//! Batch-1 decode-speed benchmark: NCM against the token baseline, random weights, in either or
//! both attention designs.
//!
//! ```text
//! ncm-bench --model both --design both --context 2048 16384 --new-tokens 256 --device cuda
//! ```
//!
//! Both models share the token vocabulary, so decode tokens per second are directly comparable.
//! Decode tok/s counts output tokens (the NCM emits `k` tokens per decoder pass).

use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::{Device, FloatDType, Int, Tensor, TensorData};
use clap::{Parser, ValueEnum};
use ncm_model::layers::set_fused_attention;
use ncm_model::quant::{self, QuantSpec};
use ncm_model::{
    Config, Design, NcmLm, TokenLm, cache_bytes, cache_bytes_with, ncm_250m_of, tiny_ncm,
    tiny_ncm_v2, tiny_tok, tiny_tok_v2, tiny_v2, tok_250m_of,
};
use serde_json::{Value, json};

pub const PASS_RATIO: f64 = 1.5;
pub const WEAK_RATIO: f64 = 1.2;
/// The v2 NCM may be at most 25% slower than the fallback NCM (plan, local test plan step 2).
pub const FALLBACK_FLOOR: f64 = 0.75;
const PREFILL_CHUNK: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ModelChoice {
    Tok,
    Ncm,
    Both,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum DesignChoice {
    /// Gated DeltaNet and sparse retrieval (the adopted design).
    V2,
    /// Sliding-window attention with a full layer every fourth (the stage 1 design).
    Fallback,
    Both,
}

impl DesignChoice {
    fn designs(self) -> Vec<Design> {
        match self {
            Self::V2 => vec![Design::V2],
            Self::Fallback => vec![Design::Fallback],
            Self::Both => vec![Design::Fallback, Design::V2],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum DeviceChoice {
    /// CUDA (build with `--features cuda`).
    Cuda,
    /// Portable CPU backend.
    Cpu,
    /// Any wgpu adapter (build with `--features wgpu`).
    Wgpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum DtypeChoice {
    Bf16,
    #[value(alias = "fp32")]
    F32,
}

/// Weight-only quantisation of the linear layers (see `ncm_model::quant`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum QuantChoice {
    /// Dense weights at the run dtype.
    None,
    /// 8-bit codes, one scale per output channel (W8A16).
    Q8,
    /// 4-bit codes, one scale per `--quant-group` inputs (W4A16).
    Q4,
}

#[derive(Parser, Debug, Clone)]
#[command(
    name = "ncm-bench",
    version,
    about = "Batch-1 decode speed of the NCM against the token baseline (random weights)."
)]
pub struct Cli {
    #[arg(long, value_enum, default_value = "both")]
    pub model: ModelChoice,
    /// Attention design(s) to run.
    #[arg(long, value_enum, default_value = "both")]
    pub design: DesignChoice,
    /// Prompt lengths in tokens.
    #[arg(long, num_args = 1.., default_values_t = [2048usize, 16384], value_name = "N")]
    pub context: Vec<usize>,
    #[arg(long, default_value_t = 256)]
    pub new_tokens: usize,
    #[arg(long, value_enum, default_value = "cuda")]
    pub device: DeviceChoice,
    /// Defaults to bf16 on cuda and f32 elsewhere.
    #[arg(long, value_enum)]
    pub dtype: Option<DtypeChoice>,
    #[arg(long, default_value_t = 3)]
    pub repeats: usize,
    /// Also write the results as JSON.
    #[arg(long, value_name = "PATH")]
    pub json: Option<PathBuf>,
    /// Tiny configs for a CPU smoke run.
    #[arg(long)]
    pub tiny: bool,
    /// Use Burn's fused attention op instead of matmul-softmax-matmul (experimental).
    #[arg(long)]
    pub fused_attention: bool,
    /// Quantise the linear layers' weights (the random weights are quantised as they are built).
    #[arg(long, value_enum, default_value = "none")]
    pub quant: QuantChoice,
    /// Inputs per scale for `--quant q4` (a multiple of 32).
    #[arg(long, default_value_t = 64)]
    pub quant_group: usize,
    /// Keep the tied output head dense when quantising.
    #[arg(long)]
    pub dense_head: bool,
}

impl Cli {
    /// The quantisation `--quant`, `--quant-group` and `--dense-head` ask for.
    pub fn quant_spec(&self) -> Result<Option<QuantSpec>, String> {
        let mode = match self.quant {
            QuantChoice::None => "none",
            QuantChoice::Q8 => "q8",
            QuantChoice::Q4 => "q4",
        };
        QuantSpec::parse(mode, self.quant_group, !self.dense_head)
    }
}

pub fn verdict(ratio: f64) -> &'static str {
    if ratio >= PASS_RATIO {
        "PASS"
    } else if ratio >= WEAK_RATIO {
        "WEAK"
    } else {
        "FAIL"
    }
}

/// Verdict on v2 NCM decode speed relative to the fallback NCM.
pub fn fallback_verdict(ratio: f64) -> &'static str {
    if ratio >= FALLBACK_FLOOR { "PASS" } else { "FAIL" }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 { values[mid] } else { (values[mid - 1] + values[mid]) / 2.0 }
}

fn make_device(choice: DeviceChoice) -> Result<Device, String> {
    match choice {
        DeviceChoice::Cpu => Ok(Device::flex()),
        #[cfg(feature = "cuda")]
        DeviceChoice::Cuda => Ok(Device::cuda(0)),
        #[cfg(not(feature = "cuda"))]
        DeviceChoice::Cuda => Err("--device cuda needs a build with `--features cuda`".into()),
        #[cfg(feature = "wgpu")]
        DeviceChoice::Wgpu => Ok(Device::wgpu(burn::tensor::DeviceKind::DefaultDevice)),
        #[cfg(not(feature = "wgpu"))]
        DeviceChoice::Wgpu => Err("--device wgpu needs a build with `--features wgpu`".into()),
    }
}

enum Model {
    Tok(Box<TokenLm>),
    Ncm(Box<NcmLm>),
}

impl Model {
    fn name(&self) -> &'static str {
        match self {
            Model::Tok(_) => "tok",
            Model::Ncm(_) => "ncm",
        }
    }

    /// Parameter bytes on the device (codes and scales when quantised).
    fn weight_bytes(&self) -> usize {
        match self {
            Model::Tok(m) => quant::weight_bytes(m.as_ref()),
            Model::Ncm(m) => quant::weight_bytes(m.as_ref()),
        }
    }
}

/// Parameter counts of `cfg` by part, from a dense model that is never initialised (a quantised
/// model's own count would see codes and scales).
fn dense_parts(cfg: &Config, device: &Device) -> Vec<(&'static str, usize)> {
    match cfg {
        Config::Tok(c) => TokenLm::new(c, device).parts().rows(),
        Config::Ncm(c) => NcmLm::new(c, device).parts().rows(),
    }
}

fn sync(device: &Device) {
    device.sync().expect("device sync");
}

/// Prefill the prompt, then greedily decode `new_tokens` tokens; returns (prefill s, decode s).
///
/// The token baseline prefills all but the last prompt token and decodes from that one, so its
/// decode loop produces exactly `new_tokens` tokens like the NCM loop does.
fn run_once(model: &Model, prompt: &Tensor<2, Int>, new_tokens: usize, device: &Device) -> Run {
    let context = prompt.dims()[1];
    let (started, prefilled) = match model {
        Model::Ncm(m) => {
            let mut cache = m.make_cache(context + new_tokens, 1, device);
            sync(device);
            let started = Instant::now();
            let mut cond = m.prefill(prompt.clone(), &mut cache, PREFILL_CHUNK);
            sync(device);
            let prefilled = Instant::now();
            let concepts = new_tokens / m.concept_size();
            for c in 0..concepts {
                let (tokens, _) = m.decode_concept(cond.clone(), &mut cache, None, false);
                if c + 1 < concepts {
                    cond = m.append_tokens(tokens, &mut cache);
                }
            }
            (started, prefilled)
        }
        Model::Tok(m) => {
            let mut caches = m.make_cache(context + new_tokens, 1, device);
            sync(device);
            let started = Instant::now();
            let _ = m.prefill(prompt.clone().narrow(1, 0, context - 1), &mut caches, PREFILL_CHUNK);
            sync(device);
            let prefilled = Instant::now();
            let mut token = prompt.clone().narrow(1, context - 1, 1);
            for _ in 0..new_tokens {
                token = m.step(token, &mut caches).argmax(2).reshape([1, 1]);
            }
            (started, prefilled)
        }
    };
    let enqueued = Instant::now();
    sync(device);
    let done = Instant::now();
    Run {
        prefill_s: prefilled.duration_since(started).as_secs_f64(),
        decode_s: done.duration_since(prefilled).as_secs_f64(),
        enqueue_s: enqueued.duration_since(prefilled).as_secs_f64(),
    }
}

/// Timings of one run. `enqueue_s` is the decode time before the final device sync: close to
/// `decode_s` means the host, not the GPU, is the bottleneck.
struct Run {
    prefill_s: f64,
    decode_s: f64,
    enqueue_s: f64,
}

fn random_prompt(device: &Device, vocab: usize, n: usize) -> Tensor<2, Int> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let data: Vec<i32> = (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % vocab as u64) as i32
        })
        .collect();
    Tensor::from_data(TensorData::new(data, [1, n]), device)
}

fn fmt_params(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else {
        format!("{:.1}K", n as f64 / 1e3)
    }
}

fn fmt_bytes(n: u64) -> String {
    format!("{:.2} GiB", n as f64 / (1u64 << 30) as f64)
}

/// One model's benchmark: params, then per-context prefill time, decode tok/s and cache bytes.
fn bench_model(
    cfg: &Config,
    cli: &Cli,
    device: &Device,
    itemsize: usize,
    spec: Option<QuantSpec>,
) -> (Value, Vec<(usize, f64)>) {
    device.seed(0);
    let model = quant::build(spec, || match cfg {
        Config::Tok(c) => Model::Tok(Box::new(TokenLm::new(c, device))),
        Config::Ncm(c) => Model::Ncm(Box::new(NcmLm::new(c, device))),
    });
    let design = cfg.design().name();
    let parts = dense_parts(cfg, device);
    let params: usize = parts.iter().map(|p| p.1).sum();
    let unit = cfg.unit();
    let new_tokens = cli.new_tokens - cli.new_tokens % unit;
    let quant = spec.map_or_else(|| "none".to_owned(), |s| s.name());
    println!(
        "{} ({design}, quant {quant}): {} params, {new_tokens} new tokens",
        model.name(),
        fmt_params(params)
    );
    for (name, n) in &parts {
        println!("  {name:<18} {:>10}", fmt_params(*n));
    }
    let mut contexts = serde_json::Map::new();
    let mut decode_rates = Vec::new();
    for &context in &cli.context {
        let context = context - context % unit;
        let prompt = random_prompt(device, cfg.vocab_size(), context);
        let warm = random_prompt(device, cfg.vocab_size(), context.min(64 * unit));
        run_once(&model, &warm, 2 * unit, device); // warm-up: compile kernels, fill pools
        let (mut prefills, mut decodes, mut enqueues) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..cli.repeats {
            let run = run_once(&model, &prompt, new_tokens, device);
            prefills.push(run.prefill_s);
            decodes.push(run.decode_s);
            enqueues.push(run.enqueue_s);
        }
        let reserved = device.memory_pool_usage().map(|u| u.bytes_reserved);
        // The design's cache (8-bit indexer keys in v2) and what the reference allocates.
        let cache = cache_bytes(cfg, context + new_tokens, itemsize, 1);
        let allocated = cache_bytes_with(cfg, context + new_tokens, itemsize, itemsize, 1);
        let decode_tok_s = new_tokens as f64 / median(&mut decodes.clone());
        let prefill_s = median(&mut prefills.clone());
        let host_share = median(&mut enqueues.clone()) / median(&mut decodes.clone());
        println!(
            "  ctx {context:>7}: prefill {prefill_s:.3}s  decode {decode_tok_s:.1} tok/s  \
             host {:.0}%  reserved {}  cache {}",
            host_share * 100.0,
            reserved.map_or("n/a".into(), fmt_bytes),
            fmt_bytes(cache as u64),
        );
        contexts.insert(
            context.to_string(),
            json!({
                "prefill_s": prefill_s,
                "decode_tok_s": decode_tok_s,
                "decode_host_share": host_share,
                "decode_s_runs": decodes,
                "prefill_s_runs": prefills,
                "reserved_memory_bytes": reserved,
                "cache_bytes": cache,
                "cache_bytes_allocated": allocated,
            }),
        );
        decode_rates.push((context, decode_tok_s));
    }
    // After the runs, so every parameter is initialised in the order the runs used them.
    let weight_bytes = model.weight_bytes();
    println!("  weights {:.1} MiB on the device", weight_bytes as f64 / (1u64 << 20) as f64);
    let result = json!({
        "name": model.name(),
        "design": design,
        "quant": quant,
        "params": params,
        "weight_bytes": weight_bytes,
        "parts": parts.iter().map(|(n, c)| json!({"name": n, "params": c})).collect::<Vec<_>>(),
        "new_tokens": new_tokens,
        "contexts": contexts,
    });
    (result, decode_rates)
}

/// The config of model `kind` ("tok" or "ncm") in `design`.
fn config_for(kind: &str, design: Design, tiny: bool) -> Config {
    match (kind, design, tiny) {
        ("tok", Design::Fallback, true) => Config::Tok(tiny_tok(8)),
        ("tok", Design::V2, true) => Config::Tok(tiny_tok_v2(tiny_v2(4, 2, 8))),
        ("tok", design, false) => Config::Tok(tok_250m_of(design)),
        (_, Design::Fallback, true) => Config::Ncm(tiny_ncm(8, 4)),
        (_, Design::V2, true) => Config::Ncm(tiny_ncm_v2(8, tiny_v2(2, 2, 4))),
        (_, design, false) => Config::Ncm(ncm_250m_of(design)),
    }
}

/// Decode tok/s by context of one model.
type Rates = Vec<(usize, f64)>;

/// `a / b` at every context both measured.
fn rate_ratios(a: &Rates, b: &Rates) -> Vec<(usize, f64)> {
    a.iter()
        .filter_map(|(context, ra)| {
            b.iter().find(|(c, _)| c == context).map(|(_, rb)| (*context, ra / rb))
        })
        .collect()
}

/// Run the benchmark described by `cli`; prints the report and returns the JSON payload.
pub fn run(cli: &Cli) -> Result<Value, String> {
    let mut device = make_device(cli.device)?;
    let dtype = cli.dtype.unwrap_or(match cli.device {
        DeviceChoice::Cuda => DtypeChoice::Bf16,
        _ => DtypeChoice::F32,
    });
    let float = match dtype {
        DtypeChoice::Bf16 => FloatDType::BF16,
        DtypeChoice::F32 => FloatDType::F32,
    };
    // A device's dtype locks on first use; a second run in the same process may reuse it.
    if let Err(e) = device.configure(float)
        && device.settings().float_dtype != float
    {
        return Err(format!("cannot use {dtype:?} on {:?}: {e}", cli.device));
    }
    if cli.repeats == 0 || cli.context.is_empty() {
        return Err("--repeats and --context need at least one value".into());
    }
    set_fused_attention(cli.fused_attention);
    let spec = cli.quant_spec()?;
    let itemsize = match dtype {
        DtypeChoice::Bf16 => 2,
        DtypeChoice::F32 => 4,
    };
    let kinds: Vec<&str> = match cli.model {
        ModelChoice::Tok => vec!["tok"],
        ModelChoice::Ncm => vec!["ncm"],
        ModelChoice::Both => vec!["tok", "ncm"],
    };
    let mut results = Vec::new();
    let mut rates: Vec<(&str, Design, Rates)> = Vec::new();
    for design in cli.design.designs() {
        for kind in &kinds {
            let cfg = config_for(kind, design, cli.tiny);
            let (result, decode) = bench_model(&cfg, cli, &device, itemsize, spec);
            results.push(result);
            rates.push((kind, design, decode));
        }
    }
    let find = |kind: &str, design: Design| {
        rates.iter().find(|r| r.0 == kind && r.1 == design).map(|r| &r.2)
    };
    let note = if cli.tiny { "  (tiny config: smoke run, not a gate result)" } else { "" };
    let mut ratios = Vec::new();
    for design in cli.design.designs() {
        let (Some(tok), Some(ncm)) = (find("tok", design), find("ncm", design)) else { continue };
        for (context, ratio) in rate_ratios(ncm, tok) {
            let v = verdict(ratio);
            println!("ratio ncm/tok ({}) @ ctx {context}: {ratio:.2}x {v}{note}", design.name());
            ratios.push(json!({
                "design": design.name(),
                "context": context,
                "ratio": ratio,
                "verdict": v,
            }));
        }
    }
    // The second gate: the v2 NCM against the fallback NCM.
    let mut design_ratios = Vec::new();
    if let (Some(v2), Some(fallback)) = (find("ncm", Design::V2), find("ncm", Design::Fallback)) {
        for (context, ratio) in rate_ratios(v2, fallback) {
            let v = fallback_verdict(ratio);
            println!(
                "ratio ncm v2/fallback @ ctx {context}: {ratio:.2}x {v} \
                 (floor {FALLBACK_FLOOR}x){note}"
            );
            design_ratios.push(json!({"context": context, "ratio": ratio, "verdict": v}));
        }
    }
    Ok(json!({
        "device": format!("{:?}", cli.device).to_lowercase(),
        "dtype": format!("{dtype:?}").to_lowercase(),
        "tiny": cli.tiny,
        "fused_attention": cli.fused_attention,
        "quant": spec.map_or_else(|| "none".to_owned(), |s| s.name()),
        "results": results,
        "ratios": ratios,
        "ncm_v2_over_fallback": design_ratios,
    }))
}
