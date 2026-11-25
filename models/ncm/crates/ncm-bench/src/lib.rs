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
use ncm_model::fused::{argmax_last, kernels_built, set_cuda_graphs, set_fused_kernels};
use ncm_model::layers::set_fused_attention;
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
    /// Run the v2 layers' reference tensor code on CUDA instead of their fused kernels (which a
    /// `--features cuda` build uses by default).
    #[arg(long)]
    pub reference_kernels: bool,
    /// Run the fused kernels eagerly instead of replaying the NCM decoder and v2 backbone steps
    /// as captured CUDA graphs.
    #[arg(long)]
    pub no_cuda_graphs: bool,
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

    fn parts(&self) -> Vec<(&'static str, usize)> {
        match self {
            Model::Tok(m) => m.parts().rows(),
            Model::Ncm(m) => m.parts().rows(),
        }
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
            // On CUDA with the fused kernels on, the decoder replays one captured graph per
            // concept (captured on the first concept once the cross-attention ring is full).
            #[cfg(feature = "cuda")]
            let mut graphed = ncm_model::ncm::GraphedDecoder::default();
            for c in 0..concepts {
                #[cfg(feature = "cuda")]
                let tokens = graphed.decode(m, cond.clone(), &mut cache);
                #[cfg(not(feature = "cuda"))]
                let (tokens, _) = m.decode_concept(cond.clone(), &mut cache, None, false);
                if c + 1 < concepts {
                    cond = m.append_tokens(tokens, &mut cache);
                }
            }
            #[cfg(feature = "cuda")]
            if let Some(why) = graphed.failure() {
                eprintln!("ncm-bench: decoder graph not captured, decoding eagerly: {why}");
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
                token = argmax_last(m.step(token, &mut caches)).reshape([1, 1]);
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
) -> (Value, Vec<(usize, f64)>) {
    device.seed(0);
    let model = match cfg {
        Config::Tok(c) => Model::Tok(Box::new(TokenLm::new(c, device))),
        Config::Ncm(c) => Model::Ncm(Box::new(NcmLm::new(c, device))),
    };
    let design = cfg.design().name();
    let parts = model.parts();
    let params: usize = parts.iter().map(|p| p.1).sum();
    let unit = cfg.unit();
    let new_tokens = cli.new_tokens - cli.new_tokens % unit;
    println!("{} ({design}): {} params, {new_tokens} new tokens", model.name(), fmt_params(params));
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
    let result = json!({
        "name": model.name(),
        "design": design,
        "params": params,
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
    set_fused_kernels(!cli.reference_kernels);
    set_cuda_graphs(!cli.no_cuda_graphs);
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
            let (result, decode) = bench_model(&cfg, cli, &device, itemsize);
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
        "fused_kernels": cli.device == DeviceChoice::Cuda && kernels_built() && !cli.reference_kernels,
        "cuda_graphs": cli.device == DeviceChoice::Cuda
            && kernels_built()
            && !cli.reference_kernels
            && !cli.no_cuda_graphs,
        "results": results,
        "ratios": ratios,
        "ncm_v2_over_fallback": design_ratios,
    }))
}
