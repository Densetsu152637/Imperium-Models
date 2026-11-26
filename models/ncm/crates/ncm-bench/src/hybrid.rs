//! The hybrid NCM's decode benchmark (local test plan step 6), on the eager reference decode
//! loop ([`ncm_model::hybrid::HybridSession`]): no fused kernels or CUDA graphs for the
//! generator yet (TODO), so these rates measure the reference, not the design.
//!
//! Random weights accept almost no drafts, so every run but the autoregressive-path one decides
//! acceptance synthetically: each draft token agrees with the check with probability p,
//! independently (geometric first rejections, the design's speed model), on the first pass and
//! after the re-decode alike. The passes run are the real ones (generator, draft, check,
//! re-decode, autoregressive fill); the tokens are not the model's output.

use std::time::Instant;

use burn::tensor::Device;
use ncm_model::hybrid::{BlockStats, DecodeOptions, Hooks, HybridLm, HybridSession};
use ncm_model::quant::{self, QuantSpec};
use ncm_model::{Config, ncm_250m_hybrid, tiny_hybrid, tiny_v2};
use serde_json::{Value, json};

use crate::Cli;

/// The step 6 gate: at a mean accepted prefix of 25.6 tokens (p = 0.986), at least this many
/// times the k = 4 NCM's decode rate.
pub const HYBRID_GATE: f64 = 2.5;
/// Per-token agreement of the gate's acceptance setting.
pub const GATE_P: f64 = 0.986;

/// The hybrid config the benchmark runs.
pub fn hybrid_config(tiny: bool) -> Config {
    Config::Ncm(if tiny { tiny_hybrid(tiny_v2(2, 2, 4)) } else { ncm_250m_hybrid() })
}

fn random_tokens(vocab: usize, n: usize, seed: u64) -> Vec<i32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % vocab as u64) as i32
        })
        .collect()
}

/// How a run decides acceptance.
#[derive(Clone, Copy, Debug)]
enum Mode {
    /// The autoregressive path only.
    ArOnly,
    /// Verified mode with each draft token agreeing with probability p.
    Synthetic(f64),
}

impl Mode {
    fn label(self) -> String {
        match self {
            Mode::ArOnly => "autoregressive path only".into(),
            Mode::Synthetic(p) => format!("SYNTHETIC per-token agreement p = {p}"),
        }
    }
}

/// One timed reply: prefill seconds, decode seconds, tokens and per-block stats.
struct Reply {
    prefill_s: f64,
    decode_s: f64,
    tokens: usize,
    blocks: Vec<BlockStats>,
}

fn reply(
    model: &HybridLm,
    prompt: &[i32],
    new_tokens: usize,
    mode: Mode,
    seed: u64,
    device: &Device,
) -> Reply {
    let l = model.config().block;
    let mut session = HybridSession::new(model, prompt.len() + new_tokens + l, device);
    let t0 = Instant::now();
    session.prefill(model, prompt);
    device.sync().expect("device sync");
    let t1 = Instant::now();
    let opts = DecodeOptions {
        verified: true,
        ar_only: matches!(mode, Mode::ArOnly),
        seed,
        ..DecodeOptions::default()
    };
    let mut state = seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1;
    let mut agree = |_pass: usize, _pos: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u = (state >> 11) as f64 / (1u64 << 53) as f64;
        match mode {
            Mode::Synthetic(p) => u < p,
            Mode::ArOnly => false,
        }
    };
    let mut hooks = Hooks { accept: Some(&mut agree), ..Hooks::default() };
    let (tokens, blocks) = session.generate_reply(model, new_tokens, &opts, &mut hooks);
    device.sync().expect("device sync");
    let t2 = Instant::now();
    Reply {
        prefill_s: t1.duration_since(t0).as_secs_f64(),
        decode_s: t2.duration_since(t1).as_secs_f64(),
        tokens: tokens.len(),
        blocks,
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 { values[mid] } else { (values[mid - 1] + values[mid]) / 2.0 }
}

/// The hybrid's benchmark: parameters, then per context the prefill time and, per acceptance
/// setting, the decode rate and what the loop did. Returns the JSON result and the decode rates
/// `(context, label, tok/s)`.
pub fn bench_hybrid(
    cli: &Cli,
    device: &Device,
    spec: Option<QuantSpec>,
) -> (Value, Vec<(usize, String, f64)>) {
    let cfg = hybrid_config(cli.tiny);
    let Config::Ncm(ncm) = &cfg else { unreachable!("the hybrid is an NCM config") };
    device.seed(0);
    let model = quant::build(spec, || HybridLm::new(ncm, device));
    let parts = HybridLm::new(ncm, device).parts();
    let l = model.config().block;
    let new_tokens = cli.new_tokens.max(l);
    println!(
        "hybrid (L = {l}, K = {}): {:.1}M params at inference ({:.1}M trained), {new_tokens} new tokens \
         (eager reference decode loop)",
        model.config().depth(),
        parts.inference() as f64 / 1e6,
        parts.trained() as f64 / 1e6,
    );
    for (name, n) in parts.rows() {
        println!("  {name:<34} {:>8.2}M", n as f64 / 1e6);
    }
    let modes: Vec<Mode> = std::iter::once(Mode::ArOnly)
        .chain(cli.hybrid_p.iter().map(|&p| Mode::Synthetic(p)))
        .chain(std::iter::once(Mode::Synthetic(0.0)))
        .collect();
    let mut contexts = serde_json::Map::new();
    let mut rates = Vec::new();
    for &context in &cli.context {
        let prompt = random_tokens(ncm.vocab_size, context, 1);
        let warm = random_tokens(ncm.vocab_size, context.min(4 * l), 2);
        let mut runs = Vec::new();
        let mut prefills = Vec::new();
        for mode in &modes {
            let _ = reply(&model, &warm, 2 * l, *mode, 0, device);
            let (mut decode_rates, mut first, mut redecodes, mut ar, mut blocks) =
                (Vec::new(), Vec::new(), 0, 0, 0);
            for r in 0..cli.repeats {
                let run = reply(&model, &prompt, new_tokens, *mode, r as u64 + 1, device);
                prefills.push(run.prefill_s);
                decode_rates.push(run.tokens as f64 / run.decode_s);
                for b in &run.blocks {
                    blocks += 1;
                    redecodes += usize::from(b.redecoded);
                    ar += b.ar_tokens;
                    if let Some(a) = b.first_accepted {
                        first.push(a as f64);
                    }
                }
            }
            let tok_s = median(&mut decode_rates);
            let mean_first =
                if first.is_empty() { 0.0 } else { first.iter().sum::<f64>() / first.len() as f64 };
            let label = mode.label();
            println!(
                "  ctx {context:>7}: {label}: decode {tok_s:.1} tok/s, mean first accepted prefix \
                 {mean_first:.1}, re-decodes {:.2} and autoregressive tokens {:.1} per block",
                redecodes as f64 / blocks.max(1) as f64,
                ar as f64 / blocks.max(1) as f64,
            );
            runs.push(json!({
                "mode": label,
                "synthetic": matches!(mode, Mode::Synthetic(_)),
                "agreement_p": match mode { Mode::Synthetic(p) => Some(*p), Mode::ArOnly => None },
                "decode_tok_s": tok_s,
                "decode_tok_s_runs": decode_rates,
                "mean_first_accepted": mean_first,
                "redecodes_per_block": redecodes as f64 / blocks.max(1) as f64,
                "ar_tokens_per_block": ar as f64 / blocks.max(1) as f64,
            }));
            rates.push((context, label, tok_s));
        }
        let prefill_s = median(&mut prefills);
        println!("  ctx {context:>7}: prefill {prefill_s:.3}s");
        contexts.insert(context.to_string(), json!({ "prefill_s": prefill_s, "runs": runs }));
    }
    let result = json!({
        "name": "hybrid",
        "design": "v2",
        "params": parts.inference(),
        "params_trained": parts.trained(),
        "parts": parts.rows().iter().map(|(n, c)| json!({"name": n, "params": c})).collect::<Vec<_>>(),
        "new_tokens": new_tokens,
        "reference_decode_loop": true,
        "contexts": contexts,
    });
    (result, rates)
}

/// The label of the gate's acceptance setting in [`bench_hybrid`]'s rates.
pub fn gate_label() -> String {
    Mode::Synthetic(GATE_P).label()
}

/// The label of the autoregressive-path run.
pub fn ar_label() -> String {
    Mode::ArOnly.label()
}
