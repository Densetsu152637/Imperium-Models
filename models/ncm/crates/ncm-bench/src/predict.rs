//! `ncm-bench predict`: the decode-speed model of the research plan (architecture.md, "Expected
//! decode speed"), pure arithmetic, so the speed gate's predictions sit next to its
//! measurements.
//!
//! At batch 1 a decode unit costs a fixed latency per layer pass plus the time to read the
//! weights it touches: `time = passes × c + weight bytes / bandwidth`, with the plan's fitted
//! `c` = 41 µs and 960 GB/s (RTX 5080) as defaults. Context adds a measured cost per backbone
//! step (k = 4 NCM: 0.15 ms at 16k, 0.30 ms at 128k, interpolated over log2 of the context and
//! scaled by the retrieval layers against its 8). Units:
//!
//! - token baseline: one token = its layers and the head;
//! - k = 4 NCM: one concept (k tokens) = the backbone once, then per token the encoder, the
//!   decoder and the head;
//! - hybrid: one block of L tokens = the backbone step and pooling, the memory, K generator
//!   levels (its layers plus one pass for sampling and reconstruction), the tokenizer draft, one
//!   check pass, and the encoder over the block; after a rejection one re-decode (generator,
//!   draft, check, known-prefix memory) when at least `redecode_min` positions remain, then the
//!   autoregressive fill at one check pass per token. Acceptance is geometric: each draft token
//!   agrees with probability p, and block times are expectations over that process.
//!
//! Weight bytes per parameter: bf16 2, q8 1 (one scale per output channel, negligible), q4
//! 0.5 plus one 2-byte scale per 64 inputs.

use clap::Parser;
use ncm_model::hybrid::HybridLm;
use ncm_model::{NcmConfig, NcmLm, TokConfig, TokenLm};
use serde_json::{Value, json};

#[derive(Parser, Debug, Clone)]
#[command(name = "ncm-bench predict", about = "Decode-speed predictions of the latency model.")]
pub struct PredictCli {
    /// Latency per layer pass, µs.
    #[arg(long, default_value_t = 41.0)]
    pub c_us: f64,
    /// Weight-read bandwidth, GB/s.
    #[arg(long, default_value_t = 960.0)]
    pub bandwidth_gbs: f64,
    /// Sizes: 250m, 1b or both.
    #[arg(long, default_value = "both")]
    pub size: String,
    /// Contexts in tokens.
    #[arg(long, num_args = 1.., default_values_t = [2048usize, 16384, 131072])]
    pub context: Vec<usize>,
    /// The hybrid's per-token agreement rates.
    #[arg(long, num_args = 1.., default_values_t = [1.0, 0.986, 0.967, 0.950])]
    pub p: Vec<f64>,
    /// Also write the predictions as JSON.
    #[arg(long)]
    pub json: Option<std::path::PathBuf>,
}

/// Weight formats.
const FORMATS: [(&str, f64); 3] = [("bf16", 2.0), ("q8", 1.0), ("q4", 0.5 + 2.0 / 64.0)];

/// Per-backbone-step context cost, ms, measured on the k = 4 NCM (8 retrieval layers).
fn context_ms(context: usize, retrieval_layers: usize) -> f64 {
    let x = (context.max(2048) as f64).log2();
    let ms = if x <= 14.0 { 0.15 * (x - 11.0) / 3.0 } else { 0.15 + 0.15 * (x - 14.0) / 3.0 };
    ms.max(0.0) * retrieval_layers as f64 / 8.0
}

/// One unit of work: passes and parameters read.
#[derive(Clone, Copy, Default)]
struct Cost {
    passes: f64,
    params: f64,
}

impl Cost {
    fn ms(self, cli: &PredictCli, bytes_per: f64) -> f64 {
        self.passes * cli.c_us / 1000.0 + self.params * bytes_per / (cli.bandwidth_gbs * 1e6)
    }
}

fn retrieval_layers(n: usize) -> usize {
    n / 2
}

fn tok_rate(c: &TokConfig, cli: &PredictCli, bytes: f64, context: usize) -> f64 {
    let device = burn::tensor::Device::flex();
    let params = TokenLm::new(c, &device).parts().total() as f64;
    let cost = Cost { passes: c.n_layers as f64 + 1.0, params };
    1000.0 / (cost.ms(cli, bytes) + context_ms(context, retrieval_layers(c.n_layers)))
}

fn ncm_rate(c: &NcmConfig, cli: &PredictCli, bytes: f64, context: usize) -> f64 {
    let device = burn::tensor::Device::flex();
    let p = NcmLm::new(c, &device).parts();
    let k = c.k as f64;
    let local_layers = (c.enc_layers + 1 + c.dec_layers + 1) as f64;
    let backbone = Cost { passes: c.bb_layers as f64, params: (p.backbone + p.projections) as f64 };
    let per_token =
        Cost { passes: local_layers, params: (p.encoder + p.decoder + p.embeddings) as f64 };
    let ms = backbone.ms(cli, bytes)
        + k * per_token.ms(cli, bytes)
        + context_ms(context, retrieval_layers(c.bb_layers));
    1000.0 * k / ms
}

/// The hybrid's costs per block part.
struct HybridCosts {
    backbone: Cost,
    memory: Cost,
    generator: Cost,
    draft: Cost,
    check: Cost,
    encoder: Cost,
    l: usize,
    redecode_min: usize,
    retrieval: usize,
}

fn hybrid_costs(c: &NcmConfig) -> HybridCosts {
    let device = burn::tensor::Device::flex();
    let h = c.hybrid.as_ref().expect("a hybrid config");
    let p = HybridLm::new(c, &device).parts();
    let k = h.depth() as f64;
    HybridCosts {
        backbone: Cost {
            passes: c.bb_layers as f64 + 1.0,
            params: (p.backbone + p.pooling) as f64,
        },
        memory: Cost { passes: 2.0, params: p.memory as f64 },
        generator: Cost {
            passes: k * (h.gen_layers as f64 + 1.0),
            params: k * (p.generator + p.generator_io) as f64 + (p.codebooks + p.phi) as f64,
        },
        draft: Cost { passes: 2.0, params: (p.tokenizer_decoder + p.embeddings) as f64 },
        check: Cost {
            passes: c.dec_layers as f64 + 2.0,
            params: (p.token_decoder + p.token_decoder_io + p.embeddings) as f64,
        },
        encoder: Cost { passes: (c.enc_layers + 1) as f64, params: p.encoder as f64 },
        l: h.block,
        redecode_min: h.redecode_min,
        retrieval: retrieval_layers(c.bb_layers),
    }
}

/// Expected ms to finish `r` remaining positions after a check pass's outcome is due, with
/// per-token agreement `p` (None: no generator, every position on the autoregressive path).
fn remaining_ms(
    r: usize,
    p: Option<f64>,
    redecoded: bool,
    redecode_ms: f64,
    ar_ms: f64,
    min: usize,
) -> f64 {
    let Some(p) = p else { return r as f64 * ar_ms };
    if r == 0 {
        return 0.0;
    }
    // a accepted then a rejection (a < r), emitting a + 1; or all r accepted.
    let mut e = 0.0;
    for a in 0..r {
        let prob = p.powi(a as i32) * (1.0 - p);
        if prob == 0.0 {
            continue;
        }
        let left = r - (a + 1);
        let next = if !redecoded && left >= min && left > 0 {
            redecode_ms + remaining_ms(left, Some(p), true, redecode_ms, ar_ms, min)
        } else {
            left as f64 * ar_ms
        };
        e += prob * next;
    }
    e
}

fn hybrid_rate(
    h: &HybridCosts,
    cli: &PredictCli,
    bytes: f64,
    context: usize,
    p: Option<f64>,
) -> f64 {
    let ms = |c: Cost| c.ms(cli, bytes);
    let fixed = ms(h.backbone) + ms(h.memory) + ms(h.encoder) + context_ms(context, h.retrieval);
    let pass = ms(h.generator) + ms(h.draft) + ms(h.check);
    let redecode = pass + ms(h.memory);
    let ar = ms(h.check);
    let block = match p {
        None => fixed + h.l as f64 * ar,
        Some(p) => fixed + pass + remaining_ms(h.l, Some(p), false, redecode, ar, h.redecode_min),
    };
    1000.0 * h.l as f64 / block
}

/// Prints and returns the predictions.
pub fn run(cli: &PredictCli) -> Value {
    println!(
        "decode-speed model: {} µs per pass, {} GB/s (estimates, tok/s at batch 1)",
        cli.c_us, cli.bandwidth_gbs
    );
    let sizes: Vec<(&str, TokConfig, NcmConfig, NcmConfig)> = [
        ("250m", ncm_model::tok_250m_v2(), ncm_model::ncm_250m_v2(), ncm_model::ncm_250m_hybrid()),
        ("1b", ncm_model::tok_1b(), ncm_model::ncm_1b(), ncm_model::ncm_hybrid_1b()),
    ]
    .into_iter()
    .filter(|(s, ..)| cli.size == "both" || cli.size == *s)
    .collect();
    let mut rows = Vec::new();
    for (size, tok, ncm, hybrid) in &sizes {
        let costs = hybrid_costs(hybrid);
        for (format, bytes) in FORMATS {
            for &context in &cli.context {
                let mut row = |model: String, rate: f64| {
                    println!(
                        "  {size:<5} {format:<5} ctx {context:>7}  {model:<28} {rate:>8.0} tok/s"
                    );
                    rows.push(json!({
                        "size": size, "format": format, "context": context,
                        "model": model, "tok_s": rate,
                    }));
                };
                row("token baseline".into(), tok_rate(tok, cli, bytes, context));
                row("ncm k=4".into(), ncm_rate(ncm, cli, bytes, context));
                row("hybrid, AR path only".into(), hybrid_rate(&costs, cli, bytes, context, None));
                for &p in cli.p.iter().chain(&[0.0]) {
                    row(
                        format!("hybrid, p = {p}"),
                        hybrid_rate(&costs, cli, bytes, context, Some(p)),
                    );
                }
            }
        }
    }
    json!({ "c_us": cli.c_us, "bandwidth_gbs": cli.bandwidth_gbs, "rows": rows })
}
