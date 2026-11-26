//! `ncm-bench train-memory`: a training-memory estimate (pure arithmetic, no GPU) for the NCM
//! hybrid and the token baseline at 250M and about 1B parameters, against a GPU's memory
//! (default 16 GiB, the RTX 5080).
//!
//! Per model: parameters (counted from the models' shapes on the CPU, never initialised),
//! weights, an optional fp32 master copy, gradients, AdamW states (fp32 moments, 8 bytes per
//! parameter, or 8-bit moments with one fp32 scale per 2,048 values, about 2 bytes), and the
//! activations one micro-batch keeps for the backward pass, without and with per-layer
//! recomputation.
//!
//! The activation model is an estimate, counted per layer in elements of the activation dtype
//! (bf16 by default), written out in [`activations`]: every pre-norm layer keeps its input, its
//! normed input, its mixer's projections (`proj` wide), its output, the MLP's gate, up and
//! activation (`3 ff`) and its output; attention keeps its probabilities (queries × visible keys
//! × heads); the vocabulary head keeps the logits (bf16) and their log-softmax (fp32). The
//! hybrid's memory attention keeps, per generator or token-decoder layer, the probabilities over
//! each block's window and entries (chunking bounds the live scores, not what autodiff keeps).
//! With recomputation only each layer's input is kept, plus the largest single layer's working
//! set (Burn's Balanced checkpointing recomputes the memory-bound part only, so it lands between
//! the two columns).

use clap::Parser;
use ncm_model::hybrid::HybridLm;
use ncm_model::{Config, NcmLm, TokenLm, V2Config};
use serde_json::{Value, json};

const GIB: f64 = (1u64 << 30) as f64;

#[derive(Parser, Debug, Clone)]
#[command(name = "ncm-bench train-memory", about = "Training-memory estimate (no GPU).")]
pub struct TrainMemoryCli {
    /// Tokens per training sequence.
    #[arg(long, default_value_t = 4096)]
    pub seq: usize,
    /// Sequences per micro-batch.
    #[arg(long, default_value_t = 1)]
    pub micro_batch: usize,
    /// Bytes per weight, gradient and activation value: 2 (bf16) or 4 (f32).
    #[arg(long, default_value_t = 2)]
    pub value_bytes: usize,
    /// Keep an fp32 master copy of the weights.
    #[arg(long)]
    pub master: bool,
    /// GPU memory to compare against, GiB.
    #[arg(long, default_value_t = 16.0)]
    pub gpu_gib: f64,
    /// Sizes to estimate: 250m, 1b or both.
    #[arg(long, default_value = "both")]
    pub size: String,
    /// Also write the estimate as JSON.
    #[arg(long)]
    pub json: Option<std::path::PathBuf>,
}

/// Activation elements of one layer kept for the backward pass.
struct Layer {
    /// Positions (rows) and width.
    rows: f64,
    d: f64,
    /// The mixer's projection width and the MLP width.
    proj: f64,
    ff: f64,
    /// Attention probabilities kept (queries × keys × heads), 0 for none.
    scores: f64,
}

impl Layer {
    /// Elements kept without recomputation.
    fn kept(&self) -> f64 {
        self.rows * (6.0 * self.d + self.proj + 3.0 * self.ff) + self.scores
    }
}

/// Activation bytes of a model's training forward for one micro-batch: (without, with
/// per-layer recomputation).
fn activations(cfg: &Config, cli: &TrainMemoryCli) -> (f64, f64) {
    let (t, b, a) = (cli.seq as f64, cli.micro_batch as f64, cli.value_bytes as f64);
    let mut layers = Vec::new();
    let mut vocab_rows = t;
    let vocab;
    match cfg {
        Config::Tok(c) => {
            vocab = c.vocab_size as f64;
            layers.extend(stack(c.d_model, c.ff_dim, c.n_layers, t, &c.v2));
        }
        Config::Ncm(c) => {
            vocab = c.vocab_size as f64;
            let d = c.local_dim as f64;
            let w = c.enc_window.min(cli.seq) as f64;
            // Local encoder: sliding-window attention layers and one DeltaNet layer.
            for _ in 0..c.enc_layers {
                let scores = t * w * c.enc_heads as f64;
                let proj = ((c.enc_heads + 2 * c.enc_kv_heads) * c.head_dim) as f64 + d;
                layers.push(Layer { rows: t, d, proj, ff: c.enc_ff_dim as f64, scores });
            }
            layers.push(Layer { rows: t, d, proj: 4.0 * d, ff: c.enc_ff_dim as f64, scores: 0.0 });
            match &c.hybrid {
                Some(h) => {
                    let l = h.block as f64;
                    let nb = (t / l).max(1.0);
                    layers.extend(stack(c.bb_dim, c.bb_ff_dim, c.bb_layers, nb, &c.v2));
                    // Generator: nb blocks of Σℓ queries; memory window + entries per block.
                    let q = h.codes_per_block() as f64;
                    let rt = (h.retrieved_blocks * h.block) as f64;
                    let window = (h.recent as f64).min(t) + (h.memory_chunk.max(1) as f64) * l;
                    let entries = h.slots as f64 + rt + (l - 1.0);
                    for _ in 0..h.gen_layers {
                        let scores = nb * q * (q + window + entries) * h.gen_heads as f64;
                        let proj =
                            ((h.gen_heads + 2 * h.gen_kv_heads) * c.head_dim) as f64 + 3.0 * d;
                        layers.push(Layer {
                            rows: nb * q,
                            d,
                            proj,
                            ff: h.gen_ff_dim as f64,
                            scores,
                        });
                    }
                    // Token decoder: causal within the block, memory without the known prefix.
                    for _ in 0..c.dec_layers {
                        let scores = t * (l + window + h.slots as f64 + rt) * c.dec_heads as f64;
                        let proj =
                            ((c.dec_heads + 2 * c.dec_kv_heads) * c.head_dim) as f64 + 3.0 * d;
                        layers.push(Layer { rows: t, d, proj, ff: c.dec_ff_dim as f64, scores });
                    }
                    // Copy head over the window and the retrieved tokens.
                    layers.push(Layer {
                        rows: t,
                        d,
                        proj: 0.0,
                        ff: 0.0,
                        scores: t * (window + rt),
                    });
                    // Code logits: nb · Σℓ rows over each level's codebook.
                    layers.push(Layer {
                        rows: nb * q,
                        d: 0.0,
                        proj: 0.0,
                        ff: 0.0,
                        scores: nb * q * h.codebook as f64 * 3.0,
                    });
                }
                None => {
                    let k = c.k as f64;
                    let m = t / k;
                    layers.extend(stack(c.bb_dim, c.bb_ff_dim, c.bb_layers, m, &c.v2));
                    for _ in 0..c.dec_layers {
                        let scores = t * (k + w) * c.dec_heads as f64;
                        let proj =
                            ((c.dec_heads + 2 * c.dec_kv_heads) * c.head_dim) as f64 + 3.0 * d;
                        layers.push(Layer { rows: t, d, proj, ff: c.dec_ff_dim as f64, scores });
                    }
                    vocab_rows = t - k;
                }
            }
        }
    }
    // The vocabulary head: bf16 logits and fp32 log-softmax.
    let head = b * vocab_rows * vocab * (a + 4.0);
    let full: f64 = layers.iter().map(Layer::kept).sum::<f64>() * b * a + head;
    let inputs: f64 = layers.iter().map(|l| l.rows * l.d).sum::<f64>() * b * a;
    let largest = layers.iter().map(Layer::kept).fold(0.0, f64::max) * b * a;
    (full, inputs + largest + head)
}

/// The v2 stack's layers (DeltaNet and retrieval alternating) over `rows` positions.
fn stack(d: usize, ff: usize, n: usize, rows: f64, v2: &V2Config) -> Vec<Layer> {
    let d = d as f64;
    (0..n)
        .map(|i| {
            if i % 2 == 0 {
                // DeltaNet: q, k, v, gate and the chunked form's intra-chunk scores.
                let heads = d / v2.dn_head_dim as f64;
                let scores = rows * v2.dn_chunk as f64 * heads * 2.0;
                Layer { rows, d, proj: 4.0 * d, ff: ff as f64, scores }
            } else {
                // Retrieval: the projection and the attention over the selected positions.
                let keys = rows.min((v2.ret_topk * v2.ret_block + v2.ret_local) as f64);
                let proj = (v2.ret_heads * (v2.ret_nope_dim + v2.ret_rope_dim + v2.ret_v_dim)
                    + v2.ret_latent
                    + v2.ret_rope_dim) as f64;
                let scores = rows * keys * v2.ret_heads as f64;
                Layer { rows, d, proj, ff: ff as f64, scores }
            }
        })
        .collect()
}

/// Parameters trained (dense count from the shapes).
fn params(cfg: &Config) -> usize {
    let device = burn::tensor::Device::flex();
    match cfg {
        Config::Tok(c) => TokenLm::new(c, &device).parts().total(),
        Config::Ncm(c) if c.hybrid.is_some() => HybridLm::new(c, &device).parts().trained(),
        Config::Ncm(c) => NcmLm::new(c, &device).parts().total(),
    }
}

/// The estimate of one model.
pub fn estimate(name: &str, cfg: &Config, cli: &TrainMemoryCli) -> Value {
    let p = params(cfg) as f64;
    let v = cli.value_bytes as f64;
    let weights = p * v;
    let master = if cli.master && cli.value_bytes < 4 { p * 4.0 } else { 0.0 };
    let grads = p * v;
    let adam = p * 8.0;
    let adam8 = p * 2.0 + p / 2048.0 * 8.0;
    let (act, act_ckpt) = activations(cfg, cli);
    let state = weights + master + grads;
    let totals = [
        ("adamw", state + adam + act),
        ("adamw_ckpt", state + adam + act_ckpt),
        ("adamw8", state + adam8 + act),
        ("adamw8_ckpt", state + adam8 + act_ckpt),
    ];
    let gib = |x: f64| x / GIB;
    println!(
        "{name:<12} {:>7.1}M params | weights {:.2} master {:.2} grads {:.2} | AdamW {:.2} (8-bit {:.2}) | \
         activations {:.2} (recompute {:.2}) GiB",
        p / 1e6,
        gib(weights),
        gib(master),
        gib(grads),
        gib(adam),
        gib(adam8),
        gib(act),
        gib(act_ckpt),
    );
    let fits: Vec<String> = totals
        .iter()
        .map(|(k, t)| {
            format!("{k} {:.1}{}", gib(*t), if gib(*t) <= cli.gpu_gib { "" } else { " (over)" })
        })
        .collect();
    println!("{:<12} totals GiB: {}", "", fits.join(", "));
    json!({
        "model": name,
        "params": p,
        "weights_gib": gib(weights),
        "master_gib": gib(master),
        "grads_gib": gib(grads),
        "adamw_gib": gib(adam),
        "adamw8_gib": gib(adam8),
        "activations_gib": gib(act),
        "activations_recompute_gib": gib(act_ckpt),
        "totals_gib": totals.iter().map(|(k, t)| json!({"setup": k, "gib": gib(*t), "fits": gib(*t) <= cli.gpu_gib})).collect::<Vec<_>>(),
    })
}

/// Runs the estimate for the four models; returns the JSON payload.
pub fn run(cli: &TrainMemoryCli) -> Value {
    println!(
        "training memory, {} tokens x {} per micro-batch, {}-byte values{}, GPU {} GiB (estimate)",
        cli.seq,
        cli.micro_batch,
        cli.value_bytes,
        if cli.master { ", fp32 master" } else { "" },
        cli.gpu_gib
    );
    let models = [
        ("250m", "tok-250m", Config::Tok(ncm_model::tok_250m_v2())),
        ("250m", "ncm-250m", Config::Ncm(ncm_model::ncm_250m_v2())),
        ("250m", "hybrid-250m", Config::Ncm(ncm_model::ncm_250m_hybrid())),
        ("1b", "tok-1b", Config::Tok(ncm_model::tok_1b())),
        ("1b", "ncm-1b", Config::Ncm(ncm_model::ncm_1b())),
        ("1b", "hybrid-1b", Config::Ncm(ncm_model::ncm_hybrid_1b())),
    ];
    let results: Vec<Value> = models
        .iter()
        .filter(|(size, _, _)| cli.size == "both" || cli.size == *size)
        .map(|(_, n, c)| estimate(n, c, cli))
        .collect();
    json!({
        "seq": cli.seq,
        "micro_batch": cli.micro_batch,
        "value_bytes": cli.value_bytes,
        "master": cli.master,
        "gpu_gib": cli.gpu_gib,
        "models": results,
    })
}
