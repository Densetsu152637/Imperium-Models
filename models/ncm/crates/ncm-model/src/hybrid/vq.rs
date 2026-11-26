//! The block tokenizer: a block-local multiscale residual VQ-VAE (Next-Concept Modeling,
//! arXiv 2610.14162, `models/quant.py` and `models/vae.py` of its code), plain Burn tensor code.
//!
//! A block of L tokens is embedded (token plus learned position embedding, no transformer
//! blocks) into latents `z` `[L, C]`. Level k (ℓ_k code positions) area-pools the residual to ℓ_k
//! vectors (the finest level, ℓ = L, takes it as is), picks each one's nearest codebook entry
//! (squared L2), upsamples the entries linearly back to L and blends them through a φ
//! convolution: `(1 - r) h + r conv(h)`. The blended vectors are level k's *contribution*; they
//! are added to the reconstruction ẑ and taken off the residual, so each level quantises what the
//! coarser ones left. One bidirectional transformer block decodes ẑ to token logits through a
//! projection onto the shared token embedding.
//!
//! Training (stage 3a): codebooks update by EMA ([`Ema`]), the encoder and φ learn from the
//! commitment and reconstruction terms and the decoder's token loss through the straight-through
//! estimator, and nested dropout truncates the hierarchy at a random coarser level
//! ([`sample_keep`]). Frozen afterwards; the hybrid reads its codebooks, φ and decoder.
//!
//! The resampling matrices are rebuilt and uploaded on every call (a reference cost: fused
//! kernels come later).

use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig, PaddingConfig1d, RmsNorm};
use burn::tensor::activation::log_softmax;
use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::config::HybridConfig;
use crate::layers::{Attention, Mlp, Rope, init_normal, linear, rms_norm};
use crate::quant::QLinear as Linear;

/// Area pooling from `l` positions down to `n` (PyTorch's `interpolate(mode="area")`, an
/// adaptive average pool): output i averages inputs `floor(i l / n) .. ceil((i + 1) l / n)`.
/// Row-major `[n, l]`.
pub fn area_matrix(n: usize, l: usize) -> Vec<f32> {
    let mut m = vec![0.0; n * l];
    for i in 0..n {
        let start = i * l / n;
        let end = ((i + 1) * l).div_ceil(n);
        let w = 1.0 / (end - start) as f32;
        for j in start..end {
            m[i * l + j] = w;
        }
    }
    m
}

/// Linear upsampling from `n` positions to `l` (PyTorch's `interpolate(mode="linear",
/// align_corners=False)`). Row-major `[l, n]`.
pub fn linear_matrix(l: usize, n: usize) -> Vec<f32> {
    let mut m = vec![0.0; l * n];
    let scale = n as f64 / l as f64;
    for j in 0..l {
        let src = ((j as f64 + 0.5) * scale - 0.5).max(0.0);
        let i0 = (src.floor() as usize).min(n - 1);
        let i1 = if i0 + 1 < n { i0 + 1 } else { i0 };
        let w1 = (src - i0 as f64) as f32;
        m[j * n + i0] += 1.0 - w1;
        m[j * n + i1] += w1;
    }
    m
}

fn matrix(values: Vec<f32>, rows: usize, cols: usize, device: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(values, [rows, cols]), device)
}

/// `[B, n, C]` from `x` `[B, l, C]` through the `[n, l]` matrix `m`.
fn resample(m: Tensor<2>, x: Tensor<3>) -> Tensor<3> {
    let [b, _, _] = x.dims();
    let [n, l] = m.dims();
    m.reshape([1, n, l]).expand([b, n, l]).matmul(x)
}

/// The φ convolutions' index for level `k` of `levels`: the one whose tick `j / (count - 1)` is
/// nearest to `k / (levels - 1)`.
pub fn phi_index(k: usize, levels: usize, count: usize) -> usize {
    if count <= 1 || levels <= 1 {
        return 0;
    }
    let at = k as f64 / (levels - 1) as f64;
    (at * (count - 1) as f64).round() as usize
}

/// One level's quantisation of a batch of blocks.
#[derive(Clone, Debug)]
pub struct Level {
    /// Codes `[B, ℓ]`.
    pub codes: Tensor<2, Int>,
    /// The level's contribution to the reconstruction `[B, L, C]`.
    pub contribution: Tensor<3>,
    /// The pooled residual the codes quantise `[B, ℓ, C]` (detached; for the EMA update).
    pub inputs: Tensor<3>,
}

/// What [`BlockTokenizer::quantise`] returns.
#[derive(Clone, Debug)]
pub struct Quantised {
    pub levels: Vec<Level>,
    /// The quantiser loss: per level `commitment * mse(ẑ_k.detach(), z) + mse(ẑ_k, z.detach())`
    /// with ẑ_k the reconstruction after level k, averaged over levels.
    pub loss: Tensor<1>,
}

impl Quantised {
    /// All codes of each block, levels concatenated coarse to fine: `[B, Σℓ]`.
    pub fn codes(&self) -> Tensor<2, Int> {
        Tensor::cat(self.levels.iter().map(|l| l.codes.clone()).collect(), 1)
    }

    /// The reconstruction after the first `keep` levels `[B, L, C]`.
    pub fn reconstruction(&self, keep: usize) -> Tensor<3> {
        let mut parts = self.levels[..keep].iter().map(|l| l.contribution.clone());
        let first = parts.next().expect("at least one level");
        parts.fold(first, |a, b| a + b)
    }
}

/// The tokenizer's training outputs for a batch of blocks.
pub struct TokenizerOut {
    /// Decoder logits `[B, L, V]`.
    pub logits: Tensor<3>,
    pub quantised: Quantised,
}

#[derive(Module, Debug)]
pub struct BlockTokenizer {
    /// Encoder: token embedding `[V, C]` (training only) and positions within the block.
    emb: Embedding,
    pos: Param<Tensor<2>>,
    /// One codebook `[n, C]` per level; trained by EMA, never by gradients.
    codebooks: Vec<Param<Tensor<2>>>,
    phis: Vec<Conv1d>,
    /// The decoder: one bidirectional block of width C and the projection onto the shared
    /// token embedding.
    n1: RmsNorm,
    attn: Attention,
    n2: RmsNorm,
    mlp: Mlp,
    norm: RmsNorm,
    head: Linear,
    levels: Vec<usize>,
    block: usize,
    latent: usize,
    head_dim: usize,
    phi_ratio: f64,
    commitment: f64,
}

impl BlockTokenizer {
    /// A tokenizer for `vocab` tokens whose decoder head projects onto a token embedding of
    /// width `token_dim`.
    pub fn new(
        cfg: &HybridConfig,
        vocab: usize,
        token_dim: usize,
        eps: f64,
        device: &Device,
    ) -> Self {
        cfg.check();
        let c = cfg.latent;
        let codebook = |_| {
            // Unit-scale entries: the encoder's latents start at the embedding's scale.
            init_normal().init([cfg.codebook, c], device)
        };
        let phi = |_| {
            Conv1dConfig::new(c, c, 3).with_padding(PaddingConfig1d::Explicit(1, 1)).init(device)
        };
        Self {
            emb: EmbeddingConfig::new(vocab, c).with_initializer(init_normal()).init(device),
            pos: init_normal().init([cfg.block, c], device),
            codebooks: (0..cfg.depth()).map(codebook).collect(),
            phis: (0..cfg.phi_count).map(phi).collect(),
            n1: rms_norm(c, eps, device),
            attn: Attention::new(c, cfg.tok_heads, cfg.tok_heads, cfg.tok_head_dim, None, device),
            n2: rms_norm(c, eps, device),
            mlp: Mlp::new(c, cfg.tok_ff_dim, device),
            norm: rms_norm(c, eps, device),
            head: linear(c, token_dim, device),
            levels: cfg.levels.clone(),
            block: cfg.block,
            latent: c,
            head_dim: cfg.tok_head_dim,
            phi_ratio: cfg.phi_ratio,
            commitment: cfg.commitment,
        }
    }

    pub fn levels(&self) -> &[usize] {
        &self.levels
    }

    pub fn latent(&self) -> usize {
        self.latent
    }

    /// Parameters loaded at inference: everything but the encoder.
    pub fn inference_params(&self) -> usize {
        self.num_params() - self.encoder_params()
    }

    /// The encoder's parameters (token and position embeddings; training only).
    pub fn encoder_params(&self) -> usize {
        self.emb.num_params() + self.pos.num_params()
    }

    /// The codebooks' parameters.
    pub fn codebook_params(&self) -> usize {
        self.codebooks.iter().map(Module::num_params).sum()
    }

    /// Level `k`'s codebook `[n, C]` (detached: codebooks take no gradient).
    pub fn codebook(&self, k: usize) -> Tensor<2> {
        self.codebooks[k].val().detach()
    }

    /// Latents `z` `[B, L, C]` of blocks `[B, L]`.
    pub fn encode_latents(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        let [b, l] = tokens.dims();
        assert_eq!(l, self.block, "the tokenizer takes whole blocks");
        let c = self.latent;
        self.emb.forward(tokens) + self.pos.val().reshape([1, l, c]).expand([b, l, c])
    }

    /// Level `k`'s nearest codebook entries for `x` `[B, n, C]`: codes `[B, n]`.
    pub fn nearest(&self, k: usize, x: Tensor<3>) -> Tensor<2, Int> {
        let [b, n, c] = x.dims();
        let e = self.codebook(k);
        // argmin ||x - e||^2 = argmax 2 x.e - ||e||^2 (||x||^2 is the same for every entry).
        let norms = e.clone().powf_scalar(2.0).sum_dim(1).reshape([1, e.dims()[0]]);
        let scores = x.reshape([b * n, c]).matmul(e.transpose()) * 2.0 - norms;
        scores.argmax(1).reshape([b, n])
    }

    /// The φ blend of level `k` over `h` `[B, L, C]`.
    fn phi(&self, k: usize, h: Tensor<3>) -> Tensor<3> {
        let conv = &self.phis[phi_index(k, self.levels.len(), self.phis.len())];
        let r = self.phi_ratio;
        let mixed = conv.forward(h.clone().swap_dims(1, 2)).swap_dims(1, 2);
        h * (1.0 - r) + mixed * r
    }

    /// Level `k`'s contribution `[B, L, C]` from its codes `[B, ℓ_k]`: codebook lookup, linear
    /// upsampling to L, φ.
    pub fn contribution(&self, k: usize, codes: Tensor<2, Int>) -> Tensor<3> {
        let device = codes.device();
        let (l, n) = (self.block, self.levels[k]);
        let entries = burn::tensor::module::embedding(self.codebook(k), codes);
        let up = if n == l {
            entries
        } else {
            resample(matrix(linear_matrix(l, n), l, n, &device), entries)
        };
        self.phi(k, up)
    }

    /// `x` `[B, L, C]` area-pooled to level `k`'s ℓ_k positions.
    pub fn pool(&self, k: usize, x: Tensor<3>) -> Tensor<3> {
        let (l, n) = (self.block, self.levels[k]);
        if n == l {
            return x;
        }
        resample(matrix(area_matrix(n, l), n, l, &x.device()), x)
    }

    /// The multiscale residual quantisation of latents `z` `[B, L, C]`.
    pub fn quantise(&self, z: Tensor<3>) -> Quantised {
        let mut rest = z.clone().detach();
        let mut zhat: Option<Tensor<3>> = None;
        let mut levels = Vec::with_capacity(self.levels.len());
        let mut loss: Option<Tensor<1>> = None;
        for k in 0..self.levels.len() {
            let inputs = self.pool(k, rest.clone()).detach();
            let codes = self.nearest(k, inputs.clone());
            let contribution = self.contribution(k, codes.clone());
            let sum = match zhat {
                Some(s) => s + contribution.clone(),
                None => contribution.clone(),
            };
            rest = rest - contribution.clone();
            let term = (sum.clone().detach() - z.clone()).powf_scalar(2.0).mean() * self.commitment
                + (sum.clone() - z.clone().detach()).powf_scalar(2.0).mean();
            loss = Some(match loss {
                Some(l) => l + term,
                None => term,
            });
            zhat = Some(sum);
            levels.push(Level { codes, contribution, inputs });
        }
        let loss = loss.expect("at least one level") / self.levels.len() as f64;
        Quantised { levels, loss }
    }

    /// The codes of blocks `[B, L]`: `[B, Σℓ]`, levels coarse to fine (no gradients needed).
    pub fn encode(&self, tokens: Tensor<2, Int>) -> Tensor<2, Int> {
        self.quantise(self.encode_latents(tokens).detach()).codes()
    }

    /// The reconstruction `[B, L, C]` from all codes `[B, Σℓ]`.
    pub fn reconstruct(&self, codes: Tensor<2, Int>) -> Tensor<3> {
        let mut out: Option<Tensor<3>> = None;
        let mut at = 0;
        for (k, &n) in self.levels.iter().enumerate() {
            let c = self.contribution(k, codes.clone().narrow(1, at, n));
            out = Some(match out {
                Some(o) => o + c,
                None => c,
            });
            at += n;
        }
        out.expect("at least one level")
    }

    /// Decoder logits `[B, L, V]` from reconstructions `[B, L, C]`, through the token embedding
    /// `emb` `[V, d]` the head projects onto.
    pub fn decode(&self, zhat: Tensor<3>, emb: Tensor<2>) -> Tensor<3> {
        let device = zhat.device();
        let rope = Rope::new(0, self.block, self.head_dim, 10_000.0, 1.0, &device);
        let x = zhat.clone()
            + self.attn.forward_masked(crate::fused::rms_norm(&self.n1, zhat), &rope, None);
        let x = x.clone() + self.mlp.forward(crate::fused::rms_norm(&self.n2, x));
        let h = self.head.forward(crate::fused::rms_norm(&self.norm, x));
        crate::fused::linear_transposed(emb, h)
    }

    /// The draft: argmax tokens `[B, L]` of [`Self::decode`].
    pub fn draft(&self, zhat: Tensor<3>, emb: Tensor<2>) -> Tensor<2, Int> {
        let [b, l, _] = zhat.dims();
        crate::fused::argmax_last(self.decode(zhat, emb)).reshape([b, l])
    }

    /// The training forward of blocks `[B, L]`: quantisation, then the decoder over the
    /// straight-through reconstruction truncated to `keep[b]` levels for block b (nested
    /// dropout, [`sample_keep`]; all levels when None).
    pub fn forward_train(
        &self,
        tokens: Tensor<2, Int>,
        keep: Option<&[usize]>,
        emb: Tensor<2>,
    ) -> TokenizerOut {
        let [b, _] = tokens.dims();
        let depth = self.levels.len();
        let z = self.encode_latents(tokens);
        let quantised = self.quantise(z.clone());
        let zhat = match keep {
            None => quantised.reconstruction(depth),
            Some(keep) => {
                assert_eq!(keep.len(), b, "one cut-off per block");
                let device = z.device();
                let mut out: Option<Tensor<3>> = None;
                for (k, level) in quantised.levels.iter().enumerate() {
                    let m: Vec<f32> = keep.iter().map(|&n| if k < n { 1.0 } else { 0.0 }).collect();
                    let m = Tensor::<3>::from_data(TensorData::new(m, [b, 1, 1]), &device);
                    let c = level.contribution.clone() * m;
                    out = Some(match out {
                        Some(o) => o + c,
                        None => c,
                    });
                }
                out.expect("at least one level")
            }
        };
        // Straight through: the forward value is ẑ, the gradient goes to z.
        let zhat = z.clone() + (zhat - z).detach();
        TokenizerOut { logits: self.decode(zhat, emb), quantised }
    }

    /// The codebooks replaced by `ema`'s (after [`Ema::update`]).
    pub fn with_codebooks(mut self, ema: &Ema) -> Self {
        for (k, book) in self.codebooks.iter_mut().enumerate() {
            *book = Param::from_tensor(ema.codebook(k));
        }
        self
    }

    /// Level `k`'s logits over its codebook for vectors `z` `[.., C]` (the generator's head tied
    /// to the codebooks, as in the paper's code): `(2 z.e - ||e||^2) / mean ||e||^2`, whose
    /// argmax is the nearest entry.
    pub fn code_logits(&self, k: usize, z: Tensor<3>) -> Tensor<3> {
        let [b, n, c] = z.dims();
        let e = self.codebook(k);
        let entries = e.dims()[0];
        let norms = e.clone().powf_scalar(2.0).sum_dim(1).reshape([1, entries]);
        let scale = norms.clone().mean().reshape([1, 1]);
        let scores = (z.reshape([b * n, c]).matmul(e.transpose()) * 2.0 - norms) / scale;
        scores.reshape([b, n, entries])
    }
}

/// The tokenizer's training loss: token cross-entropy of the decoder plus the quantiser loss.
pub fn tokenizer_loss(out: &TokenizerOut, tokens: Tensor<2, Int>) -> Tensor<1> {
    let nll = log_softmax(out.logits.clone(), 2).gather(2, tokens.unsqueeze_dim::<3>(2)).neg();
    nll.mean() + out.quantised.loss.clone()
}

/// Nested dropout cut-offs for `blocks` blocks: with probability `p` a block keeps a random
/// prefix of 1..K-1 levels (never the full stack), otherwise all K. `uniform()` draws from
/// [0, 1).
pub fn sample_keep(
    blocks: usize,
    depth: usize,
    p: f64,
    uniform: &mut dyn FnMut() -> f64,
) -> Vec<usize> {
    (0..blocks)
        .map(|_| {
            if depth > 1 && uniform() < p {
                1 + ((uniform() * (depth - 1) as f64) as usize).min(depth - 2)
            } else {
                depth
            }
        })
        .collect()
}

/// EMA state of the codebooks: per level, the smoothed entry counts `[n]` and the sums of the
/// vectors assigned to each entry `[n, C]`.
pub struct Ema {
    counts: Vec<Tensor<1>>,
    sums: Vec<Tensor<2>>,
    decay: f64,
}

impl Ema {
    /// Starts from the tokenizer's codebooks (sums = entries, counts = 1).
    pub fn new(tok: &BlockTokenizer, decay: f64) -> Self {
        let sums: Vec<Tensor<2>> = (0..tok.levels.len()).map(|k| tok.codebook(k)).collect();
        let counts = sums.iter().map(|s| Tensor::ones([s.dims()[0]], &s.device())).collect();
        Self { counts, sums, decay }
    }

    /// One update from a batch's quantisation: `counts = d counts + (1 - d) n_batch`,
    /// `sums = d sums + (1 - d) onehot^T inputs`.
    pub fn update(&mut self, quantised: &Quantised) {
        let d = self.decay;
        for (k, level) in quantised.levels.iter().enumerate() {
            let [b, n, c] = level.inputs.dims();
            let entries = self.counts[k].dims()[0];
            let onehot: Tensor<2> =
                level.codes.clone().reshape([b * n]).one_hot::<2>(entries).float();
            let batch_counts = onehot.clone().sum_dim(0).reshape([entries]);
            let batch_sums = onehot.transpose().matmul(level.inputs.clone().reshape([b * n, c]));
            self.counts[k] = self.counts[k].clone() * d + batch_counts * (1.0 - d);
            self.sums[k] = self.sums[k].clone() * d + batch_sums * (1.0 - d);
        }
    }

    /// Level `k`'s codebook: the sums over Laplace-smoothed counts.
    pub fn codebook(&self, k: usize) -> Tensor<2> {
        const EPS: f64 = 1e-5;
        let counts = self.counts[k].clone();
        let entries = counts.dims()[0];
        let total = counts.clone().sum();
        let smoothed = (counts + EPS) / (total.clone() + entries as f64 * EPS) * total;
        self.sums[k].clone() / smoothed.reshape([entries, 1])
    }
}
