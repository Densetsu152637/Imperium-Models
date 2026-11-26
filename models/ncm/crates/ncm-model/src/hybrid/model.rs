//! The hybrid model's layers and its training forward.
//!
//! * Local encoder (the k = 4 design's, causal over tokens) and **attention pooling** of each
//!   complete block's L states into one backbone input.
//! * **Backbone** (v2, one step per block): its output after block b - 1 conditions block b
//!   (block 0 gets a learned vector).
//! * **Memory** per block ([`super::memory`]): backbone slots, recent encoder states before the
//!   block, retrieved tokens and, for the generator, the block's known prefix.
//! * **Generator**: K passes coarse to fine; level 1 is a learned query, level k > 1 the
//!   cumulative reconstruction of the coarser levels area-pooled to ℓ_k positions and projected,
//!   plus a level embedding; positions restart per level; heads tied to the codebooks. Training
//!   runs all levels in one pass with block-diagonal attention between levels.
//! * **Token decoder**: the k = 4 design's local decoder over the block (causal, input = the
//!   shifted tokens), with an extra input per position, the projected reconstruction ẑ_i (absent
//!   on the autoregressive path), cross-attention to the memory without the known prefix, and a
//!   copy head.

use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, RmsNorm};
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::{Bool, Device, Int, Tensor, TensorData};

use super::memory::{CopyHead, LayerKv, Memory, MemoryAttention};
use super::vq::BlockTokenizer;
use crate::cache::build_mask;
use crate::config::{HybridConfig, NcmConfig};
use crate::layers::{Attention, Mlp, Rope, Stack, init_normal, linear, rms_norm};
use crate::ncm::ncm_specs;
use crate::quant::QLinear as Linear;

/// Attention pooling of a block's states: per head, one learned query scores the block's L
/// (normed) states, and the weighted states, merged over heads, project to the backbone width.
#[derive(Module, Debug)]
pub struct AttnPool {
    norm: RmsNorm,
    query: Param<Tensor<1>>,
    out: Linear,
    heads: usize,
}

impl AttnPool {
    fn new(d: usize, d_out: usize, heads: usize, eps: f64, device: &Device) -> Self {
        assert!(d.is_multiple_of(heads), "pooling heads divide the width");
        Self {
            norm: rms_norm(d, eps, device),
            query: init_normal().init([d], device),
            out: linear(d, d_out, device),
            heads,
        }
    }

    /// `[B, nb, d_out]` from block states `[B, nb, L, d]`.
    pub fn forward(&self, states: Tensor<4>) -> Tensor<3> {
        let [b, nb, l, d] = states.dims();
        let (h, hd) = (self.heads, d / self.heads);
        let s = crate::fused::rms_norm(&self.norm, states).reshape([b, nb, l, h, hd]);
        let q = self.query.val().reshape([1, 1, 1, h, hd]);
        let scores = (s.clone() * q).sum_dim(4) / (hd as f64).sqrt(); // [B, nb, L, H, 1]
        let a = softmax(scores, 2);
        let pooled = (s * a).sum_dim(2).reshape([b, nb, d]);
        self.out.forward(pooled)
    }
}

/// A generator or token-decoder layer: self-attention within the block (or level), cross-
/// attention to the memory, SwiGLU MLP; pre-norm residual.
#[derive(Module, Debug)]
pub struct MemBlock {
    n1: RmsNorm,
    attn: Attention,
    n2: RmsNorm,
    cross: MemoryAttention,
    n3: RmsNorm,
    mlp: Mlp,
}

/// Where a pass's queries sit: self-attention tables and mask (per block), the queries'
/// absolute positions for the memory (all blocks), and the RoPE base.
pub struct PassShape<'a> {
    pub self_rope: &'a Rope,
    pub self_mask: Option<Tensor<2, Bool>>,
    pub q_pos: &'a [usize],
    pub rope_base: f64,
}

/// One block's memory as the fast decode path holds it, per layer: the token states' keys and
/// values (the recent ring, projected once when the states entered it) and the per-block
/// entries' keys and values, with additive biases shared by the layers (0 visible, -inf
/// hidden).
pub struct FastMemory<'a> {
    pub seq: Option<&'a [(Tensor<4>, Tensor<4>)]>,
    pub seq_bias: &'a Tensor<2>,
    pub extra: &'a [(Tensor<4>, Tensor<4>)],
    pub extra_bias: &'a Tensor<2>,
    /// The queries' tables (one row per query).
    pub q_rope: &'a Rope,
}

impl MemBlock {
    /// One block's pass of `x` `[1, q, d]` over a [`FastMemory`] (layer `layer`), with the
    /// fused norm and MLP kernels where they apply (decode-sized runs on CUDA).
    fn forward_fast(
        &self,
        x: Tensor<3>,
        self_rope: &Rope,
        self_mask: Option<Tensor<2, Bool>>,
        memory: &FastMemory,
        layer: usize,
    ) -> Tensor<3> {
        let h = crate::fused::rms_norm(&self.n1, x.clone());
        let attn = self.attn.forward_masked(h, self_rope, self_mask);
        let (x, h) = crate::fused::add_rms_norm(&self.n2, x, attn);
        let seq = memory.seq.map(|s| (&s[layer].0, &s[layer].1, memory.seq_bias));
        let (ek, ev) = &memory.extra[layer];
        let crossed = self.cross.forward_biased(h, memory.q_rope, seq, (ek, ev, memory.extra_bias));
        if let Some(x) = self.mlp.residual_step(&self.n3, &x, Some(&crossed)) {
            return x;
        }
        let (x, h) = crate::fused::add_rms_norm(&self.n3, x, crossed);
        x + self.mlp.forward(h)
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        d: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        ff: usize,
        eps: f64,
        device: &Device,
    ) -> Self {
        Self {
            n1: rms_norm(d, eps, device),
            attn: Attention::new(d, heads, kv_heads, head_dim, None, device),
            n2: rms_norm(d, eps, device),
            cross: MemoryAttention::new(d, heads, kv_heads, head_dim, device),
            n3: rms_norm(d, eps, device),
            mlp: Mlp::new(d, ff, device),
        }
    }

    /// `x` `[B, nb, q, d]`.
    fn forward(&self, x: Tensor<4>, shape: &PassShape, kv: &LayerKv, memory: &Memory) -> Tensor<4> {
        let [b, nb, q, d] = x.dims();
        let h = crate::fused::rms_norm(&self.n1, x.clone()).reshape([b * nb, q, d]);
        let attn = self.attn.forward_masked(h, shape.self_rope, shape.self_mask.clone());
        let x = x + attn.reshape([b, nb, q, d]);
        let h = crate::fused::rms_norm(&self.n2, x.clone()).reshape([b, nb * q, d]);
        let crossed = self.cross.forward(h, shape.q_pos, shape.rope_base, kv, memory);
        let x = x + crossed.reshape([b, nb, q, d]);
        x.clone() + self.mlp.forward(crate::fused::rms_norm(&self.n3, x))
    }
}

/// Parameter counts by part, in the rows of the design's budget table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HybridParts {
    pub embeddings: usize,
    pub encoder: usize,
    pub pooling: usize,
    pub backbone: usize,
    pub generator: usize,
    pub generator_io: usize,
    pub codebooks: usize,
    pub phi: usize,
    pub token_decoder: usize,
    pub token_decoder_io: usize,
    pub tokenizer_decoder: usize,
    pub memory: usize,
    /// The tokenizer's encoder embedding: training only, not loaded at inference.
    pub tokenizer_encoder: usize,
}

impl HybridParts {
    /// Parameters loaded at inference (all but the tokenizer's encoder).
    pub fn inference(&self) -> usize {
        self.rows().iter().map(|r| r.1).sum()
    }

    /// Parameters trained over both stages.
    pub fn trained(&self) -> usize {
        self.inference() + self.tokenizer_encoder
    }

    pub fn rows(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("embeddings (tied)", self.embeddings),
            ("local encoder", self.encoder),
            ("block pooling", self.pooling),
            ("backbone", self.backbone),
            ("generator layers", self.generator),
            ("generator input, levels, head", self.generator_io),
            ("codebooks", self.codebooks),
            ("phi convolutions", self.phi),
            ("token decoder layers", self.token_decoder),
            ("token decoder z input, copy head", self.token_decoder_io),
            ("tokenizer decoder", self.tokenizer_decoder),
            ("memory slots and embeddings", self.memory),
        ]
    }
}

/// Random draws of one training batch: per block (row-major `[B][nb]`) the known-prefix
/// length and whether the token decoder trains without ẑ, and the corrupted conditioning codes
/// `[B * nb, Σℓ]`.
pub struct TrainPlan {
    pub known: Vec<usize>,
    pub zero: Vec<bool>,
    pub cond_codes: Tensor<2, Int>,
}

impl TrainPlan {
    /// No known prefix, ẑ everywhere, no corruption.
    pub fn exact(codes: Tensor<2, Int>) -> Self {
        let n = codes.dims()[0];
        Self { known: vec![0; n], zero: vec![false; n], cond_codes: codes }
    }

    /// The design's draws: a known prefix of 1..L-1 tokens for `known_share` of blocks, ẑ = 0
    /// for `zero_share`, and each conditioning code of level k replaced by another code of that
    /// level with the rate going linearly from `corrupt_first` to `corrupt_last`. `uniform()`
    /// draws from [0, 1).
    pub fn sample(
        h: &HybridConfig,
        codes: Tensor<2, Int>,
        uniform: &mut dyn FnMut() -> f64,
    ) -> Self {
        let [n, per] = codes.dims();
        let device = codes.device();
        let mut values: Vec<i64> =
            codes.into_data().convert::<i64>().try_to_vec::<i64>().expect("int codes");
        let depth = h.depth();
        let entries = h.codebook as i64;
        for row in 0..n {
            for k in 0..depth {
                let t = if depth > 1 { k as f64 / (depth - 1) as f64 } else { 0.0 };
                let rate = h.corrupt_first + (h.corrupt_last - h.corrupt_first) * t;
                let off = row * per + h.level_offset(k);
                for c in &mut values[off..off + h.levels[k]] {
                    if uniform() < rate && entries > 1 {
                        let shift =
                            1 + ((uniform() * (entries - 1) as f64) as i64).min(entries - 2);
                        *c = (*c + shift) % entries;
                    }
                }
            }
        }
        let known = (0..n)
            .map(|_| {
                if uniform() < h.known_share {
                    1 + ((uniform() * (h.block - 1) as f64) as usize).min(h.block - 2)
                } else {
                    0
                }
            })
            .collect();
        let zero = (0..n).map(|_| uniform() < h.zero_share).collect();
        let values: Vec<i32> = values.into_iter().map(|v| v as i32).collect();
        let cond_codes = Tensor::from_data(TensorData::new(values, [n, per]), &device);
        Self { known, zero, cond_codes }
    }
}

/// Outputs of [`HybridLm::forward_train`].
pub struct TrainOut {
    /// Per level, the generator's code logits `[B * nb, ℓ_k, n]`.
    pub code_logits: Vec<Tensor<3>>,
    /// The token decoder's log-probabilities `[B, N, V]` (row t predicts token t).
    pub token_logp: Tensor<3>,
    /// The generator's memory (the token decoder's is the same without the known prefix).
    pub memory: Memory,
}

/// The two training losses.
pub struct HybridLoss {
    /// Level-weighted code cross-entropy.
    pub code: Tensor<1>,
    /// Token cross-entropy over the positions after each block's known prefix.
    pub token: Tensor<1>,
    /// Mean code cross-entropy of each level.
    pub per_level: Vec<Tensor<1>>,
}

#[derive(Module, Debug)]
pub struct HybridLm {
    emb: Embedding,
    encoder: Stack,
    pool: AttnPool,
    backbone: Stack,
    /// The condition of block 0, which has no backbone step before it.
    start_cond: Param<Tensor<1>>,
    slots: Linear,
    known_seg: Param<Tensor<1>>,
    retrieved_pos: Param<Tensor<2>>,
    pub(crate) tokenizer: BlockTokenizer,
    gen_in: Linear,
    level_emb: Param<Tensor<2>>,
    first_query: Param<Tensor<1>>,
    gen_blocks: Vec<MemBlock>,
    gen_norm: RmsNorm,
    gen_head: Linear,
    dec_blocks: Vec<MemBlock>,
    dec_norm: RmsNorm,
    dec_start: Param<Tensor<1>>,
    z_in: Linear,
    copy: CopyHead,
    #[module(skip)]
    pub(crate) h: HybridConfig,
    pub(crate) local_dim: usize,
    pub(crate) head_dim: usize,
    pub(crate) rope_base: f64,
    pub(crate) vocab: usize,
}

impl HybridLm {
    pub fn new(cfg: &NcmConfig, device: &Device) -> Self {
        let h = cfg.hybrid.clone().expect("a hybrid config has a [hybrid] table");
        h.check();
        let (d, eps, hd) = (cfg.local_dim, cfg.norm_eps, cfg.head_dim);
        let (encoder, backbone) = ncm_specs(cfg);
        let gen_block =
            |_| MemBlock::new(d, h.gen_heads, h.gen_kv_heads, hd, h.gen_ff_dim, eps, device);
        let dec_block =
            |_| MemBlock::new(d, cfg.dec_heads, cfg.dec_kv_heads, hd, cfg.dec_ff_dim, eps, device);
        Self {
            emb: EmbeddingConfig::new(cfg.vocab_size, d)
                .with_initializer(init_normal())
                .init(device),
            encoder: Stack::new(&encoder, device),
            pool: AttnPool::new(d, cfg.bb_dim, h.pool_heads, eps, device),
            backbone: Stack::new(&backbone, device),
            start_cond: init_normal().init([cfg.bb_dim], device),
            slots: linear(cfg.bb_dim, h.slots * d, device),
            known_seg: init_normal().init([d], device),
            retrieved_pos: init_normal().init([h.block, d], device),
            tokenizer: BlockTokenizer::new(&h, cfg.vocab_size, d, eps, device),
            gen_in: linear(h.latent, d, device),
            level_emb: init_normal().init([h.depth(), d], device),
            first_query: init_normal().init([d], device),
            gen_blocks: (0..h.gen_layers).map(gen_block).collect(),
            gen_norm: rms_norm(d, eps, device),
            gen_head: linear(d, h.latent, device),
            dec_blocks: (0..cfg.dec_layers).map(dec_block).collect(),
            dec_norm: rms_norm(d, eps, device),
            dec_start: init_normal().init([d], device),
            z_in: linear(h.latent, d, device),
            copy: CopyHead::new(d, h.copy_dim, device),
            h,
            local_dim: d,
            head_dim: hd,
            rope_base: cfg.rope_base,
            vocab: cfg.vocab_size,
        }
    }

    pub fn config(&self) -> &HybridConfig {
        &self.h
    }

    pub fn tokenizer(&self) -> &BlockTokenizer {
        &self.tokenizer
    }

    /// The model with `tokenizer` (a trained, frozen one) in place of its own.
    pub fn with_tokenizer(mut self, tokenizer: BlockTokenizer) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// The tied token embedding `[V, d]`.
    pub fn embedding(&self) -> Tensor<2> {
        self.emb.weight.val()
    }

    pub fn parts(&self) -> HybridParts {
        let tok = &self.tokenizer;
        HybridParts {
            embeddings: self.emb.num_params(),
            encoder: self.encoder.num_params(),
            pooling: self.pool.num_params(),
            backbone: self.backbone.num_params(),
            generator: self.gen_blocks.num_params() + self.gen_norm.num_params(),
            generator_io: self.gen_in.num_params()
                + self.level_emb.num_params()
                + self.first_query.num_params()
                + self.gen_head.num_params(),
            codebooks: tok.codebook_params(),
            phi: tok.phi_params(),
            token_decoder: self.dec_blocks.num_params()
                + self.dec_norm.num_params()
                + self.dec_start.num_params(),
            token_decoder_io: self.z_in.num_params() + self.copy.num_params(),
            tokenizer_decoder: tok.inference_params() - tok.codebook_params() - tok.phi_params(),
            memory: self.slots.num_params()
                + self.known_seg.num_params()
                + self.retrieved_pos.num_params()
                + self.start_cond.num_params(),
            tokenizer_encoder: tok.encoder_params(),
        }
    }

    // ------------------------------------------------------------------ shared pieces

    pub(crate) fn encoder(&self) -> &Stack {
        &self.encoder
    }

    pub(crate) fn backbone(&self) -> &Stack {
        &self.backbone
    }

    pub(crate) fn embed(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        self.emb.forward(tokens)
    }

    /// Backbone inputs `[B, nb, bb]` of complete blocks' states `[B, nb * L, d]`.
    pub(crate) fn pool_blocks(&self, states: Tensor<3>) -> Tensor<3> {
        let [b, n, d] = states.dims();
        let l = self.h.block;
        self.pool.forward(states.reshape([b, n / l, l, d]))
    }

    /// The condition of block 0 `[B, 1, bb]`.
    pub(crate) fn start_condition(&self, b: usize) -> Tensor<3> {
        let bb = self.start_cond.dims()[0];
        self.start_cond.val().reshape([1, 1, bb]).expand([b, 1, bb])
    }

    /// Indices of the earlier blocks whose token ids join block `b`'s memory: the
    /// `retrieved_blocks` best by the dot product of the block's condition with their backbone
    /// inputs, among blocks entirely before the recent window, in stream order.
    ///
    /// The design picks the block positions the last retrieval layer attended to most; this
    /// reference scores them with the condition instead (a stand-in until the retrieval layer
    /// reports its attention; TODO).
    pub fn select_retrieved(&self, cond: &[f32], pooled: &[Vec<f32>], b: usize) -> Vec<usize> {
        let (l, recent) = (self.h.block, self.h.recent);
        let count = self.h.retrieved_blocks;
        if count == 0 || b * l < recent {
            return Vec::new();
        }
        let last = (b * l - recent) / l; // blocks j with (j + 1) L <= bL - recent
        let mut scored: Vec<(f32, usize)> = (0..last.min(pooled.len()))
            .map(|j| (pooled[j].iter().zip(cond).map(|(a, c)| a * c).sum::<f32>(), j))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut picked: Vec<usize> = scored.into_iter().take(count).map(|s| s.1).collect();
        picked.sort_unstable();
        picked
    }

    /// One block's per-block memory entries for both passes. `retrieved` holds per batch row
    /// the picked blocks' token ids (L each, stream order) and the tokens following them;
    /// `known` the encoder states of the known prefix `[B, L - 1, d]` (zero-padded) and each
    /// row's prefix length. Positions: slots and retrieved tokens at `start`, the known prefix at
    /// `start + i`.
    pub fn block_entries(
        &self,
        cond: Tensor<3>,
        retrieved: &[(Vec<i32>, Vec<i32>)],
        known: Option<(Tensor<3>, &[usize])>,
        start: usize,
    ) -> Entries {
        let [b, _, _] = cond.dims();
        let device = cond.device();
        let (d, l, s) = (self.local_dim, self.h.block, self.h.slots);
        let slots = self.slots.forward(cond).reshape([b, 1, s, d]);
        let mut vectors = vec![slots];
        let mut hidden: Vec<Vec<bool>> = vec![vec![false; s]; b];
        let mut pos = vec![start; s];
        let rt = self.h.retrieved_blocks * l;
        let mut copy = None;
        if rt > 0 {
            let mut ids = Vec::with_capacity(b * rt);
            let mut next = Vec::with_capacity(b * rt);
            for (row, (tokens, following)) in retrieved.iter().enumerate() {
                let have = tokens.len().min(rt);
                ids.extend_from_slice(&tokens[..have]);
                next.extend_from_slice(&following[..have]);
                ids.extend(std::iter::repeat_n(0, rt - have));
                next.extend(std::iter::repeat_n(0, rt - have));
                hidden[row].extend((0..rt).map(|i| i >= have));
            }
            let ids = Tensor::<2, Int>::from_data(TensorData::new(ids, [b, rt]), &device);
            let blocks = self.h.retrieved_blocks;
            let pos_emb = self.retrieved_pos.val().reshape([1, 1, l, d]).expand([1, blocks, l, d]);
            let vectors_r = self.emb.forward(ids).reshape([b, 1, rt, d])
                + pos_emb.reshape([1, 1, rt, d]).expand([b, 1, rt, d]);
            vectors.push(vectors_r);
            pos.extend(std::iter::repeat_n(start, rt));
            let next = Tensor::<3, Int>::from_data(TensorData::new(next, [b, 1, rt]), &device);
            copy = Some((s, rt, next));
        }
        let shared = (vectors.len(), hidden.clone(), pos.clone());
        if let Some((states, lens)) = known {
            let seg = self.known_seg.val().reshape([1, 1, 1, d]);
            vectors.push(states.unsqueeze_dim::<4>(1) + seg);
            for (row, &j) in lens.iter().enumerate() {
                hidden[row].extend((0..l - 1).map(|i| i >= j));
            }
            pos.extend((0..l - 1).map(|i| start + i));
        }
        Entries { vectors, hidden, pos, copy, shared }
    }

    // ------------------------------------------------------------------ generator

    /// The generator's input for level `k` of one block, `[B, ℓ_k, d]`: the learned query for
    /// level 0, else the reconstruction so far `zhat` `[B, L, C]` area-pooled and projected; plus
    /// the level embedding.
    pub fn level_input(&self, k: usize, zhat: Option<Tensor<3>>) -> Tensor<3> {
        let d = self.local_dim;
        let level = self.level_emb.val().narrow(0, k, 1).reshape([1, 1, d]);
        match (k, zhat) {
            (0, _) | (_, None) => {
                let b = 1;
                (self.first_query.val().reshape([1, 1, d]) + level).expand([b, 1, d])
            }
            (_, Some(z)) => {
                let x = self.gen_in.forward(self.tokenizer.pool(k, z));
                let [b, n, _] = x.dims();
                x + level.expand([b, n, d])
            }
        }
    }

    /// The generator over inputs `[B, nb, q, d]`: its states projected to the latent width
    /// `[B, nb, q, C]`.
    pub fn run_generator(
        &self,
        x: Tensor<4>,
        shape: &PassShape,
        kvs: &[LayerKv],
        memory: &Memory,
    ) -> Tensor<4> {
        let mut x = x;
        for (blk, kv) in self.gen_blocks.iter().zip(kvs) {
            x = blk.forward(x, shape, kv, memory);
        }
        self.gen_head.forward(crate::fused::rms_norm(&self.gen_norm, x))
    }

    /// Each generator layer's keys and values of `memory`.
    pub fn generator_kv(&self, memory: &Memory) -> Vec<LayerKv> {
        self.gen_blocks.iter().map(|b| b.cross.project(memory, self.rope_base)).collect()
    }

    /// Positions of the generator's training pass, levels concatenated and each restarting at 0.
    pub fn level_positions(&self) -> Vec<usize> {
        self.h.levels.iter().flat_map(|&n| 0..n).collect()
    }

    /// The block-diagonal `[Σℓ, Σℓ]` mask between levels (`true` hides).
    pub fn level_mask(&self, device: &Device) -> Tensor<2, Bool> {
        let level: Vec<usize> = self
            .h
            .levels
            .iter()
            .enumerate()
            .flat_map(|(k, &n)| std::iter::repeat_n(k, n))
            .collect();
        let q = level.len();
        let hidden: Vec<bool> = (0..q * q).map(|i| level[i / q] != level[i % q]).collect();
        Tensor::from_data(TensorData::new(hidden, [q, q]), device)
    }

    // ------------------------------------------------------------------ token decoder

    /// Token-decoder inputs `[B, nb, L, d]` for blocks whose tokens are `tokens` `[B * nb, L]`
    /// (position i reads token i - 1, position 0 a learned start), plus the projected `zhat`
    /// `[B * nb, L, C]` times `zmask` `[B * nb]` when given (None: the autoregressive path).
    pub fn decoder_input(
        &self,
        tokens: Tensor<2, Int>,
        zhat: Option<(Tensor<3>, Tensor<1>)>,
        b: usize,
    ) -> Tensor<4> {
        let [rows, l] = tokens.dims();
        let d = self.local_dim;
        let start = self.dec_start.val().reshape([1, 1, d]).expand([rows, 1, d]);
        let shifted = self.emb.forward(tokens.narrow(1, 0, l - 1));
        let mut x = Tensor::cat(vec![start, shifted], 1);
        if let Some((z, mask)) = zhat {
            x = x + self.z_in.forward(z) * mask.reshape([rows, 1, 1]);
        }
        x.reshape([b, rows / b, l, d])
    }

    /// The token decoder over inputs `[B, nb, L, d]`: log-probabilities `[B, nb * L, V]`.
    pub fn run_decoder(
        &self,
        x: Tensor<4>,
        shape: &PassShape,
        kvs: &[LayerKv],
        memory: &Memory,
    ) -> Tensor<3> {
        let [b, nb, l, d] = x.dims();
        let mut x = x;
        for (blk, kv) in self.dec_blocks.iter().zip(kvs) {
            x = blk.forward(x, shape, kv, memory);
        }
        let h = crate::fused::rms_norm(&self.dec_norm, x).reshape([b, nb * l, d]);
        let logits = crate::fused::linear_transposed(self.emb.weight.val(), h.clone());
        self.copy.mix(h, logits, memory)
    }

    /// Each token-decoder layer's keys and values of `memory`.
    pub fn decoder_kv(&self, memory: &Memory) -> Vec<LayerKv> {
        self.dec_blocks.iter().map(|b| b.cross.project(memory, self.rope_base)).collect()
    }

    /// The token decoder's self-attention tables (block positions 0..L) and causal mask.
    pub fn decoder_self(&self, device: &Device) -> (Rope, Tensor<2, Bool>) {
        let l = self.h.block;
        let pos: Vec<i32> = (0..l as i32).collect();
        let rope = Rope::new(0, l, self.head_dim, self.rope_base, 1.0, device);
        (rope, build_mask(device, &pos, &pos, None))
    }

    // ------------------------------------------------------------------ fast decode path

    /// The generator layers' memory attentions.
    pub fn generator_cross(&self) -> Vec<&MemoryAttention> {
        self.gen_blocks.iter().map(|b| &b.cross).collect()
    }

    /// The token-decoder layers' memory attentions.
    pub fn decoder_cross(&self) -> Vec<&MemoryAttention> {
        self.dec_blocks.iter().map(|b| &b.cross).collect()
    }

    pub fn copy_head(&self) -> &CopyHead {
        &self.copy
    }

    /// The backbone slots `[1, S, d]` of a condition `[1, 1, bb]`.
    pub fn slot_vectors(&self, cond: Tensor<3>) -> Tensor<3> {
        let (s, d) = (self.h.slots, self.local_dim);
        self.slots.forward(cond).reshape([1, s, d])
    }

    /// Retrieved-token entries `[1, Rt, d]` of their ids `[1, Rt]` (block-major, L per block).
    pub fn retrieved_vectors(&self, ids: Tensor<2, Int>) -> Tensor<3> {
        let (l, d) = (self.h.block, self.local_dim);
        let rt = ids.dims()[1];
        let pos = self.retrieved_pos.val().reshape([1, 1, l, d]).expand([1, rt / l, l, d]);
        self.emb.forward(ids) + pos.reshape([1, rt, d])
    }

    /// Known-prefix entries `[1, L - 1, d]` of encoder states `[1, L - 1, d]`.
    pub fn known_vectors(&self, states: Tensor<3>) -> Tensor<3> {
        states + self.known_seg.val().reshape([1, 1, self.local_dim])
    }

    /// [`Self::level_input`] with the tokenizer's tables `r`.
    pub fn level_input_with(
        &self,
        k: usize,
        zhat: Option<Tensor<3>>,
        r: &super::vq::Resampler,
    ) -> Tensor<3> {
        let d = self.local_dim;
        let level = self.level_emb.val().narrow(0, k, 1).reshape([1, 1, d]);
        match (k, zhat) {
            (0, _) | (_, None) => self.first_query.val().reshape([1, 1, d]) + level,
            (_, Some(z)) => {
                let x = self.gen_in.forward(self.tokenizer.pool_with(k, z, r));
                let n = x.dims()[1];
                x + level.expand([1, n, d])
            }
        }
    }

    /// The generator over one level's inputs `[1, q, d]` and a [`FastMemory`]: `[1, q, C]`.
    pub fn generator_fast(&self, x: Tensor<3>, self_rope: &Rope, memory: &FastMemory) -> Tensor<3> {
        let mut x = x;
        for (l, blk) in self.gen_blocks.iter().enumerate() {
            x = blk.forward_fast(x, self_rope, None, memory, l);
        }
        self.gen_head.forward(crate::fused::rms_norm(&self.gen_norm, x))
    }

    /// The token decoder over inputs `[1, L, d]` (causal) and a [`FastMemory`], with the copy
    /// head over `copy` ([`CopyHead::mix_biased`]): log-probabilities `[1, L, V]`.
    #[allow(clippy::type_complexity)]
    pub fn decoder_fast(
        &self,
        x: Tensor<3>,
        self_rope: &Rope,
        causal: Tensor<2, Bool>,
        memory: &FastMemory,
        copy: (
            Option<(&Tensor<3>, &Tensor<2, Int>, &Tensor<2>)>,
            Option<(Tensor<3>, &Tensor<2, Int>, &Tensor<2>)>,
        ),
    ) -> Tensor<3> {
        let mut x = x;
        for (l, blk) in self.dec_blocks.iter().enumerate() {
            x = blk.forward_fast(x, self_rope, Some(causal.clone()), memory, l);
        }
        let h = crate::fused::rms_norm(&self.dec_norm, x);
        let logits = crate::fused::linear_transposed(self.emb.weight.val(), h.clone());
        self.copy.mix_biased(h, logits, copy.0, copy.1)
    }

    /// Token-decoder inputs `[1, L, d]` of block tokens `[1, L]` (position i reads token
    /// i - 1), plus the projected reconstruction `[1, L, C]` when given.
    pub fn decoder_input_fast(&self, tokens: Tensor<2, Int>, zhat: Option<Tensor<3>>) -> Tensor<3> {
        let [_, l] = tokens.dims();
        let d = self.local_dim;
        let start = self.dec_start.val().reshape([1, 1, d]);
        let x = Tensor::cat(vec![start, self.emb.forward(tokens.narrow(1, 0, l - 1))], 1);
        match zhat {
            Some(z) => x + self.z_in.forward(z),
            None => x,
        }
    }

    // ------------------------------------------------------------------ training

    /// The training forward of sequences `tokens` `[B, N]` (N a multiple of L) whose blocks have
    /// codes `codes` `[B * nb, Σℓ]` (the frozen tokenizer's [`BlockTokenizer::encode`] of each
    /// block, row-major), with the batch's random draws `plan`.
    ///
    /// The causality rule holds by construction: block b's memory reads encoder states of tokens
    /// before bL (the last `recent`), the backbone output of blocks before b, retrieved blocks
    /// before the recent window, and (generator only) its own first `known` states.
    pub fn forward_train(
        &self,
        tokens: Tensor<2, Int>,
        codes: Tensor<2, Int>,
        plan: &TrainPlan,
    ) -> TrainOut {
        let [b, n] = tokens.dims();
        let l = self.h.block;
        assert!(n.is_multiple_of(l) && n >= l, "sequences are whole blocks of {l} tokens");
        let nb = n / l;
        let rows = b * nb;
        assert_eq!(codes.dims()[0], rows, "codes for every block");
        let device = tokens.device();
        let d = self.local_dim;
        let host: Vec<i32> =
            tokens.clone().into_data().convert::<i32>().try_to_vec::<i32>().expect("int tokens");

        // Encoder, pooling, backbone; block b is conditioned on the backbone after block b - 1.
        let states = self.encoder.forward(self.emb.forward(tokens.clone()), None);
        let pooled = self.pool_blocks(states.clone());
        let h = self.backbone.forward(pooled.clone(), None);
        let cond = if nb > 1 {
            Tensor::cat(vec![self.start_condition(b), h.narrow(1, 0, nb - 1)], 1)
        } else {
            self.start_condition(b)
        };

        // Memories.
        let pooled_host = rows_of(pooled, b, nb);
        let cond_host = rows_of(cond.clone(), b, nb);
        let known_states = states.clone().reshape([b, nb, l, d]).narrow(2, 0, l - 1);
        let mut per_block = Vec::with_capacity(nb);
        for blk in 0..nb {
            let retrieved: Vec<(Vec<i32>, Vec<i32>)> = (0..b)
                .map(|row| {
                    let seq = &host[row * n..(row + 1) * n];
                    let pooled = &pooled_host[row * nb..(row + 1) * nb];
                    let picked = self.select_retrieved(&cond_host[row * nb + blk], pooled, blk);
                    retrieved_tokens(seq, &picked, l)
                })
                .collect();
            let lens: Vec<usize> = (0..b).map(|row| plan.known[row * nb + blk]).collect();
            let known = known_states.clone().narrow(1, blk, 1).reshape([b, l - 1, d]);
            let c = cond.clone().narrow(1, blk, 1);
            per_block.push(self.block_entries(c, &retrieved, Some((known, &lens)), blk * l));
        }
        let windows: Vec<(usize, usize)> =
            (0..nb).map(|blk| (blk * l - (blk * l).min(self.h.recent), blk * l)).collect();
        let copy_windows: Vec<(usize, usize)> =
            windows.iter().map(|&(lo, hi)| (lo, hi.saturating_sub(1).max(lo))).collect();
        let next: Vec<i32> =
            (0..b * n).map(|i| if (i + 1) % n == 0 { 0 } else { host[i + 1] }).collect();
        let seq_next = Tensor::<2, Int>::from_data(TensorData::new(next, [b, n]), &device);
        let gen_memory = Entries::memory(
            &per_block,
            true,
            Some(states.clone()),
            0,
            &windows,
            &copy_windows,
            Some(seq_next.clone()),
            &device,
        );
        let dec_memory = Entries::memory(
            &per_block,
            false,
            Some(states),
            0,
            &windows,
            &copy_windows,
            Some(seq_next),
            &device,
        );

        // Generator: all levels in one pass, block-diagonal between levels.
        let depth = self.h.depth();
        let mut contributions = Vec::with_capacity(depth);
        for k in 0..depth {
            let at = self.h.level_offset(k);
            let level_codes = plan.cond_codes.clone().narrow(1, at, self.h.levels[k]);
            contributions.push(self.tokenizer.contribution(k, level_codes));
        }
        let mut inputs = Vec::with_capacity(depth);
        let mut zsum: Option<Tensor<3>> = None;
        for (k, c) in contributions.iter().enumerate() {
            let x = self.level_input(k, zsum.clone());
            let [_, q, _] = x.dims();
            inputs.push(x.expand([rows, q, d]));
            zsum = Some(match zsum {
                Some(z) => z + c.clone(),
                None => c.clone(),
            });
        }
        let gx = Tensor::cat(inputs, 1);
        let q = gx.dims()[1];
        let gx = gx.reshape([b, nb, q, d]);
        let self_rope = Rope::at(&self.level_positions(), self.head_dim, self.rope_base, &device);
        let q_pos: Vec<usize> = (0..nb).flat_map(|blk| std::iter::repeat_n(blk * l, q)).collect();
        let shape = PassShape {
            self_rope: &self_rope,
            self_mask: Some(self.level_mask(&device)),
            q_pos: &q_pos,
            rope_base: self.rope_base,
        };
        let kvs = self.generator_kv(&gen_memory);
        let out =
            self.run_generator(gx, &shape, &kvs, &gen_memory).reshape([rows, q, self.h.latent]);
        let code_logits = (0..depth)
            .map(|k| {
                let z = out.clone().narrow(1, self.h.level_offset(k), self.h.levels[k]);
                self.tokenizer.code_logits(k, z)
            })
            .collect();

        // Token decoder on the corrupted reconstruction (or none).
        let zmask: Vec<f32> = plan.zero.iter().map(|&z| if z { 0.0 } else { 1.0 }).collect();
        let zmask = Tensor::<1>::from_data(TensorData::new(zmask, [rows]), &device);
        let zhat = zsum.expect("at least one level");
        let x = self.decoder_input(tokens.reshape([rows, l]), Some((zhat, zmask)), b);
        let (dec_rope, causal) = self.decoder_self(&device);
        let q_pos: Vec<usize> = (0..n).collect();
        let shape = PassShape {
            self_rope: &dec_rope,
            self_mask: Some(causal),
            q_pos: &q_pos,
            rope_base: self.rope_base,
        };
        let kvs = self.decoder_kv(&dec_memory);
        let token_logp = self.run_decoder(x, &shape, &kvs, &dec_memory);
        TrainOut { code_logits, token_logp, memory: gen_memory }
    }

    /// The training losses of [`Self::forward_train`]'s outputs: the code loss (true codes
    /// `codes` `[B * nb, Σℓ]`, level weights ℓ_k^-alpha over each level's positions) and the
    /// token loss over each block's positions from its known prefix on.
    pub fn loss(
        &self,
        out: &TrainOut,
        tokens: Tensor<2, Int>,
        codes: Tensor<2, Int>,
        plan: &TrainPlan,
    ) -> HybridLoss {
        let [b, n] = tokens.dims();
        let l = self.h.block;
        let device = tokens.device();
        let alpha = self.h.level_alpha;
        let mut per_level = Vec::new();
        let mut weighted: Option<Tensor<1>> = None;
        let mut total = 0.0;
        for (k, logits) in out.code_logits.iter().enumerate() {
            let n_k = self.h.levels[k];
            let target = codes.clone().narrow(1, self.h.level_offset(k), n_k).unsqueeze_dim::<3>(2);
            let ce = log_softmax(logits.clone(), 2).gather(2, target).neg().mean();
            let w = (n_k as f64).powf(-alpha) * n_k as f64;
            total += w;
            weighted = Some(match weighted {
                Some(s) => s + ce.clone() * w,
                None => ce.clone() * w,
            });
            per_level.push(ce);
        }
        let code = weighted.expect("at least one level") / total;
        let nll =
            out.token_logp.clone().gather(2, tokens.unsqueeze_dim::<3>(2)).neg().reshape([b, n]);
        let nb = n / l;
        let mask: Vec<f32> = (0..b * n)
            .map(|i| {
                let (row, t) = (i / n, i % n);
                if t % l >= plan.known[row * nb + t / l] { 1.0 } else { 0.0 }
            })
            .collect();
        let mask = Tensor::<2>::from_data(TensorData::new(mask, [b, n]), &device);
        let token = (nll * mask.clone()).sum() / mask.sum();
        HybridLoss { code, token, per_level }
    }
}

/// The tokens of the `picked` blocks of `seq` (L each) and the token following each.
pub fn retrieved_tokens(seq: &[i32], picked: &[usize], l: usize) -> (Vec<i32>, Vec<i32>) {
    let mut ids = Vec::with_capacity(picked.len() * l);
    let mut next = Vec::with_capacity(picked.len() * l);
    for &j in picked {
        for t in j * l..(j + 1) * l {
            ids.push(seq[t]);
            next.push(seq.get(t + 1).copied().unwrap_or(0));
        }
    }
    (ids, next)
}

/// Host rows of `x` `[B, nb, w]`: `B * nb` vectors.
fn rows_of(x: Tensor<3>, b: usize, nb: usize) -> Vec<Vec<f32>> {
    let w = x.dims()[2];
    let v: Vec<f32> = x.into_data().convert::<f32>().try_to_vec::<f32>().expect("float");
    (0..b * nb).map(|r| v[r * w..(r + 1) * w].to_vec()).collect()
}

/// One block's per-block memory entries ([`HybridLm::block_entries`]): vectors `[B, 1, E_i,
/// d]` by segment (slots, retrieved, known prefix), per batch row the hidden flags, the
/// positions, the copy segment, and the shape of the part both passes share (segments, flags,
/// positions without the known prefix).
pub struct Entries {
    vectors: Vec<Tensor<4>>,
    hidden: Vec<Vec<bool>>,
    pos: Vec<usize>,
    copy: Option<(usize, usize, Tensor<3, Int>)>,
    shared: (usize, Vec<Vec<bool>>, Vec<usize>),
}

impl Entries {
    /// The memory of these blocks (one `Entries` per block, in order), with the known prefix
    /// (generator) or without (token decoder), over the token states `seq` at `seq_start`.
    #[allow(clippy::too_many_arguments)]
    pub fn memory(
        blocks: &[Entries],
        with_known: bool,
        seq: Option<Tensor<3>>,
        seq_start: usize,
        windows: &[(usize, usize)],
        copy_windows: &[(usize, usize)],
        seq_next: Option<Tensor<2, Int>>,
        device: &Device,
    ) -> Memory {
        let nb = blocks.len();
        let mut vectors = Vec::with_capacity(nb);
        let mut pos = Vec::new();
        let mut hidden_rows: Vec<Vec<bool>> = Vec::new();
        let mut copies = Vec::new();
        for (i, e) in blocks.iter().enumerate() {
            let (segments, hidden, p) = if with_known {
                (e.vectors.len(), e.hidden.clone(), e.pos.clone())
            } else {
                e.shared.clone()
            };
            vectors.push(Tensor::cat(e.vectors[..segments].to_vec(), 2));
            pos.extend(p);
            for (row, h) in hidden.into_iter().enumerate() {
                if i == 0 {
                    hidden_rows.push(h);
                } else {
                    hidden_rows[row].extend(h);
                }
            }
            if let Some(c) = &e.copy {
                copies.push(c.clone());
            }
        }
        let extra = Tensor::cat(vectors, 1);
        let [b, _, e, _] = extra.dims();
        let flat: Vec<bool> = hidden_rows.into_iter().flatten().collect();
        let extra_hidden = Tensor::from_data(TensorData::new(flat, [b, nb, e]), device);
        let copy_extra = (copies.len() == nb && nb > 0).then(|| {
            let (off, len) = (copies[0].0, copies[0].1);
            (off, len, Tensor::cat(copies.into_iter().map(|c| c.2).collect(), 1))
        });
        Memory {
            nb,
            seq,
            seq_start,
            windows: windows.to_vec(),
            copy_windows: copy_windows.to_vec(),
            seq_next,
            extra,
            extra_pos: pos,
            extra_hidden,
            copy_extra,
        }
    }
}
