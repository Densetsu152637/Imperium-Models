//! Next Concept Model: local encoder, k-token concepts, global backbone, local decoder.
//!
//! Layout of a sequence of N = M * k tokens (M concepts):
//!
//! * The local encoder (causal, sliding window) turns tokens into states.
//! * The k states of concept m are concatenated and projected into one backbone vector.
//! * The backbone (causal over concepts) output at concept m, `h_m`, has seen concepts 0..m only.
//! * The local decoder for concept m + 1 is conditioned on `h_m` and the tokens of concept m + 1
//!   it has already written, and cross-attends to encoder states of tokens up to the end of
//!   concept m (the most recent `enc_window` of them).
//!
//! Concept 0 is context only: it has no earlier concept to condition on, so logits cover tokens
//! k .. N - 1 (`logits[:, t - k]` predicts token t).

use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, RmsNorm};
use burn::tensor::{Bool, Device, Int, Tensor};
#[cfg(feature = "cuda")]
use burn::{
    backend::{Backend, BackendGraph, Dispatch},
    tensor::ExecutionError,
};

use crate::cache::{KvCache, LayerCache, build_mask, layer_cache_bytes};
use crate::config::{Design, NcmConfig};
use crate::layers::{
    Attention, CrossAttention, LayerKind, Mlp, Rope, Stack, StackSpec, fallback_kinds,
    hybrid_kinds, init_normal, linear, rms_norm,
};
use crate::quant::{QLinear as Linear, tied_head};
use crate::speculative::DraftHead;

#[derive(Module, Debug)]
pub struct DecoderBlock {
    n1: RmsNorm,
    self_attn: Attention,
    n2: RmsNorm,
    cross: CrossAttention,
    n3: RmsNorm,
    mlp: Mlp,
}

impl DecoderBlock {
    fn new(cfg: &NcmConfig, device: &Device) -> Self {
        let (d, hd, eps) = (cfg.local_dim, cfg.head_dim, cfg.norm_eps);
        Self {
            n1: rms_norm(d, eps, device),
            self_attn: Attention::new(d, cfg.dec_heads, cfg.dec_kv_heads, hd, None, device),
            n2: rms_norm(d, eps, device),
            cross: CrossAttention::new(d, d, cfg.dec_heads, cfg.dec_kv_heads, hd, device),
            n3: rms_norm(d, eps, device),
            mlp: Mlp::new(d, cfg.dec_ff_dim, device),
        }
    }

    /// `x` is `[B, M', k, d]`: every concept's decoder runs in parallel, teacher-forced.
    /// `causal` is the `[k, k]` mask inside a concept; `mask` hides encoder states per query
    /// (None: every key is visible, as for one concept against the decode cache).
    #[allow(clippy::too_many_arguments)]
    fn forward_train(
        &self,
        x: Tensor<4>,
        rope_self: &Rope,
        rope_q: &Rope,
        causal: Tensor<2, Bool>,
        keys: Tensor<4>,
        values: Tensor<4>,
        mask: Option<Tensor<2, Bool>>,
    ) -> Tensor<4> {
        let [b, m, k, d] = x.dims();
        let h = self.n1.forward(x.clone()).reshape([b * m, k, d]);
        let x = x + self.self_attn.forward_masked(h, rope_self, Some(causal)).reshape([b, m, k, d]);
        let h = self.n2.forward(x.clone()).reshape([b, m * k, d]);
        let x = x + self.cross.forward(h, rope_q, keys, values, mask).reshape([b, m, k, d]);
        x.clone() + self.mlp.forward(self.n3.forward(x))
    }

    fn step(
        &self,
        x: Tensor<3>,
        rope_self: &Rope,
        rope_q: &Rope,
        self_cache: &mut KvCache,
        cross_cache: &KvCache,
    ) -> Tensor<3> {
        let x = x.clone() + self.self_attn.forward(self.n1.forward(x), rope_self, Some(self_cache));
        let (keys, values) = cross_cache.view();
        let x = x.clone() + self.cross.forward(self.n2.forward(x), rope_q, keys, values, None);
        x.clone() + self.mlp.forward(self.n3.forward(x))
    }
}

/// Most tokens the encoder takes in one call during prefill (see [`NcmLm::append_tokens`]).
pub const ENCODER_CHUNK: usize = 1024;

/// All caches of one NCM generation run.
pub struct NcmCache {
    pub backbone: Vec<LayerCache>,
    pub encoder: Vec<LayerCache>,
    /// Per decoder layer: encoder-state keys and values, a ring of `enc_window`.
    pub cross: Vec<KvCache>,
    /// Per decoder layer: only the current concept, reset every concept.
    pub dec_self: Vec<KvCache>,
    /// Tokens consumed by the encoder so far.
    pub tokens: usize,
    /// One-concept backbone steps as a replayed CUDA graph (v2 backbones on CUDA).
    #[cfg(feature = "cuda")]
    pub backbone_graph: crate::layers::GraphedStack,
}

impl NcmCache {
    /// Bytes allocated.
    pub fn nbytes(&self) -> usize {
        let stacks = self.backbone.iter().chain(&self.encoder).map(LayerCache::nbytes);
        let decoder = self.cross.iter().chain(&self.dec_self).map(KvCache::nbytes);
        stacks.chain(decoder).sum()
    }

    /// Forget everything, ready for a new sequence.
    pub fn reset(&mut self) {
        self.backbone.iter_mut().chain(&mut self.encoder).for_each(LayerCache::reset);
        self.cross.iter_mut().chain(&mut self.dec_self).for_each(KvCache::reset);
        self.tokens = 0;
        // Resetting replaces the DeltaNet state buffers the graph was recorded against.
        #[cfg(feature = "cuda")]
        {
            self.backbone_graph = Default::default();
        }
    }
}

/// The encoder and backbone stacks of an NCM config.
pub fn ncm_specs(cfg: &NcmConfig) -> (StackSpec, StackSpec) {
    let v2 = cfg.attention == Design::V2;
    let window = LayerKind::Attention { window: Some(cfg.enc_window) };
    let mut enc_kinds = vec![window; cfg.enc_layers];
    if v2 {
        // One gated DeltaNet layer after the first sliding-window layer.
        enc_kinds.insert(1.min(cfg.enc_layers), LayerKind::DeltaNet);
    }
    let encoder = StackSpec {
        d: cfg.local_dim,
        ff: cfg.enc_ff_dim,
        kinds: enc_kinds,
        heads: cfg.enc_heads,
        kv_heads: cfg.enc_kv_heads,
        head_dim: cfg.head_dim,
        gated: v2,
        rope_base: cfg.rope_base,
        rope_scale: cfg.rope_scale,
        rope_base_full: cfg.rope_base_full,
        rope_scale_full: cfg.rope_scale_full,
        eps: cfg.norm_eps,
        v2: cfg.v2.clone(),
    };
    let bb_kinds = if v2 {
        hybrid_kinds(cfg.bb_layers)
    } else {
        fallback_kinds(cfg.bb_layers, cfg.bb_window, cfg.full_every)
    };
    let backbone = StackSpec {
        d: cfg.bb_dim,
        ff: cfg.bb_ff_dim,
        kinds: bb_kinds,
        heads: cfg.bb_heads,
        kv_heads: cfg.bb_kv_heads,
        ..encoder.clone()
    };
    (encoder, backbone)
}

pub struct NcmParts {
    pub embeddings: usize,
    pub encoder: usize,
    pub backbone: usize,
    pub decoder: usize,
    /// The parallel draft head (0 without one).
    pub draft: usize,
    pub projections: usize,
}

impl NcmParts {
    pub fn total(&self) -> usize {
        self.embeddings
            + self.encoder
            + self.backbone
            + self.decoder
            + self.draft
            + self.projections
    }

    pub fn rows(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("embeddings (tied)", self.embeddings),
            ("local encoder", self.encoder),
            ("global backbone", self.backbone),
            ("local decoder", self.decoder),
            ("draft head", self.draft),
            ("projections", self.projections),
        ]
    }
}

#[derive(Module, Debug)]
pub struct NcmLm {
    emb: Embedding,
    encoder: Stack,
    pool: Linear,
    backbone: Stack,
    cond_proj: Linear,
    start_emb: Param<Tensor<1>>,
    dec_blocks: Vec<DecoderBlock>,
    dec_norm: RmsNorm,
    draft: Option<DraftHead>,
    /// A quantised copy of the tied output head (see [`crate::quant`]); None: logits use `emb`.
    head: Option<Linear>,
    k: usize,
    local_dim: usize,
    head_dim: usize,
    enc_window: usize,
    dec_kv_heads: usize,
    rope_base: f64,
    rope_scale: f64,
}

impl NcmLm {
    pub fn new(cfg: &NcmConfig, device: &Device) -> Self {
        assert!(cfg.k >= 2, "concepts need at least two tokens");
        assert!(
            cfg.draft_layers <= cfg.dec_layers,
            "draft layer i reads decoder layer i's cross-attention cache: draft_layers <= dec_layers"
        );
        let d = cfg.local_dim;
        let (encoder, backbone) = ncm_specs(cfg);
        Self {
            emb: EmbeddingConfig::new(cfg.vocab_size, d)
                .with_initializer(init_normal())
                .init(device),
            encoder: Stack::new(&encoder, device),
            pool: linear(cfg.k * d, cfg.bb_dim, device),
            backbone: Stack::new(&backbone, device),
            cond_proj: linear(cfg.bb_dim, d, device),
            start_emb: init_normal().init([d], device),
            dec_blocks: (0..cfg.dec_layers).map(|_| DecoderBlock::new(cfg, device)).collect(),
            dec_norm: rms_norm(d, cfg.norm_eps, device),
            draft: (cfg.draft_layers > 0).then(|| DraftHead::new(cfg, device)),
            head: tied_head(cfg.vocab_size, d, device),
            k: cfg.k,
            local_dim: d,
            head_dim: cfg.head_dim,
            enc_window: cfg.enc_window,
            dec_kv_heads: cfg.dec_kv_heads,
            rope_base: cfg.rope_base,
            rope_scale: cfg.rope_scale,
        }
    }

    /// Tokens per concept.
    pub fn concept_size(&self) -> usize {
        self.k
    }

    /// Whether the model has the parallel draft head that self-speculative decoding needs.
    pub fn has_draft_head(&self) -> bool {
        self.draft.is_some()
    }

    pub fn parts(&self) -> NcmParts {
        NcmParts {
            embeddings: self.emb.num_params(),
            encoder: self.encoder.num_params(),
            backbone: self.backbone.num_params(),
            decoder: self.dec_blocks.num_params() + self.dec_norm.num_params(),
            draft: self.draft.as_ref().map_or(0, Module::num_params),
            projections: self.pool.num_params()
                + self.cond_proj.num_params()
                + self.start_emb.num_params(),
        }
    }

    pub(crate) fn rope(&self, start: usize, len: usize, device: &Device) -> Rope {
        Rope::new(start, len, self.head_dim, self.rope_base, self.rope_scale, device)
    }

    fn logits(&self, x: Tensor<3>) -> Tensor<3> {
        self.project(self.dec_norm.forward(x))
    }

    /// Normed states `[B, T, d]` to logits through the output head: its quantised copy when the
    /// model has one, else the tied embedding.
    pub(crate) fn project(&self, x: Tensor<3>) -> Tensor<3> {
        match &self.head {
            Some(head) => head.forward(x),
            None => crate::fused::linear_transposed(self.emb.weight.val(), x),
        }
    }

    // ------------------------------------------------------------------ full sequence

    /// Logits `[B, N - k, V]` for tokens k..N-1 of a sequence whose length is a multiple of k.
    pub fn forward(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        self.forward_parts(tokens, false).0
    }

    /// [`Self::forward`] and the draft head's logits `[B, N - k, V]` over the same positions
    /// (`draft[:, t - k]` is the draft head's guess for token t, made from the concept's
    /// condition alone), sharing the encoder and backbone. The training loss adds
    /// [`crate::speculative::draft_loss`] of the draft logits. Panics without a draft head.
    pub fn forward_with_draft(&self, tokens: Tensor<2, Int>) -> (Tensor<3>, Tensor<3>) {
        let (logits, draft) = self.forward_parts(tokens, true);
        (logits, draft.expect("forward_with_draft needs a model with a draft head"))
    }

    fn forward_parts(&self, tokens: Tensor<2, Int>, draft: bool) -> (Tensor<3>, Option<Tensor<3>>) {
        let k = self.k;
        let [b, n] = tokens.dims();
        assert!(
            n % k == 0 && n >= 2 * k,
            "sequence length must be a multiple of k={k} and at least {}",
            2 * k
        );
        let (m, d) = (n / k, self.local_dim);
        let device = tokens.device();
        let states = self.encoder.forward(self.emb.forward(tokens.clone()), None);
        let concepts = self.pool.forward(states.clone().reshape([b, m, k * d]));
        let h = self.backbone.forward(concepts, None);
        // [B, M-1, 1, d]: conditions concepts 1..M-1.
        let cond = self.cond_proj.forward(h.narrow(1, 0, m - 1)).unsqueeze_dim::<4>(2);
        let targets = tokens.narrow(1, k, n - k).reshape([b, m - 1, k]);
        let written = self
            .emb
            .forward(targets.narrow(2, 0, k - 1).reshape([b, (m - 1) * (k - 1)]))
            .reshape([b, m - 1, k - 1, d]);
        let start = self.start_emb.val().reshape([1, 1, 1, d]).expand([b, m - 1, 1, d]);
        let mut x = Tensor::cat(vec![start, written], 2) + cond.clone();
        let rope_self = self.rope(0, k, &device);
        let rope_q = self.rope(k, n - k, &device);
        let rope_k = self.rope(0, n, &device);
        let causal = self.causal_mask(&device);
        // Concept m' may see encoder states of tokens [k*m' - enc_window, k*m' - 1].
        let ends: Vec<i32> = (k..n).map(|t| ((t / k) * k) as i32 - 1).collect();
        let kpos: Vec<i32> = (0..n as i32).collect();
        let mask = build_mask(&device, &ends, &kpos, Some(self.enc_window));
        let mut kvs = Vec::new();
        for blk in &self.dec_blocks {
            let (keys, values) = blk.cross.make_kv(states.clone(), &rope_k);
            let (rs, rq, c) = (&rope_self, &rope_q, causal.clone());
            x = blk.forward_train(x, rs, rq, c, keys.clone(), values.clone(), Some(mask.clone()));
            // The draft head reads the decoder layers' encoder keys and values.
            if draft {
                kvs.push((keys, values));
            }
        }
        let logits = self.logits(x.reshape([b, (m - 1) * k, d]));
        let draft = draft.then(|| {
            let head = self.draft.as_ref().expect("forward_with_draft needs a draft head");
            let x = head.forward(cond, &rope_self, &rope_q, &kvs, Some(mask));
            head.logits(self, x.reshape([b, (m - 1) * k, d]))
        });
        (logits, draft)
    }

    /// The `[k, k]` causal mask of a concept's own tokens.
    pub(crate) fn causal_mask(&self, device: &Device) -> Tensor<2, Bool> {
        let pos: Vec<i32> = (0..self.k as i32).collect();
        build_mask(device, &pos, &pos, None)
    }

    // ------------------------------------------------------------------ incremental

    pub fn make_cache(&self, max_tokens: usize, batch: usize, device: &Device) -> NcmCache {
        let concepts = max_tokens.div_ceil(self.k);
        NcmCache {
            backbone: self.backbone.make_caches(concepts, batch, device),
            encoder: self.encoder.make_caches(max_tokens, batch, device),
            cross: (0..self.dec_blocks.len())
                .map(|_| {
                    KvCache::make(
                        batch,
                        self.dec_kv_heads,
                        self.head_dim,
                        max_tokens,
                        Some(self.enc_window),
                        device,
                    )
                })
                .collect(),
            dec_self: (0..self.dec_blocks.len())
                .map(|_| {
                    KvCache::make(batch, self.dec_kv_heads, self.head_dim, self.k, None, device)
                })
                .collect(),
            tokens: 0,
            #[cfg(feature = "cuda")]
            backbone_graph: Default::default(),
        }
    }

    /// Encode whole concepts and run the backbone on them. Returns the conditioning vector
    /// `[B, 1, d]` for the next concept's decoder.
    pub fn append_tokens(&self, tokens: Tensor<2, Int>, cache: &mut NcmCache) -> Tensor<3> {
        let k = self.k;
        let [b, t] = tokens.dims();
        assert!(t % k == 0, "token count must be a multiple of k={k}");
        let device = tokens.device();
        let states = self.encode(tokens, cache);
        let rope_k = self.rope(cache.tokens, t, &device);
        for (blk, kv_cache) in self.dec_blocks.iter().zip(cache.cross.iter_mut()) {
            let (keys, values) = blk.cross.make_kv(states.clone(), &rope_k);
            kv_cache.write(keys, values);
        }
        let concepts = self.pool.forward(states.reshape([b, t / k, k * self.local_dim]));
        #[cfg(feature = "cuda")]
        let h = cache.backbone_graph.forward(&self.backbone, concepts, &mut cache.backbone);
        #[cfg(not(feature = "cuda"))]
        let h = self.backbone.forward(concepts, Some(&mut cache.backbone));
        cache.tokens += t;
        self.cond_proj.forward(h.narrow(1, t / k - 1, 1))
    }

    /// The encoder states `[B, T, d]` of `tokens` `[B, T]`, continuing from its cache. Long runs
    /// go through in pieces of [`ENCODER_CHUNK`] tokens: the sliding-window layers attend over
    /// their window plus the whole piece, so one long piece would cost quadratically in it.
    fn encode(&self, tokens: Tensor<2, Int>, cache: &mut NcmCache) -> Tensor<3> {
        let t = tokens.dims()[1];
        let x = self.emb.forward(tokens);
        if t <= ENCODER_CHUNK {
            return self.encoder.forward(x, Some(&mut cache.encoder));
        }
        let pieces = (0..t)
            .step_by(ENCODER_CHUNK)
            .map(|start| {
                let len = ENCODER_CHUNK.min(t - start);
                self.encoder.forward(x.clone().narrow(1, start, len), Some(&mut cache.encoder))
            })
            .collect();
        Tensor::cat(pieces, 1)
    }

    /// Feed a prompt (length a multiple of k) in chunks; returns the next-concept condition.
    pub fn prefill(&self, tokens: Tensor<2, Int>, cache: &mut NcmCache, chunk: usize) -> Tensor<3> {
        let chunk = (chunk - chunk % self.k).max(self.k);
        let n = tokens.dims()[1];
        let mut cond = None;
        let mut start = 0;
        while start < n {
            let len = chunk.min(n - start);
            cond = Some(self.append_tokens(tokens.clone().narrow(1, start, len), cache));
            start += len;
        }
        cond.expect("prompt must not be empty")
    }

    /// Write the next k tokens greedily (or teacher-forced) with the local decoder.
    ///
    /// Returns tokens `[B, k]` and, if requested, logits `[B, k, V]`. The tokens are not yet
    /// encoded; pass them to `append_tokens` to continue.
    pub fn decode_concept(
        &self,
        cond: Tensor<3>,
        cache: &mut NcmCache,
        forced: Option<Tensor<2, Int>>,
        want_logits: bool,
    ) -> (Tensor<2, Int>, Option<Tensor<3>>) {
        self.decode_concept_with(cond, cache, want_logits, &mut |j, logits| match &forced {
            Some(forced) => forced.clone().narrow(1, j, 1),
            None => crate::fused::argmax_last(logits.clone()).reshape([logits.dims()[0], 1]),
        })
    }

    /// Like `decode_concept`, but `pick(j, logits)` chooses token `j` of the concept from its
    /// logits `[B, 1, V]` and returns it as `[B, 1]`. It is how a runtime samples, masks logits
    /// or constrains the output; it sees the tokens it picked earlier in the concept because
    /// the decoder feeds each one back before the next call.
    pub fn decode_concept_with(
        &self,
        cond: Tensor<3>,
        cache: &mut NcmCache,
        want_logits: bool,
        pick: &mut dyn FnMut(usize, &Tensor<3>) -> Tensor<2, Int>,
    ) -> (Tensor<2, Int>, Option<Tensor<3>>) {
        let device = cond.device();
        let rope_self = self.rope(0, self.k, &device);
        let rope_q = self.rope(cache.tokens, self.k, &device);
        self.decode_with_ropes(cond, cache, &rope_self, &rope_q, want_logits, pick)
    }

    /// The decoder pass of [`Self::decode_concept_with`] with the RoPE tables of the concept's
    /// own positions (`rope_self`, always 0..k) and of its absolute positions (`rope_q`).
    fn decode_with_ropes(
        &self,
        cond: Tensor<3>,
        cache: &mut NcmCache,
        rope_self: &Rope,
        rope_q: &Rope,
        want_logits: bool,
        pick: &mut dyn FnMut(usize, &Tensor<3>) -> Tensor<2, Int>,
    ) -> (Tensor<2, Int>, Option<Tensor<3>>) {
        let k = self.k;
        let [b, _, _] = cond.dims();
        for c in &mut cache.dec_self {
            c.reset();
        }
        let mut tokens = Vec::with_capacity(k);
        let mut logits = Vec::new();
        let mut inp =
            self.start_emb.val().reshape([1, 1, self.local_dim]).expand([b, 1, self.local_dim]);
        for j in 0..k {
            let mut x = inp + cond.clone();
            let (rs, rq) = (rope_self.rows(j, 1), rope_q.rows(j, 1));
            let layers =
                self.dec_blocks.iter().zip(cache.dec_self.iter_mut()).zip(cache.cross.iter());
            for ((blk, self_cache), cross_cache) in layers {
                x = blk.step(x, &rs, &rq, self_cache, cross_cache);
            }
            let lg = self.logits(x);
            let next = pick(j, &lg);
            if want_logits {
                logits.push(lg);
            }
            inp = self.emb.forward(next.clone());
            tokens.push(next);
        }
        let logits = want_logits.then(|| Tensor::cat(logits, 1));
        (Tensor::cat(tokens, 1), logits)
    }
}

/// The two passes of self-speculative decoding ([`crate::speculative`]).
impl NcmLm {
    /// The draft head's logits `[B, k, V]` for every token of the next concept, in one pass: its
    /// slot queries, conditioned on `cond` `[B, 1, d]`, attend to each other and to the encoder
    /// states held in the decoder's cross-attention caches. Panics without a draft head.
    pub fn draft_concept(&self, cond: Tensor<3>, cache: &NcmCache) -> Tensor<3> {
        let device = cond.device();
        let rope_self = self.rope(0, self.k, &device);
        let rope_q = self.rope(cache.tokens, self.k, &device);
        self.draft_with_ropes(cond, cache, &rope_self, &rope_q)
    }

    /// The local decoder's logits `[B, k, V]` for the next concept, teacher-forced on `tokens`
    /// `[B, k]` in one parallel pass, causal within the concept: row j is the decoder's
    /// distribution for token j given `tokens[.., ..j]`, as [`Self::decode_concept`] would give
    /// it step by step (the last token is not read). The decoder caches are only read.
    pub fn verify_concept(
        &self,
        cond: Tensor<3>,
        cache: &NcmCache,
        tokens: Tensor<2, Int>,
    ) -> Tensor<3> {
        let device = cond.device();
        let rope_self = self.rope(0, self.k, &device);
        let rope_q = self.rope(cache.tokens, self.k, &device);
        let causal = self.causal_mask(&device);
        self.verify_with_ropes(cond, cache, tokens, &rope_self, &rope_q, causal)
    }

    /// [`Self::draft_concept`] with the RoPE tables given.
    pub(crate) fn draft_with_ropes(
        &self,
        cond: Tensor<3>,
        cache: &NcmCache,
        rope_self: &Rope,
        rope_q: &Rope,
    ) -> Tensor<3> {
        let head = self.draft.as_ref().expect("speculative decoding needs a draft head");
        let [b, _, d] = cond.dims();
        let kvs: Vec<_> = cache.cross.iter().map(KvCache::view).collect();
        let x = head.forward(cond.unsqueeze_dim::<4>(1), rope_self, rope_q, &kvs, None);
        head.logits(self, x.reshape([b, self.k, d]))
    }

    /// [`Self::verify_concept`] with the RoPE tables and the causal mask given (no uploads, so
    /// it can be captured into a CUDA graph).
    pub(crate) fn verify_with_ropes(
        &self,
        cond: Tensor<3>,
        cache: &NcmCache,
        tokens: Tensor<2, Int>,
        rope_self: &Rope,
        rope_q: &Rope,
        causal: Tensor<2, Bool>,
    ) -> Tensor<3> {
        let (k, d) = (self.k, self.local_dim);
        let [b, _, _] = cond.dims();
        let written = self.emb.forward(tokens.narrow(1, 0, k - 1));
        let start = self.start_emb.val().reshape([1, 1, d]).expand([b, 1, d]);
        let mut x = (Tensor::cat(vec![start, written], 1) + cond).unsqueeze_dim::<4>(1);
        for (blk, cross) in self.dec_blocks.iter().zip(&cache.cross) {
            let (keys, values) = cross.view();
            x = blk.forward_train(x, rope_self, rope_q, causal.clone(), keys, values, None);
        }
        self.logits(x.reshape([b, k, d]))
    }
}

/// Cache bytes at `max_tokens`; indexer keys (v2) at `index_itemsize`, the rest at `itemsize`.
pub fn ncm_cache_bytes(
    cfg: &NcmConfig,
    max_tokens: usize,
    itemsize: usize,
    index_itemsize: usize,
    batch: usize,
) -> usize {
    let (encoder, backbone) = ncm_specs(cfg);
    let concepts = max_tokens.div_ceil(cfg.k);
    let backbone = backbone.cache_bytes(concepts, itemsize, index_itemsize, batch);
    let encoder = encoder.cache_bytes(max_tokens, itemsize, index_itemsize, batch);
    // Per decoder layer: the cross-attention ring and the per-concept self cache.
    let (kvh, hd) = (cfg.dec_kv_heads, cfg.head_dim);
    let cross = layer_cache_bytes(kvh, hd, max_tokens, Some(cfg.enc_window), itemsize, batch);
    let dec_self = layer_cache_bytes(kvh, hd, cfg.k, None, itemsize, batch);
    backbone + encoder + cfg.dec_layers * (cross + dec_self)
}

/// The NCM's local decoder as one captured CUDA graph per cache, replayed for every concept.
///
/// The decoder is most of the NCM's per-token kernel launches (four layers of attention, norms
/// and MLP for each of the k tokens, then the logits and the greedy pick), and batch-1 decode is
/// bound by issuing them. Its work is the same for every concept once the cross-attention ring
/// is full: the per-concept self-attention cache restarts at slot 0, the ring keeps its buffers
/// (written in place, [`crate::kernels::assign_into`]), and what changes, the condition and the
/// RoPE tables of the concept's positions, is copied into fixed input buffers before each
/// replay. The pick is greedy (argmax). On other devices, with the fused kernels off, before the
/// ring fills, or if the capture fails, [`GraphedDecoder::decode`] runs the decoder eagerly.
#[cfg(feature = "cuda")]
#[derive(Default)]
pub struct GraphedDecoder {
    state: GraphState,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
enum GraphState {
    #[default]
    Untried,
    Failed(String),
    Ready(Box<DecoderGraph>),
}

#[cfg(feature = "cuda")]
struct DecoderGraph {
    graph: BackendGraph<Dispatch>,
    cond: Tensor<3>,
    rope_q: Rope,
    /// Kept for the buffers the graph reads.
    _rope_self: Rope,
    tokens: Tensor<2, Int>,
}

#[cfg(feature = "cuda")]
impl GraphedDecoder {
    /// Greedy tokens `[B, k]` of the next concept, like `model.decode_concept(cond, cache, None,
    /// false)`. The returned tensor is the graph's output buffer: read it (or pass it to
    /// `append_tokens`) before the next call, which overwrites it. One `GraphedDecoder` serves
    /// one cache.
    pub fn decode(
        &mut self,
        model: &NcmLm,
        cond: Tensor<3>,
        cache: &mut NcmCache,
    ) -> Tensor<2, Int> {
        let device = cond.device();
        let ring_full = cache.cross.iter().all(|c| c.len() >= c.capacity());
        if matches!(self.state, GraphState::Untried)
            && ring_full
            && crate::fused::graphs_active(&device)
        {
            self.state = match model.capture_decoder(&cond, cache) {
                Ok(graph) => GraphState::Ready(Box::new(graph)),
                Err(e) => GraphState::Failed(e),
            };
        }
        let GraphState::Ready(g) = &self.state else {
            return model.decode_concept(cond, cache, None, false).0;
        };
        let (cos, sin) = model.rope(cache.tokens, model.k, &device).tables();
        let (q_cos, q_sin) = g.rope_q.tables();
        crate::kernels::assign_into(&g.cond, cond, 0, 0);
        crate::kernels::assign_into(&q_cos, cos, 0, 0);
        crate::kernels::assign_into(&q_sin, sin, 0, 0);
        // Safety: every buffer the graph touched is held by `g` (its inputs and output), by the
        // model (weights) or by `cache` (the decoder caches), all of which outlive the replay,
        // and all work runs on this thread's stream.
        unsafe { Dispatch::graph_replay(device.as_dispatch(), &g.graph) }
            .expect("decoder graph replay");
        g.tokens.clone()
    }

    /// Why the decoder runs eagerly, if a capture was tried and failed.
    pub fn failure(&self) -> Option<&str> {
        match &self.state {
            GraphState::Failed(e) => Some(e),
            _ => None,
        }
    }

    /// Whether the decoder replays a captured graph.
    pub fn is_captured(&self) -> bool {
        matches!(self.state, GraphState::Ready(_))
    }
}

#[cfg(feature = "cuda")]
impl NcmLm {
    /// Warms up and records the greedy decoder pass over fixed input buffers.
    fn capture_decoder(
        &self,
        cond: &Tensor<3>,
        cache: &mut NcmCache,
    ) -> Result<DecoderGraph, String> {
        let device = cond.device();
        let dispatch = device.as_dispatch();
        let cond_buffer = Tensor::<3>::zeros(cond.dims(), &device);
        crate::kernels::assign_into(&cond_buffer, cond.clone(), 0, 0);
        let (cos, sin) = self.rope(cache.tokens, self.k, &device).tables();
        let (cos_buffer, sin_buffer) =
            (Tensor::<4>::zeros(cos.dims(), &device), Tensor::<4>::zeros(sin.dims(), &device));
        crate::kernels::assign_into(&cos_buffer, cos, 0, 0);
        crate::kernels::assign_into(&sin_buffer, sin, 0, 0);
        let rope_q = Rope::from_tables(cos_buffer, sin_buffer);
        let rope_self = self.rope(0, self.k, &device);
        let run = |cache: &mut NcmCache| {
            let mut pick = |_: usize, logits: &Tensor<3>| {
                crate::fused::argmax_last(logits.clone()).reshape([logits.dims()[0], 1])
            };
            let cond = cond_buffer.clone();
            self.decode_with_ropes(cond, cache, &rope_self, &rope_q, false, &mut pick).0
        };
        let err = |what: &str, e: ExecutionError| format!("{what}: {e:?}");
        Dispatch::graph_prepare(dispatch).map_err(|e| err("graph_prepare", e))?;
        // Warm-up runs compile every kernel and fill the memory pool with every buffer the
        // capture will take, so nothing allocates mid-capture.
        for _ in 0..3 {
            let out = run(cache);
            Dispatch::sync(dispatch).map_err(|e| err("warm-up", e))?;
            drop(out);
        }
        Dispatch::graph_start_capture(dispatch).map_err(|e| err("graph_start_capture", e))?;
        let tokens = run(cache);
        let graph =
            Dispatch::graph_stop_capture(dispatch).map_err(|e| err("graph_stop_capture", e))?;
        Ok(DecoderGraph { graph, cond: cond_buffer, rope_q, _rope_self: rope_self, tokens })
    }
}
