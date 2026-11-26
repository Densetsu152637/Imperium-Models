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

use crate::cache::{KvCache, LayerCache, build_mask, layer_cache_bytes};
use crate::config::{Design, NcmConfig};
use crate::fused::{add_rms_norm, rms_norm as rms_norm_fused};
use crate::layers::{
    Attention, CrossAttention, LayerKind, Mlp, Rope, Stack, StackSpec, fallback_kinds,
    hybrid_kinds, init_normal, linear, norm_after, rms_norm,
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
        let h = rms_norm_fused(&self.n1, x.clone()).reshape([b * m, k, d]);
        let x = x + self.self_attn.forward_masked(h, rope_self, Some(causal)).reshape([b, m, k, d]);
        let h = rms_norm_fused(&self.n2, x.clone()).reshape([b, m * k, d]);
        let x = x + self.cross.forward(h, rope_q, keys, values, mask).reshape([b, m, k, d]);
        x.clone() + self.mlp.forward(rms_norm_fused(&self.n3, x))
    }

    fn step(
        &self,
        x: Tensor<3>,
        rope_self: &Rope,
        rope_q: &Rope,
        self_cache: &mut KvCache,
        cross_cache: &KvCache,
        pending: Option<Tensor<3>>,
    ) -> (Tensor<3>, Option<Tensor<3>>) {
        // As `Block::forward`: `pending`, the previous layer's MLP output, is added together with
        // the first norm, and this layer's MLP output is returned pending, or added already by
        // the fused MLP kernels (`Mlp::residual_step`).
        let (x, h) = norm_after(&self.n1, x, pending);
        let attn = self.self_attn.forward(h, rope_self, Some(self_cache));
        let (x, h) = add_rms_norm(&self.n2, x, attn);
        let (keys, values) = cross_cache.view();
        let crossed = self.cross.forward(h, rope_q, keys, values, None);
        if let Some(x) = self.mlp.residual_step(&self.n3, &x, Some(&crossed)) {
            return (x, None);
        }
        let (x, h) = add_rms_norm(&self.n3, x, crossed);
        (x, Some(self.mlp.forward(h)))
    }
}

impl DecoderBlock {
    /// [`Self::step`] for a batch whose rows are separate streams (slots): row i's
    /// self-attention cache and cross-attention ring are `caches[i]`, its query positions
    /// `rope_q[i]`. Norms and the MLP run batched, the attentions once per row.
    fn step_slots(
        &self,
        x: Tensor<3>,
        rope_self: &Rope,
        rope_q: &[Rope],
        caches: &mut [(&mut KvCache, &KvCache)],
        pending: Option<Tensor<3>>,
    ) -> (Tensor<3>, Option<Tensor<3>>) {
        // Projections once over every slot's row, the attention per slot.
        let (x, h) = norm_after(&self.n1, x, pending);
        let attn = {
            let mut own: Vec<&mut KvCache> = caches.iter_mut().map(|(own, _)| &mut **own).collect();
            self.self_attn.forward_rows(h, rope_self, &mut own)
        };
        let (x, h) = add_rms_norm(&self.n2, x, attn);
        let kvs: Vec<(Tensor<4>, Tensor<4>)> = caches.iter().map(|(_, ring)| ring.view()).collect();
        let crossed = self.cross.forward_rows(h, rope_q, &kvs);
        if let Some(x) = self.mlp.residual_step(&self.n3, &x, Some(&crossed)) {
            return (x, None);
        }
        let (x, h) = add_rms_norm(&self.n3, x, crossed);
        (x, Some(self.mlp.forward(h)))
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
    /// One-concept backbone steps of [`NcmLm::append_tokens`] as a replayed CUDA graph (on CUDA).
    #[cfg(feature = "cuda")]
    pub backbone_graph: crate::layers::GraphedStack,
    /// Unique to this cache's buffers. [`Self::reset`] and [`Self::restore`] write into the
    /// existing buffers on CUDA, so a graph captured against this cache stays valid across
    /// requests; graphs check the epoch before replaying, so another cache makes them recapture.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    epoch: u64,
}

/// What [`NcmCache::restore`] needs to return a cache to an earlier position: the stacks'
/// [`LayerSnapshot`]s, the cross-attention rings and the token count.
pub struct NcmSnapshot {
    backbone: Vec<crate::cache::LayerSnapshot>,
    encoder: Vec<crate::cache::LayerSnapshot>,
    cross: Vec<crate::cache::KvSnapshot>,
    tokens: usize,
}

/// The source of [`NcmCache`] epochs.
static EPOCHS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn next_epoch() -> u64 {
    EPOCHS.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
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
    }

    /// Makes this cache a copy of `src` (a cache of the same model and size): every stack, the
    /// cross-attention rings and the position, in this cache's own buffers on CUDA, so graphs
    /// captured against it stay valid (forking one request's prefilled prompt into another
    /// slot).
    pub fn copy_from(&mut self, src: &NcmCache) {
        for (c, s) in self.backbone.iter_mut().zip(&src.backbone) {
            c.copy_from(s);
        }
        for (c, s) in self.encoder.iter_mut().zip(&src.encoder) {
            c.copy_from(s);
        }
        for (c, s) in self.cross.iter_mut().zip(&src.cross) {
            c.copy_from(s);
        }
        self.dec_self.iter_mut().for_each(KvCache::reset);
        self.tokens = src.tokens;
    }

    /// Tokens the cache holds, and copies of what moving on would overwrite: [`Self::restore`]
    /// returns here (prompt-prefix reuse across requests).
    pub fn snapshot(&self) -> NcmSnapshot {
        NcmSnapshot {
            backbone: self.backbone.iter().map(LayerCache::snapshot).collect(),
            encoder: self.encoder.iter().map(LayerCache::snapshot).collect(),
            cross: self.cross.iter().map(KvCache::snapshot).collect(),
            tokens: self.tokens,
        }
    }

    /// Back to a [`Self::snapshot`] of this cache, in its own buffers on CUDA.
    pub fn restore(&mut self, snapshot: &NcmSnapshot) {
        for (c, s) in self.backbone.iter_mut().zip(&snapshot.backbone) {
            c.restore(s);
        }
        for (c, s) in self.encoder.iter_mut().zip(&snapshot.encoder) {
            c.restore(s);
        }
        for (c, s) in self.cross.iter_mut().zip(&snapshot.cross) {
            c.restore(s);
        }
        self.dec_self.iter_mut().for_each(KvCache::reset);
        self.tokens = snapshot.tokens;
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
        self.project(rms_norm_fused(&self.dec_norm, x))
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

    /// `slots` caches (each for batch 1) for [`Self::append_slots`] and [`Self::decode_slots`]:
    /// as many [`Self::make_cache`], except that the backbone's retrieval caches are rows of
    /// shared buffers on CUDA ([`crate::layers::Stack::make_slot_caches`]), so a batched step
    /// decodes each retrieval layer once for all slots.
    pub fn make_slot_caches(
        &self,
        slots: usize,
        max_tokens: usize,
        device: &Device,
    ) -> Vec<NcmCache> {
        let concepts = max_tokens.div_ceil(self.k);
        let backbones = self.backbone.make_slot_caches(slots, concepts, device);
        backbones
            .into_iter()
            .map(|backbone| NcmCache {
                backbone,
                ..self.make_cache_without_backbone(max_tokens, device)
            })
            .collect()
    }

    /// [`Self::make_cache`] for batch 1 with an empty backbone (filled by the caller).
    fn make_cache_without_backbone(&self, max_tokens: usize, device: &Device) -> NcmCache {
        let mut cache = self.make_cache(0, 1, device);
        cache.encoder = self.encoder.make_caches(max_tokens, 1, device);
        cache.cross = (0..self.dec_blocks.len())
            .map(|_| {
                KvCache::make(
                    1,
                    self.dec_kv_heads,
                    self.head_dim,
                    max_tokens,
                    Some(self.enc_window),
                    device,
                )
            })
            .collect();
        cache.backbone = Vec::new();
        cache
    }

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
            epoch: next_epoch(),
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

    /// Runs the prefill path once on a throwaway cache: `tokens` prompt tokens (rounded down to
    /// whole concepts) in chunks of `chunk`, then one decoded concept, so that the kernels and
    /// buffer sizes of those shapes are compiled and allocated before the first request (a
    /// server calls it at startup, after [`crate::kernels::enable_kernel_cache`] on CUDA). The
    /// model's caches are untouched; the device is synchronised before it returns.
    pub fn warm_up(&self, tokens: usize, chunk: usize, device: &Device) {
        let n = (tokens - tokens % self.k).max(self.k);
        let mut cache = self.make_cache(n + self.k, 1, device);
        let prompt = Tensor::<2, Int>::zeros([1, n], device);
        let cond = self.prefill(prompt, &mut cache, chunk);
        let (written, _) = self.decode_concept(cond, &mut cache, None, false);
        drop(written.into_data());
    }

    /// [`Self::append_tokens`] of whole concepts per batch row, the rows separate streams
    /// (slots): `tokens` `[B, t]` (`t` a multiple of k: one concept per decode step, a prefill
    /// chunk for [`Self::prefill_slots`]), row i appended to `caches[i]` (built for batch 1) at
    /// its own position. Returns each row's next condition `[B, 1, d]`.
    pub fn append_slots(&self, tokens: Tensor<2, Int>, caches: &mut [&mut NcmCache]) -> Tensor<3> {
        let k = self.k;
        let [b, t] = tokens.dims();
        assert!(t > 0 && t % k == 0 && b == caches.len(), "whole concepts for every slot");
        let device = tokens.device();
        let x = self.emb.forward(tokens);
        let states = {
            let mut enc: Vec<&mut [LayerCache]> =
                caches.iter_mut().map(|c| c.encoder.as_mut_slice()).collect();
            self.encoder.forward_slots(x, &mut enc)
        };
        // The cross-attention key/value projections once over every slot's row, RoPE and the
        // ring write per slot.
        let ropes: Vec<Rope> = caches.iter().map(|c| self.rope(c.tokens, t, &device)).collect();
        for (l, blk) in self.dec_blocks.iter().enumerate() {
            let kvs = blk.cross.make_kv_rows(states.clone(), &ropes);
            for (cache, (keys, values)) in caches.iter_mut().zip(kvs) {
                cache.cross[l].write(keys, values);
            }
        }
        let n = t / k;
        let concepts = self.pool.forward(states.reshape([b, n, k * self.local_dim]));
        let h = {
            let mut bb: Vec<&mut [LayerCache]> =
                caches.iter_mut().map(|c| c.backbone.as_mut_slice()).collect();
            self.backbone.forward_slots(concepts, &mut bb)
        };
        for cache in caches.iter_mut() {
            cache.tokens += t;
        }
        self.cond_proj.forward(if n == 1 { h } else { h.narrow(1, n - 1, 1) })
    }

    /// [`Self::prefill`] of one prompt piece per batch row, the rows separate streams (slots)
    /// each at its own position: `tokens` `[B, n]` (`n` a multiple of k), row i into
    /// `caches[i]`, in chunks of `chunk` tokens (at most [`ENCODER_CHUNK`]) through
    /// [`Self::append_slots`], so the projections and MLPs of every slot's chunk run as one
    /// batch. Returns each row's next condition `[B, 1, d]`.
    pub fn prefill_slots(
        &self,
        tokens: Tensor<2, Int>,
        caches: &mut [&mut NcmCache],
        chunk: usize,
    ) -> Tensor<3> {
        let k = self.k;
        let chunk = chunk.min(ENCODER_CHUNK);
        let chunk = (chunk - chunk % k).max(k);
        let n = tokens.dims()[1];
        let mut cond = None;
        let mut start = 0;
        while start < n {
            let len = chunk.min(n - start);
            cond = Some(self.append_slots(tokens.clone().narrow(1, start, len), caches));
            start += len;
        }
        cond.expect("prompt pieces must not be empty")
    }

    /// [`Self::decode_concept_with`] for a batch of separate streams: `cond` `[B, 1, d]`, row i
    /// decoded against `caches[i]`; `pick(j, logits)` chooses token j of every row from logits
    /// `[B, 1, V]` and returns `[B, 1]`. Returns the tokens `[B, k]`.
    pub fn decode_slots(
        &self,
        cond: Tensor<3>,
        caches: &mut [&mut NcmCache],
        pick: &mut dyn FnMut(usize, &Tensor<3>) -> Tensor<2, Int>,
    ) -> Tensor<2, Int> {
        let k = self.k;
        let device = cond.device();
        let rope_self = self.rope(0, k, &device);
        let rope_q: Vec<Rope> = caches.iter().map(|c| self.rope(c.tokens, k, &device)).collect();
        self.decode_slots_with_ropes(cond, caches, &rope_self, &rope_q, pick)
    }

    /// [`Self::decode_slots`] with the RoPE tables given: the concept's own positions
    /// (`rope_self`, 0..k) and each row's absolute positions (`rope_q[i]`). No host uploads,
    /// so it can run inside a CUDA graph capture.
    pub(crate) fn decode_slots_with_ropes(
        &self,
        cond: Tensor<3>,
        caches: &mut [&mut NcmCache],
        rope_self: &Rope,
        rope_q: &[Rope],
        pick: &mut dyn FnMut(usize, &Tensor<3>) -> Tensor<2, Int>,
    ) -> Tensor<2, Int> {
        let (k, d) = (self.k, self.local_dim);
        let [b, _, _] = cond.dims();
        for cache in caches.iter_mut() {
            cache.dec_self.iter_mut().for_each(KvCache::reset);
        }
        let mut tokens = Vec::with_capacity(k);
        let mut inp = self.start_emb.val().reshape([1, 1, d]).expand([b, 1, d]);
        for j in 0..k {
            let mut x = inp + cond.clone();
            let rs = rope_self.rows(j, 1);
            let rq: Vec<Rope> = rope_q.iter().map(|r| r.rows(j, 1)).collect();
            let mut pending = None;
            for (l, blk) in self.dec_blocks.iter().enumerate() {
                let mut layer: Vec<(&mut KvCache, &KvCache)> = caches
                    .iter_mut()
                    .map(|c| {
                        let NcmCache { dec_self, cross, .. } = &mut **c;
                        (&mut dec_self[l], &cross[l])
                    })
                    .collect();
                (x, pending) = blk.step_slots(x, &rs, &rq, &mut layer, pending);
            }
            let logits = self.project(norm_after(&self.dec_norm, x, pending).1);
            let next = pick(j, &logits);
            inp = self.emb.forward(next.clone());
            tokens.push(next);
        }
        Tensor::cat(tokens, 1)
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
            let mut pending = None;
            for ((blk, self_cache), cross_cache) in layers {
                (x, pending) = blk.step(x, &rs, &rq, self_cache, cross_cache, pending);
            }
            let lg = self.project(norm_after(&self.dec_norm, x, pending).1);
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
    graph: crate::graph::Captured,
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
        // A graph kept across requests serves only once the cross-attention ring is full
        // again: before, the eager pass attends over the filled slots alone.
        let serving = ring_full && crate::fused::graphs_active(&device);
        let (true, GraphState::Ready(g)) = (serving, &self.state) else {
            return model.decode_concept(cond, cache, None, false).0;
        };
        let (cos, sin) = model.rope(cache.tokens, model.k, &device).tables();
        let (q_cos, q_sin) = g.rope_q.tables();
        crate::kernels::assign_into(&g.cond, cond, 0, 0);
        crate::kernels::assign_into(&q_cos, cos, 0, 0);
        crate::kernels::assign_into(&q_sin, sin, 0, 0);
        // Safety: every buffer the graph touched is held by `g` (its inputs and output), by the
        // model (weights) or by `cache` (the decoder caches), all of which outlive the replay,
        // (the replay checks it runs on the capturing thread).
        unsafe { crate::graph::replay(&device, &g.graph, "decoder") };
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
            let cond = cond_buffer.clone();
            self.decode_with_ropes(cond, cache, &rope_self, &rope_q, false, &mut greedy).0
        };
        // The decoder's self caches restart every concept: nothing to undo.
        let (graph, tokens) = crate::graph::record(&device, cache, 3, run, |_| {})?;
        Ok(DecoderGraph { graph, cond: cond_buffer, rope_q, _rope_self: rope_self, tokens })
    }
}

/// Token `j` of a concept drawn by the device sampler from its logits `[B, 1, V]`.
#[cfg(feature = "cuda")]
fn sample_pick(
    logits: &Tensor<3>,
    params: &crate::sample::SampleParams,
    j: usize,
) -> Tensor<2, Int> {
    let [b, _, v] = logits.dims();
    crate::sample::sample(logits.clone().reshape([b, v]), params, j as u32)
}

/// The next concept from `cond`, each token drawn by the device sampler (eagerly).
#[cfg(feature = "cuda")]
fn sampled_concept(
    model: &NcmLm,
    cond: Tensor<3>,
    cache: &mut NcmCache,
    params: &crate::sample::SampleParams,
) -> Tensor<2, Int> {
    let mut pick = |j: usize, logits: &Tensor<3>| sample_pick(logits, params, j);
    model.decode_concept_with(cond, cache, false, &mut pick).0
}

/// The greedy pick of [`NcmLm::decode_with_ropes`] on the device.
#[cfg(feature = "cuda")]
fn greedy(_: usize, logits: &Tensor<3>) -> Tensor<2, Int> {
    crate::fused::argmax_last(logits.clone()).reshape([logits.dims()[0], 1])
}

/// A whole NCM concept step as one captured CUDA graph per cache, replayed for every concept:
/// [`NcmLm::append_tokens`] of the concept just written (encoder over its k tokens, the
/// cross-attention keys, concept pooling, the backbone step) and then the greedy decoder pass of
/// the next concept, with no host work or sync in between. Captured once the cross-attention
/// ring is full; the encoder's sliding-window rings and the backbone's caches are written and
/// masked through device cursors ([`crate::graph`]). Before each replay the written tokens, the
/// RoPE tables of the new positions and the cache positions are copied in; after it the host
/// positions advance. Both designs.
///
/// On other devices, with the fused kernels or graphs off, before the ring fills, or if the
/// capture fails, [`GraphedStep::step`] runs `append_tokens` and then [`GraphedDecoder`]. A graph
/// serves the cache it was captured against, also after [`NcmCache::reset`] or
/// [`NcmCache::restore`] (which write into its buffers on CUDA) once the ring is full again;
/// stepping another cache captures anew.
#[cfg(feature = "cuda")]
#[derive(Default)]
pub struct GraphedStep {
    state: StepState,
    decoder: GraphedDecoder,
    /// Sampled decoding ([`Self::sampled`]); None: greedy.
    sampler: Option<StepSampler>,
}

/// The device sampler of a sampled [`GraphedStep`] and the request's settings, whose counter
/// advances by `k` draws per concept.
#[cfg(feature = "cuda")]
struct StepSampler {
    params: crate::sample::SampleParams,
    settings: crate::sample::SampleSettings,
}

/// How a step graph picks the next concept's tokens.
#[cfg(feature = "cuda")]
#[derive(Clone)]
enum Pick {
    /// No decoder pass ([`GraphedAppend`]).
    None,
    Greedy,
    /// [`crate::sample::sample`] with these device settings, draw `counter + j` for token j.
    Sample(Box<crate::sample::SampleParams>),
}

/// [`NcmLm::append_tokens`] of one concept as a captured CUDA graph, like [`GraphedStep`]
/// without the decoder pass: for callers that write the next concept another way (the
/// self-speculative draft and verify passes, [`crate::Speculator`]).
#[cfg(feature = "cuda")]
#[derive(Default)]
pub struct GraphedAppend {
    state: StepState,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
enum StepState {
    #[default]
    Untried,
    Failed(String),
    Ready(Box<StepGraph>),
}

/// What a step graph leaves in its output buffers.
#[cfg(feature = "cuda")]
enum StepOut {
    /// The next concept's condition `[B, 1, d]`.
    Cond(Tensor<3>),
    /// The next concept's greedy tokens `[B, k]`.
    Tokens(Tensor<2, Int>),
}

#[cfg(feature = "cuda")]
struct StepGraph {
    graph: crate::graph::Captured,
    out: StepOut,
    inputs: StepInputs,
}

/// The fixed inputs of a step graph.
#[cfg(feature = "cuda")]
struct StepInputs {
    /// The written concept `[B, k]` (i32), copied in before each replay.
    tokens: Tensor<2, Int>,
    cursors: crate::graph::CursorSet,
    staging: crate::graph::Staging,
    encoder: crate::layers::StackStep,
    backbone: crate::layers::StackStep,
    /// Each decoder layer's cross-attention ring, in `cursors`.
    cross: Vec<usize>,
    /// RoPE tables of the written concept's positions (the cross-attention keys), and where
    /// they are staged.
    rope_k: (Rope, (usize, usize)),
    /// The decoder pass: RoPE tables of the next concept's positions, and where staged.
    rope_q: Option<(Rope, (usize, usize))>,
    /// RoPE tables of a concept's own positions (0..k) for the decoder pass.
    rope_self: Rope,
    /// The epoch of the cache the graph was captured against ([`NcmCache`]).
    epoch: u64,
}

#[cfg(feature = "cuda")]
impl GraphedStep {
    /// Appends `tokens` `[B, k]` (the concept just written) and greedily writes the next
    /// concept, like `model.decode_concept(model.append_tokens(tokens, cache), cache, None,
    /// false)`. The returned tensor is the graph's output buffer: read it (or pass it back in)
    /// before the next call, which overwrites it.
    pub fn step(
        &mut self,
        model: &NcmLm,
        tokens: Tensor<2, Int>,
        cache: &mut NcmCache,
    ) -> Tensor<2, Int> {
        let pick = self.pick();
        let out = match self.state.run(model, &tokens, cache, &pick) {
            Some(StepOut::Tokens(t)) => t,
            Some(StepOut::Cond(_)) => unreachable!("a decoding step graph writes tokens"),
            None => {
                let cond = model.append_tokens(tokens, cache);
                match &pick {
                    Pick::Sample(params) => sampled_concept(model, cond, cache, params),
                    _ => self.decoder.decode(model, cond, cache),
                }
            }
        };
        self.advance(model.k);
        out
    }

    /// A step that samples each token on the device instead of taking the argmax
    /// ([`crate::sample`]: temperature, top-k and top-p, a seeded counter-based random stream),
    /// inside the same graph. `settings.counter` is the request's first draw.
    pub fn sampled(device: &Device, settings: crate::sample::SampleSettings) -> Self {
        let params = crate::sample::SampleParams::new(device);
        Self { sampler: Some(StepSampler { params, settings }), ..Default::default() }
    }

    /// The first concept of a reply from the prompt's condition `cond`, run eagerly (the step
    /// graph serves only concepts that follow a written one). Its first tokens are `forced`
    /// (the prompt's trailing partial concept, fewer than k tokens), teacher-forced as in
    /// training; the rest are picked like every later token (argmax, or the device sampler,
    /// whose counter advances by k either way).
    pub fn first(
        &mut self,
        model: &NcmLm,
        cond: Tensor<3>,
        cache: &mut NcmCache,
        forced: &[i32],
    ) -> Tensor<2, Int> {
        assert!(forced.len() < model.k, "at most k - 1 tokens are pending");
        let device = cond.device();
        let [b, _, _] = cond.dims();
        let forced: Vec<Tensor<2, Int>> = forced
            .iter()
            .map(|&t| {
                let data = burn::tensor::TensorData::new(vec![t; b], [b, 1]);
                Tensor::<2, Int>::from_data(data, &device).cast(burn::tensor::IntDType::I32)
            })
            .collect();
        let pick = self.pick();
        let mut choose = |j: usize, logits: &Tensor<3>| match (forced.get(j), &pick) {
            (Some(token), _) => token.clone(),
            (None, Pick::Sample(params)) => sample_pick(logits, params, j),
            (None, _) => greedy(j, logits),
        };
        let out = model.decode_concept_with(cond, cache, false, &mut choose).0;
        self.advance(model.k);
        out
    }

    /// New settings for a sampled step (a new request): the captured graph keeps serving, as
    /// it reads the settings from device buffers. Panics on a greedy step.
    pub fn set_sampling(&mut self, settings: crate::sample::SampleSettings) {
        self.sampler.as_mut().expect("a sampled step").settings = settings;
    }

    /// The pick of the next step, its device settings refreshed.
    fn pick(&self) -> Pick {
        match &self.sampler {
            None => Pick::Greedy,
            Some(s) => {
                s.params.set(s.settings);
                Pick::Sample(Box::new(s.params.clone()))
            }
        }
    }

    /// Counts the draws of a written concept.
    fn advance(&mut self, k: usize) {
        if let Some(s) = &mut self.sampler {
            s.settings.counter = s.settings.counter.wrapping_add(k as u32);
        }
    }

    /// Captures the graph now (if it can serve the next [`Self::step`] with `tokens` of this
    /// shape and none is ready), so that the capture's warm-up runs stay out of a timed loop.
    /// Leaves the cache as it was; returns whether a graph is ready.
    pub fn prepare(
        &mut self,
        model: &NcmLm,
        tokens: &Tensor<2, Int>,
        cache: &mut NcmCache,
    ) -> bool {
        let pick = self.pick();
        self.state.prepare(model, tokens, cache, &pick);
        self.is_captured()
    }

    /// Why the step runs eagerly, if a capture was tried and failed.
    pub fn failure(&self) -> Option<&str> {
        self.state.failure()
    }

    /// Whether the step replays a captured graph.
    pub fn is_captured(&self) -> bool {
        matches!(self.state, StepState::Ready(_))
    }
}

#[cfg(feature = "cuda")]
impl GraphedAppend {
    /// `model.append_tokens(tokens, cache)` for one concept `[B, k]`, replayed from a graph
    /// when it can be. The returned condition is the graph's output buffer: use it before the
    /// next call, which overwrites it.
    pub fn append(
        &mut self,
        model: &NcmLm,
        tokens: Tensor<2, Int>,
        cache: &mut NcmCache,
    ) -> Tensor<3> {
        match self.state.run(model, &tokens, cache, &Pick::None) {
            Some(StepOut::Cond(c)) => c,
            Some(StepOut::Tokens(_)) => unreachable!("an append graph writes the condition"),
            None => model.append_tokens(tokens, cache),
        }
    }

    /// As [`GraphedStep::prepare`], for [`Self::append`].
    pub fn prepare(
        &mut self,
        model: &NcmLm,
        tokens: &Tensor<2, Int>,
        cache: &mut NcmCache,
    ) -> bool {
        self.state.prepare(model, tokens, cache, &Pick::None);
        self.is_captured()
    }

    /// Why the append runs eagerly, if a capture was tried and failed.
    pub fn failure(&self) -> Option<&str> {
        self.state.failure()
    }

    /// Whether the append replays a captured graph.
    pub fn is_captured(&self) -> bool {
        matches!(self.state, StepState::Ready(_))
    }
}

#[cfg(feature = "cuda")]
impl StepState {
    /// Replays the graph (capturing it first when it can be); None: the caller runs the step
    /// eagerly.
    fn run(
        &mut self,
        model: &NcmLm,
        tokens: &Tensor<2, Int>,
        cache: &mut NcmCache,
        pick: &Pick,
    ) -> Option<StepOut> {
        let device = tokens.device();
        if !crate::fused::graphs_active(&device) || tokens.dims()[1] != model.k {
            return None;
        }
        self.prepare(model, tokens, cache, pick);
        let StepState::Ready(g) = self else { return None };
        // A graph kept across requests serves only once the cross-attention ring is full again.
        if !cache.cross.iter().all(|c| c.len() >= c.capacity()) {
            return None;
        }
        g.replay(model, tokens.clone(), cache);
        Some(match &g.out {
            StepOut::Cond(c) => StepOut::Cond(c.clone()),
            StepOut::Tokens(t) => StepOut::Tokens(t.clone()),
        })
    }

    /// Captures the graph if it can serve this step now and none is ready. A graph captured
    /// against another cache, or before the cache was reset, is dropped first.
    fn prepare(
        &mut self,
        model: &NcmLm,
        tokens: &Tensor<2, Int>,
        cache: &mut NcmCache,
        pick: &Pick,
    ) {
        if matches!(self, StepState::Ready(g) if g.inputs.epoch != cache.epoch) {
            *self = StepState::Untried;
        }
        let ring_full = cache.cross.iter().all(|c| c.window().is_some() && c.len() >= c.capacity());
        let eligible = crate::fused::graphs_active(&tokens.device()) && tokens.dims()[1] == model.k;
        if matches!(self, StepState::Untried) && ring_full && eligible {
            *self = match model.capture_step(tokens, cache, pick) {
                Ok(graph) => StepState::Ready(Box::new(graph)),
                Err(e) => StepState::Failed(e),
            };
        }
    }

    fn failure(&self) -> Option<&str> {
        match self {
            StepState::Failed(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "cuda")]
impl StepGraph {
    /// Copies in the written concept and every position-dependent input, replays, and moves
    /// the host positions on.
    fn replay(&mut self, model: &NcmLm, tokens: Tensor<2, Int>, cache: &mut NcmCache) {
        let device = tokens.device();
        let k = model.k;
        let inputs = &mut self.inputs;
        let tokens = tokens.cast(burn::tensor::IntDType::I32);
        crate::kernels::assign_int_into(&inputs.tokens, tokens, 0, 0);
        inputs.refresh(model, cache);
        // Safety: every buffer the graph touched is held by `self` (inputs, outputs, RoPE
        // tables, cursors), by the model (weights) or by `cache`, all alive across the replay,
        // (the replay checks it runs on the capturing thread).
        unsafe { crate::graph::replay(&device, &self.graph, "concept step") };
        inputs.encoder.advance(&mut cache.encoder);
        for c in &mut cache.cross {
            c.advance(k);
        }
        inputs.backbone.advance(&mut cache.backbone);
        cache.tokens += k;
    }
}

#[cfg(feature = "cuda")]
impl StepInputs {
    /// Stages and uploads the RoPE tables and cache positions of the cache's current position.
    fn refresh(&mut self, model: &NcmLm, cache: &NcmCache) {
        let (k, base, scale) = (model.k, model.rope_base, model.rope_scale);
        let (cursors, staging) = (&mut self.cursors, &mut self.staging);
        self.encoder.refresh(&model.encoder, &cache.encoder, cursors, staging);
        self.backbone.refresh(&model.backbone, &cache.backbone, cursors, staging);
        for (&i, c) in self.cross.iter().zip(&cache.cross) {
            cursors.sync(i, c);
        }
        cursors.refresh(staging);
        let (rope_k, at) = &self.rope_k;
        let off = crate::layers::Yarn::OFF;
        rope_k.stage(staging, *at, cache.tokens, (base, scale, off));
        if let Some((rope_q, at)) = &self.rope_q {
            rope_q.stage(staging, *at, cache.tokens + k, (base, scale, off));
        }
        staging.upload(&self.tokens.device());
    }
}

#[cfg(feature = "cuda")]
impl NcmLm {
    /// Warms up and records one concept step (with the decoder pass when `decode`) over fixed
    /// input buffers, leaving the caches as they were.
    fn capture_step(
        &self,
        tokens: &Tensor<2, Int>,
        cache: &mut NcmCache,
        pick: &Pick,
    ) -> Result<StepGraph, String> {
        use crate::graph::{CursorSet, restore_kv, save_kv, write_kv};
        use crate::layers::StackStep;
        let device = tokens.device();
        let (k, d) = (self.k, self.local_dim);
        let [b, _] = tokens.dims();
        let input = Tensor::<2, Int>::zeros([b, k], &device).cast(burn::tensor::IntDType::I32);
        let (mut cursors, mut staging) = (CursorSet::default(), Default::default());
        let encoder = StackStep::new(&self.encoder, &cache.encoder, k, &mut cursors, &mut staging)?;
        let cross = cache
            .cross
            .iter()
            .map(|c| cursors.add(c, k, &mut staging))
            .collect::<Result<Vec<_>, _>>()?;
        let backbone =
            StackStep::new(&self.backbone, &cache.backbone, 1, &mut cursors, &mut staging)?;
        let mut rope = |start: usize| {
            let rope = self.rope(start, k, &device).fixed();
            let at = rope.add_to(&mut staging);
            (rope, at)
        };
        let rope_k = rope(cache.tokens);
        let rope_q = (!matches!(pick, Pick::None)).then(|| rope(cache.tokens + k));
        let mut inputs = StepInputs {
            tokens: input,
            cursors,
            staging,
            encoder,
            backbone,
            cross,
            rope_k,
            rope_q,
            rope_self: self.rope(0, k, &device),
            epoch: cache.epoch,
        };
        let tokens = tokens.clone().cast(burn::tensor::IntDType::I32);
        crate::kernels::assign_int_into(&inputs.tokens, tokens, 0, 0);
        inputs.refresh(self, cache);
        let saved = (
            inputs.encoder.save(&cache.encoder),
            cache.cross.iter().map(|c| save_kv(c, k)).collect::<Vec<_>>(),
            inputs.backbone.save(&cache.backbone),
        );
        let g = &inputs;
        let run = |cache: &mut NcmCache| {
            g.staging.scatter();
            let steps = g.cursors.steps();
            let x = self.emb.forward(g.tokens.clone());
            let states = g.encoder.run(&self.encoder, x, &mut cache.encoder, &steps);
            let layers = self.dec_blocks.iter().zip(cache.cross.iter_mut()).zip(&g.cross);
            for ((blk, ring), &i) in layers {
                let (keys, values) = blk.cross.make_kv(states.clone(), &g.rope_k.0);
                write_kv(ring, keys, values, &steps[i]);
            }
            let concept = self.pool.forward(states.reshape([b, 1, k * d]));
            let h = g.backbone.run(&self.backbone, concept, &mut cache.backbone, &steps);
            let cond = self.cond_proj.forward(h);
            match &g.rope_q {
                None => StepOut::Cond(cond),
                Some((q, _)) => {
                    let mut sampled = |j: usize, logits: &Tensor<3>| match pick {
                        Pick::Sample(params) => sample_pick(logits, params, j),
                        _ => greedy(j, logits),
                    };
                    let pass =
                        self.decode_with_ropes(cond, cache, &g.rope_self, q, false, &mut sampled);
                    StepOut::Tokens(pass.0)
                }
            }
        };
        let undo = |cache: &mut NcmCache| {
            StackStep::restore(&mut cache.encoder, &saved.0);
            for (ring, s) in cache.cross.iter_mut().zip(&saved.1) {
                restore_kv(ring, s);
            }
            StackStep::restore(&mut cache.backbone, &saved.2);
        };
        let (graph, out) = crate::graph::record(&device, cache, 2, run, undo)?;
        Ok(StepGraph { graph, out, inputs })
    }
}

/// NCM concept steps of several separate streams (slots) batched together, as one captured
/// CUDA graph per set of slots, captured lazily and kept (up to
/// [`crate::tok::SLOT_GRAPHS`]): [`NcmLm::append_slots`] of each slot's written concept
/// (encoder, the cross-attention keys, pooling and the backbone step, each slot's layers
/// through its own [`crate::layers::StackStep`] and cursors, the retrieval layers of pooled
/// slot caches in one set of launches) and then [`NcmLm::decode_slots`] of the next concept,
/// greedy or with the device sampler, one settings set per row. Before each replay every
/// slot's cursors, positions and RoPE tables are staged and uploaded once; after it the host
/// positions advance. A set waits until every slot's cross-attention ring is full.
#[cfg(feature = "cuda")]
#[derive(Default)]
pub struct GraphedNcmSlots {
    graphs: Vec<NcmSlotsEntry>,
}

/// A slot set and its graph (or why its capture failed).
#[cfg(feature = "cuda")]
type NcmSlotsEntry = (Vec<usize>, Result<Box<NcmSlotsGraph>, String>);

#[cfg(feature = "cuda")]
struct NcmSlotsGraph {
    graph: crate::graph::Captured,
    /// The rows' written concepts `[B, k]` (i32).
    tokens: Tensor<2, Int>,
    staging: crate::graph::Staging,
    slots: Vec<SlotInputs>,
    params: crate::sample::SampleParams,
    /// Held for its buffers, which the graph reads (the concept's own RoPE positions).
    #[allow(dead_code)]
    rope_self: Rope,
    /// The rows' next concepts `[B, k]`, and the conditions `[B, 1, d]` they were written from.
    output: (Tensor<2, Int>, Tensor<3>),
    /// The epochs of the caches the graph was captured against ([`NcmCache`]).
    epochs: Vec<u64>,
}

/// One slot's fixed inputs in a [`GraphedNcmSlots`] graph.
#[cfg(feature = "cuda")]
struct SlotInputs {
    cursors: crate::graph::CursorSet,
    encoder: crate::layers::StackStep,
    backbone: crate::layers::StackStep,
    /// Each decoder layer's cross-attention ring, in `cursors`.
    cross: Vec<usize>,
    /// RoPE tables of the written concept's positions, and of the next concept's, and where
    /// each is staged.
    rope_k: (Rope, (usize, usize)),
    rope_q: (Rope, (usize, usize)),
}

#[cfg(feature = "cuda")]
impl SlotInputs {
    /// Stages the slot's cursors, positions and RoPE tables at its cache's position.
    fn refresh(&mut self, model: &NcmLm, cache: &NcmCache, staging: &mut crate::graph::Staging) {
        let (k, base, scale) = (model.k, model.rope_base, model.rope_scale);
        let cursors = &mut self.cursors;
        self.encoder.refresh(&model.encoder, &cache.encoder, cursors, staging);
        self.backbone.refresh(&model.backbone, &cache.backbone, cursors, staging);
        for (&i, c) in self.cross.iter().zip(&cache.cross) {
            cursors.sync(i, c);
        }
        cursors.refresh(staging);
        let off = crate::layers::Yarn::OFF;
        self.rope_k.0.stage(staging, self.rope_k.1, cache.tokens, (base, scale, off));
        self.rope_q.0.stage(staging, self.rope_q.1, cache.tokens + k, (base, scale, off));
    }

    /// Moves the slot's host positions on after a replay.
    fn advance(&self, k: usize, cache: &mut NcmCache) {
        self.encoder.advance(&mut cache.encoder);
        for c in &mut cache.cross {
            c.advance(k);
        }
        self.backbone.advance(&mut cache.backbone);
        cache.tokens += k;
    }
}

#[cfg(feature = "cuda")]
impl GraphedNcmSlots {
    /// The next concept `[B, k]` of each slot of `key` (in order; `caches` theirs) after
    /// appending `tokens` `[B, k]` (each row the concept that slot just wrote), each row picked
    /// per `settings` (greedy rows take [`crate::sample::SampleSettings::GREEDY`]; counters as
    /// the requests stand, advancing by k per concept), from the slot set's graph (captured now
    /// if new and every slot's ring is full). None when no graph can serve: the caller steps
    /// eagerly ([`NcmLm::append_slots`], [`NcmLm::decode_slots`]). Also returns each row's
    /// condition `[B, 1, d]` after the append (what `append_slots` would return). Both are the
    /// graph's output buffers: read or copy them before the next call for the same slots.
    pub fn step(
        &mut self,
        model: &NcmLm,
        key: &[usize],
        tokens: Tensor<2, Int>,
        caches: &mut [&mut NcmCache],
        settings: &[crate::sample::SampleSettings],
    ) -> Option<(Tensor<2, Int>, Tensor<3>)> {
        let device = tokens.device();
        let rings_full = caches
            .iter()
            .all(|c| c.cross.iter().all(|r| r.window().is_some() && r.len() >= r.capacity()));
        if !crate::fused::graphs_active(&device) || tokens.dims()[1] != model.k || !rings_full {
            return None;
        }
        // A graph captured against other caches (or before a cache's buffers changed) is stale.
        let epochs: Vec<u64> = caches.iter().map(|c| c.epoch).collect();
        self.graphs.retain(|(k, g)| k != key || g.as_ref().map_or(true, |g| g.epochs == epochs));
        if !self.graphs.iter().any(|(k, _)| k == key) {
            if self.graphs.len() >= crate::tok::SLOT_GRAPHS {
                drop(self.graphs.remove(0));
            }
            let graph = model.capture_slots_step(&tokens, caches, settings).map(Box::new);
            self.graphs.push((key.to_vec(), graph));
        }
        let (_, Ok(g)) = self.graphs.iter_mut().find(|(k, _)| k == key)? else {
            return None;
        };
        let tokens = tokens.cast(burn::tensor::IntDType::I32);
        crate::kernels::assign_int_into(&g.tokens, tokens, 0, 0);
        g.params.set_rows(settings);
        for (slot, cache) in g.slots.iter_mut().zip(caches.iter()) {
            slot.refresh(model, cache, &mut g.staging);
        }
        g.staging.upload(&device);
        // Safety: the graph's buffers are held by `g`, the model's weights and the slots'
        // caches, all alive across the replay (which checks its thread).
        unsafe { crate::graph::replay(&device, &g.graph, "batched concept step") };
        for (slot, cache) in g.slots.iter().zip(caches.iter_mut()) {
            slot.advance(model.k, cache);
        }
        Some(g.output.clone())
    }

    /// Why a slot set's step runs eagerly, if its capture failed.
    pub fn failure(&self, key: &[usize]) -> Option<&str> {
        self.graphs
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, g)| g.as_ref().err())
            .map(String::as_str)
    }
}

#[cfg(feature = "cuda")]
impl NcmLm {
    /// Warms up and records one batched concept step of the slots `caches` over fixed input
    /// buffers, leaving the caches as they were.
    fn capture_slots_step(
        &self,
        tokens: &Tensor<2, Int>,
        caches: &mut [&mut NcmCache],
        settings: &[crate::sample::SampleSettings],
    ) -> Result<NcmSlotsGraph, String> {
        use crate::graph::{CursorSet, restore_kv, save_kv, write_kv};
        use crate::layers::StackStep;
        let device = tokens.device();
        let (k, d) = (self.k, self.local_dim);
        let b = caches.len();
        let mut staging = crate::graph::Staging::default();
        let mut slots = Vec::with_capacity(b);
        for cache in caches.iter() {
            let mut cursors = CursorSet::default();
            let encoder =
                StackStep::new(&self.encoder, &cache.encoder, k, &mut cursors, &mut staging)?;
            let cross = cache
                .cross
                .iter()
                .map(|c| cursors.add(c, k, &mut staging))
                .collect::<Result<Vec<_>, _>>()?;
            let backbone =
                StackStep::new(&self.backbone, &cache.backbone, 1, &mut cursors, &mut staging)?;
            let mut rope = |start: usize| {
                let rope = self.rope(start, k, &device).fixed();
                let at = rope.add_to(&mut staging);
                (rope, at)
            };
            let rope_k = rope(cache.tokens);
            let rope_q = rope(cache.tokens + k);
            slots.push(SlotInputs { cursors, encoder, backbone, cross, rope_k, rope_q });
        }
        let input = Tensor::<2, Int>::zeros([b, k], &device).cast(burn::tensor::IntDType::I32);
        crate::kernels::assign_int_into(
            &input,
            tokens.clone().cast(burn::tensor::IntDType::I32),
            0,
            0,
        );
        let params = crate::sample::SampleParams::rows(&device, b);
        params.set_rows(settings);
        for (slot, cache) in slots.iter_mut().zip(caches.iter()) {
            slot.refresh(self, cache, &mut staging);
        }
        staging.upload(&device);
        let rope_self = self.rope(0, k, &device).fixed();
        let saved: Vec<_> = slots
            .iter()
            .zip(caches.iter())
            .map(|(slot, c)| {
                (
                    slot.encoder.save(&c.encoder),
                    c.cross.iter().map(|r| save_kv(r, k)).collect::<Vec<_>>(),
                    slot.backbone.save(&c.backbone),
                )
            })
            .collect();
        let (g_slots, g_staging, g_input, g_params, g_rope_self) =
            (&slots, &staging, &input, &params, &rope_self);
        let run = |caches: &mut [&mut NcmCache]| {
            g_staging.scatter();
            let kv: Vec<_> = g_slots.iter().map(|s| s.cursors.steps()).collect();
            let x = self.emb.forward(g_input.clone());
            let states = {
                let steps: Vec<&StackStep> = g_slots.iter().map(|s| &s.encoder).collect();
                let mut enc: Vec<&mut [LayerCache]> =
                    caches.iter_mut().map(|c| c.encoder.as_mut_slice()).collect();
                StackStep::run_slots(&self.encoder, &steps, x, &mut enc, &kv)
            };
            // The cross-attention keys and values: the projection once over every row, RoPE
            // and the ring write per slot.
            let ropes_k: Vec<Rope> = g_slots.iter().map(|s| s.rope_k.0.clone()).collect();
            for (l, blk) in self.dec_blocks.iter().enumerate() {
                let kvs = blk.cross.make_kv_rows(states.clone(), &ropes_k);
                for (i, (cache, (keys, values))) in caches.iter_mut().zip(kvs).enumerate() {
                    write_kv(&mut cache.cross[l], keys, values, &kv[i][g_slots[i].cross[l]]);
                }
            }
            let concept = self.pool.forward(states.reshape([b, 1, k * d]));
            let h = {
                let steps: Vec<&StackStep> = g_slots.iter().map(|s| &s.backbone).collect();
                let mut bb: Vec<&mut [LayerCache]> =
                    caches.iter_mut().map(|c| c.backbone.as_mut_slice()).collect();
                StackStep::run_slots(&self.backbone, &steps, concept, &mut bb, &kv)
            };
            let cond = self.cond_proj.forward(h);
            let ropes_q: Vec<Rope> = g_slots.iter().map(|s| s.rope_q.0.clone()).collect();
            let mut pick = |j: usize, logits: &Tensor<3>| sample_pick(logits, g_params, j);
            let next = self.decode_slots_with_ropes(
                cond.clone(),
                caches,
                g_rope_self,
                &ropes_q,
                &mut pick,
            );
            (next, cond)
        };
        let undo = |caches: &mut [&mut NcmCache]| {
            for (cache, s) in caches.iter_mut().zip(&saved) {
                StackStep::restore(&mut cache.encoder, &s.0);
                for (ring, r) in cache.cross.iter_mut().zip(&s.1) {
                    restore_kv(ring, r);
                }
                StackStep::restore(&mut cache.backbone, &s.2);
            }
        };
        let (graph, output) = crate::graph::record(&device, caches, 2, run, undo)?;
        let epochs = caches.iter().map(|c| c.epoch).collect();
        Ok(NcmSlotsGraph {
            graph,
            tokens: input,
            staging,
            slots,
            params,
            rope_self,
            output,
            epochs,
        })
    }
}
