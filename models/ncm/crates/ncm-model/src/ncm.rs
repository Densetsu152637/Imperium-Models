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
use burn::nn::{Embedding, EmbeddingConfig, Linear, RmsNorm};
use burn::tensor::module::linear as linear_op;
use burn::tensor::{Bool, Device, Int, Tensor};

use crate::cache::{KvCache, build_mask, stack_cache_bytes};
use crate::config::NcmConfig;
use crate::layers::{
    Attention, CrossAttention, Mlp, Rope, Stack, StackSpec, init_normal, linear, rms_norm,
};

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

    /// `x` is `[B, M', k, d]`: every concept's decoder runs in parallel.
    fn forward_train(
        &self,
        x: Tensor<4>,
        rope_self: &Rope,
        rope_q: &Rope,
        keys: Tensor<4>,
        values: Tensor<4>,
        mask: Tensor<2, Bool>,
    ) -> Tensor<4> {
        let [b, m, k, d] = x.dims();
        let h = self.n1.forward(x.clone()).reshape([b * m, k, d]);
        let x = x + self.self_attn.forward(h, rope_self, None).reshape([b, m, k, d]);
        let h = self.n2.forward(x.clone()).reshape([b, m * k, d]);
        let x = x + self.cross.forward(h, rope_q, keys, values, Some(mask)).reshape([b, m, k, d]);
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

/// All caches of one NCM generation run.
pub struct NcmCache {
    pub backbone: Vec<KvCache>,
    pub encoder: Vec<KvCache>,
    /// Per decoder layer: encoder-state keys and values, a ring of `enc_window`.
    pub cross: Vec<KvCache>,
    /// Per decoder layer: only the current concept, reset every concept.
    pub dec_self: Vec<KvCache>,
    /// Tokens consumed by the encoder so far.
    pub tokens: usize,
}

impl NcmCache {
    pub fn nbytes(&self) -> usize {
        [&self.backbone, &self.encoder, &self.cross, &self.dec_self]
            .into_iter()
            .flatten()
            .map(KvCache::nbytes)
            .sum()
    }
}

pub struct NcmParts {
    pub embeddings: usize,
    pub encoder: usize,
    pub backbone: usize,
    pub decoder: usize,
    pub projections: usize,
}

impl NcmParts {
    pub fn total(&self) -> usize {
        self.embeddings + self.encoder + self.backbone + self.decoder + self.projections
    }

    pub fn rows(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("embeddings (tied)", self.embeddings),
            ("local encoder", self.encoder),
            ("global backbone", self.backbone),
            ("local decoder", self.decoder),
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
        let d = cfg.local_dim;
        let encoder = StackSpec {
            d,
            layers: cfg.enc_layers,
            heads: cfg.enc_heads,
            kv_heads: cfg.enc_kv_heads,
            head_dim: cfg.head_dim,
            ff: cfg.enc_ff_dim,
            window: cfg.enc_window,
            full_every: 0,
            rope_base: cfg.rope_base,
            rope_base_full: cfg.rope_base_full,
            rope_scale: cfg.rope_scale,
            rope_scale_full: cfg.rope_scale_full,
            eps: cfg.norm_eps,
        };
        let backbone = StackSpec {
            d: cfg.bb_dim,
            layers: cfg.bb_layers,
            heads: cfg.bb_heads,
            kv_heads: cfg.bb_kv_heads,
            ff: cfg.bb_ff_dim,
            window: cfg.bb_window,
            full_every: cfg.full_every,
            ..encoder.clone()
        };
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

    pub fn parts(&self) -> NcmParts {
        NcmParts {
            embeddings: self.emb.num_params(),
            encoder: self.encoder.num_params(),
            backbone: self.backbone.num_params(),
            decoder: self.dec_blocks.num_params() + self.dec_norm.num_params(),
            projections: self.pool.num_params()
                + self.cond_proj.num_params()
                + self.start_emb.num_params(),
        }
    }

    fn rope(&self, start: usize, len: usize, device: &Device) -> Rope {
        Rope::new(start, len, self.head_dim, self.rope_base, self.rope_scale, device)
    }

    fn logits(&self, x: Tensor<3>) -> Tensor<3> {
        linear_op(self.dec_norm.forward(x), self.emb.weight.val().transpose(), None)
    }

    // ------------------------------------------------------------------ full sequence

    /// Logits `[B, N - k, V]` for tokens k..N-1 of a sequence whose length is a multiple of k.
    pub fn forward(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
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
        let mut x = Tensor::cat(vec![start, written], 2) + cond;
        let rope_self = self.rope(0, k, &device);
        let rope_q = self.rope(k, n - k, &device);
        let rope_k = self.rope(0, n, &device);
        // Concept m' may see encoder states of tokens [k*m' - enc_window, k*m' - 1].
        let ends: Vec<i32> = (k..n).map(|t| ((t / k) * k) as i32 - 1).collect();
        let kpos: Vec<i32> = (0..n as i32).collect();
        let mask = build_mask(&device, &ends, &kpos, Some(self.enc_window));
        for blk in &self.dec_blocks {
            let (keys, values) = blk.cross.make_kv(states.clone(), &rope_k);
            x = blk.forward_train(x, &rope_self, &rope_q, keys, values, mask.clone());
        }
        self.logits(x.reshape([b, (m - 1) * k, d]))
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
        }
    }

    /// Encode whole concepts and run the backbone on them. Returns the conditioning vector
    /// `[B, 1, d]` for the next concept's decoder.
    pub fn append_tokens(&self, tokens: Tensor<2, Int>, cache: &mut NcmCache) -> Tensor<3> {
        let k = self.k;
        let [b, t] = tokens.dims();
        assert!(t % k == 0, "token count must be a multiple of k={k}");
        let device = tokens.device();
        let states = self.encoder.forward(self.emb.forward(tokens), Some(&mut cache.encoder));
        let rope_k = self.rope(cache.tokens, t, &device);
        for (blk, kv_cache) in self.dec_blocks.iter().zip(cache.cross.iter_mut()) {
            let (keys, values) = blk.cross.make_kv(states.clone(), &rope_k);
            kv_cache.write(keys, values);
        }
        let concepts = self.pool.forward(states.reshape([b, t / k, k * self.local_dim]));
        let h = self.backbone.forward(concepts, Some(&mut cache.backbone));
        cache.tokens += t;
        self.cond_proj.forward(h.narrow(1, t / k - 1, 1))
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
        let k = self.k;
        let [b, _, _] = cond.dims();
        let device = cond.device();
        for c in &mut cache.dec_self {
            c.reset();
        }
        let rope_self = self.rope(0, k, &device);
        let rope_q = self.rope(cache.tokens, k, &device);
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
            let next = match &forced {
                Some(forced) => forced.clone().narrow(1, j, 1),
                None => lg.clone().argmax(2).reshape([b, 1]),
            };
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

pub fn ncm_cache_bytes(cfg: &NcmConfig, max_tokens: usize, itemsize: usize, batch: usize) -> usize {
    let concepts = max_tokens.div_ceil(cfg.k);
    let hd = cfg.head_dim;
    let backbone = stack_cache_bytes(
        cfg.bb_layers,
        cfg.full_every,
        cfg.bb_window,
        concepts,
        cfg.bb_kv_heads,
        hd,
        itemsize,
        batch,
    );
    let encoder = stack_cache_bytes(
        cfg.enc_layers,
        0,
        cfg.enc_window,
        max_tokens,
        cfg.enc_kv_heads,
        hd,
        itemsize,
        batch,
    );
    // Cross-attention ring and the per-concept decoder self cache.
    let cross = stack_cache_bytes(
        cfg.dec_layers,
        0,
        cfg.enc_window,
        max_tokens,
        cfg.dec_kv_heads,
        hd,
        itemsize,
        batch,
    );
    let dec_self =
        stack_cache_bytes(cfg.dec_layers, 1, cfg.k, cfg.k, cfg.dec_kv_heads, hd, itemsize, batch);
    backbone + encoder + cross + dec_self
}
