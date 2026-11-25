//! Shared building blocks: RoPE, SwiGLU, attention, transformer blocks and stacks.
//!
//! Both models run through exactly these layers, so any kernel-level speed difference between
//! them comes from the architecture, not from a different attention path. A stack's layers
//! each pick a sequence mixer ([`LayerKind`]): grouped-query attention for the fallback design,
//! gated DeltaNet and sparse retrieval for v2.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::activation::{sigmoid, silu, softmax};
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Bool, Device, Tensor, TensorData};

use crate::cache::{KvCache, LayerCache, build_mask, layer_cache_bytes};
use crate::config::{V2Config, is_full_layer};
use crate::delta::{DeltaSpec, GatedDeltaNet};
use crate::retrieval::{RetrievalSpec, SparseRetrieval};

/// Weight initialiser used by every linear layer and embedding: N(0, 0.02).
pub fn init_normal() -> Initializer {
    Initializer::Normal { mean: 0.0, std: 0.02 }
}

pub fn linear(d_in: usize, d_out: usize, device: &Device) -> Linear {
    LinearConfig::new(d_in, d_out).with_bias(false).with_initializer(init_normal()).init(device)
}

pub fn rms_norm(d: usize, eps: f64, device: &Device) -> RmsNorm {
    RmsNormConfig::new(d).with_epsilon(eps).init(device)
}

static FUSED_ATTENTION: AtomicBool = AtomicBool::new(false);

/// Route every attention call through Burn's fused `attention` op instead of the explicit
/// matmul-softmax-matmul sequence. Process-wide, so both models always share one path.
pub fn set_fused_attention(on: bool) {
    FUSED_ATTENTION.store(on, Ordering::Relaxed);
}

/// The signed permutation `R` with `x @ R = [-x2, x1]` for `x = [x1, x2]` (the two halves of the
/// last axis), `[1, 1, D, D]`. One tiny matmul replaces two slices and a concatenation.
fn rotation(head_dim: usize, device: &Device) -> Tensor<4> {
    type Key = (String, usize, String);
    static CACHE: OnceLock<Mutex<HashMap<Key, Tensor<4>>>> = OnceLock::new();
    let dtype = device.settings().float_dtype;
    let key = (format!("{device:?}"), head_dim, format!("{dtype:?}"));
    let mut cache = CACHE.get_or_init(Default::default).lock().expect("rotation cache");
    cache
        .entry(key)
        .or_insert_with(|| {
            let half = head_dim / 2;
            let mut r = vec![0f32; head_dim * head_dim];
            for j in 0..half {
                r[(j + half) * head_dim + j] = -1.0;
                r[j * head_dim + j + half] = 1.0;
            }
            Tensor::from_data(TensorData::new(r, [1, 1, head_dim, head_dim]), device)
        })
        .clone()
}

/// RoPE tables `[1, 1, T, head_dim]`, ready to broadcast over `[B, H, T, D]`: `cos` and `sin`
/// repeated across both halves, so that `rope(x) = x * cos + (x @ R) * sin`.
#[derive(Clone)]
pub struct Rope {
    cos: Tensor<4>,
    sin: Tensor<4>,
    rot: Tensor<4>,
}

impl Rope {
    /// Tables for absolute positions `start..start + len`; `scale` divides the positions
    /// (a linear scaling hook).
    pub fn new(
        start: usize,
        len: usize,
        head_dim: usize,
        base: f64,
        scale: f64,
        device: &Device,
    ) -> Self {
        let half = head_dim / 2;
        let inv: Vec<f64> =
            (0..half).map(|i| 1.0 / base.powf((2 * i) as f64 / head_dim as f64)).collect();
        let mut cos = Vec::with_capacity(len * head_dim);
        let mut sin = Vec::with_capacity(len * head_dim);
        for pos in start..start + len {
            for _ in 0..2 {
                for f in &inv {
                    let angle = pos as f64 / scale * f;
                    cos.push(angle.cos() as f32);
                    sin.push(angle.sin() as f32);
                }
            }
        }
        let shape = [1, 1, len, head_dim];
        Self {
            cos: Tensor::from_data(TensorData::new(cos, shape), device),
            sin: Tensor::from_data(TensorData::new(sin, shape), device),
            rot: rotation(head_dim, device),
        }
    }

    /// The `cos` and `sin` tables `[1, 1, T, head_dim]`.
    #[cfg(feature = "cuda")]
    pub(crate) fn tables(&self) -> (Tensor<4>, Tensor<4>) {
        (self.cos.clone(), self.sin.clone())
    }

    /// Tables over given `cos` and `sin` buffers `[1, 1, T, head_dim]`.
    #[cfg(feature = "cuda")]
    pub(crate) fn from_tables(cos: Tensor<4>, sin: Tensor<4>) -> Self {
        let rot = rotation(cos.dims()[3], &cos.device());
        Self { cos, sin, rot }
    }

    /// Rows `start..start + len` of these tables.
    pub fn rows(&self, start: usize, len: usize) -> Self {
        Self {
            cos: self.cos.clone().narrow(2, start, len),
            sin: self.sin.clone().narrow(2, start, len),
            rot: self.rot.clone(),
        }
    }
}

/// Rotate `x` `[B, H, T, D]` by the position tables.
pub fn apply_rope(x: Tensor<4>, rope: &Rope) -> Tensor<4> {
    x.clone().matmul(rope.rot.clone()) * rope.sin.clone() + x * rope.cos.clone()
}

/// Scaled dot-product attention with grouped-query heads; the one attention path both models
/// use. `q` is `[B, H, T, D]`, `k` and `v` are `[B, KVH, S, D]`, and `mask` is `[T, S]` with
/// `true` marking keys to hide.
///
/// The query heads of one KV group are folded into the query-length axis
/// (`[B, KVH, G * T, D]`, head `h = kvh * G + g`), so K and V are read once per step rather than
/// expanded per query head.
pub fn attend(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    mask: Option<Tensor<2, Bool>>,
) -> Tensor<4> {
    let [b, h, t, d] = q.dims();
    let [_, kvh, s, _] = k.dims();
    let groups = h / kvh;
    let q = q.reshape([b, kvh, groups * t, d]);
    let mask = mask.map(|m| {
        m.repeat_dim(0, groups).reshape([1, 1, groups * t, s]).expand([b, kvh, groups * t, s])
    });
    let out = if FUSED_ATTENTION.load(Ordering::Relaxed) {
        attention(q, k, v, mask, None, AttentionModuleOptions::default())
    } else {
        let scale = 1.0 / (d as f64).sqrt();
        let mut scores = q.matmul(k.swap_dims(2, 3)) * scale;
        if let Some(mask) = mask {
            scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        }
        softmax(scores, 3).matmul(v)
    };
    out.reshape([b, h, t, d])
}

/// SwiGLU feed-forward.
#[derive(Module, Debug)]
pub struct Mlp {
    gate_up: Linear,
    down: Linear,
}

impl Mlp {
    pub fn new(d: usize, ff: usize, device: &Device) -> Self {
        Self { gate_up: linear(d, 2 * ff, device), down: linear(ff, d, device) }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let mut parts = self.gate_up.forward(x).chunk(2, D - 1).into_iter();
        let (gate, up) = (parts.next().expect("gate"), parts.next().expect("up"));
        self.down.forward(silu(gate) * up)
    }
}

/// Causal GQA self-attention; `window` None means full attention. The v2 design adds QK-norm
/// (RMSNorm of each query and key head before RoPE) and a sigmoid output gate.
#[derive(Module, Debug)]
pub struct Attention {
    qkv: Linear,
    o: Linear,
    gate: Option<Linear>,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: Option<usize>,
}

impl Attention {
    pub fn new(
        d: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        window: Option<usize>,
        device: &Device,
    ) -> Self {
        Self {
            qkv: linear(d, (heads + 2 * kv_heads) * head_dim, device),
            o: linear(heads * head_dim, d, device),
            gate: None,
            q_norm: None,
            k_norm: None,
            heads,
            kv_heads,
            head_dim,
            window,
        }
    }

    /// The same layer with an output gate and QK-norm (v2).
    pub fn gated(mut self, d: usize, eps: f64, device: &Device) -> Self {
        self.gate = Some(linear(d, self.heads * self.head_dim, device));
        self.q_norm = Some(rms_norm(self.head_dim, eps, device));
        self.k_norm = Some(rms_norm(self.head_dim, eps, device));
        self
    }

    pub fn window(&self) -> Option<usize> {
        self.window
    }

    pub fn make_cache(&self, max_positions: usize, batch: usize, device: &Device) -> KvCache {
        KvCache::make(batch, self.kv_heads, self.head_dim, max_positions, self.window, device)
    }

    pub fn forward(&self, x: Tensor<3>, rope: &Rope, cache: Option<&mut KvCache>) -> Tensor<3> {
        let [b, t, _] = x.dims();
        let (h, kvh, hd) = (self.heads, self.kv_heads, self.head_dim);
        let device = x.device();
        let gate = self.gate.as_ref().map(|g| sigmoid(g.forward(x.clone())));
        let qkv = self.qkv.forward(x);
        let (q, k) = match (&self.q_norm, &self.k_norm) {
            (Some(q_norm), Some(k_norm)) => {
                let q = qkv.clone().narrow(2, 0, h * hd).reshape([b, t, h, hd]);
                let k = qkv.clone().narrow(2, h * hd, kvh * hd).reshape([b, t, kvh, hd]);
                let q = apply_rope(q_norm.forward(q).swap_dims(1, 2), rope);
                (q, apply_rope(k_norm.forward(k).swap_dims(1, 2), rope))
            }
            _ => {
                // Queries and keys share positions, so one RoPE pass covers both.
                let qk = qkv.clone().narrow(2, 0, (h + kvh) * hd);
                let qk = apply_rope(qk.reshape([b, t, h + kvh, hd]).swap_dims(1, 2), rope);
                (qk.clone().narrow(1, 0, h), qk.narrow(1, h, kvh))
            }
        };
        let v = qkv.narrow(2, (h + kvh) * hd, kvh * hd).reshape([b, t, kvh, hd]).swap_dims(1, 2);
        let out = match cache {
            None => {
                let pos: Vec<i32> = (0..t as i32).collect();
                let mask = (t > 1).then(|| build_mask(&device, &pos, &pos, self.window));
                attend(q, k, v, mask)
            }
            Some(cache) => {
                let (keys, values, mask) = cache.append(k, v);
                attend(q, keys, values, mask)
            }
        };
        let out = out.swap_dims(1, 2).reshape([b, t, h * hd]);
        let out = match gate {
            Some(gate) => out * gate,
            None => out,
        };
        self.o.forward(out)
    }
}

/// Queries from the local decoder; keys and values from local-encoder states.
#[derive(Module, Debug)]
pub struct CrossAttention {
    q: Linear,
    kv: Linear,
    o: Linear,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl CrossAttention {
    pub fn new(
        d: usize,
        d_kv: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Self {
        Self {
            q: linear(d, heads * head_dim, device),
            kv: linear(d_kv, 2 * kv_heads * head_dim, device),
            o: linear(heads * head_dim, d, device),
            heads,
            kv_heads,
            head_dim,
        }
    }

    /// Keys (with RoPE) and values `[B, KVH, T, D]` for encoder states `[B, T, d_kv]`.
    pub fn make_kv(&self, states: Tensor<3>, rope: &Rope) -> (Tensor<4>, Tensor<4>) {
        let [b, t, _] = states.dims();
        let (kvh, hd) = (self.kv_heads, self.head_dim);
        let kv = self.kv.forward(states);
        let k = kv.clone().narrow(2, 0, kvh * hd);
        let v = kv.narrow(2, kvh * hd, kvh * hd);
        let k = apply_rope(k.reshape([b, t, kvh, hd]).swap_dims(1, 2), rope);
        (k, v.reshape([b, t, kvh, hd]).swap_dims(1, 2))
    }

    pub fn forward(
        &self,
        x: Tensor<3>,
        rope_q: &Rope,
        keys: Tensor<4>,
        values: Tensor<4>,
        mask: Option<Tensor<2, Bool>>,
    ) -> Tensor<3> {
        let [b, t, _] = x.dims();
        let (h, hd) = (self.heads, self.head_dim);
        let q = self.q.forward(x).reshape([b, t, h, hd]).swap_dims(1, 2);
        let out = attend(apply_rope(q, rope_q), keys, values, mask);
        self.o.forward(out.swap_dims(1, 2).reshape([b, t, h * hd]))
    }
}

/// The sequence mixer of one layer. A stack holds a few dozen at most, so the size gap between
/// variants costs nothing worth boxing for.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum Mixer {
    Attention(Attention),
    DeltaNet(GatedDeltaNet),
    Retrieval(SparseRetrieval),
}

/// RoPE tables of one stack forward pass, by layer type.
struct Ropes {
    local: Option<Rope>,
    full: Option<Rope>,
    retrieval: Option<Rope>,
}

#[derive(Module, Debug)]
pub struct Block {
    n1: RmsNorm,
    attn: Mixer,
    n2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    pub fn new(spec: &StackSpec, kind: LayerKind, device: &Device) -> Self {
        let mixer = match kind {
            LayerKind::Attention { window } => {
                let attn = Attention::new(
                    spec.d,
                    spec.heads,
                    spec.kv_heads,
                    spec.head_dim,
                    window,
                    device,
                );
                Mixer::Attention(if spec.gated {
                    attn.gated(spec.d, spec.eps, device)
                } else {
                    attn
                })
            }
            LayerKind::DeltaNet => Mixer::DeltaNet(GatedDeltaNet::new(&spec.delta(), device)),
            LayerKind::Retrieval => {
                Mixer::Retrieval(SparseRetrieval::new(&spec.retrieval(), device))
            }
        };
        Self {
            n1: rms_norm(spec.d, spec.eps, device),
            attn: mixer,
            n2: rms_norm(spec.d, spec.eps, device),
            mlp: Mlp::new(spec.d, spec.ff, device),
        }
    }

    fn mix(
        &self,
        x: Tensor<3>,
        start: usize,
        ropes: &Ropes,
        cache: Option<&mut LayerCache>,
    ) -> Tensor<3> {
        fn table(rope: &Option<Rope>) -> &Rope {
            rope.as_ref().expect("the stack built this table")
        }
        match &self.attn {
            Mixer::Attention(attn) => {
                let rope = if attn.window().is_some() { &ropes.local } else { &ropes.full };
                attn.forward(x, table(rope), cache.map(LayerCache::kv))
            }
            Mixer::DeltaNet(delta) => delta.forward(x, cache.map(LayerCache::delta)),
            Mixer::Retrieval(ret) => {
                ret.forward(x, start, table(&ropes.retrieval), cache.map(LayerCache::latent))
            }
        }
    }

    fn forward(
        &self,
        x: Tensor<3>,
        start: usize,
        ropes: &Ropes,
        cache: Option<&mut LayerCache>,
    ) -> Tensor<3> {
        let x = x.clone() + self.mix(self.n1.forward(x), start, ropes, cache);
        x.clone() + self.mlp.forward(self.n2.forward(x))
    }

    fn make_cache(&self, max_positions: usize, batch: usize, device: &Device) -> LayerCache {
        match &self.attn {
            Mixer::Attention(attn) => LayerCache::Kv(attn.make_cache(max_positions, batch, device)),
            Mixer::DeltaNet(delta) => LayerCache::Delta(delta.make_cache(batch, device)),
            Mixer::Retrieval(ret) => {
                LayerCache::Latent(ret.make_cache(max_positions, batch, device))
            }
        }
    }
}

/// The sequence mixer of one layer of a stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Grouped-query attention over a sliding `window`, or the whole context when None.
    Attention {
        window: Option<usize>,
    },
    DeltaNet,
    Retrieval,
}

/// The fallback layer pattern: sliding-window layers with every `full_every`-th layer full.
pub fn fallback_kinds(layers: usize, window: usize, full_every: usize) -> Vec<LayerKind> {
    (0..layers)
        .map(|i| LayerKind::Attention { window: (!is_full_layer(i, full_every)).then_some(window) })
        .collect()
}

/// The v2 backbone pattern: gated DeltaNet and sparse retrieval alternating, DeltaNet first.
pub fn hybrid_kinds(layers: usize) -> Vec<LayerKind> {
    (0..layers)
        .map(|i| if i % 2 == 0 { LayerKind::DeltaNet } else { LayerKind::Retrieval })
        .collect()
}

/// Shape of a layer stack: width, the mixer of every layer, and their settings.
#[derive(Debug, Clone)]
pub struct StackSpec {
    pub d: usize,
    pub ff: usize,
    pub kinds: Vec<LayerKind>,
    /// Attention layers.
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Output gate and QK-norm on attention layers (v2).
    pub gated: bool,
    /// Sliding attention layers.
    pub rope_base: f64,
    pub rope_scale: f64,
    /// Full attention layers and the retrieval layers' decoupled key.
    pub rope_base_full: f64,
    pub rope_scale_full: f64,
    pub eps: f64,
    /// DeltaNet and retrieval layers.
    pub v2: V2Config,
}

impl StackSpec {
    pub fn delta(&self) -> DeltaSpec {
        DeltaSpec {
            d: self.d,
            head_dim: self.v2.dn_head_dim,
            conv: self.v2.dn_conv,
            chunk: self.v2.dn_chunk,
            eps: self.eps,
        }
    }

    pub fn retrieval(&self) -> RetrievalSpec {
        let v2 = &self.v2;
        RetrievalSpec {
            d: self.d,
            heads: v2.ret_heads,
            nope_dim: v2.ret_nope_dim,
            rope_dim: v2.ret_rope_dim,
            v_dim: v2.ret_v_dim,
            latent: v2.ret_latent,
            index_heads: v2.idx_heads,
            index_dim: v2.idx_dim,
            block: v2.ret_block,
            topk: v2.ret_topk,
            local: v2.ret_local,
            dense: v2.ret_dense,
            eps: self.eps,
        }
    }

    /// Bytes of the stack's caches for a run of `max_positions`; indexer keys at
    /// `index_itemsize`, everything else at `itemsize`.
    pub fn cache_bytes(
        &self,
        max_positions: usize,
        itemsize: usize,
        index_itemsize: usize,
        batch: usize,
    ) -> usize {
        self.kinds
            .iter()
            .map(|kind| match *kind {
                LayerKind::Attention { window } => layer_cache_bytes(
                    self.kv_heads,
                    self.head_dim,
                    max_positions,
                    window,
                    itemsize,
                    batch,
                ),
                LayerKind::DeltaNet => self.delta().cache_bytes(itemsize, batch),
                LayerKind::Retrieval => {
                    self.retrieval().cache_bytes(max_positions, itemsize, index_itemsize, batch)
                }
            })
            .sum()
    }
}

/// Pre-norm residual layers, each a sequence mixer and a SwiGLU MLP.
#[derive(Module, Debug)]
pub struct Stack {
    blocks: Vec<Block>,
    norm: RmsNorm,
    head_dim: usize,
    retrieval_rope_dim: usize,
    rope_base: f64,
    rope_base_full: f64,
    rope_scale: f64,
    rope_scale_full: f64,
}

impl Stack {
    pub fn new(spec: &StackSpec, device: &Device) -> Self {
        Self {
            blocks: spec.kinds.iter().map(|&kind| Block::new(spec, kind, device)).collect(),
            norm: rms_norm(spec.d, spec.eps, device),
            head_dim: spec.head_dim,
            retrieval_rope_dim: spec.v2.ret_rope_dim,
            rope_base: spec.rope_base,
            rope_base_full: spec.rope_base_full,
            rope_scale: spec.rope_scale,
            rope_scale_full: spec.rope_scale_full,
        }
    }

    fn ropes(&self, start: usize, t: usize, device: &Device) -> Ropes {
        let (mut local, mut full, mut retrieval) = (false, false, false);
        for block in &self.blocks {
            match &block.attn {
                Mixer::Attention(a) if a.window().is_some() => local = true,
                Mixer::Attention(_) => full = true,
                Mixer::Retrieval(_) => retrieval = true,
                Mixer::DeltaNet(_) => {}
            }
        }
        let table = |on: bool, dim: usize, base: f64, scale: f64| {
            on.then(|| Rope::new(start, t, dim, base, scale, device))
        };
        Ropes {
            local: table(local, self.head_dim, self.rope_base, self.rope_scale),
            full: table(full, self.head_dim, self.rope_base_full, self.rope_scale_full),
            retrieval: table(
                retrieval,
                self.retrieval_rope_dim,
                self.rope_base_full,
                self.rope_scale_full,
            ),
        }
    }

    pub fn forward(&self, x: Tensor<3>, caches: Option<&mut [LayerCache]>) -> Tensor<3> {
        let t = x.dims()[1];
        let device = x.device();
        let start = caches.as_ref().map_or(0, |c| c[0].len());
        let ropes = self.ropes(start, t, &device);
        self.forward_with(x, start, &ropes, caches)
    }

    /// [`Self::forward`] with the RoPE tables given.
    fn forward_with(
        &self,
        x: Tensor<3>,
        start: usize,
        ropes: &Ropes,
        mut caches: Option<&mut [LayerCache]>,
    ) -> Tensor<3> {
        let mut x = x;
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(x, start, ropes, caches.as_mut().map(|c| &mut c[i]));
        }
        self.norm.forward(x)
    }

    /// One cache per layer for runs of up to `max_positions`.
    pub fn make_caches(
        &self,
        max_positions: usize,
        batch: usize,
        device: &Device,
    ) -> Vec<LayerCache> {
        self.blocks.iter().map(|b| b.make_cache(max_positions, batch, device)).collect()
    }
}

/// One-token steps of a v2 stack (DeltaNet and retrieval layers only) as one captured CUDA
/// graph per cache, replayed for every step.
///
/// With the fused kernels every launch of such a step is the same at every position: the
/// DeltaNet kernel updates its state in place, and the retrieval kernels take the position from a
/// device buffer and size their grids for the largest step. Before each replay the input, the
/// RoPE tables of the position and the caches' position buffers are copied in; after it the
/// caches' host positions advance. Stacks with attention layers, runs of more than one token,
/// other devices or a failed capture run [`Stack::forward`] eagerly.
#[cfg(feature = "cuda")]
#[derive(Default)]
pub struct GraphedStack {
    state: StackGraphState,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
enum StackGraphState {
    #[default]
    Untried,
    Failed(String),
    Ready(Box<StackGraph>),
}

#[cfg(feature = "cuda")]
struct StackGraph {
    graph: burn::backend::BackendGraph<burn::backend::Dispatch>,
    input: Tensor<3>,
    rope: Rope,
    output: Tensor<3>,
}

#[cfg(feature = "cuda")]
impl GraphedStack {
    /// `stack.forward(x, Some(caches))`, replayed from a graph when it can be. The returned
    /// tensor is the graph's output buffer: use it before the next call, which overwrites it.
    /// One `GraphedStack` serves one set of caches.
    pub fn forward(&mut self, stack: &Stack, x: Tensor<3>, caches: &mut [LayerCache]) -> Tensor<3> {
        use burn::backend::{Backend, Dispatch};
        let device = x.device();
        let eligible =
            x.dims()[1] == 1 && stack.graphable() && crate::fused::graphs_active(&device);
        if !eligible {
            return stack.forward(x, Some(caches));
        }
        if matches!(self.state, StackGraphState::Untried) {
            self.state = match stack.capture(&x, caches) {
                Ok(graph) => StackGraphState::Ready(Box::new(graph)),
                Err(e) => StackGraphState::Failed(e),
            };
        }
        let StackGraphState::Ready(g) = &self.state else {
            return stack.forward(x, Some(caches));
        };
        let start = caches[0].len();
        crate::kernels::assign_into(&g.input, x, 0, 0);
        let (cos, sin) = stack.retrieval_rope(start, &device).tables();
        let (g_cos, g_sin) = g.rope.tables();
        crate::kernels::assign_into(&g_cos, cos, 0, 0);
        crate::kernels::assign_into(&g_sin, sin, 0, 0);
        for cache in caches.iter() {
            if let LayerCache::Latent(c) = cache {
                c.refresh_position();
            }
        }
        // Safety: the graph's buffers are held by `g`, the stack's weights and the caches, all
        // alive across the replay, and everything runs on this thread's stream.
        unsafe { Dispatch::graph_replay(device.as_dispatch(), &g.graph) }
            .expect("stack graph replay");
        for cache in caches.iter_mut() {
            cache.advance(1);
        }
        g.output.clone()
    }

    /// Why the stack runs eagerly, if a capture was tried and failed.
    pub fn failure(&self) -> Option<&str> {
        match &self.state {
            StackGraphState::Failed(e) => Some(e),
            _ => None,
        }
    }

    /// Whether the stack replays a captured graph.
    pub fn is_captured(&self) -> bool {
        matches!(self.state, StackGraphState::Ready(_))
    }
}

#[cfg(feature = "cuda")]
impl Stack {
    /// Whether every layer has a position-independent fused step (DeltaNet or retrieval).
    fn graphable(&self) -> bool {
        self.blocks.iter().all(|b| matches!(b.attn, Mixer::DeltaNet(_) | Mixer::Retrieval(_)))
    }

    /// The retrieval layers' RoPE tables for the one position `start`.
    fn retrieval_rope(&self, start: usize, device: &Device) -> Rope {
        let (base, scale) = (self.rope_base_full, self.rope_scale_full);
        Rope::new(start, 1, self.retrieval_rope_dim, base, scale, device)
    }

    /// Warms up and records one step over fixed input buffers, leaving the caches as they were.
    fn capture(&self, x: &Tensor<3>, caches: &mut [LayerCache]) -> Result<StackGraph, String> {
        use burn::backend::{Backend, Dispatch};
        let device = x.device();
        let dispatch = device.as_dispatch();
        let start = caches[0].len();
        let input = Tensor::<3>::zeros(x.dims(), &device);
        crate::kernels::assign_into(&input, x.clone(), 0, 0);
        let (cos, sin) = self.retrieval_rope(start, &device).tables();
        let buffer = |t: Tensor<4>| {
            let b = Tensor::<4>::zeros(t.dims(), &device);
            crate::kernels::assign_into(&b, t, 0, 0);
            b
        };
        let rope = Rope::from_tables(buffer(cos), buffer(sin));
        let ropes = Ropes { local: None, full: None, retrieval: Some(rope.clone()) };
        // Warm-up runs are real steps: their writes are undone (DeltaNet states restored,
        // retrieval positions rewound so the real step overwrites their entries).
        let snapshot = |caches: &[LayerCache]| -> Vec<Option<(Tensor<4>, Tensor<3>, usize)>> {
            caches
                .iter()
                .map(|c| match c {
                    LayerCache::Delta(d) => Some(d.snapshot()),
                    _ => None,
                })
                .collect()
        };
        let restore = |caches: &mut [LayerCache],
                       saved: &[Option<(Tensor<4>, Tensor<3>, usize)>]| {
            for (cache, saved) in caches.iter_mut().zip(saved) {
                match (cache, saved) {
                    (LayerCache::Delta(d), Some(s)) => d.restore(s.clone()),
                    (LayerCache::Latent(l), _) => l.rewind(start),
                    _ => {}
                }
            }
        };
        let saved = snapshot(caches);
        let err = |what: &str, e: burn::tensor::ExecutionError| format!("{what}: {e:?}");
        Dispatch::graph_prepare(dispatch).map_err(|e| err("graph_prepare", e))?;
        for _ in 0..2 {
            let out = self.forward_with(input.clone(), start, &ropes, Some(caches));
            Dispatch::sync(dispatch).map_err(|e| err("warm-up", e))?;
            drop(out);
            restore(caches, &saved);
        }
        Dispatch::sync(dispatch).map_err(|e| err("restore", e))?;
        Dispatch::graph_start_capture(dispatch).map_err(|e| err("graph_start_capture", e))?;
        crate::fused::set_capturing(true);
        let output = self.forward_with(input.clone(), start, &ropes, Some(caches));
        crate::fused::set_capturing(false);
        let graph = Dispatch::graph_stop_capture(dispatch);
        // The recording ran nothing on the device; only the host positions moved.
        restore(caches, &saved);
        let graph = graph.map_err(|e| err("graph_stop_capture", e))?;
        Ok(StackGraph { graph, input, rope, output })
    }
}
