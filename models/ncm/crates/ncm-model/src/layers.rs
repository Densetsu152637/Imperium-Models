//! Shared building blocks: RoPE, SwiGLU, attention, transformer blocks and stacks.
//!
//! Both models run through exactly these layers, so any kernel-level speed difference between
//! them comes from the architecture, not from a different attention path.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Bool, Device, Tensor, TensorData};

use crate::cache::{KvCache, build_mask};
use crate::config::is_full_layer;

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

/// Causal GQA self-attention; `window` None means full attention.
#[derive(Module, Debug)]
pub struct Attention {
    qkv: Linear,
    o: Linear,
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
            heads,
            kv_heads,
            head_dim,
            window,
        }
    }

    pub fn forward(&self, x: Tensor<3>, rope: &Rope, cache: Option<&mut KvCache>) -> Tensor<3> {
        let [b, t, _] = x.dims();
        let (h, kvh, hd) = (self.heads, self.kv_heads, self.head_dim);
        let device = x.device();
        let qkv = self.qkv.forward(x);
        // Queries and keys share positions, so one RoPE pass covers both.
        let qk = qkv.clone().narrow(2, 0, (h + kvh) * hd);
        let qk = apply_rope(qk.reshape([b, t, h + kvh, hd]).swap_dims(1, 2), rope);
        let (q, k) = (qk.clone().narrow(1, 0, h), qk.narrow(1, h, kvh));
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
        self.o.forward(out.swap_dims(1, 2).reshape([b, t, h * hd]))
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

#[derive(Module, Debug)]
pub struct Block {
    n1: RmsNorm,
    attn: Attention,
    n2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        ff: usize,
        window: Option<usize>,
        eps: f64,
        device: &Device,
    ) -> Self {
        Self {
            n1: rms_norm(d, eps, device),
            attn: Attention::new(d, heads, kv_heads, head_dim, window, device),
            n2: rms_norm(d, eps, device),
            mlp: Mlp::new(d, ff, device),
        }
    }

    pub fn forward(&self, x: Tensor<3>, rope: &Rope, cache: Option<&mut KvCache>) -> Tensor<3> {
        let x = x.clone() + self.attn.forward(self.n1.forward(x), rope, cache);
        x.clone() + self.mlp.forward(self.n2.forward(x))
    }
}

/// Shape of a layer stack: width, depth, heads, window pattern and RoPE settings.
#[derive(Debug, Clone)]
pub struct StackSpec {
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ff: usize,
    pub window: usize,
    pub full_every: usize,
    pub rope_base: f64,
    pub rope_base_full: f64,
    pub rope_scale: f64,
    pub rope_scale_full: f64,
    pub eps: f64,
}

/// Pre-norm transformer layers with the sliding / every-Nth-full layer pattern.
#[derive(Module, Debug)]
pub struct Stack {
    blocks: Vec<Block>,
    norm: RmsNorm,
    kv_heads: usize,
    head_dim: usize,
    window: usize,
    full_every: usize,
    rope_base: f64,
    rope_base_full: f64,
    rope_scale: f64,
    rope_scale_full: f64,
}

impl Stack {
    pub fn new(spec: &StackSpec, device: &Device) -> Self {
        let blocks = (0..spec.layers)
            .map(|i| {
                let window = (!is_full_layer(i, spec.full_every)).then_some(spec.window);
                Block::new(
                    spec.d,
                    spec.heads,
                    spec.kv_heads,
                    spec.head_dim,
                    spec.ff,
                    window,
                    spec.eps,
                    device,
                )
            })
            .collect();
        Self {
            blocks,
            norm: rms_norm(spec.d, spec.eps, device),
            kv_heads: spec.kv_heads,
            head_dim: spec.head_dim,
            window: spec.window,
            full_every: spec.full_every,
            rope_base: spec.rope_base,
            rope_base_full: spec.rope_base_full,
            rope_scale: spec.rope_scale,
            rope_scale_full: spec.rope_scale_full,
        }
    }

    pub fn layer_window(&self, index: usize) -> Option<usize> {
        (!is_full_layer(index, self.full_every)).then_some(self.window)
    }

    pub fn forward(&self, x: Tensor<3>, mut caches: Option<&mut [KvCache]>) -> Tensor<3> {
        let t = x.dims()[1];
        let device = x.device();
        let start = caches.as_ref().map_or(0, |c| c[0].len());
        let rope_local =
            Rope::new(start, t, self.head_dim, self.rope_base, self.rope_scale, &device);
        let has_full = (0..self.blocks.len()).any(|i| is_full_layer(i, self.full_every));
        let rope_full = has_full.then(|| {
            Rope::new(start, t, self.head_dim, self.rope_base_full, self.rope_scale_full, &device)
        });
        let mut x = x;
        for (i, block) in self.blocks.iter().enumerate() {
            let rope = match (&rope_full, is_full_layer(i, self.full_every)) {
                (Some(full), true) => full,
                _ => &rope_local,
            };
            x = block.forward(x, rope, caches.as_mut().map(|c| &mut c[i]));
        }
        self.norm.forward(x)
    }

    /// One cache per layer for runs of up to `max_positions`.
    pub fn make_caches(&self, max_positions: usize, batch: usize, device: &Device) -> Vec<KvCache> {
        (0..self.blocks.len())
            .map(|i| {
                KvCache::make(
                    batch,
                    self.kv_heads,
                    self.head_dim,
                    max_positions,
                    self.layer_window(i),
                    device,
                )
            })
            .collect()
    }
}
