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
use burn::nn::{Initializer, Linear as DenseLinear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::activation::{sigmoid, silu, softmax};
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Bool, Device, Tensor, TensorData};

use crate::cache::{KvCache, LayerCache, build_mask, layer_cache_bytes};
use crate::config::{V2Config, is_full_layer};
use crate::delta::{DeltaSpec, GatedDeltaNet};
use crate::graph::KvStep;
use crate::quant::QLinear as Linear;
use crate::retrieval::{RetrievalSpec, SparseRetrieval};

/// Weight initialiser used by every linear layer and embedding: N(0, 0.02).
pub fn init_normal() -> Initializer {
    Initializer::Normal { mean: 0.0, std: 0.02 }
}

/// A bias-free linear layer: dense, or quantised inside [`crate::quant::build`].
pub fn linear(d_in: usize, d_out: usize, device: &Device) -> Linear {
    Linear::new(d_in, d_out, device)
}

/// A bias-free linear layer that is never quantised (its weight is read as a matrix).
pub fn dense_linear(d_in: usize, d_out: usize, device: &Device) -> DenseLinear {
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
///
/// Cached per thread as well: cubecl gives every thread its own stream, and a CUDA graph
/// capture that reads a buffer another thread's stream wrote is invalidated.
fn rotation(head_dim: usize, device: &Device) -> Tensor<4> {
    type Key = (String, usize, String, std::thread::ThreadId);
    static CACHE: OnceLock<Mutex<HashMap<Key, Tensor<4>>>> = OnceLock::new();
    let dtype = device.settings().float_dtype;
    let thread = std::thread::current().id();
    let key = (format!("{device:?}"), head_dim, format!("{dtype:?}"), thread);
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

/// YaRN (Peng et al., 2023) context extension of a RoPE table. A frequency whose wavelength fits
/// at least `BETA` times into the `original` context keeps its rate, one whose wavelength is
/// longer than that context (fewer than `ALPHA` turns) is slowed by `factor` (position
/// interpolation), with a linear ramp in between; both tables are scaled by the attention
/// temperature `0.1 ln(factor) + 1`, so a dot product of two rotated vectors grows by its square.
/// A `factor` of 1 is plain RoPE.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Yarn {
    pub factor: f64,
    /// Positions seen in training before the extension.
    pub original: usize,
}

impl Yarn {
    /// Wavelengths fitting fewer turns into the original context than this are interpolated.
    const ALPHA: f64 = 1.0;
    /// Wavelengths fitting more turns than this keep their rate.
    const BETA: f64 = 32.0;

    /// Plain RoPE.
    pub const OFF: Self = Self { factor: 1.0, original: 1 };

    /// The retrieval key's settings of a v2 config.
    pub fn of(v2: &V2Config) -> Self {
        Self { factor: v2.ret_yarn_factor, original: v2.ret_yarn_original }
    }

    fn on(&self) -> bool {
        self.factor > 1.0
    }

    /// The rate of frequency `theta` (radians per position) after the extension.
    fn rate(&self, theta: f64) -> f64 {
        if !self.on() {
            return theta;
        }
        let turns = self.original as f64 * theta / std::f64::consts::TAU;
        let keep = ((turns - Self::ALPHA) / (Self::BETA - Self::ALPHA)).clamp(0.0, 1.0);
        keep * theta + (1.0 - keep) * theta / self.factor
    }

    /// The factor on both tables.
    fn temperature(&self) -> f64 {
        if self.on() { 0.1 * self.factor.ln() + 1.0 } else { 1.0 }
    }
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
        Self::with_yarn(start, len, head_dim, base, scale, Yarn::OFF, device)
    }

    /// [`Self::new`] with the frequencies and temperature of `yarn`.
    pub fn with_yarn(
        start: usize,
        len: usize,
        head_dim: usize,
        base: f64,
        scale: f64,
        yarn: Yarn,
        device: &Device,
    ) -> Self {
        let mut cos = vec![0.0; len * head_dim];
        let mut sin = vec![0.0; len * head_dim];
        rope_values(start, head_dim, base, scale, yarn, &mut cos, &mut sin);
        let shape = [1, 1, len, head_dim];
        Self {
            cos: Tensor::from_data(TensorData::new(cos, shape), device),
            sin: Tensor::from_data(TensorData::new(sin, shape), device),
            rot: rotation(head_dim, device),
        }
    }

    /// Tables for the absolute positions `positions` (any order, repeats allowed), plain RoPE:
    /// row i rotates by `positions[i]`.
    pub fn at(positions: &[usize], head_dim: usize, base: f64, device: &Device) -> Self {
        let n = positions.len();
        let mut cos = vec![0.0; n * head_dim];
        let mut sin = vec![0.0; n * head_dim];
        let rows = cos.chunks_mut(head_dim).zip(sin.chunks_mut(head_dim));
        for (&pos, (c, s)) in positions.iter().zip(rows) {
            rope_values(pos, head_dim, base, 1.0, Yarn::OFF, c, s);
        }
        let shape = [1, 1, n, head_dim];
        Self {
            cos: Tensor::from_data(TensorData::new(cos, shape), device),
            sin: Tensor::from_data(TensorData::new(sin, shape), device),
            rot: rotation(head_dim, device),
        }
    }

    /// The tables `(cos, sin)` as host values, `len * head_dim` each (for tests).
    pub fn to_vecs(&self) -> (Vec<f32>, Vec<f32>) {
        let read = |t: &Tensor<4>| {
            t.clone().into_data().convert::<f32>().try_to_vec::<f32>().expect("float table")
        };
        (read(&self.cos), read(&self.sin))
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

    /// A copy of these tables in buffers of its own, for a captured graph to read.
    #[cfg(feature = "cuda")]
    pub(crate) fn fixed(&self) -> Self {
        let copy = |t: &Tensor<4>| {
            let b = Tensor::<4>::zeros(t.dims(), &t.device());
            crate::kernels::assign_into(&b, t.clone(), 0, 0);
            b
        };
        Self { cos: copy(&self.cos), sin: copy(&self.sin), rot: self.rot.clone() }
    }

    /// Stages these tables (fixed buffers, [`Self::fixed`]) in `staging`: (cos, sin) offsets,
    /// for [`Self::stage`].
    #[cfg(feature = "cuda")]
    pub(crate) fn add_to(&self, staging: &mut crate::graph::Staging) -> (usize, usize) {
        (staging.add_floats(self.cos.clone()), staging.add_floats(self.sin.clone()))
    }

    /// Stages the tables of positions `start..` (as many rows as these tables have) at `at`
    /// ([`Self::add_to`]); `base`, `scale` and `yarn` as for [`Self::with_yarn`].
    #[cfg(feature = "cuda")]
    pub(crate) fn stage(
        &self,
        staging: &mut crate::graph::Staging,
        at: (usize, usize),
        start: usize,
        (base, scale, yarn): (f64, f64, Yarn),
    ) {
        let [_, _, len, dim] = self.cos.dims();
        let mut cos = vec![0.0; len * dim];
        rope_values(start, dim, base, scale, yarn, &mut cos, staging.floats(at.1, len * dim));
        staging.floats(at.0, len * dim).copy_from_slice(&cos);
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

/// The `cos` and `sin` tables of [`Rope::with_yarn`] for positions `start..start + cos.len() /
/// head_dim`, written into `cos` and `sin` (row-major `[T, head_dim]`).
pub(crate) fn rope_values(
    start: usize,
    head_dim: usize,
    base: f64,
    scale: f64,
    yarn: Yarn,
    cos: &mut [f32],
    sin: &mut [f32],
) {
    let half = head_dim / 2;
    let temperature = yarn.temperature();
    for (row, (cos, sin)) in cos.chunks_mut(head_dim).zip(sin.chunks_mut(head_dim)).enumerate() {
        let pos = (start + row) as f64 / scale;
        for i in 0..half {
            let angle = pos * yarn.rate(1.0 / base.powf((2 * i) as f64 / head_dim as f64));
            let c = (angle.cos() * temperature) as f32;
            let s = (angle.sin() * temperature) as f32;
            (cos[i], cos[i + half], sin[i], sin[i + half]) = (c, c, s, s);
        }
    }
}

/// `x + pending` (or `x`) and its norm, through [`crate::fused::add_rms_norm`] when there is a
/// pending residual.
pub(crate) fn norm_after(
    norm: &RmsNorm,
    x: Tensor<3>,
    pending: Option<Tensor<3>>,
) -> (Tensor<3>, Tensor<3>) {
    match pending {
        Some(delta) => crate::fused::add_rms_norm(norm, x, delta),
        None => (x.clone(), crate::fused::rms_norm(norm, x)),
    }
}

/// QK-norm then RoPE: `x` `[B, T, H, D]` normed per head row by `norm`, moved to
/// `[B, H, T, D]` and rotated; one launch on CUDA with the fused kernels on.
fn norm_rope(norm: &RmsNorm, x: Tensor<4>, rope: &Rope) -> Tensor<4> {
    #[cfg(feature = "cuda")]
    if crate::fused::active(&x.device()) {
        let swapped = x.clone().swap_dims(1, 2);
        let gamma = (norm.gamma.val(), norm.epsilon);
        if let Some(y) = crate::rope_fused::rope(swapped, rope, Some(gamma)) {
            return y;
        }
    }
    apply_rope(crate::fused::rms_norm(norm, x).swap_dims(1, 2), rope)
}

/// Rotate `x` `[B, H, T, D]` by the position tables: one launch on CUDA with the fused
/// kernels on ([`crate::rope_fused::rope`]), the reference ops otherwise.
pub fn apply_rope(x: Tensor<4>, rope: &Rope) -> Tensor<4> {
    #[cfg(feature = "cuda")]
    if crate::fused::active(&x.device())
        && let Some(y) = crate::rope_fused::rope(x.clone(), rope, None)
    {
        return y;
    }
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
    // Decode steps on CUDA: one fused kernel instead of matmul, softmax and matmul.
    #[cfg(feature = "cuda")]
    if crate::kernels::attend_decode_fits(groups * t, d, s) && crate::fused::active(&q.device()) {
        return crate::kernels::attend_decode(q, k, v, mask, t).reshape([b, h, t, d]);
    }
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

    /// A pre-norm block's whole MLP half in two fused launches (CUDA, decode-sized runs):
    /// `s = x + delta` (or `x`), then `s + mlp(norm(s))`, the new residual stream
    /// ([`crate::mlp_fused`]). None when the fused kernels do not take it; the caller then runs
    /// [`crate::fused::add_rms_norm`] and [`Self::forward`].
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    pub fn residual_step(
        &self,
        norm: &RmsNorm,
        x: &Tensor<3>,
        delta: Option<&Tensor<3>>,
    ) -> Option<Tensor<3>> {
        #[cfg(feature = "cuda")]
        {
            if !crate::fused::fused_mlp_active(&x.device()) {
                return None;
            }
            let [b, t, d] = x.dims();
            let rows = b * t;
            let x2 = x.clone().reshape([rows, d]);
            let delta = delta.map(|v| v.clone().reshape([rows, d]));
            let gamma = norm.gamma.val();
            let (act, sum) =
                crate::mlp_fused::mlp_in(x2.clone(), delta, gamma, norm.epsilon, &self.gate_up)?;
            let resid = sum.unwrap_or(x2);
            let out = match crate::mlp_fused::mlp_out(act.clone(), &self.down, resid.clone()) {
                Some(out) => out,
                None => resid + self.down.forward(act),
            };
            Some(out.reshape([b, t, d]))
        }
        #[cfg(not(feature = "cuda"))]
        None
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
        self.forward_in(self.input(x), rope, cache)
    }

    /// [`Self::forward`] of `x` `[B, T, d]` whose batch rows are separate streams, row i with
    /// its own cache `caches[i]` (built for batch 1) and the same RoPE positions: the input and
    /// output projections run once over every row, the attention per row.
    pub fn forward_rows(
        &self,
        x: Tensor<3>,
        rope: &Rope,
        caches: &mut [&mut KvCache],
    ) -> Tensor<3> {
        let input = self.input(x);
        let rows: Vec<MixOut> = caches
            .iter_mut()
            .enumerate()
            .map(|(i, cache)| self.core_in(input.row(i), rope, Some(&mut **cache)))
            .collect();
        self.project_out(MixOut::cat(rows))
    }

    /// [`Self::forward`] from the input projections ([`Self::input`]).
    fn forward_in(&self, input: AttnIn, rope: &Rope, cache: Option<&mut KvCache>) -> Tensor<3> {
        self.project_out(self.core_in(input, rope, cache))
    }

    /// [`Self::forward_in`] up to the output projection.
    fn core_in(&self, input: AttnIn, rope: &Rope, cache: Option<&mut KvCache>) -> MixOut {
        let t = input.qkv.dims()[1];
        let device = input.qkv.device();
        let (gate, q, k, v) = self.project_from(input, rope);
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
        MixOut::heads(out, gate)
    }

    /// [`Self::forward`] with a cache inside a captured graph: the new keys and values land at
    /// the step's device slot and the step's device mask hides what the eager path leaves out
    /// ([`crate::graph`]). Runs of more than one token attend over the ring before the write
    /// and their own keys, like [`KvCache::append`].
    /// The output projection is left to the caller ([`Self::project_out`]).
    #[cfg(feature = "cuda")]
    fn core_step(&self, input: AttnIn, rope: &Rope, cache: &mut KvCache, step: &KvStep) -> MixOut {
        let t = input.qkv.dims()[1];
        let (gate, q, k, v) = self.project_from(input, rope);
        let (keys, values) = if t == 1 {
            crate::graph::write_kv(cache, k, v, step);
            let (ck, cv) = cache.buffers();
            (ck.clone(), cv.clone())
        } else {
            let (ck, cv) = cache.buffers();
            let keys = Tensor::cat(vec![ck.clone(), k.clone()], 2);
            let values = Tensor::cat(vec![cv.clone(), v.clone()], 2);
            crate::graph::write_kv(cache, k, v, step);
            (keys, values)
        };
        MixOut::heads(attend(q, keys, values, Some(step.mask.clone())), gate)
    }

    /// Attention over `x` itself without a cache, with the given `[T, T]` mask (`true` hides a
    /// key; None: every position sees every other). A mask built ahead of time keeps the call
    /// free of host-to-device copies, so it can run inside a CUDA graph capture.
    pub fn forward_masked(
        &self,
        x: Tensor<3>,
        rope: &Rope,
        mask: Option<Tensor<2, Bool>>,
    ) -> Tensor<3> {
        let (gate, q, k, v) = self.project(x, rope);
        self.output(attend(q, k, v, mask), gate)
    }

    /// The output gate (v2), and queries, keys (with RoPE) and values `[B, H or KVH, T, D]`.
    fn project(
        &self,
        x: Tensor<3>,
        rope: &Rope,
    ) -> (Option<Tensor<3>>, Tensor<4>, Tensor<4>, Tensor<4>) {
        self.project_from(self.input(x), rope)
    }

    /// The input projections of `x`: the sigmoid output gate (v2) and the joint QKV.
    fn input(&self, x: Tensor<3>) -> AttnIn {
        let gate = self.gate.as_ref().map(|g| sigmoid(g.forward(x.clone())));
        AttnIn { gate, qkv: self.qkv.forward(x) }
    }

    /// [`Self::input`] of the RMS-normed `x` `[B, T, d]` with the norm (and the gate's sigmoid)
    /// fused into the projections' GEMVs ([`crate::mlp_fused::norm_linear`]); None when the
    /// fused kernels do not take them.
    #[cfg(feature = "cuda")]
    fn fused_input(&self, norm: &RmsNorm, x: &Tensor<3>) -> Option<AttnIn> {
        let [b, t, d] = x.dims();
        let x2 = x.clone().reshape([b * t, d]);
        let (gamma, eps) = (norm.gamma.val(), norm.epsilon);
        let project = |layer: &Linear, sigmoid: bool| {
            let y = crate::mlp_fused::norm_linear(x2.clone(), gamma.clone(), eps, layer, sigmoid)?;
            let n = y.dims()[1];
            Some(y.reshape([b, t, n]))
        };
        let gate = match &self.gate {
            Some(g) => Some(project(g, true)?),
            None => None,
        };
        Some(AttnIn { gate, qkv: project(&self.qkv, false)? })
    }

    /// The queries, keys (with QK-norm and RoPE) and values from the input projections.
    fn project_from(
        &self,
        input: AttnIn,
        rope: &Rope,
    ) -> (Option<Tensor<3>>, Tensor<4>, Tensor<4>, Tensor<4>) {
        let AttnIn { gate, qkv } = input;
        let [b, t, _] = qkv.dims();
        let (h, kvh, hd) = (self.heads, self.kv_heads, self.head_dim);
        let (q, k) = match (&self.q_norm, &self.k_norm) {
            (Some(q_norm), Some(k_norm)) => {
                let q = qkv.clone().narrow(2, 0, h * hd).reshape([b, t, h, hd]);
                let k = qkv.clone().narrow(2, h * hd, kvh * hd).reshape([b, t, kvh, hd]);
                (norm_rope(q_norm, q, rope), norm_rope(k_norm, k, rope))
            }
            _ => {
                // Queries and keys share positions, so one RoPE pass covers both.
                let qk = qkv.clone().narrow(2, 0, (h + kvh) * hd);
                let qk = apply_rope(qk.reshape([b, t, h + kvh, hd]).swap_dims(1, 2), rope);
                (qk.clone().narrow(1, 0, h), qk.narrow(1, h, kvh))
            }
        };
        let v = qkv.narrow(2, (h + kvh) * hd, kvh * hd).reshape([b, t, kvh, hd]).swap_dims(1, 2);
        (gate, q, k, v)
    }

    /// Heads `[B, H, T, D]` merged, gated (v2) and projected out.
    fn output(&self, out: Tensor<4>, gate: Option<Tensor<3>>) -> Tensor<3> {
        self.project_out(MixOut::heads(out, gate))
    }

    /// The output projection of a [`MixOut`] (gated first).
    pub(crate) fn project_out(&self, out: MixOut) -> Tensor<3> {
        out.project(&self.o)
    }
}

/// A sequence mixer's result before its output projection: the merged heads `[B, T, W]` and,
/// when not yet applied, the sigmoid output gate of the same shape. Keeping the projection
/// apart lets a batched step ([`Stack::forward_slots`]) run each slot's mixer core on its own
/// caches and the output projection once over every slot's rows.
pub(crate) struct MixOut {
    pub(crate) x: Tensor<3>,
    pub(crate) gate: Option<Tensor<3>>,
}

impl MixOut {
    /// Heads `[B, H, T, D]` merged to `[B, T, H D]`, with the gate still to apply.
    pub(crate) fn heads(out: Tensor<4>, gate: Option<Tensor<3>>) -> Self {
        let [b, h, t, hd] = out.dims();
        Self { x: out.swap_dims(1, 2).reshape([b, t, h * hd]), gate }
    }

    /// Batch rows of several results stacked (gates kept apart only when every row has one).
    pub(crate) fn cat(rows: Vec<MixOut>) -> Self {
        let gated = rows.iter().all(|r| r.gate.is_some());
        let (mut xs, mut gates) = (Vec::with_capacity(rows.len()), Vec::new());
        for row in rows {
            match (gated, row.gate) {
                (true, Some(g)) => {
                    xs.push(row.x);
                    gates.push(g);
                }
                (_, Some(g)) => xs.push(row.x * g),
                (_, None) => xs.push(row.x),
            }
        }
        let gate = gated.then(|| Tensor::cat(gates, 0));
        Self { x: Tensor::cat(xs, 0), gate }
    }

    /// `(x * gate) W` through `o`, the gate's product in the GEMV's prologue on CUDA for
    /// decode-sized runs.
    pub(crate) fn project(self, o: &Linear) -> Tensor<3> {
        let [b, t, w] = self.x.dims();
        #[cfg(feature = "cuda")]
        if let Some(gate) = &self.gate
            && crate::fused::fused_mlp_active(&self.x.device())
        {
            let rows = b * t;
            let (x2, g2) = (self.x.clone().reshape([rows, w]), gate.clone().reshape([rows, w]));
            if let Some(y) = crate::mlp_fused::gated_linear(x2, g2, o) {
                let d = y.dims()[1];
                return y.reshape([b, t, d]);
            }
        }
        let _ = (b, t, w);
        let x = match self.gate {
            Some(gate) => self.x * gate,
            None => self.x,
        };
        o.forward(x)
    }
}

impl AttnIn {
    /// Batch row `i` (one slot's).
    fn row(&self, i: usize) -> AttnIn {
        let pick = |t: &Tensor<3>| t.clone().narrow(0, i, 1);
        AttnIn { gate: self.gate.as_ref().map(pick), qkv: pick(&self.qkv) }
    }
}

/// An attention layer's input projections: the sigmoid output gate (v2) and the joint QKV
/// `[B, T, ..]`.
pub(crate) struct AttnIn {
    gate: Option<Tensor<3>>,
    qkv: Tensor<3>,
}

/// What a block's mixer starts from: the normed residual stream, or its input projections
/// already computed with the norm fused in ([`Block::fused_input`]).
// One per block step, short-lived: not worth boxing the larger variant.
#[allow(clippy::large_enum_variant)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
enum MixInput {
    Normed(Tensor<3>),
    /// DeltaNet's or retrieval's input projection.
    Projected(Tensor<3>),
    Attention(AttnIn),
}

impl MixInput {
    /// Batch row `i` of the input (one slot's).
    fn row(&self, i: usize) -> MixInput {
        let pick = |t: &Tensor<3>| t.clone().narrow(0, i, 1);
        match self {
            Self::Normed(h) => Self::Normed(pick(h)),
            Self::Projected(p) => Self::Projected(pick(p)),
            Self::Attention(a) => Self::Attention(a.row(i)),
        }
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
        self.kv_from(self.kv.forward(states), rope)
    }

    /// [`Self::make_kv`] for batch rows that are separate streams, row i at the positions of
    /// `ropes[i]`: the key/value projection once over every row, RoPE per row.
    pub fn make_kv_rows(&self, states: Tensor<3>, ropes: &[Rope]) -> Vec<(Tensor<4>, Tensor<4>)> {
        let kv = self.kv.forward(states);
        ropes
            .iter()
            .enumerate()
            .map(|(i, rope)| self.kv_from(kv.clone().narrow(0, i, 1), rope))
            .collect()
    }

    /// Keys (with RoPE) and values `[B, KVH, T, D]` from the joint projection `kv`.
    fn kv_from(&self, kv: Tensor<3>, rope: &Rope) -> (Tensor<4>, Tensor<4>) {
        let [b, t, _] = kv.dims();
        let (kvh, hd) = (self.kv_heads, self.head_dim);
        let k = kv.clone().narrow(2, 0, kvh * hd);
        let v = kv.narrow(2, kvh * hd, kvh * hd);
        let k = apply_rope(k.reshape([b, t, kvh, hd]).swap_dims(1, 2), rope);
        (k, v.reshape([b, t, kvh, hd]).swap_dims(1, 2))
    }

    /// [`Self::forward`] for batch rows that are separate streams: row i of `x` `[B, T, d]` at
    /// the positions of `ropes[i]` against `kvs[i]`. The query and output projections run once
    /// over every row, the attention per row.
    pub fn forward_rows(
        &self,
        x: Tensor<3>,
        ropes: &[Rope],
        kvs: &[(Tensor<4>, Tensor<4>)],
    ) -> Tensor<3> {
        let [b, t, _] = x.dims();
        let (h, hd) = (self.heads, self.head_dim);
        let q = self.q.forward(x).reshape([b, t, h, hd]).swap_dims(1, 2);
        let rows: Vec<Tensor<4>> = kvs
            .iter()
            .enumerate()
            .map(|(i, (keys, values))| {
                let qi = apply_rope(q.clone().narrow(0, i, 1), &ropes[i]);
                attend(qi, keys.clone(), values.clone(), None)
            })
            .collect();
        let out = Tensor::cat(rows, 0);
        self.o.forward(out.swap_dims(1, 2).reshape([b, t, h * hd]))
    }

    pub fn forward(
        &self,
        x: Tensor<3>,
        rope_q: &Rope,
        keys: Tensor<4>,
        values: Tensor<4>,
        mask: Option<Tensor<2, Bool>>,
    ) -> Tensor<3> {
        cross_attend(&self.q, &self.o, self.heads, self.head_dim, x, rope_q, keys, values, mask)
    }
}

/// Cross-attention queries and output over keys and values that another layer projected: the
/// NCM draft head reads the local decoder's cross-attention caches through it.
#[derive(Module, Debug)]
pub struct CrossQuery {
    q: Linear,
    o: Linear,
    heads: usize,
    head_dim: usize,
}

impl CrossQuery {
    pub fn new(d: usize, heads: usize, head_dim: usize, device: &Device) -> Self {
        Self {
            q: linear(d, heads * head_dim, device),
            o: linear(heads * head_dim, d, device),
            heads,
            head_dim,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<3>,
        rope_q: &Rope,
        keys: Tensor<4>,
        values: Tensor<4>,
        mask: Option<Tensor<2, Bool>>,
    ) -> Tensor<3> {
        cross_attend(&self.q, &self.o, self.heads, self.head_dim, x, rope_q, keys, values, mask)
    }
}

/// Queries `x` `[B, T, d]` through `q`, attends over `keys` and `values` `[B, KVH, S, D]` and
/// projects out through `o`.
#[allow(clippy::too_many_arguments)]
fn cross_attend(
    q: &Linear,
    o: &Linear,
    h: usize,
    hd: usize,
    x: Tensor<3>,
    rope_q: &Rope,
    keys: Tensor<4>,
    values: Tensor<4>,
    mask: Option<Tensor<2, Bool>>,
) -> Tensor<3> {
    let [b, t, _] = x.dims();
    let q = q.forward(x).reshape([b, t, h, hd]).swap_dims(1, 2);
    let out = attend(apply_rope(q, rope_q), keys, values, mask);
    o.forward(out.swap_dims(1, 2).reshape([b, t, h * hd]))
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

    /// `step`: the layer's cache position in a captured graph (attention layers only).
    fn mix(
        &self,
        input: MixInput,
        start: usize,
        ropes: &Ropes,
        cache: Option<&mut LayerCache>,
        step: Option<&KvStep>,
    ) -> Tensor<3> {
        self.project_out(self.mix_core(input, start, ropes, cache, step))
    }

    /// The mixer's input projections of the normed input `h` (all batch rows at once).
    fn project_in(&self, h: Tensor<3>) -> MixInput {
        match &self.attn {
            Mixer::Attention(attn) => MixInput::Attention(attn.input(h)),
            Mixer::DeltaNet(delta) => MixInput::Projected(delta.input_projection().forward(h)),
            Mixer::Retrieval(ret) => MixInput::Projected(ret.input_projection().forward(h)),
        }
    }

    /// The mixer's output projection.
    fn project_out(&self, out: MixOut) -> Tensor<3> {
        match &self.attn {
            Mixer::Attention(attn) => attn.project_out(out),
            Mixer::DeltaNet(delta) => delta.project_out(out),
            Mixer::Retrieval(ret) => ret.project_out(out),
        }
    }

    /// [`Self::mix`] up to the output projection.
    fn mix_core(
        &self,
        input: MixInput,
        start: usize,
        ropes: &Ropes,
        cache: Option<&mut LayerCache>,
        step: Option<&KvStep>,
    ) -> MixOut {
        fn table(rope: &Option<Rope>) -> &Rope {
            rope.as_ref().expect("the stack built this table")
        }
        match &self.attn {
            Mixer::Attention(attn) => {
                let rope = table(if attn.window().is_some() { &ropes.local } else { &ropes.full });
                let input = match input {
                    MixInput::Normed(h) => attn.input(h),
                    MixInput::Attention(input) => input,
                    MixInput::Projected(_) => unreachable!("an attention layer's projections"),
                };
                match step {
                    #[cfg(feature = "cuda")]
                    Some(step) => {
                        let cache = cache.expect("a graphed step has caches").kv();
                        attn.core_step(input, rope, cache, step)
                    }
                    #[cfg(not(feature = "cuda"))]
                    Some(step) => match *step {},
                    None => attn.core_in(input, rope, cache.map(LayerCache::kv)),
                }
            }
            Mixer::DeltaNet(delta) => {
                let p = match input {
                    MixInput::Normed(h) => delta.input_projection().forward(h),
                    MixInput::Projected(p) => p,
                    MixInput::Attention(_) => unreachable!("a DeltaNet layer's projection"),
                };
                delta.core_projected(p, cache.map(LayerCache::delta))
            }
            Mixer::Retrieval(ret) => {
                let p = match input {
                    MixInput::Normed(h) => ret.input_projection().forward(h),
                    MixInput::Projected(p) => p,
                    MixInput::Attention(_) => unreachable!("a retrieval layer's projection"),
                };
                let rope = table(&ropes.retrieval);
                ret.core_projected(p, start, rope, cache.map(LayerCache::latent))
            }
        }
    }

    /// The mixer's input projections of the `n1`-normed `x` with the norm folded into their
    /// GEMVs (CUDA, decode-sized runs, [`crate::mlp_fused::norm_linear`]); None when the fused
    /// kernels do not take them.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn fused_input(&self, x: &Tensor<3>) -> Option<MixInput> {
        #[cfg(feature = "cuda")]
        {
            if !crate::fused::fused_mlp_active(&x.device()) {
                return None;
            }
            let [b, t, d] = x.dims();
            let single = |layer: &Linear| {
                let x2 = x.clone().reshape([b * t, d]);
                let (gamma, eps) = (self.n1.gamma.val(), self.n1.epsilon);
                let p = crate::mlp_fused::norm_linear(x2, gamma, eps, layer, false)?;
                let n = p.dims()[1];
                Some(MixInput::Projected(p.reshape([b, t, n])))
            };
            match &self.attn {
                Mixer::Attention(attn) => attn.fused_input(&self.n1, x).map(MixInput::Attention),
                Mixer::DeltaNet(delta) => single(delta.input_projection()),
                Mixer::Retrieval(ret) => single(ret.input_projection()),
            }
        }
        #[cfg(not(feature = "cuda"))]
        None
    }

    /// [`Self::forward`] for a batch whose rows are separate streams (slots), each with its own
    /// position `starts[i]`, tables `ropes[i]` and caches `caches[i][layer]` (and in a captured
    /// graph its layer's cursor step `steps[i][layer]`): the norms, the mixer's input projections
    /// and the MLP run over the whole batch, the mixer itself (caches, attention, its output
    /// projection) once per row. TODO: batch the mixers' output projections too.
    #[allow(clippy::too_many_arguments)]
    fn forward_slots(
        &self,
        x: Tensor<3>,
        pending: Option<Tensor<3>>,
        starts: &[usize],
        ropes: &[&Ropes],
        caches: &mut [&mut [LayerCache]],
        layer: usize,
        steps: Option<&[Vec<Option<KvStep>>]>,
    ) -> (Tensor<3>, Option<Tensor<3>>) {
        let fused = if pending.is_none() { self.fused_input(&x) } else { None };
        let (x, input) = match fused {
            Some(input) => (x, input),
            None => {
                // The input projections once over every slot's row.
                let (x, h) = norm_after(&self.n1, x, pending);
                (x, self.project_in(h))
            }
        };
        // Each slot's mixer core on its own caches, then the output projection once over all
        // rows: one weight read per layer instead of one per slot.
        let rows: Vec<MixOut> = caches
            .iter_mut()
            .enumerate()
            .map(|(i, c)| {
                let step = steps.and_then(|s| s[i][layer].as_ref());
                self.mix_core(input.row(i), starts[i], ropes[i], Some(&mut c[layer]), step)
            })
            .collect();
        let mixed = self.project_out(MixOut::cat(rows));
        if let Some(x) = self.mlp.residual_step(&self.n2, &x, Some(&mixed)) {
            return (x, None);
        }
        let (x, h) = crate::fused::add_rms_norm(&self.n2, x, mixed);
        (x, Some(self.mlp.forward(h)))
    }

    /// The block on `x` plus `pending` (the previous block's MLP output, not yet added: the add
    /// runs in one kernel with this block's first norm, [`crate::fused::add_rms_norm`]).
    /// Returns the residual stream after the mixer and this block's MLP output, likewise
    /// pending; or, when the fused MLP kernels ran ([`Mlp::residual_step`]), the stream with
    /// the MLP output already added and nothing pending.
    fn forward(
        &self,
        x: Tensor<3>,
        pending: Option<Tensor<3>>,
        start: usize,
        ropes: &Ropes,
        cache: Option<&mut LayerCache>,
        step: Option<&KvStep>,
    ) -> (Tensor<3>, Option<Tensor<3>>) {
        // The pre-mixer norm folded into the mixer's input projections when nothing is pending
        // (the fused MLP leaves nothing pending), else run before it.
        let fused = if pending.is_none() { self.fused_input(&x) } else { None };
        let (x, input) = match fused {
            Some(input) => (x, input),
            None => {
                let (x, h) = norm_after(&self.n1, x, pending);
                (x, MixInput::Normed(h))
            }
        };
        let mixed = self.mix(input, start, ropes, cache, step);
        if let Some(x) = self.mlp.residual_step(&self.n2, &x, Some(&mixed)) {
            return (x, None);
        }
        let (x, h) = crate::fused::add_rms_norm(&self.n2, x, mixed);
        (x, Some(self.mlp.forward(h)))
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
            index_8bit: v2.idx_8bit,
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
    /// YaRN on the retrieval layers' decoupled key.
    #[module(skip)]
    retrieval_yarn: Yarn,
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
            retrieval_yarn: Yarn::of(&spec.v2),
        }
    }

    /// The (head dim, base, scale, YaRN) of the RoPE tables of each layer type: sliding
    /// attention, full attention and retrieval.
    fn rope_params_all(&self) -> [(usize, f64, f64, Yarn); 3] {
        let (hd, full) = (self.head_dim, (self.rope_base_full, self.rope_scale_full));
        [
            (hd, self.rope_base, self.rope_scale, Yarn::OFF),
            (hd, full.0, full.1, Yarn::OFF),
            (self.retrieval_rope_dim, full.0, full.1, self.retrieval_yarn),
        ]
    }

    /// [`Self::rope_params_all`] for the layer types the stack has, None for the others.
    fn rope_params(&self) -> [Option<(usize, f64, f64, Yarn)>; 3] {
        let (mut local, mut full, mut retrieval) = (false, false, false);
        for block in &self.blocks {
            match &block.attn {
                Mixer::Attention(a) if a.window().is_some() => local = true,
                Mixer::Attention(_) => full = true,
                Mixer::Retrieval(_) => retrieval = true,
                Mixer::DeltaNet(_) => {}
            }
        }
        let [l, f, r] = self.rope_params_all();
        [local.then_some(l), full.then_some(f), retrieval.then_some(r)]
    }

    fn ropes(&self, start: usize, t: usize, device: &Device) -> Ropes {
        let [local, full, retrieval] = self.rope_params().map(|p| {
            p.map(|(dim, base, scale, yarn)| {
                Rope::with_yarn(start, t, dim, base, scale, yarn, device)
            })
        });
        Ropes { local, full, retrieval }
    }

    pub fn forward(&self, x: Tensor<3>, caches: Option<&mut [LayerCache]>) -> Tensor<3> {
        let t = x.dims()[1];
        let device = x.device();
        let start = caches.as_ref().map_or(0, |c| c[0].len());
        let ropes = self.ropes(start, t, &device);
        self.forward_with(x, start, &ropes, caches, None)
    }

    /// [`Self::forward`] with the RoPE tables given, and in a captured graph each attention
    /// layer's cache position (`steps`, by layer).
    fn forward_with(
        &self,
        x: Tensor<3>,
        start: usize,
        ropes: &Ropes,
        mut caches: Option<&mut [LayerCache]>,
        steps: Option<&[Option<KvStep>]>,
    ) -> Tensor<3> {
        let (mut x, mut pending) = (x, None);
        for (i, block) in self.blocks.iter().enumerate() {
            let cache = caches.as_mut().map(|c| &mut c[i]);
            let step = steps.and_then(|s| s[i].as_ref());
            (x, pending) = block.forward(x, pending, start, ropes, cache, step);
        }
        norm_after(&self.norm, x, pending).1
    }

    /// One step of `x` `[B, t, d]` whose batch rows are separate streams, row i with its own
    /// caches `caches[i]` (each built for batch 1) at its own position: continuous batching of
    /// requests. Each row's result equals [`Self::forward`] of that row alone up to rounding.
    pub fn forward_slots(&self, x: Tensor<3>, caches: &mut [&mut [LayerCache]]) -> Tensor<3> {
        let t = x.dims()[1];
        assert_eq!(x.dims()[0], caches.len(), "one cache set per batch row");
        let device = x.device();
        let starts: Vec<usize> = caches.iter().map(|c| c[0].len()).collect();
        let ropes: Vec<Ropes> = starts.iter().map(|&s| self.ropes(s, t, &device)).collect();
        let ropes: Vec<&Ropes> = ropes.iter().collect();
        self.forward_slots_with(x, &starts, &ropes, caches, None)
    }

    /// [`Self::forward_slots`] with the tables given, and in a captured graph each row's layer
    /// steps.
    fn forward_slots_with(
        &self,
        x: Tensor<3>,
        starts: &[usize],
        ropes: &[&Ropes],
        caches: &mut [&mut [LayerCache]],
        steps: Option<&[Vec<Option<KvStep>>]>,
    ) -> Tensor<3> {
        let (mut x, mut pending) = (x, None);
        for (i, block) in self.blocks.iter().enumerate() {
            (x, pending) = block.forward_slots(x, pending, starts, ropes, caches, i, steps);
        }
        norm_after(&self.norm, x, pending).1
    }

    /// One cache per layer for runs of up to `max_positions`.
    pub fn make_caches(
        &self,
        max_positions: usize,
        batch: usize,
        device: &Device,
    ) -> Vec<LayerCache> {
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut caches: Vec<LayerCache> =
            self.blocks.iter().map(|b| b.make_cache(max_positions, batch, device)).collect();
        // The retrieval layers are always at one position: one shared position buffer, which a
        // captured step stages once.
        #[cfg(feature = "cuda")]
        {
            let mut first: Option<usize> = None;
            for i in 0..caches.len() {
                if !matches!(caches[i], LayerCache::Latent(_)) {
                    continue;
                }
                match first {
                    None => first = Some(i),
                    Some(f) => {
                        let (head, tail) = caches.split_at_mut(i);
                        if let (LayerCache::Latent(owner), LayerCache::Latent(c)) =
                            (&head[f], &mut tail[0])
                        {
                            c.share_position(owner);
                        }
                    }
                }
            }
        }
        caches
    }
}

/// One stack's cached step over a fixed number of tokens inside a captured CUDA graph: the
/// RoPE tables in buffers of their own and each attention layer's cursor
/// ([`crate::graph::CursorSet`]). With the fused kernels every launch of such a step is the same
/// at every position: DeltaNet layers update their state in place, retrieval layers (one-token
/// steps only) take the position from a device buffer and size their grids for the largest
/// step, and attention layers write and mask through their cursor. Before each replay
/// [`Self::refresh`] stages the tables and positions ([`crate::graph::Staging`]); after it
/// [`Self::advance`] moves the caches' host positions.
#[cfg(feature = "cuda")]
pub(crate) struct StackStep {
    t: usize,
    ropes: Ropes,
    /// Where each of `ropes`' tables (local, full, retrieval) is staged: (cos, sin) offsets.
    staged: [Option<(usize, usize)>; 3],
    /// Each layer's cursor (attention layers).
    cursors: Vec<Option<usize>>,
    /// Each retrieval layer's staged position.
    positions: Vec<Option<usize>>,
}

/// What a warm-up step changed in one layer's cache, to undo.
#[cfg(feature = "cuda")]
pub(crate) enum SavedLayer {
    Kv(crate::graph::KvSaved),
    Delta((Tensor<4>, Tensor<3>, usize)),
    Latent(usize),
}

#[cfg(feature = "cuda")]
impl StackStep {
    /// Fixed inputs for runs of `t` tokens from the caches' current position, registering the
    /// attention caches in `set`. Fails for steps a graph cannot replay.
    pub(crate) fn new(
        stack: &Stack,
        caches: &[LayerCache],
        t: usize,
        set: &mut crate::graph::CursorSet,
        staging: &mut crate::graph::Staging,
    ) -> Result<Self, String> {
        let device = stack.norm.gamma.device();
        let (mut cursors, mut positions) = (Vec::new(), Vec::new());
        let mut staged_ids: Vec<(u64, usize)> = Vec::new();
        for cache in caches {
            cursors.push(match cache {
                LayerCache::Kv(kv) => Some(set.add(kv, t, staging)?),
                _ => None,
            });
            positions.push(match cache {
                LayerCache::Latent(_) if t > 1 => {
                    return Err(format!("retrieval layers step one token at a time, not {t}"));
                }
                LayerCache::Latent(c) => {
                    Some(match staged_ids.iter().find(|(id, _)| *id == c.position_id()) {
                        // A position buffer shared with an earlier layer is staged once.
                        Some(&(_, at)) => at,
                        None => {
                            let at = staging.add_ints(c.position().clone());
                            staged_ids.push((c.position_id(), at));
                            at
                        }
                    })
                }
                _ => None,
            });
        }
        let ropes = stack.ropes(caches[0].len(), t, &device);
        let mut staged = [None; 3];
        let mut fixed = |i: usize, r: Option<Rope>| {
            r.map(|r| {
                let r = r.fixed();
                staged[i] = Some(r.add_to(staging));
                r
            })
        };
        let ropes = Ropes {
            local: fixed(0, ropes.local),
            full: fixed(1, ropes.full),
            retrieval: fixed(2, ropes.retrieval),
        };
        Ok(Self { t, ropes, staged, cursors, positions })
    }

    /// Before a replay (or the warm-ups): stages the RoPE tables of the caches' position and
    /// the retrieval positions, and records the attention positions in `set` (which the caller
    /// stages with [`crate::graph::CursorSet::refresh`] before uploading).
    pub(crate) fn refresh(
        &self,
        stack: &Stack,
        caches: &[LayerCache],
        set: &mut crate::graph::CursorSet,
        staging: &mut crate::graph::Staging,
    ) {
        let start = caches[0].len();
        let ropes = [&self.ropes.local, &self.ropes.full, &self.ropes.retrieval];
        for ((params, staged), rope) in stack.rope_params().iter().zip(&self.staged).zip(ropes) {
            if let (Some((_, base, scale, yarn)), Some(at), Some(rope)) = (params, staged, rope) {
                rope.stage(staging, *at, start, (*base, *scale, *yarn));
            }
        }
        let layers = caches.iter().zip(&self.cursors).zip(&self.positions);
        for ((cache, cursor), position) in layers {
            match (cache, cursor, position) {
                (LayerCache::Kv(kv), Some(i), _) => set.sync(*i, kv),
                (LayerCache::Latent(c), _, Some(at)) => {
                    // A replay skips the eager step's bound check: refuse to write past the end.
                    assert!(c.len() < c.capacity(), "cache capacity {} exceeded", c.capacity());
                    staging.ints(*at, 1)[0] = c.len() as i32
                }
                _ => {}
            }
        }
    }

    /// The step itself (what the graph records): `x` `[B, t, d]` through the stack, with the
    /// cursors' slots and masks `steps` ([`crate::graph::CursorSet::steps`]).
    pub(crate) fn run(
        &self,
        stack: &Stack,
        x: Tensor<3>,
        caches: &mut [LayerCache],
        steps: &[KvStep],
    ) -> Tensor<3> {
        let layer_steps: Vec<Option<KvStep>> =
            self.cursors.iter().map(|c| c.map(|i| steps[i].clone())).collect();
        let start = caches[0].len();
        stack.forward_with(x, start, &self.ropes, Some(caches), Some(&layer_steps))
    }

    /// [`Self::run`] for a batch of separate streams: row i through `steps[i]` (its own fixed
    /// tables and cursors, the latter indexing `kv[i]`, its cursor set's steps) and `caches[i]`.
    pub(crate) fn run_slots(
        stack: &Stack,
        steps: &[&StackStep],
        x: Tensor<3>,
        caches: &mut [&mut [LayerCache]],
        kv: &[Vec<KvStep>],
    ) -> Tensor<3> {
        let layer_steps: Vec<Vec<Option<KvStep>>> = steps
            .iter()
            .zip(kv)
            .map(|(s, kv)| s.cursors.iter().map(|c| c.map(|i| kv[i].clone())).collect())
            .collect();
        let starts: Vec<usize> = caches.iter().map(|c| c[0].len()).collect();
        let ropes: Vec<&Ropes> = steps.iter().map(|s| &s.ropes).collect();
        stack.forward_slots_with(x, &starts, &ropes, caches, Some(&layer_steps))
    }

    /// What a warm-up run changes, to put back with [`Self::restore`].
    pub(crate) fn save(&self, caches: &[LayerCache]) -> Vec<SavedLayer> {
        caches
            .iter()
            .map(|c| match c {
                LayerCache::Kv(kv) => SavedLayer::Kv(crate::graph::save_kv(kv, self.t)),
                LayerCache::Delta(d) => SavedLayer::Delta(d.snapshot()),
                LayerCache::Latent(l) => SavedLayer::Latent(l.len()),
            })
            .collect()
    }

    /// Undoes a warm-up run: DeltaNet states restored, attention rings put back, positions
    /// rewound (entries written past them are overwritten by the real step).
    pub(crate) fn restore(caches: &mut [LayerCache], saved: &[SavedLayer]) {
        for (cache, saved) in caches.iter_mut().zip(saved) {
            match (cache, saved) {
                (LayerCache::Kv(kv), SavedLayer::Kv(s)) => crate::graph::restore_kv(kv, s),
                (LayerCache::Delta(d), SavedLayer::Delta(s)) => d.restore(s.clone()),
                (LayerCache::Latent(l), SavedLayer::Latent(len)) => l.rewind(*len),
                _ => unreachable!("saved from the same caches"),
            }
        }
    }

    /// After a replay: the caches' host positions move by the step's tokens.
    pub(crate) fn advance(&self, caches: &mut [LayerCache]) {
        for cache in caches {
            match cache {
                LayerCache::Kv(kv) => kv.advance(self.t),
                other => other.advance(self.t),
            }
        }
    }
}

/// One-token steps of a stack as one captured CUDA graph per cache, replayed for every step
/// ([`StackStep`]): the v2 backbone (DeltaNet and retrieval layers) and the fallback backbone
/// (attention layers, through device cursors). Before each replay the input, the RoPE tables of
/// the position and the caches' positions are copied in; after it the caches' host positions
/// advance. Runs of more than one token, other devices or a failed capture run
/// [`Stack::forward`] eagerly.
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
    graph: crate::graph::Captured,
    input: Tensor<3>,
    step: StackStep,
    cursors: crate::graph::CursorSet,
    staging: crate::graph::Staging,
    output: Tensor<3>,
}

#[cfg(feature = "cuda")]
impl GraphedStack {
    /// `stack.forward(x, Some(caches))`, replayed from a graph when it can be. The returned
    /// tensor is the graph's output buffer: use it before the next call, which overwrites it.
    /// One `GraphedStack` serves one set of caches.
    pub fn forward(&mut self, stack: &Stack, x: Tensor<3>, caches: &mut [LayerCache]) -> Tensor<3> {
        let device = x.device();
        if x.dims()[1] != 1 || !crate::fused::graphs_active(&device) {
            return stack.forward(x, Some(caches));
        }
        if matches!(self.state, StackGraphState::Untried) {
            self.state = match stack.capture(&x, caches) {
                Ok(graph) => StackGraphState::Ready(Box::new(graph)),
                Err(e) => StackGraphState::Failed(e),
            };
        }
        let StackGraphState::Ready(g) = &mut self.state else {
            return stack.forward(x, Some(caches));
        };
        crate::kernels::assign_into(&g.input, x, 0, 0);
        g.step.refresh(stack, caches, &mut g.cursors, &mut g.staging);
        g.cursors.refresh(&mut g.staging);
        g.staging.upload(&device);
        // Safety: the graph's buffers are held by `g`, the stack's weights and the caches, all
        // alive across the replay (the replay checks it runs on the capturing thread).
        unsafe { crate::graph::replay(&device, &g.graph, "stack") };
        g.step.advance(caches);
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
    /// Warms up and records one step over fixed input buffers, leaving the caches as they were.
    fn capture(&self, x: &Tensor<3>, caches: &mut [LayerCache]) -> Result<StackGraph, String> {
        let device = x.device();
        let (mut cursors, mut staging) = Default::default();
        let step = StackStep::new(self, caches, 1, &mut cursors, &mut staging)?;
        let input = Tensor::<3>::zeros(x.dims(), &device);
        crate::kernels::assign_into(&input, x.clone(), 0, 0);
        step.refresh(self, caches, &mut cursors, &mut staging);
        cursors.refresh(&mut staging);
        staging.upload(&device);
        let saved = step.save(caches);
        let run = |caches: &mut [LayerCache]| {
            staging.scatter();
            step.run(self, input.clone(), caches, &cursors.steps())
        };
        let undo = |caches: &mut [LayerCache]| StackStep::restore(caches, &saved);
        let (graph, output) = crate::graph::record(&device, caches, 2, run, undo)?;
        Ok(StackGraph { graph, input, step, cursors, staging, output })
    }
}
