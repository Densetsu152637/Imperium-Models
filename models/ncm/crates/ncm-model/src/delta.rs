//! Gated DeltaNet (Yang et al., 2024, as used by Qwen3-Next and Qwen3.5): linear attention with a
//! fixed-size memory per head, decayed by a learned gate and corrected by the delta rule.
//!
//! Per head, with an L2-normalised key `k_t`, value `v_t`, decay `a_t = exp(g_t)` and write
//! strength `beta_t`, the state `S` (`[dk, dv]`) and the output follow
//!
//! ```text
//! S_t = a_t S_{t-1} + k_t u_t^T,   u_t = beta_t (v_t - a_t S_{t-1}^T k_t)
//! o_t = S_t^T q_t
//! ```
//!
//! Decoding runs this recurrence one token at a time ([`delta_rule_step`]). Training and prefill
//! use the chunked form ([`delta_rule_chunked`]): inside a chunk of `C` tokens the `u` vectors
//! solve a unit lower-triangular system, inverted exactly with log2(C) matrix products, and the
//! state is carried from chunk to chunk. Both are plain Burn tensor code (reference versions).

use burn::module::{Module, Param};
use burn::nn::{Initializer, Linear, RmsNorm};
use burn::tensor::activation::{sigmoid, silu, softplus};
use burn::tensor::{Bool, Device, Tensor, TensorData};

use crate::layers::{linear, rms_norm};

/// Norm floor of the L2 normalisation of queries and keys.
const L2_EPS: f64 = 1e-6;

/// `x / |x|` over the last axis.
fn l2_normalize(x: Tensor<4>) -> Tensor<4> {
    let norm = x.clone().square().sum_dim(3).add_scalar(L2_EPS).sqrt();
    x / norm
}

/// One recurrent step for a single token: `q`, `k` `[B, H, 1, dk]`, `v` `[B, H, 1, dv]`, log
/// decay `g` and write strength `beta` `[B, H, 1]`, state `[B, H, dk, dv]`. Returns the output
/// `[B, H, 1, dv]` and the new state.
pub fn delta_rule_step(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<3>,
    beta: Tensor<3>,
    state: Tensor<4>,
) -> (Tensor<4>, Tensor<4>) {
    let [b, h, _, _] = q.dims();
    let state = state * g.exp().reshape([b, h, 1, 1]);
    let u = (v - k.clone().matmul(state.clone())) * beta.reshape([b, h, 1, 1]);
    let state = state + k.swap_dims(2, 3).matmul(u);
    (q.matmul(state.clone()), state)
}

/// The recurrence token by token over `[B, H, T, *]` inputs (`g`, `beta` are `[B, H, T]`).
/// The exact reference the chunked form is tested against.
pub fn delta_rule_recurrent(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<3>,
    beta: Tensor<3>,
    state: Tensor<4>,
) -> (Tensor<4>, Tensor<4>) {
    let t = q.dims()[2];
    let mut state = state;
    let mut outs = Vec::with_capacity(t);
    for i in 0..t {
        let (o, s) = delta_rule_step(
            q.clone().narrow(2, i, 1),
            k.clone().narrow(2, i, 1),
            v.clone().narrow(2, i, 1),
            g.clone().narrow(2, i, 1),
            beta.clone().narrow(2, i, 1),
            state,
        );
        outs.push(o);
        state = s;
    }
    (Tensor::cat(outs, 2), state)
}

/// Pads axis 2 of `x` with `pad` zeros.
fn pad_time<const D: usize>(x: Tensor<D>, pad: usize) -> Tensor<D> {
    if pad == 0 {
        return x;
    }
    let mut shape = x.dims();
    shape[2] = pad;
    let zeros = Tensor::zeros(shape, &x.device());
    Tensor::cat(vec![x, zeros], 2)
}

/// `[C, C]` mask, `true` where the column is after the row (`j > i`), or at or after it when
/// `include_diagonal`.
fn upper_mask(c: usize, include_diagonal: bool, device: &Device) -> Tensor<2, Bool> {
    let data: Vec<bool> = (0..c * c)
        .map(|n| {
            let (i, j) = (n / c, n % c);
            j > i || (include_diagonal && j == i)
        })
        .collect();
    Tensor::from_data(TensorData::new(data, [c, c]), device)
}

/// The chunked (parallel) form over `[B, H, T, *]` inputs, chunks of `chunk` tokens, starting
/// from `state`. Equal to [`delta_rule_recurrent`] up to rounding.
pub fn delta_rule_chunked(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    g: Tensor<3>,
    beta: Tensor<3>,
    state: Tensor<4>,
    chunk: usize,
) -> (Tensor<4>, Tensor<4>) {
    let [b, h, t, dk] = q.dims();
    let dv = v.dims()[3];
    let device = q.device();
    let c = chunk.min(t).max(1);
    let n = t.div_ceil(c);
    let pad = n * c - t;
    // Padding is inert: zero keys and write strengths change nothing, zero log decay keeps the
    // state, and the padded outputs are dropped.
    let q = pad_time(q, pad).reshape([b, h, n, c, dk]);
    let k = pad_time(k, pad).reshape([b, h, n, c, dk]);
    let v = pad_time(v, pad).reshape([b, h, n, c, dv]);
    let beta = pad_time(beta, pad).reshape([b, h, n, c]).unsqueeze_dim::<5>(4);
    // Cumulative log decay inside each chunk: gamma_i = sum_{j <= i} g_j.
    let gamma = pad_time(g, pad).reshape([b, h, n, c]).cumsum(3);
    // decay[i, j] = exp(gamma_i - gamma_j) for j <= i, else 0.
    let future = upper_mask(c, false, &device).reshape([1, 1, 1, c, c]).expand([b, h, n, c, c]);
    let diff = gamma.clone().unsqueeze_dim::<5>(4) - gamma.clone().unsqueeze_dim::<5>(3);
    let decay = diff.mask_fill(future, f32::NEG_INFINITY).exp();
    // u = (I + A)^-1 (beta v - beta Gamma K S0) with A[i, j] = beta_i decay[i, j] k_i.k_j, j < i.
    let kb = k.clone() * beta.clone();
    let not_below = upper_mask(c, true, &device).reshape([1, 1, 1, c, c]).expand([b, h, n, c, c]);
    let neg_a = (kb.clone().matmul(k.clone().swap_dims(3, 4)) * decay.clone())
        .mask_fill(not_below, 0.0)
        .neg();
    // A is strictly lower triangular, so (I + A)^-1 = sum_m (-A)^m
    //   = (I - A)(I + A^2)(I + A^4)...: exact once the powers reach C.
    let eye = Tensor::<2>::eye(c, &device).reshape([1, 1, 1, c, c]);
    let mut inverse = neg_a.clone() + eye;
    let mut power = neg_a;
    let mut covered = 2;
    while covered < c {
        power = power.clone().matmul(power);
        inverse = inverse.clone() + inverse.matmul(power.clone());
        covered *= 2;
    }
    let gamma_exp = gamma.clone().exp().unsqueeze_dim::<5>(4);
    let values = inverse.clone().matmul(v * beta);
    let w = inverse.matmul(kb * gamma_exp.clone());
    let qk = q.clone().matmul(k.clone().swap_dims(3, 4)) * decay;
    let qg = q * gamma_exp;
    let last = gamma.clone().narrow(3, c - 1, 1);
    let k_out = k * (last.clone() - gamma).exp().unsqueeze_dim::<5>(4);
    let carry = last.exp();

    let take = |x: &Tensor<5>, i: usize| {
        let [_, _, _, rows, cols] = x.dims();
        x.clone().narrow(2, i, 1).reshape([b, h, rows, cols])
    };
    let mut state = state;
    let mut outs = Vec::with_capacity(n);
    for i in 0..n {
        let fresh = take(&values, i) - take(&w, i).matmul(state.clone());
        outs.push(take(&qg, i).matmul(state.clone()) + take(&qk, i).matmul(fresh.clone()));
        let keep = carry.clone().narrow(2, i, 1).reshape([b, h, 1, 1]);
        state = state * keep + take(&k_out, i).swap_dims(2, 3).matmul(fresh);
    }
    let out = Tensor::cat(outs, 2);
    let out = if pad > 0 { out.narrow(2, 0, t) } else { out };
    (out, state)
}

/// Decode state of one DeltaNet layer: the memory and the convolution's last inputs.
pub struct DeltaCache {
    state: Tensor<4>,
    conv: Tensor<3>,
    length: usize,
    itemsize: usize,
}

impl DeltaCache {
    fn new(state: [usize; 4], conv: [usize; 3], device: &Device) -> Self {
        let state = Tensor::zeros(state, device);
        let itemsize = state.dtype().size();
        Self { state, conv: Tensor::zeros(conv, device), length: 0, itemsize }
    }

    /// Next absolute position.
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn reset(&mut self) {
        let device = self.state.device();
        self.state = Tensor::zeros(self.state.dims(), &device);
        self.conv = Tensor::zeros(self.conv.dims(), &device);
        self.length = 0;
    }

    pub fn nbytes(&self) -> usize {
        let count = |dims: &[usize]| dims.iter().product::<usize>();
        (count(&self.state.dims()) + count(&self.conv.dims())) * self.itemsize
    }
}

/// Shape of a gated DeltaNet layer.
#[derive(Debug, Clone)]
pub struct DeltaSpec {
    pub d: usize,
    pub head_dim: usize,
    pub conv: usize,
    pub chunk: usize,
    pub eps: f64,
}

impl DeltaSpec {
    pub fn heads(&self) -> usize {
        assert!(
            self.d.is_multiple_of(self.head_dim),
            "DeltaNet width {} is not a multiple of dn_head_dim {}",
            self.d,
            self.head_dim
        );
        self.d / self.head_dim
    }

    /// Channels of the short convolution: q, k and v.
    pub fn conv_channels(&self) -> usize {
        3 * self.heads() * self.head_dim
    }

    /// Bytes of one layer's decode state.
    pub fn cache_bytes(&self, itemsize: usize, batch: usize) -> usize {
        let h = self.heads();
        batch
            * (h * self.head_dim * self.head_dim + (self.conv - 1) * self.conv_channels())
            * itemsize
    }
}

/// Gated DeltaNet layer: input projection, short causal convolution with SiLU, L2-normalised
/// queries and keys (the DeltaNet form of QK-norm), the gated delta rule, then a per-head
/// RMSNorm, a sigmoid output gate and the output projection.
#[derive(Module, Debug)]
pub struct GatedDeltaNet {
    /// `[q | k | v | gate | a | b]`: q, k, v and gate are `heads * head_dim` wide, a and b
    /// (decay and write-strength logits) one per head.
    proj: Linear,
    /// Depthwise causal convolution weights `[conv, channels]`; the last row multiplies the
    /// current token.
    conv: Param<Tensor<2>>,
    /// Decay rate per head (Mamba-2 style): `g = -exp(a_log) * softplus(a + dt_bias)`.
    a_log: Param<Tensor<1>>,
    dt_bias: Param<Tensor<1>>,
    norm: RmsNorm,
    o: Linear,
    heads: usize,
    head_dim: usize,
    conv_size: usize,
    chunk: usize,
}

impl GatedDeltaNet {
    pub fn new(spec: &DeltaSpec, device: &Device) -> Self {
        assert!(spec.conv >= 2, "the short convolution needs a width of at least 2");
        let (h, hd) = (spec.heads(), spec.head_dim);
        let channels = spec.conv_channels();
        // Decay rates spread over [1, 16] and time steps over [0.001, 0.1] across heads, as the
        // Mamba-2 / Gated DeltaNet initialisations draw them, but deterministic.
        let spread = |i: usize| if h > 1 { i as f64 / (h - 1) as f64 } else { 0.0 };
        let a_log: Vec<f32> = (0..h).map(|i| (1.0 + 15.0 * spread(i)).ln() as f32).collect();
        let dt_bias: Vec<f32> = (0..h)
            .map(|i| {
                let dt = (0.001f64.ln() + spread(i) * (0.1f64.ln() - 0.001f64.ln())).exp();
                (dt + (-(-dt).exp_m1()).ln()) as f32 // inverse softplus
            })
            .collect();
        let bound = 1.0 / (spec.conv as f64).sqrt();
        Self {
            proj: linear(spec.d, 4 * h * hd + 2 * h, device),
            conv: Initializer::Uniform { min: -bound, max: bound }
                .init([spec.conv, channels], device),
            a_log: Param::from_tensor(Tensor::from_data(TensorData::new(a_log, [h]), device)),
            dt_bias: Param::from_tensor(Tensor::from_data(TensorData::new(dt_bias, [h]), device)),
            norm: rms_norm(hd, spec.eps, device),
            o: linear(h * hd, spec.d, device),
            heads: h,
            head_dim: hd,
            conv_size: spec.conv,
            chunk: spec.chunk,
        }
    }

    pub fn make_cache(&self, batch: usize, device: &Device) -> DeltaCache {
        let (h, hd) = (self.heads, self.head_dim);
        DeltaCache::new([batch, h, hd, hd], [batch, self.conv_size - 1, 3 * h * hd], device)
    }

    /// Causal depthwise convolution of `x` `[B, T, C]` continuing from the previous inputs
    /// `prev` `[B, conv - 1, C]`; returns the output and the new previous inputs.
    fn short_conv(&self, x: Tensor<3>, prev: Tensor<3>) -> (Tensor<3>, Tensor<3>) {
        let t = x.dims()[1];
        let width = self.conv_size;
        let padded = Tensor::cat(vec![prev, x], 1);
        let w = self.conv.val().unsqueeze_dim::<3>(0); // [1, conv, C]
        let out = if t == 1 {
            (padded.clone() * w).sum_dim(1)
        } else {
            (0..width)
                .map(|j| padded.clone().narrow(1, j, t) * w.clone().narrow(1, j, 1))
                .reduce(|a, b| a + b)
                .expect("conv width >= 2")
        };
        (out, padded.narrow(1, t, width - 1))
    }

    /// `x` `[B, T, d]`. With a cache the layer continues from its state (one-token steps use
    /// the recurrent form) and updates it.
    pub fn forward(&self, x: Tensor<3>, cache: Option<&mut DeltaCache>) -> Tensor<3> {
        let [b, t, _] = x.dims();
        let (h, hd) = (self.heads, self.head_dim);
        let device = x.device();
        let p = self.proj.forward(x);
        let mixed = p.clone().narrow(2, 0, 3 * h * hd);
        let gate = p.clone().narrow(2, 3 * h * hd, h * hd);
        let a = p.clone().narrow(2, 4 * h * hd, h);
        let write = p.narrow(2, 4 * h * hd + h, h);
        let (prev, state) = match &cache {
            Some(c) => (c.conv.clone(), c.state.clone()),
            None => (
                Tensor::zeros([b, self.conv_size - 1, 3 * h * hd], &device),
                Tensor::zeros([b, h, hd, hd], &device),
            ),
        };
        let (mixed, conv_state) = self.short_conv(mixed, prev);
        let mixed = silu(mixed);
        let heads = |x: Tensor<3>| x.reshape([b, t, h, hd]).swap_dims(1, 2);
        let q = l2_normalize(heads(mixed.clone().narrow(2, 0, h * hd)))
            .mul_scalar(1.0 / (hd as f64).sqrt());
        let k = l2_normalize(heads(mixed.clone().narrow(2, h * hd, h * hd)));
        let v = heads(mixed.narrow(2, 2 * h * hd, h * hd));
        let beta = sigmoid(write).swap_dims(1, 2);
        let rate = self.a_log.val().exp().reshape([1, 1, h]);
        let g =
            (softplus(a + self.dt_bias.val().reshape([1, 1, h]), 1.0) * rate).neg().swap_dims(1, 2);
        let (out, state) = if t == 1 {
            delta_rule_step(q, k, v, g, beta, state)
        } else {
            delta_rule_chunked(q, k, v, g, beta, state, self.chunk)
        };
        if let Some(cache) = cache {
            cache.state = state;
            cache.conv = conv_state;
            cache.length += t;
        }
        let out = self.norm.forward(out).swap_dims(1, 2).reshape([b, t, h * hd]);
        self.o.forward(out * sigmoid(gate))
    }
}
