//! Layer builders: each mirrors the Burn reference (`ncm-model`) op for op, on static shapes
//! with every cache and recurrent state an explicit graph input and output (no `Loop`, `If`,
//! `ScatterND` or stateful variables: the host writes new cache entries into the rings).
//!
//! Shapes are batch 1. Decode builders take `T = 1`; the prefill ones take the bucket length.

use crate::ExportError;
use crate::graph::{Graph, Val, WeightPool};
use crate::onnx::{DType, HostTensor};
use crate::quant::{WeightForm, weight_val};
use crate::weights::F32Tensor;

/// L2 floor of the DeltaNet's query and key normalisation (`delta.rs`).
const L2_EPS: f32 = 1e-6;

/// A graph and the weight pool it draws on.
pub struct Cx<'a> {
    pub g: &'a mut Graph,
    pub pool: &'a mut WeightPool,
    pub form: WeightForm,
}

impl Cx<'_> {
    fn f32_weight(&mut self, key: &str, t: &F32Tensor) -> Val {
        self.g.weight(self.pool, key, HostTensor::f32(&t.shape, &t.data))
    }

    /// `x W` for a bias-free linear layer of Burn's layout, quantised per the export's form.
    pub fn linear(&mut self, key: &str, x: &Val, w: &F32Tensor) -> Result<Val, ExportError> {
        let [d_in, d_out] = match w.shape[..] {
            [a, b] => [a, b],
            _ => return Err(ExportError::Shape(format!("`{key}` is not a matrix: {:?}", w.shape))),
        };
        let wv = weight_val(self.g, self.pool, key, &w.data, d_in, d_out, self.form)
            .map_err(ExportError::Quant)?;
        Ok(self.g.matmul(x, &wv))
    }

    /// `x / sqrt(mean(x^2) + eps) * gamma` over the last axis (Burn's `RmsNorm`).
    pub fn rms_norm(&mut self, key: &str, x: &Val, gamma: &F32Tensor, eps: f64) -> Val {
        let gamma = self.f32_weight(key, gamma);
        let sq = self.g.mul(x, x);
        let ms = self.g.reduce_mean(&sq, &[-1], true);
        let ms = self.g.add_scalar(&ms, eps as f32);
        let root = self.g.sqrt(&ms);
        let inv = self.g.reciprocal(&root);
        let normed = self.g.mul(x, &inv);
        self.g.mul(&normed, &gamma)
    }

    /// The SwiGLU MLP: `down(silu(gate) * up)` with `gate_up` giving `[gate | up]`.
    pub fn mlp(
        &mut self,
        key: &str,
        x: &Val,
        gate_up: &F32Tensor,
        down: &F32Tensor,
        ff: usize,
    ) -> Result<Val, ExportError> {
        let both = self.linear(&format!("{key}.gate_up.weight"), x, gate_up)?;
        let gate = self.g.slice(&both, -1, 0, ff as i64);
        let up = self.g.slice(&both, -1, ff as i64, 2 * ff as i64);
        let act = self.g.silu(&gate);
        let act = self.g.mul(&act, &up);
        self.linear(&format!("{key}.down.weight"), &act, down)
    }

    /// RoPE of `x` `[1, H, T, D]` by the tables `cos`, `sin` `[1, 1, T, D]`:
    /// `x R * sin + x * cos`, with Burn's rotation matrix `R`.
    pub fn rope(&mut self, x: &Val, cos: &Val, sin: &Val, head_dim: usize) -> Val {
        let half = head_dim / 2;
        let mut r = vec![0f32; head_dim * head_dim];
        for j in 0..half {
            r[(j + half) * head_dim + j] = -1.0;
            r[j * head_dim + j + half] = 1.0;
        }
        let rot = self.g.const_f32(&[head_dim, head_dim], &r);
        let rotated = self.g.matmul(x, &rot);
        let a = self.g.mul(&rotated, sin);
        let b = self.g.mul(x, cos);
        self.g.add(&a, &b)
    }

    /// `x / sqrt(sum(x^2, last axis) + 1e-6)` for rank-4 `x`.
    fn l2_normalize(&mut self, x: &Val) -> Val {
        let sq = self.g.mul(x, x);
        let ss = self.g.reduce_sum(&sq, &[3], true);
        let ss = self.g.add_scalar(&ss, L2_EPS);
        let norm = self.g.sqrt(&ss);
        self.g.div(x, &norm)
    }

    /// The identity `[c, c]`.
    fn eye(&mut self, c: usize) -> Val {
        let mut e = vec![0f32; c * c];
        for i in 0..c {
            e[i * c + i] = 1.0;
        }
        self.g.const_f32(&[c, c], &e)
    }

    /// `(I + A)^-1` for the strictly lower-triangular `A` whose negation is `neg_a`
    /// (`[..., c, c]`, all in f32): `(I + N)(I + N^2)(I + N^4)...`, exact once the powers reach
    /// `c`. For `c = 64` that is 5 squarings and 5 multiplies, 10 matmuls of 64 x 64
    /// (`delta.rs`'s `delta_rule_chunked`; the algebra must stay f32).
    pub fn delta_inverse(&mut self, neg_a: &Val, c: usize) -> Val {
        let eye = self.eye(c);
        let mut inverse = self.g.add(neg_a, &eye);
        let mut power = neg_a.clone();
        let mut covered = 2;
        while covered < c {
            power = self.g.matmul(&power, &power);
            let step = self.g.matmul(&inverse, &power);
            inverse = self.g.add(&inverse, &step);
            covered *= 2;
        }
        inverse
    }
}

// ---------------------------------------------------------------------- gated DeltaNet

/// A gated DeltaNet layer's shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeltaDims {
    pub d: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub conv: usize,
    pub chunk: usize,
    pub eps: f64,
}

impl DeltaDims {
    pub fn channels(&self) -> usize {
        3 * self.heads * self.head_dim
    }

    pub fn proj_width(&self) -> usize {
        4 * self.heads * self.head_dim + 2 * self.heads
    }
}

/// A gated DeltaNet layer's weights (`proj` and `o` in Burn's `[d_in, d_out]` layout).
pub struct DeltaWeights {
    pub proj: F32Tensor,
    /// `[conv, channels]`.
    pub conv: F32Tensor,
    pub a_log: F32Tensor,
    pub dt_bias: F32Tensor,
    pub norm: F32Tensor,
    pub o: F32Tensor,
}

/// What a DeltaNet layer returns.
pub struct DeltaOut {
    pub y: Val,
    pub conv: Val,
    pub state: Val,
}

impl Cx<'_> {
    /// The projections shared by the step and the chunked form, up to the per-head q, k, v, the
    /// log decay `g`, the write strength `beta` and the output gate. `x` is `[1, T, d]`;
    /// returns (q `[1,H,T,dk]`, k, v, g `[1,H,T]`, beta `[1,H,T]`, gate `[1,T,H*hd]`, the new
    /// convolution inputs).
    #[allow(clippy::type_complexity)]
    fn delta_inputs(
        &mut self,
        key: &str,
        dm: &DeltaDims,
        w: &DeltaWeights,
        x: &Val,
        conv_prev: &Val,
        t: usize,
    ) -> Result<(Val, Val, Val, Val, Val, Val, Val), ExportError> {
        let (h, hd) = (dm.heads, dm.head_dim);
        let hh = (h * hd) as i64;
        let p = self.linear(&format!("{key}.proj.weight"), x, &w.proj)?;
        let mixed = self.g.slice(&p, 2, 0, 3 * hh);
        let gate = self.g.slice(&p, 2, 3 * hh, 4 * hh);
        let a = self.g.slice(&p, 2, 4 * hh, 4 * hh + h as i64);
        let write = self.g.slice(&p, 2, 4 * hh + h as i64, 4 * hh + 2 * h as i64);
        // The short causal convolution continuing from the previous inputs.
        let padded = self.g.concat(&[conv_prev, &mixed], 1);
        let conv_w = self.f32_weight(&format!("{key}.conv"), &w.conv);
        let conv_w = self.g.reshape(&conv_w, &[1, dm.conv as i64, dm.channels() as i64]);
        let conv_out = if t == 1 {
            let prod = self.g.mul(&padded, &conv_w);
            self.g.reduce_sum(&prod, &[1], true)
        } else {
            let mut acc: Option<Val> = None;
            for j in 0..dm.conv {
                let window = self.g.slice(&padded, 1, j as i64, (j + t) as i64);
                let tap = self.g.slice(&conv_w, 1, j as i64, j as i64 + 1);
                let term = self.g.mul(&window, &tap);
                acc = Some(match acc {
                    Some(a) => self.g.add(&a, &term),
                    None => term,
                });
            }
            acc.expect("a convolution of width >= 2")
        };
        let new_conv = self.g.slice(&padded, 1, t as i64, (t + dm.conv - 1) as i64);
        let mixed = self.g.silu(&conv_out);
        let heads = |cx: &mut Self, lo: i64, hi: i64| {
            let part = cx.g.slice(&mixed, 2, lo, hi);
            let part = cx.g.reshape(&part, &[1, t as i64, h as i64, hd as i64]);
            cx.g.transpose(&part, &[0, 2, 1, 3])
        };
        let q = heads(self, 0, hh);
        let q = self.l2_normalize(&q);
        let q = self.g.mul_scalar(&q, 1.0 / (hd as f32).sqrt());
        let k = heads(self, hh, 2 * hh);
        let k = self.l2_normalize(&k);
        let v = heads(self, 2 * hh, 3 * hh);
        let beta = self.g.sigmoid(&write);
        let beta = self.g.transpose(&beta, &[0, 2, 1]);
        // g = -softplus(a + dt_bias) * exp(a_log)
        let a_log = self.f32_weight(&format!("{key}.a_log"), &w.a_log);
        let dt_bias = self.f32_weight(&format!("{key}.dt_bias"), &w.dt_bias);
        let rate = self.g.exp(&a_log);
        let rate = self.g.reshape(&rate, &[1, 1, h as i64]);
        let dt_bias = self.g.reshape(&dt_bias, &[1, 1, h as i64]);
        let a = self.g.add(&a, &dt_bias);
        let a = self.g.softplus(&a);
        let g = self.g.mul(&a, &rate);
        let g = self.g.neg(&g);
        let g = self.g.transpose(&g, &[0, 2, 1]);
        let gate = self.g.sigmoid(&gate);
        Ok((q, k, v, g, beta, gate, new_conv))
    }

    /// The layer's tail: the per-head RMS norm, merged heads, the sigmoid gate and `o`.
    fn delta_output(
        &mut self,
        key: &str,
        dm: &DeltaDims,
        w: &DeltaWeights,
        out: &Val,
        gate: &Val,
        t: usize,
    ) -> Result<Val, ExportError> {
        let (h, hd) = (dm.heads, dm.head_dim);
        let normed = self.rms_norm(&format!("{key}.norm.gamma"), out, &w.norm, dm.eps);
        let merged = self.g.transpose(&normed, &[0, 2, 1, 3]);
        let merged = self.g.reshape(&merged, &[1, t as i64, (h * hd) as i64]);
        let gated = self.g.mul(&merged, gate);
        self.linear(&format!("{key}.o.weight"), &gated, &w.o)
    }

    /// One recurrent decode step: `x` `[1, 1, d]`, the convolution's last inputs
    /// `[1, conv - 1, C]` and the state `[1, H, dk, dv]` (f32, host-owned).
    pub fn delta_step(
        &mut self,
        key: &str,
        dm: &DeltaDims,
        w: &DeltaWeights,
        x: &Val,
        conv_prev: &Val,
        state: &Val,
    ) -> Result<DeltaOut, ExportError> {
        let h = dm.heads as i64;
        let (q, k, v, g, beta, gate, conv) = self.delta_inputs(key, dm, w, x, conv_prev, 1)?;
        let decay = self.g.reshape(&g, &[1, h, 1, 1]);
        let decay = self.g.exp(&decay);
        let beta = self.g.reshape(&beta, &[1, h, 1, 1]);
        let state = self.g.mul(state, &decay);
        let read = self.g.matmul(&k, &state);
        let u = self.g.sub(&v, &read);
        let u = self.g.mul(&u, &beta);
        let kt = self.g.transpose(&k, &[0, 1, 3, 2]);
        let write = self.g.matmul(&kt, &u);
        let state = self.g.add(&state, &write);
        let out = self.g.matmul(&q, &state);
        let y = self.delta_output(key, dm, w, &out, &gate, 1)?;
        Ok(DeltaOut { y, conv, state })
    }

    /// The chunked form over `t` tokens (a multiple of the chunk length): the prefill path,
    /// equal to the recurrence up to rounding. The triangular inverse runs in the graph (f32).
    #[allow(clippy::too_many_arguments)]
    pub fn delta_chunked(
        &mut self,
        key: &str,
        dm: &DeltaDims,
        w: &DeltaWeights,
        x: &Val,
        conv_prev: &Val,
        state: &Val,
        t: usize,
    ) -> Result<DeltaOut, ExportError> {
        let (h, hd, c) = (dm.heads as i64, dm.head_dim as i64, dm.chunk);
        if !t.is_multiple_of(c) {
            return Err(ExportError::Shape(format!(
                "the DeltaNet prefill bucket {t} is not a multiple of the chunk {c}"
            )));
        }
        let n = (t / c) as i64;
        let ci = c as i64;
        let (q, k, v, g, beta, gate, conv) = self.delta_inputs(key, dm, w, x, conv_prev, t)?;
        let q5 = self.g.reshape(&q, &[1, h, n, ci, hd]);
        let k5 = self.g.reshape(&k, &[1, h, n, ci, hd]);
        let v5 = self.g.reshape(&v, &[1, h, n, ci, hd]);
        let beta5 = self.g.reshape(&beta, &[1, h, n, ci, 1]);
        // Cumulative log decay inside each chunk, and decay[i, j] = exp(gamma_i - gamma_j) for
        // j <= i, else 0.
        let g4 = self.g.reshape(&g, &[1, h, n, ci]);
        let axis = self.g.constant("axis", HostTensor::i64(&[], &[3]));
        let gamma = self.g.op("CumSum", &[&g4, &axis], &[]);
        let gamma_i = self.g.reshape(&gamma, &[1, h, n, ci, 1]);
        let gamma_j = self.g.reshape(&gamma, &[1, h, n, 1, ci]);
        let diff = self.g.sub(&gamma_i, &gamma_j);
        let future = self.mask(c, true);
        let neg_inf = self.g.scalar(f32::NEG_INFINITY);
        let masked = self.g.op("Where", &[&future, &neg_inf, &diff], &[]);
        let decay = self.g.exp(&masked);
        // u = (I + A)^-1 (beta v - beta Gamma K S0), A[i, j] = beta_i decay[i, j] k_i . k_j, j < i.
        let kb = self.g.mul(&k5, &beta5);
        let kt = self.g.transpose(&k5, &[0, 1, 2, 4, 3]);
        let a = self.g.matmul(&kb, &kt);
        let a = self.g.mul(&a, &decay);
        let not_below = self.mask(c, false);
        let zero = self.g.scalar(0.0);
        let a = self.g.op("Where", &[&not_below, &zero, &a], &[]);
        let neg_a = self.g.neg(&a);
        let inverse = self.delta_inverse(&neg_a, c);
        let gexp = self.g.exp(&gamma_i);
        let vb = self.g.mul(&v5, &beta5);
        let values = self.g.matmul(&inverse, &vb);
        let kg = self.g.mul(&kb, &gexp);
        let wv = self.g.matmul(&inverse, &kg);
        let qk = self.g.matmul(&q5, &kt);
        let qk = self.g.mul(&qk, &decay);
        let qg = self.g.mul(&q5, &gexp);
        let last = self.g.slice(&gamma, 3, ci - 1, ci);
        let last = self.g.reshape(&last, &[1, h, n, 1, 1]);
        let to_end = self.g.sub(&last, &gamma_i);
        let to_end = self.g.exp(&to_end);
        let k_out = self.g.mul(&k5, &to_end);
        let carry = self.g.exp(&last);
        let mut state = state.clone();
        let mut outs = Vec::new();
        for i in 0..n {
            let take = |cx: &mut Self, x: &Val, cols: i64| {
                let s = cx.g.slice(x, 2, i, i + 1);
                cx.g.reshape(&s, &[1, h, ci, cols])
            };
            let values_i = take(self, &values, hd);
            let w_i = take(self, &wv, hd);
            let qg_i = take(self, &qg, hd);
            let qk_i = take(self, &qk, ci);
            let k_out_i = take(self, &k_out, hd);
            let ws = self.g.matmul(&w_i, &state);
            let fresh = self.g.sub(&values_i, &ws);
            let from_state = self.g.matmul(&qg_i, &state);
            let within = self.g.matmul(&qk_i, &fresh);
            outs.push(self.g.add(&from_state, &within));
            let keep = self.g.slice(&carry, 2, i, i + 1);
            let keep = self.g.reshape(&keep, &[1, h, 1, 1]);
            let kept = self.g.mul(&state, &keep);
            let kt_i = self.g.transpose(&k_out_i, &[0, 1, 3, 2]);
            let add = self.g.matmul(&kt_i, &fresh);
            state = self.g.add(&kept, &add);
        }
        let refs: Vec<&Val> = outs.iter().collect();
        let out = self.g.concat(&refs, 2);
        let y = self.delta_output(key, dm, w, &out, &gate, t)?;
        Ok(DeltaOut { y, conv, state })
    }

    /// A `[c, c]` boolean mask: true where the column is after the row (`strict`), or at or
    /// after it.
    fn mask(&mut self, c: usize, strict: bool) -> Val {
        let data: Vec<u8> = (0..c * c)
            .map(|n| {
                let (i, j) = (n / c, n % c);
                u8::from(j > i || (!strict && j == i))
            })
            .collect();
        self.g.constant(
            "mask",
            HostTensor { dtype: DType::Bool, dims: vec![c as i64, c as i64], bytes: data },
        )
    }
}

// ---------------------------------------------------------------------- GQA attention

/// A grouped-query attention layer's shape (the fallback design: no gate, no QK-norm).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttnDims {
    pub d: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

/// The v2 design's additions to an attention layer: a sigmoid output gate (`gate`, `[d, H*D]`)
/// and QK-norm (an RMSNorm of each query and key head over `D` before RoPE). UNVERIFIED: written
/// from `layers::Attention::{input, project_from}` without a run.
pub struct AttnExtras<'a> {
    pub gate: &'a F32Tensor,
    pub q_norm: &'a F32Tensor,
    pub k_norm: &'a F32Tensor,
    pub eps: f64,
}

/// What an attention layer returns: its output and the new keys and values for the host to
/// write into its ring.
pub struct AttnOut {
    pub y: Val,
    /// `[1, KVH, T, D]`, RoPE applied.
    pub k_new: Val,
    pub v_new: Val,
}

impl Cx<'_> {
    /// Causal GQA attention of `x` `[1, T, d]` over a static ring and the new positions: keys
    /// `ring_k` `[1, KVH, S, D]` (and values) followed by the `T` new ones; `bias` `[T, S + T]`
    /// is additive (0 visible, -inf hidden). Mirrors `layers::attend` (query heads of a group
    /// folded into the query axis).
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &mut self,
        key: &str,
        dm: &AttnDims,
        qkv: &F32Tensor,
        o: &F32Tensor,
        x: &Val,
        cos: &Val,
        sin: &Val,
        ring_k: &Val,
        ring_v: &Val,
        bias: &Val,
        t: usize,
        extras: Option<&AttnExtras>,
    ) -> Result<AttnOut, ExportError> {
        let (h, kvh, hd) = (dm.heads as i64, dm.kv_heads as i64, dm.head_dim as i64);
        let groups = h / kvh;
        let ti = t as i64;
        let proj = self.linear(&format!("{key}.qkv.weight"), x, qkv)?;
        let heads = |cx: &mut Self, lo: i64, n: i64| {
            let part = cx.g.slice(&proj, 2, lo * hd, (lo + n) * hd);
            let part = cx.g.reshape(&part, &[1, ti, n, hd]);
            cx.g.transpose(&part, &[0, 2, 1, 3])
        };
        let q = heads(self, 0, h);
        let k = heads(self, h, kvh);
        let v = heads(self, h + kvh, kvh);
        // v2: QK-norm of each head row, then RoPE (`norm_rope`); the gate is computed from `x`.
        let (q, k) = match extras {
            Some(e) => (
                self.rms_norm(&format!("{key}.q_norm.gamma"), &q, e.q_norm, e.eps),
                self.rms_norm(&format!("{key}.k_norm.gamma"), &k, e.k_norm, e.eps),
            ),
            None => (q, k),
        };
        let gate = match extras {
            Some(e) => {
                let g = self.linear(&format!("{key}.gate.weight"), x, e.gate)?;
                Some(self.g.sigmoid(&g))
            }
            None => None,
        };
        let q = self.rope(&q, cos, sin, dm.head_dim);
        let k = self.rope(&k, cos, sin, dm.head_dim);
        let keys = self.g.concat(&[ring_k, &k], 2);
        let values = self.g.concat(&[ring_v, &v], 2);
        let q = self.g.reshape(&q, &[1, kvh, groups * ti, hd]);
        let kt = self.g.transpose(&keys, &[0, 1, 3, 2]);
        let scores = self.g.matmul(&q, &kt);
        let scores = self.g.mul_scalar(&scores, 1.0 / (dm.head_dim as f32).sqrt());
        let tiled = if groups > 1 {
            let reps = self.g.const_i64(&[groups, 1]);
            self.g.op("Tile", &[bias, &reps], &[])
        } else {
            bias.clone()
        };
        let scores = self.g.add(&scores, &tiled);
        let probs = self.g.softmax(&scores, 3);
        let out = self.g.matmul(&probs, &values);
        let out = self.g.reshape(&out, &[1, h, ti, hd]);
        let merged = self.g.transpose(&out, &[0, 2, 1, 3]);
        let merged = self.g.reshape(&merged, &[1, ti, h * hd]);
        let merged = match &gate {
            Some(g) => self.g.mul(&merged, g),
            None => merged,
        };
        let y = self.linear(&format!("{key}.o.weight"), &merged, o)?;
        Ok(AttnOut { y, k_new: k, v_new: v })
    }
}

// ---------------------------------------------------------------------- sparse retrieval (dense form)

/// A sparse-retrieval layer's shape (`ncm_model::retrieval::RetrievalSpec`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetDims {
    pub d: usize,
    pub heads: usize,
    pub nope_dim: usize,
    pub rope_dim: usize,
    pub v_dim: usize,
    pub latent: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub eps: f64,
}

impl RetDims {
    /// Width of the layer input projection: `[q | latent, RoPE key | gate | indexer q |
    /// indexer k | indexer weights]`.
    pub fn proj_width(&self) -> usize {
        self.heads * (self.nope_dim + self.rope_dim)
            + self.latent
            + self.rope_dim
            + self.heads * self.v_dim
            + self.index_heads * self.index_dim
            + self.index_dim
            + self.index_heads
    }

    /// Width of a latent cache entry: the latent and the RoPE key.
    pub fn entry_width(&self) -> usize {
        self.latent + self.rope_dim
    }
}

/// A retrieval layer's weights: `proj`, `o` (quantisable), `uk` `[l, H*dn]` and `uv`
/// `[l, H*dv]` (dense), and the four norms' gammas.
pub struct RetWeights {
    pub proj: F32Tensor,
    pub q_norm: F32Tensor,
    pub kv_norm: F32Tensor,
    pub rope_norm: F32Tensor,
    pub index_norm: F32Tensor,
    pub uk: F32Tensor,
    pub uv: F32Tensor,
    pub o: F32Tensor,
}

/// The part of a retrieval layer before its attention: everything the projection gives.
pub struct RetPre {
    /// The absorbed queries `[1, H, T, latent + rope_dim]`.
    pub q_all: Val,
    /// The sigmoid output gate `[1, T, H * v_dim]`.
    pub gate: Val,
    /// The new latent-cache entries `[1, T, latent + rope_dim]` for the host to store.
    pub entry: Val,
    /// The new positions' normed indexer keys `[1, T, index_dim]` (the host keeps them, 8-bit,
    /// for the split sparse path).
    pub index_key: Val,
    /// The indexer queries `[1, T, index_heads * index_dim]` and head weights
    /// `[1, T, index_heads]`, for the split sparse path.
    pub index_q: Val,
    pub index_w: Val,
}

/// What a dense retrieval layer returns.
pub struct RetOut {
    pub y: Val,
    pub pre: RetPre,
}

impl Cx<'_> {
    /// A retrieval layer up to its attention (`SparseRetrieval::core_projected`): the
    /// projection, the four norms, the decoupled RoPE, the absorbed queries (the key
    /// up-projection folded in) and the cache entry. `cos`/`sin` `[1, 1, T, rope_dim]` are the
    /// layer YaRN table.
    #[allow(clippy::too_many_arguments)]
    pub fn retrieval_pre(
        &mut self,
        key: &str,
        dm: &RetDims,
        w: &RetWeights,
        x: &Val,
        cos: &Val,
        sin: &Val,
        t: usize,
    ) -> Result<RetPre, ExportError> {
        let (h, dn, dr, dv, l) = (
            dm.heads as i64,
            dm.nope_dim as i64,
            dm.rope_dim as i64,
            dm.v_dim as i64,
            dm.latent as i64,
        );
        let (hi, di) = (dm.index_heads as i64, dm.index_dim as i64);
        let ti = t as i64;
        let qd = dn + dr;
        let p = self.linear(&format!("{key}.proj.weight"), x, &w.proj)?;
        let mut offset = 0i64;
        let mut take = |cx: &mut Self, n: i64| {
            let part = cx.g.slice(&p, 2, offset, offset + n);
            offset += n;
            part
        };
        let q = take(self, h * qd);
        let ckv = take(self, l + dr);
        let gate = take(self, h * dv);
        let index_q = take(self, hi * di);
        let index_k = take(self, di);
        let index_w = take(self, hi);
        // Queries: per-head RMS norm over (nope | rope), the rope part rotated.
        let q = self.g.reshape(&q, &[1, ti, h, qd]);
        let q = self.rms_norm(&format!("{key}.q_norm.gamma"), &q, &w.q_norm, dm.eps);
        let q = self.g.transpose(&q, &[0, 2, 1, 3]);
        let q_nope = self.g.slice(&q, 3, 0, dn);
        let q_rope = self.g.slice(&q, 3, dn, qd);
        let q_rope = self.rope(&q_rope, cos, sin, dm.rope_dim);
        // The cache entry: normed latent and the normed, rotated RoPE key.
        let latent = self.g.slice(&ckv, 2, 0, l);
        let latent = self.rms_norm(&format!("{key}.kv_norm.gamma"), &latent, &w.kv_norm, dm.eps);
        let rope_key = self.g.slice(&ckv, 2, l, l + dr);
        let rope_key =
            self.rms_norm(&format!("{key}.rope_norm.gamma"), &rope_key, &w.rope_norm, dm.eps);
        let rope_key = self.g.reshape(&rope_key, &[1, 1, ti, dr]);
        let rope_key = self.rope(&rope_key, cos, sin, dm.rope_dim);
        let rope_key = self.g.reshape(&rope_key, &[1, ti, dr]);
        let entry = self.g.concat(&[&latent, &rope_key], 2);
        let index_key =
            self.rms_norm(&format!("{key}.index_norm.gamma"), &index_k, &w.index_norm, dm.eps);
        // The key up-projection absorbed into the queries.
        let uk = self.f32_weight(&format!("{key}.uk.weight"), &w.uk);
        let uk = self.g.reshape(&uk, &[l, h, dn]);
        let uk = self.g.transpose(&uk, &[1, 2, 0]);
        let uk = self.g.reshape(&uk, &[1, h, dn, l]);
        let q_lat = self.g.matmul(&q_nope, &uk);
        let q_all = self.g.concat(&[&q_lat, &q_rope], 3);
        let gate = self.g.sigmoid(&gate);
        Ok(RetPre { q_all, gate, entry, index_key, index_q, index_w })
    }

    /// A retrieval layer attention and output projection (`attend_latent`) over `kv`
    /// `[1, N, latent + rope_dim]` (the cache and new entries, or the gathered blocks and the
    /// local window), `bias` `[T, N]` additive: scores over the latents plus RoPE keys, the
    /// weighted latent sum, the value up-projection, the gate, then `o`.
    #[allow(clippy::too_many_arguments)]
    pub fn retrieval_attend(
        &mut self,
        key: &str,
        dm: &RetDims,
        w: &RetWeights,
        pre_q: &Val,
        gate: &Val,
        kv: &Val,
        bias: &Val,
        t: usize,
    ) -> Result<Val, ExportError> {
        let (h, dv, l) = (dm.heads as i64, dm.v_dim as i64, dm.latent as i64);
        let width = (dm.latent + dm.rope_dim) as i64;
        let kv = self.g.reshape(kv, &[1, 1, -1, width]);
        let kvt = self.g.transpose(&kv, &[0, 1, 3, 2]);
        let scores = self.g.matmul(pre_q, &kvt);
        let scale = 1.0 / ((dm.nope_dim + dm.rope_dim) as f32).sqrt();
        let scores = self.g.mul_scalar(&scores, scale);
        let scores = self.g.add(&scores, bias);
        let probs = self.g.softmax(&scores, 3);
        let kv_latent = self.g.slice(&kv, 3, 0, l);
        let mixed = self.g.matmul(&probs, &kv_latent);
        let uv = self.f32_weight(&format!("{key}.uv.weight"), &w.uv);
        let uv = self.g.reshape(&uv, &[l, h, dv]);
        let uv = self.g.transpose(&uv, &[1, 0, 2]);
        let uv = self.g.reshape(&uv, &[1, h, l, dv]);
        let out = self.g.matmul(&mixed, &uv);
        let merged = self.g.transpose(&out, &[0, 2, 1, 3]);
        let merged = self.g.reshape(&merged, &[1, t as i64, h * dv]);
        let gated = self.g.mul(&merged, gate);
        self.linear(&format!("{key}.o.weight"), &gated, &w.o)
    }

    /// One retrieval layer as **dense** attention over its whole latent cache (absorbed MLA),
    /// mirroring `SparseRetrieval::core_projected` and `attend_latent`: [`Self::retrieval_pre`],
    /// then [`Self::retrieval_attend`] over `[cache | entry]`. `cache` is `[1, S, latent +
    /// rope_dim]` (the previous positions), `bias` `[T, S + T]`.
    ///
    /// Exact against the Burn layer while its sparse selection would pick every block
    /// (`positions < local + topk * block`) or when the config is `dense`; beyond that the
    /// split path (host top-k and gather) is the one that matches. UNVERIFIED: written from the
    /// Burn code without a run.
    #[allow(clippy::too_many_arguments)]
    pub fn retrieval_dense(
        &mut self,
        key: &str,
        dm: &RetDims,
        w: &RetWeights,
        x: &Val,
        cos: &Val,
        sin: &Val,
        cache: &Val,
        bias: &Val,
        t: usize,
    ) -> Result<RetOut, ExportError> {
        let pre = self.retrieval_pre(key, dm, w, x, cos, sin, t)?;
        let kv = self.g.concat(&[cache, &pre.entry], 1);
        let y = self.retrieval_attend(key, dm, w, &pre.q_all, &pre.gate, &kv, bias, t)?;
        Ok(RetOut { y, pre })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: &[usize], seed: u64) -> F32Tensor {
        let n: usize = shape.iter().product();
        let mut state = seed | 1;
        let data = (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * 0.1
            })
            .collect();
        F32Tensor { shape: shape.to_vec(), data }
    }

    fn delta_weights(dm: &DeltaDims) -> DeltaWeights {
        DeltaWeights {
            proj: tensor(&[dm.d, dm.proj_width()], 1),
            conv: tensor(&[dm.conv, dm.channels()], 2),
            a_log: tensor(&[dm.heads], 3),
            dt_bias: tensor(&[dm.heads], 4),
            norm: F32Tensor { shape: vec![dm.head_dim], data: vec![1.0; dm.head_dim] },
            o: tensor(&[dm.heads * dm.head_dim, dm.d], 5),
        }
    }

    fn dims() -> DeltaDims {
        DeltaDims { d: 32, heads: 2, head_dim: 16, conv: 4, chunk: 8, eps: 1e-6 }
    }

    #[test]
    fn the_inverse_takes_ten_matmuls_at_chunk_64() {
        let mut pool = WeightPool::default();
        let mut g = Graph::new("inv");
        let neg_a = g.input("neg_a", DType::F32, &[1, 2, 1, 64, 64]);
        let inv =
            Cx { g: &mut g, pool: &mut pool, form: WeightForm::F32 }.delta_inverse(&neg_a, 64);
        g.output("inv", &inv, DType::F32, &[1, 2, 1, 64, 64]);
        g.validate().unwrap();
        assert_eq!(g.op_types().filter(|o| *o == "MatMul").count(), 10);
    }

    #[test]
    fn the_inverse_is_small_for_small_chunks() {
        let mut pool = WeightPool::default();
        let mut g = Graph::new("inv");
        let neg_a = g.input("neg_a", DType::F32, &[1, 1, 1, 8, 8]);
        let inv = Cx { g: &mut g, pool: &mut pool, form: WeightForm::F32 }.delta_inverse(&neg_a, 8);
        g.output("inv", &inv, DType::F32, &[1, 1, 1, 8, 8]);
        // log2(8) - 1 = 2 squarings and 2 multiplies.
        assert_eq!(g.op_types().filter(|o| *o == "MatMul").count(), 4);
    }

    #[test]
    fn a_delta_step_graph_is_well_formed() {
        let dm = dims();
        let w = delta_weights(&dm);
        let mut pool = WeightPool::default();
        let mut g = Graph::new("delta_step");
        let x = g.input("x", DType::F32, &[1, 1, 32]);
        let conv = g.input("conv_in", DType::F32, &[1, 3, 96]);
        let state = g.input("state_in", DType::F32, &[1, 2, 16, 16]);
        let out = Cx { g: &mut g, pool: &mut pool, form: WeightForm::F32 }
            .delta_step("layer", &dm, &w, &x, &conv, &state)
            .unwrap();
        g.output("y", &out.y, DType::F32, &[1, 1, 32]);
        g.output("conv_out", &out.conv, DType::F32, &[1, 3, 96]);
        g.output("state_out", &out.state, DType::F32, &[1, 2, 16, 16]);
        g.validate().unwrap();
        assert!(pool.tensor("layer.proj.weight").is_some());
        assert!(pool.tensor("layer.norm.gamma").is_some());
    }

    #[test]
    fn a_chunked_graph_needs_a_multiple_of_the_chunk() {
        let dm = dims();
        let w = delta_weights(&dm);
        let mut pool = WeightPool::default();
        let mut g = Graph::new("delta_chunk");
        let x = g.input("x", DType::F32, &[1, 12, 32]);
        let conv = g.input("conv_in", DType::F32, &[1, 3, 96]);
        let state = g.input("state_in", DType::F32, &[1, 2, 16, 16]);
        let err = Cx { g: &mut g, pool: &mut pool, form: WeightForm::F32 }
            .delta_chunked("layer", &dm, &w, &x, &conv, &state, 12)
            .err()
            .expect("12 is not a multiple of 8");
        assert!(err.to_string().contains("multiple of the chunk"));
    }

    #[test]
    fn a_chunked_graph_is_well_formed() {
        let dm = dims();
        let w = delta_weights(&dm);
        let mut pool = WeightPool::default();
        let mut g = Graph::new("delta_chunk");
        let x = g.input("x", DType::F32, &[1, 16, 32]);
        let conv = g.input("conv_in", DType::F32, &[1, 3, 96]);
        let state = g.input("state_in", DType::F32, &[1, 2, 16, 16]);
        let out = Cx { g: &mut g, pool: &mut pool, form: WeightForm::F32 }
            .delta_chunked("layer", &dm, &w, &x, &conv, &state, 16)
            .unwrap();
        g.output("y", &out.y, DType::F32, &[1, 16, 32]);
        g.output("state_out", &out.state, DType::F32, &[1, 2, 16, 16]);
        g.validate().unwrap();
        assert!(g.op_types().any(|o| o == "CumSum"));
    }

    #[test]
    fn a_gated_qk_norm_attention_graph_is_well_formed() {
        let dm = AttnDims { d: 32, heads: 4, kv_heads: 2, head_dim: 8 };
        let (gate, qn, kn) = (tensor(&[32, 32], 9), tensor(&[8], 10), tensor(&[8], 11));
        let extras = AttnExtras { gate: &gate, q_norm: &qn, k_norm: &kn, eps: 1e-6 };
        let mut pool = WeightPool::default();
        let mut g = Graph::new("attn_v2");
        let x = g.input("x", DType::F32, &[1, 1, 32]);
        let cos = g.input("cos", DType::F32, &[1, 1, 1, 8]);
        let sin = g.input("sin", DType::F32, &[1, 1, 1, 8]);
        let rk = g.input("ring_k", DType::F32, &[1, 2, 5, 8]);
        let rv = g.input("ring_v", DType::F32, &[1, 2, 5, 8]);
        let bias = g.input("bias", DType::F32, &[1, 6]);
        let out = Cx { g: &mut g, pool: &mut pool, form: WeightForm::F32 }
            .attention(
                "a",
                &dm,
                &tensor(&[32, 64], 7),
                &tensor(&[32, 32], 8),
                &x,
                &cos,
                &sin,
                &rk,
                &rv,
                &bias,
                1,
                Some(&extras),
            )
            .unwrap();
        g.output("y", &out.y, DType::F32, &[1, 1, 32]);
        g.validate().unwrap();
        for k in ["a.gate.weight", "a.q_norm.gamma", "a.k_norm.gamma"] {
            assert!(pool.tensor(k).is_some(), "{k}");
        }
    }

    #[test]
    fn an_attention_graph_is_well_formed() {
        let dm = AttnDims { d: 32, heads: 4, kv_heads: 2, head_dim: 8 };
        let mut pool = WeightPool::default();
        let mut g = Graph::new("attn");
        let x = g.input("x", DType::F32, &[1, 1, 32]);
        let cos = g.input("cos", DType::F32, &[1, 1, 1, 8]);
        let sin = g.input("sin", DType::F32, &[1, 1, 1, 8]);
        let rk = g.input("ring_k", DType::F32, &[1, 2, 5, 8]);
        let rv = g.input("ring_v", DType::F32, &[1, 2, 5, 8]);
        let bias = g.input("bias", DType::F32, &[1, 6]);
        let out = Cx { g: &mut g, pool: &mut pool, form: WeightForm::F32 }
            .attention(
                "a",
                &dm,
                &tensor(&[32, 64], 7),
                &tensor(&[32, 32], 8),
                &x,
                &cos,
                &sin,
                &rk,
                &rv,
                &bias,
                1,
                None,
            )
            .unwrap();
        g.output("y", &out.y, DType::F32, &[1, 1, 32]);
        g.output("k_new", &out.k_new, DType::F32, &[1, 2, 1, 8]);
        g.validate().unwrap();
        assert!(g.op_types().any(|o| o == "Tile"), "two query heads per group tile the bias");
    }

    #[test]
    fn a_dense_retrieval_graph_is_well_formed() {
        let dm = RetDims {
            d: 32,
            heads: 2,
            nope_dim: 8,
            rope_dim: 4,
            v_dim: 8,
            latent: 16,
            index_heads: 2,
            index_dim: 8,
            eps: 1e-6,
        };
        let ones = |n: usize| F32Tensor { shape: vec![n], data: vec![1.0; n] };
        let w = RetWeights {
            proj: tensor(&[32, dm.proj_width()], 1),
            q_norm: ones(12),
            kv_norm: ones(16),
            rope_norm: ones(4),
            index_norm: ones(8),
            uk: tensor(&[16, 16], 2),
            uv: tensor(&[16, 16], 3),
            o: tensor(&[16, 32], 4),
        };
        let mut pool = WeightPool::default();
        let mut g = Graph::new("ret");
        let x = g.input("x", DType::F32, &[1, 2, 32]);
        let cos = g.input("cos", DType::F32, &[1, 1, 2, 4]);
        let sin = g.input("sin", DType::F32, &[1, 1, 2, 4]);
        let cache = g.input("cache", DType::F32, &[1, 6, 20]);
        let bias = g.input("bias", DType::F32, &[2, 8]);
        let out = Cx { g: &mut g, pool: &mut pool, form: WeightForm::F32 }
            .retrieval_dense("r", &dm, &w, &x, &cos, &sin, &cache, &bias, 2)
            .unwrap();
        g.output("y", &out.y, DType::F32, &[1, 2, 32]);
        g.output("entry", &out.pre.entry, DType::F32, &[1, 2, 20]);
        g.output("index_key", &out.pre.index_key, DType::F32, &[1, 2, 8]);
        g.validate().unwrap();
        assert_eq!(dm.entry_width(), 20);
        assert!(pool.tensor("r.uk.weight").is_some());
    }
}
