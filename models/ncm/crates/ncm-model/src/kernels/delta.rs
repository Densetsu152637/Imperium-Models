//! Fused gated DeltaNet step: short convolution, SiLU, L2 norms, gates, the delta-rule update,
//! the read-out, the per-head RMSNorm and the output gate in one kernel, token by token.
//!
//! One cube per (batch, head) with one unit per state column (`head_dim` units). Unit `j` owns
//! column `j` of the head's `[dk, dv]` state and the convolution channels `j` of q, k and v. All
//! arithmetic is in f32; tensors are read and written at the run dtype.

use burn_cubecl::cubecl;
use cubecl::prelude::*;

/// Causal depthwise convolution of one channel at one token, then SiLU. `hist` (contiguous)
/// holds the last `width - 1` inputs of every channel for this batch row (`[width - 1, channels]`
/// from `base`); they are shifted by one and `x` appended.
#[cube]
fn conv_silu<F: Float>(
    x: F,
    conv_w: &Tensor<F>,
    hist: &mut Tensor<F>,
    base: usize,
    channels: usize,
    c: usize,
    #[comptime] width: usize,
) -> f32 {
    let (w0, w1) = (conv_w.stride(0), conv_w.stride(1));
    let mut acc = f32::cast_from(x) * f32::cast_from(conv_w[(width - 1) * w0 + c * w1]);
    for w in 0..width - 1 {
        acc +=
            f32::cast_from(hist[base + w * channels + c]) * f32::cast_from(conv_w[w * w0 + c * w1]);
    }
    for w in 0..width - 2 {
        hist[base + w * channels + c] = hist[base + (w + 1) * channels + c];
    }
    hist[base + (width - 2) * channels + c] = x;
    acc / (1.0 + (-acc).exp())
}

/// `proj` `[B, T, 4 H hd + 2 H]` (q, k, v, gate, decay logits a, write logits b), `conv_w`
/// `[width, 3 H hd]`, `conv_in` `[B, width - 1, 3 H hd]`, `state_in` `[B, H, hd, hd]`, `a_log`,
/// `dt_bias` `[H]`, `norm_w` `[hd]`. Writes `out` `[B, T, H hd]` (normed and gated read-out,
/// before the output projection) and the carried `conv_out` and `state_out`. Inputs may be
/// strided (cubecl pitches rows); the outputs must be contiguous.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn delta_step_kernel<F: Float>(
    proj: &Tensor<F>,
    conv_w: &Tensor<F>,
    conv_in: &Tensor<F>,
    state_in: &Tensor<F>,
    a_log: &Tensor<F>,
    dt_bias: &Tensor<F>,
    norm_w: &Tensor<F>,
    out: &mut Tensor<F>,
    conv_out: &mut Tensor<F>,
    state_out: &mut Tensor<F>,
    steps: usize,
    q_scale: f32,
    inv_hd: f32,
    eps: f32,
    #[comptime] heads: usize,
    #[comptime] hd: usize,
    #[comptime] width: usize,
    #[define(F)] _dtype: ElemType,
) {
    let bh = CUBE_POS;
    let b = bh / heads;
    let h = bh % heads;
    let j = UNIT_POS as usize;
    let hk = heads * hd;
    let channels = 3 * hk;
    let mut q_sh = Shared::<[f32]>::new_slice(hd);
    let mut k_sh = Shared::<[f32]>::new_slice(hd);
    let mut o_sh = Shared::<[f32]>::new_slice(hd);

    // Carry the state column and the convolution history into the outputs, then update there.
    let s_base = bh * hd * hd;
    let s_in = b * state_in.stride(0) + h * state_in.stride(1) + j * state_in.stride(3);
    for i in 0..hd {
        state_out[s_base + i * hd + j] = state_in[s_in + i * state_in.stride(2)];
    }
    let c_base = b * (width - 1) * channels;
    let cq = h * hd + j;
    let (cs0, cs1, cs2) = (conv_in.stride(0), conv_in.stride(1), conv_in.stride(2));
    for w in 0..width - 1 {
        let at = c_base + w * channels + cq;
        let from = b * cs0 + w * cs1 + cq * cs2;
        conv_out[at] = conv_in[from];
        conv_out[at + hk] = conv_in[from + hk * cs2];
        conv_out[at + 2 * hk] = conv_in[from + 2 * hk * cs2];
    }
    let rate = f32::cast_from(a_log[h * a_log.stride(0)]).exp();
    let bias = f32::cast_from(dt_bias[h * dt_bias.stride(0)]);
    let gamma = f32::cast_from(norm_w[j * norm_w.stride(0)]);

    for t in 0..steps {
        let row = b * proj.stride(0) + t * proj.stride(1);
        let pc = proj.stride(2);
        let q = conv_silu::<F>(proj[row + cq * pc], conv_w, conv_out, c_base, channels, cq, width);
        let k = conv_silu::<F>(
            proj[row + (hk + cq) * pc],
            conv_w,
            conv_out,
            c_base,
            channels,
            hk + cq,
            width,
        );
        let v = conv_silu::<F>(
            proj[row + (2 * hk + cq) * pc],
            conv_w,
            conv_out,
            c_base,
            channels,
            2 * hk + cq,
            width,
        );
        q_sh[j] = q;
        k_sh[j] = k;
        sync_cube();

        let mut qq = 0.0f32;
        let mut kk = 0.0f32;
        let mut qk = 0.0f32;
        for i in 0..hd {
            let qi = q_sh[i];
            let ki = k_sh[i];
            qq += qi * qi;
            kk += ki * ki;
            qk += qi * ki;
        }
        // L2 norms (eps 1e-6 inside the root, as the reference); q also takes 1/sqrt(hd).
        let qn = q_scale / (qq + 1e-6).sqrt();
        let kn = 1.0 / (kk + 1e-6).sqrt();

        let a = f32::cast_from(proj[row + (4 * hk + h) * pc]) + bias;
        let softplus = select(a > 20.0, a, (1.0 + a.exp()).ln());
        let decay = (-rate * softplus).exp();
        let beta = 1.0 / (1.0 + (-f32::cast_from(proj[row + (4 * hk + heads + h) * pc])).exp());

        let mut ks = 0.0f32;
        let mut qs = 0.0f32;
        for i in 0..hd {
            let s = f32::cast_from(state_out[s_base + i * hd + j]);
            ks += k_sh[i] * s;
            qs += q_sh[i] * s;
        }
        // S' = decay S + k u^T with u = beta (v - decay S^T k); o = S'^T q.
        let u = beta * (v - decay * ks * kn);
        let o = decay * qs * qn + u * qk * qn * kn;
        let ku = kn * u;
        for i in 0..hd {
            let at = s_base + i * hd + j;
            let s = f32::cast_from(state_out[at]);
            state_out[at] = F::cast_from(decay * s + k_sh[i] * ku);
        }

        o_sh[j] = o;
        sync_cube();
        let mut oo = 0.0f32;
        for i in 0..hd {
            let oi = o_sh[i];
            oo += oi * oi;
        }
        let gate = f32::cast_from(proj[row + (3 * hk + cq) * pc]);
        let y = o / (oo * inv_hd + eps).sqrt() * gamma / (1.0 + (-gate).exp());
        out[(b * steps + t) * hk + cq] = F::cast_from(y);
        // Every unit is done with this token's shared values before the next overwrites them.
        sync_cube();
    }
}
