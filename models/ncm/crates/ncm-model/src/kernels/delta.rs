//! Fused gated DeltaNet step: short convolution, SiLU, L2 norms, gates, the delta-rule update,
//! the read-out, the per-head RMSNorm and the output gate in one kernel, token by token.
//!
//! One cube per (batch, head), `STEP_SPLIT` units per state column, the state in registers for
//! the whole call (see [`delta_step_kernel`]). All arithmetic is in f32; tensors are read and
//! written at the run dtype.

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

/// Units per state column of [`delta_step_kernel`]: each takes a quarter of the column's rows.
pub(crate) const STEP_SPLIT: usize = 4;

/// Sum of `v` over the cube's first `n` units (the others pass 0) through `red`, one slot per
/// plane; every unit gets the result.
#[cube]
fn head_sum(red: &mut Shared<[f32]>, v: f32) -> f32 {
    let total = plane_sum(v);
    let planes = CUBE_DIM / PLANE_DIM;
    if UNIT_POS_PLANE == 0 {
        red[(UNIT_POS / PLANE_DIM) as usize] = total;
    }
    sync_cube();
    let mut sum = 0.0f32;
    for p in 0..planes {
        sum += red[p as usize];
    }
    sync_cube();
    sum
}

/// `proj` `[B, T, 4 H hd + 2 H]` (q, k, v, gate, decay logits a, write logits b), `conv_w`
/// `[width, 3 H hd]`, `a_log`, `dt_bias` `[H]`, `norm_w` `[hd]` (all read through their strides).
/// Writes `out` `[B, T, H hd]` (normed and gated read-out, before the output projection) and
/// updates the contiguous cache `conv` `[B, width - 1, 3 H hd]` and `state` `[B, H, hd, hd]` in
/// place.
///
/// One cube per (batch, head) with `STEP_SPLIT` units per state column: unit `(j, r)` (`r` the
/// low bits of the unit index, so a column's units share a plane) holds rows
/// `r * hd / STEP_SPLIT ..` of column `j` in registers for the whole run, and the column's partial
/// products meet by plane shuffles. The first unit of each column owns its convolution channels
/// and its outputs. The state is read once and written once per call.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn delta_step_kernel<F: Float>(
    proj: &Tensor<F>,
    conv_w: &Tensor<F>,
    a_log: &Tensor<F>,
    dt_bias: &Tensor<F>,
    norm_w: &Tensor<F>,
    out: &mut Tensor<F>,
    conv: &mut Tensor<F>,
    state: &mut Tensor<F>,
    steps: usize,
    q_scale: f32,
    inv_hd: f32,
    eps: f32,
    #[comptime] heads: usize,
    #[comptime] hd: usize,
    #[comptime] width: usize,
    #[define(F)] _dtype: ElemType,
) {
    let rows = comptime!(hd / STEP_SPLIT);
    let bh = CUBE_POS;
    let b = bh / heads;
    let h = bh % heads;
    let tid = UNIT_POS as usize;
    let j = tid / STEP_SPLIT;
    let r = tid % STEP_SPLIT;
    let lead = r == 0;
    let hk = heads * hd;
    let channels = 3 * hk;
    let mut q_sh = Shared::<[f32]>::new_slice(hd);
    let mut k_sh = Shared::<[f32]>::new_slice(hd);
    let mut v_sh = Shared::<[f32]>::new_slice(hd);
    let mut o_sh = Shared::<[f32]>::new_slice(hd);
    let mut red = Shared::<[f32]>::new_slice(comptime!(hd * STEP_SPLIT / 32 + 1));

    let s_base = bh * hd * hd;
    let c_base = b * (width - 1) * channels;
    let cq = h * hd + j;
    let rate = f32::cast_from(a_log[h * a_log.stride(0)]).exp();
    let bias = f32::cast_from(dt_bias[h * dt_bias.stride(0)]);
    let gamma = f32::cast_from(norm_w[j * norm_w.stride(0)]);
    // This unit's rows of column j, kept in registers across the tokens.
    let mut s = Array::<f32>::new(rows);
    #[unroll]
    for m in 0..rows {
        s[m] = f32::cast_from(state[s_base + (r * rows + m) * hd + j]);
    }

    for t in 0..steps {
        let row = b * proj.stride(0) + t * proj.stride(1);
        let pc = proj.stride(2);
        if lead {
            q_sh[j] = conv_silu::<F>(proj[row + cq * pc], conv_w, conv, c_base, channels, cq, width);
            k_sh[j] = conv_silu::<F>(
                proj[row + (hk + cq) * pc],
                conv_w,
                conv,
                c_base,
                channels,
                hk + cq,
                width,
            );
            v_sh[j] = conv_silu::<F>(
                proj[row + (2 * hk + cq) * pc],
                conv_w,
                conv,
                c_base,
                channels,
                2 * hk + cq,
                width,
            );
        }
        sync_cube();
        // L2 norms (eps 1e-6 inside the root, as the reference); q also takes 1/sqrt(hd).
        let mut qq = 0.0f32;
        let mut kk = 0.0f32;
        let mut qk = 0.0f32;
        if tid < hd {
            let (qi, ki) = (q_sh[tid], k_sh[tid]);
            qq = qi * qi;
            kk = ki * ki;
            qk = qi * ki;
        }
        let qq = head_sum(&mut red, qq);
        let kk = head_sum(&mut red, kk);
        let qk = head_sum(&mut red, qk);
        let qn = q_scale / (qq + 1e-6).sqrt();
        let kn = 1.0 / (kk + 1e-6).sqrt();

        let a = f32::cast_from(proj[row + (4 * hk + h) * pc]) + bias;
        let softplus = select(a > 20.0, a, (1.0 + a.exp()).ln());
        let decay = (-rate * softplus).exp();
        let beta = 1.0 / (1.0 + (-f32::cast_from(proj[row + (4 * hk + heads + h) * pc])).exp());

        let mut ks = 0.0f32;
        let mut qs = 0.0f32;
        #[unroll]
        for m in 0..rows {
            let i = r * rows + m;
            ks += k_sh[i] * s[m];
            qs += q_sh[i] * s[m];
        }
        // The column's quarters meet (adjacent lanes of one plane).
        #[unroll]
        for bit in 0..comptime!(STEP_SPLIT.trailing_zeros() as usize) {
            let mask = comptime!(1u32 << bit);
            ks += plane_shuffle_xor(ks, mask);
            qs += plane_shuffle_xor(qs, mask);
        }
        // S' = decay S + k u^T with u = beta (v - decay S^T k); o = S'^T q.
        let u = beta * (v_sh[j] - decay * ks * kn);
        let o = decay * qs * qn + u * qk * qn * kn;
        let ku = kn * u;
        #[unroll]
        for m in 0..rows {
            s[m] = decay * s[m] + k_sh[r * rows + m] * ku;
        }
        if lead {
            o_sh[j] = o;
        }
        sync_cube();
        let mut oo = 0.0f32;
        if tid < hd {
            let oi = o_sh[tid];
            oo = oi * oi;
        }
        let oo = head_sum(&mut red, oo);
        if lead {
            let gate = f32::cast_from(proj[row + (3 * hk + cq) * pc]);
            let y = o / (oo * inv_hd + eps).sqrt() * gamma / (1.0 + (-gate).exp());
            out[(b * steps + t) * hk + cq] = F::cast_from(y);
        }
    }
    #[unroll]
    for m in 0..rows {
        state[s_base + (r * rows + m) * hd + j] = F::cast_from(s[m]);
    }
}

/// Units per cube of [`delta_scan_kernel`] (a multiple of every column group).
pub(crate) const SCAN_CUBE: usize = 256;
/// State columns per cube of [`delta_scan_kernel`]: few, so that a (batch, head) spreads over
/// many cubes (the scan is sequential in the chunks, so its parallelism is the columns).
pub(crate) const SCAN_COLUMNS: usize = 8;

/// The sequential part of the chunked gated delta rule (`crate::delta::delta_rule_chunked`) for
/// one (batch, head) and a group of `g` state columns per cube, chunk after chunk with the state
/// in shared memory: per chunk `i`,
///
/// ```text
/// fresh = values_i - w_i S
/// out_i = qg_i S + qk_i fresh
/// S     = carry_i S + k_out_i^T fresh
/// ```
///
/// The columns of the state never mix, so the column groups run in separate cubes. Inputs are
/// `[BH, N C, *]` (`qk` `[BH, N C, C]`, `carry` `[BH, N]`, `state` `[BH, dk, dv]`), read through
/// their strides. `out` `[BH, N C + dk, dv]` (contiguous) takes the chunk outputs in its first
/// `N C` rows and the final state in the last `dk`. The state stays in f32 between chunks.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn delta_scan_kernel<F: Float>(
    values: &Tensor<F>,
    w: &Tensor<F>,
    qk: &Tensor<F>,
    qg: &Tensor<F>,
    k_out: &Tensor<F>,
    carry: &Tensor<F>,
    state: &Tensor<F>,
    out: &mut Tensor<F>,
    chunks: usize,
    groups: usize,
    #[comptime] dk: usize,
    #[comptime] dv: usize,
    #[comptime] c: usize,
    #[comptime] g: usize,
    #[define(F)] _dtype: ElemType,
) {
    let bh = CUBE_POS / groups;
    let col0 = (CUBE_POS % groups) * g;
    let tid = UNIT_POS as usize;
    let col = tid % g;
    let r0 = tid / g;
    let rows = comptime!(SCAN_CUBE / g);
    let out_rows = chunks * c + dk;
    let mut s_sh = Shared::<[f32]>::new_slice(dk * g);
    let mut f_sh = Shared::<[f32]>::new_slice(c * g);
    let (st0, st1, st2) = (state.stride(0), state.stride(1), state.stride(2));
    let mut kk = r0;
    while kk < dk {
        s_sh[kk * g + col] = f32::cast_from(state[bh * st0 + kk * st1 + (col0 + col) * st2]);
        kk += rows;
    }
    sync_cube();
    let (v0, v1, v2) = (values.stride(0), values.stride(1), values.stride(2));
    let (w0, w1, w2) = (w.stride(0), w.stride(1), w.stride(2));
    let (a0, a1, a2) = (qk.stride(0), qk.stride(1), qk.stride(2));
    let (g0, g1, g2) = (qg.stride(0), qg.stride(1), qg.stride(2));
    let (k0, k1, k2) = (k_out.stride(0), k_out.stride(1), k_out.stride(2));
    for i in 0..chunks {
        let row0 = i * c;
        // fresh = values - w S
        let mut rr = r0;
        while rr < c {
            let wr = bh * w0 + (row0 + rr) * w1;
            let mut acc = f32::cast_from(values[bh * v0 + (row0 + rr) * v1 + (col0 + col) * v2]);
            for kk in 0..dk {
                acc -= f32::cast_from(w[wr + kk * w2]) * s_sh[kk * g + col];
            }
            f_sh[rr * g + col] = acc;
            rr += rows;
        }
        sync_cube();
        // out = qg S + qk fresh (qk is zero above the diagonal)
        let mut rr = r0;
        while rr < c {
            let gr = bh * g0 + (row0 + rr) * g1;
            let ar = bh * a0 + (row0 + rr) * a1;
            let mut acc = 0.0f32;
            for kk in 0..dk {
                acc += f32::cast_from(qg[gr + kk * g2]) * s_sh[kk * g + col];
            }
            for j in 0..c {
                if j <= rr {
                    acc += f32::cast_from(qk[ar + j * a2]) * f_sh[j * g + col];
                }
            }
            out[(bh * out_rows + row0 + rr) * dv + col0 + col] = F::cast_from(acc);
            rr += rows;
        }
        sync_cube();
        // S = carry S + k_out^T fresh
        let keep = f32::cast_from(carry[bh * carry.stride(0) + i * carry.stride(1)]);
        let mut kk = r0;
        while kk < dk {
            let mut acc = s_sh[kk * g + col] * keep;
            for j in 0..c {
                acc +=
                    f32::cast_from(k_out[bh * k0 + (row0 + j) * k1 + kk * k2]) * f_sh[j * g + col];
            }
            s_sh[kk * g + col] = acc;
            kk += rows;
        }
        sync_cube();
    }
    let mut kk = r0;
    while kk < dk {
        out[(bh * out_rows + chunks * c + kk) * dv + col0 + col] = F::cast_from(s_sh[kk * g + col]);
        kk += rows;
    }
}
