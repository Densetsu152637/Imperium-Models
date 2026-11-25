//! Fused dequantise-and-multiply kernels for quantised linear layers ([`crate::quant`]).
//!
//! * [`qgemv_kernel`]: `y = x W` for up to [`crate::quant::GEMV_ROWS`] rows of `x`. One plane
//!   (warp) per output feature streams that feature's packed row once, each lane taking every
//!   32nd word, decoding its codes in registers and accumulating every row of `x` against them;
//!   a plane sum finishes the dot products. The weight is read once per call whatever the row count, so decode
//!   moves `bits / 16` of the bf16 weight traffic.
//! * [`dequant_kernel`]: decodes the weight to `[d_in, d_out]` at the run dtype for Burn's
//!   tuned matmul when there are more rows (prefill, where compute rather than weight traffic
//!   dominates).
//!
//! Arithmetic is f32; `x` and outputs are at the run dtype, codes i32, scales f32.

use burn_cubecl::cubecl;
use cubecl::prelude::*;

/// Output features per GEMV cube (one plane each).
pub(crate) const PLANES: usize = 4;
/// Lanes per plane: the CUDA warp.
pub(crate) const PLANE: usize = 32;
/// Units per dequantisation cube.
pub(crate) const DEQUANT_CUBE: usize = 256;

/// `x` `[rows, k_len]` (any strides), `packed` `[n_len, words]` i32, `scales` `[n_len, groups]`
/// f32; writes `out` `[rows, n_len]` (contiguous). Cube `c`, plane `p` computes feature
/// `c * PLANES + p`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qgemv_kernel<F: Float>(
    x: &Tensor<F>,
    packed: &Tensor<i32>,
    scales: &Tensor<f32>,
    out: &mut Tensor<F>,
    k_len: usize,
    n_len: usize,
    words: usize,
    #[comptime] bits: u32,
    #[comptime] rows: usize,
    #[comptime] group_words: usize,
    #[define(F)] _dtype: ElemType,
) {
    let lane = UNIT_POS_X as usize;
    let n = CUBE_POS_X as usize * PLANES + UNIT_POS_Y as usize;
    if n >= n_len {
        terminate!();
    }
    let per = comptime!(32 / bits) as usize;
    let mask = comptime!((1u32 << bits) - 1) as i32;
    let offset = comptime!(1u32 << (bits - 1)) as f32;
    let (x0, x1) = (x.stride(0), x.stride(1));
    let row = n * packed.stride(0);
    let srow = n * scales.stride(0);
    let mut acc = Array::<f32>::new(rows);
    #[unroll]
    for r in 0..rows {
        acc[r] = 0.0;
    }
    let mut w = lane;
    while w < words {
        let word = packed[row + w];
        let scale = scales[srow + w / group_words];
        let k0 = w * per;
        let mut part = Array::<f32>::new(rows);
        #[unroll]
        for r in 0..rows {
            part[r] = 0.0;
        }
        #[unroll]
        for j in 0..per {
            let k = k0 + j;
            if k < k_len {
                let code = f32::cast_from((word >> comptime!(j as u32 * bits) as i32) & mask);
                let q = code - offset;
                #[unroll]
                for r in 0..rows {
                    part[r] += q * f32::cast_from(x[r * x0 + k * x1]);
                }
            }
        }
        #[unroll]
        for r in 0..rows {
            acc[r] += part[r] * scale;
        }
        w += PLANE;
    }
    #[unroll]
    for r in 0..rows {
        let total = plane_sum(acc[r]);
        if lane == 0 {
            out[r * n_len + n] = F::cast_from(total);
        }
    }
}

/// Decodes `packed` `[n_len, words]` with `scales` into `out` `[k_len, n_len]` (contiguous):
/// one unit per (feature, word), features fastest so the writes coalesce.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dequant_kernel<F: Float>(
    packed: &Tensor<i32>,
    scales: &Tensor<f32>,
    out: &mut Tensor<F>,
    k_len: usize,
    n_len: usize,
    words: usize,
    #[comptime] bits: u32,
    #[comptime] group_words: usize,
    #[define(F)] _dtype: ElemType,
) {
    let id = ABSOLUTE_POS;
    if id >= n_len * words {
        terminate!();
    }
    let n = id % n_len;
    let w = id / n_len;
    let per = comptime!(32 / bits) as usize;
    let mask = comptime!((1u32 << bits) - 1) as i32;
    let offset = comptime!(1u32 << (bits - 1)) as f32;
    let word = packed[n * packed.stride(0) + w];
    let scale = scales[n * scales.stride(0) + w / group_words];
    #[unroll]
    for j in 0..per {
        let k = w * per + j;
        if k < k_len {
            let code = f32::cast_from((word >> comptime!(j as u32 * bits) as i32) & mask);
            out[k * n_len + n] = F::cast_from((code - offset) * scale);
        }
    }
}
