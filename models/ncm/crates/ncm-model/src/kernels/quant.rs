//! Fused dequantise-and-multiply kernels for quantised linear layers ([`crate::quant`]).
//!
//! * [`qgemv_kernel`]: `y = x W` for up to [`crate::quant::GEMV_ROWS`] rows of `x`. One plane
//!   (warp) per output feature streams that feature's packed row once, each lane taking every
//!   32nd word, decoding its codes in registers and accumulating every row of `x` against them;
//!   a plane sum finishes the dot products. The weight is read once per call whatever the row count, so decode
//!   moves `bits / 16` of the bf16 weight traffic.
//! * [`qgemv_vec_kernel`]: the same product with `x` staged in shared memory once per cube of
//!   eight features and each lane's words read as one vector, when `x` fits and the word count
//!   and group length allow (decode at the models' widths).
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
/// Output features per cube of [`qgemv_vec_kernel`].
pub(crate) const QPLANES: usize = 8;
/// Most `x` slots [`qgemv_vec_kernel`] stages (36 KiB: eight rows of a 1,024-wide input with
/// its skew, so batched slot steps of eight rows keep the vector kernel).
pub(crate) const QX: usize = 9216;
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

/// [`qgemv_kernel`] with `x` staged in shared memory (f32, skewed, `rows * (k_len + k_len / 32 + 1) <= QX`) and each lane's
/// words read as one vector of `v` when the word count and the group length are multiples of
/// `v`: lane `l` takes words `(l + 32 i) * v .. + v`, which share one scale. `QPLANES` features
/// per cube share the staged `x`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qgemv_vec_kernel<F: Float, N: Size>(
    x: &Tensor<F>,
    packed: &Tensor<Vector<i32, N>>,
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
    let v = packed.vector_size().comptime();
    let lane = UNIT_POS_X as usize;
    let n = CUBE_POS_X as usize * QPLANES + UNIT_POS_Y as usize;
    let mut xs = Shared::<[f32]>::new_slice(QX);
    let (x0, x1) = (x.stride(0), x.stride(1));
    let tid = UNIT_POS as usize;
    // Skewed by one slot every 32 values: lanes reading every 4th to 32nd value hit distinct
    // banks.
    let stride = k_len + k_len / 32 + 1;
    let mut i = tid;
    while i < rows * k_len {
        let (r, k) = (i / k_len, i % k_len);
        xs[r * stride + k + k / 32] = f32::cast_from(x[r * x0 + k * x1]);
        i += PLANE * QPLANES;
    }
    sync_cube();
    if n >= n_len {
        terminate!();
    }
    let per = comptime!(32 / bits) as usize;
    let mask = comptime!((1u32 << bits) - 1) as i32;
    let offset = comptime!(1u32 << (bits - 1)) as f32;
    let row = n * packed.stride(0);
    let srow = n * scales.stride(0);
    let mut acc = Array::<f32>::new(rows);
    #[unroll]
    for r in 0..rows {
        acc[r] = 0.0;
    }
    let mut w = lane * v;
    while w < words {
        let block = packed[(row + w) / v];
        let scale = scales[srow + w / group_words];
        let mut part = Array::<f32>::new(rows);
        #[unroll]
        for r in 0..rows {
            part[r] = 0.0;
        }
        #[unroll]
        for i in 0..v {
            let word = block.extract(i);
            let k0 = (w + i) * per;
            #[unroll]
            for j in 0..per {
                let k = k0 + j;
                if k < k_len {
                    let code = f32::cast_from((word >> comptime!(j as u32 * bits) as i32) & mask);
                    let q = code - offset;
                    #[unroll]
                    for r in 0..rows {
                        part[r] += q * xs[r * stride + k + k / 32];
                    }
                }
            }
        }
        #[unroll]
        for r in 0..rows {
            acc[r] += part[r] * scale;
        }
        w += PLANE * v;
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
