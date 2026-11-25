//! Matrix-vector products for decode: `y = x W` with a few rows in `x` (one token, or a
//! concept's k tokens). Burn's tiled matmul runs these at a fraction of the memory bandwidth,
//! and at batch 1 they are most of the GPU time of a decode step.
//!
//! [`gemv_cols_kernel`] reads `W` `[K, N]` with consecutive units on consecutive columns
//! (coalesced for a row-major weight) and splits `K` over a second grid axis when `N` alone gives
//! too few cubes; [`gemv_sum_kernel`] then adds the splits. [`gemv_rows_kernel`] is for a weight
//! stored transposed (`W[k, n]` at `n * K + k`, as the tied embedding in the logits): one plane
//! per output, its units along `K`.

use burn_cubecl::cubecl;
use cubecl::prelude::*;

/// Units per cube.
pub(crate) const CUBE: usize = 256;
/// Columns per cube of [`gemv_cols_kernel`]; `CUBE / COLS` units share a column.
pub(crate) const COLS: usize = 64;
const LANES: usize = CUBE / COLS;
/// Most rows `x` may have.
pub(crate) const MAX_ROWS: usize = 4;

/// Sums the `LANES` partial values of each column held in `red` (`CUBE` slots, lane-major).
#[cube]
fn lane_sum(red: &mut Shared<[f32]>, v: f32) -> f32 {
    let tid = UNIT_POS as usize;
    red[tid] = v;
    sync_cube();
    let mut total = 0.0f32;
    if tid < COLS {
        for l in 0..LANES {
            total += red[l * COLS + tid];
        }
    }
    sync_cube();
    total
}

/// Cube `(tile, split)`: columns `tile * COLS ..` over `k` in `split * k_split ..`. With one
/// split it writes `y` `[rows, N]` at the run dtype, otherwise f32 partials `part`
/// `[splits, rows, N]`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemv_cols_kernel<F: Float>(
    x: &Tensor<F>,
    w: &Tensor<F>,
    y: &mut Tensor<F>,
    part: &mut Tensor<f32>,
    k: usize,
    n: usize,
    k_split: usize,
    #[comptime] rows: usize,
    #[comptime] split: bool,
    #[define(F)] _dtype: ElemType,
) {
    let tid = UNIT_POS as usize;
    let col = CUBE_POS_X as usize * COLS + tid % COLS;
    let lane = tid / COLS;
    let s = CUBE_POS_Y as usize;
    let k0 = s * k_split;
    let k1 = select(k0 + k_split < k, k0 + k_split, k);
    let mut xs = Shared::<[f32]>::new_slice(MAX_ROWS * 512);
    let mut red = Shared::<[f32]>::new_slice(CUBE);
    // Stage this split's slice of x (k_split <= 512).
    let (x0, x1) = (x.stride(0), x.stride(1));
    let mut i = tid;
    while i < rows * k_split {
        let r = i / k_split;
        let kk = k0 + i % k_split;
        if kk < k1 {
            xs[r * 512 + i % k_split] = f32::cast_from(x[r * x0 + kk * x1]);
        }
        i += CUBE;
    }
    sync_cube();
    let (w0, w1) = (w.stride(0), w.stride(1));
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut a3 = 0.0f32;
    if col < n {
        let mut kk = k0 + lane;
        while kk < k1 {
            let wv = f32::cast_from(w[kk * w0 + col * w1]);
            let at = kk - k0;
            a0 += xs[at] * wv;
            if comptime!(rows > 1) {
                a1 += xs[512 + at] * wv;
            }
            if comptime!(rows > 2) {
                a2 += xs[1024 + at] * wv;
            }
            if comptime!(rows > 3) {
                a3 += xs[1536 + at] * wv;
            }
            kk += LANES;
        }
    }
    let t0 = lane_sum(&mut red, a0);
    let mut t1 = f32::new(0.0f32);
    let mut t2 = f32::new(0.0f32);
    let mut t3 = f32::new(0.0f32);
    if comptime!(rows > 1) {
        t1 = lane_sum(&mut red, a1);
    }
    if comptime!(rows > 2) {
        t2 = lane_sum(&mut red, a2);
    }
    if comptime!(rows > 3) {
        t3 = lane_sum(&mut red, a3);
    }
    if tid < COLS && col < n {
        if comptime!(split) {
            let base = s * rows * n + col;
            part[base] = t0;
            if comptime!(rows > 1) {
                part[base + n] = t1;
            }
            if comptime!(rows > 2) {
                part[base + 2 * n] = t2;
            }
            if comptime!(rows > 3) {
                part[base + 3 * n] = t3;
            }
        } else {
            y[col] = F::cast_from(t0);
            if comptime!(rows > 1) {
                y[n + col] = F::cast_from(t1);
            }
            if comptime!(rows > 2) {
                y[2 * n + col] = F::cast_from(t2);
            }
            if comptime!(rows > 3) {
                y[3 * n + col] = F::cast_from(t3);
            }
        }
    }
}

/// `y[i] = sum over splits of part[s, i]` for the `count` elements of `y` `[rows, N]`.
#[cube(launch_unchecked)]
pub(crate) fn gemv_sum_kernel<F: Float>(
    part: &Tensor<f32>,
    y: &mut Tensor<F>,
    count: usize,
    splits: usize,
    #[define(F)] _dtype: ElemType,
) {
    let i = ABSOLUTE_POS;
    if i >= count {
        terminate!();
    }
    let mut total = 0.0f32;
    for s in 0..splits {
        total += part[s * count + i];
    }
    y[i] = F::cast_from(total);
}

/// One plane per output column `n` (`CUBE / PLANE_DIM` per cube), its units striding `K`:
/// `y[r, n] = sum_k x[r, k] W[k, n]` for a weight whose column `n` is contiguous over `k`.
#[cube(launch_unchecked)]
pub(crate) fn gemv_rows_kernel<F: Float>(
    x: &Tensor<F>,
    w: &Tensor<F>,
    y: &mut Tensor<F>,
    k: usize,
    n: usize,
    #[comptime] rows: usize,
    #[define(F)] _dtype: ElemType,
) {
    let planes = (CUBE_DIM / PLANE_DIM) as usize;
    let col = CUBE_POS * planes + (UNIT_POS / PLANE_DIM) as usize;
    let lane = UNIT_POS_PLANE as usize;
    let width = PLANE_DIM as usize;
    let (x0, x1) = (x.stride(0), x.stride(1));
    let (w0, w1) = (w.stride(0), w.stride(1));
    let valid = col < n;
    let c = select(valid, col, 0usize);
    for r in 0..rows {
        let mut acc = 0.0f32;
        let mut kk = lane;
        while kk < k {
            acc += f32::cast_from(x[r * x0 + kk * x1]) * f32::cast_from(w[kk * w0 + c * w1]);
            kk += width;
        }
        let total = plane_sum(acc);
        if valid && lane == 0 {
            y[r * n + col] = F::cast_from(total);
        }
    }
}
