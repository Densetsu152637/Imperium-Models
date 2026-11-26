//! Matrix-vector products for decode: `y = x W` with a few rows in `x` (one token, or a
//! concept's k tokens). Burn's tiled matmul runs these at a fraction of the memory bandwidth,
//! and at batch 1 they are most of the GPU time of a decode step.
//!
//! [`gemv_vec_kernel`] reads a row-major `W` `[K, N]` in vectors of consecutive columns (16
//! bytes per load where the width allows), eight units striding `K` per vector, and splits `K`
//! over a second grid axis for about two cubes per SM; [`gemv_sum_kernel`] then adds the splits.
//! [`gemv_cols_kernel`] is its scalar form (one column per unit, up to eight rows), for widths
//! no vector divides or strided weights. [`gemv_rows_kernel`] is for a weight
//! stored transposed (`W[k, n]` at `n * K + k`, as the tied embedding in the logits): one plane
//! per output, its units along `K`.

use burn_cubecl::cubecl;
use cubecl::prelude::*;

/// Units per cube.
pub(crate) const CUBE: usize = 256;
/// Columns per cube of [`gemv_cols_kernel`]; `CUBE / COLS` units share a column.
pub(crate) const COLS: usize = 64;
const LANES: usize = CUBE / COLS;
/// Most rows `x` may have (every kernel here).
pub(crate) const MAX_ROWS: usize = 8;
/// Most rows of [`gemv_cols_kernel`] (its staged slice is `COLS_ROWS * 512` values).
pub(crate) const COLS_ROWS: usize = MAX_ROWS;

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
    let mut xs = Shared::<[f32]>::new_slice(COLS_ROWS * 512);
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
    let mut acc = Array::<f32>::new(rows);
    #[unroll]
    for r in 0..rows {
        acc[r] = 0.0;
    }
    if col < n {
        let mut kk = k0 + lane;
        while kk < k1 {
            let wv = f32::cast_from(w[kk * w0 + col * w1]);
            let at = kk - k0;
            #[unroll]
            for r in 0..rows {
                acc[r] += xs[r * 512 + at] * wv;
            }
            kk += LANES;
        }
    }
    #[unroll]
    for r in 0..rows {
        let total = lane_sum(&mut red, acc[r]);
        if tid < COLS && col < n {
            if comptime!(split) {
                part[s * rows * n + r * n + col] = total;
            } else {
                y[r * n + col] = F::cast_from(total);
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

/// Column units per cube of [`gemv_vec_kernel`], each owning one vector of columns.
pub(crate) const VEC_COLS: usize = 32;
/// Units per column unit of [`gemv_vec_kernel`], striding `K`.
pub(crate) const VEC_LANES: usize = CUBE / VEC_COLS;
/// Largest `K` slice one cube of [`gemv_vec_kernel`] stages.
pub(crate) const VEC_K: usize = 512;

/// Writes one row's totals: the `VEC_LANES` partial vectors of each column unit, summed.
#[cube]
#[allow(clippy::too_many_arguments)]
fn vec_store<F: Float, L: Size>(
    red: &mut Shared<[f32]>,
    acc: Vector<f32, L>,
    y: &mut Tensor<F>,
    part: &mut Tensor<f32>,
    r: usize,
    rows: usize,
    n: usize,
    #[comptime] split: bool,
) {
    let v = acc.vector_size().comptime();
    let tid = UNIT_POS as usize;
    let (cu, lane) = (tid % VEC_COLS, tid / VEC_COLS);
    let width = VEC_COLS * v;
    #[unroll]
    for i in 0..v {
        red[lane * width + cu * v + i] = acc.extract(i);
    }
    sync_cube();
    if tid < width {
        let col = CUBE_POS_X as usize * width + tid;
        if col < n {
            let mut total = 0.0f32;
            for l in 0..VEC_LANES {
                total += red[l * width + tid];
            }
            if comptime!(split) {
                part[(CUBE_POS_Y as usize * rows + r) * n + col] = total;
            } else {
                y[r * n + col] = F::cast_from(total);
            }
        }
    }
    sync_cube();
}

/// `y = x L` for a packed row-major `L` `[K, L]` read as vectors of `L`'s columns (16 bytes per
/// load). Cube `(tile, split)`: column unit `u` owns the vector of columns `(tile * VEC_COLS +
/// u) * v ..`, its `VEC_LANES` units stride `K` in `split * k_split ..`, and a shared-memory sum
/// merges them. With one split it writes `y` `[rows, L]`. Otherwise it writes f32 partials
/// `part` `[splits, rows, L]`, publishes them (a device fence) and counts its arrival in
/// `counters[tile]`; the last cube of the tile to arrive adds the splits in order (as
/// [`gemv_sum_kernel`] would) and resets the counter, so no second launch is needed.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemv_vec_kernel<F: Float, L: Size>(
    x: &Tensor<F>,
    w: &Tensor<Vector<F, L>>,
    y: &mut Tensor<F>,
    part: &mut Tensor<f32>,
    counters: &mut Tensor<Atomic<i32>>,
    k: usize,
    n: usize,
    k_split: usize,
    #[comptime] rows: usize,
    #[comptime] split: bool,
    #[define(F)] _dtype: ElemType,
) {
    let v = w.vector_size().comptime();
    let tid = UNIT_POS as usize;
    let (cu, lane) = (tid % VEC_COLS, tid / VEC_COLS);
    let col0 = (CUBE_POS_X as usize * VEC_COLS + cu) * v;
    let k0 = CUBE_POS_Y as usize * k_split;
    let k1 = select(k0 + k_split < k, k0 + k_split, k);
    let mut xs = Shared::<[f32]>::new_slice(MAX_ROWS * VEC_K);
    let (x0, x1) = (x.stride(0), x.stride(1));
    let mut i = tid;
    while i < rows * k_split {
        let r = i / k_split;
        let kk = k0 + i % k_split;
        if kk < k1 {
            xs[r * VEC_K + i % k_split] = f32::cast_from(x[r * x0 + kk * x1]);
        }
        i += CUBE;
    }
    sync_cube();
    let w0 = w.stride(0);
    let zero = Vector::<f32, L>::new(0.0f32);
    let mut acc = Array::<Vector<f32, L>>::new(rows);
    #[unroll]
    for r in 0..rows {
        acc[r] = zero;
    }
    if col0 < n {
        let mut kk = k0 + lane;
        while kk < k1 {
            let wv = Vector::<f32, L>::cast_from(w[(kk * w0 + col0) / v]);
            let at = kk - k0;
            #[unroll]
            for r in 0..rows {
                acc[r] += wv * Vector::new(xs[r * VEC_K + at]);
            }
            kk += VEC_LANES;
        }
    }
    let mut red = Shared::<[f32]>::new_slice(CUBE * 8);
    #[unroll]
    for r in 0..rows {
        vec_store::<F, L>(&mut red, acc[r], y, part, r, rows, n, split);
    }
    if comptime!(split) {
        let splits = CUBE_COUNT_Y as usize;
        let tile = CUBE_POS_X as usize;
        sync_storage();
        sync_cube();
        let mut arrived = Shared::<[i32]>::new_slice(1usize);
        if tid == 0 {
            arrived[0] = counters[tile].fetch_add(1i32);
        }
        sync_cube();
        if arrived[0] == i32::cast_from(splits) - 1 {
            sync_storage();
            let width = VEC_COLS * v;
            let col = tile * width + tid;
            if tid < width && col < n {
                #[unroll]
                for r in 0..rows {
                    let mut total = 0.0f32;
                    for s in 0..splits {
                        total += part[(s * rows + r) * n + col];
                    }
                    y[r * n + col] = F::cast_from(total);
                }
            }
            if tid == 0 {
                counters[tile].store(0i32);
            }
        }
    }
}

/// Output columns per cube of [`gemm_small_kernel`].
pub(crate) const GEMM_COLS: usize = 64;
/// `K` per staged tile of [`gemm_small_kernel`].
pub(crate) const GEMM_K: usize = 32;
/// Row groups of [`gemm_small_kernel`]: unit `u` owns column `u % GEMM_COLS` of rows
/// `u / GEMM_COLS + GEMM_GROUPS i`.
pub(crate) const GEMM_GROUPS: usize = CUBE / GEMM_COLS;
/// Most rows [`gemm_small_kernel`] takes (the step sizes between the GEMV kernels' eight rows and
/// a prefill chunk: batched slot steps, verify passes, small prefill pieces).
pub(crate) const GEMM_ROWS: usize = 64;

/// Small-M GEMM `y = x W` for `x` `[m, K]` (`m <= GEMM_ROWS`) and `W` `[K, N]` read through its
/// strides (row-major, or with `kmajor` the transpose of a row-major `[N, K]`). Cube
/// `(tile, split)`: columns `tile * GEMM_COLS ..` of every row over `K` in
/// `split * k_split ..`, staging `GEMM_K`-deep tiles of `x` and `W` in shared memory (each
/// weight value read once per cube, coalesced either way round), every unit accumulating its
/// column of `rows_per` rows in registers. With one split it writes `y` `[m, N]` at the run
/// dtype, otherwise f32 partials `part` `[splits, m, N]` for [`gemv_sum_kernel`].
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_small_kernel<F: Float>(
    x: &Tensor<F>,
    w: &Tensor<F>,
    y: &mut Tensor<F>,
    part: &mut Tensor<f32>,
    m: usize,
    k: usize,
    n: usize,
    k_split: usize,
    #[comptime] rows_per: usize,
    #[comptime] kmajor: bool,
    #[comptime] split: bool,
    #[define(F)] _dtype: ElemType,
) {
    let tid = UNIT_POS as usize;
    let c = tid % GEMM_COLS;
    let g = tid / GEMM_COLS;
    let n0 = CUBE_POS_X as usize * GEMM_COLS;
    let col = n0 + c;
    let s = CUBE_POS_Y as usize;
    let k_start = s * k_split;
    let k_end = select(k_start + k_split < k, k_start + k_split, k);
    let mut xs = Shared::<[f32]>::new_slice(comptime!(GEMM_ROWS * (GEMM_K + 1)));
    let mut ws = Shared::<[f32]>::new_slice(comptime!(GEMM_K * GEMM_COLS));
    let (x0, x1) = (x.stride(0), x.stride(1));
    let (w0, w1) = (w.stride(0), w.stride(1));
    let mut acc = Array::<f32>::new(rows_per);
    #[unroll]
    for i in 0..rows_per {
        acc[i] = 0.0;
    }
    let mut kb = k_start;
    while kb < k_end {
        // x tile `[m, GEMM_K]` (padded by one per row against bank conflicts), zeros past K.
        let mut i = tid;
        while i < m * GEMM_K {
            let (r, kk) = (i / GEMM_K, i % GEMM_K);
            let at = kb + kk;
            let mut v = 0.0f32;
            if at < k_end {
                v = f32::cast_from(x[r * x0 + at * x1]);
            }
            xs[r * (GEMM_K + 1) + kk] = v;
            i += CUBE;
        }
        // W tile `[GEMM_K, GEMM_COLS]`, read along whichever axis is contiguous.
        let mut i = tid;
        while i < GEMM_K * GEMM_COLS {
            let mut kk = i / GEMM_COLS;
            let mut cc = i % GEMM_COLS;
            if comptime!(kmajor) {
                kk = i % GEMM_K;
                cc = i / GEMM_K;
            }
            let (at, cn) = (kb + kk, n0 + cc);
            let mut v = 0.0f32;
            if at < k_end && cn < n {
                v = f32::cast_from(w[at * w0 + cn * w1]);
            }
            ws[kk * GEMM_COLS + cc] = v;
            i += CUBE;
        }
        sync_cube();
        #[unroll]
        for kk in 0..GEMM_K {
            let wv = ws[kk * GEMM_COLS + c];
            #[unroll]
            for i in 0..rows_per {
                let r = g + i * GEMM_GROUPS;
                if r < m {
                    acc[i] += xs[r * (GEMM_K + 1) + kk] * wv;
                }
            }
        }
        sync_cube();
        kb += GEMM_K;
    }
    if col < n {
        #[unroll]
        for i in 0..rows_per {
            let r = g + i * GEMM_GROUPS;
            if r < m {
                if comptime!(split) {
                    part[s * m * n + r * n + col] = acc[i];
                } else {
                    y[r * n + col] = F::cast_from(acc[i]);
                }
            }
        }
    }
}
