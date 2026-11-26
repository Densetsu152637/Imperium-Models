//! Fused dequantise-and-multiply kernels for quantised linear layers ([`crate::quant`]).
//!
//! * [`qgemv_vec_kernel`]: `y = x W` for up to [`crate::quant::GEMV_ROWS`] rows of `x` (decode).
//!   One plane (warp) per output feature streams that feature's packed row once in vectors of
//!   words, decoding its codes in registers against `x` staged in shared memory (in K tiles for
//!   wide inputs); a plane sum finishes the dot products. The weight is read once per call
//!   whatever the row count, so decode moves `bits / 16` of the bf16 weight traffic.
//! * [`qgemm_small_kernel`]: 9 to [`QGEMM_ROWS`] rows (batched slot steps, verify passes, short
//!   prefill pieces): tiles of the weight decoded into shared memory on the fly and multiplied
//!   against staged tiles of `x`, the weight still read once per column tile.
//! * [`dequant_kernel`]: decodes the weight to `[d_in, d_out]` at the run dtype for Burn's
//!   tuned matmul when there are more rows (long prefill, where compute rather than weight
//!   traffic dominates).
//!
//! Arithmetic is f32; `x` and outputs are at the run dtype, codes i32, scales f32.

use burn_cubecl::cubecl;
use cubecl::prelude::*;

/// Lanes per plane: the CUDA warp.
pub(crate) const PLANE: usize = 32;
/// Output features per cube of [`qgemv_vec_kernel`].
pub(crate) const QPLANES: usize = 8;
/// Most `x` slots [`qgemv_vec_kernel`] stages at once (36 KiB); wider inputs go through in
/// tiles ([`staged_tile`]).
pub(crate) const QX: usize = 9216;
/// Units per dequantisation cube.
pub(crate) const DEQUANT_CUBE: usize = 256;

/// `y = x W` for `x` `[rows, k_len]` (any strides, `rows <= GEMV_ROWS`), `packed`
/// `[n_len, words]` i32 (contiguous along the words; [`crate::kernels::packed_int`]),
/// `scales` `[n_len, groups]` f32, into `out` `[rows, n_len]` (contiguous), with `x` staged in
/// shared memory (f32, skewed) a tile of `tile_k` inputs
/// at a time (`rows * (tile_k + tile_k / 32 + 1) <= QX`, a multiple of 512 unless it is all of
/// `k_len`), so inputs of any width keep it, and each lane's words read as one vector of `v`
/// (`v` from [`vector_words`]: as wide as the rows' alignment allows, down to one word): lane
/// `l` takes words `(l + 32 i) * v .. + v` of the tile, which share one scale unless
/// `straddle` (the group length not a multiple of `v`: then each word its own). `QPLANES`
/// features per cube share the staged `x`.
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
    tile_k: usize,
    #[comptime] bits: u32,
    #[comptime] rows: usize,
    #[comptime] group_words: usize,
    #[comptime] straddle: bool,
    #[define(F)] _dtype: ElemType,
) {
    let v = packed.vector_size().comptime();
    let lane = UNIT_POS_X as usize;
    let n = CUBE_POS_X as usize * QPLANES + UNIT_POS_Y as usize;
    // Planes past the last feature still stage and wait with the others.
    let active = n < n_len;
    let mut xs = Shared::<[f32]>::new_slice(QX);
    let (x0, x1) = (x.stride(0), x.stride(1));
    let tid = UNIT_POS as usize;
    // Skewed by one slot every 32 values: lanes reading every 4th to 32nd value hit distinct
    // banks.
    let stride = tile_k + tile_k / 32 + 1;
    let per = comptime!(32 / bits) as usize;
    let mask = comptime!((1u32 << bits) - 1) as i32;
    let offset = comptime!(1u32 << (bits - 1)) as f32;
    let row = select(active, n, 0usize) * packed.stride(0);
    let srow = select(active, n, 0usize) * scales.stride(0);
    let mut acc = Array::<f32>::new(rows);
    #[unroll]
    for r in 0..rows {
        acc[r] = 0.0;
    }
    let mut k_base = 0usize;
    while k_base < k_len {
        let tk = select(k_base + tile_k < k_len, tile_k, k_len - k_base);
        let mut i = tid;
        while i < rows * tk {
            let (r, kk) = (i / tk, i % tk);
            xs[r * stride + kk + kk / 32] = f32::cast_from(x[r * x0 + (k_base + kk) * x1]);
            i += PLANE * QPLANES;
        }
        sync_cube();
        if active {
            let w_end =
                select((k_base + tk).div_ceil(per) < words, (k_base + tk).div_ceil(per), words);
            let mut w = k_base / per + lane * v;
            while w < w_end {
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
                            let kk = k - k_base;
                            let code =
                                f32::cast_from((word >> comptime!(j as u32 * bits) as i32) & mask);
                            let q = code - offset;
                            #[unroll]
                            for r in 0..rows {
                                part[r] += q * xs[r * stride + kk + kk / 32];
                            }
                        }
                    }
                    if comptime!(straddle) {
                        // The vector may cross a group boundary: each word its own scale.
                        let ws = scales[srow + (w + i) / group_words];
                        #[unroll]
                        for r in 0..rows {
                            acc[r] += part[r] * ws;
                            part[r] = 0.0;
                        }
                    }
                }
                if comptime!(!straddle) {
                    #[unroll]
                    for r in 0..rows {
                        acc[r] += part[r] * scale;
                    }
                }
                w += PLANE * v;
            }
        }
        sync_cube();
        k_base += tile_k;
    }
    #[unroll]
    for r in 0..rows {
        let total = plane_sum(acc[r]);
        if active && lane == 0 {
            out[r * n_len + n] = F::cast_from(total);
        }
    }
}

/// Words per vector load of the staged quantised kernels for a contiguous `[n, words]` code
/// matrix with row stride `row_stride`: the preferred width (four for q8, two for q4, whose
/// decoding rather than its loads limits it) or the largest smaller power of two that keeps
/// every row's vectors aligned, down to one. Whether vectors stay inside one scale group of
/// `group_words` is the kernels' `straddle` flag ([`straddles`]), not a reason to narrow.
pub(crate) fn vector_words(bits: u32, words: usize, row_stride: usize) -> usize {
    let mut v = if bits == 4 { 2 } else { 4 };
    while v > 1 && !(words.is_multiple_of(v) && row_stride.is_multiple_of(v)) {
        v /= 2;
    }
    v
}

/// Whether vectors of `v` words may cross a scale group of `group_words` (then every word
/// looks up its own scale).
pub(crate) fn straddles(v: usize, group_words: usize) -> bool {
    !group_words.is_multiple_of(v)
}

/// The input tile of the staged quantised kernels for `rows` rows of `k_len` inputs within
/// `capacity` staged slots (skewed by one every 32): all of `k_len` when it fits, else the
/// largest multiple of 512 that does (512 aligns every q8/q4 word vector and scale group).
pub(crate) fn staged_tile(rows: usize, k_len: usize, capacity: usize) -> usize {
    let fits = |t: usize| rows * (t + t / 32 + 1) <= capacity;
    if fits(k_len) {
        return k_len;
    }
    let mut tile = 512;
    while tile + 512 < k_len && fits(tile + 512) {
        tile += 512;
    }
    tile
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

/// Output features per cube of [`qgemm_small_kernel`].
pub(crate) const QGEMM_COLS: usize = 64;
/// `K` per decoded tile of [`qgemm_small_kernel`] (a multiple of every format's codes per word).
pub(crate) const QGEMM_K: usize = 32;
/// Units per cube of [`qgemm_small_kernel`].
pub(crate) const QGEMM_CUBE: usize = 256;
/// Row groups of [`qgemm_small_kernel`]: unit `u` owns feature `u % QGEMM_COLS` of rows
/// `u / QGEMM_COLS + QGEMM_GROUPS i`.
pub(crate) const QGEMM_GROUPS: usize = QGEMM_CUBE / QGEMM_COLS;
/// Most rows [`qgemm_small_kernel`] takes.
pub(crate) const QGEMM_ROWS: usize = 64;

/// Small-M quantised GEMM `y = x W` for `x` `[m, k]` (`m <= QGEMM_ROWS`, any strides) and the
/// weight as `packed` `[n, words]` codes with `scales` `[n, groups]` (read through their
/// strides). Cube `(tile, split)`: features `tile * QGEMM_COLS ..` over `k` in
/// `split * k_split ..` (a multiple of `QGEMM_K`): per `QGEMM_K`-deep tile, `x` staged and the
/// tile's codes decoded with their scales into shared memory (each feature's words read
/// contiguously, once per cube), then every unit accumulates its feature of `rows_per` rows in
/// registers. With one split it writes `y` `[m, n]` at the run dtype, otherwise f32 partials
/// `part` `[splits, m, n]` for `gemv::gemv_sum_kernel`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qgemm_small_kernel<F: Float>(
    x: &Tensor<F>,
    packed: &Tensor<i32>,
    scales: &Tensor<f32>,
    y: &mut Tensor<F>,
    part: &mut Tensor<f32>,
    m: usize,
    k: usize,
    n: usize,
    words: usize,
    k_split: usize,
    #[comptime] bits: u32,
    #[comptime] group_words: usize,
    #[comptime] rows_per: usize,
    #[comptime] split: bool,
    #[define(F)] _dtype: ElemType,
) {
    let per = comptime!(32 / bits) as usize;
    let wpt = comptime!(QGEMM_K / (32 / bits) as usize);
    let mask = comptime!((1u32 << bits) - 1) as i32;
    let offset = comptime!(1u32 << (bits - 1)) as f32;
    let tid = UNIT_POS as usize;
    let c = tid % QGEMM_COLS;
    let g = tid / QGEMM_COLS;
    let n0 = CUBE_POS_X as usize * QGEMM_COLS;
    let col = n0 + c;
    let s = CUBE_POS_Y as usize;
    let k_start = s * k_split;
    let k_end = select(k_start + k_split < k, k_start + k_split, k);
    let mut xs = Shared::<[f32]>::new_slice(comptime!(QGEMM_ROWS * (QGEMM_K + 1)));
    let mut ws = Shared::<[f32]>::new_slice(comptime!(QGEMM_K * QGEMM_COLS));
    let (x0, x1) = (x.stride(0), x.stride(1));
    let (p0, p1) = (packed.stride(0), packed.stride(1));
    let (s0, s1) = (scales.stride(0), scales.stride(1));
    let mut acc = Array::<f32>::new(rows_per);
    #[unroll]
    for i in 0..rows_per {
        acc[i] = 0.0;
    }
    let mut kb = k_start;
    while kb < k_end {
        // x tile `[m, QGEMM_K]` (padded by one per row against bank conflicts), zeros past K.
        let mut i = tid;
        while i < m * QGEMM_K {
            let (r, kk) = (i / QGEMM_K, i % QGEMM_K);
            let at = kb + kk;
            let mut v = 0.0f32;
            if at < k_end {
                v = f32::cast_from(x[r * x0 + at * x1]);
            }
            xs[r * (QGEMM_K + 1) + kk] = v;
            i += QGEMM_CUBE;
        }
        // The weight tile `[QGEMM_K, QGEMM_COLS]` decoded: `wpt` words per feature.
        let mut i = tid;
        while i < QGEMM_COLS * wpt {
            let (cc, wi) = (i / wpt, i % wpt);
            let feature = n0 + cc;
            let w = kb / per + wi;
            let live = feature < n && w < words;
            let f = select(live, feature, 0usize);
            let wc = select(live, w, 0usize);
            let word = packed[f * p0 + wc * p1];
            let scale = select(live, scales[f * s0 + (wc / group_words) * s1], 0.0f32);
            #[unroll]
            for j in 0..per {
                let code = f32::cast_from((word >> comptime!(j as u32 * bits) as i32) & mask);
                ws[(wi * per + j) * QGEMM_COLS + cc] = (code - offset) * scale;
            }
            i += QGEMM_CUBE;
        }
        sync_cube();
        #[unroll]
        for kk in 0..QGEMM_K {
            let wv = ws[kk * QGEMM_COLS + c];
            #[unroll]
            for i in 0..rows_per {
                let r = g + i * QGEMM_GROUPS;
                if r < m {
                    acc[i] += xs[r * (QGEMM_K + 1) + kk] * wv;
                }
            }
        }
        sync_cube();
        kb += QGEMM_K;
    }
    if col < n {
        #[unroll]
        for i in 0..rows_per {
            let r = g + i * QGEMM_GROUPS;
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
