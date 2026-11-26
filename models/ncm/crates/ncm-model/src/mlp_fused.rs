//! The decode MLP in two fused launches (cargo feature `cuda`).
//!
//! A pre-norm block's MLP half is, unfused: the residual add and RMSNorm, the gate/up GEMV
//! (plus a split-K sum), SiLU(gate) * up (three elementwise ops), the down GEMV (plus a split-K
//! sum) and the residual add of its output. Here it is:
//!
//! 1. [`mlp_in`]: the gate/up GEMV with the add and the norm in its prologue (every cube
//!    computes the row's RMS itself and normalises the `K` slice it stages) and SwiGLU in its
//!    epilogue, writing the activation `[rows, ff]` and the residual sum.
//! 2. [`mlp_out`]: the down GEMV with the residual add in its epilogue.
//!
//! The same kernels serve the attention half: [`norm_linear`] folds the pre-attention norm
//! into a mixer's input projection (and the gate's sigmoid into its epilogue), and
//! [`gated_linear`] folds the output gate's product into the output projection's prologue.
//! Up to eight rows (tokens) per launch.
//!
//! Dense weights take a vectorised split-K GEMV whose splits are combined deterministically by
//! the last cube of each column tile to finish (a device fence, then an atomic arrival count in
//! a per-thread counter buffer that the last cube resets), so no separate sum launch remains.
//! q8/q4 weights take one plane per output feature (both features of a SwiGLU pair) with the
//! row staged in shared memory, as `kernels::qmatmul`'s vector kernel does.
//!
//! The arithmetic mirrors the unfused path's rounding at the run dtype `F` where it is cheap
//! to: the sum `F(x + delta)`, the norm (`rms` rounded to `F`, then `F(F(s / rms) * gamma)`),
//! the GEMV outputs rounded to `F` before SwiGLU (`F(F(silu(g)) * u)`) and before the residual
//! add (`F(resid + F(y))`). Dot products run in f32 in another order than the unfused GEMVs', so
//! results agree to rounding, not bit for bit.

use std::cell::RefCell;

use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{Backend, Dispatch, backend_extension};
use burn::tensor::{DType, Int, IntDType, Tensor};
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl::ir::{ElemType, FloatKind};
use burn_cubecl::cubecl::prelude::{CubeCount, CubeDim};
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;

use crate::quant::QLinear;

// Comptime switches nest inside runtime tests on purpose: the cube macro expands them apart.
#[allow(clippy::collapsible_if)]
mod kernel {
    use burn_cubecl::cubecl;
    use cubecl::prelude::*;

    /// Units per cube.
    pub(super) const CUBE: usize = 256;
    /// Column units per dense cube, each owning one vector of columns.
    pub(super) const COLS: usize = 32;
    /// Units per column unit, striding `K`.
    pub(super) const LANES: usize = CUBE / COLS;
    /// Largest `K` slice a dense cube stages.
    pub(super) const MAX_K: usize = 512;
    /// Most rows (tokens).
    pub(super) const MAX_ROWS: usize = 8;
    /// Output features per quantised cube (one plane each).
    pub(super) const PLANES: usize = 8;
    /// Lanes per plane.
    pub(super) const PLANE: usize = 32;
    /// Most staged `x` slots of a quantised cube.
    pub(super) const QX: usize = 8192;

    /// Sum of `v` over the cube (`CUBE` units, planes of 32).
    #[cube]
    fn cube_sum(v: f32, part: &mut Shared<[f32]>) -> f32 {
        let s = plane_sum(v);
        if UNIT_POS_PLANE == 0 {
            part[(UNIT_POS / PLANE_DIM) as usize] = s;
        }
        sync_cube();
        let mut total = 0.0f32;
        for p in 0..CUBE / 32 {
            total += part[p];
        }
        sync_cube();
        total
    }

    /// The prologue's input at `(r, kk)`: `F(x + delta)` with `add`, else `x`.
    #[cube]
    fn input<F: Float>(
        x: &Tensor<F>,
        delta: &Tensor<F>,
        r: usize,
        kk: usize,
        #[comptime] add: bool,
    ) -> F {
        let v = x[r * x.stride(0) + kk * x.stride(1)];
        if comptime!(add) {
            F::cast_from(
                f32::cast_from(v)
                    + f32::cast_from(delta[r * delta.stride(0) + kk * delta.stride(1)]),
            )
        } else {
            v
        }
    }

    /// `F(rms)` of row `r` of the prologue's input (all `k` of it), as the norm kernel rounds it.
    #[cube]
    fn row_rms<F: Float>(
        x: &Tensor<F>,
        delta: &Tensor<F>,
        r: usize,
        k: usize,
        eps: f32,
        part: &mut Shared<[f32]>,
        #[comptime] add: bool,
    ) -> f32 {
        let mut sq = 0.0f32;
        let mut kk = UNIT_POS as usize;
        while kk < k {
            let s = f32::cast_from(input(x, delta, r, kk, add));
            sq += s * s;
            kk += CUBE;
        }
        let total = cube_sum(sq, part);
        f32::cast_from(F::cast_from(f32::sqrt(total / f32::cast_from(k) + eps)))
    }

    /// The normed value of `s` (run dtype rounding as the norm kernel's).
    #[cube]
    fn normed<F: Float>(s: F, rms: f32, g: F) -> f32 {
        let n = f32::cast_from(F::cast_from(f32::cast_from(s) / rms));
        f32::cast_from(F::cast_from(n * f32::cast_from(g)))
    }

    /// The output of one element from its f32 dot products `g` (and `u`, the up half, with
    /// `glu`): SwiGLU and the residual add, rounded as the unfused ops round.
    #[cube]
    fn epilogue<F: Float>(
        g: f32,
        u: f32,
        rv: f32,
        #[comptime] glu: bool,
        #[comptime] sigmoid: bool,
        #[comptime] residual: bool,
    ) -> F {
        let mut out = f32::cast_from(F::cast_from(g));
        if comptime!(sigmoid) {
            out = f32::cast_from(F::cast_from(1.0f32 / (1.0f32 + f32::exp(-out))));
        }
        if comptime!(glu) {
            let gate = out;
            let silu = f32::cast_from(F::cast_from(gate / (1.0f32 + f32::exp(-gate))));
            out = f32::cast_from(F::cast_from(silu * f32::cast_from(F::cast_from(u))));
        }
        if comptime!(residual) {
            out += rv;
        }
        F::cast_from(out)
    }

    /// The `LANES` partial vectors of each column unit summed: unit `tid < COLS * v` gets its
    /// column's total.
    #[cube]
    fn lanes_total<L: Size>(red: &mut Shared<[f32]>, acc: Vector<f32, L>) -> f32 {
        let v = acc.vector_size().comptime();
        let tid = UNIT_POS as usize;
        let (cu, lane) = (tid % COLS, tid / COLS);
        let width = COLS * v;
        #[unroll]
        for i in 0..v {
            red[lane * width + cu * v + i] = acc.extract(i);
        }
        sync_cube();
        let mut total = 0.0f32;
        if tid < width {
            for l in 0..LANES {
                total += red[l * width + tid];
            }
        }
        sync_cube();
        total
    }

    /// Dense fused GEMV. `w` `[K, N_w]` row-major as vectors; with `glu` the output column `c`
    /// takes weight columns `c` (gate) and `c + n` (up), `N_w = 2 n`. Cube `(tile, split)` as
    /// in `kernels::gemv`'s vector kernel. `out` (flat) holds `y` `[rows, n]` at 0 and, with
    /// `add`, the residual sum `[rows, k]` at `rows * n` (written by cube (0, 0)). With splits,
    /// `part` takes f32 partials `[splits, rows, 2, n]` and the last cube of each tile to
    /// arrive (counted in `counters[tile]`, which it resets) adds them in split order.
    #[cube(launch_unchecked)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_kernel<F: Float, L: Size>(
        x: &Tensor<F>,
        delta: &Tensor<F>,
        gamma: &Tensor<F>,
        resid: &Tensor<F>,
        mulby: &Tensor<F>,
        w: &Tensor<Vector<F, L>>,
        out: &mut Tensor<F>,
        part: &mut Tensor<f32>,
        counters: &mut Tensor<Atomic<i32>>,
        k: usize,
        n: usize,
        k_split: usize,
        splits: usize,
        eps: f32,
        #[comptime] rows: usize,
        #[comptime] norm: bool,
        #[comptime] add: bool,
        #[comptime] glu: bool,
        #[comptime] residual: bool,
        #[comptime] mul: bool,
        #[comptime] sigmoid: bool,
        #[define(F)] _dtype: ElemType,
    ) {
        let v = w.vector_size().comptime();
        let tid = UNIT_POS as usize;
        let (cu, lane) = (tid % COLS, tid / COLS);
        let width = COLS * v;
        let tile = CUBE_POS_X as usize;
        let col0 = (tile * COLS + cu) * v;
        let s = CUBE_POS_Y as usize;
        let k0 = s * k_split;
        let k1 = select(k0 + k_split < k, k0 + k_split, k);
        let mut part_red = Shared::<[f32]>::new_slice(CUBE / 32);
        // Prologue: each row's RMS over the whole row, then the cube's K slice staged normed.
        let mut rms = Array::<f32>::new(rows);
        #[unroll]
        for r in 0..rows {
            rms[r] = 1.0f32;
            if comptime!(norm) {
                rms[r] = row_rms::<F>(x, delta, r, k, eps, &mut part_red, add);
            }
        }
        if comptime!(add) {
            if tile == 0 && s == 0 {
                let mut i = tid;
                while i < rows * k {
                    out[rows * n + i] = input(x, delta, i / k, i % k, add);
                    i += CUBE;
                }
            }
        }
        let mut xs = Shared::<[f32]>::new_slice(MAX_ROWS * MAX_K);
        let mut i = tid;
        while i < rows * k_split {
            let r = i / k_split;
            let kk = k0 + i % k_split;
            if kk < k1 {
                let sv = input(x, delta, r, kk, add);
                let mut val = f32::cast_from(sv);
                if comptime!(norm) {
                    val = normed::<F>(sv, rms[r], gamma[kk * gamma.stride(0)]);
                }
                if comptime!(mul) {
                    let m = f32::cast_from(mulby[r * mulby.stride(0) + kk * mulby.stride(1)]);
                    val = f32::cast_from(F::cast_from(val * m));
                }
                xs[r * MAX_K + i % k_split] = val;
            }
            i += CUBE;
        }
        sync_cube();
        // The GEMV: gate (or plain) accumulators `a`, up accumulators `b` with `glu`.
        let w0 = w.stride(0);
        let zero = Vector::<f32, L>::new(0.0f32);
        let mut a = Array::<Vector<f32, L>>::new(rows);
        let mut b = Array::<Vector<f32, L>>::new(rows);
        #[unroll]
        for r in 0..rows {
            a[r] = zero;
            b[r] = zero;
        }
        if col0 < n {
            let mut kk = k0 + lane;
            while kk < k1 {
                let at = kk - k0;
                let wg = Vector::<f32, L>::cast_from(w[(kk * w0 + col0) / v]);
                #[unroll]
                for r in 0..rows {
                    a[r] += wg * Vector::new(xs[r * MAX_K + at]);
                }
                if comptime!(glu) {
                    let wu = Vector::<f32, L>::cast_from(w[(kk * w0 + col0 + n) / v]);
                    #[unroll]
                    for r in 0..rows {
                        b[r] += wu * Vector::new(xs[r * MAX_K + at]);
                    }
                }
                kk += LANES;
            }
        }
        let mut red = Shared::<[f32]>::new_slice(CUBE * 8);
        let col = tile * width + tid;
        let mine = tid < width && col < n;
        if splits == 1 {
            #[unroll]
            for r in 0..rows {
                let g = lanes_total::<L>(&mut red, a[r]);
                let mut u = 0.0f32;
                if comptime!(glu) {
                    u = lanes_total::<L>(&mut red, b[r]);
                }
                if mine {
                    let mut rv = 0.0f32;
                    if comptime!(residual) {
                        rv = f32::cast_from(resid[r * resid.stride(0) + col * resid.stride(1)]);
                    }
                    out[r * n + col] = epilogue::<F>(g, u, rv, glu, sigmoid, residual);
                }
            }
            terminate!();
        }
        #[unroll]
        for r in 0..rows {
            let g = lanes_total::<L>(&mut red, a[r]);
            let mut u = 0.0f32;
            if comptime!(glu) {
                u = lanes_total::<L>(&mut red, b[r]);
            }
            if mine {
                part[((s * rows + r) * 2) * n + col] = g;
                part[((s * rows + r) * 2 + 1) * n + col] = u;
            }
        }
        // Publish the partials, then count this cube's arrival; the last one combines.
        sync_storage();
        sync_cube();
        let mut arrived = Shared::<[i32]>::new_slice(1usize);
        if tid == 0 {
            arrived[0] = counters[tile].fetch_add(1i32);
        }
        sync_cube();
        if arrived[0] != i32::cast_from(splits) - 1 {
            terminate!();
        }
        sync_storage();
        if mine {
            #[unroll]
            for r in 0..rows {
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for sp in 0..splits {
                    g += part[((sp * rows + r) * 2) * n + col];
                    if comptime!(glu) {
                        u += part[((sp * rows + r) * 2 + 1) * n + col];
                    }
                }
                let mut rv = 0.0f32;
                if comptime!(residual) {
                    rv = f32::cast_from(resid[r * resid.stride(0) + col * resid.stride(1)]);
                }
                out[r * n + col] = epilogue::<F>(g, u, rv, glu, sigmoid, residual);
            }
        }
        if tid == 0 {
            counters[tile].store(0i32);
        }
    }

    /// The dot product of one feature's packed row (`row`, `srow` offsets) with staged row `r`.
    #[cube]
    #[allow(clippy::too_many_arguments)]
    fn qdot<Q: Size>(
        packed: &Tensor<Vector<i32, Q>>,
        scales: &Tensor<f32>,
        xs: &Shared<[f32]>,
        feature: usize,
        k_len: usize,
        words: usize,
        stride: usize,
        r: usize,
        #[comptime] bits: u32,
        #[comptime] group_words: usize,
    ) -> f32 {
        let v = packed.vector_size().comptime();
        let lane = UNIT_POS_X as usize;
        let per = comptime!(32 / bits) as usize;
        let mask = comptime!((1u32 << bits) - 1) as i32;
        let offset = comptime!(1u32 << (bits - 1)) as f32;
        let row = feature * packed.stride(0);
        let srow = feature * scales.stride(0);
        let mut acc = 0.0f32;
        let mut w = lane * v;
        while w < words {
            let block = packed[(row + w) / v];
            let scale = scales[srow + w / group_words];
            let mut sum = 0.0f32;
            #[unroll]
            for i in 0..v {
                let word = block.extract(i);
                let k0 = (w + i) * per;
                #[unroll]
                for j in 0..per {
                    let kk = k0 + j;
                    if kk < k_len {
                        let code =
                            f32::cast_from((word >> comptime!(j as u32 * bits) as i32) & mask);
                        sum += (code - offset) * xs[r * stride + kk + kk / 32];
                    }
                }
            }
            acc += sum * scale;
            w += PLANE * v;
        }
        plane_sum(acc)
    }

    /// Quantised fused GEMV: plane `p` of cube `c` computes output feature `c * PLANES + p`
    /// (with `glu`, from packed features `f` and `f + n`). The row is staged whole in shared
    /// memory (skewed as in `kernels::quant`), normed with `norm`; `out` as for `dense_kernel`.
    #[cube(launch_unchecked)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn quant_kernel<F: Float, Q: Size>(
        x: &Tensor<F>,
        delta: &Tensor<F>,
        gamma: &Tensor<F>,
        resid: &Tensor<F>,
        mulby: &Tensor<F>,
        packed: &Tensor<Vector<i32, Q>>,
        scales: &Tensor<f32>,
        out: &mut Tensor<F>,
        k_len: usize,
        n: usize,
        words: usize,
        eps: f32,
        #[comptime] bits: u32,
        #[comptime] rows: usize,
        #[comptime] group_words: usize,
        #[comptime] norm: bool,
        #[comptime] add: bool,
        #[comptime] glu: bool,
        #[comptime] residual: bool,
        #[comptime] mul: bool,
        #[comptime] sigmoid: bool,
        #[define(F)] _dtype: ElemType,
    ) {
        let tid = UNIT_POS as usize;
        let feature = CUBE_POS_X as usize * PLANES + UNIT_POS_Y as usize;
        let mut part_red = Shared::<[f32]>::new_slice(CUBE / 32);
        let mut xs = Shared::<[f32]>::new_slice(QX);
        let stride = k_len + k_len / 32 + 1;
        #[unroll]
        for r in 0..rows {
            let mut rms = 1.0f32;
            if comptime!(norm) {
                rms = row_rms::<F>(x, delta, r, k_len, eps, &mut part_red, add);
            }
            let mut kk = tid;
            while kk < k_len {
                let sv = input(x, delta, r, kk, add);
                let mut val = f32::cast_from(sv);
                if comptime!(norm) {
                    val = normed::<F>(sv, rms, gamma[kk * gamma.stride(0)]);
                }
                if comptime!(mul) {
                    let m = f32::cast_from(mulby[r * mulby.stride(0) + kk * mulby.stride(1)]);
                    val = f32::cast_from(F::cast_from(val * m));
                }
                xs[r * stride + kk + kk / 32] = val;
                if comptime!(add) {
                    if CUBE_POS_X == 0 {
                        out[rows * n + r * k_len + kk] = sv;
                    }
                }
                kk += CUBE;
            }
        }
        sync_cube();
        if feature >= n {
            terminate!();
        }
        #[unroll]
        for r in 0..rows {
            let g =
                qdot::<Q>(packed, scales, &xs, feature, k_len, words, stride, r, bits, group_words);
            let mut u = 0.0f32;
            if comptime!(glu) {
                u = qdot::<Q>(
                    packed,
                    scales,
                    &xs,
                    feature + n,
                    k_len,
                    words,
                    stride,
                    r,
                    bits,
                    group_words,
                );
            }
            if UNIT_POS_X == 0 {
                let mut rv = 0.0f32;
                if comptime!(residual) {
                    rv = f32::cast_from(resid[r * resid.stride(0) + feature * resid.stride(1)]);
                }
                out[r * n + feature] = epilogue::<F>(g, u, rv, glu, sigmoid, residual);
            }
        }
    }
}

fn elem(dtype: DType) -> ElemType {
    match dtype {
        DType::F32 => ElemType::Float(FloatKind::F32),
        DType::F16 => ElemType::Float(FloatKind::F16),
        DType::BF16 => ElemType::Float(FloatKind::BF16),
        other => panic!("the fused MLP takes f32, f16 or bf16, got {other:?}"),
    }
}

/// What a fused launch does around its GEMV.
#[derive(Clone, Copy, Debug)]
pub struct Fusion {
    /// RMSNorm the input (`gamma`, `eps`).
    pub norm: bool,
    /// Add `delta` to the input first (and write the sum).
    pub add: bool,
    /// SwiGLU over the weight's two halves.
    pub glu: bool,
    /// Add `resid` to the output.
    pub residual: bool,
    /// Multiply the input by `mulby` elementwise first (an output gate).
    pub mul: bool,
    /// Apply the logistic sigmoid to the output (a gate projection).
    pub sigmoid: bool,
    pub eps: f64,
    /// q8/q4 weights: bits and words per scale group (0 for dense).
    pub bits: u32,
    pub group_words: usize,
    /// Output width.
    pub n: usize,
}

#[backend_extension(Cube)]
pub trait NcmFusedMlp: Backend {
    /// See [`mlp_in`] and [`mlp_out`] (dense weights).
    #[allow(clippy::too_many_arguments)]
    fn ncm_fmlp_dense(
        x: FloatTensor<Self>,
        delta: FloatTensor<Self>,
        gamma: FloatTensor<Self>,
        resid: FloatTensor<Self>,
        mulby: FloatTensor<Self>,
        w: FloatTensor<Self>,
        counters: IntTensor<Self>,
        fusion: Fusion,
    ) -> FloatTensor<Self>;

    /// See [`mlp_in`] and [`mlp_out`] (q8/q4 weights).
    #[allow(clippy::too_many_arguments)]
    fn ncm_fmlp_quant(
        x: FloatTensor<Self>,
        delta: FloatTensor<Self>,
        gamma: FloatTensor<Self>,
        resid: FloatTensor<Self>,
        mulby: FloatTensor<Self>,
        packed: IntTensor<Self>,
        scales: FloatTensor<Self>,
        fusion: Fusion,
    ) -> FloatTensor<Self>;
}

/// Cubes per SM the dense grids aim for (as `kernels::gemv`).
const WAVES: usize = 2;

fn flat_out(x: &CubeTensor, len: usize) -> CubeTensor {
    empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), vec![len].into(), x.dtype)
}

impl NcmFusedMlp for CubeBackend {
    fn ncm_fmlp_dense(
        x: CubeTensor,
        delta: CubeTensor,
        gamma: CubeTensor,
        resid: CubeTensor,
        mulby: CubeTensor,
        w: CubeTensor,
        counters: CubeTensor,
        f: Fusion,
    ) -> CubeTensor {
        use kernel::{COLS, CUBE, LANES, MAX_K};
        let [rows, k] = x.meta.shape().dims::<2>();
        let n = f.n;
        let strides = w.meta.strides().to_vec();
        debug_assert_eq!(strides[1], 1, "a QLinear weight is row-major");
        // 16-byte vectors of columns where the width allows, else 8, 4 or 2 bytes (v = 1).
        let widest = 16 / x.dtype.size();
        let v = [widest, widest / 2, widest / 4, widest / 8]
            .into_iter()
            .find(|&v| v >= 1 && n.is_multiple_of(v) && strides[0].is_multiple_of(v))
            .unwrap_or(1);
        let client = x.client.clone();
        let sms = client.properties().hardware.num_streaming_multiprocessors.unwrap_or(84);
        let tiles = n.div_ceil(COLS * v);
        let want = (WAVES * sms as usize).div_ceil(tiles);
        let most = k.div_ceil(LANES * 4).max(1);
        let splits = want.clamp(1, most).max(k.div_ceil(MAX_K));
        let k_split = k.div_ceil(splits);
        let splits = k.div_ceil(k_split);
        let out = flat_out(&x, rows * n + if f.add { rows * k } else { 0 });
        let part = empty_device_contiguous_dtype(
            client.clone(),
            x.device.clone(),
            vec![if splits > 1 { splits * rows * 2 * n } else { 1 }].into(),
            DType::F32,
        );
        let dtype = elem(x.dtype);
        unsafe {
            kernel::dense_kernel::launch_unchecked(
                &client,
                CubeCount::Static(tiles as u32, splits as u32, 1),
                CubeDim::new_1d(CUBE as u32),
                v,
                x.into_tensor_arg(),
                delta.into_tensor_arg(),
                gamma.into_tensor_arg(),
                resid.into_tensor_arg(),
                mulby.into_tensor_arg(),
                w.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                part.into_tensor_arg(),
                counters.into_tensor_arg(),
                k,
                n,
                k_split,
                splits,
                f.eps as f32,
                rows,
                f.norm,
                f.add,
                f.glu,
                f.residual,
                f.mul,
                f.sigmoid,
                dtype,
            );
        }
        out
    }

    fn ncm_fmlp_quant(
        x: CubeTensor,
        delta: CubeTensor,
        gamma: CubeTensor,
        resid: CubeTensor,
        mulby: CubeTensor,
        packed: CubeTensor,
        scales: CubeTensor,
        f: Fusion,
    ) -> CubeTensor {
        use kernel::{PLANE, PLANES};
        let [rows, k] = x.meta.shape().dims::<2>();
        let words = packed.meta.shape().dims::<2>()[1];
        let vec = if f.bits == 4 { 2 } else { 4 };
        let out = flat_out(&x, rows * f.n + if f.add { rows * k } else { 0 });
        let client = x.client.clone();
        let dtype = elem(x.dtype);
        unsafe {
            kernel::quant_kernel::launch_unchecked(
                &client,
                CubeCount::Static(f.n.div_ceil(PLANES) as u32, 1, 1),
                CubeDim::new_2d(PLANE as u32, PLANES as u32),
                vec,
                x.into_tensor_arg(),
                delta.into_tensor_arg(),
                gamma.into_tensor_arg(),
                resid.into_tensor_arg(),
                mulby.into_tensor_arg(),
                packed.into_tensor_arg(),
                scales.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                k,
                f.n,
                words,
                f.eps as f32,
                f.bits,
                rows,
                f.group_words,
                f.norm,
                f.add,
                f.glu,
                f.residual,
                f.mul,
                f.sigmoid,
                dtype,
            );
        }
        out
    }
}

thread_local! {
    /// Per thread (cubecl streams are per thread, and launches on one stream never overlap):
    /// the split-K arrival counters of the dense fused MLP kernel and `kernels::gemv`, each
    /// left at 0 by the cube that used it.
    static COUNTERS: RefCell<Option<Tensor<1, Int>>> = const { RefCell::new(None) };
}

/// Column tiles a dense launch may count (far more than any of the models' widths needs).
const MAX_TILES: usize = 4096;

pub(crate) fn counters(device: &burn::tensor::Device) -> Tensor<1, Int> {
    COUNTERS.with(|c| {
        c.borrow_mut()
            .get_or_insert_with(|| Tensor::<1, Int>::zeros([MAX_TILES], device).cast(IntDType::I32))
            .clone()
    })
}

/// How a linear layer's weight reaches the fused kernels.
#[allow(clippy::large_enum_variant)]
enum Weight {
    Dense(Tensor<2>),
    Quant { packed: Tensor<2, Int>, scales: Tensor<2>, bits: u32, group_words: usize },
}

/// The layer's weight if the fused kernels take it for `rows` rows of `x` `[rows, d_in]`.
fn weight(layer: &QLinear, rows: usize, d_in: usize, out_cols: usize) -> Option<Weight> {
    if rows == 0 || rows > kernel::MAX_ROWS {
        return None;
    }
    match layer.kernel_weight() {
        crate::quant::KernelWeight::Dense(w) => {
            // `[d_in, d_out]`, row-major as `QLinear` stores it.
            let [kk, nw] = w.dims();
            (kk == d_in && nw >= out_cols).then_some(Weight::Dense(w))
        }
        crate::quant::KernelWeight::Quant { packed, scales, format, d_in: k } => {
            let words = packed.dims()[1];
            let group_words = format.group_len(k) / format.per_word();
            let vec = if format.bits == 4 { 2 } else { 4 };
            let fits = rows * (k + k / 32 + 1) <= kernel::QX
                && words.is_multiple_of(vec)
                && group_words.is_multiple_of(vec);
            fits.then_some(Weight::Quant { packed, scales, bits: format.bits, group_words })
        }
    }
}

/// One fused launch: `out` `[rows, n]` and with `add` the sum `[rows, k]`.
#[allow(clippy::too_many_arguments)]
fn launch(
    x: Tensor<2>,
    delta: Option<Tensor<2>>,
    gamma: Option<Tensor<1>>,
    resid: Option<Tensor<2>>,
    mulby: Option<Tensor<2>>,
    weight: Weight,
    fusion: Fusion,
) -> (Tensor<2>, Option<Tensor<2>>) {
    let [rows, k] = x.dims();
    let n = fusion.n;
    let device = x.device();
    let delta = delta.unwrap_or_else(|| x.clone());
    let gamma = gamma.unwrap_or_else(|| x.clone().reshape([rows * k]));
    let resid = resid.unwrap_or_else(|| x.clone());
    let mulby = mulby.unwrap_or_else(|| x.clone());
    let out = match weight {
        Weight::Dense(w) => <Dispatch as NcmFusedMlp>::ncm_fmlp_dense(
            x.into_dispatch(),
            delta.into_dispatch(),
            gamma.into_dispatch(),
            resid.into_dispatch(),
            mulby.into_dispatch(),
            w.into_dispatch(),
            counters(&device).into_dispatch(),
            fusion,
        ),
        Weight::Quant { packed, scales, bits, group_words } => {
            let fusion = Fusion { bits, group_words, ..fusion };
            <Dispatch as NcmFusedMlp>::ncm_fmlp_quant(
                x.into_dispatch(),
                delta.into_dispatch(),
                gamma.into_dispatch(),
                resid.into_dispatch(),
                mulby.into_dispatch(),
                packed.into_dispatch(),
                scales.into_dispatch(),
                fusion,
            )
        }
    };
    let out = Tensor::<1>::from_dispatch(out);
    let y = out.clone().narrow(0, 0, rows * n).reshape([rows, n]);
    let sum = fusion.add.then(|| out.narrow(0, rows * n, rows * k).reshape([rows, k]));
    (y, sum)
}

/// The MLP's first half on `x` `[rows, d]` (plus `delta` when given), normed by `gamma`, `eps`:
/// `(silu(gate) * up, x + delta)` with `gate_up` `[d, 2 ff]`. None when the fused kernels do
/// not take this layer or row count (the caller runs the unfused ops).
pub fn mlp_in(
    x: Tensor<2>,
    delta: Option<Tensor<2>>,
    gamma: Tensor<1>,
    eps: f64,
    gate_up: &QLinear,
) -> Option<(Tensor<2>, Option<Tensor<2>>)> {
    let [rows, d] = x.dims();
    let (d_in, d_out) = gate_up.dims();
    if d_in != d || d_out % 2 != 0 {
        return None;
    }
    let ff = d_out / 2;
    let weight = weight(gate_up, rows, d, ff)?;
    let add = delta.is_some();
    let fusion = Fusion {
        norm: true,
        add,
        glu: true,
        residual: false,
        mul: false,
        sigmoid: false,
        eps,
        bits: 0,
        group_words: 0,
        n: ff,
    };
    Some(launch(x, delta, Some(gamma), None, None, weight, fusion))
}

/// The MLP's second half: `resid + h W_down` for `h` `[rows, ff]`. None when the fused kernels
/// do not take this layer or row count.
pub fn mlp_out(h: Tensor<2>, down: &QLinear, resid: Tensor<2>) -> Option<Tensor<2>> {
    let [rows, ff] = h.dims();
    let (d_in, d) = down.dims();
    if d_in != ff {
        return None;
    }
    let weight = weight(down, rows, ff, d)?;
    let fusion = Fusion {
        norm: false,
        add: false,
        glu: false,
        residual: true,
        mul: false,
        sigmoid: false,
        eps: 0.0,
        bits: 0,
        group_words: 0,
        n: d,
    };
    Some(launch(h, None, None, Some(resid), None, weight, fusion).0)
}

/// A projection of the RMS-normed `x` `[rows, d]`: `norm(x) W`, with the norm in the GEMV's
/// prologue, and with `sigmoid` the logistic function in its epilogue (an output-gate
/// projection). The pre-attention half of a block: DeltaNet's and retrieval's input
/// projections, attention's QKV and gate projections. None when the fused kernels do not take
/// this layer or row count.
pub fn norm_linear(
    x: Tensor<2>,
    gamma: Tensor<1>,
    eps: f64,
    layer: &QLinear,
    sigmoid: bool,
) -> Option<Tensor<2>> {
    let [rows, d] = x.dims();
    let (d_in, n) = layer.dims();
    if d_in != d {
        return None;
    }
    let weight = weight(layer, rows, d, n)?;
    let fusion = Fusion {
        norm: true,
        add: false,
        glu: false,
        residual: false,
        mul: false,
        sigmoid,
        eps,
        bits: 0,
        group_words: 0,
        n,
    };
    Some(launch(x, None, Some(gamma), None, None, weight, fusion).0)
}

/// An output projection of the gated attention output: `(x * gate) W` for `x`, `gate`
/// `[rows, d_in]`, the product in the GEMV's prologue. None when the fused kernels do not take
/// this layer or row count.
pub fn gated_linear(x: Tensor<2>, gate: Tensor<2>, layer: &QLinear) -> Option<Tensor<2>> {
    let [rows, d] = x.dims();
    let (d_in, n) = layer.dims();
    if d_in != d || gate.dims() != [rows, d] {
        return None;
    }
    let weight = weight(layer, rows, d, n)?;
    let fusion = Fusion {
        norm: false,
        add: false,
        glu: false,
        residual: false,
        mul: true,
        sigmoid: false,
        eps: 0.0,
        bits: 0,
        group_words: 0,
        n,
    };
    Some(launch(x, None, None, None, Some(gate), weight, fusion).0)
}
