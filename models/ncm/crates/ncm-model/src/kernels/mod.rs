//! Fused CubeCL kernels for the host-bound decode paths (cargo feature `cuda`).
//!
//! Each kernel matches a reference path in `delta.rs`, `retrieval.rs` or the linear layers
//! ([`crate::quant::QLinear`]: dense and quantised matrix-vector products) and is exposed to the
//! layers through a Burn backend extension routed to the cubecl backend. The layers call them only
//! where [`crate::fused::active`] says so; everything else runs the reference tensor code.

mod argmax;
mod assign;
mod attend;
mod delta;
mod gemv;
pub mod hybrid;
mod norm;
pub(crate) mod quant;
mod retrieval;

use burn::backend::tensor::{BoolTensor, FloatTensor, IntTensor};
use burn::backend::{Backend, Dispatch, backend_extension};
use burn::tensor::{Bool, DType, Int, Tensor};
use burn_cubecl::cubecl::prelude::*;
use burn_cubecl::cubecl::{CubeCount, CubeDim};
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{CubeBackend, cubecl};

/// [`norm::rms_norm_kernel`] over `x` `[rows, d]`, after adding `delta` when given.
fn launch_rms_norm(
    x: CubeTensor,
    delta: Option<CubeTensor>,
    gamma: CubeTensor,
    eps: f64,
) -> CubeTensor {
    let [rows, d] = x.meta.shape().dims::<2>();
    let add = delta.is_some();
    let out = empty_like(&x, if add { vec![2, rows, d] } else { vec![rows, d] });
    let units = d.div_ceil(4).next_power_of_two().clamp(32, norm::MAX_UNITS);
    let dtype = elem(x.dtype);
    let client = x.client.clone();
    let delta = delta.unwrap_or_else(|| x.clone());
    unsafe {
        norm::rms_norm_kernel::launch_unchecked(
            &client,
            CubeCount::Static(rows as u32, 1, 1),
            CubeDim::new_1d(units as u32),
            x.into_tensor_arg(),
            delta.into_tensor_arg(),
            gamma.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            d,
            if add { rows * d } else { 0 },
            eps as f32,
            add,
            dtype,
        );
    }
    out
}

/// Cubes per SM the GEMV grids aim for.
const GEMV_WAVES: usize = 4;

/// The element type a kernel is defined for, from a tensor's dtype.
fn elem(dtype: DType) -> ElemType {
    match dtype {
        DType::F32 => ElemType::Float(cubecl::ir::FloatKind::F32),
        DType::F16 => ElemType::Float(cubecl::ir::FloatKind::F16),
        DType::BF16 => ElemType::Float(cubecl::ir::FloatKind::BF16),
        DType::I32 => ElemType::Int(cubecl::ir::IntKind::I32),
        other => panic!("fused kernels take f32, f16, bf16 or i32 tensors, got {other:?}"),
    }
}

/// A new contiguous tensor (plain `empty` may pitch rows) on `t`'s device at its dtype. The
/// kernels index their outputs as contiguous and their inputs through strides.
fn empty_like(t: &CubeTensor, shape: Vec<usize>) -> CubeTensor {
    empty_device_contiguous_dtype(t.client.clone(), t.device.clone(), shape.into(), t.dtype)
}

/// A new contiguous scratch tensor on `t`'s device.
fn scratch(t: &CubeTensor, shape: Vec<usize>, dtype: DType) -> CubeTensor {
    empty_device_contiguous_dtype(t.client.clone(), t.device.clone(), shape.into(), dtype)
}

/// Shape and selection settings of a sparse retrieval layer (`RetrievalSpec`'s fields).
#[derive(Clone, Copy, Debug)]
pub struct RetrievalDims {
    pub heads: usize,
    pub nope_dim: usize,
    pub rope_dim: usize,
    pub v_dim: usize,
    pub latent: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub block: usize,
    pub topk: usize,
    pub local: usize,
    pub dense: bool,
    /// Indexer keys as 8-bit codes and one scale per position.
    pub index_8bit: bool,
    pub eps: f64,
}

impl RetrievalDims {
    /// Offsets in the input projection `[q | latent, RoPE key | gate | indexer q | indexer k |
    /// indexer weights]`: (gate, indexer q, indexer k).
    fn offsets(&self) -> (usize, usize, usize) {
        let gate = self.heads * (self.nope_dim + self.rope_dim) + self.latent + self.rope_dim;
        let index_q = gate + self.heads * self.v_dim;
        (gate, index_q, index_q + self.index_heads * self.index_dim)
    }

    /// Grid sizes that serve every position of a cache of `capacity`: the most candidate
    /// blocks; the query heads per attention cube; and the attention chunks and tiles per chunk
    /// covering the most attended slots (dense steps see `pos + 1` slots, which is at most
    /// `local + topk * block` until a step turns sparse). Short steps spread over many cubes
    /// (fewer heads each); long ones read each cached row once for every head and take two
    /// tiles per cube, so the combine merges half the partials (measured best on an RTX 5080).
    fn grids(&self, capacity: usize) -> RetrievalGrid {
        let last = capacity.saturating_sub(1);
        let blocks = if last >= self.local { (last - self.local) / self.block + 1 } else { 0 };
        let slots =
            if self.dense { capacity } else { capacity.min(self.topk * self.block + self.local) };
        let tiles = slots.div_ceil(retrieval::CHUNK).max(1);
        let long = tiles >= retrieval::LONG_TILES;
        let want = if long { retrieval::HEAD_GROUP_LONG } else { retrieval::HEAD_GROUP };
        // The largest head group up to `want` that splits the heads evenly.
        let heads_per =
            (1..=want.min(self.heads)).rev().find(|g| self.heads.is_multiple_of(*g)).unwrap_or(1);
        let per = if long { 2 } else { 1 };
        RetrievalGrid { blocks: blocks.max(1), chunks: tiles.div_ceil(per), tiles: per, heads_per }
    }
}

/// Launch shape of one retrieval decode step (see [`RetrievalDims::grids`]).
struct RetrievalGrid {
    /// Candidate blocks the indexer may score.
    blocks: usize,
    /// Attention cubes per head group.
    chunks: usize,
    /// Tiles of `retrieval::CHUNK` slots per attention cube.
    tiles: usize,
    /// Query heads per attention cube.
    heads_per: usize,
}

#[backend_extension(Cube)]
pub trait NcmKernels: Backend {
    /// See [`argmax_last`].
    fn ncm_argmax(x: FloatTensor<Self>) -> IntTensor<Self>;

    /// See [`gemv`].
    fn ncm_gemv(
        x: FloatTensor<Self>,
        w: FloatTensor<Self>,
        counters: IntTensor<Self>,
    ) -> FloatTensor<Self>;

    /// See [`qmatmul`]; `x` `[rows, d_in]` with `rows <= GEMV_ROWS`.
    fn ncm_qgemv(
        x: FloatTensor<Self>,
        packed: IntTensor<Self>,
        scales: FloatTensor<Self>,
        d_in: usize,
        bits: u32,
        group_words: usize,
    ) -> FloatTensor<Self>;

    /// See [`qmatmul`]; the decoded weight `[d_in, d_out]` at `dtype`.
    fn ncm_dequant(
        packed: IntTensor<Self>,
        scales: FloatTensor<Self>,
        d_in: usize,
        bits: u32,
        group_words: usize,
        dtype: DType,
    ) -> FloatTensor<Self>;

    /// See [`packed`].
    fn ncm_packed(x: FloatTensor<Self>) -> FloatTensor<Self>;

    /// See [`assign_int_into`].
    fn ncm_assign_int(
        dst: IntTensor<Self>,
        src: IntTensor<Self>,
        axis: usize,
        at: usize,
    ) -> IntTensor<Self>;

    /// See [`assign_into`].
    fn ncm_assign(
        dst: FloatTensor<Self>,
        src: FloatTensor<Self>,
        axis: usize,
        at: usize,
    ) -> FloatTensor<Self>;

    /// See [`rms_norm`]: `x` `[rows, d]` to `[rows, d]`.
    fn ncm_rms_norm(x: FloatTensor<Self>, gamma: FloatTensor<Self>, eps: f64) -> FloatTensor<Self>;

    /// See [`add_rms_norm`]: `x`, `delta` `[rows, d]` to `[2, rows, d]` (the sum, then the
    /// normed sum).
    fn ncm_add_rms_norm(
        x: FloatTensor<Self>,
        delta: FloatTensor<Self>,
        gamma: FloatTensor<Self>,
        eps: f64,
    ) -> FloatTensor<Self>;

    /// See [`assign_into_at`].
    fn ncm_assign_at(
        dst: FloatTensor<Self>,
        src: FloatTensor<Self>,
        axis: usize,
        at: IntTensor<Self>,
    ) -> FloatTensor<Self>;

    /// See [`delta_step`].
    #[allow(clippy::too_many_arguments)]
    fn ncm_delta_step(
        proj: FloatTensor<Self>,
        conv_w: FloatTensor<Self>,
        conv: FloatTensor<Self>,
        state: FloatTensor<Self>,
        a_log: FloatTensor<Self>,
        dt_bias: FloatTensor<Self>,
        norm_w: FloatTensor<Self>,
        heads: usize,
        eps: f64,
    ) -> FloatTensor<Self>;

    /// See [`delta_scan`]: returns `[out, new_state]` concatenated along the rows (`[BH, N C +
    /// dk, dv]`).
    #[allow(clippy::too_many_arguments)]
    fn ncm_delta_scan(
        values: FloatTensor<Self>,
        w: FloatTensor<Self>,
        qk: FloatTensor<Self>,
        qg: FloatTensor<Self>,
        k_out: FloatTensor<Self>,
        carry: FloatTensor<Self>,
        state: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    /// A new contiguous, uninitialised i32 tensor of `shape` on `like`'s device.
    fn ncm_scratch_i32(like: FloatTensor<Self>, shape: Vec<usize>) -> IntTensor<Self>;

    /// See [`retrieval_select`]; writes `picked` and `count` in place and returns `picked`.
    #[allow(clippy::too_many_arguments)]
    fn ncm_retrieval_select(
        raw: FloatTensor<Self>,
        w: FloatTensor<Self>,
        scales: FloatTensor<Self>,
        picked: IntTensor<Self>,
        count: IntTensor<Self>,
        start: usize,
        t0: usize,
        dims: RetrievalDims,
    ) -> IntTensor<Self>;

    /// See [`retrieval_prefill_attend`].
    fn ncm_retrieval_prefill_attend(
        q: FloatTensor<Self>,
        kv: FloatTensor<Self>,
        picked: IntTensor<Self>,
        count: IntTensor<Self>,
        start: usize,
        dims: RetrievalDims,
    ) -> FloatTensor<Self>;

    /// See [`attend_decode`].
    fn ncm_attend_decode(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    /// See [`attend_decode`], with a mask.
    fn ncm_attend_decode_masked(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        t: usize,
    ) -> FloatTensor<Self>;

    /// See [`retrieval_decode`].
    #[allow(clippy::too_many_arguments)]
    fn ncm_retrieval_decode(
        proj: FloatTensor<Self>,
        q_gamma: FloatTensor<Self>,
        kv_gamma: FloatTensor<Self>,
        rope_gamma: FloatTensor<Self>,
        index_gamma: FloatTensor<Self>,
        cos: FloatTensor<Self>,
        sin: FloatTensor<Self>,
        wuk: FloatTensor<Self>,
        wuv: FloatTensor<Self>,
        kv: FloatTensor<Self>,
        keys: FloatTensor<Self>,
        codes: IntTensor<Self>,
        scales: FloatTensor<Self>,
        position: IntTensor<Self>,
        dims: RetrievalDims,
    ) -> FloatTensor<Self>;
}

impl NcmKernels for CubeBackend {
    fn ncm_gemv(x: CubeTensor, w: CubeTensor, counters: CubeTensor) -> CubeTensor {
        use gemv::{COLS, CUBE};
        let [rows, k] = x.meta.shape().dims::<2>();
        let [_, n] = w.meta.shape().dims::<2>();
        let y = empty_like(&x, vec![rows, n]);
        let dtype = elem(x.dtype);
        let client = x.client.clone();
        let strides = w.meta.strides().to_vec();
        if strides[1] != 1 && strides[0] == 1 {
            // Transposed weight: one plane per output column.
            let planes = CUBE / client.properties().hardware.plane_size_max as usize;
            unsafe {
                gemv::gemv_rows_kernel::launch_unchecked(
                    &client,
                    CubeCount::Static(n.div_ceil(planes) as u32, 1, 1),
                    CubeDim::new_1d(CUBE as u32),
                    x.into_tensor_arg(),
                    w.into_tensor_arg(),
                    y.clone().into_tensor_arg(),
                    k,
                    n,
                    rows,
                    dtype,
                );
            }
            return y;
        }
        // Row-major weight: vector loads of consecutive columns, 16 bytes when the width allows
        // (else 8 or 4).
        let widest = 16 / x.dtype.size();
        let fits = |v: usize| n.is_multiple_of(v) && strides[0].is_multiple_of(v);
        let v =
            [widest, widest / 2, widest / 4, widest / 8].into_iter().find(|&v| v >= 1 && fits(v));
        if let (1, Some(v)) = (strides[1], v) {
            use gemv::{VEC_COLS, VEC_K, VEC_LANES};
            let sms = client.properties().hardware.num_streaming_multiprocessors.unwrap_or(84);
            let tiles = n.div_ceil(VEC_COLS * v);
            // About GEMV_WAVES cubes per SM, each lane at least a few K rows, at most VEC_K per cube.
            let want = (GEMV_WAVES * sms as usize).div_ceil(tiles);
            let most = k.div_ceil(VEC_LANES * 4).max(1);
            let splits = want.clamp(1, most).max(k.div_ceil(VEC_K));
            let k_split = k.div_ceil(splits);
            let splits = k.div_ceil(k_split);
            let part =
                scratch(&x, vec![if splits > 1 { splits * rows * n } else { 1 }], DType::F32);
            unsafe {
                gemv::gemv_vec_kernel::launch_unchecked(
                    &client,
                    CubeCount::Static(tiles as u32, splits as u32, 1),
                    CubeDim::new_1d(CUBE as u32),
                    v,
                    x.into_tensor_arg(),
                    w.into_tensor_arg(),
                    y.clone().into_tensor_arg(),
                    part.clone().into_tensor_arg(),
                    counters.into_tensor_arg(),
                    k,
                    n,
                    k_split,
                    rows,
                    splits > 1,
                    dtype,
                );
            }
            // With splits, the last cube of each tile combined them (no second launch).
            return y;
        }
        // A strided weight neither layout serves: the scalar column kernel (up to eight rows).
        assert!(rows <= gemv::COLS_ROWS, "a strided GEMV weight takes at most eight rows");
        // Enough cubes to keep every SM reading: split K when N alone gives too few tiles.
        let tiles = n.div_ceil(COLS);
        let mut splits = 168usize.div_ceil(tiles).clamp(1, k.div_ceil(COLS).max(1));
        splits = splits.max(k.div_ceil(512));
        let k_split = k.div_ceil(splits);
        let splits = k.div_ceil(k_split);
        let part = scratch(&x, vec![if splits > 1 { splits * rows * n } else { 1 }], DType::F32);
        unsafe {
            gemv::gemv_cols_kernel::launch_unchecked(
                &client,
                CubeCount::Static(tiles as u32, splits as u32, 1),
                CubeDim::new_1d(CUBE as u32),
                x.into_tensor_arg(),
                w.into_tensor_arg(),
                y.clone().into_tensor_arg(),
                part.clone().into_tensor_arg(),
                k,
                n,
                k_split,
                rows,
                splits > 1,
                dtype,
            );
            if splits > 1 {
                let count = rows * n;
                gemv::gemv_sum_kernel::launch_unchecked(
                    &client,
                    CubeCount::Static(count.div_ceil(CUBE) as u32, 1, 1),
                    CubeDim::new_1d(CUBE as u32),
                    part.into_tensor_arg(),
                    y.clone().into_tensor_arg(),
                    count,
                    splits,
                    dtype,
                );
            }
        }
        y
    }

    fn ncm_qgemv(
        x: CubeTensor,
        packed: CubeTensor,
        scales: CubeTensor,
        d_in: usize,
        bits: u32,
        group_words: usize,
    ) -> CubeTensor {
        use quant::{PLANE, PLANES};
        let rows = x.meta.shape().dims::<2>()[0];
        let [n_len, words] = packed.meta.shape().dims::<2>();
        let out = empty_like(&x, vec![rows, n_len]);
        let dtype = elem(x.dtype);
        let client = x.client.clone();
        let strides = packed.meta.strides().to_vec();
        // Any width (x staged a tile at a time), any word count (narrower vectors down to one
        // word) and any group length (per-word scales when vectors cross groups).
        let vec = quant::vector_words(bits, words, strides[0]);
        if rows <= crate::quant::GEMV_ROWS && strides[1] == 1 {
            unsafe {
                quant::qgemv_vec_kernel::launch_unchecked(
                    &client,
                    CubeCount::Static(n_len.div_ceil(quant::QPLANES) as u32, 1, 1),
                    CubeDim::new_2d(PLANE as u32, quant::QPLANES as u32),
                    vec,
                    x.into_tensor_arg(),
                    packed.into_tensor_arg(),
                    scales.into_tensor_arg(),
                    out.clone().into_tensor_arg(),
                    d_in,
                    n_len,
                    words,
                    quant::staged_tile(rows, d_in, quant::QX),
                    bits,
                    rows,
                    group_words,
                    quant::straddles(vec, group_words),
                    dtype,
                );
            }
            return out;
        }
        unsafe {
            quant::qgemv_kernel::launch_unchecked(
                &client,
                CubeCount::Static(n_len.div_ceil(PLANES) as u32, 1, 1),
                CubeDim::new_2d(PLANE as u32, PLANES as u32),
                x.into_tensor_arg(),
                packed.into_tensor_arg(),
                scales.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                d_in,
                n_len,
                words,
                bits,
                rows,
                group_words,
                dtype,
            );
        }
        out
    }

    fn ncm_dequant(
        packed: CubeTensor,
        scales: CubeTensor,
        d_in: usize,
        bits: u32,
        group_words: usize,
        dtype: DType,
    ) -> CubeTensor {
        use quant::DEQUANT_CUBE;
        let [n_len, words] = packed.meta.shape().dims::<2>();
        let out = scratch(&packed, vec![d_in, n_len], dtype);
        let client = packed.client.clone();
        unsafe {
            quant::dequant_kernel::launch_unchecked(
                &client,
                CubeCount::Static((n_len * words).div_ceil(DEQUANT_CUBE) as u32, 1, 1),
                CubeDim::new_1d(DEQUANT_CUBE as u32),
                packed.into_tensor_arg(),
                scales.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                d_in,
                n_len,
                words,
                bits,
                group_words,
                elem(dtype),
            );
        }
        out
    }

    fn ncm_assign_int(dst: CubeTensor, src: CubeTensor, axis: usize, at: usize) -> CubeTensor {
        Self::ncm_assign(dst, src, axis, at)
    }

    fn ncm_packed(x: CubeTensor) -> CubeTensor {
        if x.is_contiguous() {
            return x;
        }
        let out = empty_like(&x, x.meta.shape().to_vec());
        Self::ncm_assign(out, x, 0, 0)
    }

    fn ncm_assign(dst: CubeTensor, src: CubeTensor, axis: usize, at: usize) -> CubeTensor {
        let count = src.meta.shape().num_elements();
        if count == 0 {
            return dst;
        }
        let dtype = elem(dst.dtype);
        let client = dst.client.clone();
        unsafe {
            assign::assign_kernel::launch_unchecked(
                &client,
                CubeCount::Static(count.div_ceil(assign::CUBE) as u32, 1, 1),
                CubeDim::new_1d(assign::CUBE as u32),
                dst.clone().into_tensor_arg(),
                src.into_tensor_arg(),
                count,
                axis,
                at,
                dtype,
            );
        }
        dst
    }

    fn ncm_assign_at(dst: CubeTensor, src: CubeTensor, axis: usize, at: CubeTensor) -> CubeTensor {
        let count = src.meta.shape().num_elements();
        if count == 0 {
            return dst;
        }
        let dtype = elem(dst.dtype);
        let client = dst.client.clone();
        unsafe {
            assign::assign_at_kernel::launch_unchecked(
                &client,
                CubeCount::Static(count.div_ceil(assign::CUBE) as u32, 1, 1),
                CubeDim::new_1d(assign::CUBE as u32),
                dst.clone().into_tensor_arg(),
                src.into_tensor_arg(),
                at.into_tensor_arg(),
                count,
                axis,
                dtype,
            );
        }
        dst
    }

    fn ncm_rms_norm(x: CubeTensor, gamma: CubeTensor, eps: f64) -> CubeTensor {
        launch_rms_norm(x, None, gamma, eps)
    }

    fn ncm_add_rms_norm(
        x: CubeTensor,
        delta: CubeTensor,
        gamma: CubeTensor,
        eps: f64,
    ) -> CubeTensor {
        launch_rms_norm(x, Some(delta), gamma, eps)
    }

    fn ncm_argmax(x: CubeTensor) -> CubeTensor {
        let [rows, n] = x.meta.shape().dims::<2>();
        let out = scratch(&x, vec![rows, 1], DType::I32);
        let dtype = elem(x.dtype);
        let client = x.client.clone();
        unsafe {
            argmax::argmax_kernel::launch_unchecked(
                &client,
                CubeCount::Static(rows as u32, 1, 1),
                CubeDim::new_1d(argmax::CUBE as u32),
                x.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                n,
                dtype,
            );
        }
        out
    }

    fn ncm_delta_step(
        proj: CubeTensor,
        conv_w: CubeTensor,
        conv: CubeTensor,
        state: CubeTensor,
        a_log: CubeTensor,
        dt_bias: CubeTensor,
        norm_w: CubeTensor,
        heads: usize,
        eps: f64,
    ) -> CubeTensor {
        let [b, t, _] = proj.meta.shape().dims::<3>();
        let [_, _, hd, _] = state.meta.shape().dims::<4>();
        let width = conv_w.meta.shape().dims::<2>()[0];
        let out = empty_like(&proj, vec![b, t, heads * hd]);
        let dtype = elem(proj.dtype);
        let client = proj.client.clone();
        unsafe {
            delta::delta_step_kernel::launch_unchecked(
                &client,
                CubeCount::Static((b * heads) as u32, 1, 1),
                CubeDim::new_1d((hd * delta::STEP_SPLIT) as u32),
                proj.into_tensor_arg(),
                conv_w.into_tensor_arg(),
                a_log.into_tensor_arg(),
                dt_bias.into_tensor_arg(),
                norm_w.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                conv.into_tensor_arg(),
                state.into_tensor_arg(),
                t,
                1.0 / (hd as f32).sqrt(),
                1.0 / hd as f32,
                eps as f32,
                heads,
                hd,
                width,
                dtype,
            );
        }
        out
    }

    fn ncm_delta_scan(
        values: CubeTensor,
        w: CubeTensor,
        qk: CubeTensor,
        qg: CubeTensor,
        k_out: CubeTensor,
        carry: CubeTensor,
        state: CubeTensor,
    ) -> CubeTensor {
        use delta::{SCAN_COLUMNS, SCAN_CUBE};
        let [bh, rows, dv] = values.meta.shape().dims::<3>();
        let dk = w.meta.shape().dims::<3>()[2];
        let c = qk.meta.shape().dims::<3>()[2];
        let chunks = rows / c;
        let g = dv.min(SCAN_COLUMNS);
        assert!(dv.is_multiple_of(g) && SCAN_CUBE.is_multiple_of(g), "value width {dv}");
        let out = empty_like(&values, vec![bh, rows + dk, dv]);
        let dtype = elem(values.dtype);
        let client = values.client.clone();
        unsafe {
            delta::delta_scan_kernel::launch_unchecked(
                &client,
                CubeCount::Static((bh * dv / g) as u32, 1, 1),
                CubeDim::new_1d(SCAN_CUBE as u32),
                values.into_tensor_arg(),
                w.into_tensor_arg(),
                qk.into_tensor_arg(),
                qg.into_tensor_arg(),
                k_out.into_tensor_arg(),
                carry.into_tensor_arg(),
                state.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                chunks,
                dv / g,
                dk,
                dv,
                c,
                g,
                dtype,
            );
        }
        out
    }

    fn ncm_scratch_i32(like: CubeTensor, shape: Vec<usize>) -> CubeTensor {
        scratch(&like, shape, DType::I32)
    }

    fn ncm_retrieval_select(
        raw: CubeTensor,
        w: CubeTensor,
        scales: CubeTensor,
        picked: CubeTensor,
        count: CubeTensor,
        start: usize,
        t0: usize,
        dims: RetrievalDims,
    ) -> CubeTensor {
        use retrieval::CUBE;
        let [b, _, q, prefix] = raw.meta.shape().dims::<4>();
        let t_total = count.meta.shape().dims::<2>()[1];
        let nb = prefix.div_ceil(dims.block).max(1);
        let blocks = scratch(&raw, vec![b, q, nb], DType::F32);
        let dtype = elem(raw.dtype);
        let client = raw.client.clone();
        unsafe {
            retrieval::ret_select_kernel::launch_unchecked(
                &client,
                CubeCount::Static((b * q) as u32, 1, 1),
                CubeDim::new_1d(CUBE as u32),
                raw.into_tensor_arg(),
                w.into_tensor_arg(),
                scales.into_tensor_arg(),
                blocks.into_tensor_arg(),
                picked.clone().into_tensor_arg(),
                count.into_tensor_arg(),
                start,
                t0,
                q,
                t_total,
                nb,
                dims.topk,
                1.0 / ((dims.index_heads * dims.index_dim) as f32).sqrt(),
                dims.index_heads,
                dims.block,
                dims.local,
                dims.index_8bit,
                dtype,
            );
        }
        picked
    }

    fn ncm_retrieval_prefill_attend(
        q: CubeTensor,
        kv: CubeTensor,
        picked: CubeTensor,
        count: CubeTensor,
        start: usize,
        dims: RetrievalDims,
    ) -> CubeTensor {
        use retrieval::CUBE;
        let [b, h, t, _] = q.meta.shape().dims::<4>();
        let out = empty_like(&q, vec![b, t, h, dims.latent]);
        let dtype = elem(q.dtype);
        let client = q.client.clone();
        let plane = client.properties().hardware.plane_size_max as usize;
        assert_eq!(plane, retrieval::PREFILL_LANES, "the prefill kernel takes 32-lane planes");
        // The largest head group up to PREFILL_HEAD_GROUP that splits the heads evenly (16
        // heads at 1B: two groups of 8, each keeping 250M's register footprint).
        let hg = (1..=retrieval::PREFILL_HEAD_GROUP.min(h))
            .rev()
            .find(|g| h.is_multiple_of(*g))
            .unwrap_or(1);
        unsafe {
            retrieval::ret_prefill_attend_kernel::launch_unchecked(
                &client,
                CubeCount::Static((b * t * (h / hg)) as u32, 1, 1),
                CubeDim::new_1d(CUBE as u32),
                q.into_tensor_arg(),
                kv.into_tensor_arg(),
                picked.into_tensor_arg(),
                count.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                start,
                t,
                dims.topk,
                h,
                hg,
                dims.latent,
                dims.rope_dim,
                dims.block,
                dims.local,
                retrieval::PREFILL_TILE,
                dtype,
            );
        }
        out
    }

    fn ncm_attend_decode(q: CubeTensor, k: CubeTensor, v: CubeTensor) -> CubeTensor {
        launch_attend_decode(q, k, v, None, 1)
    }

    fn ncm_attend_decode_masked(
        q: CubeTensor,
        k: CubeTensor,
        v: CubeTensor,
        mask: CubeTensor,
        t: usize,
    ) -> CubeTensor {
        launch_attend_decode(q, k, v, Some(mask), t)
    }

    fn ncm_retrieval_decode(
        proj: CubeTensor,
        q_gamma: CubeTensor,
        kv_gamma: CubeTensor,
        rope_gamma: CubeTensor,
        index_gamma: CubeTensor,
        cos: CubeTensor,
        sin: CubeTensor,
        wuk: CubeTensor,
        wuv: CubeTensor,
        kv: CubeTensor,
        keys: CubeTensor,
        codes: CubeTensor,
        scales: CubeTensor,
        position: CubeTensor,
        dims: RetrievalDims,
    ) -> CubeTensor {
        use retrieval::{CUBE, INDEX_CUBE};
        let b = proj.meta.shape().dims::<3>()[0];
        let capacity = kv.meta.shape().dims::<3>()[1];
        let d = dims;
        let (h, dr, l) = (d.heads, d.rope_dim, d.latent);
        let (gate, index_q, index_k) = d.offsets();
        let RetrievalGrid { blocks, chunks, tiles, heads_per: hg } = d.grids(capacity);
        let plane = proj.client.properties().hardware.plane_size_max as usize;
        assert_eq!(plane, retrieval::LANES, "the decode kernels take 32-lane planes");
        let dtype = elem(proj.dtype);
        let client = proj.client.clone();
        let q_lat = scratch(&proj, vec![b, h, l], DType::F32);
        let q_rope = scratch(&proj, vec![b, h, dr], DType::F32);
        let scale = 1.0 / ((d.nope_dim + dr) as f32).sqrt();
        unsafe {
            retrieval::ret_prep_kernel::launch_unchecked(
                &client,
                CubeCount::Static((b * h * retrieval::PREP_SLICES + 3 * b) as u32, 1, 1),
                CubeDim::new_1d(CUBE as u32),
                proj.clone().into_tensor_arg(),
                q_gamma.into_tensor_arg(),
                kv_gamma.into_tensor_arg(),
                rope_gamma.into_tensor_arg(),
                index_gamma.into_tensor_arg(),
                cos.into_tensor_arg(),
                sin.into_tensor_arg(),
                wuk.into_tensor_arg(),
                q_lat.clone().into_tensor_arg(),
                q_rope.clone().into_tensor_arg(),
                kv.clone().into_tensor_arg(),
                keys.clone().into_tensor_arg(),
                codes.clone().into_tensor_arg(),
                scales.clone().into_tensor_arg(),
                position.clone().into_tensor_arg(),
                b,
                scale,
                d.eps as f32,
                1.0 / (d.nope_dim + dr) as f32,
                1.0 / l as f32,
                1.0 / dr as f32,
                1.0 / d.index_dim as f32,
                h,
                d.nope_dim,
                dr,
                l,
                d.index_dim,
                index_k,
                d.index_8bit,
                dtype,
            );
        }
        let picked = scratch(&proj, vec![b, d.topk], DType::U32);
        if !d.dense {
            let scores = scratch(&proj, vec![b, blocks], DType::F32);
            unsafe {
                retrieval::ret_index_kernel::launch_unchecked(
                    &client,
                    CubeCount::Static((b * blocks) as u32, 1, 1),
                    CubeDim::new_1d(INDEX_CUBE as u32),
                    proj.clone().into_tensor_arg(),
                    keys.into_tensor_arg(),
                    codes.into_tensor_arg(),
                    scales.into_tensor_arg(),
                    scores.clone().into_tensor_arg(),
                    position.clone().into_tensor_arg(),
                    blocks,
                    1.0 / ((d.index_heads * d.index_dim) as f32).sqrt(),
                    d.index_heads,
                    d.index_dim,
                    index_q,
                    d.block,
                    d.local,
                    d.index_8bit,
                    dtype,
                );
                retrieval::ret_topk_kernel::launch_unchecked(
                    &client,
                    CubeCount::Static(b as u32, 1, 1),
                    CubeDim::new_1d(CUBE as u32),
                    scores.into_tensor_arg(),
                    picked.clone().into_tensor_arg(),
                    position.clone().into_tensor_arg(),
                    blocks,
                    d.topk,
                    d.block,
                    d.local,
                );
            }
        }
        let part_max = scratch(&proj, vec![b, chunks, h], DType::F32);
        let part_sum = scratch(&proj, vec![b, chunks, h], DType::F32);
        let part_acc = scratch(&proj, vec![b, chunks, h, l], DType::F32);
        let out = empty_like(&proj, vec![b, 1, h * d.v_dim]);
        unsafe {
            retrieval::ret_attend_kernel::launch_unchecked(
                &client,
                CubeCount::Static((b * chunks * (h / hg)) as u32, 1, 1),
                CubeDim::new_1d(CUBE as u32),
                q_lat.into_tensor_arg(),
                q_rope.into_tensor_arg(),
                kv.into_tensor_arg(),
                picked.into_tensor_arg(),
                part_max.clone().into_tensor_arg(),
                part_sum.clone().into_tensor_arg(),
                part_acc.clone().into_tensor_arg(),
                position.clone().into_tensor_arg(),
                chunks,
                tiles,
                d.topk,
                h,
                dr,
                l,
                d.block,
                d.local,
                d.dense,
                hg,
                dtype,
            );
            retrieval::ret_combine_kernel::launch_unchecked(
                &client,
                CubeCount::Static((b * h * d.v_dim.div_ceil(retrieval::COMBINE_COLS)) as u32, 1, 1),
                CubeDim::new_1d(retrieval::COMBINE_CUBE as u32),
                proj.into_tensor_arg(),
                wuv.into_tensor_arg(),
                part_max.into_tensor_arg(),
                part_sum.into_tensor_arg(),
                part_acc.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                position.into_tensor_arg(),
                chunks,
                h,
                d.v_dim,
                l,
                gate,
                dtype,
            );
        }
        out
    }
}

/// Gated DeltaNet over `T` tokens from a cache, token by token in one kernel: `proj` `[B, T, W]`
/// is the layer's input projection, `conv` `[B, width - 1, 3 H hd]` and `state` `[B, H, hd, hd]`
/// the cache, both contiguous and updated in place (whoever else holds them, as a captured graph
/// does). Returns the gated, normed read-out `[B, T, H hd]` (before the output projection).
#[allow(clippy::too_many_arguments)]
pub fn delta_step(
    proj: Tensor<3>,
    conv_w: Tensor<2>,
    conv: &Tensor<3>,
    state: &Tensor<4>,
    a_log: Tensor<1>,
    dt_bias: Tensor<1>,
    norm_w: Tensor<1>,
    heads: usize,
    eps: f64,
) -> Tensor<3> {
    let out = <Dispatch as NcmKernels>::ncm_delta_step(
        proj.into_dispatch(),
        conv_w.into_dispatch(),
        conv.clone().into_dispatch(),
        state.clone().into_dispatch(),
        a_log.into_dispatch(),
        dt_bias.into_dispatch(),
        norm_w.into_dispatch(),
        heads,
        eps,
    );
    Tensor::from_dispatch(out)
}

/// The sequential part of the chunked gated delta rule in one kernel (see
/// `crate::delta::delta_rule_chunked`): `values` `[BH, N C, dv]`, `w`, `qg`, `k_out`
/// `[BH, N C, dk]`, `qk` `[BH, N C, C]`, `carry` `[BH, N]` and the initial `state`
/// `[BH, dk, dv]`. Returns the outputs `[BH, N C, dv]` and the final state `[BH, dk, dv]`.
pub fn delta_scan(
    values: Tensor<3>,
    w: Tensor<3>,
    qk: Tensor<3>,
    qg: Tensor<3>,
    k_out: Tensor<3>,
    carry: Tensor<2>,
    state: Tensor<3>,
) -> (Tensor<3>, Tensor<3>) {
    let rows = values.dims()[1];
    let dk = w.dims()[2];
    let both = <Dispatch as NcmKernels>::ncm_delta_scan(
        values.into_dispatch(),
        w.into_dispatch(),
        qk.into_dispatch(),
        qg.into_dispatch(),
        k_out.into_dispatch(),
        carry.into_dispatch(),
        state.into_dispatch(),
    );
    let both = Tensor::<3>::from_dispatch(both);
    (both.clone().narrow(1, 0, rows), both.narrow(1, rows, dk))
}

/// A new contiguous i32 tensor `[d0, d1, ..]` on `like`'s device, uninitialised (Burn's own
/// constructors may pitch rows, and the prefill kernels index these buffers as packed).
pub fn scratch_i32<const D: usize>(like: &Tensor<3>, shape: [usize; D]) -> Tensor<D, Int> {
    let t = <Dispatch as NcmKernels>::ncm_scratch_i32(like.clone().into_dispatch(), shape.to_vec());
    Tensor::from_dispatch(t)
}

/// Launches [`attend::attend_decode_kernel`] (see [`attend_decode`]).
fn launch_attend_decode(
    q: CubeTensor,
    k: CubeTensor,
    v: CubeTensor,
    mask: Option<CubeTensor>,
    t: usize,
) -> CubeTensor {
    let [b, kvh, rows, d] = q.meta.shape().dims::<4>();
    let masked = mask.is_some();
    // Without a mask the kernel reads none: any u8 tensor stands in.
    let mask = mask.unwrap_or_else(|| scratch(&q, vec![1, 1], DType::U8));
    let keys = k.meta.shape().dims::<4>()[2];
    let out = empty_like(&q, vec![b, kvh, rows, d]);
    let dtype = elem(q.dtype);
    let client = q.client.clone();
    let cubes = b * kvh * rows;
    // Split-KV: every split holds at most MAX_KEYS scores, and long caches split further
    // (splits of at least 256 keys) until about two cubes per SM are reading.
    let needed = keys.div_ceil(attend::MAX_KEYS);
    let wanted = 168usize.div_ceil(cubes).min(keys.div_ceil(256));
    let splits = needed.max(wanted).max(1);
    let split_len = keys.div_ceil(splits);
    let splits = keys.div_ceil(split_len);
    let part = scratch(&q, vec![if splits > 1 { splits * cubes * (d + 2) } else { 1 }], DType::F32);
    unsafe {
        attend::attend_decode_kernel::launch_unchecked(
            &client,
            CubeCount::Static(cubes as u32, splits as u32, 1),
            CubeDim::new_1d(attend::CUBE as u32),
            q.into_tensor_arg(),
            k.into_tensor_arg(),
            v.into_tensor_arg(),
            mask.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            part.clone().into_tensor_arg(),
            kvh,
            rows,
            keys,
            t,
            split_len,
            1.0 / (d as f32).sqrt(),
            d,
            masked,
            splits > 1,
            dtype,
        );
        if splits > 1 {
            let count = cubes * d;
            attend::attend_combine_kernel::launch_unchecked(
                &client,
                CubeCount::Static(count.div_ceil(attend::CUBE) as u32, 1, 1),
                CubeDim::new_1d(attend::CUBE as u32),
                part.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                count,
                splits,
                d,
                dtype,
            );
        }
    }
    out
}

/// Whether [`attend_decode`] takes `rows` query rows per key/value head of width `d` against
/// `keys` cached positions.
pub fn attend_decode_fits(rows: usize, d: usize, keys: usize) -> bool {
    // Any number of keys: past MAX_KEYS the kernel splits them (split-KV).
    rows <= 16 && d <= attend::CUBE && keys >= 1
}

/// Attention of a decode step in one kernel: `q` `[B, KVH, R, d]` (a group's query heads
/// folded into the rows, `R = G t`), `k`, `v` `[B, KVH, S, d]`, and optionally the `[t, S]`
/// mask (`true` hides a key) shared by the group's heads; returns
/// `softmax(q k^T / sqrt(d)) v` `[B, KVH, R, d]`. Only for shapes [`attend_decode_fits`]
/// accepts.
pub fn attend_decode(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    mask: Option<Tensor<2, Bool>>,
    t: usize,
) -> Tensor<4> {
    let (q, k, v) = (q.into_dispatch(), k.into_dispatch(), v.into_dispatch());
    let out = match mask {
        None => <Dispatch as NcmKernels>::ncm_attend_decode(q, k, v),
        Some(m) => {
            <Dispatch as NcmKernels>::ncm_attend_decode_masked(q, k, v, m.into_dispatch(), t)
        }
    };
    Tensor::from_dispatch(out)
}

/// Block selection of prefill queries `t0 .. t0 + q` of a chunk whose first query sits at
/// absolute position `start`, on the device: `raw` `[B, hi, q, S']` are the indexer dot products
/// over the candidate prefix (against the codes for 8-bit keys, whose per-position `scales`
/// `[B, S, 1]` apply after the ReLU; any float tensor otherwise), `w` `[B, T, hi]` the indexer
/// head weights of the whole chunk. Writes each query's picked blocks into `picked` `[B, T, topk]`
/// and their number into `count` `[B, T]` (both i32) in place.
#[allow(clippy::too_many_arguments)]
pub fn retrieval_select(
    raw: Tensor<4>,
    w: Tensor<3>,
    scales: Tensor<3>,
    picked: &Tensor<3, Int>,
    count: &Tensor<2, Int>,
    start: usize,
    t0: usize,
    dims: RetrievalDims,
) {
    drop(<Dispatch as NcmKernels>::ncm_retrieval_select(
        raw.into_dispatch(),
        w.into_dispatch(),
        scales.into_dispatch(),
        picked.clone().into_dispatch(),
        count.clone().into_dispatch(),
        start,
        t0,
        dims,
    ));
}

/// Sparse prefill attention of the queries `q` `[B, H, T, l + dr]` (absorbed, scale folded in)
/// at absolute positions `start ..` over the cache `kv` `[B, S, l + dr]`: each attends to its
/// `count` `picked` blocks (from [`retrieval_select`]) and its local window. Returns the
/// normalised latent sums `[B, T, H, l]` before the value up-projection.
pub fn retrieval_prefill_attend(
    q: Tensor<4>,
    kv: &Tensor<3>,
    picked: Tensor<3, Int>,
    count: Tensor<2, Int>,
    start: usize,
    dims: RetrievalDims,
) -> Tensor<4> {
    let out = <Dispatch as NcmKernels>::ncm_retrieval_prefill_attend(
        q.into_dispatch(),
        kv.clone().into_dispatch(),
        picked.into_dispatch(),
        count.into_dispatch(),
        start,
        dims,
    );
    Tensor::from_dispatch(out)
}

/// One-token sparse retrieval decode at the position held by the device tensor `position`
/// (`[1]`, i32) from the layer's input projection `proj` `[B, 1, W]`: writes the position's entry
/// into the cache in place (`kv` `[B, S, l + dr]` and the indexer `keys`, whoever else holds them),
/// selects blocks with the indexer on the device and attends over them and the local window (or
/// the whole context when it has no more candidates than `topk`, or in dense mode). Returns the
/// gated attention output `[B, 1, h dv]` before the output projection. `norms` are the q,
/// latent, RoPE-key and indexer-key RMSNorm weights; `rope` the cos and sin tables of the
/// position; `wuk`, `wuv` the latent up-projections. Every launch is the same for every
/// position, so the call can be captured into a CUDA graph.
#[allow(clippy::too_many_arguments)]
pub fn retrieval_decode(
    proj: Tensor<3>,
    norms: [Tensor<1>; 4],
    rope: (Tensor<4>, Tensor<4>),
    wuk: Tensor<2>,
    wuv: Tensor<2>,
    kv: &Tensor<3>,
    keys: &crate::retrieval::IndexKeys,
    position: &Tensor<1, Int>,
    dims: RetrievalDims,
) -> Tensor<3> {
    use crate::retrieval::IndexKeys;
    let [q, latent, rope_key, index] = norms;
    // The kernels read only the tensors of the cache's key format; the other arguments are
    // stand-ins of the right kind.
    let (keys, codes, scales) = match keys {
        IndexKeys::Float(k) => (k.clone(), position.clone().reshape([1, 1, 1]), k.clone()),
        IndexKeys::Int8 { codes, scales } => (scales.clone(), codes.clone(), scales.clone()),
    };
    let out = <Dispatch as NcmKernels>::ncm_retrieval_decode(
        proj.into_dispatch(),
        q.into_dispatch(),
        latent.into_dispatch(),
        rope_key.into_dispatch(),
        index.into_dispatch(),
        rope.0.into_dispatch(),
        rope.1.into_dispatch(),
        wuk.into_dispatch(),
        wuv.into_dispatch(),
        kv.clone().into_dispatch(),
        keys.into_dispatch(),
        codes.into_dispatch(),
        scales.into_dispatch(),
        position.clone().into_dispatch(),
        dims,
    );
    Tensor::from_dispatch(out)
}

/// `x.argmax(D - 1)` for `x` `[..., n]` in one kernel launch (one cube per row); the result has
/// int dtype i32, Burn's on CUDA.
pub fn argmax_last<const D: usize>(x: Tensor<D>) -> Tensor<D, Int> {
    let mut shape = x.dims();
    let n = shape[D - 1];
    let rows = shape.iter().product::<usize>() / n;
    let out = <Dispatch as NcmKernels>::ncm_argmax(x.reshape([rows, n]).into_dispatch());
    shape[D - 1] = 1;
    Tensor::<2, Int>::from_dispatch(out).reshape(shape)
}

/// Writes `src` into `dst` in place at offset `at` along `axis` (`dst[.., at + i, ..] =
/// src[.., i, ..]`), whoever else holds `dst`'s buffer: a captured graph keeps reading the same
/// buffer, where `slice_assign` would copy it away. The caller answers for any other holder that
/// expects the old values.
pub fn assign_into<const D: usize>(dst: &Tensor<D>, src: Tensor<D>, axis: usize, at: usize) {
    let dst = dst.clone().into_dispatch();
    drop(<Dispatch as NcmKernels>::ncm_assign(dst, src.into_dispatch(), axis, at));
}

/// [`assign_into`] at the offset held by the device tensor `at` (`[1]`, i32) rather than a launch
/// argument: a captured graph writes wherever the host last put the position.
pub fn assign_into_at<const D: usize>(
    dst: &Tensor<D>,
    src: Tensor<D>,
    axis: usize,
    at: &Tensor<1, Int>,
) {
    let (dst, at) = (dst.clone().into_dispatch(), at.clone().into_dispatch());
    drop(<Dispatch as NcmKernels>::ncm_assign_at(dst, src.into_dispatch(), axis, at));
}

/// RMSNorm of `x` `[..., d]` over its last axis with weight `gamma` `[d]` in one kernel, as
/// `burn::nn::RmsNorm` computes it at the run dtype (only the order of the f32 sum differs).
pub fn rms_norm<const D: usize>(x: Tensor<D>, gamma: Tensor<1>, eps: f64) -> Tensor<D> {
    let dims = x.dims();
    let d = dims[D - 1];
    let x = x.reshape([dims.iter().product::<usize>() / d, d]);
    let y = <Dispatch as NcmKernels>::ncm_rms_norm(x.into_dispatch(), gamma.into_dispatch(), eps);
    Tensor::<2>::from_dispatch(y).reshape(dims)
}

/// The residual add `s = x + delta` and the RMSNorm of `s` (as [`rms_norm`]) in one kernel;
/// returns `(s, norm(s))`.
pub fn add_rms_norm<const D: usize>(
    x: Tensor<D>,
    delta: Tensor<D>,
    gamma: Tensor<1>,
    eps: f64,
) -> (Tensor<D>, Tensor<D>) {
    let dims = x.dims();
    let d = dims[D - 1];
    let rows = dims.iter().product::<usize>() / d;
    let (x, delta) = (x.reshape([rows, d]), delta.reshape([rows, d]));
    let out = <Dispatch as NcmKernels>::ncm_add_rms_norm(
        x.into_dispatch(),
        delta.into_dispatch(),
        gamma.into_dispatch(),
        eps,
    );
    let out = Tensor::<3>::from_dispatch(out);
    (out.clone().narrow(0, 0, 1).reshape(dims), out.narrow(0, 1, 1).reshape(dims))
}

/// `x` itself when its layout is packed row-major, otherwise a packed copy (cubecl pitches the
/// rows of new buffers, and the in-place kernels index their caches as packed).
pub fn packed<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    Tensor::from_dispatch(<Dispatch as NcmKernels>::ncm_packed(x.into_dispatch()))
}

/// [`assign_into`] for int tensors of the same dtype (i32).
pub fn assign_int_into<const D: usize>(
    dst: &Tensor<D, Int>,
    src: Tensor<D, Int>,
    axis: usize,
    at: usize,
) {
    let dst = dst.clone().into_dispatch();
    drop(<Dispatch as NcmKernels>::ncm_assign_int(dst, src.into_dispatch(), axis, at));
}

/// `x W` for `x` `[.., K]` with at most eight rows in all and `w` `[K, N]` (row-major, or the
/// transpose of a row-major `[N, K]`), as bandwidth-bound matrix-vector kernels.
pub fn gemv<const D: usize>(x: Tensor<D>, w: Tensor<2>) -> Tensor<D> {
    let mut shape = x.dims();
    let k = shape[D - 1];
    let rows = shape.iter().product::<usize>() / k;
    debug_assert!(rows <= gemv::MAX_ROWS, "gemv takes at most {} rows", gemv::MAX_ROWS);
    shape[D - 1] = w.dims()[1];
    let counters = crate::mlp_fused::counters(&x.device()).into_dispatch();
    let y = <Dispatch as NcmKernels>::ncm_gemv(
        x.reshape([rows, k]).into_dispatch(),
        w.into_dispatch(),
        counters,
    );
    Tensor::<2>::from_dispatch(y).reshape(shape)
}

/// Whether [`gemv`] takes an input of `rows` rows.
pub fn gemv_rows_ok(rows: usize) -> bool {
    rows <= gemv::MAX_ROWS
}

/// `x` `[M, d_in]` times a quantised weight (`qweight` `[d_out, words]` i32, `scales`
/// `[d_out, groups]` f32, in `format`): one dequantise-and-multiply GEMV kernel for up to
/// [`GEMV_ROWS`](crate::quant::GEMV_ROWS) rows (decode), else the weight decoded at `x`'s dtype
/// and Burn's matmul (prefill). Codes and scales are only read, so the call can be captured into
/// a CUDA graph.
pub fn qmatmul(
    x: Tensor<2>,
    qweight: Tensor<2, Int>,
    scales: Tensor<2>,
    d_in: usize,
    format: crate::quant::Format,
) -> Tensor<2> {
    let group_words = format.group_len(d_in) / format.per_word();
    if x.dims()[0] <= crate::quant::GEMV_ROWS {
        let y = <Dispatch as NcmKernels>::ncm_qgemv(
            x.into_dispatch(),
            qweight.into_dispatch(),
            scales.into_dispatch(),
            d_in,
            format.bits,
            group_words,
        );
        return Tensor::from_dispatch(y);
    }
    let w = <Dispatch as NcmKernels>::ncm_dequant(
        qweight.into_dispatch(),
        scales.into_dispatch(),
        d_in,
        format.bits,
        group_words,
        x.dtype(),
    );
    x.matmul(Tensor::from_dispatch(w))
}

/// Version directory of the on-disk kernel cache: the crate version, the CubeCL release and a
/// hash of this crate's sources (cached kernels are keyed by name and launch settings, not by
/// their code, so another build's cache must never be read).
pub fn kernel_cache_version() -> String {
    format!("ncm-{}-cubecl-0.11.4-{}", env!("CARGO_PKG_VERSION"), env!("NCM_SOURCE_HASH"))
}

/// The default root of the on-disk kernel cache: the user's cache directory (`LOCALAPPDATA` on
/// Windows, `XDG_CACHE_HOME` or `~/.cache` elsewhere), under `ncm/kernels`.
pub fn default_kernel_cache_root() -> Option<std::path::PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("XDG_CACHE_HOME"))
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?;
    Some(base.join("ncm").join("kernels"))
}

/// Keeps compiled CUDA kernels on disk across processes, under `root` (default
/// [`default_kernel_cache_root`]) in a directory for this build ([`kernel_cache_version`]), so
/// a second process skips CubeCL's code generation for every kernel the first one compiled.
/// Call it before the first CUDA device is used: CubeCL's configuration is set once per process,
/// so after that (or after an earlier call) it does nothing. Returns the cache directory when it
/// took effect.
pub fn enable_kernel_cache(root: Option<std::path::PathBuf>) -> Option<std::path::PathBuf> {
    use burn_cubecl::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
    let dir = root.or_else(default_kernel_cache_root)?.join(kernel_cache_version());
    std::fs::create_dir_all(&dir).ok()?;
    // Whatever cubecl.toml and the CUBECL_* variables say, with the cache turned on there.
    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    config.compilation.cache = true;
    config.environment.path = CacheConfig::Directory(dir.clone());
    CubeClRuntimeConfig::try_set(config).then_some(dir)
}

/// The id of the device buffer behind a dispatch tensor (cubecl tensors; None elsewhere).
fn dispatch_buffer_id(t: burn::backend::DispatchTensor) -> Option<usize> {
    use burn::backend::{BackendTensor, DispatchTensorKind};
    let handle = match t.kind {
        DispatchTensorKind::Cube(BackendTensor::Float(c)) => c.handle,
        DispatchTensorKind::Cube(BackendTensor::Int(c)) => c.handle,
        _ => return None,
    };
    Some(handle.memory.descriptor().id.value)
}

/// The id of the device buffer behind `t` (cubecl tensors; None elsewhere): views of one
/// buffer share it, so it tells whether a view still aliases the buffer it was cut from.
pub fn buffer_id<const D: usize>(t: &Tensor<D>) -> Option<usize> {
    dispatch_buffer_id(t.clone().into_dispatch())
}

/// [`buffer_id`] of an int tensor.
pub fn int_buffer_id<const D: usize>(t: &Tensor<D, Int>) -> Option<usize> {
    dispatch_buffer_id(t.clone().into_dispatch())
}

#[cfg(test)]
mod cache_tests {
    #[test]
    fn kernel_cache_version_names_this_build() {
        let version = super::kernel_cache_version();
        assert!(version.starts_with(&format!("ncm-{}-cubecl-0.11.4-", env!("CARGO_PKG_VERSION"))));
        let hash = version.rsplit('-').next().expect("a hash");
        assert_eq!(hash.len(), 16);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn default_kernel_cache_root_is_under_ncm_kernels() {
        if let Some(root) = super::default_kernel_cache_root() {
            assert!(root.ends_with(std::path::Path::new("ncm").join("kernels")));
        }
    }
}
