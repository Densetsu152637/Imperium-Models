//! Fused CubeCL kernels for the host-bound v2 decode paths (cargo feature `cuda`).
//!
//! Each kernel matches a reference path in `delta.rs` or `retrieval.rs` and is exposed to the
//! layers through a Burn backend extension routed to the cubecl backend. The layers call them only
//! where [`crate::fused::active`] says so; everything else runs the reference tensor code.

mod argmax;
mod assign;
mod delta;
mod retrieval;

use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{Backend, Dispatch, backend_extension};
use burn::tensor::{DType, Int, Tensor};
use burn_cubecl::cubecl::prelude::*;
use burn_cubecl::cubecl::{CubeCount, CubeDim};
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{CubeBackend, cubecl};

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
    /// blocks and the most chunks of attended slots (dense steps see `pos + 1` slots, which is
    /// at most `local + topk * block` until a step turns sparse).
    fn grids(&self, capacity: usize) -> (usize, usize) {
        let last = capacity.saturating_sub(1);
        let blocks = if last >= self.local { (last - self.local) / self.block + 1 } else { 0 };
        let slots =
            if self.dense { capacity } else { capacity.min(self.topk * self.block + self.local) };
        (blocks.max(1), slots.div_ceil(retrieval::CHUNK).max(1))
    }
}

#[backend_extension(Cube)]
pub trait NcmKernels: Backend {
    /// See [`argmax_last`].
    fn ncm_argmax(x: FloatTensor<Self>) -> IntTensor<Self>;

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
        position: IntTensor<Self>,
        dims: RetrievalDims,
    ) -> FloatTensor<Self>;
}

impl NcmKernels for CubeBackend {
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
                CubeDim::new_1d(hd as u32),
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
        position: CubeTensor,
        dims: RetrievalDims,
    ) -> CubeTensor {
        use retrieval::{CUBE, INDEX_CUBE};
        let b = proj.meta.shape().dims::<3>()[0];
        let capacity = kv.meta.shape().dims::<3>()[1];
        let d = dims;
        let (h, dr, l) = (d.heads, d.rope_dim, d.latent);
        let (gate, index_q, index_k) = d.offsets();
        let (blocks, chunks) = d.grids(capacity);
        let dtype = elem(proj.dtype);
        let client = proj.client.clone();
        let q_lat = scratch(&proj, vec![b, h, l], DType::F32);
        let q_rope = scratch(&proj, vec![b, h, dr], DType::F32);
        let scale = 1.0 / ((d.nope_dim + dr) as f32).sqrt();
        unsafe {
            retrieval::ret_prep_kernel::launch_unchecked(
                &client,
                CubeCount::Static((b * h + b) as u32, 1, 1),
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
                    scores.clone().into_tensor_arg(),
                    position.clone().into_tensor_arg(),
                    blocks,
                    1.0 / ((d.index_heads * d.index_dim) as f32).sqrt(),
                    d.index_heads,
                    d.index_dim,
                    index_q,
                    d.block,
                    d.local,
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
                CubeCount::Static((b * chunks) as u32, 1, 1),
                CubeDim::new_1d(CUBE as u32),
                q_lat.into_tensor_arg(),
                q_rope.into_tensor_arg(),
                kv.into_tensor_arg(),
                picked.into_tensor_arg(),
                part_max.clone().into_tensor_arg(),
                part_sum.clone().into_tensor_arg(),
                part_acc.clone().into_tensor_arg(),
                position.into_tensor_arg(),
                chunks,
                d.topk,
                h,
                dr,
                l,
                d.block,
                d.local,
                d.dense,
                dtype,
            );
            retrieval::ret_combine_kernel::launch_unchecked(
                &client,
                CubeCount::Static((b * h) as u32, 1, 1),
                CubeDim::new_1d(CUBE as u32),
                proj.into_tensor_arg(),
                wuv.into_tensor_arg(),
                part_max.into_tensor_arg(),
                part_sum.into_tensor_arg(),
                part_acc.into_tensor_arg(),
                out.clone().into_tensor_arg(),
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

/// One-token sparse retrieval decode at the position held by the device tensor `position`
/// (`[1]`, i32) from the layer's input projection `proj` `[B, 1, W]`: writes the position's entry
/// into the cache in place (`kv` `[B, S, l + dr]`, `keys` `[B, S, di]`, whoever else holds them),
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
    keys: &Tensor<3>,
    position: &Tensor<1, Int>,
    dims: RetrievalDims,
) -> Tensor<3> {
    let [q, latent, rope_key, index] = norms;
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
        keys.clone().into_dispatch(),
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
