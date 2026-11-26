//! In-place writes into cache buffers of any element type (cargo feature `cuda`), so that a
//! cache keeps its buffers across requests and the decode graphs captured against them stay
//! valid: [`crate::kernels::assign_into`] covers float and i32 tensors; this also covers the
//! 8-bit indexer-key codes (i8).

use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{Backend, Dispatch, backend_extension};
use burn::tensor::{DType, Int, Tensor};
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl::ir::{ElemType, FloatKind, IntKind};
use burn_cubecl::cubecl::prelude::{CubeCount, CubeDim};
use burn_cubecl::tensor::CubeTensor;

mod kernel {
    use burn_cubecl::cubecl;
    use cubecl::prelude::*;

    pub(super) const CUBE: usize = 256;

    /// `dst[.., at + i, ..] = src[.., i, ..]` along `axis` (any rank, both strided).
    #[cube(launch_unchecked)]
    pub(super) fn copy_kernel<N: Numeric>(
        dst: &mut Tensor<N>,
        src: &Tensor<N>,
        count: usize,
        axis: usize,
        at: usize,
        #[define(N)] _dtype: ElemType,
    ) {
        let pos = ABSOLUTE_POS;
        if pos >= count {
            terminate!();
        }
        let mut rem = pos;
        let mut from = 0usize;
        let mut to = 0usize;
        let mut d = src.rank();
        while d > 0 {
            d -= 1;
            let n = src.shape(d);
            let c = rem % n;
            rem /= n;
            from += c * src.stride(d);
            to += (c + select(d == axis, at, 0usize)) * dst.stride(d);
        }
        dst[to] = src[from];
    }
}

fn elem(dtype: DType) -> ElemType {
    match dtype {
        DType::F32 => ElemType::Float(FloatKind::F32),
        DType::F16 => ElemType::Float(FloatKind::F16),
        DType::BF16 => ElemType::Float(FloatKind::BF16),
        DType::I32 => ElemType::Int(IntKind::I32),
        DType::I8 => ElemType::Int(IntKind::I8),
        other => panic!("in-place cache writes take f32, f16, bf16, i32 or i8, got {other:?}"),
    }
}

fn copy(dst: CubeTensor, src: CubeTensor, axis: usize, at: usize) -> CubeTensor {
    assert_eq!(dst.dtype, src.dtype, "in-place writes keep the destination's dtype");
    let count = src.meta.shape().num_elements();
    if count == 0 {
        return dst;
    }
    let client = dst.client.clone();
    let dtype = elem(dst.dtype);
    unsafe {
        kernel::copy_kernel::launch_unchecked(
            &client,
            CubeCount::Static(count.div_ceil(kernel::CUBE) as u32, 1, 1),
            CubeDim::new_1d(kernel::CUBE as u32),
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

#[backend_extension(Cube)]
pub trait NcmInPlace: Backend {
    /// See [`write_float`].
    fn ncm_write_float(
        dst: FloatTensor<Self>,
        src: FloatTensor<Self>,
        axis: usize,
        at: usize,
    ) -> FloatTensor<Self>;

    /// See [`write_int`].
    fn ncm_write_int(
        dst: IntTensor<Self>,
        src: IntTensor<Self>,
        axis: usize,
        at: usize,
    ) -> IntTensor<Self>;
}

impl NcmInPlace for CubeBackend {
    fn ncm_write_float(dst: CubeTensor, src: CubeTensor, axis: usize, at: usize) -> CubeTensor {
        copy(dst, src, axis, at)
    }

    fn ncm_write_int(dst: CubeTensor, src: CubeTensor, axis: usize, at: usize) -> CubeTensor {
        copy(dst, src, axis, at)
    }
}

/// Writes `src` into `dst`'s own buffer at offset `at` along `axis`, cast to `dst`'s dtype.
pub fn write_float<const D: usize>(dst: &Tensor<D>, src: Tensor<D>, axis: usize, at: usize) {
    let src = src.cast(dst.dtype());
    let dst = dst.clone().into_dispatch();
    drop(<Dispatch as NcmInPlace>::ncm_write_float(dst, src.into_dispatch(), axis, at));
}

/// [`write_float`] for int tensors (i32 or i8), cast to `dst`'s dtype.
pub fn write_int<const D: usize>(
    dst: &Tensor<D, Int>,
    src: Tensor<D, Int>,
    axis: usize,
    at: usize,
) {
    let src = match dst.dtype() {
        DType::I8 => src.cast(burn::tensor::IntDType::I8),
        _ => src.cast(burn::tensor::IntDType::I32),
    };
    let dst = dst.clone().into_dispatch();
    drop(<Dispatch as NcmInPlace>::ncm_write_int(dst, src.into_dispatch(), axis, at));
}
