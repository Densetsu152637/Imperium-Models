//! The hybrid generator's level tail in one launch (cargo feature `cuda`): from a level's codes
//! to its contribution added to the reconstruction.
//!
//! The reference ([`crate::hybrid::vq::BlockTokenizer::contribution_with`]) gathers the codebook
//! entries, upsamples them linearly to L positions (a matmul), runs the φ convolution (kernel
//! 3, padding 1) and blends `(1 - r) h + r conv(h)`, then the caller adds it to ẑ: about eight
//! launches per level. Here one cube per output position j builds the three upsampled rows j - 1,
//! j, j + 1 in shared memory straight from the codes and the codebook, and each unit computes
//! output channels: `acc + (1 - r) up_j + r (bias + Σ_k W[., ., k] up_{j+k-1})`, rounding at the
//! run dtype where the reference does (the upsampled rows, the convolution, the blend).

use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{Backend, Dispatch, backend_extension};
use burn::nn::conv::Conv1d;
use burn::tensor::{DType, Int, Tensor};
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl::ir::{ElemType, FloatKind};
use burn_cubecl::cubecl::prelude::{CubeCount, CubeDim};
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;

mod kernel {
    use burn_cubecl::cubecl;
    use cubecl::prelude::*;

    /// Units per cube.
    pub(super) const UNITS: usize = 256;
    /// Widest latent the kernel stages (three rows of it in shared memory).
    pub(super) const MAX_C: usize = 1024;

    /// Output row `CUBE_POS` = b * L + j of `out` `[B, L, C]`.
    #[cube(launch_unchecked)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn level_tail_kernel<F: Float>(
        codes: &Tensor<i32>,
        book: &Tensor<F>,
        up: &Tensor<F>,
        weight: &Tensor<F>,
        bias: &Tensor<F>,
        acc: &Tensor<F>,
        out: &mut Tensor<F>,
        l: usize,
        n: usize,
        c: usize,
        ratio: f32,
        #[comptime] identity: bool,
        #[comptime] add: bool,
        #[define(F)] _dtype: ElemType,
    ) {
        let row = CUBE_POS;
        let tid = UNIT_POS as usize;
        let (b, j) = (row / l, row % l);
        let mut rows = Shared::<[f32]>::new_slice(3 * MAX_C);
        // The upsampled rows j - 1, j, j + 1 (zero outside the block: the convolution's padding).
        for r in 0..3usize {
            let jj = j + r;
            let inside = jj >= 1 && jj <= l;
            let src = select(inside, jj - 1, 0);
            let mut i = tid;
            while i < c {
                let mut v = 0.0f32;
                if inside {
                    if comptime!(identity) {
                        let code =
                            usize::cast_from(codes[b * codes.stride(0) + src * codes.stride(1)]);
                        v = f32::cast_from(book[code * book.stride(0) + i * book.stride(1)]);
                    } else {
                        for m in 0..n {
                            let code =
                                usize::cast_from(codes[b * codes.stride(0) + m * codes.stride(1)]);
                            v += f32::cast_from(up[src * up.stride(0) + m * up.stride(1)])
                                * f32::cast_from(book[code * book.stride(0) + i * book.stride(1)]);
                        }
                        v = f32::cast_from(F::cast_from(v));
                    }
                }
                rows[r * MAX_C + i] = v;
                i += UNITS;
            }
        }
        sync_cube();
        let mut o = tid;
        while o < c {
            let mut s = f32::cast_from(bias[o * bias.stride(0)]);
            for k in 0..3usize {
                for i in 0..c {
                    let at = o * weight.stride(0) + i * weight.stride(1) + k * weight.stride(2);
                    s += f32::cast_from(weight[at]) * rows[k * MAX_C + i];
                }
            }
            let conv = f32::cast_from(F::cast_from(s));
            let h = rows[MAX_C + o];
            let a = f32::cast_from(F::cast_from(h * (1.0 - ratio)));
            let bb = f32::cast_from(F::cast_from(conv * ratio));
            let mut y = f32::cast_from(F::cast_from(a + bb));
            if comptime!(add) {
                y += f32::cast_from(acc[b * acc.stride(0) + j * acc.stride(1) + o * acc.stride(2)]);
            }
            out[row * c + o] = F::cast_from(y);
            o += UNITS;
        }
    }
}

fn elem(dtype: DType) -> ElemType {
    match dtype {
        DType::F32 => ElemType::Float(FloatKind::F32),
        DType::F16 => ElemType::Float(FloatKind::F16),
        DType::BF16 => ElemType::Float(FloatKind::BF16),
        other => panic!("the level tail takes f32, f16 or bf16, got {other:?}"),
    }
}

#[backend_extension(Cube)]
pub trait NcmLevelTail: Backend {
    /// See [`level_tail`].
    #[allow(clippy::too_many_arguments)]
    fn ncm_level_tail(
        codes: IntTensor<Self>,
        book: FloatTensor<Self>,
        up: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        acc: FloatTensor<Self>,
        l: usize,
        ratio: f32,
        identity: bool,
        add: bool,
    ) -> FloatTensor<Self>;
}

impl NcmLevelTail for CubeBackend {
    fn ncm_level_tail(
        codes: CubeTensor,
        book: CubeTensor,
        up: CubeTensor,
        weight: CubeTensor,
        bias: CubeTensor,
        acc: CubeTensor,
        l: usize,
        ratio: f32,
        identity: bool,
        add: bool,
    ) -> CubeTensor {
        let [b, n] = codes.meta.shape().dims::<2>();
        let [_, c] = book.meta.shape().dims::<2>();
        let out = empty_device_contiguous_dtype(
            book.client.clone(),
            book.device.clone(),
            vec![b, l, c].into(),
            book.dtype,
        );
        let client = book.client.clone();
        let dtype = elem(book.dtype);
        unsafe {
            kernel::level_tail_kernel::launch_unchecked(
                &client,
                CubeCount::Static((b * l) as u32, 1, 1),
                CubeDim::new_1d(kernel::UNITS as u32),
                codes.into_tensor_arg(),
                book.into_tensor_arg(),
                up.into_tensor_arg(),
                weight.into_tensor_arg(),
                bias.into_tensor_arg(),
                acc.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                l,
                n,
                c,
                ratio,
                identity,
                add,
                dtype,
            );
        }
        out
    }
}

/// Level contribution `[B, L, C]` of `codes` `[B, ℓ]` (i32) over `book` `[n, C]`, upsampled by
/// `up` `[L, ℓ]` (None: the identity, ℓ = L) and blended through `conv` (kernel 3, padding 1,
/// with a bias) at `ratio`, plus `acc` `[B, L, C]` when given; in one launch. None when the
/// shapes do not fit (contiguous inputs, C at most 1024, a 3-wide convolution with a bias).
pub fn level_tail(
    codes: Tensor<2, Int>,
    book: Tensor<2>,
    up: Option<Tensor<2>>,
    conv: &Conv1d,
    ratio: f64,
    acc: Option<Tensor<3>>,
) -> Option<Tensor<3>> {
    let [b, n] = codes.dims();
    let [_, c] = book.dims();
    let bias = conv.bias.as_ref()?.val();
    let weight = conv.weight.val();
    if c > kernel::MAX_C || weight.dims() != [c, c, 3] || conv.kernel_size != 3 {
        return None;
    }
    let l = up.as_ref().map_or(n, |u| u.dims()[0]);
    let identity = up.is_none();
    let up = up.unwrap_or_else(|| book.clone().narrow(0, 0, 1));
    let add = acc.is_some();
    let acc = acc.unwrap_or_else(|| Tensor::zeros([1, 1, 1], &book.device()));
    let codes = codes.cast(burn::tensor::IntDType::I32);
    let out = <Dispatch as NcmLevelTail>::ncm_level_tail(
        codes.into_dispatch(),
        book.into_dispatch(),
        up.into_dispatch(),
        weight.into_dispatch(),
        bias.into_dispatch(),
        acc.into_dispatch(),
        l,
        ratio as f32,
        identity,
        add,
    );
    let out = Tensor::from_dispatch(out);
    debug_assert_eq!(out.dims(), [b, l, c]);
    Some(out)
}
