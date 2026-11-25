//! Fused dequantise-and-multiply kernels for quantised linear layers on CUDA (cargo feature
//! `cuda`), exposed through a Burn backend extension routed to the cubecl backend.
//!
//! * [`qgemv_kernel`]: `y = x W` for up to [`GEMV_ROWS`] rows of `x`. One plane (warp) per
//!   output feature streams that feature's packed row once, each lane taking every 32nd word,
//!   decoding its codes in registers and accumulating every row of `x` against them; a plane sum
//!   finishes the dot products. The weight is read once per call whatever the row count, so decode
//!   moves `bits / 16` of the bf16 weight traffic.
//! * [`dequant_kernel`]: decodes the weight to `[d_in, d_out]` at the run dtype for Burn's
//!   tuned matmul when there are more rows (prefill, where compute rather than weight traffic
//!   dominates).
//!
//! Arithmetic is f32; `x` and outputs are at the run dtype, codes i32, scales f32.

use burn::backend::{Backend, Dispatch, backend_extension, tensor::FloatTensor, tensor::IntTensor};
use burn::tensor::{DType, Device, Int, Tensor as BurnTensor};
use burn_cubecl::cubecl::prelude::*;
use burn_cubecl::cubecl::{CubeCount, CubeDim};
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{CubeBackend, cubecl};

use super::format::Format;
use super::{GEMV_ROWS, kernels_allowed};

/// Output features per GEMV cube (one plane each).
const PLANES: usize = 4;
/// Lanes per plane: the CUDA warp.
const PLANE: usize = 32;
/// Units per dequantisation cube.
const DEQUANT_CUBE: usize = 256;

/// Whether quantised layers on `device` take these kernels now.
pub(super) fn active(device: &Device) -> bool {
    use burn::backend::DispatchDevice;
    kernels_allowed()
        && matches!(device.as_dispatch(), DispatchDevice::Cube(burn_cubecl::CubeDevice::Cuda(_)))
}

fn elem(dtype: DType) -> ElemType {
    match dtype {
        DType::F32 => ElemType::Float(cubecl::ir::FloatKind::F32),
        DType::F16 => ElemType::Float(cubecl::ir::FloatKind::F16),
        DType::BF16 => ElemType::Float(cubecl::ir::FloatKind::BF16),
        other => panic!("quantised matmul takes float activations, got {other:?}"),
    }
}

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

#[backend_extension(Cube)]
pub trait QuantKernels: Backend {
    /// See [`qgemv_kernel`]; `x` `[rows, d_in]` with `rows <= GEMV_ROWS`.
    fn ncm_qgemv(
        x: FloatTensor<Self>,
        packed: IntTensor<Self>,
        scales: FloatTensor<Self>,
        d_in: usize,
        bits: u32,
        group_words: usize,
    ) -> FloatTensor<Self>;

    /// See [`dequant_kernel`]; the result is `[d_in, d_out]` at `dtype`.
    fn ncm_dequant(
        packed: IntTensor<Self>,
        scales: FloatTensor<Self>,
        d_in: usize,
        bits: u32,
        group_words: usize,
        dtype: DType,
    ) -> FloatTensor<Self>;
}

impl QuantKernels for CubeBackend {
    fn ncm_qgemv(
        x: CubeTensor,
        packed: CubeTensor,
        scales: CubeTensor,
        d_in: usize,
        bits: u32,
        group_words: usize,
    ) -> CubeTensor {
        let rows = x.meta.shape().dims::<2>()[0];
        let [n_len, words] = packed.meta.shape().dims::<2>();
        let out = empty_device_contiguous_dtype(
            x.client.clone(),
            x.device.clone(),
            vec![rows, n_len].into(),
            x.dtype,
        );
        let dtype = elem(x.dtype);
        let client = x.client.clone();
        unsafe {
            qgemv_kernel::launch_unchecked(
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
        let [n_len, words] = packed.meta.shape().dims::<2>();
        let out = empty_device_contiguous_dtype(
            packed.client.clone(),
            packed.device.clone(),
            vec![d_in, n_len].into(),
            dtype,
        );
        let client = packed.client.clone();
        unsafe {
            dequant_kernel::launch_unchecked(
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
}

/// `x` `[M, d_in]` times the quantised weight (`qweight` `[d_out, words]`, `scales`
/// `[d_out, groups]`): the GEMV kernel for up to [`GEMV_ROWS`] rows, else a dequantised weight
/// and Burn's matmul.
pub(super) fn qmatmul(
    x: BurnTensor<2>,
    qweight: BurnTensor<2, Int>,
    scales: BurnTensor<2>,
    d_in: usize,
    format: Format,
) -> BurnTensor<2> {
    let group_words = format.group_len(d_in) / format.per_word();
    let rows = x.dims()[0];
    if rows <= GEMV_ROWS {
        let y = <Dispatch as QuantKernels>::ncm_qgemv(
            x.into_dispatch(),
            qweight.into_dispatch(),
            scales.into_dispatch(),
            d_in,
            format.bits,
            group_words,
        );
        return BurnTensor::from_dispatch(y);
    }
    let w = <Dispatch as QuantKernels>::ncm_dequant(
        qweight.into_dispatch(),
        scales.into_dispatch(),
        d_in,
        format.bits,
        group_words,
        x.dtype(),
    );
    x.matmul(BurnTensor::from_dispatch(w))
}
