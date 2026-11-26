//! RoPE, optionally after a per-head RMSNorm (QK-norm), in one launch (cargo feature `cuda`).
//!
//! The reference [`crate::layers::apply_rope`] is `(x @ R) * sin + x * cos`: a matmul by the
//! half-rotation, two products and a sum, four launches per call, and QK-norm adds a fifth.
//! Here one cube per head row loads the row (normalising it when asked), rotates it in shared
//! memory and writes the result, rounding as the reference does at the run dtype `F`:
//! `F(F(rot(x) * sin) + F(x * cos))`, with the norm as `kernels::norm`'s.

use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, Dispatch, backend_extension};
use burn::tensor::{DType, Tensor};
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl::ir::{ElemType, FloatKind};
use burn_cubecl::cubecl::prelude::{CubeCount, CubeDim};
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;

use crate::layers::Rope;

mod kernel {
    use burn_cubecl::cubecl;
    use cubecl::prelude::*;

    /// Units per cube (one cube per head row).
    pub(super) const UNITS: usize = 64;
    /// Widest head the kernel stages.
    pub(super) const MAX_D: usize = 256;

    /// Row `CUBE_POS` of `x` `[B, H, T, D]` (any strides) into the contiguous `out`: RMS-normed
    /// by `gamma` `[D]` with `norm`, then rotated by the tables `cos`, `sin` `[1, 1, T, D]` (or
    /// `[B, 1, T, D]`: per batch row).
    #[cube(launch_unchecked)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rope_kernel<F: Float>(
        x: &Tensor<F>,
        gamma: &Tensor<F>,
        cos: &Tensor<F>,
        sin: &Tensor<F>,
        out: &mut Tensor<F>,
        heads: usize,
        t: usize,
        d: usize,
        eps: f32,
        #[comptime] norm: bool,
        #[define(F)] _dtype: ElemType,
    ) {
        let row = CUBE_POS;
        let tid = UNIT_POS as usize;
        let (bi, hi, ti) = (row / (heads * t), (row / t) % heads, row % t);
        let base = bi * x.stride(0) + hi * x.stride(1) + ti * x.stride(2);
        let mut s = Shared::<[f32]>::new_slice(MAX_D);
        let mut part = Shared::<[f32]>::new_slice(UNITS / 32);
        let mut sq = 0.0f32;
        let mut i = tid;
        while i < d {
            let v = f32::cast_from(x[base + i * x.stride(3)]);
            s[i] = v;
            sq += v * v;
            i += UNITS;
        }
        if comptime!(norm) {
            let total = plane_sum(sq);
            if UNIT_POS_PLANE == 0 {
                part[(UNIT_POS / PLANE_DIM) as usize] = total;
            }
            sync_cube();
            let mut all = 0.0f32;
            for p in 0..UNITS / 32 {
                all += part[p];
            }
            let rms = f32::cast_from(F::cast_from(f32::sqrt(all / f32::cast_from(d) + eps)));
            let mut i = tid;
            while i < d {
                let n = f32::cast_from(F::cast_from(s[i] / rms));
                s[i] = f32::cast_from(F::cast_from(n * f32::cast_from(gamma[i * gamma.stride(0)])));
                i += UNITS;
            }
        }
        sync_cube();
        let half = d / 2;
        // Tables `[1 or B, 1, T, D]`: one set for every batch row, or one per row.
        let cb = select(cos.shape(0) > 1, bi, 0);
        let sb = select(sin.shape(0) > 1, bi, 0);
        let (c0, c1) = (cb * cos.stride(0) + ti * cos.stride(2), cos.stride(3));
        let (s0, s1) = (sb * sin.stride(0) + ti * sin.stride(2), sin.stride(3));
        let mut i = tid;
        while i < d {
            // The partner in the other half (indices stay inside the row either way).
            let partner = select(i < half, i + half, i - select(i < half, 0, half));
            let rot = select(i < half, -s[partner], s[partner]);
            let a = f32::cast_from(F::cast_from(rot * f32::cast_from(sin[s0 + i * s1])));
            let b = f32::cast_from(F::cast_from(s[i] * f32::cast_from(cos[c0 + i * c1])));
            out[row * d + i] = F::cast_from(a + b);
            i += UNITS;
        }
    }
}

fn elem(dtype: DType) -> ElemType {
    match dtype {
        DType::F32 => ElemType::Float(FloatKind::F32),
        DType::F16 => ElemType::Float(FloatKind::F16),
        DType::BF16 => ElemType::Float(FloatKind::BF16),
        other => panic!("the fused RoPE takes f32, f16 or bf16, got {other:?}"),
    }
}

#[backend_extension(Cube)]
pub trait NcmRope: Backend {
    /// See [`rope`].
    #[allow(clippy::too_many_arguments)]
    fn ncm_rope(
        x: FloatTensor<Self>,
        gamma: FloatTensor<Self>,
        cos: FloatTensor<Self>,
        sin: FloatTensor<Self>,
        eps: f64,
        norm: bool,
    ) -> FloatTensor<Self>;
}

impl NcmRope for CubeBackend {
    fn ncm_rope(
        x: CubeTensor,
        gamma: CubeTensor,
        cos: CubeTensor,
        sin: CubeTensor,
        eps: f64,
        norm: bool,
    ) -> CubeTensor {
        let [b, h, t, d] = x.meta.shape().dims::<4>();
        let out = empty_device_contiguous_dtype(
            x.client.clone(),
            x.device.clone(),
            vec![b, h, t, d].into(),
            x.dtype,
        );
        let client = x.client.clone();
        let dtype = elem(x.dtype);
        unsafe {
            kernel::rope_kernel::launch_unchecked(
                &client,
                CubeCount::Static((b * h * t) as u32, 1, 1),
                CubeDim::new_1d(kernel::UNITS as u32),
                x.into_tensor_arg(),
                gamma.into_tensor_arg(),
                cos.into_tensor_arg(),
                sin.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                h,
                t,
                d,
                eps as f32,
                norm,
                dtype,
            );
        }
        out
    }
}

/// `x` `[B, H, T, D]` rotated by `rope` (after an RMSNorm of each head row by `norm`'s
/// `(gamma, eps)` when given) in one launch, as [`crate::layers::apply_rope`] of the normed row
/// computes it. None for heads wider than the kernel stages (the caller runs the reference).
pub fn rope(x: Tensor<4>, rope: &Rope, norm: Option<(Tensor<1>, f64)>) -> Option<Tensor<4>> {
    let d = x.dims()[3];
    if d > kernel::MAX_D || !d.is_multiple_of(2) {
        return None;
    }
    let (cos, sin) = rope.tables();
    let (gamma, eps, on) = match norm {
        Some((gamma, eps)) => (gamma, eps, true),
        None => (cos.clone().reshape([cos.dims().iter().product::<usize>()]), 0.0, false),
    };
    let out = <Dispatch as NcmRope>::ncm_rope(
        x.into_dispatch(),
        gamma.into_dispatch(),
        cos.into_dispatch(),
        sin.into_dispatch(),
        eps,
        on,
    );
    Some(Tensor::from_dispatch(out))
}

/// `x` `[B, H, T, D]` rotated by per-row (and optionally per-batch-row) tables `cos`, `sin`
/// `[1 or B, 1, T, D]` in one launch (the hybrid's batched slots, whose positions differ per
/// row). None for heads wider than the kernel stages.
pub fn rope_tables(x: Tensor<4>, cos: Tensor<4>, sin: Tensor<4>) -> Option<Tensor<4>> {
    let [b, _, t, d] = x.dims();
    let [tb, _, tt, td] = cos.dims();
    if d > kernel::MAX_D || !d.is_multiple_of(2) || td != d || tt != t || (tb != 1 && tb != b) {
        return None;
    }
    let gamma = cos.clone().reshape([cos.dims().iter().product::<usize>()]);
    let out = <Dispatch as NcmRope>::ncm_rope(
        x.into_dispatch(),
        gamma.into_dispatch(),
        cos.into_dispatch(),
        sin.into_dispatch(),
        0.0,
        false,
    );
    Some(Tensor::from_dispatch(out))
}
