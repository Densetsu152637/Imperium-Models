//! Temperature, top-k and top-p sampling on the device (cargo feature `cuda`), so that a
//! sampled decode step can stay inside a captured CUDA graph like a greedy one.
//!
//! One cube per row of logits. The kernel keeps the same candidates as the host sampler
//! (`ncm-server`'s `sampling.rs`): the `top_k` largest logits (all for 0), then the smallest
//! prefix of them, best first, whose probability at the temperature reaches `top_p`. Both cuts
//! are found by bisection over the logits' order-preserving 32-bit keys (exact, ties kept
//! together). It then draws from the kept tokens by the Gumbel-max trick: the argmax of
//! `logit / temperature + g`, `g` standard Gumbel noise from a counter-based hash of the seed,
//! the draw's counter and the token id. That is an exact draw from the kept tokens'
//! distribution; the random stream differs from the host sampler's, so a seed gives the same
//! output on the device every time but not the host sampler's output.
//!
//! The settings and the counter live in device buffers ([`SampleParams`]): a graph replays the
//! same launches whatever a request asks for, and each draw advances the counter.

use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{Backend, Dispatch, backend_extension};
use burn::tensor::{DType, Int, IntDType, Tensor, TensorData};
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::ir::ElemType;
use burn_cubecl::cubecl::prelude::{CubeCount, CubeDim};
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;

// Kernel code: `v == v` is the NaN test, and the bisection midpoints stay in u32 range
// because finite keys never reach 0 or u32::MAX.
#[allow(clippy::eq_op, clippy::manual_div_ceil)]
mod kernel {
    use burn_cubecl::cubecl;
    use cubecl::prelude::*;

    /// Units per cube (one cube per row).
    pub(super) const CUBE: usize = 256;
    /// Planes per cube at the CUDA warp width.
    const PLANES: usize = CUBE / 32;

    /// The order-preserving key of `x`: `a < b` exactly when `key(a) < key(b)` (NaN treated as -inf
    /// by the caller).
    #[cube]
    fn key(x: f32) -> u32 {
        let bits = u32::reinterpret(x);
        select(bits >= 0x8000_0000u32, !bits, bits | 0x8000_0000u32)
    }

    /// The value of key `k` (the inverse of [`key`]).
    #[cube]
    fn value(k: u32) -> f32 {
        let bits = select(k >= 0x8000_0000u32, k & 0x7FFF_FFFFu32, !k);
        f32::reinterpret(bits)
    }

    /// Sum of `v` over the cube.
    #[cube]
    fn cube_sum(v: f32, part: &mut Shared<[f32]>) -> f32 {
        let s = plane_sum(v);
        if UNIT_POS_PLANE == 0 {
            part[(UNIT_POS / PLANE_DIM) as usize] = s;
        }
        sync_cube();
        let mut total = 0.0f32;
        for p in 0..PLANES {
            total += part[p];
        }
        sync_cube();
        total
    }

    /// Max of `v` over the cube.
    #[cube]
    fn cube_max(v: f32, part: &mut Shared<[f32]>) -> f32 {
        let s = plane_max(v);
        if UNIT_POS_PLANE == 0 {
            part[(UNIT_POS / PLANE_DIM) as usize] = s;
        }
        sync_cube();
        let mut best = f32::new(-3.0e38f32);
        for p in 0..PLANES {
            best = f32::max(best, part[p]);
        }
        sync_cube();
        best
    }

    /// Logit `i` of row `row`, NaN as -inf.
    #[cube]
    fn logit<F: Float>(x: &Tensor<F>, row: usize, i: usize) -> f32 {
        let v = f32::cast_from(x[row * x.stride(0) + i * x.stride(1)]);
        select(v == v, v, f32::new(-3.0e38f32))
    }

    /// A 32-bit mix of three words (a PCG-style hash).
    #[cube]
    fn mix(a: u32, b: u32, c: u32) -> u32 {
        let mut h = a ^ 0x9E37_79B9u32;
        h = (h ^ b) * 0x85EB_CA6Bu32;
        h ^= h >> 13;
        h = (h ^ c) * 0xC2B2_AE35u32;
        h ^= h >> 16;
        h *= 0x27D4_EB2Fu32;
        h ^ (h >> 15)
    }

    /// Row `CUBE_POS` of `x` `[rows, n]`: `params` `[temperature, top_p, top_k, 0]` (f32), `state`
    /// `[seed_lo, seed_hi, counter]` (i32 bits); draw `counter + offset + row`. Writes the token to
    /// `out[row]`.
    #[cube(launch_unchecked)]
    pub(crate) fn sample_kernel<F: Float>(
        x: &Tensor<F>,
        params: &Tensor<f32>,
        state: &Tensor<i32>,
        out: &mut Tensor<i32>,
        n: usize,
        offset: u32,
        #[define(F)] _dtype: ElemType,
    ) {
        let row = CUBE_POS;
        let tid = UNIT_POS as usize;
        let mut part = Shared::<[f32]>::new_slice(PLANES);
        let temperature = params[0];
        let top_p = params[1];
        let top_k = u32::cast_from(params[2]);
        // The largest logit and the lowest.
        let mut hi = f32::new(-3.0e38f32);
        let mut lo = f32::new(3.0e38f32);
        let mut i = tid;
        while i < n {
            let v = logit(x, row, i);
            hi = f32::max(hi, v);
            lo = f32::min(lo, v);
            i += CUBE;
        }
        let hi = cube_max(hi, &mut part);
        let lo = -cube_max(-lo, &mut part);
        // Top-k: the largest key `t` with at least `top_k` logits at or above it.
        let mut cut = key(lo);
        if top_k > 0 && (top_k as usize) < n {
            let mut a = key(lo);
            let mut b = key(hi);
            while a < b {
                let mid = a + (b - a + 1) / 2;
                let t = value(mid);
                let mut count = 0.0f32;
                let mut i = tid;
                while i < n {
                    count += select(logit(x, row, i) >= t, 1.0f32, 0.0f32);
                    i += CUBE;
                }
                if cube_sum(count, &mut part) >= f32::cast_from(top_k) {
                    a = mid;
                } else {
                    b = mid - 1;
                }
            }
            cut = a;
        }
        // Top-p: the largest key `t` whose kept mass at or above it reaches `top_p` of the total.
        if top_p < 1.0 {
            let floor = value(cut);
            let mut mass = 0.0f32;
            let mut i = tid;
            while i < n {
                let v = logit(x, row, i);
                mass += select(v >= floor, f32::exp((v - hi) / temperature), 0.0f32);
                i += CUBE;
            }
            let goal = cube_sum(mass, &mut part) * top_p;
            let mut a = cut;
            let mut b = key(hi);
            while a < b {
                let mid = a + (b - a + 1) / 2;
                let t = value(mid);
                let mut kept = 0.0f32;
                let mut i = tid;
                while i < n {
                    let v = logit(x, row, i);
                    kept += select(v >= t, f32::exp((v - hi) / temperature), 0.0f32);
                    i += CUBE;
                }
                if cube_sum(kept, &mut part) >= goal {
                    a = mid;
                } else {
                    b = mid - 1;
                }
            }
            cut = a;
        }
        // Gumbel-max over the kept logits.
        let floor = value(cut);
        let draw = u32::reinterpret(state[2]) + offset + row as u32;
        let (s0, s1) = (u32::reinterpret(state[0]), u32::reinterpret(state[1]));
        let mut best = f32::new(-3.0e38f32);
        let mut at = u32::new(0);
        let mut i = tid;
        while i < n {
            let v = logit(x, row, i);
            if v >= floor {
                let h = mix(s0 ^ draw, s1, i as u32);
                let u = (f32::cast_from(h >> 8) + 0.5f32) / 16_777_216.0f32;
                let score = v / temperature - f32::ln(-f32::ln(u));
                if score > best {
                    best = score;
                    at = i as u32;
                }
            }
            i += CUBE;
        }
        // The cube's best score; ties to the lowest id.
        let top = cube_max(best, &mut part);
        let mine = select(best == top, at, u32::new(0x7FFF_FFFF));
        let mut ids = Shared::<[u32]>::new_slice(PLANES);
        let m = plane_min(mine);
        if UNIT_POS_PLANE == 0 {
            ids[(UNIT_POS / PLANE_DIM) as usize] = m;
        }
        sync_cube();
        if tid == 0 {
            let mut pick = u32::new(0x7FFF_FFFF);
            for p in 0..PLANES {
                pick = u32::min(pick, ids[p]);
            }
            out[row] = i32::cast_from(pick);
        }
    }
}

fn elem(dtype: DType) -> ElemType {
    match dtype {
        DType::F32 => ElemType::Float(cubecl::ir::FloatKind::F32),
        DType::F16 => ElemType::Float(cubecl::ir::FloatKind::F16),
        DType::BF16 => ElemType::Float(cubecl::ir::FloatKind::BF16),
        other => panic!("sampling takes f32, f16 or bf16 logits, got {other:?}"),
    }
}

#[backend_extension(Cube)]
pub trait NcmSampling: Backend {
    /// See [`sample`].
    fn ncm_sample(
        x: FloatTensor<Self>,
        params: FloatTensor<Self>,
        state: IntTensor<Self>,
        offset: u32,
    ) -> IntTensor<Self>;
}

impl NcmSampling for CubeBackend {
    fn ncm_sample(x: CubeTensor, params: CubeTensor, state: CubeTensor, offset: u32) -> CubeTensor {
        let [rows, n] = x.meta.shape().dims::<2>();
        let out = empty_device_contiguous_dtype(
            x.client.clone(),
            x.device.clone(),
            vec![rows, 1].into(),
            DType::I32,
        );
        let dtype = elem(x.dtype);
        let client = x.client.clone();
        unsafe {
            kernel::sample_kernel::launch_unchecked(
                &client,
                CubeCount::Static(rows as u32, 1, 1),
                CubeDim::new_1d(kernel::CUBE as u32),
                x.into_tensor_arg(),
                params.into_tensor_arg(),
                state.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                n,
                offset,
                dtype,
            );
        }
        out
    }
}

/// A sampler's settings and draw counter in device buffers, read by [`sample`]: refreshed with
/// [`Self::set`] before a run (or a graph replay) and advanced with [`Self::advance`] after.
#[derive(Clone)]
pub struct SampleParams {
    /// `[temperature, top_p, top_k, 0]`, f32.
    params: Tensor<1>,
    /// `[seed_lo, seed_hi, counter]`, i32 bits.
    state: Tensor<1, Int>,
}

/// What a sampled request asks for. `temperature` must be above 0 (greedy decoding takes the
/// argmax instead); `top_k` 0 and `top_p` 1 keep every token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleSettings {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
    /// Draws made so far in the request.
    pub counter: u32,
}

impl SampleParams {
    pub fn new(device: &burn::tensor::Device) -> Self {
        let params = Tensor::<1>::zeros([4], (device, DType::F32));
        let state = Tensor::<1, Int>::zeros([3], device).cast(IntDType::I32);
        Self { params, state }
    }

    /// Copies `settings` into the device buffers (in place: a captured graph keeps reading
    /// them).
    pub fn set(&self, settings: SampleSettings) {
        let device = self.params.device();
        let p = [settings.temperature, settings.top_p, settings.top_k as f32, 0.0];
        let p = Tensor::<1>::from_data(TensorData::new(p.to_vec(), [4]), (&device, DType::F32));
        crate::kernels::assign_into(&self.params, p, 0, 0);
        let s = [
            settings.seed as u32 as i32,
            (settings.seed >> 32) as u32 as i32,
            settings.counter as i32,
        ];
        let s = Tensor::<1, Int>::from_data(TensorData::new(s.to_vec(), [3]), &device)
            .cast(IntDType::I32);
        crate::kernels::assign_int_into(&self.state, s, 0, 0);
    }
}

/// Samples one token per row of `logits` `[rows, n]` (any float dtype) per `params`, draw
/// `counter + offset + row`; returns `[rows, 1]` i32. One kernel, no host sync.
pub fn sample(logits: Tensor<2>, params: &SampleParams, offset: u32) -> Tensor<2, Int> {
    let out = <Dispatch as NcmSampling>::ncm_sample(
        logits.into_dispatch(),
        params.params.clone().into_dispatch(),
        params.state.clone().into_dispatch(),
        offset,
    );
    Tensor::from_dispatch(out)
}
