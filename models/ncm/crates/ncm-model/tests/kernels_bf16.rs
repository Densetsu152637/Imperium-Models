//! Fused decode kernels against the reference in bf16 on CUDA (the benchmark's dtype). Needs
//! `--features cuda` and a GPU; CI builds neither, so this file compiles to nothing there.
#![cfg(feature = "cuda")]

mod kernel_parity;

use burn::tensor::FloatDType;
use kernel_parity::*;

const DTYPE: FloatDType = FloatDType::BF16;

// Largest error allowed, over the largest magnitude of the f32 CPU reference run on the same
// (bf16-exact) weights and inputs. The CUDA bf16 reference rounds every intermediate tensor and
// is off by up to about 2.5e-2; the kernels keep f32 inside and store bf16 only in the caches.
const TOL: f32 = 4e-2;

#[test]
fn delta_step_matches_reference_bf16() {
    let (device, _lock) = setup(DTYPE);
    delta_matches(&device, TOL);
}

#[test]
fn retrieval_decode_matches_reference_bf16() {
    let (device, _lock) = setup(DTYPE);
    for case in retrieval_cases().into_iter().filter(|c| c.bf16_stable) {
        retrieval_matches(&device, &case, TOL);
    }
}
