//! Fused decode kernels against the reference in f32 on CUDA. Needs `--features cuda` and a GPU;
//! CI builds neither, so this file compiles to nothing there.
#![cfg(feature = "cuda")]

mod kernel_parity;

use burn::tensor::FloatDType;
use kernel_parity::*;

const DTYPE: FloatDType = FloatDType::F32;

// Largest error allowed, over the largest magnitude of the f32 CPU reference. The CUDA reference
// itself is off by up to about 2e-3 (its matmuls), so 5e-3 leaves a margin.
const TOL: f32 = 5e-3;

#[test]
fn delta_step_matches_reference_f32() {
    let (device, _lock) = setup(DTYPE);
    delta_matches(&device, TOL);
}

#[test]
fn retrieval_decode_matches_reference_f32() {
    let (device, _lock) = setup(DTYPE);
    for case in retrieval_cases() {
        retrieval_matches(&device, &case, TOL);
    }
}

#[test]
fn argmax_matches_burn_f32() {
    let (device, _lock) = setup(DTYPE);
    argmax_matches(&device);
}

#[test]
fn graphed_decode_matches_eager_f32() {
    let (device, _lock) = setup(DTYPE);
    graphed_decode_matches(&device);
}

#[test]
fn gemv_matches_cpu_f32() {
    let (device, _lock) = setup(DTYPE);
    gemv_matches(&device, TOL);
}

#[test]
fn graphed_speculation_matches_eager_f32() {
    let (device, _lock) = setup(DTYPE);
    graphed_speculation_matches(&device, true);
}
