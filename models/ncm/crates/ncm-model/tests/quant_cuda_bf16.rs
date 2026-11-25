//! Quantised-matmul kernels on CUDA in bf16, the benchmark's dtype. Needs `--features cuda`
//! and a GPU; CI builds neither, so this file compiles to nothing there.
#![cfg(feature = "cuda")]

mod quant_common;
mod quant_cuda;

use burn::tensor::FloatDType;
use quant_cuda::*;

const DTYPE: FloatDType = FloatDType::BF16;

#[test]
fn quantised_layers_match_the_host_product_bf16() {
    let (device, _lock) = setup(DTYPE);
    // The GEMV keeps f32 inside and rounds its output to bf16 (2^-9); the GEMM path also
    // rounds the decoded weight to bf16 before Burn's matmul.
    layers_match_host(&device, DTYPE, 1e-2);
}

#[test]
fn quantised_tiny_models_bf16() {
    let (device, _lock) = setup(DTYPE);
    // Every activation is rounded to bf16 on both paths, so they differ by bf16 noise.
    tiny_models_match(&device, 0.05, 0.05);
}

/// Timing only (`--ignored --nocapture`, release build): dense, q8 and q4 one-row products.
#[test]
#[ignore = "timing run, not a check"]
fn gemv_timing_bf16() {
    let (device, _lock) = setup(DTYPE);
    time_gemv(&device, DTYPE);
}
