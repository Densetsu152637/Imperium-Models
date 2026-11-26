//! Quantised-matmul kernels on CUDA in f32. Needs `--features cuda` and a GPU; CI builds
//! neither, so this file compiles to nothing there.
#![cfg(feature = "cuda")]

mod quant_common;
mod quant_cuda;

use burn::tensor::FloatDType;
use quant_cuda::*;

const DTYPE: FloatDType = FloatDType::F32;

#[test]
fn quantised_layers_match_the_host_product_f32() {
    let (device, _lock) = setup(DTYPE);
    // f32 throughout in the GEMV; Burn's f32 matmul on the GEMM path may round its inputs to
    // TF32 (10-bit mantissa) on the tensor cores.
    layers_match_host(&device, DTYPE, 1e-3);
}

#[test]
fn quantised_tiny_models_f32() {
    let (device, _lock) = setup(DTYPE);
    tiny_models_match(&device, 0.02, 1e-3);
}
