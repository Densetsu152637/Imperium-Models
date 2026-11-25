//! Quantised-matmul kernels on CUDA, shared by `quant_cuda_f32.rs` and `quant_cuda_bf16.rs` (a
//! device's float dtype is fixed per process, so each dtype gets its own test binary).
//!
//! Layers: the kernels (GEMV up to `GEMV_ROWS` rows, dequantise + matmul above) against a host
//! f64 product of the same inputs and the host-decoded weight. Models: tiny models of both kinds
//! and designs quantised from fixed dense weights, kernels against the dense model and against
//! the tensor-op reference on the same device. The kernel switch is process-wide, so the tests
//! of one binary hold a lock.

#![allow(dead_code)]

use std::sync::{Mutex, MutexGuard};

use burn::tensor::{Device, FloatDType, Tensor, TensorData};
use ncm_model::quant::format::{Format, quantize};
use ncm_model::quant::{GEMV_ROWS, QLinear, QuantSpec, kernels_active, set_quant_kernels};

use crate::quant_common::{agreement, fixed_dense, tiny_configs, values};

static SWITCH: Mutex<()> = Mutex::new(());

/// The CUDA device at `dtype`, and the lock on the kernel switch.
pub fn setup(dtype: FloatDType) -> (Device, MutexGuard<'static, ()>) {
    let guard = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
    let mut device = Device::cuda(0);
    if device.configure(dtype).is_err() {
        assert_eq!(device.settings().float_dtype, dtype, "device dtype already fixed");
    }
    set_quant_kernels(true);
    assert!(kernels_active(&device), "quantised kernels are not active on the CUDA device");
    (device, guard)
}

fn random(n: usize, seed: u64, amplitude: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * amplitude
        })
        .collect()
}

/// (d_in, d_out): ragged and small, then the 250M models' shapes (DeltaNet input projection,
/// both MLP down projections) and a wide head-like layer.
const SHAPES: [(usize, usize); 6] =
    [(70, 9), (256, 130), (1024, 4112), (1700, 1024), (2240, 1024), (512, 8192)];

/// Every layer shape, format and row count (GEMV and GEMM paths) within `tol` of the largest
/// output magnitude of the host product.
pub fn layers_match_host(device: &Device, dtype: FloatDType, tol: f64) {
    let mut worst = 0f64;
    for (i, &(d_in, d_out)) in SHAPES.iter().enumerate() {
        let w = random(d_out * d_in, i as u64 + 1, 0.05);
        for format in [Format::q8(), Format::q4(64), Format::q4(128)] {
            let packed = quantize(&w, d_out, d_in, format);
            let decoded = packed.dequantize();
            let layer = QLinear::from_packed(&packed, device);
            for rows in [1, 3, GEMV_ROWS, GEMV_ROWS + 1, 37] {
                let x = Tensor::<2>::from_data(
                    TensorData::new(random(rows * d_in, rows as u64, 1.0), [rows, d_in]),
                    device,
                )
                .cast(dtype);
                let xs = values(x.clone()); // exactly what the kernel reads
                let got = values(layer.forward(x));
                let mut want = vec![0f64; rows * d_out];
                for r in 0..rows {
                    for n in 0..d_out {
                        want[r * d_out + n] = (0..d_in)
                            .map(|k| xs[r * d_in + k] as f64 * decoded[n * d_in + k] as f64)
                            .sum();
                    }
                }
                let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
                let err =
                    got.iter().zip(&want).map(|(g, w)| (*g as f64 - w).abs()).fold(0f64, f64::max)
                        / scale;
                worst = worst.max(err);
                assert!(
                    err <= tol,
                    "[{d_in} -> {d_out}] {format:?} rows {rows}: error {err:.2e} of {scale:.3}"
                );
            }
        }
    }
    eprintln!("layers: worst error {worst:.2e} of the largest output ({dtype:?})");
}

/// Tiny models quantised on the device: the kernel path against the dense model (printed;
/// q8 asserted within `q8_tol` of the largest logit) and against the tensor-op reference on the
/// same device (within `path_tol`).
pub fn tiny_models_match(device: &Device, q8_tol: f32, path_tol: f32) {
    for (name, config) in tiny_configs() {
        let dense = fixed_dense(&config, device);
        let want = dense.logits(device);
        for spec in [
            QuantSpec { format: Format::q8(), head: false },
            QuantSpec { format: Format::q8(), head: true },
            QuantSpec { format: Format::q4(64), head: false },
            QuantSpec { format: Format::q4(64), head: true },
            QuantSpec { format: Format::q4(128), head: true },
        ] {
            let model = dense.quantized(&config, spec, device);
            let got = model.logits(device);
            set_quant_kernels(false);
            let reference = model.logits(device);
            set_quant_kernels(true);
            let a = agreement(&want, &got, config.vocab_size());
            let p = agreement(&reference, &got, config.vocab_size());
            eprintln!(
                "{name:>12} {:<12} vs dense: max {:.4} (rel {:.4}) mean {:.5} argmax {:.3}; \
                 vs reference: rel {:.4}",
                spec.name(),
                a.max_abs,
                a.max_rel,
                a.mean_abs,
                a.argmax,
                p.max_rel
            );
            assert!(p.max_rel <= path_tol, "{name} {}: kernels vs reference {p:?}", spec.name());
            if let Some(got) = model.step_logits(device) {
                set_quant_kernels(false);
                let reference = model.step_logits(device).expect("token baseline");
                set_quant_kernels(true);
                let p = agreement(&reference, &got, config.vocab_size());
                eprintln!(
                    "{name:>12} {:<12} decode steps vs reference: rel {:.4}",
                    spec.name(),
                    p.max_rel
                );
                assert!(p.max_rel <= path_tol, "{name} {}: decode {p:?}", spec.name());
            }
            if spec.format.bits == 8 {
                assert!(a.max_rel <= q8_tol, "{name} {}: {a:?}", spec.name());
            }
        }
    }
}

/// One-row products through a dense layer and its q8 and q4 versions, timed over many calls
/// queued back to back: microseconds per call and weight bytes streamed per second. Large
/// layers, so the GPU rather than the host sets the pace. Printed only.
pub fn time_gemv(device: &Device, dtype: FloatDType) {
    use std::time::Instant;
    const CALLS: usize = 200;
    // The last exceeds the 64 MiB L2 of an RTX 5080 in every format; the others fit it partly.
    for (d_in, d_out) in [(1024, 4112), (4096, 16384), (8192, 24576)] {
        let w = random(d_out * d_in, 9, 0.05);
        let dense = Tensor::<2>::from_data(TensorData::new(w.clone(), [d_out, d_in]), device);
        let layers = [
            ("dense", dense_layer(dense.cast(dtype).transpose(), d_in, d_out, device)),
            ("q8", QLinear::from_packed(&quantize(&w, d_out, d_in, Format::q8()), device)),
            ("q4-g64", QLinear::from_packed(&quantize(&w, d_out, d_in, Format::q4(64)), device)),
        ];
        let x = Tensor::<2>::from_data(TensorData::new(random(d_in, 3, 1.0), [1, d_in]), device)
            .cast(dtype);
        for (name, layer) in &layers {
            for _ in 0..10 {
                let _ = layer.forward(x.clone());
            }
            device.sync().expect("sync");
            let start = Instant::now();
            for _ in 0..CALLS {
                let _ = layer.forward(x.clone());
            }
            device.sync().expect("sync");
            let per_call = start.elapsed().as_secs_f64() / CALLS as f64;
            let bytes = ncm_model::quant::weight_bytes(layer);
            eprintln!(
                "[{d_in} -> {d_out}] {name:<7} {:>8.1} us/call  {:>6.1} GB/s of weights ({:.1} MiB)",
                per_call * 1e6,
                bytes as f64 / per_call / 1e9,
                bytes as f64 / (1u64 << 20) as f64
            );
        }
    }
}

/// A dense layer holding `w` `[d_in, d_out]`: a dense `QLinear` with its weight replaced.
fn dense_layer(w: Tensor<2>, d_in: usize, d_out: usize, device: &Device) -> QLinear {
    use burn::module::{Module, ModuleMapper, Param};
    struct Set(Option<Tensor<2>>);
    impl ModuleMapper for Set {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
            let w = self.0.take().expect("one weight");
            let dims = param.shape().dims::<D>();
            param.map(|_| w.reshape(dims))
        }
    }
    QLinear::new(d_in, d_out, device).map(&mut Set(Some(w)))
}
