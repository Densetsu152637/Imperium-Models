//! Fused decode kernels against the reference paths, shared by `kernels_f32.rs` and
//! `kernels_bf16.rs` (a device's float dtype is fixed per process, so each dtype gets its own
//! test binary).
//!
//! Every check runs one layer three ways from equal caches: the fused kernels on CUDA, the
//! reference tensor code on CUDA (same dtype), and the reference on the CPU in f32 with the same
//! weights and inputs (exact conversions from the CUDA dtype). The fused path must be within the
//! tolerance of the f32 CPU result; the CUDA reference's own error is logged next to it. The
//! process-wide switch picks the CUDA path, so the tests of one binary hold a lock while they run.

// Each test binary uses its own part of this module.
#![allow(dead_code)]

use std::sync::{Mutex, MutexGuard};

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::{Device, Distribution, FloatDType, Tensor};
use ncm_model::delta::{DeltaCache, DeltaSpec, GatedDeltaNet};
use ncm_model::fused::{active, set_fused_kernels};
use ncm_model::layers::Rope;
use ncm_model::retrieval::{LatentCache, RetrievalSpec, SparseRetrieval};

static SWITCH: Mutex<()> = Mutex::new(());

/// The CUDA device at `dtype`, and the lock on the fused-kernel switch.
pub fn setup(dtype: FloatDType) -> (Device, MutexGuard<'static, ()>) {
    let guard = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
    let mut device = Device::cuda(0);
    if device.configure(dtype).is_err() {
        assert_eq!(device.settings().float_dtype, dtype, "device dtype already fixed");
    }
    set_fused_kernels(true);
    assert!(active(&device), "fused kernels are not active on the CUDA device");
    (device, guard)
}

/// Runs `f` on the reference path (fused kernels off) and restores the switch.
pub fn reference<T>(f: impl FnOnce() -> T) -> T {
    set_fused_kernels(false);
    let out = f();
    set_fused_kernels(true);
    out
}

pub fn values<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    let data = t.try_into_data().unwrap_or_else(|e| panic!("reading a tensor failed: {e:#?}"));
    data.convert::<f32>().try_to_vec::<f32>().expect("float tensor")
}

/// The CPU device the f32 reference runs on.
pub fn cpu() -> Device {
    Device::flex()
}

/// `x` on the CPU in f32.
pub fn to_cpu<const D: usize>(x: &Tensor<D>) -> Tensor<D> {
    x.clone().to_device(&cpu()).cast(FloatDType::F32)
}

/// Moves every float parameter to the CPU in f32.
struct ToCpuF32;

impl ModuleMapper for ToCpuF32 {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        Param::from_tensor(to_cpu(&param.val()))
    }
}

/// An f32 CPU copy of `module`.
pub fn cpu_copy<M: Module>(module: &M) -> M {
    module.clone().map(&mut ToCpuF32)
}

/// Largest absolute difference over the largest magnitude of `want`.
pub fn rel_error(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0f32, |m, w| m.max(w.abs())).max(1e-6);
    got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max) / scale
}

/// Every element of `got` finite and within `tol` times the largest magnitude of `want`; logs
/// the fused error next to the CUDA reference's (`reference`).
pub fn assert_close(what: &str, got: &[f32], reference: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let scale = want.iter().fold(0f32, |m, w| m.max(w.abs())).max(1e-6);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what}: element {i} is {g}");
        assert!(
            (g - w).abs() <= tol * scale,
            "{what}: element {i}: got {g}, want {w} (tol {}, scale {scale})",
            tol * scale
        );
    }
    eprintln!(
        "{what}: error over scale {scale:.3}: fused {:.2e}, cuda reference {:.2e}",
        rel_error(got, want),
        rel_error(reference, want)
    );
}

fn delta_layer(head_dim: usize, heads: usize, device: &Device) -> GatedDeltaNet {
    let spec = DeltaSpec { d: heads * head_dim, head_dim, conv: 4, chunk: 64, eps: 1e-6 };
    GatedDeltaNet::new(&spec, device)
}

/// The fused DeltaNet step: one-token steps and 3- and 4-token runs (the NCM encoder's
/// per-concept call) after a chunked prefill, batch 2. Head widths 128 (the models'), 32 and 16
/// (the tiny configs); 96 channels also exercise cubecl's row pitching.
pub fn delta_matches(device: &Device, tol: f32) {
    for (head_dim, heads) in [(128, 3), (32, 1), (16, 2)] {
        device.seed(11);
        let layer = delta_layer(head_dim, heads, device);
        let exact = cpu_copy(&layer);
        let d = heads * head_dim;
        let normal = Distribution::Normal(0.0, 1.0);
        // Larger inputs than the 0.02-std weights alone give, so the gates and norms vary.
        let prompt = Tensor::<3>::random([2, 70, d], normal, device).mul_scalar(20.0);
        let mut fused: DeltaCache = layer.make_cache(2, device);
        let mut cuda: DeltaCache = layer.make_cache(2, device);
        let mut want: DeltaCache = exact.make_cache(2, &cpu());
        reference(|| {
            layer.forward(prompt.clone(), Some(&mut fused));
            layer.forward(prompt.clone(), Some(&mut cuda));
        });
        exact.forward(to_cpu(&prompt), Some(&mut want));
        for (step, t) in [1, 1, 4, 1, 3, 1].into_iter().enumerate() {
            let x = Tensor::<3>::random([2, t, d], normal, device).mul_scalar(20.0);
            let got = values(layer.forward(x.clone(), Some(&mut fused)));
            let base = values(reference(|| layer.forward(x.clone(), Some(&mut cuda))));
            let exp = values(exact.forward(to_cpu(&x), Some(&mut want)));
            let what = format!("delta hd {head_dim} step {step} (t {t})");
            assert_close(&what, &got, &base, &exp, tol);
        }
        assert_eq!(fused.len(), want.len());
    }
}

/// One retrieval check: layer shape, positions prefilled, positions decoded after them.
pub struct RetrievalCase {
    pub name: &'static str,
    pub spec: RetrievalSpec,
    pub prefill: usize,
    pub steps: usize,
    /// Whether the block selection is stable in bf16. With 34 candidates for 32 picks, keys
    /// rounded to bf16 in the CUDA caches (and f32 on the CPU) put the boundary blocks within
    /// rounding of each other at most steps, so all three paths may select differently; that
    /// case runs in f32 only.
    pub bf16_stable: bool,
}

fn retrieval_spec(
    model_heads: bool,
    block: usize,
    topk: usize,
    local: usize,
    dense: bool,
) -> RetrievalSpec {
    let (heads, nope_dim, rope_dim, v_dim, latent, index_heads, index_dim) =
        if model_heads { (8, 128, 64, 128, 512, 4, 128) } else { (2, 16, 8, 16, 32, 2, 16) };
    RetrievalSpec {
        d: 256,
        heads,
        nope_dim,
        rope_dim,
        v_dim,
        latent,
        index_heads,
        index_dim,
        block,
        topk,
        local,
        dense,
        eps: 1e-6,
    }
}

/// The cases: the models' head shapes with small blocks (dense, then sparse, crossing block and
/// window boundaries), the tiny configs' shapes, dense mode, and the 250M selection settings
/// (blocks of 64, top 32, window 1,024) past the point where selection starts.
pub fn retrieval_cases() -> Vec<RetrievalCase> {
    vec![
        RetrievalCase {
            name: "model heads, block 8 top 4 local 16",
            spec: retrieval_spec(true, 8, 4, 16, false),
            prefill: 30,
            steps: 70,
            bf16_stable: true,
        },
        RetrievalCase {
            name: "tiny heads, block 2 top 2 local 4",
            spec: retrieval_spec(false, 2, 2, 4, false),
            prefill: 3,
            steps: 40,
            bf16_stable: true,
        },
        RetrievalCase {
            name: "model heads, dense",
            spec: retrieval_spec(true, 8, 4, 16, true),
            prefill: 40,
            steps: 20,
            bf16_stable: true,
        },
        RetrievalCase {
            name: "model heads, 250M selection",
            spec: retrieval_spec(true, 64, 32, 1024, false),
            prefill: 3150,
            steps: 40,
            bf16_stable: false,
        },
    ]
}

/// Decodes `case.steps` positions one at a time after a chunked reference prefill, batch 2, and
/// checks every step.
pub fn retrieval_matches(device: &Device, case: &RetrievalCase, tol: f32) {
    let spec = &case.spec;
    device.seed(5);
    let layer = SparseRetrieval::new(spec, device);
    let exact = cpu_copy(&layer);
    let capacity = case.prefill + case.steps;
    let mut fused: LatentCache = layer.make_cache(capacity, 2, device);
    let mut cuda: LatentCache = layer.make_cache(capacity, 2, device);
    let mut want: LatentCache = exact.make_cache(capacity, 2, &cpu());
    let normal = Distribution::Normal(0.0, 1.0);
    let rope = |start: usize, len: usize, device: &Device| {
        Rope::new(start, len, spec.rope_dim, 1e6, 1.0, device)
    };
    let mut start = 0;
    while start < case.prefill {
        let len = 512.min(case.prefill - start);
        let x = Tensor::<3>::random([2, len, spec.d], normal, device).mul_scalar(3.0);
        reference(|| {
            layer.forward(x.clone(), start, &rope(start, len, device), Some(&mut fused));
            layer.forward(x.clone(), start, &rope(start, len, device), Some(&mut cuda));
        });
        exact.forward(to_cpu(&x), start, &rope(start, len, &cpu()), Some(&mut want));
        start += len;
    }
    // A step passes when the fused output is within `tol` of the f32 CPU result, or of the CUDA
    // reference: block scores of the CUDA paths and the CPU differ in the last bits (their
    // projections do), so at a near tie for the last selected block both CUDA paths may pick
    // another block than the CPU. Such steps are counted.
    let (mut worst, mut worst_reference, mut flips) = (0f32, 0f32, 0);
    for pos in case.prefill..capacity {
        let x = Tensor::<3>::random([2, 1, spec.d], normal, device).mul_scalar(3.0);
        let got = values(layer.forward(x.clone(), pos, &rope(pos, 1, device), Some(&mut fused)));
        let base = values(reference(|| {
            layer.forward(x.clone(), pos, &rope(pos, 1, device), Some(&mut cuda))
        }));
        let exp = values(exact.forward(to_cpu(&x), pos, &rope(pos, 1, &cpu()), Some(&mut want)));
        assert!(got.iter().all(|g| g.is_finite()), "{} pos {pos}: non-finite output", case.name);
        let (err, err_reference) = (rel_error(&got, &exp), rel_error(&base, &exp));
        let err_vs_reference = rel_error(&got, &base);
        assert!(
            err <= tol || err_vs_reference <= tol,
            "{} pos {pos}: error over scale {err:.2e} against the f32 CPU result and \
             {err_vs_reference:.2e} against the CUDA reference (which is off by \
             {err_reference:.2e})",
            case.name
        );
        if err <= tol {
            worst = worst.max(err);
            worst_reference = worst_reference.max(err_reference);
        } else {
            flips += 1;
        }
    }
    assert_eq!(fused.len(), capacity);
    assert!(flips * 10 <= case.steps, "{}: {flips} near-tie steps of {}", case.name, case.steps);
    eprintln!(
        "{}: {} steps, worst error over scale: fused {worst:.2e}, cuda reference \
         {worst_reference:.2e}; {flips} near-tie steps matched the CUDA reference instead",
        case.name, case.steps
    );
}
