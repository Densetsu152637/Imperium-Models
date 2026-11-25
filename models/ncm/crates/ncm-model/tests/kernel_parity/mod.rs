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
use burn::tensor::{Device, Distribution, FloatDType, Int, Tensor, TensorData};
use ncm_model::delta::{DeltaCache, DeltaSpec, GatedDeltaNet};
use ncm_model::fused::{active, argmax_last, set_cuda_graphs, set_fused_kernels};
use ncm_model::layers::Rope;
use ncm_model::ncm::GraphedDecoder;
use ncm_model::quant::{self, QuantSpec, format::Format};
use ncm_model::retrieval::{LatentCache, RetrievalSpec, SparseRetrieval};
use ncm_model::{Acceptance, NcmLm, SpecStats, Speculator, tiny_ncm, tiny_ncm_v2, tiny_v2};

fn ints<const D: usize>(t: Tensor<D, Int>) -> Vec<i64> {
    t.into_data().convert::<i64>().try_to_vec::<i64>().expect("int tensor")
}

/// The argmax kernel picks the same index as Burn's `argmax` (the first maximum), including on
/// rows with repeated maxima.
pub fn argmax_matches(device: &Device) {
    device.seed(3);
    let x = Tensor::<3>::random([2, 3, 32000], Distribution::Normal(0.0, 1.0), device);
    let got = ints(argmax_last(x.clone()));
    let want = ints(reference(|| x.argmax(2)));
    assert_eq!(got, want);
    // Rows of equal values: every unit holds a maximum, the first index must win.
    let flat =
        Tensor::<3>::from_data(TensorData::new(vec![1.5f32; 2 * 1000], [2, 1, 1000]), device);
    assert_eq!(ints(argmax_last(flat)), vec![0, 0]);
}

/// The matrix-vector kernels against an f32 CPU matmul: the models' decode shapes (one row and a
/// concept's four), split and unsplit `K`, a row-major and a transposed (tied-embedding) weight.
pub fn gemv_matches(device: &Device, tol: f32) {
    device.seed(4);
    let normal = Distribution::Normal(0.0, 1.0);
    for (rows, k, n) in
        [(1, 1024, 4112), (1, 2240, 1024), (4, 512, 1536), (3, 37, 70), (2, 5000, 64)]
    {
        let x = Tensor::<3>::random([1, rows, k], normal, device);
        let w = Tensor::<2>::random([k, n], normal, device);
        let got = values(ncm_model::kernels::gemv(x.clone(), w.clone()));
        let want = values(to_cpu(&x).matmul(to_cpu(&w).unsqueeze::<3>()));
        let err = rel_error(&got, &want);
        eprintln!("gemv {rows}x{k}x{n}: error over scale {err:.2e}");
        assert!(err <= tol, "gemv {rows}x{k}x{n}: error {err:.2e}");
    }
    for (rows, k, n) in [(1, 512, 32000), (4, 64, 300)] {
        let x = Tensor::<3>::random([1, rows, k], normal, device);
        let emb = Tensor::<2>::random([n, k], normal, device);
        let got = values(ncm_model::kernels::gemv(x.clone(), emb.clone().transpose()));
        let want = values(to_cpu(&x).matmul(to_cpu(&emb).transpose().unsqueeze::<3>()));
        let err = rel_error(&got, &want);
        eprintln!("gemv transposed {rows}x{k}x{n}: error over scale {err:.2e}");
        assert!(err <= tol, "gemv transposed {rows}x{k}x{n}: error {err:.2e}");
    }
}

/// Greedy NCM decode of `concepts` concepts after a 64-token prompt, through the decoder and
/// backbone graphs when `graphs` is on: the tokens and conditions of every concept.
fn ncm_decode(
    model: &NcmLm,
    graphs: bool,
    concepts: usize,
    device: &Device,
) -> (Vec<Vec<i64>>, Vec<Vec<f32>>) {
    set_cuda_graphs(graphs);
    let prompt = Tensor::<2, Int>::from_data(
        TensorData::new((0..64).map(|i: i32| i * 37 % 256).collect::<Vec<_>>(), [1, 64]),
        device,
    );
    let mut cache = model.make_cache(64 + 4 * concepts + 4, 1, device);
    let mut cond = model.prefill(prompt, &mut cache, 16);
    let mut decoder = GraphedDecoder::default();
    let (mut tokens, mut conds) = (Vec::new(), Vec::new());
    for _ in 0..concepts {
        let next = decoder.decode(model, cond, &mut cache);
        tokens.push(ints(next.clone()));
        cond = model.append_tokens(next, &mut cache);
        conds.push(values(cond.clone()));
    }
    set_cuda_graphs(true);
    if graphs {
        assert!(decoder.is_captured(), "decoder not captured: {:?}", decoder.failure());
        let backbone = &cache.backbone_graph;
        let v2 = cache.backbone.iter().all(|c| !matches!(c, ncm_model::LayerCache::Kv(_)));
        assert_eq!(backbone.is_captured(), v2, "backbone graph: {:?}", backbone.failure());
    }
    (tokens, conds)
}

/// The quantisations the graphed-decode checks run with: q8, and q4 in groups of 64 with the
/// output head quantised too.
pub fn quant_specs() -> [QuantSpec; 2] {
    [
        QuantSpec { format: Format::q8(), head: false },
        QuantSpec { format: Format::q4(64), head: true },
    ]
}

/// A tiny NCM of `cfg` with random weights, its linear layers (draft head included) quantised
/// per `spec`.
fn build_ncm(cfg: &ncm_model::NcmConfig, spec: Option<QuantSpec>, device: &Device) -> NcmLm {
    device.seed(9);
    quant::build(spec, || NcmLm::new(cfg, device))
}

/// The decoder graph (both designs) and the backbone graph (v2) give the same tokens and
/// conditions as the same kernels run eagerly, concept after concept, with the linear layers
/// dense or quantised per `spec` (the quantised GEMV kernels then run inside the graphs). Tiny
/// configs: the 8-token encoder window fills during the prompt, so both graphs are captured on
/// the first concept, and the v2 backbone's retrieval (blocks of 2, top 2, window 4) turns
/// sparse within the run.
pub fn graphed_decode_matches(device: &Device, spec: Option<QuantSpec>) {
    for cfg in [tiny_ncm(8, 4), tiny_ncm_v2(8, tiny_v2(2, 2, 4))] {
        let model = build_ncm(&cfg, spec, device);
        let (tokens_e, conds_e) = ncm_decode(&model, false, 12, device);
        let (tokens_g, conds_g) = ncm_decode(&model, true, 12, device);
        for (concept, (te, tg)) in tokens_e.iter().zip(&tokens_g).enumerate() {
            assert_eq!(tg, te, "concept {concept}: tokens differ");
            let err = rel_error(&conds_g[concept], &conds_e[concept]);
            assert!(err <= 1e-5, "concept {concept}: conditions differ by {err:.2e}");
        }
    }
}

/// Self-speculative greedy decode of `concepts` concepts after the same prompt as
/// [`ncm_decode`], with the draft and verify passes replayed from graphs when `graphs` is on:
/// the tokens of every concept and the speculator's counts.
fn spec_decode(
    model: &NcmLm,
    graphs: bool,
    acceptance: Acceptance,
    concepts: usize,
    device: &Device,
) -> (Vec<Vec<i64>>, SpecStats) {
    set_cuda_graphs(graphs);
    let prompt = Tensor::<2, Int>::from_data(
        TensorData::new((0..64).map(|i: i32| i * 37 % 256).collect::<Vec<_>>(), [1, 64]),
        device,
    );
    let mut cache = model.make_cache(64 + 4 * concepts + 4, 1, device);
    let mut cond = model.prefill(prompt, &mut cache, 16);
    let mut speculator = Speculator::new(acceptance);
    let mut tokens = Vec::new();
    for _ in 0..concepts {
        let next = speculator.decode(model, cond, &cache);
        tokens.push(ints(next.clone()));
        cond = model.append_tokens(next, &mut cache);
    }
    set_cuda_graphs(true);
    if graphs {
        let failure = speculator.failure();
        assert!(speculator.is_captured(), "speculative graphs not captured: {failure:?}");
    }
    (tokens, speculator.stats())
}

/// The draft and verify graphs (both designs; linear layers dense or quantised per `spec`) give
/// the same tokens and counts as the same kernels run eagerly, under real verification and under an assumed acceptance of one half
/// (several verify replays per concept). With `exact`, the speculative tokens also equal plain
/// greedy decoding through the decoder graph (f32; in bf16 the parallel and step-by-step passes
/// round differently and a near tie of the two best logits may go either way, so mismatching
/// concepts are only counted).
pub fn graphed_speculation_matches(device: &Device, exact: bool, spec: Option<QuantSpec>) {
    for cfg in [tiny_ncm(8, 4), tiny_ncm_v2(8, tiny_v2(2, 2, 4))] {
        let model = build_ncm(&cfg, spec, device);
        for acceptance in [Acceptance::Verify, Acceptance::Assume(0.5)] {
            let (tokens_e, stats_e) = spec_decode(&model, false, acceptance, 12, device);
            let (tokens_g, stats_g) = spec_decode(&model, true, acceptance, 12, device);
            assert_eq!(tokens_g, tokens_e, "{acceptance:?}: graphed tokens differ");
            assert_eq!(stats_g, stats_e, "{acceptance:?}: graphed passes differ");
            eprintln!("{} {acceptance:?}: {stats_g:?}", cfg.attention.name());
        }
        let (plain, _) = ncm_decode(&model, true, 12, device);
        let (spec, _) = spec_decode(&model, true, Acceptance::Verify, 12, device);
        let differ = plain.iter().zip(&spec).filter(|(p, s)| p != s).count();
        eprintln!("{}: {differ} of 12 concepts differ from plain greedy", cfg.attention.name());
        if exact {
            assert_eq!(spec, plain, "speculative greedy differs from plain greedy");
        }
    }
}

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
