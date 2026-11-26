//! The ONNX Runtime backend against the Burn reference on a tiny random token baseline (plan
//! goal 3): per-step logits, identical greedy tokens over 64 steps, prefix snapshots, and the
//! int8 QDQ export against Burn's own `--quant q8`.
//!
//! Needs the pinned ONNX Runtime shared library (`ORT_DYLIB_PATH`); CI fetches it in
//! `scripts/check.sh`. Run with `cargo test -p ncm-ort --test parity`.

use std::path::Path;

use burn::tensor::{Device, Int, Tensor, TensorData};
use ncm_exec::{ExecBackend, ExecSession, PickSpec, SnapshotInfo, StepOut};
use ncm_export::weights::Weights;
use ncm_export::{ExportOptions, Target, WeightForm, export};
use ncm_model::quant::{Format, QuantSpec, quantize_model};
use ncm_model::weights::save_weights;
use ncm_model::{Config, LayerCache, TokConfig, TokenLm, tiny_tok};
use ncm_ort::{OrtBackend, OrtOptions};

const CTX: usize = 96;
const PROMPT: usize = 21;
const STEPS: usize = 64;

/// The tests share the CPU device's random stream and ORT's one environment.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn config() -> TokConfig {
    TokConfig { vocab_size: 272, ..tiny_tok(8) }
}

fn device() -> Device {
    let device = Device::flex();
    device.seed(3);
    device
}

fn ids(device: &Device, tokens: &[i32]) -> Tensor<2, Int> {
    Tensor::from_data(TensorData::new(tokens.to_vec(), [1, tokens.len()]), device)
}

fn host(logits: Tensor<3>) -> Vec<f32> {
    logits.into_data().convert::<f32>().try_into_vec::<f32>().expect("f32")
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best })
}

fn prompt() -> Vec<i32> {
    (0..PROMPT as i32).map(|i| 16 + (i * 11) % 200).collect()
}

/// Saves the model's weights, exports them as `form` into `dir/onnx` and opens the export.
fn exported(dir: &Path, model: &TokenLm, form: WeightForm) -> OrtBackend {
    let cfg = config();
    let weights = dir.join("tiny.safetensors");
    save_weights(model, &Config::Tok(cfg.clone()), &weights).expect("saves the model");
    let out = dir.join("onnx");
    let opts = ExportOptions { ctx: CTX, buckets: vec![8, 16], form, target: Target::Cpu };
    export(&Config::Tok(cfg), &Weights::open(&weights).unwrap(), &out, &opts).expect("exports");
    OrtBackend::load(&OrtOptions { dir: out, ..Default::default() }).expect("loads the export")
}

/// Greedy decoding of `STEPS` tokens: the ORT session's logits against the Burn model's, at
/// every step; returns (max |dlogit|, the two token streams).
fn compare(model: &TokenLm, ort: &OrtBackend, device: &Device) -> (f32, Vec<usize>, Vec<usize>) {
    let mut session = ort.new_session(CTX).unwrap();
    let stream = prompt();
    session.prefill(&stream, 512).unwrap();
    let mut row = session.pending_logits().expect("logits after the prompt");

    let mut caches = model.make_cache(CTX, 1, device);
    let mut want =
        host(model.prefill(ids(device, &stream), &mut caches, 512).expect("prefill logits"));

    let (mut worst, mut ours, mut theirs) = (0f32, Vec::new(), Vec::new());
    for _ in 0..STEPS {
        worst = worst.max(row.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max));
        let (a, b) = (argmax(&row), argmax(&want));
        ours.push(a);
        theirs.push(b);
        // Both feed the Burn reference's token so one divergence does not hide later ones.
        let StepOut::Logits(next) = session.decode_step(&[b as i32], &PickSpec::host()).unwrap()
        else {
            panic!("a host pick returns logits")
        };
        row = next;
        want = host(model.step(ids(device, &[b as i32]), &mut caches));
    }
    (worst, ours, theirs)
}

#[test]
fn f32_logits_match_the_burn_reference_and_greedy_tokens_are_identical() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let device = device();
    let model = TokenLm::new(&config(), &device);
    let ort = exported(dir.path(), &model, WeightForm::F32);
    let (worst, ours, theirs) = compare(&model, &ort, &device);
    assert!(worst <= 1e-4, "max |dlogit| over {STEPS} steps = {worst}");
    assert_eq!(ours, theirs, "identical greedy tokens");
}

#[test]
fn int8_qdq_matches_burns_q8_within_the_plan_tolerance() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let device = device();
    let cfg = config();
    let dense = TokenLm::new(&cfg, &device);
    let ort = exported(dir.path(), &dense, WeightForm::Q8);
    // Burn's q8 of the same weights; the exporter leaves the tied head dense.
    let spec = QuantSpec { format: Format::q8(), head: false };
    let quant =
        quantize_model(&dense, spec, &device, || TokenLm::new(&cfg, &device)).expect("quantises");
    let (worst, ours, theirs) = compare(&quant, &ort, &device);
    assert!(worst <= 0.05, "max |dlogit| (tiny model, q8) = {worst}");
    assert_eq!(ours, theirs, "identical greedy tokens on the tiny model");
}

/// A restore returns to the snapshot's state, and a snapshot taken after the prompt carries
/// its logits (a prompt equal to the kept one resumes without a prefill).
#[test]
fn snapshots_restore_into_the_same_buffers() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let device = device();
    let model = TokenLm::new(&config(), &device);
    let ort = exported(dir.path(), &model, WeightForm::F32);
    let mut session = ort.new_session(CTX).unwrap();
    session.prefill(&prompt(), 512).unwrap();
    let snapshot = session.snapshot();
    assert!(snapshot.has_output());
    let first = session.decode_step(&[40], &PickSpec::host()).unwrap();
    session.decode_step(&[41], &PickSpec::host()).unwrap();
    session.restore(&snapshot);
    let again = session.decode_step(&[40], &PickSpec::host()).unwrap();
    assert_eq!(first, again, "the restored state replays the same step");
    session.reset();
    session.prefill(&prompt(), 512).unwrap();
    assert_eq!(session.decode_step(&[40], &PickSpec::host()).unwrap(), first);
}

/// A prompt longer than every ring (the sliding layers wrap) still matches the reference.
#[test]
fn sliding_rings_wrap_like_the_reference() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let device = device();
    let model = TokenLm::new(&config(), &device);
    let ort = exported(dir.path(), &model, WeightForm::F32);
    let mut session = ort.new_session(CTX).unwrap();
    // 40 prompt tokens against a window of 8: five ring wraps before the first decode step.
    let long: Vec<i32> = (0..40).map(|i| 16 + (i * 7) % 200).collect();
    session.prefill(&long, 512).unwrap();
    let got = session.pending_logits().unwrap();
    let mut caches: Vec<LayerCache> = model.make_cache(CTX, 1, &device);
    let want = host(model.prefill(ids(&device, &long), &mut caches, 512).unwrap());
    let d = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    assert!(d <= 1e-4, "after a wrapped prefill: max |dlogit| = {d}");
}

/// The v2 token baseline (DeltaNet and sparse-retrieval layers): ORT's dense graphs below the
/// point the selection starts to differ, the split path (host indexer scores, top-k and gather
/// between graph segments) beyond it, against Burn's own selection. UNVERIFIED: written
/// without a run.
#[test]
fn v2_retrieval_matches_beyond_the_dense_range() {
    use ncm_model::{tiny_tok_v2, tiny_v2};
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let device = device();
    // Blocks of 4, top 2, local window 8: selection differs from dense from position 16 on.
    let cfg = TokConfig { vocab_size: 272, ..tiny_tok_v2(tiny_v2(4, 2, 8)) };
    let model = TokenLm::new(&cfg, &device);
    let weights = dir.path().join("v2.safetensors");
    save_weights(&model, &Config::Tok(cfg.clone()), &weights).expect("saves the model");
    let out = dir.path().join("onnx");
    let opts =
        ExportOptions { ctx: CTX, buckets: vec![], form: WeightForm::F32, target: Target::Cpu };
    export(&Config::Tok(cfg), &Weights::open(&weights).unwrap(), &out, &opts).expect("exports");
    let ort = OrtBackend::load(&OrtOptions { dir: out, ..Default::default() }).expect("loads");
    assert!(!ort.manifest().split_segments().is_empty(), "the export has split graphs");

    let mut session = ort.new_session(CTX).unwrap();
    let stream: Vec<i32> = (0..24).map(|i| 16 + (i * 13) % 200).collect();
    session.prefill(&stream, 512).unwrap();
    let mut row = session.pending_logits().unwrap();
    let mut caches = model.make_cache(CTX, 1, &device);
    let mut want = host(model.prefill(ids(&device, &stream), &mut caches, 512).unwrap());
    let mut worst = 0f32;
    for _ in 0..24 {
        worst = worst.max(row.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max));
        let token = argmax(&want) as i32;
        let StepOut::Logits(next) = session.decode_step(&[token], &PickSpec::host()).unwrap()
        else {
            panic!("a host pick returns logits")
        };
        row = next;
        want = host(model.step(ids(&device, &[token]), &mut caches));
    }
    assert!(worst <= 1e-3, "v2 max |dlogit| across the dense/sparse boundary = {worst}");
}
