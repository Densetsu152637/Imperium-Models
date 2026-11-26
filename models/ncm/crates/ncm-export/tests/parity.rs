//! Per-builder parity of the exported graphs against the Burn reference, on ONNX Runtime's CPU
//! EP (`cargo test -p ncm-export --features ort-tests --test parity`, with `ORT_DYLIB_PATH`
//! pointing at the pinned onnxruntime shared library; CI fetches it).
//!
//! The reference is `ncm_model`'s own gated DeltaNet layer (weights saved from it and read back
//! by the exporter), so a wrong op, layout or constant in a builder shows up as a number here.
//! Tolerances follow the plan (goal 3): f32 max |delta| 1e-4 per output, and the recurrent
//! state, seen through the layer's output, after 8,192 positions.
#![cfg(feature = "ort-tests")]

use std::collections::HashMap;
use std::path::Path;

use burn::tensor::{Device, Tensor, TensorData};
use ncm_export::graph::{Graph, WeightPool};
use ncm_export::layers::{Cx, DeltaDims, DeltaWeights};
use ncm_export::onnx::DType;
use ncm_export::quant::WeightForm;
use ncm_export::weights::Weights;
use ncm_model::Config;
use ncm_model::delta::{DeltaSpec, GatedDeltaNet};
use ncm_model::tiny_tok;
use ncm_model::weights::save_weights;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor as OrtTensor;

const D: usize = 32;
const HEAD_DIM: usize = 16;
const HEADS: usize = D / HEAD_DIM;
const CONV: usize = 4;
const CHUNK: usize = 8;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn dims() -> DeltaDims {
    DeltaDims { d: D, heads: HEADS, head_dim: HEAD_DIM, conv: CONV, chunk: CHUNK, eps: 1e-6 }
}

/// The Burn layer, and the exporter's view of its weights (saved and read back).
fn layer(dir: &Path) -> (GatedDeltaNet, DeltaWeights, Device) {
    let device = Device::flex();
    device.seed(11);
    let spec = DeltaSpec { d: D, head_dim: HEAD_DIM, conv: CONV, chunk: CHUNK, eps: 1e-6 };
    let net = GatedDeltaNet::new(&spec, &device);
    let path = dir.join("delta.safetensors");
    save_weights(&net, &Config::Tok(tiny_tok(8)), &path).expect("saves the layer");
    let w = Weights::open(&path).expect("reads it back");
    let get = |n: &str| w.get(n).unwrap_or_else(|e| panic!("{n}: {e}"));
    let weights = DeltaWeights {
        proj: get("proj.weight"),
        conv: get("conv"),
        a_log: get("a_log"),
        dt_bias: get("dt_bias"),
        norm: get("norm.gamma"),
        o: get("o.weight"),
    };
    (net, weights, device)
}

fn write_graph(dir: &Path, name: &str, g: &Graph, pool: &WeightPool) -> std::path::PathBuf {
    g.validate().expect("a well-formed graph");
    std::fs::write(dir.join("weights.bin"), pool.external_bytes()).unwrap();
    let path = dir.join(format!("{name}.onnx"));
    std::fs::write(&path, g.bytes(pool, "weights.bin")).unwrap();
    path
}

/// One environment for the process (the shared library comes from `ORT_DYLIB_PATH`).
fn environment() -> &'static Environment {
    static ENV: std::sync::OnceLock<Environment> = std::sync::OnceLock::new();
    ENV.get_or_init(|| ort::init().build().expect("an ONNX Runtime environment"))
}

fn session(path: &Path) -> Session {
    Session::builder(environment())
        .expect("a session builder (is ORT_DYLIB_PATH set?)")
        .commit_from_file(path)
        .expect("ONNX Runtime loads the graph")
}

/// Runs the session on named f32 inputs; outputs by name.
fn run(session: &mut Session, inputs: &[(&str, &[usize], &[f32])]) -> HashMap<String, Vec<f32>> {
    let mut feed: Vec<(String, ort::session::SessionInputValue<'_>)> = Vec::new();
    for (name, shape, data) in inputs {
        let t = OrtTensor::from_array((shape.to_vec(), data.to_vec())).expect("an input tensor");
        feed.push(((*name).to_owned(), t.into()));
    }
    let outputs = session.run(feed).expect("the graph runs");
    outputs
        .iter()
        .map(|(name, value)| {
            let (_, data) = value.try_extract_tensor::<f32>().expect("f32 output");
            (name.to_owned(), data.to_vec())
        })
        .collect()
}

fn pseudo(seed: &mut u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            ((*seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
        })
        .collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn host(t: Tensor<3>) -> Vec<f32> {
    t.into_data().convert::<f32>().try_into_vec::<f32>().expect("f32")
}

fn step_graph(dir: &Path, w: &DeltaWeights) -> Session {
    let dm = dims();
    let mut pool = WeightPool::default();
    let mut g = Graph::new("delta_step");
    let c = dm.channels();
    let x = g.input("x", DType::F32, &[1, 1, D as i64]);
    let conv = g.input("conv_in", DType::F32, &[1, CONV as i64 - 1, c as i64]);
    let state =
        g.input("state_in", DType::F32, &[1, HEADS as i64, HEAD_DIM as i64, HEAD_DIM as i64]);
    let out = Cx { g: &mut g, pool: &mut pool, form: WeightForm::F32 }
        .delta_step("l", &dm, w, &x, &conv, &state)
        .unwrap();
    g.output("y", &out.y, DType::F32, &[1, 1, D as i64]);
    g.output("conv_out", &out.conv, DType::F32, &[1, CONV as i64 - 1, c as i64]);
    g.output(
        "state_out",
        &out.state,
        DType::F32,
        &[1, HEADS as i64, HEAD_DIM as i64, HEAD_DIM as i64],
    );
    session(&write_graph(dir, "delta_step", &g, &pool))
}

/// The DeltaNet step: the graph's output equals the Burn layer's, position by position, with
/// the convolution window and the state fed back from the graph's own outputs.
fn step_parity(positions: usize, check_every: usize) -> f32 {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let (net, w, device) = layer(dir.path());
    let mut session = step_graph(dir.path(), &w);
    let dm = dims();
    let mut cache = net.make_cache(1, &device);
    let mut conv = vec![0f32; (CONV - 1) * dm.channels()];
    let mut state = vec![0f32; HEADS * HEAD_DIM * HEAD_DIM];
    let mut seed = 0xC0FFEEu64;
    let mut worst = 0f32;
    for p in 0..positions {
        let x = pseudo(&mut seed, D);
        let got = run(
            &mut session,
            &[
                ("x", &[1, 1, D], &x),
                ("conv_in", &[1, CONV - 1, dm.channels()], &conv),
                ("state_in", &[1, HEADS, HEAD_DIM, HEAD_DIM], &state),
            ],
        );
        conv = got["conv_out"].clone();
        state = got["state_out"].clone();
        let want = net
            .forward(Tensor::from_data(TensorData::new(x, [1, 1, D]), &device), Some(&mut cache));
        if p % check_every == 0 || p + 1 == positions {
            worst = worst.max(max_diff(&got["y"], &host(want)));
        }
    }
    worst
}

#[test]
fn the_delta_step_matches_the_burn_layer_over_64_positions() {
    let worst = step_parity(64, 1);
    assert!(worst <= 1e-4, "max |dy| = {worst}");
}

/// The state's drift compounds over positions: after 8,192 recurrent steps the output still
/// matches (the state is read through the layer's output; its own max |delta| is bounded by it).
#[test]
fn the_recurrent_state_does_not_drift_over_8192_positions() {
    let worst = step_parity(8192, 256);
    assert!(worst <= 1e-3, "max |dy| after 8192 positions = {worst}");
}

/// The chunked prefill graph (with the f32 inverse in the graph) equals the Burn layer's
/// chunked forward over a bucket, and leaves the same state: one more step afterwards matches.
#[test]
fn the_chunked_prefill_matches_the_burn_layer_and_leaves_its_state() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let (net, w, device) = layer(dir.path());
    let dm = dims();
    let t = 16;
    let c = dm.channels();
    let (h, hd) = (HEADS as i64, HEAD_DIM as i64);
    let mut pool = WeightPool::default();
    let mut g = Graph::new("delta_chunk");
    let x = g.input("x", DType::F32, &[1, t as i64, D as i64]);
    let conv = g.input("conv_in", DType::F32, &[1, CONV as i64 - 1, c as i64]);
    let state = g.input("state_in", DType::F32, &[1, h, hd, hd]);
    let out = Cx { g: &mut g, pool: &mut pool, form: WeightForm::F32 }
        .delta_chunked("l", &dm, &w, &x, &conv, &state, t)
        .unwrap();
    g.output("y", &out.y, DType::F32, &[1, t as i64, D as i64]);
    g.output("conv_out", &out.conv, DType::F32, &[1, CONV as i64 - 1, c as i64]);
    g.output("state_out", &out.state, DType::F32, &[1, h, hd, hd]);
    let mut chunked = session(&write_graph(dir.path(), "delta_chunk", &g, &pool));
    let mut stepped = step_graph(dir.path(), &w);

    let mut seed = 0xBEEFu64;
    let prompt = pseudo(&mut seed, t * D);
    let conv0 = vec![0f32; (CONV - 1) * c];
    let state0 = vec![0f32; HEADS * HEAD_DIM * HEAD_DIM];
    let got = run(
        &mut chunked,
        &[
            ("x", &[1, t, D], &prompt),
            ("conv_in", &[1, CONV - 1, c], &conv0),
            ("state_in", &[1, HEADS, HEAD_DIM, HEAD_DIM], &state0),
        ],
    );
    let mut cache = net.make_cache(1, &device);
    let want = net
        .forward(Tensor::from_data(TensorData::new(prompt, [1, t, D]), &device), Some(&mut cache));
    let d = max_diff(&got["y"], &host(want));
    assert!(d <= 1e-4, "chunked prefill max |dy| = {d}");
    // The state the chunked graph leaves continues like the Burn cache does.
    let x = pseudo(&mut seed, D);
    let next = run(
        &mut stepped,
        &[
            ("x", &[1, 1, D], &x),
            ("conv_in", &[1, CONV - 1, c], &got["conv_out"]),
            ("state_in", &[1, HEADS, HEAD_DIM, HEAD_DIM], &got["state_out"]),
        ],
    );
    let want =
        net.forward(Tensor::from_data(TensorData::new(x, [1, 1, D]), &device), Some(&mut cache));
    let d = max_diff(&next["y"], &host(want));
    assert!(d <= 1e-4, "the step after the prefill: max |dy| = {d}");
}

/// The f32 triangular inverse of a 64 x 64 unit lower-triangular system, against a direct
/// forward substitution on the host, per chunk (plan: max |delta| <= 1e-4).
#[test]
fn the_f32_chunk_inverse_matches_forward_substitution() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let c = 64usize;
    let mut pool = WeightPool::default();
    let mut g = Graph::new("delta_inverse");
    let neg_a = g.input("neg_a", DType::F32, &[1, 1, 1, c as i64, c as i64]);
    let inv = Cx { g: &mut g, pool: &mut pool, form: WeightForm::F32 }.delta_inverse(&neg_a, c);
    g.output("inv", &inv, DType::F32, &[1, 1, 1, c as i64, c as i64]);
    let mut s = session(&write_graph(dir.path(), "delta_inverse", &g, &pool));
    // A strictly lower-triangular A with small entries: N = -A.
    let mut seed = 0x1234u64;
    let noise = pseudo(&mut seed, c * c);
    let mut a = vec![0f32; c * c];
    for i in 0..c {
        for j in 0..i {
            a[i * c + j] = noise[i * c + j] * 0.05;
        }
    }
    let n: Vec<f32> = a.iter().map(|v| -v).collect();
    let got = run(&mut s, &[("neg_a", &[1, 1, 1, c, c], &n)]);
    // (I + A)^-1 by forward substitution, column by column, in f64.
    let mut want = vec![0f64; c * c];
    for col in 0..c {
        for i in 0..c {
            let mut v = if i == col { 1.0 } else { 0.0 };
            for j in 0..i {
                v -= a[i * c + j] as f64 * want[j * c + col];
            }
            want[i * c + col] = v;
        }
    }
    let worst =
        got["inv"].iter().zip(&want).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max);
    assert!(worst <= 1e-4, "inverse max |delta| = {worst}");
}
