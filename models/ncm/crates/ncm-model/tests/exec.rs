//! The Burn adapter of the `ncm-exec` interface against the Burn reference, on tiny random
//! models (CPU): a session's steps give the model's own logits, restores return to the point
//! they were taken, and the hybrid's phases (`begin_block`, `pass`, `fill`, `end_block`) decode
//! the same blocks as `FastSession::decode_block`.

use burn::tensor::{Device, Int, Tensor, TensorData};
use ncm_exec::{
    Dtype, ExecBackend, ExecSession, FootprintRequest, ModelKind, PendingOutput, PickSpec, Quant,
    SnapshotInfo, StepOut, Tier,
};
use ncm_model::exec::{BurnBackend, BurnInfo, BurnModel};
use ncm_model::hybrid::{DecodeOptions, FastSession, Hooks, HybridLm};
use ncm_model::{
    Config, NcmConfig, NcmLm, TokConfig, TokenLm, tiny_hybrid, tiny_ncm_v2, tiny_tok_v2, tiny_v2,
};

const VOCAB: usize = 272;

/// The tests share the CPU device's random stream (weight init), so they run one at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn device() -> Device {
    let device = Device::flex();
    device.seed(7);
    device
}

fn info(config: Config, params: usize) -> BurnInfo {
    BurnInfo {
        config,
        dtype: Dtype::F32,
        quant: Quant::None,
        params,
        device_name: "cpu".into(),
        speculate: false,
    }
}

fn tok_backend() -> BurnBackend {
    let device = device();
    let cfg = TokConfig { vocab_size: VOCAB, ..tiny_tok_v2(tiny_v2(4, 2, 8)) };
    let model = TokenLm::new(&cfg, &device);
    let params = model.parts().total();
    BurnBackend::new(BurnModel::Tok(model), device, info(Config::Tok(cfg), params))
}

fn ncm_backend() -> BurnBackend {
    let device = device();
    let cfg = NcmConfig { vocab_size: VOCAB, ..tiny_ncm_v2(8, tiny_v2(2, 2, 4)) };
    let model = NcmLm::new(&cfg, &device);
    let params = model.parts().total();
    BurnBackend::new(BurnModel::Ncm(model), device, info(Config::Ncm(cfg), params))
}

fn hybrid_backend() -> BurnBackend {
    let device = device();
    let cfg = NcmConfig { vocab_size: VOCAB, ..tiny_hybrid(tiny_v2(2, 2, 4)) };
    let model = HybridLm::new(&cfg, &device);
    let params = model.parts().inference();
    BurnBackend::new(BurnModel::Hybrid(model), device, info(Config::Ncm(cfg), params))
}

fn ids(device: &Device, tokens: &[i32]) -> Tensor<2, Int> {
    Tensor::from_data(TensorData::new(tokens.to_vec(), [1, tokens.len()]), device)
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best })
}

fn prompt(n: usize) -> Vec<i32> {
    (0..n as i32).map(|i| 16 + (i * 7) % 200).collect()
}

fn logits_of(out: StepOut) -> Vec<f32> {
    match out {
        StepOut::Logits(row) => row,
        StepOut::Picked(_) => panic!("a host pick returns logits"),
    }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn capabilities_name_the_model_and_its_unit() {
    let _serial = serial();
    let caps = tok_backend().capabilities();
    assert_eq!((caps.model, caps.unit, caps.block), (ModelKind::Tok, 1, 0));
    assert!(!caps.graphs, "CPU devices run no graphs");
    let caps = ncm_backend().capabilities();
    assert_eq!((caps.model, caps.unit), (ModelKind::Ncm, 4));
    let caps = hybrid_backend().capabilities();
    assert_eq!((caps.model, caps.block), (ModelKind::Hybrid, 8));
}

#[test]
fn footprint_counts_weights_and_caches_in_system_ram() {
    let _serial = serial();
    let backend = tok_backend();
    let one = backend.footprint(&FootprintRequest::new(256, 1, Dtype::F32, Quant::None));
    let two = backend.footprint(&FootprintRequest::new(256, 2, Dtype::F32, Quant::None));
    assert!(one.tier_total(Tier::SystemRam) > 0);
    assert_eq!(one.tier_total(Tier::Device), 0);
    assert_eq!(
        two.part_total(ncm_exec::Part::LatentCache),
        2 * one.part_total(ncm_exec::Part::LatentCache)
    );
    let q8 =
        backend.footprint(&FootprintRequest::new(256, 1, Dtype::F32, Quant::Int8 { group: 0 }));
    assert!(q8.part_total(ncm_exec::Part::Weights) < one.part_total(ncm_exec::Part::Weights));
}

/// A token-baseline session's logits are the model's: after the prompt, and after each fed-back
/// token, against a full forward over the stream.
#[test]
fn token_session_logits_match_the_models_forward() {
    let _serial = serial();
    let backend = tok_backend();
    let BurnModel::Tok(model) = backend.model() else { unreachable!() };
    let mut session = backend.new_session(64).unwrap();
    let mut stream = prompt(10);
    let out = session.prefill(&stream, 4).unwrap();
    assert_eq!(out.output, PendingOutput::Logits);
    let mut row = session.pending_logits().expect("logits after the prompt");
    for _ in 0..6 {
        let full = model.forward(ids(backend.device(), &stream), None);
        let last = stream.len() - 1;
        let want: Vec<f32> =
            full.narrow(1, last, 1).into_data().convert::<f32>().try_into_vec::<f32>().unwrap();
        assert!(max_diff(&row, &want) < 1e-3, "the session's logits are the forward's");
        let token = argmax(&row) as i32;
        stream.push(token);
        row = logits_of(session.decode_step(&[token], &PickSpec::host()).unwrap());
    }
}

#[test]
fn token_snapshot_restores_the_point_it_was_taken() {
    let _serial = serial();
    let backend = tok_backend();
    let mut session = backend.new_session(64).unwrap();
    session.prefill(&prompt(12), 8).unwrap();
    let snapshot = session.snapshot();
    assert!(snapshot.has_output(), "the prompt's logits ride along");
    let first = logits_of(session.decode_step(&[40], &PickSpec::host()).unwrap());
    let _ = session.decode_step(&[41], &PickSpec::host()).unwrap();
    session.restore(&snapshot);
    let again = logits_of(session.decode_step(&[40], &PickSpec::host()).unwrap());
    assert!(max_diff(&first, &again) < 1e-5, "a restore returns to the snapshot's state");
    // A reset empties the stream: the same prompt gives the same logits.
    session.reset();
    session.prefill(&prompt(12), 8).unwrap();
    let fresh = logits_of(session.decode_step(&[40], &PickSpec::host()).unwrap());
    assert!(max_diff(&first, &fresh) < 1e-5);
}

/// The NCM's eager greedy concept is the model's own `decode_concept`, and the next step appends
/// it first.
#[test]
fn ncm_session_concepts_match_the_models_decode() {
    let _serial = serial();
    let backend = ncm_backend();
    let BurnModel::Ncm(model) = backend.model() else { unreachable!() };
    let device = backend.device().clone();
    let stream = prompt(16);
    let mut session = backend.new_session(64).unwrap();
    let out = session.prefill(&stream, 8).unwrap();
    assert_eq!(out.output, PendingOutput::Condition);
    let mut cache = model.make_cache(64, 1, &device);
    let mut cond = model.prefill(ids(&device, &stream), &mut cache, 8);
    let mut feed: Vec<i32> = Vec::new();
    for _ in 0..3 {
        let StepOut::Picked(got) = session.decode_step(&feed, &PickSpec::greedy()).unwrap() else {
            panic!("a greedy step picks")
        };
        let (want, _) = model.decode_concept(cond.clone(), &mut cache, None, false);
        let want: Vec<u32> = want
            .into_data()
            .convert::<i32>()
            .try_into_vec::<i32>()
            .unwrap()
            .into_iter()
            .map(|t| t as u32)
            .collect();
        assert_eq!(got, want);
        feed = got.iter().map(|&t| t as i32).collect();
        cond = model.append_tokens(ids(&device, &feed), &mut cache);
    }
}

/// Host-chosen concepts: forced tokens come first and `choose` sees the rest.
#[test]
fn ncm_host_concepts_force_the_pending_tokens() {
    let _serial = serial();
    let backend = ncm_backend();
    let mut session = backend.new_session(64).unwrap();
    session.prefill(&prompt(8), 8).unwrap();
    let mut seen = Vec::new();
    let mut choose = |j: usize, logits: &mut dyn FnMut() -> Vec<f32>| {
        seen.push(j);
        argmax(&logits()) as i32
    };
    let concept = session.decode_concept_host(&[], &[33, 34], &mut choose).unwrap();
    assert_eq!(&concept[..2], &[33, 34]);
    assert_eq!(seen, vec![2, 3], "only the free positions are chosen");
}

/// The hybrid's phases decode the same blocks as the fast session's own loop, greedy, over
/// several blocks and with a snapshot taken mid-reply.
#[test]
fn hybrid_phases_decode_the_fast_sessions_blocks() {
    let _serial = serial();
    let backend = hybrid_backend();
    let BurnModel::Hybrid(model) = backend.model() else { unreachable!() };
    let device = backend.device().clone();
    let opts = DecodeOptions { stop: vec![-1], ..Default::default() };
    let stream = prompt(21);

    let mut reference = FastSession::new(model, 128, &device);
    reference.prefill(model, &stream);
    let mut session = backend.new_session(128).unwrap();
    session.begin_request(0);
    session.prefill(&stream, 0).unwrap();
    assert_eq!(session.tokens(), stream.as_slice());

    for _ in 0..3 {
        let want = reference.decode_block(model, 8, &opts, &mut Hooks::default());
        let got = session.decode_block(8, &opts, &mut Hooks::default()).unwrap();
        assert_eq!(got.tokens, want.tokens);
        assert_eq!(got.stopped, want.stopped);
        assert_eq!(got.stats, want.stats);
        assert_eq!(session.tokens(), reference.tokens());
    }

    // Back to a mid-reply snapshot: the same next block.
    let snapshot = session.snapshot();
    assert!(snapshot.has_output());
    let next = session.decode_block(8, &opts, &mut Hooks::default()).unwrap();
    session.restore(&snapshot);
    let again = session.decode_block(8, &opts, &mut Hooks::default()).unwrap();
    assert_eq!(next.tokens, again.tokens);
}

#[test]
fn a_block_can_be_driven_phase_by_phase() {
    let _serial = serial();
    let backend = hybrid_backend();
    let mut session = backend.new_session(64).unwrap();
    session.prefill(&prompt(9), 0).unwrap();
    let opts = DecodeOptions::default();
    let mut run = session.begin_block(8, &opts).unwrap();
    let mut passes = 0;
    let mut fills = 0;
    loop {
        match session.phase(&run) {
            ncm_exec::BlockPhase::Pass => {
                let result = session.pass(&mut run, &opts, &mut Hooks::default()).unwrap();
                assert_eq!(result.ids.len(), 8, "a pass reads back L ids and L draft tokens");
                assert_eq!(result.draft.len(), 8);
                passes += 1;
            }
            ncm_exec::BlockPhase::Fill => {
                session.fill(&mut run, &opts, &mut Hooks::default()).unwrap();
                fills += 1;
            }
            ncm_exec::BlockPhase::Done => break,
        }
    }
    assert!(passes >= 1 || fills >= 1);
    let out = session.end_block(run).unwrap();
    assert_eq!(out.stats.check_passes, passes);
    assert_eq!(out.stats.ar_tokens, fills);
}
