//! The engine's graphed decode paths on CUDA (`--features cuda`, a GPU; CI builds neither, so
//! this file compiles to nothing there): greedy replies through the graphs equal the eager
//! path's, sampled replies are reproducible by seed and vary with it, both architectures and
//! designs, across requests: the graphs are captured once and kept, as the caches keep their
//! buffers.
#![cfg(feature = "cuda")]

mod common;

use burn::tensor::FloatDType;
use common::{
    request, tiny_ncm_config, tiny_ncm_v2_config, tiny_tok_config, tiny_tok_v2_config, vocab,
};
use ncm_model::fused::set_cuda_graphs;
use ncm_model::{Config, NcmConfig};
use ncm_server::engine::{
    Backend, DecodePath, DeviceChoice, Engine, EngineOptions, GenStats, make_device,
};
use ncm_server::sampling::SamplingParams;

fn engine(config: Config, speculate: bool) -> Engine {
    let options = EngineOptions {
        config,
        ctx: 256,
        device: make_device(DeviceChoice::Cuda, FloatDType::F32).expect("cuda device"),
        device_name: "cuda".into(),
        dtype: FloatDType::F32,
        weights: None,
        seed: 3,
        prefill_chunk: 16,
        quant: None,
        speculate,
        prefix_reuse: true,
        warmup: false,
    };
    Engine::new(options, vocab()).expect("a tiny engine")
}

fn run(engine: &Engine, prompt: &str, sampling: SamplingParams) -> (Vec<u32>, GenStats) {
    let prompt = vocab().encode(prompt).unwrap();
    let mut tokens = Vec::new();
    let stats = engine
        .generate(request(prompt, 40, sampling), &mut |t| {
            tokens.push(t);
            true
        })
        .unwrap();
    (tokens, stats)
}

fn configs() -> [Config; 4] {
    [tiny_tok_config(), tiny_tok_v2_config(), tiny_ncm_config(), tiny_ncm_v2_config()]
}

/// One test so the process-wide graph switch is never flipped under another test.
#[test]
fn graphed_paths_match_eager_and_sample_by_seed() {
    let prompts = ["the quick brown fox jumps over the lazy dog, twice over", "a different prompt"];
    let prompt = prompts[0];
    for config in configs() {
        let engine = engine(config.clone(), false);
        set_cuda_graphs(false);
        let eager: Vec<_> = prompts
            .iter()
            .map(|p| {
                let (tokens, stats) = run(&engine, p, SamplingParams::GREEDY);
                assert_eq!(stats.path, DecodePath::Eager);
                tokens
            })
            .collect();
        set_cuda_graphs(true);
        // The graph is captured once, then kept across requests (each resets or restores the
        // caches in place): later requests replay it without recapturing.
        for round in 0..3 {
            for (p, want) in prompts.iter().zip(&eager) {
                let (graphed, stats) = run(&engine, p, SamplingParams::GREEDY);
                assert_eq!(stats.path, DecodePath::GraphedGreedy);
                let first = round == 0 && *p == prompts[0];
                assert_eq!(stats.capture_s > 0.0, first, "{config:?} round {round}: capture");
                assert_eq!(&graphed, want, "{config:?}: graphed greedy differs from eager");
            }
        }
        let sampled =
            |seed| SamplingParams { temperature: 0.9, top_k: 40, top_p: 0.95, seed: Some(seed) };
        let (a, stats) = run(&engine, prompt, sampled(7));
        assert_eq!(stats.path, DecodePath::GraphedSampled);
        assert_eq!(run(&engine, prompt, sampled(7)).0, a, "a seed reproduces its reply");
        assert_ne!(run(&engine, prompt, sampled(8)).0, a, "another seed draws another reply");
    }
    // Requests from different threads: the engine runs them all on its own thread, so its kept
    // graphs replay on their capture stream; replies equal the eager ones.
    for config in [tiny_tok_v2_config(), tiny_ncm_v2_config()] {
        let engine = std::sync::Arc::new(engine(config.clone(), false));
        set_cuda_graphs(false);
        let eager: Vec<_> =
            prompts.iter().map(|p| run(&engine, p, SamplingParams::GREEDY).0).collect();
        set_cuda_graphs(true);
        for round in 0..2 {
            let handles: Vec<_> = (0..2)
                .map(|t| {
                    let engine = engine.clone();
                    std::thread::spawn(move || {
                        let p = prompts[(t + round) % 2];
                        (p, run(&engine, p, SamplingParams::GREEDY).0)
                    })
                })
                .collect();
            for handle in handles {
                let (p, got) = handle.join().expect("a request thread");
                let want = &eager[prompts.iter().position(|q| *q == p).unwrap()];
                assert_eq!(&got, want, "{config:?}: a request from another thread differs");
            }
        }
    }
    // Self-speculation (a model with a draft head) gives plain greedy's reply in f32.
    let Config::Ncm(base) = tiny_ncm_v2_config() else { unreachable!() };
    let with_draft = Config::Ncm(NcmConfig { draft_layers: 1, ..base });
    let plain = run(&engine(with_draft.clone(), false), prompt, SamplingParams::GREEDY).0;
    let speculative = engine(with_draft, true);
    for p in [prompt, prompts[1], prompt] {
        let (spec, stats) = run(&speculative, p, SamplingParams::GREEDY);
        assert_eq!(stats.path, DecodePath::Speculative);
        if p == prompt {
            assert_eq!(spec, plain, "speculative greedy differs from plain greedy");
        }
    }
}
