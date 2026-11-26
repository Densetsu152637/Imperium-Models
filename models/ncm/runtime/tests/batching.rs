//! Continuous batching (`--parallel N`): concurrent requests on tiny CPU models give the
//! tokens the sequential engine gives, identical prompts are prefilled once and forked, and the
//! slots report their state.

mod common;

use std::sync::Arc;
use std::thread;

use common::{
    engine as sequential, engine_with_slots, request, tiny_ncm_config, tiny_ncm_v2_config,
    tiny_tok_config, tiny_tok_v2_config, vocab,
};
use ncm_server::engine::{Backend, DecodePath, Engine, GenStats};
use ncm_server::sampling::SamplingParams;

fn prompt(text: &str) -> Vec<u32> {
    vocab().encode(text).unwrap()
}

fn run(engine: &Engine, text: &str, max: usize, sampling: SamplingParams) -> (Vec<u32>, GenStats) {
    let mut tokens = Vec::new();
    let stats = engine
        .generate(request(prompt(text), max, sampling), &mut |t| {
            tokens.push(t);
            true
        })
        .unwrap();
    (tokens, stats)
}

/// Every request of `jobs` at once from its own thread.
fn concurrently(
    engine: &Arc<Engine>,
    jobs: &[(&'static str, usize, SamplingParams)],
) -> Vec<(Vec<u32>, GenStats)> {
    let handles: Vec<_> = jobs
        .iter()
        .map(|&(text, max, sampling)| {
            let engine = engine.clone();
            thread::spawn(move || run(&engine, text, max, sampling))
        })
        .collect();
    handles.into_iter().map(|h| h.join().expect("a request thread")).collect()
}

const PROMPTS: [&str; 4] = [
    "the quick brown fox",
    "jumps over the lazy dog, twice over",
    "a",
    "lorem ipsum dolor sit amet, consectetur adipiscing elit",
];

#[test]
fn batched_requests_decode_what_sequential_ones_do() {
    for config in [tiny_tok_config(), tiny_ncm_config(), tiny_tok_v2_config(), tiny_ncm_v2_config()]
    {
        let one = sequential(config.clone(), 256);
        let want: Vec<Vec<u32>> =
            PROMPTS.iter().map(|p| run(&one, p, 20, SamplingParams::GREEDY).0).collect();
        let batched = Arc::new(engine_with_slots(config, 256, 3));
        let jobs: Vec<_> = PROMPTS.iter().map(|&p| (p, 20, SamplingParams::GREEDY)).collect();
        let got = concurrently(&batched, &jobs);
        for ((tokens, stats), want) in got.iter().zip(&want) {
            assert_eq!(tokens, want);
            assert_eq!(stats.path, DecodePath::Batched);
            assert_eq!(tokens.len(), stats.completion_tokens);
        }
        // Four requests on three slots: one waited, and some steps batched several rows.
        assert!(got.iter().any(|(_, s)| s.mean_batch > 1.0));
        let report = batched.slots().expect("a batching engine reports its slots");
        assert_eq!(report.slots.len(), 3);
        assert_eq!(report.served, 4);
        assert!(report.slots.iter().all(|s| s.state == "idle"));
        assert_eq!(report.queued, 0);
    }
}

#[test]
fn identical_prompts_are_prefilled_once_and_forked() {
    let sampled =
        |seed| SamplingParams { temperature: 0.7, top_k: 0, top_p: 0.95, seed: Some(seed) };
    for config in [tiny_tok_config(), tiny_ncm_config()] {
        let one = sequential(config.clone(), 256);
        let text = "the same prompt for every sample";
        let want: Vec<Vec<u32>> = (0..3).map(|s| run(&one, text, 16, sampled(s)).0).collect();
        let batched = Arc::new(engine_with_slots(config, 256, 3));
        let jobs: Vec<_> = (0..3).map(|s| (text, 16, sampled(s))).collect();
        let got = concurrently(&batched, &jobs);
        // Each sample is its seed's sequential output.
        for ((tokens, _), want) in got.iter().zip(&want) {
            assert_eq!(tokens, want);
        }
        let forks = got.iter().filter(|(_, s)| s.forked).count();
        let report = batched.slots().expect("slots");
        assert_eq!(report.forks as usize, forks);
        // (Whether a request is forked depends on its arrival: one that arrives while an equal
        // prompt is prefilling is; one that arrives later reuses a kept prefix or prefills.)
        for (_, stats) in got.iter().filter(|(_, s)| s.forked) {
            assert_eq!(stats.cached_tokens, stats.prompt_tokens);
        }
    }
}

#[test]
fn a_slot_reuses_its_kept_prefix() {
    let batched = engine_with_slots(tiny_tok_config(), 256, 2);
    let (first, _) = run(&batched, "a prompt and then", 8, SamplingParams::GREEDY);
    let longer = format!("a prompt and then{}", vocab().decode(&first));
    let (_, stats) = run(&batched, &longer, 4, SamplingParams::GREEDY);
    assert!(stats.cached_tokens > 0, "the second turn starts from the first one's cache");
}
