//! The NCM hybrid in the server's engine (CPU, tiny config): block decoding streams tokens,
//! stops at the limit, repeats across requests (the session is reset, its buffers kept), and
//! reports what its blocks did; constrained requests stay inside the grammar; several slots
//! decode together at block granularity.

mod common;

use common::{engine, engine_with_slots, request, vocab};
use ncm_model::{Config, NcmConfig, tiny_hybrid, tiny_v2};
use ncm_server::api::timings_json;
use ncm_server::constrain::ToolCallGuard;
use ncm_server::engine::{Backend, DecodePath, Engine, Finish, GenRequest};
use ncm_server::sampling::SamplingParams;
use ncm_server::testing::BYTE_TOKENIZER_SIZE;

fn hybrid_config() -> Config {
    Config::Ncm(NcmConfig { vocab_size: BYTE_TOKENIZER_SIZE, ..tiny_hybrid(tiny_v2(2, 2, 4)) })
}

fn run(engine: &Engine, request: GenRequest) -> (Vec<u32>, ncm_server::engine::GenStats) {
    let mut tokens = Vec::new();
    let stats = engine
        .generate(request, &mut |t| {
            tokens.push(t);
            true
        })
        .unwrap();
    (tokens, stats)
}

#[test]
fn the_hybrid_decodes_blocks_and_repeats_across_requests() {
    let engine = engine(hybrid_config(), 256);
    assert_eq!(engine.info().arch, "hybrid");
    assert_eq!(engine.info().unit, 8);
    let prompt = vocab().encode("the quick brown fox jumps").unwrap();
    let (first, stats) = run(&engine, request(prompt.clone(), 20, SamplingParams::GREEDY));
    assert_eq!(first.len(), stats.completion_tokens);
    assert!(stats.completion_tokens > 0 && stats.completion_tokens <= 20);
    if stats.finish == Finish::Length {
        assert_eq!(stats.completion_tokens, 20);
    }
    let DecodePath::Hybrid(acceptance) = stats.path else { panic!("the hybrid path") };
    assert!(acceptance.blocks >= 1);
    let timings = timings_json(&stats, engine.info().cache_bytes);
    assert_eq!(timings["decode_path"], "hybrid");
    assert!(timings["hybrid"]["mean_first_accepted"].as_f64().is_some());
    // Another request in between does not leak into the next.
    let other = vocab().encode("something else").unwrap();
    run(&engine, request(other, 5, SamplingParams::GREEDY));
    let (again, _) = run(&engine, request(prompt, 20, SamplingParams::GREEDY));
    assert_eq!(again, first);
}

#[test]
fn constrained_hybrid_requests_stay_in_the_grammar() {
    let engine = engine(hybrid_config(), 256);
    let prompt = vocab().encode("call a tool").unwrap();
    let mut req = request(prompt, 40, SamplingParams::GREEDY);
    req.constraint = Some(Box::new(ToolCallGuard::new(vocab())));
    let (tokens, _) = run(&engine, req);
    // Whatever the random weights propose, the guard's rules hold: every token was allowed.
    let mut guard = ToolCallGuard::new(vocab());
    for &t in &tokens {
        assert!(ncm_server::constrain::Constraint::allows(&guard, t), "token {t}");
        ncm_server::constrain::Constraint::advance(&mut guard, t);
    }
}

#[test]
fn the_hybrid_batches_slots_at_block_granularity() {
    use std::sync::Arc;
    let prompts = ["the quick brown fox", "jumps over the lazy dog, twice over", "x"];
    let one = engine(hybrid_config(), 256);
    let want: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| run(&one, request(vocab().encode(p).unwrap(), 20, SamplingParams::GREEDY)).0)
        .collect();
    let batched = Arc::new(engine_with_slots(hybrid_config(), 256, 3));
    let handles: Vec<_> = prompts
        .iter()
        .map(|&p| {
            let engine = batched.clone();
            std::thread::spawn(move || {
                run(&engine, request(vocab().encode(p).unwrap(), 20, SamplingParams::GREEDY))
            })
        })
        .collect();
    for (h, want) in handles.into_iter().zip(&want) {
        let (tokens, stats) = h.join().expect("a request");
        assert_eq!(&tokens, want);
        assert!(matches!(stats.path, DecodePath::Hybrid(_)));
    }
    let report = batched.slots().expect("a batching engine");
    assert_eq!(report.slots.len(), 3);
    assert_eq!(report.served, 3);
    // Identical prompts arriving together prefill once (when one waits for the other).
    let same = "the same prompt for every sample";
    let sampled =
        |seed| SamplingParams { temperature: 0.7, top_k: 0, top_p: 0.95, seed: Some(seed) };
    let handles: Vec<_> = (0..3)
        .map(|s| {
            let engine = batched.clone();
            std::thread::spawn(move || {
                run(&engine, request(vocab().encode(same).unwrap(), 16, sampled(s)))
            })
        })
        .collect();
    for h in handles {
        let (tokens, stats) = h.join().expect("a request");
        assert_eq!(tokens.len(), stats.completion_tokens);
        if stats.forked {
            assert_eq!(stats.cached_tokens, stats.prompt_tokens);
        }
    }
}

/// Prefix reuse: a next turn that repeats the prompt and the reply starts from the kept end of
/// the reply (inside an open block), and repeating it gives the same reply.
#[test]
fn the_hybrid_reuses_kept_prefixes() {
    let engine = engine(hybrid_config(), 256);
    let prompt = vocab().encode("a prompt that ends mid-block").unwrap();
    let (reply, first) = run(&engine, request(prompt.clone(), 11, SamplingParams::GREEDY));
    assert_eq!(first.cached_tokens, 0);
    let mut turn = prompt.clone();
    turn.extend_from_slice(&reply);
    turn.extend(vocab().encode(" and more").unwrap());
    let (a, stats) = run(&engine, request(turn.clone(), 9, SamplingParams::GREEDY));
    assert_eq!(stats.cached_tokens, prompt.len() + reply.len());
    // The same prompt again: a whole hit on the kept prompt point.
    let (b, again) = run(&engine, request(turn.clone(), 9, SamplingParams::GREEDY));
    assert_eq!(again.cached_tokens, turn.len());
    assert_eq!(a, b);
}
