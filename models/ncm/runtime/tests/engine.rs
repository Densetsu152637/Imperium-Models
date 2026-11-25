//! The generation loop on tiny random models (CPU): both architectures, sampling, limits,
//! context handling and constrained tool calls.

mod common;

use std::sync::Arc;

use common::{
    engine as build, request, tiny_ncm_config, tiny_ncm_v2_config, tiny_tok_config,
    tiny_tok_v2_config, vocab,
};
use ncm_server::constrain::ToolCallGuard;
use ncm_server::engine::{Backend, Engine, EngineError, Finish, GenRequest};
use ncm_server::sampling::SamplingParams;

fn prompt(text: &str) -> Vec<u32> {
    vocab().encode(text).unwrap()
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
fn both_models_decode_and_stop_at_the_token_limit() {
    for config in [tiny_tok_config(), tiny_ncm_config()] {
        let engine = build(config, 256);
        let (tokens, stats) =
            run(&engine, request(prompt("hello world"), 10, SamplingParams::GREEDY));
        assert_eq!(tokens.len(), stats.completion_tokens);
        assert!(stats.completion_tokens <= 10 && stats.completion_tokens > 0);
        assert_eq!(stats.prompt_tokens, 11);
        assert!(stats.prompt_s > 0.0);
        if stats.finish == Finish::Length {
            assert_eq!(stats.completion_tokens, 10);
        }
        assert!(tokens.iter().all(|&t| (t as usize) < 272));
    }
}

#[test]
fn greedy_decoding_is_deterministic_and_repeatable_across_requests() {
    for config in [tiny_tok_config(), tiny_ncm_config()] {
        let engine = build(config, 256);
        let first = run(&engine, request(prompt("abcdefgh"), 24, SamplingParams::GREEDY)).0;
        let second = run(&engine, request(prompt("abcdefgh"), 24, SamplingParams::GREEDY)).0;
        assert_eq!(first, second, "the caches are reset between requests");
        // Another prompt in between does not leak into the next run.
        run(&engine, request(prompt("something else entirely"), 5, SamplingParams::GREEDY));
        assert_eq!(first, run(&engine, request(prompt("abcdefgh"), 24, SamplingParams::GREEDY)).0);
    }
}

#[test]
fn v2_models_decode_and_reset_between_requests() {
    // Prompts long enough that retrieval selects blocks and DeltaNet spans several chunks.
    for config in [tiny_tok_v2_config(), tiny_ncm_v2_config()] {
        let engine = build(config, 256);
        let text = "the quick brown fox jumps over the lazy dog, twice: the quick brown fox";
        let (first, stats) = run(&engine, request(prompt(text), 12, SamplingParams::GREEDY));
        assert!(stats.completion_tokens > 0);
        let (second, _) = run(&engine, request(prompt(text), 12, SamplingParams::GREEDY));
        assert_eq!(first, second, "a request must not see the previous one's state");
    }
}

#[test]
fn a_seed_makes_sampled_decoding_repeatable() {
    let params = SamplingParams { temperature: 1.0, top_k: 0, top_p: 1.0, seed: Some(11) };
    for config in [tiny_tok_config(), tiny_ncm_config()] {
        let engine = build(config, 256);
        let a = run(&engine, request(prompt("seed"), 20, params)).0;
        let b = run(&engine, request(prompt("seed"), 20, params)).0;
        let other =
            run(&engine, request(prompt("seed"), 20, SamplingParams { seed: Some(12), ..params }))
                .0;
        assert_eq!(a, b);
        assert_ne!(a, other);
    }
}

#[test]
fn the_ncm_pads_prompts_that_are_not_whole_concepts() {
    let engine = build(tiny_ncm_config(), 256);
    // 1, 2, 3 and 4 tokens: all valid even though k = 4.
    for n in 1..=5 {
        let text: String = "abcde".chars().take(n).collect();
        let (_, stats) = run(&engine, request(prompt(&text), 4, SamplingParams::GREEDY));
        assert_eq!(stats.prompt_tokens, n);
    }
}

#[test]
fn the_context_is_enforced() {
    let tok = build(tiny_tok_config(), 32);
    assert!(matches!(
        tok.generate(request(vec![20; 40], 4, SamplingParams::GREEDY), &mut |_| true),
        Err(EngineError::ContextExceeded { prompt: 40, ctx: 32 })
    ));
    assert!(matches!(
        tok.generate(request(Vec::new(), 4, SamplingParams::GREEDY), &mut |_| true),
        Err(EngineError::EmptyPrompt)
    ));
    // A prompt that nearly fills the context stops with `length` at the end of it.
    let (tokens, stats) = run(&tok, request(vec![65; 30], usize::MAX, SamplingParams::GREEDY));
    assert!(30 + tokens.len() <= 33, "{} tokens", tokens.len());
    if stats.finish == Finish::Length {
        assert!(30 + tokens.len() >= 32);
    }

    let ncm = build(tiny_ncm_config(), 32);
    assert!(matches!(
        ncm.generate(request(vec![20; 33], 4, SamplingParams::GREEDY), &mut |_| true),
        Err(EngineError::ContextExceeded { .. })
    ));
    let (tokens, _) = run(&ncm, request(vec![65; 30], usize::MAX, SamplingParams::GREEDY));
    assert!(tokens.len() <= 8, "{} tokens", tokens.len());
}

#[test]
fn contexts_too_small_for_the_model_are_refused() {
    let options = |ctx| ncm_server::engine::EngineOptions {
        config: tiny_ncm_config(),
        ctx,
        device: ncm_server::engine::make_device(
            ncm_server::engine::DeviceChoice::Cpu,
            burn::tensor::FloatDType::F32,
        )
        .unwrap(),
        device_name: "cpu".into(),
        dtype: burn::tensor::FloatDType::F32,
        weights: None,
        seed: 0,
        prefill_chunk: 16,
    };
    assert!(matches!(
        Engine::new(options(4), vocab()),
        Err(EngineError::ContextTooSmall { ctx: 4, min: 8 })
    ));
}

#[test]
fn a_listener_that_leaves_stops_generation() {
    for config in [tiny_tok_config(), tiny_ncm_config()] {
        let engine = build(config, 256);
        let mut seen = 0;
        let stats = engine
            .generate(request(prompt("hi"), 100, SamplingParams::GREEDY), &mut |_| {
                seen += 1;
                seen < 3
            })
            .unwrap();
        assert_eq!(seen, 3);
        assert_eq!(stats.finish, Finish::Stop);
    }
}

#[test]
fn model_info_reports_the_launch_context_and_cache() {
    let engine = build(tiny_ncm_config(), 130);
    let info = engine.info();
    assert_eq!(info.arch, "ncm");
    assert_eq!(info.ctx, 128, "rounded down to whole concepts");
    assert_eq!(info.unit, 4);
    assert!(info.cache_bytes > 0);
    let big = build(tiny_tok_config(), 1024);
    assert!(big.info().cache_bytes > build(tiny_tok_config(), 64).info().cache_bytes);
}

/// With a guard that starts inside a `<tool_call>` block, whatever the random model writes is a
/// prefix of a JSON object, and a closed block parses.
#[test]
fn constrained_decoding_writes_valid_json_inside_a_tool_call() {
    let vocab = vocab();
    for config in [tiny_tok_config(), tiny_ncm_config()] {
        let engine = build(config, 512);
        let mut closed = 0;
        for seed in 0..12 {
            let params =
                SamplingParams { temperature: 1.5, top_k: 0, top_p: 1.0, seed: Some(seed) };
            let guard = ToolCallGuard::inside_call(Arc::clone(&vocab));
            let mut request = request(prompt("call"), 300, params);
            request.constraint = Some(Box::new(guard));
            let (tokens, _) = run(&engine, request);
            let text: Vec<u8> = tokens
                .iter()
                .take_while(|&&t| t != vocab.ids.tool_call_end)
                .flat_map(|&t| vocab.bytes(t).to_vec())
                .collect();
            let text = String::from_utf8_lossy(&text).into_owned();
            if tokens.contains(&vocab.ids.tool_call_end) {
                closed += 1;
                let value: serde_json::Value = serde_json::from_str(text.trim())
                    .unwrap_or_else(|e| panic!("seed {seed}: {text:?}: {e}"));
                assert!(value.is_object(), "{text:?}");
            } else {
                // Still open: must be a prefix the JSON state machine accepts.
                let mut state = ncm_server::constrain::JsonState::new();
                assert!(text.bytes().all(|b| state.step(b)), "seed {seed}: {text:?}");
            }
            assert!(
                !tokens
                    .iter()
                    .take_while(|&&t| t != vocab.ids.tool_call_end)
                    .any(|&t| vocab.is_special(t)),
                "no special token inside a call: {tokens:?}"
            );
        }
        // A random model closes some of them (an object can be as short as `{}`).
        assert!(closed > 0, "no tool call closed");
    }
}

/// Outside a tool call the guard changes nothing; the first `<tool_call>` the model writes
/// switches it on.
#[test]
fn the_guard_is_free_until_a_tool_call_opens() {
    use ncm_server::constrain::Constraint;
    let vocab = vocab();
    let mut guard = ToolCallGuard::new(Arc::clone(&vocab));
    let word = vocab.encode("hello").unwrap();
    assert!(word.iter().all(|&t| guard.allows(t)));
    assert!(guard.allows(vocab.ids.im_end));
    guard.advance(vocab.ids.tool_call);
    assert!(guard.in_call());
    assert!(!guard.allows(vocab.ids.im_end), "no escape from a call");
    assert!(!guard.allows(vocab.ids.tool_call_end), "an empty call is not closed");
    let brace = vocab.encode("{").unwrap()[0];
    assert!(guard.allows(brace));
    assert!(!guard.allows(vocab.encode("[").unwrap()[0]));
    for t in vocab.encode("{\"a\":1}").unwrap() {
        assert!(guard.allows(t));
        guard.advance(t);
    }
    assert!(guard.allows(vocab.ids.tool_call_end), "the object is complete");
    assert!(!guard.allows(brace), "nothing but whitespace after the object");
    guard.advance(vocab.ids.tool_call_end);
    assert!(!guard.in_call());
}
