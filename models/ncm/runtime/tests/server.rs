//! The HTTP layer against a scripted model: request and response JSON shapes, streaming
//! chunks, errors, and the readiness behaviour Imperium relies on.

mod common;

use std::sync::Arc;

use common::{Scripted, TestServer, sse_payloads};
use ncm_server::sampling::SamplingParams;
use ncm_server::server::AppState;
use serde_json::{Value, json};

const TOOL_REPLY: &str = "<think>\nneed the time\n</think>\nChecking.\n\
    <tool_call>\n{\"name\":\"get_time\",\"arguments\":{\"tz\":\"UTC\"}}\n</tool_call>\n<|im_end|>";

fn chat_body(extra: Value) -> Value {
    let mut body = json!({
        "model": "anything",
        "messages": [{"role": "user", "content": "What time is it?"}],
    });
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body
}

fn tools() -> Value {
    json!([{"type": "function", "function": {
        "name": "get_time", "description": "Current time.",
        "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}},
    }}])
}

async fn post(server: &TestServer, path: &str, body: &Value) -> reqwest::Response {
    reqwest::Client::new().post(server.url(path)).json(body).send().await.unwrap()
}

#[tokio::test]
async fn health_is_503_while_loading_and_200_when_ready() {
    let loading = Arc::new(AppState::loading("m", "", 1, SamplingParams::GREEDY));
    let server = TestServer::start(loading.clone()).await;
    let response = reqwest::get(server.url("/health")).await.unwrap();
    assert_eq!(response.status(), 503);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["code"], 503);
    assert_eq!(reqwest::get(server.url("/v1/models")).await.unwrap().status(), 503);
    let chat = post(&server, "/v1/chat/completions", &chat_body(json!({}))).await;
    assert_eq!(chat.status(), 503);

    loading.set_backend(Arc::new(Scripted::new("Hi.<|im_end|>", 4096)));
    let response = reqwest::get(server.url("/health")).await.unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.json::<Value>().await.unwrap(), json!({"status": "ok"}));
}

#[tokio::test]
async fn a_failed_load_is_reported() {
    let state = Arc::new(AppState::loading("m", "", 1, SamplingParams::GREEDY));
    state.set_failure("no such file".into());
    let server = TestServer::start(state).await;
    let response = reqwest::get(server.url("/health")).await.unwrap();
    assert_eq!(response.status(), 503);
    let body: Value = response.json().await.unwrap();
    assert!(body["error"]["message"].as_str().unwrap().contains("no such file"));
}

#[tokio::test]
async fn models_and_props_describe_the_model() {
    let (server, _) = TestServer::scripted("Hi.<|im_end|>", 2048).await;
    let models: Value = reqwest::get(server.url("/v1/models")).await.unwrap().json().await.unwrap();
    assert_eq!(models["object"], "list");
    assert_eq!(models["data"][0]["id"], "scripted");
    assert_eq!(models["data"][0]["object"], "model");

    let props: Value = reqwest::get(server.url("/props")).await.unwrap().json().await.unwrap();
    // Imperium reads the per-request context from `default_generation_settings.n_ctx`.
    assert_eq!(props["default_generation_settings"]["n_ctx"], 2048);
    assert_eq!(props["total_slots"], 1);
    assert_eq!(props["ncm"]["cache_bytes"], 4096);
}

#[tokio::test]
async fn tokenize_detokenize_and_apply_template() {
    let (server, _) = TestServer::scripted("Hi.<|im_end|>", 2048).await;
    // Special tokens in the text count as one token with `parse_special`.
    let tokens: Value = post(
        &server,
        "/tokenize",
        &json!({"content": "<|im_start|>user\nhi<|im_end|>", "add_special": false, "parse_special": true}),
    )
    .await
    .json()
    .await
    .unwrap();
    let tokens = tokens["tokens"].as_array().unwrap();
    assert_eq!(tokens.len(), 1 + 5 + 2 + 1, "im_start, `user\\n`, `hi`, im_end");
    let plain: Value =
        post(&server, "/tokenize", &json!({"content": "<|im_end|>", "parse_special": false}))
            .await
            .json()
            .await
            .unwrap();
    assert!(plain["tokens"].as_array().unwrap().len() > 1, "written out as text");

    let text: Value =
        post(&server, "/detokenize", &json!({"tokens": tokens})).await.json().await.unwrap();
    assert_eq!(text["content"], "<|im_start|>user\nhi<|im_end|>");

    let rendered: Value = post(
        &server,
        "/apply-template",
        &json!({"messages": [{"role": "user", "content": "hi"}], "tools": tools()}),
    )
    .await
    .json()
    .await
    .unwrap();
    let prompt = rendered["prompt"].as_str().unwrap();
    assert!(prompt.starts_with("<|im_start|>system\n<tools>\n{\"name\":\"get_time\""));
    assert!(prompt.ends_with("<|im_start|>user\nhi\n<|im_end|>\n<|im_start|>assistant\n"));
}

#[tokio::test]
async fn a_chat_completion_has_the_openai_shape() {
    let (server, backend) = TestServer::scripted("Hello there.\n<|im_end|>", 4096).await;
    let response = post(&server, "/v1/chat/completions", &chat_body(json!({}))).await;
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(body["model"], "scripted");
    let choice = &body["choices"][0];
    assert_eq!(choice["index"], 0);
    assert_eq!(choice["finish_reason"], "stop");
    assert_eq!(choice["message"]["role"], "assistant");
    assert_eq!(choice["message"]["content"], "Hello there.");
    assert!(choice["message"].get("tool_calls").is_none());
    assert!(choice["message"].get("reasoning_content").is_none());
    let usage = &body["usage"];
    assert_eq!(usage["completion_tokens"], 14);
    assert_eq!(
        usage["total_tokens"].as_u64().unwrap(),
        usage["prompt_tokens"].as_u64().unwrap() + 14
    );
    assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 0);
    // Prompt processing and decoding are timed separately, in llama-server's field names.
    let timings = &body["timings"];
    assert_eq!(timings["prompt_n"], usage["prompt_tokens"]);
    assert!(timings["prompt_ms"].as_f64().unwrap() > 0.0);
    assert!(timings["predicted_per_second"].as_f64().unwrap() > 0.0);
    assert_eq!(timings["predicted_n"], 14);
    assert_eq!(timings["cache_bytes"], 4096);

    // The model saw the rendered chat prompt.
    let vocab = common::vocab();
    let prompt = backend.prompts.lock().unwrap()[0].clone();
    assert_eq!(
        vocab.decode(&prompt),
        "<|im_start|>user\nWhat time is it?\n<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[tokio::test]
async fn reasoning_and_tool_calls_come_back_in_their_own_fields() {
    let (server, backend) = TestServer::scripted(TOOL_REPLY, 4096).await;
    let body: Value = post(&server, "/v1/chat/completions", &chat_body(json!({"tools": tools()})))
        .await
        .json()
        .await
        .unwrap();
    let choice = &body["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    let message = &choice["message"];
    assert_eq!(message["reasoning_content"], "need the time");
    assert_eq!(message["content"], "Checking.");
    let call = &message["tool_calls"][0];
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "get_time");
    assert_eq!(call["function"]["arguments"], "{\"tz\":\"UTC\"}");
    assert!(call["id"].as_str().unwrap().starts_with("call_"));
    // The tools were rendered into the system turn.
    let prompt = common::vocab().decode(&backend.prompts.lock().unwrap()[0]);
    assert!(prompt.contains("<tools>\n{\"name\":\"get_time\",\"description\":\"Current time.\""));
}

#[tokio::test]
async fn tool_choice_none_hides_the_tools() {
    let (server, backend) = TestServer::scripted("Just text.<|im_end|>", 4096).await;
    post(
        &server,
        "/v1/chat/completions",
        &chat_body(json!({"tools": tools(), "tool_choice": "none"})),
    )
    .await;
    let prompt = common::vocab().decode(&backend.prompts.lock().unwrap()[0]);
    assert!(!prompt.contains("<tools>"));
}

#[tokio::test]
async fn tool_results_and_earlier_calls_round_trip_into_the_prompt() {
    let (server, backend) = TestServer::scripted("It is noon.<|im_end|>", 4096).await;
    let body = json!({
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Time?"},
            {"role": "assistant", "content": null, "reasoning_content": "check", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "get_time", "arguments": "{\"tz\": \"UTC\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "12:00"},
        ],
    });
    let response = post(&server, "/v1/chat/completions", &body).await;
    assert_eq!(response.status(), 200);
    let prompt = common::vocab().decode(&backend.prompts.lock().unwrap()[0]);
    assert_eq!(
        prompt,
        "<|im_start|>system\nBe brief.\n<|im_end|>\n\
         <|im_start|>user\nTime?\n<|im_end|>\n\
         <|im_start|>assistant\n<think>\ncheck\n</think>\n\
         <tool_call>\n{\"name\":\"get_time\",\"arguments\":{\"tz\":\"UTC\"}}\n</tool_call>\n<|im_end|>\n\
         <|im_start|>tool\n<tool_response>\n12:00\n</tool_response>\n<|im_end|>\n\
         <|im_start|>assistant\n"
    );
}

#[tokio::test]
async fn max_tokens_ends_with_length() {
    let (server, _) = TestServer::scripted("A long answer that goes on.\n<|im_end|>", 4096).await;
    let body: Value = post(&server, "/v1/chat/completions", &chat_body(json!({"max_tokens": 5})))
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["choices"][0]["message"]["content"], "A lon");
    assert_eq!(body["usage"]["completion_tokens"], 5);
}

#[tokio::test]
async fn streaming_sends_deltas_then_finish_then_usage_then_done() {
    let (server, _) = TestServer::scripted(TOOL_REPLY, 4096).await;
    let response = post(
        &server,
        "/v1/chat/completions",
        &chat_body(
            json!({"tools": tools(), "stream": true, "stream_options": {"include_usage": true}}),
        ),
    )
    .await;
    assert_eq!(response.status(), 200);
    assert!(response.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));
    let text = response.text().await.unwrap();
    let payloads = sse_payloads(&text);
    assert_eq!(payloads.last().unwrap(), "[DONE]");
    let chunks: Vec<Value> =
        payloads[..payloads.len() - 1].iter().map(|p| serde_json::from_str(p).unwrap()).collect();

    // First chunk opens the assistant message.
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert!(chunks.iter().all(|c| c["object"] == "chat.completion.chunk"));
    let ids: std::collections::HashSet<&str> =
        chunks.iter().map(|c| c["id"].as_str().unwrap()).collect();
    assert_eq!(ids.len(), 1, "one id per stream");

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut calls = Vec::new();
    let mut finish = None;
    for chunk in &chunks {
        let Some(choice) = chunk["choices"].get(0) else { continue };
        let delta = &choice["delta"];
        content.push_str(delta["content"].as_str().unwrap_or_default());
        reasoning.push_str(delta["reasoning_content"].as_str().unwrap_or_default());
        if let Some(calls_in) = delta["tool_calls"].as_array() {
            calls.extend(calls_in.iter().cloned());
        }
        if !choice["finish_reason"].is_null() {
            finish = Some(choice["finish_reason"].clone());
            assert!(chunk["timings"]["predicted_per_second"].as_f64().unwrap() > 0.0);
        }
    }
    assert_eq!(content, "Checking.");
    assert_eq!(reasoning, "need the time");
    assert_eq!(finish, Some(json!("tool_calls")));
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["index"], 0);
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "get_time");
    assert_eq!(calls[0]["function"]["arguments"], "{\"tz\":\"UTC\"}");
    assert!(calls[0]["id"].as_str().unwrap().starts_with("call_"));

    // The usage chunk has no choices (include_usage) and comes last.
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"], json!([]));
    assert!(last["usage"]["completion_tokens"].as_u64().unwrap() > 0);
    assert_eq!(last["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
}

#[tokio::test]
async fn streaming_without_usage_ends_after_the_finish_chunk() {
    let (server, _) = TestServer::scripted("Hi.\n<|im_end|>", 4096).await;
    let text = post(&server, "/v1/chat/completions", &chat_body(json!({"stream": true})))
        .await
        .text()
        .await
        .unwrap();
    let payloads = sse_payloads(&text);
    let chunks: Vec<Value> =
        payloads[..payloads.len() - 1].iter().map(|p| serde_json::from_str(p).unwrap()).collect();
    assert!(chunks.iter().all(|c| !c["choices"].as_array().unwrap().is_empty()));
    assert_eq!(chunks.last().unwrap()["choices"][0]["finish_reason"], "stop");
    assert_eq!(payloads.last().unwrap(), "[DONE]");
}

#[tokio::test]
async fn bad_requests_get_openai_style_errors() {
    let (server, _) = TestServer::scripted("Hi.<|im_end|>", 4096).await;
    for body in [
        json!({"model": "m"}),
        json!({"messages": []}),
        json!({"messages": [{"role": "wizard", "content": "x"}]}),
        json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}),
        json!({"messages": [{"role": "user", "content": "x"}], "n": 2}),
        json!({"messages": [{"role": "user", "content": "x"}], "max_tokens": "ten"}),
    ] {
        let response = post(&server, "/v1/chat/completions", &body).await;
        assert_eq!(response.status(), 400, "{body}");
        let error: Value = response.json().await.unwrap();
        assert_eq!(error["error"]["type"], "invalid_request_error", "{body}");
        assert_eq!(error["error"]["code"], 400);
        assert!(!error["error"]["message"].as_str().unwrap().is_empty());
    }
    let response = reqwest::Client::new()
        .post(server.url("/v1/chat/completions"))
        .header("content-type", "application/json")
        .body("{not json")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    assert_eq!(response.json::<Value>().await.unwrap()["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn a_prompt_longer_than_the_context_is_refused_like_llama_server() {
    let (server, _) = TestServer::scripted("Hi.<|im_end|>", 16).await;
    let response = post(&server, "/v1/chat/completions", &chat_body(json!({}))).await;
    assert_eq!(response.status(), 400);
    let error: Value = response.json().await.unwrap();
    assert_eq!(error["error"]["type"], "exceed_context_size_error");
    assert_eq!(error["error"]["n_ctx"], 16);
    assert!(error["error"]["n_prompt_tokens"].as_u64().unwrap() > 16);
}

#[tokio::test]
async fn request_sampling_fields_reach_the_model() {
    use ncm_server::api::parse_chat_request;
    let defaults = SamplingParams::default();
    let body = chat_body(
        json!({"temperature": 0.0, "top_p": 0.5, "top_k": 7, "seed": 9, "max_completion_tokens": 33}),
    );
    let request = parse_chat_request(&body, &defaults).unwrap();
    assert_eq!(
        request.sampling,
        SamplingParams { temperature: 0.0, top_k: 7, top_p: 0.5, seed: Some(9) }
    );
    assert_eq!(request.max_tokens, Some(33));
    let plain = parse_chat_request(&chat_body(json!({})), &defaults).unwrap();
    assert_eq!(plain.sampling, defaults);
    assert_eq!(plain.max_tokens, None);
    assert_eq!(
        parse_chat_request(&chat_body(json!({"max_tokens": -1})), &defaults).unwrap().max_tokens,
        None
    );
}

#[tokio::test]
async fn concurrent_requests_all_complete() {
    let (server, _) = TestServer::scripted("Same answer.\n<|im_end|>", 4096).await;
    let client = reqwest::Client::new();
    let url = server.url("/v1/chat/completions");
    let mut joins = Vec::new();
    for _ in 0..6 {
        let (client, url) = (client.clone(), url.clone());
        joins.push(tokio::spawn(async move {
            let body: Value = client
                .post(url)
                .json(&chat_body(json!({})))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            body["choices"][0]["message"]["content"].as_str().unwrap().to_owned()
        }));
    }
    for join in joins {
        assert_eq!(join.await.unwrap(), "Same answer.");
    }
}

/// A backend whose generation panics.
struct Panicking(Scripted);

impl ncm_server::engine::Backend for Panicking {
    fn info(&self) -> &ncm_server::engine::ModelInfo {
        self.0.info()
    }

    fn vocab(&self) -> &Arc<ncm_server::vocab::Vocab> {
        self.0.vocab()
    }

    fn generate(
        &self,
        _request: ncm_server::engine::GenRequest,
        _on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<ncm_server::engine::GenStats, ncm_server::engine::EngineError> {
        panic!("a scripted failure")
    }

    fn check_prompt(&self, tokens: usize) -> Result<(), ncm_server::engine::EngineError> {
        self.0.check_prompt(tokens)
    }
}

#[tokio::test]
async fn a_panic_in_generation_fails_the_request_and_health() {
    let backend = Arc::new(Panicking(Scripted::new("Hi.<|im_end|>", 4096)));
    let state = Arc::new(AppState::ready(backend, "m", SamplingParams::GREEDY));
    let server = TestServer::start(state).await;
    assert_eq!(reqwest::get(server.url("/health")).await.unwrap().status(), 200);
    let response = post(&server, "/v1/chat/completions", &chat_body(json!({}))).await;
    assert!(!response.status().is_success(), "the panicking request fails");
    let text = response.text().await.unwrap();
    assert!(text.contains("a scripted failure"), "{text}");
    // The server is broken from now on: a supervisor restarts it.
    let health = reqwest::get(server.url("/health")).await.unwrap();
    assert_eq!(health.status(), 503);
    let again = post(&server, "/v1/chat/completions", &chat_body(json!({}))).await;
    assert_eq!(again.status(), 503);
}

/// A panic inside the real engine's worker thread (the debug-build test hook): the request
/// fails with the panic's message, a request queued behind it fails too instead of waiting
/// forever, and the server answers 503 from then on. Debug builds only, as the hook.
#[cfg(debug_assertions)]
#[tokio::test]
async fn a_panic_in_the_engine_worker_breaks_the_server() {
    let engine = Arc::new(common::engine(common::tiny_tok_config(), 256));
    let backend: Arc<dyn ncm_server::engine::Backend> = engine.clone();
    let state = Arc::new(AppState::ready(backend, "m", SamplingParams::GREEDY));
    let server = TestServer::start(state).await;
    engine.panic_next_request_after(std::time::Duration::from_millis(300));
    let body = chat_body(json!({"max_tokens": 4}));
    let first = post(&server, "/v1/chat/completions", &body);
    let queued = async {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        post(&server, "/v1/chat/completions", &body).await
    };
    let both = async { tokio::join!(first, queued) };
    let (first, queued) = tokio::time::timeout(std::time::Duration::from_secs(30), both)
        .await
        .expect("no request hangs");
    assert!(!first.status().is_success());
    assert!(first.text().await.unwrap().contains("the test hook's panic"));
    assert!(!queued.status().is_success(), "the queued request fails");
    assert!(queued.text().await.unwrap().contains("the test hook's panic"));
    assert_eq!(reqwest::get(server.url("/health")).await.unwrap().status(), 503);
}
