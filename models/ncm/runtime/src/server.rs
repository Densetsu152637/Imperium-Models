//! The HTTP server: llama-server's endpoints, as far as Imperium and OpenAI clients use them.
//!
//! | Route | Purpose |
//! |---|---|
//! | `GET /health` | 200 `{"status":"ok"}` when the model is loaded, 503 while it loads (Imperium polls this) |
//! | `GET /v1/models` | the one loaded model |
//! | `GET /props` | `default_generation_settings.n_ctx` (the context one request gets), slots, cache size |
//! | `POST /v1/chat/completions` | chat completion, optionally streamed as server-sent events |
//! | `POST /tokenize`, `POST /detokenize` | the model's own tokenizer (Imperium counts context with it) |
//! | `POST /apply-template` | the rendered prompt for `messages` and `tools` |

use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::Router;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{StatusCode, header};
use axum::response::sse::{Event as SseEvent, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::api::{
    self, ApiError, ChatRequest, chunk_json, completion_json, parse_chat_request, tool_call_id,
    tool_call_json, usage_chunk_json,
};
use crate::chat::{self, Event, OutputParser, Reply};
use crate::constrain::{Constraint, ToolCallGuard};
use crate::engine::{Backend, EngineError, GenRequest, GenStats};
use crate::sampling::SamplingParams;

/// Largest request body: long contexts are sent whole.
const MAX_BODY_BYTES: usize = 1 << 30;

/// Shared by every handler.
pub struct AppState {
    backend: OnceLock<Arc<dyn Backend>>,
    /// Why loading failed, if it did.
    failure: OnceLock<String>,
    /// Why the backend broke, if generation panicked on the caller's thread.
    broken: Arc<OnceLock<String>>,
    /// The model name in responses.
    pub alias: String,
    pub model_path: String,
    pub slots: usize,
    pub defaults: SamplingParams,
    counter: AtomicU64,
}

impl AppState {
    /// A server whose model is still loading; `set_backend` finishes it.
    pub fn loading(alias: &str, model_path: &str, slots: usize, defaults: SamplingParams) -> Self {
        Self {
            backend: OnceLock::new(),
            failure: OnceLock::new(),
            broken: Arc::new(OnceLock::new()),
            alias: alias.to_owned(),
            model_path: model_path.to_owned(),
            slots,
            defaults,
            counter: AtomicU64::new(0),
        }
    }

    pub fn ready(backend: Arc<dyn Backend>, alias: &str, defaults: SamplingParams) -> Self {
        let state = Self::loading(alias, "", 1, defaults);
        state.set_backend(backend);
        state
    }

    pub fn set_backend(&self, backend: Arc<dyn Backend>) {
        self.backend.set(backend).ok();
    }

    pub fn set_failure(&self, message: String) {
        self.failure.set(message).ok();
    }

    /// Why the model failed to load, if it did.
    pub fn failure(&self) -> Option<String> {
        self.failure.get().cloned()
    }

    /// The loaded backend, or why there is none to serve with: still loading, failed to load,
    /// or broken by a panic (then every request, `/health` included, answers 503, so that
    /// Imperium's supervisor restarts the server).
    fn backend(&self) -> Result<&Arc<dyn Backend>, ApiError> {
        let backend = self.backend.get().ok_or_else(|| match self.failure.get() {
            Some(message) => ApiError::unavailable(format!("the model failed to load: {message}")),
            None => ApiError::unavailable("Loading model"),
        })?;
        match self.broken.get().cloned().or_else(|| backend.broken()) {
            Some(message) => Err(ApiError::unavailable(format!(
                "the model stopped after an internal error: {message}"
            ))),
            None => Ok(backend),
        }
    }

    fn next_id(&self) -> String {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        format!("{:x}{n:04x}", now & 0xFFFF_FFFF_FFFF)
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/v1/models", get(models))
        .route("/models", get(models))
        .route("/props", get(props))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/apply-template", post(apply_template))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/chat/completions", post(chat_completions))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

async fn health(State(state): State<Arc<AppState>>) -> Response {
    match state.backend() {
        Ok(_) => Json(json!({"status": "ok"})).into_response(),
        Err(error) => error.into_response(),
    }
}

fn unix_time() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

async fn models(State(state): State<Arc<AppState>>) -> Result<Json<Value>, ApiError> {
    let info = state.backend()?.info();
    Ok(Json(json!({
        "object": "list",
        "data": [{
            "id": state.alias,
            "object": "model",
            "created": unix_time(),
            "owned_by": "ncm-server",
            "meta": {
                "arch": info.arch,
                "n_params": info.params,
                "n_ctx": info.ctx,
                "n_vocab": info.vocab_size,
            },
        }],
    })))
}

async fn props(State(state): State<Arc<AppState>>) -> Result<Json<Value>, ApiError> {
    let info = state.backend()?.info();
    Ok(Json(json!({
        "default_generation_settings": {
            "n_ctx": info.ctx,
            "params": {
                "n_ctx": info.ctx,
                "temperature": f64::from((state.defaults.temperature * 1e4).round()) / 1e4,
                "top_k": state.defaults.top_k,
                "top_p": f64::from((state.defaults.top_p * 1e4).round()) / 1e4,
            },
        },
        "total_slots": state.slots,
        "model_path": state.model_path,
        "model_alias": state.alias,
        "chat_template": "",
        "ncm": {
            "arch": info.arch,
            "n_params": info.params,
            "dtype": info.dtype,
            "device": info.device,
            "cache_bytes": info.cache_bytes,
            "tokens_per_pass": info.unit,
        },
    })))
}

fn engine_error(error: EngineError) -> ApiError {
    match error {
        EngineError::ContextExceeded { prompt, ctx: n_ctx } => {
            let mut api = ApiError::invalid(format!(
                "the request has {prompt} tokens but the context is {n_ctx}"
            ));
            api.kind = "exceed_context_size_error";
            api.extra = Some(json!({"n_prompt_tokens": prompt, "n_ctx": n_ctx}));
            api
        }
        EngineError::EmptyPrompt => ApiError::invalid("the prompt is empty"),
        other => {
            ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", other.to_string())
        }
    }
}

// ----------------------------------------------------------------------- tokenizer endpoints

async fn tokenize(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let backend = state.backend()?;
    let content = match body.get("content") {
        Some(Value::String(text)) => text.clone(),
        _ => return Err(ApiError::invalid("`content` must be a string")),
    };
    let parse_special = body.get("parse_special").and_then(Value::as_bool).unwrap_or(true);
    // The tokenizer always reads special tokens written in text; without `parse_special` they
    // are defused so they come out as ordinary text.
    let text = if parse_special { content } else { chat::defuse(&content) };
    let tokens = backend.vocab().encode(&text).map_err(|e| ApiError::invalid(e.to_string()))?;
    Ok(Json(json!({ "tokens": tokens })))
}

async fn detokenize(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let backend = state.backend()?;
    let tokens: Vec<u32> = body
        .get("tokens")
        .and_then(Value::as_array)
        .ok_or_else(|| ApiError::invalid("`tokens` must be a list of ids"))?
        .iter()
        .filter_map(|t| t.as_u64().map(|t| t as u32))
        .collect();
    Ok(Json(json!({ "content": backend.vocab().decode(&tokens) })))
}

async fn apply_template(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    state.backend()?;
    let request = parse_chat_request(&body, &state.defaults)?;
    let prompt = chat::render(&request.messages, &request.tools, true);
    Ok(Json(json!({ "prompt": prompt })))
}

// ------------------------------------------------------------------------- chat completions

/// What the generation thread sends back.
enum Message {
    Token(u32),
    Done(Result<GenStats, EngineError>),
}

/// Starts generation on a blocking thread and returns the channel its tokens arrive on. The
/// thread stops by itself when the receiver is dropped (the client left). A panic in the backend
/// fails the request and marks the server broken (`broken`).
fn start_generation(
    backend: Arc<dyn Backend>,
    broken: Arc<OnceLock<String>>,
    request: GenRequest,
) -> mpsc::Receiver<Message> {
    let (tx, rx) = mpsc::channel(256);
    // A client that leaves drops the receiver: abandon the request then, even mid-prefill,
    // before any token would have noticed.
    let (watch, cancel) = (tx.clone(), request.cancel.clone());
    tokio::spawn(async move {
        watch.closed().await;
        cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    tokio::task::spawn_blocking(move || {
        let sender = tx.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend
                .generate(request, &mut |token| sender.blocking_send(Message::Token(token)).is_ok())
        }));
        let result = result.unwrap_or_else(|panic| {
            let message = crate::engine::panic_message(panic.as_ref());
            broken.set(message.clone()).ok();
            Err(EngineError::Device(format!("generation panicked: {message}")))
        });
        tx.blocking_send(Message::Done(result)).ok();
    });
    rx
}

struct Prepared {
    request: ChatRequest,
    generation: GenRequest,
    parse_tools: bool,
}

fn prepare(state: &AppState, body: &Value) -> Result<(Arc<dyn Backend>, Prepared), ApiError> {
    let backend = state.backend()?.clone();
    let request = parse_chat_request(body, &state.defaults)?;
    let text = chat::render(&request.messages, &request.tools, true);
    let prompt = backend.vocab().encode(&text).map_err(|e| ApiError::invalid(e.to_string()))?;
    backend.check_prompt(prompt.len()).map_err(engine_error)?;
    let parse_tools = !request.tools.is_empty();
    let constraint: Option<Box<dyn Constraint>> = parse_tools
        .then(|| Box::new(ToolCallGuard::new(backend.vocab().clone())) as Box<dyn Constraint>);
    let generation = GenRequest {
        prompt,
        max_tokens: request.max_tokens.unwrap_or(usize::MAX),
        sampling: request.sampling,
        constraint,
        cancel: Default::default(),
    };
    Ok((backend, Prepared { request, generation, parse_tools }))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(body) = body.map_err(|e| ApiError::invalid(e.body_text()))?;
    // Rendering and tokenising a long prompt is work for a blocking thread, not the runtime's.
    let prepare_state = state.clone();
    let (backend, prepared) =
        tokio::task::spawn_blocking(move || prepare(&prepare_state, &body)).await.map_err(
            |e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string()),
        )??;
    let id = format!("chatcmpl-{}", state.next_id());
    let model = state.alias.clone();
    let created = unix_time();
    let Prepared { request, generation, parse_tools } = prepared;
    let cache_bytes = backend.info().cache_bytes;
    let rx = start_generation(backend.clone(), state.broken.clone(), generation);
    if request.stream {
        Ok(stream_response(backend, rx, request, parse_tools, id, model, created, cache_bytes))
    } else {
        collect_response(backend, rx, parse_tools, id, model, created, cache_bytes).await
    }
}

async fn collect_response(
    backend: Arc<dyn Backend>,
    mut rx: mpsc::Receiver<Message>,
    parse_tools: bool,
    id: String,
    model: String,
    created: u64,
    cache_bytes: usize,
) -> Result<Response, ApiError> {
    let vocab = backend.vocab().clone();
    let mut parser = OutputParser::new(&vocab, parse_tools);
    let mut reply = Reply::default();
    let mut stats = None;
    while let Some(message) = rx.recv().await {
        match message {
            Message::Token(token) => parser.push(token).into_iter().for_each(|e| reply.apply(e)),
            Message::Done(result) => stats = Some(result.map_err(engine_error)?),
        }
    }
    parser.finish().into_iter().for_each(|e| reply.apply(e));
    let stats = stats.ok_or_else(|| {
        ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "generation stopped early")
    })?;
    let body = completion_json(&id, created, &model, &reply, &stats, cache_bytes);
    Ok(Json(body).into_response())
}

fn sse(value: &Value) -> SseEvent {
    SseEvent::default().data(value.to_string())
}

/// Streams the reply as server-sent events: a role chunk, content / reasoning / tool-call
/// deltas as the parser finds them, a chunk with the finish reason and timings, an optional
/// usage chunk, and `[DONE]`.
#[allow(clippy::too_many_arguments)]
fn stream_response(
    backend: Arc<dyn Backend>,
    mut rx: mpsc::Receiver<Message>,
    request: ChatRequest,
    parse_tools: bool,
    id: String,
    model: String,
    created: u64,
    cache_bytes: usize,
) -> Response {
    let (out, out_rx) = mpsc::channel::<Result<SseEvent, Infallible>>(64);
    tokio::spawn(async move {
        let vocab = backend.vocab().clone();
        let mut parser = OutputParser::new(&vocab, parse_tools);
        let chunk =
            |delta: Value, finish: Option<&str>| chunk_json(&id, created, &model, delta, finish);
        let role = chunk(json!({"role": "assistant", "content": ""}), None);
        if out.send(Ok(sse(&role))).await.is_err() {
            return;
        }
        let mut calls = 0usize;
        let mut stats = None;
        let mut failure = None;
        while let Some(message) = rx.recv().await {
            let events = match message {
                Message::Token(token) => parser.push(token),
                Message::Done(result) => {
                    match result {
                        Ok(done) => stats = Some(done),
                        Err(error) => failure = Some(error),
                    }
                    break;
                }
            };
            for event in events {
                let delta = match event {
                    Event::Content(text) => json!({"content": text}),
                    Event::Reasoning(text) => json!({"reasoning_content": text}),
                    Event::ToolCall { name, arguments } => {
                        let call = tool_call_json(
                            calls,
                            &tool_call_id(&id, calls),
                            &name,
                            &arguments,
                            true,
                        );
                        calls += 1;
                        json!({"tool_calls": [call]})
                    }
                };
                if out.send(Ok(sse(&chunk(delta, None)))).await.is_err() {
                    // The client left: dropping `rx` below stops the generation.
                    return;
                }
            }
        }
        for event in parser.finish() {
            let delta = match event {
                Event::Content(text) => json!({"content": text}),
                Event::Reasoning(text) => json!({"reasoning_content": text}),
                Event::ToolCall { .. } => continue,
            };
            out.send(Ok(sse(&chunk(delta, None)))).await.ok();
        }
        if let Some(error) = failure {
            let body = engine_error(error).body();
            out.send(Ok(SseEvent::default().data(body.to_string()))).await.ok();
        } else if let Some(stats) = stats {
            let reason = api::finish_reason(stats.finish, calls > 0);
            let mut last = chunk(json!({}), Some(reason));
            last["timings"] = api::timings_json(&stats, cache_bytes);
            out.send(Ok(sse(&last))).await.ok();
            if request.include_usage {
                let usage = usage_chunk_json(&id, created, &model, &stats, cache_bytes);
                out.send(Ok(sse(&usage))).await.ok();
            }
        }
        out.send(Ok(SseEvent::default().data("[DONE]"))).await.ok();
    });
    let stream = ReceiverStream::new(out_rx).map(|event| event);
    let mut response = Sse::new(stream).into_response();
    response.headers_mut().insert(header::CACHE_CONTROL, "no-cache".parse().expect("header"));
    response
}

/// Binds the server's routes to `listener` and serves until `shutdown` resolves.
pub async fn serve(
    listener: tokio::net::TcpListener,
    state: Arc<AppState>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    axum::serve(listener, router(state)).with_graceful_shutdown(shutdown).await
}
