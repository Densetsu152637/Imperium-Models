//! The OpenAI-compatible JSON: parsing chat requests and building responses, streaming chunks
//! and errors in the shapes llama-server uses.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::chat::{Message, Reply, Role, ToolCall, ToolDef};
use crate::engine::{Finish, GenStats};
use crate::sampling::SamplingParams;

/// An error response: `{"error": {"code", "message", "type"}}`.
#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub kind: &'static str,
    pub message: String,
    /// Extra fields of the error object (llama-server adds `n_prompt_tokens` and `n_ctx`).
    pub extra: Option<Value>,
}

impl ApiError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
    }

    pub fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Self { status, kind, message: message.into(), extra: None }
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, "unavailable_error", message)
    }

    pub fn body(&self) -> Value {
        let mut error = json!({
            "code": self.status.as_u16(),
            "message": self.message,
            "type": self.kind,
        });
        if let (Some(Value::Object(extra)), Some(object)) = (&self.extra, error.as_object_mut()) {
            object.extend(extra.clone());
        }
        json!({ "error": error })
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body())).into_response()
    }
}

/// A parsed `POST /v1/chat/completions` body.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    /// Empty when the request offers no tools or sets `tool_choice: "none"`.
    pub tools: Vec<ToolDef>,
    pub stream: bool,
    pub include_usage: bool,
    /// `None` is unlimited (until a stop token or the end of the context).
    pub max_tokens: Option<usize>,
    pub sampling: SamplingParams,
}

fn text_of(value: &Value) -> Result<String, ApiError> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut text = Vec::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") | None => {
                        text.push(part.get("text").and_then(Value::as_str).unwrap_or_default())
                    }
                    Some(other) => {
                        return Err(ApiError::invalid(format!(
                            "content parts of type `{other}` are not supported: this model reads text only"
                        )));
                    }
                }
            }
            Ok(text.join("\n"))
        }
        _ => Err(ApiError::invalid("message content must be a string or a list of text parts")),
    }
}

fn parse_tool_call(value: &Value) -> Result<ToolCall, ApiError> {
    let function = value.get("function").unwrap_or(value);
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::invalid("a tool call needs a function name"))?;
    let arguments = match function.get("arguments") {
        None | Some(Value::Null) => "{}".to_owned(),
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
    };
    let id = value.get("id").and_then(Value::as_str).unwrap_or_default();
    Ok(ToolCall { id: id.to_owned(), name: name.to_owned(), arguments })
}

fn parse_message(value: &Value) -> Result<Message, ApiError> {
    let role = value
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::invalid("every message needs a `role`"))?;
    let role = Role::parse(role)
        .ok_or_else(|| ApiError::invalid(format!("unknown message role `{role}`")))?;
    let content = text_of(value.get("content").unwrap_or(&Value::Null))?;
    let reasoning = ["reasoning_content", "reasoning"]
        .iter()
        .find_map(|key| value.get(*key).and_then(Value::as_str))
        .map(str::to_owned);
    let tool_calls = match value.get("tool_calls") {
        Some(Value::Array(calls)) => calls.iter().map(parse_tool_call).collect::<Result<_, _>>()?,
        _ => Vec::new(),
    };
    Ok(Message { role, content, reasoning, tool_calls })
}

fn parse_tool(value: &Value) -> Result<ToolDef, ApiError> {
    let function = value.get("function").unwrap_or(value);
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::invalid("a tool needs a function name"))?;
    Ok(ToolDef {
        name: name.to_owned(),
        description: function
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        parameters: function
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
    })
}

/// Reads a non-negative integer field; `-1` means unlimited.
fn limit(body: &Value, key: &str) -> Result<Option<Option<usize>>, ApiError> {
    match body.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => match value.as_i64() {
            Some(-1) => Ok(Some(None)),
            Some(n) if n >= 0 => Ok(Some(Some(n as usize))),
            _ => Err(ApiError::invalid(format!("`{key}` must be a non-negative integer"))),
        },
    }
}

fn float(body: &Value, key: &str) -> Result<Option<f32>, ApiError> {
    match body.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_f64()
            .map(|v| Some(v as f32))
            .ok_or_else(|| ApiError::invalid(format!("`{key}` must be a number"))),
    }
}

/// Parses a chat request. Fields the runtime does not use (`model`, penalties, `stop`, ...) are
/// accepted and ignored; `defaults` fill the sampling fields the request leaves out.
pub fn parse_chat_request(
    body: &Value,
    defaults: &SamplingParams,
) -> Result<ChatRequest, ApiError> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| ApiError::invalid("`messages` must be a list"))?;
    if messages.is_empty() {
        return Err(ApiError::invalid("`messages` must not be empty"));
    }
    let messages = messages.iter().map(parse_message).collect::<Result<Vec<_>, _>>()?;
    let none = match body.get("tool_choice") {
        Some(Value::String(choice)) => choice == "none",
        _ => false,
    };
    let tools = match body.get("tools") {
        Some(Value::Array(tools)) if !none => {
            tools.iter().map(parse_tool).collect::<Result<Vec<_>, _>>()?
        }
        _ => Vec::new(),
    };
    if let Some(n) = body.get("n").and_then(Value::as_u64)
        && n != 1
    {
        return Err(ApiError::invalid("only n = 1 is supported"));
    }
    let max_tokens = ["max_completion_tokens", "max_tokens", "n_predict"]
        .iter()
        .find_map(|key| limit(body, key).transpose())
        .transpose()?
        .flatten();
    let mut sampling = *defaults;
    if let Some(t) = float(body, "temperature")? {
        sampling.temperature = t.max(0.0);
    }
    if let Some(p) = float(body, "top_p")? {
        sampling.top_p = p.clamp(0.0, 1.0);
    }
    if let Some(Some(k)) = limit(body, "top_k")? {
        sampling.top_k = k;
    }
    if let Some(seed) = body.get("seed").and_then(Value::as_u64) {
        sampling.seed = Some(seed);
    }
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let include_usage =
        body.pointer("/stream_options/include_usage").and_then(Value::as_bool).unwrap_or(false);
    Ok(ChatRequest { messages, tools, stream, include_usage, max_tokens, sampling })
}

/// The `finish_reason` of a reply.
pub fn finish_reason(finish: Finish, tool_calls: bool) -> &'static str {
    match (finish, tool_calls) {
        (_, true) => "tool_calls",
        (Finish::Stop, false) => "stop",
        (Finish::Length, false) => "length",
    }
}

pub fn usage_json(stats: &GenStats) -> Value {
    json!({
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "total_tokens": stats.prompt_tokens + stats.completion_tokens,
        "prompt_tokens_details": { "cached_tokens": stats.cached_tokens },
    })
}

fn per_second(count: usize, seconds: f64) -> f64 {
    if seconds > 0.0 { count as f64 / seconds } else { 0.0 }
}

fn per_token_ms(count: usize, seconds: f64) -> f64 {
    if count > 0 { seconds * 1000.0 / count as f64 } else { 0.0 }
}

/// llama-server's `timings` object, with prompt processing and decoding timed separately:
/// `cache_n` prompt tokens came from the previous request's cache (prompt-prefix reuse) and
/// `prompt_n` were prefilled.
/// `predicted_per_second` is the decode speed in tokens per second. Extensions: `cache_bytes`,
/// the key/value cache allocated at launch; `graph_capture_ms`, the decode graph's capture
/// (counted in neither `prompt_ms` nor `predicted_ms`); `decode_path`, the loop the request
/// took (`eager`, `graphed-greedy`, `graphed-sampled`, `speculative`, `batched`); with
/// continuous batching, `queue_ms` (the wait for a slot), `mean_batch` (the mean rows of the
/// decode steps it took part in) and `forked` (its prompt's prefill was another slot's).
pub fn timings_json(stats: &GenStats, cache_bytes: usize) -> Value {
    json!({
        "cache_n": stats.cached_tokens,
        "prompt_n": stats.prompt_tokens - stats.cached_tokens,
        "prompt_ms": stats.prompt_s * 1000.0,
        "prompt_per_token_ms": per_token_ms(stats.prompt_tokens - stats.cached_tokens, stats.prompt_s),
        "prompt_per_second": per_second(stats.prompt_tokens - stats.cached_tokens, stats.prompt_s),
        "predicted_n": stats.completion_tokens,
        "predicted_ms": stats.decode_s * 1000.0,
        "predicted_per_token_ms": per_token_ms(stats.decode_tokens, stats.decode_s),
        "predicted_per_second": per_second(stats.decode_tokens, stats.decode_s),
        "cache_bytes": cache_bytes,
        "graph_capture_ms": stats.capture_s * 1000.0,
        "decode_path": stats.path.name(),
        "hybrid": match stats.path {
            crate::engine::DecodePath::Hybrid(a) => json!({
                "blocks": a.blocks,
                "mean_first_accepted": a.mean_first_accepted(),
                "first_passes": a.first_passes,
                "redecodes": a.redecodes,
                "ar_tokens": a.ar_tokens,
                "graphed": a.graphed,
            }),
            _ => Value::Null,
        },
        "queue_ms": stats.queue_s * 1000.0,
        "mean_batch": stats.mean_batch,
        "forked": stats.forked,
    })
}

/// A tool call as `message.tool_calls[]` / `delta.tool_calls[]` carry it.
pub fn tool_call_json(index: usize, id: &str, name: &str, arguments: &str, delta: bool) -> Value {
    let mut call = json!({
        "id": id,
        "type": "function",
        "function": { "name": name, "arguments": arguments },
    });
    if delta {
        call["index"] = json!(index);
    }
    call
}

/// The id of tool call `index` of request `request`.
pub fn tool_call_id(request: &str, index: usize) -> String {
    format!("call_{request}_{index}")
}

/// A complete (non-streaming) chat completion.
#[allow(clippy::too_many_arguments)]
pub fn completion_json(
    id: &str,
    created: u64,
    model: &str,
    reply: &Reply,
    stats: &GenStats,
    cache_bytes: usize,
) -> Value {
    let tool_calls: Vec<Value> = reply
        .tool_calls
        .iter()
        .enumerate()
        .map(|(i, (name, arguments))| {
            tool_call_json(i, &tool_call_id(id, i), name, arguments, false)
        })
        .collect();
    let mut message = json!({
        "role": "assistant",
        "content": if reply.content.is_empty() && !tool_calls.is_empty() {
            Value::Null
        } else {
            Value::String(reply.content.clone())
        },
    });
    if !reply.reasoning.is_empty() {
        message["reasoning_content"] = json!(reply.reasoning);
    }
    let has_calls = !tool_calls.is_empty();
    if has_calls {
        message["tool_calls"] = Value::Array(tool_calls);
    }
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "system_fingerprint": concat!("ncm-server-", env!("CARGO_PKG_VERSION")),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason(stats.finish, has_calls),
        }],
        "usage": usage_json(stats),
        "timings": timings_json(stats, cache_bytes),
    })
}

/// One `chat.completion.chunk`.
pub fn chunk_json(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish: Option<&str>,
) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
    })
}

/// The usage-only chunk that closes a stream when `stream_options.include_usage` is set.
pub fn usage_chunk_json(
    id: &str,
    created: u64,
    model: &str,
    stats: &GenStats,
    cache_bytes: usize,
) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage_json(stats),
        "timings": timings_json(stats, cache_bytes),
    })
}
