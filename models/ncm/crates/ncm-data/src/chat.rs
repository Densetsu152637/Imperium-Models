//! Chat and tool-use records as one training string in the NCM chat format.
//!
//! This module only maps source rows (Nemotron tool sets, OpenThoughts-style reasoning
//! traces, ...) to structured [`Message`]s; the text itself comes from the one shared renderer,
//! `ncm_model::chat`, and every piece of source text goes through its one defusing rule.

use std::borrow::Cow;

use std::sync::atomic::{AtomicU64, Ordering};

use ncm_model::chat::{self, Message, Role, ToolCall, ToolDef};

/// Chat and trace records dropped by the no-refusals rule ([`ncm_model::refusal`]) since the
/// last [`take_refusals_dropped`]. Sources are tokenized one at a time, so this is per source.
static REFUSALS_DROPPED: AtomicU64 = AtomicU64::new(0);

/// The count of records dropped by the no-refusals rule since the last call, reset to zero.
pub fn take_refusals_dropped() -> u64 {
    REFUSALS_DROPPED.swap(0, Ordering::SeqCst)
}

/// Whether `messages` hold a topic refusal; counts it if so. Princeps learns no topic
/// refusals, in pretraining as in fine-tuning.
fn refuses(messages: &[Message]) -> bool {
    let hit = ncm_model::refusal::refusal(messages).is_some();
    if hit {
        REFUSALS_DROPPED.fetch_add(1, Ordering::SeqCst);
    }
    hit
}
use serde_json::Value;

/// Make text safe to encode: a literal special-token string in a document must not become the
/// control token (the shared rule, `< |im_end|>`).
pub fn defuse(text: &str) -> Cow<'_, str> {
    chat::defuse(text)
}

/// [`defuse`] every string inside a JSON value (keys are tag-free identifiers and stay).
fn defuse_value(value: &Value) -> Value {
    match value {
        Value::String(s) => Value::String(defuse(s).into_owned()),
        Value::Array(items) => Value::Array(items.iter().map(defuse_value).collect()),
        Value::Object(map) => {
            Value::Object(map.iter().map(|(k, v)| (k.clone(), defuse_value(v))).collect())
        }
        other => other.clone(),
    }
}

fn parse_maybe_json(v: &Value) -> Value {
    match v {
        Value::String(s) => serde_json::from_str(s).unwrap_or(Value::Null),
        other => other.clone(),
    }
}

/// Message text: a string, or a list of `{type: "text", text}` parts.
fn content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str).or_else(|| p.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn tool_list(row: &Value) -> Vec<Value> {
    let direct = row.get("tools").map(parse_maybe_json);
    let nested = row
        .get("metadata")
        .map(parse_maybe_json)
        .and_then(|m| m.get("tools").map(parse_maybe_json));
    for candidate in [direct, nested].into_iter().flatten() {
        if let Value::Array(items) = candidate
            && !items.is_empty()
        {
            return items;
        }
    }
    Vec::new()
}

/// A tool definition, bare or in OpenAI `{"type": "function", "function": {...}}` form.
fn tool_def(value: &Value) -> Option<ToolDef> {
    let body = value.get("function").unwrap_or(value);
    Some(ToolDef {
        name: defuse(body.get("name")?.as_str()?).into_owned(),
        description: body
            .get("description")
            .and_then(Value::as_str)
            .map(|d| defuse(d).into_owned())
            .unwrap_or_default(),
        parameters: body.get("parameters").map_or_else(|| serde_json::json!({}), defuse_value),
    })
}

fn tool_call(call: &Value) -> Option<ToolCall> {
    let function = call.get("function").unwrap_or(call);
    let name = function.get("name")?.as_str()?;
    let arguments = match function.get("arguments") {
        Some(Value::String(s)) => serde_json::from_str(s).unwrap_or(Value::String(s.clone())),
        Some(v) => v.clone(),
        None => Value::Object(Default::default()),
    };
    Some(ToolCall {
        id: String::new(),
        name: defuse(name).into_owned(),
        arguments: defuse_value(&arguments),
    })
}

/// Follow a dotted path through objects and arrays (`conversations.1.value`).
pub fn lookup<'a>(row: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.').try_fold(row, |v, key| match v {
        Value::Array(items) => items.get(key.parse::<usize>().ok()?),
        _ => v.get(key),
    })
}

/// Split a model response into its reasoning and its answer. R1-style responses open with
/// `<think>`; the reasoning must be closed by `</think>` or the trace is incomplete.
pub fn split_reasoning(response: &str) -> Option<(&str, &str)> {
    let body = response.trim_start();
    let body = body.strip_prefix("<think>").unwrap_or(body);
    let (think, answer) = body.split_once("</think>")?;
    let (think, answer) = (think.trim(), answer.trim());
    (!think.is_empty() && !answer.is_empty()).then_some((think, answer))
}

/// A problem and its reasoning response as one user turn and one assistant turn, with the
/// reasoning between `<think>` and `</think>` and the answer after it.
pub fn render_trace(problem: &str, response: &str) -> Option<String> {
    let (think, answer) = split_reasoning(response)?;
    let problem = problem.trim();
    if problem.is_empty() {
        return None;
    }
    let messages = [
        Message::new(Role::User, defuse(problem)),
        Message::assistant(defuse(answer), Some(&defuse(think))),
    ];
    if refuses(&messages) {
        return None;
    }
    Some(chat::render(&messages, &[]).text)
}

/// The assistant message of a record: reasoning from `reasoning_content`/`reasoning` or from an
/// inline `<think>` block of the content, then the text and the structured tool calls.
fn assistant_message(m: &Value, content: &str) -> Message {
    let field = ["reasoning_content", "reasoning"]
        .iter()
        .filter_map(|k| m.get(*k).and_then(Value::as_str))
        .find(|r| !r.trim().is_empty());
    let body = content.trim_start();
    let inline = body.strip_prefix(chat::THINK_OPEN).unwrap_or(body).split_once(chat::THINK_CLOSE);
    let (reasoning, text) = match inline {
        Some((think, answer)) => (Some(think).filter(|t| !t.trim().is_empty()).or(field), answer),
        None => (field, content),
    };
    let calls = m
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| calls.iter().filter_map(tool_call).collect())
        .unwrap_or_default();
    let reasoning = reasoning.map(|r| defuse(r).into_owned());
    Message::assistant(defuse(text), reasoning.as_deref()).with_calls(calls)
}

/// Render one record, or `None` when it has no usable messages.
pub fn render(row: &Value, messages_field: &str) -> Option<String> {
    let source = parse_maybe_json(row.get(messages_field)?);
    let source = source.as_array()?;
    let tools: Vec<ToolDef> = tool_list(row).iter().filter_map(tool_def).collect();

    let mut system = String::new();
    let mut turns: Vec<Message> = Vec::new();
    for m in source {
        let role = m.get("role")?.as_str()?;
        let content = content_text(m.get("content").unwrap_or(&Value::Null));
        match role {
            "system" => {
                let text = defuse(content.trim());
                if !text.is_empty() {
                    if !system.is_empty() {
                        system.push('\n');
                    }
                    system.push_str(&text);
                }
            }
            "user" => turns.push(Message::new(Role::User, defuse(&content))),
            "assistant" => turns.push(assistant_message(m, &content)),
            "tool" | "function" | "ipython" => {
                turns.push(Message::new(Role::Tool, defuse(&content)));
            }
            _ => return None,
        }
    }
    if turns.is_empty() {
        return None;
    }
    let mut messages = Vec::with_capacity(turns.len() + 1);
    if !system.is_empty() {
        messages.push(Message::new(Role::System, system));
    }
    messages.extend(turns);
    if refuses(&messages) {
        return None;
    }
    Some(chat::render(&messages, &tools).text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defuses_literal_specials() {
        assert_eq!(defuse("a <|im_end|> b"), "a < |im_end|> b");
        assert_eq!(defuse("<think>x</think>"), "< think>x< /think>");
        assert!(matches!(defuse("plain <b>html</b>"), Cow::Borrowed(_)));
    }

    #[test]
    fn a_tool_calling_record_renders_like_the_shared_renderer() {
        let row = json!({
            "tools": [{"type": "function", "function": {
                "name": "get_time", "description": "Clock.",
                "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}}}}],
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "time?"},
                {"role": "assistant", "content": "", "reasoning_content": "need the clock",
                 "tool_calls": [{"function": {"name": "get_time", "arguments": "{\"tz\":\"UTC\"}"}}]},
                {"role": "tool", "content": "12:00"},
                {"role": "assistant", "content": "It is 12:00."}
            ]
        });
        assert_eq!(
            render(&row, "messages").unwrap(),
            "<|im_start|>system\nBe brief.\n\n<tools>\n\
             {\"name\":\"get_time\",\"description\":\"Clock.\",\"parameters\":{\"type\":\"object\",\
             \"properties\":{\"tz\":{\"type\":\"string\"}}}}\n</tools>\n<|im_end|>\n\
             <|im_start|>user\ntime?\n<|im_end|>\n\
             <|im_start|>assistant\n<think>\nneed the clock\n</think>\n\
             <tool_call>\n{\"name\":\"get_time\",\"arguments\":{\"tz\":\"UTC\"}}\n</tool_call>\n\
             <|im_end|>\n\
             <|im_start|>tool\n<tool_response>\n12:00\n</tool_response>\n<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\nIt is 12:00.\n<|im_end|>\n"
        );
    }

    #[test]
    fn inline_think_blocks_become_reasoning_and_stray_tags_are_defused() {
        let row = json!({"messages": [
            {"role": "user", "content": "q <|im_end|>"},
            {"role": "assistant", "content": "<think>\nhmm\n</think>\n\nanswer"}
        ]});
        assert_eq!(
            render(&row, "messages").unwrap(),
            "<|im_start|>user\nq < |im_end|>\n<|im_end|>\n\
             <|im_start|>assistant\n<think>\nhmm\n</think>\nanswer\n<|im_end|>\n"
        );
    }

    #[test]
    fn a_reasoning_trace_renders_like_the_shared_renderer() {
        let t = render_trace("2+2?", "<think>\nadd them\n</think>\n\nIt is 4.").unwrap();
        assert_eq!(
            t,
            "<|im_start|>user\n2+2?\n<|im_end|>\n\
             <|im_start|>assistant\n<think>\nadd them\n</think>\nIt is 4.\n<|im_end|>\n"
        );
        let shared = chat::render(
            &[Message::new(Role::User, "2+2?"), Message::assistant("It is 4.", Some("add them"))],
            &[],
        );
        assert_eq!(t, shared.text);
        // QwQ-style inline tags work too; unfinished reasoning or no answer is rejected.
        assert!(render_trace("q", "<think> hmm </think> done").is_some());
        assert!(render_trace("q", "<think>never closed").is_none());
        assert!(render_trace("q", "<think>x</think>  ").is_none());
        assert!(render_trace("", "<think>x</think>y").is_none());
    }

    #[test]
    fn looks_up_dotted_paths() {
        let v = json!({"c": [{"v": "a"}, {"v": "b"}], "ok": [true, false]});
        assert_eq!(lookup(&v, "c.1.v").and_then(Value::as_str), Some("b"));
        assert_eq!(lookup(&v, "ok.1").and_then(Value::as_bool), Some(false));
        assert!(lookup(&v, "c.5.v").is_none());
    }

    #[test]
    fn rejects_records_without_turns() {
        assert!(render(&json!({"messages": []}), "messages").is_none());
        assert!(render(&json!({"other": 1}), "messages").is_none());
        let odd = json!({"messages": [{"role": "narrator", "content": "x"}]});
        assert!(render(&odd, "messages").is_none());
    }
}
