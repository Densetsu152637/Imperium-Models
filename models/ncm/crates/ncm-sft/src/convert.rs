//! One token-era record to one NCM fine-tuning record, or the reason it is not one.

use crate::format::{self, RESERVED, Rendered};
use crate::message::{Message, Role, ToolCall, ToolDef};
use crate::{qwen, tools};
use serde_json::{Map, Value, json};
use std::collections::HashSet;

/// Why a record does not become a fine-tuning sample.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rejection {
    /// Not training data under the token-era rules (another generator, not judge-accepted,
    /// an excluded pilot, a duplicate). Counted, not listed.
    Excluded(String),
    /// Accepted data whose structure cannot be mapped to the NCM format without loss.
    NeedsRegeneration(String),
}

impl Rejection {
    pub fn code(&self) -> &str {
        match self {
            Rejection::Excluded(c) | Rejection::NeedsRegeneration(c) => c,
        }
    }
}

fn regen<T>(code: impl Into<String>) -> Result<T, Rejection> {
    Err(Rejection::NeedsRegeneration(code.into()))
}

/// Which generators and judges produced data we may train on.
#[derive(Clone, Debug)]
pub struct Policy {
    pub allowed_generators: Vec<String>,
    pub allowed_judges: Vec<String>,
    pub excluded_pilots: HashSet<String>,
}

impl Policy {
    /// The token-era teacher (DeepSeek V4 Pro) and judge (Kimi K3.1 Pro).
    pub fn token_era(excluded_pilots: HashSet<String>) -> Policy {
        Policy {
            allowed_generators: vec!["deepseek-v4-pro".into()],
            allowed_judges: vec!["kimi-k3.1-pro".into()],
            excluded_pilots,
        }
    }
}

pub struct Converted {
    pub id: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
    pub rendered: Rendered,
    pub generator: String,
    /// Which Qwen artefacts were removed from the record.
    pub stripped: Vec<&'static str>,
    /// Provenance and bookkeeping worth keeping next to the sample.
    pub meta: Value,
}

fn text_of(meta: &Value, path: &[&str]) -> Option<String> {
    let mut value = meta;
    for key in path {
        value = value.get(key)?;
    }
    value.as_str().map(str::to_owned)
}

/// The provenance rules: the generator and judge must be the allowed ones, the judge must have
/// kept the record, and pilots that were later excluded count for nothing.
fn check_provenance(raw: &Value, policy: &Policy) -> Result<String, Rejection> {
    let excluded = |code: String| Err(Rejection::Excluded(code));
    let meta = raw.get("meta").unwrap_or(&Value::Null);
    let lowered = meta.to_string().to_lowercase();
    if lowered.contains("claude") || lowered.contains("anthropic") {
        return excluded("claude_generator".into());
    }
    let generator = text_of(meta, &["teacher"])
        .or_else(|| text_of(meta, &["generator"]))
        .unwrap_or_else(|| "unknown".into());
    if !policy.allowed_generators.contains(&generator) {
        return excluded(format!("other_generator:{generator}"));
    }
    let pilot = meta.get("pilot").map(|p| p.to_string().trim_matches('"').to_owned());
    if pilot.is_some_and(|p| policy.excluded_pilots.contains(&p)) {
        return excluded("excluded_pilot".into());
    }
    match meta.get("judge") {
        Some(judge) if judge.is_object() => {
            if judge.get("keep").and_then(Value::as_bool) != Some(true) {
                return excluded("not_judge_accepted".into());
            }
            let model = text_of(judge, &["model"]).unwrap_or_else(|| "unknown".into());
            if !policy.allowed_judges.contains(&model) {
                return excluded(format!("other_judge:{model}"));
            }
        }
        // Open-data grafts are screened by the graft pipeline, not judged.
        _ if meta.get("revision").is_some_and(Value::is_string) => {}
        _ => return excluded("not_judged".into()),
    }
    Ok(generator)
}

fn parse_arguments(value: Option<&Value>) -> Result<Value, &'static str> {
    match value {
        None | Some(Value::Null) => Ok(Value::Object(Map::new())),
        Some(Value::String(text)) if text.trim().is_empty() => Ok(Value::Object(Map::new())),
        Some(Value::String(text)) => match serde_json::from_str::<Value>(text) {
            Ok(parsed) if parsed.is_object() => Ok(parsed),
            Ok(_) => Err("arguments_not_an_object"),
            Err(_) => Err("arguments_not_json"),
        },
        Some(object @ Value::Object(_)) => Ok(object.clone()),
        Some(_) => Err("arguments_not_an_object"),
    }
}

fn parse_message(raw: &Value) -> Result<Message, &'static str> {
    let role =
        raw.get("role").and_then(Value::as_str).and_then(Role::parse).ok_or("unknown_role")?;
    let content = match raw.get("content") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(_) => return Err("non_text_content"),
    };
    let reasoning = raw
        .get("reasoning_content")
        .or_else(|| raw.get("reasoning"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let mut tool_calls = Vec::new();
    if let Some(calls) = raw.get("tool_calls").and_then(Value::as_array) {
        for (index, call) in calls.iter().enumerate() {
            let function = call.get("function").unwrap_or(call);
            let name =
                function.get("name").and_then(Value::as_str).ok_or("tool_call_without_name")?;
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .map_or_else(|| format!("call_{}", index + 1), str::to_owned);
            let arguments = parse_arguments(function.get("arguments"))?;
            tool_calls.push(ToolCall { id, name: name.to_owned(), arguments });
        }
    }
    let tool_call_id = raw.get("tool_call_id").and_then(Value::as_str).map(str::to_owned);
    Ok(Message { role, content, reasoning, tool_calls, tool_call_id })
}

fn resolve_tools(raw: &Value) -> Result<Vec<ToolDef>, String> {
    let Some(list) = raw.get("tools").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    list.iter()
        .map(|entry| match entry {
            Value::String(name) => tools::imperium_tool(name).ok_or(format!("unknown_tool:{name}")),
            other => {
                tools::from_value(other).ok_or_else(|| "unreadable_tool_definition".to_owned())
            }
        })
        .collect()
}

fn reserved_in(message: &Message) -> bool {
    let mut texts: Vec<String> = vec![message.content.clone()];
    texts.extend(message.reasoning.clone());
    texts.extend(message.tool_calls.iter().map(|c| c.arguments.to_string()));
    texts.iter().any(|t| RESERVED.iter().any(|tag| t.contains(tag)))
}

/// Check the turn order and pair every tool result with its call, putting results in the order
/// of the calls.
fn order_and_validate(
    messages: Vec<Message>,
    tools: &[ToolDef],
) -> Result<Vec<Message>, &'static str> {
    let mut out: Vec<Message> = Vec::with_capacity(messages.len());
    let mut pending: Vec<String> = Vec::new();
    let mut results: Vec<Message> = Vec::new();
    let mut seen_user = false;

    fn flush(
        out: &mut Vec<Message>,
        pending: &[String],
        results: &mut Vec<Message>,
    ) -> Result<(), &'static str> {
        if !pending.is_empty() {
            return Err("unanswered_tool_call");
        }
        out.append(results);
        Ok(())
    }
    for (index, message) in messages.into_iter().enumerate() {
        match message.role {
            Role::System => {
                if index != 0 {
                    return Err("system_not_first");
                }
                out.push(message);
            }
            Role::User => {
                flush(&mut out, &pending, &mut results)?;
                seen_user = true;
                out.push(message);
            }
            Role::Assistant => {
                flush(&mut out, &pending, &mut results)?;
                if !seen_user {
                    return Err("no_user_turn");
                }
                if message.content.trim().is_empty() && message.tool_calls.is_empty() {
                    return Err("empty_assistant_turn");
                }
                for call in &message.tool_calls {
                    if !tools.is_empty() && !tools.iter().any(|t| t.name == call.name) {
                        return Err("tool_not_offered");
                    }
                    if pending.contains(&call.id) {
                        return Err("duplicate_call_id");
                    }
                    pending.push(call.id.clone());
                }
                let calls: Vec<String> = message.tool_calls.iter().map(|c| c.id.clone()).collect();
                out.push(message);
                // Results for these calls are collected until the next non-tool message.
                results.clear();
                pending = calls;
            }
            Role::Tool => {
                let id = message.tool_call_id.clone().ok_or("tool_result_without_id")?;
                let position = pending.iter().position(|p| *p == id).ok_or("orphan_tool_result")?;
                pending.remove(position);
                results.push(message);
                // Keep results in call order: find the calls of the last assistant turn.
                let calls = out.last().map(|m| m.tool_calls.clone()).unwrap_or_default();
                results.sort_by_key(|r| {
                    calls.iter().position(|c| Some(&c.id) == r.tool_call_id.as_ref())
                });
            }
        }
    }
    flush(&mut out, &pending, &mut results)?;
    match out.last() {
        Some(last) if last.role == Role::Assistant && last.tool_calls.is_empty() => Ok(out),
        _ => Err("not_answered"),
    }
}

/// Convert one record. `raw` is a parsed JSONL line of a token-era data file.
pub fn convert(raw: &Value, policy: &Policy) -> Result<Converted, Rejection> {
    let generator = check_provenance(raw, policy)?;
    let Some(id) = raw.get("id").and_then(Value::as_str).map(str::to_owned) else {
        return regen("no_id");
    };
    let Some(raw_messages) =
        raw.get("messages").and_then(Value::as_array).filter(|m| !m.is_empty())
    else {
        return regen("no_messages");
    };
    let tool_defs = match resolve_tools(raw) {
        Ok(defs) => defs,
        Err(code) => return regen(code),
    };
    let mut stripped = Vec::new();
    let mut messages = Vec::with_capacity(raw_messages.len());
    for raw_message in raw_messages {
        let mut message = match parse_message(raw_message) {
            Ok(m) => m,
            Err(code) => return regen(code),
        };
        if let Err(code) = qwen::clean(&mut message, &mut stripped) {
            return regen(code);
        }
        if message.role == Role::System && message.content.is_empty() {
            continue;
        }
        if reserved_in(&message) {
            return regen("reserved_token_in_content");
        }
        messages.push(message);
    }
    let messages = match order_and_validate(messages, &tool_defs) {
        Ok(m) => m,
        Err(code) => return regen(code),
    };
    let rendered = format::render(&messages, &tool_defs);
    let source_meta = raw.get("meta").unwrap_or(&Value::Null);
    let mut meta = Map::new();
    meta.insert("generator".into(), json!(generator));
    for key in ["task_class", "specificity", "context_budget_k", "kind"] {
        if let Some(value) = raw.get(key) {
            meta.insert(key.into(), value.clone());
        }
    }
    if let Some(licence) = source_meta.get("licence") {
        meta.insert("licence".into(), licence.clone());
    }
    if let Some(judge) = text_of(source_meta, &["judge", "model"]) {
        meta.insert("judge".into(), json!(judge));
    }
    if let Some(mean) = source_meta.get("judge").and_then(|j| j.get("mean")) {
        meta.insert("judge_mean".into(), mean.clone());
    }
    if let Some(pilot) = source_meta.get("pilot") {
        meta.insert("pilot".into(), pilot.clone());
    }
    Ok(Converted {
        id,
        messages,
        tools: tool_defs,
        rendered,
        generator,
        stripped,
        meta: Value::Object(meta),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> Policy {
        Policy::token_era(["1".to_owned()].into_iter().collect())
    }

    fn record(messages: Value) -> Value {
        json!({
            "id": "r1", "task_class": "coding", "tools": ["web_search"],
            "messages": messages,
            "meta": {"teacher": "deepseek-v4-pro", "licence": "L", "pilot": 4,
                     "judge": {"model": "kimi-k3.1-pro", "keep": true, "mean": 4.5}}
        })
    }

    fn search_call(id: &str) -> Value {
        json!({"id": id, "type": "function",
               "function": {"name": "web_search", "arguments": "{\"query\": \"q\"}"}})
    }

    #[test]
    fn a_tool_loop_converts() {
        let raw = record(json!([
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Find"},
            {"role": "assistant", "content": "", "reasoning_content": "Think.",
             "tool_calls": [search_call("a"), search_call("b")]},
            {"role": "tool", "tool_call_id": "b", "content": "rb"},
            {"role": "tool", "tool_call_id": "a", "content": "ra"},
            {"role": "assistant", "content": "Done"}
        ]));
        let done = convert(&raw, &policy()).unwrap();
        assert_eq!(done.generator, "deepseek-v4-pro");
        assert_eq!(done.tools[0].name, "web_search");
        // Results come back in call order.
        assert_eq!(done.messages[3].tool_call_id.as_deref(), Some("a"));
        assert!(done.rendered.text.starts_with("<|im_start|>system\nSys\n\n<tools>\n"));
        assert!(done.rendered.text.contains("<think>\nThink.\n</think>\n<tool_call>"));
        assert_eq!(done.rendered.assistant_spans.len(), 2);
        assert_eq!(done.meta["judge"], "kimi-k3.1-pro");
    }

    #[test]
    fn other_generators_and_unaccepted_records_are_excluded() {
        let mut raw = record(
            json!([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]),
        );
        raw["meta"]["teacher"] = json!("gpt-oss-20b");
        assert_eq!(
            convert(&raw, &policy()).err(),
            Some(Rejection::Excluded("other_generator:gpt-oss-20b".into()))
        );
        let mut raw = record(
            json!([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]),
        );
        raw["meta"]["judge"]["keep"] = json!(false);
        assert_eq!(
            convert(&raw, &policy()).err(),
            Some(Rejection::Excluded("not_judge_accepted".into()))
        );
        raw["meta"]["judge"]["keep"] = json!(true);
        raw["meta"]["pilot"] = json!(1);
        assert_eq!(
            convert(&raw, &policy()).err(),
            Some(Rejection::Excluded("excluded_pilot".into()))
        );
        raw["meta"]["pilot"] = json!(4);
        raw["meta"]["note"] = json!("made with Claude");
        assert_eq!(
            convert(&raw, &policy()).err(),
            Some(Rejection::Excluded("claude_generator".into()))
        );
    }

    #[test]
    fn unmappable_structures_need_regeneration() {
        let cases: Vec<(Value, &str)> = vec![
            (
                json!([{"role": "user", "content": "q"},
                    {"role": "assistant", "content": "", "tool_calls": [search_call("a")]}]),
                "unanswered_tool_call",
            ),
            (
                json!([{"role": "user", "content": "q"},
                    {"role": "assistant", "content": "", "tool_calls": [search_call("a")]},
                    {"role": "tool", "tool_call_id": "a", "content": "r"}]),
                "not_answered",
            ),
            (
                json!([{"role": "user", "content": "q"},
                    {"role": "assistant", "content": "x", "tool_calls": [search_call("a")]},
                    {"role": "user", "content": "again"},
                    {"role": "assistant", "content": "y"}]),
                "unanswered_tool_call",
            ),
            (
                json!([{"role": "user", "content": "q"},
                    {"role": "tool", "tool_call_id": "z", "content": "r"},
                    {"role": "assistant", "content": "y"}]),
                "orphan_tool_result",
            ),
            (
                json!([{"role": "user", "content": "q"}, {"role": "assistant", "content": "  "}]),
                "empty_assistant_turn",
            ),
            (
                json!([{"role": "user", "content": "q <think> inside"}, {"role": "assistant", "content": "a"}]),
                "reserved_token_in_content",
            ),
            (json!([{"role": "user", "content": "q"}]), "not_answered"),
            (
                json!([{"role": "user", "content": "q"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "function": {"name": "spawn_agent", "arguments": "{}"}}]}]),
                "tool_not_offered",
            ),
            (
                json!([{"role": "user", "content": "q"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "function": {"name": "web_search", "arguments": "[1]"}}]}]),
                "arguments_not_an_object",
            ),
        ];
        for (messages, code) in cases {
            assert_eq!(
                convert(&record(messages), &policy()).err(),
                Some(Rejection::NeedsRegeneration(code.into())),
                "{code}"
            );
        }
    }

    #[test]
    fn qwen_rendered_records_are_cleaned() {
        let raw = record(json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>\nr\n</think>\n\nA<|im_end|>"}
        ]));
        let done = convert(&raw, &policy()).unwrap();
        assert!(done.stripped.contains(&"qwen_default_system_prompt"));
        assert!(done.stripped.contains(&"think_block_in_content"));
        assert_eq!(done.messages.len(), 2, "the default system prompt is dropped");
        assert!(done.rendered.text.contains("<think>\nr\n</think>\nA\n<|im_end|>"));
    }

    #[test]
    fn unknown_tool_names_are_refused() {
        let mut raw = record(
            json!([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]),
        );
        raw["tools"] = json!(["teleport"]);
        assert_eq!(
            convert(&raw, &policy()).err(),
            Some(Rejection::NeedsRegeneration("unknown_tool:teleport".into()))
        );
    }
}
