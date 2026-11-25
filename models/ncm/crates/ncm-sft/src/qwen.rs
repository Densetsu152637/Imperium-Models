//! Removing what only exists because of Qwen's chat template, so a record can be re-rendered
//! in the NCM format.
//!
//! Qwen-flavoured samples carry chat-template tokens at the edges of a message, reasoning as a
//! `<think>` block inside the content, tool calls as `<tool_call>` text, tool results as a
//! `<tool_response>` wrapper, and a system prompt with the template's tool boilerplate or its
//! default persona. The archived Emperor and Consul data holds none of these (it stores
//! `reasoning_content` and structured `tool_calls`); this module is the safety net that keeps
//! a Qwen-rendered record from leaking into NCM data, and a record it cannot map is refused.

use crate::format::{
    IM_END, IM_START, THINK_CLOSE, THINK_OPEN, TOOL_CALL_CLOSE, TOOL_CALL_OPEN,
    TOOL_RESPONSE_CLOSE, TOOL_RESPONSE_OPEN,
};
use crate::message::{Message, Role, ToolCall};
use serde_json::Value;

/// What was removed, for the stats.
pub type Stripped = Vec<&'static str>;

const DEFAULT_SYSTEMS: &[&str] = &[
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "You are a helpful assistant.",
];

fn strip_edges(text: &str) -> (String, bool) {
    let mut t = text.trim();
    let mut changed = false;
    if let Some(rest) = t.strip_prefix(IM_START)
        && let Some(newline) = rest.find('\n')
        && Role::parse(rest[..newline].trim()).is_some()
    {
        t = rest[newline + 1..].trim_start();
        changed = true;
    }
    loop {
        let next = [IM_END, "<|endoftext|>"].iter().find_map(|tag| t.strip_suffix(tag));
        match next {
            Some(shorter) => {
                t = shorter.trim_end();
                changed = true;
            }
            None => break,
        }
    }
    (t.to_owned(), changed)
}

/// `<think>reasoning</think>rest` (or, when the template opened the block, `reasoning</think>rest`).
fn split_think(text: &str) -> Option<(String, String)> {
    let t = text.trim_start();
    let (reasoning, rest) = if let Some(after) = t.strip_prefix(THINK_OPEN) {
        let end = after.find(THINK_CLOSE)?;
        (&after[..end], &after[end + THINK_CLOSE.len()..])
    } else {
        let end = t.find(THINK_CLOSE)?;
        if t[..end].contains(THINK_OPEN) {
            return None;
        }
        (&t[..end], &t[end + THINK_CLOSE.len()..])
    };
    Some((reasoning.trim().to_owned(), rest.trim().to_owned()))
}

/// Tool calls written as `<tool_call>{json}</tool_call>` text; the content without them.
fn extract_call_text(content: &str) -> Result<(Vec<ToolCall>, String), &'static str> {
    let mut calls = Vec::new();
    let mut rest = String::new();
    let mut remaining = content;
    while let Some(start) = remaining.find(TOOL_CALL_OPEN) {
        rest.push_str(&remaining[..start]);
        let after = &remaining[start + TOOL_CALL_OPEN.len()..];
        let end = after.find(TOOL_CALL_CLOSE).ok_or("unparsable_tool_call_text")?;
        let payload: Value =
            serde_json::from_str(after[..end].trim()).map_err(|_| "unparsable_tool_call_text")?;
        let name =
            payload.get("name").and_then(Value::as_str).ok_or("unparsable_tool_call_text")?;
        let arguments = match payload.get("arguments").or_else(|| payload.get("parameters")) {
            Some(Value::String(text)) => {
                serde_json::from_str(text).map_err(|_| "unparsable_tool_call_text")?
            }
            Some(value) => value.clone(),
            None => Value::Object(serde_json::Map::new()),
        };
        if !arguments.is_object() {
            return Err("unparsable_tool_call_text");
        }
        calls.push(ToolCall {
            id: format!("call_{}", calls.len() + 1),
            name: name.to_owned(),
            arguments,
        });
        remaining = &after[end + TOOL_CALL_CLOSE.len()..];
    }
    rest.push_str(remaining);
    Ok((calls, rest.trim().to_owned()))
}

fn wrapped_in_tool_response(text: &str) -> Option<&str> {
    let inner = text.trim().strip_prefix(TOOL_RESPONSE_OPEN)?.strip_suffix(TOOL_RESPONSE_CLOSE)?;
    Some(inner.trim())
}

/// The system prompt without Qwen's tool-use boilerplate. The tool definitions themselves come
/// back from the record's `tools` list, so nothing is lost.
fn strip_system(content: &str, stripped: &mut Stripped) -> String {
    let trimmed = content.trim();
    if DEFAULT_SYSTEMS.contains(&trimmed) {
        stripped.push("qwen_default_system_prompt");
        return String::new();
    }
    if let Some(open) = trimmed.find("<tools>")
        && let Some(head) = trimmed[..open].rfind("# Tools")
    {
        let end = ["</tool_call>", "</IMPORTANT>"]
            .iter()
            .filter_map(|tag| trimmed.rfind(tag).map(|i| i + tag.len()))
            .max()
            .filter(|end| *end > open);
        if let Some(end) = end {
            stripped.push("qwen_tool_boilerplate");
            return format!("{}\n{}", &trimmed[..head], &trimmed[end..]).trim().to_owned();
        }
    }
    trimmed.to_owned()
}

/// Clean one message in place. An error code means the message cannot be mapped without loss.
pub fn clean(message: &mut Message, stripped: &mut Stripped) -> Result<(), &'static str> {
    match message.role {
        Role::System => message.content = strip_system(&message.content, stripped),
        Role::User => {
            let (content, changed) = strip_edges(&message.content);
            if changed {
                stripped.push("chat_template_tokens");
            }
            if wrapped_in_tool_response(&content).is_some() {
                return Err("tool_response_as_user");
            }
            message.content = content;
        }
        Role::Tool => {
            let (content, changed) = strip_edges(&message.content);
            if changed {
                stripped.push("chat_template_tokens");
            }
            message.content = match wrapped_in_tool_response(&content) {
                Some(inner) => {
                    stripped.push("tool_response_wrapper");
                    inner.to_owned()
                }
                None => content,
            };
        }
        Role::Assistant => clean_assistant(message, stripped)?,
    }
    Ok(())
}

fn clean_assistant(message: &mut Message, stripped: &mut Stripped) -> Result<(), &'static str> {
    let (mut content, changed) = strip_edges(&message.content);
    if changed {
        stripped.push("chat_template_tokens");
    }
    if let Some((reasoning, rest)) = split_think(&content) {
        match message.reasoning.as_deref().map(str::trim) {
            None | Some("") => message.reasoning = Some(reasoning),
            Some(existing) if existing == reasoning => {}
            Some(_) => return Err("conflicting_reasoning"),
        }
        stripped.push("think_block_in_content");
        content = rest;
    }
    if content.contains(TOOL_CALL_OPEN) {
        if !message.tool_calls.is_empty() {
            return Err("duplicate_tool_calls");
        }
        let (calls, rest) = extract_call_text(&content)?;
        message.tool_calls = calls;
        stripped.push("tool_call_text");
        content = rest;
    }
    message.content = content;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn assistant(content: &str) -> Message {
        Message::assistant(content, None)
    }

    #[test]
    fn edge_tokens_are_removed() {
        let mut message = assistant("<|im_start|>assistant\nHello<|im_end|>\n<|endoftext|>");
        let mut stripped = Vec::new();
        clean(&mut message, &mut stripped).unwrap();
        assert_eq!(message.content, "Hello");
        assert_eq!(stripped, ["chat_template_tokens"]);
    }

    #[test]
    fn think_block_moves_to_reasoning() {
        let mut message = assistant("<think>\nplan it\n</think>\n\nAnswer.");
        clean(&mut message, &mut Vec::new()).unwrap();
        assert_eq!(message.reasoning.as_deref(), Some("plan it"));
        assert_eq!(message.content, "Answer.");
    }

    #[test]
    fn template_opened_think_block_is_understood() {
        let mut message = assistant("plan it\n</think>\n\nAnswer.");
        clean(&mut message, &mut Vec::new()).unwrap();
        assert_eq!(message.reasoning.as_deref(), Some("plan it"));
        assert_eq!(message.content, "Answer.");
    }

    #[test]
    fn duplicate_think_block_is_dropped_but_a_different_one_is_refused() {
        let mut same = Message::assistant("<think>a</think>b", Some("a"));
        clean(&mut same, &mut Vec::new()).unwrap();
        assert_eq!((same.reasoning.as_deref(), same.content.as_str()), (Some("a"), "b"));
        let mut other = Message::assistant("<think>a</think>b", Some("zzz"));
        assert_eq!(clean(&mut other, &mut Vec::new()), Err("conflicting_reasoning"));
    }

    #[test]
    fn tool_call_text_becomes_structured_calls() {
        let mut message = assistant(
            "Looking.\n<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"x\"}}\n</tool_call>",
        );
        clean(&mut message, &mut Vec::new()).unwrap();
        assert_eq!(message.content, "Looking.");
        assert_eq!(message.tool_calls.len(), 1);
        assert_eq!(message.tool_calls[0].name, "web_search");
        assert_eq!(message.tool_calls[0].arguments, json!({"query": "x"}));
        let mut broken = assistant("<tool_call>{nope}</tool_call>");
        assert_eq!(clean(&mut broken, &mut Vec::new()), Err("unparsable_tool_call_text"));
    }

    #[test]
    fn tool_response_wrapper_is_unwrapped_for_tool_turns_only() {
        let mut tool = Message::tool_result("1", "<tool_response>\nresult\n</tool_response>");
        clean(&mut tool, &mut Vec::new()).unwrap();
        assert_eq!(tool.content, "result");
        let mut user = Message::new(Role::User, "<tool_response>\nresult\n</tool_response>");
        assert_eq!(clean(&mut user, &mut Vec::new()), Err("tool_response_as_user"));
    }

    #[test]
    fn qwen_system_boilerplate_is_removed() {
        let system = "You are Emperor.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
            You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\"}\n</tools>\n\n\
            For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n\
            <tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
        let mut message = Message::new(Role::System, system);
        let mut stripped = Vec::new();
        clean(&mut message, &mut stripped).unwrap();
        assert_eq!(message.content, "You are Emperor.");
        assert_eq!(stripped, ["qwen_tool_boilerplate"]);
    }

    #[test]
    fn qwen_default_system_prompt_is_emptied() {
        let mut message = Message::new(
            Role::System,
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        );
        clean(&mut message, &mut Vec::new()).unwrap();
        assert!(message.content.is_empty());
    }

    #[test]
    fn plain_messages_are_untouched() {
        let mut message = Message::assistant("Just text.", Some("Reasoning."));
        let mut stripped = Vec::new();
        clean(&mut message, &mut stripped).unwrap();
        assert_eq!(message, Message::assistant("Just text.", Some("Reasoning.")));
        assert!(stripped.is_empty());
    }
}
