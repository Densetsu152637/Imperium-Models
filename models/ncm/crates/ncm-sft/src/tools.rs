//! The Imperium tool definitions the token-era records name.
//!
//! Records list tools by name only; the definitions are Imperium's tool contract, vendored from
//! `models/emperor/src/emperor/eval/vendor/imperium/tools/` (see `tools/SOURCE`).

use crate::message::ToolDef;
use serde_json::Value;

const FILES: &[(&str, &str)] = &[
    ("spawn_agent", include_str!("../tools/spawn_agent.json")),
    ("await_tasks", include_str!("../tools/await_tasks.json")),
    ("update_plan", include_str!("../tools/update_plan.json")),
    ("delegate_chunks", include_str!("../tools/delegate_chunks.json")),
    ("ask_user", include_str!("../tools/ask_user.json")),
    ("web_search", include_str!("../tools/web_search.json")),
    ("web_fetch", include_str!("../tools/web_fetch.json")),
];

/// The Imperium definition of a named tool, if it is one of the seven.
pub fn imperium_tool(name: &str) -> Option<ToolDef> {
    let (_, source) = FILES.iter().find(|(n, _)| *n == name)?;
    let value: Value = serde_json::from_str(source).expect("vendored tool schema is valid JSON");
    from_value(&value)
}

/// A definition from a JSON object: bare (`name`, `description`, `parameters`) or in OpenAI
/// format (`{"type": "function", "function": {...}}`).
pub fn from_value(value: &Value) -> Option<ToolDef> {
    let body = value.get("function").unwrap_or(value);
    Some(ToolDef {
        name: body.get("name")?.as_str()?.to_owned(),
        description: body.get("description").and_then(Value::as_str).unwrap_or("").to_owned(),
        parameters: body.get("parameters").cloned().unwrap_or_else(|| serde_json::json!({})),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_seven_resolve() {
        for (name, _) in FILES {
            let tool = imperium_tool(name).unwrap();
            assert_eq!(&tool.name, name);
            assert!(!tool.description.is_empty());
        }
        assert!(imperium_tool("nope").is_none());
    }

    #[test]
    fn openai_format_is_unwrapped() {
        let value = serde_json::json!({"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}});
        let tool = from_value(&value).unwrap();
        assert_eq!(tool.name, "f");
        assert_eq!(tool.description, "");
    }
}
