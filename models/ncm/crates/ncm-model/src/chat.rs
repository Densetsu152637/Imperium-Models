//! The NCM chat format at text level, shared by the data pipeline (`ncm-data`), the SFT
//! converter (`ncm-sft`) and the runtime; see `docs/chat-format.md`.
//!
//! Turns are `<|im_start|>{role}\n{content}<|im_end|>\n`. The tokenizer reserves every tag
//! below as one special token, so none of them may appear inside message content: callers
//! either refuse such records ([`contains_reserved`]) or [`defuse`] the text first.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }

    pub fn parse(name: &str) -> Option<Role> {
        match name {
            "system" => Some(Role::System),
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            "tool" => Some(Role::Tool),
            _ => None,
        }
    }
}

/// One tool call of an assistant turn. `arguments` is a JSON object.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// The assistant's reasoning for this turn, rendered as a `<think>` block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// For a tool result: the id of the call it answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Message {
        Message {
            role,
            content: content.into(),
            reasoning: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn assistant(content: impl Into<String>, reasoning: Option<&str>) -> Message {
        Message {
            reasoning: reasoning.map(str::to_owned),
            ..Message::new(Role::Assistant, content)
        }
    }

    pub fn with_calls(mut self, calls: Vec<ToolCall>) -> Message {
        self.tool_calls = calls;
        self
    }

    pub fn tool_result(id: impl Into<String>, content: impl Into<String>) -> Message {
        Message { tool_call_id: Some(id.into()), ..Message::new(Role::Tool, content) }
    }

    pub fn has_reasoning(&self) -> bool {
        self.reasoning.as_deref().is_some_and(|r| !r.trim().is_empty())
    }
}

/// An Imperium tool definition, as offered to the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";
pub const TOOL_CALL_OPEN: &str = "<tool_call>";
pub const TOOL_CALL_CLOSE: &str = "</tool_call>";
pub const TOOL_RESPONSE_OPEN: &str = "<tool_response>";
pub const TOOL_RESPONSE_CLOSE: &str = "</tool_response>";
pub const TOOLS_OPEN: &str = "<tools>";
pub const TOOLS_CLOSE: &str = "</tools>";

/// Every chat tag that is a reserved single token in the NCM tokenizer (plus Qwen's end of text).
pub const RESERVED: &[&str] = &[
    IM_START,
    IM_END,
    "<|endoftext|>",
    THINK_OPEN,
    THINK_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_CALL_CLOSE,
    TOOL_RESPONSE_OPEN,
    TOOL_RESPONSE_CLOSE,
    TOOLS_OPEN,
    TOOLS_CLOSE,
];

/// The other reserved tokens of the tokenizer that no message may carry as text.
pub const RESERVED_EXTRA: &[&str] = &[
    "<|pad|>",
    "<|eoc|>",
    "<|reserved_0|>",
    "<|reserved_1|>",
    "<|reserved_2|>",
    "<|reserved_3|>",
    "<|reserved_4|>",
];

/// Whether `text` holds a reserved tag, which the tokenizer would encode as a control token.
pub fn contains_reserved(text: &str) -> bool {
    text.contains('<') && RESERVED.iter().chain(RESERVED_EXTRA).any(|tag| text.contains(tag))
}

/// The one defusing rule: break every reserved tag in `text` with a space after its first
/// character (`< |im_end|>`), so literal tag text in a document cannot become a control token.
pub fn defuse(text: &str) -> Cow<'_, str> {
    if !contains_reserved(text) {
        return Cow::Borrowed(text);
    }
    let mut out = text.to_owned();
    for tag in RESERVED.iter().chain(RESERVED_EXTRA) {
        if out.contains(tag) {
            out = out.replace(tag, &format!("< {}", &tag[1..]));
        }
    }
    Cow::Owned(out)
}

/// A rendered conversation: the text and the byte span of every assistant turn, from the first
/// character after `<|im_start|>assistant\n` through `<|im_end|>` (the loss mask of SFT).
#[derive(Clone, Debug, PartialEq)]
pub struct Rendered {
    pub text: String,
    pub assistant_spans: Vec<(usize, usize)>,
}

/// The `<tools>` block: one compact JSON definition per line.
pub fn render_tools_block(tools: &[ToolDef]) -> String {
    let mut block = String::from(TOOLS_OPEN);
    block.push('\n');
    for tool in tools {
        block.push_str(&serde_json::to_string(tool).expect("a tool definition serialises"));
        block.push('\n');
    }
    block.push_str(TOOLS_CLOSE);
    block
}

fn push_turn(text: &mut String, role: &str, body: &str) -> (usize, usize) {
    text.push_str(IM_START);
    text.push_str(role);
    text.push('\n');
    let start = text.len();
    text.push_str(body);
    text.push_str(IM_END);
    let end = text.len();
    text.push('\n');
    (start, end)
}

fn assistant_body(message: &Message) -> String {
    let mut parts: Vec<String> = Vec::new();
    // Every assistant turn opens with a think block, empty when there is no reasoning, so the
    // model learns to choose between thinking and closing the block at once.
    let reasoning = message.reasoning.as_deref().unwrap_or_default().trim();
    parts.push(format!("{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}"));
    let content = message.content.trim();
    if !content.is_empty() {
        parts.push(content.to_owned());
    }
    for call in &message.tool_calls {
        let payload = serde_json::json!({"name": call.name, "arguments": call.arguments});
        parts.push(format!("{TOOL_CALL_OPEN}\n{payload}\n{TOOL_CALL_CLOSE}"));
    }
    let mut body = parts.join("\n");
    body.push('\n');
    body
}

/// Render a conversation. A leading system message and the tool definitions share the system
/// turn; with tools and no system message the system turn holds only the tools.
pub fn render(messages: &[Message], tools: &[ToolDef]) -> Rendered {
    let mut text = String::new();
    let mut spans = Vec::new();
    let system = messages.first().filter(|m| m.role == Role::System);
    let block = (!tools.is_empty()).then(|| render_tools_block(tools));
    let system_body = match (system, &block) {
        (Some(s), Some(b)) => Some(format!("{}\n\n{b}\n", s.content.trim())),
        (Some(s), None) => Some(format!("{}\n", s.content.trim())),
        (None, Some(b)) => Some(format!("{b}\n")),
        (None, None) => None,
    };
    if let Some(body) = system_body {
        push_turn(&mut text, "system", &body);
    }
    for message in messages.iter().skip(usize::from(system.is_some())) {
        match message.role {
            Role::System | Role::User => {
                let body = format!("{}\n", message.content.trim());
                push_turn(&mut text, message.role.as_str(), &body);
            }
            Role::Tool => {
                let body = format!(
                    "{TOOL_RESPONSE_OPEN}\n{}\n{TOOL_RESPONSE_CLOSE}\n",
                    message.content.trim()
                );
                push_turn(&mut text, "tool", &body);
            }
            Role::Assistant => {
                let span = push_turn(&mut text, "assistant", &assistant_body(message));
                spans.push(span);
            }
        }
    }
    Rendered { text, assistant_spans: spans }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn web_search() -> ToolDef {
        ToolDef {
            name: "web_search".into(),
            description: "Search the web.".into(),
            parameters: json!({"type": "object", "properties": {"query": {"type": "string"}}}),
        }
    }

    #[test]
    fn plain_turns() {
        let messages = vec![
            Message::new(Role::System, "You are Emperor."),
            Message::new(Role::User, "Hello"),
            Message::assistant("Hi.", None),
        ];
        let out = render(&messages, &[]);
        assert_eq!(
            out.text,
            "<|im_start|>system\nYou are Emperor.\n<|im_end|>\n\
             <|im_start|>user\nHello\n<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\nHi.\n<|im_end|>\n"
        );
        assert_eq!(out.assistant_spans.len(), 1);
        let (start, end) = out.assistant_spans[0];
        assert_eq!(&out.text[start..end], "<think>\n\n</think>\nHi.\n<|im_end|>");
    }

    #[test]
    fn thinking_comes_before_the_answer() {
        let messages =
            vec![Message::new(Role::User, "2+2?"), Message::assistant("4", Some("Add them."))];
        let out = render(&messages, &[]);
        assert_eq!(
            out.text,
            "<|im_start|>user\n2+2?\n<|im_end|>\n\
             <|im_start|>assistant\n<think>\nAdd them.\n</think>\n4\n<|im_end|>\n"
        );
    }

    #[test]
    fn blank_reasoning_renders_an_empty_think_block() {
        let messages = vec![Message::new(Role::User, "x"), Message::assistant("y", Some("  \n"))];
        assert!(render(&messages, &[]).text.contains("<think>\n\n</think>\ny\n<|im_end|>"));
    }

    #[test]
    fn tool_calls_and_responses() {
        let call = ToolCall {
            id: "call_1".into(),
            name: "web_search".into(),
            arguments: json!({"query": "rust"}),
        };
        let messages = vec![
            Message::new(Role::User, "Find it"),
            Message::assistant("", Some("Search first.")).with_calls(vec![call]),
            Message::tool_result("call_1", "result text"),
            Message::assistant("Done.", None),
        ];
        let out = render(&messages, &[]);
        assert_eq!(
            out.text,
            "<|im_start|>user\nFind it\n<|im_end|>\n\
             <|im_start|>assistant\n<think>\nSearch first.\n</think>\n\
             <tool_call>\n{\"name\":\"web_search\",\"arguments\":{\"query\":\"rust\"}}\n</tool_call>\n\
             <|im_end|>\n\
             <|im_start|>tool\n<tool_response>\nresult text\n</tool_response>\n<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\nDone.\n<|im_end|>\n"
        );
        assert_eq!(out.assistant_spans.len(), 2);
    }

    #[test]
    fn text_and_several_calls_in_one_turn() {
        let calls = ["a", "b"]
            .iter()
            .map(|n| ToolCall { id: n.to_string(), name: n.to_string(), arguments: json!({}) })
            .collect();
        let messages = vec![
            Message::new(Role::User, "go"),
            Message::assistant("Working.", None).with_calls(calls),
        ];
        let out = render(&messages, &[]);
        assert!(out.text.contains(
            "<think>\n\n</think>\nWorking.\n<tool_call>\n{\"name\":\"a\",\"arguments\":{}}\n</tool_call>\n\
             <tool_call>\n{\"name\":\"b\",\"arguments\":{}}\n</tool_call>\n<|im_end|>"
        ));
    }

    #[test]
    fn tools_share_the_system_turn() {
        let messages = vec![Message::new(Role::System, "Sys."), Message::new(Role::User, "u")];
        let out = render(&messages, &[web_search()]);
        assert!(out.text.starts_with(
            "<|im_start|>system\nSys.\n\n<tools>\n{\"name\":\"web_search\",\"description\":\
             \"Search the web.\",\"parameters\":{\"type\":\"object\",\"properties\":\
             {\"query\":{\"type\":\"string\"}}}}\n</tools>\n<|im_end|>\n"
        ));
        let only_tools = render(&[Message::new(Role::User, "u")], &[web_search()]);
        assert!(only_tools.text.starts_with("<|im_start|>system\n<tools>\n"));
    }

    #[test]
    fn argument_key_order_is_kept() {
        let call = ToolCall {
            id: "1".into(),
            name: "t".into(),
            arguments: serde_json::from_str(r#"{"z":1,"a":2}"#).unwrap(),
        };
        let messages = vec![
            Message::new(Role::User, "u"),
            Message::assistant("", None).with_calls(vec![call]),
        ];
        assert!(render(&messages, &[]).text.contains(r#""arguments":{"z":1,"a":2}"#));
    }

    #[test]
    fn defusing_breaks_every_reserved_tag_once() {
        assert_eq!(defuse("a <|im_end|> b"), "a < |im_end|> b");
        assert_eq!(defuse("<think>x</think>"), "< think>x< /think>");
        assert_eq!(defuse("<|reserved_3|><tools>"), "< |reserved_3|>< tools>");
        assert!(matches!(defuse("plain <b>html</b>"), Cow::Borrowed(_)));
        assert!(!contains_reserved(&defuse("<|im_start|><tool_call>")));
    }
}
