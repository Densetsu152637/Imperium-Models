//! The structured form of one conversation, before it is rendered to NCM text.

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
