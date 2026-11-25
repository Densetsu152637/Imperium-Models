//! The NCM chat format: rendering conversations to the text the models were fine-tuned on, and
//! parsing generated output back into reasoning, content and tool calls.
//!
//! The format is rendered by the shared `ncm_model::chat` (stage 3 defines it):
//!
//! ```text
//! <|im_start|>system
//! {system prompt}
//!
//! <tools>
//! {"name":"read_file","description":"...","parameters":{...}}
//! </tools>
//! <|im_end|>
//! <|im_start|>user
//! {text}
//! <|im_end|>
//! <|im_start|>assistant
//! <think>
//! {reasoning}
//! </think>
//! {text}
//! <tool_call>
//! {"name":"read_file","arguments":{"path":"a.txt"}}
//! </tool_call>
//! <|im_end|>
//! <|im_start|>tool
//! <tool_response>
//! {result}
//! </tool_response>
//! <|im_end|>
//! <|im_start|>assistant
//! ```
//!
//! Tool definitions live in the system turn, one compact JSON object per line between `<tools>`
//! and `</tools>`; with no system message the system turn holds only that block. Every turn
//! body ends with a newline before `<|im_end|>`. A request ends with the open generation prompt
//! `<|im_start|>assistant\n`.

use ncm_model::chat as shared;
use serde_json::Value;

use crate::vocab::{Specials, Utf8Stream, Vocab};

pub const GENERATION_PROMPT: &str = "<|im_start|>assistant\n";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        self.shared().as_str()
    }

    fn shared(self) -> shared::Role {
        match self {
            Role::System => shared::Role::System,
            Role::User => shared::Role::User,
            Role::Assistant => shared::Role::Assistant,
            Role::Tool => shared::Role::Tool,
        }
    }

    /// OpenAI role names; `developer` is the newer name of `system`.
    pub fn parse(name: &str) -> Option<Role> {
        match name {
            "system" | "developer" => Some(Role::System),
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            "tool" => Some(Role::Tool),
            _ => None,
        }
    }
}

/// A tool call as the OpenAI wire carries it: `arguments` is JSON text.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self { role, content: content.into(), reasoning: None, tool_calls: Vec::new() }
    }
}

/// A function the model may call.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Makes text safe to encode: a reserved tag written in message content must not become the
/// control token (`<|im_end|>` in a user message would end the turn). This is the shared rule
/// of `ncm_model::chat::defuse`, the same one the data pipeline applies to documents.
pub fn defuse(text: &str) -> String {
    shared::defuse(text).into_owned()
}

/// [`defuse`] every string inside a JSON value.
fn defuse_value(value: &Value) -> Value {
    match value {
        Value::String(s) => Value::String(defuse(s)),
        Value::Array(items) => Value::Array(items.iter().map(defuse_value).collect()),
        Value::Object(map) => {
            Value::Object(map.iter().map(|(k, v)| (k.clone(), defuse_value(v))).collect())
        }
        other => other.clone(),
    }
}

fn shared_tool(tool: &ToolDef) -> shared::ToolDef {
    shared::ToolDef {
        name: defuse(&tool.name),
        description: defuse(&tool.description),
        parameters: defuse_value(&tool.parameters),
    }
}

/// Arguments that parse as JSON are written compactly; anything else becomes a JSON string so
/// the line stays valid JSON.
fn shared_call(call: &ToolCall) -> shared::ToolCall {
    let arguments = serde_json::from_str::<Value>(&call.arguments)
        .unwrap_or_else(|_| Value::String(call.arguments.clone()));
    shared::ToolCall {
        id: call.id.clone(),
        name: defuse(&call.name),
        arguments: defuse_value(&arguments),
    }
}

fn shared_message(message: &Message) -> shared::Message {
    shared::Message {
        role: message.role.shared(),
        content: defuse(&message.content),
        reasoning: message.reasoning.as_deref().map(defuse),
        tool_calls: message.tool_calls.iter().map(shared_call).collect(),
        tool_call_id: None,
    }
}

/// The `<tools>` block.
pub fn tools_block(tools: &[ToolDef]) -> String {
    let tools: Vec<_> = tools.iter().map(shared_tool).collect();
    shared::render_tools_block(&tools)
}

/// Renders a conversation as NCM chat text, ending with the generation prompt when
/// `generation_prompt` is set. The text comes from the one shared renderer; this only maps the
/// wire types and defuses reserved tags in everything the client sent.
pub fn render(messages: &[Message], tools: &[ToolDef], generation_prompt: bool) -> String {
    let messages: Vec<_> = messages.iter().map(shared_message).collect();
    let tools: Vec<_> = tools.iter().map(shared_tool).collect();
    let mut text = shared::render(&messages, &tools).text;
    if generation_prompt {
        text.push_str(GENERATION_PROMPT);
    }
    text
}

// ---------------------------------------------------------------------------- output parsing

/// What the parser found in the generated tokens.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// Visible answer text.
    Content(String),
    /// Reasoning text from inside `<think>`.
    Reasoning(String),
    /// A complete tool call.
    ToolCall { name: String, arguments: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Content,
    Reasoning,
    ToolCall,
}

/// One text channel (answer or reasoning). Whitespace at the start and end of a segment is
/// formatting (`<think>\n...\n</think>\n`), so it is held back and dropped, and whitespace
/// inside is passed on once more text follows it.
#[derive(Debug, Default)]
struct Channel {
    started: bool,
    held: String,
}

impl Channel {
    fn push(&mut self, text: &str) -> Option<String> {
        let mut text = text;
        if !self.started {
            text = text.trim_start();
            if text.is_empty() {
                return None;
            }
            self.started = true;
        }
        let body = text.trim_end();
        if body.is_empty() {
            self.held.push_str(text);
            return None;
        }
        let out = format!("{}{body}", std::mem::take(&mut self.held));
        self.held = text[body.len()..].to_owned();
        Some(out)
    }

    /// Ends the segment, dropping the held trailing whitespace.
    fn end(&mut self) {
        self.started = false;
        self.held.clear();
    }
}

/// Splits generated tokens into reasoning, answer text and tool calls as they arrive.
///
/// With `parse_tools` off (a request without tools) `<tool_call>` is ordinary text. A
/// `<tool_call>` block whose JSON does not parse is returned as answer text so nothing the model
/// said is lost.
pub struct OutputParser<'a> {
    vocab: &'a Vocab,
    ids: Specials,
    parse_tools: bool,
    mode: Mode,
    utf8: Utf8Stream,
    channel: Channel,
    tool_text: String,
}

impl<'a> OutputParser<'a> {
    pub fn new(vocab: &'a Vocab, parse_tools: bool) -> Self {
        Self {
            vocab,
            ids: vocab.ids,
            parse_tools,
            mode: Mode::Content,
            utf8: Utf8Stream::default(),
            channel: Channel::default(),
            tool_text: String::new(),
        }
    }

    fn emit_text(&mut self, text: &str, out: &mut Vec<Event>) {
        match self.mode {
            Mode::ToolCall => self.tool_text.push_str(text),
            Mode::Reasoning => {
                out.extend(self.channel.push(text).map(Event::Reasoning));
            }
            Mode::Content => out.extend(self.channel.push(text).map(Event::Content)),
        }
    }

    fn flush_utf8(&mut self, out: &mut Vec<Event>) {
        let rest = self.utf8.finish();
        if !rest.is_empty() {
            self.emit_text(&rest, out);
        }
    }

    /// Feeds one generated token.
    pub fn push(&mut self, id: u32) -> Vec<Event> {
        let mut out = Vec::new();
        let ids = self.ids;
        if self.vocab.is_special(id) {
            self.flush_utf8(&mut out);
            if id == ids.think && self.mode == Mode::Content {
                self.channel.end();
                self.mode = Mode::Reasoning;
            } else if id == ids.think_end && self.mode == Mode::Reasoning {
                self.channel.end();
                self.mode = Mode::Content;
            } else if self.parse_tools && id == ids.tool_call && self.mode != Mode::ToolCall {
                self.channel.end();
                self.mode = Mode::ToolCall;
                self.tool_text.clear();
            } else if self.parse_tools && id == ids.tool_call_end && self.mode == Mode::ToolCall {
                self.mode = Mode::Content;
                let body = std::mem::take(&mut self.tool_text);
                match parse_tool_call(&body) {
                    Some((name, arguments)) => out.push(Event::ToolCall { name, arguments }),
                    None => out.extend(
                        self.channel
                            .push(&format!("<tool_call>{body}</tool_call>"))
                            .map(Event::Content),
                    ),
                }
            } else if id == ids.pad || id == ids.eoc || ids.is_stop(id) {
                // Layout tokens and the end of the reply carry no text.
            } else if let Some(text) = self.vocab.special_text(id) {
                self.emit_text(&text, &mut out);
            }
            return out;
        }
        let text = self.utf8.push(self.vocab.bytes(id));
        if !text.is_empty() {
            self.emit_text(&text, &mut out);
        }
        out
    }

    /// Ends the output. A tool call left open (the reply was cut off) is returned as text.
    pub fn finish(&mut self) -> Vec<Event> {
        let mut out = Vec::new();
        self.flush_utf8(&mut out);
        if self.mode == Mode::ToolCall {
            let body = std::mem::take(&mut self.tool_text);
            self.mode = Mode::Content;
            out.extend(self.channel.push(&format!("<tool_call>{body}")).map(Event::Content));
        }
        self.channel.end();
        out
    }
}

/// `{"name": ..., "arguments": ...}` to the name and the arguments as JSON text.
pub fn parse_tool_call(body: &str) -> Option<(String, String)> {
    let value: Value = serde_json::from_str(body.trim()).ok()?;
    let object = value.as_object()?;
    let name = object.get("name")?.as_str()?.to_owned();
    if name.is_empty() {
        return None;
    }
    let arguments = match object.get("arguments") {
        None | Some(Value::Null) => "{}".to_owned(),
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
    };
    Some((name, arguments))
}

/// A whole reply, collected from parser events.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reply {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<(String, String)>,
}

impl Reply {
    pub fn apply(&mut self, event: Event) {
        match event {
            Event::Content(text) => self.content.push_str(&text),
            Event::Reasoning(text) => self.reasoning.push_str(&text),
            Event::ToolCall { name, arguments } => self.tool_calls.push((name, arguments)),
        }
    }
}
