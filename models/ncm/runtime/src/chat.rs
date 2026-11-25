//! The NCM chat format: rendering conversations to the text the models were fine-tuned on, and
//! parsing generated output back into reasoning, content and tool calls.
//!
//! The format mirrors `ncm-sft`'s `format.rs` exactly (stage 3 defines it):
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

use serde_json::Value;

use crate::vocab::{SPECIAL_TOKENS, Specials, Utf8Stream, Vocab};

pub const GENERATION_PROMPT: &str = "<|im_start|>assistant\n";
const TOOLS_OPEN: &str = "<tools>";
const TOOLS_CLOSE: &str = "</tools>";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// Every tag that is a reserved token, plus the tool block tags.
fn reserved() -> impl Iterator<Item = &'static str> {
    SPECIAL_TOKENS.into_iter().chain([TOOLS_OPEN, TOOLS_CLOSE])
}

/// Makes text safe to encode: a reserved tag written in message content must not become the
/// control token (`<|im_end|>` in a user message would end the turn). One space after the `<`
/// defuses it, as the data pipeline does for documents.
pub fn defuse(text: &str) -> String {
    if !text.contains('<') {
        return text.to_owned();
    }
    let mut out = text.to_owned();
    for tag in reserved() {
        if out.contains(tag) {
            out = out.replace(tag, &format!("< {}", &tag[1..]));
        }
    }
    out
}

fn json_string(text: &str) -> String {
    Value::String(text.to_owned()).to_string()
}

/// One compact definition line, in the key order `name`, `description`, `parameters`.
fn tool_line(tool: &ToolDef) -> String {
    format!(
        "{{\"name\":{},\"description\":{},\"parameters\":{}}}",
        json_string(&tool.name),
        json_string(&tool.description),
        tool.parameters
    )
}

/// The `<tools>` block.
pub fn tools_block(tools: &[ToolDef]) -> String {
    let mut block = String::from(TOOLS_OPEN);
    block.push('\n');
    for tool in tools {
        block.push_str(&defuse(&tool_line(tool)));
        block.push('\n');
    }
    block.push_str(TOOLS_CLOSE);
    block
}

/// A tool call in the form the model writes it. `arguments` that parse as JSON are written
/// compactly; anything else is written as a JSON string so the line stays valid JSON.
pub fn tool_call_text(call: &ToolCall) -> String {
    let arguments = match serde_json::from_str::<Value>(&call.arguments) {
        Ok(value) => value.to_string(),
        Err(_) => json_string(&call.arguments),
    };
    format!(
        "<tool_call>\n{{\"name\":{},\"arguments\":{}}}\n</tool_call>",
        json_string(&call.name),
        defuse(&arguments)
    )
}

fn push_turn(text: &mut String, role: &str, body: &str) {
    text.push_str("<|im_start|>");
    text.push_str(role);
    text.push('\n');
    text.push_str(body);
    text.push_str("<|im_end|>\n");
}

fn assistant_body(message: &Message) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(reasoning) = message.reasoning.as_deref().map(str::trim).filter(|r| !r.is_empty()) {
        parts.push(format!("<think>\n{}\n</think>", defuse(reasoning)));
    }
    let content = message.content.trim();
    if !content.is_empty() {
        parts.push(defuse(content));
    }
    parts.extend(message.tool_calls.iter().map(tool_call_text));
    let mut body = parts.join("\n");
    body.push('\n');
    body
}

/// Renders a conversation as NCM chat text, ending with the generation prompt when
/// `generation_prompt` is set. A leading system message and the tool definitions share the
/// system turn.
pub fn render(messages: &[Message], tools: &[ToolDef], generation_prompt: bool) -> String {
    let mut text = String::new();
    let system = messages.first().filter(|m| m.role == Role::System);
    let block = (!tools.is_empty()).then(|| tools_block(tools));
    let system_body = match (system, &block) {
        (Some(s), Some(b)) => Some(format!("{}\n\n{b}\n", defuse(s.content.trim()))),
        (Some(s), None) => Some(format!("{}\n", defuse(s.content.trim()))),
        (None, Some(b)) => Some(format!("{b}\n")),
        (None, None) => None,
    };
    if let Some(body) = system_body {
        push_turn(&mut text, "system", &body);
    }
    for message in messages.iter().skip(usize::from(system.is_some())) {
        match message.role {
            Role::System | Role::User => {
                let body = format!("{}\n", defuse(message.content.trim()));
                push_turn(&mut text, message.role.as_str(), &body);
            }
            Role::Tool => {
                let body = format!(
                    "<tool_response>\n{}\n</tool_response>\n",
                    defuse(message.content.trim())
                );
                push_turn(&mut text, "tool", &body);
            }
            Role::Assistant => push_turn(&mut text, "assistant", &assistant_body(message)),
        }
    }
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
