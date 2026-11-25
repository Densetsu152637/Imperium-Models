//! Constrained decoding: a mask over the next token, applied at every decode step.
//!
//! The hook for grammar-constrained tool calls (Imperium roadmap R2.4) is the [`Constraint`]
//! trait. The sampler asks `allows(token)` for the candidates it considers, best logit first, and
//! tells the constraint which token it took with `advance`. A constraint therefore never
//! materialises a vocabulary-sized mask: it only answers for the few candidates the sampler
//! actually tries.
//!
//! Implemented now: [`ToolCallGuard`]. Free text is unconstrained; once the model writes
//! `<tool_call>`, every token must keep the text a prefix of one JSON object, and `</tool_call>`
//! is allowed only once that object is complete. A tool call is therefore always syntactically
//! valid JSON.
//!
//! TODO (schema enforcement): constrain the object to `{"name": <one of the offered tools>,
//! "arguments": <object matching that tool's `parameters` schema>}`. It plugs in as another
//! [`Constraint`]: a [`JsonState`] extended with a schema cursor (expected key set, value types,
//! enums, required keys), so `allows` rejects a key or value the schema forbids and
//! `</tool_call>` waits for every required key. `tool_choice: "required"` and a named tool
//! belong there too (force `<tool_call>` as the first token, then the name).

use std::sync::Arc;

use crate::vocab::{Specials, Vocab};

/// A mask over the next token.
pub trait Constraint: Send {
    /// Whether `token` may come next.
    fn allows(&self, token: u32) -> bool;

    /// The sampler took `token`.
    fn advance(&mut self, token: u32);
}

/// Longest mantissa (digits before and after the point) the parser accepts.
const MAX_DIGITS: u16 = 100;
/// Longest exponent: `e99` is the largest, well inside a double.
const MAX_EXPONENT_DIGITS: u16 = 2;

/// Where the JSON parser is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Before the opening `{` (whitespace allowed).
    Start,
    /// After `{`: a key or `}`.
    ObjectFirst,
    /// After `,` in an object: a key.
    ObjectKey,
    /// After a key: `:`.
    Colon,
    /// A value is expected.
    Value,
    /// After `[`: a value or `]`.
    ArrayFirst,
    /// After a value: `,` or the closing bracket.
    After,
    /// Inside a string (`key` tells whether it is an object key).
    Str {
        key: bool,
    },
    /// After a backslash in a string.
    Escape {
        key: bool,
    },
    /// Inside `\uXXXX`, with this many hex digits still to come.
    Unicode {
        key: bool,
        left: u8,
    },
    /// A number: `-`, then digits.
    NumSign,
    NumZero,
    NumInt,
    NumDot,
    NumFrac,
    NumExpMark,
    NumExpSign,
    NumExp,
    /// `true`, `false` or `null`, with the rest still to come.
    Literal {
        rest: &'static [u8],
    },
    /// The object is closed.
    Done,
}

/// A byte-level JSON parser that accepts exactly the prefixes of one JSON object.
#[derive(Clone, Debug)]
pub struct JsonState {
    mode: Mode,
    /// `{` or `[` for each open container.
    stack: Vec<u8>,
    /// Digits of the number being written (its exponent, once there is one).
    digits: u16,
}

impl Default for JsonState {
    fn default() -> Self {
        Self::new()
    }
}

fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\n' | b'\r' | b'\t')
}

impl JsonState {
    pub fn new() -> Self {
        Self { mode: Mode::Start, stack: Vec::new(), digits: 0 }
    }

    /// The object is closed; only whitespace may follow.
    pub fn is_done(&self) -> bool {
        self.mode == Mode::Done
    }

    /// Whether nothing but whitespace has been seen (before `{`) or after the object.
    pub fn is_outside(&self) -> bool {
        matches!(self.mode, Mode::Start | Mode::Done)
    }

    /// Feeds one byte; false means it cannot continue any JSON object (the state is then
    /// unspecified).
    pub fn step(&mut self, byte: u8) -> bool {
        match self.mode {
            Mode::Start => match byte {
                b'{' => self.open(b'{'),
                b if is_space(b) => true,
                _ => false,
            },
            Mode::Done => is_space(byte),
            Mode::ObjectFirst => match byte {
                b'"' => self.go(Mode::Str { key: true }),
                b'}' => self.close(b'{'),
                b if is_space(b) => true,
                _ => false,
            },
            Mode::ObjectKey => match byte {
                b'"' => self.go(Mode::Str { key: true }),
                b if is_space(b) => true,
                _ => false,
            },
            Mode::Colon => match byte {
                b':' => self.go(Mode::Value),
                b if is_space(b) => true,
                _ => false,
            },
            Mode::ArrayFirst => match byte {
                b']' => self.close(b'['),
                _ => self.value(byte),
            },
            Mode::Value => self.value(byte),
            Mode::After => match (byte, self.stack.last()) {
                (b',', Some(b'{')) => self.go(Mode::ObjectKey),
                (b',', Some(b'[')) => self.go(Mode::Value),
                (b'}', Some(b'{')) => self.close(b'{'),
                (b']', Some(b'[')) => self.close(b'['),
                (b, _) if is_space(b) => true,
                _ => false,
            },
            Mode::Str { key } => match byte {
                b'"' => self.go(if key { Mode::Colon } else { Mode::After }),
                b'\\' => self.go(Mode::Escape { key }),
                b if b < 0x20 => false,
                _ => true,
            },
            Mode::Escape { key } => match byte {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                    self.go(Mode::Str { key })
                }
                b'u' => self.go(Mode::Unicode { key, left: 4 }),
                _ => false,
            },
            Mode::Unicode { key, left } => {
                if !byte.is_ascii_hexdigit() {
                    return false;
                }
                self.go(if left == 1 {
                    Mode::Str { key }
                } else {
                    Mode::Unicode { key, left: left - 1 }
                })
            }
            Mode::NumSign => match byte {
                b'0' => self.number(Mode::NumZero),
                b'1'..=b'9' => self.number(Mode::NumInt),
                _ => false,
            },
            Mode::NumZero => self.number_tail(byte, false),
            Mode::NumInt => self.number_tail(byte, true),
            Mode::NumDot => match byte {
                b'0'..=b'9' => self.digit(MAX_DIGITS) && self.go(Mode::NumFrac),
                _ => false,
            },
            Mode::NumFrac => match byte {
                b'0'..=b'9' => self.digit(MAX_DIGITS),
                b'e' | b'E' => self.go(Mode::NumExpMark),
                _ => self.end_value(byte),
            },
            Mode::NumExpMark => match byte {
                b'+' | b'-' => self.go(Mode::NumExpSign),
                b'0'..=b'9' => self.number(Mode::NumExp),
                _ => false,
            },
            Mode::NumExpSign => match byte {
                b'0'..=b'9' => self.number(Mode::NumExp),
                _ => false,
            },
            Mode::NumExp => match byte {
                b'0'..=b'9' => self.digit(MAX_EXPONENT_DIGITS),
                _ => self.end_value(byte),
            },
            Mode::Literal { rest } => {
                if rest.first() != Some(&byte) {
                    return false;
                }
                if rest.len() == 1 {
                    self.go(Mode::After)
                } else {
                    self.go(Mode::Literal { rest: &rest[1..] })
                }
            }
        }
    }

    /// A number (or its exponent) starts with a first digit.
    fn number(&mut self, mode: Mode) -> bool {
        self.digits = 1;
        self.go(mode)
    }

    /// One more digit, unless the number is already as long as allowed: a longer one could
    /// overflow a double, and then no JSON parser reads it back.
    fn digit(&mut self, limit: u16) -> bool {
        if self.digits >= limit {
            return false;
        }
        self.digits += 1;
        true
    }

    fn go(&mut self, mode: Mode) -> bool {
        self.mode = mode;
        true
    }

    fn open(&mut self, bracket: u8) -> bool {
        self.stack.push(bracket);
        self.mode = if bracket == b'{' { Mode::ObjectFirst } else { Mode::ArrayFirst };
        true
    }

    fn close(&mut self, bracket: u8) -> bool {
        if self.stack.last() != Some(&bracket) {
            return false;
        }
        self.stack.pop();
        self.mode = if self.stack.is_empty() { Mode::Done } else { Mode::After };
        true
    }

    /// A value starts with this byte.
    fn value(&mut self, byte: u8) -> bool {
        match byte {
            b'{' => self.open(b'{'),
            b'[' => self.open(b'['),
            b'"' => self.go(Mode::Str { key: false }),
            b'-' => {
                self.digits = 0;
                self.go(Mode::NumSign)
            }
            b'0' => self.number(Mode::NumZero),
            b'1'..=b'9' => self.number(Mode::NumInt),
            b't' => self.go(Mode::Literal { rest: b"rue" }),
            b'f' => self.go(Mode::Literal { rest: b"alse" }),
            b'n' => self.go(Mode::Literal { rest: b"ull" }),
            b if is_space(b) => true,
            _ => false,
        }
    }

    /// After the integer part: a fraction, an exponent, or the end of the number.
    fn number_tail(&mut self, byte: u8, more_digits: bool) -> bool {
        match byte {
            b'0'..=b'9' if more_digits => self.digit(MAX_DIGITS),
            b'.' => self.go(Mode::NumDot),
            b'e' | b'E' => self.go(Mode::NumExpMark),
            _ => self.end_value(byte),
        }
    }

    /// The number ended; `byte` is the first byte after it.
    fn end_value(&mut self, byte: u8) -> bool {
        self.mode = Mode::After;
        self.step(byte)
    }

    /// Whether `bytes` keep the text a prefix of a JSON object.
    pub fn accepts_bytes(&self, bytes: &[u8]) -> bool {
        let mut state = self.clone();
        bytes.iter().all(|&b| state.step(b))
    }
}

/// Where the guard is in the reply.
#[derive(Clone, Debug)]
enum Phase {
    /// Free text.
    Free,
    /// Between `<tool_call>` and `</tool_call>`, with the JSON so far and the whitespace bytes
    /// written outside the object (bounded, so the model cannot stall on newlines).
    Json { state: JsonState, outside_space: usize },
}

/// Forces a syntactically valid JSON object inside every `<tool_call>` block.
pub struct ToolCallGuard {
    vocab: Arc<Vocab>,
    ids: Specials,
    phase: Phase,
}

/// Whitespace bytes allowed before the `{` and after the closing `}`.
const MAX_OUTSIDE_SPACE: usize = 2;

impl ToolCallGuard {
    pub fn new(vocab: Arc<Vocab>) -> Self {
        let ids = vocab.ids;
        Self { vocab, ids, phase: Phase::Free }
    }

    /// A guard that starts inside a `<tool_call>` block (the open tag is already written).
    pub fn inside_call(vocab: Arc<Vocab>) -> Self {
        let mut guard = Self::new(vocab);
        guard.phase = Phase::Json { state: JsonState::new(), outside_space: 0 };
        guard
    }

    /// Whether the guard is inside a tool call.
    pub fn in_call(&self) -> bool {
        matches!(self.phase, Phase::Json { .. })
    }
}

impl Constraint for ToolCallGuard {
    fn allows(&self, token: u32) -> bool {
        let Phase::Json { state, outside_space } = &self.phase else { return true };
        if token == self.ids.tool_call_end {
            return state.is_done();
        }
        let bytes = self.vocab.bytes(token);
        // Special tokens (and ids the tokenizer lacks) carry no bytes: not valid inside JSON.
        if bytes.is_empty() {
            return false;
        }
        let mut next = state.clone();
        let mut space = *outside_space;
        for &byte in bytes {
            if next.is_outside() && is_space(byte) {
                space += 1;
                if space > MAX_OUTSIDE_SPACE {
                    return false;
                }
            }
            if !next.step(byte) {
                return false;
            }
        }
        true
    }

    fn advance(&mut self, token: u32) {
        match &mut self.phase {
            Phase::Free => {
                if token == self.ids.tool_call {
                    self.phase = Phase::Json { state: JsonState::new(), outside_space: 0 };
                }
            }
            Phase::Json { state, outside_space } => {
                if token == self.ids.tool_call_end {
                    self.phase = Phase::Free;
                    return;
                }
                for &byte in self.vocab.bytes(token) {
                    if state.is_outside() && is_space(byte) {
                        *outside_space += 1;
                    }
                    state.step(byte);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(text: &str) -> Option<JsonState> {
        let mut state = JsonState::new();
        text.bytes().all(|b| state.step(b)).then_some(state)
    }

    #[test]
    fn accepts_complete_objects() {
        for text in [
            "{}",
            " {\"a\":1}\n",
            r#"{"name":"read_file","arguments":{"path":"a/b.txt","n":-12.5e+3,"ok":true,"x":null}}"#,
            r#"{"a":[1,2,[3,{"b":"\u00e9\n\"q\""}],false],"c":0}"#,
            r#"{"a":0.5}"#,
        ] {
            let state = feed(text).unwrap_or_else(|| panic!("rejected {text}"));
            assert!(state.is_done(), "{text}");
            assert!(serde_json::from_str::<serde_json::Value>(text).is_ok(), "{text}");
        }
    }

    #[test]
    fn rejects_what_is_not_json() {
        for text in [
            "[1]",
            "{1:2}",
            "{\"a\" 1}",
            "{\"a\":01}",
            "{\"a\":1,}",
            "{\"a\":tru}",
            "{\"a\":\"\u{1}\"}",
            "{\"a\":\"\\x\"}",
            "{\"a\":1}}",
            "{\"a\":-}",
            "{\"a\":1.}",
            "{\"a\":1e}",
            "{\"a\":[1,]}",
        ] {
            assert!(!feed(text).is_some_and(|state| state.is_done()), "accepted {text}");
        }
        assert!(feed("{\"a\":1,}").is_none());
        assert!(feed("[1]").is_none());
    }

    #[test]
    fn numbers_end_on_the_next_structural_byte() {
        let mut state = feed("{\"a\":12").unwrap();
        assert_eq!(state.mode, Mode::NumInt);
        assert!(!state.is_done());
        assert!(state.step(b'}'));
        assert!(state.is_done());
    }

    /// Random walks over a small token set: whenever the walk reaches a closed object, it is
    /// valid JSON, and no step the state accepted ever produced something serde rejects.
    #[test]
    fn random_accepted_walks_are_valid_json() {
        let pieces: Vec<&str> = vec![
            "{", "}", "[", "]", ",", ":", "\"a\"", "\"b c\"", "\"", "\\", "n", "1", "-", "0.5",
            "e", "true", "null", " ", "\n", "\"name\"", "\\\"", "u00e9", "x",
        ];
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut closed = 0;
        for _ in 0..4000 {
            let mut state = JsonState::new();
            let mut text = String::new();
            for _ in 0..40 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let start = (seed % pieces.len() as u64) as usize;
                // The first piece that the state accepts, from a random start.
                let Some(piece) = (0..pieces.len())
                    .map(|i| pieces[(start + i) % pieces.len()])
                    .find(|p| state.accepts_bytes(p.as_bytes()))
                else {
                    break;
                };
                for b in piece.bytes() {
                    assert!(state.step(b));
                }
                text.push_str(piece);
                if state.is_done() {
                    closed += 1;
                    serde_json::from_str::<serde_json::Value>(&text)
                        .unwrap_or_else(|e| panic!("{text:?}: {e}"));
                    break;
                }
            }
        }
        assert!(closed > 200, "only {closed} walks closed");
    }
}
