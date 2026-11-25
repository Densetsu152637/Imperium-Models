//! The tokenizer as the runtime uses it: encoding, the raw bytes of every token (for streaming
//! detokenisation and for constraining output), and the ids of the chat special tokens.
//!
//! Tokenizers are Hugging Face `tokenizer.json` files. The stage-2 tokenizer is byte-level BPE;
//! SentencePiece-style tokenizers (`▁`, `<0xNN>` byte fallback) also work.

use std::collections::HashMap;
use std::path::Path;

use tokenizers::{DecoderWrapper, Tokenizer};

/// The reserved special tokens the chat format and the NCM use. Every one must exist in the
/// tokenizer; their ids are looked up by name, not assumed.
pub const SPECIAL_TOKENS: [&str; 11] = [
    "<|endoftext|>",
    "<|pad|>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<|eoc|>",
];

#[derive(Debug, thiserror::Error)]
pub enum VocabError {
    #[error("cannot load tokenizer {path}: {message}")]
    Load { path: String, message: String },
    #[error("the tokenizer has no `{0}` token")]
    MissingSpecial(&'static str),
    #[error("the tokenizer has {tokenizer} ids but the model's vocabulary is {model}")]
    TooLarge { tokenizer: usize, model: usize },
    #[error("encoding failed: {0}")]
    Encode(String),
}

/// Ids of the special tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Specials {
    pub end_of_text: u32,
    pub pad: u32,
    pub im_start: u32,
    pub im_end: u32,
    pub tool_call: u32,
    pub tool_call_end: u32,
    pub tool_response: u32,
    pub tool_response_end: u32,
    pub think: u32,
    pub think_end: u32,
    /// The NCM's early concept end.
    pub eoc: u32,
}

impl Specials {
    /// The tokens that end a reply.
    pub fn is_stop(&self, id: u32) -> bool {
        id == self.im_end || id == self.end_of_text || id == self.im_start
    }
}

pub struct Vocab {
    tokenizer: Tokenizer,
    /// Raw bytes of each id below the model's vocabulary; empty for special and unknown ids.
    bytes: Vec<Vec<u8>>,
    special: Vec<bool>,
    pub ids: Specials,
    /// Ids that exist in the tokenizer (the rest of the model vocabulary is unused).
    known: usize,
}

/// The GPT-2 byte-to-character table of byte-level BPE, inverted.
fn byte_level_inverse() -> HashMap<char, u8> {
    let printable = |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174;
    let mut next = 256u32;
    let mut map = HashMap::new();
    for byte in 0u32..256 {
        let ch = if printable(byte) {
            char::from_u32(byte)
        } else {
            next += 1;
            char::from_u32(next - 1)
        };
        map.insert(ch.expect("a valid scalar"), byte as u8);
    }
    map
}

/// `<0xNN>` byte-fallback tokens.
fn byte_fallback(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    (hex.len() == 2).then(|| u8::from_str_radix(hex, 16).ok()).flatten()
}

impl Vocab {
    pub fn from_file(path: &Path, model_vocab: usize) -> Result<Self, VocabError> {
        let tokenizer = Tokenizer::from_file(path).map_err(|e| VocabError::Load {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        Self::new(tokenizer, model_vocab)
    }

    pub fn new(tokenizer: Tokenizer, model_vocab: usize) -> Result<Self, VocabError> {
        let known = tokenizer.get_vocab_size(true);
        let max_id =
            tokenizer.get_vocab(true).values().copied().max().map_or(0, |id| id as usize + 1);
        if max_id > model_vocab {
            return Err(VocabError::TooLarge { tokenizer: max_id, model: model_vocab });
        }
        let id = |name: &'static str| {
            tokenizer.token_to_id(name).ok_or(VocabError::MissingSpecial(name))
        };
        let ids = Specials {
            end_of_text: id(SPECIAL_TOKENS[0])?,
            pad: id(SPECIAL_TOKENS[1])?,
            im_start: id(SPECIAL_TOKENS[2])?,
            im_end: id(SPECIAL_TOKENS[3])?,
            tool_call: id(SPECIAL_TOKENS[4])?,
            tool_call_end: id(SPECIAL_TOKENS[5])?,
            tool_response: id(SPECIAL_TOKENS[6])?,
            tool_response_end: id(SPECIAL_TOKENS[7])?,
            think: id(SPECIAL_TOKENS[8])?,
            think_end: id(SPECIAL_TOKENS[9])?,
            eoc: id(SPECIAL_TOKENS[10])?,
        };
        let byte_level = matches!(tokenizer.get_decoder(), Some(DecoderWrapper::ByteLevel(_)));
        let inverse = byte_level_inverse();
        let added = tokenizer.get_added_tokens_decoder();
        let mut bytes = vec![Vec::new(); model_vocab];
        let mut special = vec![false; model_vocab];
        for (token, id) in tokenizer.get_vocab(true) {
            let slot = id as usize;
            if added.get(&id).is_some_and(|t| t.special) {
                special[slot] = true;
            } else if byte_level {
                bytes[slot] = token.chars().filter_map(|c| inverse.get(&c).copied()).collect();
            } else if let Some(byte) = byte_fallback(&token) {
                bytes[slot] = vec![byte];
            } else {
                bytes[slot] = token.replace('\u{2581}', " ").into_bytes();
            }
        }
        Ok(Self { tokenizer, bytes, special, ids, known })
    }

    /// Size of the model vocabulary this table covers.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Number of ids the tokenizer defines.
    pub fn known(&self) -> usize {
        self.known
    }

    /// Encodes `text`. Special tokens written in the text become their single ids, so message
    /// content must be defused first (see `chat::defuse`).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, VocabError> {
        let encoding =
            self.tokenizer.encode(text, false).map_err(|e| VocabError::Encode(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// The bytes a token stands for; empty for special tokens and ids the tokenizer lacks.
    pub fn bytes(&self, id: u32) -> &[u8] {
        self.bytes.get(id as usize).map_or(&[], Vec::as_slice)
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(false)
    }

    /// Whether `id` is an ordinary token that stands for at least one byte.
    pub fn is_text(&self, id: u32) -> bool {
        !self.bytes(id).is_empty()
    }

    /// A special token's literal text (`<tool_call>`), for output that is shown as plain text.
    pub fn special_text(&self, id: u32) -> Option<String> {
        self.is_special(id).then(|| self.tokenizer.id_to_token(id)).flatten()
    }

    /// The text of `ids`: ordinary tokens decoded, special tokens written out.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut raw = Vec::new();
        for &id in ids {
            if let Some(text) = self.special_text(id) {
                raw.extend_from_slice(text.as_bytes());
            } else {
                raw.extend_from_slice(self.bytes(id));
            }
        }
        String::from_utf8_lossy(&raw).into_owned()
    }
}

/// Turns a stream of token bytes into text, holding back an incomplete UTF-8 sequence until the
/// next token completes it.
#[derive(Debug, Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    /// Adds bytes and returns the text that is now complete.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            // Invalid bytes in the middle: replace them and carry on.
            Err(_) => return self.take_lossy(),
        };
        let text = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
        self.pending.drain(..valid);
        text
    }

    fn take_lossy(&mut self) -> String {
        let text = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        text
    }

    /// Whatever is left, with an incomplete sequence replaced.
    pub fn finish(&mut self) -> String {
        self.take_lossy()
    }
}
