//! A small byte-level tokenizer with the NCM special tokens, for tests and for
//! `--random-weights` runs that name no tokenizer file.
//!
//! Ids 0 to 10 are the special tokens in `SPECIAL_TOKENS` order (as in the real tokenizer),
//! 11 to 15 are reserved, and 16 to 271 are the 256 bytes. There are no merges, so every byte
//! is one token and any text round-trips.

use serde_json::{Map, Value, json};
use tokenizers::Tokenizer;

use crate::vocab::SPECIAL_TOKENS;

/// Ids the built-in tokenizer defines.
pub const BYTE_TOKENIZER_SIZE: usize = 16 + 256;

/// The GPT-2 byte-to-character table (printable bytes map to themselves).
fn byte_chars() -> Vec<char> {
    let printable = |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174;
    let mut next = 256u32;
    (0u32..256)
        .map(|byte| {
            if printable(byte) {
                char::from_u32(byte).expect("scalar")
            } else {
                next += 1;
                char::from_u32(next - 1).expect("scalar")
            }
        })
        .collect()
}

/// The tokenizer as a Hugging Face `tokenizer.json` document.
pub fn byte_tokenizer_json() -> Value {
    let mut vocab = Map::new();
    let mut added = Vec::new();
    for (id, name) in SPECIAL_TOKENS.iter().enumerate() {
        vocab.insert((*name).to_owned(), json!(id));
        added.push(json!({
            "id": id, "content": name, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true,
        }));
    }
    for id in SPECIAL_TOKENS.len()..16 {
        vocab.insert(format!("<|reserved_{}|>", id - SPECIAL_TOKENS.len()), json!(id));
    }
    for (byte, ch) in byte_chars().into_iter().enumerate() {
        vocab.insert(ch.to_string(), json!(16 + byte));
    }
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {
            "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true,
            "use_regex": false,
        },
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true,
            "use_regex": true,
        },
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null,
            "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false,
            "byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": [],
        },
    })
}

pub fn byte_tokenizer() -> Tokenizer {
    byte_tokenizer_json().to_string().parse().expect("the built-in tokenizer document is valid")
}
