//! The shared byte-level BPE tokenizer (token baseline and NCM use the same one).
//!
//! Every byte is in the base alphabet, so any text encodes and decodes losslessly (byte
//! fallback). Digits split one per token, which keeps arithmetic and code numerals regular. The
//! special tokens take the first ids so they never move when the vocabulary is retrained.

use std::path::Path;

use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::TrainerWrapper;
use tokenizers::models::bpe::{BPE, BpeTrainer};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, PreTokenizerWrapper, SplitDelimiterBehavior, Tokenizer};

use crate::error::{Error, Result};

/// Reserved ids, in id order. Ids 0 to 10 are the ones the models and chat format use; the five
/// reserved tokens leave room for new markers without retraining.
pub const SPECIAL_TOKENS: [&str; 16] = [
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
    "<|reserved_0|>",
    "<|reserved_1|>",
    "<|reserved_2|>",
    "<|reserved_3|>",
    "<|reserved_4|>",
];

pub const EOT_ID: u32 = 0;
pub const PAD_ID: u32 = 1;
pub const IM_START_ID: u32 = 2;
pub const IM_END_ID: u32 = 3;
pub const THINK_ID: u32 = 8;
pub const THINK_END_ID: u32 = 9;
/// The NCM early concept end.
pub const EOC_ID: u32 = 10;

/// Pre-tokenisation regex: English contractions, words with one leading non-letter, single
/// digits, runs of punctuation, and newlines kept with the whitespace around them.
const SPLIT_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn tok_err(e: impl std::fmt::Display) -> Error {
    Error::Tokenizer(e.to_string())
}

fn empty_tokenizer() -> Result<Tokenizer> {
    let mut tok = Tokenizer::new(BPE::default());
    let split = Split::new(
        SplitPattern::Regex(SPLIT_PATTERN.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(tok_err)?;
    tok.with_pre_tokenizer(Some(Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false)),
    ])));
    tok.with_decoder(Some(ByteLevelDecoder::default()));
    Ok(tok)
}

/// Train on `docs` (each a whole document, newlines intact) to exactly `vocab_size` ids.
pub fn train<I>(docs: I, vocab_size: usize) -> Result<Tokenizer>
where
    I: Iterator<Item = String> + Send,
{
    let mut tok = empty_tokenizer()?;
    let specials: Vec<AddedToken> =
        SPECIAL_TOKENS.iter().map(|s| AddedToken::from(*s, true)).collect();
    let mut trainer: TrainerWrapper = BpeTrainer::builder()
        .vocab_size(vocab_size)
        .min_frequency(2)
        .show_progress(false)
        .special_tokens(specials)
        .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
        .build()
        .into();
    tok.train(&mut trainer, docs).map_err(tok_err)?;
    let got = tok.get_vocab_size(true);
    if got != vocab_size {
        return Err(Error::Tokenizer(format!(
            "trained {got} ids, wanted {vocab_size}: the sample is too small or too repetitive"
        )));
    }
    check_special_ids(&tok)?;
    Ok(tok)
}

/// The special tokens must sit at their fixed ids.
pub fn check_special_ids(tok: &Tokenizer) -> Result<()> {
    for (id, name) in SPECIAL_TOKENS.iter().enumerate() {
        if tok.token_to_id(name) != Some(id as u32) {
            return Err(Error::Tokenizer(format!(
                "{name} is not at id {id} (found {:?})",
                tok.token_to_id(name)
            )));
        }
    }
    Ok(())
}

pub fn load(path: &Path) -> Result<Tokenizer> {
    let tok = Tokenizer::from_file(path).map_err(tok_err)?;
    check_special_ids(&tok)?;
    Ok(tok)
}

pub fn save(tok: &Tokenizer, path: &Path) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|source| Error::Io {
            context: format!("creating {}", dir.display()),
            source,
        })?;
    }
    tok.save(path, true).map_err(tok_err)
}

/// Encode each text and append the end-of-text id, as stored in the shards.
pub fn encode_documents(tok: &Tokenizer, texts: Vec<String>) -> Result<Vec<Vec<u16>>> {
    let vocab = tok.get_vocab_size(true);
    if vocab > u16::MAX as usize + 1 {
        return Err(Error::Tokenizer(format!("vocabulary of {vocab} ids does not fit u16 shards")));
    }
    let encodings = tok.encode_batch(texts, false).map_err(tok_err)?;
    Ok(encodings
        .iter()
        .map(|e| {
            let mut ids: Vec<u16> = e.get_ids().iter().map(|&i| i as u16).collect();
            ids.push(EOT_ID as u16);
            ids
        })
        .collect())
}
