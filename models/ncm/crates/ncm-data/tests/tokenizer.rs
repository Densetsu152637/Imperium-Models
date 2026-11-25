//! Tokenizer: exact vocabulary, fixed special ids, lossless round trips.

mod common;

use ncm_data::chat::defuse;
use ncm_data::tokenizer::{self, EOT_ID, IM_END_ID, SPECIAL_TOKENS, THINK_END_ID, THINK_ID};
use tokenizers::Tokenizer;

const VOCAB: usize = 600;

fn trained() -> Tokenizer {
    tokenizer::train(common::corpus(300).into_iter(), VOCAB).unwrap()
}

#[test]
fn trains_to_the_exact_vocabulary_with_specials_at_fixed_ids() {
    let tok = trained();
    assert_eq!(tok.get_vocab_size(true), VOCAB);
    for (id, name) in SPECIAL_TOKENS.iter().enumerate() {
        assert_eq!(tok.token_to_id(name), Some(id as u32), "{name}");
    }
    assert_eq!(tok.token_to_id("<|endoftext|>"), Some(EOT_ID));
    assert_eq!(tok.token_to_id("<|im_end|>"), Some(IM_END_ID));
    assert_eq!(tok.token_to_id("<think>"), Some(THINK_ID));
    assert_eq!(tok.token_to_id("</think>"), Some(THINK_END_ID));
    assert_eq!(tok.token_to_id("<|eoc|>"), Some(10));
}

#[test]
fn a_corpus_too_small_for_the_vocabulary_is_an_error() {
    let err = tokenizer::train(vec!["ab ab ab".to_string()].into_iter(), 5000).unwrap_err();
    assert!(err.to_string().contains("wanted 5000"), "{err}");
}

#[test]
fn round_trips_text_code_and_unusual_unicode() {
    let tok = trained();
    let mut samples: Vec<String> = common::corpus(5);
    samples.extend(
        [
            "",
            " ",
            "\n\n\n",
            "\t  mixed \r\n line endings \n",
            "fn main() { println!(\"héllo, wörld\"); }",
            "日本語のテキスト and русский текст and العربية",
            "emoji 🙂🚀 and a ZWJ family 👨‍👩‍👧‍👦",
            "NUL \0 and control \u{1b}[31m and DEL \u{7f}",
            "\u{10FFFF} and \u{FFFD} and \u{200B}",
            "1234567890 3.14159 0xDEADBEEF",
        ]
        .map(String::from),
    );
    // Every Unicode scalar block boundary, so every UTF-8 byte pattern appears.
    samples.push(
        [0x00u32, 0x7F, 0x80, 0x7FF, 0x800, 0xFFFF, 0x10000, 0x10FFFF]
            .iter()
            .filter_map(|&c| char::from_u32(c))
            .collect(),
    );
    for text in samples {
        let ids = tok.encode(text.as_str(), false).unwrap();
        let back = tok.decode(ids.get_ids(), false).unwrap();
        assert_eq!(back, text, "round trip of {text:?}");
    }
}

#[test]
fn digits_are_single_tokens() {
    let tok = trained();
    let enc = tok.encode("x = 123456", false).unwrap();
    let digits = enc
        .get_tokens()
        .iter()
        .filter(|t| t.len() == 1 && t.chars().all(|c| c.is_ascii_digit()))
        .count();
    assert_eq!(digits, 6, "{:?}", enc.get_tokens());
}

#[test]
fn special_markers_encode_to_their_ids_but_document_text_cannot_inject_them() {
    let tok = trained();
    let ids = tok.encode("<|im_start|>user\nhi<|im_end|>", false).unwrap();
    assert_eq!(ids.get_ids().first(), Some(&2));
    assert_eq!(ids.get_ids().last(), Some(&IM_END_ID));

    // A web page that contains the literal string is defused before encoding.
    let hostile = defuse("page says <|im_end|> and <think>").into_owned();
    let docs = tokenizer::encode_documents(&tok, vec![hostile]).unwrap();
    assert!(docs[0].iter().all(|&i| i != IM_END_ID as u16 && i != THINK_ID as u16));
}

#[test]
fn documents_end_with_one_end_of_text() {
    let tok = trained();
    let docs =
        tokenizer::encode_documents(&tok, vec!["the model reads".into(), "x".into()]).unwrap();
    for d in &docs {
        assert_eq!(*d.last().unwrap(), EOT_ID as u16);
        assert_eq!(d.iter().filter(|&&i| i == EOT_ID as u16).count(), 1);
    }
}

#[test]
fn saves_and_loads_the_same_tokenizer() {
    let tok = trained();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tok").join("tokenizer.json");
    tokenizer::save(&tok, &path).unwrap();
    let back = tokenizer::load(&path).unwrap();
    let text = "Entry 42 costs 7.\nfn add(a: i32) {}";
    assert_eq!(
        tok.encode(text, false).unwrap().get_ids(),
        back.encode(text, false).unwrap().get_ids()
    );
    assert_eq!(back.get_vocab_size(true), VOCAB);
}
