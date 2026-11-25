//! Shared fixtures: a tiny deterministic corpus and helpers to write fixture files.

#![allow(dead_code)]

use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use flate2::Compression;
use flate2::write::GzEncoder;
use parquet::basic::Compression as PqCompression;
use parquet::data_type::{ByteArray, ByteArrayType, Int64Type};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;

/// Deterministic text with enough repeated structure to learn a few hundred merges.
pub fn corpus(docs: usize) -> Vec<String> {
    const WORDS: [&str; 24] = [
        "the",
        "model",
        "token",
        "reads",
        "concept",
        "of",
        "four",
        "tokens",
        "and",
        "then",
        "writes",
        "an",
        "answer",
        "from",
        "memory",
        "using",
        "attention",
        "over",
        "context",
        "while",
        "training",
        "on",
        "open",
        "data",
    ];
    const SYLLABLES: [&str; 32] = [
        "ka", "to", "mi", "ra", "ne", "so", "lu", "pe", "vi", "da", "go", "he", "ri", "mo", "zu",
        "fa", "ten", "ron", "kal", "mer", "dis", "pro", "con", "sta", "ing", "tion", "ver", "ous",
        "qua", "bel", "nor", "ex",
    ];
    const CODE: [&str; 4] = [
        "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n",
        "def total(xs):\n    return sum(x * 2 for x in xs)\n",
        "for (int i = 0; i < 10; i++) {\n    printf(\"%d\\n\", i);\n}\n",
        "$$ \\int_0^1 x^2 \\, dx = \\frac{1}{3} $$\n",
    ];
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..docs)
        .map(|i| {
            let mut doc = String::new();
            for _ in 0..(8 + next() % 24) {
                doc.push_str(WORDS[(next() % WORDS.len() as u64) as usize]);
                doc.push(' ');
            }
            for _ in 0..10 {
                for _ in 0..(2 + next() % 2) {
                    doc.push_str(SYLLABLES[(next() % SYLLABLES.len() as u64) as usize]);
                }
                doc.push(' ');
            }
            doc.push_str(&format!("\n\nEntry {} costs {}.\n", i, next() % 10_000));
            doc.push_str(CODE[(next() % CODE.len() as u64) as usize]);
            doc
        })
        .collect()
}

pub fn write_jsonl(path: &Path, rows: &[serde_json::Value]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut text = String::new();
    for r in rows {
        text.push_str(&r.to_string());
        text.push('\n');
    }
    std::fs::write(path, text).unwrap();
}

pub fn gzip(bytes: &[u8]) -> Vec<u8> {
    let mut enc = GzEncoder::new(Vec::new(), Compression::fast());
    enc.write_all(bytes).unwrap();
    enc.finish().unwrap()
}

/// A Parquet file with a string column `text` and an integer column `n`.
pub fn write_parquet(path: &Path, texts: &[&str]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let schema = Arc::new(
        parse_message_type("message m { REQUIRED BYTE_ARRAY text (UTF8); REQUIRED INT64 n; }")
            .unwrap(),
    );
    let props =
        Arc::new(WriterProperties::builder().set_compression(PqCompression::SNAPPY).build());
    let file = std::fs::File::create(path).unwrap();
    let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
    let mut group = writer.next_row_group().unwrap();
    let values: Vec<ByteArray> = texts.iter().map(|t| ByteArray::from(*t)).collect();
    let mut col = group.next_column().unwrap().unwrap();
    col.typed::<ByteArrayType>().write_batch(&values, None, None).unwrap();
    col.close().unwrap();
    let nums: Vec<i64> = (0..texts.len() as i64).collect();
    let mut col = group.next_column().unwrap().unwrap();
    col.typed::<Int64Type>().write_batch(&nums, None, None).unwrap();
    col.close().unwrap();
    group.close().unwrap();
    writer.close().unwrap();
}

/// A manifest for fixture sources `a` (text) and `b` (trace, anneal) with the given shares.
pub fn mix_text(a_share: f64, b_share: f64) -> String {
    format!(
        r#"
budget_tokens = 100000
val_fraction = 0.1

[anneal]
fraction = 0.5
[anneal.shares]
a = 0.5
b = 0.5

[[source]]
name = "a"
category = "web"
phase = "main"
repo = "x/a"
revision = "0123456789abcdef0123456789abcdef01234567"
license = "MIT"
gated = false
share = {a_share}
kind = "text"
glob = "*.jsonl"
claude_check = "fixture"

[[source]]
name = "b"
category = "reasoning"
phase = "anneal"
repo = "x/b"
revision = "0123456789abcdef0123456789abcdef01234567"
license = "Apache-2.0"
gated = false
share = {b_share}
kind = "trace"
glob = "*.jsonl"
prompt_field = "q"
response_field = "r"
claude_check = "fixture"
"#
    )
}

/// One main-phase source `s` with the given extra TOML lines (kind, fields, filters).
pub fn single_source(extra: &str) -> ncm_data::manifest::Source {
    let glob = if extra.contains("glob =") {
        ""
    } else {
        "glob = \"*\"
"
    };
    let text = format!(
        r#"
budget_tokens = 1000
val_fraction = 0.1
[anneal]
fraction = 0.5
[anneal.shares]
s = 1.0

[[source]]
name = "s"
category = "web"
phase = "main"
repo = "x/s"
revision = "0123456789abcdef0123456789abcdef01234567"
license = "MIT"
gated = false
share = 1.0
{glob}claude_check = "fixture"
{extra}
"#
    );
    ncm_data::manifest::Mix::parse(&text).unwrap().sources.remove(0)
}
