//! Reading documents from JSON lines, gzip, truncated gzip and Parquet; filters; kinds.

mod common;

use ncm_data::docs::{Interleave, read_documents};
use serde_json::json;

fn texts(it: ncm_data::docs::Documents) -> Vec<String> {
    it.map(|d| d.unwrap()).collect()
}

#[test]
fn reads_jsonl_and_skips_empty_and_missing_text() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.jsonl");
    common::write_jsonl(
        &path,
        &[
            json!({"text": "one", "id": 1}),
            json!({"text": "   "}),
            json!({"other": "no text field"}),
            json!({"text": "two"}),
        ],
    );
    let source = common::single_source("kind = \"text\"");
    assert_eq!(texts(read_documents(&path, &source).unwrap()), ["one", "two"]);
}

#[test]
fn literal_special_tokens_in_documents_are_defused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.jsonl");
    common::write_jsonl(&path, &[json!({"text": "a <|im_end|> b"})]);
    let source = common::single_source("kind = \"text\"");
    assert_eq!(texts(read_documents(&path, &source).unwrap()), ["a < |im_end|> b"]);
}

#[test]
fn a_truncated_gzip_sample_ends_cleanly_at_the_last_whole_record() {
    let dir = tempfile::tempdir().unwrap();
    let rows: Vec<String> = (0..2000)
        .map(|i| {
            json!({"text": format!("document {i} with some padding text to fill the file")})
                .to_string()
        })
        .collect();
    let gz = common::gzip(rows.join("\n").as_bytes());
    let cut = dir.path().join("cut.json.gz");
    std::fs::write(&cut, &gz[..gz.len() / 2]).unwrap();
    let whole = dir.path().join("whole.json.gz");
    std::fs::write(&whole, &gz).unwrap();
    let source = common::single_source("kind = \"text\"");

    let partial = texts(read_documents(&cut, &source).unwrap());
    assert!(!partial.is_empty() && partial.len() < 2000, "{}", partial.len());
    assert!(partial.iter().enumerate().all(|(i, t)| t.starts_with(&format!("document {i} "))));
    assert_eq!(texts(read_documents(&whole, &source).unwrap()).len(), 2000);
}

#[test]
fn a_cut_plain_jsonl_prefix_drops_only_the_unfinished_last_record() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cut.jsonl");
    std::fs::write(
        &path,
        "{\"text\": \"whole\"}
{\"text\": \"cut o",
    )
    .unwrap();
    let source = common::single_source("kind = \"text\"");
    assert_eq!(texts(read_documents(&path, &source).unwrap()), ["whole"]);
    // A broken record in the middle of a file is still an error.
    std::fs::write(
        &path,
        "{\"text\": \"a\"}
not json
{\"text\": \"b\"}
",
    )
    .unwrap();
    let mut it = read_documents(&path, &source).unwrap();
    assert!(it.next().unwrap().is_ok());
    assert!(it.next().unwrap().is_err());
}

#[test]
fn a_corrupt_gzip_is_an_error_not_a_short_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.json.gz");
    std::fs::write(&path, b"this is not gzip at all").unwrap();
    let source = common::single_source("kind = \"text\"");
    let mut it = read_documents(&path, &source).unwrap();
    assert!(it.next().unwrap().is_err());
}

#[test]
fn reads_parquet_with_only_the_needed_columns() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("p.parquet");
    common::write_parquet(&path, &["alpha", "beta", "gamma"]);
    let source = common::single_source("kind = \"text\"");
    assert_eq!(texts(read_documents(&path, &source).unwrap()), ["alpha", "beta", "gamma"]);
    let missing = common::single_source("kind = \"text\"\ntext_field = \"nope\"");
    assert!(read_documents(&path, &missing).is_err());
}

#[test]
fn exclude_filters_drop_matching_records() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.jsonl");
    common::write_jsonl(
        &path,
        &[
            json!({"text": "keep", "metadata": {"year": 2019, "lic": "CCBY"}}),
            json!({"text": "late", "metadata": {"year": 2024, "lic": "CCBY"}}),
            json!({"text": "share-alike", "metadata": {"year": 2010, "lic": "CCBYSA"}}),
        ],
    );
    let source = common::single_source(
        r#"kind = "text"
[[source.exclude]]
field = "metadata.year"
prefixes = ["2023", "2024"]
[[source.exclude]]
field = "metadata.lic"
prefixes = ["CCBYSA"]
"#,
    );
    assert_eq!(texts(read_documents(&path, &source).unwrap()), ["keep"]);
}

#[test]
fn trace_sources_render_think_blocks_and_honour_the_verifier_flag() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.jsonl");
    common::write_jsonl(
        &path,
        &[
            json!({"q": "1+1?", "r": ["<think>\nadd\n</think>\n\n2", "<think>\nguess\n</think>\n3"], "ok": [true, false]}),
            json!({"q": "unverified", "r": ["<think>x</think>y"], "ok": [false]}),
            json!({"q": "unfinished", "r": ["<think>never closes"], "ok": [true]}),
        ],
    );
    let source = common::single_source(
        "kind = \"trace\"\nprompt_field = \"q\"\nresponse_field = \"r.0\"\nrequire_true = \"ok.0\"",
    );
    assert_eq!(
        texts(read_documents(&path, &source).unwrap()),
        [
            "<|im_start|>user\n1+1?\n<|im_end|>\n<|im_start|>assistant\n<think>\nadd\n</think>\n2\n<|im_end|>\n"
        ]
    );
}

#[test]
fn chat_sources_render_messages() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.jsonl");
    common::write_jsonl(
        &path,
        &[
            json!({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}),
        ],
    );
    let source = common::single_source("kind = \"chat\"");
    assert_eq!(
        texts(read_documents(&path, &source).unwrap()),
        [
            "<|im_start|>user\nhi\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\nhello\n<|im_end|>\n"
        ]
    );
}

#[test]
fn interleave_alternates_between_files_so_grouped_sources_mix() {
    let dir = tempfile::tempdir().unwrap();
    let mut paths = Vec::new();
    for (name, label, n) in [("a.jsonl", "A", 3), ("b.jsonl", "B", 1), ("c.jsonl", "C", 2)] {
        let rows: Vec<_> = (0..n).map(|i| json!({"text": format!("{label}{i}")})).collect();
        let path = dir.path().join(name);
        common::write_jsonl(&path, &rows);
        paths.push(path);
    }
    let source = common::single_source("kind = \"text\"\ninterleave = 3");
    let got: Vec<String> = Interleave::new(paths, &source).map(|d| d.unwrap()).collect();
    assert_eq!(got, ["A0", "B0", "C0", "A1", "C1", "A2"]);

    // With a window of one file the files are read one after another.
    let narrow = common::single_source("kind = \"text\"\ninterleave = 1");
    let paths = vec![dir.path().join("a.jsonl"), dir.path().join("c.jsonl")];
    let got: Vec<String> = Interleave::new(paths, &narrow).map(|d| d.unwrap()).collect();
    assert_eq!(got, ["A0", "A1", "A2", "C0", "C1"]);
}
