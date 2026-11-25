//! End to end on fixtures: lock -> sample -> train tokenizer -> tokenise -> shards -> stats.

mod common;

use ncm_data::lock::{Lock, LockEntry};
use ncm_data::manifest::Mix;
use ncm_data::sample::{self, SampleOptions};
use ncm_data::shards::{TokenIndex, read_shard};
use ncm_data::tokenize::{self, Options};
use ncm_data::{stats, tokenizer};
use serde_json::json;

const REV: &str = "0123456789abcdef0123456789abcdef01234567";

fn lock_entry(source: &str, repo: &str, path: &str, bytes: u64) -> LockEntry {
    LockEntry {
        source: source.into(),
        repo: repo.into(),
        revision: REV.into(),
        path: path.into(),
        size: bytes,
        bytes,
        sha256: None,
        verified: false,
        partial: false,
    }
}

/// A data dir with source `a` (plain documents) and `b` (reasoning traces).
fn fixture(dir: &std::path::Path) -> (Mix, Vec<String>) {
    let mix = Mix::parse(&common::mix_text(0.75, 0.25)).unwrap();
    let docs = common::corpus(400);
    let rows: Vec<_> = docs.iter().map(|d| json!({ "text": d })).collect();
    let a_path = dir.join("raw/a/part.jsonl");
    common::write_jsonl(&a_path, &rows);
    let traces: Vec<_> = (0..60)
        .map(|i| {
            json!({"q": format!("What is {i} + {i}?"),
                   "r": format!("<think>\nDouble {i} to get {}.\n</think>\nThe answer is {}.", i * 2, i * 2)})
        })
        .collect();
    let b_path = dir.join("raw/b/part.jsonl");
    common::write_jsonl(&b_path, &traces);
    let mut lock = Lock::default();
    lock.merge(vec![
        lock_entry("a", "x/a", "part.jsonl", std::fs::metadata(&a_path).unwrap().len()),
        lock_entry("b", "x/b", "part.jsonl", std::fs::metadata(&b_path).unwrap().len()),
    ]);
    lock.save(&dir.join("manifest.lock.json")).unwrap();
    (mix, docs)
}

#[test]
fn builds_tokenizer_and_shards_that_decode_back_to_the_documents() {
    let dir = tempfile::tempdir().unwrap();
    let (mix, docs) = fixture(dir.path());

    let sample_path = dir.path().join("sample/sample.jsonl");
    let reports = sample::run(
        &mix,
        dir.path(),
        &[],
        &sample_path,
        &SampleOptions { bytes: 60_000, max_doc_bytes: 4096, val_fraction: mix.val_fraction },
    )
    .unwrap();
    // The byte budget is split by share: three quarters from `a`.
    assert_eq!(reports.len(), 2);
    assert!(reports[0].bytes > 2 * reports[1].bytes);
    assert!(reports[1].documents > 0, "traces are in the sample");

    let tok = tokenizer::train(sample::read_texts(&sample_path).unwrap(), 550).unwrap();
    tokenizer::save(&tok, &dir.path().join("tokenizer/tokenizer.json")).unwrap();

    let opts = Options {
        budget_tokens: 1_000_000,
        shard_tokens: 4000,
        val_fraction: mix.val_fraction,
        max_doc_bytes: 1 << 20,
        batch_docs: 64,
    };
    let index = tokenize::run(&mix, dir.path(), &[], &opts).unwrap();

    // Both sources fully read: the target is far above what the fixtures hold.
    assert_eq!(index.sources.len(), 2);
    let a = &index.sources[0];
    assert_eq!((a.name.as_str(), a.category.as_str(), a.phase.as_str()), ("a", "web", "main"));
    assert_eq!(a.train.documents + a.val.documents, 400);
    assert!(a.val.documents > 0 && a.train.documents > a.val.documents);
    assert!(a.exhausted);
    let b = &index.sources[1];
    assert_eq!((b.category.as_str(), b.phase.as_str()), ("reasoning", "anneal"));
    assert_eq!(b.train.documents + b.val.documents, 60);

    // The shards hold exactly the counted tokens, in fixed-size files, per source and split.
    for split in ["train", "val"] {
        for s in &index.sources {
            let counted = if split == "train" { s.train.tokens } else { s.val.tokens };
            let shards: Vec<_> =
                index.shards.iter().filter(|x| x.source == s.name && x.split == split).collect();
            assert_eq!(shards.iter().map(|x| x.tokens).sum::<u64>(), counted);
            for (i, sh) in shards.iter().enumerate() {
                let path = dir.path().join("tokens").join(split).join(&sh.file);
                let ids = read_shard(&path).unwrap();
                assert_eq!(ids.len() as u64, sh.tokens);
                assert!(ids.iter().all(|&t| (t as usize) < index.vocab_size));
                if i + 1 < shards.len() {
                    assert_eq!(sh.tokens, 4000);
                }
            }
        }
    }

    // Decoding a source's train shards gives back its documents, EOT separated, in file order
    // (the split removes the validation documents).
    let mut ids = Vec::new();
    for sh in index.shards.iter().filter(|x| x.source == "a" && x.split == "train") {
        ids.extend(read_shard(&dir.path().join("tokens/train").join(&sh.file)).unwrap());
    }
    let decoded: Vec<String> = ids
        .split(|&t| t == 0)
        .filter(|d| !d.is_empty())
        .map(|d| tok.decode(&d.iter().map(|&t| t as u32).collect::<Vec<_>>(), false).unwrap())
        .collect();
    let expected: Vec<&String> =
        docs.iter().filter(|d| !ncm_data::docs::is_validation("a", d, mix.val_fraction)).collect();
    assert_eq!(decoded.len(), expected.len());
    for (got, want) in decoded.iter().zip(expected) {
        assert_eq!(got, want);
    }

    // Trace documents carry the think markers as single reserved tokens.
    let mut trace_ids = Vec::new();
    for sh in index.shards.iter().filter(|x| x.source == "b" && x.split == "train") {
        trace_ids.extend(read_shard(&dir.path().join("tokens/train").join(&sh.file)).unwrap());
    }
    let count = |id: u16| trace_ids.iter().filter(|&&t| t == id).count();
    assert_eq!(count(8), count(9));
    assert_eq!(count(8) as u64, b.train.documents);
    assert_eq!(count(2) as u64, 2 * b.train.documents, "user and assistant turns");

    // The index on disk is the one returned, and the report names every source.
    let loaded = TokenIndex::load(&dir.path().join("tokens")).unwrap();
    assert_eq!(loaded.shards, index.shards);
    let report = stats::report(&loaded);
    for needle in ["a ", "b ", "category reasoning", "phase anneal", "B/token", "validation"] {
        assert!(report.contains(needle), "report lacks {needle:?}:\n{report}");
    }

    // Tokenising again is deterministic.
    let again = tokenize::run(&mix, dir.path(), &[], &opts).unwrap();
    assert_eq!(again.shards, index.shards);
}

#[test]
fn stops_at_the_token_target_for_each_source() {
    let dir = tempfile::tempdir().unwrap();
    let (mix, _) = fixture(dir.path());
    let tok = tokenizer::train(common::corpus(300).into_iter(), 600).unwrap();
    tokenizer::save(&tok, &dir.path().join("tokenizer/tokenizer.json")).unwrap();
    let opts = Options {
        budget_tokens: 4000,
        shard_tokens: 1 << 20,
        val_fraction: 0.0,
        max_doc_bytes: 1 << 20,
        batch_docs: 16,
    };
    let index = tokenize::run(&mix, dir.path(), &["a".to_string()], &opts).unwrap();
    let a = &index.sources[0];
    assert_eq!(index.sources.len(), 1);
    assert_eq!(a.target_tokens, 3000);
    assert!(!a.exhausted);
    // A batch can overshoot by at most its own size.
    assert!(a.train.tokens >= 3000 && a.train.tokens < 3000 + 16 * 400);
    assert!(a.train.documents < 400);
}

#[test]
fn a_missing_lock_entry_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let (mix, _) = fixture(dir.path());
    std::fs::write(dir.path().join("manifest.lock.json"), "{\"files\": []}").unwrap();
    let tok = tokenizer::train(common::corpus(300).into_iter(), 600).unwrap();
    tokenizer::save(&tok, &dir.path().join("tokenizer/tokenizer.json")).unwrap();
    let opts = Options {
        budget_tokens: 4000,
        shard_tokens: 1000,
        val_fraction: 0.0,
        max_doc_bytes: 1 << 20,
        batch_docs: 16,
    };
    let err = tokenize::run(&mix, dir.path(), &[], &opts).unwrap_err().to_string();
    assert!(err.contains("run fetch"), "{err}");
}
