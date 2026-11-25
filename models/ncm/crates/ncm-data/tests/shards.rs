//! Shard writer and reader, and the deterministic validation split.

use ncm_data::docs::is_validation;
use ncm_data::shards::{ShardWriter, read_shard};
use sha2::{Digest, Sha256};

#[test]
fn shards_round_trip_across_boundaries() {
    let dir = tempfile::tempdir().unwrap();
    let mut w = ShardWriter::new(dir.path(), "web", "train", 10).unwrap();
    let docs: Vec<Vec<u16>> =
        (0..7u16).map(|d| (0..(3 + d * 2)).map(|i| d * 1000 + i).collect()).collect();
    for d in &docs {
        w.push(d).unwrap();
    }
    let shards = w.finish().unwrap();

    let all: Vec<u16> = docs.concat();
    assert_eq!(shards.len(), all.len().div_ceil(10));
    let mut back = Vec::new();
    for (i, s) in shards.iter().enumerate() {
        assert_eq!(s.file, format!("web-{i:05}.bin"));
        assert_eq!(s.source, "web");
        assert_eq!(s.split, "train");
        let path = dir.path().join(&s.file);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len() as u64, s.tokens * 2);
        assert_eq!(hex::encode(Sha256::digest(&bytes)), s.sha256);
        // Little-endian u16, every shard full except the last.
        assert_eq!(&bytes[..2], &read_shard(&path).unwrap()[0].to_le_bytes());
        if i + 1 < shards.len() {
            assert_eq!(s.tokens, 10);
        }
        back.extend(read_shard(&path).unwrap());
    }
    assert_eq!(back, all);
}

#[test]
fn an_empty_writer_makes_no_shards() {
    let dir = tempfile::tempdir().unwrap();
    let w = ShardWriter::new(dir.path(), "web", "val", 10).unwrap();
    assert!(w.finish().unwrap().is_empty());
}

#[test]
fn the_split_is_deterministic_and_close_to_the_fraction() {
    let docs: Vec<String> = (0..40_000).map(|i| format!("document number {i}")).collect();
    let first: Vec<bool> = docs.iter().map(|d| is_validation("web", d, 0.005)).collect();
    let second: Vec<bool> = docs.iter().map(|d| is_validation("web", d, 0.005)).collect();
    assert_eq!(first, second);
    let val = first.iter().filter(|&&v| v).count();
    // 0.5% of 40,000 is 200; allow generous sampling noise.
    assert!((120..300).contains(&val), "{val} validation documents");
    // The decision depends on the content, not on position or batch.
    assert!(!is_validation("web", "anything", 0.0));
    assert!(is_validation("web", "anything", 0.999999));
    // A different source hashes differently.
    let moved = docs
        .iter()
        .filter(|d| is_validation("web", d, 0.5) != is_validation("code", d, 0.5))
        .count();
    assert!(moved > 10_000);
}
