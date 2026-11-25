//! Stage 2 data pipeline for the NCM models: fetch the pinned open sources, build the
//! tokenizer-training sample, train the shared byte-level BPE, tokenise into shards, report.

pub mod chat;
pub mod docs;
pub mod error;
pub mod hub;
pub mod lock;
pub mod manifest;
pub mod sample;
pub mod shards;
pub mod stats;
pub mod tokenize;
pub mod tokenizer;

use std::path::{Path, PathBuf};

use error::{IoContext, Result};

/// The `models/ncm` directory this crate lives in.
pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// The `vocab_size` both model configs agree on.
pub fn model_vocab_size(root: &Path) -> Result<usize> {
    let mut sizes = Vec::new();
    for name in ["ncm-250m.toml", "tok-250m.toml"] {
        let path = root.join("configs").join(name);
        let text = std::fs::read_to_string(&path).ctx(format!("reading {}", path.display()))?;
        let v: toml::Table =
            toml::from_str(&text).map_err(|e| error::Error::Manifest(e.to_string()))?;
        let n = v
            .get("vocab_size")
            .and_then(toml::Value::as_integer)
            .ok_or_else(|| error::Error::Manifest(format!("{name} has no vocab_size")))?;
        sizes.push(n as usize);
    }
    if sizes[0] != sizes[1] {
        return Err(error::Error::Manifest("the two model configs disagree on vocab_size".into()));
    }
    Ok(sizes[0])
}

/// Parse a count with an optional K/M/G suffix and optional trailing `B`: `200M`, `1.5GB`, `20G`.
pub fn parse_count(text: &str) -> std::result::Result<u64, String> {
    let t = text.trim().trim_end_matches(['B', 'b']);
    let (digits, scale) = match t.chars().last() {
        Some('K' | 'k') => (&t[..t.len() - 1], 1e3),
        Some('M' | 'm') => (&t[..t.len() - 1], 1e6),
        Some('G' | 'g') => (&t[..t.len() - 1], 1e9),
        _ => (t, 1.0),
    };
    let n: f64 =
        digits.trim().parse().map_err(|_| format!("{text:?} is not a number like 200M or 1.5G"))?;
    if n < 0.0 {
        return Err(format!("{text:?} is negative"));
    }
    Ok((n * scale).round() as u64)
}

#[cfg(test)]
mod tests {
    use super::parse_count;

    #[test]
    fn parses_counts() {
        assert_eq!(parse_count("200M").unwrap(), 200_000_000);
        assert_eq!(parse_count("1.5GB").unwrap(), 1_500_000_000);
        assert_eq!(parse_count("20g").unwrap(), 20_000_000_000);
        assert_eq!(parse_count("4096").unwrap(), 4096);
        assert!(parse_count("lots").is_err());
    }
}
