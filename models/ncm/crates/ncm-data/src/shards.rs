//! Token shards: little-endian `u16` ids in fixed-size files, plus a JSON index.
//!
//! Documents are packed back to back, each ending with the end-of-text id, and may span a shard
//! boundary. A shard holds one source, so the trainer can sample sources by weight (and switch
//! weights for the anneal phase) just by choosing shards.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Error, IoContext, Result};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardInfo {
    /// File name relative to `tokens/<split>/`.
    pub file: String,
    pub source: String,
    pub split: String,
    pub tokens: u64,
    pub sha256: String,
}

/// Counts for one source and split.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Counts {
    pub documents: u64,
    /// UTF-8 bytes of the text before tokenising.
    pub bytes: u64,
    /// Tokens including one end-of-text per document.
    pub tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceStats {
    pub name: String,
    pub category: String,
    pub phase: String,
    pub target_tokens: u64,
    pub train: Counts,
    pub val: Counts,
    /// Documents skipped for exceeding the size cap.
    pub skipped_oversize: u64,
    /// Chat and trace records dropped by the no-refusals rule ([`ncm_model::refusal`]).
    #[serde(default)]
    pub refusals_dropped: u64,
    /// The files ran out before the token target was reached.
    pub exhausted: bool,
    pub files: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenIndex {
    pub dtype: String,
    pub vocab_size: usize,
    pub eot_id: u32,
    pub shard_tokens: u64,
    pub val_fraction: f64,
    /// SHA-256 of the tokenizer.json the ids belong to.
    pub tokenizer_sha256: String,
    pub sources: Vec<SourceStats>,
    pub shards: Vec<ShardInfo>,
}

pub const INDEX_FILE: &str = "index.json";

impl TokenIndex {
    pub fn load(tokens_dir: &Path) -> Result<Self> {
        let path = tokens_dir.join(INDEX_FILE);
        let text = std::fs::read_to_string(&path).ctx(format!("reading {}", path.display()))?;
        Ok(serde_json::from_str(&text)?)
    }

    pub fn save(&self, tokens_dir: &Path) -> Result<()> {
        let path = tokens_dir.join(INDEX_FILE);
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, text + "\n").ctx(format!("writing {}", path.display()))
    }
}

/// Writes one source's tokens for one split.
pub struct ShardWriter {
    dir: PathBuf,
    source: String,
    split: String,
    shard_tokens: u64,
    current: Option<Open>,
    next_index: usize,
    done: Vec<ShardInfo>,
}

struct Open {
    out: BufWriter<File>,
    hash: Sha256,
    file: String,
    tokens: u64,
}

impl ShardWriter {
    pub fn new(dir: &Path, source: &str, split: &str, shard_tokens: u64) -> Result<Self> {
        std::fs::create_dir_all(dir).ctx(format!("creating {}", dir.display()))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            source: source.to_string(),
            split: split.to_string(),
            shard_tokens: shard_tokens.max(1),
            current: None,
            next_index: 0,
            done: Vec::new(),
        })
    }

    fn open_next(&mut self) -> Result<()> {
        let file = format!("{}-{:05}.bin", self.source, self.next_index);
        self.next_index += 1;
        let path = self.dir.join(&file);
        let out = BufWriter::with_capacity(
            1 << 20,
            File::create(&path).ctx(format!("creating {}", path.display()))?,
        );
        self.current = Some(Open { out, hash: Sha256::new(), file, tokens: 0 });
        Ok(())
    }

    fn close_current(&mut self) -> Result<()> {
        if let Some(mut open) = self.current.take() {
            open.out.flush().ctx("flushing a shard")?;
            self.done.push(ShardInfo {
                file: open.file,
                source: self.source.clone(),
                split: self.split.clone(),
                tokens: open.tokens,
                sha256: hex::encode(open.hash.finalize()),
            });
        }
        Ok(())
    }

    pub fn push(&mut self, mut tokens: &[u16]) -> Result<()> {
        while !tokens.is_empty() {
            if self.current.is_none() {
                self.open_next()?;
            }
            let open = self.current.as_mut().expect("opened above");
            let room = (self.shard_tokens - open.tokens) as usize;
            let (now, rest) = tokens.split_at(room.min(tokens.len()));
            let bytes: Vec<u8> = now.iter().flat_map(|t| t.to_le_bytes()).collect();
            open.hash.update(&bytes);
            open.out.write_all(&bytes).ctx("writing a shard")?;
            open.tokens += now.len() as u64;
            tokens = rest;
            if open.tokens == self.shard_tokens {
                self.close_current()?;
            }
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<Vec<ShardInfo>> {
        self.close_current()?;
        Ok(self.done)
    }
}

/// Read a whole shard back.
pub fn read_shard(path: &Path) -> Result<Vec<u16>> {
    let mut bytes = Vec::new();
    File::open(path)
        .ctx(format!("opening {}", path.display()))?
        .read_to_end(&mut bytes)
        .ctx(format!("reading {}", path.display()))?;
    if bytes.len() % 2 != 0 {
        return Err(Error::Data(format!("{} has an odd byte count", path.display())));
    }
    let (pairs, _) = bytes.as_chunks::<2>();
    Ok(pairs.iter().map(|c| u16::from_le_bytes(*c)).collect())
}
