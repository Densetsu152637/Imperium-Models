//! Stream each source's documents through the tokenizer into shards.

use std::path::{Path, PathBuf};
use std::sync::mpsc::sync_channel;

use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::docs::{Interleave, is_validation};
use crate::error::{Error, IoContext, Result};
use crate::hub::local_path;
use crate::lock::Lock;
use crate::manifest::{Mix, Source};
use crate::shards::{Counts, ShardInfo, ShardWriter, SourceStats, TokenIndex};
use crate::tokenizer::{self, EOT_ID};

#[derive(Debug, Clone)]
pub struct Options {
    /// Total tokens that the sources' shares are fractions of.
    pub budget_tokens: u64,
    pub shard_tokens: u64,
    pub val_fraction: f64,
    /// Documents larger than this many bytes are skipped.
    pub max_doc_bytes: usize,
    /// Documents per encoding batch (a batch is also cut at 4 MiB of text).
    pub batch_docs: usize,
}

const BATCH_BYTES: usize = 4 << 20;

/// The files the lock holds for `source`, as local paths, in path order.
pub fn local_files(data_dir: &Path, lock: &Lock, source: &Source) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for e in lock.for_source(&source.name, &source.revision) {
        let path = local_path(data_dir, &source.name, &e.path);
        let len = std::fs::metadata(&path)
            .ctx(format!("{} is in the lock but missing; run fetch", path.display()))?
            .len();
        if len != e.bytes {
            return Err(Error::Data(format!(
                "{} is {len} bytes, the lock says {}; run fetch",
                path.display(),
                e.bytes
            )));
        }
        files.push(path);
    }
    if files.is_empty() {
        return Err(Error::Data(format!(
            "{}: no files in the lock at revision {}; run fetch",
            source.name, source.revision
        )));
    }
    Ok(files)
}

fn remove_old_shards(tokens_dir: &Path, source: &str) -> Result<()> {
    for split in ["train", "val"] {
        let dir = tokens_dir.join(split);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let prefix = format!("{source}-");
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let numbered = name
                .strip_prefix(&prefix)
                .and_then(|r| r.strip_suffix(".bin"))
                .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()));
            if numbered {
                std::fs::remove_file(entry.path()).ctx("removing an old shard")?;
            }
        }
    }
    Ok(())
}

/// Tokenise one source until its target is met or its files run out.
pub fn tokenize_source(
    tok: &Tokenizer,
    source: &Source,
    files: Vec<PathBuf>,
    opts: &Options,
    tokens_dir: &Path,
) -> Result<(SourceStats, Vec<ShardInfo>)> {
    remove_old_shards(tokens_dir, &source.name)?;
    let target = source.target_tokens(opts.budget_tokens);
    let mut train =
        ShardWriter::new(&tokens_dir.join("train"), &source.name, "train", opts.shard_tokens)?;
    let mut val =
        ShardWriter::new(&tokens_dir.join("val"), &source.name, "val", opts.shard_tokens)?;
    let mut stats = SourceStats {
        name: source.name.clone(),
        category: format!("{:?}", source.category).to_lowercase(),
        phase: format!("{:?}", source.phase).to_lowercase(),
        target_tokens: target,
        train: Counts::default(),
        val: Counts::default(),
        skipped_oversize: 0,
        exhausted: false,
        files: files.len(),
    };

    let (tx, rx) = sync_channel::<Result<Vec<String>>>(4);
    let reader_source = source.clone();
    let max_doc = opts.max_doc_bytes;
    let batch_docs = opts.batch_docs.max(1);
    let reader = std::thread::spawn(move || {
        let mut batch = Vec::new();
        let mut bytes = 0usize;
        let mut skipped = 0u64;
        for doc in Interleave::new(files, &reader_source) {
            match doc {
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return skipped;
                }
                Ok(text) if text.len() > max_doc => skipped += 1,
                Ok(text) => {
                    bytes += text.len();
                    batch.push(text);
                    if (bytes >= BATCH_BYTES || batch.len() >= batch_docs)
                        && tx.send(Ok(std::mem::take(&mut batch))).is_err()
                    {
                        return skipped;
                    }
                    if batch.is_empty() {
                        bytes = 0;
                    }
                }
            }
        }
        if !batch.is_empty() {
            let _ = tx.send(Ok(batch));
        }
        skipped
    });

    let mut next_report = 100_000_000u64;
    let mut outcome = Ok(());
    'batches: for batch in rx.iter() {
        let texts = match batch {
            Ok(t) => t,
            Err(e) => {
                outcome = Err(e);
                break;
            }
        };
        let lens: Vec<usize> = texts.iter().map(String::len).collect();
        let is_val: Vec<bool> =
            texts.iter().map(|t| is_validation(&source.name, t, opts.val_fraction)).collect();
        let encoded = match tokenizer::encode_documents(tok, texts) {
            Ok(e) => e,
            Err(e) => {
                outcome = Err(e);
                break;
            }
        };
        for ((ids, bytes), val_doc) in encoded.iter().zip(lens).zip(is_val) {
            let (counts, writer) =
                if val_doc { (&mut stats.val, &mut val) } else { (&mut stats.train, &mut train) };
            writer.push(ids)?;
            counts.documents += 1;
            counts.bytes += bytes as u64;
            counts.tokens += ids.len() as u64;
        }
        let total = stats.train.tokens + stats.val.tokens;
        if total >= next_report {
            eprintln!(
                "  {}: {:.2}B of {:.2}B tokens",
                source.name,
                total as f64 / 1e9,
                target as f64 / 1e9
            );
            next_report += 100_000_000;
        }
        if total >= target {
            break 'batches;
        }
    }
    let reached = stats.train.tokens + stats.val.tokens >= target;
    drop(rx);
    stats.skipped_oversize =
        reader.join().map_err(|_| Error::Data("reader thread panicked".into()))?;
    outcome?;
    stats.exhausted = !reached;
    let mut shards = train.finish()?;
    shards.extend(val.finish()?);
    Ok((stats, shards))
}

fn file_sha256(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).ctx(format!("reading {}", path.display()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

/// Tokenise the named sources (all when `only` is empty) and update `tokens/index.json`.
pub fn run(mix: &Mix, data_dir: &Path, only: &[String], opts: &Options) -> Result<TokenIndex> {
    let tok_path = data_dir.join("tokenizer").join("tokenizer.json");
    let tok = tokenizer::load(&tok_path)?;
    let lock = Lock::load(&data_dir.join("manifest.lock.json"))?;
    let tokens_dir = data_dir.join("tokens");
    std::fs::create_dir_all(&tokens_dir).ctx("creating tokens/")?;

    let mut index = TokenIndex::load(&tokens_dir).unwrap_or_else(|_| TokenIndex {
        dtype: "u16le".into(),
        vocab_size: tok.get_vocab_size(true),
        eot_id: EOT_ID,
        shard_tokens: opts.shard_tokens,
        val_fraction: opts.val_fraction,
        tokenizer_sha256: String::new(),
        sources: Vec::new(),
        shards: Vec::new(),
    });
    let sha = file_sha256(&tok_path)?;
    if !index.sources.is_empty() && index.tokenizer_sha256 != sha {
        return Err(Error::Data(
            "tokens/index.json was built with a different tokenizer; tokenise every source again \
             after clearing data/tokens"
                .into(),
        ));
    }
    index.tokenizer_sha256 = sha;
    index.vocab_size = tok.get_vocab_size(true);
    index.shard_tokens = opts.shard_tokens;
    index.val_fraction = opts.val_fraction;

    for source in mix.select(only)? {
        eprintln!(
            "tokenize {} (target {:.2}B tokens)",
            source.name,
            source.target_tokens(opts.budget_tokens) as f64 / 1e9
        );
        let files = local_files(data_dir, &lock, source)?;
        let (stats, shards) = tokenize_source(&tok, source, files, opts, &tokens_dir)?;
        index.sources.retain(|s| s.name != source.name);
        index.shards.retain(|s| s.source != source.name);
        index.sources.push(stats);
        index.shards.extend(shards);
        // Keep the index consistent after every source so an interrupted run loses little.
        index.save(&tokens_dir)?;
    }
    let order = |name: &str| mix.sources.iter().position(|s| s.name == name);
    index.sources.sort_by_key(|s| order(&s.name));
    index.shards.sort_by(|a, b| {
        (order(&a.source), &a.split, &a.file).cmp(&(order(&b.source), &b.split, &b.file))
    });
    index.save(&tokens_dir)?;
    Ok(index)
}
