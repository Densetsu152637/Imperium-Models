//! Tokenizer-training sample: a byte-limited text sample drawn from every source in proportion
//! to its share, so web, code, papers, maths, reasoning traces and chat markers all shape the
//! merges.

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::docs::{Interleave, is_validation};
use crate::error::{Error, IoContext, Result};
use crate::lock::Lock;
use crate::manifest::Mix;
use crate::tokenize::local_files;

#[derive(Debug, Clone)]
pub struct SampleOptions {
    /// Total bytes of text to draw.
    pub bytes: u64,
    /// Documents longer than this are cut so a few giant files cannot dominate.
    pub max_doc_bytes: usize,
    pub val_fraction: f64,
}

#[derive(Serialize, Deserialize)]
struct Line {
    source: String,
    text: String,
}

#[derive(Debug, Clone)]
pub struct SampleReport {
    pub source: String,
    pub documents: u64,
    pub bytes: u64,
}

/// Cut to at most `max` bytes on a character boundary.
fn truncate(text: &mut String, max: usize) {
    if text.len() > max {
        let mut cut = max;
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        text.truncate(cut);
    }
}

pub fn run(
    mix: &Mix,
    data_dir: &Path,
    only: &[String],
    out: &Path,
    opts: &SampleOptions,
) -> Result<Vec<SampleReport>> {
    let lock = Lock::load(&data_dir.join("manifest.lock.json"))?;
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).ctx(format!("creating {}", dir.display()))?;
    }
    let mut writer =
        BufWriter::new(std::fs::File::create(out).ctx(format!("creating {}", out.display()))?);
    let sources = mix.select(only)?;
    let weight: f64 = sources.iter().map(|s| s.share).sum();
    let mut reports = Vec::new();
    for source in sources {
        let budget = (opts.bytes as f64 * source.share / weight) as u64;
        let files = local_files(data_dir, &lock, source)?;
        let mut report = SampleReport { source: source.name.clone(), documents: 0, bytes: 0 };
        for doc in Interleave::new(files, source) {
            let mut text = doc?;
            if is_validation(&source.name, &text, opts.val_fraction) {
                continue;
            }
            truncate(&mut text, opts.max_doc_bytes);
            report.documents += 1;
            report.bytes += text.len() as u64;
            let line = Line { source: source.name.clone(), text };
            serde_json::to_writer(&mut writer, &line)?;
            writer.write_all(b"\n").ctx("writing the sample")?;
            if report.bytes >= budget {
                break;
            }
        }
        eprintln!(
            "sample {}: {} documents, {:.1} MB (budget {:.1} MB)",
            report.source,
            report.documents,
            report.bytes as f64 / 1e6,
            budget as f64 / 1e6
        );
        reports.push(report);
    }
    writer.flush().ctx("flushing the sample")?;
    Ok(reports)
}

/// Stream the sample's documents for tokenizer training.
pub fn read_texts(path: &Path) -> Result<impl Iterator<Item = String> + Send + use<>> {
    let file = std::fs::File::open(path).ctx(format!("opening {}", path.display()))?;
    Ok(BufReader::with_capacity(1 << 20, file)
        .lines()
        .map_while(std::result::Result::ok)
        .filter(|l| !l.is_empty())
        .filter_map(|l| serde_json::from_str::<Line>(&l).ok())
        .map(|l| l.text))
}

pub fn check_nonempty(path: &Path) -> Result<()> {
    match std::fs::metadata(path) {
        Ok(m) if m.len() > 0 => Ok(()),
        _ => {
            Err(Error::Data(format!("{} is missing or empty; run `sample` first", path.display())))
        }
    }
}
