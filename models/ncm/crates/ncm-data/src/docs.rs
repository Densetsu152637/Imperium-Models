//! Read documents out of the fetched files: line-oriented JSON (plain or gzipped) and Parquet.
//!
//! Every record becomes a `serde_json::Value`, is filtered by the source's rules, and is turned
//! into one string by the source's [`Kind`]. Plain-text documents have literal special-token
//! strings defused so web text cannot inject control tokens.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use flate2::read::MultiGzDecoder;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::reader::RowIter;
use parquet::schema::types::Type;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::chat::{self, lookup};
use crate::error::{Error, IoContext, Result};
use crate::manifest::{Kind, Source};

pub type Documents = Box<dyn Iterator<Item = Result<String>>>;

/// Deterministic train/validation split by document content: a document goes to validation when
/// its hash falls in the first `fraction` of the hash range. Exact duplicates share a side.
pub fn is_validation(source: &str, text: &str, fraction: f64) -> bool {
    let mut h = Sha256::new();
    h.update(source.as_bytes());
    h.update([0]);
    h.update(text.as_bytes());
    let d = h.finalize();
    let x = u64::from_be_bytes(d[..8].try_into().expect("8 bytes"));
    (x as f64) < fraction * (u64::MAX as f64)
}

fn json_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Apply the source's filters and kind to one record.
pub fn record_text(source: &Source, row: &mut Value) -> Option<String> {
    for ex in &source.exclude {
        if let Some(v) = lookup(row, &ex.field) {
            let s = json_string(v);
            if ex.prefixes.iter().any(|p| s.starts_with(p.as_str())) {
                return None;
            }
        }
    }
    if let Some(path) = &source.require_true
        && lookup(row, path).and_then(Value::as_bool) != Some(true)
    {
        return None;
    }
    let text = match source.kind {
        Kind::Text => match row.get_mut(&source.text_field).map(Value::take)? {
            Value::String(s) => {
                if s.trim().is_empty() {
                    return None;
                }
                match chat::defuse(&s) {
                    std::borrow::Cow::Borrowed(_) => s,
                    std::borrow::Cow::Owned(fixed) => fixed,
                }
            }
            _ => return None,
        },
        Kind::Chat => chat::render(row, &source.messages_field)?,
        Kind::Trace => {
            let problem = lookup(row, source.prompt_field.as_deref()?)?.as_str()?;
            let response = lookup(row, source.response_field.as_deref()?)?.as_str()?;
            chat::render_trace(problem, response)?
        }
    };
    Some(text)
}

pub fn read_documents(path: &Path, source: &Source) -> Result<Documents> {
    let name = path.to_string_lossy();
    if name.ends_with(".parquet") {
        parquet_documents(path, source)
    } else if name.ends_with(".gz") {
        let file = File::open(path).ctx(format!("opening {}", path.display()))?;
        json_documents(Box::new(MultiGzDecoder::new(BufReader::new(file))), source)
    } else {
        let file = File::open(path).ctx(format!("opening {}", path.display()))?;
        json_documents(Box::new(file), source)
    }
}

/// Turns an unexpected end of stream into a clean end, remembering that it happened: a sample
/// prefix of a gzipped file stops mid-stream.
struct Tolerant {
    inner: Box<dyn Read>,
    cut: Arc<AtomicBool>,
}

impl Read for Tolerant {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self.inner.read(buf) {
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                self.cut.store(true, Ordering::SeqCst);
                Ok(0)
            }
            other => other,
        }
    }
}

struct JsonLines {
    reader: BufReader<Tolerant>,
    cut: Arc<AtomicBool>,
    source: Source,
    line: Vec<u8>,
}

fn json_documents(inner: Box<dyn Read>, source: &Source) -> Result<Documents> {
    let cut = Arc::new(AtomicBool::new(false));
    let reader = BufReader::with_capacity(1 << 20, Tolerant { inner, cut: cut.clone() });
    Ok(Box::new(JsonLines { reader, cut, source: source.clone(), line: Vec::new() }))
}

impl Iterator for JsonLines {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.line.clear();
            match self.reader.read_until(b'\n', &mut self.line) {
                Err(e) => return Some(Err(Error::Data(format!("reading records: {e}")))),
                Ok(0) => return None,
                Ok(_) => {}
            }
            // A record that does not end in a newline is the cut end of a prefix sample.
            let unterminated = self.line.last() != Some(&b'\n');
            let line = self.line.trim_ascii();
            if line.is_empty() {
                continue;
            }
            let mut row: Value = match serde_json::from_slice(line) {
                Ok(v) => v,
                // The last line of a cut sample is incomplete; anything else is corrupt.
                Err(_) if unterminated || self.cut.load(Ordering::SeqCst) => return None,
                Err(e) => return Some(Err(e.into())),
            };
            if let Some(text) = record_text(&self.source, &mut row) {
                return Some(Ok(text));
            }
        }
    }
}

struct ParquetDocs {
    rows: RowIter<'static>,
    source: Source,
}

fn parquet_documents(path: &Path, source: &Source) -> Result<Documents> {
    let data_err =
        |e: parquet::errors::ParquetError| Error::Data(format!("{}: {e}", path.display()));
    let file = File::open(path).ctx(format!("opening {}", path.display()))?;
    let reader = SerializedFileReader::new(file).map_err(data_err)?;
    let wanted = source.columns();
    let schema = reader.metadata().file_metadata().schema();
    let fields: Vec<_> = schema
        .get_fields()
        .iter()
        .filter(|f| wanted.iter().any(|w| w == f.name()))
        .cloned()
        .collect();
    if fields.is_empty() {
        return Err(Error::Data(format!(
            "{}: none of the columns {wanted:?} exist",
            path.display()
        )));
    }
    let projection =
        Type::group_type_builder(schema.name()).with_fields(fields).build().map_err(data_err)?;
    let rows =
        RowIter::from_file_into(Box::new(reader)).project(Some(projection)).map_err(data_err)?;
    Ok(Box::new(ParquetDocs { rows, source: source.clone() }))
}

impl Iterator for ParquetDocs {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let row = match self.rows.next()? {
                Ok(r) => r,
                Err(e) => return Some(Err(Error::Data(format!("parquet row: {e}")))),
            };
            let mut value = row.to_json_value();
            if let Some(text) = record_text(&self.source, &mut value) {
                return Some(Ok(text));
            }
        }
    }
}

/// Read several files round-robin, one document from each in turn, so a source whose files are
/// grouped (by language, say) comes out mixed. At most `width` files are open at once.
pub struct Interleave {
    pending: std::vec::IntoIter<std::path::PathBuf>,
    open: Vec<Documents>,
    width: usize,
    next: usize,
    source: Source,
}

impl Interleave {
    pub fn new(paths: Vec<std::path::PathBuf>, source: &Source) -> Self {
        Self {
            pending: paths.into_iter(),
            open: Vec::new(),
            width: source.interleave,
            next: 0,
            source: source.clone(),
        }
    }

    fn refill(&mut self) -> Result<()> {
        while self.open.len() < self.width {
            let Some(path) = self.pending.next() else {
                break;
            };
            self.open.push(read_documents(&path, &self.source)?);
        }
        Ok(())
    }
}

impl Iterator for Interleave {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Err(e) = self.refill() {
                return Some(Err(e));
            }
            if self.open.is_empty() {
                return None;
            }
            self.next %= self.open.len();
            match self.open[self.next].next() {
                Some(item) => {
                    self.next += 1;
                    return Some(item);
                }
                None => {
                    drop(self.open.remove(self.next));
                }
            }
        }
    }
}
