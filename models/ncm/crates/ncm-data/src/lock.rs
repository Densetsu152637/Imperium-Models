//! `data/manifest.lock.json`: exactly which files were fetched, at which revision, with what
//! sizes and hashes. The tokeniser reads its input files from here, not from a directory scan.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{IoContext, Result};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Lock {
    pub files: Vec<LockEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockEntry {
    pub source: String,
    pub repo: String,
    pub revision: String,
    /// Path inside the repo; the local copy is `raw/<source>/<path>`.
    pub path: String,
    /// Size of the whole file on the Hub.
    pub size: u64,
    /// Bytes on disk; below `size` when only a prefix was fetched for a sample.
    pub bytes: u64,
    /// SHA-256 the Hub advertises for the whole file (absent for small non-LFS files).
    pub sha256: Option<String>,
    /// The whole file was downloaded and its SHA-256 matched.
    pub verified: bool,
    pub partial: bool,
}

impl Lock {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path).ctx(format!("reading {}", path.display()))?;
        Ok(serde_json::from_str(&text)?)
    }

    /// Replace or add entries (keyed by source, revision and path), keeping the file sorted.
    pub fn merge(&mut self, entries: Vec<LockEntry>) {
        for e in entries {
            self.files.retain(|o| {
                !(o.source == e.source && o.path == e.path && o.revision == e.revision)
            });
            self.files.push(e);
        }
        self.files.sort_by(|a, b| (&a.source, &a.path).cmp(&(&b.source, &b.path)));
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let tmp = path.with_extension("json.tmp");
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(&tmp, text + "\n").ctx(format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).ctx(format!("replacing {}", path.display()))
    }

    /// Entries of one source at one revision, in path order.
    pub fn for_source<'a>(
        &'a self,
        source: &'a str,
        revision: &'a str,
    ) -> impl Iterator<Item = &'a LockEntry> + 'a {
        self.files.iter().filter(move |e| e.source == source && e.revision == revision)
    }
}
