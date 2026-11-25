//! The pinned source manifest (`configs/data-mix.toml`).
//!
//! Every source names a Hugging Face dataset repo at one commit revision, its licence, the files
//! to take, and the share of the token budget it should fill.

use std::collections::BTreeMap;
use std::path::Path;

use globset::{Glob, GlobBuilder};
use serde::{Deserialize, Serialize};

use crate::error::{Error, IoContext, Result};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Mix {
    /// Total training tokens the shares are fractions of.
    pub budget_tokens: u64,
    /// Fraction of documents (by content hash) held out into `val/`.
    pub val_fraction: f64,
    /// The mix for the last stretch of pretraining, which favours reasoning-dense data.
    pub anneal: Anneal,
    #[serde(rename = "source")]
    pub sources: Vec<Source>,
}

/// The final-phase mix. Recorded here for the training loop; the data pipeline only tags
/// shards by source and phase.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Anneal {
    /// Fraction of all pretraining tokens spent in the anneal phase (the last tokens).
    pub fraction: f64,
    /// Source name to share of the anneal tokens; sums to 1.
    pub shares: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    /// Sampled throughout pretraining.
    Main,
    /// Reserved for the anneal phase: only sampled there.
    Anneal,
}

/// What a source contributes; the shares of a category add up to its target in the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Category {
    Web,
    Code,
    Research,
    Math,
    Reasoning,
    Instruct,
}

/// Drop records whose field (dotted path) starts with any of the prefixes.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Exclude {
    pub field: String,
    pub prefixes: Vec<String>,
}

/// How a record becomes text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// One string field is the document.
    Text,
    /// A list of chat messages (optionally with tools and reasoning) rendered with the NCM chat
    /// markers.
    Chat,
    /// A problem and one reasoning response, rendered as a user turn and an assistant turn whose
    /// reasoning sits between `<think>` and `</think>` followed by the answer.
    Trace,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Source {
    /// Short path-safe name; the directory under `data/raw/`.
    pub name: String,
    pub repo: String,
    /// Dataset config/subset name, for the record (the file glob does the selecting).
    #[serde(default)]
    pub subset: Option<String>,
    /// Full 40-character commit hash; files are fetched from `resolve/<revision>/`.
    pub revision: String,
    pub license: String,
    pub gated: bool,
    /// Fraction of `budget_tokens`; all shares sum to 1.
    pub share: f64,
    /// Glob over repo file paths (`*` does not cross `/`).
    pub glob: String,
    /// Keep every `stride`-th matching file (sorted by path) for an even spread.
    #[serde(default = "one")]
    pub stride: usize,
    /// Cap on selected files after striding.
    #[serde(default)]
    pub max_files: Option<usize>,
    /// How many files are read round-robin at once, so a file-grouped source is mixed.
    #[serde(default = "four")]
    pub interleave: usize,
    pub kind: Kind,
    pub category: Category,
    pub phase: Phase,
    #[serde(default = "text_name")]
    pub text_field: String,
    #[serde(default = "messages_name")]
    pub messages_field: String,
    /// Trace sources: dotted paths to the problem and to the response (`a.0.b` indexes lists).
    #[serde(default)]
    pub prompt_field: Option<String>,
    #[serde(default)]
    pub response_field: Option<String>,
    /// Trace sources: a dotted path that must be boolean true (a verifier's verdict).
    #[serde(default)]
    pub require_true: Option<String>,
    #[serde(default)]
    pub exclude: Vec<Exclude>,
    /// Fetch at most this many bytes of the source by default (a prefix of line-oriented files),
    /// when the glob matches far more data than the share needs.
    #[serde(default)]
    pub fetch_bytes: Option<u64>,
    /// What was checked for Claude/Anthropic-generated data, and the result.
    pub claude_check: String,
    /// Models that generated or filtered the data, when synthetic.
    #[serde(default)]
    pub generators: Option<String>,
    /// Estimated tokens available from the selected files, in this tokenizer.
    #[serde(default)]
    pub est_tokens: Option<u64>,
    #[serde(default)]
    pub note: String,
}

fn one() -> usize {
    1
}
fn four() -> usize {
    4
}
fn text_name() -> String {
    "text".into()
}
fn messages_name() -> String {
    "messages".into()
}

impl Mix {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).ctx(format!("reading {}", path.display()))?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let mix: Mix = toml::from_str(text).map_err(|e| Error::Manifest(e.to_string()))?;
        mix.validate()?;
        Ok(mix)
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |msg: String| Err(Error::Manifest(msg));
        if self.budget_tokens == 0 {
            return bad("budget_tokens must be positive".into());
        }
        if !(0.0..0.5).contains(&self.val_fraction) {
            return bad("val_fraction must be in [0, 0.5)".into());
        }
        if self.sources.is_empty() {
            return bad("no sources".into());
        }
        let mut share_sum = 0.0;
        for (i, s) in self.sources.iter().enumerate() {
            let n = &s.name;
            let name_ok = !n.is_empty()
                && n.bytes().all(|b| {
                    b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_'
                });
            if !name_ok {
                return bad(format!("source name {n:?} must be [a-z0-9_-]+"));
            }
            if self.sources[..i].iter().any(|o| o.name == s.name) {
                return bad(format!("duplicate source {n}"));
            }
            if s.revision.len() != 40 || !s.revision.bytes().all(|b| b.is_ascii_hexdigit()) {
                return bad(format!("{n}: revision must be a 40-character commit hash"));
            }
            if s.license.trim().is_empty() {
                return bad(format!("{n}: licence is required"));
            }
            if s.gated {
                return bad(format!(
                    "{n}: gated sources need the user's approval of their terms first"
                ));
            }
            if s.claude_check.trim().is_empty() {
                return bad(format!("{n}: record the Claude-data check"));
            }
            if s.stride == 0 || s.interleave == 0 || s.max_files == Some(0) {
                return bad(format!("{n}: stride, interleave and max_files must be positive"));
            }
            if !(s.share > 0.0 && s.share <= 1.0) {
                return bad(format!("{n}: share must be in (0, 1]"));
            }
            if s.kind == Kind::Trace && (s.prompt_field.is_none() || s.response_field.is_none()) {
                return bad(format!("{n}: trace sources need prompt_field and response_field"));
            }
            if let Err(e) = Glob::new(&s.glob) {
                return bad(format!("{n}: bad glob: {e}"));
            }
            share_sum += s.share;
        }
        if (share_sum - 1.0).abs() > 1e-6 {
            return bad(format!("shares sum to {share_sum}, not 1"));
        }
        self.anneal.validate(self)
    }

    pub fn source(&self, name: &str) -> Result<&Source> {
        self.sources
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| Error::Manifest(format!("no source named {name}")))
    }

    /// Sources named in `only`, or all of them when it is empty.
    pub fn select<'a>(&'a self, only: &[String]) -> Result<Vec<&'a Source>> {
        if only.is_empty() {
            return Ok(self.sources.iter().collect());
        }
        only.iter().map(|n| self.source(n)).collect()
    }
}

impl Anneal {
    fn validate(&self, mix: &Mix) -> Result<()> {
        let bad = |msg: String| Err(Error::Manifest(msg));
        if !(self.fraction > 0.0 && self.fraction < 1.0) {
            return bad("anneal.fraction must be in (0, 1)".into());
        }
        let sum: f64 = self.shares.values().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return bad(format!("anneal shares sum to {sum}, not 1"));
        }
        for name in self.shares.keys() {
            mix.source(name)?;
        }
        for s in &mix.sources {
            if s.phase != Phase::Anneal {
                continue;
            }
            let Some(a) = self.shares.get(&s.name) else {
                return bad(format!(
                    "{}: anneal-phase source is missing from anneal.shares",
                    s.name
                ));
            };
            // Only sampled in the anneal phase, so it cannot hold more tokens than that phase uses.
            if s.share > a * self.fraction + 1e-9 {
                return bad(format!(
                    "{}: share {} exceeds what the anneal phase consumes ({})",
                    s.name,
                    s.share,
                    a * self.fraction
                ));
            }
        }
        Ok(())
    }
}

impl Source {
    /// Top-level columns or keys a record needs, to project Parquet reads.
    pub fn columns(&self) -> Vec<String> {
        let root = |p: &str| p.split('.').next().unwrap_or(p).to_string();
        let mut cols = match self.kind {
            Kind::Text => vec![self.text_field.clone()],
            Kind::Chat => vec![self.messages_field.clone(), "tools".into(), "metadata".into()],
            Kind::Trace => [&self.prompt_field, &self.response_field]
                .into_iter()
                .flatten()
                .map(|p| root(p))
                .collect(),
        };
        cols.extend(self.require_true.iter().map(|p| root(p)));
        cols.extend(self.exclude.iter().map(|e| root(&e.field)));
        cols.sort();
        cols.dedup();
        cols
    }

    /// Token target for this source under `budget` tokens.
    pub fn target_tokens(&self, budget: u64) -> u64 {
        (budget as f64 * self.share).round() as u64
    }

    /// The repo files this source takes: glob, sorted by path, every `stride`-th, capped.
    pub fn select_files<'a, T>(&self, files: &'a [T], path: impl Fn(&T) -> &str) -> Vec<&'a T> {
        let matcher = GlobBuilder::new(&self.glob)
            .literal_separator(true)
            .build()
            .expect("glob validated")
            .compile_matcher();
        let mut hits: Vec<&T> = files.iter().filter(|f| matcher.is_match(path(f))).collect();
        hits.sort_by(|a, b| path(a).cmp(path(b)));
        let strided = hits.into_iter().step_by(self.stride);
        match self.max_files {
            Some(n) => strided.take(n).collect(),
            None => strided.collect(),
        }
    }
}
