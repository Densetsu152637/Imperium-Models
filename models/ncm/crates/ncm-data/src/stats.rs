//! `stats`: documents, bytes, tokens and bytes per token, per source, category and phase.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::shards::{Counts, TokenIndex};

fn add(into: &mut Counts, c: &Counts) {
    into.documents += c.documents;
    into.bytes += c.bytes;
    into.tokens += c.tokens;
}

fn bytes_per_token(c: &Counts) -> f64 {
    if c.tokens == 0 { 0.0 } else { c.bytes as f64 / c.tokens as f64 }
}

fn row(out: &mut String, label: &str, tag: &str, c: &Counts, target: Option<u64>) {
    let fill = target
        .filter(|t| *t > 0)
        .map_or(String::new(), |t| format!("{:>6.1}%", c.tokens as f64 * 100.0 / t as f64));
    let _ = writeln!(
        out,
        "{label:<22} {tag:<10} {:>11} {:>12.1} {:>12.3} {:>8.3} {fill:>8}",
        c.documents,
        c.bytes as f64 / 1e6,
        c.tokens as f64 / 1e6,
        bytes_per_token(c),
    );
}

/// A text table for the whole index (train and validation combined, with a validation line).
pub fn report(index: &TokenIndex) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{:<22} {:<10} {:>11} {:>12} {:>12} {:>8} {:>8}",
        "source", "category", "documents", "MB", "Mtokens", "B/token", "of target"
    );
    let mut total = Counts::default();
    let mut val = Counts::default();
    let mut by_category: BTreeMap<String, Counts> = BTreeMap::new();
    let mut by_phase: BTreeMap<String, Counts> = BTreeMap::new();
    for s in &index.sources {
        let mut c = s.train.clone();
        add(&mut c, &s.val);
        row(&mut out, &s.name, &s.category, &c, Some(s.target_tokens));
        add(&mut total, &c);
        add(&mut val, &s.val);
        add(by_category.entry(s.category.clone()).or_default(), &c);
        add(by_phase.entry(s.phase.clone()).or_default(), &c);
        if s.exhausted {
            let _ = writeln!(out, "  ^ files exhausted before the target was reached");
        }
        if s.refusals_dropped > 0 {
            let _ = writeln!(out, "  ^ {} refusal records dropped", s.refusals_dropped);
        }
        if s.skipped_oversize > 0 {
            let _ = writeln!(out, "  ^ {} oversize documents skipped", s.skipped_oversize);
        }
    }
    let _ = writeln!(out);
    for (k, c) in &by_category {
        row(&mut out, &format!("category {k}"), "", c, None);
    }
    for (k, c) in &by_phase {
        row(&mut out, &format!("phase {k}"), "", c, None);
    }
    row(&mut out, "total", "", &total, None);
    row(&mut out, "  of which validation", "", &val, None);
    let _ = writeln!(
        out,
        "vocab {} ids, {} shards of up to {} tokens",
        index.vocab_size,
        index.shards.len(),
        index.shard_tokens
    );
    out
}
