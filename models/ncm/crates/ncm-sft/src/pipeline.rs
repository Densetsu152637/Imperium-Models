//! The stage 3 build: read a model's accepted token-era files, convert every record, write the
//! dataset, the needs-regeneration list and the stats.

use crate::convert::{Policy, Rejection, convert};
use crate::message::{Message, ToolDef};
use crate::split::{Split, split_of};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum SftError {
    #[error("{path}: {source}")]
    Io { path: PathBuf, source: std::io::Error },
    #[error("{path}: {source}")]
    Json { path: PathBuf, source: serde_json::Error },
}

fn io<T>(path: &Path, result: std::io::Result<T>) -> Result<T, SftError> {
    result.map_err(|source| SftError::Io { path: path.to_owned(), source })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    Emperor,
    Consul,
}

impl Model {
    pub fn name(self) -> &'static str {
        match self {
            Model::Emperor => "emperor",
            Model::Consul => "consul",
        }
    }

    /// The files the token-era pipeline wrote as final training sets, relative to
    /// `<archive>/<model>/data`. Emperor's `mix-emperor-v2` repeats the hosted samples and
    /// adds open-data grafts; `final` is the earlier local-teacher set.
    pub fn accepted_sources(self) -> &'static [&'static str] {
        match self {
            Model::Emperor => {
                &["hosted-1/final/final.jsonl", "mix-emperor-v2/final.jsonl", "final/final.jsonl"]
            }
            Model::Consul => &["hosted-1/final/final.jsonl"],
        }
    }
}

#[derive(Default, Serialize)]
pub struct FileStats {
    pub records_in: usize,
    pub accepted: usize,
    pub needs_regeneration: BTreeMap<String, usize>,
    pub excluded: BTreeMap<String, usize>,
    pub characters: usize,
    pub with_think: usize,
    pub with_tool_calls: usize,
}

#[derive(Default, Serialize)]
pub struct Totals {
    pub records_in: usize,
    pub accepted: usize,
    pub train: usize,
    pub val: usize,
    pub needs_regeneration: usize,
    pub excluded: usize,
    pub characters: usize,
    pub with_think: usize,
    pub with_tool_calls: usize,
    pub share_with_think: f64,
    pub share_with_tool_calls: f64,
    pub assistant_turns: usize,
    pub assistant_turns_with_think: usize,
    pub tool_calls: usize,
}

#[derive(Serialize)]
pub struct Unused {
    pub kind: String,
    pub records: usize,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub reasons: BTreeMap<String, usize>,
}

#[derive(Serialize)]
pub struct Stats {
    pub model: String,
    pub allowed_generators: Vec<String>,
    pub allowed_judges: Vec<String>,
    pub excluded_pilots: Vec<String>,
    pub files: BTreeMap<String, FileStats>,
    pub totals: Totals,
    pub accepted_by_generator: BTreeMap<String, usize>,
    pub qwen_artefacts_stripped: BTreeMap<String, usize>,
    pub tools_offered: BTreeMap<String, usize>,
    /// Hosted samples whose last judgement is a rejection (never written to a final set).
    pub judged_not_accepted: usize,
    /// Token-era files that are not training sets (rejects, quarantine, traces, prompts, ...),
    /// with their record counts. None of them is read as training data.
    pub not_training_data: BTreeMap<String, Unused>,
    pub not_training_data_by_kind: BTreeMap<String, usize>,
}

#[derive(Serialize)]
struct OutRecord<'a> {
    id: &'a str,
    source_file: &'a str,
    split: Split,
    meta: &'a Value,
    tools: &'a [ToolDef],
    messages: &'a [Message],
    text: &'a str,
    /// Byte ranges of `text` that are assistant turns (the SFT loss mask), `<|im_end|>` included.
    assistant_spans: &'a [(usize, usize)],
}

fn read_lines(path: &Path, mut each: impl FnMut(usize, &str)) -> Result<(), SftError> {
    let file = io(path, File::open(path))?;
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = io(path, line)?;
        if !line.trim().is_empty() {
            each(index + 1, &line);
        }
    }
    Ok(())
}

fn jsonl_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), SftError> {
    if !dir.is_dir() {
        return Ok(());
    }
    let mut entries: Vec<PathBuf> =
        io(dir, fs::read_dir(dir))?.filter_map(Result::ok).map(|entry| entry.path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            jsonl_files(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "jsonl") {
            out.push(path);
        }
    }
    Ok(())
}

fn relative(path: &Path, base: &Path) -> String {
    path.strip_prefix(base).unwrap_or(path).to_string_lossy().replace('\\', "/")
}

fn pilots_to_skip(data: &Path) -> Result<HashSet<String>, SftError> {
    let path = data.join("hosted-1/excluded-pilots.json");
    if !path.is_file() {
        return Ok(HashSet::new());
    }
    let text = io(&path, fs::read_to_string(&path))?;
    let value: Value =
        serde_json::from_str(&text).map_err(|source| SftError::Json { path, source })?;
    Ok(value.as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default())
}

fn kind_of(relative: &str) -> &'static str {
    let parts: Vec<&str> = relative.split('/').collect();
    for (needle, kind) in [
        ("rejects", "rejects"),
        ("quarantine", "quarantine"),
        ("pilot-judge-low", "pilots"),
        ("eval", "eval"),
        ("graft-v2", "open-data-grafts"),
        ("traces", "traces"),
        ("judged", "judged"),
    ] {
        if parts.contains(&needle) {
            return kind;
        }
    }
    if relative.contains("prompts") {
        "prompts"
    } else if relative.ends_with("spend.jsonl") {
        "spend"
    } else {
        "other"
    }
}

/// Per-file survey of the unused token-era files, and record counts by kind.
type Survey = (BTreeMap<String, Unused>, BTreeMap<String, usize>);

fn survey_unused(model: Model, data: &Path) -> Result<Survey, SftError> {
    let mut files = Vec::new();
    jsonl_files(data, &mut files)?;
    let used: HashSet<&str> = model.accepted_sources().iter().copied().collect();
    let mut found = BTreeMap::new();
    let mut by_kind: BTreeMap<String, usize> = BTreeMap::new();
    for path in files {
        let rel = relative(&path, data);
        if used.contains(rel.as_str()) {
            continue;
        }
        let kind = kind_of(&rel);
        let mut records = 0;
        let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
        read_lines(&path, |_, line| {
            records += 1;
            if kind == "rejects"
                && let Ok(value) = serde_json::from_str::<Value>(line)
            {
                let reason = value.get("reason").and_then(Value::as_str).unwrap_or("unknown");
                *reasons.entry(reason.to_owned()).or_default() += 1;
            }
        })?;
        *by_kind.entry(kind.to_owned()).or_default() += records;
        found.insert(
            format!("{}/data/{rel}", model.name()),
            Unused { kind: kind.to_owned(), records, reasons },
        );
    }
    Ok((found, by_kind))
}

/// Hosted samples whose latest judgement (a retry round replaces the first) is a rejection.
fn count_judged_not_accepted(data: &Path, skip: &HashSet<String>) -> Result<usize, SftError> {
    let mut files = Vec::new();
    jsonl_files(&data.join("hosted-1"), &mut files)?;
    files.retain(|p| {
        let rel = relative(p, &data.join("hosted-1"));
        !rel.starts_with("pilot-judge-low") && rel.split('/').any(|c| c == "judged")
    });
    let mut latest: HashMap<String, (bool, String)> = HashMap::new();
    for path in files {
        read_lines(&path, |_, line| {
            let Ok(value) = serde_json::from_str::<Value>(line) else { return };
            let Some(id) = value.get("id").and_then(Value::as_str) else { return };
            let meta = value.get("meta").unwrap_or(&Value::Null);
            let keep = meta.pointer("/judge/keep").and_then(Value::as_bool) == Some(true);
            let pilot = meta.get("pilot").map(|p| p.to_string().trim_matches('"').to_owned());
            latest.insert(id.to_owned(), (keep, pilot.unwrap_or_default()));
        })?;
    }
    Ok(latest.values().filter(|(keep, pilot)| !keep && !skip.contains(pilot)).count())
}

fn bump(map: &mut BTreeMap<String, usize>, key: &str) {
    *map.entry(key.to_owned()).or_default() += 1;
}

/// Build one model's dataset into `<out>/<model>/`. `archive` is the `token-era` directory.
pub fn build(archive: &Path, out: &Path, model: Model, policy: &Policy) -> Result<Stats, SftError> {
    let data = archive.join(model.name()).join("data");
    let out_dir = out.join(model.name());
    io(&out_dir, fs::create_dir_all(&out_dir))?;
    let dataset_path = out_dir.join("sft.jsonl");
    let regen_path = out_dir.join("needs-regeneration.jsonl");
    let mut dataset = BufWriter::new(io(&dataset_path, File::create(&dataset_path))?);
    let mut regen = BufWriter::new(io(&regen_path, File::create(&regen_path))?);

    let mut stats = Stats {
        model: model.name().to_owned(),
        allowed_generators: policy.allowed_generators.clone(),
        allowed_judges: policy.allowed_judges.clone(),
        excluded_pilots: {
            let mut pilots: Vec<String> = policy.excluded_pilots.iter().cloned().collect();
            pilots.sort();
            pilots
        },
        files: BTreeMap::new(),
        totals: Totals::default(),
        accepted_by_generator: BTreeMap::new(),
        qwen_artefacts_stripped: BTreeMap::new(),
        tools_offered: BTreeMap::new(),
        judged_not_accepted: 0,
        not_training_data: BTreeMap::new(),
        not_training_data_by_kind: BTreeMap::new(),
    };
    let mut seen: HashSet<String> = HashSet::new();
    for source in model.accepted_sources() {
        let path = data.join(source);
        let label = format!("{}/data/{source}", model.name());
        let mut file_stats = FileStats::default();
        if path.is_file() {
            let mut failure: Option<SftError> = None;
            read_lines(&path, |line_no, line| {
                file_stats.records_in += 1;
                let raw: Value = match serde_json::from_str(line) {
                    Ok(v) => v,
                    Err(_) => {
                        bump(&mut file_stats.needs_regeneration, "unparsable_json");
                        let _ = writeln!(
                            regen,
                            "{}",
                            json!({"id": format!("line:{line_no}"), "source_file": label, "reason": "unparsable_json"})
                        );
                        return;
                    }
                };
                let outcome = convert(&raw, policy).and_then(|done| {
                    if seen.insert(done.id.clone()) {
                        Ok(done)
                    } else {
                        Err(Rejection::Excluded("duplicate_id".into()))
                    }
                });
                match outcome {
                    Ok(done) => {
                        file_stats.accepted += 1;
                        let text = &done.rendered.text;
                        file_stats.characters += text.chars().count();
                        let think_turns =
                            done.messages.iter().filter(|m| m.has_reasoning()).count();
                        let calls: usize = done.messages.iter().map(|m| m.tool_calls.len()).sum();
                        file_stats.with_think += usize::from(think_turns > 0);
                        file_stats.with_tool_calls += usize::from(calls > 0);
                        stats.totals.assistant_turns += done.rendered.assistant_spans.len();
                        stats.totals.assistant_turns_with_think += think_turns;
                        stats.totals.tool_calls += calls;
                        bump(&mut stats.accepted_by_generator, &done.generator);
                        for kind in &done.stripped {
                            bump(&mut stats.qwen_artefacts_stripped, kind);
                        }
                        for tool in &done.tools {
                            bump(&mut stats.tools_offered, &tool.name);
                        }
                        let split = split_of(&done.id);
                        match split {
                            Split::Train => stats.totals.train += 1,
                            Split::Val => stats.totals.val += 1,
                        }
                        let record = OutRecord {
                            id: &done.id,
                            source_file: &label,
                            split,
                            meta: &done.meta,
                            tools: &done.tools,
                            messages: &done.messages,
                            text,
                            assistant_spans: &done.rendered.assistant_spans,
                        };
                        let written = serde_json::to_writer(&mut dataset, &record)
                            .map_err(std::io::Error::from)
                            .and_then(|()| dataset.write_all(b"\n"));
                        if let Err(source) = written {
                            failure = Some(SftError::Io { path: dataset_path.clone(), source });
                        }
                    }
                    Err(Rejection::Excluded(code)) => bump(&mut file_stats.excluded, &code),
                    Err(Rejection::NeedsRegeneration(code)) => {
                        bump(&mut file_stats.needs_regeneration, &code);
                        let id = raw
                            .get("id")
                            .cloned()
                            .unwrap_or_else(|| json!(format!("line:{line_no}")));
                        let entry = json!({"id": id, "source_file": label, "reason": code});
                        if let Err(source) = writeln!(regen, "{entry}") {
                            failure = Some(SftError::Io { path: regen_path.clone(), source });
                        }
                    }
                }
            })?;
            if let Some(error) = failure {
                return Err(error);
            }
        }
        stats.files.insert(label, file_stats);
    }
    io(&dataset_path, dataset.flush())?;
    io(&regen_path, regen.flush())?;

    for file in stats.files.values() {
        stats.totals.records_in += file.records_in;
        stats.totals.accepted += file.accepted;
        stats.totals.characters += file.characters;
        stats.totals.with_think += file.with_think;
        stats.totals.with_tool_calls += file.with_tool_calls;
        stats.totals.needs_regeneration += file.needs_regeneration.values().sum::<usize>();
        stats.totals.excluded += file.excluded.values().sum::<usize>();
    }
    let accepted = stats.totals.accepted.max(1) as f64;
    stats.totals.share_with_think = stats.totals.with_think as f64 / accepted;
    stats.totals.share_with_tool_calls = stats.totals.with_tool_calls as f64 / accepted;
    stats.judged_not_accepted = count_judged_not_accepted(&data, &policy.excluded_pilots)?;
    (stats.not_training_data, stats.not_training_data_by_kind) = survey_unused(model, &data)?;

    let stats_path = out_dir.join("stats.json");
    let text = serde_json::to_string_pretty(&stats)
        .map_err(|source| SftError::Json { path: stats_path.clone(), source })?;
    io(&stats_path, fs::write(&stats_path, text + "\n"))?;
    Ok(stats)
}

/// The policy for a model's archive: the token-era generators and the pilots it excluded.
pub fn policy_for(
    archive: &Path,
    model: Model,
    extra_generators: &[String],
) -> Result<Policy, SftError> {
    let mut policy = Policy::token_era(pilots_to_skip(&archive.join(model.name()).join("data"))?);
    policy.allowed_generators.extend(extra_generators.iter().cloned());
    Ok(policy)
}
