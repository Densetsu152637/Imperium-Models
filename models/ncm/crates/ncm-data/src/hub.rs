//! Hugging Face Hub client: list a dataset revision and download its files over HTTPS.
//!
//! Downloads are resumable (`<file>.part` plus a `Range` request), checked against the SHA-256
//! the Hub advertises for LFS files, and sent with a generic User-Agent. No token is used:
//! gated datasets are out of scope.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::error::{Error, IoContext, Result};
use crate::lock::LockEntry;
use crate::manifest::Source;

pub const ENDPOINT: &str = "https://huggingface.co";
const USER_AGENT: &str = "ncm-data/0.0 (open dataset fetch)";
const ATTEMPTS: u32 = 4;

#[derive(Debug, Clone)]
pub struct RepoFile {
    pub path: String,
    pub size: u64,
    pub sha256: Option<String>,
}

#[derive(Deserialize)]
struct Listing {
    siblings: Vec<Sibling>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
    size: Option<u64>,
    lfs: Option<Lfs>,
}

#[derive(Deserialize)]
struct Lfs {
    sha256: String,
    size: u64,
}

pub struct Hub {
    agent: ureq::Agent,
    endpoint: String,
    retry_delay: Duration,
}

impl Default for Hub {
    fn default() -> Self {
        Self::new(ENDPOINT)
    }
}

impl Hub {
    pub fn new(endpoint: &str) -> Self {
        let config = ureq::Agent::config_builder()
            .user_agent(USER_AGENT)
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(120)))
            .build();
        Self {
            agent: config.into(),
            endpoint: endpoint.trim_end_matches('/').to_string(),
            retry_delay: Duration::from_secs(2),
        }
    }

    /// Base delay between attempts; it doubles each retry.
    pub fn with_retry_delay(mut self, delay: Duration) -> Self {
        self.retry_delay = delay;
        self
    }

    /// Every file of the dataset at `revision`, with sizes and LFS hashes.
    pub fn list(&self, repo: &str, revision: &str) -> Result<Vec<RepoFile>> {
        let url = format!("{}/api/datasets/{repo}/revision/{revision}?blobs=true", self.endpoint);
        let resp = self
            .agent
            .get(&url)
            .call()
            .map_err(|e| Error::Net(format!("listing {repo}@{revision}: {e}")))?;
        let listing: Listing = serde_json::from_reader(resp.into_body().into_reader())?;
        Ok(listing
            .siblings
            .into_iter()
            .map(|s| RepoFile {
                size: s.lfs.as_ref().map(|l| l.size).or(s.size).unwrap_or(0),
                sha256: s.lfs.map(|l| l.sha256),
                path: s.rfilename,
            })
            .collect())
    }

    fn file_url(&self, repo: &str, revision: &str, path: &str) -> String {
        format!("{}/datasets/{repo}/resolve/{revision}/{path}", self.endpoint)
    }

    /// Download one file (or its first `limit` bytes) to `dest`, resuming a partial copy.
    pub fn download(
        &self,
        source: &Source,
        file: &RepoFile,
        dest: &Path,
        limit: Option<u64>,
    ) -> Result<LockEntry> {
        let mut last = None;
        for attempt in 0..ATTEMPTS {
            if attempt > 0 {
                std::thread::sleep(self.retry_delay * (1 << attempt));
            }
            match self.download_once(source, file, dest, limit) {
                Ok(entry) => return Ok(entry),
                Err(e) => {
                    eprintln!(
                        "  retry {}/{}: {} (attempt {}): {e}",
                        source.name,
                        file.path,
                        attempt + 1,
                        ATTEMPTS
                    );
                    last = Some(e);
                }
            }
        }
        Err(last.expect("at least one attempt"))
    }

    fn download_once(
        &self,
        source: &Source,
        file: &RepoFile,
        dest: &Path,
        limit: Option<u64>,
    ) -> Result<LockEntry> {
        let target = limit.map_or(file.size, |l| l.min(file.size));
        let partial = target < file.size;
        let entry = |verified: bool| LockEntry {
            source: source.name.clone(),
            repo: source.repo.clone(),
            revision: source.revision.clone(),
            path: file.path.clone(),
            size: file.size,
            bytes: target,
            sha256: file.sha256.clone(),
            verified,
            partial,
        };
        if let Some(dir) = dest.parent() {
            std::fs::create_dir_all(dir).ctx(format!("creating {}", dir.display()))?;
        }
        let part = part_path(dest);
        if let Ok(meta) = std::fs::metadata(dest) {
            if meta.len() == target {
                // Complete from an earlier run. The lock records whether it was verified.
                return Ok(entry(!partial && file.sha256.is_some()));
            }
            // A shorter earlier copy (a sample prefix) can be extended.
            if meta.len() < target && !part.exists() {
                std::fs::rename(dest, &part).ctx("resuming from a shorter copy")?;
            }
        }

        let mut hasher = Sha256::new();
        let mut have = std::fs::metadata(&part).map_or(0, |m| m.len());
        if have > target {
            have = 0;
        }
        let mut out = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(have == 0)
            .open(&part)
            .ctx(format!("opening {}", part.display()))?;
        if have > 0 {
            let mut existing = File::open(&part).ctx("reading the partial copy")?;
            let mut left = have;
            let mut buf = vec![0u8; 1 << 20];
            while left > 0 {
                let want = left.min(buf.len() as u64) as usize;
                let n = existing.read(&mut buf[..want]).ctx("hashing the partial copy")?;
                if n == 0 {
                    return Err(Error::Net("partial copy shrank while hashing".into()));
                }
                hasher.update(&buf[..n]);
                left -= n as u64;
            }
        }
        out.seek(std::io::SeekFrom::Start(have)).ctx("seeking")?;

        let started = Instant::now();
        if have < target {
            let url = self.file_url(&source.repo, &source.revision, &file.path);
            let mut req = self.agent.get(&url);
            req = req.header("Range", format!("bytes={have}-{}", target - 1));
            let resp = req.call().map_err(|e| Error::Net(format!("GET {}: {e}", file.path)))?;
            if resp.status().as_u16() == 200 && have > 0 {
                // The server ignored the range: start over from the top of the body.
                hasher = Sha256::new();
                have = 0;
                out.set_len(0).ctx("truncating")?;
                out.seek(std::io::SeekFrom::Start(0)).ctx("seeking")?;
            }
            let mut body = resp.into_body().into_reader().take(target - have);
            let mut buf = vec![0u8; 1 << 18];
            let mut written = have;
            loop {
                let n = body
                    .read(&mut buf)
                    .map_err(|e| Error::Net(format!("reading {}: {e}", file.path)))?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
                out.write_all(&buf[..n]).ctx("writing")?;
                written += n as u64;
            }
            out.flush().ctx("flushing")?;
            if written != target {
                return Err(Error::Net(format!("{}: got {written} of {target} bytes", file.path)));
            }
            let secs = started.elapsed().as_secs_f64().max(0.001);
            eprintln!(
                "  got {}/{}: {:.1} MB at {:.1} MB/s",
                source.name,
                file.path,
                (target - have) as f64 / 1e6,
                (target - have) as f64 / 1e6 / secs
            );
        }
        drop(out);

        let mut verified = false;
        if !partial && let Some(want) = &file.sha256 {
            let got = hex::encode(hasher.finalize());
            if &got != want {
                let _ = std::fs::remove_file(&part);
                return Err(Error::Net(format!(
                    "{}: sha256 {got} does not match the Hub's {want}",
                    file.path
                )));
            }
            verified = true;
        }
        std::fs::rename(&part, dest).ctx(format!("finishing {}", dest.display()))?;
        Ok(entry(verified))
    }
}

fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    dest.with_file_name(name)
}

/// Local copy of a repo file: `<data>/raw/<source>/<path>`.
pub fn local_path(data_dir: &Path, source: &str, path: &str) -> PathBuf {
    data_dir.join("raw").join(source).join(path)
}

/// Whether a prefix of the file is still readable (line-oriented JSON, optionally gzipped).
pub fn supports_prefix(path: &str) -> bool {
    [".jsonl", ".json", ".jsonl.gz", ".json.gz", ".ndjson"].iter().any(|ext| path.ends_with(ext))
}

/// One planned download.
#[derive(Debug, Clone)]
pub struct Task {
    pub file: RepoFile,
    pub limit: Option<u64>,
}

/// Pick files for `source` from the repo listing, honouring the optional CLI caps.
///
/// `limit_bytes` is per source. When every selected file is line-oriented, the budget is shared
/// out evenly (smaller files give back what they do not use) and each file contributes a prefix,
/// so a file-grouped source such as code by language stays stratified. Parquet cannot be cut:
/// whole files are taken in order until the budget is spent, and the first is taken even if it
/// alone overshoots.
pub fn plan(
    source: &Source,
    listing: &[RepoFile],
    max_files: Option<usize>,
    limit_bytes: Option<u64>,
) -> Vec<Task> {
    let mut chosen = source.select_files(listing, |f| &f.path);
    if let Some(n) = max_files {
        chosen.truncate(n);
    }
    let whole = |f: &RepoFile| Task { file: f.clone(), limit: None };
    let Some(limit) = limit_bytes else {
        return chosen.into_iter().map(whole).collect();
    };
    if chosen.iter().map(|f| f.size).sum::<u64>() <= limit {
        return chosen.into_iter().map(whole).collect();
    }
    if chosen.iter().all(|f| supports_prefix(&f.path)) {
        let mut order: Vec<usize> = (0..chosen.len()).collect();
        order.sort_by_key(|&i| chosen[i].size);
        let mut left = limit;
        let mut give = vec![0u64; chosen.len()];
        for (k, &i) in order.iter().enumerate() {
            let fair = left / (order.len() - k) as u64;
            give[i] = chosen[i].size.min(fair);
            left -= give[i];
        }
        return chosen
            .iter()
            .zip(give)
            .filter(|(_, g)| *g > 0)
            .map(|(f, g)| Task { file: (*f).clone(), limit: (g < f.size).then_some(g) })
            .collect();
    }
    let mut tasks = Vec::new();
    let mut spent = 0u64;
    for f in chosen {
        if spent >= limit || (!tasks.is_empty() && spent + f.size > limit) {
            break;
        }
        spent += f.size;
        tasks.push(whole(f));
    }
    tasks
}

/// Run downloads with at most `jobs` at a time. Finished entries are returned even when some
/// downloads failed; the errors come back alongside.
pub fn fetch_all(
    hub: &Hub,
    data_dir: &Path,
    work: &[(&Source, Task)],
    jobs: usize,
) -> (Vec<LockEntry>, Vec<String>) {
    let next = AtomicUsize::new(0);
    let done = Mutex::new(Vec::new());
    let failed = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..jobs.clamp(1, work.len().max(1)) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    let Some((source, task)) = work.get(i) else {
                        break;
                    };
                    let dest = local_path(data_dir, &source.name, &task.file.path);
                    eprintln!(
                        "fetch {}/{} ({:.1} MB)",
                        source.name,
                        task.file.path,
                        task.limit.unwrap_or(task.file.size) as f64 / 1e6
                    );
                    match hub.download(source, &task.file, &dest, task.limit) {
                        Ok(entry) => done.lock().expect("lock").push(entry),
                        Err(e) => failed
                            .lock()
                            .expect("lock")
                            .push(format!("{}/{}: {e}", source.name, task.file.path)),
                    }
                }
            });
        }
    });
    (done.into_inner().expect("lock"), failed.into_inner().expect("lock"))
}
