//! Hub client against a local HTTP server (no external network): listing, resumable and
//! verified downloads, prefix samples, and the file-selection plan.

mod common;

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use ncm_data::hub::{Hub, RepoFile, plan, supports_prefix};
use ncm_data::lock::{Lock, LockEntry};
use sha2::{Digest, Sha256};

const REV: &str = "0123456789abcdef0123456789abcdef01234567";

struct Server {
    port: u16,
    stop: Arc<AtomicBool>,
    served: Arc<AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            h.join().unwrap();
        }
    }
}

fn handle(
    mut stream: TcpStream,
    files: &HashMap<String, Vec<u8>>,
    listing: &str,
    served: &AtomicU64,
) {
    stream.set_nonblocking(false).unwrap();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    reader.read_line(&mut request_line).unwrap();
    let mut range = None;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line.trim().is_empty() {
            break;
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
            let (a, b) = v.trim().split_once('-').unwrap();
            range = Some((a.parse::<usize>().unwrap(), b.parse::<usize>().ok()));
        }
    }
    let path = request_line.split_whitespace().nth(1).unwrap().to_string();
    let mut respond = |status: &str, extra: &str, body: &[u8]| {
        let head = format!(
            "HTTP/1.1 {status}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        stream.write_all(head.as_bytes()).unwrap();
        stream.write_all(body).unwrap();
    };
    if path.starts_with("/api/datasets/x/a/revision/") {
        respond("200 OK", "Content-Type: application/json\r\n", listing.as_bytes());
    } else if let Some(name) = path.strip_prefix(&format!("/datasets/x/a/resolve/{REV}/")) {
        let Some(data) = files.get(name) else {
            return respond("404 Not Found", "", b"");
        };
        match range {
            Some((from, to)) => {
                let end = to.map_or(data.len(), |t| (t + 1).min(data.len()));
                let body = &data[from..end];
                served.fetch_add(body.len() as u64, Ordering::SeqCst);
                let extra = format!("Content-Range: bytes {from}-{}/{}\r\n", end - 1, data.len());
                respond("206 Partial Content", &extra, body);
            }
            None => {
                served.fetch_add(data.len() as u64, Ordering::SeqCst);
                respond("200 OK", "", data);
            }
        }
    } else {
        respond("404 Not Found", "", b"");
    }
}

fn serve(files: HashMap<String, Vec<u8>>, listing: String) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let port = listener.local_addr().unwrap().port();
    let stop = Arc::new(AtomicBool::new(false));
    let served = Arc::new(AtomicU64::new(0));
    let (s2, v2) = (stop.clone(), served.clone());
    let handle = std::thread::spawn(move || {
        while !s2.load(Ordering::SeqCst) {
            match listener.accept() {
                Ok((stream, _)) => handle(stream, &files, &listing, &v2),
                Err(_) => std::thread::sleep(Duration::from_millis(5)),
            }
        }
    });
    Server { port, stop, served, handle: Some(handle) }
}

fn payload(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 31 % 251) as u8).collect()
}

fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn setup(data: &[u8], advertised_sha: &str) -> (Server, Hub, ncm_data::manifest::Source, RepoFile) {
    let listing = format!(
        r#"{{"siblings": [{{"rfilename": "data/part.jsonl", "size": 134, "lfs": {{"sha256": "{advertised_sha}", "size": {}}}}}]}}"#,
        data.len()
    );
    let server = serve(HashMap::from([("data/part.jsonl".to_string(), data.to_vec())]), listing);
    let hub = Hub::new(&format!("http://127.0.0.1:{}", server.port))
        .with_retry_delay(Duration::from_millis(1));
    let mut source = common::single_source("kind = \"text\"");
    source.repo = "x/a".into();
    source.revision = REV.into();
    let file = RepoFile {
        path: "data/part.jsonl".into(),
        size: data.len() as u64,
        sha256: Some(advertised_sha.into()),
    };
    (server, hub, source, file)
}

#[test]
fn lists_a_revision_with_sizes_and_hashes() {
    let data = payload(1000);
    let (_server, hub, _source, _file) = setup(&data, &sha(&data));
    let files = hub.list("x/a", REV).unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].path, "data/part.jsonl");
    assert_eq!(files[0].size, 1000);
    assert_eq!(files[0].sha256.as_deref(), Some(sha(&data).as_str()));
}

#[test]
fn downloads_verify_the_hash_and_skip_files_already_present() {
    let data = payload(300_000);
    let (server, hub, source, file) = setup(&data, &sha(&data));
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("raw").join("s").join("data").join("part.jsonl");

    let entry = hub.download(&source, &file, &dest, None).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    assert!(entry.verified && !entry.partial);
    assert_eq!(entry.bytes, 300_000);
    assert!(!dest.with_file_name("part.jsonl.part").exists());

    let before = server.served.load(Ordering::SeqCst);
    hub.download(&source, &file, &dest, None).unwrap();
    assert_eq!(server.served.load(Ordering::SeqCst), before, "no second download");
}

#[test]
fn resumes_a_partial_copy_without_refetching_it() {
    let data = payload(300_000);
    let (server, hub, source, file) = setup(&data, &sha(&data));
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("part.jsonl");
    std::fs::write(dir.path().join("part.jsonl.part"), &data[..120_000]).unwrap();

    let entry = hub.download(&source, &file, &dest, None).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), data);
    assert!(entry.verified, "the hash covers the resumed prefix too");
    assert_eq!(server.served.load(Ordering::SeqCst), 180_000);
}

#[test]
fn a_hash_mismatch_fails_and_leaves_no_file() {
    let data = payload(5000);
    let (_server, hub, source, file) = setup(&data, &"0".repeat(64));
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("part.jsonl");
    let err = hub.download(&source, &file, &dest, None).unwrap_err().to_string();
    assert!(err.contains("does not match"), "{err}");
    assert!(!dest.exists());
}

#[test]
fn a_prefix_sample_is_marked_partial_and_unverified() {
    let data = payload(50_000);
    let (_server, hub, source, file) = setup(&data, &sha(&data));
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("part.jsonl");
    let entry = hub.download(&source, &file, &dest, Some(1234)).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), &data[..1234]);
    assert!(entry.partial && !entry.verified);
    assert_eq!((entry.bytes, entry.size), (1234, 50_000));
}

fn files(spec: &[(&str, u64)]) -> Vec<RepoFile> {
    spec.iter().map(|(p, s)| RepoFile { path: (*p).into(), size: *s, sha256: None }).collect()
}

#[test]
fn selects_by_glob_stride_and_cap() {
    let listing = files(&[
        ("README.md", 1),
        ("data/a/0.parquet", 10),
        ("data/a/1.parquet", 10),
        ("data/b/0.parquet", 10),
        ("data/b/1.parquet", 10),
        ("data/b/2.parquet", 10),
        ("data/b/deep/9.parquet", 10),
    ]);
    let source = common::single_source(
        "kind = \"text\"\nglob = \"data/*/*.parquet\"\nstride = 2\nmax_files = 2",
    );
    let got: Vec<&str> =
        source.select_files(&listing, |f| &f.path).iter().map(|f| f.path.as_str()).collect();
    // Five matches (the deep file is excluded because `*` does not cross `/`), every second one.
    assert_eq!(got, ["data/a/0.parquet", "data/b/0.parquet"]);
}

#[test]
fn a_byte_limit_shares_out_prefixes_of_line_oriented_files() {
    let listing = files(&[("a.json.gz", 1000), ("b.json.gz", 1000), ("c.json.gz", 100)]);
    let source = common::single_source("kind = \"text\"\nglob = \"*.json.gz\"");
    let tasks = plan(&source, &listing, None, Some(1000));
    let by_name: HashMap<&str, Option<u64>> =
        tasks.iter().map(|t| (t.file.path.as_str(), t.limit)).collect();
    // The small file is whole and gives back what it does not use.
    assert_eq!(by_name["c.json.gz"], None);
    assert_eq!(by_name["a.json.gz"], Some(450));
    assert_eq!(by_name["b.json.gz"], Some(450));
    // Under the limit everything is taken whole.
    assert!(plan(&source, &listing, None, Some(5000)).iter().all(|t| t.limit.is_none()));
}

#[test]
fn a_byte_limit_takes_whole_parquet_files_and_at_least_one() {
    let listing = files(&[("a.parquet", 900), ("b.parquet", 900), ("c.parquet", 900)]);
    let source = common::single_source("kind = \"text\"\nglob = \"*.parquet\"");
    assert_eq!(plan(&source, &listing, None, Some(100)).len(), 1);
    assert_eq!(plan(&source, &listing, None, Some(1800)).len(), 2);
    assert_eq!(plan(&source, &listing, Some(1), None).len(), 1);
    assert!(plan(&source, &listing, None, Some(100)).iter().all(|t| t.limit.is_none()));
    assert!(supports_prefix("x.json.gz") && !supports_prefix("x.parquet"));
}

#[test]
fn the_lock_replaces_entries_and_keeps_order() {
    let entry = |path: &str, bytes: u64| LockEntry {
        source: "s".into(),
        repo: "x/a".into(),
        revision: REV.into(),
        path: path.into(),
        size: 100,
        bytes,
        sha256: None,
        verified: false,
        partial: bytes < 100,
    };
    let mut lock = Lock::default();
    lock.merge(vec![entry("b", 10), entry("a", 100)]);
    lock.merge(vec![entry("b", 100)]);
    let paths: Vec<_> = lock.files.iter().map(|e| (e.path.as_str(), e.bytes)).collect();
    assert_eq!(paths, [("a", 100), ("b", 100)]);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("manifest.lock.json");
    lock.save(&path).unwrap();
    assert_eq!(Lock::load(&path).unwrap().files.len(), 2);
    let mut text = String::new();
    std::fs::File::open(&path).unwrap().read_to_string(&mut text).unwrap();
    assert!(text.contains("\"revision\""));
}
