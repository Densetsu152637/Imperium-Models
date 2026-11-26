//! Fingerprints the crate's sources for the on-disk kernel cache (`kernels::enable_kernel_cache`):
//! cached kernels are keyed by name and launch settings, not by their code, so a cache written
//! by another build must never be read.

use std::path::Path;

fn visit(dir: &Path, hash: &mut u64) {
    let mut entries: Vec<_> = std::fs::read_dir(dir).into_iter().flatten().flatten().collect();
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            visit(&path, hash);
        } else if let Ok(bytes) = std::fs::read(&path) {
            for b in path.to_string_lossy().bytes().chain(bytes) {
                // FNV-1a, 64-bit.
                *hash ^= u64::from(b);
                *hash = hash.wrapping_mul(0x100_0000_01b3);
            }
        }
    }
}

fn main() {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    visit(Path::new("src"), &mut hash);
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rustc-env=NCM_SOURCE_HASH={hash:016x}");
}
