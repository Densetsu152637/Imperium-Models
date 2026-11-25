//! The deterministic train/validation split: a hash of the record id, so a record keeps its
//! side when others are added or removed.

use serde::Serialize;
use sha2::{Digest, Sha256};

/// Share of records that go to validation.
pub const VAL_FRACTION: f64 = 0.02;

const SALT: &str = "ncm-sft-split-v1:";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Val,
}

/// A number in [0, 1) derived from the id.
pub fn bucket(id: &str) -> f64 {
    let digest = Sha256::digest(format!("{SALT}{id}").as_bytes());
    let head = u64::from_be_bytes(digest[..8].try_into().expect("a SHA-256 digest has 32 bytes"));
    head as f64 / 18_446_744_073_709_551_616.0
}

pub fn split_of(id: &str) -> Split {
    if bucket(id) < VAL_FRACTION { Split::Val } else { Split::Train }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_split_is_deterministic() {
        for id in ["p-b63df6fdaf7c", "ct-report-fa91a056e6", "x"] {
            assert_eq!(split_of(id), split_of(id));
            assert_eq!(bucket(id), bucket(id));
        }
        // Pinned values: changing the hash or salt would silently move records across the
        // split, so a change here must be deliberate.
        assert_eq!(format!("{:.6}", bucket("p-b63df6fdaf7c")), PINNED);
    }

    const PINNED: &str = "0.532454";

    #[test]
    fn about_two_percent_go_to_validation() {
        let val = (0..20_000).filter(|i| split_of(&format!("id-{i}")) == Split::Val).count();
        assert!((300..=500).contains(&val), "{val} of 20000");
    }
}
