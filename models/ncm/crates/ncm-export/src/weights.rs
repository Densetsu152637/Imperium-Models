//! Reading a weight file for export: the safetensors header parsed directly (no tensors are
//! decoded until asked for), f32, bf16 and f16 tensors converted to f32 on read.
//!
//! Tensor names are Burn module paths (`stack.blocks.0.n1.gamma`, ...). A module enum may or
//! may not put its variant in the path (`attn.DeltaNet.proj.weight` or `attn.proj.weight`), so
//! lookups take several candidate names ([`Weights::find`]).

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::ExportError;

/// One tensor's place in the file.
#[derive(Clone, Debug)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// A weight file's header and bytes.
pub struct Weights {
    entries: HashMap<String, Entry>,
    data: Vec<u8>,
    /// The `__metadata__` map.
    pub metadata: HashMap<String, String>,
}

/// A tensor read as f32.
#[derive(Clone, Debug, PartialEq)]
pub struct F32Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self, ExportError> {
        let bytes =
            std::fs::read(path).map_err(|e| ExportError::Io(format!("{}: {e}", path.display())))?;
        Self::from_bytes(bytes, &path.display().to_string())
    }

    pub fn from_bytes(bytes: Vec<u8>, what: &str) -> Result<Self, ExportError> {
        let bad = |why: &str| ExportError::Weights(format!("{what}: {why}"));
        let len: [u8; 8] =
            bytes.get(..8).ok_or_else(|| bad("shorter than a header"))?.try_into().unwrap();
        let header_len = u64::from_le_bytes(len) as usize;
        let header = bytes.get(8..8 + header_len).ok_or_else(|| bad("the header is truncated"))?;
        let header: Value = serde_json::from_slice(header).map_err(|e| bad(&e.to_string()))?;
        let object = header.as_object().ok_or_else(|| bad("the header is not an object"))?;
        let mut entries = HashMap::new();
        let mut metadata = HashMap::new();
        for (name, value) in object {
            if name == "__metadata__" {
                for (k, v) in value.as_object().into_iter().flatten() {
                    if let Some(v) = v.as_str() {
                        metadata.insert(k.clone(), v.to_owned());
                    }
                }
                continue;
            }
            let dtype = value["dtype"].as_str().ok_or_else(|| bad("a tensor has no dtype"))?;
            let shape: Vec<usize> = value["shape"]
                .as_array()
                .ok_or_else(|| bad("a tensor has no shape"))?
                .iter()
                .map(|d| d.as_u64().unwrap_or(0) as usize)
                .collect();
            let offsets = value["data_offsets"].as_array().ok_or_else(|| bad("no data_offsets"))?;
            let (start, end) = (
                offsets.first().and_then(Value::as_u64).unwrap_or(0) as usize,
                offsets.get(1).and_then(Value::as_u64).unwrap_or(0) as usize,
            );
            entries.insert(name.clone(), Entry { dtype: dtype.to_owned(), shape, start, end });
        }
        let data = bytes[8 + header_len..].to_vec();
        Ok(Self { entries, data, metadata })
    }

    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Names, sorted.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.entries.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// The tensor `name` as f32.
    pub fn get(&self, name: &str) -> Result<F32Tensor, ExportError> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| ExportError::Weights(format!("the weight file has no `{name}`")))?;
        let raw = self
            .data
            .get(e.start..e.end)
            .ok_or_else(|| ExportError::Weights(format!("`{name}` is outside the file")))?;
        let data: Vec<f32> = match e.dtype.as_str() {
            "F32" => raw.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
            "BF16" => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
                .collect(),
            "F16" => {
                raw.as_chunks::<2>().0.iter().map(|b| f16_to_f32(u16::from_le_bytes(*b))).collect()
            }
            other => {
                return Err(ExportError::Weights(format!(
                    "`{name}` is {other}: only dense f32, bf16 and f16 weight files export \
                     (a quantised file holds packed codes; export from the dense file with --quant)"
                )));
            }
        };
        Ok(F32Tensor { shape: e.shape.clone(), data })
    }

    /// The first of `names` the file has.
    pub fn find(&self, names: &[String]) -> Result<F32Tensor, ExportError> {
        for n in names {
            if self.has(n) {
                return self.get(n);
            }
        }
        Err(ExportError::Weights(format!("the weight file has none of {names:?}")))
    }
}

/// IEEE half to single.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign << 31,
        (0, f) => {
            // Subnormal: normalise.
            let mut e = 127 - 15 + 1;
            let mut f = f;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((f & 0x3FF) << 13)
        }
        (0x1F, 0) => (sign << 31) | 0x7F80_0000,
        (0x1F, f) => (sign << 31) | 0x7F80_0000 | (f << 13),
        (e, f) => (sign << 31) | ((e + 127 - 15) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}

/// Builds a safetensors file in memory (f32 tensors), for tests and synthetic exports.
pub fn safetensors_bytes(
    tensors: &[(&str, &[usize], &[f32])],
    metadata: &[(&str, &str)],
) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        let m: serde_json::Map<String, Value> =
            metadata.iter().map(|(k, v)| (k.to_string(), Value::String(v.to_string()))).collect();
        header.insert("__metadata__".into(), Value::Object(m));
    }
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        data.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let header = serde_json::to_vec(&Value::Object(header)).expect("a header");
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header);
    out.extend(data);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_what_it_wrote() {
        let bytes = safetensors_bytes(
            &[("a.weight", &[2, 2], &[1.0, 2.0, 3.0, 4.0]), ("b", &[3], &[0.5, 0.25, 0.125])],
            &[("ncm_format", "x")],
        );
        let w = Weights::from_bytes(bytes, "mem").unwrap();
        assert_eq!(w.names(), vec!["a.weight", "b"]);
        assert_eq!(w.get("a.weight").unwrap().shape, vec![2, 2]);
        assert_eq!(w.get("b").unwrap().data, vec![0.5, 0.25, 0.125]);
        assert_eq!(w.metadata["ncm_format"], "x");
        assert!(w.get("missing").is_err());
        let found = w.find(&["nope".into(), "b".into()]).unwrap();
        assert_eq!(found.data.len(), 3);
    }

    #[test]
    fn half_floats_convert() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12);
    }
}
