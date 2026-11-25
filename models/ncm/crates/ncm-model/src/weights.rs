//! Weight files: the on-disk format shared by training, evaluation and the `ncm-server` runtime.
//!
//! A weight file is one [safetensors](https://github.com/huggingface/safetensors) file written
//! by Burn's `SafetensorsStore`. Tensor names are Burn module paths (`emb.weight`,
//! `backbone.layers.0.attn.q.weight`, ...), so a file only loads into the same architecture.
//! The header's `__metadata__` map carries `ncm_format` (the format tag, [`FORMAT`]) and
//! `ncm_config` (the model config as TOML, as `Config::to_toml` writes it), so a weight file
//! says which architecture it is without a separate config file. Tensors keep the dtype they
//! were saved in (bf16 from training); loading casts them to the dtype the run uses.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use burn::module::{Module, ModuleMapper, Param};
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::{FloatDType, Tensor};

use crate::config::{Config, ConfigError};

/// Metadata key holding the format tag.
pub const FORMAT_KEY: &str = "ncm_format";
/// Metadata key holding the model config as TOML.
pub const CONFIG_KEY: &str = "ncm_config";
/// The format tag written to `ncm_format`.
pub const FORMAT: &str = "ncm-safetensors-v1";
/// Headers larger than this are not safetensors headers.
const MAX_HEADER_BYTES: u64 = 64 << 20;

#[derive(Debug, thiserror::Error)]
pub enum WeightsError {
    #[error("{path}: {source}")]
    Io { path: String, source: std::io::Error },
    #[error("{path} is not a safetensors file: {reason}")]
    Header { path: String, reason: String },
    #[error("{path}: {message}")]
    Store { path: String, message: String },
    #[error("{path} does not match the model: {details}")]
    Mismatch { path: String, details: String },
    #[error("{path} has an invalid `ncm_config`: {source}")]
    Config { path: String, source: ConfigError },
}

fn io_error(path: &Path, source: std::io::Error) -> WeightsError {
    WeightsError::Io { path: path.display().to_string(), source }
}

/// Saves `module` to `path` with `config` in the header. An existing file is replaced.
pub fn save_weights<M: Module>(
    module: &M,
    config: &Config,
    path: &Path,
) -> Result<(), WeightsError> {
    let mut store = SafetensorsStore::from_file(path)
        .overwrite(true)
        .metadata(FORMAT_KEY, FORMAT)
        .metadata(CONFIG_KEY, config.to_toml());
    module.save_into(&mut store).map_err(|e| WeightsError::Store {
        path: path.display().to_string(),
        message: e.to_string(),
    })
}

/// The header's `__metadata__` map, read without touching the tensors.
pub fn read_metadata(path: &Path) -> Result<HashMap<String, String>, WeightsError> {
    let header_error = |reason: &str| WeightsError::Header {
        path: path.display().to_string(),
        reason: reason.into(),
    };
    let mut file = std::fs::File::open(path).map_err(|e| io_error(path, e))?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len).map_err(|_| header_error("shorter than a header"))?;
    let len = u64::from_le_bytes(len);
    if len > MAX_HEADER_BYTES {
        return Err(header_error("the header length is implausible"));
    }
    let mut header = vec![0u8; len as usize];
    file.read_exact(&mut header).map_err(|_| header_error("the header is truncated"))?;
    let header: serde_json::Value =
        serde_json::from_slice(&header).map_err(|e| header_error(&e.to_string()))?;
    let metadata = header.get("__metadata__").and_then(|m| m.as_object());
    Ok(metadata
        .into_iter()
        .flatten()
        .filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_owned())))
        .collect())
}

/// The model config stored in the weight file's header, if it has one.
pub fn read_config(path: &Path) -> Result<Option<Config>, WeightsError> {
    let Some(text) = read_metadata(path)?.remove(CONFIG_KEY) else { return Ok(None) };
    Config::parse(&text)
        .map(Some)
        .map_err(|source| WeightsError::Config { path: path.display().to_string(), source })
}

/// Casts every float parameter to one dtype.
struct CastFloats(FloatDType);

impl ModuleMapper for CastFloats {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        param.map(|tensor| tensor.cast(self.0))
    }
}

/// Loads the weights at `path` into `module` (built with the same config) and casts the float
/// parameters to `dtype`. Every tensor must match: a missing, unused or mismatched tensor is an
/// error, so a file from another architecture never loads half-way.
pub fn load_weights<M: Module>(
    mut module: M,
    path: &Path,
    dtype: FloatDType,
) -> Result<M, WeightsError> {
    let shown = path.display().to_string();
    let mut store = SafetensorsStore::from_file(path);
    let result = module
        .load_from(&mut store)
        .map_err(|e| WeightsError::Store { path: shown.clone(), message: e.to_string() })?;
    if !result.is_success() || !result.unused.is_empty() {
        return Err(WeightsError::Mismatch { path: shown, details: result.to_string() });
    }
    Ok(module.map(&mut CastFloats(dtype)))
}
