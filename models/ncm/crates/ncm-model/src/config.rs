//! Model configurations and TOML loading.
//!
//! The same `configs/*.toml` files are read by the PyTorch reference and by this crate, so the
//! field names and defaults mirror `src/ncm/config.py`.

use std::path::Path;

use serde::Deserialize;

/// The 250M token-baseline config, embedded so binaries need no config directory.
pub const TOK_250M_TOML: &str = include_str!("../../../configs/tok-250m.toml");
/// The 250M NCM config.
pub const NCM_250M_TOML: &str = include_str!("../../../configs/ncm-250m.toml");

/// Every `full_every`-th layer (1-based) attends over the whole context.
pub fn is_full_layer(index: usize, full_every: usize) -> bool {
    full_every > 0 && (index + 1).is_multiple_of(full_every)
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("cannot read config {path}: {source}")]
    Read { path: String, source: std::io::Error },
    #[error("invalid config TOML: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("config has no `model` key")]
    MissingModel,
    #[error("unknown model kind `{0}` (expected `tok` or `ncm`)")]
    UnknownModel(String),
}

/// Plain decoder-only baseline.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TokConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ff_dim: usize,
    /// Tokens, sliding layers.
    pub window: usize,
    pub full_every: usize,
    pub rope_base: f64,
    pub rope_base_full: f64,
    pub rope_scale: f64,
    pub rope_scale_full: f64,
    pub norm_eps: f64,
}

impl Default for TokConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 1024,
            n_layers: 20,
            n_heads: 16,
            n_kv_heads: 4,
            head_dim: 64,
            ff_dim: 2688,
            window: 4096,
            full_every: 4,
            rope_base: 10_000.0,
            rope_base_full: 1_000_000.0,
            rope_scale: 1.0,
            rope_scale_full: 1.0,
            norm_eps: 1e-6,
        }
    }
}

/// Hierarchical Next Concept Model with fixed k-token concepts.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NcmConfig {
    pub vocab_size: usize,
    pub k: usize,
    pub head_dim: usize,
    /// Encoder, decoder and embedding width.
    pub local_dim: usize,
    pub enc_layers: usize,
    pub enc_heads: usize,
    pub enc_kv_heads: usize,
    pub enc_ff_dim: usize,
    /// Tokens; the decoder cross-attends to the same window.
    pub enc_window: usize,
    pub dec_layers: usize,
    pub dec_heads: usize,
    pub dec_kv_heads: usize,
    pub dec_ff_dim: usize,
    pub bb_dim: usize,
    pub bb_layers: usize,
    pub bb_heads: usize,
    pub bb_kv_heads: usize,
    pub bb_ff_dim: usize,
    /// Concepts, sliding layers.
    pub bb_window: usize,
    pub full_every: usize,
    pub rope_base: f64,
    pub rope_base_full: f64,
    pub rope_scale: f64,
    pub rope_scale_full: f64,
    pub norm_eps: f64,
}

impl Default for NcmConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            k: 4,
            head_dim: 64,
            local_dim: 512,
            enc_layers: 2,
            enc_heads: 8,
            enc_kv_heads: 4,
            enc_ff_dim: 1536,
            enc_window: 512,
            dec_layers: 4,
            dec_heads: 8,
            dec_kv_heads: 4,
            dec_ff_dim: 1536,
            bb_dim: 1024,
            bb_layers: 16,
            bb_heads: 16,
            bb_kv_heads: 4,
            bb_ff_dim: 3200,
            bb_window: 1024,
            full_every: 4,
            rope_base: 10_000.0,
            rope_base_full: 1_000_000.0,
            rope_scale: 1.0,
            rope_scale_full: 1.0,
            norm_eps: 1e-6,
        }
    }
}

/// Either model configuration; the `model` key of a config file picks the variant.
#[derive(Debug, Clone, PartialEq)]
pub enum Config {
    Tok(TokConfig),
    Ncm(NcmConfig),
}

impl Config {
    pub fn parse(text: &str) -> Result<Self, ConfigError> {
        let mut table: toml::Table = toml::from_str(text)?;
        let kind = match table.remove("model") {
            Some(toml::Value::String(kind)) => kind,
            _ => return Err(ConfigError::MissingModel),
        };
        match kind.as_str() {
            "tok" => Ok(Self::Tok(table.try_into()?)),
            "ncm" => Ok(Self::Ncm(table.try_into()?)),
            _ => Err(ConfigError::UnknownModel(kind)),
        }
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|source| ConfigError::Read { path: path.display().to_string(), source })?;
        Self::parse(&text)
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Tok(c) => c.vocab_size,
            Self::Ncm(c) => c.vocab_size,
        }
    }

    /// Tokens per decode unit: 1 for the baseline, `k` for the NCM.
    pub fn unit(&self) -> usize {
        match self {
            Self::Tok(_) => 1,
            Self::Ncm(c) => c.k,
        }
    }
}

/// The 250M token baseline.
pub fn tok_250m() -> TokConfig {
    match Config::parse(TOK_250M_TOML) {
        Ok(Config::Tok(c)) => c,
        _ => unreachable!("configs/tok-250m.toml is a tok config (checked by tests)"),
    }
}

/// The 250M NCM.
pub fn ncm_250m() -> NcmConfig {
    match Config::parse(NCM_250M_TOML) {
        Ok(Config::Ncm(c)) => c,
        _ => unreachable!("configs/ncm-250m.toml is an ncm config (checked by tests)"),
    }
}

/// Small config for CPU tests and smoke runs; `window` is short enough to wrap.
pub fn tiny_tok(window: usize) -> TokConfig {
    TokConfig {
        vocab_size: 256,
        d_model: 64,
        n_layers: 4,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 16,
        ff_dim: 128,
        window,
        full_every: 2,
        ..TokConfig::default()
    }
}

pub fn tiny_ncm(enc_window: usize, bb_window: usize) -> NcmConfig {
    NcmConfig {
        vocab_size: 256,
        k: 4,
        head_dim: 16,
        local_dim: 32,
        enc_layers: 2,
        enc_heads: 2,
        enc_kv_heads: 1,
        enc_ff_dim: 64,
        enc_window,
        dec_layers: 2,
        dec_heads: 2,
        dec_kv_heads: 1,
        dec_ff_dim: 64,
        bb_dim: 64,
        bb_layers: 4,
        bb_heads: 4,
        bb_kv_heads: 2,
        bb_ff_dim: 128,
        bb_window,
        full_every: 2,
        ..NcmConfig::default()
    }
}
