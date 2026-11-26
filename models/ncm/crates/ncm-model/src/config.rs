//! Model configurations and TOML loading.
//!
//! Field names and defaults are those of the first (PyTorch-era) configs. `attention` picks the
//! attention design and the `[v2]` table holds the settings of the newer one; see
//! `docs/attention.md`.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// The 250M token-baseline config, embedded so binaries need no config directory.
pub const TOK_250M_TOML: &str = include_str!("../../../configs/tok-250m.toml");
/// The 250M NCM config.
pub const NCM_250M_TOML: &str = include_str!("../../../configs/ncm-250m.toml");
/// The 250M token baseline in the v2 attention design.
pub const TOK_250M_V2_TOML: &str = include_str!("../../../configs/tok-250m-v2.toml");
/// The 250M NCM in the v2 attention design.
pub const NCM_250M_V2_TOML: &str = include_str!("../../../configs/ncm-250m-v2.toml");

/// Every `full_every`-th layer (1-based) attends over the whole context.
pub fn is_full_layer(index: usize, full_every: usize) -> bool {
    full_every > 0 && (index + 1).is_multiple_of(full_every)
}

/// Which attention design a model uses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Design {
    /// The first design: sliding-window attention with a full-attention layer every
    /// `full_every`-th layer and grouped-query KV caches.
    #[default]
    Fallback,
    /// Gated DeltaNet alternating with sparse retrieval attention (indexer, top-k blocks, MLA
    /// cache), with output gates and QK-norm. Settings in [`V2Config`].
    V2,
}

impl Design {
    pub fn name(self) -> &'static str {
        match self {
            Self::Fallback => "fallback",
            Self::V2 => "v2",
        }
    }
}

/// Settings of the v2 attention design, shared by both models (`[v2]` in the config files).
/// Fallback models ignore them.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct V2Config {
    /// Gated DeltaNet key and value width per head; heads = layer width / `dn_head_dim`.
    pub dn_head_dim: usize,
    /// Width of the short causal depthwise convolution over q, k and v.
    pub dn_conv: usize,
    /// Chunk length of the chunked (parallel) DeltaNet form.
    pub dn_chunk: usize,
    /// Retrieval attention query heads.
    pub ret_heads: usize,
    /// Per-head query/key width without position; absorbed into the latent for decode.
    pub ret_nope_dim: usize,
    /// Decoupled RoPE key width, shared by all heads (partial RoPE).
    pub ret_rope_dim: usize,
    pub ret_v_dim: usize,
    /// MLA compressed key-value latent per position.
    pub ret_latent: usize,
    pub idx_heads: usize,
    /// Indexer key width.
    pub idx_dim: usize,
    /// Positions per retrievable block.
    pub ret_block: usize,
    /// Blocks each query retrieves.
    pub ret_topk: usize,
    /// Most recent positions every query sees.
    pub ret_local: usize,
    /// Dense attention over the whole causal context (top-k = all): the training warm-up.
    pub ret_dense: bool,
    /// Indexer keys cached as 8-bit codes with one scale per position (the design, as in
    /// DeepSeek-V3.2); `false` keeps them at the run dtype (the parity option).
    pub idx_8bit: bool,
    /// YaRN context extension of the retrieval layers' decoupled RoPE key: positions are
    /// stretched by this factor (1 = plain RoPE, the default for training at 4k tokens).
    pub ret_yarn_factor: f64,
    /// Positions (of the stack: concepts in the NCM backbone, tokens in the baseline) the model
    /// was trained at before YaRN; only read when `ret_yarn_factor` is above 1.
    pub ret_yarn_original: usize,
}

impl Default for V2Config {
    fn default() -> Self {
        Self {
            dn_head_dim: 128,
            dn_conv: 4,
            dn_chunk: 64,
            ret_heads: 8,
            ret_nope_dim: 128,
            ret_rope_dim: 64,
            ret_v_dim: 128,
            ret_latent: 512,
            idx_heads: 4,
            idx_dim: 128,
            ret_block: 64,
            ret_topk: 32,
            ret_local: 1024,
            ret_dense: false,
            idx_8bit: true,
            ret_yarn_factor: 1.0,
            ret_yarn_original: 4096,
        }
    }
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

/// Decoder-only baseline.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct TokConfig {
    pub attention: Design,
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
    /// Full-attention layers; in v2, the retrieval layers' decoupled key.
    pub rope_base_full: f64,
    pub rope_scale: f64,
    pub rope_scale_full: f64,
    pub norm_eps: f64,
    pub v2: V2Config,
}

impl Default for TokConfig {
    fn default() -> Self {
        Self {
            attention: Design::Fallback,
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
            v2: V2Config::default(),
        }
    }
}

/// Hierarchical Next Concept Model with fixed k-token concepts.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct NcmConfig {
    pub attention: Design,
    pub vocab_size: usize,
    pub k: usize,
    pub head_dim: usize,
    /// Encoder, decoder and embedding width.
    pub local_dim: usize,
    /// Sliding-window layers; v2 adds one gated DeltaNet layer after the first.
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
    /// Layers of the parallel draft head on the local decoder, which proposes a whole concept
    /// in one pass for self-speculative decoding (`src/speculative.rs`); at most `dec_layers`.
    /// 0, the default, builds no draft head (configs from before it).
    pub draft_layers: usize,
    pub bb_dim: usize,
    pub bb_layers: usize,
    pub bb_heads: usize,
    pub bb_kv_heads: usize,
    pub bb_ff_dim: usize,
    /// Concepts, sliding layers.
    pub bb_window: usize,
    pub full_every: usize,
    pub rope_base: f64,
    /// Full-attention layers; in v2, the retrieval layers' decoupled key.
    pub rope_base_full: f64,
    pub rope_scale: f64,
    pub rope_scale_full: f64,
    pub norm_eps: f64,
    pub v2: V2Config,
}

impl Default for NcmConfig {
    fn default() -> Self {
        Self {
            attention: Design::Fallback,
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
            draft_layers: 0,
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
            v2: V2Config::default(),
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

    /// The config as TOML, with the `model` key first, in the form `parse` reads back.
    pub fn to_toml(&self) -> String {
        let (kind, table) = match self {
            Self::Tok(c) => ("tok", toml::Table::try_from(c)),
            Self::Ncm(c) => ("ncm", toml::Table::try_from(c)),
        };
        let table = toml::to_string(&table.expect("a config is a table of plain values"));
        format!("model = \"{kind}\"\n{}", table.expect("a config serialises to TOML"))
    }

    pub fn design(&self) -> Design {
        match self {
            Self::Tok(c) => c.attention,
            Self::Ncm(c) => c.attention,
        }
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

fn parse_tok(text: &str) -> TokConfig {
    match Config::parse(text) {
        Ok(Config::Tok(c)) => c,
        _ => unreachable!("the shipped tok configs parse (checked by tests)"),
    }
}

fn parse_ncm(text: &str) -> NcmConfig {
    match Config::parse(text) {
        Ok(Config::Ncm(c)) => c,
        _ => unreachable!("the shipped ncm configs parse (checked by tests)"),
    }
}

/// The 250M token baseline.
pub fn tok_250m() -> TokConfig {
    parse_tok(TOK_250M_TOML)
}

/// The 250M NCM.
pub fn ncm_250m() -> NcmConfig {
    parse_ncm(NCM_250M_TOML)
}

/// The 250M token baseline in the v2 attention design.
pub fn tok_250m_v2() -> TokConfig {
    parse_tok(TOK_250M_V2_TOML)
}

/// The 250M NCM in the v2 attention design.
pub fn ncm_250m_v2() -> NcmConfig {
    parse_ncm(NCM_250M_V2_TOML)
}

/// The 250M token baseline of `design`.
pub fn tok_250m_of(design: Design) -> TokConfig {
    match design {
        Design::Fallback => tok_250m(),
        Design::V2 => tok_250m_v2(),
    }
}

/// The 250M NCM of `design`.
pub fn ncm_250m_of(design: Design) -> NcmConfig {
    match design {
        Design::Fallback => ncm_250m(),
        Design::V2 => ncm_250m_v2(),
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
        draft_layers: 1,
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

/// v2 settings small enough for CPU tests: retrieval over blocks of `block` positions, `topk`
/// of them plus a `local` window, and DeltaNet chunks of 4 so short runs span several chunks.
pub fn tiny_v2(block: usize, topk: usize, local: usize) -> V2Config {
    V2Config {
        dn_head_dim: 16,
        dn_conv: 4,
        dn_chunk: 4,
        ret_heads: 2,
        ret_nope_dim: 16,
        ret_rope_dim: 8,
        ret_v_dim: 16,
        ret_latent: 32,
        idx_heads: 2,
        idx_dim: 16,
        ret_block: block,
        ret_topk: topk,
        ret_local: local,
        ..V2Config::default()
    }
}

/// The tiny token baseline in the v2 design: 4 layers, 2 DeltaNet and 2 retrieval.
pub fn tiny_tok_v2(v2: V2Config) -> TokConfig {
    TokConfig { attention: Design::V2, v2, ..tiny_tok(8) }
}

/// The tiny NCM in the v2 design; the encoder's sliding window is `enc_window` tokens.
pub fn tiny_ncm_v2(enc_window: usize, v2: V2Config) -> NcmConfig {
    NcmConfig { attention: Design::V2, v2, ..tiny_ncm(enc_window, 4) }
}
