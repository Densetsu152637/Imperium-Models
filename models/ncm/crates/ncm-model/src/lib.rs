//! Next Concept Model (NCM) and the same-size token baseline, built on Burn.
//!
//! Burn 0.22 picks the compute backend (CUDA, CPU, wgpu) from the `Device` at run time, so
//! every type here works on whichever backends the final binary enables. Tests and CI use the
//! CPU backend; `ncm-bench` adds CUDA behind a cargo feature.

pub mod cache;
pub mod config;
pub mod layers;
pub mod ncm;
pub mod tok;

pub use cache::KvCache;
pub use config::{
    Config, ConfigError, NcmConfig, TokConfig, ncm_250m, tiny_ncm, tiny_tok, tok_250m,
};
pub use ncm::{NcmCache, NcmLm, NcmParts, ncm_cache_bytes};
pub use tok::{TokParts, TokenLm, tok_cache_bytes};

/// KV-cache bytes at `max_tokens` for either model config.
pub fn cache_bytes(cfg: &Config, max_tokens: usize, itemsize: usize, batch: usize) -> usize {
    match cfg {
        Config::Tok(c) => tok_cache_bytes(c, max_tokens, itemsize, batch),
        Config::Ncm(c) => ncm_cache_bytes(c, max_tokens, itemsize, batch),
    }
}
