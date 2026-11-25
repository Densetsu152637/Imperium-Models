//! Next Concept Model (NCM) and the same-size token baseline, built on Burn.
//!
//! Burn 0.22 picks the compute backend (CUDA, CPU, wgpu) from the `Device` at run time, so
//! every type here works on whichever backends the final binary enables. Tests and CI use the
//! CPU backend; `ncm-bench` adds CUDA behind a cargo feature.
//!
//! Both models come in two attention designs, picked by the config's `attention` key: `v2`
//! (gated DeltaNet and sparse retrieval, the adopted design) and `fallback` (sliding-window
//! attention with periodic full layers). See `docs/attention.md`.

pub mod cache;
pub mod chat;
pub mod config;
pub mod delta;
pub mod fused;
#[cfg(feature = "cuda")]
pub mod kernels;
pub mod layers;
pub mod ncm;
pub mod retrieval;
pub mod tok;
pub mod weights;

pub use cache::{KvCache, LayerCache};
pub use config::{
    Config, ConfigError, Design, NcmConfig, TokConfig, V2Config, ncm_250m, ncm_250m_of,
    ncm_250m_v2, tiny_ncm, tiny_ncm_v2, tiny_tok, tiny_tok_v2, tiny_v2, tok_250m, tok_250m_of,
    tok_250m_v2,
};
pub use ncm::{NcmCache, NcmLm, NcmParts, ncm_cache_bytes};
pub use tok::{TokParts, TokenLm, tok_cache_bytes};

/// Bytes per indexer-key value in the v2 design: 8-bit keys, as in DeepSeek-V3.2. The reference
/// implementation stores them at the run dtype until a fused kernel reads 8-bit keys.
pub const INDEX_KEY_BYTES: usize = 1;

/// Cache bytes of the design at `max_tokens` for either model config, with indexer keys in
/// [`INDEX_KEY_BYTES`] and everything else at `itemsize`.
pub fn cache_bytes(cfg: &Config, max_tokens: usize, itemsize: usize, batch: usize) -> usize {
    cache_bytes_with(cfg, max_tokens, itemsize, INDEX_KEY_BYTES, batch)
}

/// Cache bytes with indexer keys at `index_itemsize`: what the reference allocates when that is
/// the run's float size.
pub fn cache_bytes_with(
    cfg: &Config,
    max_tokens: usize,
    itemsize: usize,
    index_itemsize: usize,
    batch: usize,
) -> usize {
    match cfg {
        Config::Tok(c) => tok_cache_bytes(c, max_tokens, itemsize, index_itemsize, batch),
        Config::Ncm(c) => ncm_cache_bytes(c, max_tokens, itemsize, index_itemsize, batch),
    }
}
