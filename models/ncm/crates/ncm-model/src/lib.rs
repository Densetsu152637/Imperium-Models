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
pub(crate) mod graph;
#[cfg(feature = "cuda")]
pub mod kernels;
pub mod layers;
pub mod ncm;
pub mod quant;
pub mod retrieval;
pub mod speculative;
pub mod tok;
pub mod weights;

pub use cache::{KvCache, LayerCache};
pub use config::{
    Config, ConfigError, Design, NcmConfig, TokConfig, V2Config, ncm_250m, ncm_250m_of,
    ncm_250m_v2, tiny_ncm, tiny_ncm_v2, tiny_tok, tiny_tok_v2, tiny_v2, tok_250m, tok_250m_of,
    tok_250m_v2,
};
pub use ncm::{NcmCache, NcmLm, NcmParts, ncm_cache_bytes};
pub use speculative::{Acceptance, DraftLoss, SpecStats, Speculator, draft_loss};
pub use tok::{TokParts, TokenLm, tok_cache_bytes};

/// Bytes per indexer-key value in the v2 design: 8-bit keys, as in DeepSeek-V3.2, with one scale
/// per position at the run dtype (`V2Config::idx_8bit`).
pub const INDEX_KEY_BYTES: usize = 1;

/// Cache bytes of the design at `max_tokens` for either model config: indexer keys in
/// [`INDEX_KEY_BYTES`] plus a scale when the config caches them in 8 bits (the default),
/// otherwise at `itemsize` like everything else. What the cache allocates.
pub fn cache_bytes(cfg: &Config, max_tokens: usize, itemsize: usize, batch: usize) -> usize {
    let v2 = match cfg {
        Config::Tok(c) => &c.v2,
        Config::Ncm(c) => &c.v2,
    };
    let index = if v2.idx_8bit { INDEX_KEY_BYTES } else { itemsize };
    cache_bytes_with(cfg, max_tokens, itemsize, index, batch)
}

/// Cache bytes with indexer keys at `index_itemsize` per value (narrower than `itemsize`: 8-bit,
/// plus one scale per position at `itemsize`), whatever the config says.
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
