//! The NCM hybrid: the v2 backbone over L-token blocks and the coarse-to-fine block decoder of
//! Next-Concept Modeling (arXiv 2610.14162). Design of record: the research repository's
//! `docs/next-concept-modelling/architecture.md`; settings in [`crate::HybridConfig`].
//!
//! Plain Burn tensor code (the reference); fused kernels and CUDA graphs come later.

pub mod decode;
pub mod fast;
pub mod memory;
pub mod model;
pub mod vq;

pub use decode::{BlockOut, BlockStats, DecodeOptions, Hooks, HybridSession, HybridSnapshot};
pub use fast::{FastSession, PassMode, PassResult};
pub use memory::Memory;
pub use model::{HybridLm, HybridLoss, HybridParts, TrainOut, TrainPlan};
pub use vq::{BlockTokenizer, Ema, Quantised, TokenizerOut, sample_keep, tokenizer_loss};

use crate::config::NcmConfig;

/// Decode-state bytes of the hybrid at `max_tokens`: the backbone's caches over block
/// positions (indexer keys at `index_itemsize`, the rest at `itemsize`), the encoder's, the
/// recent encoder states plus one open block, and the token ids of the whole context (4 bytes
/// each, for retrieved tokens).
pub fn hybrid_cache_bytes(
    cfg: &NcmConfig,
    max_tokens: usize,
    itemsize: usize,
    index_itemsize: usize,
) -> usize {
    let h = cfg.hybrid.as_ref().expect("a hybrid config");
    let (encoder, backbone) = crate::ncm::ncm_specs(cfg);
    let blocks = max_tokens.div_ceil(h.block);
    let backbone = backbone.cache_bytes(blocks, itemsize, index_itemsize, 1);
    let encoder = encoder.cache_bytes(max_tokens, itemsize, index_itemsize, 1);
    let states = (h.recent + h.block) * cfg.local_dim * itemsize;
    backbone + encoder + states + 4 * max_tokens
}
