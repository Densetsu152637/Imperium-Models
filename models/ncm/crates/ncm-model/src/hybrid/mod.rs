//! The NCM hybrid: the v2 backbone over L-token blocks and the coarse-to-fine block decoder of
//! Next-Concept Modeling (arXiv 2610.14162). Design of record: the research repository's
//! `docs/next-concept-modelling/architecture.md`; settings in [`crate::HybridConfig`].
//!
//! The reference is plain Burn tensor code ([`decode`]); [`fast`] is the decode path for speed
//! (memory rings, fixed-buffer passes, device acceptance, CUDA graphs).
//!
//! Open TODOs of the fast path:
//!
//! - **One fused kernel per generator layer** (block-diagonal level attention, the joint softmax
//!   over the memory ring and the per-block entries, then the MLP), to bring a level to at most
//!   three launches. Today a layer runs the existing fused norm, RoPE and MLP kernels and Burn's
//!   matmuls; the level tail is one kernel (`kernels/hybrid.rs`). Deferred: a kernel that size
//!   needs runs against the reference to be trusted.
//! - **GPU grammar scan.** The tool-call guard is a byte-level JSON automaton with a bracket stack
//!   and digit counters, so a token × state table does not stay small even at depth 16; a
//!   one-thread walk of the draft's bytes on the GPU would work but gains nothing while the host
//!   reads the draft back anyway (2 + 2L ints) and scans at most L tokens. Revisit when
//!   acceptance, scan and fill all stay on the device.

pub mod decode;
pub mod fast;
pub mod memory;
pub mod model;
pub mod train;
pub mod vq;

pub use decode::{BlockOut, BlockStats, DecodeOptions, Hooks, HybridSession, HybridSnapshot};
#[cfg(feature = "cuda")]
pub(crate) use fast::sample_settings;
pub use fast::{
    BlockPhase, BlockRun, DrawRow, DrawSettings, FastSession, PassGraphs, PassMode, PassResult,
    ar_pass_slots, block_pass_slots,
};
pub use memory::{Memory, RopeTables};
pub use model::{HybridLm, HybridLoss, HybridParts, TrainOut, TrainPlan, TrainTargets};
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
