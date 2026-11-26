//! The NCM hybrid: the v2 backbone over L-token blocks and the coarse-to-fine block decoder of
//! Next-Concept Modeling (arXiv 2610.14162). Design of record: the research repository's
//! `docs/next-concept-modelling/architecture.md`; settings in [`crate::HybridConfig`].
//!
//! Plain Burn tensor code (the reference); fused kernels and CUDA graphs come later.

pub mod vq;

pub use vq::{BlockTokenizer, Ema, Quantised, TokenizerOut, sample_keep, tokenizer_loss};
