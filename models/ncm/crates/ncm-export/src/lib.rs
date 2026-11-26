//! `ncm-export`: writes the NCM runtime's per-phase static-shape ONNX graphs (opset 21) with
//! standard QDQ int8 / int4 weights, plus the manifest `ncm-ort` reads (`docs/plans/npu.md`,
//! decision 6).
//!
//! - [`onnx`]: the protobuf schema subset, written by hand on `prost`.
//! - [`graph`]: a small graph builder and the shared weight pool (one external-data file).
//! - [`quant`]: QDQ weights from the repository's own host quantiser.
//! - [`layers`]: builders mirroring the Burn reference: RMSNorm, RoPE, linear, SwiGLU MLP, GQA
//!   attention over a static ring, gated DeltaNet (decode step and chunked prefill with the f32
//!   triangular inverse).
//! - [`model`]: the token baseline's phase graphs and the export driver.
//!
//! **Coverage.** The token baseline in the fallback design (DeltaNet and GQA attention layers)
//! exports completely. The v2 design's sparse-retrieval layers (absorbed MLA, the indexer) and
//! the hybrid's phases (backbone step, block pass, fill, ...) are described by
//! [`model::describe`] but not yet written: [`ExportError::Unsupported`] names what is missing.

pub mod graph;
pub mod layers;
pub mod model;
pub mod onnx;
pub mod quant;
pub mod weights;

pub use model::{ExportOptions, PhasePlan, Support, Target, describe, export};
pub use quant::WeightForm;

/// Why an export failed.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("{0}")]
    Io(String),
    #[error("weights: {0}")]
    Weights(String),
    #[error("shape: {0}")]
    Shape(String),
    #[error("quantisation: {0}")]
    Quant(String),
    #[error("not supported yet: {0}")]
    Unsupported(String),
    #[error("graph `{graph}`: {problem}")]
    Graph { graph: String, problem: String },
    #[error("config: {0}")]
    Config(String),
}
