//! The backend-agnostic engine interface of the NCM runtime (`docs/plans/npu.md`, decision 1).
//!
//! This crate depends on nothing model-specific: no Burn, no ONNX Runtime. The dependency
//! direction is `ncm-model -> ncm-exec` (the Burn adapter lives there), `ncm-ort -> ncm-exec`,
//! and the runtime depends on all three.
//!
//! - [`footprint`]: what a model costs in bytes, by [`Tier`] and [`Part`], shared with the CPU
//!   offload planner.

pub mod footprint;

pub use footprint::{Dtype, Footprint, FootprintRequest, Part, Placement, Quant, Tier};
