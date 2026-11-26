//! The backend-agnostic engine interface of the NCM runtime (`docs/plans/npu.md`, decision 1).
//!
//! This crate depends on nothing model-specific: no Burn, no ONNX Runtime. The dependency
//! direction is `ncm-model -> ncm-exec` (the Burn adapter lives there), `ncm-ort -> ncm-exec`,
//! and the runtime depends on all three.
//!
//! - [`footprint`]: what a model costs in bytes, by [`Tier`] and [`Part`], shared with the CPU
//!   offload planner.
//! - [`exec`]: the [`ExecBackend`] and [`ExecSession`] traits and their plain-data types.
//! - [`manifest`]: what `ncm-export` writes and `ncm-ort` reads; [`rope`]: host RoPE tables.
//! - [`hybrid`]: the hybrid's block types and the host half of its block state machine
//!   ([`hybrid::BlockLogic`]), shared by every backend's block run.

pub mod exec;
pub mod footprint;
pub mod hybrid;
pub mod manifest;
pub mod rope;

pub use exec::{
    Capabilities, ExecBackend, ExecError, ExecSession, HostChoice, ModelKind, PendingOutput,
    PickMode, PickSpec, PrefillOut, SnapshotInfo, StepOut, drive_block,
};
pub use footprint::{Dtype, Footprint, FootprintRequest, Part, Placement, Quant, Tier};
pub use hybrid::{
    BlockLogic, BlockOut, BlockPhase, BlockStats, DecodeOptions, DrawRow, DrawSettings, Hooks,
    PassMode, PassResult,
};
