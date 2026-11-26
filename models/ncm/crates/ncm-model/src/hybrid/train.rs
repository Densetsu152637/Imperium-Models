//! Training-side settings of the hybrid.
//!
//! **Gradient checkpointing.** Burn 0.22 sets the checkpointing strategy per autodiff device
//! (`Device::autodiff().gradient_checkpointing()`: memory-bound operations are recomputed in
//! the backward pass, compute-bound ones keep their outputs), and tensors of devices with
//! different strategies cannot meet in one graph. So the switch covers the whole model
//! (backbone, generator and token decoder together): [`HybridConfig::checkpointing`] picks it
//! through [`training_device`]. Per-component recomputation (a segment's forward rerun in the
//! backward pass) needs a custom backward operation, which the unified tensor API does not
//! expose yet (TODO).
//!
//! **Memory attention** runs per chunk of [`HybridConfig::memory_chunk`] blocks over each
//! chunk's window ([`super::memory::Memory::chunk_blocks`]), and the token loss reads only the
//! true tokens' log-probabilities ([`super::HybridLm::forward_train_targets`]), so a 4k-token
//! sequence's training forward holds no `nb · q · N` score matrix and no `[N, V]` mixture.

#[cfg(feature = "train")]
use burn::tensor::Device;

#[cfg(doc)]
use crate::config::HybridConfig;

/// The autodiff device to train on: `device` with autodiff, and gradient checkpointing when
/// `checkpointing` ([`HybridConfig::checkpointing`]).
#[cfg(feature = "train")]
pub fn training_device(device: &Device, checkpointing: bool) -> Device {
    let device = device.clone().autodiff();
    if checkpointing { device.gradient_checkpointing() } else { device }
}
