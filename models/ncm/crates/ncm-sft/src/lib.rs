//! Stage 3 of the NCM plan: fine-tuning data. Converts the archived token-era Emperor and
//! Consul teacher data (DeepSeek V4 Pro traces judged by Kimi K3.1 Pro) to the NCM chat format.
//! The two sets are per-harness sets for Princeps, the single model that runs both the Emperor
//! and the Consul harness; stage 7 fine-tunes on their union.
//! No teacher calls and no training happen here.

pub mod convert;
pub mod format;
pub mod message;
pub mod pipeline;
pub mod qwen;
pub mod split;
pub mod tools;
