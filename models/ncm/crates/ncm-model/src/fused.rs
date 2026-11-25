//! When the v2 layers take their fused decode kernels instead of the reference tensor code.
//!
//! The kernels (`src/kernels`, cargo feature `cuda`) cover the host-bound decode paths: a gated
//! DeltaNet step over a few tokens and one-token sparse retrieval. They run on CUDA devices only;
//! every other device, and every build without the feature (CI), takes the reference path. The
//! process-wide switch below turns them off on CUDA as well, for A/B runs and the parity tests.

use std::sync::atomic::{AtomicBool, Ordering};

use burn::tensor::{Device, Int, Tensor};

static FUSED_KERNELS: AtomicBool = AtomicBool::new(true);

/// Allow (the default) or forbid the fused kernels. Process-wide; affects calls made after it.
pub fn set_fused_kernels(on: bool) {
    FUSED_KERNELS.store(on, Ordering::Relaxed);
}

/// Whether this build has the fused kernels at all (the `cuda` feature).
pub const fn kernels_built() -> bool {
    cfg!(feature = "cuda")
}

/// Whether layers on `device` take the fused kernels now.
pub fn active(device: &Device) -> bool {
    FUSED_KERNELS.load(Ordering::Relaxed) && on_cuda(device)
}

#[cfg(feature = "cuda")]
fn on_cuda(device: &Device) -> bool {
    use burn::backend::DispatchDevice;
    matches!(device.as_dispatch(), DispatchDevice::Cube(burn_cubecl::CubeDevice::Cuda(_)))
}

#[cfg(not(feature = "cuda"))]
fn on_cuda(_device: &Device) -> bool {
    false
}

/// `x.argmax(D - 1)`, the greedy token pick: one kernel on CUDA when the fused kernels are
/// active ([`crate::kernels::argmax_last`]), Burn's reduction otherwise.
pub fn argmax_last<const D: usize>(x: Tensor<D>) -> Tensor<D, Int> {
    #[cfg(feature = "cuda")]
    if active(&x.device()) {
        return crate::kernels::argmax_last(x);
    }
    x.argmax(D - 1)
}
