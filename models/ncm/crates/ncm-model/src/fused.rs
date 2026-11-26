//! When the v2 layers take their fused decode kernels instead of the reference tensor code.
//!
//! The kernels (`src/kernels`, cargo feature `cuda`) cover the host-bound decode paths: a gated
//! DeltaNet step over a few tokens and one-token sparse retrieval. They run on CUDA devices only;
//! every other device, and every build without the feature (CI), takes the reference path. The
//! process-wide switch below turns them off on CUDA as well, for A/B runs and the parity tests.

use std::sync::atomic::{AtomicBool, Ordering};

use burn::nn::RmsNorm;
use burn::tensor::{Device, Int, Tensor};

static FUSED_KERNELS: AtomicBool = AtomicBool::new(true);
#[cfg(feature = "cuda")]
static CAPTURING: AtomicBool = AtomicBool::new(false);
static CUDA_GRAPHS: AtomicBool = AtomicBool::new(true);
static FUSED_MLP: AtomicBool = AtomicBool::new(true);

/// Allow (the default) or forbid the fused MLP kernels ([`crate::mlp_fused`]: the norm folded
/// into the gate/up GEMV, SwiGLU and the residual add into its epilogues); without them the MLP
/// runs as separate norm, GEMV and elementwise ops. Process-wide, for A/B runs.
pub fn set_fused_mlp(on: bool) {
    FUSED_MLP.store(on, Ordering::Relaxed);
}

/// Whether decode MLPs on `device` take the fused MLP kernels now.
pub fn fused_mlp_active(device: &Device) -> bool {
    FUSED_MLP.load(Ordering::Relaxed) && active(device)
}

/// Allow (the default) or forbid replaying decode steps as captured CUDA graphs
/// ([`crate::ncm::GraphedDecoder`], [`crate::layers::GraphedStack`]); without them the same
/// fused kernels run eagerly. Process-wide; a graph already captured keeps replaying.
pub fn set_cuda_graphs(on: bool) {
    CUDA_GRAPHS.store(on, Ordering::Relaxed);
}

/// Whether decode steps on `device` may be captured into CUDA graphs now.
pub fn graphs_active(device: &Device) -> bool {
    CUDA_GRAPHS.load(Ordering::Relaxed) && active(device)
}

/// Marks a CUDA graph capture under way: layers then leave out the host-to-device refreshes
/// (positions) that the graph driver issues before each replay instead.
#[cfg(feature = "cuda")]
pub(crate) fn set_capturing(on: bool) {
    CAPTURING.store(on, Ordering::Relaxed);
}

/// Whether a CUDA graph capture is under way.
#[cfg(feature = "cuda")]
pub(crate) fn capturing() -> bool {
    CAPTURING.load(Ordering::Relaxed)
}

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

/// `norm.forward(x)`: one kernel on CUDA when the fused kernels are active
/// ([`crate::kernels::rms_norm`]), Burn's ops otherwise.
pub fn rms_norm<const D: usize>(norm: &RmsNorm, x: Tensor<D>) -> Tensor<D> {
    #[cfg(feature = "cuda")]
    if active(&x.device()) {
        return crate::kernels::rms_norm(x, norm.gamma.val(), norm.epsilon);
    }
    norm.forward(x)
}

/// The residual add `x + delta` and its norm, `(s, norm.forward(s))`: one kernel on CUDA when
/// the fused kernels are active ([`crate::kernels::add_rms_norm`]), Burn's ops otherwise.
pub fn add_rms_norm<const D: usize>(
    norm: &RmsNorm,
    x: Tensor<D>,
    delta: Tensor<D>,
) -> (Tensor<D>, Tensor<D>) {
    #[cfg(feature = "cuda")]
    if active(&x.device()) {
        return crate::kernels::add_rms_norm(x, delta, norm.gamma.val(), norm.epsilon);
    }
    let s = x + delta;
    (s.clone(), norm.forward(s))
}

/// `x W` for a dense weight `w` `[K, N]` (row-major, or the transpose of a row-major `[N, K]`):
/// on CUDA with the fused kernels on, inputs of at most eight rows (decode steps) take the
/// matrix-vector kernels ([`crate::kernels::gemv`]), up to 64 rows the small-M GEMM
/// ([`crate::kernels::gemm_small`]); everything else Burn's matmul. Dense
/// [`crate::quant::QLinear`] layers and the tied output head multiply through it.
pub fn linear<const D: usize>(x: Tensor<D>, w: Tensor<2>) -> Tensor<D> {
    #[cfg(feature = "cuda")]
    if active(&x.device()) {
        let rows = x.dims()[..D - 1].iter().product::<usize>();
        if crate::kernels::gemv_rows_ok(rows) {
            return crate::kernels::gemv(x, w);
        }
        // Past the GEMV kernels' rows, the small-M GEMM up to 64.
        if crate::kernels::gemm_small_rows_ok(rows) {
            return crate::kernels::gemm_small(x, w);
        }
    }
    burn::tensor::module::linear(x, w, None)
}

/// `x W^T` for the tied embedding `emb` `[V, d]` (the logits), through [`linear`].
pub fn linear_transposed<const D: usize>(emb: Tensor<2>, x: Tensor<D>) -> Tensor<D> {
    linear(x, emb.transpose())
}
