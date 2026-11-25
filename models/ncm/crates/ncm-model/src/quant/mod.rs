//! Weight-only quantisation of the linear layers: 8-bit codes with one scale per output channel
//! (W8A16) or 4-bit codes with one scale per group of 64 or 128 inputs (W4A16). Activations stay
//! at the run dtype. See `docs/quantisation.md`.
//!
//! Every linear layer of both models and both designs is a [`QLinear`], dense unless the model
//! was built inside [`build`] with a [`QuantSpec`]. A dense model converts with
//! [`quantize_model`]; quantised weight files save and load with [`save_quantized`] and
//! [`load_quantized`]. Kept dense: embeddings, norms, convolutions, gates' per-head parameters,
//! the retrieval layers' latent up-projections (read as matrices by the absorbed attention) and,
//! unless the spec asks for it, the tied output head.
//!
//! On CUDA (cargo feature `cuda`, with [`crate::fused::active`]) quantised layers multiply
//! through fused CubeCL kernels ([`crate::kernels::qmatmul`]): a dequantise-and-multiply GEMV for
//! up to [`GEMV_ROWS`] rows (decode) and, for more rows (prefill), a dequantising kernel feeding
//! Burn's matmul; dense layers take the dense GEMV kernel for decode ([`crate::fused::linear`]).
//! Both read their weights in place, so the captured decode graphs (NCM decoder, v2 backbone,
//! draft and verify passes) replay quantised layers like dense ones. Elsewhere quantised layers
//! decode the weight with tensor ops and multiply.

mod convert;
pub mod format;
mod linear;

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

use burn::module::{Module, ModuleVisitor, Param};
use burn::tensor::{Device, Int, Tensor};

pub use convert::{QuantError, load_quantized, quantize_model, read_spec, save_quantized};
pub use format::Format;
pub use linear::{QLinear, dequantize};

/// Rows (tokens) up to which a quantised layer takes the GEMV kernel on CUDA.
pub const GEMV_ROWS: usize = 8;

/// Sums the bytes of every parameter at its dtype.
struct Bytes(usize);

impl ModuleVisitor for Bytes {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let t = param.val();
        self.0 += t.shape().num_elements() * t.dtype().size();
    }

    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<D, Int>>) {
        let t = param.val();
        self.0 += t.shape().num_elements() * t.dtype().size();
    }
}

/// Bytes of a model's parameters as held on the device (codes, scales and dense tensors).
/// Initialises any parameter not initialised yet.
pub fn weight_bytes<M: Module>(model: &M) -> usize {
    let mut bytes = Bytes(0);
    model.visit(&mut bytes);
    bytes.0
}

/// How a model's linear layers are quantised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantSpec {
    pub format: Format,
    /// Also quantise the tied output head (a separate quantised copy; the embedding table
    /// itself stays dense for lookups).
    pub head: bool,
}

impl QuantSpec {
    /// The spec named by a `--quant` value (`none`, `q8`, `q4`), with the q4 `group` size and
    /// whether to quantise the output head; None for `none`.
    pub fn parse(mode: &str, group: usize, head: bool) -> Result<Option<Self>, String> {
        let format = match mode {
            "none" => return Ok(None),
            "q8" => Format::q8(),
            "q4" if group > 0 && group.is_multiple_of(32) => Format::q4(group),
            "q4" => return Err(format!("q4 group size {group} is not a multiple of 32")),
            other => return Err(format!("unknown quantisation `{other}` (none, q8, q4)")),
        };
        Ok(Some(Self { format, head }))
    }

    /// Short name: `q8`, `q4-g64`, with `+head` when the head is quantised.
    pub fn name(&self) -> String {
        let base = match self.format.group {
            None => format!("q{}", self.format.bits),
            Some(g) => format!("q{}-g{g}", self.format.bits),
        };
        if self.head { format!("{base}+head") } else { base }
    }
}

/// What [`QLinear::new`] builds on this thread.
#[derive(Clone, Copy)]
struct Building {
    spec: QuantSpec,
    /// Zero codes instead of quantised random weights: the layers are about to be overwritten.
    placeholder: bool,
}

thread_local! {
    static BUILDING: Cell<Option<Building>> = const { Cell::new(None) };
}

/// Restores the previous build mode, also when the builder panics.
struct Restore(Option<Building>);

impl Drop for Restore {
    fn drop(&mut self) {
        BUILDING.set(self.0);
    }
}

fn build_as<T>(mode: Option<Building>, f: impl FnOnce() -> T) -> T {
    let _restore = Restore(BUILDING.replace(mode));
    f()
}

/// Runs `f` (a model constructor) with every [`QLinear`] it creates quantised per `spec`, from
/// the usual N(0, 0.02) random weights; `None` builds a dense model.
pub fn build<T>(spec: Option<QuantSpec>, f: impl FnOnce() -> T) -> T {
    build_as(spec.map(|spec| Building { spec, placeholder: false }), f)
}

/// [`build`] with zero codes in place of random weights, for a model about to be loaded.
fn build_placeholder<T>(spec: QuantSpec, f: impl FnOnce() -> T) -> T {
    build_as(Some(Building { spec, placeholder: true }), f)
}

fn building() -> Option<Building> {
    BUILDING.get()
}

/// The quantised output head of a model with tied embeddings (`d` to `vocab`), when the model is
/// being built with a spec that quantises the head; None otherwise (logits then use the
/// embedding table directly).
pub fn tied_head(vocab: usize, d: usize, device: &Device) -> Option<QLinear> {
    building().filter(|b| b.spec.head).map(|_| QLinear::new(d, vocab, device))
}

static KERNELS: AtomicBool = AtomicBool::new(true);

/// Allow (the default) or forbid the CUDA quantised-matmul kernels; forbidden, CUDA runs the
/// tensor-op reference like every other device. Process-wide; [`crate::fused::set_fused_kernels`]
/// turning the fused kernels off turns these off as well.
pub fn set_quant_kernels(on: bool) {
    KERNELS.store(on, Ordering::Relaxed);
}

/// Whether quantised layers on `device` multiply through the CUDA kernels now (a `cuda` build,
/// a CUDA device, the fused kernels on and these allowed).
pub fn kernels_active(device: &Device) -> bool {
    KERNELS.load(Ordering::Relaxed) && crate::fused::active(device)
}
