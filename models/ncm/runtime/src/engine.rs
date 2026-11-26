//! The model, its caches and the generation loop for both architectures.
//!
//! One [`Engine`] owns one model and one set of caches allocated at launch for the per-request
//! context size. Requests run one at a time (batch size 1, like the decode-speed gate); each
//! prefills its prompt and decodes until a stop token, the token limit, or the end of the
//! context. A prompt that extends the previous request's prompt, or its prompt and reply,
//! restores the cache as it stood there and prefills only the rest; any other prompt resets the
//! caches. The NCM reads its prompt in k-token concepts from the start of the token stream; a
//! trailing partial concept is forced as the start of the first decoded one. Resets and restores
//! write into the caches' own buffers on CUDA, so the decode graphs are captured once and kept.
//! All GPU work runs on one worker thread ([`Engine`]), as the kept graphs require.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::Instant;

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::{Device, FloatDType, Tensor};
use ncm_exec::{
    Capabilities, Dtype, ExecBackend, ExecSession, ModelKind, PickMode, PickSpec, Quant,
    SnapshotInfo, StepOut,
};
use ncm_model::exec::{BurnBackend, BurnInfo, BurnModel};
use ncm_model::hybrid::HybridLm;
use ncm_model::quant::{self, QuantError, QuantSpec};
use ncm_model::weights::{self, WeightsError};
use ncm_model::{Config, NcmLm, TokenLm, cache_bytes};

use crate::constrain::Constraint;
use crate::sampling::{Sampler, SamplingParams};
use crate::vocab::{Specials, Vocab, VocabError};

/// Prompt tokens fed to the model at once: the fallback design's default.
pub const DEFAULT_PREFILL_CHUNK: usize = 512;
/// The v2 design's default: its fused prefill kernels are fastest on long chunks.
pub const DEFAULT_PREFILL_CHUNK_V2: usize = 8192;
/// Concepts in a row with no output before an NCM reply is cut off.
pub(crate) const MAX_EMPTY_CONCEPTS: usize = 16;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("the prompt has {prompt} tokens but the context holds {ctx}")]
    ContextExceeded { prompt: usize, ctx: usize },
    #[error("the prompt is empty")]
    EmptyPrompt,
    #[error("the context size {ctx} is too small for this model (needs at least {min})")]
    ContextTooSmall { ctx: usize, min: usize },
    #[error("the token table covers {table} ids but the model's vocabulary is {model}")]
    VocabMismatch { table: usize, model: usize },
    #[error(transparent)]
    Vocab(#[from] VocabError),
    #[error(transparent)]
    Weights(#[from] WeightsError),
    #[error(transparent)]
    Quant(#[from] QuantError),
    #[error("the weight file is quantised as {file}, but {asked} was asked for")]
    QuantMismatch { file: String, asked: String },
    #[error("device: {0}")]
    Device(String),
    #[error(transparent)]
    Exec(#[from] ncm_exec::ExecError),
    #[error("{0}")]
    Unsupported(String),
}

/// Why generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// A stop token, or the client went away.
    Stop,
    /// The token limit or the end of the context.
    Length,
}

/// What a generation did and how long it took.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenStats {
    /// Prompt tokens as the request sent them (before any NCM alignment padding).
    pub prompt_tokens: usize,
    /// Tokens generated, including the stop token.
    pub completion_tokens: usize,
    pub finish: Finish,
    /// Seconds to process the prompt.
    pub prompt_s: f64,
    /// Seconds spent decoding after the prompt.
    pub decode_s: f64,
    /// Tokens produced in `decode_s` (the token baseline's first token comes from the prompt
    /// pass, so it is not counted here).
    pub decode_tokens: usize,
    /// Prompt tokens served from the previous request's cache instead of being prefilled again
    /// (prompt-prefix reuse); `prompt_s` covers the rest.
    pub cached_tokens: usize,
    /// Seconds spent capturing the decode graph (outside `prompt_s` and `decode_s`; 0 when
    /// none was captured: the engine keeps its graphs across requests).
    pub capture_s: f64,
    /// How the reply was decoded ([`DecodePath`]).
    pub path: DecodePath,
    /// Seconds the request waited for a slot (continuous batching; 0 with one slot).
    pub queue_s: f64,
    /// Mean rows of the batched decode steps the request took part in (1 with one slot).
    pub mean_batch: f64,
    /// The prompt's prefill was another slot's, copied (shared-prefix fork).
    pub forked: bool,
}

/// The decode loop a request took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodePath {
    /// Per-token steps with the host choosing every token: grammar-constrained requests, the
    /// CPU, and CUDA with the fused kernels off.
    Eager,
    /// One captured CUDA graph replay per step (per concept for the NCM), greedy picks on the
    /// device.
    GraphedGreedy,
    /// The same with temperature, top-k and top-p sampling on the device.
    GraphedSampled,
    /// Self-speculative greedy decoding (NCM with a draft head, `--speculate`): draft and
    /// verify passes, each concept appended through a graph.
    Speculative,
    /// Continuous batching (`--parallel` above 1, [`crate::batch`]): batched eager steps over
    /// the decoding slots.
    Batched,
    /// The NCM hybrid's block decode loop (`ncm_model::hybrid::FastSession`), with what its
    /// blocks did.
    Hybrid(HybridAcceptance),
}

/// What the hybrid's blocks did in one request: acceptance is the draft tokens the first
/// check pass of each block accepted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HybridAcceptance {
    pub blocks: u32,
    /// Draft tokens accepted by first check passes, summed, and the passes counted.
    pub first_accepted: u32,
    pub first_passes: u32,
    pub redecodes: u32,
    /// Tokens written on the autoregressive path one per pass.
    pub ar_tokens: u32,
    /// The block and autoregressive passes replayed captured graphs.
    pub graphed: bool,
}

impl HybridAcceptance {
    pub(crate) fn add(&mut self, stats: &ncm_model::hybrid::BlockStats) {
        self.blocks += 1;
        if let Some(a) = stats.first_accepted {
            self.first_accepted += a as u32;
            self.first_passes += 1;
        }
        self.redecodes += u32::from(stats.redecoded);
        self.ar_tokens += stats.ar_tokens as u32;
    }

    /// Mean draft tokens accepted per first check pass.
    pub fn mean_first_accepted(&self) -> f64 {
        if self.first_passes == 0 {
            0.0
        } else {
            self.first_accepted as f64 / self.first_passes as f64
        }
    }
}

impl DecodePath {
    pub fn name(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::GraphedGreedy => "graphed-greedy",
            Self::GraphedSampled => "graphed-sampled",
            Self::Speculative => "speculative",
            Self::Batched => "batched",
            Self::Hybrid(_) => "hybrid",
        }
    }
}

pub struct GenRequest {
    pub prompt: Vec<u32>,
    /// Most tokens to generate.
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub constraint: Option<Box<dyn Constraint>>,
    /// Set to abandon the request (its client left): generation stops before the next prefill
    /// chunk or decode step. The engine also sets it when `on_token` returns false.
    pub cancel: Arc<AtomicBool>,
}

/// What `/props` and `/v1/models` report about the loaded model.
#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub arch: &'static str,
    pub params: usize,
    /// Context tokens one request may use.
    pub ctx: usize,
    /// Key/value cache bytes allocated at launch.
    pub cache_bytes: usize,
    pub dtype: String,
    pub device: String,
    pub vocab_size: usize,
    /// Tokens produced per decoder pass (1, or `k` for the NCM).
    pub unit: usize,
}

/// What the HTTP layer needs from a model. [`Engine`] is the real one; tests script others.
pub trait Backend: Send + Sync {
    fn info(&self) -> &ModelInfo;

    fn vocab(&self) -> &Arc<Vocab>;

    /// Generates a reply, calling `on_token` with each token as it is chosen. `on_token`
    /// returns false when nobody is listening any more, which ends the generation.
    fn generate(
        &self,
        request: GenRequest,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenStats, EngineError>;

    /// Rejects a prompt that cannot be served, before any generation starts.
    fn check_prompt(&self, tokens: usize) -> Result<(), EngineError>;

    /// Why the backend can serve no more requests (it panicked; its state may be broken), or
    /// None while it is healthy. The server then reports itself unavailable so that a
    /// supervisor restarts it.
    fn broken(&self) -> Option<String> {
        None
    }

    /// The slots and the queue of a continuously batching backend (`--parallel` above 1), or
    /// None.
    fn slots(&self) -> Option<crate::batch::SlotsReport> {
        None
    }
}

/// A request whose listener left during the prefill: nothing generated.
pub(crate) fn aborted(prompt_tokens: usize) -> GenStats {
    GenStats {
        prompt_tokens,
        completion_tokens: 0,
        finish: Finish::Stop,
        prompt_s: 0.0,
        decode_s: 0.0,
        decode_tokens: 0,
        cached_tokens: 0,
        capture_s: 0.0,
        path: DecodePath::Eager,
        queue_s: 0.0,
        mean_batch: 0.0,
        forked: false,
    }
}

/// The message of a panic payload.
pub fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic without a message".into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceChoice {
    Cpu,
    Cuda,
}

/// The compute device, with float dtype `dtype` configured.
pub fn make_device(choice: DeviceChoice, dtype: FloatDType) -> Result<Device, EngineError> {
    #[allow(unused_mut)]
    let mut device = match choice {
        DeviceChoice::Cpu => Device::flex(),
        #[cfg(feature = "cuda")]
        DeviceChoice::Cuda => Device::cuda(0),
        #[cfg(not(feature = "cuda"))]
        DeviceChoice::Cuda => {
            return Err(EngineError::Device(
                "this build has no CUDA support; rebuild with `--features cuda`".into(),
            ));
        }
    };
    // A device's default dtype is fixed process-wide by its first use, so a second engine in
    // one process (tests) can only reuse the dtype that is already set.
    if let Err(e) = device.configure(dtype)
        && device.settings().float_dtype != dtype
    {
        return Err(EngineError::Device(format!("cannot use {dtype:?} on {choice:?}: {e}")));
    }
    Ok(device)
}

pub struct EngineOptions {
    pub config: Config,
    /// Context tokens one request may use; the caches are allocated for it.
    pub ctx: usize,
    pub device: Device,
    pub device_name: String,
    pub dtype: FloatDType,
    /// Weight file, or `None` for random weights.
    pub weights: Option<PathBuf>,
    /// Seed for random weights.
    pub seed: u64,
    pub prefill_chunk: usize,
    /// Quantise the linear layers' weights at load (see `ncm_model::quant`). A quantised weight
    /// file loads as it is; this must then be None or the file's own spec.
    pub quant: Option<QuantSpec>,
    /// Decode greedy, unconstrained NCM requests by self-speculation when the model has a draft
    /// head (CUDA only). The parallel verify pass rounds differently from step-by-step decoding,
    /// so in bf16 a near tie may go the other way: off by default.
    pub speculate: bool,
    /// Keep the previous request's cache and prefill only the new part of a prompt that extends
    /// the previous prompt (see [`Engine`]).
    pub prefix_reuse: bool,
    /// Run a short greedy and a short sampled request at launch (`Engine::new` returns after
    /// them), so that the first real request does not pay for compiling the kernels and
    /// capturing the decode graphs.
    pub warmup: bool,
    /// Requests served at once (continuous batching, [`crate::batch`]); each slot gets caches
    /// for `ctx`. 1: one at a time.
    pub slots: usize,
}

/// A cache state kept from the previous request by the continuous batcher: the tokens the cache
/// held, a snapshot to restore it, and the output at that point (the next token's logits, or
/// the NCM's next condition) when known. A later prompt that extends `tokens` restores the
/// snapshot and prefills only its new tokens. The batcher keeps two per slot: right after the
/// prompt's prefill, and at the end of the reply (whose fed-back tokens a next turn's prompt
/// usually repeats). The sequential engine keeps the same two as [`SeqKept`].
pub(crate) struct Kept<S, O> {
    pub(crate) tokens: Vec<i32>,
    pub(crate) snapshot: S,
    pub(crate) out: Option<O>,
}

/// What the engine reads of a backend, fixed at launch.
pub(crate) struct Settings {
    pub(crate) info: ModelInfo,
    pub(crate) vocab: Arc<Vocab>,
    pub(crate) chunk: usize,
    /// Tokens the caches hold.
    pub(crate) capacity: usize,
    pub(crate) prefix_reuse: bool,
    /// Self-speculative NCM decoding where the model has a draft head.
    pub(crate) speculate: bool,
}

/// A kept point of the sequential engine's session: the tokens its caches held and a snapshot
/// to restore it (which carries the next step's output when the session had one).
struct SeqKept<P> {
    tokens: Vec<i32>,
    snapshot: P,
}

/// The sequential engine's mutable state: the one session and its kept points.
struct Seq<S: ExecSession> {
    session: S,
    kept: Vec<SeqKept<S::Snapshot>>,
}

/// The backend, its session and the generation loop for every model kind, on the engine's
/// worker thread ([`Engine`]). Requests run one at a time.
pub(crate) struct Core<B: ExecBackend> {
    pub(crate) cfg: Settings,
    backend: B,
    caps: Capabilities,
    seq: Mutex<Seq<B::Session>>,
}

/// What the continuous batcher reads of a core: the settings and the Burn device.
pub(crate) struct BatchCore {
    pub(crate) vocab: Arc<Vocab>,
    pub(crate) device: Device,
    pub(crate) chunk: usize,
    pub(crate) capacity: usize,
    pub(crate) prefix_reuse: bool,
    pub(crate) speculate: bool,
}

impl BatchCore {
    pub(crate) fn sync(&self) {
        self.device.sync().ok();
    }
}

impl Core<BurnBackend> {
    /// The Burn backend's model and first session's caches, for the continuous batcher (the
    /// Burn-only side door: `ncm-exec` has no batch API).
    pub(crate) fn into_batcher_parts(self) -> (BatchCore, BurnModel, ncm_model::exec::BurnCaches) {
        let Self { cfg, backend, seq, .. } = self;
        let seq = seq.into_inner().unwrap_or_else(|p| p.into_inner());
        let device = backend.device().clone();
        let (model, caches) = backend.into_parts(seq.session);
        let core = BatchCore {
            vocab: cfg.vocab,
            device,
            chunk: cfg.chunk,
            capacity: cfg.capacity,
            prefix_reuse: cfg.prefix_reuse,
            speculate: cfg.speculate,
        };
        (core, model, caches)
    }
}

/// The hybrid's decode options for a request: its sampling settings (seeded), verified mode
/// for constrained requests, the stop tokens, and `<tool_call>` / `</tool_call>` as the
/// mixed-block switch.
pub(crate) fn hybrid_options(
    params: SamplingParams,
    constrained: bool,
    ids: Specials,
) -> ncm_model::hybrid::DecodeOptions {
    ncm_model::hybrid::DecodeOptions {
        temperature: if params.is_greedy() { 0.0 } else { f64::from(params.temperature) },
        top_p: f64::from(params.top_p),
        top_k: params.top_k,
        seed: params.seed.unwrap_or_else(crate::sampling::entropy),
        verified: constrained,
        stop: [ids.im_end, ids.end_of_text, ids.im_start].iter().map(|&t| t as i32).collect(),
        switch_on: vec![ids.tool_call as i32],
        switch_off: vec![ids.tool_call_end as i32],
        ..Default::default()
    }
}

/// Burn initialises parameters when first used, drawing from the device's generator then.
/// Touching them all right after construction makes random weights depend on the seed alone.
struct Materialise;

impl ModuleMapper for Materialise {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        param.map(|tensor| tensor)
    }
}

/// Builds a model with `build` and fills it: random weights (quantised per `quant`), a dense
/// weight file (quantised after loading per `quant`), or a quantised weight file.
fn load_model<M: Module>(
    weights: Option<&Path>,
    dtype: FloatDType,
    quant: Option<QuantSpec>,
    device: &Device,
    build: impl Fn() -> M,
) -> Result<M, EngineError> {
    let Some(path) = weights else {
        return Ok(quant::build(quant, &build).map(&mut Materialise));
    };
    if let Some(file) = quant::read_spec(path)? {
        if let Some(asked) = quant.filter(|q| *q != file) {
            return Err(EngineError::QuantMismatch { file: file.name(), asked: asked.name() });
        }
        return Ok(quant::load_quantized(path, dtype, &build)?.0);
    }
    let model = weights::load_weights(build(), path, dtype)?;
    match quant {
        Some(spec) => Ok(quant::quantize_model(&model, spec, device, &build)?),
        None => Ok(model),
    }
}

fn itemsize(dtype: FloatDType) -> usize {
    match dtype {
        FloatDType::F64 => 8,
        FloatDType::F32 | FloatDType::Flex32 => 4,
        _ => 2,
    }
}

/// The interface's float type of a Burn dtype.
fn neutral_dtype(dtype: FloatDType) -> Dtype {
    match dtype {
        FloatDType::F16 => Dtype::F16,
        FloatDType::BF16 => Dtype::Bf16,
        _ => Dtype::F32,
    }
}

/// The interface's weight quantisation of a Burn spec.
fn neutral_quant(spec: Option<QuantSpec>) -> Quant {
    match spec {
        None => Quant::None,
        Some(s) if s.format.bits == 4 => Quant::Int4 { group: s.format.group.unwrap_or(0) },
        Some(s) => Quant::Int8 { group: s.format.group.unwrap_or(0) },
    }
}

/// Builds the Burn backend (loading `options.weights` or initialising random weights) and
/// allocates the caches for `options.ctx` tokens.
fn build_burn(options: EngineOptions, vocab: Arc<Vocab>) -> Result<Core<BurnBackend>, EngineError> {
    let EngineOptions {
        config,
        ctx,
        device,
        device_name,
        dtype,
        weights,
        seed,
        prefill_chunk,
        quant,
        speculate,
        prefix_reuse,
        warmup,
        slots,
    } = options;
    if vocab.len() != config.vocab_size() {
        return Err(EngineError::VocabMismatch { table: vocab.len(), model: config.vocab_size() });
    }
    let unit = config.unit();
    let min = 2 * unit;
    if ctx < min {
        return Err(EngineError::ContextTooSmall { ctx, min });
    }
    // Whole concepts only: the NCM reads and writes `k` tokens at a time.
    let capacity = ctx - ctx % unit;
    device.seed(seed);
    let (model, params, arch) = match &config {
        Config::Tok(cfg) => {
            let build = || TokenLm::new(cfg, &device);
            let model = load_model(weights.as_deref(), dtype, quant, &device, build)?;
            // Counted on a dense model that is never initialised: a quantised one holds codes.
            let params = TokenLm::new(cfg, &device).parts().total();
            (BurnModel::Tok(model), params, "tok")
        }
        Config::Ncm(cfg) if cfg.hybrid.is_some() => {
            let build = || HybridLm::new(cfg, &device);
            let model = load_model(weights.as_deref(), dtype, quant, &device, build)?;
            let params = HybridLm::new(cfg, &device).parts().inference();
            (BurnModel::Hybrid(model), params, "hybrid")
        }
        Config::Ncm(cfg) => {
            let build = || NcmLm::new(cfg, &device);
            let model = load_model(weights.as_deref(), dtype, quant, &device, build)?;
            let params = NcmLm::new(cfg, &device).parts().total();
            (BurnModel::Ncm(model), params, "ncm")
        }
    };
    let info = ModelInfo {
        arch,
        params,
        ctx: capacity,
        cache_bytes: match &config {
            Config::Ncm(c) if c.hybrid.is_some() => {
                slots.max(1)
                    * ncm_model::hybrid::hybrid_cache_bytes(c, capacity, itemsize(dtype), 1)
            }
            _ => slots.max(1) * cache_bytes(&config, capacity, itemsize(dtype), 1),
        },
        dtype: format!("{dtype:?}").to_lowercase(),
        device: device_name.clone(),
        vocab_size: config.vocab_size(),
        unit,
    };
    let backend = BurnBackend::new(
        model,
        device,
        BurnInfo {
            config,
            dtype: neutral_dtype(dtype),
            quant: neutral_quant(quant),
            params,
            device_name,
            speculate,
        },
    );
    Core::new(backend, info, vocab, prefill_chunk, prefix_reuse, speculate, warmup)
}

impl<B: ExecBackend> Core<B> {
    /// A core over a loaded `backend`: its one session holds caches for `info.ctx` tokens. With
    /// `warmup`, runs a short greedy and a short sampled request first.
    pub(crate) fn new(
        backend: B,
        info: ModelInfo,
        vocab: Arc<Vocab>,
        prefill_chunk: usize,
        prefix_reuse: bool,
        speculate: bool,
        warmup: bool,
    ) -> Result<Self, EngineError> {
        let caps = backend.capabilities();
        let capacity = info.ctx;
        let session = backend.new_session(capacity)?;
        let cfg = Settings {
            chunk: prefill_chunk.max(info.unit),
            capacity,
            prefix_reuse,
            speculate,
            info,
            vocab,
        };
        let core = Self { cfg, backend, caps, seq: Mutex::new(Seq { session, kept: Vec::new() }) };
        if warmup {
            core.warm_up();
        }
        Ok(core)
    }

    /// Compiles the prefill kernels for a full prefill chunk (on throwaway state), then serves
    /// two throwaway requests (greedy, then sampled) long enough for the decode graphs to be
    /// captured (past the NCM's cross-attention ring), compiling every decode kernel on the
    /// way, and forgets them. Errors are ignored: a real request would meet them again.
    fn warm_up(&self) {
        let ids = self.cfg.vocab.ids;
        let unit = self.cfg.info.unit;
        let capacity = self.cfg.capacity;
        let shapes = self.cfg.chunk.min(capacity.saturating_sub(2 * unit)).max(unit);
        self.backend.warm_up(shapes, self.cfg.chunk);
        let len = (capacity / 2).clamp(unit, 1024);
        let never = AtomicBool::new(false);
        for sampling in [
            SamplingParams::GREEDY,
            SamplingParams { temperature: 0.8, top_k: 40, top_p: 0.95, seed: Some(0) },
        ] {
            let request = GenRequest {
                prompt: vec![ids.pad; len],
                max_tokens: 4 * unit,
                sampling,
                constraint: None,
                cancel: Default::default(),
            };
            self.generate(request, &never, &mut |_| true).ok();
        }
        let mut seq = self.seq.lock().unwrap_or_else(|p| p.into_inner());
        seq.session.reset();
        seq.kept.clear();
    }

    pub(crate) fn sync(&self) {
        self.backend.sync();
    }
}

/// The device sampler's settings for a sampled request.
#[cfg(feature = "cuda")]
pub(crate) fn device_sampling(params: SamplingParams) -> ncm_model::sample::SampleSettings {
    ncm_model::sample::SampleSettings {
        temperature: params.temperature,
        top_k: params.top_k,
        top_p: params.top_p,
        seed: params.seed.unwrap_or_else(crate::sampling::entropy),
        counter: 0,
    }
}

pub(crate) use ncm_model::exec::{read_ids, read_logits};

/// Mutable state of one generation, shared by both architectures.
struct Run<'a> {
    sampler: Sampler,
    params: SamplingParams,
    constraint: Option<Box<dyn Constraint>>,
    ids: Specials,
    on_token: &'a mut dyn FnMut(u32) -> bool,
    /// Set when the listener left: the decode loops check it before each step.
    cancel: &'a AtomicBool,
    max_tokens: usize,
    emitted: usize,
}

impl Run<'_> {
    /// Chooses a token from `logits`; with nothing allowed, ends the reply.
    fn choose(&mut self, logits: &[f32]) -> u32 {
        let token = self.sampler.pick(logits, self.constraint.as_deref());
        token.unwrap_or(self.ids.im_end)
    }

    fn advance(&mut self, token: u32) {
        if let Some(constraint) = &mut self.constraint {
            constraint.advance(token);
        }
    }

    /// Hands a token to the listener and says whether to go on: stop for a stop token or when
    /// nobody is listening, limit when the token limit is reached.
    fn emit(&mut self, token: u32) -> Emit {
        self.emitted += 1;
        if !(self.on_token)(token) {
            return Emit::Stop;
        }
        if self.ids.is_stop(token) {
            return Emit::Stop;
        }
        if self.emitted >= self.max_tokens { Emit::Limit } else { Emit::Go }
    }
}

enum Emit {
    Go,
    Stop,
    Limit,
}

/// Rejects a prompt of `tokens` that a model of `info` cannot serve.
fn check_prompt(info: &ModelInfo, tokens: usize) -> Result<(), EngineError> {
    if tokens == 0 {
        return Err(EngineError::EmptyPrompt);
    }
    if tokens.next_multiple_of(info.unit) > info.ctx {
        return Err(EngineError::ContextExceeded { prompt: tokens, ctx: info.ctx });
    }
    Ok(())
}

/// What a worker thread serves requests with.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Worker {
    /// One request at a time through the execution interface.
    Sequential(Box<dyn Serve>),
    /// Continuous batching (Burn only).
    Batched(crate::batch::Batcher),
}

/// The model, its caches and the generation loop, served from one long-lived worker thread.
///
/// cubecl gives every OS thread its own stream, and a captured CUDA graph replays in order only
/// with work on its capture stream. The engine keeps its decode graphs across requests, so every
/// request's GPU work (building the model, staging inputs, replays, reading tokens back) must run
/// on the same thread: [`Engine::new`] starts that thread and builds the model there, and
/// [`Backend::generate`] hands it each request and relays its tokens, from whichever thread the
/// caller runs on.
pub struct Engine {
    info: ModelInfo,
    vocab: Arc<Vocab>,
    jobs: Mutex<mpsc::Sender<Job>>,
    /// Set (with the panic's message) when a request panicked on the worker thread, which then
    /// stops: its model, caches and graphs may be inconsistent.
    dead: Arc<OnceLock<String>>,
    /// The batcher's slots and queue (more than one slot), for `/props`.
    report: Option<Arc<Mutex<crate::batch::SlotsReport>>>,
    /// Events a request may have in flight: [`EVENTS_IN_FLIGHT`] with one slot; with more, the
    /// whole reply, so that a slow listener never stalls the batch.
    in_flight: usize,
    /// Debug builds only: the next request panics on the worker after this delay (ms; 0 off).
    #[cfg(debug_assertions)]
    panic_after: Arc<std::sync::atomic::AtomicU64>,
}

/// Events in flight between the worker and a request's caller: about two NCM concepts'
/// tokens, so the worker cannot run far ahead of a slow listener.
const EVENTS_IN_FLIGHT: usize = 8;

/// A request for the worker thread, the channel its events go back on, and the flag the caller
/// sets when its listener left.
pub(crate) struct Job {
    pub(crate) request: GenRequest,
    pub(crate) events: mpsc::SyncSender<Event>,
    pub(crate) cancel: Arc<AtomicBool>,
}

pub(crate) enum Event {
    Token(u32),
    Done(Result<GenStats, EngineError>),
}

impl Engine {
    /// Builds the model (loading `options.weights` or initialising random weights) and
    /// allocates the caches for `options.ctx` tokens, on a new worker thread that then serves
    /// every request.
    pub fn new(options: EngineOptions, vocab: Arc<Vocab>) -> Result<Self, EngineError> {
        let slots = options.slots.max(1);
        let worker_vocab = vocab.clone();
        Self::spawn(slots, vocab, move |report| {
            let core = build_burn(options, worker_vocab)?;
            let info = core.cfg.info.clone();
            Ok(match report {
                // The Burn-only side door: continuous batching over the Burn types.
                Some(report) => {
                    (info, Worker::Batched(crate::batch::Batcher::new(core, slots, report)))
                }
                None => (info, Worker::Sequential(Box::new(core))),
            })
        })
    }

    /// Starts the worker thread, on which `build` runs (every request's device work must run
    /// there). `build` gets the slots report when `slots` is above 1, and returns the model's
    /// info and what serves its requests.
    pub(crate) fn spawn(
        slots: usize,
        vocab: Arc<Vocab>,
        build: impl FnOnce(
            Option<Arc<Mutex<crate::batch::SlotsReport>>>,
        ) -> Result<(ModelInfo, Worker), EngineError>
        + Send
        + 'static,
    ) -> Result<Self, EngineError> {
        let slots = slots.max(1);
        let report =
            (slots > 1).then(|| Arc::new(Mutex::new(crate::batch::SlotsReport::default())));
        let worker_report = report.clone();
        let (ready, built) = mpsc::channel();
        let (jobs, queue) = mpsc::channel::<Job>();
        let dead = Arc::new(OnceLock::new());
        let worker_dead = dead.clone();
        #[cfg(debug_assertions)]
        let panic_after = Arc::new(std::sync::atomic::AtomicU64::new(0));
        #[cfg(debug_assertions)]
        let worker_panic_after = panic_after.clone();
        std::thread::Builder::new()
            .name("ncm-engine".into())
            .spawn(move || {
                let built = catch_unwind(AssertUnwindSafe(|| build(worker_report)));
                let worker = match built {
                    Ok(Ok((info, worker))) => {
                        ready.send(Ok(info)).ok();
                        worker
                    }
                    Ok(Err(e)) => {
                        ready.send(Err(e)).ok();
                        return;
                    }
                    Err(panic) => {
                        let message = panic_message(panic.as_ref());
                        ready
                            .send(Err(EngineError::Device(format!("loading panicked: {message}"))))
                            .ok();
                        return;
                    }
                };
                let core = match worker {
                    Worker::Batched(batcher) => {
                        batcher.serve(queue, worker_dead);
                        return;
                    }
                    Worker::Sequential(core) => core,
                };
                // Ends when the engine (its sender) is dropped, or after a panic.
                for Job { request, events, cancel } in queue {
                    let mut on_token = |token| {
                        !cancel.load(Ordering::Relaxed) && events.send(Event::Token(token)).is_ok()
                    };
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        #[cfg(debug_assertions)]
                        {
                            let ms = worker_panic_after.swap(0, Ordering::Relaxed);
                            if ms > 0 {
                                std::thread::sleep(std::time::Duration::from_millis(ms));
                                panic!("the test hook's panic");
                            }
                        }
                        core.generate(request, &cancel, &mut on_token)
                    }));
                    match result {
                        Ok(result) => {
                            events.send(Event::Done(result)).ok();
                        }
                        Err(panic) => {
                            let message = panic_message(panic.as_ref());
                            worker_dead.set(message.clone()).ok();
                            let error =
                                EngineError::Device(format!("generation panicked: {message}"));
                            events.send(Event::Done(Err(error))).ok();
                            return;
                        }
                    }
                }
            })
            .map_err(|e| EngineError::Device(format!("cannot start the engine thread: {e}")))?;
        let info = built
            .recv()
            .map_err(|_| EngineError::Device("the engine thread stopped while loading".into()))??;
        Ok(Self {
            info,
            vocab,
            jobs: Mutex::new(jobs),
            dead,
            report,
            in_flight: if slots > 1 { usize::MAX } else { EVENTS_IN_FLIGHT },
            #[cfg(debug_assertions)]
            panic_after,
        })
    }

    /// Test hook, debug builds only: the next request panics on the worker thread after
    /// `delay` (as a bug in generation would), so tests can check that the engine reports
    /// itself broken and fails the requests queued behind it.
    #[cfg(debug_assertions)]
    pub fn panic_next_request_after(&self, delay: std::time::Duration) {
        self.panic_after.store(delay.as_millis().max(1) as u64, Ordering::Relaxed);
    }
}

impl Backend for Engine {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }

    fn check_prompt(&self, tokens: usize) -> Result<(), EngineError> {
        check_prompt(&self.info, tokens)
    }

    fn broken(&self) -> Option<String> {
        self.dead.get().cloned()
    }

    fn slots(&self) -> Option<crate::batch::SlotsReport> {
        self.report.as_ref().map(|r| r.lock().unwrap_or_else(|p| p.into_inner()).clone())
    }

    fn generate(
        &self,
        request: GenRequest,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenStats, EngineError> {
        check_prompt(&self.info, request.prompt.len())?;
        if let Some(message) = self.dead.get() {
            return Err(EngineError::Device(format!(
                "the engine stopped after a panic: {message}"
            )));
        }
        // With batching, room for the whole reply: the worker never waits on one listener.
        let room = self.in_flight.min(request.max_tokens.saturating_add(self.info.unit + 2));
        let (events, received) = mpsc::sync_channel(room);
        let cancel = request.cancel.clone();
        let job = Job { request, events, cancel: cancel.clone() };
        // A worker that stopped after a panic drops the queued jobs: say why.
        let gone = || match self.dead.get() {
            Some(message) => {
                EngineError::Device(format!("the engine stopped after a panic: {message}"))
            }
            None => EngineError::Device("the engine thread stopped".into()),
        };
        self.jobs.lock().unwrap_or_else(|p| p.into_inner()).send(job).map_err(|_| gone())?;
        let mut listening = true;
        loop {
            match received.recv().map_err(|_| gone())? {
                // Tokens the worker sent before it saw the cancel are dropped.
                Event::Token(token) if listening => {
                    if !on_token(token) {
                        listening = false;
                        cancel.store(true, Ordering::Relaxed);
                    }
                }
                Event::Token(_) => {}
                Event::Done(result) => return result,
            }
        }
    }
}

/// What the worker thread serves one request at a time with.
pub(crate) trait Serve {
    fn generate(
        &self,
        request: GenRequest,
        cancel: &AtomicBool,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenStats, EngineError>;
}

impl<B: ExecBackend> Serve for Core<B> {
    fn generate(
        &self,
        request: GenRequest,
        cancel: &AtomicBool,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenStats, EngineError> {
        Core::generate(self, request, cancel, on_token)
    }
}

/// The device draws' settings of a sampled request (counter 0).
fn draw_row(params: SamplingParams) -> ncm_exec::DrawRow {
    ncm_exec::DrawRow {
        temperature: params.temperature,
        top_k: params.top_k,
        top_p: params.top_p,
        seed: params.seed.unwrap_or_else(crate::sampling::entropy),
        counter: 0,
    }
}

/// The tokens a device pick returned.
fn picked(out: StepOut) -> Result<Vec<u32>, EngineError> {
    match out {
        StepOut::Picked(tokens) => Ok(tokens),
        StepOut::Logits(_) => {
            Err(EngineError::Unsupported("the backend returned logits for a device pick".into()))
        }
    }
}

/// Emits a decoded concept's tokens after its first `skip` (forced prompt tokens), up to an
/// end-of-concept token; returns the concept to append (cut after that token and padded to
/// `k`), or why generation ends.
fn emit_concept(
    picked: &[u32],
    skip: usize,
    k: usize,
    run: &mut Run<'_>,
) -> Result<Vec<i32>, Finish> {
    let ids = run.ids;
    let mut concept: Vec<i32> = Vec::with_capacity(k);
    for (j, &token) in picked.iter().enumerate() {
        concept.push(token as i32);
        if j < skip {
            continue;
        }
        if token == ids.eoc {
            break;
        }
        match run.emit(token) {
            Emit::Stop => return Err(Finish::Stop),
            Emit::Limit => return Err(Finish::Length),
            Emit::Go => {}
        }
    }
    concept.resize(k, ids.pad as i32);
    Ok(concept)
}

impl<B: ExecBackend> Core<B> {
    fn generate(
        &self,
        request: GenRequest,
        cancel: &AtomicBool,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenStats, EngineError> {
        check_prompt(&self.cfg.info, request.prompt.len())?;
        let mut run = Run {
            sampler: Sampler::new(request.sampling),
            params: request.sampling,
            constraint: request.constraint,
            ids: self.cfg.vocab.ids,
            on_token,
            cancel,
            max_tokens: request.max_tokens.max(1),
            emitted: 0,
        };
        let mut seq = self.seq.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        match self.caps.model {
            ModelKind::Tok => self.generate_tok(&mut seq, &request.prompt, &mut run),
            ModelKind::Ncm => self.generate_ncm(&mut seq, &request.prompt, &mut run),
            ModelKind::Hybrid => self.generate_hybrid(&mut seq, &request.prompt, &mut run),
        }
    }

    /// The kept state a prompt `ids` can start from: the longest one whose tokens `ids`
    /// extends (with its stored output when `ids` is exactly its tokens), as (index, whole).
    fn pick_kept<P: SnapshotInfo>(
        &self,
        ids: &[i32],
        kept: &[SeqKept<P>],
    ) -> Option<(usize, bool)> {
        if !self.cfg.prefix_reuse {
            return None;
        }
        kept.iter()
            .enumerate()
            .filter(|(_, k)| ids.starts_with(&k.tokens))
            .filter(|(_, k)| ids.len() > k.tokens.len() || k.snapshot.has_output())
            .max_by_key(|(_, k)| k.tokens.len())
            .map(|(i, k)| (i, ids.len() == k.tokens.len()))
    }

    /// Prefills `ids` in chunks of `chunk`, checking `cancel` between them: false when the
    /// listener left first (or there was nothing to prefill).
    fn prefill_chunks(
        &self,
        session: &mut B::Session,
        ids: &[i32],
        chunk: usize,
        cancel: &AtomicBool,
    ) -> Result<bool, EngineError> {
        for piece in ids.chunks(chunk) {
            if cancel.load(Ordering::Relaxed) {
                return Ok(false);
            }
            session.prefill(piece, chunk)?;
        }
        Ok(!ids.is_empty())
    }

    /// Keeps the prompt's state `next` and, when the reply fed tokens back, the state at the
    /// end of the reply (`history`).
    fn keep_reply(
        &self,
        kept: &mut Vec<SeqKept<<B::Session as ExecSession>::Snapshot>>,
        mut next: Vec<SeqKept<<B::Session as ExecSession>::Snapshot>>,
        history: Vec<i32>,
        session: &B::Session,
    ) {
        if self.cfg.prefix_reuse && next.first().is_some_and(|k| history.len() > k.tokens.len()) {
            next.push(SeqKept { tokens: history, snapshot: session.snapshot() });
        }
        *kept = next;
    }

    fn generate_tok(
        &self,
        seq: &mut Seq<B::Session>,
        prompt: &[u32],
        run: &mut Run<'_>,
    ) -> Result<GenStats, EngineError> {
        let Seq { session, kept } = seq;
        let ids: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        let chunk = self.cfg.chunk;
        self.sync();
        let started = Instant::now();
        let (whole_prompt, cached) = match self.pick_kept(&ids, kept) {
            Some((i, true)) => {
                session.restore(&kept[i].snapshot);
                (true, ids.len())
            }
            Some((i, false)) => {
                let n = kept[i].tokens.len();
                session.restore(&kept[i].snapshot);
                (self.prefill_chunks(session, &ids[n..], chunk, run.cancel)?, n)
            }
            None => {
                session.reset();
                (self.prefill_chunks(session, &ids, chunk, run.cancel)?, 0)
            }
        };
        if !whole_prompt {
            // The listener left during the prefill: the cache holds part of the prompt, so no
            // kept state may claim it.
            kept.clear();
            return Ok(aborted(prompt.len()));
        }
        let mut next = Vec::new();
        if self.cfg.prefix_reuse {
            next.push(SeqKept { tokens: ids.clone(), snapshot: session.snapshot() });
        }
        // The tokens in the cache: the prompt, then every token fed back.
        let mut history = ids.clone();
        if self.caps.graphs && run.constraint.is_none() {
            self.sync();
            let prompt_s = started.elapsed().as_secs_f64();
            let sampled = !run.params.is_greedy();
            let pick = if sampled {
                PickSpec::new(PickMode::Sampled(draw_row(run.params)))
            } else {
                PickSpec::greedy()
            };
            // The first token comes from the prompt's logits; the first call also captures the
            // graph, timed apart from decoding.
            let mut token = picked(session.decode_step(&[], &pick)?)?[0];
            let capture_s = session.take_capture_s();
            let decode_started = Instant::now();
            let finish = loop {
                if run.cancel.load(Ordering::Relaxed) {
                    break Finish::Stop;
                }
                match run.emit(token) {
                    Emit::Stop => break Finish::Stop,
                    Emit::Limit => break Finish::Length,
                    Emit::Go => {}
                }
                if history.len() >= self.cfg.capacity {
                    break Finish::Length;
                }
                let next_token = picked(session.decode_step(&[token as i32], &pick)?)?[0];
                history.push(token as i32);
                token = next_token;
            };
            let decode_s = decode_started.elapsed().as_secs_f64();
            self.keep_reply(kept, next, history, session);
            return Ok(GenStats {
                prompt_tokens: prompt.len(),
                completion_tokens: run.emitted,
                finish,
                prompt_s,
                decode_s,
                decode_tokens: run.emitted.saturating_sub(1),
                cached_tokens: cached,
                capture_s,
                path: if sampled { DecodePath::GraphedSampled } else { DecodePath::GraphedGreedy },
                queue_s: 0.0,
                mean_batch: 1.0,
                forked: false,
            });
        }
        let mut row = session
            .pending_logits()
            .ok_or_else(|| EngineError::Unsupported("the prompt left no logits".into()))?;
        let prompt_s = started.elapsed().as_secs_f64();
        let decode_started = Instant::now();
        let host = PickSpec::host();
        let finish = loop {
            if run.cancel.load(Ordering::Relaxed) {
                break Finish::Stop;
            }
            let token = run.choose(&row);
            run.advance(token);
            match run.emit(token) {
                Emit::Stop => break Finish::Stop,
                Emit::Limit => break Finish::Length,
                Emit::Go => {}
            }
            if history.len() >= self.cfg.capacity {
                break Finish::Length;
            }
            row = match session.decode_step(&[token as i32], &host)? {
                StepOut::Logits(row) => row,
                StepOut::Picked(_) => {
                    return Err(EngineError::Unsupported("a host pick returned a token".into()));
                }
            };
            history.push(token as i32);
        };
        let decode_s = decode_started.elapsed().as_secs_f64();
        if self.cfg.prefix_reuse && history.len() > ids.len() {
            next.push(SeqKept { tokens: history, snapshot: session.snapshot() });
        }
        *kept = next;
        Ok(GenStats {
            prompt_tokens: prompt.len(),
            completion_tokens: run.emitted,
            finish,
            prompt_s,
            decode_s,
            decode_tokens: run.emitted.saturating_sub(1),
            cached_tokens: cached,
            capture_s: 0.0,
            path: DecodePath::Eager,
            queue_s: 0.0,
            mean_batch: 1.0,
            forked: false,
        })
    }

    /// The NCM. Concepts are fixed k-token chunks from the start of the token stream, as in
    /// pretraining: the prompt's whole concepts are prefilled, and its trailing partial concept
    /// (fewer than k tokens) stays pending: it is forced as the first tokens of the first
    /// decoded concept, which the decoder then completes. A prompt shorter than one concept is
    /// left-padded to one.
    fn generate_ncm(
        &self,
        seq: &mut Seq<B::Session>,
        prompt: &[u32],
        run: &mut Run<'_>,
    ) -> Result<GenStats, EngineError> {
        let Seq { session, kept } = seq;
        let k = self.caps.unit;
        let ids = run.ids;
        let pad = k.saturating_sub(prompt.len());
        let mut stream: Vec<i32> = vec![ids.pad as i32; pad];
        stream.extend(prompt.iter().map(|&t| t as i32));
        let whole = stream.len() - stream.len() % k;
        let (head, pending) = stream.split_at(whole);
        // Chunks are whole concepts too.
        let chunk = (self.cfg.chunk - self.cfg.chunk % k).max(k);
        self.sync();
        let started = Instant::now();
        let (whole_prompt, cached) = match self.pick_kept(head, kept) {
            Some((i, true)) => {
                session.restore(&kept[i].snapshot);
                (true, whole)
            }
            Some((i, false)) => {
                let n = kept[i].tokens.len();
                session.restore(&kept[i].snapshot);
                (self.prefill_chunks(session, &head[n..], chunk, run.cancel)?, n)
            }
            None => {
                session.reset();
                (self.prefill_chunks(session, head, chunk, run.cancel)?, 0)
            }
        };
        if !whole_prompt {
            kept.clear();
            return Ok(aborted(prompt.len()));
        }
        // Prompt tokens served from the cache, padding excluded.
        let cached = cached.saturating_sub(pad);
        let mut next = Vec::new();
        if self.cfg.prefix_reuse {
            next.push(SeqKept { tokens: head.to_vec(), snapshot: session.snapshot() });
        }
        let mut history = head.to_vec();
        self.sync();
        let prompt_s = started.elapsed().as_secs_f64();
        let capacity = self.cfg.capacity;
        let mut empty_concepts = 0;
        if self.caps.graphs && run.constraint.is_none() {
            let sampled = !run.params.is_greedy();
            let speculate = !sampled && self.cfg.speculate && self.caps.speculative;
            let mut pick = PickSpec {
                mode: if sampled {
                    PickMode::Sampled(draw_row(run.params))
                } else {
                    PickMode::Greedy
                },
                forced: pending.to_vec(),
                speculate,
            };
            // Decoding starts with the first concept, whose tokens count (the eager path
            // likewise).
            let decode_started = Instant::now();
            let mut tokens = picked(session.decode_step(&[], &pick)?)?;
            pick.forced.clear();
            let mut skip = pending.len();
            let finish = loop {
                if run.cancel.load(Ordering::Relaxed) {
                    break Finish::Stop;
                }
                let before = run.emitted;
                let concept = match emit_concept(&tokens, skip, k, run) {
                    Ok(concept) => concept,
                    Err(finish) => break finish,
                };
                skip = 0;
                empty_concepts = if run.emitted == before { empty_concepts + 1 } else { 0 };
                if empty_concepts >= MAX_EMPTY_CONCEPTS {
                    break Finish::Stop;
                }
                if history.len() + k > capacity {
                    break Finish::Length;
                }
                // Appends the concept just written and decodes the next one; the first step
                // that can be graphed captures its graph, timed apart from decoding.
                tokens = picked(session.decode_step(&concept, &pick)?)?;
                history.extend_from_slice(&concept);
            };
            let capture_s = session.take_capture_s();
            self.keep_reply(kept, next, history, session);
            let path = match (speculate, sampled) {
                (true, _) => DecodePath::Speculative,
                (false, true) => DecodePath::GraphedSampled,
                (false, false) => DecodePath::GraphedGreedy,
            };
            return Ok(GenStats {
                prompt_tokens: prompt.len(),
                completion_tokens: run.emitted,
                finish,
                prompt_s,
                decode_s: (decode_started.elapsed().as_secs_f64() - capture_s).max(0.0),
                decode_tokens: run.emitted,
                cached_tokens: cached,
                capture_s,
                path,
                queue_s: 0.0,
                mean_batch: 1.0,
                forked: false,
            });
        }
        let decode_started = Instant::now();
        let host_sampling = !run.params.is_greedy() || run.constraint.is_some();
        let mut forced: Vec<i32> = pending.to_vec();
        // The concept written last, appended by the next step.
        let mut feed: Vec<i32> = Vec::new();
        let finish = 'generate: loop {
            if run.cancel.load(Ordering::Relaxed) {
                break Finish::Stop;
            }
            // A concept's tokens: the pending prompt tokens first, then chosen on the host
            // when sampling, constrained, or completing a forced start, else by argmax on the
            // device.
            let tokens = if host_sampling || !forced.is_empty() {
                let mut ended = false;
                let mut choose = |_: usize, logits: &mut dyn FnMut() -> Vec<f32>| -> i32 {
                    if ended {
                        return ids.pad as i32;
                    }
                    let token = run.choose(&logits());
                    if token == ids.eoc || ids.is_stop(token) {
                        ended = true;
                    } else {
                        run.advance(token);
                    }
                    token as i32
                };
                session.decode_concept_host(&feed, &forced, &mut choose)?
            } else {
                picked(session.decode_step(&feed, &PickSpec::greedy())?)?
            };
            history.extend_from_slice(&feed);
            let skip = forced.len();
            forced.clear();
            let before = run.emitted;
            let concept = match emit_concept(&tokens, skip, k, run) {
                Ok(concept) => concept,
                Err(finish) => break 'generate finish,
            };
            empty_concepts = if run.emitted == before { empty_concepts + 1 } else { 0 };
            if empty_concepts >= MAX_EMPTY_CONCEPTS {
                break Finish::Stop;
            }
            if history.len() + k > capacity {
                break Finish::Length;
            }
            feed = concept;
        };
        let decode_s = decode_started.elapsed().as_secs_f64();
        if self.cfg.prefix_reuse && history.len() > whole {
            next.push(SeqKept { tokens: history, snapshot: session.snapshot() });
        }
        *kept = next;
        Ok(GenStats {
            prompt_tokens: prompt.len(),
            completion_tokens: run.emitted,
            finish,
            prompt_s,
            decode_s,
            decode_tokens: run.emitted,
            cached_tokens: cached,
            capture_s: 0.0,
            path: DecodePath::Eager,
            queue_s: 0.0,
            mean_batch: 1.0,
            forked: false,
        })
    }

    /// The NCM hybrid: the prompt restored from the longest kept point it extends (the last
    /// prompt, or the end of its reply, both kept at any offset in a block) and prefilled from
    /// there, or from scratch; then block after block ([`ExecSession::decode_block`]), each
    /// block's tokens streamed as it closes. Verified mode for constrained requests, whose
    /// grammar scans every check pass's tokens and picks every autoregressive token;
    /// `<tool_call>` switches a block to verified mode until the block in which `</tool_call>`
    /// appears ends.
    fn generate_hybrid(
        &self,
        seq: &mut Seq<B::Session>,
        prompt: &[u32],
        run: &mut Run<'_>,
    ) -> Result<GenStats, EngineError> {
        use std::cell::RefCell;
        let Seq { session, kept } = seq;
        let ids = run.ids;
        self.sync();
        let started = Instant::now();
        if run.cancel.load(Ordering::Relaxed) {
            return Ok(aborted(prompt.len()));
        }
        let stream: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        let cached = match self.pick_kept(&stream, kept) {
            Some((i, _)) => {
                session.restore(&kept[i].snapshot);
                kept[i].tokens.len()
            }
            None => {
                session.reset();
                0
            }
        };
        session.prefill(&stream[cached..], self.cfg.chunk)?;
        let mut next = Vec::new();
        if self.cfg.prefix_reuse {
            next.push(SeqKept { tokens: session.tokens().to_vec(), snapshot: session.snapshot() });
        }
        self.sync();
        let prompt_s = started.elapsed().as_secs_f64();
        let params = run.params;
        let opts = hybrid_options(params, run.constraint.is_some(), ids);
        session.begin_request(opts.seed);
        let constraint = RefCell::new(run.constraint.take());
        let sampler = RefCell::new(Sampler::new(params));
        let constrained = constraint.borrow().is_some();
        let mut acceptance = HybridAcceptance::default();
        let decode_started = Instant::now();
        let finish = 'reply: loop {
            if run.cancel.load(Ordering::Relaxed) {
                break Finish::Stop;
            }
            let room = self.cfg.capacity.saturating_sub(session.tokens().len());
            let remaining = run.max_tokens.saturating_sub(run.emitted).min(room);
            if remaining == 0 {
                break Finish::Length;
            }
            let mut scan = |proposed: &[i32]| {
                let mut c = constraint.borrow_mut();
                let Some(c) = c.as_mut() else { return proposed.len() };
                let mut n = 0;
                for &t in proposed {
                    if !c.allows(t as u32) {
                        break;
                    }
                    c.advance(t as u32);
                    n += 1;
                }
                n
            };
            let mut pick = |row: &[f32]| {
                let mut c = constraint.borrow_mut();
                let token = sampler.borrow_mut().pick(row, c.as_deref()).unwrap_or(ids.im_end);
                if let Some(c) = c.as_mut() {
                    c.advance(token);
                }
                token as i32
            };
            let mut hooks = ncm_exec::Hooks::default();
            if constrained {
                hooks.scan = Some(&mut scan);
                hooks.pick = Some(&mut pick);
            }
            let block = session.decode_block(remaining, &opts, &mut hooks)?;
            acceptance.add(&block.stats);
            for &t in &block.tokens {
                match run.emit(t as u32) {
                    Emit::Stop => break 'reply Finish::Stop,
                    Emit::Limit => break 'reply Finish::Length,
                    Emit::Go => {}
                }
            }
            if block.stopped || block.tokens.is_empty() {
                break Finish::Length;
            }
        };
        if self.cfg.prefix_reuse && session.tokens().len() > stream.len() {
            next.push(SeqKept { tokens: session.tokens().to_vec(), snapshot: session.snapshot() });
        }
        *kept = next;
        self.sync();
        acceptance.graphed = self.caps.graphs && session.failure().is_none();
        Ok(GenStats {
            prompt_tokens: prompt.len(),
            completion_tokens: run.emitted,
            finish,
            prompt_s,
            decode_s: decode_started.elapsed().as_secs_f64(),
            decode_tokens: run.emitted,
            cached_tokens: cached,
            capture_s: 0.0,
            path: DecodePath::Hybrid(acceptance),
            queue_s: 0.0,
            mean_batch: 1.0,
            forked: false,
        })
    }
}
