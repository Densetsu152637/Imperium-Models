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
use burn::tensor::{Device, FloatDType, Int, Tensor, TensorData};
use ncm_model::cache::LayerSnapshot;
use ncm_model::ncm::NcmSnapshot;
use ncm_model::quant::{self, QuantError, QuantSpec};
use ncm_model::weights::{self, WeightsError};
use ncm_model::{Config, LayerCache, NcmCache, NcmLm, TokenLm, cache_bytes};

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
}

impl DecodePath {
    pub fn name(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::GraphedGreedy => "graphed-greedy",
            Self::GraphedSampled => "graphed-sampled",
            Self::Speculative => "speculative",
            Self::Batched => "batched",
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

#[allow(clippy::large_enum_variant)]
pub(crate) enum State {
    Tok {
        model: TokenLm,
        caches: Vec<LayerCache>,
        kept: Vec<TokKept>,
        graphs: TokGraphs,
    },
    Ncm {
        model: NcmLm,
        cache: NcmCache,
        kept: Vec<NcmKept>,
        graphs: NcmGraphs,
    },
    /// Moved into the continuous batcher ([`crate::batch::Batcher`]).
    Empty,
}

/// A cache state kept from the previous request: the tokens the cache held, a snapshot to
/// restore it, and the output at that point (the next token's logits, or the NCM's next
/// condition) when known. A later prompt that extends `tokens` restores the snapshot and
/// prefills only its new tokens. The engine keeps two: right after the prompt's prefill, and
/// at the end of the reply (whose fed-back tokens a next turn's prompt usually repeats).
pub(crate) struct Kept<S, O> {
    pub(crate) tokens: Vec<i32>,
    pub(crate) snapshot: S,
    pub(crate) out: Option<O>,
}

type TokKept = Kept<Vec<LayerSnapshot>, Tensor<3>>;
type NcmKept = Kept<NcmSnapshot, Tensor<3>>;

/// The token baseline's decode graphs, kept across requests (their caches keep their buffers).
#[derive(Default)]
pub(crate) struct TokGraphs {
    #[cfg(feature = "cuda")]
    greedy: ncm_model::tok::GraphedTok,
    #[cfg(feature = "cuda")]
    sampled: Option<ncm_model::tok::GraphedTok>,
}

/// The NCM's decode graphs, kept across requests.
pub(crate) struct NcmGraphs {
    #[cfg(feature = "cuda")]
    greedy: ncm_model::ncm::GraphedStep,
    #[cfg(feature = "cuda")]
    sampled: Option<ncm_model::ncm::GraphedStep>,
    #[cfg(feature = "cuda")]
    append: ncm_model::ncm::GraphedAppend,
    #[cfg(feature = "cuda")]
    speculator: ncm_model::Speculator,
}

// Derivable only without CUDA, where the struct is empty.
#[cfg_attr(not(feature = "cuda"), allow(clippy::derivable_impls))]
impl Default for NcmGraphs {
    fn default() -> Self {
        Self {
            #[cfg(feature = "cuda")]
            greedy: Default::default(),
            #[cfg(feature = "cuda")]
            sampled: None,
            #[cfg(feature = "cuda")]
            append: Default::default(),
            #[cfg(feature = "cuda")]
            speculator: ncm_model::Speculator::new(ncm_model::Acceptance::Verify),
        }
    }
}

/// The model and its caches, on the engine's worker thread ([`Engine`]).
pub(crate) struct Core {
    info: ModelInfo,
    pub(crate) vocab: Arc<Vocab>,
    pub(crate) device: Device,
    pub(crate) chunk: usize,
    /// Tokens the caches hold.
    pub(crate) capacity: usize,
    pub(crate) prefix_reuse: bool,
    /// Read by the graphed paths (CUDA builds).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    speculate: bool,
    pub(crate) state: Mutex<State>,
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

impl Core {
    /// Builds the model (loading `options.weights` or initialising random weights) and
    /// allocates the caches for `options.ctx` tokens.
    fn build(options: EngineOptions, vocab: Arc<Vocab>) -> Result<Self, EngineError> {
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
            return Err(EngineError::VocabMismatch {
                table: vocab.len(),
                model: config.vocab_size(),
            });
        }
        let unit = config.unit();
        let min = 2 * unit;
        if ctx < min {
            return Err(EngineError::ContextTooSmall { ctx, min });
        }
        // Whole concepts only: the NCM reads and writes `k` tokens at a time.
        let capacity = ctx - ctx % unit;
        device.seed(seed);
        let (state, params, arch) = match &config {
            Config::Tok(cfg) => {
                let build = || TokenLm::new(cfg, &device);
                let model = load_model(weights.as_deref(), dtype, quant, &device, build)?;
                let caches = model.make_cache(capacity, 1, &device);
                // Counted on a dense model that is never initialised: a quantised one holds codes.
                let params = TokenLm::new(cfg, &device).parts().total();
                let state =
                    State::Tok { model, caches, kept: Vec::new(), graphs: TokGraphs::default() };
                (state, params, "tok")
            }
            Config::Ncm(cfg) if cfg.hybrid.is_some() => {
                // TODO: serve the hybrid (block decode loop, batched at block granularity);
                // its reference loop (`ncm_model::hybrid::HybridSession`) runs at batch 1.
                return Err(EngineError::Unsupported(
                    "ncm-server does not serve the NCM hybrid yet ([hybrid] config)".into(),
                ));
            }
            Config::Ncm(cfg) => {
                let build = || NcmLm::new(cfg, &device);
                let model = load_model(weights.as_deref(), dtype, quant, &device, build)?;
                let cache = model.make_cache(capacity, 1, &device);
                let params = NcmLm::new(cfg, &device).parts().total();
                let state =
                    State::Ncm { model, cache, kept: Vec::new(), graphs: NcmGraphs::default() };
                (state, params, "ncm")
            }
        };
        let info = ModelInfo {
            arch,
            params,
            ctx: capacity,
            cache_bytes: slots.max(1) * cache_bytes(&config, capacity, itemsize(dtype), 1),
            dtype: format!("{dtype:?}").to_lowercase(),
            device: device_name,
            vocab_size: config.vocab_size(),
            unit,
        };
        let core = Self {
            info,
            vocab,
            device,
            chunk: prefill_chunk.max(unit),
            capacity,
            prefix_reuse,
            speculate,
            state: Mutex::new(state),
        };
        if warmup {
            core.warm_up();
        }
        Ok(core)
    }

    /// Compiles the prefill kernels for a full prefill chunk (on a throwaway cache:
    /// [`NcmLm::warm_up`], or the token baseline's equivalent), then serves two throwaway
    /// requests (greedy, then sampled) long enough for the decode graphs to be captured (past
    /// the NCM's cross-attention ring), compiling every decode kernel on the way, and forgets
    /// them. Errors are ignored: a real request would meet them again.
    fn warm_up(&self) {
        let ids = self.vocab.ids;
        let unit = self.info.unit;
        let shapes = self.chunk.min(self.capacity.saturating_sub(2 * unit)).max(unit);
        {
            let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            match &*state {
                State::Ncm { model, .. } => model.warm_up(shapes, self.chunk, &self.device),
                State::Empty => {}
                State::Tok { model, .. } => {
                    // A throwaway cache: a prefill of `shapes` tokens, then one step.
                    let mut caches = model.make_cache(shapes + 1, 1, &self.device);
                    let prompt = self.int_tensor(&vec![ids.pad as i32; shapes]);
                    model.prefill(prompt, &mut caches, self.chunk);
                    let logits = model.step(self.int_tensor(&[ids.pad as i32]), &mut caches);
                    drop(logits.into_data());
                }
            }
        }
        let len = (self.capacity / 2).clamp(unit, 1024);
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
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        match &mut *state {
            State::Tok { caches, kept, .. } => {
                caches.iter_mut().for_each(LayerCache::reset);
                kept.clear();
            }
            State::Ncm { cache, kept, .. } => {
                cache.reset();
                kept.clear();
            }
            State::Empty => {}
        }
    }

    fn int_tensor(&self, ids: &[i32]) -> Tensor<2, Int> {
        Tensor::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), &self.device)
    }

    pub(crate) fn sync(&self) {
        self.device.sync().ok();
    }

    /// Whether requests may take the graphed device paths: CUDA with the fused kernels and
    /// graphs on.
    #[cfg(feature = "cuda")]
    fn graphs(&self) -> bool {
        ncm_model::fused::graphs_active(&self.device)
    }
}

/// The device sampler's settings for a sampled request.
#[cfg(feature = "cuda")]
fn device_sampling(params: SamplingParams) -> ncm_model::sample::SampleSettings {
    ncm_model::sample::SampleSettings {
        temperature: params.temperature,
        top_k: params.top_k,
        top_p: params.top_p,
        seed: params.seed.unwrap_or_else(crate::sampling::entropy),
        counter: 0,
    }
}

pub(crate) fn read_logits(logits: &Tensor<3>) -> Vec<f32> {
    logits.to_data_as::<f32>().try_into_vec::<f32>().expect("logits are floats")
}

pub(crate) fn read_ids(tokens: &Tensor<2, Int>) -> Vec<u32> {
    tokens
        .to_data_as::<i32>()
        .try_into_vec::<i32>()
        .expect("tokens are integers")
        .into_iter()
        .map(|id| id.max(0) as u32)
        .collect()
}

/// Mutable state of one generation, shared by both architectures.
struct Run<'a> {
    sampler: Sampler,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
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
        let report =
            (slots > 1).then(|| Arc::new(Mutex::new(crate::batch::SlotsReport::default())));
        let worker_report = report.clone();
        let (ready, built) = mpsc::channel();
        let (jobs, queue) = mpsc::channel::<Job>();
        let worker_vocab = vocab.clone();
        let dead = Arc::new(OnceLock::new());
        let worker_dead = dead.clone();
        #[cfg(debug_assertions)]
        let panic_after = Arc::new(std::sync::atomic::AtomicU64::new(0));
        #[cfg(debug_assertions)]
        let worker_panic_after = panic_after.clone();
        std::thread::Builder::new()
            .name("ncm-engine".into())
            .spawn(move || {
                let built = catch_unwind(AssertUnwindSafe(|| Core::build(options, worker_vocab)));
                let core = match built {
                    Ok(Ok(core)) => {
                        ready.send(Ok(core.info.clone())).ok();
                        core
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
                if let Some(report) = worker_report {
                    crate::batch::Batcher::new(core, slots, report).serve(queue, worker_dead);
                    return;
                }
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

impl Core {
    fn generate(
        &self,
        request: GenRequest,
        cancel: &AtomicBool,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenStats, EngineError> {
        check_prompt(&self.info, request.prompt.len())?;
        let mut run = Run {
            sampler: Sampler::new(request.sampling),
            params: request.sampling,
            constraint: request.constraint,
            ids: self.vocab.ids,
            on_token,
            cancel,
            max_tokens: request.max_tokens.max(1),
            emitted: 0,
        };
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        match &mut *state {
            State::Tok { model, caches, kept, graphs } => {
                self.generate_tok(model, caches, kept, graphs, &request.prompt, &mut run)
            }
            State::Ncm { model, cache, kept, graphs } => {
                self.generate_ncm(model, cache, kept, graphs, &request.prompt, &mut run)
            }
            State::Empty => unreachable!("the sequential loop keeps its model"),
        }
    }
}

impl Core {
    /// The kept state a prompt `ids` can start from: the longest one whose tokens `ids`
    /// extends (with its stored output when `ids` is exactly its tokens), as (index, whole).
    fn pick_kept<S, O>(&self, ids: &[i32], kept: &[Kept<S, O>]) -> Option<(usize, bool)> {
        if !self.prefix_reuse {
            return None;
        }
        kept.iter()
            .enumerate()
            .filter(|(_, k)| ids.starts_with(&k.tokens))
            .filter(|(_, k)| ids.len() > k.tokens.len() || k.out.is_some())
            .max_by_key(|(_, k)| k.tokens.len())
            .map(|(i, k)| (i, ids.len() == k.tokens.len()))
    }

    fn generate_tok(
        &self,
        model: &TokenLm,
        caches: &mut [LayerCache],
        kept: &mut Vec<TokKept>,
        #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] graphs: &mut TokGraphs,
        prompt: &[u32],
        run: &mut Run<'_>,
    ) -> Result<GenStats, EngineError> {
        let ids: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        self.sync();
        let started = Instant::now();
        let restore = |caches: &mut [LayerCache], k: &TokKept| {
            caches.iter_mut().zip(&k.snapshot).for_each(|(c, s)| c.restore(s));
        };
        let (logits, cached) = match self.pick_kept(&ids, kept) {
            Some((i, true)) => {
                restore(caches, &kept[i]);
                (Some(kept[i].out.clone().expect("a whole hit has its output")), ids.len())
            }
            Some((i, false)) => {
                let n = kept[i].tokens.len();
                restore(caches, &kept[i]);
                (self.prefill_tok(model, &ids[n..], caches, run.cancel), n)
            }
            None => {
                caches.iter_mut().for_each(LayerCache::reset);
                (self.prefill_tok(model, &ids, caches, run.cancel), 0)
            }
        };
        let Some(logits) = logits else {
            // The listener left during the prefill: the cache holds part of the prompt, so no
            // kept state may claim it.
            kept.clear();
            return Ok(aborted(prompt.len()));
        };
        let mut next = Vec::new();
        if self.prefix_reuse {
            let snapshot = caches.iter().map(LayerCache::snapshot).collect();
            next.push(Kept { tokens: ids.clone(), snapshot, out: Some(logits.clone()) });
        }
        // The tokens in the cache: the prompt, then every token fed back.
        let mut history = ids.clone();
        #[cfg(feature = "cuda")]
        if self.graphs() && run.constraint.is_none() {
            self.sync();
            let prompt_s = started.elapsed().as_secs_f64();
            let stats = self.graphed_tok(model, caches, graphs, logits, &mut history, run);
            self.keep_reply(kept, next, history, || {
                caches.iter().map(LayerCache::snapshot).collect()
            });
            return Ok(GenStats {
                prompt_tokens: prompt.len(),
                prompt_s,
                cached_tokens: cached,
                ..stats
            });
        }
        let mut row = read_logits(&logits);
        let prompt_s = started.elapsed().as_secs_f64();
        let decode_started = Instant::now();
        let mut last = Some(logits);
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
            if history.len() >= self.capacity {
                break Finish::Length;
            }
            let logits = model.step(self.int_tensor(&[token as i32]), caches);
            history.push(token as i32);
            row = read_logits(&logits);
            last = Some(logits);
        };
        let decode_s = decode_started.elapsed().as_secs_f64();
        if self.prefix_reuse && history.len() > ids.len() {
            let snapshot = caches.iter().map(LayerCache::snapshot).collect();
            next.push(Kept { tokens: history, snapshot, out: last });
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

    /// Prefills the token baseline with `ids` chunk by chunk, checking `cancel` between chunks:
    /// the logits of the last position, or None when the listener left first.
    fn prefill_tok(
        &self,
        model: &TokenLm,
        ids: &[i32],
        caches: &mut [LayerCache],
        cancel: &AtomicBool,
    ) -> Option<Tensor<3>> {
        let mut last = None;
        for piece in ids.chunks(self.chunk) {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            last = model.prefill(self.int_tensor(piece), caches, self.chunk);
        }
        last
    }

    /// [`Self::prefill_tok`] for the NCM (`ids` whole concepts; chunks are whole concepts too):
    /// the next concept's condition, or None when the listener left first.
    fn prefill_ncm(
        &self,
        model: &NcmLm,
        ids: &[i32],
        cache: &mut NcmCache,
        cancel: &AtomicBool,
    ) -> Option<Tensor<3>> {
        let k = model.concept_size();
        let chunk = (self.chunk - self.chunk % k).max(k);
        let mut cond = None;
        for piece in ids.chunks(chunk) {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            cond = Some(model.prefill(self.int_tensor(piece), cache, chunk));
        }
        cond
    }

    /// Keeps the prompt's state `next` and, when the reply fed tokens back, the state at the
    /// end of the reply (`history`, without a stored output: the graphs keep it on the device).
    #[cfg(feature = "cuda")]
    fn keep_reply<S, O>(
        &self,
        kept: &mut Vec<Kept<S, O>>,
        mut next: Vec<Kept<S, O>>,
        history: Vec<i32>,
        snapshot: impl FnOnce() -> S,
    ) {
        if self.prefix_reuse && next.first().is_some_and(|k| history.len() > k.tokens.len()) {
            next.push(Kept { tokens: history, snapshot: snapshot(), out: None });
        }
        *kept = next;
    }

    /// The NCM. Concepts are fixed k-token chunks from the start of the token stream, as in
    /// pretraining: the prompt's whole concepts are prefilled, and its trailing partial concept
    /// (fewer than k tokens) stays pending: it is forced as the first tokens of the first
    /// decoded concept, which the decoder then completes. A prompt shorter than one concept is
    /// left-padded to one.
    fn generate_ncm(
        &self,
        model: &NcmLm,
        cache: &mut NcmCache,
        kept: &mut Vec<NcmKept>,
        #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] graphs: &mut NcmGraphs,
        prompt: &[u32],
        run: &mut Run<'_>,
    ) -> Result<GenStats, EngineError> {
        let k = model.concept_size();
        let ids = run.ids;
        let pad = k.saturating_sub(prompt.len());
        let mut stream: Vec<i32> = vec![ids.pad as i32; pad];
        stream.extend(prompt.iter().map(|&t| t as i32));
        let whole = stream.len() - stream.len() % k;
        let (head, pending) = stream.split_at(whole);
        self.sync();
        let started = Instant::now();
        let (cond, cached) = match self.pick_kept(head, kept) {
            Some((i, true)) => {
                cache.restore(&kept[i].snapshot);
                (Some(kept[i].out.clone().expect("a whole hit has its output")), whole)
            }
            Some((i, false)) => {
                let n = kept[i].tokens.len();
                cache.restore(&kept[i].snapshot);
                (self.prefill_ncm(model, &head[n..], cache, run.cancel), n)
            }
            None => {
                cache.reset();
                (self.prefill_ncm(model, head, cache, run.cancel), 0)
            }
        };
        let Some(mut cond) = cond else {
            kept.clear();
            return Ok(aborted(prompt.len()));
        };
        // Prompt tokens served from the cache, padding excluded.
        let cached = cached.saturating_sub(pad);
        let mut next = Vec::new();
        if self.prefix_reuse {
            let snapshot = cache.snapshot();
            next.push(Kept { tokens: head.to_vec(), snapshot, out: Some(cond.clone()) });
        }
        let mut history = head.to_vec();
        self.sync();
        let prompt_s = started.elapsed().as_secs_f64();
        #[cfg(feature = "cuda")]
        if self.graphs() && run.constraint.is_none() {
            let stats = self.graphed_ncm(model, cache, graphs, cond, pending, &mut history, run);
            self.keep_reply(kept, next, history, || cache.snapshot());
            return Ok(GenStats {
                prompt_tokens: prompt.len(),
                prompt_s,
                cached_tokens: cached,
                ..stats
            });
        }
        let decode_started = Instant::now();
        let device = self.device.clone();
        let host_sampling = !run.sampler.is_greedy() || run.constraint.is_some();
        let mut empty_concepts = 0;
        let mut forced: &[i32] = pending;
        let finish = 'generate: loop {
            if run.cancel.load(Ordering::Relaxed) {
                break Finish::Stop;
            }
            // A concept's tokens: the pending prompt tokens first, then chosen on the host
            // when sampling, constrained, or completing a forced start, else by argmax on the
            // device.
            let host = host_sampling || !forced.is_empty();
            let mut ended = false;
            let (tokens, _) =
                model.decode_concept_with(cond.clone(), cache, false, &mut |j, logits| {
                    if let Some(&token) = forced.get(j) {
                        return Tensor::from_data(TensorData::new(vec![token], [1, 1]), &device);
                    }
                    if !host {
                        return logits.clone().argmax(2).reshape([1, 1]);
                    }
                    let token = if ended {
                        ids.pad
                    } else {
                        let token = run.choose(&read_logits(logits));
                        if token == ids.eoc || ids.is_stop(token) {
                            ended = true;
                        } else {
                            run.advance(token);
                        }
                        token
                    };
                    Tensor::from_data(TensorData::new(vec![token as i32], [1, 1]), &device)
                });
            let skip = forced.len();
            forced = &[];
            let picked = read_ids(&tokens);
            let before = run.emitted;
            let concept = match self.emit_concept(&picked, skip, k, run) {
                Ok(concept) => concept,
                Err(finish) => break 'generate finish,
            };
            empty_concepts = if run.emitted == before { empty_concepts + 1 } else { 0 };
            if empty_concepts >= MAX_EMPTY_CONCEPTS {
                break Finish::Stop;
            }
            if history.len() + k > self.capacity {
                break Finish::Length;
            }
            cond = model.append_tokens(self.int_tensor(&concept), cache);
            history.extend_from_slice(&concept);
        };
        let decode_s = decode_started.elapsed().as_secs_f64();
        if self.prefix_reuse && history.len() > whole {
            next.push(Kept { tokens: history, snapshot: cache.snapshot(), out: Some(cond) });
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

    /// Emits a decoded concept's tokens after its first `skip` (forced prompt tokens), up to an
    /// end-of-concept token; returns the concept to append (cut after that token and padded to
    /// `k`), or why generation ends.
    fn emit_concept(
        &self,
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
}

/// The graphed decode loops (CUDA): one graph replay per step, the token or concept read back
/// to the host once per step to stream it.
#[cfg(feature = "cuda")]
impl Core {
    /// The token baseline after its prompt pass (`logits` of the last prompt position); each
    /// token fed back is appended to `history`. The caller fills in the prompt's fields.
    fn graphed_tok(
        &self,
        model: &TokenLm,
        caches: &mut [LayerCache],
        graphs: &mut TokGraphs,
        logits: Tensor<3>,
        history: &mut Vec<i32>,
        run: &mut Run<'_>,
    ) -> GenStats {
        use ncm_model::tok::GraphedTok;
        let sampled = !run.params.is_greedy();
        let graphed = if sampled {
            let settings = device_sampling(run.params);
            let step =
                graphs.sampled.get_or_insert_with(|| GraphedTok::sampled(&self.device, settings));
            step.set_sampling(settings);
            step
        } else {
            &mut graphs.greedy
        };
        let mut token = graphed.first(logits);
        let fresh = !graphed.is_captured();
        let capture_started = Instant::now();
        graphed.prepare(model, &token, caches);
        self.sync();
        let capture_s = if fresh && graphed.is_captured() {
            capture_started.elapsed().as_secs_f64()
        } else {
            0.0
        };
        let decode_started = Instant::now();
        let finish = loop {
            if run.cancel.load(Ordering::Relaxed) {
                break Finish::Stop;
            }
            let id = read_ids(&token)[0];
            match run.emit(id) {
                Emit::Stop => break Finish::Stop,
                Emit::Limit => break Finish::Length,
                Emit::Go => {}
            }
            if history.len() >= self.capacity {
                break Finish::Length;
            }
            token = graphed.step(model, token, caches);
            history.push(id as i32);
        };
        GenStats {
            prompt_tokens: 0,
            completion_tokens: run.emitted,
            finish,
            prompt_s: 0.0,
            decode_s: decode_started.elapsed().as_secs_f64(),
            decode_tokens: run.emitted.saturating_sub(1),
            cached_tokens: 0,
            capture_s,
            path: if sampled { DecodePath::GraphedSampled } else { DecodePath::GraphedGreedy },
            queue_s: 0.0,
            mean_batch: 1.0,
            forked: false,
        }
    }

    /// The NCM after its prompt (`cond`; the pending prompt tokens `pending` start the first
    /// concept). Greedy requests replay [`ncm_model::ncm::GraphedStep`] (append the concept
    /// just written and decode the next one in one graph), or decode by self-speculation with
    /// `--speculate` (a first concept with pending tokens is decoded plainly, forcing them);
    /// sampled ones replay its device-sampling variant. Concepts are cut at the end-of-concept
    /// token and padded before they are appended, as on the eager path; each appended concept
    /// is added to `history`.
    #[allow(clippy::too_many_arguments)]
    fn graphed_ncm(
        &self,
        model: &NcmLm,
        cache: &mut NcmCache,
        graphs: &mut NcmGraphs,
        cond: Tensor<3>,
        pending: &[i32],
        history: &mut Vec<i32>,
        run: &mut Run<'_>,
    ) -> GenStats {
        use ncm_model::ncm::GraphedStep;
        let k = model.concept_size();
        let sampled = !run.params.is_greedy();
        let speculate = !sampled && self.speculate && model.has_draft_head();
        let NcmGraphs { greedy, sampled: sampled_step, append, speculator } = graphs;
        let step = if sampled {
            let settings = device_sampling(run.params);
            let step =
                sampled_step.get_or_insert_with(|| GraphedStep::sampled(&self.device, settings));
            step.set_sampling(settings);
            step
        } else {
            greedy
        };
        // Decoding starts with the first concept, whose tokens count (the eager path likewise).
        let decode_started = Instant::now();
        let mut tokens = if speculate && pending.is_empty() {
            speculator.decode(model, cond, cache)
        } else {
            step.first(model, cond, cache, pending)
        };
        let mut skip = pending.len();
        let (mut capture_s, mut captured) = (0.0, false);
        let fresh = !(if speculate { append.is_captured() } else { step.is_captured() });
        let mut empty_concepts = 0;
        let finish = loop {
            if run.cancel.load(Ordering::Relaxed) {
                break Finish::Stop;
            }
            let picked = read_ids(&tokens);
            let before = run.emitted;
            let concept = match self.emit_concept(&picked, skip, k, run) {
                Ok(concept) => concept,
                Err(finish) => break finish,
            };
            skip = 0;
            empty_concepts = if run.emitted == before { empty_concepts + 1 } else { 0 };
            if empty_concepts >= MAX_EMPTY_CONCEPTS {
                break Finish::Stop;
            }
            if history.len() + k > self.capacity {
                break Finish::Length;
            }
            // Fed back as decoded unless the concept ended early (then padded).
            let unchanged = picked.iter().zip(&concept).all(|(&p, &c)| p as i32 == c);
            let input = if unchanged { tokens } else { self.int_tensor(&concept) };
            // The first step that can be graphed captures its graph (once the cross-attention
            // ring is full): timed apart from decoding.
            let failed = step.failure().is_some() || append.failure().is_some();
            if !captured && !failed {
                let capture_started = Instant::now();
                captured = if speculate {
                    append.prepare(model, &input, cache)
                } else {
                    step.prepare(model, &input, cache)
                };
                if captured && fresh {
                    self.sync();
                    capture_s = capture_started.elapsed().as_secs_f64();
                }
            }
            tokens = if speculate {
                let cond = append.append(model, input, cache);
                speculator.decode(model, cond, cache)
            } else {
                step.step(model, input, cache)
            };
            history.extend_from_slice(&concept);
        };
        let path = match (speculate, sampled) {
            (true, _) => DecodePath::Speculative,
            (false, true) => DecodePath::GraphedSampled,
            (false, false) => DecodePath::GraphedGreedy,
        };
        GenStats {
            prompt_tokens: 0,
            completion_tokens: run.emitted,
            finish,
            prompt_s: 0.0,
            decode_s: (decode_started.elapsed().as_secs_f64() - capture_s).max(0.0),
            decode_tokens: run.emitted,
            cached_tokens: 0,
            capture_s,
            path,
            queue_s: 0.0,
            mean_batch: 1.0,
            forked: false,
        }
    }
}
