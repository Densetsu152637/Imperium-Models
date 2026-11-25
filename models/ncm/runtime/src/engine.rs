//! The model, its caches and the generation loop for both architectures.
//!
//! One [`Engine`] owns one model and one set of caches allocated at launch for the per-request
//! context size. Requests run one at a time (batch size 1, like the decode-speed gate); each
//! resets the caches, prefills its prompt and decodes until a stop token, the token limit, or the
//! end of the context.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::{Device, FloatDType, Int, Tensor, TensorData};
use ncm_model::quant::{self, QuantError, QuantSpec};
use ncm_model::weights::{self, WeightsError};
use ncm_model::{Config, LayerCache, NcmCache, NcmLm, TokenLm, cache_bytes};

use crate::constrain::Constraint;
use crate::sampling::{Sampler, SamplingParams};
use crate::vocab::{Specials, Vocab, VocabError};

/// Prompt tokens fed to the model at once.
pub const DEFAULT_PREFILL_CHUNK: usize = 512;
/// Concepts in a row with no output before an NCM reply is cut off.
const MAX_EMPTY_CONCEPTS: usize = 16;

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
}

pub struct GenRequest {
    pub prompt: Vec<u32>,
    /// Most tokens to generate.
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub constraint: Option<Box<dyn Constraint>>,
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
}

#[allow(clippy::large_enum_variant)]
enum State {
    Tok { model: TokenLm, caches: Vec<LayerCache> },
    Ncm { model: NcmLm, cache: NcmCache },
}

pub struct Engine {
    info: ModelInfo,
    vocab: Arc<Vocab>,
    device: Device,
    chunk: usize,
    /// Tokens the caches hold.
    capacity: usize,
    state: Mutex<State>,
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

impl Engine {
    /// Builds the model (loading `options.weights` or initialising random weights) and
    /// allocates the caches for `options.ctx` tokens.
    pub fn new(options: EngineOptions, vocab: Arc<Vocab>) -> Result<Self, EngineError> {
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
                (State::Tok { model, caches }, params, "tok")
            }
            Config::Ncm(cfg) => {
                let build = || NcmLm::new(cfg, &device);
                let model = load_model(weights.as_deref(), dtype, quant, &device, build)?;
                let cache = model.make_cache(capacity, 1, &device);
                let params = NcmLm::new(cfg, &device).parts().total();
                (State::Ncm { model, cache }, params, "ncm")
            }
        };
        let info = ModelInfo {
            arch,
            params,
            ctx: capacity,
            cache_bytes: cache_bytes(&config, capacity, itemsize(dtype), 1),
            dtype: format!("{dtype:?}").to_lowercase(),
            device: device_name,
            vocab_size: config.vocab_size(),
            unit,
        };
        Ok(Self {
            info,
            vocab,
            device,
            chunk: prefill_chunk.max(unit),
            capacity,
            state: Mutex::new(state),
        })
    }

    fn int_tensor(&self, ids: &[i32]) -> Tensor<2, Int> {
        Tensor::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), &self.device)
    }

    fn sync(&self) {
        self.device.sync().ok();
    }
}

fn read_logits(logits: &Tensor<3>) -> Vec<f32> {
    logits.to_data_as::<f32>().try_into_vec::<f32>().expect("logits are floats")
}

fn read_ids(tokens: &Tensor<2, Int>) -> Vec<u32> {
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
    constraint: Option<Box<dyn Constraint>>,
    ids: Specials,
    on_token: &'a mut dyn FnMut(u32) -> bool,
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

impl Backend for Engine {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }

    fn check_prompt(&self, tokens: usize) -> Result<(), EngineError> {
        if tokens == 0 {
            return Err(EngineError::EmptyPrompt);
        }
        let padded = tokens.next_multiple_of(self.info.unit);
        if padded > self.capacity {
            return Err(EngineError::ContextExceeded { prompt: tokens, ctx: self.capacity });
        }
        Ok(())
    }

    fn generate(
        &self,
        request: GenRequest,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenStats, EngineError> {
        self.check_prompt(request.prompt.len())?;
        let mut run = Run {
            sampler: Sampler::new(request.sampling),
            constraint: request.constraint,
            ids: self.vocab.ids,
            on_token,
            max_tokens: request.max_tokens.max(1),
            emitted: 0,
        };
        let mut state = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        match &mut *state {
            State::Tok { model, caches } => {
                self.generate_tok(model, caches, &request.prompt, &mut run)
            }
            State::Ncm { model, cache } => {
                self.generate_ncm(model, cache, &request.prompt, &mut run)
            }
        }
    }
}

impl Engine {
    fn generate_tok(
        &self,
        model: &TokenLm,
        caches: &mut [LayerCache],
        prompt: &[u32],
        run: &mut Run<'_>,
    ) -> Result<GenStats, EngineError> {
        for cache in caches.iter_mut() {
            cache.reset();
        }
        let ids: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        self.sync();
        let started = Instant::now();
        let logits = model
            .prefill(self.int_tensor(&ids), caches, self.chunk)
            .ok_or(EngineError::EmptyPrompt)?;
        let mut row = read_logits(&logits);
        let prompt_s = started.elapsed().as_secs_f64();
        let decode_started = Instant::now();
        let mut written = prompt.len();
        let finish = loop {
            let token = run.choose(&row);
            run.advance(token);
            match run.emit(token) {
                Emit::Stop => break Finish::Stop,
                Emit::Limit => break Finish::Length,
                Emit::Go => {}
            }
            if written >= self.capacity {
                break Finish::Length;
            }
            let logits = model.step(self.int_tensor(&[token as i32]), caches);
            written += 1;
            row = read_logits(&logits);
        };
        Ok(GenStats {
            prompt_tokens: prompt.len(),
            completion_tokens: run.emitted,
            finish,
            prompt_s,
            decode_s: decode_started.elapsed().as_secs_f64(),
            decode_tokens: run.emitted.saturating_sub(1),
        })
    }

    fn generate_ncm(
        &self,
        model: &NcmLm,
        cache: &mut NcmCache,
        prompt: &[u32],
        run: &mut Run<'_>,
    ) -> Result<GenStats, EngineError> {
        let k = model.concept_size();
        let ids = run.ids;
        cache.reset();
        // The prompt must be whole concepts: left-pad with `<|pad|>` so that it still ends
        // exactly where the reply starts.
        let pad = (k - prompt.len() % k) % k;
        let mut padded: Vec<i32> = vec![ids.pad as i32; pad];
        padded.extend(prompt.iter().map(|&t| t as i32));
        self.sync();
        let started = Instant::now();
        let mut cond = model.prefill(self.int_tensor(&padded), cache, self.chunk);
        self.sync();
        let prompt_s = started.elapsed().as_secs_f64();
        let decode_started = Instant::now();
        let mut written = padded.len();
        let device = self.device.clone();
        let host_sampling = !run.sampler.is_greedy() || run.constraint.is_some();
        let mut empty_concepts = 0;
        let finish = 'generate: loop {
            // A concept's tokens: chosen on the host when sampling or constrained, else by
            // argmax on the device.
            let mut ended = false;
            let (tokens, _) =
                model.decode_concept_with(cond.clone(), cache, false, &mut |_, logits| {
                    if !host_sampling {
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
            let tokens = read_ids(&tokens);
            let before = run.emitted;
            let mut concept: Vec<i32> = Vec::with_capacity(k);
            for &token in &tokens {
                if token == ids.eoc {
                    concept.push(token as i32);
                    break;
                }
                concept.push(token as i32);
                match run.emit(token) {
                    Emit::Stop => break 'generate Finish::Stop,
                    Emit::Limit => break 'generate Finish::Length,
                    Emit::Go => {}
                }
            }
            empty_concepts = if run.emitted == before { empty_concepts + 1 } else { 0 };
            if empty_concepts >= MAX_EMPTY_CONCEPTS {
                break Finish::Stop;
            }
            // Feed the concept back: everything after an early end is padding.
            concept.resize(k, ids.pad as i32);
            if written + k > self.capacity {
                break Finish::Length;
            }
            cond = model.append_tokens(self.int_tensor(&concept), cache);
            written += k;
        };
        Ok(GenStats {
            prompt_tokens: prompt.len(),
            completion_tokens: run.emitted,
            finish,
            prompt_s,
            decode_s: decode_started.elapsed().as_secs_f64(),
            decode_tokens: run.emitted,
        })
    }
}
