//! The Burn adapter of the `ncm-exec` interface ([`BurnBackend`]): CPU and CUDA, the token
//! baseline, the k = 4 NCM and the hybrid.
//!
//! This is the sequential engine's old decode machinery (eager steps, the captured CUDA graphs,
//! prefix snapshots, the hybrid's fast session) moved behind [`ExecSession`], so that the
//! runtime drives every backend the same way. Nothing here changes how a step runs:
//!
//! - A session keeps the output its next step needs (the last logits, or the NCM's condition) on
//!   the device. A graphed step's token stays in the graph's output buffer and is fed back from
//!   there (the host copy only has to match what the engine feeds).
//! - Resets and restores write into the caches' own buffers, so captured graphs stay valid.
//! - The batcher (`ncm-server`'s continuous batching) is a Burn-only side door: it takes the
//!   model and the first session's caches out with [`BurnBackend::into_parts`].

use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::time::Instant;

use burn::tensor::{Device, Int, Tensor, TensorData};
use ncm_exec::hybrid::{BlockLogic, BlockOut, BlockPhase, DecodeOptions, Hooks, PassResult};
use ncm_exec::{
    Capabilities, Dtype, ExecBackend, ExecError, ExecSession, Footprint, FootprintRequest,
    HostChoice, ModelKind, Part, PendingOutput, PickMode, PickSpec, PrefillOut, Quant,
    SnapshotInfo, StepOut, Tier,
};

use crate::cache::LayerSnapshot;
use crate::config::Config;
use crate::hybrid::{BlockRun, FastSession, HybridLm, HybridSnapshot};
use crate::ncm::NcmSnapshot;
use crate::{LayerCache, NcmCache, NcmLm, TokenLm, cache_bytes};

/// The model a [`BurnBackend`] runs.
#[allow(clippy::large_enum_variant)]
pub enum BurnModel {
    Tok(TokenLm),
    Ncm(NcmLm),
    Hybrid(HybridLm),
}

/// The caches of one session, for the continuous batcher.
#[allow(clippy::large_enum_variant)]
pub enum BurnCaches {
    Tok(Vec<LayerCache>),
    Ncm(NcmCache),
    Hybrid(Box<FastSession>),
}

/// Launch-time facts the backend reports.
#[derive(Clone, Debug)]
pub struct BurnInfo {
    pub config: Config,
    pub dtype: Dtype,
    pub quant: Quant,
    /// Parameters of the dense model (a quantised model holds codes, so this is counted on a
    /// dense one).
    pub params: usize,
    /// `cpu` or `cuda`.
    pub device_name: String,
    /// `--speculate`: self-speculative NCM decoding where the model has a draft head.
    pub speculate: bool,
}

/// The Burn backend over one loaded model.
pub struct BurnBackend {
    model: Arc<BurnModel>,
    device: Device,
    info: BurnInfo,
}

impl BurnBackend {
    pub fn new(model: BurnModel, device: Device, info: BurnInfo) -> Self {
        Self { model: Arc::new(model), device, info }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn info(&self) -> &BurnInfo {
        &self.info
    }

    pub fn model(&self) -> &BurnModel {
        &self.model
    }

    /// The model and the session's caches, for the continuous batcher. The session must be
    /// this backend's own (the last one standing).
    pub fn into_parts(self, session: BurnSession) -> (BurnModel, BurnCaches) {
        let caches = session.into_caches();
        let model = Arc::try_unwrap(self.model)
            .unwrap_or_else(|_| panic!("a session outlived its backend's hand-over"));
        (model, caches)
    }

    fn kind(&self) -> ModelKind {
        match &*self.model {
            BurnModel::Tok(_) => ModelKind::Tok,
            BurnModel::Ncm(_) => ModelKind::Ncm,
            BurnModel::Hybrid(_) => ModelKind::Hybrid,
        }
    }

    /// Whether requests may take the graphed device paths: CUDA with the fused kernels and
    /// graphs on.
    fn graphs(&self) -> bool {
        graphs_on(&self.device)
    }
}

#[cfg(feature = "cuda")]
fn graphs_on(device: &Device) -> bool {
    crate::fused::graphs_active(device)
}

#[cfg(not(feature = "cuda"))]
fn graphs_on(_device: &Device) -> bool {
    false
}

impl ExecBackend for BurnBackend {
    type Session = BurnSession;

    fn capabilities(&self) -> Capabilities {
        let cfg = &self.info.config;
        let graphs = self.graphs();
        Capabilities {
            backend: format!("burn-{}", self.info.device_name),
            model: self.kind(),
            unit: cfg.unit(),
            block: match cfg {
                Config::Ncm(c) => c.hybrid.as_ref().map_or(0, |h| h.block),
                Config::Tok(_) => 0,
            },
            prefill_buckets: Vec::new(),
            dtypes: vec![self.info.dtype],
            device_sampling: graphs,
            graphs,
            speculative: cfg!(feature = "cuda")
                && matches!(&*self.model, BurnModel::Ncm(m) if m.has_draft_head()),
            batching: true,
        }
    }

    fn footprint(&self, req: &FootprintRequest) -> Footprint {
        let tier = if self.info.device_name == "cuda" { Tier::Device } else { Tier::SystemRam };
        let items = req.dtype.itemsize();
        let weight_bits = req.quant.bits(8 * items);
        let weights = (self.info.params as u64 * weight_bits as u64).div_ceil(8);
        let per_slot = match &self.info.config {
            Config::Ncm(c) if c.hybrid.is_some() => {
                crate::hybrid::hybrid_cache_bytes(c, req.ctx, items, 1)
            }
            cfg => cache_bytes(cfg, req.ctx, items, 1),
        };
        // The cache is reported as one figure: the per-part split (rings, indexer keys, DeltaNet
        // state) belongs to the offload planner, which refines it.
        Footprint::new().with(tier, Part::Weights, weights).with(
            tier,
            Part::LatentCache,
            per_slot as u64 * req.slots.max(1) as u64,
        )
    }

    fn new_session(&self, ctx: usize) -> Result<BurnSession, ExecError> {
        let device = &self.device;
        let inner = match &*self.model {
            BurnModel::Tok(m) => Inner::Tok(TokSession {
                caches: m.make_cache(ctx, 1, device),
                pending: None,
                #[cfg(feature = "cuda")]
                graphs: TokGraphs::default(),
                #[cfg(feature = "cuda")]
                last: None,
                #[cfg(feature = "cuda")]
                sampled_run: false,
                capture_s: 0.0,
            }),
            BurnModel::Ncm(m) => Inner::Ncm(NcmSession {
                cache: m.make_cache(ctx, 1, device),
                pending: None,
                #[cfg(feature = "cuda")]
                graphs: NcmGraphs::default(),
                #[cfg(feature = "cuda")]
                last: None,
                #[cfg(feature = "cuda")]
                run: NcmRun::default(),
                capture_s: 0.0,
            }),
            BurnModel::Hybrid(m) => Inner::Hybrid(Box::new(FastSession::new(m, ctx, device))),
        };
        Ok(BurnSession {
            model: self.model.clone(),
            device: self.device.clone(),
            speculate: self.info.speculate,
            inner,
        })
    }

    fn warm_up(&self, shapes: usize, chunk: usize) {
        match &*self.model {
            BurnModel::Ncm(m) => m.warm_up(shapes, chunk, &self.device),
            BurnModel::Hybrid(_) => {}
            BurnModel::Tok(m) => {
                // A throwaway cache: a prefill of `shapes` tokens, then one step.
                let mut caches = m.make_cache(shapes + 1, 1, &self.device);
                let prompt = int_tensor(&self.device, &vec![0; shapes]);
                m.prefill(prompt, &mut caches, chunk);
                let logits = m.step(int_tensor(&self.device, &[0]), &mut caches);
                drop(logits.into_data());
            }
        }
    }

    fn sync(&self) {
        self.device.sync().ok();
    }

    fn describe(&self) -> String {
        self.info.device_name.clone()
    }
}

// ---------------------------------------------------------------------- helpers

fn int_tensor(device: &Device, ids: &[i32]) -> Tensor<2, Int> {
    Tensor::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), device)
}

/// A `[1, 1, V]` logits tensor as a host row.
pub fn read_logits(logits: &Tensor<3>) -> Vec<f32> {
    logits.to_data_as::<f32>().try_into_vec::<f32>().expect("logits are floats")
}

/// A `[1, n]` token tensor as ids.
pub fn read_ids(tokens: &Tensor<2, Int>) -> Vec<u32> {
    tokens
        .to_data_as::<i32>()
        .try_into_vec::<i32>()
        .expect("tokens are integers")
        .into_iter()
        .map(|id| id.max(0) as u32)
        .collect()
}

/// The device sampler's settings of a backend-neutral draw row.
#[cfg(feature = "cuda")]
fn sample_settings(r: &ncm_exec::DrawRow) -> crate::sample::SampleSettings {
    crate::hybrid::sample_settings(r)
}

#[cfg(feature = "cuda")]
fn row_of(pick: &PickSpec) -> Option<ncm_exec::DrawRow> {
    match &pick.mode {
        PickMode::Sampled(row) => Some(*row),
        _ => None,
    }
}

fn unsupported<T>(what: &str) -> Result<T, ExecError> {
    Err(ExecError::Unsupported(what.into()))
}

// ---------------------------------------------------------------------- the session

/// A Burn decode session: one request stream's caches and kept graphs.
pub struct BurnSession {
    model: Arc<BurnModel>,
    device: Device,
    speculate: bool,
    inner: Inner,
}

#[allow(clippy::large_enum_variant)]
enum Inner {
    Tok(TokSession),
    Ncm(NcmSession),
    Hybrid(Box<FastSession>),
}

/// The token baseline's decode graphs, kept across requests (their caches keep their buffers).
#[cfg(feature = "cuda")]
#[derive(Default)]
struct TokGraphs {
    greedy: crate::tok::GraphedTok,
    sampled: Option<crate::tok::GraphedTok>,
}

struct TokSession {
    caches: Vec<LayerCache>,
    /// The next token's logits, once the prompt (or an eager step) is in.
    pending: Option<Tensor<3>>,
    #[cfg(feature = "cuda")]
    graphs: TokGraphs,
    /// The graphed step's last token: its ids and the graph's output buffer.
    #[cfg(feature = "cuda")]
    last: Option<(Vec<i32>, Tensor<2, Int>)>,
    #[cfg(feature = "cuda")]
    sampled_run: bool,
    capture_s: f64,
}

/// The NCM's decode graphs, kept across requests.
#[cfg(feature = "cuda")]
struct NcmGraphs {
    greedy: crate::ncm::GraphedStep,
    sampled: Option<crate::ncm::GraphedStep>,
    append: crate::ncm::GraphedAppend,
    speculator: crate::Speculator,
}

#[cfg(feature = "cuda")]
impl Default for NcmGraphs {
    fn default() -> Self {
        Self {
            greedy: Default::default(),
            sampled: None,
            append: Default::default(),
            speculator: crate::Speculator::new(crate::Acceptance::Verify),
        }
    }
}

/// What the current graphed NCM request is doing.
#[cfg(feature = "cuda")]
#[derive(Default)]
struct NcmRun {
    sampled: bool,
    speculate: bool,
    /// The step (or append) graph was not captured yet when the request began.
    fresh: bool,
    captured: bool,
}

struct NcmSession {
    cache: NcmCache,
    /// The next concept's condition.
    pending: Option<Tensor<3>>,
    #[cfg(feature = "cuda")]
    graphs: NcmGraphs,
    #[cfg(feature = "cuda")]
    last: Option<(Vec<i32>, Tensor<2, Int>)>,
    #[cfg(feature = "cuda")]
    run: NcmRun,
    capture_s: f64,
}

/// A point of a [`BurnSession`] ([`ExecSession::snapshot`]).
#[allow(clippy::large_enum_variant)]
pub enum BurnSnapshot {
    Tok { layers: Vec<LayerSnapshot>, out: Option<Tensor<3>> },
    Ncm { cache: NcmSnapshot, out: Option<Tensor<3>> },
    Hybrid(Box<HybridSnapshot>),
}

impl SnapshotInfo for BurnSnapshot {
    fn has_output(&self) -> bool {
        match self {
            Self::Tok { out, .. } | Self::Ncm { out, .. } => out.is_some(),
            // The hybrid decodes from its state.
            Self::Hybrid(_) => true,
        }
    }
}

impl BurnSession {
    /// The session's caches, for the continuous batcher.
    pub fn into_caches(self) -> BurnCaches {
        match self.inner {
            Inner::Tok(t) => BurnCaches::Tok(t.caches),
            Inner::Ncm(n) => BurnCaches::Ncm(n.cache),
            Inner::Hybrid(h) => BurnCaches::Hybrid(h),
        }
    }

    fn tok(&mut self) -> Result<(&TokenLm, &mut TokSession, &Device), ExecError> {
        match (&*self.model, &mut self.inner) {
            (BurnModel::Tok(m), Inner::Tok(s)) => Ok((m, s, &self.device)),
            _ => unsupported("this session is not the token baseline's"),
        }
    }

    fn ncm(&mut self) -> Result<(&NcmLm, &mut NcmSession, &Device), ExecError> {
        match (&*self.model, &mut self.inner) {
            (BurnModel::Ncm(m), Inner::Ncm(s)) => Ok((m, s, &self.device)),
            _ => unsupported("this session is not the k = 4 NCM's"),
        }
    }

    fn hybrid(&mut self) -> Result<(&HybridLm, &mut FastSession), ExecError> {
        match (&*self.model, &mut self.inner) {
            (BurnModel::Hybrid(m), Inner::Hybrid(s)) => Ok((m, s)),
            _ => unsupported("this session is not the hybrid's"),
        }
    }
}

impl ExecSession for BurnSession {
    type BlockRun = BlockRun;
    type Snapshot = BurnSnapshot;

    fn begin_request(&mut self, seed: u64) {
        if let Inner::Hybrid(s) = &mut self.inner {
            s.begin_request(seed);
        }
    }

    fn reset(&mut self) {
        match (&*self.model, &mut self.inner) {
            (BurnModel::Tok(_), Inner::Tok(s)) => {
                s.caches.iter_mut().for_each(LayerCache::reset);
                s.pending = None;
                #[cfg(feature = "cuda")]
                {
                    s.last = None;
                }
            }
            (BurnModel::Ncm(_), Inner::Ncm(s)) => {
                s.cache.reset();
                s.pending = None;
                #[cfg(feature = "cuda")]
                {
                    s.last = None;
                }
            }
            (BurnModel::Hybrid(m), Inner::Hybrid(s)) => s.reset(m),
            _ => {}
        }
    }

    fn snapshot(&self) -> BurnSnapshot {
        match &self.inner {
            Inner::Tok(s) => BurnSnapshot::Tok {
                layers: s.caches.iter().map(LayerCache::snapshot).collect(),
                out: s.pending.clone(),
            },
            Inner::Ncm(s) => {
                BurnSnapshot::Ncm { cache: s.cache.snapshot(), out: s.pending.clone() }
            }
            Inner::Hybrid(s) => BurnSnapshot::Hybrid(Box::new(s.snapshot())),
        }
    }

    fn restore(&mut self, snapshot: &BurnSnapshot) {
        match (&mut self.inner, snapshot) {
            (Inner::Tok(s), BurnSnapshot::Tok { layers, out }) => {
                s.caches.iter_mut().zip(layers).for_each(|(c, l)| c.restore(l));
                s.pending = out.clone();
                #[cfg(feature = "cuda")]
                {
                    s.last = None;
                }
            }
            (Inner::Ncm(s), BurnSnapshot::Ncm { cache, out }) => {
                s.cache.restore(cache);
                s.pending = out.clone();
                #[cfg(feature = "cuda")]
                {
                    s.last = None;
                }
            }
            (Inner::Hybrid(s), BurnSnapshot::Hybrid(h)) => s.restore(h),
            _ => panic!("a snapshot of another model kind"),
        }
    }

    fn tokens(&self) -> &[i32] {
        match &self.inner {
            Inner::Hybrid(s) => s.tokens(),
            _ => &[],
        }
    }

    fn prefill(&mut self, tokens: &[i32], chunk: usize) -> Result<PrefillOut, ExecError> {
        let n = tokens.len();
        match &*self.model {
            BurnModel::Tok(_) => {
                let (m, s, device) = self.tok()?;
                s.pending = m.prefill(int_tensor(device, tokens), &mut s.caches, chunk);
                Ok(PrefillOut { prefilled: n, output: PendingOutput::Logits })
            }
            BurnModel::Ncm(_) => {
                let (m, s, device) = self.ncm()?;
                s.pending = Some(m.prefill(int_tensor(device, tokens), &mut s.cache, chunk));
                Ok(PrefillOut { prefilled: n, output: PendingOutput::Condition })
            }
            BurnModel::Hybrid(_) => {
                let (m, s) = self.hybrid()?;
                s.prefill(m, tokens);
                Ok(PrefillOut { prefilled: n, output: PendingOutput::None })
            }
        }
    }

    fn pending_logits(&mut self) -> Option<Vec<f32>> {
        match &self.inner {
            Inner::Tok(s) => s.pending.as_ref().map(read_logits),
            _ => None,
        }
    }

    fn decode_step(&mut self, feed: &[i32], pick: &PickSpec) -> Result<StepOut, ExecError> {
        match &*self.model {
            BurnModel::Tok(_) => self.tok_step(feed, pick),
            BurnModel::Ncm(_) => self.ncm_step(feed, pick),
            BurnModel::Hybrid(_) => unsupported("the hybrid decodes in blocks"),
        }
    }

    fn decode_concept_host(
        &mut self,
        feed: &[i32],
        forced: &[i32],
        choose: &mut HostChoice<'_>,
    ) -> Result<Vec<u32>, ExecError> {
        let (m, s, device) = self.ncm()?;
        let cond = ncm_cond(m, s, device, feed)?;
        let (tokens, _) = m.decode_concept_with(cond, &mut s.cache, false, &mut |j, logits| {
            if let Some(&token) = forced.get(j) {
                return Tensor::from_data(TensorData::new(vec![token], [1, 1]), device);
            }
            let token = choose(j, &mut || read_logits(logits));
            Tensor::from_data(TensorData::new(vec![token], [1, 1]), device)
        });
        Ok(read_ids(&tokens))
    }

    fn append(&mut self, tokens: &[i32]) -> Result<(), ExecError> {
        let (m, s, device) = self.ncm()?;
        ncm_cond(m, s, device, tokens).map(|_| ())
    }

    fn take_capture_s(&mut self) -> f64 {
        match &mut self.inner {
            Inner::Tok(s) => std::mem::take(&mut s.capture_s),
            Inner::Ncm(s) => std::mem::take(&mut s.capture_s),
            Inner::Hybrid(_) => 0.0,
        }
    }

    fn begin_block(&mut self, limit: usize, opts: &DecodeOptions) -> Result<BlockRun, ExecError> {
        let (m, s) = self.hybrid()?;
        Ok(s.begin_block(m, limit, opts))
    }

    fn phase(&self, run: &BlockRun) -> BlockPhase {
        run.phase()
    }

    fn logic<'a>(&self, run: &'a BlockRun) -> &'a BlockLogic {
        run
    }

    fn pass(
        &mut self,
        run: &mut BlockRun,
        opts: &DecodeOptions,
        hooks: &mut Hooks,
    ) -> Result<PassResult, ExecError> {
        let (m, s) = self.hybrid()?;
        Ok(s.pass_once(m, run, opts, hooks))
    }

    fn fill(
        &mut self,
        run: &mut BlockRun,
        opts: &DecodeOptions,
        hooks: &mut Hooks,
    ) -> Result<i32, ExecError> {
        let (m, s) = self.hybrid()?;
        Ok(s.fill_once(m, run, opts, hooks))
    }

    fn end_block(&mut self, run: BlockRun) -> Result<BlockOut, ExecError> {
        let (m, s) = self.hybrid()?;
        Ok(s.end_block(m, run))
    }

    fn decode_block(
        &mut self,
        limit: usize,
        opts: &DecodeOptions,
        hooks: &mut Hooks,
    ) -> Result<BlockOut, ExecError> {
        let (m, s) = self.hybrid()?;
        // Options the fast passes do not serve (host draws on the CPU, the test hooks that
        // rewrite drafts or codes) take the reference loop.
        if !s.serves(opts) || hooks.draft.is_some() || hooks.codes.is_some() {
            return Ok(s.decode_block(m, limit, opts, hooks));
        }
        ncm_exec::drive_block(self, limit, opts, hooks)
    }

    fn failure(&self) -> Option<String> {
        match &self.inner {
            Inner::Hybrid(s) => s.failure(),
            #[cfg(feature = "cuda")]
            Inner::Tok(s) => s
                .graphs
                .greedy
                .failure()
                .or_else(|| s.graphs.sampled.as_ref().and_then(|g| g.failure()))
                .map(String::from),
            #[cfg(feature = "cuda")]
            Inner::Ncm(s) => s
                .graphs
                .greedy
                .failure()
                .or(s.graphs.append.failure())
                .or_else(|| s.graphs.sampled.as_ref().and_then(|g| g.failure()))
                .map(String::from),
            #[cfg(not(feature = "cuda"))]
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------- token baseline steps

impl BurnSession {
    fn tok_step(&mut self, feed: &[i32], pick: &PickSpec) -> Result<StepOut, ExecError> {
        let (m, s, device) = self.tok()?;
        match &pick.mode {
            PickMode::Host => {
                // The host chooses: an eager step, then the logits.
                if let Some(&token) = feed.first() {
                    s.pending = Some(m.step(int_tensor(device, &[token]), &mut s.caches));
                }
                match &s.pending {
                    Some(logits) => Ok(StepOut::Logits(read_logits(logits))),
                    None => unsupported("a step without a prompt"),
                }
            }
            #[cfg(feature = "cuda")]
            _ if graphs_on(device) => tok_graphed(m, s, device, feed, pick),
            PickMode::Greedy => {
                if let Some(&token) = feed.first() {
                    s.pending = Some(m.step(int_tensor(device, &[token]), &mut s.caches));
                }
                let Some(logits) = &s.pending else {
                    return unsupported("a step without a prompt");
                };
                let row = read_logits(logits);
                Ok(StepOut::Picked(vec![argmax(&row) as u32]))
            }
            PickMode::Sampled(_) => unsupported("device sampling needs CUDA with graphs on"),
        }
    }
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

/// The token baseline's graphed step: the first call (no `feed`) picks from the held logits and
/// captures the graph; later calls feed the token back and replay it.
#[cfg(feature = "cuda")]
fn tok_graphed(
    model: &TokenLm,
    s: &mut TokSession,
    device: &Device,
    feed: &[i32],
    pick: &PickSpec,
) -> Result<StepOut, ExecError> {
    use crate::tok::GraphedTok;
    let row = row_of(pick);
    if feed.is_empty() {
        let Some(logits) = s.pending.take() else {
            return unsupported("a step without a prompt");
        };
        s.sampled_run = row.is_some();
        let graphed = match row.as_ref().map(sample_settings) {
            Some(settings) => {
                let step =
                    s.graphs.sampled.get_or_insert_with(|| GraphedTok::sampled(device, settings));
                step.set_sampling(settings);
                step
            }
            None => &mut s.graphs.greedy,
        };
        let token = graphed.first(logits);
        let fresh = !graphed.is_captured();
        let started = Instant::now();
        graphed.prepare(model, &token, &mut s.caches);
        device.sync().ok();
        s.capture_s =
            if fresh && graphed.is_captured() { started.elapsed().as_secs_f64() } else { 0.0 };
        let ids = read_ids(&token);
        s.last = Some((ids.iter().map(|&i| i as i32).collect(), token));
        return Ok(StepOut::Picked(ids));
    }
    let input = match &s.last {
        Some((host, tensor)) if host.as_slice() == feed => tensor.clone(),
        _ => int_tensor(device, feed),
    };
    let graphed = if s.sampled_run {
        s.graphs.sampled.as_mut().expect("a sampled request began")
    } else {
        &mut s.graphs.greedy
    };
    let token = graphed.step(model, input, &mut s.caches);
    let ids = read_ids(&token);
    s.last = Some((ids.iter().map(|&i| i as i32).collect(), token));
    Ok(StepOut::Picked(ids))
}

// ---------------------------------------------------------------------- NCM steps

/// The condition the next concept decodes from: the held one, or the one after appending
/// `feed` (a concept just written).
fn ncm_cond(
    model: &NcmLm,
    s: &mut NcmSession,
    device: &Device,
    feed: &[i32],
) -> Result<Tensor<3>, ExecError> {
    if feed.is_empty() {
        return match &s.pending {
            Some(cond) => Ok(cond.clone()),
            None => unsupported("a step without a prompt"),
        };
    }
    let cond = model.append_tokens(int_tensor(device, feed), &mut s.cache);
    s.pending = Some(cond.clone());
    Ok(cond)
}

impl BurnSession {
    fn ncm_step(&mut self, feed: &[i32], pick: &PickSpec) -> Result<StepOut, ExecError> {
        let speculate_asked = pick.speculate && self.speculate;
        let (m, s, device) = self.ncm()?;
        if pick.mode == PickMode::Host {
            return unsupported("host-chosen concepts go through decode_concept_host");
        }
        #[cfg(feature = "cuda")]
        if graphs_on(device) {
            return ncm_graphed(m, s, device, feed, pick, speculate_asked);
        }
        let _ = speculate_asked;
        if matches!(pick.mode, PickMode::Sampled(_)) {
            return unsupported("device sampling needs CUDA with graphs on");
        }
        // Eager greedy: argmax on the device, after any forced start.
        let cond = ncm_cond(m, s, device, feed)?;
        let forced = &pick.forced;
        let (tokens, _) =
            m.decode_concept_with(
                cond,
                &mut s.cache,
                false,
                &mut |j, logits| match forced.get(j) {
                    Some(&token) => Tensor::from_data(TensorData::new(vec![token], [1, 1]), device),
                    None => crate::fused::argmax_last(logits.clone()).reshape([1, 1]),
                },
            );
        Ok(StepOut::Picked(read_ids(&tokens)))
    }
}

/// The NCM's graphed step. Greedy requests replay [`crate::ncm::GraphedStep`] (append the concept
/// just written and decode the next one in one graph), or decode by self-speculation (a first
/// concept with forced tokens is decoded plainly); sampled ones replay its device-sampling
/// variant. The first call (no `feed`) decodes from the held condition.
#[cfg(feature = "cuda")]
fn ncm_graphed(
    model: &NcmLm,
    s: &mut NcmSession,
    device: &Device,
    feed: &[i32],
    pick: &PickSpec,
    speculate_asked: bool,
) -> Result<StepOut, ExecError> {
    use crate::ncm::GraphedStep;
    let row = row_of(pick);
    let NcmGraphs { greedy, sampled: sampled_step, append, speculator } = &mut s.graphs;
    if feed.is_empty() {
        let Some(cond) = s.pending.take() else {
            return unsupported("a step without a prompt");
        };
        let sampled = row.is_some();
        let speculate = !sampled && speculate_asked && model.has_draft_head();
        let step = match row.as_ref().map(sample_settings) {
            Some(settings) => {
                let step =
                    sampled_step.get_or_insert_with(|| GraphedStep::sampled(device, settings));
                step.set_sampling(settings);
                step
            }
            None => greedy,
        };
        let fresh = !(if speculate { append.is_captured() } else { step.is_captured() });
        s.run = NcmRun { sampled, speculate, fresh, captured: false };
        s.capture_s = 0.0;
        let tokens = if speculate && pick.forced.is_empty() {
            speculator.decode(model, cond, &s.cache)
        } else {
            step.first(model, cond, &mut s.cache, &pick.forced)
        };
        let ids = read_ids(&tokens);
        s.last = Some((ids.iter().map(|&i| i as i32).collect(), tokens));
        return Ok(StepOut::Picked(ids));
    }
    let input = match &s.last {
        Some((host, tensor)) if host.as_slice() == feed => tensor.clone(),
        _ => int_tensor(device, feed),
    };
    let step = if s.run.sampled {
        sampled_step.as_mut().expect("a sampled request began")
    } else {
        greedy
    };
    // The first step that can be graphed captures its graph (once the cross-attention ring is
    // full): timed apart from decoding.
    let failed = step.failure().is_some() || append.failure().is_some();
    if !s.run.captured && !failed {
        let started = Instant::now();
        s.run.captured = if s.run.speculate {
            append.prepare(model, &input, &mut s.cache)
        } else {
            step.prepare(model, &input, &mut s.cache)
        };
        if s.run.captured && s.run.fresh {
            device.sync().ok();
            s.capture_s = started.elapsed().as_secs_f64();
        }
    }
    let tokens = if s.run.speculate {
        let cond = append.append(model, input, &mut s.cache);
        speculator.decode(model, cond, &s.cache)
    } else {
        step.step(model, input, &mut s.cache)
    };
    let ids = read_ids(&tokens);
    s.last = Some((ids.iter().map(|&i| i as i32).collect(), tokens));
    Ok(StepOut::Picked(ids))
}
