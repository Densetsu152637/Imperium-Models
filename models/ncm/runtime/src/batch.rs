//! Continuous batching (`--parallel N`, N > 1): up to N requests decode together, one batched
//! model step per iteration, on the engine's single worker thread.
//!
//! Each slot owns its caches (sized for the per-request context) and its kept prompt prefixes
//! ([`Kept`]). Every iteration of [`Batcher::serve`]:
//!
//! 1. takes new requests into a FIFO queue, and drops requests whose listener left;
//! 2. starts queued requests on free slots: the slot whose kept prefix the prompt extends
//!    furthest (restored, then only the rest prefilled), else the least recently used one. A
//!    request whose prompt equals one a slot is prefilling right now waits instead, and is
//!    **forked** from that slot when its prefill ends: the caches are copied into the waiting
//!    request's slot ([`ncm_model::NcmCache::copy_from`]), so identical prompts (the samples of
//!    a self-consistency call) are prefilled once;
//! 3. runs one prefill chunk of every prefilling slot (round robin from the next one, up to
//!    [`PREFILL_BUDGET`] tokens in all): each slot's chunk as long as its prompt has left, up
//!    to the chunk size, all packed into one varlen forward ([`TokenLm::prefill_packed`],
//!    [`NcmLm::append_packed`], [`HybridSession::prefill_packed`]: one row, each slot's segment
//!    at its own position, nothing padded), so that long prompts interleave with the others'
//!    decoding;
//! 4. runs one decode step over every decoding slot: one token per slot for the token baseline,
//!    one concept for the NCM. The batched step ([`TokenLm::step_slots`],
//!    [`NcmLm::append_slots`] / [`NcmLm::decode_slots`]) runs the norms, input projections and
//!    MLPs over all rows at once and each slot's attention against its own caches.
//!
//! Tokens are chosen per slot with each request's own settings. On CUDA with graphs on, the
//! token baseline's unconstrained rows replay one captured graph per set of decoding slots
//! ([`ncm_model::tok::GraphedTokSlots`]: batched step and the device sampler with one settings
//! set and counter per row), the rest step eagerly with host samplers; the NCM steps eagerly,
//! greedy unconstrained tokens picked on the device (TODO: NCM graphs per slot set); with
//! `--speculate` and a draft head, all-greedy slot sets write their concepts by batched
//! self-speculation instead ([`ncm_model::speculate_slots`]). With one
//! slot the engine runs [`crate::engine`]'s sequential loop instead, unchanged.
//!
//! The NCM hybrid batches at block granularity: each decoding slot has an open block
//! ([`ncm_model::hybrid::BlockRun`]); per iteration one batched block pass per pass mode runs
//! over the slots due one (first pass, re-decode or recheck: rows = slots × block, per-slot
//! memory rings and known prefixes, acceptance per row on the device), then one batched fill
//! pass over the slots on their autoregressive fill, both replayed from graphs per slot set on
//! CUDA ([`ncm_model::hybrid::PassGraphs`]); prefill chunks batch through one encoder pass
//! ([`ncm_model::hybrid::HybridSession::prefill_packed`]), and identical prompts fork.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::Instant;

use burn::tensor::{Device, Int, Tensor, TensorData};
use ncm_model::hybrid::{
    BlockPhase, BlockRun, DecodeOptions, DrawSettings, FastSession, HybridLm, HybridSession,
    HybridSnapshot, PassGraphs, PassMode, ar_pass_slots, block_pass_slots,
};
use ncm_model::{LayerCache, NcmCache, NcmLm, TokenLm};

use crate::constrain::Constraint;
use crate::engine::{
    Core, DecodePath, EngineError, Event, Finish, GenStats, HybridAcceptance, Job, Kept,
    MAX_EMPTY_CONCEPTS, State, panic_message, read_ids, read_logits,
};
use crate::sampling::{Sampler, argmax};
use crate::vocab::Specials;

/// What `/props` reports about the slots and the queue ([`crate::engine::Backend::slots`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SlotsReport {
    pub slots: Vec<SlotView>,
    /// Requests waiting for a slot (or for the prefill they will be forked from).
    pub queued: usize,
    /// Requests finished since launch.
    pub served: u64,
    /// Batched decode steps run, and the rows they decoded (their mean is the mean batch).
    pub decode_steps: u64,
    pub decode_rows: u64,
    /// Requests started by forking another slot's prefill.
    pub forks: u64,
}

/// One slot in [`SlotsReport`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SlotView {
    pub id: usize,
    /// `idle`, `prefill` or `decode`.
    pub state: &'static str,
    /// Tokens the slot's caches hold.
    pub tokens: usize,
    pub prompt_tokens: usize,
    pub generated: usize,
}

/// One slot's caches.
enum Caches {
    Tok(Vec<LayerCache>),
    Ncm(NcmCache),
    Hybrid(Box<FastSession>),
}

/// A kept prefix of one slot (see [`crate::engine`]): its snapshot restores only that slot.
enum SlotKept {
    Tok(Kept<Vec<ncm_model::cache::LayerSnapshot>, Tensor<3>>),
    Ncm(Kept<ncm_model::ncm::NcmSnapshot, Tensor<3>>),
    /// The hybrid's points need no output: decoding starts from the session.
    Hybrid(Box<HybridSnapshot>),
}

impl SlotKept {
    fn tokens(&self) -> &[i32] {
        match self {
            Self::Tok(k) => &k.tokens,
            Self::Ncm(k) => &k.tokens,
            Self::Hybrid(k) => k.tokens(),
        }
    }

    fn has_out(&self) -> bool {
        match self {
            Self::Tok(k) => k.out.is_some(),
            Self::Ncm(k) => k.out.is_some(),
            Self::Hybrid(_) => true,
        }
    }
}

/// A hybrid request's block state: its options, the open block's decode, what its blocks did.
struct HybridRequest {
    opts: DecodeOptions,
    run: Option<BlockRun>,
    acceptance: HybridAcceptance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Prefill,
    Decode,
}

/// A request on a slot.
struct Active {
    events: mpsc::SyncSender<Event>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    sampler: Sampler,
    constraint: Option<Box<dyn Constraint>>,
    max_tokens: usize,
    emitted: usize,
    prompt_tokens: usize,
    /// What the caches must hold before decoding: the prompt (for the NCM its whole concepts,
    /// after any left padding), of which `prefilled` are in.
    stream: Vec<i32>,
    prefilled: usize,
    /// The NCM's trailing partial concept, forced as the start of the first decoded one, and
    /// the left padding of a prompt shorter than one concept.
    pending: Vec<i32>,
    pad: usize,
    phase: Phase,
    /// The next token's logits (token baseline) or the next concept's condition (NCM), each
    /// `[1, 1, ..]`, once the prompt is in.
    out: Option<Tensor<3>>,
    /// The tokens in the caches: the stream, then every token fed back.
    history: Vec<i32>,
    /// The prompt's kept state, waiting to replace the slot's list when the request ends.
    prompt_kept: Option<SlotKept>,
    cached: usize,
    queued_s: f64,
    started: Instant,
    prompt_s: f64,
    decode_started: Option<Instant>,
    /// Decode steps taken and the batch rows they had in total.
    steps: usize,
    rows: usize,
    empty_concepts: usize,
    forked: bool,
    /// The next token, already drawn on the device by a batched graph replay.
    next: Option<u32>,
    /// NCM: the concept just written, appended at the start of the next step (with the next
    /// concept's decoder pass in one graph replay when the slot set has one).
    unappended: Option<Vec<i32>>,
    /// The device sampler's settings (greedy: top-k 1), counter advanced per draw.
    #[cfg(feature = "cuda")]
    device: ncm_model::sample::SampleSettings,
    /// The hybrid's block decode (None for the other models).
    hybrid: Option<HybridRequest>,
}

struct Slot {
    caches: Caches,
    kept: Vec<SlotKept>,
    last_used: u64,
    active: Option<Active>,
}

/// Most prompt tokens one prefill step packs over all prefilling slots: it bounds how long a
/// prefill step holds up the decoding slots.
const PREFILL_BUDGET: usize = 8192;

/// A queued request.
struct Waiting {
    job: Job,
    queued: Instant,
}

/// The model, its slots and the queue; owns the worker thread's state.
pub(crate) struct Batcher {
    core: Core,
    model: Model,
    slots: Vec<Slot>,
    queue: VecDeque<Waiting>,
    report: Arc<Mutex<SlotsReport>>,
    clock: u64,
    /// The next prefilling slot to serve (round robin).
    next_prefill: usize,
    /// The token baseline's batched step graphs, per slot set.
    #[cfg(feature = "cuda")]
    tok_graphs: ncm_model::tok::GraphedTokSlots,
    /// The NCM's batched concept-step graphs, per slot set.
    #[cfg(feature = "cuda")]
    ncm_graphs: ncm_model::ncm::GraphedNcmSlots,
    /// The hybrid's batched block and fill passes' graphs, per slot set and mode.
    hybrid_graphs: PassGraphs,
}

#[allow(clippy::large_enum_variant)]
enum Model {
    Tok(TokenLm),
    Ncm(NcmLm),
    Hybrid(HybridLm),
}

impl Batcher {
    /// `count` slots over `core`'s model: the core's own caches become slot 0, the rest are
    /// allocated for the same context.
    pub(crate) fn new(mut core: Core, count: usize, report: Arc<Mutex<SlotsReport>>) -> Self {
        // The hybrid: one fast decode session per slot (the core's is the first).
        if let Some((model, first)) = core.take_hybrid() {
            let device = core.device.clone();
            let capacity = core.capacity;
            // One pool of device buffers (rows per slot) and pooled backbone caches.
            let sessions = if count <= 1 {
                vec![first]
            } else {
                drop(first);
                FastSession::pooled(&model, count, capacity, &device)
            };
            let slots = sessions
                .into_iter()
                .map(|session| Slot {
                    caches: Caches::Hybrid(Box::new(session)),
                    kept: Vec::new(),
                    last_used: 0,
                    active: None,
                })
                .collect();
            return Self {
                core,
                model: Model::Hybrid(model),
                slots,
                queue: VecDeque::new(),
                report,
                clock: 0,
                next_prefill: 0,
                #[cfg(feature = "cuda")]
                tok_graphs: Default::default(),
                #[cfg(feature = "cuda")]
                ncm_graphs: Default::default(),
                hybrid_graphs: PassGraphs::default(),
            };
        }
        let state = std::mem::replace(
            &mut *core.state.lock().unwrap_or_else(|p| p.into_inner()),
            State::Empty,
        );
        let device = core.device.clone();
        let capacity = core.capacity;
        let (model, first) = match state {
            State::Tok { model, caches, .. } => (Model::Tok(model), Caches::Tok(caches)),
            State::Ncm { model, cache, .. } => (Model::Ncm(model), Caches::Ncm(cache)),
            State::Empty => unreachable!("a built core has a model"),
        };
        let hybrid_graphs = PassGraphs::default();
        // With several slots, all of them get caches allocated together (on CUDA the retrieval
        // layers' rows of shared buffers, so a batched step decodes them in one set of
        // launches); one slot keeps the core's.
        let slot = |caches| Slot { caches, kept: Vec::new(), last_used: 0, active: None };
        let slots: Vec<Slot> = if count <= 1 {
            vec![slot(first)]
        } else {
            drop(first);
            match &model {
                Model::Tok(m) => m
                    .make_slot_caches(count, capacity, &device)
                    .into_iter()
                    .map(|c| slot(Caches::Tok(c)))
                    .collect(),
                Model::Ncm(m) => m
                    .make_slot_caches(count, capacity, &device)
                    .into_iter()
                    .map(|c| slot(Caches::Ncm(c)))
                    .collect(),
                Model::Hybrid(_) => unreachable!("the hybrid's slots are built above"),
            }
        };
        Self {
            core,
            model,
            slots,
            queue: VecDeque::new(),
            report,
            clock: 0,
            next_prefill: 0,
            #[cfg(feature = "cuda")]
            tok_graphs: Default::default(),
            #[cfg(feature = "cuda")]
            ncm_graphs: Default::default(),
            hybrid_graphs,
        }
    }

    /// Serves `jobs` until the engine is dropped, or a panic (recorded in `dead`; every request
    /// in flight or queued then fails).
    pub(crate) fn serve(mut self, jobs: mpsc::Receiver<Job>, dead: Arc<OnceLock<String>>) {
        loop {
            let idle = self.queue.is_empty() && self.slots.iter().all(|s| s.active.is_none());
            if idle {
                match jobs.recv() {
                    Ok(job) => self.queue.push_back(Waiting { job, queued: Instant::now() }),
                    Err(_) => return,
                }
            }
            while let Ok(job) = jobs.try_recv() {
                self.queue.push_back(Waiting { job, queued: Instant::now() });
            }
            let step = catch_unwind(AssertUnwindSafe(|| self.iterate()));
            if let Err(panic) = step {
                let message = panic_message(panic.as_ref());
                dead.set(message.clone()).ok();
                let error = || EngineError::Device(format!("generation panicked: {message}"));
                for slot in &mut self.slots {
                    if let Some(active) = slot.active.take() {
                        active.events.send(Event::Done(Err(error()))).ok();
                    }
                }
                for waiting in self.queue.drain(..) {
                    waiting.job.events.send(Event::Done(Err(error()))).ok();
                }
                return;
            }
        }
    }

    /// One scheduling iteration (module docs).
    fn iterate(&mut self) {
        self.drop_cancelled();
        self.start_queued();
        self.prefill_step();
        self.decode_step();
        self.publish();
    }

    fn ids(&self) -> Specials {
        self.core.vocab.ids
    }

    fn device(&self) -> &Device {
        &self.core.device
    }

    fn int_tensor(&self, ids: &[i32], rows: usize) -> Tensor<2, Int> {
        let cols = ids.len() / rows.max(1);
        Tensor::from_data(TensorData::new(ids.to_vec(), [rows, cols]), self.device())
    }

    // ------------------------------------------------------------------ queue and slots

    /// Requests whose listener left: queued ones are answered at once, active ones end.
    fn drop_cancelled(&mut self) {
        let mut kept = VecDeque::new();
        for waiting in self.queue.drain(..) {
            if waiting.job.cancel.load(Ordering::Relaxed) {
                let stats = crate::engine::aborted(waiting.job.request.prompt.len());
                waiting.job.events.send(Event::Done(Ok(stats))).ok();
            } else {
                kept.push_back(waiting);
            }
        }
        self.queue = kept;
        for i in 0..self.slots.len() {
            let cancelled =
                self.slots[i].active.as_ref().is_some_and(|a| a.cancel.load(Ordering::Relaxed));
            if cancelled {
                self.finish(i, Finish::Stop);
            }
        }
    }

    /// The stream a prompt needs in the caches, its pending tokens and its padding.
    fn stream_of(&self, prompt: &[u32]) -> (Vec<i32>, Vec<i32>, usize) {
        let ids: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        match &self.model {
            Model::Tok(_) | Model::Hybrid(_) => (ids, Vec::new(), 0),
            Model::Ncm(m) => {
                let k = m.concept_size();
                let pad = k.saturating_sub(ids.len());
                let mut stream = vec![self.ids().pad as i32; pad];
                stream.extend(ids);
                let whole = stream.len() - stream.len() % k;
                let pending = stream.split_off(whole);
                (stream, pending, pad)
            }
        }
    }

    /// Starts queued requests on free slots, in order; a request whose prompt a slot is
    /// prefilling waits to be forked from it.
    fn start_queued(&mut self) {
        let mut waiting = VecDeque::new();
        while let Some(w) = self.queue.pop_front() {
            let (stream, pending, _) = self.stream_of(&w.job.request.prompt);
            let leader = self.slots.iter().any(|s| {
                s.active.as_ref().is_some_and(|a| {
                    a.phase == Phase::Prefill && a.stream == stream && a.pending == pending
                })
            });
            if leader {
                waiting.push_back(w);
                continue;
            }
            match self.pick_slot(&stream) {
                Some(i) => self.start(i, w),
                None => waiting.push_back(w),
            }
        }
        self.queue = waiting;
    }

    /// The free slot for `stream`: the one whose kept prefix covers the most of it, else the
    /// least recently used.
    fn pick_slot(&self, stream: &[i32]) -> Option<usize> {
        let reuse = self.core.prefix_reuse;
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.active.is_none())
            .map(|(i, s)| {
                let hit =
                    if reuse { kept_match(&s.kept, stream).map_or(0, |(_, n)| n + 1) } else { 0 };
                (i, hit, s.last_used)
            })
            .max_by(|a, b| a.1.cmp(&b.1).then(b.2.cmp(&a.2)))
            .map(|(i, _, _)| i)
    }

    /// Puts request `w` on slot `i`: restores its best kept prefix (or resets the caches) and
    /// leaves the rest of the prompt to prefill.
    fn start(&mut self, i: usize, w: Waiting) {
        let Waiting { job, queued } = w;
        let Job { request, events, cancel } = job;
        let (stream, pending, pad) = self.stream_of(&request.prompt);
        let reuse = self.core.prefix_reuse;
        let slot = &mut self.slots[i];
        let hit = if reuse { kept_match(&slot.kept, &stream) } else { None };
        let (prefilled, out) = match hit {
            Some((k, _)) => {
                let out = match (&slot.kept[k], &mut slot.caches) {
                    (SlotKept::Tok(kept), Caches::Tok(c)) => {
                        c.iter_mut().zip(&kept.snapshot).for_each(|(c, s)| c.restore(s));
                        kept.out.clone()
                    }
                    (SlotKept::Ncm(kept), Caches::Ncm(c)) => {
                        c.restore(&kept.snapshot);
                        kept.out.clone()
                    }
                    (SlotKept::Hybrid(kept), Caches::Hybrid(c)) => {
                        c.restore(kept);
                        None
                    }
                    _ => unreachable!("a slot keeps states of its own caches"),
                };
                let n = slot.kept[k].tokens().len();
                (n, if n == stream.len() { out } else { None })
            }
            None => {
                match (&mut slot.caches, &self.model) {
                    (Caches::Tok(c), _) => c.iter_mut().for_each(LayerCache::reset),
                    (Caches::Ncm(c), _) => c.reset(),
                    (Caches::Hybrid(c), Model::Hybrid(m)) => c.reset(m),
                    _ => unreachable!("slots hold caches of the model"),
                }
                (0, None)
            }
        };
        // The hybrid decodes from its session: a whole hit needs no stored output.
        let is_hybrid = matches!(self.model, Model::Hybrid(_));
        let whole = prefilled >= stream.len();
        let phase =
            if out.is_some() || (is_hybrid && whole) { Phase::Decode } else { Phase::Prefill };
        let hybrid = is_hybrid.then(|| HybridRequest {
            opts: crate::engine::hybrid_options(
                request.sampling,
                request.constraint.is_some(),
                self.core.vocab.ids,
            ),
            run: None,
            acceptance: HybridAcceptance::default(),
        });
        if let (Some(h), Caches::Hybrid(c)) = (&hybrid, &mut slot.caches) {
            c.begin_request(h.opts.seed);
        }
        #[cfg(feature = "cuda")]
        let device = if request.sampling.is_greedy() {
            ncm_model::sample::SampleSettings::GREEDY
        } else {
            crate::engine::device_sampling(request.sampling)
        };
        let cached = prefilled.saturating_sub(pad);
        let history = stream.clone();
        slot.active = Some(Active {
            events,
            cancel,
            sampler: Sampler::new(request.sampling),
            constraint: request.constraint,
            max_tokens: request.max_tokens.max(1),
            emitted: 0,
            prompt_tokens: request.prompt.len(),
            stream,
            prefilled,
            pending,
            pad,
            phase,
            out,
            history,
            prompt_kept: None,
            cached,
            queued_s: queued.elapsed().as_secs_f64(),
            started: Instant::now(),
            prompt_s: 0.0,
            decode_started: None,
            steps: 0,
            rows: 0,
            empty_concepts: 0,
            forked: false,
            next: None,
            unappended: None,
            #[cfg(feature = "cuda")]
            device,
            hybrid,
        });
        if phase == Phase::Decode {
            self.prompt_done(i);
        }
    }

    // ------------------------------------------------------------------ prefill and forks

    /// One prefill chunk of every prefilling slot, round robin from the next one, up to
    /// [`PREFILL_BUDGET`] tokens in all: chunks of different lengths (each up to the chunk size
    /// and to what its prompt has left) packed into one varlen forward.
    fn prefill_step(&mut self) {
        let n = self.slots.len();
        let prefilling = |s: &Slot| s.active.as_ref().is_some_and(|a| a.phase == Phase::Prefill);
        let k = match &self.model {
            Model::Tok(_) | Model::Hybrid(_) => 1,
            Model::Ncm(m) => m.concept_size(),
        };
        // The NCM's and the hybrid's encoders take at most ENCODER_CHUNK tokens per segment
        // (their sliding windows attend over the window plus the whole piece).
        let chunk = match &self.model {
            Model::Tok(_) => self.core.chunk,
            _ => self.core.chunk.min(ncm_model::ncm::ENCODER_CHUNK),
        };
        // The NCM's chunks are whole concepts (its streams are too).
        let chunk = (chunk - chunk % k).max(k);
        let mut rows = Vec::new();
        let mut lens = Vec::new();
        let mut budget = PREFILL_BUDGET.max(chunk);
        let mut last = None;
        for d in 0..n {
            let i = (self.next_prefill + d) % n;
            if !prefilling(&self.slots[i]) {
                continue;
            }
            let a = self.slots[i].active.as_ref().expect("prefilling");
            let left = a.stream.len() - a.prefilled;
            let len = chunk.min(left).min(budget - budget % k);
            if len == 0 {
                break;
            }
            rows.push(i);
            lens.push(len);
            last = Some(i);
            budget -= len;
        }
        let Some(last) = last else { return };
        self.next_prefill = (last + 1) % n;
        let device = self.device().clone();
        if rows.len() == 1 {
            self.prefill_single(rows[0], lens[0], &device);
        } else {
            self.prefill_packed(&rows, &lens, &device);
        }
        for &i in &rows {
            let active = self.slots[i].active.as_ref().expect("prefilling");
            if active.prefilled >= active.stream.len() {
                self.slots[i].active.as_mut().expect("prefilling").phase = Phase::Decode;
                self.prompt_done(i);
                self.fork_from(i);
            }
        }
    }

    /// The next `len` prompt tokens of slot `i` alone; the output once the prompt is in.
    fn prefill_single(&mut self, i: usize, len: usize, device: &Device) {
        let slot = &mut self.slots[i];
        let active = slot.active.as_mut().expect("prefilling");
        let start = active.prefilled;
        let piece = &active.stream[start..start + len];
        let t = Tensor::from_data(TensorData::new(piece.to_vec(), [1, len]), device);
        let out = match (&self.model, &mut slot.caches) {
            (Model::Tok(m), Caches::Tok(c)) => m.prefill(t, c, len),
            (Model::Ncm(m), Caches::Ncm(c)) => Some(m.prefill(t, c, len)),
            (Model::Hybrid(m), Caches::Hybrid(c)) => {
                c.prefill(m, piece);
                None
            }
            _ => unreachable!("slots hold caches of the model"),
        };
        active.prefilled += len;
        if active.prefilled >= active.stream.len() {
            active.out = out;
        }
    }

    /// The next `lens[r]` prompt tokens of each slot `rows[r]`, packed into one varlen forward.
    fn prefill_packed(&mut self, rows: &[usize], lens: &[usize], device: &Device) {
        let total: usize = lens.iter().sum();
        let mut ids = Vec::with_capacity(total);
        for (&i, &len) in rows.iter().zip(lens) {
            let a = self.slots[i].active.as_ref().expect("prefilling");
            ids.extend_from_slice(&a.stream[a.prefilled..a.prefilled + len]);
        }
        // The slots of `rows` in that order: their caches and requests.
        let mut picked: Vec<Option<&mut Slot>> = self.slots.iter_mut().map(Some).collect();
        let mut slots: Vec<&mut Slot> =
            rows.iter().map(|&i| picked[i].take().expect("one row per slot")).collect();
        let out = match &self.model {
            Model::Tok(m) => {
                let t = Tensor::from_data(TensorData::new(ids, [1, total]), device);
                let mut caches: Vec<&mut [LayerCache]> = slots
                    .iter_mut()
                    .map(|s| match &mut s.caches {
                        Caches::Tok(c) => c.as_mut_slice(),
                        _ => unreachable!("slots hold caches of the model"),
                    })
                    .collect();
                Some(m.prefill_packed(t, lens, &mut caches))
            }
            Model::Ncm(m) => {
                let t = Tensor::from_data(TensorData::new(ids, [1, total]), device);
                let mut caches: Vec<&mut NcmCache> = slots
                    .iter_mut()
                    .map(|s| match &mut s.caches {
                        Caches::Ncm(c) => c,
                        _ => unreachable!("slots hold caches of the model"),
                    })
                    .collect();
                Some(m.append_packed(t, lens, &mut caches))
            }
            Model::Hybrid(m) => {
                let mut at = 0;
                let pieces: Vec<&[i32]> = lens
                    .iter()
                    .map(|&len| {
                        let piece = &ids[at..at + len];
                        at += len;
                        piece
                    })
                    .collect();
                let mut sessions: Vec<&mut HybridSession> = slots
                    .iter_mut()
                    .map(|s| match &mut s.caches {
                        Caches::Hybrid(c) => &mut c.session,
                        _ => unreachable!("slots hold caches of the model"),
                    })
                    .collect();
                HybridSession::prefill_packed(m, &mut sessions, &pieces);
                None
            }
        };
        for (r, (slot, &len)) in slots.into_iter().zip(lens).enumerate() {
            let active = slot.active.as_mut().expect("prefilling");
            active.prefilled += len;
            if active.prefilled >= active.stream.len() {
                active.out = out.as_ref().map(|o| o.clone().narrow(0, r, 1));
            }
        }
    }

    /// The prompt of slot `i` is in: keep its state (prefix reuse) and start the clock.
    fn prompt_done(&mut self, i: usize) {
        self.core.sync();
        let reuse = self.core.prefix_reuse;
        let slot = &mut self.slots[i];
        let active = slot.active.as_mut().expect("active");
        active.prompt_s = active.started.elapsed().as_secs_f64();
        active.decode_started = Some(Instant::now());
        if reuse {
            let out = active.out.clone();
            let tokens = active.stream.clone();
            active.prompt_kept = Some(match &slot.caches {
                Caches::Tok(c) => SlotKept::Tok(Kept {
                    tokens,
                    snapshot: c.iter().map(LayerCache::snapshot).collect(),
                    out,
                }),
                Caches::Ncm(c) => SlotKept::Ncm(Kept { tokens, snapshot: c.snapshot(), out }),
                Caches::Hybrid(c) => SlotKept::Hybrid(Box::new(c.snapshot())),
            });
        }
    }

    /// Forks every queued request whose prompt slot `leader` just prefilled onto free slots:
    /// copies of the leader's caches, decoding from the same output.
    fn fork_from(&mut self, leader: usize) {
        let (stream, pending, out) = {
            let a = self.slots[leader].active.as_ref().expect("the leader");
            (a.stream.clone(), a.pending.clone(), a.out.clone())
        };
        let mut rest = VecDeque::new();
        while let Some(w) = self.queue.pop_front() {
            let (s, p, _) = self.stream_of(&w.job.request.prompt);
            let free = (0..self.slots.len())
                .filter(|&j| j != leader && self.slots[j].active.is_none())
                .min_by_key(|&j| self.slots[j].last_used);
            match free {
                Some(j) if s == stream && p == pending => {
                    // Its kept prefixes describe the old contents, which the copy replaces.
                    self.slots[j].kept.clear();
                    self.start(j, w);
                    let (a, b) = if j < leader {
                        let (lo, hi) = self.slots.split_at_mut(leader);
                        (&hi[0], &mut lo[j])
                    } else {
                        let (lo, hi) = self.slots.split_at_mut(j);
                        (&lo[leader], &mut hi[0])
                    };
                    match (&a.caches, &mut b.caches) {
                        (Caches::Tok(src), Caches::Tok(dst)) => {
                            dst.iter_mut().zip(src).for_each(|(d, s)| d.copy_from(s));
                        }
                        (Caches::Ncm(src), Caches::Ncm(dst)) => dst.copy_from(src),
                        (Caches::Hybrid(src), Caches::Hybrid(dst)) => dst.copy_from(src),
                        _ => unreachable!("slots hold caches of one model"),
                    }
                    let active = self.slots[j].active.as_mut().expect("just started");
                    active.prefilled = active.stream.len();
                    active.cached = active.stream.len().saturating_sub(active.pad);
                    active.out = out.clone();
                    active.phase = Phase::Decode;
                    active.forked = true;
                    self.prompt_done(j);
                    self.report.lock().unwrap_or_else(|p| p.into_inner()).forks += 1;
                }
                _ => rest.push_back(w),
            }
        }
        self.queue = rest;
    }

    // ------------------------------------------------------------------ decoding

    fn decode_step(&mut self) {
        let rows: Vec<usize> = (0..self.slots.len())
            .filter(|&i| self.slots[i].active.as_ref().is_some_and(|a| a.phase == Phase::Decode))
            .collect();
        if rows.is_empty() {
            return;
        }
        {
            let mut report = self.report.lock().unwrap_or_else(|p| p.into_inner());
            report.decode_steps += 1;
            report.decode_rows += rows.len() as u64;
        }
        for &i in &rows {
            let a = self.slots[i].active.as_mut().expect("decoding");
            a.steps += 1;
            a.rows += rows.len();
        }
        match &self.model {
            Model::Tok(_) => self.tok_step(&rows),
            Model::Ncm(_) => self.ncm_step(&rows),
            Model::Hybrid(_) => self.hybrid_step(&rows),
        }
    }

    /// The hybrid at block granularity: every decoding slot opens its block if it has none,
    /// then one batched block pass per pass mode over the slots due one (a first pass, a
    /// re-decode or a recheck), and one batched fill pass over the slots on their
    /// autoregressive fill; slots whose block is done emit its tokens.
    fn hybrid_step(&mut self, rows: &[usize]) {
        let capacity = self.core.capacity;
        let ids = self.ids();
        let Model::Hybrid(model) = &self.model else { unreachable!("the hybrid") };
        // Open the blocks.
        let mut done = Vec::new();
        for &i in rows {
            let Slot { caches: Caches::Hybrid(session), active: Some(a), .. } = &mut self.slots[i]
            else {
                unreachable!("a decoding hybrid slot")
            };
            let h = a.hybrid.as_mut().expect("a hybrid request");
            if h.run.is_none() {
                let room = capacity.saturating_sub(session.tokens().len());
                let limit = a.max_tokens.saturating_sub(a.emitted).min(room);
                if limit == 0 {
                    done.push((i, Finish::Length));
                    continue;
                }
                h.run = Some(session.begin_block(model, limit, &h.opts));
            }
        }
        // Block passes, one batch per mode.
        let mut due: Vec<(usize, PassMode, DrawSettings)> = Vec::new();
        for &i in rows {
            let Slot { caches: Caches::Hybrid(session), active: Some(a), .. } = &mut self.slots[i]
            else {
                continue;
            };
            let Some(h) = a.hybrid.as_mut() else { continue };
            let Some(run) = &h.run else { continue };
            if run.phase() == BlockPhase::Pass {
                let (mode, settings) = session.prepare_pass(model, run, &h.opts);
                due.push((i, mode, settings));
            }
        }
        let mut modes: Vec<PassMode> = due.iter().map(|d| d.1).collect();
        modes.dedup();
        modes.sort_by_key(|m| format!("{m:?}"));
        modes.dedup();
        for mode in modes {
            let group: Vec<&(usize, PassMode, DrawSettings)> =
                due.iter().filter(|d| d.1 == mode).collect();
            let key: Vec<usize> = group.iter().map(|d| d.0).collect();
            let settings: Vec<DrawSettings> = group.iter().map(|d| d.2.clone()).collect();
            let results = {
                let sessions = hybrid_sessions(&self.slots, &key);
                block_pass_slots(model, &sessions, mode, &settings, &mut self.hybrid_graphs)
            };
            for (&i, result) in key.iter().zip(results) {
                let Slot { caches: Caches::Hybrid(session), active: Some(a), .. } =
                    &mut self.slots[i]
                else {
                    unreachable!("a decoding hybrid slot")
                };
                let Active { constraint, sampler, hybrid, .. } = a;
                let h = hybrid.as_mut().expect("a hybrid request");
                let run = h.run.as_mut().expect("an open block");
                with_hooks(constraint, sampler, ids, |hooks| {
                    session.apply_pass(model, run, result, &h.opts, hooks)
                });
            }
        }
        // Fill passes, one batch per draw kind.
        for sampled in [false, true] {
            let mut key = Vec::new();
            let mut settings = Vec::new();
            for &i in rows {
                let Slot { caches: Caches::Hybrid(session), active: Some(a), .. } =
                    &mut self.slots[i]
                else {
                    continue;
                };
                let Some(h) = a.hybrid.as_mut() else { continue };
                let Some(run) = &h.run else { continue };
                if run.phase() == BlockPhase::Fill && run.fill_sampled(&h.opts) == sampled {
                    settings.push(session.prepare_fill(model, run, &h.opts));
                    key.push(i);
                }
            }
            if key.is_empty() {
                continue;
            }
            let (tokens, logp) = {
                let sessions = hybrid_sessions(&self.slots, &key);
                ar_pass_slots(model, &sessions, sampled, &settings, &mut self.hybrid_graphs)
            };
            for ((&i, row_tokens), logp) in key.iter().zip(tokens).zip(logp) {
                let Slot { caches: Caches::Hybrid(session), active: Some(a), .. } =
                    &mut self.slots[i]
                else {
                    unreachable!("a decoding hybrid slot")
                };
                let Active { constraint, sampler, hybrid, .. } = a;
                let h = hybrid.as_mut().expect("a hybrid request");
                let run = h.run.as_mut().expect("an open block");
                let p = run.next_position();
                let mut row = || -> Vec<f32> {
                    logp.clone()
                        .narrow(1, p, 1)
                        .into_data()
                        .convert::<f32>()
                        .try_to_vec::<f32>()
                        .expect("f32")
                };
                with_hooks(constraint, sampler, ids, |hooks| {
                    session.apply_fill(run, row_tokens[p], &mut row, &h.opts, hooks)
                });
            }
        }
        // Blocks done: close them together (one batched backbone step) and emit their tokens.
        let closing: Vec<usize> =
            rows.iter()
                .copied()
                .filter(|&i| !done.iter().any(|d| d.0 == i))
                .filter(|&i| {
                    self.slots[i].active.as_ref().and_then(|a| a.hybrid.as_ref()).is_some_and(|h| {
                        h.run.as_ref().is_some_and(|r| r.phase() == BlockPhase::Done)
                    })
                })
                .collect();
        if !closing.is_empty() {
            let runs: Vec<BlockRun> = closing
                .iter()
                .map(|&i| {
                    let a = self.slots[i].active.as_mut().expect("decoding");
                    a.hybrid.as_mut().expect("hybrid").run.take().expect("a done block")
                })
                .collect();
            let blocks = {
                let mut picked: Vec<Option<&mut Slot>> = self.slots.iter_mut().map(Some).collect();
                let mut sessions: Vec<&mut FastSession> = closing
                    .iter()
                    .map(|&i| match &mut picked[i].take().expect("one row per slot").caches {
                        Caches::Hybrid(c) => &mut **c,
                        _ => unreachable!("slots hold caches of the hybrid"),
                    })
                    .collect();
                FastSession::end_blocks(model, &mut sessions, runs)
            };
            for (&i, block) in closing.iter().zip(blocks) {
                let a = self.slots[i].active.as_mut().expect("decoding");
                a.hybrid.as_mut().expect("hybrid").acceptance.add(&block.stats);
                let mut ended = None;
                for &t in &block.tokens {
                    if let Some(f) = emit(a, t as u32, ids) {
                        ended = Some(f);
                        break;
                    }
                }
                if ended.is_none() && (block.stopped || block.tokens.is_empty()) {
                    ended = Some(Finish::Length);
                }
                if let Some(f) = ended {
                    done.push((i, f));
                }
            }
        }
        for (i, f) in done {
            self.finish(i, f);
        }
    }

    /// Token baseline: each slot takes its next token from its logits (host sampler), emits it,
    /// and the slots that go on feed their tokens back in one batched step.
    fn tok_step(&mut self, rows: &[usize]) {
        let ids = self.ids();
        let capacity = self.core.capacity;
        let mut feed = Vec::new();
        for &i in rows {
            let a = self.slots[i].active.as_mut().expect("decoding");
            let token = match a.next.take() {
                Some(token) => token,
                None => {
                    let logits = read_logits(a.out.as_ref().expect("logits after the prompt"));
                    a.sampler.pick(&logits, a.constraint.as_deref()).unwrap_or(ids.im_end)
                }
            };
            if let Some(c) = &mut a.constraint {
                c.advance(token);
            }
            match emit(a, token, ids) {
                Some(finish) => self.finish(i, finish),
                None if a.history.len() >= capacity => self.finish(i, Finish::Length),
                None => feed.push((i, token as i32)),
            }
        }
        #[cfg(feature = "cuda")]
        let feed = self.tok_graph_step(feed);
        if feed.is_empty() {
            return;
        }
        let tokens: Vec<i32> = feed.iter().map(|f| f.1).collect();
        let input = self.int_tensor(&tokens, tokens.len());
        let Model::Tok(model) = &self.model else { unreachable!("the token baseline") };
        let logits = {
            let mut caches = slot_caches_tok(&mut self.slots, &feed);
            model.step_slots(input, &mut caches)
        };
        for (row, &(i, token)) in feed.iter().enumerate() {
            let a = self.slots[i].active.as_mut().expect("decoding");
            a.history.push(token);
            a.out = Some(logits.clone().narrow(0, row, 1));
        }
    }

    /// Feeds the unconstrained rows of `feed` through their slot set's graph (CUDA, graphs on),
    /// which also draws their next tokens; returns the rows left to step eagerly.
    #[cfg(feature = "cuda")]
    fn tok_graph_step(&mut self, feed: Vec<(usize, i32)>) -> Vec<(usize, i32)> {
        if !ncm_model::fused::graphs_active(self.device()) {
            return feed;
        }
        let (graphed, eager): (Vec<_>, Vec<_>) = feed.into_iter().partition(|&(i, _)| {
            self.slots[i].active.as_ref().is_some_and(|a| a.constraint.is_none())
        });
        if graphed.is_empty() {
            return eager;
        }
        let key: Vec<usize> = graphed.iter().map(|f| f.0).collect();
        let tokens: Vec<i32> = graphed.iter().map(|f| f.1).collect();
        let input = self.int_tensor(&tokens, tokens.len());
        let settings: Vec<_> =
            key.iter().map(|&i| self.slots[i].active.as_ref().expect("decoding").device).collect();
        let Model::Tok(model) = &self.model else { unreachable!("the token baseline") };
        let out = {
            let mut caches = slot_caches_tok(&mut self.slots, &graphed);
            self.tok_graphs.step(model, &key, input, &mut caches, &settings)
        };
        let Some(out) = out else {
            // No graph for this set: everything steps eagerly.
            return graphed.into_iter().chain(eager).collect();
        };
        let next = read_ids(&out);
        for (row, &(i, token)) in graphed.iter().enumerate() {
            let a = self.slots[i].active.as_mut().expect("decoding");
            a.history.push(token);
            a.next = Some(next[row]);
            a.out = None;
            a.device.counter = a.device.counter.wrapping_add(1);
        }
        eager
    }

    /// NCM: one concept per slot. The concept each slot wrote last step is appended first
    /// (deferred, so that on CUDA the append and the next decoder pass run as one graph replay
    /// of the slot set, [`ncm_model::ncm::GraphedNcmSlots`]); then the next concept is written
    /// from each slot's condition (pending prompt tokens forced first).
    fn ncm_step(&mut self, rows: &[usize]) {
        if self.core.speculate && self.ncm_speculate_step(rows) {
            return;
        }
        #[cfg(feature = "cuda")]
        if let Some(picked) = self.ncm_graph_step(rows) {
            self.ncm_emit(rows, &picked, &vec![0; rows.len()]);
            return;
        }
        self.ncm_append_deferred(rows);
        let ids = self.ids();
        let device = self.device().clone();
        let Model::Ncm(model) = &self.model else { unreachable!("the NCM") };
        let cond = Tensor::cat(
            rows.iter()
                .map(|&i| {
                    self.slots[i].active.as_ref().expect("decoding").out.clone().expect("cond")
                })
                .collect(),
            0,
        );
        let b = rows.len();
        // Per row: forced tokens, host choice (sampled or constrained), or the device argmax.
        let forced: Vec<Vec<i32>> = rows
            .iter()
            .map(|&i| std::mem::take(&mut self.slots[i].active.as_mut().expect("decoding").pending))
            .collect();
        let host: Vec<bool> = rows
            .iter()
            .map(|&i| {
                let a = self.slots[i].active.as_ref().expect("decoding");
                !a.sampler.is_greedy() || a.constraint.is_some()
            })
            .collect();
        let mut ended = vec![false; b];
        let picked = {
            let (mut caches, mut actives) = slot_caches_ncm(&mut self.slots, rows);
            let mut pick = |j: usize, logits: &Tensor<3>| -> Tensor<2, Int> {
                let device_only = (0..b).all(|r| !host[r] && forced[r].get(j).is_none());
                if device_only {
                    return logits.clone().argmax(2).reshape([b, 1]);
                }
                let v = logits.dims()[2];
                let all = read_logits(logits);
                let tokens: Vec<i32> = (0..b)
                    .map(|r| {
                        if let Some(&t) = forced[r].get(j) {
                            return t;
                        }
                        let row = &all[r * v..(r + 1) * v];
                        if !host[r] {
                            return argmax(row).unwrap_or(ids.im_end) as i32;
                        }
                        if ended[r] {
                            return ids.pad as i32;
                        }
                        let a = &mut *actives[r];
                        let token =
                            a.sampler.pick(row, a.constraint.as_deref()).unwrap_or(ids.im_end);
                        if token == ids.eoc || ids.is_stop(token) {
                            ended[r] = true;
                        } else if let Some(c) = &mut a.constraint {
                            c.advance(token);
                        }
                        token as i32
                    })
                    .collect();
                Tensor::from_data(TensorData::new(tokens, [b, 1]), &device)
            };
            model.decode_slots(cond, &mut caches, &mut pick)
        };
        let picked = read_ids(&picked);
        let skips: Vec<usize> = forced.iter().map(Vec::len).collect();
        self.ncm_emit(rows, &picked, &skips);
    }

    /// `--speculate`: when every slot of `rows` decodes greedily, unconstrained and with no
    /// pending prompt tokens, and the model has a draft head, writes their next concepts by
    /// batched self-speculation ([`ncm_model::speculate_slots`]: one draft pass over all the
    /// slots, then verify passes over the slots still open, the decoder's cross-attention per
    /// slot and everything else batched). Returns false (nothing done) otherwise.
    fn ncm_speculate_step(&mut self, rows: &[usize]) -> bool {
        let Model::Ncm(model) = &self.model else { return false };
        if !model.has_draft_head() {
            return false;
        }
        let plain = rows.iter().all(|&i| {
            let a = self.slots[i].active.as_ref().expect("decoding");
            a.sampler.is_greedy() && a.constraint.is_none() && a.pending.is_empty()
        });
        if !plain {
            return false;
        }
        self.ncm_append_deferred(rows);
        let Model::Ncm(model) = &self.model else { unreachable!("the NCM") };
        let cond = Tensor::cat(
            rows.iter()
                .map(|&i| {
                    self.slots[i].active.as_ref().expect("decoding").out.clone().expect("cond")
                })
                .collect(),
            0,
        );
        let mut stats = ncm_model::SpecStats::default();
        let picked = {
            let (caches, _) = slot_caches_ncm(&mut self.slots, rows);
            let caches: Vec<&NcmCache> = caches.into_iter().map(|c| &*c).collect();
            ncm_model::speculate_slots(model, cond, &caches, &mut stats)
        };
        let picked: Vec<u32> = picked.into_iter().map(|t| t as u32).collect();
        self.ncm_emit(rows, &picked, &vec![0; rows.len()]);
        true
    }

    /// Appends the concepts the slots of `rows` wrote last step (those that have one) in one
    /// batched step; their conditions become the slots' `out`.
    fn ncm_append_deferred(&mut self, rows: &[usize]) {
        let appends: Vec<(usize, Vec<i32>)> = rows
            .iter()
            .filter_map(|&i| {
                let a = self.slots[i].active.as_mut().expect("decoding");
                a.unappended.take().map(|c| (i, c))
            })
            .collect();
        if appends.is_empty() {
            return;
        }
        let flat: Vec<i32> = appends.iter().flat_map(|a| a.1.iter().copied()).collect();
        let input = self.int_tensor(&flat, appends.len());
        let slots: Vec<usize> = appends.iter().map(|a| a.0).collect();
        let Model::Ncm(model) = &self.model else { unreachable!("the NCM") };
        let cond = {
            let (mut caches, _) = slot_caches_ncm(&mut self.slots, &slots);
            model.append_slots(input, &mut caches)
        };
        for (row, (i, concept)) in appends.into_iter().enumerate() {
            let a = self.slots[i].active.as_mut().expect("decoding");
            a.history.extend_from_slice(&concept);
            a.out = Some(cond.clone().narrow(0, row, 1));
        }
    }

    /// A slot set's step as one graph replay (CUDA, graphs on): when every row has a written
    /// concept to append, nothing pending or constrained (the device sampler draws every row,
    /// greedy rows by argmax). Returns the next concepts `[rows * k]`, or None to step eagerly.
    #[cfg(feature = "cuda")]
    fn ncm_graph_step(&mut self, rows: &[usize]) -> Option<Vec<u32>> {
        if !ncm_model::fused::graphs_active(self.device()) {
            return None;
        }
        // Greedy rows only: a sampled request's draws must come from one generator for the whole
        // reply, and its first concept (forced prompt tokens) and any step shared with a row
        // the graph cannot take run eagerly with the host sampler, so a sampled row on the
        // graph's device sampler would draw from two streams in a batch-dependent mix.
        let eligible = rows.iter().all(|&i| {
            let a = self.slots[i].active.as_ref().expect("decoding");
            a.unappended.is_some()
                && a.pending.is_empty()
                && a.constraint.is_none()
                && a.sampler.is_greedy()
        });
        if !eligible {
            return None;
        }
        let flat: Vec<i32> = rows
            .iter()
            .flat_map(|&i| {
                let a = self.slots[i].active.as_ref().expect("decoding");
                a.unappended.clone().expect("checked")
            })
            .collect();
        let input = self.int_tensor(&flat, rows.len());
        let settings: Vec<_> =
            rows.iter().map(|&i| self.slots[i].active.as_ref().expect("decoding").device).collect();
        let Model::Ncm(model) = &self.model else { unreachable!("the NCM") };
        let k = model.concept_size();
        let (next, cond) = {
            let (mut caches, _) = slot_caches_ncm(&mut self.slots, rows);
            self.ncm_graphs.step(model, rows, input, &mut caches, &settings)?
        };
        let picked = read_ids(&next);
        for (row, &i) in rows.iter().enumerate() {
            let a = self.slots[i].active.as_mut().expect("decoding");
            let concept = a.unappended.take().expect("checked");
            a.history.extend_from_slice(&concept);
            // A view of the graph's output buffer: copied if the request ends (`finish`).
            a.out = Some(cond.clone().narrow(0, row, 1));
            a.device.counter = a.device.counter.wrapping_add(k as u32);
        }
        Some(picked)
    }

    /// Streams the concepts `picked` (`[rows * k]`, the first `skips[r]` of row r forced prompt
    /// tokens) and keeps each continuing slot's concept to append next step.
    fn ncm_emit(&mut self, rows: &[usize], picked: &[u32], skips: &[usize]) {
        let ids = self.ids();
        let capacity = self.core.capacity;
        let Model::Ncm(model) = &self.model else { unreachable!("the NCM") };
        let k = model.concept_size();
        for (r, &i) in rows.iter().enumerate() {
            let skip = skips[r];
            let concept_ids = &picked[r * k..(r + 1) * k];
            let a = self.slots[i].active.as_mut().expect("decoding");
            let before = a.emitted;
            match emit_concept(a, concept_ids, skip, k, ids) {
                Err(finish) => self.finish(i, finish),
                Ok(concept) => {
                    a.empty_concepts = if a.emitted == before { a.empty_concepts + 1 } else { 0 };
                    if a.empty_concepts >= MAX_EMPTY_CONCEPTS {
                        self.finish(i, Finish::Stop);
                    } else if a.history.len() + k > capacity {
                        self.finish(i, Finish::Length);
                    } else {
                        a.unappended = Some(concept);
                    }
                }
            }
        }
    }

    /// Ends the request on slot `i`: its stats go back, and the slot keeps the prompt's state
    /// and the state at the end of the reply for the next requests.
    fn finish(&mut self, i: usize, finish: Finish) {
        self.clock += 1;
        let reuse = self.core.prefix_reuse;
        let slot = &mut self.slots[i];
        slot.last_used = self.clock;
        let Some(a) = slot.active.take() else { return };
        let mut next = Vec::new();
        if let (Some(_), Caches::Hybrid(c)) = (&a.prompt_kept, &slot.caches) {
            // The hybrid keeps the point after the prompt and the end of the reply (an open
            // block's tokens included).
            if reuse && c.tokens().len() > a.stream.len() {
                next.push(SlotKept::Hybrid(Box::new(c.snapshot())));
            }
        }
        let hybrid_path = a.hybrid.as_ref().map(|h| DecodePath::Hybrid(h.acceptance));
        if let Some(prompt) = a.prompt_kept {
            if reuse && hybrid_path.is_none() && a.history.len() > a.stream.len() {
                let tokens = a.history.clone();
                // A copy: the condition may be a view of a step graph's output buffer.
                let out = a.out.clone().map(|o| o.mul_scalar(1.0));
                next.push(match &slot.caches {
                    Caches::Tok(c) => SlotKept::Tok(Kept {
                        tokens,
                        snapshot: c.iter().map(LayerCache::snapshot).collect(),
                        out,
                    }),
                    Caches::Ncm(c) => SlotKept::Ncm(Kept { tokens, snapshot: c.snapshot(), out }),
                    Caches::Hybrid(_) => unreachable!("the hybrid keeps its reply above"),
                });
            }
            next.insert(0, prompt);
        }
        // A request that ended during its prefill leaves part of a prompt: keep nothing.
        slot.kept = next;
        let decode_s = a.decode_started.map_or(0.0, |t| t.elapsed().as_secs_f64());
        let stats = GenStats {
            prompt_tokens: a.prompt_tokens,
            completion_tokens: a.emitted,
            finish,
            prompt_s: a.prompt_s,
            decode_s,
            decode_tokens: a.emitted,
            cached_tokens: a.cached,
            capture_s: 0.0,
            path: hybrid_path.unwrap_or(DecodePath::Batched),
            queue_s: a.queued_s,
            mean_batch: if a.steps == 0 { 0.0 } else { a.rows as f64 / a.steps as f64 },
            forked: a.forked,
        };
        a.events.send(Event::Done(Ok(stats))).ok();
        self.report.lock().unwrap_or_else(|p| p.into_inner()).served += 1;
    }

    /// Copies the slots' state into the shared report.
    fn publish(&self) {
        let views = self
            .slots
            .iter()
            .enumerate()
            .map(|(id, s)| {
                let tokens = match &s.caches {
                    Caches::Tok(c) => c.first().map_or(0, LayerCache::len),
                    Caches::Ncm(c) => c.tokens,
                    Caches::Hybrid(c) => c.tokens().len(),
                };
                match &s.active {
                    None => SlotView { id, state: "idle", tokens, ..SlotView::default() },
                    Some(a) => SlotView {
                        id,
                        state: if a.phase == Phase::Prefill { "prefill" } else { "decode" },
                        tokens,
                        prompt_tokens: a.prompt_tokens,
                        generated: a.emitted,
                    },
                }
            })
            .collect();
        let mut report = self.report.lock().unwrap_or_else(|p| p.into_inner());
        report.slots = views;
        report.queued = self.queue.len();
    }
}

/// The longest kept prefix `stream` extends (with its output when `stream` is exactly it):
/// (index, its length).
fn kept_match(kept: &[SlotKept], stream: &[i32]) -> Option<(usize, usize)> {
    kept.iter()
        .enumerate()
        .filter(|(_, k)| stream.starts_with(k.tokens()))
        .filter(|(_, k)| stream.len() > k.tokens().len() || k.has_out())
        .max_by_key(|(_, k)| k.tokens().len())
        .map(|(i, k)| (i, k.tokens().len()))
}

/// Hands `token` to the request's listener; Some(why) when the request ends with it.
fn emit(a: &mut Active, token: u32, ids: Specials) -> Option<Finish> {
    a.emitted += 1;
    if a.events.send(Event::Token(token)).is_err() || a.cancel.load(Ordering::Relaxed) {
        return Some(Finish::Stop);
    }
    if ids.is_stop(token) {
        return Some(Finish::Stop);
    }
    (a.emitted >= a.max_tokens).then_some(Finish::Length)
}

/// [`crate::engine`]'s concept emission for a slot: the concept to append (cut after an
/// end-of-concept token and padded to `k`), or why the request ends.
fn emit_concept(
    a: &mut Active,
    picked: &[u32],
    skip: usize,
    k: usize,
    ids: Specials,
) -> Result<Vec<i32>, Finish> {
    let mut concept = Vec::with_capacity(k);
    for (j, &token) in picked.iter().enumerate() {
        concept.push(token as i32);
        if j < skip {
            continue;
        }
        if token == ids.eoc {
            break;
        }
        if let Some(finish) = emit(a, token, ids) {
            return Err(finish);
        }
    }
    concept.resize(k, ids.pad as i32);
    Ok(concept)
}

/// The token-baseline caches of the slots in `feed`, in order.
fn slot_caches_tok<'a>(slots: &'a mut [Slot], feed: &[(usize, i32)]) -> Vec<&'a mut [LayerCache]> {
    let mut picked: Vec<Option<&'a mut [LayerCache]>> = slots
        .iter_mut()
        .map(|s| match &mut s.caches {
            Caches::Tok(c) => Some(c.as_mut_slice()),
            _ => None,
        })
        .collect();
    feed.iter().map(|&(i, _)| picked[i].take().expect("one row per slot")).collect()
}

/// The NCM caches and requests of slots `rows`, in order.
fn slot_caches_ncm<'a>(
    slots: &'a mut [Slot],
    rows: &[usize],
) -> (Vec<&'a mut NcmCache>, Vec<&'a mut Active>) {
    let mut picked: Vec<Option<(&'a mut NcmCache, Option<&'a mut Active>)>> = slots
        .iter_mut()
        .map(|s| match &mut s.caches {
            Caches::Ncm(c) => Some((c, s.active.as_mut())),
            _ => None,
        })
        .collect();
    rows.iter()
        .map(|&i| {
            let (c, a) = picked[i].take().expect("one row per slot");
            (c, a.expect("an active slot"))
        })
        .unzip()
}

/// The hybrid sessions of slots `key`, in order.
fn hybrid_sessions<'a>(slots: &'a [Slot], key: &[usize]) -> Vec<&'a FastSession> {
    key.iter()
        .map(|&i| match &slots[i].caches {
            Caches::Hybrid(c) => &**c,
            _ => unreachable!("slots hold caches of the hybrid"),
        })
        .collect()
}

/// Runs `f` with hooks over a request's grammar constraint (the scan over each check pass's
/// tokens, the pick of every autoregressive token, both advancing the constraint) when it has
/// one; plain hooks otherwise.
fn with_hooks<T>(
    constraint: &mut Option<Box<dyn Constraint>>,
    sampler: &mut Sampler,
    ids: Specials,
    f: impl FnOnce(&mut ncm_model::hybrid::Hooks) -> T,
) -> T {
    use std::cell::RefCell;
    if constraint.is_none() {
        return f(&mut ncm_model::hybrid::Hooks::default());
    }
    let shared = RefCell::new((constraint.take(), sampler));
    let mut scan = |proposed: &[i32]| {
        let mut s = shared.borrow_mut();
        let Some(c) = s.0.as_mut() else { return proposed.len() };
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
        let mut s = shared.borrow_mut();
        let (c, sampler) = &mut *s;
        let token = sampler.pick(row, c.as_deref()).unwrap_or(ids.im_end);
        if let Some(c) = c.as_mut() {
            c.advance(token);
        }
        token as i32
    };
    let out = {
        let mut hooks = ncm_model::hybrid::Hooks {
            scan: Some(&mut scan),
            pick: Some(&mut pick),
            ..Default::default()
        };
        f(&mut hooks)
    };
    *constraint = shared.into_inner().0;
    out
}
