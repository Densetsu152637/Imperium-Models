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
//! 3. runs one prefill chunk of one prefilling slot (round robin), so that a long prompt
//!    interleaves with the others' decoding;
//! 4. runs one decode step over every decoding slot: one token per slot for the token baseline,
//!    one concept for the NCM. The batched step ([`TokenLm::step_slots`],
//!    [`NcmLm::append_slots`] / [`NcmLm::decode_slots`]) runs the norms, input projections and
//!    MLPs over all rows at once and each slot's attention against its own caches.
//!
//! Tokens are chosen on the host per slot (each request's own sampler, seed and constraint),
//! except greedy unconstrained NCM tokens, picked on the device. The batched steps run eagerly
//! (the captured graphs serve one slot: TODO graphs per slot set). With one slot the engine runs
//! [`crate::engine`]'s sequential loop instead, unchanged.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::Instant;

use burn::tensor::{Device, Int, Tensor, TensorData};
use ncm_model::{LayerCache, NcmCache, NcmLm, TokenLm};

use crate::constrain::Constraint;
use crate::engine::{
    Core, DecodePath, EngineError, Event, Finish, GenStats, Job, Kept, MAX_EMPTY_CONCEPTS, State,
    panic_message, read_ids, read_logits,
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
}

/// A kept prefix of one slot (see [`crate::engine`]): its snapshot restores only that slot.
enum SlotKept {
    Tok(Kept<Vec<ncm_model::cache::LayerSnapshot>, Tensor<3>>),
    Ncm(Kept<ncm_model::ncm::NcmSnapshot, Tensor<3>>),
}

impl SlotKept {
    fn tokens(&self) -> &[i32] {
        match self {
            Self::Tok(k) => &k.tokens,
            Self::Ncm(k) => &k.tokens,
        }
    }

    fn has_out(&self) -> bool {
        match self {
            Self::Tok(k) => k.out.is_some(),
            Self::Ncm(k) => k.out.is_some(),
        }
    }
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
}

struct Slot {
    caches: Caches,
    kept: Vec<SlotKept>,
    last_used: u64,
    active: Option<Active>,
}

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
}

#[allow(clippy::large_enum_variant)]
enum Model {
    Tok(TokenLm),
    Ncm(NcmLm),
}

impl Batcher {
    /// `count` slots over `core`'s model: the core's own caches become slot 0, the rest are
    /// allocated for the same context.
    pub(crate) fn new(core: Core, count: usize, report: Arc<Mutex<SlotsReport>>) -> Self {
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
        let mut slots = vec![Slot { caches: first, kept: Vec::new(), last_used: 0, active: None }];
        for _ in 1..count {
            let caches = match &model {
                Model::Tok(m) => Caches::Tok(m.make_cache(capacity, 1, &device)),
                Model::Ncm(m) => Caches::Ncm(m.make_cache(capacity, 1, &device)),
            };
            slots.push(Slot { caches, kept: Vec::new(), last_used: 0, active: None });
        }
        Self { core, model, slots, queue: VecDeque::new(), report, clock: 0, next_prefill: 0 }
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
        self.prefill_one();
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
            Model::Tok(_) => (ids, Vec::new(), 0),
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
                    _ => unreachable!("a slot keeps states of its own caches"),
                };
                let n = slot.kept[k].tokens().len();
                (n, if n == stream.len() { out } else { None })
            }
            None => {
                match &mut slot.caches {
                    Caches::Tok(c) => c.iter_mut().for_each(LayerCache::reset),
                    Caches::Ncm(c) => c.reset(),
                }
                (0, None)
            }
        };
        let phase = if out.is_some() { Phase::Decode } else { Phase::Prefill };
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
        });
        if phase == Phase::Decode {
            self.prompt_done(i);
        }
    }

    // ------------------------------------------------------------------ prefill and forks

    /// One prefill chunk of the next prefilling slot (round robin).
    fn prefill_one(&mut self) {
        let n = self.slots.len();
        let Some(i) = (0..n)
            .map(|d| (self.next_prefill + d) % n)
            .find(|&i| self.slots[i].active.as_ref().is_some_and(|a| a.phase == Phase::Prefill))
        else {
            return;
        };
        self.next_prefill = (i + 1) % n;
        let chunk = self.core.chunk;
        let device = self.device().clone();
        let slot = &mut self.slots[i];
        let active = slot.active.as_mut().expect("prefilling");
        let start = active.prefilled;
        let out = match (&self.model, &mut slot.caches) {
            (Model::Tok(m), Caches::Tok(c)) => {
                let len = chunk.min(active.stream.len() - start);
                let piece = &active.stream[start..start + len];
                let t = Tensor::from_data(TensorData::new(piece.to_vec(), [1, len]), &device);
                active.prefilled += len;
                m.prefill(t, c, chunk)
            }
            (Model::Ncm(m), Caches::Ncm(c)) => {
                let k = m.concept_size();
                let chunk = (chunk - chunk % k).max(k);
                let len = chunk.min(active.stream.len() - start);
                let piece = &active.stream[start..start + len];
                let t = Tensor::from_data(TensorData::new(piece.to_vec(), [1, len]), &device);
                active.prefilled += len;
                Some(m.prefill(t, c, chunk))
            }
            _ => unreachable!("slots hold caches of the model"),
        };
        if active.prefilled >= active.stream.len() {
            active.out = out;
            active.phase = Phase::Decode;
            self.prompt_done(i);
            self.fork_from(i);
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
                        _ => unreachable!("slots hold caches of one model"),
                    }
                    // Its kept prefixes described the old contents.
                    b.kept.clear();
                    self.start(j, w);
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
            let logits = read_logits(a.out.as_ref().expect("logits after the prompt"));
            let token = a.sampler.pick(&logits, a.constraint.as_deref()).unwrap_or(ids.im_end);
            if let Some(c) = &mut a.constraint {
                c.advance(token);
            }
            match emit(a, token, ids) {
                Some(finish) => self.finish(i, finish),
                None if a.history.len() >= capacity => self.finish(i, Finish::Length),
                None => feed.push((i, token as i32)),
            }
        }
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

    /// NCM: one concept per slot from its condition (pending prompt tokens forced first), then
    /// the written concepts appended in one batched step.
    fn ncm_step(&mut self, rows: &[usize]) {
        let ids = self.ids();
        let capacity = self.core.capacity;
        let device = self.device().clone();
        let Model::Ncm(model) = &self.model else { unreachable!("the NCM") };
        let k = model.concept_size();
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
        let mut appends = Vec::new();
        for (r, &i) in rows.iter().enumerate() {
            let skip = forced[r].len();
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
                        appends.push((i, concept));
                    }
                }
            }
        }
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

    /// Ends the request on slot `i`: its stats go back, and the slot keeps the prompt's state
    /// and the state at the end of the reply for the next requests.
    fn finish(&mut self, i: usize, finish: Finish) {
        self.clock += 1;
        let reuse = self.core.prefix_reuse;
        let slot = &mut self.slots[i];
        slot.last_used = self.clock;
        let Some(a) = slot.active.take() else { return };
        let mut next = Vec::new();
        if let Some(prompt) = a.prompt_kept {
            if reuse && a.history.len() > a.stream.len() {
                let tokens = a.history.clone();
                let out = a.out.clone();
                next.push(match &slot.caches {
                    Caches::Tok(c) => SlotKept::Tok(Kept {
                        tokens,
                        snapshot: c.iter().map(LayerCache::snapshot).collect(),
                        out,
                    }),
                    Caches::Ncm(c) => SlotKept::Ncm(Kept { tokens, snapshot: c.snapshot(), out }),
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
            path: DecodePath::Batched,
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
            Caches::Ncm(_) => None,
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
            Caches::Tok(_) => None,
        })
        .collect();
    rows.iter()
        .map(|&i| {
            let (c, a) = picked[i].take().expect("one row per slot");
            (c, a.expect("an active slot"))
        })
        .unzip()
}
