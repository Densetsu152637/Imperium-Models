//! The hybrid's fast block decode path ([`FastSession`]): fixed device buffers, passes free of
//! host data (so they can be captured as CUDA graphs), acceptance on the device, and the same
//! passes batched over several slots' sessions ([`block_pass_slots`], [`ar_pass_slots`]).
//!
//! What it changes against the reference loop ([`super::decode`]):
//!
//! * **Memory rings.** Every generator and token-decoder layer keeps the keys and values of the
//!   recent encoder states in a ring of `recent + L` slots, projected once when a state enters
//!   it (with RoPE at its position), as do the copy head's keys and the tokens that followed each
//!   state. Per block only the per-block entries (slots, retrieved tokens, known prefix) are
//!   projected. Visibility is an additive bias over the ring, staged per block.
//! * **Block pass.** One pass runs the generator's K levels (level inputs pooled and projected,
//!   six layers, tied codebook logits, the code, the codebook gather, upsampling and φ), the
//!   tokenizer decoder's draft with the known prefix forced, the token decoder's check pass over
//!   the draft, and the acceptance: the first rejected position at or after the known prefix
//!   (verified: the autoregressive path's token differs from the draft; parallel: the pass with
//!   ẑ disagrees or its top probability is below τ). The host reads back `[first, any, ids (L),
//!   draft (L)]`, `2 + 2L` integers per row.
//! * **Autoregressive pass.** One token-decoder pass on the autoregressive path over the
//!   block's fixed tokens: the token (argmax or device draw) and log-probabilities of every
//!   position.
//! * **Batching.** A pass runs over a batch of slots: every input is the slots' fixed buffers
//!   stacked along the batch (rings, biases, positions' tables, known prefixes), so the norms,
//!   projections, MLPs, codebook logits and acceptance run once for `slots × block` rows. A
//!   block's host logic is a state machine ([`BlockRun`]): each slot is in a pass (first, a
//!   re-decode or a recheck) or in its autoregressive fill, and the batcher runs one batched
//!   pass per group.
//! * **Graphs** (CUDA, graphs on): each pass is captured once per mode and slot set over the
//!   fixed buffers and replayed; every other device, or a failed capture, runs the same passes
//!   eagerly.
//!
//! Scope: temperature 0 (verified or parallel mode) everywhere; on CUDA also T > 0: parallel
//! mode samples the codes on the device (tokens argmax), verified mode samples each position's
//! token from the autoregressive path on the device and accepts the draft token where the draw
//! equals it. For a one-token (point-mass) draft that is exactly speculative sampling: the draft
//! token is accepted with probability `min(1, p_ar(d) / 1) = p_ar(d)`, and a rejected position's
//! draw is distributed as the residual `p_ar` without `d`, renormalised; the output has the
//! autoregressive path's distribution. Draws are seeded per request (counter-based, one settings
//! set per row). The test hooks that rewrite drafts or codes stay on the reference loop.

use burn::tensor::{Bool, Device, Int, Tensor, TensorData};

use super::decode::{BlockOut, BlockStats, DecodeOptions, Hooks, HybridSession};
use super::memory::RopeTables;
use super::model::{FastMemory, HybridLm, retrieved_tokens};
use super::vq::Resampler;
use crate::layers::Rope;

/// Writes `src` into row `row` of the pooled `dst` (axis 0), at offset `at` along `axis`: in
/// `dst`'s own buffer on CUDA with the fused kernels on (a graph reads it), by replacing it
/// elsewhere.
fn put<const D: usize>(dst: &mut Tensor<D>, row: usize, src: Tensor<D>, axis: usize, at: usize) {
    #[cfg(feature = "cuda")]
    if crate::fused::active(&dst.device()) {
        crate::kernels::assign_into(&dst.clone().narrow(0, row, 1), src, axis, at);
        return;
    }
    let n = src.dims()[axis];
    let mut ranges: Vec<std::ops::Range<usize>> = dst.dims().iter().map(|&d| 0..d).collect();
    ranges[0] = row..row + 1;
    ranges[axis] = at..at + n;
    let ranges: [std::ops::Range<usize>; D] = ranges.try_into().expect("rank");
    *dst = dst.clone().slice_assign(ranges, src);
}

/// [`put`] for int tensors.
fn put_int<const D: usize>(
    dst: &mut Tensor<D, Int>,
    row: usize,
    src: Tensor<D, Int>,
    axis: usize,
    at: usize,
) {
    #[cfg(feature = "cuda")]
    if crate::fused::active(&dst.device()) {
        crate::kernels::assign_int_into(&dst.clone().narrow(0, row, 1), src, axis, at);
        return;
    }
    let n = src.dims()[axis];
    let mut ranges: Vec<std::ops::Range<usize>> = dst.dims().iter().map(|&d| 0..d).collect();
    ranges[0] = row..row + 1;
    ranges[axis] = at..at + n;
    let ranges: [std::ops::Range<usize>; D] = ranges.try_into().expect("rank");
    *dst = dst.clone().slice_assign(ranges, src);
}

fn floats<const D: usize>(values: Vec<f32>, shape: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::from_data(TensorData::new(values, shape), device)
}

fn ints<const D: usize>(values: Vec<i32>, shape: [usize; D], device: &Device) -> Tensor<D, Int> {
    Tensor::from_data(TensorData::new(values, shape), device)
}

/// Rewrites row `row` of the pooled `rope` tables with those of `positions`.
fn restage(
    rope: &mut RopeTables,
    row: usize,
    positions: &[usize],
    head_dim: usize,
    base: f64,
    device: &Device,
) {
    let fresh = RopeTables::at(positions, head_dim, base, device);
    put(&mut rope.cos, row, fresh.cos, 2, 0);
    put(&mut rope.sin, row, fresh.sin, 2, 0);
}

/// The recent encoder states' memory keys and values, per layer, in rings of `recent + L`
/// slots (token t in slot t mod capacity), with the copy keys and the tokens that followed;
/// one row per slot of the pool.
struct Rings {
    generator: Vec<(Tensor<4>, Tensor<4>)>,
    dec: Vec<(Tensor<4>, Tensor<4>)>,
    copy_keys: Tensor<3>,
    next: Tensor<2, Int>,
    /// Visibility for the memory attention and for the copy head `[rows, capacity]`.
    seq_bias: Tensor<2>,
    copy_bias: Tensor<2>,
    capacity: usize,
}

/// The per-block inputs, in fixed buffers, one row per slot of the pool.
struct Inputs {
    cond: Tensor<3>,
    retrieved: Tensor<2, Int>,
    retrieved_next: Tensor<2, Int>,
    retrieved_bias: Tensor<2>,
    known_states: Tensor<3>,
    known_tokens: Tensor<2, Int>,
    /// The known-prefix length `[1, 1]`.
    p: Tensor<2, Int>,
    /// The per-block entries' tables (slots and retrieved at the block's start, the known
    /// prefix at its positions) and the queries' (generator: the start, L rows; token decoder:
    /// start + i).
    extra_rope: RopeTables,
    gen_q: RopeTables,
    dec_q: RopeTables,
    /// The autoregressive pass's input tokens `[1, L]`.
    ar_tokens: Tensor<2, Int>,
    /// A draft to check again without generating (after a mixed-block switch) `[1, L]`.
    draft: Tensor<2, Int>,
}

/// Tables fixed for the model.
struct Consts {
    resampler: Resampler,
    level_ropes: Vec<Rope>,
    tok_rope: Rope,
    dec_rope: Rope,
    causal: Tensor<2, Bool>,
    /// Positions 0..L `[1, L]` and 0..L-1 `[1, L - 1]`.
    pos: Tensor<2, Int>,
    known_pos: Tensor<2, Int>,
    /// `L - pos`, the first-rejection score `[1, L]`.
    rank: Tensor<2, Int>,
    slots_bias: Tensor<2>,
    /// ln τ `[1, 1]` for parallel mode.
    ln_tau: Tensor<2>,
}

/// The fast path's device buffers of `rows` slots, each slot's in one row of shared pools
/// (memory rings, visibility biases, per-block inputs, RoPE tables): a pass over any of the
/// slots reads them in place, all rows at once, and the rows of slots not in the pass are
/// computed and ignored (their slot entries keep every softmax finite). A graph captured over
/// the pool serves every slot set.
pub struct SlotPool {
    rows: usize,
    rings: Rings,
    inputs: Inputs,
    consts: Consts,
    device: Device,
}

/// A pool shared by its slots' sessions (one thread: the engine's worker).
pub type SharedPool = std::rc::Rc<std::cell::RefCell<SlotPool>>;

impl SlotPool {
    pub fn new(model: &HybridLm, rows: usize, device: &Device) -> SharedPool {
        let h = model.config().clone();
        let (l, d, hd) = (h.block, model.local_dim, model.head_dim);
        let capacity = h.recent + l;
        let ring = |kvh: usize| {
            (
                Tensor::<4>::zeros([rows, kvh, capacity, hd], device),
                Tensor::<4>::zeros([rows, kvh, capacity, hd], device),
            )
        };
        let generator = model.generator_cross().iter().map(|c| ring(c.kv_heads())).collect();
        let dec = model.decoder_cross().iter().map(|c| ring(c.kv_heads())).collect();
        let neg = vec![f32::NEG_INFINITY; rows * capacity];
        let rings = Rings {
            generator,
            dec,
            copy_keys: Tensor::zeros([rows, capacity, h.copy_dim], device),
            next: Tensor::zeros([rows, capacity], device),
            seq_bias: floats(neg.clone(), [rows, capacity], device),
            copy_bias: floats(neg, [rows, capacity], device),
            capacity,
        };
        let rt = h.retrieved_blocks * l;
        let e = h.slots + rt + (l - 1);
        let base = model.rope_base;
        let tables = |n: usize| {
            let t = RopeTables::at(&vec![0usize; n], hd, base, device);
            RopeTables { cos: t.cos.repeat_dim(0, rows), sin: t.sin.repeat_dim(0, rows) }
        };
        let width = rt.max(1);
        let inputs = Inputs {
            cond: Tensor::zeros([rows, 1, model.start_condition(1).dims()[2]], device),
            retrieved: Tensor::zeros([rows, width], device),
            retrieved_next: Tensor::zeros([rows, width], device),
            retrieved_bias: floats(vec![f32::NEG_INFINITY; rows * width], [rows, width], device),
            known_states: Tensor::zeros([rows, l - 1, d], device),
            known_tokens: Tensor::zeros([rows, l], device),
            p: Tensor::zeros([rows, 1], device),
            extra_rope: tables(e),
            gen_q: tables(l),
            dec_q: tables(l),
            ar_tokens: Tensor::zeros([rows, l], device),
            draft: Tensor::zeros([rows, l], device),
        };
        let (dec_rope, causal) = model.decoder_self(device);
        let pos: Vec<i32> = (0..l as i32).collect();
        let consts = Consts {
            resampler: model.tokenizer().resampler(device),
            level_ropes: h
                .levels
                .iter()
                .map(|&n| Rope::at(&(0..n).collect::<Vec<_>>(), hd, base, device))
                .collect(),
            tok_rope: model.tokenizer().decoder_rope(device),
            dec_rope,
            causal,
            pos: ints(pos.clone(), [1, l], device),
            known_pos: ints(pos[..l - 1].to_vec(), [1, l - 1], device),
            rank: ints(pos.iter().map(|p| l as i32 - p).collect(), [1, l], device),
            slots_bias: Tensor::zeros([1, h.slots], device),
            ln_tau: floats(vec![(h.tau.max(1e-9)).ln() as f32], [1, 1], device),
        };
        std::rc::Rc::new(std::cell::RefCell::new(SlotPool {
            rows,
            rings,
            inputs,
            consts,
            device: device.clone(),
        }))
    }

    /// Rows (slots) of the pool.
    pub fn rows(&self) -> usize {
        self.rows
    }
}

/// How a block pass judges the draft.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PassMode {
    /// Against the autoregressive path (no ẑ), greedy.
    Verified,
    /// Against the autoregressive path, its tokens drawn on the device (speculative sampling,
    /// CUDA).
    VerifiedSampled,
    /// The staged draft checked again against the autoregressive path, no generator (after a
    /// mixed-block switch), greedy or sampled.
    Recheck,
    RecheckSampled,
    /// With ẑ, greedy codes.
    Parallel,
    /// With ẑ, codes sampled on the device (CUDA).
    ParallelSampled,
}

impl PassMode {
    fn verified(self) -> bool {
        matches!(
            self,
            Self::Verified | Self::VerifiedSampled | Self::Recheck | Self::RecheckSampled
        )
    }

    fn sampled_tokens(self) -> bool {
        matches!(self, Self::VerifiedSampled | Self::RecheckSampled)
    }

    fn sampled_codes(self) -> bool {
        self == Self::ParallelSampled
    }

    fn generates(self) -> bool {
        !matches!(self, Self::Recheck | Self::RecheckSampled)
    }
}

/// What a block pass reports for one row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassResult {
    /// The first rejected position at or after the known prefix, if any.
    pub rejected: Option<usize>,
    /// The pass's token per position.
    pub ids: Vec<i32>,
    pub draft: Vec<i32>,
}

/// The device draws' settings of one pass, one per row (CUDA): per level `ℓ_k` code rows, and
/// L token rows. Empty for greedy passes.
#[derive(Clone, Debug, Default)]
pub struct DrawSettings {
    #[cfg(feature = "cuda")]
    codes: Vec<Vec<crate::sample::SampleSettings>>,
    #[cfg(feature = "cuda")]
    tokens: Vec<crate::sample::SampleSettings>,
}

impl DrawSettings {
    /// The settings of every row of a pool of `rows` slots: each slot of `parts` (its row and
    /// settings) at its row, greedy draws (top-k 1) for the other rows.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn rows(
        parts: &[(usize, &DrawSettings)],
        rows: usize,
        levels: &[usize],
        l: usize,
    ) -> DrawSettings {
        #[cfg(feature = "cuda")]
        {
            use crate::sample::SampleSettings;
            let fill = |n: usize, get: &dyn Fn(&DrawSettings) -> Option<Vec<SampleSettings>>| {
                let mut out = Vec::with_capacity(rows * n);
                for r in 0..rows {
                    let own = parts.iter().find(|p| p.0 == r).and_then(|p| get(p.1));
                    match own {
                        Some(v) if v.len() == n => out.extend(v),
                        _ => out.extend(std::iter::repeat_n(SampleSettings::GREEDY, n)),
                    }
                }
                out
            };
            let sampled_codes = parts.iter().any(|p| !p.1.codes.is_empty());
            let sampled_tokens = parts.iter().any(|p| !p.1.tokens.is_empty());
            let codes = if sampled_codes {
                levels
                    .iter()
                    .enumerate()
                    .map(|(k, &n)| fill(n, &|d| d.codes.get(k).cloned()))
                    .collect()
            } else {
                Vec::new()
            };
            let tokens = if sampled_tokens {
                fill(l, &|d| (!d.tokens.is_empty()).then(|| d.tokens.clone()))
            } else {
                Vec::new()
            };
            DrawSettings { codes, tokens }
        }
        #[cfg(not(feature = "cuda"))]
        DrawSettings {}
    }
}

/// Device samplers with one settings set per row (CUDA), held by a pass (or its graph).
struct Samplers {
    #[cfg(feature = "cuda")]
    codes: Vec<crate::sample::SampleParams>,
    #[cfg(feature = "cuda")]
    tokens: crate::sample::SampleParams,
}

impl Samplers {
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn new(levels: &[usize], rows: usize, l: usize, device: &Device) -> Self {
        Samplers {
            #[cfg(feature = "cuda")]
            codes: levels
                .iter()
                .map(|&n| crate::sample::SampleParams::rows(device, rows * n))
                .collect(),
            #[cfg(feature = "cuda")]
            tokens: crate::sample::SampleParams::rows(device, rows * l),
        }
    }

    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn set(&self, settings: &DrawSettings) {
        #[cfg(feature = "cuda")]
        {
            for (p, s) in self.codes.iter().zip(&settings.codes) {
                if !s.is_empty() {
                    p.set_rows(s);
                }
            }
            if !settings.tokens.is_empty() {
                self.tokens.set_rows(&settings.tokens);
            }
        }
    }
}

/// Captured passes per mode (CUDA) over one [`SlotPool`], kept by the pool's owner: a
/// [`FastSession`] for its own pool, the batcher for its slots'. A pass reads every row of the
/// pool, so one graph per mode serves every slot set.
#[derive(Default)]
pub struct PassGraphs {
    #[cfg(feature = "cuda")]
    block: Vec<(PassMode, Result<BlockGraph, String>)>,
    #[cfg(feature = "cuda")]
    ar: Vec<(bool, Result<ArGraph, String>)>,
}

/// A captured block pass, its samplers and output.
#[cfg(feature = "cuda")]
type BlockGraph = (crate::graph::Captured, Samplers, Tensor<2, Int>);
/// A captured autoregressive pass, its samplers and outputs (tokens, log-probabilities).
#[cfg(feature = "cuda")]
type ArGraph = (crate::graph::Captured, Samplers, (Tensor<2, Int>, Tensor<3>));

impl PassGraphs {
    /// Why a pass runs eagerly, if a capture failed.
    pub fn failure(&self) -> Option<String> {
        #[cfg(feature = "cuda")]
        {
            let block = self.block.iter().find_map(|(_, g)| g.as_ref().err().cloned());
            let ar = self.ar.iter().find_map(|(_, g)| g.as_ref().err().cloned());
            block.or(ar)
        }
        #[cfg(not(feature = "cuda"))]
        None
    }
}

/// The inputs of a pass: the pool's buffers themselves (handles, no copies).
struct PassInputs {
    b: usize,
    cond: Tensor<3>,
    retrieved: Tensor<2, Int>,
    retrieved_next: Tensor<2, Int>,
    retrieved_bias: Tensor<2>,
    known_states: Tensor<3>,
    known_tokens: Tensor<2, Int>,
    p: Tensor<2, Int>,
    extra_rope: RopeTables,
    gen_q: RopeTables,
    dec_q: RopeTables,
    ar_tokens: Tensor<2, Int>,
    draft: Tensor<2, Int>,
    generator: Vec<(Tensor<4>, Tensor<4>)>,
    dec: Vec<(Tensor<4>, Tensor<4>)>,
    copy_keys: Tensor<3>,
    next: Tensor<2, Int>,
    seq_bias: Tensor<2>,
    copy_bias: Tensor<2>,
}

impl PassInputs {
    fn of(pool: &SlotPool) -> Self {
        let (i, r) = (&pool.inputs, &pool.rings);
        PassInputs {
            b: pool.rows,
            cond: i.cond.clone(),
            retrieved: i.retrieved.clone(),
            retrieved_next: i.retrieved_next.clone(),
            retrieved_bias: i.retrieved_bias.clone(),
            known_states: i.known_states.clone(),
            known_tokens: i.known_tokens.clone(),
            p: i.p.clone(),
            extra_rope: i.extra_rope.clone(),
            gen_q: i.gen_q.clone(),
            dec_q: i.dec_q.clone(),
            ar_tokens: i.ar_tokens.clone(),
            draft: i.draft.clone(),
            generator: r.generator.clone(),
            dec: r.dec.clone(),
            copy_keys: r.copy_keys.clone(),
            next: r.next.clone(),
            seq_bias: r.seq_bias.clone(),
            copy_bias: r.copy_bias.clone(),
        }
    }
}

// ---------------------------------------------------------------------- the passes

/// The token decoder over inputs `x` with the batch's decoder memory.
fn decoder_pass(
    model: &HybridLm,
    c: &Consts,
    i: &PassInputs,
    x: Tensor<3>,
    dec_kv: &[(Tensor<4>, Tensor<4>)],
    dec_bias: &Tensor<2>,
    retrieved: Option<Tensor<3>>,
) -> Tensor<3> {
    let memory = FastMemory {
        seq: Some(&i.dec),
        seq_bias: &i.seq_bias,
        extra: dec_kv,
        extra_bias: dec_bias,
        q_rope: &i.dec_q,
    };
    let copy_seq = Some((&i.copy_keys, &i.next, &i.copy_bias));
    let copy_ret = retrieved.map(|v| (v, &i.retrieved_next, &i.retrieved_bias));
    model.decoder_fast(x, &c.dec_rope, c.causal.clone(), &memory, (copy_seq, copy_ret))
}

/// The per-block entries shared by both passes (slots, retrieved tokens) with their biases,
/// and the retrieved entries alone.
fn shared_entries(
    model: &HybridLm,
    c: &Consts,
    i: &PassInputs,
) -> (Vec<Tensor<3>>, Vec<Tensor<2>>, Option<Tensor<3>>) {
    let h = model.config();
    let rt = h.retrieved_blocks * h.block;
    let slots = model.slot_vectors(i.cond.clone());
    let retrieved = (rt > 0).then(|| model.retrieved_vectors(i.retrieved.clone()));
    let mut shared = vec![slots];
    let mut bias = vec![c.slots_bias.clone().expand([i.b, h.slots])];
    if let Some(v) = &retrieved {
        shared.push(v.clone());
        bias.push(i.retrieved_bias.clone());
    }
    (shared, bias, retrieved)
}

/// The block pass (module docs) over a batch: `[B, 2 + 2L]` = per row the first rejected
/// position, whether any, the tokens, the draft. Only device buffers: capturable.
fn block_pass(
    model: &HybridLm,
    c: &Consts,
    i: &PassInputs,
    mode: PassMode,
    samplers: &Samplers,
) -> Tensor<2, Int> {
    let h = model.config();
    let (l, b) = (h.block, i.b);
    let device = i.cond.device();
    let (mut shared, mut shared_bias, retrieved) = shared_entries(model, c, i);
    let known = model.known_vectors(i.known_states.clone());
    let p_known = i.p.clone().expand([b, l - 1]);
    let known_bias = Tensor::<2>::zeros([b, l - 1], &device).mask_fill(
        c.known_pos.clone().expand([b, l - 1]).greater_equal(p_known),
        f32::NEG_INFINITY,
    );
    let n_shared = shared.iter().map(|t| t.dims()[1]).sum::<usize>();
    let dec_extra = Tensor::cat(shared.clone(), 1);
    let dec_bias = Tensor::cat(shared_bias.clone(), 1);
    shared.push(known);
    shared_bias.push(known_bias);
    let gen_extra = Tensor::cat(shared, 1);
    let gen_bias = Tensor::cat(shared_bias, 1);
    let dec_rope_rows = i.extra_rope.rows(0, n_shared);
    let gen_kv: Vec<_> = model
        .generator_cross()
        .iter()
        .map(|x| x.project_rows_tables(gen_extra.clone(), &i.extra_rope))
        .collect();
    let dec_kv: Vec<_> = model
        .decoder_cross()
        .iter()
        .map(|x| x.project_rows_tables(dec_extra.clone(), &dec_rope_rows))
        .collect();
    // The generator, level by level (none when the staged draft is checked again).
    let tok = model.tokenizer();
    let mut zhat: Option<Tensor<3>> = None;
    let levels: &[usize] = if mode.generates() { &h.levels } else { &[] };
    for (k, &n) in levels.iter().enumerate() {
        let q_rope = i.gen_q.rows(0, n);
        let memory = FastMemory {
            seq: Some(&i.generator),
            seq_bias: &i.seq_bias,
            extra: &gen_kv,
            extra_bias: &gen_bias,
            q_rope: &q_rope,
        };
        let x = model.level_input_with(k, zhat.clone(), &c.resampler);
        let [xb, xq, xd] = x.dims();
        let x = if xb == b { x } else { x.expand([b, xq, xd]) };
        let out = model.generator_fast(x, &c.level_ropes[k], &memory);
        let logits = tok.code_logits_with(k, out, &c.resampler);
        let codes = pick_codes(k, logits, mode, samplers);
        zhat = Some(tok.add_contribution(k, codes, &c.resampler, zhat));
    }
    // The draft, the known prefix forced.
    let draft = match &zhat {
        Some(z) => {
            crate::fused::argmax_last(tok.decode_with(z.clone(), model.embedding(), &c.tok_rope))
                .reshape([b, l])
        }
        None => i.draft.clone(),
    };
    let p = i.p.clone().expand([b, l]);
    let pos = c.pos.clone().expand([b, l]);
    let draft = draft.mask_where(pos.clone().lower(p.clone()), i.known_tokens.clone());
    // The check pass and the acceptance.
    let verified = mode.verified();
    let x = model.decoder_input_fast(draft.clone(), if verified { None } else { zhat });
    let logp = decoder_pass(model, c, i, x, &dec_kv, &dec_bias, retrieved);
    // Each position's token: the argmax, or a draw from the autoregressive path (where a draw
    // equal to the draft token accepts it: speculative sampling for a one-token draft).
    let ids = pick_tokens(logp.clone(), mode.sampled_tokens(), samplers);
    let mut reject = ids.clone().not_equal(draft.clone());
    if !verified {
        let low = logp.max_dim(2).reshape([b, l]).lower(c.ln_tau.clone().expand([b, l]));
        reject = reject.bool_or(low);
    }
    let reject = reject.bool_and(pos.greater_equal(p));
    let score = reject.int() * c.rank.clone().expand([b, l]);
    let first = score.clone().argmax(1).reshape([b, 1]);
    let any = score.max_dim(1).reshape([b, 1]);
    Tensor::cat(vec![first, any, ids, draft], 1)
}

/// The autoregressive pass over `ar_tokens` for a batch: tokens `[B, L]` and log-probabilities.
fn ar_pass(
    model: &HybridLm,
    c: &Consts,
    i: &PassInputs,
    sampled: bool,
    samplers: &Samplers,
) -> (Tensor<2, Int>, Tensor<3>) {
    let (shared, bias, retrieved) = shared_entries(model, c, i);
    let n = shared.iter().map(|t| t.dims()[1]).sum::<usize>();
    let extra = Tensor::cat(shared, 1);
    let bias = Tensor::cat(bias, 1);
    let rope = i.extra_rope.rows(0, n);
    let kv: Vec<_> =
        model.decoder_cross().iter().map(|x| x.project_rows_tables(extra.clone(), &rope)).collect();
    let x = model.decoder_input_fast(i.ar_tokens.clone(), None);
    let logp = decoder_pass(model, c, i, x, &kv, &bias, retrieved);
    let ids = pick_tokens(logp.clone(), sampled, samplers);
    (ids, logp)
}

/// Every position's token `[B, L]` from log-probabilities `[B, L, V]`: the argmax, or a draw
/// by the device token sampler (one settings set per row, CUDA).
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
fn pick_tokens(logp: Tensor<3>, sampled: bool, samplers: &Samplers) -> Tensor<2, Int> {
    let [b, l, v] = logp.dims();
    #[cfg(feature = "cuda")]
    if sampled {
        return crate::sample::sample(logp.reshape([b * l, v]), &samplers.tokens, 0)
            .reshape([b, l]);
    }
    crate::fused::argmax_last(logp).reshape([b, l])
}

/// Level `k`'s codes `[B, ℓ]` from its logits: the argmax, or a device draw.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
fn pick_codes(k: usize, logits: Tensor<3>, mode: PassMode, samplers: &Samplers) -> Tensor<2, Int> {
    let [b, n, e] = logits.dims();
    #[cfg(feature = "cuda")]
    if mode.sampled_codes() {
        return crate::sample::sample(logits.reshape([b * n, e]), &samplers.codes[k], 0)
            .reshape([b, n]);
    }
    crate::fused::argmax_last(logits).reshape([b, n])
}

fn read_results(out: Tensor<2, Int>, l: usize) -> Vec<PassResult> {
    let b = out.dims()[0];
    let v: Vec<i32> = out.into_data().convert::<i32>().try_to_vec::<i32>().expect("ints");
    let w = 2 + 2 * l;
    (0..b)
        .map(|r| {
            let row = &v[r * w..(r + 1) * w];
            PassResult {
                rejected: (row[1] > 0).then_some(row[0] as usize),
                ids: row[2..2 + l].to_vec(),
                draft: row[2 + l..2 + 2 * l].to_vec(),
            }
        })
        .collect()
}

/// The block pass over the slots `sessions` (staged with [`FastSession::prepare_pass`], all of
/// one pool) with their draws' settings: one result per slot. The pass runs over every row of
/// the pool (a captured graph per mode serves any slot set).
pub fn block_pass_slots(
    model: &HybridLm,
    sessions: &[&FastSession],
    mode: PassMode,
    settings: &[DrawSettings],
    graphs: &mut PassGraphs,
) -> Vec<PassResult> {
    let h = model.config();
    let l = h.block;
    let pool = sessions[0].pool.borrow();
    let device = pool.device.clone();
    let parts: Vec<(usize, &DrawSettings)> =
        sessions.iter().map(|s| s.row).zip(settings.iter()).collect();
    let settings = DrawSettings::rows(&parts, pool.rows, &h.levels, l);
    let pick = |all: Vec<PassResult>| -> Vec<PassResult> {
        sessions.iter().map(|s| all[s.row].clone()).collect()
    };
    #[cfg(feature = "cuda")]
    if crate::fused::graphs_active(&device) {
        if !graphs.block.iter().any(|(m, _)| *m == mode) {
            let samplers = Samplers::new(&h.levels, pool.rows, l, &device);
            samplers.set(&settings);
            let p = &*pool;
            let captured = crate::graph::record(
                &device,
                &mut (),
                2,
                |_| block_pass(model, &p.consts, &PassInputs::of(p), mode, &samplers),
                |_| {},
            )
            .map(|(g, out)| (g, samplers, out));
            graphs.block.push((mode, captured));
        }
        if let Some((_, Ok((graph, samplers, out)))) = graphs.block.iter().find(|(m, _)| *m == mode)
        {
            samplers.set(&settings);
            // Safety: the graph reads the pool's buffers, its samplers and the model's weights,
            // all alive (the replay checks its thread).
            unsafe { crate::graph::replay(&device, graph, "hybrid block") };
            return pick(read_results(out.clone(), l));
        }
    }
    let _ = &graphs;
    let samplers = Samplers::new(&h.levels, pool.rows, l, &device);
    samplers.set(&settings);
    pick(read_results(block_pass(model, &pool.consts, &PassInputs::of(&pool), mode, &samplers), l))
}

/// The autoregressive pass over the slots `sessions` (staged with [`FastSession::prepare_fill`],
/// all of one pool): per slot its tokens `[L]` (argmax or device draw) and log-probabilities
/// `[1, L, V]`.
pub fn ar_pass_slots(
    model: &HybridLm,
    sessions: &[&FastSession],
    sampled: bool,
    settings: &[DrawSettings],
    graphs: &mut PassGraphs,
) -> (Vec<Vec<i32>>, Vec<Tensor<3>>) {
    let h = model.config();
    let l = h.block;
    let pool = sessions[0].pool.borrow();
    let device = pool.device.clone();
    let parts: Vec<(usize, &DrawSettings)> =
        sessions.iter().map(|s| s.row).zip(settings.iter()).collect();
    let settings = DrawSettings::rows(&parts, pool.rows, &h.levels, l);
    let pick = |ids: Tensor<2, Int>, logp: Tensor<3>| -> (Vec<Vec<i32>>, Vec<Tensor<3>>) {
        let v: Vec<i32> = ids.into_data().convert::<i32>().try_to_vec::<i32>().expect("ints");
        let rows: Vec<Vec<i32>> = v.chunks(l).map(<[i32]>::to_vec).collect();
        (
            sessions.iter().map(|s| rows[s.row].clone()).collect(),
            sessions.iter().map(|s| logp.clone().narrow(0, s.row, 1)).collect(),
        )
    };
    #[cfg(feature = "cuda")]
    if crate::fused::graphs_active(&device) {
        if !graphs.ar.iter().any(|(s, _)| *s == sampled) {
            let samplers = Samplers::new(&h.levels, pool.rows, l, &device);
            samplers.set(&settings);
            let p = &*pool;
            let captured = crate::graph::record(
                &device,
                &mut (),
                2,
                |_| ar_pass(model, &p.consts, &PassInputs::of(p), sampled, &samplers),
                |_| {},
            )
            .map(|(g, out)| (g, samplers, out));
            graphs.ar.push((sampled, captured));
        }
        if let Some((_, Ok((graph, samplers, out)))) = graphs.ar.iter().find(|(s, _)| *s == sampled)
        {
            samplers.set(&settings);
            // Safety: as for the block graph.
            unsafe { crate::graph::replay(&device, graph, "hybrid ar") };
            return pick(out.0.clone(), out.1.clone());
        }
    }
    let _ = &graphs;
    let samplers = Samplers::new(&h.levels, pool.rows, l, &device);
    samplers.set(&settings);
    let (ids, logp) = ar_pass(model, &pool.consts, &PassInputs::of(&pool), sampled, &samplers);
    pick(ids, logp)
}

// ---------------------------------------------------------------------- one block, host side

/// Where a block's decode stands ([`FastSession::begin_block`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockPhase {
    /// A block pass is due (first, re-decode or recheck).
    Pass,
    /// The autoregressive fill: one token per pass.
    Fill,
    /// The block is decoded ([`FastSession::end_block`]).
    Done,
}

/// The host state of one block's decode on the fast path.
#[derive(Clone, Debug)]
pub struct BlockRun {
    start: usize,
    known: usize,
    limit: usize,
    fixed: Vec<i32>,
    pub stats: BlockStats,
    verified: bool,
    recheck: bool,
    redecode: bool,
    passes: usize,
    stopped: bool,
    close_span: bool,
    phase: BlockPhase,
}

impl BlockRun {
    pub fn phase(&self) -> BlockPhase {
        self.phase
    }

    /// The block position the next fill pass writes.
    pub fn next_position(&self) -> usize {
        self.fixed.len()
    }

    /// The pass mode due (in [`BlockPhase::Pass`]).
    pub fn mode(&self, opts: &DecodeOptions) -> PassMode {
        match (self.recheck, self.verified, opts.temperature > 0.0) {
            (true, _, false) => PassMode::Recheck,
            (true, _, true) => PassMode::RecheckSampled,
            (false, true, false) => PassMode::Verified,
            (false, true, true) => PassMode::VerifiedSampled,
            (false, false, false) => PassMode::Parallel,
            (false, false, true) => PassMode::ParallelSampled,
        }
    }

    /// Whether the fill draws its tokens on the device.
    pub fn fill_sampled(&self, opts: &DecodeOptions) -> bool {
        opts.temperature > 0.0 && cfg!(feature = "cuda")
    }
}

/// Decode state of one stream at batch 1 on the fast path; the reference session underneath
/// keeps the tokens, the encoder and backbone caches and the retrieval history.
pub struct FastSession {
    pub session: HybridSession,
    /// The pool holding this slot's buffers, and its row there.
    pool: SharedPool,
    row: usize,
    /// Tokens projected into this slot's rings so far (token t is in its slot when `t < upto`).
    upto: usize,
    block: usize,
    /// Device draws of the request so far (the token draws' counters).
    draws: u32,
    /// This session's own captured passes (as a slot of a batcher, the batcher's are used).
    graphs: PassGraphs,
    device: Device,
}

impl FastSession {
    pub fn new(model: &HybridLm, max_tokens: usize, device: &Device) -> Self {
        let pool = SlotPool::new(model, 1, device);
        Self::in_pool(model, pool, 0, HybridSession::new(model, max_tokens, device))
    }

    /// `slots` sessions over one shared [`SlotPool`] (rows 0..slots) and pooled backbone
    /// caches ([`HybridSession::slots`]): a batcher's slots.
    pub fn pooled(model: &HybridLm, slots: usize, max_tokens: usize, device: &Device) -> Vec<Self> {
        let pool = SlotPool::new(model, slots, device);
        HybridSession::slots(model, slots, max_tokens, device)
            .into_iter()
            .enumerate()
            .map(|(row, session)| Self::in_pool(model, pool.clone(), row, session))
            .collect()
    }

    fn in_pool(model: &HybridLm, pool: SharedPool, row: usize, session: HybridSession) -> Self {
        let device = pool.borrow().device.clone();
        Self {
            session,
            pool,
            row,
            upto: 0,
            block: model.config().block,
            draws: 0,
            graphs: PassGraphs::default(),
            device,
        }
    }

    /// Appends prompt tokens ([`HybridSession::prefill`]).
    pub fn prefill(&mut self, model: &HybridLm, tokens: &[i32]) {
        self.session.prefill(model, tokens);
    }

    pub fn tokens(&self) -> &[i32] {
        self.session.tokens()
    }

    /// The session's point to return to ([`HybridSession::snapshot`]).
    pub fn snapshot(&self) -> super::decode::HybridSnapshot {
        self.session.snapshot()
    }

    /// Back to a [`Self::snapshot`] of this session: the rings are projected again from the
    /// held encoder states at the next block (slots written since may hold later tokens).
    pub fn restore(&mut self, snapshot: &super::decode::HybridSnapshot) {
        self.session.restore(snapshot);
        self.upto = self.session.states_start;
    }

    /// This session made a copy of `src` (another session of the same model and size: a
    /// forked request), in its own buffers; the rings are projected again from the copied
    /// states at the next block.
    pub fn copy_from(&mut self, src: &FastSession) {
        self.session.copy_from(&src.session);
        self.upto = self.session.states_start;
        self.draws = 0;
    }

    /// Back to an empty stream, keeping every buffer (and so the captured graphs): the rings'
    /// contents are hidden by the next block's staged visibility.
    pub fn reset(&mut self, model: &HybridLm) {
        self.session.reset(model);
        self.upto = 0;
    }

    /// Starts a new request's draws (seeded by `seed`).
    pub fn begin_request(&mut self, seed: u64) {
        self.session.rng = seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1;
        self.draws = 0;
    }

    // ------------------------------------------------------------------ rings and inputs

    /// Projects the encoded states not yet in the rings (those still held) into their slots.
    fn sync_rings(&mut self, model: &HybridLm) {
        let s = &self.session;
        let mut pool = self.pool.borrow_mut();
        let row = self.row;
        let from = self.upto.max(s.states_start);
        let to = s.encoded;
        let cap = pool.rings.capacity;
        // Only the last `capacity` tokens survive in the ring.
        let from = from.max(to.saturating_sub(cap));
        if to <= from {
            self.upto = self.upto.max(to);
            return;
        }
        let states = s.states(from, to).expect("held states");
        let hd = model.head_dim;
        let rope = Rope::new(from, to - from, hd, model.rope_base, 1.0, &self.device);
        let copy = model.copy_head().keys(states.clone());
        // Pieces that do not wrap around the ring's end.
        let mut pieces = Vec::new();
        let mut t = from;
        while t < to {
            let slot = t % cap;
            let len = (cap - slot).min(to - t);
            pieces.push((t - from, slot, len));
            t += len;
        }
        let write = |ring: &mut (Tensor<4>, Tensor<4>), kv: (Tensor<4>, Tensor<4>)| {
            for &(off, slot, len) in &pieces {
                put(&mut ring.0, row, kv.0.clone().narrow(2, off, len), 2, slot);
                put(&mut ring.1, row, kv.1.clone().narrow(2, off, len), 2, slot);
            }
        };
        let rings = &mut pool.rings;
        for (ring, cross) in rings.generator.iter_mut().zip(model.generator_cross()) {
            write(ring, cross.project_rows(states.clone(), &rope));
        }
        for (ring, cross) in rings.dec.iter_mut().zip(model.decoder_cross()) {
            write(ring, cross.project_rows(states.clone(), &rope));
        }
        for &(off, slot, len) in &pieces {
            put(&mut rings.copy_keys, row, copy.clone().narrow(1, off, len), 1, slot);
        }
        self.upto = to;
    }

    /// Stages the open block's inputs: ring visibility, the condition, the retrieved tokens,
    /// the known prefix `fixed`, and the tables of the block's positions.
    fn stage(&mut self, model: &HybridLm, fixed: &[i32]) {
        self.sync_rings(model);
        let h = model.config();
        let (l, hd, base) = (h.block, model.head_dim, model.rope_base);
        let device = self.device.clone();
        let start = self.session.blocks() * l;
        let row = self.row;
        let cap = self.pool.borrow().rings.capacity;
        let tokens = self.session.tokens().to_vec();
        // Ring visibility: slot s holds the latest projected token t = s mod cap.
        let upto = self.upto;
        let (mut seq, mut copy, mut next) =
            (vec![f32::NEG_INFINITY; cap], vec![f32::NEG_INFINITY; cap], vec![0i32; cap]);
        for t in upto.saturating_sub(cap)..upto {
            let slot = t % cap;
            let visible = t < start && t + h.recent >= start;
            seq[slot] = if visible { 0.0 } else { f32::NEG_INFINITY };
            copy[slot] = if visible && t + 1 < start { 0.0 } else { f32::NEG_INFINITY };
            next[slot] = tokens.get(t + 1).copied().unwrap_or(0);
        }
        {
            let mut pool = self.pool.borrow_mut();
            put(&mut pool.rings.seq_bias, row, floats(seq, [1, cap], &device), 1, 0);
            put(&mut pool.rings.copy_bias, row, floats(copy, [1, cap], &device), 1, 0);
            put_int(&mut pool.rings.next, row, ints(next, [1, cap], &device), 1, 0);
        }
        // The condition and the retrieved blocks.
        let cond = self.session.condition();
        let cond_host: Vec<f32> =
            cond.clone().into_data().convert::<f32>().try_to_vec::<f32>().expect("f32");
        put(&mut self.pool.borrow_mut().inputs.cond, row, cond, 2, 0);
        let rt = h.retrieved_blocks * l;
        if rt > 0 {
            let picked =
                model.select_retrieved(&cond_host, &self.session.pooled, self.session.blocks());
            let (mut ids, mut following) = retrieved_tokens(&tokens, &picked, l);
            let have = ids.len();
            let bias: Vec<f32> =
                (0..rt).map(|i| if i < have { 0.0 } else { f32::NEG_INFINITY }).collect();
            ids.resize(rt, 0);
            following.resize(rt, 0);
            let mut pool = self.pool.borrow_mut();
            put_int(&mut pool.inputs.retrieved, row, ints(ids, [1, rt], &device), 1, 0);
            put_int(&mut pool.inputs.retrieved_next, row, ints(following, [1, rt], &device), 1, 0);
            put(&mut pool.inputs.retrieved_bias, row, floats(bias, [1, rt], &device), 1, 0);
        }
        self.stage_known(model, fixed);
        let e_pos: Vec<usize> =
            std::iter::repeat_n(start, h.slots + rt).chain((0..l - 1).map(|i| start + i)).collect();
        let mut pool = self.pool.borrow_mut();
        let inputs = &mut pool.inputs;
        restage(&mut inputs.extra_rope, row, &e_pos, hd, base, &device);
        restage(&mut inputs.gen_q, row, &vec![start; l], hd, base, &device);
        let dec_pos: Vec<usize> = (start..start + l).collect();
        restage(&mut inputs.dec_q, row, &dec_pos, hd, base, &device);
    }

    /// Stages the known prefix `fixed`: its states, tokens and length.
    fn stage_known(&mut self, model: &HybridLm, fixed: &[i32]) {
        let l = model.config().block;
        let device = self.device.clone();
        let states = self.session.known_states(model, fixed);
        let row = self.row;
        let mut pool = self.pool.borrow_mut();
        put(&mut pool.inputs.known_states, row, states, 1, 0);
        let mut known = fixed.to_vec();
        known.resize(l, 0);
        put_int(&mut pool.inputs.known_tokens, row, ints(known, [1, l], &device), 1, 0);
        put_int(&mut pool.inputs.p, row, ints(vec![fixed.len() as i32], [1, 1], &device), 1, 0);
    }

    /// The token draws' settings of a pass of `L` rows (counters advance), and the code draws'
    /// (CUDA; empty for greedy passes).
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn draw_settings(
        &mut self,
        model: &HybridLm,
        opts: &DecodeOptions,
        codes: bool,
        tokens: bool,
    ) -> DrawSettings {
        let h = model.config();
        let l = h.block;
        #[cfg(feature = "cuda")]
        {
            use crate::sample::SampleSettings;
            let row = |t: f64, counter: u32| SampleSettings {
                temperature: t as f32,
                top_k: opts.top_k,
                top_p: opts.top_p as f32,
                seed: opts.seed,
                counter,
            };
            let mut out = DrawSettings::default();
            if tokens {
                out.tokens = (0..l)
                    .map(|j| row(opts.temperature, self.draws.wrapping_add(j as u32)))
                    .collect();
                self.draws = self.draws.wrapping_add(l as u32);
            }
            if codes {
                out.codes = h
                    .levels
                    .iter()
                    .enumerate()
                    .map(|(k, &n)| {
                        let t = level_temperature(opts.temperature, k, h.depth());
                        let base = self.draws;
                        self.draws = self.draws.wrapping_add(n as u32);
                        (0..n).map(|j| row(t, base.wrapping_add(j as u32))).collect()
                    })
                    .collect();
            }
            out
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = l;
            DrawSettings::default()
        }
    }

    // ------------------------------------------------------------------ the block state machine

    /// Whether the fast path serves `opts` (else [`Self::decode_block`] takes the reference
    /// loop).
    pub fn serves(&self, opts: &DecodeOptions) -> bool {
        let sampled = opts.temperature > 0.0;
        let device_sampling = cfg!(feature = "cuda") && crate::fused::active(&self.device);
        !sampled || device_sampling
    }

    /// Starts decoding the open block, at most `limit` tokens: stages its inputs.
    pub fn begin_block(
        &mut self,
        model: &HybridLm,
        limit: usize,
        opts: &DecodeOptions,
    ) -> BlockRun {
        let l = model.config().block;
        let start = self.session.blocks() * l;
        let fixed: Vec<i32> = self.session.tokens()[start..].to_vec();
        let known = fixed.len();
        self.stage(model, &fixed);
        BlockRun {
            start,
            known,
            limit,
            fixed,
            stats: BlockStats { known, ..BlockStats::default() },
            verified: opts.verified || self.session.verified_span,
            recheck: false,
            redecode: false,
            passes: 0,
            stopped: false,
            close_span: false,
            phase: if opts.ar_only { BlockPhase::Fill } else { BlockPhase::Pass },
        }
    }

    /// Stages a due pass (a re-decode's or recheck's known prefix) and returns its mode and
    /// draws' settings.
    pub fn prepare_pass(
        &mut self,
        model: &HybridLm,
        run: &BlockRun,
        opts: &DecodeOptions,
    ) -> (PassMode, DrawSettings) {
        if run.redecode || run.recheck {
            let fixed = run.fixed.clone();
            self.stage_known(model, &fixed);
        }
        let mode = run.mode(opts);
        let settings = self.draw_settings(model, opts, mode.sampled_codes(), mode.sampled_tokens());
        (mode, settings)
    }

    /// Applies a block pass's result (module docs; decoding step 5): accepted tokens, the
    /// rejected position's token, stops, a mixed-block switch, the constraint scan, then a
    /// re-decode, a recheck, the fill or the block's end.
    pub fn apply_pass(
        &mut self,
        model: &HybridLm,
        run: &mut BlockRun,
        result: PassResult,
        opts: &DecodeOptions,
        hooks: &mut Hooks,
    ) {
        let h = model.config();
        let l = h.block;
        let redecode_min = opts.redecode_min.unwrap_or(h.redecode_min);
        let mode = run.mode(opts);
        run.recheck = false;
        run.redecode = false;
        run.stats.generator_runs += usize::from(mode.generates());
        run.stats.check_passes += 1;
        let p = run.fixed.len();
        let mut cut = match hooks.accept.as_mut() {
            Some(f) => (p..l).find(|&i| !f(run.passes, i)),
            None => result.rejected,
        };
        if cut.is_some_and(|c| c < p) {
            cut = Some(p);
        }
        let accepted = cut.unwrap_or(l) - p;
        match run.passes {
            0 => run.stats.first_accepted = Some(accepted),
            1 => run.stats.second_accepted = Some(accepted),
            _ => {}
        }
        run.passes += 1;
        let mut emitted: Vec<i32> = result.draft[p..p + accepted].to_vec();
        if let Some(c) = cut {
            emitted.push(result.ids[c]);
        }
        emitted.truncate((run.known + run.limit).saturating_sub(run.fixed.len()));
        let mut keep = emitted.len();
        if let Some(s) = emitted.iter().position(|t| opts.stop.contains(t)) {
            keep = s + 1;
            run.stopped = true;
        }
        let switch = (!run.verified)
            .then(|| emitted[..keep].iter().position(|t| opts.switch_on.contains(t)))
            .flatten();
        if let Some(s) = switch {
            keep = s + 1;
            run.stopped = false;
        }
        emitted.truncate(keep);
        let constrained = super::decode::scan_cut(hooks, &mut emitted);
        if constrained {
            run.stopped = false;
        }
        if emitted.iter().any(|t| opts.switch_off.contains(t)) {
            run.close_span = true;
        }
        run.fixed.extend_from_slice(&emitted);
        if run.fixed.len() >= run.known + run.limit {
            run.stopped = true;
        }
        run.phase = if run.stopped || run.fixed.len() >= l {
            BlockPhase::Done
        } else if constrained {
            // The grammar cut the block: the rest on the autoregressive path.
            BlockPhase::Fill
        } else if let Some(s) = switch {
            run.stats.switched = Some(p + s);
            run.verified = true;
            self.session.verified_span = true;
            // One more check of the same draft, in verified mode.
            let device = self.device.clone();
            let row = self.row;
            put_int(
                &mut self.pool.borrow_mut().inputs.draft,
                row,
                ints(result.draft.clone(), [1, l], &device),
                1,
                0,
            );
            run.recheck = true;
            BlockPhase::Pass
        } else if cut.is_none() {
            BlockPhase::Pass
        } else if !run.stats.redecoded && l - run.fixed.len() >= redecode_min {
            run.stats.redecoded = true;
            run.redecode = true;
            BlockPhase::Pass
        } else {
            BlockPhase::Fill
        };
    }

    /// Stages the fill's next pass (the block's fixed tokens) and returns the draws' settings.
    pub fn prepare_fill(
        &mut self,
        model: &HybridLm,
        run: &BlockRun,
        opts: &DecodeOptions,
    ) -> DrawSettings {
        let l = model.config().block;
        let mut inputs = run.fixed.clone();
        inputs.resize(l, 0);
        let device = self.device.clone();
        let row = self.row;
        put_int(
            &mut self.pool.borrow_mut().inputs.ar_tokens,
            row,
            ints(inputs, [1, l], &device),
            1,
            0,
        );
        self.draw_settings(model, opts, false, run.fill_sampled(opts))
    }

    /// Applies one fill pass: the token at the next position, from the pass's token `drawn`
    /// (argmax or device draw) or, when the host must choose (a grammar pick, a host draw, an
    /// observer), from its log-probabilities `row()`.
    pub fn apply_fill(
        &mut self,
        run: &mut BlockRun,
        drawn: i32,
        row: &mut dyn FnMut() -> Vec<f32>,
        opts: &DecodeOptions,
        hooks: &mut Hooks,
    ) {
        let i = run.fixed.len();
        let host_draw = opts.temperature > 0.0 && !run.fill_sampled(opts);
        let token = if host_draw || hooks.ar_rows.is_some() || hooks.pick.is_some() {
            let row = row();
            if let Some(f) = hooks.ar_rows.as_mut() {
                f(&run.fixed, i, &row);
            }
            if let Some(pick) = hooks.pick.as_mut() {
                pick(&row)
            } else if host_draw {
                let mut uniform = || self.session.uniform();
                super::decode::sample_row(
                    &row,
                    opts.temperature,
                    opts.top_p,
                    opts.top_k,
                    &mut uniform,
                ) as i32
            } else {
                drawn
            }
        } else {
            drawn
        };
        run.stats.ar_tokens += 1;
        run.fixed.push(token);
        if opts.switch_on.contains(&token) {
            self.session.verified_span = true;
        }
        if opts.switch_off.contains(&token) {
            run.close_span = true;
        }
        if opts.stop.contains(&token) || run.fixed.len() >= run.known + run.limit {
            run.stopped = true;
        }
        if run.stopped || run.fixed.len() >= self.block {
            run.phase = BlockPhase::Done;
        }
    }

    /// Ends the block: encodes what was written and steps the backbone over a complete block.
    pub fn end_block(&mut self, model: &HybridLm, run: BlockRun) -> BlockOut {
        let emitted = run.fixed[run.known..].to_vec();
        self.session.close_block(model, run.start, &emitted, run.stopped, run.close_span);
        BlockOut { tokens: emitted, stopped: run.stopped, stats: run.stats }
    }

    /// [`Self::end_block`] for several sessions at once: the backbone steps of the blocks they
    /// complete run as one batched forward ([`HybridSession::close_blocks`]).
    pub fn end_blocks(
        model: &HybridLm,
        sessions: &mut [&mut FastSession],
        runs: Vec<BlockRun>,
    ) -> Vec<BlockOut> {
        let closes: Vec<(usize, Vec<i32>, bool, bool)> = runs
            .iter()
            .map(|r| (r.start, r.fixed[r.known..].to_vec(), r.stopped, r.close_span))
            .collect();
        {
            let mut inner: Vec<&mut HybridSession> =
                sessions.iter_mut().map(|s| &mut s.session).collect();
            HybridSession::close_blocks(model, &mut inner, &closes);
        }
        runs.into_iter()
            .zip(closes)
            .map(|(run, (_, tokens, stopped, _))| BlockOut { tokens, stopped, stats: run.stats })
            .collect()
    }

    // ------------------------------------------------------------------ one slot

    /// Decodes the open block, at most `limit` tokens, as [`HybridSession::decode_block`] does
    /// (same steps, same results at temperature 0 up to rounding), on the fast passes. Only the
    /// `accept`, `ar_rows`, `scan` and `pick` hooks apply here; options the fast path does not
    /// serve take the reference loop.
    pub fn decode_block(
        &mut self,
        model: &HybridLm,
        limit: usize,
        opts: &DecodeOptions,
        hooks: &mut Hooks,
    ) -> BlockOut {
        if !self.serves(opts) || hooks.draft.is_some() || hooks.codes.is_some() {
            return self.session.decode_block(model, limit, opts, hooks);
        }
        let mut run = self.begin_block(model, limit, opts);
        let mut graphs = std::mem::take(&mut self.graphs);
        loop {
            match run.phase {
                BlockPhase::Pass => {
                    let (mode, settings) = self.prepare_pass(model, &run, opts);
                    let result = block_pass_slots(model, &[&*self], mode, &[settings], &mut graphs)
                        .pop()
                        .expect("one row");
                    self.apply_pass(model, &mut run, result, opts, hooks);
                }
                BlockPhase::Fill => {
                    let settings = self.prepare_fill(model, &run, opts);
                    let sampled = run.fill_sampled(opts);
                    let (ids, logp) =
                        ar_pass_slots(model, &[&*self], sampled, &[settings], &mut graphs);
                    let i = run.fixed.len();
                    let drawn = ids[0][i];
                    let logp = logp[0].clone();
                    let mut row = || -> Vec<f32> {
                        logp.clone()
                            .narrow(1, i, 1)
                            .into_data()
                            .convert::<f32>()
                            .try_to_vec::<f32>()
                            .expect("f32")
                    };
                    self.apply_fill(&mut run, drawn, &mut row, opts, hooks);
                }
                BlockPhase::Done => break,
            }
        }
        self.graphs = graphs;
        self.end_block(model, run)
    }

    /// Why a pass of this session's own runs eagerly, if a capture failed.
    pub fn failure(&self) -> Option<String> {
        self.graphs.failure()
    }

    /// Decodes up to `max_tokens` tokens of a reply ([`HybridSession::generate_reply`]).
    pub fn generate_reply(
        &mut self,
        model: &HybridLm,
        max_tokens: usize,
        opts: &DecodeOptions,
        hooks: &mut Hooks,
    ) -> (Vec<i32>, Vec<BlockStats>) {
        self.begin_request(opts.seed);
        let mut out = Vec::new();
        let mut stats = Vec::new();
        while out.len() < max_tokens {
            let block = self.decode_block(model, max_tokens - out.len(), opts, hooks);
            out.extend_from_slice(&block.tokens);
            stats.push(block.stats);
            if block.stopped {
                break;
            }
        }
        (out, stats)
    }
}

/// Code temperature of level `k` ([`super::decode`]'s schedule).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn level_temperature(t: f64, k: usize, depth: usize) -> f64 {
    if k < 3 || depth <= 3 {
        return t;
    }
    let span = (depth - 3).max(1) as f64;
    t * (1.0 - 0.95 * (k - 2) as f64 / span).max(0.05)
}
