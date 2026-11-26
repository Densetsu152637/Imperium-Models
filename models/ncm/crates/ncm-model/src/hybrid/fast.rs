//! The hybrid's fast block decode path ([`FastSession`]): fixed device buffers, passes free of
//! host data (so they can be captured as CUDA graphs), and acceptance on the device.
//!
//! What it changes against the reference loop ([`super::decode`]):
//!
//! * **Memory rings.** Every generator and token-decoder layer keeps the keys and values of the
//!   recent encoder states in a ring of `recent + L` slots, projected once when a state enters
//!   it (with RoPE at its position), as do the copy head's keys and the tokens that followed each
//!   state. Per block only the per-block entries (slots, retrieved tokens, known prefix) are
//!   projected. Visibility is an additive bias over the ring, staged per block.
//! * **Block pass.** One pass runs the generator's K levels (level inputs pooled and projected,
//!   six layers, tied codebook logits, the argmax code, the codebook gather, upsampling and φ),
//!   the tokenizer decoder's draft with the known prefix forced, the token decoder's check pass
//!   over the draft, and the acceptance: the first rejected position at or after the known
//!   prefix (verified: the autoregressive path's argmax differs from the draft; parallel: the
//!   pass with ẑ disagrees or its top probability is below τ). The host reads back
//!   `[first, any, argmax ids (L), draft (L)]`, `2 + 2L` integers.
//! * **Autoregressive pass.** One token-decoder pass on the autoregressive path over the
//!   block's fixed tokens: argmax ids (and log-probabilities for sampling) of every position.
//! * **Graphs** (CUDA, graphs on): both passes are captured once per mode over the fixed
//!   buffers and replayed; a re-decode replays the block pass with the new known prefix. Every
//!   other device, or a failed capture, runs the same passes eagerly.
//!
//! Scope: temperature 0 (verified or parallel mode) and parallel mode at any temperature on
//! CUDA (codes then sampled on the device, tokens argmax); verified sampling at T > 0 and the
//! test hooks that rewrite drafts or codes stay on the reference loop.

use burn::tensor::{Bool, Device, Int, Tensor, TensorData};

use super::decode::{BlockOut, BlockStats, DecodeOptions, Hooks, HybridSession};
use super::model::{FastMemory, HybridLm, retrieved_tokens};
use super::vq::Resampler;
use crate::layers::Rope;

/// Writes `src` into `dst`: in `dst`'s own buffer on CUDA with the fused kernels on (a graph
/// reads it), by replacing it elsewhere. `axis`/`at`: where along `dst`.
fn put<const D: usize>(dst: &mut Tensor<D>, src: Tensor<D>, axis: usize, at: usize) {
    #[cfg(feature = "cuda")]
    if crate::fused::active(&dst.device()) {
        crate::kernels::assign_into(dst, src, axis, at);
        return;
    }
    let n = src.dims()[axis];
    let mut ranges: Vec<std::ops::Range<usize>> = dst.dims().iter().map(|&d| 0..d).collect();
    ranges[axis] = at..at + n;
    let ranges: [std::ops::Range<usize>; D] = ranges.try_into().expect("rank");
    *dst = dst.clone().slice_assign(ranges, src);
}

/// [`put`] for int tensors.
fn put_int<const D: usize>(dst: &mut Tensor<D, Int>, src: Tensor<D, Int>, axis: usize, at: usize) {
    #[cfg(feature = "cuda")]
    if crate::fused::active(&dst.device()) {
        crate::kernels::assign_int_into(dst, src, axis, at);
        return;
    }
    let n = src.dims()[axis];
    let mut ranges: Vec<std::ops::Range<usize>> = dst.dims().iter().map(|&d| 0..d).collect();
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

/// Rewrites `rope`'s tables with those of `positions` (in place on CUDA, where a graph reads
/// them; replaced elsewhere).
fn restage(rope: &mut Rope, positions: &[usize], head_dim: usize, base: f64, device: &Device) {
    let fresh = Rope::at(positions, head_dim, base, device);
    #[cfg(feature = "cuda")]
    if crate::fused::active(device) {
        let ((cos, sin), (new_cos, new_sin)) = (rope.tables(), fresh.tables());
        crate::kernels::assign_into(&cos, new_cos, 0, 0);
        crate::kernels::assign_into(&sin, new_sin, 0, 0);
        return;
    }
    *rope = fresh;
}

/// The recent encoder states' memory keys and values, per layer, in rings of `recent + L`
/// slots (token t in slot t mod capacity), with the copy keys and the tokens that followed.
struct Rings {
    generator: Vec<(Tensor<4>, Tensor<4>)>,
    dec: Vec<(Tensor<4>, Tensor<4>)>,
    copy_keys: Tensor<3>,
    next: Tensor<2, Int>,
    /// Visibility for the memory attention and for the copy head `[1, capacity]`.
    seq_bias: Tensor<2>,
    copy_bias: Tensor<2>,
    capacity: usize,
    /// Tokens projected so far (token t is in its slot when `t < upto`).
    upto: usize,
}

/// The per-block inputs, in fixed buffers.
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
    extra_rope: Rope,
    gen_q: Rope,
    dec_q: Rope,
    /// The autoregressive pass's input tokens `[1, L]`.
    ar_tokens: Tensor<2, Int>,
}

/// Tables fixed for the session.
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

/// How a block pass judges the draft.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PassMode {
    /// Against the autoregressive path (no ẑ), greedy.
    Verified,
    /// With ẑ, greedy codes.
    Parallel,
    /// With ẑ, codes sampled on the device (CUDA).
    ParallelSampled,
}

/// What a block pass reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassResult {
    /// The first rejected position at or after the known prefix, if any.
    pub rejected: Option<usize>,
    /// The pass's argmax token per position.
    pub ids: Vec<i32>,
    pub draft: Vec<i32>,
}

/// Decode state of one stream at batch 1 on the fast path; the reference session underneath
/// keeps the tokens, the encoder and backbone caches and the retrieval history.
pub struct FastSession {
    pub session: HybridSession,
    rings: Rings,
    inputs: Inputs,
    consts: Consts,
    /// The device code samplers, one per level (CUDA).
    #[cfg(feature = "cuda")]
    samplers: Vec<crate::sample::SampleParams>,
    #[cfg(feature = "cuda")]
    graphs: Graphs,
    device: Device,
}

/// The captured passes, per mode (CUDA).
#[cfg(feature = "cuda")]
#[derive(Default)]
struct Graphs {
    block: Vec<(PassMode, Result<BlockGraph, String>)>,
    ar: Option<Result<ArGraph, String>>,
}

/// A captured block pass and its output buffer.
#[cfg(feature = "cuda")]
type BlockGraph = (crate::graph::Captured, Tensor<2, Int>);
/// A captured autoregressive pass and its outputs (ids, log-probabilities).
#[cfg(feature = "cuda")]
type ArGraph = (crate::graph::Captured, (Tensor<2, Int>, Tensor<3>));

impl FastSession {
    pub fn new(model: &HybridLm, max_tokens: usize, device: &Device) -> Self {
        let h = model.config().clone();
        let (l, d, hd) = (h.block, model.local_dim, model.head_dim);
        let capacity = h.recent + l;
        let ring = |kvh: usize| {
            (
                Tensor::<4>::zeros([1, kvh, capacity, hd], device),
                Tensor::<4>::zeros([1, kvh, capacity, hd], device),
            )
        };
        let generator = model.generator_cross().iter().map(|c| ring(c.kv_heads())).collect();
        let dec = model.decoder_cross().iter().map(|c| ring(c.kv_heads())).collect();
        let neg = vec![f32::NEG_INFINITY; capacity];
        let rings = Rings {
            generator,
            dec,
            copy_keys: Tensor::zeros([1, capacity, h.copy_dim], device),
            next: Tensor::zeros([1, capacity], device),
            seq_bias: floats(neg.clone(), [1, capacity], device),
            copy_bias: floats(neg, [1, capacity], device),
            capacity,
            upto: 0,
        };
        let rt = h.retrieved_blocks * l;
        let e = h.slots + rt + (l - 1);
        let zeros_pos = vec![0usize; e];
        let inputs = Inputs {
            cond: Tensor::zeros([1, 1, model.start_condition(1).dims()[2]], device),
            retrieved: Tensor::zeros([1, rt.max(1)], device),
            retrieved_next: Tensor::zeros([1, rt.max(1)], device),
            retrieved_bias: floats(vec![f32::NEG_INFINITY; rt.max(1)], [1, rt.max(1)], device),
            known_states: Tensor::zeros([1, l - 1, d], device),
            known_tokens: Tensor::zeros([1, l], device),
            p: Tensor::zeros([1, 1], device),
            extra_rope: Rope::at(&zeros_pos, hd, model.rope_base, device),
            gen_q: Rope::at(&vec![0; l], hd, model.rope_base, device),
            dec_q: Rope::at(&(0..l).collect::<Vec<_>>(), hd, model.rope_base, device),
            ar_tokens: Tensor::zeros([1, l], device),
        };
        let (dec_rope, causal) = model.decoder_self(device);
        let pos: Vec<i32> = (0..l as i32).collect();
        let consts = Consts {
            resampler: model.tokenizer().resampler(device),
            level_ropes: h
                .levels
                .iter()
                .map(|&n| Rope::at(&(0..n).collect::<Vec<_>>(), hd, model.rope_base, device))
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
        Self {
            session: HybridSession::new(model, max_tokens, device),
            rings,
            inputs,
            consts,
            #[cfg(feature = "cuda")]
            samplers: (0..h.depth()).map(|_| crate::sample::SampleParams::new(device)).collect(),
            #[cfg(feature = "cuda")]
            graphs: Graphs::default(),
            device: device.clone(),
        }
    }

    /// Appends prompt tokens ([`HybridSession::prefill`]).
    pub fn prefill(&mut self, model: &HybridLm, tokens: &[i32]) {
        self.session.prefill(model, tokens);
    }

    pub fn tokens(&self) -> &[i32] {
        self.session.tokens()
    }

    // ------------------------------------------------------------------ rings and inputs

    /// Projects the encoded states not yet in the rings (those still held) into their slots.
    fn sync_rings(&mut self, model: &HybridLm) {
        let s = &self.session;
        let from = self.rings.upto.max(s.states_start);
        let to = s.encoded;
        let cap = self.rings.capacity;
        // Only the last `capacity` tokens survive in the ring.
        let from = from.max(to.saturating_sub(cap));
        if to <= from {
            self.rings.upto = self.rings.upto.max(to);
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
                put(&mut ring.0, kv.0.clone().narrow(2, off, len), 2, slot);
                put(&mut ring.1, kv.1.clone().narrow(2, off, len), 2, slot);
            }
        };
        for (ring, cross) in self.rings.generator.iter_mut().zip(model.generator_cross()) {
            write(ring, cross.project_rows(states.clone(), &rope));
        }
        for (ring, cross) in self.rings.dec.iter_mut().zip(model.decoder_cross()) {
            write(ring, cross.project_rows(states.clone(), &rope));
        }
        for &(off, slot, len) in &pieces {
            put(&mut self.rings.copy_keys, copy.clone().narrow(1, off, len), 1, slot);
        }
        self.rings.upto = to;
    }

    /// Stages the open block's inputs: ring visibility, the condition, the retrieved tokens,
    /// the known prefix `fixed`, and the tables of the block's positions.
    fn stage(&mut self, model: &HybridLm, fixed: &[i32]) {
        self.sync_rings(model);
        let h = model.config();
        let (l, hd, base) = (h.block, model.head_dim, model.rope_base);
        let device = self.device.clone();
        let start = self.session.blocks() * l;
        let cap = self.rings.capacity;
        let tokens = self.session.tokens().to_vec();
        // Ring visibility: slot s holds the latest projected token t = s mod cap.
        let upto = self.rings.upto;
        let (mut seq, mut copy, mut next) =
            (vec![f32::NEG_INFINITY; cap], vec![f32::NEG_INFINITY; cap], vec![0i32; cap]);
        for t in upto.saturating_sub(cap)..upto {
            let slot = t % cap;
            let visible = t < start && t + h.recent >= start;
            seq[slot] = if visible { 0.0 } else { f32::NEG_INFINITY };
            copy[slot] = if visible && t + 1 < start { 0.0 } else { f32::NEG_INFINITY };
            next[slot] = tokens.get(t + 1).copied().unwrap_or(0);
        }
        put(&mut self.rings.seq_bias, floats(seq, [1, cap], &device), 1, 0);
        put(&mut self.rings.copy_bias, floats(copy, [1, cap], &device), 1, 0);
        put_int(&mut self.rings.next, ints(next, [1, cap], &device), 1, 0);
        // The condition and the retrieved blocks.
        let cond = self.session.condition();
        let cond_host: Vec<f32> =
            cond.clone().into_data().convert::<f32>().try_to_vec::<f32>().expect("f32");
        put(&mut self.inputs.cond, cond, 2, 0);
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
            put_int(&mut self.inputs.retrieved, ints(ids, [1, rt], &device), 1, 0);
            put_int(&mut self.inputs.retrieved_next, ints(following, [1, rt], &device), 1, 0);
            put(&mut self.inputs.retrieved_bias, floats(bias, [1, rt], &device), 1, 0);
        }
        self.stage_known(model, fixed);
        let e_pos: Vec<usize> =
            std::iter::repeat_n(start, h.slots + rt).chain((0..l - 1).map(|i| start + i)).collect();
        restage(&mut self.inputs.extra_rope, &e_pos, hd, base, &device);
        restage(&mut self.inputs.gen_q, &vec![start; l], hd, base, &device);
        restage(&mut self.inputs.dec_q, &(start..start + l).collect::<Vec<_>>(), hd, base, &device);
    }

    /// Stages the known prefix `fixed`: its states, tokens and length.
    fn stage_known(&mut self, model: &HybridLm, fixed: &[i32]) {
        let l = model.config().block;
        let device = self.device.clone();
        let states = self.session.known_states(model, fixed);
        put(&mut self.inputs.known_states, states, 1, 0);
        let mut known = fixed.to_vec();
        known.resize(l, 0);
        put_int(&mut self.inputs.known_tokens, ints(known, [1, l], &device), 1, 0);
        put_int(&mut self.inputs.p, ints(vec![fixed.len() as i32], [1, 1], &device), 1, 0);
    }

    // ------------------------------------------------------------------ the passes

    /// The block pass (module docs): `[1, 2 + 2L]` = first rejected position, whether any,
    /// the argmax ids, the draft. Only device buffers: capturable.
    fn block_pass(&self, model: &HybridLm, mode: PassMode) -> Tensor<2, Int> {
        let h = model.config();
        let (l, rt) = (h.block, h.retrieved_blocks * h.block);
        let (i, c, r) = (&self.inputs, &self.consts, &self.rings);
        let slots = model.slot_vectors(i.cond.clone());
        let retrieved = (rt > 0).then(|| model.retrieved_vectors(i.retrieved.clone()));
        let known = model.known_vectors(i.known_states.clone());
        let p_known = i.p.clone().expand([1, l - 1]);
        let known_bias = Tensor::<2>::zeros([1, l - 1], &self.device)
            .mask_fill(c.known_pos.clone().greater_equal(p_known), f32::NEG_INFINITY);
        let mut shared = vec![slots];
        let mut shared_bias = vec![c.slots_bias.clone()];
        if let Some(v) = &retrieved {
            shared.push(v.clone());
            shared_bias.push(i.retrieved_bias.clone());
        }
        let n_shared = shared.iter().map(|t| t.dims()[1]).sum::<usize>();
        let dec_extra = Tensor::cat(shared.clone(), 1);
        let dec_bias = Tensor::cat(shared_bias.clone(), 1);
        shared.push(known);
        shared_bias.push(known_bias);
        let gen_extra = Tensor::cat(shared, 1);
        let gen_bias = Tensor::cat(shared_bias, 1);
        let gen_rope = &i.extra_rope;
        let dec_rope_rows = i.extra_rope.rows(0, n_shared);
        let gen_kv: Vec<_> = model
            .generator_cross()
            .iter()
            .map(|x| x.project_rows(gen_extra.clone(), gen_rope))
            .collect();
        let dec_kv: Vec<_> = model
            .decoder_cross()
            .iter()
            .map(|x| x.project_rows(dec_extra.clone(), &dec_rope_rows))
            .collect();
        // The generator, level by level.
        let tok = model.tokenizer();
        let mut zhat: Option<Tensor<3>> = None;
        for (k, &n) in h.levels.iter().enumerate() {
            let q_rope = i.gen_q.rows(0, n);
            let memory = FastMemory {
                seq: Some(&r.generator),
                seq_bias: &r.seq_bias,
                extra: &gen_kv,
                extra_bias: &gen_bias,
                q_rope: &q_rope,
            };
            let x = model.level_input_with(k, zhat.clone(), &c.resampler);
            let out = model.generator_fast(x, &c.level_ropes[k], &memory);
            let logits = tok.code_logits_with(k, out, &c.resampler);
            let codes = self.pick_codes(k, logits, mode);
            let contribution = tok.contribution_with(k, codes, &c.resampler);
            zhat = Some(match zhat {
                Some(z) => z + contribution,
                None => contribution,
            });
        }
        let zhat = zhat.expect("at least one level");
        // The draft, the known prefix forced.
        let draft = crate::fused::argmax_last(tok.decode_with(
            zhat.clone(),
            model.embedding(),
            &c.tok_rope,
        ))
        .reshape([1, l]);
        let p = i.p.clone().expand([1, l]);
        let draft = draft.mask_where(c.pos.clone().lower(p.clone()), i.known_tokens.clone());
        // The check pass and the acceptance.
        let verified = mode == PassMode::Verified;
        let x = model.decoder_input_fast(draft.clone(), (!verified).then_some(zhat));
        let logp = self.decoder_pass(model, x, &dec_kv, &dec_bias, retrieved);
        let ids = crate::fused::argmax_last(logp.clone()).reshape([1, l]);
        let mut reject = ids.clone().not_equal(draft.clone());
        if !verified {
            let low = logp.max_dim(2).reshape([1, l]).lower(c.ln_tau.clone().expand([1, l]));
            reject = reject.bool_or(low);
        }
        let reject = reject.bool_and(c.pos.clone().greater_equal(p));
        let score = reject.int() * c.rank.clone();
        let first = score.clone().argmax(1).reshape([1, 1]);
        let any = score.max_dim(1).reshape([1, 1]);
        Tensor::cat(vec![first, any, ids, draft], 1)
    }

    /// The token decoder over inputs `x` with the block's decoder memory.
    fn decoder_pass(
        &self,
        model: &HybridLm,
        x: Tensor<3>,
        dec_kv: &[(Tensor<4>, Tensor<4>)],
        dec_bias: &Tensor<2>,
        retrieved: Option<Tensor<3>>,
    ) -> Tensor<3> {
        let (i, c, r) = (&self.inputs, &self.consts, &self.rings);
        let memory = FastMemory {
            seq: Some(&r.dec),
            seq_bias: &r.seq_bias,
            extra: dec_kv,
            extra_bias: dec_bias,
            q_rope: &i.dec_q,
        };
        let copy_seq = Some((&r.copy_keys, &r.next, &r.copy_bias));
        let copy_ret = retrieved.map(|v| (v, &i.retrieved_next, &i.retrieved_bias));
        model.decoder_fast(x, &c.dec_rope, c.causal.clone(), &memory, (copy_seq, copy_ret))
    }

    /// The autoregressive pass over `ar_tokens`: argmax ids `[1, L]` and log-probabilities.
    fn ar_pass(&self, model: &HybridLm) -> (Tensor<2, Int>, Tensor<3>) {
        let h = model.config();
        let (l, rt) = (h.block, h.retrieved_blocks * h.block);
        let i = &self.inputs;
        let slots = model.slot_vectors(i.cond.clone());
        let retrieved = (rt > 0).then(|| model.retrieved_vectors(i.retrieved.clone()));
        let mut shared = vec![slots];
        let mut bias = vec![self.consts.slots_bias.clone()];
        if let Some(v) = &retrieved {
            shared.push(v.clone());
            bias.push(i.retrieved_bias.clone());
        }
        let n = shared.iter().map(|t| t.dims()[1]).sum::<usize>();
        let extra = Tensor::cat(shared, 1);
        let bias = Tensor::cat(bias, 1);
        let rope = i.extra_rope.rows(0, n);
        let kv: Vec<_> =
            model.decoder_cross().iter().map(|x| x.project_rows(extra.clone(), &rope)).collect();
        let x = model.decoder_input_fast(i.ar_tokens.clone(), None);
        let logp = self.decoder_pass(model, x, &kv, &bias, retrieved);
        let ids = crate::fused::argmax_last(logp.clone()).reshape([1, l]);
        (ids, logp)
    }

    /// Level `k`'s codes `[1, ℓ]` from its logits: the argmax, or a device draw.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn pick_codes(&self, k: usize, logits: Tensor<3>, mode: PassMode) -> Tensor<2, Int> {
        let [_, n, e] = logits.dims();
        #[cfg(feature = "cuda")]
        if mode == PassMode::ParallelSampled {
            return crate::sample::sample(logits.reshape([n, e]), &self.samplers[k], 0)
                .reshape([1, n]);
        }
        crate::fused::argmax_last(logits).reshape([1, n])
    }

    /// Runs the block pass (replaying its graph when it can) and reads its result.
    fn run_block(&mut self, model: &HybridLm, mode: PassMode) -> PassResult {
        let l = model.config().block;
        let out = self.block_output(model, mode);
        let v: Vec<i32> = out.into_data().convert::<i32>().try_to_vec::<i32>().expect("ints");
        let rejected = (v[1] > 0).then_some(v[0] as usize);
        PassResult { rejected, ids: v[2..2 + l].to_vec(), draft: v[2 + l..2 + 2 * l].to_vec() }
    }

    fn block_output(&mut self, model: &HybridLm, mode: PassMode) -> Tensor<2, Int> {
        #[cfg(feature = "cuda")]
        if crate::fused::graphs_active(&self.device) {
            if !self.graphs.block.iter().any(|(m, _)| *m == mode) {
                let captured = {
                    let this = &*self;
                    crate::graph::record(
                        &self.device,
                        &mut (),
                        2,
                        |_| this.block_pass(model, mode),
                        |_| {},
                    )
                };
                self.graphs.block.push((mode, captured));
            }
            if let Some((_, Ok((graph, out)))) = self.graphs.block.iter().find(|(m, _)| *m == mode)
            {
                // Safety: the graph reads the session's fixed buffers, the rings and the
                // model's weights, all alive (the replay checks its thread).
                unsafe { crate::graph::replay(&self.device, graph, "hybrid block") };
                return out.clone();
            }
        }
        self.block_pass(model, mode)
    }

    fn ar_output(&mut self, model: &HybridLm) -> (Tensor<2, Int>, Tensor<3>) {
        #[cfg(feature = "cuda")]
        if crate::fused::graphs_active(&self.device) {
            if self.graphs.ar.is_none() {
                let captured = {
                    let this = &*self;
                    crate::graph::record(&self.device, &mut (), 2, |_| this.ar_pass(model), |_| {})
                };
                self.graphs.ar = Some(captured);
            }
            if let Some(Ok((graph, out))) = &self.graphs.ar {
                // Safety: as for the block graph.
                unsafe { crate::graph::replay(&self.device, graph, "hybrid ar") };
                return out.clone();
            }
        }
        self.ar_pass(model)
    }

    /// Why a pass runs eagerly, if a capture failed.
    pub fn failure(&self) -> Option<String> {
        #[cfg(feature = "cuda")]
        {
            let block = self.graphs.block.iter().find_map(|(_, g)| g.as_ref().err().cloned());
            let ar = self.graphs.ar.as_ref().and_then(|g| g.as_ref().err().cloned());
            block.or(ar)
        }
        #[cfg(not(feature = "cuda"))]
        None
    }

    // ------------------------------------------------------------------ the loop

    /// Whether the fast path serves `opts` (else [`Self::decode_block`] takes the reference
    /// loop).
    pub fn serves(&self, opts: &DecodeOptions) -> bool {
        let sampled = opts.temperature > 0.0;
        let device_sampling = cfg!(feature = "cuda") && crate::fused::active(&self.device);
        !sampled || (!opts.verified && !opts.ar_only && device_sampling)
    }

    /// Decodes the open block, at most `limit` tokens, as [`HybridSession::decode_block`] does
    /// (same steps, same results at temperature 0 up to rounding), on the fast passes. Only the
    /// `accept` and `ar_rows` hooks apply here; options the fast path does not serve take the
    /// reference loop.
    pub fn decode_block(
        &mut self,
        model: &HybridLm,
        limit: usize,
        opts: &DecodeOptions,
        hooks: &mut Hooks,
    ) -> BlockOut {
        if !self.serves(opts) || hooks.draft.is_some() || hooks.codes.is_some() {
            let out = self.session.decode_block(model, limit, opts, hooks);
            return out;
        }
        let h = model.config().clone();
        let l = h.block;
        let start = self.session.blocks() * l;
        let mut fixed: Vec<i32> = self.session.tokens()[start..].to_vec();
        let known = fixed.len();
        let redecode_min = opts.redecode_min.unwrap_or(h.redecode_min);
        let mut stats = BlockStats { known, ..BlockStats::default() };
        let mut verified = opts.verified || self.session.verified_span;
        let (mut stopped, mut close_span) = (false, false);
        self.stage(model, &fixed);
        let mut passes = 0;
        let mut redecode = false;
        let budget = |fixed: &Vec<i32>| (known + limit).saturating_sub(fixed.len());
        #[cfg(feature = "cuda")]
        let sampled = opts.temperature > 0.0;
        while !opts.ar_only && fixed.len() < l && !stopped {
            if redecode {
                self.stage_known(model, &fixed);
            }
            #[cfg(feature = "cuda")]
            if sampled {
                for (k, s) in self.samplers.iter().enumerate() {
                    let t = level_temperature(opts.temperature, k, h.depth());
                    let counter =
                        (self.session.rng as u32).wrapping_add((passes * 1000 + k * 100) as u32);
                    s.set(crate::sample::SampleSettings {
                        temperature: t as f32,
                        top_k: 0,
                        top_p: opts.top_p as f32,
                        seed: opts.seed,
                        counter,
                    });
                }
            }
            let mode = match (verified, opts.temperature > 0.0) {
                (true, _) => PassMode::Verified,
                (false, false) => PassMode::Parallel,
                (false, true) => PassMode::ParallelSampled,
            };
            let result = self.run_block(model, mode);
            stats.generator_runs += 1;
            stats.check_passes += 1;
            let p = fixed.len();
            let mut cut = match hooks.accept.as_mut() {
                Some(f) => (p..l).find(|&i| !f(passes, i)),
                None => result.rejected,
            };
            if cut.is_some_and(|c| c < p) {
                cut = Some(p);
            }
            let accepted = cut.unwrap_or(l) - p;
            match passes {
                0 => stats.first_accepted = Some(accepted),
                1 => stats.second_accepted = Some(accepted),
                _ => {}
            }
            passes += 1;
            let mut emitted: Vec<i32> = result.draft[p..p + accepted].to_vec();
            if let Some(c) = cut {
                emitted.push(result.ids[c]);
            }
            emitted.truncate(budget(&fixed));
            let mut keep = emitted.len();
            if let Some(s) = emitted.iter().position(|t| opts.stop.contains(t)) {
                keep = s + 1;
                stopped = true;
            }
            let switch = (!verified)
                .then(|| emitted[..keep].iter().position(|t| opts.switch_on.contains(t)))
                .flatten();
            if let Some(s) = switch {
                keep = s + 1;
                stopped = false;
            }
            emitted.truncate(keep);
            if emitted.iter().any(|t| opts.switch_off.contains(t)) {
                close_span = true;
            }
            fixed.extend_from_slice(&emitted);
            if fixed.len() >= known + limit {
                stopped = true;
            }
            if stopped || fixed.len() >= l {
                break;
            }
            if let Some(s) = switch {
                stats.switched = Some(p + s);
                verified = true;
                self.session.verified_span = true;
                // One more pass over the block in verified mode (a fresh draft: TODO reuse).
                redecode = true;
                continue;
            }
            if cut.is_none() {
                continue;
            }
            if !stats.redecoded && l - fixed.len() >= redecode_min {
                stats.redecoded = true;
                redecode = true;
                continue;
            }
            break;
        }
        // The autoregressive fill.
        while fixed.len() < l && !stopped {
            let i = fixed.len();
            let mut inputs = fixed.clone();
            inputs.resize(l, 0);
            let device = self.device.clone();
            put_int(&mut self.inputs.ar_tokens, ints(inputs, [1, l], &device), 1, 0);
            let (ids, logp) = self.ar_output(model);
            let token = if opts.temperature > 0.0 || hooks.ar_rows.is_some() {
                let row: Vec<f32> = logp
                    .narrow(1, i, 1)
                    .into_data()
                    .convert::<f32>()
                    .try_to_vec::<f32>()
                    .expect("f32");
                if let Some(f) = hooks.ar_rows.as_mut() {
                    f(&fixed, i, &row);
                }
                if opts.temperature > 0.0 {
                    let mut uniform = || self.session.uniform();
                    super::decode::sample_row(&row, opts.temperature, opts.top_p, &mut uniform)
                        as i32
                } else {
                    super::decode::argmax_row(&row) as i32
                }
            } else {
                let v: Vec<i32> = ids
                    .narrow(1, i, 1)
                    .into_data()
                    .convert::<i32>()
                    .try_to_vec::<i32>()
                    .expect("ints");
                v[0]
            };
            stats.ar_tokens += 1;
            fixed.push(token);
            if opts.switch_on.contains(&token) {
                self.session.verified_span = true;
            }
            if opts.switch_off.contains(&token) {
                close_span = true;
            }
            if opts.stop.contains(&token) || fixed.len() >= known + limit {
                stopped = true;
            }
        }
        let emitted = fixed[known..].to_vec();
        self.session.close_block(model, start, &emitted, stopped, close_span);
        BlockOut { tokens: emitted, stopped, stats }
    }

    /// Decodes up to `max_tokens` tokens of a reply ([`HybridSession::generate_reply`]).
    pub fn generate_reply(
        &mut self,
        model: &HybridLm,
        max_tokens: usize,
        opts: &DecodeOptions,
        hooks: &mut Hooks,
    ) -> (Vec<i32>, Vec<BlockStats>) {
        self.session.rng = opts.seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1;
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
