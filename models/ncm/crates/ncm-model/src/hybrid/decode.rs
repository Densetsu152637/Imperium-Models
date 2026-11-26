//! The hybrid's block decode loop, an eager reference at batch 1 ([`HybridSession`]).
//!
//! Per block (design: "Decoding one block" and "Exactness"):
//!
//! 1. The backbone step of the last complete block has run (at its close); the memory is built
//!    from the condition, the recent encoder states before the block, the retrieved blocks and
//!    (generator only) the known prefix: the block's tokens already fixed, from the prompt or
//!    from an earlier pass over the block.
//! 2. The generator runs K passes coarse to fine, sampling each level's codes; their
//!    reconstruction ẑ goes through the tokenizer decoder into a draft (argmax), with the known
//!    prefix forced.
//! 3. One check pass of the token decoder over the block, input the shifted draft: in
//!    **verified** mode on the autoregressive path (no ẑ; a draft token is accepted when it is
//!    the path's argmax at temperature 0, by the speculative-sampling rule above 0), in
//!    **parallel** mode with ẑ (accepted while the pass agrees with the draft and its top
//!    probability is at least τ). The first rejected position emits the pass's own token.
//! 4. If at least `redecode_min` positions remain, one re-decode: steps 2 and 3 with everything
//!    fixed so far as the known prefix. Otherwise, or after it, the rest of the block is filled on
//!    the autoregressive path one token per pass.
//! 5. A switch token (tool call or code-fence opener) in a parallel-mode result ends parallel
//!    mode at it: the rest of the block runs in verified mode (one extra check pass on the same
//!    draft), and so do later blocks until a switch-off token's block ends.
//! 6. Stop tokens and the token limit truncate the block: the block stays open (no backbone
//!    step) and the next prefill fills it. A complete block is encoded, pooled and stepped.
//!
//! The autoregressive path reads only the memory built before the block and the block's tokens
//! before each position, through passes of one fixed shape (the whole block, causal), so its
//! output at a position depends on nothing else bit for bit: verified output at temperature 0 is
//! its greedy output whatever the codes, ẑ, drafts or re-decodes.
//!
//! Every pass recomputes the memory's keys and the whole block (no incremental caches inside the
//! block); fused kernels, ring-projected memory keys and CUDA graphs come later (TODO).

use burn::tensor::{Device, Int, Tensor, TensorData};

use super::memory::{LayerKv, Memory};
use super::model::{Entries, HybridLm, PassShape, retrieved_tokens};
use crate::cache::{LayerCache, LayerSnapshot};
use crate::layers::Rope;
use crate::ncm::ENCODER_CHUNK;

// The block types are backend-neutral and live in `ncm-exec`; re-exported here and from
// [`crate::hybrid`] so existing paths keep working.
pub use ncm_exec::hybrid::{
    BlockOut, BlockStats, DecodeOptions, Hooks, Pick, Rewrite, RowObserver, Scan, scan_cut,
};

/// A point of a [`HybridSession`] to return to ([`HybridSession::snapshot`]).
pub struct HybridSnapshot {
    tokens: Vec<i32>,
    encoder: Vec<LayerSnapshot>,
    backbone: Vec<LayerSnapshot>,
    states: Option<Tensor<3>>,
    states_start: usize,
    encoded: usize,
    pooled: Vec<Vec<f32>>,
    cond: Tensor<3>,
    blocks: usize,
}

impl HybridSnapshot {
    /// The tokens the session held.
    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }
}

/// Decode state of one stream at batch 1: the tokens, the encoder and backbone caches, the
/// encoder states of the recent tokens and the open block, the backbone inputs of every block
/// (for retrieval) and the condition of the open block.
pub struct HybridSession {
    pub(crate) tokens: Vec<i32>,
    encoder: Vec<LayerCache>,
    backbone: Vec<LayerCache>,
    /// Encoder states `[1, n, d]` of tokens `states_start..tokens.len()` minus any not yet
    /// encoded (see `encoded`).
    states: Option<Tensor<3>>,
    pub(crate) states_start: usize,
    /// Tokens the encoder has consumed.
    pub(crate) encoded: usize,
    pub(crate) pooled: Vec<Vec<f32>>,
    cond: Tensor<3>,
    blocks: usize,
    pub(crate) rng: u64,
    /// Verified mode carried across blocks after a switch token; ends with the block in which a
    /// switch-off token appears.
    pub(crate) verified_span: bool,
    device: Device,
}

impl HybridSession {
    /// An empty stream with caches for up to `max_tokens`.
    pub fn new(model: &HybridLm, max_tokens: usize, device: &Device) -> Self {
        let l = model.config().block;
        let backbone = model.backbone().make_caches(max_tokens / l + 1, 1, device);
        Self::with_backbone(model, max_tokens, backbone, device)
    }

    /// `slots` sessions for a batcher: their backbone caches are rows of shared buffers on
    /// CUDA ([`crate::layers::Stack::make_slot_caches`]), so a batched backbone step over
    /// closing slots decodes each retrieval layer in one set of launches.
    pub fn slots(model: &HybridLm, slots: usize, max_tokens: usize, device: &Device) -> Vec<Self> {
        let l = model.config().block;
        model
            .backbone()
            .make_slot_caches(slots, max_tokens / l + 1, device)
            .into_iter()
            .map(|backbone| Self::with_backbone(model, max_tokens, backbone, device))
            .collect()
    }

    fn with_backbone(
        model: &HybridLm,
        max_tokens: usize,
        backbone: Vec<LayerCache>,
        device: &Device,
    ) -> Self {
        Self {
            tokens: Vec::new(),
            encoder: model.encoder().make_caches(max_tokens, 1, device),
            backbone,
            states: None,
            states_start: 0,
            encoded: 0,
            pooled: Vec::new(),
            cond: model.start_condition(1),
            blocks: 0,
            rng: 0x9E37_79B9_7F4A_7C15,
            verified_span: false,
            device: device.clone(),
        }
    }

    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    /// What [`Self::restore`] needs to return to this point: the tokens, the encoder and
    /// backbone caches' snapshots, the held encoder states, the retrieval history and the open
    /// block's condition. Taken at any point (a block boundary or inside the open block).
    pub fn snapshot(&self) -> HybridSnapshot {
        HybridSnapshot {
            tokens: self.tokens.clone(),
            encoder: self.encoder.iter().map(LayerCache::snapshot).collect(),
            backbone: self.backbone.iter().map(LayerCache::snapshot).collect(),
            states: self.states.clone(),
            states_start: self.states_start,
            encoded: self.encoded,
            pooled: self.pooled.clone(),
            cond: self.cond.clone(),
            blocks: self.blocks,
        }
    }

    /// Back to a [`Self::snapshot`] of this session (taken since its caches last moved past
    /// it on another path: append-only caches only rewind), in the caches' own buffers on CUDA.
    pub fn restore(&mut self, snapshot: &HybridSnapshot) {
        self.tokens = snapshot.tokens.clone();
        for (c, s) in self.encoder.iter_mut().zip(&snapshot.encoder) {
            c.restore(s);
        }
        for (c, s) in self.backbone.iter_mut().zip(&snapshot.backbone) {
            c.restore(s);
        }
        self.states = snapshot.states.clone();
        self.states_start = snapshot.states_start;
        self.encoded = snapshot.encoded;
        self.pooled = snapshot.pooled.clone();
        self.cond = snapshot.cond.clone();
        self.blocks = snapshot.blocks;
        self.verified_span = false;
    }

    /// This session made a copy of `src` (a session of the same model and size: a request
    /// forked from another slot's prefill), in its own caches' buffers on CUDA.
    pub fn copy_from(&mut self, src: &HybridSession) {
        self.tokens = src.tokens.clone();
        for (c, s) in self.encoder.iter_mut().zip(&src.encoder) {
            c.copy_from(s);
        }
        for (c, s) in self.backbone.iter_mut().zip(&src.backbone) {
            c.copy_from(s);
        }
        self.states = src.states.clone();
        self.states_start = src.states_start;
        self.encoded = src.encoded;
        self.pooled = src.pooled.clone();
        self.cond = src.cond.clone();
        self.blocks = src.blocks;
        self.verified_span = false;
    }

    /// Appends `tokens` `[B, len]` (row i to `sessions[i]`, all of them encoded up to their
    /// ends) with one batched encoder pass over the rows ([`crate::layers::Stack::forward_slots`]),
    /// then steps each session's backbone over the blocks its row completes.
    pub fn prefill_slots(
        model: &HybridLm,
        sessions: &mut [&mut HybridSession],
        tokens: Tensor<2, Int>,
    ) {
        let [b, len] = tokens.dims();
        assert_eq!(b, sessions.len(), "one row per session");
        let host: Vec<i32> =
            tokens.clone().into_data().convert::<i32>().try_to_vec::<i32>().expect("ints");
        for (s, row) in sessions.iter_mut().zip(host.chunks(len)) {
            assert_eq!(s.encoded, s.tokens.len(), "sessions encoded up to their ends");
            s.tokens.extend_from_slice(row);
        }
        let x = model.embed(tokens);
        let states = {
            let mut caches: Vec<&mut [LayerCache]> =
                sessions.iter_mut().map(|s| s.encoder.as_mut_slice()).collect();
            model.encoder().forward_slots(x, &mut caches)
        };
        for (i, s) in sessions.iter_mut().enumerate() {
            let row = states.clone().narrow(0, i, 1);
            s.states = Some(match s.states.take() {
                Some(old) => Tensor::cat(vec![old, row], 1),
                None => row,
            });
            s.encoded += len;
            s.step_complete_blocks(model);
        }
    }

    /// [`Self::prefill_slots`] for pieces of different lengths (varlen): `pieces[i]` (each
    /// non-empty, at most [`crate::ncm::ENCODER_CHUNK`] tokens) continues `sessions[i]`, all
    /// encoded in one packed encoder pass ([`crate::layers::Stack::forward_packed`]: projections
    /// and MLPs once over every token, the mixers per piece, nothing padded).
    pub fn prefill_packed(
        model: &HybridLm,
        sessions: &mut [&mut HybridSession],
        pieces: &[&[i32]],
    ) {
        assert_eq!(pieces.len(), sessions.len(), "one piece per session");
        assert!(pieces.iter().all(|p| !p.is_empty()), "no empty piece");
        let lens: Vec<usize> = pieces.iter().map(|p| p.len()).collect();
        let ids: Vec<i32> = pieces.concat();
        let total = ids.len();
        let device = sessions[0].device.clone();
        for (s, piece) in sessions.iter_mut().zip(pieces) {
            assert_eq!(s.encoded, s.tokens.len(), "sessions encoded up to their ends");
            s.tokens.extend_from_slice(piece);
        }
        let tokens =
            Tensor::<2, Int>::from_data(burn::tensor::TensorData::new(ids, [1, total]), &device);
        let x = model.embed(tokens);
        let states = {
            let mut caches: Vec<&mut [LayerCache]> =
                sessions.iter_mut().map(|s| s.encoder.as_mut_slice()).collect();
            model.encoder().forward_packed(x, &lens, &mut caches)
        };
        let mut at = 0;
        for (s, &len) in sessions.iter_mut().zip(&lens) {
            let piece = states.clone().narrow(1, at, len);
            at += len;
            s.states = Some(match s.states.take() {
                Some(old) => Tensor::cat(vec![old, piece], 1),
                None => piece,
            });
            s.encoded += len;
            s.step_complete_blocks(model);
        }
    }

    /// Back to an empty stream, in the caches' own buffers on CUDA (a server reuses one
    /// session, and the graphs captured against it, across requests).
    pub fn reset(&mut self, model: &HybridLm) {
        self.tokens.clear();
        self.encoder.iter_mut().for_each(LayerCache::reset);
        self.backbone.iter_mut().for_each(LayerCache::reset);
        self.states = None;
        self.states_start = 0;
        self.encoded = 0;
        self.pooled.clear();
        self.cond = model.start_condition(1);
        self.blocks = 0;
        self.verified_span = false;
    }

    /// Complete blocks (backbone steps run).
    pub fn blocks(&self) -> usize {
        self.blocks
    }

    /// The open block's condition `[1, 1, bb]`.
    pub fn condition(&self) -> Tensor<3> {
        self.cond.clone()
    }

    /// The encoder states of tokens `from..to` `[1, to - from, d]` (all already encoded).
    pub fn states(&self, from: usize, to: usize) -> Option<Tensor<3>> {
        assert!(from >= self.states_start && to <= self.encoded, "states {from}..{to} not held");
        (to > from).then(|| {
            let s = self.states.clone().expect("states");
            s.narrow(1, from - self.states_start, to - from)
        })
    }

    /// Appends prompt tokens: encodes them, and runs the backbone over every block they
    /// complete. A block left incomplete stays open for the next prefill or decode.
    pub fn prefill(&mut self, model: &HybridLm, tokens: &[i32]) {
        self.tokens.extend_from_slice(tokens);
        self.encode_pending(model);
        self.step_complete_blocks(model);
    }

    /// Encodes the tokens not yet encoded, in pieces of [`ENCODER_CHUNK`].
    pub(crate) fn encode_pending(&mut self, model: &HybridLm) {
        while self.encoded < self.tokens.len() {
            let len = ENCODER_CHUNK.min(self.tokens.len() - self.encoded);
            let ids = self.tokens[self.encoded..self.encoded + len].to_vec();
            let x = model.embed(ids_tensor(&ids, &self.device));
            let s = model.encoder().forward(x, Some(&mut self.encoder));
            self.states = Some(match self.states.take() {
                Some(old) => Tensor::cat(vec![old, s], 1),
                None => s,
            });
            self.encoded += len;
        }
    }

    /// Backbone steps for every complete block, then drops states older than the next block's
    /// recent window.
    pub(crate) fn step_complete_blocks(&mut self, model: &HybridLm) {
        let l = model.config().block;
        let complete = self.encoded / l;
        if complete > self.blocks {
            let s = self.states(self.blocks * l, complete * l).expect("block states");
            let pooled = model.pool_blocks(s);
            let h = model.backbone().forward(pooled.clone(), Some(&mut self.backbone));
            let n = complete - self.blocks;
            self.cond = h.narrow(1, n - 1, 1);
            let host: Vec<f32> =
                pooled.into_data().convert::<f32>().try_to_vec::<f32>().expect("f32");
            let w = host.len() / n;
            self.pooled.extend(host.chunks(w).map(<[f32]>::to_vec));
            self.blocks = complete;
        }
        let keep_from = (self.blocks * l).saturating_sub(model.config().recent);
        if keep_from > self.states_start {
            let s = self.states.take().expect("states");
            let held = s.dims()[1];
            let drop = keep_from - self.states_start;
            self.states = (held > drop).then(|| s.narrow(1, drop, held - drop));
            self.states_start = keep_from;
        }
    }

    /// [`Self::close_block`] for several sessions at once: each encodes what it wrote, and the
    /// sessions whose block is now complete (exactly one new block each) take their backbone
    /// step together, one batched forward over their pooled blocks
    /// ([`crate::layers::Stack::forward_slots`]). `closes[i]`: (block start, emitted tokens,
    /// stopped, close span) of `sessions[i]`.
    pub(crate) fn close_blocks(
        model: &HybridLm,
        sessions: &mut [&mut HybridSession],
        closes: &[(usize, Vec<i32>, bool, bool)],
    ) {
        let l = model.config().block;
        let mut stepping = Vec::new();
        for (i, (s, (start, emitted, stopped, close_span))) in
            sessions.iter_mut().zip(closes).enumerate()
        {
            s.tokens.extend_from_slice(emitted);
            s.encode_pending(model);
            let complete = s.encoded / l;
            if s.tokens.len() - start >= l && complete == s.blocks + 1 {
                stepping.push(i);
            } else if s.tokens.len() - start >= l {
                // Not a single new block (cannot happen after a block's decode): alone.
                s.step_complete_blocks(model);
            }
            if s.tokens.len() - start >= l && *close_span {
                s.verified_span = false;
            }
            if *stopped {
                s.verified_span = false;
            }
        }
        if stepping.is_empty() {
            return;
        }
        let pooled: Vec<Tensor<3>> = stepping
            .iter()
            .map(|&i| {
                let s = &sessions[i];
                model.pool_blocks(s.states(s.blocks * l, (s.blocks + 1) * l).expect("block states"))
            })
            .collect();
        let x = Tensor::cat(pooled.clone(), 0);
        let h = {
            let mut picked: Vec<Option<&mut HybridSession>> =
                sessions.iter_mut().map(|s| Some(&mut **s)).collect();
            let mut caches: Vec<&mut [LayerCache]> = stepping
                .iter()
                .map(|&i| picked[i].take().expect("one row per session").backbone.as_mut_slice())
                .collect();
            model.backbone().forward_slots(x, &mut caches)
        };
        for (r, &i) in stepping.iter().enumerate() {
            let s = &mut sessions[i];
            s.cond = h.clone().narrow(0, r, 1);
            let host: Vec<f32> =
                pooled[r].clone().into_data().convert::<f32>().try_to_vec::<f32>().expect("f32");
            s.pooled.push(host);
            s.blocks += 1;
            // Trim the states as `step_complete_blocks` does.
            s.step_complete_blocks(model);
        }
    }

    pub(crate) fn uniform(&mut self) -> f64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 11) as f64 / (1u64 << 53) as f64
    }

    /// The open block's memories (generator with the known prefix `fixed`, token decoder
    /// without) and both stacks' keys and values.
    fn memories(
        &mut self,
        model: &HybridLm,
        fixed: &[i32],
    ) -> (Memory, Vec<LayerKv>, Memory, Vec<LayerKv>) {
        let h = model.config();
        let l = h.block;
        let start = self.blocks * l;
        let lo = start.saturating_sub(h.recent).max(self.states_start);
        let seq = self.states(lo, start);
        let n = start - lo;
        let next: Vec<i32> =
            (lo..start).map(|t| self.tokens.get(t + 1).copied().unwrap_or(0)).collect();
        let seq_next = (n > 0).then(|| ids_tensor(&next, &self.device));
        let windows = [(0, n)];
        let copy_windows = [(0, n.saturating_sub(1))];
        let cond_host: Vec<f32> =
            self.cond.clone().into_data().convert::<f32>().try_to_vec::<f32>().expect("f32");
        let picked = model.select_retrieved(&cond_host, &self.pooled, self.blocks);
        let retrieved = [retrieved_tokens(&self.tokens, &picked, l)];
        let j = fixed.len();
        let known = self.known_states(model, fixed);
        let entries =
            [model.block_entries(self.cond.clone(), &retrieved, Some((known, &[j])), start)];
        let memory = |with_known| {
            Entries::memory(
                &entries,
                with_known,
                seq.clone(),
                lo,
                &windows,
                &copy_windows,
                seq_next.clone(),
                &self.device,
            )
        };
        let (memory_g, dec) = (memory(true), memory(false));
        let (gk, dk) = (model.generator_kv(&memory_g), model.decoder_kv(&dec));
        (memory_g, gk, dec, dk)
    }

    /// The open block's known prefix `fixed` as encoder states `[1, L - 1, d]`, zero-padded:
    /// the encoded ones from the buffer, the rest (this block's emitted tokens, not yet
    /// encoded) by a throwaway encoder run.
    pub(crate) fn known_states(&mut self, model: &HybridLm, fixed: &[i32]) -> Tensor<3> {
        let (l, d) = (model.config().block, model.local_dim);
        let start = self.blocks * l;
        let j = fixed.len();
        let have = self.encoded.min(start + j).saturating_sub(start);
        let mut parts = Vec::new();
        if let Some(s) = self.states(start, start + have) {
            parts.push(s);
        }
        if j > have {
            let saved: Vec<LayerSnapshot> = self.encoder.iter().map(LayerCache::snapshot).collect();
            let x = model.embed(ids_tensor(&fixed[have..], &self.device));
            parts.push(model.encoder().forward(x, Some(&mut self.encoder)));
            for (c, s) in self.encoder.iter_mut().zip(&saved) {
                c.restore(s);
            }
        }
        if j < l - 1 {
            parts.push(Tensor::zeros([1, l - 1 - j, d], &self.device));
        }
        Tensor::cat(parts, 1)
    }

    /// The autoregressive path's log-probabilities `[L * V]` over the open block, input
    /// `inputs` (the block's tokens; position i reads token i - 1, later entries are padding),
    /// with the token decoder's memory `dec`.
    pub fn ar_pass(
        &self,
        model: &HybridLm,
        dec: &Memory,
        kvs: &[LayerKv],
        inputs: &[i32],
    ) -> Vec<f32> {
        self.decoder_pass(model, dec, kvs, inputs, None)
    }

    fn decoder_pass(
        &self,
        model: &HybridLm,
        dec: &Memory,
        kvs: &[LayerKv],
        inputs: &[i32],
        zhat: Option<Tensor<3>>,
    ) -> Vec<f32> {
        let l = model.config().block;
        let start = self.blocks * l;
        let mut padded = inputs.to_vec();
        padded.resize(l, 0);
        let ones = Tensor::<1>::ones([1], &self.device);
        let x = model.decoder_input(ids_tensor(&padded, &self.device), zhat.map(|z| (z, ones)), 1);
        let (rope, causal) = model.decoder_self(&self.device);
        let q_pos: Vec<usize> = (start..start + l).collect();
        let shape = PassShape {
            self_rope: &rope,
            self_mask: Some(causal),
            q_pos: &q_pos,
            rope_base: model.rope_base,
        };
        let logp = model.run_decoder(x, &shape, kvs, dec);
        logp.into_data().convert::<f32>().try_to_vec::<f32>().expect("f32")
    }

    /// The open block's memories as the decode loop builds them with known prefix `fixed`
    /// (generator, token decoder), for tests.
    pub fn block_memories(&mut self, model: &HybridLm, fixed: &[i32]) -> (Memory, Memory) {
        let (g, _, d, _) = self.memories(model, fixed);
        (g, d)
    }

    /// The autoregressive path's log-probabilities `[L * V]` of the open block for its tokens
    /// `inputs` (for tests).
    pub fn ar_logprobs(&mut self, model: &HybridLm, inputs: &[i32]) -> Vec<f32> {
        let (_, _, dec, dk) = self.memories(model, &[]);
        self.ar_pass(model, &dec, &dk, inputs)
    }

    /// Code temperature of level `k` of `depth` at request temperature `t`.
    fn level_temperature(t: f64, k: usize, depth: usize) -> f64 {
        if k < 3 || depth <= 3 {
            return t;
        }
        let span = (depth - 3).max(1) as f64;
        t * (1.0 - 0.95 * (k - 2) as f64 / span).max(0.05)
    }

    /// The generator's K passes over the open block's memory: codes per level and the
    /// reconstruction `[1, L, C]`.
    fn generate(
        &mut self,
        model: &HybridLm,
        memory_g: &Memory,
        kvs: &[LayerKv],
        opts: &DecodeOptions,
        hooks: &mut Hooks,
    ) -> Tensor<3> {
        let h = model.config();
        let depth = h.depth();
        let start = self.blocks * h.block;
        let mut zhat: Option<Tensor<3>> = None;
        for k in 0..depth {
            let logits = model.level_logits(memory_g, kvs, k, zhat.clone(), start);
            let n = h.codebook;
            let host: Vec<f32> =
                logits.into_data().convert::<f32>().try_to_vec::<f32>().expect("f32");
            let t = Self::level_temperature(opts.temperature, k, depth);
            let mut codes: Vec<i32> = host
                .chunks(n)
                .map(|row| sample(row, t, opts.top_p, opts.top_k, &mut || self.uniform()) as i32)
                .collect();
            if let Some(f) = hooks.codes.as_mut() {
                f(k, &mut codes);
            }
            let c = model.tokenizer().contribution(
                k,
                Tensor::<2, Int>::from_data(TensorData::new(codes, [1, h.levels[k]]), &self.device),
            );
            zhat = Some(match zhat {
                Some(z) => z + c,
                None => c,
            });
        }
        zhat.expect("at least one level")
    }

    /// Decodes the open block (or what is left of it): at most `limit` tokens. See the module
    /// docs for the steps.
    pub fn decode_block(
        &mut self,
        model: &HybridLm,
        limit: usize,
        opts: &DecodeOptions,
        hooks: &mut Hooks,
    ) -> BlockOut {
        let h = model.config().clone();
        let l = h.block;
        let start = self.blocks * l;
        assert_eq!(self.encoded, self.tokens.len(), "the stream is encoded up to its end");
        let mut fixed: Vec<i32> = self.tokens[start..].to_vec();
        let known = fixed.len();
        assert!(known < l, "the open block is incomplete");
        let redecode_min = opts.redecode_min.unwrap_or(h.redecode_min);
        let tau = opts.tau.unwrap_or(h.tau);
        let mut stats = BlockStats { known, ..BlockStats::default() };
        let mut verified = opts.verified || self.verified_span;
        let mut close_span = false;
        let mut stopped = false;
        let (gen0, gk0, dec, dk) = self.memories(model, &fixed);
        let mut gen_mem = Some((gen0, gk0));
        let mut draft: Option<(Vec<i32>, Tensor<3>)> = None;
        let mut passes = 0;
        let emit_budget = |fixed: &Vec<i32>| (known + limit).saturating_sub(fixed.len());
        // Generator passes and check passes.
        while !opts.ar_only && fixed.len() < l && !stopped {
            if draft.is_none() {
                let (memory_g, gk) = match gen_mem.take() {
                    Some(m) => m,
                    None => {
                        let (g, gk, _, _) = self.memories(model, &fixed);
                        (g, gk)
                    }
                };
                let zhat = self.generate(model, &memory_g, &gk, opts, hooks);
                stats.generator_runs += 1;
                let mut tokens: Vec<i32> = model
                    .tokenizer()
                    .draft(zhat.clone(), model.embedding())
                    .into_data()
                    .convert::<i32>()
                    .try_to_vec::<i32>()
                    .expect("int draft");
                if let Some(f) = hooks.draft.as_mut() {
                    f(passes, &mut tokens);
                }
                tokens[..fixed.len()].copy_from_slice(&fixed);
                draft = Some((tokens, zhat));
            }
            let (mut d, zhat) = draft.clone().expect("a draft");
            let p = fixed.len();
            // Inputs before p are the fixed tokens (after a switch, the pass's own token at the
            // switch may differ from the draft's).
            d[..p].copy_from_slice(&fixed);
            let rows = self.decoder_pass(model, &dec, &dk, &d, (!verified).then_some(zhat));
            stats.check_passes += 1;
            let v = model.vocab;
            let mut accepted = 0;
            let mut rejected = None;
            for i in p..l {
                let row = &rows[i * v..(i + 1) * v];
                if verified && let Some(f) = hooks.ar_rows.as_mut() {
                    f(&d[..i], i, row);
                }
                let ok = match hooks.accept.as_mut() {
                    Some(f) => f(passes, i),
                    None if verified => self.verify(row, d[i], opts),
                    None => {
                        let top = argmax(row);
                        top == d[i] as usize && (row[top] as f64).exp() >= tau
                    }
                };
                if ok {
                    accepted += 1;
                } else {
                    let token =
                        if verified { self.residual_sample(row, d[i], opts) } else { argmax(row) };
                    rejected = Some(token as i32);
                    break;
                }
            }
            match passes {
                0 => stats.first_accepted = Some(accepted),
                1 => stats.second_accepted = Some(accepted),
                _ => {}
            }
            passes += 1;
            let mut emitted: Vec<i32> = d[p..p + accepted].to_vec();
            emitted.extend(rejected);
            emitted.truncate(emit_budget(&fixed));
            let mut cut = emitted.len();
            if let Some(s) = emitted.iter().position(|t| opts.stop.contains(t)) {
                cut = s + 1;
                stopped = true;
            }
            let switch = (!verified)
                .then(|| emitted[..cut].iter().position(|t| opts.switch_on.contains(t)))
                .flatten();
            if let Some(s) = switch {
                cut = s + 1;
                stopped = false;
            }
            emitted.truncate(cut);
            let constrained = scan_cut(hooks, &mut emitted);
            if constrained {
                stopped = false;
            }
            if emitted.iter().any(|t| opts.switch_off.contains(t)) {
                close_span = true;
            }
            fixed.extend_from_slice(&emitted);
            if fixed.len() >= known + limit {
                stopped = true;
            }
            if stopped || fixed.len() >= l || constrained {
                break;
            }
            if let Some(s) = switch {
                // From the switch on, verified mode: one more check pass on the same draft.
                stats.switched = Some(p + s);
                verified = true;
                self.verified_span = true;
                continue;
            }
            if rejected.is_none() {
                continue;
            }
            if !stats.redecoded && l - fixed.len() >= redecode_min {
                stats.redecoded = true;
                draft = None;
                continue;
            }
            break;
        }
        // The autoregressive fill.
        while fixed.len() < l && !stopped {
            let i = fixed.len();
            let rows = self.ar_pass(model, &dec, &dk, &fixed);
            let v = model.vocab;
            let row = &rows[i * v..(i + 1) * v];
            if let Some(f) = hooks.ar_rows.as_mut() {
                f(&fixed, i, row);
            }
            let token = match hooks.pick.as_mut() {
                Some(pick) => pick(row),
                None => {
                    sample(row, opts.temperature, opts.top_p, opts.top_k, &mut || self.uniform())
                        as i32
                }
            };
            stats.ar_tokens += 1;
            fixed.push(token);
            if opts.switch_on.contains(&token) {
                self.verified_span = true;
            }
            if opts.switch_off.contains(&token) {
                close_span = true;
            }
            if opts.stop.contains(&token) || fixed.len() >= known + limit {
                stopped = true;
            }
        }
        let emitted = fixed[known..].to_vec();
        self.close_block(model, start, &emitted, stopped, close_span);
        BlockOut { tokens: emitted, stopped, stats }
    }

    /// After a block's decode: encodes what was written, and steps the backbone over the block
    /// once it is complete; the verified span ends with a stop, or with the block in which a
    /// switch-off token appeared.
    pub(crate) fn close_block(
        &mut self,
        model: &HybridLm,
        start: usize,
        emitted: &[i32],
        stopped: bool,
        close_span: bool,
    ) {
        let l = model.config().block;
        self.tokens.extend_from_slice(emitted);
        self.encode_pending(model);
        if self.tokens.len() - start >= l {
            self.step_complete_blocks(model);
            if close_span {
                self.verified_span = false;
            }
        }
        if stopped {
            self.verified_span = false;
        }
    }

    /// Decodes up to `max_tokens` tokens of a reply, block after block, until a stop token.
    pub fn generate_reply(
        &mut self,
        model: &HybridLm,
        max_tokens: usize,
        opts: &DecodeOptions,
        hooks: &mut Hooks,
    ) -> (Vec<i32>, Vec<BlockStats>) {
        self.rng = opts.seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1;
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

    /// Verified acceptance of draft token `d` against the autoregressive row `row`: argmax at
    /// temperature 0, else accept with the path's (tempered, nucleus) probability of `d`.
    fn verify(&mut self, row: &[f32], d: i32, opts: &DecodeOptions) -> bool {
        if opts.temperature <= 0.0 {
            return argmax(row) == d as usize;
        }
        let p = distribution(row, opts.temperature, opts.top_p, opts.top_k);
        self.uniform() < p[d as usize]
    }

    /// The token at a rejected position: the path's argmax at temperature 0, else a draw from
    /// its distribution with the rejected draft token removed (the speculative-sampling residual
    /// for a one-token draft).
    fn residual_sample(&mut self, row: &[f32], d: i32, opts: &DecodeOptions) -> usize {
        if opts.temperature <= 0.0 {
            return argmax(row);
        }
        let mut p = distribution(row, opts.temperature, opts.top_p, opts.top_k);
        p[d as usize] = 0.0;
        let total: f64 = p.iter().sum();
        if total <= 0.0 {
            return argmax(row);
        }
        draw(&p, total, self.uniform())
    }
}

impl HybridLm {
    /// The generator's per-level inference pass: level `k`'s code logits `[1, ℓ_k, n]` over the
    /// one-block memory `memory_g` (keys `kvs`) given the reconstruction of the coarser levels `zhat`
    /// `[1, L, C]` (None for level 0), for a block starting at `start`.
    pub fn level_logits(
        &self,
        memory_g: &Memory,
        kvs: &[LayerKv],
        k: usize,
        zhat: Option<Tensor<3>>,
        start: usize,
    ) -> Tensor<3> {
        let device = memory_g.extra.device();
        let n = self.config().levels[k];
        let x = self.level_input(k, zhat);
        let [_, _, d] = x.dims();
        let pos: Vec<usize> = (0..n).collect();
        let rope = Rope::at(&pos, self.head_dim, self.rope_base, &device);
        let q_pos = vec![start; n];
        let shape = PassShape {
            self_rope: &rope,
            self_mask: None,
            q_pos: &q_pos,
            rope_base: self.rope_base,
        };
        let out = self.run_generator(x.reshape([1, 1, n, d]), &shape, kvs, memory_g);
        self.tokenizer().code_logits(k, out.reshape([1, n, self.config().latent]))
    }
}

fn ids_tensor(ids: &[i32], device: &Device) -> Tensor<2, Int> {
    Tensor::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), device)
}

/// Index of the largest value (the first on ties).
fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in row.iter().enumerate() {
        if x > row[best] {
            best = i;
        }
    }
    best
}

/// Probabilities of `scores` (logits or log-probabilities) at temperature `t` (> 0), cut to
/// the smallest set holding `top_p` of the mass and renormalised.
fn distribution(scores: &[f32], t: f64, top_p: f64, top_k: usize) -> Vec<f64> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut p: Vec<f64> = scores.iter().map(|&s| ((s as f64 - max) / t).exp()).collect();
    // Top-k: only the k largest (ties at the cut kept together, as the device sampler does).
    if top_k > 0 && top_k < p.len() {
        let mut sorted = p.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let floor = sorted[top_k - 1];
        p.iter_mut().filter(|x| **x < floor).for_each(|x| *x = 0.0);
    }
    let total: f64 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= total);
    if top_p < 1.0 {
        let mut order: Vec<usize> = (0..p.len()).collect();
        order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
        let mut mass = 0.0;
        let mut keep = order.len();
        for (n, &i) in order.iter().enumerate() {
            mass += p[i];
            if mass >= top_p {
                keep = n + 1;
                break;
            }
        }
        for &i in &order[keep..] {
            p[i] = 0.0;
        }
        let total: f64 = p.iter().sum();
        p.iter_mut().for_each(|x| *x /= total);
    }
    p
}

/// A draw from unnormalised probabilities `p` summing to `total`, at `u` in [0, 1).
fn draw(p: &[f64], total: f64, u: f64) -> usize {
    let mut acc = 0.0;
    let target = u * total;
    for (i, &x) in p.iter().enumerate() {
        acc += x;
        if acc > target {
            return i;
        }
    }
    p.iter().rposition(|&x| x > 0.0).unwrap_or(0)
}

/// A token or code from `scores`: argmax at temperature 0, else a draw.
fn sample(
    scores: &[f32],
    t: f64,
    top_p: f64,
    top_k: usize,
    uniform: &mut dyn FnMut() -> f64,
) -> usize {
    if t <= 0.0 {
        return argmax(scores);
    }
    let p = distribution(scores, t, top_p, top_k);
    draw(&p, 1.0, uniform())
}

/// [`sample`] for the fast path.
pub(crate) fn sample_row(
    row: &[f32],
    t: f64,
    top_p: f64,
    top_k: usize,
    uniform: &mut dyn FnMut() -> f64,
) -> usize {
    sample(row, t, top_p, top_k, uniform)
}
