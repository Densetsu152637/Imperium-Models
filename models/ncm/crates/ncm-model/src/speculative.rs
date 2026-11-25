//! Self-speculative decoding of NCM concepts: a parallel draft head proposes all k tokens of
//! the next concept in one pass, and the local decoder checks them in one teacher-forced pass.
//!
//! The local decoder writes a concept token by token: k passes of its layers, each waiting for
//! the token before. Here a draft pass guesses the whole concept at once, and one verify pass
//! runs the decoder over the guess, causal within the concept, which gives the decoder's own
//! prediction at every position of the guess. The longest prefix where the guess equals the
//! decoder's pick is kept, the first mismatch is replaced by the decoder's pick, and if the
//! concept is not finished another verify pass runs on the repaired guess. Every verify pass
//! settles at least one more token, so a concept takes between one and k verify passes, and a
//! fully accepted draft costs one draft and one verify pass instead of k decoder steps.
//!
//! Greedy output equals plain greedy decoding ([`NcmLm::decode_concept`]): a position is only
//! kept when its whole prefix is settled and the token is the decoder's argmax there. (The
//! parallel pass rounds differently from the step-by-step one, so at an exact near-tie of the
//! two best logits the two paths may still pick different tokens, as with any batched
//! verification.) For sampling, [`Speculator::decode_sampled`] uses standard speculative
//! acceptance (accept a drafted token with probability `min(1, p/q)`, else resample from the
//! normalised `max(0, p - q)`), whose output is distributed exactly as sampling from the
//! decoder.
//!
//! The draft head ([`DraftHead`]) is `draft_layers` blocks at the decoder's width: learned
//! position queries for slots 0..k-1 plus the concept's condition, self-attention between the
//! slots (bidirectional: the slots carry no tokens), cross-attention over the encoder states
//! through the keys and values of decoder layer i's cross-attention cache (no cache of its
//! own), a SwiGLU MLP, its own final norm and the tied embedding. Training adds
//! [`draft_loss`] on [`NcmLm::forward_with_draft`]'s draft logits.

use burn::module::{Module, Param};
use burn::nn::{Embedding, RmsNorm};
use burn::tensor::activation::log_softmax;
use burn::tensor::{Bool, Int, Tensor, TensorData};
#[cfg(feature = "cuda")]
use burn::{
    backend::{Backend, BackendGraph, Dispatch},
    tensor::ExecutionError,
};

use crate::config::NcmConfig;
use crate::layers::{Attention, CrossQuery, Mlp, Rope, init_normal, rms_norm};
use crate::ncm::{NcmCache, NcmLm};

// ------------------------------------------------------------------------------ draft head

#[derive(Module, Debug)]
pub struct DraftBlock {
    n1: RmsNorm,
    self_attn: Attention,
    n2: RmsNorm,
    cross: CrossQuery,
    n3: RmsNorm,
    mlp: Mlp,
}

impl DraftBlock {
    fn new(cfg: &NcmConfig, device: &burn::tensor::Device) -> Self {
        let (d, hd, eps) = (cfg.local_dim, cfg.head_dim, cfg.norm_eps);
        Self {
            n1: rms_norm(d, eps, device),
            self_attn: Attention::new(d, cfg.dec_heads, cfg.dec_kv_heads, hd, None, device),
            n2: rms_norm(d, eps, device),
            cross: CrossQuery::new(d, cfg.dec_heads, hd, device),
            n3: rms_norm(d, eps, device),
            mlp: Mlp::new(d, cfg.dec_ff_dim, device),
        }
    }

    /// `x` is `[B, M, k, d]`, the slots of M concepts.
    fn forward(
        &self,
        x: Tensor<4>,
        rope_self: &Rope,
        rope_q: &Rope,
        keys: Tensor<4>,
        values: Tensor<4>,
        mask: Option<Tensor<2, Bool>>,
    ) -> Tensor<4> {
        let [b, m, k, d] = x.dims();
        let h = self.n1.forward(x.clone()).reshape([b * m, k, d]);
        // Every slot sees every other: no slot holds a token yet.
        let x = x + self.self_attn.forward_masked(h, rope_self, None).reshape([b, m, k, d]);
        let h = self.n2.forward(x.clone()).reshape([b, m * k, d]);
        let x = x + self.cross.forward(h, rope_q, keys, values, mask).reshape([b, m, k, d]);
        x.clone() + self.mlp.forward(self.n3.forward(x))
    }
}

/// The parallel draft head on the local decoder: every token of a concept in one pass.
#[derive(Module, Debug)]
pub struct DraftHead {
    /// Position queries of slots 0..k-1, `[k, d]`.
    slots: Param<Tensor<2>>,
    blocks: Vec<DraftBlock>,
    norm: RmsNorm,
}

impl DraftHead {
    pub(crate) fn new(cfg: &NcmConfig, device: &burn::tensor::Device) -> Self {
        Self {
            slots: init_normal().init([cfg.k, cfg.local_dim], device),
            blocks: (0..cfg.draft_layers).map(|_| DraftBlock::new(cfg, device)).collect(),
            norm: rms_norm(cfg.local_dim, cfg.norm_eps, device),
        }
    }

    /// Final states `[B, M, k, d]` of the slots of M concepts with conditions `cond`
    /// `[B, M, 1, d]`. Block i attends to `kvs[i]`, decoder layer i's encoder keys and values;
    /// `mask` hides encoder states per query, as in the decoder.
    pub(crate) fn forward(
        &self,
        cond: Tensor<4>,
        rope_self: &Rope,
        rope_q: &Rope,
        kvs: &[(Tensor<4>, Tensor<4>)],
        mask: Option<Tensor<2, Bool>>,
    ) -> Tensor<4> {
        let [b, m, _, d] = cond.dims();
        let k = self.slots.dims()[0];
        let slots = self.slots.val().reshape([1, 1, k, d]).expand([b, m, k, d]);
        let mut x = slots + cond;
        for (blk, (keys, values)) in self.blocks.iter().zip(kvs) {
            x = blk.forward(x, rope_self, rope_q, keys.clone(), values.clone(), mask.clone());
        }
        x
    }

    /// Logits over the tied embedding `emb` for states `[B, T, d]`.
    pub(crate) fn logits(&self, emb: &Embedding, x: Tensor<3>) -> Tensor<3> {
        crate::fused::linear_transposed(emb.weight.val(), self.norm.forward(x))
    }
}

/// The draft head's cross-entropy: `mean` over every position, and `per_slot` `[k]`, the mean
/// at each slot of the concept (later slots are harder: the head never sees the tokens before
/// them).
pub struct DraftLoss {
    pub mean: Tensor<1>,
    pub per_slot: Tensor<1>,
}

/// Cross-entropy of the draft logits `[B, N - k, V]` ([`NcmLm::forward_with_draft`]) against
/// `tokens` `[B, N]`, the sequence they were computed from, for concepts of `k` tokens. The
/// training crate adds `mean` (weighted) to the decoder's loss.
pub fn draft_loss(draft_logits: Tensor<3>, tokens: Tensor<2, Int>, k: usize) -> DraftLoss {
    let [b, t, _] = draft_logits.dims();
    assert_eq!(t % k, 0, "draft logits cover whole concepts");
    assert_eq!(tokens.dims()[1], t + k, "the draft logits start at token k");
    let targets = tokens.narrow(1, k, t).unsqueeze_dim::<3>(2);
    let nll = log_softmax(draft_logits, 2).gather(2, targets).neg();
    let per_slot = nll.reshape([b, t / k, k]).mean_dim(0).mean_dim(1).reshape([k]);
    DraftLoss { mean: per_slot.clone().mean(), per_slot }
}

// ------------------------------------------------------------------------------ decoding

/// How drafted tokens are judged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Acceptance {
    /// Against the decoder's own pick: the real thing, output equal to plain decoding.
    Verify,
    /// Synthetic, for benchmarks only: keep this share of drafted tokens whatever the decoder
    /// says. Decision n keeps the token when `frac(n * (sqrt(5) - 1) / 2)` is below the rate: a
    /// quasi-random sequence, so the kept share matches the rate closely even over a short
    /// run, without the periodic pattern of a plain counter. Runs the same passes and
    /// transfers as a real run with that acceptance rate, but the output is not the model's.
    Assume(f64),
}

/// `(sqrt(5) - 1) / 2`, the step of the synthetic decisions' sequence.
const GOLDEN: f64 = 0.618_033_988_749_894_8;

/// Counts of a [`Speculator`]'s work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecStats {
    pub concepts: usize,
    pub verify_passes: usize,
    /// Drafted tokens judged (each position of each concept once).
    pub judged: usize,
    /// Drafted tokens kept as drafted.
    pub accepted: usize,
}

impl SpecStats {
    /// Share of judged drafted tokens that were kept.
    pub fn acceptance(&self) -> f64 {
        self.accepted as f64 / self.judged.max(1) as f64
    }

    /// Verify passes per concept (1 to k).
    pub fn passes_per_concept(&self) -> f64 {
        self.verify_passes as f64 / self.concepts.max(1) as f64
    }
}

/// Writes NCM concepts by self-speculative decoding. One `Speculator` serves one cache.
///
/// On CUDA with the fused kernels and graphs on, the draft pass and the verify pass are each
/// captured as a CUDA graph once the cross-attention ring is full (like
/// [`crate::ncm::GraphedDecoder`]) and replayed for every concept; elsewhere they run eagerly.
pub struct Speculator {
    acceptance: Acceptance,
    /// Point of the synthetic decisions' sequence.
    phase: f64,
    stats: SpecStats,
    #[cfg(feature = "cuda")]
    graphs: GraphState,
}

impl Speculator {
    pub fn new(acceptance: Acceptance) -> Self {
        if let Acceptance::Assume(rate) = acceptance {
            assert!((0.0..=1.0).contains(&rate), "assumed acceptance {rate} is not in 0..=1");
        }
        Self {
            acceptance,
            phase: 0.0,
            stats: SpecStats::default(),
            #[cfg(feature = "cuda")]
            graphs: GraphState::Untried,
        }
    }

    pub fn stats(&self) -> SpecStats {
        self.stats
    }

    /// Whether the decision on a drafted token that does (`matches`) or does not equal the
    /// decoder's pick is to keep it.
    fn accepts(&mut self, matches: bool) -> bool {
        match self.acceptance {
            Acceptance::Verify => matches,
            Acceptance::Assume(rate) => {
                self.phase = (self.phase + GOLDEN).fract();
                self.phase < rate
            }
        }
    }

    /// The next concept's tokens `[B, k]` by greedy self-speculative decoding, equal to
    /// `model.decode_concept(cond, cache, None, false)` under [`Acceptance::Verify`]. Pass them
    /// to `append_tokens` to continue, as with plain decoding.
    pub fn decode(&mut self, model: &NcmLm, cond: Tensor<3>, cache: &NcmCache) -> Tensor<2, Int> {
        self.prepare(model, &cond, cache);
        let drafted = self.draft(model, &cond, cache);
        self.verify_draft(model, cond, cache, drafted)
    }

    /// [`Self::decode`] from the draft `drafted` `[B, k]` instead of the draft head's: the
    /// verification alone (tests force full, partial and no acceptance through it).
    pub fn decode_from(
        &mut self,
        model: &NcmLm,
        cond: Tensor<3>,
        cache: &NcmCache,
        drafted: Tensor<2, Int>,
    ) -> Tensor<2, Int> {
        self.prepare(model, &cond, cache);
        self.verify_draft(model, cond, cache, drafted)
    }

    fn verify_draft(
        &mut self,
        model: &NcmLm,
        cond: Tensor<3>,
        cache: &NcmCache,
        drafted: Tensor<2, Int>,
    ) -> Tensor<2, Int> {
        let [b, _, _] = cond.dims();
        let k = model.concept_size();
        let device = cond.device();
        let mut guess_dev = drafted.clone();
        let mut guess: Vec<i64> = Vec::new();
        let mut settled = vec![0usize; b];
        loop {
            let verified = self.verify(model, &cond, cache, guess_dev);
            // One device-to-host transfer per pass; the first also brings the draft.
            let picks = if guess.is_empty() {
                let both = ints(Tensor::cat(vec![drafted.clone(), verified.clone()], 1));
                let (mut draft, mut picks) = (Vec::new(), Vec::new());
                for row in both.chunks(2 * k) {
                    draft.extend_from_slice(&row[..k]);
                    picks.extend_from_slice(&row[k..]);
                }
                guess = draft;
                picks
            } else {
                ints(verified.clone())
            };
            self.stats.verify_passes += 1;
            for row in 0..b {
                let (g, v) = (&mut guess[row * k..(row + 1) * k], &picks[row * k..(row + 1) * k]);
                let mut j = settled[row];
                while j < k {
                    self.stats.judged += 1;
                    let keep = self.accepts(g[j] == v[j]);
                    if !keep {
                        g[j] = v[j];
                    }
                    self.stats.accepted += keep as usize;
                    j += 1;
                    if !keep {
                        break;
                    }
                }
                settled[row] = j;
            }
            if settled.iter().all(|&s| s == k) {
                self.stats.concepts += 1;
                // Under real verification the last pass's picks are the concept (settled
                // positions are recomputed from the same prefix), so the device buffer serves
                // without an upload. It is the graph's output buffer on CUDA: the caller uses
                // it before the next call, as with `GraphedDecoder`.
                return if guess == picks { verified } else { upload(&guess, [b, k], &device) };
            }
            guess_dev = upload(&guess, [b, k], &device);
        }
    }

    /// The next concept's tokens `[1, k]` by speculative sampling (batch 1): drafted tokens
    /// are drawn from the draft head's distributions and accepted with probability
    /// `min(1, p/q)`, a rejected one is redrawn from the normalised `max(0, p - q)`, so the
    /// concept is distributed as if drawn token by token from the decoder.
    ///
    /// `dist(j, logits)` turns the logits of slot j into the sampling distribution
    /// (temperature, top-k, top-p ...); it is applied to the draft head's and to the decoder's
    /// logits alike, and called once per slot for each. `uniform()` draws from [0, 1).
    /// Runs eagerly: the distributions are read on the host anyway.
    pub fn decode_sampled(
        &mut self,
        model: &NcmLm,
        cond: Tensor<3>,
        cache: &NcmCache,
        dist: &mut dyn FnMut(usize, &[f32]) -> Vec<f32>,
        uniform: &mut dyn FnMut() -> f32,
    ) -> Tensor<2, Int> {
        assert_eq!(cond.dims()[0], 1, "speculative sampling runs batch 1");
        let k = model.concept_size();
        let device = cond.device();
        let q_logits = floats(model.draft_concept(cond.clone(), cache));
        let vocab = q_logits.len() / k;
        let q: Vec<Vec<f32>> =
            (0..k).map(|j| dist(j, &q_logits[j * vocab..(j + 1) * vocab])).collect();
        let mut guess: Vec<i64> = q.iter().map(|qj| sample(qj, uniform()) as i64).collect();
        let mut settled = 0;
        while settled < k {
            let tokens = upload(&guess, [1, k], &device);
            let p_logits = floats(model.verify_concept(cond.clone(), cache, tokens));
            self.stats.verify_passes += 1;
            let mut j = settled;
            while j < k {
                let p = dist(j, &p_logits[j * vocab..(j + 1) * vocab]);
                self.stats.judged += 1;
                let fix = accept_or_resample(&p, &q[j], guess[j] as usize, uniform(), uniform());
                j += 1;
                match fix {
                    None => self.stats.accepted += 1,
                    Some(token) => {
                        guess[j - 1] = token as i64;
                        break;
                    }
                }
            }
            settled = j;
        }
        self.stats.concepts += 1;
        upload(&guess, [1, k], &device)
    }

    /// Readies the passes for a new concept: on CUDA, captures the graphs if they can serve it
    /// and copies the condition and the concept's RoPE tables into their input buffers.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn prepare(&mut self, model: &NcmLm, cond: &Tensor<3>, cache: &NcmCache) {
        #[cfg(feature = "cuda")]
        if let Some(g) = self.graphs(model, cond, cache) {
            let device = cond.device();
            let (cos, sin) = model.rope(cache.tokens, model.concept_size(), &device).tables();
            let (q_cos, q_sin) = g.rope_q.tables();
            crate::kernels::assign_into(&g.cond, cond.clone(), 0, 0);
            crate::kernels::assign_into(&q_cos, cos, 0, 0);
            crate::kernels::assign_into(&q_sin, sin, 0, 0);
        }
    }

    /// Greedy draft tokens `[B, k]` on the device. Follows [`Self::prepare`].
    fn draft(&mut self, model: &NcmLm, cond: &Tensor<3>, cache: &NcmCache) -> Tensor<2, Int> {
        #[cfg(feature = "cuda")]
        if let GraphState::Ready(g) = &self.graphs {
            // Safety: every buffer the graph touched is held by `g`, by the model (weights) or
            // by `cache` (the cross-attention rings), all alive across the replay, and all work
            // runs on this thread's stream.
            unsafe { Dispatch::graph_replay(cond.device().as_dispatch(), &g.draft) }
                .expect("draft graph replay");
            return g.drafted.clone();
        }
        let [b, _, _] = cond.dims();
        let logits = model.draft_concept(cond.clone(), cache);
        crate::fused::argmax_last(logits).reshape([b, model.concept_size()])
    }

    /// The decoder's greedy picks `[B, k]` on the device, teacher-forced on `guess`. Follows
    /// [`Self::prepare`] of the same concept.
    fn verify(
        &mut self,
        model: &NcmLm,
        cond: &Tensor<3>,
        cache: &NcmCache,
        guess: Tensor<2, Int>,
    ) -> Tensor<2, Int> {
        #[cfg(feature = "cuda")]
        if let GraphState::Ready(g) = &self.graphs {
            crate::kernels::assign_int_into(&g.guess, guess, 0, 0);
            // Safety: as for the draft graph.
            unsafe { Dispatch::graph_replay(cond.device().as_dispatch(), &g.verify) }
                .expect("verify graph replay");
            return g.verified.clone();
        }
        let [b, k] = guess.dims();
        crate::fused::argmax_last(model.verify_concept(cond.clone(), cache, guess)).reshape([b, k])
    }
}

/// Keeps drafted token `x` (None) with probability `min(1, p[x] / q[x])` (`u` uniform in
/// [0, 1)), or returns a replacement drawn with `u2` from the normalised `max(0, p - q)`.
pub fn accept_or_resample(p: &[f32], q: &[f32], x: usize, u: f32, u2: f32) -> Option<usize> {
    if q[x] > 0.0 && u * q[x] < p[x] {
        return None;
    }
    let residual: Vec<f32> = p.iter().zip(q).map(|(p, q)| (p - q).max(0.0)).collect();
    let total: f32 = residual.iter().sum();
    // p == q up to rounding: the token was kept with probability 1 in exact arithmetic.
    if total <= 0.0 {
        return Some(sample(p, u2));
    }
    Some(sample(&residual, u2 * total))
}

/// The index where the running sum of `weights` first exceeds `u` (the last positive weight if
/// rounding leaves `u` beyond the sum).
fn sample(weights: &[f32], u: f32) -> usize {
    let mut acc = 0.0;
    let mut last = 0;
    for (i, &w) in weights.iter().enumerate() {
        if w > 0.0 {
            acc += w;
            last = i;
            if u < acc {
                return i;
            }
        }
    }
    last
}

fn ints(t: Tensor<2, Int>) -> Vec<i64> {
    t.into_data().convert::<i64>().try_to_vec::<i64>().expect("int tensor")
}

fn floats(t: Tensor<3>) -> Vec<f32> {
    t.into_data().convert::<f32>().try_to_vec::<f32>().expect("float tensor")
}

fn upload(tokens: &[i64], shape: [usize; 2], device: &burn::tensor::Device) -> Tensor<2, Int> {
    let data: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    Tensor::from_data(TensorData::new(data, shape), device)
}

// ------------------------------------------------------------------------------ CUDA graphs

#[cfg(feature = "cuda")]
enum GraphState {
    Untried,
    Failed(String),
    Ready(Box<SpecGraphs>),
}

/// The draft and verify passes captured over fixed buffers: the condition and the concept's
/// RoPE tables (refreshed before the draft replay), the guess (refreshed before each verify
/// replay), and the two passes' greedy picks.
#[cfg(feature = "cuda")]
struct SpecGraphs {
    draft: BackendGraph<Dispatch>,
    verify: BackendGraph<Dispatch>,
    cond: Tensor<3>,
    rope_q: Rope,
    guess: Tensor<2, Int>,
    drafted: Tensor<2, Int>,
    verified: Tensor<2, Int>,
    /// Kept for the buffers the graphs read.
    _rope_self: Rope,
    _causal: Tensor<2, Bool>,
}

#[cfg(feature = "cuda")]
impl Speculator {
    /// The captured graphs, capturing them first if this is the first concept they can serve.
    fn graphs(&mut self, model: &NcmLm, cond: &Tensor<3>, cache: &NcmCache) -> Option<&SpecGraphs> {
        let ring_full = cache.cross.iter().all(|c| c.len() >= c.capacity());
        if matches!(self.graphs, GraphState::Untried)
            && ring_full
            && crate::fused::graphs_active(&cond.device())
        {
            self.graphs = match capture(model, cond, cache) {
                Ok(g) => GraphState::Ready(Box::new(g)),
                Err(e) => GraphState::Failed(e),
            };
        }
        match &self.graphs {
            GraphState::Ready(g) => Some(g),
            _ => None,
        }
    }

    /// Why the passes run eagerly, if a capture was tried and failed.
    pub fn failure(&self) -> Option<&str> {
        match &self.graphs {
            GraphState::Failed(e) => Some(e),
            _ => None,
        }
    }

    /// Whether the passes replay captured graphs.
    pub fn is_captured(&self) -> bool {
        matches!(self.graphs, GraphState::Ready(_))
    }
}

/// Warms up and records the draft pass and the verify pass. Neither writes a cache, so the
/// warm-up runs leave nothing to undo.
#[cfg(feature = "cuda")]
fn capture(model: &NcmLm, cond: &Tensor<3>, cache: &NcmCache) -> Result<SpecGraphs, String> {
    let device = cond.device();
    let dispatch = device.as_dispatch();
    let k = model.concept_size();
    let [b, _, _] = cond.dims();
    let cond_buffer = Tensor::<3>::zeros(cond.dims(), &device);
    crate::kernels::assign_into(&cond_buffer, cond.clone(), 0, 0);
    let (cos, sin) = model.rope(cache.tokens, k, &device).tables();
    let buffer = |t: Tensor<4>| {
        let b = Tensor::<4>::zeros(t.dims(), &device);
        crate::kernels::assign_into(&b, t, 0, 0);
        b
    };
    let rope_q = Rope::from_tables(buffer(cos), buffer(sin));
    let rope_self = model.rope(0, k, &device);
    let causal = model.causal_mask(&device);
    let guess = Tensor::<2, Int>::zeros([b, k], &device);
    let draft = || {
        let logits = model.draft_with_ropes(cond_buffer.clone(), cache, &rope_self, &rope_q);
        crate::fused::argmax_last(logits).reshape([b, k])
    };
    let verify = || {
        let (cond, guess, causal) = (cond_buffer.clone(), guess.clone(), causal.clone());
        let logits = model.verify_with_ropes(cond, cache, guess, &rope_self, &rope_q, causal);
        crate::fused::argmax_last(logits).reshape([b, k])
    };
    let err = |what: &str, e: ExecutionError| format!("{what}: {e:?}");
    let record = |pass: &dyn Fn() -> Tensor<2, Int>| {
        Dispatch::graph_prepare(dispatch).map_err(|e| err("graph_prepare", e))?;
        // Warm-up runs compile every kernel and fill the memory pool with every buffer the
        // capture will take, so nothing allocates mid-capture.
        for _ in 0..3 {
            let out = pass();
            Dispatch::sync(dispatch).map_err(|e| err("warm-up", e))?;
            drop(out);
        }
        Dispatch::graph_start_capture(dispatch).map_err(|e| err("graph_start_capture", e))?;
        let out = pass();
        let graph =
            Dispatch::graph_stop_capture(dispatch).map_err(|e| err("graph_stop_capture", e))?;
        Ok::<_, String>((graph, out))
    };
    let (draft_graph, drafted) = record(&draft).map_err(|e| format!("draft pass: {e}"))?;
    let (verify_graph, verified) = record(&verify).map_err(|e| format!("verify pass: {e}"))?;
    Ok(SpecGraphs {
        draft: draft_graph,
        verify: verify_graph,
        cond: cond_buffer,
        rope_q,
        guess,
        drafted,
        verified,
        _rope_self: rope_self,
        _causal: causal,
    })
}
