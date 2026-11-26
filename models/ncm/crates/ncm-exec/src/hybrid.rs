//! The hybrid's backend-neutral block types and the host half of its block state machine.
//!
//! Per block (design: "Decoding one block"): a block pass (the generator and a check pass of the
//! token decoder, or the check alone in verified mode) proposes a draft and says where it first
//! disagrees with the autoregressive path; the host accepts the draft up to there, applies stops,
//! the mixed-block switch and the grammar scan, then asks for a re-decode, a recheck, the
//! autoregressive fill (one token per pass) or ends the block.
//!
//! Everything in [`BlockLogic`] works on the read-back integers and the fixed-token list only,
//! so it is shared by every backend: a backend's `BlockRun` embeds one and keeps its own staging
//! and pass execution (`ncm_model::hybrid::BlockRun` for Burn, the ORT backend's for ONNX).

/// How a reply decodes.
#[derive(Clone, Debug)]
pub struct DecodeOptions {
    /// Request temperature: codes of levels 1-3 sample at it, finer levels at a linearly
    /// decreasing share of it (to 5% at the finest); verified mode and the autoregressive fill
    /// sample tokens at it. 0: argmax everywhere.
    pub temperature: f64,
    /// Nucleus mass for codes and tokens (1: off).
    pub top_p: f64,
    /// Most candidates for codes and tokens before the nucleus cut (0: off).
    pub top_k: usize,
    pub seed: u64,
    /// Verified mode for the whole reply.
    pub verified: bool,
    /// The autoregressive path only (no generator): the Block Transformer-style variant.
    pub ar_only: bool,
    /// Tokens that end the reply (kept in the stream).
    pub stop: Vec<i32>,
    /// Tokens that switch to verified mode from the next position (tool-call and code-fence
    /// openers), and tokens whose block's end switches back (closers).
    pub switch_on: Vec<i32>,
    pub switch_off: Vec<i32>,
    /// Re-decode after a rejection when at least this many positions remain (None: the
    /// config's; above the block length: never).
    pub redecode_min: Option<usize>,
    /// Parallel mode's confidence threshold (None: the config's).
    pub tau: Option<f64>,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
            verified: false,
            ar_only: false,
            stop: Vec::new(),
            switch_on: Vec::new(),
            switch_off: Vec::new(),
            redecode_min: None,
            tau: None,
        }
    }
}

/// Rewrites a list in place: `(pass or level, values)`.
pub type Rewrite<'a> = &'a mut dyn FnMut(usize, &mut Vec<i32>);
/// Sees an autoregressive-path row: `(the block's tokens before the position, position,
/// log-probabilities)`.
pub type RowObserver<'a> = &'a mut dyn FnMut(&[i32], usize, &[f32]);

/// The grammar scan over proposed tokens (see [`Hooks::scan`]).
pub type Scan<'a> = &'a mut dyn FnMut(&[i32]) -> usize;
/// An autoregressive-path token choice (see [`Hooks::pick`]).
pub type Pick<'a> = &'a mut dyn FnMut(&[f32]) -> i32;

/// Observers and overrides for tests and benchmarks.
#[derive(Default)]
pub struct Hooks<'a> {
    /// Rewrites each pass's draft (before the known prefix is forced): `(pass, draft)`.
    pub draft: Option<Rewrite<'a>>,
    /// Rewrites each level's sampled codes: `(level, codes)`.
    pub codes: Option<Rewrite<'a>>,
    /// Sees every autoregressive-path row: `(the block's tokens before the position, position,
    /// log-probabilities)`.
    pub ar_rows: Option<RowObserver<'a>>,
    /// Decides acceptance instead of the rule, for synthetic acceptance in benchmarks:
    /// `(pass, position) -> accept`. The output is then not the model's.
    pub accept: Option<&'a mut dyn FnMut(usize, usize) -> bool>,
    /// The grammar constraint scan: given the tokens a check pass would emit, in order, returns
    /// how many of them the grammar allows (advancing its state over those). The block is cut
    /// there and continues on the autoregressive path, where `pick` chooses every token.
    pub scan: Option<Scan<'a>>,
    /// Chooses an autoregressive-path token from its log-probabilities (a grammar mask, the
    /// request's sampler), in place of the argmax or the plain draw.
    pub pick: Option<Pick<'a>>,
}

/// Applies the constraint scan to `emitted`: true when it cut them (the block continues on the
/// autoregressive path).
pub fn scan_cut(hooks: &mut Hooks, emitted: &mut Vec<i32>) -> bool {
    let Some(scan) = hooks.scan.as_mut() else { return false };
    let allowed = scan(emitted).min(emitted.len());
    let cut = allowed < emitted.len();
    emitted.truncate(allowed);
    cut
}

/// What one block's decode did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockStats {
    /// Positions the block started with fixed (the known prefix).
    pub known: usize,
    /// Draft tokens accepted by the first and the second check pass (before the rejection).
    pub first_accepted: Option<usize>,
    pub second_accepted: Option<usize>,
    pub redecoded: bool,
    /// Where parallel mode switched to verified mode (the switch token's position).
    pub switched: Option<usize>,
    pub generator_runs: usize,
    pub check_passes: usize,
    /// Tokens written on the autoregressive path one per pass.
    pub ar_tokens: usize,
}

/// One block's decode.
#[derive(Clone, Debug, Default)]
pub struct BlockOut {
    /// Tokens emitted (after the known prefix).
    pub tokens: Vec<i32>,
    /// A stop token or the limit ended the reply inside this block.
    pub stopped: bool,
    pub stats: BlockStats,
}

/// How a block pass judges the draft.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PassMode {
    /// Against the autoregressive path (no z-hat), greedy.
    Verified,
    /// Against the autoregressive path, its tokens drawn on the device (speculative sampling,
    /// CUDA).
    VerifiedSampled,
    /// The staged draft checked again against the autoregressive path, no generator (after a
    /// mixed-block switch), greedy or sampled.
    Recheck,
    RecheckSampled,
    /// With z-hat, greedy codes.
    Parallel,
    /// With z-hat, codes sampled on the device (CUDA).
    ParallelSampled,
}

impl PassMode {
    pub fn verified(self) -> bool {
        matches!(
            self,
            Self::Verified | Self::VerifiedSampled | Self::Recheck | Self::RecheckSampled
        )
    }

    pub fn sampled_tokens(self) -> bool {
        matches!(self, Self::VerifiedSampled | Self::RecheckSampled)
    }

    pub fn sampled_codes(self) -> bool {
        self == Self::ParallelSampled
    }

    pub fn generates(self) -> bool {
        !matches!(self, Self::Recheck | Self::RecheckSampled)
    }
}

/// What a block pass reports for one row: the read-back `[first, any, ids(L), draft(L)]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassResult {
    /// The first rejected position at or after the known prefix, if any.
    pub rejected: Option<usize>,
    /// The pass's token per position.
    pub ids: Vec<i32>,
    pub draft: Vec<i32>,
}

/// One row of device-draw settings: temperature, top-k, top-p, a counter-based seed and the
/// draws made so far in the request. Backend-neutral: the CUDA backend maps it to its sampler's
/// settings; other backends use it on the host or as graph inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawRow {
    pub temperature: f32,
    /// 0 keeps every token.
    pub top_k: usize,
    /// 1 keeps every token.
    pub top_p: f32,
    pub seed: u64,
    /// Draws made so far in the request.
    pub counter: u32,
}

impl DrawRow {
    /// The largest logit (top-k 1), as a row of a batch of per-row settings.
    pub const GREEDY: Self = Self { temperature: 1.0, top_k: 1, top_p: 1.0, seed: 0, counter: 0 };
}

/// The draws' settings of one pass: per level `l_k` code rows, and `L` token rows. Empty for
/// greedy passes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DrawSettings {
    pub codes: Vec<Vec<DrawRow>>,
    pub tokens: Vec<DrawRow>,
}

impl DrawSettings {
    /// The settings of every row of a pool of `rows` slots: each slot of `parts` (its row and
    /// settings) at its row, greedy draws (top-k 1) for the other rows.
    pub fn rows(
        parts: &[(usize, &DrawSettings)],
        rows: usize,
        levels: &[usize],
        l: usize,
    ) -> DrawSettings {
        let fill = |n: usize, get: &dyn Fn(&DrawSettings) -> Option<Vec<DrawRow>>| {
            let mut out = Vec::with_capacity(rows * n);
            for r in 0..rows {
                let own = parts.iter().find(|p| p.0 == r).and_then(|p| get(p.1));
                match own {
                    Some(v) if v.len() == n => out.extend(v),
                    _ => out.extend(std::iter::repeat_n(DrawRow::GREEDY, n)),
                }
            }
            out
        };
        let sampled_codes = parts.iter().any(|p| !p.1.codes.is_empty());
        let sampled_tokens = parts.iter().any(|p| !p.1.tokens.is_empty());
        let codes = if sampled_codes {
            levels.iter().enumerate().map(|(k, &n)| fill(n, &|d| d.codes.get(k).cloned())).collect()
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

    /// Whether no draw is sampled (a greedy pass).
    pub fn is_greedy(&self) -> bool {
        self.codes.is_empty() && self.tokens.is_empty()
    }
}

/// Where a block's decode stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockPhase {
    /// A block pass is due (first, re-decode or recheck).
    Pass,
    /// The autoregressive fill: one token per pass.
    Fill,
    /// The block is decoded (the backend's `end_block` closes it).
    Done,
}

/// What a backend must do after [`BlockLogic::apply_pass`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PassEffects {
    /// The session enters verified mode across blocks (a switch token was accepted).
    pub verified_span: bool,
    /// The draft to stage as the pass input of the recheck that now follows.
    pub stage_draft: Option<Vec<i32>>,
}

/// What a backend must do after [`BlockLogic::apply_fill`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FillEffects {
    /// The session enters verified mode across blocks (the fill wrote a switch token).
    pub verified_span: bool,
}

/// What closing a block needs: the block's first position, the tokens it wrote after the known
/// prefix, whether it stopped, and whether a switch-off token closes the verified span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockClose {
    pub start: usize,
    pub emitted: Vec<i32>,
    pub stopped: bool,
    pub close_span: bool,
}

/// The host state of one block's decode, shared by every backend.
#[derive(Clone, Debug)]
pub struct BlockLogic {
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

impl BlockLogic {
    /// Starts a block at token position `start` whose first `fixed.len()` tokens are known
    /// (from the prompt or an earlier pass), decoding at most `limit` more. `verified`: the
    /// reply or the carried span is in verified mode; `ar_only`: no generator.
    pub fn begin(
        start: usize,
        fixed: Vec<i32>,
        limit: usize,
        verified: bool,
        ar_only: bool,
    ) -> Self {
        let known = fixed.len();
        Self {
            start,
            known,
            limit,
            fixed,
            stats: BlockStats { known, ..BlockStats::default() },
            verified,
            recheck: false,
            redecode: false,
            passes: 0,
            stopped: false,
            close_span: false,
            phase: if ar_only { BlockPhase::Fill } else { BlockPhase::Pass },
        }
    }

    pub fn phase(&self) -> BlockPhase {
        self.phase
    }

    /// The block position the next fill pass writes.
    pub fn next_position(&self) -> usize {
        self.fixed.len()
    }

    /// The block's first token position in the stream.
    pub fn start(&self) -> usize {
        self.start
    }

    /// The tokens fixed so far (the known prefix and everything written since).
    pub fn fixed(&self) -> &[i32] {
        &self.fixed
    }

    /// Whether the next pass re-decodes or rechecks, so its known prefix must be staged again.
    pub fn restage_known(&self) -> bool {
        self.redecode || self.recheck
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

    /// Whether the fill draws its tokens on the device (`device_draws`: the backend has a
    /// device sampler).
    pub fn fill_sampled(&self, opts: &DecodeOptions, device_draws: bool) -> bool {
        opts.temperature > 0.0 && device_draws
    }

    /// Applies a block pass's result: accepted tokens, the rejected position's token, stops, a
    /// mixed-block switch, the constraint scan, then a re-decode, a recheck, the fill or the
    /// block's end. `l` is the block length and `redecode_min` the config's re-decode threshold
    /// (`opts.redecode_min` overrides it).
    pub fn apply_pass(
        &mut self,
        l: usize,
        redecode_min: usize,
        result: PassResult,
        opts: &DecodeOptions,
        hooks: &mut Hooks,
    ) -> PassEffects {
        let redecode_min = opts.redecode_min.unwrap_or(redecode_min);
        let mode = self.mode(opts);
        self.recheck = false;
        self.redecode = false;
        self.stats.generator_runs += usize::from(mode.generates());
        self.stats.check_passes += 1;
        let p = self.fixed.len();
        let mut cut = match hooks.accept.as_mut() {
            Some(f) => (p..l).find(|&i| !f(self.passes, i)),
            None => result.rejected,
        };
        if cut.is_some_and(|c| c < p) {
            cut = Some(p);
        }
        let accepted = cut.unwrap_or(l) - p;
        match self.passes {
            0 => self.stats.first_accepted = Some(accepted),
            1 => self.stats.second_accepted = Some(accepted),
            _ => {}
        }
        self.passes += 1;
        let mut emitted: Vec<i32> = result.draft[p..p + accepted].to_vec();
        if let Some(c) = cut {
            emitted.push(result.ids[c]);
        }
        emitted.truncate((self.known + self.limit).saturating_sub(self.fixed.len()));
        let mut keep = emitted.len();
        if let Some(s) = emitted.iter().position(|t| opts.stop.contains(t)) {
            keep = s + 1;
            self.stopped = true;
        }
        let switch = (!self.verified)
            .then(|| emitted[..keep].iter().position(|t| opts.switch_on.contains(t)))
            .flatten();
        if let Some(s) = switch {
            keep = s + 1;
            self.stopped = false;
        }
        emitted.truncate(keep);
        let constrained = scan_cut(hooks, &mut emitted);
        if constrained {
            self.stopped = false;
        }
        if emitted.iter().any(|t| opts.switch_off.contains(t)) {
            self.close_span = true;
        }
        self.fixed.extend_from_slice(&emitted);
        if self.fixed.len() >= self.known + self.limit {
            self.stopped = true;
        }
        let mut effects = PassEffects::default();
        self.phase = if self.stopped || self.fixed.len() >= l {
            BlockPhase::Done
        } else if constrained {
            // The grammar cut the block: the rest on the autoregressive path.
            BlockPhase::Fill
        } else if let Some(s) = switch {
            self.stats.switched = Some(p + s);
            self.verified = true;
            effects.verified_span = true;
            // One more check of the same draft, in verified mode.
            effects.stage_draft = Some(result.draft.clone());
            self.recheck = true;
            BlockPhase::Pass
        } else if cut.is_none() {
            BlockPhase::Pass
        } else if !self.stats.redecoded && l - self.fixed.len() >= redecode_min {
            self.stats.redecoded = true;
            self.redecode = true;
            BlockPhase::Pass
        } else {
            BlockPhase::Fill
        };
        effects
    }

    /// Applies one fill pass: the token at the next position, from the pass's token `drawn`
    /// (argmax or device draw) or, when the host must choose (a grammar pick, a host draw, an
    /// observer), from its log-probabilities `row()`. `host_draw` draws from a row on the host
    /// (called only at temperature above 0 without device draws); `block` is the block length.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_fill(
        &mut self,
        block: usize,
        drawn: i32,
        row: &mut dyn FnMut() -> Vec<f32>,
        host_draw: &mut dyn FnMut(&[f32]) -> i32,
        opts: &DecodeOptions,
        hooks: &mut Hooks,
        device_draws: bool,
    ) -> FillEffects {
        let i = self.fixed.len();
        let host_sampling = opts.temperature > 0.0 && !self.fill_sampled(opts, device_draws);
        let token = if host_sampling || hooks.ar_rows.is_some() || hooks.pick.is_some() {
            let row = row();
            if let Some(f) = hooks.ar_rows.as_mut() {
                f(&self.fixed, i, &row);
            }
            if let Some(pick) = hooks.pick.as_mut() {
                pick(&row)
            } else if host_sampling {
                host_draw(&row)
            } else {
                drawn
            }
        } else {
            drawn
        };
        self.stats.ar_tokens += 1;
        self.fixed.push(token);
        let mut effects = FillEffects::default();
        if opts.switch_on.contains(&token) {
            effects.verified_span = true;
        }
        if opts.switch_off.contains(&token) {
            self.close_span = true;
        }
        if opts.stop.contains(&token) || self.fixed.len() >= self.known + self.limit {
            self.stopped = true;
        }
        if self.stopped || self.fixed.len() >= block {
            self.phase = BlockPhase::Done;
        }
        effects
    }

    /// The tokens written after the known prefix.
    pub fn emitted(&self) -> &[i32] {
        &self.fixed[self.known..]
    }

    /// What closing the block needs.
    pub fn close(&self) -> BlockClose {
        BlockClose {
            start: self.start,
            emitted: self.emitted().to_vec(),
            stopped: self.stopped,
            close_span: self.close_span,
        }
    }

    /// The block's result (after the backend closed it).
    pub fn into_out(self) -> BlockOut {
        let tokens = self.fixed[self.known..].to_vec();
        BlockOut { tokens, stopped: self.stopped, stats: self.stats }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const L: usize = 8;

    fn result(rejected: Option<usize>) -> PassResult {
        PassResult {
            rejected,
            ids: (100..100 + L as i32).collect(),
            draft: (0..L as i32).collect(),
        }
    }

    fn plain() -> DecodeOptions {
        DecodeOptions { stop: vec![-1], ..Default::default() }
    }

    fn opts() -> DecodeOptions {
        DecodeOptions {
            stop: vec![-1],
            switch_on: vec![5],
            switch_off: vec![6],
            ..Default::default()
        }
    }

    #[test]
    fn full_acceptance_ends_the_block() {
        let mut b = BlockLogic::begin(0, vec![], L, false, false);
        assert_eq!(b.phase(), BlockPhase::Pass);
        b.apply_pass(L, 2, result(None), &plain(), &mut Hooks::default());
        assert_eq!(b.phase(), BlockPhase::Done);
        assert_eq!(b.stats.first_accepted, Some(L));
        let out = b.into_out();
        assert_eq!(out.tokens, (0..L as i32).collect::<Vec<_>>());
        assert!(!out.stopped || out.tokens.len() == L);
    }

    #[test]
    fn a_rejection_emits_the_passes_token_then_redecodes() {
        let mut b = BlockLogic::begin(0, vec![], L, false, false);
        b.apply_pass(L, 2, result(Some(3)), &plain(), &mut Hooks::default());
        assert_eq!(b.fixed(), &[0, 1, 2, 103]);
        assert_eq!(b.phase(), BlockPhase::Pass);
        assert!(b.restage_known(), "a re-decode stages its known prefix again");
        assert!(b.stats.redecoded);
        // The re-decode accepts nothing more: the fill takes over.
        b.apply_pass(L, 2, result(Some(4)), &plain(), &mut Hooks::default());
        assert_eq!(b.stats.second_accepted, Some(0));
        assert_eq!(b.fixed().len(), 5);
    }

    #[test]
    fn a_late_rejection_goes_to_the_fill() {
        let mut b = BlockLogic::begin(0, vec![], L, false, false);
        b.apply_pass(L, 2, result(Some(7)), &plain(), &mut Hooks::default());
        // Position 7 was the last: nothing left to fill.
        assert_eq!(b.phase(), BlockPhase::Done);
        let mut b = BlockLogic::begin(0, vec![], L, false, false);
        b.apply_pass(L, 3, result(Some(6)), &plain(), &mut Hooks::default());
        assert_eq!(b.phase(), BlockPhase::Fill, "fewer than redecode_min positions remain");
    }

    #[test]
    fn fill_writes_one_token_and_stops_on_a_stop_token() {
        let mut b = BlockLogic::begin(0, vec![1, 2], L, false, true);
        assert_eq!(b.phase(), BlockPhase::Fill);
        assert_eq!(b.next_position(), 2);
        let o = opts();
        let mut row = || vec![0.0f32];
        let mut draw = |_: &[f32]| 0;
        b.apply_fill(L, 9, &mut row, &mut draw, &o, &mut Hooks::default(), false);
        assert_eq!(b.fixed(), &[1, 2, 9]);
        assert_eq!(b.phase(), BlockPhase::Fill);
        b.apply_fill(L, -1, &mut row, &mut draw, &o, &mut Hooks::default(), false);
        assert_eq!(b.phase(), BlockPhase::Done);
        let out = b.into_out();
        assert!(out.stopped);
        assert_eq!(out.tokens, vec![9, -1]);
        assert_eq!(out.stats.ar_tokens, 2);
    }

    #[test]
    fn a_switch_token_rechecks_the_same_draft_in_verified_mode() {
        let mut b = BlockLogic::begin(0, vec![], L, false, false);
        // The draft is 0..8: token 5 is the switch-on token; the pass accepted everything.
        let effects = b.apply_pass(L, 2, result(None), &opts(), &mut Hooks::default());
        assert!(effects.verified_span);
        assert_eq!(effects.stage_draft, Some((0..L as i32).collect()));
        assert_eq!(b.stats.switched, Some(5));
        assert_eq!(b.phase(), BlockPhase::Pass);
        assert_eq!(b.mode(&opts()), PassMode::Recheck);
        assert_eq!(b.fixed().len(), 6, "kept up to and including the switch token");
    }

    #[test]
    fn the_grammar_scan_cuts_the_block_to_the_fill() {
        let mut b = BlockLogic::begin(0, vec![], L, false, false);
        let mut scan = |t: &[i32]| t.len().min(2);
        let mut hooks = Hooks { scan: Some(&mut scan), ..Default::default() };
        b.apply_pass(L, 2, result(None), &opts(), &mut hooks);
        assert_eq!(b.fixed(), &[0, 1]);
        assert_eq!(b.phase(), BlockPhase::Fill);
    }

    #[test]
    fn the_token_limit_stops_the_block() {
        let mut b = BlockLogic::begin(0, vec![], 3, false, false);
        b.apply_pass(L, 2, result(None), &plain(), &mut Hooks::default());
        assert_eq!(b.phase(), BlockPhase::Done);
        let out = b.into_out();
        assert_eq!(out.tokens.len(), 3);
        assert!(out.stopped);
    }

    #[test]
    fn draw_rows_fill_greedy_for_other_rows() {
        let mine = DrawSettings {
            codes: vec![vec![DrawRow { temperature: 0.5, ..DrawRow::GREEDY }; 2]],
            tokens: vec![DrawRow { temperature: 0.5, ..DrawRow::GREEDY }; 3],
        };
        let rows = DrawSettings::rows(&[(1, &mine)], 3, &[2], 3);
        assert_eq!(rows.tokens.len(), 9);
        assert_eq!(rows.codes[0].len(), 6);
        assert_eq!(rows.tokens[0], DrawRow::GREEDY);
        assert_eq!(rows.tokens[3].temperature, 0.5);
        assert!(DrawSettings::rows(&[], 2, &[2], 3).is_greedy());
    }
}
