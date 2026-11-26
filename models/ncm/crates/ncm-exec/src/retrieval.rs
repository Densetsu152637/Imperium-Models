//! The host half of sparse retrieval at decode (`docs/plans/npu.md`, decision 4): the indexer
//! scores over the cached 8-bit indexer keys, the block maxima, the top-k blocks, and the
//! gather of their latent entries plus the local window. It mirrors
//! `ncm_model::retrieval::SparseRetrieval::attend_selected` (one query, batch 1) so the split
//! graphs (`pre` on the accelerator, this on the CPU, `attend` on the accelerator) pick what
//! the Burn reference picks. UNVERIFIED: written from that code without a run.

/// Code range of an 8-bit indexer key (`KEY_CODE_MAX`).
const KEY_CODE_MAX: f32 = 127.0;

/// Blocks a query at `pos` may retrieve: those starting before its local window.
pub fn candidates(pos: usize, block: usize, local: usize) -> usize {
    if pos >= local { (pos - local) / block + 1 } else { 0 }
}

/// One indexer key as 8-bit codes and a scale: `scale = max |k| / 127` (floored at 1e-30),
/// `code = round(k / scale)` clamped to the range (`retrieval::quantize_keys`).
pub fn quantize_key(key: &[f32]) -> (Vec<i8>, f32) {
    let max = key.iter().fold(0f32, |m, v| m.max(v.abs()));
    let scale = (max / KEY_CODE_MAX).max(1e-30);
    let codes =
        key.iter().map(|v| (v / scale).round().clamp(-KEY_CODE_MAX, KEY_CODE_MAX) as i8).collect();
    (codes, scale)
}

/// A retrieval layer's cached indexer keys, one row per position.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IndexCache {
    dim: usize,
    codes: Vec<i8>,
    scales: Vec<f32>,
}

impl IndexCache {
    pub fn new(dim: usize) -> Self {
        Self { dim, codes: Vec::new(), scales: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.scales.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scales.is_empty()
    }

    /// Keeps the first `n` positions (a rewind).
    pub fn truncate(&mut self, n: usize) {
        self.codes.truncate(n * self.dim);
        self.scales.truncate(n);
    }

    /// Stores the key of position `pos` (positions arrive in order; a re-written position
    /// replaces the tail from there).
    pub fn put(&mut self, pos: usize, key: &[f32]) {
        assert_eq!(key.len(), self.dim, "an indexer key row");
        self.truncate(pos);
        let (codes, scale) = quantize_key(key);
        self.codes.extend(codes);
        self.scales.push(scale);
    }

    /// The indexer scores of the first `n` positions for one query: per head `j`,
    /// `relu(q_j . c_s) * w_j / sqrt(heads * dim)`, summed over heads, times the position's
    /// scale (`SparseRetrieval::index_scores`). `q` is `[heads * dim]`, `w` `[heads]`.
    pub fn scores(&self, q: &[f32], w: &[f32], n: usize) -> Vec<f32> {
        let heads = w.len();
        let dim = self.dim;
        let norm = 1.0 / ((heads * dim) as f32).sqrt();
        (0..n)
            .map(|s| {
                let code = &self.codes[s * dim..(s + 1) * dim];
                let mut sum = 0f32;
                for j in 0..heads {
                    let qj = &q[j * dim..(j + 1) * dim];
                    let dot: f32 = qj.iter().zip(code).map(|(a, &c)| a * f32::from(c)).sum();
                    sum += dot.max(0.0) * (w[j] * norm);
                }
                sum * self.scales[s]
            })
            .collect()
    }
}

/// What the host gathers for one retrieval layer and query.
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    /// The cache positions of the entries to gather, in order: the picked blocks'
    /// positions (`topk * block`), then the local window (`local`).
    pub positions: Vec<usize>,
    /// The additive mask over those entries: a picked position that is inside the local
    /// window is hidden (the window already holds it), everything else is visible.
    pub bias: Vec<f32>,
}

/// The selection for a query at `pos`, or None when every candidate block would be selected
/// (`candidates <= topk`: the whole causal context is visible, so the dense graph is exact).
/// `scores` are the indexer scores of positions `0..candidates * block`.
pub fn select(
    pos: usize,
    block: usize,
    topk: usize,
    local: usize,
    scores: &[f32],
) -> Option<Selection> {
    let n = candidates(pos, block, local);
    if n <= topk {
        return None;
    }
    assert!(scores.len() >= n * block, "scores for every candidate position");
    // Block scores: the maximum inside each block.
    let blocks: Vec<f32> = (0..n)
        .map(|b| {
            scores[b * block..(b + 1) * block].iter().copied().fold(f32::NEG_INFINITY, f32::max)
        })
        .collect();
    // The top-k blocks, best first; ties go to the earlier block.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| blocks[b].total_cmp(&blocks[a]).then(a.cmp(&b)));
    order.truncate(topk);
    let mut positions: Vec<usize> = Vec::with_capacity(topk * block + local);
    for &b in &order {
        positions.extend(b * block..(b + 1) * block);
    }
    let mut bias: Vec<f32> =
        positions.iter().map(|&p| if p > pos - local { f32::NEG_INFINITY } else { 0.0 }).collect();
    // The local window: the `local` positions ending at `pos`.
    positions.extend(pos + 1 - local..pos + 1);
    bias.extend(std::iter::repeat_n(0.0, local));
    Some(Selection { positions, bias })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_are_the_blocks_before_the_window() {
        assert_eq!(candidates(5, 4, 16), 0);
        assert_eq!(candidates(16, 4, 16), 1);
        assert_eq!(candidates(35, 4, 16), 5);
    }

    #[test]
    fn keys_round_trip_within_half_a_step() {
        let key = [0.5, -1.0, 0.25, 0.0];
        let (codes, scale) = quantize_key(&key);
        assert_eq!(codes[1], -127);
        for (c, k) in codes.iter().zip(&key) {
            assert!((f32::from(*c) * scale - k).abs() <= scale / 2.0 + 1e-7);
        }
        let (zero, s) = quantize_key(&[0.0; 4]);
        assert!(zero.iter().all(|&c| c == 0) && s > 0.0);
    }

    #[test]
    fn scores_follow_the_indexer_formula() {
        let mut cache = IndexCache::new(2);
        cache.put(0, &[1.0, 0.0]);
        cache.put(1, &[-1.0, 0.0]);
        // One head, q = (2, 0), w = 1: relu(2) on position 0, relu(-2) = 0 on position 1.
        let s = cache.scores(&[2.0, 0.0], &[1.0], 2);
        let norm = 1.0 / 2f32.sqrt();
        assert!((s[0] - 2.0 * 127.0 * norm * (1.0 / 127.0)).abs() < 1e-5, "{s:?}");
        assert_eq!(s[1], 0.0);
    }

    #[test]
    fn a_rewrite_replaces_the_tail() {
        let mut cache = IndexCache::new(1);
        for p in 0..4 {
            cache.put(p, &[p as f32]);
        }
        cache.put(2, &[9.0]);
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn few_candidates_select_nothing() {
        // pos 20, block 4, local 16: one candidate block, topk 2: dense is exact.
        assert!(select(20, 4, 2, 16, &[0.0; 4]).is_none());
    }

    #[test]
    fn the_best_blocks_and_the_window_are_gathered() {
        let (block, topk, local, pos) = (2, 2, 4, 15);
        // candidates: (15 - 4) / 2 + 1 = 6 blocks over positions 0..12.
        let mut scores = vec![0.0f32; 12];
        scores[5] = 3.0; // block 2
        scores[8] = 5.0; // block 4
        scores[11] = 1.0; // block 5
        let sel = select(pos, block, topk, local, &scores).unwrap();
        assert_eq!(&sel.positions[..4], &[8, 9, 4, 5], "best block first");
        assert_eq!(&sel.positions[4..], &[12, 13, 14, 15], "the window ends at pos");
        assert!(sel.bias.iter().all(|&b| b == 0.0), "no picked block lies inside the window");
    }

    #[test]
    fn a_picked_position_inside_the_window_is_hidden() {
        // pos 13, local 5: three candidate blocks of 4 (positions 0..12); the last block
        // (8..12) starts before the window (>= 8) but reaches into it.
        let mut scores = vec![0.0f32; 12];
        scores[9] = 1.0;
        let sel = select(13, 4, 1, 5, &scores).unwrap();
        assert_eq!(&sel.positions[..4], &[8, 9, 10, 11]);
        assert_eq!(sel.bias[0], 0.0, "position 8 is not newer than pos - local = 8");
        assert!(sel.bias[1..4].iter().all(|b| b.is_infinite()), "9, 10, 11 are in the window");
        assert_eq!(&sel.positions[4..], &[9, 10, 11, 12, 13]);
    }

    #[test]
    fn ties_go_to_the_earlier_block() {
        let scores = vec![1.0f32; 12];
        let sel = select(15, 2, 2, 4, &scores).unwrap();
        assert_eq!(&sel.positions[..4], &[0, 1, 2, 3]);
    }
}
