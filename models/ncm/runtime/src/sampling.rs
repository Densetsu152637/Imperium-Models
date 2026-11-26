//! Choosing the next token from logits: greedy, or temperature with top-k and top-p, with an
//! optional [`Constraint`] mask.

use crate::constrain::Constraint;

/// Most allowed candidates a constrained step looks at before the top-k and top-p cuts.
const CONSTRAINED_CANDIDATES: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    /// 0 (or below) is greedy.
    pub temperature: f32,
    /// 0 keeps every token.
    pub top_k: usize,
    /// 1 keeps every token.
    pub top_p: f32,
    pub seed: Option<u64>,
}

impl SamplingParams {
    pub const GREEDY: SamplingParams =
        SamplingParams { temperature: 0.0, top_k: 0, top_p: 1.0, seed: None };

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

impl Default for SamplingParams {
    /// llama-server's defaults, so the two runtimes answer alike when nothing is asked.
    fn default() -> Self {
        Self { temperature: 0.8, top_k: 40, top_p: 0.95, seed: None }
    }
}

/// xorshift64*: small, fast, and good enough to draw a token.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // splitmix64 so that small seeds do not start in a degenerate corner of xorshift.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self((z ^ (z >> 31)) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub struct Sampler {
    params: SamplingParams,
    rng: Rng,
}

/// A fresh seed when the request names none.
pub fn entropy() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0x5EED, |d| d.as_nanos() as u64)
}

fn clean(logit: f32) -> f32 {
    if logit.is_nan() { f32::NEG_INFINITY } else { logit }
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        Self { params, rng: Rng::new(params.seed.unwrap_or_else(entropy)) }
    }

    pub fn is_greedy(&self) -> bool {
        self.params.is_greedy()
    }

    /// The token to take, or `None` when the constraint allows no token at all.
    pub fn pick(&mut self, logits: &[f32], constraint: Option<&dyn Constraint>) -> Option<u32> {
        if self.params.is_greedy() {
            return match constraint {
                None => argmax(logits),
                Some(mask) => ranked(logits).find(|&id| mask.allows(id)),
            };
        }
        let limit = match self.params.top_k {
            0 => usize::MAX,
            k => k,
        };
        let mut candidates: Vec<(u32, f32)> = match constraint {
            None => top(logits, limit.min(logits.len())),
            Some(mask) => ranked(logits)
                .filter(|&id| mask.allows(id))
                .take(limit.min(CONSTRAINED_CANDIDATES))
                .map(|id| (id, clean(logits[id as usize])))
                .collect(),
        };
        candidates.retain(|c| c.1.is_finite());
        let &(best, max) = candidates.first()?;
        let temperature = self.params.temperature;
        let mut weights: Vec<f32> =
            candidates.iter().map(|c| ((c.1 - max) / temperature).exp()).collect();
        let total: f32 = weights.iter().sum();
        // Top-p: the smallest prefix whose probability mass reaches p.
        let mut keep = candidates.len();
        if self.params.top_p < 1.0 {
            let mut mass = 0.0;
            for (i, w) in weights.iter().enumerate() {
                mass += w / total;
                if mass >= self.params.top_p {
                    keep = i + 1;
                    break;
                }
            }
        }
        weights.truncate(keep);
        let total: f32 = weights.iter().sum();
        let mut draw = self.rng.unit() * total;
        for (candidate, weight) in candidates.iter().zip(&weights) {
            draw -= weight;
            if draw < 0.0 {
                return Some(candidate.0);
            }
        }
        Some(best)
    }
}

/// The index of the largest logit.
pub fn argmax(logits: &[f32]) -> Option<u32> {
    logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, clean(l)))
        .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
        .map(|(i, _)| i as u32)
}

/// The `k` largest logits, best first.
fn top(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut all: Vec<(u32, f32)> =
        logits.iter().enumerate().map(|(i, &l)| (i as u32, clean(l))).collect();
    if k < all.len() {
        all.select_nth_unstable_by(k, |a, b| b.1.total_cmp(&a.1));
        all.truncate(k);
    }
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all
}

/// Every id, best logit first (ties by id).
fn ranked(logits: &[f32]) -> impl Iterator<Item = u32> {
    let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
    ids.sort_by(|&a, &b| {
        clean(logits[b as usize]).total_cmp(&clean(logits[a as usize])).then(a.cmp(&b))
    });
    ids.into_iter()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Only(Vec<u32>);

    impl Constraint for Only {
        fn allows(&self, token: u32) -> bool {
            self.0.contains(&token)
        }

        fn advance(&mut self, _: u32) {}
    }

    #[test]
    fn greedy_takes_the_largest_logit() {
        let mut sampler = Sampler::new(SamplingParams::GREEDY);
        assert_eq!(sampler.pick(&[0.1, 2.0, -1.0, 1.9], None), Some(1));
        assert_eq!(argmax(&[f32::NAN, 1.0, 1.0]), Some(1), "first of equals, NaN last");
    }

    #[test]
    fn greedy_respects_the_constraint() {
        let mut sampler = Sampler::new(SamplingParams::GREEDY);
        let mask = Only(vec![0, 2]);
        assert_eq!(sampler.pick(&[0.1, 2.0, 1.0, 1.9], Some(&mask)), Some(2));
        assert_eq!(sampler.pick(&[0.1, 2.0], Some(&Only(vec![]))), None);
    }

    #[test]
    fn a_seed_makes_sampling_repeatable() {
        let logits: Vec<f32> = (0..50).map(|i| (i % 7) as f32 * 0.3).collect();
        let params = SamplingParams { temperature: 1.0, top_k: 0, top_p: 1.0, seed: Some(7) };
        let run = |params| {
            let mut sampler = Sampler::new(params);
            (0..20).map(|_| sampler.pick(&logits, None).unwrap()).collect::<Vec<_>>()
        };
        assert_eq!(run(params), run(params));
        assert_ne!(run(params), run(SamplingParams { seed: Some(8), ..params }));
    }

    #[test]
    fn top_k_and_top_p_cut_the_tail() {
        let logits = [5.0, 4.9, -20.0, -20.0, -20.0];
        for seed in 0..50 {
            let params =
                SamplingParams { temperature: 1.0, top_k: 2, top_p: 1.0, seed: Some(seed) };
            assert!(Sampler::new(params).pick(&logits, None).unwrap() < 2);
        }
        // With p small enough only the best token is kept.
        let params = SamplingParams { temperature: 1.0, top_k: 0, top_p: 0.3, seed: Some(1) };
        for _ in 0..20 {
            assert_eq!(Sampler::new(params).pick(&logits, None), Some(0));
        }
    }

    #[test]
    fn sampling_only_returns_allowed_tokens() {
        let logits = [5.0, 4.9, 1.0, 0.5];
        let mask = Only(vec![2, 3]);
        for seed in 0..50 {
            let params =
                SamplingParams { temperature: 0.8, top_k: 40, top_p: 0.95, seed: Some(seed) };
            let token = Sampler::new(params).pick(&logits, Some(&mask)).unwrap();
            assert!(token == 2 || token == 3);
        }
    }

    #[test]
    fn a_tiny_temperature_is_nearly_greedy() {
        let logits = [1.0, 3.0, 2.0];
        let params = SamplingParams { temperature: 0.01, top_k: 0, top_p: 1.0, seed: Some(3) };
        for _ in 0..20 {
            assert_eq!(Sampler::new(params).pick(&logits, None), Some(1));
        }
    }
}
