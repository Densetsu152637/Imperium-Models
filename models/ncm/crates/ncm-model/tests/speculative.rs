//! Self-speculative decoding: the draft head and the parallel verify pass against plain
//! step-by-step greedy decoding. CPU backend, tiny configs of both attention designs.

use burn::tensor::{Device, Int, Tensor, TensorData};
use ncm_model::speculative::accept_or_resample;
use ncm_model::{
    Acceptance, NcmCache, NcmConfig, NcmLm, Speculator, draft_loss, tiny_ncm, tiny_ncm_v2, tiny_v2,
};

const PROMPT: usize = 24;
const CONCEPTS: usize = 10;

fn device() -> Device {
    Device::flex()
}

fn designs() -> [NcmConfig; 2] {
    [tiny_ncm(8, 4), tiny_ncm_v2(8, tiny_v2(2, 2, 4))]
}

fn tokens(device: &Device, vocab: usize, n: usize, seed: u64) -> Tensor<2, Int> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let data: Vec<i32> = (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % vocab as u64) as i32
        })
        .collect();
    Tensor::from_data(TensorData::new(data, [1, n]), device)
}

fn ints(t: Tensor<2, Int>) -> Vec<i64> {
    t.into_data().convert::<i64>().try_to_vec::<i64>().expect("ints")
}

fn values<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().expect("f32 tensor")
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
}

/// Writes one concept: `(model, cond, cache, plain)`, where `plain` is the concept plain
/// decoding writes.
type Concept<'a> = dyn FnMut(&NcmLm, Tensor<3>, &mut NcmCache, &[i64]) -> Vec<i64> + 'a;

/// Greedy decode of `CONCEPTS` concepts after a random prompt, each from `concept`.
fn generate(model: &NcmLm, cfg: &NcmConfig, seed: u64, concept: &mut Concept<'_>) -> Vec<i64> {
    let device = device();
    let prompt = tokens(&device, cfg.vocab_size, PROMPT, seed);
    let mut cache = model.make_cache(PROMPT + cfg.k * CONCEPTS, 1, &device);
    let mut cond = model.prefill(prompt, &mut cache, 8);
    let mut out = Vec::new();
    for _ in 0..CONCEPTS {
        let (plain, _) = model.decode_concept(cond.clone(), &mut cache, None, false);
        let plain = ints(plain);
        let got = concept(model, cond.clone(), &mut cache, &plain);
        let next = Tensor::from_data(
            TensorData::new(got.iter().map(|&t| t as i32).collect::<Vec<_>>(), [1, cfg.k]),
            &device,
        );
        cond = model.append_tokens(next, &mut cache);
        out.extend(got);
    }
    out
}

fn plain(model: &NcmLm, cfg: &NcmConfig, seed: u64) -> Vec<i64> {
    generate(model, cfg, seed, &mut |_, _, _, plain| plain.to_vec())
}

#[test]
fn speculative_greedy_equals_plain_greedy() {
    for cfg in designs() {
        for seed in 0..6 {
            let device = device();
            device.seed(seed);
            let model = NcmLm::new(&cfg, &device);
            let want = plain(&model, &cfg, seed);
            let mut spec = Speculator::new(Acceptance::Verify);
            let got = generate(&model, &cfg, seed, &mut |m, cond, cache, _| {
                ints(spec.decode(m, cond, cache))
            });
            assert_eq!(got, want, "{} seed {seed}", cfg.attention.name());
            let stats = spec.stats();
            assert_eq!(stats.concepts, CONCEPTS);
            assert_eq!(stats.judged, CONCEPTS * cfg.k);
            assert!((CONCEPTS..=CONCEPTS * cfg.k).contains(&stats.verify_passes));
        }
    }
}

/// Batched self-speculation over slots at different positions equals each slot's plain greedy
/// concept, and the slot draft and verify passes equal the single-stream ones.
#[test]
fn slot_speculation_equals_plain_greedy() {
    for cfg in designs() {
        let device = device();
        device.seed(3);
        let model = NcmLm::new(&cfg, &device);
        let mut caches = Vec::new();
        let mut conds = Vec::new();
        for (i, len) in [2 * cfg.k, 5 * cfg.k, 3 * cfg.k].into_iter().enumerate() {
            let mut cache = model.make_cache(len + cfg.k * CONCEPTS, 1, &device);
            conds.push(model.prefill(
                tokens(&device, cfg.vocab_size, len, i as u64),
                &mut cache,
                8,
            ));
            caches.push(cache);
        }
        for step in 0..4 {
            let want: Vec<i64> = conds
                .iter()
                .zip(caches.iter_mut())
                .flat_map(|(c, cache)| ints(model.decode_concept(c.clone(), cache, None, false).0))
                .collect();
            let cond = Tensor::cat(conds.clone(), 0);
            let refs: Vec<&NcmCache> = caches.iter().collect();
            let draft = values(model.draft_slots(cond.clone(), &refs));
            let single: Vec<f32> = conds
                .iter()
                .zip(&caches)
                .flat_map(|(c, cache)| values(model.draft_concept(c.clone(), cache)))
                .collect();
            assert!(max_diff(&draft, &single) < 1e-4, "{} draft step {step}", cfg.attention.name());
            let mut stats = ncm_model::SpecStats::default();
            let got = ncm_model::speculate_slots(&model, cond, &refs, &mut stats);
            assert_eq!(got, want, "{} step {step}", cfg.attention.name());
            assert_eq!(stats.concepts, 3);
            for (r, cache) in caches.iter_mut().enumerate() {
                let next: Vec<i32> =
                    got[r * cfg.k..(r + 1) * cfg.k].iter().map(|&t| t as i32).collect();
                let next = Tensor::from_data(TensorData::new(next, [1, cfg.k]), &device);
                conds[r] = model.append_tokens(next, cache);
            }
        }
    }
}

/// Drafts that are right, right up to slot `good`, or wrong everywhere: the output is plain
/// greedy decoding every time, and a concept takes one verify pass per rejected slot before
/// the last, plus one.
#[test]
fn forced_full_partial_and_zero_acceptance_equal_plain_greedy() {
    for cfg in designs() {
        let k = cfg.k;
        for good in [k, 2, 1, 0] {
            for seed in 0..3 {
                let device = device();
                device.seed(seed);
                let model = NcmLm::new(&cfg, &device);
                let want = plain(&model, &cfg, seed);
                let mut spec = Speculator::new(Acceptance::Verify);
                let vocab = cfg.vocab_size as i64;
                let got = generate(&model, &cfg, seed, &mut |m, cond, cache, plain| {
                    let draft: Vec<i32> = plain
                        .iter()
                        .enumerate()
                        .map(|(j, &t)| if j < good { t } else { (t + 1) % vocab } as i32)
                        .collect();
                    let draft = Tensor::from_data(TensorData::new(draft, [1, k]), &cond.device());
                    ints(spec.decode_from(m, cond, cache, draft))
                });
                assert_eq!(got, want, "{} draft good to {good}", cfg.attention.name());
                let stats = spec.stats();
                let passes = 1 + (good..k - 1).count();
                assert_eq!(stats.verify_passes, CONCEPTS * passes, "good to {good}");
                assert_eq!(stats.accepted, CONCEPTS * good);
            }
        }
    }
}

#[test]
fn assumed_acceptance_runs_the_matching_passes() {
    let cfg = tiny_ncm(8, 4);
    let device = device();
    device.seed(0);
    let model = NcmLm::new(&cfg, &device);
    let mut cache = model.make_cache(64, 1, &device);
    let cond = model.prefill(tokens(&device, cfg.vocab_size, PROMPT, 1), &mut cache, 8);
    // Close to independent decisions: a concept takes one pass plus one per rejection before
    // slot k-1.
    let k = cfg.k as f64;
    for rate in [1.0, 0.75, 0.5, 0.0] {
        let mut spec = Speculator::new(Acceptance::Assume(rate));
        for _ in 0..200 {
            spec.decode(&model, cond.clone(), &cache);
        }
        let stats = spec.stats();
        assert!((stats.acceptance() - rate).abs() < 0.01, "{rate}: {stats:?}");
        let passes = 1.0 + (k - 1.0) * (1.0 - rate);
        assert!((stats.passes_per_concept() - passes).abs() < 0.2, "{rate}: {stats:?}");
    }
}

/// With one-hot distributions speculative sampling is greedy verification.
#[test]
fn speculative_sampling_with_one_hot_distributions_is_greedy() {
    for cfg in designs() {
        let device = device();
        device.seed(4);
        let model = NcmLm::new(&cfg, &device);
        let want = plain(&model, &cfg, 4);
        let mut spec = Speculator::new(Acceptance::Verify);
        let mut one_hot = |_: usize, logits: &[f32]| {
            let best = (0..logits.len()).fold(0, |b, i| if logits[i] > logits[b] { i } else { b });
            (0..logits.len()).map(|i| (i == best) as u32 as f32).collect()
        };
        let mut state = 7u64;
        let mut uniform = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        let got = generate(&model, &cfg, 4, &mut |m, cond, cache, _| {
            ints(spec.decode_sampled(m, cond, cache, &mut one_hot, &mut uniform))
        });
        assert_eq!(got, want, "{}", cfg.attention.name());
    }
}

/// Drawing from q and then accepting or resampling gives draws distributed as p.
#[test]
fn speculative_acceptance_samples_the_target_distribution() {
    let p = [0.5f32, 0.2, 0.0, 0.25, 0.05];
    let q = [0.1f32, 0.3, 0.3, 0.2, 0.1];
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut uniform = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let n = 200_000;
    let mut counts = [0usize; 5];
    for _ in 0..n {
        let u = uniform();
        let mut acc = 0.0;
        let x = q.iter().position(|&w| {
            acc += w;
            u < acc
        });
        let x = x.unwrap_or(4);
        let token = accept_or_resample(&p, &q, x, uniform(), uniform()).unwrap_or(x);
        counts[token] += 1;
    }
    for (i, (&c, &want)) in counts.iter().zip(&p).enumerate() {
        let got = c as f32 / n as f32;
        assert!((got - want).abs() < 0.006, "token {i}: {got} vs {want}");
    }
}

/// The verify pass gives the decoder's teacher-forced logits, as the full forward pass does;
/// the draft head gives the same logits incrementally as in the training forward.
#[test]
fn verify_and_draft_passes_match_the_training_forward() {
    for cfg in designs() {
        let device = device();
        device.seed(2);
        let model = NcmLm::new(&cfg, &device);
        let (k, n, vocab) = (cfg.k, 48, cfg.vocab_size);
        let toks = tokens(&device, vocab, n, 5);
        let (logits, draft) = model.forward_with_draft(toks.clone());
        assert_eq!(logits.dims(), [1, n - k, vocab]);
        assert_eq!(draft.dims(), [1, n - k, vocab]);
        let (logits, draft) = (values(logits), values(draft));
        let mut cache = model.make_cache(n, 1, &device);
        let mut cond = model.prefill(toks.clone().narrow(1, 0, PROMPT), &mut cache, 8);
        for start in (PROMPT..n).step_by(k) {
            let concept = toks.clone().narrow(1, start, k);
            let at = (start - k) * vocab..(start - k + k) * vocab;
            let verify = values(model.verify_concept(cond.clone(), &cache, concept.clone()));
            assert!(max_diff(&verify, &logits[at.clone()]) < 1e-4, "{}", cfg.attention.name());
            let drafted = values(model.draft_concept(cond.clone(), &cache));
            assert!(max_diff(&drafted, &draft[at]) < 1e-4, "{}", cfg.attention.name());
            if start + k < n {
                cond = model.append_tokens(concept, &mut cache);
            }
        }
    }
}

#[test]
fn draft_loss_is_the_cross_entropy_per_slot() {
    let device = device();
    device.seed(3);
    let cfg = tiny_ncm(8, 4);
    let model = NcmLm::new(&cfg, &device);
    let (k, n, vocab) = (cfg.k, 32, cfg.vocab_size);
    let toks = tokens(&device, vocab, n, 9);
    let (_, draft) = model.forward_with_draft(toks.clone());
    let loss = draft_loss(draft.clone(), toks.clone(), k);
    assert_eq!(loss.per_slot.dims(), [k]);
    let per_slot = values(loss.per_slot);
    let mean = values(loss.mean)[0];
    // By hand: -log softmax at the target, averaged per slot.
    let (logits, targets) = (values(draft), ints(toks));
    let mut want = vec![0f64; k];
    for t in 0..n - k {
        let row = &logits[t * vocab..(t + 1) * vocab];
        let max = row.iter().fold(f32::MIN, |m, &v| m.max(v)) as f64;
        let lse = max + row.iter().map(|&v| (v as f64 - max).exp()).sum::<f64>().ln();
        want[t % k] += lse - row[targets[t + k] as usize] as f64;
    }
    for (slot, (got, want)) in per_slot.iter().zip(&want).enumerate() {
        let want = want / ((n - k) / k) as f64;
        assert!((*got as f64 - want).abs() < 1e-4, "slot {slot}: {got} vs {want}");
    }
    let avg = per_slot.iter().sum::<f32>() / k as f32;
    assert!((mean - avg).abs() < 1e-5);
    // Random weights over 256 tokens: about ln 256.
    assert!((mean - (vocab as f32).ln()).abs() < 1.0, "{mean}");
}

#[test]
fn draft_head_is_optional() {
    let device = device();
    let cfg = NcmConfig { draft_layers: 0, ..tiny_ncm(8, 4) };
    let model = NcmLm::new(&cfg, &device);
    assert!(!model.has_draft_head());
    assert_eq!(model.parts().draft, 0);
    let with = NcmLm::new(&tiny_ncm(8, 4), &device);
    assert!(with.has_draft_head());
    assert_eq!(with.parts().total() - with.parts().draft, model.parts().total());
}
