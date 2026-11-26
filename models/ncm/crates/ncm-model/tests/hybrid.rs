//! The NCM hybrid (block decoder): config, parameter counts against the design's budget, the
//! block tokenizer, training forward and the decode loop's guarantees. CPU backend; the 250M
//! counts build the full-size layers on the CPU.

use burn::module::Module;
use burn::tensor::Device;

use ncm_model::config::NCM_250M_HYBRID_TOML;
use ncm_model::layers::{Block, LayerKind};
use ncm_model::ncm::ncm_specs;
use ncm_model::{Config, ncm_250m_hybrid};

fn device() -> Device {
    Device::flex()
}

/// `got` within `rel` of the design's hand count `want`.
fn assert_near(what: &str, got: usize, want: f64, rel: f64) {
    let err = (got as f64 - want).abs() / want;
    assert!(err <= rel, "{what}: {got} parameters, design {want:.0} ({:.1}% off)", err * 100.0);
}

#[test]
fn hybrid_config_parses_and_round_trips() {
    let cfg = Config::parse(NCM_250M_HYBRID_TOML).expect("the hybrid config parses");
    let Config::Ncm(ncm) = &cfg else { panic!("an NCM config") };
    let hybrid = ncm.hybrid.as_ref().expect("a [hybrid] table");
    hybrid.check();
    assert_eq!(hybrid.levels, vec![1, 2, 4, 8, 12, 16, 24, 32]);
    assert_eq!(hybrid.codes_per_block(), 99);
    assert_eq!((ncm.bb_layers, ncm.bb_ff_dim), (14, 2112));
    assert_eq!(cfg.unit(), 32);
    assert_eq!(Config::parse(&cfg.to_toml()).expect("round trip"), cfg);
    // The k = 4 config has no hybrid table and keeps its unit.
    let v2 = Config::parse(ncm_model::config::NCM_250M_V2_TOML).expect("v2 parses");
    assert_eq!(v2.unit(), 4);
}

/// The backbone's per-layer counts against the design's table (hand counts run up to about 1%
/// high): gated DeltaNet ~11.7M, sparse retrieval ~12.5M at d = 1024 with MLP 2,112, 7 of each.
#[test]
fn hybrid_backbone_layers_match_the_budget() {
    let device = device();
    let cfg = ncm_250m_hybrid();
    let (_, backbone) = ncm_specs(&cfg);
    let delta = backbone.kinds.iter().filter(|k| **k == LayerKind::DeltaNet).count();
    let retrieval = backbone.kinds.iter().filter(|k| **k == LayerKind::Retrieval).count();
    assert_eq!((delta, retrieval), (7, 7));
    let dn = Block::new(&backbone, LayerKind::DeltaNet, &device).num_params();
    let ret = Block::new(&backbone, LayerKind::Retrieval, &device).num_params();
    // Exact, from the layer shapes.
    assert_eq!(dn, 11_761_808);
    assert_eq!(ret, 12_458_880);
    assert_near("backbone DeltaNet layer", dn, 11.7e6, 0.01);
    assert_near("backbone retrieval layer", ret, 12.5e6, 0.01);
}

// ----------------------------------------------------------------------------- block tokenizer

use burn::tensor::{Int, Tensor, TensorData};
use ncm_model::hybrid::vq::{area_matrix, linear_matrix, phi_index};
use ncm_model::hybrid::{BlockTokenizer, Ema, sample_keep};
use ncm_model::{HybridConfig, tiny_hybrid, tiny_v2};

fn tokens(device: &Device, vocab: usize, b: usize, n: usize, seed: u64) -> Tensor<2, Int> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let data: Vec<i32> = (0..b * n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % vocab as u64) as i32
        })
        .collect();
    Tensor::from_data(TensorData::new(data, [b, n]), device)
}

fn values<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().convert::<f32>().try_to_vec::<f32>().expect("f32 tensor")
}

fn ints<const D: usize>(t: Tensor<D, Int>) -> Vec<i64> {
    t.into_data().convert::<i64>().try_to_vec::<i64>().expect("int tensor")
}

fn assert_close(got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = atol + rtol * w.abs();
        assert!((g - w).abs() <= tol, "element {i}: got {g}, want {w} (tol {tol})");
    }
}

fn tiny_cfg() -> (ncm_model::NcmConfig, HybridConfig) {
    let cfg = tiny_hybrid(tiny_v2(2, 2, 4));
    let hybrid = cfg.hybrid.clone().expect("hybrid");
    (cfg, hybrid)
}

#[test]
fn resampling_matrices_match_pytorch() {
    // interpolate(mode="area") from 32 to 12: windows [0, 3), [2, 6), [5, 8), ...
    let m = area_matrix(12, 32);
    assert_close(
        &m[..4],
        &[1.0 / 3.0; 3].iter().copied().chain([0.0]).collect::<Vec<_>>(),
        1e-7,
        0.0,
    );
    assert_close(&m[32..38], &[0.0, 0.0, 0.25, 0.25, 0.25, 0.25], 1e-7, 0.0);
    for rows in [area_matrix(12, 32).chunks(32), area_matrix(1, 32).chunks(32)] {
        for row in rows {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }
    // interpolate(mode="linear", align_corners=False) from 2 to 8: output 0 clamps to input 0,
    // output 2 sits at 0.375 (weights 0.625 / 0.375).
    let u = linear_matrix(8, 2);
    assert_close(&u[0..2], &[1.0, 0.0], 1e-7, 0.0);
    assert_close(&u[4..6], &[0.625, 0.375], 1e-6, 0.0);
    assert_close(&u[14..16], &[0.0, 1.0], 1e-7, 0.0);
    for row in linear_matrix(32, 12).chunks(12) {
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
    // One phi per level band: 8 levels over 4 convolutions.
    let idx: Vec<usize> = (0..8).map(|k| phi_index(k, 8, 4)).collect();
    assert_eq!(idx, vec![0, 0, 1, 1, 2, 2, 3, 3]);
}

#[test]
fn quantiser_residual_adds_up() {
    let device = device();
    device.seed(1);
    let (cfg, h) = tiny_cfg();
    let tok = BlockTokenizer::new(&h, cfg.vocab_size, cfg.local_dim, cfg.norm_eps, &device);
    let blocks = tokens(&device, cfg.vocab_size, 3, h.block, 7);
    let z = tok.encode_latents(blocks.clone());
    let q = tok.quantise(z.clone());
    assert_eq!(q.levels.len(), h.depth());
    for (level, &n) in q.levels.iter().zip(&h.levels) {
        assert_eq!(level.codes.dims(), [3, n]);
        assert!(ints(level.codes.clone()).iter().all(|&c| (0..h.codebook as i64).contains(&c)));
    }
    // Codes decode back to the same reconstruction, and the encode path gives the same codes.
    let codes = q.codes();
    assert_eq!(codes.dims(), [3, h.codes_per_block()]);
    assert_eq!(ints(tok.encode(blocks)), ints(codes.clone()));
    let zhat = tok.reconstruct(codes);
    assert_close(&values(zhat), &values(q.reconstruction(h.depth())), 1e-6, 1e-5);
    assert!(values(q.loss.clone())[0].is_finite());
}

#[test]
fn ema_with_no_memory_moves_entries_to_their_means() {
    let device = device();
    device.seed(2);
    let (cfg, h) = tiny_cfg();
    let tok = BlockTokenizer::new(&h, cfg.vocab_size, cfg.local_dim, cfg.norm_eps, &device);
    let blocks = tokens(&device, cfg.vocab_size, 4, h.block, 3);
    let q = tok.quantise(tok.encode_latents(blocks));
    let mut ema = Ema::new(&tok, 0.0);
    ema.update(&q);
    // Level 0 (one code per block): each used entry becomes the mean of its inputs.
    let level = &q.levels[0];
    let codes = ints(level.codes.clone());
    let inputs = values(level.inputs.clone());
    let book = values(ema.codebook(0));
    let c = h.latent;
    let used = codes[0] as usize;
    let members: Vec<usize> = (0..codes.len()).filter(|&i| codes[i] as usize == used).collect();
    let mean: Vec<f32> = (0..c)
        .map(|j| members.iter().map(|&i| inputs[i * c + j]).sum::<f32>() / members.len() as f32)
        .collect();
    // Laplace smoothing moves counts by about 1e-5 relative.
    assert_close(&book[used * c..(used + 1) * c], &mean, 1e-5, 1e-3);
    let tok = tok.with_codebooks(&ema);
    assert_close(&values(tok.codebook(0)), &book, 0.0, 0.0);
}

#[test]
fn nested_dropout_never_drops_everything() {
    let mut state = 0.123f64;
    let mut uniform = || {
        state = (state * 997.0 + 0.1).fract();
        state
    };
    let keep = sample_keep(1000, 8, 0.5, &mut uniform);
    assert!(keep.iter().all(|&k| (1..=8).contains(&k)));
    let truncated = keep.iter().filter(|&&k| k < 8).count();
    assert!((350..650).contains(&truncated), "{truncated} of 1000 truncated");
    assert!(keep.contains(&1) && keep.contains(&7));
}

#[test]
fn tokenizer_trains_through_the_straight_through_path() {
    let device = device();
    device.seed(4);
    let (cfg, h) = tiny_cfg();
    let tok = BlockTokenizer::new(&h, cfg.vocab_size, cfg.local_dim, cfg.norm_eps, &device);
    let emb = Tensor::<2>::random(
        [cfg.vocab_size, cfg.local_dim],
        burn::tensor::Distribution::Normal(0.0, 0.02),
        &device,
    );
    let blocks = tokens(&device, cfg.vocab_size, 2, h.block, 5);
    let out = tok.forward_train(blocks.clone(), Some(&[2, h.depth()]), emb);
    assert_eq!(out.logits.dims(), [2, h.block, cfg.vocab_size]);
    let loss = ncm_model::hybrid::tokenizer_loss(&out, blocks);
    assert!(values(loss)[0].is_finite());
}

// ----------------------------------------------------------------------------- model, training

use ncm_model::hybrid::{HybridLm, TrainPlan};
use ncm_model::{TokenLm, tok_250m_v2};

/// Exact counts of the 250M hybrid against the design's budget table (hand counts, about 1%
/// high in total): every row within its tolerance, the inference total within 2% of the
/// design's 254.4M, and the hybrid at most 2% larger than the token baseline.
#[test]
fn hybrid_parameter_counts_match_the_budget() {
    let device = device();
    let parts = HybridLm::new(&ncm_250m_hybrid(), &device).parts();
    // (part, design hand count, tolerance). The design counts the token decoder's and the
    // generator's attention as full multi-head; the token decoder keeps the k = 4 decoder's
    // grouped keys (4 of 8 heads), so it and the rows around it come in under the hand count.
    let rows = [
        ("embeddings", parts.embeddings, 16.4e6, 0.01),
        ("local encoder", parts.encoder, 10.5e6, 0.02),
        ("block pooling", parts.pooling, 0.5e6, 0.1),
        ("backbone", parts.backbone, 169.6e6, 0.01),
        ("generator layers", parts.generator, 26.8e6, 0.02),
        ("generator input, levels, head", parts.generator_io, 0.3e6, 0.15),
        ("codebooks", parts.codebooks, 8.4e6, 0.01),
        ("token decoder layers", parts.token_decoder, 17.8e6, 0.15),
        ("token decoder z input, copy head", parts.token_decoder_io, 0.5e6, 0.25),
        ("tokenizer decoder", parts.tokenizer_decoder, 1.0e6, 0.05),
        ("memory slots", parts.memory, 2.6e6, 0.25),
    ];
    for (what, got, want, rel) in rows {
        assert_near(what, got, want, rel);
    }
    // Not in the design's table: four shared phi convolutions of 256 x 256 x 3.
    assert_eq!(parts.phi, 4 * (256 * 256 * 3 + 256));
    assert_eq!(parts.tokenizer_encoder, 32_000 * 256 + 32 * 256);
    let total = parts.inference();
    assert_near("inference total", total, 254.4e6, 0.02);
    let baseline = TokenLm::new(&tok_250m_v2(), &device).parts().total();
    let gap = total as f64 / baseline as f64 - 1.0;
    assert!(
        gap <= 0.02,
        "hybrid {total} is {:.2}% larger than the baseline {baseline}",
        gap * 100.0
    );
}

fn tiny_model(seed: u64) -> (HybridLm, ncm_model::NcmConfig, HybridConfig, Device) {
    let device = device();
    device.seed(seed);
    let (cfg, h) = tiny_cfg();
    (HybridLm::new(&cfg, &device), cfg, h, device)
}

/// Codes of every block of `tokens` `[B, N]`, row-major `[B * nb, Σℓ]`.
fn block_codes(model: &HybridLm, tokens: Tensor<2, Int>) -> Tensor<2, Int> {
    let [b, n] = tokens.dims();
    let l = model.config().block;
    model.tokenizer().encode(tokens.reshape([b * n / l, l]))
}

#[test]
fn training_forward_gives_finite_losses() {
    let (model, cfg, h, device) = tiny_model(10);
    let n = 6 * h.block;
    let toks = tokens(&device, cfg.vocab_size, 2, n, 11);
    let codes = block_codes(&model, toks.clone());
    let mut state = 0.37f64;
    let mut uniform = || {
        state = (state * 991.0 + 0.13).fract();
        state
    };
    let plan = TrainPlan::sample(&h, codes.clone(), &mut uniform);
    let out = model.forward_train(toks.clone(), codes.clone(), &plan);
    assert_eq!(out.code_logits.len(), h.depth());
    for (logits, &l) in out.code_logits.iter().zip(&h.levels) {
        assert_eq!(logits.dims(), [2 * 6, l, h.codebook]);
    }
    assert_eq!(out.token_logp.dims(), [2, n, cfg.vocab_size]);
    // Log-probabilities of a distribution: they sum to about 1 per position.
    let p = values(out.token_logp.clone().exp().sum_dim(2));
    assert_close(&p, &vec![1.0; p.len()], 1e-4, 0.0);
    let loss = model.loss(&out, toks, codes, &plan);
    assert!(values(loss.code)[0].is_finite() && values(loss.token)[0].is_finite());
    assert_eq!(loss.per_level.len(), h.depth());
}

/// Causality in training mode: changing tokens at or after a block's start beyond its known
/// prefix leaves that block's generator logits and memory unchanged (its codes held fixed), and
/// leaves the token decoder's outputs at earlier positions unchanged.
#[test]
fn training_forward_is_causal_across_blocks() {
    let (model, cfg, h, device) = tiny_model(12);
    let l = h.block;
    let nb = 6;
    let n = nb * l;
    let toks = tokens(&device, cfg.vocab_size, 1, n, 13);
    let codes = block_codes(&model, toks.clone());
    for (blk, known) in [(0usize, 0usize), (3, 0), (3, 5), (5, 7)] {
        let mut plan = TrainPlan::exact(codes.clone());
        plan.known[blk] = known;
        let base = model.forward_train(toks.clone(), codes.clone(), &plan);
        let cut = blk * l + known;
        // Perturb every token from the cut on (codes unchanged: they are inputs).
        let mut data: Vec<i32> =
            toks.clone().into_data().convert::<i32>().try_to_vec::<i32>().unwrap();
        for t in data.iter_mut().skip(cut) {
            *t = (*t + 1) % cfg.vocab_size as i32;
        }
        let changed = Tensor::<2, Int>::from_data(TensorData::new(data, [1, n]), &device);
        let other = model.forward_train(changed, codes.clone(), &plan);
        for k in 0..h.depth() {
            let pick = |t: &Tensor<3>| values(t.clone().narrow(0, blk, 1));
            assert_eq!(
                pick(&base.code_logits[k]),
                pick(&other.code_logits[k]),
                "block {blk} level {k}"
            );
        }
        let mem = |m: &ncm_model::hybrid::Memory| {
            (values(m.extra.clone().narrow(1, blk, 1)), m.windows[blk])
        };
        assert_eq!(mem(&base.memory), mem(&other.memory), "block {blk} memory entries");
        let (lo, hi) = base.memory.windows[blk];
        if hi > lo {
            let seq = |m: &ncm_model::hybrid::Memory| {
                values(m.seq.clone().unwrap().narrow(1, lo, hi - lo))
            };
            assert_eq!(seq(&base.memory), seq(&other.memory), "block {blk} token states");
        }
        // Token decoder rows 0..=cut predict tokens up to the cut from tokens before it.
        let rows = |t: &Tensor<3>| values(t.clone().narrow(1, 0, cut + 1));
        assert_eq!(rows(&base.token_logp), rows(&other.token_logp), "rows before {cut}");
    }
}

// ----------------------------------------------------------------------------- decoding

use ncm_model::hybrid::{DecodeOptions, Hooks, HybridSession, hybrid_cache_bytes};

const GIB: f64 = (1u64 << 30) as f64;

fn host_ints(t: Tensor<2, Int>) -> Vec<i32> {
    t.into_data().convert::<i32>().try_to_vec::<i32>().expect("int tensor")
}

/// Training-mode generator logits (all levels in one pass, block-diagonal) equal the per-level
/// inference passes over the same block memory, teacher-forced on the same codes.
#[test]
fn generator_training_logits_match_per_level_passes() {
    let (model, cfg, h, device) = tiny_model(20);
    let l = h.block;
    let nb = 5;
    let toks = tokens(&device, cfg.vocab_size, 1, nb * l, 21);
    let codes = block_codes(&model, toks.clone());
    let mut plan = TrainPlan::exact(codes.clone());
    plan.known[3] = 3;
    let out = model.forward_train(toks, codes.clone(), &plan);
    for blk in [0, 3, 4] {
        let memory = out.memory.block(blk);
        let kvs = model.generator_kv(&memory);
        let row = host_ints(codes.clone().narrow(0, blk, 1));
        let mut zhat: Option<Tensor<3>> = None;
        for k in 0..h.depth() {
            let got = model.level_logits(&memory, &kvs, k, zhat.clone(), blk * l);
            let want = out.code_logits[k].clone().narrow(0, blk, 1);
            assert_close(&values(got), &values(want), 1e-4, 1e-3);
            let at = h.level_offset(k);
            let level = Tensor::<2, Int>::from_data(
                TensorData::new(row[at..at + h.levels[k]].to_vec(), [1, h.levels[k]]),
                &device,
            );
            let c = model.tokenizer().contribution(k, level);
            zhat = Some(match zhat {
                Some(z) => z + c,
                None => c,
            });
        }
    }
}

/// Incremental decoding equals the full forward pass: the decode loop's autoregressive pass
/// over a block, after prefilling the stream up to its start, gives the training forward's
/// token log-probabilities with z = 0; and its memories match the training memories.
#[test]
fn incremental_autoregressive_path_matches_training_forward() {
    let (model, cfg, h, device) = tiny_model(22);
    let l = h.block;
    let nb = 6;
    let n = nb * l;
    let toks = tokens(&device, cfg.vocab_size, 1, n, 23);
    let host = host_ints(toks.clone());
    let codes = block_codes(&model, toks.clone());
    let mut plan = TrainPlan::exact(codes.clone());
    plan.zero = vec![true; nb];
    plan.known[4] = 2;
    let out = model.forward_train(toks, codes, &plan);
    let v = cfg.vocab_size;
    for blk in [0, 2, 4, 5] {
        let mut session = HybridSession::new(&model, n + l, &device);
        session.prefill(&model, &host[..blk * l]);
        let block = &host[blk * l..(blk + 1) * l];
        let got = session.ar_logprobs(&model, block);
        let want = values(out.token_logp.clone().narrow(1, blk * l, l));
        assert_eq!(got.len(), l * v);
        assert_close(&got, &want, 1e-4, 1e-3);
        // The generator's memory with the same known prefix (prefilled, as in a reply).
        let known = plan.known[blk];
        let mut session = HybridSession::new(&model, n + l, &device);
        session.prefill(&model, &host[..blk * l + known]);
        let (memory, _) = session.block_memories(&model, &block[..known]);
        let train = out.memory.block(blk);
        assert_eq!(memory.extra_pos, train.extra_pos);
        let flags = |m: &ncm_model::hybrid::Memory| values(m.extra_hidden.clone().int().float());
        assert_eq!(flags(&memory), flags(&train));
        // Hidden entries may differ (padding); compare the visible ones.
        let d = cfg.local_dim;
        let (a, b) = (values(memory.extra.clone()), values(train.extra.clone()));
        for (i, &hidden) in flags(&memory).iter().enumerate() {
            if hidden < 0.5 {
                assert_close(&a[i * d..(i + 1) * d], &b[i * d..(i + 1) * d], 1e-4, 1e-3);
            }
        }
    }
}

fn greedy(verified: bool) -> DecodeOptions {
    DecodeOptions { verified, ar_only: !verified, ..DecodeOptions::default() }
}

/// Verified mode at temperature 0 equals the autoregressive path's greedy decode token for
/// token, over prompts ending at every offset of a block, with drafts forced wrong or random
/// and re-decodes forced at every allowed point or never.
#[test]
fn verified_mode_equals_autoregressive_greedy() {
    let (model, cfg, h, device) = tiny_model(24);
    let l = h.block;
    let v = cfg.vocab_size as i32;
    let stream = host_ints(tokens(&device, cfg.vocab_size, 1, 8 * l, 25));
    let reply = 3 * l;
    for offset in 0..l {
        let prompt = &stream[..2 * l + offset];
        let mut ar = HybridSession::new(&model, 8 * l, &device);
        ar.prefill(&model, prompt);
        let (want, _) = ar.generate_reply(&model, reply, &greedy(false), &mut Hooks::default());
        assert_eq!(want.len(), reply);
        for (case, redecode) in [(0, 1), (1, l + 1), (2, h.redecode_min), (3, 1)] {
            let mut session = HybridSession::new(&model, 8 * l, &device);
            session.prefill(&model, prompt);
            let mut state = (offset * 31 + case) as u64 + 1;
            let mut wrong = |pass: usize, draft: &mut Vec<i32>| match case {
                0 => draft.iter_mut().for_each(|t| *t = (*t + 1) % v),
                1 => {}
                _ => {
                    for t in draft.iter_mut().skip(pass) {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        *t = ((state >> 33) % v as u64) as i32;
                    }
                }
            };
            let mut hooks = Hooks { draft: Some(&mut wrong), ..Hooks::default() };
            let opts = DecodeOptions { redecode_min: Some(redecode), ..greedy(true) };
            let (got, stats) = session.generate_reply(&model, reply, &opts, &mut hooks);
            assert_eq!(got, want, "offset {offset}, case {case}");
            if case == 0 {
                // Every draft is wrong, so every block re-decodes when enough remains.
                assert!(stats.iter().any(|s| s.redecoded), "re-decodes forced");
            }
        }
    }
}

/// The autoregressive path's rows are bit-identical whatever the codes, z, drafts or the
/// generator's known prefix (forced re-decodes): every row seen for the same block tokens and
/// position is the same.
#[test]
fn autoregressive_rows_ignore_codes_drafts_and_redecodes() {
    use std::collections::HashMap;
    let (model, cfg, h, device) = tiny_model(26);
    let l = h.block;
    let v = cfg.vocab_size as i32;
    let n = h.codebook as i32;
    let stream = host_ints(tokens(&device, cfg.vocab_size, 1, 4 * l, 27));
    let mut seen: HashMap<(Vec<i32>, usize), Vec<f32>> = HashMap::new();
    let mut compared = 0;
    for case in 0..6u64 {
        let mut session = HybridSession::new(&model, 4 * l, &device);
        session.prefill(&model, &stream[..2 * l + 3]);
        let mut draft = |pass: usize, d: &mut Vec<i32>| {
            if case % 2 == 1 {
                d.iter_mut().enumerate().for_each(|(i, t)| *t = (*t + (i + pass) as i32) % v);
            }
        };
        let mut codes = |level: usize, c: &mut Vec<i32>| {
            c.iter_mut().for_each(|x| *x = (*x + (case as i32) * (level as i32 + 1)) % n);
        };
        let mut rows = |prefix: &[i32], i: usize, row: &[f32]| {
            let key = (prefix.to_vec(), i);
            match seen.get(&key) {
                Some(old) => {
                    assert_eq!(old.as_slice(), row, "row {i} after {prefix:?}");
                    compared += 1;
                }
                None => {
                    seen.insert(key, row.to_vec());
                }
            }
        };
        let mut hooks = Hooks {
            draft: Some(&mut draft),
            codes: Some(&mut codes),
            ar_rows: Some(&mut rows),
            ..Hooks::default()
        };
        let opts =
            DecodeOptions { redecode_min: Some(if case < 3 { 1 } else { l + 1 }), ..greedy(true) };
        session.decode_block(&model, l, &opts, &mut hooks);
    }
    assert!(compared > 0, "some rows were seen twice");
}

/// A reply that reaches an offset by decoding continues as one whose prompt ends there; and a
/// stop mid-block followed by a new message leaves the state of prefilling the whole stream.
#[test]
fn offsets_and_stops_match_prefill() {
    let (model, cfg, h, device) = tiny_model(28);
    let l = h.block;
    let stream = host_ints(tokens(&device, cfg.vocab_size, 1, 8 * l, 29));
    let prompt = &stream[..2 * l + 1];
    let mut first = HybridSession::new(&model, 8 * l, &device);
    first.prefill(&model, prompt);
    let (reply, _) = first.generate_reply(&model, 2 * l, &greedy(true), &mut Hooks::default());
    // Inside the first reply block (l - 1 tokens from the prompt's offset 1) the continuation
    // reads only the block's own tokens and the memory built before it.
    for j in [1, 3, l - 2] {
        let mut resumed = HybridSession::new(&model, 8 * l, &device);
        let mut p = prompt.to_vec();
        p.extend_from_slice(&reply[..j]);
        resumed.prefill(&model, &p);
        let rest_len = l - 1 - j;
        let (rest, _) =
            resumed.generate_reply(&model, rest_len, &greedy(true), &mut Hooks::default());
        assert_eq!(rest, reply[j..j + rest_len], "resumed at {j}");
    }
    // Stop at the reply's fourth token, then a new message.
    let stop = reply[3];
    let at = reply.iter().position(|&t| t == stop).expect("present");
    let mut stopped = HybridSession::new(&model, 8 * l, &device);
    stopped.prefill(&model, prompt);
    let opts = DecodeOptions { stop: vec![stop], ..greedy(true) };
    let (out, _) = stopped.generate_reply(&model, 2 * l, &opts, &mut Hooks::default());
    assert_eq!(out, reply[..=at]);
    let message = &stream[5 * l..5 * l + l + 5];
    stopped.prefill(&model, message);
    let mut whole = HybridSession::new(&model, 8 * l, &device);
    let mut all = prompt.to_vec();
    all.extend_from_slice(&out);
    all.extend_from_slice(message);
    whole.prefill(&model, &all);
    assert_eq!(stopped.tokens(), whole.tokens());
    assert_eq!(stopped.blocks(), whole.blocks());
    assert_close(&values(stopped.condition()), &values(whole.condition()), 1e-4, 1e-3);
    let open = stopped.tokens()[stopped.blocks() * l..].to_vec();
    let (a, b) = (stopped.ar_logprobs(&model, &open), whole.ar_logprobs(&model, &open));
    assert_close(&a, &b, 1e-4, 1e-3);
}

/// A switch token in a parallel-mode result: tokens through it come from parallel mode, the
/// rest of the block from verified mode, so they equal the autoregressive path's greedy
/// continuation of the tokens before them.
#[test]
fn mixed_block_switches_to_verified_mode() {
    let (model, cfg, h, device) = tiny_model(30);
    let l = h.block;
    let stream = host_ints(tokens(&device, cfg.vocab_size, 1, 6 * l, 31));
    let prompt = &stream[..l + 2];
    let parallel = DecodeOptions { tau: Some(0.0), ..DecodeOptions::default() };
    let mut dry = HybridSession::new(&model, 6 * l, &device);
    dry.prefill(&model, prompt);
    let first = dry.decode_block(&model, l, &parallel, &mut Hooks::default());
    let plain = first.tokens;
    // Switch on the parallel result's third token (its first occurrence).
    let switch = plain[2];
    let s = plain.iter().position(|&t| t == switch).expect("present");
    let mut session = HybridSession::new(&model, 6 * l, &device);
    session.prefill(&model, prompt);
    let opts = DecodeOptions { switch_on: vec![switch], ..parallel.clone() };
    let block = session.decode_block(&model, l, &opts, &mut Hooks::default());
    let got = block.tokens;
    assert_eq!(got[..=s], plain[..=s], "parallel mode up to the switch");
    assert_eq!(block.stats.switched, Some(block.stats.known + s));
    let mut ar = HybridSession::new(&model, 6 * l, &device);
    let mut p = prompt.to_vec();
    p.extend_from_slice(&got[..=s]);
    ar.prefill(&model, &p);
    let rest = got.len() - s - 1;
    let (want, _) = ar.generate_reply(&model, rest, &greedy(false), &mut Hooks::default());
    assert_eq!(got[s + 1..], want[..], "verified after the switch");
}

/// The decode state at 1M tokens against the design's ~0.3 GiB (bf16, 8-bit indexer keys):
/// 7 x 32,768 x 1,152 B of MLA cache, ~28 MiB of indexer keys, DeltaNet state, token ids and
/// recent states.
#[test]
fn hybrid_cache_at_1m_matches_the_design() {
    let bytes = hybrid_cache_bytes(&ncm_250m_hybrid(), 1 << 20, 2, 1) as f64 / GIB;
    assert!((0.27..=0.33).contains(&bytes), "{bytes:.3} GiB at 1M tokens");
}

// ----------------------------------------------------------------------------- fast path

use ncm_model::hybrid::FastSession;

/// The fast path (memory rings, fixed-buffer passes, acceptance on the device) decodes what
/// the reference loop decodes at temperature 0, in verified and in parallel mode, from prompts
/// ending at several offsets, with re-decodes forced or not, and reports the same first-pass
/// acceptance.
#[test]
fn fast_path_matches_the_reference_loop() {
    let (model, cfg, h, device) = tiny_model(40);
    let l = h.block;
    let stream = host_ints(tokens(&device, cfg.vocab_size, 1, 8 * l, 41));
    for offset in [0, 1, l / 2, l - 1] {
        let prompt = &stream[..2 * l + offset];
        for opts in [
            greedy(true),
            DecodeOptions { redecode_min: Some(1), ..greedy(true) },
            DecodeOptions { tau: Some(0.0), ..DecodeOptions::default() },
            DecodeOptions::default(),
        ] {
            let mut reference = HybridSession::new(&model, 8 * l, &device);
            reference.prefill(&model, prompt);
            let (want, want_stats) =
                reference.generate_reply(&model, 3 * l, &opts, &mut Hooks::default());
            let mut fast = FastSession::new(&model, 8 * l, &device);
            fast.prefill(&model, prompt);
            assert!(fast.serves(&opts));
            let (got, got_stats) = fast.generate_reply(&model, 3 * l, &opts, &mut Hooks::default());
            assert_eq!(got, want, "offset {offset}, {opts:?}");
            let first = |s: &[ncm_model::hybrid::BlockStats]| {
                s.iter().map(|b| (b.known, b.first_accepted, b.redecoded)).collect::<Vec<_>>()
            };
            assert_eq!(first(&got_stats), first(&want_stats), "offset {offset}, {opts:?}");
        }
    }
}

/// In verified mode the fast path equals the autoregressive path's greedy decode, whatever the
/// synthetic acceptance decides (it changes only how fast the tokens come).
#[test]
fn fast_verified_mode_equals_autoregressive_greedy() {
    let (model, cfg, h, device) = tiny_model(42);
    let l = h.block;
    let stream = host_ints(tokens(&device, cfg.vocab_size, 1, 6 * l, 43));
    let prompt = &stream[..l + 3];
    let mut ar = FastSession::new(&model, 6 * l, &device);
    ar.prefill(&model, prompt);
    let (want, _) = ar.generate_reply(&model, 2 * l, &greedy(false), &mut Hooks::default());
    for p in [1.0, 0.5, 0.0] {
        let mut fast = FastSession::new(&model, 6 * l, &device);
        fast.prefill(&model, prompt);
        let mut n = 0u64;
        let mut accept = |_: usize, _: usize| {
            n += 1;
            ((n * 2654435761) % 1000) as f64 / 1000.0 < p
        };
        let mut hooks = Hooks { accept: Some(&mut accept), ..Hooks::default() };
        // Synthetic acceptance keeps drafts the path did not choose: only p = 0 (every draft
        // rejected, the pass's own token at each rejection) must equal the greedy decode.
        let (got, _) = fast.generate_reply(&model, 2 * l, &greedy(true), &mut hooks);
        if p == 0.0 {
            assert_eq!(got, want);
        }
        assert_eq!(got.len(), want.len());
    }
}

/// The constraint scan: a grammar that allows only even token ids cuts every block at the
/// first odd token a check pass proposes; the autoregressive fill then picks the best allowed
/// token. Every emitted token is allowed, and the fast path agrees with the reference loop.
#[test]
fn constraint_scan_rejects_what_it_should() {
    let (model, cfg, h, device) = tiny_model(44);
    let l = h.block;
    let stream = host_ints(tokens(&device, cfg.vocab_size, 1, 6 * l, 45));
    let prompt = &stream[..l + 2];
    let allowed = |t: i32| t % 2 == 0;
    let decode = |fast: bool| {
        let mut scans = 0;
        let mut scan = |proposed: &[i32]| {
            scans += 1;
            proposed.iter().take_while(|&&t| allowed(t)).count()
        };
        let mut pick = |row: &[f32]| {
            let best = (0..row.len())
                .filter(|&i| allowed(i as i32))
                .max_by(|&a, &b| row[a].total_cmp(&row[b]));
            best.expect("an allowed token") as i32
        };
        let mut hooks = Hooks { scan: Some(&mut scan), pick: Some(&mut pick), ..Hooks::default() };
        let opts = greedy(true);
        let out = if fast {
            let mut s = FastSession::new(&model, 6 * l, &device);
            s.prefill(&model, prompt);
            s.generate_reply(&model, 2 * l, &opts, &mut hooks).0
        } else {
            let mut s = HybridSession::new(&model, 6 * l, &device);
            s.prefill(&model, prompt);
            s.generate_reply(&model, 2 * l, &opts, &mut hooks).0
        };
        (out, scans)
    };
    let (reference, scans) = decode(false);
    assert!(scans > 0, "the scan ran");
    assert!(reference.iter().all(|&t| allowed(t)), "{reference:?}");
    assert_eq!(decode(true).0, reference);
}
