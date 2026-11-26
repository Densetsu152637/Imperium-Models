//! Incremental decode equals a full forward pass; NCM causality; sizes. CPU backend, tiny
//! configs. Mirrors `tests/test_models.py` of the PyTorch reference.

use ncm_model::config::{NCM_250M_TOML, TOK_250M_TOML};
use ncm_model::{
    Config, LayerCache, NcmConfig, NcmLm, TokenLm, cache_bytes, ncm_250m, tiny_ncm, tiny_tok,
    tok_250m,
};

use burn::tensor::{Device, Int, Tensor, TensorData};

const GIB: f64 = (1u64 << 30) as f64;
const MIB: f64 = (1u64 << 20) as f64;

fn device() -> Device {
    Device::flex()
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

fn values<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().expect("f32 tensor")
}

fn assert_close(got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = atol + rtol * w.abs();
        assert!((g - w).abs() <= tol, "element {i}: got {g}, want {w} (tol {tol})");
    }
}

// ----------------------------------------------------------------------------- token baseline

fn tok_incremental(chunk: usize, window: usize, n: usize, prompt: usize) {
    let device = device();
    device.seed(0);
    let model = TokenLm::new(&tiny_tok(window), &device);
    let toks = tokens(&device, 256, n, 0);
    let full = values(model.forward(toks.clone(), None));

    // The prompt in chunks (T > 1 ring path), then single-token decode past the wrap.
    let mut caches = model.make_cache(n, 1, &device);
    let mut parts = Vec::new();
    let mut start = 0;
    while start < prompt {
        let len = chunk.min(prompt - start);
        parts.push(model.forward(toks.clone().narrow(1, start, len), Some(&mut caches)));
        start += len;
    }
    for t in prompt..n {
        parts.push(model.step(toks.clone().narrow(1, t, 1), &mut caches));
    }
    let got = values(Tensor::cat(parts, 1));
    assert_close(&got, &full, 2e-4, 1e-3);
}

#[test]
fn tok_incremental_matches_full() {
    // 40 tokens against window 8: the ring wraps several times.
    for chunk in [1, 3, 8, 13] {
        tok_incremental(chunk, 8, 40, 17);
    }
}

#[test]
fn tok_short_run_uses_plain_cache() {
    tok_incremental(1, 64, 20, 0);
}

// ----------------------------------------------------------------------------- NCM

fn ncm_incremental(
    model: &NcmLm,
    toks: &Tensor<2, Int>,
    k: usize,
    prompt: usize,
    chunk: usize,
) -> Vec<f32> {
    let device = toks.device();
    let n = toks.dims()[1];
    let mut cache = model.make_cache(n, 1, &device);
    let mut cond = model.prefill(toks.clone().narrow(1, 0, prompt), &mut cache, chunk);
    let mut outs = Vec::new();
    for start in (prompt..n).step_by(k) {
        let forced = toks.clone().narrow(1, start, k);
        let (_, logits) =
            model.decode_concept(cond.clone(), &mut cache, Some(forced.clone()), true);
        outs.push(logits.expect("logits"));
        if start + k < n {
            cond = model.append_tokens(forced, &mut cache);
        }
    }
    values(Tensor::cat(outs, 1))
}

#[test]
fn ncm_incremental_matches_full() {
    for chunk in [4, 12, 32] {
        let device = device();
        device.seed(0);
        // 64 tokens = 16 concepts: wraps the encoder ring (8 tokens) and backbone ring (4).
        let cfg = tiny_ncm(8, 4);
        let model = NcmLm::new(&cfg, &device);
        let toks = tokens(&device, 256, 64, 0);
        let prompt = 24;
        let full = model.forward(toks.clone()); // logits for tokens 4..63
        let vocab = cfg.vocab_size;
        let tail = full.narrow(1, prompt - cfg.k, 64 - prompt);
        assert_eq!(tail.dims(), [1, 64 - prompt, vocab]);
        let got = ncm_incremental(&model, &toks, cfg.k, prompt, chunk);
        assert_close(&got, &values(tail), 2e-4, 1e-3);
    }
}

#[test]
fn ncm_decode_greedy_is_consistent_with_teacher_forcing() {
    let device = device();
    device.seed(3);
    let cfg = tiny_ncm(8, 4);
    let model = NcmLm::new(&cfg, &device);
    let toks = tokens(&device, 256, 24, 5);
    let mut cache = model.make_cache(48, 1, &device);
    let mut cond = model.prefill(toks.clone(), &mut cache, 8);
    let mut generated = Vec::new();
    for _ in 0..3 {
        let (next, _) = model.decode_concept(cond, &mut cache, None, false);
        assert_eq!(next.dims(), [1, cfg.k]);
        generated.push(next.clone());
        cond = model.append_tokens(next, &mut cache);
    }
    // Running the whole sequence in one go must reproduce the greedy choices.
    let all = Tensor::cat([vec![toks], generated.clone()].concat(), 1);
    let logits = model.forward(all.clone());
    let n = all.dims()[1];
    let predicted = logits.argmax(2).reshape([1, n - cfg.k]);
    let want: Vec<i32> = all.narrow(1, 24, 12).into_data().try_to_vec().expect("ints");
    let got: Vec<i32> = predicted.narrow(1, 24 - cfg.k, 12).into_data().try_to_vec().expect("ints");
    assert_eq!(got, want);
}

#[test]
fn ncm_causality() {
    let device = device();
    device.seed(0);
    let cfg = tiny_ncm(8, 4);
    let model = NcmLm::new(&cfg, &device);
    let (k, n) = (cfg.k, 48);
    let base_tokens = tokens(&device, 256, n, 0);
    let (concept, pos) = (5, 2); // change token `pos` of concept 5
    let t_edit = concept * k + pos;
    let mut edited: Vec<i32> = base_tokens.clone().into_data().try_to_vec().expect("ints");
    edited[t_edit] = (edited[t_edit] + 1) % 256;
    let edited = Tensor::<2, Int>::from_data(TensorData::new(edited, [1, n]), &device);
    let vocab = cfg.vocab_size;
    let base = values(model.forward(base_tokens));
    let other = values(model.forward(edited));
    // logits[t - k] predicts token t. Concepts before the edit are untouched, and inside the
    // edited concept the decoder has only seen tokens before `pos`, so it predicts up to and
    // including token `pos` identically.
    let same = (t_edit + 1 - k) * vocab;
    assert_close(&other[..same], &base[..same], 1e-5, 1e-5);
    // Predictions after the edit are allowed to change, and do.
    let moved = other[same..].iter().zip(&base[same..]).any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(moved, "later predictions must depend on the edited token");
}

#[test]
fn ncm_conditioning_excludes_the_concept_being_written() {
    // The backbone vector used for concept m+1 comes from concepts <= m only.
    let device = device();
    device.seed(2);
    let cfg = tiny_ncm(8, 4);
    let model = NcmLm::new(&cfg, &device);
    let toks = tokens(&device, 256, 40, 0);
    let mut edited: Vec<i32> = toks.clone().into_data().try_to_vec().expect("ints");
    for t in &mut edited[20..24] {
        *t = (*t + 7) % 256; // the *next* concept
    }
    let edited = Tensor::<2, Int>::from_data(TensorData::new(edited, [1, 40]), &device);
    let mut c1 = model.make_cache(40, 1, &device);
    let cond_a = model.prefill(toks.narrow(1, 0, 20), &mut c1, 8);
    let mut c2 = model.make_cache(40, 1, &device);
    let cond_b = model.prefill(edited.narrow(1, 0, 20), &mut c2, 8);
    assert_close(&values(cond_a), &values(cond_b), 1e-6, 1e-6);
}

// ----------------------------------------------------------------------------- sizes

/// Parameter counts of the PyTorch build (`uv run ncm-bench --json`).
const TORCH_TOK_PARAMS: usize = 250_389_504;
const TORCH_NCM_PARAMS: usize = 240_298_496;

#[test]
fn full_config_param_counts() {
    let device = device();
    // Parameters are initialised lazily, so counting does not allocate 250M floats.
    let tok = TokenLm::new(&tok_250m(), &device).parts();
    let ncm = NcmLm::new(&ncm_250m(), &device).parts();
    // The PyTorch build had no draft head.
    let ncm_torch_parts = ncm.total() - ncm.draft;
    for (name, total, torch) in
        [("tok", tok.total(), TORCH_TOK_PARAMS), ("ncm", ncm_torch_parts, TORCH_NCM_PARAMS)]
    {
        assert!((225_000_000..=265_000_000).contains(&total), "{name}: {total}");
        let off = (total as f64 - torch as f64).abs() / torch as f64;
        assert!(off < 1e-3, "{name}: {total} vs PyTorch {torch}");
    }
    let m = |n: usize| (n as f64 / 1e5).round() / 10.0;
    assert_eq!(
        [
            m(ncm.embeddings),
            m(ncm.encoder),
            m(ncm.backbone),
            m(ncm.decoder),
            m(ncm.draft),
            m(ncm.projections)
        ],
        [16.4, 6.3, 199.3, 15.7, 3.7, 2.6]
    );
    assert_eq!(ncm.rows().iter().map(|r| r.1).sum::<usize>(), ncm.total());
    assert_eq!(tok.rows().iter().map(|r| r.1).sum::<usize>(), tok.total());
}

#[test]
fn cache_bytes_at_one_million_tokens() {
    let n = 1_048_576;
    let ncm = cache_bytes(&Config::Ncm(ncm_250m()), n, 2, 1) as f64;
    let tok = cache_bytes(&Config::Tok(tok_250m()), n, 2, 1) as f64;
    let (ncm_doc, tok_doc) = (GIB + 12.0 * MIB, 5.0 * GIB + 60.0 * MIB);
    assert!((ncm - ncm_doc).abs() / ncm_doc < 0.05, "ncm {}", ncm / GIB);
    assert!((tok - tok_doc).abs() / tok_doc < 0.05, "tok {}", tok / GIB);
    assert!(tok / ncm > 4.5);
}

#[test]
fn cache_bytes_match_allocation() {
    let device = device();
    for n in [20, 64] {
        let (ncm_cfg, tok_cfg) = (tiny_ncm(8, 4), tiny_tok(8));
        let ncm = NcmLm::new(&ncm_cfg, &device).make_cache(n, 1, &device);
        assert_eq!(ncm.nbytes(), cache_bytes(&Config::Ncm(ncm_cfg), n, 4, 1));
        let tok = TokenLm::new(&tok_cfg, &device).make_cache(n, 1, &device);
        let got: usize = tok.iter().map(LayerCache::nbytes).sum();
        assert_eq!(got, cache_bytes(&Config::Tok(tok_cfg), n, 4, 1));
    }
}

// ----------------------------------------------------------------------------- configs

#[test]
fn shipped_configs_parse() {
    assert!(matches!(Config::parse(TOK_250M_TOML), Ok(Config::Tok(_))));
    assert!(matches!(Config::parse(NCM_250M_TOML), Ok(Config::Ncm(_))));
}

#[test]
fn config_rejects_unknown_keys_and_kinds() {
    assert!(Config::parse("model = \"ncm\"\nbogus = 1\n").is_err());
    assert!(Config::parse("model = \"nope\"\n").is_err());
    assert!(Config::parse("k = 4\n").is_err());
    let cfg = Config::parse("model = \"ncm\"\nk = 8\n");
    assert_eq!(cfg.map(|c| c.unit()).ok(), Some(NcmConfig { k: 8, ..NcmConfig::default() }.k));
}

#[test]
fn ncm_warm_up_leaves_the_model_usable() {
    let device = Device::flex();
    let cfg = ncm_model::tiny_ncm_v2(8, ncm_model::tiny_v2(2, 2, 4));
    let model = ncm_model::NcmLm::new(&cfg, &device);
    model.warm_up(37, 16, &device);
    model.warm_up(1, 16, &device);
}

/// Batched prefill chunks of several slots at different positions equal each slot prefilled
/// alone, for both models in both designs.
#[test]
fn slot_prefill_equals_single_prefill() {
    let device = device();
    let v2 = ncm_model::tiny_v2(2, 2, 4);
    for cfg in [tiny_tok(8), ncm_model::tiny_tok_v2(ncm_model::tiny_v2(4, 2, 8))] {
        let model = TokenLm::new(&cfg, &device);
        let heads = [tokens(&device, cfg.vocab_size, 5, 1), tokens(&device, cfg.vocab_size, 11, 2)];
        let piece = tokens(&device, cfg.vocab_size, 12, 3).reshape([2, 6]);
        let mut single: Vec<Vec<LayerCache>> = Vec::new();
        let mut batched: Vec<Vec<LayerCache>> = Vec::new();
        for h in &heads {
            for set in [&mut single, &mut batched] {
                let mut c = model.make_cache(32, 1, &device);
                model.prefill(h.clone(), &mut c, 4);
                set.push(c);
            }
        }
        let want: Vec<f32> = (0..2)
            .flat_map(|i| {
                values(model.prefill(piece.clone().narrow(0, i, 1), &mut single[i], 4).unwrap())
            })
            .collect();
        let mut rows: Vec<&mut [LayerCache]> =
            batched.iter_mut().map(|c| c.as_mut_slice()).collect();
        let got = values(model.prefill_slots(piece, &mut rows, 4));
        assert_close(&got, &want, 1e-4, 1e-4);
    }
    for cfg in [tiny_ncm(8, 4), ncm_model::tiny_ncm_v2(8, v2)] {
        let model = NcmLm::new(&cfg, &device);
        let k = cfg.k;
        let heads =
            [tokens(&device, cfg.vocab_size, 2 * k, 4), tokens(&device, cfg.vocab_size, 5 * k, 5)];
        let piece = tokens(&device, cfg.vocab_size, 6 * k, 6).reshape([2, 3 * k]);
        let mut single = Vec::new();
        let mut batched = Vec::new();
        for h in &heads {
            for set in [&mut single, &mut batched] {
                let mut c = model.make_cache(16 * k, 1, &device);
                model.prefill(h.clone(), &mut c, 2 * k);
                set.push(c);
            }
        }
        let want: Vec<f32> = (0..2)
            .flat_map(|i| {
                values(model.prefill(piece.clone().narrow(0, i, 1), &mut single[i], 2 * k))
            })
            .collect();
        let mut rows: Vec<&mut ncm_model::NcmCache> = batched.iter_mut().collect();
        let got = values(model.prefill_slots(piece, &mut rows, 2 * k));
        assert_close(&got, &want, 1e-4, 1e-4);
    }
}

/// Batched slot steps (projections once over all rows, attention per row) equal each row
/// stepped alone, for both models in both designs: slots at different positions.
#[test]
fn slot_steps_equal_single_steps() {
    let device = device();
    let v2 = ncm_model::tiny_v2(2, 2, 4);
    for cfg in [tiny_tok(8), ncm_model::tiny_tok_v2(ncm_model::tiny_v2(4, 2, 8))] {
        let model = TokenLm::new(&cfg, &device);
        let prompts =
            [tokens(&device, cfg.vocab_size, 9, 1), tokens(&device, cfg.vocab_size, 14, 2)];
        let mut single: Vec<Vec<LayerCache>> = Vec::new();
        let mut batched: Vec<Vec<LayerCache>> = Vec::new();
        for p in &prompts {
            for set in [&mut single, &mut batched] {
                let mut c = model.make_cache(32, 1, &device);
                model.prefill(p.clone(), &mut c, 4);
                set.push(c);
            }
        }
        let next = Tensor::<2, Int>::from_data(TensorData::new(vec![3i32, 7], [2, 1]), &device);
        let want: Vec<f32> = (0..2)
            .flat_map(|i| values(model.step(next.clone().narrow(0, i, 1), &mut single[i])))
            .collect();
        let mut rows: Vec<&mut [LayerCache]> =
            batched.iter_mut().map(|c| c.as_mut_slice()).collect();
        let got = values(model.step_slots(next, &mut rows));
        assert_close(&got, &want, 1e-4, 1e-4);
    }
    for cfg in [tiny_ncm(8, 4), ncm_model::tiny_ncm_v2(8, v2)] {
        let model = NcmLm::new(&cfg, &device);
        let k = cfg.k;
        let prompts =
            [tokens(&device, cfg.vocab_size, 3 * k, 3), tokens(&device, cfg.vocab_size, 5 * k, 4)];
        let mut single = Vec::new();
        let mut batched = Vec::new();
        for p in &prompts {
            for set in [&mut single, &mut batched] {
                let mut c = model.make_cache(16 * k, 1, &device);
                model.prefill(p.clone(), &mut c, 2 * k);
                set.push(c);
            }
        }
        let concept = tokens(&device, cfg.vocab_size, 2 * k, 5).reshape([2, k]);
        let want: Vec<f32> = (0..2)
            .flat_map(|i| {
                values(model.append_tokens(concept.clone().narrow(0, i, 1), &mut single[i]))
            })
            .collect();
        let mut rows: Vec<&mut ncm_model::NcmCache> = batched.iter_mut().collect();
        let cond = model.append_slots(concept, &mut rows);
        assert_close(&values(cond.clone()), &want, 1e-4, 1e-4);
        // Greedy decoding of the next concept, batched against row by row.
        let mut greedy =
            |_: usize, logits: &Tensor<3>| logits.clone().argmax(2).reshape([logits.dims()[0], 1]);
        let got = model.decode_slots(cond.clone(), &mut rows, &mut greedy);
        let ints = |t: Tensor<2, Int>| t.into_data().convert::<i64>().try_to_vec::<i64>().unwrap();
        for (i, cache) in single.iter_mut().enumerate() {
            let row = cond.clone().narrow(0, i, 1);
            let (want, _) = model.decode_concept(row, cache, None, false);
            assert_eq!(ints(got.clone().narrow(0, i, 1)), ints(want));
        }
    }
}
