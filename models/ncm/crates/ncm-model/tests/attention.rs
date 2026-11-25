//! The v2 attention design: DeltaNet forms agree, incremental decode equals a full forward pass,
//! sparse retrieval with every block selected equals dense attention, NCM causality, sizes and
//! cache bytes. CPU backend, tiny configs.

use burn::module::Module;
use burn::tensor::activation::sigmoid;
use burn::tensor::{Device, Distribution, Int, Tensor, TensorData};
use ncm_model::config::{NCM_250M_V2_TOML, TOK_250M_V2_TOML};
use ncm_model::delta::{delta_rule_chunked, delta_rule_recurrent};
use ncm_model::{
    Config, Design, LayerCache, NcmConfig, NcmLm, TokConfig, TokenLm, V2Config, cache_bytes,
    cache_bytes_with, ncm_250m_v2, tiny_ncm_v2, tiny_tok_v2, tiny_v2, tok_250m_v2,
};

const GIB: f64 = (1u64 << 30) as f64;

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

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Retrieval over blocks of 4 tokens, top 2 blocks plus the last 8 tokens.
fn tok_v2() -> TokConfig {
    tiny_tok_v2(tiny_v2(4, 2, 8))
}

/// Retrieval over blocks of 2 concepts, top 2 blocks plus the last 4 concepts.
fn ncm_v2() -> NcmConfig {
    tiny_ncm_v2(8, tiny_v2(2, 2, 4))
}

// ----------------------------------------------------------------------------- gated DeltaNet

#[test]
fn delta_rule_chunked_matches_recurrent() {
    let device = device();
    device.seed(7);
    let (b, h, t, dk, dv) = (2, 3, 13, 8, 6);
    let normal = Distribution::Normal(0.0, 1.0);
    let unit = |x: Tensor<4>| {
        let n = x.clone().square().sum_dim(3).sqrt();
        x / n
    };
    let q = unit(Tensor::random([b, h, t, dk], normal, &device));
    let k = unit(Tensor::random([b, h, t, dk], normal, &device));
    let v = Tensor::<4>::random([b, h, t, dv], normal, &device);
    // Log decays in about [-1.5, 0]: some heads forget fast, some barely.
    let g = Tensor::<3>::random([b, h, t], Distribution::Uniform(-1.5, 0.0), &device);
    let beta = sigmoid(Tensor::<3>::random([b, h, t], normal, &device));
    let s0 = Tensor::<4>::random([b, h, dk, dv], normal, &device);
    let (want_o, want_s) =
        delta_rule_recurrent(q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), s0.clone());
    // Chunks that divide T, that leave a ragged tail, of one token, and longer than T.
    for chunk in [1, 4, 5, 13, 64] {
        let (o, s) = delta_rule_chunked(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            s0.clone(),
            chunk,
        );
        assert_close(&values(o), &values(want_o.clone()), 1e-4, 1e-4);
        assert_close(&values(s), &values(want_s.clone()), 1e-4, 1e-4);
    }
}

// ----------------------------------------------------------------------------- token baseline

fn tok_incremental(cfg: &TokConfig, chunk: usize, n: usize, prompt: usize) {
    let device = device();
    device.seed(0);
    let model = TokenLm::new(cfg, &device);
    let toks = tokens(&device, cfg.vocab_size, n, 0);
    let full = values(model.forward(toks.clone(), None));
    // The prompt in chunks (prefill path), then one-token steps (recurrent DeltaNet, gathered
    // retrieval blocks).
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
    assert_close(&values(Tensor::cat(parts, 1)), &full, 2e-4, 1e-3);
}

#[test]
fn tok_v2_incremental_matches_full() {
    // 40 tokens: 10 blocks of 4, so from position 16 on queries have more candidate blocks
    // than the 2 they keep; prefill chunks cross chunk, block and window boundaries.
    for chunk in [1, 3, 8, 13] {
        tok_incremental(&tok_v2(), chunk, 40, 17);
    }
}

#[test]
fn tok_v2_dense_incremental_matches_full() {
    let mut cfg = tok_v2();
    cfg.v2.ret_dense = true;
    tok_incremental(&cfg, 5, 30, 11);
}

/// Logits of `cfg` with the parameters of `model`.
fn tok_logits_as(model: &TokenLm, cfg: &TokConfig, toks: &Tensor<2, Int>) -> Vec<f32> {
    let device = toks.device();
    let other = TokenLm::new(cfg, &device).load_record(model.clone().into_record());
    values(other.forward(toks.clone(), None))
}

#[test]
fn tok_v2_sparse_with_every_block_equals_dense() {
    let device = device();
    device.seed(4);
    let cfg = tok_v2();
    let model = TokenLm::new(&cfg, &device);
    let toks = tokens(&device, cfg.vocab_size, 48, 3);
    let all = TokConfig { v2: V2Config { ret_topk: 1000, ..cfg.v2.clone() }, ..cfg.clone() };
    let dense = TokConfig { v2: V2Config { ret_dense: true, ..cfg.v2.clone() }, ..cfg.clone() };
    let want = tok_logits_as(&model, &dense, &toks);
    assert_close(&tok_logits_as(&model, &all, &toks), &want, 1e-5, 1e-5);
    // With 2 blocks per query the late positions see less, so the logits move.
    let sparse = tok_logits_as(&model, &cfg, &toks);
    assert!(max_diff(&sparse, &want) > 1e-4, "top-2 retrieval must differ from dense");
}

// ----------------------------------------------------------------------------- NCM

fn ncm_incremental(model: &NcmLm, toks: &Tensor<2, Int>, k: usize, prompt: usize, chunk: usize) {
    let device = toks.device();
    let n = toks.dims()[1];
    let want = values(model.forward(toks.clone()).narrow(1, prompt - k, n - prompt));
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
    assert_close(&values(Tensor::cat(outs, 1)), &want, 2e-4, 1e-3);
}

#[test]
fn ncm_v2_incremental_matches_full() {
    // 64 tokens = 16 concepts: 8 blocks of 2 against top 2 plus 4 recent concepts, and the
    // encoder ring (8 tokens) wraps.
    let cfg = ncm_v2();
    for chunk in [4, 12, 32] {
        let device = device();
        device.seed(0);
        let model = NcmLm::new(&cfg, &device);
        let toks = tokens(&device, cfg.vocab_size, 64, 0);
        ncm_incremental(&model, &toks, cfg.k, 24, chunk);
    }
}

#[test]
fn ncm_v2_dense_incremental_matches_full() {
    let mut cfg = ncm_v2();
    cfg.v2.ret_dense = true;
    let device = device();
    device.seed(1);
    let model = NcmLm::new(&cfg, &device);
    let toks = tokens(&device, cfg.vocab_size, 48, 2);
    ncm_incremental(&model, &toks, cfg.k, 20, 8);
}

#[test]
fn ncm_v2_sparse_with_every_block_equals_dense() {
    let device = device();
    device.seed(5);
    let cfg = ncm_v2();
    let model = NcmLm::new(&cfg, &device);
    let toks = tokens(&device, cfg.vocab_size, 64, 1);
    let logits_as = |cfg: &NcmConfig| {
        let other = NcmLm::new(cfg, &device).load_record(model.clone().into_record());
        values(other.forward(toks.clone()))
    };
    let all = NcmConfig { v2: V2Config { ret_topk: 1000, ..cfg.v2.clone() }, ..cfg.clone() };
    let dense = NcmConfig { v2: V2Config { ret_dense: true, ..cfg.v2.clone() }, ..cfg.clone() };
    let want = logits_as(&dense);
    assert_close(&logits_as(&all), &want, 1e-5, 1e-5);
    assert!(max_diff(&logits_as(&cfg), &want) > 1e-4, "top-2 retrieval must differ from dense");
}

#[test]
fn ncm_v2_causality() {
    let device = device();
    device.seed(0);
    let cfg = ncm_v2();
    let model = NcmLm::new(&cfg, &device);
    let (k, n) = (cfg.k, 48);
    let base_tokens = tokens(&device, cfg.vocab_size, n, 0);
    let (concept, pos) = (5, 2);
    let t_edit = concept * k + pos;
    let mut edited: Vec<i32> = base_tokens.clone().into_data().try_to_vec().expect("ints");
    edited[t_edit] = (edited[t_edit] + 1) % cfg.vocab_size as i32;
    let edited = Tensor::<2, Int>::from_data(TensorData::new(edited, [1, n]), &device);
    let base = values(model.forward(base_tokens));
    let other = values(model.forward(edited));
    // logits[t - k] predicts token t; everything up to and including the prediction of the
    // edited token is unchanged.
    let same = (t_edit + 1 - k) * cfg.vocab_size;
    assert_close(&other[..same], &base[..same], 1e-5, 1e-5);
    assert!(max_diff(&other[same..], &base[same..]) > 1e-6, "later predictions must change");
}

#[test]
fn ncm_v2_conditioning_excludes_the_concept_being_written() {
    let device = device();
    device.seed(2);
    let cfg = ncm_v2();
    let model = NcmLm::new(&cfg, &device);
    let toks = tokens(&device, cfg.vocab_size, 40, 0);
    let mut edited: Vec<i32> = toks.clone().into_data().try_to_vec().expect("ints");
    for t in &mut edited[20..24] {
        *t = (*t + 7) % cfg.vocab_size as i32;
    }
    let edited = Tensor::<2, Int>::from_data(TensorData::new(edited, [1, 40]), &device);
    let mut c1 = model.make_cache(40, 1, &device);
    let cond_a = model.prefill(toks.narrow(1, 0, 20), &mut c1, 8);
    let mut c2 = model.make_cache(40, 1, &device);
    let cond_b = model.prefill(edited.narrow(1, 0, 20), &mut c2, 8);
    assert_close(&values(cond_a), &values(cond_b), 1e-6, 1e-6);
}

// ----------------------------------------------------------------------------- sizes

#[test]
fn v2_full_config_param_counts() {
    let device = device();
    // Parameters are initialised lazily, so counting does not allocate 250M floats.
    let tok = TokenLm::new(&tok_250m_v2(), &device).parts();
    let ncm = NcmLm::new(&ncm_250m_v2(), &device).parts();
    for (name, total) in [("tok", tok.total()), ("ncm", ncm.total())] {
        println!("{name} v2: {total} parameters");
        assert!((230_000_000..=260_000_000).contains(&total), "{name}: {total}");
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
        [16.4, 10.5, 200.1, 15.7, 3.7, 2.6]
    );
    assert_eq!([m(tok.embeddings), m(tok.transformer)], [32.8, 216.9]);
}

#[test]
fn v2_cache_bytes_at_one_million_tokens() {
    let n = 1_048_576;
    let ncm = cache_bytes(&Config::Ncm(ncm_250m_v2()), n, 2, 1) as f64;
    let tok = cache_bytes(&Config::Tok(tok_250m_v2()), n, 2, 1) as f64;
    // architecture.md: about 2.5 GiB for the NCM and 12.5 GiB for the token baseline.
    let (ncm_doc, tok_doc) = (2.5 * GIB, 12.5 * GIB);
    assert!((ncm - ncm_doc).abs() / ncm_doc < 0.05, "ncm {}", ncm / GIB);
    assert!((tok - tok_doc).abs() / tok_doc < 0.05, "tok {}", tok / GIB);
    assert!(tok / ncm > 4.5);
}

#[test]
fn v2_cache_bytes_match_allocation() {
    let device = device();
    for n in [20, 64] {
        let (ncm_cfg, tok_cfg) = (ncm_v2(), tok_v2());
        let ncm = NcmLm::new(&ncm_cfg, &device).make_cache(n, 1, &device);
        // The reference stores indexer keys at the run dtype (f32 here).
        assert_eq!(ncm.nbytes(), cache_bytes_with(&Config::Ncm(ncm_cfg), n, 4, 4, 1));
        let tok = TokenLm::new(&tok_cfg, &device).make_cache(n, 1, &device);
        let got: usize = tok.iter().map(LayerCache::nbytes).sum();
        assert_eq!(got, cache_bytes_with(&Config::Tok(tok_cfg), n, 4, 4, 1));
    }
}

#[test]
fn shipped_v2_configs_parse() {
    for text in [TOK_250M_V2_TOML, NCM_250M_V2_TOML] {
        let cfg = Config::parse(text).expect("v2 config");
        assert_eq!(cfg.design(), Design::V2);
        assert_eq!(Config::parse(&cfg.to_toml()).expect("round trip"), cfg);
    }
    assert_eq!(Config::parse(ncm_model::config::NCM_250M_TOML).unwrap().design(), Design::Fallback);
    assert!(Config::parse("model = \"tok\"\nattention = \"v3\"\n").is_err());
    assert!(Config::parse("model = \"tok\"\n[v2]\nbogus = 1\n").is_err());
}
