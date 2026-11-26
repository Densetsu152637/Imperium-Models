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
