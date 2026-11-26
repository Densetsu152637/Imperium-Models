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
