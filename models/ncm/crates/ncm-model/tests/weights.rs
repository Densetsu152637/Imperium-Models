//! Weight files: save and load round trips, header metadata, and refusing a wrong architecture.
//! CPU backend, tiny configs.

use burn::tensor::{Device, FloatDType, Int, Tensor, TensorData};
use ncm_model::weights::{self, CONFIG_KEY, FORMAT, FORMAT_KEY};
use ncm_model::{Config, NcmLm, TokenLm, tiny_ncm, tiny_tok};

fn device() -> Device {
    Device::flex()
}

fn tokens(device: &Device, n: usize) -> Tensor<2, Int> {
    let data: Vec<i32> = (0..n as i32).map(|i| (i * 37 + 5) % 256).collect();
    Tensor::from_data(TensorData::new(data, [1, n]), device)
}

fn values<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().convert::<f32>().try_into_vec::<f32>().expect("f32 tensor")
}

#[test]
fn the_token_baseline_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tok.safetensors");
    let device = device();
    let cfg = tiny_tok(8);
    device.seed(1);
    let model = TokenLm::new(&cfg, &device);
    let before = values(model.forward(tokens(&device, 12), None));
    weights::save_weights(&model, &Config::Tok(cfg.clone()), &path).unwrap();

    device.seed(2);
    let fresh = TokenLm::new(&cfg, &device);
    assert_ne!(before, values(fresh.forward(tokens(&device, 12), None)), "different init");
    let loaded = weights::load_weights(fresh, &path, FloatDType::F32).unwrap();
    assert_eq!(before, values(loaded.forward(tokens(&device, 12), None)));
}

#[test]
fn the_ncm_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ncm.safetensors");
    let device = device();
    let cfg = tiny_ncm(8, 4);
    device.seed(1);
    let model = NcmLm::new(&cfg, &device);
    let before = values(model.forward(tokens(&device, 16)));
    weights::save_weights(&model, &Config::Ncm(cfg.clone()), &path).unwrap();

    device.seed(2);
    let loaded = weights::load_weights(NcmLm::new(&cfg, &device), &path, FloatDType::F32).unwrap();
    assert_eq!(before, values(loaded.forward(tokens(&device, 16))));
}

#[test]
fn v2_models_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let device = device();
    let v2 = ncm_model::tiny_v2(2, 2, 4);
    let tok_cfg = ncm_model::tiny_tok_v2(v2.clone());
    let path = dir.path().join("tok-v2.safetensors");
    device.seed(1);
    let model = TokenLm::new(&tok_cfg, &device);
    let before = values(model.forward(tokens(&device, 12), None));
    weights::save_weights(&model, &Config::Tok(tok_cfg.clone()), &path).unwrap();
    assert_eq!(weights::read_config(&path).unwrap(), Some(Config::Tok(tok_cfg.clone())));
    device.seed(2);
    let loaded =
        weights::load_weights(TokenLm::new(&tok_cfg, &device), &path, FloatDType::F32).unwrap();
    assert_eq!(before, values(loaded.forward(tokens(&device, 12), None)));

    let ncm_cfg = ncm_model::tiny_ncm_v2(8, v2);
    let path = dir.path().join("ncm-v2.safetensors");
    device.seed(1);
    let model = NcmLm::new(&ncm_cfg, &device);
    let before = values(model.forward(tokens(&device, 16)));
    weights::save_weights(&model, &Config::Ncm(ncm_cfg.clone()), &path).unwrap();
    device.seed(2);
    let loaded =
        weights::load_weights(NcmLm::new(&ncm_cfg, &device), &path, FloatDType::F32).unwrap();
    assert_eq!(before, values(loaded.forward(tokens(&device, 16))));
    // A fallback file does not load into a v2 model.
    let fallback = dir.path().join("tok.safetensors");
    weights::save_weights(
        &TokenLm::new(&tiny_tok(8), &device),
        &Config::Tok(tiny_tok(8)),
        &fallback,
    )
    .unwrap();
    let v2_model = TokenLm::new(&tok_cfg, &device);
    assert!(weights::load_weights(v2_model, &fallback, FloatDType::F32).is_err());
}

#[test]
fn the_header_carries_the_format_and_the_config() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tok.safetensors");
    let cfg = Config::Tok(tiny_tok(8));
    weights::save_weights(&TokenLm::new(&tiny_tok(8), &device()), &cfg, &path).unwrap();
    let metadata = weights::read_metadata(&path).unwrap();
    assert_eq!(metadata.get(FORMAT_KEY).map(String::as_str), Some(FORMAT));
    assert!(metadata[CONFIG_KEY].starts_with("model = \"tok\""));
    assert_eq!(weights::read_config(&path).unwrap(), Some(cfg));
}

#[test]
fn a_config_survives_a_toml_round_trip() {
    for cfg in [Config::Tok(tiny_tok(8)), Config::Ncm(tiny_ncm(8, 4))] {
        assert_eq!(Config::parse(&cfg.to_toml()).unwrap(), cfg);
    }
}

#[test]
fn another_architecture_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tok.safetensors");
    let device = device();
    weights::save_weights(&TokenLm::new(&tiny_tok(8), &device), &Config::Tok(tiny_tok(8)), &path)
        .unwrap();
    let wide = ncm_model::TokConfig { n_layers: 6, ..tiny_tok(8) };
    let wide = TokenLm::new(&wide, &device);
    assert!(weights::load_weights(wide, &path, FloatDType::F32).is_err(), "fewer layers");
    let ncm = NcmLm::new(&tiny_ncm(8, 4), &device);
    assert!(weights::load_weights(ncm, &path, FloatDType::F32).is_err(), "not an NCM");
}

#[test]
fn bf16_weights_load_as_f32() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tok.safetensors");
    let device = device();
    let cfg = tiny_tok(8);
    let model = TokenLm::new(&cfg, &device);
    // Save the way training does: bf16 tensors.
    let bf16 = burn::module::Module::map(model, &mut Bf16);
    weights::save_weights(&bf16, &Config::Tok(cfg.clone()), &path).unwrap();
    let loaded =
        weights::load_weights(TokenLm::new(&cfg, &device), &path, FloatDType::F32).unwrap();
    let logits = loaded.forward(tokens(&device, 8), None);
    assert_eq!(logits.dtype(), burn::tensor::DType::F32);
    assert!(values(logits).iter().all(|v| v.is_finite()));
}

struct Bf16;

impl burn::module::ModuleMapper for Bf16 {
    fn map_float<const D: usize>(
        &mut self,
        param: burn::module::Param<Tensor<D>>,
    ) -> burn::module::Param<Tensor<D>> {
        param.map(|t| t.cast(FloatDType::BF16))
    }
}
