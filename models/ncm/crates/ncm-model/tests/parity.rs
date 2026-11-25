//! The Rust models reproduce the PyTorch reference: tiny-config weights and logits exported from
//! the since-deleted reference at commit 2fa4688. The fixture is frozen (see `fixtures/README.md`).

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::{Device, Int, Tensor, TensorData};
use ncm_model::{NcmConfig, NcmLm, TokConfig, TokenLm};
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/parity.json");

#[derive(Deserialize)]
struct Fixture<C> {
    config: C,
    tokens: Vec<i32>,
    logits: Vec<f32>,
    params: Vec<FixtureTensor>,
}

#[derive(Deserialize, Clone)]
struct FixtureTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Deserialize)]
struct Parity {
    tok: Fixture<TokConfig>,
    ncm: Fixture<NcmConfig>,
}

/// Overwrites every float parameter, in module field order, with the next fixture tensor.
struct Loader {
    params: std::vec::IntoIter<FixtureTensor>,
    device: Device,
}

impl ModuleMapper for Loader {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let fixture = self.params.next().expect("fixture has fewer tensors than the model");
        assert_eq!(param.shape().dims::<D>().to_vec(), fixture.shape, "parameter shape");
        let device = self.device.clone();
        param.map(|_| Tensor::from_data(TensorData::new(fixture.data, fixture.shape), &device))
    }
}

fn load<M: Module>(model: M, params: Vec<FixtureTensor>, device: &Device) -> M {
    let mut loader = Loader { params: params.into_iter(), device: device.clone() };
    let model = model.map(&mut loader);
    assert!(loader.params.next().is_none(), "model has fewer tensors than the fixture");
    model
}

fn assert_close(got: Vec<f32>, want: &[f32]) {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(scale > 0.5, "fixture logits are degenerate");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 1e-4 + 1e-3 * w.abs(), "logit {i}: got {g}, PyTorch {w}");
    }
}

fn int_tokens(tokens: &[i32], device: &Device) -> Tensor<2, Int> {
    Tensor::from_data(TensorData::new(tokens.to_vec(), [1, tokens.len()]), device)
}

fn fixture() -> Parity {
    serde_json::from_str(FIXTURE).expect("parity fixture")
}

#[test]
fn token_baseline_matches_pytorch() {
    let device = Device::flex();
    let Fixture { config, tokens, logits, params } = fixture().tok;
    let model = load(TokenLm::new(&config, &device), params, &device);
    let got = model.forward(int_tokens(&tokens, &device), None);
    assert_close(got.into_data().try_to_vec::<f32>().expect("f32"), &logits);
}

#[test]
fn ncm_matches_pytorch() {
    let device = Device::flex();
    let Fixture { config, tokens, logits, params } = fixture().ncm;
    let model = load(NcmLm::new(&config, &device), params, &device);
    let got = model.forward(int_tokens(&tokens, &device));
    assert_close(got.into_data().try_to_vec::<f32>().expect("f32"), &logits);
}
