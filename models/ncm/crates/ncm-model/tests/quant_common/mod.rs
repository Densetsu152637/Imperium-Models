//! Shared by the quantisation tests: tiny models in both designs, dense and quantised, and
//! logit agreement statistics.

#![allow(dead_code)]

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::{Device, Int, Tensor, TensorData};
use ncm_model::quant::{self, QuantSpec};
use ncm_model::{Config, NcmLm, TokenLm, tiny_ncm, tiny_ncm_v2, tiny_tok, tiny_tok_v2, tiny_v2};

pub fn tokens(device: &Device, vocab: usize, n: usize, seed: u64) -> Tensor<2, Int> {
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

/// Overwrites every matrix parameter with fixed pseudo-random values (uniform, standard
/// deviation 0.02, the init's), so a model's weights do not depend on the device's random state
/// that concurrently running tests share.
struct Fixed {
    next: u64,
    device: Device,
}

impl ModuleMapper for Fixed {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        if D != 2 {
            return param;
        }
        let shape = param.shape();
        let mut state = self.next.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        self.next += 1;
        let amplitude = 0.02 * 3f32.sqrt();
        let data: Vec<f32> = (0..shape.num_elements())
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * amplitude
            })
            .collect();
        let device = self.device.clone();
        param.map(|_| Tensor::from_data(TensorData::new(data, shape), &device))
    }
}

/// `model` with fixed matrix weights (see [`Fixed`]).
pub fn fixed<M: Module>(model: M, device: &Device) -> M {
    model.map(&mut Fixed { next: 1, device: device.clone() })
}

pub fn values<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().convert::<f32>().try_into_vec::<f32>().expect("f32 tensor")
}

/// A tiny model of either kind and design, dense or quantised.
#[allow(clippy::large_enum_variant)]
pub enum Tiny {
    Tok(TokenLm),
    Ncm(NcmLm),
}

/// The four tiny configs, by name.
pub fn tiny_configs() -> Vec<(&'static str, Config)> {
    let v2 = tiny_v2(2, 2, 4);
    vec![
        ("tok fallback", Config::Tok(tiny_tok(8))),
        ("tok v2", Config::Tok(tiny_tok_v2(v2.clone()))),
        ("ncm fallback", Config::Ncm(tiny_ncm(8, 4))),
        ("ncm v2", Config::Ncm(tiny_ncm_v2(8, v2))),
    ]
}

/// A dense model of `config` with fixed weights (see [`fixed`]).
pub fn fixed_dense(config: &Config, device: &Device) -> Tiny {
    match config {
        Config::Tok(c) => Tiny::Tok(fixed(TokenLm::new(c, device), device)),
        Config::Ncm(c) => Tiny::Ncm(fixed(NcmLm::new(c, device), device)),
    }
}

/// A model of `config`: dense random weights, or quantised random weights inside `spec`.
pub fn build(config: &Config, spec: Option<QuantSpec>, device: &Device) -> Tiny {
    quant::build(spec, || match config {
        Config::Tok(c) => Tiny::Tok(TokenLm::new(c, device)),
        Config::Ncm(c) => Tiny::Ncm(NcmLm::new(c, device)),
    })
}

impl Tiny {
    /// The same architecture quantised per `spec` from this model's weights.
    pub fn quantized(&self, config: &Config, spec: QuantSpec, device: &Device) -> Tiny {
        match (self, config) {
            (Tiny::Tok(m), Config::Tok(c)) => Tiny::Tok(
                quant::quantize_model(m, spec, device, || TokenLm::new(c, device)).unwrap(),
            ),
            (Tiny::Ncm(m), Config::Ncm(c)) => {
                Tiny::Ncm(quant::quantize_model(m, spec, device, || NcmLm::new(c, device)).unwrap())
            }
            _ => panic!("config of another model kind"),
        }
    }

    /// Token-baseline logits of the same 16 tokens fed one at a time through the caches (every
    /// linear layer sees one row: the decode path); None for the NCM.
    pub fn step_logits(&self, device: &Device) -> Option<Vec<f32>> {
        let Tiny::Tok(m) = self else { return None };
        let t = tokens(device, 256, 16, 3);
        let mut caches = m.make_cache(16, 1, device);
        let mut out = Vec::new();
        for i in 0..16 {
            out.extend(values(m.step(t.clone().narrow(1, i, 1), &mut caches)));
        }
        Some(out)
    }

    /// Logits of a 16-token sequence, flattened.
    pub fn logits(&self, device: &Device) -> Vec<f32> {
        match self {
            Tiny::Tok(m) => values(m.forward(tokens(device, 256, 16, 3), None)),
            Tiny::Ncm(m) => values(m.forward(tokens(device, 256, 16, 3))),
        }
    }
}

/// Agreement of quantised logits with the dense ones.
#[derive(Debug, Clone, Copy)]
pub struct Agreement {
    /// Largest absolute error over the largest dense logit magnitude.
    pub max_rel: f32,
    pub max_abs: f32,
    pub mean_abs: f32,
    /// Share of positions whose argmax token agrees.
    pub argmax: f32,
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    best
}

pub fn agreement(dense: &[f32], quant: &[f32], vocab: usize) -> Agreement {
    assert_eq!(dense.len(), quant.len());
    let scale = dense.iter().fold(0f32, |m, v| m.max(v.abs()));
    let errors: Vec<f32> = dense.iter().zip(quant).map(|(a, b)| (a - b).abs()).collect();
    let max_abs = errors.iter().copied().fold(0f32, f32::max);
    let rows = dense.len() / vocab;
    let same = (0..rows)
        .filter(|r| {
            let span = r * vocab..(r + 1) * vocab;
            argmax(&dense[span.clone()]) == argmax(&quant[span])
        })
        .count();
    Agreement {
        max_rel: max_abs / scale,
        max_abs,
        mean_abs: errors.iter().sum::<f32>() / errors.len() as f32,
        argmax: same as f32 / rows as f32,
    }
}
