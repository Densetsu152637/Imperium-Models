//! Weight-only quantisation on the CPU reference path: the quantised layer against a host
//! decode, tiny models of both kinds and designs against their dense weights, conversion
//! errors and quantised weight-file round trips.

mod quant_common;

use burn::tensor::{Device, FloatDType, Tensor, TensorData};
use ncm_model::quant::format::{Format, quantize};
use ncm_model::quant::{self, QLinear, QuantSpec};
use ncm_model::{Config, NcmLm, TokenLm, tiny_tok, weights};
use quant_common::*;

fn device() -> Device {
    Device::flex()
}

fn spec(format: Format, head: bool) -> QuantSpec {
    QuantSpec { format, head }
}

fn random(n: usize, seed: u64, amplitude: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * amplitude
        })
        .collect()
}

#[test]
fn the_quantised_layer_multiplies_by_the_decoded_weight() {
    let device = device();
    // Ragged widths (not multiples of a group or a word) and every row count of the GEMV path.
    for (d_in, d_out) in [(70, 9), (64, 33), (1700 / 10, 16)] {
        let w = random(d_out * d_in, 1, 0.05);
        for format in [Format::q8(), Format::q4(64), Format::q4(128)] {
            let packed = quantize(&w, d_out, d_in, format);
            let decoded = packed.dequantize(); // [d_out, d_in]
            let layer = QLinear::from_packed(&packed, &device);
            for rows in [1, 3, 8, 13] {
                let x = random(rows * d_in, rows as u64 + 7, 1.0);
                let got = values(layer.forward(Tensor::<2>::from_data(
                    TensorData::new(x.clone(), [rows, d_in]),
                    &device,
                )));
                for r in 0..rows {
                    for n in 0..d_out {
                        let want: f32 =
                            (0..d_in).map(|k| x[r * d_in + k] * decoded[n * d_in + k]).sum();
                        let g = got[r * d_out + n];
                        assert!((g - want).abs() <= 1e-4, "{format:?} [{r}, {n}]: {g} vs {want}");
                    }
                }
            }
        }
    }
}

#[test]
fn a_dense_build_keeps_the_dense_weight_format() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dense.safetensors");
    let device = device();
    let cfg = tiny_tok(8);
    let t = tokens(&device, 256, 12, 1);
    let built = quant::build(None, || TokenLm::new(&cfg, &device));
    let want = values(built.forward(t.clone(), None));
    weights::save_weights(&built, &Config::Tok(cfg.clone()), &path).unwrap();
    let plain = weights::load_weights(TokenLm::new(&cfg, &device), &path, FloatDType::F32).unwrap();
    assert_eq!(want, values(plain.forward(t, None)));
}

/// Tiny models quantised from fixed dense weights stay close to them. Asserted for q8 (within 2%
/// of the largest logit, the same argmax at 90% of positions or more); q4 only has to stay
/// recognisably the same model (within 30%): untrained tiny models have flat logits, so their q4
/// numbers say little about trained weights. Printed for the record (`--nocapture`).
#[test]
fn quantised_tiny_models_track_their_dense_weights() {
    let device = device();
    for (name, config) in tiny_configs() {
        let dense = fixed_dense(&config, &device);
        let want = dense.logits(&device);
        for spec in [
            spec(Format::q8(), false),
            spec(Format::q8(), true),
            spec(Format::q4(64), false),
            spec(Format::q4(64), true),
            spec(Format::q4(128), true),
        ] {
            let got = dense.quantized(&config, spec, &device).logits(&device);
            let a = agreement(&want, &got, config.vocab_size());
            eprintln!(
                "{name:>12} {:<12} max {:.4} (rel {:.4})  mean {:.5}  argmax {:.3}",
                spec.name(),
                a.max_abs,
                a.max_rel,
                a.mean_abs,
                a.argmax
            );
            let ok = if spec.format.bits == 8 {
                a.max_rel <= 0.02 && a.argmax >= 0.9
            } else {
                a.max_rel <= 0.3
            };
            assert!(ok, "{name} {}: {a:?}", spec.name());
        }
    }
}

#[test]
fn quantised_random_models_run_both_designs() {
    let device = device();
    for (name, config) in tiny_configs() {
        let model = build(&config, Some(spec(Format::q4(64), true)), &device);
        let logits = model.logits(&device);
        assert!(logits.iter().all(|v| v.is_finite()), "{name}");
        assert!(logits.iter().any(|v| *v != 0.0), "{name}: random weights give nonzero logits");
    }
}

#[test]
fn quantised_weight_files_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let device = device();
    for (i, (name, config)) in tiny_configs().into_iter().enumerate() {
        for spec in [spec(Format::q8(), false), spec(Format::q4(128), true)] {
            let path = dir.path().join(format!("{i}-{}.safetensors", spec.name()));
            device.seed(5);
            let model = build(&config, Some(spec), &device);
            let before = model.logits(&device);
            match &model {
                Tiny::Tok(m) => quant::save_quantized(m, &config, spec, &path),
                Tiny::Ncm(m) => quant::save_quantized(m, &config, spec, &path),
            }
            .unwrap();
            assert_eq!(quant::read_spec(&path).unwrap(), Some(spec));
            assert_eq!(weights::read_config(&path).unwrap(), Some(config.clone()));
            let loaded = match &config {
                Config::Tok(c) => {
                    let (m, s) =
                        quant::load_quantized(&path, FloatDType::F32, || TokenLm::new(c, &device))
                            .unwrap();
                    assert_eq!(s, spec);
                    Tiny::Tok(m)
                }
                Config::Ncm(c) => {
                    let (m, s) =
                        quant::load_quantized(&path, FloatDType::F32, || NcmLm::new(c, &device))
                            .unwrap();
                    assert_eq!(s, spec);
                    Tiny::Ncm(m)
                }
            };
            assert_eq!(before, loaded.logits(&device), "{name} {}", spec.name());
        }
    }
}

#[test]
fn quantised_and_dense_files_do_not_mix() {
    let dir = tempfile::tempdir().unwrap();
    let device = device();
    let cfg = tiny_tok(8);
    let dense_path = dir.path().join("dense.safetensors");
    let dense = TokenLm::new(&cfg, &device);
    weights::save_weights(&dense, &Config::Tok(cfg.clone()), &dense_path).unwrap();
    assert_eq!(quant::read_spec(&dense_path).unwrap(), None);
    let err = quant::load_quantized(&dense_path, FloatDType::F32, || TokenLm::new(&cfg, &device));
    assert!(matches!(err, Err(quant::QuantError::NotQuantized { .. })));

    // A quantised file does not load into a dense model.
    let q_path = dir.path().join("q8.safetensors");
    let s = spec(Format::q8(), false);
    let q = quant::build(Some(s), || TokenLm::new(&cfg, &device));
    quant::save_quantized(&q, &Config::Tok(cfg.clone()), s, &q_path).unwrap();
    let err = weights::load_weights(TokenLm::new(&cfg, &device), &q_path, FloatDType::F32);
    assert!(
        matches!(
            err,
            Err(weights::WeightsError::Mismatch { .. } | weights::WeightsError::Store { .. })
        ),
        "{err:?}"
    );
}

#[test]
fn conversion_refuses_another_architecture() {
    let device = device();
    let small = TokenLm::new(&tiny_tok(8), &device);
    let other = ncm_model::TokConfig { ff_dim: 96, ..tiny_tok(8) };
    let err = quant::quantize_model(&small, spec(Format::q8(), false), &device, || {
        TokenLm::new(&other, &device)
    });
    assert!(matches!(err, Err(quant::QuantError::Mismatch(_))));
}

#[test]
fn spec_names_round_trip() {
    for spec in [
        spec(Format::q8(), false),
        spec(Format::q8(), true),
        spec(Format::q4(64), false),
        spec(Format::q4(128), true),
    ] {
        assert_eq!(QuantSpec::from_name(&spec.name()), Some(spec));
    }
    assert_eq!(QuantSpec::parse("none", 64, true).unwrap(), None);
    assert_eq!(QuantSpec::parse("q4", 128, false).unwrap(), Some(spec(Format::q4(128), false)));
    assert!(QuantSpec::parse("q4", 48, false).is_err());
    assert!(QuantSpec::parse("q3", 64, false).is_err());
    assert_eq!(QuantSpec::from_name("q4-g48"), None);
}
