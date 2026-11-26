//! Fused decode kernels against the reference paths, shared by `kernels_f32.rs` and
//! `kernels_bf16.rs` (a device's float dtype is fixed per process, so each dtype gets its own
//! test binary).
//!
//! Every check runs one layer three ways from equal caches: the fused kernels on CUDA, the
//! reference tensor code on CUDA (same dtype), and the reference on the CPU in f32 with the same
//! weights and inputs (exact conversions from the CUDA dtype). The fused path must be within the
//! tolerance of the f32 CPU result; the CUDA reference's own error is logged next to it. The
//! process-wide switch picks the CUDA path, so the tests of one binary hold a lock while they run.

// Each test binary uses its own part of this module.
#![allow(dead_code)]

use std::sync::{Mutex, MutexGuard};

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::{Bool, Device, Distribution, FloatDType, Int, Tensor, TensorData};
use ncm_model::delta::{
    DeltaCache, DeltaSpec, GatedDeltaNet, delta_rule_chunked, delta_rule_recurrent,
};
use ncm_model::fused::{active, argmax_last, set_cuda_graphs, set_fused_kernels};
use ncm_model::layers::Rope;
use ncm_model::ncm::{GraphedAppend, GraphedDecoder, GraphedStep};
use ncm_model::quant::{self, QuantSpec, format::Format};
use ncm_model::retrieval::{LatentCache, RetrievalSpec, SparseRetrieval};
use ncm_model::sample::SampleSettings;
use ncm_model::tok::GraphedTok;
use ncm_model::{
    Acceptance, NcmLm, SpecStats, Speculator, TokenLm, tiny_ncm, tiny_ncm_v2, tiny_tok,
    tiny_tok_v2, tiny_v2,
};

fn ints<const D: usize>(t: Tensor<D, Int>) -> Vec<i64> {
    t.into_data().convert::<i64>().try_to_vec::<i64>().expect("int tensor")
}

/// The argmax kernel picks the same index as Burn's `argmax` (the first maximum), including on
/// rows with repeated maxima.
pub fn argmax_matches(device: &Device) {
    device.seed(3);
    let x = Tensor::<3>::random([2, 3, 32000], Distribution::Normal(0.0, 1.0), device);
    let got = ints(argmax_last(x.clone()));
    let want = ints(reference(|| x.argmax(2)));
    assert_eq!(got, want);
    // Rows of equal values: every unit holds a maximum, the first index must win.
    let flat =
        Tensor::<3>::from_data(TensorData::new(vec![1.5f32; 2 * 1000], [2, 1, 1000]), device);
    assert_eq!(ints(argmax_last(flat)), vec![0, 0]);
}

/// The matrix-vector kernels against an f32 CPU matmul: the models' decode shapes (one row and a
/// concept's four), split and unsplit `K`, a row-major and a transposed (tied-embedding) weight.
pub fn gemv_matches(device: &Device, tol: f32) {
    device.seed(4);
    let normal = Distribution::Normal(0.0, 1.0);
    for (rows, k, n) in [
        (1, 1024, 4112),
        (1, 2240, 1024),
        (4, 512, 1536),
        (1, 1024, 3780),
        (3, 37, 70),
        (2, 5000, 64),
        (1, 2048, 77),
        (8, 512, 1536),
        (6, 1024, 1000),
    ] {
        let x = Tensor::<3>::random([1, rows, k], normal, device);
        let w = Tensor::<2>::random([k, n], normal, device);
        let got = values(ncm_model::kernels::gemv(x.clone(), w.clone()));
        // Split-K combines in the last cube to arrive, which resets its counter: a second call
        // of the same shape gives the same result.
        let again = values(ncm_model::kernels::gemv(x.clone(), w.clone()));
        assert_eq!(got, again, "gemv {rows}x{k}x{n}: a repeated call differs");
        let want = values(to_cpu(&x).matmul(to_cpu(&w).unsqueeze::<3>()));
        let err = rel_error(&got, &want);
        eprintln!("gemv {rows}x{k}x{n}: error over scale {err:.2e}");
        assert!(err <= tol, "gemv {rows}x{k}x{n}: error {err:.2e}");
    }
    for (rows, k, n) in [(1, 512, 32000), (4, 64, 300)] {
        let x = Tensor::<3>::random([1, rows, k], normal, device);
        let emb = Tensor::<2>::random([n, k], normal, device);
        let got = values(ncm_model::kernels::gemv(x.clone(), emb.clone().transpose()));
        let want = values(to_cpu(&x).matmul(to_cpu(&emb).transpose().unsqueeze::<3>()));
        let err = rel_error(&got, &want);
        eprintln!("gemv transposed {rows}x{k}x{n}: error over scale {err:.2e}");
        assert!(err <= tol, "gemv transposed {rows}x{k}x{n}: error {err:.2e}");
    }
}

/// The fused MLP half ([`ncm_model::layers::Mlp::residual_step`]: add, norm, gate/up GEMV with
/// SwiGLU, down GEMV with the residual add) against the same block run unfused on the CUDA
/// path (`fused::set_fused_mlp(false)`) and on the CPU in f32: the models' widths, one and
/// four rows, dense and quantised weights, with and without the residual delta.
pub fn fused_mlp_matches(device: &Device, tol: f32) {
    use ncm_model::fused::set_fused_mlp;
    use ncm_model::layers::{Mlp, rms_norm};
    let normal = Distribution::Normal(0.0, 1.0);
    for (rows, d, ff) in [(1, 1024, 2688), (4, 512, 1536), (1, 64, 128), (2, 1000, 600)] {
        for spec in [None, Some(quant_specs()[0]), Some(quant_specs()[1])] {
            device.seed(6);
            let mlp = quant::build(spec, || Mlp::new(d, ff, device));
            let norm = rms_norm(d, 1e-6, device);
            let x = Tensor::<3>::random([1, rows, d], normal, device);
            let delta = Tensor::<3>::random([1, rows, d], normal, device);
            for with_delta in [true, false] {
                let delta = with_delta.then(|| delta.clone());
                let got = mlp
                    .residual_step(&norm, &x, delta.as_ref())
                    .expect("the fused MLP takes the models' shapes");
                set_fused_mlp(false);
                assert!(mlp.residual_step(&norm, &x, delta.as_ref()).is_none());
                let s = match &delta {
                    Some(delta) => x.clone() + delta.clone(),
                    None => x.clone(),
                };
                let unfused = s.clone() + mlp.forward(norm.forward(s.clone()));
                set_fused_mlp(true);
                let cpu_mlp = cpu_copy(&mlp);
                let cpu_norm = cpu_copy(&norm);
                let s_cpu = to_cpu(&s);
                let want = values(s_cpu.clone() + cpu_mlp.forward(cpu_norm.forward(s_cpu)));
                let what = format!("fused mlp {rows}x{d}x{ff} {spec:?} delta {with_delta}");
                assert_close(&what, &values(got), &values(unfused), &want, tol);
            }
        }
    }
}

/// The fused projections of the attention half against the same ops unfused, on CUDA and on
/// the CPU in f32: [`ncm_model::mlp_fused::norm_linear`] (the pre-norm in the GEMV's prologue,
/// with and without the gate's sigmoid in its epilogue) and
/// [`ncm_model::mlp_fused::gated_linear`] (the output gate's product in the prologue), for one,
/// three and eight rows (a block decoder's levels), dense and quantised weights.
pub fn fused_projections_match(device: &Device, tol: f32) {
    use burn::tensor::activation::sigmoid;
    use ncm_model::layers::rms_norm;
    use ncm_model::mlp_fused::{gated_linear, norm_linear};
    use ncm_model::quant::QLinear;
    let normal = Distribution::Normal(0.0, 1.0);
    // Eight rows of 960 still fit the quantised kernel's staged row (8 * (960 + 31) slots).
    for (rows, d, n) in [(1, 1024, 4112), (3, 512, 1536), (8, 960, 1024), (8, 128, 72)] {
        for spec in [None, Some(quant_specs()[0]), Some(quant_specs()[1])] {
            device.seed(7);
            let layer = quant::build(spec, || QLinear::new(d, n, device));
            let norm = rms_norm(d, 1e-6, device);
            let x = Tensor::<2>::random([rows, d], normal, device);
            let cpu_layer = cpu_copy(&layer);
            let cpu_norm = cpu_copy(&norm);
            for gate in [false, true] {
                let got = norm_linear(x.clone(), norm.gamma.val(), 1e-6, &layer, gate)
                    .expect("the fused projection takes these shapes");
                let mut unfused = layer.forward(norm.forward(x.clone()));
                let mut want = cpu_layer.forward(cpu_norm.forward(to_cpu(&x)));
                if gate {
                    unfused = sigmoid(unfused);
                    want = sigmoid(want);
                }
                let what = format!("norm_linear {rows}x{d}x{n} {spec:?} sigmoid {gate}");
                assert_close(&what, &values(got), &values(unfused), &values(want), tol);
            }
            let g = sigmoid(Tensor::<2>::random([rows, d], normal, device));
            let got = gated_linear(x.clone(), g.clone(), &layer).expect("these shapes");
            let unfused = layer.forward(x.clone() * g.clone());
            let want = cpu_layer.forward(to_cpu(&x) * to_cpu(&g));
            let what = format!("gated_linear {rows}x{d}x{n} {spec:?}");
            assert_close(&what, &values(got), &values(unfused), &values(want), tol);
        }
    }
}

/// The fused RoPE kernel ([`ncm_model::rope_fused::rope`], plain and after QK-norm) against
/// `layers::apply_rope` of the (normed) heads on the CUDA reference path and on the CPU in f32:
/// strided (swapped) inputs, head widths 64, 128 and 16, a few positions.
pub fn fused_rope_matches(device: &Device, tol: f32) {
    use ncm_model::layers::{apply_rope, rms_norm};
    let normal = Distribution::Normal(0.0, 1.0);
    for (b, h, t, d) in [(1, 8, 1, 64), (2, 4, 3, 128), (1, 2, 5, 16)] {
        device.seed(8);
        let x = Tensor::<4>::random([b, t, h, d], normal, device);
        let swapped = x.clone().swap_dims(1, 2);
        let rope = Rope::new(37, t, d, 10_000.0, 1.0, device);
        let cpu_rope = Rope::new(37, t, d, 10_000.0, 1.0, &cpu());
        let norm = rms_norm(d, 1e-6, device);
        let cpu_norm = cpu_copy(&norm);
        let got = ncm_model::rope_fused::rope(swapped.clone(), &rope, None).expect("these widths");
        let cuda_ref = reference(|| apply_rope(swapped.clone(), &rope));
        let want = apply_rope(to_cpu(&swapped), &cpu_rope);
        let what = format!("rope {b}x{h}x{t}x{d}");
        assert_close(&what, &values(got), &values(cuda_ref), &values(want), tol);
        let gamma = Some((norm.gamma.val(), 1e-6));
        let got = ncm_model::rope_fused::rope(swapped.clone(), &rope, gamma).expect("these widths");
        let cuda_ref = reference(|| apply_rope(norm.forward(x.clone()).swap_dims(1, 2), &rope));
        let want = apply_rope(cpu_norm.forward(to_cpu(&x)).swap_dims(1, 2), &cpu_rope);
        let what = format!("qk-norm rope {b}x{h}x{t}x{d}");
        assert_close(&what, &values(got), &values(cuda_ref), &values(want), tol);
    }
}

/// The fused RMSNorm kernels (plain, and after a residual add) against Burn's `RmsNorm` on the
/// CPU in f32 with the same weights and inputs: the models' widths at one and four rows,
/// QK-norm's per-head rows, and a strided (narrowed) input. Also against the CUDA reference
/// path, whose rounding the kernel mirrors.
pub fn norm_matches(device: &Device, tol: f32) {
    use burn::nn::RmsNormConfig;
    device.seed(5);
    let normal = Distribution::Normal(0.0, 1.0);
    for (rows, d) in [(1, 1024), (4, 512), (32, 64), (3, 1000)] {
        let norm = RmsNormConfig::new(d).with_epsilon(1e-6).init(device);
        let gamma = Tensor::<1>::random([d], normal, device);
        let norm = burn::nn::RmsNorm { gamma: Param::from_tensor(gamma), ..norm };
        let wide = Tensor::<3>::random([1, rows, d + 8], normal, device);
        let x = wide.narrow(2, 4, d);
        let delta = Tensor::<3>::random([1, rows, d], normal, device);
        let cpu_norm = cpu_copy(&norm);
        let want = values(cpu_norm.forward(to_cpu(&x)));
        let cuda_ref = values(reference(|| norm.forward(x.clone())));
        let got = values(ncm_model::fused::rms_norm(&norm, x.clone()));
        assert_close(&format!("rms_norm {rows}x{d}"), &got, &cuda_ref, &want, tol);
        let sum_want = to_cpu(&x) + to_cpu(&delta);
        let want = values(cpu_norm.forward(sum_want.clone()));
        let (sum, normed) = ncm_model::fused::add_rms_norm(&norm, x.clone(), delta.clone());
        let cuda_ref = values(reference(|| norm.forward(x.clone() + delta.clone())));
        assert_close(
            &format!("add_rms_norm sum {rows}x{d}"),
            &values(sum),
            &values(sum_want.clone()),
            &values(sum_want),
            tol,
        );
        assert_close(&format!("add_rms_norm {rows}x{d}"), &values(normed), &cuda_ref, &want, tol);
    }
}

/// Greedy NCM decode of `concepts` concepts after a 64-token prompt, through the decoder and
/// backbone graphs when `graphs` is on: the tokens and conditions of every concept.
fn ncm_decode(
    model: &NcmLm,
    graphs: bool,
    concepts: usize,
    device: &Device,
) -> (Vec<Vec<i64>>, Vec<Vec<f32>>) {
    set_cuda_graphs(graphs);
    let prompt = Tensor::<2, Int>::from_data(
        TensorData::new((0..64).map(|i: i32| i * 37 % 256).collect::<Vec<_>>(), [1, 64]),
        device,
    );
    let mut cache = model.make_cache(64 + 4 * concepts + 4, 1, device);
    let mut cond = model.prefill(prompt, &mut cache, 16);
    let mut decoder = GraphedDecoder::default();
    let (mut tokens, mut conds) = (Vec::new(), Vec::new());
    for _ in 0..concepts {
        let next = decoder.decode(model, cond, &mut cache);
        tokens.push(ints(next.clone()));
        cond = model.append_tokens(next, &mut cache);
        conds.push(values(cond.clone()));
    }
    set_cuda_graphs(true);
    if graphs {
        assert!(decoder.is_captured(), "decoder not captured: {:?}", decoder.failure());
        let backbone = &cache.backbone_graph;
        assert!(backbone.is_captured(), "backbone graph: {:?}", backbone.failure());
    }
    (tokens, conds)
}

/// The 64-token prompt of the NCM checks.
fn ncm_prompt(device: &Device) -> Tensor<2, Int> {
    Tensor::<2, Int>::from_data(
        TensorData::new((0..64).map(|i: i32| i * 37 % 256).collect::<Vec<_>>(), [1, 64]),
        device,
    )
}

/// Greedy NCM decode of `concepts` concepts after the prompt of [`ncm_decode`]: the first
/// concept eagerly, every later one as one whole concept step ([`GraphedStep`]: append, then
/// decode), replayed from a graph when `graphs` is on. The tokens of every concept.
fn ncm_step_decode(model: &NcmLm, graphs: bool, concepts: usize, device: &Device) -> Vec<Vec<i64>> {
    set_cuda_graphs(graphs);
    let mut cache = model.make_cache(64 + 4 * concepts + 4, 1, device);
    let cond = model.prefill(ncm_prompt(device), &mut cache, 16);
    let (mut next, _) = model.decode_concept(cond, &mut cache, None, false);
    let mut graphed = GraphedStep::default();
    let mut tokens = vec![ints(next.clone())];
    for _ in 1..concepts {
        next = graphed.step(model, next, &mut cache);
        tokens.push(ints(next.clone()));
    }
    set_cuda_graphs(true);
    assert_eq!(graphed.is_captured(), graphs, "concept step graph: {:?}", graphed.failure());
    tokens
}

/// Sampled NCM decode through [`GraphedStep::sampled`] (graphs on or off): the tokens of every
/// concept.
fn ncm_sampled_decode(model: &NcmLm, graphs: bool, device: &Device) -> Vec<Vec<i64>> {
    set_cuda_graphs(graphs);
    let mut cache = model.make_cache(64 + 4 * 12 + 4, 1, device);
    let cond = model.prefill(ncm_prompt(device), &mut cache, 16);
    let settings =
        SampleSettings { temperature: 0.9, top_k: 40, top_p: 0.95, seed: 1234, counter: 0 };
    let mut graphed = GraphedStep::sampled(device, settings);
    let mut next = graphed.first(model, cond, &mut cache, &[]);
    let mut tokens = vec![ints(next.clone())];
    for _ in 1..12 {
        next = graphed.step(model, next, &mut cache);
        tokens.push(ints(next.clone()));
    }
    set_cuda_graphs(true);
    assert_eq!(graphed.is_captured(), graphs, "sampled step graph: {:?}", graphed.failure());
    tokens
}

/// The sampled concept step replayed from a graph draws the same tokens as the same device
/// sampler run eagerly (same seed, same draw counters), both designs; the draws vary.
pub fn graphed_sampled_step_matches(device: &Device) {
    for cfg in [tiny_ncm(8, 4), tiny_ncm_v2(8, tiny_v2(2, 2, 4))] {
        let model = build_ncm(&cfg, None, device);
        let eager = ncm_sampled_decode(&model, false, device);
        let graphed = ncm_sampled_decode(&model, true, device);
        assert_eq!(graphed, eager, "{}: sampled tokens differ", cfg.attention.name());
        let distinct: std::collections::BTreeSet<_> = eager.iter().flatten().collect();
        assert!(distinct.len() > 8, "sampling should vary the tokens: {eager:?}");
    }
}

/// The device sampler's draws follow the host sampler's distribution: the `top_k` largest
/// logits, then the smallest prefix reaching `top_p` at the temperature. 20,000 draws of one
/// 12-token row (one row per draw, each its own counter) against the exact probabilities; a
/// near-zero temperature takes the argmax.
pub fn sampler_matches_distribution(device: &Device) {
    use ncm_model::sample::{SampleParams, sample};
    let logits: Vec<f32> = vec![1.2, -0.4, 2.0, 0.3, 1.9, -2.0, 0.0, 1.1, 0.8, -1.0, 0.5, 1.95];
    let v = logits.len();
    let rows = 20_000;
    let data: Vec<f32> = (0..rows).flat_map(|_| logits.clone()).collect();
    let x = Tensor::<2>::from_data(TensorData::new(data, [rows, v]), device);
    let params = SampleParams::new(device);
    for (temperature, top_k, top_p) in
        [(1.0, 0, 1.0), (0.7, 3, 1.0), (1.3, 0, 0.6), (0.5, 5, 0.8), (2.0, 7, 0.9)]
    {
        params.set(SampleSettings { temperature, top_k, top_p, seed: 77, counter: 5 });
        let drawn = ints(sample(x.clone(), &params, 0));
        // The host sampler's kept set and probabilities.
        let mut order: Vec<usize> = (0..v).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let k = if top_k == 0 { v } else { top_k };
        let kept = &order[..k];
        let max = logits[kept[0]];
        let w: Vec<f32> = kept.iter().map(|&i| ((logits[i] - max) / temperature).exp()).collect();
        let total: f32 = w.iter().sum();
        let mut keep = k;
        if top_p < 1.0 {
            let mut mass = 0.0;
            for (i, wi) in w.iter().enumerate() {
                mass += wi / total;
                if mass >= top_p {
                    keep = i + 1;
                    break;
                }
            }
        }
        let z: f32 = w[..keep].iter().sum();
        let mut want = vec![0f32; v];
        for (i, &id) in kept[..keep].iter().enumerate() {
            want[id] = w[i] / z;
        }
        let mut got = vec![0f32; v];
        for &t in &drawn {
            got[t as usize] += 1.0 / rows as f32;
        }
        for id in 0..v {
            let sd = (want[id] * (1.0 - want[id]) / rows as f32).sqrt();
            assert!(
                (got[id] - want[id]).abs() <= 5.0 * sd + 1e-4,
                "T {temperature} k {top_k} p {top_p}: token {id} drawn {:.4}, want {:.4}",
                got[id],
                want[id]
            );
        }
    }
    params.set(SampleSettings { temperature: 1e-4, top_k: 0, top_p: 1.0, seed: 1, counter: 0 });
    let drawn = ints(sample(x.narrow(0, 0, 100), &params, 0));
    assert!(drawn.iter().all(|&t| t == 2), "near-zero temperature takes the argmax");
}

/// The whole-concept-step graph (encoder, cross-attention keys, pooling, backbone step,
/// decoder, argmax in one replay) gives the same tokens as the same kernels run eagerly,
/// concept after concept, in both designs, with the linear layers dense or quantised per
/// `spec`. The tiny configs fill the 8-token encoder ring during the prompt, so the step is
/// captured on the second concept; the fallback backbone's sliding rings (4 concepts) wrap and
/// its full layers grow inside the graph, and the v2 retrieval turns sparse within the run.
pub fn graphed_step_matches(device: &Device, spec: Option<QuantSpec>) {
    for cfg in [tiny_ncm(8, 4), tiny_ncm_v2(8, tiny_v2(2, 2, 4)), tiny_ncm_v2(8, yarn_v2())] {
        let model = build_ncm(&cfg, spec, device);
        let eager = ncm_step_decode(&model, false, 12, device);
        let graphed = ncm_step_decode(&model, true, 12, device);
        for (concept, (te, tg)) in eager.iter().zip(&graphed).enumerate() {
            assert_eq!(tg, te, "{} concept {concept}: tokens differ", cfg.attention.name());
        }
    }
}

/// One [`GraphedStep`] kept across [`ncm_model::NcmCache::reset`]: the reset writes into the
/// buffers the graph was captured against, so the graph keeps serving and decodes the same
/// prompt to the same tokens as before (both designs).
pub fn graphed_step_survives_reset(device: &Device) {
    for cfg in [tiny_ncm(8, 4), tiny_ncm_v2(8, tiny_v2(2, 2, 4))] {
        let model = build_ncm(&cfg, None, device);
        let mut cache = model.make_cache(64 + 4 * 8 + 4, 1, device);
        let mut graphed = GraphedStep::default();
        let mut run = |cache: &mut ncm_model::NcmCache| {
            let cond = model.prefill(ncm_prompt(device), cache, 16);
            let (mut next, _) = model.decode_concept(cond, cache, None, false);
            let mut tokens = Vec::new();
            for _ in 0..8 {
                next = graphed.step(&model, next, cache);
                tokens.push(ints(next.clone()));
            }
            assert!(graphed.is_captured(), "concept step graph: {:?}", graphed.failure());
            tokens
        };
        let first = run(&mut cache);
        cache.reset();
        let second = run(&mut cache);
        assert_eq!(second, first, "{}: tokens after a reset differ", cfg.attention.name());
    }
}

/// The append-only graph ([`GraphedAppend`], what self-speculation uses) gives the same
/// conditions as `append_tokens` run eagerly, concept after concept, both designs.
pub fn graphed_append_matches(device: &Device, tol: f32) {
    for cfg in [tiny_ncm(8, 4), tiny_ncm_v2(8, tiny_v2(2, 2, 4))] {
        let model = build_ncm(&cfg, None, device);
        let run = |graphs: bool| {
            set_cuda_graphs(graphs);
            let mut cache = model.make_cache(64 + 4 * 12 + 4, 1, device);
            model.prefill(ncm_prompt(device), &mut cache, 16);
            let mut append = GraphedAppend::default();
            let mut conds = Vec::new();
            for concept in 0..12 {
                // Fixed tokens: the conditions alone are compared.
                let written: Vec<i32> = (0..4).map(|j| (concept * 11 + j * 5) % 256).collect();
                let written = Tensor::<2, Int>::from_data(TensorData::new(written, [1, 4]), device);
                let cond = append.append(&model, written, &mut cache);
                conds.push(values(cond.clone()));
            }
            set_cuda_graphs(true);
            assert_eq!(append.is_captured(), graphs, "append graph: {:?}", append.failure());
            conds
        };
        let (eager, graphed) = (run(false), run(true));
        for (concept, (e, g)) in eager.iter().zip(&graphed).enumerate() {
            let err = rel_error(g, e);
            eprintln!("{} concept {concept}: condition error {err:.2e}", cfg.attention.name());
            assert!(err <= tol, "concept {concept}: conditions differ by {err:.2e}");
        }
    }
}

/// Greedy token-baseline decode of `steps` tokens after a 16-token prompt, through
/// [`GraphedTok`] (graphs on) or the same kernels eagerly.
fn tok_decode(model: &TokenLm, graphs: bool, steps: usize, device: &Device) -> Vec<i64> {
    set_cuda_graphs(graphs);
    let prompt = Tensor::<2, Int>::from_data(
        TensorData::new((0..16).map(|i: i32| i * 37 % 256).collect::<Vec<_>>(), [1, 16]),
        device,
    );
    let mut caches = model.make_cache(16 + steps + 1, 1, device);
    model.prefill(prompt.clone().narrow(1, 0, 15), &mut caches, 8);
    let mut token = prompt.narrow(1, 15, 1);
    let mut graphed = GraphedTok::default();
    let mut tokens = Vec::new();
    for _ in 0..steps {
        token = graphed.step(model, token, &mut caches);
        tokens.extend(ints(token.clone()));
    }
    set_cuda_graphs(true);
    assert_eq!(graphed.is_captured(), graphs, "token step graph: {:?}", graphed.failure());
    tokens
}

/// A tiny v2 config with YaRN on the retrieval key (factor 4 over 16 positions, so it changes
/// the tables within the tests' runs); its indexer keys are 8-bit, as by default.
fn yarn_v2() -> ncm_model::V2Config {
    ncm_model::V2Config { ret_yarn_factor: 4.0, ret_yarn_original: 16, ..tiny_v2(2, 2, 4) }
}

/// The token baseline's graphed step (embedding, stack, logits, argmax) gives the same tokens
/// as the same kernels run eagerly, both designs: the fallback's 8-token sliding rings wrap
/// and its full layers grow inside the graph; the v2 retrieval turns sparse.
pub fn graphed_tok_matches(device: &Device, spec: Option<QuantSpec>) {
    for cfg in [tiny_tok(8), tiny_tok_v2(tiny_v2(4, 2, 8)), tiny_tok_v2(yarn_v2())] {
        device.seed(9);
        let model = quant::build(spec, || TokenLm::new(&cfg, device));
        let eager = tok_decode(&model, false, 40, device);
        let graphed = tok_decode(&model, true, 40, device);
        assert_eq!(graphed, eager, "{}: graphed tokens differ", cfg.attention.name());
    }
}

/// A captured [`GraphedDecoder`] kept across a reset to a prompt shorter than the
/// cross-attention ring must not replay (the graph attends over the whole ring, stale slots
/// included): it decodes eagerly until the ring is full again, giving the eager tokens.
pub fn graphed_decoder_waits_for_a_full_ring(device: &Device) {
    for cfg in [tiny_ncm(8, 4), tiny_ncm_v2(8, tiny_v2(2, 2, 4))] {
        let model = build_ncm(&cfg, None, device);
        let mut cache = model.make_cache(64 + 4 * 4, 1, device);
        let mut decoder = GraphedDecoder::default();
        let cond = model.prefill(ncm_prompt(device), &mut cache, 16);
        decoder.decode(&model, cond, &mut cache);
        assert!(decoder.is_captured(), "decoder graph: {:?}", decoder.failure());
        cache.reset();
        // One concept: the 8-token ring holds 4.
        let short =
            Tensor::<2, Int>::from_data(TensorData::new(vec![5i32, 9, 2, 7], [1, 4]), device);
        let cond = model.prefill(short, &mut cache, 16);
        let want = ints(model.decode_concept(cond.clone(), &mut cache, None, false).0);
        let got = ints(decoder.decode(&model, cond, &mut cache));
        assert_eq!(
            got,
            want,
            "{}: a kept decoder graph replayed over a part-filled ring",
            cfg.attention.name()
        );
    }
}

/// The quantisations the graphed-decode checks run with: q8, and q4 in groups of 64 with the
/// output head quantised too.
pub fn quant_specs() -> [QuantSpec; 2] {
    [
        QuantSpec { format: Format::q8(), head: false },
        QuantSpec { format: Format::q4(64), head: true },
    ]
}

/// A tiny NCM of `cfg` with random weights, its linear layers (draft head included) quantised
/// per `spec`.
fn build_ncm(cfg: &ncm_model::NcmConfig, spec: Option<QuantSpec>, device: &Device) -> NcmLm {
    device.seed(9);
    quant::build(spec, || NcmLm::new(cfg, device))
}

/// The decoder graph (both designs) and the backbone graph (v2) give the same tokens and
/// conditions as the same kernels run eagerly, concept after concept, with the linear layers
/// dense or quantised per `spec` (the quantised GEMV kernels then run inside the graphs). Tiny
/// configs: the 8-token encoder window fills during the prompt, so both graphs are captured on
/// the first concept, and the v2 backbone's retrieval (blocks of 2, top 2, window 4) turns
/// sparse within the run.
pub fn graphed_decode_matches(device: &Device, spec: Option<QuantSpec>) {
    for cfg in [tiny_ncm(8, 4), tiny_ncm_v2(8, tiny_v2(2, 2, 4))] {
        let model = build_ncm(&cfg, spec, device);
        let (tokens_e, conds_e) = ncm_decode(&model, false, 12, device);
        let (tokens_g, conds_g) = ncm_decode(&model, true, 12, device);
        for (concept, (te, tg)) in tokens_e.iter().zip(&tokens_g).enumerate() {
            assert_eq!(tg, te, "concept {concept}: tokens differ");
            let err = rel_error(&conds_g[concept], &conds_e[concept]);
            assert!(err <= 1e-5, "concept {concept}: conditions differ by {err:.2e}");
        }
    }
}

/// Self-speculative greedy decode of `concepts` concepts after the same prompt as
/// [`ncm_decode`], with the draft and verify passes replayed from graphs when `graphs` is on:
/// the tokens of every concept and the speculator's counts.
fn spec_decode(
    model: &NcmLm,
    graphs: bool,
    acceptance: Acceptance,
    concepts: usize,
    device: &Device,
) -> (Vec<Vec<i64>>, SpecStats) {
    set_cuda_graphs(graphs);
    let prompt = Tensor::<2, Int>::from_data(
        TensorData::new((0..64).map(|i: i32| i * 37 % 256).collect::<Vec<_>>(), [1, 64]),
        device,
    );
    let mut cache = model.make_cache(64 + 4 * concepts + 4, 1, device);
    let mut cond = model.prefill(prompt, &mut cache, 16);
    let mut speculator = Speculator::new(acceptance);
    let mut tokens = Vec::new();
    for _ in 0..concepts {
        let next = speculator.decode(model, cond, &cache);
        tokens.push(ints(next.clone()));
        cond = model.append_tokens(next, &mut cache);
    }
    set_cuda_graphs(true);
    if graphs {
        let failure = speculator.failure();
        assert!(speculator.is_captured(), "speculative graphs not captured: {failure:?}");
    }
    (tokens, speculator.stats())
}

/// The draft and verify graphs (both designs; linear layers dense or quantised per `spec`) give
/// the same tokens and counts as the same kernels run eagerly, under real verification and under an assumed acceptance of one half
/// (several verify replays per concept). With `exact`, the speculative tokens also equal plain
/// greedy decoding through the decoder graph (f32; in bf16 the parallel and step-by-step passes
/// round differently and a near tie of the two best logits may go either way, so mismatching
/// concepts are only counted).
pub fn graphed_speculation_matches(device: &Device, exact: bool, spec: Option<QuantSpec>) {
    for cfg in [tiny_ncm(8, 4), tiny_ncm_v2(8, tiny_v2(2, 2, 4))] {
        let model = build_ncm(&cfg, spec, device);
        for acceptance in [Acceptance::Verify, Acceptance::Assume(0.5)] {
            let (tokens_e, stats_e) = spec_decode(&model, false, acceptance, 12, device);
            let (tokens_g, stats_g) = spec_decode(&model, true, acceptance, 12, device);
            assert_eq!(tokens_g, tokens_e, "{acceptance:?}: graphed tokens differ");
            assert_eq!(stats_g, stats_e, "{acceptance:?}: graphed passes differ");
            eprintln!("{} {acceptance:?}: {stats_g:?}", cfg.attention.name());
        }
        let (plain, _) = ncm_decode(&model, true, 12, device);
        let (spec, _) = spec_decode(&model, true, Acceptance::Verify, 12, device);
        let differ = plain.iter().zip(&spec).filter(|(p, s)| p != s).count();
        eprintln!("{}: {differ} of 12 concepts differ from plain greedy", cfg.attention.name());
        if exact {
            assert_eq!(spec, plain, "speculative greedy differs from plain greedy");
        }
    }
}

static SWITCH: Mutex<()> = Mutex::new(());

/// The CUDA device at `dtype`, and the lock on the fused-kernel switch.
pub fn setup(dtype: FloatDType) -> (Device, MutexGuard<'static, ()>) {
    let guard = SWITCH.lock().unwrap_or_else(|e| e.into_inner());
    let mut device = Device::cuda(0);
    if device.configure(dtype).is_err() {
        assert_eq!(device.settings().float_dtype, dtype, "device dtype already fixed");
    }
    set_fused_kernels(true);
    assert!(active(&device), "fused kernels are not active on the CUDA device");
    (device, guard)
}

/// Runs `f` on the reference path (fused kernels off) and restores the switch.
pub fn reference<T>(f: impl FnOnce() -> T) -> T {
    set_fused_kernels(false);
    let out = f();
    set_fused_kernels(true);
    out
}

pub fn values<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    let data = t.try_into_data().unwrap_or_else(|e| panic!("reading a tensor failed: {e:#?}"));
    data.convert::<f32>().try_to_vec::<f32>().expect("float tensor")
}

/// The CPU device the f32 reference runs on.
pub fn cpu() -> Device {
    Device::flex()
}

/// `x` on the CPU in f32.
pub fn to_cpu<const D: usize>(x: &Tensor<D>) -> Tensor<D> {
    x.clone().to_device(&cpu()).cast(FloatDType::F32)
}

/// Moves every float parameter to the CPU in f32.
struct ToCpuF32;

impl ModuleMapper for ToCpuF32 {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        Param::from_tensor(to_cpu(&param.val()))
    }
}

/// An f32 CPU copy of `module`.
pub fn cpu_copy<M: Module>(module: &M) -> M {
    module.clone().map(&mut ToCpuF32)
}

/// Largest absolute difference over the largest magnitude of `want`.
pub fn rel_error(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0f32, |m, w| m.max(w.abs())).max(1e-6);
    got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max) / scale
}

/// Every element of `got` finite and within `tol` times the largest magnitude of `want`; logs
/// the fused error next to the CUDA reference's (`reference`).
pub fn assert_close(what: &str, got: &[f32], reference: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let scale = want.iter().fold(0f32, |m, w| m.max(w.abs())).max(1e-6);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what}: element {i} is {g}");
        assert!(
            (g - w).abs() <= tol * scale,
            "{what}: element {i}: got {g}, want {w} (tol {}, scale {scale})",
            tol * scale
        );
    }
    eprintln!(
        "{what}: error over scale {scale:.3}: fused {:.2e}, cuda reference {:.2e}",
        rel_error(got, want),
        rel_error(reference, want)
    );
}

/// Like [`assert_close`] for paths whose algorithm has its own rounding error on the device
/// (the chunked delta rule inverts a triangular matrix by repeated squaring): `got` may be off
/// the f32 CPU result by `tol` or by 1.5 times the CUDA reference's own error, whichever is
/// larger, so the fused kernel adds no error of its own.
pub fn assert_no_worse(what: &str, got: &[f32], reference: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    assert!(got.iter().all(|g| g.is_finite()), "{what}: non-finite output");
    let (err, err_reference) = (rel_error(got, want), rel_error(reference, want));
    assert!(
        err <= tol.max(1.5 * err_reference),
        "{what}: error over scale {err:.2e}, the CUDA reference's {err_reference:.2e}"
    );
    eprintln!("{what}: error over scale: fused {err:.2e}, cuda reference {err_reference:.2e}");
}

fn delta_layer(head_dim: usize, heads: usize, device: &Device) -> GatedDeltaNet {
    let spec = DeltaSpec { d: heads * head_dim, head_dim, conv: 4, chunk: 64, eps: 1e-6 };
    GatedDeltaNet::new(&spec, device)
}

/// Random inputs of the gated delta rule over `t` positions: L2-normalised queries and keys,
/// values, log decays in (-0.5, 0), write strengths in (0, 1) and a starting state.
fn delta_inputs(
    b: usize,
    h: usize,
    t: usize,
    dk: usize,
    dv: usize,
    device: &Device,
) -> [Tensor<4>; 4] {
    let normal = Distribution::Normal(0.0, 1.0);
    let unit = |x: Tensor<4>| {
        let norm = x.clone().square().sum_dim(3).sqrt();
        x / norm
    };
    let q = unit(Tensor::random([b, h, t, dk], normal, device));
    let k = unit(Tensor::random([b, h, t, dk], normal, device));
    let v = Tensor::random([b, h, t, dv], normal, device);
    let state = Tensor::random([b, h, dk, dv], normal, device).mul_scalar(0.1);
    [q, k, v, state]
}

/// The chunked delta rule with the fused chunk scan, against the same chunked form on the
/// reference path and the token-by-token recurrence on the CPU in f32: outputs and final state,
/// for lengths that pad the last chunk, fill whole chunks, and fit in one.
pub fn delta_chunked_matches(device: &Device, tol: f32) {
    for (dk, dv, chunk) in [(128, 128, 64), (16, 16, 4), (32, 32, 64)] {
        for t in [200, 128, 37] {
            device.seed(t as u64);
            let (b, h) = (2, 3);
            let [q, k, v, state] = delta_inputs(b, h, t, dk, dv, device);
            let uniform = |lo: f64, hi: f64| {
                Tensor::<3>::random([b, h, t], Distribution::Uniform(lo, hi), device)
            };
            let (g, beta) = (uniform(-0.5, 0.0), uniform(0.0, 1.0));
            let args = (q, k, v, g, beta, state);
            let run = |a: &(Tensor<4>, Tensor<4>, Tensor<4>, Tensor<3>, Tensor<3>, Tensor<4>)| {
                let (q, k, v, g, beta, s) = a.clone();
                delta_rule_chunked(q, k, v, g, beta, s, chunk)
            };
            let (got, got_state) = run(&args);
            let (base, base_state) = reference(|| run(&args));
            let (q, k, v, g, beta, s) = args;
            let (want, want_state) = delta_rule_recurrent(
                to_cpu(&q),
                to_cpu(&k),
                to_cpu(&v),
                to_cpu(&g),
                to_cpu(&beta),
                to_cpu(&s),
            );
            let what = format!("delta chunked dk {dk} chunk {chunk} t {t}");
            assert_no_worse(&what, &values(got), &values(base), &values(want), tol);
            let what = format!("{what} state");
            let (got, base, want) = (values(got_state), values(base_state), values(want_state));
            assert_no_worse(&what, &got, &base, &want, tol);
        }
    }
}

/// A gated DeltaNet layer prefilled through the fused chunk scan (more than a chunk per call,
/// so not the step kernel), then stepped: outputs against the f32 CPU layer.
pub fn delta_prefill_matches(device: &Device, tol: f32) {
    for (head_dim, heads) in [(128, 2), (16, 2)] {
        device.seed(17);
        let layer = delta_layer(head_dim, heads, device);
        let exact = cpu_copy(&layer);
        let d = heads * head_dim;
        let normal = Distribution::Normal(0.0, 1.0);
        let mut fused: DeltaCache = layer.make_cache(2, device);
        let mut cuda: DeltaCache = layer.make_cache(2, device);
        let mut want: DeltaCache = exact.make_cache(2, &cpu());
        for (step, t) in [150, 70, 1, 4, 300].into_iter().enumerate() {
            let x = Tensor::<3>::random([2, t, d], normal, device).mul_scalar(20.0);
            let got = values(layer.forward(x.clone(), Some(&mut fused)));
            let base = values(reference(|| layer.forward(x.clone(), Some(&mut cuda))));
            let exp = values(exact.forward(to_cpu(&x), Some(&mut want)));
            let what = format!("delta prefill hd {head_dim} call {step} (t {t})");
            assert_no_worse(&what, &got, &base, &exp, tol);
        }
    }
}

/// The fused DeltaNet step: one-token steps and 3- and 4-token runs (the NCM encoder's
/// per-concept call) after a chunked prefill, batch 2. Head widths 128 (the models'), 32 and 16
/// (the tiny configs); 96 channels also exercise cubecl's row pitching.
pub fn delta_matches(device: &Device, tol: f32) {
    for (head_dim, heads) in [(128, 3), (32, 1), (16, 2)] {
        device.seed(11);
        let layer = delta_layer(head_dim, heads, device);
        let exact = cpu_copy(&layer);
        let d = heads * head_dim;
        let normal = Distribution::Normal(0.0, 1.0);
        // Larger inputs than the 0.02-std weights alone give, so the gates and norms vary.
        let prompt = Tensor::<3>::random([2, 70, d], normal, device).mul_scalar(20.0);
        let mut fused: DeltaCache = layer.make_cache(2, device);
        let mut cuda: DeltaCache = layer.make_cache(2, device);
        let mut want: DeltaCache = exact.make_cache(2, &cpu());
        reference(|| {
            layer.forward(prompt.clone(), Some(&mut fused));
            layer.forward(prompt.clone(), Some(&mut cuda));
        });
        exact.forward(to_cpu(&prompt), Some(&mut want));
        for (step, t) in [1, 1, 4, 1, 3, 1].into_iter().enumerate() {
            let x = Tensor::<3>::random([2, t, d], normal, device).mul_scalar(20.0);
            let got = values(layer.forward(x.clone(), Some(&mut fused)));
            let base = values(reference(|| layer.forward(x.clone(), Some(&mut cuda))));
            let exp = values(exact.forward(to_cpu(&x), Some(&mut want)));
            let what = format!("delta hd {head_dim} step {step} (t {t})");
            assert_close(&what, &got, &base, &exp, tol);
        }
        assert_eq!(fused.len(), want.len());
    }
}

/// One retrieval check: layer shape, positions prefilled, positions decoded after them.
#[derive(Clone)]
pub struct RetrievalCase {
    pub name: &'static str,
    pub spec: RetrievalSpec,
    pub prefill: usize,
    pub steps: usize,
    /// Whether the block selection is stable in bf16. With 34 candidates for 32 picks, keys
    /// rounded to bf16 in the CUDA caches (and f32 on the CPU) put the boundary blocks within
    /// rounding of each other at most steps, so all three paths may select differently; that
    /// case runs in f32 only.
    pub bf16_stable: bool,
}

fn retrieval_spec(
    model_heads: bool,
    block: usize,
    topk: usize,
    local: usize,
    dense: bool,
) -> RetrievalSpec {
    let (heads, nope_dim, rope_dim, v_dim, latent, index_heads, index_dim) =
        if model_heads { (8, 128, 64, 128, 512, 4, 128) } else { (2, 16, 8, 16, 32, 2, 16) };
    RetrievalSpec {
        d: 256,
        heads,
        nope_dim,
        rope_dim,
        v_dim,
        latent,
        index_heads,
        index_dim,
        block,
        topk,
        local,
        dense,
        index_8bit: false,
        eps: 1e-6,
    }
}

/// `case` with its indexer keys cached as 8-bit codes. The fused kernel normalises the key in
/// f32 and the reference in bf16 tensor ops, so in bf16 a few codes differ by one between the
/// paths, which moves near ties of the block scores often enough that these cases run in f32
/// only (the kernels are the same code at both dtypes).
fn int8(case: RetrievalCase) -> RetrievalCase {
    let name = match case.name {
        "model heads, block 8 top 4 local 16" => "model heads, block 8 top 4 local 16, 8-bit keys",
        "tiny heads, block 2 top 2 local 4" => "tiny heads, block 2 top 2 local 4, 8-bit keys",
        "model heads, 250M selection" => "model heads, 250M selection, 8-bit keys",
        other => other,
    };
    let spec = RetrievalSpec { index_8bit: true, ..case.spec };
    RetrievalCase { name, spec, bf16_stable: false, ..case }
}

/// The cases: the models' head shapes with small blocks (dense, then sparse, crossing block and
/// window boundaries), the tiny configs' shapes, dense mode, and the 250M selection settings
/// (blocks of 64, top 32, window 1,024) past the point where selection starts; the sparse ones
/// again with 8-bit indexer keys.
pub fn retrieval_cases() -> Vec<RetrievalCase> {
    let cases = vec![
        RetrievalCase {
            name: "model heads, block 8 top 4 local 16",
            spec: retrieval_spec(true, 8, 4, 16, false),
            prefill: 30,
            steps: 70,
            bf16_stable: true,
        },
        RetrievalCase {
            name: "tiny heads, block 2 top 2 local 4",
            spec: retrieval_spec(false, 2, 2, 4, false),
            prefill: 3,
            steps: 40,
            bf16_stable: true,
        },
        RetrievalCase {
            name: "model heads, dense",
            spec: retrieval_spec(true, 8, 4, 16, true),
            prefill: 40,
            steps: 20,
            bf16_stable: true,
        },
        RetrievalCase {
            name: "model heads, 250M selection",
            spec: retrieval_spec(true, 64, 32, 1024, false),
            prefill: 3150,
            steps: 40,
            bf16_stable: false,
        },
    ];
    let quantised: Vec<_> = cases.iter().filter(|c| !c.spec.dense).cloned().map(int8).collect();
    cases.into_iter().chain(quantised).collect()
}

/// Decodes `case.steps` positions one at a time after a chunked reference prefill, batch 2, and
/// checks every step.
pub fn retrieval_matches(device: &Device, case: &RetrievalCase, tol: f32) {
    let spec = &case.spec;
    device.seed(5);
    let layer = SparseRetrieval::new(spec, device);
    let exact = cpu_copy(&layer);
    let capacity = case.prefill + case.steps;
    let mut fused: LatentCache = layer.make_cache(capacity, 2, device);
    let mut cuda: LatentCache = layer.make_cache(capacity, 2, device);
    let mut want: LatentCache = exact.make_cache(capacity, 2, &cpu());
    let normal = Distribution::Normal(0.0, 1.0);
    let rope = |start: usize, len: usize, device: &Device| {
        Rope::new(start, len, spec.rope_dim, 1e6, 1.0, device)
    };
    let mut start = 0;
    while start < case.prefill {
        let len = 512.min(case.prefill - start);
        let x = Tensor::<3>::random([2, len, spec.d], normal, device).mul_scalar(3.0);
        reference(|| {
            layer.forward(x.clone(), start, &rope(start, len, device), Some(&mut fused));
            layer.forward(x.clone(), start, &rope(start, len, device), Some(&mut cuda));
        });
        exact.forward(to_cpu(&x), start, &rope(start, len, &cpu()), Some(&mut want));
        start += len;
    }
    // A step passes when the fused output is within `tol` of the f32 CPU result, or of the CUDA
    // reference: block scores of the CUDA paths and the CPU differ in the last bits (their
    // projections do), so at a near tie for the last selected block both CUDA paths may pick
    // another block than the CPU. Such steps are counted.
    let (mut worst, mut worst_reference, mut flips) = (0f32, 0f32, 0);
    for pos in case.prefill..capacity {
        let x = Tensor::<3>::random([2, 1, spec.d], normal, device).mul_scalar(3.0);
        let got = values(layer.forward(x.clone(), pos, &rope(pos, 1, device), Some(&mut fused)));
        let base = values(reference(|| {
            layer.forward(x.clone(), pos, &rope(pos, 1, device), Some(&mut cuda))
        }));
        let exp = values(exact.forward(to_cpu(&x), pos, &rope(pos, 1, &cpu()), Some(&mut want)));
        assert!(got.iter().all(|g| g.is_finite()), "{} pos {pos}: non-finite output", case.name);
        let (err, err_reference) = (rel_error(&got, &exp), rel_error(&base, &exp));
        let err_vs_reference = rel_error(&got, &base);
        assert!(
            err <= tol || err_vs_reference <= tol,
            "{} pos {pos}: error over scale {err:.2e} against the f32 CPU result and \
             {err_vs_reference:.2e} against the CUDA reference (which is off by \
             {err_reference:.2e})",
            case.name
        );
        if err <= tol {
            worst = worst.max(err);
            worst_reference = worst_reference.max(err_reference);
        } else {
            flips += 1;
        }
    }
    assert_eq!(fused.len(), capacity);
    assert!(flips * 10 <= case.steps, "{}: {flips} near-tie steps of {}", case.name, case.steps);
    eprintln!(
        "{}: {} steps, worst error over scale: fused {worst:.2e}, cuda reference \
         {worst_reference:.2e}; {flips} near-tie steps matched the CUDA reference instead",
        case.name, case.steps
    );
}

/// The fused sparse prefill (indexer selection and attention on the device) against the
/// reference's masked attention over the whole context, on CUDA at the same dtype, and the f32
/// CPU reference: the case's positions plus its decode steps fed in uneven chunks (37, 300 and
/// 512 positions), batch 2. An output row passes when it is within `tol` of the CPU result or of
/// the CUDA reference; rows matching neither (a near tie of block scores picked differently) may
/// be at most 2% of all.
pub fn retrieval_prefill_matches(device: &Device, case: &RetrievalCase, tol: f32) {
    let spec = &case.spec;
    device.seed(7);
    let layer = SparseRetrieval::new(spec, device);
    let exact = cpu_copy(&layer);
    let capacity = case.prefill + case.steps;
    let mut fused: LatentCache = layer.make_cache(capacity, 2, device);
    let mut cuda: LatentCache = layer.make_cache(capacity, 2, device);
    let mut want: LatentCache = exact.make_cache(capacity, 2, &cpu());
    let normal = Distribution::Normal(0.0, 1.0);
    let rope = |start: usize, len: usize, device: &Device| {
        Rope::new(start, len, spec.rope_dim, 1e6, 1.0, device)
    };
    let (mut start, mut call) = (0, 0);
    let (mut rows, mut flips, mut worst, mut worst_reference) = (0, 0, 0f32, 0f32);
    while start < capacity {
        let len = [37, 300, 512][call % 3].min(capacity - start).max(2);
        let len = len.min(capacity - start);
        let x = Tensor::<3>::random([2, len, spec.d], normal, device).mul_scalar(3.0);
        let got =
            values(layer.forward(x.clone(), start, &rope(start, len, device), Some(&mut fused)));
        let base = values(reference(|| {
            layer.forward(x.clone(), start, &rope(start, len, device), Some(&mut cuda))
        }));
        let exp =
            values(exact.forward(to_cpu(&x), start, &rope(start, len, &cpu()), Some(&mut want)));
        assert!(got.iter().all(|g| g.is_finite()), "{} at {start}: non-finite output", case.name);
        let scale = exp.iter().fold(0f32, |m, w| m.max(w.abs())).max(1e-6);
        for ((g, b), e) in got.chunks(spec.d).zip(base.chunks(spec.d)).zip(exp.chunks(spec.d)) {
            let diff = |a: &[f32], z: &[f32]| {
                a.iter().zip(z).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) / scale
            };
            let (err, err_base, err_reference) = (diff(g, e), diff(g, b), diff(b, e));
            rows += 1;
            if err <= tol {
                worst = worst.max(err);
                worst_reference = worst_reference.max(err_reference);
            } else if err_base > tol {
                flips += 1;
            }
        }
        start += len;
        call += 1;
    }
    assert_eq!(fused.len(), capacity);
    assert!(flips * 50 <= rows, "{}: {flips} of {rows} prefill rows match neither", case.name);
    eprintln!(
        "{} prefill: {rows} rows, worst error over scale: fused {worst:.2e}, cuda reference \
         {worst_reference:.2e}; {flips} rows matched neither",
        case.name
    );
}

/// The fused decode attention against matmul-softmax-matmul on the CPU in f32: the NCM
/// decoder's shapes (cross-attention over a 512-position ring, self-attention over a few
/// positions), grouped heads folded into the rows, strided (narrowed) caches.
pub fn attend_decode_matches(device: &Device, tol: f32) {
    use burn::tensor::activation::softmax;
    let normal = Distribution::Normal(0.0, 1.0);
    for (b, kvh, rows, keys, d) in [(1, 4, 2, 512, 64), (2, 4, 8, 3, 64), (1, 1, 2, 37, 16)] {
        device.seed(keys as u64);
        let q = Tensor::<4>::random([b, kvh, rows, d], normal, device);
        // Caches with spare capacity, narrowed like a growing cache's view.
        let k = Tensor::<4>::random([b, kvh, keys + 5, d], normal, device).narrow(2, 0, keys);
        let v = Tensor::<4>::random([b, kvh, keys + 5, d], normal, device).narrow(2, 0, keys);
        assert!(ncm_model::kernels::attend_decode_fits(rows, d, keys));
        let (qc, kc, vc) = (to_cpu(&q), to_cpu(&k), to_cpu(&v));
        let scores = qc.matmul(kc.swap_dims(2, 3)) * (1.0 / (d as f64).sqrt());
        let kernel = |mask: Option<Tensor<2, Bool>>, t: usize| {
            values(ncm_model::kernels::attend_decode(q.clone(), k.clone(), v.clone(), mask, t))
        };
        let got = kernel(None, 1);
        let want = values(softmax(scores.clone(), 3).matmul(vc.clone()));
        let what = format!("attend decode rows {rows} keys {keys} d {d}");
        assert_close(&what, &got, &want, &want, tol);
        // Masked, two query positions (rows = groups x 2): each hides every third key and the
        // keys after its own position (the last one, then the last two).
        let t = 2;
        let hidden = |pos: usize, key: usize| key % 3 == 1 || key + t > keys + pos;
        let hide: Vec<bool> = (0..t * keys).map(|i| hidden(i / keys, i % keys)).collect();
        let mask = Tensor::<2, Bool>::from_data(TensorData::new(hide, [t, keys]), device);
        let got = kernel(Some(mask), t);
        let rows_mask: Vec<bool> =
            (0..rows * keys).map(|i| hidden((i / keys) % t, i % keys)).collect();
        let mask_cpu =
            Tensor::<2, Bool>::from_data(TensorData::new(rows_mask, [rows, keys]), &cpu())
                .reshape([1, 1, rows, keys])
                .expand([b, kvh, rows, keys]);
        let want = values(softmax(scores.mask_fill(mask_cpu, f32::NEG_INFINITY), 3).matmul(vc));
        let what = format!("attend decode masked rows {rows} keys {keys} d {d}");
        assert_close(&what, &got, &want, &want, tol);
    }
}

/// Per-row settings ([`ncm_model::sample::SampleParams::rows`]): each row samples with its own
/// settings and counter, a greedy row (top-k 1) takes its argmax, and a row's draws do not
/// depend on the other rows (the same row alone, with the same counter, draws the same token).
pub fn sampler_rows_are_independent(device: &Device) {
    use ncm_model::sample::{SampleParams, sample};
    let logits: Vec<f32> = vec![1.2, -0.4, 2.0, 0.3, 1.9, -2.0, 0.0, 1.1, 0.8, -1.0, 0.5, 1.95];
    let v = logits.len();
    let rows = 4;
    let data: Vec<f32> = (0..rows).flat_map(|_| logits.clone()).collect();
    let x = Tensor::<2>::from_data(TensorData::new(data, [rows, v]), device);
    let settings = [
        SampleSettings::GREEDY,
        SampleSettings { temperature: 0.7, top_k: 0, top_p: 0.95, seed: 1, counter: 3 },
        SampleSettings { temperature: 1.3, top_k: 5, top_p: 1.0, seed: 2, counter: 9 },
        SampleSettings { temperature: 0.7, top_k: 0, top_p: 0.95, seed: 1, counter: 3 },
    ];
    let params = SampleParams::rows(device, rows);
    params.set_rows(&settings);
    let drawn = ints(sample(x.clone(), &params, 0));
    assert_eq!(drawn[0], 2, "the greedy row takes the argmax");
    assert_eq!(drawn[1], drawn[3], "equal settings and counters draw alike");
    for (r, s) in settings.iter().enumerate() {
        let one = SampleParams::rows(device, 1);
        one.set_rows(&[*s]);
        let alone = ints(sample(x.clone().narrow(0, r, 1), &one, 0));
        assert_eq!(alone[0], drawn[r], "row {r} alone");
    }
}

/// Slot caches allocated together ([`ncm_model::TokenLm::make_slot_caches`], the retrieval
/// layers' rows of shared buffers, decoded in one set of launches) give the same batched steps
/// as separately allocated slot caches, for slots at different positions and past the point
/// where retrieval turns sparse (blocks of 4, top 2, window 8).
pub fn pooled_slot_steps_match(device: &Device, tol: f32) {
    let cfg = ncm_model::tiny_tok_v2(tiny_v2(4, 2, 8));
    device.seed(3);
    let model = ncm_model::TokenLm::new(&cfg, device);
    let mut pooled = model.make_slot_caches(3, 96, device);
    let mut separate: Vec<_> = (0..3).map(|_| model.make_cache(96, 1, device)).collect();
    for (i, len) in [21usize, 40, 33].into_iter().enumerate() {
        let data: Vec<i32> = (0..len).map(|j| ((j * 7 + i * 13) % cfg.vocab_size) as i32).collect();
        let prompt = Tensor::<2, Int>::from_data(TensorData::new(data, [1, len]), device);
        model.prefill(prompt.clone(), &mut pooled[i], 16);
        model.prefill(prompt, &mut separate[i], 16);
    }
    for step in 0..12 {
        let ids: Vec<i32> = (0..3).map(|i| ((step * 5 + i * 3) % cfg.vocab_size) as i32).collect();
        let token = Tensor::<2, Int>::from_data(TensorData::new(ids, [3, 1]), device);
        let mut rows: Vec<&mut [ncm_model::LayerCache]> =
            pooled.iter_mut().map(|c| c.as_mut_slice()).collect();
        let got = values(model.step_slots(token.clone(), &mut rows));
        let mut rows: Vec<&mut [ncm_model::LayerCache]> =
            separate.iter_mut().map(|c| c.as_mut_slice()).collect();
        let want = values(model.step_slots(token, &mut rows));
        assert_close(&format!("pooled slot step {step}"), &got, &want, &want, tol);
    }
}
