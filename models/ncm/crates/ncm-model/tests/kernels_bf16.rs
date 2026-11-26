//! Fused decode kernels against the reference in bf16 on CUDA (the benchmark's dtype). Needs
//! `--features cuda` and a GPU; CI builds neither, so this file compiles to nothing there.
#![cfg(feature = "cuda")]

mod kernel_parity;

use burn::tensor::FloatDType;
use kernel_parity::*;

const DTYPE: FloatDType = FloatDType::BF16;

// Largest error allowed, over the largest magnitude of the f32 CPU reference run on the same
// (bf16-exact) weights and inputs. The CUDA bf16 reference rounds every intermediate tensor and
// is off by up to about 2.5e-2; the kernels keep f32 inside and store bf16 only in the caches.
const TOL: f32 = 4e-2;

// Conditions of the append graph against the eager append: the graph attends over a growing
// cache's whole buffer with the tail masked where the eager path takes the filled part, which
// may reorder a sum.
const APPEND_TOL: f32 = 1e-5;

#[test]
fn delta_chunked_matches_recurrent_bf16() {
    let (device, _lock) = setup(DTYPE);
    delta_chunked_matches(&device, TOL);
}

#[test]
fn delta_prefill_matches_reference_bf16() {
    let (device, _lock) = setup(DTYPE);
    delta_prefill_matches(&device, TOL);
}

#[test]
fn delta_step_matches_reference_bf16() {
    let (device, _lock) = setup(DTYPE);
    delta_matches(&device, TOL);
}

#[test]
fn retrieval_decode_matches_reference_bf16() {
    let (device, _lock) = setup(DTYPE);
    for case in retrieval_cases().into_iter().filter(|c| c.bf16_stable) {
        retrieval_matches(&device, &case, TOL);
    }
}

#[test]
fn retrieval_prefill_matches_reference_bf16() {
    let (device, _lock) = setup(DTYPE);
    for case in retrieval_cases().into_iter().filter(|c| !c.spec.dense && c.bf16_stable) {
        retrieval_prefill_matches(&device, &case, TOL);
    }
}

#[test]
fn argmax_matches_burn_bf16() {
    let (device, _lock) = setup(DTYPE);
    argmax_matches(&device);
}

#[test]
fn graphed_decode_matches_eager_bf16() {
    let (device, _lock) = setup(DTYPE);
    graphed_decode_matches(&device, None);
}

#[test]
fn gemv_matches_cpu_bf16() {
    let (device, _lock) = setup(DTYPE);
    gemv_matches(&device, TOL);
}

#[test]
fn graphed_speculation_matches_eager_bf16() {
    let (device, _lock) = setup(DTYPE);
    graphed_speculation_matches(&device, false, None);
}

#[test]
fn graphed_quantised_decode_matches_eager_bf16() {
    let (device, _lock) = setup(DTYPE);
    for spec in quant_specs() {
        graphed_decode_matches(&device, Some(spec));
    }
}

#[test]
fn graphed_quantised_speculation_matches_eager_bf16() {
    let (device, _lock) = setup(DTYPE);
    for spec in quant_specs() {
        graphed_speculation_matches(&device, false, Some(spec));
    }
}

#[test]
fn graphed_step_matches_eager_bf16() {
    let (device, _lock) = setup(DTYPE);
    graphed_step_matches(&device, None);
}

#[test]
fn graphed_quantised_step_matches_eager_bf16() {
    let (device, _lock) = setup(DTYPE);
    for spec in quant_specs() {
        graphed_step_matches(&device, Some(spec));
    }
}

#[test]
fn graphed_append_matches_eager_bf16() {
    let (device, _lock) = setup(DTYPE);
    graphed_append_matches(&device, APPEND_TOL);
}

#[test]
fn graphed_tok_step_matches_eager_bf16() {
    let (device, _lock) = setup(DTYPE);
    graphed_tok_matches(&device, None);
    for spec in quant_specs() {
        graphed_tok_matches(&device, Some(spec));
    }
}

#[test]
fn norm_matches_reference_bf16() {
    let (device, _lock) = setup(DTYPE);
    norm_matches(&device, TOL);
}

#[test]
fn graphed_step_survives_reset_bf16() {
    let (device, _lock) = setup(DTYPE);
    graphed_step_survives_reset(&device);
}

#[test]
fn graphed_sampled_step_matches_eager_bf16() {
    let (device, _lock) = setup(DTYPE);
    graphed_sampled_step_matches(&device);
}

#[test]
fn sampler_matches_distribution_bf16() {
    let (device, _lock) = setup(DTYPE);
    sampler_matches_distribution(&device);
}

#[test]
fn attend_decode_matches_cpu_bf16() {
    let (device, _lock) = setup(DTYPE);
    attend_decode_matches(&device, TOL);
}

#[test]
fn graphed_decoder_waits_for_a_full_ring_bf16() {
    let (device, _lock) = setup(DTYPE);
    graphed_decoder_waits_for_a_full_ring(&device);
}

#[test]
fn fused_mlp_matches_unfused_bf16() {
    let (device, _lock) = setup(DTYPE);
    fused_mlp_matches(&device, TOL);
}

#[test]
fn fused_projections_match_unfused_bf16() {
    let (device, _lock) = setup(DTYPE);
    fused_projections_match(&device, TOL);
}

#[test]
fn fused_rope_matches_reference_bf16() {
    let (device, _lock) = setup(DTYPE);
    fused_rope_matches(&device, TOL);
}

#[test]
fn sampler_rows_are_independent_bf16() {
    let (device, _lock) = setup(DTYPE);
    sampler_rows_are_independent(&device);
}

#[test]
fn pooled_slot_steps_match_separate_bf16() {
    let (device, _lock) = setup(DTYPE);
    pooled_slot_steps_match(&device, TOL);
}
