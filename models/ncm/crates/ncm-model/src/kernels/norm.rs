//! RMSNorm in one kernel, optionally after the residual add before it: Burn's `RmsNorm` is a
//! square, a mean reduction, a square root and three elementwise ops, about six launches per
//! norm, and decode runs a few hundred norms per token.
//!
//! The arithmetic mirrors the reference (`burn::nn::RmsNorm`) at the run dtype `F`: the sum
//! `s = F(x + delta)`, the mean square in f32, `rms = F(sqrt(mean + eps))`, and
//! `y = F(F(s / rms) * gamma)`. Only the order of the f32 sum differs.

use burn_cubecl::cubecl;
use cubecl::prelude::*;

/// Most units per cube (one cube per row).
pub(crate) const MAX_UNITS: usize = 256;

/// Row `CUBE_POS` of `x` `[rows, d]` (any strides) into the contiguous `out`: the normed row at
/// `y_at + row * d`, and with `add` the sum `s = x + delta` at `row * d` first.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn rms_norm_kernel<F: Float>(
    x: &Tensor<F>,
    delta: &Tensor<F>,
    gamma: &Tensor<F>,
    out: &mut Tensor<F>,
    d: usize,
    y_at: usize,
    eps: f32,
    #[comptime] add: bool,
    #[define(F)] _dtype: ElemType,
) {
    let row = CUBE_POS;
    let tid = UNIT_POS as usize;
    let units = CUBE_DIM as usize;
    let (x0, x1) = (x.stride(0), x.stride(1));
    let mut sq = 0.0f32;
    let mut i = tid;
    while i < d {
        let mut v = x[row * x0 + i * x1];
        if comptime!(add) {
            let dv = delta[row * delta.stride(0) + i * delta.stride(1)];
            v = F::cast_from(f32::cast_from(v) + f32::cast_from(dv));
            out[row * d + i] = v;
        }
        let f = f32::cast_from(v);
        sq += f * f;
        i += units;
    }
    let mut part = Shared::<[f32]>::new_slice(MAX_UNITS / 32);
    let total = plane_sum(sq);
    let planes = units / PLANE_DIM as usize;
    let plane = tid / PLANE_DIM as usize;
    if UNIT_POS_PLANE == 0 {
        part[plane] = total;
    }
    sync_cube();
    let mut all = 0.0f32;
    for p in 0..planes {
        all += part[p];
    }
    let rms = f32::cast_from(F::cast_from(f32::sqrt(all / f32::cast_from(d) + eps)));
    let g0 = gamma.stride(0);
    let mut i = tid;
    while i < d {
        let v = if comptime!(add) { out[row * d + i] } else { x[row * x0 + i * x1] };
        let normed = f32::cast_from(F::cast_from(f32::cast_from(v) / rms));
        out[y_at + row * d + i] = F::cast_from(normed * f32::cast_from(gamma[i * g0]));
        i += units;
    }
}
