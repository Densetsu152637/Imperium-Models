//! Greedy token pick: argmax over the last axis, one cube per row. Burn's generic reduction
//! scans a 32,000-wide row with too little parallelism at batch 1 (about 0.4 ms per call on an
//! RTX 5080); a cube-wide reduction takes a few microseconds.

use burn_cubecl::cubecl;
use cubecl::prelude::*;

/// Units per cube (a power of two).
pub(crate) const CUBE: usize = 256;

/// `out[r]` is the index of the largest value of row `r` of `x` `[rows, n]` (the first one on
/// ties, as Burn's `argmax`).
#[cube(launch_unchecked)]
pub(crate) fn argmax_kernel<F: Float>(
    x: &Tensor<F>,
    out: &mut Tensor<i32>,
    n: usize,
    #[define(F)] _dtype: ElemType,
) {
    let row = CUBE_POS;
    let tid = UNIT_POS as usize;
    let mut vals = Shared::<[f32]>::new_slice(CUBE);
    let mut idxs = Shared::<[u32]>::new_slice(CUBE);
    let base = row * x.stride(0);
    let xs = x.stride(1);
    let mut best = f32::new(-3.0e38f32);
    let mut at = u32::new(u32::MAX as i64);
    let mut i = tid;
    while i < n {
        let v = f32::cast_from(x[base + i * xs]);
        // Units visit increasing indices, so a strictly larger value keeps the first maximum.
        if v > best {
            best = v;
            at = u32::cast_from(i);
        }
        i += CUBE;
    }
    vals[tid] = best;
    idxs[tid] = at;
    sync_cube();
    let mut half = CUBE_DIM / 2;
    while half > 0 {
        if UNIT_POS < half {
            let o = tid + half as usize;
            let (v, w) = (vals[o], vals[tid]);
            if v > w || (v == w && idxs[o] < idxs[tid]) {
                vals[tid] = v;
                idxs[tid] = idxs[o];
            }
        }
        sync_cube();
        half /= 2;
    }
    if tid == 0 {
        out[row] = i32::cast_from(idxs[0]);
    }
}
