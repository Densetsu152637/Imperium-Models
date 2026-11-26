//! Decode attention without a mask in one kernel: the few query rows of a decode step against
//! a short key/value cache (the NCM decoder's cross-attention ring and its per-concept
//! self-attention), in place of Burn's matmul, softmax reductions and second matmul.

use burn_cubecl::cubecl;
use cubecl::prelude::*;

/// Units per cube of [`attend_decode_kernel`] (whole planes).
pub(crate) const CUBE: usize = 256;
/// Most keys [`attend_decode_kernel`] takes (their scores stay in shared memory).
pub(crate) const MAX_KEYS: usize = 2048;

/// Maximum of `v` over the cube through `red` (one slot per plane); every unit gets it.
#[cube]
fn cube_max(red: &mut Shared<[f32]>, v: f32) -> f32 {
    let top = plane_max(v);
    if UNIT_POS_PLANE == 0 {
        red[(UNIT_POS / PLANE_DIM) as usize] = top;
    }
    sync_cube();
    let mut all = f32::new(-3.0e38f32);
    for p in 0..CUBE_DIM / PLANE_DIM {
        let x = red[p as usize];
        all = select(x > all, x, all);
    }
    sync_cube();
    all
}

/// Sum of `v` over the cube through `red`, as [`cube_max`].
#[cube]
fn cube_sum(red: &mut Shared<[f32]>, v: f32) -> f32 {
    let total = plane_sum(v);
    if UNIT_POS_PLANE == 0 {
        red[(UNIT_POS / PLANE_DIM) as usize] = total;
    }
    sync_cube();
    let mut all = 0.0f32;
    for p in 0..CUBE_DIM / PLANE_DIM {
        all += red[p as usize];
    }
    sync_cube();
    all
}

/// One cube per (batch row, key/value head, query row): `q` `[B, KVH, R, d]` (the query heads
/// of a group folded into the rows, `R = G T`), `k`, `v` `[B, KVH, S, d]` with
/// `S <= MAX_KEYS`, and with `masked` the `[T, S]` mask (nonzero hides a key; query row `r`
/// takes mask row `r % T`), all read through their strides. Writes `softmax(q k^T * scale) v` into `out` `[B, KVH, R, d]`
/// (contiguous). Scores: one unit per key; weighted sum: each unit owns a column and every
/// `CUBE / d`-th key, the groups added in shared memory.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn attend_decode_kernel<F: Float>(
    q: &Tensor<F>,
    k: &Tensor<F>,
    v: &Tensor<F>,
    mask: &Tensor<u8>,
    out: &mut Tensor<F>,
    kv_heads: usize,
    rows: usize,
    keys: usize,
    t_len: usize,
    scale: f32,
    #[comptime] d: usize,
    #[comptime] masked: bool,
    #[define(F)] _dtype: ElemType,
) {
    let r = CUBE_POS % rows;
    let bk = CUBE_POS / rows;
    let b = bk / kv_heads;
    let kh = bk % kv_heads;
    let tid = UNIT_POS as usize;
    let mut qs = Shared::<[f32]>::new_slice(d);
    let mut sc = Shared::<[f32]>::new_slice(MAX_KEYS);
    let mut red = Shared::<[f32]>::new_slice(CUBE);
    if tid < d {
        let at = b * q.stride(0) + kh * q.stride(1) + r * q.stride(2) + tid * q.stride(3);
        qs[tid] = f32::cast_from(q[at]) * scale;
    }
    sync_cube();
    let kb = b * k.stride(0) + kh * k.stride(1);
    let (k2, k3) = (k.stride(2), k.stride(3));
    let mut top = f32::new(-3.0e38f32);
    let mut s = tid;
    while s < keys {
        let row = kb + s * k2;
        let mut acc = 0.0f32;
        #[unroll]
        for i in 0..d {
            acc += qs[i] * f32::cast_from(k[row + i * k3]);
        }
        if masked {
            let hidden = mask[(r % t_len) * mask.stride(0) + s * mask.stride(1)];
            acc = select(hidden != 0u8, f32::new(-3.0e38f32), acc);
        }
        sc[s] = acc;
        top = select(acc > top, acc, top);
        s += CUBE;
    }
    let top = cube_max(&mut red, top);
    let mut total = 0.0f32;
    let mut s = tid;
    while s < keys {
        let e = select(sc[s] > -1.0e38f32, (sc[s] - top).exp(), 0.0f32);
        sc[s] = e;
        total += e;
        s += CUBE;
    }
    let total = cube_sum(&mut red, total);
    // Weighted sum of the values (cube_sum ended with a barrier, so every weight is in place).
    let groups = comptime!(CUBE / d);
    let col = tid % d;
    let group = tid / d;
    let vb = b * v.stride(0) + kh * v.stride(1);
    let (v2, v3) = (v.stride(2), v.stride(3));
    let mut acc = 0.0f32;
    if group < groups {
        let mut s = group;
        while s < keys {
            acc += sc[s] * f32::cast_from(v[vb + s * v2 + col * v3]);
            s += groups;
        }
    }
    red[tid] = acc;
    sync_cube();
    if tid < d {
        let mut sum = 0.0f32;
        for g in 0..groups {
            sum += red[g * d + tid];
        }
        out[((b * kv_heads + kh) * rows + r) * d + tid] = F::cast_from(sum / total);
    }
}
