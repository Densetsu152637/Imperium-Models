//! Decode attention without a mask in one kernel: the few query rows of a decode step against
//! a short key/value cache (the NCM decoder's cross-attention ring and its per-concept
//! self-attention), in place of Burn's matmul, softmax reductions and second matmul.

use burn_cubecl::cubecl;
use cubecl::prelude::*;

/// Units per cube of [`attend_decode_kernel`] (whole planes).
pub(crate) const CUBE: usize = 256;
/// Most keys per cube of [`attend_decode_kernel`] (their scores stay in shared memory); more
/// keys run as several splits merged by [`attend_combine_kernel`].
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

/// One cube per (batch row, key/value head, query row) and key split: `q` `[B, KVH, R, d]`
/// (the query heads of a group folded into the rows, `R = G T`), `k`, `v` `[B, KVH, S, d]`,
/// and with `masked` the `[T, S]` mask (nonzero hides a key; query row `r` takes mask row
/// `r % T`), all read through their strides. Split `y` (`CUBE_POS_Y`) takes keys
/// `y * split_len ..` (`split_len <= MAX_KEYS`). Without `split` (one split) it writes
/// `softmax(q k^T * scale) v` into `out` `[B, KVH, R, d]` (contiguous); with `split`, each
/// split's unnormalised weighted sum, score maximum and weight total into `part`
/// `[splits, B KVH R, d + 2]` for [`attend_combine_kernel`] (split-KV with an online-softmax
/// merge, so any number of keys runs). Scores: one unit per key; weighted sum: each unit owns a
/// column and every `CUBE / d`-th key, the groups added in shared memory.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn attend_decode_kernel<F: Float>(
    q: &Tensor<F>,
    k: &Tensor<F>,
    v: &Tensor<F>,
    mask: &Tensor<u8>,
    out: &mut Tensor<F>,
    part: &mut Tensor<f32>,
    kv_heads: usize,
    rows: usize,
    keys: usize,
    t_len: usize,
    split_len: usize,
    scale: f32,
    #[comptime] d: usize,
    #[comptime] masked: bool,
    #[comptime] split: bool,
    #[define(F)] _dtype: ElemType,
) {
    let cube = CUBE_POS_X as usize;
    let r = cube % rows;
    let bk = cube / rows;
    let b = bk / kv_heads;
    let kh = bk % kv_heads;
    let tid = UNIT_POS as usize;
    let s0 = CUBE_POS_Y as usize * split_len;
    let s1 = select(s0 + split_len < keys, s0 + split_len, keys);
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
    let mut s = s0 + tid;
    while s < s1 {
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
        sc[s - s0] = acc;
        top = select(acc > top, acc, top);
        s += CUBE;
    }
    let top = cube_max(&mut red, top);
    let mut total = 0.0f32;
    let mut s = tid;
    while s < s1 - s0 {
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
        let mut s = s0 + group;
        while s < s1 {
            acc += sc[s - s0] * f32::cast_from(v[vb + s * v2 + col * v3]);
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
        if comptime!(split) {
            let base = (CUBE_POS_Y as usize * CUBE_COUNT_X as usize + cube) * (d + 2);
            part[base + tid] = sum;
            if tid == 0 {
                part[base + d] = top;
                part[base + d + 1] = total;
            }
        } else {
            out[cube * d + tid] = F::cast_from(sum / total);
        }
    }
}

/// Merges the key splits of [`attend_decode_kernel`]: unit `i` of `count = B KVH R d` outputs
/// rescales each split's sum and total by `exp(max_s - max)` and writes their ratio into
/// `out` `[B, KVH, R, d]`.
#[cube(launch_unchecked)]
pub(crate) fn attend_combine_kernel<F: Float>(
    part: &Tensor<f32>,
    out: &mut Tensor<F>,
    count: usize,
    splits: usize,
    #[comptime] d: usize,
    #[define(F)] _dtype: ElemType,
) {
    let i = ABSOLUTE_POS;
    if i >= count {
        terminate!();
    }
    let (cube, col) = (i / d, i % d);
    let cubes = count / d;
    let mut top = f32::new(-3.0e38f32);
    for s in 0..splits {
        let m = part[(s * cubes + cube) * (d + 2) + d];
        top = select(m > top, m, top);
    }
    let mut sum = 0.0f32;
    let mut total = 0.0f32;
    for s in 0..splits {
        let base = (s * cubes + cube) * (d + 2);
        let m = part[base + d];
        let f = select(m > -1.0e38f32, (m - top).exp(), 0.0f32);
        sum += part[base + col] * f;
        total += part[base + d + 1] * f;
    }
    out[i] = F::cast_from(sum / total);
}
