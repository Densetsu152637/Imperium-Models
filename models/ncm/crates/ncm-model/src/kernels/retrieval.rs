//! Fused one-token sparse retrieval decode (`SparseRetrieval::attend_selected` and the dense
//! case of `attend_latent`), in four kernels launched back to back:
//!
//! 1. [`ret_prep_kernel`]: QK-norms, partial RoPE, the absorbed query `q_nope W_uk^T` (with the
//!    softmax scale folded in), and the new position's cache entry (latent, RoPE key, indexer
//!    key) written in place.
//! 2. [`ret_index_kernel`] and [`ret_topk_kernel`]: indexer scores and block maxima of every
//!    candidate block, then the top-k blocks, on the device (both do nothing on dense steps).
//! 3. [`ret_attend_kernel`]: split attention over chunks of the visible positions (the selected
//!    blocks and the local window, or the whole context), partial softmax per chunk.
//! 4. [`ret_combine_kernel`]: merges the chunks, applies the value up-projection and the output
//!    gate.
//!
//! The position comes from a device tensor and every grid is sized for the largest step, so the
//! same launches serve every position and can be captured into a CUDA graph; which blocks are
//! candidates, whether the step is sparse and which slots are visible are decided on the device.
//! Inputs are read through their strides (cubecl pitches rows); scratch and outputs are
//! contiguous. Arithmetic is f32 throughout.

/// Candidate blocks of a query at `pos`: those starting at least `local` positions before it.
#[cube]
fn candidate_blocks(pos: usize, #[comptime] block: usize, #[comptime] local: usize) -> usize {
    select(pos >= local, (pos - local) / block + 1, 0usize)
}

use burn_cubecl::cubecl;
use cubecl::prelude::*;

/// Units per cube of the prep, top-k, attention and combine kernels (a power of two).
pub(crate) const CUBE: usize = 256;
/// Units per cube of the indexer kernel (a power of two).
pub(crate) const INDEX_CUBE: usize = 64;
/// Positions per attention chunk.
pub(crate) const CHUNK: usize = 64;
// Hidden positions score -1e30 (finite, so no inf - inf); anything below -1e29 counts as hidden.

/// Sum of `v` over the cube through `red` (one slot per unit; the cube size a power of two);
/// every unit gets the result.
#[cube]
fn cube_sum(red: &mut Shared<[f32]>, v: f32) -> f32 {
    let tid = UNIT_POS as usize;
    red[tid] = v;
    sync_cube();
    let mut half = CUBE_DIM / 2;
    while half > 0 {
        if UNIT_POS < half {
            let other = red[tid + half as usize];
            red[tid] += other;
        }
        sync_cube();
        half /= 2;
    }
    let total = red[0];
    sync_cube();
    total
}

/// Maximum of `v` over the cube through `red`, as [`cube_sum`].
#[cube]
fn cube_max(red: &mut Shared<[f32]>, v: f32) -> f32 {
    let tid = UNIT_POS as usize;
    red[tid] = v;
    sync_cube();
    let mut half = CUBE_DIM / 2;
    while half > 0 {
        if UNIT_POS < half {
            let other = red[tid + half as usize];
            red[tid] = select(other > red[tid], other, red[tid]);
        }
        sync_cube();
        half /= 2;
    }
    let total = red[0];
    sync_cube();
    total
}

/// This unit's share of the sum of squares of `p[base + i * ps]` for `i < n`.
#[cube]
fn sum_squares<F: Float>(
    p: &Tensor<F>,
    base: usize,
    ps: usize,
    n: usize,
    #[comptime] cube: usize,
) -> f32 {
    let mut ss = 0.0f32;
    let mut i = UNIT_POS as usize;
    while i < n {
        let x = f32::cast_from(p[base + i * ps]);
        ss += x * x;
        i += cube;
    }
    ss
}

/// Rotate-half RoPE of element `x` with partner `other` (element `(i + dr / 2) % dr`).
#[cube]
fn rope_value<F: Float>(
    x: f32,
    other: f32,
    i: usize,
    dr: usize,
    cos: &Tensor<F>,
    sin: &Tensor<F>,
) -> f32 {
    let rot = select(i < dr / 2, -other, other);
    x * f32::cast_from(cos[i * cos.stride(3)]) + rot * f32::cast_from(sin[i * sin.stride(3)])
}

/// Cubes `0 .. B * heads`: one query head each. Cubes `B * heads ..`: one batch row's cache
/// entry each. `p` `[B, 1, W]` is the layer's input projection, `cos`/`sin` `[1, 1, 1, dr]`
/// the RoPE tables of position `pos`, `wuk` `[l, heads * dn]`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ret_prep_kernel<F: Float>(
    p: &Tensor<F>,
    q_gamma: &Tensor<F>,
    kv_gamma: &Tensor<F>,
    rope_gamma: &Tensor<F>,
    index_gamma: &Tensor<F>,
    cos: &Tensor<F>,
    sin: &Tensor<F>,
    wuk: &Tensor<F>,
    q_lat: &mut Tensor<f32>,
    q_rope: &mut Tensor<f32>,
    kv: &mut Tensor<F>,
    keys: &mut Tensor<F>,
    position: &Tensor<i32>,
    batch: usize,
    scale: f32,
    eps: f32,
    inv_qd: f32,
    inv_latent: f32,
    inv_dr: f32,
    inv_index: f32,
    #[comptime] heads: usize,
    #[comptime] dn: usize,
    #[comptime] dr: usize,
    #[comptime] latent: usize,
    #[comptime] index_dim: usize,
    #[comptime] index_offset: usize,
    #[define(F)] _dtype: ElemType,
) {
    let cube_id = CUBE_POS;
    let tid = UNIT_POS as usize;
    let pos = usize::cast_from(position[0]);
    let qd = dn + dr;
    let mut red = Shared::<[f32]>::new_slice(CUBE);
    let mut xs = Shared::<[f32]>::new_slice(qd);
    let mut rot = Shared::<[f32]>::new_slice(dr);
    let ps = p.stride(2);
    if cube_id < heads * batch {
        let b = cube_id / heads;
        let hh = cube_id % heads;
        let base = b * p.stride(0) + hh * qd * ps;
        let r = 1.0
            / (cube_sum(&mut red, sum_squares::<F>(p, base, ps, qd, CUBE)) * inv_qd + eps).sqrt();
        let mut i = tid;
        while i < qd {
            xs[i] = f32::cast_from(p[base + i * ps])
                * r
                * f32::cast_from(q_gamma[i * q_gamma.stride(0)]);
            i += CUBE;
        }
        sync_cube();
        let rope_base = (b * heads + hh) * dr;
        let mut i = tid;
        while i < dr {
            rot[i] = xs[dn + i];
            i += CUBE;
        }
        sync_cube();
        let mut i = tid;
        while i < dr {
            q_rope[rope_base + i] =
                rope_value::<F>(rot[i], rot[(i + dr / 2) % dr], i, dr, cos, sin) * scale;
            i += CUBE;
        }
        // q_lat[c] = sum_n q_nope[n] W_uk[c, hh * dn + n]
        let (w0, w1) = (wuk.stride(0), wuk.stride(1));
        let lat_base = (b * heads + hh) * latent;
        let mut c = tid;
        while c < latent {
            let row = c * w0 + hh * dn * w1;
            let mut acc = 0.0f32;
            for n in 0..dn {
                acc += xs[n] * f32::cast_from(wuk[row + n * w1]);
            }
            q_lat[lat_base + c] = acc * scale;
            c += CUBE;
        }
    } else {
        let b = cube_id - heads * batch;
        let row = b * p.stride(0);
        let (k0, k1, k2) = (kv.stride(0), kv.stride(1), kv.stride(2));
        let at = b * k0 + pos * k1;
        // Latent: RMSNorm, stored as is.
        let lat = row + heads * qd * ps;
        let r = 1.0
            / (cube_sum(&mut red, sum_squares::<F>(p, lat, ps, latent, CUBE)) * inv_latent + eps)
                .sqrt();
        let mut i = tid;
        while i < latent {
            let x = f32::cast_from(p[lat + i * ps])
                * r
                * f32::cast_from(kv_gamma[i * kv_gamma.stride(0)]);
            kv[at + i * k2] = F::cast_from(x);
            i += CUBE;
        }
        // Decoupled RoPE key: RMSNorm, then RoPE.
        let rk = lat + latent * ps;
        let r =
            1.0 / (cube_sum(&mut red, sum_squares::<F>(p, rk, ps, dr, CUBE)) * inv_dr + eps).sqrt();
        let mut i = tid;
        while i < dr {
            rot[i] = f32::cast_from(p[rk + i * ps])
                * r
                * f32::cast_from(rope_gamma[i * rope_gamma.stride(0)]);
            i += CUBE;
        }
        sync_cube();
        let mut i = tid;
        while i < dr {
            kv[at + (latent + i) * k2] =
                F::cast_from(rope_value::<F>(rot[i], rot[(i + dr / 2) % dr], i, dr, cos, sin));
            i += CUBE;
        }
        // Indexer key: RMSNorm.
        let ik = row + index_offset * ps;
        let r = 1.0
            / (cube_sum(&mut red, sum_squares::<F>(p, ik, ps, index_dim, CUBE)) * inv_index + eps)
                .sqrt();
        let kat = b * keys.stride(0) + pos * keys.stride(1);
        let ks2 = keys.stride(2);
        let mut i = tid;
        while i < index_dim {
            let x = f32::cast_from(p[ik + i * ps])
                * r
                * f32::cast_from(index_gamma[i * index_gamma.stride(0)]);
            keys[kat + i * ks2] = F::cast_from(x);
            i += CUBE;
        }
    }
}

/// One cube per (batch row, candidate block): `scores[b, blk]` is the largest indexer score
/// `sum_j w_j relu(q_j . k_s) / sqrt(hi di)` over the block's positions.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ret_index_kernel<F: Float>(
    p: &Tensor<F>,
    keys: &Tensor<F>,
    scores: &mut Tensor<f32>,
    position: &Tensor<i32>,
    blocks: usize,
    w_scale: f32,
    #[comptime] index_heads: usize,
    #[comptime] index_dim: usize,
    #[comptime] q_offset: usize,
    #[comptime] block: usize,
    #[comptime] local: usize,
    #[define(F)] _dtype: ElemType,
) {
    let b = CUBE_POS / blocks;
    let blk = CUBE_POS % blocks;
    // Uniform over the cube: blocks past the candidates are left alone.
    if blk >= candidate_blocks(usize::cast_from(position[0]), block, local) {
        terminate!();
    }
    let tid = UNIT_POS as usize;
    let mut q = Shared::<[f32]>::new_slice(index_heads * index_dim);
    let mut w = Shared::<[f32]>::new_slice(index_heads);
    let mut red = Shared::<[f32]>::new_slice(INDEX_CUBE);
    let ps = p.stride(2);
    let row = b * p.stride(0);
    let mut i = tid;
    while i < index_heads * index_dim {
        q[i] = f32::cast_from(p[row + (q_offset + i) * ps]);
        i += INDEX_CUBE;
    }
    if tid < index_heads {
        w[tid] =
            f32::cast_from(p[row + (q_offset + index_heads * index_dim + index_dim + tid) * ps])
                * w_scale;
    }
    sync_cube();
    let (s0, s1, s2) = (keys.stride(0), keys.stride(1), keys.stride(2));
    let mut best = f32::new(-1.0e30f32);
    let mut s = tid;
    while s < block {
        let at = b * s0 + (blk * block + s) * s1;
        let mut score = 0.0f32;
        for j in 0..index_heads {
            let mut dot = 0.0f32;
            for d in 0..index_dim {
                dot += q[j * index_dim + d] * f32::cast_from(keys[at + d * s2]);
            }
            score += w[j] * select(dot > 0.0f32, dot, 0.0f32);
        }
        best = select(score > best, score, best);
        s += INDEX_CUBE;
    }
    let top = cube_max(&mut red, best);
    if tid == 0 {
        scores[b * blocks + blk] = top;
    }
}

/// One cube per batch row: `picked[b, r]` is the candidate block of rank `r < k` by score (ties
/// to the lower block). Nothing to pick on a dense step (at most `k` candidates). `scores` has
/// `blocks` columns, of which the first `n` are this step's candidates.
#[cube(launch_unchecked)]
pub(crate) fn ret_topk_kernel(
    scores: &Tensor<f32>,
    picked: &mut Tensor<u32>,
    position: &Tensor<i32>,
    blocks: usize,
    k: usize,
    #[comptime] block: usize,
    #[comptime] local: usize,
) {
    let b = CUBE_POS;
    let row = b * blocks;
    let n = candidate_blocks(usize::cast_from(position[0]), block, local);
    if n <= k {
        terminate!();
    }
    let mut e = UNIT_POS as usize;
    while e < n {
        let v = scores[row + e];
        let mut rank = 0usize;
        for o in 0..n {
            let w = scores[row + o];
            if w > v || (w == v && o < e) {
                rank += 1;
            }
        }
        if rank < k {
            picked[b * k + rank] = u32::cast_from(e);
        }
        e += CUBE;
    }
}

/// One cube per (batch row, chunk of `CHUNK` slots). A step is sparse when it has more than `k`
/// candidate blocks (never in `dense` mode); then slot `u` is
/// position `picked[u / block] * block + u % block` for `u < k * block` (visible when at most
/// `pos - local`) and window position `pos + 1 - local + (u - k * block)` after that; in the
/// dense layout slot `u` is position `u`. Writes the chunk's running max, sum of exponentials
/// and unnormalised latent sum per head.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ret_attend_kernel<F: Float>(
    q_lat: &Tensor<f32>,
    q_rope: &Tensor<f32>,
    kv: &Tensor<F>,
    picked: &Tensor<u32>,
    part_max: &mut Tensor<f32>,
    part_sum: &mut Tensor<f32>,
    part_acc: &mut Tensor<f32>,
    position: &Tensor<i32>,
    chunks: usize,
    k: usize,
    #[comptime] heads: usize,
    #[comptime] dr: usize,
    #[comptime] latent: usize,
    #[comptime] block: usize,
    #[comptime] local: usize,
    #[comptime] dense: bool,
    #[define(F)] _dtype: ElemType,
) {
    let b = CUBE_POS / chunks;
    let chunk = CUBE_POS % chunks;
    let pos = usize::cast_from(position[0]);
    let mut sparse = candidate_blocks(pos, block, local) > k;
    if dense {
        sparse = false;
    }
    let slots = select(sparse, k * block + local, pos + 1);
    let tid = UNIT_POS as usize;
    let mut ql = Shared::<[f32]>::new_slice(heads * latent);
    let mut qr = Shared::<[f32]>::new_slice(heads * dr);
    let mut sc = Shared::<[f32]>::new_slice(heads * CHUNK);
    let mut at = Shared::<[u32]>::new_slice(CHUNK);
    let mut ok = Shared::<[u32]>::new_slice(CHUNK);
    let mut i = tid;
    while i < heads * latent {
        ql[i] = q_lat[b * heads * latent + i];
        i += CUBE;
    }
    let mut i = tid;
    while i < heads * dr {
        qr[i] = q_rope[b * heads * dr + i];
        i += CUBE;
    }
    // Map the chunk's slots to cache positions.
    if tid < CHUNK {
        let u = chunk * CHUNK + tid;
        let mut position = 0usize;
        let mut visible = u < slots;
        if visible {
            if sparse {
                if u < k * block {
                    let blk = usize::cast_from(picked[b * k + u / block]);
                    position = blk * block + u % block;
                    visible = position + local <= pos;
                } else {
                    position = pos + 1 - local + (u - k * block);
                }
            } else {
                position = u;
            }
        }
        at[tid] = u32::cast_from(position);
        ok[tid] = select(visible, 1u32, 0u32);
    }
    sync_cube();
    let (k0, k1, k2) = (kv.stride(0), kv.stride(1), kv.stride(2));
    let row0 = b * k0;
    // Scores, one (slot, head) pair per unit at a time.
    let mut pair = tid;
    while pair < heads * CHUNK {
        let s = pair / heads;
        let hh = pair % heads;
        let mut score = f32::new(-1.0e30f32);
        if ok[s] == 1 {
            let row = row0 + usize::cast_from(at[s]) * k1;
            let mut acc = 0.0f32;
            for c in 0..latent {
                acc += ql[hh * latent + c] * f32::cast_from(kv[row + c * k2]);
            }
            for c in 0..dr {
                acc += qr[hh * dr + c] * f32::cast_from(kv[row + (latent + c) * k2]);
            }
            score = acc;
        }
        sc[hh * CHUNK + s] = score;
        pair += CUBE;
    }
    sync_cube();
    // Per head: chunk max, then exponentials in place and their sum.
    if tid < heads {
        let mut m = f32::new(-1.0e30f32);
        for s in 0..CHUNK {
            let v = sc[tid * CHUNK + s];
            m = select(v > m, v, m);
        }
        let mut total = 0.0f32;
        for s in 0..CHUNK {
            let v = sc[tid * CHUNK + s];
            let e = select(v > -1.0e29f32, (v - m).exp(), 0.0f32);
            sc[tid * CHUNK + s] = e;
            total += e;
        }
        let o = (b * chunks + chunk) * heads + tid;
        part_max[o] = m;
        part_sum[o] = total;
    }
    sync_cube();
    // Unnormalised weighted latent sums.
    let mut c = tid;
    while c < latent {
        for hh in 0..heads {
            let mut acc = 0.0f32;
            for s in 0..CHUNK {
                let e = sc[hh * CHUNK + s];
                if e > 0.0 {
                    acc += e * f32::cast_from(kv[row0 + usize::cast_from(at[s]) * k1 + c * k2]);
                }
            }
            part_acc[((b * chunks + chunk) * heads + hh) * latent + c] = acc;
        }
        c += CUBE;
    }
}

/// One cube per (batch row, head): merges the chunks' partial softmax, applies the value
/// up-projection `wuv` `[l, heads * dv]` and the sigmoid output gate from `p`. Writes `out`
/// `[B, 1, heads * dv]`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ret_combine_kernel<F: Float>(
    p: &Tensor<F>,
    wuv: &Tensor<F>,
    part_max: &Tensor<f32>,
    part_sum: &Tensor<f32>,
    part_acc: &Tensor<f32>,
    out: &mut Tensor<F>,
    chunks: usize,
    #[comptime] heads: usize,
    #[comptime] dv: usize,
    #[comptime] latent: usize,
    #[comptime] gate_offset: usize,
    #[define(F)] _dtype: ElemType,
) {
    let b = CUBE_POS / heads;
    let hh = CUBE_POS % heads;
    let tid = UNIT_POS as usize;
    let mut o = Shared::<[f32]>::new_slice(latent);
    let mut m = f32::new(-1.0e30f32);
    for ch in 0..chunks {
        let v = part_max[(b * chunks + ch) * heads + hh];
        m = select(v > m, v, m);
    }
    let mut total = 0.0f32;
    for ch in 0..chunks {
        let i = (b * chunks + ch) * heads + hh;
        total += part_sum[i] * (part_max[i] - m).exp();
    }
    let mut c = tid;
    while c < latent {
        let mut acc = 0.0f32;
        for ch in 0..chunks {
            let i = (b * chunks + ch) * heads + hh;
            acc += part_acc[i * latent + c] * (part_max[i] - m).exp();
        }
        o[c] = acc / total;
        c += CUBE;
    }
    sync_cube();
    let (w0, w1) = (wuv.stride(0), wuv.stride(1));
    let ps = p.stride(2);
    let mut v = tid;
    while v < dv {
        let col = (hh * dv + v) * w1;
        let mut acc = 0.0f32;
        for c in 0..latent {
            acc += o[c] * f32::cast_from(wuv[c * w0 + col]);
        }
        let gate = f32::cast_from(p[b * p.stride(0) + (gate_offset + hh * dv + v) * ps]);
        out[b * heads * dv + hh * dv + v] = F::cast_from(acc / (1.0 + (-gate).exp()));
        v += CUBE;
    }
}
