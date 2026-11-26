//! Fused one-token sparse retrieval decode (`SparseRetrieval::attend_selected` and the dense
//! case of `attend_latent`), in four kernels launched back to back:
//!
//! 1. [`ret_prep_kernel`]: QK-norms, partial RoPE, the absorbed query `q_nope W_uk^T` (with the
//!    softmax scale folded in), and the new position's cache entry (latent, RoPE key, indexer
//!    key at the run dtype or as 8-bit codes and a scale) written in place.
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
/// Units per cube of the indexer kernel (a power of two, whole planes).
pub(crate) const INDEX_CUBE: usize = 256;
/// Positions per attention tile; a decode attention cube takes one or more tiles.
pub(crate) const CHUNK: usize = 32;
/// Steps attending over at least this many tiles count as long (see `RetrievalDims::grids`).
pub(crate) const LONG_TILES: usize = 64;
/// Lanes per plane the decode attention and combine kernels are written for (NVIDIA warps).
pub(crate) const LANES: usize = 32;
/// Planes per cube of the decode attention and combine kernels.
pub(crate) const PLANES: usize = CUBE / LANES;
/// Units per cube of [`ret_combine_kernel`].
pub(crate) const COMBINE_CUBE: usize = 512;
/// Value columns per cube of [`ret_combine_kernel`].
pub(crate) const COMBINE_COLS: usize = 16;
/// Query heads per cube of the decode attention kernel on short steps (more cubes in flight).
pub(crate) const HEAD_GROUP: usize = 4;
/// Query heads per cube on long steps (each cached row read once for all of them).
pub(crate) const HEAD_GROUP_LONG: usize = 8;
/// Cubes per query head of the prep kernel, each computing a slice of the absorbed query.
pub(crate) const PREP_SLICES: usize = 16;
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

/// The indexer key element `i` of position `at` (row offset of `keys`, or of `codes` and the
/// position's slot `scale_at` in `scales` for 8-bit keys) as f32.
#[cube]
#[allow(clippy::too_many_arguments)]
fn index_key<F: Float>(
    keys: &Tensor<F>,
    codes: &Tensor<i8>,
    scales: &Tensor<F>,
    at: usize,
    scale_at: usize,
    i: usize,
    #[comptime] int8: bool,
) -> f32 {
    if int8 {
        f32::cast_from(codes[at + i * codes.stride(2)]) * f32::cast_from(scales[scale_at])
    } else {
        f32::cast_from(keys[at + i * keys.stride(2)])
    }
}

/// The next position of batch row `b`: `position` holds one per row (separate streams, slot
/// batching) or one for every row.
#[cube]
fn row_position(position: &Tensor<i32>, b: usize) -> usize {
    let i = select(position.shape(0) > 1, b, 0usize);
    usize::cast_from(position[i * position.stride(0)])
}

/// Whether batch row `b` takes part in this step: a negative position marks an idle row of a
/// slot pool (it neither reads nor writes; its output rows are left unset).
#[cube]
fn row_active(position: &Tensor<i32>, b: usize) -> bool {
    let i = select(position.shape(0) > 1, b, 0usize);
    position[i * position.stride(0)] >= 0
}

/// Rotate-half RoPE of element `x` of batch row `b` with partner `other` (element
/// `(i + dr / 2) % dr`); the tables hold one position per row or one for every row.
#[cube]
#[allow(clippy::too_many_arguments)]
fn rope_value<F: Float>(
    x: f32,
    other: f32,
    i: usize,
    dr: usize,
    cos: &Tensor<F>,
    sin: &Tensor<F>,
    b: usize,
) -> f32 {
    let rot = select(i < dr / 2, -other, other);
    let c = select(cos.shape(0) > 1, b, 0usize) * cos.stride(0) + i * cos.stride(3);
    let s = select(sin.shape(0) > 1, b, 0usize) * sin.stride(0) + i * sin.stride(3);
    x * f32::cast_from(cos[c]) + rot * f32::cast_from(sin[s])
}

/// Cubes `0 .. B * heads * PREP_SLICES`: a slice of one query head's absorbed query each (the
/// first slice also writes the RoPE part). Cubes after them, three per batch row: its cache
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
    codes: &mut Tensor<i8>,
    scales: &mut Tensor<F>,
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
    #[comptime] int8: bool,
    #[define(F)] _dtype: ElemType,
) {
    let cube_id = CUBE_POS;
    let tid = UNIT_POS as usize;
    let qd = dn + dr;
    let mut red = Shared::<[f32]>::new_slice(CUBE);
    let mut xs = Shared::<[f32]>::new_slice(qd);
    let mut rot = Shared::<[f32]>::new_slice(dr);
    let ps = p.stride(2);
    if cube_id < heads * batch * PREP_SLICES {
        let b = cube_id / (heads * PREP_SLICES);
        if !row_active(position, b) {
            terminate!();
        }
        let hh = (cube_id / PREP_SLICES) % heads;
        let slice = cube_id % PREP_SLICES;
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
        if slice == 0 {
            let mut i = tid;
            while i < dr {
                q_rope[rope_base + i] =
                    rope_value::<F>(rot[i], rot[(i + dr / 2) % dr], i, dr, cos, sin, b) * scale;
                i += CUBE;
            }
        }
        // q_lat[c] = sum_n q_nope[n] W_uk[c, hh * dn + n] for this cube's slice of c: one plane
        // per c, its lanes reading the row's n coalesced.
        let (w0, w1) = (wuk.stride(0), wuk.stride(1));
        let lat_base = (b * heads + hh) * latent;
        let lane = UNIT_POS_PLANE as usize;
        let per_slice = comptime!(latent.div_ceil(PREP_SLICES));
        let end = select((slice + 1) * per_slice < latent, (slice + 1) * per_slice, latent);
        let mut c = slice * per_slice + tid / LANES;
        while c < end {
            let row = c * w0 + hh * dn * w1;
            let mut acc = 0.0f32;
            let mut n = lane;
            while n < dn {
                acc += xs[n] * f32::cast_from(wuk[row + n * w1]);
                n += LANES;
            }
            let total = plane_sum(acc);
            if lane == 0 {
                q_lat[lat_base + c] = total * scale;
            }
            c += PLANES;
        }
    } else {
        // Three cubes per batch row: the latent, the RoPE key, the indexer key.
        let entry = cube_id - heads * batch * PREP_SLICES;
        let b = entry / 3;
        let part = entry % 3;
        if !row_active(position, b) {
            terminate!();
        }
        let pos = row_position(position, b);
        let row = b * p.stride(0);
        let (k0, k1, k2) = (kv.stride(0), kv.stride(1), kv.stride(2));
        let at = b * k0 + pos * k1;
        // Latent: RMSNorm, stored as is.
        let lat = row + heads * qd * ps;
        if part == 0 {
            let r = 1.0
                / (cube_sum(&mut red, sum_squares::<F>(p, lat, ps, latent, CUBE)) * inv_latent
                    + eps)
                    .sqrt();
            let mut i = tid;
            while i < latent {
                let x = f32::cast_from(p[lat + i * ps])
                    * r
                    * f32::cast_from(kv_gamma[i * kv_gamma.stride(0)]);
                kv[at + i * k2] = F::cast_from(x);
                i += CUBE;
            }
        }
        // Decoupled RoPE key: RMSNorm, then RoPE.
        let rk = lat + latent * ps;
        if part == 1 {
            let r = 1.0
                / (cube_sum(&mut red, sum_squares::<F>(p, rk, ps, dr, CUBE)) * inv_dr + eps).sqrt();
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
                kv[at + (latent + i) * k2] = F::cast_from(rope_value::<F>(
                    rot[i],
                    rot[(i + dr / 2) % dr],
                    i,
                    dr,
                    cos,
                    sin,
                    b,
                ));
                i += CUBE;
            }
        }
        // Indexer key: RMSNorm, then stored as is or as 8-bit codes with the scale
        // max |k| / 127 (at the run dtype, as the reference stores it).
        let ik = row + index_offset * ps;
        if part == 2 {
            let r = 1.0
                / (cube_sum(&mut red, sum_squares::<F>(p, ik, ps, index_dim, CUBE)) * inv_index
                    + eps)
                    .sqrt();
            if int8 {
                let mut top = 0.0f32;
                let mut i = tid;
                while i < index_dim {
                    let x = f32::cast_from(p[ik + i * ps])
                        * r
                        * f32::cast_from(index_gamma[i * index_gamma.stride(0)]);
                    let x = f32::cast_from(F::cast_from(x));
                    top = select(x.abs() > top, x.abs(), top);
                    i += CUBE;
                }
                let top = cube_max(&mut red, top);
                let step = f32::cast_from(F::cast_from(top / 127.0f32));
                let step = select(step > 1.0e-30f32, step, 1.0e-30f32);
                let cat = b * codes.stride(0) + pos * codes.stride(1);
                let cs2 = codes.stride(2);
                let mut i = tid;
                while i < index_dim {
                    let x = f32::cast_from(p[ik + i * ps])
                        * r
                        * f32::cast_from(index_gamma[i * index_gamma.stride(0)]);
                    // Rounded to the run dtype at each step, as the reference's tensor ops are.
                    let x = f32::cast_from(F::cast_from(x));
                    let code =
                        f32::cast_from(F::cast_from(x / step)).round().clamp(-127.0f32, 127.0f32);
                    codes[cat + i * cs2] = i8::cast_from(code);
                    i += CUBE;
                }
                let sat = b * scales.stride(0) + pos * scales.stride(1);
                if tid == 0 {
                    scales[sat] = F::cast_from(step);
                }
            } else {
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
    }
}

/// One cube per (batch row, candidate block): `scores[b, blk]` is the largest indexer score
/// `sum_j w_j relu(q_j . k_s) / sqrt(hi di)` over the block's positions.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ret_index_kernel<F: Float>(
    p: &Tensor<F>,
    keys: &Tensor<F>,
    codes: &Tensor<i8>,
    scales: &Tensor<F>,
    scores: &mut Tensor<f32>,
    position: &Tensor<i32>,
    blocks: usize,
    w_scale: f32,
    #[comptime] index_heads: usize,
    #[comptime] index_dim: usize,
    #[comptime] q_offset: usize,
    #[comptime] block: usize,
    #[comptime] local: usize,
    #[comptime] int8: bool,
    #[define(F)] _dtype: ElemType,
) {
    let b = CUBE_POS / blocks;
    let blk = CUBE_POS % blocks;
    if !row_active(position, b) {
        terminate!();
    }
    // Uniform over the cube: blocks past the candidates are left alone.
    if blk >= candidate_blocks(row_position(position, b), block, local) {
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
    // One plane per position, its lanes splitting the key (coalesced, unrolled), the heads'
    // partial dot products in registers.
    let lane = UNIT_POS_PLANE as usize;
    let steps = comptime!(index_dim.div_ceil(LANES));
    let mut best = f32::new(-1.0e30f32);
    let mut s = tid / LANES;
    while s < block {
        let position = blk * block + s;
        let at = select(
            int8,
            b * codes.stride(0) + position * codes.stride(1),
            b * keys.stride(0) + position * keys.stride(1),
        );
        let scale_at = b * scales.stride(0) + position * scales.stride(1);
        let mut dots = Array::<f32>::new(index_heads);
        #[unroll]
        for j in 0..index_heads {
            dots[j] = 0.0f32;
        }
        #[unroll]
        for m in 0..steps {
            let d = lane + m * LANES;
            if d < index_dim {
                let x = index_key::<F>(keys, codes, scales, at, scale_at, d, int8);
                #[unroll]
                for j in 0..index_heads {
                    dots[j] += q[j * index_dim + d] * x;
                }
            }
        }
        let mut score = 0.0f32;
        #[unroll]
        for j in 0..index_heads {
            let dot = plane_sum(dots[j]);
            score += w[j] * select(dot > 0.0f32, dot, 0.0f32);
        }
        best = select(score > best, score, best);
        s += INDEX_CUBE / LANES;
    }
    let top = cube_max(&mut red, best);
    if tid == 0 {
        scores[b * blocks + blk] = top;
    }
}

/// A score's bits as a key that orders like the score: flip every bit of a negative float, only
/// the sign bit of a positive one.
#[cube]
fn order_key(x: f32) -> u32 {
    let bits = u32::reinterpret(x);
    select(bits >> 31u32 == 1u32, !bits, bits | 0x8000_0000u32)
}

/// One cube per batch row: `picked[b, ..k]` gets the `k` candidate blocks with the highest
/// scores (ties to the lower block), in block order. Nothing to pick on a dense step (at most
/// `k` candidates). `scores` has `blocks` columns, of which the first `n` are this step's
/// candidates.
///
/// A radix select on the scores' order keys: four passes over 8-bit digits, most significant
/// first, each a shared-memory histogram of the candidates matching the digits chosen so far,
/// find the `k`-th highest key `T` and how many of the keys equal to it make the cut. Then each
/// unit takes a contiguous range of blocks, counts its keys above `T` and equal to it, and two
/// prefix sums over the units give every unit its share of the tied keys (the lowest blocks
/// first) and where its picks go. About `4 + 2` reads of each score instead of one per pair.
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
    if !row_active(position, b) {
        terminate!();
    }
    let n = candidate_blocks(row_position(position, b), block, local);
    if n <= k {
        terminate!();
    }
    let tid = UNIT_POS as usize;
    let hist = Shared::<[Atomic<u32>]>::new_slice(256usize);
    let mut state = Shared::<[u32]>::new_slice(2usize);
    let mut gt_counts = Shared::<[u32]>::new_slice(CUBE);
    let mut eq_counts = Shared::<[u32]>::new_slice(CUBE);
    let mut prefix = 0u32;
    let mut mask = 0u32;
    let mut remaining = u32::cast_from(k);
    #[unroll]
    for pass in 0..4u32 {
        let shift = comptime!(24u32 - 8u32 * pass);
        hist[tid].store(0u32);
        sync_cube();
        let mut e = tid;
        while e < n {
            let key = order_key(scores[row + e]);
            if key & mask == prefix {
                hist[((key >> shift) & 255u32) as usize].fetch_add(1u32);
            }
            e += CUBE;
        }
        sync_cube();
        if tid == 0 {
            // The highest digit whose bin, added to the bins above it, reaches `remaining`.
            let mut left = remaining;
            let mut digit = 255u32;
            let mut found = false;
            for i in 0..256u32 {
                let d = 255u32 - i;
                let count = hist[d as usize].load();
                if !found {
                    if count >= left {
                        digit = d;
                        found = true;
                    } else {
                        left -= count;
                    }
                }
            }
            state[0] = digit;
            state[1] = left;
        }
        sync_cube();
        prefix |= state[0] << shift;
        mask |= 255u32 << shift;
        remaining = state[1];
        sync_cube();
    }
    // `prefix` is now the k-th highest key; `remaining` of the keys equal to it are picked.
    let per = n.div_ceil(CUBE);
    let lo = select(tid * per < n, tid * per, n);
    let hi = select(lo + per < n, lo + per, n);
    let mut gt = 0u32;
    let mut eq = 0u32;
    for e in lo..hi {
        let key = order_key(scores[row + e]);
        gt += select(key > prefix, 1u32, 0u32);
        eq += select(key == prefix, 1u32, 0u32);
    }
    gt_counts[tid] = gt;
    eq_counts[tid] = eq;
    sync_cube();
    if tid == 0 {
        // Exclusive prefix sums: first of the tied keys before each range (then how many of
        // them the range takes), then of the picks before each range.
        let mut ties_before = 0u32;
        let mut picks_before = 0u32;
        for u in 0..CUBE {
            let ties = eq_counts[u];
            let needed = select(ties_before < remaining, remaining - ties_before, 0u32);
            let take = select(ties < needed, ties, needed);
            ties_before += ties;
            let picks = gt_counts[u] + take;
            eq_counts[u] = take;
            gt_counts[u] = picks_before;
            picks_before += picks;
        }
    }
    sync_cube();
    let mut at = gt_counts[tid] as usize;
    let mut take = eq_counts[tid];
    for e in lo..hi {
        let key = order_key(scores[row + e]);
        let tied = key == prefix && take > 0u32;
        if key > prefix || tied {
            picked[b * k + at] = u32::cast_from(e);
            at += 1;
        }
        if tied {
            take -= 1u32;
        }
    }
}

/// One cube per (batch row, chunk of `tiles` tiles of `CHUNK` slots, group of `hg` query
/// heads). A step is sparse when it has more than `k` candidate blocks (never in `dense` mode);
/// then slot `u` is position `picked[u / block] * block + u % block` for `u < k * block`
/// (visible when at most `pos - local`) and window position `pos + 1 - local + (u - k * block)`
/// after that; in the dense layout slot `u` is position `u`. Writes the chunk's max, sum of
/// exponentials and unnormalised latent sum for each of its heads.
///
/// Per tile, scores: each plane takes `CHUNK / 8` slots and its lanes split the row width
/// (coalesced, the loads unrolled), every lane keeping the partial dot products of those slots
/// and the group's heads in registers, then one plane sum each. Softmax: one plane per head,
/// one lane per slot, a running max carried from tile to tile. Weighted sums: each unit owns
/// latent columns, one cached value feeding every head. Several tiles per cube at long contexts
/// keep the partials the combine merges few.
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
    tiles: usize,
    k: usize,
    #[comptime] heads: usize,
    #[comptime] dr: usize,
    #[comptime] latent: usize,
    #[comptime] block: usize,
    #[comptime] local: usize,
    #[comptime] dense: bool,
    #[comptime] hg: usize,
    #[define(F)] _dtype: ElemType,
) {
    let groups = comptime!(heads / hg);
    let width = comptime!(latent + dr);
    let per = comptime!(CHUNK / PLANES);
    let steps = comptime!(width.div_ceil(LANES));
    let cols = comptime!(latent.div_ceil(CUBE));
    let b = CUBE_POS / (chunks * groups);
    if !row_active(position, b) {
        terminate!();
    }
    let rest = CUBE_POS % (chunks * groups);
    let chunk = rest / groups;
    let h0 = (rest % groups) * hg;
    let pos = row_position(position, b);
    let mut sparse = candidate_blocks(pos, block, local) > k;
    if dense {
        sparse = false;
    }
    let slots = select(sparse, k * block + local, pos + 1);
    let tid = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane = tid / LANES;
    let mut q_sh = Shared::<[f32]>::new_slice(hg * width);
    let mut sc = Shared::<[f32]>::new_slice(hg * CHUNK);
    let mut at = Shared::<[u32]>::new_slice(CHUNK);
    let mut ok = Shared::<[u32]>::new_slice(CHUNK);
    // Per head: running max and sum, and this tile's rescale of the sums so far.
    let mut run_m = Shared::<[f32]>::new_slice(hg);
    let mut run_l = Shared::<[f32]>::new_slice(hg);
    let mut corr = Shared::<[f32]>::new_slice(hg);
    let mut i = tid;
    while i < hg * width {
        let h = h0 + i / width;
        let d = i % width;
        if d < latent {
            q_sh[i] = q_lat[(b * heads + h) * latent + d];
        } else {
            q_sh[i] = q_rope[(b * heads + h) * dr + d - latent];
        }
        i += CUBE;
    }
    if tid < hg {
        run_m[tid] = f32::new(-1.0e30f32);
        run_l[tid] = 0.0f32;
    }
    let mut acc = Array::<f32>::new(comptime!(hg * cols));
    #[unroll]
    for a in 0..hg * cols {
        acc[a] = 0.0f32;
    }
    let (k0, k1, k2) = (kv.stride(0), kv.stride(1), kv.stride(2));
    let row0 = b * k0;
    let first = chunk * tiles * CHUNK;
    // Tiles past the step's slots are skipped (uniform over the cube).
    let mut t = 0usize;
    while t < tiles && first + t * CHUNK < slots {
        // Map the tile's slots to cache positions.
        if tid < CHUNK {
            let u = first + t * CHUNK + tid;
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
            // Hidden slots read position 0 and weigh zero.
            at[tid] = u32::cast_from(select(visible, position, 0usize));
            ok[tid] = select(visible, 1u32, 0u32);
        }
        sync_cube();
        let mut rows = Array::<usize>::new(per);
        #[unroll]
        for j in 0..per {
            rows[j] = row0 + usize::cast_from(at[plane * per + j]) * k1;
        }
        let mut part = Array::<f32>::new(comptime!(per * hg));
        #[unroll]
        for a in 0..per * hg {
            part[a] = 0.0f32;
        }
        #[unroll]
        for m in 0..steps {
            let d = lane + m * LANES;
            if d < width {
                let mut qv = Array::<f32>::new(hg);
                #[unroll]
                for h in 0..hg {
                    qv[h] = q_sh[h * width + d];
                }
                #[unroll]
                for j in 0..per {
                    let x = f32::cast_from(kv[rows[j] + d * k2]);
                    #[unroll]
                    for h in 0..hg {
                        part[j * hg + h] += qv[h] * x;
                    }
                }
            }
        }
        #[unroll]
        for j in 0..per {
            let s = plane * per + j;
            #[unroll]
            for h in 0..hg {
                let total = plane_sum(part[j * hg + h]);
                if lane == 0 {
                    sc[h * CHUNK + s] = select(ok[s] == 1, total, f32::new(-1.0e30f32));
                }
            }
        }
        sync_cube();
        // Per head: the new running max, exponentials in place, the running sum.
        if plane < hg {
            let v = sc[plane * CHUNK + lane];
            let old = run_m[plane];
            let tile_top = plane_max(v);
            let top = select(tile_top > old, tile_top, old);
            let e = select(v > -1.0e29f32, (v - top).exp(), 0.0f32);
            sc[plane * CHUNK + lane] = e;
            let total = plane_sum(e);
            if lane == 0 {
                let factor = (old - top).exp();
                run_l[plane] = run_l[plane] * factor + total;
                run_m[plane] = top;
                corr[plane] = factor;
            }
        }
        sync_cube();
        // Unnormalised weighted latent sums.
        #[unroll]
        for j in 0..cols {
            let c = tid + j * CUBE;
            if c < latent {
                #[unroll]
                for h in 0..hg {
                    acc[h * cols + j] *= corr[h];
                }
                #[unroll]
                for s in 0..CHUNK {
                    let x = f32::cast_from(kv[row0 + usize::cast_from(at[s]) * k1 + c * k2]);
                    #[unroll]
                    for h in 0..hg {
                        acc[h * cols + j] += sc[h * CHUNK + s] * x;
                    }
                }
            }
        }
        sync_cube();
        t += 1;
    }
    if tid < hg {
        let o = (b * chunks + chunk) * heads + h0 + tid;
        part_max[o] = run_m[tid];
        part_sum[o] = run_l[tid];
    }
    #[unroll]
    for j in 0..cols {
        let c = tid + j * CUBE;
        if c < latent {
            #[unroll]
            for h in 0..hg {
                part_acc[((b * chunks + chunk) * heads + h0 + h) * latent + c] = acc[h * cols + j];
            }
        }
    }
}

/// One cube of `COMBINE_CUBE` units per (batch row, head, tile of `COMBINE_COLS` value
/// columns): merges the chunks' partial softmax, then the tile's columns of the value
/// up-projection `wuv` `[l, heads * dv]` and the sigmoid output gate from `p`. Writes `out`
/// `[B, 1, heads * dv]`.
///
/// Every plane finds the overall max and normaliser itself (lanes striding the chunks: no
/// cube-wide reduction), each unit merges latent columns over the chunks (four per round, the
/// loads overlapping), and the projection gives each unit a column and every
/// `COMBINE_CUBE / COMBINE_COLS`-th latent row (two rows of contiguous columns per plane load).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ret_combine_kernel<F: Float>(
    p: &Tensor<F>,
    wuv: &Tensor<F>,
    part_max: &Tensor<f32>,
    part_sum: &Tensor<f32>,
    part_acc: &Tensor<f32>,
    out: &mut Tensor<F>,
    position: &Tensor<i32>,
    chunks: usize,
    #[comptime] heads: usize,
    #[comptime] dv: usize,
    #[comptime] latent: usize,
    #[comptime] gate_offset: usize,
    #[define(F)] _dtype: ElemType,
) {
    let tiles = comptime!(dv.div_ceil(COMBINE_COLS));
    let groups = comptime!(COMBINE_CUBE / COMBINE_COLS);
    let cols = comptime!(latent.div_ceil(COMBINE_CUBE));
    let bh = CUBE_POS / tiles;
    let tile = CUBE_POS % tiles;
    let b = bh / heads;
    let hh = bh % heads;
    if !row_active(position, b) {
        terminate!();
    }
    let tid = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let mut o = Shared::<[f32]>::new_slice(latent);
    let mut red = Shared::<[f32]>::new_slice(COMBINE_CUBE);
    let mut m = f32::new(-1.0e30f32);
    let mut ch = lane;
    while ch < chunks {
        let v = part_max[(b * chunks + ch) * heads + hh];
        m = select(v > m, v, m);
        ch += LANES;
    }
    let m = plane_max(m);
    let mut total = 0.0f32;
    let mut ch = lane;
    while ch < chunks {
        let i = (b * chunks + ch) * heads + hh;
        total += part_sum[i] * (part_max[i] - m).exp();
        ch += LANES;
    }
    let inv = 1.0f32 / plane_sum(total);
    let mut acc = Array::<f32>::new(cols);
    #[unroll]
    for j in 0..cols {
        acc[j] = 0.0f32;
    }
    let stride = heads * latent;
    let mut j = 0usize;
    while j < chunks {
        let quad = j + 4 <= chunks;
        let i0 = (b * chunks + j) * heads + hh;
        let w0 = (part_max[i0] - m).exp();
        let mut w1 = 0.0f32;
        let mut w2 = 0.0f32;
        let mut w3 = 0.0f32;
        if quad {
            w1 = (part_max[i0 + heads] - m).exp();
            w2 = (part_max[i0 + 2 * heads] - m).exp();
            w3 = (part_max[i0 + 3 * heads] - m).exp();
        }
        let base = i0 * latent;
        #[unroll]
        for cc in 0..cols {
            let c = tid + cc * COMBINE_CUBE;
            if c < latent {
                if quad {
                    acc[cc] += part_acc[base + c] * w0
                        + part_acc[base + stride + c] * w1
                        + part_acc[base + 2 * stride + c] * w2
                        + part_acc[base + 3 * stride + c] * w3;
                } else {
                    acc[cc] += part_acc[base + c] * w0;
                }
            }
        }
        j += select(quad, 4usize, 1usize);
    }
    #[unroll]
    for j in 0..cols {
        let c = tid + j * COMBINE_CUBE;
        if c < latent {
            o[c] = acc[j] * inv;
        }
    }
    sync_cube();
    let col = tid % COMBINE_COLS;
    let group = tid / COMBINE_COLS;
    let v = tile * COMBINE_COLS + col;
    let (w0, w1) = (wuv.stride(0), wuv.stride(1));
    let at = (hh * dv + v) * w1;
    let mut sum = 0.0f32;
    if v < dv {
        let mut c = group;
        while c + 3 * groups < latent {
            sum += o[c] * f32::cast_from(wuv[c * w0 + at])
                + o[c + groups] * f32::cast_from(wuv[(c + groups) * w0 + at])
                + o[c + 2 * groups] * f32::cast_from(wuv[(c + 2 * groups) * w0 + at])
                + o[c + 3 * groups] * f32::cast_from(wuv[(c + 3 * groups) * w0 + at]);
            c += 4 * groups;
        }
        while c < latent {
            sum += o[c] * f32::cast_from(wuv[c * w0 + at]);
            c += groups;
        }
    }
    red[tid] = sum;
    sync_cube();
    if tid < COMBINE_COLS && v < dv {
        let mut total = 0.0f32;
        for g in 0..groups {
            total += red[g * COMBINE_COLS + tid];
        }
        let ps = p.stride(2);
        let gate = f32::cast_from(p[b * p.stride(0) + (gate_offset + hh * dv + v) * ps]);
        out[b * heads * dv + hh * dv + v] = F::cast_from(total / (1.0 + (-gate).exp()));
    }
}

/// Lanes per plane the prefill attention kernel is written for (NVIDIA warps).
pub(crate) const PREFILL_LANES: usize = 32;
/// Planes per cube of [`ret_prefill_attend_kernel`].
pub(crate) const PREFILL_PLANES: usize = CUBE / PREFILL_LANES;

/// Positions per tile of [`ret_prefill_attend_kernel`]: one plane.
pub(crate) const PREFILL_TILE: usize = PREFILL_LANES;

/// Prefill selection for a group of `q` queries (chunk rows `t0 ..`, absolute positions
/// `start + t0 ..`), one cube per (batch row, query). `raw` `[B, hi, q, S']` holds the indexer
/// dot products `q_j . k_s` over the candidate prefix (for 8-bit keys against the codes, the
/// position's scale in `scales` `[B, S, 1]` applied after the ReLU), `w` `[B, T, hi]` the head
/// weights. A query's block scores go to its row of the scratch `blocks` `[B, q, nb]`; then
/// `picked[b, t, ..]` gets its top `k` candidate blocks (ties to the lower block) or, with at
/// most `k` candidates, all of them, and `count[b, t]` how many.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ret_select_kernel<F: Float>(
    raw: &Tensor<F>,
    w: &Tensor<F>,
    scales: &Tensor<F>,
    blocks: &mut Tensor<f32>,
    picked: &mut Tensor<i32>,
    count: &mut Tensor<i32>,
    start: usize,
    t0: usize,
    q: usize,
    t_total: usize,
    nb: usize,
    k: usize,
    w_scale: f32,
    #[comptime] index_heads: usize,
    #[comptime] block: usize,
    #[comptime] local: usize,
    #[comptime] int8: bool,
    #[define(F)] _dtype: ElemType,
) {
    let b = CUBE_POS / q;
    let i = CUBE_POS % q;
    let t = t0 + i;
    let pos = start + t;
    let n = candidate_blocks(pos, block, local);
    let tid = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let width = PLANE_DIM as usize;
    let planes = CUBE / width;
    let plane = tid / width;
    let mut wts = Shared::<[f32]>::new_slice(index_heads);
    let mut red_v = Shared::<[f32]>::new_slice(CUBE);
    let mut red_i = Shared::<[u32]>::new_slice(CUBE);
    if tid < index_heads {
        wts[tid] =
            f32::cast_from(w[b * w.stride(0) + t * w.stride(1) + tid * w.stride(2)]) * w_scale;
    }
    sync_cube();
    let (r0, r1, r2, r3) = (raw.stride(0), raw.stride(1), raw.stride(2), raw.stride(3));
    let row = (b * q + i) * nb;
    // Block maxima, one plane per block.
    let mut blk = plane;
    while blk < n {
        let mut best = f32::new(-1.0e30f32);
        let mut s = lane;
        while s < block {
            let position = blk * block + s;
            let at = b * r0 + i * r2 + position * r3;
            let mut score = 0.0f32;
            for j in 0..index_heads {
                let dot = f32::cast_from(raw[at + j * r1]);
                score += wts[j] * select(dot > 0.0f32, dot, 0.0f32);
            }
            if int8 {
                score *= f32::cast_from(scales[b * scales.stride(0) + position * scales.stride(1)]);
            }
            best = select(score > best, score, best);
            s += width;
        }
        let top = plane_max(best);
        if lane == 0 {
            blocks[row + blk] = top;
        }
        blk += planes;
    }
    sync_cube();
    let out = (b * t_total + t) * k;
    if n <= k {
        let mut r = tid;
        while r < n {
            picked[out + r] = i32::cast_from(r);
            r += CUBE;
        }
        if tid == 0 {
            count[b * t_total + t] = i32::cast_from(n);
        }
    } else {
        for r in 0..k {
            // The best block left: highest score, then lowest index.
            let mut v = f32::new(-3.0e38f32);
            let mut e_best = u32::cast_from(n);
            let mut e = tid;
            while e < n {
                let x = blocks[row + e];
                if x > v {
                    v = x;
                    e_best = u32::cast_from(e);
                }
                e += CUBE;
            }
            red_v[tid] = v;
            red_i[tid] = e_best;
            sync_cube();
            let mut half = CUBE_DIM / 2;
            while half > 0 {
                if UNIT_POS < half {
                    let ov = red_v[tid + half as usize];
                    let oi = red_i[tid + half as usize];
                    if ov > red_v[tid] || (ov == red_v[tid] && oi < red_i[tid]) {
                        red_v[tid] = ov;
                        red_i[tid] = oi;
                    }
                }
                sync_cube();
                half /= 2;
            }
            if tid == 0 {
                let e = usize::cast_from(red_i[0]);
                picked[out + r] = i32::cast_from(e);
                blocks[row + e] = f32::new(-3.0e38f32);
            }
            sync_cube();
        }
        if tid == 0 {
            count[b * t_total + t] = i32::cast_from(k);
        }
    }
}

/// Sparse prefill attention on the cached latents, one cube per (batch row, query of the chunk
/// at absolute position `start + t`): the query's `count` picked blocks (their positions at
/// least `local` before it) and its local window (the last `min(local, pos + 1)` positions),
/// with an online softmax over tiles of `tile` positions. `q` `[B, H, T, l + dr]` is the
/// absorbed query with the softmax scale folded in, `kv` `[B, S, l + dr]` the cache (both read
/// through their strides). Writes the normalised latent sums `out` `[B, T, H, l]`
/// (contiguous), before the value up-projection.
///
/// The scores of a tile: each plane takes `tile / 8` positions and its lanes split the row
/// width, every lane keeping the partial dot products of those positions and all heads in
/// registers (one query value and one cached value feed many products, the loads unrolled so
/// they overlap). The weighted sums: each unit owns latent columns of every head. The tile is
/// a plane wide, so the softmax update is one lane per position.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ret_prefill_attend_kernel<F: Float>(
    q: &Tensor<F>,
    kv: &Tensor<F>,
    picked: &Tensor<i32>,
    count: &Tensor<i32>,
    out: &mut Tensor<F>,
    start: usize,
    t_total: usize,
    k: usize,
    #[comptime] heads: usize,
    #[comptime] latent: usize,
    #[comptime] dr: usize,
    #[comptime] block: usize,
    #[comptime] local: usize,
    #[comptime] tile: usize,
    #[define(F)] _dtype: ElemType,
) {
    let b = CUBE_POS / t_total;
    let t = CUBE_POS % t_total;
    let pos = start + t;
    let width = comptime!(latent + dr);
    let cols = comptime!(latent.div_ceil(CUBE));
    let per = comptime!(tile / PREFILL_PLANES);
    let steps = comptime!(width.div_ceil(PREFILL_LANES));
    let tid = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane = tid / PREFILL_LANES;
    let mut q_sh = Shared::<[F]>::new_slice(heads * width);
    let mut sc = Shared::<[f32]>::new_slice(heads * tile);
    let mut at = Shared::<[u32]>::new_slice(tile);
    let mut ok = Shared::<[u32]>::new_slice(tile);
    let mut m_sh = Shared::<[f32]>::new_slice(heads);
    let mut l_sh = Shared::<[f32]>::new_slice(heads);
    let mut corr = Shared::<[f32]>::new_slice(heads);
    let (q0, q1, q2, q3) = (q.stride(0), q.stride(1), q.stride(2), q.stride(3));
    let mut idx = tid;
    while idx < heads * width {
        let h = idx / width;
        let d = idx % width;
        q_sh[idx] = q[b * q0 + h * q1 + t * q2 + d * q3];
        idx += CUBE;
    }
    if tid < heads {
        m_sh[tid] = f32::new(-1.0e30f32);
        l_sh[tid] = 0.0f32;
    }
    let mut acc = Array::<f32>::new(comptime!(heads * cols));
    #[unroll]
    for a in 0..heads * cols {
        acc[a] = 0.0f32;
    }
    let n_sel = usize::cast_from(count[b * t_total + t]);
    let window = select(pos + 1 < local, pos + 1, local);
    let sel_slots = n_sel * block;
    let slots = sel_slots + window;
    let win_start = pos + 1 - window;
    let pick_row = (b * t_total + t) * k;
    let (k0, k1, k2) = (kv.stride(0), kv.stride(1), kv.stride(2));
    let row0 = b * k0;
    sync_cube();
    let mut base = 0usize;
    while base < slots {
        if tid < tile {
            let u = base + tid;
            let mut position = 0usize;
            let mut visible = u < slots;
            if visible {
                if u < sel_slots {
                    let blk = usize::cast_from(picked[pick_row + u / block]);
                    position = blk * block + u % block;
                    visible = position + local <= pos;
                } else {
                    position = win_start + (u - sel_slots);
                }
            }
            at[tid] = u32::cast_from(position);
            ok[tid] = select(visible, 1u32, 0u32);
        }
        sync_cube();
        // Scores: plane `plane` takes positions `plane * per ..`, its lanes splitting the width.
        let mut part = Array::<f32>::new(comptime!(per * heads));
        #[unroll]
        for a in 0..per * heads {
            part[a] = 0.0f32;
        }
        let mut rows = Array::<usize>::new(per);
        #[unroll]
        for j in 0..per {
            rows[j] = row0 + usize::cast_from(at[plane * per + j]) * k1;
        }
        #[unroll]
        for m in 0..steps {
            let d = lane + m * PREFILL_LANES;
            if d < width {
                let mut qv = Array::<f32>::new(heads);
                #[unroll]
                for h in 0..heads {
                    qv[h] = f32::cast_from(q_sh[h * width + d]);
                }
                #[unroll]
                for j in 0..per {
                    // Hidden positions read row 0 (always cached) and are masked below.
                    let x = f32::cast_from(kv[rows[j] + d * k2]);
                    #[unroll]
                    for h in 0..heads {
                        part[j * heads + h] += qv[h] * x;
                    }
                }
            }
        }
        #[unroll]
        for j in 0..per {
            let s = plane * per + j;
            #[unroll]
            for h in 0..heads {
                let total = plane_sum(part[j * heads + h]);
                if lane == 0 {
                    sc[h * tile + s] = select(ok[s] == 1, total, f32::new(-1.0e30f32));
                }
            }
        }
        sync_cube();
        // Online softmax per head, one plane per head and one lane per position: new running
        // max, weights in place, rescale factor.
        let mut h = plane;
        while h < heads {
            let mut v = f32::new(-1.0e30f32);
            if lane < tile {
                v = sc[h * tile + lane];
            }
            let old = m_sh[h];
            let tile_top = plane_max(v);
            let top = select(tile_top > old, tile_top, old);
            let e = select(v > -1.0e29f32, (v - top).exp(), 0.0f32);
            if lane < tile {
                sc[h * tile + lane] = e;
            }
            let total = plane_sum(e);
            if lane == 0 {
                let factor = (old - top).exp();
                l_sh[h] = l_sh[h] * factor + total;
                m_sh[h] = top;
                corr[h] = factor;
            }
            h += PREFILL_PLANES;
        }
        sync_cube();
        // Weighted latent sums, each unit owning `cols` latent columns of every head.
        #[unroll]
        for j in 0..cols {
            let c = tid + j * CUBE;
            if c < latent {
                #[unroll]
                for h in 0..heads {
                    acc[h * cols + j] *= corr[h];
                }
                #[unroll]
                for s in 0..tile {
                    // Hidden positions weigh zero.
                    let x = f32::cast_from(kv[row0 + usize::cast_from(at[s]) * k1 + c * k2]);
                    #[unroll]
                    for h in 0..heads {
                        acc[h * cols + j] += sc[h * tile + s] * x;
                    }
                }
            }
        }
        sync_cube();
        base += tile;
    }
    #[unroll]
    for j in 0..cols {
        let c = tid + j * CUBE;
        if c < latent {
            #[unroll]
            for h in 0..heads {
                out[((b * t_total + t) * heads + h) * latent + c] =
                    F::cast_from(acc[h * cols + j] / l_sh[h]);
            }
        }
    }
}
