//! Sparse retrieval attention: exact recall from anywhere in the context at a bounded cost.
//!
//! * **MLA cache** (DeepSeek-V2): each position stores one compressed latent (`ret_latent`
//!   values) plus one decoupled RoPE key (`ret_rope_dim`, shared by all heads) instead of
//!   per-head keys and values. Keys and values are up-projections of the latent; with a cache
//!   the projections are absorbed into the query and the output, so attention runs directly
//!   on the latent.
//! * **Indexer** (DeepSeek-V3.2 lightning indexer): a few small heads score every cached
//!   position, `I(t, s) = sum_j w_j(t) relu(q_j(t) . k(s))`. A block's score is the maximum over
//!   its positions.
//! * **Selection**: each query attends to the `ret_topk` highest-scoring blocks of `ret_block`
//!   positions that start before its local window, plus its most recent `ret_local` positions.
//!   With `ret_dense` (the training warm-up) it attends to the whole causal context.
//! * A sigmoid output gate (Qwen3-Next), RMSNorm on the queries, the latent and the RoPE key
//!   (QK-norm), and RoPE on the decoupled 64-dimensional part only (partial RoPE).
//!
//! Reference version in plain Burn tensor ops. One-token decode gathers the selected blocks
//! from the cache; prefill and training compute the selection as a mask over the whole
//! context, which is exact but costs as much as dense attention.

use burn::module::Module;
use burn::nn::{Linear as DenseLinear, RmsNorm};
use burn::tensor::activation::{relu, sigmoid, softmax};
use burn::tensor::{Bool, Device, Int, IntDType, Tensor, s};

use crate::layers::{Rope, apply_rope, dense_linear, linear, rms_norm};
use crate::quant::QLinear as Linear;

/// Shape of a sparse retrieval layer.
#[derive(Debug, Clone)]
pub struct RetrievalSpec {
    pub d: usize,
    pub heads: usize,
    pub nope_dim: usize,
    pub rope_dim: usize,
    pub v_dim: usize,
    pub latent: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub block: usize,
    pub topk: usize,
    pub local: usize,
    pub dense: bool,
    /// Indexer keys cached as 8-bit codes and one scale per position.
    pub index_8bit: bool,
    pub eps: f64,
}

impl RetrievalSpec {
    /// Bytes of one layer's cache for `max_positions`: the latent and RoPE key at `itemsize`,
    /// the indexer key at `index_itemsize` per value; keys narrower than `itemsize` (8-bit) add
    /// one scale per position at `itemsize`.
    pub fn cache_bytes(
        &self,
        max_positions: usize,
        itemsize: usize,
        index_itemsize: usize,
        batch: usize,
    ) -> usize {
        let scale = if index_itemsize < itemsize { itemsize } else { 0 };
        batch
            * max_positions
            * ((self.latent + self.rope_dim) * itemsize + self.index_dim * index_itemsize + scale)
    }
}

/// Largest 8-bit code of an indexer key (symmetric, so -127..=127).
const KEY_CODE_MAX: f64 = 127.0;

/// Indexer keys `[B, T, di]` as 8-bit codes (i8) and one scale per position `[B, T, 1]` at the
/// run dtype: `scale = max |k| / 127`, `code = round(k / scale)`.
pub fn quantize_keys(keys: Tensor<3>) -> (Tensor<3, Int>, Tensor<3>) {
    let scale = keys.clone().abs().max_dim(2).div_scalar(KEY_CODE_MAX).clamp_min(1e-30);
    let codes = (keys / scale.clone()).round().clamp(-KEY_CODE_MAX, KEY_CODE_MAX);
    (codes.int().cast(IntDType::I8), scale)
}

/// The keys 8-bit `codes` and their `scales` stand for.
pub fn dequantize_keys(codes: Tensor<3, Int>, scales: Tensor<3>) -> Tensor<3> {
    codes.float() * scales
}

/// Indexer keys as the scores read them: `keys` `[B, S, di]`, or for 8-bit keys their codes (as
/// floats, exact) with the per-position `scales` `[B, S, 1]` applied after the dot products
/// (`relu(s q.c) = s relu(q.c)` for a positive scale), so the codes are never rounded back to
/// the run dtype.
struct ScoreKeys {
    keys: Tensor<3>,
    scales: Option<Tensor<3>>,
}

impl ScoreKeys {
    /// Keys computed in this pass, rounded to 8 bits as a cache would store them when `int8`.
    /// Under autodiff (training) the rounding is straight-through instead.
    fn fresh(keys: Tensor<3>, int8: bool) -> Self {
        if !int8 {
            return Self { keys, scales: None };
        }
        let (codes, scales) = quantize_keys(keys.clone());
        if keys.is_require_grad() {
            let rounded = dequantize_keys(codes, scales);
            return Self { keys: keys.clone() + (rounded - keys).detach(), scales: None };
        }
        Self { keys: codes.float(), scales: Some(scales) }
    }

    fn positions(&self) -> usize {
        self.keys.dims()[1]
    }

    /// The first `n` positions.
    fn narrow(self, n: usize) -> Self {
        Self { keys: self.keys.narrow(1, 0, n), scales: self.scales.map(|s| s.narrow(1, 0, n)) }
    }
}

/// A retrieval cache's indexer keys.
// One per retrieval layer: the size gap between the variants is not worth boxing for.
#[allow(clippy::large_enum_variant)]
pub enum IndexKeys {
    /// At the run dtype, `[B, S, di]`.
    Float(Tensor<3>),
    /// 8-bit codes `[B, S, di]` (i8) and one scale per position `[B, S, 1]` (run dtype).
    Int8 { codes: Tensor<3, Int>, scales: Tensor<3> },
}

impl IndexKeys {
    fn new(batch: usize, capacity: usize, index_dim: usize, int8: bool, device: &Device) -> Self {
        if int8 {
            let codes = Tensor::<3, Int>::zeros([batch, capacity, index_dim], device);
            let scales = Tensor::<3>::zeros([batch, capacity, 1], device);
            Self::Int8 { codes: codes.cast(IntDType::I8), scales }
        } else {
            Self::Float(Tensor::zeros([batch, capacity, index_dim], device))
        }
    }

    fn nbytes(&self) -> usize {
        let bytes = |dims: [usize; 3], size: usize| dims.iter().product::<usize>() * size;
        match self {
            Self::Float(k) => bytes(k.dims(), k.dtype().size()),
            Self::Int8 { codes, scales } => {
                bytes(codes.dims(), codes.dtype().size())
                    + bytes(scales.dims(), scales.dtype().size())
            }
        }
    }

    /// Writes `keys` `[B, T, di]` (run dtype) at positions `start..start + T`.
    fn write(self, start: usize, keys: Tensor<3>) -> Self {
        let t = keys.dims()[1];
        match self {
            Self::Float(k) => Self::Float(k.slice_assign(s![.., start..start + t], keys)),
            Self::Int8 { codes, scales } => {
                let (new_codes, new_scales) = quantize_keys(keys);
                Self::Int8 {
                    codes: codes.slice_assign(s![.., start..start + t], new_codes),
                    scales: scales.slice_assign(s![.., start..start + t], new_scales),
                }
            }
        }
    }

    /// [`Self::write`] into the existing buffers (CUDA).
    #[cfg(feature = "cuda")]
    fn write_into(&self, start: usize, keys: Tensor<3>) {
        match self {
            Self::Float(k) => crate::inplace::write_float(k, keys, 1, start),
            Self::Int8 { codes, scales } => {
                let (new_codes, new_scales) = quantize_keys(keys);
                crate::inplace::write_int(codes, new_codes, 1, start);
                crate::inplace::write_float(scales, new_scales, 1, start);
            }
        }
    }

    /// The first `n` positions' keys for scoring.
    fn view(&self, n: usize) -> ScoreKeys {
        match self {
            Self::Float(k) => ScoreKeys { keys: k.clone().narrow(1, 0, n), scales: None },
            Self::Int8 { codes, scales } => ScoreKeys {
                keys: codes.clone().narrow(1, 0, n).float(),
                scales: Some(scales.clone().narrow(1, 0, n)),
            },
        }
    }
}

/// One retrieval layer's cache: `[latent | RoPE key]` and the indexer key per position. It
/// grows into a preallocated capacity; nothing is ever evicted.
pub struct LatentCache {
    kv: Option<Tensor<3>>,
    keys: Option<IndexKeys>,
    /// The next position on the device (`[1]`, i32), read by the fused decode kernels.
    #[cfg(feature = "cuda")]
    position: Tensor<1, Int>,
    capacity: usize,
    length: usize,
    itemsize: usize,
}

impl LatentCache {
    fn new(
        batch: usize,
        width: usize,
        index_dim: usize,
        int8: bool,
        capacity: usize,
        device: &Device,
    ) -> Self {
        let kv = Tensor::<3>::zeros([batch, capacity, width], device);
        let itemsize = kv.dtype().size();
        let keys = IndexKeys::new(batch, capacity, index_dim, int8, device);
        Self {
            kv: Some(kv),
            keys: Some(keys),
            #[cfg(feature = "cuda")]
            position: Tensor::<1, Int>::zeros([1], device).cast(burn::tensor::IntDType::I32),
            capacity,
            length: 0,
            itemsize,
        }
    }

    /// Next absolute position.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Copies the next position into the device buffer the fused kernels read.
    #[cfg(feature = "cuda")]
    pub(crate) fn refresh_position(&self) {
        let device = self.position.device();
        let pos = burn::tensor::TensorData::new(vec![self.length as i32], [1]);
        let pos = Tensor::<1, Int>::from_data(pos, &device).cast(burn::tensor::IntDType::I32);
        crate::kernels::assign_int_into(&self.position, pos, 0, 0);
    }

    /// The device buffer of the next position (`[1]`, i32) the fused kernels read; a captured
    /// graph copies staged positions into it ([`crate::graph::Staging`]).
    #[cfg(feature = "cuda")]
    pub(crate) fn position(&self) -> &Tensor<1, Int> {
        &self.position
    }

    /// Advances the position by `t` without writing (a replayed graph wrote the entries).
    #[cfg(feature = "cuda")]
    pub(crate) fn advance(&mut self, t: usize) {
        self.length += t;
    }

    /// Moves the position back to `len` (entries past it are overwritten when written again).
    pub fn rewind(&mut self, len: usize) {
        self.length = len;
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn reset(&mut self) {
        self.length = 0;
    }

    /// Bytes allocated: the latent and RoPE key at the run dtype, the indexer keys at the run
    /// dtype or as 8-bit codes with one scale per position.
    pub fn nbytes(&self) -> usize {
        let kv = self.kv.as_ref().expect("cache").dims().iter().product::<usize>();
        kv * self.itemsize + self.keys.as_ref().expect("cache").nbytes()
    }

    fn write(&mut self, kv: Tensor<3>, keys: Tensor<3>) {
        let (start, t) = (self.length, kv.dims()[1]);
        assert!(start + t <= self.capacity, "cache capacity {} exceeded", self.capacity);
        #[cfg(feature = "cuda")]
        if crate::fused::active(&kv.device()) {
            // In the cache's own buffers, which captured decode graphs keep reading.
            crate::inplace::write_float(self.kv.as_ref().expect("cache"), kv, 1, start);
            self.keys.as_ref().expect("cache").write_into(start, keys);
            self.length += t;
            return;
        }
        let old_kv = self.kv.take().expect("cache");
        self.kv = Some(old_kv.slice_assign(s![.., start..start + t], kv));
        self.keys = Some(self.keys.take().expect("cache").write(start, keys));
        self.length += t;
    }

    /// The cached latents and the indexer keys of every position so far.
    fn view(&self) -> (Tensor<3>, ScoreKeys) {
        let n = self.length;
        let kv = self.kv.clone().expect("cache");
        let kv = if n == self.capacity { kv } else { kv.narrow(1, 0, n) };
        (kv, self.keys.as_ref().expect("cache").view(n))
    }
}

/// Number of blocks a query at `pos` may retrieve: those starting before its local window.
fn candidates(pos: usize, block: usize, local: usize) -> usize {
    if pos >= local { (pos - local) / block + 1 } else { 0 }
}

#[derive(Module, Debug)]
pub struct SparseRetrieval {
    /// `[q | latent, RoPE key | gate | indexer q | indexer k | indexer weights]` from `x`.
    proj: Linear,
    q_norm: RmsNorm,
    kv_norm: RmsNorm,
    rope_norm: RmsNorm,
    index_norm: RmsNorm,
    /// Latent to per-head keys (no position) and values.
    uk: DenseLinear,
    uv: DenseLinear,
    o: Linear,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    latent: usize,
    index_heads: usize,
    index_dim: usize,
    block: usize,
    topk: usize,
    local: usize,
    dense: bool,
    index_8bit: bool,
}

impl SparseRetrieval {
    pub fn new(spec: &RetrievalSpec, device: &Device) -> Self {
        assert!(spec.local >= spec.block, "ret_local must cover at least one block");
        assert!(spec.block >= 1 && spec.topk >= 1, "ret_block and ret_topk must be positive");
        let (h, qd) = (spec.heads, spec.nope_dim + spec.rope_dim);
        let width = h * qd
            + spec.latent
            + spec.rope_dim
            + h * spec.v_dim
            + spec.index_heads * spec.index_dim
            + spec.index_dim
            + spec.index_heads;
        Self {
            proj: linear(spec.d, width, device),
            q_norm: rms_norm(qd, spec.eps, device),
            kv_norm: rms_norm(spec.latent, spec.eps, device),
            rope_norm: rms_norm(spec.rope_dim, spec.eps, device),
            index_norm: rms_norm(spec.index_dim, spec.eps, device),
            uk: dense_linear(spec.latent, h * spec.nope_dim, device),
            uv: dense_linear(spec.latent, h * spec.v_dim, device),
            o: linear(h * spec.v_dim, spec.d, device),
            heads: h,
            nope_dim: spec.nope_dim,
            rope_dim: spec.rope_dim,
            v_dim: spec.v_dim,
            latent: spec.latent,
            index_heads: spec.index_heads,
            index_dim: spec.index_dim,
            block: spec.block,
            topk: spec.topk,
            local: spec.local,
            dense: spec.dense,
            index_8bit: spec.index_8bit,
        }
    }

    pub fn make_cache(&self, max_positions: usize, batch: usize, device: &Device) -> LatentCache {
        let width = self.latent + self.rope_dim;
        let int8 = self.index_8bit;
        LatentCache::new(batch, width, self.index_dim, int8, max_positions, device)
    }

    fn scale(&self) -> f64 {
        1.0 / ((self.nope_dim + self.rope_dim) as f64).sqrt()
    }

    /// The layer's shape and selection settings for the fused kernels.
    #[cfg(feature = "cuda")]
    fn kernel_dims(&self) -> crate::kernels::RetrievalDims {
        crate::kernels::RetrievalDims {
            heads: self.heads,
            nope_dim: self.nope_dim,
            rope_dim: self.rope_dim,
            v_dim: self.v_dim,
            latent: self.latent,
            index_heads: self.index_heads,
            index_dim: self.index_dim,
            block: self.block,
            topk: self.topk,
            local: self.local,
            dense: self.dense,
            index_8bit: self.index_8bit,
            eps: self.q_norm.epsilon,
        }
    }

    /// The fused one-token decode ([`crate::kernels::retrieval_decode`]): the same cache write,
    /// selection and attention as the reference, between the two projections.
    #[cfg(feature = "cuda")]
    fn decode_fused(&self, x: Tensor<3>, rope: &Rope, cache: &mut LatentCache) -> Tensor<3> {
        let pos = cache.length;
        assert!(pos < cache.capacity, "cache capacity {} exceeded", cache.capacity);
        let dims = self.kernel_dims();
        let norms =
            [&self.q_norm, &self.kv_norm, &self.rope_norm, &self.index_norm].map(|n| n.gamma.val());
        if !crate::fused::capturing() {
            cache.refresh_position();
        }
        let out = crate::kernels::retrieval_decode(
            self.proj.forward(x),
            norms,
            rope.tables(),
            self.uk.weight.val(),
            self.uv.weight.val(),
            cache.kv.as_ref().expect("cache"),
            cache.keys.as_ref().expect("cache"),
            &cache.position,
            dims,
        );
        cache.length += 1;
        self.o.forward(out)
    }

    /// `x` `[B, T, d]` at absolute positions `start..start + T`; `rope` holds the tables of those
    /// positions for the decoupled key width. With a cache, the new positions are appended and
    /// the queries see everything cached before them. On CUDA with the fused kernels on,
    /// one-token steps with a cache take [`Self::decode_fused`].
    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    pub fn forward(
        &self,
        x: Tensor<3>,
        start: usize,
        rope: &Rope,
        mut cache: Option<&mut LatentCache>,
    ) -> Tensor<3> {
        #[cfg(feature = "cuda")]
        if let Some(c) = cache.as_deref_mut()
            && x.dims()[1] == 1
            && crate::fused::active(&x.device())
        {
            return self.decode_fused(x, rope, c);
        }
        let [b, t, _] = x.dims();
        let (h, dn, dr, dv, l) =
            (self.heads, self.nope_dim, self.rope_dim, self.v_dim, self.latent);
        let (hi, di) = (self.index_heads, self.index_dim);
        let p = self.proj.forward(x);
        let mut offset = 0;
        let mut take = |n: usize| {
            let part = p.clone().narrow(2, offset, n);
            offset += n;
            part
        };
        let q = take(h * (dn + dr));
        let ckv = take(l + dr);
        let gate = take(h * dv);
        let index_q = take(hi * di);
        let index_k = take(di);
        let index_w = take(hi);

        let q = self.q_norm.forward(q.reshape([b, t, h, dn + dr])).swap_dims(1, 2);
        let q_nope = q.clone().narrow(3, 0, dn);
        let q_rope = apply_rope(q.narrow(3, dn, dr), rope);
        let latent = self.kv_norm.forward(ckv.clone().narrow(2, 0, l));
        let rope_key = self.rope_norm.forward(ckv.narrow(2, l, dr)).unsqueeze_dim::<4>(1);
        let rope_key = apply_rope(rope_key, rope).reshape([b, t, dr]);
        let entry = Tensor::cat(vec![latent, rope_key], 2); // [B, T, l + dr]
        let index_k = self.index_norm.forward(index_k);
        let index = Indexer { q: index_q, w: index_w };

        let out = match cache {
            None => {
                // The keys rounded as an 8-bit cache would store them, so a forward pass and
                // cached decoding select alike.
                let keys = ScoreKeys::fresh(index_k, self.index_8bit);
                let hidden = self.hidden_mask(&index, keys, start, t);
                self.attend_expanded(q_nope, q_rope, entry, hidden)
            }
            Some(cache) => {
                cache.write(entry, index_k);
                #[cfg(feature = "cuda")]
                if t > 1 && !self.dense && crate::fused::active(&q_nope.device()) {
                    let out = self.attend_prefill(q_nope, q_rope, &index, cache, start);
                    return self.output(out, gate);
                }
                let (kv, keys) = cache.view();
                if t == 1 && !self.dense {
                    self.attend_selected(q_nope, q_rope, kv, &index, keys, start)
                } else {
                    let hidden = self.hidden_mask(&index, keys, start, t);
                    self.attend_latent(q_nope, q_rope, kv, Some(hidden))
                }
            }
        };
        self.output(out, gate)
    }

    /// The attention output `[B, H, T, dv]` gated by `gate` `[B, T, H dv]` and projected.
    fn output(&self, out: Tensor<4>, gate: Tensor<3>) -> Tensor<3> {
        let [b, h, t, dv] = out.dims();
        self.o.forward(out.swap_dims(1, 2).reshape([b, t, h * dv]) * sigmoid(gate))
    }

    /// Sparse prefill of `T > 1` queries at positions `start..` whose entries are already in
    /// the cache, on the device ([`crate::kernels::retrieval_select`] and
    /// [`crate::kernels::retrieval_prefill_attend`]): the same selection and attention as
    /// [`Self::hidden_mask`] with [`Self::attend_latent`], but each query reads only its picked
    /// blocks and its window, and no `[T, S]` mask or score matrix of the attention is built.
    /// The indexer dot products and the absorbed projections stay matrix products. Returns
    /// `[B, H, T, dv]`.
    #[cfg(feature = "cuda")]
    fn attend_prefill(
        &self,
        q_nope: Tensor<4>,
        q_rope: Tensor<4>,
        index: &Indexer,
        cache: &LatentCache,
        start: usize,
    ) -> Tensor<4> {
        /// Most indexer dot products held at once (`[B, hi, queries, positions]`).
        const SELECT_BUDGET: usize = 1 << 26;
        let [b, h, t, _] = q_nope.dims();
        let (dn, dv, l) = (self.nope_dim, self.v_dim, self.latent);
        let (hi, di, block) = (self.index_heads, self.index_dim, self.block);
        let dims = self.kernel_dims();
        let wuk = self.uk.weight.val().reshape([l, h, dn]).permute([1, 2, 0]).unsqueeze_dim::<4>(0);
        let q = Tensor::cat(vec![q_nope.matmul(wuk), q_rope], 3).mul_scalar(self.scale());
        let kv = cache.kv.as_ref().expect("cache");
        // Packed scratch the selection fills for every query (its count, then that many blocks).
        let picked = crate::kernels::scratch_i32(kv, [b, t, self.topk]);
        let count = crate::kernels::scratch_i32(kv, [b, t]);
        // Candidate blocks of the last query; earlier queries have a prefix of them.
        let n = candidates(start + t - 1, block, self.local);
        if n == 0 {
            // No query retrieves yet: the selection only records zero blocks for each.
            let none = q.clone();
            crate::kernels::retrieval_select(
                none,
                index.w.clone(),
                kv.clone(),
                &picked,
                &count,
                start,
                0,
                dims,
            );
        } else {
            let keys = cache.keys.as_ref().expect("cache").view(n * block);
            let iq = index.q.clone().reshape([b, t, hi, di]).swap_dims(1, 2); // [B, hi, T, di]
            let kt = keys.keys.clone().swap_dims(1, 2).unsqueeze_dim::<4>(1); // [B, 1, di, S']
            let scales = keys.scales.unwrap_or(keys.keys);
            let group = (SELECT_BUDGET / (b * hi * n * block)).clamp(1, t);
            let mut t0 = 0;
            while t0 < t {
                let len = group.min(t - t0);
                let raw = iq.clone().narrow(2, t0, len).matmul(kt.clone());
                let w = index.w.clone();
                crate::kernels::retrieval_select(
                    raw,
                    w,
                    scales.clone(),
                    &picked,
                    &count,
                    start,
                    t0,
                    dims,
                );
                t0 += len;
            }
        }
        let latent = crate::kernels::retrieval_prefill_attend(q, kv, picked, count, start, dims);
        let wuv = self.uv.weight.val().reshape([l, h, dv]).swap_dims(0, 1).unsqueeze_dim::<4>(0);
        latent.swap_dims(1, 2).matmul(wuv)
    }

    /// Indexer scores `[B, T, S]` of queries against cached indexer keys.
    fn index_scores(&self, index: &Indexer, keys: ScoreKeys) -> Tensor<3> {
        let [b, t, _] = index.q.dims();
        let s = keys.positions();
        let ScoreKeys { keys, scales } = keys;
        let (hi, di) = (self.index_heads, self.index_dim);
        let q = index.q.clone().reshape([b, t, hi, di]).swap_dims(1, 2); // [B, hi, T, di]
        let w = index.w.clone().swap_dims(1, 2).unsqueeze_dim::<4>(3) // [B, hi, T, 1]
            * (1.0 / ((hi * di) as f64).sqrt());
        let keys = keys.swap_dims(1, 2).unsqueeze_dim::<4>(1); // [B, 1, di, S]
        // One head at a time keeps the transient at [B, T, S].
        let scores = (0..hi)
            .map(|j| {
                let qj = q.clone().narrow(1, j, 1);
                relu(qj.matmul(keys.clone())) * w.clone().narrow(1, j, 1)
            })
            .reduce(|a, b| a + b)
            .expect("at least one indexer head")
            .reshape([b, t, s]);
        match scales {
            Some(scales) => scores * scales.reshape([b, 1, s]),
            None => scores,
        }
    }

    /// Block scores `[B, T, nb]`: the maximum indexer score inside each block.
    fn block_scores(&self, scores: Tensor<3>) -> Tensor<3> {
        let [b, t, s] = scores.dims();
        let nb = s.div_ceil(self.block);
        let pad = nb * self.block - s;
        let scores = if pad > 0 {
            let fill = Tensor::full([b, t, pad], f32::NEG_INFINITY, &scores.device());
            Tensor::cat(vec![scores, fill], 2)
        } else {
            scores
        };
        scores.reshape([b, t, nb, self.block]).max_dim(3).reshape([b, t, nb])
    }

    /// `[B, T, S]` mask, `true` where query `start + i` must not see position `s`: in the
    /// future, or older than its local window and outside every block it selected.
    fn hidden_mask(
        &self,
        index: &Indexer,
        keys: ScoreKeys,
        start: usize,
        t: usize,
    ) -> Tensor<3, Bool> {
        let [b, s, _] = keys.keys.dims();
        let device = keys.keys.device();
        let (block, local) = (self.block, self.local);
        // Query minus key position, `[T, S]`, built on the device.
        let queries = Tensor::<1, Int>::arange(start as i64..(start + t) as i64, &device);
        let queries = queries.reshape([t, 1]);
        let age = queries.clone() - Tensor::<1, Int>::arange(0..s as i64, &device).reshape([1, s]);
        let causal = age.clone().lower_elem(0).unsqueeze_dim::<3>(0).expand([b, t, s]);
        if self.dense || candidates(start + t - 1, block, local) == 0 {
            return causal;
        }
        let nb = s.div_ceil(block);
        let scores = self.block_scores(self.index_scores(index, keys));
        // A block is a candidate when it starts at least `local` positions before the query.
        let block_starts = Tensor::<1, Int>::arange(0..nb as i64, &device).mul_scalar(block as i64);
        let candidate = (queries - block_starts.reshape([1, nb]))
            .greater_equal_elem(local as i64)
            .unsqueeze_dim::<3>(0)
            .expand([b, t, nb]);
        let scores = scores.mask_fill(candidate.clone().bool_not(), f32::NEG_INFINITY);
        let k = self.topk.min(nb);
        let (_, picked) = scores.topk_with_indices(k, 2); // [B, T, k]
        let blocks = Tensor::<1, Int>::arange(0..nb as i64, &device)
            .reshape([1, 1, 1, nb])
            .expand([b, t, k, nb]);
        let selected = picked
            .unsqueeze_dim::<4>(3)
            .expand([b, t, k, nb])
            .equal(blocks)
            .any_dim(2)
            .reshape([b, t, nb])
            .bool_and(candidate);
        let selected = selected
            .unsqueeze_dim::<4>(3)
            .expand([b, t, nb, block])
            .reshape([b, t, nb * block])
            .narrow(2, 0, s);
        let old = age.greater_equal_elem(local as i64).unsqueeze_dim::<3>(0).expand([b, t, s]);
        causal.bool_or(old.bool_and(selected.bool_not()))
    }

    /// Attention with keys and values expanded per head from the latent (training form).
    fn attend_expanded(
        &self,
        q_nope: Tensor<4>,
        q_rope: Tensor<4>,
        entry: Tensor<3>,
        hidden: Tensor<3, Bool>,
    ) -> Tensor<4> {
        let [b, h, t, _] = q_nope.dims();
        let s = entry.dims()[1];
        let (dn, dr, dv, l) = (self.nope_dim, self.rope_dim, self.v_dim, self.latent);
        let latent = entry.clone().narrow(2, 0, l);
        let rope_key = entry.narrow(2, l, dr).unsqueeze_dim::<4>(1).expand([b, h, s, dr]);
        let k_nope = self.uk.forward(latent.clone()).reshape([b, s, h, dn]).swap_dims(1, 2);
        let v = self.uv.forward(latent).reshape([b, s, h, dv]).swap_dims(1, 2);
        let q = Tensor::cat(vec![q_nope, q_rope], 3);
        let k = Tensor::cat(vec![k_nope, rope_key], 3);
        let scores = q.matmul(k.swap_dims(2, 3)) * self.scale();
        let hidden = hidden.unsqueeze_dim::<4>(1).expand([b, h, t, s]);
        softmax(scores.mask_fill(hidden, f32::NEG_INFINITY), 3).matmul(v)
    }

    /// Attention directly on cached latents `[B, S, l + dr]`, with the key up-projection absorbed
    /// into the queries and the value up-projection applied after the weighted sum.
    fn attend_latent(
        &self,
        q_nope: Tensor<4>,
        q_rope: Tensor<4>,
        kv: Tensor<3>,
        hidden: Option<Tensor<3, Bool>>,
    ) -> Tensor<4> {
        let [b, h, t, _] = q_nope.dims();
        let s = kv.dims()[1];
        let (dn, dv, l) = (self.nope_dim, self.v_dim, self.latent);
        // uk weight is [l, h * dn]: per head, q_lat = q_nope W_h^T.
        let wuk = self.uk.weight.val().reshape([l, h, dn]).permute([1, 2, 0]).unsqueeze_dim::<4>(0);
        let wuv = self.uv.weight.val().reshape([l, h, dv]).swap_dims(0, 1).unsqueeze_dim::<4>(0);
        let q = Tensor::cat(vec![q_nope.matmul(wuk), q_rope], 3); // [B, h, T, l + dr]
        let kv = kv.unsqueeze_dim::<4>(1); // [B, 1, S, l + dr]
        let mut scores = q.matmul(kv.clone().swap_dims(2, 3)) * self.scale();
        if let Some(hidden) = hidden {
            let hidden = hidden.unsqueeze_dim::<4>(1).expand([b, h, t, s]);
            scores = scores.mask_fill(hidden, f32::NEG_INFINITY);
        }
        softmax(scores, 3).matmul(kv.narrow(3, 0, l)).matmul(wuv)
    }

    /// One-token decode at position `pos`: gather the selected blocks and the local window from
    /// the cache instead of attending over all of it.
    fn attend_selected(
        &self,
        q_nope: Tensor<4>,
        q_rope: Tensor<4>,
        kv: Tensor<3>,
        index: &Indexer,
        keys: ScoreKeys,
        pos: usize,
    ) -> Tensor<4> {
        let (block, local) = (self.block, self.local);
        let n = candidates(pos, block, local);
        if n <= self.topk {
            // Every candidate block is selected: the whole causal context is visible.
            return self.attend_latent(q_nope, q_rope, kv, None);
        }
        let [b, _, width] = kv.dims();
        let device = kv.device();
        let k = self.topk;
        let scores = self.index_scores(index, keys.narrow(n * block)); // [B, 1, n*block]
        let scores = self.block_scores(scores).reshape([b, n]);
        let (_, picked) = scores.topk_with_indices(k, 1); // [B, k]
        let offsets = Tensor::<1, Int>::arange(0..block as i64, &device).reshape([1, 1, block]);
        let positions =
            (picked.mul_scalar(block as i64).reshape([b, k, 1]) + offsets).reshape([b, k * block]);
        let gathered = kv
            .clone()
            .gather(1, positions.clone().unsqueeze_dim::<3>(2).expand([b, k * block, width]));
        let recent = kv.narrow(1, pos + 1 - local, local);
        let keys = Tensor::cat(vec![gathered, recent], 1);
        // Selected positions inside the local window are already in it, so they are hidden; the
        // window itself stands in as position 0, which never is.
        let window = Tensor::<2, Int>::zeros([b, local], &device);
        let hidden = Tensor::cat(vec![positions, window], 1)
            .greater_elem((pos - local) as i64)
            .unsqueeze_dim::<3>(1);
        self.attend_latent(q_nope, q_rope, keys, Some(hidden))
    }
}

/// Indexer queries `[B, T, hi * di]` and head weights `[B, T, hi]` of the current positions.
struct Indexer {
    q: Tensor<3>,
    w: Tensor<3>,
}
