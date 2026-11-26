//! The memory the generator and the token decoder cross-attend to, one per block, and the
//! attention over it.
//!
//! A memory holds, for each of `nb` blocks:
//!
//! * **token states** shared by the blocks: local-encoder states of a token range (the whole
//!   training sequence, or the recent ring at decode time), of which each block sees only its own
//!   window, the last `recent` tokens before its start (the causality rule);
//! * **per-block entries**: the backbone slots, the retrieved tokens (embedded ids plus their
//!   position within their block) and, for the generator only, the block's known prefix (encoder
//!   states plus a segment embedding); hidden entries are masked.
//!
//! Queries attend over both parts with one softmax. Keys carry RoPE at absolute token positions:
//! token states at their own, slots and retrieved tokens at the block's start, known-prefix states
//! at theirs; queries sit at the block's start (generator) or at their token (token decoder).
//! The copy head reads the same token entries, each standing for the token that followed it.

use burn::module::Module;
use burn::tensor::activation::softmax;
use burn::tensor::{Bool, Device, Int, Tensor, TensorData};

use crate::layers::{Rope, apply_rope, linear};
use crate::quant::QLinear as Linear;

/// The layer-independent memory of `nb` blocks (see the module docs).
#[derive(Clone, Debug)]
pub struct Memory {
    pub nb: usize,
    /// Token states `[B, N, d]` at absolute positions `seq_start..`, None when there are none.
    pub seq: Option<Tensor<3>>,
    pub seq_start: usize,
    /// Per block, the visible range of `seq` (indices into it) and the range the copy head may
    /// point into (an entry is copyable when the token after it is before the block).
    pub windows: Vec<(usize, usize)>,
    pub copy_windows: Vec<(usize, usize)>,
    /// The token that followed each `seq` entry `[B, N]` (0 where none is known).
    pub seq_next: Option<Tensor<2, Int>>,
    /// Per-block entries `[B, nb, E, d]`, their absolute positions (`nb * E`, block-major) and
    /// which are hidden `[B, nb, E]` (`true` hides).
    pub extra: Tensor<4>,
    pub extra_pos: Vec<usize>,
    pub extra_hidden: Tensor<3, Bool>,
    /// The retrieved-token segment of `extra`: offset, length and the token that followed each
    /// entry `[B, nb, len]`.
    pub copy_extra: Option<(usize, usize, Tensor<3, Int>)>,
}

impl Memory {
    /// Per-block entries.
    pub fn extra_len(&self) -> usize {
        self.extra.dims()[2]
    }

    /// Block `i`'s memory alone (one block, the same token states).
    pub fn block(&self, i: usize) -> Memory {
        let e = self.extra_len();
        Memory {
            nb: 1,
            seq: self.seq.clone(),
            seq_start: self.seq_start,
            windows: vec![self.windows[i]],
            copy_windows: vec![self.copy_windows[i]],
            seq_next: self.seq_next.clone(),
            extra: self.extra.clone().narrow(1, i, 1),
            extra_pos: self.extra_pos[i * e..(i + 1) * e].to_vec(),
            extra_hidden: self.extra_hidden.clone().narrow(1, i, 1),
            copy_extra: self
                .copy_extra
                .as_ref()
                .map(|(o, l, n)| (*o, *l, n.clone().narrow(1, i, 1))),
        }
    }

    /// The `[nb * q, N]` mask of the token states for `q` queries per block over `windows`
    /// (`true` hides).
    fn seq_mask(&self, windows: &[(usize, usize)], q: usize, device: &Device) -> Tensor<2, Bool> {
        let n = self.seq.as_ref().map_or(0, |s| s.dims()[1]);
        let mut hidden = Vec::with_capacity(self.nb * q * n);
        for &(lo, hi) in windows {
            for _ in 0..q {
                hidden.extend((0..n).map(|i| i < lo || i >= hi));
            }
        }
        Tensor::from_data(TensorData::new(hidden, [self.nb * q, n]), device)
    }
}

/// One cross-attention layer's keys and values of a [`Memory`]: token states `[B, KVH, N, D]`
/// and per-block entries `[B, KVH, nb * E, D]`.
pub struct LayerKv {
    seq: Option<(Tensor<4>, Tensor<4>)>,
    extra: (Tensor<4>, Tensor<4>),
}

/// Queries from the generator or the token decoder over a [`Memory`].
#[derive(Module, Debug)]
pub struct MemoryAttention {
    q: Linear,
    kv: Linear,
    o: Linear,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl MemoryAttention {
    pub fn new(d: usize, heads: usize, kv_heads: usize, head_dim: usize, device: &Device) -> Self {
        assert!(heads.is_multiple_of(kv_heads), "query heads come in groups per key head");
        Self {
            q: linear(d, heads * head_dim, device),
            kv: linear(d, 2 * kv_heads * head_dim, device),
            o: linear(heads * head_dim, d, device),
            heads,
            kv_heads,
            head_dim,
        }
    }

    /// Keys (with RoPE at `rope_base`) and values of `memory` for this layer.
    pub fn project(&self, memory: &Memory, rope_base: f64) -> LayerKv {
        let device = memory.extra.device();
        let hd = self.head_dim;
        let split = |x: Tensor<3>, rope: Rope| {
            let [b, t, _] = x.dims();
            let kv = self.kv.forward(x);
            let k = kv.clone().narrow(2, 0, self.kv_heads * hd);
            let v = kv.narrow(2, self.kv_heads * hd, self.kv_heads * hd);
            let k = apply_rope(k.reshape([b, t, self.kv_heads, hd]).swap_dims(1, 2), &rope);
            (k, v.reshape([b, t, self.kv_heads, hd]).swap_dims(1, 2))
        };
        let seq = memory.seq.as_ref().map(|s| {
            let n = s.dims()[1];
            split(s.clone(), Rope::new(memory.seq_start, n, hd, rope_base, 1.0, &device))
        });
        let [b, nb, e, d] = memory.extra.dims();
        let extra = split(
            memory.extra.clone().reshape([b, nb * e, d]),
            Rope::at(&memory.extra_pos, hd, rope_base, &device),
        );
        LayerKv { seq, extra }
    }

    /// `x` `[B, nb * q, d]` (q queries per block, block-major) at absolute positions `q_pos`
    /// attending over the layer's memory keys `kv`: `[B, nb * q, d]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
        q_pos: &[usize],
        rope_base: f64,
        kv: &LayerKv,
        memory: &Memory,
    ) -> Tensor<3> {
        let [b, nq, _] = x.dims();
        let (h, hd, nb) = (self.heads, self.head_dim, memory.nb);
        let q_per = nq / nb;
        let device = x.device();
        let groups = h / self.kv_heads;
        let expand = |t: &Tensor<4>| {
            let [b, kvh, n, d] = t.dims();
            t.clone().unsqueeze_dim::<5>(2).expand([b, kvh, groups, n, d]).reshape([b, h, n, d])
        };
        let q = self.q.forward(x).reshape([b, nq, h, hd]).swap_dims(1, 2);
        let q = apply_rope(q, &Rope::at(q_pos, hd, rope_base, &device));
        let scale = 1.0 / (hd as f64).sqrt();
        let e = memory.extra_len();
        let (ek, ev) = (expand(&kv.extra.0), expand(&kv.extra.1));
        let s_extra = q
            .clone()
            .reshape([b, h, nb, q_per, hd])
            .matmul(ek.reshape([b, h, nb, e, hd]).swap_dims(3, 4))
            * scale;
        let hidden =
            memory.extra_hidden.clone().reshape([b, 1, nb, 1, e]).expand([b, h, nb, q_per, e]);
        let s_extra = s_extra.mask_fill(hidden, f32::NEG_INFINITY).reshape([b, h, nq, e]);
        let out = match &kv.seq {
            None => {
                let p = softmax(s_extra, 3).reshape([b, h, nb, q_per, e]);
                p.matmul(ev.reshape([b, h, nb, e, hd])).reshape([b, h, nq, hd])
            }
            Some((sk, sv)) => {
                let (sk, sv) = (expand(sk), expand(sv));
                let n = sk.dims()[2];
                let mask = memory.seq_mask(&memory.windows, q_per, &device);
                let mask = mask.reshape([1, 1, nq, n]).expand([b, h, nq, n]);
                let s_seq =
                    (q.matmul(sk.swap_dims(2, 3)) * scale).mask_fill(mask, f32::NEG_INFINITY);
                let p = softmax(Tensor::cat(vec![s_seq, s_extra], 3), 3);
                let p_seq = p.clone().narrow(3, 0, n);
                let p_extra = p.narrow(3, n, e).reshape([b, h, nb, q_per, e]);
                p_seq.matmul(sv)
                    + p_extra.matmul(ev.reshape([b, h, nb, e, hd])).reshape([b, h, nq, hd])
            }
        };
        self.o.forward(out.swap_dims(1, 2).reshape([b, nq, h * hd]))
    }
}

/// The token decoder's copy (pointer) head: a distribution over the memory's token entries,
/// each standing for the token that followed it, mixed with the vocabulary distribution by a
/// learned gate.
#[derive(Module, Debug)]
pub struct CopyHead {
    q: Linear,
    k: Linear,
    gate: Linear,
    dim: usize,
}

impl CopyHead {
    pub fn new(d: usize, dim: usize, device: &Device) -> Self {
        Self {
            q: linear(d, dim, device),
            k: linear(d, dim, device),
            gate: linear(d, 1, device),
            dim,
        }
    }

    /// Log-probabilities `[B, nb * q, V]` of decoder states `h` `[B, nb * q, d]` with vocabulary
    /// logits `logits` `[B, nb * q, V]`: `(1 - g) p_vocab + g p_copy`, where the copy softmax
    /// also has a null entry whose mass falls back to `p_vocab` (a block with nothing to copy).
    pub fn mix(&self, h: Tensor<3>, logits: Tensor<3>, memory: &Memory) -> Tensor<3> {
        let [b, nq, v] = logits.dims();
        let nb = memory.nb;
        let q_per = nq / nb;
        let device = h.device();
        let gate = burn::tensor::activation::sigmoid(self.gate.forward(h.clone()));
        let cq = self.q.forward(h);
        let scale = 1.0 / (self.dim as f64).sqrt();
        let mut scores = Vec::new();
        let mut ids = Vec::new();
        if let (Some(seq), Some(next)) = (&memory.seq, &memory.seq_next) {
            let n = seq.dims()[1];
            let ck = self.k.forward(seq.clone());
            let mask = memory.seq_mask(&memory.copy_windows, q_per, &device);
            let s = (cq.clone().matmul(ck.swap_dims(1, 2)) * scale)
                .mask_fill(mask.reshape([1, nq, n]).expand([b, nq, n]), f32::NEG_INFINITY);
            scores.push(s);
            ids.push(next.clone().reshape([b, 1, n]).expand([b, nq, n]));
        }
        if let Some((off, len, next)) = &memory.copy_extra {
            let (off, len) = (*off, *len);
            let entries = memory.extra.clone().narrow(2, off, len);
            let ce = self.k.forward(entries.reshape([b, nb * len, memory.extra.dims()[3]]));
            let ce = ce.reshape([b, nb, len, self.dim]);
            let s = cq.clone().reshape([b, nb, q_per, self.dim]).matmul(ce.swap_dims(2, 3)) * scale;
            let hidden = memory.extra_hidden.clone().narrow(2, off, len);
            let hidden = hidden.reshape([b, nb, 1, len]).expand([b, nb, q_per, len]);
            scores.push(s.mask_fill(hidden, f32::NEG_INFINITY).reshape([b, nq, len]));
            ids.push(
                next.clone()
                    .reshape([b, nb, 1, len])
                    .expand([b, nb, q_per, len])
                    .reshape([b, nq, len]),
            );
        }
        let p_vocab = softmax(logits, 2);
        let mixed = if scores.is_empty() {
            p_vocab
        } else {
            scores.push(Tensor::zeros([b, nq, 1], &device));
            let a = softmax(Tensor::cat(scores, 2), 2);
            let entries = a.dims()[2] - 1;
            let null = a.clone().narrow(2, entries, 1);
            let copy = Tensor::<3>::zeros([b, nq, v], &device).scatter(
                2,
                Tensor::cat(ids, 2),
                a.narrow(2, 0, entries),
                burn::tensor::IndexingUpdateOp::Add,
            );
            let keep = gate.clone().neg() + 1.0 + gate.clone() * null;
            p_vocab * keep + copy * gate
        };
        mixed.clamp_min(1e-30).log()
    }
}
