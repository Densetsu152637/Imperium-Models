//! Decode caches shared by both models: preallocated KV caches for attention layers, and
//! [`LayerCache`], one layer's cache whatever its mixer (KV, DeltaNet state or MLA latent).
//!
//! Full-attention layers keep `capacity` positions and grow into it. Sliding layers keep a ring
//! buffer of exactly `window` positions. Positions are contiguous: the cache itself tracks the
//! next absolute position (`length`), so callers never pass positions.
//!
//! Masks follow the Burn convention: `true` marks a key the query must not see.

use burn::tensor::{Bool, Device, Int, Tensor, TensorData, s};

use crate::delta::DeltaCache;
use crate::retrieval::LatentCache;

/// Key position stored for an empty ring slot. Far ahead of every query, so the causal test
/// masks it.
const EMPTY_SLOT: i32 = i32::MAX / 2;

/// Boolean `[Tq, Tk]` mask, `true` where the key is not visible: it is missing, in the future,
/// or outside the window.
pub fn build_mask(
    device: &Device,
    qpos: &[i32],
    kpos: &[i32],
    window: Option<usize>,
) -> Tensor<2, Bool> {
    let q = Tensor::<2, Int>::from_data(TensorData::new(qpos.to_vec(), [qpos.len(), 1]), device);
    let k = Tensor::<2, Int>::from_data(TensorData::new(kpos.to_vec(), [1, kpos.len()]), device);
    let diff = q - k;
    let mut masked = diff.clone().lower_elem(0);
    if let Some(window) = window {
        masked = masked.bool_or(diff.greater_equal_elem(window as i64));
    }
    masked
}

pub struct KvCache {
    k: Option<Tensor<4>>,
    v: Option<Tensor<4>>,
    window: Option<usize>,
    capacity: usize,
    length: usize,
    batch: usize,
    kv_heads: usize,
    head_dim: usize,
    itemsize: usize,
    device: Device,
}

impl KvCache {
    /// `window` None keeps `capacity` positions; otherwise a ring of exactly `window` positions.
    pub fn new(
        batch: usize,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        window: Option<usize>,
        device: &Device,
    ) -> Self {
        let capacity = window.unwrap_or(capacity);
        let shape = [batch, kv_heads, capacity, head_dim];
        let k = Tensor::<4>::zeros(shape, device);
        let v = Tensor::<4>::zeros(shape, device);
        let itemsize = k.dtype().size();
        Self {
            k: Some(k),
            v: Some(v),
            window,
            capacity,
            length: 0,
            batch,
            kv_heads,
            head_dim,
            itemsize,
            device: device.clone(),
        }
    }

    /// A ring when the window is smaller than the run, otherwise a plain growing cache.
    pub fn make(
        batch: usize,
        kv_heads: usize,
        head_dim: usize,
        max_positions: usize,
        window: Option<usize>,
        device: &Device,
    ) -> Self {
        match window {
            Some(w) if max_positions > w => {
                Self::new(batch, kv_heads, head_dim, w, Some(w), device)
            }
            _ => Self::new(batch, kv_heads, head_dim, max_positions, None, device),
        }
    }

    /// Next absolute position.
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn nbytes(&self) -> usize {
        self.batch * self.kv_heads * self.capacity * self.head_dim * self.itemsize * 2
    }

    pub fn reset(&mut self) {
        self.length = 0;
    }

    /// The ring's window, or None for a plain growing cache.
    pub fn window(&self) -> Option<usize> {
        self.window
    }

    /// The key and value buffers `[B, KVH, capacity, D]` themselves (graphed steps write them
    /// in place, [`crate::graph`]).
    #[cfg(feature = "cuda")]
    pub(crate) fn buffers(&self) -> (&Tensor<4>, &Tensor<4>) {
        (self.k.as_ref().expect("cache"), self.v.as_ref().expect("cache"))
    }

    /// Moves the position by `t` without writing (a graphed step wrote the entries).
    #[cfg(feature = "cuda")]
    pub(crate) fn advance(&mut self, t: usize) {
        self.length += t;
    }

    /// Moves the position back to `len` (entries past it are overwritten when written again).
    #[cfg(feature = "cuda")]
    pub(crate) fn rewind(&mut self, len: usize) {
        self.length = len;
    }

    /// Makes this cache a copy of `src` (same shape): contents and position, in this cache's
    /// own buffers on CUDA (forking one request's state into another slot).
    pub fn copy_from(&mut self, src: &KvCache) {
        // A growing cache copies only the positions `src` holds; a ring all of it.
        let held = match self.window {
            None => src.length.min(self.capacity),
            Some(_) => self.capacity,
        };
        store_prefix(self.k.as_mut().expect("cache"), src.k.clone().expect("cache"), 2, held);
        store_prefix(self.v.as_mut().expect("cache"), src.v.clone().expect("cache"), 2, held);
        self.length = src.length;
    }

    /// What [`Self::restore`] needs to return here.
    pub fn snapshot(&self) -> KvSnapshot {
        let ring = self.window.map(|_| {
            (copy_of(self.k.as_ref().expect("cache")), copy_of(self.v.as_ref().expect("cache")))
        });
        KvSnapshot { len: self.length, ring }
    }

    /// Back to a [`Self::snapshot`] of this cache (ring contents written into its own buffers
    /// on CUDA).
    pub fn restore(&mut self, snapshot: &KvSnapshot) {
        if let Some((k, v)) = &snapshot.ring {
            store(self.k.as_mut().expect("cache"), k.clone());
            store(self.v.as_mut().expect("cache"), v.clone());
        }
        self.length = snapshot.len;
    }

    fn held(&self) -> usize {
        self.length.min(self.capacity)
    }

    /// Everything currently held; unordered for rings (RoPE is already baked into the keys).
    pub fn view(&self) -> (Tensor<4>, Tensor<4>) {
        let (k, v) = (self.k.clone().expect("cache"), self.v.clone().expect("cache"));
        let n = self.held();
        if n == self.capacity {
            (k, v)
        } else {
            (k.slice(s![.., .., 0..n]), v.slice(s![.., .., 0..n]))
        }
    }

    fn assign(&mut self, at: usize, k: Tensor<4>, v: Tensor<4>) {
        // On CUDA the write always lands in the cache's own buffers, which a captured decoder
        // graph ([`crate::ncm::GraphedDecoder`]) keeps reading.
        #[cfg(feature = "cuda")]
        if crate::fused::active(&self.device) {
            crate::kernels::assign_into(self.k.as_ref().expect("cache"), k, 2, at);
            crate::kernels::assign_into(self.v.as_ref().expect("cache"), v, 2, at);
            return;
        }
        let n = k.dims()[2];
        let (ck, cv) = (self.k.take().expect("cache"), self.v.take().expect("cache"));
        self.k = Some(ck.slice_assign(s![.., .., at..at + n], k));
        self.v = Some(cv.slice_assign(s![.., .., at..at + n], v));
    }

    /// Store new keys and values `[B, KVH, T, D]` and advance.
    pub fn write(&mut self, k: Tensor<4>, v: Tensor<4>) {
        let t = k.dims()[2];
        let start = self.length;
        if self.window.is_none() {
            assert!(start + t <= self.capacity, "cache capacity {} exceeded", self.capacity);
            self.assign(start, k, v);
        } else if t == 1 {
            self.assign(start % self.capacity, k, v);
        } else {
            // Only the last `capacity` positions survive; they may wrap around the ring end.
            let n = t.min(self.capacity);
            let (k, v) = (k.narrow(2, t - n, n), v.narrow(2, t - n, n));
            let slot = (start + t - n) % self.capacity;
            let head = n.min(self.capacity - slot);
            if head == n {
                self.assign(slot, k, v);
            } else {
                self.assign(slot, k.clone().narrow(2, 0, head), v.clone().narrow(2, 0, head));
                self.assign(0, k.narrow(2, head, n - head), v.narrow(2, head, n - head));
            }
        }
        self.length += t;
    }

    /// Absolute position held by each ring slot; empty slots hold a far-future position.
    fn slot_positions(&self) -> Vec<i32> {
        let cap = self.capacity as i64;
        let last = self.length as i64 - 1;
        (0..cap)
            .map(|slot| {
                let pos = last - (last - slot).rem_euclid(cap);
                if pos < 0 { EMPTY_SLOT } else { pos as i32 }
            })
            .collect()
    }

    /// A growing cache's keys and values for a masked multi-token step. On CUDA the view is
    /// padded to [`key_bucket`] slots (the padding masked by the caller), so a prefill over a
    /// growing cache meets a few key lengths, each compiled and allocated once, instead of a new
    /// one per chunk; elsewhere it is exactly [`Self::view`].
    fn padded_view(&self) -> (Tensor<4>, Tensor<4>) {
        #[cfg(feature = "cuda")]
        if crate::fused::active(&self.device) {
            let n = key_bucket(self.length, self.capacity);
            let (k, v) = (self.k.clone().expect("cache"), self.v.clone().expect("cache"));
            if n == self.capacity {
                return (k, v);
            }
            return (k.slice(s![.., .., 0..n]), v.slice(s![.., .., 0..n]));
        }
        self.view()
    }

    /// Add new keys and values; return (K, V, mask) for the queries at the same positions.
    /// Single-token steps need no mask: everything held is visible.
    pub fn append(
        &mut self,
        k: Tensor<4>,
        v: Tensor<4>,
    ) -> (Tensor<4>, Tensor<4>, Option<Tensor<2, Bool>>) {
        let t = k.dims()[2];
        let start = self.length;
        if t == 1 {
            // The slot being overwritten holds position start - window: already out of window.
            self.write(k, v);
            let (keys, values) = self.view();
            return (keys, values, None);
        }
        let qpos: Vec<i32> = (start..start + t).map(|p| p as i32).collect();
        if self.window.is_none() {
            self.write(k, v);
            let (keys, values) = self.padded_view();
            let held = keys.dims()[2];
            // Padding slots (past the length) hold a far-future position: always masked.
            let kpos: Vec<i32> =
                (0..held).map(|p| if p < self.length { p as i32 } else { EMPTY_SLOT }).collect();
            let mask = build_mask(&self.device, &qpos, &kpos, None);
            return (keys, values, Some(mask));
        }
        let keys = Tensor::cat(vec![self.k.clone().expect("cache"), k.clone()], 2);
        let values = Tensor::cat(vec![self.v.clone().expect("cache"), v.clone()], 2);
        let mut kpos = self.slot_positions();
        kpos.extend_from_slice(&qpos);
        let mask = build_mask(&self.device, &qpos, &kpos, self.window);
        self.write(k, v);
        (keys, values, Some(mask))
    }
}

/// Key slots a masked step over `n` cached positions attends to (at most `capacity`): `n`
/// rounded up to an eighth of its next power of two (at least 512), so lengths fall into four
/// buckets per doubling and the padding adds at most a quarter of the keys.
pub fn key_bucket(n: usize, capacity: usize) -> usize {
    const MIN: usize = 512;
    let bucket = if n <= MIN {
        MIN
    } else {
        let step = (n.next_power_of_two() / 8).max(1);
        n.div_ceil(step) * step
    };
    bucket.min(capacity).max(n)
}

/// A copy of `t` in a buffer of its own (on the CPU a clone suffices: writes there copy a
/// buffer another tensor holds).
pub(crate) fn copy_of<const D: usize>(t: &Tensor<D>) -> Tensor<D> {
    #[cfg(feature = "cuda")]
    if crate::fused::active(&t.device()) {
        let c = Tensor::<D>::zeros(t.dims(), &t.device()).cast(t.dtype());
        crate::kernels::assign_into(&c, t.clone(), 0, 0);
        return c;
    }
    t.clone()
}

/// Sets the first `len` entries along `axis` of `slot` to `value`'s (a forked cache's used
/// prefix; the rest of `slot` is left as it is, as positions past a cache's length are never
/// read): written into `slot`'s own buffer on CUDA with the fused kernels on, so a fork copies
/// what its source holds rather than its whole capacity. Elsewhere `slot` becomes `value`.
pub(crate) fn store_prefix<const D: usize>(
    slot: &mut Tensor<D>,
    value: Tensor<D>,
    axis: usize,
    len: usize,
) {
    #[cfg(feature = "cuda")]
    if crate::fused::active(&slot.device()) {
        if len > 0 {
            crate::inplace::write_float(slot, value.narrow(axis, 0, len), axis, 0);
        }
        return;
    }
    let _ = (axis, len);
    *slot = value;
}

/// [`store_prefix`] for int tensors (8-bit indexer-key codes).
pub(crate) fn store_int_prefix<const D: usize>(
    slot: &mut Tensor<D, Int>,
    value: Tensor<D, Int>,
    axis: usize,
    len: usize,
) {
    #[cfg(feature = "cuda")]
    if crate::fused::active(&slot.device()) {
        if len > 0 {
            crate::inplace::write_int(slot, value.narrow(axis, 0, len), axis, 0);
        }
        return;
    }
    let _ = (axis, len);
    *slot = value;
}

/// Sets `slot` to `value`: written into `slot`'s own buffer on CUDA with the fused kernels on,
/// so that graphs captured against the buffer keep seeing the cache, replaced elsewhere.
pub(crate) fn store<const D: usize>(slot: &mut Tensor<D>, value: Tensor<D>) {
    #[cfg(feature = "cuda")]
    if crate::fused::active(&slot.device()) {
        crate::inplace::write_float(slot, value, 0, 0);
        return;
    }
    *slot = value;
}

/// What a cache needs to return to an earlier position: its position, and the contents that
/// moving on would overwrite (sliding rings and DeltaNet states). Append-only storage (growing
/// key/value caches, retrieval latents) only rewinds.
pub enum LayerSnapshot {
    Kv(KvSnapshot),
    Delta((Tensor<4>, Tensor<3>, usize)),
    Latent(usize),
}

/// A [`KvCache`]'s position, and for a ring its keys and values.
pub struct KvSnapshot {
    len: usize,
    ring: Option<(Tensor<4>, Tensor<4>)>,
}

/// Bytes of one layer's cache for a run of `max_positions`.
pub fn layer_cache_bytes(
    kv_heads: usize,
    head_dim: usize,
    max_positions: usize,
    window: Option<usize>,
    itemsize: usize,
    batch: usize,
) -> usize {
    let cap = window.map_or(max_positions, |w| w.min(max_positions));
    2 * batch * kv_heads * cap * head_dim * itemsize
}

/// The decode cache of one layer of a stack, by its sequence mixer.
// One per layer, a few dozen at most: the size gap between variants is not worth boxing for.
#[allow(clippy::large_enum_variant)]
pub enum LayerCache {
    Kv(KvCache),
    Delta(DeltaCache),
    Latent(LatentCache),
}

impl LayerCache {
    /// Next absolute position.
    pub fn len(&self) -> usize {
        match self {
            Self::Kv(c) => c.len(),
            Self::Delta(c) => c.len(),
            Self::Latent(c) => c.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn reset(&mut self) {
        match self {
            Self::Kv(c) => c.reset(),
            Self::Delta(c) => c.reset(),
            Self::Latent(c) => c.reset(),
        }
    }

    /// Makes this cache a copy of `src`, a cache of the same layer and shape (see
    /// [`KvCache::copy_from`]).
    pub fn copy_from(&mut self, src: &LayerCache) {
        match (self, src) {
            (Self::Kv(c), Self::Kv(s)) => c.copy_from(s),
            (Self::Delta(c), Self::Delta(s)) => c.copy_from(s),
            (Self::Latent(c), Self::Latent(s)) => c.copy_from(s),
            _ => panic!("copies come from a cache of the same layer"),
        }
    }

    /// What [`Self::restore`] needs to return here.
    pub fn snapshot(&self) -> LayerSnapshot {
        match self {
            Self::Kv(c) => LayerSnapshot::Kv(c.snapshot()),
            Self::Delta(c) => LayerSnapshot::Delta(c.snapshot()),
            Self::Latent(c) => LayerSnapshot::Latent(c.len()),
        }
    }

    /// Back to a [`Self::snapshot`] of this cache.
    pub fn restore(&mut self, snapshot: &LayerSnapshot) {
        match (self, snapshot) {
            (Self::Kv(c), LayerSnapshot::Kv(s)) => c.restore(s),
            (Self::Delta(c), LayerSnapshot::Delta(s)) => c.restore(s.clone()),
            (Self::Latent(c), LayerSnapshot::Latent(len)) => c.rewind(*len),
            _ => panic!("a snapshot restores the cache it was taken of"),
        }
    }

    /// Bytes allocated.
    pub fn nbytes(&self) -> usize {
        match self {
            Self::Kv(c) => c.nbytes(),
            Self::Delta(c) => c.nbytes(),
            Self::Latent(c) => c.nbytes(),
        }
    }

    /// Advances the position of a DeltaNet or retrieval cache by `t` after a replayed graph
    /// did the step's writes.
    #[cfg(feature = "cuda")]
    pub(crate) fn advance(&mut self, t: usize) {
        match self {
            Self::Delta(c) => c.advance(t),
            Self::Latent(c) => c.advance(t),
            Self::Kv(_) => unreachable!("attention caches are not replayed"),
        }
    }

    pub(crate) fn kv(&mut self) -> &mut KvCache {
        match self {
            Self::Kv(c) => c,
            _ => unreachable!("an attention layer gets the cache it made"),
        }
    }

    pub(crate) fn delta(&mut self) -> &mut DeltaCache {
        match self {
            Self::Delta(c) => c,
            _ => unreachable!("a DeltaNet layer gets the cache it made"),
        }
    }

    pub(crate) fn latent(&mut self) -> &mut LatentCache {
        match self {
            Self::Latent(c) => c,
            _ => unreachable!("a retrieval layer gets the cache it made"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::key_bucket;

    #[test]
    fn key_buckets_are_few_and_cover() {
        assert_eq!(key_bucket(1, 100_000), 512);
        assert_eq!(key_bucket(512, 100_000), 512);
        assert_eq!(key_bucket(513, 100_000), 640);
        assert_eq!(key_bucket(1100, 100_000), 1280);
        assert_eq!(key_bucket(40_000, 100_000), 40_960);
        assert_eq!(key_bucket(300, 200), 300);
        assert_eq!(key_bucket(150, 200), 200);
        let mut seen = std::collections::BTreeSet::new();
        for n in 1..=131_072 {
            let b = key_bucket(n, 1 << 20);
            assert!(b >= n && (b <= 512 || b * 4 <= n * 5));
            seen.insert(b);
        }
        // Four per doubling from 512 to 128k, plus the minimum.
        assert!(seen.len() <= 4 * 8 + 1, "{} buckets", seen.len());
        assert!(seen.len() >= 8);
    }
}
