//! Preallocated KV caches shared by both models.
//!
//! Full-attention layers keep `capacity` positions and grow into it. Sliding layers keep a ring
//! buffer of exactly `window` positions. Positions are contiguous: the cache itself tracks the
//! next absolute position (`length`), so callers never pass positions.
//!
//! Masks follow the Burn convention: `true` marks a key the query must not see.

use burn::tensor::{Bool, Device, Int, Tensor, TensorData, s};

use crate::config::is_full_layer;

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
            let (keys, values) = self.view();
            let kpos: Vec<i32> = (0..self.length).map(|p| p as i32).collect();
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

/// Bytes of a layer stack's caches.
#[allow(clippy::too_many_arguments)]
pub fn stack_cache_bytes(
    n_layers: usize,
    full_every: usize,
    window: usize,
    max_positions: usize,
    kv_heads: usize,
    head_dim: usize,
    itemsize: usize,
    batch: usize,
) -> usize {
    (0..n_layers)
        .map(|i| {
            let w = (!is_full_layer(i, full_every)).then_some(window);
            layer_cache_bytes(kv_heads, head_dim, max_positions, w, itemsize, batch)
        })
        .sum()
}
