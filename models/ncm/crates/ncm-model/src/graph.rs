//! Pieces shared by the decode steps captured as CUDA graphs ([`crate::layers::GraphedStack`],
//! [`crate::ncm::GraphedStep`], [`crate::tok::GraphedTok`]).
//!
//! A captured graph replays its launches with the buffers and scalar arguments of the capture,
//! so whatever moves from step to step must sit in a device buffer that the host refreshes
//! before each replay. For attention caches that is the write position and the mask: a
//! [`Cursor`] holds `[slot, position]` on the device for a group of caches that move together,
//! and the step derives the mask from it on the device ([`Cursor::step`]), so one small upload
//! per group replaces the per-layer masks the eager path builds on the host.
//!
//! Every per-step input (RoPE tables, cursor positions, retrieval positions) is staged: the host
//! writes the values into two host vectors and uploads each in one copy ([`Staging::upload`]),
//! and the graph itself begins by copying the staged values into the buffers the layers read
//! ([`Staging::scatter`]). A replay thus costs the host two uploads whatever the model's depth.

/// Without CUDA nothing is captured, so no step carries a cursor.
#[cfg(not(feature = "cuda"))]
pub(crate) enum KvStep {}

#[cfg(feature = "cuda")]
pub(crate) use cuda::*;

#[cfg(feature = "cuda")]
mod cuda {
    use burn::backend::{Backend, BackendGraph, Dispatch};
    use burn::tensor::{Bool, Device, ExecutionError, Int, IntDType, Tensor, TensorData};

    use crate::cache::{KvCache, build_mask};

    /// One attention cache's write slot (`[1]`, i32) and mask (`[T, S]`, `true` hides a key)
    /// for a graphed step, both on the device.
    #[derive(Clone)]
    pub(crate) struct KvStep {
        pub slot: Tensor<1, Int>,
        pub mask: Tensor<2, Bool>,
    }

    /// The device position of a group of attention caches of equal shape that always move
    /// together (one stack's layers, or the NCM encoder and the decoder's cross-attention rings).
    ///
    /// One-token steps (`t == 1`) cover growing caches and rings in any state: the new entry is
    /// written at `slot` and the step attends over the whole buffer, hiding the slots past the
    /// position. Runs of `t > 1` cover only full rings whose size is a multiple of `t`, at a
    /// position that is a multiple of `t` (the NCM encoder): the run attends over the ring
    /// before the write and its own keys, as the eager path does, with the window mask the
    /// eager path builds on the host derived from the position instead.
    struct Cursor {
        /// `[slot, position]` (i32), staged before each replay.
        info: Tensor<1, Int>,
        /// Where `info` is staged.
        staged: usize,
        capacity: usize,
        ring: bool,
        t: usize,
        /// The host position the next refresh uploads.
        len: usize,
        /// `[1, capacity]`: slot indices.
        cols: Tensor<2, Int>,
        /// `t > 1`: `[t, 1]` query offsets and the `[t, t]` causal mask of the run's own keys.
        rows: Option<(Tensor<2, Int>, Tensor<2, Bool>)>,
    }

    impl Cursor {
        fn new(
            cache: &KvCache,
            t: usize,
            device: &Device,
            staging: &mut Staging,
        ) -> Result<Self, String> {
            let (capacity, ring, len) = (cache.capacity(), cache.window().is_some(), cache.len());
            if t > 1 && !(ring && len >= capacity && capacity % t == 0 && len % t == 0) {
                return Err(format!(
                    "attention cache of {capacity} (ring: {ring}) at {len}: graphed runs of {t} \
                     need a full ring whose size and position are multiples of {t}"
                ));
            }
            let ints = |n: usize| {
                Tensor::<1, Int>::arange(0..n as i64, device).cast(IntDType::I32).reshape([1, n])
            };
            let rows = (t > 1).then(|| {
                let pos: Vec<i32> = (0..t as i32).collect();
                (ints(t).reshape([t, 1]), build_mask(device, &pos, &pos, None))
            });
            let info = Tensor::<1, Int>::zeros([2], device).cast(IntDType::I32);
            Ok(Self {
                staged: staging.add_ints(info.clone()),
                info,
                capacity,
                ring,
                t,
                len,
                cols: ints(capacity),
                rows,
            })
        }

        fn matches(&self, cache: &KvCache, t: usize) -> bool {
            let ring = cache.window().is_some();
            (self.capacity, self.ring, self.t, self.len) == (cache.capacity(), ring, t, cache.len())
        }

        fn refresh(&self, staging: &mut Staging) {
            let (len, cap) = (self.len, self.capacity);
            if self.t > 1 {
                assert!(len % self.t == 0, "graphed runs of {} at position {len}", self.t);
            } else {
                assert!(self.ring || len < cap, "cache capacity {cap} exceeded");
            }
            let slot = if self.ring { len % cap } else { len };
            staging.ints(self.staged, 2).copy_from_slice(&[slot as i32, len as i32]);
        }

        /// The slot and the mask, as device ops (recorded into the graph).
        fn step(&self) -> KvStep {
            let cap = self.capacity;
            let slot = self.info.clone().narrow(0, 0, 1);
            let len = self.info.clone().narrow(0, 1, 1).reshape([1, 1]);
            let mask = match &self.rows {
                // Slots past the position are empty (a ring that wrapped has none).
                None => (self.cols.clone() - len).greater_elem(0),
                Some((rows, causal)) => {
                    // Slot s holds position len - 1 - d, d = (len - 1 - s) mod cap; query i
                    // (position len + i) sees it while (len + i) - (len - 1 - d) < cap.
                    let d = (len - self.cols.clone()).add_scalar(cap as i64 - 1);
                    let d = d.remainder_scalar(cap as i64);
                    let hidden = (d + rows.clone()).greater_equal_elem(cap as i64 - 1);
                    Tensor::cat(vec![hidden, causal.clone()], 1)
                }
            };
            KvStep { slot, mask }
        }
    }

    /// The cursors of one captured graph.
    #[derive(Default)]
    pub(crate) struct CursorSet {
        cursors: Vec<Cursor>,
    }

    impl CursorSet {
        /// The cursor of `cache` for runs of `t`: an existing one of the same shape and
        /// position, or a new one. Fails for caches a graph cannot step ([`Cursor`]).
        pub(crate) fn add(
            &mut self,
            cache: &KvCache,
            t: usize,
            staging: &mut Staging,
        ) -> Result<usize, String> {
            if let Some(i) = self.cursors.iter().position(|c| c.matches(cache, t)) {
                return Ok(i);
            }
            let device = cache.buffers().0.device();
            self.cursors.push(Cursor::new(cache, t, &device, staging)?);
            Ok(self.cursors.len() - 1)
        }

        /// Records the host position of cursor `i` (a cache of its group), for [`Self::refresh`].
        pub(crate) fn sync(&mut self, i: usize, cache: &KvCache) {
            self.cursors[i].len = cache.len();
        }

        /// Stages every cursor's slot and position (before a replay).
        pub(crate) fn refresh(&self, staging: &mut Staging) {
            self.cursors.iter().for_each(|c| c.refresh(staging));
        }

        /// Every cursor's slot and mask, indexed as [`Self::add`] returned.
        pub(crate) fn steps(&self) -> Vec<KvStep> {
            self.cursors.iter().map(Cursor::step).collect()
        }
    }

    /// The per-step inputs of one graph, staged on the host and uploaded in one copy per type
    /// (module docs). Targets are registered while the graph is built; the device buffers are
    /// made by the first upload and never move after it.
    #[derive(Default)]
    pub(crate) struct Staging {
        floats: Vec<f32>,
        ints: Vec<i32>,
        float_targets: Vec<(usize, Tensor<4>)>,
        int_targets: Vec<(usize, Tensor<1, Int>)>,
        buffers: Option<(Tensor<1>, Tensor<1, Int>)>,
    }

    impl Staging {
        /// Stages the whole of `target` (a fixed float buffer); returns its offset.
        pub(crate) fn add_floats(&mut self, target: Tensor<4>) -> usize {
            assert!(self.buffers.is_none(), "staging targets are fixed by the first upload");
            let at = self.floats.len();
            self.floats.resize(at + target.shape().num_elements(), 0.0);
            self.float_targets.push((at, target));
            at
        }

        /// Stages the whole of `target` (a fixed i32 buffer); returns its offset.
        pub(crate) fn add_ints(&mut self, target: Tensor<1, Int>) -> usize {
            assert!(self.buffers.is_none(), "staging targets are fixed by the first upload");
            let at = self.ints.len();
            self.ints.resize(at + target.dims()[0], 0);
            self.int_targets.push((at, target));
            at
        }

        /// The host values `at..at + n` of the float stage.
        pub(crate) fn floats(&mut self, at: usize, n: usize) -> &mut [f32] {
            &mut self.floats[at..at + n]
        }

        /// The host values `at..at + n` of the int stage.
        pub(crate) fn ints(&mut self, at: usize, n: usize) -> &mut [i32] {
            &mut self.ints[at..at + n]
        }

        /// Copies the host values to the device stage (before a replay, never inside a
        /// capture).
        pub(crate) fn upload(&mut self, device: &Device) {
            let (floats, ints) = self.buffers.get_or_insert_with(|| {
                let floats = Tensor::<1>::zeros([self.floats.len().max(1)], device);
                let ints = Tensor::<1, Int>::zeros([self.ints.len().max(1)], device);
                (floats, ints.cast(IntDType::I32))
            });
            if !self.floats.is_empty() {
                let data = TensorData::new(self.floats.clone(), [self.floats.len()]);
                crate::kernels::assign_into(floats, Tensor::from_data(data, device), 0, 0);
            }
            if !self.ints.is_empty() {
                let data = TensorData::new(self.ints.clone(), [self.ints.len()]);
                let data = Tensor::<1, Int>::from_data(data, device).cast(IntDType::I32);
                crate::kernels::assign_int_into(ints, data, 0, 0);
            }
        }

        /// Copies the device stage into every target: the first thing a captured graph does.
        pub(crate) fn scatter(&self) {
            let (floats, ints) = self.buffers.as_ref().expect("staged values are uploaded first");
            for (at, target) in &self.float_targets {
                let dims = target.dims();
                let n = dims.iter().product();
                let src = floats.clone().narrow(0, *at, n).reshape(dims);
                crate::kernels::assign_into(target, src, 0, 0);
            }
            for (at, target) in &self.int_targets {
                let src = ints.clone().narrow(0, *at, target.dims()[0]);
                crate::kernels::assign_int_into(target, src, 0, 0);
            }
        }
    }

    /// Writes `k`, `v` `[B, KVH, T, D]` at the step's slot of `cache`'s own buffers and
    /// advances its host position.
    pub(crate) fn write_kv(cache: &mut KvCache, k: Tensor<4>, v: Tensor<4>, step: &KvStep) {
        let t = k.dims()[2];
        let (ck, cv) = cache.buffers();
        crate::kernels::assign_into_at(ck, k, 2, &step.slot);
        crate::kernels::assign_into_at(cv, v, 2, &step.slot);
        cache.advance(t);
    }

    /// What a warm-up step changed in an attention cache, to undo: the position, and for runs
    /// of more than one token the buffers (such a run reads ring entries it then overwrites).
    pub(crate) struct KvSaved {
        len: usize,
        buffers: Option<(Tensor<4>, Tensor<4>)>,
    }

    pub(crate) fn save_kv(cache: &KvCache, t: usize) -> KvSaved {
        let copy = |x: &Tensor<4>| {
            let c = Tensor::<4>::zeros(x.dims(), &x.device());
            crate::kernels::assign_into(&c, x.clone(), 0, 0);
            c
        };
        let (k, v) = cache.buffers();
        KvSaved { len: cache.len(), buffers: (t > 1).then(|| (copy(k), copy(v))) }
    }

    pub(crate) fn restore_kv(cache: &mut KvCache, saved: &KvSaved) {
        if let Some((k, v)) = &saved.buffers {
            let (ck, cv) = cache.buffers();
            crate::kernels::assign_into(ck, k.clone(), 0, 0);
            crate::kernels::assign_into(cv, v.clone(), 0, 0);
        }
        cache.rewind(saved.len);
    }

    /// Records `run(ctx)` as a CUDA graph on `device`'s stream: `warmups` real runs first
    /// (compiling every kernel and filling the memory pool with every buffer the capture will
    /// take, so nothing allocates mid-capture), each undone by `undo(ctx)`, then the capture,
    /// whose host-side effects are undone as well (recording runs nothing on the device).
    /// Returns the graph and the output tensors of the recorded run, which every replay writes.
    pub(crate) fn record<C: ?Sized, T>(
        device: &Device,
        ctx: &mut C,
        warmups: usize,
        run: impl Fn(&mut C) -> T,
        undo: impl Fn(&mut C),
    ) -> Result<(Captured, T), String> {
        let dispatch = device.as_dispatch();
        let err = |what: &str, e: ExecutionError| format!("{what}: {e:?}");
        Dispatch::graph_prepare(dispatch).map_err(|e| err("graph_prepare", e))?;
        for _ in 0..warmups {
            let out = run(ctx);
            Dispatch::sync(dispatch).map_err(|e| err("warm-up", e))?;
            drop(out);
            undo(ctx);
        }
        Dispatch::sync(dispatch).map_err(|e| err("restore", e))?;
        Dispatch::graph_start_capture(dispatch).map_err(|e| err("graph_start_capture", e))?;
        crate::fused::set_capturing(true);
        let out = run(ctx);
        crate::fused::set_capturing(false);
        let graph = Dispatch::graph_stop_capture(dispatch);
        undo(ctx);
        let graph = graph.map_err(|e| err("graph_stop_capture", e))?;
        Ok((Captured::new(graph), out))
    }

    /// A captured graph and the thread that captured it. cubecl gives every OS thread its own
    /// stream, and a replay is ordered only against the capture's stream: the inputs a caller
    /// stages and the outputs it reads are ordered with the replay only on that thread. So a
    /// graph replays only on its capturing thread ([`replay`] checks); a server runs all of a
    /// model's GPU work on one thread.
    pub(crate) struct Captured {
        graph: BackendGraph<Dispatch>,
        thread: std::thread::ThreadId,
    }

    impl Captured {
        pub(crate) fn new(graph: BackendGraph<Dispatch>) -> Self {
            Self { graph, thread: std::thread::current().id() }
        }
    }

    /// Replays `graph` on `device`'s stream. Panics on a thread other than the capturing one.
    ///
    /// # Safety
    ///
    /// Every buffer the graph touched must still be alive (held by the caller's graph inputs
    /// and outputs, the model's weights or the caches).
    pub(crate) unsafe fn replay(device: &Device, graph: &Captured, what: &str) {
        assert_eq!(
            graph.thread,
            std::thread::current().id(),
            "the {what} graph replays only on the thread that captured it (its stream)"
        );
        unsafe { Dispatch::graph_replay(device.as_dispatch(), &graph.graph) }
            .unwrap_or_else(|e| panic!("{what} graph replay: {e:?}"));
    }
}
