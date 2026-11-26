# Plan: CPU offload for the NCM runtime

Revision: 2026-11-26, written against `b1f20c6` (feat/ncm-review-fixes tip), last updated 2026-11-26 (rev 2: plan-gate review of `9eb3b68`; rev 3: footprint types and Stage 1/5 ordering aligned with docs/plans/npu.md). Nothing in it is implemented or measured yet. Every number marked *est.* is an estimate.

> **Note, 2026-11-26: NCM numbers are provisional.** The user chose a hybrid NCM. The 4-token local decoder is replaced by the Stanford NCM paper's coarse-to-fine parallel decoder over blocks of about 32–64 tokens, and the backbone runs once per block. The design is in progress in `Imperium-Research/docs/next-concept-modelling/architecture.md`.
>
> Until the block shape is fixed, every NCM-specific number below is provisional:
> - whether the NCM needs tiering at 1M tokens at all;
> - the token thresholds where retrieval traffic starts and goes flat;
> - the NCM estimates in Stages 3, 4 and 6;
> - which stage owns `GraphedStep` and `speculative.rs`.
>
> The token-baseline numbers stand.
>
> With the backbone running once per 32–64 tokens, 1M tokens is only 16k–32k backbone positions (*est.* about 0.16–0.31 GiB of NCM retrieval cache), so the NCM may never need tiering at 1M. Streaming backbone layers (Stage 6), on the other hand, gets cheaper per token, so it may move ahead of Stage 4 for the NCM once re-estimated. Stage 0 redoes these numbers.
>
> The planner and footprint take `tokens_per_position` as a parameter instead of hard-coding k = 4: 1 for the token baseline, k or the block length for the NCM.

Goals (each one passes or fails):

1. With the v2 retrieval caches tiered, cache VRAM at 1,048,576 tokens of context stays under the bounds below. Cache VRAM here means every device byte the caches own:
   - indexer keys;
   - the decode tail rings;
   - the prefill chunk-staging buffers;
   - the union slab, at its minimum size of 1,024 blocks.

   Bounds:
   - token baseline: under 2 GiB (*est.* 1.45 GiB; 12.52 GiB today);
   - NCM: under 0.4 GiB (*est.* 0.36 GiB at the current 4-token shape; 2.51 GiB today; provisional).

   Check: `ncm-server --print-footprint --arch tok --random-weights -c 1048576 --cache-offload on` shows the device cache total, with one row per part, under 2 GiB. The same command with `--arch ncm` shows under 0.4 GiB.
2. Tiered decode returns the same results as device-resident decode: identical logits in f32 and identical greedy tokens in bf16, both eager and graphed, for both models. Check: `cargo test -p ncm-model --features cuda --test offload`.
3. With a forced VRAM budget, `ncm-server` launches, places the model by itself and serves requests at a context that does not fit fully resident on the GPU. Check: `cargo test -p ncm-server --features cuda --test offload_e2e` (it starts the server with `--vram-budget`).
4. NCM v2 decode at 128k context with the cache tiered on PCIe 5.0 x16 runs at 85% or more of resident decode speed. Check: `ncm-bench --model ncm --design v2 --context 131072 --cache-offload on` against the same run with `off`.
   - At the current shape this needs at least about 35 GB/s on cold-block reads: 18.9 MB in at most 0.54 ms per concept.
   - The decode-path gate in Stage 0a uses this same 35 GB/s threshold (see Risks).
   - The threshold is re-derived in Stage 0 once the block shape is fixed.
5. `-ngl N` with N below the backbone depth streams the remaining backbone layers over PCIe and still decodes correctly. Check: `cargo test -p ncm-model --features cuda --test offload streamed_layers`.
6. CI stays green on CPU: `scripts/check.sh`. Placement, footprint and flag parsing are all tested without a GPU.

**Scope.** `crates/ncm-model` (caches, retrieval kernels, placement, host memory), `runtime` (CLI flags, engine start-up, footprint report) and `crates/ncm-bench` (offload flags). Both models, v2 design. The fallback design gets footprint reporting only.

**Non-goals.**
- Tiering the indexer keys. Every step scans all of them (for the token baseline at 1M, that is 1.3 GB per token over PCIe), so they stay on the GPU.
- Moving DeltaNet state, or the encoder/decoder rings, off the GPU (a few MiB in total).
- Quantising the latent cache. It could later halve offload traffic, and `--cache-type-k` is the natural flag for it, but it is a separate plan.
- Running host layers on the CPU backend (rejected below; revisit conditions given).
- A wgpu implementation. The design keeps a slot open for it.
- Multi-GPU split.
- Fallback-design KV offload.
- Changing Imperium's launch line.

**Assumptions.**
- The in-flight optimisation branches have landed first:
  - shape-stable prefill, the persistent compile cache and top-k select (touch `cache.rs`, `layers.rs` attention and `retrieval.rs`);
  - the fused norm/GEMV/MLP work (touches `kernels/gemv.rs`, `kernels/quant.rs`, `kernels/norm.rs` and the `layers.rs` MLP).

  Stage 0 rebases this plan's paths onto the result.
- Only the NCM-specific work waits for the hybrid NCM block decoder (see the note above); Stage 0b re-derives those items once it lands. Stage 1 and the token-baseline path of Stages 2–5 depend only on the optimisation branches and Stage 0a.
- Burn 0.22.1 with burn-cuda/cubecl 0.11.4. A CUDA build sits on a PCIe 4.0 or 5.0 x16 link.
- Effective bandwidth over PCIe *est.*:
  - PCIe 5.0 x16: about 40 GB/s for zero-copy kernel reads and about 50 GB/s for pinned memcpy.
  - PCIe 4.0: half of each.
  - The RTX 5080's own VRAM: about 900 GB/s.
- Imperium passes `--n-gpu-layers 999` (`imperium-local/src/backend.rs`, which clamps to 999) and a `--ctx-size`.
- Testing is paused. Stages that need a GPU run once the user resumes testing, following the local-GPU-gates rule; CI covers the CPU side.

**Relevant paths** (under `models/ncm/`):
- `crates/ncm-model/src/`: `cache.rs`, `retrieval.rs`, `kernels/retrieval.rs`, `kernels/mod.rs`, `inplace.rs`, `graph.rs`, `ncm.rs`, `tok.rs`, `layers.rs`, `quant/linear.rs`, `kernels/quant.rs`, `kernels/gemv.rs`, `weights.rs`, `fused.rs`, `lib.rs`. New files: `placement.rs`, `host.rs`.
- `runtime/src/`: `cli.rs`, `engine.rs`, `app.rs`, `server.rs`.
- `crates/ncm-bench/src/lib.rs`.

**Related docs.** Where they conflict, current code and `docs/attention.md` win over this plan, and this plan wins over the research docs for runtime placement.
- `docs/attention.md`: kernel table, cache at 1M tokens.
- `docs/quantisation.md`
- `runtime/README.md`: the llama-server contract.
- Read-only research docs: `Imperium-Research/docs/next-concept-modelling/architecture.md` (cache-at-1M table) and `runtime.md`.

## Decision record

### Where VRAM goes, and which part to offload first

Footprint at 1M tokens, bf16, batch 1:

| Part | NCM v2 | Token v2 |
|---|---|---|
| Weights (bf16 / q4 g64) | 0.50 / about 0.15 GB | 0.50 / about 0.15 GB |
| Retrieval latents (MLA 512 + RoPE 64) | 2.25 GiB | 11.25 GiB |
| Indexer keys (8-bit + scale) | 0.254 GiB | 1.27 GiB |
| DeltaNet, encoder/decoder rings | about 5 MiB | about 3 MiB |
| Prefill transients (indexer score groups of at most 64M f32 scores, chunk activations) *est.* | 0.3–0.6 GB | 0.3–0.6 GB |
| CUDA context, graph pools *est.* | 0.3–0.5 GB | 0.3–0.5 GB |

At long context the latent cache is the dominant term, and it grows with both context and depth. Weights are small at 250M; they only start to matter for larger future models.

Order:
1. Cache tiering first: most benefit, and the only offload that makes the token baseline at 1M fit under 16 GB.
2. Automatic fit next.
3. Weight streaming last.

### (b) Retrieval-cache tiering: chosen

Each retrieval layer's `LatentCache` splits into two tiers:
- **Device tail ring** of `ret_local + ret_block` positions (1,088), sized for decode only. It always holds the local window and the block being filled.
- **Cold store** of complete blocks in pinned host memory.

Prefill adds a third, transient device buffer per layer, described under "Prefill staging" below.

The indexer keys stay whole on the device, so scoring, block maxima and top-k (`ret_index_kernel` and `ret_topk_kernel`) are unchanged.

Why this works:
- A block only becomes a candidate once it starts before the local window.
- A block is spilled to the host as soon as it completes. The host knows when (`length % ret_block == 0`), so the spill is an async copy of 73,728 B issued between graph replays, about 1,000 steps before any query can select that block.

How the attend kernels read the tiers:
- **Decode (`ret_attend_kernel`):** reads selected cold blocks directly from host-mapped pinned memory (zero-copy over UVA, at a fixed device pointer) and the local window from the tail ring.
- **Prefill (`ret_prefill_attend_kernel`):** reads the chunk-staging buffer plus the union slab (Stage 4), because many queries in a chunk re-read the same blocks.

### Prefill staging and the union slab

A prefill chunk (8,192 positions for the token baseline; 2,048 concepts for the NCM at k = 4) is far larger than the 1,088-position ring. So each retrieval layer gets a **device chunk-staging buffer** of `(ret_local + ret_block + chunk) × 1,152 B`.

How a chunk runs:
1. At chunk start, the ring's pre-chunk tail is copied into the head of the staging buffer (device to device). Then `ret_prep` writes the chunk's entries after it, so one buffer covers every query's local window.
2. Picked blocks fall into two groups:

   | Picked block | Where it is read |
   |---|---|
   | Inside staging (index ≥ `staging_first_block`) | Directly from staging |
   | Older (cold) | Through the union slab |

   The union slab works like this:
   - A device kernel builds the union of the older blocks the chunk's queries picked.
   - A gather copies those blocks from the cold store into one **union slab** shared by all layers. Its size is chosen by the planner: at least 1,024 blocks (75.5 MB), at most every candidate block.
   - The attend kernel reads the slab through a block-to-slot remap.
   - If the union overflows the slab, the query chunk is split.
3. After the layer's chunk:
   - every block completed inside the chunk is spilled from staging to the cold store;
   - the last 1,088 positions are copied back into the ring.

Footprint term per model: `retrieval_layers × (ret_local + ret_block + chunk) × 1,152 B`.
- Token baseline: 10 × 9,280 × 1,152 B ≈ 107 MB.
- NCM: 8 × 3,136 × 1,152 B ≈ 29 MB (provisional).

Halving the prefill chunk (fit step 3) shrinks it.

**Decision:** the union slab and the staging buffers count as cache VRAM, both in goal 1 and in `--print-footprint`. The planner sizes the slab from the budget left over after everything else.

Why zero-copy:
- Selection happens on the device.
- Explicit host-issued copies would need the picked indices on the host. That is a device-to-host sync per layer per step, which breaks CUDA graphs and adds about 8 × 20 µs per concept *est.*
- A kernel that reads mapped host memory needs no sync. It also keeps fixed addresses, so it is graph-safe.

Device VRAM after tiering, at 1M, with the minimum 1,024-block slab:

| | Indexer keys | Rings | Staging | Slab | Total | Today |
|---|---|---|---|---|---|---|
| NCM (provisional) | 0.254 GiB | 10 MB | 29 MB | 72 MiB | about **0.36 GiB** | 2.51 GiB |
| Token baseline | 1.27 GiB | 12.5 MB | 107 MB | 72 MiB | about **1.45 GiB** | 12.52 GiB |

Host memory (pinned): 2.25 GiB for the NCM and 11.25 GiB for the token baseline.

Per-step PCIe reads, worst case with no reuse: `topk × block × 1,152 B` per retrieval layer, which is 2.36 MB.

| | Bytes per step | PCIe 5.0 (40 GB/s) | PCIe 4.0 (20 GB/s) | Effect *est.* |
|---|---|---|---|---|
| NCM (8 layers, per 4-token concept) | 18.9 MB | +0.47 ms per concept | +0.94 ms per concept | 128k: 1,304 → about 1,130 tok/s (−13%) on PCIe 5, about 1,000 (−23%) on PCIe 4 |
| Token baseline (10 layers, per token) | 23.6 MB | +0.59 ms per token | +1.18 ms per token | about −40% to −60% at today's per-token cost (its offload traffic is per token, not per concept) |

Traffic depends on context length:
- It is zero until blocks become candidates: NCM below 4,096 tokens, token baseline below 1,024 tokens.
- It reaches its flat maximum once 32 blocks are candidates: NCM at about 12k tokens, token baseline at about 3k tokens.
- Beyond that, the cost does not grow with context.
- The 2k numbers are therefore unaffected.

Today the NCM at 1M fits resident on any card of 8 GB or more, so in `auto` mode tiering only switches on when it must (Stage 5).

### Alternatives considered for the cache

| Option | Verdict | Why |
|---|---|---|
| Keep everything resident (status quo) | Kept as the default when it fits | Fastest; already done |
| Host-issued block copies after reading the picked indices back | Rejected for CUDA; kept as the design for a future wgpu store | Sync per layer, no graphs |
| Zero-copy reads of cold blocks in the attend kernels | **Chosen** for decode | No sync, graph-safe, uses the existing top-k |
| Union staging (gather the chunk's picked blocks into a device slab, then attend through a remap table) | **Chosen** for prefill (Stage 4) | Each block crosses PCIe once per chunk and layer |
| GPU hot-block cache reusing the previous step's selection | Deferred (Stage 7, gated) | Its gain depends on how much the trained model's selection churns between steps, which cannot be known from untrained weights |
| Tiering the indexer keys too | Rejected (non-goal) | Every step scans every key |

### (c) DeltaNet state

It stays with its layer, on the GPU: about 0.5 MiB per layer. Under the chosen weight design every layer computes on the GPU, so the state never moves.

### (a) Weight offload by layer: stream, don't compute on the CPU

Behaviour of `-ngl N`:
- N ≥ number of backbone layers (16 for the NCM, 20 for the token baseline), or `all`, or `999`: everything on the GPU, as today.
- 0: the CPU device, as today.
- In between: the **first** `layers − N` backbone layers are *streamed*. This follows llama.cpp, which also keeps the first layers off the GPU.

A streamed layer works like this:
- Its weights (dense or q8/q4 codes) live in pinned host memory.
- Decode reads them zero-copy straight into the fused GEMV kernels (`kernels/gemv.rs`, `kernels/quant.rs`): batch-1 decode reads each weight once anyway.
- Prefill copies the layer into a fixed, double-buffered device slab before its GEMMs, because GEMMs reuse each weight across rows. Prefill runs eagerly, so the next layer's copy overlaps the current layer's compute through stream order.

The NCM's local encoder, decoder, draft head and output head always stay on the GPU when N > 0:
- The decoder runs k times per concept, and the draft head runs on the same per-token decode path.
- Streaming the decoder would cost 31 MB per token.

Cost per concept for streamed NCM layers (12.5M parameters each) *est.*:

| Weights | Per layer | Half the backbone (8 layers) | Speed with 8 layers streamed |
|---|---|---|---|
| bf16 | about 25 MB, +0.63 ms | +5 ms | about 500 tok/s |
| q4 | about 7 MB, +0.18 ms | +1.4 ms | about 900 tok/s |

That is why the fit loop (Stage 5) prefers q4 hints and cache tiering before it streams weights.

Rejected: running host layers on burn-flex.
- Batch-1 decode is memory-bound either way: dual-channel DDR5 is about 60–80 GB/s peak against PCIe 5.0 at about 40–50 GB/s.
- But on the CPU:
  - burn-flex has no fused or quantised GEMV;
  - weights would likely run as f32, doubling their bytes;
  - prefill would be compute-bound on the CPU;
  - every crossing between CPU and GPU would split the captured graph into segments with host syncs.
- Streaming keeps one kernel set, quantised bytes and graphs.
- **Revisit** when the link is PCIe x4 or an eGPU (under about 8 GB/s), or if burn-flex gains a quantised GEMV.

### (d) Placement: one planner, explicit flags, automatic fit

`ncm_model::placement` is plain Rust with no backend types. From a config, context, quantisation and VRAM budget it computes a `Placement`:
- per backbone layer: weights `Device` or `Streamed`;
- per retrieval cache: `Resident` or `Tiered`;
- the prefill chunk;
- a `Footprint` broken down by tier and by part.

The footprint types (`Footprint`, `Tier`, `Part`, `FootprintRequest` with `tokens_per_position`, and `Placement`) live in the Burn-free `ncm-exec` crate, which the NPU backend-interface plan (`docs/plans/npu.md`, decision 2) creates in its Stage 1a. The tiers are `Device`, `PinnedHost`, `SystemRam` and `NpuCompileCache`. `ncm-model` depends on `ncm-exec` unconditionally, and the planner itself stays in `ncm_model::placement`.

How `--fit` (default `on`) chooses:
1. Everything resident, if it fits under `free VRAM − --fit-target` (MiB, default 1024).
2. Otherwise tier the retrieval caches, cheapest per byte saved.
3. Otherwise halve the prefill chunk, down to 1,024.
4. Otherwise stream backbone layers, one at a time from the first.
5. Otherwise fail with a message that gives the largest context that fits.

Pinned host memory cannot be swapped, so the planner also checks available physical RAM. It keeps a host margin of at least 25% of RAM or 4 GiB.

Rules for the planner (root decisions, 2026-11-26):
- Explicit flags from Imperium win: an explicit `-ngl` below the layer count, `--cache-offload on|off` and `-nkvo` are always respected.
- ncm-server's own fit applies only where placement is unspecified or `auto`.
- `-ngl` ≥ the layer count (Imperium's 999), `all` or `auto` means "as many as fit": fit may lower it, and says so on stderr.

Everything is parameterised by `tokens_per_position`, so the hybrid NCM's block length drops in without planner changes.

New flags:

| Flag | Values | Notes |
|---|---|---|
| `--fit` | `on` / `off` | Fit to free VRAM at launch |
| `--fit-target` | MiB | VRAM margin kept free |
| `--cache-offload` | `auto` / `on` / `off` | `-nkvo` / `--no-kv-offload` is an alias for `on` |
| `--vram-budget` | MiB | Pretends only that much VRAM is free. This is how tests exercise offload on a 16 GB card |
| `--print-footprint` | | The queued flag, extended with a tier column and the placement decision |

`-fit` and `-fitt` are normalised like `-ngl`. Free VRAM comes from `cuMemGetInfo` through cudarc. The wgpu backend has no portable query, so fit there uses `--vram-budget` or the adapter limits.

### (e) Interactions

- **CUDA graphs**
  - Cold-store and streamed-weight buffers are allocated once, at fixed pinned addresses, so a replay keeps reading the right memory and the cache epoch is unchanged.
  - Spills are host-issued between replays, ordered on the same stream.
  - The tail-ring slot is derived on the device from the existing `position` buffer.
  - Nothing new has to be staged for graph replays (`graph.rs` `Staging`).
  - Prefill union staging runs only in the eager or shape-stable prefill path. If shape-stable prefill is graphed, the union size must be capped at a fixed slab, with the chunk split on overflow.
  - Prefetch overlaps compute differently on each path:

    | Path | How the copies overlap compute |
    |---|---|
    | Decode | No separate prefetch: the zero-copy reads happen inside the memory-bound attend kernel. Stage 7 may prefetch reused blocks |
    | Prefill | The double-buffered slab overlaps the next layer's copy with the current layer's compute |
- **Quantisation**
  - q8/q4 cut streamed-weight bytes by 2–3.5×.
  - Indexer keys are already 8-bit.
  - Latents stay at the run dtype (non-goal).
  - The footprint counts quantised bytes.
- **Prefix-reuse snapshots**
  - `LayerCache::snapshot` takes `&self` (`cache.rs:342`), so a snapshot cannot flush into the cold store.
  - Instead, a tiered latent snapshot becomes `LayerSnapshot::Latent { len, tail }`, where `tail` is a device copy of the partial block: at most 63 positions × 1,152 B, taken with `copy_of`, as `KvCache::snapshot` already does for rings.
  - Positions in complete blocks are already on the host, because they were spilled when the block completed.
  - `restore(&mut self)` does two things:
    - refills the ring's complete-block part from the cold store (at most 1,088 × 1,152 B = 1.25 MB per layer);
    - writes `tail` back into the ring.
  - Resident caches keep the plain length (`tail: None`).
  - Both `Kept` entries in `engine.rs` therefore stay valid.
- **Single engine thread.**
  - Placement, pinned allocation (at `Core` construction) and spills all run on `ncm-engine`, so no new OS threads are added.
  - Pinning 11 GiB is expected to take about 1–3 s at launch *est.*; it is logged.
- **Footprint reporting.** `--print-footprint` and the start-up line print device bytes, pinned-host bytes and the placement, from the same `ncm_exec::Footprint` the planner returns, so the report and the allocation cannot disagree.
- **Backend-agnostic.**
  - Placement and footprint carry no backend types.
  - The cold store sits behind a `ColdStore` trait with three implementations:

    | Implementation | Backend | Behaviour |
    |---|---|---|
    | `MappedHost` | CUDA | Zero-copy reads |
    | `Resident` | Every backend | Today's behaviour; the only one on CPU, where host memory is the device |
    | `StagedHost` | wgpu, later | Index readback plus `copy_buffer_to_buffer` into the same slab and remap that union staging uses, without graphs |

  - The kernels take a "cold blocks" binding plus an optional remap table, so the wgpu path reuses the Stage 4 kernel variant.

### Risks and revisit conditions

- **Can cubecl 0.11.4 bind host-mapped memory?** This is the main risk. Stage 0a spikes it, with no dependencies: `cuMemHostAlloc(DEVICEMAP)` or `cuMemHostRegister`, then the device pointer as a cubecl binding. If it cannot, the fallbacks are, in order:
  1. a small cudarc PTX gather kernel on the same stream, which copies the picked blocks into a device slab, followed by the remap variant of the attend kernel;
  2. patching cubecl-cuda;
  3. host index readback (eager only, slower).
- **Decode gate, tied to goal 4.** Stage 0a measures two bandwidths on PCIe 5:
  - zero-copy reads from inside an attend-shaped kernel;
  - a dedicated wide-load gather kernel into a slab.

  Then:
  - If the in-attend read reaches at least 35 GB/s, decode reads cold blocks in place.
  - Below that, decode uses gather-into-slab plus the remap attend, provided the gather reaches 35 GB/s.
  - If neither reaches 35 GB/s, goal 4 is reported as unmet and its threshold goes back to root. It is not loosened silently.
- **Selection churn** of the trained model decides whether Stage 7 is worth doing.

## Stage 0a: Host-mapped memory spike
- **Objective:**
  - Prove a cubecl 0.11.4 kernel can read and write pinned host memory mapped into the device at a fixed address, eagerly and inside a captured graph.
  - Measure the two decode-gate bandwidths: in-attend zero-copy reads of 64-position blocks, and a dedicated gather kernel into a slab.
  - It is a throwaway spike. Only the result and the chosen mechanism are kept.
- **Depends on:** none. It can run as soon as GPU testing resumes, in parallel with everything else.
- **Owned paths:** a scratch spike crate or branch, not merged; its result is recorded in this plan.
- **Acceptance checks:**
  - The spike's eager and replayed reads match the host data.
  - Both bandwidths and the resulting decode path (in place, gather-into-slab, or escalate to root) are recorded in Outcome, against the 35 GB/s gate.
- **Status:** pending
- **Outcome:**
- **Handoff notes:** picks the fallback order in Risks. It also fixes the decode path that Stages 2 and 3 build.

## Stage 0: Rebase onto the landed optimisation round
- **Objective:** confirm that the optimisation branches have merged, then update this plan's owned paths, kernel names and token-baseline speed baselines to what landed.
- **Depends on:** the optimisation branches being merged to `main`.
- **Owned paths:** `models/ncm/docs/plans/cpu-offload.md`
- **Acceptance checks:**
  - Every path and kernel named below exists on `main`: `rg -n "ret_attend_kernel|ret_prefill_attend_kernel|LatentCache" models/ncm/crates`.
  - The token-baseline speed baselines are re-recorded with `ncm-bench --model tok --design v2 --context 2048 131072`.
- **Status:** pending
- **Outcome:**
- **Handoff notes:**
  - Docs only.
  - If top-k select moved out of `kernels/retrieval.rs`, re-point Stage 3.

## Stage 0b: Re-derive the NCM numbers after the block decoder
- **Objective:** once the hybrid NCM block decoder has landed, re-derive every NCM-specific number and ownership item, then replace the provisional note at the top with the settled values. The items to re-derive:
  - NCM cache sizes at 1M tokens, and whether the NCM needs tiering at all;
  - the token thresholds where retrieval traffic starts and goes flat;
  - goal 1's NCM bound, and goal 4's threshold (35 GB/s at the current shape);
  - the NCM estimates in Stage 3, the NCM prefill estimate in Stage 4's handoff, and Stage 6's per-token cost;
  - Stage 5's budget table, especially case a: at 16k–32k backbone positions the NCM at 1M may fit resident in 3 GiB and stop exercising tiering, in which case raise the context or lower the budget;
  - which stage owns `GraphedStep` and `speculative.rs`;
  - for the NCM, whether Stage 6 comes before Stage 4.
- **Depends on:** stage 0; the block decoder having landed.
- **Owned paths:** `models/ncm/docs/plans/cpu-offload.md`
- **Acceptance checks:**
  - The NCM speed baselines are re-recorded with `ncm-bench --model ncm --design v2 --context 2048 131072`.
  - No NCM number in the plan is still marked provisional.
- **Status:** pending
- **Outcome:**
- **Handoff notes:**
  - Docs only.
  - The NCM-specific parts of Stages 3–6 wait for this stage; the token-baseline parts do not.

## Stage 1: Placement model, footprint and flags (no behaviour change)
- **Objective:**
  - Add `placement.rs`: the planner from the decision record. It fills the `ncm_exec` `Placement` and `Footprint` types created by NPU Stage 1a, and takes RAM and VRAM inputs as plain numbers.
  - Parse `-ngl` (N / `all` / `auto` / 999), `--fit`, `--fit-target`, `--cache-offload`, `-nkvo` and `--vram-budget`, and normalise `-fit` and `-fitt`.
  - Extend `--print-footprint` with the tier breakdown.
  - With an offload decision not yet backed by code, print the plan and fall back to resident with a warning.
- **Depends on:** stage 0 and NPU plan Stage 1a, which creates `ncm-exec`. This stage runs before NPU Stage 1b, which owns `engine.rs` after it. It does not wait for the block decoder: `tokens_per_position` keeps the planner shape-agnostic. The queued `--print-footprint` should merge first or be folded in here.
- **Owned paths:**
  - `crates/ncm-model/src/{placement.rs, lib.rs}`
  - `crates/ncm-model/src/ncm.rs` (`ncm_cache_bytes`) and `tok.rs`: per-tier splits
  - `runtime/src/{cli.rs, engine.rs (start-up line only), app.rs}`
- **Acceptance checks:**
  - Planner unit tests give the expected placements for 4, 8 and 16 GB budgets at 2k, 128k and 1M for both models. They also check:
    - the per-tier sums match `ncm_cache_bytes`;
    - the staging and slab terms are included;
    - results are right for several values of `tokens_per_position`;
    - the planner refuses, naming the largest context that fits, when a budget cannot hold even the tiered placement (for example, the token baseline at 1M under 2 GiB).

    Command: `cargo test -p ncm-model placement`.
  - CLI parsing tests pass: `cargo test -p ncm-server cli`.
  - `scripts/check.sh` passes.
- **Status:** pending
- **Outcome:**
- **Handoff notes:**
  - Free-VRAM and free-RAM queries live in the runtime, not ncm-model: `cuMemGetInfo` behind `cuda`, and free RAM through `windows-sys` (`GlobalMemoryStatusEx`) or `sysinfo`. Both dependencies were approved by root on 2026-11-26.
  - Prefill transients and the CUDA context are estimated constants until Stage 5 measures them.

## Stage 2: Host-mapped memory primitive (module)
- **Objective:**
  - Turn the mechanism Stage 0a chose into `host.rs` (`HostBuffer`: alloc, device view, async copy-in and copy-out on the run stream, drop) and the `ColdStore` trait with the `MappedHost` and `Resident` implementations.
- **Depends on:** stage 0a; stage 1 (types only).
- **Owned paths:**
  - `crates/ncm-model/src/{host.rs, kernels/mod.rs (binding helper), inplace.rs (copy helpers)}`
  - `crates/ncm-model/Cargo.toml`: cudarc as a direct dependency if needed (approved)
- **Acceptance checks:** a round-trip test passes: a kernel reads a mapped buffer, the result equals the host data, eager and replayed: `cargo test -p ncm-model --features cuda --test offload host_buffer`.
- **Status:** pending
- **Outcome:**
- **Handoff notes:** if Stage 0a chose gather-into-slab, this stage also owns the gather kernel.

## Stage 3: Tiered retrieval cache, decode
- **Objective:**
  - Split `LatentCache` into the device tail ring and a `ColdStore` of complete blocks:
    - spill each completed block from the host-tracked position;
    - `snapshot(&self)` copies the partial block on the device into `LayerSnapshot::Latent { len, tail }`;
    - `restore` refills the ring from the cold store plus `tail`;
    - `nbytes` reports both tiers.
  - Teach `ret_prep_kernel` to write into the ring slot (`position mod ring`).
  - Teach `ret_attend_kernel` to read selected cold blocks through the cold binding and the window from the ring.
  - Make the reference path (`attend_selected`) do the same, so CPU tests cover the logic with the `Resident` store.
  - Apply all of this to the NCM backbone and the token baseline (shared `LatentCache`) and to `GraphedStep`/`GraphedTok`.
- **Depends on:** stage 2. The NCM `GraphedStep` part also waits for stage 0b.
- **Owned paths:** `crates/ncm-model/src/`:
  - `retrieval.rs`, `kernels/retrieval.rs` (prep, attend)
  - `cache.rs` (`LayerCache` snapshot/restore)
  - `ncm.rs` (`make_cache`, `NcmCache`, `GraphedStep` refresh), `tok.rs`, `graph.rs`
  - `crates/ncm-bench/src/lib.rs`: `--cache-offload`, `--vram-budget`, `--n-gpu-layers`, which pass a `Placement` straight through
- **Acceptance checks:**
  - Tiered decode matches resident decode (f32 logits bitwise, bf16 greedy tokens) at contexts crossing 0, 1 and 40 candidate blocks, eager and graphed, both models: `cargo test -p ncm-model --features cuda --test offload decode`.
  - Snapshot, restore and prefix reuse across tiers: `cargo test -p ncm-server --features cuda --test engine_cuda prefix_reuse_tiered`.
  - CPU parity for the reference path: `cargo test -p ncm-model tiered`.
  - Speed is recorded: `ncm-bench --cache-offload on|off --context 16384 131072`.
- **Status:** pending
- **Outcome:**
- **Handoff notes:**
  - Ring size `ret_local + ret_block`, for decode only. The slot arithmetic must match `candidate_blocks`.
  - Until Stage 4 lands, a tiered cache is prefilled resident and then converted: blocks are spilled and the ring is filled after prefill. That costs full VRAM during prefill, but it is correct, and it is only allowed with an explicit `--cache-offload on`.
  - The NCM `GraphedStep` part is provisional until Stage 0 settles the block decoder.

## Stage 4: Tiered prefill (chunk staging and union slab)
- **Objective:** per retrieval layer and prefill chunk, as in "Prefill staging and the union slab":
  1. copy the ring's tail into the head of the chunk-staging buffer, and have `ret_prep` write the chunk after it;
  2. a device kernel builds the bitmap and the compacted list of cold blocks (those older than `staging_first_block`) picked by any query, from the `ret_select_kernel` output;
  3. a gather copies those blocks into the shared union slab (planner-sized, at least 1,024 blocks);
  4. `ret_prefill_attend_kernel` reads blocks inside staging directly, and cold blocks through the block-to-slot remap;
  5. after the chunk, spill the blocks completed inside it from staging to the cold store, and copy the last 1,088 positions back into the ring.

  On slab overflow, split the query chunk.
- **Depends on:** stage 3. NCM numbers wait for stage 0b.
- **Owned paths:**
  - `crates/ncm-model/src/{retrieval.rs (prefill), kernels/retrieval.rs (prep target, select, prefill attend, new union/gather kernels)}`
  - `crates/ncm-model/src/cache.rs` (staging buffer in `LatentCache`)
  - the `placement.rs` staging and slab terms
- **Acceptance checks:**
  - Tiered prefill equals resident prefill (f32 bitwise, bf16 greedy tokens afterwards) in each of these cases:
    - a chunk larger than the ring, whose queries pick blocks inside the chunk itself;
    - a union that overflows the slab;
    - a union that does not;
    - a prompt ending mid-block, followed by decode.

    Command: `cargo test -p ncm-model --features cuda --test offload prefill`.
  - The footprint of staging plus slab matches the allocated bytes: `cargo test -p ncm-model --features cuda --test offload prefill_footprint`.
  - The 1M-token token-baseline prefill completes under a 6 GiB `--vram-budget`, and both the time and the number of union-overflow sub-chunks are logged: `ncm-bench --model tok --design v2 --context 1048576 --vram-budget 6144`.
- **Status:** pending
- **Outcome:**
- **Handoff notes:**
  - Worst-case PCIe traffic for a full 1M prefill *est.*:
    - NCM: 144 GB, about 3 s. Provisional: Stage 0b re-derives it.
    - Token baseline: about 700 GB, about 14 s at 50 GB/s.
  - Both estimates hold only when the slab is close to the union size, so that each block crosses PCIe once per chunk and layer.
  - With the minimum 1,024-block slab, an overflowing union splits the chunk into sub-chunks, and blocks shared between sub-chunks cross PCIe again. That is why the 1M test logs the sub-chunk count.
  - The union is usually far smaller than the worst case.
  - The same slab-and-remap variant is what a wgpu `StagedHost` would use.

## Stage 5: Automatic fit at launch
- **Objective:**
  - Wire the planner into `BurnBackend` construction (`crates/ncm-model/src/exec.rs`, created by NPU Stage 1b): query free VRAM and RAM, apply `--fit`, `--fit-target` and `--vram-budget`, allocate per `Placement`, and log the decision.
  - `engine.rs` only passes the flags through.
  - Replace the estimated constants for prefill transients and the CUDA context with measured ones.
  - Document the flags in `runtime/README.md`.
- **Depends on:** stages 3 and 4, and NPU plan Stage 1b. Cases b and c below need only the token-baseline path; case a waits for stage 0b.
- **Owned paths:**
  - `crates/ncm-model/src/{exec.rs (BurnBackend construction), placement.rs}`
  - `runtime/src/{cli.rs, app.rs, server.rs (props/footprint in health, if exposed)}`; `engine.rs` gets flag pass-through only, coordinated with the NPU plan, which owns `engine.rs`
  - `runtime/README.md`
- **Acceptance checks:**
  - End to end, the server starts, picks tiering, and its completions match a resident run of the same model on a short prompt, in each case below. All use `--random-weights` and `--fit-target 512`. Command: `cargo test -p ncm-server --features cuda --test offload_e2e`.

    | Case | Model | Context | `--vram-budget` | Resident would need *est.* | Tiered needs *est.* |
    |---|---|---|---|---|---|
    | a | NCM v2 (provisional) | 1,048,576 | 3 GiB | about 4.5 GiB | about 2.4 GiB |
    | b | Token v2 | 262,144 | 4 GiB | about 5.1 GiB | about 2.3 GiB |
    | c | Token v2 | 1,048,576 | 6 GiB | about 14.5 GiB | about 3.5 GiB, plus 11.25 GiB pinned host RAM |

    Case c uses Imperium's launch line (`-ngl 999 -c 1048576`) and must also log the tiering. It needs enough free RAM; on a machine without it, the test checks the planner's clear refusal instead.
  - The token baseline at 1M is not tested under 2 GiB; Stage 1's planner unit tests cover that refusal.
  - Peak VRAM, read from `cuMemGetInfo` in the test, stays within budget minus target.
- **Status:** pending
- **Outcome:**
- **Handoff notes:** after release, notify the research and Imperium sessions (harness behaviour change: auto-tiering at long context).

## Stage 6: Streamed backbone layers (`-ngl` below depth)
- **Objective:**
  - Streamed layers keep their weights (dense or quantised codes) in `HostBuffer`s.
  - In decode, the fused GEMV and quant GEMV kernels read those weights zero-copy.
  - In prefill, each layer is copied into a double-buffered device slab ahead of its GEMMs.
  - The planner's step 4 is enabled. Only backbone layers are streamed; the encoder, decoder, draft and head stay on the device.
- **Depends on:** stage 5; the fused GEMV work landed; stage 0b, for the NCM order and estimates.
- **Owned paths:** `crates/ncm-model/src/`:
  - `quant/linear.rs` (`QLinear` storage tier), `kernels/{gemv.rs, quant.rs}`
  - `layers.rs` (`Stack` placement), `delta.rs` (projections only), `weights.rs`/`quant/convert.rs` (load into host buffers)
  - `ncm.rs`, `tok.rs`, `placement.rs`
- **Acceptance checks:**
  - Streamed and resident layers give identical outputs, dense and q4, eager and graphed: `cargo test -p ncm-model --features cuda --test offload streamed_layers`.
  - `-ngl 8` speed is recorded against the estimate: `ncm-bench --model ncm --design v2 --quant q4 --n-gpu-layers 8`.
- **Status:** pending
- **Outcome:**
- **Handoff notes:**
  - Mostly for larger future models; the gain at 250M is small.
  - Under the hybrid NCM, the backbone runs once per 32–64-token block, so streaming costs 8–16 times less per token than at k = 4. Streaming 8 bf16 layers would cost about 0.08–0.16 ms per token *est.* This stage may move ahead of Stage 4 for the NCM once Stage 0 re-estimates.
  - If zero-copy GEMV measures under about 60% of pinned memcpy bandwidth, stage each layer through the slab in decode too, inside the graph as a captured memcpy from pinned memory.

## Stage 7 (gated): Hot-block reuse
- **Objective:** keep the previous step's picked blocks in a device slab and fetch only the newly picked ones, cutting decode PCIe traffic by the measured reuse rate.
- **Depends on:** stage 3, and a trained checkpoint whose measured selection churn shows at least 50% step-to-step block reuse. Measure it with a one-off counter in `ret_topk_kernel` behind a test feature.
- **Owned paths:** `crates/ncm-model/src/{kernels/retrieval.rs, retrieval.rs}`
- **Acceptance checks:**
  - Parity with Stage 3: `cargo test -p ncm-model --features cuda --test offload decode`.
  - Decode PCIe bytes per concept, from an Nsight or kernel counter, drop by at least the reuse rate.
- **Status:** pending
- **Outcome:**
- **Handoff notes:** skip if reuse is low. Then revisit quantising the latent cache instead.
