# Plan: NCM engine backend interface, with NPUs as its first new backend

Revision: 2026-11-26, written against `9613f29` (feat/ncm-varlen-batch tip). This is a plan only: nothing in it is implemented, built or measured. Every number marked *est.* is an estimate. External facts carry their source; those marked *unverified* were not confirmed in this session.

**Goals** (each one passes or fails):

1. One backend-agnostic engine interface (`ncm-exec`) covers every phase the runtime uses:
   - load;
   - prefill chunk;
   - decode step (token baseline, k = 4 NCM);
   - hybrid block pass and fill pass;
   - snapshot, restore and reset;
   - footprint and capabilities.

   `ncm-server`'s sequential engine runs every request through it. Check: the existing runtime tests pass unchanged on the Burn backend: `cargo test -p ncm-server` (CPU, CI) and `cargo test -p ncm-server --features cuda` (local GPU gate).
2. The trait has no Burn types in its signatures, so a non-Burn backend compiles against it. Check: `cargo tree -p ncm-exec` shows no `burn*` dependency.
3. The exported ONNX graphs match the Burn reference on ONNX Runtime's CPU execution provider (EP), for every phase of the token baseline v2 and the hybrid. Bounds: f32 max |Δlogit| ≤ 1e-4, identical greedy tokens over 64 steps, on tiny random models in CI and the 250M configs locally. Check: `cargo test -p ncm-ort --test parity`.
4. `ncm-server --device ort-cpu` serves the OpenAI-compatible API end to end, through the same e2e tests as the Burn backend. Check: `cargo test -p ncm-server --features ort --test e2e -- --device ort-cpu`.
5. On the first NPU target, `ncm-server --device npu:<target>` passes the same e2e tests and logs a speed run. The target is the user's hardware (open question 1). Check: the hardware-harness JSON from `ncm-bench --device npu:<target> --model hybrid --json <path>` reports both parity and speed.
6. CI stays green without an NPU: `scripts/check.sh`, with the ORT CPU EP parity tests added.

**Scope.** All of these, for the token baseline v2 and the NCM hybrid (Princeps):
- the interface crate;
- a Burn adapter for today's CPU and CUDA paths (and later wgpu, which arrives as a Burn device);
- an ONNX exporter;
- an ONNX Runtime backend, run on the CPU EP for parity and on one NPU EP first;
- `ncm-server`/Imperium wiring;
- footprint reporting.

The k = 4 NCM gets the interface through the Burn adapter. Its NPU export is deferred, since the hybrid replaces it as the product NCM.

**Non-goals.**
- Apple Neural Engine. Core ML has no maintained ONNX import, so the ANE would need a separate coremltools path, and the user is on Windows. Revisit if macOS becomes a target.
- Training on an NPU.
- DirectML. Microsoft marks it "legacy" in Windows ML ([accelerate AI models](https://learn.microsoft.com/windows/ai/new-windows-ml/accelerate-ai-models)).
- NPU continuous batching in v1. The NPU backend is batch 1, and the batcher falls back to sequential.
- Splitting one model across NPU and GPU.
- CUDA graphs or device sampling on the NPU.
- An NPU path for the fallback attention design.

**Assumptions.**
- The runtime on `feat/ncm-varlen-batch` merges first: the k = 4 NCM, the hybrid in `src/hybrid`, the token baseline, fused CUDA kernels and graphs, batching, quantisation.
- Burn has no ONNX *export*. `burn-import` / `burn-onnx` go ONNX → Burn only (*unverified for 0.22.1*; Stage 0 confirms). An exporter is therefore ours to write.
- NPUs share system DRAM with the CPU. The relevant bandwidths are the laptop's LPDDR5X: about 120–136 GB/s peak on Lunar Lake, Strix Point and Snapdragon X (vendor specifications, *unverified*), and about 60 GB/s effective for NPU weight streaming (*est.*).
- Testing is paused, and the user has the GPU right now. Every stage that needs a GPU or an NPU runs later under the local gates. CI covers the CPU EP.

**Relevant paths** (under `models/ncm/`):

| Area | Paths |
|---|---|
| Runtime engine | `runtime/src/{engine.rs (Backend trait, Core, Run, make_device, DeviceChoice), batch.rs (Batcher, Model enum), cli.rs, app.rs, server.rs, sampling.rs, constrain.rs}` |
| Model | `crates/ncm-model/src/{tok.rs, ncm.rs, cache.rs, delta.rs, retrieval.rs, layers.rs, quant/*, weights.rs, sample.rs}` |
| Hybrid | `crates/ncm-model/src/hybrid/{fast.rs (FastSession, BlockRun, block_pass, ar_pass), model.rs (HybridLm), memory.rs, vq.rs, decode.rs}` |
| Bench | `crates/ncm-bench` |
| New crates | `crates/ncm-exec` (interface), `crates/ncm-export` (ONNX emitter), `crates/ncm-ort` (ONNX Runtime backend) |

**Related docs.** Current code wins over every doc. This plan wins over the research docs for runtime architecture. On footprint and placement types, the CPU offload plan and this one must agree (Stage 1 settles where they live).
- `docs/attention.md`
- `docs/quantisation.md`
- `runtime/README.md` (the llama-server contract)
- `docs/plans/cpu-offload.md` (branch `docs/ncm-cpu-offload`)
- `Imperium-Research/docs/next-concept-modelling/architecture.md` (the hybrid; read-only)

## Decision record

### 1. The interface sits under the HTTP `Backend` trait, not in place of it

`runtime/src/engine.rs` already has `trait Backend` (`generate`, `check_prompt`, `slots`), the seam the HTTP layer and its test doubles use. Below it, `Core` and `Batcher` call concrete Burn models: `enum Model { Tok, Ncm, Hybrid }`, `LayerCache`, `FastSession`, and the graph types.

The new interface lives in `ncm-exec`, a crate with no Burn dependency. It sits between `Core` and the models:

```rust
pub trait ExecBackend: Send {
    type Session: ExecSession;
    fn capabilities(&self) -> Capabilities;        // phases, buckets, max batch, dtypes, device sampling, graphs
    fn footprint(&self, ctx: usize, slots: usize) -> Footprint; // per memory tier, per part
    fn new_session(&self, ctx: usize) -> Result<Self::Session, ExecError>;
}
pub trait ExecSession: Send {
    fn prefill(&mut self, tokens: &[i32]) -> Result<StepOut, ExecError>;        // backend chunks to its buckets
    fn decode_step(&mut self, feed: &[i32], pick: &PickSpec) -> Result<StepOut, ExecError>; // token / k=4 NCM
    fn block_pass(&mut self, input: &BlockPassIn, mode: PassMode) -> Result<BlockPassOut, ExecError>; // hybrid
    fn fill_pass(&mut self, input: &FillIn) -> Result<FillOut, ExecError>;      // hybrid autoregressive fill
    fn snapshot(&self) -> Snapshot;                // opaque, backend-owned
    fn restore(&mut self, s: &Snapshot);
    fn reset(&mut self);
}
```

Supporting types:

| Type | What it carries |
|---|---|
| `StepOut` | Either `Picked(Vec<u32>)` (the backend sampled on its device: CUDA graphs) or `Logits(HostLogits)` (the engine samples with `sampling.rs`) |
| `PickSpec` | Greedy or seeded sampling, and whether host logits are required (grammar-constrained requests) |
| `BlockPassIn` / `BlockPassOut` | The host-visible state of today's `BlockRun`: known prefix, positions, mode. The output is the `[first, any, ids (L), draft (L)]` read-back that `fast.rs` already returns |

Batched passes become a capability (`max_batch`) plus `ExecBackend::step_batch(&mut [&mut Session], …)`. `Batcher` uses that only when `max_batch > 1`; otherwise the engine runs sequentially.

Why this shape:
- The block state machine (`BlockRun`) and grammar stay host-side and shared, so every backend gets acceptance and tool calls for free.
- Device-resident fast paths (CUDA graphs, device sampling) stay behind `Picked`, so the CUDA path loses nothing.
- Snapshots stay opaque. Prefix reuse keeps working whatever the cache layout, including the offload plan's tiers.

Rejected:
- **Making Burn's `Backend` the seam**, that is, writing an NPU Burn backend. ORT/OpenVINO execute whole graphs, not Burn's op-by-op dispatch, and there is no Burn NPU backend.
- **Replacing the HTTP `Backend` trait.** It is the right boundary for HTTP tests, and it stays.
- **One trait per model.** Three near-identical traits; phases are the natural unit.

Where existing backends land:
- `BurnBackend` (in `ncm-model`, feature `exec`) wraps today's code for `Device::flex()` and CUDA. The planned wgpu backend is just another Burn device, so it implements the interface with no new adapter.
- The offload plan's `ColdStore`/tiering stays inside `BurnBackend`. Only its `Footprint` crosses the interface.

### 2. What NPU stacks offer (researched 2026-11-26)

| Target | Runtime path | Shapes | Precision | State | Notes |
|---|---|---|---|---|---|
| Intel NPU (Core Ultra) | OpenVINO, directly or as the ORT OpenVINO EP | Static. The GenAI NPU pipeline fixes `MAX_PROMPT_LEN` / `MIN_RESPONSE_LEN`. Chunked dynamic prompts since 2025.3 via `NPUW_LLM_PREFILL_CHUNK_SIZE` ([OpenVINO GenAI on NPU](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html)) | Symmetric INT4 weights for GenAI models (same source); KV u8/u4 ([OVMS LLM reference](https://docs.openvino.ai/2025/model-server/ovms_docs_llm_reference.html)) | Stateful servables only, in OVMS (same OVMS source). One paper reports the NPU compiler rejecting ReadValue/Assign outside that path ([arXiv 2608.19147](https://arxiv.org/pdf/2608.19147)) | Compile caching through `CACHE_DIR` |
| AMD XDNA (Ryzen AI 300) | ORT + Vitis AI EP; OGA for LLMs, NPU-only or hybrid NPU+iGPU ([Ryzen AI OGA flow](https://ryzenai.docs.amd.com/en/latest/hybrid_oga.html)) | Static (*unverified* for custom models) | INT4 weights with BF16 activations for LLMs ([AMD GPT-OSS article](https://www.amd.com/en/developer/resources/technical-articles/2026/accelerating-gpt-oss-20b-on-amd-ryzen-ai-npus.html)); AWQ g128 int4 models | OGA-managed | Hybrid models are not compatible across releases (same OGA docs) |
| Qualcomm Hexagon (Snapdragon X) | ORT QNN EP; Genie for LLMs | Static only: "does not support models with dynamic shapes" ([QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)) | int8/int16 QDQ; fp16 via `enable_htp_fp16_precision` (same QNN EP doc) | Explicit I/O. Genie bundles fix context and KV size ([Qualcomm AI Hub](https://dragonwingdocs.qualcomm.com/Key-Documents/AI-Developer-Workflow/topic/genai-prepare-ai-hub)); prefill and AR-1 graphs share one weight set (third-party example) | Context binaries are ahead-of-time compiled per QAIRT version. Large models are split into parts (AI Hub) |
| Apple ANE | Core ML | Enumerated or fixed shapes | fp16; int8/int4 palettised | `MLState` for in-place KV ([HF Mistral Core ML](https://huggingface.co/blog/mistral-coreml)) | Non-goal (above) |
| Windows-wide | Windows ML: a bundled ORT plus EPs (OpenVINO, QNN, VitisAI) downloaded and updated through `ExecutionProviderCatalog` on Windows 11 24H2+ ([supported EPs](https://learn.microsoft.com/windows/ai/new-windows-ml/supported-execution-providers)) | Per EP | Per EP | Per EP | Apps may also bundle EPs themselves |

What this means for the design:
1. Every NPU path wants **static shapes per graph**, ahead-of-time compilation cached on disk, and quantised weights (int8, or symmetric int4).
2. **ONNX plus ONNX Runtime EPs** reaches Intel, AMD and Qualcomm through one exporter and one runtime crate.
3. **Recurrent and KV state is safest as explicit graph inputs and outputs**, with fixed shapes and host-owned buffers. Stateful variables are vendor-specific. Our caches are small per step, and the NPUs share DRAM anyway.

Intel through OpenVINO directly is the fallback if the ORT OpenVINO EP lags. It is reachable from the same ONNX files.

### 3. How the model maps onto an NPU

**Good fit** (dense, static):
- linear layers and MLPs;
- RMSNorm and RoPE (precomputed tables passed in);
- the encoder's sliding attention over a fixed ring;
- the hybrid generator's K levels;
- codebook logits and the codebook lookup (as `Gather`, or as a one-hot × codebook matmul if `Gather` falls back to the CPU);
- the token decoder's check pass;
- retrieval attention over the local ring (dense, fixed length);
- the indexer scores (a GEMV, as a fixed-bucket matmul with a validity mask).

**DeltaNet:**
- *Decode step:* standard ops (MatMul, Mul, Add, Sigmoid). The state `[heads, 128, 128]` f32 is about 0.5 MiB per layer, as explicit I/O.
- *Chunked prefill:* the WY form needs `(I − A)⁻¹` for a strictly lower-triangular 64×64 `A`. ONNX has no triangular solve, so the inverse is computed exactly as `(I + A)(I + A²)(I + A⁴)…(I + A³²)`: six matmuls, exact because `A` is nilpotent. Parity is checked in fp16 (Stage 2).

**Poor fit, so on the CPU:**

| Step | Why it moves | Cost on the CPU |
|---|---|---|
| Retrieval top-k block selection | Data-dependent | The CPU picks the blocks and gathers them into a fixed `[topk × block]` input for the NPU's attention |
| Sampling | Needs host-side randomness and settings | Negligible |
| Grammar constraints | Host automaton | Unchanged from today |
| The hybrid's acceptance scan | Reads draft and check ids | The host already reads back 2 + 2L ints per pass |

ONNX Runtime's own partitioner also assigns nodes the EP rejects to the CPU EP. The graph is designed so this never happens inside hot loops, and Stage 4 checks the partition report.

Context thresholds for the hybrid (L = 32, `ret_local` = 1,024 block positions):
- Up to 32k tokens, retrieval is dense over the local window. No top-k is needed and the whole backbone step is one static graph. That covers Imperium's common contexts.
- Beyond 32k, each retrieval layer runs in three steps:
  1. the indexer runs on the NPU;
  2. top-k and the gather run on the CPU: 7 layers × 2,048 × 1,152 B = 16.5 MB copied per 32-token block (*est.*, trivial);
  3. attention runs on the NPU.

  That costs about 2 invocations per retrieval layer per block. ORT's `TopK` may run in-graph on some EPs; Stage 4 measures which is faster.

**Why the hybrid suits NPUs better than token-by-token decode.** Per 32-token block, the hybrid makes about 2–4 graph invocations: the backbone step, one block pass with all K levels unrolled, and any re-decode or fill passes. Their matmuls have 32–99 rows. Token-by-token decode makes one invocation per token, with 1-row GEMVs.

NPUs have a fixed dispatch cost per invocation (*unknown*, likely 0.1–1 ms; measured in Stage 0 for the CPU EP and Stage 4 on hardware) and favour many rows. So the hybrid amortises both.

The caveat: fill passes. A rejected draft position costs an autoregressive fill pass, so a low acceptance rate turns the hybrid back into per-token invocations. Speed depends on the trained model's acceptance rate.

**Few graphs.** Each static graph is compiled separately and may duplicate weights in memory (QNN shares weights between the AR-N and AR-1 graphs in Genie; ORT session-level sharing is *unverified* per EP). So the graph set is kept small:

| Graph | Shape |
|---|---|
| Prefill | Buckets of 1,024 and 128 tokens (the 128 bucket for short tails) |
| Hybrid backbone step | 1 block position |
| Block pass | All K levels padded to one shape, so one graph serves all levels |
| Fill pass | One graph |
| Token baseline decode | 1 token |

Padding the levels to 32 rows costs about 2.6× the compute of the true 99 rows (8 levels × 32 = 256 rows). That is acceptable while bandwidth-bound; Stage 5 revisits it with two level buckets if compute-bound.

### 4. Expected speed and memory (*est.*, unmeasured; 250M configs)

Ceilings below assume weights streamed from DRAM at about 60 GB/s effective. The "planning guess" column adds dispatch overhead and acceptance below 1.

| Path | Weight bytes per step | Ceiling | Planning guess |
|---|---|---|---|
| Token baseline, int4 decode | about 0.13 GB per token | ≤ about 460 tok/s | 150–350 tok/s |
| Hybrid, int8 | about 0.38 GB per 32-token block (backbone 0.2 GB + generator 8 levels × 20 MB + token decoder 16 MB) | about 5,000 tok/s | 500–2,000 tok/s |
| Hybrid, int4 | about 0.2 GB per block | about 9,700 tok/s | 500–2,000 tok/s |
| Prefill (compute-bound) | 0.5 GFLOP per token | about 20k tok/s at 10 effective TOPS | not estimated |

The NPUs are rated at 45–50 TOPS int8 (vendor; *unverified*), against the 10 effective TOPS assumed above.

Comparison: today's CUDA NCM v2 runs at about 1,400 tok/s on an RTX 5080. The NPU's value is running Princeps without a discrete GPU, or alongside one that is busy, at laptop power. It will not beat the 5080.

Memory: everything is system RAM.
- Weights: 0.13–0.25 GB per graph set (×N if a target duplicates weights per graph).
- Caches: the hybrid at 1M tokens is about 0.2–0.3 GiB; the token baseline at 1M is 12.5 GiB (it needs RAM, not tiering).
- The NPU compile cache on disk: *unknown*, likely tens to hundreds of MB.

### 5. Exporter: hand-built ONNX from Rust, not a PyTorch mirror

`ncm-export` writes ONNX protobuf (opset 21, which has int4 types) through a small graph builder. There is one builder per layer kind, mirroring the Burn reference code:
- `Linear` / `QLinear`
- `RmsNorm`
- `Rope`
- GQA attention over a ring
- DeltaNet step and chunk
- MLA absorbed attention
- indexer
- VQ level
- copy head

Weights come from the safetensors file as ONNX external data. Quantised weights are exported as:
- QDQ int8: `DequantizeLinear` with per-channel scales;
- int4: opset-21 int4 `DequantizeLinear` with block size = `quant_group`, or ORT's `MatMulNBits`, whichever the target EP accepts (*unverified* per EP; Stage 5).

Rejected:
- **A PyTorch re-implementation plus `torch.onnx.export`.** It is a second model implementation in another language, which drifts from the Burn reference.
- **Tracing Burn.** It has no exporter (Stage 0 confirms).

Parity tests against the Burn reference (goal 3) catch drift in the builders.

### 6. Server, Imperium and other plans

**ncm-server flags:**
- `--device` adds `ort-cpu` (parity and CI) and `npu[:intel|amd|qualcomm|auto]`.
- `-ngl` is ignored on NPU devices, with a note on stderr.
- `--npu-cache <dir>` sets the compile cache; the default is beside the model.

**Readiness:**
- `/health` stays 503 until the graphs are compiled. A first compile may take minutes (*unverified*); later loads hit the cache.
- `/props` `ncm.device` reports the EP and its version.

**Footprint:** `--print-footprint` reports system-memory weights per graph set, caches and the compile cache, from `ncm_exec::Footprint`.

**Imperium:** its `runtime.ncm.device` setting maps to `--device` in the extra args. Stage 6 notifies the Imperium and research sessions (it is a harness release).

**CPU offload plan:** `Footprint` (and the placement types it reports) move into `ncm-exec`, so both plans share one definition. The offload plan's Stage 1 should create them there. On NPUs there is no VRAM tier, so the offload plan's tiering applies only to `BurnBackend`.

**wgpu:** arrives as a Burn device through `BurnBackend`, so the interface needs nothing extra. Its capabilities (no graphs, no device sampling) are just a different `Capabilities` value.

### 7. Dependencies, licences, size

| Dependency | Licence | Notes |
|---|---|---|
| `ort` 2.0 (currently 2.0.0-rc.12; [docs.rs](https://docs.rs/crate/ort/latest)) | MIT / Apache-2.0 (*unverified*) | `load-dynamic` loads `onnxruntime.dll` at run time, from `ORT_DYLIB_PATH` or an explicit path, so `ncm-server` links nothing and NPU support is an optional download. Pyke's prebuilt binaries cover only a few EPs; NPU EPs come from Windows ML or vendor packages. It sits behind the `ort` feature of `ncm-server` |
| ONNX Runtime | MIT | About 10–20 MB for the CPU build (*est.*) |
| OpenVINO (runtime + NPU plugin) | Apache-2.0 | Hundreds of MB (*est.*); via Windows ML or Intel's package |
| QNN / QAIRT | Qualcomm proprietary | Distributed through Windows ML; we never redistribute it |
| Vitis AI EP / Ryzen AI | AMD terms | Distributed through Windows ML |
| ONNX protobuf (`ncm-export`) | Apache-2.0 (`onnx.proto`) | Generated with `prost`; no runtime dependency |

**Windows ML or bundled EPs?** Windows ML keeps vendor libraries out of our distribution and keeps them updated. Whether `ort` can attach to Windows ML's ORT and register catalog EPs is *unverified* (Stage 4 spike). If it cannot, use vendor packages located at run time.

**CI:** Linux, CPU EP only:
- `ncm-ort` parity tests, against an ONNX Runtime release pinned by version and checksum and fetched by the test setup;
- the e2e tests with `--device ort-cpu`.

**Hardware:** a self-hosted harness on the target device runs the same parity and e2e suite plus `ncm-bench --device npu:<target> --json`. That fits the user's no-manual-checks rule.

## Stage 0: Spike: one decode step to ONNX, run on the ORT CPU EP
- **Objective:**
  - Confirm Burn 0.22.1 has no ONNX export.
  - Hand-build the ONNX graph of one token-baseline v2 decode step at the tiny config (DeltaNet step, dense local retrieval, MLP, logits), with the caches as explicit inputs and outputs, using a throwaway builder.
  - Run it on ORT's CPU EP through `ort` and compare against the Burn reference.
  - Measure the overhead of one session run.
  - Also check whether `ort` loads a given `onnxruntime.dll` by path (`load-dynamic`).
- **Depends on:** none. Needs no GPU, so it can run while the user holds the GPU, once testing resumes.
- **Owned paths:** a scratch spike branch, not merged. Results are recorded here.
- **Acceptance checks:**
  - Parity holds: f32 max |Δlogit| ≤ 1e-4, greedy tokens identical over 64 steps.
  - The per-run overhead and graph size are recorded in Outcome.
- **Status:** pending
- **Outcome:**
- **Handoff notes:** if Burn gained an exporter, re-decide decision 5. If the overhead per run is above about 1 ms on the CPU EP, flag the invocation-count budget for Stage 4.

## Stage 1: The `ncm-exec` interface and the Burn adapter
- **Objective:**
  - Create `crates/ncm-exec`: the traits and types of decision 1, and `Footprint`/`Capabilities`, agreed with the CPU offload plan.
  - Implement `BurnBackend` for CPU and CUDA, wrapping `TokenLm`, `NcmLm` and `HybridLm`/`FastSession` without changing them.
  - Route `Core`'s sequential paths in `runtime/src/engine.rs` through the trait.
  - `Batcher` keeps using Burn directly, gated on `capabilities().max_batch > 1`.
- **Depends on:** stage 0 (the interface shape is unaffected, but the spike's findings on state I/O feed `ExecSession`); `feat/ncm-varlen-batch` merged.
- **Owned paths:**
  - `crates/ncm-exec/**`
  - `crates/ncm-model/src/exec.rs` (adapter, feature `exec`)
  - `runtime/src/{engine.rs, app.rs, batch.rs (gate only)}`
  - the workspace `Cargo.toml`
- **Acceptance checks:**
  - Runtime tests unchanged and passing on CPU (`cargo test -p ncm-server`) and CUDA (`cargo test -p ncm-server --features cuda`, local gate).
  - Speed unchanged within ±2%: `ncm-bench --model both --design v2 --context 2048 131072`, and the hybrid run.
  - `cargo tree -p ncm-exec` has no `burn`.
- **Status:** pending
- **Outcome:**
- **Handoff notes:**
  - This is the riskiest refactor, because `engine.rs` has many paths (eager, graphed, speculative, hybrid).
  - Move one path at a time, each with its existing test.
  - Snapshots are the opaque `Snapshot(Box<dyn Any + Send>)`.

## Stage 2: ONNX exporter
- **Objective:**
  - `crates/ncm-export`: an opset-21 protobuf writer and the layer builders of decision 5.
  - Phase graphs:
    - token baseline: prefill buckets, decode step;
    - hybrid: encoder step, backbone step (dense local), block pass with all levels unrolled, fill pass;
    - retrieval beyond the local window: split indexer and attention graphs.
  - DeltaNet chunked prefill uses the nilpotent product inverse.
  - Quantised export: QDQ int8, int4 `DequantizeLinear` or `MatMulNBits`.
  - Command: `ncm-export --model <weights> --out <dir> --buckets 128,1024 [--quant q8|q4]`.
- **Depends on:** stage 0.
- **Owned paths:** `crates/ncm-export/**`. Read-only use of `crates/ncm-model/src/{config.rs, weights.rs, quant/format.rs}`.
- **Acceptance checks:**
  - The graphs load in the ORT CPU EP.
  - Per-builder unit parity against the Burn reference: `cargo test -p ncm-export`.
  - The DeltaNet chunk inverse matches the reference in f32 and fp16 within tolerance.
- **Status:** pending
- **Outcome:**
- **Handoff notes:** keep the graphs free of dynamic shapes, `Loop`/`If`, ScatterND on caches and stateful variables. The host writes new cache entries into the ring.

## Stage 3: ONNX Runtime backend on the CPU EP (`--device ort-cpu`)
- **Objective:**
  - `crates/ncm-ort`: `OrtBackend` implementing `ExecBackend` over the exported graphs, with:
    - host-owned fixed cache buffers bound as inputs and outputs (IO binding);
    - CPU top-k and gather beyond the local window;
    - prefill bucket scheduling;
    - host acceptance for the hybrid through the shared `BlockRun` logic;
    - an opaque snapshot (a copy of the cache buffers or their lengths).
  - `ncm-server --features ort --device ort-cpu`.
  - Footprint and `/props` reporting.
- **Depends on:** stages 1 and 2.
- **Owned paths:**
  - `crates/ncm-ort/**`
  - `runtime/src/{cli.rs, engine.rs (device selection), app.rs, server.rs (/props)}`
  - `runtime/Cargo.toml`, `scripts/check.sh` (the CI ORT fetch)
  - the `runtime/README.md` device section
- **Acceptance checks:**
  - Goal 3 parity for every phase, both models: `cargo test -p ncm-ort --test parity`.
  - Goal 4 e2e: `cargo test -p ncm-server --features ort --test e2e -- --device ort-cpu`.
  - Prefix reuse across requests matches the Burn backend's cached-token counts.
  - CI green: `scripts/check.sh`.
- **Status:** pending
- **Outcome:**
- **Handoff notes:** adding `ort` and the CI ORT download needs the user's approval (open question 2).

## Stage 4: First NPU target
- **Objective:**
  - Bring up `--device npu:<target>` on the user's hardware, via Windows ML's EP catalog or the vendor package, whichever the `ort` attach spike allows.
  - Set EP options: compile cache (`CACHE_DIR` / `ep.context_enable`), precision.
  - Report the partition (no CPU fallback inside hot graphs).
  - Hold `/health` at 503 during compilation.
  - Measure dispatch overhead per invocation, and in-graph `TopK` against the CPU split.
- **Depends on:** stage 3; target hardware available (open question 1).
- **Owned paths:**
  - `crates/ncm-ort/src/ep/*`
  - `runtime/src/{cli.rs, server.rs (readiness)}`
  - a hardware-harness script under `scripts/`
- **Acceptance checks:**
  - Goal 5 on the target: parity within the quantised tolerance, e2e passing, speed JSON recorded.
  - The partition report shows every hot graph fully on the NPU.
- **Status:** pending
- **Outcome:**
- **Handoff notes:**
  - The default first target is Intel (OpenVINO is the best-documented static LLM path, with a direct-OpenVINO fallback), unless the user owns a different NPU.
  - Record measured numbers against decision 4 and correct the estimates.

## Stage 5: NPU quantisation and graph-set tuning
- **Objective:**
  - Per target, pick the weight format: symmetric int4, QDQ int8, or `MatMulNBits`.
  - Pick the activation precision: fp16, bf16 (XDNA), or int16 (HTP).
  - Choose the level bucketing (one padded graph or two buckets) and the prefill buckets from Stage 4's measurements.
  - Enable weight sharing between graphs where the EP supports it.
- **Depends on:** stage 4.
- **Owned paths:**
  - `crates/ncm-export/src/quant.rs`
  - `crates/ncm-ort/src/ep/*`
  - `docs/quantisation.md` (the NPU section)
- **Acceptance checks:**
  - Quantised parity on the target: greedy tokens match the Burn quantised reference on the tiny model and on a 64-step 250M sample.
  - Weights in memory are at most 1.5× one copy.
  - The `ncm-bench` speed JSON improves on Stage 4.
- **Status:** pending
- **Outcome:**
- **Handoff notes:** acceptance-rate data from a trained hybrid decides how much the fill-pass invocation count matters.

## Stage 6: More targets, Imperium release
- **Objective:**
  - Add the remaining Windows NPU targets (QNN, VitisAI) through the same `ncm-ort` EP layer.
  - Map Imperium's `runtime.ncm.device` to `--device`.
  - Document everything in `runtime/README.md`.
  - Notify the Imperium and research sessions.
- **Depends on:** stage 5; hardware access for each target.
- **Owned paths:**
  - `crates/ncm-ort/src/ep/*`
  - `runtime/README.md`
  - an Imperium PR for the `runtime.ncm.device` mapping, if needed (owned by the Imperium implementation session)
- **Acceptance checks:** goal 5 on each added target, from the harness JSON.
- **Status:** pending
- **Outcome:**
- **Handoff notes:** NPU batching (`max_batch > 1`) and Core ML are later, separate decisions.
