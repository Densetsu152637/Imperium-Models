# Plan: NCM engine backend interface, with NPU backends as its first new implementation

Revision: 2026-11-26, written against `9613f29` (feat/ncm-varlen-batch tip). Last updated 2026-11-26, rev 3: re-review of `d238768` (rev 2 was the plan-gate review of `1fe0b52` and the user's answers).

Nothing here is implemented, built or measured. Every number marked *est.* is an estimate. External facts carry their source; those marked *unverified* were not confirmed in this session.

**User decisions (2026-11-26):**
- Targets: as many NPUs as possible. All three are planned (Intel OpenVINO EP, Qualcomm QNN EP, AMD VitisAI EP), ordered by risk and value.
- The `ort` crate and a pinned ONNX Runtime CPU build in CI: approved.
- Packaging: Windows ML's downloadable EPs.
- Core ML / ANE: not needed.
- A batch-1 NPU backend in v1: accepted.

> **Hardware: no NPU is available.** The development machine is an Intel i7-13700KF (no NPU) on Windows build 26300.
> - Every on-NPU step is **blocked on hardware access**: Stage 0b, Stage 4, Stage 5, Stages 6a–6b, goal 5, and the NPU-specific parts of Stages 2–3.
> - The generic work runs on the CPU and in CI: the interface, the exporter, the ORT CPU backend.
> - Ways to get hardware are open question 1.

**Goals** (each one passes or fails):

1. One backend-agnostic engine interface (`ncm-exec`) covers every phase the runtime uses: load, prefill, decode step (token baseline, k = 4 NCM), the hybrid's request and block cycle, snapshot/restore/reset, footprint and capabilities. `ncm-server`'s sequential engine runs every request through it.
   - Check: the runtime tests pass unchanged on the Burn backend, `cargo test -p ncm-server` (CPU, CI) and `cargo test -p ncm-server --features cuda` (local GPU gate).
   - Check: speed stays within ±2% under the Stage 1b procedure.
2. `ncm-exec` does not depend on Burn. The dependency direction is `ncm-model → ncm-exec`, `ncm-ort → ncm-exec`, and the runtime depends on all three.
   - Check: `! cargo tree -p ncm-exec -e normal | grep -q burn` succeeds.
3. The exported ONNX graphs match the Burn reference on ORT's CPU EP for every phase of the token baseline v2 and the hybrid:
   - f32 max |Δlogit| ≤ 1e-4;
   - identical greedy tokens over 64 steps;
   - DeltaNet state max |Δ| ≤ 1e-3 after 8,192 recurrent positions;
   - tiny random models in CI, 250M configs locally.

   Check: `cargo test -p ncm-ort --test parity`.
4. `ncm-server --device ort-cpu` serves the OpenAI-compatible API end to end, passing the same e2e tests as the Burn backend.
   - Check: `NCM_E2E_DEVICE=ort-cpu cargo test -p ncm-server --features ort --test e2e`.
5. On each NPU target (hardware-gated), `ncm-server --device npu:<target>` passes the e2e tests, and the harness writes the speed and parity record of Stage 4.
   - Check: `ncm-bench --device npu:<target> --model hybrid --json <path>` validates against the Stage 4 schema.
6. CI stays green without an NPU, with the ORT CPU EP parity and e2e tests added. Check: `scripts/check.sh`.

**Scope.** All of these, for the token baseline v2 and the NCM hybrid (Princeps). The k = 4 NCM gets the interface through the Burn adapter only.
- the interface crate;
- a Burn adapter for CPU and CUDA (wgpu later, as one more Burn device);
- an ONNX exporter;
- an ONNX Runtime backend (CPU EP for parity; NPU EPs through Windows ML);
- `ncm-server` / `ncm-bench` / Imperium wiring;
- shared footprint types with the CPU offload plan.

**Non-goals.**
- Core ML / ANE (user decision: not needed).
- DirectML. Windows ML lists it as "legacy" ([accelerate AI models](https://learn.microsoft.com/windows/ai/new-windows-ml/accelerate-ai-models)).
- Training on an NPU.
- Batched NPU decode in v1, and any general batch API in the interface (decision 1).
- Splitting one model across NPU and GPU.
- An NPU path for the fallback attention design, or for the k = 4 NCM.

**Assumptions.**
- `feat/ncm-varlen-batch` merges first.
- Burn 0.22.1 has no ONNX export: `burn-import` / `burn-onnx` go ONNX → Burn only (*unverified*; Stage 0 confirms).
- NPUs share system DRAM. Effective NPU weight streaming is about 60 GB/s (*est.*), against laptop LPDDR5X peaks of about 120–136 GB/s (vendor specifications, *unverified*).
- Testing is paused and the user holds the GPU. GPU stages run later under the local gates; NPU stages need hardware (above).

**Relevant paths** (under `models/ncm/`):

| Area | Paths |
|---|---|
| Runtime | `runtime/src/{engine.rs (HTTP Backend trait, Core, Run, make_device, DeviceChoice), batch.rs (Batcher, Model enum), cli.rs, app.rs, server.rs, sampling.rs, constrain.rs}`, `runtime/tests/{e2e.rs, engine.rs, hybrid.rs, batching.rs}`, `runtime/Cargo.toml` |
| Model | `crates/ncm-model/src/{tok.rs, ncm.rs, cache.rs, delta.rs (chunk inverse, lines 143–160), retrieval.rs, layers.rs, quant/*, weights.rs, sample.rs, config.rs}` |
| Hybrid | `crates/ncm-model/src/hybrid/{fast.rs (FastSession: begin_request, begin_block, prepare_pass/apply_pass, prepare_fill/apply_fill, end_block, DrawSettings, failure), decode.rs, model.rs, memory.rs, vq.rs}` |
| Bench | `crates/ncm-bench/src/lib.rs` |
| New crates | `crates/ncm-exec` (interface, footprint types, the hybrid's host block logic), `crates/ncm-export` (ONNX emitter), `crates/ncm-ort` (ONNX Runtime backend) |

**Related docs.** Current code wins over every doc, and this plan wins over the research docs for runtime architecture. For footprint types and the order of the two plans' Stage 1, this plan and `docs/plans/cpu-offload.md` (branch `docs/ncm-cpu-offload`) state the same thing; if they ever disagree, this plan wins.
- `docs/attention.md`
- `docs/quantisation.md`
- `runtime/README.md`
- `docs/plans/cpu-offload.md`
- `Imperium-Research/docs/next-concept-modelling/architecture.md` (read-only)

## Decision record

### 1. The interface sits under the HTTP `Backend` trait

`runtime/src/engine.rs`'s `trait Backend` (`generate`, `check_prompt`, `slots`) stays. It is the seam for the HTTP layer and its test doubles.

The new `ncm-exec` interface sits between `Core` and the models:

```rust
pub trait ExecBackend: Send {
    type Session: ExecSession;
    fn capabilities(&self) -> Capabilities;   // phases, prefill buckets, dtypes, device sampling, graphs
    fn footprint(&self, req: &FootprintRequest) -> Footprint; // ctx, slots, tokens_per_position, placement
    fn new_session(&self, ctx: usize) -> Result<Self::Session, ExecError>;
}

pub trait ExecSession: Send {
    type BlockRun: Send;                      // opaque, backend-owned (hybrid only)
    fn begin_request(&mut self, seed: u64);
    fn prefill(&mut self, tokens: &[i32], chunk: usize) -> Result<PrefillOut, ExecError>;
    fn decode_step(&mut self, feed: &[i32], pick: &PickSpec) -> Result<StepOut, ExecError>;
    // Hybrid block cycle, mirroring FastSession:
    fn begin_block(&mut self, limit: usize, opts: &DecodeOptions) -> Self::BlockRun;
    fn phase(&self, run: &Self::BlockRun) -> BlockPhase;          // Pass | Fill | Done
    fn pass(&mut self, run: &mut Self::BlockRun, opts: &DecodeOptions, draws: &DrawSettings)
        -> Result<PassReadBack, ExecError>;                         // [first, any, ids(L), draft(L)]
    fn fill(&mut self, run: &mut Self::BlockRun, opts: &DecodeOptions, draws: &DrawSettings,
        constrain: &mut dyn FnMut(&[f32]) -> Option<i32>) -> Result<i32, ExecError>;
    fn end_block(&mut self, run: Self::BlockRun) -> BlockOut;
    fn snapshot(&self) -> Snapshot;           // opaque
    fn restore(&mut self, s: &Snapshot);
    fn reset(&mut self);
    fn failure(&self) -> Option<String>;      // a failed capture or EP error, reported in /props
}
```

**Return types.** `StepOut` is one of:
- `Picked(Vec<u32>)`: the backend sampled on its device (CUDA graphs);
- `Logits(HostLogits)`: the engine samples with `sampling.rs`.

`PickSpec` says whether host logits are required (grammar-constrained requests).

**Prefill** returns what each model's next step needs:
- token baseline: the last logits;
- k = 4 NCM: the condition;
- hybrid: nothing beyond the session state.

The cached-token count for prefix reuse is returned for every model.

**DrawSettings** is a backend-neutral type: per-row temperature, top-k, top-p and counter-based seed. The session derives it from `DecodeOptions` and `begin_request(seed)`. The CUDA backend maps it to `crate::sample::SampleSettings`; other backends use it on the host or as graph inputs (decision 4).

**The hybrid's block logic.** Today the block state machine lives inside `FastSession`: `BlockRun` plus the apply logic for acceptance, stops, mixed-block switches, constraint scans, re-decode, recheck and fill. Its *host* part works only on the read-back integers and the fixed-token list, so Stage 1b moves it into `ncm_exec::hybrid::BlockLogic`, which every backend's `BlockRun` embeds. Each backend keeps its own staging and pass execution, and `BlockRun` stays opaque to the engine.

**Batching.** The interface has no batch API in v1. `Batcher` (`runtime/src/batch.rs`) remains an explicit **Burn-only side door**: when the backend is `BurnBackend` and `--parallel` is above 1, the runtime builds the batcher over the Burn types, as today. Every other backend serves requests sequentially.

**Snapshots.**

| Backend | Snapshot | Restore |
|---|---|---|
| `BurnBackend` | Copies on the device, as today: ring KV copies, DeltaNet state clones, latent lengths | Writes into the session's existing buffers, so CUDA graph addresses stay fixed and no graph recapture is needed |
| `OrtBackend` | Copies in system RAM: cache lengths plus ring and state contents | `memcpy`s into the same IO-bound host buffers, whose addresses stay fixed |

**Dependency direction.**
- `ncm-exec` depends on nothing model-specific: no Burn, no ORT.
- `ncm-model` depends on `ncm-exec` unconditionally, for the footprint types and `BlockLogic`.
- `BurnBackend` lives in `ncm-model` (`src/exec.rs`).
- `ncm-ort` depends on `ncm-exec` only, and reads configs from the exporter's manifest.
- The runtime depends on all three; `ncm-ort` is optional, behind the `ort` feature.

**Rejected:**
- An NPU *Burn* backend. EPs run whole graphs, not Burn's op dispatch.
- Replacing the HTTP `Backend` trait.
- One trait per model.
- A batch API now. With only Burn implementing it, it would be speculative design.

### 2. Footprint types shared with the CPU offload plan

`ncm-exec` owns the types:
- `Footprint`: bytes by `Tier` × `Part`.
- `Tier`: `Device`, `PinnedHost`, `SystemRam`, `NpuCompileCache`. The last is on disk and reported, not budgeted.
- `Part`: weights, latent cache, indexer keys, rings, DeltaNet state, prefill staging, union slab, graph workspace.
- `FootprintRequest`: ctx, slots, `tokens_per_position`, dtype and quantisation, and an optional `Placement` (per-layer weights tier and per-cache tier).

The offload planner (which turns budgets into placements) stays in `ncm-model` and returns these types.

**Order of the two plans' Stage 1, and who owns `engine.rs`:**
1. **NPU Stage 1a** creates `ncm-exec` with only these types plus the `ncm-model → ncm-exec` dependency. It does not touch `engine.rs`.
2. **Offload Stage 1** adds the planner and flags in `ncm-model`. Its `engine.rs` edit is limited to the start-up line.
3. **NPU Stage 1b** refactors `engine.rs` onto the traits. It owns `engine.rs` from then on.

Offload Stage 5 (placement at launch) then lands in `BurnBackend` construction (`ncm-model/src/exec.rs`) rather than in `engine.rs`. The CPU offload plan is edited to match (separate commit on `docs/ncm-cpu-offload`).

### 3. What NPU stacks offer (researched 2026-11-26), and target order

| Target | Path | Shapes | Precision | State | Notes and risk |
|---|---|---|---|---|---|
| **Intel NPU** (Core Ultra) | ORT OpenVINO EP via Windows ML; direct OpenVINO as the fallback | Static. The GenAI NPU pipeline fixes `MAX_PROMPT_LEN` / `MIN_RESPONSE_LEN`, with chunked dynamic prompts since 2025.3 ([OpenVINO GenAI on NPU](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html)) | Symmetric INT4 weights (same source); KV u8/u4 ([OVMS](https://docs.openvino.ai/2025/model-server/ovms_docs_llm_reference.html)) | OVMS NPU is stateful-only. One paper reports the NPU compiler rejecting ReadValue/Assign elsewhere ([arXiv 2608.19147](https://arxiv.org/pdf/2608.19147)), so we use explicit I/O | `CACHE_DIR` compile cache. **Lowest risk:** the best-documented generic ONNX → NPU path |
| **Qualcomm Hexagon** (Snapdragon X) | ORT QNN EP via Windows ML | Static only ([QNN EP](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)) | QDQ int8/int16; fp16 via `enable_htp_fp16_precision` (same doc) | Explicit I/O; Genie bundles fix context and KV ([AI Hub](https://dragonwingdocs.qualcomm.com/Key-Documents/AI-Developer-Workflow/topic/genai-prepare-ai-hub)) | Context binaries are tied to a QAIRT version. Large models are split into parts. **Medium risk** |
| **AMD XDNA** (Ryzen AI 300) | ORT VitisAI EP via Windows ML | Static (*unverified* for custom graphs) | LLMs: INT4 weights with BF16 activations ([AMD GPT-OSS article](https://www.amd.com/en/developer/resources/technical-articles/2026/accelerating-gpt-oss-20b-on-amd-ryzen-ai-npus.html)); int8 QDQ for other models | OGA-managed for LLMs ([OGA flow](https://ryzenai.docs.amd.com/en/latest/hybrid_oga.html)) | **Highest risk:** LLM-style NPU execution is documented only through OGA's prepared models, and hybrid models break across releases (same source). Custom ONNX LLM graphs on the VitisAI EP are unproven |

Packaging: Windows ML downloads and updates the OpenVINO, QNN and VitisAI EPs through `ExecutionProviderCatalog` on Windows 11 24H2+ ([supported EPs](https://learn.microsoft.com/windows/ai/new-windows-ml/supported-execution-providers)).

What this means:
- Static graphs, compiled ahead of time and cached.
- Quantised weights as standard QDQ.
- All state as explicit inputs and outputs in host-owned buffers.
- Target order: **Intel → Qualcomm → AMD.**

### 4. How the model maps onto an NPU

**Good fit:**
- linear layers, MLPs, RMSNorm;
- RoPE (tables passed in);
- encoder ring attention;
- the hybrid generator's K levels;
- codebook logits and lookup (`Gather`, or one-hot × codebook if `Gather` falls back);
- the token decoder's check pass;
- dense retrieval attention over the local ring;
- the indexer GEMV (a fixed bucket plus a mask).

**DeltaNet:**
- *Decode step:* standard ops, with the `[heads, 128, 128]` f32 state (about 0.5 MiB per layer) as explicit I/O.
- *Chunked prefill:* follows the code's convention (`delta.rs`). `neg_a = −A` for a strictly lower-triangular `A`, and `(I + A)⁻¹ = (I + neg_a)(I + neg_a²)(I + neg_a⁴)(I + neg_a⁸)(I + neg_a¹⁶)(I + neg_a³²)`, exact for C = 64. That is **10 matmuls** of 64×64: 5 squarings and 5 multiplies.
  - The code runs this algebra in f32 because bf16 loses most of its accuracy (`delta.rs` lines 116–117). The exported inverse subgraph must also be **f32**.
  - On an EP without f32 matmul on the NPU (likely QNN HTP, which runs int8/int16/fp16), that subgraph runs on the CPU EP. This is an explicit partition: per chunk and layer, 6 × 64×64 tensors.
  - Tolerance: max |Δ| ≤ 1e-4 against the Burn f32 inverse per chunk.
  - Long-context check: the DeltaNet state after 8,192 recurrent positions (goal 3) catches drift that compounds over chunks.

**On the CPU:**
- top-k block selection plus the gather into a fixed `[topk × block]` input, beyond the local window;
- sampling with top-k or top-p;
- grammar;
- the hybrid's acceptance logic, in `BlockLogic` (the host reads back 2 + 2L ints per pass anyway).

**Sampling inside the graph.** Temperature-only sampling can stay in the graph: the host passes counter-based Gumbel noise as an input, and the graph takes `argmax(logits / T + g)`. That is deterministic per seed. Requests with top-k or top-p split the block pass per level, with draws on the host. Greedy is one pass.

**Context thresholds for the hybrid** (L = 32, `ret_local` 1,024 positions):
- Up to 32k tokens, retrieval is dense over the local window, so there is no top-k and the backbone step is one static graph.
- Beyond 32k, each retrieval layer runs as indexer on the NPU, then top-k and gather on the CPU (16.5 MB per 32-token block, *est.*), then attention on the NPU.

**Why the hybrid suits NPUs:**
- It makes about 2–4 graph invocations per 32-token block (backbone step, the block pass with all levels unrolled, any re-decode or fill) with 32–99-row matmuls.
- Token-by-token decode makes one 1-row invocation per token.
- Dispatch cost per invocation is *unknown* (likely 0.1–1 ms), and the hybrid amortises it.
- Every rejected draft position costs a fill pass, so the trained model's acceptance rate decides how much of that advantage survives.

**Graph set:**
- prefill buckets of 1,024 and 128 tokens;
- backbone step;
- block pass with levels padded to 32 rows: one graph, about 2.6× the compute of 99 rows;
- fill pass;
- token decode step.

Weight duplication across graphs is checked per EP (Stage 0b and Stage 5).

### 4a. How NPU placement and NPU parity are checked

**Placement**, the source of the partition acceptance:
- Every graph meant for the NPU runs in its own ORT session with the session config `session.disable_cpu_ep_fallback = 1`. With that set, a node the EP rejects makes session creation fail instead of silently running on the CPU, so creating the session at all proves the graph is fully on the NPU.
- The CPU parts get their own sessions on the CPU EP, so their placement holds by construction:
  - the f32 DeltaNet inverse, on targets where Stage 0b puts it on the CPU;
  - top-k and gather;
  - host sampling.
- The node counts in the harness JSON (`partition.<graph>.npu_nodes` / `cpu_nodes`) come from parsing ORT's verbose-log "Node placements" lines, with log severity set to verbose for the harness run only.

**Parity on the NPU**, against the Burn reference with the *same* quantisation (`--quant q8` / `q4`, same group), over 64 greedy steps from fixed prompts:

| Weights | max \|Δlogit\| (tiny model / 250M) | Greedy match |
|---|---|---|
| int8 | ≤ 0.05 / ≤ 0.1 | identical tokens for all 64 steps on the tiny model; ≥ 62/64 on the 250M sample |
| int4 | ≤ 0.1 / ≤ 0.25 | identical tokens for all 64 steps on the tiny model; ≥ 60/64 on the 250M sample |

These thresholds are *est.*; Stage 0b and Stage 4 re-derive them from the measured spread and record them with the result. A mismatch must also be explainable: the reference's top-2 logit margin at the first diverging step must be below the |Δlogit| bound.

**Recurrent-state drift:**
- The DeltaNet state buffers stay **f32** and **host-owned**: graph inputs and outputs in f32, even when activations are fp16 or int16. So rounding cannot compound in the stored state between invocations.
- A long-context check on each target compares the NPU state against Burn f32 after 8,192 recurrent positions: max |Δ| ≤ 1e-2 *est.*, looser than the CPU EP's 1e-3 because the step's internal math runs at the NPU precision.

### 5. Expected speed and memory (*est.*, unmeasured; 250M configs)

| Path | Weight bytes per step | Bandwidth ceiling (60 GB/s) | Planning guess |
|---|---|---|---|
| Token baseline int8 | 0.25 GB per token | about 240 tok/s | 100–200 tok/s |
| Token baseline int4 | 0.13 GB per token | about 460 tok/s | 150–350 tok/s |
| Hybrid int8 | about 0.38 GB per 32-token block | about 5,000 tok/s | 500–1,500 tok/s |
| Hybrid int4 | about 0.2 GB per block | about 9,700 tok/s | 600–2,000 tok/s |

Why the hybrid's int4 guess gains so little over int8: the hybrid's ceilings are far above its guesses. Its speed is expected to be bound by dispatch overhead, fill passes and int4 dequantisation cost, not by bytes, so halving the bytes moves the guess only modestly. The token baseline is bytes-bound, so int4 helps it more.

Prefill is compute-bound: about 20k tok/s at 10 effective TOPS (rated 45–50 TOPS int8, vendor, *unverified*).

Comparison: today's CUDA NCM v2 runs at about 1,400 tok/s on an RTX 5080. The NPU's value is running Princeps without a free discrete GPU, at laptop power.

Memory is all system RAM:
- weights: 0.13–0.25 GB per graph set (more if an EP duplicates weights per graph);
- hybrid caches at 1M tokens: about 0.2–0.3 GiB;
- token baseline caches at 1M tokens: 12.5 GiB;
- compile cache: *unknown*.

### 6. Exporter: hand-built ONNX from Rust

`ncm-export` writes ONNX (opset 21) protobuf through `prost`-generated types from `onnx.proto` (Apache-2.0). Each layer kind has a builder mirroring the Burn reference:
- `Linear` / `QLinear`, `RmsNorm`, `Rope`;
- ring GQA attention;
- DeltaNet step and chunk;
- MLA absorbed attention;
- indexer, VQ level, copy head.

**Output.** Weights go out as external data, plus a JSON manifest (config, graphs, buckets, input/output names) that `ncm-ort` reads.

**Quantisation:** standard QDQ only.
- int8: `DequantizeLinear` with per-channel scales.
- int4: opset-21 int4 `DequantizeLinear` with `block_size = quant_group`, symmetric. Symmetric is what OpenVINO's NPU path requires.
- ORT's `MatMulNBits` is a **contrib op** and is unlikely to be supported on NPU EPs. It is used only for the CPU EP, and only if QDQ int4 is slow there.

**Rejected:**
- A PyTorch mirror with `torch.onnx.export`: a second implementation that drifts.
- Tracing Burn: no exporter (Stage 0 confirms).

### 7. Server, bench, Imperium, wgpu

**ncm-server:**
- `--device ort-cpu | npu[:intel|qualcomm|amd|auto]`.
- `-ngl` is ignored on these devices, with a note.
- `--npu-cache <dir>` sets the compile cache.
- `/health` stays 503 while graphs compile.
- `/props` `ncm.device` reports the EP, the EP version and the ORT version; `failure()` text appears there too.
- `--print-footprint` prints the `ncm-exec` footprint.

**ncm-bench:**
- Stage 3 adds `--device ort-cpu`.
- Stage 4 adds `--device npu:<target>` and the JSON schema.

**Imperium:** `runtime.ncm.device` maps to `--device` (Stage 6c), then the Imperium and research sessions are notified.

**wgpu:** arrives as another Burn device in `BurnBackend`, with its own `Capabilities`.

### 8. Dependencies, licences, version skew, size

**Dependencies:**
- **`ort` 2.0.0-rc.15**, targeting ONNX Runtime 1.31. Licence MIT / Apache-2.0 (*unverified*). Use `load-dynamic`, so `ncm-server` links nothing. Behind the runtime's `ort` feature.
- ONNX Runtime: MIT.
- OpenVINO: Apache-2.0.
- QNN/QAIRT and VitisAI: vendor terms. They arrive only through Windows ML, so we never redistribute them.

**Version skew (risk).** Windows ML ships its own ORT build, which may lag or lead the version `ort` targets. Whether `ort` can load Windows ML's `onnxruntime.dll` and register catalog EPs is *unverified*. Mitigations:
- at start-up, compare `OrtGetApiBase()->GetVersionString` against the version `ort` needs, and refuse with a clear message on a mismatch;
- pin `ort` to the Windows ML ORT line where possible;
- Stage 0b records the exact versions per target.

**Size:**
- `onnxruntime.dll` CPU: about 10–20 MB (*est.*);
- NPU EPs: tens to hundreds of MB (*est.*), downloaded by Windows ML.

**CI:** Linux, with an ONNX Runtime 1.31 CPU release pinned by version and SHA-256 and fetched by `scripts/check.sh` (approved).

## Stage 0: Spike: one decode step to ONNX on the ORT CPU EP
- **Objective:**
  - Confirm Burn 0.22.1 has no ONNX export.
  - Hand-build one tiny token-baseline v2 decode step (DeltaNet step, dense local retrieval, MLP, logits) with explicit cache I/O.
  - Run it through `ort` 2.0.0-rc.15 on the CPU EP and compare against the Burn reference.
  - Time one session run, and check that `load-dynamic` loads a given DLL.
- **Depends on:** none. No GPU or NPU needed.
- **Owned paths:**
  - Branch `spike/ncm-onnx-step` (throwaway, not merged).
  - Results file `models/ncm/docs/plans/npu-results/stage0.md`, committed on `docs/ncm-npu`: versions, parity numbers, run overhead, graph size.
- **Acceptance checks:**
  - f32 max |Δlogit| ≤ 1e-4, and greedy tokens identical over 64 steps.
  - The results file exists and holds the numbers.
- **Status:** pending
- **Outcome:**
- **Handoff notes:**
  - If Burn has an exporter after all, re-decide decision 6.
  - If a CPU-EP run costs over about 1 ms, raise the invocation-budget question before Stage 4.

## Stage 0b: NPU smoke test, per target (hardware-gated)
- **Objective:** on each target's hardware, run a tiny graph through ORT and that target's Windows ML EP, then record the results. The graph contains:
  - a matmul;
  - RMSNorm;
  - int8 per-channel and int4 block-wise symmetric `DequantizeLinear`;
  - explicit cache input → output with a fixed shape;
  - an f32 64×64 inverse subgraph.

  Record per target:
  - the ORT and EP versions and whether `ort` attached to Windows ML's ORT;
  - the EP partition report (which nodes ran on the NPU);
  - the compile time;
  - the per-run overhead.
- **Depends on:** stage 0; NPU hardware per target (open question 1).
- **Owned paths:** `models/ncm/docs/plans/npu-results/stage0b-<target>.md`; the smoke graph and runner on the spike branch.
- **Acceptance checks:**
  - Per target: the smoke graph's outputs match the CPU EP within the quantised tolerance.
  - The partition report and versions are recorded.
- **Status:** pending (**blocked: no NPU hardware**)
- **Outcome:**
- **Handoff notes:**
  - This **gates the NPU-specific parts of Stages 2–3**: which DQ form to export, which ops to avoid, and whether the inverse runs on the CPU.
  - The generic exporter and the CPU-EP backend proceed without it.

## Stage 1a: `ncm-exec` footprint types
- **Objective:**
  - Create `crates/ncm-exec` holding `Footprint`, `Tier`, `Part`, `FootprintRequest` (with `tokens_per_position`) and `Placement` (decision 2).
  - Make `ncm-model` depend on it unconditionally.
- **Depends on:** `feat/ncm-varlen-batch` merged. It goes before CPU offload Stage 1 (decision 2).
- **Owned paths:** `crates/ncm-exec/{Cargo.toml, src/lib.rs, src/footprint.rs}`, the workspace `Cargo.toml`, `crates/ncm-model/Cargo.toml`.
- **Acceptance checks:**
  - Goal 2's check succeeds: `! cargo tree -p ncm-exec -e normal | grep -q burn`.
  - Unit tests of the byte sums: `cargo test -p ncm-exec`.
  - `scripts/check.sh` passes.
- **Status:** pending
- **Outcome:**
- **Handoff notes:** no behaviour change. Tell the offload plan's implementer the types are in.

## Stage 1b: Interface traits, hybrid `BlockLogic`, Burn adapter, engine routing
- **Objective:**
  - Add the traits of decision 1 to `ncm-exec`.
  - Move the host part of the hybrid block state machine into `ncm_exec::hybrid::BlockLogic`. `FastSession` keeps staging and passes and embeds `BlockLogic` in its `BlockRun`.
  - Implement `BurnBackend` for CPU and CUDA.
  - Route `Core`'s sequential paths through the traits, one path at a time: eager, graphed greedy, graphed sampled, speculative, hybrid.
  - Keep the batcher as the Burn-only side door.
- **Depends on:** stage 1a.
  - CPU offload Stage 1 is a soft ordering only: it touches just the `engine.rs` start-up line.
  - If offload Stage 1 has landed, 1b keeps its footprint start-up line and routes it through `ExecBackend::footprint`.
  - If it has not landed, 1b goes ahead, and offload Stage 1 later adds its line on top of the refactored `engine.rs` (a one-line edit there).
- **Owned paths:**
  - `crates/ncm-exec/src/{exec.rs, hybrid.rs}`
  - `crates/ncm-model/src/{exec.rs, hybrid/fast.rs}`
  - `runtime/src/{engine.rs, app.rs, batch.rs (side-door construction only)}`
- **Acceptance checks:**
  - Goal 1's runtime tests pass on CPU and CUDA.
  - **Speed within ±2%.** Before the refactor, build `ncm-bench` at the parent commit in `--release`. Record `--model both --design v2 --context 2048 131072 --repeats 5 --json base.json` and a hybrid run at 2k and 32k. Then repeat on the refactored commit on the same machine, with the GPU idle and at the same driver. Every configuration's median decode tok/s must be within ±2% of `base.json`.
- **Status:** pending
- **Outcome:**
- **Handoff notes:**
  - This is the riskiest refactor, so it is staged path by path, each with its existing test.
  - After this stage, `engine.rs` changes belong to this plan's stages; CPU offload Stage 5 goes in `ncm-model/src/exec.rs`.

## Stage 2: ONNX exporter
- **Objective:**
  - `crates/ncm-export`: the opset-21 writer, the layer builders, the manifest, and every phase graph of decision 4, including:
    - the DeltaNet f32 inverse (10 matmuls);
    - the split retrieval graphs beyond the local window;
    - QDQ int8 and int4 export.
  - Command: `ncm-export --model <weights> --out <dir> --buckets 128,1024 [--quant q8|q4] [--target cpu|intel|qualcomm|amd]`. A per-target profile sets the NPU-specific choices.
- **Depends on:** stage 0. The per-target profiles also depend on stage 0b for that target.
- **Owned paths:** `crates/ncm-export/**`. Read-only use of `crates/ncm-model/src/{config.rs, weights.rs, quant/format.rs}`.
- **Acceptance checks:**
  - Per-builder parity on the CPU EP: `cargo test -p ncm-export`.
  - DeltaNet: the chunk inverse matches within 1e-4 in f32, and the state matches within 1e-3 after 8,192 recurrent positions.
- **Status:** pending
- **Outcome:**
- **Handoff notes:** no dynamic shapes, `Loop`/`If`, ScatterND on caches or stateful variables. The host writes new cache entries into the rings.

## Stage 3: ONNX Runtime backend on the CPU EP
- **Objective:**
  - `crates/ncm-ort`: `OrtBackend` implementing the traits over the manifest, with:
    - IO-bound host cache buffers;
    - CPU top-k and gather;
    - prefill buckets;
    - `BlockLogic` for the hybrid;
    - snapshots in system RAM;
    - the version check of decision 8.
  - Before adding `ort` 2.0.0-rc.15, check its MSRV (`rust-version` in its `Cargo.toml`) against the workspace toolchain (`rust-toolchain.toml`, 1.99.1). If it needs a newer toolchain, stop and report instead of bumping the toolchain.
  - Add the `ort` feature to `runtime/Cargo.toml` (optional `ncm-ort` with `ort/load-dynamic`) and `--device ort-cpu`.
  - Make `runtime/tests/e2e.rs` take its device from the `NCM_E2E_DEVICE` environment variable (default: Burn CPU).
  - Add `--device ort-cpu` to `ncm-bench`.
  - Make `scripts/check.sh` fetch the pinned ORT CPU release and run the parity and e2e tests.
- **Depends on:** stages 1b and 2. The NPU-specific EP option code waits for stage 0b.
- **Owned paths:**
  - `crates/ncm-ort/**`
  - `runtime/{Cargo.toml, src/cli.rs, src/engine.rs (device selection), src/app.rs, src/server.rs (/props), tests/e2e.rs, README.md}`
  - `crates/ncm-bench/src/lib.rs` (`--device ort-cpu`)
  - `scripts/check.sh`
- **Acceptance checks:**
  - Goal 3: `cargo test -p ncm-ort --test parity`.
  - Goal 4: `NCM_E2E_DEVICE=ort-cpu cargo test -p ncm-server --features ort --test e2e`.
  - Prefix reuse gives the same `cached_tokens` as the Burn backend.
  - `scripts/check.sh` passes in CI.
- **Status:** pending
- **Outcome:**
- **Handoff notes:** keep the ORT version in one place, the `scripts/check.sh` pin, alongside the `ort` version in `Cargo.toml`.

## Stage 4: First NPU target: Intel (hardware-gated)
- **Objective:**
  - `--device npu:intel` through Windows ML's OpenVINO EP, with direct OpenVINO as the fallback if the EP or version skew blocks it.
  - Compile cache, partition check, readiness at 503.
  - Measure dispatch overhead and in-graph `TopK` against the CPU split.
  - Add `--device npu:<target>` to `ncm-bench`, with this JSON schema:
    - `{target, ep, ep_version, ort_version, model, quant, context, prompt_tokens, prefill_tok_s, decode_tok_s, invocations_per_block, dispatch_ms_p50, acceptance, partition: {<graph>: {npu_nodes, cpu_nodes}}, parity: {max_abs_dlogit, greedy_match_steps}, compile_s}`
  - Add a harness script that runs the e2e suite plus the bench on the device.
- **Depends on:** stage 3; stage 0b for Intel; Intel NPU hardware.
- **Owned paths:**
  - `crates/ncm-ort/src/ep/{mod.rs, openvino.rs}`
  - `crates/ncm-bench/src/lib.rs` (`npu:<target>`, the schema)
  - `runtime/src/{cli.rs, server.rs (readiness)}`
  - `scripts/npu-harness.ps1`
- **Acceptance checks:**
  - Goal 5 for Intel: the harness JSON validates against the schema, and the e2e tests pass.
  - Parity meets the int8 and int4 thresholds of decision 4a.
  - The 8,192-position state-drift check passes.
  - Every NPU session is created with `session.disable_cpu_ep_fallback = 1`.
  - The parsed node placements show 0 CPU nodes in the NPU sessions.
- **Status:** pending (**blocked: no NPU hardware**)
- **Outcome:**
- **Handoff notes:** correct the decision 5 estimates with the measured numbers.

## Stage 5: NPU quantisation and graph-set tuning, Intel (hardware-gated)
- **Objective:**
  - On Intel, choose:
    - the weight form: symmetric int4 or int8 QDQ;
    - the activation precision;
    - the level bucketing: one padded graph or two buckets;
    - the prefill buckets.
  - Enable weight sharing between graphs where the EP supports it.
  - Record the choices as the `intel` target profile.
- **Depends on:** stage 4.
- **Owned paths:** `crates/ncm-export/src/{quant.rs, targets.rs}`, `crates/ncm-ort/src/ep/openvino.rs`, `docs/quantisation.md` (NPU section).
- **Acceptance checks:**
  - Decision 4a parity for both weight forms.
  - Weights in memory are at most 1.5× one copy.
  - The harness JSON improves on Stage 4's decode or prefill tok/s.
- **Status:** pending (**blocked: no NPU hardware**)
- **Outcome:**
- **Handoff notes:** acceptance data from a trained hybrid decides how much fill-pass count matters. The profile format becomes the template for 6a and 6b.

## Stage 6a: Qualcomm bring-up and tuning (hardware-gated)
- **Objective:**
  - Bring up `--device npu:qualcomm` through the QNN EP.
  - Cache the context binaries per QAIRT version.
  - Export with int8/int16 QDQ and fp16 options.
  - Run the f32 inverse in its CPU session.
  - Tune as in Stage 5 and record the `qualcomm` profile.
- **Depends on:** stage 5 (profile format, harness); stage 0b for Qualcomm; Snapdragon X hardware.
- **Owned paths:** `crates/ncm-ort/src/ep/qnn.rs`, the `qualcomm` profile in `crates/ncm-export/src/targets.rs`.
- **Acceptance checks:**
  - Goal 5 for Qualcomm.
  - Decision 4a parity and state drift.
  - `disable_cpu_ep_fallback` sessions are created, with 0 CPU nodes in NPU sessions.
- **Status:** pending (**blocked: no NPU hardware**)
- **Outcome:**
- **Handoff notes:** record the QAIRT and ORT versions with the result. Context binaries must be rebuilt when QAIRT changes.

## Stage 6b: AMD bring-up and tuning (hardware-gated)
- **Objective:**
  - Bring up `--device npu:amd` through the VitisAI EP with the exported graphs, then tune as in Stage 5 and record the `amd` profile.
  - If custom graphs do not compile with CPU fallback disabled, stop and report. OGA-prepared models are the only documented LLM route, and adopting OGA is a separate decision.
- **Depends on:** stage 5; stage 0b for AMD; Ryzen AI 300 hardware. It is independent of 6a.
- **Owned paths:** `crates/ncm-ort/src/ep/vitisai.rs`, the `amd` profile in `crates/ncm-export/src/targets.rs`.
- **Acceptance checks:** goal 5 for AMD, plus decision 4a parity and state drift, and the fallback-disabled sessions. Otherwise, a recorded stop report naming the rejected ops.
- **Status:** pending (**blocked: no NPU hardware**)
- **Outcome:**
- **Handoff notes:** Ryzen AI releases break compatibility between versions, so record the exact release.

## Stage 6c: Imperium mapping and release
- **Objective:**
  - Map Imperium's `runtime.ncm.device` to `ncm-server --device`, via an Imperium PR owned by the Imperium implementation session.
  - Document the devices in `runtime/README.md`.
  - Write release notes listing the supported targets and their status.
  - Notify the Imperium and research sessions.
- **Depends on:** stage 3, for `ort-cpu`, and stage 4, for the first NPU target. Later targets are added to the notes as 6a and 6b land.
- **Owned paths:** `runtime/README.md` and the release notes. Also a launch-line test in an Imperium-AI PR under `crates/imperium-local` tests, modelled on PR #335's launch-line test, which asserts that `runtime.ncm.device` becomes the matching `--device` argument. This stage owns that test; the Imperium implementation session reviews and merges the PR.
- **Acceptance checks:**
  - With Imperium set to `runtime.ncm.device = "npu:intel"` (or `ort-cpu` without hardware), Imperium launches `ncm-server` with the matching `--device` and completes a chat.
  - The new `crates/imperium-local` launch-line test passes in Imperium-AI CI.
- **Status:** pending
- **Outcome:**
- **Handoff notes:** NPU batching stays out of scope until a target shows batch-1 is dispatch-bound and the EP supports it.
