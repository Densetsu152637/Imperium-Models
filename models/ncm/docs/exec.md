# The engine interface (`ncm-exec`), the exporter and the ONNX Runtime backend

Plan of record: `docs/plans/npu.md` (branch `docs/ncm-npu`). This page says what the code does
today; the plan says what it is for.

Nothing here has been run on a GPU or on ONNX Runtime yet: the stages were implemented
speculatively and checked with `cargo check`, `clippy` and `fmt` only (see "Verification status").

## Crates and dependency direction

```
ncm-exec   (no Burn, no ORT)  footprint types, ExecBackend/ExecSession, hybrid BlockLogic, manifest, rope
  ^   ^
  |   +-- ncm-ort     ONNX Runtime backend (CPU EP; NPU EPs later), reads the exporter's manifest
  |   +-- ncm-export  hand-built ONNX graphs + QDQ weights + manifest   (also reads ncm-model's config)
ncm-model  BurnBackend (src/exec.rs) for CPU and CUDA
ncm-server (runtime) Core<B: ExecBackend>; the batcher is a Burn-only side door
```

`! cargo tree -p ncm-exec -e normal | grep -q burn` holds.

## The interface

`ExecBackend` owns one loaded model (`capabilities`, `footprint`, `new_session`, `warm_up`,
`sync`); `ExecSession` owns one request stream's caches.

| Call | Token baseline | k = 4 NCM (Burn only) | Hybrid |
|---|---|---|---|
| `prefill(tokens, chunk)` | one chunk, logits held | one chunk of whole concepts, condition held | any length |
| `decode_step(feed, pick)` | feed one token (none: from the held logits), pick the next | feed the concept just written, pick the next concept (`forced` starts it) | not used |
| `decode_concept_host` | not used | host chooses each token of the concept | not used |
| `begin_block` / `phase` / `pass` / `fill` / `end_block` / `decode_block` | not used | not used | the block cycle |
| `snapshot` / `restore` / `reset` | all | all | all |

`PickSpec { mode: Greedy | Sampled(DrawRow) | Host, forced, speculate }`: `Greedy` and `Sampled`
pick on the backend (CUDA graphs; `StepOut::Picked`), `Host` returns logits for the engine's
sampler and grammar (`StepOut::Logits`). A session keeps the output its next step needs on its
device (`PendingOutput`); `pending_logits()` reads it back on demand, so graphed paths never
read logits. Snapshots carry that output (`SnapshotInfo::has_output`), which is how a prompt
exactly equal to a kept prefix resumes without a prefill.

### Where this differs from the plan's sketch (decisions the plan left open)

- `Snapshot` and `BlockRun` are associated types, not fixed opaque ones (`ExecSession::Snapshot:
  SnapshotInfo`), so the engine can ask `has_output()` and a backend can embed `BlockLogic`.
- `pass` and `fill` execute **and apply** the pass (the backend's `BlockRun` embeds a
  `BlockLogic`), with the `Hooks` (grammar scan and pick) as an argument. The plan's `draws`
  argument is derived inside the session from the options and `begin_request(seed)`;
  `DrawSettings`/`DrawRow` are backend-neutral and CUDA maps them to its sampler settings.
- `ExecSession::decode_block` drives a whole block (`drive_block` by default); the Burn backend
  overrides it only for options the fast passes do not serve (host draws on the CPU, the draft
  and code rewrite hooks), which take the reference loop as before.
- `ExecSession::tokens()` (hybrid) gives the stream so far, for prefix reuse and the room left.
- `ExecBackend::warm_up`/`sync`/`describe` exist so the engine's start-up and timing boundaries
  stay backend-neutral.
- `FootprintRequest::tokens_per_position` is a `(numerator, denominator)` pair to stay exact
  (1/1 for the token baseline, 1/block for hybrid backbone positions).
- `BurnBackend::footprint` reports the cache as one `LatentCache` figure per slot: the per-part
  split (rings, indexer keys, DeltaNet state) belongs to the offload planner.
- `ExecSession::append` adds a concept without decoding (the NCM's eager path appends at once, as it always did, so a cancel keeps that concept in the kept prefix; graphed steps append and decode together).
- The hybrid session now calls `begin_request(opts.seed)` per request, as the batcher already
  did (the sequential engine never did, so seeded sampled requests were not reproducible).

## The Burn adapter (`ncm_model::exec`)

`BurnBackend` + `BurnSession` are the old sequential engine's decode machinery moved behind the
traits (eager steps, graphed greedy/sampled steps, speculative NCM, snapshots, the hybrid's fast
session). `Core<B>` in `ncm-server` runs the three model kinds through them; `Core<BurnBackend>`
hands its model and first session's caches to the continuous batcher
(`into_batcher_parts`), which is unchanged.

## The exporter (`ncm-export`)

Weight names are exact Burn module paths. `Mixer` is a module enum and `save_weights` leaves `skip_enum_variants` off, so the variant is a path segment (`stack.blocks.0.attn.DeltaNet.proj.weight`, `...attn.Attention.qkv.weight`); this follows burn-derive's `codegen_enum` and burn-store 0.22.1 but was not checked against a real file. `ncm-export --model f.safetensors --list-tensors` prints a header's names, dtypes and shapes to check it.

```
ncm-export --model weights.safetensors --out dir --buckets 128,1024 --ctx 4096 \
           [--quant none|q8|q4 --group 64] [--target cpu] [--describe]
```

Writes `decode.onnx`, `prefill_<bucket>.onnx`, one shared external `weights.bin` (offsets
aligned to 4 KiB; a weight appears once whatever number of graphs use it) and `manifest.json`
(`ncm_exec::manifest`: layers, rope tables, every graph input and output with its role).

- **Opset 21, IR 10**, static shapes, no `Loop`/`If`/`ScatterND`/stateful variables: caches and
  recurrent state are explicit inputs and outputs the host owns; the host writes new ring
  entries from the `k_new`/`v_new` outputs.
- **Quantisation** is standard QDQ from the repository's own host quantiser, so an exported
  weight dequantises to exactly what the Burn backend computes at the same `--quant`/`--group`:
  int8 per output channel (`DequantizeLinear`, axis 1), int4 block-wise symmetric (opset-21 int4
  codes, `block_size = group`, axis 0; the input width must be a multiple of the group).
  `MatMulNBits` is not used.
- **DeltaNet**: the decode step, and the chunked prefill with the f32 triangular inverse in the
  graph: 5 squarings and 5 multiplies of 64 x 64 at chunk 64 (`(I+N)(I+N^2)...(I+N^32)`).
- **Coverage today** (everything below is **unverified**: written from the Burn code, compiled
  and linted, never run):
  - the fallback-design token baseline (GQA attention with sliding or full rings, SwiGLU MLP,
    DeltaNet): decode and prefill-bucket graphs;
  - the v2 gated attention with QK-norm (`layers::AttnExtras`);
  - **sparse retrieval** (absorbed MLA, the four norms, decoupled partial RoPE with YaRN, the
    8-bit indexer keys as outputs) as **dense attention over the whole latent cache**
    (`retrieval_dense`): exact while the Burn selection would pick every block
    (`position < local + topk * block`, recorded per layer as `dense_exact_until`);
  - **the split sparse path** for decode beyond that: the decode step is cut at every
    retrieval layer (`decode_split_<k>` graphs, `Phase::DecodeSplit`: segment `k` ends at the
    layer's projection, the next starts at its attention). Between segments `ncm-ort` scores
    the cached 8-bit indexer keys, takes block maxima and the top-k, and gathers the picked
    blocks and the local window (`ncm_exec::retrieval`, mirroring `attend_selected`: ties go to
    the earlier block, picked positions inside the window are masked). Prefill buckets are used
    only below the dense range; beyond it prefill runs token by token through the split step.
    The index cache is host RAM, snapshotted with the layer.
- **Not written**: every **hybrid** phase graph (encoder and backbone step, the block pass with
  its K generator levels, the codebook lookups, the tokenizer decoder and check pass, the fill
  pass, the copy head) and `OrtBlockRun::begin_block`; the k = 4 NCM (no ONNX path by plan);
  the NPU target profiles (they wait for plan stage 0b). The hybrid is not a graph translation
  alone: `FastSession` stages ring projections, visibility biases, retrieved tokens and known
  prefixes per block (about 1,400 lines plus `HybridSession::close_block`, the pooled retrieval
  history and `select_retrieved`), all of which the ORT session would have to reimplement
  against graphs nobody has run. That was judged not writable correctly blind.

## The ONNX Runtime backend (`ncm-ort`)

`OrtBackend::load(dir)` reads `manifest.json`, creates one ONNX Runtime session per graph on the
CPU execution provider (`ort` 2.0.0-rc.15, `load-dynamic`: nothing is linked; the library comes
from `--ort-lib`, `ORT_DYLIB_PATH` or the platform name) and refuses a library whose minor
version is not the one `ort` targets (1.31) unless `--ort-allow-version-skew`. The pin lives in
`crates/ncm-ort/Cargo.toml` and `scripts/check.sh`.

- **State is host-owned**: attention rings are written in place at `position % ring` (so a
  restore is a `memcpy` into the same buffers), DeltaNet states stay f32, convolution windows
  are kept per layer. Each run copies buffers into input tensors and the new entries out of the
  outputs; ONNX Runtime IO binding (zero-copy inputs) is the obvious next optimisation.
- **Masks**: the additive bias of an attention layer is built on the host per call (ring slot
  `s` holds the latest position `p < len` with `p % ring == s`; a query sees the previous
  `ring` positions and itself), for decode steps and prefill buckets alike.
- **Prefill** takes the largest exported bucket that fits and finishes token by token on the
  decode graph. **Sampling and grammar** run on the host.
- **Hybrid**: `OrtBlockRun` embeds the shared `BlockLogic`, but there are no block graphs, so
  `begin_block` reports that. The NPU execution providers (`ep/` modules, `disable_cpu_ep_fallback`
  sessions, partition reports, compile caches) are plan stages 4-6 and need hardware.
- **Server**: `--device ort-cpu --model <export dir>` (needs `--features ort`); `ncm-bench
  --device ort-cpu --onnx-dir <dir>`. `scripts/check.sh` fetches the pinned CPU release by
  version and SHA-256 and runs the parity and e2e tests; **its `ORT_SHA256` is empty and must
  be pinned from the official release** (until then the script skips those tests and says so).

## Verification status

Checked: `cargo check`, `cargo clippy -D warnings` (CPU, `--features cuda`, `ort`, `ort-tests`),
`cargo fmt`. **Not run**: any test binary, benchmark, server, GPU path or ONNX Runtime. The
tests are written and wait for a run: `crates/ncm-model/tests/exec.rs` (adapter vs the Burn
reference), `crates/ncm-export` unit tests and `tests/parity.rs` (`--features ort-tests`),
`crates/ncm-ort` tests, `NCM_E2E_DEVICE=ort-cpu` in `runtime/tests/e2e.rs`. The Stage 1b speed
gate (every `ncm-bench` configuration's median decode tok/s within +-2% of the parent commit,
same machine and driver) has not been run.
