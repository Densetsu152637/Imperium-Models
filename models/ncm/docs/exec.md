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
- The hybrid session now calls `begin_request(opts.seed)` per request, as the batcher already
  did (the sequential engine never did, so seeded sampled requests were not reproducible).

## The Burn adapter (`ncm_model::exec`)

`BurnBackend` + `BurnSession` are the old sequential engine's decode machinery moved behind the
traits (eager steps, graphed greedy/sampled steps, speculative NCM, snapshots, the hybrid's fast
session). `Core<B>` in `ncm-server` runs the three model kinds through them; `Core<BurnBackend>`
hands its model and first session's caches to the continuous batcher
(`into_batcher_parts`), which is unchanged.

## The exporter (`ncm-export`)

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
- **Coverage today**: the token baseline in the fallback design (GQA attention with sliding or
  full rings, SwiGLU MLP, DeltaNet) exports completely. **Not written yet** (`--describe` lists
  them; `export` fails naming the layer): the v2 design's sparse-retrieval layers (absorbed MLA
  and the indexer), the v2 gated attention with QK-norm, every hybrid phase graph (backbone
  step, block pass, fill, token check, the split retrieval graphs beyond the local window), the
  k = 4 NCM (no ONNX path by plan), and the NPU target profiles (they wait for plan stage 0b).

## The ONNX Runtime backend (`ncm-ort`)

See `crates/ncm-ort/src/lib.rs` and `runtime/README.md`: `--device ort-cpu` (build with
`--features ort`), `ONNXRUNTIME` shared library from `ORT_DYLIB_PATH`, the version pin in
`scripts/check.sh`.

## Verification status

Checked: `cargo check`, `cargo clippy -D warnings` (CPU, `--features cuda`, `ort`, `ort-tests`),
`cargo fmt`. **Not run**: any test binary, benchmark, server, GPU path or ONNX Runtime. The
tests are written and wait for a run: `crates/ncm-model/tests/exec.rs` (adapter vs the Burn
reference), `crates/ncm-export` unit tests and `tests/parity.rs` (`--features ort-tests`),
`crates/ncm-ort` tests, `NCM_E2E_DEVICE=ort-cpu` in `runtime/tests/e2e.rs`. The Stage 1b speed
gate (every `ncm-bench` configuration's median decode tok/s within +-2% of the parent commit,
same machine and driver) has not been run.
