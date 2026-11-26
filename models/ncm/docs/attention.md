# Attention designs

Both models (the NCM and the token baseline) come in two attention designs, picked by the config's `attention` key:

| `attention` | Design | Configs |
|---|---|---|
| `"v2"` | Gated DeltaNet alternating with sparse retrieval attention (adopted 2026-11-25) | `configs/ncm-250m-v2.toml`, `configs/tok-250m-v2.toml` |
| `"fallback"` (default) | Sliding-window attention with a full-attention layer every fourth (the stage 1 design) | `configs/ncm-250m.toml`, `configs/tok-250m.toml` |

The design itself is settled in Imperium-Research (`docs/next-concept-modelling/architecture.md`). This page records how `ncm-model` implements v2 and the choices made where the design leaves details open. The layers are written as **reference** code: plain Burn tensor ops, written for correctness, and the only path on the CPU and in CI. On CUDA, decode runs fused CubeCL kernels and replays CUDA graphs instead ([Fast decode on CUDA](#fast-decode-on-cuda-srckernels-srcfusedrs), local test plan step 3), tested against the reference.

v2 settings live in the `[v2]` table of a config (`V2Config`); fallback models ignore them. A weight file's header carries the whole config, so `ncm-server` loads either design from the weights alone.

## Layer layout

| Model | Stack | Layers |
|---|---|---|
| NCM | Local encoder, d = 512 | sliding window (512 tokens), gated DeltaNet, sliding window |
| NCM | Global backbone, d = 1024, over concepts | 16: DeltaNet, retrieval, DeltaNet, retrieval, ... (8 + 8) |
| NCM | Local decoder, d = 512 | unchanged from the fallback: 4 layers, self-attention over the concept, cross-attention to the last 512 encoder states |
| NCM | Draft head, d = 512 | 1 layer (`draft_layers`): the whole next concept in one pass, for self-speculative decoding ([Local decoder and speculative decoding](#local-decoder-and-speculative-decoding-srcspeculativers)) |
| Token baseline | d = 1024, over tokens | 20: DeltaNet and retrieval alternating (10 + 10) |

Every layer is pre-norm (RMSNorm) with a residual mixer and a residual SwiGLU MLP. The backbone alternates starting with DeltaNet, so the first retrieval layer already sees order from the convolution and recurrence. In v2 the encoder's sliding-window layers also get the output gate and QK-norm.

## Gated DeltaNet (`src/delta.rs`)

Follows Gated DeltaNet (Yang et al., 2024) as Qwen3-Next and Qwen3.5 use it.

- **Heads:** layer width / `dn_head_dim` (128): 8 heads in the backbone, 4 in the encoder. Keys and values are both 128 wide per head, so the state is 128 × 128 per head. Qwen3-Next groups two value heads per key head; equal heads keep the reference simple and cost about 1M parameters more per layer.
- **Input path:** one projection gives q, k, v, the output gate and the per-head logits `a` and `b`. q, k and v pass a causal depthwise convolution of width 4 (`dn_conv`) and SiLU, as in Qwen3-Next. q and k are L2-normalised per head (DeltaNet's form of QK-norm) and q is scaled by 1/sqrt(128).
- **Gates:** write strength `beta = sigmoid(b)`; log decay `g = -exp(A_log) * softplus(a + dt_bias)` (Mamba-2 parameterisation, as in Qwen3-Next). `A_log` starts at ln(1..16) and `dt_bias` at the inverse softplus of time steps 0.001..0.1, spread across heads deterministically.
- **Output:** per-head RMSNorm of the read-out, times a **sigmoid** gate of the input, then the output projection. The architecture doc asks for a sigmoid output gate on every DeltaNet output; Qwen3-Next itself uses SiLU on this gate. Switching is one line if the learning check (step 4) prefers it.
- **Recurrent form** (`delta_rule_step`): one token, used for decode. The cache holds the state `[B, H, 128, 128]` and the last 3 convolution inputs.
- **Chunked form** (`delta_rule_chunked`): training and prefill, chunks of 64 (`dn_chunk`). Inside a chunk the corrected values solve a unit lower-triangular system `(I + A) u = ...`. Because `A` is strictly lower triangular, `(I + A)^-1 = (I - A)(I + A^2)(I + A^4)...` is exact after log2(64) = 6 squarings, so no row-by-row substitution loop is needed. Chunks are processed in order, carrying the state. A ragged tail is padded with inert positions (zero key, zero write strength, zero log decay).
- No position encoding: the convolution and the recurrence carry order.

## Sparse retrieval attention (`src/retrieval.rs`)

Follows DeepSeek-V2 multi-head latent attention for the cache and DeepSeek-V3.2 sparse attention for the indexer, with the doc's block selection.

- **Heads:** 8 query heads, each with a 128-dimensional part without position and a 64-dimensional RoPE part; values 128 wide (DeepSeek's per-head sizes at half the head count, for d = 1024). Queries are not compressed (as in DeepSeek-V2-Lite).
- **MLA cache:** per position, a 512-value latent plus one 64-value RoPE key shared by all heads: 576 values, 1,152 bytes in bf16. Keys without position and values are up-projections of the latent. With a cache (every prefill chunk and decode step) the key up-projection is **absorbed** into the queries and the value up-projection is applied after the weighted sum, so attention runs on the latent directly. The training forward expands per-head keys and values instead; the two are algebraically equal and the incremental-decode tests check it.
- **QK-norm:** RMSNorm on each query head, on the latent (MLA's own `kv_norm`) and on the RoPE key. Per-head key norms after the up-projection would break absorption, so the latent norm stands in for them.
- **Partial RoPE:** only the 64-dimensional decoupled key and the matching query part rotate, with base 1,000,000 (`rope_base_full`); YaRN extension to 1M belongs to the long-context stage.
- **Indexer:** 4 heads (`idx_heads`) with 128-dimensional queries and one shared 128-dimensional key per position (RMSNorm on the key), scored as in DeepSeek-V3.2: `I(t, s) = sum_j w_j(t) relu(q_j(t) . k(s))`, with learned per-head weights `w`. No RoPE in the indexer: block retrieval is meant to be content based. A block's score is the **maximum** of its positions' scores.
- **Selection:** a query at position `p` always sees its last 1,024 positions (`ret_local`). Blocks of 64 positions (`ret_block`) that start before that window are candidates; the query attends to the top 32 (`ret_topk`) by block score, restricted to positions older than the window (positions inside the window are already visible). When a query has 32 candidates or fewer it sees its whole causal context. Selection is per query and shared by all heads (MLA keys are shared).
- **Dense mode:** `ret_dense = true` makes every query attend to its whole causal context: the training warm-up, and the reference the sparse path is tested against.
- **Output gate:** elementwise sigmoid gate on the attention output (heads × 128), from the layer input, as in Qwen3-Next's gated attention. The same gate is on the encoder's sliding-window layers.
- **Indexer keys** are 8-bit in the design (as DeepSeek-V3.2 keeps them in FP8). The reference stores them at the run dtype (bf16 on CUDA); `cache_bytes` reports the design figure and `cache_bytes_with` what the reference allocates.

Execution paths:

| Call | Path |
|---|---|
| One-token decode | Indexer scan over all cached positions, block max, top-k, gather the selected blocks and the local window from the cache, absorbed attention over about 3,072 positions (on CUDA: the fused kernels, which select on the device and index the selected blocks in place instead of gathering them) |
| Prefill chunk or training forward | Indexer scores for every query, selection as a `[T, S]` mask, masked attention over all `S` positions. Exact, but as costly as dense attention |

## Parameters

| Part | NCM v2 | Token baseline v2 |
|---|---|---|
| Embeddings (tied) | 16.4M | 32.8M |
| Local encoder | 10.5M | |
| Backbone / layers | 200.1M | 216.9M |
| Local decoder | 15.7M | |
| Draft head | 3.7M (3,674,112) | |
| Projections | 2.6M | |
| **Total** | **249.0M** (248,973,320) | **249.7M** (249,662,624) |

Per layer at d = 1024: a DeltaNet mixer is 5.27M, a retrieval mixer 5.97M (of which the indexer 0.66M). MLP widths fill the budget: `bb_ff_dim = 2240` for the NCM backbone (6.88M per MLP), `ff_dim = 1700` for the token baseline (5.22M). The encoder is 10.5M against the doc's ~9M because the DeltaNet layer comes with its own MLP and the sliding layers gained gates.

## Cache at 1M tokens (bf16, batch 1)

| Model | Retrieval (MLA) | Indexer keys, 8-bit | DeltaNet states | Encoder and decoder | Design total | Reference allocation (bf16 indexer keys) |
|---|---|---|---|---|---|---|
| NCM v2 | 8 × 262,144 × 1,152 B = 2.25 GiB | 0.25 GiB | 2.3 MiB | 3.0 MiB | **2.505 GiB** | 2.755 GiB |
| Token v2 | 10 × 1,048,576 × 1,152 B = 11.25 GiB | 1.25 GiB | 2.7 MiB | | **12.503 GiB** | 13.753 GiB |

DeltaNet states are kept at the run dtype in the reference; Qwen keeps them in f32, which would double that column (still under 10 MiB).

## Tests

`tests/attention.rs` (CPU, tiny configs): the chunked DeltaNet equals the recurrent one (ragged tails, chunk 1, chunk longer than the run); incremental decode equals a full forward pass for both v2 models, sparse and dense, across chunk, block and local-window boundaries; sparse retrieval with every block selected equals dense attention, and top-2 differs from it; NCM causality and conditioning; full-config parameter counts; cache bytes at 1M within 5% of the architecture doc; cache bytes equal the allocation. `tests/weights.rs` round-trips v2 weights. `ncm-server` has a v2 engine test.

## Fast decode on CUDA (`src/kernels`, `src/fused.rs`)

Batch-1 decode with the reference code is bound by the host issuing small tensor ops (about 10 µs each), not by the GPU: the reference v2 NCM issued 825 kernel launches per decoded token at 2k context. Three layers of work make v2 decode fast on CUDA, each behind the same model API and each with a switch; the CPU and CI run only the reference.

### Fused kernels

| Kernel | Replaces | Launches per layer and step |
|---|---|---|
| `delta_step_kernel` | Gated DeltaNet with a cache over up to `dn_chunk` tokens: short convolution, SiLU, L2 norms, gates, delta-rule update and read-out, per-head RMSNorm, output gate. One cube per (batch, head), one unit per state column; the tokens of a run go one by one inside the kernel, so the NCM encoder's 4-token call per concept is one launch too. Updates the cache's (packed) state and convolution history in place | 1 (+ the 2 projections) |
| `ret_prep_kernel` | Retrieval QK-norms, partial RoPE, the absorbed query `q_nope W_uk^T` (softmax scale folded in) and the new cache entry (latent, RoPE key, indexer key) written in place | 1 |
| `ret_index_kernel`, `ret_topk_kernel` | Indexer scores of every candidate block (block maximum), then the top `ret_topk` blocks by rank counting, on the device: no host round trip. Both do nothing on dense steps | 2 |
| `ret_attend_kernel`, `ret_combine_kernel` | Absorbed attention over the selected blocks and the local window (or the whole context), split into chunks of 64 positions with a partial softmax each, then merged, up-projected by `W_uv` and gated | 2 |
| `gemv_vec_kernel` (+ `gemv_sum_kernel`), `gemv_cols_kernel`, `gemv_rows_kernel` | Dense linear layers on at most four rows (`fused::linear`, called by every dense `quant::QLinear`: MLPs, attention, DeltaNet and retrieval projections, NCM pool and condition projection; `fused::linear_transposed`: the tied-embedding logits). Bandwidth-bound matrix-vector products. Row-major weights take `gemv_vec_kernel`: each unit loads 16 bytes of consecutive columns (8 or 4 when the width needs it), eight units stride `K` per column vector and sum in shared memory, and `K` is split over about two cubes per SM with an f32 partial sum; 600 to 860 GB/s at the 250M shapes on an RTX 5080, against 240 to 340 GB/s for the scalar `gemv_cols_kernel` it replaces (kept for widths no vector divides). Transposed weights (the logits) take `gemv_rows_kernel`, one warp per output, at about 850 GB/s | 1 or 2 |
| `qgemv_vec_kernel`, `qgemv_kernel`, `dequant_kernel` | Quantised (q8, q4) linear layers (`kernels::qmatmul`, called by quantised `quant::QLinear`s): a dequantise-and-multiply GEMV for up to eight rows, else the weight decoded for Burn's matmul; see `docs/quantisation.md` | 1 |
| `rms_norm_kernel` | RMSNorm in one launch (`fused::rms_norm`; `fused::add_rms_norm` adds the pending residual first): every block norm of the stacks and the local decoder, the final norms and QK-norm. Burn's `RmsNorm` is about six launches. The arithmetic mirrors it at the run dtype (`rms` rounded to the run dtype, then `F(F(s / rms) * gamma)`); only the order of the f32 sum differs. Each block hands its MLP output to the next block's first norm, which adds it to the residual stream in the same launch | 1 |
| `argmax_kernel` | The greedy pick (`fused::argmax_last`): Burn's reduction took about 0.4 ms per token over the 32k vocabulary | 1 |
| `assign_kernel`, `assign_at_kernel` | In-place writes into a buffer whoever else holds it (KV-cache rows, graph inputs), at an offset given as a launch argument or held in a device buffer (`kernels::assign_into_at`: graphed attention-cache writes); see below | 1 |

The retrieval kernels take the position from a device buffer (`LatentCache`'s `position`) and size their grids for the largest step of the cache: the candidate count, whether a step is sparse and which slots are visible are decided on the device, so every position issues the same launches.

### CUDA graphs

Burn 0.22 exposes CUDA graph capture (`Backend::graph_prepare`, `graph_start_capture`, `graph_stop_capture`, `graph_replay`, on cubecl's CUDA runtime). A graph replays the recorded launches against the exact buffers and scalar arguments of the capture, so a step is graphable only when its launches do not change from step to step:

- `ncm::GraphedStep`: a whole NCM concept step in one graph: `append_tokens` of the concept just written (encoder over its k tokens, the decoder layers' cross-attention keys, concept pooling, the backbone step, the condition) and then the greedy decoder pass of the next concept (k tokens through four layers, logits, argmax). No host work or sync inside a step; the written tokens are copied into a fixed buffer before each replay, so feeding the returned tokens back costs one device copy. Captured once the cross-attention ring is full; until then, or if the capture fails, `append_tokens` and `GraphedDecoder` run instead. Both designs. `ncm-bench` decodes the first concept from the prompt eagerly and every later one through it.
- `ncm::GraphedAppend`: the same graph without the decoder pass (written concept to condition), which self-speculative decoding uses between its draft and verify passes.
- `tok::GraphedTok`: the token baseline's greedy step (embedding, stack step, logits, argmax) in one graph, both designs, so `ncm-bench` compares like with like.
- `ncm::GraphedDecoder`: the NCM's local decoder alone, the eager fallback of `GraphedStep`. The per-concept self cache restarts at slot 0; the condition and the RoPE tables of the concept's positions are copied into fixed buffers before each replay.
- `speculative::Speculator`: the draft pass and the verify pass of self-speculative decoding, one graph each, captured once the ring is full ([below](#local-decoder-and-speculative-decoding-srcspeculativers)). Both designs.
- `layers::GraphedStack`: one-token steps of a stack (`NcmLm::append_tokens`' backbone step when the whole step is not graphed), both designs.

The stack steps inside these graphs are `layers::StackStep`s: the RoPE tables of the step's positions in fixed buffers, DeltaNet and retrieval layers as above, and attention layers through **cursors** (`graph.rs`). A cursor holds `[slot, position]` on the device for a group of attention caches that move together; the step writes its keys and values at `slot` (`assign_at_kernel`) and derives its mask from `position` with a few device ops instead of the eager path's host-built mask. One-token steps (backbones, the token baseline) attend over the whole cache buffer and hide the slots past the position (a ring that wrapped hides none); the NCM encoder's four-token runs over its full sliding-window rings attend over the ring before the write and their own keys, exactly as the eager `KvCache::append`, with the window mask `(position - 1 - slot) mod window >= window - 1 - i` for query i. Warm-up steps are undone before the capture (DeltaNet states restored, attention rings copied back, positions rewound).

Every per-step input is staged (`graph::Staging`): before a replay the host writes the RoPE tables, cursor positions and retrieval positions into one float and one int vector and uploads each in one copy, and the graph begins by copying the staged values into the buffers the layers read. A replay costs the host two uploads, one token copy and the launch (about 50 us for the token baseline), whatever the depth of the model; after it the caches' host positions advance.

What Burn's graph API requires, and how it is met here:

- **Fixed buffers.** A graph holds every buffer it touched, so Burn sees them as shared (`can_mut` false) and its `slice_assign` would copy a cache or input to a new buffer, leaving the graph reading the stale one. Cache rows, graph inputs and DeltaNet states are therefore written with `assign_kernel` and the in-place DeltaNet kernel, which write the buffer itself. The caller answers for other holders.
- **Fixed scalars and shapes.** Anything that varies per step must live in device memory: positions (device buffers), RoPE tables (copied in), selection and slot counts (computed on the device), grid sizes (the largest step).
- **No host-to-device copies inside a capture.** Uploads (the staged positions and RoPE tables) are issued before each replay; `fused::capturing` tells the layers to leave out their own while recording. Masks a captured pass needs are built before the capture (the verify pass's causal mask) or on the device from a cursor.
- **One thread.** cubecl gives every thread its own stream, and a capture that reads a buffer last written on another thread's stream is invalidated (`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`). The cached RoPE rotation matrix is therefore cached per thread. Capture and replay on the thread that built the model and its caches.
- **Not graphable as written:** prefill (multi-concept runs, retrieval over more than one token), the NCM encoder before its ring fills, and caches that cannot take device cursors (multi-token runs over growing caches). They run the fused kernels eagerly.

`fused::set_fused_kernels(false)` (`ncm-bench --reference-kernels`) turns the kernels and graphs off on CUDA; `fused::set_cuda_graphs(false)` (`ncm-bench --no-cuda-graphs`) keeps the kernels but runs them eagerly. Burn's fusion backend (`burn/fusion`) was measured and not adopted: it slowed the fallback NCM from 171 to 127 tok/s at 2k, and left the reference v2 at 84.

### Numerics and tests

- f32 inside every kernel. The DeltaNet state and the caches stay at the run dtype, as in the reference.
- `tests/kernels_f32.rs` and `tests/kernels_bf16.rs` (`cargo test --release -p ncm-model --features cuda --test kernels_f32 --test kernels_bf16`, a GPU and the CUDA toolkit needed) run each layer three ways from equal caches: fused on CUDA, reference on CUDA, and reference on the CPU in f32 with the same weights and inputs. The fused output must be within the tolerance of the f32 CPU result at every step, as a share of its largest magnitude: **5e-3 in f32 and 4e-2 in bf16** (the CUDA reference itself is off by up to about 2e-3 and 2.5e-2 away from selection near ties). Cases: DeltaNet at head widths 128, 32 and 16 with 1-, 3- and 4-token runs after a chunked prefill; retrieval with the models' head shapes and small blocks (dense steps, then sparse ones, crossing block and window boundaries), the tiny configs' shapes, dense mode, and the 250M selection (blocks of 64, top 32, window 1,024) past 3,150 positions. Batch 2 throughout. At a near tie for the last selected block, the CUDA paths (keys from the same GPU projection) may pick another block than the CPU; such a step passes when the fused output matches the CUDA reference instead, and at most one step in ten may. The 250M selection case runs in f32 only: with bf16 keys most of its steps are near ties.
- The GEMV kernels against an f32 CPU matmul (one to four rows, split and unsplit K, row-major weights at 16-, 8- and 4-byte vectors and transposed ones: about 1e-6 of scale in f32, 3e-3 in bf16); the norm kernel, plain and after a residual add, against Burn's `RmsNorm` on the CPU in f32 (the models' widths, QK-norm's per-head rows, a strided input), within the layer tolerances; a `GraphedStep` kept across `NcmCache::reset` captures anew and decodes the same tokens; the argmax kernel against Burn's `argmax` (ties to the first index); graphed decode against the same kernels run eagerly, token for token, for 12 concepts of both tiny NCM designs (the v2 one turning sparse within the run): the decoder graph with the backbone graph, the whole concept step (dense and quantised; the fallback backbone's sliding rings wrap and its full layers grow inside the graph), the append graph's conditions (equal to the eager ones), and the token baseline's graphed step for 40 tokens of both tiny designs (dense and quantised).

### Effect

RTX 5080, bf16, batch 1, short `ncm-bench` runs (128 new tokens) on a busy CPU; decode tok/s of the NCM v2:

| Step | 2k | 16k | Host share (2k) |
|---|---|---|---|
| Reference | 89.0 (825 launches per token) | 12.3 | 100% |
| Fused DeltaNet and retrieval kernels | 213.6 (507 launches per token) | 155.2 | 100% |
| + decoder graph, argmax kernel | 305.4 | 206.8 | 97% |
| + v2 backbone graph | 331.1 (194.3 with `--no-cuda-graphs`) | | 90% |
| + GEMV kernels | **453.3** | **375.9** | 93% |

The fallback NCM, which shares the decoder, the GEMV kernels and the argmax, made 251.4 tok/s at 2k in the last run.

**Whole-step graphs.** Short dev runs (2k context, 256 new tokens, 3 repeats, bf16, random weights; indicative, not the gate): NCM v2 605 tok/s (479 in the last gate with the decoder and backbone graphs only; 198 with `--no-cuda-graphs`), fallback NCM 711 (408); token baseline v2 240 (77.7 ungraphed in the last gate; 147 with `--no-cuda-graphs` now that its logits take the GEMV kernel like the NCM's), fallback 259 (111.8). The capture (two warm-up steps and the recording) fell inside the timed decode of these runs. Decode is now bound by the GPU, not the host: Nsight Systems puts most of a step in the GEMV kernels (about a third) and the retrieval kernels (`ret_attend` about 85 us and `ret_combine` about 47 us per layer step at 2k).

**Vectorised GEMV and fused norms.** `ncm-bench` now captures the decode graphs before the clock starts (`prepare`) and times the steps after it, so the figures below exclude the capture. Same dev runs (2k, 256 new tokens, 3 repeats, two rounds, bf16), before (the kernels above) and after:

| Model | Before | After |
|---|---|---|
| NCM v2 | 652 | 962 |
| NCM v2, q4-g64 | 920 | 1123 |
| Token baseline v2 | 227 | 312 |
| NCM fallback | | 1303 |
| Token baseline fallback | | 380 |

NCM/token at 2k: 3.08x in v2, 3.43x in the fallback design. GPU time of a whole run (prefill included), after: NCM v2 retrieval kernels 29%, GEMV 27%, prefill matmuls 14%, norms 5.5%, DeltaNet 4.6%; token v2 retrieval 55%, GEMV 22%, DeltaNet 6%, norms 3%. The retrieval attention (`ret_attend`, `ret_combine`, together about 135 us per layer step at 2k) is now the largest share.

## Local decoder and speculative decoding (`src/speculative.rs`)

The local decoder writes a concept token by token: k = 4 passes of its four layers, each waiting for the token before. Self-speculative decoding replaces them with two parallel passes per concept:

1. **Draft pass.** The draft head (`DraftHead`, `draft_layers` blocks at the decoder width) proposes all k tokens at once. Its input at slot j is the concept's condition plus a learned position query for slot j (`slots`, `[k, d]`). Each block has self-attention between the k slots (bidirectional: no slot holds a token), cross-attention over the encoder states and a SwiGLU MLP; then its own RMSNorm and the tied embedding. Block i has its own queries and output projection (`layers::CrossQuery`) but reads the keys and values of decoder layer i's cross-attention cache, so it needs no cache or key-value projection of its own (`draft_layers <= dec_layers`). 3.67M parameters for one block at d = 512; `draft_layers = 0` builds no head (configs written before it).
2. **Verify pass.** The decoder runs once over the drafted concept, teacher-forced and causal within the concept (`NcmLm::verify_concept`), exactly as the training forward does: row j is the decoder's distribution for token j given the drafted tokens before it. The decoder's caches are only read.
3. **Accept.** The longest prefix where the draft equals the decoder's argmax is kept; the first mismatch is replaced by the decoder's argmax. If positions remain, the repaired guess goes through another verify pass. Every pass settles at least one token, so a concept takes 1 to k verify passes (one plus one per rejected slot before the last).

Greedy output equals plain greedy decoding: a token is kept only where its whole prefix is settled and it is the decoder's argmax. The parallel pass rounds differently from the step-by-step one, so at an exact near tie of the two best logits the two may differ; the GPU tests find no difference in f32 and count differences in bf16. For sampling, `Speculator::decode_sampled` applies standard speculative acceptance: a drafted token x is kept with probability min(1, p(x)/q(x)), a rejected one is redrawn from the normalised max(0, p - q), with p and q the decoder's and the draft head's distributions after the caller's transform (temperature, top-p, ...). Because the draft head's slots do not depend on earlier tokens, the drafted tokens after a correction remain valid proposals. `ncm-server` does not use speculation yet.

**Training hook.** `NcmLm::forward_with_draft` returns the decoder's logits and the draft head's logits over the same positions, sharing the encoder and backbone; `speculative::draft_loss` is the cross-entropy of the draft logits, overall and per slot. The training crate adds it (weighted) to the decoder loss. Its gradient reaches the decoder layers' cross-attention key-value projections, which the head shares.

**Execution.** One device-to-host transfer per verify pass (the first also brings the draft). On CUDA with the fused kernels and graphs on, the draft and verify passes are captured once the cross-attention ring is full and replayed per concept: the condition and RoPE tables are copied in before the draft replay and the guess before each verify replay. Without CUDA, or before the ring fills, the same passes run eagerly. Under real verification the last verify pass's output buffer is the concept, so no upload follows. Appending each written concept replays `ncm::GraphedAppend` (encoder, cross-attention keys, pooling and backbone step in one graph).

**Tests.** `tests/speculative.rs` (CPU, tiny models of both designs): speculative greedy equals plain greedy over 10 concepts for 6 seeds, and with forced drafts that are right everywhere, right up to slot 2 or 1, or wrong everywhere (also checking the verify-pass count); assumed acceptance gives the rate and pass count it should; speculative sampling with one-hot distributions equals greedy, and accept-or-resample draws reproduce the target distribution (200,000 draws); the verify and draft passes equal the training forward's logits; `draft_loss` against a hand computation; the head is optional. On the GPU (`kernels_f32.rs`, `kernels_bf16.rs`): the graphed draft and verify passes give the same tokens and pass counts as the same kernels run eagerly, under real verification and an assumed rate of 0.5, and in f32 the same tokens as plain greedy decoding through the decoder graph.

**Benchmark.** `ncm-bench --speculate on` adds, per context, a speculative run with real verification and one per `--assume-accept` rate. An assumed rate keeps that share of drafted tokens whatever the decoder says (decision n keeps when frac(n (sqrt 5 - 1)/2) is below the rate), so it times the passes and transfers a trained model with that acceptance would run; its tokens are not the model's output and the report labels it SYNTHETIC. Random weights accept few drafts (2 to 18% at 250M), so the real run measures little more than the worst case. RTX 5080, bf16, v2 NCM, 256 new tokens (2k: median of 3; 16k: one run):

| Run | 2k tok/s | 16k tok/s |
|---|---|---|
| Plain (decoder graph) | 499.8 | 474.8 |
| Speculative, real verification (random weights: 3.5% and 18% kept, 3.9 and 3.5 passes per concept) | 462.4 (0.93x) | 463.4 (0.98x) |
| Speculative, assumed acceptance 1.0 (1 pass per concept) | 585.9 (1.17x) | 585.1 (1.23x) |
| Speculative, assumed acceptance 0.75 (1.7 passes) | 540.3 (1.08x) | 553.0 (1.16x) |
| Speculative, assumed acceptance 0.5 (2.5 passes) | 511.4 (1.02x) | 502.8 (1.06x) |

The fallback NCM at 2k: 430.0 plain, 486.8 (1.13x) at acceptance 1.0, 458.0 at 0.75, 407.9 at 0.5. The gain is bounded because greedy decode is host-bound outside the decoder (95% host share): the plain decoder already replays as one graph whose GPU time overlaps the host issuing the next concept's encoder and backbone work, which speculation does not touch, while each verify pass adds a host sync. Runtimes that sample on the host save more: plain decoding syncs once per token there, speculation once per verify pass.

## Known slow paths

- **Chunked DeltaNet (prefill):** the chunk loop runs on the host, about seven tensor ops per chunk per layer; the intra-chunk inverse is six batched matmul squarings.
- **Sparse prefill:** prefill and training compute the selection as a mask over the whole context and then attend densely, so they cost as much as dense attention; a block-sparse kernel is needed to make them sparse in compute. A 16k-token NCM v2 prefill takes about 28 s against 1 s for the fallback.
- **Indexer keys** are read at bf16, twice the design's traffic.
- **Reference decode** (CPU, and CUDA with the kernels off): about a dozen small ops per DeltaNet step, the indexer scan as a matmul plus pad, block max, top-k and a gather, and the absorbed projections rebuilt from the weights on every call.

## Not yet implemented

- The indexer's training objective (DeepSeek-V3.2 matches the indexer to the dense attention distribution with a KL loss). Top-k selection passes no gradient to the indexer, so the dense warm-up must add that loss in the training crate.
- 8-bit storage of indexer keys.
- YaRN scaling of the retrieval RoPE for 1M contexts.
