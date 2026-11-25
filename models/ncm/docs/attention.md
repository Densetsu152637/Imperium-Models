# Attention designs

Both models (the NCM and the token baseline) come in two attention designs, picked by the config's `attention` key:

| `attention` | Design | Configs |
|---|---|---|
| `"v2"` | Gated DeltaNet alternating with sparse retrieval attention (adopted 2026-11-25) | `configs/ncm-250m-v2.toml`, `configs/tok-250m-v2.toml` |
| `"fallback"` (default) | Sliding-window attention with a full-attention layer every fourth (the stage 1 design) | `configs/ncm-250m.toml`, `configs/tok-250m.toml` |

The design itself is settled in Imperium-Research (`docs/next-concept-modelling/architecture.md`). This page records how `ncm-model` implements v2 and the choices made where the design leaves details open. The code is the **reference** version: plain Burn tensor ops, written for correctness. Fused kernels come later (local test plan, step 3).

v2 settings live in the `[v2]` table of a config (`V2Config`); fallback models ignore them. A weight file's header carries the whole config, so `ncm-server` loads either design from the weights alone.

## Layer layout

| Model | Stack | Layers |
|---|---|---|
| NCM | Local encoder, d = 512 | sliding window (512 tokens), gated DeltaNet, sliding window |
| NCM | Global backbone, d = 1024, over concepts | 16: DeltaNet, retrieval, DeltaNet, retrieval, ... (8 + 8) |
| NCM | Local decoder, d = 512 | unchanged from the fallback: 4 layers, self-attention over the concept, cross-attention to the last 512 encoder states |
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
| One-token decode | Indexer scan over all cached positions, block max, top-k, gather the selected blocks and the local window from the cache, absorbed attention over about 3,072 positions |
| Prefill chunk or training forward | Indexer scores for every query, selection as a `[T, S]` mask, masked attention over all `S` positions. Exact, but as costly as dense attention |

## Parameters

| Part | NCM v2 | Token baseline v2 |
|---|---|---|
| Embeddings (tied) | 16.4M | 32.8M |
| Local encoder | 10.5M | |
| Backbone / layers | 200.1M | 216.9M |
| Local decoder | 15.7M | |
| Projections | 2.6M | |
| **Total** | **245.3M** (245,299,208) | **249.7M** (249,662,624) |

Per layer at d = 1024: a DeltaNet mixer is 5.27M, a retrieval mixer 5.97M (of which the indexer 0.66M). MLP widths fill the budget: `bb_ff_dim = 2240` for the NCM backbone (6.88M per MLP), `ff_dim = 1700` for the token baseline (5.22M). The encoder is 10.5M against the doc's ~9M because the DeltaNet layer comes with its own MLP and the sliding layers gained gates.

## Cache at 1M tokens (bf16, batch 1)

| Model | Retrieval (MLA) | Indexer keys, 8-bit | DeltaNet states | Encoder and decoder | Design total | Reference allocation (bf16 indexer keys) |
|---|---|---|---|---|---|---|
| NCM v2 | 8 × 262,144 × 1,152 B = 2.25 GiB | 0.25 GiB | 2.3 MiB | 3.0 MiB | **2.505 GiB** | 2.755 GiB |
| Token v2 | 10 × 1,048,576 × 1,152 B = 11.25 GiB | 1.25 GiB | 2.7 MiB | | **12.503 GiB** | 13.753 GiB |

DeltaNet states are kept at the run dtype in the reference; Qwen keeps them in f32, which would double that column (still under 10 MiB).

## Tests

`tests/attention.rs` (CPU, tiny configs): the chunked DeltaNet equals the recurrent one (ragged tails, chunk 1, chunk longer than the run); incremental decode equals a full forward pass for both v2 models, sparse and dense, across chunk, block and local-window boundaries; sparse retrieval with every block selected equals dense attention, and top-2 differs from it; NCM causality and conditioning; full-config parameter counts; cache bytes at 1M within 5% of the architecture doc; cache bytes equal the allocation. `tests/weights.rs` round-trips v2 weights. `ncm-server` has a v2 engine test.

## Known slow paths (for fused kernels, step 3)

- **Chunked DeltaNet:** the chunk loop runs on the host, about seven tensor ops per chunk per layer; the intra-chunk inverse is six batched matmul squarings.
- **Recurrent DeltaNet step:** about a dozen small ops per layer per token, so decode is host-dispatch bound.
- **Indexer scan and top-k:** the scan is a matmul over every cached key per indexer head, then pad, block max, top-k, index arithmetic and a gather of the selected blocks. Keys are read at bf16, twice the design's traffic.
- **Sparse prefill:** prefill and training compute the selection as a mask over the whole context and then attend densely, so they cost as much as dense attention; a block-sparse kernel is needed to make them sparse in compute.
- **Absorbed projections** are rebuilt from the weights (reshape and permute) on every call.

## Not yet implemented

- The indexer's training objective (DeepSeek-V3.2 matches the indexer to the dense attention distribution with a KL loss). Top-k selection passes no gradient to the indexer, so the dense warm-up must add that loss in the training crate.
- 8-bit storage of indexer keys.
- YaRN scaling of the retrieval RoPE for 1M contexts.
