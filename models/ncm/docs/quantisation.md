# Weight-only quantisation

Both models, in both attention designs, can run their linear layers on 8-bit or 4-bit weights with bf16 (or f32) activations: W8A16 and W4A16. Batch-1 decode reads every weight once per token, so smaller weights mean less memory traffic per token once decode is no longer bound by host dispatch. The code is `crates/ncm-model/src/quant/`.

| `--quant` | Codes | Scales | Bits per weight (with scales) |
|---|---|---|---|
| `none` (default) | dense at the run dtype | | 16 (bf16) |
| `q8` | int8, symmetric (`absmax / 127`) | one f32 per output channel | about 8 |
| `q4` | 4-bit, all 16 levels (the group's largest-magnitude value maps to -8, as llama.cpp's Q4_0) | one f32 per `--quant-group` inputs (64, default, or 128) | 4.5 (g64), 4.25 (g128) |

## What is quantised

Every linear layer of the four stacks: attention q/k/v, output and gate projections, cross-attention, the gated DeltaNet and sparse-retrieval input and output projections, every SwiGLU MLP, and the NCM's pooling and conditioning projections. They are `quant::QLinear`, dense unless the model is built quantised.

Kept dense: the embedding table, norms, the DeltaNet convolution and per-head gate parameters, and the retrieval layers' latent up-projections `uk` and `uv` (the absorbed attention reads them as matrices, and the fused retrieval kernels take them dense). The tied output head is quantised by default as a separate quantised copy of the embedding table (`head`); the table itself stays dense for lookups. `--dense-head` keeps the head on the embedding table.

## Format

A quantised layer stores, in place of `weight` `[d_in, d_out]`:

* `qweight` `[d_out, words]`, i32: each output row's codes packed little end first, 4 (q8) or 8 (q4) per word. Rows are padded with zero codes to a multiple of the group (q4) or of 64 (q8), so a word never straddles two groups and ragged widths such as the token baseline's 1700-wide MLP work.
* `scales` `[d_out, groups]`, f32 (one group per row for q8).

A code `u` decodes to `(u - 2^(bits-1)) * scale`. Quantisation happens on the host from the dense weights (`quant::format`), so it is identical on every device.

## Kernels (CUDA)

On a `--features cuda` build the quantised layers multiply through two CubeCL kernels (`src/kernels/quant.rs`, launched by `kernels::qmatmul` through the same Burn backend extension as the other fused decode kernels, see `docs/attention.md`):

* **GEMV** (up to 8 rows, `quant::GEMV_ROWS`: decode steps, the NCM's per-concept calls): one warp per output feature streams that feature's packed row once, each lane decoding its words in registers and accumulating every input row against them in f32; a warp sum finishes each dot product. The weight is read once per call whatever the row count.
* **GEMM fallback** (more rows: prefill and training-length runs): a kernel decodes the weight to `[d_in, d_out]` at the run dtype and Burn's tuned matmul multiplies. Prefill is compute-bound, so the extra weight write and read cost little there.

Everywhere else (CPU, CI, wgpu, with the fused kernels off by `fused::set_fused_kernels(false)` or `ncm-bench --reference-kernels`, or with `quant::set_quant_kernels(false)`) the reference path decodes the weight with tensor ops (`quant::dequantize`) and multiplies.

`QLinear::forward` is the one linear path of both models: dense, it multiplies through `fused::linear` (the dense GEMV kernel for up to four rows on CUDA, Burn's matmul otherwise); quantised, through `kernels::qmatmul` as above. The tied output head (quantised copy or embedding table) is shared by the decoder and the draft head.

**CUDA graphs.** The kernels only read their codes and scales, which are parameters and keep their buffers, and index every operand through its strides (cubecl pitches the rows of new 2-D buffers), so the captured decode graphs (NCM decoder, v2 backbone step, draft and verify passes of self-speculation) record and replay quantised layers as they do dense ones; `ncm-bench --quant q8|q4` runs them graphed by default. Short runs on an RTX 5080 (`--model ncm --design v2 --device cuda --context 2048 --new-tokens 128`, bf16, graphs on, no speculation): about 410 to 450 tok/s dense, 480 q8, 495 q4-g64, all at a 92% host share, so decode remains bound by host dispatch outside the graphs and the saved weight traffic shows only as a modest gain.

One-row products on an RTX 5080 in bf16 (`tests/quant_cuda_bf16.rs`, `gemv_timing_bf16`, release build, calls queued back to back):

| Layer `d_in -> d_out` | dense (Burn matmul) | q8 GEMV | q4-g64 GEMV |
|---|---|---|---|
| 1024 -> 4112 (a DeltaNet input projection) | 21 us | 9.5 us | 12 us |
| 4096 -> 16384 (fits the 64 MiB L2 when quantised) | 337 us | 46 us | 39 us |
| 8192 -> 24576 (exceeds L2 in every format) | 1210 us, 333 GB/s | 216 us, 932 GB/s | 145 us, 780 GB/s |

The 250M models' layers are small (4 to 9 MiB dense), so each call is launch-bound and whole-model decode stays host-dispatch-bound until decode runs as a CUDA graph; the table is the per-layer headroom that quantisation then releases. Burn's dense one-row matmul streams weights at well under the card's bandwidth, so even q8 is several times faster per layer than the dense path.

## Loading and files

* **Random weights** (`ncm-bench`, `ncm-server --random-weights`): `quant::build(spec, || Model::new(...))` builds every `QLinear` quantised from the usual N(0, 0.02) init.
* **Dense weight file** with `--quant q8|q4` (`ncm-server`): the file loads dense, then `quant::quantize_model` quantises it on the host.
* **Quantised weight file**: `quant::save_quantized` writes the weight-file format (see `runtime/README.md`) with `<layer>.qweight` and `<layer>.scales` in place of `<layer>.weight` (and `head.qweight`, `head.scales` for a quantised head) and adds `ncm_quant` to the header (`q8`, `q4-g64`, with `+head` when the head is quantised). `quant::load_quantized` reads the spec from the header, builds the model quantised and loads the tensors; floats other than the scales are cast to the run dtype, scales stay f32. `ncm-server` loads such a file as it is; `--quant` must then be `none` or name the same spec. A quantised file does not load as dense or the other way round.

## Accuracy

Tiny models (CPU, f32, fixed uniform weights with the init's standard deviation; `tests/quant.rs`), quantised logits against the dense model's over 16 tokens, error relative to the largest dense logit:

| Model | q8 max (mean abs) | q8 argmax | q4-g64 max (mean abs) | q4-g64 argmax |
|---|---|---|---|---|
| tok fallback | 0.45% (0.0009) | 100% | 5.9% (0.014) | 94% |
| tok v2 | 0.47% (0.0008) | 100% | 6.9% (0.015) | 100% |
| ncm fallback | 0.66% (0.0006) | 100% | 14% (0.012) | 92% |
| ncm v2 | 0.79% (0.0006) | 100% | 11% (0.009) | 92% |

Quantising the head adds little (q8 at most 0.85%). Untrained models have flat logits, so argmax agreement under q4 says little about trained weights; the q4 numbers are for the record and want re-measuring on trained checkpoints (perplexity, not logits). On CUDA the kernels agree with the tensor-op reference to 1e-3 of the largest logit in f32 and within bf16 noise in bf16 (`tests/quant_cuda_*.rs`).

## Tests

`tests/quant.rs` (CPU, CI): the layer against a host product of the decoded weight for ragged widths and the GEMV and GEMM row counts; dense builds keep the dense format; tiny models in both designs against their dense weights; quantised random models run; quantised files round trip exactly, record their spec and refuse to mix with dense files; conversion refuses another architecture. `src/quant/format.rs` unit tests: rounding bounds, padding, exact values, zero rows. `tests/quant_cuda_f32.rs` and `tests/quant_cuda_bf16.rs` (`--features cuda`, a GPU): kernels against an f64 host product for the 250M shapes, every format and row count; tiny models (full forward and token-by-token decode) against the reference path. `tests/kernels_f32.rs` and `tests/kernels_bf16.rs` (`--features cuda`, a GPU): with q8 and with q4-g64 (head quantised), graphed decode and graphed draft and verify passes give the same tokens (and pass counts) as the same quantised kernels run eagerly, in both NCM designs. `runtime/tests/engine.rs`: an engine quantising a dense file at load equals one loading the saved quantised file.
