# ncm-server

An OpenAI-compatible HTTP server over the NCM and the token baseline (`ncm-model`, Burn 0.22), started and spoken to like llama.cpp's `llama-server`. It is stage 8 of the NCM plan, started early: it works today with random weights and loads trained weights once they exist.

```bash
# CPU (what CI builds)
cargo run --release -p ncm-server -- --arch ncm --random-weights -c 4096 --port 8080
# GPU: fused CUDA kernels and one captured CUDA graph replay per decode step (see "Decode paths")
cargo run --release -p ncm-server --features cuda -- --arch ncm --random-weights -c 4096 --port 8080
# ... with the v2 design, 4-bit weights and self-speculative greedy decoding (needs a draft head)
cargo run --release -p ncm-server --features cuda -- --arch ncm --random-weights \
    --config configs/ncm-250m-v2.toml --quant q4 --speculate -c 4096 --port 8080
# Trained weights (config in the file's header; tokenizer.json beside it or --tokenizer)
ncm-server --model weights/ncm-250m.safetensors --tokenizer tokenizer.json -c 1000000 --port 8080
curl localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
```

## What Imperium needs (the `llama-server` contract)

Imperium (`imperium-local`) starts `llama-server --model <path> --host 127.0.0.1 --ctx-size <total> --parallel <slots> --n-gpu-layers <n> --cache-type-k <t> --cache-type-v <t> --jinja [--mmproj <path>] <extra args> --port <port>`, polls `GET /health` until it answers 2xx (503 while loading; it gives up if the process exits), then talks to `<base>/v1`. The agent loop streams `POST /v1/chat/completions` (`stream: true`, `stream_options.include_usage`, `tools`) and reads `delta.content`, `delta.tool_calls[]` (`index`, `id`, `function.name`, `function.arguments`) and the final `usage` chunk (`prompt_tokens`, `completion_tokens`, `prompt_tokens_details.cached_tokens`). For context breakdowns it also calls `POST /tokenize` (`content`, `add_special`, `parse_special`), `POST /apply-template` (`messages`, `tools` to `prompt`) and `GET /props` (`default_generation_settings.n_ctx`); all three are optional on Imperium's side.

| Endpoint | Status |
|---|---|
| `GET /health` | 200 `{"status":"ok"}` once loaded; 503 while loading. The listener opens at once, the model loads in the background; a failed load prints the reason and exits non-zero. |
| `POST /v1/chat/completions` | Non-streaming and SSE streaming, `tools`, `tool_choice: "none"`, `max_tokens` / `max_completion_tokens` / `n_predict`, `temperature`, `top_p`, `top_k`, `seed`, `stream_options.include_usage`. Other fields are accepted and ignored. Errors use llama-server's `{"error":{"code","message","type"}}`; a prompt over the context is a 400 `exceed_context_size_error` with `n_prompt_tokens` and `n_ctx`. |
| `GET /v1/models` | The one model (id is `--alias`, default the weight file's name). |
| `GET /props` | `default_generation_settings.n_ctx` (per request), `total_slots`, plus an `ncm` object: arch, parameters, dtype, device, `cache_bytes`, tokens per pass. |
| `POST /tokenize`, `/detokenize`, `/apply-template` | As above. |

Not implemented: `stop` strings, penalties, `response_format`, images (`--mmproj` is ignored with a note), `/slots`, `/metrics`, an API key, `tool_choice` other than `none`/auto.

## Decode paths

Every request resets or restores the caches (prefix reuse, below), prefills the rest of its prompt and decodes one of four ways (`timings.decode_path`). All of a model's GPU work runs on one engine thread, whatever thread the HTTP layer serves a request on: cubecl gives each thread its own stream, and the kept graphs replay in order only on the thread that captured them (a replay on another thread panics).

| Path | Requests | How |
|---|---|---|
| `graphed-greedy` | CUDA (fused kernels and graphs on), greedy (`temperature` 0), no tool-call constraint | NCM: `ncm_model::ncm::GraphedStep`, one CUDA graph replay per concept that appends the concept just written (encoder, cross-attention keys, pooling, backbone) and decodes the next (decoder, argmax); the first concept, from the prompt, runs eagerly. Token baseline: `ncm_model::tok::GraphedTok`, one replay per token. |
| `graphed-sampled` | The same, sampled | The same graphs with the pick replaced by the device sampler (`ncm_model::sample`): temperature, top-k and top-p as the host sampler keeps them, drawn by the Gumbel-max trick from a counter-based hash of the request's `seed` (a fresh one when none is given). A seed reproduces its reply on the device; it is not the host sampler's random stream. |
| `speculative` | `--speculate`, CUDA, an NCM with a draft head, greedy, unconstrained | Draft and verify passes per concept (`ncm_model::Speculator`, graphed), each written concept appended through `GraphedAppend`. Equal to plain greedy in f32; in bf16 a near tie of the two best logits may go the other way, hence off by default. |
| `eager` | Tool-call constrained requests (the grammar runs on the host), the CPU, CUDA with graphs off | The host picks every token from the logits (`sampling.rs`). |

Each graph is captured once, on the first request that can use it (for the NCM, once the cross-attention ring is full; about 40 to 80 ms at the 250M size, more on the very first request, which also compiles the kernels), and kept: resetting and restoring the caches writes into their own buffers on CUDA (DeltaNet states, attention rings, retrieval latents and 8-bit indexer keys), so later requests replay the same graph (`graph_capture_ms` 0). A kept NCM graph waits until a request's cross-attention ring is full again before it serves. Each step reads its token or concept back to stream it, and a concept cut by the end-of-concept token is padded on the host before it is appended, as on the eager path.

**Prompt-prefix reuse** (on by default, `--no-prefix-reuse` turns it off). The engine keeps two snapshots of the cache from the previous request (DeltaNet states and attention rings copied; append-only storage only remembers its length): right after the prompt's prefill, with the prompt's output, and at the end of the reply, holding the prompt and every token fed back. A request whose prompt starts with either restores the longer one and prefills only the new tokens; the same prompt again prefills nothing. Agent loops hit the reply snapshot: the next turn's prompt repeats the previous prompt and reply and appends the tool's answer. For the NCM the snapshots hold whole concepts (a pending partial concept is prefilled with the new tokens), so any extension reuses the previous prompt's whole concepts; the reply snapshot serves while the next prompt repeats the reply token for token, which a concept cut short by `<|eoc|>` (fed back padded) breaks. The token baseline reuses the prompt and all but the last reply token. `timings.cache_n` and `usage.prompt_tokens_details.cached_tokens` count the reused prompt tokens, `timings.prompt_n` the prefilled ones. Short runs (RTX 5080, bf16, a 2k-token chat prompt extended by about 220 tokens, warm): prefill 17 ms with reuse against 49 ms without for the NCM v2, 45 against 103 ms for the token baseline v2.

Short runs on an RTX 5080 (random weights, bf16, a 2,048-byte chat prompt, 256 tokens streamed, `predicted_per_second` of later requests): NCM v2 about 1,230 greedy and 1,080 sampled, with `--quant q4` about 1,430 and 1,400; token baseline v2 about 515 greedy and 495 sampled. `ncm-bench` measured 1,244, 1,588 and 564 greedy on the same card earlier.

## Command line

`--model/-m`, `--host`, `--port` (0 picks a free one), `--ctx-size/-c` (total, allocated at launch; each request gets `ctx / parallel`), `--parallel/-np` (slots decoding together, below), `--n-gpu-layers/-ngl` (0 means CPU), `--alias/-a`, `--temp`, `--top-k`, `--top-p` (llama-server's defaults: 0.8, 40, 0.95) are llama-server's. `--cache-type-k/-v`, `--jinja`, `--threads`, `--batch-size`, `--flash-attn`, `--mlock`, `--no-mmap` and a few more are accepted and ignored so the launch line works unchanged (`-ngl`, `-np`, `-fa`, `-ctk`, `-ctv`, `-ub`, `-tb`, `-cb` are mapped to their long forms). NCM-specific: `--arch tok|ncm`, `--random-weights` (+ `--weights-seed`), `--tokenizer`, `--config`, `--device cpu|cuda`, `--dtype bf16|f32`, `--prefill-chunk` (default 8192 for the v2 design, whose fused prefill kernels want long chunks; 512 for the fallback), `--quant none|q8|q4` (+ `--quant-group`, `--dense-head`; weight-only quantisation at load, a dense bf16 file converted as it loads, see `../docs/quantisation.md`), `--speculate` (self-speculative decoding, below). Defaults: CUDA with bf16 when built with `--features cuda` and `-ngl` is not 0, else CPU with f32.

The architecture and config come from `--config`, else the weight file's header, else `config.toml` beside the weights, else the 250M config of `--arch` (the fallback attention design; pass `--config configs/ncm-250m-v2.toml` or `tok-250m-v2.toml` for v2, see `../docs/attention.md`). The tokenizer is `--tokenizer`, else `tokenizer.json` beside the weights; with `--random-weights` and none, a built-in byte-level tokenizer (ids 0 to 10 are the special tokens, 11 to 15 reserved, 16 to 271 bytes) is used.

## Context and cache

The key/value caches are allocated at launch for the per-request context, rounded down to whole concepts for the NCM. `-c 1000000` allocates the 1M-token caches the architecture doc sizes (`ncm_model::cache_bytes`; about 1 GiB for the NCM and 5 GiB for the baseline in bf16); the launch line reports the bytes, `/props` and every response's `timings.cache_bytes` repeat them. Requests reset or restore the caches (prefix reuse); generation ends with `finish_reason: "length"` at the end of the context.

## Weight format

One safetensors file, written by `ncm_model::weights::save_weights` and read by `load_weights` (Burn's `SafetensorsStore`). Tensor names are Burn module paths, so a file loads only into the architecture and config it was saved from; a missing, extra or mismatched tensor is an error. The header's `__metadata__` holds `ncm_format = "ncm-safetensors-v1"` and `ncm_config` (the model config as TOML), so a file says which model it is. Tensors keep their saved dtype (bf16 from training); loading casts floats to the run's dtype (f32 on CPU, bf16 on CUDA). `save_weights` is for the training code to call. A quantised file (`ncm_model::quant::save_quantized`) holds `qweight` and `scales` in place of each quantised layer's `weight` and adds `ncm_quant` to the header; the server loads it as it is (`--quant` must then be `none` or the file's own spec), and `--quant` on a dense file quantises it at load (`../docs/quantisation.md`).

## Chat format

Identical to `ncm-sft`'s `format.rs` (stage 3), which the models are fine-tuned on:

```text
<|im_start|>system
{system}

<tools>
{"name":"get_time","description":"...","parameters":{...}}
</tools>
<|im_end|>
<|im_start|>user
{text}
<|im_end|>
<|im_start|>assistant
<think>
{reasoning}
</think>
{text}
<tool_call>
{"name":"get_time","arguments":{"tz":"UTC"}}
</tool_call>
<|im_end|>
<|im_start|>tool
<tool_response>
{result}
</tool_response>
<|im_end|>
<|im_start|>assistant
```

Tool definitions go in the system turn (one compact `{"name","description","parameters"}` line each between `<tools>` and `</tools>`; with no system message the turn holds only that block). A request ends with `<|im_start|>assistant\n`. Reserved tags written inside message content are defused (`< |im_end|>`) so a user cannot end a turn. Generated output is parsed from token ids: `<think>…</think>` becomes `reasoning_content` (the field llama-server uses; Imperium does not read it yet), `<tool_call>` JSON becomes `tool_calls` (`finish_reason: "tool_calls"`; ids are `call_<request>_<index>`), and leading and trailing whitespace of each segment is dropped. A tool call whose JSON does not parse, or that is cut off, is returned as content text. Tool calls are parsed only when the request has tools. Generation stops at `<|im_end|>`, `<|endoftext|>` or `<|im_start|>`.

## NCM specifics (assumptions to confirm against the trained model)

- The decoder writes `k` tokens per pass. Output is streamed per concept. A `<|eoc|>` ends a concept early: tokens after it are dropped, and the concept is fed back as the tokens up to `<|eoc|>` followed by `<|pad|>`.
- Concepts are fixed `k`-token chunks from the start of the token stream, as in pretraining. The prompt's whole concepts are prefilled; its trailing partial concept (fewer than `k` tokens) stays pending and is forced, teacher-forced as in training, as the first tokens of the first decoded concept, which the decoder completes (those forced tokens are prompt, not output). Every path handles it: eager, graphed (the first concept is decoded eagerly anyway) and speculative (a first concept with pending tokens is decoded plainly). A prompt shorter than one concept is left-padded with `<|pad|>` to one.
- Sampling inside a concept uses `NcmLm::decode_concept_with`, which hands each position's logits to a callback (greedy unconstrained decoding stays on the device).

## Metrics

Every response has llama-server's `timings` object: `prompt_n`, `prompt_ms`, `prompt_per_second` (prompt processing) and `predicted_n`, `predicted_ms`, `predicted_per_second` (decoding), timed separately with a device sync, plus `cache_bytes`, `graph_capture_ms` (capturing the request's decode graph, counted in neither `prompt_ms` nor `predicted_ms`) and `decode_path` (below). Tokens are reference-tokenizer tokens (the NCM's `k` per pass are counted individually). `predicted_per_second` excludes the token baseline's first token, which comes out of the prompt pass. The streaming finish chunk carries `timings` and the usage chunk (with `include_usage`) repeats them.

## Grammar-constrained tool calls (R2.4)

The hook is `constrain::Constraint`: `allows(token)` is asked for the candidates the sampler tries, best logit first, and `advance(token)` follows each choice; no vocabulary-sized mask is built. `ToolCallGuard` is on whenever a request has tools: free text is unconstrained, and after `<tool_call>` every token must keep the text a prefix of one JSON object (a byte-level parser, `JsonState`), `</tool_call>` is allowed only once the object is closed, and special tokens and long runs of whitespace outside the object are refused, so a tool call is always syntactically valid JSON. With constraints active sampling looks at the top 256 allowed candidates. TODO: schema enforcement (a `JsonState` that also tracks the tool name, required keys and value types from the request's `tools`), `tool_choice: "required"` and a named tool (force `<tool_call>` first), and `response_format`.

## Out of scope for now

Quantised weights and cache (`--cache-type-k q8_0` warns), batching or concurrent requests, prompt reuse beyond the previous request's prompt, parity tests against the training model (they need trained weights), and the 1M-context check (a launch option; the plan checks it separately).

## Tests

`cargo test -p ncm-server` (CPU, tiny configs, no network): chat rendering and parsing, JSON shapes including SSE chunks, the sampler and JSON constraint, both architectures' generation loops, weight save and load through the server, and in-process servers on ephemeral ports. `ncm-model`'s `tests/weights.rs` covers the weight file.

## Continuous batching (`--parallel N`)

With `--parallel N` above 1 the engine serves up to N requests at once (`src/batch.rs`), all on its one worker thread. Each slot has caches for `ctx-size / N` tokens and its own kept prompt prefixes. Every iteration starts queued requests on free slots (the slot whose kept prefix the prompt extends furthest, else the least recently used), runs one prefill chunk of one prefilling slot, so long prompts interleave with decoding, then one decode step over every decoding slot: one token each for the token baseline, one concept each for the NCM. The batched step runs the norms, input projections and MLPs over all rows at once and each slot's attention against its own caches. A request whose prompt equals one a slot is prefilling waits and is forked from it when that prefill ends (the caches are copied into its slot), so the samples of a self-consistency call are prefilled once. Tokens are chosen per request (own settings, seed and constraint). On CUDA the token baseline's unconstrained rows replay one captured graph per set of decoding slots (batched step plus the device sampler with one settings set and counter per row; up to 16 sets kept); constrained rows and the NCM step eagerly (TODO: NCM graphs per slot set), and self-speculation serves the one-slot engine only. `timings` add `queue_ms`, `mean_batch` and `forked`; `/props` adds `is_processing`, `n_processing` and `batching` (each slot's state and tokens, the queue, requests served, forks, mean batch). `--parallel 1` (the default) keeps the sequential engine unchanged. The NCM hybrid (a `[hybrid]` config) is served on one slot only (TODO: batching at block granularity): each request resets one fast decode session (`ncm_model::hybrid::FastSession`, graphed on CUDA), prefills from scratch (no prefix reuse yet), and streams each block's tokens as it closes; constrained requests run in verified mode with the grammar scanning every check pass's tokens, `<tool_call>` switches a block to verified mode, and `timings.hybrid` reports blocks, mean first-pass accepted draft tokens, re-decodes, autoregressive tokens and whether the passes were graphed. Sampling applies temperature, top-k and top-p to the codes and, in verified mode, to the tokens (speculative sampling against the autoregressive path, on the device on CUDA).
