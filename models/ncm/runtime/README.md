# ncm-server

An OpenAI-compatible HTTP server over the NCM and the token baseline (`ncm-model`, Burn 0.22), started and spoken to like llama.cpp's `llama-server`. It is stage 8 of the NCM plan, started early: it works today with random weights and loads trained weights once they exist.

```bash
# CPU (what CI builds)
cargo run --release -p ncm-server -- --arch ncm --random-weights -c 4096 --port 8080
# GPU
cargo run --release -p ncm-server --features cuda -- --arch ncm --random-weights -c 4096 --port 8080
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

Not implemented: `stop` strings, penalties, `response_format`, images (`--mmproj` is ignored with a note), `/slots`, `/metrics`, an API key, prompt-cache reuse across requests (`cached_tokens` is always 0), several requests at once (`--parallel N` only divides the context; requests queue and run one at a time), `tool_choice` other than `none`/auto.

## Command line

`--model/-m`, `--host`, `--port` (0 picks a free one), `--ctx-size/-c` (total, allocated at launch; each request gets `ctx / parallel`), `--parallel/-np`, `--n-gpu-layers/-ngl` (0 means CPU), `--alias/-a`, `--temp`, `--top-k`, `--top-p` (llama-server's defaults: 0.8, 40, 0.95) are llama-server's. `--cache-type-k/-v`, `--jinja`, `--threads`, `--batch-size`, `--flash-attn`, `--mlock`, `--no-mmap` and a few more are accepted and ignored so the launch line works unchanged (`-ngl`, `-np`, `-fa`, `-ctk`, `-ctv`, `-ub`, `-tb`, `-cb` are mapped to their long forms). NCM-specific: `--arch tok|ncm`, `--random-weights` (+ `--weights-seed`), `--tokenizer`, `--config`, `--device cpu|cuda`, `--dtype bf16|f32`, `--prefill-chunk`, `--quant none|q8|q4` (+ `--quant-group`, `--dense-head`; weight-only quantisation at load, see `../docs/quantisation.md`). Defaults: CUDA with bf16 when built with `--features cuda` and `-ngl` is not 0, else CPU with f32.

The architecture and config come from `--config`, else the weight file's header, else `config.toml` beside the weights, else the 250M config of `--arch` (the fallback attention design; pass `--config configs/ncm-250m-v2.toml` or `tok-250m-v2.toml` for v2, see `../docs/attention.md`). The tokenizer is `--tokenizer`, else `tokenizer.json` beside the weights; with `--random-weights` and none, a built-in byte-level tokenizer (ids 0 to 10 are the special tokens, 11 to 15 reserved, 16 to 271 bytes) is used.

## Context and cache

The key/value caches are allocated at launch for the per-request context, rounded down to whole concepts for the NCM. `-c 1000000` allocates the 1M-token caches the architecture doc sizes (`ncm_model::cache_bytes`; about 1 GiB for the NCM and 5 GiB for the baseline in bf16); the launch line reports the bytes, `/props` and every response's `timings.cache_bytes` repeat them. Requests reset the caches; generation ends with `finish_reason: "length"` at the end of the context.

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
- The prompt must be whole concepts, so it is left-padded with `<|pad|>`: the prompt still ends exactly where the reply starts. If training pads the other way (or ends a partial concept with `<|eoc|>`), change `generate_ncm` in `src/engine.rs`.
- Sampling inside a concept uses `NcmLm::decode_concept_with`, which hands each position's logits to a callback (greedy unconstrained decoding stays on the device).

## Metrics

Every response has llama-server's `timings` object: `prompt_n`, `prompt_ms`, `prompt_per_second` (prompt processing) and `predicted_n`, `predicted_ms`, `predicted_per_second` (decoding), timed separately with a device sync, plus `cache_bytes`. Tokens are reference-tokenizer tokens (the NCM's `k` per pass are counted individually). `predicted_per_second` excludes the token baseline's first token, which comes out of the prompt pass. The streaming finish chunk carries `timings` and the usage chunk (with `include_usage`) repeats them.

## Grammar-constrained tool calls (R2.4)

The hook is `constrain::Constraint`: `allows(token)` is asked for the candidates the sampler tries, best logit first, and `advance(token)` follows each choice; no vocabulary-sized mask is built. `ToolCallGuard` is on whenever a request has tools: free text is unconstrained, and after `<tool_call>` every token must keep the text a prefix of one JSON object (a byte-level parser, `JsonState`), `</tool_call>` is allowed only once the object is closed, and special tokens and long runs of whitespace outside the object are refused, so a tool call is always syntactically valid JSON. With constraints active sampling looks at the top 256 allowed candidates. TODO: schema enforcement (a `JsonState` that also tracks the tool name, required keys and value types from the request's `tools`), `tool_choice: "required"` and a named tool (force `<tool_call>` first), and `response_format`.

## Out of scope for now

Quantised weights and cache (`--cache-type-k q8_0` warns), batching or concurrent requests, prompt-cache reuse, parity tests against the training model (they need trained weights), and the 1M-context check (a launch option; the plan checks it separately).

## Tests

`cargo test -p ncm-server` (CPU, tiny configs, no network): chat rendering and parsing, JSON shapes including SSE chunks, the sampler and JSON constraint, both architectures' generation loops, weight save and load through the server, and in-process servers on ephemeral ports. `ncm-model`'s `tests/weights.rs` covers the weight file.
