# NCM: Next Concept Modelling models for Imperium

Training code, configs and the inference runtime for the 250M Next Concept Modelling (NCM) versions of Emperor and Consul, plus the same-size token baselines they are compared with. Written in Rust on Burn 0.22. Nothing is trained yet.

The design, plan and acceptance checks live in Imperium-Research under `docs/next-concept-modelling/` (roadmap R4). This folder holds code and data only.

## Layout

| Path | Contents |
|---|---|
| `Cargo.toml` | Rust workspace. CPU is the default backend; CUDA is the `cuda` feature |
| `crates/ncm-model/` | Configs, layers, caches, the token baseline and the NCM, each in two attention designs (`attention = "v2"` or `"fallback"`, see `docs/attention.md`). Device-agnostic: Burn picks CUDA, CPU or wgpu from the `Device` |
| `crates/ncm-bench/` | `ncm-bench`, the batch-1 decode-speed benchmark |
| `crates/ncm-data/` | `ncm-data`, the stage 2 data pipeline: fetch, sample, train the tokenizer, tokenise, stats |
| `configs/` | Model configs (TOML), embedded into `ncm-model`: `*-250m-v2.toml` (the adopted attention design) and `*-250m.toml` (the fallback); and `data-mix.toml`, the pinned pretraining source manifest |
| `docs/` | `attention.md` (the two attention designs as implemented), `chat-format.md` |
| `runtime/` | `ncm-server`, an OpenAI-compatible inference server launched by Imperium like `llama-server` (see `runtime/README.md`). A training crate (Burn autodiff) will also build on `ncm-model` |
| `scripts/check.sh` | CI checks: `cargo fmt --check`, `clippy`, `test` on the CPU backend, and a tiny benchmark run |
| `data/`, `runs/`, `outputs/`, `weights/` | Local only, ignored by Git |

## Build and run

```
cargo build --release -p ncm-bench --features cuda
target/release/ncm-bench --device cuda --context 2048 16384 131072 --new-tokens 256   # speed gate, all four models
target/release/ncm-bench --device cuda --design fallback --context 2048 16384          # the stage 1 gate
target/release/ncm-bench --device cpu --tiny --context 64 --new-tokens 8              # smoke run
```

`--design v2|fallback|both` (default `both`) and `--model tok|ncm|both` pick the models. With both designs the report adds the second gate of the local test plan: v2 NCM decode speed over fallback NCM, which passes at 0.75x or more.

`cuda` builds need a local CUDA toolkit. `.cargo/config.toml` sets `CUDARC_CUDA_VERSION=13040` because cudarc (under burn-cuda) only lists CUDA up to 13.4 and rejects 13.5; the 13.x runtime is ABI compatible. CI never builds the `cuda` feature.

The benchmark uses random weights, batch 1 and greedy decode with caches. Prefill is timed separately, decode tok/s counts output tokens (median of `--repeats`), and the NCM/token ratio is graded PASS (1.5x or more), WEAK (1.2x to 1.5x) or FAIL. The `host` column is the share of decode time spent enqueueing work: near 100% means the run is bound by host op dispatch, not the GPU.

## Stage 1 result

The speed gate passed. NCM against the token baseline: 1.45x to 2.02x at 2k context (1.45x, 1.84x and 2.02x across runs) and 1.83x to 2.30x at 16k. Decode is host-dispatch-bound in eager mode, so these figures likely understate the NCM advantage on the GPU.

## Models

| Model | Architecture | Starts from |
|---|---|---|
| `tok-250m-base` | Token baseline | Scratch |
| `ncm-250m-base` | NCM | Scratch |
| `emperor-tok-250m`, `consul-tok-250m` | Token baseline | `tok-250m-base` |
| `emperor-ncm-250m`, `consul-ncm-250m` | NCM | `ncm-250m-base` |

All models support a context of at least 1M tokens. Tests allocate 2k to 16k.

## Data

Fine-tuning reuses the token-era DeepSeek teacher traces cached in `../../archive/token-era/`. Pretraining data is built by `ncm-data` (stage 2 of the plan) from the open sources pinned in `configs/data-mix.toml`.

### Source rules

- Not gated: nothing needs an accepted agreement or a token.
- Licences allow training and redistributing derived models: ODC-By, CC BY, Apache-2.0, or per-file permissive code licences. Each source records its licence.
- No Claude data: every dataset card was read for Claude or Anthropic as a data generator or filter, and the result is in the source's `claude_check`. Synthetic sets name their generators (DeepSeek-R1, QwQ-32B, Qwen3, GPT-OSS, Llama 3), recorded under `generators`.
- Web text and papers dated March 2023 or later are dropped (Claude 1 shipped then; AI-written text on the web grows afterwards). The exclusion rules live in the manifest.
- Every source is pinned to a 40-character commit revision; files come from `resolve/<revision>/` and are checked against the Hub's SHA-256.

### The mix (20B tokens)

| Source | Category | Phase | Share | Licence | Tokens available | B/token |
|---|---|---|---|---|---|---|
| `fineweb-edu` (pre-2023 crawls) | web | main | 45% | ODC-By | 10.3B | 4.34 |
| `stack-edu` (Common Pile Stack v2 Edu) | code | main | 25% | per file, Blue Oak permissive | 6.3B | 3.65 |
| `pes2o` (Common Pile, open-licence papers) | research | main | 10% | CC BY, CC0 | 2.4B | 4.34 |
| `finemath` (4plus) | math | main | 10% | ODC-By | 2.4B | 3.30 |
| `openthoughts3` | reasoning | main | 3% | Apache-2.0 | 0.64B | 3.70 |
| `openmathreasoning` (cot) | reasoning | anneal | 2% | CC BY 4.0 | 0.51B | 2.88 |
| `openr1-math` | reasoning | anneal | 1% | Apache-2.0 | 0.24B | 2.86 |
| `nemotron-tools` | tool use | main | 2% | CC BY 4.0 | 0.48B | 4.14 |
| `nemotron-agentic` | tool use | main | 2% | CC BY 4.0 | 0.46B | 4.44 |

Tokens available is an estimate for the files the manifest selects, scaled from the proof run; bytes per token (B/token) was measured with the proof tokenizer. Reasoning traces and chat records are rendered with the chat markers: `<|im_start|>user`, the problem, `<|im_end|>`, `<|im_start|>assistant`, then `<think>` reasoning `</think>` and the answer, so the model sees that structure from the start of pretraining. Tool calls use `<tool_call>` and `<tool_response>`. Literal special-token strings inside web text or code are defused (`< |im_end|>`) so a document cannot inject a control token.

The `[anneal]` table records the mix for the last 10% of pretraining, which favours maths, papers and reasoning. `phase = "anneal"` sources are reserved for it. The data pipeline only tags shards by source and phase; the training loop will sample them.

### Pipeline

All commands are `cargo run --release -p ncm-data --` followed by:

| Command | What it does |
|---|---|
| `manifest` | Validate `configs/data-mix.toml` and print the mix |
| `fetch [--source S] [--max-files N] [--limit-bytes 200M] [--jobs 4] [--dry-run]` | Download into `data/raw/<source>/`, resumable, SHA-256 checked, and record exact files in `data/manifest.lock.json`. `--limit-bytes` takes prefixes of line-oriented files evenly, and whole Parquet files |
| `sample [--bytes 100M]` | Draw a text sample from every source in proportion to its share into `data/sample/sample.jsonl` |
| `tokenizer train` | Train the byte-level BPE on the sample to the models' `vocab_size` (32000), into `data/tokenizer/tokenizer.json` |
| `tokenizer info` | Print the vocabulary size and special token ids |
| `tokenize [--budget-tokens N] [--source S]` | Write `data/tokens/{train,val}/<source>-NNNNN.bin` and `data/tokens/index.json` |
| `stats [--json]` | Documents, bytes, tokens and bytes per token, per source, category and phase |

Tokenizer: byte-level BPE (Hugging Face `tokenizers`), every byte in the base alphabet so any text round-trips. Digits split one per token. Ids 0 to 15 are special: `<|endoftext|>` 0, `<|pad|>` 1, `<|im_start|>` 2, `<|im_end|>` 3, `<tool_call>` 4, `</tool_call>` 5, `<tool_response>` 6, `</tool_response>` 7, `<think>` 8, `</think>` 9, the NCM concept end `<|eoc|>` 10, then five reserved ids.

Shards: little-endian `u16` token ids, one source per file, each document followed by `<|endoftext|>`, documents packed back to back (they may span a shard boundary). `index.json` lists every shard with its source, split, token count and SHA-256, plus per-source counts and the tokenizer's hash. About 0.5% of documents, chosen by a hash of the source name and text, go to `val/`, so the split is deterministic and duplicates stay on one side. `tokenize` stops each source at `share x budget` tokens; files read round-robin (`interleave`) so a source grouped by language or domain comes out mixed.

Typical full run, in order: `fetch`, `sample --bytes 1G`, `tokenizer train`, `tokenize`, `stats`. Everything under `data/` is ignored by Git. The tests and `scripts/check.sh` never touch the network; the fetch tests use a local HTTP server.
