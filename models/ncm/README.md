# NCM: Next Concept Modelling models for Imperium

Training code, configs and the inference runtime for the 250M Next Concept Modelling (NCM) versions of Emperor and Consul, plus the same-size token baselines they are compared with. Written in Rust on Burn 0.22. Nothing is trained yet.

The design, plan and acceptance checks live in Imperium-Research under `docs/next-concept-modelling/` (roadmap R4). This folder holds code and data only.

## Layout

| Path | Contents |
|---|---|
| `Cargo.toml` | Rust workspace. CPU is the default backend; CUDA is the `cuda` feature |
| `crates/ncm-model/` | Configs, layers, KV caches, the token baseline and the NCM. Device-agnostic: Burn picks CUDA, CPU or wgpu from the `Device` |
| `crates/ncm-bench/` | `ncm-bench`, the batch-1 decode-speed benchmark |
| `configs/` | Model configs (TOML), embedded into `ncm-model` |
| `runtime/` | Planned: `ncm-server`, an OpenAI-compatible inference server launched by Imperium like `llama-server`. A training crate (Burn autodiff) will also build on `ncm-model` |
| `scripts/check.sh` | CI checks: `cargo fmt --check`, `clippy`, `test` on the CPU backend, and a tiny benchmark run |
| `data/`, `runs/`, `outputs/`, `weights/` | Local only, ignored by Git |

## Build and run

```
cargo build --release -p ncm-bench --features cuda
target/release/ncm-bench --device cuda --context 2048 16384 --new-tokens 256   # the stage 1 gate
target/release/ncm-bench --device cpu --tiny --context 64 --new-tokens 8       # smoke run
```

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

Fine-tuning reuses the token-era DeepSeek teacher traces cached in `../../archive/token-era/`. Pretraining data is built from open datasets in stage 2 of the plan.
