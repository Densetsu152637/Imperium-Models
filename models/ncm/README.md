# NCM: Next Concept Modelling models for Imperium

Training code, configs and the inference runtime for the 250M Next Concept Modelling (NCM) versions of Emperor and Consul, plus the same-size token baselines they are compared with. Stage 1 (design freeze and speed gate) code is in, in Rust on Burn: both models with random weights, tests (including a parity test against the PyTorch reference) and the `ncm-bench` decode-speed benchmark. Nothing is trained yet.

The design, plan and acceptance checks live in Imperium-Research under `docs/next-concept-modelling/` (roadmap R4). This folder holds code and data only.

## Layout

| Path | Contents |
|---|---|
| `Cargo.toml` | Rust workspace (Burn 0.22). The models are Rust; CPU is the default backend, CUDA is the `cuda` feature |
| `crates/ncm-model/` | Configs, layers, KV caches, the token baseline and the NCM, device-agnostic (Burn picks CUDA, CPU or wgpu from the `Device`). The training crate (Burn autodiff) and `runtime/` (`ncm-server`, an OpenAI-compatible inference server launched by Imperium like `llama-server`) build on it later |
| `crates/ncm-bench/` | `ncm-bench`, the batch-1 decode-speed benchmark |
| `configs/` | Model, data and training configs, one per stage; read by both the Rust and the PyTorch code |
| `src/ncm/`, `tests/`, `pyproject.toml` | PyTorch reference implementation, kept as the numerical reference until removed. `scripts/export_parity_fixture.py` exports tiny-config weights and logits for the Rust parity test |
| `scripts/check.sh` | CI checks for this model (Python reference, then `cargo fmt`, `clippy`, `test` on the CPU backend) |
| `data/`, `runs/`, `outputs/`, `weights/` | Local only, ignored by Git |

## Benchmark

```
cargo build --release -p ncm-bench --features cuda
target/release/ncm-bench --device cuda --context 2048 16384 --new-tokens 256   # the stage 1 gate
target/release/ncm-bench --device cpu --tiny --context 64 --new-tokens 8       # smoke run
```

Random weights, batch 1, greedy decode with caches. Prefill is timed separately, decode tok/s is output tokens per second (median of `--repeats`), and the NCM/token-baseline ratio is graded PASS (1.5x or more), WEAK (1.2x to 1.5x) or FAIL. The `host` column is the share of decode time spent enqueueing work before the final device sync: near 100% means the run is bound by host-side op dispatch rather than by the GPU, which understates the NCM's advantage.

`cuda` builds need a local CUDA toolkit. `.cargo/config.toml` pins `CUDARC_CUDA_VERSION` because cudarc does not list CUDA 13.5 yet.

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
