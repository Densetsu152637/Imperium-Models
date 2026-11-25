# NCM: Next Concept Modelling models for Imperium

Training code, configs and the inference runtime for the 250M Next Concept Modelling (NCM) versions of Emperor and Consul, plus the same-size token baselines they are compared with. Nothing is built or trained yet.

The design, plan and acceptance checks live in Imperium-Research under `docs/next-concept-modelling/` (roadmap R4). This folder holds code and data only.

## Layout (planned)

| Path | Contents |
|---|---|
| `src/` | PyTorch models (NCM and token baseline), tokenizer training, data pipeline, training loops |
| `configs/` | Model, data and training configs, one per stage |
| `runtime/` | `ncm-server`: Rust inference server with an OpenAI-compatible API, launched by Imperium like `llama-server` |
| `scripts/check.sh` | CI checks for this model |
| `data/`, `runs/`, `outputs/`, `weights/` | Local only, ignored by Git |

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
