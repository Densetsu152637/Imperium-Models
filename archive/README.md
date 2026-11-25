# Archive

Cached data from earlier model generations. Contents are local only and ignored by Git, except these READMEs.

## token-era

Training data for the token-based Emperor and Consul (Qwen3.5-4B LoRA students), moved here on 2026-11-25 when the research switched to Next Concept Modelling (NCM) models trained from scratch.

| Path | Contents |
|---|---|
| `token-era/emperor/data/` | Prompts, DeepSeek teacher traces, judged and rejected samples, open-data grafts and the final training mixes (`final`, `mix-emperor-v2`, `full-1`, `hosted-1`, `graft-v2`, `judged`, `quarantine`, `rejects`, `traces`). About 5.5 GB, 95 files. |
| `token-era/consul/data/` | Prompts, teacher traces, eval data and rejects (`eval`, `hosted-1`, `rejects`, `traces`). About 0.15 GB, 43 files. |

Each model's `models/<name>/data/` directory is now empty. The token-era pipeline still reads from there, so to rerun it, move or link the archived directory back first.

The teacher traces stay useful: the NCM models reuse them as sequence-level distillation and fine-tuning data. See the NCM research docs in Imperium-Research (`docs/next-concept-modelling/`).

Runs, outputs and weights for the token-era models were not moved; they stay under each `models/<name>/` folder.
