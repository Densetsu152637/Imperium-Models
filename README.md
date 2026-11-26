# Imperium-Models

Open-weight models trained for [Imperium](https://github.com/Densetsu152637/Imperium-AI)'s
agent roles. Each model has its own folder, with its own Python project, configs, tests and
release cards:

| Folder | Model | Role |
| --- | --- | --- |
| [`models/emperor`](models/emperor) | Emperor (4B) and Emperor Lite (2B) | Root agent: plans, delegates, checks |
| [`models/consul`](models/consul) | Consul (9B) and Consul Lite (4B) | Critic of plans, diffs and agent reports |
| [`models/ncm`](models/ncm) | Princeps (250M, NCM, from scratch) | One model that runs both the Emperor and Consul harnesses |

**Current line (2026-11-26):** Princeps, in `models/ncm`. There is one model; Emperor and Consul now name Imperium harnesses: the Emperor harness (root agent: plans, delegates, checks) and the Consul harness (critic of plans, diffs and agent reports). The Emperor and Consul Qwen LoRA folders above are retired history. Further NCM training is on hold until the Imperium Harness Research session has finished building the harnesses.

The shared data, training, eval and release pipeline lives in the `emperor` package. Other models
depend on it as a sibling path package, so a change to the pipeline lands in one commit. A new,
more specialised model gets its own `models/<name>` folder with the same layout.

- Plan from now to release and v2: [models/emperor/docs/plan.md](models/emperor/docs/plan.md)
- How to run each step: [models/emperor/docs/runbook.md](models/emperor/docs/runbook.md)
- Checks for one model: `bash models/<name>/scripts/check.sh` (CI runs each model)
