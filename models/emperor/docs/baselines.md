# Zero-shot baselines

Untrained candidate bases, run on 2026-11-15 on Emperor's held-out suite. These scores are the
baseline for the trained models. Consul's held-out baselines are still to come (see
[Not yet run](#not-yet-run)).

## Candidates

| Model | Parameters | Licence (from the Hugging Face card) | Size class |
| --- | --- | --- | --- |
| Qwen/Qwen3.5-4B | 4.2B (language model) | Apache-2.0 | 4B |
| microsoft/Phi-4-mini-reasoning | 3.8B | MIT | 4B |
| Qwen/Qwen3.5-2B | 1.9B (language model) | Apache-2.0 | 2B |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.8B | MIT | 2B |

## Setup

- Each model ran from its own bf16 GGUF, converted from the Hugging Face weights with llama.cpp
  `convert_hf_to_gguf.py`. They were served by Imperium's llama.cpp b12180 `llama-server`
  (`--jinja`, the model's own chat template) with a 32,768-token context, on the RTX 5080.
  One model ran at a time.
- The harness was `uv run eval` (replay backend, `--suite heldout`, `--context-tier 32k`) with
  the default 16,384 completion tokens per reply. That is 15 tasks with scripted subagent
  replies, scored by heuristics, with no judge model and no hosted calls.
- A task "fails" when the model does not finish within 16 turns. Its metrics still count.
- "n/a" means the metric had nothing to score. For example, tool_validity is n/a when the
  model made no tool calls at all.
- The reports are kept locally in `models/emperor/runs/baselines-20261115/`.

## Emperor held-out suite

| Metric | Goal | Target | Qwen3.5-4B | Phi-4-mini-reasoning | Qwen3.5-2B | R1-Distill-Qwen-1.5B |
| --- | --- | --- | --- | --- | --- | --- |
| tool_validity | G1 | 0.98 | **1.000** | n/a (no tool calls) | **1.000** | n/a (no tool calls) |
| routing | G2 | 0.85 | **0.636** | 0.273 | 0.273 | 0.273 |
| task_success | G3 | (proxy) | **0.636** | 0.364 | 0.455 | 0.182 |
| brief | G4 | 0.90 | 0.000 | n/a | 0.000 | n/a |
| leakage | G5 | 1.00 | 0.867 | 0.867 | 0.933 | **1.000** |
| retrieval | G6 | 0.90 | 0.000 | 1.000 | 0.000 | 1.000 |
| plan | G10 | 4.0 | **1.522** | 0.000 | 0.606 | 0.000 |
| clarify | G11 | 0.90 | 0.000 | 0.000 | 0.000 | 0.000 |
| tasks without a final answer | | | 4 / 15 | 0 / 15 | 7 / 15 | 0 / 15 |

Notes:

- **Neither non-Qwen model delegates.** Phi-4-mini-reasoning and R1-Distill-Qwen-1.5B never call
  a tool, so they answer every task themselves. That is why they "finish" every task, and why
  leakage and retrieval look good: these are single-task metrics, and answering directly passes
  them. Both are worse than Qwen in their size class on routing, task success and planning,
  the root agent's job.
- Retrieval is one task, and plan and clarify are three each, so those rows are coarse.
- The Qwen models run out of turns on long planning and research tasks (looping on subagent
  calls). This is the behaviour the hosted training data targets.

## Decision

No non-Qwen candidate beats Qwen at the same size. The bases stay as planned:

| Model | Base |
| --- | --- |
| Emperor | Qwen3.5-4B |
| Emperor Lite | Qwen3.5-2B |
| Consul | Qwen3.5-4B |
| Consul Lite | Qwen3.5-2B |

## Not yet run

- **Consul's held-out set** (60 seeded-defect critique tasks, `models/consul/data/hosted-1/eval`)
  for all four candidates. With the 16,384-token completion budget, some critiques run to the
  limit, so one model takes about 50 minutes. That run was moved after Emperor 4B's training so
  it would not cost a training night.
- A frontier root (DeepSeek V4 Pro) in the same harness. It needs the hosted API, and the
  DeepSeek balance is empty.
