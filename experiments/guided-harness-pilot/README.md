# Guided harness pilot

A small pilot that measures Qwen3.5-4B (bf16 GGUF, thinking off) with and without Imperium's agent loop (a baseline for the guided reasoning harness proposed in `Imperium-Research/docs/human-reasoning-models/harness-proposal.md`. The write-up with
results is `Imperium-Research/docs/human-reasoning-models/experiments/guided-harness-pilot.md`.

The suite (135 items, 9 categories, fictional-world knowledge base) was authored by Claude. It is evaluation data,
not training data. Scoring is deterministic; there is no LLM judge.

## Arms

| Arm | Description |
|---|---|
| `plain` | One call, question only, greedy. |
| `rag` | One call, question plus top-5 BM25 passages, greedy. |
| `harness_refined` | Imperium with the harness variant in `config/harness-variants/refined.toml` (evidence policy, 4 tools, `--permission-mode full` so `run_command` works in the scratch workspace). |
| `harness` | Imperium's own agent loop via `imperium run --headless`; KB as markdown files in a per-item scratch workspace copy (`kb_files.py`, `imperium_backend.py`). `harness_note` is the same with one extra prompt sentence (real-world, arithmetic and logic questions need no workspace), for wording parity with `rag`. |

## Layout

- `suite/build_suite.py` generates `suite/kb.jsonl` (327 passages) and `suite/tasks.jsonl` (135 items), fixed seed.
- `validate_suite.py` checks ids, support ids, and that unanswerable items are truly absent from the knowledge base.
- `scoring.py`, `test_scorer.py`: deterministic scorer and its self-test.
- `bm25.py`, `llm.py`, `arms.py`: retrieval, server client, arms `plain` and `rag`.
- `kb_files.py`, `imperium_backend.py`: knowledge base as files and the Imperium headless backend.
- `run.py`: runs one arm, appends one record per item to `results/<arm>.jsonl` (resumable).
- `summarize.py`: writes `results/summary.json` and `results/summary.md`.

## Rerun

```
python validate_suite.py
python test_scorer.py
```

Start exactly one server (the runtime build used was llama.cpp b12180, CUDA 13):

```
llama-server.exe -m Qwen3.5-4B-bf16.gguf --jinja -ngl 99 -c 16384 -np 1 --host 127.0.0.1 --port 8099 --reasoning-budget 0
```

Every request also sets `chat_template_kwargs {"enable_thinking": false}`. Then run the arms one after another:

```
python run.py --arm plain
python run.py --arm rag
python run.py --arm harness
python run.py --arm harness_note
python summarize.py
```

`--per-cat N --out results_pilot` runs a small subset into a separate directory. Stop the server afterwards and
confirm the GPU memory returned to idle.

Python 3.13, standard library only.

## Harness arm: pluggable backend

`run.py` registers arms as `fn(task, bm25, meter) -> dict`. The dict must hold `final_text` (ending in a
`Final answer: ...` line) and may hold `steps`, `stop_reason`, `trace`, `output`. The runner measures wall time and
fills `model_calls`, `prompt_tokens` and `completion_tokens` from the `llm.Meter`; a backend that does not call
`llm.chat` (such as `imperium run --headless`) must add the ledger numbers to the meter itself. To add the
arm, create `imperium_backend.py` exposing `run_imperium(task, bm25, meter)`; `run.py` registers it as `harness`
when the module imports.

Requires the Imperium daemon's local pipe (`daemon.socket = true`; Imperium-AI PR #273). Flags used: `--local --thinking off --context-tokens 16384 --timeout-s 300 --flag max_steps=20 --permission-mode default`. Run `python run.py --arm harness` and `--arm harness_note` after starting the same llama-server.

## Dev and hard suites

`suite/tasks.jsonl` (135 items, dev, used to tune the variant) and `suite/hard_tasks.jsonl` (72 items, held out,
authored after the variant was frozen; knowledge base `suite/hard_merged_kb.jsonl` = `kb.jsonl` + `hard_kb.jsonl`).
Select with `GHP_SUITE=hard` (also for `validate_suite.py` and `rescore.py`) and `--out results_hard`.
Scoring is strict (see `scoring.py`); `rescore.py` re-scores saved outputs, keeping the earlier verdict as
`correct_lenient`. `build_variant.py` writes a variant TOML; iteration bodies are in
`config/harness-variants/iterations/`, iteration results in `results/iterations/`.
