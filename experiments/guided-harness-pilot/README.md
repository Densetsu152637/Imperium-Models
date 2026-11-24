# Guided harness pilot

A small pilot that measures Qwen3.5-4B (bf16 GGUF, thinking off) with and without the guided
reasoning harness (to be run through Imperium's agent loop) proposed in `Imperium-Research/docs/human-reasoning-models/harness-proposal.md`. The write-up with
results is `Imperium-Research/docs/human-reasoning-models/experiments/guided-harness-pilot.md`.

The suite (135 items, 9 categories, fictional-world knowledge base) was authored by Claude. It is evaluation data,
not training data. Scoring is deterministic; there is no LLM judge.

## Arms

| Arm | Description |
|---|---|
| `plain` | One call, question only, greedy. |
| `rag` | One call, question plus top-5 BM25 passages, greedy. |
| `harness` | Not run yet. Planned backend: Imperium's own agent loop via `imperium run --headless` against a per-run copy of the KB as plain files. See below. |

## Layout

- `suite/build_suite.py` generates `suite/kb.jsonl` (327 passages) and `suite/tasks.jsonl` (135 items), fixed seed.
- `validate_suite.py` checks ids, support ids, and that unanswerable items are truly absent from the knowledge base.
- `scoring.py`, `test_scorer.py`: deterministic scorer and its self-test.
- `bm25.py`, `llm.py`, `arms.py`: retrieval, server client, arms `plain` and `rag`.
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

Status: as of the first run, `imperium.exe run --headless --print-prompt` is refused by the running daemon with
error -32005 ("evaluation sessions are for the imperium command on this machine, not for phones, mods or the
API"), so the backend has not been written or run.
