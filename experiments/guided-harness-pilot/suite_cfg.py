"""Which suite a run uses: dev (default) or hard (GHP_SUITE=hard, knowledge base = kb + hard_kb)."""
import os
import pathlib

HERE = pathlib.Path(__file__).parent
HARD = os.environ.get("GHP_SUITE") == "hard"
KB = HERE / ("suite/hard_merged_kb.jsonl" if HARD else "suite/kb.jsonl")
TASKS = HERE / ("suite/hard_tasks.jsonl" if HARD else "suite/tasks.jsonl")
