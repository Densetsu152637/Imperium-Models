"""Run one arm over the suite, appending one JSON record per item (resumable).

  python run.py --arm plain|rag|harness [--out results] [--per-cat N] [--ids a,b]
Requires llama-server on 127.0.0.1:8099 (see README.md).
"""
import argparse
import json
import pathlib
import sys
import time

import arms
import llm
from bm25 import BM25
from grounding import grounded, grounded_all
from scoring import score

import suite_cfg

HERE = pathlib.Path(__file__).parent
# Pluggable backends: fn(task, bm25, meter) -> {"final_text": str, ...optional extras}. The harness arm is
# registered below once its backend (Imperium headless) is available.
ARMS = {"plain": arms.run_plain, "rag": arms.run_rag}
try:
    import imperium_backend
    ARMS["harness"] = imperium_backend.run_imperium
    ARMS["harness_note"] = imperium_backend.run_imperium_note
    ARMS["harness_refined"] = imperium_backend.run_imperium_refined
except ImportError:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--out", default="results")
    ap.add_argument("--per-cat", type=int, default=0)
    ap.add_argument("--ids", default="")
    a = ap.parse_args()
    tasks = [json.loads(line) for line in open(suite_cfg.TASKS, encoding="utf-8")]
    if a.ids:
        want = set(a.ids.split(","))
        tasks = [t for t in tasks if t["id"] in want]
    if a.per_cat:
        seen, keep = {}, []
        for t in tasks:
            seen[t["category"]] = seen.get(t["category"], 0) + 1
            if seen[t["category"]] <= a.per_cat:
                keep.append(t)
        tasks = keep
    out_dir = HERE / a.out
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{a.arm}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["id"] for line in open(path, encoding="utf-8")}
    bm25 = BM25(suite_cfg.KB)
    llm.chat([{"role": "user", "content": "Say OK."}], llm.Meter(), max_tokens=4)  # warm-up, not recorded
    fn = ARMS[a.arm]
    t_start = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for i, t in enumerate(tasks):
            if t["id"] in done:
                continue
            meter = llm.Meter()
            rec = {"id": t["id"], "category": t["category"], "arm": a.arm, "gold": t["answer"], "error": None,
                   "escalated": None, "steps": None}
            t0 = time.perf_counter()
            try:
                r = fn(t, bm25, meter)
                text = r["final_text"]
                correct, parsed = score(t, text)
                rec.update(final_answer=parsed, correct=bool(correct), escalated=r.get("escalated"),
                           steps=r.get("steps"), abstained=r.get("abstained"), stop_reason=r.get("stop_reason"),
                           escalation_reason=r.get("escalation_reason"), samples=r.get("samples"),
                           retrieved=r.get("retrieved"), trace=r.get("trace"), output=r.get("output"))
                for k in ("exit_code", "exit_reason", "tool_calls", "failed_tool_calls", "imperium_wall_ms",
                          "prompt_hash", "variant_hash", "variant_warnings", "cached_tokens", "runtime_error", "endpoint_errors", "stderr"):
                    if k in r:
                        rec[k] = r[k]
                if parsed is None:
                    rec["error"] = "no_final_answer_line"
            except Exception as e:  # an erroring item counts as wrong and is recorded
                rec.update(final_answer=None, correct=False, error=f"{type(e).__name__}: {e}")
            rec["grounded"] = grounded(t, rec)
            rec["grounded_all"] = grounded_all(t, rec)
            rec.update(wall_time_s=round(time.perf_counter() - t0, 3), model_calls=meter.calls,
                       prompt_tokens=meter.prompt_tokens, completion_tokens=meter.completion_tokens)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{a.arm}] {i + 1}/{len(tasks)} {t['id']} correct={rec['correct']} {rec['wall_time_s']}s "
                  f"calls={meter.calls} esc={rec['escalated']} ans={rec['final_answer']!r}", flush=True)
    print(f"[{a.arm}] finished in {time.time() - t_start:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
