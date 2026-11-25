"""Rescore saved result files with the current (strict) scorer from the saved outputs; no model calls.

  python rescore.py <tasks.jsonl> <results.jsonl> [...]
Keeps the earlier verdict as `correct_lenient` (first time only), writes the strict verdict as `correct`, adds
`grounded`, and writes a flip log to results/rescore_flips.json.
"""
import json
import pathlib
import sys

from grounding import grounded, grounded_all
from scoring import score

HERE = pathlib.Path(__file__).parent
tasks = {}
tasks_path, files = sys.argv[1], sys.argv[2:]
for line in open(tasks_path, encoding="utf-8"):
    t = json.loads(line)
    tasks[t["id"]] = t
flips = []
for f in files:
    rows = [json.loads(line) for line in open(f, encoding="utf-8")]
    for r in rows:
        t = tasks[r["id"]]
        r.setdefault("correct_lenient", r["correct"])
        new, parsed = score(t, r.get("output") or "")
        if r["correct"] != new:
            flips.append({"file": pathlib.Path(f).name, "id": r["id"], "category": r["category"], "arm": r["arm"],
                          "was": r["correct"], "now": new, "answer": parsed, "gold": t["answer"]})
        r["correct"] = bool(new)
        r["final_answer"] = parsed
        r["grounded"] = grounded(t, r)
        r["grounded_all"] = grounded_all(t, r)
    with open(f, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
out = HERE / "results/rescore_flips.json"
old = json.load(open(out)) if out.exists() else []
keep = [x for x in old if x["file"] not in {pathlib.Path(f).name for f in files}]
json.dump(keep + flips, open(out, "w"), indent=1)
print(len(flips), "flips")
