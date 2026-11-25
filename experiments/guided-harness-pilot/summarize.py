"""Build summary.json and summary.md from <dir>/<arm>.jsonl.

  python summarize.py [results_dir] [title]

Strict scores (`correct`) are primary. Where records also carry `correct_lenient` (the earlier scorer) a
comparison table is added, and `grounded` rates are reported alongside accuracy.
"""
import json
import pathlib
import statistics as st
import sys
from collections import Counter

HERE = pathlib.Path(__file__).parent
RES = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results")
TITLE = sys.argv[2] if len(sys.argv) > 2 else "Guided harness pilot: results"
ARMS = ["plain", "rag", "harness", "harness_note", "harness_refined"]


def load(arm):
    p = RES / f"{arm}.jsonl"
    return [json.loads(line) for line in open(p, encoding="utf-8")] if p.exists() else []


data = {a: load(a) for a in ARMS}
arms = [a for a in ARMS if data[a]]
CATS = []
for a in arms:
    for r in data[a]:
        if r["category"] not in CATS:
            CATS.append(r["category"])


def agg(rows):
    if not rows:
        return None
    t = [r["wall_time_s"] for r in rows]
    kb = [r for r in rows if r.get("grounded") is not None]
    return {"n": len(rows), "correct": sum(r["correct"] for r in rows),
            "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
            "correct_lenient": sum(r.get("correct_lenient", r["correct"]) for r in rows),
            "mean_time_s": round(st.mean(t), 2), "median_time_s": round(st.median(t), 2),
            "total_time_s": round(sum(t), 1),
            "mean_calls": round(st.mean(r["model_calls"] for r in rows), 2),
            "mean_prompt_tokens": round(st.mean(r["prompt_tokens"] for r in rows)),
            "mean_completion_tokens": round(st.mean(r["completion_tokens"] for r in rows)),
            "errors": sum(1 for r in rows if r.get("error")),
            "grounded": sum(1 for r in kb if r["grounded"]), "grounded_n": len(kb),
            "grounded_all": sum(1 for r in kb if r.get("grounded_all"))}


summary = {"arms": {}}
for a in arms:
    rows = data[a]
    e = {"overall": agg(rows), "by_category": {c: agg([r for r in rows if r["category"] == c]) for c in CATS}}
    if a.startswith("harness"):
        e["exit_reasons"] = dict(Counter(str(r.get("exit_reason")) for r in rows))
        e["exit_codes"] = dict(Counter(str(r.get("exit_code")) for r in rows))
        e["mean_tool_calls"] = round(st.mean(r.get("tool_calls") or 0 for r in rows), 2)
        e["mean_tool_calls_by_category"] = {
            c: round(st.mean((r.get("tool_calls") or 0) for r in rows if r["category"] == c), 2) for c in CATS}
        e["failed_tool_calls_total"] = sum(r.get("failed_tool_calls") or 0 for r in rows)
        e["max_wall_time_s"] = max(r["wall_time_s"] for r in rows)
    summary["arms"][a] = e
json.dump(summary, open(RES / "summary.json", "w"), indent=2)

nl = chr(10)
L = [f"# {TITLE}", "", "Strict scores are primary. Time is wall-clock seconds per item (mean / median).", "",
     "## Overall", "", "| Arm | Accuracy (strict) | Lenient (old scorer) | Mean s | Median s | Total wall time | Mean calls | "
     "Mean prompt tok | Mean completion tok | Errors | Grounded (KB items) |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
for a in arms:
    o = summary["arms"][a]["overall"]
    g = f"{o['grounded']} any, {o['grounded_all']} all of {o['grounded_n']}" if o["grounded_n"] else "n/a"
    L.append(f"| {a} | {o['correct']}/{o['n']} ({o['accuracy']:.1%}) | {o['correct_lenient']}/{o['n']} | "
             f"{o['mean_time_s']} | {o['median_time_s']} | {o['total_time_s'] / 60:.1f} min | {o['mean_calls']} | "
             f"{o['mean_prompt_tokens']} | {o['mean_completion_tokens']} | {o['errors']} | {g} |")
for a in arms:
    if a.startswith("harness"):
        h = summary["arms"][a]
        L.append(f"- {a}: exit reasons {h['exit_reasons']}; exit codes {h['exit_codes']}; mean tool calls "
                 f"{h['mean_tool_calls']}; failed tool calls {h['failed_tool_calls_total']}; slowest item "
                 f"{h['max_wall_time_s']} s.")


def table(title, fn):
    L.extend(["", f"## {title}", "", "| Category | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)])
    for c in CATS:
        L.append(f"| {c} | " + " | ".join(fn(summary["arms"][a], c) for a in arms) + " |")


table("Accuracy by category (strict)", lambda s, c: f"{s['by_category'][c]['correct']}/{s['by_category'][c]['n']}")
table("Grounded by category (KB-backed items: at least one / all supporting passages retrieved or read, of n)",
      lambda s, c: (f"{s['by_category'][c]['grounded']} / {s['by_category'][c]['grounded_all']} of {s['by_category'][c]['grounded_n']}"
                    if s['by_category'][c]['grounded_n'] else "n/a"))
table("Mean wall time per item by category (s, median in brackets)",
      lambda s, c: f"{s['by_category'][c]['mean_time_s']} ({s['by_category'][c]['median_time_s']})")
table("Mean model calls per item by category", lambda s, c: str(s['by_category'][c]['mean_calls']))
table("Mean prompt tokens per item by category", lambda s, c: str(s['by_category'][c]['mean_prompt_tokens']))
L.append("")
out = nl.join(L) + nl
open(RES / "summary.md", "w", encoding="utf-8").write(out)
print(out)
