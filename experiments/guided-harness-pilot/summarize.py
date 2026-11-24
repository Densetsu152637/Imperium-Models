"""Build results/summary.json and results/summary.md from results/<arm>.jsonl.  python summarize.py [results_dir]"""
import json
import pathlib
import statistics as st
import sys

HERE = pathlib.Path(__file__).parent
RES = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results")
ARMS = ["plain", "rag", "harness"]
CATS = ["single_hop", "multi_hop", "compare_aggregate", "arithmetic", "logic", "unanswerable", "kb_override",
        "compute", "general_knowledge"]


def load(arm):
    p = RES / f"{arm}.jsonl"
    return [json.loads(line) for line in open(p, encoding="utf-8")] if p.exists() else []


def agg(rows):
    if not rows:
        return None
    t = [r["wall_time_s"] for r in rows]
    return {"n": len(rows), "correct": sum(r["correct"] for r in rows),
            "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
            "mean_time_s": round(st.mean(t), 2), "median_time_s": round(st.median(t), 2),
            "total_time_s": round(sum(t), 1),
            "mean_calls": round(st.mean(r["model_calls"] for r in rows), 2),
            "mean_prompt_tokens": round(st.mean(r["prompt_tokens"] for r in rows)),
            "mean_completion_tokens": round(st.mean(r["completion_tokens"] for r in rows)),
            "errors": sum(1 for r in rows if r.get("error"))}


data = {a: load(a) for a in ARMS}
summary = {"arms": {}}
for a in ARMS:
    rows = data[a]
    if not rows:
        continue
    entry = {"overall": agg(rows), "by_category": {c: agg([r for r in rows if r["category"] == c]) for c in CATS}}
    if a == "harness":
        esc = [r for r in rows if r.get("escalated")]
        entry["escalation_rate"] = round(len(esc) / len(rows), 4)
        entry["escalated"] = agg(esc)
        entry["fast_path"] = agg([r for r in rows if r.get("escalated") is False])
        entry["escalation_rate_by_category"] = {
            c: round(sum(1 for r in rows if r["category"] == c and r.get("escalated")) /
                     max(1, sum(1 for r in rows if r["category"] == c)), 3) for c in CATS}
        entry["abstain_final_rate"] = round(sum(1 for r in rows if r.get("abstained")) / len(rows), 4)
    summary["arms"][a] = entry
json.dump(summary, open(RES / "summary.json", "w"), indent=2)

arms = [a for a in ARMS if a in summary["arms"]]
L = ["# Guided harness pilot: results", "",
     "Accuracy is n correct / n. Time is wall-clock seconds per item (mean / median).", ""]
L += ["## Overall", "", "| Arm | Accuracy | Mean s | Median s | Total wall time | Mean calls | Mean prompt tok | "
      "Mean completion tok | Errors |", "|---|---|---|---|---|---|---|---|---|"]
for a in arms:
    o = summary["arms"][a]["overall"]
    L.append(f"| {a} | {o['correct']}/{o['n']} ({o['accuracy']:.1%}) | {o['mean_time_s']} | {o['median_time_s']} | "
             f"{o['total_time_s'] / 60:.1f} min | {o['mean_calls']} | {o['mean_prompt_tokens']} | "
             f"{o['mean_completion_tokens']} | {o['errors']} |")
if "harness" in summary["arms"]:
    h = summary["arms"]["harness"]
    L += ["", f"Harness escalation rate: {h['escalation_rate']:.1%}; final abstain rate: {h['abstain_final_rate']:.1%}."]
    for k, name in (("fast_path", "Fast path only"), ("escalated", "Escalated to step loop")):
        if h[k]:
            L.append(f"- {name}: {h[k]['correct']}/{h[k]['n']} correct, mean {h[k]['mean_time_s']} s, "
                     f"mean {h[k]['mean_calls']} calls.")
L += ["", "## Accuracy by category", "", "| Category | " + " | ".join(arms) + " |",
      "|---|" + "---|" * len(arms)]
for c in CATS:
    L.append(f"| {c} | " + " | ".join(
        f"{summary['arms'][a]['by_category'][c]['correct']}/{summary['arms'][a]['by_category'][c]['n']}"
        for a in arms) + " |")
L += ["", "## Mean wall time per item by category (s)", "", "| Category | " + " | ".join(arms) + " |",
      "|---|" + "---|" * len(arms)]
for c in CATS:
    L.append(f"| {c} | " + " | ".join(
        f"{summary['arms'][a]['by_category'][c]['mean_time_s']} (median {summary['arms'][a]['by_category'][c]['median_time_s']})"
        for a in arms) + " |")
L += ["", "## Mean model calls per item by category", "", "| Category | " + " | ".join(arms) + " |",
      "|---|" + "---|" * len(arms)]
for c in CATS:
    L.append(f"| {c} | " + " | ".join(str(summary['arms'][a]['by_category'][c]['mean_calls']) for a in arms) + " |")
if "harness" in summary["arms"]:
    L += ["", "## Harness escalation rate by category", "", "| Category | Escalated |", "|---|---|"]
    for c in CATS:
        L.append(f"| {c} | {summary['arms']['harness']['escalation_rate_by_category'][c]:.0%} |")
open(RES / "summary.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
