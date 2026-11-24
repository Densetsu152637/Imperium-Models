"""Validate suite/kb.jsonl and suite/tasks.jsonl. Exit status 1 on any problem."""
import json
import pathlib
import sys
from collections import Counter

HERE = pathlib.Path(__file__).parent
kb = [json.loads(line) for line in open(HERE / "suite/kb.jsonl", encoding="utf-8")]
tasks = [json.loads(line) for line in open(HERE / "suite/tasks.jsonl", encoding="utf-8")]
errors = []
kb_ids = [p["id"] for p in kb]
if len(set(kb_ids)) != len(kb_ids):
    errors.append("duplicate kb ids")
if len({p["text"] for p in kb}) != len(kb):
    errors.append("duplicate kb texts")
task_ids = [t["id"] for t in tasks]
if len(set(task_ids)) != len(task_ids):
    errors.append("duplicate task ids")
text = {p["id"]: p["text"] for p in kb}
KB_CATS = {"single_hop", "multi_hop", "compare_aggregate", "kb_override", "compute"}
for t in tasks:
    for k in ("id", "category", "question", "answer", "answer_type", "support"):
        if k not in t:
            errors.append(f"{t.get('id')}: missing {k}")
    if t["answer_type"] not in ("string", "number", "abstain"):
        errors.append(f"{t['id']}: bad answer_type")
    for s in t["support"]:
        if s not in text:
            errors.append(f"{t['id']}: support id {s} not in kb")
    if t["category"] in KB_CATS and not t["support"]:
        errors.append(f"{t['id']}: KB item without support")
    if t["category"] in ("arithmetic", "logic", "general_knowledge", "unanswerable") and t["support"]:
        errors.append(f"{t['id']}: should have no support")
    if t["category"] in ("single_hop", "multi_hop", "kb_override") and t["answer_type"] == "string":
        blob = " ".join(text[s] for s in t["support"]).lower()
        if str(t["answer"]).lower() not in blob and t["category"] != "kb_override":
            errors.append(f"{t['id']}: gold {t['answer']!r} not in its support passages")
    if t["category"] == "unanswerable":
        ent, kws = t["absent"]["entity"].lower(), [k.lower() for k in t["absent"]["keywords"]]
        for p in kb:
            low = p["text"].lower()
            if ent in low and any(k in low for k in kws):
                errors.append(f"{t['id']}: answer may be present in {p['id']}: {p['text']}")
        if t["answer_type"] != "abstain":
            errors.append(f"{t['id']}: unanswerable must be abstain type")
cats = Counter(t["category"] for t in tasks)
if len(tasks) < 120 or len(cats) < 6:
    errors.append(f"too few items/categories: {len(tasks)} / {len(cats)}")
print(f"{len(kb)} passages, {len(tasks)} tasks, categories: {dict(cats)}")
if errors:
    print("\n".join(errors))
    sys.exit(1)
print("suite OK")
