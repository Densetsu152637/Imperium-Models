"""Validate suite/kb.jsonl and suite/tasks.jsonl. Exit status 1 on any problem."""
import json
import pathlib
import sys
from collections import Counter

import suite_cfg

HERE = pathlib.Path(__file__).parent
kb = [json.loads(line) for line in open(suite_cfg.KB, encoding="utf-8")]
tasks = [json.loads(line) for line in open(suite_cfg.TASKS, encoding="utf-8")]
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
KB_CATS = {"single_hop", "multi_hop", "compare_aggregate", "kb_override", "compute", "hard_chain", "hard_distractor",
           "hard_temporal", "hard_filter", "hard_compute", "hard_counterfactual", "hard_mixed"}
GOLD_IN_SUPPORT = {"single_hop", "multi_hop", "hard_chain", "hard_distractor", "hard_temporal", "hard_filter"}
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
    if t["category"] in GOLD_IN_SUPPORT and t["answer_type"] == "string":
        blob = " ".join(text[s] for s in t["support"]).lower()
        if str(t["answer"]).lower() not in blob:
            errors.append(f"{t['id']}: gold {t['answer']!r} not in its support passages")
    if t["category"] == "unanswerable" or t["answer_type"] == "abstain":
        ent, kws = t["absent"]["entity"].lower(), [k.lower() for k in t["absent"]["keywords"]]
        for p in kb:
            low = p["text"].lower()
            if ent in low and any(k in low for k in kws):
                errors.append(f"{t['id']}: answer may be present in {p['id']}: {p['text']}")
        if t["answer_type"] != "abstain" or t["support"]:
            errors.append(f"{t['id']}: unanswerable must be abstain type without support")
cats = Counter(t["category"] for t in tasks)
if len(tasks) < (60 if suite_cfg.HARD else 120) or len(cats) < 6:
    errors.append(f"too few items/categories: {len(tasks)} / {len(cats)}")
print(f"{len(kb)} passages, {len(tasks)} tasks, categories: {dict(cats)}")
if errors:
    print("\n".join(errors))
    sys.exit(1)
print("suite OK")
