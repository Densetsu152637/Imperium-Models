"""Grounding (reported, not part of correctness): did the answerer see a passage that supports the item?

rag: a supporting passage is in the retrieved top-5.  Harness arms: the agent read a topic file that contains a
supporting passage, or a grep pattern it ran matched a supporting line (reconstructed from the recorded tool-call
arguments against the same files; run_command calls that name a file count as reading it).
"""
import json
import pathlib
import re

import kb_files
import suite_cfg

HERE = pathlib.Path(__file__).parent


def _files():
    kb = [json.loads(line) for line in open(suite_cfg.KB, encoding="utf-8")]
    buckets = {f: [] for f, _, _ in kb_files.TOPICS}
    buckets["other-facts.md"] = []
    for p in kb:
        if p.get("topic"):
            buckets.setdefault(p["topic"], []).append(p)
            continue
        for f, _, rx in kb_files.TOPICS:
            if re.search(rx, p["text"]):
                buckets[f].append(p)
                break
        else:
            buckets["other-facts.md"].append(p)
    return buckets


FILES = _files()
FILE_OF = {p["id"]: f for f, ps in FILES.items() for p in ps}
TEXT = {p["id"]: p["text"] for ps in FILES.values() for p in ps}


def _args(a):
    try:
        return json.loads(a)
    except Exception:
        m = re.search(r'"pattern":"((?:[^"\\]|\\.)*)', a)
        if not m:
            return {}
        try:
            return {"pattern": json.loads('"' + m.group(1) + '"')}
        except Exception:
            return {"pattern": m.group(1)}


def seen_support(task, rec):
    """Set of supporting passage ids the answerer saw (retrieved or read), or None for items with no support."""
    support = task.get("support") or []
    if not support:
        return None
    if rec["arm"] == "rag":
        return {s for s in support if s in (rec.get("retrieved") or [])}
    if rec["arm"] == "plain":
        return None
    seen = set()
    for tc in rec.get("trace") or []:
        tool, raw = tc.get("tool"), tc.get("args") or ""
        a = _args(raw)
        if tool == "read_file":
            path = str(a.get("path", "")).replace("\\", "/").split("/")[-1]
            seen |= {s for s in support if FILE_OF[s] == path}
        elif tool == "grep":
            pat = a.get("pattern")
            if not pat:
                continue
            try:
                rx = re.compile(pat, re.I if a.get("case_insensitive") else 0)
            except re.error:
                continue
            scope = str(a.get("path", "")).replace("\\", "/").split("/")[-1]
            for s in support:
                if scope.endswith(".md") and scope != FILE_OF[s]:
                    continue
                if rx.search(f"[{s}] {TEXT[s]}"):
                    seen.add(s)
        elif tool == "run_command":
            seen |= {s for s in support if FILE_OF[s] in raw}
    return seen


def grounded(task, rec):
    """At least one supporting passage seen (None when not applicable)."""
    s = seen_support(task, rec)
    return None if s is None else bool(s)


def grounded_all(task, rec):
    """Every supporting passage seen (None when not applicable)."""
    s = seen_support(task, rec)
    return None if s is None else s >= set(task["support"])
