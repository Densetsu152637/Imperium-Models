"""Mixing our hosted data with the grafted sources at target shares.

The hosted data is the anchor: all of it is used, and every other bucket is sized from it by
the target shares. A bucket with no data yet (research planning, before it is generated) drops
out and the remaining shares are renormalised, so the mix still holds the same proportions
among the sources that exist. A bucket with fewer samples than its target uses all it has.
The report records the targets, what was available and what was taken.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

# The plan's mix: ~45% hosted, ~30% research planning, ~15% Nemotron, ~10% OpenThoughts.
TARGET_SHARES = {"hosted": 0.45, "research": 0.30, "nemotron": 0.15, "openthoughts": 0.10}
ANCHOR = "hosted"


def read_jsonl(paths: Sequence[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if not Path(path).is_file():
            continue
        with Path(path).open(encoding="utf-8") as handle:
            rows += [json.loads(line) for line in handle if line.strip()]
    return rows


def plan_counts(available: dict[str, int], shares: dict[str, float] = TARGET_SHARES) -> dict:
    """How many samples to take from each bucket. Buckets without data drop out."""
    present = {name: share for name, share in shares.items() if available.get(name, 0) > 0}
    if ANCHOR not in present:
        raise ValueError("the hosted data is the anchor of the mix and is missing")
    total_share = sum(present.values())
    normalised = {name: share / total_share for name, share in present.items()}
    anchor = available[ANCHOR]
    total = anchor / normalised[ANCHOR]
    counts = {}
    for name, share in normalised.items():
        want = anchor if name == ANCHOR else round(total * share)
        counts[name] = {
            "target_share": shares[name],
            "normalised_share": round(share, 4),
            "available": available.get(name, 0),
            "wanted": want,
            "taken": min(want, available.get(name, 0)),
        }
    for name in shares:
        if name not in counts:
            counts[name] = {
                "target_share": shares[name],
                "normalised_share": 0.0,
                "available": 0,
                "wanted": 0,
                "taken": 0,
                "missing": True,
            }
    return counts


def _order(sample: dict, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample['id']}".encode()).hexdigest()


def take(rows: list[dict], count: int, seed: int, balance_key: str | None = None) -> list[dict]:
    """A deterministic sample of `count` rows; with `balance_key` (a meta field), round-robin
    across its values so no single subset dominates."""
    ordered = sorted(rows, key=lambda r: _order(r, seed))
    if not balance_key:
        return ordered[:count]
    groups: dict[str, list[dict]] = {}
    for row in ordered:
        groups.setdefault(str((row.get("meta") or {}).get(balance_key)), []).append(row)
    out: list[dict] = []
    queues = [groups[k] for k in sorted(groups)]
    while len(out) < count and any(queues):
        for queue in queues:
            if queue and len(out) < count:
                out.append(queue.pop(0))
    return out


MODELS = ("emperor", "consul")


def user_text(row: dict) -> str:
    return "\n".join(
        m.get("content") or ""
        for m in row.get("messages") or []
        if m.get("role") == "user" and isinstance(m.get("content"), str)
    )


GRAFT_BUCKETS = ("nemotron", "openthoughts")


def graft_rejection(row: dict) -> str | None:
    """A grafted row must come from a current source: its subset is registered and usable, its
    revision is the pinned one, and its generator and upstream source are allowed now."""
    from emperor.graft.sources import SOURCES, allowed, upstream_allowed

    meta = row.get("meta") or {}
    source = SOURCES.get(meta.get("subset"))
    if source is None or not source.usable:
        return "graft_unknown_or_excluded_source"
    if meta.get("revision") != source.revision:
        return "graft_stale_revision"
    if not allowed(meta.get("generator")):
        return "graft_generator_not_allowed"
    if not upstream_allowed(source, meta.get("upstream_source")):
        return "graft_upstream_not_allowed"
    return None


def rejection(row: dict, model: str, heldout, bucket: str = "") -> str | None:
    """Why a row cannot enter this model's mix: not a chat sample ending in an assistant turn,
    made for the other model (`meta.for_model`), a graft from a stale or disallowed source, a
    generated row that was not judged, or overlapping a held-out eval task."""
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return "no_messages"
    if messages[-1].get("role") != "assistant":
        return "not_answered"
    meta = row.get("meta") or {}
    target = meta.get("for_model")
    if target and target != model:
        return f"for_{target}"
    if bucket in GRAFT_BUCKETS:
        why = graft_rejection(row)
        if why:
            return why
    elif bucket != ANCHOR and meta.get("judged") is not True:
        return "not_judged"
    if heldout is not None and heldout.leaks(user_text(row)):
        return "heldout_overlap"
    return None


def screen(
    buckets: dict[str, list[dict]], model: str, heldout
) -> tuple[dict[str, list[dict]], dict[str, dict[str, int]]]:
    """Each bucket without the rows that cannot enter the mix, and the counts of why."""
    kept: dict[str, list[dict]] = {}
    rejected: dict[str, dict[str, int]] = {}
    for name, rows in buckets.items():
        kept[name] = []
        for row in rows:
            why = rejection(row, model, heldout, name)
            if why:
                rejected.setdefault(name, {})
                rejected[name][why] = rejected[name].get(why, 0) + 1
            else:
                kept[name].append(row)
    return kept, rejected


def mix(
    buckets: dict[str, list[dict]],
    seed: int = 1337,
    shares: dict[str, float] = TARGET_SHARES,
    *,
    model: str = "emperor",
    heldout="bundled",
) -> tuple[list[dict], dict]:
    """The mixed samples for `model` (shuffled deterministically) and a report. `heldout` is a
    `HeldoutIndex` (default: the bundled eval tasks) or None to skip the overlap check."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model}")
    if model == "consul" and heldout == "bundled":
        # The bundled set is Emperor's; Consul's held-out tasks live in the consul package.
        raise ValueError("Consul's held-out overlap check is not wired up; pass heldout")
    if heldout == "bundled":
        from emperor.data.prompts import HeldoutIndex

        heldout = HeldoutIndex.bundled()
    buckets, rejected = screen(buckets, model, heldout)
    counts = plan_counts({k: len(v) for k, v in buckets.items()}, shares)
    chosen: list[dict] = []
    seen: set[str] = set()
    for name, plan in counts.items():
        if not plan["taken"]:
            continue
        rows = take(buckets[name], plan["taken"], seed, None if name == ANCHOR else "subset")
        for row in rows:
            if row["id"] not in seen:
                seen.add(row["id"])
                chosen.append(row)
    chosen.sort(key=lambda r: _order(r, seed + 1))
    by_source: dict[str, int] = {}
    by_licence: dict[str, int] = {}
    by_generator: dict[str, int] = {}
    for row in chosen:
        meta = row.get("meta") or {}
        by_source[meta.get("subset") or meta.get("source") or ANCHOR] = (
            by_source.get(meta.get("subset") or meta.get("source") or ANCHOR, 0) + 1
        )
        by_licence[meta.get("licence", "?")] = by_licence.get(meta.get("licence", "?"), 0) + 1
        generator = meta.get("generator") or meta.get("teacher") or "?"
        by_generator[generator] = by_generator.get(generator, 0) + 1
    grafts: dict[str, dict] = {}
    for row in chosen:
        meta = row.get("meta") or {}
        if not meta.get("revision"):
            continue  # not an open-data graft
        entry = grafts.setdefault(
            meta["subset"],
            {
                "dataset": meta["source"],
                "subset": meta["subset"],
                "revision": meta["revision"],
                "licence": meta["licence"],
                "generators": set(),
                "question_licences": set(),
                "samples": 0,
            },
        )
        entry["generators"].add(meta["generator"])
        entry["samples"] += 1
        if meta.get("question_licence"):
            entry["question_licences"].add(f"{meta['upstream_source']}: {meta['question_licence']}")
    report = {
        "model": model,
        "grafts": [
            {
                **g,
                "generators": sorted(g["generators"]),
                "question_licences": sorted(g["question_licences"]),
            }
            for _, g in sorted(grafts.items())
        ],
        "seed": seed,
        "total": len(chosen),
        "rejected": rejected,
        "heldout_checked": heldout is not None,
        "buckets": counts,
        "by_source": dict(sorted(by_source.items())),
        "by_licence": dict(sorted(by_licence.items())),
        "by_generator": dict(sorted(by_generator.items())),
    }
    return chosen, report
