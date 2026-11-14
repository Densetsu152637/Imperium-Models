"""Generators that need the hosted teacher (DeepSeek, never Claude or another closed model).
They are written but not run until the DeepSeek balance is topped up and the user asks:
`graft generate <kind> ... --confirm-spend` refuses to start without the flag.

Kinds, each turning an input JSONL file into teacher requests and parsed records:

- `extract`: a paper's own text (`{"paper_id", "title", "licence", "url", "text"}`, licence
  CC-BY-4.0 or CC0-1.0 only: no SA, ND or NC) -> the paper row the research generator needs:
  `problem`, `prior_work` and `actual_path`, extracted by DeepSeek (recorded as `extractor`).
  Claude or a retrieval agent may pick papers; it never writes or extracts their content.
- `research`: an extracted paper -> an Emperor task (only the problem and prior work; the
  reference is the paper's path as atomic steps with checks, dependencies and stages) and a
  Consul task (critique a decomposition with one seeded defect). Papers on the held-out list
  (`--heldout-papers`) are refused: they form the eval split.
- `paraphrase`: a clear task -> ~10 vague or messy prompts with the same intent and plan.
- `coref`: a scenario -> a multi-turn conversation whose later turns use coreference and
  ellipsis, including across a summary that replaced earlier turns.
- `solve`: a research record (a `messages` prefix ending in a user turn) -> the same record
  with the teacher's answer (its reasoning as `reasoning_content`) appended. Research planning
  needs no tools; paraphrase and coref prompts need them, so they go through the hosted
  pipeline (tools, simulator, judge) instead. Records carry `meta.for_model`, and the mixer
  takes only judged ones.
- `ask-dpo`: preference pairs on asking versus guessing (see `ask_pairs`). They are not SFT
  samples (no `messages`), must be judged (`dpo_judge_messages`, Kimi) before use, and need a
  DPO stage (bf16 LoRA, TRL DPOTrainer) that is not written yet.

Every record carries `meta` with the generator, the teacher's licence and, for papers, the
paper's licence and URL for attribution.
"""

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from emperor.data.teacher import TeacherOutputError
from emperor.eval.contract import ALL_TOOLS
from emperor.graft.sources import allowed

TEACHER_LICENCE = "DeepSeek Open Platform Terms of Service 4.2"
OPEN_TEACHERS = ("deepseek-",)  # generation runs on DeepSeek only
PAPER_KINDS = ("extract", "research")
PAPER_LICENCES = ("CC-BY-4.0", "CC0-1.0")
DEFECTS = ("missing_step", "wrong_order", "too_coarse", "unverifiable_claim", "broken_dependency")


class GeneratorError(ValueError):
    """An input row or a teacher answer is unusable."""


def require_open_teacher(model: str) -> str:
    """DeepSeek only, and only models on the generator allow-list."""
    if not model.lower().startswith(OPEN_TEACHERS) or not allowed(model):
        raise GeneratorError(f"{model}: generation runs on allow-listed DeepSeek models only")
    return model


def load_heldout_papers(path: Path | None) -> set[str]:
    if not path:
        return set()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def _id(prefix: str, *parts: str) -> str:
    return f"{prefix}-{hashlib.sha256('|'.join(parts).encode()).hexdigest()[:12]}"


def parse_json(text: str) -> dict:
    """The JSON object in a teacher answer (a fenced block or the bare object)."""
    text = (text or "").strip()
    if "```" in text:
        text = text.split("```", 2)[1]
        text = text.split("\n", 1)[1] if text.startswith(("json", "JSON")) else text
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise GeneratorError("the answer holds no JSON object")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise GeneratorError(f"the answer's JSON does not parse: {error}") from error


def _meta(model: str, **extra) -> dict:
    return {
        "generator": require_open_teacher(model),
        "licence": TEACHER_LICENCE,
        "judged": False,
        **extra,
    }


def _check_paper_licence(row: dict) -> None:
    if row.get("licence") not in PAPER_LICENCES:
        raise GeneratorError(
            f"{row.get('paper_id')}: licence {row.get('licence')} is not one of {PAPER_LICENCES}"
        )


# -- extracting a paper ----------------------------------------------------------------------

EXTRACT_SYSTEM = """You read a research paper and extract, as one JSON object: "problem": the \
problem as it stood before this paper (no hint of its method or results); "prior_work": what \
was known and tried before, as the paper describes it; "actual_path": the ordered steps the \
authors actually took, from hypothesis through experiments to verification, each one short. \
Answer with the JSON object only."""


def extract_messages(paper: dict) -> list[dict]:
    for key in ("paper_id", "title", "licence", "url", "text"):
        if not paper.get(key):
            raise GeneratorError(f"paper row is missing {key}")
    _check_paper_licence(paper)
    return [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": f"# {paper['title']}\n\n{paper['text']}"},
    ]


def extract_records(paper: dict, answer: dict, model: str) -> list[dict]:
    data = parse_json(answer.get("content") or "")
    path = data.get("actual_path")
    if not data.get("problem") or not data.get("prior_work") or not isinstance(path, list):
        raise GeneratorError("the extraction lacks problem, prior_work or actual_path")
    return [
        {
            "id": paper["paper_id"],
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "licence": paper["licence"],
            "url": paper["url"],
            "problem": data["problem"],
            "prior_work": data["prior_work"],
            "actual_path": path,
            "meta": _meta(model, extractor=model),
        }
    ]


# -- research planning ---------------------------------------------------------------------

RESEARCH_SYSTEM = """You write training tasks about research planning. You are given a paper's \
problem statement, the prior work it built on, and the path the paper actually took. Write two \
tasks as one JSON object.

"emperor": {"prompt": the problem and prior work only, phrased as a request to plan the \
investigation (never reveal the paper's method or results), "reference_plan": the paper's \
actual path as a list of atomic steps, each {"id", "action", "check" (a concrete, runnable or \
observable test that the step succeeded), "depends_on": [ids], "stage": "hypothesis" | \
"experiment" | "verification"}}. Steps must be small enough to check one at a time.

"consul": {"plan": the reference plan with exactly one seeded defect of the kind given, \
"defect": {"kind", "step_ids", "explanation"}}.

Answer with the JSON object only."""

CONSUL_RESEARCH_PROMPT = """Critique this research plan for the problem below. Name every \
step that is missing, out of order, too coarse to check, unverifiable, or has a broken \
dependency, with the step ids and a fix.

## Problem
{problem}

## Plan
{plan}"""


def research_messages(paper: dict, defect: str) -> list[dict]:
    for key in ("paper_id", "title", "licence", "url", "problem", "prior_work", "actual_path"):
        if not paper.get(key):
            raise GeneratorError(f"paper row is missing {key}")
    _check_paper_licence(paper)
    if not (paper.get("meta") or {}).get("extractor", "").lower().startswith(OPEN_TEACHERS):
        raise GeneratorError(f"{paper['paper_id']}: not extracted by DeepSeek (run `extract`)")
    if defect not in DEFECTS:
        raise GeneratorError(f"unknown defect {defect}")
    body = {
        "title": paper["title"],
        "problem": paper["problem"],
        "prior_work": paper["prior_work"],
        "actual_path": paper["actual_path"],
        "defect_to_seed": defect,
    }
    return [
        {"role": "system", "content": RESEARCH_SYSTEM},
        {"role": "user", "content": json.dumps(body, ensure_ascii=False, indent=1)},
    ]


def check_plan(steps: list[dict]) -> None:
    ids = [s.get("id") for s in steps]
    if not steps or len(set(ids)) != len(ids):
        raise GeneratorError("plan steps need unique ids")
    known: set = set()
    for step in steps:
        if not step.get("action") or not step.get("check"):
            raise GeneratorError(f"step {step.get('id')} needs an action and a check")
        if step.get("stage") not in ("hypothesis", "experiment", "verification"):
            raise GeneratorError(f"step {step.get('id')} has no valid stage")
        missing = [d for d in step.get("depends_on") or [] if d not in known]
        if missing:
            raise GeneratorError(f"step {step.get('id')} depends on later or unknown {missing}")
        known.add(step["id"])


def research_records(paper: dict, defect: str, answer: dict, model: str) -> list[dict]:
    task = parse_json(answer.get("content") or "")
    emperor, consul = task.get("emperor") or {}, task.get("consul") or {}
    if not emperor.get("prompt") or not consul.get("plan"):
        raise GeneratorError("the answer lacks the emperor prompt or the consul plan")
    check_plan(emperor.get("reference_plan") or [])
    if (consul.get("defect") or {}).get("kind") != defect:
        raise GeneratorError("the seeded defect is not the kind asked for")
    paper_meta = {k: paper[k] for k in ("paper_id", "title", "licence", "url")}
    paper_meta["extractor"] = paper["meta"]["extractor"]
    critique = CONSUL_RESEARCH_PROMPT.format(
        problem=paper["problem"], plan=json.dumps(consul["plan"], ensure_ascii=False, indent=1)
    )
    return [
        {
            "id": _id("rp-e", paper["paper_id"]),
            "kind": "research/emperor",
            "messages": [{"role": "user", "content": emperor["prompt"]}],
            "reference_plan": emperor["reference_plan"],
            "meta": _meta(model, paper=paper_meta, for_model="emperor"),
        },
        {
            "id": _id("rp-c", paper["paper_id"], defect),
            "kind": "research/consul",
            "messages": [{"role": "user", "content": critique}],
            "defect": consul["defect"],
            "reference_plan": emperor["reference_plan"],
            "meta": _meta(model, paper=paper_meta, for_model="consul"),
        },
    ]


# -- messy-prompt paraphrases --------------------------------------------------------------

PARAPHRASE_SYSTEM = """Rewrite a clear multi-step software task the way real users write: \
vague, messy, typo-ridden, out of order, missing context that can be looked up, or wrapped in \
chatter. Write {n} different rewrites. Every rewrite must keep the same intent: a reader who \
investigates the repository can recover exactly the original task. Answer with JSON: \
{{"intent": one-sentence statement of the task, "plan": [short ordered steps], \
"rewrites": [{{"prompt", "style"}}]}}."""


def paraphrase_messages(task: dict, n: int = 10) -> list[dict]:
    if not task.get("id") or not task.get("prompt"):
        raise GeneratorError("task row needs id and prompt")
    return [
        {"role": "system", "content": PARAPHRASE_SYSTEM.format(n=n)},
        {"role": "user", "content": task["prompt"]},
    ]


def paraphrase_records(task: dict, answer: dict, model: str, n: int = 10) -> list[dict]:
    data = parse_json(answer.get("content") or "")
    rewrites = [r for r in data.get("rewrites") or [] if (r.get("prompt") or "").strip()]
    if not data.get("intent") or not data.get("plan") or len(rewrites) < n // 2:
        raise GeneratorError("too few rewrites, or no intent and plan")
    return [
        {
            "id": _id("pp", task["id"], str(i)),
            "kind": "paraphrase",
            "source_task": task["id"],
            "messages": [{"role": "user", "content": r["prompt"]}],
            "style": r.get("style", ""),
            "intent": data["intent"],
            "plan": data["plan"],
            "meta": _meta(model, for_model="emperor"),
        }
        for i, r in enumerate(rewrites)
    ]


# -- asking versus guessing (DPO) ----------------------------------------------------------

ASK_SYSTEM = """You write preference data for Emperor, a root agent that never edits files \
itself: it delegates with spawn_agent (a self-contained brief for a subagent that can read, \
search and run things), searches with web_search and web_fetch, and asks the user with \
ask_user. Given a task, answer with JSON: "hidden_constraint": a decision only the user can \
make (a preference, a policy, a target) that the task depends on and no tool can discover; \
"discoverable_fact": a fact a subagent could look up in the repository; and four assistant \
turns, each {"content": short text, "tool_calls": [{"name", "arguments": {...}}]}: "ask" \
calls ask_user with one sharp question about the hidden constraint only; "guess" invents the \
constraint and acts on it with spawn_agent; "look_up" has a subagent look the discoverable \
fact up with spawn_agent; "needless_question" calls ask_user to ask for the discoverable \
fact."""
# Which tool each turn must use.
TURN_TOOLS = {
    "ask": {"ask_user"},
    "needless_question": {"ask_user"},
    "guess": set(ALL_TOOLS) - {"ask_user"},
    "look_up": set(ALL_TOOLS) - {"ask_user"},
}

DPO_JUDGE_SYSTEM = """You judge one preference pair for a root coding agent. Answer JSON \
{"valid": true|false, "reason"}. Valid only if the chosen turn is clearly better: an ask about \
something no tool can discover beats guessing it; looking up a discoverable fact beats asking \
the user for it; and the chosen turn is itself correct and specific."""


def ask_messages(task: dict) -> list[dict]:
    if not task.get("id") or not task.get("prompt"):
        raise GeneratorError("task row needs id and prompt")
    return [
        {"role": "system", "content": ASK_SYSTEM},
        {"role": "user", "content": task["prompt"]},
    ]


def _assistant_turn(raw, key: str) -> dict:
    """A teacher-written turn as an assistant message with tool calls."""
    if not isinstance(raw, dict) or not raw.get("tool_calls"):
        raise GeneratorError(f"{key} must be an assistant turn with tool calls")
    calls = []
    for i, call in enumerate(raw["tool_calls"]):
        arguments = call.get("arguments")
        if not call.get("name") or not isinstance(arguments, dict):
            raise GeneratorError(f"{key}: each tool call needs a name and object arguments")
        if call["name"] not in TURN_TOOLS.get(key, set(ALL_TOOLS)):
            raise GeneratorError(f"{key}: {call['name']} is not a tool this turn may use")
        calls.append(
            {
                "id": f"call_{key}_{i}",
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(arguments)},
            }
        )
    return {"role": "assistant", "content": raw.get("content") or "", "tool_calls": calls}


def _words(text: str) -> set[str]:
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return {w for w in cleaned.split() if len(w) > 3}


def ask_pairs(task: dict, answer: dict, model: str) -> list[dict]:
    data = parse_json(answer.get("content") or "")
    for key in ("hidden_constraint", "discoverable_fact"):
        if not data.get(key):
            raise GeneratorError(f"the answer lacks {key}")
    turns = {
        key: _assistant_turn(data.get(key), key)
        for key in ("ask", "guess", "look_up", "needless_question")
    }
    fact = _words(data["discoverable_fact"])
    if fact and len(fact & _words(json.dumps(turns["ask"]))) / len(fact) >= 0.5:
        raise GeneratorError("the ask mentions the discoverable fact")
    meta = _meta(model, for_model="emperor", needs_judge=True)
    prompt = [{"role": "user", "content": task["prompt"]}]
    return [
        {
            "id": _id("dpo-ask", task["id"]),
            "kind": "dpo/ask_vs_guess",
            "prompt": prompt,
            "chosen": turns["ask"],
            "rejected": turns["guess"],
            "meta": {**meta, "hidden_constraint": data["hidden_constraint"]},
        },
        {
            "id": _id("dpo-look", task["id"]),
            "kind": "dpo/look_up_vs_ask",
            "prompt": prompt,
            "chosen": turns["look_up"],
            "rejected": turns["needless_question"],
            "meta": {**meta, "discoverable_fact": data["discoverable_fact"]},
        },
    ]


def dpo_judge_messages(pair: dict) -> list[dict]:
    """The judge's request for one pair (Kimi; a different family from the teacher)."""
    body = {k: pair[k] for k in ("prompt", "chosen", "rejected")}
    return [
        {"role": "system", "content": DPO_JUDGE_SYSTEM},
        {"role": "user", "content": json.dumps(body, ensure_ascii=False, indent=1)},
    ]


def apply_dpo_judgement(pair: dict, answer: dict, judge: str) -> dict | None:
    """The pair marked judged, or None when the judge rejects it."""
    verdict = parse_json(answer.get("content") or "")
    if verdict.get("valid") is not True:
        return None
    return {**pair, "meta": {**pair["meta"], "judged": True, "needs_judge": False, "judge": judge}}


# -- coreference and ellipsis --------------------------------------------------------------

COREF_SYSTEM = """Write a multi-turn conversation between a user and a root coding agent in \
which later user turns depend on earlier ones through coreference and ellipsis ("the other \
one", "same for the tests", "and that file?"). Include one point where earlier turns are \
replaced by a compact summary, after which a reference still points into the summarised \
part. End on a user turn. Answer with JSON: {"turns": [{"role": "user" | "assistant" | \
"summary", "content"}], "resolutions": [{"turn": index of a user turn, "refers_to": what the \
reference means}]}."""


def coref_messages(seed: dict) -> list[dict]:
    if not seed.get("id") or not seed.get("scenario"):
        raise GeneratorError("seed row needs id and scenario")
    return [
        {"role": "system", "content": COREF_SYSTEM},
        {"role": "user", "content": seed["scenario"]},
    ]


def coref_records(seed: dict, answer: dict, model: str) -> list[dict]:
    data = parse_json(answer.get("content") or "")
    turns = data.get("turns") or []
    if sum(1 for t in turns if t.get("role") == "user") < 3:
        raise GeneratorError("a coreference task needs at least three user turns")
    if not any(t.get("role") == "summary" for t in turns):
        raise GeneratorError("the conversation has no summary point")
    if not data.get("resolutions"):
        raise GeneratorError("no resolutions recorded")
    if turns[-1].get("role") != "user":
        raise GeneratorError("the conversation must end on a user turn")
    messages: list[dict] = []
    for turn in turns:
        if turn.get("role") == "summary":
            # What compaction leaves: the earlier turns replaced by their summary.
            summary = f"Summary of the conversation so far:\n{turn.get('content') or ''}"
            messages = [{"role": "system", "content": summary}]
        else:
            messages.append({"role": turn["role"], "content": turn.get("content") or ""})
    return [
        {
            "id": _id("coref", seed["id"]),
            "kind": "coref",
            "messages": messages,
            "resolutions": data["resolutions"],
            "meta": _meta(model, for_model="emperor"),
        }
    ]


# -- solving: a record's prefix answered by the teacher -------------------------------------


def solve_messages(record: dict) -> list[dict]:
    if not str(record.get("kind", "")).startswith("research/"):
        raise GeneratorError("solve takes research records only (others need tools)")
    messages = record.get("messages") or []
    if not messages or messages[-1].get("role") != "user":
        raise GeneratorError("solve needs a messages prefix ending in a user turn")
    return messages


def solve_records(record: dict, answer: dict, model: str) -> list[dict]:
    reasoning = (answer.get("reasoning_content") or "").strip()
    content = (answer.get("content") or "").strip()
    if not reasoning or not content:
        raise GeneratorError("the teacher's answer lacks reasoning or content")
    turn = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    meta = {**(record.get("meta") or {}), "solver": require_open_teacher(model)}
    return [{**record, "messages": [*record["messages"], turn], "meta": meta}]


# -- running -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Kind:
    messages: Callable[..., list[dict]]
    records: Callable[..., list[dict]]


KINDS = {
    "extract": Kind(extract_messages, extract_records),
    "research": Kind(research_messages, research_records),
    "paraphrase": Kind(paraphrase_messages, paraphrase_records),
    "ask-dpo": Kind(ask_messages, ask_pairs),
    "coref": Kind(coref_messages, coref_records),
    "solve": Kind(solve_messages, solve_records),
}


def input_text(row: dict) -> str:
    """The user-facing text of an input row, for the held-out overlap check."""
    parts = [row.get(k) or "" for k in ("prompt", "scenario", "problem", "title")]
    parts += [
        m.get("content") or ""
        for m in row.get("messages") or []
        if m.get("role") == "user" and isinstance(m.get("content"), str)
    ]
    return "\n".join(p for p in parts if isinstance(p, str) and p)


def _paper_of(row: dict) -> str | None:
    return row.get("paper_id") or ((row.get("meta") or {}).get("paper") or {}).get("paper_id")


def run(
    kind: str,
    rows: Iterable[dict],
    chat: Callable[[list[dict]], dict],
    model: str,
    done: set[str] | None = None,
    *,
    heldout_papers: set[str] | None = None,
    heldout_tasks=None,
) -> Iterable[tuple[dict | None, str | None]]:
    """For each input row: (record, None) per record made, or (None, error). `chat` is the
    teacher (`HostedClient.chat`, bound to its options); rows whose id is in `done` are skipped,
    so a run resumes. A bad teacher answer for one row is reported and the run goes on."""
    require_open_teacher(model)
    spec = KINDS[kind]
    done = done or set()
    if kind in PAPER_KINDS and heldout_papers is None:
        raise GeneratorError(f"{kind} needs the held-out paper list")
    heldout_papers = heldout_papers or set()
    for n, row in enumerate(rows):
        key = row.get("id") or row.get("paper_id") or str(n)
        if key in done:
            continue
        paper = _paper_of(row)
        if paper and paper in heldout_papers:
            yield None, f"{key}: paper {paper} is held out for the eval split"
            continue
        if heldout_tasks is not None and heldout_tasks.leaks(input_text(row)):
            yield None, f"{key}: overlaps a held-out eval task"
            continue
        try:
            if kind == "research":
                defect = DEFECTS[int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(DEFECTS)]
                answer = chat(spec.messages(row, defect))
                made = spec.records(row, defect, answer, model)
            else:
                answer = chat(spec.messages(row))
                made = spec.records(row, answer, model)
        except (GeneratorError, TeacherOutputError) as error:
            yield None, f"{key}: {error}"
            continue
        for record in made:
            record.setdefault("input_id", key)
            yield record, None
