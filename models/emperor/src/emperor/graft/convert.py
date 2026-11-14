"""Open-dataset records -> training samples in our chat format.

A training sample is `{"id", "task_class", "tools", "messages", "meta"}`. Messages are
OpenAI-style (`system`/`user`/`assistant`/`tool`), the assistant's thinking is in
`reasoning_content`, tool-call arguments are JSON text, and `tools` holds full OpenAI function
schemas (our own samples name Emperor's tools instead). `meta` records the source, licence,
generator and URL of every sample.

A sample is dropped, with the reason counted, when its generator is blocked or unknown, when the
first assistant turn after the last user message has no reasoning (a thinking base would learn
to act without thinking), or when its thinking is longer than the cap.
"""

import hashlib
import json
import re

from emperor.graft.sources import Source, blocked

BLOCKED = "blocked_generator"
NO_REASONING = "no_reasoning"
LONG_THINKING = "thinking_over_cap"
MALFORMED = "malformed"
_THINK = re.compile(r"^\s*<think>(.*?)</think>\s*(.*)$", re.DOTALL)
CHARS_PER_TOKEN = 4  # rough, for the thinking cap only


def approx_tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)


def sample_id(source: Source, record: dict, messages: list[dict]) -> str:
    key = record.get("uuid") or (record.get("metadata") or {}).get("uuid")
    if not key:
        key = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    return f"g-{source.key}-{hashlib.sha256(str(key).encode()).hexdigest()[:12]}"


def _arguments(arguments) -> str:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        json.loads(arguments or "{}")  # must parse
        return arguments or "{}"
    return json.dumps(arguments, ensure_ascii=False)


def _tool_calls(calls: list[dict], turn: int) -> list[dict]:
    out = []
    for n, call in enumerate(calls):
        function = call["function"]
        out.append(
            {
                "id": call.get("id") or f"call_{turn}_{n}",
                "type": "function",
                "function": {
                    "name": function["name"],
                    "arguments": _arguments(function.get("arguments")),
                },
            }
        )
    return out


def _tool_schema(tool: dict) -> dict:
    function = dict(tool.get("function") or tool)
    parameters = function.get("parameters") or {"type": "object", "properties": {}}
    if isinstance(parameters, str):
        parameters = json.loads(parameters)
    return {
        "type": "function",
        "function": {
            "name": function["name"],
            "description": function.get("description") or "",
            "parameters": parameters,
        },
    }


def last_query(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            return i
    return -1


def thinks_first(messages: list[dict]) -> bool:
    """The first assistant turn after the last user message carries reasoning."""
    for message in messages[last_query(messages) + 1 :]:
        if message["role"] == "assistant":
            return bool((message.get("reasoning_content") or "").strip())
    return False


def max_thinking_tokens(messages: list[dict]) -> int:
    return max(
        (
            approx_tokens(m.get("reasoning_content") or "")
            for m in messages
            if m["role"] == "assistant"
        ),
        default=0,
    )


def convert_nemotron(
    record: dict, source: Source, max_thinking: int
) -> tuple[dict | None, str | None]:
    """A Nemotron agentic record -> (sample, None) or (None, why it was dropped)."""
    generator = source.generator_of(record)
    if blocked(generator):
        return None, BLOCKED
    try:
        call_ids: list[str] = []
        messages = []
        for turn, message in enumerate(record["messages"]):
            role = message["role"]
            if role in ("system", "user"):
                if role == "system" and not (message.get("content") or "").strip():
                    continue  # an empty system prompt adds nothing
                messages.append({"role": role, "content": (message.get("content") or "").strip()})
            elif role == "assistant":
                out = {"role": "assistant", "content": message.get("content") or ""}
                reasoning = (message.get("reasoning_content") or "").strip()
                if reasoning:
                    out["reasoning_content"] = reasoning
                if message.get("tool_calls"):
                    out["tool_calls"] = _tool_calls(message["tool_calls"], turn)
                    call_ids = [c["id"] for c in out["tool_calls"]]
                messages.append(out)
            elif role == "tool":
                content = message.get("content")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                tool_id = message.get("tool_call_id") or (call_ids.pop(0) if call_ids else None)
                if tool_id is None:
                    return None, MALFORMED
                if tool_id in call_ids:
                    call_ids.remove(tool_id)
                messages.append({"role": "tool", "tool_call_id": tool_id, "content": content})
            else:
                return None, MALFORMED
        tools = [_tool_schema(t) for t in record.get("tools") or []]
    except (KeyError, TypeError, ValueError):
        return None, MALFORMED
    if not any(m["role"] == "user" for m in messages) or messages[-1]["role"] != "assistant":
        return None, MALFORMED
    if not thinks_first(messages):
        return None, NO_REASONING
    if max_thinking_tokens(messages) > max_thinking:
        return None, LONG_THINKING
    return {
        "id": sample_id(source, record, messages),
        "task_class": f"graft/{source.key}",
        "tools": tools,
        "messages": messages,
        "meta": {
            "source": source.dataset,
            "subset": source.key,
            "licence": source.licence,
            "generator": generator,
            "url": source.url,
        },
    }, None


def convert_openthoughts(
    record: dict, source: Source, max_thinking: int
) -> tuple[dict | None, str | None]:
    """An OpenThoughts3 row (`conversations` of from/value, the answer holding `<think>`)."""
    generator = source.generator_of(record)
    if blocked(generator):
        return None, BLOCKED
    turns = record.get("conversations") or []
    if len(turns) != 2 or turns[0].get("from") not in ("human", "user"):
        return None, MALFORMED
    match = _THINK.match(turns[1].get("value") or "")
    if not match or not match.group(1).strip() or not match.group(2).strip():
        return None, NO_REASONING
    reasoning, answer = match.group(1).strip(), match.group(2).strip()
    if approx_tokens(reasoning) > max_thinking:
        return None, LONG_THINKING
    messages = [
        {"role": "user", "content": turns[0]["value"].strip()},
        {"role": "assistant", "content": answer, "reasoning_content": reasoning},
    ]
    return {
        "id": sample_id(source, {"uuid": turns[0]["value"]}, messages),
        "task_class": f"graft/{source.key}/{record.get('domain') or 'unknown'}",
        "tools": [],
        "messages": messages,
        "meta": {
            "source": source.dataset,
            "subset": source.key,
            "licence": source.licence,
            "generator": generator,
            "url": source.url,
            "domain": record.get("domain"),
            "upstream_source": record.get("source"),
        },
    }, None


CONVERTERS = {"nemotron": convert_nemotron, "openthoughts": convert_openthoughts}


def convert(record: dict, source: Source, max_thinking: int) -> tuple[dict | None, str | None]:
    return CONVERTERS[source.kind](record, source, max_thinking)
