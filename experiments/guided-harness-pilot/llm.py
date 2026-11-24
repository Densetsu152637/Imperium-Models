"""Thin client for llama-server's OpenAI-compatible endpoint (stdlib only). Thinking is always off."""
import json
import time
import urllib.request

URL = "http://127.0.0.1:8099/v1/chat/completions"


class Meter:
    """Accumulates model calls and token usage for one item."""

    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0


def chat(messages, meter, max_tokens=1024, temperature=0.0, seed=None, schema=None, timeout=600):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}, "stream": False}
    if seed is not None:
        body["seed"] = seed
    if temperature > 0:
        body["top_p"] = 0.95
    if schema is not None:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "step", "strict": True, "schema": schema}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    meter.calls += 1
    u = data.get("usage", {})
    meter.prompt_tokens += u.get("prompt_tokens", 0)
    meter.completion_tokens += u.get("completion_tokens", 0)
    return data["choices"][0]["message"].get("content") or ""
