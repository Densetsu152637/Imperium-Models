"""Harness arm backend: Imperium's own agent loop via `imperium run --headless`.

Each item gets a fresh scratch COPY of the knowledge base as plain markdown files (kb_files.py) used as the
workspace; the agent finds facts with its own tools. Harness aids stay at their `--local` defaults. The same
llama-server and model as the other arms; thinking off. Errors and timeouts are recorded and count as wrong.
"""
import json
import os
import pathlib
import shutil
import subprocess
import tempfile

import kb_files
import suite_cfg

HERE = pathlib.Path(__file__).parent
IMPERIUM = os.environ.get("IMPERIUM_EXE", r"C:\Users\Nicholas\AppData\Local\Imperium\imperium.exe")
ENDPOINT = "http://127.0.0.1:8099/v1"
MODEL_ID = "qwen3.5-4b"
CONTEXT_TOKENS = 16384
TIMEOUT_S = 300
MAX_STEPS = 20
PERMISSION_MODE = "default"  # a call that would ask is denied; the agent needs only read tools
FLAGS = ["--local", "--thinking", "off", "--context-tokens", str(CONTEXT_TOKENS), "--timeout-s", str(TIMEOUT_S),
         "--flag", f"max_steps={MAX_STEPS}", "--permission-mode", PERMISSION_MODE]

INSTRUCTION = ("Answer the question below. Facts about the fictional world it concerns are in the workspace files "
               "(start with INDEX.md). Then end your reply with a final line exactly of the form\n"
               "Final answer: <answer>\nwhere <answer> is as short as possible: a number (no units, no thousands "
               "separators), a name, or a short phrase. If you do not know the answer, or it cannot be determined "
               "from the information available, end with\nFinal answer: Unknown")

NOTE = (" Questions about the real world, pure arithmetic or pure logic can be answered from your own knowledge "
        "and reasoning; only facts about the fictional world need the workspace files.")

_scratch = pathlib.Path(tempfile.gettempdir()) / "ghp-workspaces"


def _compact_trace(transcript):
    out = []
    for m in transcript or []:
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", {})
            out.append({"tool": fn.get("name"), "args": str(fn.get("arguments"))[:600]})
    return out


def run_imperium(task, bm25, meter, note=False, variant=None, permission="default"):
    ws = _scratch / task["id"]
    kb_files.write(ws, suite_cfg.KB)
    prompt = f"{INSTRUCTION}{NOTE if note else ''}\n\nQuestion: {task['question']}"
    cmd = [IMPERIUM, "run", "--headless", "--json", "--root-model", ENDPOINT, "--model-id", MODEL_ID,
           "--workspace", str(ws), "--task", prompt] + FLAGS
    if permission != "default":
        # `full` lets the agent run commands (python3 for arithmetic) in its scratch workspace copy; the engine's
        # safety floor still stops destructive commands. In `default`, commands ask and are denied headless.
        cmd[cmd.index("--permission-mode") + 1] = permission
    if variant:
        cmd += ["--config-home", str(HERE), "--variant", variant]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=TIMEOUT_S + 60)
        code, stdout, stderr = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        shutil.rmtree(ws, ignore_errors=True)
        return {"final_text": "", "exit_code": None, "exit_reason": "outer_timeout", "stop_reason": "outer_timeout"}
    shutil.rmtree(ws, ignore_errors=True)
    try:
        j = json.loads(stdout)
    except Exception:
        return {"final_text": "", "exit_code": code, "exit_reason": "no_json", "stop_reason": "no_json",
                "stderr": (stderr or "")[:500]}
    tot = j.get("ledger", {}).get("total", {})
    meter.calls += tot.get("model_calls", 0)
    meter.prompt_tokens += tot.get("prompt_tokens", 0)
    meter.completion_tokens += tot.get("completion_tokens", 0)
    trace = _compact_trace(j.get("transcript"))
    return {"final_text": j.get("final_reply") or "", "exit_code": code, "exit_reason": j.get("exit_reason"),
            "stop_reason": j.get("exit_reason"), "tool_calls": j.get("root_tool_calls"),
            "failed_tool_calls": j.get("failed_tool_calls"), "imperium_wall_ms": j.get("wall_time_ms"),
            "prompt_hash": j.get("prompt_hash"), "variant_hash": (j.get("variant") or {}).get("config_hash"),
            "variant_warnings": (j.get("variant") or {}).get("warnings"), "cached_tokens": tot.get("cached_tokens"),
            "trace": trace, "output": j.get("final_reply"), "runtime_error": j.get("error"),
            "endpoint_errors": len(j.get("endpoint_errors") or [])}


def run_imperium_note(task, bm25, meter):
    """Sensitivity variant: the prompt carries the same real-world/arithmetic/logic note the rag arm has."""
    return run_imperium(task, bm25, meter, note=True)


def run_imperium_refined(task, bm25, meter):
    """Imperium with the refined harness variant (config/harness-variants/<name>.toml); neutral task prompt."""
    return run_imperium(task, bm25, meter, variant=os.environ.get("HARNESS_VARIANT", "refined"), permission="full")


def run_imperium_full(task, bm25, meter):
    """Stock Imperium, commands allowed (permission mode full)."""
    return run_imperium(task, bm25, meter, permission="full")


def run_imperium_note_full(task, bm25, meter):
    return run_imperium(task, bm25, meter, note=True, permission="full")


def run_imperium_refined_v4(task, bm25, meter):
    return run_imperium(task, bm25, meter, variant="refined_v4", permission="full")
