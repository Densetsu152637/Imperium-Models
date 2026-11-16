"""harness backend: run each task through Imperium's real agent loop with `imperium run --headless`.

The model is the root agent of an ordinary Imperium session: the engine's own root prompt, tool
schemas and loop (Imperium-AI docs/03 "Running evals through the harness"), not the vendored
copy the replay backend uses. Subagents are answered from the task's recordings
(`--subagent-replay`), so a run is deterministic and calls no hosted model. The harness output
is turned back into a `Transcript`, so every metric scores it unchanged.

Differences from the replay backend, set by the harness rather than by this package:
- the root is offered the engine's whole root tool set, not only the task's `tools`;
- `ask_user` is answered with the recommended answers (the task's `user_answers` are unused);
- `web_search` returns an error and `web_fetch` is denied, so the task's `web` is unused;
- the model's reasoning text is not kept in the transcript.
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from emperor.eval.contract import SchemaError, validate
from emperor.eval.models import Reply, ToolCall, split_thinking
from emperor.eval.runner import CallRecord, Transcript
from emperor.eval.tasks import Task, render_prompt


class BackendUnavailable(RuntimeError):
    """The harness cannot run (no `imperium` binary, or one without `run --headless`)."""


@dataclass
class HarnessConfig:
    binary: list[str]  # the `imperium` command, e.g. ["imperium"]
    base_url: str
    model_id: str
    context_tokens: int
    variant: str | None = None
    flags: list[str] = field(default_factory=list)
    thinking: str | None = None  # "on" or "off"; None leaves the endpoint's default
    local: bool = True
    timeout_s: int = 1800
    api_key_env: str | None = None


def replay_file(task: Task) -> dict:
    """The task's recordings in the harness's replay format.

    A recording with a `match` word answers the first subagent whose brief contains it; one
    without answers any task subagent. The harness does not use a recording up, so the order is
    matched first, then unmatched. `delegate_chunks` recordings answer chunk 1, 2, ... in order,
    and the merge agent gets their reports joined.
    """
    matched, unmatched, chunks = [], [], []
    for reply in task.replies:
        if reply.get("tool", "spawn_agent") == "delegate_chunks":
            chunks.append(reply["report"])
            continue
        words = reply.get("match", [])
        if words:
            if len(words) > 1:
                raise ValueError(f"task {task.id}: the harness matches one word per recording")
            matched.append({"brief_contains": words[0], "report": reply["report"], "ok": True})
        else:
            unmatched.append({"role": "task", "report": reply["report"], "ok": True})
    replies = matched + unmatched
    for number, report in enumerate(chunks, start=1):
        replies.append({"role": "chunk", "chunk": number, "report": report, "ok": True})
    if chunks:
        replies.append({"role": "reduce", "report": "\n\n".join(chunks), "ok": True})
    return {"replies": replies}


def _parse_arguments(
    name: str, raw: str, schemas: dict[str, dict]
) -> tuple[dict | None, list[str]]:
    """Parse one call and validate it against the schema the engine offered (from the run's
    output, so a variant's renames and the engine's own tools are covered)."""
    if name not in schemas:
        return None, [f"tool {name!r} was not offered"]
    try:
        arguments = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as error:
        return None, [f"arguments are not JSON: {error.msg}"]
    try:
        return arguments, validate(arguments, schemas[name])
    except SchemaError:
        # A keyword our small validator does not know: keep the parse check only.
        return arguments, [] if isinstance(arguments, dict) else ["arguments are not an object"]


def to_transcript(task: Task, output: dict) -> Transcript:
    """Rebuild the scored transcript from the harness's OpenAI-shaped root conversation."""
    transcript = Transcript(task_id=task.id)
    schemas = {
        t["function"]["name"]: t["function"].get("parameters") or {}
        for t in output.get("tools", [])
        if "function" in t
    }
    results = {
        m.get("tool_call_id"): m.get("content") or ""
        for m in output.get("transcript", [])
        if m.get("role") == "tool"
    }
    turn = 0
    for message in output.get("transcript", []):
        if message.get("role") != "assistant":
            continue
        reasoning, content = split_thinking(message.get("content") or "")
        calls = [
            ToolCall(
                id=c.get("id", ""),
                name=c["function"]["name"],
                arguments=c["function"].get("arguments") or "",
            )
            for c in message.get("tool_calls") or []
        ]
        transcript.replies.append(Reply(content=content, tool_calls=calls, reasoning=reasoning))
        for call in calls:
            arguments, errors = _parse_arguments(call.name, call.arguments, schemas)
            transcript.calls.append(
                CallRecord(
                    turn, call.name, call.arguments, arguments, errors, results.get(call.id, "")
                )
            )
        turn += 1
    _, transcript.final = split_thinking(output.get("final_reply") or "")
    root = output.get("ledger", {}).get("root", {})
    transcript.prompt_tokens = root.get("prompt_tokens", 0)
    transcript.completion_tokens = root.get("completion_tokens", 0)
    reason = output.get("exit_reason")
    if reason != "completed":
        detail = output.get("error") or ""
        transcript.error = f"{reason}: {detail}".rstrip(": ")
        transcript.overflowed = "context" in detail.lower()
    return transcript


@dataclass
class HarnessRun:
    """What the report keeps of one harness run besides the transcript."""

    transcript: Transcript
    prompt_hash: str | None
    variant: dict
    engine: dict
    replay_unused: list[int]
    permission_requests: int
    questions_asked: int
    compactions: int


class HarnessBackend:
    def __init__(self, config: HarnessConfig):
        self.config = config

    def command(self, task_file: Path, replay: Path, workspace: Path) -> list[str]:
        c = self.config
        cmd = [
            *c.binary,
            "run",
            "--headless",
            "--json",
            "--task",
            str(task_file),
            "--workspace",
            str(workspace),
            "--root-model",
            f"{c.base_url}#{c.model_id}",
            "--context-tokens",
            str(c.context_tokens),
            "--subagent-replay",
            str(replay),
            "--timeout-s",
            str(c.timeout_s),
        ]
        if c.local:
            cmd.append("--local")
        if c.thinking:
            cmd += ["--thinking", c.thinking]
        if c.variant:
            cmd += ["--variant", c.variant]
        if c.api_key_env:
            cmd += ["--api-key-env", c.api_key_env]
        for flag in c.flags:
            cmd += ["--flag", flag]
        return cmd

    def run_task(self, task: Task) -> HarnessRun:
        with tempfile.TemporaryDirectory(prefix=f"emperor-eval-{task.id}-") as scratch:
            root = Path(scratch)
            workspace = root / "workspace"
            workspace.mkdir()
            task_file = root / "task.json"
            task_file.write_text(
                json.dumps(
                    {"id": task.id, "prompt": render_prompt(task, self.config.context_tokens)}
                ),
                encoding="utf-8",
            )
            replay = root / "replay.json"
            replay.write_text(json.dumps(replay_file(task)), encoding="utf-8")
            cmd = self.command(task_file, replay, workspace)
            try:
                done = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=self.config.timeout_s + 60,
                )
            except FileNotFoundError as error:
                raise BackendUnavailable(f"cannot run {self.config.binary[0]}") from error
            except subprocess.TimeoutExpired:
                transcript = Transcript(task_id=task.id, error="timeout: the harness did not exit")
                return HarnessRun(transcript, None, {}, {}, [], 0, 0, 0)
        if done.returncode not in (0, 3):
            stderr = done.stderr.strip()
            if "unrecognized subcommand 'run'" in stderr:
                raise BackendUnavailable(
                    f"{self.config.binary[0]} has no `run --headless` "
                    "(needs Imperium-AI bdc237a or later)"
                )
            raise BackendUnavailable(f"the harness could not start: {stderr[-500:]}")
        output = json.loads(done.stdout)
        return HarnessRun(
            transcript=to_transcript(task, output),
            prompt_hash=output.get("prompt_hash"),
            variant=output.get("variant") or {},
            engine=output.get("engine") or {},
            replay_unused=(output.get("replay") or {}).get("unused", []),
            permission_requests=len(output.get("permission_requests") or []),
            questions_asked=len(output.get("questions_asked") or []),
            compactions=output.get("compactions") or 0,
        )
