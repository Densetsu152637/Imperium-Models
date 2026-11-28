"""Imperium backend for `uv run eval`: Consul answers through Imperium's headless agent loop.

This is the R5 eval H2 (Imperium-Research `docs/emperor-consul-harnesses/design.md`). Each
held-out task runs as one `imperium run --headless --json` turn. The task's artifact and sources
go in the user message exactly as `prompts.user_message` builds them, and the critique is read
from the reply. The workspace is an empty scratch folder per task: Consul needs no project files,
and an empty folder means the agent can never touch a real repository.

The Consul harness is chosen with `harness` (`--harness <profile>`) once Imperium ships harness
profiles. Until then `harness=None` runs a stand-in: the stock loop with Consul's system prompt
appended as a prompt overlay. The stand-in measures the prompt, not the harness, and its runs are
labelled with the backend `imperium-overlay` so the two are never mixed up in a report.
"""

import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from consul import prompts
from consul.tasks import CritiqueTask

DEFAULT_TIMEOUT_S = 600


@dataclass
class ImperiumSettings:
    exe: Path
    root_model: str  # an OpenAI-compatible endpoint, as `--root-model` takes it
    model_id: str | None = None
    harness: str | None = None  # a harness profile; None runs the prompt-overlay stand-in
    permission_mode: str = "plan"  # Consul only reads
    local: bool = True
    thinking: str | None = None  # "on" | "off" | None (leave the model's default)
    context_tokens: int | None = None
    timeout_s: int = DEFAULT_TIMEOUT_S
    extra_args: Sequence[str] = field(default_factory=tuple)

    @property
    def backend(self) -> str:
        return f"imperium-harness:{self.harness}" if self.harness else "imperium-overlay"


@dataclass
class RunRecord:
    """What one task's run returned, kept beside the score so a report can say which build ran."""

    task_id: str
    exit_code: int | None
    exit_reason: str | None
    prompt_hash: str | None
    engine: dict | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


def build_command(
    s: ImperiumSettings, task: CritiqueTask, workspace: Path, overlay: Path | None
) -> list[str]:
    cmd = [str(s.exe), "run", "--headless", "--json", "--root-model", s.root_model]
    if s.model_id:
        cmd += ["--model-id", s.model_id]
    cmd += ["--workspace", str(workspace), "--task", prompts.user_message(task)]
    cmd += ["--permission-mode", s.permission_mode, "--timeout-s", str(s.timeout_s)]
    if s.local:
        cmd.append("--local")
    if s.thinking:
        cmd += ["--thinking", s.thinking]
    if s.context_tokens:
        cmd += ["--context-tokens", str(s.context_tokens)]
    if s.harness:
        cmd += ["--harness", s.harness]
    elif overlay is not None:
        cmd += ["--prompt-overlay", str(overlay)]
    cmd += list(s.extra_args)
    return cmd


def parse_result(task_id: str, code: int | None, stdout: str) -> tuple[str, RunRecord]:
    """The reply text and the run record; an unreadable result gives an empty reply."""
    try:
        j = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return "", RunRecord(task_id, code, "no_json", None, None, error="no JSON on stdout")
    total = (j.get("ledger") or {}).get("total") or {}
    record = RunRecord(
        task_id=task_id,
        exit_code=code,
        exit_reason=j.get("exit_reason"),
        prompt_hash=j.get("prompt_hash"),
        engine=j.get("engine"),
        prompt_tokens=total.get("prompt_tokens", 0),
        completion_tokens=total.get("completion_tokens", 0),
        error=j.get("error"),
    )
    return j.get("final_reply") or "", record


def imperium_answer(
    s: ImperiumSettings,
    records: list[RunRecord],
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Callable[[CritiqueTask], str]:
    """An `answer(task)` for `run_suite` that appends each task's run record to `records`."""

    def answer(task: CritiqueTask) -> str:
        scratch = Path(tempfile.mkdtemp(prefix="consul-h2-"))
        workspace = scratch / "workspace"
        workspace.mkdir()
        overlay = None
        if not s.harness:
            overlay = scratch / "consul-overlay.md"
            overlay.write_text(prompts.CONSUL_SYSTEM + "\n", encoding="utf-8")
        try:
            p = run(
                build_command(s, task, workspace, overlay),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=s.timeout_s + 60,
            )
            text, record = parse_result(task.id, p.returncode, p.stdout)
        except subprocess.TimeoutExpired:
            text, record = "", RunRecord(task.id, None, "outer_timeout", None, None)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        records.append(record)
        return text

    return answer
