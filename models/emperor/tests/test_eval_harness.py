"""The harness backend, driven against a stand-in `imperium run --headless` script."""

import json
import sys
import textwrap

import pytest

from emperor.eval.cli import main
from emperor.eval.contract import openai_tools
from emperor.eval.harness import (
    BackendUnavailable,
    HarnessBackend,
    HarnessConfig,
    replay_file,
    to_transcript,
)
from emperor.eval.tasks import load_suite, task_from_dict

SPAWN = {"brief": "Rename read_inventory to load_inventory everywhere.", "tier": "standard"}


def _output(**overrides) -> dict:
    """A schema-version-1 harness result in which the root spawns one agent and answers."""
    output = {
        "schema_version": 1,
        "mode": "run",
        "engine": {"imperium_version": "0.2.0", "protocol_version": "0.2"},
        "variant": {"id": "default", "config_hash": "sha256:00"},
        "prompt_hash": "sha256:abc",
        "tools": openai_tools(["spawn_agent", "await_tasks", "update_plan"]),
        "transcript": [
            {"role": "system", "content": "You are Emperor"},
            {"role": "user", "content": "rename it"},
            {
                "role": "assistant",
                "content": "<think>plan it</think>Delegating.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "spawn_agent", "arguments": json.dumps(SPAWN)},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "spawn_agent", "arguments": '{"tier": 3}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"task_id": "t1"}'},
            {"role": "tool", "tool_call_id": "c2", "content": "Error: invalid"},
            {"role": "assistant", "content": "Renamed in 4 files."},
        ],
        "final_reply": "Renamed in 4 files.",
        "ledger": {"root": {"prompt_tokens": 120, "completion_tokens": 30}},
        "exit_reason": "completed",
        "error": None,
        "replay": {"replies": 1, "unused": []},
        "permission_requests": [],
        "questions_asked": [],
        "compactions": 0,
    }
    output.update(overrides)
    return output


def _fake_imperium(tmp_path, output: dict, code: int = 0, stderr: str = "") -> list[str]:
    """A command that records its arguments and prints `output` as the harness would."""
    script = tmp_path / "fake_imperium.py"
    (tmp_path / "out.json").write_text(json.dumps(output), encoding="utf-8")
    script.write_text(
        textwrap.dedent(
            f"""
            import json, pathlib, sys
            here = pathlib.Path({str(tmp_path)!r})
            args = sys.argv[1:]
            replay = json.loads(pathlib.Path(args[args.index("--subagent-replay") + 1]).read_text())
            task = json.loads(pathlib.Path(args[args.index("--task") + 1]).read_text())
            seen = {{"args": args, "replay": replay, "task": task}}
            (here / "args.json").write_text(json.dumps(seen))
            sys.stderr.write({stderr!r})
            if {code} != 1:
                sys.stdout.write((here / "out.json").read_text())
            sys.exit({code})
            """
        ),
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


def _config(binary: list[str], **overrides) -> HarnessConfig:
    values = {
        "binary": binary,
        "base_url": "http://127.0.0.1:8080/v1",
        "model_id": "emperor",
        "context_tokens": 32768,
    }
    values.update(overrides)
    return HarnessConfig(**values)


def test_replay_file_maps_match_words_roles_and_chunks():
    task = task_from_dict(
        {
            "id": "t",
            "prompt": "p",
            "replies": [
                {"report": "any"},
                {"report": "httpx notes", "match": ["httpx"]},
                {"report": "chunk one", "tool": "delegate_chunks"},
                {"report": "chunk two", "tool": "delegate_chunks"},
            ],
        },
        "heldout",
    )
    replies = replay_file(task)["replies"]
    assert replies[0] == {"brief_contains": "httpx", "report": "httpx notes", "ok": True}
    assert replies[1] == {"role": "task", "report": "any", "ok": True}
    assert replies[2]["role"] == "chunk" and replies[2]["chunk"] == 1
    assert replies[3]["chunk"] == 2
    assert replies[4] == {"role": "reduce", "report": "chunk one\n\nchunk two", "ok": True}


def test_every_bundled_task_converts_to_a_replay_file():
    for suite in ("heldout", "longctx"):
        for task in load_suite(suite):
            assert isinstance(replay_file(task)["replies"], list)


def test_to_transcript_scores_against_the_offered_schemas():
    task = task_from_dict({"id": "t", "prompt": "p"}, "heldout")
    transcript = to_transcript(task, _output())
    assert [c.name for c in transcript.calls] == ["spawn_agent", "spawn_agent"]
    assert transcript.calls[0].valid and transcript.calls[0].result == '{"task_id": "t1"}'
    assert not transcript.calls[1].valid
    assert transcript.replies[0].content == "Delegating."
    assert transcript.replies[0].reasoning == "plan it"
    assert transcript.final == "Renamed in 4 files."
    assert transcript.prompt_tokens == 120 and transcript.error is None


def test_to_transcript_flags_a_tool_the_engine_did_not_offer():
    task = task_from_dict({"id": "t", "prompt": "p"}, "heldout")
    output = _output()
    output["transcript"][2]["tool_calls"][0]["function"]["name"] = "made_up"
    assert not to_transcript(task, output).calls[0].valid


def test_a_run_that_did_not_complete_is_an_error():
    task = task_from_dict({"id": "t", "prompt": "p"}, "heldout")
    transcript = to_transcript(task, _output(exit_reason="step_limit", error="50 model calls"))
    assert transcript.error == "step_limit: 50 model calls"


def test_backend_passes_the_run_options(tmp_path):
    binary = _fake_imperium(tmp_path, _output())
    config = _config(binary, variant="lean", flags=["max_steps=30"], thinking="on")
    task = task_from_dict(
        {"id": "t", "prompt": "rename it", "replies": [{"report": "done", "match": ["rename"]}]},
        "heldout",
    )
    run = HarnessBackend(config).run_task(task)
    seen = json.loads((tmp_path / "args.json").read_text())
    args = seen["args"]
    assert args[:3] == ["run", "--headless", "--json"]
    assert args[args.index("--root-model") + 1] == "http://127.0.0.1:8080/v1#emperor"
    assert args[args.index("--context-tokens") + 1] == "32768"
    assert args[args.index("--variant") + 1] == "lean"
    assert args[args.index("--flag") + 1] == "max_steps=30"
    assert args[args.index("--thinking") + 1] == "on"
    assert "--local" in args
    assert seen["task"] == {"id": "t", "prompt": "rename it"}
    assert seen["replay"]["replies"][0]["brief_contains"] == "rename"
    assert run.prompt_hash == "sha256:abc" and run.transcript.final == "Renamed in 4 files."


def test_backend_keeps_a_run_that_ended_early(tmp_path):
    binary = _fake_imperium(tmp_path, _output(exit_reason="timeout", error="1800 s"), code=3)
    task = task_from_dict({"id": "t", "prompt": "p"}, "heldout")
    assert HarnessBackend(_config(binary)).run_task(task).transcript.error == "timeout: 1800 s"


def test_backend_names_an_imperium_without_headless_run(tmp_path):
    binary = _fake_imperium(tmp_path, {}, code=2, stderr="error: unrecognized subcommand 'run'\n")
    task = task_from_dict({"id": "t", "prompt": "p"}, "heldout")
    with pytest.raises(BackendUnavailable, match="no `run --headless`"):
        HarnessBackend(_config(binary)).run_task(task)


def test_cli_harness_report(tmp_path, monkeypatch):
    binary = _fake_imperium(tmp_path, _output())
    monkeypatch.setattr("emperor.eval.cli.default_imperium", lambda: binary[0])
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"id": "t", "prompt": "rename it"}), encoding="utf-8")
    report = tmp_path / "report.json"
    # The stand-in is a Python script, so the binary is the interpreter plus the script; the
    # CLI takes one path, so run it through a one-line wrapper module.
    wrapper = tmp_path / "imperium.cmd" if sys.platform == "win32" else tmp_path / "imperium"
    if sys.platform == "win32":
        wrapper.write_text(f'@"{binary[0]}" "{binary[1]}" %*\n', encoding="utf-8")
    else:
        wrapper.write_text(f'#!/bin/sh\nexec "{binary[0]}" "{binary[1]}" "$@"\n', encoding="utf-8")
        wrapper.chmod(0o755)
    code = main(
        [
            "--model", "llamacpp:emperor", "--tasks", str(task), "--imperium", str(wrapper),
            "--context-tier", "32k", "--output", str(report),
        ]
    )  # fmt: skip
    assert code == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["backend"] == "harness"
    assert data["harness"]["prompt_hashes"] == ["sha256:abc"]
    assert data["tasks"][0]["harness"]["prompt_hash"] == "sha256:abc"
