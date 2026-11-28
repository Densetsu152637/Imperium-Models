"""The Imperium backend of `uv run eval` (R5 eval H2), with a fake `imperium` process."""

import json
import subprocess
from pathlib import Path

from consul.eval.imperium import ImperiumSettings, build_command, imperium_answer, parse_result
from consul.score import reference_critique, score_critique
from consul.tasks import make_task


def _settings(**kw) -> ImperiumSettings:
    return ImperiumSettings(exe=Path("imperium"), root_model="http://127.0.0.1:8080/v1", **kw)


def test_overlay_stand_in_without_a_harness(tmp_path):
    task = make_task(7, held=True)
    cmd = build_command(_settings(model_id="consul"), task, tmp_path, tmp_path / "o.md")
    assert cmd[:4] == ["imperium", "run", "--headless", "--json"]
    assert cmd[cmd.index("--prompt-overlay") + 1] == str(tmp_path / "o.md")
    assert "--harness" not in cmd
    assert cmd[cmd.index("--permission-mode") + 1] == "plan"
    assert task.artifact in cmd[cmd.index("--task") + 1]
    assert _settings().backend == "imperium-overlay"


def test_harness_profile_replaces_the_overlay(tmp_path):
    s = _settings(harness="consul")
    cmd = build_command(s, make_task(3, held=True), tmp_path, tmp_path / "o.md")
    assert cmd[cmd.index("--harness") + 1] == "consul"
    assert "--prompt-overlay" not in cmd
    assert s.backend == "imperium-harness:consul"


def test_answer_scores_the_reply_and_records_the_run():
    task = make_task(11, held=True)
    reply = json.dumps(reference_critique(task))
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        overlay = Path(cmd[cmd.index("--prompt-overlay") + 1])
        seen["overlay"] = overlay.read_text(encoding="utf-8")
        out = {
            "final_reply": reply,
            "exit_reason": "done",
            "prompt_hash": "sha256:abc",
            "engine": {"imperium_version": "0.2.0"},
            "ledger": {"total": {"prompt_tokens": 120, "completion_tokens": 30}},
        }
        return subprocess.CompletedProcess(cmd, 0, json.dumps(out), "")

    records = []
    text = imperium_answer(_settings(), records, run=fake_run)(task)
    assert (
        score_critique(task, json.loads(text)).to_dict()
        == score_critique(task, reference_critique(task)).to_dict()
    )
    assert "Consul" in seen["overlay"]
    workspace = Path(seen["cmd"][seen["cmd"].index("--workspace") + 1])
    assert not workspace.exists()  # the scratch folder is removed after the run
    (r,) = records
    assert (r.exit_reason, r.prompt_hash, r.prompt_tokens) == ("done", "sha256:abc", 120)


def test_unreadable_output_is_an_empty_reply():
    text, record = parse_result("t", 1, "not json")
    assert text == "" and record.exit_reason == "no_json"


def test_timeout_is_recorded():
    def fake_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)

    records = []
    assert imperium_answer(_settings(), records, run=fake_run)(make_task(5, held=True)) == ""
    assert records[0].exit_reason == "outer_timeout"
