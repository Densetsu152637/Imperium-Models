"""Benchmark CLI: help and a tiny CPU run."""

from __future__ import annotations

import json

import pytest

from ncm.bench import main, verdict


def test_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--context" in capsys.readouterr().out


def test_verdict_thresholds() -> None:
    assert verdict(1.5) == "PASS"
    assert verdict(1.3) == "WEAK"
    assert verdict(1.19) == "FAIL"


def test_tiny_cpu_run(capsys: pytest.CaptureFixture[str], tmp_path) -> None:
    out = tmp_path / "bench.json"
    argv = ["--tiny", "--device", "cpu", "--dtype", "fp32", "--context", "64", "--new-tokens", "8"]
    assert main([*argv, "--repeats", "1", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "tok/s" in text and "ratio ncm/tok @ ctx 64" in text
    data = json.loads(out.read_text())
    assert {r["name"] for r in data["results"]} == {"tok", "ncm"}
    assert len(data["ratios"]) == 1
