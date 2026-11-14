import json
import tomllib
from pathlib import Path

from emperor.train import stages
from emperor.train.cli import main as train_main
from emperor.train.config import parse_config

ROOT = Path(__file__).resolve().parent.parent


def test_caps_rise_a_step_at_a_time_up_to_the_longest_sample():
    assert stages.stage_caps(19453, 12288, 1024, 32768) == [
        13312, 14336, 15360, 16384, 17408, 18432, 19456,
    ]  # fmt: skip
    assert stages.stage_caps(12000, 12288, 1024, 32768) == []  # nothing was dropped
    assert stages.stage_caps(40000, 12288, 8192, 24576) == [20480, 24576]  # never past max


def records(lengths):
    return [{"id": f"s{i}", "n": n} for i, n in enumerate(lengths)], {
        f"s{i}": n for i, n in enumerate(lengths)
    }


def test_a_stage_takes_its_length_band_once_plus_a_short_replay():
    train, lengths = records([1000, 2000, 3000, 13000, 13300, 14000, 0])
    long, replay = stages.plan_stage(train, lengths, 12288, 13312, set(), 12288, 1.0, 1)
    assert [r["id"] for r in long] == ["s3", "s4"]
    assert len(replay) == 2 and all(lengths[r["id"]] <= 12288 for r in replay)
    again, _ = stages.plan_stage(train, lengths, 12288, 13312, {"s3"}, 12288, 1.0, 1)
    assert [r["id"] for r in again] == ["s4"]  # never trained twice
    long, replay = stages.plan_stage(train, lengths, 13312, 14336, set(), 12288, 0.5, 1)
    assert [r["id"] for r in long] == ["s5"] and len(replay) == 0  # round(0.5) == 0
    one, _ = stages.plan_stage(train, lengths, 12288, 13312, set(), 12288, 1.0, 1)
    two, _ = stages.plan_stage(train, lengths, 12288, 13312, set(), 12288, 1.0, 1)
    assert one == two  # deterministic


def test_stage_and_final_configs(tmp_path):
    raw = tomllib.loads((ROOT / "configs" / "emperor-4b.toml").read_text(encoding="utf-8"))
    stage_dir = tmp_path / "runs" / "x" / "cap-13312"
    data = stages.stage_config(raw, tmp_path, stage_dir, 13312, Path("outputs/a/adapter"), 0.5)
    config = parse_config(tomllib.loads(stages.to_toml(data)))
    assert config.data.path == "runs/x/cap-13312/data.jsonl" and config.data.max_length == 13312
    assert not config.data.max_length_auto and config.data.val_fraction == 0.0
    assert config.train.learning_rate == raw["train"]["learning_rate"] * 0.5
    assert config.train.init_adapter == "outputs/a/adapter"
    assert config.train.output_dir == "runs/x/cap-13312/out"
    assert config.merge.adapter_dir == "runs/x/cap-13312/out/adapter"
    assert config.lora.dropout == raw["lora"]["dropout"]  # the recipe is otherwise the same
    final_raw = stages.final_config(raw, "A", "long", "runs/x")
    final = parse_config(tomllib.loads(stages.to_toml(final_raw)))
    assert final.release.name == "emperor-4b-long" and final.merge.adapter_dir == "A"
    assert final.merge.output_dir == "weights/emperor-4b-long-bf16"
    # Never the base run's outputs, nor a stage's adapter, even if someone trained with it.
    assert final.train.output_dir == "runs/x/final-no-train"
    assert final.train.output_dir != raw["train"]["output_dir"] and not final.train.init_adapter


def test_state_keeps_the_last_good_adapter_and_summarises(tmp_path):
    state = stages.State(12288, "base")
    state.stages = [
        stages.Stage(13312, 4, 4, "done", 1, 13.1, 0.7, "a1"),
        stages.Stage(14336, 0, 0, "skipped"),
        stages.Stage(15360, 3, 3, "failed", 2, note="out of memory"),
    ]
    state.trained = ["s1", "s2", "s3", "s4"]
    state.save(tmp_path / "s.json")
    loaded = stages.State.load(tmp_path / "s.json", 0, "")
    assert loaded.last_adapter() == "a1" and loaded.trained == state.trained
    text = stages.summary(loaded)
    assert "Long samples recovered: 4" in text and "| 15360 | 3 | 3 | failed | 2 |" in text
    assert stages.State(12288, "base").last_adapter() == "base"


def test_plan_only_writes_each_stage_and_the_final_config(tmp_path, monkeypatch):
    sample = json.loads(
        (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8") and '{"id": "x"}'
    )
    rows = [dict(sample, id=f"s{i}", messages=[]) for i in range(8)]
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    config = tmp_path / "c.toml"
    config.write_text(
        '[data]\npath = "data.jsonl"\nmax_length_auto = true\nmax_length_cap = 12288\n'
        'val_fraction = 0.0\n[release]\nname = "m"\n',
        encoding="utf-8",
    )
    sizes = dict(
        zip([f"s{i}" for i in range(8)], [100, 200, 300, 400, 13000, 13100, 15000, 0], strict=True)
    )
    monkeypatch.setattr(stages, "measure_lengths", lambda config, train: sizes)
    code = train_main(
        [
            "stages", "--config", str(config), "--root", str(tmp_path),
            "--base-adapter", "outputs/m/adapter", "--run-dir", "runs/m-long", "--plan-only",
        ]
    )  # fmt: skip
    assert code == 0
    run = tmp_path / "runs" / "m-long"
    state = json.loads((run / "stages.json").read_text(encoding="utf-8"))
    assert [(s["cap"], s["long"], s["replay"], s["status"]) for s in state["stages"]] == [
        (13312, 2, 2, "pending"),
        (14336, 0, 0, "skipped"),
        (15360, 1, 1, "pending"),
    ]
    stage = tomllib.loads((run / "cap-13312" / "stage.toml").read_text(encoding="utf-8"))
    assert stage["train"]["init_adapter"] == "outputs/m/adapter"
    assert len((run / "cap-13312" / "data.jsonl").read_text(encoding="utf-8").splitlines()) == 4
    final = tomllib.loads((run / "final.toml").read_text(encoding="utf-8"))
    assert final["release"]["name"] == "m-long"


def test_init_adapter_is_a_config_key():
    assert parse_config({"train": {"init_adapter": "a"}}).train.init_adapter == "a"
    assert parse_config({}).train.init_adapter == ""


def load_runner():
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_all", ROOT / "scripts" / "train_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_final_step_runs_merge_export_and_eval_only(tmp_path):
    runner = load_runner()
    command, final = runner.final_step(tmp_path, "emperor.pipeline.cli", "runs/m-long", "data")
    phases = [command[i + 1] for i, arg in enumerate(command) if arg == "--only"]
    assert phases == ["merge", "export", "eval"] and "train" not in phases
    assert command[command.index("--train-config") + 1] == str(final)
    assert command[command.index("--resume") + 1] == "runs/m-long-final"
    base = runner.pipeline(tmp_path, "emperor.pipeline.cli", "c.toml", "runs/m-12k", "data")
    assert [base[i + 1] for i, arg in enumerate(base) if arg == "--only"][0] == "train"


def test_the_child_trainer_gets_the_same_root(tmp_path):
    command = stages.fit_command(Path("c.toml"), Path("PAUSE"), tmp_path)
    assert command[command.index("--root") + 1] == str(tmp_path)
    assert command[command.index("--config") + 1] == "c.toml"


def stage_run(tmp_path, monkeypatch, outcomes):
    """Run `train stages` on a tiny plan with `run_fit` replaced by scripted outcomes:
    (exit code, log text); a successful outcome writes the adapter's train_result."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = [{"id": f"s{i}", "messages": []} for i in range(4)]
    (tmp_path / "data.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    config = tmp_path / "c.toml"
    config.write_text(
        '[data]\npath = "data.jsonl"\nmax_length_auto = true\nmax_length_cap = 12288\n'
        'val_fraction = 0.0\n[release]\nname = "m"\n',
        encoding="utf-8",
    )
    sizes = {"s0": 100, "s1": 200, "s2": 13000, "s3": 0}
    monkeypatch.setattr(stages, "measure_lengths", lambda config, train: sizes)
    calls = []

    def fake_run_fit(config_path, log_path, gate, pause_file, poll, root):
        code, text = outcomes.pop(0)
        calls.append(log_path.name)
        log_path.write_text(text, encoding="utf-8")
        if code == 0:
            out = config_path.parent / "out"
            out.mkdir(exist_ok=True)
            (out / "train_result.json").write_text(
                json.dumps({"metrics": {"train_loss": 0.5}, "max_memory_gib": 13.1}),
                encoding="utf-8",
            )
        return code

    monkeypatch.setattr(stages, "run_fit", fake_run_fit)
    train_main(
        [
            "stages", "--config", str(config), "--root", str(tmp_path),
            "--base-adapter", "outputs/m/adapter", "--run-dir", "runs/m-long", "--asleep",
        ]
    )  # fmt: skip
    state = json.loads((tmp_path / "runs" / "m-long" / "stages.json").read_text("utf-8"))
    return state["stages"][0], calls


OOM = "Traceback (most recent call last):\ntorch.OutOfMemoryError: CUDA out of memory."


def test_an_interrupted_attempt_runs_again_without_counting(tmp_path, monkeypatch):
    stage, calls = stage_run(tmp_path, monkeypatch, [(1, "killed"), (0, "ok")])
    assert stage["status"] == "done" and stage["attempts"] == 1
    assert stage["peak_vram_gib"] == 13.1 and len(calls) == 2


def test_out_of_memory_is_retried_once_then_fails(tmp_path, monkeypatch):
    stage, _ = stage_run(tmp_path, monkeypatch, [(1, OOM), (0, "ok")])
    assert stage["status"] == "done" and stage["attempts"] == 2
    stage, _ = stage_run(tmp_path / "b", monkeypatch, [(1, OOM), (1, OOM)])
    assert stage["status"] == "failed" and stage["attempts"] == 2
    assert stage["note"] == "out of memory"


def test_another_child_failure_is_not_retried(tmp_path, monkeypatch):
    stage, _ = stage_run(
        tmp_path, monkeypatch, [(1, "Traceback (most recent call last):\nKeyError")]
    )
    assert stage["status"] == "failed" and stage["attempts"] == 1


def test_a_busy_gpu_is_not_now_not_a_failure(monkeypatch, tmp_path):
    from emperor.pipeline import processes
    from emperor.train.sft import VRAM_BUSY_EXIT_CODE

    codes = [VRAM_BUSY_EXIT_CODE, 0]

    class FakeProcess:
        def __init__(self, command, log_path, own):
            self.pid = 1
            self.code = codes.pop(0)

        def start(self):
            return self

        def poll(self):
            return self.code

    class Gate:
        def can_start(self):
            return type("S", (), {"ok": True, "reason": ""})()

        can_continue = can_start

    monkeypatch.setattr(processes, "ManagedProcess", FakeProcess)
    monkeypatch.setattr(stages.time, "sleep", lambda seconds: None)
    code = stages.run_fit(Path("c.toml"), tmp_path / "l.log", Gate(), tmp_path / "P", 0, tmp_path)
    assert code == 0 and codes == []  # waited, started again, finished


def test_clear_failure_codes_are_not_retried_as_interruptions(tmp_path, monkeypatch):
    for code in (2, 3):
        outcome = [(code, "train: error: no adapter")]
        stage, calls = stage_run(tmp_path / str(code), monkeypatch, outcome)
        assert stage["status"] == "failed" and stage["attempts"] == 1 and len(calls) == 1


def test_a_refusal_with_a_reason_exits_with_its_own_code(tmp_path, monkeypatch, capsys):
    from emperor.train import cli, sft

    def refuse(config, root, pause_file):
        raise SystemExit("no adapter to continue at x")

    monkeypatch.setattr(sft, "train", refuse)
    config = tmp_path / "c.toml"
    config.write_text("", encoding="utf-8")
    assert cli.main(["fit", "--config", str(config)]) == cli.TRAIN_ERROR_EXIT_CODE
    assert "no adapter to continue" in capsys.readouterr().err


def test_a_run_dir_outside_the_root_still_gets_a_final_config(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    (root / "data.jsonl").write_text(json.dumps({"id": "s0", "messages": []}) + "\n", "utf-8")
    config = root / "c.toml"
    config.write_text(
        '[data]\npath = "data.jsonl"\nmax_length_auto = true\nmax_length_cap = 12288\n'
        'val_fraction = 0.0\n[release]\nname = "m"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(stages, "measure_lengths", lambda config, train: {"s0": 100})
    outside = tmp_path / "elsewhere"
    train_main(
        [
            "stages", "--config", str(config), "--root", str(root),
            "--base-adapter", "a", "--run-dir", str(outside), "--asleep",
        ]
    )  # fmt: skip
    final = (outside / "final.toml").read_text(encoding="utf-8")
    assert "final-no-train" in final and outside.resolve().as_posix() in final
