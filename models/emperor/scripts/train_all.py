"""Train the four models one after another: each at 12K, then its long-sample stages, then
merge, export (bf16 + GGUF) and eval of the final adapter beside the 12K-only one.

    python scripts/train_all.py --repo "F:/Git Repositories/Imperium-Models"

Every step is a separate command that is itself gated on an idle machine (the pipeline's and
the stages' idle gates), so this script only sequences them and records what finished in
`models/emperor/runs/train-all/state.json`; rerunning it skips finished steps. One model is on
the GPU at a time because the steps run strictly in order. Nothing is published.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

MODELS = [
    # (name, model folder, config, pipeline module, eval data dir)
    ("emperor-4b", "emperor", "configs/emperor-4b.toml", "emperor.pipeline.cli", "data"),
    ("emperor-lite-2b", "emperor", "configs/emperor-lite-2b.toml", "emperor.pipeline.cli", "data"),
    ("consul-4b", "consul", "configs/consul-4b.toml", "consul.pipeline.cli", "data/hosted-1"),
    (
        "consul-lite-2b",
        "consul",
        "configs/consul-lite-2b.toml",
        "consul.pipeline.cli",
        "data/hosted-1",
    ),
]
EVAL_TIER = "32k"  # the zero-shot baselines ran at 32K


def python_of(folder: Path) -> str:
    exe = folder / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(exe)


BASE_PHASES = ("train", "merge", "export", "eval")
FINAL_PHASES = ("merge", "export", "eval")  # the final adapter is never trained again


def pipeline(
    folder: Path, module: str, config: str, run: str, data_dir: str, phases=BASE_PHASES
) -> list[str]:
    only = [arg for phase in phases for arg in ("--only", phase)]
    return [
        python_of(folder), "-m", module, "--root", str(folder),
        "--train-config", config, "--resume", run, "--data-dir", data_dir,
        "--eval-tier", EVAL_TIER, *only,
    ]  # fmt: skip


def final_step(folder: Path, module: str, stage_run: str, data_dir: str):
    final = folder / stage_run / "final.toml"
    return (
        pipeline(folder, module, str(final), f"{stage_run}-final", data_dir, FINAL_PHASES),
        final,
    )


def steps(
    repo: Path, name: str, folder_name: str, config: str, module: str, data_dir: str, env=None
):
    folder = repo / "models" / folder_name
    base_run = f"runs/{name}-12k"
    stage_run = f"runs/{name}-long"
    yield (
        f"{name}: 12K train, merge, export, eval",
        pipeline(folder, module, config, base_run, data_dir),
        folder,
    )
    adapter = json.loads(
        subprocess.run(
            [python_of(folder), "-m", "emperor.train.cli", "show", "--config", config],
            cwd=folder, capture_output=True, text=True, check=True, env=env,
        ).stdout
    )["merge"]["adapter_dir"]  # fmt: skip
    yield (
        f"{name}: long-sample stages",
        [
            python_of(folder), "-m", "emperor.train.cli", "stages", "--config", config,
            "--root", str(folder), "--base-adapter", adapter, "--run-dir", stage_run,
        ],
        folder,
    )  # fmt: skip
    command, _ = final_step(folder, module, stage_run, data_dir)
    yield (f"{name}: final adapter merge, export, eval", command, folder)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[], help="model names to run")
    parser.add_argument(
        "--emperor-src",
        type=Path,
        help="run Emperor's package from this src directory (PYTHONPATH), e.g. a reviewed branch",
    )
    parser.add_argument(
        "--done",
        action="append",
        default=[],
        help="a step label to record as done without running it (e.g. a 12K run made by hand)",
    )
    args = parser.parse_args()
    env = dict(os.environ)
    if args.emperor_src:
        env["PYTHONPATH"] = str(args.emperor_src)
    state_dir = args.repo / "models" / "emperor" / "runs" / "train-all"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    for label in args.done:
        state.setdefault(label, {"status": "done", "note": "recorded by hand (--done)"})
    for name, folder_name, config, module, data_dir in MODELS:
        if args.only and name not in args.only:
            continue
        for label, command, cwd in steps(
            args.repo, name, folder_name, config, module, data_dir, env
        ):
            if state.get(label, {}).get("status") == "done":
                continue
            if (
                "final adapter" in label
                and not Path(command[command.index("--train-config") + 1]).exists()
            ):
                state[label] = {"status": "skipped", "reason": "no stages config"}
                continue
            if "final adapter" in label:
                stages_state = json.loads(
                    (
                        Path(command[command.index("--train-config") + 1]).parent / "stages.json"
                    ).read_text(encoding="utf-8")
                )
                if not any(s["status"] == "done" for s in stages_state["stages"]):
                    state[label] = {"status": "skipped", "reason": "no stage trained"}
                    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                    continue
            started = datetime.now().isoformat(timespec="seconds")
            print(f"{started} {label}", flush=True)
            code = subprocess.run(command, cwd=cwd, env=env).returncode
            state[label] = {
                "status": "done" if code == 0 else "failed",
                "code": code,
                "started": started,
                "finished": datetime.now().isoformat(timespec="seconds"),
            }
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            if code != 0:
                print(f"{label}: exit code {code}; stopping", flush=True)
                return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
