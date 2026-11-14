"""Long-sample stages: train the samples the base run dropped as too long, raising the cap 1K
at a time.

`train stages --config C --base-adapter A --run-dir R` takes the base run's training config and
its finished adapter. Each stage:

- holds the train-split samples longer than the previous cap and at most this stage's cap that
  no earlier stage trained on, plus a replay of short samples (at most the base cap; about
  1:1) so the adapter does not drift towards long traces only;
- continues the previous stage's adapter at half the base learning rate;
- runs in a fresh training process (`train fit`), so GPU memory starts unfragmented, and only
  while the idle gate allows it (no input, no foreign GPU load); the trainer keeps the free-VRAM
  check, the VRAM clamp and the allocator settings;
- on an out-of-memory error is retried once in a fresh process; a second failure stops the
  stages and the last good adapter is the result.

Everything is recorded in `R/stages.json` (resumable: finished stages are skipped) and
summarised in `R/STAGES.md`. `R/final.toml` is the base config pointed at the last good
adapter, for merge, export and eval through the pipeline (`--only merge --only export
--only eval`), with its own weights and release names so the base run's are kept.
"""

import hashlib
import json
import logging
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from emperor.train.config import SftConfig
from emperor.train.data import load_combined, split_records

log = logging.getLogger("emperor.stages")
PAUSED = 75
OOM_MARKER = "OutOfMemoryError"
MAX_INTERRUPTIONS = 3


def add_arguments(p) -> None:
    p.add_argument("--base-adapter", type=Path, required=True, help="the base run's adapter")
    p.add_argument("--run-dir", type=Path, required=True, help="stage data, configs and logs")
    p.add_argument("--step", type=int, default=1024, help="tokens added to the cap per stage")
    p.add_argument("--max-cap", type=int, default=32768, help="never train above this length")
    p.add_argument("--replay", type=float, default=1.0, help="short samples per long sample")
    p.add_argument("--lr-scale", type=float, default=0.5, help="of the base learning rate")
    p.add_argument("--suffix", default="long", help="added to the final weights and names")
    p.add_argument("--asleep", action="store_true", help="the user is asleep: ignore the gate")
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--plan-only", action="store_true", help="write the plan, train nothing")


# -- planning (no torch, no GPU) -------------------------------------------------------------


def stage_caps(longest: int, base_cap: int, step: int, max_cap: int) -> list[int]:
    """The caps above the base cap, `step` apart, up to the first that holds `longest`."""
    caps = []
    cap = base_cap
    while cap < min(longest, max_cap):
        cap = min(cap + step, max_cap)
        caps.append(cap)
    return caps


def _order(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def plan_stage(
    train: list[dict],
    lengths: dict[str, int],
    low: int,
    cap: int,
    done: set[str],
    base_cap: int,
    replay: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """(long samples for this stage, replay samples): long ones are above `low` and at most
    `cap`, not yet trained; replay ones are at most `base_cap`, chosen deterministically."""
    long = [r for r in train if low < lengths.get(r["id"], 0) <= cap and r["id"] not in done]
    short = [r for r in train if 0 < lengths.get(r["id"], 0) <= base_cap]
    short.sort(key=lambda r: _order(r["id"], seed + cap))
    return long, short[: round(len(long) * replay)]


def to_toml(data: dict) -> str:
    def value(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int | float):
            return repr(v)
        if isinstance(v, list):
            return "[" + ", ".join(value(x) for x in v) + "]"
        return json.dumps(v)

    lines = []
    for section, table in data.items():
        lines.append(f"[{section}]")
        lines += [f"{k} = {value(v)}" for k, v in table.items()]
        lines.append("")
    return "\n".join(lines)


def stage_config(
    raw: dict, root: Path, stage_dir: Path, cap: int, start_adapter: Path, lr_scale: float
) -> dict:
    """The base config, trained on this stage's data at `cap`, continuing `start_adapter`."""
    data = json.loads(json.dumps(raw))  # deep copy
    rel = stage_dir.resolve().relative_to(root.resolve()).as_posix()
    data.setdefault("data", {}).update(
        path=f"{rel}/data.jsonl",
        extra_paths=[],
        max_length_auto=False,
        max_length=cap,
        val_fraction=0.0,
    )
    train = data.setdefault("train", {})
    base_lr = train.get("learning_rate", 1.0e-4)
    train.update(
        output_dir=f"{rel}/out",
        learning_rate=base_lr * lr_scale,
        init_adapter=Path(start_adapter).as_posix(),
        save_steps=min(int(train.get("save_steps", 25)), 10),
    )
    data.setdefault("merge", {})["adapter_dir"] = f"{rel}/out/adapter"
    return data


def final_config(raw: dict, adapter: str, suffix: str, run_dir: str = "") -> dict:
    """The base config with merge, export and release names for the stages' final adapter.
    It is for merge, export and eval only; its training output points into the stage run
    (`<run_dir>/final-no-train`), so even a mistaken `train fit` with it could never reach the
    base run's outputs or a stage's adapter."""
    data = json.loads(json.dumps(raw))
    release = data.setdefault("release", {})
    name = f"{release.get('name', 'emperor-4b')}-{suffix}"
    release["name"] = name
    data.setdefault("merge", {}).update(adapter_dir=adapter, output_dir=f"weights/{name}-bf16")
    train = data.setdefault("train", {})
    train["output_dir"] = f"{run_dir or 'runs/' + name}/final-no-train"
    train.pop("init_adapter", None)
    return data


# -- state --------------------------------------------------------------------------------------


@dataclass
class Stage:
    cap: int
    long: int = 0
    replay: int = 0
    status: str = "pending"  # pending, done, skipped (nothing to train), failed
    attempts: int = 0
    peak_vram_gib: float | None = None
    train_loss: float | None = None
    adapter: str = ""
    note: str = ""


@dataclass
class State:
    base_cap: int
    base_adapter: str
    stages: list[Stage] = field(default_factory=list)
    trained: list[str] = field(default_factory=list)  # long sample ids trained so far

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path, base_cap: int, base_adapter: str) -> "State":
        if not path.is_file():
            return cls(base_cap, base_adapter)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            raw["base_cap"],
            raw["base_adapter"],
            [Stage(**s) for s in raw.get("stages", [])],
            raw.get("trained", []),
        )

    def last_adapter(self) -> str:
        done = [s for s in self.stages if s.status == "done" and s.adapter]
        return done[-1].adapter if done else self.base_adapter


def summary(state: State) -> str:
    rows = [
        "# Long-sample stages",
        "",
        f"Base cap {state.base_cap}; base adapter `{state.base_adapter}`.",
        f"Long samples recovered: {len(state.trained)}. Final adapter: `{state.last_adapter()}`.",
        "",
        "| Cap | Long | Replay | Status | Attempts | Peak VRAM (GiB) | Train loss | Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in state.stages:
        rows.append(
            f"| {s.cap} | {s.long} | {s.replay} | {s.status} | {s.attempts} | "
            f"{'' if s.peak_vram_gib is None else s.peak_vram_gib} | "
            f"{'' if s.train_loss is None else round(s.train_loss, 4)} | {s.note} |"
        )
    return "\n".join(rows) + "\n"


# -- running -----------------------------------------------------------------------------------


def measure_lengths(config: SftConfig, records: list[dict]) -> dict[str, int]:
    """Each sample's token length under the base's chat template (0: not trainable)."""
    from transformers import AutoTokenizer

    from emperor.train.data import encode_sample

    tokenizer = AutoTokenizer.from_pretrained(config.model.base)
    lengths = {}
    for record in records:
        row, _ = encode_sample(record, tokenizer, 1 << 20)
        lengths[record["id"]] = len(row["input_ids"]) if row else 0
    return lengths


def fit_command(config_path: Path, pause_file: Path, root: Path) -> list[str]:
    """The child `train fit`, with the same root, so its relative paths resolve alike."""
    return [
        sys.executable, "-m", "emperor.train.cli", "fit",
        "--config", str(config_path), "--root", str(root), "--pause-file", str(pause_file),
    ]  # fmt: skip


def run_fit(
    config_path: Path, log_path: Path, gate, pause_file: Path, poll: float, root: Path
) -> int:
    """`train fit` in a fresh process, paused (checkpoint, exit 75) when the gate says so and
    resumed when it allows again. A refusal because another process holds the VRAM is "not
    now": it waits and starts again. Returns the final exit code."""
    from emperor.pipeline.processes import ManagedProcess, OwnProcesses
    from emperor.train.sft import VRAM_BUSY_EXIT_CODE

    own = OwnProcesses()
    gate.own_pids = own.pids
    while True:
        while not (status := gate.can_start()).ok:
            log.info("waiting for an idle machine: %s", status.reason)
            time.sleep(poll)
        pause_file.unlink(missing_ok=True)
        command = fit_command(config_path, pause_file, root)
        trainer = ManagedProcess(command, log_path, own).start()
        log.info("stage trainer pid %s", trainer.pid)
        while trainer.poll() is None:
            time.sleep(poll)
            if not pause_file.exists() and not (status := gate.can_continue()).ok:
                log.info("pausing: %s", status.reason)
                pause_file.touch()
        code = trainer.poll()
        if code == VRAM_BUSY_EXIT_CODE:
            log.info("another process holds the VRAM; waiting before starting again")
            time.sleep(poll)
            continue
        if code != PAUSED:
            return code


def main(args, config: SftConfig) -> int:
    from emperor.pipeline.idle import IdleGate

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = args.root.resolve()
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    raw = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    base_cap = config.data.max_length_cap if config.data.max_length_auto else config.data.max_length
    state_path = run_dir / "stages.json"
    state = State.load(state_path, base_cap, Path(args.base_adapter).as_posix())

    paths = [
        p if Path(p).is_absolute() else root / p
        for p in (config.data.path, *config.data.extra_paths)
    ]
    records = load_combined(paths)
    train, _ = split_records(records, config.data.val_fraction, config.data.split_seed)
    lengths = measure_lengths(config, train)
    longest = max(lengths.values(), default=0)
    caps = stage_caps(longest, base_cap, args.step, args.max_cap)
    log.info(
        "%d train samples over the base cap %d (longest %d); stage caps %s",
        sum(1 for n in lengths.values() if n > base_cap),
        base_cap,
        longest,
        caps,
    )
    gate = IdleGate(asleep=args.asleep)
    previous = base_cap
    for cap in caps:
        stage = next((s for s in state.stages if s.cap == cap), None)
        if stage is None:
            stage = Stage(cap)
            state.stages.append(stage)
        if stage.status in ("done", "skipped"):
            previous = cap
            continue
        if stage.status == "failed":
            log.info("stage %d failed before; caps are not raised further", cap)
            break
        long, replay = plan_stage(
            train, lengths, previous, cap, set(state.trained), base_cap, args.replay,
            config.data.split_seed,
        )  # fmt: skip
        stage.long, stage.replay = len(long), len(replay)
        if not long:
            stage.status, stage.note = "skipped", "no samples in this length band"
            state.save(state_path)
            previous = cap
            continue
        stage_dir = run_dir / f"cap-{cap}"
        stage_dir.mkdir(exist_ok=True)
        with (stage_dir / "data.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for record in long + replay:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        start = state.last_adapter()
        config_path = stage_dir / "stage.toml"
        config_path.write_text(
            to_toml(stage_config(raw, root, stage_dir, cap, Path(start), args.lr_scale)),
            encoding="utf-8",
        )
        state.save(state_path)
        if args.plan_only:
            previous = cap
            continue
        # An attempt counts only once its result is known: success, an out-of-memory error or
        # another failure of the child (a traceback). An attempt cut short by a kill or a reboot
        # (no traceback) runs again without counting, up to MAX_INTERRUPTIONS times in a row.
        interruptions = 0
        while stage.attempts < 2:
            log_path = stage_dir / f"train-{stage.attempts + 1}-{interruptions}.log"
            code = run_fit(
                config_path, log_path, gate, stage_dir / "PAUSE", args.poll_seconds, root
            )
            text = (
                log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            )
            if code != 0 and "Traceback" not in text and interruptions < MAX_INTERRUPTIONS:
                interruptions += 1
                log.warning("stage %d was interrupted (exit code %s); starting it again", cap, code)
                continue
            interruptions = 0
            stage.attempts += 1
            state.save(state_path)
            if code == 0:
                result = json.loads((stage_dir / "out" / "train_result.json").read_text("utf-8"))
                stage.status = "done"
                stage.peak_vram_gib = result.get("max_memory_gib")
                stage.train_loss = (result.get("metrics") or {}).get("train_loss")
                stage.adapter = (stage_dir / "out" / "adapter").resolve().as_posix()
                state.trained += [r["id"] for r in long]
                break
            stage.note = "out of memory" if OOM_MARKER in text else f"exit code {code}"
            log.warning("stage %d attempt %d failed: %s", cap, stage.attempts, stage.note)
            if OOM_MARKER not in text:
                break  # not a memory problem: a retry would fail the same way
        if stage.status != "done":
            stage.status = "failed"
        state.save(state_path)
        (run_dir / "STAGES.md").write_text(summary(state), encoding="utf-8")
        if stage.status == "failed":
            log.warning("stopping: the last good adapter is %s", state.last_adapter())
            break
        previous = cap
    state.save(state_path)
    (run_dir / "STAGES.md").write_text(summary(state), encoding="utf-8")
    rel_run = run_dir.resolve().relative_to(root).as_posix()
    final = final_config(raw, state.last_adapter(), args.suffix, rel_run)
    (run_dir / "final.toml").write_text(to_toml(final), encoding="utf-8")
    print(
        json.dumps(
            {
                "recovered": len(state.trained),
                "final_adapter": state.last_adapter(),
                "final_config": str(run_dir / "final.toml"),
                "stages": [asdict(s) for s in state.stages],
            }
        )
    )
    return 0
