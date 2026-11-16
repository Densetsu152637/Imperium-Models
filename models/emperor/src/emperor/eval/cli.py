"""Command line for the evaluation harness (plan stage 2b).

`--dry-run` replays every task with a scripted fake model that plays the task's reference
turns, so it needs no model server and still emits every metric. Without it, the model behind an
OpenAI-compatible endpoint (e.g. a llama.cpp server) is the root agent, and subagents are
answered from the task's recordings: deterministic and free.

- `harness` (default): each task runs through Imperium's own agent loop, root prompt and tool
  schemas with `imperium run --headless --json`. Scores that are compared or published come
  from this backend. `--variant` and `--flag` pick a harness variant for A/B runs.
- `replay`: this package's own loop with the vendored prompt and tool schemas; no Imperium
  needed. Kept for tests and for comparing against older reports.

Only `--suite e2e` spends money, and it refuses to run unless an API ceiling is set.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from emperor.eval.harness import BackendUnavailable, HarnessBackend, HarnessConfig
from emperor.eval.metrics import METRICS as METRIC_DEFS
from emperor.eval.models import DEFAULT_MAX_TOKENS, OpenAICompatModel, ScriptedModel
from emperor.eval.report import CeilingExceeded, CostMeter, run_suite
from emperor.eval.tasks import TaskError, load_suite, load_tasks

# Metric name -> plan goal it verifies.
METRICS = {name: metric.goal for name, metric in METRIC_DEFS.items()}
# Suite name -> plan goals it verifies.
SUITES = {
    "heldout": "G1, G2, G4, G5, G10, G11 (plus a G3 replay proxy and a G6 smoke test)",
    "e2e": "G3",
    "longctx": "G6",
}
# Context tier name -> active context in tokens; the same weights run at each tier.
CONTEXT_TIERS = {
    "32k": 32_768,
    "64k": 65_536,
    "96k": 98_304,
    "128k": 131_072,
    "192k": 196_608,
    "256k": 262_144,
}
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"


def default_imperium() -> str:
    """`IMPERIUM_BIN`, else the Windows install's `imperium.exe` when present, else `imperium`."""
    if os.environ.get("IMPERIUM_BIN"):
        return os.environ["IMPERIUM_BIN"]
    installed = Path(os.environ.get("LOCALAPPDATA", "")) / "Imperium" / "imperium.exe"
    return str(installed) if os.environ.get("LOCALAPPDATA") and installed.exists() else "imperium"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval",
        description="Score a model as the root agent of Imperium sessions.",
    )
    parser.add_argument("--model", help="model spec as Imperium names it, e.g. llamacpp:emperor")
    parser.add_argument("--suite", choices=sorted(SUITES), default="heldout")
    parser.add_argument(
        "--context-tier",
        choices=list(CONTEXT_TIERS),
        default="64k",
        help="active context the model runs with (default: 64k)",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRICS),
        action="append",
        help="limit scoring to these metrics (repeatable); default: all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="score the tasks' reference turns with a scripted fake model; no server needed",
    )
    parser.add_argument(
        "--backend",
        choices=["harness", "replay"],
        default="harness",
        help="harness: through Imperium's agent loop with `imperium run --headless` (default); "
        "replay: this package's own loop with the vendored prompt",
    )
    parser.add_argument(
        "--imperium",
        default=default_imperium(),
        help="the imperium binary for the harness backend (default: IMPERIUM_BIN, the installed "
        "imperium.exe, or `imperium` on PATH)",
    )
    parser.add_argument("--variant", help="harness variant: a name or a TOML path")
    parser.add_argument("--flag", action="append", default=[], help="harness flag k=v (repeatable)")
    parser.add_argument(
        "--thinking",
        choices=["on", "off"],
        help="the harness's thinking passthrough (the 2B models need on)",
    )
    parser.add_argument(
        "--remote-model",
        action="store_true",
        help="the root model is not on this machine (the harness's `--local` is left off)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("EMPEROR_EVAL_BASE_URL", DEFAULT_BASE_URL),
        help=f"OpenAI-compatible endpoint for the replay backend (default: {DEFAULT_BASE_URL}, "
        "or EMPEROR_EVAL_BASE_URL)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"tokens per reply, thinking included (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--tasks", type=Path, help="task directory or file to use instead of the bundled suite"
    )
    parser.add_argument(
        "--api-ceiling",
        type=float,
        default=_env_float("EMPEROR_EVAL_API_CEILING", 0.0),
        help="most USD a run may spend on hosted models (default 0, or EMPEROR_EVAL_API_CEILING); "
        "--suite e2e refuses to run while it is 0",
    )
    parser.add_argument("--output", type=Path, help="write the report here instead of stdout")
    return parser


def _fail(message: str) -> int:
    print(f"eval: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = args.metric or list(METRICS)
    context_tokens = CONTEXT_TIERS[args.context_tier]

    spends_money = args.suite == "e2e" and not args.dry_run
    if spends_money and args.api_ceiling <= 0:
        return _fail(
            "--suite e2e uses real hosted subagents and spends money; set --api-ceiling "
            "(or EMPEROR_EVAL_API_CEILING) to the most it may spend"
        )
    if spends_money:
        return _fail("--suite e2e needs real subagents, which the eval does not run yet")

    try:
        tasks = load_tasks(args.tasks, args.suite) if args.tasks else load_suite(args.suite)
    except (TaskError, OSError) as error:
        return _fail(f"cannot load tasks: {error}")

    harness = None
    backend = "replay" if args.dry_run else args.backend

    def model_for(task):
        return ScriptedModel(task.reference)

    if args.dry_run:
        pass
    elif args.backend == "harness":
        if not args.model:
            return _fail("--model is required without --dry-run")
        harness = HarnessBackend(
            HarnessConfig(
                binary=[args.imperium],
                base_url=args.base_url,
                model_id=args.model.split(":", 1)[-1],
                context_tokens=context_tokens,
                variant=args.variant,
                flags=args.flag,
                thinking=args.thinking,
                local=not args.remote_model,
            )
        )
    else:
        if not args.model:
            return _fail("--model is required without --dry-run")
        client = OpenAICompatModel(
            args.base_url, args.model.split(":", 1)[-1], max_tokens=args.max_tokens
        )

        def model_for(task):
            return client

    try:
        report = run_suite(
            tasks,
            model_for,
            model=args.model,
            suite=args.suite,
            tier=args.context_tier,
            context_tokens=context_tokens,
            metrics=metrics,
            backend=backend,
            dry_run=args.dry_run,
            goals=SUITES[args.suite],
            meter=CostMeter(args.api_ceiling),
            harness=harness,
        )
    except (CeilingExceeded, BackendUnavailable) as error:
        return _fail(str(error))

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 1 if report["tasks_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
