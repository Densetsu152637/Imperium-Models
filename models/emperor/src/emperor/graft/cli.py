"""`uv run --group graft graft {sources,fetch,mix,generate}`: open-data grafts.

    graft sources                        the registry: licence, generators, usable or why not
    graft fetch --out data/graft --config configs/emperor-4b.toml
                                         sample, convert and filter every usable source; with
                                         --config, keep only samples that fit its training
                                         length under its base's chat template
    graft mix --model emperor --hosted data/hosted-1/final/final.jsonl --graft data/graft \\
              --out data/mix             the training mix for one model and its report
    graft generate research --input papers.jsonl --out data/research --confirm-spend
                                         DeepSeek-written records; refuses without
                                         --confirm-spend

Fetch and mix are CPU and network only. Fetch writes `<out>/<source>.jsonl` (converted samples)
and `<out>/report.json` (counts and drop reasons per source, including the generators dropped);
downloads are cached in `<out>/cache`.
"""

import argparse
import json
import sys
from pathlib import Path

from emperor.data.prompts import HeldoutIndex
from emperor.graft import fetch
from emperor.graft.convert import BLOCKED, convert
from emperor.graft.generators import KINDS, PAPER_KINDS, load_heldout_papers
from emperor.graft.generators import run as run_generator
from emperor.graft.mix import mix, read_jsonl
from emperor.graft.sources import BUCKETS, SOURCES, Source

# Per source: samples to keep after conversion and filtering, before mixing (the mixer takes
# what the target shares ask for; a margin lets it balance subsets).
DEFAULT_KEEP = 1500
MAX_THINKING_TOKENS = 8192  # OpenThoughts: at 4096 only ~1% of code traces fit
AGENT_MAX_THINKING_TOKENS = 8192
# The hosted runner's ledger, so generators share its budget caps instead of keeping their own.
DEFAULT_LEDGER = Path("data/hosted-1/spend.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graft", description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sources", help="list the sources, their licences and generators")
    f = sub.add_parser("fetch", help="sample, convert and filter the usable sources")
    f.add_argument("--out", type=Path, default=Path("data/graft"))
    f.add_argument("--source", action="append", choices=sorted(SOURCES), help="only these")
    f.add_argument("--keep", type=int, default=DEFAULT_KEEP, help="samples per source")
    f.add_argument("--seed", type=int, default=1337)
    f.add_argument("--slices", type=int, default=400, help="JSONL range reads per file")
    f.add_argument("--slice-bytes", type=int, default=262144)
    f.add_argument("--max-thinking", type=int, default=MAX_THINKING_TOKENS)
    f.add_argument("--agent-max-thinking", type=int, default=AGENT_MAX_THINKING_TOKENS)
    f.add_argument(
        "--config",
        type=Path,
        required=True,
        help="training config whose base tokenizer and training length every sample must fit "
        "(needs the train group), e.g. configs/emperor-4b.toml",
    )
    m = sub.add_parser("mix", help="mix the hosted data with the grafts at the target shares")
    m.add_argument(
        "--model",
        choices=["emperor"],
        required=True,
        help="whose training mix (Consul's needs its own held-out check, not wired up yet)",
    )
    m.add_argument("--hosted", type=Path, action="append", required=True)
    m.add_argument("--graft", type=Path, default=Path("data/graft"))
    m.add_argument("--research", type=Path, action="append", default=[], help="solved records")
    m.add_argument("--out", type=Path, default=Path("data/mix"))
    m.add_argument("--seed", type=int, default=1337)
    g = sub.add_parser("generate", help="DeepSeek-generated records (spends money)")
    g.add_argument("kind", choices=sorted(KINDS))
    g.add_argument("--input", type=Path, required=True, help="JSONL rows for the generator")
    g.add_argument("--out", type=Path, required=True, help="directory for records.jsonl")
    g.add_argument("--hosted-config", type=Path, default=Path("configs/hosted.toml"))
    g.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help=f"spend ledger shared with the hosted runner (default {DEFAULT_LEDGER})",
    )
    g.add_argument(
        "--heldout-papers",
        type=Path,
        help="paper ids (one per line) kept for the eval split: never made into training data "
        "(required for extract and research)",
    )
    g.add_argument(
        "--budget",
        type=float,
        help="DeepSeek budget in USD for this run, overriding budget_usd (the shared ledger "
        "counts every earlier hosted run)",
    )
    g.add_argument("--limit", type=int, default=0, help="at most this many input rows")
    g.add_argument(
        "--confirm-spend", action="store_true", help="required: the teacher is a paid API"
    )
    return parser


def generate(args) -> int:
    if not args.confirm_spend:
        print("graft: generate calls the paid DeepSeek API; pass --confirm-spend", file=sys.stderr)
        return 2
    from emperor.data.hosted import Ledger, load_config, make_clients

    config = load_config(args.hosted_config)
    role, provider = config.role("teacher")
    if provider.name != "deepseek":
        print(f"graft: the teacher must be DeepSeek, not {provider.name}", file=sys.stderr)
        return 2
    if args.kind in PAPER_KINDS and not args.heldout_papers:
        print(f"graft: {args.kind} needs --heldout-papers (the eval split)", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    budgets = {name: p.budget_usd for name, p in config.providers.items()}
    if args.budget is not None:
        budgets[provider.name] = args.budget
    ledger = Ledger(args.ledger, budgets)
    client = make_clients(config, ledger, ("teacher",))["teacher"]
    records_path, errors_path = args.out / "records.jsonl", args.out / "errors.jsonl"
    done = {r.get("input_id") for r in read_jsonl([records_path])}
    rows = read_jsonl([args.input])[: args.limit or None]
    heldout = load_heldout_papers(args.heldout_papers)
    made = failed = 0
    with (
        records_path.open("a", encoding="utf-8", newline="\n") as out,
        errors_path.open("a", encoding="utf-8", newline="\n") as err,
    ):
        for record, error in run_generator(
            args.kind,
            rows,
            lambda m: client.chat(m, max_tokens=16384),
            role.model,
            done,
            heldout_papers=heldout,
            heldout_tasks=HeldoutIndex.bundled(),
        ):
            if record is None:
                failed += 1
                err.write(json.dumps({"error": error}) + "\n")
            else:
                made += 1
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
    client.close()
    print(json.dumps({"made": made, "failed": failed, "spend": ledger.summary()}))
    return 0


def describe(source: Source) -> dict:
    return {
        "dataset": source.dataset,
        "revision": source.revision,
        "files": list(source.files),
        "licence": source.licence,
        "generators": list(source.card_generators) or f"per sample ({source.generator_field})",
        "upstream_allowed": sorted(source.upstream_allow) or None,
        "usable": source.usable,
        "excluded_reason": source.excluded_reason or None,
        "note": source.note or None,
    }


def records_of(source: Source, args, cache: Path):
    """Records from every file of the source, taken in turn (one from each file, then the
    next), so a caller that stops early still covers every file."""
    streams = []
    for file in source.files:
        url = fetch.file_url(source.dataset, file, source.revision)
        if file.endswith(".parquet"):
            path = fetch.download(
                url, cache / source.dataset.replace("/", "__") / source.revision[:12] / file
            )
            streams.append(fetch.parquet_rows(path))
        else:
            streams.append(fetch.sample_jsonl(url, args.slices, args.slice_bytes, args.seed))
    yield from interleave(streams)


def interleave(streams):
    streams = list(streams)
    while streams:
        for stream in list(streams):
            try:
                yield next(stream)
            except StopIteration:
                streams.remove(stream)


class FitCheck:
    """Whether a sample encodes within a training config's length under its base's chat
    template (the same encoder the trainer uses); loaded only when asked for."""

    def __init__(self, config_path: Path):
        from transformers import AutoTokenizer

        from emperor.train.config import load_config
        from emperor.train.data import encode_sample

        config = load_config(config_path)
        self.path = str(config_path)
        self.length = (
            config.data.max_length_cap if config.data.max_length_auto else (config.data.max_length)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.base)
        self._encode = encode_sample

    def __call__(self, sample: dict) -> str | None:
        try:
            row, reason = self._encode(sample, self.tokenizer, self.length)
        except Exception as error:  # a template that cannot render it
            return f"fit:{type(error).__name__}"
        return None if row is not None else f"fit:{reason}"


def fetch_source(source: Source, args, out: Path, fit=None) -> dict:
    cap = args.max_thinking if source.kind == "openthoughts" else args.agent_max_thinking
    kept: list[dict] = []
    seen: set[str] = set()
    dropped: dict[str, int] = {}
    generators: dict[str, int] = {}
    dropped_generators: dict[str, int] = {}
    upstream: dict[str, int] = {}
    read = 0
    for record in records_of(source, args, out / "cache"):
        read += 1
        sample, reason = convert(record, source, cap)
        if sample is None and reason == BLOCKED:
            name = source.generator_of(record) or "unknown"
            dropped_generators[name] = dropped_generators.get(name, 0) + 1
        if sample is not None and sample["id"] in seen:
            sample, reason = None, "duplicate"
        if sample is not None and fit is not None:
            reason = fit(sample)
            sample = None if reason else sample
        if sample is None:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        seen.add(sample["id"])
        kept.append(sample)
        meta = sample["meta"]
        generators[meta["generator"]] = generators.get(meta["generator"], 0) + 1
        if meta.get("upstream_source"):
            upstream[meta["upstream_source"]] = upstream.get(meta["upstream_source"], 0) + 1
        if len(kept) >= args.keep:
            break
    path = out / f"{source.key}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in kept:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return {
        "dataset": source.dataset,
        "revision": source.revision,
        "read": read,
        "kept": len(kept),
        "dropped": dict(sorted(dropped.items())),
        "generators": generators,
        "dropped_generators": dropped_generators,
        "upstream_sources": upstream or None,
        "licence": source.licence,
        "thinking_cap_tokens": cap,
        "fit_config": fit.path if fit else None,
        "fit_length": fit.length if fit else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "sources":
        print(json.dumps({k: describe(s) for k, s in SOURCES.items()}, indent=2))
        return 0
    if args.command == "generate":
        return generate(args)
    if args.command == "fetch":
        args.out.mkdir(parents=True, exist_ok=True)
        fit = FitCheck(args.config) if args.config else None
        report_path = args.out / "report.json"
        report = (
            json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        )
        for key in args.source or sorted(SOURCES):
            source = SOURCES[key]
            if not source.usable:
                report[key] = {"excluded": source.excluded_reason, "licence": source.licence}
                (args.out / f"{key}.jsonl").unlink(missing_ok=True)  # no stale samples
                continue
            print(f"graft: {key} ...", file=sys.stderr, flush=True)
            report[key] = fetch_source(source, args, args.out, fit)
            print(f"graft: {key}: {report[key]['kept']} kept", file=sys.stderr, flush=True)
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v.get("kept", v.get("excluded")) for k, v in report.items()}))
        return 0
    buckets = {
        "hosted": read_jsonl(args.hosted),
        "research": read_jsonl(args.research),
        **{
            bucket: read_jsonl([args.graft / f"{key}.jsonl" for key in keys])
            for bucket, keys in BUCKETS.items()
        },
    }
    samples, report = mix(buckets, args.seed, model=args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "final.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v["taken"] for k, v in report["buckets"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
