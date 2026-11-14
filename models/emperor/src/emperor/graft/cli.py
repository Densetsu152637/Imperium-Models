"""`uv run --group graft graft {sources,fetch,mix}`: open-data grafts for Emperor and Consul.

    graft sources                        the registry: licence, generators, usable or why not
    graft fetch --out data/graft         sample, convert and filter every usable source
    graft mix --hosted data/hosted-1/final/final.jsonl --graft data/graft --out data/mix
                                         the training mix and its report
    graft generate research --input papers.jsonl --out data/research --confirm-spend
                                         DeepSeek-written tasks (research, paraphrase, ask-dpo,
                                         coref); refuses without --confirm-spend

Everything is CPU and network only. Fetch writes `<out>/<source>.jsonl` (converted samples) and
`<out>/report.json` (counts and drop reasons per source); downloads are cached in `<out>/cache`.
"""

import argparse
import json
import sys
from pathlib import Path

from emperor.graft import fetch
from emperor.graft.convert import convert
from emperor.graft.generators import KINDS
from emperor.graft.generators import run as run_generator
from emperor.graft.mix import mix, read_jsonl
from emperor.graft.sources import BUCKETS, SOURCES, Source

# Per source: samples to keep after conversion and filtering, before mixing (the mixer takes
# what the target shares ask for; a margin lets it balance subsets).
DEFAULT_KEEP = 1500
MAX_THINKING_TOKENS = 4096  # the OpenThoughts thinking cap (longer samples are dropped)
AGENT_MAX_THINKING_TOKENS = 8192


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
    m = sub.add_parser("mix", help="mix the hosted data with the grafts at the target shares")
    m.add_argument("--hosted", type=Path, action="append", required=True)
    m.add_argument("--graft", type=Path, default=Path("data/graft"))
    m.add_argument("--research", type=Path, action="append", default=[], help="if generated")
    m.add_argument("--out", type=Path, default=Path("data/mix"))
    m.add_argument("--seed", type=int, default=1337)
    g = sub.add_parser("generate", help="DeepSeek-generated tasks (spends money)")
    g.add_argument("kind", choices=sorted(KINDS))
    g.add_argument("--input", type=Path, required=True, help="JSONL rows for the generator")
    g.add_argument("--out", type=Path, required=True, help="directory for records.jsonl")
    g.add_argument("--hosted-config", type=Path, default=Path("configs/hosted.toml"))
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
    args.out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(args.out / "spend.jsonl", {provider.name: provider.budget_usd})
    client = make_clients(config, ledger, ("teacher",))["teacher"]
    records_path, errors_path = args.out / "records.jsonl", args.out / "errors.jsonl"
    done = {r.get("input_id") for r in read_jsonl([records_path])}
    rows = read_jsonl([args.input])[: args.limit or None]
    made = failed = 0
    with (
        records_path.open("a", encoding="utf-8", newline="\n") as out,
        errors_path.open("a", encoding="utf-8", newline="\n") as err,
    ):
        for record, error in run_generator(
            args.kind, rows, lambda m: client.chat(m, max_tokens=16384), role.model, done
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
        "files": list(source.files),
        "licence": source.licence,
        "generators": list(source.card_generators) or f"per sample ({source.generator_field})",
        "usable": source.usable,
        "excluded_reason": source.excluded_reason or None,
        "note": source.note or None,
    }


def records_of(source: Source, args, cache: Path):
    for file in source.files:
        url = fetch.file_url(source.dataset, file)
        if file.endswith(".parquet"):
            path = fetch.download(url, cache / source.dataset.replace("/", "__") / file)
            yield from fetch.parquet_rows(path)
        else:
            yield from fetch.sample_jsonl(url, args.slices, args.slice_bytes, args.seed)


def fetch_source(source: Source, args, out: Path) -> dict:
    cap = args.max_thinking if source.kind == "openthoughts" else args.agent_max_thinking
    kept: list[dict] = []
    seen: set[str] = set()
    dropped: dict[str, int] = {}
    generators: dict[str, int] = {}
    read = 0
    for record in records_of(source, args, out / "cache"):
        read += 1
        sample, reason = convert(record, source, cap)
        if sample is None:
            dropped[reason] = dropped.get(reason, 0) + 1
            generator = source.generator_of(record) or "unknown"
            if reason == "blocked_generator":
                generators[f"dropped: {generator}"] = generators.get(f"dropped: {generator}", 0) + 1
            continue
        if sample["id"] in seen:
            dropped["duplicate"] = dropped.get("duplicate", 0) + 1
            continue
        seen.add(sample["id"])
        kept.append(sample)
        generators[sample["meta"]["generator"]] = generators.get(sample["meta"]["generator"], 0) + 1
        if len(kept) >= args.keep:
            break
    path = out / f"{source.key}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in kept:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return {
        "read": read,
        "kept": len(kept),
        "dropped": dropped,
        "generators": generators,
        "licence": source.licence,
        "thinking_cap_tokens": cap,
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
        report_path = args.out / "report.json"
        report = (
            json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        )
        for key in args.source or sorted(SOURCES):
            source = SOURCES[key]
            if not source.usable:
                report[key] = {"excluded": source.excluded_reason, "licence": source.licence}
                continue
            print(f"graft: {key} ...", file=sys.stderr, flush=True)
            report[key] = fetch_source(source, args, args.out)
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
    samples, report = mix(buckets, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "final.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v["taken"] for k, v in report["buckets"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
