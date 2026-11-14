"""`uv run --group train train {show,fit,merge} --config configs/sft.toml`."""

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

from emperor.train.config import ConfigError, load_config

DEFAULT_ALLOC_CONF = "max_split_size_mb:128,garbage_collection_threshold:0.7"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train", description="Emperor bf16 LoRA fine-tuning.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        ("show", "print the validated configuration"),
        ("fit", "train (or resume from the last checkpoint)"),
        ("merge", "merge the adapter into the base and save bf16 safetensors"),
        ("stages", "train the samples too long for the base run, raising the cap 1K a stage"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--config", type=Path, default=Path("configs/sft.toml"))
        p.add_argument("--root", type=Path, default=Path.cwd(), help="base for relative paths")
        if name == "stages":
            from emperor.train.stages import add_arguments

            add_arguments(p)
        if name == "fit":
            p.add_argument(
                "--pause-file",
                type=Path,
                help="when this file appears, save a checkpoint and exit with code 75",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as error:
        print(f"train: {error}", file=sys.stderr)
        return 2
    if args.command == "show":
        print(json.dumps(dataclasses.asdict(config), indent=2))
        return 0
    # Long samples fragment PyTorch's CUDA cache: at 16K tokens 1.2 GiB sat reserved but unused
    # and the next allocation ran out of memory. Smaller split blocks and earlier collection
    # keep it usable (expandable segments would too, but are not supported on Windows). A value
    # the user sets in either variable wins.
    if not (os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or os.environ.get("PYTORCH_ALLOC_CONF")):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = DEFAULT_ALLOC_CONF
    if args.command == "stages":
        from emperor.train.stages import main as stages_main

        return stages_main(args, config)
    from emperor.train import sft

    if args.command == "fit":
        return sft.train(config, args.root, args.pause_file)
    print(sft.merge(config, args.root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
