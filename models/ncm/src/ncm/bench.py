"""Batch-1 decode-speed benchmark: NCM against the token baseline, random weights.

    uv run ncm-bench --model both --context 2048 16384 --new-tokens 256 --device cuda

Both models share the token vocabulary, so decode tokens per second are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from . import (
    CONFIG_DIR,
    NcmLM,
    build_model,
    cache_bytes,
    load_config,
    param_count,
    tiny_ncm,
    tiny_tok,
)
from .layers import Mlp

PASS_RATIO = 1.5
WEAK_RATIO = 1.2
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def verdict(ratio: float) -> str:
    return "PASS" if ratio >= PASS_RATIO else "WEAK" if ratio >= WEAK_RATIO else "FAIL"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_once(
    model: torch.nn.Module, prompt: torch.Tensor, new_tokens: int, device: torch.device
) -> tuple[float, float]:
    """Prefill the prompt, then greedily decode `new_tokens` tokens; returns (prefill s, decode s).

    The token baseline prefills all but the last prompt token and decodes from that one, so its
    decode loop produces exactly `new_tokens` tokens like the NCM loop does.
    """
    dtype = next(model.parameters()).dtype
    context = prompt.shape[1]
    cache = model.make_cache(context + new_tokens, 1, dtype, device)
    _sync(device)
    t0 = time.perf_counter()
    if isinstance(model, NcmLM):
        cond = model.prefill(prompt, cache)
        _sync(device)
        t1 = time.perf_counter()
        concepts = new_tokens // model.cfg.k
        for c in range(concepts):
            tokens, _ = model.decode_concept(cond, cache)
            if c < concepts - 1:
                cond = model.append_tokens(tokens, cache)
    else:
        model.prefill(prompt[:, :-1], cache)
        _sync(device)
        t1 = time.perf_counter()
        token = prompt[:, -1:]
        for _ in range(new_tokens):
            token = model.step(token, cache).argmax(dim=-1)
    _sync(device)
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1


def bench_model(
    name: str, cfg, args: argparse.Namespace, device: torch.device, dtype: torch.dtype
) -> dict:
    torch.manual_seed(0)
    model = build_model(cfg).to(device=device, dtype=dtype).eval()
    if args.compile:
        for m in model.modules():
            if isinstance(m, Mlp):
                m.forward = torch.compile(m.forward)
    n_params = param_count(model)
    itemsize = torch.empty((), dtype=dtype).element_size()
    new_tokens = args.new_tokens - args.new_tokens % getattr(cfg, "k", 1)
    result = {"name": name, "params": n_params, "new_tokens": new_tokens, "contexts": {}}
    for context in args.context:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        k = getattr(cfg, "k", 1)
        context -= context % k
        gen = torch.Generator().manual_seed(1)
        prompt = torch.randint(0, cfg.vocab_size, (1, context), generator=gen).to(device)
        run_once(model, prompt[:, : min(context, 64 * k)], 2 * k, device)  # warm-up
        prefills, decodes = [], []
        for _ in range(args.repeats):
            p, d = run_once(model, prompt, new_tokens, device)
            prefills.append(p)
            decodes.append(d)
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        result["contexts"][context] = {
            "prefill_s": statistics.median(prefills),
            "decode_tok_s": new_tokens / statistics.median(decodes),
            "peak_memory_bytes": peak,
            "cache_bytes": cache_bytes(cfg, context + new_tokens, itemsize),
        }
    return result


def _fmt_mem(n: int | None) -> str:
    return "n/a" if n is None else f"{n / 2**30:.2f} GiB"


def report(results: list[dict], tiny: bool) -> list[dict]:
    """Print per-model rows and the NCM/tok ratio per context; returns the ratio records."""
    for res in results:
        print(f"{res['name']}: {res['params'] / 1e6:.1f}M params, {res['new_tokens']} new tokens")
        for ctx, r in res["contexts"].items():
            print(
                f"  ctx {ctx:>7}: prefill {r['prefill_s']:.3f}s"
                f"  decode {r['decode_tok_s']:.1f} tok/s"
                f"  peak {_fmt_mem(r['peak_memory_bytes'])}  cache {_fmt_mem(r['cache_bytes'])}"
            )
    by_name = {r["name"]: r for r in results}
    ratios = []
    if "tok" in by_name and "ncm" in by_name:
        for ctx, ncm in by_name["ncm"]["contexts"].items():
            tok = by_name["tok"]["contexts"].get(ctx)
            if tok is None:
                continue
            ratio = ncm["decode_tok_s"] / tok["decode_tok_s"]
            v = verdict(ratio)
            note = "  (tiny config: smoke run, not a gate result)" if tiny else ""
            print(f"ratio ncm/tok @ ctx {ctx}: {ratio:.2f}x {v}{note}")
            ratios.append({"context": ctx, "ratio": ratio, "verdict": v})
    return ratios


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ncm-bench",
        description="Batch-1 decode speed of the NCM against the token baseline (random weights).",
    )
    ap.add_argument("--model", choices=["tok", "ncm", "both"], default="both")
    ap.add_argument("--context", type=int, nargs="+", default=[2048, 16384], metavar="N")
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    ap.add_argument(
        "--compile", action="store_true", help="torch.compile the MLPs only (experimental)"
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", type=Path, metavar="PATH", help="also write results as JSON")
    ap.add_argument("--tiny", action="store_true", help="tiny configs for a CPU smoke run")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        ap.error("--device cuda requested but CUDA is not available")
    dtype = DTYPES[args.dtype]
    kinds = ["tok", "ncm"] if args.model == "both" else [args.model]
    configs = {
        "tok": tiny_tok() if args.tiny else load_config(CONFIG_DIR / "tok-250m.toml"),
        "ncm": tiny_ncm() if args.tiny else load_config(CONFIG_DIR / "ncm-250m.toml"),
    }
    with torch.inference_mode():
        results = [bench_model(kind, configs[kind], args, device, dtype) for kind in kinds]
    ratios = report(results, args.tiny)
    if args.json:
        payload = {
            "device": args.device,
            "dtype": args.dtype,
            "tiny": args.tiny,
            "results": results,
            "ratios": ratios,
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
