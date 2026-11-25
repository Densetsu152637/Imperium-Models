"""Export a tiny-config weight and logit fixture so the Rust models can be checked against the
PyTorch reference.

    uv run python scripts/export_parity_fixture.py

Writes `crates/ncm-model/tests/fixtures/parity.json`. Linear weights are stored transposed
(`[in, out]`, the Burn layout); every other tensor keeps its PyTorch shape. Parameters are
listed in module registration order, which is also the field order of the Rust modules.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from ncm import NcmConfig, NcmLM, TokConfig, TokenLM

OUT = Path(__file__).resolve().parents[1] / "crates/ncm-model/tests/fixtures/parity.json"


def micro_tok() -> TokConfig:
    return TokConfig(
        vocab_size=64, d_model=16, n_layers=4, n_heads=2, n_kv_heads=1, head_dim=8,
        ff_dim=24, window=6, full_every=2,
    )  # fmt: skip


def micro_ncm() -> NcmConfig:
    return NcmConfig(
        vocab_size=64, k=4, head_dim=8, local_dim=16, enc_layers=2, enc_heads=2,
        enc_kv_heads=1, enc_ff_dim=24, enc_window=8, dec_layers=2, dec_heads=2,
        dec_kv_heads=1, dec_ff_dim=24, bb_dim=16, bb_layers=4, bb_heads=2, bb_kv_heads=1,
        bb_ff_dim=24, bb_window=3, full_every=2,
    )  # fmt: skip


def flat(t: torch.Tensor) -> list[float]:
    return [float(f"{v:.7g}") for v in t.detach().reshape(-1).tolist()]


def params(model: nn.Module) -> list[dict]:
    """Parameters in Rust field order: `named_parameters` lists a module's own parameters
    first, but the Rust NCM keeps `start_emb` right after `cond_proj`."""
    linear_weights = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)}
    named = list(model.named_parameters())
    names = [n for n, _ in named]
    if "start_emb" in names:
        entry = named.pop(names.index("start_emb"))
        named.insert([n for n, _ in named].index("cond_proj.weight") + 1, entry)
    out = []
    for _, p in named:
        t = p.detach().t().contiguous() if id(p) in linear_weights else p.detach()
        out.append({"shape": list(t.shape), "data": flat(t)})
    return out


def sample_tokens(vocab: int, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (1, n), generator=g)


def main() -> None:
    torch.manual_seed(1234)
    tok_cfg, ncm_cfg = micro_tok(), micro_ncm()
    tok_model, ncm_model = TokenLM(tok_cfg).eval(), NcmLM(ncm_cfg).eval()
    # Larger than the init scale so that RoPE and attention do real work.
    with torch.no_grad():
        for model in (tok_model, ncm_model):
            for name, p in model.named_parameters():
                if p.dim() > 1 or name == "start_emb":  # leave the norm gains at 1
                    p.mul_(25.0)
    tok_tokens = sample_tokens(tok_cfg.vocab_size, 20, 0)
    ncm_tokens = sample_tokens(ncm_cfg.vocab_size, 28, 1)
    with torch.no_grad():
        tok_logits = tok_model(tok_tokens)
        ncm_logits = ncm_model(ncm_tokens)
    payload = {
        "tok": {
            "config": asdict(tok_cfg),
            "tokens": tok_tokens[0].tolist(),
            "logits": flat(tok_logits),
            "params": params(tok_model),
        },
        "ncm": {
            "config": asdict(ncm_cfg),
            "tokens": ncm_tokens[0].tolist(),
            "logits": flat(ncm_logits),
            "params": params(ncm_model),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
