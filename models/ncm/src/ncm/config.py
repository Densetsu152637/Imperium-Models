"""Model configurations (dataclasses) and TOML loading."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


def is_full_layer(index: int, full_every: int) -> bool:
    """Every `full_every`-th layer (1-based) attends over the whole context."""
    return full_every > 0 and (index + 1) % full_every == 0


@dataclass(frozen=True)
class TokConfig:
    """Plain decoder-only baseline."""

    vocab_size: int = 32000
    d_model: int = 1024
    n_layers: int = 20
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 64
    ff_dim: int = 2688
    window: int = 4096  # tokens, sliding layers
    full_every: int = 4
    rope_base: float = 10_000.0
    rope_base_full: float = 1_000_000.0
    rope_scale: float = 1.0
    rope_scale_full: float = 1.0
    norm_eps: float = 1e-6


@dataclass(frozen=True)
class NcmConfig:
    """Hierarchical Next Concept Model with fixed k-token concepts."""

    vocab_size: int = 32000
    k: int = 4
    head_dim: int = 64
    local_dim: int = 512  # encoder, decoder and embedding width
    enc_layers: int = 2
    enc_heads: int = 8
    enc_kv_heads: int = 4
    enc_ff_dim: int = 1536
    enc_window: int = 512  # tokens
    dec_layers: int = 4
    dec_heads: int = 8
    dec_kv_heads: int = 4
    dec_ff_dim: int = 1536
    bb_dim: int = 1024
    bb_layers: int = 16
    bb_heads: int = 16
    bb_kv_heads: int = 4
    bb_ff_dim: int = 3200
    bb_window: int = 1024  # concepts, sliding layers
    full_every: int = 4
    rope_base: float = 10_000.0
    rope_base_full: float = 1_000_000.0
    rope_scale: float = 1.0
    rope_scale_full: float = 1.0
    norm_eps: float = 1e-6


Config = TokConfig | NcmConfig


def load_config(path: str | Path) -> Config:
    """Load a `configs/*.toml` file; its `model` key picks the dataclass."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    kind = data.pop("model")
    cls = {"tok": TokConfig, "ncm": NcmConfig}[kind]
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown config keys for {kind}: {sorted(unknown)}")
    return cls(**data)


def tiny_tok(window: int = 8) -> TokConfig:
    """Small config for CPU tests and smoke runs; the window is short enough to wrap."""
    return TokConfig(
        vocab_size=256, d_model=64, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=16,
        ff_dim=128, window=window, full_every=2,
    )  # fmt: skip


def tiny_ncm(enc_window: int = 8, bb_window: int = 4) -> NcmConfig:
    return NcmConfig(
        vocab_size=256, k=4, head_dim=16, local_dim=32, enc_layers=2, enc_heads=2,
        enc_kv_heads=1, enc_ff_dim=64, enc_window=enc_window, dec_layers=2, dec_heads=2,
        dec_kv_heads=1, dec_ff_dim=64, bb_dim=64, bb_layers=4, bb_heads=4, bb_kv_heads=2,
        bb_ff_dim=128, bb_window=bb_window, full_every=2,
    )  # fmt: skip
