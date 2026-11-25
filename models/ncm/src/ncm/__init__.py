"""Next Concept Modelling (NCM) and the same-size token baseline."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .cache import KVCache
from .config import Config, NcmConfig, TokConfig, load_config, tiny_ncm, tiny_tok
from .ncm import NcmCache, NcmLM, ncm_cache_bytes
from .tok import TokenLM, tok_cache_bytes

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def build_model(cfg: Config) -> nn.Module:
    return NcmLM(cfg) if isinstance(cfg, NcmConfig) else TokenLM(cfg)


def build_meta(cfg: Config) -> nn.Module:
    """The model on the meta device: parameter counts without allocating weights."""
    with torch.device("meta"):
        return build_model(cfg)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def cache_bytes(cfg: Config, max_tokens: int, itemsize: int = 2, batch: int = 1) -> int:
    """Analytic KV cache size for a run of `max_tokens` tokens."""
    fn = ncm_cache_bytes if isinstance(cfg, NcmConfig) else tok_cache_bytes
    return fn(cfg, max_tokens, itemsize, batch)


__all__ = [
    "CONFIG_DIR",
    "Config",
    "KVCache",
    "NcmCache",
    "NcmConfig",
    "NcmLM",
    "TokConfig",
    "TokenLM",
    "build_meta",
    "build_model",
    "cache_bytes",
    "load_config",
    "param_count",
    "tiny_ncm",
    "tiny_tok",
]
