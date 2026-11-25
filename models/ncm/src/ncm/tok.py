"""Token baseline: decoder-only transformer with tied embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .cache import KVCache, stack_cache_bytes
from .config import TokConfig
from .layers import Stack, init_weights


class TokenLM(nn.Module):
    def __init__(self, cfg: TokConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.stack = Stack(
            cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.ff_dim,
            cfg.window, cfg.full_every, cfg.rope_base, cfg.rope_base_full, cfg.rope_scale,
            cfg.rope_scale_full, cfg.norm_eps,
        )  # fmt: skip
        self.apply(init_weights)

    def parts(self) -> dict[str, int]:
        count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        return {"embeddings (tied)": count(self.emb), "transformer": count(self.stack)}

    def forward(self, tokens: torch.Tensor, caches: list[KVCache] | None = None) -> torch.Tensor:
        """Logits [B, T, V]; position t predicts token t + 1. Pass caches to run incrementally."""
        x = self.stack(self.emb(tokens), caches)
        return F.linear(x, self.emb.weight)

    def make_cache(
        self, max_tokens: int, batch: int = 1, dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> list[KVCache]:  # fmt: skip
        return self.stack.make_caches(max_tokens, batch, dtype, device)

    @torch.no_grad()
    def prefill(
        self, tokens: torch.Tensor, caches: list[KVCache], chunk: int = 512
    ) -> torch.Tensor | None:
        """Feed the prompt in chunks; returns the logits of the last chunk, or None if empty."""
        logits = None
        for i in range(0, tokens.shape[1], chunk):
            logits = self(tokens[:, i : i + chunk], caches)
        return logits

    @torch.no_grad()
    def step(self, token: torch.Tensor, caches: list[KVCache]) -> torch.Tensor:
        """One decode step for tokens [B, 1]; returns logits [B, 1, V]."""
        return self(token, caches)


def tok_cache_bytes(cfg: TokConfig, max_tokens: int, itemsize: int = 2, batch: int = 1) -> int:
    return stack_cache_bytes(
        cfg.n_layers, cfg.full_every, cfg.window, max_tokens, cfg.n_kv_heads, cfg.head_dim,
        itemsize, batch,
    )  # fmt: skip
