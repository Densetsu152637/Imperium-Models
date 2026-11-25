"""Next Concept Model: local encoder, k-token concepts, global backbone, local decoder.

Layout of a sequence of N = M * k tokens (M concepts):

* The local encoder (causal, sliding window) turns tokens into states.
* The k states of concept m are concatenated and projected into one backbone vector.
* The backbone (causal over concepts) output at concept m, `h_m`, has seen concepts 0..m only.
* The local decoder for concept m + 1 is conditioned on `h_m` and the tokens of concept m + 1
  it has already written, and cross-attends to encoder states of tokens up to the end of
  concept m (the most recent `enc_window` of them).

Concept 0 is context only: it has no earlier concept to condition on, so logits cover tokens
k .. N - 1 (`logits[:, t - k]` predicts token t).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .cache import KVCache, make_cache, stack_cache_bytes
from .config import NcmConfig
from .layers import Attention, CrossAttention, Mlp, Rope, Stack, init_weights, rope_tables


class DecoderBlock(nn.Module):
    def __init__(self, cfg: NcmConfig):
        super().__init__()
        d, hd, eps = cfg.local_dim, cfg.head_dim, cfg.norm_eps
        self.n1 = nn.RMSNorm(d, eps=eps)
        self.self_attn = Attention(d, cfg.dec_heads, cfg.dec_kv_heads, hd, None)
        self.n2 = nn.RMSNorm(d, eps=eps)
        self.cross = CrossAttention(d, d, cfg.dec_heads, cfg.dec_kv_heads, hd)
        self.n3 = nn.RMSNorm(d, eps=eps)
        self.mlp = Mlp(d, cfg.dec_ff_dim)

    def forward_train(
        self,
        x: torch.Tensor,
        rope_self: Rope,
        rope_q: Rope,
        kv: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """x is [B, M', k, d]: every concept's decoder runs in parallel."""
        b, m, k, d = x.shape
        h = self.n1(x).reshape(b * m, k, d)
        x = x + self.self_attn(h, rope_self).reshape(b, m, k, d)
        h = self.n2(x).reshape(b, m * k, d)
        x = x + self.cross(h, rope_q, kv=kv, mask=mask).reshape(b, m, k, d)
        return x + self.mlp(self.n3(x))

    def step(
        self,
        x: torch.Tensor,
        rope_self: Rope,
        rope_q: Rope,
        self_cache: KVCache,
        cross_cache: KVCache,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.n1(x), rope_self, self_cache)
        x = x + self.cross(self.n2(x), rope_q, cache=cross_cache)
        return x + self.mlp(self.n3(x))


@dataclass
class NcmCache:
    backbone: list[KVCache]
    encoder: list[KVCache]
    cross: list[KVCache]  # per decoder layer: encoder-state keys and values, ring of enc_window
    dec_self: list[KVCache]  # per decoder layer: only the current concept, reset every concept
    tokens: int = 0  # tokens consumed by the encoder so far
    _all: list[KVCache] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._all = [*self.backbone, *self.encoder, *self.cross, *self.dec_self]

    @property
    def nbytes(self) -> int:
        return sum(c.nbytes for c in self._all)


class NcmLM(nn.Module):
    def __init__(self, cfg: NcmConfig):
        super().__init__()
        self.cfg = cfg
        d, bb = cfg.local_dim, cfg.bb_dim
        self.emb = nn.Embedding(cfg.vocab_size, d)
        self.encoder = Stack(
            d, cfg.enc_layers, cfg.enc_heads, cfg.enc_kv_heads, cfg.head_dim, cfg.enc_ff_dim,
            cfg.enc_window, 0, cfg.rope_base, cfg.rope_base_full, cfg.rope_scale,
            cfg.rope_scale_full, cfg.norm_eps,
        )  # fmt: skip
        self.pool = nn.Linear(cfg.k * d, bb, bias=False)
        self.backbone = Stack(
            bb, cfg.bb_layers, cfg.bb_heads, cfg.bb_kv_heads, cfg.head_dim, cfg.bb_ff_dim,
            cfg.bb_window, cfg.full_every, cfg.rope_base, cfg.rope_base_full, cfg.rope_scale,
            cfg.rope_scale_full, cfg.norm_eps,
        )  # fmt: skip
        self.cond_proj = nn.Linear(bb, d, bias=False)
        self.start_emb = nn.Parameter(torch.empty(d))
        self.dec_blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.dec_layers))
        self.dec_norm = nn.RMSNorm(d, eps=cfg.norm_eps)
        self.apply(init_weights)
        nn.init.normal_(self.start_emb, mean=0.0, std=0.02)

    def parts(self) -> dict[str, int]:
        count = lambda *ms: sum(p.numel() for m in ms for p in m.parameters())  # noqa: E731
        return {
            "embeddings (tied)": count(self.emb),
            "local encoder": count(self.encoder),
            "global backbone": count(self.backbone),
            "local decoder": count(self.dec_blocks, self.dec_norm),
            "projections": count(self.pool, self.cond_proj) + self.start_emb.numel(),
        }

    def _rope_local(self, positions: torch.Tensor, dtype: torch.dtype) -> Rope:
        cfg = self.cfg
        return rope_tables(positions, cfg.head_dim, cfg.rope_base, cfg.rope_scale, dtype)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.dec_norm(x), self.emb.weight)

    # ------------------------------------------------------------------ full sequence

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Logits [B, N - k, V] for tokens k..N-1 of a sequence whose length is a multiple of k."""
        cfg = self.cfg
        b, n = tokens.shape
        k = cfg.k
        if n % k or n < 2 * k:
            raise ValueError(f"sequence length must be a multiple of k={k} and at least {2 * k}")
        m = n // k
        states = self.encoder(self.emb(tokens))  # [B, N, d]
        concepts = self.pool(states.reshape(b, m, k * cfg.local_dim))
        h = self.backbone(concepts)  # [B, M, bb]
        cond = self.cond_proj(h[:, :-1])  # [B, M-1, d], conditions concepts 1..M-1
        targets = tokens[:, k:].reshape(b, m - 1, k)
        start = self.start_emb.expand(b, m - 1, 1, -1)
        x = torch.cat([start, self.emb(targets[..., :-1])], dim=2) + cond[:, :, None]
        device, dtype = tokens.device, x.dtype
        rope_self = self._rope_local(torch.arange(k, device=device), dtype)
        qpos = torch.arange(k, n, device=device)
        rope_q = self._rope_local(qpos, dtype)
        rope_k = self._rope_local(torch.arange(n, device=device), dtype)
        # Concept m' may see encoder states of tokens [k*m' - enc_window, k*m' - 1].
        end = (qpos // k) * k - 1
        kpos = torch.arange(n, device=device)
        mask = (kpos[None] <= end[:, None]) & (kpos[None] > end[:, None] - cfg.enc_window)
        for blk in self.dec_blocks:
            kv = blk.cross.make_kv(states, rope_k)
            x = blk.forward_train(x, rope_self, rope_q, kv, mask)
        return self._logits(x.reshape(b, (m - 1) * k, -1))

    # ------------------------------------------------------------------ incremental

    def make_cache(
        self, max_tokens: int, batch: int = 1, dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> NcmCache:  # fmt: skip
        cfg = self.cfg
        concepts = math.ceil(max_tokens / cfg.k)
        return NcmCache(
            backbone=self.backbone.make_caches(concepts, batch, dtype, device),
            encoder=self.encoder.make_caches(max_tokens, batch, dtype, device),
            cross=[
                make_cache(
                    batch, cfg.dec_kv_heads, cfg.head_dim, max_tokens, cfg.enc_window, dtype, device
                )
                for _ in range(cfg.dec_layers)
            ],
            dec_self=[
                make_cache(batch, cfg.dec_kv_heads, cfg.head_dim, cfg.k, None, dtype, device)
                for _ in range(cfg.dec_layers)
            ],
        )  # fmt: skip

    @torch.no_grad()
    def append_tokens(self, tokens: torch.Tensor, cache: NcmCache) -> torch.Tensor:
        """Encode whole concepts and run the backbone on them.

        Returns the conditioning vector [B, 1, d] for the next concept's decoder.
        """
        cfg = self.cfg
        b, t = tokens.shape
        if t % cfg.k:
            raise ValueError(f"token count must be a multiple of k={cfg.k}")
        states = self.encoder(self.emb(tokens), cache.encoder)
        positions = torch.arange(cache.tokens, cache.tokens + t, device=tokens.device)
        rope_k = self._rope_local(positions, states.dtype)
        for blk, kv_cache in zip(self.dec_blocks, cache.cross, strict=True):
            kv_cache.write(*blk.cross.make_kv(states, rope_k))
        concepts = self.pool(states.reshape(b, t // cfg.k, cfg.k * cfg.local_dim))
        h = self.backbone(concepts, cache.backbone)
        cache.tokens += t
        return self.cond_proj(h[:, -1:])

    @torch.no_grad()
    def prefill(self, tokens: torch.Tensor, cache: NcmCache, chunk: int = 512) -> torch.Tensor:
        """Feed a prompt (length a multiple of k) in chunks; returns the next-concept condition."""
        chunk -= chunk % self.cfg.k
        cond = None
        for i in range(0, tokens.shape[1], chunk):
            cond = self.append_tokens(tokens[:, i : i + chunk], cache)
        return cond

    @torch.no_grad()
    def decode_concept(
        self,
        cond: torch.Tensor,
        cache: NcmCache,
        forced: torch.Tensor | None = None,
        want_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Write the next k tokens greedily (or teacher-forced) with the local decoder.

        Returns tokens [B, k] and, if requested, logits [B, k, V]. The tokens are not yet
        encoded; pass them to `append_tokens` to continue.
        """
        cfg = self.cfg
        b = cond.shape[0]
        dtype, device = cond.dtype, cond.device
        for c in cache.dec_self:
            c.reset()
        rope_self = self._rope_local(torch.arange(cfg.k, device=device), dtype)
        rope_q = self._rope_local(
            torch.arange(cache.tokens, cache.tokens + cfg.k, device=device), dtype
        )
        tokens, logits = [], []
        inp = self.start_emb.expand(b, 1, -1)
        for j in range(cfg.k):
            x = inp + cond
            rs = (rope_self[0][j : j + 1], rope_self[1][j : j + 1])
            rq = (rope_q[0][j : j + 1], rope_q[1][j : j + 1])
            for blk, sc, cc in zip(self.dec_blocks, cache.dec_self, cache.cross, strict=True):
                x = blk.step(x, rs, rq, sc, cc)
            lg = self._logits(x)
            nxt = forced[:, j : j + 1] if forced is not None else lg.argmax(dim=-1)
            tokens.append(nxt)
            if want_logits:
                logits.append(lg)
            inp = self.emb(nxt)
        return torch.cat(tokens, dim=1), (torch.cat(logits, dim=1) if want_logits else None)


def ncm_cache_bytes(cfg: NcmConfig, max_tokens: int, itemsize: int = 2, batch: int = 1) -> int:
    concepts = math.ceil(max_tokens / cfg.k)
    hd = cfg.head_dim
    total = stack_cache_bytes(
        cfg.bb_layers, cfg.full_every, cfg.bb_window, concepts, cfg.bb_kv_heads, hd, itemsize, batch
    )
    total += stack_cache_bytes(
        cfg.enc_layers, 0, cfg.enc_window, max_tokens, cfg.enc_kv_heads, hd, itemsize, batch
    )
    # Cross-attention ring and the per-concept decoder self cache.
    total += stack_cache_bytes(
        cfg.dec_layers, 0, cfg.enc_window, max_tokens, cfg.dec_kv_heads, hd, itemsize, batch
    )
    total += stack_cache_bytes(
        cfg.dec_layers, 1, cfg.k, cfg.k, cfg.dec_kv_heads, hd, itemsize, batch
    )
    return total
