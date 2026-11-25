"""Shared building blocks: RoPE, SwiGLU, attention, transformer blocks and stacks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .cache import KVCache, build_mask, make_cache
from .config import is_full_layer

Rope = tuple[torch.Tensor, torch.Tensor]
_INV_FREQ: dict[tuple, torch.Tensor] = {}


def rope_tables(
    positions: torch.Tensor, head_dim: int, base: float, scale: float, dtype: torch.dtype
) -> Rope:
    """cos and sin tables [T, head_dim / 2]; `scale` divides positions (linear scaling hook)."""
    key = (positions.device, head_dim, base)
    inv = _INV_FREQ.get(key)
    if inv is None:
        steps = torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32)
        inv = _INV_FREQ[key] = 1.0 / (base ** (steps / head_dim))
    angles = (positions.to(torch.float32) / scale)[:, None] * inv[None, :]
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: torch.Tensor, rope: Rope) -> torch.Tensor:
    cos, sin = rope
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    is_causal: bool = False,
) -> torch.Tensor:
    """SDPA with grouped-query heads; the one attention path both models use."""
    if q.device.type == "cuda":
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal, enable_gqa=True
        )
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)


class Mlp(nn.Module):
    """SwiGLU."""

    def __init__(self, d: int, ff: int):
        super().__init__()
        self.gate_up = nn.Linear(d, 2 * ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Attention(nn.Module):
    """Causal GQA self-attention; `window` None means full attention."""

    def __init__(self, d: int, heads: int, kv_heads: int, head_dim: int, window: int | None):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim, self.window = heads, kv_heads, head_dim, window
        self.qkv = nn.Linear(d, (heads + 2 * kv_heads) * head_dim, bias=False)
        self.o = nn.Linear(heads * head_dim, d, bias=False)

    def forward(self, x: torch.Tensor, rope: Rope, cache: KVCache | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        h, kvh, hd = self.heads, self.kv_heads, self.head_dim
        q, k, v = self.qkv(x).split([h * hd, kvh * hd, kvh * hd], dim=-1)
        q = apply_rope(q.reshape(b, t, h, hd).transpose(1, 2), rope)
        k = apply_rope(k.reshape(b, t, kvh, hd).transpose(1, 2), rope)
        v = v.reshape(b, t, kvh, hd).transpose(1, 2)
        if cache is None:
            if self.window is None:
                out = attend(q, k, v, None, is_causal=True)
            else:
                pos = torch.arange(t, device=x.device)
                out = attend(q, k, v, build_mask(pos, pos, self.window))
        else:
            keys, values, mask = cache.append(k, v)
            out = attend(q, keys, values, mask)
        return self.o(out.transpose(1, 2).reshape(b, t, h * hd))


class CrossAttention(nn.Module):
    """Queries from the local decoder; keys and values from local-encoder states."""

    def __init__(self, d: int, d_kv: int, heads: int, kv_heads: int, head_dim: int):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.q = nn.Linear(d, heads * head_dim, bias=False)
        self.kv = nn.Linear(d_kv, 2 * kv_heads * head_dim, bias=False)
        self.o = nn.Linear(heads * head_dim, d, bias=False)

    def make_kv(self, states: torch.Tensor, rope: Rope) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = states.shape
        k, v = self.kv(states).chunk(2, dim=-1)
        k = apply_rope(k.reshape(b, t, self.kv_heads, self.head_dim).transpose(1, 2), rope)
        return k, v.reshape(b, t, self.kv_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        rope_q: Rope,
        kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache: KVCache | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q(x).reshape(b, t, self.heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, rope_q)
        keys, values = cache.view() if cache is not None else kv
        out = attend(q, keys, values, mask)
        return self.o(out.transpose(1, 2).reshape(b, t, self.heads * self.head_dim))


class Block(nn.Module):
    def __init__(
        self,
        d: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        ff: int,
        window: int | None,
        eps: float,
    ):
        super().__init__()
        self.n1 = nn.RMSNorm(d, eps=eps)
        self.attn = Attention(d, heads, kv_heads, head_dim, window)
        self.n2 = nn.RMSNorm(d, eps=eps)
        self.mlp = Mlp(d, ff)

    def forward(self, x: torch.Tensor, rope: Rope, cache: KVCache | None = None) -> torch.Tensor:
        x = x + self.attn(self.n1(x), rope, cache)
        return x + self.mlp(self.n2(x))


class Stack(nn.Module):
    """Pre-norm transformer layers with the sliding / every-Nth-full layer pattern."""

    def __init__(
        self,
        d: int,
        n_layers: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        ff: int,
        window: int,
        full_every: int,
        rope_base: float,
        rope_base_full: float,
        rope_scale: float,
        rope_scale_full: float,
        eps: float,
    ):
        super().__init__()
        self.kv_heads, self.head_dim = kv_heads, head_dim
        self.window, self.full_every = window, full_every
        self.rope = (rope_base, rope_scale)
        self.rope_full = (rope_base_full, rope_scale_full)
        self.blocks = nn.ModuleList(
            Block(d, heads, kv_heads, head_dim, ff, self.layer_window(i), eps)
            for i in range(n_layers)
        )
        self.norm = nn.RMSNorm(d, eps=eps)

    def layer_window(self, index: int) -> int | None:
        return None if is_full_layer(index, self.full_every) else self.window

    def forward(self, x: torch.Tensor, caches: list[KVCache] | None = None) -> torch.Tensor:
        t = x.shape[1]
        start = caches[0].length if caches else 0
        pos = torch.arange(start, start + t, device=x.device)
        rope_local = rope_tables(pos, self.head_dim, *self.rope, x.dtype)
        has_full = any(is_full_layer(i, self.full_every) for i in range(len(self.blocks)))
        rope_full = rope_tables(pos, self.head_dim, *self.rope_full, x.dtype) if has_full else None
        for i, block in enumerate(self.blocks):
            rope = rope_full if is_full_layer(i, self.full_every) else rope_local
            x = block(x, rope, caches[i] if caches else None)
        return self.norm(x)

    def make_caches(
        self, max_positions: int, batch: int, dtype: torch.dtype, device: torch.device | str
    ) -> list[KVCache]:
        return [
            make_cache(
                batch,
                self.kv_heads,
                self.head_dim,
                max_positions,
                self.layer_window(i),
                dtype,
                device,
            )
            for i in range(len(self.blocks))
        ]


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear | nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
