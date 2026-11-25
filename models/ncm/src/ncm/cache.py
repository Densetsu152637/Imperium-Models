"""Preallocated KV caches shared by both models.

Full-attention layers keep `capacity` positions and grow into it. Sliding layers keep a ring
buffer of exactly `window` positions. Positions are contiguous: the cache itself tracks the next
absolute position (`length`), so callers never pass positions.
"""

from __future__ import annotations

import torch

from .config import is_full_layer


def build_mask(qpos: torch.Tensor, kpos: torch.Tensor, window: int | None) -> torch.Tensor:
    """Boolean [Tq, Tk] mask: key exists (>= 0), is causal, and is inside the window."""
    mask = (kpos[None, :] >= 0) & (kpos[None, :] <= qpos[:, None])
    if window is not None:
        mask &= (qpos[:, None] - kpos[None, :]) < window
    return mask


class KVCache:
    def __init__(
        self,
        batch: int,
        n_kv_heads: int,
        head_dim: int,
        capacity: int,
        window: int | None,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.window = window
        self.capacity = capacity if window is None else window
        shape = (batch, n_kv_heads, self.capacity, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.length = 0  # next absolute position

    @property
    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

    def reset(self) -> None:
        self.length = 0

    def view(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Everything currently held, unordered for rings (RoPE is already baked into keys)."""
        n = min(self.length, self.capacity)
        return self.k[:, :, :n], self.v[:, :, :n]

    def write(self, k: torch.Tensor, v: torch.Tensor) -> None:
        t = k.shape[2]
        start = self.length
        if self.window is None:
            if start + t > self.capacity:
                raise ValueError(f"cache capacity {self.capacity} exceeded")
            self.k[:, :, start : start + t] = k
            self.v[:, :, start : start + t] = v
        elif t == 1:
            slot = start % self.capacity
            self.k[:, :, slot : slot + 1] = k
            self.v[:, :, slot : slot + 1] = v
        else:
            n = min(t, self.capacity)
            pos = torch.arange(start + t - n, start + t, device=k.device)
            slots = pos % self.capacity
            self.k.index_copy_(2, slots, k[:, :, t - n :])
            self.v.index_copy_(2, slots, v[:, :, t - n :])
        self.length += t

    def _slot_positions(self) -> torch.Tensor:
        """Absolute position held by each ring slot, -1 when empty."""
        slots = torch.arange(self.capacity, device=self.k.device)
        last = self.length - 1
        pos = last - ((last - slots) % self.capacity)
        return pos.masked_fill(pos < 0, -1)

    def append(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Add new keys and values; return (K, V, mask) for the queries at the same positions."""
        t = k.shape[2]
        start = self.length
        device = k.device
        qpos = torch.arange(start, start + t, device=device) if t > 1 else None
        if self.window is None:
            self.write(k, v)
            keys, values = self.view()
            if t == 1:
                return keys, values, None
            return keys, values, build_mask(qpos, torch.arange(self.length, device=device), None)
        if t == 1:
            # The slot being overwritten holds position start - window: already out of window.
            self.write(k, v)
            keys, values = self.view()
            return keys, values, None
        keys = torch.cat([self.k, k], dim=2)
        values = torch.cat([self.v, v], dim=2)
        kpos = torch.cat([self._slot_positions(), qpos])
        mask = build_mask(qpos, kpos, self.window)
        self.write(k, v)
        return keys, values, mask


def make_cache(
    batch: int,
    n_kv_heads: int,
    head_dim: int,
    max_positions: int,
    window: int | None,
    dtype: torch.dtype,
    device: torch.device | str,
) -> KVCache:
    """A ring when the window is smaller than the run, otherwise a plain growing cache."""
    if window is None or max_positions <= window:
        return KVCache(batch, n_kv_heads, head_dim, max_positions, None, dtype, device)
    return KVCache(batch, n_kv_heads, head_dim, window, window, dtype, device)


def layer_cache_bytes(
    n_kv_heads: int,
    head_dim: int,
    max_positions: int,
    window: int | None,
    itemsize: int,
    batch: int,
) -> int:
    cap = max_positions if window is None else min(window, max_positions)
    return 2 * batch * n_kv_heads * cap * head_dim * itemsize


def stack_cache_bytes(
    n_layers: int,
    full_every: int,
    window: int,
    max_positions: int,
    n_kv_heads: int,
    head_dim: int,
    itemsize: int = 2,
    batch: int = 1,
) -> int:
    total = 0
    for i in range(n_layers):
        w = None if is_full_layer(i, full_every) else window
        total += layer_cache_bytes(n_kv_heads, head_dim, max_positions, w, itemsize, batch)
    return total
