"""Incremental decode equals a full forward pass; NCM causality; sizes."""

from __future__ import annotations

import pytest
import torch

from ncm import (
    CONFIG_DIR,
    NcmLM,
    TokenLM,
    build_meta,
    cache_bytes,
    load_config,
    param_count,
    tiny_ncm,
    tiny_tok,
)

GIB = 2**30
MIB = 2**20


def _tokens(vocab: int, n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (1, n), generator=g)


# ----------------------------------------------------------------------------- token baseline


@pytest.mark.parametrize("chunk", [1, 3, 8, 13])
def test_tok_incremental_matches_full(chunk: int) -> None:
    torch.manual_seed(0)
    model = TokenLM(tiny_tok(window=8)).eval()
    n, prompt = 40, 17  # 40 tokens against window 8: the ring wraps several times
    tokens = _tokens(256, n)
    with torch.no_grad():
        full = model(tokens)
        # Prompt in chunks of `chunk` (T > 1 ring path), then single-token decode past the wrap.
        cache = model.make_cache(n)
        head = torch.cat(
            [model(tokens[:, i : min(i + chunk, prompt)], cache) for i in range(0, prompt, chunk)],
            dim=1,
        )
        tail = [model.step(tokens[:, t : t + 1], cache) for t in range(prompt, n)]
    out = torch.cat([head, *tail], dim=1)
    torch.testing.assert_close(out, full, atol=2e-4, rtol=1e-3)


def test_tok_short_run_uses_plain_cache() -> None:
    """A run shorter than the window needs no ring and must still match."""
    torch.manual_seed(1)
    model = TokenLM(tiny_tok(window=64)).eval()
    tokens = _tokens(256, 20)
    with torch.no_grad():
        full = model(tokens)
        cache = model.make_cache(20)
        out = torch.cat([model.step(tokens[:, t : t + 1], cache) for t in range(20)], dim=1)
    torch.testing.assert_close(out, full, atol=2e-4, rtol=1e-3)


# ----------------------------------------------------------------------------- NCM


def _ncm_incremental(model: NcmLM, tokens: torch.Tensor, prompt: int, chunk: int) -> torch.Tensor:
    k = model.cfg.k
    cache = model.make_cache(tokens.shape[1])
    cond = model.prefill(tokens[:, :prompt], cache, chunk=chunk)
    outs = []
    for start in range(prompt, tokens.shape[1], k):
        forced = tokens[:, start : start + k]
        _, logits = model.decode_concept(cond, cache, forced=forced, want_logits=True)
        outs.append(logits)
        if start + k < tokens.shape[1]:
            cond = model.append_tokens(forced, cache)
    return torch.cat(outs, dim=1)


@pytest.mark.parametrize("chunk", [4, 12, 32])
def test_ncm_incremental_matches_full(chunk: int) -> None:
    torch.manual_seed(0)
    # 64 tokens = 16 concepts: wraps the encoder ring (8 tokens) and backbone ring (4 concepts).
    model = NcmLM(tiny_ncm(enc_window=8, bb_window=4)).eval()
    tokens = _tokens(256, 64)
    prompt = 24
    with torch.no_grad():
        full = model(tokens)  # logits for tokens 4..63
        got = _ncm_incremental(model, tokens, prompt, chunk)
    torch.testing.assert_close(got, full[:, prompt - 4 :], atol=2e-4, rtol=1e-3)


def test_ncm_causality() -> None:
    torch.manual_seed(0)
    cfg = tiny_ncm(enc_window=8, bb_window=4)
    model = NcmLM(cfg).eval()
    k, n = cfg.k, 48
    tokens = _tokens(256, n)
    concept, pos = 5, 2  # change token `pos` of concept 5
    edited = tokens.clone()
    t_edit = concept * k + pos
    edited[0, t_edit] = (edited[0, t_edit] + 1) % 256
    with torch.no_grad():
        base, other = model(tokens), model(edited)
    # logits[:, t - k] predicts token t. Concepts before the edit are untouched, and inside the
    # edited concept the decoder has only seen tokens before `pos`, so it predicts up to and
    # including token `pos` identically.
    same_until = t_edit  # tokens < t_edit + 1
    torch.testing.assert_close(other[:, : same_until + 1 - k], base[:, : same_until + 1 - k])
    # Predictions after the edit are allowed to change, and do.
    assert not torch.allclose(other[:, t_edit + 1 - k :], base[:, t_edit + 1 - k :])


def test_ncm_conditioning_excludes_the_concept_being_written() -> None:
    """The backbone vector used for concept m+1 comes from concepts <= m only."""
    torch.manual_seed(2)
    cfg = tiny_ncm()
    model = NcmLM(cfg).eval()
    tokens = _tokens(256, 40)
    with torch.no_grad():
        c1 = model.make_cache(40)
        cond_a = model.prefill(tokens[:, :20], c1)
        edited = tokens.clone()
        edited[0, 20:24] = (edited[0, 20:24] + 7) % 256  # the *next* concept
        c2 = model.make_cache(40)
        cond_b = model.prefill(edited[:, :20], c2)
    torch.testing.assert_close(cond_a, cond_b)


# ----------------------------------------------------------------------------- sizes


@pytest.mark.parametrize("name", ["tok-250m", "ncm-250m"])
def test_full_config_param_count(name: str) -> None:
    cfg = load_config(CONFIG_DIR / f"{name}.toml")
    model = build_meta(cfg)
    total = param_count(model)
    print(name, f"{total / 1e6:.1f}M", {k: f"{v / 1e6:.1f}M" for k, v in model.parts().items()})
    assert 225e6 <= total <= 265e6
    assert sum(model.parts().values()) == total


def test_cache_bytes_at_one_million_tokens() -> None:
    n = 1_048_576
    ncm = cache_bytes(load_config(CONFIG_DIR / "ncm-250m.toml"), n)
    tok = cache_bytes(load_config(CONFIG_DIR / "tok-250m.toml"), n)
    assert abs(ncm - (1 * GIB + 12 * MIB)) / (1 * GIB + 12 * MIB) < 0.05
    assert abs(tok - (5 * GIB + 60 * MIB)) / (5 * GIB + 60 * MIB) < 0.05
    assert tok / ncm > 4.5


@pytest.mark.parametrize("n", [20, 64])
def test_cache_bytes_match_allocation(n: int) -> None:
    ncm_cfg, tok_cfg = tiny_ncm(), tiny_tok()
    ncm = NcmLM(ncm_cfg).make_cache(n)
    assert ncm.nbytes == cache_bytes(ncm_cfg, n, itemsize=4)
    tok = TokenLM(tok_cfg).make_cache(n)
    assert sum(c.nbytes for c in tok) == cache_bytes(tok_cfg, n, itemsize=4)
