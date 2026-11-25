#!/usr/bin/env bash
# What CI runs: the PyTorch reference checks (while it remains), then the Rust workspace on the
# CPU backend only (CI has no GPU or CUDA toolkit, so the `cuda` feature is never built here).
set -euo pipefail
cd "$(dirname "$0")/.."

# PyTorch reference
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run ncm-bench --help > /dev/null
uv run ncm-bench --tiny --device cpu --dtype fp32 --context 64 --new-tokens 8 --repeats 1 > /dev/null

# Rust (Burn). rustup 1.28+ no longer installs the toolchain named by rust-toolchain.toml itself.
export CARGO_INCREMENTAL=0
if command -v rustup > /dev/null; then
    rustup show active-toolchain > /dev/null 2>&1 || rustup toolchain install
fi
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo run --locked --quiet -p ncm-bench -- --tiny --device cpu --context 64 --new-tokens 8 \
    --repeats 1 > /dev/null
