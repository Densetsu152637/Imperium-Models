#!/usr/bin/env bash
# What CI runs: lint, format check, tests, and the benchmark CLI smoke tests.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run ncm-bench --help > /dev/null
uv run ncm-bench --tiny --device cpu --dtype fp32 --context 64 --new-tokens 8 --repeats 1 > /dev/null
