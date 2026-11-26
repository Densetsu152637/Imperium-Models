#!/usr/bin/env bash
# What CI runs: the Rust workspace on the CPU backend only. CI has no GPU or CUDA toolkit, so
# the `cuda` feature is never built here.
set -euo pipefail
cd "$(dirname "$0")/.."

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

# ONNX Runtime (plan docs/plans/npu.md, stage 3). The feature builds link nothing, so clippy
# always covers them; the parity and e2e tests need the shared library, a CPU release fetched
# here by version and SHA-256. Keep ORT_VERSION in step with the `ort` crate's pin in
# crates/ncm-ort/Cargo.toml (rc.15 targets ONNX Runtime 1.31).
cargo clippy --workspace --all-targets --locked --features ncm-server/ort,ncm-bench/ort -- -D warnings
cargo clippy -p ncm-export -p ncm-ort --all-targets --locked --features ncm-export/ort-tests,ncm-ort/ort-tests -- -D warnings
ORT_VERSION="1.31.0"
# SHA-256 of onnxruntime-linux-x64-${ORT_VERSION}.tgz from the official release (to be pinned).
ORT_SHA256=""
if [ -n "$ORT_SHA256" ]; then
    ort_dir="$(mktemp -d)"
    trap 'rm -rf "$ort_dir"' EXIT
    ort_tgz="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
    curl -fsSL -o "$ort_dir/$ort_tgz"         "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ort_tgz}"
    echo "${ORT_SHA256}  $ort_dir/$ort_tgz" | sha256sum -c -
    tar -xzf "$ort_dir/$ort_tgz" -C "$ort_dir"
    export ORT_DYLIB_PATH="$ort_dir/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so"
    cargo test --locked -p ncm-export --features ort-tests --test parity
    cargo test --locked -p ncm-ort --features ort-tests --test parity
    NCM_E2E_DEVICE=ort-cpu cargo test --locked -p ncm-server --features ort --test e2e
else
    echo "check.sh: ORT_SHA256 is not pinned: skipping the ONNX Runtime parity and e2e tests" >&2
fi
