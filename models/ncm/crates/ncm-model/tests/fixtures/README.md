# Fixtures

`parity.json` holds tiny-config weights, tokens and logits exported from the PyTorch reference
implementation at commit 2fa4688 (since deleted). It is frozen: `tests/parity.rs` checks that the
Rust models reproduce those logits. Linear weights are stored transposed (`[in, out]`, the Burn
layout); tensors are listed in the field order of the Rust modules.
