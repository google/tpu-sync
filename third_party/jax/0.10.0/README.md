# JAX 0.10.0

Revisions and patches for JAX 0.10.0 support. Uses the RAIDEN_JAX=1000 compat branch.

## Patches
- `rules_ml_toolchain/no_register_toolchains.patch`: Local copy of toolchain patch adjusted for line offsets.
- `third_party/abseil/status_macros.patch`: Backports `status_macros.h` to Abseil 20260107.1. See [shared README](../README.md#when-a-revision-is-older-than-raidens-code).
