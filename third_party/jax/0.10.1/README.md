# JAX 0.10.1

Revisions and patches for JAX 0.10.1 support. Shares the C++ compat definitions with 0.10.0.

## Patches
- `patches/rules_ml_toolchain/no_register_toolchains.patch`: Toolchain patch adjusted for this pin, whose `register_toolchains` call differs from the shared patch.
- `patches/py/jax_remove_local_wheels.patch`: Strips `local_wheels` from JAX pip parse.
- `third_party/abseil/status_macros.patch`: Backports `status_macros.h` to Abseil 20260107.1. See [shared README](../README.md#when-a-revision-is-older-than-raidens-code).
