# JAX 0.10.2

Revisions and patches for JAX 0.10.2 support. Uses RAIDEN_JAX=1002 with the polymorphic sibling base cast in `xla_compat.cc`.

## Patches
- `patches/py/jax_remove_local_wheels.patch`: Drops `dev_dependency` and `local_wheels`.
- `third_party/abseil/status_macros.patch`: Backports `status_macros.h` to Abseil 20260107.1. See [shared README](../README.md#when-a-revision-is-older-than-raidens-code).
