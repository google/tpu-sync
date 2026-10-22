# JAX 0.12.0

- JAX commit: `8dcb7aeced770bd22531b42d0b925c7342787c27`
- XLA commit: `d867969800d4a8fd045505eabc65e1e66f30794f`
- rules_ml_toolchain: `3a4df0c8e9e51687ac4c62c793a00a27286e5e83`
- abseil-cpp: `20260526.0`
- libtpu: `0.0.50`
- `RAIDEN_JAX`: `1200`

## Patches
- `third_party/py/jax_remove_local_wheels.patch`: Removes local wheel references from JAX's Python rules.
- `third_party/rules_ml_toolchain/no_register_toolchains.patch`: Removes duplicate toolchain registration.
