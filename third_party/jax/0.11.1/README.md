# JAX 0.11.1

- JAX commit: `2d66622450e2c8633cda2307688ef7aa294bd6eb`
- XLA commit: `f85cfbe2e260907a52bbbe85942462377c8fdb62`
- rules_ml_toolchain: `73cb731fed3c9215033c5e21e64906f376cf47e8`
- abseil-cpp: `20260526.0`
- libtpu: `0.0.48`
- `RAIDEN_JAX`: `1101`

## Patches
- `patches/py/jax_remove_local_wheels.patch`: Custom patch for JAX 0.11.1 wheel dependencies.
- `patches/rules_ml_toolchain/no_register_toolchains.patch`: Removes duplicate toolchain registration.
