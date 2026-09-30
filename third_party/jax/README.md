# JAX Versions

TPU Raiden builds against a selectable JAX version:

```bash
RAIDEN_JAX_VERSION=0.11.2 ./build.sh jax  # defaults to DEFAULT_VERSION in versions.bzl
```

Each subdirectory corresponds to a supported version containing its dependency pins:

| File in `third_party/jax/<version>/` | Description |
| :--- | :--- |
| `deps.bzl` | Upstream revisions for JAX, jaxlib, XLA, rules_ml_toolchain, abseil, libtpu, and the `RAIDEN_JAX` macro. |
| `README.md` | Version-specific notes and patch rationales (optional). |
| `patches/` | Version-specific patches (optional). |

Root directory files:
- `versions.bzl`: Defines `DEFAULT_VERSION` and `SUPPORTED_VERSIONS`.
- `BUILD`: Defines `:raiden_jax_version` setting the C++ `-DRAIDEN_JAX` flag.

## Python Environment

`build.sh` replaces three dependency pins in `requirements.txt` using `deps.bzl`:
- `jax==<version>` -> `jax_version`
- `jaxlib==<version>` -> `jaxlib_version`
- `libtpu==<version>` -> `libtpu_version`

`tools/jax/jax_deps.sh` renders these substitutions into a temporary file for `pip install -r`. The root `requirements.txt` remains unchanged.

## Dependency Co-evolution

Pinned revisions must match upstream JAX's `MODULE.bazel` exactly:
- **jax**: Python wheel.
- **xla**: PJRT C++ headers.
- **rules_ml_toolchain**: C++ toolchain for XLA rules.
- **abseil**: Must match `jaxlib`'s Abseil version to prevent runtime ABI mismatches.
- **libtpu**: Runtime PJRT plugin wheel.

## Adding a Version

1. Read `xla`, `rules_ml_toolchain`, and `abseil-cpp` revisions from upstream JAX `MODULE.bazel`.
2. Create `third_party/jax/<version>/deps.bzl`. Set `raiden_jax = major*10000 + minor*100 + patch`.
3. Add any version-specific patches under `third_party/jax/<version>/patches/`.
4. Add a `config_setting` and `select` branch in `third_party/jax/BUILD`.
5. Add the version to `SUPPORTED_VERSIONS` in `third_party/jax/versions.bzl`.
6. Run tests:
   ```bash
   RAIDEN_JAX_VERSION=<version> ./build.sh jax
   RAIDEN_JAX_VERSION=<version> ./tools/run_cc_tests.sh -- //tpu_sync/frameworks/jax:raw_transfer_test
   ./run_tests.sh jax
   ```
7. Confine all version-specific C++ adaptations to `xla_compat.*` and `jax_compat.*`.
