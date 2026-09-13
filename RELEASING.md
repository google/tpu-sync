# Releasing tpu_sync

## Versioning scheme

The base version is single-sourced from `[project].version` in
[pyproject.toml](pyproject.toml) (read by the Bazel repo rule in
[bazel/wheel_version.bzl](bazel/wheel_version.bzl)). A suffix from the
`WHEEL_VERSION_EXTRAS` environment variable is appended at build time:

| Channel | Version format            | Built by                                    | Published to |
|---------|---------------------------|---------------------------------------------|--------------|
| nightly | `X.Y.Z.devYYYYMMDDHHMMSS` | `Nightly Wheels` workflow (daily 08:00 UTC) | Artifact Registry |
| release | `X.Y.Z`                   | `Release Wheels` workflow (on `vX.Y.Z` tag) | PyPI + GitHub Release |

Two wheels are built per version, one per framework: `tpu_sync_jax` and
`tpu_sync_torch`. On PyPI they are the projects `tpu-sync-jax` and
`tpu-sync-torch`.

```bash
# JAX: self-contained, pulls the pinned jax/jaxlib/libtpu stack.
pip install tpu-sync-jax

# Torch: requires torch_tpu (and its matching torch) to be installed first,
# from the torch_tpu registry — see "The torch_tpu ABI pin" below for how the
# compatible torch_tpu version is recorded per release.
pip install --pre torch_tpu \
  --index-url "https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/"
pip install tpu-sync-torch
```

Nightlies live in the tpu-sync Artifact Registry instead (default
`https://us-python.pkg.dev/cloud-tpu-inference-test/tpu-raiden/`, overridable
via the `RAIDEN_REGISTRY_URL` repository variable); pip's pre-release rules
keep the channels apart, since `.dev` versions are only ever selected with
`--pre`:

```bash
pip install --pre tpu_sync_jax \
  --extra-index-url "https://us-python.pkg.dev/cloud-tpu-inference-test/tpu-raiden/simple/"
```

## How a release reaches PyPI

The `Release Wheels` workflow never talks to PyPI directly. It stages the
wheels in the project's OSS exit-gate Artifact Registry repository and writes
a publishing manifest that names exactly the packages and versions it built
to the exit-gate trigger bucket. The gate verifies the staged files,
publishes them to PyPI with its own credentials, removes them from the
staging repository on success, and emails the outcome (with logs) to the
project's notification address. There is no status file to poll: watch for
the two emails (one when the release is accepted, one when it finishes), then
check `https://pypi.org/project/tpu-sync-jax/` and
`https://pypi.org/project/tpu-sync-torch/`.

A failed or abandoned run leaves its files in the staging repository. The
workflow's manifest only ever lists the versions it built, so leftovers are
never published by a later run; a builder identity can delete them from the
repository if they get in the way.

### The PyPI file-size limit

PyPI accepts files up to 100 MB per file unless the project has been granted
a higher limit, and the exit gate uploads a release file by file, so a file
over the limit fails the release after the smaller files are already on PyPI
(and PyPI releases cannot be replaced). The workflow therefore refuses to
stage any wheel above the limit it knows about. Every build job prints the
wheel sizes in its summary; the torch wheel bundles one extension per
supported torch release and exceeds 100 MB when it carries more than one.

PyPI only raises a project's limit once the project exists with at least one
release under the current limit, so the first publication is:

1. Run `Release Wheels` by hand with `publish=no` and `torch_abis` set to a
   single release (`2.11.0`) and read the wheel sizes off the job summary;
   a wheel must be under the limit to be published (a single-ABI torch
   wheel is the smallest torch wheel the build makes). The `frameworks`
   input narrows the run to the wheels that fit.
2. Run it again with `publish=yes` and the same inputs. This publishes a
   `.dev` pre-release of each selected package, which creates its PyPI
   project.
3. File a file-size limit increase with PyPI for each project, following
   https://docs.pypi.org/project-management/storage-limits/. Approval can take
   weeks.
4. Once granted, record the new limit in the `PYPI_FILE_LIMIT_MB` repository
   variable. Tag-driven releases then publish the full multi-ABI torch wheel.

## The torch_tpu pairing

`tpu_sync_torch` is ABI-coupled to `torch_tpu`: the extension compiles
against torch_tpu's public tensor-buffer header and binds those symbols at
import to the installed torch_tpu wheel, and both must resolve the same
libtorch symbols. The build needs no torch_tpu checkout: the wheel-backed
module under [shims/torch_tpu](shims/torch_tpu) takes
the header and the XLA revision the extension must share with torch_tpu
from the installed torch_tpu wheel, which ships both under
`torch_tpu/include/`. The wheel ships one extension variant
per torch release in `RAIDEN_TORCH_ABIS` (see
[ci/build_wheel_impl.sh](ci/build_wheel_impl.sh)), and the loader picks the
variant matching the installed torch.

torch_tpu publishes nightlies only (`torch_tpu==0.1.1.devYYYYMMDDHHMMSS`), so
every release records the torch_tpu wheel it was validated with:

- [torch_tpu.version](torch_tpu.version) at the repo root names that
  torch_tpu wheel version; the build installs exactly that wheel, so it is
  the pairing at build time as well. Update it before tagging.
- The GitHub Release notes carry the same version, and the CHANGELOG entry
  lists it so users can install a compatible pair.

## The JAX stack pin

`tpu_sync_jax` is version-coupled to jax/jaxlib/libtpu, but unlike the torch
side this pin is enforced by pip itself: the exact versions are baked into
the wheel's `Requires-Dist` metadata (from `JAX_REQUIRES` in
[ci/wheel/BUILD.bazel](ci/wheel/BUILD.bazel), kept in lockstep with
[pyproject.toml](pyproject.toml)), so installing the wheel installs the
matching stack. For visibility, both workflows also extract these pins from
the built wheel into a `jax_pins.txt` beside it, and the GitHub Release notes
list them next to the torch_tpu pin. Bumping the JAX stack for a release
means updating `JAX_REQUIRES` and `pyproject.toml` together in the release
PR.

## Cutting a release

1. Pick the release candidate commit on `main` — normally the last-known-good
   commit already published in [lkg.version](lkg.version).
2. Open a release PR that:
   - updates [torch_tpu.version](torch_tpu.version) to the torch_tpu wheel
     the release was validated with,
   - bumps `[project].version` in `pyproject.toml` to `X.Y.Z`,
   - adds the `X.Y.Z` section to [CHANGELOG.md](CHANGELOG.md), including the
     compatible `torch_tpu` nightly version.
3. (Optional) Dry run: trigger the `Release Wheels` workflow via
   `workflow_dispatch` with `publish=no` — it builds `.dev`-versioned wheels
   as workflow artifacts without publishing anything.
4. Merge the PR, then tag and push:

   ```bash
   git tag vX.Y.Z <merge-commit>
   git push origin vX.Y.Z
   ```

   The `Release Wheels` workflow verifies the tag (must equal the pyproject
   version and be reachable from `main`), builds both wheels with an empty
   version suffix, stages them and triggers the exit gate, and creates the
   GitHub Release with the wheels attached.
5. Wait for the exit-gate emails and confirm both projects on pypi.org show
   `X.Y.Z`.
6. Bump `[project].version` on `main` to the next patch version
   `X.Y.(Z+1)`. This keeps nightlies (`X.Y.(Z+1).devN`) sorting *above* the
   just-released `X.Y.Z`, so `--pre` users keep receiving fresh builds.

## Prerequisites (one-time repo setup)

- The GitHub Actions runner service account needs
  `roles/artifactregistry.writer` on the nightly registry (or set
  `RAIDEN_REGISTRY_URL` to a registry it can write to), and must be registered
  with the OSS exit gate as a builder and triggerer for this project, which
  grants it write access to the staging repository and the manifest folder.
- `PYPI_FILE_LIMIT_MB` repository variable: the per-file limit PyPI has
  granted the project, once it differs from the default 100.
