# tpu-raiden CI/CD

GitHub Actions CI for tpu-raiden, modeled on
[`torch_tpu`](https://github.com/google-pytorch/torch_tpu)'s setup. The existing
Kokoro jobs under [`kokoro/`](../kokoro) are kept and run alongside these.

## Runners

- **CPU** — `linux-x86-n2-32` runs the presubmit + nightly build/unit-test jobs.
- **TPU v5e** — `linux-x86-ct5lp-224-8tpu` runs the device tests (`device_test.yml`).
  v5e is used because the v7 (`linux-x86-tpu7x-224-4tpu`) scale-set is not
  available to this repo.

Both are self-hosted scale-sets and can take several minutes to cold-start a
runner, so a freshly triggered job may sit `queued` for a while before a runner
picks it up — that is normal, not a failure. CPU and TPU jobs are split across
separate workflows so a slow TPU runner never blocks the CPU gate.

## Workflows

| Workflow | Trigger | What it does |
| --- | --- | --- |
| [`lint.yml`](workflows/lint.yml) | PR | Apache license headers (`addlicense`) + commit-message check. GitHub-hosted. |
| [`clang_format.yml`](workflows/clang_format.yml) | PR | Google C++ style on changed `.cc/.h` via pinned `clang-format==18`. GitHub-hosted. |
| [`presubmit.yml`](workflows/presubmit.yml) | PR | Build JAX extension + OSS-safe CPU unit tests + import smoke, on `n2-32`. |
| [`nightly.yml`](workflows/nightly.yml) | push to `main` / daily cron 09:00 UTC / dispatch | CPU build + unit tests + `tpu_raiden_jax` wheel (twine-checked, uploaded as an artifact), file a tracking issue on failure, on `n2-32`. |
| [`device_test.yml`](workflows/device_test.yml) | PR / daily cron / dispatch | Build JAX extension, verify TPU visible, run device tests on v5e (`ct5lp`). |

Shared logic lives in [`ci/tools/`](../ci/tools): `install_clang18.sh` (the
clang-18 the build needs) and `bazel_test.sh` (bazel binary + dummy torch_tpu
override + flags, matching `build.sh`).

## Scope notes

- **JAX-only build.** The JAX path generates a dummy `torch_tpu` Bazel module
  (same as `build.sh` / `kokoro/.../presubmit.sh`), so no secrets are needed. The
  **Torch** extension/wheel additionally needs a `torch_tpu` checkout + a local
  `torch` (deploy key) — out of scope here for now.
- **Test set.** `CPU_TEST_TARGETS` are device-free; `DEVICE_TEST_TARGETS` exercise
  the v5e. Both are a curated OSS-loadable set. `//rpc/...` and
  `//kv_cache/global_registry/...` (and the `kv_cache_store` tests that depend on
  them) load Google-internal gRPC rules and are excluded by design — see
  [`ci/wheel/BUILD.bazel`](../ci/wheel/BUILD.bazel). Expand the lists as coverage
  grows; for multiple parallel device tests, add a `--run_under` accelerator-lock
  helper.

## One-time setup to turn this on

1. **Runner** — `google/tpu-raiden` must be granted the
   `linux-x86-ct5lp-224-8tpu` scale-set (confirmed picking up jobs).
2. **Bazel remote cache** — bucket `gs://tpu-raiden-bazel-cache` (already used by
   Kokoro); the runner service account needs object read (and write for the
   nightly's read-write cache).
3. **Labels** — create `ci:nightly-failed` for the failure-issue action.
   `GITHUB_TOKEN` is provided automatically.

## Landing & running

`google/tpu-raiden` is a one-way Copybara mirror of an internal google3 repo
(commits carry `PiperOrigin-RevId:`), so these files land **internally** and sync
out — there is no external push/PR path for merges. Opening a PR still triggers
the workflows for validation. Once on `main`, presubmit fires on PRs and
`nightly.yml` can be kicked with `gh workflow run "CI - Nightly"` by anyone with
write access.
