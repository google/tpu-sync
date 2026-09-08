#!/bin/bash

# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Runs `bazel test` over a set of raiden targets, reusing the same bazel binary,
# torch_tpu module override, and flags that build.sh uses. Used by the GitHub
# Actions CPU workflows so test selection stays in one place rather than
# duplicated across YAML.
#
# Usage:
#   ci/tools/bazel_test.sh <jax|torch|both> [extra_bazel_flags...] -- <target...>
#
# Env:
#   TORCH_TPU_MODULE_PATH   Path to a torch_tpu checkout (default ../torch_tpu).
#                           Only needed for `torch`/`both`; `jax` uses a dummy.
#   RAIDEN_REMOTE_CACHE     If set, used as --remote_cache (e.g.
#                           https://storage.googleapis.com/tpu-raiden-bazel-cache).
#   RAIDEN_REMOTE_UPLOAD    "true"/"false" -> --remote_upload_local_results
#                           (default false; presubmit should stay read-only).

set -exu -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:?usage: bazel_test.sh <jax|torch|both> [flags...] -- <target...>}"
shift

EXTRA_FLAGS=()
while [[ "$#" -gt 0 && "$1" != "--" ]]; do
  EXTRA_FLAGS+=("$1"); shift
done
[[ "${1:-}" == "--" ]] && shift
TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "ERROR: no test targets provided (expected '... -- //pkg:target ...')." >&2
  exit 1
fi

# Bootstrap the same standalone Bazel build.sh uses (honors .bazelversion).
BAZEL_VERSION="$(tr -d '\r\n ' < "${REPO_ROOT}/.bazelversion")"
BAZEL_BIN="/tmp/bazel-bootstrap-${BAZEL_VERSION}"
if [[ ! -x "${BAZEL_BIN}" ]]; then
  curl -Lo "${BAZEL_BIN}" \
    "https://storage.googleapis.com/bazel/${BAZEL_VERSION}/release/bazel-${BAZEL_VERSION}-linux-x86_64"
  chmod +x "${BAZEL_BIN}"
fi
"${BAZEL_BIN}" --version

# torch_tpu module override: a real checkout for torch builds, otherwise a
# generated dummy module so JAX-only graph resolution succeeds (mirrors
# build.sh and kokoro/gcp_ubuntu/presubmit.sh).
TORCH_TPU_MODULE_PATH="${TORCH_TPU_MODULE_PATH:-${REPO_ROOT}/../torch_tpu}"
DEFINE_FLAGS=()
if [[ "${MODE}" == "jax" ]]; then
  DUMMY_MODULE="$(mktemp -d)/dummy_torch_tpu_module"
  mkdir -p "${DUMMY_MODULE}"
  echo 'module(name = "torch_tpu", version = "0.1.1")' > "${DUMMY_MODULE}/MODULE.bazel"
  MODULE_OVERRIDE="--override_module=torch_tpu=${DUMMY_MODULE}"
  DEFINE_FLAGS+=(--define with_torch=false)
else
  if [[ ! -f "${TORCH_TPU_MODULE_PATH}/MODULE.bazel" ]]; then
    echo "ERROR: ${MODE} mode needs a torch_tpu checkout at ${TORCH_TPU_MODULE_PATH}." >&2
    exit 1
  fi
  TORCH_TPU_MODULE_PATH="$(cd "${TORCH_TPU_MODULE_PATH}" && pwd)"
  MODULE_OVERRIDE="--override_module=torch_tpu=${TORCH_TPU_MODULE_PATH}"
  TORCH_SOURCE="$(python3 -c 'import importlib.util,pathlib;s=importlib.util.find_spec("torch");print(pathlib.Path(next(iter(s.submodule_search_locations))).resolve().parent)')"
  export TORCH_SOURCE
  DEFINE_FLAGS+=(--define=TORCH_SOURCE=local --repo_env=TORCH_SOURCE="${TORCH_SOURCE}")
  [[ "${MODE}" == "torch" ]] && DEFINE_FLAGS+=(--define with_jax=false)
fi

REMOTE_FLAGS=()
if [[ -n "${RAIDEN_REMOTE_CACHE:-}" ]]; then
  REMOTE_FLAGS+=(
    "--remote_cache=${RAIDEN_REMOTE_CACHE}"
    "--google_default_credentials"
    "--remote_upload_local_results=${RAIDEN_REMOTE_UPLOAD:-false}"
    "--remote_max_connections=25"
    "--remote_timeout=300s"
    "--remote_retries=3"
  )
fi

"${BAZEL_BIN}" test -c opt --check_visibility=false --verbose_failures \
  --experimental_repo_remote_exec --incompatible_disallow_empty_glob=false \
  --repo_env=HERMETIC_PYTHON_VERSION="${HERMETIC_PYTHON_VERSION:-3.12}" \
  "${MODULE_OVERRIDE}" \
  "${DEFINE_FLAGS[@]}" \
  "${REMOTE_FLAGS[@]}" \
  "${EXTRA_FLAGS[@]}" \
  "${TARGETS[@]}"
