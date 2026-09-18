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
# CPU-only unit test and E2E JAX build verification for GitHub PRs and Copybara
# presubmits. Runs inside the ml-build container (glibc 2.35, the same image
# the wheel builds use): either as a GitHub Actions job step whose job declares
# that image as its container (see .github/workflows/unit_tests.yml), or by hand:
#
#   docker run --rm -v "${PWD}:/src" -w /src \
#     us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest \
#     ci/run_unit_tests.sh
#
# Requires root (installs clang-18 and libstdc++-12 through apt), like
# ci/build_wheel_impl.sh.
#
# Execution phases (when invoked without explicit target arguments):
#   Phase 1: Every cc_test under the package patterns in RAIDEN_TEST_SCOPE
#            that carries none of the exclusion tags (no_oss, notap, manual,
#            or any requires-<resource> tag).
#   Phase 2: E2E JAX validation build via `./build.sh jax`, compiling
#            _tpu_raiden_jax.so, C++ control-plane service binaries, and
#            Python protobuf modules, reusing the warm Bazel server and
#            Skyframe analysis cache from Phase 1.
#   Phase 3: Python 3.12 dynamic module binding linkage check (`import
#            kv_cache_manager` with JAX mocked on CPU) and pure-Python CPU
#            unit tests (`nd_slice_math_test.py`).
#
# Naming targets as arguments (e.g. `ci/run_unit_tests.sh //tpu_sync/core:foo_test`)
# overrides the query and skips Phases 2 & 3 unless RAIDEN_RUN_E2E_BUILD=true.
#
# Environment:
#   RAIDEN_TEST_SCOPE       space-separated bazel package patterns to search
#                           (default "//tpu_sync/core:all //tpu_sync/kv_cache:all")
#   RAIDEN_RUN_E2E_BUILD    true/false to run Phases 2 & 3 (default: true when
#                           no target args are passed, false otherwise)
#   BAZEL_CACHE_DIR         bazel disk/repo cache root (default /cache)
#   BAZEL_OUTPUT_BASE       bazel output base (default <cache>/output_base)
#   EXTRA_BAZEL_FLAGS       extra bazel flags, space-separated (optional; the
#                           workflow passes --config=ci for the remote cache when
#                           the run has credentials for it)
set -exu -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export BAZEL_CACHE_DIR="${BAZEL_CACHE_DIR:-${RUNNER_TEMP:-/cache}/bazel_cache}"
export BAZEL_OUTPUT_BASE="${BAZEL_OUTPUT_BASE:-${BAZEL_CACHE_DIR}/output_base}"
EXTRA_BAZEL_FLAGS="${EXTRA_BAZEL_FLAGS:-}"
export HERMETIC_PYTHON_VERSION="${HERMETIC_PYTHON_VERSION:-3.12}"
RAIDEN_TEST_SCOPE="${RAIDEN_TEST_SCOPE:-//tpu_sync/core:all //tpu_sync/kv_cache:all}"
mkdir -p "${BAZEL_CACHE_DIR}/disk_cache" "${BAZEL_CACHE_DIR}/repo_cache" "$(dirname "${BAZEL_OUTPUT_BASE}")"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq wget gnupg ca-certificates >/dev/null
# Add the LLVM jammy-18 apt repo manually (the container's add-apt-repository is
# broken: python apt_pkg is missing for python3.12) unless the image already
# carries it: apt refuses to read its sources when the same suite is listed
# twice under different keyrings.
if ! grep -rqs 'apt.llvm.org/jammy/ llvm-toolchain-jammy-18' /etc/apt/sources.list /etc/apt/sources.list.d/; then
  wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg
  echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" \
    > /etc/apt/sources.list.d/llvm18.list
fi
apt-get update -qq
# clang ships no standard library of its own: it compiles against the newest
# GCC installation present. The container carries only libstdc++ 11, which
# predates parts of the C++20 library this code uses, so 12 is installed
# alongside it and clang selects it automatically.
apt-get install -y -qq clang-18 libstdc++-12-dev >/dev/null
ln -sf /usr/bin/clang-18 /usr/bin/clang
ln -sf /usr/bin/clang++-18 /usr/bin/clang++
export CC=clang-18
export CXX=clang++-18
clang --version | head -1

BAZEL_VERSION="$(tr -d '\r\n ' < .bazelversion)"
# Use the exact same binary path and startup options as build.sh so both
# scripts share the downloaded binary and connect to the same running Bazel
# server without triggering a JVM server restart.
BAZEL_BIN="/tmp/bazel-bootstrap-${BAZEL_VERSION}"
if [[ ! -x "${BAZEL_BIN}" ]]; then
  wget -qO "${BAZEL_BIN}" \
    "https://storage.googleapis.com/bazel/${BAZEL_VERSION}/release/bazel-${BAZEL_VERSION}-linux-x86_64"
  chmod +x "${BAZEL_BIN}"
fi
"${BAZEL_BIN}" --version

BAZEL_STARTUP_FLAGS=(
  "--install_base=${BAZEL_OUTPUT_BASE}/install_base"
  "--output_base=${BAZEL_OUTPUT_BASE}"
  "--host_jvm_args=-Xmx32g"
  "--host_jvm_args=-Xms2g"
)
# Align repository environment flags across query, test, and build.sh (including
# --repo_env=CC=clang-18 from .bazelrc build:oss) so Skyframe never invalidates
# RepoEnvironmentFunction between commands.
BAZEL_REPO_ENV_FLAGS=(
  "--repo_env=CC=clang-18"
  "--repo_env=HERMETIC_PYTHON_VERSION=${HERMETIC_PYTHON_VERSION}"
  "--repo_env=PIP_INDEX_URL=https://pypi.org/simple"
  "--repo_env=PIP_EXTRA_INDEX_URL="
  "--repo_env=PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring"
  "--repo_env=PIP_CONFIG_FILE=/dev/null"
)
# Include --define=raiden_wheel_build=true and --define=with_torch=false to
# match `./build.sh jax` BuildOptions byte-for-byte, preventing Skyframe from
# discarding its in-memory analysis cache between Phase 1 and Phase 2.
BAZEL_COMMON_FLAGS=(
  "--config=oss"
  "${BAZEL_REPO_ENV_FLAGS[@]}"
  "--define=raiden_wheel_build=true"
  "--define=with_torch=false"
  "--disk_cache=${BAZEL_CACHE_DIR}/disk_cache"
  "--repository_cache=${BAZEL_CACHE_DIR}/repo_cache"
)
# The target query takes its own flags: .bazelrc defines the oss config under
# `build:`, and query inherits no build config, so `--config=oss` there is an
# error ("Config value 'oss' is not defined in any .rc file").
BAZEL_QUERY_FLAGS=(
  "--experimental_repo_remote_exec"
  "--override_module=torch_tpu=${REPO_ROOT}/shims/torch_tpu"
  "${BAZEL_REPO_ENV_FLAGS[@]}"
  "--repository_cache=${BAZEL_CACHE_DIR}/repo_cache"
)

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  RUN_E2E_DEFAULT="true"
  read -r -a SCOPE_PATTERNS <<< "${RAIDEN_TEST_SCOPE}"
  UNIVERSE="$(IFS='+'; echo "${SCOPE_PATTERNS[*]}")"
  QUERY="kind('cc_test rule', ${UNIVERSE}) except attr(tags, 'no_oss|notap|manual|requires-', ${UNIVERSE})"
  QUERY_OUTPUT="$(
    "${BAZEL_BIN}" "${BAZEL_STARTUP_FLAGS[@]}" query "${QUERY}" \
      "${BAZEL_QUERY_FLAGS[@]}" --noshow_progress
  )"
  if [[ -n "${QUERY_OUTPUT}" ]]; then
    mapfile -t TARGETS <<< "${QUERY_OUTPUT}"
  fi
  TARGETS+=("//tpu_sync/kv_cache:nd_slice_math_test")
else
  RUN_E2E_DEFAULT="false"
fi
RAIDEN_RUN_E2E_BUILD="${RAIDEN_RUN_E2E_BUILD:-${RUN_E2E_DEFAULT}}"

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "ERROR: no test targets selected from '${RAIDEN_TEST_SCOPE}'" >&2
  exit 1
fi

echo "=== Phase 1: Running Bazel CPU Unit Test Suite (${#TARGETS[@]} targets) ==="
printf '  %s\n' "${TARGETS[@]}"

# --keep_going: report every target that fails to build or test, rather than
# stopping at the first one.
if ! "${BAZEL_BIN}" "${BAZEL_STARTUP_FLAGS[@]}" test -c opt \
  "${BAZEL_COMMON_FLAGS[@]}" \
  --keep_going \
  --verbose_failures \
  --test_output=errors \
  --test_summary=detailed \
  ${EXTRA_BAZEL_FLAGS} \
  "${TARGETS[@]}"; then
  echo "=== Bazel CPU Unit Test Suite Failed: Extracting Test Logs ==="
  for xml_file in $(find -L "${REPO_ROOT}/bazel-testlogs" -name "test.xml" 2>/dev/null); do
    if grep -qE '<(failure|error)' "${xml_file}"; then
      log_file="${xml_file%/test.xml}/test.log"
      test_name="${xml_file#${REPO_ROOT}/bazel-testlogs/}"
      test_name="${test_name%/test.xml}"
      if [[ -f "${log_file}" ]]; then
        snippet="$(tail -n 40 "${log_file}" | tr '\n' ' ' | sed 's/::/:/g')"
        echo "::error title=${test_name}::${snippet}"
        {
          echo "### Failed Test: \`${test_name}\`"
          echo '```text'
          tail -n 100 "${log_file}"
          echo '```'
        } >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
      fi
    fi
  done
  exit 1
fi

if [[ "${RAIDEN_RUN_E2E_BUILD}" == "true" ]]; then
  echo "=== Phase 2: E2E JAX Validation Build (./build.sh jax) ==="
  ./build.sh jax --config=oss --remote_download_outputs=toplevel ${EXTRA_BAZEL_FLAGS}

  echo "=== Verifying E2E Build Artifacts ==="
  EXPECTED_ARTIFACTS=(
    "${REPO_ROOT}/tpu_sync/frameworks/jax/_tpu_raiden_jax.so"
    "${REPO_ROOT}/tpu_sync/kv_cache/global_registry/global_registry_server"
    "${REPO_ROOT}/tpu_sync/store_node/kv_cache_host_store_node_main"
  )
  for artifact in "${EXPECTED_ARTIFACTS[@]}"; do
    if [[ ! -e "${artifact}" ]]; then
      echo "ERROR: Expected build artifact missing: ${artifact}" >&2
      exit 1
    fi
  done

  echo "=== Phase 3: Verifying Dynamic Module Binding Linkage ==="
  export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/bazel-bin:${REPO_ROOT}/tpu_sync/api/jax:${REPO_ROOT}/tpu_sync/frameworks/jax:${PYTHONPATH:-}"
  HERMETIC_PYTHON_BIN="$(find "${BAZEL_OUTPUT_BASE}/external" -path "*/bin/python3" -executable 2>/dev/null | grep -E "python_3_12|rules_python" | head -n 1 || true)"
  if [[ -z "${HERMETIC_PYTHON_BIN}" ]] || ! "${HERMETIC_PYTHON_BIN}" --version &>/dev/null; then
    HERMETIC_PYTHON_BIN="python3"
  fi
  echo "Using Python interpreter: ${HERMETIC_PYTHON_BIN} ($(${HERMETIC_PYTHON_BIN} --version))"

  "${HERMETIC_PYTHON_BIN}" -c "
import sys
from unittest.mock import MagicMock
sys.modules['jax'] = MagicMock()
sys.modules['jax.core'] = MagicMock()
sys.modules['jax.extend'] = MagicMock()
sys.modules['jax.extend.ffi'] = MagicMock()

import kv_cache_manager
print('Dynamic linkage verified: kv_cache_manager (_tpu_raiden_jax.so) imported successfully!')
"
fi

echo "=== CI Verification Complete! ==="
