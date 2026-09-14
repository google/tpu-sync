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
# CPU-only unit test run for GitHub PR verification. Runs inside the ml-build
# container (glibc 2.35, the same image the wheel builds use): either as a
# GitHub Actions job step whose job declares that image as its container (see
# .github/workflows/unit_tests.yml), or by hand:
#
#   docker run --rm -v "${PWD}:/src" -w /src \
#     us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest \
#     ci/run_unit_tests.sh
#
# Requires root (installs clang-18 and libstdc++-12 through apt), like
# ci/build_wheel_impl.sh.
#
# Test selection: every cc_test under the package patterns in
# RAIDEN_TEST_SCOPE that carries none of the exclusion tags (no_oss, notap,
# manual, or any requires-<resource> tag). Tagging a target no_oss therefore
# keeps it out of GitHub CI, and the TPU-bound tests (requires-ghostfish) stay
# out on their own, since these runners have no TPU. New tests are picked up
# automatically -- no list to update here. Naming targets as arguments
# overrides the query entirely.
#
# Environment:
#   RAIDEN_TEST_SCOPE   space-separated bazel package patterns to search
#                       (default "//tpu_sync/core:all //tpu_sync/kv_cache:all")
#   BAZEL_CACHE_DIR     bazel disk/repo cache root (default /cache)
#   BAZEL_OUTPUT_BASE   bazel output base (default <cache>/output_base)
#   EXTRA_BAZEL_FLAGS   extra bazel flags, space-separated (optional; the
#                       workflow passes --config=ci for the remote cache when
#                       the run has credentials for it)
set -exu -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BAZEL_CACHE_DIR="${BAZEL_CACHE_DIR:-/cache}"
BAZEL_OUTPUT_BASE="${BAZEL_OUTPUT_BASE:-${BAZEL_CACHE_DIR}/output_base}"
EXTRA_BAZEL_FLAGS="${EXTRA_BAZEL_FLAGS:-}"
HERMETIC_PYTHON_VERSION="${HERMETIC_PYTHON_VERSION:-3.12}"
RAIDEN_TEST_SCOPE="${RAIDEN_TEST_SCOPE:-//tpu_sync/core:all //tpu_sync/kv_cache:all}"
mkdir -p "${BAZEL_CACHE_DIR}/disk_cache" "${BAZEL_CACHE_DIR}/repo_cache"

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
clang --version | head -1

BAZEL_VERSION="$(tr -d '\r\n ' < .bazelversion)"
BAZEL_BIN="/tmp/bazel-${BAZEL_VERSION}"
if [[ ! -x "${BAZEL_BIN}" ]]; then
  wget -qO "${BAZEL_BIN}" \
    "https://storage.googleapis.com/bazel/${BAZEL_VERSION}/release/bazel-${BAZEL_VERSION}-linux-x86_64"
  chmod +x "${BAZEL_BIN}"
fi
"${BAZEL_BIN}" --version

BAZEL_STARTUP_FLAGS=(
  "--output_base=${BAZEL_OUTPUT_BASE}"
)
BAZEL_COMMON_FLAGS=(
  "--config=oss"
  "--repo_env=HERMETIC_PYTHON_VERSION=${HERMETIC_PYTHON_VERSION}"
  "--disk_cache=${BAZEL_CACHE_DIR}/disk_cache"
  "--repository_cache=${BAZEL_CACHE_DIR}/repo_cache"
)
# The target query takes its own flags: .bazelrc defines the oss config under
# `build:`, and query inherits no build config, so `--config=oss` there is an
# error ("Config value 'oss' is not defined in any .rc file"). A query runs no
# action either, which leaves the disk cache and the compiler settings nothing
# to do. What remains is what the loading phase needs: the torch_tpu override,
# without which module resolution -- which precedes every command, query
# included -- fails, as the root module depends on torch_tpu and no registry
# carries it; it lives in shims/ (the oss config points there the same way).
BAZEL_QUERY_FLAGS=(
  "--experimental_repo_remote_exec"
  "--override_module=torch_tpu=${REPO_ROOT}/shims/torch_tpu"
  "--repo_env=HERMETIC_PYTHON_VERSION=${HERMETIC_PYTHON_VERSION}"
  "--repository_cache=${BAZEL_CACHE_DIR}/repo_cache"
)

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  read -r -a SCOPE_PATTERNS <<< "${RAIDEN_TEST_SCOPE}"
  UNIVERSE="$(IFS='+'; echo "${SCOPE_PATTERNS[*]}")"
  QUERY="kind('cc_test rule', ${UNIVERSE}) except attr(tags, 'no_oss|notap|manual|requires-', ${UNIVERSE})"
  # Read the query into a variable rather than piping it straight into
  # mapfile: a command substitution fails the script under `set -e`, whereas
  # `mapfile < <(...)` reports mapfile's own status and a broken query would
  # reach the emptiness check below disguised as "no tests match".
  QUERY_OUTPUT="$(
    "${BAZEL_BIN}" "${BAZEL_STARTUP_FLAGS[@]}" query "${QUERY}" \
      "${BAZEL_QUERY_FLAGS[@]}" --noshow_progress
  )"
  if [[ -n "${QUERY_OUTPUT}" ]]; then
    mapfile -t TARGETS <<< "${QUERY_OUTPUT}"
  fi
fi
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "ERROR: no test targets selected from '${RAIDEN_TEST_SCOPE}'" >&2
  exit 1
fi

printf '===> testing %d target(s):\n' "${#TARGETS[@]}"
printf '  %s\n' "${TARGETS[@]}"

# --keep_going: report every target that fails to build or test, rather than
# stopping at the first one. A contributor reading the pull request gets the
# whole list in one run instead of one failure per push.
"${BAZEL_BIN}" "${BAZEL_STARTUP_FLAGS[@]}" test -c opt \
  "${BAZEL_COMMON_FLAGS[@]}" \
  --keep_going \
  --verbose_failures \
  --test_output=errors \
  --test_summary=detailed \
  ${EXTRA_BAZEL_FLAGS} \
  "${TARGETS[@]}"
