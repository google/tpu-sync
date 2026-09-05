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
# Runs C++ unit tests using the bootstrap Bazel toolchain and caches.
# Configures the build with --config=oss and host Clang required by XLA CPU codegen.
#
# Usage: tools/run_cc_tests.sh [extra bazel args...] [-- targets...]
# With no targets it runs every cc_test in the tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKSPACE_DIR}"

# Match build.sh so the two share their caches.
if [[ -d "/mnt/disks/persistent/${USER}" && -w "/mnt/disks/persistent/${USER}" ]] ||
  [[ -d /mnt/disks/persistent && -w /mnt/disks/persistent ]]; then
  DEFAULT_BAZEL_CACHE_BASE="/mnt/disks/persistent/${USER}/tpu-raiden-bazel-cache"
  DEFAULT_BAZEL_OUTPUT_BASE="/mnt/disks/persistent/${USER}/bazel-output-user-root/tpu_raiden_${USER}"
elif [[ -d "/mnt/disk/${USER}" && -w "/mnt/disk/${USER}" ]] || [[ -d /mnt/disk && -w /mnt/disk ]]; then
  DEFAULT_BAZEL_CACHE_BASE="/mnt/disk/${USER}/tpu-raiden-bazel-cache"
  DEFAULT_BAZEL_OUTPUT_BASE="/mnt/disk/${USER}/bazel-output-user-root/tpu_raiden_${USER}"
else
  DEFAULT_BAZEL_CACHE_BASE="${HOME}/.bazel_cache"
  DEFAULT_BAZEL_OUTPUT_BASE="/tmp/tpu_raiden_bazel_output_${USER}"
fi
BAZEL_CACHE_BASE="${BAZEL_CACHE_DIR:-${DEFAULT_BAZEL_CACHE_BASE}}"
BAZEL_OUTPUT_BASE="${BAZEL_OUTPUT_BASE:-${DEFAULT_BAZEL_OUTPUT_BASE}}"

BAZEL_VERSION="$(tr -d '[:space:]' < .bazelversion)"
BAZEL_BIN="/tmp/bazel-bootstrap-${BAZEL_VERSION}"
if [[ ! -x "${BAZEL_BIN}" ]]; then
  echo "Error: ${BAZEL_BIN} not found. Run ./build.sh once to bootstrap it." >&2
  exit 1
fi

# The C++ gate for the JAX version being tested, so the compat layer compiles
# the same branch the .so was built with.
DEFAULT_JAX_VERSION="$(tr -d '[:space:]' < third_party/jax/DEFAULT)"
# shellcheck source=tools/pins.sh
source "${SCRIPT_DIR}/pins.sh"
eval "$(raiden_read_pins "${WORKSPACE_DIR}" "${RAIDEN_JAX_VERSION:-${DEFAULT_JAX_VERSION}}")"

# --config=oss compiles with the host clang, which finds libstdc++ by guessing
# the newest gcc directory under /usr/lib/gcc. On a machine where that guess
# lands on a gcc whose libstdc++ headers are not installed -- Ubuntu leaves an
# empty gcc-14 directory when only libstdc++-13-dev is present -- every C++
# compile fails with "'algorithm' file not found". Point clang at a gcc that
# does have headers instead of requiring a machine change.
CLANG_FIX=()
CLANG_BIN="$(command -v clang-18 || true)"
if [[ -n "${CLANG_BIN}" ]] &&
  ! echo '#include <algorithm>' | "${CLANG_BIN}" -x c++ -fsyntax-only - 2> /dev/null; then
  for gcc_version in $(ls -1 /usr/include/c++ 2> /dev/null | sort -rV); do
    gcc_dir="/usr/lib/gcc/x86_64-linux-gnu/${gcc_version}"
    if [[ -d "${gcc_dir}" ]] &&
      echo '#include <algorithm>' |
      "${CLANG_BIN}" -x c++ "--gcc-install-dir=${gcc_dir}" -fsyntax-only - 2> /dev/null; then
      echo "Host clang cannot find libstdc++; using ${gcc_dir}."
      # The link driver is clang, not clang++, so it does not pull the C++
      # runtime in by itself; say so explicitly. Both the target and the exec
      # (`[for tool]`) configuration need every one of these -- llvm-tblgen is
      # built for the host and links C++ just as much as the test does.
      CLANG_FIX=(
        "--cxxopt=--gcc-install-dir=${gcc_dir}"
        "--host_cxxopt=--gcc-install-dir=${gcc_dir}"
        "--linkopt=--gcc-install-dir=${gcc_dir}"
        "--linkopt=-lstdc++"
        "--host_linkopt=--gcc-install-dir=${gcc_dir}"
        "--host_linkopt=-lstdc++"
      )
      break
    fi
  done
  if [[ ${#CLANG_FIX[@]} -eq 0 ]]; then
    echo "Error: clang-18 cannot compile C++ on this host and no usable gcc" \
      "install was found. Install libstdc++-<version>-dev." >&2
    exit 1
  fi
fi

ARGS=()
TARGETS=()
seen_separator=false
for arg in "$@"; do
  if [[ "${arg}" == "--" ]]; then
    seen_separator=true
    continue
  fi
  if [[ "${seen_separator}" == true ]]; then
    TARGETS+=("${arg}")
  else
    ARGS+=("${arg}")
  fi
done
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=("//...")
  ARGS+=("--build_tests_only")
fi

mkdir -p "${BAZEL_CACHE_BASE}/disk_cache" "${BAZEL_CACHE_BASE}/repo_cache"

set -x
exec "${BAZEL_BIN}" \
  --install_base="${BAZEL_OUTPUT_BASE}/install_base" \
  --output_base="${BAZEL_OUTPUT_BASE}" \
  --host_jvm_args="-Xmx32g" --host_jvm_args="-Xms2g" \
  test -c opt --config=oss \
  --define with_torch=false \
  --define "raiden_jax=${RAIDEN_PIN_RAIDEN_JAX}" \
  --repo_env=HERMETIC_PYTHON_VERSION="${HERMETIC_PYTHON_VERSION:-3.12}" \
  --disk_cache="${BAZEL_CACHE_BASE}/disk_cache" \
  --repository_cache="${BAZEL_CACHE_BASE}/repo_cache" \
  --test_output=errors \
  "${CLANG_FIX[@]+"${CLANG_FIX[@]}"}" \
  "${ARGS[@]}" \
  "${TARGETS[@]}"
