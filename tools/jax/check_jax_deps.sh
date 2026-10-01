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
# Verifies consistency of JAX version dependency pins, allowlists, and requirements.
#
# Usage: tools/jax/check_jax_deps.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${WORKSPACE_DIR}"

# shellcheck source=tools/jax/jax_deps.sh
source "${SCRIPT_DIR}/jax_deps.sh"

DEFAULT_VERSION="$(raiden_jax_default_version "${WORKSPACE_DIR}")"

FAILURES=0
fail() {
  echo "FAIL: $*" >&2
  FAILURES=$((FAILURES + 1))
}

# --- Validate allowlist against directories on disk ---------------------------

mapfile -t VERSIONS < <(raiden_jax_entry_versions "${WORKSPACE_DIR}")
mapfile -t SUPPORTED < <(raiden_jax_supported_versions "${WORKSPACE_DIR}")
if [[ ${#VERSIONS[@]} -eq 0 ]]; then
  echo "ERROR: no versions under third_party/jax/." >&2
  exit 2
fi

for version in "${SUPPORTED[@]}"; do
  [[ -f "$(raiden_jax_deps_dir "${WORKSPACE_DIR}" "${version}")/deps.bzl" ]] ||
    fail "versions.bzl lists ${version}, but third_party/jax/${version}/deps.bzl does not exist"
done
for version in "${VERSIONS[@]}"; do
  printf '%s\n' "${SUPPORTED[@]}" | grep -qxF "${version}" ||
    fail "third_party/jax/${version}/ exists but versions.bzl does not list it, so nothing can select it"
done

printf '%s\n' "${SUPPORTED[@]}" | grep -qxF "${DEFAULT_VERSION}" ||
  fail "versions.bzl sets DEFAULT_VERSION to ${DEFAULT_VERSION}, which its own SUPPORTED_VERSIONS does not list"

# --- Validate self-consistency for each version -------------------------------

for version in "${VERSIONS[@]}"; do
  deps_dir="$(raiden_jax_deps_dir "${WORKSPACE_DIR}" "${version}")"
  eval "$(raiden_jax_read_deps "${WORKSPACE_DIR}" "${version}")" || exit 2

  [[ "${RAIDEN_JAX_DEP_JAX_VERSION}" == "${version}" ]] ||
    fail "${version}: deps.bzl says jax_version=${RAIDEN_JAX_DEP_JAX_VERSION}"

  expected_macro="$(python3 -c '
import sys
parts = (sys.argv[1].split(".") + ["0", "0", "0"])[:3]
print(int(parts[0]) * 10000 + int(parts[1]) * 100 + int(parts[2]))' "${version}")"
  [[ "${RAIDEN_JAX_DEP_RAIDEN_JAX}" == "${expected_macro}" ]] ||
    fail "${version}: raiden_jax=${RAIDEN_JAX_DEP_RAIDEN_JAX}, expected ${expected_macro}"

  if [[ -z "${RAIDEN_JAX_DEP_JAXLIB_VERSION:-}" ]]; then
    fail "${version}: deps.bzl states no jaxlib_version"
  elif [[ "${RAIDEN_JAX_DEP_JAXLIB_VERSION}" != "${version}" ]]; then
    fail "${version}: jaxlib_version is ${RAIDEN_JAX_DEP_JAXLIB_VERSION}, which no version has needed yet -- if that is deliberate, record why in README.md and relax this check"
  fi

  for patch in ${RAIDEN_JAX_DEP_JAX_PATCHES} ${RAIDEN_JAX_DEP_XLA_PATCHES} \
    ${RAIDEN_JAX_DEP_RULES_ML_TOOLCHAIN_PATCHES} ${RAIDEN_JAX_DEP_ABSL_PATCHES}; do
    [[ -f "${WORKSPACE_DIR}/${patch}" ]] ||
      fail "${version}: deps.bzl names a patch that does not exist: ${patch}"
  done

  rendered="$(mktemp "${TMPDIR:-/tmp}/raiden_req_check.XXXXXX")"
  if raiden_jax_render_requirements "${WORKSPACE_DIR}" "${version}" "${rendered}"; then
    for spec in "jax==${version}" "jaxlib==${RAIDEN_JAX_DEP_JAXLIB_VERSION}" \
      "libtpu==${RAIDEN_JAX_DEP_LIBTPU_VERSION}"; do
      grep -qxF "${spec}" "${rendered}" ||
        fail "${version}: rendered requirements do not state ${spec}"
    done
  else
    fail "${version}: could not render requirements"
  fi
  rm -f "${rendered}"
done

# --- Validate default version matches MODULE.bazel and requirements.txt -------

eval "$(raiden_jax_read_deps "${WORKSPACE_DIR}" "${DEFAULT_VERSION}")" || exit 2

module_has() {
  grep -qF "$1" MODULE.bazel
}

module_has "commit = \"${RAIDEN_JAX_DEP_JAX_COMMIT}\"" ||
  fail "MODULE.bazel does not pin jax at ${RAIDEN_JAX_DEP_JAX_COMMIT} (default version ${DEFAULT_VERSION})"
module_has "commit = \"${RAIDEN_JAX_DEP_XLA_COMMIT}\"" ||
  fail "MODULE.bazel does not pin xla at ${RAIDEN_JAX_DEP_XLA_COMMIT}"
module_has "rules_ml_toolchain-${RAIDEN_JAX_DEP_RULES_ML_TOOLCHAIN_COMMIT}" ||
  fail "MODULE.bazel does not pin rules_ml_toolchain at ${RAIDEN_JAX_DEP_RULES_ML_TOOLCHAIN_COMMIT}"
module_has "integrity = \"${RAIDEN_JAX_DEP_RULES_ML_TOOLCHAIN_INTEGRITY}\"" ||
  fail "MODULE.bazel's rules_ml_toolchain integrity does not match deps.bzl"
module_has "name = \"abseil-cpp\", version = \"${RAIDEN_JAX_DEP_ABSL_VERSION}\"" ||
  fail "MODULE.bazel does not pin abseil-cpp at ${RAIDEN_JAX_DEP_ABSL_VERSION}"

grep -qE '^jax>=0\.11\.0' requirements.txt ||
  fail "requirements.txt does not satisfy floor bound jax>=0.11.0"
grep -qE '^jaxlib>=0\.11\.0' requirements.txt ||
  fail "requirements.txt does not satisfy floor bound jaxlib>=0.11.0"
grep -qE '^libtpu>=0\.0\.47' requirements.txt ||
  fail "requirements.txt does not satisfy floor bound libtpu>=0.0.47"

if [[ ${FAILURES} -gt 0 ]]; then
  echo >&2
  echo "${FAILURES} dependency consistency check(s) failed." >&2
  exit 1
fi

echo "jax deps OK (default ${DEFAULT_VERSION}; versions: ${VERSIONS[*]})"
