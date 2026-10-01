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
# Computes Bazel module overrides for the selected JAX version: downloads each
# pinned module into the cache, applies its patches, and emits --override_module.
# Sourced by build.sh and tools/run_cc_tests.sh.

# shellcheck source=tools/jax/jax_deps.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/jax_deps.sh"

download_file() {
  local url="$1"
  local dest="$2"
  if command -v curl > /dev/null; then
    curl -Lo "$dest" "$url"
  elif command -v wget > /dev/null; then
    wget -O "$dest" "$url"
  elif command -v python3 > /dev/null; then
    python3 -c "import urllib.request; urllib.request.urlretrieve('$url', '$dest')"
  elif command -v python > /dev/null; then
    python -c "import urllib; urllib.urlretrieve('$url', '$dest')"
  else
    echo "Error: No download tool found (curl, wget, python3, python)." >&2
    return 1
  fi
}

# Downloads <repo>/archive/<ref>.tar.gz into bazel cache and applies patches.
#
#   materialize_module <cache_base> <repo_url> <ref> [patch_file...]
materialize_module() {
  local cache_base="$1" repo="$2" ref="$3"
  shift 3
  local name; name="$(basename "${repo}")"
  local patch_key="nopatch"
  if [[ "$#" -gt 0 ]]; then
    patch_key="$(cat "$@" | sha256sum | cut -c1-16)"
  fi
  MATERIALIZED_DIR="${cache_base}/pinned_modules/${name}/${ref}-${patch_key}"
  if [[ ! -f "${MATERIALIZED_DIR}/MODULE.bazel" ]]; then
    echo "Materializing ${name} @ ${ref} ($# patch(es))..." >&2
    mkdir -p "${cache_base}"
    local stage; stage="$(mktemp -d "${cache_base}/${name}_stage.XXXXXX")"
    download_file "${repo}/archive/${ref}.tar.gz" "${stage}/src.tar.gz" >&2
    tar -xzf "${stage}/src.tar.gz" -C "${stage}"
    local p
    for p in "$@"; do
      if [[ ! -f "$p" ]]; then
        echo "Error: materialize_module: patch file not found: '${p}'" >&2
        return 1
      fi
      patch -p1 -s -d "${stage}/${name}-${ref}" < "$p" >&2 || {
        echo "Error: failed to apply patch '${p}' to ${name}" >&2
        return 1
      }
    done
    mkdir -p "$(dirname "${MATERIALIZED_DIR}")"
    rm -rf "${MATERIALIZED_DIR}"
    mv "${stage}/${name}-${ref}" "${MATERIALIZED_DIR}"
    rm -rf "${stage}"
  fi
}

# Prints one --override_module flag per line for modules moving with JAX.
#
#   raiden_jax_module_overrides <workspace> <version> <bazel_cache_base>
raiden_jax_module_overrides() {
  local workspace="$1" version="$2" cache_base="$3"
  eval "$(raiden_jax_read_deps "${workspace}" "${version}")" || return 1

  local module patches_var patch_list
  for module in jax:JAX xla:XLA rules_ml_toolchain:RULES_ML_TOOLCHAIN \
    abseil-cpp:ABSL; do
    local bazel_name="${module%%:*}" key="${module##*:}"
    local repo_var="RAIDEN_JAX_DEP_${key}_REPO"
    local patches_key="RAIDEN_JAX_DEP_${key}_PATCHES"
    local ref_var="RAIDEN_JAX_DEP_${key}_COMMIT"
    [[ "${key}" == "ABSL" ]] && ref_var="RAIDEN_JAX_DEP_${key}_VERSION"

    patch_list=()
    for patches_var in ${!patches_key}; do
      patch_list+=("${workspace}/${patches_var}")
    done

    materialize_module "${cache_base}" "${!repo_var}" "${!ref_var}" \
      "${patch_list[@]+"${patch_list[@]}"}" || return 1
    echo "--override_module=${bazel_name}=${MATERIALIZED_DIR}"
  done
}
