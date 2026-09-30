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
# Reads JAX version dependency revisions from third_party/jax/<version>/deps.bzl.
# Source this file; do not run it directly.

# Directory of one version: raiden_jax_deps_dir <workspace> <version>
raiden_jax_deps_dir() {
  echo "${1}/third_party/jax/${2}"
}

# Reads a variable out of a .bzl file: raiden_jax_read_bzl_value <bzl_path> <var_name>
raiden_jax_read_bzl_value() {
  python3 - "$1" "$2" <<'PY'
import sys

namespace = {}
with open(sys.argv[1]) as f:
    exec(compile(f.read(), sys.argv[1], "exec"), {}, namespace)  # noqa: S102
value = namespace.get(sys.argv[2])
if value is None:
    raise SystemExit("%s does not define %s" % (sys.argv[1], sys.argv[2]))
if isinstance(value, (list, tuple)):
    print("\n".join(str(item) for item in value))
else:
    print(value)
PY
}

# Supported versions listed in versions.bzl.
raiden_jax_supported_versions() {
  raiden_jax_read_bzl_value "${1}/third_party/jax/versions.bzl" SUPPORTED_VERSIONS
}

# Default version from versions.bzl.
raiden_jax_default_version() {
  raiden_jax_read_bzl_value "${1}/third_party/jax/versions.bzl" DEFAULT_VERSION
}

# Existing version directories under third_party/jax/.
raiden_jax_entry_versions() {
  local workspace="$1" d
  for d in "${workspace}"/third_party/jax/*/deps.bzl; do
    [[ -f "$d" ]] || continue
    basename "$(dirname "$d")"
  done | sort -V
}

# Validates that <version> is in SUPPORTED_VERSIONS.
raiden_jax_validate_version() {
  local workspace="$1" version="$2"
  if raiden_jax_supported_versions "${workspace}" | grep -qxF "${version}"; then
    return 0
  fi
  echo "Error: RAIDEN_JAX_VERSION='${version}' is not a supported JAX version." >&2
  echo "Supported: $(raiden_jax_supported_versions "${workspace}" | tr '\n' ' ')" >&2
  echo "See third_party/jax/README.md for how to add one." >&2
  return 1
}

# Renders requirements.txt for <version> to <outfile> by replacing jax, jaxlib, and libtpu.
raiden_jax_render_requirements() {
  local workspace="$1" version="$2" outfile="$3"
  eval "$(raiden_jax_read_deps "${workspace}" "${version}")" || return 1

  local missing=()
  [[ -n "${RAIDEN_JAX_DEP_JAX_VERSION:-}" ]] || missing+=("jax_version")
  [[ -n "${RAIDEN_JAX_DEP_JAXLIB_VERSION:-}" ]] || missing+=("jaxlib_version")
  [[ -n "${RAIDEN_JAX_DEP_LIBTPU_VERSION:-}" ]] || missing+=("libtpu_version")
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Error: third_party/jax/${version}/deps.bzl states no ${missing[*]}." >&2
    return 1
  fi

  sed -E \
      -e "s#^jax(==|>=).*#jax==${RAIDEN_JAX_DEP_JAX_VERSION}#" \
      -e "s#^jaxlib(==|>=).*#jaxlib==${RAIDEN_JAX_DEP_JAXLIB_VERSION}#" \
      -e "s#^libtpu(==|>=).*#libtpu==${RAIDEN_JAX_DEP_LIBTPU_VERSION}#" \
      "${workspace}/requirements.txt" > "${outfile}"
}

# Reads all keys from deps.bzl into RAIDEN_JAX_DEP_<KEY> variables.
raiden_jax_read_deps() {
  local workspace="$1" version="$2"
  local deps_file="$(raiden_jax_deps_dir "${workspace}" "${version}")/deps.bzl"
  if [[ ! -f "${deps_file}" ]]; then
    echo "Error: no deps for JAX ${version} (${deps_file} not found)." >&2
    echo "On disk: $(raiden_jax_entry_versions "${workspace}" | tr '\n' ' ')" >&2
    return 1
  fi
  python3 - "${deps_file}" <<'PY'
import shlex
import sys

namespace = {}
with open(sys.argv[1]) as f:
    exec(compile(f.read(), sys.argv[1], "exec"), {}, namespace)  # noqa: S102
deps = namespace.get("DEPS")
if not isinstance(deps, dict):
    raise SystemExit("%s does not define a DEPS dict" % sys.argv[1])
for key, value in deps.items():
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value)
    print("RAIDEN_JAX_DEP_%s=%s" % (key.upper(), shlex.quote(str(value))))
PY
}
