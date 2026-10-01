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
# Verifies that version-fragile JAX/XLA types remain confined to the compat layer.
#
# Usage: tools/jax/check_compat_boundaries.sh [--verbose]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${WORKSPACE_DIR}"

VERBOSE=false
[[ "${1:-}" == "--verbose" ]] && VERBOSE=true

# Compat layer files permitted to access fragile types and macros.
COMPAT_FILES=(
  "tpu_sync/core/xla_compat.h"
  "tpu_sync/core/xla_compat.cc"
  "tpu_sync/frameworks/jax/jax_compat.h"
  "tpu_sync/frameworks/jax/jax_compat.cc"
)

# Version-fragile XLA and jaxlib headers.
FRAGILE_HEADERS=(
  "xla/pjrt/raw_buffer.h"
  "xla/pjrt/abstract_tracked_device_buffer.h"
  "xla/pjrt/c_api_client/pjrt_c_api_client.h"
  "jaxlib/py_array.h"
)

# Searches C++ source files excluding third_party and build symlinks.
sources() {
  find . \( -name '*.h' -o -name '*.cc' \) \
    -not -path './bazel-*' \
    -not -path './third_party/*' \
    -not -path './.git/*' \
    -not -path './tpu_sync/internal/*' \
    -not -path './tpu_sync/google/*' \
    -not -path './tpu_sync/experimental/*' \
    | sed 's|^\./||' | sort
}

is_compat_file() {
  local f="$1" c
  for c in "${COMPAT_FILES[@]}"; do
    [[ "$f" == "$c" ]] && return 0
  done
  return 1
}

# Call sites subject to boundary enforcement.
call_sites() {
  local f
  while read -r f; do
    is_compat_file "$f" || echo "$f"
  done < <(sources)
}

FAILURES=0
report() {
  local rule="$1"
  shift
  echo "FAIL [${rule}]" >&2
  printf '  %s\n' "$@" >&2
  FAILURES=$((FAILURES + 1))
}

mapfile -t CALL_SITES < <(call_sites)
if [[ ${#CALL_SITES[@]} -eq 0 ]]; then
  echo "ERROR: found no sources to check; run this from the repository." >&2
  exit 2
fi

# Rule 1: Fragile headers are included only from the compat layer.
hits=()
for header in "${FRAGILE_HEADERS[@]}"; do
  while IFS= read -r line; do
    [[ -n "$line" ]] && hits+=("$line")
  done < <(grep -n -E "#include \"([^\"]*/)?${header}\"" "${CALL_SITES[@]}" 2>/dev/null)
done
if [[ ${#hits[@]} -gt 0 ]]; then
  report "fragile-include" \
    "These headers may be included only from the compat layer; reach them" \
    "through tpu_sync/core/xla_compat.h or tpu_sync/frameworks/jax/jax_compat.h:" \
    "${hits[@]}"
fi

# Rule 2: RAIDEN_JAX version macro is read only inside the compat layer.
mapfile -t macro_hits < <(
  grep -n -E '\bRAIDEN_JAX\b' "${CALL_SITES[@]}" 2>/dev/null |
    grep -v -E '^tools/jax/check_compat_boundaries\.sh:'
)
if [[ ${#macro_hits[@]} -gt 0 ]]; then
  report "version-macro" \
    "RAIDEN_JAX may be tested only inside the compat layer:" \
    "${macro_hits[@]}"
fi

# Rule 3: No call site directly names XLA's raw-buffer classes.
mapfile -t type_hits < <(
  grep -n -E 'xla::(PjRtRawBuffer|PjRtRawBufferInterface|CommonPjRtRawBuffer|PjRtRawBufferRef)\b' \
    "${CALL_SITES[@]}" 2>/dev/null
)
if [[ ${#type_hits[@]} -gt 0 ]]; then
  report "raw-buffer-type" \
    "Use raiden::RawBuffer / raiden::RawBufferRef instead of XLA's own names:" \
    "${type_hits[@]}"
fi

# Rule 4: No call site directly names hold or C-API client classes.
mapfile -t hold_hits < <(
  grep -n -E 'xla::(CommonPjRtBuffer|PjRtCApiBuffer|PjRtCApiClient)\b' \
    "${CALL_SITES[@]}" 2>/dev/null
)
if [[ ${#hold_hits[@]} -gt 0 ]]; then
  report "erased-type" \
    "Use raiden::ClassifyBuffer / raiden::AcquireCommonRawBuffer /" \
    "raiden::CreateCApiRawAlias instead of XLA's own names:" \
    "${hold_hits[@]}"
fi

if [[ ${FAILURES} -gt 0 ]]; then
  echo >&2
  echo "${FAILURES} compat-boundary rule(s) violated." >&2
  exit 1
fi

if [[ "${VERBOSE}" == true ]]; then
  echo "Checked ${#CALL_SITES[@]} sources against 4 compat-boundary rules."
fi
echo "compat boundaries OK"
