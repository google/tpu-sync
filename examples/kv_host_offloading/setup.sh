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

# One-click environment setup for the KV host-offloading example.
#
# Clones vLLM and tpu-inference (which ships the RaidenOffloadConnector) into a
# hidden, in-tree `.src/` dir and installs them editable into the CURRENTLY
# ACTIVE venv. Self-contained: does not depend on any other example's setup.
#
# Prerequisites:
#   1. A python3.12 venv created and ACTIVATED.
#   2. tpu_raiden already installed into that venv via EITHER supported path:
#        - build from source: `./build.sh jax` from the repo root, or
#        - wheel:             `pip install --pre tpu-sync-jax`
# Then, from this directory:
#   bash setup.sh
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIDEN_ROOT="${RAIDEN_ROOT:-$(cd "${SCRIPTS_DIR}/../.." && pwd)}"
SRC_DIR="${SRC_DIR:-${SCRIPTS_DIR}/.src}"

# Pinned sources. The RaidenOffloadConnector is on tpu-inference main; these
# pinned commits include it. vLLM's pin is coupled with tpu-inference (see
# tpu-inference/.buildkite/vllm_lkg.version), so keep the two in lockstep if you
# bump either.
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_COMMIT="${VLLM_COMMIT:-239fc7355361ee60ea10756c10944041d8e38ded}"
TPU_INFERENCE_REPO="${TPU_INFERENCE_REPO:-https://github.com/vllm-project/tpu-inference.git}"
TPU_INFERENCE_COMMIT="${TPU_INFERENCE_COMMIT:-24c556240b044b52071b21674e82fc9fb068daf4}"

OFFLOAD_CONNECTOR_REL="tpu_inference/offload/raiden_offload_connector.py"

# --- Preconditions --------------------------------------------------------
if [ -z "${VIRTUAL_ENV:-}" ] && ! python -c "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)" 2>/dev/null; then
  echo "ERROR: no Python venv appears to be active." >&2
  echo "Create and activate the venv per the repo README, then re-run." >&2
  exit 1
fi

# tpu_raiden must already be available via EITHER supported path -- this script
# does NOT install it. The run scripts resolve whichever is present at launch
# (see raiden_env.sh).
if ! python -c "import tpu_sync.rpc.coordination_pb2" 2>/dev/null \
   && [ ! -f "${RAIDEN_ROOT}/tpu_sync/frameworks/jax/_tpu_raiden_jax.so" ]; then
  echo "ERROR: tpu_raiden is not available. Install it via one of:" >&2
  echo "  - build from source: run \`./build.sh jax\` from the repo root, or" >&2
  echo "  - wheel:             \`pip install --pre tpu-sync-jax\`" >&2
  exit 1
fi

echo "venv:            ${VIRTUAL_ENV:-$(python -c 'import sys; print(sys.prefix)')}"
echo "raiden root:     ${RAIDEN_ROOT}"
echo "src dir:         ${SRC_DIR}"
echo "tpu-inference:   ${TPU_INFERENCE_REPO} @ ${TPU_INFERENCE_COMMIT}"
mkdir -p "${SRC_DIR}"

# --- Clone (or update) at the pinned ref ----------------------------------
clone_pinned() {
  local name="$1" repo="$2" ref="$3" dest="${SRC_DIR}/$1"
  if [ -d "${dest}/.git" ]; then
    echo "=== ${name}: existing checkout, fetching ${ref} ==="
    git -C "${dest}" fetch --quiet origin "${ref}" 2>/dev/null || git -C "${dest}" fetch --quiet origin
  else
    echo "=== ${name}: cloning ${repo} ==="
    git clone "${repo}" "${dest}"
    git -C "${dest}" fetch --quiet origin "${ref}" 2>/dev/null || true
  fi
  # Works for a commit SHA, a tag, or a branch (via FETCH_HEAD when fetched).
  git -C "${dest}" checkout --quiet "${ref}" 2>/dev/null \
    || git -C "${dest}" checkout --quiet FETCH_HEAD
  echo "${name} @ $(git -C "${dest}" rev-parse --short HEAD)"
}

clone_pinned vllm "${VLLM_REPO}" "${VLLM_COMMIT}"
clone_pinned tpu-inference "${TPU_INFERENCE_REPO}" "${TPU_INFERENCE_COMMIT}"

# --- Verify the offload connector is present in this tpu-inference ref -----
if [ ! -f "${SRC_DIR}/tpu-inference/${OFFLOAD_CONNECTOR_REL}" ]; then
  echo "ERROR: ${OFFLOAD_CONNECTOR_REL} is missing from the tpu-inference" >&2
  echo "checkout at ref '${TPU_INFERENCE_COMMIT}'. Use a ref that contains the" >&2
  echo "RaidenOffloadConnector (the pinned default does), e.g.:" >&2
  echo "  TPU_INFERENCE_COMMIT=<commit-with-connector> bash setup.sh" >&2
  exit 1
fi

# --- Install into the active venv (vllm first; tpu-inference depends on it) -
echo "=== Installing vLLM (TPU target, editable) ==="
pip install -r "${SRC_DIR}/vllm/requirements/tpu.txt"
# Default PyPI serves a CUDA torch, whose Torch CMake config makes vLLM's
# find_package(Torch) demand CUDA libs (build fails). Pin the matching CPU build
# (paired with torchvision==0.25.0 from tpu-inference's requirements).
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0+cpu torchvision==0.25.0+cpu
# --no-build-isolation: build against the venv's CPU torch. Without it, pip
# provisions a fresh CUDA torch in an isolated overlay and CMake fails on CUDA.
VLLM_TARGET_DEVICE="tpu" pip install -e "${SRC_DIR}/vllm" --no-build-isolation

echo "=== Installing tpu-inference (editable) ==="
pip install -r "${SRC_DIR}/tpu-inference/requirements.txt"
pip install -e "${SRC_DIR}/tpu-inference"

# The benchmark client (benchmark.sh) uses tpu-inference's benchmark_serving.py,
# which imports these two eval-only helpers.
echo "=== Installing benchmark client deps (evaluate, nltk) ==="
pip install evaluate nltk

echo ""
echo "=== Setup complete! ==="
echo "vllm + tpu_inference are installed in the active venv; tpu_raiden is"
echo "resolved at run time by raiden_env.sh (site-packages if wheel-installed, or"
echo "the source tree via PYTHONPATH if built from source)."
echo "Next: run the offloading demo with"
echo "    bash ${SCRIPTS_DIR}/run_offload.sh"
