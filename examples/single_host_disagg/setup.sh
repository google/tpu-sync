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
# One-click environment setup for the single-host disaggregated-serving example.
#
# Clones vLLM and tpu-inference (vLLM's TPU backend, which ships the Raiden
# KV-connector) at their pinned commits into a hidden, in-tree `.src/` dir and
# installs them editable into the CURRENTLY ACTIVE venv.
#
# Prerequisites:
#   1. A python3.12 venv created and ACTIVATED.
#   2. tpu_raiden already installed into that venv via EITHER supported path:
#        - build from source: `./build.sh jax` from the repo root, or
#        - wheel:             `pip install --pre tpu-sync-jax`
#      (Both work at run time; `--pre` is needed until the first stable
#      tpu-sync-jax release is published to PyPI.)
# Then, from this directory:
#   bash setup.sh
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIDEN_ROOT="${RAIDEN_ROOT:-$(cd "${SCRIPTS_DIR}/../.." && pwd)}"
SRC_DIR="${SRC_DIR:-${SCRIPTS_DIR}/.src}"

# Pinned commits -- keep these three in lockstep (jax/jaxlib/libtpu are coupled).
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_COMMIT="${VLLM_COMMIT:-6185d73882c0cdfd9ee13cea16a9b50d2b5267be}"
TPU_INFERENCE_REPO="${TPU_INFERENCE_REPO:-https://github.com/vllm-project/tpu-inference.git}"
TPU_INFERENCE_COMMIT="${TPU_INFERENCE_COMMIT:-7ed46a69e4bf6900af69ad681e01df40a355be85}"

# --- Preconditions --------------------------------------------------------
if [ -z "${VIRTUAL_ENV:-}" ] && ! python -c "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)" 2>/dev/null; then
  echo "ERROR: no Python venv appears to be active." >&2
  echo "Create and activate the venv per the repo README, then re-run." >&2
  exit 1
fi

# tpu_raiden must already be available via EITHER supported path -- this script
# does NOT install it:
#   - wheel install: tpu_raiden (with bundled protos) is already importable, or
#   - build-from-source: the engine .so exists in the source tree.
# The run scripts resolve whichever is present at launch -- see raiden_env.sh.
if ! python -c "import tpu_sync.rpc.coordination_pb2" 2>/dev/null \
   && [ ! -f "${RAIDEN_ROOT}/tpu_sync/frameworks/jax/_tpu_raiden_jax.so" ]; then
  echo "ERROR: tpu_raiden is not available. Install it via one of:" >&2
  echo "  - build from source: run \`./build.sh jax\` from the repo root, or" >&2
  echo "  - wheel:             \`pip install --pre tpu-sync-jax\`" >&2
  exit 1
fi

echo "venv:        ${VIRTUAL_ENV:-$(python -c 'import sys; print(sys.prefix)')}"
echo "raiden root: ${RAIDEN_ROOT}"
echo "src dir:     ${SRC_DIR}"
mkdir -p "${SRC_DIR}"

# --- Clone (or update) at the pinned commit -------------------------------
clone_pinned() {
  local name="$1" repo="$2" commit="$3" dest="${SRC_DIR}/$1"
  if [ -d "${dest}/.git" ]; then
    echo "=== ${name}: existing checkout, fetching ${commit} ==="
    git -C "${dest}" fetch --quiet origin "${commit}" 2>/dev/null || git -C "${dest}" fetch --quiet origin
  else
    echo "=== ${name}: cloning ${repo} ==="
    git clone "${repo}" "${dest}"
  fi
  git -C "${dest}" checkout --quiet "${commit}"
  echo "${name} @ $(git -C "${dest}" rev-parse --short HEAD)"
}

clone_pinned vllm "${VLLM_REPO}" "${VLLM_COMMIT}"
clone_pinned tpu-inference "${TPU_INFERENCE_REPO}" "${TPU_INFERENCE_COMMIT}"

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

echo ""
echo "=== Setup complete! ==="
echo "vllm + tpu_inference are installed in the active venv; tpu_raiden is"
echo "resolved at run time by raiden_env.sh (site-packages if wheel-installed, or"
echo "the source tree via PYTHONPATH if built from source)."
echo "Next: run the disaggregated demo with"
echo "    bash ${SCRIPTS_DIR}/run_all.sh"