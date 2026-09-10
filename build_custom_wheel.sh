#!/usr/bin/env bash
# ==============================================================================
# Script: build_custom_wheel.sh
# Purpose: Build TPU-Sync PyTorch wheel guaranteed against specific dependency versions
# ==============================================================================
set -euo pipefail

# 1. Configuration & Directories
WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/pd/tpu-sync}"
TORCH_TPU_DIR="${TORCH_TPU_DIR:-/home/wenjung_google_com/torch_tpu}"
VERSIONS_FILE="${VERSIONS_FILE:-/tmp/versions.txt}"
VENV_DIR="${VENV_DIR:-/home/wenjung_google_com/venv_torch212}"
CACHE_BASE="${CACHE_BASE:-/mnt/pd/temp/bazel_cache}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/pd/temp/bazel_output_torch212}"
DIST_DIR="${WORKSPACE_DIR}/dist"

echo "=== [1/6] Setting up build directories and prerequisites ==="
mkdir -p "${CACHE_BASE}" "${OUTPUT_BASE}" "${DIST_DIR}"

# Ensure torch_tpu symlink is accessible relative to workspace parent if needed
if [[ ! -e "/mnt/pd/torch_tpu" ]]; then
  ln -sf "${TORCH_TPU_DIR}" /mnt/pd/torch_tpu
fi

echo "=== [2/6] Provisioning Isolated Virtual Environment ==="
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR}..."
  python3.12 -m venv "${VENV_DIR}"
fi

# Activate the target environment
source "${VENV_DIR}/bin/activate"

echo "=== [3/6] Installing Target Dependency Versions ==="
# Extract exact PyTorch version from versions.txt if provided
if [[ -f "${VERSIONS_FILE}" ]]; then
  TARGET_TORCH_VERSION=$(grep -E '^\s*torch\s+' "${VERSIONS_FILE}" | awk '{print $2}' || true)
fi
TARGET_TORCH_VERSION="${TARGET_TORCH_VERSION:-2.12.0+cpu}"

echo "Installing PyTorch version: ${TARGET_TORCH_VERSION}..."
pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TARGET_TORCH_VERSION}"

# Verify active PyTorch version
ACTUAL_TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
echo "Active PyTorch version confirmed: ${ACTUAL_TORCH_VERSION}"

# Compute ABI / Glue version tag (e.g. 2.12.0 -> 2_12_0)
TORCH_GLUE_SUFFIX=$(python3 -c 'import torch, re; v = re.match(r"(\d+)\.(\d+)\.(\d+)", torch.__version__); print(f"{v.group(1)}_{v.group(2)}_{v.group(3)}")')
SHORT_VERSION_TAG="torch${TORCH_GLUE_SUFFIX//_/}"

echo "=== [4/6] Configuring Bazel Build Environment ==="
cd "${WORKSPACE_DIR}"

export TORCH_TPU_MODULE_PATH="${TORCH_TPU_DIR}"
export BAZEL_CACHE_DIR="${CACHE_BASE}"
export BAZEL_OUTPUT_BASE="${OUTPUT_BASE}"

# Generate PEP-440 compliant wheel version tag (e.g. 0.0.1.dev<timestamp>+torch212)
TIMESTAMP=$(date +%Y%m%d%H%M%S)
export WHEEL_VERSION_EXTRAS=".dev${TIMESTAMP}+${SHORT_VERSION_TAG}"

echo "Wheel version extra: ${WHEEL_VERSION_EXTRAS}"

echo "=== [5/6] Building Wheel via Option 2 (build.sh) ==="
./build.sh torch //ci/wheel:raiden_torch_wheel --repo_env=WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS}"

# Copy newly built wheel to dist/
WHEEL_SRC=$(find "${OUTPUT_BASE}/execroot/_main/bazel-out/k8-opt/bin/ci/wheel/" -name "tpu_raiden_torch-*${TIMESTAMP}*.whl" | head -n 1)
if [[ -z "${WHEEL_SRC}" ]]; then
  # Fallback to latest wheel if exact timestamp match differed
  WHEEL_SRC=$(find "${OUTPUT_BASE}/execroot/_main/bazel-out/k8-opt/bin/ci/wheel/" -name "tpu_raiden_torch-*.whl" -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")
fi

cp -f "${WHEEL_SRC}" "${DIST_DIR}/"
WHEEL_PATH="${DIST_DIR}/$(basename "${WHEEL_SRC}")"

echo "=== [6/6] Verifying Build Artifacts ==="
echo "Verifying dynamic loader symbols on native extension..."
patchelf --print-needed "${WORKSPACE_DIR}/tpu_sync/frameworks/torch/_tpu_raiden_torch.so" | grep "libpywrap_${TORCH_GLUE_SUFFIX}_common.so"

echo "Verifying Python import test..."
python3 -c "import torch; print('PyTorch loaded:', torch.__version__); from tpu_sync.frameworks.torch import _tpu_raiden_host; print('TPU Sync host module verified successfully!')"

echo "=============================================================================="
echo "Build Successful!"
echo "Generated Wheel: ${WHEEL_PATH}"
echo "=============================================================================="