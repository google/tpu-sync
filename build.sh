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
set -e

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

check_disk_space() {
  local dir="$1"
  local desc="$2"
  # Warn if free disk space is under 20 GB (20971520 KB)
  local min_kb=20971520
  mkdir -p "$dir" 2>/dev/null || true
  if command -v df > /dev/null && command -v awk > /dev/null; then
    local free_kb
    free_kb="$(df -P "$dir" 2>/dev/null | awk 'NR==2 {print $4}')"
    if [[ "$free_kb" =~ ^[0-9]+$ ]] && (( free_kb < min_kb )); then
      local free_gb=$((free_kb / 1048576))
      echo "WARNING: Low disk space on ${desc} (${dir}). Only ~${free_gb} GB free (< 20 GB recommended for Bazel)." >&2
    fi
  fi
}

# Define directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
DEFAULT_WORKSPACE_DIR="$SCRIPT_DIR"
WORKSPACE_DIR="${WORKSPACE_DIR:-${DEFAULT_WORKSPACE_DIR}}"
# Check for persistent storage mounted on Cloud TPU VMs (3-4 TB SSDs)
if [[ -d "/mnt/disks/persistent/${USER}" && -w "/mnt/disks/persistent/${USER}" ]] || [[ -d /mnt/disks/persistent && -w /mnt/disks/persistent ]]; then
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
BAZEL_DISK_CACHE="${BAZEL_CACHE_BASE}/disk_cache"
BAZEL_REPO_CACHE="${BAZEL_CACHE_BASE}/repo_cache"
BAZEL_OUTPUT_BASE="${BAZEL_OUTPUT_BASE:-${DEFAULT_BAZEL_OUTPUT_BASE}}"

check_disk_space "${WORKSPACE_DIR}" "Workspace Directory"
check_disk_space "${BAZEL_CACHE_BASE}" "Bazel Cache Base"
check_disk_space "${BAZEL_OUTPUT_BASE}" "Bazel Output Base"
check_disk_space "/tmp" "Temporary Directory (/tmp)"

echo "=== Navigating to workspace directory ==="
cd "${WORKSPACE_DIR}"
TORCH_TPU_MODULE_PATH="${TORCH_TPU_MODULE_PATH:-../torch_tpu}"

# 0. Set up standalone Bazel environment in /tmp
BAZEL_VERSION="$(cat .bazelversion | tr -d '[:space:]')"

BAZEL_BIN="/tmp/bazel-bootstrap-${BAZEL_VERSION}"
if [[ ! -f "${BAZEL_BIN}" ]]; then
  echo "Bootstrapping standalone Bazel ${BAZEL_VERSION} to temporary folder ${BAZEL_BIN}..."
  download_file "https://storage.googleapis.com/bazel/${BAZEL_VERSION}/release/bazel-${BAZEL_VERSION}-linux-x86_64" "${BAZEL_BIN}"
  chmod +x "${BAZEL_BIN}"
fi

"${BAZEL_BIN}" --version

# Default behavior based on auto-detection
BUILD_JAX=true
BUILD_TORCH=true

if [ ! -f "${TORCH_TPU_MODULE_PATH}/MODULE.bazel" ]; then
  echo "torch_tpu checkout not found at ${TORCH_TPU_MODULE_PATH}. Defaulting to JAX-only build."
  BUILD_TORCH=false
fi

# Parse command line arguments
if [ "$#" -gt 0 ]; then
  case "$1" in
    jax)
      BUILD_JAX=true
      BUILD_TORCH=false
      shift
      ;;
    torch)
      BUILD_JAX=false
      BUILD_TORCH=true
      shift
      ;;
    both)
      BUILD_JAX=true
      BUILD_TORCH=true
      shift
      ;;
    -*)
      # Standard Bazel/environment flags: keep defaults and do not shift.
      ;;
    *)
      echo "Usage: $0 [jax|torch|both] [bazel_flags...]"
      exit 1
      ;;
  esac
fi

if [ "${BUILD_TORCH}" = true ]; then
  # Ensure clang-18 is installed on the host (required for PyTorch C++ extensions)
  if ! command -v clang-18 > /dev/null || ! command -v clang++-18 > /dev/null; then
    echo "clang-18 / clang++-18 not found on host. Attempting automatic installation..."
    if command -v apt-get > /dev/null && command -v sudo > /dev/null; then
      sudo apt-get update && sudo apt-get install -y clang-18
    else
      echo "Error: clang-18 / clang++-18 is required to build Torch extensions." >&2
      echo "Please install clang-18 manually on your system." >&2
      exit 1
    fi
  fi

  if [[ -z "${CC:-}" ]]; then
    export CC="$(command -v clang-18)"
  fi
  if [[ -z "${CXX:-}" ]]; then
    export CXX="$(command -v clang++-18)"
  fi
fi

BAZEL_TARGETS=(
  "//tpu_sync/rpc:raiden_service_py_pb2"
  "//tpu_sync/rpc:coordination_py_pb2"
  "//tpu_sync/rpc:coordination_py_pb2_grpc"
  "//tpu_sync/rpc:controller_service_py_pb2"
  # C++ control-plane service binaries. These do not depend on JAX or Torch,
  # so they are always built.
  "//tpu_sync/kv_cache/global_registry:global_registry_server"
  "//tpu_sync/store_node:kv_cache_host_store_node_main"
)
DEFINE_FLAGS=""
BAZEL_MODULE_FLAGS=()
TORCH_REPO_ENV_FLAGS=()

if [ "$BUILD_JAX" = true ]; then
  echo "Configuring build for JAX..."
  BAZEL_TARGETS+=(
    "//tpu_sync/frameworks/jax:_tpu_raiden_jax"
  )
else
  DEFINE_FLAGS+=" --define with_jax=false"
fi

if [ "$BUILD_TORCH" = true ]; then
  echo "Configuring build for Torch..."
  if [[ ! -f "${TORCH_TPU_MODULE_PATH}/MODULE.bazel" ]]; then
    echo "Error: Torch build requires a torch_tpu checkout at ${TORCH_TPU_MODULE_PATH}." >&2
    echo "Set TORCH_TPU_MODULE_PATH to override the default ../torch_tpu location." >&2
    exit 1
  fi
  TORCH_TPU_MODULE_PATH="$(cd "${TORCH_TPU_MODULE_PATH}" && pwd)"
  BAZEL_MODULE_FLAGS+=("--override_module=torch_tpu=${TORCH_TPU_MODULE_PATH}")

  # The torch extension calls virtual methods on PJRT objects that torch_tpu
  # constructs, so both binaries must be compiled from XLA revisions with the
  # same C++ ABI on that surface. MODULE.bazel's xla pin follows JAX's and
  # covers only the JAX leg; torch-only builds therefore compile against the
  # exact XLA revision the torch_tpu checkout pins, materialized locally with
  # this repo's xla patches applied (a path override bypasses git_override
  # patching). RAIDEN_TORCH_XLA=<dir> supplies a pre-patched tree instead;
  # RAIDEN_TORCH_XLA=module keeps MODULE.bazel's pin. A combined jax+torch
  # invocation has a single xla module and cannot express both pins.
  if [ "$BUILD_JAX" = true ]; then
    echo "WARNING: combined jax+torch build uses MODULE.bazel's xla pin for" \
         "both legs; the torch leg is NOT built against torch_tpu's XLA" \
         "revision. Release torch builds must be torch-only." >&2
  elif [[ "${RAIDEN_TORCH_XLA:-}" == "module" ]]; then
    echo "RAIDEN_TORCH_XLA=module: torch leg uses MODULE.bazel's xla pin."
  else
    # Downloads <repo>/archive/<commit>.tar.gz into the bazel cache (keyed by
    # commit, reused across builds) and applies every patch in <patches_dir>;
    # a path override bypasses bzlmod's own patching, so the tree must be
    # patched here. Sets MATERIALIZED_DIR to the result.
    materialize_pinned_module() {
      local repo="$1" commit="$2" patches_dir="$3"
      local name; name="$(basename "${repo}")"
      MATERIALIZED_DIR="${BAZEL_CACHE_BASE}/torch_leg_deps/${name}/${commit}"
      if [[ ! -f "${MATERIALIZED_DIR}/MODULE.bazel" ]]; then
        echo "Materializing ${name} @ ${commit} with patches..."
        mkdir -p "${BAZEL_CACHE_BASE}"
        local stage; stage="$(mktemp -d "${BAZEL_CACHE_BASE}/${name}_stage.XXXXXX")"
        download_file "${repo}/archive/${commit}.tar.gz" "${stage}/src.tar.gz"
        tar -xzf "${stage}/src.tar.gz" -C "${stage}"
        if [[ -n "${patches_dir}" ]]; then
          local p
          for p in "${patches_dir}"/*.patch; do
            patch -p1 -s -d "${stage}/${name}-${commit}" < "$p"
          done
        fi
        mkdir -p "$(dirname "${MATERIALIZED_DIR}")"
        mv "${stage}/${name}-${commit}" "${MATERIALIZED_DIR}"
        rm -rf "${stage}"
      fi
    }

    if [[ -n "${RAIDEN_TORCH_XLA:-}" ]]; then
      TORCH_XLA_DIR="$(cd "${RAIDEN_TORCH_XLA}" && pwd)"
    else
      XLA_REVISION_FILE="${TORCH_TPU_MODULE_PATH}/bazel/xla_revision.bzl"
      XLA_COMMIT="$(sed -n -E 's/^XLA_COMMIT = "([0-9a-f]{40})".*/\1/p' "${XLA_REVISION_FILE}")"
      if [[ -z "${XLA_COMMIT}" ]]; then
        echo "Error: could not read XLA_COMMIT from ${XLA_REVISION_FILE}." >&2
        exit 1
      fi
      materialize_pinned_module "https://github.com/openxla/xla" \
        "${XLA_COMMIT}" "${WORKSPACE_DIR}/third_party/xla"
      TORCH_XLA_DIR="${MATERIALIZED_DIR}"
    fi
    echo "torch leg xla override: ${TORCH_XLA_DIR}"
    BAZEL_MODULE_FLAGS+=("--override_module=xla=${TORCH_XLA_DIR}")

    # xla's module extensions call rules_ml_toolchain rules and the two move
    # together upstream, so the revision must match the xla override rather
    # than MODULE.bazel's pin. The xla tree pins its compatible revision in
    # its own MODULE.bazel; mirror that pin. Used unpatched: the load fix in
    # third_party/rules_ml_toolchain/*.patch targets the older revision
    # MODULE.bazel pins and is already upstream at the revisions xla pins.
    RMT_COMMIT="$(sed -n -E 's/.*strip_prefix = "rules_ml_toolchain-([0-9a-f]{40})".*/\1/p' \
      "${TORCH_XLA_DIR}/MODULE.bazel" | head -1)"
    if [[ -z "${RMT_COMMIT}" ]]; then
      echo "Error: could not read the rules_ml_toolchain pin from ${TORCH_XLA_DIR}/MODULE.bazel." >&2
      exit 1
    fi
    materialize_pinned_module "https://github.com/google-ml-infra/rules_ml_toolchain" \
      "${RMT_COMMIT}" ""
    echo "torch leg rules_ml_toolchain override: ${MATERIALIZED_DIR}"
    BAZEL_MODULE_FLAGS+=("--override_module=rules_ml_toolchain=${MATERIALIZED_DIR}")
  fi
  if [[ -z "${TORCH_SOURCE:-}" ]]; then
    TORCH_SOURCE="$(python3 - <<'PY'
import importlib.util
import pathlib

spec = importlib.util.find_spec("torch")
if spec is None or not spec.submodule_search_locations:
  raise SystemExit("torch package not found on Python path")
print(pathlib.Path(next(iter(spec.submodule_search_locations))).resolve().parent)
PY
)"
  fi
  export TORCH_SOURCE
  echo "Using local torch from: ${TORCH_SOURCE}"
  # Local-torch mode needs two switches. The define drives raiden's own
  # //ci/wheel config (the wheel then declares no torch requirement). The
  # @torch_tpu//shims/torch:local_torch flag drives torch_tpu's torch shims;
  # torch_tpu's own --config=local_torch sets it via its .bazelrc, which does
  # not apply when torch_tpu is a dependency module, so pass it explicitly.
  # Without the flag, the shims resolve to torch_tpu's pinned pypi torch and
  # TORCH_SOURCE never enters the build.
  DEFINE_FLAGS+=" --define=TORCH_SOURCE=local"
  TORCH_REPO_ENV_FLAGS+=("--@torch_tpu//shims/torch:local_torch=True")
  TORCH_REPO_ENV_FLAGS+=("--repo_env=TORCH_SOURCE=${TORCH_SOURCE}")
  BAZEL_TARGETS+=(
    "//tpu_sync/frameworks/torch:_tpu_raiden_host"
    "//tpu_sync/frameworks/torch:_tpu_raiden_torch"
    "//tpu_sync/frameworks/torch:_torch_raw_transfer"
  )
else
  DEFINE_FLAGS+=" --define with_torch=false"
  DUMMY_TORCH_TPU_MODULE="${BAZEL_CACHE_BASE}/dummy_torch_tpu_module"
  mkdir -p "${DUMMY_TORCH_TPU_MODULE}"
  cat > "${DUMMY_TORCH_TPU_MODULE}/MODULE.bazel" <<'EOF'
module(name = "torch_tpu", version = "0.1.1")
EOF
  BAZEL_MODULE_FLAGS+=("--override_module=torch_tpu=${DUMMY_TORCH_TPU_MODULE}")
fi

if [ ${#BAZEL_TARGETS[@]} -eq 0 ]; then
  echo "No targets selected to build!"
  exit 1
fi

mkdir -p "${BAZEL_DISK_CACHE}" "${BAZEL_REPO_CACHE}" "$(dirname "${BAZEL_OUTPUT_BASE}")"

echo "=== Building targets with Bazel ==="
"${BAZEL_BIN}" --install_base="${BAZEL_OUTPUT_BASE}/install_base" --output_base="${BAZEL_OUTPUT_BASE}" --host_jvm_args="-Xmx32g" --host_jvm_args="-Xms2g" build -c opt --check_visibility=false --verbose_failures --experimental_repo_remote_exec --incompatible_disallow_empty_glob=false \
  --repo_env=HERMETIC_PYTHON_VERSION=${HERMETIC_PYTHON_VERSION:-3.12} \
  --repo_env=PIP_INDEX_URL="https://pypi.org/simple" \
  --repo_env=PIP_EXTRA_INDEX_URL="" \
  --repo_env=PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring" \
  --repo_env=PIP_CONFIG_FILE="/dev/null" \
  "${BAZEL_MODULE_FLAGS[@]}" \
  "${TORCH_REPO_ENV_FLAGS[@]}" \
  "${BAZEL_TARGETS[@]}" \
  ${DEFINE_FLAGS} \
  --disk_cache=${BAZEL_DISK_CACHE} \
  --repository_cache=${BAZEL_REPO_CACHE} \
  "$@"


echo "=== Copying generated protobuf Python modules ==="
cp -f "${WORKSPACE_DIR}/bazel-bin/tpu_sync/rpc/raiden_service_pb2.py" "${WORKSPACE_DIR}/tpu_sync/rpc/" 2>/dev/null || true
cp -f "${WORKSPACE_DIR}/bazel-bin/tpu_sync/rpc/coordination_pb2.py" "${WORKSPACE_DIR}/tpu_sync/rpc/" 2>/dev/null || true
cp -f "${WORKSPACE_DIR}/bazel-bin/tpu_sync/rpc/coordination_pb2_grpc.py" "${WORKSPACE_DIR}/tpu_sync/rpc/" 2>/dev/null || true
cp -f "${WORKSPACE_DIR}/bazel-bin/tpu_sync/rpc/controller_service_pb2.py" "${WORKSPACE_DIR}/tpu_sync/rpc/" 2>/dev/null || true

echo "=== Linking C++ control-plane service binaries into source tree ==="
# The KVCacheStore tests (kv_cache_store_test.py / kv_cache_store_e2e_test.py)
# launch these binaries from their SOURCE-tree path, but Bazel emits them under
# bazel-bin/. Symlink them so a fresh checkout can run the tests without extra
# steps.
link_service_binary() {
  local rel="$1"
  local src="${WORKSPACE_DIR}/bazel-bin/${rel}"
  local dst="${WORKSPACE_DIR}/${rel}"
  if [ -e "${src}" ]; then
    ln -sf "${src}" "${dst}"
  fi
}
link_service_binary "tpu_sync/kv_cache/global_registry/global_registry_server"
link_service_binary "tpu_sync/store_node/kv_cache_host_store_node_main"

echo "=== Copying compiled shared libraries to source directory ==="
if [ "$BUILD_JAX" = true ]; then
  echo "Copying JAX artifacts..."
  cp -f "${WORKSPACE_DIR}/bazel-bin/tpu_sync/frameworks/jax/_tpu_raiden_jax.so" "${WORKSPACE_DIR}/tpu_sync/frameworks/jax/"
fi

if [ "${BUILD_TORCH}" = true ]; then
  echo "=== Copying torch extension modules into source tree ==="
  # Python tests and source-checkout consumers import the extensions from the
  # package directory; Bazel emits them under bazel-bin/.
  for so in _tpu_raiden_host.so _tpu_raiden_torch.so _torch_raw_transfer.so; do
    src="${WORKSPACE_DIR}/bazel-bin/tpu_sync/frameworks/torch/${so}"
    if [ -e "${src}" ]; then
      cp -f "${src}" "${WORKSPACE_DIR}/tpu_sync/frameworks/torch/${so}"
      chmod u+w "${WORKSPACE_DIR}/tpu_sync/frameworks/torch/${so}"
    fi
  done
  # The torch-linked extensions are built with --allow-shlib-undefined and no
  # NEEDED entries; the wheel packaging adds the per-version glue dependency.
  # For source-tree runs, add a NEEDED on the unversioned common library so
  # dlopen resolves torch_tpu/torch symbols through the dependency chain that
  # tpu_sync.api.torch.torch_tpu_common_loader preloads (RTLD_LOCAL; a global
  # load would collide with the extension's statically linked gRPC/XLA).
  if command -v patchelf > /dev/null; then
    for so in _tpu_raiden_torch.so _torch_raw_transfer.so; do
      dst="${WORKSPACE_DIR}/tpu_sync/frameworks/torch/${so}"
      if [ -e "${dst}" ] && \
         ! readelf -d "${dst}" | grep -q "libpywrap_torch_tpu_common.so"; then
        patchelf --add-needed libpywrap_torch_tpu_common.so "${dst}"
      fi
    done
  else
    echo "WARNING: patchelf not found; in-tree torch extensions will not" \
         "import outside bazel (missing libpywrap NEEDED)." >&2
  fi
fi


echo "=== Build Complete! ==="
if [ "$BUILD_JAX" = true ]; then
  echo "JAX Artifacts are located in: ${WORKSPACE_DIR}/tpu_sync/frameworks/jax/"
fi
if [ "$BUILD_TORCH" = true ]; then
  echo "Torch Artifacts are located in: ${WORKSPACE_DIR}/tpu_sync/frameworks/torch/"
fi

echo "=== Install Python Dependencies! ==="
echo "Using Python interpreter: $(which python3) ($(python3 --version))"
python3 -m pip install --index-url=https://pypi.org/simple -r requirements.txt || echo "Warning: pip installation returned a non-zero status. Proceeding anyway."

echo "=== Installation Complete! ==="
