#!/bin/bash
# Builds the tpu_raiden wheel hermetically inside the ml-build container
# (glibc 2.35, matching the TPU runtime), mirroring torch_tpu/ci/build_wheel.sh.
#
# Unlike torch_tpu, raiden needs (a) clang-18 for XLA's .ll codegen targets and
# (b) a local torch for the torch_tpu shim headers, plus the torch_tpu module.
# This script installs clang-18 + CPU torch into the container, mounts a sibling
# torch_tpu checkout, and runs build.sh for the wheel target.
#
# One wheel per invocation:
#   ci/build_wheel.sh [jax]            # tpu_raiden_jax
#   ci/build_wheel.sh torch            # tpu_raiden_torch (needs ../torch_tpu)
#   TORCH_TPU_SRC=/path ci/build_wheel.sh torch

set -exu -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS:-.dev$(date +%Y%m%d%H%M%S)}"
export WHEEL_VERSION_EXTRAS
echo "WHEEL_VERSION_EXTRAS: ${WHEEL_VERSION_EXTRAS}"

RAIDEN_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null)" ]]; then
  RAIDEN_COMMIT="${RAIDEN_COMMIT}-dirty"
fi
echo "raiden source commit: ${RAIDEN_COMMIT}"

BUILD_MODE="${1:-jax}"
if [[ "${BUILD_MODE}" != "torch" && "${BUILD_MODE}" != "jax" ]]; then
  echo "Usage: ci/build_wheel.sh [torch|jax]" >&2
  exit 1
fi

# Torch releases bundled as extra ABI variants next to the primary (the
# container's torch). Mirrors torch_tpu's supported-torch glue set, limited to
# releases installable from the PyTorch CPU wheel index (torch_tpu also carries
# glue for unreleased torches, which have no wheel to compile a variant
# against). Space-separated; set to "" for a single-ABI build.
RAIDEN_EXTRA_TORCH_ABIS="${RAIDEN_EXTRA_TORCH_ABIS-2.12.0 2.13.0}"
TORCH_TPU_SRC="${TORCH_TPU_SRC:-${REPO_ROOT}/../torch_tpu}"
WHEEL_DIR="${KOKORO_ARTIFACTS_DIR:-${HOME}/raiden_artifacts}/dist"
CACHE_DIR="${RAIDEN_CONTAINER_CACHE:-${HOME}/.bazel_cache_container}"
mkdir -p "${WHEEL_DIR}" "${REPO_ROOT}/dist" "${CACHE_DIR}"

CONTAINER_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest"
echo "===> Pulling ${CONTAINER_IMAGE}..."
docker pull "${CONTAINER_IMAGE}"

DOCKER_MOUNTS=(
  -v "${REPO_ROOT}:/workspace"
  -v "${CACHE_DIR}:/cache"
)
# The torch wheel has no jax deps; a torch-only build.sh invocation also lets
# the torch leg compile against the mounted torch_tpu's pinned XLA revision
# (see the xla override in build.sh), which a combined jax+torch invocation
# cannot do.
if [[ "${BUILD_MODE}" == "torch" ]]; then
  if [[ ! -f "${TORCH_TPU_SRC}/MODULE.bazel" ]]; then
    echo "ERROR: torch build needs a torch_tpu checkout at ${TORCH_TPU_SRC}" >&2
    echo "       set TORCH_TPU_SRC=<path> or build with 'ci/build_wheel.sh jax'." >&2
    exit 1
  fi
  TORCH_TPU_SRC="$(cd "${TORCH_TPU_SRC}" && pwd)"
  echo "torch_tpu source commit: $(git -C "${TORCH_TPU_SRC}" rev-parse HEAD 2>/dev/null || echo unknown)"
  DOCKER_MOUNTS+=(-v "${TORCH_TPU_SRC}:/torch_tpu")  # sibling ../torch_tpu == /torch_tpu
fi

# The in-container build: install clang-18 (XLA .ll targets), libstdc++-12 (the
# standard library clang compiles against) + CPU torch (shim headers), then
# drive the existing build.sh for the wheel target.
read -r -d '' INNER <<'INNER_EOF' || true
set -exu -o pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq wget gnupg ca-certificates patchelf patch >/dev/null
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
# alongside it and clang selects it automatically. Its runtime is already the
# one the container ships (libstdc++.so.6.0.30), so nothing new is required at
# load time.
apt-get install -y -qq clang-18 libstdc++-12-dev >/dev/null
ln -sf /usr/bin/clang-18 /usr/bin/clang
ln -sf /usr/bin/clang++-18 /usr/bin/clang++
clang --version | head -1

if [[ "${BUILD_MODE}" == "torch" ]]; then
  # raiden is built on top of torch_tpu: raiden's _tpu_raiden_torch.so and
  # torch_tpu's libpywrap_torch_tpu_common.so must resolve the SAME libtorch
  # symbols at runtime. So raiden MUST compile against the EXACT torch that
  # torch_tpu was built against — never a floating `torch>=X` specifier, which
  # drifts to the latest release and breaks ABI (e.g. torch 2.13.x drops
  # `torch::autograd::deleteNode`, which torch_tpu's libpywrap needs → dlopen
  # `undefined symbol` at import). torch_tpu's source of truth is its per-Python
  # requirements lock, which pins an exact `torch==VERSION+cpu`.
  PYTAG="$(python3 -c 'import sys;print(f"{sys.version_info.major}_{sys.version_info.minor}")')"
  TORCH_REQ_FILE="/torch_tpu/requirements/requirements_${PYTAG}.txt"
  TORCH_PIN=""
  if [[ -f "${TORCH_REQ_FILE}" ]]; then
    # e.g. line `torch==2.11.0+cpu \` -> `torch==2.11.0+cpu`
    TORCH_PIN=$(sed -n -E 's/^(torch==[0-9][0-9A-Za-z.+_-]*).*/\1/p' "${TORCH_REQ_FILE}" | head -1 || true)
  fi
  if [[ -n "${TORCH_PIN}" ]]; then
    echo "Installing torch pinned by torch_tpu (${TORCH_REQ_FILE}): ${TORCH_PIN}"
    pip install -q "${TORCH_PIN}" --index-url https://download.pytorch.org/whl/cpu
  else
    # Fallback: the (looser) specifier from torch_tpu's pyproject.toml. This can
    # float to the latest release and may NOT match torch_tpu's ABI, so warn.
    TORCH_VERSION=""
    if [[ -f /torch_tpu/pyproject.toml ]]; then
      TORCH_VERSION=$(sed -n -E 's/.*["'\''`]torch[[:space:]]*([>=<~=]+[0-9.a-zA-Z+-]+)["'\''`].*/\1/p' /torch_tpu/pyproject.toml 2>/dev/null | head -1 || true)
    fi
    if [[ -z "${TORCH_VERSION}" ]]; then
      echo "WARNING: could not determine torch pin from ${TORCH_REQ_FILE} or /torch_tpu/pyproject.toml. Installing latest torch — this may NOT match torch_tpu's ABI." >&2
      pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    else
      echo "WARNING: no exact pin in ${TORCH_REQ_FILE}; falling back to torch_tpu pyproject specifier 'torch${TORCH_VERSION}', which may float to a torch that does not match torch_tpu's ABI." >&2
      pip install -q "torch${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
    fi
  fi
  TORCH_SOURCE="$(python3 -c 'import torch,pathlib;print(pathlib.Path(torch.__file__).resolve().parent.parent)')"
  export TORCH_SOURCE
  export TORCH_TPU_MODULE_PATH=/torch_tpu
fi

# Persistent, resumable bazel cache + output base on the mounted volume.
export BAZEL_CACHE_DIR=/cache
export BAZEL_OUTPUT_BASE=/cache/output_base

# Separate per-framework wheels: tpu_raiden_torch (no jax deps) vs
# tpu_raiden_jax. Pick by BUILD_MODE.
if [[ "${BUILD_MODE}" == "torch" ]]; then
  WHEEL_TARGET="//ci/wheel:raiden_torch_wheel"
  WHEEL_DIST="tpu_raiden_torch"
else
  WHEEL_TARGET="//ci/wheel:raiden_jax_wheel"
  WHEEL_DIST="tpu_raiden_jax"
fi
# Match ONLY the wheel this build just produced. cache/output_base is shared
# across builds, so its bin/ci/wheel/ dir accumulates wheels from earlier runs,
# each with a distinct .dev<timestamp>. A broad "${WHEEL_DIST}-*.whl" glob would
# also match those stale wheels and hand multiple paths to the single-wheel
# patchelf step below (which then fails). WHEEL_VERSION_EXTRAS (.dev<timestamp>)
# is unique per build and appears verbatim in the filename, so scope to it.
WHEEL_GLOB="${WHEEL_DIST}-*${WHEEL_VERSION_EXTRAS}-*.whl"

cd /workspace
./build.sh "${BUILD_MODE}" "${WHEEL_TARGET}" \
  --repo_env=WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS}"

mkdir -p /workspace/dist
cp /cache/output_base/execroot/_main/bazel-out/k8-opt/bin/ci/wheel/${WHEEL_GLOB} /workspace/dist/

# The bazel-built _tpu_raiden_torch.so does not link libpywrap; the torch
# extension loader (tpu_sync/api/torch/torch_abi.py) requires a NEEDED on
# torch_tpu's per-torch-version glue so the torch_tpu symbols resolve in
# RTLD_LOCAL scope at import. The wheel ships one version-suffixed extension
# per torch ABI (_tpu_raiden_torch_<v>.so); torch_abi.load_extension picks
# the variant matching the installed torch. RAIDEN_TORCH_ABIS lists the
# torch releases to build variants for; the first entry doubles as the
# torch the rest of the wheel build compiles against.
torch_suffix() {
  python3 -c 'import torch, re; v = re.match(r"(\d+)\.(\d+)\.(\d+)", torch.__version__); print(f"{v.group(1)}_{v.group(2)}_{v.group(3)}")'
}
if [[ "${BUILD_MODE}" == "torch" ]]; then
  pip install -q wheel
  WHL="$(ls /workspace/dist/${WHEEL_GLOB} | head -1)"
  UNPACK_DIR="$(mktemp -d)"
  wheel unpack "${WHL}" -d "${UNPACK_DIR}"
  PKG_DIR="$(ls -d "${UNPACK_DIR}"/*/)"
  EXT_DIR="${PKG_DIR}tpu_sync/frameworks/torch"

  # Primary variant: the torch this wheel build compiled against.
  SUFFIX="$(torch_suffix)"
  mv "${EXT_DIR}/_tpu_raiden_torch.so" "${EXT_DIR}/_tpu_raiden_torch_${SUFFIX}.so"
  patchelf --add-needed "libpywrap_${SUFFIX}_common.so" \
    "${EXT_DIR}/_tpu_raiden_torch_${SUFFIX}.so"
  echo "wheel variant: _tpu_raiden_torch_${SUFFIX}.so (NEEDED libpywrap_${SUFFIX}_common.so)"

  # Extra variants: rebuild the extension against each additional torch in an
  # isolated venv (a fresh TORCH_SOURCE path forces the bazel torch repo to
  # re-resolve; an in-place pip swap at the same path would be reused stale).
  for V in ${RAIDEN_EXTRA_TORCH_ABIS:-}; do
    python3 -m venv "/tmp/torch-abi-${V}"
    "/tmp/torch-abi-${V}/bin/pip" install -q "torch==${V}" \
      --index-url https://download.pytorch.org/whl/cpu
    TORCH_SOURCE="$("/tmp/torch-abi-${V}/bin/python3" -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).resolve().parent.parent)')"
    export TORCH_SOURCE
    SUFFIX="$("/tmp/torch-abi-${V}/bin/python3" -c 'import torch, re; v = re.match(r"(\d+)\.(\d+)\.(\d+)", torch.__version__); print(f"{v.group(1)}_{v.group(2)}_{v.group(3)}")')"
    # build.sh derives the glue suffix from the system python's torch, which
    # is still the primary variant's -- pin the override to this variant.
    export RAIDEN_PYWRAP_SONAME="libpywrap_${SUFFIX}_common.so"
    ./build.sh torch
    unset RAIDEN_PYWRAP_SONAME
    cp /workspace/tpu_sync/frameworks/torch/_tpu_raiden_torch.so \
      "${EXT_DIR}/_tpu_raiden_torch_${SUFFIX}.so"
    echo "wheel variant: _tpu_raiden_torch_${SUFFIX}.so"
  done

  rm -f "${WHL}"
  wheel pack "${PKG_DIR}" -d /workspace/dist
fi
INNER_EOF

echo "===> Building ${BUILD_MODE} wheel in ${CONTAINER_IMAGE}..."
docker run --rm \
  "${DOCKER_MOUNTS[@]}" \
  -w /workspace \
  -e WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS}" \
  -e RAIDEN_EXTRA_TORCH_ABIS="${RAIDEN_EXTRA_TORCH_ABIS:-}" \
  -e BUILD_MODE="${BUILD_MODE}" \
  "${CONTAINER_IMAGE}" \
  bash -c "${INNER}"

# Scope to THIS build's wheel(s) (.dev<timestamp>); REPO_ROOT/dist and WHEEL_DIR
# are persistent and may hold wheels from earlier runs.
if [[ -n "$(ls -A "${REPO_ROOT}"/dist/*"${WHEEL_VERSION_EXTRAS}"-*.whl 2>/dev/null)" ]]; then
  cp "${REPO_ROOT}"/dist/*"${WHEEL_VERSION_EXTRAS}"-*.whl "${WHEEL_DIR}/"
  echo "===> Wheel(s) built:"; ls -lh "${WHEEL_DIR}"/*"${WHEEL_VERSION_EXTRAS}"-*.whl
else
  echo "ERROR: wheel build produced no .whl for ${WHEEL_VERSION_EXTRAS} in dist/" >&2; exit 1
fi

echo "===> twine check..."
docker run --rm -v "${WHEEL_DIR}:/dist" "${CONTAINER_IMAGE}" \
  bash -c "uv run --isolated --with twine twine check /dist/*${WHEEL_VERSION_EXTRAS}-*.whl"
echo "===> raiden wheel build successful!"
