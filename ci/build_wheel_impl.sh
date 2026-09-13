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
# In-container build of one tpu_sync wheel. Runs inside the ml-build container
# (glibc 2.35, matching the TPU runtime): either through the docker wrapper
# ci/build_wheel.sh, or directly as a GitHub Actions job step whose job
# declares the ml-build image as its container (see .github/workflows/).
# Requires root (installs clang-18 and libstdc++-12 through apt).
#
# One wheel per invocation, selected by BUILD_MODE:
#   BUILD_MODE=jax    tpu_sync_jax
#   BUILD_MODE=torch  tpu_sync_torch
# The wheel lands in <repo root>/dist/, named by WHEEL_VERSION_EXTRAS.
#
# The torch wheel needs no torch_tpu checkout: build.sh resolves the
# torch_tpu module from shims/torch_tpu, which takes the API
# header and the XLA pin from the torch_tpu wheel this script installs, and
# the extension binds torch_tpu's symbols at import to the installed wheel.
# With TORCH_TPU_CHECKOUT set, the torch wheel is built against that
# torch_tpu checkout instead (torch_tpu's own Kokoro job builds this way).
#
# Environment:
#   BUILD_MODE               jax (default) or torch
#   WHEEL_VERSION_EXTRAS     suffix appended to the pyproject.toml base version
#                            (default .dev<UTC timestamp>; set to the empty
#                            string for a release wheel)
#   RAIDEN_TORCH_ABIS        torch releases the torch wheel ships an extension
#                            variant for, space-separated, the first one being
#                            the torch the rest of the wheel is compiled
#                            against (default "2.11.0 2.12.0 2.13.0", the
#                            releases torch_tpu ships glue for; "2.11.0" alone
#                            builds a single-ABI wheel)
#   TORCH_TPU_INDEX_URL      pip index the torch_tpu wheel named by
#                            torch_tpu.version is installed from (default:
#                            the torch_tpu testing registry)
#   TORCH_TPU_CHECKOUT       a torch_tpu source checkout to build the torch
#                            wheel against instead of the installed wheel
#                            (optional)
#   BAZEL_CACHE_DIR          bazel disk/repo cache root (default /cache)
#   BAZEL_OUTPUT_BASE        bazel output base (default <cache>/output_base)
#   EXTRA_BAZEL_FLAGS        extra bazel flags, space-separated (optional)

set -exu -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BUILD_MODE="${BUILD_MODE:-jax}"
if [[ "${BUILD_MODE}" != "torch" && "${BUILD_MODE}" != "jax" ]]; then
  echo "ERROR: BUILD_MODE must be 'torch' or 'jax', got '${BUILD_MODE}'" >&2
  exit 1
fi
# Default only when UNSET: an empty-but-set WHEEL_VERSION_EXTRAS is a release
# wheel with the bare pyproject version.
WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS-.dev$(date -u +%Y%m%d%H%M%S)}"
export WHEEL_VERSION_EXTRAS
read -r -a TORCH_ABIS <<< "${RAIDEN_TORCH_ABIS:-2.11.0 2.12.0 2.13.0}"
export BAZEL_CACHE_DIR="${BAZEL_CACHE_DIR:-/cache}"
export BAZEL_OUTPUT_BASE="${BAZEL_OUTPUT_BASE:-${BAZEL_CACHE_DIR}/output_base}"
EXTRA_BAZEL_FLAGS="${EXTRA_BAZEL_FLAGS:-}"
export TORCH_TPU_INDEX_URL="${TORCH_TPU_INDEX_URL:-https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-testing-registry/simple/}"
echo "WHEEL_VERSION_EXTRAS: ${WHEEL_VERSION_EXTRAS}"

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
  if [[ ${#TORCH_ABIS[@]} -eq 0 ]]; then
    echo "ERROR: RAIDEN_TORCH_ABIS must name at least one torch release" >&2
    exit 1
  fi
  # raiden's _tpu_raiden_torch.so and torch_tpu's per-torch-version glue must
  # resolve the SAME libtorch symbols at runtime, so each extension variant is
  # compiled against the exact torch release it is named for, never a
  # floating specifier (torch 2.13.x, for one, drops symbols the 2.11 glue
  # needs, which shows up as an `undefined symbol` at import).
  echo "Installing torch ${TORCH_ABIS[0]} (primary variant)"
  pip install -q "torch==${TORCH_ABIS[0]}" --index-url https://download.pytorch.org/whl/cpu
  TORCH_SOURCE="$(python3 -c 'import torch,pathlib;print(pathlib.Path(torch.__file__).resolve().parent.parent)')"
  export TORCH_SOURCE
  if [[ -n "${TORCH_TPU_CHECKOUT:-}" ]]; then
    echo "Building against the torch_tpu checkout at ${TORCH_TPU_CHECKOUT}"
    export TORCH_TPU_MODULE_PATH="${TORCH_TPU_CHECKOUT}"
  else
    # The torch_tpu wheel this build pairs with: its torch_tpu/include/ holds
    # the API header and the XLA pin the extension compiles against. Only
    # those files are read at build time, so the wheel's own requirements
    # (torch, libtpu) are not installed.
    TORCH_TPU_VERSION="$(tr -d '\r\n ' < "${REPO_ROOT}/torch_tpu.version")"
    # The torch_tpu registry is read with a Google Cloud identity: pip takes it
    # from a netrc entry holding the environment's access token (written from
    # Python so the token never reaches the shell trace). --no-input turns a
    # missing credential into a failure instead of a prompt.
    if [[ "${TORCH_TPU_INDEX_URL}" == https://*.pkg.dev/* ]]; then
      python3 - <<'PY2'
import os, pathlib, subprocess
token = subprocess.run(["gcloud", "auth", "print-access-token"],
                       capture_output=True, text=True, check=True).stdout.strip()
host = os.environ["TORCH_TPU_INDEX_URL"].split("/")[2]
netrc = pathlib.Path.home() / ".netrc"
with netrc.open("a") as f:
  f.write(f"machine {host} login oauth2accesstoken password {token}\n")
netrc.chmod(0o600)
PY2
    fi
    echo "Installing torch_tpu ${TORCH_TPU_VERSION}"
    pip install --no-input --no-deps "torch_tpu==${TORCH_TPU_VERSION}" --index-url "${TORCH_TPU_INDEX_URL}"
    # With torch_tpu installed, `import torch` autoloads the torch_tpu
    # backend, whose runtime dependencies are not installed here; the build
    # only reads the wheel's header and pin.
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  fi
fi

# Separate per-framework wheels: tpu_sync_torch (no jax deps) vs tpu_sync_jax.
# Pick by BUILD_MODE.
if [[ "${BUILD_MODE}" == "torch" ]]; then
  WHEEL_TARGET="//ci/wheel:raiden_torch_wheel"
  WHEEL_DIST="tpu_sync_torch"
else
  WHEEL_TARGET="//ci/wheel:raiden_jax_wheel"
  WHEEL_DIST="tpu_sync_jax"
fi
# Match ONLY the wheel this build just produced. The output directory is shared
# across builds, so bazel-bin/ci/wheel/ accumulates wheels from earlier runs,
# each with a distinct .dev<timestamp>. A broad "${WHEEL_DIST}-*.whl" glob would
# also match those stale wheels and hand multiple paths to the single-wheel
# patchelf step below (which then fails). WHEEL_VERSION_EXTRAS (.dev<timestamp>)
# is unique per build and appears verbatim in the filename, so scope to it.
WHEEL_GLOB="${WHEEL_DIST}-*${WHEEL_VERSION_EXTRAS}-*.whl"

./build.sh "${BUILD_MODE}" "${WHEEL_TARGET}" \
  --repo_env=WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS}" \
  ${EXTRA_BAZEL_FLAGS}

mkdir -p "${REPO_ROOT}/dist"
# bazel-bin is the convenience symlink build.sh leaves in the workspace; it
# points at the output directory of whatever configuration the build used.
cp "${REPO_ROOT}"/bazel-bin/ci/wheel/${WHEEL_GLOB} "${REPO_ROOT}/dist/"

# The bazel-built _tpu_raiden_torch.so does not link libpywrap; the torch
# extension loader (tpu_sync/api/torch/torch_abi.py) requires a NEEDED on
# torch_tpu's per-torch-version glue so the torch_tpu symbols resolve in
# RTLD_LOCAL scope at import. The wheel ships one version-suffixed extension
# per torch release in RAIDEN_TORCH_ABIS (_tpu_raiden_torch_<v>.so);
# torch_abi.load_extension picks the variant matching the installed torch.
torch_suffix() {
  "$1" -c 'import torch, re; v = re.match(r"(\d+)\.(\d+)\.(\d+)", torch.__version__); print(f"{v.group(1)}_{v.group(2)}_{v.group(3)}")'
}
if [[ "${BUILD_MODE}" == "torch" ]]; then
  pip install -q wheel
  WHL="$(ls "${REPO_ROOT}"/dist/${WHEEL_GLOB} | head -1)"
  UNPACK_DIR="$(mktemp -d)"
  wheel unpack "${WHL}" -d "${UNPACK_DIR}"
  PKG_DIR="$(ls -d "${UNPACK_DIR}"/*/)"
  EXT_DIR="${PKG_DIR}tpu_sync/frameworks/torch"

  # Primary variant: the torch this wheel build compiled against.
  SUFFIX="$(torch_suffix python3)"
  mv "${EXT_DIR}/_tpu_raiden_torch.so" "${EXT_DIR}/_tpu_raiden_torch_${SUFFIX}.so"
  patchelf --add-needed "libpywrap_${SUFFIX}_common.so" \
    "${EXT_DIR}/_tpu_raiden_torch_${SUFFIX}.so"
  echo "wheel variant: _tpu_raiden_torch_${SUFFIX}.so (NEEDED libpywrap_${SUFFIX}_common.so)"

  # Extra variants: rebuild the extension against each additional torch in an
  # isolated venv (a fresh TORCH_SOURCE path forces the bazel torch repo to
  # re-resolve; an in-place pip swap at the same path would be reused stale).
  for V in "${TORCH_ABIS[@]:1}"; do
    python3 -m venv "/tmp/torch-abi-${V}"
    "/tmp/torch-abi-${V}/bin/pip" install -q "torch==${V}" \
      --index-url https://download.pytorch.org/whl/cpu
    TORCH_SOURCE="$("/tmp/torch-abi-${V}/bin/python3" -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).resolve().parent.parent)')"
    export TORCH_SOURCE
    SUFFIX="$(torch_suffix "/tmp/torch-abi-${V}/bin/python3")"
    # build.sh derives the glue suffix from the system python's torch, which
    # is still the primary variant's -- pin the override to this variant.
    export RAIDEN_PYWRAP_SONAME="libpywrap_${SUFFIX}_common.so"
    ./build.sh torch ${EXTRA_BAZEL_FLAGS}
    unset RAIDEN_PYWRAP_SONAME
    cp "${REPO_ROOT}/tpu_sync/frameworks/torch/_tpu_raiden_torch.so" \
      "${EXT_DIR}/_tpu_raiden_torch_${SUFFIX}.so"
    echo "wheel variant: _tpu_raiden_torch_${SUFFIX}.so"
  done

  rm -f "${WHL}"
  wheel pack "${PKG_DIR}" -d "${REPO_ROOT}/dist"
fi

echo "===> Wheel(s) built:"
ls -lh "${REPO_ROOT}"/dist/${WHEEL_GLOB}
