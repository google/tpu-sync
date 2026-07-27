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
# Builds one tpu_sync wheel hermetically inside the ml-build container
# (glibc 2.35, matching the TPU runtime), mirroring torch_tpu/ci/build_wheel.sh.
#
# The in-container work (clang-18 install, torch per extension variant,
# build.sh for the wheel target, per-torch-ABI repack) lives in
# ci/build_wheel_impl.sh. This script mounts the checkout and the bazel cache
# into the container and runs that script there; GitHub Actions runs the same
# script directly inside a job container. No torch_tpu checkout is involved:
# build.sh resolves the torch_tpu module from bazel/torch_tpu_standin.
#
# One wheel per invocation:
#   ci/build_wheel.sh [jax]            # tpu_sync_jax
#   ci/build_wheel.sh torch            # tpu_sync_torch
#
# WHEEL_VERSION_EXTRAS defaults to .dev<UTC timestamp> when unset; an empty
# value builds a wheel with the bare pyproject.toml version (a release).
# RAIDEN_TORCH_ABIS lists the torch releases the torch wheel ships an
# extension variant for (see ci/build_wheel_impl.sh).

set -exu -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS-.dev$(date -u +%Y%m%d%H%M%S)}"
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

RAIDEN_TORCH_ABIS="${RAIDEN_TORCH_ABIS:-2.11.0 2.12.0 2.13.0}"
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

echo "===> Building ${BUILD_MODE} wheel in ${CONTAINER_IMAGE}..."
docker run --rm \
  "${DOCKER_MOUNTS[@]}" \
  -w /workspace \
  -e BUILD_MODE="${BUILD_MODE}" \
  -e WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS}" \
  -e RAIDEN_TORCH_ABIS="${RAIDEN_TORCH_ABIS}" \
  -e BAZEL_CACHE_DIR=/cache \
  "${CONTAINER_IMAGE}" \
  bash ci/build_wheel_impl.sh

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
