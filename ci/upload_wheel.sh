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
# Uploads the tpu_raiden wheel to Google Artifact Registry via twine, mirroring
# torch_tpu/ci/upload_wheel.sh.

set -exu -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEEL_DIR="${KOKORO_ARTIFACTS_DIR:-${HOME}/raiden_artifacts}/dist"

# Dedicated raiden Python registry in the cloud-tpu-inference-test project.
REGISTRY_URL="${RAIDEN_REGISTRY_URL:-https://us-python.pkg.dev/cloud-tpu-inference-test/tpu-raiden/}"
CONTAINER_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest"

export UPLOAD_WHEEL_TO_AR="${UPLOAD_WHEEL_TO_AR:-true}"
if [[ "${UPLOAD_WHEEL_TO_AR}" == "true" ]]; then
  echo "===> Uploading tpu_raiden wheel(s) to ${REGISTRY_URL} ..."
  docker run --rm \
    -v "${WHEEL_DIR}:/dist" \
    -v "${HOME}/.config/gcloud:/root/.config/gcloud:ro" \
    "${CONTAINER_IMAGE}" \
    bash -c "
      uv run --isolated \
        --with twine \
        --with keyrings.google-artifactregistry-auth \
        twine upload --repository-url ${REGISTRY_URL} /dist/tpu_*.whl
    "
  echo "===> Upload complete."
fi