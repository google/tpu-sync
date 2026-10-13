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
# kokoro/gcp_ubuntu/build_docker.sh
# Kokoro entrypoint script for building and pushing custom Docker image.
set -e
set -o pipefail
set -x

# Resolve paths
export WORK_DIR="${KOKORO_ARTIFACTS_DIR}/workspace"
mkdir -p "${WORK_DIR}"

export REPO_ROOT="${KOKORO_ARTIFACTS_DIR}/github/tpu-raiden"
cd "${REPO_ROOT}"

# Default registry and image name
# Customize these via Kokoro environment variables if needed
REGISTRY="${DOCKER_REGISTRY:-us-docker.pkg.dev/cloud-tpu-inference-test/tpu-raiden}"
IMAGE_NAME="${DOCKER_IMAGE_NAME:-ml-build-custom}"
TAG="${DOCKER_TAG:-latest}"

# If running as presubmit (GitHub PR), tag with PR number
if [[ -n "${KOKORO_GITHUB_PULL_REQUEST_NUMBER}" ]]; then
  TAG="pr-${KOKORO_GITHUB_PULL_REQUEST_NUMBER}"
elif [[ -n "${KOKORO_PIPER_CHANGELIST}" ]]; then
  TAG="cl-${KOKORO_PIPER_CHANGELIST}"
fi

FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "=== Building Docker image: ${FULL_IMAGE_NAME} ==="
docker build -t "${FULL_IMAGE_NAME}" .

echo "=== Authenticating to Artifact Registry ==="
# Assuming the environment has credentials or is configured to access the registry
# For Google Artifact Registry:
gcloud auth configure-docker us-docker.pkg.dev --quiet

echo "=== Pushing Docker image ==="
docker push "${FULL_IMAGE_NAME}"

echo "=== Docker image pushed successfully! ==="
