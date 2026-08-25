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

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Point to the directory containing the compiled extension libraries and source files
export PYTHONPATH="${WORKSPACE_DIR}:${WORKSPACE_DIR}/bazel-bin:${PYTHONPATH}"

RUN_JAX=true
RUN_TORCH=true

if [ "$#" -gt 0 ]; then
  case "$1" in
    jax)
      RUN_JAX=true
      RUN_TORCH=false
      shift
      ;;
    torch)
      RUN_JAX=false
      RUN_TORCH=true
      shift
      ;;
    both)
      RUN_JAX=true
      RUN_TORCH=true
      shift
      ;;
    -*)
      ;;
    *)
      echo "Usage: $0 [jax|torch|both]"
      exit 1
      ;;
  esac
fi

if [ "$RUN_JAX" = true ]; then
  echo "=== Running JAX Python Unit Tests ==="
  python "${WORKSPACE_DIR}/tpu_sync/frameworks/jax/kv_cache_manager_test.py"
  python "${WORKSPACE_DIR}/tpu_sync/frameworks/jax/resharding_planner_test.py"
  python "${WORKSPACE_DIR}/tpu_sync/api/jax/kv_cache_manager_test.py"
  python "${WORKSPACE_DIR}/tpu_sync/api/jax/kv_cache_store_test.py"
  python "${WORKSPACE_DIR}/tpu_sync/api/jax/kv_cache_manager_transfer_test.py"
  python "${WORKSPACE_DIR}/tpu_sync/api/jax/weight_synchronizer_test.py"
  python "${WORKSPACE_DIR}/tpu_sync/api/jax/kv_cache_manager_mpmd_test.py"
fi

if [ "$RUN_TORCH" = true ]; then
  echo "=== Running Torch Python Unit Tests ==="
  python "${WORKSPACE_DIR}/tpu_sync/frameworks/torch/kv_cache_manager_test.py"
  python "${WORKSPACE_DIR}/tpu_sync/api/torch/kv_cache_manager_test.py"
  python "${WORKSPACE_DIR}/tpu_sync/api/torch/kv_cache_store_test.py"
  python "${WORKSPACE_DIR}/tpu_sync/api/torch/kv_cache_manager_transfer_test.py"
  python "${WORKSPACE_DIR}/tpu_sync/api/torch/weight_synchronizer_test.py"
  python "${WORKSPACE_DIR}/tpu_sync/api/torch/kv_cache_manager_mpmd_test.py"
fi
