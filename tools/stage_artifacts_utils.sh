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
#
# Shared Bazel target lists and post-build source-tree staging helpers for
# build.sh and ci/run_unit_tests.sh.
# Source this file; do not run it directly.

RAIDEN_PROTO_BAZEL_TARGETS=(
  "//tpu_sync/rpc:raiden_service_py_pb2"
  "//tpu_sync/rpc:coordination_py_pb2"
  "//tpu_sync/proto:control_pipe_py_pb2"
  "//tpu_sync/proto:control_pipe_py_pb2_grpc"
  "//tpu_sync/rpc:coordination_py_pb2_grpc"
  "//tpu_sync/rpc:controller_service_py_pb2"
)

# C++ control-plane service binaries. These do not depend on JAX or Torch.
RAIDEN_SERVER_BAZEL_TARGETS=(
  "//tpu_sync/kv_cache/global_registry:global_registry_server"
  "//tpu_sync/store_node:kv_cache_host_store_node_main"
)

RAIDEN_JAX_EXTENSION_BAZEL_TARGETS=(
  "//tpu_sync/frameworks/jax:_tpu_raiden_jax"
  "//tpu_sync/frameworks/jax:_weight_synchronizer_ffi"
)

RAIDEN_JAX_E2E_BAZEL_TARGETS=(
  "${RAIDEN_PROTO_BAZEL_TARGETS[@]}"
  "${RAIDEN_SERVER_BAZEL_TARGETS[@]}"
  "${RAIDEN_JAX_EXTENSION_BAZEL_TARGETS[@]}"
)

# Copies generated protobuf Python modules from bazel-bin into the source tree.
raiden_stage_proto_artifacts() {
  local workspace_dir="$1"
  echo "=== Copying generated protobuf Python modules ==="
  cp -f "${workspace_dir}/bazel-bin/tpu_sync/rpc/raiden_service_pb2.py" "${workspace_dir}/tpu_sync/rpc/" 2>/dev/null || true
  cp -f "${workspace_dir}/bazel-bin/tpu_sync/rpc/coordination_pb2.py" "${workspace_dir}/tpu_sync/rpc/" 2>/dev/null || true
  cp -f "${workspace_dir}/bazel-bin/tpu_sync/rpc/coordination_pb2_grpc.py" "${workspace_dir}/tpu_sync/rpc/" 2>/dev/null || true
  cp -f "${workspace_dir}/bazel-bin/tpu_sync/rpc/controller_service_pb2.py" "${workspace_dir}/tpu_sync/rpc/" 2>/dev/null || true
  cp -f "${workspace_dir}/bazel-bin/tpu_sync/proto/control_pipe_pb2.py" "${workspace_dir}/tpu_sync/proto/" 2>/dev/null || true
  cp -f "${workspace_dir}/bazel-bin/tpu_sync/proto/control_pipe_pb2_grpc.py" "${workspace_dir}/tpu_sync/proto/" 2>/dev/null || true
}

# Symlinks C++ control-plane service binaries from bazel-bin into the source
# tree. KVCacheStore tests (kv_cache_store_test.py / kv_cache_store_e2e_test.py)
# launch these binaries from their source-tree path, but Bazel emits them under
# bazel-bin/. Symlink them so a fresh checkout can run the tests without extra
# steps.
raiden_stage_server_binaries() {
  local workspace_dir="$1"
  echo "=== Linking C++ control-plane service binaries into source tree ==="
  local rel src dst
  for rel in \
    "tpu_sync/kv_cache/global_registry/global_registry_server" \
    "tpu_sync/store_node/kv_cache_host_store_node_main"; do
    src="${workspace_dir}/bazel-bin/${rel}"
    dst="${workspace_dir}/${rel}"
    if [[ -e "${src}" ]]; then
      ln -sf "${src}" "${dst}"
    fi
  done
}

# Copies compiled JAX shared libraries from bazel-bin into the source tree.
raiden_stage_jax_artifacts() {
  local workspace_dir="$1"
  echo "Copying JAX artifacts..."
  cp -f "${workspace_dir}/bazel-bin/tpu_sync/frameworks/jax/_tpu_raiden_jax.so" "${workspace_dir}/tpu_sync/frameworks/jax/"
  cp -f "${workspace_dir}/bazel-bin/tpu_sync/frameworks/jax/_weight_synchronizer_ffi.so" "${workspace_dir}/tpu_sync/frameworks/jax/"
}

# Stages all artifacts produced by a full JAX build (protos, server binaries,
# and JAX shared libraries) into the source tree.
raiden_stage_jax_e2e_artifacts() {
  local workspace_dir="$1"
  raiden_stage_proto_artifacts "${workspace_dir}"
  raiden_stage_server_binaries "${workspace_dir}"
  raiden_stage_jax_artifacts "${workspace_dir}"
}
