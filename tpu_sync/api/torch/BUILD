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

load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "torch_abi",
    srcs = ["torch_abi.py"],
)

py_library(
    name = "torch_tpu_common_loader",
    srcs = ["torch_tpu_common_loader.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_abi",
        "@torch_tpu//src/torch_tpu",
    ],
)

py_library(
    name = "pool_layout",
    srcs = ["pool_layout.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "kv_cache_manager_torch_py",
    srcs = ["kv_cache_manager.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":pool_layout",
        ":torch_abi",
        ":torch_tpu_common_loader",
        "//tpu_sync/frameworks/torch:_tpu_raiden_torch",
        "//tpu_sync/rpc:raiden_service_py_pb2",
    ],
)

py_library(
    name = "raw_transfer",
    srcs = ["raw_transfer.py"],
    tags = ["nobuilder"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_abi",
        ":torch_tpu_common_loader",
        "//tpu_sync/frameworks/torch:_tpu_raiden_torch",
    ],
)

# Shim for _tpu_raiden_torch.fault_injection (no extra .so).
py_library(
    name = "fault_injection",
    srcs = ["fault_injection.py"],
    tags = ["nobuilder"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_abi",
        ":torch_tpu_common_loader",
        "//tpu_sync/frameworks/torch:_tpu_raiden_torch",
    ],
)

py_library(
    name = "kv_cache_manager_host_py",
    srcs = ["kv_cache_manager.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":pool_layout",
        "//tpu_sync/frameworks/torch:_tpu_raiden_host",
        "//tpu_sync/rpc:raiden_service_py_pb2",
    ],
)

py_library(
    name = "weight_synchronizer_torch_py",
    srcs = ["weight_synchronizer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_abi",
        ":torch_tpu_common_loader",
        "@torch_tpu//shims/torch:pytorch",
        # buildcleaner: keep
        "//tpu_sync/frameworks/torch:_tpu_raiden_torch",
        "//tpu_sync/api:common",
        "//tpu_sync/api:weight_sync_common",
    ],
)

py_library(
    name = "kv_cache_store",
    srcs = ["kv_cache_store.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/api:common",
        ":torch_abi",
        ":torch_tpu_common_loader",
        "@torch_tpu//src/torch_tpu",
        # buildcleaner: keep
        "//tpu_sync/frameworks/torch:_tpu_raiden_torch",
    ],
)

py_library(
    name = "reshard_client",
    srcs = ["reshard_client.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_abi",
        ":torch_tpu_common_loader",
        "//tpu_sync/frameworks/torch:_tpu_raiden_torch",
        "//tpu_sync/rpc:raiden_service_py_pb2",
        "@torch_tpu//src/torch_tpu",
    ],
)

py_library(
    name = "reshard_store",
    srcs = ["reshard_store.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_abi",
        ":torch_tpu_common_loader",
        "//tpu_sync/api:common",
        "//tpu_sync/frameworks/torch:_tpu_raiden_torch",
    ],
)
