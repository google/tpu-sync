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

load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "torch_abi",
    srcs = ["torch_abi.py"],
)

# Targets tagged "nobuilder" compile against a torch_tpu source checkout and
# are built only by the external wheel build (build:oss in .bazelrc).
py_library(
    name = "torch_tpu_common_loader",
    srcs = ["torch_tpu_common_loader.py"],
    tags = ["nobuilder"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_abi",
        "@torch_tpu//torch_tpu",
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
    tags = ["nobuilder"],
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
    name = "kv_cache_manager_host_py",
    srcs = ["kv_cache_manager.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":pool_layout",
        "//tpu_sync/frameworks/torch:_tpu_raiden_host",
        "//tpu_sync/rpc:raiden_service_py_pb2",
    ],
)

py_test(
    name = "kv_cache_manager_host_test",
    srcs = ["kv_cache_manager_host_test.py"],
    deps = [
        ":kv_cache_manager_host_py",
        ":pool_layout",
    ],
)

py_library(
    name = "weight_synchronizer_torch_py",
    srcs = ["weight_synchronizer.py"],
    tags = ["nobuilder"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_abi",
        ":torch_tpu_common_loader",
        "@torch_tpu//shims/torch:pytorch",
        # buildcleaner: keep
        "//tpu_sync/frameworks/torch:_tpu_raiden_torch",
    ],
)

py_library(
    name = "kv_cache_store",
    srcs = ["kv_cache_store.py"],
    tags = ["nobuilder"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/api:common",
        ":torch_abi",
        ":torch_tpu_common_loader",
        "@torch_tpu//torch_tpu",
        # buildcleaner: keep
        "//tpu_sync/frameworks/torch:_tpu_raiden_torch",
    ],
)

py_library(
    name = "reshard_client",
    srcs = ["reshard_client.py"],
    tags = ["nobuilder"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_abi",
        ":torch_tpu_common_loader",
        "//tpu_sync/frameworks/torch:_tpu_raiden_torch",
        "//tpu_sync/rpc:raiden_service_py_pb2",
        "@torch_tpu//torch_tpu",
    ],
)

py_library(
    name = "reshard_store",
    srcs = ["reshard_store.py"],
    tags = ["nobuilder"],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_store",
        ":torch_abi",
        ":torch_tpu_common_loader",
        "//tpu_sync/frameworks/torch:_tpu_raiden_torch",
    ],
)

py_test(
    name = "kv_cache_store_test",
    srcs = ["kv_cache_store_test.py"],
    data = [
        "//tpu_sync/kv_cache/global_registry:global_registry_server",
    ],
    tags = ["nobuilder"],
    deps = [
        ":kv_cache_store",
        "//pyglib:resources",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "weight_synchronizer_test",
    srcs = ["weight_synchronizer_test.py"],
    tags = [
        "cpu:8",
        "nobuilder",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":weight_synchronizer_torch_py",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

py_test(
    name = "kv_cache_store_recovery_e2e_test",
    srcs = ["kv_cache_store_recovery_e2e_test.py"],
    tags = [
        "cpu:8",
        "nobuilder",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":kv_cache_manager_torch_py",
        ":kv_cache_store",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/testing:absltest",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

py_test(
    name = "kv_cache_manager_transfer_test",
    srcs = ["kv_cache_manager_transfer_test.py"],
    tags = [
        "cpu:8",
        "nobuilder",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":kv_cache_manager_torch_py",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

py_test(
    name = "kv_cache_store_e2e_test",
    timeout = "long",
    srcs = ["kv_cache_store_e2e_test.py"],
    data = [
        "//tpu_sync/kv_cache/global_registry:global_registry_server",
    ],
    tags = [
        "manual",
    ],
    deps = [
        ":kv_cache_manager_torch_py",
        ":kv_cache_store",
        "//pyglib:resources",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

py_test(
    name = "kv_cache_store_mpmd_e2e_test",
    timeout = "long",
    srcs = ["kv_cache_store_mpmd_e2e_test.py"],
    data = [
        "//tpu_sync/kv_cache/global_registry:global_registry_server",
    ],
    tags = [
        "manual",
    ],
    deps = [
        ":kv_cache_manager_torch_py",
        ":kv_cache_store",
        "//pyglib:resources",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)
