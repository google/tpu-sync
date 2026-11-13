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

load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "kv_cache_manager_host_test",
    srcs = ["kv_cache_manager_host_test.py"],
    deps = [
        "//tpu_sync/api/torch:kv_cache_manager_host_py",
        "//tpu_sync/api/torch:pool_layout",
    ],
)

py_test(
    name = "reshard_store_test",
    srcs = ["reshard_store_test.py"],
    tags = ["nobuilder"],
    deps = [
        "//tpu_sync/api/torch:reshard_client",
        "//tpu_sync/api/torch:reshard_store",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "kv_cache_store_test",
    srcs = ["kv_cache_store_test.py"],
    data = [
        "//tpu_sync/kv_cache/global_registry:global_registry_server",
    ],
    deps = [
        "//pyglib:resources",
        "//tpu_sync/api/torch:kv_cache_store",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "weight_synchronizer_test",
    srcs = ["weight_synchronizer_test.py"],
    tags = [
        "cpu:8",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        "//tpu_sync/api/torch:weight_synchronizer_torch_py",
        "//tpu_sync/common/control_pipe:control_pipe_client",
        "//tpu_sync/rpc:raiden_controller",
        "//tpu_sync/rpc:raiden_service_py_pb2",
        "@com_google_absl_py//absl/logging",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//src/torch_tpu",
    ],
)

# Base accelerator test requiring physical TPU devices via torch_tpu.
# Tagged "manual" because generic CPU Forge runners lack TPU hardware.
# For remote execution on Forge, use the _gf (TPU v4) or _gl (TPU v5e) targets below.
py_test(
    name = "kv_cache_store_recovery_e2e_test",
    srcs = ["kv_cache_store_recovery_e2e_test.py"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tpu_sync/api/torch:kv_cache_manager_torch_py",
        "//tpu_sync/api/torch:kv_cache_store",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/testing:absltest",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//src/torch_tpu",
    ],
)

py_test(
    name = "kv_cache_manager_transfer_test",
    srcs = ["kv_cache_manager_transfer_test.py"],
    tags = [
        "cpu:8",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        "//tpu_sync/api/torch:kv_cache_manager_torch_py",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//src/torch_tpu",
    ],
)

# Note: blaze shares one FaultInjector across extensions, so only
# test_wrapper_resolves_to_torch_extension guards the wheel's wiring bug.
py_test(
    name = "kv_cache_manager_fault_injection_test",
    srcs = ["kv_cache_manager_fault_injection_test.py"],
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        "//tpu_sync/api/torch:fault_injection",
        "//tpu_sync/api/torch:kv_cache_manager_torch_py",
        "@com_google_absl_py//absl/testing:absltest",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//src/torch_tpu",
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
        "notap",
    ],
    deps = [
        "//pyglib:resources",
        "//tpu_sync/api/torch:kv_cache_manager_torch_py",
        "//tpu_sync/api/torch:kv_cache_store",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//src/torch_tpu",
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
        "notap",
    ],
    deps = [
        "//pyglib:resources",
        "//tpu_sync/api/torch:kv_cache_manager_torch_py",
        "//tpu_sync/api/torch:kv_cache_store",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//src/torch_tpu",
    ],
)
