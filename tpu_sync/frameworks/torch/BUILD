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

load("@nanobind_bazel//:build_defs.bzl", "nanobind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@rules_python//python:defs.bzl", "py_test")
load("//tpu_sync/core:header_only_cc_info.bzl", "header_only_cc_info")

package(default_visibility = ["//visibility:public"])

# Targets tagged "nobuilder" compile against a torch_tpu source checkout and
# are built only by the external wheel build (build:oss in .bazelrc).
header_only_cc_info(
    name = "torch_tpu_device_buffer_headers",
    tags = ["nobuilder"],
    deps = [
        "@torch_tpu//torch_tpu/csrc/eager:device_buffer",
    ],
)

header_only_cc_info(
    name = "torch_tpu_tensor_to_buffer_headers",
    tags = ["nobuilder"],
    deps = [
        "@torch_tpu//torch_tpu/csrc/eager:tensor_to_buffer",
    ],
)

header_only_cc_info(
    name = "torch_tpu_materialize_headers",
    tags = ["nobuilder"],
    deps = [
        "@torch_tpu//torch_tpu/csrc/eager:materialize",
    ],
)

cc_library(
    name = "weight_synchronizer_torch",
    srcs = ["weight_synchronizer.cc"],
    hdrs = ["weight_synchronizer.h"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    linkopts = ["-Wl,--allow-shlib-undefined"],
    tags = ["nobuilder"],
    visibility = ["//visibility:public"],
    deps = [
        # buildcleaner: keep
        ":torch_tpu_device_buffer_headers",
        ":torch_tpu_tensor_to_buffer_headers",
        ":torch_utils",
        "@torch_tpu//shims/torch:aten_headers",
        "//tpu_sync/weight_sync:weight_synchronizer_base",
    ],
)

# Removed _weight_synchronizer (consolidated into _tpu_raiden_torch)

cc_library(
    name = "torch_tpu_utils",
    srcs = ["torch_tpu_utils.cc"],
    hdrs = ["torch_tpu_utils.h"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    tags = ["nobuilder"],
    deps = [
        ":torch_tpu_device_buffer_headers",
        ":torch_tpu_materialize_headers",
        ":torch_tpu_tensor_to_buffer_headers",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:xla_raw_transfer_headers",
        "@com_google_absl//absl/strings",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
        "@torch_tpu//shims/torch/torch/headeronly:torch_headeronly",
    ],
)

cc_library(
    name = "torch_utils",
    srcs = ["torch_utils.cc"],
    hdrs = ["torch_utils.h"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    tags = ["nobuilder"],
    deps = [
        ":torch_tpu_tensor_to_buffer_headers",
        ":torch_tpu_utils",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:xla_raw_transfer_headers",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
    ],
)

cc_library(
    name = "torch_nanobind_utils",
    hdrs = ["torch_nanobind_utils.h"],
    copts = [
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
    ],
    tags = ["nobuilder"],
    visibility = ["//visibility:public"],
    deps = [
        "@nanobind",
        "@pybind11",  # buildcleaner: keep
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
    ],
)

cc_library(
    name = "pool_layout_nanobind",
    hdrs = ["pool_layout_nanobind.h"],
    copts = [
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/kv_cache:pool_layout",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@nanobind",
    ],
)

cc_library(
    name = "kv_cache_manager_torch",
    srcs = ["kv_cache_manager.cc"],
    hdrs = ["kv_cache_manager.h"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    linkopts = ["-Wl,--allow-shlib-undefined"],
    tags = ["nobuilder"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_tpu_tensor_to_buffer_headers",
        ":torch_utils",
        "//tpu_sync/core:kv_cache_manager_with_transfer",
        "//tpu_sync/core:kv_manager_holder",
        "//tpu_sync/core:tpu_utils",
        "//tpu_sync/core:utils",
        "//tpu_sync/core:xla_raw_transfer_headers",
        "//tpu_sync/core/controller:controller_client",
        "//tpu_sync/core/controller:worker_service_server",
        "//tpu_sync/kv_cache:kv_cache_listener",
        "//tpu_sync/transport:block_transport",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@torch_tpu//shims/torch:aten_headers",
    ],
)

nanobind_extension(
    name = "_tpu_raiden_torch",
    srcs = ["tpu_raiden_torch_module.cc"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    linkopts = ["-Wl,--allow-shlib-undefined"],
    tags = ["nobuilder"],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_torch",
        ":pool_layout_nanobind",
        ":torch_nanobind_utils",
        ":weight_synchronizer_torch",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/kv_cache:kv_cache_manager_base",
        "//tpu_sync/kv_cache:kv_cache_store",
        "//tpu_sync/kv_cache:kv_cache_store_wrapper",
        "//tpu_sync/kv_cache/reshard:reshard_client",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "//tpu_sync/telemetry/python:telemetry_binding",
        "//tpu_sync/weight_sync:weight_synchronizer_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@nanobind",
        "@torch_tpu//shims/torch:aten_headers",
    ],
)

nanobind_extension(
    name = "_tpu_raiden_host",
    srcs = ["tpu_raiden_host_module.cc"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":pool_layout_nanobind",
        "//tpu_sync/core:kv_cache_manager_with_transfer",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/kv_cache:kv_cache_manager_base",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "//tpu_sync/transport:block_transport",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@nanobind",
    ],
)

cc_library(
    name = "torch_raw_transfer_torch",
    srcs = ["torch_raw_transfer.cc"],
    hdrs = ["torch_raw_transfer.h"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = ["-use_header_modules"],
    tags = ["nobuilder"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_tpu_device_buffer_headers",
        ":torch_tpu_tensor_to_buffer_headers",
        ":torch_tpu_utils",
        "//tpu_sync/core:host_memory_allocator",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:utils",
        "//tpu_sync/core:xla_raw_transfer_headers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch/c10/core:base_headers",
        "@torch_tpu//shims/torch/torch/headeronly:torch_headeronly",
    ],
    alwayslink = 1,
)

nanobind_extension(
    name = "_torch_raw_transfer",
    srcs = ["torch_raw_transfer_module.cc"],
    features = ["-use_header_modules"],
    linkopts = ["-Wl,--allow-shlib-undefined"],
    tags = ["nobuilder"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_nanobind_utils",
        ":torch_raw_transfer_torch",
        "//tpu_sync/core:host_memory_allocator",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:utils",
        "//tpu_sync/core:xla_raw_transfer_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@torch_tpu//shims/torch:aten_headers",
    ],
)

cc_library(
    name = "torch_raw_transfer_torch_mock",
    srcs = ["torch_raw_transfer.cc"],
    hdrs = ["torch_raw_transfer.h"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = ["-use_header_modules"],
    tags = ["nobuilder"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_tpu_device_buffer_headers",
        ":torch_tpu_tensor_to_buffer_headers",
        ":torch_tpu_utils_mock",
        "//tpu_sync/core:host_memory_allocator",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:utils",
        "//tpu_sync/core:xla_raw_transfer_headers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch/c10/core:base_headers",
        "@torch_tpu//shims/torch/torch/headeronly:torch_headeronly",
    ],
)

cc_library(
    name = "torch_tpu_utils_mock",
    srcs = ["torch_tpu_utils_mock.cc"],
    hdrs = [
        "torch_tpu_utils.h",
        "torch_tpu_utils_mock.h",
    ],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = ["-use_header_modules"],
    tags = ["nobuilder"],
    deps = [
        ":torch_tpu_device_buffer_headers",
        "//tpu_sync/core:raw_transfer_core",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "torch_utils_mock",
    srcs = ["torch_utils.cc"],
    hdrs = ["torch_utils.h"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = ["-use_header_modules"],
    tags = ["nobuilder"],
    deps = [
        ":torch_tpu_tensor_to_buffer_headers",
        ":torch_tpu_utils_mock",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:xla_raw_transfer_headers",
        "@torch_tpu//shims/torch:aten_headers",
    ],
)

cc_library(
    name = "kv_cache_manager_torch_mock",
    srcs = ["kv_cache_manager.cc"],
    hdrs = ["kv_cache_manager.h"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = ["-use_header_modules"],
    tags = ["nobuilder"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_tpu_tensor_to_buffer_headers",
        ":torch_utils_mock",
        "//tpu_sync/core:kv_cache_manager_with_transfer",
        "//tpu_sync/core:kv_manager_holder",
        "//tpu_sync/core:tpu_utils",
        "//tpu_sync/core:utils",
        "//tpu_sync/core:xla_raw_transfer_headers",
        "//tpu_sync/core/controller:controller_client",
        "//tpu_sync/core/controller:worker_service_server",
        "//tpu_sync/kv_cache:kv_cache_listener",
        "//tpu_sync/transport:block_transport",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
    ],
)

cc_library(
    name = "weight_synchronizer_torch_mock",
    srcs = ["weight_synchronizer.cc"],
    hdrs = ["weight_synchronizer.h"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = ["-use_header_modules"],
    tags = ["nobuilder"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_tpu_tensor_to_buffer_headers",
        ":torch_utils_mock",
        "//tpu_sync/weight_sync:weight_synchronizer_base",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
    ],
)

cc_test(
    name = "kv_cache_manager_test",
    srcs = ["kv_cache_manager_test.cc"],
    copts = [
        "-std=c++20",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    tags = ["nobuilder"],
    deps = [
        ":kv_cache_manager_torch_mock",
        ":torch_tpu_utils_mock",
        "//tpu_sync/core/controller:controller_service",
        "//tpu_sync/core/controller:test_util",
        "@com_google_absl//absl/strings",
        "@torch_tpu//shims/torch",
        "@torch_tpu//shims/torch:torch_headers",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_test(
    name = "kv_cache_manager_transfer_test",
    srcs = ["kv_cache_manager_transfer_test.cc"],
    copts = [
        "-std=c++20",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    tags = ["nobuilder"],
    deps = [
        ":kv_cache_manager_torch_mock",
        ":torch_tpu_utils_mock",
        "@torch_tpu//shims/torch",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
        "@torch_tpu//shims/torch/c10/core:base_headers",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_test(
    name = "torch_raw_transfer_test",
    srcs = ["torch_raw_transfer_test.cc"],
    copts = [
        "-std=c++20",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    tags = ["nobuilder"],
    deps = [
        ":torch_raw_transfer_torch_mock",
        ":torch_tpu_utils_mock",
        "@torch_tpu//shims/torch",
        "@torch_tpu//shims/torch:torch_headers",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

py_test(
    name = "kv_cache_manager_test_py",
    srcs = ["kv_cache_manager_test.py"],
    main = "kv_cache_manager_test.py",
    tags = [
        "cpu:8",
        "nobuilder",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":_tpu_raiden_torch",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
        "@torch_tpu//torch_tpu/_internal/execution_mode",
        "@torch_tpu//torch_tpu/_internal/sync",
    ],
)

py_test(
    name = "torch_raw_transfer_perf_test_gl",
    timeout = "long",
    srcs = ["torch_raw_transfer_perf_test.py"],
    main = "torch_raw_transfer_perf_test.py",
    tags = [
        "cpu:8",
        "nobuilder",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":_torch_raw_transfer",
        "//pyglib/contrib/g3_multiprocessing",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

py_test(
    name = "torch_raw_transfer_perf_test_gf",
    timeout = "eternal",
    srcs = ["torch_raw_transfer_perf_test.py"],
    main = "torch_raw_transfer_perf_test.py",
    tags = [
        "cpu:8",
        "nobuilder",
        "notap",
        "requires-accel-hbm",
        "requires-ghostfish:4",
        "requires-net:external",
    ],
    deps = [
        ":_torch_raw_transfer",
        "//pyglib/contrib/g3_multiprocessing",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

test_suite(
    name = "torch_raw_transfer_perf_test",
    tags = ["nobuilder"],
    tests = [
        ":torch_raw_transfer_perf_test_gf",
        ":torch_raw_transfer_perf_test_gl",
    ],
)

py_test(
    name = "torch_raw_transfer_test_gl",
    timeout = "long",
    srcs = ["torch_raw_transfer_test.py"],
    main = "torch_raw_transfer_test.py",
    tags = [
        "cpu:8",
        "nobuilder",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":_torch_raw_transfer",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

py_test(
    name = "torch_raw_transfer_test_gf",
    timeout = "long",
    srcs = ["torch_raw_transfer_test.py"],
    main = "torch_raw_transfer_test.py",
    tags = [
        "cpu:8",
        "nobuilder",
        "notap",
        "requires-accel-hbm",
        "requires-ghostfish:4",
        "requires-net:external",
    ],
    deps = [
        ":_torch_raw_transfer",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

test_suite(
    name = "torch_raw_transfer_test_py",
    tags = ["nobuilder"],
    tests = [
        ":torch_raw_transfer_test_gf",
        ":torch_raw_transfer_test_gl",
    ],
)
