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

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tiling_utils",
    srcs = ["tiling_utils.cc"],
    hdrs = ["tiling_utils.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//tpu_sync/core:numa_thread_pool",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@highway//:hwy",
        "@xla//xla:shape_util",
        "@xla//xla:util",
    ],
)

cc_library(
    name = "weight_synchronizer_base",
    srcs = [
        "weight_synchronization_worker_service.cc",
        "weight_synchronizer_base.cc",
        "weight_synchronizer_listener.cc",
    ],
    hdrs = [
        "weight_synchronization_worker_service.h",
        "weight_synchronizer_base.h",
        "weight_synchronizer_listener.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":tiling_utils",
        "//tpu_sync/core:host_memory_allocator",
        "//tpu_sync/core:numa_thread_pool",
        "//tpu_sync/core:raiden_manager_base",
        "//tpu_sync/core:raiden_transfer_endpoint",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:status_macros",
        "//tpu_sync/core:xla_raw_transfer_headers",
        "//tpu_sync/rpc:raiden_service_cc_grpc",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "//tpu_sync/transport:buffer_push_task",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = 1,
)

cc_test(
    name = "weight_synchronizer_test",
    srcs = [
        "weight_synchronizer_test.cc",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    linkopts = ["-Wl,--allow-shlib-undefined"],
    deps = [
        ":weight_synchronizer_base",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
    ],
)

cc_test(
    name = "weight_synchronizer_listener_test",
    srcs = ["weight_synchronizer_listener_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":weight_synchronization_worker_service_client",
        ":weight_synchronizer_base",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tiling_utils_test",
    srcs = ["tiling_utils_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":tiling_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@xla//xla:shape_util",
    ],
)

cc_library(
    name = "weight_synchronization_worker_service",
    hdrs = ["weight_synchronization_worker_service.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":weight_synchronizer_base",
        "//tpu_sync/rpc:raiden_service_cc_grpc",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

cc_test(
    name = "weight_synchronization_worker_service_test",
    srcs = ["weight_synchronization_worker_service_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":weight_synchronization_worker_service",
        ":weight_synchronizer_base",
        "//tpu_sync/rpc:raiden_service_cc_grpc",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "weight_synchronization_worker_service_client",
    srcs = ["weight_synchronization_worker_service_client.cc"],
    hdrs = ["weight_synchronization_worker_service_client.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/rpc:raiden_service_cc_grpc",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@xla//xla/tsl/concurrency:future",
    ],
)

cc_test(
    name = "weight_synchronization_worker_service_client_test",
    srcs = ["weight_synchronization_worker_service_client_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":weight_synchronization_worker_service",
        ":weight_synchronization_worker_service_client",
        ":weight_synchronizer_base",
        "//tpu_sync/rpc:raiden_service_cc_grpc",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/concurrency:future",
    ],
)

py_library(
    name = "weight_synchronization_worker_service_client_py",
    srcs = ["weight_synchronization_worker_service_client.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/rpc:raiden_service_py_pb2",
        "//tpu_sync/rpc:raiden_service_py_pb2_grpc",
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
    ],
)

py_test(
    name = "weight_synchronization_worker_service_client_py_test",
    srcs = ["weight_synchronization_worker_service_client_test.py"],
    main = "weight_synchronization_worker_service_client_test.py",
    deps = [
        ":weight_synchronization_worker_service_client_py",
        "//net/grpc/python:use_insecure_channel_for_test",
        "//tpu_sync/rpc:raiden_service_py_pb2",
        "//tpu_sync/rpc:raiden_service_py_pb2_grpc",
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)
