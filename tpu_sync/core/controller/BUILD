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
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "worker_service_client",
    srcs = ["worker_service_client.cc"],
    hdrs = ["worker_service_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/proto:worker_service_cc_grpc",
        "//tpu_sync/proto:worker_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@xla//xla/tsl/concurrency:future",
    ],
)

cc_library(
    name = "worker_service_impl",
    srcs = ["worker_service_impl.cc"],
    hdrs = ["worker_service_impl.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/core:host_memory_allocator",
        "//tpu_sync/core:kv_manager_holder",
        "//tpu_sync/core:raiden_transfer_endpoint",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:transfer_program_reshard",
        "//tpu_sync/proto:transfer_program_cc_proto",
        "//tpu_sync/proto:worker_service_cc_grpc",
        "//tpu_sync/proto:worker_service_cc_proto",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "worker_service_server",
    srcs = ["worker_service_server.cc"],
    hdrs = ["worker_service_server.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":worker_service_impl",
        "//tpu_sync/core:host_memory_allocator",
        "//tpu_sync/core:kv_manager_holder",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "raiden_controller",
    srcs = ["raiden_controller.cc"],
    hdrs = ["raiden_controller.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":controller_server",
        ":controller_service",
        ":worker_registry",
        ":worker_service_client",
        "//tpu_sync/common:raiden_id",
        "//tpu_sync/core:buffer",
        "//tpu_sync/core:raiden_transfer_endpoint",
        "//tpu_sync/core:status_macros",
        "//tpu_sync/kv_cache:logical_block_manager",
        "//tpu_sync/proto:controller_service_cc_grpc",
        "//tpu_sync/proto:controller_service_cc_proto",
        "//tpu_sync/proto:worker_service_cc_proto",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@xla//xla/tsl/concurrency:future",
    ],
)

cc_library(
    name = "controller_client",
    srcs = ["controller_client.cc"],
    hdrs = ["controller_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/core:raiden_transfer_endpoint",
        "//tpu_sync/proto:controller_service_cc_grpc",
        "//tpu_sync/proto:controller_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "worker_registry",
    srcs = ["worker_registry.cc"],
    hdrs = ["worker_registry.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":worker_service_client",
        "//tpu_sync/core:raiden_transfer_endpoint",
        "//tpu_sync/proto:worker_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "controller_service",
    srcs = ["controller_service.cc"],
    hdrs = ["controller_service.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":worker_registry",
        "//tpu_sync/core:buffer",
        "//tpu_sync/core:raiden_transfer_endpoint",
        "//tpu_sync/proto:controller_service_cc_grpc",
        "//tpu_sync/proto:controller_service_cc_proto",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@xla//xla/tsl/concurrency:future",
    ],
)

cc_library(
    name = "controller_server",
    srcs = ["controller_server.cc"],
    hdrs = ["controller_server.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":controller_service",
        ":worker_registry",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "test_util",
    testonly = True,
    hdrs = ["test_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":controller_client",
        ":controller_service",
        ":worker_service_client",
        ":worker_service_impl",
        "//tpu_sync/core:raiden_transfer_endpoint",
        "//tpu_sync/core:raw_transfer_core",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "worker_service_test",
    srcs = ["worker_service_test.cc"],
    deps = [
        ":test_util",
        ":worker_service_client",
        ":worker_service_impl",
        "//tpu_sync/core:kv_manager_holder",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/proto:worker_service_cc_proto",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "raiden_controller_test",
    srcs = ["raiden_controller_test.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    local_defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":controller_client",
        ":controller_server",
        ":raiden_controller",
        ":test_util",
        "//tpu_sync/common:raiden_id",
        "//tpu_sync/core:buffer",
        "//tpu_sync/core:kv_manager_holder",
        "//tpu_sync/core:raiden_transfer_endpoint",
        "//tpu_sync/proto:worker_service_cc_proto",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "controller_service_test",
    srcs = ["controller_service_test.cc"],
    deps = [
        ":controller_client",
        ":controller_service",
        ":test_util",
        "//tpu_sync/core:raiden_transfer_endpoint",
        "//tpu_sync/proto:controller_service_cc_grpc",
        "//tpu_sync/proto:controller_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "worker_service_server_test",
    srcs = ["worker_service_server_test.cc"],
    deps = [
        ":worker_service_client",
        ":worker_service_server",
        "//tpu_sync/proto:worker_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "worker_registry_test",
    srcs = ["worker_registry_test.cc"],
    deps = [
        ":worker_registry",
        "//tpu_sync/core:raiden_transfer_endpoint",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "controller_server_test",
    srcs = ["controller_server_test.cc"],
    deps = [
        ":controller_client",
        ":controller_server",
        "//tpu_sync/core:raiden_transfer_endpoint",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

# Console-script launchers for the control-plane binaries bundled in the
# tpu_raiden_torch wheel. The binaries ride along as data at their source
# paths, so pip installs are sufficient to run the registry and the host
# store node — see //ci/wheel:BUILD.bazel. (The orchestrator the launcher
# originally also wrapped was deprecated upstream in ff56427.)
py_library(
    name = "binary_launcher",
    srcs = ["binary_launcher.py"],
    data = [
        "//tpu_sync/kv_cache/global_registry:global_registry_server",
        "//tpu_sync/store_node:kv_cache_host_store_node_main",
    ],
)
