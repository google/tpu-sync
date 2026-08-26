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

py_library(
    name = "nd_slice_math",
    srcs = ["nd_slice_math.py"],
    visibility = ["//visibility:public"],
    deps = ["//tpu_sync/rpc:raiden_service_py_pb2"],
)

py_test(
    name = "nd_slice_math_test",
    srcs = ["nd_slice_math_test.py"],
    deps = [
        ":nd_slice_math",
        "//tpu_sync/rpc:raiden_controller",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

cc_test(
    name = "kv_cache_listener_test",
    srcs = ["kv_cache_listener_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":kv_cache_listener",
        ":kv_cache_manager_base",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "logical_block_manager",
    srcs = ["logical_block_manager.cc"],
    hdrs = ["logical_block_manager.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "logical_block_manager_test",
    srcs = ["logical_block_manager_test.cc"],
    deps = [
        ":logical_block_manager",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "kv_cache_metadata",
    srcs = ["kv_cache_metadata.cc"],
    hdrs = ["kv_cache_metadata.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "kv_cache_metadata_test",
    srcs = ["kv_cache_metadata_test.cc"],
    deps = [
        ":kv_cache_metadata",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "kv_cache_metadata_shm",
    srcs = ["kv_cache_metadata_shm.cc"],
    hdrs = ["kv_cache_metadata_shm.h"],
    linkopts = ["-lrt"],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_metadata",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "kv_cache_store_wrapper",
    srcs = ["kv_cache_store_wrapper.cc"],
    hdrs = ["kv_cache_store_wrapper.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_metadata",
        ":kv_cache_metadata_shm",
        ":kv_cache_store",
        ":kv_cache_store_backend_factory",
        ":raiden_id",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "kv_cache_store_wrapper_test",
    srcs = ["kv_cache_store_wrapper_test.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        ":kv_cache_store",
        ":kv_cache_store_backend",
        ":kv_cache_store_backend_factory",
        ":kv_cache_store_wrapper",
        ":raiden_id",
        "//tpu_sync/kv_cache/global_registry:test_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "kv_cache_metadata_shm_test",
    srcs = ["kv_cache_metadata_shm_test.cc"],
    deps = [
        ":kv_cache_metadata",
        ":kv_cache_metadata_shm",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "pool_layout",
    srcs = ["pool_layout.cc"],
    hdrs = ["pool_layout.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "pool_layout_test",
    srcs = ["pool_layout_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":pool_layout",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "kv_cache_manager_base",
    srcs = [
        "kv_cache_manager_base.cc",
    ],
    hdrs = [
        "kv_cache_manager_base.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":logical_block_manager",
        ":pool_layout",
        "//tpu_sync/core:host_memory_allocator",
        "//tpu_sync/core:numa_thread_pool",
        "//tpu_sync/core:raiden_manager_base",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:status_macros",
        "//tpu_sync/core:tpu_utils",
        "//tpu_sync/core:xla_raw_transfer_headers",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "//tpu_sync/telemetry:metrics_api",
        "//tpu_sync/telemetry:metrics_backend",
        "//tpu_sync/transport:block_transport",
        "//tpu_sync/transport:block_transport_delegate",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@xla//xla/pjrt:semaphore",
        "@xla//xla/stream_executor:device_address",
    ],
    alwayslink = 1,
)

# JAX test targets successfully relocated to api/jax/BUILD.

cc_library(
    name = "lru_cache",
    hdrs = ["lru_cache.h"],
    deps = ["@com_google_absl//absl/container:flat_hash_map"],
)

cc_test(
    name = "lru_cache_test",
    srcs = ["lru_cache_test.cc"],
    deps = [
        ":lru_cache",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "raiden_id",
    hdrs = ["raiden_id.h"],
    visibility = ["//visibility:public"],
    deps = ["@com_google_absl//absl/hash"],
)

cc_library(
    name = "kv_cache_store_backend",
    hdrs = ["kv_cache_store_backend.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":raiden_id",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@xla//xla/tsl/concurrency:future",
    ],
)

cc_library(
    name = "host_offload_backend",
    srcs = ["host_offload_backend.cc"],
    hdrs = ["host_offload_backend.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_metadata",
        ":kv_cache_store_backend",
        ":kv_cache_store_backend_factory",
        ":kv_cache_store_client",
        ":kv_cache_store_server",
        ":lru_cache",
        ":raiden_id",
        "//tpu_sync/core:buffer",
        "//tpu_sync/core:status_macros",
        "//tpu_sync/core/controller:raiden_controller",
        "//tpu_sync/core/controller:worker_registry",
        "//tpu_sync/kv_cache/global_registry:global_registry_client_cc",
        "//tpu_sync/proto:kv_cache_store_service_cc_proto",
        "//tpu_sync/proto:worker_service_cc_proto",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@xla//xla/tsl/concurrency:future",
    ],
    alwayslink = 1,
)

cc_library(
    name = "kv_cache_store_backend_factory",
    srcs = ["kv_cache_store_backend_factory.cc"],
    hdrs = ["kv_cache_store_backend_factory.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_metadata",
        ":kv_cache_store_backend",
        ":raiden_id",
        "//tpu_sync/core:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

cc_test(
    name = "kv_cache_store_backend_factory_test",
    srcs = ["kv_cache_store_backend_factory_test.cc"],
    deps = [
        ":host_offload_backend",
        ":kv_cache_store",
        ":kv_cache_store_backend",
        ":kv_cache_store_backend_factory",
        ":raiden_id",
        "//tpu_sync/core/controller:raiden_controller",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "store_monitor",
    srcs = ["store_monitor.cc"],
    hdrs = ["store_monitor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":raiden_id",
        "//tpu_sync/kv_cache/global_registry:global_registry_cc_proto",
        "//tpu_sync/kv_cache/global_registry:global_registry_client_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "store_monitor_test",
    srcs = ["store_monitor_test.cc"],
    deps = [
        ":raiden_id",
        ":store_monitor",
        "//tpu_sync/kv_cache/global_registry:global_registry_cc_proto",
        "//tpu_sync/kv_cache/global_registry:global_registry_client_cc",
        "//tpu_sync/kv_cache/global_registry:global_registry_server_lib",
        "//tpu_sync/kv_cache/global_registry:test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "kv_cache_store",
    srcs = ["kv_cache_store.cc"],
    hdrs = ["kv_cache_store.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":host_offload_backend",
        ":kv_cache_metadata",
        ":kv_cache_store_backend",
        ":kv_cache_store_backend_factory",
        ":kv_cache_store_server",
        ":lru_cache",
        ":raiden_id",
        ":store_monitor",
        "//tpu_sync/core:buffer",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:status_macros",
        "//tpu_sync/core/controller:raiden_controller",
        "//tpu_sync/kv_cache/global_registry:global_registry_client_cc",
        "//tpu_sync/kv_cache/reshard:reshard_service",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
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

cc_test(
    name = "kv_cache_store_test",
    srcs = ["kv_cache_store_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    local_defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":host_offload_backend",
        ":kv_cache_metadata",
        ":kv_cache_store",
        ":kv_cache_store_backend",
        ":kv_cache_store_backend_factory",
        ":kv_cache_store_client",
        ":lru_cache",
        ":raiden_id",
        "//tpu_sync/core:kv_manager_holder",
        "//tpu_sync/core:raiden_transfer_endpoint",
        "//tpu_sync/core/controller:controller_client",
        "//tpu_sync/core/controller:raiden_controller",
        "//tpu_sync/core/controller:test_util",
        "//tpu_sync/kv_cache/global_registry:global_registry_cc_grpc",
        "//tpu_sync/kv_cache/global_registry:global_registry_client_cc",
        "//tpu_sync/kv_cache/global_registry:global_registry_server_lib",
        "//tpu_sync/kv_cache/global_registry:test_util",
        "//tpu_sync/proto:kv_cache_store_service_cc_grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
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
    name = "host_offload_backend_test",
    srcs = ["host_offload_backend_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    local_defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":host_offload_backend",
        ":kv_cache_store_backend",
        ":kv_cache_store_backend_factory",
        ":kv_cache_store_client",
        ":kv_cache_store_server",
        ":raiden_id",
        "//tpu_sync/core:kv_manager_holder",
        "//tpu_sync/core/controller:controller_client",
        "//tpu_sync/core/controller:raiden_controller",
        "//tpu_sync/core/controller:test_util",
        "//tpu_sync/core/controller:worker_registry",
        "//tpu_sync/kv_cache/global_registry:global_registry_client_cc",
        "//tpu_sync/kv_cache/global_registry:test_util",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "kv_cache_store_service",
    srcs = ["kv_cache_store_service.cc"],
    hdrs = ["kv_cache_store_service.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_store_backend",
        ":raiden_id",
        "//tpu_sync/core:buffer",
        "//tpu_sync/core:raiden_transfer_endpoint",
        "//tpu_sync/core/controller:raiden_controller",
        "//tpu_sync/proto:kv_cache_store_service_cc_grpc",
        "//tpu_sync/proto:kv_cache_store_service_cc_proto",
        "//tpu_sync/proto:worker_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
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
    name = "kv_cache_store_server",
    srcs = ["kv_cache_store_server.cc"],
    hdrs = ["kv_cache_store_server.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_store_backend",
        ":kv_cache_store_service",
        "//tpu_sync/core/controller:raiden_controller",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "kv_cache_store_server_test",
    srcs = ["kv_cache_store_server_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    local_defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":host_offload_backend",
        ":kv_cache_metadata",
        ":kv_cache_store",
        ":kv_cache_store_backend",
        ":kv_cache_store_client",
        ":kv_cache_store_server",
        ":raiden_id",
        "//tpu_sync/core:kv_manager_holder",
        "//tpu_sync/core/controller:controller_client",
        "//tpu_sync/core/controller:raiden_controller",
        "//tpu_sync/core/controller:test_util",
        "//tpu_sync/kv_cache/global_registry:global_registry_server_lib",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "kv_cache_store_client",
    srcs = ["kv_cache_store_client.cc"],
    hdrs = ["kv_cache_store_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/proto:kv_cache_store_service_cc_grpc",
        "//tpu_sync/proto:kv_cache_store_service_cc_proto",
        "//tpu_sync/proto:worker_service_cc_proto",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@xla//xla/tsl/concurrency:future",
    ],
)

cc_test(
    name = "kv_cache_store_client_test",
    srcs = ["kv_cache_store_client_test.cc"],
    local_defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":kv_cache_store_client",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/proto:kv_cache_store_service_cc_grpc",
        "//tpu_sync/proto:kv_cache_store_service_cc_proto",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/concurrency:future",
    ],
)

cc_test(
    name = "kv_cache_store_service_test",
    srcs = ["kv_cache_store_service_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    local_defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":host_offload_backend",
        ":kv_cache_store",
        ":kv_cache_store_backend",
        ":kv_cache_store_client",
        ":kv_cache_store_service",
        ":raiden_id",
        "//tpu_sync/core:buffer",
        "//tpu_sync/core:kv_manager_holder",
        "//tpu_sync/core/controller:controller_client",
        "//tpu_sync/core/controller:raiden_controller",
        "//tpu_sync/core/controller:test_util",
        "//tpu_sync/kv_cache/global_registry:global_registry_server_lib",
        "//tpu_sync/kv_cache/global_registry:test_util",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/concurrency:future",
    ],
)

py_library(
    name = "kv_cache_controller",
    srcs = ["kv_cache_controller.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/rpc:raiden_controller",
        "//tpu_sync/rpc:raiden_service_py_pb2",
    ],
)

cc_library(
    name = "kv_cache_listener",
    srcs = ["kv_cache_listener.cc"],
    hdrs = ["kv_cache_listener.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_base",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)
