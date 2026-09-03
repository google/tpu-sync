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
load(":header_only_cc_info.bzl", "header_only_cc_info")

package(default_visibility = ["//visibility:public"])

header_only_cc_info(
    name = "tsl_platform_headers",
    deps = [
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "raiden_transfer_endpoint",
    hdrs = [
        "raiden_transfer_endpoint.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "status_macros",
    hdrs = [
        "status_macros.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "buffer",
    srcs = [
        "buffer.cc",
    ],
    hdrs = [
        "buffer.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":raiden_transfer_endpoint",
        "//tpu_sync/proto:worker_service_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "buffer_utils",
    srcs = ["buffer_utils.cc"],
    hdrs = ["buffer_utils.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":raw_transfer_core",
        "@com_google_absl//absl/strings",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_extension_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_internal",
    ],
)

cc_test(
    name = "buffer_utils_test",
    srcs = ["buffer_utils_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":buffer_utils",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "raiden_manager_base",
    srcs = [
        "raiden_manager_base.cc",
    ],
    hdrs = [
        "raiden_manager_base.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":raw_transfer_core",
        ":tpu_utils",
        ":xla_raw_transfer_headers",
        "//tpu_sync/transport:block_transport",
        "//tpu_sync/transport:block_transport_delegate",
        "//tpu_sync/transport:buffer_push_task",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

header_only_cc_info(
    name = "xla_raw_transfer_headers",
    deps = [
        "@xla//xla:future",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:common_pjrt_client",
        "@xla//xla/pjrt:host_memory_allocator",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_layout",
        "@xla//xla/pjrt:raw_buffer",
        "@xla//xla/pjrt:semaphore",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_extension_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_external",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/stream_executor:stream",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "host_memory_allocator",
    srcs = [
        "host_memory_allocator.cc",
    ],
    hdrs = [
        "host_memory_allocator.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_utils",
        ":xla_raw_transfer_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tpu_utils",
    srcs = [
        "tpu_utils.cc",
    ],
    hdrs = [
        "tpu_utils.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "numa_thread_pool",
    srcs = [
        "numa_thread_pool.cc",
    ],
    hdrs = [
        "numa_thread_pool.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_utils",
        "@xla//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "utils",
    hdrs = [
        "utils.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":host_memory_allocator",
        ":xla_raw_transfer_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "raw_transfer_core",
    hdrs = [
        "raiden_future.h",
        "raw_transfer_core.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
        "-Wno-unused-variable",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":xla_raw_transfer_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        # buildcleaner: keep
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_external",
        "@xla//xla/tsl/concurrency:async_value",
    ],
)

cc_library(
    name = "raw_transfer_impl",
    hdrs = [
        "raw_transfer_impl.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
        "-Wno-unused-variable",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":raw_transfer_core",
        ":status_macros",
        ":xla_raw_transfer_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "kv_manager_holder",
    hdrs = ["kv_manager_holder.h"],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":raiden_transfer_endpoint",
        ":raw_transfer_core",
        ":status_macros",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "transfer_program_reshard_test",
    srcs = ["transfer_program_reshard_test.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    linkstatic = True,
    deps = [
        ":transfer_program_reshard",
        "//tpu_sync/proto:transfer_program_cc_proto",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//src/google/protobuf/io",
    ],
)

cc_library(
    name = "transfer_program_reshard",
    srcs = ["transfer_program_reshard.cc"],
    hdrs = ["transfer_program_reshard.h"],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/proto:transfer_program_cc_proto",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "kv_manager_holder_test",
    srcs = ["kv_manager_holder_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":kv_manager_holder",
        ":raiden_transfer_endpoint",
        "//tpu_sync/core/controller:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

# Pure helper: how many (pool, peer) push completions a sender-side pool
# reshard plan produces (multi-destination aware). Header-only over the
# service proto so it unit-tests without a device-attached manager.
cc_library(
    name = "pool_reshard_send_slots",
    srcs = ["pool_reshard_send_slots.cc"],
    hdrs = ["pool_reshard_send_slots.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/rpc:raiden_service_cc_proto",
    ],
)

cc_test(
    name = "pool_reshard_send_slots_test",
    srcs = ["pool_reshard_send_slots_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":pool_reshard_send_slots",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "kv_cache_manager_with_transfer",
    srcs = [
        "kv_cache_manager_with_transfer.cc",
    ],
    hdrs = [
        "kv_cache_manager_with_transfer.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":host_memory_allocator",
        ":metrics_collector",
        ":pool_reshard_send_slots",
        ":raiden_manager_base",
        ":raiden_transfer_endpoint",
        ":raw_transfer_core",
        ":status_macros",
        ":tpu_utils",
        "//tpu_sync/kv_cache:kv_cache_manager_base",
        "//tpu_sync/kv_cache:pool_layout",
        "//tpu_sync/telemetry:metrics_api",
        "//tpu_sync/telemetry:metrics_backend",
        "//tpu_sync/transport:block_transport",
        "//tpu_sync/transport:block_transport_delegate",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "tpu_pjrt_manager",
    srcs = [
        "tpu_pjrt_manager.cc",
    ],
    hdrs = [
        "tpu_pjrt_manager.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",
    ],
)

cc_test(
    name = "tpu_pjrt_manager_test",
    srcs = ["tpu_pjrt_manager_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    tags = [
        "no_oss",
        "requires-jellyfish",
    ],
    deps = [
        ":tpu_pjrt_manager",
        "@xla//xla:literal",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_test(
    name = "buffer_test",
    srcs = ["buffer_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":buffer",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "host_memory_allocator_test",
    srcs = ["host_memory_allocator_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    tags = [
        "no_oss",
        "requires-jellyfish",
    ],
    deps = [
        ":host_memory_allocator",
        ":tpu_pjrt_manager",
        ":utils",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_test(
    name = "host_memory_allocator_shm_key_test",
    srcs = ["host_memory_allocator_shm_key_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    deps = [
        ":host_memory_allocator",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_test(
    name = "tpu_utils_test",
    srcs = ["tpu_utils_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    tags = [
        "no_oss",
        "requires-ghostfish:4",
    ],
    deps = [
        ":tpu_pjrt_manager",
        ":tpu_utils",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_test(
    name = "raiden_manager_base_test",
    srcs = ["raiden_manager_base_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    tags = [
        "no_oss",
        "requires-jellyfish",
    ],
    deps = [
        ":host_memory_allocator",
        ":raiden_manager_base",
        ":raw_transfer_core",
        ":tpu_pjrt_manager",
        ":tpu_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@xla//xla:future",
        "@xla//xla:literal",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_external",  # build_cleaner: keep
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_test(
    name = "raw_transfer_core_test",
    srcs = ["raw_transfer_core_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    tags = [
        "no_oss",
        "requires-jellyfish",
    ],
    deps = [
        ":raw_transfer_core",
        ":tpu_pjrt_manager",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@xla//xla:future",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_test(
    name = "raw_transfer_impl_test",
    srcs = ["raw_transfer_impl_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    tags = [
        "no_oss",
        "requires-jellyfish",
    ],
    deps = [
        ":raw_transfer_core",
        ":raw_transfer_impl",
        ":tpu_pjrt_manager",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_test(
    name = "raw_transfer_perf_test",
    srcs = ["raw_transfer_perf_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    tags = [
        "no_oss",
        "requires-ghostfish:4",
    ],
    deps = [
        ":host_memory_allocator",
        ":raw_transfer_impl",
        ":tpu_pjrt_manager",
        ":tpu_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
    ],
)

cc_test(
    name = "kv_cache_manager_with_transfer_test",
    srcs = ["kv_cache_manager_with_transfer_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    tags = [
        "no_oss",
        "requires-jellyfish",
    ],
    deps = [
        ":kv_cache_manager_with_transfer",
        ":raw_transfer_core",
        ":tpu_pjrt_manager",
        "//tpu_sync/kv_cache:kv_cache_manager_base",
        "//tpu_sync/telemetry:metrics_api",
        "//tpu_sync/telemetry:mock_metrics_backend",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@xla//xla:future",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

# Hardware-free unit test for the ip wire-encoding.
cc_test(
    name = "kv_cache_manager_with_transfer_ip_test",
    srcs = ["kv_cache_manager_with_transfer_ip_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    deps = [
        ":kv_cache_manager_with_transfer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "kv_cache_manager_with_transfer_pool_reshard_test",
    srcs = ["kv_cache_manager_with_transfer_pool_reshard_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    deps = [
        ":kv_cache_manager_with_transfer",
        "//tpu_sync/kv_cache:kv_cache_manager_base",
        "//tpu_sync/kv_cache:pool_layout",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "//tpu_sync/telemetry:metrics_api",
        "//tpu_sync/telemetry:mock_metrics_backend",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "kv_cache_manager_perf_test",
    timeout = "long",
    srcs = ["kv_cache_manager_perf_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    tags = [
        "no_oss",
        "notap",
        "requires-ghostfish:4",
    ],
    deps = [
        ":host_memory_allocator",
        ":raw_transfer_core",
        ":tpu_pjrt_manager",
        ":tpu_utils",
        "//tpu_sync/kv_cache:kv_cache_manager_base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@xla//xla:future",
        "@xla//xla:types",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_test(
    name = "kv_cache_manager_with_transfer_perf_test",
    timeout = "long",
    srcs = ["kv_cache_manager_with_transfer_perf_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    tags = [
        "no_oss",
        "notap",
        "requires-ghostfish:4",
    ],
    deps = [
        ":host_memory_allocator",
        ":kv_cache_manager_with_transfer",
        ":raw_transfer_core",
        ":tpu_pjrt_manager",
        "//tpu_sync/kv_cache:kv_cache_manager_base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@xla//xla:future",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_library(
    name = "metrics_collector",
    srcs = ["metrics_collector.cc"],
    hdrs = ["metrics_collector.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "metrics_collector_test",
    srcs = ["metrics_collector_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":metrics_collector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)
