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
    tags = ["no_oss"],
    deps = [
        ":kv_cache_listener",
        ":kv_cache_manager_base",
        ":pool_layout",
        "//tpu_sync/common/control_pipe",
        "//tpu_sync/common/control_pipe:control_pipe_interfaces",
        "//tpu_sync/common/control_pipe:control_pipe_types",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
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
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/platform:statusor",
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
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "pool_staging_test",
    srcs = ["pool_staging_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":kv_cache_manager_base",
        ":kv_cache_manager_with_transfer",
        ":pool_layout",
        ":transfer_plan",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "//tpu_sync/transport:block_transport_delegate",
        "//tpu_sync/transport:transfer_session",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/platform:statusor",
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
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "transfer_plan",
    srcs = ["transfer_plan.cc"],
    hdrs = ["transfer_plan.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":pool_layout",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "//tpu_sync/transport:block_transport_delegate",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "host_meminfo",
    srcs = ["host_meminfo.cc"],
    hdrs = ["host_meminfo.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "host_meminfo_test",
    srcs = ["host_meminfo_test.cc"],
    deps = [
        ":host_meminfo",
        "@com_google_absl//absl/strings:string_view",
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
        ":host_meminfo",
        ":logical_block_manager",
        ":pool_layout",
        ":transfer_plan",
        "//tpu_sync/core:host_memory_allocator",
        "//tpu_sync/core:numa_thread_pool",
        "//tpu_sync/core:raiden_manager_base",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:staging_arena",
        "//tpu_sync/core:tpu_utils",
        "//tpu_sync/core:xla_raw_transfer_headers",
        "//tpu_sync/kv_cache_offload:kv_backend_factory",
        "//tpu_sync/kv_cache_offload:kv_cache_store_backend_factory",
        "//tpu_sync/kv_cache_offload/backends:backend",
        "//tpu_sync/kv_cache_offload/backends/storage:posix_backend",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "//tpu_sync/telemetry:metrics_api",
        "//tpu_sync/telemetry:metrics_backend",
        "//tpu_sync/transport:block_transport",
        "//tpu_sync/transport:block_transport_delegate",
        "//tpu_sync/transport:block_transport_session",
        "//tpu_sync/transport:peer_endpoint",
        "//tpu_sync/transport:transfer_session",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@xla//xla/stream_executor:device_address",
        "@xla//xla/tsl/concurrency:future",
    ],
    alwayslink = 1,
)

# JAX test targets successfully relocated to api/jax/tests/BUILD.

cc_library(
    name = "staging_manager",
    srcs = ["staging_manager.cc"],
    hdrs = ["staging_manager.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "staging_manager_test",
    size = "small",
    srcs = ["staging_manager_test.cc"],
    deps = [
        ":staging_manager",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "kv_cache_manager_with_transfer",
    srcs = [
        "data_connection_policy.cc",
        "kv_cache_manager_with_transfer.cc",
        "reshard_receive_session.cc",
        "reshard_send_session.cc",
        "transfer_receive_session.cc",
        "transfer_send_session.cc",
    ],
    hdrs = [
        "data_connection_policy.h",
        "kv_cache_manager_with_transfer.h",
        "reshard_receive_session.h",
        "reshard_send_session.h",
        "transfer_receive_session.h",
        "transfer_send_session.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_base",
        ":pool_layout",
        ":staging_manager",
        ":transfer_plan",
        "//tpu_sync/common:trace",
        "//tpu_sync/core:control_plane_backend",
        "//tpu_sync/core:host_memory_allocator",
        "//tpu_sync/core:metrics_collector",
        "//tpu_sync/core:pool_reshard_send_slots",
        "//tpu_sync/core:raiden_manager_base",
        "//tpu_sync/core:raiden_transfer_endpoint",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:staging_arena",
        "//tpu_sync/core:tsl_platform_headers",
        "//tpu_sync/core:utils",
        "//tpu_sync/fault_injection:hooks",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "//tpu_sync/telemetry:metrics_api",
        "//tpu_sync/telemetry:metrics_backend",
        "//tpu_sync/transport:block_transport",
        "//tpu_sync/transport:block_transport_session",
        "//tpu_sync/transport:peer_endpoint",
        "//tpu_sync/transport:transfer_session",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@xla//xla/tsl/concurrency:future",
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
        ":kv_cache_manager_base",
        ":kv_cache_manager_with_transfer",
        "//tpu_sync/core:host_memory_allocator",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:tpu_pjrt_manager",
        "//tpu_sync/telemetry:metrics_api",
        "//tpu_sync/telemetry:metrics_backend",
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
        ":kv_cache_manager_base",
        ":kv_cache_manager_with_transfer",
        ":pool_layout",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/fault_injection:hooks",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "//tpu_sync/telemetry:metrics_backend",
        "//tpu_sync/telemetry:mock_metrics_backend",
        "//tpu_sync/transport:peer_endpoint",
        "//tpu_sync/transport:transfer_session",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@xla//xla:future",
    ],
)

cc_test(
    name = "reshard_receive_session_test",
    size = "small",
    srcs = ["reshard_receive_session_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    deps = [
        ":kv_cache_manager_base",
        ":kv_cache_manager_with_transfer",
        ":pool_layout",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:staging_arena",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "//tpu_sync/transport:block_transport_delegate",
        "//tpu_sync/transport:transfer_session",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@xla//xla:future",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "reshard_send_session_test",
    size = "small",
    srcs = ["reshard_send_session_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    deps = [
        ":kv_cache_manager_base",
        ":kv_cache_manager_with_transfer",
        ":pool_layout",
        ":transfer_plan",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:staging_arena",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "//tpu_sync/transport:block_transport",
        "//tpu_sync/transport:transfer_session",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@xla//xla:future",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "data_connection_policy_test",
    size = "small",
    srcs = ["data_connection_policy_test.cc"],
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
        "//tpu_sync/transport:peer_endpoint",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "transfer_send_session_test",
    size = "small",
    srcs = ["transfer_send_session_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    deps = [
        ":kv_cache_manager_base",
        ":kv_cache_manager_with_transfer",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:staging_arena",
        "//tpu_sync/transport:block_transport",
        "//tpu_sync/transport:peer_endpoint",
        "//tpu_sync/transport:transfer_session",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@xla//xla:future",
    ],
)

cc_test(
    name = "kv_cache_manager_with_transfer_control_test",
    size = "small",
    srcs = ["kv_cache_manager_with_transfer_control_test.cc"],
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
        "//tpu_sync/core:control_plane_backend",
        "//tpu_sync/core:grpc_control_plane_backend",
        "//tpu_sync/core:numa_thread_pool",
        "//tpu_sync/core:tcp_control_plane_backend",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "kv_cache_manager_with_transfer_host_memory_test",
    size = "small",
    srcs = ["kv_cache_manager_with_transfer_host_memory_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    deps = [
        ":kv_cache_manager_base",
        ":kv_cache_manager_with_transfer",
        "//tpu_sync/core:kv_manager_holder",
        "//tpu_sync/core:metrics_collector",
        "//tpu_sync/core:raiden_transfer_endpoint",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/fault_injection:hooks",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "kv_cache_manager_with_transfer_send_drain_test",
    size = "small",
    srcs = ["kv_cache_manager_with_transfer_send_drain_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    deps = [
        ":kv_cache_manager_base",
        ":kv_cache_manager_with_transfer",
        "//tpu_sync/core:metrics_collector",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:staging_arena",
        "//tpu_sync/fault_injection:hooks",
        "//tpu_sync/transport:block_transport_delegate",
        "//tpu_sync/transport:peer_endpoint",
        "//tpu_sync/transport:transfer_session",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:scoped_mock_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@xla//xla:future",
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
        ":kv_cache_manager_base",
        "//tpu_sync/core:host_memory_allocator",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:tpu_pjrt_manager",
        "//tpu_sync/core:tpu_utils",
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
        ":kv_cache_manager_base",
        ":kv_cache_manager_with_transfer",
        "//tpu_sync/core:host_memory_allocator",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:tpu_pjrt_manager",
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
    name = "raiden_id",
    visibility = ["//visibility:public"],
    deps = ["@com_google_absl//absl/hash"],
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
        "//tpu_sync/common/control_pipe",
        "//tpu_sync/common/control_pipe:control_dispatcher",
        "//tpu_sync/common/control_pipe:control_pipe_interfaces",
        "//tpu_sync/common/control_pipe:control_pipe_types",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)
