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

# BUILD file for tpu_raiden BlockTransport package.

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "block_transport_delegate",
    hdrs = ["block_transport_delegate.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/transport/lib:raw_buffer_transport_delegate",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "buffer_push_task",
    hdrs = ["buffer_push_task.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "block_transport",
    srcs = ["block_transport.cc"],
    hdrs = ["block_transport.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":block_transport_delegate",
        ":buffer_push_task",
        "//tpu_sync/core:status_macros",
        "//tpu_sync/core:tsl_platform_headers",
        "//tpu_sync/telemetry:metrics_api",
        "//tpu_sync/telemetry:metrics_backend",
        "//tpu_sync/transport/lib:chunk",
        "//tpu_sync/transport/lib:chunk_serializer",
        "//tpu_sync/transport/lib:peregrine_control_service",
        "//tpu_sync/transport/lib:raw_buffer_transport",
        "//tpu_sync/transport/lib:transport_adapter",
        "//tpu_sync/transport/peregrine/src/api:socket_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "block_transport_test",
    srcs = ["block_transport_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":block_transport",
        ":block_transport_delegate",
        ":buffer_push_task",
        "//tpu_sync/telemetry:metrics_3p_prometheus_exporter",
        "//tpu_sync/telemetry:metrics_api",
        "//tpu_sync/telemetry:metrics_backend",
        "//tpu_sync/transport/lib/socket:psp_syscall_mock",
        "//tpu_sync/transport/lib/socket:tcp_psp_helper",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "h2h_strided_bench",
    srcs = ["h2h_strided_bench.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//tpu_sync/transport/lib:chunk",
        "//tpu_sync/transport/lib:chunk_serializer",
        "//tpu_sync/transport/peregrine/src/api:socket_util",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)
