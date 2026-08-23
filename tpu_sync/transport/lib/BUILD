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

# BUILD file for tpu_raiden transport library package.
# It is only visible to the BlockTransport package.

load("@com_github_google_flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

flatbuffer_cc_library(
    name = "chunk_cc_fbs",
    srcs = ["chunk.fbs"],
)

cc_library(
    name = "chunk",
    hdrs = ["chunk.h"],
)

cc_library(
    name = "transport_adapter",
    hdrs = ["transport_adapter.h"],
)

cc_library(
    name = "raw_buffer_transport_delegate",
    hdrs = ["raw_buffer_transport_delegate.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "chunk_serializer",
    srcs = ["chunk_serializer.cc"],
    hdrs = ["chunk_serializer.h"],
    deps = [
        ":chunk",
        ":chunk_cc_fbs",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "chunk_serializer_test",
    srcs = ["chunk_serializer_test.cc"],
    deps = [
        ":chunk",
        ":chunk_serializer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "raw_buffer_transport",
    srcs = ["raw_buffer_transport.cc"],
    hdrs = ["raw_buffer_transport.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":chunk",
        ":chunk_serializer",
        ":raw_buffer_transport_delegate",
        ":transport_adapter",
        "//tpu_sync/core:numa_thread_pool",
        "//tpu_sync/core:status_macros",
        "//tpu_sync/transport:buffer_push_task",
        "//tpu_sync/transport/lib/conn:pool",
        "//tpu_sync/transport/peregrine/src/api:socket_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "raw_buffer_transport_test",
    srcs = ["raw_buffer_transport_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":raw_buffer_transport",
        ":raw_buffer_transport_delegate",
        "//tpu_sync/transport:buffer_push_task",
        "//tpu_sync/transport/lib/conn:pool",
        "//tpu_sync/transport/peregrine/src/util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "peregrine_control_service",
    hdrs = ["peregrine_control_service.h"],
    deps = [
        ":raw_buffer_transport",
        "//tpu_sync/transport/peregrine/src/internal/control:service_cc_grpc",
        "//tpu_sync/transport/peregrine/src/internal/control:service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "peregrine_control_service_test",
    srcs = ["peregrine_control_service_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":peregrine_control_service",
        ":raw_buffer_transport",
        ":raw_buffer_transport_delegate",
        "//tpu_sync/transport/peregrine/src/internal/control:service_cc_grpc",
        "//tpu_sync/transport/peregrine/src/internal/control:service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
