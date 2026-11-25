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

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "grpc_transport_types",
    srcs = ["grpc_transport_types.cc"],
    hdrs = ["grpc_transport_types.h"],
    features = ["-use_header_modules"],
    deps = [
        "//tpu_sync/proto:grpc_transport_service_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "grpc_transport_service",
    srcs = ["grpc_transport_service.cc"],
    hdrs = ["grpc_transport_service.h"],
    features = ["-use_header_modules"],
    deps = [
        ":grpc_transport_types",
        "//tpu_sync/platform:grpc_credential_factory",
        "//tpu_sync/proto:grpc_transport_service_cc_grpc",
        "//tpu_sync/proto:grpc_transport_service_cc_proto",
        "//tpu_sync/rpc:weight_transfer_common_cc_proto",
        "//tpu_sync/transport:transfer_session",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "grpc_transport_client",
    srcs = ["grpc_transport_client.cc"],
    hdrs = ["grpc_transport_client.h"],
    features = ["-use_header_modules"],
    deps = [
        ":grpc_transport_types",
        "//tpu_sync/platform:grpc_credential_factory",
        "//tpu_sync/proto:grpc_transport_service_cc_grpc",
        "//tpu_sync/proto:grpc_transport_service_cc_proto",
        "//tpu_sync/rpc:weight_transfer_common_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "grpc_transport_test",
    srcs = ["grpc_transport_test.cc"],
    features = ["-use_header_modules"],
    deps = [
        ":grpc_transport_client",
        ":grpc_transport_service",
        ":grpc_transport_types",
        "//tpu_sync/transport:transfer_session",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/concurrency:future",
    ],
)
