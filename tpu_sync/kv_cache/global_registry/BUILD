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

load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "global_registry_proto",
    srcs = ["global_registry.proto"],
    deps = ["//tpu_sync/rpc:raiden_service_proto"],
)

cc_proto_library(
    name = "global_registry_cc_proto",
    deps = [":global_registry_proto"],
)

cc_grpc_library(
    name = "global_registry_cc_grpc",
    srcs = [":global_registry_proto"],
    generate_mocks = True,
    grpc_only = True,
    deps = [":global_registry_cc_proto"],
)

cc_library(
    name = "global_registry_server_lib",
    srcs = ["global_registry_server.cc"],
    hdrs = ["global_registry_server.h"],
    deps = [
        ":global_registry_cc_grpc",
        ":global_registry_cc_proto",
        "//tpu_sync/kv_cache:raiden_id",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_binary(
    name = "global_registry_server",
    srcs = ["global_registry_server_main.cc"],
    deps = [
        ":global_registry_server_lib",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "global_registry_client_cc",
    srcs = ["global_registry_client.cc"],
    hdrs = ["global_registry_client.h"],
    deps = [
        ":global_registry_cc_grpc",
        ":global_registry_cc_proto",
        "//tpu_sync/kv_cache:raiden_id",
        "@boringssl//:crypto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@xla//xla/tsl/concurrency:future",
    ],
)

cc_library(
    name = "test_util",
    testonly = True,
    hdrs = ["test_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":global_registry_cc_grpc",
        ":global_registry_client_cc",
        ":global_registry_server_lib",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "global_registry_test",
    srcs = ["global_registry_test.cc"],
    deps = [
        ":global_registry_cc_grpc",
        ":global_registry_cc_proto",
        ":global_registry_client_cc",
        ":global_registry_server_lib",
        ":test_util",
        "//tpu_sync/kv_cache:raiden_id",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/concurrency:future",
    ],
)
