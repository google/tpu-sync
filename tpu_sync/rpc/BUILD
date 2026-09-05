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
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@rules_python//python:proto.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "coordination_proto",
    srcs = ["coordination.proto"],
)

proto_library(
    name = "raiden_service_proto",
    srcs = ["raiden_service.proto"],
)

cc_proto_library(
    name = "raiden_service_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":raiden_service_proto"],
)

cc_grpc_library(
    name = "raiden_service_cc_grpc",
    srcs = [":raiden_service_proto"],
    generate_mocks = True,
    visibility = ["//visibility:public"],
    deps = [":raiden_service_cc_proto"],
)

py_proto_library(
    name = "raiden_service_py_pb2",
    visibility = ["//visibility:public"],
    deps = [":raiden_service_proto"],
)

py_grpc_library(
    name = "raiden_service_py_pb2_grpc",
    srcs = [":raiden_service_proto"],
    visibility = ["//visibility:public"],
    deps = [":raiden_service_py_pb2"],
)

proto_library(
    name = "controller_service_proto",
    srcs = ["controller_service.proto"],
    deps = [":raiden_service_proto"],
)

py_proto_library(
    name = "controller_service_py_pb2",
    visibility = ["//visibility:public"],
    deps = [":controller_service_proto"],
)

cc_proto_library(
    name = "controller_service_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":controller_service_proto"],
)

cc_proto_library(
    name = "coordination_cc_proto",
    deps = [":coordination_proto"],
)

cc_grpc_library(
    name = "coordination_cc_grpc",
    srcs = [":coordination_proto"],
    grpc_only = True,
    deps = [":coordination_cc_proto"],
)

py_proto_library(
    name = "coordination_py_pb2",
    deps = [":coordination_proto"],
)

py_grpc_library(
    name = "coordination_py_pb2_grpc",
    srcs = [":coordination_proto"],
    deps = [":coordination_py_pb2"],
)

py_library(
    name = "coordination_helper",
    srcs = ["coordination_helper.py"],
    deps = [
        ":coordination_py_pb2",
        ":coordination_py_pb2_grpc",
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
    ],
)

py_library(
    name = "raiden_controller",
    srcs = ["raiden_controller.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":controller_service_py_pb2",
        ":raiden_service_py_pb2",
        "//tpu_sync/api:common",
        "//tpu_sync/kv_cache:nd_slice_math",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "raiden_controller_test",
    srcs = ["raiden_controller_test.py"],
    deps = [
        ":raiden_controller",
        ":raiden_service_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "coordination_helper_test",
    srcs = ["coordination_helper_test.py"],
    deps = [
        ":coordination_helper",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

cc_library(
    name = "coordination_client",
    srcs = ["coordination_client.cc"],
    hdrs = ["coordination_client.h"],
    deps = [
        ":coordination_cc_grpc",
        ":coordination_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "coordination_client_test",
    srcs = ["coordination_client_test.cc"],
    deps = [
        ":coordination_cc_grpc",
        ":coordination_cc_proto",
        ":coordination_client",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
