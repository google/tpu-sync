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
load("@rules_python//python:proto.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "transfer_control_proto",
    srcs = ["transfer_control.proto"],
)

cc_proto_library(
    name = "transfer_control_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":transfer_control_proto"],
)

cc_grpc_library(
    name = "transfer_control_cc_grpc",
    srcs = [":transfer_control_proto"],
    visibility = ["//visibility:public"],
    deps = [":transfer_control_cc_proto"],
)

py_proto_library(
    name = "transfer_control_py_pb2",
    visibility = ["//visibility:public"],
    deps = [":transfer_control_proto"],
)

proto_library(
    name = "transfer_program_proto",
    srcs = ["transfer_program.proto"],
    deps = ["//tpu_sync/rpc:raiden_service_proto"],
)

cc_proto_library(
    name = "transfer_program_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":transfer_program_proto"],
)

py_proto_library(
    name = "transfer_program_py_pb2",
    visibility = ["//visibility:public"],
    deps = [":transfer_program_proto"],
)

proto_library(
    name = "worker_service_proto",
    srcs = ["worker_service.proto"],
    deps = [
        ":transfer_program_proto",
        "//tpu_sync/rpc:raiden_service_proto",
    ],
)

cc_proto_library(
    name = "worker_service_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":worker_service_proto"],
)

cc_grpc_library(
    name = "worker_service_cc_grpc",
    srcs = [":worker_service_proto"],
    grpc_only = True,
    visibility = ["//visibility:public"],
    deps = [":worker_service_cc_proto"],
)

py_proto_library(
    name = "worker_service_py_pb2",
    visibility = ["//visibility:public"],
    deps = [":worker_service_proto"],
)

py_grpc_library(
    name = "worker_service_py_pb2_grpc",
    srcs = [":worker_service_proto"],
    visibility = ["//visibility:public"],
    deps = [":worker_service_py_pb2"],
)

proto_library(
    name = "controller_service_proto",
    srcs = ["controller_service.proto"],
    deps = [
        ":worker_service_proto",
        "//tpu_sync/rpc:raiden_service_proto",
    ],
)

cc_proto_library(
    name = "controller_service_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":controller_service_proto"],
)

cc_grpc_library(
    name = "controller_service_cc_grpc",
    srcs = [":controller_service_proto"],
    generate_mocks = True,
    grpc_only = True,
    visibility = ["//visibility:public"],
    deps = [":controller_service_cc_proto"],
)

proto_library(
    name = "kv_cache_store_service_proto",
    srcs = ["kv_cache_store_service.proto"],
    deps = [
        ":worker_service_proto",
        "//tpu_sync/rpc:raiden_service_proto",
    ],
)

cc_proto_library(
    name = "kv_cache_store_service_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":kv_cache_store_service_proto"],
)

cc_grpc_library(
    name = "kv_cache_store_service_cc_grpc",
    srcs = [":kv_cache_store_service_proto"],
    generate_mocks = True,
    grpc_only = True,
    visibility = ["//visibility:public"],
    deps = [":kv_cache_store_service_cc_proto"],
)

py_proto_library(
    name = "kv_cache_store_service_py_pb2",
    visibility = ["//visibility:public"],
    deps = [":kv_cache_store_service_proto"],
)

py_grpc_library(
    name = "kv_cache_store_service_py_pb2_grpc",
    srcs = [":kv_cache_store_service_proto"],
    visibility = ["//visibility:public"],
    deps = [":kv_cache_store_service_py_pb2"],
)
