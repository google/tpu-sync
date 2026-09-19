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

cc_library(
    name = "control_pipe_types",
    srcs = ["control_pipe_types.cc"],
    hdrs = ["control_pipe_types.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "control_pipe_types_test",
    srcs = ["control_pipe_types_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":control_pipe_types",
        "//tpu_sync/proto:control_pipe_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "control_dispatcher",
    srcs = ["control_dispatcher.cc"],
    hdrs = ["control_dispatcher.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":control_pipe_types",
        "//tpu_sync/proto:control_pipe_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "control_dispatcher_test",
    srcs = ["control_dispatcher_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":control_dispatcher",
        ":control_pipe_types",
        "//tpu_sync/proto:control_pipe_cc_proto",
        "//tpu_sync/proto:kv_cache_control_plane_service_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "control_pipe_interfaces",
    hdrs = [
        "control_pipe_client.h",
        "control_pipe_server.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":control_dispatcher",
        ":control_pipe_types",
        "//tpu_sync/proto:control_pipe_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "tcp_control_pipe",
    srcs = ["tcp_control_pipe.cc"],
    hdrs = ["tcp_control_pipe.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":control_dispatcher",
        ":control_pipe_interfaces",
        ":control_pipe_types",
        "//tpu_sync/proto:control_pipe_cc_proto",
        "//tpu_sync/proto:kv_cache_control_plane_service_cc_proto",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/base:core_headers",
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
    ],
)

cc_test(
    name = "tcp_control_pipe_test",
    srcs = ["tcp_control_pipe_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":control_dispatcher",
        ":control_pipe_types",
        ":tcp_control_pipe",
        "//tpu_sync/proto:kv_cache_control_plane_service_cc_proto",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "grpc_control_pipe",
    srcs = ["grpc_control_pipe.cc"],
    hdrs = ["grpc_control_pipe.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":control_dispatcher",
        ":control_pipe_interfaces",
        ":control_pipe_types",
        "//tpu_sync/proto:control_pipe_cc_grpc",
        "//tpu_sync/proto:control_pipe_cc_proto",
        "//tpu_sync/proto:kv_cache_control_plane_service_cc_grpc",
        "//tpu_sync/proto:kv_cache_control_plane_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "zmq_control_pipe",
    srcs = ["zmq_control_pipe.cc"],
    hdrs = ["zmq_control_pipe.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":control_dispatcher",
        ":control_pipe_interfaces",
        ":control_pipe_types",
        "//tpu_sync/proto:control_pipe_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@libzmq",
    ],
)

cc_test(
    name = "zmq_control_pipe_test",
    srcs = ["zmq_control_pipe_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":control_dispatcher",
        ":control_pipe_types",
        ":zmq_control_pipe",
        "//tpu_sync/proto:control_pipe_cc_proto",
        "//tpu_sync/proto:kv_cache_control_plane_service_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@libzmq",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "control_pipe",
    srcs = ["control_pipe_factory.cc"],
    hdrs = [
        "control_dispatcher.h",
        "control_pipe_client.h",
        "control_pipe_server.h",
        "control_pipe_types.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":control_dispatcher",
        ":control_pipe_interfaces",
        ":control_pipe_types",
        ":grpc_control_pipe",
        ":tcp_control_pipe",
        ":zmq_control_pipe",
        "//tpu_sync/proto:control_pipe_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "control_pipe_test",
    srcs = ["control_pipe_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    deps = [
        ":control_dispatcher",
        ":control_pipe",
        ":control_pipe_types",
        "//tpu_sync/proto:control_pipe_cc_grpc",
        "//tpu_sync/proto:control_pipe_cc_proto",
        "//tpu_sync/proto:kv_cache_control_plane_service_cc_grpc",
        "//tpu_sync/proto:kv_cache_control_plane_service_cc_proto",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/platform:statusor",
    ],
)

py_library(
    name = "control_pipe_client",
    srcs = ["control_pipe_client.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/proto:control_pipe_py_pb2",
        "//tpu_sync/proto:control_pipe_py_pb2_grpc",
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
    ],
)

py_test(
    name = "control_pipe_client_test",
    srcs = ["control_pipe_client_test.py"],
    deps = [
        ":control_pipe_client",
        "//tpu_sync/proto:control_pipe_py_pb2",
        "//tpu_sync/proto:control_pipe_py_pb2_grpc",
        "//tpu_sync/rpc:raiden_service_py_pb2",
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)
