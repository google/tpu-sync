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

load("@nanobind_bazel//:build_defs.bzl", "nanobind_extension")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "logical_types",
    hdrs = ["logical_types.h"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "entity_registry",
    srcs = ["entity_registry.cc"],
    hdrs = ["entity_registry.h"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":logical_types",
        "//tpu_sync/common:raiden_id",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "logical_reshard_planner",
    srcs = ["logical_reshard_planner.cc"],
    hdrs = ["logical_reshard_planner.h"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":logical_types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "pull_scheduler",
    srcs = ["pull_scheduler.cc"],
    hdrs = ["pull_scheduler.h"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":logical_types",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dynamic_pull_engine",
    srcs = ["dynamic_pull_engine.cc"],
    hdrs = ["dynamic_pull_engine.h"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":entity_registry",
        ":logical_types",
        ":pull_scheduler",
        "//tpu_sync/common:raiden_id",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "fake_sampler_fleet",
    testonly = True,
    srcs = ["fake_sampler_fleet.cc"],
    hdrs = ["fake_sampler_fleet.h"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":logical_types",
        ":pull_scheduler",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_binary(
    name = "pull_scheduler_benchmark",
    testonly = True,
    srcs = ["pull_scheduler_benchmark.cc"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":fake_sampler_fleet",
        ":logical_reshard_planner",
        ":logical_types",
        ":pull_scheduler",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "async_control_server",
    srcs = ["async_control_server.cc"],
    hdrs = ["async_control_server.h"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//tpu_sync/common/control_pipe:control_pipe_types",
        "//tpu_sync/proto:control_pipe_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "async_control_server_test",
    srcs = ["async_control_server_test.cc"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":async_control_server",
        "//tpu_sync/common/control_pipe",
        "//tpu_sync/common/control_pipe:control_pipe_types",
        "//tpu_sync/proto:control_pipe_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "controller_v3",
    srcs = ["controller_v3.cc"],
    hdrs = ["controller_v3.h"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":async_control_server",
        ":dynamic_pull_engine",
        ":entity_registry",
        ":logical_reshard_planner",
        ":logical_types",
        ":pull_scheduler",
        "//tpu_sync/common:detached_thread_group",
        "//tpu_sync/common:raiden_id",
        "//tpu_sync/common/control_pipe",
        "//tpu_sync/common/control_pipe:control_pipe_interfaces",
        "//tpu_sync/common/control_pipe:control_pipe_types",
        "//tpu_sync/proto:control_pipe_cc_proto",
        "//tpu_sync/rpc:controller_service_cc_proto",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "test_pull_wire_codec",
    testonly = True,
    srcs = ["test_pull_wire_codec.cc"],
    hdrs = ["test_pull_wire_codec.h"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":controller_v3",
        ":dynamic_pull_engine",
        ":pull_scheduler",
        "//tpu_sync/common:raiden_id",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
    ],
)

cc_binary(
    name = "controller_pull_rpc_benchmark",
    testonly = True,
    srcs = ["controller_pull_rpc_benchmark.cc"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":async_control_server",
        ":controller_v3",
        ":dynamic_pull_engine",
        ":entity_registry",
        ":pull_scheduler",
        ":test_pull_wire_codec",
        "//tpu_sync/common:raiden_id",
        "//tpu_sync/proto:control_pipe_cc_proto",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

nanobind_extension(
    name = "_controller_v3",
    srcs = ["controller_v3_binding.cc"],
    copts = [
        "-fexceptions",
        "-frtti",
        "-fvisibility=hidden",
    ],
    features = ["-use_header_modules"],
    linkopts = ["-Wl,--exclude-libs,ALL"],
    visibility = ["//visibility:public"],
    deps = [
        ":controller_v3",
        ":dynamic_pull_engine",
        ":entity_registry",
        ":logical_reshard_planner",
        ":logical_types",
        ":pull_scheduler",
        "//tpu_sync/common:raiden_id",
        "//tpu_sync/common/control_pipe:control_pipe_interfaces",
        "//tpu_sync/common/control_pipe:control_pipe_types",
        "//tpu_sync/rpc:controller_service_cc_proto",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@nanobind",
    ],
)

cc_test(
    name = "entity_registry_test",
    srcs = ["entity_registry_test.cc"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":entity_registry",
        "//tpu_sync/common:raiden_id",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "logical_reshard_planner_test",
    srcs = ["logical_reshard_planner_test.cc"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":logical_reshard_planner",
        ":logical_types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "pull_scheduler_test",
    srcs = ["pull_scheduler_test.cc"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":fake_sampler_fleet",
        ":logical_reshard_planner",
        ":logical_types",
        ":pull_scheduler",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "dynamic_pull_engine_test",
    srcs = ["dynamic_pull_engine_test.cc"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":dynamic_pull_engine",
        ":entity_registry",
        ":logical_reshard_planner",
        ":logical_types",
        ":pull_scheduler",
        "//tpu_sync/common:raiden_id",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "controller_v3_test",
    srcs = ["controller_v3_test.cc"],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":async_control_server",
        ":controller_v3",
        ":dynamic_pull_engine",
        ":entity_registry",
        ":logical_types",
        ":pull_scheduler",
        ":test_pull_wire_codec",
        "//tpu_sync/common:raiden_id",
        "//tpu_sync/common/control_pipe",
        "//tpu_sync/common/control_pipe:control_pipe_types",
        "//tpu_sync/proto:control_pipe_cc_proto",
        "//tpu_sync/rpc:raiden_service_cc_proto",
        "//tpu_sync/weight_sync:weight_synchronizer_base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

py_library(
    name = "types",
    srcs = [
        "slice_math.py",
        "types.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":_controller_v3",
        "//tpu_sync/api:common",
        "//tpu_sync/rpc:controller_service_py_pb2",
        "//tpu_sync/rpc:raiden_service_py_pb2",
        "//tpu_sync/weight_sync/manager:controller_types",
    ],
)

py_test(
    name = "types_py_test",
    srcs = ["types_py_test.py"],
    tags = ["no_oss"],
    deps = [
        ":_controller_v3",
        ":types",
        "//tpu_sync/api:common",
        "//tpu_sync/rpc:controller_service_py_pb2",
        "//tpu_sync/rpc:raiden_service_py_pb2",
        "//tpu_sync/weight_sync/manager:controller_types",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)
