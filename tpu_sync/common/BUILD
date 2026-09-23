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
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "version",
    hdrs = ["version.h"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "version_test",
    srcs = ["version_test.cc"],
    deps = [
        ":version",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "raiden_id",
    hdrs = ["raiden_id.h"],
    visibility = ["//visibility:public"],
    deps = ["@com_google_absl//absl/hash"],
)

cc_test(
    name = "raiden_id_test",
    srcs = ["raiden_id_test.cc"],
    deps = [
        ":raiden_id",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "grpc_util",
    hdrs = ["grpc_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "grpc_util_test",
    srcs = ["grpc_util_test.cc"],
    deps = [
        ":grpc_util",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "detached_thread_group",
    srcs = ["detached_thread_group.cc"],
    hdrs = ["detached_thread_group.h"],
    # std::thread's constructor reports pthread_create failures by throwing.
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "detached_thread_group_test",
    srcs = ["detached_thread_group_test.cc"],
    deps = [
        ":detached_thread_group",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "trace",
    srcs = ["trace.cc"],
    hdrs = ["trace.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@perfetto//:libperfetto_client_experimental",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/profiler/lib:traceme_encode",
    ],
)

cc_test(
    name = "trace_test",
    srcs = ["trace_test.cc"],
    deps = [
        ":trace",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@perfetto//:libperfetto_client_experimental",
    ],
)

nanobind_extension(
    name = "_raiden_id",
    srcs = ["raiden_id_binding.cc"],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":raiden_id",
        "@com_google_absl//absl/strings:str_format",
    ],
)
