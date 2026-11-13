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
    name = "peregrine_control_service",
    srcs = ["peregrine_control_service.cc"],
    hdrs = ["peregrine_control_service.h"],
    deps = [
        "//tpu_sync/common:grpc_util",
        "//tpu_sync/transport/lib:raw_buffer_transport",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status:statusor",
        "@peregrine//peregrine/src/api/v1:psp",
        "@peregrine//peregrine/src/internal/control:message_cc_proto",
        "@peregrine//peregrine/src/internal/control:message_internal_cc_proto",
        "@peregrine//peregrine/src/internal/control:service_cc_grpc_proto",
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
        ":psp_syscall_mock",
        "//tpu_sync/transport/lib:raw_buffer_transport",
        "//tpu_sync/transport/lib:raw_buffer_transport_delegate",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@peregrine//peregrine/src/api/v1:psp",
        "@peregrine//peregrine/src/internal/control:message_cc_proto",
        "@peregrine//peregrine/src/internal/control:message_internal_cc_proto",
        "@peregrine//peregrine/src/internal/control:service_cc_grpc_proto",
    ],
)

cc_library(
    name = "psp_syscall_mock",
    testonly = 1,
    hdrs = ["psp_syscall_mock.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
)
