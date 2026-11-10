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
load("//tpu_sync/platform:build_config.bzl", "platform_deps")

package(default_visibility = ["//visibility:public"])

exports_files(
    ["grpc_credential_factory.h"],
    visibility = [
        "//tpu_sync/platform/cloud:__pkg__",
    ],
)

cc_library(
    name = "grpc_credential_factory",
    hdrs = ["grpc_credential_factory.h"],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status:statusor",
    ] + platform_deps("grpc_credential_factory"),
)

cc_test(
    name = "grpc_credential_factory_test",
    srcs = ["grpc_credential_factory_test.cc"],
    deps = [
        ":grpc_credential_factory",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
