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
