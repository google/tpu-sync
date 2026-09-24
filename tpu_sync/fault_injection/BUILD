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
load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "hooks",
    srcs = ["fault_injector.cc"],
    hdrs = [
        "fault_injector.h",
        "hooks.h",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "fault_injector_test",
    srcs = ["fault_injector_test.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        ":hooks",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

# Link and mount inside every extension that also links :hooks.
cc_library(
    name = "fault_injection_bindings",
    srcs = ["fault_injection_bindings.cc"],
    hdrs = ["fault_injection_bindings.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":hooks",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@nanobind",
    ],
)

nanobind_extension(
    name = "fault_injection",
    srcs = ["binding.cc"],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [":fault_injection_bindings"],
)

py_test(
    name = "binding_test",
    srcs = ["binding_test.py"],
    deps = [
        ":fault_injection",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)
