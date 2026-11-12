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
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

_X86_MARCH_COPTS = select({
    "@platforms//cpu:x86_64": ["-march=x86-64-v2"],
    "//conditions:default": [],
})

# TPU buffer provider for Mooncake TENT. Mooncake dlopen()s this extension's
# .so and looks up mc_tpu_buffer_provider_get, so mooncake_provider.cc is
# compiled into the extension itself: from a cc_library dep, the symbol would
# be hidden by -Wl,--exclude-libs,ALL.
nanobind_extension(
    name = "_mooncake_provider",
    srcs = [
        "mooncake_provider.cc",
        "mooncake_provider.h",
        "mooncake_provider_module.cc",
    ],
    copts = [
        "-fexceptions",
        "-frtti",
        "-fvisibility=hidden",
    ] + _X86_MARCH_COPTS,
    features = ["-use_header_modules"],
    linkopts = ["-Wl,--exclude-libs,ALL"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:tpu_utils",
        "//tpu_sync/core:xla_compat",
        "//tpu_sync/core:xla_pjrt_headers",
        "//tpu_sync/frameworks/jax:jax_compat",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@nanobind",
        "@rules_python//python/cc:current_py_cc_headers",
    ],
)

# mooncake_provider.cc without Python, for mooncake_provider_test.
cc_library(
    name = "mooncake_provider_lib",
    testonly = True,
    srcs = ["mooncake_provider.cc"],
    hdrs = ["mooncake_provider.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:tpu_utils",
        "//tpu_sync/core:xla_compat",
        "//tpu_sync/core:xla_pjrt_headers",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "mooncake_provider_test",
    srcs = ["mooncake_provider_test.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        ":mooncake_provider_lib",
        "//tpu_sync/core:raw_transfer_core",
        "//tpu_sync/core:tpu_utils",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:scoped_mock_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:raw_buffer",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

# TPU buffer provider for Mooncake TENT, for jax.Arrays: the module Mooncake
# imports as tpu_sync.solutions.mooncake.jax.mooncake_provider.
py_library(
    name = "mooncake_provider",
    srcs = ["mooncake_provider.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_mooncake_provider",
        "@jax//jax",
    ],
)
