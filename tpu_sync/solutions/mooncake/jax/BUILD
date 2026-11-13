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
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

_X86_MARCH_COPTS = select({
    "@platforms//cpu:x86_64": ["-march=x86-64-v2"],
    "//conditions:default": [],
})

# TPU buffer provider for Mooncake TENT. Mooncake dlopen()s this extension's
# .so and looks up mc_tpu_buffer_provider_get, so mooncake_provider_module.cc
# defines it; see tpu_sync/solutions/mooncake/mooncake_provider.h.
nanobind_extension(
    name = "_mooncake_provider",
    srcs = ["mooncake_provider_module.cc"],
    copts = [
        "-fexceptions",
        "-frtti",
        "-fvisibility=hidden",
    ] + _X86_MARCH_COPTS,
    features = ["-use_header_modules"],
    linkopts = ["-Wl,--exclude-libs,ALL"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/frameworks/jax:jax_compat",
        "//tpu_sync/solutions/mooncake:mooncake_provider",
        "@com_google_absl//absl/status:statusor",
        "@nanobind",
        "@rules_python//python/cc:current_py_cc_headers",
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
