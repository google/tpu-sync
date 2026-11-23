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

load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

_X86_MARCH_COPTS = select({
    "@platforms//cpu:x86_64": ["-march=x86-64-v2"],
    "//conditions:default": [],
})

# The JAX half of the Mooncake TPU buffer provider, which _tpu_raiden_jax
# mounts as its mooncake_provider submodule.
cc_library(
    name = "mooncake_provider_bindings",
    srcs = ["mooncake_provider_bindings.cc"],
    hdrs = ["mooncake_provider_bindings.h"],
    copts = [
        "-fexceptions",
        "-frtti",
    ] + _X86_MARCH_COPTS,
    features = ["-use_header_modules"],
    deps = [
        "//tpu_sync/frameworks/jax:jax_compat",
        "//tpu_sync/solutions/mooncake:mooncake_provider",
        "@com_google_absl//absl/status:statusor",
        "@nanobind",
        "@rules_python//python/cc:current_py_cc_headers",
    ],
)

# TPU buffer provider for Mooncake TENT, for jax.Arrays: the module Mooncake
# imports as tpu_sync.solutions.mooncake.jax.mooncake_provider. Its C table is
# served by _tpu_raiden_jax (no extra .so).
py_library(
    name = "mooncake_provider",
    srcs = ["mooncake_provider.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/frameworks/jax:_tpu_raiden_jax",
        "@jax//jax",
    ],
)
