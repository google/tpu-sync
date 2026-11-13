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

# The torch half of the Mooncake TPU buffer provider, which _tpu_raiden_torch
# mounts as its mooncake_provider submodule.
cc_library(
    name = "mooncake_provider_bindings",
    srcs = ["mooncake_provider_bindings.cc"],
    hdrs = ["mooncake_provider_bindings.h"],
    copts = [
        "-std=c++20",
        "-fexceptions",
    ] + _X86_MARCH_COPTS,
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nanobind",
        "@torch_tpu//shims/torch:aten_headers",
        "//tpu_sync/frameworks/torch:torch_nanobind_utils",
        # buildcleaner: keep
        "//tpu_sync/frameworks/torch:torch_tpu_tensor_buffer_headers",
        "//tpu_sync/frameworks/torch:torch_tpu_utils",
        "//tpu_sync/solutions/mooncake:mooncake_provider",
    ],
)

# TPU buffer provider for Mooncake TENT, for torch tensors: the module to pass
# to mooncake.tpu as its provider. Its C table is served by _tpu_raiden_torch
# (no extra .so).
py_library(
    name = "mooncake_provider",
    srcs = ["mooncake_provider.py"],
    tags = ["nobuilder"],
    visibility = ["//visibility:public"],
    deps = [
        "@torch_tpu//shims/torch:pytorch",
        "//tpu_sync/api/torch:torch_abi",
        "//tpu_sync/api/torch:torch_tpu_common_loader",
        # buildcleaner: keep
        "//tpu_sync/frameworks/torch:_tpu_raiden_torch",
    ],
)
