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

load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "sources",
    srcs = glob(["**"]),
    visibility = ["//:internal"],
)

filegroup(
    name = "gating_baselines",
    srcs = [
        "h2d_d2h_gating_jax_baselines.json",
        "h2d_d2h_gating_torch_baselines.json",
    ],
)

py_library(
    name = "bap_metrics",
    testonly = True,
    srcs = ["bap_metrics.py"],
    deps = ["@pypi//tensorflow"],
)

py_library(
    name = "perf_core",
    testonly = True,
    srcs = ["perf_core.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/frameworks/jax:_tpu_raiden_jax",
        "//tpu_sync/frameworks/jax:jax_test_utils",
        "@jax//jax",
        "@pypi//libtpu",
        "@pypi//numpy",
    ],
)

py_binary(
    name = "h2d_d2h_perf_test",
    testonly = True,
    srcs = ["h2d_d2h_perf_test.py"],
    deps = [
        ":bap_metrics",
        ":perf_core",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_binary(
    name = "h2d_d2h_benchmark_gating",
    testonly = True,
    srcs = ["h2d_d2h_benchmark_gating.py"],
    data = [":gating_baselines"],
    deps = [
        ":bap_metrics",
        ":perf_core",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@pypi//numpy",
    ],
)

py_binary(
    name = "h2h_cpp_gate",
    testonly = True,
    srcs = ["h2h_cpp_gate.py"],
    data = [
        "//examples/microbenchmarks:h2h_benchmark_runner",
    ],
    deps = [
        ":bap_metrics",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)
