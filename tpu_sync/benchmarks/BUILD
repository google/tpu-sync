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
    data = ["h2d_d2h_gating_baselines.json"],
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

# Stats, gate floors, config ranking and SVG plots (stdlib only).
py_library(
    name = "h2h_dist",
    testonly = True,
    srcs = ["h2h_dist.py"],
)

# Cross-host H2H driver: --stage=analyze|record|gate on a BAP multi-host job.
py_binary(
    name = "h2h_multihost_bench",
    testonly = True,
    srcs = ["h2h_multihost_bench.py"],
    data = [
        "h2h_multihost_baselines.json",
        "//examples/microbenchmarks:h2h_benchmark_runner",
    ],
    deps = [
        ":bap_metrics",
        ":h2h_dist",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)
