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
    name = "metrics_backend",
    hdrs = ["metrics_backend.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "metrics_api",
    srcs = ["metrics_api.cc"],
    hdrs = ["metrics_api.h"],
    deps = [
        ":buffered_metrics_exporter",
        ":metrics_3p_prometheus_exporter",
        ":metrics_backend",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "test_util",
    testonly = True,
    hdrs = ["test_util.h"],
)

cc_library(
    name = "mock_metrics_backend",
    testonly = True,
    hdrs = ["mock_metrics_backend.h"],
    deps = [
        ":metrics_api",
        ":metrics_backend",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_for_library_testonly",
    ],
)

cc_test(
    name = "metrics_api_test",
    srcs = ["metrics_api_test.cc"],
    deps = [
        ":metrics_api",
        ":metrics_backend",
        ":mock_metrics_backend",
        ":test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "metrics_3p_prometheus_exporter",
    srcs = ["prometheus_exporter.cc"],
    hdrs = ["prometheus_exporter.h"],
    # -fexceptions is required because prometheus-cpp Exposer throws C++
    # exceptions (std::runtime_error) on socket binding and initialization failure.
    copts = ["-fexceptions"],
    # Disables Clang header modules due to incompatibility with prometheus-cpp / CivetWeb headers.
    features = ["-use_header_modules"],
    deps = [
        ":metrics_backend",
        "@com_github_jupp0r_prometheus_cpp//core",
        "@com_github_jupp0r_prometheus_cpp//pull",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "prometheus_exporter_test",
    srcs = ["prometheus_exporter_test.cc"],
    deps = [
        ":metrics_3p_prometheus_exporter",
        ":metrics_backend",
        ":test_util",
        "@com_github_jupp0r_prometheus_cpp//core",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "buffered_metrics_exporter",
    srcs = ["buffered_metrics_exporter.cc"],
    hdrs = ["buffered_metrics_exporter.h"],
    deps = [
        ":metrics_backend",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "buffered_metrics_exporter_test",
    srcs = ["buffered_metrics_exporter_test.cc"],
    deps = [
        ":buffered_metrics_exporter",
        ":metrics_backend",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "label_util",
    srcs = ["label_util.cc"],
    hdrs = ["label_util.h"],
    deps = [
        ":metrics_backend",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "label_util_test",
    srcs = ["label_util_test.cc"],
    deps = [
        ":label_util",
        ":metrics_backend",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
