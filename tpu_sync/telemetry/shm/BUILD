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
    name = "shm_layout",
    hdrs = ["shm_layout.h"],
    deps = [
        "//tpu_sync/telemetry:metrics_backend",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "shm_layout_test",
    srcs = ["shm_layout_test.cc"],
    deps = [
        ":shm_layout",
        "//tpu_sync/telemetry:metrics_backend",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "shm_writer",
    srcs = ["shm_writer.cc"],
    hdrs = ["shm_writer.h"],
    deps = [
        ":shm_layout",
        "//tpu_sync/telemetry:metrics_backend",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "shm_writer_test",
    srcs = ["shm_writer_test.cc"],
    deps = [
        ":shm_layout",
        ":shm_writer",
        "//tpu_sync/telemetry:metrics_backend",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "shm_collector",
    srcs = ["shm_collector.cc"],
    hdrs = ["shm_collector.h"],
    deps = [
        ":shm_layout",
        "//tpu_sync/telemetry:metrics_backend",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "shm_collector_test",
    srcs = ["shm_collector_test.cc"],
    deps = [
        ":shm_collector",
        ":shm_layout",
        ":shm_writer",
        "//tpu_sync/telemetry:metrics_backend",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
