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
    name = "posix_backend",
    srcs = ["posix_backend.cc"],
    hdrs = ["posix_backend.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/common:raiden_id",
        "//tpu_sync/core:numa_thread_pool",
        "//tpu_sync/core/controller:raiden_controller",
        "//tpu_sync/kv_cache:kv_cache_store_backend",
        "//tpu_sync/kv_cache:kv_cache_store_backend_factory",
        "//tpu_sync/kv_cache/backends:backend",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@xla//xla/tsl/concurrency:future",
    ],
    alwayslink = 1,
)

cc_test(
    name = "posix_backend_test",
    srcs = ["posix_backend_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    deps = [
        ":posix_backend",
        "//tpu_sync/core/controller:raiden_controller",
        "//tpu_sync/kv_cache:kv_cache_store_backend",
        "//tpu_sync/kv_cache:kv_cache_store_backend_factory",
        "//tpu_sync/kv_cache/backends:backend",
        "//tpu_sync/proto:worker_service_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
