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

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "raiden_storage_plugin_abi",
    hdrs = ["raiden_storage_plugin_abi.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "plugin_kv_backend",
    srcs = ["plugin_kv_backend.cc"],
    hdrs = ["plugin_kv_backend.h"],
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
        ":raiden_storage_plugin_abi",
        "//tpu_sync/kv_cache/backends:backend",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "plugin_kv_cache_store_backend",
    srcs = ["plugin_kv_cache_store_backend.cc"],
    hdrs = ["plugin_kv_cache_store_backend.h"],
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
        ":plugin_kv_backend",
        "//tpu_sync/kv_cache:kv_cache_store_backend",
        "//tpu_sync/kv_cache/backends:backend",
        "//tpu_sync/kv_cache/backends/storage:storage_lookup_helper",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "plugin_registry",
    srcs = ["plugin_registry.cc"],
    hdrs = ["plugin_registry.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    linkopts = ["-ldl"],
    visibility = ["//visibility:public"],
    deps = [
        ":plugin_kv_backend",
        ":plugin_kv_cache_store_backend",
        ":raiden_storage_plugin_abi",
        "//tpu_sync/kv_cache:kv_backend_factory",
        "//tpu_sync/kv_cache:kv_cache_store_backend",
        "//tpu_sync/kv_cache:kv_cache_store_backend_factory",
        "//tpu_sync/kv_cache/backends:backend",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "libmock_storage_plugin.so",
    testonly = True,
    srcs = ["mock_storage_plugin.cc"],
    copts = ["-fvisibility=hidden"],
    linkshared = True,
    deps = [
        ":raiden_storage_plugin_abi",
    ],
)

# Loads the mock plugin in tests. Depending on it puts the plugin in the
# test's runfiles.
cc_library(
    name = "mock_storage_plugin_loader",
    testonly = True,
    hdrs = ["mock_storage_plugin_loader.h"],
    data = [":libmock_storage_plugin.so"],
    linkopts = ["-ldl"],
    deps = [
        ":raiden_storage_plugin_abi",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "plugin_kv_backend_test",
    srcs = ["plugin_kv_backend_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    deps = [
        ":mock_storage_plugin_loader",
        ":plugin_kv_backend",
        ":raiden_storage_plugin_abi",
        "//tpu_sync/core:host_memory_allocator",
        "//tpu_sync/kv_cache/backends:backend",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "plugin_kv_cache_store_backend_test",
    srcs = ["plugin_kv_cache_store_backend_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    deps = [
        ":mock_storage_plugin_loader",
        ":plugin_kv_backend",
        ":plugin_kv_cache_store_backend",
        ":raiden_storage_plugin_abi",
        "//tpu_sync/kv_cache:kv_cache_store_backend",
        "//tpu_sync/kv_cache/backends:backend",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "plugin_registry_test",
    srcs = ["plugin_registry_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    deps = [
        ":mock_storage_plugin_loader",
        ":plugin_kv_backend",
        ":plugin_kv_cache_store_backend",
        ":plugin_registry",
        ":raiden_storage_plugin_abi",
        "//tpu_sync/kv_cache:kv_backend_factory",
        "//tpu_sync/kv_cache:kv_cache_store_backend",
        "//tpu_sync/kv_cache:kv_cache_store_backend_factory",
        "//tpu_sync/kv_cache/backends:backend",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/platform:statusor",
    ],
)
