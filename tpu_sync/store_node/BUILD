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
    name = "kv_transfer_spec_source",
    hdrs = ["kv_transfer_spec_source.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "grs_kv_transfer_spec_source",
    srcs = ["grs_kv_transfer_spec_source.cc"],
    hdrs = ["grs_kv_transfer_spec_source.h"],
    deps = [
        ":kv_transfer_spec_source",
        "//tpu_sync/core:status_macros",
        "//tpu_sync/kv_cache/global_registry:global_registry_cc_proto",
        "//tpu_sync/kv_cache/global_registry:global_registry_client_cc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "grs_kv_transfer_spec_source_test",
    srcs = ["grs_kv_transfer_spec_source_test.cc"],
    deps = [
        ":grs_kv_transfer_spec_source",
        ":kv_transfer_spec_source",
        "//tpu_sync/kv_cache/global_registry:global_registry_cc_proto",
        "//tpu_sync/kv_cache/global_registry:global_registry_client_cc",
        "//tpu_sync/kv_cache/global_registry:global_registry_server_lib",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "kv_cache_host_store_node",
    srcs = ["kv_cache_host_store_node.cc"],
    hdrs = ["kv_cache_host_store_node.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":kv_transfer_spec_source",
        "//tpu_sync/common:raiden_id",
        "//tpu_sync/core:kv_cache_manager_with_transfer",
        "//tpu_sync/core:kv_manager_holder",
        "//tpu_sync/core:status_macros",
        "//tpu_sync/core/controller:controller_client",
        "//tpu_sync/core/controller:worker_service_server",
        "//tpu_sync/kv_cache:kv_cache_store",
        "//tpu_sync/kv_cache:kv_cache_store_backend_factory",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_binary(
    name = "kv_cache_host_store_node_main",
    srcs = ["kv_cache_host_store_node_main.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":grs_kv_transfer_spec_source",
        ":kv_cache_host_store_node",
        "//tpu_sync/common:raiden_id",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "kv_cache_host_store_node_test",
    srcs = ["kv_cache_host_store_node_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":grs_kv_transfer_spec_source",
        ":kv_cache_host_store_node",
        ":kv_transfer_spec_source",
        "//tpu_sync/common:raiden_id",
        "//tpu_sync/kv_cache/global_registry:global_registry_cc_proto",
        "//tpu_sync/kv_cache/global_registry:global_registry_client_cc",
        "//tpu_sync/kv_cache/global_registry:global_registry_server_lib",
        "//tpu_sync/kv_cache/global_registry:test_util",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "kv_cache_evict_e2e_test",
    srcs = ["kv_cache_evict_e2e_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":kv_cache_host_store_node",
        ":kv_transfer_spec_source",
        "//tpu_sync/common:raiden_id",
        "//tpu_sync/core:kv_cache_manager_with_transfer",
        "//tpu_sync/core:kv_manager_holder",
        "//tpu_sync/core:status_macros",
        "//tpu_sync/core/controller:controller_client",
        "//tpu_sync/core/controller:worker_service_server",
        "//tpu_sync/kv_cache:kv_cache_store",
        "//tpu_sync/kv_cache:kv_cache_store_backend",
        "//tpu_sync/kv_cache:kv_cache_store_backend_factory",
        "//tpu_sync/kv_cache/global_registry:test_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
