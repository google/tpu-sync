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

load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "kv_cache_manager_jax_py",
    srcs = ["kv_cache_manager.py"],
    visibility = ["//visibility:public"],
    deps = [
        # Framework-neutral despite living under api/torch: pure dataclasses,
        # no torch import. Shared rather than duplicated so the JAX and torch
        # pool descriptors cannot drift.
        "//tpu_sync/api/torch:pool_layout",
        "//tpu_sync/frameworks/jax:_tpu_raiden_jax",
        "//tpu_sync/frameworks/jax:jax_test_utils",
    ],
)

py_test(
    name = "kv_cache_manager_pool_test",
    srcs = ["kv_cache_manager_pool_test.py"],
    args = ["--device_type=cpu"],
    # TODO(jawadamin): drop "notap" once //tpu_sync/kv_cache:kv_cache_store
    # compiles again. Every py_test in this package links it and none of them build today.
    tags = [
        "cpu:2",
        "notap",
    ],
    deps = [
        ":kv_cache_manager_jax_py",
        "//tpu_sync/api/torch:pool_layout",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/testing:absltest",
        "@jax//jax",
        "@pypi//numpy",
    ],
)

py_library(
    name = "kv_cache_store",
    srcs = ["kv_cache_store.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/api:common",
        "//tpu_sync/frameworks/jax:_tpu_raiden_jax",
    ],
)

py_test(
    name = "kv_cache_store_test",
    srcs = ["kv_cache_store_test.py"],
    data = [
        "//tpu_sync/kv_cache/global_registry:global_registry_server",
    ],
    tags = [
        "manual",
    ],
    deps = [
        ":kv_cache_store",
        ":kv_cache_manager_jax_py",
        "//pyglib:resources",
        "@com_google_absl_py//absl/testing:absltest",
        "@jax//jax",
        "@jax//jax/experimental:mesh_utils",
        "@pypi//numpy",
        # buildcleaner: keep
        "//tpu_sync/frameworks/jax:_raw_transfer",
    ],
)

py_test(
    name = "kv_cache_store_recovery_e2e_test",
    srcs = ["kv_cache_store_recovery_e2e_test.py"],
    deps = [
        ":kv_cache_store",
        ":kv_cache_manager_jax_py",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl:app",
        "@jax//jax",
        "@pypi//numpy",
        # buildcleaner: keep
        "//tpu_sync/frameworks/jax:_raw_transfer",
    ],
)

py_library(
    name = "weight_synchronizer_jax_py",
    srcs = [
        "weight_synchronizer.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/frameworks/jax:_tpu_raiden_jax",
        "//tpu_sync/frameworks/jax:jax_test_utils",
    ],
)

py_test(
    name = "weight_synchronizer_test",
    srcs = ["weight_synchronizer_test.py"],
    tags = [],
    deps = [
        ":weight_synchronizer_jax_py",
        "//tpu_sync/frameworks/jax:jax_test_utils",
        "//tpu_sync/rpc:raiden_service_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
        "@jax//jax",
        "@pypi//numpy",
    ],
)

# Keep Copybara happy by maintaining at least one internal reference target prefix reference:
# //:dummy_target
