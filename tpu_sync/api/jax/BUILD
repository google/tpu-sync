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

load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "kv_cache_manager_jax_py",
    srcs = ["kv_cache_manager.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/frameworks/jax:_tpu_raiden_jax",
        "//tpu_sync/frameworks/jax:jax_test_utils",
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

py_library(
    name = "weight_synchronizer_jax_py",
    srcs = [
        "weight_synchronizer.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/api:common",
        "//tpu_sync/api:weight_sync_common",
        "//tpu_sync/frameworks/jax:_tpu_raiden_jax",
        "//tpu_sync/frameworks/jax:jax_test_utils",
        "@jax//jax",
    ],
)

# Keep Copybara happy by maintaining at least one internal reference target prefix reference:
# //:dummy_target
