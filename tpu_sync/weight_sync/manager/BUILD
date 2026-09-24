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
    name = "broadcast_engine",
    srcs = ["broadcast_engine.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_sync/api:common",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "broadcast_engine_test",
    srcs = ["broadcast_engine_test.py"],
    deps = [
        ":broadcast_engine",
        "//tpu_sync/api:common",
        "//tpu_sync/rpc:raiden_controller",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "controller_types",
    srcs = ["controller_types.py"],
    deps = [
        "//tpu_sync/api:common",
        "//tpu_sync/rpc:raiden_service_py_pb2",
    ],
)

py_library(
    name = "job_entity",
    srcs = ["job_entity.py"],
    deps = [
        ":controller_types",
        "//tpu_sync/api:common",
        "//tpu_sync/common/control_pipe:control_pipe_client",
        "//tpu_sync/rpc:raiden_service_py_pb2",
        "@com_google_absl_py//absl/logging",
    ],
)

py_library(
    name = "reshard_planner",
    srcs = ["reshard_planner.py"],
    deps = [
        ":broadcast_engine",
        ":controller_types",
        ":job_entity",
        "//tpu_sync/api:common",
        "//tpu_sync/kv_cache:nd_slice_math",
        "@com_google_absl_py//absl/logging",
    ],
)

py_test(
    name = "reshard_planner_test",
    timeout = "long",
    srcs = ["reshard_planner_test.py"],
    deps = [
        ":controller_types",
        ":job_entity",
        ":reshard_planner",
        "//tpu_sync/api:common",
        "//tpu_sync/rpc:raiden_service_py_pb2",
        "@com_google_absl_py//absl/logging",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "controller_rpc_service",
    srcs = ["controller_rpc_service.py"],
    deps = [
        ":controller_types",
        "//tpu_sync/api:common",
        "//tpu_sync/common/control_pipe:control_pipe_client",
        "//tpu_sync/rpc:controller_service_py_pb2",
        "//tpu_sync/rpc:raiden_service_py_pb2",
    ],
)
