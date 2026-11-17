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

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

# Release verification and preflight tooling behind
# .github/workflows/release_wheels.yml; see RELEASING.md. Standard library
# only, so the OSS workflow runs it on a stock python3 without a Bazel build.
py_library(
    name = "release_checks_lib",
    srcs = ["release_checks.py"],
)

py_binary(
    name = "release_checks",
    srcs = ["release_checks.py"],
    deps = [":release_checks_lib"],
)

py_test(
    name = "release_checks_test",
    srcs = ["release_checks_test.py"],
    deps = [":release_checks_lib"],
)
