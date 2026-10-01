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

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Carries the C++ version gate. Anything that includes the compat layer gets
# -DRAIDEN_JAX from here, and nothing else in the build does.
cc_library(
    name = "raiden_jax_version",
    defines = select({
        ":jax_1000": ["RAIDEN_JAX=1000"],
        ":jax_1001": ["RAIDEN_JAX=1001"],
        ":jax_1002": ["RAIDEN_JAX=1002"],
        ":jax_1100": ["RAIDEN_JAX=1100"],
        ":jax_1101": ["RAIDEN_JAX=1101"],
        ":jax_1102": ["RAIDEN_JAX=1102"],
        "//conditions:default": ["RAIDEN_JAX=1102"],
    }),
)

config_setting(
    name = "jax_1000",
    define_values = {"raiden_jax": "1000"},
)

config_setting(
    name = "jax_1001",
    define_values = {"raiden_jax": "1001"},
)

config_setting(
    name = "jax_1002",
    define_values = {"raiden_jax": "1002"},
)

config_setting(
    name = "jax_1100",
    define_values = {"raiden_jax": "1100"},
)

config_setting(
    name = "jax_1101",
    define_values = {"raiden_jax": "1101"},
)

config_setting(
    name = "jax_1102",
    define_values = {"raiden_jax": "1102"},
)
