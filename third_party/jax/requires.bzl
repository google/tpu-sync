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

"""JAX stack requirements for TPU Raiden wheels based on selected JAX version."""

load(":0.10.0/deps.bzl", _deps_0_10_0 = "DEPS")
load(":0.10.1/deps.bzl", _deps_0_10_1 = "DEPS")
load(":0.10.2/deps.bzl", _deps_0_10_2 = "DEPS")
load(":0.11.0/deps.bzl", _deps_0_11_0 = "DEPS")
load(":0.11.1/deps.bzl", _deps_0_11_1 = "DEPS")
load(":0.11.2/deps.bzl", _deps_0_11_2 = "DEPS")

_DEPS_BY_SETTING = {
    "//third_party/jax:jax_1000": _deps_0_10_0,
    "//third_party/jax:jax_1001": _deps_0_10_1,
    "//third_party/jax:jax_1002": _deps_0_10_2,
    "//third_party/jax:jax_1100": _deps_0_11_0,
    "//third_party/jax:jax_1101": _deps_0_11_1,
    "//third_party/jax:jax_1102": _deps_0_11_2,
}

_DEFAULT_DEPS = _deps_0_11_2

def _specs(deps):
    """Returns wheel requirement specs for jax, jaxlib, and libtpu."""
    return [
        "jax==" + deps["jax_version"],
        "jaxlib==" + deps["jaxlib_version"],
        "libtpu==" + deps["libtpu_version"],
    ]

def jax_stack_requires():
    """Selects the JAX wheel requirements for the configured `--define raiden_jax`.

    Returns:
      A select() mapping each JAX version setting to its jax, jaxlib, and
      libtpu requirement specs, defaulting to the default version's specs.
    """
    branches = {}
    for setting, deps in _DEPS_BY_SETTING.items():
        branches[setting] = _specs(deps)
    branches["//conditions:default"] = _specs(_DEFAULT_DEPS)
    return select(branches)
