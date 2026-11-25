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

"""Shared py_wheel template for the per-framework tpu_sync wheels.

Keeps the tpu_sync_jax and tpu_sync_torch wheels identical except for their
distribution name, requires list, and package contents.
"""

load("@raiden_version//:version.bzl", "SUPPORTED_PYTHON_VERSIONS", "WHEEL_ABI_TAG", "WHEEL_PYTHON_TAG")
load("@rules_python//python:packaging.bzl", "py_wheel")

def raiden_framework_wheel(
        name,
        distribution,
        package,
        requires,
        version,
        entry_points = {}):
    """A tpu_raiden wheel for a single framework (jax or torch).

    Args:
      name: wheel target name (e.g. raiden_torch_wheel).
      distribution: PyPI distribution name (e.g. tpu_sync_torch).
      package: the py_package target whose tpu_raiden files go in the wheel.
      requires: runtime dependency list for the wheel metadata.
      version: wheel version (typically @raiden_version//:WHEEL_VERSION).
      entry_points: py_wheel entry_points dict, e.g.
        {"console_scripts": ["name = module:func"]}.
    """
    py_wheel(
        entry_points = entry_points,
        name = name,
        abi = WHEEL_ABI_TAG,
        author = "TPU Raiden team",
        classifiers = [
            "Development Status :: 3 - Alpha",
            "License :: OSI Approved :: Apache Software License",
        ] + [
            "Programming Language :: Python :: {}".format(v)
            for v in SUPPORTED_PYTHON_VERSIONS
        ] + [
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        description_file = "//:README.md",
        distribution = distribution,
        homepage = "https://github.com/google/tpu-raiden",
        license = "Apache 2.0",
        platform = "manylinux_2_31_x86_64",
        python_requires = ">=3.12",
        python_tag = WHEEL_PYTHON_TAG,
        requires = requires,
        summary = "High-bandwidth PJRT raw-transfer / KV-cache DMA engine for TPU.",
        tags = ["manual"],
        # Base version comes from pyproject.toml; .dev<timestamp> appended via
        # WHEEL_VERSION_EXTRAS at build time.
        version = version,
        visibility = ["//visibility:public"],
        deps = [package],
    )
