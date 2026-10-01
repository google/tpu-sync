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

"""Dependency revisions for jax 0.11.2 (default version).

Values are sourced from JAX's MODULE.bazel to maintain ABI compatibility with
jaxlib. Default values mirror root MODULE.bazel and requirements.txt.
"""

DEPS = {
    "jax_version": "0.11.2",
    "jaxlib_version": "0.11.2",
    "raiden_jax": 1102,
    "jax_repo": "https://github.com/jax-ml/jax",
    "jax_commit": "32544801e26115ac1794926d027148abf3baf009",
    "jax_patches": [
        "third_party/py/jax_remove_local_wheels.patch",
        "third_party/py/jax_mosaic_gpu_mbarrier_layout.patch",
    ],
    "xla_repo": "https://github.com/openxla/xla",
    "xla_commit": "f60be94c9c5d1e1340a6259433ead57824c37293",
    "xla_patches": [
        "third_party/xla/future_sfinae.patch",
        "third_party/xla/attribute_map_static_assert.patch",
        "third_party/xla/attribute_map_cc_static_assert.patch",
    ],
    "rules_ml_toolchain_repo": "https://github.com/google-ml-infra/rules_ml_toolchain",
    "rules_ml_toolchain_commit": "c0eb2743b7b12b2bbcf0e1888e26d36ba6b093de",
    "rules_ml_toolchain_integrity": "sha256-AD+fcA+AoQ9wvsdh0AUpI30WpuIMBjEkyf/6mKjArzU=",
    "rules_ml_toolchain_patches": [
        "third_party/rules_ml_toolchain/no_register_toolchains.patch",
    ],
    "absl_repo": "https://github.com/abseil/abseil-cpp",
    "absl_version": "20260526.0",
    "absl_patches": [],
    "libtpu_version": "0.0.48",
}
