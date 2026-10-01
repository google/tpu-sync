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

"""Dependency revisions for JAX 0.10.1.

Every value except libtpu, raiden_jax, and abseil patches is read from JAX
619764c1's MODULE.bazel. See README.md for patch details.
"""

DEPS = {
    "jax_version": "0.10.1",
    "jaxlib_version": "0.10.1",
    "raiden_jax": 1001,
    "jax_repo": "https://github.com/jax-ml/jax",
    "jax_commit": "619764c15117fbefc4ba13ab941871cb514c23f6",
    "jax_patches": [
        "third_party/jax/0.10.1/patches/py/jax_remove_local_wheels.patch",
    ],
    "xla_repo": "https://github.com/openxla/xla",
    "xla_commit": "9b635916ecc6df6efee62d8e4b0c7ef87ef84d69",
    "xla_integrity": "sha256-SrTyccoJ+p5DG5r13tgj/PA5MtlK2Y9pV/E1I/U7oHA=",
    "xla_patches": [
        "third_party/xla/future_sfinae.patch",
        "third_party/xla/attribute_map_static_assert.patch",
        "third_party/xla/attribute_map_cc_static_assert.patch",
    ],
    "rules_ml_toolchain_repo": "https://github.com/google-ml-infra/rules_ml_toolchain",
    "rules_ml_toolchain_commit": "84ac62e4db38215a2a7d3ad8cd4d7452134e3ec6",
    "rules_ml_toolchain_integrity": "sha256-y3Y1Yi6lJsiSOl6tbpncK6Vc3mrpYTv5stNLPwAPKc4=",
    "rules_ml_toolchain_patches": [
        "third_party/jax/0.10.1/patches/rules_ml_toolchain/no_register_toolchains.patch",
    ],
    "absl_repo": "https://github.com/abseil/abseil-cpp",
    "absl_version": "20260107.1",
    "absl_patches": [
        "third_party/abseil/status_macros.patch",
    ],
    "libtpu_version": "0.0.41",
}
