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

"""Dependency revisions for JAX 0.10.2.

Every value except libtpu, raiden_jax, and abseil patches is read from JAX
990e6a0b's MODULE.bazel. See README.md for patch details.
"""

DEPS = {
    "jax_version": "0.10.2",
    "jaxlib_version": "0.10.2",
    "raiden_jax": 1002,
    "jax_repo": "https://github.com/jax-ml/jax",
    "jax_commit": "990e6a0b84138346e6a38785412f36356e0e5dc3",
    "jax_patches": [
        "third_party/jax/0.10.2/patches/py/jax_remove_local_wheels.patch",
    ],
    "xla_repo": "https://github.com/openxla/xla",
    "xla_commit": "5a9e73cbd92530cac2ac36f4736a774b2412afe2",
    "xla_integrity": "sha256-CKUiEKBM1o049iAdVic9BKwPi04Nqfcmd8dKSMxjdCI=",
    "xla_patches": [
        "third_party/xla/future_sfinae.patch",
        "third_party/xla/attribute_map_static_assert.patch",
        "third_party/xla/attribute_map_cc_static_assert.patch",
    ],
    "rules_ml_toolchain_repo": "https://github.com/google-ml-infra/rules_ml_toolchain",
    "rules_ml_toolchain_commit": "cad1047facbac4fb3c1124da68bf2cb36c7eb9ac",
    "rules_ml_toolchain_integrity": "sha256-QJY+S8Ji36mkMUb2EBQK8AaLAjrOjzxQ8XBae1DeCDA=",
    "rules_ml_toolchain_patches": [
        "third_party/jax/0.10.2/patches/rules_ml_toolchain/no_register_toolchains.patch",
    ],
    "absl_repo": "https://github.com/abseil/abseil-cpp",
    "absl_version": "20260107.1",
    "absl_patches": [
        "third_party/abseil/status_macros.patch",
    ],
    "libtpu_version": "0.0.42.1",
}
