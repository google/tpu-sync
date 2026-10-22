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

"""Dependency revisions for jax 0.12.0."""

DEPS = {
    "jax_version": "0.12.0",
    "jaxlib_version": "0.12.0",
    "raiden_jax": 1200,
    "jax_repo": "https://github.com/jax-ml/jax",
    "jax_commit": "8dcb7aeced770bd22531b42d0b925c7342787c27",
    "jax_patches": [
        "third_party/py/jax_remove_local_wheels.patch",
    ],
    "xla_repo": "https://github.com/openxla/xla",
    "xla_commit": "d867969800d4a8fd045505eabc65e1e66f30794f",
    "xla_patches": [
        "third_party/xla/future_sfinae.patch",
        "third_party/xla/attribute_map_static_assert.patch",
        "third_party/xla/attribute_map_cc_static_assert.patch",
    ],
    "rules_ml_toolchain_repo": "https://github.com/google-ml-infra/rules_ml_toolchain",
    "rules_ml_toolchain_commit": "3a4df0c8e9e51687ac4c62c793a00a27286e5e83",
    "rules_ml_toolchain_integrity": "sha256-9+4rpwQMv77PxA+AGMWpwcjioakWHi4nJ7UQ0iPHrJo=",
    "rules_ml_toolchain_patches": [
        "third_party/rules_ml_toolchain/no_register_toolchains.patch",
    ],
    "absl_repo": "https://github.com/abseil/abseil-cpp",
    "absl_version": "20260526.0",
    "absl_patches": [],
    "libtpu_version": "0.0.50",
}
