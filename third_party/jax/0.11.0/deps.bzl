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

"""Dependency revisions for jax 0.11.0."""

DEPS = {
    "jax_version": "0.11.0",
    "jaxlib_version": "0.11.0",
    "raiden_jax": 1100,
    "jax_repo": "https://github.com/jax-ml/jax",
    "jax_commit": "a1521744c6dc074443fe549f19f48d7197abf759",
    "jax_patches": [
        "third_party/jax/0.11.0/patches/py/jax_remove_local_wheels.patch",
    ],
    "xla_repo": "https://github.com/openxla/xla",
    "xla_commit": "131bf41acb4650e4391a640c3f1859c1c86ad74b",
    "xla_integrity": "sha256-Mr7ZKBuhU8Z6dbZY4kDdvNFzbnkprU7T0T22kba/ybM=",
    "xla_patches": [
        "third_party/xla/future_sfinae.patch",
        "third_party/xla/attribute_map_static_assert.patch",
        "third_party/xla/attribute_map_cc_static_assert.patch",
    ],
    "rules_ml_toolchain_repo": "https://github.com/google-ml-infra/rules_ml_toolchain",
    "rules_ml_toolchain_commit": "b11745590f513ec55b32e2d126073576fde18c71",
    "rules_ml_toolchain_integrity": "sha256-axspTL7LmKCLz+WWQUYfcuET2zD6kbRPzEnZgJm1umU=",
    "rules_ml_toolchain_patches": [
        "third_party/jax/0.11.0/patches/rules_ml_toolchain/no_register_toolchains.patch",
    ],
    "absl_repo": "https://github.com/abseil/abseil-cpp",
    "absl_version": "20260526.0",
    "absl_patches": [],
    "libtpu_version": "0.0.47",
}
