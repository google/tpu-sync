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

"""Dependency revisions for JAX 0.10.0.

Every value except libtpu, raiden_jax, and abseil patches is read from JAX
a33ed614's MODULE.bazel. See README.md for patch details.
"""

DEPS = {
    "jax_version": "0.10.0",
    "jaxlib_version": "0.10.0",
    "raiden_jax": 1000,
    "jax_repo": "https://github.com/jax-ml/jax",
    "jax_commit": "a33ed614c58ee8a10d0b7536c50c2609c38500c1",
    "jax_patches": [],
    "xla_repo": "https://github.com/openxla/xla",
    "xla_commit": "b6f37ab7767f428fd6f993de5e211643d47d4deb",
    "xla_integrity": "sha256-aiX/5xLCdGj+ZCyNB/TeVgEJK/MfTR7dSD7u+Ws54tk=",
    "xla_patches": [
        "third_party/xla/future_sfinae.patch",
        "third_party/xla/attribute_map_static_assert.patch",
        "third_party/xla/attribute_map_cc_static_assert.patch",
    ],
    "rules_ml_toolchain_repo": "https://github.com/google-ml-infra/rules_ml_toolchain",
    "rules_ml_toolchain_commit": "99c43dfe995a0e81c767d5b6d686191992672fe6",
    "rules_ml_toolchain_integrity": "sha256-8skk6Foiui6qDAhlfl9UZ/7bw9BQb5zAxp3Zftn7ryg=",
    "rules_ml_toolchain_patches": [
        "third_party/jax/0.10.0/patches/rules_ml_toolchain/no_register_toolchains.patch",
    ],
    "absl_repo": "https://github.com/abseil/abseil-cpp",
    "absl_version": "20260107.1",
    "absl_patches": [
        "third_party/abseil/status_macros.patch",
    ],
    "libtpu_version": "0.0.40",
}
