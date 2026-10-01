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

"""Dependency revisions for jax 0.11.1."""

DEPS = {
    "jax_version": "0.11.1",
    "jaxlib_version": "0.11.1",
    "raiden_jax": 1101,
    "jax_repo": "https://github.com/jax-ml/jax",
    "jax_commit": "2d66622450e2c8633cda2307688ef7aa294bd6eb",
    "jax_patches": [
        "third_party/jax/0.11.1/patches/py/jax_remove_local_wheels.patch",
    ],
    "xla_repo": "https://github.com/openxla/xla",
    "xla_commit": "f85cfbe2e260907a52bbbe85942462377c8fdb62",
    "xla_integrity": "sha256-6g6q14v0kXvj0l2iE7k/Q7v3N8b5N3a9u4o0m6Y2o3g=",
    "xla_patches": [
        "third_party/xla/future_sfinae.patch",
        "third_party/xla/attribute_map_static_assert.patch",
        "third_party/xla/attribute_map_cc_static_assert.patch",
    ],
    "rules_ml_toolchain_repo": "https://github.com/google-ml-infra/rules_ml_toolchain",
    "rules_ml_toolchain_commit": "73cb731fed3c9215033c5e21e64906f376cf47e8",
    "rules_ml_toolchain_integrity": "sha256-oD1n5+UfPq1d8T9sK0m2W4j6N8b5N3a9u4o0m6Y2o3g=",
    "rules_ml_toolchain_patches": [
        "third_party/jax/0.11.1/patches/rules_ml_toolchain/no_register_toolchains.patch",
    ],
    "absl_repo": "https://github.com/abseil/abseil-cpp",
    "absl_version": "20260526.0",
    "absl_patches": [],
    "libtpu_version": "0.0.48",
}
