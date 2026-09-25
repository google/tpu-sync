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

"""Unit tests for weight_synchronizer_ffi APIs on TPU."""

import os
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tpu_sync.frameworks.jax import weight_synchronizer_ffi


class WeightSynchronizerFfiTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.devices = jax.devices()
    self.num_devices = len(self.devices)
    self.mesh = jax.sharding.Mesh(np.array(self.devices), ("x",))
    self.weight_sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec("x", None)
    )
    self.shard_idx = jax.device_put(
        jnp.arange(self.num_devices, dtype=jnp.int32),
        jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec("x")),
    )

  def tearDown(self):
    weight_synchronizer_ffi.destroy_weight_synchronizer()
    os.environ.pop("RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER", None)
    super().tearDown()

  @parameterized.named_parameters(
      ("legacy", "0"),
      ("zerocopy", "1"),
  )
  def test_init_weight_synchronizer_and_d2h(self, direct_buffer_env):
    os.environ["RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER"] = direct_buffer_env
    w0 = jax.device_put(
        jnp.arange(self.num_devices * 16, dtype=jnp.bfloat16).reshape(
            self.num_devices, 16
        ),
        self.weight_sharding,
    )
    w1 = jax.device_put(
        jnp.ones((self.num_devices, 32), dtype=jnp.bfloat16),
        self.weight_sharding,
    )
    device_arrays = [w0, w1]
    slice_byte_sizes = jax.device_put(
        jnp.array([16 * 2, 32 * 2], dtype=jnp.int32),
        jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(None)),
    )

    meta = weight_synchronizer_ffi.init_weight_synchronizer_and_d2h(
        device_arrays=device_arrays,
        shard_idx=self.shard_idx,
        mesh=self.mesh,
        slice_byte_sizes=slice_byte_sizes,
        local_port=0,
        parallelism=1,
        num_layers=len(device_arrays),
        listener_port=0,
        num_shards=self.num_devices,
    )
    jax.block_until_ready(meta)

    meta_np = np.asarray(meta)
    self.assertEqual(meta_np.shape, (self.num_devices, 6))
    for rank in range(self.num_devices):
      self.assertGreater(int(meta_np[rank, 4]), 0)
      self.assertGreater(int(meta_np[rank, 5]), 0)

  @parameterized.named_parameters(
      ("legacy", "0"),
      ("zerocopy", "1"),
  )
  def test_d2h_and_h2d(self, direct_buffer_env):
    os.environ["RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER"] = direct_buffer_env
    src_w0 = jax.device_put(
        jnp.arange(self.num_devices * 16, dtype=jnp.bfloat16).reshape(
            self.num_devices, 16
        ),
        self.weight_sharding,
    )
    src_w1 = jax.device_put(
        jnp.full((self.num_devices, 32), 3.5, dtype=jnp.bfloat16),
        self.weight_sharding,
    )
    slice_byte_sizes = jax.device_put(
        jnp.array([16 * 2, 32 * 2], dtype=jnp.int32),
        jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(None)),
    )

    meta = weight_synchronizer_ffi.init_weight_synchronizer(
        device_array=src_w0,
        shard_idx=self.shard_idx,
        mesh=self.mesh,
        slice_byte_sizes=slice_byte_sizes,
        local_port=0,
        parallelism=1,
        num_layers=2,
        listener_port=0,
        num_shards=self.num_devices,
    )
    jax.block_until_ready(meta)

    d2h_w0 = weight_synchronizer_ffi.d2h(
        src_w0, self.shard_idx, self.mesh, layer_idx=0
    )
    d2h_w1 = weight_synchronizer_ffi.d2h(
        src_w1, self.shard_idx, self.mesh, layer_idx=1
    )
    jax.block_until_ready(d2h_w0)
    jax.block_until_ready(d2h_w1)

    dst_w0_init = jax.device_put(
        jnp.zeros_like(src_w0),
        self.weight_sharding,
    )
    dst_w1_init = jax.device_put(
        jnp.zeros_like(src_w1),
        self.weight_sharding,
    )

    dst_w0 = weight_synchronizer_ffi.h2d(
        dst_w0_init, self.shard_idx, self.mesh, layer_idx=0
    )
    dst_w1 = weight_synchronizer_ffi.h2d(
        dst_w1_init, self.shard_idx, self.mesh, layer_idx=1
    )
    jax.block_until_ready(dst_w0)
    jax.block_until_ready(dst_w1)

    np.testing.assert_array_equal(np.asarray(dst_w0), np.asarray(src_w0))
    np.testing.assert_array_equal(np.asarray(dst_w1), np.asarray(src_w1))

  @parameterized.named_parameters(
      ("legacy", "0"),
      ("zerocopy", "1"),
  )
  def test_multi_h2d(self, direct_buffer_env):
    os.environ["RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER"] = direct_buffer_env
    src_w0 = jax.device_put(
        jnp.arange(self.num_devices * 16, dtype=jnp.bfloat16).reshape(
            self.num_devices, 16
        ),
        self.weight_sharding,
    )
    src_w1 = jax.device_put(
        jnp.full((self.num_devices, 32), 7.25, dtype=jnp.bfloat16),
        self.weight_sharding,
    )
    src_arrays = [src_w0, src_w1]
    slice_byte_sizes = jax.device_put(
        jnp.array([16 * 2, 32 * 2], dtype=jnp.int32),
        jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(None)),
    )

    meta = weight_synchronizer_ffi.init_weight_synchronizer_and_d2h(
        device_arrays=src_arrays,
        shard_idx=self.shard_idx,
        mesh=self.mesh,
        slice_byte_sizes=slice_byte_sizes,
        local_port=0,
        parallelism=1,
        num_layers=len(src_arrays),
        listener_port=0,
        num_shards=self.num_devices,
    )
    jax.block_until_ready(meta)

    dst_templates = [
        jax.device_put(jnp.zeros_like(arr), self.weight_sharding)
        for arr in src_arrays
    ]
    restored = weight_synchronizer_ffi.multi_h2d(
        dst_templates, self.shard_idx, self.mesh
    )
    for orig, rest in zip(src_arrays, restored):
      jax.block_until_ready(rest)
      np.testing.assert_array_equal(np.asarray(rest), np.asarray(orig))


if __name__ == "__main__":
  absltest.main()
