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

"""Unit tests for JAX utils."""

import os
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from tpu_sync.frameworks.jax import utils

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


class UtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    try:
      self.devices = jax.devices("tpu")
    except RuntimeError:
      self.devices = jax.devices("cpu")
    self.mesh_2d = jax.sharding.Mesh(
        np.array(self.devices[:4]).reshape(2, 2), ("x", "y")
    )

  def test_get_host_subgrid_3d(self):
    # 4-chip host on 4x4x4 physical TPU torus mesh -> [1, 2, 2]
    self.assertEqual(utils._get_host_subgrid([4, 4, 4], 4), [1, 2, 2])
    # 4-chip host on 2x4x4 physical TPU torus mesh -> [1, 2, 2]
    self.assertEqual(utils._get_host_subgrid([2, 4, 4], 4), [1, 2, 2])
    # 8-chip host on 4x4x4 physical TPU torus mesh -> [1, 2, 4]
    self.assertEqual(utils._get_host_subgrid([4, 4, 4], 8), [1, 2, 4])
    # 4-chip host on 2D mesh [4, 8] -> [1, 4]
    self.assertEqual(utils._get_host_subgrid([4, 8], 4), [1, 4])
    # 4-chip host on 1D mesh [16] -> [4]
    self.assertEqual(utils._get_host_subgrid([16], 4), [4])

  def test_get_shard_sorting_permutation_aligned_2d(self):
    sharding = jax.sharding.NamedSharding(
        self.mesh_2d, jax.sharding.PartitionSpec("x", "y")
    )
    arr = jax.device_put(jnp.zeros((8, 8)), sharding)
    perm = utils.get_shard_sorting_permutation(arr)
    self.assertEqual(perm, [])

  def test_get_shard_sorting_permutation_transposed_2d(self):
    sharding = jax.sharding.NamedSharding(
        self.mesh_2d, jax.sharding.PartitionSpec("y", "x")
    )
    arr = jax.device_put(jnp.zeros((8, 8)), sharding)
    perm = utils.get_shard_sorting_permutation(arr)
    self.assertEqual(perm, [])

  def test_get_shard_sorting_permutation_replicated(self):
    sharding = jax.sharding.NamedSharding(
        self.mesh_2d, jax.sharding.PartitionSpec("x")
    )
    arr = jax.device_put(jnp.zeros((8, 8)), sharding)
    perm = utils.get_shard_sorting_permutation(arr)
    self.assertEqual(perm, [])

    sharding_empty = jax.sharding.NamedSharding(
        self.mesh_2d, jax.sharding.PartitionSpec()
    )
    arr_empty = jax.device_put(jnp.zeros((8, 8)), sharding_empty)
    perm_empty = utils.get_shard_sorting_permutation(arr_empty)
    self.assertEqual(perm_empty, [])

  def test_get_shard_sorting_permutation_single_shard(self):
    single_device = self.devices[0]
    sharding = jax.sharding.SingleDeviceSharding(single_device)
    arr = jax.device_put(jnp.zeros((8, 8)), sharding)
    perm = utils.get_shard_sorting_permutation(arr)
    self.assertEqual(perm, [])

  def test_strict_bijection_failure_raises_value_error(self):
    sharding = jax.sharding.NamedSharding(
        self.mesh_2d, jax.sharding.PartitionSpec("x", "y")
    )
    arr = jax.device_put(jnp.zeros((8, 8)), sharding)

    mock_arr = mock.MagicMock()
    mock_arr.shape = arr.shape
    mock_arr.sharding = arr.sharding
    mock_arr.addressable_shards = [
        arr.addressable_shards[0],
        arr.addressable_shards[0],
        arr.addressable_shards[2],
        arr.addressable_shards[3],
    ]
    with self.assertRaises(ValueError) as ctx:
      utils.get_shard_sorting_permutation(mock_arr)
    self.assertIn("Strict bijection failed", str(ctx.exception))

  def test_non_contiguous_subgrid_4x4x4_mesh_permutation(self):
    mock_devices = np.empty((4, 4, 4), dtype=object)
    for i in range(4):
      for j in range(4):
        for k in range(4):
          mock_devices[i, j, k] = mock.MagicMock(spec=jax.Device)

    mesh_3d = jax.sharding.Mesh(mock_devices, ("x", "y", "z"))
    sharding_3d = jax.sharding.NamedSharding(
        mesh_3d, jax.sharding.PartitionSpec("x", "y", "z")
    )

    # Host 0 chips in 1x2x2 subgrid: (0,0,0)->0, (0,0,1)->1, (0,1,0)->4, (0,1,1)->5
    host0_devices = [
        mock_devices[0, 0, 0],
        mock_devices[0, 0, 1],
        mock_devices[0, 1, 0],
        mock_devices[0, 1, 1],
    ]
    host0_shards = [mock.MagicMock(device=d) for d in host0_devices]

    mock_arr0 = mock.MagicMock()
    mock_arr0.shape = (16, 16, 16)
    mock_arr0.sharding = sharding_3d
    mock_arr0.addressable_shards = host0_shards

    with mock.patch.object(
        jax, "process_index", return_value=0
    ), mock.patch.object(jax, "process_count", return_value=16):
      perm0 = utils.get_shard_sorting_permutation(mock_arr0)
      self.assertEqual(perm0, [])

    # Permuted addressable shards: should compute the exact correcting permutation
    permuted_shards = [
        host0_shards[1],
        host0_shards[0],
        host0_shards[3],
        host0_shards[2],
    ]
    mock_arr0.addressable_shards = permuted_shards
    with mock.patch.object(
        jax, "process_index", return_value=0
    ), mock.patch.object(jax, "process_count", return_value=16):
      perm_reordered = utils.get_shard_sorting_permutation(mock_arr0)
      self.assertEqual(perm_reordered, [1, 0, 3, 2])


if __name__ == "__main__":
  absltest.main()
