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

"""Pool-admission and control-plane parity tests for the JAX manager.

These mirror api/torch/kv_cache_manager_host_test.py: the JAX manager must
expose the same pool surface, with the same shapes and the same errors, so a
caller (e.g. the TPU vLLM connector) can drive either framework identically.
"""

from absl import flags
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from tpu_sync.api.jax import kv_cache_manager as jax_kv
from tpu_sync.api.torch import pool_layout

flags.DEFINE_string(
    "device_type",
    "tpu",
    "JAX backend to run against (e.g. 'tpu', 'cpu').",
)

_NUM_BLOCKS = 4
_BLOCK_SHAPE = (8, 4, 128)
_DTYPE = jnp.float32


def _block_bytes() -> int:
  return int(np.prod(_BLOCK_SHAPE)) * jnp.dtype(_DTYPE).itemsize


def _dense_pool(tag: str, storage_index: int) -> pool_layout.PoolSpec:
  """One pool spanning every byte of every block of one storage."""
  block_bytes = _block_bytes()
  return pool_layout.PoolSpec(
      tag=tag,
      storage_index=storage_index,
      base_offset_bytes=0,
      block_stride_bytes=block_bytes,
      num_blocks=_NUM_BLOCKS,
      regions=(
          pool_layout.RegionSpec(
              name="payload",
              offset_bytes=0,
              stride_bytes=block_bytes,
              unit_bytes=block_bytes,
              num_units=1,
          ),
      ),
      dtype_tag="float32",
  )


def _half_block_pool(tag: str, storage_index: int) -> pool_layout.PoolSpec:
  """A pool declaring only the first half of each block as live.

  Exercises the case the reshard path actually uses: block stride and live
  extent differ, so a wrong offset shows up as corruption in the untouched
  half rather than as an error.
  """
  block_bytes = _block_bytes()
  return pool_layout.PoolSpec(
      tag=tag,
      storage_index=storage_index,
      base_offset_bytes=0,
      block_stride_bytes=block_bytes,
      num_blocks=_NUM_BLOCKS,
      regions=(
          pool_layout.RegionSpec(
              name="payload",
              offset_bytes=0,
              stride_bytes=block_bytes,
              unit_bytes=block_bytes // 2,
              num_units=1,
          ),
      ),
      dtype_tag="float32",
  )


class _ManagerTestBase(absltest.TestCase):
  """Shared fixture: one single-device, single-shard JAX manager."""

  def setUp(self):
    super().setUp()
    device_type = flags.FLAGS.device_type
    try:
      self.devices = jax.devices(device_type)
    except RuntimeError as exc:
      raise AssertionError(f"No {device_type} devices found") from exc
    if not self.devices:
      raise AssertionError(f"No {device_type} devices found")

  def _make_manager(self, num_layers=2, listener_port=None):
    """Builds a single-device, single-shard manager over `num_layers` arrays."""
    key = jax.random.key(17)
    refs = []
    arrays = []
    for layer in range(num_layers):
      base = jax.random.uniform(
          jax.random.fold_in(key, layer),
          (_NUM_BLOCKS,) + _BLOCK_SHAPE,
          dtype=_DTYPE,
      )
      refs.append(np.asarray(base))
      arrays.append(jax.device_put(base, self.devices[0]))
    jax.block_until_ready(arrays)
    manager = jax_kv.KVCacheManager(
        kv_caches=arrays,
        local_control_port=0,
        host_blocks_to_allocate=_NUM_BLOCKS,
        unsafe_skip_buffer_lock=True,
        listener_port=listener_port,
    )
    return manager, arrays, refs


class JaxPoolApiTest(_ManagerTestBase):

  def test_register_pools_round_trip(self):
    manager, _, _ = self._make_manager(num_layers=2)
    pools = (_dense_pool("fa", 0), _dense_pool("fa", 1))

    summary = manager.register_pools(pools)

    self.assertEqual(
        summary,
        {"admitted": True, "pools": 2, "storages": 2, "tags": {"fa": 2}},
    )
    self.assertEqual(manager.admission_summary(), summary)
    self.assertTrue(manager.has_explicit_pools())
    self.assertEqual(manager.num_pools(), 2)
    self.assertEqual(manager.pool_ids_with_tag("fa"), [0, 1])
    self.assertEmpty(manager.pool_ids_with_tag("state"))

  def test_pool_spec_echoes_the_admitted_descriptor(self):
    manager, _, _ = self._make_manager(num_layers=1)
    pool = _half_block_pool("fa", 0)
    manager.register_pools((pool,))

    spec = manager.pool_spec(0)

    self.assertEqual(spec["tag"], "fa")
    self.assertEqual(spec["storage_index"], 0)
    self.assertEqual(spec["base_offset_bytes"], 0)
    self.assertEqual(spec["block_stride_bytes"], _block_bytes())
    self.assertEqual(spec["num_blocks"], _NUM_BLOCKS)
    self.assertEqual(spec["dtype_tag"], "float32")
    self.assertLen(spec["regions"], 1)
    region = spec["regions"][0]
    self.assertEqual(region["name"], "payload")
    self.assertEqual(region["unit_bytes"], _block_bytes() // 2)
    self.assertEqual(region["num_units"], 1)
    self.assertEqual(region["units_per_stride"], 1)

  def test_block_refs_advance_by_the_declared_block_stride(self):
    manager, _, _ = self._make_manager(num_layers=1)
    manager.register_pools((_dense_pool("fa", 0),))

    first = manager.get_block_ref(pool_idx=0, block_id=0)
    second = manager.get_block_ref(pool_idx=0, block_id=1)

    self.assertEqual(first["tag"], "fa")
    self.assertEqual(first["block_stride_bytes"], _block_bytes())
    self.assertEqual(second["ptr"] - first["ptr"], _block_bytes())

  def test_register_pools_rejects_out_of_range_storage_index(self):
    manager, _, _ = self._make_manager(num_layers=1)

    with self.assertRaisesRegex(ValueError, "out of range"):
      manager.register_pools((_dense_pool("fa", 3),))

  def test_register_pools_rejects_an_empty_table(self):
    manager, _, _ = self._make_manager(num_layers=1)

    with self.assertRaisesRegex(ValueError, "non-empty"):
      manager.register_pools(())

  def test_pool_d2h_h2d_round_trip_leaves_the_device_untouched(self):
    """The byte oracle: pool-addressed copies must be offset-exact.

    Pool D2H/H2D mirror host and device at the *same* byte offsets, so a
    correct round trip is the identity on the device array. Any error in the
    base offset, block stride, or region extent scrambles the array instead.
    """
    manager, arrays, refs = self._make_manager(num_layers=2)
    manager.register_pools((_half_block_pool("fa", 0), _dense_pool("fa", 1)))

    for pool_idx in (0, 1):
      manager.d2h_pool_blocks(pool_idx, list(range(_NUM_BLOCKS))).Await()
    for pool_idx in (0, 1):
      manager.h2d_pool_blocks(pool_idx, list(range(_NUM_BLOCKS))).Await()
    jax.block_until_ready(arrays)

    for layer, (array, ref) in enumerate(zip(arrays, refs)):
      np.testing.assert_array_equal(
          np.asarray(array), ref, err_msg=f"layer {layer} corrupted"
      )

  def test_pool_d2h_accepts_a_block_subset(self):
    manager, arrays, refs = self._make_manager(num_layers=1)
    manager.register_pools((_dense_pool("fa", 0),))

    manager.d2h_pool_blocks(0, [1, 3]).Await()
    manager.h2d_pool_blocks(0, [1, 3]).Await()
    jax.block_until_ready(arrays)

    np.testing.assert_array_equal(np.asarray(arrays[0]), refs[0])

  def test_pool_index_out_of_range_raises(self):
    manager, _, _ = self._make_manager(num_layers=1)
    manager.register_pools((_dense_pool("fa", 0),))

    with self.assertRaises((IndexError, RuntimeError)):
      manager.pool_spec(7)


class JaxControlListenerTest(_ManagerTestBase):
  """The control plane the RaidenController drives, absent from JAX until now."""

  def test_listener_is_off_by_default(self):
    manager, _, _ = self._make_manager(num_layers=1)

    self.assertIsNone(manager.listener_port)
    self.assertFalse(manager.is_control_listener_active)
    self.assertEqual(manager.listener_address, "")

  def test_listener_binds_an_ephemeral_port(self):
    manager, _, _ = self._make_manager(num_layers=1, listener_port=0)

    port = manager.listener_port
    self.assertIsNotNone(port)
    self.assertGreater(port, 0)
    self.assertTrue(manager.is_control_listener_active)
    self.assertTrue(
        manager.listener_address.endswith(f":{port}"),
        msg=f"listener_address={manager.listener_address!r} port={port}",
    )
    # The control listener and the WorkerService gRPC server are separate
    # sockets; the latter is off here, and that must not affect the former.
    self.assertEqual(manager.get_raiden_worker_port(), 0)
    self.assertFalse(manager.is_listener_active)


if __name__ == "__main__":
  absltest.main()
