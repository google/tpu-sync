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

"""E2E unit tests for repurposed JAX KVCacheManager."""

import os
import time

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from tpu_sync.api.jax import kv_cache_manager

KVCacheManager = kv_cache_manager.KVCacheManager
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


class KVCacheManagerJaxTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    device_type = "tpu"
    try:
      self.devices = jax.devices(device_type)
    except RuntimeError as exc:
      raise AssertionError(f"No {device_type} devices found") from exc

    if not self.devices:
      raise AssertionError(f"No {device_type} devices found")

    self.num_devices = len(self.devices)
    self.num_layers = 2
    self.block_size = 1
    self.skip_lock = True

  def create_mesh(self, axis_shapes, axis_names, devices=None):
    try:
      num_required_devices = np.prod(axis_shapes)
      if devices is None:
        devices = self.devices
      devices = np.array(devices)
      if len(devices) < num_required_devices:
        raise AssertionError(
            f"Need {num_required_devices} devices, got {len(devices)}"
        )
      device_array = devices[:num_required_devices].reshape(axis_shapes)
      return jax.sharding.Mesh(device_array, axis_names)
    except RuntimeError:
      self.skipTest("Cannot create mesh.")
      return None

  def setup_shardings(self):
    axis_shapes = (1, self.num_devices)
    axis_names = ("data", "model")
    mesh = self.create_mesh(axis_shapes, axis_names)
    spec = jax.sharding.PartitionSpec(None, None, "model", None, None)
    tpu_sharding = jax.sharding.NamedSharding(mesh, spec)
    return tpu_sharding

  def _generate_random_cache(self, key, shape, dtype, sharding):
    """Generates random cache data on device and returns (device_array, host_numpy_array)."""
    host_floats = jax.random.uniform(key, shape, dtype=dtype)
    dev_arr = jax.device_put(host_floats, sharding)
    return dev_arr, np.asarray(host_floats)

  def _verify_cache_equal(self, dev_arr, expected_numpy):
    """Verifies that the device array values match the expected numpy array."""
    actual_numpy = np.asarray(dev_arr)
    np.testing.assert_array_equal(actual_numpy, expected_numpy)

  def _to_loopback_endpoints(self, endpoints):
    return endpoints

  def test_initialization(self):
    tpu_sharding = self.setup_shardings()
    shape = (4, 128, 8, 8, 128)
    kv_caches = [jax.device_put(jnp.zeros(shape), tpu_sharding)]

    engine = KVCacheManager(
        kv_caches=kv_caches,
        local_control_port=0,
        max_blocks=4,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
    )
    self.assertIsNotNone(engine)

  @parameterized.parameters(jnp.float32, jnp.bfloat16)
  def test_e2e_transfer_polling(self, dtype):
    tpu_sharding = self.setup_shardings()
    num_blocks = 2
    shape = (num_blocks, 128, 8, 8, 128)

    src_refs = []
    src_caches = []
    key = jax.random.key(123)
    for i in range(self.num_layers):
      sub_key = jax.random.fold_in(key, i)
      dev_arr, host_ref = self._generate_random_cache(
          sub_key, shape, dtype, tpu_sharding
      )
      src_caches.append(dev_arr)
      src_refs.append(host_ref)

    dst_caches = []
    for _ in range(self.num_layers):
      dst_caches.append(
          jax.device_put(jnp.zeros(shape, dtype=dtype), tpu_sharding)
      )

    jax.block_until_ready(src_caches)
    jax.block_until_ready(dst_caches)

    producer = KVCacheManager(
        kv_caches=src_caches,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        node_id=0,
    )

    consumer = KVCacheManager(
        kv_caches=dst_caches,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        node_id=1,
    )

    endpoints = self._to_loopback_endpoints(producer.get_local_endpoints())
    self.assertGreater(len(endpoints), 0)

    req_id = "test_req_poll_jax"
    uuid = 12345
    producer.register_read(req_id, uuid, [0, 1])

    consumer.start_read(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=endpoints,
        remote_block_ids=[0, 1],
        local_block_ids=[0, 1],
    )

    # Poll until consumer is done receiving
    done = False
    for _ in range(50):
      _, done_recving, failed_recving = consumer.poll_stats()
      if req_id in failed_recving:
        self.fail("Transfer failed")
      if req_id in done_recving:
        done = True
        break
      time.sleep(0.1)

    self.assertTrue(done, "Consumer did not finish transfer in time")

    # Check that consumer correctly loaded the values
    for idx, t in enumerate(dst_caches):
      self._verify_cache_equal(t, src_refs[idx])

    # Poll producer until it's done sending
    done_prod = False
    for _ in range(50):
      done_sending, _, _ = producer.poll_stats()
      if req_id in done_sending:
        done_prod = True
        break
      time.sleep(0.1)

    self.assertTrue(done_prod, "Producer did not finish sending in time")

  @parameterized.parameters(jnp.float32, jnp.bfloat16)
  def test_non_contiguous_blocks(self, dtype):
    tpu_sharding = self.setup_shardings()
    num_blocks = 3
    shape = (num_blocks, 128, 8, 8, 128)

    src_refs = []
    src_caches = []
    key = jax.random.key(123)
    for i in range(self.num_layers):
      sub_key = jax.random.fold_in(key, i)
      dev_arr, host_ref = self._generate_random_cache(
          sub_key, shape, dtype, tpu_sharding
      )
      src_caches.append(dev_arr)
      src_refs.append(host_ref)

    dst_caches = []
    for _ in range(self.num_layers):
      dst_caches.append(
          jax.device_put(jnp.zeros(shape, dtype=dtype), tpu_sharding)
      )

    jax.block_until_ready(src_caches)
    jax.block_until_ready(dst_caches)

    producer = KVCacheManager(
        kv_caches=src_caches,
        local_control_port=0,
        max_blocks=3,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        node_id=0,
    )

    consumer = KVCacheManager(
        kv_caches=dst_caches,
        local_control_port=0,
        max_blocks=3,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        node_id=1,
    )

    endpoints = self._to_loopback_endpoints(producer.get_local_endpoints())
    self.assertGreater(len(endpoints), 0)

    req_id = "test_req_non_contig"
    uuid = 54321
    producer.register_read(req_id, uuid, [0, 2])

    consumer.start_read(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=endpoints,
        remote_block_ids=[0, 2],
        local_block_ids=[0, 1],
    )

    done = False
    for _ in range(50):
      _, done_recving, failed_recving = consumer.poll_stats()
      if req_id in failed_recving:
        self.fail("Transfer failed")
      if req_id in done_recving:
        done = True
        break
      time.sleep(0.1)

    self.assertTrue(done, "Consumer did not finish transfer in time")

    for idx, t in enumerate(dst_caches):
      # local block 0 <- remote block 0
      self._verify_cache_equal(t[0], src_refs[idx][0])
      # local block 1 <- remote block 2
      self._verify_cache_equal(t[1], src_refs[idx][2])
      # local block 2 was not copied, should remain 0
      np.testing.assert_allclose(np.asarray(t)[2], 0.0, atol=1e-5)

    done_prod = False
    for _ in range(50):
      done_sending, _, _ = producer.poll_stats()
      if req_id in done_sending:
        done_prod = True
        break
      time.sleep(0.1)

    self.assertTrue(done_prod, "Producer did not finish sending in time")

  @parameterized.parameters(jnp.float32, jnp.bfloat16)
  def test_host_reordering(self, dtype):
    tpu_sharding = self.setup_shardings()
    num_blocks = 2
    shape = (num_blocks, 128, 8, 8, 128)

    src_refs = []
    src_caches = []
    key = jax.random.key(123)
    for i in range(self.num_layers):
      sub_key = jax.random.fold_in(key, i)
      dev_arr, host_ref = self._generate_random_cache(
          sub_key, shape, dtype, tpu_sharding
      )
      src_caches.append(dev_arr)
      src_refs.append(host_ref)

    dst_caches = []
    for _ in range(self.num_layers):
      dst_caches.append(
          jax.device_put(jnp.zeros(shape, dtype=dtype), tpu_sharding)
      )

    jax.block_until_ready(src_caches)
    jax.block_until_ready(dst_caches)

    producer = KVCacheManager(
        kv_caches=src_caches,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        node_id=0,
    )

    consumer = KVCacheManager(
        kv_caches=dst_caches,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        node_id=1,
    )

    endpoints = self._to_loopback_endpoints(producer.get_local_endpoints())
    self.assertGreater(len(endpoints), 0)

    req_id = "test_req_reorder"
    uuid = 98765
    producer.register_read(req_id, uuid, [0, 1])

    consumer.start_read(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=endpoints,
        remote_block_ids=[1, 0],
        local_block_ids=[0, 1],
    )

    done = False
    for _ in range(50):
      _, done_recving, failed_recving = consumer.poll_stats()
      if req_id in failed_recving:
        self.fail("Transfer failed")
      if req_id in done_recving:
        done = True
        break
      time.sleep(0.1)

    self.assertTrue(done, "Consumer did not finish transfer in time")

    for idx, t in enumerate(dst_caches):
      # local block 0 <- remote block 1
      self._verify_cache_equal(t[0], src_refs[idx][1])
      # local block 1 <- remote block 0
      self._verify_cache_equal(t[1], src_refs[idx][0])

    done_prod = False
    for _ in range(50):
      done_sending, _, _ = producer.poll_stats()
      if req_id in done_sending:
        done_prod = True
        break
      time.sleep(0.1)

    self.assertTrue(done_prod, "Producer did not finish sending in time")

  @parameterized.parameters(jnp.float32, jnp.bfloat16)
  def test_large_complex_non_contiguous_and_reorder(self, dtype):
    tpu_sharding = self.setup_shardings()
    num_blocks = 16
    shape = (num_blocks, 128, 8, 8, 128)

    src_refs = []
    src_caches = []
    key = jax.random.key(123)
    for i in range(self.num_layers):
      sub_key = jax.random.fold_in(key, i)
      dev_arr, host_ref = self._generate_random_cache(
          sub_key, shape, dtype, tpu_sharding
      )
      src_caches.append(dev_arr)
      src_refs.append(host_ref)

    dst_caches = []
    for _ in range(self.num_layers):
      dst_caches.append(
          jax.device_put(jnp.zeros(shape, dtype=dtype), tpu_sharding)
      )

    jax.block_until_ready(src_caches)
    jax.block_until_ready(dst_caches)

    producer = KVCacheManager(
        kv_caches=src_caches,
        local_control_port=0,
        max_blocks=16,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        node_id=0,
    )

    consumer = KVCacheManager(
        kv_caches=dst_caches,
        local_control_port=0,
        max_blocks=16,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        node_id=1,
    )

    endpoints = self._to_loopback_endpoints(producer.get_local_endpoints())
    self.assertGreater(len(endpoints), 0)

    req_id = "test_req_large_complex"
    uuid = 13579

    remote_blocks = [0, 2, 3, 5, 6, 7, 9, 11, 12, 14]
    requested_remote = list(reversed(remote_blocks))
    local_blocks = list(range(len(remote_blocks)))

    producer.register_read(req_id, uuid, remote_blocks)

    consumer.start_read(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=endpoints,
        remote_block_ids=requested_remote,
        local_block_ids=local_blocks,
    )

    done = False
    for _ in range(100):
      _, done_recving, failed_recving = consumer.poll_stats()
      if req_id in failed_recving:
        self.fail("Transfer failed")
      if req_id in done_recving:
        done = True
        break
      time.sleep(0.1)

    self.assertTrue(done, "Consumer did not finish transfer in time")

    for idx, t in enumerate(dst_caches):
      for local_idx, local_block in enumerate(local_blocks):
        remote_block = requested_remote[local_idx]
        self._verify_cache_equal(t[local_block], src_refs[idx][remote_block])

      for local_block in range(len(local_blocks), num_blocks):
        np.testing.assert_allclose(np.asarray(t)[local_block], 0.0, atol=1e-5)

    done_prod = False
    for _ in range(50):
      done_sending, _, _ = producer.poll_stats()
      if req_id in done_sending:
        done_prod = True
        break
      time.sleep(0.1)

    self.assertTrue(done_prod, "Producer did not finish sending in time")

  @parameterized.parameters(jnp.float32, jnp.bfloat16)
  def test_parallel_pull(self, dtype):
    tpu_sharding = self.setup_shardings()
    num_blocks = 2
    shape = (num_blocks, 128, 8, 8, 128)

    src_refs = []
    src_caches = []
    key = jax.random.key(999)
    for i in range(self.num_layers):
      sub_key = jax.random.fold_in(key, i)
      dev_arr, host_ref = self._generate_random_cache(
          sub_key, shape, dtype, tpu_sharding
      )
      src_caches.append(dev_arr)
      src_refs.append(host_ref)

    dst_caches = []
    for _ in range(self.num_layers):
      dst_caches.append(
          jax.device_put(jnp.zeros(shape, dtype=dtype), tpu_sharding)
      )

    jax.block_until_ready(src_caches)
    jax.block_until_ready(dst_caches)

    producer = KVCacheManager(
        kv_caches=src_caches,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        node_id=0,
    )

    consumer = KVCacheManager(
        kv_caches=dst_caches,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        node_id=1,
    )

    endpoints = self._to_loopback_endpoints(producer.get_local_endpoints())
    self.assertGreater(len(endpoints), 0)

    req_id = "test_req_parallel"
    uuid = 77777
    producer.register_read(req_id, uuid, [0, 1])

    consumer.start_read(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=endpoints,
        remote_block_ids=[0, 1],
        local_block_ids=[0, 1],
        parallelism=2,
    )

    done = False
    for _ in range(50):
      _, done_recving, failed_recving = consumer.poll_stats()
      if req_id in failed_recving:
        self.fail("Transfer failed")
      if req_id in done_recving:
        done = True
        break
      time.sleep(0.1)

    self.assertTrue(done, "Consumer did not finish transfer in time")

    for idx, t in enumerate(dst_caches):
      self._verify_cache_equal(t, src_refs[idx])

    done_prod = False
    for _ in range(50):
      done_sending, _, _ = producer.poll_stats()
      if req_id in done_sending:
        done_prod = True
        break
      time.sleep(0.1)

    self.assertTrue(done_prod, "Producer did not finish sending in time")

  def test_hbm_staging_fragmented_d2h_and_h2d_with_donated_caches(self):
    # Verifies the HBM-staged bulk transfer path (`enable_hbm_staging=True`)
    # end to end across:
    #   1. Non-power-of-two (5-block) scattered `d2h_auto_allocate` -> `h2d`.
    #   2. JAX buffer donation (`donate_argnums`) invalidating initial
    #      `kv_caches` before D2H and again between consecutive H2D scatters.
    #   3. Safe no-op handling of empty (`[]`) and zero-size (`copy_sizes=[0]`)
    #      transfers without allocating host blocks or clobbering caches.
    tpu_sharding = self.setup_shardings()
    shape = (8, 128, 8, 8, 128)
    key = jax.random.key(2026)

    src_refs = []
    caches = []
    for i in range(self.num_layers):
      sub_key = jax.random.fold_in(key, i)
      dev_arr, host_ref = self._generate_random_cache(
          sub_key, shape, jnp.bfloat16, tpu_sharding
      )
      caches.append(dev_arr)
      src_refs.append(host_ref)
    jax.block_until_ready(caches)

    engine = KVCacheManager(
        kv_caches=caches,
        local_control_port=0,
        host_blocks_to_allocate=8,
        unsafe_skip_buffer_lock=self.skip_lock,
        enable_hbm_staging=True,
    )

    # Simulate an inference forward step donating and invalidating the initial
    # `caches` buffers that `engine` saw at construction time.
    donate_step = jax.jit(lambda xs: [x for x in xs], donate_argnums=(0,))
    step_caches = donate_step(caches)
    jax.block_until_ready(step_caches)

    # Offload 5 scattered HBM blocks (non-power-of-two count: padded to an
    # 8-block gather bucket in HBM, streamed via a 5-block bulk D2H DMA).
    src_blocks = [7, 2, 5, 0, 3]
    allocated_chunks, save_fut = engine.d2h_auto_allocate(
        src_blocks, kv_caches=step_caches
    )
    save_fut.Await()

    # Empty / zero-size `d2h_auto_allocate`, `d2h`, and `h2d` calls must return
    # immediately ready futures without allocating host slots or overwriting
    # previously saved host blocks or live device caches.
    empty_chunks, empty_fut = engine.d2h_auto_allocate([], kv_caches=step_caches)
    self.assertEqual(empty_chunks, [])
    self.assertTrue(empty_fut.IsReady())
    empty_fut.Await()
    zero_chunks, zero_fut = engine.d2h_auto_allocate(
        [0], [0], kv_caches=step_caches
    )
    self.assertEqual(zero_chunks, [])
    zero_fut.Await()
    engine.d2h([0], [0], [0], kv_caches=step_caches).Await()

    # Recall the 5 host chunks into 5 different scattered HBM destination blocks
    # `[6, 1, 4, 0, 5]` (decomposed into 4 + 1 power-of-two scatter buckets),
    # then donate `engine.kv_caches` in a second step and repeat H2D to confirm
    # no stale/donated buffer reference is retained across steps.
    dst_caches = [
        jax.device_put(jnp.zeros(shape, dtype=jnp.bfloat16), tpu_sharding)
        for _ in range(self.num_layers)
    ]
    jax.block_until_ready(dst_caches)
    engine.h2d([0], [0], [0], kv_caches=dst_caches).Await()
    dst_blocks = [6, 1, 4, 0, 5]
    engine.h2d(allocated_chunks, dst_blocks, kv_caches=dst_caches).Await()
    step2_caches = donate_step(engine.kv_caches)
    jax.block_until_ready(step2_caches)
    engine.h2d(allocated_chunks, dst_blocks, kv_caches=step2_caches).Await()

    # Targeted blocks match the offloaded source blocks; untouched HBM blocks
    # remain zeroed.
    untouched = sorted(set(range(8)) - set(dst_blocks))
    for layer_idx in range(self.num_layers):
      actual = np.asarray(engine.kv_caches[layer_idx])
      for dst_blk, orig_blk in zip(dst_blocks, src_blocks):
        np.testing.assert_array_equal(
            actual[dst_blk], src_refs[layer_idx][orig_blk]
        )
      for blk in untouched:
        np.testing.assert_array_equal(actual[blk], 0)

    engine.unlock_blocks(allocated_chunks)


if __name__ == "__main__":
  absltest.main()
