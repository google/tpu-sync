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

"""Integration tests for JAX WeightSynchronizer Python API."""

import os
import socket
import threading

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
import jax
import jax.numpy as jnp
import numpy as np

from tpu_sync.api.jax import weight_synchronizer
from tpu_sync.frameworks.jax import utils
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.weight_sync.manager import reshard_planner


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

WeightSynchronizer = weight_synchronizer.WeightSynchronizer


class WeightSynchronizerIntegrationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    try:
      self.devices = jax.devices("tpu")
    except RuntimeError:
      self.devices = jax.devices("cpu")
    self.mesh = jax.sharding.Mesh(np.array(self.devices), ("data",))
    self.sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec("data")
    )
    self.mesh_2d = jax.sharding.Mesh(
        np.array(self.devices[:4]).reshape(2, 2), ("x", "y")
    )
    self.shape = (8, 128)
    self.dtype = jnp.float32

  def test_push_synchronization(self):
    src_arrs = [
        jax.device_put(
            jnp.ones(self.shape, dtype=self.dtype) * 5.0, self.sharding
        )
    ]
    dst1_arrs = [
        jax.device_put(jnp.zeros(self.shape, dtype=self.dtype), self.sharding)
    ]
    dst2_arrs = [
        jax.device_put(jnp.zeros(self.shape, dtype=self.dtype), self.sharding)
    ]

    for arr in src_arrs:
      arr.block_until_ready()
    for arr in dst1_arrs:
      arr.block_until_ready()
    for arr in dst2_arrs:
      arr.block_until_ready()

    ws_source = WeightSynchronizer(
        jax_arrays=src_arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
        listener_port=0,
        bind_ip="127.0.0.1",
    )
    ws_dest1 = WeightSynchronizer(
        jax_arrays=dst1_arrs, local_port=0, unsafe_skip_buffer_lock=True,
        bind_ip="127.0.0.1",
    )
    ws_dest2 = WeightSynchronizer(
        jax_arrays=dst2_arrs, local_port=0, unsafe_skip_buffer_lock=True,
        bind_ip="127.0.0.1",
    )

    req = raiden_service_pb2.ControlRequest(
        command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
        peers=[
            f"127.0.0.1:{ws_dest1.local_port}",
            f"127.0.0.1:{ws_dest2.local_port}",
        ],
        start_transfer_request=raiden_service_pb2.StartTransferRequest(
            is_sender=True
        ),
    )
    payload = req.SerializeToString()

    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM, 0)
    sock.connect(("::1", ws_source.listener_port))
    sock.sendall(len(payload).to_bytes(4, "big") + payload)

    resp_len = int.from_bytes(sock.recv(4), "big")
    resp_bytes = sock.recv(resp_len)
    resp = raiden_service_pb2.ControlResponse()
    resp.ParseFromString(resp_bytes)
    assert resp.success
    sock.close()

    ws_dest1.h2d()
    ws_dest2.h2d()

    for arr in dst1_arrs:
      np.testing.assert_array_equal(np.asarray(arr), 5.0)
    for arr in dst2_arrs:
      np.testing.assert_array_equal(np.asarray(arr), 5.0)

  def test_create_cpu_instance(self):
    ws = WeightSynchronizer.test_only_create_cpu_instance(
        num_layers=2,
        num_shards=1,
        slice_byte_size=1024,
        local_port=0,
        listener_port=0,
        bind_ip="127.0.0.1",
    )
    self.assertIsNotNone(ws.local_port)
    self.assertIsNotNone(ws.listener_port)
    self.assertEqual(ws.num_layers, 2)
    self.assertEqual(ws.num_shards, 1)
    self.assertEqual(ws.slice_byte_size, 1024)

    buf = ws.get_host_buffer(layer_idx=0, shard_idx=0)
    self.assertGreaterEqual(len(buf), 1024)
    buf[:10] = 42
    self.assertEqual(buf[0], 42)

    ws.shutdown()
    self.assertIsNone(ws.local_port)

  def test_create_cpu_instance_heterogeneous(self):
    ws = WeightSynchronizer.test_only_create_cpu_instance(
        num_layers=2,
        num_shards=1,
        slice_byte_size=[512, 2048],
        local_port=0,
        listener_port=0,
        bind_ip="127.0.0.1",
    )
    self.assertIsNotNone(ws.local_port)
    self.assertIsNotNone(ws.listener_port)
    self.assertEqual(ws.num_layers, 2)
    self.assertEqual(ws.num_shards, 1)

    buf0 = ws.get_host_buffer(layer_idx=0, shard_idx=0)
    buf1 = ws.get_host_buffer(layer_idx=1, shard_idx=0)
    self.assertGreaterEqual(len(buf0), 512)
    self.assertGreaterEqual(len(buf1), 2048)
    buf0[:4] = 11
    buf1[:4] = 22
    self.assertEqual(buf0[0], 11)
    self.assertEqual(buf1[0], 22)

    ws.shutdown()
    self.assertIsNone(ws.local_port)

  def test_wait_for_transfer_completion_api_exists(self):
    arrs = [
        jax.device_put(jnp.zeros(self.shape, dtype=self.dtype), self.sharding)
    ]
    ws = WeightSynchronizer(jax_arrays=arrs, local_port=0)
    self.assertTrue(hasattr(ws, "wait_for_transfer_completion"))
    self.assertTrue(callable(ws.wait_for_transfer_completion))

  def test_bind_weights(self):
    src_arrs = [
        jax.device_put(
            jnp.ones(self.shape, dtype=self.dtype) * 5.0, self.sharding
        )
    ]
    dst_arrs = [
        jax.device_put(jnp.zeros(self.shape, dtype=self.dtype), self.sharding)
    ]

    for arr in src_arrs:
      arr.block_until_ready()
    for arr in dst_arrs:
      arr.block_until_ready()

    ws_source = WeightSynchronizer(
        jax_arrays=src_arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
        listener_port=0,
        bind_ip="127.0.0.1",
    )
    ws_dest = WeightSynchronizer(
        jax_arrays=dst_arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
        bind_ip="127.0.0.1",
    )

    # --- Sync 1 (V1: 5.0 -> 0.0) ---
    req = raiden_service_pb2.ControlRequest(
        command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
        peers=[
            f"127.0.0.1:{ws_dest.local_port}",
        ],
        start_transfer_request=raiden_service_pb2.StartTransferRequest(
            is_sender=True
        ),
    )
    payload = req.SerializeToString()

    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM, 0)
    sock.connect(("::1", ws_source.listener_port))
    sock.sendall(len(payload).to_bytes(4, "big") + payload)

    resp_len = int.from_bytes(sock.recv(4), "big")
    resp_bytes = sock.recv(resp_len)
    resp = raiden_service_pb2.ControlResponse()
    resp.ParseFromString(resp_bytes)
    assert resp.success
    sock.close()

    ws_dest.h2d()

    # Verify Sync 1
    for arr in dst_arrs:
      np.testing.assert_array_equal(np.asarray(arr), 5.0)

    # --- Bind weights to V2 ---
    new_src_arrs = [
        jax.device_put(
            jnp.ones(self.shape, dtype=self.dtype) * 10.0, self.sharding
        )
    ]
    for arr in new_src_arrs:
      arr.block_until_ready()

    ws_source.bind_weights(new_src_arrs)
    ws_source.d2h()  # Stage the V2 weights

    new_dst_arrs = [
        jax.device_put(
            jnp.ones(self.shape, dtype=self.dtype) * -1.0, self.sharding
        )
    ]
    for arr in new_dst_arrs:
      arr.block_until_ready()

    ws_dest.bind_weights(new_dst_arrs)

    # --- Sync 2 (V2: 10.0 -> -1.0) ---
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM, 0)
    sock.connect(("::1", ws_source.listener_port))
    sock.sendall(len(payload).to_bytes(4, "big") + payload)

    resp_len = int.from_bytes(sock.recv(4), "big")
    resp_bytes = sock.recv(resp_len)
    resp = raiden_service_pb2.ControlResponse()
    resp.ParseFromString(resp_bytes)
    assert resp.success
    sock.close()

    ws_dest.h2d()

    # Verify Sync 2
    for arr in new_dst_arrs:
      np.testing.assert_array_equal(np.asarray(arr), 10.0)

    # Verify original V1 arrays were NOT overwritten by Sync 2
    # (should still be 5.0)
    for arr in dst_arrs:
      np.testing.assert_array_equal(np.asarray(arr), 5.0)

  def _run_resharding_test(self, src_sharding, dst_sharding, shape):
    src_arrs = [
        jax.device_put(
            jnp.arange(np.prod(shape), dtype=self.dtype).reshape(shape),
            src_sharding,
        )
    ]
    dst_arrs = [
        jax.device_put(jnp.zeros(shape, dtype=self.dtype), dst_sharding)
    ]

    for arr in src_arrs:
      arr.block_until_ready()
    for arr in dst_arrs:
      arr.block_until_ready()

    ws_source = WeightSynchronizer(
        jax_arrays=src_arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
        listener_port=0,
        bind_ip="127.0.0.1",
    )
    ws_dest = WeightSynchronizer(
        jax_arrays=dst_arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
        bind_ip="127.0.0.1",
    )

    req = raiden_service_pb2.ControlRequest(
        command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
        peers=[
            f"127.0.0.1:{ws_dest.local_port}",
        ],
        start_transfer_request=raiden_service_pb2.StartTransferRequest(
            is_sender=True
        ),
    )
    payload = req.SerializeToString()

    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM, 0)
    sock.connect(("::1", ws_source.listener_port))
    sock.sendall(len(payload).to_bytes(4, "big") + payload)

    resp_len = int.from_bytes(sock.recv(4), "big")
    resp_bytes = sock.recv(resp_len)
    resp = raiden_service_pb2.ControlResponse()
    resp.ParseFromString(resp_bytes)
    self.assertTrue(resp.success)
    sock.close()

    ws_dest.h2d()

    # Verify data integrity
    np.testing.assert_array_equal(
        np.asarray(dst_arrs[0]), np.asarray(src_arrs[0])
    )

  def test_push_sync_aligned_to_aligned(self):
    src_sharding = jax.sharding.NamedSharding(
        self.mesh_2d, jax.sharding.PartitionSpec("x", "y")
    )
    dst_sharding = jax.sharding.NamedSharding(
        self.mesh_2d, jax.sharding.PartitionSpec("x", "y")
    )
    self._run_resharding_test(src_sharding, dst_sharding, (8, 8))

  def _run_heterogeneous_push_sync_test(self, shapes):
    src_arrs = [
        jax.device_put(
            jnp.ones(shape, dtype=self.dtype) * (i + 1.0), self.sharding
        )
        for i, shape in enumerate(shapes)
    ]
    dst_arrs = [
        jax.device_put(jnp.zeros(shape, dtype=self.dtype), self.sharding)
        for shape in shapes
    ]

    for arr in src_arrs:
      arr.block_until_ready()
    for arr in dst_arrs:
      arr.block_until_ready()

    ws_source = WeightSynchronizer(
        jax_arrays=src_arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
        listener_port=0,
        bind_ip="127.0.0.1",
    )
    ws_dest = WeightSynchronizer(
        jax_arrays=dst_arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
        bind_ip="127.0.0.1",
    )

    req = raiden_service_pb2.ControlRequest(
        command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
        peers=[f"127.0.0.1:{ws_dest.local_port}"],
        start_transfer_request=raiden_service_pb2.StartTransferRequest(
            is_sender=True
        ),
    )
    payload = req.SerializeToString()

    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM, 0)
    sock.connect(("::1", ws_source.listener_port))
    sock.sendall(len(payload).to_bytes(4, "big") + payload)

    resp_len = int.from_bytes(sock.recv(4), "big")
    resp_bytes = sock.recv(resp_len)
    resp = raiden_service_pb2.ControlResponse()
    resp.ParseFromString(resp_bytes)
    self.assertTrue(resp.success)
    sock.close()

    ws_dest.h2d()

    for i in range(len(shapes)):
      np.testing.assert_array_equal(
          np.asarray(dst_arrs[i]), np.asarray(src_arrs[i])
      )

  def test_heterogeneous_layers_small_first(self):
    self._run_heterogeneous_push_sync_test(
        [(1024,), (1024, 3072), (2048, 2048)]
    )

  def test_heterogeneous_layers_large_first(self):
    self._run_heterogeneous_push_sync_test([(1024, 3072), (1024,), (128,)])

  def test_heterogeneous_layers_local_roundtrip(self):
    shapes = [(1024,), (1024, 3072), (2048, 2048)]
    arrs = [
        jax.device_put(
            jnp.ones(shape, dtype=self.dtype) * (i + 10.0), self.sharding
        )
        for i, shape in enumerate(shapes)
    ]
    for arr in arrs:
      arr.block_until_ready()

    ws = WeightSynchronizer(
        jax_arrays=arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
    )
    ws.d2h()

    # Mutate device arrays to 0
    zero_arrs = [
        jax.device_put(jnp.zeros(shape, dtype=self.dtype), self.sharding)
        for shape in shapes
    ]
    for arr in zero_arrs:
      arr.block_until_ready()
    ws.bind_weights(zero_arrs)

    # Ingest staged weights back to device
    ws.h2d()

    for i in range(len(shapes)):
      np.testing.assert_array_equal(np.asarray(zero_arrs[i]), i + 10.0)

  def test_explicit_global_shard_indices(self):
    sharding_2d = jax.sharding.NamedSharding(
        self.mesh_2d, jax.sharding.PartitionSpec("x", "y")
    )
    arrs = [jax.device_put(jnp.zeros((8, 8), dtype=self.dtype), sharding_2d)]
    for arr in arrs:
      arr.block_until_ready()

    ws = WeightSynchronizer(
        jax_arrays=arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
        global_shard_indices=[0, 1, 4, 5],
    )
    eps = ws.get_local_endpoints()
    self.assertNotEmpty(eps)
    self.assertEqual(eps[0]["shards"], [0, 1, 4, 5])

  def test_explicit_global_shard_indices_8_shards(self):
    arrs = [
        jax.device_put(jnp.zeros(self.shape, dtype=self.dtype), self.sharding)
    ]
    for arr in arrs:
      arr.block_until_ready()

    expected_indices = [0, 1, 4, 5, 8, 9, 12, 13]
    ws = WeightSynchronizer(
        jax_arrays=arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
        global_shard_indices=expected_indices,
    )
    eps = ws.get_local_endpoints()
    self.assertNotEmpty(eps)
    self.assertEqual(eps[0]["shards"], expected_indices)

  def test_automatic_global_shard_indices_derivation(self):
    sharding_2d = jax.sharding.NamedSharding(
        self.mesh_2d, jax.sharding.PartitionSpec("x", "y")
    )
    arrs = [jax.device_put(jnp.zeros((8, 8), dtype=self.dtype), sharding_2d)]
    for arr in arrs:
      arr.block_until_ready()

    ws = WeightSynchronizer(
        jax_arrays=arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
    )
    eps = ws.get_local_endpoints()
    self.assertNotEmpty(eps)
    # 2x2 mesh with 4 local devices should automatically derive flat indices [0, 1, 2, 3]
    self.assertEqual(eps[0]["shards"], [0, 1, 2, 3])

  def test_non_contiguous_subgrid_transfer(self):
    sharding_2d = jax.sharding.NamedSharding(
        self.mesh_2d, jax.sharding.PartitionSpec("x", "y")
    )
    src_arrs = [
        jax.device_put(
            jnp.arange(64, dtype=self.dtype).reshape(8, 8), sharding_2d
        )
    ]
    dst_arrs = [
        jax.device_put(jnp.zeros((8, 8), dtype=self.dtype), sharding_2d)
    ]
    for arr in src_arrs:
      arr.block_until_ready()
    for arr in dst_arrs:
      arr.block_until_ready()

    # Configure non-contiguous global shard indices [0, 1, 4, 5] on both sides
    ws_source = WeightSynchronizer(
        jax_arrays=src_arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
        listener_port=0,
        bind_ip="127.0.0.1",
        global_shard_indices=[0, 1, 4, 5],
    )
    ws_dest = WeightSynchronizer(
        jax_arrays=dst_arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
        bind_ip="127.0.0.1",
        global_shard_indices=[0, 1, 4, 5],
    )

    req = raiden_service_pb2.ControlRequest(
        command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
        peers=[f"127.0.0.1:{ws_dest.local_port}"],
        start_transfer_request=raiden_service_pb2.StartTransferRequest(
            is_sender=True
        ),
    )
    payload = req.SerializeToString()

    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM, 0)
    sock.connect(("::1", ws_source.listener_port))
    sock.sendall(len(payload).to_bytes(4, "big") + payload)

    resp_len = int.from_bytes(sock.recv(4), "big")
    resp_bytes = sock.recv(resp_len)
    resp = raiden_service_pb2.ControlResponse()
    resp.ParseFromString(resp_bytes)
    self.assertTrue(resp.success)
    sock.close()

    ws_dest.h2d()

    np.testing.assert_array_equal(
        np.asarray(dst_arrs[0]), np.asarray(src_arrs[0])
    )

  def test_metrics_accumulation_bandwidth_and_reset(self):
    ws_source = WeightSynchronizer.test_only_create_cpu_instance(
        num_layers=1,
        num_shards=1,
        slice_byte_size=1024,
        local_port=0,
        listener_port=0,
        bind_ip="127.0.0.1",
    )
    ws_dest = WeightSynchronizer.test_only_create_cpu_instance(
        num_layers=1,
        num_shards=1,
        slice_byte_size=1024,
        local_port=0,
        listener_port=0,
        bind_ip="127.0.0.1",
    )
    self.addCleanup(ws_source.shutdown)
    self.addCleanup(ws_dest.shutdown)

    m0 = ws_source.get_metrics()
    self.assertEqual(m0["total_h2h_bytes"], 0)
    self.assertEqual(m0["total_h2h_time_ms"], 0.0)
    self.assertEqual(m0["total_h2h_bandwidth_gbps"], 0.0)

    def _send_ctrl_req(port: int, req: raiden_service_pb2.ControlRequest):
      payload = req.SerializeToString()
      sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM, 0)
      sock.connect(("::1", port))
      sock.sendall(len(payload).to_bytes(4, "big") + payload)
      resp_len = int.from_bytes(sock.recv(4), "big")
      resp_bytes = sock.recv(resp_len)
      resp = raiden_service_pb2.ControlResponse()
      resp.ParseFromString(resp_bytes)
      self.assertTrue(resp.success, resp.message)
      sock.close()

    for step in range(2):
      uuid = 70001 + step
      dst_req = raiden_service_pb2.ControlRequest(
          command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
          start_transfer_request=raiden_service_pb2.StartTransferRequest(
              is_sender=False,
              uuid=uuid,
              expected_block_count=1,
          ),
      )
      _send_ctrl_req(ws_dest.listener_port, dst_req)

      src_transfer_req = raiden_service_pb2.StartTransferRequest(
          is_sender=True,
          uuid=uuid,
          skip_d2h=True,
      )
      sched = src_transfer_req.shard_push_schedules[0]
      entry = sched.entries.add()
      entry.dst_peer = f"127.0.0.1:{ws_dest.local_port}"
      entry.dst_shard_idx = 0
      entry.src_offset_bytes = 0
      entry.dst_offset_bytes = 0
      entry.size_bytes = 1024
      entry.count = 1
      entry.layer_idx = 0

      src_req = raiden_service_pb2.ControlRequest(
          command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
          start_transfer_request=src_transfer_req,
      )
      _send_ctrl_req(ws_source.listener_port, src_req)
      ws_dest.wait_for_transfer_completion(uuid)

    m2 = ws_source.get_metrics()
    self.assertEqual(m2["total_h2h_bytes"], 2048)
    self.assertGreater(m2["total_h2h_time_ms"], 0.0)
    self.assertGreater(m2["total_h2h_bandwidth_gbps"], 0.0)
    self.assertEqual(m2["push_resharded_call_count"], 2)

    ws_source.reset_metrics()
    m_reset = ws_source.get_metrics()
    self.assertEqual(m_reset["total_h2h_bytes"], 0)
    self.assertEqual(m_reset["total_h2h_time_ms"], 0.0)
    self.assertEqual(m_reset["total_h2h_bandwidth_gbps"], 0.0)
    self.assertEqual(m_reset["push_resharded_call_count"], 0)

  def test_bounded_ring_buffer_pool_jax_e2e(self):
    num_layers = 6
    ring_size = 2
    src_arrs = [
        jax.device_put(
            jnp.ones(self.shape, dtype=self.dtype) * float(i + 1),
            self.sharding,
        )
        for i in range(num_layers)
    ]
    dst_arrs = [
        jax.device_put(jnp.zeros(self.shape, dtype=self.dtype), self.sharding)
        for _ in range(num_layers)
    ]
    for arr in src_arrs + dst_arrs:
      arr.block_until_ready()

    ws_unbounded = WeightSynchronizer(
        jax_arrays=src_arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
        bind_ip="127.0.0.1",
    )
    ws_source = WeightSynchronizer(
        jax_arrays=src_arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
        listener_port=0,
        bind_ip="127.0.0.1",
        ring_buffer_size=ring_size,
    )
    ws_dest = WeightSynchronizer(
        jax_arrays=dst_arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
        listener_port=0,
        bind_ip="127.0.0.1",
        ring_buffer_size=ring_size,
    )
    self.addCleanup(ws_unbounded.shutdown)
    self.addCleanup(ws_source.shutdown)
    self.addCleanup(ws_dest.shutdown)

    self.assertEqual(ws_source.ring_buffer_size, ring_size)
    self.assertEqual(ws_dest.ring_buffer_size, ring_size)
    self.assertEqual(
        ws_source.allocated_host_dram_bytes * (num_layers // ring_size),
        ws_unbounded.allocated_host_dram_bytes,
    )

    def _send_ctrl_req(port: int, req: raiden_service_pb2.ControlRequest):
      payload = req.SerializeToString()
      sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM, 0)
      sock.connect(("::1", port))
      sock.sendall(len(payload).to_bytes(4, "big") + payload)
      resp_len = int.from_bytes(sock.recv(4), "big")
      resp_bytes = sock.recv(resp_len)
      resp = raiden_service_pb2.ControlResponse()
      resp.ParseFromString(resp_bytes)
      self.assertTrue(resp.success, resp.message)
      sock.close()

    uuid = 75001
    num_shards = ws_source.num_shards
    slice_bytes = ws_source.slice_byte_size
    dst_start_req = raiden_service_pb2.StartTransferRequest(
        is_sender=False,
        uuid=uuid,
        expected_block_count=num_layers * num_shards,
    )
    for l in range(num_layers):
      dst_start_req.expected_layer_chunk_counts[l] = num_shards
    _send_ctrl_req(
        ws_dest.listener_port,
        raiden_service_pb2.ControlRequest(
            command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
            start_transfer_request=dst_start_req,
        ),
    )

    src_start_req = raiden_service_pb2.StartTransferRequest(
        is_sender=True,
        uuid=uuid,
        skip_d2h=False,
    )
    for s in range(num_shards):
      sched = src_start_req.shard_push_schedules[s]
      for l in range(num_layers):
        entry = sched.entries.add()
        entry.dst_peer = f"127.0.0.1:{ws_dest.local_port}"
        entry.dst_shard_idx = s
        entry.src_offset_bytes = 0
        entry.dst_offset_bytes = 0
        entry.size_bytes = slice_bytes
        entry.count = 1
        entry.layer_idx = l
    _send_ctrl_req(
        ws_source.listener_port,
        raiden_service_pb2.ControlRequest(
            command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
            start_transfer_request=src_start_req,
        ),
    )
    ws_dest.wait_for_transfer_completion(uuid)
    ws_dest.h2d()

    for i in range(num_layers):
      np.testing.assert_array_equal(
          np.asarray(dst_arrs[i]), np.asarray(src_arrs[i])
      )

  def test_dp_subsharding_and_post_h2d_ici_all_gather_e2e(self):
    # Source mesh: (4, 2) along ("fsdp", "tp")
    # Destination mesh: (2, 2, 2) along ("dp", "fsdp", "tp") -> DP=2 replicated!
    src_mesh = jax.sharding.Mesh(
        np.array(self.devices[:8]).reshape(4, 2), ("fsdp", "tp")
    )
    dst_mesh = jax.sharding.Mesh(
        np.array(self.devices[:8]).reshape(2, 2, 2), ("dp", "fsdp", "tp")
    )
    src_sharding = jax.sharding.NamedSharding(
        src_mesh, jax.sharding.PartitionSpec("fsdp", "tp")
    )
    dst_sharding = jax.sharding.NamedSharding(
        dst_mesh, jax.sharding.PartitionSpec("fsdp", "tp")
    )
    shape = (64, 32)
    src_arrs = [
        jax.device_put(
            jnp.arange(np.prod(shape), dtype=self.dtype).reshape(shape),
            src_sharding,
        )
    ]
    dst_arrs = [
        jax.device_put(jnp.zeros(shape, dtype=self.dtype), dst_sharding)
    ]
    for arr in src_arrs + dst_arrs:
      arr.block_until_ready()

    ws_source = WeightSynchronizer(
        jax_arrays=src_arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
        listener_port=0,
        bind_ip="127.0.0.1",
    )
    ws_dest = WeightSynchronizer(
        jax_arrays=dst_arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
        listener_port=0,
        bind_ip="127.0.0.1",
    )
    self.addCleanup(ws_source.shutdown)
    self.addCleanup(ws_dest.shutdown)

    # Disable CPU tiling on CPU devices since XLA CPU arrays are untiled.
    ws_source.test_only_set_skip_tiling(True)
    ws_dest.test_only_set_skip_tiling(True)

    src_id = reshard_planner.RaidenId("trainer", "0", "weights", 0)
    dst_id = reshard_planner.RaidenId("sampler", "0", "weights", 0)
    var_src = raiden_service_pb2.VariableMetadataProto(
        name="w0",
        shape=list(shape),
        mesh_shape=[4, 2],
        layout=[1, 0],
        item_size=4,
        layer_idx=0,
        sharding_spec=["fsdp", "tp"],
    )
    var_dst = raiden_service_pb2.VariableMetadataProto(
        name="w0",
        shape=list(shape),
        mesh_shape=[2, 2],
        layout=[1, 0],
        item_size=4,
        layer_idx=0,
        sharding_spec=["fsdp", "tp"],
    )
    src_shards = [f"127.0.0.1:{ws_source.local_port}"] * 8
    dst_shards = [f"127.0.0.1:{ws_dest.local_port}"] * 8
    dst_meta_proto = raiden_service_pb2.RegisterWorkUnitRequest(
        unit=raiden_service_pb2.RaidenIdProto(
            job_name=dst_id.job_name,
            job_replica_id=str(dst_id.job_replica_id),
            data_name=dst_id.data_name,
            data_replica_idx=dst_id.data_replica_idx,
        ),
        shards=dst_shards,
        control_plane_rpc_address=f"127.0.0.1:{ws_dest.listener_port}",
        variables=[var_dst],
        mesh_shape=[2, 2, 2],
        mesh_axes=["dp", "fsdp", "tp"],
    )
    common_kwargs = dict(
        src_units=[src_id],
        dst_units=[dst_id],
        dst_metadata=[dst_meta_proto],
        entities={
            src_id: reshard_planner.JobEntity(unit=src_id, shards=src_shards)
        },
        registered_variables={src_id: [var_src]},
        registered_global_shapes={},
        registered_mesh_shapes={src_id: [4, 2], dst_id: [2, 2, 2]},
        registered_mesh_axes={
            src_id: ["fsdp", "tp"],
            dst_id: ["dp", "fsdp", "tp"],
        },
        registered_host_subgrids={},
        registered_layouts={},
        registered_itemsizes={},
        registered_shards={src_id: src_shards, dst_id: dst_shards},
        computed_phys_meshes={},
        worker_endpoints={
            src_id: f"127.0.0.1:{ws_source.listener_port}",
            dst_id: f"127.0.0.1:{ws_dest.listener_port}",
        },
        broadcast_k=64,
        lock=threading.Lock(),
        skip_tiling={0: True},
    )
    sched_full = (
        reshard_planner.ReshardPlanner.compute_transfer_schedule_from_metadata(
            **common_kwargs,
            enable_dp_subsharding=False,
        )
    )
    sched_sub = (
        reshard_planner.ReshardPlanner.compute_transfer_schedule_from_metadata(
            **common_kwargs,
            enable_dp_subsharding=True,
        )
    )

    full_bytes = 0
    for s_entries in sched_full.direct_schedules[src_id].values():
      for e in s_entries:
        full_bytes += e[4] * e[9]
    sub_bytes = 0
    for s_entries in sched_sub.direct_schedules[src_id].values():
      for e in s_entries:
        sub_bytes += e[4] * e[9]
    tensor_bytes = int(np.prod(shape)) * 4
    self.assertEqual(full_bytes, 2 * tensor_bytes)
    self.assertEqual(sub_bytes, tensor_bytes)

    def _send_ctrl_req(port: int, req: raiden_service_pb2.ControlRequest):
      payload = req.SerializeToString()
      sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM, 0)
      sock.connect(("::1", port))
      sock.sendall(len(payload).to_bytes(4, "big") + payload)
      resp_len = int.from_bytes(sock.recv(4), "big")
      resp_bytes = sock.recv(resp_len)
      resp = raiden_service_pb2.ControlResponse()
      resp.ParseFromString(resp_bytes)
      self.assertTrue(resp.success, resp.message)
      sock.close()

    uuid = 76001
    dst_start_req = raiden_service_pb2.StartTransferRequest(
        is_sender=False,
        uuid=uuid,
        expected_block_count=sched_sub.dst_unit_counts[dst_id],
    )
    dst_start_req.skip_tiling[0] = True
    for l_idx, cnt in sched_sub.dst_unit_layer_counts.get(dst_id, {}).items():
      dst_start_req.expected_layer_chunk_counts[l_idx] = cnt
    _send_ctrl_req(
        ws_dest.listener_port,
        raiden_service_pb2.ControlRequest(
            command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
            start_transfer_request=dst_start_req,
        ),
    )

    src_start_req = raiden_service_pb2.StartTransferRequest(
        is_sender=True,
        uuid=uuid,
        skip_d2h=False,
    )
    src_start_req.skip_tiling[0] = True
    for src_s, entries in sched_sub.direct_schedules[src_id].items():
      sched_proto = src_start_req.shard_push_schedules[src_s]
      for e in entries:
        entry = sched_proto.entries.add()
        entry.dst_peer = e[0]
        entry.dst_shard_idx = e[1]
        entry.dst_offset_bytes = e[2]
        entry.src_offset_bytes = e[3]
        entry.size_bytes = e[4]
        entry.src_block_id = e[5]
        entry.dst_block_id = e[6]
        entry.src_stride_bytes = e[7]
        entry.dst_stride_bytes = e[8]
        entry.count = e[9]
        entry.layer_idx = e[10]
    _send_ctrl_req(
        ws_source.listener_port,
        raiden_service_pb2.ControlRequest(
            command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
            start_transfer_request=src_start_req,
        ),
    )

    ws_dest.wait_for_transfer_completion(uuid)
    gathered_arrs = ws_dest.h2d(ici_all_gather=True)
    self.assertIsNotNone(gathered_arrs)
    self.assertLen(gathered_arrs, 1)
    gathered_arrs[0].block_until_ready()

    # Verify the full tensor and every DP=2 replica shard matches src_arrs[0]
    np.testing.assert_array_equal(
        np.asarray(gathered_arrs[0]), np.asarray(src_arrs[0])
    )
    for shard in gathered_arrs[0].addressable_shards:
      expected_slice = np.asarray(src_arrs[0])[shard.index]
      np.testing.assert_array_equal(np.asarray(shard.data), expected_slice)


class ShardSortingUtilTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    try:
      self.devices = jax.devices("tpu")
    except RuntimeError:
      self.devices = jax.devices("cpu")
    self.mesh_2d = jax.sharding.Mesh(
        np.array(self.devices[:4]).reshape(2, 2), ("x", "y")
    )

  def test_aligned_sharding_permutation(self):
    sharding = jax.sharding.NamedSharding(
        self.mesh_2d, jax.sharding.PartitionSpec("x", "y")
    )
    arr = jax.device_put(jnp.zeros((8, 8)), sharding)
    perm = utils.get_shard_sorting_permutation(arr)
    self.assertEqual(perm, [])

  def test_transposed_sharding_permutation(self):
    sharding = jax.sharding.NamedSharding(
        self.mesh_2d, jax.sharding.PartitionSpec("y", "x")
    )
    arr = jax.device_put(jnp.zeros((8, 8)), sharding)
    perm = utils.get_shard_sorting_permutation(arr)
    # No permutation is needed because JAX device order matches the
    # controller's physical mapping order for this transposed sharding.
    self.assertEqual(perm, [])

  def test_replicated_sharding_permutation(self):
    sharding = jax.sharding.NamedSharding(
        self.mesh_2d, jax.sharding.PartitionSpec("x")
    )
    arr = jax.device_put(jnp.zeros((8, 8)), sharding)
    perm = utils.get_shard_sorting_permutation(arr)
    self.assertEqual(perm, [])


if __name__ == "__main__":
  absltest.main()
