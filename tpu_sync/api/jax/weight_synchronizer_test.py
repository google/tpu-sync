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

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
import jax
import jax.numpy as jnp
import numpy as np

from tpu_sync.api.jax import weight_synchronizer
from tpu_sync.frameworks.jax import utils
from tpu_sync.rpc import raiden_service_pb2


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
