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

"""Tests for Raiden Controller high-level transfer API under rpc/."""

import asyncio
import math
import os
import socket
import time
from unittest import mock
from absl.testing import absltest
from tpu_sync.rpc import raiden_controller
from tpu_sync.rpc import raiden_service_pb2


class DummyWorkerRpcClient(raiden_controller.WorkerRpcClient):

  async def start_transfer(
      self, target_id, transfer_plan, address=None
  ) -> None:
    pass


class RecordingWorkerRpcClient(raiden_controller.WorkerRpcClient):

  def __init__(self, event_log=None, label=""):
    super().__init__()
    self.calls = []
    self.event_log = event_log
    self.label = label

  async def start_transfer(
      self, target_id, transfer_plan, address=None
  ) -> None:
    self.calls.append((target_id, transfer_plan))
    if self.event_log is not None:
      self.event_log.append((self.label, target_id))


class RaidenControllerTest(absltest.TestCase):

  def test_register_work_unit_accepts_duplicate_endpoints_for_shared_port(
      self,
  ):
    # JAX/Pathways: one process serves several local devices behind a single
    # transfer port, so a unit's shards legitimately register identical
    # endpoints. The list length is the shard count; the addresses coincide.
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10000, worker_rpc_client=client
    )
    unit = raiden_controller.RaidenId(
        job_name="pathways",
        job_replica_id="host0",
        data_name="kv_cache",
        data_replica_idx=0,
    )
    shared_endpoint = "10.0.0.1:8000"
    controller.register_work_unit(unit, [shared_endpoint] * 4)
    self.assertEqual(controller._resolve_shards(unit), [shared_endpoint] * 4)

  def test_registration_metadata_round_trip_and_replacement(self):
    bind_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bind_sock.bind(("127.0.0.1", 0))
    port = bind_sock.getsockname()[1]
    bind_sock.close()

    controller = raiden_controller.RaidenController(port=port)
    server = raiden_controller.RaidenControllerServer(controller)
    server.start()
    facade = raiden_controller.RaidenControllerClientFacade(f"127.0.0.1:{port}")
    unit = raiden_controller.RaidenId("prefill", "engine-rank0", "kv.fa", 0)
    pool = raiden_service_pb2.PoolSpecProto(
        tag="fa",
        storage_index=0,
        block_stride_bytes=4096 * 1024,
        num_blocks=4096,
        dtype_tag="fp8",
    )
    try:
      facade.register_work_unit(
          unit,
          ["127.0.0.1:8100"],
          control_plane_rpc_address="127.0.0.1:9100",
          pool_manifest=[pool],
          layout_fingerprint="layout-v1",
          page_tokens=4096,
          transfer_parallelism=8,
          transfer_rank=0,
      )
      metadata = facade.get_metadata()
      self.assertLen(metadata, 1)
      self.assertEqual(
          raiden_controller._raiden_id_from_proto(metadata[0].unit), unit
      )
      self.assertLen(metadata[0].pools, 1)
      self.assertEqual(metadata[0].layout_fingerprint, "layout-v1")
      self.assertEqual(metadata[0].page_tokens, 4096)
      self.assertEqual(metadata[0].transfer_parallelism, 8)
      self.assertEqual(metadata[0].transfer_rank, 0)

      # Registration is replacement, not a patch. Re-registration
      # removes every optional field and stale control endpoint.
      facade.register_work_unit(unit, ["127.0.0.1:8200"])
      replaced = facade.get_metadata()[0]
      self.assertEmpty(replaced.pools)
      self.assertEmpty(replaced.layout_fingerprint)
      self.assertEqual(replaced.page_tokens, 0)
      self.assertEmpty(replaced.control_plane_rpc_address)
    finally:
      server.stop()
      server._thread.join(timeout=2)

  def test_variables_registration_metadata_round_trip(self):
    bind_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bind_sock.bind(("127.0.0.1", 0))
    port = bind_sock.getsockname()[1]
    bind_sock.close()

    controller = raiden_controller.RaidenController(port=port)
    server = raiden_controller.RaidenControllerServer(controller)
    server.start()
    facade = raiden_controller.RaidenControllerClientFacade(f"127.0.0.1:{port}")
    unit = raiden_controller.RaidenId("prefill", "engine-rank0", "kv.fa", 0)

    v1 = raiden_service_pb2.VariableMetadataProto(
        name="weights_0",
        shape=[128, 1024],
        mesh_shape=[2, 2],
        layout=[0, 1],
        item_size=4,
        layer_idx=0,
    )
    v2 = raiden_service_pb2.VariableMetadataProto(
        name="weights_1",
        shape=[512],
        mesh_shape=[2, 2],
        layout=[0],
        item_size=2,
        layer_idx=1,
    )

    try:
      facade.register_work_unit(
          unit,
          ["127.0.0.1:8100"],
          control_plane_rpc_address="127.0.0.1:9100",
          variables=[v1, v2],
      )
      metadata = facade.get_metadata()
      self.assertLen(metadata, 1)
      self.assertEqual(
          raiden_controller._raiden_id_from_proto(metadata[0].unit), unit
      )
      self.assertLen(metadata[0].variables, 2)
      self.assertEqual(metadata[0].variables[0].name, "weights_0")
      self.assertEqual(list(metadata[0].variables[0].shape), [128, 1024])
      self.assertEqual(list(metadata[0].variables[0].mesh_shape), [2, 2])
      self.assertEqual(list(metadata[0].variables[0].layout), [0, 1])
      self.assertEqual(metadata[0].variables[0].item_size, 4)
      self.assertEqual(metadata[0].variables[0].layer_idx, 0)

      self.assertEqual(metadata[0].variables[1].name, "weights_1")
      self.assertEqual(list(metadata[0].variables[1].shape), [512])
      self.assertEqual(list(metadata[0].variables[1].mesh_shape), [2, 2])
      self.assertEqual(list(metadata[0].variables[1].layout), [0])
      self.assertEqual(metadata[0].variables[1].item_size, 2)
      self.assertEqual(metadata[0].variables[1].layer_idx, 1)

    finally:
      server.stop()
      server._thread.join(timeout=2)

  def test_global_shard_indices_registration_round_trip(self):
    bind_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bind_sock.bind(("127.0.0.1", 0))
    port = bind_sock.getsockname()[1]
    bind_sock.close()

    controller = raiden_controller.RaidenController(port=port)
    server = raiden_controller.RaidenControllerServer(controller)
    server.start()
    facade = raiden_controller.RaidenControllerClientFacade(f"127.0.0.1:{port}")
    unit = raiden_controller.RaidenId("trainer", "0", "weights", 0)

    # Variables specify explicit global_shard_indices.
    # Note: host_subgrid, mesh_axes, and top-level mesh_shape are no longer
    # necessary.
    v1 = raiden_service_pb2.VariableMetadataProto(
        name="weights_0",
        shape=[128, 1024],
        mesh_shape=[2, 2],
        layout=[0, 1],
        item_size=4,
        layer_idx=0,
        global_shard_indices=[0, 2],
    )
    v2 = raiden_controller._VariableMetadata(
        name="weights_1",
        shape=[512],
        mesh_shape=[2, 2],
        layout=[0],
        item_size=2,
        layer_idx=1,
        global_shard_indices=[1, 3],
    )

    try:
      facade.register_work_unit(
          unit,
          ["127.0.0.1:8100", "127.0.0.1:8101"],
          control_plane_rpc_address="127.0.0.1:9100",
          variables=[v1, v2],
      )
      metadata = facade.get_metadata()
      self.assertLen(metadata, 1)
      self.assertEqual(
          raiden_controller._raiden_id_from_proto(metadata[0].unit), unit
      )
      self.assertLen(metadata[0].variables, 2)
      self.assertEqual(metadata[0].variables[0].name, "weights_0")
      self.assertEqual(
          list(metadata[0].variables[0].global_shard_indices), [0, 2]
      )
      self.assertEqual(metadata[0].variables[1].name, "weights_1")
      self.assertEqual(
          list(metadata[0].variables[1].global_shard_indices), [1, 3]
      )
    finally:
      server.stop()
      server._thread.join(timeout=2)

  def test_dynamic_balancing_and_overlap_planner(self):
    dummy_client = DummyWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10000, worker_rpc_client=dummy_client
    )

    src_unit_0 = raiden_controller.RaidenId(
        job_name="sampler",
        job_replica_id="0",
        data_name="kv_cache",
    )
    src_unit_1 = raiden_controller.RaidenId(
        job_name="sampler",
        job_replica_id="1",
        data_name="kv_cache",
    )
    target_unit = raiden_controller.RaidenId(
        job_name="inference_server",
        job_replica_id="215",
        data_name="kv_cache",
    )

    controller.register_work_unit(
        src_unit_0, ["10.0.0.1:8000", "10.0.0.2:8000"]
    )
    controller.register_work_unit(
        src_unit_1, ["10.0.0.3:8000", "10.0.0.4:8000"]
    )
    controller.register_work_unit(
        target_unit, ["10.0.0.5:8000", "10.0.0.6:8000"]
    )

    # First transfer routes to src_unit_0
    future_1 = controller.start_transfer(
        src_units=[src_unit_0, src_unit_1],
        dst_units=[target_unit],
    )
    asyncio.run(future_1.wait())

    self.assertTrue(future_1.done())
    self.assertEqual(future_1.session_id, 0)

    plan_1 = controller.get_plan("req_0")
    self.assertEqual(plan_1.src_units[0].job_replica_id, "0")

    # Verify generalized NDSlice fully qualified overlap push schedule dict
    self.assertIn(src_unit_0, plan_1.plan)
    unit_0_plan = plan_1.plan[src_unit_0]
    self.assertLen(unit_0_plan, 2)
    self.assertEqual(unit_0_plan[0], [(target_unit, 0, [[(0, 2)]])])
    self.assertEqual(unit_0_plan[1], [(target_unit, 1, [[(0, 2)]])])

    # Second transfer routes dynamically to least-loaded src_unit_1
    future_2 = controller.start_transfer(
        src_units=[src_unit_0, src_unit_1],
        dst_units=[target_unit],
    )
    self.assertEqual(future_2.session_id, 1)

    plan_2 = controller.get_plan("req_1")
    self.assertEqual(plan_2.src_units[0].job_replica_id, "1")
    asyncio.run(future_2.wait())

  def test_fan_out_multiple_targets(self):
    controller = raiden_controller.RaidenController(
        port=10001, worker_rpc_client=DummyWorkerRpcClient()
    )

    src = raiden_controller.RaidenId(
        job_name="trainer",
        job_replica_id="0",
        data_name="layer0.weights",
    )
    target_0 = raiden_controller.RaidenId(
        job_name="inference_server",
        job_replica_id="10",
        data_name="layer0.weights",
    )
    target_1 = raiden_controller.RaidenId(
        job_name="inference_server",
        job_replica_id="11",
        data_name="layer0.weights",
    )

    controller.register_work_unit(src, ["10.0.0.1:8000"])
    controller.register_work_unit(target_0, ["10.0.0.2:8000"])
    controller.register_work_unit(target_1, ["10.0.0.3:8000"])

    future_multi = controller.start_transfer(
        src_units=[src],
        dst_units=[target_0, target_1],
    )
    self.assertEqual(future_multi.session_id, 0)

    plan_multi = controller.get_plan("req_0")
    self.assertLen(plan_multi.dst_units, 2)
    asyncio.run(future_multi.wait())

  def test_rpc_client_push_coordination(self):
    recorded_actions = []

    class MockWorkerClient(raiden_controller.WorkerRpcClient):

      async def start_transfer(
          self, target_id, transfer_plan, address=None
      ) -> None:
        recorded_actions.append(("start", [target_id]))

    mock_client = MockWorkerClient()
    controller = raiden_controller.RaidenController(
        port=10002, worker_rpc_client=mock_client
    )

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    dst = raiden_controller.RaidenId(
        job_name="sampler", job_replica_id="0", data_name="weights"
    )
    controller.register_work_unit(src, ["10.0.0.1:8000"])
    controller.register_work_unit(dst, ["10.0.0.2:8000"])

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[dst],
    )

    asyncio.run(future.wait())

    self.assertEqual(
        recorded_actions,
        [
            ("start", [dst]),
            ("start", [src]),
        ],
    )

  def test_enforce_metadata_completeness(self):
    controller = raiden_controller.RaidenController(port=10003)
    unit = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )

    # 1. Failure if some shape/layout fields are missing
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda e: "all of them must be provided" in str(e)
    ):
      controller.register_work_unit(
          unit,
          ["10.0.0.1:8000"],
          mesh_shape=[1, 1, 4, 1, 1],
          # layout and global_shape are missing
      )

    # 2. Failure if itemsize is missing when metadata is provided
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda e: "itemsize must be provided" in str(e)
    ):
      controller.register_work_unit(
          unit,
          ["10.0.0.1:8000"],
          mesh_shape=[1, 1, 4, 1, 1],
          layout=[4, 3, 2, 1, 0],
          global_shape=[128, 16, 8, 2, 128],
          # itemsize is missing
      )

    # 3. Success if all are provided
    controller.register_work_unit(
        unit,
        ["10.0.0.1:8000"],
        mesh_shape=[1, 1, 4, 1, 1],
        layout=[4, 3, 2, 1, 0],
        global_shape=[128, 16, 8, 2, 128],
        itemsize=4,
    )

  def test_multi_shard_worker_resharding(self):
    dummy_client = DummyWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10004, worker_rpc_client=dummy_client
    )
    src = raiden_controller.RaidenId("trainer", "0", "weights")
    dst = raiden_controller.RaidenId("sampler", "0", "weights")

    controller.register_work_unit(
        src,
        ["10.0.0.1:8000", "10.0.0.1:8001", "10.0.0.1:8002", "10.0.0.1:8003"],
        mesh_shape=(2, 2),
        layout=(1, 0),
        global_shape=(128, 1024),
        itemsize=4,
    )
    controller.register_work_unit(
        dst,
        ["10.0.0.2:8000", "10.0.0.2:8001", "10.0.0.2:8002", "10.0.0.2:8003"],
        mesh_shape=(1, 4),
        layout=(1, 0),
        global_shape=(128, 1024),
        itemsize=4,
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[dst],
        use_block_chunks=True,
    )
    asyncio.run(future.wait())
    plan = controller.get_plan("req_0")
    self.assertIn(src, plan.shard_push_schedules)
    schedules = plan.shard_push_schedules[src]
    self.assertSetEqual(set(schedules.keys()), {0, 1, 2, 3})

  def test_resharding_with_layout_permutation_different_meshes(self):
    dummy_client = DummyWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10004, worker_rpc_client=dummy_client
    )

    # Mesh [2, 3, 4], layout [0, 2, 1], num_hosts=3.
    # Host axis is dim 1 (size 3).
    # We register 3 replicas for src and dst.
    src_units = []
    dst_units = []
    for r in range(3):
      src_id = raiden_controller.RaidenId("trainer", str(r), "weights")
      dst_id = raiden_controller.RaidenId("sampler", str(r), "weights")
      src_units.append(src_id)
      dst_units.append(dst_id)

      # 8 shards per replica
      src_shards = [f"10.0.0.1:{8000 + r*8 + i}" for i in range(8)]
      dst_shards = [f"10.0.0.2:{8000 + r*8 + i}" for i in range(8)]

      controller.register_work_unit(
          src_id,
          src_shards,
          mesh_shape=(2, 3, 4),
          layout=(0, 2, 1),
          global_shape=(4, 3, 4),
          itemsize=4,
      )
      controller.register_work_unit(
          dst_id,
          dst_shards,
          mesh_shape=(4, 3, 2),
          layout=(0, 2, 1),
          global_shape=(4, 3, 4),
          itemsize=4,
      )

    future = controller.start_transfer(
        src_units=src_units,
        dst_units=dst_units,
        use_block_chunks=True,
    )
    asyncio.run(future.wait())

    # Assert on schedules for replica 0
    plan = controller.get_plan("req_0")
    self.assertIn(src_units[0], plan.shard_push_schedules)
    schedules = plan.shard_push_schedules[src_units[0]]

    expected_matches = {
        0: [0, 2],
        1: [0, 2],
        2: [1, 3],
        3: [1, 3],
        4: [4, 6],
        5: [4, 6],
        6: [5, 7],
        7: [5, 7],
    }

    for src_idx, expected_dst_indices in expected_matches.items():
      self.assertIn(src_idx, schedules)
      entries = schedules[src_idx]
      self.assertEqual(len(entries), 2)

      actual_dst_indices = sorted([entry[1] for entry in entries])
      self.assertEqual(actual_dst_indices, expected_dst_indices)

      for entry in entries:
        dst_peer = entry[0]
        dst_shard_idx = entry[1]
        self.assertEqual(dst_peer, f"10.0.0.2:{8000 + dst_shard_idx}")

  def test_greedy_tree_broadcast(self):
    recorded_calls = []

    class MockBroadcastWorkerClient(raiden_controller.WorkerRpcClient):

      def __init__(self):
        super().__init__()
        self.endpoints = {}

      def get_worker_endpoints(self):
        return self.endpoints

      async def start_transfer(
          self, target_id, transfer_plan, address=None
      ) -> None:
        is_sender = transfer_plan.is_sender
        recorded_calls.append((target_id, is_sender))

    mock_client = MockBroadcastWorkerClient()
    controller = raiden_controller.RaidenController(
        port=10004, worker_rpc_client=mock_client
    )
    controller.broadcast_k = 2

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target_0 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )
    target_1 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="1", data_name="weights"
    )
    target_2 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="2", data_name="weights"
    )

    controller.register_work_unit(
        src,
        ["10.0.0.1:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.1:9000",
    )
    controller.register_work_unit(
        target_0,
        ["10.0.0.2:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
    )
    controller.register_work_unit(
        target_1,
        ["10.0.0.3:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.3:9000",
    )
    controller.register_work_unit(
        target_2,
        ["10.0.0.4:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.4:9000",
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[target_0, target_1, target_2],
        use_block_chunks=True,
    )
    asyncio.run(future.wait())

    self.assertTrue(future.done())
    self.assertGreater(len(recorded_calls), 0)

  def test_greedy_tree_broadcast_exception_propagation(self):
    class MockFailureWorkerClient(raiden_controller.WorkerRpcClient):

      async def start_transfer(
          self, target_id, transfer_plan, address=None
      ) -> None:
        if target_id.job_replica_id == "1":
          raise RuntimeError("Simulated start_transfer failure")

    mock_client = MockFailureWorkerClient()
    controller = raiden_controller.RaidenController(
        port=10005, worker_rpc_client=mock_client
    )
    controller.broadcast_k = 2

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target_0 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )
    target_1 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="1", data_name="weights"
    )
    target_2 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="2", data_name="weights"
    )

    controller.register_work_unit(
        src,
        ["10.0.0.1:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.1:9000",
    )
    controller.register_work_unit(
        target_0,
        ["10.0.0.2:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
    )
    controller.register_work_unit(
        target_1,
        ["10.0.0.3:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.3:9000",
    )
    controller.register_work_unit(
        target_2,
        ["10.0.0.4:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.4:9000",
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[target_0, target_1, target_2],
        use_block_chunks=True,
    )
    with self.assertRaisesRegex(
        RuntimeError, "Simulated start_transfer failure"
    ):
      asyncio.run(future.wait())

  def test_register_transfer_schedule_skip_d2h_propagation(self):
    facade = raiden_controller.RaidenControllerClientFacade("127.0.0.1:0")
    calls = []

    def mock_send(req):
      calls.append(req)
      return True

    facade._send_raiden_protobuf_rpc = mock_send

    src_unit = raiden_controller.RaidenId("prefill", "engine-rank0", "kv.fa", 0)
    dst_unit = raiden_controller.RaidenId("decode", "engine-rank0", "kv.fa", 0)

    # Test with skip_d2h=True
    facade.register_transfer_schedule(
        src_units=[src_unit],
        dst_units=[dst_unit],
        req_id="req1",
        skip_d2h=True,
    )
    self.assertEqual(len(calls), 1)
    req = calls[0]
    self.assertEqual(
        req.command,
        raiden_service_pb2.ControlRequest.COMMAND_REGISTER_TRANSFER_SCHEDULE,
    )
    self.assertTrue(req.start_transfer_request.skip_d2h)

    # Test with skip_d2h=False (default)
    calls.clear()
    facade.register_transfer_schedule(
        src_units=[src_unit],
        dst_units=[dst_unit],
        req_id="req2",
    )
    self.assertEqual(len(calls), 1)
    req = calls[0]
    self.assertFalse(req.start_transfer_request.skip_d2h)

  def test_server_skip_d2h_propagation(self):
    controller = raiden_controller.RaidenController(port=0)
    server = raiden_controller.RaidenControllerServer(controller)
    server.start()

    facade = raiden_controller.RaidenControllerClientFacade(
        f"127.0.0.1:{server.port}",
        name_resolver=controller.worker_rpc_client.name_resolver,
    )

    calls = []
    original_start_transfer = controller.start_transfer

    def mock_start_transfer(*args, **kwargs):
      calls.append(kwargs)
      return raiden_controller.RaidenFuture(0, None)

    controller.start_transfer = mock_start_transfer

    src_unit = raiden_controller.RaidenId("prefill", "engine-rank0", "kv.fa", 0)
    dst_unit = raiden_controller.RaidenId("decode", "engine-rank0", "kv.fa", 0)

    try:
      # Test with skip_d2h=True
      facade.register_transfer_schedule(
          src_units=[src_unit],
          dst_units=[dst_unit],
          req_id="req1",
          skip_d2h=True,
      )
      self.assertEqual(len(calls), 1)
      self.assertTrue(calls[0].get("skip_d2h"))

      # Test with skip_d2h=False (default)
      calls.clear()
      facade.register_transfer_schedule(
          src_units=[src_unit],
          dst_units=[dst_unit],
          req_id="req2",
      )
      self.assertEqual(len(calls), 1)
      self.assertFalse(calls[0].get("skip_d2h"))

    finally:
      controller.start_transfer = original_start_transfer
      server.stop()

  def test_multi_variable_resharding_planning(self):
    """Verifies resharding planning for multiple variables using absolute offsets."""
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10000, worker_rpc_client=client
    )

    src_unit = raiden_controller.RaidenId(
        "prefill", "engine-rank0", "weights", 0
    )
    dst_unit = raiden_controller.RaidenId(
        "decode", "engine-rank0", "weights", 0
    )

    # 1. Register source unit with 2 variables
    controller.register_work_unit(
        src_unit,
        shards=["127.0.0.1:8000"] * 4,
        control_plane_rpc_address="127.0.0.1:9000",
        variables=[
            raiden_service_pb2.VariableMetadataProto(
                name="weights_0",
                shape=[128, 1024],
                mesh_shape=[1, 4],
                layout=[0, 1],
                item_size=4,
                layer_idx=0,
            ),
            raiden_service_pb2.VariableMetadataProto(
                name="weights_1",
                shape=[4, 512],
                mesh_shape=[1, 4],
                layout=[0, 1],
                item_size=2,
                layer_idx=1,
            ),
        ],
    )

    # 2. Register destination unit with matching variables but different sharding
    controller.register_work_unit(
        dst_unit,
        shards=["127.0.0.1:8001"] * 4,
        control_plane_rpc_address="127.0.0.1:9001",
        variables=[
            raiden_service_pb2.VariableMetadataProto(
                name="weights_0",
                shape=[128, 1024],
                mesh_shape=[2, 2],
                layout=[0, 1],
                item_size=4,
                layer_idx=0,
            ),
            raiden_service_pb2.VariableMetadataProto(
                name="weights_1",
                shape=[4, 512],
                mesh_shape=[2, 2],
                layout=[0, 1],
                item_size=2,
                layer_idx=1,
            ),
        ],
    )

    # 3. Trigger transfer
    future = controller.start_transfer(
        src_units=[src_unit],
        dst_units=[dst_unit],
        req_id="multi-var-req",
        use_block_chunks=True,
    )
    asyncio.run(future.wait())

    # Verify calls
    self.assertNotEmpty(client.calls)

    # 4. Verify that the generated schedules in the plan contain entries for both variables.
    plan = controller.get_plan("multi-var-req")
    self.assertIsNotNone(plan)

    unit_entries = plan.shard_push_schedules[src_unit][0]
    self.assertNotEmpty(unit_entries)

    # 5. Assert that the entries have the correct layer_idx (0 and 1 respectively)
    layer_indices = {entry[10] for entry in unit_entries}
    self.assertEqual(layer_indices, {0, 1})

    # 6. Assert that absolute offsets are used (since both register variables, they are not legacy).
    has_large_offset = False
    for shard_idx, entries in plan.shard_push_schedules[src_unit].items():
      for entry in entries:
        dst_offset = entry[2]
        src_offset = entry[3]
        layer_idx = entry[10]
        if layer_idx == 0:
          # Block sizes (inner dimensions): src = 512, dst = 256.
          # Absolute offset (e.g. 65536) strictly exceeds these.
          if dst_offset >= 2048 or src_offset >= 1024:
            has_large_offset = True
        elif layer_idx == 1:
          # Block sizes (inner dimensions): src = 8, dst = 4.
          # Absolute offset (e.g. 512) strictly exceeds these.
          if dst_offset >= 512 or src_offset >= 256:
            has_large_offset = True

    self.assertTrue(
        has_large_offset, "Expected absolute offsets to exceed block sizes"
    )

  def test_upfront_d2h_optimization_legacy_path(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10004, worker_rpc_client=client
    )
    controller.broadcast_k = 2

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target_0 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )
    target_1 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="1", data_name="weights"
    )
    target_2 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="2", data_name="weights"
    )

    controller.register_work_unit(
        src,
        ["10.0.0.1:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.1:9000",
    )
    controller.register_work_unit(
        target_0,
        ["10.0.0.2:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
    )
    controller.register_work_unit(
        target_1,
        ["10.0.0.3:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.3:9000",
    )
    controller.register_work_unit(
        target_2,
        ["10.0.0.4:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.4:9000",
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[target_0, target_1, target_2],
        use_block_chunks=True,
    )
    asyncio.run(future.wait())

    self.assertTrue(len(client.calls) > 0)
    src_calls = [call for call in client.calls if call[0] == src]
    self.assertNotEmpty(src_calls)
    self.assertFalse(src_calls[0][1].skip_d2h)

  def test_skip_d2h_true_no_upfront_copy_legacy_path(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10004, worker_rpc_client=client
    )
    controller.broadcast_k = 2

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target_0 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )
    target_1 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="1", data_name="weights"
    )
    target_2 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="2", data_name="weights"
    )

    controller.register_work_unit(
        src,
        ["10.0.0.1:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.1:9000",
    )
    controller.register_work_unit(
        target_0,
        ["10.0.0.2:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
    )
    controller.register_work_unit(
        target_1,
        ["10.0.0.3:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.3:9000",
    )
    controller.register_work_unit(
        target_2,
        ["10.0.0.4:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.4:9000",
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[target_0, target_1, target_2],
        use_block_chunks=True,
        skip_d2h=True,
    )
    asyncio.run(future.wait())

    self.assertTrue(len(client.calls) > 0)
    for target_id, plan in client.calls:
      self.assertTrue(plan.skip_d2h)
      self.assertNotEqual(plan.shard_push_schedules, {target_id: {0: []}})

  def test_auto_calculate_expected_block_count(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10004, worker_rpc_client=client
    )

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )

    controller.register_work_unit(
        src,
        ["10.0.0.1:8000", "10.0.0.1:8001"],
        mesh_shape=[2],
        layout=[0],
        global_shape=[128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.1:9000",
    )
    controller.register_work_unit(
        target,
        ["10.0.0.2:8000", "10.0.0.2:8001"],
        mesh_shape=[2],
        layout=[0],
        global_shape=[128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[target],
        use_block_chunks=True,
    )
    asyncio.run(future.wait())

    self.assertTrue(len(client.calls) > 0)
    dst_calls = [call for call in client.calls if call[0] == target]
    self.assertLen(dst_calls, 1)
    target_id, plan = dst_calls[0]
    self.assertEqual(plan.expected_block_count, 2)

  def test_auto_calculate_expected_block_count_strided(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10004, worker_rpc_client=client
    )

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )
    # Src: 1x2 mesh, layout [-1, 0], global [8, 8] -> shard [8, 4]
    controller.register_work_unit(
        src,
        ["10.0.0.1:8000"],
        mesh_shape=[1, 2],
        layout=[-1, 0],
        global_shape=[8, 8],
        itemsize=4,
        control_plane_rpc_address="10.0.0.1:9000",
    )
    # Dst: 1x4 mesh, layout [-1, 0], global [8, 8] -> shard [8, 2]
    controller.register_work_unit(
        target,
        ["10.0.0.2:8000"],
        mesh_shape=[1, 4],
        layout=[-1, 0],
        global_shape=[8, 8],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[target],
        use_block_chunks=True,
    )
    asyncio.run(future.wait())

    self.assertTrue(len(client.calls) > 0)
    dst_calls = [call for call in client.calls if call[0] == target]
    self.assertLen(dst_calls, 1)
    target_id, plan = dst_calls[0]
    self.assertEqual(plan.expected_block_count, 1)

  def test_auto_calculate_expected_block_count_strided_skip_tiling(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10004, worker_rpc_client=client
    )

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )

    controller.register_work_unit(
        src,
        ["10.0.0.1:8000"],
        mesh_shape=[1, 2],
        layout=[-1, 0],
        global_shape=[8, 8],
        itemsize=4,
        control_plane_rpc_address="10.0.0.1:9000",
    )
    controller.register_work_unit(
        target,
        ["10.0.0.2:8000"],
        mesh_shape=[1, 4],
        layout=[-1, 0],
        global_shape=[8, 8],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[target],
        use_block_chunks=True,
        skip_tiling={0: True},
    )
    asyncio.run(future.wait())

    self.assertTrue(len(client.calls) > 0)
    dst_calls = [call for call in client.calls if call[0] == target]
    self.assertLen(dst_calls, 1)
    target_id, plan = dst_calls[0]
    self.assertEqual(plan.expected_block_count, 1)

  def test_auto_calculate_expected_block_count_strided_contiguous_src(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10004, worker_rpc_client=client
    )

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )
    # Src: 1x4 mesh, layout [-1, 0], global [8, 8] -> shard [8, 2]
    # Dst: 1x2 mesh, layout [-1, 0], global [8, 8] -> shard [8, 4]
    # Each row slice has 2 elements (8 bytes).
    # Since src shard has 2 elements per row, src_stride == size == 8 bytes.
    # Dst shard has 4 elements per row, so dst_stride == 16 bytes != size.
    # Under Path A (Approach 1), contiguous source with strided dst counts as 1 task per block!
    controller.register_work_unit(
        src,
        ["10.0.0.1:8000"],
        mesh_shape=[1, 4],
        layout=[-1, 0],
        global_shape=[8, 8],
        itemsize=4,
        control_plane_rpc_address="10.0.0.1:9000",
    )
    controller.register_work_unit(
        target,
        ["10.0.0.2:8000"],
        mesh_shape=[1, 2],
        layout=[-1, 0],
        global_shape=[8, 8],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[target],
        use_block_chunks=True,
    )
    asyncio.run(future.wait())

    self.assertTrue(len(client.calls) > 0)
    dst_calls = [call for call in client.calls if call[0] == target]
    self.assertLen(dst_calls, 1)
    target_id, plan = dst_calls[0]
    # With 1 single strided task emitted for contiguous src (instead of 8 unrolled tasks), expected is 1.
    self.assertEqual(plan.expected_block_count, 1)

  def test_auto_calculate_expected_block_count_mixed(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10004, worker_rpc_client=client
    )

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )

    variables_src = [
        raiden_service_pb2.VariableMetadataProto(
            name="weights_0",
            shape=[8, 8],
            mesh_shape=[1, 2],
            layout=[-1, 0],
            item_size=4,
            layer_idx=0,
        ),
        raiden_service_pb2.VariableMetadataProto(
            name="weights_1",
            shape=[8, 8],
            mesh_shape=[1, 2],
            layout=[-1, 0],
            item_size=4,
            layer_idx=1,
        ),
    ]

    variables_dst = [
        raiden_service_pb2.VariableMetadataProto(
            name="weights_0",
            shape=[8, 8],
            mesh_shape=[1, 4],
            layout=[-1, 0],
            item_size=4,
            layer_idx=0,
        ),
        raiden_service_pb2.VariableMetadataProto(
            name="weights_1",
            shape=[8, 8],
            mesh_shape=[1, 4],
            layout=[-1, 0],
            item_size=4,
            layer_idx=1,
        ),
    ]

    controller.register_work_unit(
        src,
        ["10.0.0.1:8000"],
        mesh_shape=[1, 2],
        layout=[-1, 0],
        global_shape=[8, 8],
        itemsize=4,
        control_plane_rpc_address="10.0.0.1:9000",
        variables=variables_src,
    )
    controller.register_work_unit(
        target,
        ["10.0.0.2:8000"],
        mesh_shape=[1, 4],
        layout=[-1, 0],
        global_shape=[8, 8],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
        variables=variables_dst,
    )

    # Mixed skip tiling: 1 strided task for layer 0, 1 strided task for layer 1.
    # Total expected = 1 + 1 = 2.
    future = controller.start_transfer(
        src_units=[src],
        dst_units=[target],
        use_block_chunks=True,
        skip_tiling={0: True, 1: False},
    )
    asyncio.run(future.wait())

    self.assertTrue(len(client.calls) > 0)
    dst_calls = [call for call in client.calls if call[0] == target]
    self.assertLen(dst_calls, 1)
    target_id, plan = dst_calls[0]
    self.assertEqual(plan.expected_block_count, 2)

  def test_plan_caching_hit_and_reuse(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10005, worker_rpc_client=client, enable_plan_cache=True
    )

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )

    controller.register_work_unit(
        src,
        ["10.0.0.1:8000", "10.0.0.1:8001"],
        mesh_shape=[2],
        layout=[0],
        global_shape=[128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.1:9000",
    )
    controller.register_work_unit(
        target,
        ["10.0.0.2:8000", "10.0.0.2:8001"],
        mesh_shape=[2],
        layout=[0],
        global_shape=[128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
    )

    self.assertEqual(controller.get_plan_cache_size(), 0)

    # First transfer: cache miss, computes and caches schedule
    future_1 = controller.start_transfer(
        src_units=[src],
        dst_units=[target],
        use_block_chunks=True,
        req_id="iter_0",
        uuid=1001,
    )
    asyncio.run(future_1.wait())
    self.assertEqual(controller.get_plan_cache_size(), 1)
    plan_1 = controller.get_plan("iter_0")
    self.assertEqual(plan_1.uuid, 1001)
    self.assertEqual(plan_1.expected_block_count, 2)

    # Second transfer: cache hit, reuses pre-computed schedule
    future_2 = controller.start_transfer(
        src_units=[src],
        dst_units=[target],
        use_block_chunks=True,
        req_id="iter_1",
        uuid=1002,
    )
    asyncio.run(future_2.wait())
    self.assertEqual(controller.get_plan_cache_size(), 1)
    plan_2 = controller.get_plan("iter_1")
    self.assertEqual(plan_2.uuid, 1002)
    self.assertEqual(plan_2.expected_block_count, 2)
    self.assertEqual(plan_1.shard_push_schedules, plan_2.shard_push_schedules)

  def test_plan_caching_invalidation_on_reregistration(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10006, worker_rpc_client=client
    )

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )

    controller.register_work_unit(
        src,
        ["10.0.0.1:8000"],
        mesh_shape=[1],
        layout=[0],
        global_shape=[64],
        itemsize=4,
        control_plane_rpc_address="10.0.0.1:9000",
    )
    controller.register_work_unit(
        target,
        ["10.0.0.2:8000"],
        mesh_shape=[1],
        layout=[0],
        global_shape=[64],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[target],
        use_block_chunks=True,
    )
    asyncio.run(future.wait())
    self.assertEqual(controller.get_plan_cache_size(), 1)

    # Re-registering target with identical metadata (heartbeat keepalive)
    # preserves plan cache.
    controller.register_work_unit(
        target,
        ["10.0.0.2:8000"],
        mesh_shape=[1],
        layout=[0],
        global_shape=[64],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
    )
    self.assertEqual(controller.get_plan_cache_size(), 1)

    # Re-registering target with changed metadata (e.g. shard changed)
    # invalidates plan cache.
    controller.register_work_unit(
        target,
        ["10.0.0.2:8001"],
        mesh_shape=[1],
        layout=[0],
        global_shape=[64],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
    )
    self.assertEqual(controller.get_plan_cache_size(), 0)

    # Next transfer repopulates cache
    future_2 = controller.start_transfer(
        src_units=[src],
        dst_units=[target],
        use_block_chunks=True,
    )
    asyncio.run(future_2.wait())
    self.assertEqual(controller.get_plan_cache_size(), 1)

  def test_plan_caching_preserved_on_idempotent_reregistration_via_facade(self):
    bind_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bind_sock.bind(("127.0.0.1", 0))
    port = bind_sock.getsockname()[1]
    bind_sock.close()

    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=port, worker_rpc_client=client, enable_plan_cache=True
    )
    server = raiden_controller.RaidenControllerServer(controller)
    server.start()
    facade = raiden_controller.RaidenControllerClientFacade(f"127.0.0.1:{port}")

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )
    v1 = raiden_service_pb2.VariableMetadataProto(
        name="layer_0.weight",
        shape=[64],
        mesh_shape=[1],
        layout=[0],
        item_size=4,
        layer_idx=0,
    )

    try:
      facade.register_work_unit(
          src,
          ["127.0.0.1:8000"],
          control_plane_rpc_address="127.0.0.1:9000",
          mesh_shape=[1],
          variables=[v1],
      )
      facade.register_work_unit(
          target,
          ["127.0.0.1:8001"],
          control_plane_rpc_address="127.0.0.1:9001",
          mesh_shape=[1],
          variables=[v1],
      )

      # Precompute transfer plan via warmup
      asyncio.run(
          controller.warmup_transfer_plan(
              src_units=[src],
              dst_units=[target],
          )
      )
      self.assertEqual(controller.get_plan_cache_size(), 1)

      # Periodic keepalive re-registration from workers over RPC (facade)
      for _ in range(3):
        facade.register_work_unit(
            src,
            ["127.0.0.1:8000"],
            control_plane_rpc_address="127.0.0.1:9000",
            mesh_shape=[1],
            variables=[v1],
        )
        facade.register_work_unit(
            target,
            ["127.0.0.1:8001"],
            control_plane_rpc_address="127.0.0.1:9001",
            mesh_shape=[1],
            variables=[v1],
        )

      # Plan cache must remain preserved (not invalidated by keepalive RPCs)
      self.assertEqual(controller.get_plan_cache_size(), 1)

      # Starting transfer reuses the cached plan
      future = controller.start_transfer(
          src_units=[src],
          dst_units=[target],
          use_block_chunks=True,
          req_id="bench_iter",
      )
      asyncio.run(future.wait())
      self.assertEqual(controller.get_plan_cache_size(), 1)

      # Re-registering with changed shards via facade invalidates the cache
      facade.register_work_unit(
          target,
          ["127.0.0.1:8002"],
          control_plane_rpc_address="127.0.0.1:9001",
          mesh_shape=[1],
          variables=[v1],
      )
      self.assertEqual(controller.get_plan_cache_size(), 0)
    finally:
      server.stop()
      server._thread.join(timeout=2)

  def test_plan_caching_unrelated_unit_registration_does_not_invalidate(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10009, worker_rpc_client=client, enable_plan_cache=True
    )

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )
    controller.register_work_unit(
        src,
        ["10.0.0.1:8000"],
        mesh_shape=[1],
        layout=[0],
        global_shape=[64],
        itemsize=4,
    )
    controller.register_work_unit(
        target,
        ["10.0.0.2:8000"],
        mesh_shape=[1],
        layout=[0],
        global_shape=[64],
        itemsize=4,
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[target],
        use_block_chunks=True,
    )
    asyncio.run(future.wait())
    self.assertEqual(controller.get_plan_cache_size(), 1)

    # Registering a brand new unrelated work unit does NOT invalidate
    # existing plan.
    other_unit = raiden_controller.RaidenId(
        job_name="monitor", job_replica_id="0", data_name="metrics"
    )
    controller.register_work_unit(
        other_unit,
        ["10.0.0.3:8000"],
        mesh_shape=[1],
        layout=[0],
        global_shape=[64],
        itemsize=4,
    )
    self.assertEqual(controller.get_plan_cache_size(), 1)

  def test_plan_caching_clear_cache(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10007, worker_rpc_client=client
    )

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )

    controller.register_work_unit(
        src,
        ["10.0.0.1:8000"],
        mesh_shape=[1],
        layout=[0],
        global_shape=[64],
        itemsize=4,
        control_plane_rpc_address="10.0.0.1:9000",
    )
    controller.register_work_unit(
        target,
        ["10.0.0.2:8000"],
        mesh_shape=[1],
        layout=[0],
        global_shape=[64],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[target],
        use_block_chunks=True,
    )
    asyncio.run(future.wait())
    self.assertEqual(controller.get_plan_cache_size(), 1)

    controller.clear_plan_cache()
    self.assertEqual(controller.get_plan_cache_size(), 0)

  def test_plan_caching_disabled(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10008, worker_rpc_client=client, enable_plan_cache=False
    )

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )

    controller.register_work_unit(
        src,
        ["10.0.0.1:8000"],
        mesh_shape=[1],
        layout=[0],
        global_shape=[64],
        itemsize=4,
        control_plane_rpc_address="10.0.0.1:9000",
    )
    controller.register_work_unit(
        target,
        ["10.0.0.2:8000"],
        mesh_shape=[1],
        layout=[0],
        global_shape=[64],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[target],
        use_block_chunks=True,
    )
    asyncio.run(future.wait())
    self.assertEqual(controller.get_plan_cache_size(), 0)

  def test_plan_caching_multi_target_distinct_keys(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10009, worker_rpc_client=client
    )

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target_0 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )
    target_1 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="1", data_name="weights"
    )

    controller.register_work_unit(
        src,
        ["10.0.0.1:8000"],
        mesh_shape=[1],
        layout=[0],
        global_shape=[64],
        itemsize=4,
        control_plane_rpc_address="10.0.0.1:9000",
    )
    controller.register_work_unit(
        target_0,
        ["10.0.0.2:8000"],
        mesh_shape=[1],
        layout=[0],
        global_shape=[64],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
    )
    controller.register_work_unit(
        target_1,
        ["10.0.0.3:8000"],
        mesh_shape=[1],
        layout=[0],
        global_shape=[64],
        itemsize=4,
        control_plane_rpc_address="10.0.0.3:9000",
    )

    # Transfer to target_0
    future_0 = controller.start_transfer(
        src_units=[src],
        dst_units=[target_0],
        use_block_chunks=True,
    )
    asyncio.run(future_0.wait())
    self.assertEqual(controller.get_plan_cache_size(), 1)

    # Transfer to target_1
    future_1 = controller.start_transfer(
        src_units=[src],
        dst_units=[target_1],
        use_block_chunks=True,
    )
    asyncio.run(future_1.wait())
    self.assertEqual(controller.get_plan_cache_size(), 2)

    # Transfer to both target_0 and target_1
    future_both = controller.start_transfer(
        src_units=[src],
        dst_units=[target_0, target_1],
        use_block_chunks=True,
    )
    asyncio.run(future_both.wait())
    self.assertEqual(controller.get_plan_cache_size(), 3)

  def test_pipelined_d2h_and_push_execution(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10010, worker_rpc_client=client
    )

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )

    controller.register_work_unit(
        src,
        ["10.0.0.1:8000"],
        mesh_shape=[1],
        layout=[0],
        global_shape=[64],
        itemsize=4,
        control_plane_rpc_address="10.0.0.1:9000",
    )
    controller.register_work_unit(
        target,
        ["10.0.0.2:8000"],
        mesh_shape=[1],
        layout=[0],
        global_shape=[64],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[target],
        use_block_chunks=True,
        skip_d2h=False,
    )
    asyncio.run(future.wait())

    self.assertEqual(len(client.calls), 2)
    # Call 1: Destination receiver arming
    dst_target, dst_plan = client.calls[0]
    self.assertEqual(dst_target, target)

    # Call 2: Source P2P push execution with skip_d2h=False for C++ pipelining
    src_target, src_plan = client.calls[1]
    self.assertEqual(src_target, src)
    self.assertFalse(src_plan.skip_d2h)
    self.assertIn(src, src_plan.shard_push_schedules)


class GetGlobalIndicesTest(absltest.TestCase):

  def test_global_shard_indices_bypasses_mesh_geometry(self):
    unit = raiden_controller.RaidenId("trainer", "1", "weights")
    shards = ["10.0.0.2:8000"] * 4
    # Explicit global_shard_indices bypasses all geometry.
    # Note that host_subgrid, mesh_axes, sharding_spec, and physical_mesh_shape
    # are all omitted/None, demonstrating they are no longer necessary.
    indices = raiden_controller._get_global_indices(
        unit,
        shards,
        logical_mesh_shape=[8, 2],
        layout=[1, 0],
        num_physical_hosts=4,
        global_shard_indices=[12, 13, 14, 15],
    )
    self.assertEqual(indices, [(0, 12), (1, 13), (2, 14), (3, 15)])

  def test_global_shard_indices_strided_ep_mapping(self):
    # Strided expert parallelism: host 0 gets shards [0, 2, 4, 6],
    # host 1 gets shards [1, 3, 5, 7].
    unit0 = raiden_controller.RaidenId("trainer", "0", "weights")
    unit1 = raiden_controller.RaidenId("trainer", "1", "weights")
    shards = ["10.0.0.1:8000"] * 4

    indices0 = raiden_controller._get_global_indices(
        unit0,
        shards,
        logical_mesh_shape=[8],
        layout=[0],
        num_physical_hosts=2,
        global_shard_indices=[0, 2, 4, 6],
    )
    self.assertEqual(indices0, [(0, 0), (1, 2), (2, 4), (3, 6)])

    indices1 = raiden_controller._get_global_indices(
        unit1,
        shards,
        logical_mesh_shape=[8],
        layout=[0],
        num_physical_hosts=2,
        global_shard_indices=[1, 3, 5, 7],
    )
    self.assertEqual(indices1, [(0, 1), (1, 3), (2, 5), (3, 7)])

  def test_global_shard_indices_mismatched_length_fallback(self):
    unit = raiden_controller.RaidenId("trainer", "0", "weights")
    shards = ["10.0.0.1:8000"] * 4
    # Length of global_shard_indices is 2, while num_shards is 4.
    # Falls back to legacy geometry calculation without crashing.
    indices = raiden_controller._get_global_indices(
        unit,
        shards,
        logical_mesh_shape=[1, 4],
        layout=[1, 0],
        num_physical_hosts=1,
        global_shard_indices=[0, 1],
    )
    self.assertEqual(indices, [(0, 0), (1, 1), (2, 2), (3, 3)])

  def test_single_host(self):
    unit = raiden_controller.RaidenId("trainer", "0", "weights")
    shards = ["10.0.0.1:8000"] * 4
    indices = raiden_controller._get_global_indices(
        unit,
        shards,
        logical_mesh_shape=[1, 4],
        layout=[1, 0],
        num_physical_hosts=1,
    )
    self.assertEqual(indices, [(0, 0), (1, 1), (2, 2), (3, 3)])

  def test_multi_host_matching_axis(self):
    unit = raiden_controller.RaidenId("trainer", "1", "weights")
    shards = ["10.0.0.2:8000"] * 4
    indices = raiden_controller._get_global_indices(
        unit,
        shards,
        logical_mesh_shape=[4, 4],
        layout=[1, 0],
        num_physical_hosts=4,
    )
    self.assertEqual(indices, [(0, 4), (1, 5), (2, 6), (3, 7)])

  def test_multi_host_non_matching_axis_fallback_without_spec(self):
    unit = raiden_controller.RaidenId("trainer", "1", "weights")
    shards = ["10.0.0.2:8000"] * 4
    indices = raiden_controller._get_global_indices(
        unit,
        shards,
        logical_mesh_shape=[8, 2],
        layout=[1, 0],
        num_physical_hosts=4,
    )
    self.assertEqual(indices, [(0, 0), (1, 1), (2, 2), (3, 3)])

  def test_multi_host_non_matching_axis_with_spec(self):
    unit = raiden_controller.RaidenId("trainer", "1", "weights")
    shards = ["10.0.0.2:8000"] * 4
    indices = raiden_controller._get_global_indices(
        unit,
        shards,
        logical_mesh_shape=[8, 2],
        layout=[1, 0],
        num_physical_hosts=4,
        sharding_spec=["tp", "fsdp"],
        mesh_axes=["fsdp", "tp"],
        physical_mesh_shape=[2, 8],
        host_subgrid=[1, 4],
    )
    self.assertEqual(indices, [(0, 8), (1, 10), (2, 12), (3, 14)])

  def test_multi_host_non_matching_axis_with_spec_fsdp_major_var(self):
    unit = raiden_controller.RaidenId("trainer", "1", "weights")
    shards = ["10.0.0.2:8000"] * 4
    indices = raiden_controller._get_global_indices(
        unit,
        shards,
        logical_mesh_shape=[2, 8],
        layout=[1, 0],
        num_physical_hosts=4,
        sharding_spec=["fsdp", "tp"],
        mesh_axes=["fsdp", "tp"],
        physical_mesh_shape=[2, 8],
        host_subgrid=[1, 4],
    )
    self.assertEqual(indices, [(0, 4), (1, 5), (2, 6), (3, 7)])

  def test_multi_host_2d_mesh_with_spec_fsdp_tp(self):
    unit0 = raiden_controller.RaidenId("trainer", "0", "weights")
    unit1 = raiden_controller.RaidenId("trainer", "1", "weights")
    shards = ["10.0.0.1:8000"] * 8
    indices0 = raiden_controller._get_global_indices(
        unit0,
        shards,
        logical_mesh_shape=[2, 8],
        layout=[1, 0],
        num_physical_hosts=2,
        sharding_spec=["fsdp", "tp"],
        mesh_axes=["fsdp", "tp"],
        physical_mesh_shape=[2, 8],
        host_subgrid=[1, 8],
    )
    self.assertEqual(indices0, [(i, i) for i in range(8)])
    indices1 = raiden_controller._get_global_indices(
        unit1,
        shards,
        logical_mesh_shape=[2, 8],
        layout=[1, 0],
        num_physical_hosts=2,
        sharding_spec=["fsdp", "tp"],
        mesh_axes=["fsdp", "tp"],
        physical_mesh_shape=[2, 8],
        host_subgrid=[1, 8],
    )
    self.assertEqual(indices1, [(i, i + 8) for i in range(8)])

  def test_multi_host_2d_mesh_square_mesh_single_axis_partitioned(self):
    """Tests 4x4 physical mesh across 4 hosts (4 devices/host)."""
    unit0 = raiden_controller.RaidenId("trainer", "0", "weights")
    unit1 = raiden_controller.RaidenId("trainer", "1", "weights")
    unit2 = raiden_controller.RaidenId("trainer", "2", "weights")
    unit3 = raiden_controller.RaidenId("trainer", "3", "weights")
    shards = ["10.0.0.1:8000"] * 4
    expected_indices = {
        unit0: [(0, 0), (1, 1), (2, 4), (3, 5)],
        unit1: [(0, 2), (1, 3), (2, 6), (3, 7)],
        unit2: [(0, 8), (1, 9), (2, 12), (3, 13)],
        unit3: [(0, 10), (1, 11), (2, 14), (3, 15)],
    }
    for unit in [unit0, unit1, unit2, unit3]:
      indices = raiden_controller._get_global_indices(
          unit,
          shards,
          logical_mesh_shape=[4, 4],
          layout=[1, 0],
          num_physical_hosts=4,
          sharding_spec=["fsdp", "tp"],
          mesh_axes=["fsdp", "tp"],
          physical_mesh_shape=[4, 4],
          host_subgrid=[2, 2],
      )
      self.assertEqual(indices, expected_indices[unit])

  def test_multi_host_2d_mesh_with_spec_tp_fsdp(self):
    unit0 = raiden_controller.RaidenId("trainer", "0", "weights")
    unit1 = raiden_controller.RaidenId("trainer", "1", "weights")
    shards = ["10.0.0.1:8000"] * 8
    indices0 = raiden_controller._get_global_indices(
        unit0,
        shards,
        logical_mesh_shape=[8, 2],
        layout=[0, 1],
        num_physical_hosts=2,
        sharding_spec=["tp", "fsdp"],
        mesh_axes=["fsdp", "tp"],
        physical_mesh_shape=[2, 8],
        host_subgrid=[1, 8],
    )
    self.assertEqual(indices0, [(i, i * 2) for i in range(8)])
    indices1 = raiden_controller._get_global_indices(
        unit1,
        shards,
        logical_mesh_shape=[8, 2],
        layout=[0, 1],
        num_physical_hosts=2,
        sharding_spec=["tp", "fsdp"],
        mesh_axes=["fsdp", "tp"],
        physical_mesh_shape=[2, 8],
        host_subgrid=[1, 8],
    )
    self.assertEqual(indices1, [(i, i * 2 + 1) for i in range(8)])

  def test_multi_host_2d_mesh_1d_tensor_with_spec(self):
    unit0 = raiden_controller.RaidenId("trainer", "0", "weights")
    unit1 = raiden_controller.RaidenId("trainer", "1", "weights")
    shards = ["10.0.0.1:8000"] * 8
    indices0 = raiden_controller._get_global_indices(
        unit0,
        shards,
        logical_mesh_shape=[2],
        layout=[0],
        num_physical_hosts=2,
        sharding_spec=["fsdp"],
        mesh_axes=["fsdp", "tp"],
        physical_mesh_shape=[2, 8],
        host_subgrid=[1, 8],
    )
    self.assertEqual(indices0, [(i, 0) for i in range(8)])
    indices1 = raiden_controller._get_global_indices(
        unit1,
        shards,
        logical_mesh_shape=[2],
        layout=[0],
        num_physical_hosts=2,
        sharding_spec=["fsdp"],
        mesh_axes=["fsdp", "tp"],
        physical_mesh_shape=[2, 8],
        host_subgrid=[1, 8],
    )
    self.assertEqual(indices1, [(i, 1) for i in range(8)])

  def test_multi_host_3d_mesh_data_parallelism_with_spec(self):
    unit0 = raiden_controller.RaidenId("sampler", "0", "weights")
    unit1 = raiden_controller.RaidenId("sampler", "1", "weights")
    shards = ["10.0.0.1:8000"] * 8
    indices0 = raiden_controller._get_global_indices(
        unit0,
        shards,
        logical_mesh_shape=[2, 4],
        layout=[1, 0],
        num_physical_hosts=2,
        sharding_spec=["fsdp", "tp"],
        mesh_axes=["data", "fsdp", "tp"],
        physical_mesh_shape=[2, 2, 4],
        host_subgrid=[1, 2, 4],
    )
    self.assertEqual(indices0, [(i, i) for i in range(8)])
    indices1 = raiden_controller._get_global_indices(
        unit1,
        shards,
        logical_mesh_shape=[2, 4],
        layout=[1, 0],
        num_physical_hosts=2,
        sharding_spec=["fsdp", "tp"],
        mesh_axes=["data", "fsdp", "tp"],
        physical_mesh_shape=[2, 2, 4],
        host_subgrid=[1, 2, 4],
    )
    self.assertEqual(indices1, [(i, i) for i in range(8)])

  def test_multi_host_3d_mesh_1d_tensor_with_spec(self):
    unit0 = raiden_controller.RaidenId("sampler", "0", "weights")
    unit1 = raiden_controller.RaidenId("sampler", "1", "weights")
    shards = ["10.0.0.1:8000"] * 8
    indices0 = raiden_controller._get_global_indices(
        unit0,
        shards,
        logical_mesh_shape=[2],
        layout=[0],
        num_physical_hosts=2,
        sharding_spec=["fsdp"],
        mesh_axes=["data", "fsdp", "tp"],
        physical_mesh_shape=[2, 2, 4],
        host_subgrid=[1, 2, 4],
    )
    self.assertEqual(
        indices0,
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (6, 1), (7, 1)],
    )
    indices1 = raiden_controller._get_global_indices(
        unit1,
        shards,
        logical_mesh_shape=[2],
        layout=[0],
        num_physical_hosts=2,
        sharding_spec=["fsdp"],
        mesh_axes=["data", "fsdp", "tp"],
        physical_mesh_shape=[2, 2, 4],
        host_subgrid=[1, 2, 4],
    )
    self.assertEqual(
        indices1,
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (6, 1), (7, 1)],
    )

  def test_multi_host_non_contiguous_subgrid_4x4x4_mesh_3d_tensor(self):
    """Tests 1x2x2 host subgrids on a 4x4x4 physical mesh (16 hosts, 4 chips/host)."""
    unit0 = raiden_controller.RaidenId("trainer", "0", "weights")
    unit1 = raiden_controller.RaidenId("trainer", "1", "weights")
    unit2 = raiden_controller.RaidenId("trainer", "2", "weights")
    shards = ["10.0.0.1:8000"] * 4

    # Full 3D sharding across (x, y, z)
    indices0 = raiden_controller._get_global_indices(
        unit0,
        shards,
        logical_mesh_shape=[4, 4, 4],
        layout=[2, 1, 0],
        num_physical_hosts=16,
        sharding_spec=["x", "y", "z"],
        mesh_axes=["x", "y", "z"],
        physical_mesh_shape=[4, 4, 4],
        host_subgrid=[1, 2, 2],
    )
    # Host 0 (0,0,0) -> chips (0,0,0)->0, (0,0,1)->1, (0,1,0)->4, (0,1,1)->5
    self.assertEqual(indices0, [(0, 0), (1, 1), (2, 4), (3, 5)])

    indices1 = raiden_controller._get_global_indices(
        unit1,
        shards,
        logical_mesh_shape=[4, 4, 4],
        layout=[2, 1, 0],
        num_physical_hosts=16,
        sharding_spec=["x", "y", "z"],
        mesh_axes=["x", "y", "z"],
        physical_mesh_shape=[4, 4, 4],
        host_subgrid=[1, 2, 2],
    )
    # Host 1 (0,0,1) -> chips (0,0,2)->2, (0,0,3)->3, (0,1,2)->6, (0,1,3)->7
    self.assertEqual(indices1, [(0, 2), (1, 3), (2, 6), (3, 7)])

    indices2 = raiden_controller._get_global_indices(
        unit2,
        shards,
        logical_mesh_shape=[4, 4, 4],
        layout=[2, 1, 0],
        num_physical_hosts=16,
        sharding_spec=["x", "y", "z"],
        mesh_axes=["x", "y", "z"],
        physical_mesh_shape=[4, 4, 4],
        host_subgrid=[1, 2, 2],
    )
    # Host 2 (0,1,0) -> chips (0,2,0)->8, (0,2,1)->9, (0,3,0)->12, (0,3,1)->13
    self.assertEqual(indices2, [(0, 8), (1, 9), (2, 12), (3, 13)])

  def test_multi_host_non_contiguous_subgrid_4x4x4_mesh_2d_tensor(self):
    """Tests 2D sharded tensor on 1x2x2 host subgrids on a 4x4x4 physical mesh."""
    unit0 = raiden_controller.RaidenId("trainer", "0", "weights")
    unit1 = raiden_controller.RaidenId("trainer", "1", "weights")
    shards = ["10.0.0.1:8000"] * 4

    indices0 = raiden_controller._get_global_indices(
        unit0,
        shards,
        logical_mesh_shape=[4, 4],
        layout=[1, 0],
        num_physical_hosts=16,
        sharding_spec=["x", "y"],
        mesh_axes=["x", "y", "z"],
        physical_mesh_shape=[4, 4, 4],
        host_subgrid=[1, 2, 2],
    )
    # Host 0 coords (0,0,0) -> (x,y)=(0,0), (0,0), (0,1), (0,1) -> global (0, 0, 1, 1)
    self.assertEqual(indices0, [(0, 0), (1, 0), (2, 1), (3, 1)])

    indices1 = raiden_controller._get_global_indices(
        unit1,
        shards,
        logical_mesh_shape=[4, 4],
        layout=[1, 0],
        num_physical_hosts=16,
        sharding_spec=["x", "y"],
        mesh_axes=["x", "y", "z"],
        physical_mesh_shape=[4, 4, 4],
        host_subgrid=[1, 2, 2],
    )
    # Host 1 coords (0,0,1) -> (x,y)=(0,0), (0,0), (0,1), (0,1) -> global (0, 0, 1, 1)
    self.assertEqual(indices1, [(0, 0), (1, 0), (2, 1), (3, 1)])

  def test_multi_host_non_contiguous_subgrid_4x4x4_mesh_trailing_unpartitioned(
      self,
  ):
    """Tests 2D tensor on (4, 4, 4) mesh partitioned only on first dimension."""
    unit0 = raiden_controller.RaidenId("trainer", "0", "weights")
    shards = ["10.0.0.1:8000"] * 4
    indices0 = raiden_controller._get_global_indices(
        unit0,
        shards,
        logical_mesh_shape=[4, 2],
        layout=[1, 0],
        num_physical_hosts=16,
        sharding_spec=["x"],
        mesh_axes=["x", "y", "z"],
        physical_mesh_shape=[4, 4, 4],
        host_subgrid=[1, 2, 2],
    )
    # x=0 on (4, 2) mesh with trailing unpartitioned dim -> global_idx = 0*2 + 0 = 0
    self.assertEqual(indices0, [(0, 0), (1, 0), (2, 0), (3, 0)])

    unit4 = raiden_controller.RaidenId("trainer", "4", "weights")
    # Host 4 coords in host_grid (4, 2, 2): temp_h=4 -> (1, 0, 0) -> x=1 -> global_idx = 1*2 + 0 = 2
    indices4 = raiden_controller._get_global_indices(
        unit4,
        shards,
        logical_mesh_shape=[4, 2],
        layout=[1, 0],
        num_physical_hosts=16,
        sharding_spec=["x"],
        mesh_axes=["x", "y", "z"],
        physical_mesh_shape=[4, 4, 4],
        host_subgrid=[1, 2, 2],
    )
    self.assertEqual(indices4, [(0, 2), (1, 2), (2, 2), (3, 2)])

  def test_replicated_variable(self):
    unit = raiden_controller.RaidenId("trainer", "1", "weights")
    shards = ["10.0.0.2:8000"] * 4
    indices = raiden_controller._get_global_indices(
        unit,
        shards,
        logical_mesh_shape=[1, 1],
        layout=[1, 0],
        num_physical_hosts=4,
    )
    self.assertEqual(indices, [(0, 0), (1, 0), (2, 0), (3, 0)])

  def test_variable_resharding_with_grouping(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10005, worker_rpc_client=client
    )
    controller.broadcast_k = 1

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target_0 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )
    target_1 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="1", data_name="weights"
    )

    variables_src = [
        raiden_service_pb2.VariableMetadataProto(
            name=f"weights_{i}",
            shape=[8, 8],
            mesh_shape=[1, 2],
            layout=[-1, 0],
            item_size=4,
            layer_idx=i,
        )
        for i in range(4)
    ]

    variables_dst = [
        raiden_service_pb2.VariableMetadataProto(
            name=f"weights_{i}",
            shape=[8, 8],
            mesh_shape=[
                1,
                1,
            ],  # Use [1, 1] to trigger replicated branch in _get_global_indices
            layout=[-1, -1],
            item_size=4,
            layer_idx=i,
        )
        for i in range(4)
    ]

    controller.register_work_unit(
        src,
        ["10.0.0.1:8000", "10.0.0.1:8000"],  # 2 shards
        mesh_shape=[1, 2],
        layout=[-1, 0],
        global_shape=[8, 8],
        itemsize=4,
        control_plane_rpc_address="10.0.0.1:9000",
        variables=variables_src,
    )
    controller.register_work_unit(
        target_0,
        ["10.0.0.2:8000", "10.0.0.2:8000"],  # 2 shards
        mesh_shape=[1, 2],
        layout=[-1, -1],
        global_shape=[8, 8],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
        variables=variables_dst,
    )
    controller.register_work_unit(
        target_1,
        ["10.0.0.3:8000", "10.0.0.3:8000"],  # 2 shards
        mesh_shape=[1, 2],
        layout=[-1, -1],
        global_shape=[8, 8],
        itemsize=4,
        control_plane_rpc_address="10.0.0.3:9000",
        variables=variables_dst,
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[target_0, target_1],
        use_block_chunks=True,
        group_size=2,
    )
    asyncio.run(future.wait())

    # Filter out D2H calls
    src_calls = [
        call
        for call in client.calls
        if call[0] == src and "_d2h_" not in call[1].req_id
    ]
    target_0_calls = [
        call
        for call in client.calls
        if call[0] == target_0 and "_d2h_" not in call[1].req_id
    ]
    target_1_calls = [
        call
        for call in client.calls
        if call[0] == target_1 and "_d2h_" not in call[1].req_id
    ]

    # Under tree broadcast with broadcast_k=1, src sends only to target_0 (4 calls),
    # target_0 relays to target_1 (4 calls as receiver + 4 calls as sender = 8 calls),
    # and target_1 receives from target_0 (4 calls).
    self.assertLen(src_calls, 4)
    self.assertLen(target_0_calls, 8)
    self.assertLen(target_1_calls, 4)

    for _, plan in src_calls:
      self.assertIsNotNone(plan.shard_push_schedules)
      schedules = plan.shard_push_schedules.get(src)
      if schedules:
        for shard_idx, entries in schedules.items():
          self.assertLen(entries, 4)  # 2 variables * 2 devices
          dst_shards = [entry[1] for entry in entries]
          self.assertEqual(sorted(dst_shards), [0, 0, 1, 1])
          layer_indices = [entry[10] for entry in entries]
          self.assertEqual(len(set(layer_indices)), 2)
          self.assertEqual(layer_indices[0] // 2, layer_indices[2] // 2)

  def test_is_nd_slice_tile_aligned_1d(self):
    src_slice = [(0, 2304)]
    dst_slice = [(0, 1152)]
    intersection = [(0, 1152)]
    self.assertFalse(
        raiden_controller.is_nd_slice_tile_aligned(
            src_slice, dst_slice, intersection
        )
    )
    # 0-D scalar slices
    self.assertFalse(raiden_controller.is_nd_slice_tile_aligned([], [], []))
    self.assertEqual(
        raiden_controller.generate_strided_copy_chunks_tile_aware(
            [], [], [], itemsize=2
        ),
        [(0, 0, 2, 0, 0, 1)],
    )

  def test_is_nd_slice_tile_aligned_2d(self):
    # Aligned 2D slice: multiples of (8, 128)
    src_slice = [(0, 64032), (0, 2304)]
    dst_slice = [(0, 128064), (0, 1152)]
    intersection = [(0, 64032), (0, 1152)]
    self.assertTrue(
        raiden_controller.is_nd_slice_tile_aligned(
            src_slice, dst_slice, intersection, tile_shape=(8, 128)
        )
    )

    # Unaligned column start
    unaligned_col_start_int = [(0, 64032), (10, 1162)]
    self.assertFalse(
        raiden_controller.is_nd_slice_tile_aligned(
            src_slice, dst_slice, unaligned_col_start_int, tile_shape=(8, 128)
        )
    )

    # Unaligned row length (not divisible by 8)
    unaligned_row_len_int = [(0, 64035), (0, 1152)]
    self.assertFalse(
        raiden_controller.is_nd_slice_tile_aligned(
            src_slice, dst_slice, unaligned_row_len_int, tile_shape=(8, 128)
        )
    )

    # Unaligned column length (not divisible by 128)
    unaligned_col_len_int = [(0, 64032), (0, 1000)]
    self.assertFalse(
        raiden_controller.is_nd_slice_tile_aligned(
            src_slice, dst_slice, unaligned_col_len_int, tile_shape=(8, 128)
        )
    )

  def test_generate_strided_copy_chunks_1d(self):
    src_slice = [(0, 5376)]
    dst_slice_0 = [(0, 2688)]
    intersection_0 = [(0, 2688)]
    chunks_0 = raiden_controller.generate_strided_copy_chunks(
        src_slice, dst_slice_0, intersection_0, itemsize=2
    )
    self.assertEqual(chunks_0, [(0, 0, 2688 * 2, 0, 0, 1)])

    dst_slice_1 = [(2688, 5376)]
    intersection_1 = [(2688, 5376)]
    chunks_1 = raiden_controller.generate_strided_copy_chunks(
        src_slice, dst_slice_1, intersection_1, itemsize=2
    )
    self.assertEqual(chunks_1, [(2688 * 2, 0, 2688 * 2, 0, 0, 1)])

  def test_generate_strided_copy_chunks_tile_aware_1d(self):
    src_slice = [(0, 2304)]
    dst_slice = [(0, 1152)]
    intersection = [(0, 1152)]
    chunks = raiden_controller.generate_strided_copy_chunks_tile_aware(
        src_slice, dst_slice, intersection, itemsize=4
    )
    self.assertEqual(chunks, [(0, 0, 1152 * 4, 0, 0, 1)])

  def test_generate_strided_copy_chunks_tile_aware_2d_reshard(self):
    # Embedding resharding: [64032, 2304] -> [128064, 1152]
    # Source shard 0 -> Destination shard 0
    src_0 = [(0, 64032), (0, 2304)]
    dst_0 = [(0, 128064), (0, 1152)]
    int_0 = [(0, 64032), (0, 1152)]

    chunks_0 = raiden_controller.generate_strided_copy_chunks_tile_aware(
        src_0, dst_0, int_0, itemsize=4, tile_shape=(8, 128)
    )
    # Expected: size = 1152 * 8 * 4 = 36864, src_stride = 2304 * 8 * 4 = 73728,
    # dst_stride = 1152 * 8 * 4 = 36864, count = 64032 / 8 = 8004
    self.assertEqual(
        chunks_0,
        [(0, 0, 36864, 73728, 36864, 8004)],
    )

    # Source shard 0 -> Destination shard 1 (offset col = 1152)
    dst_1 = [(0, 128064), (1152, 2304)]
    int_1 = [(0, 64032), (1152, 2304)]
    chunks_1 = raiden_controller.generate_strided_copy_chunks_tile_aware(
        src_0, dst_1, int_1, itemsize=4, tile_shape=(8, 128)
    )
    # local_src_col = 1152 -> src_offset = 1152 * 8 * 4 = 36864
    self.assertEqual(
        chunks_1,
        [(36864, 0, 36864, 73728, 36864, 8004)],
    )

    # Source shard 1 -> Destination shard 0 (offset row = 64032)
    src_1 = [(64032, 128064), (0, 2304)]
    int_src1_dst0 = [(64032, 128064), (0, 1152)]
    chunks_src1_dst0 = (
        raiden_controller.generate_strided_copy_chunks_tile_aware(
            src_1, dst_0, int_src1_dst0, itemsize=4, tile_shape=(8, 128)
        )
    )
    # local_dst_row = 64032 -> dst_offset = 64032 * 1152 * 4 = 295059456
    self.assertEqual(
        chunks_src1_dst0,
        [(0, 295059456, 36864, 73728, 36864, 8004)],
    )

  def test_generate_strided_copy_chunks_tile_aware_row_parallel_merge(self):
    # When sharding only along row (FSDP), size == src_stride == dst_stride -> merge to 1 chunk
    src_slice = [(0, 1024), (0, 512)]
    dst_slice = [(0, 1024), (0, 512)]
    intersection = [(0, 1024), (0, 512)]
    chunks = raiden_controller.generate_strided_copy_chunks_tile_aware(
        src_slice, dst_slice, intersection, itemsize=4, tile_shape=(8, 128)
    )
    self.assertEqual(
        chunks,
        [(0, 0, 1024 * 512 * 4, 0, 0, 1)],
    )

  def test_generate_strided_copy_chunks_tile_aware_3d_batch(self):
    # 3D tensor: [batch=2, H=16, W=128]
    src_slice = [(0, 2), (0, 16), (0, 128)]
    dst_slice = [(0, 2), (0, 16), (0, 128)]
    intersection = [(0, 2), (0, 16), (0, 128)]
    chunks = raiden_controller.generate_strided_copy_chunks_tile_aware(
        src_slice, dst_slice, intersection, itemsize=4, tile_shape=(8, 128)
    )
    self.assertEqual(chunks, [(0, 0, 16384, 0, 0, 1)])

  def test_generate_strided_copy_chunks_tile_aware_3d_sharded_batch(self):
    # 3D tensor: [B=8, H=16, W=128]. Source has B=[0, 4), Dest has B=[2, 6) -> Intersect B=[2, 4)
    src_slice = [(0, 4), (0, 16), (0, 128)]
    dst_slice = [(2, 6), (0, 16), (0, 128)]
    intersection = [(2, 4), (0, 16), (0, 128)]
    chunks = raiden_controller.generate_strided_copy_chunks_tile_aware(
        src_slice, dst_slice, intersection, itemsize=4, tile_shape=(8, 128)
    )
    self.assertEqual(chunks, [(16384, 0, 16384, 0, 0, 1)])

  def test_generate_strided_copy_chunks_tile_aware_3d_strided_outer_dst(self):
    # 3D tensor: [E=4, H=16, W=128]. Source holds [4, 8, 128], Dest holds [4, 16, 128].
    # Intersection is [4, 8, 128]. Inner 2D slice [8, 128] is contiguous (4096 bytes),
    # while outer dimension E=4 has src_stride=4096 and dst_stride=8192.
    src_slice = [(0, 4), (0, 8), (0, 128)]
    dst_slice = [(0, 4), (0, 16), (0, 128)]
    intersection = [(0, 4), (0, 8), (0, 128)]
    chunks = raiden_controller.generate_strided_copy_chunks_tile_aware(
        src_slice, dst_slice, intersection, itemsize=4, tile_shape=(8, 128)
    )
    self.assertEqual(chunks, [(0, 0, 4096, 4096, 8192, 4)])

  def test_generate_strided_copy_chunks_tile_aware_4d_tensor(self):
    # 4D tensor: [Experts=2, Heads=2, H=16, W=128]
    src_slice = [(0, 2), (0, 2), (0, 16), (0, 128)]
    dst_slice = [(0, 2), (0, 2), (0, 16), (0, 128)]
    intersection = [(0, 2), (0, 2), (0, 16), (0, 128)]
    chunks = raiden_controller.generate_strided_copy_chunks_tile_aware(
        src_slice, dst_slice, intersection, itemsize=4, tile_shape=(8, 128)
    )
    # 2 * 2 = 4 outer slices
    self.assertLen(chunks, 4)
    # Each matrix is 16 * 128 * 4 = 8192 bytes
    for i in range(4):
      self.assertEqual(
          chunks[i],
          (i * 8192, i * 8192, 128 * 8 * 4, 128 * 8 * 4, 128 * 8 * 4, 2),
      )

  def test_replicated_source_resharding_deduplication(self):
    """Tests that when source ranks replicate a tensor, only one source pushes to each destination."""
    dummy_client = DummyWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10010, worker_rpc_client=dummy_client
    )

    # 4 Source units with mesh_shape=[1, 4] (fsdp=1, tp=4)
    # 1D layernorm tensor (2304,) has sharding_spec=['fsdp'] -> mesh_shape=[1]
    # Replicated on all 4 source units: each holds [0, 2304]
    src_units = []
    for r in range(4):
      u = raiden_controller.RaidenId("trainer", str(r), "model_weights")
      src_units.append(u)
      var = raiden_service_pb2.VariableMetadataProto(
          name="layer_19.input_layernorm",
          shape=[2304],
          mesh_shape=[1],
          layout=[0],
          item_size=2,
          layer_idx=198,
          sharding_spec=["fsdp"],
      )
      controller.register_work_unit(
          u,
          [f"10.0.0.1:{8000 + r}"],
          control_plane_rpc_address=f"10.0.0.1:{9000 + r}",
          mesh_shape=[1, 4],
          mesh_axes=["fsdp", "tp"],
          variables=[var],
      )

    # 4 Destination units with mesh_shape=[2, 2] (fsdp=2, tp=2)
    # 1D layernorm tensor (2304,) has sharding_spec=['fsdp'] -> mesh_shape=[2]
    # Sampler ranks 0 & 1 hold slice 0 [0, 1152]; ranks 2 & 3 hold slice 1 [1152, 2304]
    dst_units = []
    for r in range(4):
      u = raiden_controller.RaidenId("sampler", str(r), "model_weights")
      dst_units.append(u)
      var = raiden_service_pb2.VariableMetadataProto(
          name="layer_19.input_layernorm",
          shape=[2304],
          mesh_shape=[2],
          layout=[0],
          item_size=2,
          layer_idx=198,
          sharding_spec=["fsdp"],
      )
      controller.register_work_unit(
          u,
          [f"10.0.0.2:{8000 + r}"],
          control_plane_rpc_address=f"10.0.0.2:{9000 + r}",
          mesh_shape=[2, 2],
          mesh_axes=["fsdp", "tp"],
          variables=[var],
      )

    future = controller.start_transfer(
        src_units=src_units,
        dst_units=dst_units,
        use_block_chunks=True,
    )
    asyncio.run(future.wait())

    plan = controller.get_plan("req_0")

    # Each destination rank must expect exactly 1 chunk for layer 198 (not 4)
    for dst_u in dst_units:
      layer_counts = plan.dst_expected_layer_chunk_counts.get(dst_u, {})
      self.assertEqual(
          layer_counts.get(198, 0),
          1,
          f"Destination unit {dst_u} expected chunk count for layer 198 should"
          f" be 1, but got {layer_counts.get(198, 0)}",
      )

    # Count how many times each destination endpoint is targeted across all source schedules
    dst_target_counts = {f"10.0.0.2:{8000 + r}": 0 for r in range(4)}
    for src_u, schedules in plan.shard_push_schedules.items():
      for shard_idx, entries in schedules.items():
        for entry in entries:
          dst_peer = entry[0]
          layer_idx = entry[10] if len(entry) > 10 else 0
          if layer_idx == 198 and dst_peer in dst_target_counts:
            dst_target_counts[dst_peer] += 1

    for peer, count in dst_target_counts.items():
      self.assertEqual(
          count,
          1,
          f"Destination peer {peer} should receive exactly 1 push for layer"
          f" 198, but got {count}",
      )

  def test_parallel_multi_variable_resharding_plan(self):
    """Tests parallel resharding schedule generation across multiple variables and ranks."""
    dummy_client = DummyWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10011, worker_rpc_client=dummy_client, enable_plan_cache=True
    )

    num_vars = 60
    # 4 Source units (fsdp=1, tp=4)
    src_units = []
    for r in range(4):
      u = raiden_controller.RaidenId("trainer", str(r), "model_weights")
      src_units.append(u)
      vars_list = []
      for v_idx in range(num_vars):
        if v_idx % 2 == 0:
          # 2D matrix (2048, 2048) column sharded
          vars_list.append(
              raiden_service_pb2.VariableMetadataProto(
                  name=f"layer_{v_idx}.mlp.gate_proj",
                  shape=[2048, 2048],
                  mesh_shape=[1, 4],
                  layout=[1, 0],
                  item_size=2,
                  layer_idx=v_idx,
                  sharding_spec=["fsdp", "tp"],
              )
          )
        else:
          # 1D vector (2048,) replicated
          vars_list.append(
              raiden_service_pb2.VariableMetadataProto(
                  name=f"layer_{v_idx}.input_norm",
                  shape=[2048],
                  mesh_shape=[1],
                  layout=[0],
                  item_size=2,
                  layer_idx=v_idx,
                  sharding_spec=["fsdp"],
              )
          )
      controller.register_work_unit(
          u,
          [f"10.0.0.1:{8000 + r}"],
          control_plane_rpc_address=f"10.0.0.1:{9000 + r}",
          mesh_shape=[1, 4],
          mesh_axes=["fsdp", "tp"],
          variables=vars_list,
      )

    # 4 Destination units (fsdp=2, tp=2)
    dst_units = []
    for r in range(4):
      u = raiden_controller.RaidenId("sampler", str(r), "model_weights")
      dst_units.append(u)
      vars_list = []
      for v_idx in range(num_vars):
        if v_idx % 2 == 0:
          vars_list.append(
              raiden_service_pb2.VariableMetadataProto(
                  name=f"layer_{v_idx}.mlp.gate_proj",
                  shape=[2048, 2048],
                  mesh_shape=[2, 2],
                  layout=[1, 0],
                  item_size=2,
                  layer_idx=v_idx,
                  sharding_spec=["fsdp", "tp"],
              )
          )
        else:
          vars_list.append(
              raiden_service_pb2.VariableMetadataProto(
                  name=f"layer_{v_idx}.input_norm",
                  shape=[2048],
                  mesh_shape=[2],
                  layout=[0],
                  item_size=2,
                  layer_idx=v_idx,
                  sharding_spec=["fsdp"],
              )
          )
      controller.register_work_unit(
          u,
          [f"10.0.0.2:{8000 + r}"],
          control_plane_rpc_address=f"10.0.0.2:{9000 + r}",
          mesh_shape=[2, 2],
          mesh_axes=["fsdp", "tp"],
          variables=vars_list,
      )

    future = controller.start_transfer(
        src_units=src_units,
        dst_units=dst_units,
        use_block_chunks=True,
        uuid=5555,
        req_id="parallel_test_req_0",
    )
    asyncio.run(future.wait())

    plan = controller.get_plan("parallel_test_req_0")
    self.assertIsNotNone(plan)

    # Check all variables exist in the generated schedules
    scheduled_layers = set()
    for src_u, schedules in plan.shard_push_schedules.items():
      for shard_idx, entries in schedules.items():
        for entry in entries:
          layer_idx = entry[10] if len(entry) > 10 else 0
          scheduled_layers.add(layer_idx)

    self.assertEqual(len(scheduled_layers), num_vars)

    # Verify each destination unit has all layers in its expected chunk counts
    for dst_u in dst_units:
      layer_counts = plan.dst_expected_layer_chunk_counts.get(dst_u, {})
      for v_idx in range(num_vars):
        self.assertGreater(
            layer_counts.get(v_idx, 0),
            0,
            f"Destination {dst_u} missing expected chunks for layer {v_idx}",
        )

    # Verify second transfer invocation reuses cached schedule seamlessly
    future2 = controller.start_transfer(
        src_units=src_units,
        dst_units=dst_units,
        use_block_chunks=True,
        uuid=5556,
        req_id="parallel_test_req_1",
    )
    asyncio.run(future2.wait())
    plan2 = controller.get_plan("parallel_test_req_1")
    self.assertEqual(
        len(plan2.shard_push_schedules), len(plan.shard_push_schedules)
    )

  def test_zero_copy_plan_cache_immutability_and_reuse(self):
    dummy_client = DummyWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10012, worker_rpc_client=dummy_client, enable_plan_cache=True
    )
    src_unit = raiden_controller.RaidenId(
        job_name="trainer",
        job_replica_id="0",
        data_name="weights_0",
        data_replica_idx=0,
    )
    dst_unit = raiden_controller.RaidenId(
        job_name="inference",
        job_replica_id="0",
        data_name="weights_0",
        data_replica_idx=0,
    )
    vars_list = [
        raiden_service_pb2.VariableMetadataProto(
            name="layer_0.weight",
            shape=[1024, 1024],
            mesh_shape=[1, 1],
            layout=[1, 0],
            item_size=4,
            layer_idx=0,
            sharding_spec=["fsdp", "tp"],
        )
    ]
    controller.register_work_unit(
        src_unit,
        ["10.0.0.1:8000"],
        control_plane_rpc_address="10.0.0.1:9000",
        mesh_shape=[1, 1],
        mesh_axes=["fsdp", "tp"],
        variables=vars_list,
    )
    controller.register_work_unit(
        dst_unit,
        ["10.0.0.2:8000"],
        control_plane_rpc_address="10.0.0.2:9000",
        mesh_shape=[1, 1],
        mesh_axes=["fsdp", "tp"],
        variables=vars_list,
    )

    # First transfer compiles and caches the plan
    fut1 = controller.start_transfer(
        src_units=[src_unit],
        dst_units=[dst_unit],
        use_block_chunks=True,
        uuid=1001,
        req_id="zero_copy_req_0",
    )
    asyncio.run(fut1.wait())
    plan1 = controller.get_plan("zero_copy_req_0")
    self.assertIsNotNone(plan1)

    # Second transfer reuses cached plan with zero-copy
    fut2 = controller.start_transfer(
        src_units=[src_unit],
        dst_units=[dst_unit],
        use_block_chunks=True,
        uuid=1002,
        req_id="zero_copy_req_1",
    )
    asyncio.run(fut2.wait())
    plan2 = controller.get_plan("zero_copy_req_1")
    self.assertIsNotNone(plan2)

    # Verify requests have independent dynamic IDs but share identical
    # pre-computed schedules
    self.assertEqual(plan1.uuid, 1001)
    self.assertEqual(plan2.uuid, 1002)
    self.assertEqual(plan1.shard_push_schedules, plan2.shard_push_schedules)

  def test_1d_rank1_tensor_resharding_offsets(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10000, worker_rpc_client=client
    )
    num_hosts = 2
    num_shards_per_host = 8
    src_units = [
        raiden_controller.RaidenId("trainer", str(r), "model_weights")
        for r in range(num_hosts)
    ]
    dst_units = [
        raiden_controller.RaidenId("sampler", str(r), "model_weights")
        for r in range(num_hosts)
    ]

    # 1D layernorm (shape (5376,), P("fsdp"), item_size 2)
    # 2D projection matrix (shape (5376, 4096), P("fsdp", "tp"), item_size 2)
    src_vars = [
        raiden_service_pb2.VariableMetadataProto(
            name="layer_0.input_layernorm",
            shape=[5376],
            mesh_shape=[1],
            layout=[0],
            item_size=2,
            layer_idx=0,
            sharding_spec=["fsdp"],
        ),
        raiden_service_pb2.VariableMetadataProto(
            name="layer_0.attn.q_proj",
            shape=[5376, 4096],
            mesh_shape=[1, 16],
            layout=[1, 0],
            item_size=2,
            layer_idx=1,
            sharding_spec=["fsdp", "tp"],
        ),
    ]

    dst_vars = [
        raiden_service_pb2.VariableMetadataProto(
            name="layer_0.input_layernorm",
            shape=[5376],
            mesh_shape=[2],
            layout=[0],
            item_size=2,
            layer_idx=0,
            sharding_spec=["fsdp"],
        ),
        raiden_service_pb2.VariableMetadataProto(
            name="layer_0.attn.q_proj",
            shape=[5376, 4096],
            mesh_shape=[2, 8],
            layout=[1, 0],
            item_size=2,
            layer_idx=1,
            sharding_spec=["fsdp", "tp"],
        ),
    ]

    for r in range(num_hosts):
      shards_src = [f"10.0.0.{r}:{s}" for s in range(num_shards_per_host)]
      controller.register_work_unit(
          src_units[r],
          shards_src,
          f"10.0.0.{r}:10000",
          mesh_shape=[1, 16],
          variables=src_vars,
          mesh_axes=["fsdp", "tp"],
      )
      shards_dst = [f"10.0.1.{r}:{s}" for s in range(num_shards_per_host)]
      controller.register_work_unit(
          dst_units[r],
          shards_dst,
          f"10.0.1.{r}:10000",
          mesh_shape=[2, 8],
          variables=dst_vars,
          mesh_axes=["fsdp", "tp"],
      )

    fut = controller.start_transfer(
        src_units=src_units,
        dst_units=dst_units,
        use_block_chunks=True,
        uuid=1001,
        req_id="test_req_0",
    )
    asyncio.run(fut.wait())

    plans_sent = {target: plan for target, plan in client.calls}
    self.assertIn(src_units[0], plans_sent)
    self.assertIn(src_units[1], plans_sent)

    # Validate 1D layernorm schedule entries:
    # Destination Host 0 (fsdp=0) slice [0:2688] -> src_block_offset=0, dst_block_offset=0, size=5376 bytes
    # Destination Host 1 (fsdp=1) slice [2688:5376] -> src_block_offset=5376, dst_block_offset=0, size=5376 bytes
    host0_layernorm_entries = []
    host1_layernorm_entries = []
    for s_unit in src_units:
      plan = plans_sent[s_unit]
      for shard_idx, entries in plan.shard_push_schedules[s_unit].items():
        for entry in entries:
          # entry tuple format:
          # (dst_peer, local_dst_idx, dst_block_offset, src_block_offset, size, src_block_id, dst_block_id, src_stride, dst_stride, count, layer_idx, 0)
          if entry[10] == 0:  # layer_idx == 0 (input_layernorm)
            dst_peer = entry[0]
            if "10.0.1.0" in dst_peer:
              host0_layernorm_entries.append(entry)
            elif "10.0.1.1" in dst_peer:
              host1_layernorm_entries.append(entry)

    self.assertLen(host0_layernorm_entries, num_shards_per_host)
    self.assertLen(host1_layernorm_entries, num_shards_per_host)

    for entry in host0_layernorm_entries:
      self.assertEqual(entry[2], 0)  # dst_block_offset
      self.assertEqual(entry[3], 0)  # src_block_offset
      self.assertEqual(entry[4], 2688 * 2)  # size in bytes
      self.assertEqual(entry[5], 0)  # src_block_id
      self.assertEqual(entry[6], 0)  # dst_block_id

    for entry in host1_layernorm_entries:
      self.assertEqual(entry[2], 0)  # dst_block_offset
      self.assertEqual(entry[3], 2688 * 2)  # src_block_offset = 5376 bytes
      self.assertEqual(entry[4], 2688 * 2)  # size in bytes
      self.assertEqual(entry[5], 0)  # src_block_id
      self.assertEqual(entry[6], 0)  # dst_block_id

  def test_skip_tiling_2d_vs_1d_behavior(self):
    """Verifies that 2D layers skip tiling while 1D layers use CPU tiling."""
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10000,
        worker_rpc_client=client,
    )

    src_units = [
        raiden_controller.RaidenId("trainer", "0", "weights"),
        raiden_controller.RaidenId("trainer", "1", "weights"),
    ]
    dst_units = [
        raiden_controller.RaidenId("sampler", "0", "weights"),
    ]

    vars_list = [
        raiden_service_pb2.VariableMetadataProto(
            name="layer_0.input_layernorm",
            shape=[5376],
            mesh_shape=[2],
            layout=[0],
            item_size=2,
            layer_idx=0,
            sharding_spec=["fsdp"],
        ),
        raiden_service_pb2.VariableMetadataProto(
            name="layer_0.attn.q_proj",
            shape=[5376, 4096],
            mesh_shape=[2, 8],
            layout=[1, 0],
            item_size=2,
            layer_idx=1,
            sharding_spec=["fsdp", "tp"],
        ),
    ]

    # Register Source: (fsdp=2, tp=8) across 2 hosts
    for r in range(2):
      shards_src = [f"10.0.0.{r}:{s}" for s in range(8)]
      controller.register_work_unit(
          src_units[r],
          shards_src,
          f"10.0.0.{r}:10000",
          mesh_shape=[2, 8],
          variables=vars_list,
          mesh_axes=["fsdp", "tp"],
      )

    # Register Destination: (fsdp=2, tp=4) on 1 host
    shards_dst = [f"10.0.1.0:{s}" for s in range(8)]
    dst_vars = [
        raiden_service_pb2.VariableMetadataProto(
            name="layer_0.input_layernorm",
            shape=[5376],
            mesh_shape=[2],
            layout=[0],
            item_size=2,
            layer_idx=0,
            sharding_spec=["fsdp"],
        ),
        raiden_service_pb2.VariableMetadataProto(
            name="layer_0.attn.q_proj",
            shape=[5376, 4096],
            mesh_shape=[2, 4],
            layout=[1, 0],
            item_size=2,
            layer_idx=1,
            sharding_spec=["fsdp", "tp"],
        ),
    ]
    controller.register_work_unit(
        dst_units[0],
        shards_dst,
        "10.0.1.0:10000",
        mesh_shape=[2, 4],
        variables=dst_vars,
        mesh_axes=["fsdp", "tp"],
    )

    fut = controller.start_transfer(
        src_units=src_units,
        dst_units=dst_units,
        use_block_chunks=True,
        uuid=2002,
        req_id="test_req_skip_tiling",
    )
    asyncio.run(fut.wait())

    plans_sent = {target: plan for target, plan in client.calls}
    for s_unit in src_units:
      plan = plans_sent[s_unit]
      # Layer 0 (1D Layernorm) MUST NOT skip tiling (skip_tiling=False)
      self.assertFalse(plan.skip_tiling.get(0, False))
      # Layer 1 (2D Q_Proj) MUST skip tiling (skip_tiling=True)
      self.assertTrue(plan.skip_tiling.get(1, False))

  def test_skip_tiling_identical_reference_case(self):
    """Verifies reference case (identical topology): 2D weights skip tiling, 1D do not."""
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10000,
        worker_rpc_client=client,
    )

    src_units = [
        raiden_controller.RaidenId("trainer", "0", "weights"),
        raiden_controller.RaidenId("trainer", "1", "weights"),
    ]
    dst_units = [
        raiden_controller.RaidenId("sampler", "0", "weights"),
        raiden_controller.RaidenId("sampler", "1", "weights"),
    ]

    vars_list = [
        raiden_service_pb2.VariableMetadataProto(
            name="layer_0.input_layernorm",
            shape=[5376],
            mesh_shape=[2],
            layout=[0],
            item_size=2,
            layer_idx=0,
            sharding_spec=["fsdp"],
        ),
        raiden_service_pb2.VariableMetadataProto(
            name="layer_0.attn.q_proj",
            shape=[5376, 4096],
            mesh_shape=[2, 8],
            layout=[1, 0],
            item_size=2,
            layer_idx=1,
            sharding_spec=["fsdp", "tp"],
        ),
    ]

    # Register Source: (fsdp=2, tp=8) across 2 hosts
    for r in range(2):
      shards_src = [f"10.0.0.{r}:{s}" for s in range(8)]
      controller.register_work_unit(
          src_units[r],
          shards_src,
          f"10.0.0.{r}:10000",
          mesh_shape=[2, 8],
          variables=vars_list,
          mesh_axes=["fsdp", "tp"],
      )

    # Register Destination with identical topology and variables
    for r in range(2):
      shards_dst = [f"10.0.1.{r}:{s}" for s in range(8)]
      controller.register_work_unit(
          dst_units[r],
          shards_dst,
          f"10.0.1.{r}:10000",
          mesh_shape=[2, 8],
          variables=vars_list,
          mesh_axes=["fsdp", "tp"],
      )

    fut = controller.start_transfer(
        src_units=src_units,
        dst_units=dst_units,
        use_block_chunks=True,
        uuid=2003,
        req_id="test_req_identical_ref_case",
    )
    asyncio.run(fut.wait())

    plans_sent = {target: plan for target, plan in client.calls}
    for s_unit in src_units:
      plan = plans_sent[s_unit]
      # 1D Layernorm has rank=1: must NOT skip tiling
      self.assertFalse(plan.skip_tiling.get(0, False))
      # 2D Q_Proj is identical: MUST skip tiling (zero copy)
      self.assertTrue(plan.skip_tiling.get(1, False))

  def test_multi_endpoint_registration_and_get_worker_endpoints(self):
    controller = raiden_controller.RaidenController(port=10099)
    unit = raiden_controller.RaidenId("trainer", "0", "weights", 0)
    controller.register_work_unit(
        unit,
        ["10.0.0.1:8000", "10.0.0.2:8000"],
        control_plane_rpc_address="10.0.0.1:9001,10.0.0.2:9002",
    )
    self.assertEqual(
        controller.worker_rpc_client.get_registered_endpoints(unit),
        ["10.0.0.1:9001", "10.0.0.2:9002"],
    )
    self.assertEqual(
        controller.worker_rpc_client.get_worker_endpoints()[unit],
        "10.0.0.1:9001,10.0.0.2:9002",
    )
    # Re-registration replaces endpoints
    controller.register_work_unit(
        unit,
        ["10.0.0.3:8000"],
        control_plane_rpc_address="10.0.0.3:9003",
    )
    self.assertEqual(
        controller.worker_rpc_client.get_registered_endpoints(unit),
        ["10.0.0.3:9003"],
    )

  def test_start_transfer_multi_endpoint_broadcast(self):
    recorded_calls = []

    class MockWorkerClient(raiden_controller.WorkerRpcClient):

      def _encode_start_transfer(self, target_id, transfer_plan):
        return b"dummy_payload"

      async def _send_and_verify(self, addr, payload):
        recorded_calls.append((addr, payload))

    client = MockWorkerClient()
    unit = raiden_controller.RaidenId("trainer", "0", "weights", 0)
    client.register_worker_endpoint(unit, "10.0.0.1:9001")
    client.register_worker_endpoint(unit, "10.0.0.2:9002")

    plan = mock.MagicMock(spec=raiden_controller.TransferPlan)
    asyncio.run(client.start_transfer(unit, plan))
    self.assertEqual(
        recorded_calls,
        [
            ("10.0.0.1:9001", b"dummy_payload"),
            ("10.0.0.2:9002", b"dummy_payload"),
        ],
    )

    # Test explicit comma-separated address parameter
    recorded_calls.clear()
    asyncio.run(
        client.start_transfer(unit, plan, address="10.0.0.3:9003,10.0.0.4:9004")
    )
    self.assertEqual(
        recorded_calls,
        [
            ("10.0.0.3:9003", b"dummy_payload"),
            ("10.0.0.4:9004", b"dummy_payload"),
        ],
    )

  def test_resharding_plan3_fsdp16_tp4_to_fsdp2_tp4(self):
    dummy_client = DummyWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10042, worker_rpc_client=dummy_client
    )

    src_unit = raiden_controller.RaidenId("trainer", "0", "gemma2_2b_weights")
    dst_unit = raiden_controller.RaidenId("sampler", "0", "gemma2_2b_weights")

    src_shards = ["10.0.0.1:8000"] * 64
    dst_shards = ["10.0.0.2:8000"] * 8

    src_var = raiden_service_pb2.VariableMetadataProto(
        name="layer_0.attn.q_proj",
        shape=[2304, 2048],
        mesh_shape=[16, 4],
        layout=[1, 0],
        item_size=2,
        layer_idx=0,
        sharding_spec=["fsdp", "tp"],
    )

    dst_var = raiden_service_pb2.VariableMetadataProto(
        name="layer_0.attn.q_proj",
        shape=[2304, 2048],
        mesh_shape=[2, 4],
        layout=[1, 0],
        item_size=2,
        layer_idx=0,
        sharding_spec=["fsdp", "tp"],
    )

    controller.register_work_unit(
        src_unit,
        src_shards,
        control_plane_rpc_address="10.0.0.1:9000",
        variables=[src_var],
        mesh_shape=[16, 4],
        mesh_axes=["fsdp", "tp"],
    )

    controller.register_work_unit(
        dst_unit,
        dst_shards,
        control_plane_rpc_address="10.0.0.2:9000",
        variables=[dst_var],
        mesh_shape=[2, 4],
        mesh_axes=["fsdp", "tp"],
    )

    future = controller.start_transfer(
        src_units=[src_unit],
        dst_units=[dst_unit],
        use_block_chunks=True,
        req_id="plan3_test_req",
    )
    asyncio.run(future.wait())
    plan = controller.get_plan("plan3_test_req")
    self.assertIsNotNone(plan)
    self.assertIn(src_unit, plan.shard_push_schedules)
    schedules = plan.shard_push_schedules[src_unit]
    self.assertEqual(len(schedules), 64)
    # Each destination shard in (2, 4) corresponds to 8 source shards in (16, 4)
    for shard_idx, entries in schedules.items():
      self.assertNotEmpty(entries)

  def test_resharding_plan3_multi_source_tasks_to_single_dst_task(self):
    dummy_client = DummyWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10043, worker_rpc_client=dummy_client
    )

    src_units = [
        raiden_controller.RaidenId(
            "pathways_trainer", str(i), "gemma2_2b_weights"
        )
        for i in range(32)
    ]
    dst_unit = raiden_controller.RaidenId(
        "mc_jax_sampler", "0", "gemma2_2b_weights"
    )

    src_embed_var = raiden_service_pb2.VariableMetadataProto(
        name="embedder.input_embedding",
        shape=[256128, 2304],
        mesh_shape=[4, 16],
        layout=[1, 0],
        item_size=2,
        layer_idx=0,
        sharding_spec=["tp", "fsdp"],
    )
    src_q_var = raiden_service_pb2.VariableMetadataProto(
        name="layer_0.attn.q_proj",
        shape=[2304, 2048],
        mesh_shape=[16, 4],
        layout=[1, 0],
        item_size=2,
        layer_idx=1,
        sharding_spec=["fsdp", "tp"],
    )

    dst_embed_var = raiden_service_pb2.VariableMetadataProto(
        name="embedder.input_embedding",
        shape=[256128, 2304],
        mesh_shape=[4, 2],
        layout=[1, 0],
        item_size=2,
        layer_idx=0,
        sharding_spec=["tp", "fsdp"],
    )
    dst_q_var = raiden_service_pb2.VariableMetadataProto(
        name="layer_0.attn.q_proj",
        shape=[2304, 2048],
        mesh_shape=[2, 4],
        layout=[1, 0],
        item_size=2,
        layer_idx=1,
        sharding_spec=["fsdp", "tp"],
    )

    for i, u in enumerate(src_units):
      controller.register_work_unit(
          u,
          [f"10.0.0.{i+1}:8000", f"10.0.0.{i+1}:8000"],
          control_plane_rpc_address=f"10.0.0.{i+1}:9000",
          variables=[src_embed_var, src_q_var],
          mesh_shape=[16, 4],
          mesh_axes=["fsdp", "tp"],
      )

    controller.register_work_unit(
        dst_unit,
        [f"10.0.1.1:{8000+j}" for j in range(8)],
        control_plane_rpc_address="10.0.1.1:9000",
        variables=[dst_embed_var, dst_q_var],
        mesh_shape=[2, 4],
        mesh_axes=["fsdp", "tp"],
    )

    future = controller.start_transfer(
        src_units=src_units,
        dst_units=[dst_unit],
        use_block_chunks=True,
        req_id="plan3_multi_task_req",
    )
    asyncio.run(future.wait())
    plan = controller.get_plan("plan3_multi_task_req")
    self.assertIsNotNone(plan)

    total_scheduled_src_shards = 0
    for u in src_units:
      self.assertIn(u, plan.shard_push_schedules)
      schedules = plan.shard_push_schedules[u]
      self.assertEqual(len(schedules), 2)
      for shard_idx, entries in schedules.items():
        self.assertNotEmpty(entries)
        total_scheduled_src_shards += 1
    self.assertEqual(total_scheduled_src_shards, 64)

  def test_compute_host_subgrid(self):
    test_cases = [
        # (physical_mesh_shape, devices_per_host, expected_subgrid, expected_grid)
        # 1D meshes
        ([16], 4, [4], [4]),
        ([16], 8, [8], [2]),
        ([8], 2, [2], [4]),
        # 2D meshes (row-major minor-to-major factoring)
        ([4, 4], 2, [1, 2], [4, 2]),
        ([4, 4], 4, [2, 2], [2, 2]),
        ([4, 4], 8, [2, 4], [2, 1]),
        ([4, 4], 16, [4, 4], [1, 1]),
        ([4, 8], 4, [1, 4], [4, 2]),
        ([4, 8], 8, [2, 4], [2, 2]),
        ([4, 8], 16, [2, 8], [2, 1]),
        ([8, 4], 8, [2, 4], [4, 1]),
        ([8, 8], 8, [2, 4], [4, 2]),
        ([2, 8], 4, [1, 4], [2, 2]),
        ([2, 8], 8, [1, 8], [2, 1]),
        ([8, 2], 4, [2, 2], [4, 1]),
        ([8, 2], 8, [4, 2], [2, 1]),
        ([16, 4], 2, [1, 2], [16, 2]),
        ([2, 4], 8, [2, 4], [1, 1]),
        ([16, 1], 4, [4, 1], [4, 1]),
        # 3D meshes
        ([2, 2, 2], 2, [1, 1, 2], [2, 2, 1]),
        ([2, 2, 2], 4, [1, 2, 2], [2, 1, 1]),
        ([2, 4, 4], 4, [1, 2, 2], [2, 2, 2]),
        ([4, 4, 4], 4, [1, 2, 2], [4, 2, 2]),
        ([4, 4, 4], 8, [2, 2, 2], [2, 2, 2]),
        ([4, 4, 4], 16, [2, 2, 4], [2, 2, 1]),
        ([2, 4, 8], 8, [1, 2, 4], [2, 2, 2]),
        ([2, 4, 8], 16, [1, 2, 8], [2, 2, 1]),
        # 4D meshes
        ([2, 2, 4, 4], 4, [1, 1, 2, 2], [2, 2, 2, 2]),
        ([2, 2, 4, 4], 8, [1, 1, 2, 4], [2, 2, 2, 1]),
        ([2, 2, 4, 4], 16, [1, 1, 4, 4], [2, 2, 1, 1]),
        ([4, 4, 4, 4], 4, [1, 1, 2, 2], [4, 4, 2, 2]),
        ([4, 4, 4, 4], 8, [1, 2, 2, 2], [4, 2, 2, 2]),
        ([4, 4, 4, 4], 16, [2, 2, 2, 2], [2, 2, 2, 2]),
        ([2, 2, 2, 4], 4, [1, 1, 1, 4], [2, 2, 2, 1]),
        ([2, 4, 4, 8], 8, [1, 2, 2, 2], [2, 2, 2, 4]),
        ([2, 4, 4, 8], 16, [1, 2, 2, 4], [2, 2, 2, 2]),
        # Edge cases
        ([], 4, [], []),
        ([4, 4], 0, [], []),
        ([4, 4], -1, [], []),
        ([2, 2], 8, [2, 2], [1, 1]),
    ]
    for mesh, k, expected_subgrid, expected_grid in test_cases:
      with self.subTest(mesh=mesh, k=k):
        subgrid, grid = raiden_controller.compute_host_subgrid(mesh, k)
        self.assertEqual(subgrid, expected_subgrid)
        self.assertEqual(grid, expected_grid)
        if subgrid:
          self.assertEqual(math.prod(subgrid), min(k, math.prod(mesh)))
          self.assertEqual([s * g for s, g in zip(subgrid, grid)], mesh)


class FormatUnitHelpersTest(absltest.TestCase):

  def test_format_unit(self):
    u1 = raiden_controller.RaidenId(
        job_name="trainer",
        job_replica_id="0",
        data_name="weights",
        data_replica_idx=0,
    )
    self.assertEqual(raiden_controller._format_unit(u1), "trainer:0[weights]")

    u2 = raiden_controller.RaidenId(
        job_name="trainer",
        job_replica_id="0",
        data_name="weights",
        data_replica_idx=1,
    )
    self.assertEqual(raiden_controller._format_unit(u2), "trainer:0[weights#1]")

    u3 = raiden_controller.RaidenId(
        job_name="actor", job_replica_id="1", data_name="", data_replica_idx=0
    )
    self.assertEqual(raiden_controller._format_unit(u3), "actor:1")

    u4 = raiden_controller.RaidenId(
        job_name="actor", job_replica_id="", data_name="", data_replica_idx=0
    )
    self.assertEqual(raiden_controller._format_unit(u4), "actor")

    self.assertEqual(
        raiden_controller._format_unit("endpoint:1000"), "endpoint:1000"
    )

  def test_format_units(self):
    u1 = raiden_controller.RaidenId(
        job_name="trainer",
        job_replica_id="0",
        data_name="weights",
        data_replica_idx=0,
    )
    u2 = raiden_controller.RaidenId(
        job_name="trainer",
        job_replica_id="1",
        data_name="weights",
        data_replica_idx=1,
    )

    # List of units
    self.assertEqual(
        raiden_controller._format_units([u1, u2]),
        "[trainer:0[weights], trainer:1[weights#1]]",
    )

    # Generator / custom iterable
    self.assertEqual(
        raiden_controller._format_units(u for u in [u1]),
        "[trainer:0[weights]]",
    )

    # Single unit
    self.assertEqual(raiden_controller._format_units(u1), "trainer:0[weights]")

    # Single string / bytes (must not be iterated as chars)
    self.assertEqual(raiden_controller._format_units("unit_str"), "unit_str")
    self.assertEqual(
        raiden_controller._format_units(b"unit_bytes"), "b'unit_bytes'"
    )

  def test_heterogeneous_host_mesh_weight_transfer(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10000, worker_rpc_client=client
    )
    vars_metadata = [
        raiden_service_pb2.VariableMetadataProto(
            name="w_in",
            shape=[8, 8],
            mesh_shape=[4, 4],
            layout=[1, 0],
            item_size=4,
            layer_idx=0,
            sharding_spec=["fsdp", "tp"],
        ),
        raiden_service_pb2.VariableMetadataProto(
            name="w_out",
            shape=[8, 8],
            mesh_shape=[4, 4],
            layout=[1, 0],
            item_size=4,
            layer_idx=1,
            sharding_spec=["tp", "fsdp"],
        ),
        raiden_service_pb2.VariableMetadataProto(
            name="norm",
            shape=[8],
            mesh_shape=[4],
            layout=[0],
            item_size=4,
            layer_idx=2,
            sharding_spec=["fsdp"],
        ),
    ]

    src_units = []
    for i in range(4):
      u = raiden_controller.RaidenId("src_job", str(i), "weights", 0)
      src_units.append(u)
      controller.register_work_unit(
          u,
          [f"10.0.0.{i+1}:{8000+j}" for j in range(4)],
          control_plane_rpc_address=f"10.0.0.{i+1}:9000",
          mesh_shape=[4, 4],
          variables=vars_metadata,
          mesh_axes=["fsdp", "tp"],
      )

    dst_units = []
    for i in range(2):
      u = raiden_controller.RaidenId("dst_job", str(i), "weights", 0)
      dst_units.append(u)
      controller.register_work_unit(
          u,
          [f"10.0.1.{i+1}:{8000+j}" for j in range(8)],
          control_plane_rpc_address=f"10.0.1.{i+1}:9000",
          mesh_shape=[4, 4],
          variables=vars_metadata,
          mesh_axes=["fsdp", "tp"],
      )

    loop = asyncio.new_event_loop()
    try:
      future = controller.start_transfer(
          src_units=src_units,
          dst_units=dst_units,
          dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
          use_block_chunks=True,
          uuid=123456,
          req_id="hetero_mesh_test",
          expected_block_count=0,
          group_size=1,
      )
      loop.run_until_complete(future.wait())
    finally:
      loop.close()

    # Verify calls received by worker RPC client
    self.assertEqual(len(client.calls), 6)  # 2 dst + 4 src
    dst_calls = [c for c in client.calls if c[0] in dst_units]
    src_calls = [c for c in client.calls if c[0] in src_units]
    self.assertEqual(len(dst_calls), 2)
    self.assertEqual(len(src_calls), 4)

    # Check start_transfer_request proto formatting for both src and dst
    for unit, plan in client.calls:
      req_bytes = controller.worker_rpc_client._encode_start_transfer(
          unit, plan
      )
      req = raiden_service_pb2.ControlRequest()
      req.ParseFromString(req_bytes)
      start_req = req.start_transfer_request
      print(f"\n--- Call for {unit} ---")
      print(f"expected_block_count={start_req.expected_block_count}")
      print(f"num_shard_push_schedules={len(start_req.shard_push_schedules)}")
      for shard_idx, sched in start_req.shard_push_schedules.items():
        print(f"  shard {shard_idx}: {len(sched.entries)} entries")
        for e in sched.entries[:2]:
          print(
              f"    dst_peer={e.dst_peer} dst_shard={e.dst_shard_idx}"
              f" layer={e.layer_idx} size={e.size_bytes}"
          )


class RaidenPlanWarmupTest(absltest.TestCase):
  """Tests for background schedule warmup and plan caching."""

  def test_warmup_transfer_plan_explicit(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=0, worker_rpc_client=client, enable_plan_cache=True
    )
    vars_metadata = [
        raiden_service_pb2.VariableMetadataProto(
            name="layer0",
            shape=[16, 16],
            mesh_shape=[2, 2],
            layout=[1, 0],
            item_size=4,
            layer_idx=0,
            sharding_spec=["fsdp", "tp"],
        ),
    ]
    src_units = []
    for i in range(2):
      u = raiden_controller.RaidenId("src_job", str(i), "weights", 0)
      src_units.append(u)
      controller.register_work_unit(
          u,
          [f"10.0.0.{i+1}:{8000+j}" for j in range(2)],
          control_plane_rpc_address=f"10.0.0.{i+1}:9000",
          mesh_shape=[2, 2],
          variables=vars_metadata,
          mesh_axes=["fsdp", "tp"],
      )

    dst_units = []
    for i in range(2):
      u = raiden_controller.RaidenId("dst_job", str(i), "weights", 0)
      dst_units.append(u)
      controller.register_work_unit(
          u,
          [f"10.0.1.{i+1}:{8000+j}" for j in range(2)],
          control_plane_rpc_address=f"10.0.1.{i+1}:9000",
          mesh_shape=[2, 2],
          variables=vars_metadata,
          mesh_axes=["fsdp", "tp"],
      )

    self.assertEqual(controller.get_plan_cache_size(), 0)

    # Explicit warmup outside critical path (e.g. during model compilation)
    loop = asyncio.new_event_loop()
    try:
      cached = loop.run_until_complete(
          controller.warmup_transfer_plan(
              src_units=src_units,
              dst_units=dst_units,
          )
      )
      self.assertEqual(controller.get_plan_cache_size(), 1)
      self.assertIsNotNone(cached)
      self.assertGreater(cached.expected_block_count, 0)

      # Subsequent start_transfer (with HBM) reuses the cached schedule
      transfer_future = controller.start_transfer(
          src_units=src_units,
          dst_units=dst_units,
          dst_mem_type="HBM",
          use_block_chunks=True,
          uuid=999,
          req_id="test_req_hbm",
      )
      loop.run_until_complete(transfer_future.wait())

      # A subsequent transfer to DRAM ALSO reuses the exact same cached schedule
      transfer_future_dram = controller.start_transfer(
          src_units=src_units,
          dst_units=dst_units,
          dst_mem_type="DRAM",
          use_block_chunks=True,
          uuid=1000,
          req_id="test_req_dram",
      )
      loop.run_until_complete(transfer_future_dram.wait())
    finally:
      loop.close()

    # Cache size remains 1 (both HBM and DRAM hit the same cached plan)
    self.assertEqual(controller.get_plan_cache_size(), 1)
    self.assertLen(client.calls, 8)

  def test_transfer_schedule_with_global_shard_indices_expert_parallelism(
      self,
  ):
    """Verifies routing of strided expert-parallelism shards across hosts.

    Uses global_shard_indices without host_subgrid or mesh_axes.
    """
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=0, worker_rpc_client=client, enable_plan_cache=False
    )

    # 4-way sharded tensor: shape [64, 64], mesh_shape [4, 1], layout [1, 0].
    # Total 4 slices along dim 0: [0:16], [16:32], [32:48], [48:64].
    #
    # Source has 2 hosts, 2 shards each.
    # Host 0 owns shards [0, 2] (strided EP layout across hosts).
    # Host 1 owns shards [1, 3].
    # Neither host specifies host_subgrid, mesh_axes, sharding_spec,
    # or top-level mesh_shape (obsolete fields).
    src_unit_0 = raiden_controller.RaidenId("src_ep", "0", "weights", 0)
    src_var_0 = raiden_service_pb2.VariableMetadataProto(
        name="experts",
        shape=[64, 64],
        mesh_shape=[4, 1],
        layout=[1, 0],
        item_size=4,
        layer_idx=0,
        global_shard_indices=[0, 2],
    )
    controller.register_work_unit(
        src_unit_0,
        ["10.0.0.1:8000", "10.0.0.1:8001"],
        control_plane_rpc_address="10.0.0.1:9000",
        variables=[src_var_0],
    )

    src_unit_1 = raiden_controller.RaidenId("src_ep", "1", "weights", 0)
    src_var_1 = raiden_service_pb2.VariableMetadataProto(
        name="experts",
        shape=[64, 64],
        mesh_shape=[4, 1],
        layout=[1, 0],
        item_size=4,
        layer_idx=0,
        global_shard_indices=[1, 3],
    )
    controller.register_work_unit(
        src_unit_1,
        ["10.0.0.2:8000", "10.0.0.2:8001"],
        control_plane_rpc_address="10.0.0.2:9000",
        variables=[src_var_1],
    )

    # Destination has 1 host with 4 shards owning global shards [0, 1, 2, 3].
    dst_unit_0 = raiden_controller.RaidenId("dst_ep", "0", "weights", 0)
    dst_var_0 = raiden_service_pb2.VariableMetadataProto(
        name="experts",
        shape=[64, 64],
        mesh_shape=[4, 1],
        layout=[1, 0],
        item_size=4,
        layer_idx=0,
        global_shard_indices=[0, 1, 2, 3],
    )
    controller.register_work_unit(
        dst_unit_0,
        [
            "10.0.1.1:8000",
            "10.0.1.1:8001",
            "10.0.1.1:8002",
            "10.0.1.1:8003",
        ],
        control_plane_rpc_address="10.0.1.1:9000",
        variables=[dst_var_0],
    )

    loop = asyncio.new_event_loop()
    try:
      cached = loop.run_until_complete(
          controller._compute_transfer_schedule(
              src_units=[src_unit_0, src_unit_1],
              dst_units=[dst_unit_0],
          )
      )
    finally:
      loop.close()

    self.assertIsNotNone(cached)
    sched = cached.computed_schedules

    # Host 0 local shard 0 (global shard 0) -> dst shard 0
    self.assertIn(src_unit_0, sched)
    self.assertIn(0, sched[src_unit_0])
    push_0_0 = sched[src_unit_0][0]
    self.assertLen(push_0_0, 1)
    self.assertEqual(push_0_0[0][0], "10.0.1.1:8000")
    self.assertEqual(push_0_0[0][1], 0)

    # Host 0 local shard 1 (global shard 2) -> dst shard 2
    self.assertIn(1, sched[src_unit_0])
    push_0_1 = sched[src_unit_0][1]
    self.assertLen(push_0_1, 1)
    self.assertEqual(push_0_1[0][0], "10.0.1.1:8002")
    self.assertEqual(push_0_1[0][1], 2)

    # Host 1 local shard 0 (global shard 1) -> dst shard 1
    self.assertIn(src_unit_1, sched)
    self.assertIn(0, sched[src_unit_1])
    push_1_0 = sched[src_unit_1][0]
    self.assertLen(push_1_0, 1)
    self.assertEqual(push_1_0[0][0], "10.0.1.1:8001")
    self.assertEqual(push_1_0[0][1], 1)

    # Host 1 local shard 1 (global shard 3) -> dst shard 3
    self.assertIn(1, sched[src_unit_1])
    push_1_1 = sched[src_unit_1][1]
    self.assertLen(push_1_1, 1)
    self.assertEqual(push_1_1[0][0], "10.0.1.1:8003")
    self.assertEqual(push_1_1[0][1], 3)

  def test_worker_rpc_client_executor_concurrency_defaults_to_at_least_128(
      self,
  ):
    client = raiden_controller.WorkerRpcClient()
    try:
      self.assertIsNotNone(client.executor)
      self.assertGreaterEqual(client.executor._max_workers, 128)
    finally:
      client.close()

  def test_worker_rpc_client_custom_max_workers_and_env_var(self):
    client = raiden_controller.WorkerRpcClient(max_workers=64)
    try:
      self.assertEqual(client.executor._max_workers, 64)
    finally:
      client.close()

    with mock.patch.dict(os.environ, {"RAIDEN_RPC_CONCURRENCY": "256"}):
      env_client = raiden_controller.WorkerRpcClient()
      try:
        self.assertEqual(env_client.executor._max_workers, 256)
      finally:
        env_client.close()

  def test_weight_sync_worker_rpc_client_inherits_concurrency(self):
    client = raiden_controller.WeightSyncWorkerRpcClient()
    try:
      self.assertIsNotNone(client.executor)
      self.assertGreaterEqual(client.executor._max_workers, 128)
    finally:
      client.close()

  def test_worker_rpc_client_send_rpc_runs_concurrently(self):
    client = raiden_controller.WorkerRpcClient(max_workers=32)
    try:
      # Mock _send_rpc_sync to simulate 30ms network latency per worker
      def mock_send(addr, payload, timeout=600.0):
        del addr, payload, timeout
        time.sleep(0.03)
        return b"ok"

      client._send_rpc_sync = mock_send

      async def _run_all():
        tasks = [client._send_rpc(f"worker_{i}", b"data") for i in range(32)]
        return await asyncio.gather(*tasks)

      loop = asyncio.new_event_loop()
      try:
        start_time = time.perf_counter()
        results = loop.run_until_complete(_run_all())
        elapsed = time.perf_counter() - start_time
        self.assertEqual(results, [b"ok"] * 32)
        # 32 serialized calls would take 32 * 0.03s = 0.96s.
        # Concurrently on 32 workers, it should take < 0.25s.
        self.assertLess(elapsed, 0.5)
      finally:
        loop.close()
    finally:
      client.close()

  def test_host_subgrid_enumeration_all_meshes_and_devices_per_host(self):
    """Verifies that _get_global_indices forms a strict disjoint bijection.

    Enumerates real multi-host TPU configurations across 1D, 2D, and 3D meshes
    (16 to 128 chips), physical TPU host device counts (4 chips/host on
    v4/v5p/v6e
    and 8 chips/host on v2/v3/v5e/v6e), and all valid host_subgrid
    factorizations.
    Ensures compute_host_subgrid is bypassed and all devices in the cluster are
    covered without gaps or collisions.
    """
    mesh_shapes = [
        # 1D meshes (pure data-parallel or tensor-parallel)
        [16],
        [32],
        [64],
        [128],
        # 2D meshes (e.g. FSDP x TP, Data x Model on 16, 32, 64, 128 chips)
        [2, 8],
        [8, 2],
        [4, 4],
        [4, 8],
        [8, 4],
        [8, 8],
        [16, 4],
        [4, 16],
        [16, 8],
        [8, 16],
        # 3D meshes (physical 3D torus topologies or Data x FSDP x TP)
        [2, 2, 4],
        [2, 4, 4],
        [4, 4, 4],
        [2, 4, 8],
        [4, 4, 8],
    ]
    # Real TPU host VM architectures have strictly 4 or 8 chips per host.
    devices_per_host_candidates = [4, 8]

    def find_all_subgrids(
        shape: list[int], target_prod: int
    ) -> list[list[int]]:
      results = []

      def backtrack(dim: int, remaining: int, current: list[int]):
        if dim == len(shape) - 1:
          if remaining <= shape[dim] and shape[dim] % remaining == 0:
            results.append(current + [remaining])
          return
        limit = min(remaining, shape[dim])
        for s in range(1, limit + 1):
          if remaining % s == 0 and shape[dim] % s == 0:
            backtrack(dim + 1, remaining // s, current + [s])

      backtrack(0, target_prod, [])
      return results

    evaluated_cases = 0
    with mock.patch.object(
        raiden_controller,
        "compute_host_subgrid",
        side_effect=AssertionError(
            "compute_host_subgrid should not be called!"
        ),
    ):
      for mesh_shape in mesh_shapes:
        total_devices = math.prod(mesh_shape)
        mesh_axes = [f"dim_{d}" for d in range(len(mesh_shape))]
        layout = list(range(len(mesh_shape) - 1, -1, -1))

        for devices_per_host in devices_per_host_candidates:
          if (
              devices_per_host > total_devices
              or total_devices % devices_per_host != 0
          ):
            continue
          total_hosts = total_devices // devices_per_host
          if total_hosts < 2:
            continue
          subgrids = find_all_subgrids(mesh_shape, devices_per_host)

          for subgrid in subgrids:
            evaluated_cases += 1
            all_global_indices = set()
            for host_id in range(total_hosts):
              unit = raiden_controller.RaidenId(
                  "worker", str(host_id), "weights"
              )
              indices = raiden_controller._get_global_indices(
                  unit=unit,
                  shards=["10.0.0.1:8000"] * devices_per_host,
                  logical_mesh_shape=mesh_shape,
                  layout=layout,
                  num_physical_hosts=total_hosts,
                  sharding_spec=mesh_axes,
                  mesh_axes=mesh_axes,
                  physical_mesh_shape=mesh_shape,
                  host_subgrid=subgrid,
              )
              self.assertLen(
                  indices,
                  devices_per_host,
                  f"Wrong shard count for mesh={mesh_shape}, subgrid={subgrid}",
              )
              for shard_idx, global_idx in indices:
                self.assertNotIn(
                    global_idx,
                    all_global_indices,
                    f"Duplicate global index {global_idx} encountered for"
                    f" mesh={mesh_shape}, dph={devices_per_host},"
                    f" subgrid={subgrid}, host={host_id}, shard={shard_idx}",
                )
                all_global_indices.add(global_idx)

            self.assertEqual(
                all_global_indices,
                set(range(total_devices)),
                "Global indices do not form a complete bijection for"
                f" mesh={mesh_shape}, dph={devices_per_host},"
                f" subgrid={subgrid}",
            )

    self.assertGreater(evaluated_cases, 50)

  def test_register_work_unit_stores_and_serializes_host_subgrid(self):
    controller = raiden_controller.RaidenController(port=10099)
    unit = raiden_controller.RaidenId("trainer", "0", "weights")
    controller.register_work_unit(
        unit=unit,
        shards=["10.0.0.1:8000"] * 4,
        mesh_shape=[4, 8],
        layout=[1, 0],
        global_shape=[128, 64],
        itemsize=4,
        mesh_axes=["fsdp", "tp"],
        host_subgrid=[2, 2],
    )
    with controller._lock:
      self.assertEqual(controller._registered_host_subgrids.get(unit), [2, 2])
      proto = controller._metadata_proto_locked(unit)
      self.assertEqual(list(proto.host_subgrid), [2, 2])

  def test_qwen_norm_scale_schedule(self):
    controller = raiden_controller.RaidenController(port=10100)
    src_units = [
        raiden_controller.RaidenId("pathways_trainer", str(i), "weights")
        for i in range(16)
    ]
    dst_units = [
        raiden_controller.RaidenId("mc_jax_sampler", str(i), "weights")
        for i in range(16)
    ]

    for i, s_unit in enumerate(src_units):
      controller.register_work_unit(
          unit=s_unit,
          shards=[f"10.0.0.{i}:8000", f"10.0.0.{i}:8001"],
          mesh_shape=[16, 2],
          variables=[
              raiden_service_pb2.VariableMetadataProto(
                  name="decoder.decoder_norm.scale",
                  shape=[2048],
                  mesh_shape=[16],
                  layout=[0],
                  item_size=2,
                  layer_idx=0,
                  sharding_spec=["fsdp"],
              )
          ],
          mesh_axes=["fsdp", "tp"],
      )

    for i, d_unit in enumerate(dst_units):
      controller.register_work_unit(
          unit=d_unit,
          shards=[
              f"10.0.1.{i}:8000",
              f"10.0.1.{i}:8001",
              f"10.0.1.{i}:8002",
              f"10.0.1.{i}:8003",
          ],
          mesh_shape=[32, 2],
          variables=[
              raiden_service_pb2.VariableMetadataProto(
                  name="decoder.decoder_norm.scale",
                  shape=[2048],
                  mesh_shape=[1],
                  layout=[0],
                  item_size=2,
                  layer_idx=0,
                  sharding_spec=[""],
              )
          ],
          mesh_axes=["fsdp", "tp"],
      )

    sched = asyncio.run(
        controller._compute_transfer_schedule(src_units, dst_units)
    )
    # Check coverage for each destination unit and local shard
    # Each dst shard must receive all 2048 elements (4096 bytes: 2048 * 2 bytes)
    coverage_by_dst = {}
    for d_unit in dst_units:
      for s_idx in range(4):
        coverage_by_dst[(d_unit, s_idx)] = set()
    for unit_sched in sched.direct_schedules.values():
      for entries in unit_sched.values():
        for entry in entries:
          dst_peer = entry[0]
          local_dst_idx = entry[1]
          dst_offset = entry[2]
          size = entry[4]
          d_unit = sched.data_address_to_unit.get(dst_peer)
          if d_unit:
            coverage_by_dst[(d_unit, local_dst_idx)].update(
                range(dst_offset, dst_offset + size)
            )

    for (d_unit, local_dst_idx), covered_bytes in coverage_by_dst.items():
      self.assertLen(
          covered_bytes,
          4096,
          f"Dst {d_unit} shard {local_dst_idx} received {len(covered_bytes)}"
          " bytes instead of 4096! Missing bytes:"
          f" {set(range(4096)) - covered_bytes}",
      )


class WeightSyncReceiverAndCacheLeakTest(absltest.TestCase):
  """Tests for receiver push schedule omission and step-growth memory/cache fixes."""

  def test_weight_sync_worker_rpc_client_skips_receiver_push_schedules(self):
    src_unit = raiden_controller.RaidenId("src_job", "0", "weights", 0)
    dst_unit = raiden_controller.RaidenId("dst_job", "0", "weights", 0)
    schedule_entries = [
        ("10.0.1.1:8000", 0, 0, 0, 1024, 0, 0, 1024, 1024, 1, 2, 0),
        ("10.0.1.1:8000", 0, 1024, 1024, 1024, 0, 0, 1024, 1024, 1, 2, 0),
    ]
    plan = raiden_controller.TransferPlan(
        src_units=[src_unit],
        dst_units=[dst_unit],
        plan=None,
        shard_push_schedules={src_unit: {0: schedule_entries}},
        worker_rpc_addresses={
            src_unit: "10.0.0.1:9000",
            dst_unit: "10.0.1.1:9000",
        },
        worker_data_addresses={
            src_unit: ["10.0.0.1:8000"],
            dst_unit: ["10.0.1.1:8000"],
        },
        uuid=42,
        use_block_chunks=True,
        is_sender=True,
        expected_block_count=10,
        dst_expected_block_counts={dst_unit: 8},
        dst_expected_layer_chunk_counts={dst_unit: {2: 8}},
        skip_tiling={2: True},
        req_id="wsync_test",
    )

    ws_client = raiden_controller.WeightSyncWorkerRpcClient()
    base_client = raiden_controller.WorkerRpcClient()
    try:
      # 1. WeightSyncWorkerRpcClient on receiver (dst_unit):
      # shard_push_schedules MUST be empty, while expected counts & skip_tiling
      # ARE populated.
      dst_bytes = ws_client._encode_start_transfer(dst_unit, plan)
      dst_req = raiden_service_pb2.ControlRequest()
      dst_req.ParseFromString(dst_bytes)
      start_dst = dst_req.start_transfer_request
      self.assertFalse(start_dst.is_sender)
      self.assertEmpty(start_dst.shard_push_schedules)
      self.assertEqual(start_dst.expected_block_count, 8)
      self.assertEqual(dict(start_dst.expected_layer_chunk_counts), {2: 8})
      self.assertEqual(dict(start_dst.skip_tiling), {2: True})

      # 2. WeightSyncWorkerRpcClient on sender (src_unit):
      # shard_push_schedules MUST be populated with local schedule.
      src_bytes = ws_client._encode_start_transfer(src_unit, plan)
      src_req = raiden_service_pb2.ControlRequest()
      src_req.ParseFromString(src_bytes)
      start_src = src_req.start_transfer_request
      self.assertTrue(start_src.is_sender)
      self.assertLen(start_src.shard_push_schedules, 1)
      self.assertLen(start_src.shard_push_schedules[0].entries, 2)

      # 3. Base WorkerRpcClient on receiver (dst_unit) when is_weight_sync=False
      # preserves legacy behavior (populates filtered receiver schedules).
      legacy_dst_bytes = base_client._encode_start_transfer(dst_unit, plan)
      legacy_dst_req = raiden_service_pb2.ControlRequest()
      legacy_dst_req.ParseFromString(legacy_dst_bytes)
      self.assertFalse(legacy_dst_req.start_transfer_request.is_sender)
      self.assertLen(
          legacy_dst_req.start_transfer_request.shard_push_schedules, 1
      )

      # 4. Base WorkerRpcClient on receiver (dst_unit) when is_weight_sync=True:
      # automatically skips receiver shard_push_schedules even with default
      # WorkerRpcClient!
      plan.is_weight_sync = True
      ws_auto_dst_bytes = base_client._encode_start_transfer(dst_unit, plan)
      ws_auto_dst_req = raiden_service_pb2.ControlRequest()
      ws_auto_dst_req.ParseFromString(ws_auto_dst_bytes)
      self.assertFalse(ws_auto_dst_req.start_transfer_request.is_sender)
      self.assertEmpty(
          ws_auto_dst_req.start_transfer_request.shard_push_schedules
      )

      # 5. Sender path caches pre-built ShardPushScheduleProto in
      # sender_push_schedule_protos (already populated by step 2 above).
      self.assertIn(src_unit, plan.sender_push_schedule_protos)
      cached_proto = plan.sender_push_schedule_protos[src_unit][0]
      # Clear raw tuple schedule to prove subsequent calls reuse cached_proto
      # without reading tuples.
      plan.shard_push_schedules[src_unit][0] = []
      src_bytes_2 = base_client._encode_start_transfer(src_unit, plan)
      self.assertEqual(src_bytes, src_bytes_2)
      self.assertIs(plan.sender_push_schedule_protos[src_unit][0], cached_proto)
    finally:
      ws_client.close()
      base_client.close()

  def test_raiden_id_has_slots_for_gc_untracking(self):
    rid = raiden_controller.RaidenId("job", "0", "weights", 0)
    self.assertFalse(hasattr(rid, "__dict__"))

  def test_multi_host_receiver_endpoint_counts_specialization(self):
    ws_client = raiden_controller.WeightSyncWorkerRpcClient()
    dst_unit = raiden_controller.RaidenId("rollout", "", "weights", 0)
    src_unit = raiden_controller.RaidenId("trainer", "0", "weights", 0)
    plan = raiden_controller.TransferPlan(
        src_units=[src_unit],
        dst_units=[dst_unit],
        plan=None,
        worker_data_addresses={
            dst_unit: ["10.0.1.2:8001", "10.0.1.3:8001"],
            src_unit: ["10.0.1.1:8001"],
        },
        use_block_chunks=True,
        is_sender=True,
        expected_block_count=200,
        dst_expected_block_counts={dst_unit: 200},
        dst_endpoint_counts={
            "10.0.1.2": 120,
            "10.0.1.3": 80,
        },
        dst_endpoint_layer_counts={
            "10.0.1.2": {0: 60, 1: 60},
            "10.0.1.3": {0: 40, 1: 40},
        },
        is_weight_sync=True,
    )
    try:
      dst_bytes_host2 = ws_client._encode_start_transfer(
          dst_unit, plan, address="10.0.1.2:8000"
      )
      req2 = raiden_service_pb2.ControlRequest()
      req2.ParseFromString(dst_bytes_host2)
      start_dst2 = req2.start_transfer_request
      self.assertFalse(start_dst2.is_sender)
      self.assertEqual(start_dst2.expected_block_count, 120)
      self.assertEqual(
          dict(start_dst2.expected_layer_chunk_counts), {0: 60, 1: 60}
      )

      dst_bytes_host3 = ws_client._encode_start_transfer(
          dst_unit, plan, address="10.0.1.3:8000"
      )
      req3 = raiden_service_pb2.ControlRequest()
      req3.ParseFromString(dst_bytes_host3)
      start_dst3 = req3.start_transfer_request
      self.assertFalse(start_dst3.is_sender)
      self.assertEqual(start_dst3.expected_block_count, 80)
      self.assertEqual(
          dict(start_dst3.expected_layer_chunk_counts), {0: 40, 1: 40}
      )

      dst_bytes_default = ws_client._encode_start_transfer(dst_unit, plan)
      req_default = raiden_service_pb2.ControlRequest()
      req_default.ParseFromString(dst_bytes_default)
      self.assertEqual(
          req_default.start_transfer_request.expected_block_count, 200
      )

      # When a single control listener manages multiple NUMA NICs on one host,
      # _encode_start_transfer should preserve the full host expected counts.
      ws_client._endpoints[dst_unit] = ["10.0.1.2:8000"]
      dst_bytes_single_listener = ws_client._encode_start_transfer(
          dst_unit, plan, address="10.0.1.2:8000"
      )
      req_single = raiden_service_pb2.ControlRequest()
      req_single.ParseFromString(dst_bytes_single_listener)
      self.assertEqual(
          req_single.start_transfer_request.expected_block_count, 200
      )
    finally:
      ws_client.close()

  def test_source_ephemeral_port_reregistration_preserves_plan_cache(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=0, worker_rpc_client=client, enable_plan_cache=True
    )
    vars_metadata = [
        raiden_service_pb2.VariableMetadataProto(
            name="layer0",
            shape=[16, 16],
            mesh_shape=[1, 1],
            layout=[1, 0],
            item_size=4,
            layer_idx=0,
        ),
    ]
    src_unit = raiden_controller.RaidenId("trainer", "0", "weights", 0)
    dst_unit = raiden_controller.RaidenId("rollout", "0", "weights", 0)

    controller.register_work_unit(
        src_unit,
        ["10.0.0.1:8000"],
        control_plane_rpc_address="10.0.0.1:9000",
        mesh_shape=[1, 1],
        variables=vars_metadata,
    )
    controller.register_work_unit(
        dst_unit,
        ["10.0.1.1:8000"],
        control_plane_rpc_address="10.0.1.1:9000",
        mesh_shape=[1, 1],
        variables=vars_metadata,
    )

    fut1 = controller.start_transfer(
        src_units=[src_unit],
        dst_units=[dst_unit],
        use_block_chunks=True,
        req_id="wsync-1",
    )
    asyncio.run(fut1.wait())
    self.assertIsNone(fut1._transfer_task)
    self.assertEqual(controller.get_plan_cache_size(), 1)
    self.assertIn("wsync-1", controller._active_transfers)
    self.assertTrue(controller.get_plan("wsync-1").is_weight_sync)

    # Simulate Tunix d2h() re-registering source trainer with new ephemeral TCP
    # ports while keeping tensor/mesh topology identical.
    controller.register_work_unit(
        src_unit,
        ["10.0.0.1:18543"],
        control_plane_rpc_address="10.0.0.1:19543",
        mesh_shape=[1, 1],
        variables=vars_metadata,
    )

    # 1. Plan cache MUST be preserved (not invalidated by source ephemeral port
    # change).
    self.assertEqual(controller.get_plan_cache_size(), 1)
    # 2. Previous transfer wsync-1 MUST be cleaned up from _active_transfers
    # upon unit replacement.
    self.assertNotIn("wsync-1", controller._active_transfers)

    # 3. Next transfer reuses cached plan and updates worker_rpc_addresses to
    # new port.
    fut2 = controller.start_transfer(
        src_units=[src_unit],
        dst_units=[dst_unit],
        use_block_chunks=True,
        req_id="wsync-2",
    )
    asyncio.run(fut2.wait())
    self.assertIsNone(fut2._transfer_task)
    plan2 = controller.get_plan("wsync-2")
    self.assertEqual(plan2.worker_rpc_addresses[src_unit], "10.0.0.1:19543")

  def test_completed_transfers_pruned_across_many_steps(self):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=0, worker_rpc_client=client, enable_plan_cache=True
    )
    src_unit = raiden_controller.RaidenId("src", "0", "weights", 0)
    dst_unit = raiden_controller.RaidenId("dst", "0", "weights", 0)
    controller.register_work_unit(
        src_unit,
        ["10.0.0.1:8000"],
        mesh_shape=[1],
        layout=[0],
        global_shape=[16],
        itemsize=4,
    )
    controller.register_work_unit(
        dst_unit,
        ["10.0.1.1:8000"],
        mesh_shape=[1],
        layout=[0],
        global_shape=[16],
        itemsize=4,
    )

    for step in range(60):
      fut = controller.start_transfer(
          src_units=[src_unit],
          dst_units=[dst_unit],
          use_block_chunks=True,
          req_id=f"wsync-{step}",
      )
      asyncio.run(fut.wait())
      self.assertIsNone(fut._transfer_task)
      self.assertEqual(fut.session_id, step)

    # Completed transfers are bounded to <= 16 entries, preventing linear
    # memory/GC growth.
    self.assertLessEqual(len(controller._active_transfers), 16)
    self.assertLessEqual(len(controller._active_tasks), 16)
    self.assertLessEqual(len(controller._task_units), 16)

  def test_group_size_must_be_positive(self):
    controller = raiden_controller.RaidenController(
        port=0, worker_rpc_client=RecordingWorkerRpcClient()
    )
    src_unit = raiden_controller.RaidenId("src", "0", "weights", 0)
    dst_unit = raiden_controller.RaidenId("dst", "0", "weights", 0)

    for invalid_group_size in [0, -1, -128]:
      with self.assertRaisesRegex(ValueError, "group_size must be positive"):
        controller.start_transfer(
            src_units=[src_unit],
            dst_units=[dst_unit],
            use_block_chunks=True,
            group_size=invalid_group_size,
        )

      with self.assertRaisesRegex(ValueError, "group_size must be positive"):
        controller._make_plan_cache_key(
            src_units=[src_unit],
            dst_units=[dst_unit],
            group_size=invalid_group_size,
        )

      with self.assertRaisesRegex(ValueError, "group_size must be positive"):
        asyncio.run(
            controller.warmup_transfer_plan(
                src_units=[src_unit],
                dst_units=[dst_unit],
                group_size=invalid_group_size,
            )
        )

      with self.assertRaisesRegex(ValueError, "group_size must be positive"):
        asyncio.run(
            controller._compute_transfer_schedule(
                src_units=[src_unit],
                dst_units=[dst_unit],
                group_size=invalid_group_size,
            )
        )

  def test_ep_multi_host_dst_endpoint_counts_matches_push_tasks(self):
    """Verifies dst_endpoint_counts exactly matches tasks dispatched for EP and replicated tensors."""
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=0, worker_rpc_client=client, enable_plan_cache=False
    )

    # Source has 1 unit with 4 shards:
    # 1. Sharded EP tensor: [64, 64], mesh [4, 1], layout [1, 0]
    # 2. Replicated tensor: [32, 32], mesh [1, 1], layout [1, 0]
    src_unit = raiden_controller.RaidenId("src", "0", "weights", 0)
    src_var_ep = raiden_service_pb2.VariableMetadataProto(
        name="experts",
        shape=[64, 64],
        mesh_shape=[4, 1],
        layout=[1, 0],
        item_size=4,
        layer_idx=0,
        global_shard_indices=[0, 1, 2, 3],
    )
    src_var_rep = raiden_service_pb2.VariableMetadataProto(
        name="attention",
        shape=[32, 32],
        mesh_shape=[1, 1],
        layout=[1, 0],
        item_size=4,
        layer_idx=1,
        global_shard_indices=[0, 0, 0, 0],
    )
    controller.register_work_unit(
        src_unit,
        ["10.0.0.1:8000", "10.0.0.1:8001", "10.0.0.1:8002", "10.0.0.1:8003"],
        control_plane_rpc_address="10.0.0.1:9000",
        variables=[src_var_ep, src_var_rep],
    )

    # Destination has 2 hosts, 2 shards each, with strided EP global_shard_indices:
    # Host 0 (IP 10.0.1.1): global_shard_indices [0, 2]
    # Host 1 (IP 10.0.1.2): global_shard_indices [1, 3]
    dst_unit_0 = raiden_controller.RaidenId("dst", "0", "weights", 0)
    dst_var_ep_0 = raiden_service_pb2.VariableMetadataProto(
        name="experts",
        shape=[64, 64],
        mesh_shape=[4, 1],
        layout=[1, 0],
        item_size=4,
        layer_idx=0,
        global_shard_indices=[0, 2],
    )
    dst_var_rep_0 = raiden_service_pb2.VariableMetadataProto(
        name="attention",
        shape=[32, 32],
        mesh_shape=[1, 1],
        layout=[1, 0],
        item_size=4,
        layer_idx=1,
        global_shard_indices=[0, 0],
    )
    controller.register_work_unit(
        dst_unit_0,
        ["10.0.1.1:8000", "10.0.1.1:8001"],
        control_plane_rpc_address="10.0.1.1:9000",
        variables=[dst_var_ep_0, dst_var_rep_0],
    )

    dst_unit_1 = raiden_controller.RaidenId("dst", "1", "weights", 0)
    dst_var_ep_1 = raiden_service_pb2.VariableMetadataProto(
        name="experts",
        shape=[64, 64],
        mesh_shape=[4, 1],
        layout=[1, 0],
        item_size=4,
        layer_idx=0,
        global_shard_indices=[1, 3],
    )
    dst_var_rep_1 = raiden_service_pb2.VariableMetadataProto(
        name="attention",
        shape=[32, 32],
        mesh_shape=[1, 1],
        layout=[1, 0],
        item_size=4,
        layer_idx=1,
        global_shard_indices=[0, 0],
    )
    controller.register_work_unit(
        dst_unit_1,
        ["10.0.1.2:8000", "10.0.1.2:8001"],
        control_plane_rpc_address="10.0.1.2:9000",
        variables=[dst_var_ep_1, dst_var_rep_1],
    )

    loop = asyncio.new_event_loop()
    try:
      cached = loop.run_until_complete(
          controller._compute_transfer_schedule(
              src_units=[src_unit],
              dst_units=[dst_unit_0, dst_unit_1],
          )
      )
    finally:
      loop.close()

    self.assertIsNotNone(cached)
    self.assertIn("10.0.1.1", cached.dst_endpoint_counts)
    self.assertIn("10.0.1.2", cached.dst_endpoint_counts)

    # Count tasks actually in direct_schedules targeting each host
    host_tasks = {}
    for s_unit, scheds in cached.direct_schedules.items():
      for shard_idx, entries in scheds.items():
        for entry in entries:
          dst_peer = entry[0]
          dst_host = raiden_controller._extract_host_ip(dst_peer)
          tasks_count = 1
          host_tasks[dst_host] = host_tasks.get(dst_host, 0) + tasks_count

    for host, expected_count in cached.dst_endpoint_counts.items():
      self.assertEqual(
          expected_count,
          host_tasks.get(host, 0),
          f"Mismatch in expected tasks for host {host}",
      )

    # Also verify that when _encode_start_transfer is called with the host's control-plane address,
    # the StartTransferRequest carries the customized expected_block_count.
    ws_client = raiden_controller.WeightSyncWorkerRpcClient()
    try:
      plan = raiden_controller.TransferPlan(
          src_units=[src_unit],
          dst_units=[dst_unit_0, dst_unit_1],
          plan=None,
          shard_push_schedules=cached.direct_schedules,
          worker_data_addresses=cached.data_addresses,
          use_block_chunks=True,
          is_sender=False,
          expected_block_count=cached.expected_block_count,
          dst_expected_block_counts=cached.dst_unit_counts,
          dst_endpoint_counts=cached.dst_endpoint_counts,
          dst_endpoint_layer_counts=cached.dst_endpoint_layer_counts,
          is_weight_sync=True,
      )
      for dst_unit, host in [
          (dst_unit_0, "10.0.1.1"),
          (dst_unit_1, "10.0.1.2"),
      ]:
        encoded = ws_client._encode_start_transfer(
            dst_unit, plan, address=f"{host}:9000"
        )
        req = raiden_service_pb2.ControlRequest()
        req.ParseFromString(encoded)
        self.assertEqual(
            req.start_transfer_request.expected_block_count,
            cached.dst_endpoint_counts[host],
        )
    finally:
      ws_client.close()


class RaidenMultiPeerBroadcastDeduplicationTest(absltest.TestCase):
  """Tests for multi-peer broadcast destination deduplication in ShardPushEntryProto."""

  def test_build_sender_push_schedule_protos_deduplication(self):
    client = raiden_controller.WorkerRpcClient()
    try:
      # 3 destination peers targeting the exact same slicing parameters
      push_schedules = {
          0: [
              ("10.0.0.1:8000", 0, 100, 200, 1024, 1, 2, 64, 64, 16, 5, 0),
              ("10.0.0.2:8000", 0, 100, 200, 1024, 1, 2, 64, 64, 16, 5, 0),
              ("10.0.0.3:8000", 0, 100, 200, 1024, 1, 2, 64, 64, 16, 5, 0),
              # A distinct entry with different slicing parameters
              ("10.0.0.1:8000", 1, 300, 400, 2048, 3, 4, 128, 128, 8, 6, 0),
          ]
      }
      protos = client.build_sender_push_schedule_protos(push_schedules)
      self.assertIn(0, protos)
      sched_proto = protos[0]
      self.assertLen(sched_proto.entries, 2)

      # Check deduplicated broadcast entry
      e0 = sched_proto.entries[0]
      self.assertEqual(e0.dst_peer, "10.0.0.1:8000")
      self.assertEqual(
          list(e0.dst_peers),
          ["10.0.0.1:8000", "10.0.0.2:8000", "10.0.0.3:8000"],
      )
      self.assertEqual(e0.dst_shard_idx, 0)
      self.assertEqual(e0.dst_offset_bytes, 100)
      self.assertEqual(e0.src_offset_bytes, 200)
      self.assertEqual(e0.size_bytes, 1024)
      self.assertEqual(e0.src_block_id, 1)
      self.assertEqual(e0.dst_block_id, 2)
      self.assertEqual(e0.src_stride_bytes, 64)
      self.assertEqual(e0.dst_stride_bytes, 64)
      self.assertEqual(e0.count, 16)
      self.assertEqual(e0.layer_idx, 5)

      # Check distinct entry
      e1 = sched_proto.entries[1]
      self.assertEqual(e1.dst_peer, "10.0.0.1:8000")
      self.assertEqual(list(e1.dst_peers), ["10.0.0.1:8000"])
      self.assertEqual(e1.dst_shard_idx, 1)
      self.assertEqual(e1.size_bytes, 2048)
    finally:
      client.close()

  def test_build_sender_push_schedule_protos_duplicate_peers(self):
    client = raiden_controller.WorkerRpcClient()
    try:
      # Duplicate destination peers should not produce duplicates in dst_peers
      push_schedules = {
          0: [
              ("10.0.0.1:8000", 0, 0, 0, 512, 0, 0, 0, 0, 1, 0, 0),
              ("10.0.0.1:8000", 0, 0, 0, 512, 0, 0, 0, 0, 1, 0, 0),
              ("10.0.0.2:8000", 0, 0, 0, 512, 0, 0, 0, 0, 1, 0, 0),
          ]
      }
      protos = client.build_sender_push_schedule_protos(push_schedules)
      e = protos[0].entries[0]
      self.assertEqual(list(e.dst_peers), ["10.0.0.1:8000", "10.0.0.2:8000"])
      self.assertEqual(e.dst_peer, "10.0.0.1:8000")
    finally:
      client.close()

  def test_build_sender_push_schedule_protos_outer_stride_folding(self):
    client = raiden_controller.WorkerRpcClient()
    try:
      push_schedules = {
          0: [
              (
                  "10.0.0.1:8000",
                  0,
                  i * 524288,
                  i * 65536,
                  8192,
                  i,
                  i,
                  8192,
                  16384,
                  8,
                  2,
                  0,
              )
              for i in range(32)
          ]
      }
      protos = client.build_sender_push_schedule_protos(push_schedules)
      self.assertLen(protos[0].entries, 1)
      e = protos[0].entries[0]
      self.assertEqual(e.dst_offset_bytes, 0)
      self.assertEqual(e.src_offset_bytes, 0)
      self.assertEqual(e.size_bytes, 8192)
      self.assertEqual(e.src_stride_bytes, 8192)
      self.assertEqual(e.dst_stride_bytes, 16384)
      self.assertEqual(e.count, 8)
      self.assertEqual(list(e.outer_counts), [32])
      self.assertEqual(list(e.outer_src_strides_bytes), [65536])
      self.assertEqual(list(e.outer_dst_strides_bytes), [524288])
    finally:
      client.close()

  def test_encode_start_transfer_receiver_filtering_with_dst_peers(self):
    client = raiden_controller.WorkerRpcClient()
    try:
      src_unit = raiden_controller.RaidenId("src_job", "0", "data", 0)
      dst_unit_1 = raiden_controller.RaidenId("dst_job_1", "0", "data", 0)
      dst_unit_2 = raiden_controller.RaidenId("dst_job_2", "0", "data", 0)
      dst_unit_3 = raiden_controller.RaidenId("dst_job_3", "0", "data", 0)

      # Create pre-grouped entry proto with multiple dst_peers
      sched_proto = raiden_service_pb2.ShardPushScheduleProto()
      entry = sched_proto.entries.add()
      entry.dst_peer = "10.0.0.1:8000"
      entry.dst_peers.extend(["10.0.0.1:8000", "10.0.0.2:8000"])
      entry.dst_shard_idx = 0
      entry.dst_offset_bytes = 0
      entry.src_offset_bytes = 0
      entry.size_bytes = 1024
      entry.count = 1

      plan = raiden_controller.TransferPlan(
          src_units=[src_unit],
          dst_units=[dst_unit_1, dst_unit_2, dst_unit_3],
          plan={},
          shard_push_schedules={src_unit: {0: [entry]}},
          worker_data_addresses={
              dst_unit_1: ["10.0.0.1:8000"],
              dst_unit_2: ["10.0.0.2:8000"],
              dst_unit_3: ["10.0.0.3:8000"],
          },
          is_weight_sync=False,
      )

      # Test receiver 1 (matches 10.0.0.1:8000)
      encoded1 = client._encode_start_transfer(
          dst_unit_1, plan, address="10.0.0.1:9000"
      )
      req1 = raiden_service_pb2.ControlRequest()
      req1.ParseFromString(encoded1)
      self.assertLen(
          req1.start_transfer_request.shard_push_schedules[0].entries, 1
      )
      e1 = req1.start_transfer_request.shard_push_schedules[0].entries[0]
      self.assertEqual(e1.dst_peer, "10.0.0.1:8000")
      self.assertEqual(list(e1.dst_peers), ["10.0.0.1:8000"])

      # Test receiver 2 (matches 10.0.0.2:8000)
      encoded2 = client._encode_start_transfer(
          dst_unit_2, plan, address="10.0.0.2:9000"
      )
      req2 = raiden_service_pb2.ControlRequest()
      req2.ParseFromString(encoded2)
      self.assertLen(
          req2.start_transfer_request.shard_push_schedules[0].entries, 1
      )
      e2 = req2.start_transfer_request.shard_push_schedules[0].entries[0]
      self.assertEqual(e2.dst_peer, "10.0.0.2:8000")
      self.assertEqual(list(e2.dst_peers), ["10.0.0.2:8000"])

      # Test receiver 3 (no match, should be empty)
      encoded3 = client._encode_start_transfer(
          dst_unit_3, plan, address="10.0.0.3:9000"
      )
      req3 = raiden_service_pb2.ControlRequest()
      req3.ParseFromString(encoded3)
      self.assertEmpty(req3.start_transfer_request.shard_push_schedules)
    finally:
      client.close()


class SenderScheduleSlicingAndPayloadCachingTest(absltest.TestCase):

  def test_per_worker_schedule_slicing_pathways_multinuma(self):
    """Verifies that in Pathways multi-NUMA mode (2 workers per host, sharing IP with different ports),

    each worker receives ONLY its own local shards in ShardPushScheduleProto.
    """
    client = raiden_controller.WorkerRpcClient()
    try:
      src_unit = raiden_controller.RaidenId("trainer", "0", "weights", 0)
      dst_unit = raiden_controller.RaidenId("rollout", "0", "weights", 0)

      # 2 hosts, 2 workers per host = 4 endpoints, 8 shards (2 shards per worker).
      endpoints = [
          "10.0.0.1:9000",
          "10.0.0.1:9001",
          "10.0.0.2:9000",
          "10.0.0.2:9001",
      ]
      shards = [
          "10.0.0.1:8000",
          "10.0.0.1:8001",
          "10.0.0.1:8002",
          "10.0.0.1:8003",
          "10.0.0.2:8000",
          "10.0.0.2:8001",
          "10.0.0.2:8002",
          "10.0.0.2:8003",
      ]
      for ep in endpoints:
        client.register_worker_endpoint(src_unit, ep)

      # Build 8 shard push schedules
      shard_schedules = {}
      for shard_idx in range(8):
        sched_proto = raiden_service_pb2.ShardPushScheduleProto()
        entry = sched_proto.entries.add()
        entry.dst_peer = "10.0.1.1:8000"
        entry.dst_shard_idx = shard_idx
        entry.size_bytes = 4096
        shard_schedules[shard_idx] = sched_proto

      plan = raiden_controller.TransferPlan(
          src_units=[src_unit],
          dst_units=[dst_unit],
          plan={},
          shard_push_schedules={src_unit: {}},
          worker_data_addresses={
              src_unit: shards,
              dst_unit: ["10.0.1.1:8000"],
          },
          is_sender=True,
          is_weight_sync=True,
          sender_push_schedule_protos={src_unit: shard_schedules},
      )

      # Worker 0 (host 1 port 9000): should own shards [0, 1]
      encoded0 = client._encode_start_transfer(
          src_unit, plan, address="10.0.0.1:9000"
      )
      req0 = raiden_service_pb2.ControlRequest()
      req0.ParseFromString(encoded0)
      self.assertEqual(
          set(req0.start_transfer_request.shard_push_schedules.keys()), {0, 1}
      )

      # Worker 1 (host 1 port 9001): should own shards [2, 3]
      encoded1 = client._encode_start_transfer(
          src_unit, plan, address="10.0.0.1:9001"
      )
      req1 = raiden_service_pb2.ControlRequest()
      req1.ParseFromString(encoded1)
      self.assertEqual(
          set(req1.start_transfer_request.shard_push_schedules.keys()), {2, 3}
      )

      # Worker 2 (host 2 port 9000): should own shards [4, 5]
      encoded2 = client._encode_start_transfer(
          src_unit, plan, address="10.0.0.2:9000"
      )
      req2 = raiden_service_pb2.ControlRequest()
      req2.ParseFromString(encoded2)
      self.assertEqual(
          set(req2.start_transfer_request.shard_push_schedules.keys()), {4, 5}
      )

      # Worker 3 (host 2 port 9001): should own shards [6, 7]
      encoded3 = client._encode_start_transfer(
          src_unit, plan, address="10.0.0.2:9001"
      )
      req3 = raiden_service_pb2.ControlRequest()
      req3.ParseFromString(encoded3)
      self.assertEqual(
          set(req3.start_transfer_request.shard_push_schedules.keys()), {6, 7}
      )
    finally:
      client.close()

  def test_serialize_once_when_payload_invariant(self):
    """Verifies that when payload is invariant across workers, _encode_start_transfer is called once."""

    class CountingWorkerRpcClient(raiden_controller.WorkerRpcClient):

      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encode_count = 0
        self.dispatched = []

      def _encode_start_transfer(self, target_id, transfer_plan, address=None):
        self.encode_count += 1
        return super()._encode_start_transfer(
            target_id, transfer_plan, address=address
        )

      async def _send_rpc(self, addr, payload, timeout=600.0):
        self.dispatched.append((addr, payload))
        resp = raiden_service_pb2.ControlResponse(success=True)
        return resp.SerializeToString()

    client = CountingWorkerRpcClient()
    try:
      src_unit = raiden_controller.RaidenId("trainer", "0", "weights", 0)
      dst_unit = raiden_controller.RaidenId("rollout", "0", "weights", 0)

      # 4 worker endpoints on receiver (no endpoint specialization)
      for i in range(4):
        client.register_worker_endpoint(dst_unit, f"10.0.0.{i+1}:9000")

      plan = raiden_controller.TransferPlan(
          src_units=[src_unit],
          dst_units=[dst_unit],
          plan={},
          worker_data_addresses={
              dst_unit: ["10.0.0.1:8000"],
          },
          is_sender=False,
          expected_block_count=10,
          is_weight_sync=True,
      )

      asyncio.run(client.start_transfer(dst_unit, plan))
      # Invariant across 4 workers: encode MUST be called only 1 time
      self.assertEqual(client.encode_count, 1)
      self.assertEqual(len(client.dispatched), 4)
      # All 4 workers must receive the exact same payload bytes
      for _, payload in client.dispatched:
        self.assertEqual(payload, client.dispatched[0][1])
    finally:
      client.close()

  def test_steady_state_payload_caching_step1_reuses_bytes(self):
    """Verifies that steady-state step 1+ reuses cached serialized bytes without re-serializing."""
    client = raiden_controller.WorkerRpcClient()
    try:
      src_unit = raiden_controller.RaidenId("trainer", "0", "weights", 0)
      dst_unit = raiden_controller.RaidenId("rollout", "0", "weights", 0)

      client.register_worker_endpoint(src_unit, "10.0.0.1:9000")
      client.register_worker_endpoint(src_unit, "10.0.0.1:9001")

      sched0 = raiden_service_pb2.ShardPushScheduleProto()
      sched0.entries.add(
          dst_peer="10.0.1.1:8000", dst_shard_idx=0, size_bytes=1024
      )
      sched1 = raiden_service_pb2.ShardPushScheduleProto()
      sched1.entries.add(
          dst_peer="10.0.1.1:8000", dst_shard_idx=1, size_bytes=1024
      )

      plan = raiden_controller.TransferPlan(
          src_units=[src_unit],
          dst_units=[dst_unit],
          plan={},
          worker_data_addresses={
              src_unit: ["10.0.0.1:8000", "10.0.0.1:8001"],
              dst_unit: ["10.0.1.1:8000"],
          },
          is_sender=True,
          is_weight_sync=True,
          uuid=42,
          req_id="steady_step_0",
          sender_push_schedule_protos={src_unit: {0: sched0, 1: sched1}},
      )

      # Step 0: Initial serialization populates plan.cached_serialized_payloads
      bytes_w0 = client._encode_start_transfer(
          src_unit, plan, address="10.0.0.1:9000"
      )
      bytes_w1 = client._encode_start_transfer(
          src_unit, plan, address="10.0.0.1:9001"
      )
      self.assertTrue(len(plan.cached_serialized_payloads) > 0)

      # Clear sender_push_schedule_protos to prove Step 1 reuses cached bytes
      plan.sender_push_schedule_protos.clear()

      bytes_w0_step1 = client._encode_start_transfer(
          src_unit, plan, address="10.0.0.1:9000"
      )
      bytes_w1_step1 = client._encode_start_transfer(
          src_unit, plan, address="10.0.0.1:9001"
      )

      self.assertEqual(bytes_w0, bytes_w0_step1)
      self.assertEqual(bytes_w1, bytes_w1_step1)
    finally:
      client.close()

  def test_serialize_once_when_payload_invariant_sender_full_schedule(self):
    """Verifies that when sender sends full schedule, encode runs once."""

    class CountingWorkerRpcClient(raiden_controller.WorkerRpcClient):

      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encode_count = 0
        self.dispatched = []

      def _encode_start_transfer(self, target_id, transfer_plan, address=None):
        self.encode_count += 1
        return super()._encode_start_transfer(
            target_id, transfer_plan, address=address
        )

      async def _send_rpc(self, addr, payload, timeout=600.0):
        self.dispatched.append((addr, payload))
        resp = raiden_service_pb2.ControlResponse(success=True)
        return resp.SerializeToString()

    client = CountingWorkerRpcClient()
    try:
      src_unit = raiden_controller.RaidenId("trainer", "0", "weights", 0)
      dst_unit = raiden_controller.RaidenId("rollout", "0", "weights", 0)

      # 4 addresses, but only 1 endpoint registered on target_id (cannot slice,
      # sends full schedule).
      client.register_worker_endpoint(src_unit, "10.0.0.1:9000")

      sched = raiden_service_pb2.ShardPushScheduleProto()
      sched.entries.add(
          dst_peer="10.0.1.1:8000", dst_shard_idx=0, size_bytes=1024
      )

      plan = raiden_controller.TransferPlan(
          src_units=[src_unit],
          dst_units=[dst_unit],
          plan={},
          worker_data_addresses={
              src_unit: ["10.0.0.1:8000"],
              dst_unit: ["10.0.1.1:8000"],
          },
          is_sender=True,
          is_weight_sync=True,
          sender_push_schedule_protos={src_unit: {0: sched}},
      )

      # When address="10.0.0.1:9000, 10.0.0.2:9000" but slicing is inactive,
      # payload is invariant.
      asyncio.run(
          client.start_transfer(
              src_unit, plan, address="10.0.0.1:9000, 10.0.0.2:9000"
          )
      )
      self.assertEqual(client.encode_count, 1)
      self.assertEqual(len(client.dispatched), 2)
      self.assertEqual(client.dispatched[0][1], client.dispatched[1][1])
    finally:
      client.close()

  def test_steady_state_payload_caching_step1_reuses_bytes_across_plans(self):
    """Verifies that across distinct plans, cached bytes are reused."""
    client = raiden_controller.WorkerRpcClient()
    try:
      src_unit = raiden_controller.RaidenId("trainer", "0", "weights", 0)
      dst_unit = raiden_controller.RaidenId("rollout", "0", "weights", 0)

      client.register_worker_endpoint(src_unit, "10.0.0.1:9000")
      client.register_worker_endpoint(src_unit, "10.0.0.1:9001")

      sched0 = raiden_service_pb2.ShardPushScheduleProto()
      sched0.entries.add(
          dst_peer="10.0.1.1:8000", dst_shard_idx=0, size_bytes=1024
      )
      sched1 = raiden_service_pb2.ShardPushScheduleProto()
      sched1.entries.add(
          dst_peer="10.0.1.1:8000", dst_shard_idx=1, size_bytes=1024
      )

      shared_payload_cache = {}
      plan_step0 = raiden_controller.TransferPlan(
          src_units=[src_unit],
          dst_units=[dst_unit],
          plan={},
          worker_data_addresses={
              src_unit: ["10.0.0.1:8000", "10.0.0.1:8001"],
              dst_unit: ["10.0.1.1:8000"],
          },
          is_sender=True,
          is_weight_sync=True,
          skip_d2h=True,
          uuid=100,
          req_id="step_0",
          sender_push_schedule_protos={src_unit: {0: sched0, 1: sched1}},
          cached_serialized_payloads=shared_payload_cache,
      )

      bytes_w0_step0 = client._encode_start_transfer(
          src_unit, plan_step0, address="10.0.0.1:9000"
      )
      bytes_w1_step0 = client._encode_start_transfer(
          src_unit, plan_step0, address="10.0.0.1:9001"
      )
      self.assertNotEmpty(shared_payload_cache)

      # Step 1: Newly instantiated TransferPlan with different req_id and uuid
      plan_step1 = raiden_controller.TransferPlan(
          src_units=[src_unit],
          dst_units=[dst_unit],
          plan={},
          worker_data_addresses={
              src_unit: ["10.0.0.1:8000", "10.0.0.1:8001"],
              dst_unit: ["10.0.1.1:8000"],
          },
          is_sender=True,
          is_weight_sync=True,
          skip_d2h=True,
          uuid=101,
          req_id="step_1",
          # Empty: would fail/empty if re-serialized
          sender_push_schedule_protos={},
          cached_serialized_payloads=shared_payload_cache,
      )

      bytes_w0_step1 = client._encode_start_transfer(
          src_unit, plan_step1, address="10.0.0.1:9000"
      )
      bytes_w1_step1 = client._encode_start_transfer(
          src_unit, plan_step1, address="10.0.0.1:9001"
      )

      req_w0_step0 = raiden_service_pb2.ControlRequest()
      req_w0_step0.ParseFromString(bytes_w0_step0)
      req_w0_step1 = raiden_service_pb2.ControlRequest()
      req_w0_step1.ParseFromString(bytes_w0_step1)
      self.assertEqual(req_w0_step1.start_transfer_request.uuid, 101)
      self.assertEqual(req_w0_step1.start_transfer_request.req_id, "step_1")
      self.assertEqual(
          req_w0_step1.start_transfer_request.shard_push_schedules,
          req_w0_step0.start_transfer_request.shard_push_schedules,
      )

      req_w1_step0 = raiden_service_pb2.ControlRequest()
      req_w1_step0.ParseFromString(bytes_w1_step0)
      req_w1_step1 = raiden_service_pb2.ControlRequest()
      req_w1_step1.ParseFromString(bytes_w1_step1)
      self.assertEqual(req_w1_step1.start_transfer_request.uuid, 101)
      self.assertEqual(
          req_w1_step1.start_transfer_request.shard_push_schedules,
          req_w1_step0.start_transfer_request.shard_push_schedules,
      )

      # Step 2: When uuid and skip_d2h match Step 0 (only req_id differs),
      # exact serialized bytes are returned without re-encoding.
      plan_step2 = raiden_controller.TransferPlan(
          src_units=[src_unit],
          dst_units=[dst_unit],
          plan={},
          worker_data_addresses={
              src_unit: ["10.0.0.1:8000", "10.0.0.1:8001"],
              dst_unit: ["10.0.1.1:8000"],
          },
          is_sender=True,
          is_weight_sync=True,
          skip_d2h=True,
          uuid=100,
          req_id="step_2",
          sender_push_schedule_protos={},
          cached_serialized_payloads=shared_payload_cache,
      )
      bytes_w0_step2 = client._encode_start_transfer(
          src_unit, plan_step2, address="10.0.0.1:9000"
      )
      bytes_w1_step2 = client._encode_start_transfer(
          src_unit, plan_step2, address="10.0.0.1:9001"
      )
      self.assertEqual(bytes_w0_step0, bytes_w0_step2)
      self.assertEqual(bytes_w1_step0, bytes_w1_step2)
    finally:
      client.close()

  def test_per_worker_schedule_slicing_uneven_shards(self):
    """Verifies that uneven shard-to-worker division accounts for all shards."""
    client = raiden_controller.WorkerRpcClient()
    try:
      src_unit = raiden_controller.RaidenId("trainer", "0", "weights", 0)
      dst_unit = raiden_controller.RaidenId("rollout", "0", "weights", 0)

      # 2 workers, 5 shards
      client.register_worker_endpoint(src_unit, "10.0.0.1:9000")
      client.register_worker_endpoint(src_unit, "10.0.0.1:9001")

      schedules = {}
      for s in range(5):
        sched = raiden_service_pb2.ShardPushScheduleProto()
        sched.entries.add(
            dst_peer="10.0.1.1:8000", dst_shard_idx=s, size_bytes=1024
        )
        schedules[s] = sched

      plan = raiden_controller.TransferPlan(
          src_units=[src_unit],
          dst_units=[dst_unit],
          plan={},
          worker_data_addresses={
              src_unit: [f"10.0.0.1:800{s}" for s in range(5)],
              dst_unit: ["10.0.1.1:8000"],
          },
          is_sender=True,
          is_weight_sync=True,
          sender_push_schedule_protos={src_unit: schedules},
      )

      owned0 = client._get_worker_owned_shards(
          src_unit, plan, address="10.0.0.1:9000"
      )
      owned1 = client._get_worker_owned_shards(
          src_unit, plan, address="10.0.0.1:9001"
      )

      self.assertEqual(owned0, {0, 1})
      self.assertEqual(owned1, {2, 3, 4})
      self.assertEqual(owned0 | owned1, set(range(5)))
      self.assertEqual(len(owned0 & owned1), 0)
    finally:
      client.close()

  def test_per_worker_schedule_slicing_more_workers_than_shards(self):
    """Verifies slicing when there are more workers than shards."""
    client = raiden_controller.WorkerRpcClient()
    try:
      src_unit = raiden_controller.RaidenId("trainer", "0", "weights", 0)
      dst_unit = raiden_controller.RaidenId("rollout", "0", "weights", 0)

      # 4 workers, 2 shards
      for i in range(4):
        client.register_worker_endpoint(src_unit, f"10.0.0.1:900{i}")

      plan = raiden_controller.TransferPlan(
          src_units=[src_unit],
          dst_units=[dst_unit],
          plan={},
          worker_data_addresses={
              src_unit: ["10.0.0.1:8000", "10.0.0.1:8001"],
              dst_unit: ["10.0.1.1:8000"],
          },
          is_sender=True,
          is_weight_sync=True,
      )

      owned0 = client._get_worker_owned_shards(
          src_unit, plan, address="10.0.0.1:9000"
      )
      owned1 = client._get_worker_owned_shards(
          src_unit, plan, address="10.0.0.1:9001"
      )
      owned2 = client._get_worker_owned_shards(
          src_unit, plan, address="10.0.0.1:9002"
      )
      owned3 = client._get_worker_owned_shards(
          src_unit, plan, address="10.0.0.1:9003"
      )

      self.assertEqual(owned0, {0})
      self.assertEqual(owned1, {1})
      self.assertEqual(owned2, set())
      self.assertEqual(owned3, set())
    finally:
      client.close()

  def test_single_address_dispatch_preserves_slicing(self):
    """Verifies that start_transfer with a single address preserves slicing."""

    class SingleAddrClient(raiden_controller.WorkerRpcClient):

      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dispatched_reqs = []

      async def _send_rpc(self, addr, payload, timeout=600.0):
        req = raiden_service_pb2.ControlRequest()
        req.ParseFromString(payload)
        self.dispatched_reqs.append((addr, req))
        resp = raiden_service_pb2.ControlResponse(success=True)
        return resp.SerializeToString()

    client = SingleAddrClient()
    try:
      src_unit = raiden_controller.RaidenId("trainer", "0", "weights", 0)
      dst_unit = raiden_controller.RaidenId("rollout", "0", "weights", 0)

      # 2 workers registered on src_unit
      client.register_worker_endpoint(src_unit, "10.0.0.1:9000")
      client.register_worker_endpoint(src_unit, "10.0.0.1:9001")

      s0 = raiden_service_pb2.ShardPushScheduleProto()
      s0.entries.add(dst_peer="10.0.1.1:8000", dst_shard_idx=0, size_bytes=1024)
      s1 = raiden_service_pb2.ShardPushScheduleProto()
      s1.entries.add(dst_peer="10.0.1.1:8000", dst_shard_idx=1, size_bytes=1024)

      plan = raiden_controller.TransferPlan(
          src_units=[src_unit],
          dst_units=[dst_unit],
          plan={},
          worker_data_addresses={
              src_unit: ["10.0.0.1:8000", "10.0.0.1:8001"],
              dst_unit: ["10.0.1.1:8000"],
          },
          is_sender=True,
          is_weight_sync=True,
          sender_push_schedule_protos={src_unit: {0: s0, 1: s1}},
      )

      # Dispatch to single address directly
      asyncio.run(
          client.start_transfer(src_unit, plan, address="10.0.0.1:9001")
      )
      self.assertEqual(len(client.dispatched_reqs), 1)
      addr, req = client.dispatched_reqs[0]
      self.assertEqual(addr, "10.0.0.1:9001")
      # Worker 1 must only receive shard 1, not shard 0
      self.assertEqual(
          set(req.start_transfer_request.shard_push_schedules.keys()), {1}
      )
    finally:
      client.close()

  def test_endpoint_to_shards_direct_address_key(self):
    """Verifies endpoint_to_shards when keyed directly by address string."""
    client = raiden_controller.WorkerRpcClient()
    try:
      src_unit = raiden_controller.RaidenId("trainer", "0", "weights", 0)
      dst_unit = raiden_controller.RaidenId("rollout", "0", "weights", 0)

      plan = raiden_controller.TransferPlan(
          src_units=[src_unit],
          dst_units=[dst_unit],
          plan={},
          is_sender=True,
          is_weight_sync=True,
          endpoint_to_shards={"10.0.0.1:9000": [3, 7]},
      )

      owned = client._get_worker_owned_shards(
          src_unit, plan, address="10.0.0.1:9000"
      )
      self.assertEqual(owned, {3, 7})
    finally:
      client.close()


class JobEntityTest(absltest.TestCase):
  """Tests for multi-host JobEntity / HostGroup abstraction and ControlPipeClient ownership."""

  def test_eleven_entities_multi_host_delegation(self):
    """Verifies 1 trainer (10 hosts) + 10 samplers (4 hosts each) = 11 entities in RaidenController."""
    controller = raiden_controller.RaidenController(port=0)
    try:
      # 1 Trainer job (data_replica_idx=0) with 10 hosts
      # (job_replica_id="0".."9").
      trainer_host_units = [
          raiden_controller.RaidenId(
              job_name="trainer",
              job_replica_id=str(h),
              data_name="weights",
              data_replica_idx=0,
          )
          for h in range(10)
      ]
      trainer_hosts = [f"10.0.0.{h}:9000" for h in range(10)]
      trainer_shards: list[str] = []
      for h, host_unit in enumerate(trainer_host_units):
        host_shards = [f"10.0.0.{h}:{8000 + local_i}" for local_i in range(4)]
        trainer_shards.extend(host_shards)
        controller.register_work_unit(
            host_unit,
            shards=host_shards,
            control_plane_rpc_address=trainer_hosts[h],
            mesh_shape=[40],
            layout=[0],
            global_shape=[400],
            itemsize=4,
        )

      # 10 Sampler jobs (job_name="sampler_0".."sampler_9"), each with 4
      # attached hosts (job_replica_id="0".."3", 4 shards/host = 16 shards/job).
      sampler_representative_units = []
      for s_idx in range(10):
        sampler_host_units = [
            raiden_controller.RaidenId(
                job_name=f"sampler_{s_idx}",
                job_replica_id=str(h),
                data_name="weights",
                data_replica_idx=0,
            )
            for h in range(4)
        ]
        sampler_representative_units.append(sampler_host_units[0])
        for h, host_unit in enumerate(sampler_host_units):
          host_shards = [
              f"10.1.{s_idx}.{h}:{8000 + local_i}" for local_i in range(4)
          ]
          controller.register_work_unit(
              host_unit,
              shards=host_shards,
              control_plane_rpc_address=f"10.1.{s_idx}.{h}:9000",
              mesh_shape=[16],
              layout=[0],
              global_shape=[400],
              itemsize=4,
          )

      # RaidenController manages 11 entities (1 trainer + 10 samplers),
      # indexed by RaidenId(job_name=...).
      entities = controller.entities
      self.assertLen(entities, 11)
      self.assertIn(raiden_controller.RaidenId(job_name="trainer"), entities)
      self.assertIn(raiden_controller.RaidenId(job_name="sampler_1"), entities)

      trainer_entity = controller.get_entity(
          raiden_controller.RaidenId(job_name="trainer")
      )
      self.assertIsNotNone(trainer_entity)
      self.assertIsInstance(trainer_entity, raiden_controller.JobEntity)
      self.assertEqual(trainer_entity.num_hosts, 10)
      self.assertLen(trainer_entity.hosts, 10)
      self.assertEqual(trainer_entity.host_endpoints, trainer_hosts)
      self.assertIsNotNone(trainer_entity.worker_rpc_client)
      self.assertIsNotNone(trainer_entity.control_pipe_client)
      self.assertIs(
          trainer_entity.control_pipe_client,
          trainer_entity.worker_rpc_client.control_pipe_client,
      )

      # Verify each trainer host owns 4 distinct shards (40 shards / 10 hosts)
      for host_idx, host_desc in enumerate(trainer_entity.hosts):
        self.assertEqual(host_desc.job_replica_id, str(host_idx))
        self.assertEqual(host_desc.control_address, trainer_hosts[host_idx])
        self.assertLen(host_desc.shards, 4)

      for s_idx, sampler_unit in enumerate(sampler_representative_units):
        sampler_entity = controller.get_entity(
            raiden_controller.RaidenId(job_name=f"sampler_{s_idx}")
        )
        self.assertIs(sampler_entity, controller.get_entity(sampler_unit))
        self.assertIsNotNone(sampler_entity)
        self.assertEqual(sampler_entity.num_hosts, 4)
        self.assertLen(sampler_entity.hosts, 4)
        self.assertIsNotNone(sampler_entity.worker_rpc_client)
        self.assertIsNotNone(sampler_entity.control_pipe_client)
        self.assertIs(
            sampler_entity.control_pipe_client,
            sampler_entity.worker_rpc_client.control_pipe_client,
        )
        for host_idx, host_desc in enumerate(sampler_entity.hosts):
          self.assertEqual(host_desc.job_replica_id, str(host_idx))
          self.assertLen(host_desc.shards, 4)

      # Verify transfer command dispatch via JobEntity's owned ControlPipeClient
      dispatched_by_entity: dict[raiden_controller.RaidenId, list[str]] = {}
      ok_resp = raiden_service_pb2.ControlResponse(
          success=True
      ).SerializeToString()

      for unit_id, ent in entities.items():
        dispatched_by_entity[unit_id] = []

        def make_fake_sync(u):
          def fake_send_sync(addr, payload, timeout=600.0, message_type=""):
            del payload, timeout, message_type
            dispatched_by_entity[u].append(addr)
            return ok_resp

          return fake_send_sync

        ent.control_pipe_client.send_raw_bytes_sync = make_fake_sync(unit_id)

      # Dispatch transfer across all 10 hosts of trainer_entity via entity key
      trainer_key = trainer_entity.unit
      sampler0_key = controller.get_entity(sampler_representative_units[0]).unit
      s0_plan = raiden_controller.TransferPlan(
          src_units=[trainer_key],
          dst_units=[sampler0_key],
          plan={},
          worker_data_addresses={
              trainer_key: trainer_shards,
              sampler0_key: controller.get_entity(sampler0_key).shards,
          },
          is_sender=True,
          is_weight_sync=True,
      )
      asyncio.run(trainer_entity.start_transfer(s0_plan))
      self.assertCountEqual(dispatched_by_entity[trainer_key], trainer_hosts)
    finally:
      controller.worker_rpc_client.close()

  def test_incremental_attach_host_on_entity(self):
    """Verifies incremental host attachment via controller.attach_host."""
    controller = raiden_controller.RaidenController(port=0)
    try:
      for h in range(4):
        host_unit = raiden_controller.RaidenId(
            job_name="sampler",
            job_replica_id=str(h),
            data_name="weights",
            data_replica_idx=0,
        )
        controller.attach_host(
            host_unit,
            control_address=f"10.2.0.{h}:9000",
            shards=[f"10.2.0.{h}:8000", f"10.2.0.{h}:8001"],
        )

      sampler_key = raiden_controller.RaidenId("sampler", "", "weights", 0)
      entity = controller.get_entity(sampler_key)
      self.assertIsNotNone(entity)
      self.assertLen(controller.entities, 1)
      self.assertEqual(entity.num_hosts, 4)
      self.assertLen(entity.shards, 8)
    finally:
      controller.worker_rpc_client.close()

  def test_dst_indices_hoisted_and_computed_once_per_variable(self):
    """Verifies dst_indices is computed once per (dst_unit, var_name) instead of per source shard."""
    controller = raiden_controller.RaidenController(
        port=0, enable_plan_cache=False
    )
    try:
      num_src_shards = 16
      src_unit = raiden_controller.RaidenId("src", "0", "weights", 0)
      v0 = raiden_service_pb2.VariableMetadataProto(
          name="var0",
          shape=[64, 64],
          mesh_shape=[num_src_shards, 1],
          layout=[1, 0],
          item_size=4,
          layer_idx=0,
          global_shard_indices=list(range(num_src_shards)),
      )
      v1 = raiden_service_pb2.VariableMetadataProto(
          name="var1",
          shape=[32, 32],
          mesh_shape=[num_src_shards, 1],
          layout=[1, 0],
          item_size=4,
          layer_idx=1,
          global_shard_indices=list(range(num_src_shards)),
      )
      src_shards = [f"10.0.0.1:{8000 + i}" for i in range(num_src_shards)]
      controller.register_work_unit(
          src_unit,
          src_shards,
          control_plane_rpc_address="10.0.0.1:9000",
          variables=[v0, v1],
      )

      dst_unit_0 = raiden_controller.RaidenId("dst", "0", "weights", 0)
      dst_v0_0 = raiden_service_pb2.VariableMetadataProto(
          name="var0",
          shape=[64, 64],
          mesh_shape=[8, 1],
          layout=[1, 0],
          item_size=4,
          layer_idx=0,
          global_shard_indices=[0, 1, 2, 3],
      )
      dst_v1_0 = raiden_service_pb2.VariableMetadataProto(
          name="var1",
          shape=[32, 32],
          mesh_shape=[8, 1],
          layout=[1, 0],
          item_size=4,
          layer_idx=1,
          global_shard_indices=[0, 1, 2, 3],
      )
      controller.register_work_unit(
          dst_unit_0,
          [f"10.0.1.1:{8000 + i}" for i in range(4)],
          control_plane_rpc_address="10.0.1.1:9000",
          variables=[dst_v0_0, dst_v1_0],
      )

      dst_unit_1 = raiden_controller.RaidenId("dst", "1", "weights", 0)
      dst_v0_1 = raiden_service_pb2.VariableMetadataProto(
          name="var0",
          shape=[64, 64],
          mesh_shape=[8, 1],
          layout=[1, 0],
          item_size=4,
          layer_idx=0,
          global_shard_indices=[4, 5, 6, 7],
      )
      dst_v1_1 = raiden_service_pb2.VariableMetadataProto(
          name="var1",
          shape=[32, 32],
          mesh_shape=[8, 1],
          layout=[1, 0],
          item_size=4,
          layer_idx=1,
          global_shard_indices=[4, 5, 6, 7],
      )
      controller.register_work_unit(
          dst_unit_1,
          [f"10.0.1.2:{8000 + i}" for i in range(4)],
          control_plane_rpc_address="10.0.1.2:9000",
          variables=[dst_v0_1, dst_v1_1],
      )

      original_get_global_indices = (
          raiden_controller.reshard_planner._get_global_indices
      )
      call_units = []

      def counting_get_global_indices(unit, *args, **kwargs):
        call_units.append(unit)
        return original_get_global_indices(unit, *args, **kwargs)

      with mock.patch.object(
          raiden_controller.reshard_planner,
          "_get_global_indices",
          side_effect=counting_get_global_indices,
      ):
        loop = asyncio.new_event_loop()
        try:
          schedule = loop.run_until_complete(
              controller._compute_transfer_schedule(
                  src_units=[src_unit],
                  dst_units=[dst_unit_0, dst_unit_1],
              )
          )
        finally:
          loop.close()

      self.assertIsNotNone(schedule)
      dst_calls = [u for u in call_units if u in (dst_unit_0, dst_unit_1)]
      self.assertLen(dst_calls, 4)
      self.assertIn(src_unit, schedule.computed_schedules)
      self.assertLen(schedule.computed_schedules[src_unit], num_src_shards)
      total_entries = sum(
          len(entries)
          for entries in schedule.computed_schedules[src_unit].values()
      )
      self.assertGreater(total_entries, 0)
    finally:
      controller.worker_rpc_client.close()

  def test_dst_indices_cached_across_multiple_source_units(self):
    """Verifies dst_indices is cached across multiple source units."""
    controller = raiden_controller.RaidenController(
        port=0, enable_plan_cache=False
    )
    try:
      num_src_shards = 8
      src_unit_0 = raiden_controller.RaidenId("src", "0", "weights", 0)
      src_unit_1 = raiden_controller.RaidenId("src", "1", "weights", 0)
      v0_0 = raiden_service_pb2.VariableMetadataProto(
          name="var0",
          shape=[64, 64],
          mesh_shape=[16, 1],
          layout=[1, 0],
          item_size=4,
          layer_idx=0,
          global_shard_indices=list(range(0, 8)),
      )
      v0_1 = raiden_service_pb2.VariableMetadataProto(
          name="var0",
          shape=[64, 64],
          mesh_shape=[16, 1],
          layout=[1, 0],
          item_size=4,
          layer_idx=0,
          global_shard_indices=list(range(8, 16)),
      )
      controller.register_work_unit(
          src_unit_0,
          [f"10.0.0.1:{8000 + i}" for i in range(num_src_shards)],
          control_plane_rpc_address="10.0.0.1:9000",
          variables=[v0_0],
      )
      controller.register_work_unit(
          src_unit_1,
          [f"10.0.0.2:{8000 + i}" for i in range(num_src_shards)],
          control_plane_rpc_address="10.0.0.2:9000",
          variables=[v0_1],
      )

      dst_unit_0 = raiden_controller.RaidenId("dst", "0", "weights", 0)
      dst_v0_0 = raiden_service_pb2.VariableMetadataProto(
          name="var0",
          shape=[64, 64],
          mesh_shape=[8, 1],
          layout=[1, 0],
          item_size=4,
          layer_idx=0,
          global_shard_indices=[0, 1, 2, 3],
      )
      controller.register_work_unit(
          dst_unit_0,
          [f"10.0.1.1:{8000 + i}" for i in range(4)],
          control_plane_rpc_address="10.0.1.1:9000",
          variables=[dst_v0_0],
      )

      dst_unit_1 = raiden_controller.RaidenId("dst", "1", "weights", 0)
      dst_v0_1 = raiden_service_pb2.VariableMetadataProto(
          name="var0",
          shape=[64, 64],
          mesh_shape=[8, 1],
          layout=[1, 0],
          item_size=4,
          layer_idx=0,
          global_shard_indices=[4, 5, 6, 7],
      )
      controller.register_work_unit(
          dst_unit_1,
          [f"10.0.1.2:{8000 + i}" for i in range(4)],
          control_plane_rpc_address="10.0.1.2:9000",
          variables=[dst_v0_1],
      )

      original_get_global_indices = (
          raiden_controller.reshard_planner._get_global_indices
      )
      call_units = []

      def counting_get_global_indices(unit, *args, **kwargs):
        call_units.append(unit)
        return original_get_global_indices(unit, *args, **kwargs)

      with mock.patch.object(
          raiden_controller.reshard_planner,
          "_get_global_indices",
          side_effect=counting_get_global_indices,
      ):
        loop = asyncio.new_event_loop()
        try:
          schedule = loop.run_until_complete(
              controller._compute_transfer_schedule(
                  src_units=[src_unit_0, src_unit_1],
                  dst_units=[dst_unit_0, dst_unit_1],
              )
          )
        finally:
          loop.close()

      self.assertIsNotNone(schedule)
      dst_calls = [u for u in call_units if u in (dst_unit_0, dst_unit_1)]
      # Exactly 2 calls (1 for dst_unit_0, 1 for dst_unit_1 for var0) across
      # both src units!
      self.assertLen(dst_calls, 2)
      self.assertIn(src_unit_0, schedule.computed_schedules)
      self.assertIn(src_unit_1, schedule.computed_schedules)
      self.assertLen(schedule.computed_schedules[src_unit_0], num_src_shards)
      self.assertLen(schedule.computed_schedules[src_unit_1], num_src_shards)
    finally:
      controller.worker_rpc_client.close()

  def test_variable_signature_deduplication_across_layers(self):
    """Verifies identical-shape variables across layers reuse schedule templates."""
    controller = raiden_controller.RaidenController(
        port=0, enable_plan_cache=False
    )
    try:
      num_src_shards = 8
      num_layers = 10
      src_unit = raiden_controller.RaidenId("src", "0", "weights", 0)
      dst_unit = raiden_controller.RaidenId("dst", "0", "weights", 0)

      src_vars = [
          raiden_service_pb2.VariableMetadataProto(
              name=f"layer_{l}_weight",
              shape=[64, 64],
              mesh_shape=[8, 1],
              layout=[1, 0],
              item_size=4,
              layer_idx=l,
              global_shard_indices=list(range(num_src_shards)),
          )
          for l in range(num_layers)
      ]
      dst_vars = [
          raiden_service_pb2.VariableMetadataProto(
              name=f"layer_{l}_weight",
              shape=[64, 64],
              mesh_shape=[4, 1],
              layout=[1, 0],
              item_size=4,
              layer_idx=l,
              global_shard_indices=[0, 1, 2, 3],
          )
          for l in range(num_layers)
      ]

      controller.register_work_unit(
          src_unit,
          [f"10.0.0.1:{8000 + i}" for i in range(num_src_shards)],
          control_plane_rpc_address="10.0.0.1:9000",
          variables=src_vars,
      )
      controller.register_work_unit(
          dst_unit,
          [f"10.0.1.1:{8000 + i}" for i in range(4)],
          control_plane_rpc_address="10.0.1.1:9000",
          variables=dst_vars,
      )

      original_get_global_indices = (
          raiden_controller.reshard_planner._get_global_indices
      )
      call_units = []

      def counting_get_global_indices(unit, *args, **kwargs):
        call_units.append(unit)
        return original_get_global_indices(unit, *args, **kwargs)

      with mock.patch.object(
          raiden_controller.reshard_planner,
          "_get_global_indices",
          side_effect=counting_get_global_indices,
      ):
        loop = asyncio.new_event_loop()
        try:
          schedule = loop.run_until_complete(
              controller._compute_transfer_schedule(
                  src_units=[src_unit],
                  dst_units=[dst_unit],
              )
          )
        finally:
          loop.close()

      # Despite 10 layers, _get_global_indices is called only once for src_unit
      # and once for dst_unit because all 10 layers share the same variable
      # signature, and only 1 unique plan_id is stored in variable_plans.
      self.assertEqual(call_units, [src_unit, dst_unit])
      self.assertLen(schedule.variable_plans[src_unit], 1)
      self.assertEqual(
          schedule.variable_to_plan_id[src_unit],
          {i: 0 for i in range(num_layers)},
      )
      src_ent = controller.get_or_create_entity(src_unit)
      protos = src_ent.build_sender_push_schedule_protos(
          schedule.computed_schedules[src_unit]
      )
      self.assertEqual(
          sorted({e.layer_idx for e in protos[0].entries}),
          list(range(num_layers)),
      )
      for local_idx in range(num_src_shards):
        entries = schedule.computed_schedules[src_unit][local_idx]
        layers_seen = sorted({e[10] for e in entries})
        self.assertEqual(layers_seen, list(range(num_layers)))
    finally:
      controller.worker_rpc_client.close()

  def test_controller_offline_plan_save_and_load_transfer(self):
    """Verifies offline plan generation, parallel worker plan loading, and step-0 transfer without online reshard math."""
    offline_controller = raiden_controller.RaidenController(
        port=0, enable_plan_cache=False
    )
    src_units = [
        raiden_controller.RaidenId("trainer", str(i), "weights", 0)
        for i in range(2)
    ]
    dst_units = [
        raiden_controller.RaidenId("sampler", str(j), "weights", 0)
        for j in range(2)
    ]
    src_vars = [
        raiden_service_pb2.VariableMetadataProto(
            name="layer_0_w",
            shape=[64, 64],
            mesh_shape=[2, 4],
            layout=[1, 0],
            item_size=2,
            layer_idx=0,
            sharding_spec=["fsdp", "tp"],
        ),
        raiden_service_pb2.VariableMetadataProto(
            name="layer_1_w",
            shape=[64, 64],
            mesh_shape=[2, 4],
            layout=[1, 0],
            item_size=2,
            layer_idx=1,
            sharding_spec=["fsdp", "tp"],
        ),
    ]
    dst_vars = [
        raiden_service_pb2.VariableMetadataProto(
            name="layer_0_w",
            shape=[64, 64],
            mesh_shape=[1, 4],
            layout=[1, 0],
            item_size=2,
            layer_idx=0,
            sharding_spec=["fsdp", "tp"],
        ),
        raiden_service_pb2.VariableMetadataProto(
            name="layer_1_w",
            shape=[64, 64],
            mesh_shape=[1, 4],
            layout=[1, 0],
            item_size=2,
            layer_idx=1,
            sharding_spec=["fsdp", "tp"],
        ),
    ]
    try:
      # Register with placeholder offline addresses
      for i, u in enumerate(src_units):
        offline_controller.register_work_unit(
            u,
            [f"0.0.0.{i + 1}:{8000 + d}" for d in range(4)],
            control_plane_rpc_address=f"0.0.0.{i + 1}:9000",
            variables=src_vars,
            mesh_shape=[2, 4],
            mesh_axes=["fsdp", "tp"],
        )
      for j, u in enumerate(dst_units):
        offline_controller.register_work_unit(
            u,
            [f"0.0.1.{j + 1}:{8000 + d}" for d in range(4)],
            control_plane_rpc_address=f"0.0.1.{j + 1}:9000",
            variables=dst_vars,
            mesh_shape=[1, 4],
            mesh_axes=["fsdp", "tp"],
        )

      plan_dir = self.create_tempdir().full_path
      offline_controller.save_offline_plan(plan_dir, src_units, dst_units)
    finally:
      offline_controller.worker_rpc_client.close()

    # Start a runtime controller configured with offline_plan_path and live IPs.
    client = RecordingWorkerRpcClient()
    runtime_controller = raiden_controller.RaidenController(
        port=0,
        worker_rpc_client=client,
        offline_plan_path=plan_dir,
    )
    try:
      for i, u in enumerate(src_units):
        runtime_controller.register_work_unit(
            u,
            [f"10.20.0.{i + 1}:{18000 + d}" for d in range(4)],
            control_plane_rpc_address=f"10.20.0.{i + 1}:19000",
            variables=src_vars,
            mesh_shape=[2, 4],
            mesh_axes=["fsdp", "tp"],
        )
      for j, u in enumerate(dst_units):
        runtime_controller.register_work_unit(
            u,
            [f"10.30.0.{j + 1}:{28000 + d}" for d in range(4)],
            control_plane_rpc_address=f"10.30.0.{j + 1}:29000",
            variables=dst_vars,
            mesh_shape=[1, 4],
            mesh_axes=["fsdp", "tp"],
        )

      # Verify parallel worker loading binds the new live 10.30.0.x endpoints
      worker_plans = runtime_controller.load_offline_worker_plans_parallel(
          plan_dir, src_units, req_id="offline_req", uuid=999
      )
      self.assertLen(worker_plans, 2)
      for u in src_units:
        for (
            _,
            sched_proto,
        ) in worker_plans[
            u
        ].start_transfer_request.shard_push_schedules.items():
          for entry in sched_proto.entries:
            self.assertTrue(entry.dst_peer.startswith("10.30.0."))

      # Verify start_transfer uses the offline plan without calling
      # compute_transfer_schedule_from_metadata.
      with mock.patch.object(
          runtime_controller._planner,
          "compute_transfer_schedule_from_metadata",
          side_effect=AssertionError("Should not run online planning math!"),
      ):
        fut = runtime_controller.start_transfer(
            src_units=src_units,
            dst_units=dst_units,
            use_block_chunks=True,
            req_id="step0_offline",
        )
        asyncio.run(fut.wait())

      plan = runtime_controller.get_plan("step0_offline")
      self.assertIsNotNone(plan)
      for u in src_units:
        for _, entries in plan.shard_push_schedules[u].items():
          for entry in entries:
            self.assertTrue(entry[0].startswith("10.30.0."))
    finally:
      runtime_controller.worker_rpc_client.close()

  def test_controller_parallel_worker_planning_transfer(self):
    """Verifies RaidenController with parallel_worker_planning=True produces identical plans to sequential planning."""
    src_units = [
        raiden_controller.RaidenId("trainer", str(i), "weights", 0)
        for i in range(2)
    ]
    dst_units = [
        raiden_controller.RaidenId("sampler", str(j), "weights", 0)
        for j in range(2)
    ]
    src_vars = [
        raiden_service_pb2.VariableMetadataProto(
            name="layer_0_w",
            shape=[64, 64],
            mesh_shape=[2, 4],
            layout=[1, 0],
            item_size=2,
            layer_idx=0,
            sharding_spec=["fsdp", "tp"],
        ),
    ]
    dst_vars = [
        raiden_service_pb2.VariableMetadataProto(
            name="layer_0_w",
            shape=[64, 64],
            mesh_shape=[1, 4],
            layout=[1, 0],
            item_size=2,
            layer_idx=0,
            sharding_spec=["fsdp", "tp"],
        ),
    ]

    plans = []
    for parallel_flag in (False, True):
      client = RecordingWorkerRpcClient()
      ctrl = raiden_controller.RaidenController(
          port=0,
          worker_rpc_client=client,
          enable_plan_cache=False,
          parallel_worker_planning=parallel_flag,
      )
      try:
        for i, u in enumerate(src_units):
          ctrl.register_work_unit(
              u,
              [f"10.0.0.{i + 1}:{8000 + d}" for d in range(4)],
              control_plane_rpc_address=f"10.0.0.{i + 1}:9000",
              variables=src_vars,
              mesh_shape=[2, 4],
              mesh_axes=["fsdp", "tp"],
          )
        for j, u in enumerate(dst_units):
          ctrl.register_work_unit(
              u,
              [f"10.0.1.{j + 1}:{8000 + d}" for d in range(4)],
              control_plane_rpc_address=f"10.0.1.{j + 1}:9000",
              variables=dst_vars,
              mesh_shape=[1, 4],
              mesh_axes=["fsdp", "tp"],
          )
        fut = ctrl.start_transfer(
            src_units=src_units,
            dst_units=dst_units,
            use_block_chunks=True,
            req_id="req_cmp",
            uuid=12345,
        )
        asyncio.run(fut.wait())
        plans.append(ctrl.get_plan("req_cmp"))
      finally:
        ctrl.worker_rpc_client.close()

    self.assertLen(plans, 2)
    seq_plan = plans[0]
    par_plan = plans[1]
    self.assertEqual(
        seq_plan.expected_block_count, par_plan.expected_block_count
    )
    self.assertEqual(seq_plan.dst_endpoint_counts, par_plan.dst_endpoint_counts)
    for u in src_units:
      self.assertEqual(
          {k: list(v) for k, v in seq_plan.shard_push_schedules[u].items()},
          {k: list(v) for k, v in par_plan.shard_push_schedules[u].items()},
      )


if __name__ == "__main__":
  absltest.main()
