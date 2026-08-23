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
import socket
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
    self.assertEqual(plan.expected_block_count, 8)

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
    self.assertEqual(plan.expected_block_count, 8)

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

    # Mixed skip tiling: 8 strided tasks for layer 0, 8 strided tasks for layer 1.
    # Total expected = 8 + 8 = 16.
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
    self.assertEqual(plan.expected_block_count, 16)

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

    # Re-registering target invalidates plan cache
    controller.register_work_unit(
        target,
        ["10.0.0.2:8000"],
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
    )
    self.assertEqual(indices, [(0, 4), (1, 5), (2, 6), (3, 7)])

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

    self.assertLen(src_calls, 8)
    self.assertLen(target_0_calls, 4)
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


if __name__ == "__main__":
  absltest.main()
