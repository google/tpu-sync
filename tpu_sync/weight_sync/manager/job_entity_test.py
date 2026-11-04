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

"""Unit tests for JobEntity host dispatch and request encoding."""

import asyncio
from typing import Any

from absl.testing import absltest

from tpu_sync.api.common import RaidenId
from tpu_sync.common.control_pipe import control_pipe_client
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.weight_sync.manager import controller_types
from tpu_sync.weight_sync.manager import job_entity


class StubControlPipeClient:
  """Stub ControlPipeClient that records sent requests and returns success."""

  def __init__(self) -> None:
    self.backend = control_pipe_client.ControlPipeBackendType.TCP
    self.sent_requests: list[tuple[str, bytes]] = []

  def send_raw_bytes_sync(
      self,
      endpoint: str,
      payload: bytes,
      timeout: float = 600.0,
      message_type: str = "",
  ) -> bytes:
    del timeout, message_type
    self.sent_requests.append((endpoint, payload))
    resp = raiden_service_pb2.ControlResponse(success=True)
    return resp.SerializeToString()

  def close(self) -> None:
    pass


class JobEntityTest(absltest.TestCase):
  """Unit tests for JobEntity."""

  def setUp(self):
    super().setUp()
    self.src_unit = RaidenId("trainer", "0", "weights")
    self.dst_unit = RaidenId("sampler", "0", "weights")
    self.pipe_stub = StubControlPipeClient()

  def _make_dummy_entry(self, shard_idx: int = 0) -> tuple[Any, ...]:
    return (
        "10.11.0.3:8000",  # dst_peer
        0,  # dst_shard_idx
        0,  # dst_block_offset
        0,  # src_block_offset
        1024,  # size
        0,  # src_block_id
        0,  # dst_block_id
        1024,  # src_stride
        1024,  # dst_stride
        1,  # count
        0,  # layer_idx
        0,  # pool_group
    )

  def test_encode_start_transfer_skips_idle_sender_in_block_chunk_plan(self):
    """Verifies idle sender host with no push schedules returns None to skip dispatch."""
    entity = job_entity.JobEntity(
        unit=self.src_unit,
        shards=["10.11.0.1:8000"],
        control_endpoints=["10.11.0.1:9000"],
        control_pipe=self.pipe_stub,
    )
    self.addCleanup(entity.worker_rpc_client.close)

    # Shard push schedules has work for shard 99, but this host owns shard 0
    plan = controller_types.TransferPlan(
        src_units=[self.src_unit],
        dst_units=[self.dst_unit],
        plan=None,
        shard_push_schedules={
            self.src_unit: {99: [self._make_dummy_entry(shard_idx=99)]}
        },
        worker_data_addresses={
            self.src_unit: ["10.11.0.1:8000"],
            self.dst_unit: ["10.11.0.3:8000"],
        },
        endpoint_to_shards={(self.src_unit, "10.11.0.1:9000"): [0]},
        use_block_chunks=True,
        is_sender=True,
    )

    payload = entity.encode_start_transfer(
        plan, address="10.11.0.1:9000", unit=self.src_unit
    )
    self.assertIsNone(
        payload,
        "Idle sender host with empty shard_push_schedules must return None",
    )

  def test_encode_start_transfer_includes_active_sender_in_block_chunk_plan(
      self,
  ):
    """Verifies active sender host with push schedules serializes non-None request."""
    entity = job_entity.JobEntity(
        unit=self.src_unit,
        shards=["10.11.0.1:8000"],
        control_endpoints=["10.11.0.1:9000"],
        control_pipe=self.pipe_stub,
    )
    self.addCleanup(entity.worker_rpc_client.close)

    plan = controller_types.TransferPlan(
        src_units=[self.src_unit],
        dst_units=[self.dst_unit],
        plan=None,
        shard_push_schedules={
            self.src_unit: {0: [self._make_dummy_entry(shard_idx=0)]}
        },
        worker_data_addresses={
            self.src_unit: ["10.11.0.1:8000"],
            self.dst_unit: ["10.11.0.3:8000"],
        },
        endpoint_to_shards={(self.src_unit, "10.11.0.1:9000"): [0]},
        use_block_chunks=True,
        is_sender=True,
    )

    payload = entity.encode_start_transfer(
        plan, address="10.11.0.1:9000", unit=self.src_unit
    )
    self.assertIsNotNone(payload)

    req = raiden_service_pb2.ControlRequest()
    req.ParseFromString(payload)
    self.assertEqual(
        req.command, raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER
    )
    self.assertIn(0, req.start_transfer_request.shard_push_schedules)

  def test_encode_start_transfer_peers_empty_for_block_chunk_plan(self):
    """Verifies peers is empty for block-chunk transfers to prevent legacy PushWeights."""
    entity = job_entity.JobEntity(
        unit=self.src_unit,
        shards=["10.11.0.1:8000"],
        control_endpoints=["10.11.0.1:9000"],
        control_pipe=self.pipe_stub,
    )
    self.addCleanup(entity.worker_rpc_client.close)

    plan = controller_types.TransferPlan(
        src_units=[self.src_unit],
        dst_units=[self.dst_unit],
        plan=None,
        shard_push_schedules={
            self.src_unit: {0: [self._make_dummy_entry(shard_idx=0)]}
        },
        worker_data_addresses={
            self.src_unit: ["10.11.0.1:8000"],
            self.dst_unit: ["10.11.0.3:8000"],
        },
        endpoint_to_shards={(self.src_unit, "10.11.0.1:9000"): [0]},
        use_block_chunks=True,
        is_sender=True,
    )

    payload = entity.encode_start_transfer(
        plan, address="10.11.0.1:9000", unit=self.src_unit
    )
    self.assertIsNotNone(payload)

    req = raiden_service_pb2.ControlRequest()
    req.ParseFromString(payload)
    self.assertEmpty(
        req.peers,
        "req.peers must be empty for block-chunk transfers so C++ listener"
        " does not trigger unresharded legacy PushWeights fallback",
    )

  def test_encode_start_transfer_peers_populated_for_legacy_plan(self):
    """Verifies peers is populated for legacy plan-less transfers without block chunks."""
    entity = job_entity.JobEntity(
        unit=self.src_unit,
        shards=["10.11.0.1:8000"],
        control_endpoints=["10.11.0.1:9000"],
        control_pipe=self.pipe_stub,
    )
    self.addCleanup(entity.worker_rpc_client.close)

    plan = controller_types.TransferPlan(
        src_units=[self.src_unit],
        dst_units=[self.dst_unit],
        plan={},
        shard_push_schedules={},
        worker_data_addresses={
            self.src_unit: ["10.11.0.1:8000"],
            self.dst_unit: ["10.11.0.3:8000"],
        },
        use_block_chunks=False,
        is_sender=True,
    )

    payload = entity.encode_start_transfer(
        plan, address="10.11.0.1:9000", unit=self.src_unit
    )
    self.assertIsNotNone(payload)

    req = raiden_service_pb2.ControlRequest()
    req.ParseFromString(payload)
    self.assertEqual(list(req.peers), ["10.11.0.3:8000"])

  def test_encode_start_transfer_receiver_not_skipped(self):
    """Verifies receiver hosts (is_sender=False) are never skipped even with empty push schedules."""
    entity = job_entity.JobEntity(
        unit=self.dst_unit,
        shards=["10.11.0.3:8000"],
        control_endpoints=["10.11.0.3:9000"],
        control_pipe=self.pipe_stub,
    )
    self.addCleanup(entity.worker_rpc_client.close)

    plan = controller_types.TransferPlan(
        src_units=[self.src_unit],
        dst_units=[self.dst_unit],
        plan=None,
        shard_push_schedules={},
        worker_data_addresses={
            self.src_unit: ["10.11.0.1:8000"],
            self.dst_unit: ["10.11.0.3:8000"],
        },
        use_block_chunks=True,
        is_sender=False,
        expected_block_count=10,
        expected_layer_chunk_counts={0: 6, 1: 4},
    )

    payload = entity.encode_start_transfer(
        plan, address="10.11.0.3:9000", unit=self.dst_unit
    )
    self.assertIsNotNone(
        payload, "Receivers must not be skipped even without push schedules"
    )

    req = raiden_service_pb2.ControlRequest()
    req.ParseFromString(payload)
    self.assertEqual(req.start_transfer_request.expected_block_count, 10)
    self.assertEqual(
        dict(req.start_transfer_request.expected_layer_chunk_counts),
        {0: 6, 1: 4},
    )
    self.assertEmpty(req.peers)
    self.assertEqual(
        req.start_transfer_request.host_tiling_mode,
        raiden_service_pb2.HOST_TILING_MODE_UNSPECIFIED,
    )

  def test_encode_start_transfer_receiver_host_tiling_mode(self):
    """Verifies host_tiling_mode is encoded for receivers."""
    entity = job_entity.JobEntity(
        unit=self.dst_unit,
        shards=["10.11.0.3:8000"],
        control_endpoints=["10.11.0.3:9000"],
        control_pipe=self.pipe_stub,
    )
    self.addCleanup(entity.worker_rpc_client.close)

    for mode in (
        raiden_service_pb2.HOST_TILING_MODE_DEFERRED,
        raiden_service_pb2.HOST_TILING_MODE_ON_ARRIVAL,
        raiden_service_pb2.HOST_TILING_MODE_PRE_TILED,
    ):
      with self.subTest(mode=mode):
        plan = controller_types.TransferPlan(
            src_units=[self.src_unit],
            dst_units=[self.dst_unit],
            plan=None,
            shard_push_schedules={},
            worker_data_addresses={
                self.src_unit: ["10.11.0.1:8000"],
                self.dst_unit: ["10.11.0.3:8000"],
            },
            use_block_chunks=True,
            is_sender=False,
            expected_block_count=10,
            expected_layer_chunk_counts={0: 6, 1: 4},
            host_tiling_mode=mode,
        )

        payload = entity.encode_start_transfer(
            plan, address="10.11.0.3:9000", unit=self.dst_unit
        )
        self.assertIsNotNone(payload)
        req = raiden_service_pb2.ControlRequest()
        req.ParseFromString(payload)
        self.assertEqual(req.start_transfer_request.host_tiling_mode, mode)

  def test_encode_start_transfer_receiver_requires_layer_chunk_counts(self):
    """Verifies receivers fail fast without per-layer counts summing to total."""
    entity = job_entity.JobEntity(
        unit=self.dst_unit,
        shards=["10.11.0.3:8000"],
        control_endpoints=["10.11.0.3:9000"],
        control_pipe=self.pipe_stub,
    )
    self.addCleanup(entity.worker_rpc_client.close)

    for layer_counts in ({}, {0: 6, 1: 3}):
      with self.subTest(layer_counts=layer_counts):
        plan = controller_types.TransferPlan(
            src_units=[self.src_unit],
            dst_units=[self.dst_unit],
            plan=None,
            shard_push_schedules={},
            worker_data_addresses={
                self.src_unit: ["10.11.0.1:8000"],
                self.dst_unit: ["10.11.0.3:8000"],
            },
            use_block_chunks=True,
            is_sender=False,
            expected_block_count=10,
            expected_layer_chunk_counts=layer_counts,
        )
        with self.assertRaisesRegex(ValueError, "per-layer chunk counts"):
          entity.encode_start_transfer(
              plan, address="10.11.0.3:9000", unit=self.dst_unit
          )

  def test_start_transfer_dispatches_only_to_active_hosts(self):
    """Verifies start_transfer only dispatches RPCs to hosts that own active push schedules."""
    entity = job_entity.JobEntity(
        unit=self.src_unit,
        shards=["10.11.0.1:8000", "10.11.0.2:8000"],
        control_endpoints=["10.11.0.1:9000", "10.11.0.2:9000"],
        control_pipe=self.pipe_stub,
    )
    self.addCleanup(entity.worker_rpc_client.close)

    # Only shard 0 (on 10.11.0.1:9000) has work; shard 1 (on 10.11.0.2:9000) is idle
    plan = controller_types.TransferPlan(
        src_units=[self.src_unit],
        dst_units=[self.dst_unit],
        plan=None,
        shard_push_schedules={
            self.src_unit: {0: [self._make_dummy_entry(shard_idx=0)]}
        },
        worker_data_addresses={
            self.src_unit: ["10.11.0.1:8000", "10.11.0.2:8000"],
            self.dst_unit: ["10.11.0.3:8000"],
        },
        endpoint_to_shards={
            (self.src_unit, "10.11.0.1:9000"): [0],
            (self.src_unit, "10.11.0.2:9000"): [1],
        },
        use_block_chunks=True,
        is_sender=True,
    )

    asyncio.run(entity.start_transfer(plan))

    dispatched_endpoints = [ep for ep, _ in self.pipe_stub.sent_requests]
    self.assertIn("10.11.0.1:9000", dispatched_endpoints)
    self.assertNotIn(
        "10.11.0.2:9000",
        dispatched_endpoints,
        "Idle host 10.11.0.2:9000 must NOT receive an RPC transfer command",
    )

  def test_multi_endpoint_proxy_without_endpoint_to_shards_broadcasts_all_schedules(
      self,
  ):
    """Verifies multi-endpoint proxy without endpoint_to_shards broadcasts all schedules."""
    endpoints = ["10.11.0.1:9000", "10.11.0.2:9000"]
    entity = job_entity.JobEntity(
        unit=self.src_unit,
        shards=["10.11.0.1:8000", "10.11.0.2:8000"],
        control_endpoints=endpoints,
        control_pipe=self.pipe_stub,
    )
    self.addCleanup(entity.worker_rpc_client.close)

    plan = controller_types.TransferPlan(
        src_units=[self.src_unit],
        dst_units=[self.dst_unit],
        plan=None,
        shard_push_schedules={
            self.src_unit: {
                0: [self._make_dummy_entry(shard_idx=0)],
                1: [self._make_dummy_entry(shard_idx=1)],
            }
        },
        worker_data_addresses={
            self.src_unit: ["10.11.0.1:8000", "10.11.0.2:8000"],
            self.dst_unit: ["10.11.0.3:8000"],
        },
        use_block_chunks=True,
        is_sender=True,
    )

    self.assertIsNone(
        entity.get_host_owned_shards(
            plan, address="10.11.0.1:9000", unit=self.src_unit
        )
    )
    self.assertIsNone(
        entity.get_host_owned_shards(
            plan, address="10.11.0.2:9000", unit=self.src_unit
        )
    )
    self.assertTrue(
        entity.is_payload_invariant_across_hosts(
            plan, endpoints, unit=self.src_unit
        )
    )

    asyncio.run(entity.start_transfer(plan))

    dispatched = {ep: payload for ep, payload in self.pipe_stub.sent_requests}
    self.assertIn("10.11.0.1:9000", dispatched)
    self.assertIn("10.11.0.2:9000", dispatched)

    for ep in endpoints:
      req = raiden_service_pb2.ControlRequest()
      req.ParseFromString(dispatched[ep])
      self.assertEqual(
          req.command, raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER
      )
      self.assertEqual(
          set(req.start_transfer_request.shard_push_schedules.keys()),
          {0, 1},
          f"Host {ep} must receive all sender schedules under broadcast"
          " contract",
      )

  def test_tree_hop_template_cache_differentiates_stages_and_reuses_on_warm_sync(
      self,
  ):
    """Verifies shared cached_serialized_payloads differentiates stages/destinations and reuses templates on warm syncs."""
    entity = job_entity.JobEntity(
        unit=self.src_unit,
        shards=["10.11.0.1:8000"],
        control_endpoints=["10.11.0.1:9000"],
        control_pipe=self.pipe_stub,
        weight_sync_mode=True,
    )
    self.addCleanup(entity.worker_rpc_client.close)

    dst_0 = RaidenId("sampler", "0", "weights")
    dst_1 = RaidenId("sampler", "1", "weights")
    shared_cache: dict[Any, Any] = {}

    def _make_hop_plan(
        dst_unit: RaidenId,
        dst_peer: str,
        layer_idx: int,
        uuid_val: int,
        req_id: str,
    ) -> controller_types.TransferPlan:
      ref_sched = controller_types.PlanReferencedShardSchedule(
          {1: [(dst_peer, 0, 0, 0, 1024, 0, 0, 1024, 1024, 1)]},
          {layer_idx: 1},
          [(layer_idx, 1)],
      )
      return controller_types.TransferPlan(
          src_units=[self.src_unit],
          dst_units=[dst_unit],
          plan=None,
          shard_push_schedules={self.src_unit: {0: ref_sched}},
          worker_data_addresses={
              self.src_unit: ["10.11.0.1:8000"],
              dst_0: ["10.11.0.3:8000"],
              dst_1: ["10.11.0.4:8000"],
          },
          use_block_chunks=True,
          is_sender=True,
          is_weight_sync=True,
          uuid=uuid_val,
          req_id=req_id,
          cached_serialized_payloads=shared_cache,
      )

    # Cold sync (uuid=100): Stage 0 -> dst_0, Stage 1 -> dst_0, Stage 0 -> dst_1
    p_s0_d0 = entity.encode_start_transfer(
        _make_hop_plan(dst_0, "10.11.0.3:8000", 0, 100, "sync0_s0_d0"),
        address="10.11.0.1:9000",
        unit=self.src_unit,
    )
    p_s1_d0 = entity.encode_start_transfer(
        _make_hop_plan(dst_0, "10.11.0.3:8000", 5, 100, "sync0_s1_d0"),
        address="10.11.0.1:9000",
        unit=self.src_unit,
    )
    p_s0_d1 = entity.encode_start_transfer(
        _make_hop_plan(dst_1, "10.11.0.4:8000", 0, 100, "sync0_s0_d1"),
        address="10.11.0.1:9000",
        unit=self.src_unit,
    )

    self.assertNotEqual(p_s0_d0, p_s1_d0)
    self.assertNotEqual(p_s0_d0, p_s0_d1)

    # Warm sync (uuid=200): verify template hit skips rebuilding schedule protos
    build_calls = 0
    orig_build = entity.build_sender_push_schedule_protos

    def _counting_build(push_schedules):
      nonlocal build_calls
      build_calls += 1
      return orig_build(push_schedules)

    entity.build_sender_push_schedule_protos = _counting_build
    warm_s1_d0 = entity.encode_start_transfer(
        _make_hop_plan(dst_0, "10.11.0.3:8000", 5, 200, "sync1_s1_d0"),
        address="10.11.0.1:9000",
        unit=self.src_unit,
    )
    self.assertEqual(build_calls, 0)

    req_warm = raiden_service_pb2.ControlRequest()
    req_warm.ParseFromString(warm_s1_d0)
    self.assertEqual(req_warm.start_transfer_request.uuid, 200)
    self.assertEqual(req_warm.start_transfer_request.req_id, "sync1_s1_d0")
    entry = req_warm.start_transfer_request.shard_push_schedules[0].entries[0]
    self.assertEqual(entry.layer_idx, 5)
    self.assertEqual(entry.dst_peer, "10.11.0.3:8000")

  def test_compute_endpoint_to_shards_topologies(self):
    """Verifies compute_endpoint_to_shards across multi-host, NUMA, and torus-interleaved topologies."""
    unit = RaidenId("trainer", "0", "weights")
    ent_key = controller_types.entity_key_from_unit(unit)

    # 1. Multi-host multi-NUMA: 2 hosts, 2 control endpoints/host, 2 unique
    # data endpoints/host, 8 shards total.
    ctrl_numa = [
        "10.0.0.1:9000,10.0.0.1:9001,10.0.0.2:9000,10.0.0.2:9001",
    ]
    data_numa = [
        "10.0.0.1:8000",
        "10.0.0.1:8000",
        "10.0.0.1:8001",
        "10.0.0.1:8001",
        "10.0.0.2:8000",
        "10.0.0.2:8000",
        "10.0.0.2:8001",
        "10.0.0.2:8001",
    ]
    res_numa = controller_types.compute_endpoint_to_shards(
        unit, ctrl_numa, data_numa
    )
    self.assertEqual(res_numa[(unit, "10.0.0.1:9000")], {0, 1})
    self.assertEqual(res_numa[(ent_key, "10.0.0.1:9000")], {0, 1})
    self.assertEqual(res_numa[(unit, "10.0.0.1:9001")], {2, 3})
    self.assertEqual(res_numa[(unit, "10.0.0.2:9000")], {4, 5})
    self.assertEqual(res_numa[(unit, "10.0.0.2:9001")], {6, 7})

    # 2. Torus-interleaved TP=2 on loopback
    ctrl_torus = ["127.0.0.1:9000,127.0.0.1:9001"]
    data_torus = [
        "127.0.0.1:8000",
        "127.0.0.1:8000",
        "127.0.0.1:8001",
        "127.0.0.1:8001",
        "127.0.0.1:8000",
        "127.0.0.1:8000",
        "127.0.0.1:8001",
        "127.0.0.1:8001",
    ]
    res_torus = controller_types.compute_endpoint_to_shards(
        unit, ctrl_torus, data_torus
    )
    self.assertEqual(res_torus[(unit, "127.0.0.1:9000")], {0, 1, 4, 5})
    self.assertEqual(res_torus[(unit, "127.0.0.1:9001")], {2, 3, 6, 7})

    # 3. Multi-host single control endpoint per host with per-chip data ports
    ctrl_single = ["10.0.0.1:9000,10.0.0.2:9000"]
    data_single = [f"10.0.0.1:{8000 + i}" for i in range(4)] + [
        f"10.0.0.2:{8000 + i}" for i in range(4)
    ]
    res_single = controller_types.compute_endpoint_to_shards(
        unit, ctrl_single, data_single
    )
    self.assertEqual(res_single[(unit, "10.0.0.1:9000")], {0, 1, 2, 3})
    self.assertEqual(res_single[(unit, "10.0.0.2:9000")], {4, 5, 6, 7})

  def test_encode_start_transfer_builds_only_host_owned_shards(self):
    """Verifies encode_start_transfer with endpoint_to_shards only builds protos for owned_shards."""
    entity = job_entity.JobEntity(
        unit=self.src_unit,
        shards=[
            "10.11.0.1:8000",
            "10.11.0.1:8000",
            "10.11.0.2:8000",
            "10.11.0.2:8000",
        ],
        control_endpoints=["10.11.0.1:9000", "10.11.0.2:9000"],
        control_pipe=self.pipe_stub,
    )
    self.addCleanup(entity.worker_rpc_client.close)

    ep_to_shards = controller_types.compute_endpoint_to_shards(
        self.src_unit,
        ["10.11.0.1:9000,10.11.0.2:9000"],
        [
            "10.11.0.1:8000",
            "10.11.0.1:8000",
            "10.11.0.2:8000",
            "10.11.0.2:8000",
        ],
    )
    plan = controller_types.TransferPlan(
        src_units=[self.src_unit],
        dst_units=[self.dst_unit],
        plan=None,
        shard_push_schedules={
            self.src_unit: {i: [self._make_dummy_entry(i)] for i in range(4)}
        },
        worker_data_addresses={
            self.src_unit: [
                "10.11.0.1:8000",
                "10.11.0.1:8000",
                "10.11.0.2:8000",
                "10.11.0.2:8000",
            ],
            self.dst_unit: ["10.11.0.3:8000"],
        },
        endpoint_to_shards=ep_to_shards,
        use_block_chunks=True,
        is_sender=True,
    )

    built_shard_sets: list[set[int]] = []
    orig_build = entity.build_sender_push_schedule_protos

    def _recording_build(push_schedules):
      built_shard_sets.append(set(push_schedules.keys()))
      return orig_build(push_schedules)

    entity.build_sender_push_schedule_protos = _recording_build

    payload_h0 = entity.encode_start_transfer(
        plan, address="10.11.0.1:9000", unit=self.src_unit
    )
    self.assertEqual(built_shard_sets, [{0, 1}])
    req_h0 = raiden_service_pb2.ControlRequest()
    req_h0.ParseFromString(payload_h0)
    self.assertEqual(
        set(req_h0.start_transfer_request.shard_push_schedules.keys()),
        {0, 1},
    )

    payload_h1 = entity.encode_start_transfer(
        plan, address="10.11.0.2:9000", unit=self.src_unit
    )
    self.assertEqual(built_shard_sets, [{0, 1}, {2, 3}])
    req_h1 = raiden_service_pb2.ControlRequest()
    req_h1.ParseFromString(payload_h1)
    self.assertEqual(
        set(req_h1.start_transfer_request.shard_push_schedules.keys()),
        {2, 3},
    )


if __name__ == "__main__":
  absltest.main()
