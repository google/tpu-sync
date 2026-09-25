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
    self.assertEmpty(req.peers)

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


if __name__ == "__main__":
  absltest.main()
