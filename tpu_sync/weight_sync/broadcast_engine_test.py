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

"""Unit tests for broadcast_engine."""

import asyncio
from typing import Any, Optional

from absl.testing import absltest

from tpu_sync.api.common import RaidenId
from tpu_sync.rpc import raiden_controller
from tpu_sync.weight_sync import broadcast_engine


class RecordingWorkerRpcClient(raiden_controller.WeightSyncWorkerRpcClient):
  """WorkerRpcClient stub that records all start_transfer invocations."""

  def __init__(self) -> None:
    super().__init__()
    self.invocations: list[tuple[RaidenId, Any]] = []

  async def start_transfer(
      self,
      target_id: RaidenId,
      transfer_plan: Any,
      address: Optional[str] = None,
  ) -> None:
    del address
    self.invocations.append((target_id, transfer_plan))


class BroadcastEngineTest(absltest.TestCase):
  """Tests for BroadcastEngine group partitioning and multi-hop tree broadcast."""

  def test_partition_direct_and_broadcast_groups(self) -> None:
    """Verifies partition_direct_and_broadcast_groups separates direct vs tree groups."""
    src = RaidenId(job_name="src", job_replica_id="0", data_name="w")
    dst1 = RaidenId(job_name="dst", job_replica_id="0", data_name="w")
    dst2 = RaidenId(job_name="dst", job_replica_id="1", data_name="w")

    key1 = (src, 0, 0, 0, 1024, 0, 1, 0, 0)
    targets1 = [(dst1, "127.0.0.1:8001", 0, 0, 0, 0)]

    key2 = (src, 0, 0, 1024, 1024, 0, 1, 1, 0)
    targets2 = [
        (dst1, "127.0.0.1:8001", 0, 0, 1024, 0),
        (dst2, "127.0.0.1:8002", 0, 0, 1024, 0),
    ]

    groups = {key1: targets1, key2: targets2}
    direct, bcast = (
        broadcast_engine.BroadcastEngine.partition_direct_and_broadcast_groups(
            groups, broadcast_k=1
        )
    )
    self.assertIn(src, direct)
    self.assertLen(bcast, 1)

  def test_execute_slice_broadcast_multihop(self) -> None:
    """Verifies multi-hop fanout execution with fanout_k=1, 2, 4 and node promotion."""
    for fanout_k in (1, 2, 4):
      with self.subTest(fanout_k=fanout_k):
        rpc_client = RecordingWorkerRpcClient()
        self.addCleanup(rpc_client.close)
        engine = broadcast_engine.BroadcastEngine(rpc_client)

        src = RaidenId(job_name="src", job_replica_id="0", data_name="w")
        dsts = [
            RaidenId(job_name="dst", job_replica_id=str(i), data_name="w")
            for i in range(4)
        ]

        key = (src, 0, 0, 0, 1024, 0, 1, 0, 0)
        targets: list[tuple[Any, ...]] = [
            (dsts[i], f"127.0.0.1:800{i}", 0, 0, 0, 0) for i in range(4)
        ]

        final_plan = raiden_controller.TransferPlan(
            src_units=[src],
            dst_units=dsts,
            plan=None,
            worker_data_addresses={
                u: [f"127.0.0.1:800{i}"] for i, u in enumerate([src] + dsts)
            },
        )
        registered_shards = {u: ["s0"] for u in [src] + dsts}

        asyncio.run(
            engine.execute_slice_broadcast(
                keys_and_targets=[(key, targets)],
                final_plan=final_plan,
                fanout_k=fanout_k,
                req_id="req_test",
                dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
                registered_shards=registered_shards,
            )
        )

        dst_received = {
            plan.dst_units[0]
            for target_id, plan in rpc_client.invocations
            if target_id == plan.dst_units[0]
        }
        self.assertEqual(dst_received, set(dsts))


if __name__ == "__main__":
  absltest.main()
