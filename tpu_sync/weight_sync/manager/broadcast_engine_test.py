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
from tpu_sync.weight_sync.manager import broadcast_engine


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

  def test_execute_slice_broadcast_pipeline_removes_barriers_and_balances_relays(
      self,
  ) -> None:
    """Verifies pipeline eliminates barriers, limits src to 2 copies, and balances relays."""
    src = RaidenId(job_name="src", job_replica_id="0", data_name="w")
    dsts = [
        RaidenId(job_name="dst", job_replica_id=str(i), data_name="w")
        for i in range(4)
    ]

    class BarrierFreeTestingRpcClient(
        raiden_controller.WeightSyncWorkerRpcClient
    ):
      """Client that records invocations and delays Group 0 relay transfers."""

      def __init__(self) -> None:
        super().__init__()
        self.invocations: list[tuple[RaidenId, Any]] = []
        self.group0_relays_can_finish = asyncio.Event()
        self.group1_src_started_before_group0_relay_done = False

      async def start_transfer(
          self,
          target_id: RaidenId,
          transfer_plan: Any,
          address: Optional[str] = None,
      ) -> None:
        del address
        self.invocations.append((target_id, transfer_plan))
        sender = transfer_plan.src_units[0]
        # Check if Group 1 started by src
        if sender == src and "_1_" in transfer_plan.req_id:
          if not self.group0_relays_can_finish.is_set():
            self.group1_src_started_before_group0_relay_done = True
            self.group0_relays_can_finish.set()

        # If this is a relay transfer for Group 0 (sender != src), pause until Group 1 is started by src
        if (
            sender != src
            and "_0_" in transfer_plan.req_id
            and target_id == sender
        ):
          await self.group0_relays_can_finish.wait()

    rpc_client = BarrierFreeTestingRpcClient()
    self.addCleanup(rpc_client.close)
    engine = broadcast_engine.BroadcastEngine(rpc_client)

    # 2 groups: Group 0 (layer 0) and Group 1 (layer 1)
    key0 = (src, 0, 0, 0, 1024, 0, 1, 0, 0)
    targets0: list[tuple[Any, ...]] = [
        (dsts[i], f"127.0.0.1:800{i}", 0, 0, 0, 0) for i in range(4)
    ]
    key1 = (src, 0, 0, 0, 1024, 0, 1, 1, 0)
    targets1: list[tuple[Any, ...]] = [
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
        engine.execute_slice_broadcast_pipeline(
            groups_list=[[(key0, targets0)], [(key1, targets1)]],
            final_plan=final_plan,
            fanout_k=2,
            req_id="req_pipeline_test",
            dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
            registered_shards=registered_shards,
        )
    )

    # 1. Verify src starts Group 1 before Group 0's relay transfers complete
    self.assertTrue(
        rpc_client.group1_src_started_before_group0_relay_done,
        "Source must immediately advance to Group 1 while Group 0's relay is"
        " in-flight.",
    )

    # 2. Verify src sends only 2 copies per group (4 total pushes across 2 groups, NOT 6!)
    src_sender_pushes = [
        plan
        for target_id, plan in rpc_client.invocations
        if target_id == plan.src_units[0] and plan.src_units[0] == src
    ]
    g0_src_pushes = [p for p in src_sender_pushes if "_0_" in p.req_id]
    g1_src_pushes = [p for p in src_sender_pushes if "_1_" in p.req_id]
    self.assertLen(
        g0_src_pushes,
        2,
        "Source must dispatch exactly fanout_k=2 copies for Group 0, got"
        f" {len(g0_src_pushes)}",
    )
    self.assertLen(
        g1_src_pushes,
        2,
        "Source must dispatch exactly fanout_k=2 copies for Group 1, got"
        f" {len(g1_src_pushes)}",
    )
    self.assertLen(
        src_sender_pushes,
        4,
        "Source must dispatch exactly 4 total pushes across 2 groups, got"
        f" {len(src_sender_pushes)}",
    )

    # 3. Verify both dst_0 and dst_1 act as relays
    relay_sender_pushes = [
        plan
        for target_id, plan in rpc_client.invocations
        if target_id == plan.src_units[0] and plan.src_units[0] != src
    ]
    relay_senders = {p.src_units[0] for p in relay_sender_pushes}
    self.assertIn(
        dsts[0],
        relay_senders,
        "dst_0 must act as a relay in the broadcast pipeline.",
    )
    self.assertIn(
        dsts[1],
        relay_senders,
        "dst_1 must act as a relay in the broadcast pipeline.",
    )

    # 4. Verify all 4 destinations received data for both groups
    for g_idx in (0, 1):
      dsts_received_g = {
          plan.dst_units[0]
          for target_id, plan in rpc_client.invocations
          if target_id == plan.dst_units[0] and f"_{g_idx}_" in plan.req_id
      }
      self.assertEqual(
          dsts_received_g,
          set(dsts),
          f"All destinations must receive data for Group {g_idx}",
      )

  def test_coalesce_contiguous_relay_entries(self) -> None:
    """Verifies relay coalescing behavior for weight sync and non-weight sync."""
    chunk_size = 64 * 1024

    # 1. Non-weight sync (is_weight_sync=False):
    # Entries with identical block IDs coalesce
    same_block_entries = [
        (
            "127.0.0.1:8001",
            0,
            i * chunk_size,
            i * chunk_size,
            chunk_size,
            5,
            5,
            chunk_size,
            chunk_size,
            1,
            0,
            0,
        )
        for i in range(2)
    ]
    coalesced_same = broadcast_engine._coalesce_contiguous_relay_entries(
        same_block_entries, is_weight_sync=False
    )
    self.assertLen(coalesced_same, 1)
    self.assertEqual(coalesced_same[0][4], 2 * chunk_size)
    self.assertEqual(coalesced_same[0][5], 5)
    self.assertEqual(coalesced_same[0][6], 5)

    # Entries with differing block IDs do NOT coalesce when is_weight_sync=False
    diff_block_entries = [
        (
            "127.0.0.1:8001",
            0,
            i * chunk_size,
            i * chunk_size,
            chunk_size,
            i,
            i,
            chunk_size,
            chunk_size,
            1,
            0,
            0,
        )
        for i in range(2)
    ]
    coalesced_diff = broadcast_engine._coalesce_contiguous_relay_entries(
        diff_block_entries, is_weight_sync=False
    )
    self.assertLen(coalesced_diff, 2)

    # 2. Weight sync (is_weight_sync=True):
    # 64 entries with varying block IDs (0..63) coalesce into 1 4MB entry with block_id = 0
    varying_block_entries = [
        (
            "127.0.0.1:8001",
            0,
            i * chunk_size,
            i * chunk_size,
            chunk_size,
            i,
            i,
            chunk_size,
            chunk_size,
            1,
            0,
            0,
        )
        for i in range(64)
    ]
    coalesced_ws = broadcast_engine._coalesce_contiguous_relay_entries(
        varying_block_entries,
        max_chunk_bytes=4 * 1024 * 1024,
        is_weight_sync=True,
    )
    self.assertLen(coalesced_ws, 1)
    entry = coalesced_ws[0]
    self.assertEqual(entry[4], 4 * 1024 * 1024)
    self.assertEqual(entry[5], 0)
    self.assertEqual(entry[6], 0)
    self.assertEqual(entry[7], 4 * 1024 * 1024)
    self.assertEqual(entry[8], 4 * 1024 * 1024)
    self.assertEqual(entry[9], 1)

    hop_expected_block_count = sum(
        1 if (e[9] == 1 or (e[7] == e[4] and e[8] == e[4])) else e[9]
        for e in coalesced_ws
    )
    self.assertEqual(hop_expected_block_count, 1)

  def test_coalesce_pipeline_groups(self) -> None:
    """Verifies 40 single-group stages coalesce into 8 stages of 5 groups."""
    src = RaidenId(job_name="src", job_replica_id="0", data_name="w")
    dsts = [
        RaidenId(job_name="dst", job_replica_id=str(i), data_name="w")
        for i in range(4)
    ]
    targets = [(dsts[i], f"127.0.0.1:800{i}", 0, 0, 0, 0) for i in range(4)]

    # 40 groups sharing the same (src, shard_idx=0, targets)
    groups_list = [
        [((src, 0, 0, 0, 1024, 0, 1, layer_idx, 0), targets)]
        for layer_idx in range(40)
    ]
    coalesced = broadcast_engine._coalesce_pipeline_groups(
        groups_list, target_stages=8
    )
    self.assertLen(coalesced, 8)
    for stage in coalesced:
      self.assertLen(stage, 5)

  def test_execute_slice_broadcast_pipeline_cancels_siblings_on_failure(
      self,
  ) -> None:
    """Verifies that sibling transfer tasks are cancelled when one hop fails."""
    src = RaidenId(job_name="src", job_replica_id="0", data_name="w")
    dst0 = RaidenId(job_name="dst", job_replica_id="0", data_name="w")
    dst1 = RaidenId(job_name="dst", job_replica_id="1", data_name="w")

    sibling_task_was_cancelled = False

    class FailureRpcClient(raiden_controller.WeightSyncWorkerRpcClient):

      async def start_transfer(
          self,
          target_id: RaidenId,
          transfer_plan: Any,
          address: Optional[str] = None,
      ) -> None:
        del address
        nonlocal sibling_task_was_cancelled
        if target_id == dst0:
          await asyncio.sleep(0.01)
          raise RuntimeError("Simulated transfer failure")
        elif target_id == dst1:
          try:
            await asyncio.sleep(10.0)
          except asyncio.CancelledError:
            sibling_task_was_cancelled = True
            raise

    rpc_client = FailureRpcClient()
    self.addCleanup(rpc_client.close)
    engine = broadcast_engine.BroadcastEngine(rpc_client)

    key0 = (src, 0, 0, 0, 1024, 0, 1, 0, 0)
    targets0 = [
        (dst0, "127.0.0.1:8000", 0, 0, 0, 0),
        (dst1, "127.0.0.1:8001", 0, 0, 0, 0),
    ]
    final_plan = raiden_controller.TransferPlan(
        src_units=[src],
        dst_units=[dst0, dst1],
        plan=None,
        worker_data_addresses={
            src: ["127.0.0.1:8000"],
            dst0: ["127.0.0.1:8001"],
            dst1: ["127.0.0.1:8002"],
        },
    )
    registered_shards = {src: ["s0"], dst0: ["s0"], dst1: ["s0"]}

    with self.assertRaisesRegex(RuntimeError, "Simulated transfer failure"):
      asyncio.run(
          engine.execute_slice_broadcast_pipeline(
              groups_list=[[(key0, targets0)]],
              final_plan=final_plan,
              fanout_k=2,
              req_id="req_failure_test",
              dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
              registered_shards=registered_shards,
          )
      )

    self.assertTrue(
        sibling_task_was_cancelled,
        "Sibling transfer task must be cancelled when one hop fails.",
    )

  def test_execute_slice_broadcast_pipeline_invalid_fanout(self) -> None:
    """Verifies ValueError is raised when fanout_k <= 0."""
    src = RaidenId(job_name="src", job_replica_id="0", data_name="w")
    dst0 = RaidenId(job_name="dst", job_replica_id="0", data_name="w")
    rpc_client = raiden_controller.WeightSyncWorkerRpcClient()
    self.addCleanup(rpc_client.close)
    engine = broadcast_engine.BroadcastEngine(rpc_client)

    key0 = (src, 0, 0, 0, 1024, 0, 1, 0, 0)
    targets0 = [(dst0, "127.0.0.1:8000", 0, 0, 0, 0)]
    final_plan = raiden_controller.TransferPlan(
        src_units=[src],
        dst_units=[dst0],
        plan=None,
        worker_data_addresses={
            src: ["127.0.0.1:8000"],
            dst0: ["127.0.0.1:8001"],
        },
    )
    registered_shards = {src: ["s0"], dst0: ["s0"]}

    for invalid_k in (0, -1):
      with self.assertRaisesRegex(ValueError, "fanout_k must be >= 1"):
        asyncio.run(
            engine.execute_slice_broadcast_pipeline(
                groups_list=[[(key0, targets0)]],
                final_plan=final_plan,
                fanout_k=invalid_k,
                req_id="req_invalid_k",
                dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
                registered_shards=registered_shards,
            )
        )


if __name__ == "__main__":
  absltest.main()
