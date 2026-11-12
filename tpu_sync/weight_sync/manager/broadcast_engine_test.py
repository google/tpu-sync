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
import collections
import threading
from typing import Any, Optional
from unittest import mock

from absl.testing import absltest

from tpu_sync.api.common import RaidenId
from tpu_sync.rpc import raiden_controller
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.weight_sync.manager import broadcast_engine
from tpu_sync.weight_sync.manager import controller_types
from tpu_sync.weight_sync.manager import job_entity
from tpu_sync.weight_sync.manager import reshard_planner


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
            groups, n_seed=1
        )
    )
    self.assertEmpty(direct)
    self.assertLen(bcast, 2)

    with self.assertRaises(ValueError):
      broadcast_engine.BroadcastEngine.partition_direct_and_broadcast_groups(
          groups, n_seed=0
      )

  def test_execute_slice_broadcast_multihop(self) -> None:
    """Verifies multi-hop fanout execution with n_seed=1, 2, 4 and node promotion."""
    for n_seed in (1, 2, 4):
      with self.subTest(n_seed=n_seed):
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
                n_seed=n_seed,
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
            n_seed=2,
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
        "Source must dispatch exactly n_seed=2 copies for Group 0, got"
        f" {len(g0_src_pushes)}",
    )
    self.assertLen(
        g1_src_pushes,
        2,
        "Source must dispatch exactly n_seed=2 copies for Group 1, got"
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
        same_block_entries
    )
    self.assertLen(coalesced_same, 1)
    self.assertEqual(coalesced_same[0][4], 2 * chunk_size)
    self.assertEqual(coalesced_same[0][5], 5)
    self.assertEqual(coalesced_same[0][6], 5)

    # Entries with differing block IDs do NOT coalesce
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
        diff_block_entries
    )
    self.assertLen(coalesced_diff, 2)

    # 2. Weight sync (is_weight_sync=True):
    # 64 entries with varying block IDs (0..63) do not merge across block boundaries.
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
    )
    self.assertLen(coalesced_ws, 64)
    for i, entry in enumerate(coalesced_ws):
      self.assertEqual(entry[4], chunk_size)
      self.assertEqual(entry[5], i)
      self.assertEqual(entry[6], i)

    hop_expected_block_count = sum(
        1 if (e[9] == 1 or (e[7] == e[4] and e[8] == e[4])) else e[9]
        for e in coalesced_ws
    )
    self.assertEqual(hop_expected_block_count, 64)

  def test_coalesce_contiguous_relay_entries_preserves_block_ids_in_weight_sync(
      self,
  ) -> None:
    """Verifies that distinct block IDs are preserved in weight sync coalescing."""
    entries = [
        (
            "127.0.0.1:8001",
            0,
            i * 512,
            i * 512,
            512,
            i,
            i,
            512,
            512,
            1,
            0,
            0,
        )
        for i in range(16)
    ]
    coalesced = broadcast_engine._coalesce_contiguous_relay_entries(entries)
    self.assertLen(coalesced, 16)
    for i, entry in enumerate(coalesced):
      self.assertEqual(entry[4], 512)
      self.assertEqual(entry[5], i)
      self.assertEqual(entry[6], i)

    within_block_entries = [
        (
            "127.0.0.1:8001",
            0,
            0,
            0,
            256,
            0,
            0,
            256,
            256,
            1,
            0,
            0,
        ),
        (
            "127.0.0.1:8001",
            0,
            256,
            256,
            256,
            0,
            0,
            256,
            256,
            1,
            0,
            0,
        ),
    ]
    coalesced_within = broadcast_engine._coalesce_contiguous_relay_entries(
        within_block_entries
    )
    self.assertLen(coalesced_within, 1)
    self.assertEqual(coalesced_within[0][4], 512)
    self.assertEqual(coalesced_within[0][5], 0)
    self.assertEqual(coalesced_within[0][6], 0)

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

  def test_coalesce_pipeline_groups_multiple_shard_runs(self) -> None:
    """Verifies each (src, shard) run coalesces into target_stages independently."""
    dsts = [
        RaidenId(job_name="dst", job_replica_id=str(i), data_name="w")
        for i in range(4)
    ]
    targets = [(dsts[i], f"127.0.0.1:800{i}", 0, 0, 0, 0) for i in range(4)]

    # 8 shards, each with 8 layer groups (total 64 groups)
    groups_list = []
    for shard_idx in range(8):
      src = RaidenId(
          job_name="src", job_replica_id=str(shard_idx), data_name="w"
      )
      for layer_idx in range(8):
        groups_list.append(
            [((src, shard_idx, 0, 0, 1024, 0, 1, layer_idx, 0), targets)]
        )

    coalesced = broadcast_engine._coalesce_pipeline_groups(
        groups_list, target_stages=4
    )
    # Each of the 8 shards should produce 4 pipeline stages (total 32 stages)
    self.assertLen(coalesced, 32)
    for stage in coalesced:
      self.assertLen(stage, 2)

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
              n_seed=2,
              req_id="req_failure_test",
              dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
              registered_shards=registered_shards,
          )
      )

    self.assertTrue(
        sibling_task_was_cancelled,
        "Sibling transfer task must be cancelled when one hop fails.",
    )

  def test_execute_slice_broadcast_pipeline_invalid_seed(self) -> None:
    """Verifies ValueError is raised when n_seed <= 0."""
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

    for invalid_seed in (0, -1):
      with self.assertRaisesRegex(ValueError, "n_seed must be >= 1"):
        asyncio.run(
            engine.execute_slice_broadcast_pipeline(
                groups_list=[[(key0, targets0)]],
                final_plan=final_plan,
                n_seed=invalid_seed,
                req_id="req_invalid_seed",
                dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
                registered_shards=registered_shards,
            )
        )

  def test_execute_slice_broadcast_multishard_relay(self) -> None:
    """Verifies relay nodes dispatch push schedules partitioned by actual shard index."""
    rpc_client = RecordingWorkerRpcClient()
    self.addCleanup(rpc_client.close)
    engine = broadcast_engine.BroadcastEngine(rpc_client)

    src = RaidenId(job_name="src", job_replica_id="0", data_name="w")
    relay = RaidenId(job_name="relay", job_replica_id="0", data_name="w")
    dst = RaidenId(job_name="dst", job_replica_id="0", data_name="w")

    # key0 maps to relay shard 0 (e.g. Host 0)
    key0 = (src, 0, 0, 0, 1024, 0, 1, 0, 0)
    # key1 maps to relay shard 1 (e.g. Host 1)
    key1 = (src, 0, 1, 0, 1024, 0, 1, 0, 0)

    targets0: list[tuple[Any, ...]] = [
        (relay, "127.0.0.1:8001", 0, 0, 0, 0),
        (dst, "127.0.0.1:8002", 0, 0, 0, 0),
    ]
    targets1: list[tuple[Any, ...]] = [
        (relay, "127.0.0.1:8001", 1, 1, 0, 0),
        (dst, "127.0.0.1:8002", 1, 1, 0, 0),
    ]

    final_plan = raiden_controller.TransferPlan(
        src_units=[src],
        dst_units=[relay, dst],
        plan=None,
        worker_data_addresses={
            src: ["127.0.0.1:8000"],
            relay: ["127.0.0.1:8001"],
            dst: ["127.0.0.1:8002"],
        },
        is_weight_sync=True,
        uuid=42,
    )
    registered_shards = {src: ["s0"], relay: ["s0", "s1"], dst: ["s0", "s1"]}

    asyncio.run(
        engine.execute_slice_broadcast(
            keys_and_targets=[(key0, targets0), (key1, targets1)],
            final_plan=final_plan,
            n_seed=1,
            req_id="req_multishard_test",
            dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
            registered_shards=registered_shards,
        )
    )

    # Find the relay's forward transfer plan
    # (where relay is sender, dst is receiver)
    relay_forward_plans = [
        plan
        for target_id, plan in rpc_client.invocations
        if target_id == relay
        and plan.src_units[0] == relay
        and plan.dst_units[0] == dst
    ]
    self.assertLen(relay_forward_plans, 1)
    relay_plan = relay_forward_plans[0]
    self.assertEqual(
        relay_plan.uuid, 42, "Relay plan must preserve final_plan.uuid"
    )

    # Verify both shard 0 and shard 1 are present in the relay's push schedule
    self.assertIn(relay, relay_plan.shard_push_schedules)
    relay_schedules = relay_plan.shard_push_schedules[relay]
    self.assertIn(
        0, relay_schedules, "Relay must include push schedules for shard 0"
    )
    self.assertIn(
        1, relay_schedules, "Relay must include push schedules for shard 1"
    )

    # Verify key0 is under shard 0 and key1 is under shard 1
    self.assertLen(relay_schedules[0], 1)
    entry0 = relay_schedules[0][0]
    self.assertEqual(
        entry0[1], 0, "Destination shard index for key0 should be 0"
    )
    self.assertEqual(entry0[5], 0, "Source block ID for key0 should be 0")
    self.assertEqual(entry0[6], 0, "Destination block ID for key0 should be 0")

    self.assertLen(relay_schedules[1], 1)
    entry1 = relay_schedules[1][0]
    self.assertEqual(
        entry1[1], 1, "Destination shard index for key1 should be 1"
    )
    self.assertEqual(entry1[5], 1, "Source block ID for key1 should be 1")
    self.assertEqual(entry1[6], 1, "Destination block ID for key1 should be 1")

    self.assertEqual(relay_plan.expected_block_count, 2)

  def test_execute_slice_broadcast_multishard_relay_coalescing(self) -> None:
    """Verifies contiguous slices on multiple shards coalesce independently on relay."""
    rpc_client = RecordingWorkerRpcClient()
    self.addCleanup(rpc_client.close)
    engine = broadcast_engine.BroadcastEngine(rpc_client)

    src = RaidenId(job_name="src", job_replica_id="0", data_name="w")
    relay = RaidenId(job_name="relay", job_replica_id="0", data_name="w")
    dst = RaidenId(job_name="dst", job_replica_id="0", data_name="w")

    # Two contiguous slices for shard 0
    # (block_id 0, offsets 0 and 512, size 512)
    key0_a = (src, 0, 0, 0, 512, 0, 1, 0, 0)
    key0_b = (src, 0, 0, 512, 512, 0, 1, 0, 0)
    # Two contiguous slices for shard 1
    # (block_id 1, offsets 0 and 512, size 512)
    key1_a = (src, 0, 1, 0, 512, 0, 1, 0, 0)
    key1_b = (src, 0, 1, 512, 512, 0, 1, 0, 0)

    targets0_a: list[tuple[Any, ...]] = [
        (relay, "127.0.0.1:8001", 0, 0, 0, 0),
        (dst, "127.0.0.1:8002", 0, 0, 0, 0),
    ]
    targets0_b: list[tuple[Any, ...]] = [
        (relay, "127.0.0.1:8001", 0, 0, 512, 0),
        (dst, "127.0.0.1:8002", 0, 0, 512, 0),
    ]
    targets1_a: list[tuple[Any, ...]] = [
        (relay, "127.0.0.1:8001", 1, 1, 0, 0),
        (dst, "127.0.0.1:8002", 1, 1, 0, 0),
    ]
    targets1_b: list[tuple[Any, ...]] = [
        (relay, "127.0.0.1:8001", 1, 1, 512, 0),
        (dst, "127.0.0.1:8002", 1, 1, 512, 0),
    ]

    final_plan = raiden_controller.TransferPlan(
        src_units=[src],
        dst_units=[relay, dst],
        plan=None,
        worker_data_addresses={
            src: ["127.0.0.1:8000"],
            relay: ["127.0.0.1:8001"],
            dst: ["127.0.0.1:8002"],
        },
        is_weight_sync=True,
    )
    registered_shards = {src: ["s0"], relay: ["s0", "s1"], dst: ["s0", "s1"]}

    asyncio.run(
        engine.execute_slice_broadcast(
            keys_and_targets=[
                (key0_a, targets0_a),
                (key0_b, targets0_b),
                (key1_a, targets1_a),
                (key1_b, targets1_b),
            ],
            final_plan=final_plan,
            n_seed=1,
            req_id="req_coalesce_test",
            dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
            registered_shards=registered_shards,
        )
    )

    relay_forward_plans = [
        plan
        for target_id, plan in rpc_client.invocations
        if target_id == relay
        and plan.src_units[0] == relay
        and plan.dst_units[0] == dst
    ]
    self.assertLen(relay_forward_plans, 1)
    relay_plan = relay_forward_plans[0]
    relay_schedules = relay_plan.shard_push_schedules[relay]

    # Both shards must have their 2 slices coalesced into 1 contiguous transfer
    # of 1024 bytes
    self.assertLen(relay_schedules[0], 1)
    self.assertEqual(relay_schedules[0][0][4], 1024)
    self.assertEqual(relay_schedules[0][0][5], 0)

    self.assertLen(relay_schedules[1], 1)
    self.assertEqual(relay_schedules[1][0][4], 1024)
    self.assertEqual(relay_schedules[1][0][5], 1)

    self.assertEqual(relay_plan.expected_block_count, 2)

  def test_stage_broadcast_group_whole_block_sampler_relay(self) -> None:
    """Verifies 1-to-1 whole-block sampler relays across 2 trainers and 4 samplers."""
    rpc_client = RecordingWorkerRpcClient()
    engine = broadcast_engine.BroadcastEngine(worker_rpc_client=rpc_client)

    train_0 = RaidenId("trainer", "0", "weights", 0)
    train_1 = RaidenId("trainer", "1", "weights", 0)
    src_units = [train_0, train_1]
    dst_units = [RaidenId("sampler", str(i), "weights", 0) for i in range(4)]

    data_addresses = {
        train_0: ["10.0.0.1:8000"],
        train_1: ["10.0.0.2:8000"],
        **{u: [f"10.0.1.{i}:8000"] for i, u in enumerate(dst_units)},
    }
    worker_rpc_addresses = {
        train_0: "10.0.0.1:9000",
        train_1: "10.0.0.2:9000",
        **{u: f"10.0.1.{i}:9000" for i, u in enumerate(dst_units)},
    }

    # Two trainers each contribute half of a 1024-byte block (512 bytes each)
    # 9-tuple: (local_dst_idx, dst_block_offset, src_block_offset, size, src_block_id, dst_block_id, src_stride, dst_stride, count)
    canonical_var_plans = {
        train_0: {100: {0: [(0, 0, 0, 512, 0, 0, 512, 512, 1)]}},
        train_1: {100: {0: [(0, 512, 0, 512, 0, 0, 512, 512, 1)]}},
    }
    # Relay spans (min_offset, block_size, dst_block_id); relays collapse them
    # into one whole-shard push.
    canonical_relay = {100: {0: [(0, 512, 0), (512, 512, 0)]}}

    stage_group = controller_types.StageBroadcastGroup(
        pool_group=0,
        layer_group_idx=0,
        src_units=src_units,
        dst_units=dst_units,
        stage_ordered_vars_by_unit={
            train_0: [(0, 100)],
            train_1: [(0, 100)],
        },
        canonical_variable_plans=canonical_var_plans,
        canonical_relay_plans=canonical_relay,
        data_addresses=data_addresses,
    )

    final_plan = raiden_controller.TransferPlan(
        src_units=src_units,
        dst_units=dst_units,
        plan=None,
        worker_data_addresses=data_addresses,
        worker_rpc_addresses=worker_rpc_addresses,
        is_weight_sync=True,
    )

    asyncio.run(
        engine.execute_slice_broadcast_pipeline(
            groups_list=[stage_group],
            final_plan=final_plan,
            n_seed=2,
            req_id="req_stage_relay_test",
            dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
            registered_shards={
                u: data_addresses[u] for u in src_units + dst_units
            },
        )
    )

    # 1. Verify Trainer -> Seed Samplers (dst_units[0] and dst_units[1])
    # Both train_0 and train_1 must have pushed to each seed sampler.
    for seed in (dst_units[0], dst_units[1]):
      seed_receiver_calls = [
          plan
          for target_id, plan in rpc_client.invocations
          if target_id == seed and set(plan.src_units) == set(src_units)
      ]
      self.assertLen(seed_receiver_calls, 1)
      self.assertEqual(seed_receiver_calls[0].expected_block_count, 2)
      # Seeds tile their host buffers in place as layers arrive.
      self.assertEqual(
          seed_receiver_calls[0].host_tiling_mode,
          raiden_service_pb2.HOST_TILING_MODE_ON_ARRIVAL,
      )

    # 2. Verify Sampler -> Sampler Relay transfers
    # With n_seed=2 and 4 samplers:
    # dst_units[0] and dst_units[1] relay to dst_units[2] and dst_units[3].
    relay_sender_calls = [
        plan
        for target_id, plan in rpc_client.invocations
        if target_id in dst_units and plan.src_units[0] in dst_units
    ]
    self.assertNotEmpty(relay_sender_calls)
    for relay_plan in relay_sender_calls:
      # Relays forward the seeds' tiled shards as is.
      self.assertEqual(
          relay_plan.host_tiling_mode,
          raiden_service_pb2.HOST_TILING_MODE_PRE_TILED,
      )
      self.assertEqual(relay_plan.expected_layer_chunk_counts, {0: 1})
      s_u = relay_plan.src_units[0]
      sched = relay_plan.shard_push_schedules[s_u]
      for s_idx, entries in sched.items():
        self.assertLen(entries, 1)
        for entry in entries:
          (
              dst_peer,
              local_dst_idx,
              dst_block_offset,
              src_block_offset,
              size,
              src_block_id,
              dst_block_id,
              src_stride,
              dst_stride,
              count,
              layer_idx,
              pool_group,
          ) = entry
          # Verify 1-to-1 whole-block relay invariants:
          self.assertEqual(count, 1, "Relay transfers must have count = 1")
          self.assertEqual(size, 1024, "Whole-block size must be 1024")
          self.assertEqual(dst_block_offset, 0)
          self.assertEqual(src_block_offset, 0)
          self.assertEqual(src_block_id, 0)
          self.assertEqual(dst_block_id, 0)
          self.assertEqual(src_stride, 1024)
          self.assertEqual(dst_stride, 1024)

    # 3. Verify all 4 samplers received the stage
    receivers = {
        plan.dst_units[0]
        for target_id, plan in rpc_client.invocations
        if target_id == plan.dst_units[0] and target_id in dst_units
    }
    self.assertEqual(receivers, set(dst_units))

    # 4. Relay receivers get pre-tiled data and do not tile it again.
    for leaf in (dst_units[2], dst_units[3]):
      leaf_receiver_calls = [
          plan
          for target_id, plan in rpc_client.invocations
          if target_id == leaf and plan.dst_units[0] == leaf
      ]
      self.assertNotEmpty(leaf_receiver_calls)
      for plan in leaf_receiver_calls:
        self.assertEqual(
            plan.host_tiling_mode,
            raiden_service_pb2.HOST_TILING_MODE_PRE_TILED,
        )

  def test_all_source_binomial_tree_population_and_destinations(self) -> None:
    """Verifies population counts across rounds and round-grouped destination logging."""
    rpc_client = RecordingWorkerRpcClient()
    self.addCleanup(rpc_client.close)
    engine = broadcast_engine.BroadcastEngine(rpc_client)

    src = RaidenId(job_name="trainer", job_replica_id="0", data_name="w")
    # 7 samplers with n_seed=1 -> Round 0: 1, Round 1: 3, Round 2: 7
    dsts = [
        RaidenId(job_name="sampler", job_replica_id=str(i), data_name="w")
        for i in range(7)
    ]

    key = (src, 0, 0, 0, 1024, 0, 1, 0, 0)
    targets = [(dsts[i], f"127.0.0.1:800{i}", 0, 0, 0, 0) for i in range(7)]

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
            n_seed=1,
            req_id="req_tree_audit",
            dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
            registered_shards=registered_shards,
        )
    )

    # Verify all 7 destinations received the transfer
    dst_received = {
        plan.dst_units[0]
        for target_id, plan in rpc_client.invocations
        if target_id == plan.dst_units[0]
    }
    self.assertEqual(dst_received, set(dsts))

    # Verify every dispatched sub_plan has broadcast_round and broadcast_round_destinations
    for target_id, plan in rpc_client.invocations:
      self.assertIsNotNone(plan.broadcast_round)
      self.assertIn(plan.broadcast_round, (0, 1, 2))
      self.assertNotEmpty(plan.broadcast_round_destinations)
      for rd in plan.broadcast_round_destinations:
        self.assertGreaterEqual(rd.round_idx, 0)
        self.assertNotEmpty(rd.dst_units)
        self.assertNotEmpty(rd.dst_peers)

  def test_all_source_binomial_tree_newest_first_partial_round(self) -> None:
    """Verifies newest-first parent assignment on partial final rounds (N=5, n_seed=1)."""
    rpc_client = RecordingWorkerRpcClient()
    self.addCleanup(rpc_client.close)
    engine = broadcast_engine.BroadcastEngine(rpc_client)

    src = RaidenId(job_name="trainer", job_replica_id="0", data_name="w")
    dsts = [
        RaidenId(job_name="sampler", job_replica_id=str(i), data_name="w")
        for i in range(5)
    ]

    key = (src, 0, 0, 0, 1024, 0, 1, 0, 0)
    targets = [(dsts[i], f"127.0.0.1:800{i}", 0, 0, 0, 0) for i in range(5)]

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
            n_seed=1,
            req_id="req_partial_audit",
            dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
            registered_shards=registered_shards,
        )
    )

    receiver_to_sender = {}
    for _, plan in rpc_client.invocations:
      if plan.is_sender:
        receiver_to_sender[plan.dst_units[0]] = plan.src_units[0]

    # Round 0: Trainer -> dsts[0]
    self.assertEqual(receiver_to_sender[dsts[0]], src)
    # Round 1: dsts[0] -> dsts[1], Trainer -> dsts[2]
    self.assertEqual(receiver_to_sender[dsts[1]], dsts[0])
    self.assertEqual(receiver_to_sender[dsts[2]], src)
    # Round 2: Newest first: dsts[2] -> dsts[3], dsts[1] -> dsts[4]
    self.assertEqual(receiver_to_sender[dsts[3]], dsts[2])
    self.assertEqual(receiver_to_sender[dsts[4]], dsts[1])

    # The slice-list path never tiles host buffers in place.
    for _, plan in rpc_client.invocations:
      self.assertEqual(
          plan.host_tiling_mode, raiden_service_pb2.HOST_TILING_MODE_UNSPECIFIED
      )

  def test_stream_ordered_multi_chunk_dispatch(self) -> None:
    """Verifies (child_order, g_idx) stream ordering across 3 binomial rounds with M=3 chunks."""
    src = RaidenId(job_name="trainer", job_replica_id="0", data_name="w")
    dsts = [
        RaidenId(job_name="sampler", job_replica_id=str(i), data_name="w")
        for i in range(7)
    ]

    dispatches_by_sender: dict[RaidenId, list[tuple[RaidenId, int]]] = (
        collections.defaultdict(list)
    )

    class OrderingRpcClient(raiden_controller.WeightSyncWorkerRpcClient):

      async def start_transfer(
          self,
          target_id: RaidenId,
          transfer_plan: Any,
          address: Optional[str] = None,
      ) -> None:
        del address
        sender = transfer_plan.src_units[0]
        receiver = transfer_plan.dst_units[0]
        if target_id == sender:
          g_idx = int(
              transfer_plan.req_id.split("reqordered_")[1].split("_")[0]
          )
          dispatches_by_sender[sender].append((receiver, g_idx))

    rpc_client = OrderingRpcClient()
    self.addCleanup(rpc_client.close)
    engine = broadcast_engine.BroadcastEngine(rpc_client)

    groups_list = []
    for g_idx in range(3):
      key = (src, 0, 0, 0, 1024, 0, 1, g_idx, 0)
      targets = [(dsts[i], f"127.0.0.1:800{i}", 0, 0, 0, 0) for i in range(7)]
      groups_list.append([(key, targets)])

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
            groups_list=groups_list,
            final_plan=final_plan,
            n_seed=1,
            req_id="reqordered",
            dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
            registered_shards=registered_shards,
            pipeline_target_stages=10,
        )
    )

    # Trainer dispatches (t0, g0), (t0, g1), (t0, g2) before (t2, g0), (t2, g1), (t2, g2)
    # and before (t6, g0), (t6, g1), (t6, g2).
    trainer_dispatches = dispatches_by_sender[src]
    t0_pushes = [(dsts[0], 0), (dsts[0], 1), (dsts[0], 2)]
    t2_pushes = [(dsts[2], 0), (dsts[2], 1), (dsts[2], 2)]
    t6_pushes = [(dsts[6], 0), (dsts[6], 1), (dsts[6], 2)]

    for p in t0_pushes + t2_pushes + t6_pushes:
      self.assertIn(p, trainer_dispatches)

    t0_indices = [trainer_dispatches.index(p) for p in t0_pushes]
    t2_indices = [trainer_dispatches.index(p) for p in t2_pushes]
    t6_indices = [trainer_dispatches.index(p) for p in t6_pushes]

    self.assertLess(max(t0_indices), min(t2_indices))
    self.assertLess(max(t2_indices), min(t6_indices))

    # t0 dispatches (t1, g0), (t1, g1), (t1, g2) before (t5, g0), (t5, g1), (t5, g2)
    t0_dispatches = dispatches_by_sender[dsts[0]]
    t1_pushes = [(dsts[1], 0), (dsts[1], 1), (dsts[1], 2)]
    t5_pushes = [(dsts[5], 0), (dsts[5], 1), (dsts[5], 2)]

    for p in t1_pushes + t5_pushes:
      self.assertIn(p, t0_dispatches)

    t1_indices = [t0_dispatches.index(p) for p in t1_pushes]
    t5_indices = [t0_dispatches.index(p) for p in t5_pushes]

    self.assertLess(max(t1_indices), min(t5_indices))

  def test_full_duplex_chunk_pipeline_overlap(self) -> None:
    """Verifies t0 relays g0 to t1 while Trainer concurrently sends g1 to t0."""
    src = RaidenId(job_name="trainer", job_replica_id="0", data_name="w")
    dsts = [
        RaidenId(job_name="sampler", job_replica_id=str(i), data_name="w")
        for i in range(7)
    ]

    class OverlapRpcClient(raiden_controller.WeightSyncWorkerRpcClient):

      def __init__(self) -> None:
        super().__init__()
        self.t0_relaying_g0_to_t1 = asyncio.Event()
        self.trainer_sending_g1_to_t0 = asyncio.Event()
        self.full_duplex_overlap_observed = False
        self.allow_g1_to_complete = asyncio.Event()

      async def start_transfer(
          self,
          target_id: RaidenId,
          transfer_plan: Any,
          address: Optional[str] = None,
      ) -> None:
        del address
        sender = transfer_plan.src_units[0]
        receiver = transfer_plan.dst_units[0]
        is_sender_call = target_id == sender

        # Detect Trainer -> t0 for Chunk 1
        if (
            is_sender_call
            and sender == src
            and receiver == dsts[0]
            and "_1_" in transfer_plan.req_id
        ):
          self.trainer_sending_g1_to_t0.set()
          try:
            await asyncio.wait_for(
                self.t0_relaying_g0_to_t1.wait(), timeout=5.0
            )
            self.full_duplex_overlap_observed = True
          finally:
            self.allow_g1_to_complete.set()

        # Detect t0 -> t1 for Chunk 0
        if (
            is_sender_call
            and sender == dsts[0]
            and receiver == dsts[1]
            and "_0_" in transfer_plan.req_id
        ):
          self.t0_relaying_g0_to_t1.set()

    rpc_client = OverlapRpcClient()
    self.addCleanup(rpc_client.close)
    engine = broadcast_engine.BroadcastEngine(rpc_client)

    groups_list = []
    for g_idx in range(2):
      key = (src, 0, 0, 0, 1024, 0, 1, g_idx, 0)
      targets = [(dsts[i], f"127.0.0.1:800{i}", 0, 0, 0, 0) for i in range(7)]
      groups_list.append([(key, targets)])

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
            groups_list=groups_list,
            final_plan=final_plan,
            n_seed=1,
            req_id="reqoverlap",
            dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
            registered_shards=registered_shards,
            pipeline_target_stages=10,
        )
    )

    self.assertTrue(
        rpc_client.full_duplex_overlap_observed,
        "t0 must start relaying g0 to t2 concurrently while Trainer sends g1 to"
        " t0 (full-duplex pipelining).",
    )

  def test_coalesce_granularity_and_pipeline_validation(self) -> None:
    """Verifies configurable target_stages and max_chunk_bytes plus validations."""
    src = RaidenId(job_name="src", job_replica_id="0", data_name="w")
    dsts = [
        RaidenId(job_name="dst", job_replica_id=str(i), data_name="w")
        for i in range(4)
    ]
    targets = [(dsts[i], f"127.0.0.1:800{i}", 0, 0, 0, 0) for i in range(4)]
    groups_list = [
        [((src, 0, 0, 0, 1024, 0, 1, layer_idx, 0), targets)]
        for layer_idx in range(40)
    ]

    # Valid coalescing to 4 stages
    coalesced = broadcast_engine._coalesce_pipeline_groups(
        groups_list, target_stages=4
    )
    self.assertLen(coalesced, 4)
    for stage in coalesced:
      self.assertLen(stage, 10)

    # Validation on _coalesce_pipeline_groups
    with self.assertRaisesRegex(ValueError, "target_stages must be >= 1"):
      broadcast_engine._coalesce_pipeline_groups(groups_list, target_stages=0)
    with self.assertRaisesRegex(ValueError, "target_stages must be >= 1"):
      broadcast_engine._coalesce_pipeline_groups(groups_list, target_stages=-2)

    # Validation on _coalesce_contiguous_relay_entries
    with self.assertRaisesRegex(ValueError, "max_chunk_bytes must be > 0"):
      broadcast_engine._coalesce_contiguous_relay_entries([], max_chunk_bytes=0)
    with self.assertRaisesRegex(ValueError, "max_chunk_bytes must be > 0"):
      broadcast_engine._coalesce_contiguous_relay_entries(
          [], max_chunk_bytes=-1
      )

    # Validation on execute_slice_broadcast_pipeline
    engine = broadcast_engine.BroadcastEngine(RecordingWorkerRpcClient())
    final_plan = raiden_controller.TransferPlan(
        src_units=[src], dst_units=dsts, plan=None
    )
    with self.assertRaisesRegex(
        ValueError, "pipeline_target_stages must be >= 1"
    ):
      asyncio.run(
          engine.execute_slice_broadcast_pipeline(
              groups_list=groups_list,
              final_plan=final_plan,
              n_seed=1,
              req_id="req_val",
              dst_mem_type=0,
              registered_shards={},
              pipeline_target_stages=0,
          )
      )
    with self.assertRaisesRegex(ValueError, "max_chunk_bytes must be > 0"):
      asyncio.run(
          engine.execute_slice_broadcast_pipeline(
              groups_list=groups_list,
              final_plan=final_plan,
              n_seed=1,
              req_id="req_val",
              dst_mem_type=0,
              registered_shards={},
              max_chunk_bytes=0,
          )
      )

  def test_execute_slice_broadcast_multistage_tree_propagates_parallelism(
      self,
  ) -> None:
    """Verifies that all sender hop plans in a multi-stage tree inherit parallelism."""
    for expected_parallelism in (4, 0):
      with self.subTest(parallelism=expected_parallelism):
        rpc_client = RecordingWorkerRpcClient()
        self.addCleanup(rpc_client.close)
        engine = broadcast_engine.BroadcastEngine(rpc_client)

        src = RaidenId(job_name="src", job_replica_id="0", data_name="w")
        relay = RaidenId(job_name="relay", job_replica_id="0", data_name="w")
        dst = RaidenId(job_name="dst", job_replica_id="0", data_name="w")

        key0 = (src, 0, 0, 0, 1024, 0, 1, 0, 0)
        key1 = (src, 0, 1, 0, 1024, 0, 1, 0, 0)
        targets0: list[tuple[Any, ...]] = [
            (relay, "127.0.0.1:8001", 0, 0, 0, 0),
            (dst, "127.0.0.1:8002", 0, 0, 0, 0),
        ]
        targets1: list[tuple[Any, ...]] = [
            (relay, "127.0.0.1:8001", 1, 1, 0, 0),
            (dst, "127.0.0.1:8002", 1, 1, 0, 0),
        ]

        final_plan = raiden_controller.TransferPlan(
            src_units=[src],
            dst_units=[relay, dst],
            plan=None,
            worker_data_addresses={
                src: ["127.0.0.1:8000"],
                relay: ["127.0.0.1:8001"],
                dst: ["127.0.0.1:8002"],
            },
            is_weight_sync=True,
            parallelism=expected_parallelism,
            uuid=101,
        )
        registered_shards = {
            src: ["s0"],
            relay: ["s0", "s1"],
            dst: ["s0", "s1"],
        }

        asyncio.run(
            engine.execute_slice_broadcast(
                keys_and_targets=[(key0, targets0), (key1, targets1)],
                final_plan=final_plan,
                n_seed=1,
                req_id=f"req_parallelism_{expected_parallelism}",
                dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
                registered_shards=registered_shards,
            )
        )

        seed_sender_plans = [
            plan
            for target_id, plan in rpc_client.invocations
            if target_id == src and plan.src_units[0] == src and plan.is_sender
        ]
        relay_sender_plans = [
            plan
            for target_id, plan in rpc_client.invocations
            if target_id == relay
            and plan.src_units[0] == relay
            and plan.is_sender
        ]

        self.assertNotEmpty(seed_sender_plans)
        self.assertNotEmpty(relay_sender_plans)

        for plan in seed_sender_plans:
          self.assertEqual(
              plan.parallelism,
              expected_parallelism,
              "Seed sender hop plan must carry"
              f" parallelism={expected_parallelism}",
          )

        for plan in relay_sender_plans:
          self.assertEqual(
              plan.parallelism,
              expected_parallelism,
              "Relay sender hop plan must carry"
              f" parallelism={expected_parallelism}",
          )

  def test_stage_broadcast_group_trainer_seed_latin_square_circular_shift(
      self,
  ) -> None:
    """Verifies Round 0 trainer seeds push to samplers in Latin Square circular shift."""
    dst_units = [RaidenId("sampler", str(j), "weights", 0) for j in range(8)]
    dst_shards = {u: [f"10.1.0.{j + 1}:8000"] for j, u in enumerate(dst_units)}

    src_var = controller_types.VariableMetadata(
        name="weight",
        shape=[4096, 4096],
        mesh_shape=[4, 1],
        layout=[1, 0],
        item_size=2,
        layer_idx=0,
        sharding_spec=["fsdp", ""],
    )
    dst_var = controller_types.VariableMetadata(
        name="weight",
        shape=[4096, 4096],
        mesh_shape=[1, 1],
        layout=[1, 0],
        item_size=2,
        layer_idx=0,
        sharding_spec=["", ""],
    )

    for mode in ("multi_unit_trainer", "pathways_single_unit_trainer"):
      with self.subTest(mode=mode):
        if mode == "multi_unit_trainer":
          src_units = [
              RaidenId("trainer", str(h), "weights", 0) for h in range(4)
          ]
          registered_shards = {
              u: [f"10.0.0.{h + 1}:8000"] for h, u in enumerate(src_units)
          }
          worker_endpoints = {
              u: f"10.0.0.{h + 1}:9000" for h, u in enumerate(src_units)
          }
          entities = {
              u: job_entity.JobEntity(unit=u, shards=registered_shards[u])
              for u in src_units
          }
          registered_variables = {u: [src_var] for u in src_units}
          registered_mesh_shapes = {u: [4, 1] for u in src_units}
        else:
          src_unit = RaidenId("trainer", "", "weights", 0)
          src_units = [src_unit]
          src_shards = [f"10.0.0.{h + 1}:8000" for h in range(4)]
          registered_shards = {src_unit: src_shards}
          worker_endpoints = {
              src_unit: (
                  "10.0.0.1:9000,10.0.0.2:9000,10.0.0.3:9000,10.0.0.4:9000"
              )
          }
          entities = {
              src_unit: job_entity.JobEntity(unit=src_unit, shards=src_shards)
          }
          registered_variables = {src_unit: [src_var]}
          registered_mesh_shapes = {src_unit: [4, 1]}

        registered_shards.update(dst_shards)
        dst_metadata = []
        for j, u in enumerate(dst_units):
          worker_endpoints[u] = f"10.1.0.{j + 1}:9000"
          meta = raiden_service_pb2.RegisterWorkUnitRequest(
              unit=raiden_service_pb2.RaidenIdProto(
                  job_name=u.job_name,
                  job_replica_id=str(u.job_replica_id),
                  data_name=u.data_name,
                  data_replica_idx=u.data_replica_idx,
              ),
              control_plane_rpc_address=f"10.1.0.{j + 1}:9000",
          )
          meta.shards.extend(dst_shards[u])
          meta.mesh_shape.extend([1, 1])
          meta.mesh_axes.extend(["tp", "tp_wo"])
          vp = meta.variables.add()
          vp.name = dst_var.name
          vp.shape.extend(dst_var.shape)
          vp.mesh_shape.extend(dst_var.mesh_shape)
          vp.layout.extend(dst_var.layout)
          vp.item_size = dst_var.item_size
          vp.layer_idx = dst_var.layer_idx
          vp.sharding_spec.extend(dst_var.sharding_spec)
          dst_metadata.append(meta)

        sched = reshard_planner.ReshardPlanner.compute_transfer_schedule_from_metadata(
            src_units=src_units,
            dst_units=dst_units,
            dst_metadata=dst_metadata,
            entities=entities,
            registered_variables=registered_variables,
            registered_global_shapes={},
            registered_mesh_shapes=registered_mesh_shapes,
            registered_layouts={},
            registered_itemsizes={},
            registered_shards=registered_shards,
            computed_phys_meshes={},
            worker_endpoints=worker_endpoints,
            broadcast_host_ratio=1.0,
            lock=threading.Lock(),
        )
        self.assertEqual(sched.n_seed, 4)
        self.assertNotEmpty(sched.broadcast_groups)

        rpc_client = RecordingWorkerRpcClient()
        self.addCleanup(rpc_client.close)
        engine = broadcast_engine.BroadcastEngine(rpc_client)

        final_plan = raiden_controller.TransferPlan(
            src_units=src_units,
            dst_units=dst_units,
            plan=None,
            worker_data_addresses=sched.data_addresses,
            worker_rpc_addresses=sched.rpc_addresses,
            is_weight_sync=True,
        )

        asyncio.run(
            engine.execute_slice_broadcast_pipeline(
                groups_list=list(sched.broadcast_groups.values()),
                final_plan=final_plan,
                n_seed=sched.n_seed,
                req_id="req_stage_latin_square",
                dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
                registered_shards=dict(sched.data_addresses),
            )
        )

        # 1. Verify all 4 Round-0 seed samplers (sampler:0..3) receive
        # receiver plans.
        for seed_idx in range(4):
          seed = dst_units[seed_idx]
          recv_plans = [
              plan
              for target_id, plan in rpc_client.invocations
              if target_id == seed and not plan.is_sender
          ]
          self.assertNotEmpty(
              recv_plans,
              f"Seed sampler {seed} must receive receiver plan",
          )

        # Verify all 4 Round-1 relay samplers (sampler:4..7) receive whole-block
        # relay transfers from sampler:0..3.
        for relay_idx in range(4, 8):
          relay_dst = dst_units[relay_idx]
          recv_plans = [
              plan
              for target_id, plan in rpc_client.invocations
              if target_id == relay_dst
              and plan.dst_units[0] == relay_dst
              and plan.src_units[0] in dst_units[:4]
          ]
          self.assertNotEmpty(
              recv_plans,
              f"Relay sampler {relay_dst} must receive relay transfer from seed"
              " samplers",
          )

        relay_sender_plans = [
            plan
            for target_id, plan in rpc_client.invocations
            if target_id in dst_units[:4] and plan.is_sender
        ]
        self.assertNotEmpty(relay_sender_plans)
        for r_plan in relay_sender_plans:
          s_u = r_plan.src_units[0]
          s_sched = r_plan.shard_push_schedules[s_u]
          for entries in s_sched.values():
            for entry in entries:
              self.assertEqual(
                  entry[9], 1, "Relay transfers must have count = 1"
              )

        # 2. In Round 0, for each of the 4 trainer hosts h in {0, 1, 2, 3},
        # the trainer sender schedule pushes to all 4 Round-0 seed samplers
        # in the circularly shifted Latin Square order
        # [sampler:h, sampler:(h+1)%4, sampler:(h+2)%4, sampler:(h+3)%4].
        first_seed_by_host = {}
        for h in range(4):
          expected_seeds = [dst_units[(h + i) % 4] for i in range(4)]
          if mode == "multi_unit_trainer":
            u = src_units[h]
            u_sender_plans = [
                plan
                for target_id, plan in rpc_client.invocations
                if target_id == u and plan.is_sender
            ]
            self.assertNotEmpty(u_sender_plans)
            sender_plan = u_sender_plans[0]
            self.assertNotEmpty(
                sender_plan.broadcast_round_destinations,
                f"Trainer unit {u} must have non-empty"
                " broadcast_round_destinations",
            )
            sched_map = sender_plan.shard_push_schedules[u]
            entries = sched_map[0]
            dst_peers = [e[0] for e in entries]
            dst_units_order = [sched.data_address_to_unit[p] for p in dst_peers]
            self.assertEqual(
                dst_units_order,
                expected_seeds,
                f"Trainer unit {u} (host {h}) must push seeds in Latin Square"
                " order",
            )
            protos = entities[u].build_sender_push_schedule_protos(sched_map)
            proto_dst_peers = list(protos[0].entries[0].dst_peers)
            proto_dst_units = [
                sched.data_address_to_unit[p] for p in proto_dst_peers
            ]
            self.assertEqual(proto_dst_units, expected_seeds)
            first_seed_by_host[h] = proto_dst_units[0]
          else:
            u = src_units[0]
            u_sender_plans = [
                plan
                for target_id, plan in rpc_client.invocations
                if target_id == u and plan.is_sender
            ]
            self.assertNotEmpty(u_sender_plans)
            sender_plan = u_sender_plans[0]
            sched_map = sender_plan.shard_push_schedules[u]
            entries = sched_map[h]
            dst_peers = [e[0] for e in entries]
            dst_units_order = [sched.data_address_to_unit[p] for p in dst_peers]
            self.assertEqual(
                dst_units_order,
                expected_seeds,
                f"Trainer shard {h} (host {h}) must push seeds in Latin Square"
                " order",
            )
            protos = entities[u].build_sender_push_schedule_protos(sched_map)
            proto_dst_peers = list(protos[h].entries[0].dst_peers)
            proto_dst_units = [
                sched.data_address_to_unit[p] for p in proto_dst_peers
            ]
            self.assertEqual(proto_dst_units, expected_seeds)
            first_seed_by_host[h] = proto_dst_units[0]

        self.assertLen(
            set(first_seed_by_host.values()),
            4,
            "All 4 trainer hosts must start with distinct seed samplers",
        )

  def test_stage_broadcast_group_multistage_cumulative_receiver_counts(
      self,
  ) -> None:
    """Verifies multi-stage StageBroadcastGroup arms seed and relay receivers with full-transfer block and layer chunk counts."""
    rpc_client = RecordingWorkerRpcClient()
    self.addCleanup(rpc_client.close)
    engine = broadcast_engine.BroadcastEngine(worker_rpc_client=rpc_client)

    train_0 = RaidenId("trainer", "0", "weights", 0)
    train_1 = RaidenId("trainer", "1", "weights", 0)
    src_units = [train_0, train_1]
    dst_units = [RaidenId("sampler", str(i), "weights", 0) for i in range(4)]

    data_addresses = {
        train_0: ["10.0.0.1:8000"],
        train_1: ["10.0.0.2:8000"],
        **{
            u: [f"10.0.1.{i}:8000", f"10.0.1.{i}:8001"]
            for i, u in enumerate(dst_units)
        },
    }
    worker_rpc_addresses = {
        train_0: "10.0.0.1:9000",
        train_1: "10.0.0.2:9000",
        **{u: f"10.0.1.{i}:9000" for i, u in enumerate(dst_units)},
    }

    # Each trainer pushes 1 slice to dst shard 0 and 1 slice to dst shard 1
    # (4 slices per layer for seed samplers, 2 whole blocks per layer for relay samplers)
    canonical_var_plans = {
        train_0: {
            100: {
                0: [
                    (0, 0, 0, 512, 0, 0, 512, 512, 1),
                    (1, 0, 0, 512, 1, 1, 512, 512, 1),
                ]
            }
        },
        train_1: {
            100: {
                0: [
                    (0, 512, 0, 512, 0, 0, 512, 512, 1),
                    (1, 512, 0, 512, 1, 1, 512, 512, 1),
                ]
            }
        },
    }
    canonical_relay = {
        100: {
            0: [(0, 1024, 0)],
            1: [(0, 1024, 1)],
        }
    }

    # 3 stages: Stage 0 (layers 0, 1), Stage 1 (layers 2, 3), Stage 2 (layers 4, 5)
    groups_list = []
    for stage_idx, layers in enumerate([(0, 1), (2, 3), (4, 5)]):
      groups_list.append(
          controller_types.StageBroadcastGroup(
              pool_group=0,
              layer_group_idx=stage_idx,
              src_units=src_units,
              dst_units=dst_units,
              stage_ordered_vars_by_unit={
                  train_0: [(l, 100) for l in layers],
                  train_1: [(l, 100) for l in layers],
              },
              canonical_variable_plans=canonical_var_plans,
              canonical_relay_plans=canonical_relay,
              data_addresses=data_addresses,
          )
      )

    final_plan = raiden_controller.TransferPlan(
        src_units=src_units,
        dst_units=dst_units,
        plan=None,
        worker_data_addresses=data_addresses,
        worker_rpc_addresses=worker_rpc_addresses,
        uuid=777,
        is_weight_sync=True,
    )

    asyncio.run(
        engine.execute_slice_broadcast_pipeline(
            groups_list=groups_list,
            final_plan=final_plan,
            n_seed=2,
            req_id="req_multistage_sbg",
            dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
            registered_shards={
                u: data_addresses[u] for u in src_units + dst_units
            },
            pipeline_target_stages=3,
        )
    )

    # Seed samplers (dst_units[0], dst_units[1]):
    # 6 layers total * 4 chunks/layer = 24 total chunks
    expected_seed_layers = {l: 4 for l in range(6)}
    for i in (0, 1):
      seed = dst_units[i]
      host_ip = f"10.0.1.{i}"
      seed_recv_plans = [
          plan
          for target_id, plan in rpc_client.invocations
          if target_id == seed and not plan.is_sender
      ]
      self.assertLen(seed_recv_plans, 3)
      for plan in seed_recv_plans:
        self.assertEqual(plan.uuid, 777)
        self.assertEqual(plan.expected_block_count, 24)
        self.assertEqual(plan.dst_expected_block_counts[seed], 24)
        self.assertEqual(plan.expected_layer_chunk_counts, expected_seed_layers)
        self.assertEqual(
            plan.dst_expected_layer_chunk_counts[seed], expected_seed_layers
        )
        self.assertEqual(plan.dst_endpoint_counts, {host_ip: 24})
        self.assertEqual(
            plan.dst_endpoint_layer_counts, {host_ip: expected_seed_layers}
        )

    # Relay samplers (dst_units[2], dst_units[3]):
    # 6 layers total * 2 whole blocks/layer (1 per shard) = 12 total chunks
    expected_relay_layers = {l: 2 for l in range(6)}
    for i in (2, 3):
      relay_dst = dst_units[i]
      host_ip = f"10.0.1.{i}"
      relay_recv_plans = [
          plan
          for target_id, plan in rpc_client.invocations
          if target_id == relay_dst and plan.dst_units == [relay_dst]
      ]
      self.assertLen(relay_recv_plans, 3)
      for plan in relay_recv_plans:
        self.assertEqual(plan.uuid, 777)
        self.assertEqual(plan.expected_block_count, 12)
        self.assertEqual(plan.dst_expected_block_counts[relay_dst], 12)
        self.assertEqual(
            plan.expected_layer_chunk_counts, expected_relay_layers
        )
        self.assertEqual(
            plan.dst_expected_layer_chunk_counts[relay_dst],
            expected_relay_layers,
        )
        self.assertEqual(plan.dst_endpoint_counts, {host_ip: 12})
        self.assertEqual(
            plan.dst_endpoint_layer_counts, {host_ip: expected_relay_layers}
        )

  def test_slice_broadcast_pipeline_multistage_cumulative_receiver_counts(
      self,
  ) -> None:
    """Verifies legacy slice-list multi-stage pipeline arms seed and relay receivers with full-transfer block and layer counts."""
    rpc_client = RecordingWorkerRpcClient()
    self.addCleanup(rpc_client.close)
    engine = broadcast_engine.BroadcastEngine(worker_rpc_client=rpc_client)

    src = RaidenId(job_name="src", job_replica_id="0", data_name="w")
    seed = RaidenId(job_name="sampler", job_replica_id="0", data_name="w")
    relay = RaidenId(job_name="sampler", job_replica_id="1", data_name="w")

    # 2 stages, each with 1 layer having 2 contiguous 512B slices on shard 0:
    # Seed receives 2 slices per layer (4 total across 2 layers);
    # Relay coalesces the 2 contiguous slices into 1 chunk per layer (2 total across 2 layers).
    groups_list = []
    for layer_idx in (0, 1):
      key_a = (src, 0, layer_idx, 0, 512, 0, 1, layer_idx, 0)
      key_b = (src, 0, layer_idx, 512, 512, 0, 1, layer_idx, 0)
      targets_a = [
          (seed, "10.0.2.0:8000", 0, layer_idx, 0, 0),
          (relay, "10.0.2.1:8000", 0, layer_idx, 0, 0),
      ]
      targets_b = [
          (seed, "10.0.2.0:8000", 0, layer_idx, 512, 0),
          (relay, "10.0.2.1:8000", 0, layer_idx, 512, 0),
      ]
      groups_list.append([(key_a, targets_a), (key_b, targets_b)])

    final_plan = raiden_controller.TransferPlan(
        src_units=[src],
        dst_units=[seed, relay],
        plan=None,
        worker_data_addresses={
            src: ["10.0.0.1:8000"],
            seed: ["10.0.2.0:8000"],
            relay: ["10.0.2.1:8000"],
        },
        uuid=888,
        is_weight_sync=True,
    )

    asyncio.run(
        engine.execute_slice_broadcast_pipeline(
            groups_list=groups_list,
            final_plan=final_plan,
            n_seed=1,
            req_id="req_multistage_slice",
            dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
            registered_shards={src: ["s0"], seed: ["s0"], relay: ["s0"]},
            pipeline_target_stages=2,
        )
    )

    seed_recv_plans = [
        plan
        for target_id, plan in rpc_client.invocations
        if target_id == seed and plan.dst_units == [seed]
    ]
    self.assertLen(seed_recv_plans, 2)
    for plan in seed_recv_plans:
      self.assertEqual(plan.uuid, 888)
      self.assertEqual(plan.expected_block_count, 4)
      self.assertEqual(plan.dst_expected_block_counts[seed], 4)
      self.assertEqual(plan.expected_layer_chunk_counts, {0: 2, 1: 2})
      self.assertEqual(plan.dst_endpoint_counts, {"10.0.2.0": 4})
      self.assertEqual(
          plan.dst_endpoint_layer_counts, {"10.0.2.0": {0: 2, 1: 2}}
      )

    relay_recv_plans = [
        plan
        for target_id, plan in rpc_client.invocations
        if target_id == relay and plan.dst_units == [relay]
    ]
    self.assertLen(relay_recv_plans, 2)
    for plan in relay_recv_plans:
      self.assertEqual(plan.uuid, 888)
      self.assertEqual(plan.expected_block_count, 2)
      self.assertEqual(plan.dst_expected_block_counts[relay], 2)
      self.assertEqual(plan.expected_layer_chunk_counts, {0: 1, 1: 1})
      self.assertEqual(plan.dst_endpoint_counts, {"10.0.2.1": 2})
      self.assertEqual(
          plan.dst_endpoint_layer_counts, {"10.0.2.1": {0: 1, 1: 1}}
      )

  def test_multi_equivalence_class_trainer_seeding_across_rounds(self) -> None:
    """Verifies Trainer seeds n_seed replicas per equivalence class in Rounds 0 and 1+."""
    rpc_client = RecordingWorkerRpcClient()
    self.addCleanup(rpc_client.close)
    engine = broadcast_engine.BroadcastEngine(rpc_client)

    src_0 = RaidenId(job_name="trainer", job_replica_id="0", data_name="w")
    src_1 = RaidenId(job_name="trainer", job_replica_id="1", data_name="w")
    src_units = [src_0, src_1]
    eq0_units = [
        RaidenId(job_name=f"roll-{i}", job_replica_id="0", data_name="w")
        for i in range(6)
    ]
    eq1_units = [
        RaidenId(job_name=f"roll-{i}", job_replica_id="1", data_name="w")
        for i in range(6)
    ]
    all_dst_units = eq0_units + eq1_units

    data_addresses = {
        src_0: ["10.0.0.1:8000"],
        src_1: ["10.0.0.2:8000"],
    }
    for i, u in enumerate(eq0_units):
      data_addresses[u] = [f"10.0.1.{i}:8000"]
    for i, u in enumerate(eq1_units):
      data_addresses[u] = [f"10.0.2.{i}:8000"]

    canonical_vars = {
        src_0: {1: {0: [(0, 0, 0, 512, 0, 0, 512, 512, 1)]}},
        src_1: {1: {0: [(0, 512, 0, 512, 0, 0, 512, 512, 1)]}},
    }
    canonical_relays = {1: {0: [(0, 1024, 0)]}}

    sg_eq0 = controller_types.StageBroadcastGroup(
        pool_group=0,
        layer_group_idx=0,
        src_units=src_units,
        dst_units=eq0_units,
        stage_ordered_vars_by_unit={src_0: [(0, 1)], src_1: [(0, 1)]},
        canonical_variable_plans=canonical_vars,
        canonical_relay_plans=canonical_relays,
        data_addresses=data_addresses,
    )
    sg_eq1 = controller_types.StageBroadcastGroup(
        pool_group=1,
        layer_group_idx=0,
        src_units=src_units,
        dst_units=eq1_units,
        stage_ordered_vars_by_unit={src_0: [(0, 1)], src_1: [(0, 1)]},
        canonical_variable_plans=canonical_vars,
        canonical_relay_plans=canonical_relays,
        data_addresses=data_addresses,
    )

    final_plan = raiden_controller.TransferPlan(
        src_units=src_units,
        dst_units=all_dst_units,
        plan=None,
        worker_data_addresses=data_addresses,
        uuid=999,
        is_weight_sync=True,
    )

    asyncio.run(
        engine.execute_slice_broadcast_pipeline(
            groups_list=[sg_eq0, sg_eq1],
            final_plan=final_plan,
            n_seed=2,
            req_id="req_multi_eq_seed",
            dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
            registered_shards={u: ["s0"] for u in src_units + all_dst_units},
            pipeline_target_stages=1,
        )
    )

    # With n_seed=2 and 6 units per equivalence class:
    # Round 0: Trainer seeds 2 units per eq class (4 total).
    # Round 1: 2 Round-0 seeds relay to 2 units, and Trainer seeds the remaining
    #          2 units per eq class (4 total).
    trainer_seeded_by_round = {0: set(), 1: set()}
    for target_id, plan in rpc_client.invocations:
      if target_id == src_0 and plan.is_sender:
        trainer_seeded_by_round[plan.broadcast_round].update(plan.dst_units)

    self.assertLen(trainer_seeded_by_round[0], 4)
    self.assertLen(trainer_seeded_by_round[1], 4)

  def test_coalesced_stage_pool_group_isolation_and_stage_ordering(
      self,
  ) -> None:
    """Verifies coalesced stages get distinct pool_group values and interleave by stage_idx across eq classes."""
    rpc_client = RecordingWorkerRpcClient()
    self.addCleanup(rpc_client.close)
    engine = broadcast_engine.BroadcastEngine(rpc_client)

    src_0 = RaidenId(job_name="trainer", job_replica_id="0", data_name="w")
    eq0_units = [
        RaidenId(job_name="sampler", job_replica_id=f"0_{i}", data_name="w")
        for i in range(2)
    ]
    eq1_units = [
        RaidenId(job_name="sampler", job_replica_id=f"1_{i}", data_name="w")
        for i in range(2)
    ]
    all_dst_units = eq0_units + eq1_units

    data_addresses = {src_0: ["10.0.0.1:8000"]}
    for i, u in enumerate(eq0_units):
      data_addresses[u] = [f"10.0.1.{i}:8000"]
    for i, u in enumerate(eq1_units):
      data_addresses[u] = [f"10.0.2.{i}:8000"]

    canonical_vars = {
        src_0: {1: {0: [(0, 0, 0, 1024, 0, 0, 1024, 1024, 1)]}},
    }
    canonical_relays = {1: {0: [(0, 1024, 0)]}}

    # 4 layer groups per eq class, coalesced into target_stages=2 stages.
    groups_list = []
    for eq_idx, dsts in [(0, eq0_units), (1, eq1_units)]:
      for lg_idx in range(4):
        groups_list.append(
            controller_types.StageBroadcastGroup(
                pool_group=eq_idx,
                layer_group_idx=lg_idx,
                src_units=[src_0],
                dst_units=dsts,
                stage_ordered_vars_by_unit={src_0: [(lg_idx, 1)]},
                canonical_variable_plans=canonical_vars,
                canonical_relay_plans=canonical_relays,
                data_addresses=data_addresses,
            )
        )

    final_plan = raiden_controller.TransferPlan(
        src_units=[src_0],
        dst_units=all_dst_units,
        plan=None,
        worker_data_addresses=data_addresses,
        uuid=1001,
        is_weight_sync=True,
    )

    asyncio.run(
        engine.execute_slice_broadcast_pipeline(
            groups_list=groups_list,
            final_plan=final_plan,
            n_seed=1,
            req_id="req_stage_pg",
            dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
            registered_shards={u: ["s0"] for u in [src_0] + all_dst_units},
            pipeline_target_stages=2,
        )
    )

    sender_plans = [
        plan
        for target_id, plan in rpc_client.invocations
        if target_id == src_0 and plan.is_sender and plan.broadcast_round == 0
    ]
    self.assertLen(sender_plans, 4)

    # Extract (pool_group, first_layer_idx) in dispatch order for Round 0:
    # Stage 0 of eq0 (pg=0, layer=0) and Stage 0 of eq1 (pg=2, layer=0) must
    # dispatch before Stage 1 of eq0 (pg=1, l=2) and Stage 1 of eq1 (pg=3, l=2).
    dispatched_pg_and_layer = []
    for plan in sender_plans:
      sched = plan.shard_push_schedules[src_0][0]
      first_entry = sched[0]
      dispatched_pg_and_layer.append((first_entry[11], first_entry[10]))

    self.assertEqual(
        dispatched_pg_and_layer,
        [(0, 0), (2, 0), (1, 2), (3, 2)],
    )

  def test_stage_broadcast_group_reuses_cached_hop_schedules_across_rounds(
      self,
  ):
    """Verifies StageBroadcastGroup caches hop schedules across rounds."""
    rpc_client = RecordingWorkerRpcClient()
    self.addCleanup(rpc_client.close)
    engine = broadcast_engine.BroadcastEngine(rpc_client)

    src_0 = RaidenId(job_name="trainer", job_replica_id="0", data_name="w")
    dst_units = [
        RaidenId(job_name="sampler", job_replica_id=str(i), data_name="w")
        for i in range(4)
    ]
    data_addresses = {src_0: ["10.0.0.1:8000", "10.0.0.2:8000"]}
    for i, u in enumerate(dst_units):
      data_addresses[u] = [f"10.0.1.{i}:8000", f"10.0.1.{i}:8001"]

    canonical_vars = {
        src_0: {
            1: {
                0: [(0, 0, 0, 1024, 0, 0, 1024, 1024, 1)],
                1: [(1, 0, 0, 1024, 0, 0, 1024, 1024, 1)],
            }
        }
    }
    canonical_relays = {
        1: {
            0: [(0, 1024, 0)],
            1: [(0, 1024, 0)],
        }
    }
    stage_group = controller_types.StageBroadcastGroup(
        pool_group=0,
        layer_group_idx=0,
        src_units=[src_0],
        dst_units=dst_units,
        stage_ordered_vars_by_unit={src_0: [(0, 1), (1, 1), (2, 1)]},
        canonical_variable_plans=canonical_vars,
        canonical_relay_plans=canonical_relays,
        data_addresses=data_addresses,
    )
    shared_payload_cache: dict[Any, Any] = {}
    registered_shards = {u: list(addrs) for u, addrs in data_addresses.items()}

    for round_idx, uuid in enumerate([1001, 1002]):
      final_plan = raiden_controller.TransferPlan(
          src_units=[src_0],
          dst_units=dst_units,
          plan=None,
          worker_data_addresses=data_addresses,
          uuid=uuid,
          is_weight_sync=True,
          cached_serialized_payloads=shared_payload_cache,
      )
      asyncio.run(
          engine.execute_slice_broadcast_pipeline(
              groups_list=[stage_group],
              final_plan=final_plan,
              n_seed=1,
              req_id=f"req_warm_{round_idx}",
              dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
              registered_shards=registered_shards,
              pipeline_target_stages=2,
          )
      )

    self.assertIn(("__layer_counts__", 0, 0), stage_group.cached_hop_schedules)
    r0_sender_plans = [
        plan
        for _, plan in rpc_client.invocations
        if plan.is_sender and plan.uuid == 1001
    ]
    r1_sender_plans = [
        plan
        for _, plan in rpc_client.invocations
        if plan.is_sender and plan.uuid == 1002
    ]
    self.assertLen(r0_sender_plans, len(r1_sender_plans))
    for p0, p1 in zip(r0_sender_plans, r1_sender_plans):
      self.assertIs(
          p0.cached_serialized_payloads, p1.cached_serialized_payloads
      )
      for s_u in p0.shard_push_schedules:
        for shard_idx in p0.shard_push_schedules[s_u]:
          self.assertIs(
              p0.shard_push_schedules[s_u][shard_idx],
              p1.shard_push_schedules[s_u][shard_idx],
          )

  def test_multi_equivalence_class_1hop_coalesces_into_single_stage(
      self,
  ) -> None:
    """Verifies multiple 1-hop equivalence classes coalesce into a single stage without cross-eq_class count inflation."""
    rpc_client = RecordingWorkerRpcClient()
    self.addCleanup(rpc_client.close)
    engine = broadcast_engine.BroadcastEngine(rpc_client)

    src_0 = RaidenId(job_name="trainer", job_replica_id="0", data_name="w")
    dst_0 = RaidenId(job_name="sampler", job_replica_id="0", data_name="w")
    dst_1 = RaidenId(job_name="sampler", job_replica_id="1", data_name="w")
    data_addresses = {
        src_0: ["10.0.0.1:8000", "10.0.0.2:8000"],
        dst_0: ["10.0.1.0:8000"],
        dst_1: ["10.0.1.1:8000"],
    }

    # eq_idx=0 has 1 chunk on shard 0; eq_idx=1 has 2 chunks on shard 1.
    canonical_vars_eq0 = {
        src_0: {1: {0: [(0, 0, 0, 512, 0, 0, 512, 512, 1)]}},
    }
    canonical_vars_eq1 = {
        src_0: {
            1: {
                1: [
                    (0, 0, 0, 256, 0, 0, 256, 256, 1),
                    (0, 256, 256, 256, 0, 0, 256, 256, 1),
                ]
            }
        },
    }
    groups_list = []
    for lg_idx in range(2):
      groups_list.append(
          controller_types.StageBroadcastGroup(
              pool_group=0,
              layer_group_idx=lg_idx,
              src_units=[src_0],
              dst_units=[dst_0],
              stage_ordered_vars_by_unit={src_0: [(lg_idx, 1)]},
              canonical_variable_plans=canonical_vars_eq0,
              canonical_relay_plans={},
              data_addresses=data_addresses,
          )
      )
    for lg_idx in range(2):
      groups_list.append(
          controller_types.StageBroadcastGroup(
              pool_group=1,
              layer_group_idx=lg_idx,
              src_units=[src_0],
              dst_units=[dst_1],
              stage_ordered_vars_by_unit={src_0: [(lg_idx, 1)]},
              canonical_variable_plans=canonical_vars_eq1,
              canonical_relay_plans={},
              data_addresses=data_addresses,
          )
      )

    coalesced = broadcast_engine._coalesce_pipeline_groups(
        groups_list, target_stages=1, n_seed=1
    )
    self.assertLen(coalesced, 1)
    self.assertEqual(coalesced[0].dst_units, [dst_0, dst_1])
    self.assertLen(coalesced[0].eq_class_groups, 2)

    precomputed_scheds = (
        broadcast_engine.BroadcastEngine.build_stage_trainer_push_schedules(
            coalesced[0],
            coalesced[0].dst_units,
            registered_shards=data_addresses,
        )
    )
    # 2 layers * 1 chunk for shard 0 -> dst_0; 2 layers * 2 chunks for shard 1 -> dst_1
    self.assertLen(precomputed_scheds[src_0][0], 2)
    self.assertLen(precomputed_scheds[src_0][1], 4)

    sentinel_protos = {src_0: {0: "proto0", 1: "proto1"}}
    final_plan = raiden_controller.TransferPlan(
        src_units=[src_0],
        dst_units=[dst_0, dst_1],
        plan=None,
        shard_push_schedules=precomputed_scheds,
        worker_data_addresses=data_addresses,
        uuid=2026,
        is_weight_sync=True,
        sender_push_schedule_protos=sentinel_protos,
    )
    asyncio.run(
        engine.execute_slice_broadcast_pipeline(
            groups_list=groups_list,
            final_plan=final_plan,
            n_seed=1,
            req_id="req_1hop_multi_eq",
            dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
            registered_shards=data_addresses,
            pipeline_target_stages=2,
        )
    )

    sender_plans = [
        plan
        for target_id, plan in rpc_client.invocations
        if target_id == src_0 and plan.is_sender
    ]
    # Single 1-hop stage dispatch for the trainer source unit!
    self.assertLen(sender_plans, 1)
    self.assertIs(sender_plans[0].sender_push_schedule_protos, sentinel_protos)
    self.assertEqual(sender_plans[0].expected_block_count, 6)

    dst0_plans = [
        plan
        for target_id, plan in rpc_client.invocations
        if target_id == dst_0 and not plan.is_sender
    ]
    dst1_plans = [
        plan
        for target_id, plan in rpc_client.invocations
        if target_id == dst_1 and not plan.is_sender
    ]
    self.assertLen(dst0_plans, 1)
    self.assertLen(dst1_plans, 1)
    # dst_0 only receives eq_idx=0's 2 blocks (1 per layer), NOT eq_idx=1's blocks!
    self.assertEqual(dst0_plans[0].expected_block_count, 2)
    self.assertEqual(dst0_plans[0].expected_layer_chunk_counts, {0: 1, 1: 1})
    # dst_1 only receives eq_idx=1's 4 blocks (2 per layer), NOT eq_idx=0's blocks!
    self.assertEqual(dst1_plans[0].expected_block_count, 4)
    self.assertEqual(dst1_plans[0].expected_layer_chunk_counts, {0: 2, 1: 2})

  def test_coalesce_pipeline_groups_cached_reuses_result_across_transfers(
      self,
  ) -> None:
    """Verifies planner stage groups are coalesced once per cached plan.

    ReshardPlanner emits one StageBroadcastGroup per (eq_class, layer_group)
    sharing a single canonical plan dict per eq_class. Warm transfers must
    reuse the coalesced result (and the shared plan dicts) instead of
    re-merging O(eq_classes x layers x src_units) dict entries per transfer.
    """
    src_0 = RaidenId(job_name="trainer", job_replica_id="0", data_name="w")
    dst_units = [
        RaidenId(job_name="sampler", job_replica_id=str(i), data_name="w")
        for i in range(3)
    ]
    data_addresses = {src_0: ["10.0.0.1:8000"]}
    data_addresses.update(
        {d: [f"10.0.1.{i}:8000"] for i, d in enumerate(dst_units)}
    )
    num_layers = 50
    plans_by_eq = []
    groups_list = []
    for eq_idx, d in enumerate(dst_units):
      eq_plans = {
          src_0: {
              pid: {0: [(0, pid * 64, pid * 64, 64, 0, 0, 64, 64, 1)]}
              for pid in range(4)
          }
      }
      plans_by_eq.append(eq_plans)
      for l_idx in range(num_layers):
        groups_list.append(
            controller_types.StageBroadcastGroup(
                pool_group=eq_idx,
                layer_group_idx=l_idx,
                src_units=[src_0],
                dst_units=[d],
                stage_ordered_vars_by_unit={src_0: [(l_idx, l_idx % 4)]},
                canonical_variable_plans=eq_plans,
                canonical_relay_plans={},
                data_addresses=data_addresses,
            )
        )

    stages_a, coalesced_a = (
        broadcast_engine.BroadcastEngine.coalesce_pipeline_groups_cached(
            groups_list, 4, 1
        )
    )
    self.assertEqual(stages_a, 1)
    self.assertLen(coalesced_a, 1)
    self.assertEqual(coalesced_a[0].dst_units, dst_units)
    self.assertLen(coalesced_a[0].eq_class_groups, 3)
    # The merged per-eq_class groups reuse the planner's shared plan dicts
    # rather than copying them layer by layer.
    for eq_idx, sub_g in enumerate(coalesced_a[0].eq_class_groups):
      self.assertIs(sub_g.canonical_variable_plans, plans_by_eq[eq_idx])
      self.assertEqual(
          sub_g.stage_ordered_vars_by_unit[src_0],
          [(l_idx, l_idx % 4) for l_idx in range(num_layers)],
      )

    with mock.patch.object(
        broadcast_engine,
        "_coalesce_pipeline_groups",
        side_effect=AssertionError("must not re-coalesce a cached plan"),
    ):
      stages_b, coalesced_b = (
          broadcast_engine.BroadcastEngine.coalesce_pipeline_groups_cached(
              groups_list, 4, 1
          )
      )
    self.assertEqual(stages_b, 1)
    self.assertIs(coalesced_b, coalesced_a)

    # A different (pipeline_target_stages, n_seed) is a separate memo entry.
    stages_c, coalesced_c = (
        broadcast_engine.BroadcastEngine.coalesce_pipeline_groups_cached(
            groups_list, 4, 2
        )
    )
    self.assertEqual(stages_c, 1)
    self.assertIsNot(coalesced_c, coalesced_a)

    # Two back-to-back pipeline executions over the same planner groups only
    # coalesce once.
    rpc_client = RecordingWorkerRpcClient()
    self.addCleanup(rpc_client.close)
    engine = broadcast_engine.BroadcastEngine(rpc_client)
    fresh_groups = [
        controller_types.StageBroadcastGroup(
            pool_group=g.pool_group,
            layer_group_idx=g.layer_group_idx,
            src_units=list(g.src_units),
            dst_units=list(g.dst_units),
            stage_ordered_vars_by_unit=g.stage_ordered_vars_by_unit,
            canonical_variable_plans=g.canonical_variable_plans,
            canonical_relay_plans=g.canonical_relay_plans,
            data_addresses=g.data_addresses,
        )
        for g in groups_list
    ]
    real_coalesce = broadcast_engine._coalesce_pipeline_groups
    with mock.patch.object(
        broadcast_engine, "_coalesce_pipeline_groups", wraps=real_coalesce
    ) as coalesce_mock:
      for uuid in (11, 12):
        final_plan = raiden_controller.TransferPlan(
            src_units=[src_0],
            dst_units=list(dst_units),
            plan=None,
            shard_push_schedules={},
            worker_data_addresses=data_addresses,
            uuid=uuid,
            is_weight_sync=True,
        )
        asyncio.run(
            engine.execute_slice_broadcast_pipeline(
                groups_list=fresh_groups,
                final_plan=final_plan,
                n_seed=1,
                req_id=f"req_cached_{uuid}",
                dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
                registered_shards=data_addresses,
                pipeline_target_stages=4,
            )
        )
    self.assertEqual(coalesce_mock.call_count, 1)
    sender_plans = [
        plan
        for target_id, plan in rpc_client.invocations
        if target_id == src_0 and plan.is_sender
    ]
    self.assertLen(sender_plans, 2)
    for plan in sender_plans:
      self.assertEqual(plan.expected_block_count, num_layers * 3)
    receiver_plans = [
        plan for _, plan in rpc_client.invocations if not plan.is_sender
    ]
    self.assertLen(receiver_plans, 6)
    for plan in receiver_plans:
      self.assertEqual(plan.expected_block_count, num_layers)

  def test_peer_control_endpoints_passed_by_ref_on_senders_only(
      self,
  ) -> None:
    """Verifies sender plans share endpoints by ref and receivers omit them."""
    src = RaidenId("trainer", "0", "w")
    seed = RaidenId("sampler", "0", "w")
    relay = RaidenId("sampler", "1", "w")
    endpoint_map = {f"10.0.1.{i}:8000": f"10.0.1.{i}:9000" for i in (0, 1)}
    endpoint_key = tuple(sorted(endpoint_map.items()))
    data_addresses = {
        src: ["10.0.0.1:8000"],
        seed: ["10.0.1.0:8000"],
        relay: ["10.0.1.1:8000"],
    }
    stage_group = controller_types.StageBroadcastGroup(
        pool_group=0,
        layer_group_idx=0,
        src_units=[src],
        dst_units=[seed, relay],
        stage_ordered_vars_by_unit={src: [(0, 10)]},
        canonical_variable_plans={
            src: {10: {0: [(0, 0, 0, 512, 0, 0, 512, 512, 1)]}}
        },
        canonical_relay_plans={10: {0: [(0, 512, 0)]}},
        data_addresses=data_addresses,
    )
    final_plan = raiden_controller.TransferPlan(
        src_units=[src],
        dst_units=[seed, relay],
        plan=None,
        worker_data_addresses=data_addresses,
        peer_control_endpoints=endpoint_map,
        peer_control_endpoints_key=endpoint_key,
        uuid=101,
        is_weight_sync=True,
    )

    def _run_pipeline(groups, reg_shards, req_id):
      rpc = RecordingWorkerRpcClient()
      self.addCleanup(rpc.close)
      engine = broadcast_engine.BroadcastEngine(rpc)
      asyncio.run(
          engine.execute_slice_broadcast_pipeline(
              groups_list=groups,
              final_plan=final_plan,
              n_seed=1,
              req_id=req_id,
              dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
              registered_shards=reg_shards,
              pipeline_target_stages=1,
          )
      )
      return rpc.invocations

    stage_invocations = _run_pipeline([stage_group], data_addresses, "req_sbg")
    self.assertEqual(
        {t for t, p in stage_invocations if p.is_sender and t in p.src_units},
        {src, seed},
    )
    self.assertTrue(any(not p.is_sender for _, p in stage_invocations))
    for _, plan in stage_invocations:
      if plan.is_sender:
        self.assertIs(plan.peer_control_endpoints, endpoint_map)
        self.assertIs(plan.peer_control_endpoints_key, endpoint_key)
      else:
        self.assertEmpty(plan.peer_control_endpoints)
        self.assertIsNone(plan.peer_control_endpoints_key)

    slice_group = [(
        (src, 0, 0, 0, 512, 0, 1, 0, 0),
        [
            (seed, "10.0.1.0:8000", 0, 0, 0, 0),
            (relay, "10.0.1.1:8000", 0, 0, 0, 0),
        ],
    )]
    slice_invocations = _run_pipeline(
        [slice_group], {u: ["s0"] for u in (src, seed, relay)}, "req_slice"
    )
    relay_plans = [
        p
        for t, p in slice_invocations
        if t == seed and p.is_sender and p.dst_units == [relay]
    ]
    self.assertLen(relay_plans, 1)
    self.assertIs(relay_plans[0].peer_control_endpoints, endpoint_map)
    self.assertIs(relay_plans[0].peer_control_endpoints_key, endpoint_key)


if __name__ == "__main__":
  absltest.main()
