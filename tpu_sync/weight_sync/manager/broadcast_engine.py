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

"""Pipelined multi-hop tree broadcast scheduler and execution engine."""

import asyncio
import collections
import functools
import random
from typing import Any, Callable, Optional

from absl import logging

from tpu_sync.api.common import RaidenId


# TODO(fhzhang): Make max_chunk_bytes and target_stages configurable via RaidenController options.
def _coalesce_contiguous_relay_entries(
    entries: list[tuple[Any, ...]],
    max_chunk_bytes: int = 64 * 1024 * 1024,
    is_weight_sync: bool = False,
) -> list[tuple[Any, ...]]:
  """Coalesces adjacent contiguous relay entries into larger multi-MB blocks.

  Args:
    entries: List of 12-tuples: (dst_peer, dst_shard_idx, dst_block_offset,
      src_block_offset, size, src_block_id, dst_block_id, src_stride,
      dst_stride, count, layer_idx, pool_group).
    max_chunk_bytes: Maximum size in bytes of a coalesced chunk (default 64MB).
    is_weight_sync: Whether this transfer is weight synchronization, where block
      IDs are normalized to 0 because flat layer offsets are used.

  Returns:
    Coalesced list of 12-tuples.
  """
  if not entries:
    return []

  normalized: list[tuple[Any, ...]] = []
  for e in entries:
    (
        dst_peer,
        dst_shard_idx,
        dst_offset,
        src_offset,
        size,
        src_block_id,
        dst_block_id,
        src_stride,
        dst_stride,
        count,
        layer_idx,
        pool_group,
    ) = e
    if count == 1 or (src_stride == size and dst_stride == size):
      total_size = count * size
      norm_src_block_id = 0 if is_weight_sync else src_block_id
      norm_dst_block_id = 0 if is_weight_sync else dst_block_id
      normalized.append((
          dst_peer,
          dst_shard_idx,
          dst_offset,
          src_offset,
          total_size,
          norm_src_block_id,
          norm_dst_block_id,
          total_size,
          total_size,
          1,
          layer_idx,
          pool_group,
      ))
    else:
      normalized.append(e)

  def sort_key(x: tuple[Any, ...]) -> tuple[Any, ...]:
    return (x[10], x[11], x[0], x[1], x[5], x[6], x[3], x[2])

  sorted_entries = sorted(normalized, key=sort_key)

  coalesced: list[tuple[Any, ...]] = []
  for e in sorted_entries:
    if not coalesced:
      coalesced.append(e)
      continue

    curr = coalesced[-1]
    if (
        curr[9] == 1
        and e[9] == 1
        and curr[0] == e[0]
        and curr[1] == e[1]
        and curr[5] == e[5]
        and curr[6] == e[6]
        and curr[10] == e[10]
        and curr[11] == e[11]
        and e[3] == curr[3] + curr[4]
        and e[2] == curr[2] + curr[4]
        and curr[4] + e[4] <= max_chunk_bytes
    ):
      new_size = curr[4] + e[4]
      coalesced[-1] = (
          curr[0],
          curr[1],
          curr[2],
          curr[3],
          new_size,
          curr[5],
          curr[6],
          new_size,
          new_size,
          1,
          curr[10],
          curr[11],
      )
    else:
      coalesced.append(e)

  return coalesced


# TODO(fhzhang): Make max_chunk_bytes and target_stages configurable via RaidenController options.
def _coalesce_pipeline_groups(
    groups_list: list[list[tuple[tuple[Any, ...], list[tuple[Any, ...]]]]],
    target_stages: int = 8,
) -> list[list[tuple[tuple[Any, ...], list[tuple[Any, ...]]]]]:
  """Coalesces fine-grained groups into fewer pipeline stages if len > target_stages.

  Args:
    groups_list: List of pipeline stages, where each stage is a list of (key,
      targets) slice transfer tuples.
    target_stages: Maximum number of coalesced pipeline stages to produce.

  Returns:
    Coalesced list of pipeline stages.
  """
  if len(groups_list) <= target_stages:
    return groups_list

  max_groups_per_stage = (len(groups_list) + target_stages - 1) // target_stages
  coalesced: list[list[tuple[tuple[Any, ...], list[tuple[Any, ...]]]]] = []
  current_stage: list[tuple[tuple[Any, ...], list[tuple[Any, ...]]]] = []
  current_key: Optional[tuple[Any, ...]] = None
  current_count = 0

  for g in groups_list:
    if not g:
      continue
    ref_key, ref_targets = g[0]
    src_unit = ref_key[0]
    shard_idx = ref_key[1]
    sorted_targets = sorted(ref_targets, key=lambda t: (t[1], t[2]))
    targets_routing_key = tuple((t[0], t[1], t[2]) for t in sorted_targets)
    routing_key = (src_unit, shard_idx, targets_routing_key)

    if current_key == routing_key and current_count < max_groups_per_stage:
      current_stage.extend(g)
      current_count += 1
    else:
      if current_stage:
        coalesced.append(current_stage)
      current_stage = list(g)
      current_key = routing_key
      current_count = 1

  if current_stage:
    coalesced.append(current_stage)

  return coalesced


class _HopTask:
  """A single transfer hop in the deterministic broadcast tree."""

  def __init__(
      self,
      group: "_GroupBroadcastState",
      sender: RaidenId,
      receiver: RaidenId,
      dst_indices: list[int],
  ) -> None:
    self.group = group
    self.sender = sender
    self.receiver = receiver
    self.dst_indices = dst_indices
    self.children: list[_HopTask] = []
    self.sub_plan: Any = None


def _populate_receiver_offsets(hop: _HopTask) -> None:
  """Populates node_slice_offsets for the hop's receiver."""
  ref_idx = hop.dst_indices[0]
  hop.group.node_slice_offsets[hop.receiver] = {}
  for key, k_targets in hop.group.keys_and_sorted_targets:
    k_target = k_targets[ref_idx]
    (
        _,
        _,
        k_dst_shard_idx,
        k_dst_block_id,
        k_dst_block_offset,
        k_dst_stride,
    ) = k_target
    hop.group.node_slice_offsets[hop.receiver][key] = (
        k_dst_shard_idx,
        k_dst_block_id,
        k_dst_block_offset,
        k_dst_stride,
    )


def _build_hop_sub_plan(
    hop: _HopTask,
    final_plan: Any,
    dst_mem_type: int,
) -> None:
  """Pre-computes and caches the sub_plan for a single tree hop."""
  group = hop.group
  s = hop.sender
  dst_unit = hop.receiver
  dst_indices = hop.dst_indices

  ref_offset = group.node_slice_offsets[s][group.ref_key]
  s_shard_idx = ref_offset[0]

  sub_schedule: dict[RaidenId, dict[int, list[Any]]] = {s: {s_shard_idx: []}}
  for key, k_targets in group.keys_and_sorted_targets:
    _, k_s_block_id, k_s_block_offset, k_s_stride = group.node_slice_offsets[s][
        key
    ]
    k_size = key[4]
    k_count = key[6]
    k_layer_idx = key[7]
    k_pool_group = key[8]

    for idx in dst_indices:
      k_target = k_targets[idx]
      (
          _,
          k_dst_peer,
          k_dst_shard_idx,
          k_dst_block_id,
          k_dst_block_offset,
          k_dst_stride,
      ) = k_target

      entry = (
          k_dst_peer,
          k_dst_shard_idx,
          k_dst_block_offset,
          k_s_block_offset,
          k_size,
          k_s_block_id,
          k_dst_block_id,
          k_s_stride,
          k_dst_stride,
          k_count,
          k_layer_idx,
          k_pool_group,
      )
      sub_schedule[s][s_shard_idx].append(entry)

  if s != group.src_unit:
    sub_schedule[s][s_shard_idx] = _coalesce_contiguous_relay_entries(
        sub_schedule[s][s_shard_idx],
        is_weight_sync=bool(final_plan.is_weight_sync),
    )

  hop_expected_layer_chunk_counts: dict[int, int] = {}
  for e in sub_schedule[s][s_shard_idx]:
    layer_idx = e[10]
    push_count = 1 if (e[9] == 1 or (e[7] == e[4] and e[8] == e[4])) else e[9]
    hop_expected_layer_chunk_counts[layer_idx] = (
        hop_expected_layer_chunk_counts.get(layer_idx, 0) + push_count
    )

  hop_expected_block_count = sum(hop_expected_layer_chunk_counts.values())

  sub_plan = type(final_plan)(
      src_units=[s],
      dst_units=[dst_unit],
      plan=None,
      shard_push_schedules=sub_schedule,
      worker_rpc_addresses=(
          dict(final_plan.worker_rpc_addresses)
          if final_plan.worker_rpc_addresses is not None
          else {}
      ),
      worker_data_addresses=(
          dict(final_plan.worker_data_addresses)
          if final_plan.worker_data_addresses is not None
          else {}
      ),
      uuid=0,
      dst_mem_type=dst_mem_type,
      use_block_chunks=True,
      is_sender=True,
      expected_block_count=hop_expected_block_count,
      expected_layer_chunk_counts=hop_expected_layer_chunk_counts,
      dst_expected_layer_chunk_counts={
          dst_unit: hop_expected_layer_chunk_counts
      },
      req_id="",
      skip_d2h=final_plan.skip_d2h or (s != group.src_unit),
      skip_tiling=final_plan.skip_tiling,
      is_weight_sync=final_plan.is_weight_sync,
      cached_serialized_payloads={},
  )
  hop.sub_plan = sub_plan


class _GroupBroadcastState:
  """State for a single broadcast group in the broadcast pipeline."""

  def __init__(
      self,
      group_idx: int,
      keys_and_sorted_targets: list[
          tuple[tuple[Any, ...], list[tuple[Any, ...]]]
      ],
      src_unit: RaidenId,
      shard_idx: int,
  ) -> None:
    self.group_idx = group_idx
    self.keys_and_sorted_targets = keys_and_sorted_targets
    self.src_unit = src_unit
    self.shard_idx = shard_idx
    self.available_sources: list[RaidenId] = [src_unit]

    ref_key, ref_targets = keys_and_sorted_targets[0]
    self.ref_key = ref_key
    self.ref_targets = ref_targets

    self.node_slice_offsets: dict[
        RaidenId, dict[Any, tuple[int, int, int, int]]
    ] = {
        src_unit: {
            key: (shard_idx, key[2], key[3], key[5])
            for key, _ in keys_and_sorted_targets
        }
    }

    self.dst_unit_to_indices: dict[RaidenId, list[int]] = {}
    self.pending_dst_units: list[RaidenId] = []
    for idx, target in enumerate(ref_targets):
      dst_u = target[0]
      if dst_u not in self.dst_unit_to_indices:
        self.dst_unit_to_indices[dst_u] = []
        self.pending_dst_units.append(dst_u)
      self.dst_unit_to_indices[dst_u].append(idx)


class BroadcastEngine:
  """Executes pipelined multi-hop tree broadcasts for grouped tensor slices."""

  _coalesce_contiguous_relay_entries = staticmethod(
      _coalesce_contiguous_relay_entries
  )
  _coalesce_pipeline_groups = staticmethod(_coalesce_pipeline_groups)

  def __init__(
      self,
      worker_rpc_client: Any,
      remote_controller_client_factory: Optional[Callable[..., Any]] = None,
  ) -> None:
    self._worker_rpc_client = worker_rpc_client
    self._remote_client_factory = remote_controller_client_factory
    self._pipeline_cache: dict[tuple[Any, ...], Any] = {}

  @classmethod
  def partition_direct_and_broadcast_groups(
      cls,
      groups: dict[tuple[Any, ...], list[tuple[Any, ...]]],
      broadcast_k: int,
      group_size: int = 1,
  ) -> tuple[
      dict[RaidenId, dict[int, list[Any]]], dict[tuple[Any, ...], list[Any]]
  ]:
    """Partitions slice transfer groups into direct pushes vs.

    tree-broadcast groups.
    """
    direct_schedules: dict[RaidenId, dict[int, list[Any]]] = {}
    broadcast_groups: dict[tuple[Any, ...], list[Any]] = {}

    for key, targets in groups.items():
      unique_dst_units = set(t[0] for t in targets)
      is_tree_broadcast = (
          len(unique_dst_units) > 1 and len(unique_dst_units) > broadcast_k
      )
      if not is_tree_broadcast:
        (
            src_unit,
            shard_idx,
            src_block_id,
            src_block_offset,
            size,
            src_stride,
            count,
            layer_idx,
            pool_group,
        ) = key
        for (
            _,
            dst_peer,
            dst_shard_idx,
            dst_block_id,
            dst_block_offset,
            dst_stride,
        ) in targets:
          entry = (
              dst_peer,
              dst_shard_idx,
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
          )
          direct_schedules.setdefault(src_unit, {}).setdefault(
              shard_idx, []
          ).append(entry)
      else:
        (
            src_unit,
            shard_idx,
            src_block_id,
            src_block_offset,
            size,
            src_stride,
            count,
            layer_idx,
            pool_group,
        ) = key

        layer_group_idx = (
            layer_idx // group_size if group_size > 1 else layer_idx
        )

        sorted_targets = sorted(targets, key=lambda t: t[1])
        targets_routing_key = tuple((t[0], t[1], t[2]) for t in sorted_targets)

        group_key = (
            src_unit,
            shard_idx,
            pool_group,
            layer_group_idx,
            targets_routing_key,
        )
        broadcast_groups.setdefault(group_key, []).append((key, targets))

    return direct_schedules, broadcast_groups

  async def execute_slice_broadcast_pipeline(
      self,
      groups_list: list[list[tuple[tuple[Any, ...], list[tuple[Any, ...]]]]],
      final_plan: Any,
      fanout_k: int,
      req_id: str,
      dst_mem_type: int,
      registered_shards: dict[RaidenId, list[str]],
      dst_controller_address: Optional[str] = None,
      src_controller_address: Optional[str] = None,
  ) -> None:
    """Executes a pipelined multi-hop tree broadcast across multiple groups."""
    if fanout_k <= 0:
      raise ValueError(f"fanout_k must be >= 1, got {fanout_k}")

    pipeline_cache_key = (
        id(groups_list),
        fanout_k,
        dst_mem_type,
        bool(final_plan.is_weight_sync),
        bool(final_plan.skip_d2h),
    )
    cached_entry = self._pipeline_cache.get(pipeline_cache_key)
    if cached_entry is not None and cached_entry[0] is groups_list:
      _, groups, all_workers, hop_trees = cached_entry
    else:
      coalesced_groups_list = _coalesce_pipeline_groups(
          groups_list, target_stages=8
      )

      groups = []
      all_workers = []

      def _add_worker(u: RaidenId) -> None:
        if u not in all_workers:
          all_workers.append(u)

      for g_idx, keys_and_targets in enumerate(coalesced_groups_list):
        if not keys_and_targets:
          continue
        keys_and_sorted_targets = []
        for key, targets in keys_and_targets:
          sorted_t = sorted(targets, key=lambda t: (t[1], t[2]))
          keys_and_sorted_targets.append((key, sorted_t))

        ref_key, ref_targets = keys_and_sorted_targets[0]
        src_unit = ref_key[0]
        shard_idx = ref_key[1]

        _add_worker(src_unit)
        for t in ref_targets:
          _add_worker(t[0])

        groups.append(
            _GroupBroadcastState(
                group_idx=g_idx,
                keys_and_sorted_targets=keys_and_sorted_targets,
                src_unit=src_unit,
                shard_idx=shard_idx,
            )
        )

      hop_trees: list[list[_HopTask]] = []
      # Build deterministic balanced k-ary trees for each group
      for g in groups:
        dst_units = g.pending_dst_units
        n = len(dst_units)
        if n == 0:
          continue

        k_seed = min(fanout_k, n)
        level_1_hops: list[_HopTask] = []
        for i in range(k_seed):
          d = dst_units[i]
          hop = _HopTask(g, g.src_unit, d, g.dst_unit_to_indices[d])
          _populate_receiver_offsets(hop)
          _build_hop_sub_plan(hop, final_plan, dst_mem_type)
          level_1_hops.append(hop)

        prev_level_hops = level_1_hops
        next_idx = k_seed
        while next_idx < n:
          curr_level_hops: list[_HopTask] = []
          max_level_targets = min(len(prev_level_hops) * fanout_k, n - next_idx)
          for j in range(max_level_targets):
            parent_hop = prev_level_hops[j % len(prev_level_hops)]
            d = dst_units[next_idx]
            child_hop = _HopTask(
                g, parent_hop.receiver, d, g.dst_unit_to_indices[d]
            )
            _populate_receiver_offsets(child_hop)
            _build_hop_sub_plan(child_hop, final_plan, dst_mem_type)
            parent_hop.children.append(child_hop)
            curr_level_hops.append(child_hop)
            next_idx += 1
          prev_level_hops = curr_level_hops
        hop_trees.append(level_1_hops)

      if groups:
        if len(self._pipeline_cache) >= 64:
          self._pipeline_cache.pop(next(iter(self._pipeline_cache)))
        self._pipeline_cache[pipeline_cache_key] = (
            groups_list,
            groups,
            all_workers,
            hop_trees,
        )

    if not groups:
      return

    active_pushes: dict[RaidenId, int] = {u: 0 for u in all_workers}
    ready_queue: dict[RaidenId, collections.deque[_HopTask]] = {
        u: collections.deque() for u in all_workers
    }
    transfers_in_progress: dict[asyncio.Task[None], _HopTask] = {}

    for level_1_hops in hop_trees:
      for hop in level_1_hops:
        ready_queue[hop.sender].append(hop)

    async def _run_single_transfer(
        s_node: RaidenId, d_node: RaidenId, plan: Any
    ) -> None:
      if dst_controller_address:
        if self._remote_client_factory is None:
          raise RuntimeError(
              "remote_controller_client_factory is required for remote"
              " controller broadcast"
          )
        dst_facade = self._remote_client_factory(
            dst_controller_address,
            name_resolver=self._worker_rpc_client.name_resolver,
        )
        loop = asyncio.get_running_loop()
        rpc_executor = getattr(self._worker_rpc_client, "executor", None)
        remote_is_sender = s_node not in registered_shards
        if (
            remote_is_sender
            or self._worker_rpc_client.include_receiver_push_schedules(plan)
        ):
          remote_schedules = plan.shard_push_schedules
        else:
          remote_schedules = None
        success = await loop.run_in_executor(
            rpc_executor,
            functools.partial(
                dst_facade.register_transfer_schedule,
                [s_node],
                [d_node],
                plan.req_id,
                True,
                remote_is_sender,
                plan.expected_block_count,
                plan.uuid,
                dst_controller_address,
                src_controller_address,
                remote_schedules,
                dst_mem_type,
                skip_d2h=plan.skip_d2h,
            ),
        )
        if not success:
          raise RuntimeError("Failed remote prepare in slice tree broadcast")
        if s_node in registered_shards:
          await self._worker_rpc_client.start_transfer(s_node, plan)
      else:
        if s_node in registered_shards and int(dst_mem_type) in (0, 1):
          await asyncio.gather(
              self._worker_rpc_client.start_transfer(d_node, plan),
              self._worker_rpc_client.start_transfer(s_node, plan),
          )
        else:
          await self._worker_rpc_client.start_transfer(d_node, plan)
          if s_node in registered_shards:
            await self._worker_rpc_client.start_transfer(s_node, plan)

    def _dispatch_hop(hop: _HopTask) -> None:
      active_pushes[hop.sender] += 1
      hop_uuid = random.randint(1, 2**63 - 1)
      hop.sub_plan.uuid = hop_uuid
      hop.sub_plan.req_id = f"{req_id}__stage_{hop.group.group_idx}__{hop_uuid}"
      hop.sub_plan.skip_d2h = final_plan.skip_d2h or (
          hop.sender != hop.group.src_unit
      )
      hop.sub_plan.skip_tiling = final_plan.skip_tiling

      task = asyncio.create_task(
          _run_single_transfer(hop.sender, hop.receiver, hop.sub_plan)
      )
      transfers_in_progress[task] = hop

    try:
      while any(ready_queue.values()) or transfers_in_progress:
        while True:
          scheduled_any = False
          for u in all_workers:
            while active_pushes[u] < fanout_k and ready_queue[u]:
              hop = ready_queue[u].popleft()
              _dispatch_hop(hop)
              scheduled_any = True
          if not scheduled_any:
            break

        if transfers_in_progress:
          done, _ = await asyncio.wait(
              transfers_in_progress.keys(), return_when=asyncio.FIRST_COMPLETED
          )
          for fut in done:
            if fut.cancelled():
              raise asyncio.CancelledError()
            exc = fut.exception()
            if exc is not None:
              logging.error("Slice transfer failed in broadcast tree: %s", exc)
              raise exc
            hop = transfers_in_progress.pop(fut)
            active_pushes[hop.sender] -= 1

            for child_hop in hop.children:
              ready_queue[hop.receiver].append(child_hop)
    finally:
      pending = [f for f in transfers_in_progress.keys() if not f.done()]
      for f in pending:
        f.cancel()
      if pending:
        await asyncio.gather(*pending, return_exceptions=True)

  async def execute_slice_broadcast(
      self,
      keys_and_targets: list[tuple[tuple[Any, ...], list[tuple[Any, ...]]]],
      final_plan: Any,
      fanout_k: int,
      req_id: str,
      dst_mem_type: int,
      registered_shards: dict[RaidenId, list[str]],
      dst_controller_address: Optional[str] = None,
      src_controller_address: Optional[str] = None,
  ) -> None:
    """Executes a pipelined tree broadcast for a group of variables."""
    await self.execute_slice_broadcast_pipeline(
        groups_list=[keys_and_targets],
        final_plan=final_plan,
        fanout_k=fanout_k,
        req_id=req_id,
        dst_mem_type=dst_mem_type,
        registered_shards=registered_shards,
        dst_controller_address=dst_controller_address,
        src_controller_address=src_controller_address,
    )
