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
import heapq
import itertools
import random
import time
from typing import Any, Callable, Optional

from absl import logging

from tpu_sync.api.common import RaidenId
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.weight_sync.manager import controller_types

_PIPELINE_TARGET_STAGES: int = 4
_RELAY_MAX_COALESCED_CHUNK_BYTES: int = 4 * 1024 * 1024


def _coalesce_contiguous_relay_entries(
    entries: list[tuple[Any, ...]],
    max_chunk_bytes: int = _RELAY_MAX_COALESCED_CHUNK_BYTES,
) -> list[tuple[Any, ...]]:
  """Coalesces adjacent contiguous relay entries into larger multi-MB blocks.

  Args:
    entries: List of 12-tuples: (dst_peer, dst_shard_idx, dst_block_offset,
      src_block_offset, size, src_block_id, dst_block_id, src_stride,
      dst_stride, count, layer_idx, pool_group).
    max_chunk_bytes: Maximum size in bytes of a coalesced chunk (default 4MB).

  Returns:
    Coalesced list of 12-tuples.

  Raises:
    ValueError: If max_chunk_bytes <= 0.
  """
  if max_chunk_bytes <= 0:
    raise ValueError(f"max_chunk_bytes must be > 0, got {max_chunk_bytes}")
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
      norm_src_block_id = src_block_id
      norm_dst_block_id = dst_block_id
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


def _coalesce_pipeline_groups(
    groups_list: list[Any],
    target_stages: int = _PIPELINE_TARGET_STAGES,
    n_seed: int = 1,
) -> list[Any]:
  """Coalesces fine-grained groups into fewer pipeline stages if len > target_stages.

  Args:
    groups_list: List of pipeline stages, where each stage is either a
      StageBroadcastGroup or a list of (key, targets) slice transfer tuples.
    target_stages: Maximum number of coalesced pipeline stages to produce.
    n_seed: Maximum number of destination units seeded per equivalence class in
      a single 1-hop trainer broadcast round.

  Returns:
    Coalesced list of pipeline stages.
  Raises:
    ValueError: If target_stages <= 0.
  """
  if target_stages <= 0:
    raise ValueError(f"target_stages must be >= 1, got {target_stages}")
  if not groups_list:
    return []

  if isinstance(groups_list[0], controller_types.StageBroadcastGroup):
    def _stage_routing_key(
        g: controller_types.StageBroadcastGroup,
    ) -> tuple[Any, ...]:
      return (tuple(g.src_units), tuple(g.dst_units), g.pool_group)

    coalesced_stage_groups: list[controller_types.StageBroadcastGroup] = []
    i = 0
    n = len(groups_list)
    while i < n:
      if not groups_list[i]:
        i += 1
        continue
      cur_key = _stage_routing_key(groups_list[i])
      run: list[controller_types.StageBroadcastGroup] = [groups_list[i]]
      j = i + 1
      while j < n:
        if not groups_list[j]:
          j += 1
          continue
        if _stage_routing_key(groups_list[j]) == cur_key:
          run.append(groups_list[j])
          j += 1
        else:
          break

      run_len = len(run)
      if run_len <= target_stages:
        if run_len == 1:
          coalesced_stage_groups.extend(run)
        else:
          for stage_slot, g in enumerate(run):
            coalesced_stage_groups.append(
                controller_types.StageBroadcastGroup(
                    pool_group=g.pool_group * target_stages + stage_slot,
                    layer_group_idx=stage_slot,
                    src_units=list(g.src_units),
                    dst_units=list(g.dst_units),
                    stage_ordered_vars_by_unit=g.stage_ordered_vars_by_unit,
                    canonical_variable_plans=g.canonical_variable_plans,
                    canonical_relay_plans=g.canonical_relay_plans,
                    data_addresses=dict(g.data_addresses),
                    cached_hop_schedules=g.cached_hop_schedules,
                )
            )
      else:
        max_per_stage = (run_len + target_stages - 1) // target_stages
        for k in range(0, run_len, max_per_stage):
          chunk = run[k : k + max_per_stage]
          first = chunk[0]
          merged_stage_ordered_vars: dict[RaidenId, list[tuple[int, int]]] = {
              u: [] for u in first.src_units
          }
          for u in first.src_units:
            u_vars = merged_stage_ordered_vars[u]
            for g in chunk:
              u_vars.extend(g.stage_ordered_vars_by_unit.get(u, ()))

          # ReshardPlanner emits one StageBroadcastGroup per (eq_class,
          # layer_group) but shares a single canonical plan dict per eq_class
          # across all of its layer groups. Re-merging those identical dicts
          # per layer group is O(layers x src_units x plan_ids) per transfer
          # (tens of millions of dict ops for 60-layer Qwen 397B at 16+16), so
          # reuse the shared object when every group in the chunk points at it.
          first_vars = first.canonical_variable_plans
          if all(g.canonical_variable_plans is first_vars for g in chunk):
            merged_canonical_vars = first_vars
          else:
            merged_canonical_vars = {u: {} for u in first.src_units}
            for g in chunk:
              for u in first.src_units:
                merged_canonical_vars[u].update(
                    g.canonical_variable_plans.get(u, {})
                )
          first_relays = first.canonical_relay_plans
          if all(g.canonical_relay_plans is first_relays for g in chunk):
            merged_canonical_relays = first_relays
          else:
            merged_canonical_relays = {}
            for g in chunk:
              merged_canonical_relays.update(g.canonical_relay_plans)

          stage_slot = k // max_per_stage
          coalesced_stage_groups.append(
              controller_types.StageBroadcastGroup(
                  pool_group=first.pool_group * target_stages + stage_slot,
                  layer_group_idx=stage_slot,
                  src_units=list(first.src_units),
                  dst_units=list(first.dst_units),
                  stage_ordered_vars_by_unit=merged_stage_ordered_vars,
                  canonical_variable_plans=merged_canonical_vars,
                  canonical_relay_plans=merged_canonical_relays,
                  data_addresses=dict(first.data_addresses),
                  cached_hop_schedules=first.cached_hop_schedules,
              )
          )
      i = j

    if target_stages == 1 and len(coalesced_stage_groups) > 1:
      eq_groups: list[controller_types.StageBroadcastGroup] = []
      for g in coalesced_stage_groups:
        sub = getattr(g, "eq_class_groups", None)
        if sub:
          eq_groups.extend(sub)
        else:
          eq_groups.append(g)
      first = eq_groups[0]
      first_src = tuple(first.src_units)
      if all(
          tuple(g.src_units) == first_src and len(g.dst_units) <= n_seed
          for g in eq_groups
      ):
        merged_dst_units: list[RaidenId] = []
        seen_dst: set[RaidenId] = set()
        merged_data_addresses: dict[RaidenId, list[str]] = {}
        for g in eq_groups:
          merged_data_addresses.update(g.data_addresses)
          for d in g.dst_units:
            if d not in seen_dst:
              seen_dst.add(d)
              merged_dst_units.append(d)
        merged_ordered_vars: dict[RaidenId, list[tuple[int, int]]] = {}
        for u in first.src_units:
          seen_vars: set[tuple[int, int]] = set()
          u_vars: list[tuple[int, int]] = []
          for g in eq_groups:
            for vp in g.stage_ordered_vars_by_unit.get(u, []):
              if vp not in seen_vars:
                seen_vars.add(vp)
                u_vars.append(vp)
          merged_ordered_vars[u] = u_vars
        return [
            controller_types.StageBroadcastGroup(
                pool_group=0,
                layer_group_idx=0,
                src_units=list(first.src_units),
                dst_units=merged_dst_units,
                stage_ordered_vars_by_unit=merged_ordered_vars,
                canonical_variable_plans=first.canonical_variable_plans,
                canonical_relay_plans={},
                data_addresses=merged_data_addresses,
                cached_hop_schedules=first.cached_hop_schedules,
                eq_class_groups=eq_groups,
            )
        ]

    return coalesced_stage_groups

  def _routing_key(
      g: list[tuple[tuple[Any, ...], list[tuple[Any, ...]]]],
  ) -> tuple[Any, ...]:
    ref_key, ref_targets = g[0]
    src_unit = ref_key[0]
    shard_idx = ref_key[1]
    sorted_targets = sorted(ref_targets, key=lambda t: (t[1], t[2]))
    targets_routing_key = tuple((t[0], t[1], t[2]) for t in sorted_targets)
    return (src_unit, shard_idx, targets_routing_key)

  coalesced: list[list[tuple[tuple[Any, ...], list[tuple[Any, ...]]]]] = []
  i = 0
  n = len(groups_list)
  while i < n:
    if not groups_list[i]:
      i += 1
      continue
    cur_key = _routing_key(groups_list[i])
    run: list[list[tuple[tuple[Any, ...], list[tuple[Any, ...]]]]] = [
        groups_list[i]
    ]
    j = i + 1
    while j < n:
      if not groups_list[j]:
        j += 1
        continue
      if _routing_key(groups_list[j]) == cur_key:
        run.append(groups_list[j])
        j += 1
      else:
        break

    run_len = len(run)
    if run_len <= target_stages:
      coalesced.extend(run)
    else:
      max_per_stage = (run_len + target_stages - 1) // target_stages
      for k in range(0, run_len, max_per_stage):
        stage: list[tuple[tuple[Any, ...], list[tuple[Any, ...]]]] = []
        for g in run[k : k + max_per_stage]:
          stage.extend(g)
        coalesced.append(stage)

    i = j

  return coalesced


def _groups_need_multi_hop(groups_list: list[Any], n_seed: int) -> bool:
  """Returns True if any stage group fans out to more than n_seed receivers."""
  for g in groups_list:
    if not g:
      continue
    if isinstance(g, controller_types.StageBroadcastGroup):
      sub_groups = getattr(g, "eq_class_groups", None)
      if sub_groups:
        if any(len(sub.dst_units) > n_seed for sub in sub_groups):
          return True
      elif len(g.dst_units) > n_seed:
        return True
    elif len({t[0] for _, targets in g for t in targets}) > n_seed:
      return True
  return False


def _coalesce_pipeline_groups_cached(
    groups_list: list[Any],
    pipeline_target_stages: int,
    n_seed: int,
) -> tuple[int, list[Any]]:
  """Coalesces planner stage groups into pipeline stages, memoized per plan.

  ReshardPlanner emits one StageBroadcastGroup per (eq_class, layer_group),
  which is O(100k) groups for a 60-layer model at 128 destination units. The
  coalesced result only depends on the (immutable) planner groups, n_seed and
  pipeline_target_stages, so it is memoized on the first group's
  `cached_hop_schedules` dict, which lives as long as the cached transfer
  schedule. This keeps warm transfers from re-walking every planner group.

  Args:
    groups_list: Planner stage groups (StageBroadcastGroup or legacy lists).
    pipeline_target_stages: Requested number of pipeline stages.
    n_seed: Seed fan-out per equivalence class.

  Returns:
    (effective_target_stages, coalesced_groups_list).
  """
  memo: Optional[dict[Any, Any]] = None
  memo_key = None
  if groups_list and isinstance(
      groups_list[0], controller_types.StageBroadcastGroup
  ):
    memo = getattr(groups_list[0], "cached_hop_schedules", None)
    if memo is not None:
      memo_key = (
          "__coalesced_pipeline__",
          pipeline_target_stages,
          n_seed,
          len(groups_list),
          id(groups_list[-1]),
      )
      cached = memo.get(memo_key)
      if cached is not None:
        return cached

  effective_target_stages = (
      pipeline_target_stages
      if _groups_need_multi_hop(groups_list, n_seed)
      else 1
  )
  coalesced = _coalesce_pipeline_groups(
      groups_list, target_stages=effective_target_stages, n_seed=n_seed
  )
  result = (effective_target_stages, coalesced)
  if memo is not None and memo_key is not None:
    memo[memo_key] = result
  return result


class _HopTask:
  """A single transfer hop in the deterministic broadcast tree."""

  def __init__(
      self,
      group: "_GroupBroadcastState",
      sender: RaidenId,
      receiver: RaidenId,
      dst_indices: list[int],
      round_idx: int = 0,
      child_order: int = 0,
      receivers: Optional[list[RaidenId]] = None,
      seed_hops: Optional[list["_HopTask"]] = None,
  ) -> None:
    self.group = group
    self.sender = sender
    self.receiver = receiver
    self.dst_indices = dst_indices
    self.round_idx = round_idx
    self.child_order = child_order
    self.receivers: list[RaidenId] = (
        list(receivers) if receivers is not None else [receiver]
    )
    self.seed_hops: list[_HopTask] = (
        list(seed_hops) if seed_hops is not None else []
    )
    self.children: list[_HopTask] = []
    self.enqueued_time: float = 0.0
    self.dispatch_time: float = 0.0
    self.queue_wait_ms: float = 0.0
    self.plan_build_ms: float = 0.0
    self.recv_arm_ms: float = 0.0
    self.sender_rpc_ms: float = 0.0
    self.active_pushes_at_dispatch: int = 0
    self.hop_req_id: str = ""


class _GroupBroadcastState:
  """State for a single broadcast group in the broadcast pipeline."""

  def __init__(
      self,
      group_idx: int,
      stage_group: Optional[controller_types.StageBroadcastGroup] = None,
      keys_and_sorted_targets: Optional[
          list[tuple[tuple[Any, ...], list[tuple[Any, ...]]]]
      ] = None,
      src_unit: Optional[RaidenId] = None,
      shard_idx: int = 0,
  ) -> None:
    self.group_idx = group_idx
    self.stage_group = stage_group
    self.stage_idx = (
        stage_group.layer_group_idx if stage_group is not None else group_idx
    )
    self.keys_and_sorted_targets = keys_and_sorted_targets or []

    if stage_group is not None:
      self.src_units = list(stage_group.src_units)
      self.primary_src_unit = self.src_units[0]
      self.src_unit = self.primary_src_unit
      self.shard_idx = shard_idx
      self.available_sources = [self.primary_src_unit]
      self.pending_dst_units = list(stage_group.dst_units)
      self.dst_unit_to_indices = {
          u: [i] for i, u in enumerate(self.pending_dst_units)
      }
    else:
      self.src_unit = src_unit
      self.primary_src_unit = src_unit
      self.shard_idx = shard_idx
      self.src_units = [src_unit] if src_unit else []
      self.available_sources = [src_unit] if src_unit else []
      ref_key, ref_targets = self.keys_and_sorted_targets[0]
      self.ref_key = ref_key
      self.ref_targets = ref_targets
      self.dst_unit_to_indices = {}
      self.pending_dst_units = []
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
  coalesce_pipeline_groups_cached = staticmethod(
      _coalesce_pipeline_groups_cached
  )

  def __init__(
      self,
      worker_rpc_client: Any,
      remote_controller_client_factory: Optional[Callable[..., Any]] = None,
  ) -> None:
    self._worker_rpc_client = worker_rpc_client
    self._remote_client_factory = remote_controller_client_factory

  @classmethod
  def build_stage_trainer_push_schedules(
      cls,
      stage_group: controller_types.StageBroadcastGroup,
      receivers: list[RaidenId],
      registered_shards: Optional[dict[RaidenId, list[str]]] = None,
  ) -> dict[RaidenId, dict[int, Any]]:
    """Builds trainer source push schedules for a StageBroadcastGroup targeting receivers."""
    if not receivers:
      return {s_u: {} for s_u in stage_group.src_units}
    eq_groups = getattr(stage_group, "eq_class_groups", None)
    s_u_schedules: dict[RaidenId, dict[int, Any]] = {}
    for s_u_idx, s_u in enumerate(stage_group.src_units):
      s_u_shards = (
          stage_group.data_addresses[s_u]
          if s_u in stage_group.data_addresses
          else (
              registered_shards[s_u]
              if registered_shards is not None and s_u in registered_shards
              else []
          )
      )
      shard_host_ranks = controller_types.compute_shard_host_ranks(
          s_u_shards, len(stage_group.src_units), s_u_idx
      )
      var_list = stage_group.stage_ordered_vars_by_unit.get(s_u, [])
      stage_var_to_pid = dict(var_list)
      unique_pids = set(stage_var_to_pid.values())
      shard_plans_by_id: dict[int, dict[int, list[Any]]] = {}
      if eq_groups:
        receiver_set = set(receivers)
        all_local_src_idxs: set[int] = set()
        for sub_g in eq_groups:
          sub_var_plans = sub_g.canonical_variable_plans.get(s_u, {})
          for pid in unique_pids:
            all_local_src_idxs.update(sub_var_plans.get(pid, {}).keys())
        for pid in unique_pids:
          for local_src_idx in sorted(all_local_src_idxs):
            src_host_rank = shard_host_ranks.get(local_src_idx, s_u_idx)
            active_targets: list[tuple[RaidenId, list[str], list[Any]]] = []
            for sub_g in eq_groups:
              tuples_9 = (
                  sub_g.canonical_variable_plans.get(s_u, {})
                  .get(pid, {})
                  .get(local_src_idx, [])
              )
              if not tuples_9:
                continue
              sub_addrs = sub_g.data_addresses
              for d_unit in sub_g.dst_units:
                if d_unit in receiver_set:
                  active_targets.append((d_unit, sub_addrs[d_unit], tuples_9))
            if active_targets:
              shift = src_host_rank % len(active_targets)
              shifted_targets = active_targets[shift:] + active_targets[:shift]
              combined_entries = [
                  (d_addrs[t9[0]], *t9)
                  for _, d_addrs, tuples_9 in shifted_targets
                  for t9 in tuples_9
              ]
              shard_plans_by_id.setdefault(local_src_idx, {})[
                  pid
              ] = combined_entries
      else:
        var_plans = stage_group.canonical_variable_plans.get(s_u, {})
        for pid in unique_pids:
          p_dict = var_plans.get(pid, {})
          for local_src_idx, tuples_9 in p_dict.items():
            if tuples_9:
              src_host_rank = shard_host_ranks[local_src_idx]
              shift = src_host_rank % len(receivers)
              shifted_receivers = receivers[shift:] + receivers[:shift]
              shard_plans_by_id.setdefault(local_src_idx, {})[pid] = [
                  (stage_group.data_addresses[d_unit][t9[0]], *t9)
                  for t9 in tuples_9
                  for d_unit in shifted_receivers
              ]
      s_u_sched = {}
      for local_src_idx in sorted(shard_plans_by_id.keys()):
        ref_sched = controller_types.PlanReferencedShardSchedule(
            shard_plans_by_id[local_src_idx],
            stage_var_to_pid,
            var_list,
            pool_group=stage_group.pool_group,
        )
        s_u_sched[local_src_idx] = ref_sched
      s_u_schedules[s_u] = s_u_sched
    return s_u_schedules

  @classmethod
  def partition_direct_and_broadcast_groups(
      cls,
      groups: dict[tuple[Any, ...], list[tuple[Any, ...]]],
      n_seed: int,
      group_size: int = 1,
  ) -> tuple[
      dict[RaidenId, dict[int, list[Any]]], dict[tuple[Any, ...], list[Any]]
  ]:
    """Partitions slice transfer groups into direct pushes vs.

    tree-broadcast groups.
    """
    if n_seed <= 0:
      raise ValueError(f"n_seed must be positive, got {n_seed}")
    direct_schedules: dict[RaidenId, dict[int, list[Any]]] = {}
    broadcast_groups: dict[tuple[Any, ...], list[Any]] = {}

    for key, targets in groups.items():
      src_unit = key[0]
      shard_idx = key[1]
      layer_idx = key[7]
      pool_group = key[8]

      layer_group_idx = layer_idx // group_size if group_size > 1 else layer_idx

      sorted_targets = sorted(targets, key=lambda t: (t[1], t[2]))
      targets_routing_key = tuple(t[0] for t in sorted_targets)

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
      n_seed: int,
      req_id: str,
      dst_mem_type: int,
      registered_shards: dict[RaidenId, list[str]],
      dst_controller_address: Optional[str] = None,
      src_controller_address: Optional[str] = None,
      pipeline_target_stages: int = _PIPELINE_TARGET_STAGES,
      max_chunk_bytes: int = _RELAY_MAX_COALESCED_CHUNK_BYTES,
  ) -> None:
    """Executes a pipelined multi-hop tree broadcast across multiple groups."""
    diag_vlog = logging.vlog_is_on(1)
    if n_seed <= 0:
      raise ValueError(f"n_seed must be >= 1, got {n_seed}")
    if pipeline_target_stages <= 0:
      raise ValueError(
          f"pipeline_target_stages must be >= 1, got {pipeline_target_stages}"
      )
    if max_chunk_bytes <= 0:
      raise ValueError(f"max_chunk_bytes must be > 0, got {max_chunk_bytes}")

    effective_target_stages, coalesced_groups_list = (
        _coalesce_pipeline_groups_cached(
            groups_list, pipeline_target_stages, n_seed
        )
    )

    groups: list[_GroupBroadcastState] = []
    all_workers: list[RaidenId] = []

    def _add_worker(u: RaidenId) -> None:
      if u not in all_workers:
        all_workers.append(u)

    for g_idx, item in enumerate(coalesced_groups_list):
      if not item:
        continue
      if isinstance(item, controller_types.StageBroadcastGroup):
        for u in item.src_units:
          _add_worker(u)
        for u in item.dst_units:
          _add_worker(u)
        groups.append(_GroupBroadcastState(group_idx=g_idx, stage_group=item))
      else:
        keys_and_sorted_targets = []
        for key, targets in item:
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

    if not groups:
      return

    active_pushes: dict[RaidenId, int] = {u: 0 for u in all_workers}
    ready_queue: dict[RaidenId, list[tuple[int, int, int, int, _HopTask]]] = {
        u: [] for u in all_workers
    }
    hop_counter = itertools.count()
    transfers_in_progress: dict[asyncio.Task[None], _HopTask] = {}

    # Build deterministic All-Source Binomial Tree for each group.
    worker_round_destinations: dict[
        RaidenId, dict[int, tuple[list[str], list[str]]]
    ] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: ([], []))
    )
    pipeline_uuid = (
        final_plan.uuid
        if (final_plan.uuid is not None and final_plan.uuid > 0)
        else random.randint(1, 2**63 - 1)
    )
    receiver_block_counts: dict[RaidenId, int] = collections.defaultdict(int)
    receiver_layer_counts: dict[RaidenId, dict[int, int]] = (
        collections.defaultdict(lambda: collections.defaultdict(int))
    )
    receiver_endpoint_counts: dict[RaidenId, dict[str, int]] = (
        collections.defaultdict(lambda: collections.defaultdict(int))
    )
    receiver_endpoint_layer_counts: dict[
        RaidenId, dict[str, dict[int, int]]
    ] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(int))
    )

    # Receivers fed directly by the trainer (seeds) and by another receiver
    # (relay receivers); see raiden_service_pb2.HostTilingMode.
    seed_receivers: set[RaidenId] = set()
    relay_receivers: set[RaidenId] = set()

    for g in groups:
      dst_units = g.pending_dst_units
      n = len(dst_units)
      if n == 0:
        continue

      sender_child_count: dict[RaidenId, int] = collections.defaultdict(int)
      populated_hops: list[_HopTask] = []
      next_idx = 0
      round_idx = 0

      def _record_hop_destinations(
          s: RaidenId, d: RaidenId, r: int, cur_g: _GroupBroadcastState = g
      ) -> None:
        u_str = str(d)
        dests = worker_round_destinations[s][r]
        seen_pairs = set(zip(dests[0], dests[1]))
        added = False
        if cur_g.stage_group is not None:
          peers = cur_g.stage_group.data_addresses.get(d, [])
          for peer in peers:
            if peer and (u_str, peer) not in seen_pairs:
              seen_pairs.add((u_str, peer))
              dests[0].append(u_str)
              dests[1].append(peer)
              added = True
        else:
          for idx in cur_g.dst_unit_to_indices[d]:
            peer = cur_g.ref_targets[idx][1]
            if peer and (u_str, peer) not in seen_pairs:
              seen_pairs.add((u_str, peer))
              dests[0].append(u_str)
              dests[1].append(peer)
              added = True
        if not added and u_str not in dests[0]:
          dests[0].append(u_str)

      while next_idx < n:
        new_round_hops: list[_HopTask] = []

        # 1. All previously populated samplers each send 1:1 to 1 new sampler
        #    (ordered newest-first so partial final rounds use earliest-free TX NICs).
        for parent_hop in reversed(populated_hops):
          if next_idx >= n:
            break
          d = dst_units[next_idx]
          child_hop = _HopTask(
              group=g,
              sender=parent_hop.receiver,
              receiver=d,
              dst_indices=g.dst_unit_to_indices[d],
              round_idx=round_idx,
              child_order=sender_child_count[parent_hop.receiver],
          )
          sender_child_count[parent_hop.receiver] += 1
          parent_hop.children.append(child_hop)
          new_round_hops.append(child_hop)
          _record_hop_destinations(parent_hop.receiver, d, round_idx)
          next_idx += 1

        # 2. Trainer sends to up to n_seed new samplers in this round.
        is_coalesced_1hop = g.stage_group is not None and bool(
            getattr(g.stage_group, "eq_class_groups", None)
        )
        k_train = (
            (n - next_idx) if is_coalesced_1hop else min(n_seed, n - next_idx)
        )
        train_child_order = sender_child_count[g.primary_src_unit]
        if g.stage_group is not None:
          if k_train > 0:
            round_train_dsts = [dst_units[next_idx + i] for i in range(k_train)]
            seed_hops = []
            combined_dst_indices = []
            for d in round_train_dsts:
              s_hop = _HopTask(
                  group=g,
                  sender=g.primary_src_unit,
                  receiver=d,
                  dst_indices=g.dst_unit_to_indices[d],
                  round_idx=round_idx,
                  child_order=train_child_order,
              )
              seed_hops.append(s_hop)
              new_round_hops.append(s_hop)
              combined_dst_indices.extend(g.dst_unit_to_indices[d])
              _record_hop_destinations(g.primary_src_unit, d, round_idx)
            trainer_hop = _HopTask(
                group=g,
                sender=g.primary_src_unit,
                receiver=round_train_dsts[0],
                dst_indices=combined_dst_indices,
                round_idx=round_idx,
                child_order=train_child_order,
                receivers=round_train_dsts,
                seed_hops=seed_hops,
            )
            if diag_vlog:
              trainer_hop.enqueued_time = time.monotonic()
            heapq.heappush(
                ready_queue[g.primary_src_unit],
                (
                    trainer_hop.child_order,
                    g.stage_idx,
                    g.group_idx,
                    next(hop_counter),
                    trainer_hop,
                ),
            )
            next_idx += k_train
            sender_child_count[g.primary_src_unit] += 1
        else:
          for _ in range(k_train):
            d = dst_units[next_idx]
            s_hop = _HopTask(
                group=g,
                sender=g.primary_src_unit,
                receiver=d,
                dst_indices=g.dst_unit_to_indices[d],
                round_idx=round_idx,
                child_order=train_child_order,
            )
            new_round_hops.append(s_hop)
            if diag_vlog:
              s_hop.enqueued_time = time.monotonic()
            heapq.heappush(
                ready_queue[g.primary_src_unit],
                (
                    s_hop.child_order,
                    g.stage_idx,
                    g.group_idx,
                    next(hop_counter),
                    s_hop,
                ),
            )
            next_idx += 1
            _record_hop_destinations(g.primary_src_unit, d, round_idx)
            sender_child_count[g.primary_src_unit] += 1

        populated_hops.extend(new_round_hops)
        round_idx += 1

      # Accumulate full-transfer receiver chunk expectations across stages so
      # OnLayerDataReceived fires once per layer and OnDataReceived fires only
      # after the final pipeline stage completes for the shared transfer UUID.
      if g.stage_group is not None:
        stage_group = g.stage_group
        sub_groups = getattr(stage_group, "eq_class_groups", None) or [
            stage_group
        ]
        for sub_g in sub_groups:
          sub_dst_set = set(sub_g.dst_units)
          hop_cache = getattr(sub_g, "cached_hop_schedules", None)
          if hop_cache is None:
            hop_cache = getattr(stage_group, "cached_hop_schedules", None)
          layer_counts_key = (
              "__layer_counts__",
              sub_g.pool_group,
              sub_g.layer_group_idx,
          )
          if hop_cache is not None and layer_counts_key in hop_cache:
            (
                seed_shard_layer_counts,
                seed_total_blocks,
                seed_layer_totals,
                relay_shard_layer_counts,
                relay_total_blocks,
                relay_layer_totals,
            ) = hop_cache[layer_counts_key]
          else:
            seed_shard_layer_counts: dict[int, dict[int, int]] = (
                collections.defaultdict(lambda: collections.defaultdict(int))
            )
            relay_shard_layer_counts: dict[int, dict[int, int]] = (
                collections.defaultdict(lambda: collections.defaultdict(int))
            )
            seen_relay_layers: set[int] = set()
            for s_u in sub_g.src_units:
              var_list = sub_g.stage_ordered_vars_by_unit.get(s_u, [])
              var_plans = sub_g.canonical_variable_plans.get(s_u, {})
              pid_seed_dst_counts: dict[int, dict[int, int]] = {}
              for layer_idx, plan_id in var_list:
                dst_counts = pid_seed_dst_counts.get(plan_id)
                if dst_counts is None:
                  dst_counts = collections.defaultdict(int)
                  for tuples_9 in var_plans.get(plan_id, {}).values():
                    for t9 in tuples_9:
                      dst_counts[t9[0]] += 1
                  pid_seed_dst_counts[plan_id] = dst_counts
                for local_dst_idx, cnt in dst_counts.items():
                  seed_shard_layer_counts[local_dst_idx][layer_idx] += cnt
                if layer_idx not in seen_relay_layers:
                  seen_relay_layers.add(layer_idx)
                  relay_shards = sub_g.canonical_relay_plans.get(plan_id, {})
                  # Relays push each pre-tiled shard as a single chunk.
                  for local_dst_idx, blocks in relay_shards.items():
                    if blocks:
                      relay_shard_layer_counts[local_dst_idx][layer_idx] += 1
            seed_layer_totals: dict[int, int] = collections.defaultdict(int)
            seed_total_blocks = 0
            for layer_dict in seed_shard_layer_counts.values():
              for layer_idx, cnt in layer_dict.items():
                seed_layer_totals[layer_idx] += cnt
                seed_total_blocks += cnt
            relay_layer_totals: dict[int, int] = collections.defaultdict(int)
            relay_total_blocks = 0
            for layer_dict in relay_shard_layer_counts.values():
              for layer_idx, cnt in layer_dict.items():
                relay_layer_totals[layer_idx] += cnt
                relay_total_blocks += cnt
            if hop_cache is not None:
              hop_cache[layer_counts_key] = (
                  seed_shard_layer_counts,
                  seed_total_blocks,
                  seed_layer_totals,
                  relay_shard_layer_counts,
                  relay_total_blocks,
                  relay_layer_totals,
              )

          for hop in populated_hops:
            dst_unit = hop.receiver
            if dst_unit not in sub_dst_set:
              continue
            is_seed_hop = hop.sender == g.primary_src_unit
            (seed_receivers if is_seed_hop else relay_receivers).add(dst_unit)
            shard_layer_counts = (
                seed_shard_layer_counts
                if is_seed_hop
                else relay_shard_layer_counts
            )
            hop_total_blocks = (
                seed_total_blocks if is_seed_hop else relay_total_blocks
            )
            hop_layer_totals = (
                seed_layer_totals if is_seed_hop else relay_layer_totals
            )
            receiver_block_counts[dst_unit] += hop_total_blocks
            dst_lc = receiver_layer_counts[dst_unit]
            for layer_idx, cnt in hop_layer_totals.items():
              dst_lc[layer_idx] += cnt

            dst_addrs = sub_g.data_addresses.get(
                dst_unit, stage_group.data_addresses.get(dst_unit, [])
            )
            unique_host_ips = {
                controller_types.extract_host_ip(dst_addrs[local_dst_idx])
                if local_dst_idx < len(dst_addrs) and dst_addrs[local_dst_idx]
                else ""
                for local_dst_idx in shard_layer_counts
            }
            if len(unique_host_ips) == 1:
              only_ip = next(iter(unique_host_ips))
              if only_ip:
                receiver_endpoint_counts[dst_unit][only_ip] += hop_total_blocks
                ep_lc = receiver_endpoint_layer_counts[dst_unit][only_ip]
                for layer_idx, cnt in hop_layer_totals.items():
                  ep_lc[layer_idx] += cnt
            else:
              for local_dst_idx, layer_dict in shard_layer_counts.items():
                dst_peer = (
                    dst_addrs[local_dst_idx]
                    if local_dst_idx < len(dst_addrs)
                    else ""
                )
                host_ip = (
                    controller_types.extract_host_ip(dst_peer)
                    if dst_peer
                    else ""
                )
                if host_ip:
                  ep_lc = receiver_endpoint_layer_counts[dst_unit][host_ip]
                  for layer_idx, cnt in layer_dict.items():
                    receiver_endpoint_counts[dst_unit][host_ip] += cnt
                    ep_lc[layer_idx] += cnt
      else:
        for hop in populated_hops:
          dst_unit = hop.receiver
          s = hop.sender
          if s == g.primary_src_unit:
            for key, k_targets in g.keys_and_sorted_targets:
              k_size = key[4]
              k_s_stride = key[5]
              k_count = key[6]
              k_layer_idx = key[7]
              is_contiguous = (k_count == 1) or (k_s_stride == k_size)
              for idx in hop.dst_indices:
                k_target = k_targets[idx]
                k_dst_peer = k_target[1]
                k_dst_stride = k_target[5]
                entry_contiguous = is_contiguous and (
                    k_count == 1 or k_dst_stride == k_size
                )
                push_count = 1 if entry_contiguous else k_count
                receiver_block_counts[dst_unit] += push_count
                receiver_layer_counts[dst_unit][k_layer_idx] += push_count
                host_ip = (
                    controller_types._extract_host_ip(k_dst_peer)
                    if k_dst_peer
                    else ""
                )
                if host_ip:
                  receiver_endpoint_counts[dst_unit][host_ip] += push_count
                  receiver_endpoint_layer_counts[dst_unit][host_ip][
                      k_layer_idx
                  ] += push_count
          else:
            ref_idx = g.dst_unit_to_indices[s][0]
            tmp_sched: dict[int, list[Any]] = {}
            for key, k_targets in g.keys_and_sorted_targets:
              k_s_target = k_targets[ref_idx]
              k_s_shard_idx = k_s_target[2]
              k_s_block_id = k_s_target[3]
              k_s_block_offset = k_s_target[4]
              k_s_stride = k_s_target[5]
              for idx in hop.dst_indices:
                k_dst_target = k_targets[idx]
                entry = (
                    k_dst_target[1],
                    k_dst_target[2],
                    k_dst_target[4],
                    k_s_block_offset,
                    key[4],
                    k_s_block_id,
                    k_dst_target[3],
                    k_s_stride,
                    k_dst_target[5],
                    key[6],
                    key[7],
                    key[8],
                )
                tmp_sched.setdefault(k_s_shard_idx, []).append(entry)
            for entries in tmp_sched.values():
              coalesced_entries = _coalesce_contiguous_relay_entries(
                  entries, max_chunk_bytes=max_chunk_bytes
              )
              for e in coalesced_entries:
                push_count = (
                    1
                    if (e[9] == 1 or (e[7] == e[4] and e[8] == e[4]))
                    else e[9]
                )
                layer_idx = e[10]
                receiver_block_counts[dst_unit] += push_count
                receiver_layer_counts[dst_unit][layer_idx] += push_count
                host_ip = (
                    controller_types._extract_host_ip(e[0]) if e[0] else ""
                )
                if host_ip:
                  receiver_endpoint_counts[dst_unit][host_ip] += push_count
                  receiver_endpoint_layer_counts[dst_unit][host_ip][
                      layer_idx
                  ] += push_count

      hops_by_round: dict[int, list[str]] = collections.defaultdict(list)
      for tree_hop in populated_hops:
        hops_by_round[tree_hop.round_idx].append(
            f"{controller_types.format_unit(tree_hop.sender)}->"
            f"{controller_types.format_unit(tree_hop.receiver)}"
        )
      for r in sorted(hops_by_round):
        logging.info(
            "Broadcast tree %s: stage %d round %d (%d hop(s)): %s",
            req_id,
            g.group_idx,
            r,
            len(hops_by_round[r]),
            ", ".join(hops_by_round[r]),
        )

    all_pending_dst_units = set()
    for grp in groups:
      all_pending_dst_units.update(grp.pending_dst_units)
    logging.info(
        "Broadcast tree %s: n_seed=%d, %d planner stage group(s) coalesced"
        " into %d pipeline stage(s) (target %d), %d destination unit(s)",
        req_id,
        n_seed,
        len(groups_list),
        len(groups),
        pipeline_target_stages,
        len(all_pending_dst_units),
    )
    round_dests_by_sender: dict[RaidenId, list[Any]] = {}
    for s, rounds_dict in worker_round_destinations.items():
      sorted_rounds = sorted(rounds_dict.keys())
      round_dests_by_sender[s] = [
          controller_types.BroadcastRoundDestinations(
              round_idx=r,
              dst_units=rounds_dict[r][0],
              dst_peers=rounds_dict[r][1],
          )
          for r in sorted_rounds
      ]
    # Seeds tile their host buffers in place and relays forward the tiled
    # bytes. Every stage re-arms a receiver under the same uuid, so it must
    # have the same role in all stages.
    mixed_receivers = seed_receivers & relay_receivers
    if mixed_receivers:
      raise ValueError(
          "Broadcast receivers are fed by both the trainer and relays: "
          f"{sorted(str(u) for u in mixed_receivers)}"
      )

    async def _run_single_transfer(
        s_node: RaidenId,
        d_node: RaidenId,
        plan: Any,
        s_u_plans: Optional[dict[RaidenId, Any]] = None,
        hop: Optional[_HopTask] = None,
        receiver_plans: Optional[dict[RaidenId, Any]] = None,
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
        recv_arm_start = time.monotonic() if diag_vlog else 0.0
        if receiver_plans and hop is not None:
          for d_unit in hop.receivers:
            r_plan = receiver_plans[d_unit]
            if (
                remote_is_sender
                or self._worker_rpc_client.include_receiver_push_schedules(
                    r_plan
                )
            ):
              remote_schedules = r_plan.shard_push_schedules
            else:
              remote_schedules = None
            success = await loop.run_in_executor(
                rpc_executor,
                functools.partial(
                    dst_facade.register_transfer_schedule,
                    r_plan.src_units,
                    [d_unit],
                    r_plan.req_id,
                    True,
                    remote_is_sender,
                    r_plan.expected_block_count,
                    r_plan.uuid,
                    dst_controller_address,
                    src_controller_address,
                    remote_schedules,
                    dst_mem_type,
                    skip_d2h=r_plan.skip_d2h,
                    expected_layer_chunk_counts=(
                        r_plan.dst_expected_layer_chunk_counts.get(
                            d_unit, r_plan.expected_layer_chunk_counts
                        )
                    ),
                    host_tiling_mode=r_plan.host_tiling_mode,
                ),
            )
            if not success:
              raise RuntimeError(
                  "Failed remote prepare in slice tree broadcast"
              )
        else:
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
                  plan.src_units,
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
                  expected_layer_chunk_counts=(
                      plan.dst_expected_layer_chunk_counts.get(
                          d_node, plan.expected_layer_chunk_counts
                      )
                  ),
                  host_tiling_mode=plan.host_tiling_mode,
              ),
          )
          if not success:
            raise RuntimeError("Failed remote prepare in slice tree broadcast")
        if diag_vlog and hop is not None:
          hop.recv_arm_ms = (time.monotonic() - recv_arm_start) * 1000.0
        sender_rpc_start = time.monotonic() if diag_vlog else 0.0
        if s_u_plans:
          await asyncio.gather(*[
              self._worker_rpc_client.start_transfer(s_u, s_u_plans[s_u])
              for s_u in (s_u_plans.keys() if s_u_plans else plan.src_units)
              if s_u in registered_shards
          ])
        elif s_node in registered_shards:
          await self._worker_rpc_client.start_transfer(s_node, plan)
        if diag_vlog and hop is not None:
          hop.sender_rpc_ms = (time.monotonic() - sender_rpc_start) * 1000.0
      else:
        if s_u_plans:
          # Arm destination receiver first, then concurrently dispatch all trainer sources
          recv_arm_start = time.monotonic() if diag_vlog else 0.0
          if receiver_plans and hop is not None:
            await asyncio.gather(*[
                self._worker_rpc_client.start_transfer(
                    d_unit, receiver_plans[d_unit]
                )
                for d_unit in hop.receivers
            ])
          else:
            await self._worker_rpc_client.start_transfer(d_node, plan)
          if diag_vlog and hop is not None:
            hop.recv_arm_ms = (time.monotonic() - recv_arm_start) * 1000.0
          sender_rpc_start = time.monotonic() if diag_vlog else 0.0
          await asyncio.gather(*[
              self._worker_rpc_client.start_transfer(s_u, s_u_plans[s_u])
              for s_u in s_u_plans
              if s_u in registered_shards
          ])
          if diag_vlog and hop is not None:
            hop.sender_rpc_ms = (time.monotonic() - sender_rpc_start) * 1000.0
        else:
          # Arm destination receiver first, then dispatch sender
          recv_arm_start = time.monotonic() if diag_vlog else 0.0
          await self._worker_rpc_client.start_transfer(d_node, plan)
          if diag_vlog and hop is not None:
            hop.recv_arm_ms = (time.monotonic() - recv_arm_start) * 1000.0
          if s_node in registered_shards:
            sender_rpc_start = time.monotonic() if diag_vlog else 0.0
            await self._worker_rpc_client.start_transfer(s_node, plan)
            if diag_vlog and hop is not None:
              hop.sender_rpc_ms = (time.monotonic() - sender_rpc_start) * 1000.0

    def _dispatch_hop(hop: _HopTask) -> None:
      group = hop.group
      s = hop.sender
      dst_unit = hop.receiver
      dst_indices = hop.dst_indices

      if diag_vlog:
        dispatch_time = time.monotonic()
        hop.dispatch_time = dispatch_time
        hop.queue_wait_ms = (
            (dispatch_time - hop.enqueued_time) * 1000.0
            if hop.enqueued_time > 0
            else 0.0
        )
        hop.active_pushes_at_dispatch = active_pushes[s]
      active_pushes[s] += len(hop.receivers)
      hop_uuid = pipeline_uuid
      hop_req_id = f"{req_id}_{group.group_idx}_{dst_unit}_{hop_uuid}"
      hop.hop_req_id = hop_req_id

      plan_build_start = time.monotonic() if diag_vlog else 0.0

      dst_total_blocks = receiver_block_counts[dst_unit]
      dst_layer_counts = dict(receiver_layer_counts[dst_unit])
      dst_ep_counts = dict(receiver_endpoint_counts[dst_unit])
      dst_ep_layer_counts = {
          h: dict(lc)
          for h, lc in receiver_endpoint_layer_counts[dst_unit].items()
      }

      if group.stage_group is not None:
        stage_group = group.stage_group
        hop_cache = getattr(stage_group, "cached_hop_schedules", None)
        dst_addrs = stage_group.data_addresses[dst_unit]
        if s == group.primary_src_unit:
          hop_receivers = hop.receivers
          is_single_stage_1hop = (
              len(groups) == 1
              and hop.round_idx == 0
              and len(hop_receivers) == len(group.pending_dst_units)
          )
          seed_cache_key = (
              "seed",
              stage_group.pool_group,
              stage_group.layer_group_idx,
              tuple(hop_receivers),
              tuple(
                  tuple(stage_group.data_addresses.get(d_u, ()))
                  for d_u in hop_receivers
              ),
          )
          if hop_cache is not None and seed_cache_key in hop_cache:
            s_u_schedules, s_u_expected_blocks = hop_cache[seed_cache_key]
          else:
            if (
                getattr(final_plan, "has_explicit_shard_push_schedules", False)
                or is_single_stage_1hop
            ) and final_plan.shard_push_schedules:
              s_u_schedules = {
                  s_u: final_plan.shard_push_schedules.get(s_u, {})
                  for s_u in stage_group.src_units
              }
            else:
              s_u_schedules = self.build_stage_trainer_push_schedules(
                  stage_group,
                  hop_receivers,
                  registered_shards=registered_shards,
              )
            s_u_expected_blocks = {
                s_u: sum(len(sched) for sched in sched_dict.values())
                for s_u, sched_dict in s_u_schedules.items()
            }
            if hop_cache is not None:
              hop_cache[seed_cache_key] = (s_u_schedules, s_u_expected_blocks)

          use_explicit_plan_counts = bool(
              getattr(final_plan, "has_explicit_shard_push_schedules", False)
              or (
                  stage_group is not None
                  and not getattr(stage_group, "canonical_variable_plans", None)
              )
          )
          receiver_plans = {}
          for d_unit in hop_receivers:
            if use_explicit_plan_counts:
              d_total_blocks = (final_plan.dst_expected_block_counts or {}).get(
                  d_unit, final_plan.expected_block_count or 0
              )
              d_layer_counts = dict(
                  (final_plan.dst_expected_layer_chunk_counts or {}).get(
                      d_unit, {}
                  )
              )
              d_ep_counts = dict(final_plan.dst_endpoint_counts or {})
              d_ep_layer_counts = dict(
                  final_plan.dst_endpoint_layer_counts or {}
              )
            else:
              d_total_blocks = receiver_block_counts[d_unit]
              d_layer_counts = dict(receiver_layer_counts[d_unit])
              d_ep_counts = dict(receiver_endpoint_counts[d_unit])
              d_ep_layer_counts = {
                  h: dict(lc)
                  for h, lc in receiver_endpoint_layer_counts[d_unit].items()
              }
            receiver_plans[d_unit] = type(final_plan)(
                src_units=list(stage_group.src_units),
                dst_units=[d_unit],
                plan=None,
                shard_push_schedules={},
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
                uuid=hop_uuid,
                dst_mem_type=dst_mem_type,
                use_block_chunks=True,
                is_sender=False,
                expected_block_count=d_total_blocks,
                dst_expected_block_counts={d_unit: d_total_blocks},
                expected_layer_chunk_counts=d_layer_counts,
                dst_expected_layer_chunk_counts={d_unit: d_layer_counts},
                dst_endpoint_counts=d_ep_counts,
                dst_endpoint_layer_counts=d_ep_layer_counts,
                req_id=f"{req_id}_{group.group_idx}_{d_unit}_{hop_uuid}",
                skip_d2h=final_plan.skip_d2h,
                skip_tiling=final_plan.skip_tiling,
                parallelism=final_plan.parallelism,
                is_weight_sync=final_plan.is_weight_sync,
                broadcast_round=hop.round_idx,
                broadcast_round_destinations=round_dests_by_sender.get(
                    group.primary_src_unit, []
                ),
                endpoint_to_shards=getattr(
                    final_plan, "endpoint_to_shards", {}
                ),
                host_tiling_mode=raiden_service_pb2.HOST_TILING_MODE_ON_ARRIVAL,
            )

          s_u_plans = {}
          for s_u_idx, s_u in enumerate(stage_group.src_units):
            s_u_shards = (
                stage_group.data_addresses[s_u]
                if s_u in stage_group.data_addresses
                else registered_shards[s_u]
            )
            shard_host_ranks = controller_types.compute_shard_host_ranks(
                s_u_shards, len(stage_group.src_units), s_u_idx
            )
            if len(set(shard_host_ranks.values())) == 1:
              s_u_dsts = (
                  hop_receivers[(s_u_idx % len(hop_receivers)) :]
                  + hop_receivers[: (s_u_idx % len(hop_receivers))]
              )
            else:
              s_u_dsts = list(hop_receivers)

            s_u_plans[s_u] = type(final_plan)(
                src_units=[s_u],
                dst_units=s_u_dsts,
                plan=None,
                shard_push_schedules={s_u: s_u_schedules[s_u]},
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
                peer_control_endpoints=final_plan.peer_control_endpoints,
                peer_control_endpoints_key=(
                    final_plan.peer_control_endpoints_key
                ),
                uuid=hop_uuid,
                dst_mem_type=dst_mem_type,
                use_block_chunks=True,
                is_sender=True,
                expected_block_count=s_u_expected_blocks[s_u],
                req_id=hop_req_id,
                skip_d2h=final_plan.skip_d2h,
                skip_tiling=final_plan.skip_tiling,
                parallelism=final_plan.parallelism,
                is_weight_sync=final_plan.is_weight_sync,
                broadcast_round=hop.round_idx,
                broadcast_round_destinations=round_dests_by_sender.get(
                    group.primary_src_unit, []
                ),
                cached_serialized_payloads=(
                    getattr(final_plan, "cached_serialized_payloads", None)
                    if is_single_stage_1hop
                    else getattr(final_plan, "cached_serialized_payloads", {})
                ),
                sender_push_schedule_protos=(
                    getattr(final_plan, "sender_push_schedule_protos", None)
                    if is_single_stage_1hop
                    else None
                ),
                endpoint_to_shards=dict(
                    getattr(final_plan, "endpoint_to_shards", None) or {}
                ),
            )

          if diag_vlog:
            hop.plan_build_ms = (time.monotonic() - plan_build_start) * 1000.0
          task = asyncio.create_task(
              _run_single_transfer(
                  group.primary_src_unit,
                  hop_receivers[0],
                  receiver_plans[hop_receivers[0]],
                  s_u_plans=s_u_plans,
                  hop=hop,
                  receiver_plans=receiver_plans,
              )
          )
          transfers_in_progress[task] = hop
        else:
          # Sampler -> Sampler relay: 1-to-1 whole-block transfer
          relay_cache_key = (
              "relay",
              stage_group.pool_group,
              stage_group.layer_group_idx,
              dst_unit,
              tuple(dst_addrs),
          )
          if hop_cache is not None and relay_cache_key in hop_cache:
            relay_shard_sched = hop_cache[relay_cache_key]
          else:
            seen_relay_layers = set()
            ordered_relay_vars = []
            for s_u in stage_group.src_units:
              for (
                  layer_idx,
                  plan_id,
              ) in stage_group.stage_ordered_vars_by_unit.get(s_u, []):
                if layer_idx not in seen_relay_layers:
                  seen_relay_layers.add(layer_idx)
                  ordered_relay_vars.append((layer_idx, plan_id))
            relay_var_to_pid = dict(ordered_relay_vars)
            unique_relay_pids = set(relay_var_to_pid.values())
            relay_shard_plans_by_id: dict[int, dict[int, list[Any]]] = {}
            for pid in unique_relay_pids:
              relay_shards = stage_group.canonical_relay_plans.get(pid, {})
              for local_dst_idx, blocks in relay_shards.items():
                if blocks:
                  # The relay holds this shard in device layout, so it sends
                  # the whole shard in one chunk; the sender resolves the
                  # physical size (HOST_TILING_MODE_PRE_TILED), so the
                  # logical span here is informational only.
                  span = max(off + size for off, size, _ in blocks)
                  dst_block_id = blocks[0][2]
                  relay_shard_plans_by_id.setdefault(local_dst_idx, {})[pid] = (
                      [(
                          dst_addrs[local_dst_idx],
                          local_dst_idx,
                          0,
                          0,
                          span,
                          dst_block_id,
                          dst_block_id,
                          span,
                          span,
                          1,
                      )]
                  )
            relay_shard_sched = {
                local_dst_idx: controller_types.PlanReferencedShardSchedule(
                    relay_shard_plans_by_id[local_dst_idx],
                    relay_var_to_pid,
                    ordered_relay_vars,
                    pool_group=stage_group.pool_group,
                )
                for local_dst_idx in sorted(relay_shard_plans_by_id.keys())
            }
            if hop_cache is not None:
              hop_cache[relay_cache_key] = relay_shard_sched
          sub_schedule: dict[RaidenId, dict[int, Any]] = {s: relay_shard_sched}

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
              peer_control_endpoints=final_plan.peer_control_endpoints,
              peer_control_endpoints_key=(
                  final_plan.peer_control_endpoints_key
              ),
              uuid=hop_uuid,
              dst_mem_type=dst_mem_type,
              use_block_chunks=True,
              is_sender=True,
              expected_block_count=dst_total_blocks,
              dst_expected_block_counts={dst_unit: dst_total_blocks},
              expected_layer_chunk_counts=dst_layer_counts,
              dst_expected_layer_chunk_counts={dst_unit: dst_layer_counts},
              dst_endpoint_counts=dst_ep_counts,
              dst_endpoint_layer_counts=dst_ep_layer_counts,
              req_id=hop_req_id,
              skip_d2h=True,
              skip_tiling=final_plan.skip_tiling,
              parallelism=final_plan.parallelism,
              is_weight_sync=final_plan.is_weight_sync,
              broadcast_round=hop.round_idx,
              broadcast_round_destinations=round_dests_by_sender.get(s, []),
              cached_serialized_payloads=getattr(
                  final_plan, "cached_serialized_payloads", {}
              ),
              endpoint_to_shards=dict(
                  getattr(final_plan, "endpoint_to_shards", None) or {}
              ),
              host_tiling_mode=raiden_service_pb2.HOST_TILING_MODE_PRE_TILED,
          )
          if diag_vlog:
            hop.plan_build_ms = (time.monotonic() - plan_build_start) * 1000.0
          task = asyncio.create_task(
              _run_single_transfer(s, dst_unit, sub_plan, hop=hop)
          )
          transfers_in_progress[task] = hop
      else:
        # Legacy slice-list mode
        sub_schedule: dict[RaidenId, dict[int, list[Any]]] = {s: {}}
        if s == group.primary_src_unit:
          for key, k_targets in group.keys_and_sorted_targets:
            k_s_shard_idx = key[1]
            k_s_block_id = key[2]
            k_s_block_offset = key[3]
            k_size = key[4]
            k_s_stride = key[5]
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
              sub_schedule[s].setdefault(k_s_shard_idx, []).append(entry)
        else:
          ref_idx = group.dst_unit_to_indices[s][0]
          for key, k_targets in group.keys_and_sorted_targets:
            k_s_target = k_targets[ref_idx]
            k_s_shard_idx = k_s_target[2]
            k_s_block_id = k_s_target[3]
            k_s_block_offset = k_s_target[4]
            k_s_stride = k_s_target[5]

            for idx in dst_indices:
              k_dst_target = k_targets[idx]
              (
                  _,
                  k_dst_peer,
                  k_dst_shard_idx,
                  k_dst_block_id,
                  k_dst_block_offset,
                  k_dst_stride,
              ) = k_dst_target

              entry = (
                  k_dst_peer,
                  k_dst_shard_idx,
                  k_dst_block_offset,
                  k_s_block_offset,
                  key[4],
                  k_s_block_id,
                  k_dst_block_id,
                  k_s_stride,
                  k_dst_stride,
                  key[6],
                  key[7],
                  key[8],
              )
              sub_schedule[s].setdefault(k_s_shard_idx, []).append(entry)

          for shard_idx, entries in list(sub_schedule[s].items()):
            sub_schedule[s][shard_idx] = _coalesce_contiguous_relay_entries(
                entries,
                max_chunk_bytes=max_chunk_bytes,
            )

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
            peer_control_endpoints=final_plan.peer_control_endpoints,
            peer_control_endpoints_key=final_plan.peer_control_endpoints_key,
            uuid=hop_uuid,
            dst_mem_type=dst_mem_type,
            use_block_chunks=True,
            is_sender=True,
            expected_block_count=dst_total_blocks,
            dst_expected_block_counts={dst_unit: dst_total_blocks},
            expected_layer_chunk_counts=dst_layer_counts,
            dst_expected_layer_chunk_counts={dst_unit: dst_layer_counts},
            dst_endpoint_counts=dst_ep_counts,
            dst_endpoint_layer_counts=dst_ep_layer_counts,
            req_id=hop_req_id,
            skip_d2h=final_plan.skip_d2h or (s != group.src_unit),
            skip_tiling=final_plan.skip_tiling,
            parallelism=final_plan.parallelism,
            is_weight_sync=final_plan.is_weight_sync,
            broadcast_round=hop.round_idx,
            broadcast_round_destinations=round_dests_by_sender.get(s, []),
            cached_serialized_payloads=getattr(
                final_plan, "cached_serialized_payloads", {}
            ),
            endpoint_to_shards=dict(
                getattr(final_plan, "endpoint_to_shards", None) or {}
            ),
        )
        if diag_vlog:
          hop.plan_build_ms = (time.monotonic() - plan_build_start) * 1000.0
        task = asyncio.create_task(
            _run_single_transfer(s, dst_unit, sub_plan, hop=hop)
        )
        transfers_in_progress[task] = hop

    async def _monitor_event_loop_lag() -> None:
      step = 0.1
      threshold = 0.25
      try:
        while True:
          t0 = time.monotonic()
          await asyncio.sleep(step)
          elapsed = time.monotonic() - t0
          lag = elapsed - step
          if lag > threshold:
            logging.vlog(
                1,
                "RAIDEN_DIAG loop_lag req_id=%s lag_ms=%.2f elapsed_ms=%.2f"
                " expected_ms=%.2f",
                req_id,
                lag * 1000.0,
                elapsed * 1000.0,
                step * 1000.0,
            )
      except asyncio.CancelledError:
        pass

    lag_monitor_task = (
        asyncio.create_task(_monitor_event_loop_lag())
        if logging.vlog_is_on(1)
        else None
    )

    src_units = set(g.src_unit for g in groups)
    num_dst_eq_classes = max(
        1,
        len({
            tuple(sub.dst_units)
            for g in groups
            if g.stage_group is not None
            for sub in getattr(g.stage_group, "eq_class_groups", None) or [
                g.stage_group
            ]
        }),
    )
    try:
      while any(ready_queue.values()) or transfers_in_progress:
        while True:
          scheduled_any = False
          for u in all_workers:
            max_active = n_seed * num_dst_eq_classes if u in src_units else 1
            while (
                ready_queue[u]
                and active_pushes[u] + len(ready_queue[u][0][4].receivers)
                <= max_active
            ):
              _, _, _, _, hop = heapq.heappop(ready_queue[u])
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
            hop = transfers_in_progress.pop(fut)
            active_pushes[hop.sender] -= len(hop.receivers)
            if diag_vlog:
              total_ms = (time.monotonic() - hop.dispatch_time) * 1000.0
              sender_str = controller_types.format_unit(hop.sender)
              receiver_str = (
                  controller_types.format_unit(hop.receiver)
                  if len(hop.receivers) == 1
                  else controller_types.format_units(hop.receivers)
              )

            exc = fut.exception()
            if exc is not None:
              if diag_vlog:
                logging.vlog(
                    1,
                    "RAIDEN_DIAG hop status=FAILED hop_req_id=%s %s->%s"
                    " round=%d stage_group=%d queue_wait_ms=%.2f"
                    " plan_build_ms=%.2f recv_arm_ms=%.2f sender_rpc_ms=%.2f"
                    " total_ms=%.2f active_pushes=%d error=%s",
                    hop.hop_req_id,
                    sender_str,
                    receiver_str,
                    hop.round_idx,
                    hop.group.group_idx,
                    hop.queue_wait_ms,
                    hop.plan_build_ms,
                    hop.recv_arm_ms,
                    hop.sender_rpc_ms,
                    total_ms,
                    hop.active_pushes_at_dispatch,
                    exc,
                )
              logging.error("Slice transfer failed in broadcast tree: %s", exc)
              raise exc

            if diag_vlog:
              logging.vlog(
                  1,
                  "RAIDEN_DIAG hop status=COMPLETE hop_req_id=%s %s->%s"
                  " round=%d stage_group=%d queue_wait_ms=%.2f"
                  " plan_build_ms=%.2f recv_arm_ms=%.2f sender_rpc_ms=%.2f"
                  " total_ms=%.2f active_pushes=%d",
                  hop.hop_req_id,
                  sender_str,
                  receiver_str,
                  hop.round_idx,
                  hop.group.group_idx,
                  hop.queue_wait_ms,
                  hop.plan_build_ms,
                  hop.recv_arm_ms,
                  hop.sender_rpc_ms,
                  total_ms,
                  hop.active_pushes_at_dispatch,
              )

            hops_to_propagate = hop.seed_hops if hop.seed_hops else [hop]
            for completed_hop in hops_to_propagate:
              for child_hop in completed_hop.children:
                if diag_vlog:
                  child_hop.enqueued_time = time.monotonic()
                heapq.heappush(
                    ready_queue[completed_hop.receiver],
                    (
                        child_hop.child_order,
                        child_hop.group.stage_idx,
                        child_hop.group.group_idx,
                        next(hop_counter),
                        child_hop,
                    ),
                )
    finally:
      if lag_monitor_task is not None:
        lag_monitor_task.cancel()
        try:
          await lag_monitor_task
        except asyncio.CancelledError:
          pass
      pending = [f for f in transfers_in_progress.keys() if not f.done()]
      for f in pending:
        f.cancel()
      if pending:
        await asyncio.gather(*pending, return_exceptions=True)

  async def execute_slice_broadcast(
      self,
      keys_and_targets: list[tuple[tuple[Any, ...], list[tuple[Any, ...]]]],
      final_plan: Any,
      n_seed: int,
      req_id: str,
      dst_mem_type: int,
      registered_shards: dict[RaidenId, list[str]],
      dst_controller_address: Optional[str] = None,
      src_controller_address: Optional[str] = None,
      pipeline_target_stages: int = _PIPELINE_TARGET_STAGES,
      max_chunk_bytes: int = _RELAY_MAX_COALESCED_CHUNK_BYTES,
  ) -> None:
    """Executes a pipelined tree broadcast for a group of variables."""
    await self.execute_slice_broadcast_pipeline(
        groups_list=[keys_and_targets],
        final_plan=final_plan,
        n_seed=n_seed,
        req_id=req_id,
        dst_mem_type=dst_mem_type,
        registered_shards=registered_shards,
        dst_controller_address=dst_controller_address,
        src_controller_address=src_controller_address,
        pipeline_target_stages=pipeline_target_stages,
        max_chunk_bytes=max_chunk_bytes,
    )
