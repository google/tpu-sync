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
import functools
import random
from typing import Any, Callable, Optional

from absl import logging

from tpu_sync.api.common import RaidenId


class BroadcastEngine:
  """Executes pipelined multi-hop tree broadcasts for grouped tensor slices."""

  def __init__(
      self,
      worker_rpc_client: Any,
      remote_controller_client_factory: Optional[Callable[..., Any]] = None,
  ) -> None:
    self._worker_rpc_client = worker_rpc_client
    self._remote_client_factory = remote_controller_client_factory

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
    keys_and_sorted_targets = []
    for key, targets in keys_and_targets:
      sorted_t = sorted(targets, key=lambda t: (t[1], t[2]))
      keys_and_sorted_targets.append((key, sorted_t))

    ref_key, ref_targets = keys_and_sorted_targets[0]
    (
        src_unit,
        shard_idx,
        _,  # src_block_id
        _,  # src_block_offset
        _,  # size
        _,  # src_stride
        _,  # count
        _,  # layer_idx
        _,  # pool_group
    ) = ref_key

    available_sources = [src_unit]
    node_slice_offsets = {
        src_unit: {
            key: (shard_idx, key[2], key[3], key[5])
            for key, _ in keys_and_sorted_targets
        }
    }

    pending_indices = list(range(len(ref_targets)))
    active_pushes = {u: 0 for u in [src_unit] + [t[0] for t in ref_targets]}
    transfers_in_progress: dict[
        RaidenId, tuple[RaidenId, RaidenId, list[int], asyncio.Task[None]]
    ] = {}

    while pending_indices or transfers_in_progress:
      # 1. Greedy assignment step
      scheduled_any = False
      for s in list(available_sources):
        while active_pushes[s] < fanout_k and pending_indices:
          first_idx = pending_indices[0]
          first_target = ref_targets[first_idx]
          dst_unit = first_target[0]

          dst_indices = [
              idx for idx in pending_indices if ref_targets[idx][0] == dst_unit
          ]
          pending_indices = [
              idx for idx in pending_indices if idx not in dst_indices
          ]

          active_pushes[s] += 1
          scheduled_any = True

          ref_offset = node_slice_offsets[s][ref_key]
          s_shard_idx = ref_offset[0]

          sub_schedule: dict[RaidenId, dict[int, list[Any]]] = {
              s: {s_shard_idx: []}
          }
          hop_expected_block_count = 0
          for key, k_targets in keys_and_sorted_targets:
            _, k_s_block_id, k_s_block_offset, k_s_stride = node_slice_offsets[
                s
            ][key]
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

              is_contiguous = (k_count == 1) or (
                  k_s_stride == k_size and k_dst_stride == k_size
              )
              push_count = 1 if is_contiguous else k_count
              hop_expected_block_count += push_count

          hop_uuid = random.randint(1, 2**63 - 1)
          hop_req_id = f"{req_id}_{hop_uuid}"

          sub_plan = type(final_plan)(
              src_units=[s],
              dst_units=[dst_unit],
              plan=None,
              shard_push_schedules=sub_schedule,
              worker_rpc_addresses=dict(final_plan.worker_rpc_addresses),
              worker_data_addresses=dict(final_plan.worker_data_addresses),
              uuid=hop_uuid,
              dst_mem_type=dst_mem_type,
              use_block_chunks=True,
              is_sender=True,
              expected_block_count=hop_expected_block_count,
              req_id=hop_req_id,
              skip_d2h=final_plan.skip_d2h or (s != src_unit),
              skip_tiling=final_plan.skip_tiling,
              is_weight_sync=final_plan.is_weight_sync,
          )

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
                  or self._worker_rpc_client.include_receiver_push_schedules(
                      plan
                  )
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
                raise RuntimeError(
                    "Failed remote prepare in slice tree broadcast"
                )
            else:
              await self._worker_rpc_client.start_transfer(d_node, plan)

            if s_node in registered_shards:
              await self._worker_rpc_client.start_transfer(s_node, plan)

          task = asyncio.create_task(
              _run_single_transfer(s, dst_unit, sub_plan)
          )
          transfers_in_progress[dst_unit] = (s, dst_unit, dst_indices, task)

      # 2. Wait step
      if transfers_in_progress:
        futures_to_dsts = {
            info[3]: dst for dst, info in transfers_in_progress.items()
        }
        done, _ = await asyncio.wait(
            futures_to_dsts.keys(), return_when=asyncio.FIRST_COMPLETED
        )

        # 3. Promotion step
        for fut in done:
          exc = fut.exception()
          if exc is not None:
            logging.error("Slice transfer failed in broadcast tree: %s", exc)
            for info in transfers_in_progress.values():
              info[3].cancel()
            raise exc
          dst_unit = futures_to_dsts[fut]
          s, _, dst_indices, _ = transfers_in_progress.pop(dst_unit)
          active_pushes[s] -= 1
          available_sources.append(dst_unit)

          ref_idx = dst_indices[0]
          node_slice_offsets[dst_unit] = {}
          for key, k_targets in keys_and_sorted_targets:
            k_target = k_targets[ref_idx]
            (
                _,
                _,
                k_dst_shard_idx,
                k_dst_block_id,
                k_dst_block_offset,
                k_dst_stride,
            ) = k_target
            node_slice_offsets[dst_unit][key] = (
                k_dst_shard_idx,
                k_dst_block_id,
                k_dst_block_offset,
                k_dst_stride,
            )
      elif not scheduled_any:
        break
