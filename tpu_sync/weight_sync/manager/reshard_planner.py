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

"""Resharding plan and N-D slice math for RaidenController."""

import collections
import math
import threading
from typing import Any, Mapping, Optional, Sequence

from absl import logging

from tpu_sync.api.common import RaidenId
from tpu_sync.kv_cache import nd_slice_math
from tpu_sync.weight_sync.manager import broadcast_engine
from tpu_sync.weight_sync.manager import controller_types
from tpu_sync.weight_sync.manager import job_entity

BroadcastEngine = broadcast_engine.BroadcastEngine
JobEntity = job_entity.JobEntity
_CachedTransferSchedule = controller_types.CachedTransferSchedule
_PlanReferencedShardSchedule = controller_types.PlanReferencedShardSchedule
_VariableMetadata = controller_types.VariableMetadata
_extract_host_ip = controller_types.extract_host_ip
_format_units = controller_types.format_units
_is_variable_spec_identical = controller_types.is_variable_spec_identical
_proto_to_nd_slice = controller_types.proto_to_nd_slice
_raiden_id_from_proto = controller_types.raiden_id_from_proto
_StageBroadcastGroup = controller_types.StageBroadcastGroup

_PIPELINE_TARGET_STAGES: int = 4
_RELAY_MAX_COALESCED_CHUNK_BYTES: int = 4 * 1024 * 1024


def to_physical(logical_shape, logical_mesh_shape, minor_to_major):
  """Maps logical tensor and mesh shapes to physical memory layout."""
  logical_shape = list(logical_shape)
  logical_mesh_shape = list(logical_mesh_shape)
  minor_to_major = list(minor_to_major)
  major_to_minor = list(reversed(minor_to_major))
  rank = len(logical_shape)
  if sorted(minor_to_major) == list(range(rank)):
    physical_shape = tuple(logical_shape[d] for d in major_to_minor)
    physical_mesh_shape = tuple(logical_mesh_shape[d] for d in major_to_minor)
  else:
    physical_shape = tuple(
        logical_shape[minor_to_major.index(d)] for d in major_to_minor
    )
    physical_mesh_shape = tuple(logical_mesh_shape[d] for d in major_to_minor)
  return physical_shape, physical_mesh_shape


def _get_global_indices(
    unit: RaidenId,
    shards: list[str],
    logical_mesh_shape: list[int],
    layout: list[int],
    num_physical_hosts: int,
    global_shard_indices: Optional[list[int]] = None,
) -> list[tuple[int, int]]:
  """Maps local shard indices to global slice indices, handling replication.

  When global_shard_indices is provided, this function bypasses all mesh
  geometry and coordinate mapping, directly returning the 1:1 mapping between
  local shards and their declared global slice indices. Otherwise the legacy
  host-axis mapping is used: the logical mesh axis whose size equals
  num_physical_hosts is indexed by the unit's job_replica_id and the remaining
  axes are enumerated in row-major order.

  Args:
    unit: The RaidenId of the work unit.
    shards: List of data-plane shard endpoint strings.
    logical_mesh_shape: Logical device mesh dimensions.
    layout: Shard dimension permutation mapping.
    num_physical_hosts: Total physical host count.
    global_shard_indices: Explicit vector of global shard indices.

  Returns:
    List of (local_shard_idx, global_slice_idx) pairs.
  """
  num_shards = len(shards)
  if num_shards == 0:
    return []

  if global_shard_indices is not None:
    if len(global_shard_indices) == num_shards:
      return [(j, int(g_idx)) for j, g_idx in enumerate(global_shard_indices)]
    logging.warning(
        "global_shard_indices length (%d) does not match num_shards (%d) for"
        " unit %s. Falling back to mesh geometry calculation.",
        len(global_shard_indices),
        num_shards,
        unit,
    )

  try:
    replica_id = int(unit.job_replica_id)
  except ValueError:
    replica_id = 0

  if not logical_mesh_shape:
    return [(i, i) for i in range(num_shards)]

  if all(d == 1 for d in logical_mesh_shape):
    return [(i, 0) for i in range(num_shards)]

  major_to_minor = list(reversed(layout))
  phys_mesh = [logical_mesh_shape[d] for d in major_to_minor]

  host_axis_logical = None
  for d, size in enumerate(logical_mesh_shape):
    if size == num_physical_hosts:
      host_axis_logical = d
      break

  if host_axis_logical is None:
    logging.warning(
        "No logical mesh axis of size num_physical_hosts=%d for %s. Falling"
        " back to legacy mapping.",
        num_physical_hosts,
        unit,
    )

  non_host_axes = [
      d for d in range(len(logical_mesh_shape)) if d != host_axis_logical
  ]

  indices = []
  for j in range(num_shards):
    local_coords = {}
    temp = j
    for d in reversed(non_host_axes):
      size = logical_mesh_shape[d]
      local_coords[d] = temp % size
      temp = temp // size

    full_coords = [0] * len(logical_mesh_shape)
    for d in range(len(logical_mesh_shape)):
      if d == host_axis_logical:
        full_coords[d] = replica_id
      else:
        full_coords[d] = local_coords[d]

    tensor_coords = [full_coords[m_axis] for m_axis in major_to_minor]

    global_idx = 0
    stride = 1
    for val, size in zip(reversed(tensor_coords), reversed(phys_mesh)):
      global_idx += val * stride
      stride *= size
    indices.append((j, global_idx))
  return indices


get_global_indices = _get_global_indices


def intersect_nd_slices(
    slice1: list[tuple[int, int]], slice2: list[tuple[int, int]]
) -> Optional[list[tuple[int, int]]]:
  """Computes the precise N-dimensional intersection bounding box between two multi-dimensional slices.

  Each slice is represented as a list of coordinate bounds (start, end) for
  each dimension.

  Args:
    slice1: First N-dimensional slice bounding box.
    slice2: Second N-dimensional slice bounding box.

  Returns:
    A list of (start, end) coordinate bounds representing the intersecting
    subgrid, or None if the slices do not overlap in any dimension.
  """
  result = []
  for (s1, e1), (s2, e2) in zip(slice1, slice2):
    start = max(s1, s2)
    end = min(e1, e2)
    if start >= end:
      return None
    result.append((start, end))
  return result


def generate_strided_copy_chunks(
    src_shard_slice: list[tuple[int, int]],
    dst_shard_slice: list[tuple[int, int]],
    intersection_slice: list[tuple[int, int]],
    itemsize: int,
) -> list[tuple[int, int, int, int, int, int]]:
  """Translates an N-dimensional grid intersection into strided memory copy chunks."""
  rank = len(src_shard_slice)
  if rank == 0:
    return [(0, 0, itemsize, 0, 0, 1)]
  if rank == 1:
    s_s, _ = src_shard_slice[0]
    d_s, _ = dst_shard_slice[0]
    i_s, i_e = intersection_slice[0]
    size = (i_e - i_s) * itemsize
    return [((i_s - s_s) * itemsize, (i_s - d_s) * itemsize, size, 0, 0, 1)]
  src_shape = [e - s for s, e in src_shard_slice]
  dst_shape = [e - s for s, e in dst_shard_slice]
  int_shape = [e - s for s, e in intersection_slice]

  # Calculate how many inner dimensions can be merged into a contiguous chunk
  split_dim = -1
  for d in range(rank - 1, -1, -1):
    dim_size = int_shape[d]
    src_full = dim_size == src_shape[d]
    dst_full = dim_size == dst_shape[d]
    if not (src_full and dst_full):
      split_dim = d
      break

  if split_dim != -1:
    contiguous_elements = math.prod(int_shape[max(1, split_dim) :])
    stride_dim = max(0, split_dim - 1)
  else:
    contiguous_elements = math.prod(int_shape)
    stride_dim = -1

  contiguous_bytes = contiguous_elements * itemsize

  src_strides = [1] * rank
  for i in range(rank - 2, -1, -1):
    src_strides[i] = src_strides[i + 1] * src_shape[i + 1]

  dst_strides = [1] * rank
  for i in range(rank - 2, -1, -1):
    dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1]

  if stride_dim >= 0:
    count = int_shape[stride_dim]
    src_stride = src_strides[stride_dim] * itemsize
    dst_stride = dst_strides[stride_dim] * itemsize
    outer_shape = int_shape[:stride_dim]
  else:
    count = 1
    src_stride = 0
    dst_stride = 0
    outer_shape = []

  num_outer_elements = math.prod(outer_shape) if outer_shape else 1

  src_local_int_slice = [
      (int_s - src_s, int_e - src_s)
      for (src_s, _), (int_s, int_e) in zip(src_shard_slice, intersection_slice)
  ]
  dst_local_int_slice = [
      (int_s - dst_s, int_e - dst_s)
      for (dst_s, _), (int_s, int_e) in zip(dst_shard_slice, intersection_slice)
  ]

  chunks = []
  for i in range(num_outer_elements):
    multi_index = []
    temp = i
    for dim_size in reversed(outer_shape):
      multi_index.append(temp % dim_size)
      temp //= dim_size
    multi_index.reverse()

    src_offset_items = 0
    dst_offset_items = 0

    for d in range(len(outer_shape)):
      src_idx = src_local_int_slice[d][0] + multi_index[d]
      src_offset_items += src_idx * src_strides[d]

      dst_idx = dst_local_int_slice[d][0] + multi_index[d]
      dst_offset_items += dst_idx * dst_strides[d]

    start_d = len(outer_shape)
    for d in range(start_d, rank):
      src_offset_items += src_local_int_slice[d][0] * src_strides[d]
      dst_offset_items += dst_local_int_slice[d][0] * dst_strides[d]

    chunks.append((
        src_offset_items * itemsize,
        dst_offset_items * itemsize,
        contiguous_bytes,
        src_stride,
        dst_stride,
        count,
    ))

  return chunks


def is_nd_slice_tile_aligned(
    src_shard_slice: list[tuple[int, int]],
    dst_shard_slice: list[tuple[int, int]],
    intersection_slice: list[tuple[int, int]],
    tile_shape: tuple[int, int] = (8, 128),
) -> bool:
  """Checks if slices and their intersection align with hardware tile boundaries."""
  rank = len(src_shard_slice)
  if rank < 2:
    return False
  t_row, t_col = tile_shape
  s_row_s, s_row_e = src_shard_slice[-2]
  s_col_s, s_col_e = src_shard_slice[-1]
  d_row_s, d_row_e = dst_shard_slice[-2]
  d_col_s, d_col_e = dst_shard_slice[-1]
  i_row_s, i_row_e = intersection_slice[-2]
  i_col_s, i_col_e = intersection_slice[-1]

  if (i_row_s - s_row_s) % t_row != 0 or (i_row_s - d_row_s) % t_row != 0:
    return False
  if (i_col_s - s_col_s) % t_col != 0 or (i_col_s - d_col_s) % t_col != 0:
    return False
  if (i_row_e - i_row_s) % t_row != 0:
    return False
  if (i_col_e - i_col_s) % t_col != 0:
    return False
  if (s_col_e - s_col_s) % t_col != 0 or (d_col_e - d_col_s) % t_col != 0:
    return False
  if (s_row_e - s_row_s) % t_row != 0 or (d_row_e - d_row_s) % t_row != 0:
    return False
  return True


def generate_strided_copy_chunks_tile_aware(
    src_shard_slice: list[tuple[int, int]],
    dst_shard_slice: list[tuple[int, int]],
    intersection_slice: list[tuple[int, int]],
    itemsize: int,
    tile_shape: tuple[int, int] = (8, 128),
) -> list[tuple[int, int, int, int, int, int]]:
  """Translates an N-dimensional grid intersection into tile-aware physical strided copy chunks."""
  rank = len(src_shard_slice)
  if rank == 0:
    return [(0, 0, itemsize, 0, 0, 1)]
  if rank == 1:
    s_s, _ = src_shard_slice[0]
    d_s, _ = dst_shard_slice[0]
    i_s, i_e = intersection_slice[0]
    size = (i_e - i_s) * itemsize
    return [((i_s - s_s) * itemsize, (i_s - d_s) * itemsize, size, 0, 0, 1)]

  t_row, _ = tile_shape
  s_row_s, s_row_e = src_shard_slice[-2]
  s_col_s, s_col_e = src_shard_slice[-1]
  d_row_s, d_row_e = dst_shard_slice[-2]
  d_col_s, d_col_e = dst_shard_slice[-1]
  i_row_s, i_row_e = intersection_slice[-2]
  i_col_s, i_col_e = intersection_slice[-1]

  w_src = s_col_e - s_col_s
  w_dst = d_col_e - d_col_s
  w_int = i_col_e - i_col_s
  h_int = i_row_e - i_row_s

  local_src_row = i_row_s - s_row_s
  local_src_col = i_col_s - s_col_s
  local_dst_row = i_row_s - d_row_s
  local_dst_col = i_col_s - d_col_s

  size_bytes = w_int * t_row * itemsize
  src_stride = w_src * t_row * itemsize
  dst_stride = w_dst * t_row * itemsize
  count = h_int // t_row

  src_offset = (
      local_src_row * w_src * itemsize + local_src_col * t_row * itemsize
  )
  dst_offset = (
      local_dst_row * w_dst * itemsize + local_dst_col * t_row * itemsize
  )

  if rank > 2:
    num_outer_dims = rank - 2
    outer_shape = [
        intersection_slice[d][1] - intersection_slice[d][0]
        for d in range(num_outer_dims)
    ]
    src_outer_strides = [1] * num_outer_dims
    src_outer_strides[-1] = (s_row_e - s_row_s) * (s_col_e - s_col_s)
    for d in range(num_outer_dims - 2, -1, -1):
      dim_len = src_shard_slice[d + 1][1] - src_shard_slice[d + 1][0]
      src_outer_strides[d] = src_outer_strides[d + 1] * dim_len

    dst_outer_strides = [1] * num_outer_dims
    dst_outer_strides[-1] = (d_row_e - d_row_s) * (d_col_e - d_col_s)
    for d in range(num_outer_dims - 2, -1, -1):
      dim_len = dst_shard_slice[d + 1][1] - dst_shard_slice[d + 1][0]
      dst_outer_strides[d] = dst_outer_strides[d + 1] * dim_len

    src_local_outer_start = [
        intersection_slice[d][0] - src_shard_slice[d][0]
        for d in range(num_outer_dims)
    ]
    dst_local_outer_start = [
        intersection_slice[d][0] - dst_shard_slice[d][0]
        for d in range(num_outer_dims)
    ]

    num_outer = math.prod(outer_shape) if outer_shape else 1
    if (
        count == 1 or (size_bytes == src_stride and size_bytes == dst_stride)
    ) and num_outer_dims == 1:
      inner_size = size_bytes * count
      s_stride_outer = src_outer_strides[-1] * itemsize
      d_stride_outer = dst_outer_strides[-1] * itemsize
      base_src = src_offset + src_local_outer_start[0] * s_stride_outer
      base_dst = dst_offset + dst_local_outer_start[0] * d_stride_outer
      if inner_size == s_stride_outer and inner_size == d_stride_outer:
        return [(base_src, base_dst, inner_size * num_outer, 0, 0, 1)]
      else:
        return [(
            base_src,
            base_dst,
            inner_size,
            s_stride_outer,
            d_stride_outer,
            num_outer,
        )]

    chunks = []
    for i in range(num_outer):
      multi_idx = []
      temp = i
      for dim_size in reversed(outer_shape):
        multi_idx.append(temp % dim_size)
        temp //= dim_size
      multi_idx.reverse()

      outer_src_bytes = sum(
          (src_local_outer_start[d] + multi_idx[d])
          * src_outer_strides[d]
          * itemsize
          for d in range(num_outer_dims)
      )
      outer_dst_bytes = sum(
          (dst_local_outer_start[d] + multi_idx[d])
          * dst_outer_strides[d]
          * itemsize
          for d in range(num_outer_dims)
      )

      curr_src_offset = src_offset + outer_src_bytes
      curr_dst_offset = dst_offset + outer_dst_bytes
      chunks.append((
          curr_src_offset,
          curr_dst_offset,
          size_bytes,
          src_stride,
          dst_stride,
          count,
      ))
    return chunks

  if size_bytes == src_stride and size_bytes == dst_stride:
    return [(src_offset, dst_offset, size_bytes * count, 0, 0, 1)]

  return [(src_offset, dst_offset, size_bytes, src_stride, dst_stride, count)]


def compute_bandwidth_matched_seed(
    broadcast_host_ratio: Optional[float] = 1.0,
    num_src_hosts: int = 1,
    num_dst_hosts_per_unit: int = 1,
    num_dst_units: int = 1,
) -> int:
  """Computes bandwidth-matched seed sampler count N_seed.

  Args:
    broadcast_host_ratio: Ratio of trainer TX host bandwidth to sampler RX host
      bandwidth (K0 = B_train_TX / B_sample_RX). Values <= 0.0 or None default
      to 1.0.
    num_src_hosts: Number of trainer host endpoints (H_train).
    num_dst_hosts_per_unit: Number of sampler host endpoints per unit
      (H_sample).
    num_dst_units: Number of destination sampler units (N_samplers).

  Returns:
    N_seed: Number of seed samplers to populate directly in Round 0.
  """
  if num_src_hosts <= 0:
    raise ValueError(f"num_src_hosts must be positive, got {num_src_hosts}")
  if num_dst_hosts_per_unit <= 0:
    raise ValueError(
        f"num_dst_hosts_per_unit must be positive, got {num_dst_hosts_per_unit}"
    )
  if num_dst_units < 0:
    raise ValueError(f"num_dst_units must be non-negative, got {num_dst_units}")
  effective_ratio = (
      1.0
      if broadcast_host_ratio is None or broadcast_host_ratio <= 0.0
      else float(broadcast_host_ratio)
  )
  return max(
      1,
      int(
          math.floor(effective_ratio * (num_src_hosts / num_dst_hosts_per_unit))
      ),
  )


def pair_host_endpoints(
    unit: RaidenId,
    hosts: Sequence[Any],
    warned_units: Optional[set[RaidenId]] = None,
) -> dict[str, str]:
  """Pairs |unit|'s shard data endpoints with its single control endpoint.

  Args:
    unit: Work unit identifier whose host record is being paired.
    hosts: Sequence of HostEndpointsProto-like objects with |control_address|
      and |shards| attributes.
    warned_units: Optional per-instance set of units already warned about to
      suppress repeated warnings across calls. Uses a call-local set when None,
      so separate calls with None will log again.

  Returns:
    Mapping from shard data endpoint ("ip:port") to its single control-plane
    listener address ("ip:port"), or {} when no unambiguous 1-to-N pairing
    exists (logging a warning deduplicated against |warned_units| when provided,
    or once within the call when None).
  """
  warned = warned_units if warned_units is not None else set()

  def _warn_once(fmt: str, *args: Any) -> None:
    if unit not in warned:
      warned.add(unit)
      logging.warning(fmt, *args)

  host_list = list(hosts or ())
  if not host_list:
    _warn_once(
        "Unit %s has no HostEndpointsProto registration record; omitting"
        " peer_control_endpoints.",
        unit,
    )
    return {}
  if len(host_list) > 1:
    control_addresses = [getattr(h, "control_address", "") for h in host_list]
    shards = []
    for h in host_list:
      shards.extend(str(s) for s in getattr(h, "shards", ()) if s)
    _warn_once(
        "Unit %s has %d HostEndpointsProto records (control_addresses=%s,"
        " shards=%s), expected 1; omitting peer_control_endpoints.",
        unit,
        len(host_list),
        control_addresses,
        shards,
    )
    return {}

  host = host_list[0]
  control_address = str(getattr(host, "control_address", "") or "")
  endpoints = [ep.strip() for ep in control_address.split(",") if ep.strip()]
  shards = [str(s) for s in getattr(host, "shards", ()) if s]
  if not endpoints:
    _warn_once(
        "Unit %s has no registered control endpoint (control_address=%r,"
        " shards=%s); omitting peer_control_endpoints.",
        unit,
        control_address,
        shards,
    )
    return {}
  if len(endpoints) > 1:
    _warn_once(
        "Unit %s has %d control endpoints (%s) for shards=%s, cannot tell"
        " which serves which data endpoint; omitting peer_control_endpoints.",
        unit,
        len(endpoints),
        endpoints,
        shards,
    )
    return {}
  warned.discard(unit)
  return {shard: endpoints[0] for shard in shards}


class ReshardPlanner:
  """Computes resharding schedules across registered source and destination JobEntities."""

  @classmethod
  def compute_bandwidth_matched_seed(
      cls,
      broadcast_host_ratio: Optional[float] = 1.0,
      num_src_hosts: int = 1,
      num_dst_hosts_per_unit: int = 1,
      num_dst_units: int = 1,
  ) -> int:
    del cls
    return compute_bandwidth_matched_seed(
        broadcast_host_ratio=broadcast_host_ratio,
        num_src_hosts=num_src_hosts,
        num_dst_hosts_per_unit=num_dst_hosts_per_unit,
        num_dst_units=num_dst_units,
    )

  @classmethod
  def make_plan_cache_key(
      cls,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      group_size: int = 1,
      skip_tiling: Optional[dict[int, bool]] = None,
      dst_controller_address: Optional[str] = None,
      src_controller_address: Optional[str] = None,
      broadcast_host_ratio: Optional[float] = 1.0,
      broadcast_pipeline_stages: int = 4,
  ) -> tuple[Any, ...]:
    """Builds a hashable plan cache key from transfer arguments."""
    del cls
    if group_size <= 0:
      raise ValueError("group_size must be positive")
    if broadcast_pipeline_stages <= 0:
      raise ValueError(
          "broadcast_pipeline_stages must be >= 1, got"
          f" {broadcast_pipeline_stages}"
      )
    effective_ratio = (
        1.0
        if broadcast_host_ratio is None or broadcast_host_ratio <= 0.0
        else float(broadcast_host_ratio)
    )
    return (
        tuple(src_units),
        tuple(dst_units),
        group_size,
        tuple(sorted(skip_tiling.items())) if skip_tiling is not None else None,
        dst_controller_address,
        src_controller_address,
        effective_ratio,
        int(broadcast_pipeline_stages),
    )

  @classmethod
  def build_default_1d_plan(
      cls,
      selected_src: RaidenId,
      num_src: int,
      dst_shard_counts: list[tuple[RaidenId, int]],
  ) -> dict[RaidenId, list[list[tuple[RaidenId, int, list[Any]]]]]:
    """Builds the legacy synchronous 1D intersection plan."""
    del cls
    default_plan_dict = {}
    src_plan = [[] for _ in range(num_src)]
    for dst_unit, num_dst in dst_shard_counts:
      for i in range(num_src):
        src_start = i * num_dst
        src_end = (i + 1) * num_dst
        for j in range(num_dst):
          dst_start = j * num_src
          dst_end = (j + 1) * num_src
          intersect_start = max(src_start, dst_start)
          intersect_end = min(src_end, dst_end)
          if intersect_start < intersect_end:
            local_start = intersect_start - src_start
            local_end = intersect_end - src_start
            nd_slice = [(local_start, local_end)]
            src_plan[i].append((dst_unit, j, [nd_slice]))
    default_plan_dict[selected_src] = src_plan
    return default_plan_dict

  @classmethod
  def compute_transfer_schedule_from_metadata(
      cls,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      dst_metadata: list[Any],
      entities: Mapping[RaidenId, JobEntity],
      registered_variables: Mapping[RaidenId, list[Any]],
      registered_global_shapes: Mapping[RaidenId, list[int]],
      registered_mesh_shapes: Mapping[RaidenId, list[int]],
      registered_layouts: Mapping[RaidenId, list[int]],
      registered_itemsizes: Mapping[RaidenId, int],
      registered_shards: Mapping[RaidenId, list[str]],
      computed_phys_meshes: dict[RaidenId, list[int]],
      worker_endpoints: dict[RaidenId, str],
      broadcast_host_ratio: Optional[float] = 1.0,
      lock: Optional[threading.Lock] = None,
      group_size: int = 1,
      skip_tiling: Optional[dict[int, bool]] = None,
      shard_push_schedules: Optional[
          dict[RaidenId, dict[int, list[Any]]]
      ] = None,
      req_id: str = "warmup",
      uuid: Any = "",
      pipeline_target_stages: int = _PIPELINE_TARGET_STAGES,
      warned_units: Optional[set[RaidenId]] = None,
  ) -> _CachedTransferSchedule:
    """Computes transfer schedule math and returns a _CachedTransferSchedule."""
    if group_size <= 0:
      raise ValueError("group_size must be positive")
    if pipeline_target_stages <= 0:
      raise ValueError(
          f"pipeline_target_stages must be >= 1, got {pipeline_target_stages}"
      )

    if lock is None:
      lock = threading.Lock()

    computed_schedules = {}
    computed_slices = {}
    data_address_to_unit = {}
    num_vars = 0

    def resolve_shards_locked(u: RaidenId) -> list[str]:
      with lock:
        ent = entities.get(u)
        shards = (
            ent.shards if (ent and ent.shards) else registered_shards.get(u)
        )
        if not shards:
          raise ValueError(f"Work unit is not registered: {u}")
        return list(shards)

    def _count_hosts_for_shards(shards: Sequence[str]) -> int:
      hosts = {_extract_host_ip(s) for s in shards if _extract_host_ip(s)}
      return max(1, len(hosts))

    variable_plans = {}
    variable_to_plan_id = {}
    canonical_relay_plans = {}
    local_skip_tiling = dict(skip_tiling) if skip_tiling else {}

    if not src_units:
      raise ValueError("src_units must not be empty")
    total_src_hosts = sum(
        _count_hosts_for_shards(resolve_shards_locked(u)) for u in src_units
    )
    num_src_hosts = total_src_hosts

    if dst_metadata:
      if dst_metadata[0].shards:
        dst_shards_0 = list(dst_metadata[0].shards)
      else:
        dst_shards_0 = resolve_shards_locked(
            _raiden_id_from_proto(dst_metadata[0].unit)
        )
      num_dst_units = len(dst_metadata)
    elif dst_units:
      dst_shards_0 = resolve_shards_locked(dst_units[0])
      num_dst_units = len(dst_units)
    else:
      raise ValueError("dst_metadata or dst_units must not be empty")
    num_dst_hosts_per_unit = _count_hosts_for_shards(dst_shards_0)

    n_seed = compute_bandwidth_matched_seed(
        broadcast_host_ratio=broadcast_host_ratio,
        num_src_hosts=num_src_hosts,
        num_dst_hosts_per_unit=num_dst_hosts_per_unit,
        num_dst_units=num_dst_units,
    )

    if shard_push_schedules:
      logging.info("Using pre-computed shard_push_schedules")
      computed_schedules = shard_push_schedules
      for meta in dst_metadata:
        unit = _raiden_id_from_proto(meta.unit)
        for shard in meta.shards:
          data_address_to_unit[shard] = unit
    else:
      is_legacy_by_unit = {}
      slices_by_logical_spec = {}
      slices_by_phys_spec = {}
      nd_slices_by_phys_spec = {}
      computed_nd_slices = {}

      def _var_spec_sig(var: Any) -> tuple[Any, ...]:
        return (
            tuple(var.shape),
            tuple(var.mesh_shape),
            tuple(var.layout),
            tuple(getattr(var, "sharding_spec", None) or ()),
            tuple(getattr(var, "global_shard_indices", None) or ()),
            getattr(var, "item_size", 4) or 4,
        )

      def _get_or_compute_slices(
          shape: Sequence[int],
          mesh_shape: Sequence[int],
          layout: Sequence[int],
      ) -> tuple[list[Any], list[tuple[tuple[int, int], ...]], tuple[int, ...]]:
        logical_key = (tuple(shape), tuple(mesh_shape), tuple(layout))
        cached = slices_by_logical_spec.get(logical_key)
        if cached is not None:
          return cached
        phys_shape, phys_mesh = to_physical(
            logical_key[0], logical_key[1], logical_key[2]
        )
        phys_key = (tuple(phys_shape), tuple(phys_mesh))
        slices = slices_by_phys_spec.get(phys_key)
        if slices is None:
          slices = nd_slice_math.compute_nd_shard_slices(phys_shape, phys_mesh)
          slices_by_phys_spec[phys_key] = slices
          nd_slices_by_phys_spec[phys_key] = [
              tuple(_proto_to_nd_slice(p)) for p in slices
          ]
        res = (slices, nd_slices_by_phys_spec[phys_key], phys_mesh)
        slices_by_logical_spec[logical_key] = res
        return res

      # Source slices (always local to sender controller)
      src_vars_by_unit = {}
      src_sig_by_unit_and_name = {}
      src_unit_bundle_cache = {}
      for unit in src_units:
        with lock:
          variables = registered_variables.get(unit)
        if variables:
          src_vars = variables
          is_legacy_by_unit[unit] = False
        else:
          with lock:
            global_shape = registered_global_shapes.get(unit)
            mesh_shape = registered_mesh_shapes.get(unit)
            layout = registered_layouts.get(unit)
            itemsize = registered_itemsizes.get(unit) or 4
          if global_shape and mesh_shape and layout:
            src_vars = [
                _VariableMetadata(
                    name=unit.data_name,
                    shape=global_shape,
                    mesh_shape=mesh_shape,
                    layout=layout,
                    item_size=itemsize,
                    layer_idx=0,
                )
            ]
          else:
            src_vars = []
          is_legacy_by_unit[unit] = True

        src_vars_by_unit[unit] = src_vars
        num_vars = max(num_vars, len(src_vars))
        src_bundle = src_unit_bundle_cache.get(id(src_vars))
        if src_bundle is None:
          unit_slices = {}
          unit_nd_slices = {}
          unit_sigs = {}
          last_phys_mesh = None
          for var in src_vars:
            s_sig = _var_spec_sig(var)
            unit_sigs[var.name] = s_sig
            slices, nd_slices, last_phys_mesh = _get_or_compute_slices(
                s_sig[0], s_sig[1], s_sig[2]
            )
            unit_slices[var.name] = slices
            unit_nd_slices[var.name] = nd_slices
          src_bundle = (unit_slices, unit_nd_slices, unit_sigs, last_phys_mesh)
          src_unit_bundle_cache[id(src_vars)] = src_bundle
        else:
          unit_slices, unit_nd_slices, unit_sigs, last_phys_mesh = src_bundle
        computed_slices[unit] = unit_slices
        computed_nd_slices[unit] = unit_nd_slices
        src_sig_by_unit_and_name[unit] = unit_sigs
        if last_phys_mesh is not None:
          with lock:
            computed_phys_meshes[unit] = last_phys_mesh

      # Destination slices
      dst_vars_by_unit = {}
      dst_vars_by_unit_and_name = {}
      dst_sig_by_unit_and_name = {}
      data_address_to_host = {}
      ref_dst_vars: Any = None
      ref_dst_bundle: tuple[Any, Any, Any, Any, Any] = (
          {},
          {},
          {},
          {},
          None,
      )
      for meta in dst_metadata:
        unit = _raiden_id_from_proto(meta.unit)
        for shard in meta.shards:
          data_address_to_unit[shard] = unit
          if shard not in data_address_to_host:
            data_address_to_host[shard] = _extract_host_ip(shard)
        if meta.variables:
          dst_vars = meta.variables
          is_legacy_by_unit[unit] = False
        else:
          global_shape = list(meta.global_shape) if meta.global_shape else []
          mesh_shape = list(meta.mesh_shape) if meta.mesh_shape else []
          layout = list(meta.layout) if meta.layout else []
          itemsize = meta.itemsize if meta.itemsize else 4
          if global_shape and mesh_shape and layout:
            dst_vars = [
                _VariableMetadata(
                    name=unit.data_name,
                    shape=global_shape,
                    mesh_shape=mesh_shape,
                    layout=layout,
                    item_size=itemsize,
                    layer_idx=0,
                )
            ]
          else:
            dst_vars = []
          is_legacy_by_unit[unit] = True

        dst_vars_by_unit[unit] = dst_vars
        if (
            ref_dst_vars is not None
            and dst_vars
            and len(dst_vars) == len(ref_dst_vars)
            and not getattr(dst_vars[0], "global_shard_indices", None)
            and dst_vars == ref_dst_vars
        ):
          (
              unit_slices,
              unit_nd_slices,
              unit_sigs,
              unit_vars_by_name,
              last_phys_mesh,
          ) = ref_dst_bundle
        else:
          unit_slices = {}
          unit_nd_slices = {}
          unit_sigs = {}
          unit_vars_by_name = {}
          last_phys_mesh = None
          for var in dst_vars:
            v_name = var.name
            unit_vars_by_name[v_name] = var
            d_sig = _var_spec_sig(var)
            unit_sigs[v_name] = d_sig
            slices, nd_slices, last_phys_mesh = _get_or_compute_slices(
                d_sig[0], d_sig[1], d_sig[2]
            )
            unit_slices[v_name] = slices
            unit_nd_slices[v_name] = nd_slices
          if (
              ref_dst_bundle is None
              and dst_vars
              and not getattr(dst_vars[0], "global_shard_indices", None)
          ):
            ref_dst_vars = dst_vars
            ref_dst_bundle = (
                unit_slices,
                unit_nd_slices,
                unit_sigs,
                unit_vars_by_name,
                last_phys_mesh,
            )
        computed_slices[unit] = unit_slices
        computed_nd_slices[unit] = unit_nd_slices
        dst_sig_by_unit_and_name[unit] = unit_sigs
        dst_vars_by_unit_and_name[unit] = unit_vars_by_name
        if last_phys_mesh is not None:
          with lock:
            computed_phys_meshes[unit] = last_phys_mesh

      # Compute skip_tiling if not provided
      local_skip_tiling = skip_tiling
      if local_skip_tiling is None:
        local_skip_tiling = {}
        if src_units and dst_units:
          reference_src_unit = src_units[0]
          reference_src_vars = src_vars_by_unit.get(reference_src_unit, [])
          reference_dst_unit = dst_units[0]
          reference_dst_vars = dst_vars_by_unit.get(reference_dst_unit, [])
          ref_dst_var_by_layer = {v.layer_idx: v for v in reference_dst_vars}
          skip_tiling_by_geom = {}

          for src_var in reference_src_vars:
            layer_idx = src_var.layer_idx
            dst_var = ref_dst_var_by_layer.get(layer_idx)
            if dst_var:
              geom_key = (
                  tuple(src_var.shape),
                  tuple(src_var.mesh_shape),
                  tuple(src_var.layout),
                  tuple(getattr(src_var, "sharding_spec", None) or ()),
                  tuple(dst_var.shape),
                  tuple(dst_var.mesh_shape),
                  tuple(dst_var.layout),
                  tuple(getattr(dst_var, "sharding_spec", None) or ()),
              )
              aligned_decision = skip_tiling_by_geom.get(geom_key)
              if aligned_decision is None:
                s_nd_slices = computed_nd_slices.get(
                    reference_src_unit, {}
                ).get(src_var.name, [])
                d_nd_slices = computed_nd_slices.get(
                    reference_dst_unit, {}
                ).get(dst_var.name, [])
                all_aligned = bool(s_nd_slices) and bool(d_nd_slices)
                for s_sl in s_nd_slices:
                  for d_sl in d_nd_slices:
                    inter = intersect_nd_slices(s_sl, d_sl)
                    if inter:
                      if not is_nd_slice_tile_aligned(
                          s_sl, d_sl, inter, tile_shape=(8, 128)
                      ):
                        all_aligned = False
                        break
                  if not all_aligned:
                    break
                is_2d_or_more = len(src_var.shape) >= 2
                aligned_decision = is_2d_or_more and all_aligned
                skip_tiling_by_geom[geom_key] = aligned_decision
              local_skip_tiling[layer_idx] = aligned_decision

      # Pre-index source slice holders to deduplicate and load-balance
      # across replicated source shards, caching _get_global_indices by
      # (s_unit, s_sig).
      src_slice_holders = {}
      src_indices_cache = {}
      for s_unit in src_units:
        with lock:
          variables = registered_variables.get(s_unit)
        s_vars = variables if variables else []
        s_shards = resolve_shards_locked(s_unit)
        with lock:
          s_job_reps = {
              u.job_replica_id
              for u in registered_shards
              if u.job_name == s_unit.job_name
          }
        num_src_hosts = max(
            1,
            len(s_job_reps),
            len({
                u.job_replica_id
                for u in src_units
                if u.job_name == s_unit.job_name
            }),
        )
        for s_var in s_vars:
          s_nd_slices = computed_nd_slices.get(s_unit, {}).get(s_var.name)
          if not s_nd_slices:
            continue
          s_sig = src_sig_by_unit_and_name[s_unit][s_var.name]
          s_cache_key = (s_unit, s_sig)
          if s_cache_key in src_indices_cache:
            continue
          s_global_shard_indices = (
              list(s_var.global_shard_indices)
              if getattr(s_var, "global_shard_indices", None)
              else None
          )
          s_indices_list = _get_global_indices(
              s_unit,
              s_shards,
              list(s_var.mesh_shape),
              list(s_var.layout),
              num_src_hosts,
              global_shard_indices=s_global_shard_indices,
          )
          s_ranked_indices = []
          for l_s_idx, g_s_idx in s_indices_list:
            if g_s_idx < len(s_nd_slices):
              sl = s_nd_slices[g_s_idx]
              k = (s_sig, sl)
              holders = src_slice_holders.setdefault(k, [])
              rank = len(holders)
              holders.append((s_unit, l_s_idx))
              s_ranked_indices.append((l_s_idx, g_s_idx, rank))
          src_indices_cache[s_cache_key] = s_ranked_indices

      # Pre-index destination shard lists and job replicas for O(1) lookups.
      dst_shards_by_unit = {}
      dst_job_replicas_by_job = {}
      for meta in dst_metadata:
        m_unit = _raiden_id_from_proto(meta.unit)
        if m_unit not in dst_shards_by_unit:
          dst_shards_by_unit[m_unit] = (
              list(meta.shards) if meta.shards else ["127.0.0.1:8000"]
          )
        dst_job_replicas_by_job.setdefault(meta.unit.job_name, set()).add(
            meta.unit.job_replica_id
        )

      dst_group_sig_by_name = {}
      dst_units_index = {u: i for i, u in enumerate(dst_units)}
      dst_indices_cache = {}

      # Partition dst_units into shard-slice equivalence classes so multi-host
      # destination replicas (e.g. :0 holding shards 0..7 and :1 holding shards
      # 8..15) form disjoint broadcast trees per replica-host role.
      dst_eq_classes: list[list[RaidenId]] = []
      dst_unit_to_eq_class: dict[RaidenId, int] = {}
      eq_sig_to_class_idx: dict[tuple[Any, ...], int] = {}
      expected_num_key_parts: Optional[int] = None
      for d_u in dst_units:
        if d_u not in dst_shards_by_unit:
          raise ValueError(
              f"Destination unit {d_u} is missing from dst_metadata"
          )
        d_vars = dst_vars_by_unit.get(d_u, [])
        d_shards = dst_shards_by_unit[d_u]
        num_d_phys_hosts = max(
            1, len(dst_job_replicas_by_job.get(d_u.job_name, set()))
        )
        seen_d_sigs = set()
        unit_eq_key_parts = []
        for d_var in d_vars:
          d_sig = dst_sig_by_unit_and_name.get(d_u, {}).get(d_var.name)
          if d_sig is None:
            raise ValueError(
                f"Variable {d_var.name} on destination unit {d_u} is missing"
                " signature"
            )
          if d_sig in seen_d_sigs:
            continue
          seen_d_sigs.add(d_sig)
          d_nd_slices = computed_nd_slices.get(d_u, {}).get(d_var.name)
          if not d_nd_slices:
            raise ValueError(
                f"Variable {d_var.name} on destination unit {d_u} is missing"
                " computed nd_slices"
            )
          cache_key = (d_u, d_sig)
          d_indices = dst_indices_cache.get(cache_key)
          if d_indices is None:
            d_global_shard_indices = (
                list(d_var.global_shard_indices)
                if getattr(d_var, "global_shard_indices", None)
                else None
            )
            d_indices = _get_global_indices(
                d_u,
                d_shards,
                list(d_var.mesh_shape),
                list(d_var.layout),
                num_d_phys_hosts,
                global_shard_indices=d_global_shard_indices,
            )
            dst_indices_cache[cache_key] = d_indices
          slice_pairs = tuple(
              (l_idx, d_nd_slices[g_idx])
              for l_idx, g_idx in d_indices
              if g_idx < len(d_nd_slices)
          )
          unit_eq_key_parts.append((d_sig[:4], slice_pairs))
        if expected_num_key_parts is None:
          expected_num_key_parts = len(unit_eq_key_parts)
        elif len(unit_eq_key_parts) != expected_num_key_parts:
          raise ValueError(
              f"Destination unit {d_u} produced {len(unit_eq_key_parts)}"
              f" equivalence key parts, expected {expected_num_key_parts}"
          )
        eq_sig = tuple(unit_eq_key_parts)
        eq_idx = eq_sig_to_class_idx.get(eq_sig)
        if eq_idx is None:
          eq_idx = len(dst_eq_classes)
          eq_sig_to_class_idx[eq_sig] = eq_idx
          dst_eq_classes.append([])
        dst_eq_classes[eq_idx].append(d_u)
        dst_unit_to_eq_class[d_u] = eq_idx

      if not dst_eq_classes:
        raise ValueError(
            "Destination equivalence class partitioning produced no classes"
        )

      max_eq_class_len = max((len(c) for c in dst_eq_classes), default=0)
      if len(dst_eq_classes) > 1 and max_eq_class_len > 0:
        num_dst_hosts_per_replica = sum(
            _count_hosts_for_shards(dst_shards_by_unit[c[0]])
            for c in dst_eq_classes
            if c
        )
        n_seed = compute_bandwidth_matched_seed(
            broadcast_host_ratio=broadcast_host_ratio,
            num_src_hosts=total_src_hosts,
            num_dst_hosts_per_unit=max(1, num_dst_hosts_per_replica),
            num_dst_units=max_eq_class_len,
        )

      # 3. Generate plan (Intersection) with plan_id dictionary deduplication:
      # Each unique (src_sig, dst_group_sig, skip_tile_flag) is assigned a
      # unique integer `plan_id`. The calculated shard schedule for `plan_id`
      # is stored once in `variable_plans[src_unit][plan_id]`, and every
      # variable (`layer_idx`) stores `variable_to_plan_id[src_unit][layer_idx]`
      # referring to `plan_id`.
      dst_targets_cache = {}
      chunk_descriptor_cache = {}
      active_slice_chunks_cache = {}
      slice_candidate_cache = {}
      global_sig_to_plan_id = {}
      num_eq_classes = len(dst_eq_classes)
      variable_plans_by_eq_class: list[dict[RaidenId, dict[int, Any]]] = [
          {} for _ in range(num_eq_classes)
      ]
      variable_plans = variable_plans_by_eq_class[0]
      variable_to_plan_id = {}
      unit_ordered_vars_by_unit = {}
      canonical_relay_plans_by_eq_class: list[dict[int, Any]] = [
          {} for _ in range(num_eq_classes)
      ]
      canonical_relay_plans = canonical_relay_plans_by_eq_class[0]
      plan_classification_cache = {}

      for src_unit_idx, src_unit in enumerate(src_units):
        src_vars = src_vars_by_unit.get(src_unit, [])
        src_shards = resolve_shards_locked(src_unit)
        shard_host_ranks = controller_types.compute_shard_host_ranks(
            src_shards, len(src_units), src_unit_idx
        )
        unit_plans_by_eq_and_id: list[dict[int, Any]] = [
            {} for _ in range(num_eq_classes)
        ]

        with lock:
          src_job_replicas = {
              u.job_replica_id
              for u in registered_shards
              if u.job_name == src_unit.job_name
          }
        num_src_physical_hosts = max(
            1,
            len(src_job_replicas),
            len({
                u.job_replica_id
                for u in src_units
                if u.job_name == src_unit.job_name
            }),
        )

        unit_sig_map = src_sig_by_unit_and_name.get(src_unit, {})
        unit_nd_slices_map = computed_nd_slices.get(src_unit, {})

        # Classify variables into unique plan_ids (reusing classification when
        # src_unit shares the same variable signature sequence as a prior unit).
        vars_cache_key = (
            id(src_vars),
            tuple(
                (v.name, v.layer_idx, unit_sig_map.get(v.name))
                for v in src_vars
            ),
        )
        classified = plan_classification_cache.get(vars_cache_key)
        if classified is None:
          unit_var_to_plan_id = {}
          unit_ordered_vars = []
          unique_vars_to_compute = []
          unit_layers_by_pid = {}
          seen_pids_in_unit = set()
          for src_var in src_vars:
            var_name = src_var.name
            if not unit_nd_slices_map.get(var_name):
              continue
            layer_idx = src_var.layer_idx
            s_sig = unit_sig_map[var_name]
            dst_group_sig = dst_group_sig_by_name.get(var_name)
            if dst_group_sig is None:
              dst_group_sig = tuple(
                  (dst_unit, dst_sig_by_unit_and_name[dst_unit][var_name])
                  for dst_unit in dst_units
                  if var_name in dst_sig_by_unit_and_name.get(dst_unit, {})
              )
              dst_group_sig_by_name[var_name] = dst_group_sig
            skip_tile_flag = (
                bool(local_skip_tiling.get(layer_idx, False))
                if local_skip_tiling
                else False
            )
            var_plan_sig = (s_sig, dst_group_sig, skip_tile_flag)
            plan_id = global_sig_to_plan_id.get(var_plan_sig)
            if plan_id is None:
              plan_id = len(global_sig_to_plan_id)
              global_sig_to_plan_id[var_plan_sig] = plan_id
            # Store the variable's plan by referring to `plan_id`.
            unit_var_to_plan_id[layer_idx] = plan_id
            unit_ordered_vars.append((layer_idx, plan_id))
            unit_layers_by_pid.setdefault(plan_id, []).append(layer_idx)
            if plan_id not in seen_pids_in_unit:
              seen_pids_in_unit.add(plan_id)
              unique_vars_to_compute.append(
                  (plan_id, src_var, s_sig, dst_group_sig, skip_tile_flag)
              )
          unit_layers_tuple_by_pid = {
              pid: tuple(l_list) for pid, l_list in unit_layers_by_pid.items()
          }
          classified = (
              unit_var_to_plan_id,
              unit_ordered_vars,
              unique_vars_to_compute,
              unit_layers_tuple_by_pid,
          )
          plan_classification_cache[vars_cache_key] = classified
        else:
          (
              unit_var_to_plan_id,
              unit_ordered_vars,
              unique_vars_to_compute,
              unit_layers_tuple_by_pid,
          ) = classified

        # Calculate the plan ONLY ONCE per unique `plan_id` on this src_unit.
        # Any subsequent variable sharing the same `plan_id` already points to
        # `plan_id` via `unit_var_to_plan_id` and skips calculation completely.
        for (
            plan_id,
            src_var,
            s_sig,
            dst_group_sig,
            skip_tile_flag,
        ) in unique_vars_to_compute:
          itemsize = src_var.item_size
          var_name = src_var.name
          src_nd_slices = unit_nd_slices_map.get(var_name)
          if not src_nd_slices:
            continue

          s_cache_key = (src_unit, s_sig)
          src_ranked_indices = src_indices_cache.get(s_cache_key)
          if src_ranked_indices is None:
            src_global_shard_indices = (
                list(src_var.global_shard_indices)
                if getattr(src_var, "global_shard_indices", None)
                else None
            )
            raw_indices = _get_global_indices(
                src_unit,
                src_shards,
                list(src_var.mesh_shape),
                list(src_var.layout),
                num_src_physical_hosts,
                global_shard_indices=src_global_shard_indices,
            )
            src_ranked_indices = [
                (l_idx, g_idx, 0)
                for l_idx, g_idx in raw_indices
                if g_idx < len(src_nd_slices)
            ]
            src_indices_cache[s_cache_key] = src_ranked_indices

          # Resolve dst_indices and destination metadata for dst_group_sig once
          # across all variables and src_units sharing the same dst_group_sig.
          dst_target_bundle = dst_targets_cache.get(dst_group_sig)
          if dst_target_bundle is None:
            dst_targets = []
            dst_targets_by_slice = {}
            rep_targets_by_eq = {}
            for dst_unit in dst_units:
              dst_var = dst_vars_by_unit_and_name.get(dst_unit, {}).get(
                  var_name
              )
              if not dst_var:
                continue

              d_nd_slices = computed_nd_slices.get(dst_unit, {}).get(var_name)
              if not d_nd_slices:
                continue

              dst_shards = dst_shards_by_unit[dst_unit]

              d_sig = dst_sig_by_unit_and_name[dst_unit][var_name]
              cache_key = (dst_unit, d_sig)
              if cache_key in dst_indices_cache:
                dst_indices = dst_indices_cache[cache_key]
              else:
                with lock:
                  dst_job_replicas = dst_job_replicas_by_job.get(
                      dst_unit.job_name, set()
                  )
                num_dst_physical_hosts = max(1, len(dst_job_replicas))

                dst_global_shard_indices = (
                    list(dst_var.global_shard_indices)
                    if getattr(dst_var, "global_shard_indices", None)
                    else None
                )
                dst_indices = _get_global_indices(
                    dst_unit,
                    dst_shards,
                    list(dst_var.mesh_shape),
                    list(dst_var.layout),
                    num_dst_physical_hosts,
                    global_shard_indices=dst_global_shard_indices,
                )
                dst_indices_cache[cache_key] = dst_indices

              dst_unit_idx = dst_units_index.get(dst_unit, 0)
              is_dst_legacy = is_legacy_by_unit.get(dst_unit, True)
              num_dst_shards = max(1, len(dst_shards))

              dst_shard_items = []
              for local_dst_idx, global_dst_idx in dst_indices:
                if global_dst_idx >= len(d_nd_slices):
                  continue
                dst_slice = d_nd_slices[global_dst_idx]
                dst_peer = (
                    dst_shards[local_dst_idx]
                    if local_dst_idx < len(dst_shards)
                    else dst_shards[0]
                )
                dst_shard_items.append((local_dst_idx, dst_slice, dst_peer))
                dst_global_idx = dst_unit_idx * num_dst_shards + local_dst_idx
                dst_targets_by_slice.setdefault(
                    (dst_slice, is_dst_legacy), []
                ).append((
                    dst_unit,
                    dst_global_idx,
                    local_dst_idx,
                    dst_peer,
                    data_address_to_unit.get(dst_peer),
                    data_address_to_host.get(dst_peer),
                ))

              target_tuple = (
                  dst_unit,
                  dst_unit_idx,
                  is_dst_legacy,
                  num_dst_shards,
                  dst_shard_items,
              )
              dst_targets.append(target_tuple)
              eq_idx = dst_unit_to_eq_class.get(dst_unit, 0)
              if eq_idx not in rep_targets_by_eq:
                rep_targets_by_eq[eq_idx] = target_tuple
            dst_target_bundle = (
                dst_targets,
                dst_targets_by_slice,
                rep_targets_by_eq,
            )
            dst_targets_cache[dst_group_sig] = dst_target_bundle
          else:
            dst_targets, dst_targets_by_slice, rep_targets_by_eq = (
                dst_target_bundle
            )

          canonical_templates_by_eq = [{} for _ in range(num_eq_classes)]
          is_src_legacy = is_legacy_by_unit.get(src_unit, True)
          for (
              local_src_idx,
              global_src_idx,
              candidate_rank,
          ) in src_ranked_indices:
            src_host_rank = shard_host_ranks[local_src_idx]
            src_slice = src_nd_slices[global_src_idx]
            s_key = (s_sig, src_slice)
            candidates = src_slice_holders.get(s_key)
            num_candidates = len(candidates) if candidates else 1

            # Cache active intersecting (dst_slice, is_dst_legacy) chunks per
            # (plan_id, global_src_idx, is_src_legacy) so replicated/multi-host
            # source shards do not re-scan dst_targets_by_slice.
            active_cache_key = (plan_id, global_src_idx, is_src_legacy)
            active_slice_chunks = active_slice_chunks_cache.get(
                active_cache_key
            )
            if active_slice_chunks is None:
              active_slice_chunks = {}
              for dst_slice, is_dst_legacy in dst_targets_by_slice:
                is_legacy = is_src_legacy or is_dst_legacy
                chunk_key = (
                    src_slice,
                    dst_slice,
                    is_legacy,
                    skip_tile_flag,
                    itemsize,
                )
                converted_chunks = chunk_descriptor_cache.get(chunk_key)
                if converted_chunks is None:
                  intersection = intersect_nd_slices(src_slice, dst_slice)
                  if not intersection:
                    converted_chunks = ()
                  else:
                    is_tile_aware = skip_tile_flag and is_nd_slice_tile_aligned(
                        src_slice,
                        dst_slice,
                        intersection,
                        tile_shape=(8, 128),
                    )
                    if is_tile_aware:
                      chunks = generate_strided_copy_chunks_tile_aware(
                          src_slice,
                          dst_slice,
                          intersection,
                          itemsize,
                          tile_shape=(8, 128),
                      )
                    else:
                      chunks = generate_strided_copy_chunks(
                          src_slice, dst_slice, intersection, itemsize
                      )
                    if len(src_slice) > 1:
                      src_block_bytes = (
                          math.prod([e - s for s, e in src_slice[1:]])
                          * itemsize
                      )
                      src_multidim = True
                    else:
                      src_block_bytes = (
                          src_slice[0][1] - src_slice[0][0]
                      ) * itemsize
                      src_multidim = False

                    if len(dst_slice) > 1:
                      dst_block_bytes = (
                          math.prod([e - s for s, e in dst_slice[1:]])
                          * itemsize
                      )
                      dst_multidim = True
                    else:
                      dst_block_bytes = (
                          dst_slice[0][1] - dst_slice[0][0]
                      ) * itemsize
                      dst_multidim = False

                    built_chunks = []
                    for (
                        src_offset,
                        dst_offset,
                        size,
                        src_stride,
                        dst_stride,
                        count,
                    ) in chunks:
                      if src_multidim:
                        src_block_id = src_offset // src_block_bytes
                        src_block_offset = (
                            src_offset % src_block_bytes
                            if is_legacy
                            else src_offset
                        )
                      else:
                        src_block_id = 0
                        src_block_offset = src_offset

                      if dst_multidim:
                        dst_block_id = dst_offset // dst_block_bytes
                        dst_block_offset = (
                            dst_offset % dst_block_bytes
                            if is_legacy
                            else dst_offset
                        )
                      else:
                        dst_block_id = 0
                        dst_block_offset = dst_offset

                      built_chunks.append((
                          dst_block_offset,
                          src_block_offset,
                          size,
                          src_block_id,
                          dst_block_id,
                          src_stride,
                          dst_stride,
                          count,
                      ))
                    converted_chunks = tuple(built_chunks)
                  chunk_descriptor_cache[chunk_key] = converted_chunks
                if converted_chunks:
                  active_slice_chunks[(dst_slice, is_dst_legacy)] = (
                      converted_chunks
                  )
              active_slice_chunks_cache[active_cache_key] = active_slice_chunks

            if not active_slice_chunks:
              continue

            for eq_idx, rep_target in rep_targets_by_eq.items():
              (
                  _,
                  _,
                  is_canonical_legacy,
                  rep_num_shards,
                  canonical_shard_items,
              ) = rep_target
              eq_tmpl = canonical_templates_by_eq[eq_idx]
              for local_dst_idx, dst_slice, _ in canonical_shard_items:
                converted_chunks = active_slice_chunks.get(
                    (dst_slice, is_canonical_legacy)
                )
                if not converted_chunks:
                  continue
                if num_candidates > 1:
                  dst_global_idx = eq_idx * rep_num_shards + local_dst_idx
                  if dst_global_idx % num_candidates != candidate_rank:
                    continue
                for chunk_desc in converted_chunks:
                  eq_tmpl.setdefault(local_src_idx, []).append(
                      (local_dst_idx, *chunk_desc)
                  )

          for eq_idx in range(num_eq_classes):
            unit_plans_by_eq_and_id[eq_idx][plan_id] = (
                canonical_templates_by_eq[eq_idx]
            )

        for eq_idx in range(num_eq_classes):
          variable_plans_by_eq_class[eq_idx][src_unit] = (
              unit_plans_by_eq_and_id[eq_idx]
          )
        variable_to_plan_id[src_unit] = dict(unit_var_to_plan_id)
        unit_ordered_vars_by_unit[src_unit] = unit_ordered_vars

      # Map layer_idx -> unique list of (u, pid) across src_units
      layer_to_unit_pids: dict[int, list[tuple[RaidenId, int]]] = (
          collections.defaultdict(list)
      )
      seen_u_layer_pids: set[tuple[int, RaidenId, int]] = set()
      for u in src_units:
        for l_idx, pid in unit_ordered_vars_by_unit.get(u, []):
          if (l_idx, u, pid) not in seen_u_layer_pids:
            seen_u_layer_pids.add((l_idx, u, pid))
            layer_to_unit_pids[l_idx].append((u, pid))

      for eq_idx, eq_var_plans in enumerate(variable_plans_by_eq_class):
        eq_dst_units = dst_eq_classes[eq_idx]
        eq_is_legacy = (
            is_legacy_by_unit.get(eq_dst_units[0], True)
            if eq_dst_units
            else True
        )
        all_plan_ids = set()
        for u_plans in eq_var_plans.values():
          all_plan_ids.update(u_plans.keys())

        eq_relay_plans = canonical_relay_plans_by_eq_class[eq_idx]
        for l_idx, u_pid_list in layer_to_unit_pids.items():
          blocks_by_dst: dict[tuple[int, int], list[int]] = {}
          for u, pid in u_pid_list:
            u_plans = eq_var_plans.get(u, {})
            tuples_by_shard = u_plans.get(pid, {})
            for local_src_idx, tuples_9 in tuples_by_shard.items():
              for t9 in tuples_9:
                local_dst_idx = t9[0]
                dst_block_offset = t9[1]
                size = t9[3]
                # For non-legacy variable weight sync, each variable occupies a
                # single contiguous device buffer and dst_block_offset is an
                # absolute byte offset into that buffer (rather than a
                # block-relative offset). Normalizing dst_block_id to 0 allows
                # row chunks arriving from distinct FSDP source shards (which
                # carry different t9[5] = dst_offset // dst_block_bytes sub-row
                # indices) to coalesce into a single whole-buffer relay range.
                dst_block_id = 0 if not eq_is_legacy else t9[5]
                dst_stride = t9[7]
                count = t9[8]
                cur_end = dst_block_offset + (count - 1) * dst_stride + size
                key = (local_dst_idx, dst_block_id)
                if key not in blocks_by_dst:
                  blocks_by_dst[key] = [dst_block_offset, cur_end]
                else:
                  cur = blocks_by_dst[key]
                  if dst_block_offset < cur[0]:
                    cur[0] = dst_block_offset
                  if cur_end > cur[1]:
                    cur[1] = cur_end

          shard_blocks_map: dict[int, list[tuple[int, int, int]]] = {}
          for (local_dst_idx, dst_block_id), (min_off, max_end) in sorted(
              blocks_by_dst.items(),
              key=lambda item: (item[0][0], item[1][0], item[0][1]),
          ):
            shard_blocks = shard_blocks_map.setdefault(local_dst_idx, [])
            if not eq_is_legacy and shard_blocks:
              prev_off, prev_size, prev_bid = shard_blocks[-1]
              prev_end = prev_off + prev_size
              if min_off <= prev_end:
                new_end = max(prev_end, max_end)
                shard_blocks[-1] = (prev_off, new_end - prev_off, prev_bid)
                continue
            shard_blocks.append((min_off, max_end - min_off, dst_block_id))

          for _, pid in u_pid_list:
            eq_relay_plans[pid] = shard_blocks_map

        for pid in all_plan_ids:
          if pid not in eq_relay_plans:
            eq_relay_plans[pid] = {}
      canonical_relay_plans = canonical_relay_plans_by_eq_class[0]

    # Build rpc_addresses for local source workers
    rpc_addresses = dict(worker_endpoints)
    warned = warned_units if warned_units is not None else set()
    dst_unit_set = set(dst_units)
    dst_meta_by_unit: dict[RaidenId, Any] = {}
    # Merge destination rpc addresses from metadata
    for meta in dst_metadata:
      unit = _raiden_id_from_proto(meta.unit)
      if meta.control_plane_rpc_address:
        rpc_addresses[unit] = meta.control_plane_rpc_address
      if unit in dst_unit_set:
        dst_meta_by_unit[unit] = meta

    data_addresses = {unit: [] for unit in dst_units}
    for meta in dst_metadata:
      unit = _raiden_id_from_proto(meta.unit)
      if unit in data_addresses:
        data_addresses[unit] = list(meta.shards)
    for unit in src_units:
      with lock:
        if unit in registered_shards:
          data_addresses[unit] = list(registered_shards[unit])

    peer_control_endpoints: dict[str, str] = {}
    with lock:
      for unit in dst_units:
        meta = dst_meta_by_unit.get(unit)
        hosts = getattr(meta, "hosts", ()) if meta is not None else ()
        peer_control_endpoints.update(
            pair_host_endpoints(unit, hosts, warned_units=warned)
        )

    if not shard_push_schedules:
      direct_schedules = {}
      broadcast_groups = {}

      all_ordered_vars = []
      for u in src_units:
        all_ordered_vars.extend(unit_ordered_vars_by_unit.get(u, []))
      unique_layer_idxs = sorted(
          dict.fromkeys(l_idx for l_idx, _ in all_ordered_vars)
      )

      layer_group_map: dict[int, list[int]] = {}
      for l_idx in unique_layer_idxs:
        lg_idx = l_idx // group_size if group_size > 1 else l_idx
        layer_group_map.setdefault(lg_idx, []).append(l_idx)

      dst_unit_counts = {}
      dst_unit_layer_counts = {}
      dst_endpoint_counts = {}
      dst_endpoint_layer_counts = {}
      for eq_idx, eq_dst_units in enumerate(dst_eq_classes):
        eq_var_plans = variable_plans_by_eq_class[eq_idx]
        eq_relay_plans = canonical_relay_plans_by_eq_class[eq_idx]
        eq_total_seed_blocks = 0
        for lg_idx, l_indices in layer_group_map.items():
          l_set = set(l_indices)
          stage_ordered_vars_by_unit = {}
          stage_blocks = 0
          for u in src_units:
            u_vars = [
                (l_idx, pid)
                for (l_idx, pid) in unit_ordered_vars_by_unit.get(u, [])
                if l_idx in l_set
            ]
            stage_ordered_vars_by_unit[u] = u_vars
            u_plans = eq_var_plans.get(u, {})
            for l_idx, pid in u_vars:
              tuples_by_shard = u_plans.get(pid, {})
              layer_cnt = 0
              for local_src_idx, tuples_9 in tuples_by_shard.items():
                layer_cnt += len(tuples_9)
              stage_blocks += layer_cnt
              for d_u in eq_dst_units:
                dst_layer_map = dst_unit_layer_counts.setdefault(d_u, {})
                dst_layer_map[l_idx] = dst_layer_map.get(l_idx, 0) + layer_cnt

          eq_total_seed_blocks += stage_blocks
          stage_group = _StageBroadcastGroup(
              pool_group=eq_idx,
              layer_group_idx=lg_idx,
              src_units=list(src_units),
              dst_units=list(eq_dst_units),
              stage_ordered_vars_by_unit=stage_ordered_vars_by_unit,
              canonical_variable_plans=eq_var_plans,
              canonical_relay_plans=eq_relay_plans,
              data_addresses=dict(data_addresses),
          )
          broadcast_groups[(eq_idx, lg_idx)] = stage_group
        for d_u in eq_dst_units:
          dst_unit_counts[d_u] = eq_total_seed_blocks
          d_shards = data_addresses.get(d_u, [])
          d_host = (
              data_address_to_host.get(d_shards[0])
              or _extract_host_ip(d_shards[0])
              if d_shards
              else None
          )
          if d_host:
            dst_endpoint_counts[d_host] = (
                dst_endpoint_counts.get(d_host, 0) + eq_total_seed_blocks
            )
            dst_ep_layer = dst_endpoint_layer_counts.setdefault(d_host, {})
            for l_idx, cnt in dst_unit_layer_counts.get(d_u, {}).items():
              dst_ep_layer[l_idx] = dst_ep_layer.get(l_idx, 0) + cnt

      computed_expected_block_count = (
          max(dst_unit_counts.values()) if dst_unit_counts else 0
      )
      direct_dsts = []
      vars_info = f"{num_vars} variable(s), " if num_vars > 0 else ""
      logging.info(
          "Transfer %s (uuid=%s): generated tree broadcast schedule for %s ->"
          " %s (%s%d expected blocks, %d stage groups)",
          req_id,
          uuid,
          _format_units(src_units),
          _format_units(dst_units),
          vars_info,
          computed_expected_block_count,
          len(broadcast_groups),
      )
      if logging.vlog_is_on(1):
        logging.vlog(
            1,
            "RAIDEN_DIAG tree_schedule req_id=%s uuid=%s n_seed=%d"
            " broadcast_host_ratio=%s num_src_hosts=%d"
            " num_dst_hosts_per_unit=%d num_dst_units=%d",
            req_id,
            uuid,
            n_seed,
            broadcast_host_ratio,
            num_src_hosts,
            num_dst_hosts_per_unit,
            num_dst_units,
        )
    else:
      # Group flat entries into slices for broadcast
      groups = {}
      for src_unit, schedules in computed_schedules.items():
        for shard_idx, entries in schedules.items():
          for entry in entries:
            (
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
            ) = entry
            dst_unit = data_address_to_unit.get(dst_peer)
            if not dst_unit:
              continue
            key = (
                src_unit,
                shard_idx,
                src_block_id,
                src_block_offset,
                size,
                src_stride,
                count,
                layer_idx,
                pool_group,
            )
            val = (
                dst_unit,
                dst_peer,
                dst_shard_idx,
                dst_block_id,
                dst_block_offset,
                dst_stride,
            )
            groups.setdefault(key, []).append(val)

      direct_schedules = {}
      if groups:
        stage_group = controller_types.StageBroadcastGroup(
            pool_group=0,
            layer_group_idx=0,
            src_units=list(src_units),
            dst_units=list(dst_units),
            stage_ordered_vars_by_unit={},
            canonical_variable_plans={},
            canonical_relay_plans={},
            data_addresses=dict(data_addresses),
        )
        broadcast_groups = {(0, 0): stage_group}
      else:
        broadcast_groups = {}

      dst_unit_counts = {}
      dst_unit_layer_counts = {}
      dst_endpoint_counts = {}
      dst_endpoint_layer_counts = {}
      computed_expected_block_count = 0
      for key, targets in groups.items():
        layer_idx = key[7]
        for val in targets:
          dst_unit = val[0]
          dst_peer = val[1]
          dst_unit_counts[dst_unit] = dst_unit_counts.get(dst_unit, 0) + 1
          dst_unit_layer_counts.setdefault(dst_unit, {})
          dst_unit_layer_counts[dst_unit][layer_idx] = (
              dst_unit_layer_counts[dst_unit].get(layer_idx, 0) + 1
          )
          dst_host = _extract_host_ip(dst_peer)
          if dst_host:
            dst_endpoint_counts[dst_host] = (
                dst_endpoint_counts.get(dst_host, 0) + 1
            )
            dst_endpoint_layer_counts.setdefault(dst_host, {})
            dst_endpoint_layer_counts[dst_host][layer_idx] = (
                dst_endpoint_layer_counts[dst_host].get(layer_idx, 0) + 1
            )
      if dst_unit_counts:
        computed_expected_block_count = max(dst_unit_counts.values())
      vars_info = f"{num_vars} variable(s), " if num_vars > 0 else ""
      logging.info(
          "Transfer %s (uuid=%s): generated schedule for %s -> %s"
          " (%s%d expected blocks)",
          req_id,
          uuid,
          _format_units(src_units),
          _format_units(dst_units),
          vars_info,
          computed_expected_block_count,
      )

      direct_dsts = []

    return _CachedTransferSchedule(
        computed_schedules=computed_schedules,
        direct_schedules=direct_schedules,
        broadcast_groups=broadcast_groups,
        local_skip_tiling=(
            dict(local_skip_tiling) if local_skip_tiling else {}
        ),
        expected_block_count=computed_expected_block_count,
        dst_unit_layer_counts=dst_unit_layer_counts,
        data_address_to_unit=dict(data_address_to_unit),
        direct_dsts=list(direct_dsts),
        rpc_addresses=dict(rpc_addresses),
        data_addresses=data_addresses,
        peer_control_endpoints=peer_control_endpoints,
        dst_unit_counts=dst_unit_counts,
        dst_endpoint_counts=dst_endpoint_counts,
        dst_endpoint_layer_counts=dst_endpoint_layer_counts,
        is_weight_sync=bool(num_vars > 0 or local_skip_tiling),
        variable_plans=variable_plans,
        variable_to_plan_id=variable_to_plan_id,
        canonical_relay_plans=canonical_relay_plans,
        n_seed=n_seed,
    )
