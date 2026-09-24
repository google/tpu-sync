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

import math
import sys
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
_VariableMetadata = controller_types.VariableMetadata
_extract_host_ip = controller_types.extract_host_ip
_format_units = controller_types.format_units
_is_variable_spec_identical = controller_types.is_variable_spec_identical
_proto_to_nd_slice = controller_types.proto_to_nd_slice
_raiden_id_from_proto = controller_types.raiden_id_from_proto


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


def compute_host_subgrid(
    physical_mesh_shape: Sequence[int],
    devices_per_host: int,
) -> tuple[list[int], list[int]]:
  """Computes (host_subgrid, host_grid) for a physical TPU mesh shape and devices per host.

  Args:
    physical_mesh_shape: The shape of the physical mesh.
    devices_per_host: The number of devices assigned to each host process.

  Returns:
    A tuple of (host_subgrid, host_grid) where:
      - math.prod(host_subgrid) == devices_per_host (or total devices if fewer)
      - host_grid[i] = physical_mesh_shape[i] // host_subgrid[i]
  """
  num_dims = len(physical_mesh_shape)
  if num_dims == 0 or devices_per_host <= 0:
    return [], []

  total_devices = math.prod(physical_mesh_shape)
  if total_devices <= devices_per_host:
    subgrid = list(physical_mesh_shape)
    grid = [1] * num_dims
    return subgrid, grid

  if num_dims == 1:
    subgrid = [devices_per_host]
    grid = [physical_mesh_shape[0] // devices_per_host]
    return subgrid, grid

  candidates: list[list[int]] = []

  def find_factors(dim_idx: int, rem_k: int, current: list[int]):
    if dim_idx == num_dims - 1:
      if physical_mesh_shape[dim_idx] % rem_k == 0:
        candidates.append(current + [rem_k])
      return
    p_dim = physical_mesh_shape[dim_idx]
    divisors = [
        d for d in range(1, int(math.isqrt(rem_k)) + 1) if rem_k % d == 0
    ]
    all_divisors = sorted(set(divisors + [rem_k // d for d in divisors]))
    for d in all_divisors:
      if p_dim % d == 0:
        find_factors(dim_idx + 1, rem_k // d, current + [d])

  find_factors(0, devices_per_host, [])
  if not candidates:
    subgrid = [1] * num_dims
    subgrid[-1] = min(devices_per_host, physical_mesh_shape[-1])
    grid = [
        p // s if s > 0 else 1 for p, s in zip(physical_mesh_shape, subgrid)
    ]
    return subgrid, grid

  def score(s: list[int]):
    h = [p // si for p, si in zip(physical_mesh_shape, s)]
    h_monotonic_violations = sum(
        1 for i in range(len(h) - 1) if h[i] < h[i + 1]
    )
    s_monotonic_violations = sum(
        1 for i in range(len(s) - 1) if s[i] > s[i + 1]
    )
    return (
        h_monotonic_violations,
        s_monotonic_violations,
        max(s),
        max(h),
        tuple(-x for x in reversed(s)),
    )

  candidates.sort(key=score)
  chosen_subgrid = candidates[0]
  chosen_grid = [
      p // s if s > 0 else 1
      for p, s in zip(physical_mesh_shape, chosen_subgrid)
  ]
  return chosen_subgrid, chosen_grid


def _get_global_indices(
    unit: RaidenId,
    shards: list[str],
    logical_mesh_shape: list[int],
    layout: list[int],
    num_physical_hosts: int,
    sharding_spec: Optional[list[str]] = None,
    mesh_axes: Optional[list[str]] = None,
    physical_mesh_shape: Optional[list[int]] = None,
    host_subgrid: Optional[list[int]] = None,
    global_shard_indices: Optional[list[int]] = None,
) -> list[tuple[int, int]]:
  """Maps local shard indices to global slice indices, handling replication.

  When global_shard_indices is provided, this function
  bypasses all mesh geometry, subgrid calculation, and coordinate mapping,
  directly returning the 1:1 mapping between local shards and their declared
  global slice indices.

  Note: When global_shard_indices is used:
    - host_subgrid is no longer necessary.
    - mesh_axes is no longer necessary.
    - sharding_spec is no longer necessary.
    - physical_mesh_shape is no longer necessary.
    - num_physical_hosts, layout, and logical_mesh_shape are not used for index
      calculation.

  Args:
    unit: The RaidenId of the work unit.
    shards: List of data-plane shard endpoint strings.
    logical_mesh_shape: Logical device mesh dimensions.
    layout: Shard dimension permutation mapping.
    num_physical_hosts: Total physical host count.
    sharding_spec: Optional axis names for sharding dimensions.
    mesh_axes: Optional named mesh axis labels.
    physical_mesh_shape: Physical topology shape.
    host_subgrid: Host device coordinate subgrid bounding box.
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

  use_spec_mapping = bool(sharding_spec and mesh_axes and physical_mesh_shape)

  if use_spec_mapping:
    devices_per_host = num_shards
    if (
        host_subgrid is not None
        and len(host_subgrid) == len(physical_mesh_shape)
        and math.prod(host_subgrid) == devices_per_host
        and all(p % s == 0 for p, s in zip(physical_mesh_shape, host_subgrid))
    ):
      subgrid = list(host_subgrid)
      grid = [p // s for p, s in zip(physical_mesh_shape, subgrid)]
    else:
      logging.warning(
          "host_subgrid not provided or invalid in _get_global_indices; falling"
          " back to compute_host_subgrid. This fallback is deprecated."
      )
      rc_mod = sys.modules.get(
          "tpu_sync.rpc.raiden_controller"
      )
      subgrid_fn = getattr(rc_mod, "compute_host_subgrid", compute_host_subgrid)
      subgrid, grid = subgrid_fn(physical_mesh_shape, devices_per_host)
    host_subgrid, host_grid = subgrid, grid

    host_coords = []
    temp_h = replica_id
    for size in reversed(host_grid):
      host_coords.append(temp_h % size if size > 0 else 0)
      temp_h //= size if size > 0 else 1
    host_coords.reverse()

    indices = []
    for j in range(num_shards):
      local_coords = []
      temp_l = j
      for size in reversed(host_subgrid):
        local_coords.append(temp_l % size if size > 0 else 0)
        temp_l //= size if size > 0 else 1
      local_coords.reverse()

      phys_coords = [
          h * s + l for h, s, l in zip(host_coords, host_subgrid, local_coords)
      ]

      # Map physical coordinates to tensor dimensions using sharding_spec
      tensor_coords = []
      for axis_name in sharding_spec:
        if not axis_name:
          tensor_coords.append(0)
        elif "," in axis_name:
          sub_axes = [a.strip() for a in axis_name.split(",") if a.strip()]
          coord = 0
          for sub_a in sub_axes:
            try:
              phys_axis_idx = mesh_axes.index(sub_a)
              sub_size = physical_mesh_shape[phys_axis_idx]
              coord = coord * sub_size + phys_coords[phys_axis_idx]
            except ValueError:
              logging.warning(
                  "Sub-spec axis %s not found in mesh axes %s", sub_a, mesh_axes
              )
          tensor_coords.append(coord)
        else:
          try:
            phys_axis_idx = mesh_axes.index(axis_name)
            tensor_coords.append(phys_coords[phys_axis_idx])
          except ValueError:
            logging.warning(
                "Spec axis %s not found in mesh axes %s", axis_name, mesh_axes
            )
            tensor_coords.append(0)

      # Compute flat index in logical_mesh_shape (row-major)
      if len(tensor_coords) < len(logical_mesh_shape):
        tensor_coords = tensor_coords + [0] * (
            len(logical_mesh_shape) - len(tensor_coords)
        )
      global_idx = 0
      stride = 1
      for val, size in zip(
          reversed(tensor_coords), reversed(logical_mesh_shape)
      ):
        global_idx += val * stride
        stride *= size
      indices.append((j, global_idx))

    return indices

  else:
    if host_axis_logical is None:
      logging.warning(
          "host_axis_logical is None and sharding_spec, mesh_axes, or"
          " physical_mesh_shape is missing for %s. Falling back to legacy"
          " mapping.",
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


class ReshardPlanner:
  """Computes resharding schedules across registered source and destination JobEntities."""

  @classmethod
  def make_plan_cache_key(
      cls,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      group_size: int = 1,
      skip_tiling: Optional[dict[int, bool]] = None,
      dst_controller_address: Optional[str] = None,
      src_controller_address: Optional[str] = None,
  ) -> tuple[Any, ...]:
    """Builds a hashable plan cache key from transfer arguments."""
    del cls
    if group_size <= 0:
      raise ValueError("group_size must be positive")
    return (
        tuple(src_units),
        tuple(dst_units),
        group_size,
        tuple(sorted(skip_tiling.items())) if skip_tiling is not None else None,
        dst_controller_address,
        src_controller_address,
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
      registered_mesh_axes: Mapping[RaidenId, list[str]],
      registered_host_subgrids: Mapping[RaidenId, list[int]],
      registered_layouts: Mapping[RaidenId, list[int]],
      registered_itemsizes: Mapping[RaidenId, int],
      registered_shards: Mapping[RaidenId, list[str]],
      computed_phys_meshes: dict[RaidenId, list[int]],
      worker_endpoints: dict[RaidenId, str],
      broadcast_k: int,
      lock: threading.Lock,
      group_size: int = 1,
      skip_tiling: Optional[dict[int, bool]] = None,
      shard_push_schedules: Optional[
          dict[RaidenId, dict[int, list[Any]]]
      ] = None,
      req_id: str = "warmup",
      uuid: Any = "",
  ) -> _CachedTransferSchedule:
    """Computes transfer schedule math and returns a _CachedTransferSchedule."""
    if group_size <= 0:
      raise ValueError("group_size must be positive")

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

    if shard_push_schedules:
      logging.info("Using pre-computed shard_push_schedules")
      computed_schedules = shard_push_schedules
      for meta in dst_metadata:
        unit = _raiden_id_from_proto(meta.unit)
        for shard in meta.shards:
          data_address_to_unit[shard] = unit
    else:
      is_legacy_by_unit = {}
      # Source slices (always local to sender controller)
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

        num_vars = max(num_vars, len(src_vars))
        computed_slices[unit] = {}
        for var in src_vars:
          phys_shape, phys_mesh = to_physical(
              var.shape, var.mesh_shape, var.layout
          )
          with lock:
            computed_phys_meshes[unit] = phys_mesh
          slices = nd_slice_math.compute_nd_shard_slices(phys_shape, phys_mesh)
          computed_slices[unit][var.name] = slices
          logging.debug(
              "Computed source slices for %s var %s: %s",
              unit,
              var.name,
              nd_slice_math.format_nd_slices(slices),
          )

      # Destination slices
      dst_vars_by_unit = {}
      for meta in dst_metadata:
        unit = _raiden_id_from_proto(meta.unit)
        for shard in meta.shards:
          data_address_to_unit[shard] = unit
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
        computed_slices[unit] = {}
        for var in dst_vars:
          phys_shape, phys_mesh = to_physical(
              list(var.shape),
              list(var.mesh_shape),
              list(var.layout),
          )
          with lock:
            computed_phys_meshes[unit] = phys_mesh
          slices = nd_slice_math.compute_nd_shard_slices(phys_shape, phys_mesh)
          computed_slices[unit][var.name] = slices
          logging.debug(
              "Computed destination slices for %s var %s: %s",
              unit,
              var.name,
              nd_slice_math.format_nd_slices(slices),
          )

      # Compute skip_tiling if not provided
      local_skip_tiling = skip_tiling
      if local_skip_tiling is None:
        local_skip_tiling = {}
        if src_units and dst_units:
          reference_src_unit = src_units[0]
          with lock:
            reference_src_vars = registered_variables.get(reference_src_unit)
          if not reference_src_vars:
            with lock:
              global_shape = registered_global_shapes.get(reference_src_unit)
              mesh_shape = registered_mesh_shapes.get(reference_src_unit)
              layout = registered_layouts.get(reference_src_unit)
              itemsize = registered_itemsizes.get(reference_src_unit) or 4
            if global_shape and mesh_shape and layout:
              reference_src_vars = [
                  _VariableMetadata(
                      name=reference_src_unit.data_name,
                      shape=global_shape,
                      mesh_shape=mesh_shape,
                      layout=layout,
                      item_size=itemsize,
                      layer_idx=0,
                  )
              ]
            else:
              reference_src_vars = []

          reference_dst_unit = dst_units[0]
          reference_dst_vars = dst_vars_by_unit.get(reference_dst_unit, [])

          for src_var in reference_src_vars:
            layer_idx = src_var.layer_idx
            dst_var = next(
                (v for v in reference_dst_vars if v.layer_idx == layer_idx),
                None,
            )
            if dst_var:
              is_identical = _is_variable_spec_identical(src_var, dst_var)
              s_slices = computed_slices.get(reference_src_unit, {}).get(
                  src_var.name, []
              )
              d_slices = computed_slices.get(reference_dst_unit, {}).get(
                  dst_var.name, []
              )
              all_aligned = bool(s_slices) and bool(d_slices)
              for s_proto in s_slices:
                s_sl = _proto_to_nd_slice(s_proto)
                for d_proto in d_slices:
                  d_sl = _proto_to_nd_slice(d_proto)
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
              is_2d_identical = is_identical and is_2d_or_more
              local_skip_tiling[layer_idx] = is_2d_or_more and (
                  is_2d_identical or all_aligned
              )

      # Pre-index source slice holders to deduplicate and load-balance
      # across replicated source shards.
      src_slice_holders = {}
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
          s_phys_mesh = registered_mesh_shapes.get(s_unit)
          s_mesh_axes = registered_mesh_axes.get(s_unit)
          s_host_subgrid = registered_host_subgrids.get(s_unit)
        num_src_hosts = max(1, len(s_job_reps))
        for s_var in s_vars:
          s_slices_list = computed_slices.get(s_unit, {}).get(s_var.name)
          if not s_slices_list:
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
              sharding_spec=list(s_var.sharding_spec),
              mesh_axes=s_mesh_axes,
              physical_mesh_shape=s_phys_mesh,
              host_subgrid=s_host_subgrid,
              global_shard_indices=s_global_shard_indices,
          )
          for l_s_idx, g_s_idx in s_indices_list:
            if g_s_idx < len(s_slices_list):
              s_proto = s_slices_list[g_s_idx]
              sl = tuple(_proto_to_nd_slice(s_proto))
              k = (s_var.name, sl)
              src_slice_holders.setdefault(k, []).append((s_unit, l_s_idx))

      # Pre-index destination metadata and variables for O(1) lookups.
      dst_meta_info = {}
      dst_job_replicas_by_job = {}
      for meta in dst_metadata:
        m_unit = _raiden_id_from_proto(meta.unit)
        if m_unit not in dst_meta_info:
          dst_meta_info[m_unit] = (
              list(meta.shards) if meta.shards else ["127.0.0.1:8000"],
              list(meta.mesh_shape) if meta.mesh_shape else None,
              list(meta.mesh_axes) if meta.mesh_axes else None,
              list(meta.host_subgrid) if meta.host_subgrid else None,
          )
        dst_job_replicas_by_job.setdefault(meta.unit.job_name, set()).add(
            meta.unit.job_replica_id
        )

      dst_vars_by_unit_and_name = {
          u: {v.name: v for v in vars} for u, vars in dst_vars_by_unit.items()
      }
      dst_units_index = {u: i for i, u in enumerate(dst_units)}

      # 3. Generate plan (Intersection)
      dst_indices_cache = {}
      dst_targets_cache = {}
      for src_unit in src_units:
        with lock:
          variables = registered_variables.get(src_unit)
        if variables:
          src_vars = variables
        else:
          with lock:
            global_shape = registered_global_shapes.get(src_unit)
            mesh_shape = registered_mesh_shapes.get(src_unit)
            layout = registered_layouts.get(src_unit)
            itemsize = registered_itemsizes.get(src_unit) or 4
          if global_shape and mesh_shape and layout:
            src_vars = [
                _VariableMetadata(
                    name=src_unit.data_name,
                    shape=global_shape,
                    mesh_shape=mesh_shape,
                    layout=layout,
                    item_size=itemsize,
                    layer_idx=0,
                )
            ]
          else:
            src_vars = []

        src_shards = resolve_shards_locked(src_unit)
        unit_schedules = {}

        for src_var in src_vars:
          itemsize = src_var.item_size
          layer_idx = src_var.layer_idx
          var_name = src_var.name

          src_slices = computed_slices.get(src_unit, {}).get(var_name)
          if not src_slices:
            continue

          with lock:
            src_job_replicas = {
                u.job_replica_id
                for u in registered_shards
                if u.job_name == src_unit.job_name
            }
            src_phys_mesh_shape = registered_mesh_shapes.get(src_unit)
            src_mesh_axes = registered_mesh_axes.get(src_unit)
            src_host_subgrid = registered_host_subgrids.get(src_unit)
          num_src_physical_hosts = max(1, len(src_job_replicas))
          src_logical_mesh = list(src_var.mesh_shape)
          src_layout = list(src_var.layout)

          src_global_shard_indices = (
              list(src_var.global_shard_indices)
              if getattr(src_var, "global_shard_indices", None)
              else None
          )
          src_indices = _get_global_indices(
              src_unit,
              src_shards,
              src_logical_mesh,
              src_layout,
              num_src_physical_hosts,
              sharding_spec=list(src_var.sharding_spec),
              mesh_axes=src_mesh_axes,
              physical_mesh_shape=src_phys_mesh_shape,
              host_subgrid=src_host_subgrid,
              global_shard_indices=src_global_shard_indices,
          )

          # Resolve dst_indices and destination metadata for var_name once
          # per var across dst_units and cache per var_name across src_units.
          dst_targets = dst_targets_cache.get(var_name)
          if dst_targets is None:
            dst_targets = []
            for dst_unit in dst_units:
              dst_var = dst_vars_by_unit_and_name.get(dst_unit, {}).get(
                  var_name
              )
              if not dst_var:
                continue

              d_slices = computed_slices.get(dst_unit, {}).get(var_name)
              if not d_slices:
                continue

              meta_tuple = dst_meta_info.get(dst_unit)
              if meta_tuple:
                (
                    dst_shards,
                    dst_phys_mesh_shape,
                    dst_mesh_axes,
                    dst_host_subgrid,
                ) = meta_tuple
              else:
                dst_shards = ["127.0.0.1:8000"]
                dst_phys_mesh_shape = None
                dst_mesh_axes = None
                dst_host_subgrid = None

              cache_key = (dst_unit, var_name)
              if cache_key in dst_indices_cache:
                dst_indices = dst_indices_cache[cache_key]
              else:
                with lock:
                  dst_job_replicas = dst_job_replicas_by_job.get(
                      dst_unit.job_name, set()
                  )
                num_dst_physical_hosts = max(1, len(dst_job_replicas))

                dst_logical_mesh = list(dst_var.mesh_shape)
                dst_layout = list(dst_var.layout)

                dst_global_shard_indices = (
                    list(dst_var.global_shard_indices)
                    if getattr(dst_var, "global_shard_indices", None)
                    else None
                )
                dst_indices = _get_global_indices(
                    dst_unit,
                    dst_shards,
                    dst_logical_mesh,
                    dst_layout,
                    num_dst_physical_hosts,
                    sharding_spec=list(dst_var.sharding_spec),
                    mesh_axes=dst_mesh_axes,
                    physical_mesh_shape=dst_phys_mesh_shape,
                    host_subgrid=dst_host_subgrid,
                    global_shard_indices=dst_global_shard_indices,
                )
                dst_indices_cache[cache_key] = dst_indices

              dst_unit_idx = dst_units_index.get(dst_unit, 0)
              is_dst_legacy = is_legacy_by_unit.get(dst_unit, True)
              num_dst_shards = max(1, len(dst_shards))

              # Pre-convert destination slice protos to coordinate intervals
              # and pre-resolve destination peer endpoints.
              dst_shard_items = []
              for local_dst_idx, global_dst_idx in dst_indices:
                if global_dst_idx >= len(d_slices):
                  continue
                dst_slice_proto = d_slices[global_dst_idx]
                dst_slice = _proto_to_nd_slice(dst_slice_proto)
                dst_peer = (
                    dst_shards[local_dst_idx]
                    if local_dst_idx < len(dst_shards)
                    else dst_shards[0]
                )
                dst_shard_items.append((local_dst_idx, dst_slice, dst_peer))

              dst_targets.append((
                  dst_unit,
                  dst_unit_idx,
                  is_dst_legacy,
                  num_dst_shards,
                  dst_shard_items,
              ))
            dst_targets_cache[var_name] = dst_targets

          for local_src_idx, global_src_idx in src_indices:
            if global_src_idx >= len(src_slices):
              continue

            src_slice_proto = src_slices[global_src_idx]
            src_slice = _proto_to_nd_slice(src_slice_proto)
            shard_entries = unit_schedules.setdefault(local_src_idx, [])

            for (
                dst_unit,
                dst_unit_idx,
                is_dst_legacy,
                num_dst_shards,
                dst_shard_items,
            ) in dst_targets:
              is_legacy = is_legacy_by_unit.get(src_unit, True) or is_dst_legacy
              for local_dst_idx, dst_slice, dst_peer in dst_shard_items:
                intersection = intersect_nd_slices(src_slice, dst_slice)
                if intersection:
                  s_key = (var_name, tuple(src_slice))
                  candidates = src_slice_holders.get(
                      s_key, [(src_unit, local_src_idx)]
                  )
                  if len(candidates) > 1:
                    dst_global_idx = (
                        dst_unit_idx * num_dst_shards + local_dst_idx
                    )
                    chosen_src = candidates[dst_global_idx % len(candidates)]
                    if (src_unit, local_src_idx) != chosen_src:
                      continue

                  is_tile_aware = (
                      local_skip_tiling.get(layer_idx, False)
                      if local_skip_tiling
                      else False
                  ) and is_nd_slice_tile_aligned(
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
                  for (
                      src_offset,
                      dst_offset,
                      size,
                      src_stride,
                      dst_stride,
                      count,
                  ) in chunks:

                    if len(src_slice) > 1:
                      src_block_bytes = (
                          math.prod([e - s for s, e in src_slice[1:]])
                          * itemsize
                      )
                      src_block_id = src_offset // src_block_bytes
                      src_block_offset = (
                          src_offset % src_block_bytes
                          if is_legacy
                          else src_offset
                      )
                    else:
                      src_block_bytes = (
                          src_slice[0][1] - src_slice[0][0]
                      ) * itemsize
                      src_block_id = 0
                      src_block_offset = src_offset

                    if len(dst_slice) > 1:
                      dst_block_bytes = (
                          math.prod([e - s for s, e in dst_slice[1:]])
                          * itemsize
                      )
                      dst_block_id = dst_offset // dst_block_bytes
                      dst_block_offset = (
                          dst_offset % dst_block_bytes
                          if is_legacy
                          else dst_offset
                      )
                    else:
                      dst_block_bytes = (
                          dst_slice[0][1] - dst_slice[0][0]
                      ) * itemsize
                      dst_block_id = 0
                      dst_block_offset = dst_offset

                    shard_entries.append((
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
                        0,
                    ))

        if unit_schedules:
          computed_schedules[src_unit] = unit_schedules

    # Build rpc_addresses for local source workers
    rpc_addresses = dict(worker_endpoints)
    # Merge destination rpc addresses from metadata
    for meta in dst_metadata:
      unit = _raiden_id_from_proto(meta.unit)
      if meta.control_plane_rpc_address:
        rpc_addresses[unit] = meta.control_plane_rpc_address

    data_addresses = {unit: [] for unit in dst_units}
    for meta in dst_metadata:
      unit = _raiden_id_from_proto(meta.unit)
      if unit in data_addresses:
        data_addresses[unit] = list(meta.shards)
    for unit in src_units:
      with lock:
        if unit in registered_shards:
          data_addresses[unit] = list(registered_shards[unit])

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

    direct_schedules, broadcast_groups = (
        BroadcastEngine.partition_direct_and_broadcast_groups(
            groups, broadcast_k, group_size
        )
    )

    dst_unit_counts = {}
    dst_unit_layer_counts = {}
    dst_endpoint_counts = {}
    dst_endpoint_layer_counts = {}
    computed_expected_block_count = 0
    if direct_schedules:
      for src_unit, schedules in direct_schedules.items():
        for shard_idx, entries in schedules.items():
          for entry in entries:
            dst_peer = entry[0]
            dst_unit = data_address_to_unit.get(dst_peer)
            if dst_unit:
              layer_idx = entry[10] if len(entry) > 10 else 0
              tasks_count = 1
              dst_unit_counts[dst_unit] = (
                  dst_unit_counts.get(dst_unit, 0) + tasks_count
              )
              dst_unit_layer_counts.setdefault(dst_unit, {})
              dst_unit_layer_counts[dst_unit][layer_idx] = (
                  dst_unit_layer_counts[dst_unit].get(layer_idx, 0)
                  + tasks_count
              )
              dst_host = _extract_host_ip(dst_peer)
              if dst_host:
                dst_endpoint_counts[dst_host] = (
                    dst_endpoint_counts.get(dst_host, 0) + tasks_count
                )
                dst_endpoint_layer_counts.setdefault(dst_host, {})
                dst_endpoint_layer_counts[dst_host][layer_idx] = (
                    dst_endpoint_layer_counts[dst_host].get(layer_idx, 0)
                    + tasks_count
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
    for scheds in direct_schedules.values():
      for entries in scheds.values():
        for entry in entries:
          dst_peer = entry[0]
          d_node = data_address_to_unit.get(dst_peer)
          if d_node and d_node not in direct_dsts:
            direct_dsts.append(d_node)

    return _CachedTransferSchedule(
        computed_schedules=computed_schedules,
        direct_schedules=direct_schedules,
        broadcast_groups=broadcast_groups,
        local_skip_tiling=dict(local_skip_tiling) if local_skip_tiling else {},
        expected_block_count=computed_expected_block_count,
        dst_unit_layer_counts=dst_unit_layer_counts,
        data_address_to_unit=dict(data_address_to_unit),
        direct_dsts=list(direct_dsts),
        rpc_addresses=dict(rpc_addresses),
        data_addresses=data_addresses,
        dst_unit_counts=dst_unit_counts,
        dst_endpoint_counts=dst_endpoint_counts,
        dst_endpoint_layer_counts=dst_endpoint_layer_counts,
        is_weight_sync=bool(num_vars > 0 or local_skip_tiling),
    )
