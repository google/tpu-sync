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

import concurrent.futures
import dataclasses
import json
import math
import os
import sys
import threading
from typing import Any, Mapping, Optional, Sequence

from absl import logging

from tpu_sync.api.common import RaidenId
from tpu_sync.kv_cache import nd_slice_math
from tpu_sync.rpc import raiden_service_pb2
from tpu_sync.weight_sync.manager import broadcast_engine
from tpu_sync.weight_sync.manager import controller_types
from tpu_sync.weight_sync.manager import job_entity

BroadcastEngine = broadcast_engine.BroadcastEngine
JobEntity = job_entity.JobEntity
_CachedTransferSchedule = controller_types.CachedTransferSchedule
_PlanReferencedShardSchedule = controller_types.PlanReferencedShardSchedule
_VariableMetadata = controller_types.VariableMetadata
_bind_symbolic_endpoints_in_proto = (
    controller_types.bind_symbolic_endpoints_in_proto
)
_extract_host_ip = controller_types.extract_host_ip
_format_units = controller_types.format_units
_is_symbolic_endpoint = controller_types.is_symbolic_endpoint
_is_variable_spec_identical = controller_types.is_variable_spec_identical
_make_symbolic_endpoint = controller_types.make_symbolic_endpoint
_make_symbolic_shards = controller_types.make_symbolic_shards
_parse_symbolic_endpoint = controller_types.parse_symbolic_endpoint
_proto_to_nd_slice = controller_types.proto_to_nd_slice
_raiden_id_from_proto = controller_types.raiden_id_from_proto
_resolve_symbolic_endpoint = controller_types.resolve_symbolic_endpoint
_unit_filename_stem = controller_types.unit_filename_stem


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
      target_src_units: Optional[Sequence[RaidenId]] = None,
      parallel_worker_planning: Optional[bool] = None,
  ) -> _CachedTransferSchedule:
    """Computes transfer schedule math and returns a _CachedTransferSchedule."""
    if group_size <= 0:
      raise ValueError("group_size must be positive")

    if parallel_worker_planning is None:
      env_parallel = (
          os.environ.get("RAIDEN_PARALLEL_WORKER_PLANNING", "").strip().lower()
      )
      env_mode = os.environ.get("RAIDEN_PLANNING_MODE", "").strip().lower()
      parallel_worker_planning = env_parallel in (
          "1",
          "true",
          "yes",
      ) or env_mode in ("parallel", "parallel_worker")

    if (
        parallel_worker_planning
        and target_src_units is None
        and not shard_push_schedules
        and len(src_units) > 1
    ):
      return cls.compute_schedules_in_parallel_workers(
          src_units=src_units,
          dst_units=dst_units,
          dst_metadata=dst_metadata,
          entities=entities,
          registered_variables=registered_variables,
          registered_global_shapes=registered_global_shapes,
          registered_mesh_shapes=registered_mesh_shapes,
          registered_mesh_axes=registered_mesh_axes,
          registered_host_subgrids=registered_host_subgrids,
          registered_layouts=registered_layouts,
          registered_itemsizes=registered_itemsizes,
          registered_shards=registered_shards,
          computed_phys_meshes=computed_phys_meshes,
          worker_endpoints=worker_endpoints,
          broadcast_k=broadcast_k,
          lock=lock,
          group_size=group_size,
          skip_tiling=skip_tiling,
          req_id=req_id,
          uuid=uuid,
      )

    target_src_set = (
        set(target_src_units) if target_src_units is not None else None
    )

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

    variable_plans = {}
    variable_to_plan_id = {}
    local_skip_tiling = dict(skip_tiling) if skip_tiling else {}
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
                is_identical = _is_variable_spec_identical(src_var, dst_var)
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
                is_2d_identical = is_identical and is_2d_or_more
                aligned_decision = is_2d_or_more and (
                    is_2d_identical or all_aligned
                )
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
          s_phys_mesh = registered_mesh_shapes.get(s_unit)
          s_mesh_axes = registered_mesh_axes.get(s_unit)
          s_host_subgrid = registered_host_subgrids.get(s_unit)
        num_src_hosts = max(1, len(s_job_reps))
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
              sharding_spec=list(s_var.sharding_spec),
              mesh_axes=s_mesh_axes,
              physical_mesh_shape=s_phys_mesh,
              host_subgrid=s_host_subgrid,
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

      # Pre-index destination metadata, signatures, and variables for O(1)
      # lookups.
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

      dst_group_sig_by_name = {}
      dst_units_index = {u: i for i, u in enumerate(dst_units)}

      # Track direct block counts during template expansion when tree broadcast
      # cannot be triggered (len(dst_meta_info) <= max(1, broadcast_k)).
      can_fast_path_direct = len(dst_meta_info) <= max(1, broadcast_k)
      fast_dst_unit_counts = {}
      fast_dst_unit_layer_counts = {}
      fast_dst_endpoint_counts = {}
      fast_dst_endpoint_layer_counts = {}
      fast_direct_dsts = []
      fast_direct_dsts_set = set()

      # 3. Generate plan (Intersection) with plan_id dictionary deduplication:
      # Each unique (src_sig, dst_group_sig, skip_tile_flag) is assigned a
      # unique integer `plan_id`. The calculated shard schedule for `plan_id`
      # is stored once in `variable_plans[src_unit][plan_id]`, and every
      # variable (`layer_idx`) stores `variable_to_plan_id[src_unit][layer_idx]`
      # referring to `plan_id`.
      dst_indices_cache = {}
      dst_targets_cache = {}
      chunk_descriptor_cache = {}
      active_slice_chunks_cache = {}
      slice_candidate_cache = {}
      global_sig_to_plan_id = {}
      variable_plans = {}
      variable_to_plan_id = {}
      plan_classification_cache = {}
      plan_unit_counts_by_pid = {}
      plan_host_counts_by_pid = {}

      for src_unit_idx, src_unit in enumerate(src_units):
        src_vars = src_vars_by_unit.get(src_unit, [])
        src_shards = resolve_shards_locked(src_unit)
        unit_plans_by_id = {}
        unit_shard_plans_by_id = {}
        unit_shard_pid_dst_units = {}

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

        if target_src_set is not None and src_unit not in target_src_set:
          continue

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
                sharding_spec=list(src_var.sharding_spec),
                mesh_axes=src_mesh_axes,
                physical_mesh_shape=src_phys_mesh_shape,
                host_subgrid=src_host_subgrid,
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
            for dst_unit in dst_units:
              dst_var = dst_vars_by_unit_and_name.get(dst_unit, {}).get(
                  var_name
              )
              if not dst_var:
                continue

              d_nd_slices = computed_nd_slices.get(dst_unit, {}).get(var_name)
              if not d_nd_slices:
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

              dst_targets.append((
                  dst_unit,
                  dst_unit_idx,
                  is_dst_legacy,
                  num_dst_shards,
                  dst_shard_items,
              ))
            dst_target_bundle = (dst_targets, dst_targets_by_slice)
            dst_targets_cache[dst_group_sig] = dst_target_bundle
          else:
            dst_targets, dst_targets_by_slice = dst_target_bundle

          template_for_var = {}
          tmpl_unit_counts = {}
          tmpl_host_counts = {}
          is_src_legacy = is_legacy_by_unit.get(src_unit, True)
          for (
              local_src_idx,
              global_src_idx,
              candidate_rank,
          ) in src_ranked_indices:
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

            template_entries = []
            shard_dst_units_order = []
            shard_dst_units_seen = set()

            if len(active_slice_chunks) == 1:
              only_key, converted_chunks = next(
                  iter(active_slice_chunks.items())
              )
              if num_candidates > 1:
                cand_cache_key = (dst_group_sig, only_key, num_candidates)
                by_cand = slice_candidate_cache.get(cand_cache_key)
                if by_cand is None:
                  by_cand = {}
                  for item in dst_targets_by_slice[only_key]:
                    by_cand.setdefault(item[1] % num_candidates, []).append(
                        item
                    )
                  slice_candidate_cache[cand_cache_key] = by_cand
                matched_items = by_cand.get(candidate_rank)
              else:
                matched_items = dst_targets_by_slice[only_key]

              if not matched_items:
                continue

              if len(matched_items) > 1:
                shift = src_unit_idx % len(matched_items)
                shifted_matched_items = (
                    matched_items[shift:] + matched_items[:shift]
                )
              else:
                shifted_matched_items = matched_items

              num_c = len(converted_chunks)
              first_chunk = converted_chunks[0]
              for (
                  _,
                  _,
                  local_dst_idx,
                  dst_peer,
                  d_u,
                  d_h,
              ) in shifted_matched_items:
                if d_u is not None:
                  tmpl_unit_counts[d_u] = tmpl_unit_counts.get(d_u, 0) + num_c
                  if d_u not in shard_dst_units_seen:
                    shard_dst_units_seen.add(d_u)
                    shard_dst_units_order.append(d_u)
                  if d_h:
                    tmpl_host_counts[d_h] = tmpl_host_counts.get(d_h, 0) + num_c
                if num_c == 1:
                  template_entries.append(
                      (dst_peer, local_dst_idx, *first_chunk)
                  )
                else:
                  template_entries.extend(
                      (dst_peer, local_dst_idx, *chunk_desc)
                      for chunk_desc in converted_chunks
                  )
            else:
              if len(dst_targets) > 1:
                shift = src_unit_idx % len(dst_targets)
                shifted_dst_targets = dst_targets[shift:] + dst_targets[:shift]
              else:
                shifted_dst_targets = dst_targets
              for (
                  dst_unit,
                  dst_unit_idx,
                  is_dst_legacy,
                  num_dst_shards,
                  dst_shard_items,
              ) in shifted_dst_targets:
                for local_dst_idx, dst_slice, dst_peer in dst_shard_items:
                  converted_chunks = active_slice_chunks.get(
                      (dst_slice, is_dst_legacy)
                  )
                  if not converted_chunks:
                    continue
                  if num_candidates > 1:
                    dst_global_idx = (
                        dst_unit_idx * num_dst_shards + local_dst_idx
                    )
                    if dst_global_idx % num_candidates != candidate_rank:
                      continue
                  num_c = len(converted_chunks)
                  d_u = data_address_to_unit.get(dst_peer)
                  if d_u is not None:
                    tmpl_unit_counts[d_u] = tmpl_unit_counts.get(d_u, 0) + num_c
                    if d_u not in shard_dst_units_seen:
                      shard_dst_units_seen.add(d_u)
                      shard_dst_units_order.append(d_u)
                    d_h = data_address_to_host.get(dst_peer)
                    if d_h:
                      tmpl_host_counts[d_h] = (
                          tmpl_host_counts.get(d_h, 0) + num_c
                      )
                  template_entries.extend(
                      (dst_peer, local_dst_idx, *chunk_desc)
                      for chunk_desc in converted_chunks
                  )

            if template_entries:
              template_for_var[local_src_idx] = template_entries
              unit_shard_plans_by_id.setdefault(local_src_idx, {})[
                  plan_id
              ] = template_entries
              unit_shard_pid_dst_units[(local_src_idx, plan_id)] = (
                  shard_dst_units_order
              )

          unit_plans_by_id[plan_id] = template_for_var
          if can_fast_path_direct:
            layers_tuple = unit_layers_tuple_by_pid.get(plan_id, ())
            if layers_tuple:
              pid_key = (plan_id, layers_tuple)
              pid_u_counts = plan_unit_counts_by_pid.setdefault(pid_key, {})
              for d_u, cnt in tmpl_unit_counts.items():
                pid_u_counts[d_u] = pid_u_counts.get(d_u, 0) + cnt
              pid_h_counts = plan_host_counts_by_pid.setdefault(pid_key, {})
              for d_h, cnt in tmpl_host_counts.items():
                pid_h_counts[d_h] = pid_h_counts.get(d_h, 0) + cnt

        variable_plans[src_unit] = unit_plans_by_id
        variable_to_plan_id[src_unit] = dict(unit_var_to_plan_id)
        if unit_shard_plans_by_id:
          sorted_local_idxs = sorted(unit_shard_plans_by_id.keys())
          computed_schedules[src_unit] = {
              local_src_idx: _PlanReferencedShardSchedule(
                  unit_shard_plans_by_id[local_src_idx],
                  unit_var_to_plan_id,
                  unit_ordered_vars,
              )
              for local_src_idx in sorted_local_idxs
          }
          if can_fast_path_direct and len(fast_direct_dsts_set) < len(
              dst_units
          ):
            unique_ordered_pids = list(
                dict.fromkeys(pid for _, pid in unit_ordered_vars)
            )
            for local_src_idx in sorted_local_idxs:
              for pid in unique_ordered_pids:
                for d_u in unit_shard_pid_dst_units.get(
                    (local_src_idx, pid), ()
                ):
                  if d_u not in fast_direct_dsts_set:
                    fast_direct_dsts_set.add(d_u)
                    fast_direct_dsts.append(d_u)
              if len(fast_direct_dsts_set) == len(dst_units):
                break

      if can_fast_path_direct:
        for (_, layers_tuple), pid_u_counts in plan_unit_counts_by_pid.items():
          num_layers_for_pid = len(layers_tuple)
          for d_u, cnt in pid_u_counts.items():
            fast_dst_unit_counts[d_u] = (
                fast_dst_unit_counts.get(d_u, 0) + cnt * num_layers_for_pid
            )
            layer_map = fast_dst_unit_layer_counts.setdefault(d_u, {})
            for l_idx in layers_tuple:
              layer_map[l_idx] = layer_map.get(l_idx, 0) + cnt
        for (_, layers_tuple), pid_h_counts in plan_host_counts_by_pid.items():
          num_layers_for_pid = len(layers_tuple)
          for d_h, cnt in pid_h_counts.items():
            fast_dst_endpoint_counts[d_h] = (
                fast_dst_endpoint_counts.get(d_h, 0) + cnt * num_layers_for_pid
            )
            h_layer_map = fast_dst_endpoint_layer_counts.setdefault(d_h, {})
            for l_idx in layers_tuple:
              h_layer_map[l_idx] = h_layer_map.get(l_idx, 0) + cnt

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

    if not shard_push_schedules and can_fast_path_direct:
      direct_schedules = {
          u: {s_idx: entries for s_idx, entries in scheds.items() if entries}
          for u, scheds in computed_schedules.items()
          if any(scheds.values())
      }
      broadcast_groups = {}
      dst_unit_counts = fast_dst_unit_counts
      dst_unit_layer_counts = fast_dst_unit_layer_counts
      dst_endpoint_counts = fast_dst_endpoint_counts
      dst_endpoint_layer_counts = fast_dst_endpoint_layer_counts
      computed_expected_block_count = (
          max(dst_unit_counts.values()) if dst_unit_counts else 0
      )
      direct_dsts = fast_direct_dsts
      if direct_schedules:
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
        variable_plans=variable_plans,
        variable_to_plan_id=variable_to_plan_id,
    )

  @classmethod
  def merge_worker_schedules(
      cls,
      worker_schedules: Sequence[_CachedTransferSchedule],
      src_units: Sequence[RaidenId],
      dst_units: Sequence[RaidenId],
      broadcast_k: int = 64,
      group_size: int = 1,
  ) -> _CachedTransferSchedule:
    """Merges independently computed per-worker schedules into a unified schedule."""
    del cls
    if not worker_schedules:
      return _CachedTransferSchedule(
          computed_schedules={},
          direct_schedules={},
          broadcast_groups={},
          local_skip_tiling={},
          expected_block_count=0,
          dst_unit_layer_counts={},
          data_address_to_unit={},
          direct_dsts=[],
          rpc_addresses={},
          data_addresses={u: [] for u in dst_units},
      )

    computed_schedules: dict[Any, Any] = {}
    variable_plans: dict[Any, Any] = {}
    variable_to_plan_id: dict[Any, Any] = {}
    local_skip_tiling: dict[int, bool] = {}
    data_address_to_unit: dict[str, Any] = {}
    rpc_addresses: dict[Any, str] = {}
    data_addresses: dict[Any, list[str]] = {u: [] for u in dst_units}
    is_weight_sync = False

    can_fast_path_direct = len(set(dst_units)) <= max(1, broadcast_k)
    direct_schedules: dict[Any, Any] = {}
    direct_dsts: list[Any] = []
    direct_dsts_set: set[Any] = set()
    dst_unit_counts: dict[Any, int] = {}
    dst_unit_layer_counts: dict[Any, dict[int, int]] = {}
    dst_endpoint_counts: dict[str, int] = {}
    dst_endpoint_layer_counts: dict[str, dict[int, int]] = {}

    sched_by_src: dict[RaidenId, _CachedTransferSchedule] = {}
    for w_sched in worker_schedules:
      for u in w_sched.computed_schedules:
        sched_by_src[u] = w_sched
      for u in w_sched.variable_plans:
        if u not in sched_by_src:
          sched_by_src[u] = w_sched

    ordered_scheds = [
        sched_by_src[u] for u in src_units if u in sched_by_src
    ] or list(worker_schedules)

    for w_sched in ordered_scheds:
      computed_schedules.update(w_sched.computed_schedules)
      variable_plans.update(w_sched.variable_plans)
      variable_to_plan_id.update(w_sched.variable_to_plan_id)
      local_skip_tiling.update(w_sched.local_skip_tiling)
      data_address_to_unit.update(w_sched.data_address_to_unit)
      rpc_addresses.update(w_sched.rpc_addresses)
      for k, v in w_sched.data_addresses.items():
        if v:
          data_addresses[k] = list(v)
      if w_sched.is_weight_sync:
        is_weight_sync = True

      if can_fast_path_direct and not w_sched.broadcast_groups:
        direct_schedules.update(w_sched.direct_schedules)
        for d_u in w_sched.direct_dsts:
          if d_u not in direct_dsts_set:
            direct_dsts_set.add(d_u)
            direct_dsts.append(d_u)
        for d_u, cnt in w_sched.dst_unit_counts.items():
          dst_unit_counts[d_u] = dst_unit_counts.get(d_u, 0) + cnt
        for d_u, l_map in w_sched.dst_unit_layer_counts.items():
          target_l_map = dst_unit_layer_counts.setdefault(d_u, {})
          for l_idx, cnt in l_map.items():
            target_l_map[l_idx] = target_l_map.get(l_idx, 0) + cnt
        for d_h, cnt in w_sched.dst_endpoint_counts.items():
          dst_endpoint_counts[d_h] = dst_endpoint_counts.get(d_h, 0) + cnt
        for d_h, l_map in w_sched.dst_endpoint_layer_counts.items():
          target_h_map = dst_endpoint_layer_counts.setdefault(d_h, {})
          for l_idx, cnt in l_map.items():
            target_h_map[l_idx] = target_h_map.get(l_idx, 0) + cnt

    if can_fast_path_direct and all(
        not w.broadcast_groups for w in ordered_scheds
    ):
      broadcast_groups = {}
      expected_block_count = (
          max(dst_unit_counts.values()) if dst_unit_counts else 0
      )
    else:
      groups = {}
      for src_unit in src_units:
        schedules = computed_schedules.get(src_unit, {})
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
      expected_block_count = 0
      direct_dsts = []
      if direct_schedules:
        for _, schedules in direct_schedules.items():
          for _, entries in schedules.items():
            for entry in entries:
              dst_peer = entry[0]
              dst_unit = data_address_to_unit.get(dst_peer)
              if dst_unit:
                if dst_unit not in direct_dsts:
                  direct_dsts.append(dst_unit)
                layer_idx = entry[10] if len(entry) > 10 else 0
                dst_unit_counts[dst_unit] = dst_unit_counts.get(dst_unit, 0) + 1
                dst_unit_layer_counts.setdefault(dst_unit, {})[layer_idx] = (
                    dst_unit_layer_counts.get(dst_unit, {}).get(layer_idx, 0)
                    + 1
                )
                dst_host = _extract_host_ip(dst_peer)
                if dst_host:
                  dst_endpoint_counts[dst_host] = (
                      dst_endpoint_counts.get(dst_host, 0) + 1
                  )
                  dst_endpoint_layer_counts.setdefault(dst_host, {})[
                      layer_idx
                  ] = (
                      dst_endpoint_layer_counts.get(dst_host, {}).get(
                          layer_idx, 0
                      )
                      + 1
                  )
        if dst_unit_counts:
          expected_block_count = max(dst_unit_counts.values())

    return _CachedTransferSchedule(
        computed_schedules=computed_schedules,
        direct_schedules=direct_schedules,
        broadcast_groups=broadcast_groups,
        local_skip_tiling=local_skip_tiling,
        expected_block_count=expected_block_count,
        dst_unit_layer_counts=dst_unit_layer_counts,
        data_address_to_unit=data_address_to_unit,
        direct_dsts=direct_dsts,
        rpc_addresses=rpc_addresses,
        data_addresses=data_addresses,
        dst_unit_counts=dst_unit_counts,
        dst_endpoint_counts=dst_endpoint_counts,
        dst_endpoint_layer_counts=dst_endpoint_layer_counts,
        is_weight_sync=is_weight_sync,
        variable_plans=variable_plans,
        variable_to_plan_id=variable_to_plan_id,
    )

  @classmethod
  def compute_schedules_in_parallel_workers(
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
      req_id: str = "warmup",
      uuid: Any = "",
      max_workers: Optional[int] = None,
  ) -> _CachedTransferSchedule:
    """Computes each source worker's schedule in parallel and merges the results."""

    def _compute_for_unit(unit: RaidenId) -> _CachedTransferSchedule:
      local_phys_meshes: dict[RaidenId, list[int]] = {}
      sched = cls.compute_transfer_schedule_from_metadata(
          src_units=src_units,
          dst_units=dst_units,
          dst_metadata=dst_metadata,
          entities=entities,
          registered_variables=registered_variables,
          registered_global_shapes=registered_global_shapes,
          registered_mesh_shapes=registered_mesh_shapes,
          registered_mesh_axes=registered_mesh_axes,
          registered_host_subgrids=registered_host_subgrids,
          registered_layouts=registered_layouts,
          registered_itemsizes=registered_itemsizes,
          registered_shards=registered_shards,
          computed_phys_meshes=local_phys_meshes,
          worker_endpoints=worker_endpoints,
          broadcast_k=broadcast_k,
          lock=lock,
          group_size=group_size,
          skip_tiling=skip_tiling,
          req_id=req_id,
          uuid=uuid,
          target_src_units=[unit],
          parallel_worker_planning=False,
      )
      with lock:
        computed_phys_meshes.update(local_phys_meshes)
      return sched

    num_threads = max_workers or min(32, max(1, len(src_units)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
      worker_schedules = list(pool.map(_compute_for_unit, src_units))

    return cls.merge_worker_schedules(
        worker_schedules=worker_schedules,
        src_units=src_units,
        dst_units=dst_units,
        broadcast_k=broadcast_k,
        group_size=group_size,
    )

  @classmethod
  def compute_offline_schedule(
      cls,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      src_variables: Optional[Mapping[RaidenId, list[Any]]] = None,
      dst_variables: Optional[Mapping[RaidenId, list[Any]]] = None,
      num_src_shards_per_unit: int = 8,
      num_dst_shards_per_unit: int = 8,
      src_mesh_shapes: Optional[Mapping[RaidenId, Sequence[int]]] = None,
      src_mesh_axes: Optional[Mapping[RaidenId, Sequence[str]]] = None,
      src_host_subgrids: Optional[Mapping[RaidenId, Sequence[int]]] = None,
      dst_mesh_shapes: Optional[Mapping[RaidenId, Sequence[int]]] = None,
      dst_mesh_axes: Optional[Mapping[RaidenId, Sequence[str]]] = None,
      dst_host_subgrids: Optional[Mapping[RaidenId, Sequence[int]]] = None,
      num_shards_by_unit: Optional[Mapping[RaidenId, int]] = None,
      broadcast_k: int = 64,
      group_size: int = 1,
      skip_tiling: Optional[dict[int, bool]] = None,
      target_src_units: Optional[Sequence[RaidenId]] = None,
      parallel_worker_planning: bool = False,
      **legacy_kwargs: Any,
  ) -> _CachedTransferSchedule:
    """Computes a symbolic transfer schedule offline without live worker IP:ports."""
    lock = legacy_kwargs.get("lock") or threading.Lock()
    if src_variables is None:
      src_variables = legacy_kwargs.get("registered_variables", {})
    if src_mesh_shapes is None:
      src_mesh_shapes = legacy_kwargs.get("registered_mesh_shapes")
    if src_mesh_axes is None:
      src_mesh_axes = legacy_kwargs.get("registered_mesh_axes")
    if src_host_subgrids is None:
      src_host_subgrids = legacy_kwargs.get("registered_host_subgrids")
    legacy_shards = legacy_kwargs.get("registered_shards", {})
    legacy_dst_metadata = legacy_kwargs.get("dst_metadata")

    entities = {}
    registered_shards = {}
    registered_mesh_shapes = {}
    registered_mesh_axes = {}
    registered_host_subgrids = {}
    worker_endpoints = {}

    for u in src_units:
      if num_shards_by_unit and u in num_shards_by_unit:
        n_shards = num_shards_by_unit[u]
      elif u in legacy_shards and legacy_shards[u]:
        n_shards = len(legacy_shards[u])
      else:
        n_shards = num_src_shards_per_unit
      shards = _make_symbolic_shards(u, n_shards)
      registered_shards[u] = shards
      entities[u] = JobEntity(unit=u, shards=shards)
      if src_mesh_shapes and u in src_mesh_shapes:
        registered_mesh_shapes[u] = list(src_mesh_shapes[u])
      if src_mesh_axes and u in src_mesh_axes:
        registered_mesh_axes[u] = list(src_mesh_axes[u])
      if src_host_subgrids and u in src_host_subgrids:
        registered_host_subgrids[u] = list(src_host_subgrids[u])
      worker_endpoints[u] = _make_symbolic_endpoint(u, 0)

    dst_metadata = []
    if legacy_dst_metadata is not None and dst_variables is None:
      for item in legacy_dst_metadata:
        u = controller_types.raiden_id_from_proto(item.unit)
        if num_shards_by_unit and u in num_shards_by_unit:
          n_shards = num_shards_by_unit[u]
        elif item.shards:
          n_shards = len(item.shards)
        else:
          n_shards = num_dst_shards_per_unit
        sym_shards = _make_symbolic_shards(u, n_shards)
        registered_shards[u] = sym_shards
        meta = raiden_service_pb2.RegisterWorkUnitRequest()
        meta.CopyFrom(item)
        del meta.shards[:]
        meta.shards.extend(sym_shards)
        meta.control_plane_rpc_address = _make_symbolic_endpoint(u, 0)
        dst_metadata.append(meta)
    else:
      dst_vars_map = dst_variables or {}
      proto_vars_cache = {}
      for u in dst_units:
        if num_shards_by_unit and u in num_shards_by_unit:
          n_shards = num_shards_by_unit[u]
        elif u in legacy_shards and legacy_shards[u]:
          n_shards = len(legacy_shards[u])
        else:
          n_shards = num_dst_shards_per_unit
        shards = _make_symbolic_shards(u, n_shards)
        registered_shards[u] = shards
        meta = raiden_service_pb2.RegisterWorkUnitRequest(
            unit=raiden_service_pb2.RaidenIdProto(
                job_name=u.job_name,
                job_replica_id=str(u.job_replica_id),
                data_name=u.data_name,
                data_replica_idx=u.data_replica_idx,
            ),
            control_plane_rpc_address=_make_symbolic_endpoint(u, 0),
        )
        meta.shards.extend(shards)
        if dst_mesh_shapes and u in dst_mesh_shapes:
          meta.mesh_shape.extend(dst_mesh_shapes[u])
        if dst_mesh_axes and u in dst_mesh_axes:
          meta.mesh_axes.extend(dst_mesh_axes[u])
        if dst_host_subgrids and u in dst_host_subgrids:
          meta.host_subgrid.extend(dst_host_subgrids[u])
        u_vars = dst_vars_map.get(u, [])
        proto_vars = proto_vars_cache.get(id(u_vars))
        if proto_vars is None:
          tmpl = raiden_service_pb2.RegisterWorkUnitRequest()
          for v in u_vars:
            vp = tmpl.variables.add()
            vp.CopyFrom(
                controller_types.coerce_variable_proto(v, raiden_service_pb2)
            )
          proto_vars = list(tmpl.variables)
          proto_vars_cache[id(u_vars)] = proto_vars
        meta.variables.extend(proto_vars)
        dst_metadata.append(meta)

    return cls.compute_transfer_schedule_from_metadata(
        src_units=src_units,
        dst_units=dst_units,
        dst_metadata=dst_metadata,
        entities=entities,
        registered_variables=src_variables,
        registered_global_shapes=legacy_kwargs.get(
            "registered_global_shapes", {}
        ),
        registered_mesh_shapes=registered_mesh_shapes,
        registered_mesh_axes=registered_mesh_axes,
        registered_host_subgrids=registered_host_subgrids,
        registered_layouts=legacy_kwargs.get("registered_layouts", {}),
        registered_itemsizes=legacy_kwargs.get("registered_itemsizes", {}),
        registered_shards=registered_shards,
        computed_phys_meshes={},
        worker_endpoints=worker_endpoints,
        broadcast_k=broadcast_k,
        lock=lock,
        group_size=group_size,
        skip_tiling=skip_tiling,
        req_id="offline",
        uuid=0,
        target_src_units=target_src_units,
        parallel_worker_planning=parallel_worker_planning,
    )

  @classmethod
  def bind_symbolic_schedule(
      cls,
      schedule: _CachedTransferSchedule,
      live_data_addresses: Mapping[RaidenId, Sequence[str]],
      live_rpc_addresses: Optional[Mapping[RaidenId, str]] = None,
  ) -> _CachedTransferSchedule:
    """Binds symbolic endpoints in an offline _CachedTransferSchedule to live IP:ports."""
    del cls
    if not live_data_addresses:
      return schedule

    bound_variable_plans: dict[Any, dict[int, dict[int, list[Any]]]] = {}
    bound_computed_schedules: dict[Any, Any] = {}

    for src_unit, unit_plans_by_id in schedule.variable_plans.items():
      new_unit_plans_by_id: dict[int, dict[int, list[Any]]] = {}
      new_unit_shard_plans_by_id: dict[int, dict[int, list[Any]]] = {}
      for pid, tmpl_for_var in unit_plans_by_id.items():
        new_tmpl_for_var: dict[int, list[Any]] = {}
        for local_src_idx, entries in tmpl_for_var.items():
          bound_entries = [
              (_resolve_symbolic_endpoint(e[0], live_data_addresses), *e[1:])
              for e in entries
          ]
          new_tmpl_for_var[local_src_idx] = bound_entries
          new_unit_shard_plans_by_id.setdefault(local_src_idx, {})[
              pid
          ] = bound_entries
        new_unit_plans_by_id[pid] = new_tmpl_for_var
      bound_variable_plans[src_unit] = new_unit_plans_by_id

      orig_unit_scheds = schedule.computed_schedules.get(src_unit, {})
      unit_var_to_pid = schedule.variable_to_plan_id.get(src_unit, {})
      new_unit_scheds = {}
      for local_src_idx, orig_sched in orig_unit_scheds.items():
        ordered_vars = getattr(orig_sched, "_ordered_vars", None)
        if local_src_idx in new_unit_shard_plans_by_id:
          new_unit_scheds[local_src_idx] = _PlanReferencedShardSchedule(
              new_unit_shard_plans_by_id[local_src_idx],
              unit_var_to_pid,
              ordered_vars,
          )
        elif isinstance(orig_sched, (list, tuple)):
          new_unit_scheds[local_src_idx] = [
              (_resolve_symbolic_endpoint(e[0], live_data_addresses), *e[1:])
              for e in orig_sched
          ]
      if new_unit_scheds:
        bound_computed_schedules[src_unit] = new_unit_scheds

    for src_unit, orig_unit_scheds in schedule.computed_schedules.items():
      if src_unit not in bound_computed_schedules:
        new_unit_scheds = {}
        for local_src_idx, orig_sched in orig_unit_scheds.items():
          new_unit_scheds[local_src_idx] = [
              (_resolve_symbolic_endpoint(e[0], live_data_addresses), *e[1:])
              for e in orig_sched
          ]
        bound_computed_schedules[src_unit] = new_unit_scheds

    if not schedule.broadcast_groups:
      bound_direct_schedules = {
          u: {s_idx: entries for s_idx, entries in scheds.items() if entries}
          for u, scheds in bound_computed_schedules.items()
          if any(scheds.values())
      }
      bound_broadcast_groups = {}
    else:
      bound_direct_schedules = {}
      for src_unit, scheds in schedule.direct_schedules.items():
        if (
            src_unit in bound_computed_schedules
            and scheds == schedule.computed_schedules.get(src_unit)
        ):
          bound_direct_schedules[src_unit] = bound_computed_schedules[src_unit]
        else:
          bound_direct_schedules[src_unit] = {
              s_idx: [
                  (
                      _resolve_symbolic_endpoint(e[0], live_data_addresses),
                      *e[1:],
                  )
                  for e in entries
              ]
              for s_idx, entries in scheds.items()
          }
      bound_broadcast_groups = {}
      for group_key, k_and_t_list in schedule.broadcast_groups.items():
        bound_list = []
        for k, targets in k_and_t_list:
          bound_targets = [
              (
                  t[0],
                  _resolve_symbolic_endpoint(t[1], live_data_addresses),
                  *t[2:],
              )
              for t in targets
          ]
          bound_list.append((k, bound_targets))
        bound_broadcast_groups[group_key] = bound_list

    bound_data_addresses = dict(schedule.data_addresses)
    bound_data_address_to_unit = {}
    for u, shards in bound_data_addresses.items():
      if u in live_data_addresses and live_data_addresses[u]:
        bound_shards = list(live_data_addresses[u])
      else:
        bound_shards = [
            _resolve_symbolic_endpoint(s, live_data_addresses) for s in shards
        ]
      bound_data_addresses[u] = bound_shards
      for s in bound_shards:
        bound_data_address_to_unit[s] = u
    for u, shards in live_data_addresses.items():
      if u not in bound_data_addresses and shards:
        bound_data_addresses[u] = list(shards)
      for s in shards:
        bound_data_address_to_unit[s] = u

    bound_dst_endpoint_counts: dict[str, int] = {}
    for sym_host, cnt in schedule.dst_endpoint_counts.items():
      parsed = _parse_symbolic_endpoint(f"{sym_host}:0")
      if parsed is not None and parsed[0] in live_data_addresses:
        live_shards = live_data_addresses[parsed[0]]
        live_host = (
            _extract_host_ip(live_shards[0]) if live_shards else sym_host
        )
      else:
        live_host = sym_host
      bound_dst_endpoint_counts[live_host] = (
          bound_dst_endpoint_counts.get(live_host, 0) + cnt
      )

    bound_dst_endpoint_layer_counts: dict[str, dict[int, int]] = {}
    for sym_host, l_map in schedule.dst_endpoint_layer_counts.items():
      parsed = _parse_symbolic_endpoint(f"{sym_host}:0")
      if parsed is not None and parsed[0] in live_data_addresses:
        live_shards = live_data_addresses[parsed[0]]
        live_host = (
            _extract_host_ip(live_shards[0]) if live_shards else sym_host
        )
      else:
        live_host = sym_host
      target_map = bound_dst_endpoint_layer_counts.setdefault(live_host, {})
      for l_idx, cnt in l_map.items():
        target_map[l_idx] = target_map.get(l_idx, 0) + cnt

    bound_rpc_addresses = dict(schedule.rpc_addresses)
    if live_rpc_addresses:
      bound_rpc_addresses.update(live_rpc_addresses)

    return _CachedTransferSchedule(
        computed_schedules=bound_computed_schedules,
        direct_schedules=bound_direct_schedules,
        broadcast_groups=bound_broadcast_groups,
        local_skip_tiling=dict(schedule.local_skip_tiling),
        expected_block_count=schedule.expected_block_count,
        dst_unit_layer_counts={
            u: dict(m) for u, m in schedule.dst_unit_layer_counts.items()
        },
        data_address_to_unit=bound_data_address_to_unit,
        direct_dsts=list(schedule.direct_dsts),
        rpc_addresses=bound_rpc_addresses,
        data_addresses=bound_data_addresses,
        dst_unit_counts=dict(schedule.dst_unit_counts),
        dst_endpoint_counts=bound_dst_endpoint_counts,
        dst_endpoint_layer_counts=bound_dst_endpoint_layer_counts,
        is_weight_sync=schedule.is_weight_sync,
        variable_plans=bound_variable_plans,
        variable_to_plan_id={
            u: dict(m) for u, m in schedule.variable_to_plan_id.items()
        },
    )

  @classmethod
  def _to_json_serializable(cls, obj: Any) -> Any:
    """Recursively converts a schedule structure to JSON-compatible primitives."""
    if isinstance(obj, RaidenId):
      return {
          "__raiden_id__": [
              obj.job_name,
              str(obj.job_replica_id),
              obj.data_name,
              int(obj.data_replica_idx),
          ]
      }
    if isinstance(obj, _PlanReferencedShardSchedule):
      ordered_vars = getattr(obj, "_ordered_vars", [])
      return {
          "__plan_ref__": [
              [int(l_idx), int(pid)] for l_idx, pid in ordered_vars
          ]
      }
    if isinstance(obj, _CachedTransferSchedule):
      skip_fields = {
          "sender_push_schedule_protos",
          "cached_serialized_payloads",
      }
      return {
          "__cached_schedule__": {
              f.name: cls._to_json_serializable(getattr(obj, f.name))
              for f in dataclasses.fields(obj)
              if f.name not in skip_fields
          }
      }
    if isinstance(obj, tuple):
      return {"__tuple__": [cls._to_json_serializable(x) for x in obj]}
    if isinstance(obj, list):
      return [cls._to_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
      if all(isinstance(k, str) for k in obj.keys()):
        return {k: cls._to_json_serializable(v) for k, v in obj.items()}
      return {
          "__dict__": [
              [
                  cls._to_json_serializable(k),
                  cls._to_json_serializable(v),
              ]
              for k, v in obj.items()
          ]
      }
    return obj

  @classmethod
  def _from_json_serializable(cls, obj: Any) -> Any:
    """Recursively reconstructs a schedule structure from JSON primitives."""
    if isinstance(obj, list):
      return [cls._from_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
      if "__raiden_id__" in obj:
        parts = obj["__raiden_id__"]
        return RaidenId(
            str(parts[0]), str(parts[1]), str(parts[2]), int(parts[3])
        )
      if "__tuple__" in obj:
        return tuple(cls._from_json_serializable(x) for x in obj["__tuple__"])
      if "__dict__" in obj:
        return {
            cls._from_json_serializable(k): cls._from_json_serializable(v)
            for k, v in obj["__dict__"]
        }
      if "__plan_ref__" in obj:
        return {"__plan_ref__": [tuple(pair) for pair in obj["__plan_ref__"]]}
      if "__cached_schedule__" in obj:
        fields_dict = {
            k: cls._from_json_serializable(v)
            for k, v in obj["__cached_schedule__"].items()
        }
        var_plans = fields_dict.get("variable_plans", {})
        var_to_pid = fields_dict.get("variable_to_plan_id", {})
        for src_unit, unit_plans_by_id in var_plans.items():
          shard_plans_by_id: dict[int, dict[int, list[Any]]] = {}
          for pid, tmpl_for_var in unit_plans_by_id.items():
            for local_src_idx, entries in tmpl_for_var.items():
              shard_plans_by_id.setdefault(local_src_idx, {})[pid] = entries
          unit_var_to_pid = var_to_pid.get(src_unit, {})
          for sched_key in ("computed_schedules", "direct_schedules"):
            unit_scheds = fields_dict.get(sched_key, {}).get(src_unit)
            if not unit_scheds:
              continue
            for local_src_idx, val in list(unit_scheds.items()):
              if isinstance(val, dict) and "__plan_ref__" in val:
                unit_scheds[local_src_idx] = _PlanReferencedShardSchedule(
                    shard_plans_by_id.get(local_src_idx, {}),
                    unit_var_to_pid,
                    val["__plan_ref__"],
                )
        return _CachedTransferSchedule(**fields_dict)
      return {k: cls._from_json_serializable(v) for k, v in obj.items()}
    return obj

  @classmethod
  def save_offline_plan(
      cls,
      schedule: _CachedTransferSchedule,
      path: str,
      src_units: Optional[Sequence[RaidenId]] = None,
      dst_units: Optional[Sequence[RaidenId]] = None,
      dst_mem_type: int = controller_types.RaidenMemoryType.DRAM,
      parallelism: int = 1,
  ) -> dict[Any, str]:
    """Saves per-worker ControlRequest protobufs and schedule bundle to `path`."""
    os.makedirs(path, exist_ok=True)
    if src_units is not None:
      src_list = list(src_units)
    else:
      src_keys = (
          schedule.computed_schedules.keys() or schedule.direct_schedules.keys()
      )
      src_list = list(src_keys)
    src_set = set(src_list)
    if dst_units is not None:
      dst_list = list(dst_units)
    else:
      inferred_dsts = [u for u in schedule.data_addresses if u not in src_set]
      dst_list = inferred_dsts or list(schedule.dst_unit_counts.keys())

    raw_schedules = schedule.direct_schedules or schedule.computed_schedules
    transfer_plan = controller_types.TransferPlan(
        src_units=src_list,
        dst_units=dst_list,
        plan=None,
        shard_push_schedules=raw_schedules,
        worker_rpc_addresses=dict(schedule.rpc_addresses),
        worker_data_addresses=dict(schedule.data_addresses),
        uuid=0,
        dst_mem_type=dst_mem_type,
        use_block_chunks=True,
        is_sender=True,
        expected_block_count=schedule.expected_block_count,
        dst_expected_layer_chunk_counts=schedule.dst_unit_layer_counts,
        dst_expected_block_counts=schedule.dst_unit_counts,
        dst_endpoint_counts=schedule.dst_endpoint_counts,
        dst_endpoint_layer_counts=schedule.dst_endpoint_layer_counts,
        src_schedule_keys={u: i for i, u in enumerate(src_list)},
        req_id="offline",
        skip_d2h=False,
        skip_tiling=schedule.local_skip_tiling,
        parallelism=parallelism,
        is_weight_sync=schedule.is_weight_sync,
        variable_plans=schedule.variable_plans,
        variable_to_plan_id=schedule.variable_to_plan_id,
    )

    written_files: dict[Any, str] = {}
    for u in src_list:
      ent = JobEntity(
          unit=u,
          shards=schedule.data_addresses.get(u) or _make_symbolic_shards(u, 1),
          weight_sync_mode=schedule.is_weight_sync,
      )
      payload = ent.encode_start_transfer(transfer_plan, unit=u)
      if payload:
        file_path = os.path.join(path, f"{_unit_filename_stem(u)}.pb")
        with open(file_path, "wb") as f:
          f.write(payload)
        written_files[u] = file_path
        written_files[controller_types.format_unit(u)] = file_path

    for u in dst_list:
      ent = JobEntity(
          unit=u,
          shards=schedule.data_addresses.get(u) or _make_symbolic_shards(u, 1),
          weight_sync_mode=schedule.is_weight_sync,
      )
      payload = ent.encode_start_transfer(transfer_plan, unit=u)
      if payload:
        file_path = os.path.join(path, f"{_unit_filename_stem(u)}.pb")
        with open(file_path, "wb") as f:
          f.write(payload)
        written_files[u] = file_path
        written_files[controller_types.format_unit(u)] = file_path

    bundle_path = os.path.join(path, "offline_schedule.json")
    payload_dict = cls._to_json_serializable({
        "schedule": schedule,
        "src_units": src_list,
        "dst_units": dst_list,
    })
    with open(bundle_path, "w", encoding="utf-8") as f:
      json.dump(payload_dict, f)
    written_files["__schedule_bundle__"] = bundle_path
    return written_files

  @classmethod
  def load_offline_schedule(
      cls,
      path: str,
      live_data_addresses: Optional[Mapping[RaidenId, Sequence[str]]] = None,
      live_rpc_addresses: Optional[Mapping[RaidenId, str]] = None,
      registered_shards: Optional[Mapping[RaidenId, Sequence[str]]] = None,
      worker_endpoints: Optional[Mapping[RaidenId, str]] = None,
      entities: Optional[Mapping[RaidenId, Any]] = None,
  ) -> _CachedTransferSchedule:
    """Loads an offline _CachedTransferSchedule from `path` and binds live endpoints."""
    del entities
    bundle_path = (
        os.path.join(path, "offline_schedule.json")
        if os.path.isdir(path)
        else path
    )
    with open(bundle_path, "r", encoding="utf-8") as f:
      raw_bundle = json.load(f)
    bundle = cls._from_json_serializable(raw_bundle)
    schedule = bundle["schedule"] if isinstance(bundle, dict) else bundle
    effective_data = (
        live_data_addresses
        if live_data_addresses is not None
        else registered_shards
    )
    effective_rpc = (
        live_rpc_addresses
        if live_rpc_addresses is not None
        else worker_endpoints
    )
    if effective_data:
      return cls.bind_symbolic_schedule(
          schedule,
          live_data_addresses=effective_data,
          live_rpc_addresses=effective_rpc,
      )
    return schedule

  @classmethod
  def load_offline_worker_plan(
      cls,
      path: str,
      unit: RaidenId,
      live_data_addresses: Optional[Mapping[RaidenId, Sequence[str]]] = None,
      registered_shards: Optional[Mapping[RaidenId, Sequence[str]]] = None,
      uuid: Optional[int] = None,
      req_id: Optional[str] = None,
      skip_d2h: Optional[bool] = None,
  ) -> raiden_service_pb2.ControlRequest:
    """Loads a single worker's precomputed ControlRequest proto and binds live endpoints."""
    del cls
    file_path = (
        os.path.join(path, f"{_unit_filename_stem(unit)}.pb")
        if os.path.isdir(path)
        else path
    )
    with open(file_path, "rb") as f:
      raw_bytes = f.read()
    req = raiden_service_pb2.ControlRequest()
    req.ParseFromString(raw_bytes)
    effective_data = (
        live_data_addresses
        if live_data_addresses is not None
        else registered_shards
    )
    if effective_data:
      _bind_symbolic_endpoints_in_proto(req, effective_data)
    if uuid is not None:
      req.start_transfer_request.uuid = int(uuid)
    if req_id is not None:
      req.start_transfer_request.req_id = str(req_id)
    if skip_d2h is not None:
      req.start_transfer_request.skip_d2h = bool(skip_d2h)
    return req

  @classmethod
  def load_offline_worker_plans_parallel(
      cls,
      path: str,
      units: Sequence[RaidenId],
      live_data_addresses: Optional[Mapping[RaidenId, Sequence[str]]] = None,
      registered_shards: Optional[Mapping[RaidenId, Sequence[str]]] = None,
      uuid: Optional[int] = None,
      req_id: Optional[str] = None,
      skip_d2h: Optional[bool] = None,
      max_workers: Optional[int] = None,
  ) -> dict[RaidenId, raiden_service_pb2.ControlRequest]:
    """Loads and binds precomputed ControlRequest plans for `units` in parallel."""
    unit_list = list(units)
    if not unit_list:
      return {}
    effective_data = (
        live_data_addresses
        if live_data_addresses is not None
        else registered_shards
    )

    def _load_one(
        u: RaidenId,
    ) -> tuple[RaidenId, raiden_service_pb2.ControlRequest]:
      return (
          u,
          cls.load_offline_worker_plan(
              path=path,
              unit=u,
              live_data_addresses=effective_data,
              uuid=uuid,
              req_id=req_id,
              skip_d2h=skip_d2h,
          ),
      )

    num_threads = max_workers or min(32, max(1, len(unit_list)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
      return dict(pool.map(_load_one, unit_list))
