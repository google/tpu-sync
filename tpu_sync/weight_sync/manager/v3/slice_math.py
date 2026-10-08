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

"""Thin Python wrappers around C++ `_controller_v3` N-D slice and reshard math."""

from typing import Any, Optional, Sequence

from tpu_sync.weight_sync.manager.v3 import _controller_v3
from tpu_sync.weight_sync.manager.v3 import types

NDSlice = types.NDSlice
VariableBundleSpec = types.VariableBundleSpec
_bounds_to_cpp_nd_slice = types.bounds_to_cpp_nd_slice


def compute_seed_sampler_count(
    num_samplers: int,
    broadcast_host_ratio: float,
    trainer_hosts: int = 1,
    sampler_hosts: int = 1,
) -> int:
  """Computes `n_seed` matching trainer egress bandwidth to seed sampler ingress."""
  return int(
      _controller_v3.compute_seed_sampler_count(
          int(num_samplers),
          float(broadcast_host_ratio),
          int(trainer_hosts),
          int(sampler_hosts),
      )
  )


def intersect_nd_slices(
    slice_a: Any, slice_b: Any
) -> Optional[list[tuple[int, int]]]:
  """Computes intersection of two N-D slices via C++ V3 planner."""
  nd_a = _bounds_to_cpp_nd_slice(slice_a)
  nd_b = _bounds_to_cpp_nd_slice(slice_b)
  res = _controller_v3.intersect_nd_slices(nd_a, nd_b)
  if res is None:
    return None
  return [(int(off), int(off + sz)) for off, sz in zip(res.offsets, res.sizes)]


def is_nd_slice_tile_aligned(
    src_shard_slice: Any,
    dst_shard_slice: Any,
    intersection_slice: Any,
    tile_shape: tuple[int, int] = (8, 128),
) -> bool:
  """Checks whether slices and their intersection align to hardware tiles."""
  src_b = list(
      src_shard_slice.dims
      if isinstance(src_shard_slice, NDSlice)
      else src_shard_slice
  )
  dst_b = list(
      dst_shard_slice.dims
      if isinstance(dst_shard_slice, NDSlice)
      else dst_shard_slice
  )
  int_b = list(
      intersection_slice.dims
      if isinstance(intersection_slice, NDSlice)
      else intersection_slice
  )
  rank = len(src_b)
  if rank < 2:
    return False
  t_row, t_col = tile_shape
  s_row_s, s_row_e = src_b[-2]
  s_col_s, s_col_e = src_b[-1]
  d_row_s, d_row_e = dst_b[-2]
  d_col_s, d_col_e = dst_b[-1]
  i_row_s, i_row_e = int_b[-2]
  i_col_s, i_col_e = int_b[-1]

  if (i_row_s - s_row_s) % t_row != 0 or (i_row_s - d_row_s) % t_row != 0:
    return False
  if (i_col_s - s_col_s) % t_col != 0 or (i_col_s - d_col_s) % t_col != 0:
    return False
  if (i_row_e - i_row_s) % t_row != 0 or (i_col_e - i_col_s) % t_col != 0:
    return False
  if (s_col_e - s_col_s) % t_col != 0 or (d_col_e - d_col_s) % t_col != 0:
    return False
  if (s_row_e - s_row_s) % t_row != 0 or (d_row_e - d_row_s) % t_row != 0:
    return False
  return True


def to_physical(
    logical_shape: Sequence[int],
    logical_mesh_shape: Sequence[int],
    minor_to_major: Optional[Sequence[int]] = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
  """Maps logical tensor and mesh shapes to major-to-minor physical layout.

  Args:
    logical_shape: Logical tensor shape.
    logical_mesh_shape: Logical mesh shape, one entry per tensor dimension.
    minor_to_major: Layout as dimension indices, minor first. Defaults to
      row-major (`[rank - 1, ..., 0]`), whose physical shapes are the logical
      ones.

  Returns:
    `(physical_shape, physical_mesh_shape)`, major to minor.
  """
  if minor_to_major is None:
    minor_to_major = range(len(logical_shape) - 1, -1, -1)
  logical_shape_list = list(logical_shape)
  logical_mesh_list = list(logical_mesh_shape)
  m2m = list(minor_to_major)
  maj2min = list(reversed(m2m))
  rank = len(logical_shape_list)
  if sorted(m2m) == list(range(rank)):
    phys_shape = tuple(logical_shape_list[d] for d in maj2min)
    phys_mesh = tuple(logical_mesh_list[d] for d in maj2min)
  else:
    phys_shape = tuple(logical_shape_list[m2m.index(d)] for d in maj2min)
    phys_mesh = tuple(logical_mesh_list[d] for d in maj2min)
  return phys_shape, phys_mesh


def get_global_indices(idx: int, shape: Sequence[int]) -> list[int]:
  """Computes row-major N-D mesh coordinates for `idx` via C++ V3 planner."""
  return list(
      _controller_v3.get_global_indices(int(idx), [int(x) for x in shape])
  )


def generate_strided_copy_chunks(
    src_slice: Any,
    dst_slice: Any,
    intersection: Any,
    itemsize: int = 1,
    src_layout: Optional[Sequence[int]] = None,
    dst_layout: Optional[Sequence[int]] = None,
) -> list[tuple[int, int, int, int, int, int]]:
  """Generates strided copy chunks `(src_off, dst_off, size, src_stride, dst_stride, count)`."""
  nd_src = _bounds_to_cpp_nd_slice(src_slice)
  nd_dst = _bounds_to_cpp_nd_slice(dst_slice)
  nd_int = _bounds_to_cpp_nd_slice(intersection)
  rank = len(nd_src.sizes)
  default_layout = list(range(rank - 1, -1, -1))
  s_lay = (
      [int(x) for x in src_layout] if src_layout is not None else default_layout
  )
  d_lay = (
      [int(x) for x in dst_layout] if dst_layout is not None else default_layout
  )
  return list(
      _controller_v3.generate_strided_copy_chunks(
          nd_src, nd_dst, nd_int, s_lay, d_lay, int(itemsize)
      )
  )


def generate_strided_copy_chunks_tile_aware(
    src_slice: Any,
    dst_slice: Any,
    intersection: Any,
    itemsize: int = 1,
    tile_shape: tuple[int, int] = (8, 128),
    src_layout: Optional[Sequence[int]] = None,
    dst_layout: Optional[Sequence[int]] = None,
) -> list[tuple[int, int, int, int, int, int]]:
  """Generates tile-aware strided copy chunks for TPU `(8, 128)` tiled buffers."""
  del tile_shape
  nd_src = _bounds_to_cpp_nd_slice(src_slice)
  nd_dst = _bounds_to_cpp_nd_slice(dst_slice)
  nd_int = _bounds_to_cpp_nd_slice(intersection)
  rank = len(nd_src.sizes)
  default_layout = list(range(rank - 1, -1, -1))
  s_lay = (
      [int(x) for x in src_layout] if src_layout is not None else default_layout
  )
  d_lay = (
      [int(x) for x in dst_layout] if dst_layout is not None else default_layout
  )
  return list(
      _controller_v3.generate_strided_copy_chunks_tile_aware(
          nd_src, nd_dst, nd_int, s_lay, d_lay, int(itemsize)
      )
  )


def partition_variable_bundles(
    num_layers: int,
    dst_layer_shard_bytes: Sequence[int],
    num_bundle_groups: int = 8,
) -> list[VariableBundleSpec]:
  """Partitions variables into contiguous bundles via C++ V3 planner."""
  cpp_bundles = _controller_v3.partition_variable_bundles(
      int(num_layers),
      [int(b) for b in dst_layer_shard_bytes],
      int(num_bundle_groups),
  )
  return [VariableBundleSpec.from_cpp(b) for b in cpp_bundles]
