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

"""Framework-neutral helpers for computing logical N-D shard slices."""

import itertools
from typing import List, Tuple

from tpu_sync.rpc import raiden_service_pb2


def compute_nd_shard_slices(
    global_shape: Tuple[int, ...],
    mesh_shape: Tuple[int, ...],
) -> List[raiden_service_pb2.NDSliceProto]:
  """Computes N-dimensional logical tensor bounding boxes for a sharded grid.

  This function derives the exact coordinate intervals along every dimension
  for every logical accelerator shard in canonical row-major order.

  Args:
    global_shape: The global multi-dimensional shape of the tensor.
    mesh_shape: The sharding grid configuration (number of devices per
      dimension).

  Returns:
    A list of NDSliceProto messages containing the multi-dimensional bounding
    box for each logical device shard.
  """
  if len(global_shape) != len(mesh_shape):
    raise ValueError(
        f"Tensor rank ({len(global_shape)}) and sharding mesh rank"
        f" ({len(mesh_shape)}) must match exactly."
    )

  rank = len(global_shape)
  tile_sizes = []
  for d in range(rank):
    if mesh_shape[d] <= 0:
      raise ValueError(f"Mesh shape at dimension {d} must be positive.")
    tile_sizes.append(global_shape[d] // mesh_shape[d])

  # Generate all multi-dimensional device coordinates in row-major sequence.
  coordinate_ranges = [range(mesh_shape[d]) for d in range(rank)]

  shard_slices = []
  for device_coord in itertools.product(*coordinate_ranges):
    slice_proto = raiden_service_pb2.NDSliceProto()
    for d in range(rank):
      c = device_coord[d]
      start = c * tile_sizes[d]
      # Put any remainder in the last physical mesh shard.
      end = (
          (c + 1) * tile_sizes[d] if c < mesh_shape[d] - 1 else global_shape[d]
      )
      slice_proto.dimensions.add(start=start, end=end)
    shard_slices.append(slice_proto)

  return shard_slices


def format_nd_slice(slice_proto: raiden_service_pb2.NDSliceProto) -> str:
  """Formats an NDSliceProto into a compact single-line slice string e.g.

  [0:512, 0:2048].
  """
  dims = ", ".join(f"{d.start}:{d.end}" for d in slice_proto.dimensions)
  return f"[{dims}]"


def format_nd_slices(slices: List[raiden_service_pb2.NDSliceProto]) -> str:
  """Formats a list of NDSliceProto into a compact single-line string representation.

  Examples:
    1D: [[0:512], [512:1024], [1024:1536], [1536:2048]]
    2D: [[0:1536, 0:2048], [1536:3072, 0:2048], [3072:4608, 0:2048], [4608:6144,
    0:2048]]
  """
  return f"[{', '.join(format_nd_slice(s) for s in slices)}]"
