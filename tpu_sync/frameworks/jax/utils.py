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

"""Utilities for JAX."""

import contextlib
import math
from typing import Tuple

from absl import flags
import jax
import jax.numpy as jnp
import numpy as np

# JAX-native open-source profiler imports loaded dynamically on export


flags.DEFINE_string(
    'profiling_log_dir',
    '/tmp/tb_logs',
    'Target directory matching TB profiles logs tracking maps.',
)

P = jax.sharding.PartitionSpec
Mesh = jax.sharding.Mesh
host_sharding = None


def get_local_shape(
    global_shape: Tuple[int, ...], spec: P, mesh: Mesh
) -> Tuple[int, ...]:
  """Calculates the local shard shape for a given global shape and sharding."""
  local_shape = list(global_shape)
  for i, axis_name in enumerate(spec):
    if i < len(local_shape) and axis_name is not None:
      if isinstance(axis_name, str):
        local_shape[i] //= mesh.shape[axis_name]
      elif isinstance(axis_name, tuple):
        for sub_axis in axis_name:
          local_shape[i] //= mesh.shape[sub_axis]
  return tuple(local_shape)


def get_host_sharding(memory_kind='pinned_host'):
  """Returns a sharding for pinned host memory."""
  global host_sharding

  if host_sharding is not None:
    return host_sharding
  mesh = jax.sharding.Mesh(jax.devices(), ('x',))
  host_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec(), memory_kind=memory_kind
  )
  return host_sharding


def _get_reshaped_shape(shape: Tuple[int, ...], threshold: int = 128):
  """Calculates a reshaped shape where the minor dimension is >= threshold."""
  if not shape or shape[-1] >= threshold:
    return shape

  total_elements = 1
  for i in range(len(shape) - 1, -1, -1):
    total_elements *= shape[i]
    if total_elements % threshold == 0:
      prefix = shape[:i]
      if total_elements // threshold > 1:
        return prefix + (total_elements // threshold, threshold)
      else:
        return prefix + (threshold,)
  return None


def _update_sharding_spec(spec, new_rank: int):
  """Adjusts a PartitionSpec to match a new rank."""
  if spec is None:
    return P()
  if len(spec) == new_rank:
    return spec
  if len(spec) > new_rank:
    return P(*spec[:new_rank])
  else:
    return P(*(spec + (None,) * (new_rank - len(spec))))


@contextlib.contextmanager
def xprof_session_manager(dtype):
  """Context manager for XProf session."""
  import os
  log_dir = f"{flags.FLAGS.profiling_log_dir}_{dtype}"
  os.makedirs(log_dir, exist_ok=True)
  jax.profiler.start_trace(log_dir)
  try:
    yield
  finally:
    jax.profiler.stop_trace()
    print("*" * 80)
    print(
        "Hardware trace capture completed successfully! "
        f"Target metrics logged to {log_dir}"
    )
    print(
        "Visualize trace metrics natively by executing: "
        f"tensorboard --logdir={log_dir}"
    )
    print("*" * 80)


@contextlib.contextmanager
def trace_annotation_context(name: str):
  """Platform-agnostic JAX/XProf profiling trace annotations context manager."""
  with jax.profiler.TraceAnnotation(name):
    yield


def create_mesh(axis_shapes, axis_names, explicit_axis: bool = False):
  """Creates a JAX device mesh with the default device order."""
  try:
    num_required_devices = np.prod(axis_shapes)
    devices = np.array(jax.devices())
    if len(devices) < num_required_devices:
      print(
          f'Expected at least {num_required_devices} devices, but only found'
          f' {len(devices)}. This script requires more devices.'
      )
      return None

    device_array = devices[:num_required_devices].reshape(axis_shapes)
    axis_types = (
        tuple([jax.sharding.AxisType.Explicit] * len(axis_shapes))
        if explicit_axis
        else None
    )
    return jax.sharding.Mesh(device_array, axis_names, axis_types=axis_types)
  except RuntimeError:
    print('No TPU devices found. This script must be run on a TPU node.')
    return None


def create_single_layer_kv_cache(
    cache_shape: Tuple[int, ...],
    cache_dtype: jnp.dtype,
    cache_sharding: jax.sharding.NamedSharding,
    init_zeros: bool = False,
) -> jax.Array:
  """Creates a single layer KV cache.

  Args:
    cache_shape: The shape of the cache.
    cache_dtype: The dtype of the cache.
    cache_sharding: The sharding specification for the cache.
    init_zeros: If True, initialize the cache with zeros. Otherwise, use random
      uniform values.

  Returns:
    A sharded jax.Array representing the KV cache.
  """

  def _allocate() -> jax.Array:
    if init_zeros:
      return jnp.zeros(shape=cache_shape, dtype=cache_dtype)
    return jax.random.uniform(
        jax.random.key(1), shape=cache_shape, dtype=cache_dtype
    )

  sharded_allocate = jax.jit(_allocate, out_shardings=cache_sharding)
  return sharded_allocate()


def _get_host_subgrid(
    physical_mesh_shape: list[int], devices_per_host: int
) -> list[int]:
  """Computes host subgrid cuboid shape within the physical TPU mesh."""
  if len(physical_mesh_shape) == 3:
    if (
        devices_per_host == 4
        and physical_mesh_shape[1] % 2 == 0
        and physical_mesh_shape[2] % 2 == 0
    ):
      return [1, 2, 2]
  # General fallback: factor devices_per_host starting from minor dimension
  subgrid = [1] * len(physical_mesh_shape)
  rem = devices_per_host
  for i in range(len(physical_mesh_shape) - 1, -1, -1):
    dim = physical_mesh_shape[i]
    factor = math.gcd(dim, rem)
    subgrid[i] = factor
    rem //= factor
  if rem != 1:
    subgrid = [1] * len(physical_mesh_shape)
    subgrid[-1] = devices_per_host
  return subgrid


def get_shard_sorting_permutation(arr: jax.Array) -> list[int]:
  """Computes the permutation to sort JAX shards for Raiden Controller."""
  if len(arr.addressable_shards) <= 1:
    return []
  sharding = arr.sharding
  if not isinstance(sharding, jax.sharding.NamedSharding):
    return []

  # Inline get_sharding_metadata logic
  shape = arr.shape
  layout = list(range(len(shape) - 1, -1, -1))
  local_shard_shape = sharding.shard_shape(shape)
  logical_mesh_shape = [g // l for g, l in zip(shape, local_shard_shape)]

  mesh = sharding.mesh
  spec = sharding.spec
  major_to_minor = list(reversed(layout))

  # 2. Compute controller's expected global indices
  num_shards = len(arr.addressable_shards)
  num_physical_hosts = jax.process_count()
  replica_id = jax.process_index()

  phys_mesh = [logical_mesh_shape[d] for d in major_to_minor]

  use_spec_mapping = bool(spec is not None and mesh is not None)
  if use_spec_mapping:
    # Use physical mesh mapping (matching raiden_controller.py use_spec_mapping)
    physical_mesh_shape = list(mesh.devices.shape)
    devices_per_host = num_shards
    host_subgrid = _get_host_subgrid(physical_mesh_shape, devices_per_host)
    host_grid = [
        p // s if s > 0 else 1
        for p, s in zip(physical_mesh_shape, host_subgrid)
    ]

    host_coords = []
    temp_h = replica_id
    for size in reversed(host_grid):
      host_coords.append(temp_h % size if size > 0 else 0)
      temp_h //= size if size > 0 else 1
    host_coords.reverse()

    controller_global_indices = []
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

      tensor_coords = []
      tensor_shape = []
      for axis in spec:
        if axis is None:
          tensor_coords.append(0)
          tensor_shape.append(1)
        elif isinstance(axis, str):
          tensor_coords.append(phys_coords[mesh.axis_names.index(axis)])
          tensor_shape.append(mesh.shape[axis])
        else:
          sub_coord = 0
          sub_size = 1
          for ax in axis:
            ax_idx = mesh.axis_names.index(ax)
            sub_coord = (
                sub_coord * mesh.devices.shape[ax_idx] + phys_coords[ax_idx]
            )
            sub_size *= mesh.shape[ax]
          tensor_coords.append(sub_coord)
          tensor_shape.append(sub_size)

      global_idx = 0
      stride = 1
      for val, size in zip(reversed(tensor_coords), reversed(tensor_shape)):
        global_idx += val * stride
        stride *= size
      controller_global_indices.append(global_idx)
  else:
    # Legacy mapping fallback
    host_axis_logical = None
    for d, size in enumerate(logical_mesh_shape):
      if size == num_physical_hosts:
        host_axis_logical = d
        break
    non_host_axes = [
        d for d in range(len(logical_mesh_shape)) if d != host_axis_logical
    ]

    controller_global_indices = []
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
          full_coords[d] = local_coords.get(d, 0)

      tensor_coords = [full_coords[m_axis] for m_axis in major_to_minor]

      global_idx = 0
      stride = 1
      for val, size in zip(reversed(tensor_coords), reversed(phys_mesh)):
        global_idx += val * stride
        stride *= size
      controller_global_indices.append(global_idx)

  # 3. Compute ACTUAL JAX global shard indices
  jax_shard_global_indices = []
  for shard in arr.addressable_shards:
    device = shard.device
    coords = np.argwhere(mesh.devices == device)
    if coords.size == 0:
      raise ValueError(f"Device {device} not found in mesh")
    m_coords = coords[0]
    full_coords = list(m_coords)

    # Map array dimensions to mesh axes using spec
    tensor_coords = []
    tensor_shape = []
    for axis in spec:
      if axis is None:
        tensor_coords.append(0)
        tensor_shape.append(1)
      elif isinstance(axis, str):
        tensor_coords.append(full_coords[mesh.axis_names.index(axis)])
        tensor_shape.append(mesh.shape[axis])
      else:
        sub_coord = 0
        sub_size = 1
        for ax in axis:
          ax_idx = mesh.axis_names.index(ax)
          sub_coord = (
              sub_coord * mesh.devices.shape[ax_idx] + full_coords[ax_idx]
          )
          sub_size *= mesh.shape[ax]
        tensor_coords.append(sub_coord)
        tensor_shape.append(sub_size)

    # Compute flat index (row-major)
    global_idx = 0
    stride = 1
    for val, size in zip(reversed(tensor_coords), reversed(tensor_shape)):
      global_idx += val * stride
      stride *= size
    jax_shard_global_indices.append(global_idx)

  # 4. Construct permutation matching controller slots to addressable shards
  perm = []
  used_indices = set()
  for j, expected_g in enumerate(controller_global_indices):
    occurrence = controller_global_indices[:j].count(expected_g)
    matched_idx = None
    curr_occ = 0
    for idx, actual_g in enumerate(jax_shard_global_indices):
      if actual_g == expected_g:
        if curr_occ == occurrence:
          matched_idx = idx
          break
        curr_occ += 1
    if matched_idx is None or matched_idx in used_indices:
      raise ValueError(
          'Strict bijection failed for shard sorting: cannot map controller'
          f' slot {j} (expected global index {expected_g}) to an unused JAX'
          f' shard. JAX shards: {jax_shard_global_indices}, Controller'
          f' expected: {controller_global_indices}'
      )
    perm.append(matched_idx)
    used_indices.add(matched_idx)

  if len(perm) != len(arr.addressable_shards) or len(used_indices) != len(
      arr.addressable_shards
  ):
    raise ValueError(
        f'Strict bijection failed: perm {perm} does not cover all'
        f' {len(arr.addressable_shards)} addressable shards.'
    )

  if perm == list(range(len(arr.addressable_shards))):
    return []
  return perm
