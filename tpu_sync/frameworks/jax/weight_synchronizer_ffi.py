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

"""JAX bindings for WeightSynchronizer FFI, enabling host/device weight synchronization."""

import math
import os

import jax
from jax.experimental import compute_on
from jax.experimental import xla_metadata
import jax.numpy as jnp
import numpy as np

from tpu_sync.frameworks.jax import _weight_synchronizer_ffi
from tpu_sync.frameworks.jax import utils

_DIRECT_DEVICE_BUFFER_ENV = "RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER"

_orig_compute_on = compute_on.compute_on


def _compat_compute_on(f=None, *, compute_type="device_host", **kwargs):
  """Compatibility wrapper around compute_on.compute_on for Google3 and OSS JAX."""
  try:
    if f is not None:
      return _orig_compute_on(f, compute_type=compute_type, **kwargs)
    return _orig_compute_on(compute_type=compute_type, **kwargs)
  except TypeError:
    cm = _orig_compute_on(compute_type=compute_type)
    return cm(f) if f is not None else cm


compute_on.compute_on = _compat_compute_on


def _use_direct_device_buffer() -> bool:
  """Returns whether zero-copy direct device buffer mode is enabled."""
  val = os.environ.get(_DIRECT_DEVICE_BUFFER_ENV, "0").strip().lower()
  return val in ("1", "true", "yes", "on")


def _prepare_shard_info(
    shard_idx: jax.Array,
    mesh: jax.sharding.Mesh,
    num_shards: int,
    host_subgrid: list[int] | None = None,
) -> jax.Array:
  """Packs [shard_idx, local_slot, host_idx] for FFI custom call.

  If shard_idx is already packed (trailing dimension >= 3), returns shard_idx.
  Otherwise, computes local_slot and host_idx based on the mesh physical layout
  and host subgrid decomposition, and returns a sharded array with shape
  `mesh.devices.shape + (3,)` and PartitionSpec(*mesh.axis_names, None).

  Args:
    shard_idx: Shard index array.
    mesh: JAX device mesh.
    num_shards: Number of local shards (devices per host).
    host_subgrid: Optional host subgrid shape for shard decomposition.

  Returns:
    A sharded array containing packed shard information.
  """
  if shard_idx.ndim > len(mesh.axis_names) and shard_idx.shape[-1] >= 3:
    return shard_idx

  physical_mesh_shape = list(mesh.devices.shape)
  devices_per_host = num_shards
  local_subgrid = None
  try:
    if (
        hasattr(mesh, "local_mesh")
        and mesh.local_mesh is not None
        and hasattr(mesh.local_mesh, "devices")
    ):
      local_subgrid = list(mesh.local_mesh.devices.shape)
  except (AttributeError, ValueError, TypeError):
    local_subgrid = None

  if (
      host_subgrid is not None
      and len(host_subgrid) == len(physical_mesh_shape)
      and math.prod(host_subgrid) == devices_per_host
      and all(p % s == 0 for p, s in zip(physical_mesh_shape, host_subgrid))
  ):
    subgrid = list(host_subgrid)
    grid = [p // s for p, s in zip(physical_mesh_shape, subgrid)]
  elif (
      local_subgrid is not None
      and len(local_subgrid) == len(physical_mesh_shape)
      and math.prod(local_subgrid) == devices_per_host
      and all(p % s == 0 for p, s in zip(physical_mesh_shape, local_subgrid))
  ):
    subgrid = local_subgrid
    grid = [p // s for p, s in zip(physical_mesh_shape, subgrid)]
  else:
    subgrid, grid = utils.compute_host_subgrid(
        physical_mesh_shape, devices_per_host
    )
  host_subgrid, host_grid = subgrid, grid

  local_slots_np = np.zeros(mesh.devices.shape, dtype=np.int32)
  host_indices_np = np.zeros(mesh.devices.shape, dtype=np.int32)

  for coord in np.ndindex(*mesh.devices.shape):
    h_idx = 0
    for c, s, g in zip(coord, host_subgrid, host_grid):
      h = c // s if s > 0 else 0
      h_idx = h_idx * g + h
    l_slot = 0
    for c, s in zip(coord, host_subgrid):
      l = c % s if s > 0 else 0
      l_slot = l_slot * s + l
    local_slots_np[coord] = l_slot
    host_indices_np[coord] = h_idx

  spec = jax.sharding.PartitionSpec(*mesh.axis_names)
  sharding = jax.sharding.NamedSharding(mesh, spec)

  local_slots = jax.device_put(
      jnp.array(local_slots_np, dtype=jnp.int32), sharding
  )
  host_indices = jax.device_put(
      jnp.array(host_indices_np, dtype=jnp.int32), sharding
  )

  global_mesh_indices_np = np.arange(mesh.devices.size, dtype=np.int32).reshape(
      mesh.devices.shape
  )
  global_mesh_indices = jax.device_put(
      jnp.array(global_mesh_indices_np, dtype=jnp.int32), sharding
  )

  if shard_idx.shape != tuple(physical_mesh_shape):
    shard_idx = shard_idx.reshape(tuple(physical_mesh_shape))

  return jnp.stack(
      [shard_idx, local_slots, host_indices, global_mesh_indices], axis=-1
  )


def init_weight_synchronizer(
    device_arrays,  # List of sharded arrays
    shard_idx,
    mesh,
    slice_byte_sizes,  # JAX array of int32
    local_port: int = 0,
    parallelism: int = 1,
    num_layers: int = 1,
    listener_port: int = -1,
    num_shards: int | None = None,
    host_subgrid: list[int] | None = None,
) -> jax.Array:
  """Registers and executes init_weight_synchronizer FFI custom call on each device rank.

  Args:
    device_arrays: List of sharded input device arrays (or a single sharded
      device array) serving as the FFI target anchor(s).
    shard_idx: Sharding index array representing shard IDs on each local rank.
    mesh: JAX device mesh across all participating physical devices/hosts.
    slice_byte_sizes: Sharded 1D int32 array of physical weight slice sizes per
      layer.
    local_port: Port number for the local sender transport server (`0` for
      auto-assign).
    parallelism: Number of parallel TCP/IPC streams to use for memory transfer.
    num_layers: Total number of network layers or synchronization iterations.
    listener_port: Optional pre-assigned listener port to share across ranks
      (`-1` to initialize new).
    num_shards: Number of local shards (devices per host). If None, calculated
      from mesh.
    host_subgrid: Optional host subgrid shape for shard decomposition.

  Returns:
    A sharded 1D int32 array containing synchronization metadata (`out_dim=6` if
    `listener_port >= 0`).
  """
  if isinstance(device_arrays, (list, tuple)):
    anchors_list = list(device_arrays)
  else:
    anchors_list = [device_arrays]

  if num_shards is None:
    num_processes = len(set(d.process_index for d in mesh.devices.flatten()))
    num_shards = mesh.devices.size // num_processes

  shard_info = _prepare_shard_info(
      shard_idx, mesh, num_shards, host_subgrid=host_subgrid
  )
  use_direct = _use_direct_device_buffer()
  ffi_name = (
      "raiden_weight_synchronizer_create"
      if use_direct
      else "init_weight_synchronizer"
  )

  @compute_on.compute_on(
      compute_type="device_host", out_memory_spaces=jax.memory.Space.Device
  )
  def _local_init(s_idx, sizes, *anchors):
    axis_names = mesh.axis_names
    out_dim = 6 if listener_port >= 0 else 5
    out_shape = tuple([1] * len(axis_names)) + (out_dim,)
    ffi_args = (
        (anchors[0], s_idx, sizes, *anchors[1:])
        if use_direct
        else (anchors[0], s_idx, sizes)
    )
    return jax.ffi.ffi_call(
        ffi_name,
        jax.ShapeDtypeStruct(out_shape, jnp.int32),
        has_side_effect=True,
    )(
        *ffi_args,
        local_port=np.int32(local_port),
        parallelism=np.int32(parallelism),
        num_layers=np.int32(num_layers),
        listener_port=np.int32(listener_port),
        num_shards=np.int32(num_shards),
    )

  def _shard_init(s_idx, sizes, *anchors):
    if use_direct:
      new_anchors = []
      for a in anchors:
        a = a.reshape(-1).view(jnp.uint32)
        one = jax.lax.optimization_barrier(jnp.ones_like(a))
        with xla_metadata.set_xla_metadata(_xla_device_buffer="true"):
          a = jax.lax.mul(a, one)
        new_anchors.append(a)
      anchors = tuple(new_anchors)
    return _local_init(s_idx, sizes, *anchors)

  axis_names = mesh.axis_names
  anchor_specs = tuple(a.sharding.spec for a in anchors_list)
  index_spec = jax.sharding.PartitionSpec(*axis_names, None)
  sizes_spec = jax.sharding.PartitionSpec(None)
  out_spec = jax.sharding.PartitionSpec(*axis_names, None)

  return jax.jit(
      jax.shard_map(
          _shard_init,
          mesh=mesh,
          in_specs=(index_spec, sizes_spec) + anchor_specs,
          out_specs=out_spec,
      )
  )(shard_info, slice_byte_sizes, *anchors_list)


def init_weight_synchronizer_and_d2h(
    device_arrays,  # List of sharded arrays
    shard_idx,
    mesh,
    slice_byte_sizes,  # JAX array of int32
    local_port: int = 0,
    parallelism: int = 1,
    num_layers: int = 1,
    listener_port: int = -1,
    num_shards: int | None = None,
    host_subgrid: list[int] | None = None,
) -> jax.Array:
  """Registers and executes init_weight_synchronizer_and_d2h FFI custom call on each device rank.

  Args:
    device_arrays: List of sharded input device arrays.
    shard_idx: Sharding index array representing shard IDs on each local rank.
    mesh: JAX device mesh across all participating physical devices/hosts.
    slice_byte_sizes: Sharded 1D int32 array of physical weight slice sizes per
      layer.
    local_port: Port number for the local sender transport server (`0` for
      auto-assign).
    parallelism: Number of parallel TCP/IPC streams to use for memory transfer.
    num_layers: Total number of network layers or synchronization iterations.
    listener_port: Optional pre-assigned listener port to share across ranks
      (`-1` to initialize new).
    num_shards: Number of local shards (devices per host). If None, calculated
      from mesh.
    host_subgrid: Optional host subgrid shape for shard decomposition.

  Returns:
    A sharded 1D int32 array containing synchronization metadata (`out_dim=6` if
    `listener_port >= 0`).
  """
  if num_shards is None:
    num_processes = len(set(d.process_index for d in mesh.devices.flatten()))
    num_shards = mesh.devices.size // num_processes

  shard_info = _prepare_shard_info(
      shard_idx, mesh, num_shards, host_subgrid=host_subgrid
  )
  ffi_name = (
      "raiden_weight_synchronizer_create_and_d2h"
      if _use_direct_device_buffer()
      else "init_weight_synchronizer_and_d2h"
  )

  @compute_on.compute_on(
      compute_type="device_host", out_memory_spaces=jax.memory.Space.Device
  )
  def _local_init_and_d2h(s_idx, sizes, *anchors):
    axis_names = mesh.axis_names
    out_dim = 6 if listener_port >= 0 else 5
    out_shape = tuple([1] * len(axis_names)) + (out_dim,)
    return jax.ffi.ffi_call(
        ffi_name,
        jax.ShapeDtypeStruct(out_shape, jnp.int32),
        has_side_effect=True,
    )(
        s_idx,
        sizes,
        *anchors,
        local_port=np.int32(local_port),
        parallelism=np.int32(parallelism),
        num_layers=np.int32(num_layers),
        listener_port=np.int32(listener_port),
        num_shards=np.int32(num_shards),
    )

  def _shard_init_and_d2h(s_idx, sizes, *anchors):
    if _use_direct_device_buffer():
      tagged_anchors = []
      for a in anchors:
        a = a.reshape(-1).view(jnp.uint32)
        one = jax.lax.optimization_barrier(jnp.ones_like(a))
        with xla_metadata.set_xla_metadata(_xla_device_buffer="true"):
          a = jax.lax.mul(a, one)
        tagged_anchors.append(a)
      anchors = tagged_anchors
    return _local_init_and_d2h(s_idx, sizes, *anchors)

  axis_names = mesh.axis_names
  index_spec = jax.sharding.PartitionSpec(*axis_names, None)
  sizes_spec = jax.sharding.PartitionSpec(None)
  out_spec = jax.sharding.PartitionSpec(*axis_names, None)

  in_specs = (index_spec, sizes_spec) + tuple(
      arr.sharding.spec for arr in device_arrays
  )

  return jax.jit(
      jax.shard_map(
          _shard_init_and_d2h,
          mesh=mesh,
          in_specs=in_specs,
          out_specs=out_spec,
      )
  )(shard_info, slice_byte_sizes, *device_arrays)


def prepare_extended_info(
    gathered_info, device_ids, r_starts, r_ends, c_starts, c_ends
):
  """Packs metadata for coordination."""
  return _weight_synchronizer_ffi.prepare_extended_info(
      gathered_info, device_ids, r_starts, r_ends, c_starts, c_ends
  )


def destroy_weight_synchronizer():
  """Cleans up WeightSynchronizer instances."""
  _weight_synchronizer_ffi.destroy_weight_synchronizer()


def is_listener_active(shard_idx: int = 0) -> bool:
  """Returns whether the native C++ listener for the shard is active."""
  return _weight_synchronizer_ffi.is_listener_active(shard_idx)


def h2d(device_array, shard_idx, mesh, layer_idx: int = 0) -> jax.Array:
  """Executes asynchronous Host-to-Device (H2D) copy from local staging buffer directly onto device memory via FFI.

  Args:
    device_array: Sharded destination device array onto which host memory is
      copied.
    shard_idx: Sharding index array representing shard IDs on each local rank.
    mesh: JAX device mesh across all participating physical devices/hosts.
    layer_idx: Target layer index to copy.

  Returns:
    The updated sharded device array with data copied from the local host
    buffer.
  """
  ffi_name = (
      "raiden_weight_synchronizer_h2d"
      if _use_direct_device_buffer()
      else "ws_h2d"
  )
  sharding = device_array.sharding
  local_shape = sharding.shard_shape(device_array.shape)
  dtype = device_array.dtype

  @compute_on.compute_on(
      compute_type="device_host", out_memory_spaces=jax.memory.Space.Device
  )
  def _local_h2d(s_idx):
    return jax.ffi.ffi_call(
        ffi_name,
        jax.ShapeDtypeStruct(local_shape, dtype),
        has_side_effect=True,
    )(s_idx, layer_idx=np.int32(layer_idx))

  axis_names = mesh.axis_names
  index_spec = (
      jax.sharding.PartitionSpec(*axis_names, None)
      if shard_idx.ndim > len(axis_names)
      else jax.sharding.PartitionSpec(*axis_names)
  )
  out_spec = sharding.spec

  return jax.jit(
      jax.shard_map(
          _local_h2d,
          mesh=mesh,
          in_specs=(index_spec,),
          out_specs=out_spec,
      )
  )(shard_idx)


def multi_h2d(device_arrays, shard_idx, mesh) -> list[jax.Array]:
  """Executes asynchronous Host-to-Device (H2D) copy from local staging buffers directly onto device memory via FFI.

  Args:
    device_arrays: List of sharded destination device arrays onto which host
      memory is copied.
    shard_idx: Sharding index array representing shard IDs on each local rank.
    mesh: JAX device mesh across all participating physical devices/hosts.

  Returns:
    A list of updated sharded device arrays with data copied from the local host
    buffers.
  """
  ffi_name = (
      "raiden_weight_synchronizer_multi_h2d"
      if _use_direct_device_buffer()
      else "ws_multi_h2d"
  )
  out_types = []
  out_specs = []
  for arr in device_arrays:
    sharding = arr.sharding
    local_shape = sharding.shard_shape(arr.shape)
    out_types.append(jax.ShapeDtypeStruct(local_shape, arr.dtype))
    out_specs.append(sharding.spec)

  out_types = tuple(out_types)
  out_specs = tuple(out_specs)

  @compute_on.compute_on(
      compute_type="device_host", out_memory_spaces=jax.memory.Space.Device
  )
  def _local_multi_h2d(s_idx):
    return jax.ffi.ffi_call(
        ffi_name,
        out_types,
        has_side_effect=True,
    )(s_idx)

  axis_names = mesh.axis_names
  index_spec = (
      jax.sharding.PartitionSpec(*axis_names, None)
      if shard_idx.ndim > len(axis_names)
      else jax.sharding.PartitionSpec(*axis_names)
  )

  return jax.jit(
      jax.shard_map(
          _local_multi_h2d,
          mesh=mesh,
          in_specs=(index_spec,),
          out_specs=out_specs,
      )
  )(shard_idx)


def d2h(device_array, shard_idx, mesh, layer_idx: int = 0) -> jax.Array:
  """Executes asynchronous Device-to-Host (D2H) copy from device memory directly onto local staging buffer via FFI.

  Args:
    device_array: Sharded source device array from which data is copied.
    shard_idx: Sharding index array representing shard IDs on each local rank.
    mesh: JAX device mesh across all participating physical devices/hosts.
    layer_idx: Target layer index to copy.

  Returns:
    The same sharded device array (serving as an execution anchor).
  """
  ffi_name = (
      "raiden_weight_synchronizer_d2h"
      if _use_direct_device_buffer()
      else "ws_d2h"
  )

  @compute_on.compute_on(
      compute_type="device_host", out_memory_spaces=jax.memory.Space.Device
  )
  def _local_d2h(anchor, s_idx):
    return jax.ffi.ffi_call(
        ffi_name,
        jax.ShapeDtypeStruct(anchor.shape, anchor.dtype),
        has_side_effect=True,
    )(anchor, s_idx, layer_idx=np.int32(layer_idx))

  def _shard_d2h(anchor, s_idx):
    orig_shape, orig_dtype = anchor.shape, anchor.dtype
    if _use_direct_device_buffer():
      anchor = anchor.reshape(-1).view(jnp.uint32)
      one = jax.lax.optimization_barrier(jnp.ones_like(anchor))
      with xla_metadata.set_xla_metadata(_xla_device_buffer="true"):
        anchor = jax.lax.mul(anchor, one)
    return _local_d2h(anchor, s_idx).view(orig_dtype).reshape(orig_shape)

  axis_names = mesh.axis_names
  anchor_spec = device_array.sharding.spec
  index_spec = (
      jax.sharding.PartitionSpec(*axis_names, None)
      if shard_idx.ndim > len(axis_names)
      else jax.sharding.PartitionSpec(*axis_names)
  )

  return jax.jit(
      jax.shard_map(
          _shard_d2h,
          mesh=mesh,
          in_specs=(anchor_spec, index_spec),
          out_specs=anchor_spec,
      )
  )(device_array, shard_idx)
