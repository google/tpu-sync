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

from collections.abc import Sequence
from typing import Any, Dict, List, Optional

import jax
from jax.experimental import compute_on
import jax.numpy as jnp
import numpy as np

from tpu_sync.frameworks.jax import _weight_synchronizer_ffi

__all__ = ["WeightSynchronizer"]


def init_weight_synchronizer(
    device_array,
    shard_idx,
    mesh,
    slice_byte_sizes,  # JAX array of int32
    local_port: int = 0,
    parallelism: int = 1,
    num_layers: int = 1,
    listener_port: int = -1,
    num_shards: int | None = None,
) -> jax.Array:
  """Registers and executes init_weight_synchronizer FFI custom call on each device rank.

  Args:
    device_array: Sharded input device array serving as the FFI target anchor.
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

  Returns:
    A sharded 1D int32 array containing synchronization metadata (`out_dim=6` if
    `listener_port >= 0`).
  """
  if num_shards is None:
    num_processes = len(set(d.process_index for d in mesh.devices.flatten()))
    num_shards = mesh.devices.size // num_processes

  @compute_on.compute_on(
      compute_type="device_host", out_memory_spaces=jax.memory.Space.Device
  )
  def _local_init(anchor, s_idx, sizes):
    axis_names = mesh.axis_names
    out_dim = 6 if listener_port >= 0 else 5
    out_shape = tuple([1] * len(axis_names)) + (out_dim,)
    return jax.ffi.ffi_call(
        "init_weight_synchronizer",
        jax.ShapeDtypeStruct(out_shape, jnp.int32),
        has_side_effect=True,
    )(
        anchor,
        s_idx,
        sizes,
        local_port=np.int32(local_port),
        parallelism=np.int32(parallelism),
        num_layers=np.int32(num_layers),
        listener_port=np.int32(listener_port),
        num_shards=np.int32(num_shards),
    )

  axis_names = mesh.axis_names
  anchor_spec = jax.sharding.PartitionSpec(
      *axis_names, *([None] * (len(device_array.shape) - len(axis_names)))
  )
  index_spec = jax.sharding.PartitionSpec(*axis_names)
  sizes_spec = jax.sharding.PartitionSpec(None)
  out_spec = jax.sharding.PartitionSpec(*axis_names, None)

  return jax.shard_map(
      _local_init,
      mesh=mesh,
      in_specs=(anchor_spec, index_spec, sizes_spec),
      out_specs=out_spec,
  )(device_array, shard_idx, slice_byte_sizes)


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

  Returns:
    A sharded 1D int32 array containing synchronization metadata (`out_dim=6` if
    `listener_port >= 0`).
  """
  if num_shards is None:
    num_processes = len(set(d.process_index for d in mesh.devices.flatten()))
    num_shards = mesh.devices.size // num_processes

  @compute_on.compute_on(
      compute_type="device_host", out_memory_spaces=jax.memory.Space.Device
  )
  def _local_init_and_d2h(s_idx, sizes, *anchors):
    axis_names = mesh.axis_names
    out_dim = 6 if listener_port >= 0 else 5
    out_shape = tuple([1] * len(axis_names)) + (out_dim,)
    return jax.ffi.ffi_call(
        "init_weight_synchronizer_and_d2h",
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

  axis_names = mesh.axis_names
  index_spec = jax.sharding.PartitionSpec(*axis_names)
  sizes_spec = jax.sharding.PartitionSpec(None)
  out_spec = jax.sharding.PartitionSpec(*axis_names, None)

  in_specs = (index_spec, sizes_spec) + tuple(
      arr.sharding.spec for arr in device_arrays
  )

  return jax.shard_map(
      _local_init_and_d2h,
      mesh=mesh,
      in_specs=in_specs,
      out_specs=out_spec,
  )(shard_idx, slice_byte_sizes, *device_arrays)


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

  sharding = device_array.sharding
  local_shape = sharding.shard_shape(device_array.shape)
  dtype = device_array.dtype

  @compute_on.compute_on(
      compute_type="device_host", out_memory_spaces=jax.memory.Space.Device
  )
  def _local_h2d(s_idx):
    return jax.ffi.ffi_call(
        "ws_h2d",
        jax.ShapeDtypeStruct(local_shape, dtype),
        has_side_effect=True,
    )(s_idx, layer_idx=np.int32(layer_idx))

  axis_names = mesh.axis_names
  index_spec = jax.sharding.PartitionSpec(*axis_names)
  out_spec = sharding.spec

  return jax.shard_map(
      _local_h2d,
      mesh=mesh,
      in_specs=(index_spec,),
      out_specs=out_spec,
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
        "ws_multi_h2d",
        out_types,
        has_side_effect=True,
    )(s_idx)

  axis_names = mesh.axis_names
  index_spec = jax.sharding.PartitionSpec(*axis_names)

  return jax.shard_map(
      _local_multi_h2d,
      mesh=mesh,
      in_specs=(index_spec,),
      out_specs=out_specs,
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

  @compute_on.compute_on(
      compute_type="device_host", out_memory_spaces=jax.memory.Space.Device
  )
  def _local_d2h(anchor, s_idx):
    return jax.ffi.ffi_call(
        "ws_d2h",
        jax.ShapeDtypeStruct(anchor.shape, anchor.dtype),
        has_side_effect=True,
    )(anchor, s_idx, layer_idx=np.int32(layer_idx))

  axis_names = mesh.axis_names
  anchor_spec = device_array.sharding.spec
  index_spec = jax.sharding.PartitionSpec(*axis_names)

  return jax.shard_map(
      _local_d2h,
      mesh=mesh,
      in_specs=(anchor_spec, index_spec),
      out_specs=anchor_spec,
  )(device_array, shard_idx)


class WeightSynchronizer:
  """FFI-based distributed Weight Synchronizer for JAX / Pathways."""

  def __init__(
      self,
      jax_arrays: Sequence[Any],
      local_port: Optional[int] = None,
      parallelism: int = 1,
      unsafe_skip_buffer_lock: bool = False,
      listener_port: Optional[int] = None,
      bind_ip: Optional[str] = None,
      auto_h2d: bool = False,
      mesh: Optional[jax.sharding.Mesh] = None,
      num_shards: Optional[int] = None,
  ):
    """Instantiates the FFI-based Weight Synchronizer on a JAX weights list.

    Args:
      jax_arrays: A sequence of JAX arrays representing the sharded model
        weights.
      local_port: Sockets server port for incoming pulls (inference mode).
      parallelism: Number of parallel network stream TCP sockets workers.
      unsafe_skip_buffer_lock: Skip PJRT buffer locks during weights unpack.
      listener_port: Sockets server port for incoming C++ Listener commands.
      bind_ip: Sockets server bind IP address.
      auto_h2d: Automatically execute H2D ingestion upon data arrival.
      mesh: Optional JAX device mesh. If omitted, inferred from jax_arrays.
      num_shards: Number of local shards per host. If omitted, inferred from
        mesh.
    """
    if not jax_arrays:
      raise ValueError("jax_arrays list cannot be empty")
    self._jax_arrays = list(jax_arrays)
    self._parallelism = parallelism
    self._unsafe_skip_buffer_lock = unsafe_skip_buffer_lock
    self._bind_ip = bind_ip
    self._auto_h2d = auto_h2d
    self._destroyed = False

    # 1. Resolve mesh
    if mesh is not None:
      self._mesh = mesh
    else:
      first_sharding = self._jax_arrays[0].sharding
      if hasattr(first_sharding, "mesh") and first_sharding.mesh is not None:
        self._mesh = first_sharding.mesh
      else:
        devices = jax.devices()
        self._mesh = jax.sharding.Mesh(np.array(devices), ("devices",))

    # 2. Compute slice byte sizes
    self._slice_byte_sizes = [
        int(np.prod(arr.sharding.shard_shape(arr.shape)) * arr.dtype.itemsize)
        for arr in self._jax_arrays
    ]
    sizes_sharding = jax.sharding.NamedSharding(
        self._mesh, jax.sharding.PartitionSpec(None)
    )
    self._slice_byte_sizes_sharded = jax.device_put(
        jnp.array(self._slice_byte_sizes, dtype=jnp.int32), sizes_sharding
    )

    # 3. Create shard_idx
    global_ids = jnp.array(
        [d.id for d in self._mesh.devices.flatten()], dtype=jnp.int32
    ).reshape(self._mesh.devices.shape)
    self._shard_idx = jax.device_put(
        global_ids,
        jax.sharding.NamedSharding(
            self._mesh, jax.sharding.PartitionSpec(*self._mesh.axis_names)
        ),
    )

    # 4. Resolve num_shards
    if num_shards is None:
      num_processes = len(
          set(d.process_index for d in self._mesh.devices.flatten())
      )
      if num_processes > 0:
        self._num_shards = self._mesh.devices.size // num_processes
      else:
        self._num_shards = self._mesh.devices.size
    else:
      self._num_shards = num_shards

    # 5. Initialize via FFI
    self._init_info = init_weight_synchronizer(
        device_array=self._jax_arrays[0],
        shard_idx=self._shard_idx,
        mesh=self._mesh,
        slice_byte_sizes=self._slice_byte_sizes_sharded,
        local_port=local_port if local_port is not None else 0,
        parallelism=self._parallelism,
        num_layers=len(self._jax_arrays),
        listener_port=listener_port if listener_port is not None else -1,
        num_shards=self._num_shards,
    )
    self._init_info.block_until_ready()

    try:
      info_np = np.array(self._init_info)
      flat_info = info_np.flatten()
      self._local_port = (
          int(flat_info[4]) if len(flat_info) >= 5 else (local_port or 0)
      )
      self._listener_port = (
          int(flat_info[5]) if len(flat_info) >= 6 else (listener_port or 0)
      )
    except Exception:
      self._local_port = local_port or 0
      self._listener_port = listener_port or 0

  def d2h(self) -> None:
    """Executes asynchronous Device-to-Host (D2H) copy from device memory to local staging buffer."""
    if self._destroyed:
      raise RuntimeError("Cannot invoke d2h on destroyed WeightSynchronizer")
    for layer_idx, arr in enumerate(self._jax_arrays):
      d2h(
          device_array=arr,
          shard_idx=self._shard_idx,
          mesh=self._mesh,
          layer_idx=layer_idx,
      ).block_until_ready()

  def h2d(self) -> None:
    """Executes asynchronous Host-to-Device (H2D) copy from local staging buffer back to device memory."""
    if self._destroyed:
      raise RuntimeError("Cannot invoke h2d on destroyed WeightSynchronizer")
    res = multi_h2d(
        device_arrays=self._jax_arrays,
        shard_idx=self._shard_idx,
        mesh=self._mesh,
    )
    for arr in res:
      arr.block_until_ready()

  def bind_weights(self, jax_arrays: Sequence[Any]) -> None:
    """Binds updated JAX arrays to the weight synchronizer in-place."""
    if not jax_arrays:
      raise ValueError("jax_arrays list cannot be empty")
    self._jax_arrays = list(jax_arrays)

  def test_only_set_skip_tiling(self, skip: bool | Sequence[bool]) -> None:
    """Sets whether D2H/H2D should skip CPU tiling/detiling."""
    pass

  def get_host_buffer(self, layer_idx: int = 0, shard_idx: int = 0) -> Any:
    """Returns a zero-copy Host-side CPU NumPy ndarray view of the staging buffer."""
    raise NotImplementedError(
        "Direct host buffer NumPy mapping is not supported on remote Pathways"
        " FFI."
    )

  def get_local_endpoints(self) -> List[Dict[str, Any]]:
    """Returns the list of transfer endpoints advertised by this instance."""
    return [{
        "endpoint": f"{self._bind_ip or 'localhost'}:{self.local_port}",
        "shards": list(range(self.num_shards)),
    }]

  @property
  def local_port(self) -> Optional[int]:
    """Returns the active local port assigned to the transceiving sockets server."""
    return self._local_port

  @property
  def listener_port(self) -> Optional[int]:
    """Returns the active local port assigned to the C++ Listener."""
    return self._listener_port

  @property
  def is_listener_active(self) -> bool:
    """Returns whether the native C++ Listener is actively running."""
    return is_listener_active()

  @property
  def num_layers(self) -> int:
    """Returns the total number of model weight layers registered."""
    return len(self._jax_arrays)

  @property
  def num_shards(self) -> int:
    """Returns the sharded devices count per layer."""
    return self._num_shards

  @property
  def slice_byte_size(self) -> int:
    """Returns the slice capacity per device block."""
    return self._slice_byte_sizes[0] if self._slice_byte_sizes else 0

  def get_metrics(self) -> Any:
    """Returns a dictionary of internal performance metrics."""
    return {}

  def reset_metrics(self) -> None:
    """Resets all recorded internal metrics."""
    pass

  def destroy(self) -> None:
    """Cleans up and deallocates WeightSynchronizer FFI resources."""
    if not self._destroyed:
      destroy_weight_synchronizer()
      self._destroyed = True

  def close(self) -> None:
    """Alias for destroy()."""
    self.destroy()

  def __del__(self) -> None:
    try:
      self.destroy()
    except Exception:
      pass
