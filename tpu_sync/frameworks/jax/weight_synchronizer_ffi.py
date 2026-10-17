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

import collections
import dataclasses
import os
import socket
import struct
from typing import Any, Sequence

from absl import logging
import jax
from jax.experimental import compute_on
from jax.experimental import multihost_utils
from jax.experimental import xla_metadata
import jax.numpy as jnp
import numpy as np

from tpu_sync.frameworks.jax import _weight_synchronizer_ffi

ENDPOINTS_API_VERSION: int = 1


class EndpointError(RuntimeError):
  pass


@dataclasses.dataclass(frozen=True)
class ShardEndpoint:
  mesh_index: int
  device: Any
  data_address: str


@dataclasses.dataclass(frozen=True)
class HostEndpoints:
  host_index: int
  control_address: str
  shards: tuple[ShardEndpoint, ...]

  @property
  def devices(self) -> tuple[Any, ...]:
    return tuple(s.device for s in self.shards)

  @property
  def data_addresses(self) -> tuple[str, ...]:
    return tuple(s.data_address for s in self.shards)


@dataclasses.dataclass(frozen=True)
class SyncEndpoints:
  """Grouped per-host control and per-shard data endpoints for a mesh."""

  hosts: tuple[HostEndpoints, ...]

  def single_unit_registration(self) -> tuple[list[str], str]:
    """Returns flat data addresses and comma-joined control addresses."""
    if not self.hosts:
      raise EndpointError("No hosts available.")
    shards_per_host = len(self.hosts[0].shards)
    if shards_per_host == 0:
      raise EndpointError("Host 0 has no shards.")
    shards = []
    for k, host in enumerate(self.hosts):
      if len(host.shards) != shards_per_host:
        raise EndpointError(
            f"Host {k} has {len(host.shards)} shards, expected"
            f" {shards_per_host}."
        )
      for i, s in enumerate(host.shards):
        expected_index = k * shards_per_host + i
        if s.mesh_index != expected_index:
          raise EndpointError(
              f"Host {k} shard {i} has mesh_index {s.mesh_index}, expected"
              f" {expected_index}."
          )
        shards.append(s.data_address)
    control_addresses = ",".join(h.control_address for h in self.hosts)
    return shards, control_addresses


def _decode_ip(words: Sequence[int] | np.ndarray) -> str:
  raw_bytes = struct.pack("<4i", *(int(x) for x in words[:4]))
  if raw_bytes == b"\x00" * 16:
    return ""
  if raw_bytes[:10] == b"\x00" * 10 and raw_bytes[10:12] == b"\xff\xff":
    return socket.inet_ntop(socket.AF_INET, raw_bytes[12:16])
  addr_str = socket.inet_ntop(socket.AF_INET6, raw_bytes)
  return f"[{addr_str}]"


def _default_shard_idx(mesh: jax.sharding.Mesh) -> jax.Array:
  indices = np.arange(mesh.devices.size, dtype=np.int32).reshape(
      mesh.devices.shape
  )
  return jax.device_put(
      indices,
      jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec(*mesh.axis_names)
      ),
  )


def _default_slice_byte_sizes(
    device_arrays: Sequence[jax.Array], mesh: jax.sharding.Mesh
) -> jax.Array:
  sizes = np.array(
      [
          int(np.prod(arr.sharding.shard_shape(arr.shape))) * arr.dtype.itemsize
          for arr in device_arrays
      ],
      dtype=np.int32,
  )
  return jax.device_put(
      sizes, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))
  )


def start_weight_synchronizer(
    device_arrays,
    mesh: jax.sharding.Mesh,
    *,
    d2h: bool = False,  # pylint: disable=redefined-outer-name
    shard_idx: jax.Array | None = None,
    slice_byte_sizes: jax.Array | None = None,
    local_port: int = 0,
    parallelism: int = 1,
    num_layers: int | None = None,
    listener_port: int = 0,
    num_shards: int | None = None,
) -> SyncEndpoints:
  """Initializes WeightSynchronizer across a mesh and returns host endpoints."""
  if isinstance(device_arrays, (list, tuple)):
    anchors_list = list(device_arrays)
  else:
    anchors_list = [device_arrays]
  if not anchors_list:
    raise ValueError("device_arrays must be non-empty")
  if listener_port < 0:
    raise ValueError(f"listener_port cannot be < 0, got {listener_port}")
  if shard_idx is not None and shard_idx.ndim > len(mesh.axis_names):
    raise ValueError(
        "start_weight_synchronizer does not accept pre-packed shard_idx"
    )

  if num_shards is None:
    num_processes = len(set(int(d.process_index) for d in mesh.devices.flat))
    num_shards = mesh.devices.size // num_processes

  host_indices_np, _ = _derive_host_slots(mesh.devices, num_shards)

  if shard_idx is None:
    shard_idx = _default_shard_idx(mesh)
  if slice_byte_sizes is None:
    slice_byte_sizes = _default_slice_byte_sizes(anchors_list, mesh)
  if num_layers is None:
    num_layers = len(anchors_list)

  if d2h:
    ws_info = init_weight_synchronizer_and_d2h(
        anchors_list,
        shard_idx,
        mesh,
        slice_byte_sizes,
        local_port=local_port,
        parallelism=parallelism,
        num_layers=num_layers,
        listener_port=listener_port,
        num_shards=num_shards,
    )
  else:
    ws_info = init_weight_synchronizer(
        anchors_list,
        shard_idx,
        mesh,
        slice_byte_sizes,
        local_port=local_port,
        parallelism=parallelism,
        num_layers=num_layers,
        listener_port=listener_port,
        num_shards=num_shards,
    )

  if isinstance(ws_info, jax.Array) and jax.process_count() > 1:
    gathered = multihost_utils.process_allgather(ws_info, tiled=True)
    rows = np.asarray(gathered)
  elif isinstance(ws_info, jax.Array):
    rows = np.asarray(jax.device_get(ws_info))
  else:
    rows = np.asarray(ws_info)

  if rows.ndim >= 2:
    rows = rows.reshape(-1, rows.shape[-1])
  if rows.ndim != 2 or rows.shape[0] != mesh.devices.size or rows.shape[1] < 10:
    raise EndpointError(
        f"Gathered endpoint rows have shape {rows.shape}, expected at least"
        f" ({mesh.devices.size}, 10)"
    )

  by_host = collections.defaultdict(list)
  for m, row in enumerate(rows):
    data_ip = _decode_ip(row[0:4])
    data_port = int(row[4])
    ctrl_port = int(row[5])
    ctrl_ip = _decode_ip(row[6:10])

    if not data_ip or not ctrl_ip or data_port <= 0 or ctrl_port <= 0:
      host_index = int(host_indices_np.flat[m])
      raise EndpointError(
          f"Host {host_index} mesh_index {m} returned invalid endpoint: data"
          f" {data_ip}:{data_port}, control {ctrl_ip}:{ctrl_port}"
      )

    host_index = int(host_indices_np.flat[m])
    device = mesh.devices.flat[m]
    shard_ep = ShardEndpoint(
        mesh_index=m, device=device, data_address=f"{data_ip}:{data_port}"
    )
    control_address = f"{ctrl_ip}:{ctrl_port}"
    by_host[host_index].append((shard_ep, control_address))

  hosts = []
  for host_idx in sorted(by_host.keys()):
    host_data = by_host[host_idx]
    shards = tuple(sd[0] for sd in host_data)
    control_addresses = set(sd[1] for sd in host_data)
    if len(control_addresses) > 1:
      raise EndpointError(
          f"Host {host_idx} reported conflicting control addresses:"
          f" {control_addresses}"
      )
    hosts.append(
        HostEndpoints(
            host_index=host_idx,
            control_address=control_addresses.pop(),
            shards=shards,
        )
    )

  return SyncEndpoints(hosts=tuple(hosts))


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


def _derive_host_slots(
    devices: np.ndarray, num_shards: int
) -> tuple[np.ndarray, np.ndarray]:
  """Derives per-device (host_idx, local_slot) for the FFI synchronizer layout.

  `host_idx` selects the WeightSynchronizer instance (submanager) a device
  joins and `local_slot` is its buffer slot inside that instance, which must
  have exactly `num_shards` slots. The grouping is defined by one of two
  explicit rules keyed on the process topology of `devices`:

  * Multi-process mesh (more than one distinct `process_index`, e.g. McJAX):
    `host_idx` is the device's `process_index` and `local_slot` is the rank of
    the device among the devices of that process in flat row-major order.
    Every process must own exactly `num_shards` devices.
  * Single-process mesh (all `process_index` equal, e.g. single-controller
    Pathways): synchronizers are contiguous runs of `num_shards` devices in
    flat row-major order, i.e. flat index `f` maps to
    `host_idx = f // num_shards`, `local_slot = f % num_shards`. The device
    count must be a multiple of `num_shards`.

  Args:
    devices: N-D array of devices exposing `process_index` (e.g. mesh.devices).
    num_shards: Number of devices per synchronizer instance.

  Returns:
    A tuple (host_indices, local_slots) of int32 arrays shaped like `devices`.

  Raises:
    ValueError: If `num_shards` is not positive or the device topology does not
      satisfy the rule selected above.
  """
  if num_shards <= 0:
    raise ValueError(f"num_shards must be positive, got {num_shards}")
  process_indices = [int(dev.process_index) for dev in devices.flat]
  num_devices = len(process_indices)
  host_indices = np.empty(num_devices, dtype=np.int32)
  local_slots = np.empty(num_devices, dtype=np.int32)

  if len(set(process_indices)) > 1:
    devices_per_process = collections.Counter(process_indices)
    for process_index, count in sorted(devices_per_process.items()):
      if count != num_shards:
        raise ValueError(
            f"process_index {process_index} owns {count} mesh devices but"
            f" num_shards={num_shards}; every process must own exactly"
            " num_shards devices."
        )
    next_slot_by_process = {p: 0 for p in devices_per_process}
    for flat_idx, process_index in enumerate(process_indices):
      host_indices[flat_idx] = process_index
      local_slots[flat_idx] = next_slot_by_process[process_index]
      next_slot_by_process[process_index] += 1
  else:
    if num_devices % num_shards != 0:
      raise ValueError(
          f"Single-process mesh has {num_devices} devices, which is not a"
          f" multiple of num_shards={num_shards}."
      )
    # Contiguous-run contract: this arithmetic *defines* the documented
    # grouping for single-process meshes; it does not infer a hidden layout.
    for flat_idx in range(num_devices):
      host_indices[flat_idx] = flat_idx // num_shards
      local_slots[flat_idx] = flat_idx % num_shards

  return (
      host_indices.reshape(devices.shape),
      local_slots.reshape(devices.shape),
  )


def _resolve_device_numa_nodes(mesh: jax.sharding.Mesh) -> np.ndarray:
  """Returns the NUMA node (socket) of every mesh device, -1 where unknown.

  The only source is the device's own `numa_node` attribute, which McJAX TPU
  clients publish and Pathways (IFRT proxy) clients do not. Sockets are never
  inferred from device coordinates, slot order, or PCI order. Any further
  source for still-unknown devices belongs in this function.

  Args:
    mesh: JAX device mesh.

  Returns:
    An int32 array shaped like `mesh.devices`.
  """
  numa_nodes = np.full(mesh.devices.shape, -1, dtype=np.int32)
  for idx, device in np.ndenumerate(mesh.devices):
    numa_node = getattr(device, "numa_node", None)
    if (
        isinstance(numa_node, (int, np.integer))
        and not isinstance(numa_node, bool)
        and numa_node >= 0
    ):
      numa_nodes[idx] = numa_node
  logging.info(
      "weight_synchronizer_ffi: %d of %d mesh devices report numa_node; the"
      " rest have an unknown socket (-1).",
      np.count_nonzero(numa_nodes >= 0),
      numa_nodes.size,
  )
  return numa_nodes


def _derive_slot_numa_nodes(
    numa_by_device: np.ndarray,
    host_indices: np.ndarray,
    local_slots: np.ndarray,
    num_shards: int,
) -> np.ndarray:
  """Gives every device the sockets of all local slots of its host.

  Args:
    numa_by_device: NUMA node of each device (-1 if unknown).
    host_indices: Host index of each device, from `_derive_host_slots`.
    local_slots: Local slot of each device, from `_derive_host_slots`.
    num_shards: Number of local slots per host.

  Returns:
    An int32 array of shape `host_indices.shape + (num_shards,)` whose entry
    for a device lists, by local slot, the NUMA node of every device on that
    device's host.

  Raises:
    ValueError: If the three input arrays differ in shape.
  """
  if not numa_by_device.shape == host_indices.shape == local_slots.shape:
    raise ValueError(
        "numa_by_device, host_indices and local_slots must have the same"
        f" shape, got {numa_by_device.shape}, {host_indices.shape} and"
        f" {local_slots.shape}."
    )
  slot_numa_by_host = collections.defaultdict(
      lambda: np.full(num_shards, -1, dtype=np.int32)
  )
  for host_idx, slot, numa_node in zip(
      host_indices.flat, local_slots.flat, numa_by_device.flat
  ):
    slot_numa_by_host[int(host_idx)][int(slot)] = numa_node
  return np.stack(
      [slot_numa_by_host[int(host_idx)] for host_idx in host_indices.flat]
  ).reshape(host_indices.shape + (num_shards,))


def _prepare_shard_info(
    shard_idx: jax.Array,
    mesh: jax.sharding.Mesh,
    num_shards: int,
) -> jax.Array:
  """Packs the per-device shard descriptor for the init FFI calls.

  The descriptor is `[shard_idx, local_slot, host_idx, global_mesh_idx,
  numa_0, ..., numa_{num_shards-1}]`, where `numa_s` is the NUMA node of the
  device in local slot `s` of the device's host, or -1 if unknown (see
  `_resolve_device_numa_nodes`). The synchronizer binds the data NICs of the
  known sockets; with every socket unknown it binds only the host's first
  data NIC.

  If shard_idx is already packed (trailing dimension >= 3), returns shard_idx.
  Otherwise derives local_slot and host_idx from the mesh's process topology
  (see `_derive_host_slots`) and returns a sharded array with shape
  `mesh.devices.shape + (4 + num_shards,)` and
  PartitionSpec(*mesh.axis_names, None).

  `num_shards` is the number of devices per synchronizer; callers needing a
  non-contiguous grouping must pass a pre-packed `shard_idx` (trailing
  dim >= 3), which is already supported.

  Args:
    shard_idx: Shard index array.
    mesh: JAX device mesh.
    num_shards: Number of devices per synchronizer instance.

  Returns:
    A sharded array containing packed shard information.
  """
  if shard_idx.ndim > len(mesh.axis_names) and shard_idx.shape[-1] >= 3:
    return shard_idx

  host_indices_np, local_slots_np = _derive_host_slots(mesh.devices, num_shards)
  slot_numa_nodes_np = _derive_slot_numa_nodes(
      _resolve_device_numa_nodes(mesh),
      host_indices_np,
      local_slots_np,
      num_shards,
  )

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
  slot_numa_nodes = jax.device_put(
      jnp.array(slot_numa_nodes_np, dtype=jnp.int32),
      jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec(*mesh.axis_names, None)
      ),
  )

  if shard_idx.shape != mesh.devices.shape:
    shard_idx = shard_idx.reshape(mesh.devices.shape)

  return jnp.concatenate(
      [
          jnp.stack(
              [shard_idx, local_slots, host_indices, global_mesh_indices],
              axis=-1,
          ),
          slot_numa_nodes,
      ],
      axis=-1,
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
    num_shards: Number of devices per synchronizer instance. If None, calculated
      as mesh devices per process. Device-to-synchronizer grouping follows
      `_prepare_shard_info`; callers needing a non-contiguous grouping must pass
      a pre-packed `shard_idx` (trailing dim >= 3), which is already supported.

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

  shard_info = _prepare_shard_info(shard_idx, mesh, num_shards)
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
    out_dim = 10 if listener_port >= 0 else 5
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
    num_shards: Number of devices per synchronizer instance. If None, calculated
      as mesh devices per process. Device-to-synchronizer grouping follows
      `_prepare_shard_info`; callers needing a non-contiguous grouping must pass
      a pre-packed `shard_idx` (trailing dim >= 3), which is already supported.

  Returns:
    A sharded 1D int32 array containing synchronization metadata (`out_dim=10`
    if `listener_port >= 0`).
  """
  if num_shards is None:
    num_processes = len(set(d.process_index for d in mesh.devices.flatten()))
    num_shards = mesh.devices.size // num_processes

  shard_info = _prepare_shard_info(shard_idx, mesh, num_shards)
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
    out_dim = 10 if listener_port >= 0 else 5
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


def h2d(
    device_array,
    shard_idx: jax.Array | None = None,
    mesh: jax.sharding.Mesh | None = None,
    layer_idx: int = 0,
) -> jax.Array:
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
  if mesh is None:
    mesh = device_array.sharding.mesh
  if shard_idx is None:
    shard_idx = _default_shard_idx(mesh)
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


def multi_h2d(
    device_arrays,
    shard_idx: jax.Array | None = None,
    mesh: jax.sharding.Mesh | None = None,
) -> list[jax.Array]:
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
  if mesh is None:
    mesh = device_arrays[0].sharding.mesh
  if shard_idx is None:
    shard_idx = _default_shard_idx(mesh)
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


def d2h(
    device_array,
    shard_idx: jax.Array | None = None,
    mesh: jax.sharding.Mesh | None = None,
    layer_idx: int = 0,
) -> jax.Array:
  """Executes asynchronous Device-to-Host (D2H) copy from device memory directly onto local staging buffer via FFI.

  Args:
    device_array: Sharded source device array from which data is copied.
    shard_idx: Sharding index array representing shard IDs on each local rank.
    mesh: JAX device mesh across all participating physical devices/hosts.
    layer_idx: Target layer index to copy.

  Returns:
    The same sharded device array (serving as an execution anchor).
  """
  if mesh is None:
    mesh = device_array.sharding.mesh
  if shard_idx is None:
    shard_idx = _default_shard_idx(mesh)
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
