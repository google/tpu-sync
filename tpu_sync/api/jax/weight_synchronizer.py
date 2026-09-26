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

"""High-performance JAX Weight Synchronizer for RL Trainer-Inference Pipelines."""

import functools
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Union

import jax

from tpu_sync.api import common
# Import Nanobind binary library directly E2E!
from tpu_sync.frameworks.jax import _tpu_raiden_jax as _weight_synchronizer

common.register_telemetry_callbacks(
    increment_counter=_weight_synchronizer.increment_counter,
    set_gauge=_weight_synchronizer.set_gauge,
    observe_histogram=_weight_synchronizer.observe_histogram,
)

configure_telemetry = _weight_synchronizer.configure_telemetry
get_and_reset_metric_samples = _weight_synchronizer.get_and_reset_metric_samples
get_raiden_metrics_prometheus_text = (
    _weight_synchronizer.get_raiden_metrics_prometheus_text
)


def find_dp_subshard_split_dim(
    local_shape: Sequence[int],
    k: int,
    tile_rows: int = 8,
) -> Optional[int]:
  """Returns the dimension `d` to split a local shard into `k` contiguous sub-shards."""
  if k <= 1 or not local_shape:
    return None
  rank = len(local_shape)
  max_dim = 0 if rank == 1 else rank - 2
  for d in range(max_dim + 1):
    dim_len = int(local_shape[d])
    if dim_len == 1:
      continue
    if dim_len % k != 0:
      return None
    sub_dim = dim_len // k
    if (
        rank >= 2
        and d == rank - 2
        and tile_rows > 1
        and sub_dim % tile_rows != 0
    ):
      return None
    return d
  return None


def _resolve_enable_ici_all_gather(
    ici_all_gather: Optional[bool],
) -> bool:
  """Resolves `ici_all_gather` from explicit bool or environment variables."""
  if ici_all_gather is not None:
    return bool(ici_all_gather)
  for env_name in (
      "RAIDEN_ENABLE_DP_SUBSHARDING",
      "RAIDEN_ENABLE_ICI_ALL_GATHER",
  ):
    val = os.environ.get(env_name, "").strip().lower()
    if val in ("1", "true", "yes"):
      return True
    if val in ("0", "false", "no"):
      return False
  return False


def _get_replicated_mesh_axes(
    mesh: jax.sharding.Mesh,
    spec: jax.sharding.PartitionSpec,
) -> tuple[str, ...]:
  """Returns mesh axes with size > 1 that are not referenced in `spec`."""
  used_axes = set()
  for entry in spec:
    if entry is None:
      continue
    if isinstance(entry, (tuple, list)):
      for sub in entry:
        if sub:
          used_axes.update(s.strip() for s in str(sub).split(",") if s.strip())
    else:
      used_axes.update(s.strip() for s in str(entry).split(",") if s.strip())
  return tuple(
      ax for ax in mesh.axis_names if ax not in used_axes and mesh.shape[ax] > 1
  )


@functools.lru_cache(maxsize=64)
def _get_compiled_shard_all_gather(
    mesh: jax.sharding.Mesh,
    specs: tuple[jax.sharding.PartitionSpec, ...],
    split_meta: tuple[tuple[int, int, Any], ...],
):
  """Returns a cached JIT-compiled shard_map collective for `split_meta`."""

  def _shard_all_gather(*xs):
    outs = []
    for x, (d_split, sub_dim, axis_arg) in zip(xs, split_meta):
      sub_x = jax.lax.slice_in_dim(x, 0, sub_dim, axis=d_split)
      outs.append(
          jax.lax.all_gather(
              sub_x, axis_name=axis_arg, axis=d_split, tiled=True
          )
      )
    return tuple(outs)

  return jax.jit(
      jax.shard_map(
          _shard_all_gather,
          mesh=mesh,
          in_specs=specs,
          out_specs=specs,
          check_vma=False,
      )
  )


def ici_all_gather_weights(
    jax_arrays: Sequence[jax.Array],
    tile_rows: int = 8,
) -> List[jax.Array]:
  """Reconstructs full shards from DP sub-shards using direct ICI all_gather."""
  if not jax_arrays:
    return []
  result = list(jax_arrays)

  # Group arrays by mesh so all layers on the same mesh run in one compiled
  # shard_map collective.
  indices_by_mesh: dict[Any, list[tuple[int, int, int, Any]]] = {}
  for idx, arr in enumerate(result):
    sharding = getattr(arr, "sharding", None)
    mesh = getattr(sharding, "mesh", None)
    spec = getattr(sharding, "spec", None)
    if mesh is None or spec is None:
      continue
    replicated_axes = _get_replicated_mesh_axes(mesh, spec)
    if not replicated_axes:
      continue
    k = math.prod(mesh.shape[ax] for ax in replicated_axes)
    if k <= 1:
      continue
    local_shape = sharding.shard_shape(arr.shape)
    d_split = find_dp_subshard_split_dim(local_shape, k, tile_rows=tile_rows)
    if d_split is None:
      continue
    sub_dim = local_shape[d_split] // k
    axis_arg = (
        replicated_axes[0] if len(replicated_axes) == 1 else replicated_axes
    )
    indices_by_mesh.setdefault(mesh, []).append(
        (idx, d_split, sub_dim, axis_arg)
    )

  for mesh, items in indices_by_mesh.items():
    group_arrays = tuple(result[item[0]] for item in items)
    specs = tuple(arr.sharding.spec for arr in group_arrays)
    split_meta = tuple((item[1], item[2], item[3]) for item in items)
    compiled_fn = _get_compiled_shard_all_gather(mesh, specs, split_meta)
    gathered = compiled_fn(*group_arrays)
    for (idx, _, _, _), g_arr in zip(items, gathered):
      result[idx] = g_arr

  return result


class WeightSynchronizer:
  """Zero-copy distributed Weight Synchronizer for JAX."""

  def __init__(
      self,
      jax_arrays: List[any],
      local_port: Optional[int] = None,
      parallelism: int = 1,
      unsafe_skip_buffer_lock: bool = False,
      listener_port: Optional[int] = None,
      bind_ip: Optional[str] = None,
      auto_h2d: bool = False,
      global_shard_indices: Optional[List[int]] = None,
      ring_buffer_size: Optional[int] = None,
  ):
    """Instantiates the Weight Synchronizer on a JAX weights list.

    Args:
      jax_arrays: A list of JAX arrays representing the sharded model weights.
      local_port: Sockets server port for incoming pulls (inference mode).
      parallelism: Number of parallel network stream TCP sockets workers.
      unsafe_skip_buffer_lock: Skip PJRT buffer locks during weights unpack.
      listener_port: Sockets server port for incoming C++ Listener commands.
      bind_ip: Sockets server bind IP address.
      auto_h2d: Automatically execute H2D ingestion upon data arrival.
      global_shard_indices: Explicit vector of global shard indices.
      ring_buffer_size: Optional bounded number of host staging buffers per
        shard (`0` or `None` uses all layers unless
        `RAIDEN_WEIGHT_SYNC_RING_BUFFER_SIZE` is set).
    """
    self._jax_arrays = list(jax_arrays) if jax_arrays is not None else []
    self._unsafe_skip_buffer_lock = unsafe_skip_buffer_lock
    self._has_explicit_global_shard_indices = global_shard_indices is not None
    if global_shard_indices is None:
      if jax_arrays and hasattr(jax_arrays[0], "addressable_shards"):
        arr = jax_arrays[0]
        if (
            hasattr(arr, "sharding")
            and getattr(arr.sharding, "mesh", None) is not None
        ):
          mesh = arr.sharding.mesh
          flat_devices = list(mesh.devices.flat)
          indices = []
          for s in arr.addressable_shards:
            try:
              indices.append(flat_devices.index(s.device))
            except (ValueError, AttributeError):
              pass
          if len(indices) == len(arr.addressable_shards):
            global_shard_indices = indices

    if global_shard_indices is None:
      num_shards = (
          len(jax_arrays[0].addressable_shards)
          if jax_arrays and hasattr(jax_arrays[0], "addressable_shards")
          else len(jax.local_devices())
      )
      offset = (
          jax.process_index() * len(jax.local_devices())
          if jax.process_count() > 1
          else 0
      )
      global_shard_indices = [offset + i for i in range(num_shards)]

    self._global_shard_indices = (
        list(global_shard_indices) if global_shard_indices is not None else []
    )

    self._impl = _weight_synchronizer.WeightSynchronizer(
        jax_arrays,
        local_port,
        parallelism,
        unsafe_skip_buffer_lock,
        listener_port,
        bind_ip,
        auto_h2d,
        global_shard_indices,
        ring_buffer_size,
    )

  @classmethod
  def test_only_create_cpu_instance(
      cls,
      num_layers: int,
      num_shards: int,
      slice_byte_size: Union[int, List[int]],
      local_port: Optional[int] = None,
      parallelism: int = 1,
      listener_port: Optional[int] = None,
      bind_ip: Optional[str] = "127.0.0.1",
      auto_h2d: bool = False,
      global_shard_indices: Optional[List[int]] = None,
      test_only_simulated_egress_gbps: float = 0.0,
      test_only_simulated_ingress_gbps: float = 0.0,
      ring_buffer_size: Optional[int] = None,
  ) -> "WeightSynchronizer":
    """Instantiates a CPU-only WeightSynchronizer allocating host DRAM without TPU devices."""
    instance = cls.__new__(cls)
    instance._impl = (
        _weight_synchronizer.WeightSynchronizer.test_only_create_cpu_instance(
            num_layers,
            num_shards,
            slice_byte_size,
            local_port,
            parallelism,
            listener_port,
            bind_ip,
            auto_h2d,
            global_shard_indices,
            test_only_simulated_egress_gbps,
            test_only_simulated_ingress_gbps,
            ring_buffer_size,
        )
    )
    instance._jax_arrays = []
    instance._unsafe_skip_buffer_lock = False
    instance._global_shard_indices = list(global_shard_indices or [])
    instance._has_explicit_global_shard_indices = (
        global_shard_indices is not None
    )
    return instance

  def test_only_set_bandwidth_limit(
      self,
      test_only_simulated_egress_gbps: float = 0.0,
      test_only_simulated_ingress_gbps: float = 0.0,
  ) -> None:
    """Sets the simulated transport bandwidth limit in Gbps (for testing only)."""
    self._impl.test_only_set_bandwidth_limit(
        test_only_simulated_egress_gbps,
        test_only_simulated_ingress_gbps,
    )

  def d2h(self) -> None:
    """Triggers asynchronous Device-to-Host (D2H) copy of current weights to Host buffer."""
    self._impl.D2h()

  def h2d(
      self, ici_all_gather: Optional[bool] = None
  ) -> Optional[List[jax.Array]]:
    """Triggers Host-to-Device (H2D) copy and optional post-H2D ICI all-gather."""
    self._impl.H2d()
    if _resolve_enable_ici_all_gather(ici_all_gather) and self._jax_arrays:
      return self.ici_all_gather()
    return None

  def ici_all_gather(
      self,
      jax_arrays: Optional[Sequence[jax.Array]] = None,
      rebind: bool = False,
  ) -> List[jax.Array]:
    """Runs direct ICI all-gather across replicated mesh axes on sub-sharded weights."""
    target_arrays = (
        list(jax_arrays) if jax_arrays is not None else self._jax_arrays
    )
    updated = ici_all_gather_weights(target_arrays)
    if rebind and updated:
      self.bind_weights(updated)
    return updated

  def wait_for_transfer_completion(self, uuid: Optional[int] = None) -> None:
    """Blocks until the transfer with the given UUID (or any transfer if None) has finished ingestion."""
    self._impl.wait_for_transfer_completion(uuid)

  def test_only_set_skip_tiling(self, skip: bool | List[bool]) -> None:
    """Sets whether D2H/H2D should skip CPU tiling/detiling (for testing only)."""
    if isinstance(skip, bool):
      self._impl.set_skip_tiling(skip)
    else:
      self._impl.set_skip_tiling(list(skip))

  def bind_weights(self, jax_arrays: List[any]) -> None:
    """Binds the JAX arrays to the weight synchronizer in-place.

    Args:
      jax_arrays: A list of JAX arrays representing the updated sharded model
        weights.
    """
    self._jax_arrays = list(jax_arrays) if jax_arrays is not None else []
    self._impl.bind_weights(jax_arrays)

  def get_host_buffer(self, layer_idx: int = 0, shard_idx: int = 0) -> any:
    """Returns a zero-copy Host-side CPU NumPy ndarray view of the C++ staging buffer.

    Args:
      layer_idx: Target layer index to fetch.
      shard_idx: Target shard index to fetch.
    """
    return self._impl.get_host_buffer(layer_idx, shard_idx)

  def get_local_endpoints(self) -> List[Dict[str, Any]]:
    """Returns the list of transfer endpoints advertised by this instance."""
    eps = self._impl.get_local_endpoints()
    if (
        not self._has_explicit_global_shard_indices
        and self._global_shard_indices
    ):
      g_to_l = {int(g): idx for idx, g in enumerate(self._global_shard_indices)}
      normalized = []
      for ep in eps:
        raw_shards = ep.get("shards", [])
        local_shards = [g_to_l.get(int(s), int(s)) for s in raw_shards]
        normalized.append({
            **ep,
            "shards": local_shards,
            "global_shards": list(raw_shards),
        })
      return normalized
    return eps

  @property
  def local_port(self) -> Optional[int]:
    """Returns the active local port assigned to the transceiving sockets server."""
    if self._impl is None:
      return None
    return self._impl.local_port

  @property
  def listener_port(self) -> Optional[int]:
    """Returns the active local port assigned to the C++ Listener."""
    if self._impl is None:
      return None
    return self._impl.listener_port

  @property
  def is_listener_active(self) -> bool:
    """Returns whether the native C++ Listener is actively running."""
    if self._impl is None:
      return False
    return self._impl.is_listener_active

  @property
  def num_layers(self) -> int:
    """Returns the total number of model weight layers registered."""
    return self._impl.num_layers

  @property
  def num_shards(self) -> int:
    """Returns the sharded devices count per layer."""
    return self._impl.num_shards

  @property
  def slice_byte_size(self) -> int:
    """Returns the slice capacity per device block."""
    return self._impl.slice_byte_size

  @property
  def ring_buffer_size(self) -> int:
    """Returns the configured host staging ring-buffer pool size (0 if disabled)."""
    return self._impl.ring_buffer_size

  @property
  def allocated_host_dram_bytes(self) -> int:
    """Returns the total host DRAM bytes allocated by this synchronizer."""
    return self._impl.allocated_host_dram_bytes

  def get_metrics(self) -> dict[str, float | int]:
    """Returns a dictionary of internal performance metrics."""
    m = self._impl.get_metrics()
    d2h_time_s = max(m.last_d2h_time_ms / 1000.0, 1e-9)
    h2h_time_s = max(m.last_h2h_time_ms / 1000.0, 1e-9)
    tiling_time_s = max(m.last_tiling_time_ms / 1000.0, 1e-9)
    detiling_time_s = max(m.last_detiling_time_ms / 1000.0, 1e-9)

    d2h_bytes_gb = m.last_d2h_bytes / 1e9
    h2h_bytes_gb = m.last_h2h_bytes / 1e9
    tiled_bytes_gb = m.last_tiled_bytes / 1e9
    detiled_bytes_gb = m.last_detiled_bytes / 1e9

    total_h2h_time_s = max(m.total_h2h_time_ms / 1000.0, 1e-9)
    total_h2h_bytes_gb = m.total_h2h_bytes / 1e9

    return {
        "last_d2h_time_ms": m.last_d2h_time_ms,
        "last_h2h_time_ms": m.last_h2h_time_ms,
        "last_staging_time_ms": m.last_staging_time_ms,
        "last_tiling_time_ms": m.last_tiling_time_ms,
        "last_detiling_time_ms": m.last_detiling_time_ms,
        "last_total_push_resharded_time_ms": (
            m.last_total_push_resharded_time_ms
        ),
        "last_d2h_bytes": m.last_d2h_bytes,
        "last_h2h_bytes": m.last_h2h_bytes,
        "last_tiled_bytes": m.last_tiled_bytes,
        "last_detiled_bytes": m.last_detiled_bytes,
        "total_d2h_time_ms": m.total_d2h_time_ms,
        "total_h2h_time_ms": m.total_h2h_time_ms,
        "total_staging_time_ms": m.total_staging_time_ms,
        "total_tiling_time_ms": m.total_tiling_time_ms,
        "total_detiling_time_ms": m.total_detiling_time_ms,
        "total_push_resharded_time_ms": m.total_push_resharded_time_ms,
        "total_d2h_bytes": m.total_d2h_bytes,
        "total_h2h_bytes": m.total_h2h_bytes,
        "total_tiled_bytes": m.total_tiled_bytes,
        "total_detiled_bytes": m.total_detiled_bytes,
        "d2h_call_count": m.d2h_call_count,
        "push_resharded_call_count": m.push_resharded_call_count,
        "d2h_bandwidth_gbps": (
            d2h_bytes_gb / d2h_time_s if m.last_d2h_bytes > 0 else 0.0
        ),
        "h2h_bandwidth_gbps": (
            h2h_bytes_gb / h2h_time_s if m.last_h2h_bytes > 0 else 0.0
        ),
        "total_h2h_bandwidth_gbps": (
            total_h2h_bytes_gb / total_h2h_time_s
            if m.total_h2h_bytes > 0
            else 0.0
        ),
        "tiling_bandwidth_gbps": (
            tiled_bytes_gb / tiling_time_s if m.last_tiled_bytes > 0 else 0.0
        ),
        "detiling_bandwidth_gbps": (
            detiled_bytes_gb / detiling_time_s
            if m.last_detiled_bytes > 0
            else 0.0
        ),
    }

  def reset_metrics(self) -> None:
    """Resets all recorded internal metrics."""
    self._impl.reset_metrics()

  @classmethod
  def configure_telemetry(cls, backends: Optional[List[str]] = None) -> None:
    """Configures active C++ telemetry backends."""
    _weight_synchronizer.configure_telemetry(backends)

  @classmethod
  def get_and_reset_metric_samples(cls) -> Dict[str, List[float]]:
    """Extracts and resets buffered telemetry metric samples across all backends."""
    return _weight_synchronizer.get_and_reset_metric_samples()

  @classmethod
  def get_raiden_metrics_prometheus_text(cls) -> str:
    """Exports Prometheus text snapshot of TPU Raiden metrics."""
    return _weight_synchronizer.get_raiden_metrics_prometheus_text()

  def shutdown(self) -> None:
    """Releases and shuts down the underlying C++ synchronizer instance."""
    self._impl = None
