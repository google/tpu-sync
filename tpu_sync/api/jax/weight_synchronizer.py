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

from typing import Any, Dict, List, Optional, Union

import jax

from tpu_sync.api import common
from tpu_sync.api import weight_sync_common

# Import Nanobind binary library directly E2E!
from tpu_sync.frameworks.jax import _tpu_raiden_jax as _weight_synchronizer

RaidenFuture = _weight_synchronizer.RaidenFuture

common.register_telemetry_callbacks(
    increment_counter=_weight_synchronizer.increment_counter,
    set_gauge=_weight_synchronizer.set_gauge,
    observe_histogram=_weight_synchronizer.observe_histogram,
    flush_cloud_logging=_weight_synchronizer.flush_cloud_logging,
)

configure_telemetry = _weight_synchronizer.configure_telemetry
get_and_reset_metric_samples = _weight_synchronizer.get_and_reset_metric_samples
get_raiden_metrics_prometheus_text = (
    _weight_synchronizer.get_raiden_metrics_prometheus_text
)
flush_cloud_logging = _weight_synchronizer.flush_cloud_logging


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
    """
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
        )
    )
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

  def d2h(self) -> RaidenFuture:
    """Schedules a D2H copy of the bound weights into host staging buffers.

    The returned future completes after all layers have been copied and
    detiled. Call its `wait()` method to wait for completion and raise any
    asynchronous copy error. `get_host_buffer()` waits for its requested
    layer, and `h2d()` waits for all layers, without an explicit `wait()`.
    A subsequent `d2h()` or `bind_weights()` waits for the in-flight copy.
    Use `d2h().wait()` before reading a previously obtained host view or
    inspecting completed-copy metrics.
    """
    return self._impl.D2h()

  def h2d(self) -> None:
    """Triggers asynchronous Host-to-Device (H2D) copy of staged host buffer back to Device memory E2E."""
    self._impl.H2d()

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
    self._impl.bind_weights(jax_arrays)

  def get_host_buffer(self, layer_idx: int = 0, shard_idx: int = 0) -> any:
    """Returns a zero-copy Host-side CPU NumPy ndarray view of the C++ staging buffer.

    Waits for an in-flight D2H copy of the requested layer and raises if that
    copy failed before returning the view.

    The view keeps the native synchronizer, with its ports, threads and host
    staging memory, alive for as long as the view or any array sharing its
    memory exists, even after `shutdown()`. Do not hand the view to
    asynchronous consumers (for example raw-transfer copies) that can outlive
    the caller's references: the final release then tears the synchronizer
    down on that consumer's thread.

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

  def get_metrics(self) -> dict[str, float | int]:
    """Returns a dictionary of internal performance metrics."""
    return weight_sync_common.metrics_to_dict(self._impl.get_metrics())

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

  @classmethod
  def flush_cloud_logging(
      cls, phase: str, uuid: int = 0, req_id: str = ""
  ) -> str:
    """Flushes buffered telemetry metrics to Cloud Logging (stdout JSON) if enabled."""
    return _weight_synchronizer.flush_cloud_logging(phase, uuid, req_id)

  def shutdown(self) -> None:
    """Drops this wrapper's reference to the C++ synchronizer instance.

    The native synchronizer, with its ports, threads and host staging memory,
    is destroyed once its last reference is gone, including any view returned
    by `get_host_buffer()`. Until then it keeps running.
    """
    self._impl = None
