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

from typing import Any, Dict, List, Optional

# Import Nanobind binary library directly E2E!
from tpu_sync.frameworks.jax import _tpu_raiden_jax as _weight_synchronizer


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
    """
    self._impl = _weight_synchronizer.WeightSynchronizer(
        jax_arrays,
        local_port,
        parallelism,
        unsafe_skip_buffer_lock,
        listener_port,
        bind_ip,
        auto_h2d,
    )

  def d2h(self) -> None:
    """Triggers asynchronous Device-to-Host (D2H) copy of current weights to Host buffer."""
    self._impl.D2h()

  def h2d(self) -> None:
    """Triggers asynchronous Host-to-Device (H2D) copy of staged host buffer back to Device memory E2E."""
    self._impl.H2d()

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

    Args:
      layer_idx: Target layer index to fetch.
      shard_idx: Target shard index to fetch.
    """
    return self._impl.get_host_buffer(layer_idx, shard_idx)

  def get_local_endpoints(self) -> List[Dict[str, Any]]:
    """Returns the list of transfer endpoints advertised by this instance."""
    return self._impl.get_local_endpoints()

  @property
  def local_port(self) -> Optional[int]:
    """Returns the active local port assigned to the transceiving sockets server."""
    return self._impl.local_port

  @property
  def listener_port(self) -> Optional[int]:
    """Returns the active local port assigned to the C++ Listener."""
    return self._impl.listener_port

  @property
  def is_listener_active(self) -> bool:
    """Returns whether the native C++ Listener is actively running."""
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
    m = self._impl.get_metrics()
    d2h_time_s = max(m.last_d2h_time_ms / 1000.0, 1e-9)
    h2h_time_s = max(m.last_h2h_time_ms / 1000.0, 1e-9)
    tiling_time_s = max(m.last_tiling_time_ms / 1000.0, 1e-9)
    detiling_time_s = max(m.last_detiling_time_ms / 1000.0, 1e-9)

    d2h_bytes_gb = m.last_d2h_bytes / 1e9
    h2h_bytes_gb = m.last_h2h_bytes / 1e9
    tiled_bytes_gb = m.last_tiled_bytes / 1e9
    detiled_bytes_gb = m.last_detiled_bytes / 1e9

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
