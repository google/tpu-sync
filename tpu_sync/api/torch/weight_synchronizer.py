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

"""High-performance PyTorch Weight Synchronizer for Trainer-Inference Pipelines."""

import math
from typing import Any, Dict, List, Optional, Union

import torch

from tpu_sync.api import common
from tpu_sync.api import weight_sync_common
from tpu_sync.api.torch import torch_abi
from tpu_sync.api.torch import torch_tpu_common_loader

torch_tpu_common_loader.load_torch_tpu_common()

_weight_synchronizer = torch_abi.load_extension(
    "tpu_sync.frameworks.torch",
    "_tpu_raiden_torch",
)

common.register_telemetry_callbacks(
    increment_counter=_weight_synchronizer.increment_counter,
    set_gauge=_weight_synchronizer.set_gauge,
    observe_histogram=_weight_synchronizer.observe_histogram,
    flush_cloud_logging=_weight_synchronizer.flush_cloud_logging,
)


def configure_telemetry(exporter_types: Optional[list[str]] = None) -> None:
  """Configures the telemetry backend exporters in RaidenMetricStore."""
  _weight_synchronizer.configure_telemetry(exporter_types)


def get_and_reset_metric_samples() -> dict[str, Any]:
  """Retrieves and clears buffered metric samples from RaidenMetricStore."""
  return _weight_synchronizer.get_and_reset_metric_samples()


def get_raiden_metrics_prometheus_text() -> str:
  """Returns current Prometheus metric exposition text from RaidenMetricStore."""
  return _weight_synchronizer.get_raiden_metrics_prometheus_text()


def flush_cloud_logging(
    phase: str,
    uuid: int = 0,
    req_id: str = "",
) -> str:
  """Flushes buffered telemetry metrics to Cloud Logging (stdout JSON) if enabled."""
  return _weight_synchronizer.flush_cloud_logging(phase, uuid, req_id)


class WeightSynchronizer:
  """Zero-copy PyTorch Weight Synchronizer."""

  def __init__(
      self,
      device_tensors: List[List[torch.Tensor]],
      local_port: Optional[int] = None,
      parallelism: int = 1,
      listener_port: Optional[int] = None,
      bind_ip: Optional[str] = None,
      unsafe_skip_buffer_lock: bool = True,
      auto_h2d: bool = False,
      global_shard_indices: Optional[List[int]] = None,
  ):
    """Instantiates the PyTorch Weight Synchronizer shims.

    Args:
      device_tensors: List of list of device-placed contiguous Tensors
        representing the sharded model weights [layers, shards].
      local_port: Sockets listener port assigned for incoming pulls (inference
        mode).
      parallelism: Parallel TCP sockets workers count.
      listener_port: RPC control listener port.
      bind_ip: Sockets server bind IP address.
      unsafe_skip_buffer_lock: Whether to bypass buffer lock safety.
      auto_h2d: Automatically execute H2D ingestion upon data arrival.
      global_shard_indices: Optional list of global shard indices corresponding
        to each local shard on this process. If None, defaults to [rank *
        num_shards + i for i in range(num_shards)] when torch.distributed is
        initialized, or [0, ..., num_shards - 1] otherwise.
    """
    self._has_explicit_global_shard_indices = global_shard_indices is not None
    if global_shard_indices is None:
      num_shards = len(device_tensors[0]) if device_tensors else 0
      offset = 0
      if (
          torch.distributed.is_available()
          and torch.distributed.is_initialized()
      ):
        offset = torch.distributed.get_rank() * num_shards
      global_shard_indices = [offset + i for i in range(num_shards)]

    self._global_shard_indices = (
        list(global_shard_indices) if global_shard_indices is not None else []
    )

    self._impl = _weight_synchronizer.WeightSynchronizer(
        device_tensors,
        local_port,
        parallelism,
        listener_port,
        bind_ip,
        unsafe_skip_buffer_lock,
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

  def push_weights(self, peers: List[str]) -> None:
    """Trainer pushes model weights to peer inference server coordinates."""
    self._impl.PushWeights(peers)

  def test_only_set_skip_tiling(self, skip: bool | List[bool]) -> None:
    """Sets whether D2H/H2D should skip CPU tiling/detiling (for testing only)."""
    if isinstance(skip, bool):
      self._impl.set_skip_tiling(skip)
    else:
      self._impl.set_skip_tiling(list(skip))

  def bind_weights(self, device_tensors: List[List[torch.Tensor]]) -> None:
    """Dynamically re-binds new device weights in-place without daemon restart.

    Args:
      device_tensors: New per-layer, per-shard device tensors to bind. Must
        match the layer/shard/size configuration from construction. An empty
        list is equivalent to calling `unbind_weights()`.
    """
    if not device_tensors:
      self.unbind_weights()
      return
    self._impl.bind_weights(device_tensors)

  def unbind_weights(self) -> None:
    """Releases all bound device tensors and their TPU HBM holds immediately.

    The synchronizer itself (pinned host staging buffers, listeners, controller
    registration) stays alive, so a later `bind_weights()` is cheap and does not
    require re-creating or re-registering the WeightSynchronizer. Use this right
    after a transfer completes—or, during a pipelined transfer
    (`skip_d2h=False`), immediately after `wait_for_d2h_completion(uuid)`
    returns while H2H network broadcast continues in the background—so temporary
    send tensors do not stay pinned in HBM during the next training step.

    After unbinding, `d2h()`, `h2d()`, and any controller-driven push that
    performs D2H raise until `bind_weights()` is called again.

    Must not be called while local D2H or H2D copies on this synchronizer are
    in flight.
    """
    self._impl.unbind_weights()

  def d2h(self) -> None:
    """Triggers asynchronous D2H copy of current weights to Host buffer."""
    self._impl.D2h()

  def h2d(self) -> None:
    """Triggers asynchronous H2D copy of weights from Host buffer to Device."""
    self._impl.H2d()

  def wait_for_transfer_completion(self, uuid: Optional[int] = None) -> None:
    """Blocks until the transfer with the given UUID (or any transfer if None) has finished ingestion."""
    self._impl.wait_for_transfer_completion(uuid)

  def wait_for_d2h_completion(
      self, uuid: int, timeout_s: Optional[float] = None
  ) -> None:
    """Blocks until local D2H (and host detiling) completes for `uuid`.

    During a pipelined transfer (`skip_d2h=False`), callers pass the same
    `uuid` they gave `controller.start_transfer(..., uuid=...)` (identical on
    every trainer host) to wait only until local D2H finishes, then immediately
    call `unbind_weights()` while H2H network broadcast continues in the
    background.

    Caller requirements:
      1. This call does not drive the transfer: the future returned by
         `controller.start_transfer(...)` must be awaited concurrently (e.g.
         `fut.wait()` on a background thread/event loop), otherwise no D2H runs
         and this wait blocks until `timeout_s`.
      2. H2H keeps reading this synchronizer's host staging buffers until that
         future completes, so await it before starting the next transfer on this
         synchronizer (the next D2H would overwrite those buffers);
         `bind_weights()` and training may proceed meanwhile.

    Args:
      uuid: Positive 64-bit transfer UUID passed to
        `controller.start_transfer(..., uuid=...)` (identical across all trainer
        hosts in the sync).
      timeout_s: Optional non-negative timeout in seconds. If None, blocks until
        local D2H finishes or fails.

    Raises:
      ValueError: If `uuid` is not a positive integer or `timeout_s` is negative
        or non-finite.
      TimeoutError: If `timeout_s` elapses before local D2H completes for
        `uuid`.
      RuntimeError: If local D2H fails or device weights are unbound.
    """
    if isinstance(uuid, bool) or not isinstance(uuid, int) or uuid <= 0:
      raise ValueError(
          "wait_for_d2h_completion requires a positive integer uuid; got"
          f" {uuid!r}"
      )
    if timeout_s is not None:
      if (
          isinstance(timeout_s, bool)
          or not isinstance(timeout_s, (int, float))
          or not math.isfinite(timeout_s)
          or timeout_s < 0.0
      ):
        raise ValueError(
            "wait_for_d2h_completion requires a finite non-negative timeout_s"
            f" for uuid={uuid}; got {timeout_s!r}"
        )
    self._impl.wait_for_d2h_completion(
        uuid, float(timeout_s) if timeout_s is not None else None
    )

  def get_host_buffer(
      self, layer_idx: int = 0, shard_idx: int = 0
  ) -> torch.Tensor:
    """Returns a zero-copy Host CPU PyTorch Tensor view of staging buffer.

    The view keeps the native synchronizer, with its ports, threads and host
    staging memory, alive for as long as the view or any tensor sharing its
    storage exists, even after `shutdown()`. Do not hand the view to
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
      g_to_l = {g: i for i, g in enumerate(self._global_shard_indices)}
      normalized = []
      for ep in eps:
        raw_shards = ep["shards"]
        local_shards = [g_to_l.get(s, s) for s in raw_shards]
        normalized.append({
            "endpoint": ep["endpoint"],
            "shards": local_shards,
            "global_shards": list(raw_shards),
        })
      return normalized
    return eps

  @property
  def local_port(self) -> Optional[int]:
    """Returns assigned ephemeral listener port coordinates."""
    if self._impl is None:
      return None
    return self._impl.local_port

  @property
  def listener_port(self) -> Optional[int]:
    """Returns assigned RPC listener port coordinate."""
    if self._impl is None:
      return None
    return self._impl.listener_port

  @property
  def is_listener_active(self) -> bool:
    """Returns whether the native C++ listener thread is actively running."""
    if self._impl is None:
      return False
    return self._impl.is_listener_active

  @property
  def num_layers(self) -> int:
    """Returns total layers registered."""
    return self._impl.num_layers

  @property
  def num_shards(self) -> int:
    """Returns physical devices shards count."""
    return self._impl.num_shards

  @property
  def slice_byte_size(self) -> int:
    """Returns individual major dimension slice byte size capacity."""
    return self._impl.slice_byte_size

  def get_metrics(self) -> Dict[str, float | int]:
    """Returns a dictionary of internal performance metrics."""
    return weight_sync_common.metrics_to_dict(self._impl.get_metrics())

  def reset_metrics(self) -> None:
    """Resets all recorded internal metrics."""
    self._impl.reset_metrics()

  @classmethod
  def configure_telemetry(
      cls, exporter_types: Optional[list[str]] = None
  ) -> None:
    """Configures the telemetry backend exporters in RaidenMetricStore."""
    configure_telemetry(exporter_types)

  @classmethod
  def get_and_reset_metric_samples(cls) -> dict[str, Any]:
    """Retrieves and clears buffered metric samples from RaidenMetricStore."""
    return get_and_reset_metric_samples()

  @classmethod
  def get_raiden_metrics_prometheus_text(cls) -> str:
    """Returns current Prometheus metric exposition text from RaidenMetricStore."""
    return get_raiden_metrics_prometheus_text()

  @classmethod
  def flush_cloud_logging(
      cls, phase: str, uuid: int = 0, req_id: str = ""
  ) -> str:
    """Flushes buffered telemetry metrics to Cloud Logging (stdout JSON) if enabled."""
    return flush_cloud_logging(phase, uuid, req_id)

  def shutdown(self) -> None:
    """Drops this wrapper's reference to the C++ synchronizer instance.

    The native synchronizer, with its ports, threads and host staging memory,
    is destroyed once its last reference is gone, including any view returned
    by `get_host_buffer()`. Until then it keeps running.
    """
    self._impl = None
