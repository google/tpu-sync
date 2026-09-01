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

from typing import List, Optional

import torch

# Import Pybind11 dynamic binary extension E2E!
from tpu_sync.api.torch import torch_tpu_common_loader

torch_tpu_common_loader.load_torch_tpu_common()

# pylint: disable=g-import-not-at-top
from tpu_sync.api.torch import torch_abi

_weight_synchronizer = torch_abi.load_extension(
    "tpu_sync.frameworks.torch",
    "_tpu_raiden_torch",
)
# pylint: enable=g-import-not-at-top


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
    """
    self._impl = _weight_synchronizer.WeightSynchronizer(
        device_tensors,
        local_port,
        parallelism,
        listener_port,
        bind_ip,
        unsafe_skip_buffer_lock,
        auto_h2d,
    )

  def push_weights(self, peers: List[str]) -> None:
    """Trainer pushes model weights to peer inference server coordinates."""
    self._impl.PushWeights(peers)

  def bind_weights(self, device_tensors: List[List[torch.Tensor]]) -> None:
    """Dynamically re-binds new device weights in-place without daemon restart."""
    self._impl.bind_weights(device_tensors)

  def d2h(self) -> None:
    """Triggers asynchronous D2H copy of current weights to Host buffer."""
    self._impl.D2h()

  def h2d(self) -> None:
    """Triggers asynchronous H2D copy of weights from Host buffer to Device."""
    self._impl.H2d()

  def get_host_buffer(
      self, layer_idx: int = 0, shard_idx: int = 0
  ) -> torch.Tensor:
    """Returns a zero-copy Host CPU PyTorch Tensor view of staging buffer.

    Args:
      layer_idx: Target layer index to fetch.
      shard_idx: Target shard index to fetch.
    """
    return self._impl.get_host_buffer(layer_idx, shard_idx)

  @property
  def local_port(self) -> Optional[int]:
    """Returns assigned ephemeral listener port coordinates."""
    return self._impl.local_port

  @property
  def listener_port(self) -> Optional[int]:
    """Returns assigned RPC listener port coordinate."""
    return self._impl.listener_port

  @property
  def is_listener_active(self) -> bool:
    """Returns whether the native C++ listener thread is actively running."""
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
