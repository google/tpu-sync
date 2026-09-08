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

"""Thin store hosting RaidenController and ReshardService in-engine."""

from typing import Optional
from tpu_sync.api.torch import kv_cache_store
from tpu_sync.api.torch import torch_abi
from tpu_sync.api.torch import torch_tpu_common_loader

torch_tpu_common_loader.load_torch_tpu_common()

# pylint: disable=g-import-not-at-top
_impl = torch_abi.load_extension(
    "tpu_sync.frameworks.torch",
    "_tpu_raiden_torch",
)
# pylint: enable=g-import-not-at-top


class ReshardStore:
  """Thin store hosting RaidenController and ReshardService in-engine.

  Eliminates the separate Python or C++ sidecar binary: the engine process
  hosts both the controller (which workers register with via WorkerService)
  and the reshard control plane.
  """

  def __init__(
      self,
      raiden_id: kv_cache_store.RaidenId,
      store_server_ip: str,
      raiden_controller_port: int = 0,
      reshard_service_port: int = 0,
  ):
    # The extension exposes the reshard-store factory as a free function
    # (create_reshard_store) returning the KVCacheStore binding: a second
    # nb::class_ registration of the same wrapper type would be silently
    # dropped by nanobind.
    self._impl = _impl.create_reshard_store(
        getattr(raiden_id, "_impl", raiden_id),
        store_server_ip,
        raiden_controller_port,
        reshard_service_port,
    )

  @property
  def raiden_controller_address(self) -> str:
    return self._impl.raiden_controller_address

  @property
  def reshard_service_port(self) -> int:
    return self._impl.reshard_service_port
