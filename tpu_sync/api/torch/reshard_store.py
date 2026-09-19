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

from tpu_sync.api import common
from tpu_sync.api.torch import torch_abi
from tpu_sync.api.torch import torch_tpu_common_loader

torch_tpu_common_loader.load_torch_tpu_common()

# pylint: disable=g-import-not-at-top
_impl = torch_abi.load_extension(
    "tpu_sync.frameworks.torch",
    "_tpu_raiden_torch",
)
# pylint: enable=g-import-not-at-top

RaidenId = getattr(_impl, "RaidenId", common.RaidenId)

# True if the reshard store accepts request_registry_ttl_s; missing on older
# extensions.
SUPPORTS_REQUEST_REGISTRY_TTL = bool(
    getattr(_impl, "reshard_store_supports_request_registry_ttl", False)
)


class ReshardStore:
  """Thin store hosting RaidenController and ReshardService in-engine.

  Eliminates the separate Python or C++ sidecar binary: the engine process
  hosts both the controller (which workers register with via WorkerService)
  and the reshard control plane.
  """

  def __init__(
      self,
      raiden_id: RaidenId,
      store_server_ip: str,
      raiden_controller_port: int = 0,
      reshard_service_port: int = 0,
      *,
      request_registry_ttl_s: float = 600.0,
  ):
    """Args: raiden_id: Identity of the hosting engine's reshard store.

    store_server_ip: Routable address the store advertises.
    raiden_controller_port: Dispatch controller port.
    reshard_service_port: Reshard service port.
    request_registry_ttl_s: Seconds an unclaimed request-block registration
    stays in the registry.
    """
    request_registry_ttl_s = float(request_registry_ttl_s)
    if request_registry_ttl_s <= 0:
      raise ValueError("request_registry_ttl_s must be positive")
    # The extension exposes the reshard-store factory as a free function
    # (create_reshard_store) returning the KVCacheStore binding: a second
    # nb::class_ registration of the same wrapper type would be silently
    # dropped by nanobind.
    self._impl = _impl.create_reshard_store(
        raiden_id,
        store_server_ip,
        raiden_controller_port,
        reshard_service_port,
        request_registry_ttl_s,
    )

  @property
  def raiden_id(self) -> RaidenId:
    return self._impl.raiden_id

  @property
  def raiden_controller_address(self) -> str:
    return self._impl.raiden_controller_address

  @property
  def reshard_service_port(self) -> int:
    return self._impl.reshard_service_port

  @property
  def request_registry_ttl_s(self) -> float:
    return float(self._impl.request_registry_ttl_s)
