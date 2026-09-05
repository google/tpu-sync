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

"""PyTorch KV cache offloader backed by an externally managed shm pool."""

from typing import Any, Sequence

_TORCH_IMPL = None


def _torch_impl():
  """Loads the consolidated Torch extension after the torch_tpu runtime."""
  global _TORCH_IMPL
  if _TORCH_IMPL is None:
    # pylint: disable=g-import-not-at-top
    from tpu_sync.api.torch import torch_abi
    from tpu_sync.api.torch import torch_tpu_common_loader

    torch_tpu_common_loader.load_torch_tpu_common()
    _TORCH_IMPL = torch_abi.load_extension(
        "tpu_sync.frameworks.torch",
        "_tpu_raiden_torch",
    )
    # pylint: enable=g-import-not-at-top
  return _TORCH_IMPL


class KVCacheOffloader:
  """Moves KV pages between TPU buffers and complete host object views.

  Each same-sized TPU physical buffer is treated as a raw byte array split
  into ``page_nbytes`` pages; logical tensor dimensions are not interpreted.
  Host-pool allocation and object-view creation belong to the external
  SharedMemoryPool. This wrapper only forwards the caller's existing mapping
  address for one whole-pool DMA registration and forwards complete object
  tensor views to the native copy engine. The pool must provide mutually
  disjoint object-tensor storage for all active copies.
  """

  def __init__(
      self,
      kv_cache_tensors: Sequence[Any],
      page_nbytes: int,
  ) -> None:
    self._impl = _torch_impl().KVCacheOffloader(
        list(kv_cache_tensors), page_nbytes
    )

  def map_shared_memory(
      self, mapped_address: int, pool_size_bytes: int
  ) -> None:
    """Registers an existing whole-pool mapping for TPU DMA.

    The external pool owner must keep this process-local virtual address
    mapped and page-locked until ``unmap_shared_memory`` succeeds.
    """
    self._impl.map_shared_memory(mapped_address, pool_size_bytes)

  def unmap_shared_memory(self) -> None:
    """Drains submitted copies and releases the whole-pool registration."""
    self._impl.unmap_shared_memory()

  def h2d(
      self,
      block_ids: Sequence[int],
      object_tensors: Sequence[Any],
      rank_id: int,
  ) -> Any:
    """Starts host-to-device copies from complete object tensor views."""
    return self._impl.h2d(list(block_ids), list(object_tensors), rank_id)

  def d2h(
      self,
      block_ids: Sequence[int],
      object_tensors: Sequence[Any],
      rank_id: int,
  ) -> Any:
    """Starts device-to-host copies into complete object tensor views."""
    return self._impl.d2h(list(block_ids), list(object_tensors), rank_id)

  @property
  def num_layers(self) -> int:
    return self._impl.num_layers

  @property
  def num_blocks(self) -> int:
    return self._impl.num_blocks

  @property
  def page_nbytes(self) -> int:
    return self._impl.page_nbytes

  @property
  def is_shared_memory_mapped(self) -> bool:
    return self._impl.is_shared_memory_mapped
