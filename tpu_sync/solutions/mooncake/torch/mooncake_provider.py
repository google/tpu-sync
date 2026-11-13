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

"""tpu-sync's TPU buffer provider for Mooncake TENT, for torch tensors.

Mooncake moves TPU data by staging it through host DRAM. Its adapter cannot
reach torch_tpu's device buffers itself, so pass this module to ``mooncake.tpu``
as the provider that turns TPU tensors into provider handles:

  >>> from mooncake import tpu
  >>> from tpu_sync.solutions.mooncake.torch import mooncake_provider
  >>> shards = tpu.prepare(kv_caches, provider=mooncake_provider)
  >>> # ... create the Mooncake TENT TransferEngine ...
  >>> tpu.register_with_engine(engine, shards)

Mooncake then copies byte ranges of each tensor's on-device representation to
and from host memory through the C table that ``library_path()`` exports. The
contract and the lifetime rules are in
``tpu_sync/solutions/mooncake/mooncake_provider.h``.

The table's ``dma_map`` registers TENT's host staging buffers with the TPU DMA
engine, using the PJRT client of the exported tensors, so export before
creating the engine, as above.
"""

import atexit
import functools
import os

import torch

from tpu_sync.api.torch import torch_abi
from tpu_sync.api.torch import torch_tpu_common_loader

torch_tpu_common_loader.load_torch_tpu_common()
_extension = torch_abi.load_extension(
    "tpu_sync.frameworks.torch",
    "_tpu_raiden_torch",
)


def library_path() -> str:
  """Returns the path of the library exporting ``mc_tpu_buffer_provider_get``.

  It is the _tpu_raiden_torch extension that this module has already loaded,
  so when Mooncake dlopen()s the path it gets the loaded image back, and with
  it the handles that export_shards() created.
  """
  return os.path.realpath(_extension.__file__)


def export_shards(tensor: torch.Tensor) -> list[int]:
  """Exports one Mooncake provider handle for ``tensor``'s device buffer.

  A torch tensor lives on one device, so there is one handle. Its device index
  (TENT's ``tpu:N``) is ``tensor.device.index``.

  The caller owns the returned handle and passes it to Mooncake, which releases
  it exactly once.

  The handle refers to the device buffer that backs ``tensor`` when it is
  exported, and keeps that buffer's HBM allocated until it is released, even
  if ``tensor`` is deleted. TorchTPU computes an in-place op out of place, then
  either reuses the tensor's buffer for the result, which the handle sees, or
  moves the tensor to a new buffer, which the handle does not see. Which of the
  two happens is TorchTPU's decision. Currently, ops that read the tensor, such
  as ``add_()``, ``index_copy_()`` and index assignment
  (``t[indices] = values``), reuse its buffer, while ops that write it without
  reading it, such as ``fill_()``, ``copy_()`` and slice assignment of a value
  (``t[:n] = value``), move it. Another storage that shares the buffer, such
  as an unwritten ``clone()`` of the tensor or a tensor saved for backward,
  keeps TorchTPU from reusing it. So after an in-place op, unregister the
  tensor's shards and export it again, unless you have verified that its
  buffer stays put.

  The handle does not order Mooncake's copies with torch: the caller must keep
  torch from writing the tensor while Mooncake copies out of it, and from using
  it while Mooncake copies into it.

  Args:
    tensor: A contiguous TPU tensor that starts at offset 0 of its storage and
      spans all of it.

  Returns:
    ``[handle]``, with a non-zero handle.

  Raises:
    TypeError: ``tensor`` is not a torch.Tensor.
    ValueError: ``tensor`` is not such a tensor.
    RuntimeError: TorchTPU failed to produce the tensor's buffer.
  """
  handle = _extension.mooncake_provider.export_tensor(tensor)
  _register_exit_hook()
  return [handle]


@functools.cache
def _register_exit_hook() -> None:
  """Registers the provider's exit hook with atexit, once.

  TorchTPU's own atexit handler shuts its PJRT runtime down, after which the
  client that dma_map and dma_unmap call can be destroyed. atexit runs handlers
  in reverse order of registration, so this hook, registered later, runs first
  and makes them stop calling PJRT. TorchTPU registers its handler when it
  loads, before any TPU tensor exists; registering this one at the first
  export keeps it after TorchTPU's even if TorchTPU registered its handler
  only when it initializes the TPU. Nothing can be DMA-mapped before the first
  export.
  """
  atexit.register(_extension.mooncake_provider.mark_process_exiting)
