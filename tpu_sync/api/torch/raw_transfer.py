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

"""Raw device<->host DMA transfers for torch tensors.

Torch-side twin of the JAX ``_raw_transfer`` module: byte-level PJRT raw-buffer
copies between a TPU tensor's device buffer and pinned host memory, addressed
in whole major-dimension slices via ``*_offsets_major_dim`` /
``copy_sizes_major_dim`` (empty = full physical buffer). Host data keeps the
physical (tiled) on-device layout; partial copies require rank >= 3 and
4 KiB-aligned slices.

Semantics:
  * Every transfer materializes the TPU tensor's PjRtBuffer and awaits its
    ready future before issuing DMA, so data produced by a just-launched op is
    committed. Later torch ops on the tensor create *new* device buffers;
    re-issue the transfer to observe them.
  * Host tensors must be CPU + contiguous and should be pinned
    (``pin_memory=True`` or ``RawHostBuffer``); pageable memory falls back to
    staged copies (warning logged; ``TPU_RAIDEN_RAW_REQUIRE_PINNED_HOST=1``
    makes it an error).
  * Async variants return a ``PjRtCopyFuture`` (``wait()`` / ``is_ready()``);
    ``await_all`` / ``is_ready`` accept a future or a list of futures.

The bindings live inside the ABI-dispatched ``_tpu_raiden_torch`` extension as
its ``raw_transfer`` submodule; this shim selects the variant matching the
installed torch and loads torch_tpu's common runtime first.
"""

from typing import Any, List, Sequence

_IMPL = None


def _impl():
  """Returns the raw_transfer submodule of the torch-backed extension."""
  global _IMPL
  if _IMPL is None:
    # pylint: disable=g-import-not-at-top
    from tpu_sync.api.torch import torch_abi
    from tpu_sync.api.torch import torch_tpu_common_loader

    torch_tpu_common_loader.load_torch_tpu_common()
    ext = torch_abi.load_extension(
        "tpu_sync.frameworks.torch",
        "_tpu_raiden_torch",
    )
    # pylint: enable=g-import-not-at-top
    _IMPL = getattr(ext, "raw_transfer")
  return _IMPL


def transfer_d2h_async(
    src_arr: Any,
    dst_arr: Any,
    *,
    src_offsets_major_dim: Sequence[int] = (),
    dst_offsets_major_dim: Sequence[int] = (),
    copy_sizes_major_dim: Sequence[int] = (),
    unsafe_skip_buffer_lock: bool = False,
) -> Any:
  """Asynchronous device-to-host raw copy; returns a PjRtCopyFuture."""
  return _impl().transfer_d2h_async(
      src_arr,
      dst_arr,
      src_offsets_major_dim=list(src_offsets_major_dim),
      dst_offsets_major_dim=list(dst_offsets_major_dim),
      copy_sizes_major_dim=list(copy_sizes_major_dim),
      unsafe_skip_buffer_lock=unsafe_skip_buffer_lock,
  )


def transfer_h2d_async(
    src_arr: Any,
    dst_arr: Any,
    *,
    src_offsets_major_dim: Sequence[int] = (),
    dst_offsets_major_dim: Sequence[int] = (),
    copy_sizes_major_dim: Sequence[int] = (),
    unsafe_skip_buffer_lock: bool = False,
) -> Any:
  """Asynchronous host-to-device raw copy; returns a PjRtCopyFuture."""
  return _impl().transfer_h2d_async(
      src_arr,
      dst_arr,
      src_offsets_major_dim=list(src_offsets_major_dim),
      dst_offsets_major_dim=list(dst_offsets_major_dim),
      copy_sizes_major_dim=list(copy_sizes_major_dim),
      unsafe_skip_buffer_lock=unsafe_skip_buffer_lock,
  )


def transfer_d2h(
    src_arr: Any,
    dst_arr: Any,
    *,
    src_offsets_major_dim: Sequence[int] = (),
    dst_offsets_major_dim: Sequence[int] = (),
    copy_sizes_major_dim: Sequence[int] = (),
    unsafe_skip_buffer_lock: bool = False,
) -> None:
  """Synchronous device-to-host raw copy (issues then awaits)."""
  _impl().transfer_d2h(
      src_arr,
      dst_arr,
      src_offsets_major_dim=list(src_offsets_major_dim),
      dst_offsets_major_dim=list(dst_offsets_major_dim),
      copy_sizes_major_dim=list(copy_sizes_major_dim),
      unsafe_skip_buffer_lock=unsafe_skip_buffer_lock,
  )


def transfer_h2d(
    src_arr: Any,
    dst_arr: Any,
    *,
    src_offsets_major_dim: Sequence[int] = (),
    dst_offsets_major_dim: Sequence[int] = (),
    copy_sizes_major_dim: Sequence[int] = (),
    unsafe_skip_buffer_lock: bool = False,
) -> None:
  """Synchronous host-to-device raw copy (issues then awaits)."""
  _impl().transfer_h2d(
      src_arr,
      dst_arr,
      src_offsets_major_dim=list(src_offsets_major_dim),
      dst_offsets_major_dim=list(dst_offsets_major_dim),
      copy_sizes_major_dim=list(copy_sizes_major_dim),
      unsafe_skip_buffer_lock=unsafe_skip_buffer_lock,
  )


def transfer_d2h_batch_async(
    src_arrs: List[Any],
    dst_arrs: List[Any],
    *,
    src_offsets_major_dim: Sequence[int] = (),
    dst_offsets_major_dim: Sequence[int] = (),
    copy_sizes_major_dim: Sequence[int] = (),
    unsafe_skip_buffer_lock: bool = False,
) -> Any:
  """Asynchronous batched d2h over parallel tensor lists; one joined future."""
  return _impl().transfer_d2h_batch_async(
      src_arrs,
      dst_arrs,
      src_offsets_major_dim=list(src_offsets_major_dim),
      dst_offsets_major_dim=list(dst_offsets_major_dim),
      copy_sizes_major_dim=list(copy_sizes_major_dim),
      unsafe_skip_buffer_lock=unsafe_skip_buffer_lock,
  )


def transfer_h2d_batch_async(
    src_arrs: List[Any],
    dst_arrs: List[Any],
    *,
    src_offsets_major_dim: Sequence[int] = (),
    dst_offsets_major_dim: Sequence[int] = (),
    copy_sizes_major_dim: Sequence[int] = (),
    unsafe_skip_buffer_lock: bool = False,
) -> Any:
  """Asynchronous batched h2d over parallel tensor lists; one joined future."""
  return _impl().transfer_h2d_batch_async(
      src_arrs,
      dst_arrs,
      src_offsets_major_dim=list(src_offsets_major_dim),
      dst_offsets_major_dim=list(dst_offsets_major_dim),
      copy_sizes_major_dim=list(copy_sizes_major_dim),
      unsafe_skip_buffer_lock=unsafe_skip_buffer_lock,
  )


def transfer_d2h_batch(
    src_arrs: List[Any],
    dst_arrs: List[Any],
    *,
    src_offsets_major_dim: Sequence[int] = (),
    dst_offsets_major_dim: Sequence[int] = (),
    copy_sizes_major_dim: Sequence[int] = (),
    unsafe_skip_buffer_lock: bool = False,
) -> None:
  """Synchronous batched d2h (issues all copies then awaits the join)."""
  _impl().transfer_d2h_batch(
      src_arrs,
      dst_arrs,
      src_offsets_major_dim=list(src_offsets_major_dim),
      dst_offsets_major_dim=list(dst_offsets_major_dim),
      copy_sizes_major_dim=list(copy_sizes_major_dim),
      unsafe_skip_buffer_lock=unsafe_skip_buffer_lock,
  )


def transfer_h2d_batch(
    src_arrs: List[Any],
    dst_arrs: List[Any],
    *,
    src_offsets_major_dim: Sequence[int] = (),
    dst_offsets_major_dim: Sequence[int] = (),
    copy_sizes_major_dim: Sequence[int] = (),
    unsafe_skip_buffer_lock: bool = False,
) -> None:
  """Synchronous batched h2d (issues all copies then awaits the join)."""
  _impl().transfer_h2d_batch(
      src_arrs,
      dst_arrs,
      src_offsets_major_dim=list(src_offsets_major_dim),
      dst_offsets_major_dim=list(dst_offsets_major_dim),
      copy_sizes_major_dim=list(copy_sizes_major_dim),
      unsafe_skip_buffer_lock=unsafe_skip_buffer_lock,
  )


def await_all(futures: Any) -> None:
  """Awaits one PjRtCopyFuture or a list of them; raises on the first error."""
  _impl().await_all(futures)


def is_ready(futures: Any) -> bool:
  """True iff the future (or every future in the list) has completed."""
  return _impl().is_ready(futures)


def __getattr__(name: str) -> Any:
  """Delegates classes (RawHostBuffer, PreparedTorchRawTransfer, ...) lazily."""
  if name.startswith("_"):
    raise AttributeError(name)
  return getattr(_impl(), name)
