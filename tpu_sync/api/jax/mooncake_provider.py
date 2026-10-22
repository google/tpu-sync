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

"""tpu-sync's TPU buffer provider for Mooncake TENT.

Mooncake moves TPU data by staging it through host DRAM. Its adapter cannot
reach JAX device buffers itself, so ``mooncake.tpu`` imports this module by
default and uses it to turn the shards of a jax.Array into provider handles:

  >>> from mooncake import tpu
  >>> shards = tpu.prepare(kv_caches)  # Calls library_path(), export_shards().
  >>> # ... create the Mooncake TENT TransferEngine ...
  >>> tpu.register_with_engine(engine, shards)

Mooncake then copies byte ranges of each shard's on-device representation to
and from host memory through the C table that ``library_path()`` exports. The
contract is Mooncake's ``tent/platform/tpu_buffer_provider.h`` (included here
as ``third_party/mooncake/tpu_buffer_provider.h``); the lifetime rules are in
``tpu_sync/frameworks/jax/mooncake_provider.h``.
"""

import os

import jax

from tpu_sync.frameworks.jax import _mooncake_provider


def library_path() -> str:
  """Returns the path of the library exporting ``mc_tpu_buffer_provider_get``.

  It is the extension module that this module has already imported, so when
  Mooncake dlopen()s the path it gets the loaded image back, and with it the
  handles that export_shards() created.
  """
  return os.path.realpath(_mooncake_provider.__file__)


def export_shards(array: jax.Array) -> list[int]:
  """Exports one Mooncake provider handle per addressable shard of ``array``.

  Handle ``k`` belongs to ``array.addressable_shards[k]``. Its device index
  (TENT's ``tpu:N``) is the position of that shard's device in
  ``jax.local_devices(backend="tpu")``: dense from 0 and stable for the life
  of the process.

  The caller owns the returned handles and passes them to Mooncake, which
  releases each exactly once. All-or-nothing: if any shard cannot be exported,
  this raises and no handle is left behind.

  A handle keeps its shard's HBM allocated until it is released, even if
  ``array`` is deleted. It does not order Mooncake's copies with JAX: the
  caller must keep JAX from writing the array while Mooncake copies it, and
  must not donate the array while its handles are live.

  Args:
    array: A committed jax.Array whose shards live in TPU device memory.

  Returns:
    One non-zero handle per addressable shard, in ``addressable_shards`` order.

  Raises:
    TypeError: ``array`` is not a jax.Array.
    ValueError: ``array`` is deleted, uncommitted, or not in TPU HBM.
  """
  _check_exportable(array)
  # Copies through a handle do not wait for the computation that produces the
  # array, so it must be complete before the first one.
  array.block_until_ready()
  shards = array.addressable_shards
  if not shards:
    return []
  device_indices = {
      device: index
      for index, device in enumerate(jax.local_devices(backend="tpu"))
  }
  return _mooncake_provider.export_buffers(
      [(shard.data, device_indices[shard.device]) for shard in shards]
  )


def _check_exportable(array: jax.Array) -> None:
  """Raises unless ``array`` is a live, committed jax.Array in TPU HBM."""
  if not isinstance(array, jax.Array):
    raise TypeError(f"Expected a jax.Array, got {type(array).__name__}.")
  if array.is_deleted():
    raise ValueError("Cannot export a deleted array.")
  if not array.committed:
    raise ValueError(
        "Cannot export an uncommitted array; place it explicitly, e.g. with"
        " jax.device_put(x, sharding)."
    )
  memory_kind = array.sharding.memory_kind
  if memory_kind not in (None, "device"):
    raise ValueError(
        f"Cannot export an array in {memory_kind!r} memory; only TPU device"
        " memory (HBM) is supported."
    )
  platforms = {shard.device.platform for shard in array.addressable_shards}
  if platforms - {"tpu"}:
    raise ValueError(
        f"Cannot export an array on {sorted(platforms)}; only TPU is supported."
    )
