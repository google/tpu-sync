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

"""The Mooncake provider's C table as Mooncake sees it, for the TPU tests."""

from collections.abc import Callable
import ctypes
import mmap
import os

import numpy as np

from tpu_sync.api.jax import mooncake_provider

Handle = ctypes.c_uint64


class ProviderTable(ctypes.Structure):
  """ctypes mirror of mc_tpu_buffer_provider_t, ABI version 2."""

  _fields_ = [
      ("abi_version", ctypes.c_uint32),
      ("name", ctypes.c_char_p),
      ("size", ctypes.CFUNCTYPE(ctypes.c_size_t, Handle)),
      ("device_index", ctypes.CFUNCTYPE(ctypes.c_int, Handle)),
      ("numa_node", ctypes.CFUNCTYPE(ctypes.c_int, Handle)),
      (
          "copy_to_host",
          ctypes.CFUNCTYPE(
              ctypes.c_int,
              Handle,
              ctypes.c_size_t,
              ctypes.c_void_p,
              ctypes.c_size_t,
          ),
      ),
      (
          "copy_from_host",
          ctypes.CFUNCTYPE(
              ctypes.c_int,
              Handle,
              ctypes.c_size_t,
              ctypes.c_void_p,
              ctypes.c_size_t,
          ),
      ),
      ("release", ctypes.CFUNCTYPE(None, Handle)),
      # Version 2.
      (
          "dma_map",
          ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t),
      ),
      ("dma_unmap", ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)),
  ]


def provider_get() -> Callable[..., int]:
  """Looks up mc_tpu_buffer_provider_get as Mooncake does: dlopen(), dlsym().

  Returns:
    The entry point, called as `get(version, ctypes.byref(table))`.
  """
  lib = ctypes.CDLL(mooncake_provider.library_path(), mode=os.RTLD_LOCAL)
  get = lib.mc_tpu_buffer_provider_get
  get.argtypes = [ctypes.c_uint32, ctypes.POINTER(ProviderTable)]
  get.restype = ctypes.c_int
  return get


def load_table() -> ProviderTable:
  """Fetches the table as Mooncake does, asking for version 2."""
  table = ProviderTable()
  if provider_get()(2, ctypes.byref(table)) != 0:
    raise AssertionError("mc_tpu_buffer_provider_get(2) failed")
  if table.abi_version != 2:
    raise AssertionError(f"Asked for ABI version 2, got {table.abi_version}")
  return table


def staging_pool(size: int) -> np.ndarray:
  """Host memory like a TENT staging pool: anonymous, page-aligned, prefaulted.

  Args:
    size: Bytes needed; rounded up to whole pages.

  Returns:
    A uint8 array over the memory, which is unmapped when the array is freed.
  """
  size = -(-size // mmap.PAGESIZE) * mmap.PAGESIZE
  pool = np.frombuffer(
      mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS),
      np.uint8,
  )
  pool[:: mmap.PAGESIZE] = 0  # TENT touches every page before mapping it.
  return pool
