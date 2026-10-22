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

"""TPU tests for the Mooncake TPU buffer provider.

Drives the provider's C table through ctypes, the way Mooncake's adapter does,
on real TPU arrays.
"""

import concurrent.futures
import ctypes
import gc
import os
import time

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from tpu_sync.api.jax import mooncake_provider
from tpu_sync.frameworks.jax import _mooncake_provider

_Handle = ctypes.c_uint64


class _ProviderTable(ctypes.Structure):
  """ctypes mirror of mc_tpu_buffer_provider_t, ABI version 1."""

  _fields_ = [
      ("abi_version", ctypes.c_uint32),
      ("name", ctypes.c_char_p),
      ("size", ctypes.CFUNCTYPE(ctypes.c_size_t, _Handle)),
      ("device_index", ctypes.CFUNCTYPE(ctypes.c_int, _Handle)),
      ("numa_node", ctypes.CFUNCTYPE(ctypes.c_int, _Handle)),
      (
          "copy_to_host",
          ctypes.CFUNCTYPE(
              ctypes.c_int,
              _Handle,
              ctypes.c_size_t,
              ctypes.c_void_p,
              ctypes.c_size_t,
          ),
      ),
      (
          "copy_from_host",
          ctypes.CFUNCTYPE(
              ctypes.c_int,
              _Handle,
              ctypes.c_size_t,
              ctypes.c_void_p,
              ctypes.c_size_t,
          ),
      ),
      ("release", ctypes.CFUNCTYPE(None, _Handle)),
  ]


def _load_table() -> _ProviderTable:
  """Fetches the table as Mooncake does: dlopen(), dlsym(), call with v1."""
  lib = ctypes.CDLL(mooncake_provider.library_path(), mode=os.RTLD_LOCAL)
  get = lib.mc_tpu_buffer_provider_get
  get.argtypes = [ctypes.c_uint32, ctypes.POINTER(_ProviderTable)]
  get.restype = ctypes.c_int
  table = _ProviderTable()
  if get(1, ctypes.byref(table)) != 0:
    raise AssertionError("mc_tpu_buffer_provider_get(1) failed")
  return table


def _host_buffer(size: int, misalignment: int) -> np.ndarray:
  """A uint8 host buffer whose address is `misalignment` past a 64-byte one."""
  raw = np.empty(size + 64 + misalignment, np.uint8)
  start = -raw.ctypes.data % 64 + misalignment
  return raw[start : start + size]


def _cpu_devices() -> list[jax.Device]:
  try:
    return jax.devices("cpu")
  except RuntimeError:  # This process has no CPU backend.
    return []


def _bytes_in_use() -> int:
  return sum(d.memory_stats()["bytes_in_use"] for d in jax.local_devices())


def _wait_for(predicate, timeout_s: float = 30.0) -> bool:
  """Polls `predicate`: device frees can complete asynchronously."""
  deadline = time.monotonic() + timeout_s
  while time.monotonic() < deadline:
    if predicate():
      return True
    time.sleep(0.05)
  return predicate()


class MooncakeProviderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.assertEqual(jax.default_backend(), "tpu")
    self.devices = jax.local_devices(backend="tpu")
    self.mesh = jax.sharding.Mesh(np.array(self.devices), ("x",))
    self.sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec("x")
    )
    self.table = _load_table()

  def _array(self, shape, dtype, seed=0):
    values = np.random.default_rng(seed).uniform(-100, 100, size=shape)
    return jax.device_put(values.astype(dtype), self.sharding)

  def _copy_to_host(self, handle: int, offset: int, out: np.ndarray) -> None:
    rc = self.table.copy_to_host(
        handle, offset, out.ctypes.data_as(ctypes.c_void_p), out.nbytes
    )
    self.assertEqual(rc, 0)

  def _copy_from_host(self, handle: int, offset: int, src: np.ndarray) -> None:
    rc = self.table.copy_from_host(
        handle, offset, src.ctypes.data_as(ctypes.c_void_p), src.nbytes
    )
    self.assertEqual(rc, 0)

  def test_library_is_the_loaded_extension(self):
    path = mooncake_provider.library_path()
    self.assertTrue(os.path.isfile(path), path)
    # RTLD_NOLOAD succeeds only if this exact file is already mapped, i.e.
    # Python's import loaded it and dlopen(path) returns that same image.
    ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
    self.assertEqual(self.table.abi_version, 1)
    self.assertEqual(self.table.name, b"tpu-sync")

    # State is shared too: a handle exported through Python and released
    # through the dlopen()ed table is gone from Python's count.
    before = _mooncake_provider.live_buffer_count()
    handles = mooncake_provider.export_shards(self._array((8, 128), np.float32))
    self.assertEqual(
        _mooncake_provider.live_buffer_count(), before + len(handles)
    )
    for handle in handles:
      self.table.release(handle)
    self.assertEqual(_mooncake_provider.live_buffer_count(), before)

  # Per-device shard shapes (3-D shapes get tiled, padded TPU layouts), and
  # host pointers that are DMA-aligned or not (the latter take the bounce path).
  @parameterized.product(
      shard_shape_and_dtype=(
          ((16, 256), np.float32),
          ((4, 3, 1000), jnp.bfloat16),
          ((8, 8, 130), np.int8),
      ),
      host_misalignment=(0, 1),
  )
  def test_round_trip_through_host_restores_array(
      self, shard_shape_and_dtype, host_misalignment
  ):
    shard_shape, dtype = shard_shape_and_dtype
    shape = (len(self.devices) * shard_shape[0],) + shard_shape[1:]
    src = self._array(shape, dtype, seed=1)
    dst = jax.device_put(jnp.zeros(shape, dtype), self.sharding)
    expected = np.asarray(src)
    src_handles = mooncake_provider.export_shards(src)
    dst_handles = mooncake_provider.export_shards(dst)
    self.assertLen(src_handles, len(src.addressable_shards))

    for shard, src_handle, dst_handle in zip(
        src.addressable_shards, src_handles, dst_handles
    ):
      size = self.table.size(src_handle)
      self.assertEqual(size, shard.data.on_device_size_in_bytes())
      self.assertEqual(self.table.size(dst_handle), size)
      self.assertEqual(
          self.table.device_index(src_handle), self.devices.index(shard.device)
      )
      # The node the device publishes, or -1 if it publishes none.
      self.assertEqual(
          self.table.numa_node(src_handle),
          getattr(shard.device, "numa_node", -1),
      )

      # Several chunks, as TENT stages a buffer in pieces. Device offsets must
      # be multiples of 32 bytes; host pointers need not be.
      host = _host_buffer(size, host_misalignment)
      chunk = (size // 3 // 32 + 1) * 32
      for offset in range(0, size, chunk):
        self._copy_to_host(src_handle, offset, host[offset : offset + chunk])
      for offset in range(0, size, chunk):
        self._copy_from_host(dst_handle, offset, host[offset : offset + chunk])

    for handle in src_handles + dst_handles:
      self.table.release(handle)
    np.testing.assert_array_equal(np.asarray(dst), expected)

  def test_concurrent_copies_on_one_handle(self):
    src = self._array((len(self.devices) * 64, 1024), np.float32, seed=2)
    dst = jax.device_put(jnp.zeros(src.shape, src.dtype), self.sharding)
    expected = np.asarray(src)
    src_handles = mooncake_provider.export_shards(src)
    dst_handles = mooncake_provider.export_shards(dst)

    # ctypes releases the GIL around each call, like Mooncake's C++ threads.
    chunk = 32 * 1024
    jobs = []
    for src_handle, dst_handle in zip(src_handles, dst_handles):
      size = self.table.size(src_handle)
      for offset in range(0, size, chunk):
        jobs.append((src_handle, dst_handle, offset, min(chunk, size - offset)))

    def move(job):
      src_handle, dst_handle, offset, length = job
      host = np.empty(length, np.uint8)
      self._copy_to_host(src_handle, offset, host)
      self._copy_from_host(dst_handle, offset, host)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
      list(pool.map(move, jobs))

    for handle in src_handles + dst_handles:
      self.table.release(handle)
    np.testing.assert_array_equal(np.asarray(dst), expected)

  def test_any_length_at_aligned_offset(self):
    src = self._array((len(self.devices) * 8, 128), np.float32, seed=6)
    dst = jax.device_put(jnp.zeros(src.shape, src.dtype), self.sharding)
    src_handles = mooncake_provider.export_shards(src)
    dst_handles = mooncake_provider.export_shards(dst)
    src_handle, dst_handle = src_handles[0], dst_handles[0]
    ranges = ((32, 1), (64, 33), (160, 31), (256, 100))
    for offset, length in ranges:
      host = _host_buffer(length, 0)
      self._copy_to_host(src_handle, offset, host)
      self._copy_from_host(dst_handle, offset, host)
    # Compare the raw on-device bytes of both buffers over the copied ranges.
    size = self.table.size(src_handle)
    src_bytes, dst_bytes = _host_buffer(size, 0), _host_buffer(size, 0)
    self._copy_to_host(src_handle, 0, src_bytes)
    self._copy_to_host(dst_handle, 0, dst_bytes)
    for offset, length in ranges:
      np.testing.assert_array_equal(
          dst_bytes[offset : offset + length],
          src_bytes[offset : offset + length],
      )
    for handle in src_handles + dst_handles:
      self.table.release(handle)

  def test_misaligned_device_offset_fails_cleanly(self):
    handles = mooncake_provider.export_shards(self._array((8, 128), np.float32))
    host = _host_buffer(64, 0)
    ptr = host.ctypes.data_as(ctypes.c_void_p)
    self.assertNotEqual(self.table.copy_to_host(handles[0], 1, ptr, 32), 0)
    self.assertNotEqual(self.table.copy_from_host(handles[0], 4, ptr, 32), 0)
    self.assertEqual(self.table.copy_to_host(handles[0], 32, ptr, 32), 0)
    for handle in handles:
      self.table.release(handle)

  def test_out_of_range_copy_fails(self):
    handles = mooncake_provider.export_shards(self._array((8, 128), np.float32))
    size = self.table.size(handles[0])
    host = np.empty(size + 1, np.uint8)
    ptr = host.ctypes.data_as(ctypes.c_void_p)
    self.assertNotEqual(
        self.table.copy_to_host(handles[0], 0, ptr, size + 1), 0
    )
    self.assertNotEqual(self.table.copy_from_host(handles[0], size, ptr, 1), 0)
    for handle in handles:
      self.table.release(handle)

  def test_handle_keeps_hbm_alive_until_released(self):
    shape = (len(self.devices) * 1024, 2048)  # 8 MiB of float32 per device.
    src = self._array(shape, np.float32, seed=3)
    expected = [np.asarray(shard.data) for shard in src.addressable_shards]
    nbytes = src.nbytes
    handles = mooncake_provider.export_shards(src)
    in_use_with_array = _bytes_in_use()

    src.delete()
    del src
    gc.collect()
    # Tempt the allocator to reuse the memory if the handle did not hold it.
    decoy = self._array(shape, np.float32, seed=4)
    decoy.block_until_ready()
    self.assertGreaterEqual(_bytes_in_use(), in_use_with_array + decoy.nbytes)

    # Copy each shard back out through its handle and compare: the bytes are
    # the original ones, not the decoy's.
    restored = jax.device_put(jnp.zeros(shape, np.float32), self.sharding)
    restored_handles = mooncake_provider.export_shards(restored)
    for handle, restored_handle in zip(handles, restored_handles):
      host = np.empty(self.table.size(handle), np.uint8)
      self._copy_to_host(handle, 0, host)
      self._copy_from_host(restored_handle, 0, host)
    for handle in restored_handles:
      self.table.release(handle)
    for shard, want in zip(restored.addressable_shards, expected):
      np.testing.assert_array_equal(np.asarray(shard.data), want)

    in_use_before_release = _bytes_in_use()
    for handle in handles:
      self.table.release(handle)
    self.assertTrue(
        _wait_for(lambda: _bytes_in_use() <= in_use_before_release - nbytes),
        "releasing the handles did not free the deleted array's HBM",
    )

  def test_failed_export_leaks_nothing(self):
    shape = (len(self.devices) * 1024, 2048)
    baseline_count = _mooncake_provider.live_buffer_count()
    baseline_bytes = _bytes_in_use()
    array = self._array(shape, np.float32, seed=5)
    nbytes = array.nbytes

    # Every shard but the last one is valid, so the failure happens after the
    # earlier shards' buffers were already acquired.
    shards = [(s.data, 0) for s in array.addressable_shards]
    shards.append((object(), 0))
    with self.assertRaises(AttributeError):  # object() has no raw buffer.
      _mooncake_provider.export_buffers(shards)
    del shards
    self.assertEqual(_mooncake_provider.live_buffer_count(), baseline_count)

    # With no buffer reference left behind, deleting the array frees its HBM.
    array.delete()
    del array
    gc.collect()
    self.assertTrue(
        _wait_for(lambda: _bytes_in_use() < baseline_bytes + nbytes),
        "a failed export kept the array's HBM alive",
    )

  def test_rejects_unsupported_arrays(self):
    with self.assertRaises(TypeError):
      mooncake_provider.export_shards(np.zeros(8, np.float32))

    deleted = self._array((8, 128), np.float32)
    deleted.delete()
    with self.assertRaisesRegex(ValueError, "deleted"):
      mooncake_provider.export_shards(deleted)

    uncommitted = jnp.zeros((8, 128), np.float32)
    self.assertFalse(uncommitted.committed)
    with self.assertRaisesRegex(ValueError, "uncommitted"):
      mooncake_provider.export_shards(uncommitted)

  def test_rejects_array_on_cpu(self):
    cpu_devices = _cpu_devices()
    if not cpu_devices:
      self.skipTest("No CPU backend in this process.")
    on_cpu = jax.device_put(np.zeros(8, np.float32), cpu_devices[0])
    with self.assertRaisesRegex(ValueError, "only TPU"):
      mooncake_provider.export_shards(on_cpu)


if __name__ == "__main__":
  absltest.main()
