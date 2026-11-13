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

"""TPU tests for the Mooncake TPU buffer provider for torch tensors.

Drives the provider's C table through ctypes, the way Mooncake's adapter does,
on real TPU tensors.
"""

import concurrent.futures
import ctypes
import gc
import os
import time

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch

from tpu_sync.frameworks.torch import _tpu_raiden_torch
from tpu_sync.solutions.mooncake import mooncake_provider_test_util as test_util
from tpu_sync.solutions.mooncake.torch import mooncake_provider

_impl = _tpu_raiden_torch.mooncake_provider


def _host_buffer(size: int, misalignment: int) -> np.ndarray:
  """A uint8 host buffer whose address is `misalignment` past a 64-byte one."""
  raw = np.empty(size + 64 + misalignment, np.uint8)
  start = -raw.ctypes.data % 64 + misalignment
  return raw[start : start + size]


def _wait_for(predicate, timeout_s: float = 30.0) -> bool:
  """Polls `predicate`: device frees can complete asynchronously."""
  deadline = time.monotonic() + timeout_s
  while time.monotonic() < deadline:
    if predicate():
      return True
    time.sleep(0.05)
  return predicate()


def _random(
    shape: tuple[int, ...], dtype: torch.dtype, seed: int
) -> torch.Tensor:
  """A CPU tensor of values in [-100, 100)."""
  generator = torch.Generator().manual_seed(seed)
  return (torch.rand(shape, generator=generator) * 200 - 100).to(dtype)


class MooncakeProviderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu:0")
    self.table = test_util.load_table(mooncake_provider.library_path())

  def _export(self, tensor: torch.Tensor) -> int:
    handles = mooncake_provider.export_shards(tensor)
    self.assertLen(handles, 1)
    return handles[0]

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

  def _contents(self, handle: int, like: torch.Tensor) -> torch.Tensor:
    """What `handle`'s buffer holds, as a CPU tensor shaped like `like`.

    Moves the buffer's on-device bytes through the host into a fresh tensor,
    so that it does not depend on the device layout.
    """
    restored = torch.zeros(like.shape, dtype=like.dtype, device=self.device)
    restored_handle = self._export(restored)
    host = np.empty(self.table.size(handle), np.uint8)
    self._copy_to_host(handle, 0, host)
    self._copy_from_host(restored_handle, 0, host)
    self.table.release(restored_handle)
    return restored.cpu()

  def test_library_is_the_loaded_extension(self):
    path = mooncake_provider.library_path()
    self.assertTrue(os.path.isfile(path), path)
    # RTLD_NOLOAD succeeds only if this exact file is already mapped, i.e.
    # importing the provider loaded it and dlopen(path) returns that image.
    ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
    self.assertEqual(self.table.name, b"tpu-sync")

    # State is shared too: a handle exported through Python and released
    # through the dlopen()ed table is gone from Python's count.
    before = _impl.live_buffer_count()
    handle = self._export(torch.zeros((8, 128), device=self.device))
    self.assertEqual(_impl.live_buffer_count(), before + 1)
    self.table.release(handle)
    self.assertEqual(_impl.live_buffer_count(), before)

  # Shapes that get tiled, padded TPU layouts, and host pointers that are
  # DMA-aligned or not (the latter take the bounce path).
  @parameterized.product(
      shape_and_dtype=(
          ((16, 256), torch.float32),
          ((4, 3, 1000), torch.bfloat16),
          ((8, 8, 130), torch.int8),
      ),
      host_misalignment=(0, 1),
  )
  def test_round_trip_through_host_restores_tensor(
      self, shape_and_dtype, host_misalignment
  ):
    shape, dtype = shape_and_dtype
    expected = _random(shape, dtype, seed=1)
    src = expected.to(self.device)
    dst = torch.zeros(shape, dtype=dtype, device=self.device)
    src_handle = self._export(src)
    dst_handle = self._export(dst)

    size = self.table.size(src_handle)
    self.assertGreaterEqual(size, src.nbytes)
    self.assertEqual(self.table.size(dst_handle), size)
    self.assertEqual(self.table.device_index(src_handle), src.device.index)
    # Both live on one device, whose node is published or -1.
    self.assertEqual(
        self.table.numa_node(dst_handle), self.table.numa_node(src_handle)
    )
    self.assertGreaterEqual(self.table.numa_node(src_handle), -1)

    # Several chunks, as TENT stages a buffer in pieces. Device offsets must
    # be multiples of 32 bytes; host pointers need not be.
    host = _host_buffer(size, host_misalignment)
    chunk = (size // 3 // 32 + 1) * 32
    for offset in range(0, size, chunk):
      self._copy_to_host(src_handle, offset, host[offset : offset + chunk])
    for offset in range(0, size, chunk):
      self._copy_from_host(dst_handle, offset, host[offset : offset + chunk])

    self.table.release(src_handle)
    self.table.release(dst_handle)
    self.assertTrue(torch.equal(dst.cpu(), expected))

  def test_concurrent_copies_on_one_handle(self):
    expected = _random((256, 1024), torch.float32, seed=2)
    src = expected.to(self.device)
    dst = torch.zeros_like(src)
    src_handle = self._export(src)
    dst_handle = self._export(dst)

    # ctypes releases the GIL around each call, like Mooncake's C++ threads.
    size = self.table.size(src_handle)
    chunk = 32 * 1024
    jobs = [
        (offset, min(chunk, size - offset)) for offset in range(0, size, chunk)
    ]

    def move(job):
      offset, length = job
      host = np.empty(length, np.uint8)
      self._copy_to_host(src_handle, offset, host)
      self._copy_from_host(dst_handle, offset, host)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
      list(pool.map(move, jobs))
      # Mooncake releases handles on its own threads too.
      list(pool.map(self.table.release, (src_handle, dst_handle)))
    self.assertTrue(torch.equal(dst.cpu(), expected))

  def test_misaligned_device_offset_fails_cleanly(self):
    handle = self._export(torch.zeros((8, 128), device=self.device))
    host = _host_buffer(64, 0)
    ptr = host.ctypes.data_as(ctypes.c_void_p)
    self.assertNotEqual(self.table.copy_to_host(handle, 1, ptr, 32), 0)
    self.assertNotEqual(self.table.copy_from_host(handle, 4, ptr, 32), 0)
    self.assertEqual(self.table.copy_to_host(handle, 32, ptr, 32), 0)
    self.table.release(handle)

  def test_dma_map_round_trip(self):
    expected = _random((1024, 1024), torch.float32, seed=7)  # 4 MiB.
    src = expected.to(self.device)
    dst = torch.zeros_like(src)
    src_handle = self._export(src)
    dst_handle = self._export(dst)
    size = self.table.size(src_handle)
    pool = test_util.staging_pool(size)
    ptr = pool.ctypes.data
    before = _impl.live_dma_mapping_count()

    # The export gave the provider TorchTPU's PJRT client to map with.
    self.assertEqual(self.table.dma_map(ptr, pool.nbytes), 0)
    try:
      self.assertEqual(_impl.live_dma_mapping_count(), before + 1)
      self._copy_to_host(src_handle, 0, pool[:size])
      self._copy_from_host(dst_handle, 0, pool[:size])
    finally:
      self.assertEqual(self.table.dma_unmap(ptr), 0)
    self.assertEqual(_impl.live_dma_mapping_count(), before)

    self.table.release(src_handle)
    self.table.release(dst_handle)
    self.assertTrue(torch.equal(dst.cpu(), expected))

  def test_handle_keeps_hbm_alive_until_released(self):
    shape = (1024, 2048)  # 8 MiB of float32.
    expected = _random(shape, torch.float32, seed=3)
    tensor = expected.to(self.device)
    nbytes = tensor.nbytes
    # Allocated up front, so that nothing else is allocated or freed while the
    # memory in use is compared below.
    restored = torch.zeros(shape, device=self.device)
    restored_handle = self._export(restored)
    handle = self._export(tensor)
    torch.accelerator.synchronize()
    in_use_with_tensor = torch.accelerator.memory_allocated()

    del tensor
    gc.collect()
    # Tempt the allocator to reuse the memory if the handle did not hold it.
    decoy = _random(shape, torch.float32, seed=4).to(self.device)
    torch.accelerator.synchronize()
    self.assertGreaterEqual(
        torch.accelerator.memory_allocated(), in_use_with_tensor + decoy.nbytes
    )

    # The handle's bytes are the original ones, not the decoy's.
    host = np.empty(self.table.size(handle), np.uint8)
    self._copy_to_host(handle, 0, host)
    self._copy_from_host(restored_handle, 0, host)
    self.assertTrue(torch.equal(restored.cpu(), expected))

    in_use_before_release = torch.accelerator.memory_allocated()
    self.table.release(handle)
    self.assertTrue(
        _wait_for(
            lambda: torch.accelerator.memory_allocated()
            <= in_use_before_release - nbytes
        ),
        "releasing the handle did not free the deleted tensor's HBM",
    )
    self.table.release(restored_handle)
    del decoy, restored

  # A clone of a view that folds leading dimensions is backed by the original
  # [E, F, D] buffer (torch_tpu_utils.h), which holds the bytes where an
  # [E * F, D] buffer would, so its handle moves them like any other.
  def test_round_trip_of_tensor_folding_its_buffer(self):
    stack = _random((8, 64, 256), torch.float32, seed=8)
    expected = stack.reshape(8 * 64, 256)
    src = stack.to(self.device).reshape(8 * 64, 256).clone()
    dst = torch.zeros(8 * 64, 256, device=self.device)
    src_handle = self._export(src)
    dst_handle = self._export(dst)
    size = self.table.size(src_handle)
    self.assertEqual(self.table.size(dst_handle), size)

    host = np.empty(size, np.uint8)
    self._copy_to_host(src_handle, 0, host)
    self._copy_from_host(dst_handle, 0, host)
    self.table.release(src_handle)
    self.table.release(dst_handle)
    self.assertTrue(torch.equal(dst.cpu(), expected))

  # TorchTPU reuses the tensor's buffer for some in-place ops and moves the
  # tensor to a new buffer, which only a new export reaches, for others. These
  # are the examples that export_shards names as current behavior, so a failure
  # here means that TorchTPU changed and the docstring must too.
  @parameterized.named_parameters(
      dict(testcase_name="add_", op=lambda t: t.add_(1), in_place=True),
      dict(
          testcase_name="index_copy_",
          op=lambda t: t.index_copy_(
              0,
              torch.arange(8, device=t.device),
              torch.ones((8, t.shape[1]), device=t.device),
          ),
          in_place=True,
      ),
      dict(
          testcase_name="index_assignment",
          op=lambda t: t.__setitem__(
              torch.arange(8, device=t.device),
              torch.ones((8, t.shape[1]), device=t.device),
          ),
          in_place=True,
      ),
      dict(testcase_name="fill_", op=lambda t: t.fill_(1), in_place=False),
      dict(
          testcase_name="copy_",
          op=lambda t: t.copy_(torch.ones_like(t)),
          in_place=False,
      ),
      dict(
          testcase_name="slice_assignment",
          op=lambda t: t.__setitem__(slice(0, 8), 1),
          in_place=False,
      ),
  )
  def test_in_place_op_after_export(self, op, in_place):
    # Not zeros, which _contents starts its copy from.
    tensor = torch.full((1024, 1024), 2.0, device=self.device)
    before = tensor.cpu()
    handle = self._export(tensor)
    op(tensor)
    after = tensor.cpu()
    self.assertFalse(torch.equal(after, before))

    self.assertTrue(
        torch.equal(
            self._contents(handle, tensor), after if in_place else before
        )
    )
    new_handle = self._export(tensor)
    self.assertTrue(torch.equal(self._contents(new_handle, tensor), after))
    self.table.release(handle)
    self.table.release(new_handle)

  def test_rejects_unsupported_tensors(self):
    tensor = _random((8, 128), torch.float32, seed=5).to(self.device)
    # An f32 [4, 5, 256] buffer cannot hold its 5-row dimension the way a
    # [20, 256] buffer does: 5 rows are not a whole (8, 128) tile.
    unaligned_stack = _random((4, 5, 256), torch.float32, seed=6).to(
        self.device
    )
    before = _impl.live_buffer_count()
    with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
      mooncake_provider.export_shards(np.zeros((8, 128), np.float32))
    for name, unsupported, message in (
        ("on CPU", torch.zeros((8, 128)), "must reside on TPU"),
        ("not contiguous", tensor.t(), "must be contiguous"),
        ("an offset view", tensor[1:], "must start at storage offset 0"),
        ("a partial view", tensor[:4], "does not match its base storage"),
        (
            "a fold with another layout",
            unaligned_stack.view(20, 256),
            r"does not hold its bytes as a \[20, 256\] buffer would",
        ),
    ):
      with self.subTest(name):
        with self.assertRaisesRegex(ValueError, message):
          mooncake_provider.export_shards(unsupported)
    self.assertEqual(_impl.live_buffer_count(), before)


if __name__ == "__main__":
  absltest.main()
