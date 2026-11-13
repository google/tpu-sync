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

"""Real-TPU tests for shared memory DMA mapping and object tensor transfers.

These exercise KVCacheManager.experimental_map_shared_memory plus the
object-tensor H2D/D2H overloads against a live PJRT client, including pages
that span several device blocks.

test_object_tensor_bandwidth additionally measures throughput, and
test_multi_block_page_bandwidth compares one DMA per multi-block page against
one DMA per device block for the same bytes. Their threshold is deliberately
loose: it only catches a silent fall-off to staged pageable copies, an order of
magnitude below DMA speed. It is not a regression gate, and scheduling jitter
must not flake it.
"""

import os
import statistics
import time

from absl import flags
from absl.testing import absltest
import torch

from tpu_sync.api.torch import kv_cache_manager
from tpu_sync.api.torch import torch_tpu_common_loader

KVCacheManager = kv_cache_manager.KVCacheManager

_NUM_LAYERS = 2
_NUM_BLOCKS = 4
_NUM_RANKS = 2
# Per-layer device buffer: [num_blocks, ...] float32, so one block per layer is
# 8 * 8 * 128 * 4 = 32 KiB.
_BLOCK_SHAPE = (_NUM_BLOCKS, 8, 8, 128)

# The correctness tests above move 32 KiB per layer, where per-call overhead
# dominates and the measured rate says nothing about DMA bandwidth. The
# bandwidth test uses its own, much larger geometry.
_PERF_LAYERS = flags.DEFINE_integer(
    "perf_layers", 8, "Layers (device buffers) for the bandwidth test."
)
_PERF_BLOCKS = flags.DEFINE_integer(
    "perf_blocks",
    8,
    "Blocks per h2d/d2h call in the bandwidth tests; pages in "
    "test_multi_block_page_bandwidth.",
)
_PERF_BLOCK_MIB = flags.DEFINE_integer(
    "perf_block_mib", 4, "MiB per block per layer in the bandwidth test."
)
_PERF_ITERS = flags.DEFINE_integer(
    "perf_iters", 8, "Timed iterations per direction."
)
_PERF_WARMUP = flags.DEFINE_integer(
    "perf_warmup", 2, "Untimed warmup iterations."
)
_MIN_GBPS = flags.DEFINE_float(
    "min_gbps",
    2.0,
    "Fail below this bandwidth (GB/s, decimal). Loose on purpose: it catches a "
    "fall-off to staged pageable copies, not a drift regression.",
)

# test_multi_block_page_bandwidth mirrors a connector whose block spans
# several device blocks: 512 KiB device blocks grouped into 2 MiB pages.
_PAGE_DEVICE_BLOCKS = 4
# float32, so one device block is 512 * 256 * 4 bytes == 512 KiB.
_PAGE_TEST_BLOCK_SHAPE = (512, 256)


class SharedMemoryDmaTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Registers the "tpu" device name.  Importing torch_tpu is not enough: the
    # privateuse1 rename happens inside torch_tpu's loader, which
    # kv_cache_manager only calls lazily from the manager constructor, i.e.
    # after setUp needs the device.
    torch_tpu_common_loader.load_torch_tpu_common()
    self.device = torch.device("tpu")
    self.page_size = os.sysconf("SC_PAGE_SIZE")

  def _make_manager(self) -> KVCacheManager:
    kv_caches = [
        torch.zeros(_BLOCK_SHAPE, dtype=torch.float32, device=self.device)
        for _ in range(_NUM_LAYERS)
    ]
    return KVCacheManager(
        kv_caches=kv_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=_NUM_BLOCKS,
        num_slots=2,
    )

  def _alloc_pool(self, min_bytes: int):
    """Returns (backing_tensor, pad, pool_addr, pool_size) for a page-aligned pool.

    experimental_map_shared_memory requires both the address and the size to
    be page aligned, so over-allocate by one page and start the pool at the
    first aligned offset inside it.

    Args:
      min_bytes: smallest usable pool size; rounded up to a page multiple.
    """
    pool_size = (
        (min_bytes + self.page_size - 1) // self.page_size
    ) * self.page_size
    backing = torch.empty(pool_size + self.page_size, dtype=torch.int8)
    base = backing.data_ptr()
    pad = (-base) % self.page_size
    return backing, pad, base + pad, pool_size

  def _bench(self, fn) -> float:
    """Returns the median wall time of fn over _PERF_ITERS timed calls."""
    times = []
    for i in range(_PERF_WARMUP.value + _PERF_ITERS.value):
      start = time.perf_counter()
      fn()
      elapsed = time.perf_counter() - start
      if i >= _PERF_WARMUP.value:
        times.append(elapsed)
    return statistics.median(times)

  def _bench_object_transfers(self, manager, block_ids, tensors):
    """Returns median (h2d_s, d2h_s) for one call over all of `tensors`."""
    h2d_s = self._bench(
        lambda: manager.experimental_h2d(block_ids, tensors, 0).wait()
    )
    d2h_s = self._bench(
        lambda: manager.experimental_d2h(block_ids, tensors, 0).wait()
    )
    return h2d_s, d2h_s

  def test_h2d_d2h_roundtrip_through_mapped_pool(self):
    manager = self._make_manager()
    page_nbytes = manager._impl.slice_byte_size  # pylint: disable=protected-access
    obj_nbytes = _NUM_RANKS * _NUM_LAYERS * page_nbytes

    # src and dst object tensors both live inside the registered pool, which is
    # how a caller is expected to use this API.
    backing, pad, pool_addr, pool_size = self._alloc_pool(2 * obj_nbytes)
    src = backing[pad : pad + obj_nbytes].view(
        _NUM_RANKS, _NUM_LAYERS, page_nbytes
    )
    dst = backing[pad + obj_nbytes : pad + 2 * obj_nbytes].view(
        _NUM_RANKS, _NUM_LAYERS, page_nbytes
    )
    src[0].fill_(0x42)
    src[1].fill_(0x77)
    dst.fill_(0)

    manager.experimental_map_shared_memory(pool_addr, pool_size)
    self.assertTrue(manager.experimental_is_shared_memory_mapped)

    try:
      # Rank 0 -> block 3 -> back again.
      manager.experimental_h2d([3], [src], 0).wait()
      manager.experimental_d2h([3], [dst], 0).wait()
      self.assertTrue(
          torch.equal(dst[0], src[0]),
          "rank 0 did not round-trip through the device",
      )

      # Rank 1 lands in a different block and must not disturb rank 0.
      manager.experimental_h2d([1], [src], 1).wait()
      manager.experimental_d2h([1], [dst], 1).wait()
      self.assertTrue(
          torch.equal(dst[1], src[1]),
          "rank 1 did not round-trip through the device",
      )
      self.assertTrue(torch.equal(dst[0], src[0]))
    finally:
      manager.experimental_unmap_shared_memory()

    self.assertFalse(manager.experimental_is_shared_memory_mapped)
    del backing

  def test_tensor_outside_mapped_pool_is_rejected(self):
    manager = self._make_manager()
    page_nbytes = manager._impl.slice_byte_size  # pylint: disable=protected-access
    obj_nbytes = _NUM_RANKS * _NUM_LAYERS * page_nbytes

    backing, pad, pool_addr, pool_size = self._alloc_pool(obj_nbytes)
    inside = backing[pad : pad + obj_nbytes].view(
        _NUM_RANKS, _NUM_LAYERS, page_nbytes
    )
    # Separate allocation, so it cannot overlap the pool.
    outside = torch.zeros(
        (_NUM_RANKS, _NUM_LAYERS, page_nbytes), dtype=torch.int8
    )

    manager.experimental_map_shared_memory(pool_addr, pool_size)
    try:
      # Sanity: the in-pool tensor is accepted, so a rejection below is really
      # about the address and not about some other argument.
      manager.experimental_h2d([0], [inside], 0).wait()

      with self.assertRaisesRegex(Exception, "outside the DMA mapped pool"):
        manager.experimental_h2d([0], [outside], 0)
      with self.assertRaisesRegex(Exception, "outside the DMA mapped pool"):
        manager.experimental_d2h([0], [outside], 0)
    finally:
      manager.experimental_unmap_shared_memory()
    del backing

  def test_transfers_refused_when_not_mapped(self):
    manager = self._make_manager()
    page_nbytes = manager._impl.slice_byte_size  # pylint: disable=protected-access
    obj_nbytes = _NUM_RANKS * _NUM_LAYERS * page_nbytes

    backing, pad, pool_addr, pool_size = self._alloc_pool(obj_nbytes)
    obj = backing[pad : pad + obj_nbytes].view(
        _NUM_RANKS, _NUM_LAYERS, page_nbytes
    )

    # Before mapping.
    self.assertFalse(manager.experimental_is_shared_memory_mapped)
    with self.assertRaisesRegex(Exception, "must be mapped"):
      manager.experimental_h2d([0], [obj], 0)

    # And again after the pool is released.
    manager.experimental_map_shared_memory(pool_addr, pool_size)
    manager.experimental_h2d([0], [obj], 0).wait()
    manager.experimental_unmap_shared_memory()

    self.assertFalse(manager.experimental_is_shared_memory_mapped)
    with self.assertRaisesRegex(Exception, "must be mapped"):
      manager.experimental_d2h([0], [obj], 0)
    del backing

  def test_remap_after_unmap(self):
    manager = self._make_manager()
    page_nbytes = manager._impl.slice_byte_size  # pylint: disable=protected-access
    obj_nbytes = _NUM_RANKS * _NUM_LAYERS * page_nbytes

    backing, pad, pool_addr, pool_size = self._alloc_pool(obj_nbytes)
    obj = backing[pad : pad + obj_nbytes].view(
        _NUM_RANKS, _NUM_LAYERS, page_nbytes
    )
    obj[0].fill_(0x5A)

    for _ in range(2):
      manager.experimental_map_shared_memory(pool_addr, pool_size)
      self.assertTrue(manager.experimental_is_shared_memory_mapped)
      manager.experimental_h2d([2], [obj], 0).wait()
      manager.experimental_d2h([2], [obj], 0).wait()
      manager.experimental_unmap_shared_memory()
      self.assertFalse(manager.experimental_is_shared_memory_mapped)
    del backing

  def test_multi_block_page_roundtrip(self):
    manager = self._make_manager()
    # Let the zero fill of the device KV caches land before reading them.
    torch.accelerator.synchronize()
    block_nbytes = manager._impl.slice_byte_size  # pylint: disable=protected-access
    # _NUM_BLOCKS device blocks per layer, so two-block pages leave two pages.
    page_nbytes = 2 * block_nbytes
    page_shape = (_NUM_RANKS, _NUM_LAYERS, page_nbytes)
    block_shape = (_NUM_RANKS, _NUM_LAYERS, block_nbytes)
    page_obj_nbytes = _NUM_RANKS * _NUM_LAYERS * page_nbytes
    block_obj_nbytes = _NUM_RANKS * _NUM_LAYERS * block_nbytes

    # A source and a destination page, then two one-block objects for reading
    # device blocks back individually. All of them live inside the pool.
    backing, pad, pool_addr, pool_size = self._alloc_pool(
        2 * page_obj_nbytes + 2 * block_obj_nbytes
    )
    backing.zero_()
    start = pad
    src = backing[start : start + page_obj_nbytes].view(page_shape)
    start += page_obj_nbytes
    dst = backing[start : start + page_obj_nbytes].view(page_shape)
    start += page_obj_nbytes
    first = backing[start : start + block_obj_nbytes].view(block_shape)
    start += block_obj_nbytes
    second = backing[start : start + block_obj_nbytes].view(block_shape)

    # The page's first device block is 0x11 and its second 0x22, so the
    # readback can tell where each half landed.
    src[0, :, :block_nbytes].fill_(0x11)
    src[0, :, block_nbytes:].fill_(0x22)

    manager.experimental_map_shared_memory(
        pool_addr, pool_size, page_nbytes=page_nbytes
    )
    try:
      # Page 1 is device blocks 2 and 3.
      manager.experimental_h2d([1], [src], 0).wait()
      manager.experimental_d2h([1], [dst], 0).wait()
      self.assertTrue(
          torch.equal(dst[0], src[0]),
          "page 1 did not round-trip through the device",
      )

      # The mapping pins the page size, so a one-block object is refused
      # instead of block id 1 silently meaning device block 1.
      with self.assertRaisesRegex(ValueError, "does not match"):
        manager.experimental_h2d([1], [first], 0)
    finally:
      manager.experimental_unmap_shared_memory()

    # Re-map with the default one-block page and read device blocks 1 to 3
    # back one at a time.
    manager.experimental_map_shared_memory(pool_addr, pool_size)
    try:
      manager.experimental_d2h([2, 3], [first, second], 0).wait()
      self.assertTrue(
          torch.all(first[0] == 0x11).item(),
          "device block 2 does not hold the page's first half",
      )
      self.assertTrue(
          torch.all(second[0] == 0x22).item(),
          "device block 3 does not hold the page's second half",
      )
      manager.experimental_d2h([1], [first], 0).wait()
      self.assertEqual(
          torch.count_nonzero(first[0]).item(),
          0,
          "device block 1, below page 1, was written",
      )
    finally:
      manager.experimental_unmap_shared_memory()
    del backing

  def test_object_tensor_bandwidth(self):
    num_layers = _PERF_LAYERS.value
    num_blocks = _PERF_BLOCKS.value
    # float32, so 1024 * 256 * 4 == 1 MiB per unit of dim 1.
    block_shape = (num_blocks, _PERF_BLOCK_MIB.value, 1024, 256)

    kv_caches = [
        torch.zeros(block_shape, dtype=torch.float32, device=self.device)
        for _ in range(num_layers)
    ]
    torch.accelerator.synchronize()
    manager = KVCacheManager(
        kv_caches=kv_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
    )
    page_nbytes = manager._impl.slice_byte_size  # pylint: disable=protected-access

    # One object tensor per block, each [num_ranks=1, num_layers, page_nbytes].
    # Only rank 0's slice is transferred, so that is the whole tensor here.
    obj_nbytes = num_layers * page_nbytes
    backing, pad, pool_addr, pool_size = self._alloc_pool(
        num_blocks * obj_nbytes
    )
    tensors = [
        backing[pad + i * obj_nbytes : pad + (i + 1) * obj_nbytes].view(
            1, num_layers, page_nbytes
        )
        for i in range(num_blocks)
    ]
    block_ids = list(range(num_blocks))
    payload_bytes = num_blocks * obj_nbytes

    manager.experimental_map_shared_memory(pool_addr, pool_size)
    try:
      h2d_s, d2h_s = self._bench_object_transfers(manager, block_ids, tensors)
    finally:
      manager.experimental_unmap_shared_memory()

    mib = payload_bytes / (1024.0 * 1024.0)
    print(
        f"[object-tensor perf] layers={num_layers} blocks={num_blocks} "
        f"page={page_nbytes / (1024.0 * 1024.0):.3f} MiB "
        f"payload={mib:.1f} MiB iters={_PERF_ITERS.value}"
    )
    rates = {}
    for name, seconds in (("H2D", h2d_s), ("D2H", d2h_s)):
      gbps = payload_bytes / seconds / 1e9
      gibps = payload_bytes / seconds / (1024.0**3)
      rates[name] = gbps
      print(
          f"[object-tensor perf] {name} median {seconds * 1e3:.2f} ms -> "
          f"{gbps:.1f} GB/s ({gibps:.1f} GiB/s)"
      )

    self.assertGreater(rates["H2D"], _MIN_GBPS.value)
    self.assertGreater(rates["D2H"], _MIN_GBPS.value)
    del backing

  def test_multi_block_page_bandwidth(self):
    num_layers = _PERF_LAYERS.value
    num_pages = _PERF_BLOCKS.value
    kv_caches = [
        torch.zeros(
            (num_pages * _PAGE_DEVICE_BLOCKS,) + _PAGE_TEST_BLOCK_SHAPE,
            dtype=torch.float32,
            device=self.device,
        )
        for _ in range(num_layers)
    ]
    torch.accelerator.synchronize()
    # Staging slots are unused by object transfers; keep them minimal.
    manager = KVCacheManager(
        kv_caches=kv_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=1,
        num_slots=1,
    )
    block_nbytes = manager._impl.slice_byte_size  # pylint: disable=protected-access
    page_nbytes = _PAGE_DEVICE_BLOCKS * block_nbytes
    payload_bytes = num_pages * num_layers * page_nbytes
    backing, pad, pool_addr, pool_size = self._alloc_pool(payload_bytes)

    print(
        f"[multi-block page perf] layers={num_layers} pages={num_pages} "
        f"page={page_nbytes / (1024.0 * 1024.0):.3f} MiB "
        f"device_block={block_nbytes / 1024.0:.0f} KiB "
        f"payload={payload_bytes / (1024.0 * 1024.0):.1f} MiB "
        f"iters={_PERF_ITERS.value}"
    )
    # Both cases move the same bytes between the same pool and device range;
    # only the transfer unit differs. Each object is [1, num_layers, unit] and
    # object i holds unit i of every layer, so the device-block case needs
    # _PAGE_DEVICE_BLOCKS times as many block ids and DMAs.
    rates = {}
    for name, unit_nbytes, mapped_page_nbytes in (
        ("page", page_nbytes, page_nbytes),
        ("device_block", block_nbytes, None),
    ):
      obj_nbytes = num_layers * unit_nbytes
      tensors = [
          backing[pad + i * obj_nbytes : pad + (i + 1) * obj_nbytes].view(
              1, num_layers, unit_nbytes
          )
          for i in range(payload_bytes // obj_nbytes)
      ]
      manager.experimental_map_shared_memory(
          pool_addr, pool_size, page_nbytes=mapped_page_nbytes
      )
      try:
        h2d_s, d2h_s = self._bench_object_transfers(
            manager, list(range(len(tensors))), tensors
        )
      finally:
        manager.experimental_unmap_shared_memory()
      for direction, seconds in (("H2D", h2d_s), ("D2H", d2h_s)):
        gbps = payload_bytes / seconds / 1e9
        gibps = payload_bytes / seconds / (1024.0**3)
        rates[f"{name} {direction}"] = gbps
        print(
            f"[multi-block page perf] {name} ids={len(tensors)} {direction} "
            f"median {seconds * 1e3:.2f} ms -> {gbps:.1f} GB/s "
            f"({gibps:.1f} GiB/s)"
        )

    print(
        "[multi-block page perf] page/device_block "
        f"H2D {rates['page H2D'] / rates['device_block H2D']:.2f}x "
        f"D2H {rates['page D2H'] / rates['device_block D2H']:.2f}x"
    )
    for case, gbps in rates.items():
      self.assertGreater(gbps, _MIN_GBPS.value, case)
    del backing


if __name__ == "__main__":
  absltest.main()
