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

"""Contract tests for the thin PyTorch KV cache offloader wrapper."""

import unittest

from tpu_sync.api.torch import kv_cache_offloader


class _FakeNativeOffloader:

  def __init__(self, kv_cache_tensors, page_nbytes):
    self.calls = [("init", kv_cache_tensors, page_nbytes)]
    self.num_layers = len(kv_cache_tensors)
    self.num_blocks = 13
    self.page_nbytes = page_nbytes
    self.is_shared_memory_mapped = False

  def map_shared_memory(self, mapped_address, pool_size_bytes):
    self.calls.append(("map", mapped_address, pool_size_bytes))
    self.is_shared_memory_mapped = True

  def unmap_shared_memory(self):
    self.calls.append(("unmap",))
    self.is_shared_memory_mapped = False

  def h2d(self, block_ids, object_tensors, rank_id):
    self.calls.append(("h2d", block_ids, object_tensors, rank_id))
    return "h2d-future"

  def d2h(self, block_ids, object_tensors, rank_id):
    self.calls.append(("d2h", block_ids, object_tensors, rank_id))
    return "d2h-future"


class _FakeExtension:
  KVCacheOffloader = _FakeNativeOffloader


class KVCacheOffloaderWrapperTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._old_impl = kv_cache_offloader._TORCH_IMPL  # pylint: disable=protected-access
    kv_cache_offloader._TORCH_IMPL = _FakeExtension  # pylint: disable=protected-access

  def tearDown(self):
    kv_cache_offloader._TORCH_IMPL = self._old_impl  # pylint: disable=protected-access
    super().tearDown()

  def test_forwards_full_object_views_and_mapping_contract(self):
    device_tensors = [object(), object()]
    object_tensors = [object(), object()]
    offloader = kv_cache_offloader.KVCacheOffloader(
        (tensor for tensor in device_tensors), page_nbytes=4096
    )

    offloader.map_shared_memory(mapped_address=0x40000000, pool_size_bytes=8192)
    self.assertTrue(offloader.is_shared_memory_mapped)
    self.assertEqual(
        offloader.h2d((3, 5), (tensor for tensor in object_tensors), rank_id=1),
        "h2d-future",
    )
    self.assertEqual(
        offloader.d2h([3, 5], object_tensors, rank_id=0), "d2h-future"
    )
    offloader.unmap_shared_memory()

    self.assertEqual(
        offloader._impl.calls,  # pylint: disable=protected-access
        [
            ("init", device_tensors, 4096),
            ("map", 0x40000000, 8192),
            ("h2d", [3, 5], object_tensors, 1),
            ("d2h", [3, 5], object_tensors, 0),
            ("unmap",),
        ],
    )
    self.assertFalse(offloader.is_shared_memory_mapped)

  def test_exposes_geometry(self):
    offloader = kv_cache_offloader.KVCacheOffloader([object()], 4096)

    self.assertEqual(offloader.num_layers, 1)
    self.assertEqual(offloader.num_blocks, 13)
    self.assertEqual(offloader.page_nbytes, 4096)


if __name__ == "__main__":
  unittest.main()
