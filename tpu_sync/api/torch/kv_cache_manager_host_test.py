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

"""Host-only tests for the Python pool-admission surface."""

import unittest

from tpu_sync.api.torch import kv_cache_manager
from tpu_sync.api.torch import pool_layout


class KVCacheManagerHostTest(unittest.TestCase):

  def test_aliased_pool_admits_last_live_region_without_trailing_padding(self):
    manager = kv_cache_manager.KVCacheManager.create_host_only_for_testing(
        num_layers=1,
        num_shards=1,
        slice_byte_size=128,
        node_id=7,
        host_blocks=2,
        parallelism=1,
    )
    pool = pool_layout.PoolSpec(
        tag="aliased",
        storage_index=0,
        base_offset_bytes=32,
        block_stride_bytes=128,
        num_blocks=2,
        regions=(
            pool_layout.RegionSpec(
                name="payload",
                offset_bytes=0,
                stride_bytes=64,
                unit_bytes=32,
                num_units=2,
            ),
        ),
    )

    self.assertEqual(pool.storage_extent_end_bytes, 256)
    pool.validate(storage_bytes=256)
    self.assertTrue(manager.register_pools((pool,))["admitted"])
    first = manager.get_block_ref(pool_idx=0, block_id=0)
    last = manager.get_block_ref(pool_idx=0, block_id=1)
    self.assertEqual(last["ptr"], first["ptr"] + 128)

  def test_register_refs_tags_and_admission_summary(self):
    manager = kv_cache_manager.KVCacheManager.create_host_only_for_testing(
        num_layers=1,
        num_shards=1,
        slice_byte_size=256,
        node_id=7,
        host_blocks=4,
        parallelism=1,
    )
    first_pool = pool_layout.PoolSpec(
        tag="kind_a",
        storage_index=0,
        base_offset_bytes=32,
        block_stride_bytes=128,
        num_blocks=4,
        regions=(
            pool_layout.RegionSpec(
                name="payload",
                offset_bytes=0,
                stride_bytes=128,
                unit_bytes=128,
                num_units=1,
            ),
        ),
        dtype_tag="dtype_a",
    )
    second_pool = pool_layout.PoolSpec(
        tag="kind_b",
        storage_index=0,
        base_offset_bytes=640,
        block_stride_bytes=64,
        num_blocks=4,
        regions=(
            pool_layout.RegionSpec(
                name="payload",
                offset_bytes=0,
                stride_bytes=64,
                unit_bytes=64,
                num_units=1,
            ),
        ),
        dtype_tag="dtype_b",
    )

    summary = manager.register_pools((first_pool, second_pool))

    self.assertEqual(
        summary,
        {
            "admitted": True,
            "pools": 2,
            "storages": 1,
            "tags": {"kind_a": 1, "kind_b": 1},
            "host_staging": {
                "mode": "full",
                "leases": 0,
                "bounded_storage_bytes_per_shard": 0,
                "full_storage_bytes_per_shard": 1024,
                "storages": [{
                    "storage_index": 0,
                    "bounded": False,
                    "stride_bytes": 0,
                    "num_slots": 0,
                    "blocks_per_lease": 0,
                    "host_bytes_per_shard": 1024,
                    "free_slots": 0,
                }],
            },
        },
    )
    self.assertEqual(manager.admission_summary(), summary)
    self.assertEqual(manager.num_pools(), 2)
    self.assertEqual(manager.pool_ids_with_tag("kind_a"), [0])
    self.assertEqual(manager.pool_ids_with_tag("kind_b"), [1])
    self.assertEqual(manager.pool_ids_with_tag("missing"), [])

    first_ref = manager.get_block_ref(pool_idx=0, block_id=0)
    first_ref_last = manager.get_block_ref(pool_idx=0, block_id=3)
    second_ref = manager.get_block_ref(pool_idx=1, block_id=2)
    self.assertEqual(first_ref_last["ptr"], first_ref["ptr"] + 3 * 128)
    self.assertEqual(second_ref["ptr"], first_ref["ptr"] + (640 - 32) + 2 * 64)
    self.assertEqual(first_ref["block_stride_bytes"], 128)
    self.assertEqual(second_ref["block_stride_bytes"], 64)
    self.assertEqual(first_ref["tag"], "kind_a")
    self.assertEqual(second_ref["tag"], "kind_b")
    self.assertEqual(second_ref["dtype_tag"], "dtype_b")
    self.assertEqual(manager.pool_spec(1)["base_offset_bytes"], 640)

    with self.assertRaisesRegex(RuntimeError, "host-only manager"):
      manager.d2h_pool_blocks(pool_idx=0, block_ids=[0])
    with self.assertRaisesRegex(RuntimeError, "host-only manager"):
      manager.h2d_pool_blocks(pool_idx=0, block_ids=[0])


if __name__ == "__main__":
  unittest.main()
