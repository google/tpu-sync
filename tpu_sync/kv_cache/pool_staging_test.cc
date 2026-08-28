// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Bounded host staging for the pool-reshard path: arena sizing, per-uuid
// leases, host addressing through the lease, capacity enforcement and the
// full-mirror fallbacks. Host-only manager with device-backed storages declared
// via physical sizes (no TPU needed).

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/block_transport_delegate.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

class StagingTestManager : public KVCacheManagerBase {
 public:
  StagingTestManager(size_t num_layers, size_t num_shards,
                     size_t slice_byte_size, int host_blocks)
      : KVCacheManagerBase(num_layers, num_shards, slice_byte_size,
                           /*local_port=*/std::nullopt,
                           std::make_optional(host_blocks)) {
    buffer_holds_.resize(num_layers);
    for (size_t l = 0; l < num_layers; ++l) {
      buffer_holds_[l].holds.resize(num_shards);
      for (size_t sh = 0; sh < num_shards; ++sh) {
        layers_[l].shards[sh].device_size = host_blocks * bytes_per_block();
      }
    }
  }

  // Declares storage `layer_idx` device-backed with `physical_size` bytes.
  void SetDeviceBacked(size_t layer_idx, size_t physical_size) {
    buffer_holds_[layer_idx].physical_size = physical_size;
    major_dim_size_ = 1;
    for (auto& shard : layers_[layer_idx].shards) {
      shard.device_size = physical_size;
    }
  }
};

PoolSpec DensePool(const std::string& tag, size_t storage_index,
                   int64_t base_offset, int64_t stride, int64_t num_blocks,
                   int64_t staging_blocks_per_request) {
  return PoolSpec{
      .tag = tag,
      .storage_index = storage_index,
      .base_offset_bytes = base_offset,
      .block_stride_bytes = stride,
      .num_blocks = num_blocks,
      .regions = {RegionSpec{
          .name = "block",
          .offset_bytes = 0,
          .stride_bytes = stride,
          .unit_bytes = stride,
          .num_units = 1,
          .units_per_stride = 1,
      }},
      .dtype_tag = "dtype_a",
      .staging_blocks_per_request = staging_blocks_per_request,
  };
}

// Pool with one strided live region per block: live [0, 16) and [32, 48)
// within each 64-byte block.
PoolSpec StridedPool(const std::string& tag, size_t storage_index,
                     int64_t base_offset, int64_t stride, int64_t num_blocks,
                     int64_t staging_blocks_per_request) {
  return PoolSpec{
      .tag = tag,
      .storage_index = storage_index,
      .base_offset_bytes = base_offset,
      .block_stride_bytes = stride,
      .num_blocks = num_blocks,
      .regions = {RegionSpec{
          .name = "payload",
          .offset_bytes = 0,
          .stride_bytes = 32,
          .unit_bytes = 16,
          .num_units = 2,
          .units_per_stride = 1,
      }},
      .dtype_tag = "dtype_a",
      .staging_blocks_per_request = staging_blocks_per_request,
  };
}

TEST(PoolStagingTest, BoundedArenaLeasesAndAddressing) {
  constexpr int64_t kStride = 64;
  constexpr int64_t kNumBlocks = 16;
  StagingTestManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/kStride, /*host_blocks=*/1);
  manager.SetDeviceBacked(0, kStride * kNumBlocks);
  // 2 leases x 2 blocks per lease = 4 slots, well under the 16-block pool.
  absl::Status status =
      manager.RegisterPools({DensePool("fa", 0, 0, kStride, kNumBlocks,
                                       /*staging_blocks_per_request=*/2)},
                            /*staging_leases=*/2);
  ASSERT_TRUE(status.ok()) << status.ToString();
  EXPECT_TRUE(manager.PoolStorageStagingBounded(0));
  const size_t host_size = manager.GetHostSize(/*layer_idx=*/0, 0);
  EXPECT_GE(host_size, static_cast<size_t>(4 * kStride));
  EXPECT_LT(host_size, static_cast<size_t>(kNumBlocks * kStride));
  auto summary = manager.PoolStagingSummary();
  ASSERT_EQ(summary.size(), 1u);
  EXPECT_TRUE(summary[0].bounded);
  EXPECT_EQ(summary[0].num_slots, 4);
  EXPECT_EQ(summary[0].blocks_per_lease, 2);
  EXPECT_EQ(summary[0].free_slots, 4);
  // The transport's bounds span is the whole arena.
  EXPECT_EQ(manager.GetBlockArrayHostSize(/*block_array_idx=*/0, 0),
            static_cast<size_t>(4 * kStride));

  // No standing host residency on a bounded storage.
  EXPECT_EQ(manager.GetPoolBlockRef(0, 0, 5).status().code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(manager.GetBlockHostPointer(/*layer_idx=*/0, 0, /*block_id=*/5),
            nullptr);

  // Lease blocks 5 and 9 for uuid 7: slot 0 then slot 1 (LIFO free list).
  status = manager.AcquirePoolStagingLease(/*uuid=*/7, /*storage_idx=*/0,
                                           std::vector<int64_t>{5, 9},
                                           absl::Milliseconds(50));
  ASSERT_TRUE(status.ok()) << status.ToString();
  // Re-acquiring the same ids is a no-op; adding one more takes slot 2.
  status = manager.AcquirePoolStagingLease(7, 0, std::vector<int64_t>{9, 11},
                                           absl::Milliseconds(50));
  ASSERT_TRUE(status.ok()) << status.ToString();
  EXPECT_EQ(manager.PoolStagingSummary()[0].free_slots, 1);

  // Receiver-side chunk resolution lands dst block 9 at its slot, not at
  // block 9 * stride (which is outside the arena).
  tpu_sync::rpc::StartTransferRequest request;
  request.set_uuid(7);
  request.set_is_sender(false);
  auto* entry = (*request.mutable_shard_push_schedules())[0].add_entries();
  entry->set_dst_peer("127.0.0.1:1");
  entry->set_dst_shard_idx(0);
  entry->set_dst_block_id(9);
  entry->set_src_block_id(3);
  entry->set_dst_offset_bytes(8);
  entry->set_src_offset_bytes(0);
  entry->set_size_bytes(16);
  status = manager.RegisterActivePlan(7, request, /*is_sender=*/false);
  ASSERT_TRUE(status.ok()) << status.ToString();
  uint8_t* host_base = manager.GetHostPointer(/*layer_idx=*/0, 0);
  std::vector<transport::BlockChunk> chunks = manager.GetBlockChunks(
      /*layer_idx=*/0, /*shard_idx=*/0, std::vector<int64_t>{9},
      /*total_bytes=*/16, /*uuid=*/7, /*sender_node_id=*/0);
  ASSERT_EQ(chunks.size(), 1u);
  EXPECT_EQ(chunks[0].size, 16u);
  EXPECT_EQ(chunks[0].ptr, host_base + 1 * kStride + 8);  // slot 1
  // A block outside the lease resolves to nothing.
  EXPECT_TRUE(manager
                  .GetBlockChunks(0, 0, std::vector<int64_t>{4}, 16, 7,
                                  /*sender_node_id=*/0)
                  .empty());

  // Another transfer needing more slots than are free waits, then fails.
  status = manager.AcquirePoolStagingLease(
      /*uuid=*/8, 0, std::vector<int64_t>{1, 2}, absl::Milliseconds(20));
  EXPECT_EQ(status.code(), absl::StatusCode::kResourceExhausted)
      << status.ToString();
  // More blocks than the whole arena is rejected outright.
  status = manager.AcquirePoolStagingLease(/*uuid=*/9, 0,
                                           std::vector<int64_t>{1, 2, 3, 4, 6},
                                           absl::Milliseconds(20));
  EXPECT_EQ(status.code(), absl::StatusCode::kResourceExhausted);

  // Releasing uuid 7 returns its three slots; uuid 8 now fits.
  ASSERT_TRUE(manager.UnregisterActivePlan(7).ok());
  manager.ReleasePoolStagingLeases(7);
  EXPECT_EQ(manager.PoolStagingSummary()[0].free_slots, 4);
  status = manager.AcquirePoolStagingLease(8, 0, std::vector<int64_t>{1, 2},
                                           absl::Milliseconds(20));
  ASSERT_TRUE(status.ok()) << status.ToString();
  EXPECT_EQ(manager.PoolStagingSummary()[0].free_slots, 2);
  manager.ReleasePoolStagingLeases(8);
  // Releasing an unknown uuid is a no-op.
  manager.ReleasePoolStagingLeases(12345);
  EXPECT_EQ(manager.PoolStagingSummary()[0].free_slots, 4);
}

// Without hints (or without leases) the storage keeps the full mirror and
// the identity addressing, i.e. the pre-existing behaviour.
TEST(PoolStagingTest, FallsBackToFullMirrorWithoutHintsOrLeases) {
  constexpr int64_t kStride = 64;
  constexpr int64_t kNumBlocks = 16;
  for (int variant = 0; variant < 2; ++variant) {
    StagingTestManager manager(/*num_layers=*/1, /*num_shards=*/1,
                               /*slice_byte_size=*/kStride, /*host_blocks=*/1);
    manager.SetDeviceBacked(0, kStride * kNumBlocks);
    PoolSpec pool = DensePool("fa", 0, 0, kStride, kNumBlocks,
                              /*staging_blocks_per_request=*/
                              variant == 0 ? 0 : 2);
    absl::Status status =
        manager.RegisterPools({pool}, /*staging_leases=*/variant == 0 ? 2 : 0);
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_FALSE(manager.PoolStorageStagingBounded(0));
    EXPECT_GE(manager.GetHostSize(0, 0),
              static_cast<size_t>(kStride * kNumBlocks));
    auto ref = manager.GetPoolBlockRef(0, 0, 9);
    ASSERT_TRUE(ref.ok()) << ref.status().ToString();
    EXPECT_EQ(ref->ptr, manager.GetHostPointer(0, 0) + 9 * kStride);
    // Leases are no-ops on unbounded storages.
    EXPECT_TRUE(manager
                    .AcquirePoolStagingLease(1, 0, std::vector<int64_t>{9},
                                             absl::Milliseconds(1))
                    .ok());
    EXPECT_FALSE(manager.PoolStagingSummary()[0].bounded);
  }
}

// An arena that would be at least as large as the pool keeps the identity
// mapping (small pools); pools sharing a storage share one page lease; pools
// disagreeing on stride cannot share a page lease and stay on the full mirror.
TEST(PoolStagingTest, SmallPoolIdentitySharedStorageAndStrideMismatch) {
  constexpr int64_t kStride = 64;
  StagingTestManager small(/*num_layers=*/1, /*num_shards=*/1,
                           /*slice_byte_size=*/kStride, /*host_blocks=*/1);
  small.SetDeviceBacked(0, kStride * 4);
  // 2 leases x 2 >= 4 blocks -> identity.
  absl::Status status =
      small.RegisterPools({DensePool("fa", 0, 0, kStride, /*num_blocks=*/4, 2)},
                          /*staging_leases=*/2);
  ASSERT_TRUE(status.ok()) << status.ToString();
  EXPECT_FALSE(small.PoolStorageStagingBounded(0));

  StagingTestManager shared(/*num_layers=*/1, /*num_shards=*/1,
                            /*slice_byte_size=*/kStride, /*host_blocks=*/1);
  shared.SetDeviceBacked(0, kStride * 32);
  status = shared.RegisterPools(
      {StridedPool("gdn.conv", 0, /*base_offset=*/0, kStride, 32, 1),
       StridedPool("gdn.ssm", 0, /*base_offset=*/16, kStride, 32, 1)},
      /*staging_leases=*/3);
  ASSERT_TRUE(status.ok()) << status.ToString();
  EXPECT_TRUE(shared.PoolStorageStagingBounded(0));
  EXPECT_EQ(shared.PoolStagingSummary()[0].num_slots, 3);
  ASSERT_TRUE(shared
                  .AcquirePoolStagingLease(/*uuid=*/5, 0,
                                           std::vector<int64_t>{20},
                                           absl::Milliseconds(10))
                  .ok());
  EXPECT_EQ(shared.PoolStagingSummary()[0].free_slots, 2);
  // Both pools of the storage address device page 20 through the same slot
  // (slot 0), each at its own base offset.
  tpu_sync::rpc::StartTransferRequest request;
  request.set_uuid(5);
  request.set_is_sender(false);
  for (int pool_idx = 0; pool_idx < 2; ++pool_idx) {
    auto* entry = (*request.mutable_shard_push_schedules())[0].add_entries();
    entry->set_dst_peer("127.0.0.1:1");
    entry->set_dst_shard_idx(0);
    entry->set_dst_block_id(20);
    entry->set_src_block_id(1);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(16);
    entry->set_layer_idx(pool_idx);
  }
  ASSERT_TRUE(shared.RegisterActivePlan(5, request, /*is_sender=*/false).ok());
  uint8_t* base = shared.GetHostPointer(0, 0);
  auto conv_chunks =
      shared.GetBlockChunks(/*layer_idx=*/0, 0, std::vector<int64_t>{20}, 16, 5,
                            /*sender_node_id=*/0);
  auto ssm_chunks =
      shared.GetBlockChunks(/*layer_idx=*/1, 0, std::vector<int64_t>{20}, 16, 5,
                            /*sender_node_id=*/0);
  ASSERT_FALSE(conv_chunks.empty());
  ASSERT_FALSE(ssm_chunks.empty());
  EXPECT_EQ(conv_chunks[0].ptr, base + 0 * kStride + 0);
  EXPECT_EQ(ssm_chunks[0].ptr, base + 0 * kStride + 16);

  StagingTestManager mixed(/*num_layers=*/1, /*num_shards=*/1,
                           /*slice_byte_size=*/kStride, /*host_blocks=*/1);
  mixed.SetDeviceBacked(0, kStride * 32);
  status = mixed.RegisterPools({DensePool("a", 0, 0, kStride, 32, 1),
                                DensePool("b", 0, 0, kStride / 2, 64, 1)},
                               /*staging_leases=*/3);
  ASSERT_TRUE(status.ok()) << status.ToString();
  EXPECT_FALSE(mixed.PoolStorageStagingBounded(0));
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
