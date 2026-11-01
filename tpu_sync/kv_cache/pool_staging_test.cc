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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/kv_cache/transfer_plan.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/transport/transfer_session.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::absl_testing::StatusIs;

class StagingTestManager : public KVCacheManagerBase {
 public:
  StagingTestManager(size_t num_layers, size_t num_shards,
                     size_t slice_byte_size, int host_blocks,
                     std::optional<int> num_staging_layers = std::nullopt)
      : KVCacheManagerBase(num_layers, num_shards, slice_byte_size,
                           /*local_port=*/std::nullopt,
                           std::make_optional(host_blocks), /*parallelism=*/1,
                           /*host_allocator=*/nullptr, /*bind_ip=*/std::nullopt,
                           num_staging_layers) {
    buffer_holds_.resize(num_layers);
    for (size_t l = 0; l < num_layers; ++l) {
      buffer_holds_[l].holds.resize(num_shards);
      if (l < layers_.size()) {
        for (size_t sh = 0; sh < num_shards; ++sh) {
          layers_[l].shards[sh].device_size = host_blocks * slice_byte_size;
        }
      }
    }
  }

  // Declares storage `layer_idx` device-backed with `physical_size` bytes.
  void SetDeviceBacked(size_t layer_idx, size_t physical_size) {
    buffer_holds_[layer_idx].physical_size = physical_size;
    major_dim_size_ = 1;
    if (layer_idx < layers_.size()) {
      for (auto& shard : layers_[layer_idx].shards) {
        shard.device_size = physical_size;
      }
    }
  }
};

// Hosts a |Base| test manager behind KVCacheManagerWithTransfer so that
// registered plans live in transfer sessions, the only place plans are
// stored. Base-level plan calls on manager() route through its hooks.
template <typename Base>
class WithPlanSessions : public KVCacheManagerWithTransfer {
 public:
  template <typename... Args>
  explicit WithPlanSessions(Args&&... args)
      : KVCacheManagerWithTransfer(
            std::make_unique<Base>(std::forward<Args>(args)...)) {}

  Base& manager() { return static_cast<Base&>(*base_); }

  // Stages plans in per-transfer host blocks, as
  // TPU_RAIDEN_DYNAMIC_HOST_STAGING=1 does.
  void EnableDemandStaging() {
    staging_allocator_ = CreateStagingBlockAllocator(
        base_.get(), staging_allocator_->num_slots(),
        staging_allocator_->max_blocks(), /*dynamic_host_staging=*/true);
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
  WithPlanSessions<StagingTestManager> manager_host(
      /*num_layers=*/1, /*num_shards=*/1,
      /*slice_byte_size=*/kStride, /*host_blocks=*/1);
  StagingTestManager& manager = manager_host.manager();
  manager.SetDeviceBacked(0, kStride * kNumBlocks);
  // 2 leases x 2 blocks per lease = 4 slots, well under the 16-block pool.
  absl::Status status =
      manager.RegisterPools({DensePool("fa", 0, 0, kStride, kNumBlocks,
                                       /*staging_blocks_per_request=*/2)},
                            /*staging_leases=*/2);
  ABSL_ASSERT_OK(status);
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
  EXPECT_THAT(manager.GetPoolBlockRef(0, 0, 5),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_EQ(manager.GetBlockHostPointer(/*layer_idx=*/0, 0, /*block_id=*/5),
            nullptr);

  // Lease blocks 5 and 9 for uuid 7: slot 0 then slot 1 (LIFO free list).
  status = manager.AcquirePoolStagingLease(/*uuid=*/7, /*storage_idx=*/0,
                                           std::vector<int64_t>{5, 9},
                                           absl::Milliseconds(50));
  ABSL_ASSERT_OK(status);
  // Re-acquiring the same ids is a no-op; adding one more takes slot 2.
  status = manager.AcquirePoolStagingLease(7, 0, std::vector<int64_t>{9, 11},
                                           absl::Milliseconds(50));
  ABSL_ASSERT_OK(status);
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
  ABSL_ASSERT_OK(status);
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
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kResourceExhausted));
  // More blocks than the whole arena is rejected outright.
  status = manager.AcquirePoolStagingLease(/*uuid=*/9, 0,
                                           std::vector<int64_t>{1, 2, 3, 4, 6},
                                           absl::Milliseconds(20));
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kResourceExhausted));

  // Releasing uuid 7 returns its three slots; uuid 8 now fits.
  ABSL_ASSERT_OK(manager.UnregisterActivePlan(7));
  manager.ReleasePoolStagingLeases(7);
  EXPECT_EQ(manager.PoolStagingSummary()[0].free_slots, 4);
  status = manager.AcquirePoolStagingLease(8, 0, std::vector<int64_t>{1, 2},
                                           absl::Milliseconds(20));
  ABSL_ASSERT_OK(status);
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
    ABSL_ASSERT_OK(status);
    EXPECT_FALSE(manager.PoolStorageStagingBounded(0));
    EXPECT_GE(manager.GetHostSize(0, 0),
              static_cast<size_t>(kStride * kNumBlocks));
    TF_ASSERT_OK_AND_ASSIGN(auto ref, manager.GetPoolBlockRef(0, 0, 9));
    EXPECT_EQ(ref.ptr, manager.GetHostPointer(0, 0) + 9 * kStride);
    // Leases are no-ops on unbounded storages.
    ABSL_EXPECT_OK(manager.AcquirePoolStagingLease(
        1, 0, std::vector<int64_t>{9}, absl::Milliseconds(1)));
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
  ABSL_ASSERT_OK(status);
  EXPECT_FALSE(small.PoolStorageStagingBounded(0));

  WithPlanSessions<StagingTestManager> shared_host(
      /*num_layers=*/1, /*num_shards=*/1,
      /*slice_byte_size=*/kStride, /*host_blocks=*/1);
  StagingTestManager& shared = shared_host.manager();
  shared.SetDeviceBacked(0, kStride * 32);
  status = shared.RegisterPools(
      {StridedPool("gdn.conv", 0, /*base_offset=*/0, kStride, 32, 1),
       StridedPool("gdn.ssm", 0, /*base_offset=*/16, kStride, 32, 1)},
      /*staging_leases=*/3);
  ABSL_ASSERT_OK(status);
  EXPECT_TRUE(shared.PoolStorageStagingBounded(0));
  EXPECT_EQ(shared.PoolStagingSummary()[0].num_slots, 3);
  ABSL_ASSERT_OK(shared.AcquirePoolStagingLease(
      /*uuid=*/5, 0, std::vector<int64_t>{20}, absl::Milliseconds(10)));
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
  ABSL_ASSERT_OK(shared.RegisterActivePlan(5, request, /*is_sender=*/false));
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
  ABSL_ASSERT_OK(status);
  EXPECT_FALSE(mixed.PoolStorageStagingBounded(0));
}

// A sender given the receiver's pool addresses (receiver_addrs) resolves each
// chunk's remote address (raddr) to exactly the pointer the receiver itself
// resolves, even though the two sides' pool geometries differ.
TEST(PoolStagingTest, SenderRaddrMatchesReceiverPointer) {
  constexpr int64_t kDstBase = 128;
  constexpr int64_t kDstStride = 64;
  constexpr int64_t kDstBlocks = 8;
  WithPlanSessions<StagingTestManager> receiver_host(
      /*num_layers=*/1, /*num_shards=*/1,
      /*slice_byte_size=*/kDstStride,
      /*host_blocks=*/1);
  StagingTestManager& receiver = receiver_host.manager();
  receiver.SetDeviceBacked(0, kDstBase + kDstStride * kDstBlocks);
  ABSL_ASSERT_OK(receiver.RegisterPools(
      {DensePool("fa", 0, kDstBase, kDstStride, kDstBlocks, 0)},
      /*staging_leases=*/0));
  ASSERT_FALSE(receiver.PoolStorageStagingBounded(0));

  constexpr int64_t kSrcStride = 32;
  constexpr int64_t kSrcBlocks = 16;
  WithPlanSessions<StagingTestManager> sender_host(
      /*num_layers=*/1, /*num_shards=*/1,
      /*slice_byte_size=*/kSrcStride, /*host_blocks=*/1);
  StagingTestManager& sender = sender_host.manager();
  sender.SetDeviceBacked(0, kSrcStride * kSrcBlocks);
  ABSL_ASSERT_OK(
      sender.RegisterPools({DensePool("fa", 0, 0, kSrcStride, kSrcBlocks, 0)},
                           /*staging_leases=*/0));

  constexpr char kPeer[] = "127.0.0.1:1";
  tpu_sync::rpc::StartTransferRequest request;
  request.set_uuid(11);
  request.set_req_id("req");
  request.add_transfer_pool_indices(0);
  auto* group = request.add_pool_groups();
  group->add_pool_indices(0);
  group->set_expected_pushes(1);
  auto* entry = (*request.mutable_shard_push_schedules())[0].add_entries();
  entry->set_dst_peer(kPeer);
  entry->set_dst_shard_idx(0);
  entry->set_src_block_id(3);
  entry->set_dst_block_id(5);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(8);
  entry->set_size_bytes(16);
  entry->set_count(2);
  entry->set_src_stride_bytes(16);
  entry->set_dst_stride_bytes(24);

  tpu_sync::rpc::StartTransferRequest recv_request = request;
  recv_request.set_is_sender(false);
  ABSL_ASSERT_OK(receiver.RegisterActivePlan(11, recv_request,
                                             /*is_sender=*/false));
  std::vector<transport::BlockChunk> recv_chunks = receiver.GetBlockChunks(
      /*layer_idx=*/0, /*shard_idx=*/0, std::vector<int64_t>{5},
      /*total_bytes=*/32, /*uuid=*/11, /*sender_node_id=*/0, /*peer=*/"",
      /*src_block_id=*/3);
  ASSERT_EQ(recv_chunks.size(), 2u);

  tpu_sync::rpc::StartTransferRequest send_request = request;
  send_request.set_is_sender(true);
  tpu_sync::rpc::PoolHostAddrsProto& dst_pool =
      (*(*send_request.mutable_receiver_addrs())[kPeer].mutable_pools())[0];
  dst_pool.set_block_stride_bytes(kDstStride);
  dst_pool.set_num_blocks(kDstBlocks);
  TF_ASSERT_OK_AND_ASSIGN(tpu_sync::rpc::PoolHostAddrsProto dst_bases,
                          receiver.PoolHostBaseAddrs(/*uuid=*/11, 0));
  for (uint64_t addr : dst_bases.host_base_addrs()) {
    dst_pool.add_host_base_addrs(addr);
  }
  ABSL_ASSERT_OK(sender.RegisterActivePlan(11, send_request,
                                           /*is_sender=*/true));
  std::vector<transport::BlockChunk> send_chunks = sender.GetBlockChunks(
      /*layer_idx=*/0, /*shard_idx=*/0, std::vector<int64_t>{3},
      /*total_bytes=*/32, /*uuid=*/11, /*sender_node_id=*/-1, kPeer,
      /*src_block_id=*/-1, /*dst_block_id=*/5);
  ASSERT_EQ(send_chunks.size(), recv_chunks.size());
  for (size_t i = 0; i < send_chunks.size(); ++i) {
    EXPECT_EQ(send_chunks[i].size, recv_chunks[i].size);
    EXPECT_EQ(send_chunks[i].raddr, recv_chunks[i].ptr);
  }

  // No peer: the sender cannot tell which receiver's addresses apply.
  send_chunks = sender.GetBlockChunks(0, 0, std::vector<int64_t>{3}, 32,
                                      /*uuid=*/11, -1, /*peer=*/"", -1, 5);
  ASSERT_EQ(send_chunks.size(), 2u);
  for (const auto& chunk : send_chunks) {
    EXPECT_EQ(chunk.raddr, nullptr);
  }

  // No addresses for this peer.
  tpu_sync::rpc::StartTransferRequest other_request = send_request;
  other_request.set_uuid(12);
  other_request.mutable_receiver_addrs()->clear();
  (*other_request.mutable_receiver_addrs())["127.0.0.1:2"] =
      send_request.receiver_addrs().at(kPeer);
  ABSL_ASSERT_OK(sender.RegisterActivePlan(12, other_request,
                                           /*is_sender=*/true));
  send_chunks = sender.GetBlockChunks(0, 0, std::vector<int64_t>{3}, 32,
                                      /*uuid=*/12, -1, kPeer, -1, 5);
  ASSERT_EQ(send_chunks.size(), 2u);
  for (const auto& chunk : send_chunks) {
    EXPECT_EQ(chunk.raddr, nullptr);
  }

  // Without host bases the sender cannot address the receiver.
  send_request.set_uuid(13);
  (*(*send_request.mutable_receiver_addrs())[kPeer].mutable_pools())[0]
      .clear_host_base_addrs();
  ABSL_ASSERT_OK(sender.RegisterActivePlan(13, send_request,
                                           /*is_sender=*/true));
  send_chunks = sender.GetBlockChunks(0, 0, std::vector<int64_t>{3}, 32,
                                      /*uuid=*/13, -1, kPeer, -1, 5);
  ASSERT_EQ(send_chunks.size(), 2u);
  for (const auto& chunk : send_chunks) {
    EXPECT_EQ(chunk.raddr, nullptr);
  }
}

// On a bounded receiver the sender addresses the block's leased slot.
TEST(PoolStagingTest, SenderRaddrUsesReceiverLeaseSlot) {
  constexpr int64_t kStride = 64;
  constexpr int64_t kBlocks = 16;
  WithPlanSessions<StagingTestManager> receiver_host(
      /*num_layers=*/1, /*num_shards=*/1,
      /*slice_byte_size=*/kStride, /*host_blocks=*/1);
  StagingTestManager& receiver = receiver_host.manager();
  receiver.SetDeviceBacked(0, kStride * kBlocks);
  ABSL_ASSERT_OK(
      receiver.RegisterPools({DensePool("fa", 0, 0, kStride, kBlocks,
                                        /*staging_blocks_per_request=*/2)},
                             /*staging_leases=*/2));
  ASSERT_TRUE(receiver.PoolStorageStagingBounded(0));
  ABSL_ASSERT_OK(receiver.AcquirePoolStagingLease(
      /*uuid=*/11, /*storage_idx=*/0, std::vector<int64_t>{5, 9},
      absl::Milliseconds(50)));

  WithPlanSessions<StagingTestManager> sender_host(
      /*num_layers=*/1, /*num_shards=*/1,
      /*slice_byte_size=*/kStride, /*host_blocks=*/1);
  StagingTestManager& sender = sender_host.manager();
  sender.SetDeviceBacked(0, kStride * kBlocks);
  ABSL_ASSERT_OK(sender.RegisterPools(
      {DensePool("fa", 0, 0, kStride, kBlocks, 0)}, /*staging_leases=*/0));

  constexpr char kPeer[] = "127.0.0.1:1";
  tpu_sync::rpc::StartTransferRequest request;
  request.set_uuid(11);
  request.set_req_id("req");
  request.add_transfer_pool_indices(0);
  auto* group = request.add_pool_groups();
  group->add_pool_indices(0);
  group->set_expected_pushes(1);
  auto* entry = (*request.mutable_shard_push_schedules())[0].add_entries();
  entry->set_dst_peer(kPeer);
  entry->set_dst_shard_idx(0);
  entry->set_src_block_id(3);
  entry->set_dst_block_id(9);
  entry->set_dst_offset_bytes(8);
  entry->set_size_bytes(16);

  tpu_sync::rpc::StartTransferRequest recv_request = request;
  recv_request.set_is_sender(false);
  ABSL_ASSERT_OK(receiver.RegisterActivePlan(11, recv_request,
                                             /*is_sender=*/false));
  std::vector<transport::BlockChunk> recv_chunks = receiver.GetBlockChunks(
      /*layer_idx=*/0, /*shard_idx=*/0, std::vector<int64_t>{9},
      /*total_bytes=*/16, /*uuid=*/11, /*sender_node_id=*/0, /*peer=*/"",
      /*src_block_id=*/3);
  ASSERT_EQ(recv_chunks.size(), 1u);

  tpu_sync::rpc::StartTransferRequest send_request = request;
  send_request.set_is_sender(true);
  TF_ASSERT_OK_AND_ASSIGN(
      (*(*send_request.mutable_receiver_addrs())[kPeer].mutable_pools())[0],
      receiver.PoolHostBaseAddrs(/*uuid=*/11, /*pool_idx=*/0));
  ABSL_ASSERT_OK(sender.RegisterActivePlan(11, send_request,
                                           /*is_sender=*/true));
  std::vector<transport::BlockChunk> send_chunks = sender.GetBlockChunks(
      /*layer_idx=*/0, /*shard_idx=*/0, std::vector<int64_t>{3},
      /*total_bytes=*/16, /*uuid=*/11, /*sender_node_id=*/-1, kPeer,
      /*src_block_id=*/-1, /*dst_block_id=*/9);
  ASSERT_EQ(send_chunks.size(), 1u);
  EXPECT_EQ(send_chunks[0].raddr, recv_chunks[0].ptr);

  // A transfer without a lease has no host addresses.
  TF_ASSERT_OK_AND_ASSIGN(tpu_sync::rpc::PoolHostAddrsProto unleased,
                          receiver.PoolHostBaseAddrs(12, 0));
  EXPECT_TRUE(unleased.host_base_addrs().empty());
}

// PoolHostBaseAddrs reports, per local shard, each pool's base address
// (storage host pointer + base offset), is empty for bounded staging, and
// rejects unknown pools.
TEST(PoolStagingTest, PoolHostBaseAddrs) {
  constexpr int64_t kStride = 64;
  StagingTestManager full(/*num_layers=*/1, /*num_shards=*/2,
                          /*slice_byte_size=*/kStride, /*host_blocks=*/1);
  full.SetDeviceBacked(0, 32 + kStride * 8);
  ABSL_ASSERT_OK(full.RegisterPools(
      {DensePool("a", 0, /*base_offset=*/0, kStride, 8, 0),
       StridedPool("b", 0, /*base_offset=*/32, kStride, 8, 0)},
      /*staging_leases=*/0));
  for (size_t pool_idx = 0; pool_idx < 2; ++pool_idx) {
    TF_ASSERT_OK_AND_ASSIGN(tpu_sync::rpc::PoolHostAddrsProto addrs,
                            full.PoolHostBaseAddrs(/*uuid=*/1, pool_idx));
    ASSERT_EQ(addrs.host_base_addrs_size(), 2);
    for (size_t sh = 0; sh < 2; ++sh) {
      EXPECT_EQ(addrs.host_base_addrs(sh),
                reinterpret_cast<uint64_t>(full.GetHostPointer(0, sh)) +
                    full.pool(pool_idx)->base_offset_bytes);
    }
  }
  EXPECT_THAT(full.PoolHostBaseAddrs(/*uuid=*/1, 2),
              StatusIs(absl::StatusCode::kOutOfRange));

  StagingTestManager bounded(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/kStride, /*host_blocks=*/1);
  bounded.SetDeviceBacked(0, kStride * 32);
  ABSL_ASSERT_OK(bounded.RegisterPools(
      {DensePool("fa", 0, 0, kStride, 32, /*staging_blocks_per_request=*/2)},
      /*staging_leases=*/2));
  ASSERT_TRUE(bounded.PoolStorageStagingBounded(0));
  TF_ASSERT_OK_AND_ASSIGN(tpu_sync::rpc::PoolHostAddrsProto addrs,
                          bounded.PoolHostBaseAddrs(/*uuid=*/1, 0));
  EXPECT_TRUE(addrs.host_base_addrs().empty());
}

TEST(PoolStagingTest, BoundedHostLayersPoolStaging) {
  constexpr int64_t kStride = 64;
  constexpr int64_t kNumBlocks = 16;
  constexpr size_t kNumLayers = 4;
  StagingTestManager manager(/*num_layers=*/kNumLayers, /*num_shards=*/1,
                             /*slice_byte_size=*/kStride, /*host_blocks=*/1,
                             /*num_staging_layers=*/2);

  EXPECT_EQ(manager.num_layers(), kNumLayers);
  EXPECT_EQ(manager.num_host_layers(), 2u);

  for (size_t i = 0; i < kNumLayers; ++i) {
    manager.SetDeviceBacked(i, kStride * kNumBlocks);
  }

  // Register 4 pools, one per storage, with bounded staging (2 leases x 2
  // blocks = 4 slots).
  std::vector<PoolSpec> pools;
  pools.reserve(kNumLayers);
  for (size_t i = 0; i < kNumLayers; ++i) {
    pools.push_back(DensePool("pool_" + std::to_string(i), /*storage_index=*/i,
                              /*base_offset=*/0, kStride, kNumBlocks,
                              /*staging_blocks_per_request=*/2));
  }
  ABSL_ASSERT_OK(manager.RegisterPools(pools, /*staging_leases=*/2));

  for (size_t i = 0; i < kNumLayers; ++i) {
    EXPECT_EQ(manager.GetBlockArrayHostSize(i, 0),
              static_cast<size_t>(4 * kStride));
  }

  // Receiver plan for uuid 42.
  tpu_sync::rpc::StartTransferRequest request;
  request.set_uuid(42);
  request.set_is_sender(false);
  auto* entry = (*request.mutable_shard_push_schedules())[0].add_entries();
  entry->set_dst_peer("127.0.0.1:1");
  entry->set_dst_shard_idx(0);
  entry->set_dst_block_id(4);
  entry->set_src_block_id(1);
  entry->set_dst_offset_bytes(0);
  entry->set_src_offset_bytes(0);
  entry->set_size_bytes(16);
  absl::StatusOr<TransferPlan> plan =
      TransferPlan::Create(request, /*is_sender=*/false, manager.num_pools(),
                           manager.has_explicit_pools(), manager.pools());
  ABSL_ASSERT_OK(plan);

  // A session that carries the plan and leases host layer 1 for pool 3.
  class StubPoolTransferSession : public TransferSession {
   public:
    explicit StubPoolTransferSession(std::shared_ptr<const TransferPlan> plan)
        : plan_(std::move(plan)) {}
    bool Done() const override { return false; }
    void Finish(const absl::Status& status) override {}
    bool TryBeginOp() override { return true; }
    void EndOp() override {}
    absl::Status GetStatus() const override { return absl::OkStatus(); }
    absl::Status AwaitForDone() override { return absl::OkStatus(); }
    bool IsDraining() const override { return false; }
    absl::StatusOr<size_t> ResolveStagingSlot(size_t unit) override {
      return unit % 2;
    }
    absl::Status OnDataReceived(size_t idx) override {
      return absl::OkStatus();
    }
    absl::optional<std::shared_ptr<const TransferPlan>> plan() const override {
      return plan_;
    }

   private:
    std::shared_ptr<const TransferPlan> plan_;
  };
  auto session = std::make_shared<StubPoolTransferSession>(
      std::make_shared<const TransferPlan>(*std::move(plan)));
  KVCacheManagerBase::TransferEventHooks hooks;
  hooks.get_session = [session](uint64_t uuid)
      -> absl::StatusOr<std::shared_ptr<TransferSession>> { return session; };
  manager.SetTransferEventHooks(std::move(hooks));

  // Lease blocks 2 and 4 on storage 3 under uuid 42.
  ABSL_ASSERT_OK(
      manager.AcquirePoolStagingLease(42, 3, {2, 4}, absl::Milliseconds(50)));

  // For pool 3, the hook returns host slot 1 (pool 3 % 2 = 1).
  // Destination block 4 was leased as the 2nd block (slot 1 within the lease
  // arena).
  std::vector<transport::BlockChunk> chunks = manager.GetBlockChunks(
      /*layer_idx=*/3, /*shard_idx=*/0, {4}, /*total_bytes=*/16, /*uuid=*/42,
      /*sender_node_id=*/0);
  ASSERT_EQ(chunks.size(), 1u);
  EXPECT_EQ(chunks[0].size, 16u);
  uint8_t* expected_host_base = manager.GetHostPointer(/*layer_idx=*/1, 0);
  EXPECT_EQ(chunks[0].ptr, expected_host_base + 1 * kStride);

  manager.ReleasePoolStagingLeases(42);
}

TEST(PoolStagingTest, GetBlockChunksLargePlanLinearScaling) {
  constexpr size_t kNumShards = 8;
  constexpr size_t kNumFaPools = 6;
  constexpr size_t kTotalPools = 93;
  constexpr int64_t kStride = 64;
  constexpr int64_t kMaxBlocks = 4096;
  constexpr int64_t kNumBlocks = 3072;
  constexpr uint64_t kUuid = 101;
  constexpr char kPeer[] = "127.0.0.1:1";

  WithPlanSessions<StagingTestManager> receiver_host(1, kNumShards, kStride,
                                                     /*host_blocks=*/1);
  StagingTestManager& receiver = receiver_host.manager();
  WithPlanSessions<StagingTestManager> sender_host(1, kNumShards, kStride,
                                                   /*host_blocks=*/1);
  StagingTestManager& sender = sender_host.manager();
  const size_t storage_bytes = kTotalPools * kStride * kMaxBlocks;
  receiver.SetDeviceBacked(0, storage_bytes);
  sender.SetDeviceBacked(0, storage_bytes);

  std::vector<PoolSpec> specs;
  specs.reserve(kTotalPools);
  for (size_t pool_idx = 0; pool_idx < kTotalPools; ++pool_idx) {
    specs.push_back(DensePool("pool_" + std::to_string(pool_idx), 0,
                              pool_idx * kStride * kMaxBlocks, kStride,
                              kMaxBlocks, /*staging_blocks_per_request=*/0));
  }
  ABSL_ASSERT_OK(receiver.RegisterPools(specs, /*staging_leases=*/0));
  ABSL_ASSERT_OK(sender.RegisterPools(specs, /*staging_leases=*/0));

  tpu_sync::rpc::StartTransferRequest request;
  request.set_uuid(kUuid);
  request.set_req_id("req");
  tpu_sync::rpc::PoolGroupProto* fa_group = request.add_pool_groups();
  tpu_sync::rpc::PoolGroupProto* swa_group = request.add_pool_groups();
  fa_group->set_expected_pushes(1);
  swa_group->set_expected_pushes(1);
  for (size_t pool_idx = 0; pool_idx < kTotalPools; ++pool_idx) {
    request.add_transfer_pool_indices(pool_idx);
    (pool_idx < kNumFaPools ? fa_group : swa_group)->add_pool_indices(pool_idx);
  }

  std::vector<int64_t> dst_blocks(kNumBlocks);
  for (size_t shard_idx = 0; shard_idx < kNumShards; ++shard_idx) {
    tpu_sync::rpc::ShardPushScheduleProto& schedule =
        (*request.mutable_shard_push_schedules())[shard_idx];
    for (int64_t block_id = 0; block_id < kNumBlocks; ++block_id) {
      dst_blocks[block_id] = block_id + 100;
      tpu_sync::rpc::ShardPushEntryProto* entry = schedule.add_entries();
      entry->set_dst_peer(kPeer);
      entry->set_dst_shard_idx(shard_idx);
      entry->set_src_block_id(block_id);
      entry->set_dst_block_id(dst_blocks[block_id]);
      entry->set_size_bytes(kStride);
      entry->set_pool_group(0);
    }
  }
  ABSL_ASSERT_OK(
      receiver.RegisterActivePlan(kUuid, request, /*is_sender=*/false));
  for (size_t pool_idx = 0; pool_idx < kNumFaPools; ++pool_idx) {
    TF_ASSERT_OK_AND_ASSIGN(tpu_sync::rpc::PoolHostAddrsProto pool_addrs,
                            receiver.PoolHostBaseAddrs(kUuid, pool_idx));
    (*(*request.mutable_receiver_addrs())[kPeer].mutable_pools())[pool_idx] =
        pool_addrs;
  }
  ABSL_ASSERT_OK(sender.RegisterActivePlan(kUuid, request, /*is_sender=*/true));

  for (size_t shard_idx = 0; shard_idx < kNumShards; ++shard_idx) {
    for (size_t pool_idx = 0; pool_idx < kNumFaPools; ++pool_idx) {
      std::vector<transport::BlockChunk> recv_chunks = receiver.GetBlockChunks(
          pool_idx, shard_idx, dst_blocks, kNumBlocks * kStride, kUuid,
          /*sender_node_id=*/shard_idx);
      ASSERT_EQ(recv_chunks.size(), static_cast<size_t>(kNumBlocks));
      for (int64_t block_id = 0; block_id < kNumBlocks; ++block_id) {
        std::vector<transport::BlockChunk> send_chunks = sender.GetBlockChunks(
            pool_idx, shard_idx, {block_id}, kStride, kUuid,
            /*sender_node_id=*/-1, kPeer, block_id, dst_blocks[block_id]);
        ASSERT_EQ(send_chunks.size(), 1u);
        EXPECT_EQ(send_chunks[0].raddr, recv_chunks[block_id].ptr);
      }
    }
  }
  EXPECT_TRUE(receiver
                  .GetBlockChunks(kNumFaPools, /*shard_idx=*/0, dst_blocks,
                                  kNumBlocks * kStride, kUuid,
                                  /*sender_node_id=*/0)
                  .empty());
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
