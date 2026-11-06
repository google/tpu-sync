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

#include "tpu_sync/kv_cache/reshard_receive_session.h"

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {

namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::tpu_sync::rpc::MEMORY_TYPE_HBM;
using ::tpu_sync::rpc::ShardPushEntryProto;
using ::tpu_sync::rpc::StartTransferRequest;

kv_cache::KVCacheManagerBase MakeTestBase() {
  return kv_cache::KVCacheManagerBase(
      /*num_layers=*/1, /*num_shards=*/1,
      /*slice_byte_size=*/128,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/std::make_optional(4));
}

kv_cache::PoolSpec DensePool(std::string tag, int64_t block_stride = 128,
                             std::string dtype_tag = "bf16") {
  return kv_cache::PoolSpec{
      .tag = std::move(tag),
      .storage_index = 0,
      .base_offset_bytes = 0,
      .block_stride_bytes = block_stride,
      .num_blocks = 4,
      .regions = {kv_cache::RegionSpec{
          .name = "block",
          .offset_bytes = 0,
          .stride_bytes = block_stride,
          .unit_bytes = block_stride,
          .num_units = 1,
          .units_per_stride = 1,
      }},
      .dtype_tag = std::move(dtype_tag),
  };
}

StartTransferRequest ValidPlan(
    int64_t uuid, const std::vector<std::string>& dtype_tags = {"bf16"},
    const std::vector<int32_t>& transferred_pools = {0}) {
  StartTransferRequest plan;
  plan.set_uuid(uuid);
  plan.set_req_id("pool_reshard_req_" + std::to_string(uuid));
  plan.set_dst_mem_type(MEMORY_TYPE_HBM);
  plan.set_use_block_chunks(true);
  plan.set_parallelism(1);
  for (int32_t pool_idx : transferred_pools) {
    plan.add_transfer_pool_indices(pool_idx);
  }
  for (const std::string& tag : dtype_tags) {
    plan.add_pool_dtype_tags(tag);
  }
  auto* group = plan.add_pool_groups();
  for (int32_t pool_idx : transferred_pools) {
    group->add_pool_indices(pool_idx);
  }
  group->add_dst_device_block_ids(0);
  group->set_expected_pushes(1);
  group->add_dst_expected_extent_bytes(16);
  auto* entry = (*plan.mutable_shard_push_schedules())[0].add_entries();
  entry->set_dst_peer("127.0.0.1:1");
  entry->set_dst_shard_idx(0);
  entry->set_src_block_id(0);
  entry->set_dst_block_id(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(16);
  entry->set_src_stride_bytes(0);
  entry->set_dst_stride_bytes(0);
  entry->set_count(1);
  return plan;
}

ShardPushEntryProto* AddEntry(StartTransferRequest& plan, int32_t schedule_key,
                              int64_t dst_block_id, int64_t dst_offset,
                              int64_t size, int32_t group_idx = 0) {
  auto* entry =
      (*plan.mutable_shard_push_schedules())[schedule_key].add_entries();
  entry->set_dst_peer("127.0.0.1:1");
  entry->set_dst_shard_idx(0);
  entry->set_src_block_id(0);
  entry->set_dst_block_id(dst_block_id);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(dst_offset);
  entry->set_size_bytes(size);
  entry->set_src_stride_bytes(0);
  entry->set_dst_stride_bytes(0);
  entry->set_count(1);
  entry->set_pool_group(group_idx);
  return entry;
}

void ExpectInvalid(const absl::Status& status, const std::string& fragment) {
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument)
      << status.ToString();
  EXPECT_NE(std::string(status.message()).find(fragment), std::string::npos)
      << status.ToString();
}

// Returns the status of creating a receive session for `plan` on `base`.
// Create validates the plan before anything else that can fail here; the
// placeholder device hold passes its device-attachment check for plans that
// validate.
absl::Status CreateStatus(kv_cache::KVCacheManagerBase& base,
                          const StartTransferRequest& plan,
                          absl::Span<const int64_t> chip_blocks) {
  base.AttachPlaceholderDeviceHoldForTest();
  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(&base, /*num_slots=*/0, /*max_blocks=*/0);
  return CreateReshardReceiveSession(
             &base, allocator.get(), plan, chip_blocks,
             std::chrono::steady_clock::now() + std::chrono::seconds(10))
      .status();
}

// A CPU-only receiver base whose pool host-to-device uploads need no device:
// every upload completes OK as soon as it is issued.
class FakeH2dBase : public kv_cache::KVCacheManagerBase {
 public:
  using kv_cache::KVCacheManagerBase::KVCacheManagerBase;

  absl::StatusOr<raiden::PjRtCopyFuture> H2dPoolBlocks(
      size_t pool_idx, absl::Span<const int64_t> block_ids,
      std::optional<size_t> shard_idx, std::optional<uint64_t> uuid,
      std::optional<size_t> host_layer_idx) override {
    if (host_layer_idx.has_value()) {
      uploaded_host_slot_by_pool_.emplace_back(pool_idx, *host_layer_idx);
    }
    auto [promise, future] = xla::MakePromise();
    promise.Set();
    return raiden::PjRtCopyFuture(std::move(future), raiden::BufferHolders{});
  }

  const std::vector<std::pair<size_t, size_t>>& uploaded_host_slot_by_pool()
      const {
    return uploaded_host_slot_by_pool_;
  }

 private:
  std::vector<std::pair<size_t, size_t>> uploaded_host_slot_by_pool_;
};

TEST(ReshardReceiveSessionTest, AcceptsCanonicalPlanOnExplicitPools) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1001);

  EXPECT_TRUE(CreateStatus(base, plan, std::vector<int64_t>{0}).ok());
}

TEST(ReshardReceiveSessionTest, AcceptsImplicitPools) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  StartTransferRequest plan = ValidPlan(/*uuid=*/1002, /*dtype_tags=*/{""});

  EXPECT_TRUE(CreateStatus(base, plan, std::vector<int64_t>{0}).ok());
}

TEST(ReshardReceiveSessionTest, HasNoTagPolicy) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("gdn.conv")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1003);

  EXPECT_TRUE(CreateStatus(base, plan, std::vector<int64_t>{0}).ok());
}

TEST(ReshardReceiveSessionTest, RejectsMissingIdentityAndPoolFields) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());

  StartTransferRequest plan = ValidPlan(/*uuid=*/1005);
  plan.clear_req_id();
  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}), "req_id");

  plan = ValidPlan(/*uuid=*/0);
  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
                "uuid must be positive");

  plan = ValidPlan(/*uuid=*/1006);
  plan.clear_transfer_pool_indices();
  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
                "transfer_pool_indices");

  plan = ValidPlan(/*uuid=*/1007);
  plan.clear_pool_groups();
  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
                "must declare pool_groups");

  plan = ValidPlan(/*uuid=*/1008);
  plan.set_use_block_chunks(false);
  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
                "use_block_chunks");
}

TEST(ReshardReceiveSessionTest, RejectsOutOfRangeDuplicateAndDtypeMismatch) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());

  StartTransferRequest plan = ValidPlan(/*uuid=*/1009, /*dtype_tags=*/{"bf16"},
                                        /*transferred_pools=*/{1});
  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
                "out of range");

  plan = ValidPlan(/*uuid=*/1010, /*dtype_tags=*/{"bf16"},
                   /*transferred_pools=*/{0, 0});
  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
                "duplicate transfer pool index");

  plan = ValidPlan(/*uuid=*/1011, /*dtype_tags=*/{"fp8"});
  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
                "dtype tag mismatch");

  plan = ValidPlan(/*uuid=*/1012, /*dtype_tags=*/{"bf16", "bf16"});
  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
                "one dtype tag per pool");
}

TEST(ReshardReceiveSessionTest, ChecksEveryPoolSpanAndDestinationZeroCover) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(
      base.RegisterPools({DensePool("fa", 128), DensePool("fa", 64)}).ok());
  StartTransferRequest plan =
      ValidPlan(/*uuid=*/1013, /*dtype_tags=*/{"bf16", "bf16"},
                /*transferred_pools=*/{0, 1});
  auto* entry = plan.mutable_shard_push_schedules()->at(0).mutable_entries(0);
  entry->set_dst_offset_bytes(48);
  entry->set_size_bytes(32);
  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
                "destination span exceeds declared pool 1");

  plan = ValidPlan(/*uuid=*/1014, /*dtype_tags=*/{"bf16", "bf16"},
                   /*transferred_pools=*/{0, 1});
  entry = plan.mutable_shard_push_schedules()->at(0).mutable_entries(0);
  entry->set_dst_offset_bytes(16);
  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
                "no transfer entry starting at offset 0");

  plan = ValidPlan(/*uuid=*/1015, /*dtype_tags=*/{"bf16", "bf16"},
                   /*transferred_pools=*/{0, 1});
  plan.mutable_shard_push_schedules()->at(0).clear_entries();
  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
                "no entries");
}

TEST(ReshardReceiveSessionTest, RejectsBlockIdsOutsideDeclaredPool) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1017);
  plan.mutable_pool_groups(0)->set_dst_device_block_ids(0, 9);
  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{9}),
                "out of range for pool");

  plan = ValidPlan(/*uuid=*/1018);
  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{8}),
                "must concatenate to the plan's local block ids");
  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0, 0}),
                "must cover the plan's local block ids");
  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{}),
                "must not be empty");
}

TEST(ReshardReceiveSessionTest, RejectsCoverageGap) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1101);
  plan.mutable_pool_groups(0)->clear_dst_expected_extent_bytes();
  plan.mutable_pool_groups(0)->add_dst_expected_extent_bytes(48);
  AddEntry(plan, 0, /*dst_block_id=*/0, /*dst_offset=*/32, /*size=*/16);

  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
                "coverage gap");
}

TEST(ReshardReceiveSessionTest, RejectsCoverageOverlap) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1102);
  plan.mutable_pool_groups(0)->clear_dst_expected_extent_bytes();
  plan.mutable_pool_groups(0)->add_dst_expected_extent_bytes(24);
  AddEntry(plan, 0, /*dst_block_id=*/0, /*dst_offset=*/8, /*size=*/16);

  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
                "coverage overlap");
}

TEST(ReshardReceiveSessionTest, RejectsShortCoverage) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1103);
  plan.mutable_pool_groups(0)->clear_dst_expected_extent_bytes();
  plan.mutable_pool_groups(0)->add_dst_expected_extent_bytes(24);

  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
                "does not cover the exact live bytes");
}

TEST(ReshardReceiveSessionTest, RejectsNonPrefixExtents) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1104);
  auto* group = plan.mutable_pool_groups(0);
  group->add_dst_device_block_ids(1);
  group->clear_dst_expected_extent_bytes();
  group->add_dst_expected_extent_bytes(16);
  group->add_dst_expected_extent_bytes(16);
  AddEntry(plan, 0, /*dst_block_id=*/1, /*dst_offset=*/0, /*size=*/16);

  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0, 1}),
                "except the final one");
}

TEST(ReshardReceiveSessionTest, RejectsMissingExtents) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1105);
  plan.mutable_pool_groups(0)->clear_dst_expected_extent_bytes();

  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
                "require dst_expected_extent_bytes");
}

TEST(ReshardReceiveSessionTest, RejectsMissingParallelism) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1106);
  plan.set_parallelism(0);

  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
                "positive parallelism");
}

TEST(ReshardReceiveSessionTest, RejectsExpectedPushMismatch) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1107);
  plan.mutable_pool_groups(0)->set_expected_pushes(2);

  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
                "do not match the received schedules");
}

TEST(ReshardReceiveSessionTest, RejectsWriteBeyondDeclaredLiveTail) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  kv_cache::PoolSpec pool = DensePool("fa");
  pool.regions = {
      kv_cache::RegionSpec{
          .name = "run_0",
          .offset_bytes = 0,
          .stride_bytes = 32,
          .unit_bytes = 32,
          .num_units = 1,
          .units_per_stride = 1,
      },
      kv_cache::RegionSpec{
          .name = "run_1",
          .offset_bytes = 64,
          .stride_bytes = 32,
          .unit_bytes = 32,
          .num_units = 1,
          .units_per_stride = 1,
      },
  };
  ASSERT_TRUE(base.RegisterPools({pool}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1108);
  plan.mutable_pool_groups(0)->clear_dst_expected_extent_bytes();
  plan.mutable_pool_groups(0)->add_dst_expected_extent_bytes(40);
  plan.mutable_shard_push_schedules()->at(0).clear_entries();
  AddEntry(plan, 0, /*dst_block_id=*/0, /*dst_offset=*/0, /*size=*/32);
  AddEntry(plan, 0, /*dst_block_id=*/0, /*dst_offset=*/64, /*size=*/16);

  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
                "declared live tail");

  plan.mutable_shard_push_schedules()->at(0).mutable_entries(1)->set_size_bytes(
      8);
  EXPECT_TRUE(CreateStatus(base, plan, std::vector<int64_t>{0}).ok());
}

TEST(ReshardReceiveSessionTest, GroupScopesDestinationBlocks) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(
      base.RegisterPools({DensePool("fa", 128), DensePool("gdn", 64)}).ok());
  StartTransferRequest plan = ValidPlan(
      /*uuid=*/1109, /*dtype_tags=*/{"bf16", "bf16"},
      /*transferred_pools=*/{0, 1});
  plan.clear_pool_groups();
  plan.mutable_shard_push_schedules()->at(0).clear_entries();
  AddEntry(plan, 0, /*dst_block_id=*/2, /*dst_offset=*/0, /*size=*/16,
           /*group_idx=*/0);
  AddEntry(plan, 0, /*dst_block_id=*/3, /*dst_offset=*/0, /*size=*/8,
           /*group_idx=*/1);
  auto* fa_group = plan.add_pool_groups();
  fa_group->add_pool_indices(0);
  fa_group->add_dst_device_block_ids(2);
  fa_group->set_expected_pushes(1);
  fa_group->add_dst_expected_extent_bytes(16);
  auto* state_group = plan.add_pool_groups();
  state_group->add_pool_indices(1);
  state_group->add_dst_device_block_ids(3);
  state_group->set_expected_pushes(1);
  state_group->add_dst_expected_extent_bytes(8);
  state_group->set_order_rank(1);

  EXPECT_TRUE(CreateStatus(base, plan, std::vector<int64_t>{2, 3}).ok());

  AddEntry(plan, 0, /*dst_block_id=*/2, /*dst_offset=*/0, /*size=*/8,
           /*group_idx=*/1);
  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{2, 3}),
                "outside its group");
}

TEST(ReshardReceiveSessionTest, BoundedStagingSlotAcquisitionAndRelease) {
  FakeH2dBase base(
      /*num_layers=*/2, /*num_shards=*/1,
      /*slice_byte_size=*/128,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/std::make_optional(4),
      /*parallelism=*/1, /*host_allocator=*/nullptr,
      /*bind_ip=*/std::nullopt, /*num_staging_layers=*/1);
  base.AttachPlaceholderDeviceHoldForTest();
  ASSERT_EQ(base.num_host_layers(), 1u);
  ASSERT_EQ(base.num_layers(), 2u);

  std::vector<kv_cache::PoolSpec> pools = {
      DensePool("fa", 128),
      DensePool("state", 128),
  };
  pools[1].storage_index = 1;
  ABSL_ASSERT_OK(base.RegisterPools(pools));

  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(&base, /*num_slots=*/0, /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(
      /*uuid=*/1301, /*dtype_tags=*/{"bf16", "bf16"},
      /*transferred_pools=*/{0, 1});
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<TransferSession> session,
      CreateReshardReceiveSession(
          &base, allocator.get(), plan, std::vector<int64_t>{0},
          std::chrono::steady_clock::now() + std::chrono::seconds(2)));

  // Pool 0 leases slot 0 (the only allocated host layer) and keeps it.
  EXPECT_THAT(session->ResolveStagingSlot(0), IsOkAndHolds(0u));
  EXPECT_THAT(session->ResolveStagingSlot(0), IsOkAndHolds(0u));

  // Pool 1 blocks waiting for slot 0 until pool 0's upload releases it.
  absl::StatusOr<size_t> slot1;
  std::thread release_waiter([&]() { slot1 = session->ResolveStagingSlot(1); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  ABSL_EXPECT_OK(session->OnDataReceived(/*major_dimension_idx=*/0,
                                         /*minor_dimension_indices=*/{},
                                         /*shard_ids=*/{}));
  release_waiter.join();
  EXPECT_THAT(slot1, IsOkAndHolds(0u));
  EXPECT_THAT(session->ResolveStagingSlot(1), IsOkAndHolds(0u));

  // Cancelling the session while pool 0 is blocked waiting for a slot wakes
  // the waiter with CancelledError and releases all held slots once settled.
  absl::StatusOr<size_t> slot0_cancelled;
  std::thread cancel_waiter(
      [&]() { slot0_cancelled = session->ResolveStagingSlot(0); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  session->Finish(absl::CancelledError("cancelled"));
  cancel_waiter.join();
  EXPECT_THAT(slot0_cancelled, StatusIs(absl::StatusCode::kCancelled));
  EXPECT_TRUE(session->Done());
  EXPECT_THAT(session->ResolveStagingSlot(1),
              StatusIs(absl::StatusCode::kCancelled));

  // Pool 1 cannot lease while slot 0 is held by pool 0 and deadline expires;
  // timeout terminally drains the session and releases all held slots.
  StartTransferRequest timeout_plan = ValidPlan(
      /*uuid=*/1302, /*dtype_tags=*/{"bf16", "bf16"},
      /*transferred_pools=*/{0, 1});
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<TransferSession> timeout_session,
      CreateReshardReceiveSession(
          &base, allocator.get(), timeout_plan, std::vector<int64_t>{0},
          std::chrono::steady_clock::now() + std::chrono::milliseconds(30)));
  EXPECT_THAT(timeout_session->ResolveStagingSlot(0), IsOkAndHolds(0u));
  EXPECT_THAT(timeout_session->ResolveStagingSlot(1),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_TRUE(timeout_session->Done());
  EXPECT_THAT(timeout_session->GetStatus(),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_THAT(timeout_session->ResolveStagingSlot(0),
              StatusIs(absl::StatusCode::kCancelled));
}

TEST(ReshardReceiveSessionTest,
     OutOfOrderPoolArrivalResolvesLeasedHostBufferForValidation) {
  constexpr uint64_t kUuid = 1303;
  FakeH2dBase base(
      /*num_layers=*/2, /*num_shards=*/1,
      /*slice_byte_size=*/128,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/std::make_optional(4));
  base.AttachPlaceholderDeviceHoldForTest();
  ASSERT_EQ(base.num_host_layers(), 2u);
  ASSERT_EQ(base.num_layers(), 2u);

  std::vector<kv_cache::PoolSpec> pools = {
      DensePool("fa", 128),
      DensePool("state", 128),
  };
  pools[0].storage_index = 0;
  pools[0].base_offset_bytes = 16;
  pools[0].num_blocks = 3;
  pools[1].storage_index = 1;
  pools[1].base_offset_bytes = 32;
  pools[1].num_blocks = 3;
  ABSL_ASSERT_OK(base.RegisterPools(pools));

  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(&base, /*num_slots=*/0, /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(
      /*uuid=*/kUuid, /*dtype_tags=*/{"bf16", "bf16"},
      /*transferred_pools=*/{0, 1});
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<TransferSession> session,
      CreateReshardReceiveSession(
          &base, allocator.get(), plan, std::vector<int64_t>{0},
          std::chrono::steady_clock::now() + std::chrono::seconds(2)));
  kv_cache::KVCacheManagerBase::TransferEventHooks hooks;
  hooks.get_session = [session](uint64_t uuid)
      -> absl::StatusOr<std::shared_ptr<TransferSession>> { return session; };
  base.SetTransferEventHooks(std::move(hooks));

  // Pool 1 arrives before Pool 0 and leases host slot 0 (even though
  // pools[1].storage_index == 1).
  std::vector<transport::BlockChunk> chunks1 = base.GetBlockChunks(
      /*layer_idx=*/1, /*shard_idx=*/0, std::vector<int64_t>{0},
      /*total_bytes=*/16, kUuid, /*sender_node_id=*/0);
  ASSERT_EQ(chunks1.size(), 1u);
  EXPECT_THAT(session->ResolveStagingSlot(1), IsOkAndHolds(0u));
  TF_ASSERT_OK_AND_ASSIGN(const absl::Span<const char> pool1_span,
                       base.GetBlockArrayHostSpan(/*block_array_idx=*/1,
                                                  /*shard_idx=*/0, kUuid));
  EXPECT_EQ(pool1_span.data(),
            reinterpret_cast<const char*>(
                base.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0) +
                pools[1].base_offset_bytes));
  EXPECT_EQ(
      pool1_span.size(),
      static_cast<size_t>(pools[1].num_blocks * pools[1].block_stride_bytes));
  const char* chunk1_begin = reinterpret_cast<const char*>(chunks1[0].ptr);
  EXPECT_GE(chunk1_begin, pool1_span.data());
  EXPECT_LE(chunk1_begin + chunks1[0].size,
            pool1_span.data() + pool1_span.size());

  // Pool 0 arrives second and leases host slot 1.
  std::vector<transport::BlockChunk> chunks0 = base.GetBlockChunks(
      /*layer_idx=*/0, /*shard_idx=*/0, std::vector<int64_t>{0},
      /*total_bytes=*/16, kUuid, /*sender_node_id=*/0);
  ASSERT_EQ(chunks0.size(), 1u);
  EXPECT_THAT(session->ResolveStagingSlot(0), IsOkAndHolds(1u));
  TF_ASSERT_OK_AND_ASSIGN(const absl::Span<const char> pool0_span,
                       base.GetBlockArrayHostSpan(/*block_array_idx=*/0,
                                                  /*shard_idx=*/0, kUuid));
  EXPECT_EQ(pool0_span.data(),
            reinterpret_cast<const char*>(
                base.GetHostPointer(/*layer_idx=*/1, /*shard_idx=*/0) +
                pools[0].base_offset_bytes));
  EXPECT_EQ(
      pool0_span.size(),
      static_cast<size_t>(pools[0].num_blocks * pools[0].block_stride_bytes));
  const char* chunk0_begin = reinterpret_cast<const char*>(chunks0[0].ptr);
  EXPECT_GE(chunk0_begin, pool0_span.data());
  EXPECT_LE(chunk0_begin + chunks0[0].size,
            pool0_span.data() + pool0_span.size());

  ABSL_EXPECT_OK(session->OnDataReceived(/*major_dimension_idx=*/1,
                                         /*minor_dimension_indices=*/{},
                                         /*shard_ids=*/{}));
  ABSL_EXPECT_OK(session->OnDataReceived(/*major_dimension_idx=*/0,
                                         /*minor_dimension_indices=*/{},
                                         /*shard_ids=*/{}));
  ABSL_EXPECT_OK(session->AwaitForDone());
  const std::vector<std::pair<size_t, size_t>> expected_uploads = {{1, 0},
                                                                   {0, 1}};
  EXPECT_EQ(base.uploaded_host_slot_by_pool(), expected_uploads);
}

}  // namespace
}  // namespace tpu_raiden
