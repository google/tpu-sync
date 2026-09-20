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

#include "tpu_sync/core/reshard_receive_session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {

struct ReshardReceiveSessionTestPeer {
  static absl::Status ValidatePlan(
      const kv_cache::KVCacheManagerBase& base,
      const ::tpu_sync::rpc::StartTransferRequest& plan,
      absl::Span<const int64_t> chip_blocks) {
    return ReshardReceiveSession::ValidatePlan(base, plan, chip_blocks);
  }
};

namespace {

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

TEST(ReshardReceiveSessionTest, AcceptsCanonicalPlanOnExplicitPools) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1001);

  EXPECT_TRUE(ReshardReceiveSessionTestPeer::ValidatePlan(
                  base, plan, std::vector<int64_t>{0})
                  .ok());
}

TEST(ReshardReceiveSessionTest, AcceptsImplicitPools) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  StartTransferRequest plan = ValidPlan(/*uuid=*/1002, /*dtype_tags=*/{""});

  EXPECT_TRUE(ReshardReceiveSessionTestPeer::ValidatePlan(
                  base, plan, std::vector<int64_t>{0})
                  .ok());
}

TEST(ReshardReceiveSessionTest, HasNoTagPolicy) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("gdn.conv")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1003);

  EXPECT_TRUE(ReshardReceiveSessionTestPeer::ValidatePlan(
                  base, plan, std::vector<int64_t>{0})
                  .ok());
}

TEST(ReshardReceiveSessionTest, RejectsMissingIdentityAndPoolFields) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());

  StartTransferRequest plan = ValidPlan(/*uuid=*/1005);
  plan.clear_req_id();
  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "req_id");

  plan = ValidPlan(/*uuid=*/0);
  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "uuid must be positive");

  plan = ValidPlan(/*uuid=*/1006);
  plan.clear_transfer_pool_indices();
  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "transfer_pool_indices");

  plan = ValidPlan(/*uuid=*/1007);
  plan.clear_pool_groups();
  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "must declare pool_groups");

  plan = ValidPlan(/*uuid=*/1008);
  plan.set_use_block_chunks(false);
  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "use_block_chunks");
}

TEST(ReshardReceiveSessionTest, RejectsOutOfRangeDuplicateAndDtypeMismatch) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());

  StartTransferRequest plan = ValidPlan(/*uuid=*/1009, /*dtype_tags=*/{"bf16"},
                                        /*transferred_pools=*/{1});
  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "out of range");

  plan = ValidPlan(/*uuid=*/1010, /*dtype_tags=*/{"bf16"},
                   /*transferred_pools=*/{0, 0});
  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "duplicate transfer pool index");

  plan = ValidPlan(/*uuid=*/1011, /*dtype_tags=*/{"fp8"});
  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "dtype tag mismatch");

  plan = ValidPlan(/*uuid=*/1012, /*dtype_tags=*/{"bf16", "bf16"});
  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
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
  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "destination span exceeds declared pool 1");

  plan = ValidPlan(/*uuid=*/1014, /*dtype_tags=*/{"bf16", "bf16"},
                   /*transferred_pools=*/{0, 1});
  entry = plan.mutable_shard_push_schedules()->at(0).mutable_entries(0);
  entry->set_dst_offset_bytes(16);
  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "no transfer entry starting at offset 0");

  plan = ValidPlan(/*uuid=*/1015, /*dtype_tags=*/{"bf16", "bf16"},
                   /*transferred_pools=*/{0, 1});
  plan.mutable_shard_push_schedules()->at(0).clear_entries();
  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "no entries");
}

TEST(ReshardReceiveSessionTest, RejectsBlockIdsOutsideDeclaredPool) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1017);
  plan.mutable_pool_groups(0)->set_dst_device_block_ids(0, 9);
  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{9}),
                "out of range for pool");

  plan = ValidPlan(/*uuid=*/1018);
  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{8}),
                "must concatenate to the plan's local block ids");
  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0, 0}),
                "must cover the plan's local block ids");
  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{}),
                "must not be empty");
}

TEST(ReshardReceiveSessionTest, RejectsCoverageGap) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1101);
  plan.mutable_pool_groups(0)->clear_dst_expected_extent_bytes();
  plan.mutable_pool_groups(0)->add_dst_expected_extent_bytes(48);
  AddEntry(plan, 0, /*dst_block_id=*/0, /*dst_offset=*/32, /*size=*/16);

  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "coverage gap");
}

TEST(ReshardReceiveSessionTest, RejectsCoverageOverlap) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1102);
  plan.mutable_pool_groups(0)->clear_dst_expected_extent_bytes();
  plan.mutable_pool_groups(0)->add_dst_expected_extent_bytes(24);
  AddEntry(plan, 0, /*dst_block_id=*/0, /*dst_offset=*/8, /*size=*/16);

  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "coverage overlap");
}

TEST(ReshardReceiveSessionTest, RejectsShortCoverage) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1103);
  plan.mutable_pool_groups(0)->clear_dst_expected_extent_bytes();
  plan.mutable_pool_groups(0)->add_dst_expected_extent_bytes(24);

  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
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

  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0, 1}),
                "except the final one");
}

TEST(ReshardReceiveSessionTest, RejectsMissingExtents) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1105);
  plan.mutable_pool_groups(0)->clear_dst_expected_extent_bytes();

  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "require dst_expected_extent_bytes");
}

TEST(ReshardReceiveSessionTest, RejectsMissingParallelism) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1106);
  plan.set_parallelism(0);

  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "positive parallelism");
}

TEST(ReshardReceiveSessionTest, RejectsExpectedPushMismatch) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1107);
  plan.mutable_pool_groups(0)->set_expected_pushes(2);

  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
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

  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "declared live tail");

  plan.mutable_shard_push_schedules()->at(0).mutable_entries(1)->set_size_bytes(
      8);
  EXPECT_TRUE(ReshardReceiveSessionTestPeer::ValidatePlan(
                  base, plan, std::vector<int64_t>{0})
                  .ok());
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

  EXPECT_TRUE(ReshardReceiveSessionTestPeer::ValidatePlan(
                  base, plan, std::vector<int64_t>{2, 3})
                  .ok());

  AddEntry(plan, 0, /*dst_block_id=*/2, /*dst_offset=*/0, /*size=*/8,
           /*group_idx=*/1);
  ExpectInvalid(ReshardReceiveSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{2, 3}),
                "outside its group");
}

}  // namespace
}  // namespace tpu_raiden
