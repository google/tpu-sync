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

// Deviceless executor unit tier (X1/X4): the ValidatePoolReshardPlan
// accept/reject table, the device-only rejection at the public entry points,
// the tag-neutral skip summary, and the sender-side pool selection (including
// the no-bytes-owned sender completing without device work). The executor
// byte path and the pool receive lifecycle are device-only by design and are
// exercised on real chips by the D-series harness.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/reshard_receive_session.h"
#include "tpu_sync/core/reshard_send_session.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/mock_metrics_backend.h"

namespace tpu_raiden {

struct PoolReshardRecvTestPeer {
  static void FinishPoolH2d(ReshardReceiveSession& session,
                            KVCacheManagerWithTransfer& manager,
                            size_t pool_idx, const absl::Status& status) {
    session.FinishPoolH2d(manager, pool_idx, status);
  }
};

namespace {

using ::testing::_;
using ::testing::Ge;
using ::testing::Contains;
using ::testing::IsEmpty;
using ::tpu_sync::rpc::MEMORY_TYPE_DRAM;
using ::tpu_sync::rpc::MEMORY_TYPE_HBM;
using ::tpu_sync::rpc::ShardPushEntryProto;
using ::tpu_sync::rpc::StartTransferRequest;

class TestManager : public KVCacheManagerWithTransfer {
 public:
  explicit TestManager(double timeout_s = 30.0)
      : KVCacheManagerWithTransfer(
            /*num_layers=*/1, /*num_shards=*/1,
            /*slice_byte_size=*/128,
            /*local_port=*/std::nullopt,
            /*host_blocks_to_allocate=*/std::make_optional(4),
            /*parallelism=*/1, /*node_id=*/0,
            /*local_control_port=*/-1, /*max_blocks=*/0, /*num_slots=*/0,
            timeout_s) {}

  void FinishPoolReshardRecvPool(uint64_t uuid, size_t pool_idx,
                                 const absl::Status& status) {
    std::shared_ptr<ReshardReceiveSession> session;
    {
      absl::MutexLock lock(mu_);
      auto it = active_pool_reshard_recvs_.find(uuid);
      if (it == active_pool_reshard_recvs_.end()) return;
      session = it->second;
    }
    PoolReshardRecvTestPeer::FinishPoolH2d(*session, *this, pool_idx, status);
  }
  using KVCacheManagerWithTransfer::PoolReshardRegisterRecv;

  // Passes the device-attached gate with no real device state. Only paths
  // that never touch the holds (validation and the no-bytes-owned sender
  // completion) may rely on it.
  void AttachPlaceholderDeviceHold() {
    base()->AttachPlaceholderDeviceHoldForTest();
  }
  // Stages plans in per-transfer host blocks, as
  // TPU_RAIDEN_DYNAMIC_HOST_STAGING=1 does for device-attached managers.
  void EnableDemandStaging() {
    staging_allocator_ = StagingBlockAllocator::Create(
        base_.get(), staging_allocator_->num_slots(),
        staging_allocator_->max_blocks(), /*dynamic_host_staging=*/true);
  }
};

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

// A block-addressed plan (no pool groups) that moves `src` to `dst` on one
// shard, destined for HBM or for host memory.
StartTransferRequest BlockPlan(int64_t uuid, const std::vector<int32_t>& src,
                               const std::vector<int32_t>& dst,
                               ::tpu_sync::rpc::MemoryType dst_mem_type) {
  StartTransferRequest plan;
  plan.set_uuid(uuid);
  plan.set_req_id("block_plan_req_" + std::to_string(uuid));
  plan.set_dst_mem_type(dst_mem_type);
  plan.set_use_block_chunks(true);
  plan.set_parallelism(1);
  auto* schedule = &(*plan.mutable_shard_push_schedules())[0];
  for (size_t i = 0; i < src.size(); ++i) {
    auto* entry = schedule->add_entries();
    entry->set_dst_peer("127.0.0.1:1");
    entry->set_dst_shard_idx(0);
    entry->set_src_block_id(src[i]);
    entry->set_dst_block_id(dst[i]);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(16);
    entry->set_src_stride_bytes(0);
    entry->set_dst_stride_bytes(0);
    entry->set_count(1);
  }
  return plan;
}

TEST(ExpectedPushSendersTest, CountsReceiverPlanSchedules) {
  TestManager manager;
  EXPECT_EQ(manager.base()->ExpectedPushSenders(/*uuid=*/1), std::nullopt);

  // A block-addressed receive plan assembled from two source ranks, each
  // writing its own head slice of the same destination block.
  StartTransferRequest receive_plan;
  receive_plan.set_uuid(1);
  receive_plan.set_use_block_chunks(true);
  for (int source_rank : {4, 5}) {
    auto* entry =
        (*receive_plan.mutable_shard_push_schedules())[source_rank].add_entries();
    entry->set_dst_peer("127.0.0.1:1");
    entry->set_src_block_id(0);
    entry->set_dst_block_id(1);
    entry->set_dst_offset_bytes(source_rank == 4 ? 0 : 16);
    entry->set_size_bytes(16);
    entry->set_src_stride_bytes(16);
    entry->set_dst_stride_bytes(32);
    entry->set_count(2);
  }
  ASSERT_TRUE(
      manager.RegisterActivePlan(1, receive_plan, /*is_sender=*/false).ok());
  EXPECT_EQ(manager.base()->ExpectedPushSenders(1), std::optional<size_t>(2));

  // A sender's own plan never gates what it receives.
  StartTransferRequest send_plan;
  send_plan.set_uuid(2);
  send_plan.set_is_sender(true);
  send_plan.set_use_block_chunks(true);
  auto* entry = (*send_plan.mutable_shard_push_schedules())[0].add_entries();
  entry->set_dst_peer("127.0.0.1:1");
  entry->set_src_block_id(0);
  entry->set_dst_block_id(0);
  entry->set_size_bytes(16);
  entry->set_count(1);
  ASSERT_TRUE(manager.RegisterActivePlan(2, send_plan, /*is_sender=*/true).ok());
  EXPECT_EQ(manager.base()->ExpectedPushSenders(2), std::nullopt);

  // A plan without schedules declares nothing about its senders.
  StartTransferRequest bare_plan;
  ASSERT_TRUE(
      manager.RegisterActivePlan(3, bare_plan, /*is_sender=*/false).ok());
  EXPECT_EQ(manager.base()->ExpectedPushSenders(3), std::nullopt);

  ASSERT_TRUE(manager.UnregisterActivePlan(1).ok());
  EXPECT_EQ(manager.base()->ExpectedPushSenders(1), std::nullopt);
}

TEST(PoolReshardValidationTest, DeviceOnlyRejectionAtPublicEntryPoints) {
  TestManager manager;
  ASSERT_TRUE(manager.base()->RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1004);

  const absl::Status recv_status =
      manager.PoolReshardRegisterRecv(plan, std::vector<int64_t>{0});
  EXPECT_EQ(recv_status.code(), absl::StatusCode::kFailedPrecondition)
      << recv_status.ToString();
  EXPECT_NE(std::string(recv_status.message()).find("device-attached"),
            std::string::npos);

  const absl::Status push_status =
      manager.PoolReshardPush(plan, std::vector<int64_t>{0});
  EXPECT_EQ(push_status.code(), absl::StatusCode::kFailedPrecondition)
      << push_status.ToString();
}

// Extends ValidPlan's single-group shape with a second group holding pool 1;
// the base plan's only schedule entry stays in group 0.
StartTransferRequest TwoGroupPlan(int64_t uuid) {
  StartTransferRequest plan = ValidPlan(uuid, /*dtype_tags=*/{"bf16", "bf16"},
                                        /*transferred_pools=*/{0});
  auto* group = plan.add_pool_groups();
  group->add_pool_indices(1);
  group->add_dst_device_block_ids(0);
  group->set_expected_pushes(1);
  group->add_dst_expected_extent_bytes(16);
  return plan;
}

TEST(PoolReshardSendTest,
     SenderWithNoBytesForAnyTransferredPoolRefusedUpFront) {
  TestManager manager;
  ASSERT_TRUE(manager.base()
                  ->RegisterPools({DensePool("fa"), DensePool("state")})
                  .ok());
  manager.AttachPlaceholderDeviceHold();

  // The sender's only schedule entry names group 1, while the transfer set
  // holds just pool 0 (group 0) — the shape of a PCP rank owning no bytes of
  // any transferred pool. With CountPoolReshardSendSlots, a plan that
  // schedules zero pushes is refused up front (and unregisters itself).
  StartTransferRequest plan = TwoGroupPlan(/*uuid=*/2004);
  (*plan.mutable_shard_push_schedules())[0].mutable_entries(0)->set_pool_group(
      1);

  const absl::Status push_status =
      manager.PoolReshardPush(plan, std::vector<int64_t>{0});
  EXPECT_FALSE(push_status.ok());
  EXPECT_EQ(push_status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(push_status.message(),
              ::testing::HasSubstr("sender plan schedules no pushes"));
  EXPECT_FALSE(manager.base()->HasActivePlan(plan.uuid()));
}

TEST(PoolReshardRecvTest, FinishPoolReshardRecvRecordsDurationMetric) {
  TestManager manager;
  ASSERT_TRUE(manager.base()->RegisterPools({DensePool("fa")}).ok());
  manager.AttachPlaceholderDeviceHold();

  auto mock_backend = std::make_unique<telemetry::MockMetricsBackend>();
  telemetry::MockMetricsBackend* raw_mock = mock_backend.get();
  EXPECT_CALL(*raw_mock,
              ObserveHistogram(telemetry::metric_names::kTransferDurationMs,
                               IsEmpty(), Ge(0.0)))
      .Times(1);
  telemetry::ScopedMetricsBackendReset scoped_metrics_reset(
      std::move(mock_backend));

  StartTransferRequest plan = ValidPlan(/*uuid=*/3001);
  ASSERT_TRUE(
      manager.PoolReshardRegisterRecv(plan, std::vector<int64_t>{0}).ok());

  // Simulate pool completion
  manager.FinishPoolReshardRecvPool(3001, /*pool_idx=*/0, absl::OkStatus());
}

TEST(PoolReshardRecvTest, FinishPoolReshardRecvDoesNotRecordMetricOnFailure) {
  TestManager manager;
  ASSERT_TRUE(manager.base()->RegisterPools({DensePool("fa")}).ok());
  manager.AttachPlaceholderDeviceHold();

  auto mock_backend = std::make_unique<telemetry::MockMetricsBackend>();
  telemetry::MockMetricsBackend* raw_mock = mock_backend.get();
  EXPECT_CALL(*raw_mock, ObserveHistogram(_, _, _)).Times(0);
  telemetry::ScopedMetricsBackendReset scoped_metrics_reset(
      std::move(mock_backend));

  StartTransferRequest plan = ValidPlan(/*uuid=*/3002);
  ASSERT_TRUE(
      manager.PoolReshardRegisterRecv(plan, std::vector<int64_t>{0}).ok());

  // Simulate pool failure
  manager.FinishPoolReshardRecvPool(3002, /*pool_idx=*/0,
                                    absl::InternalError("simulated failure"));
}

TEST(SendDeadlineTest, ExpiredSendSessionFailsInsteadOfReportingDone) {
  TestManager manager(/*timeout_s=*/0.05);
  ASSERT_GT(manager.NotifyForRead("expired_send_req", 31, {0, 1}), 0);

  absl::SleepFor(absl::Milliseconds(120));
  const auto [done_sending, done_recving, failed_recving] =
      manager.CompleteReadRaw();
  EXPECT_THAT(done_sending, IsEmpty());
  EXPECT_THAT(failed_recving, Contains("expired_send_req"));
}

TEST(DemandStagingTest, SenderPlanReturnsStagingOnUnregister) {
  TestManager manager;
  manager.EnableDemandStaging();
  auto* pool = manager.base()->host_block_manager();
  const int free_before = pool->num_free_blocks();
  ASSERT_TRUE(manager
                  .RegisterActivePlan(21, BlockPlan(21, {0, 1}, {2, 3},
                                                    MEMORY_TYPE_HBM),
                                      /*is_sender=*/true)
                  .ok());
  EXPECT_EQ(pool->num_free_blocks(), free_before - 2);
  EXPECT_EQ(pool->num_locked_blocks(), 2);
  ASSERT_TRUE(manager.UnregisterActivePlan(21).ok());
  EXPECT_EQ(pool->num_free_blocks(), free_before);
  EXPECT_EQ(pool->num_locked_blocks(), 0);
}

TEST(DemandStagingTest, HostReceiverPlanReturnsStagingOnUnregister) {
  TestManager manager;
  manager.EnableDemandStaging();
  auto* pool = manager.base()->host_block_manager();
  const int free_before = pool->num_free_blocks();
  ASSERT_TRUE(manager
                  .RegisterActivePlan(22, BlockPlan(22, {0, 1}, {2, 3},
                                                    MEMORY_TYPE_DRAM),
                                      /*is_sender=*/false)
                  .ok());
  EXPECT_EQ(pool->num_free_blocks(), free_before - 2);
  EXPECT_EQ(pool->num_locked_blocks(), 2);
  ASSERT_TRUE(manager.UnregisterActivePlan(22).ok());
  EXPECT_EQ(pool->num_free_blocks(), free_before);
  EXPECT_EQ(pool->num_locked_blocks(), 0);
}

TEST(DemandStagingTest, DuplicateSenderRegistrationKeepsOriginalStaging) {
  TestManager manager;
  manager.EnableDemandStaging();
  auto* pool = manager.base()->host_block_manager();
  const int free_before = pool->num_free_blocks();
  const StartTransferRequest plan =
      BlockPlan(23, {0, 1}, {2, 3}, MEMORY_TYPE_HBM);
  ASSERT_TRUE(manager.RegisterActivePlan(23, plan, /*is_sender=*/true).ok());
  const int free_registered = pool->num_free_blocks();
  EXPECT_EQ(free_registered, free_before - 2);

  // The second registration is refused and must neither keep its own
  // staging nor disturb the first plan's.
  const absl::Status duplicate =
      manager.RegisterActivePlan(23, plan, /*is_sender=*/true);
  EXPECT_EQ(duplicate.code(), absl::StatusCode::kAlreadyExists)
      << duplicate.ToString();
  EXPECT_EQ(pool->num_free_blocks(), free_registered);
  EXPECT_EQ(pool->num_locked_blocks(), 2);

  ASSERT_TRUE(manager.UnregisterActivePlan(23).ok());
  EXPECT_EQ(pool->num_free_blocks(), free_before);
  EXPECT_EQ(pool->num_locked_blocks(), 0);
}


TEST(DemandStagingTest, UnregisteringInFlightReceiverDefersUntilItSettles) {
  TestManager manager(/*timeout_s=*/0.05);
  manager.EnableDemandStaging();
  manager.AttachPlaceholderDeviceHold();
  auto* pool = manager.base()->host_block_manager();
  const int free_before = pool->num_free_blocks();
  ASSERT_TRUE(manager
                  .RegisterActivePlan(24, BlockPlan(24, {0, 1}, {2, 3},
                                                    MEMORY_TYPE_HBM),
                                      /*is_sender=*/false)
                  .ok());
  EXPECT_EQ(pool->num_free_blocks(), free_before - 2);
  const auto staged = manager.base()->PlanHostBlocks(24, {2, 3});
  ASSERT_TRUE(staged.ok()) << staged.status();

  // Unregistering while the receive is in flight keeps the plan's mapping
  // and its staging, so pushes already accepted still land in the plan's
  // blocks ...
  ASSERT_TRUE(manager.UnregisterActivePlan(24).ok());
  const auto still_staged = manager.base()->PlanHostBlocks(24, {2, 3});
  ASSERT_TRUE(still_staged.ok()) << still_staged.status();
  EXPECT_EQ(*still_staged, *staged);
  EXPECT_EQ(pool->num_free_blocks(), free_before - 2);

  // ... until the receive settles; here it times out. Then the plan, its
  // staging and the receive session go together.
  absl::SleepFor(absl::Milliseconds(120));
  const auto [done_sending, done_recving, failed_recving] =
      manager.CompleteReadRaw();
  EXPECT_THAT(failed_recving, Contains("block_plan_req_24"));
  EXPECT_EQ(manager.base()->PlanHostBlocks(24, {2, 3}).status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(pool->num_free_blocks(), free_before);
  EXPECT_EQ(pool->num_locked_blocks(), 0);
}

TEST(DemandStagingTest, PlanHostBlocksFailsClosed) {
  TestManager manager;
  manager.EnableDemandStaging();
  auto* pool = manager.base()->host_block_manager();
  EXPECT_EQ(manager.base()->PlanHostBlocks(25, {0}).status().code(),
            absl::StatusCode::kNotFound);

  // Occupy the identity blocks first, so the plan's host blocks provably
  // differ from its device blocks.
  const auto occupied = pool->Allocate(2, /*lock=*/true);
  ASSERT_TRUE(occupied.ok());
  ASSERT_TRUE(manager
                  .RegisterActivePlan(25, BlockPlan(25, {0, 1}, {2, 3},
                                                    MEMORY_TYPE_HBM),
                                      /*is_sender=*/true)
                  .ok());
  const auto staged = manager.base()->PlanHostBlocks(25, {0, 1});
  ASSERT_TRUE(staged.ok()) << staged.status();
  ASSERT_EQ(staged->size(), 2u);
  EXPECT_NE(*staged, (std::vector<int64_t>{0, 1}));
  EXPECT_TRUE(pool->IsLocked(static_cast<int>((*staged)[0])));
  EXPECT_TRUE(pool->IsLocked(static_cast<int>((*staged)[1])));
  // A block the plan does not stage is refused rather than passed through.
  EXPECT_EQ(manager.base()->PlanHostBlocks(25, {0, 7}).status().code(),
            absl::StatusCode::kInvalidArgument);

  ASSERT_TRUE(manager.UnregisterActivePlan(25).ok());
  EXPECT_EQ(manager.base()->PlanHostBlocks(25, {0}).status().code(),
            absl::StatusCode::kNotFound);
}

TEST(DemandStagingTest, PlanHostBlocksIsIdentityForFixedStaging) {
  TestManager manager;
  ASSERT_TRUE(manager
                  .RegisterActivePlan(26, BlockPlan(26, {0, 1}, {2, 3},
                                                    MEMORY_TYPE_HBM),
                                      /*is_sender=*/true)
                  .ok());
  const auto staged = manager.base()->PlanHostBlocks(26, {0, 1});
  ASSERT_TRUE(staged.ok()) << staged.status();
  EXPECT_EQ(*staged, (std::vector<int64_t>{0, 1}));
  // Identity covers only the blocks the plan names.
  EXPECT_EQ(manager.base()->PlanHostBlocks(26, {7}).status().code(),
            absl::StatusCode::kInvalidArgument);
  ASSERT_TRUE(manager.UnregisterActivePlan(26).ok());
}

TEST(DemandStagingTest, PlanHostBlocksRejectsBlocksOfAnEmptyPlan) {
  TestManager manager;
  manager.EnableDemandStaging();
  ASSERT_TRUE(manager
                  .RegisterActivePlan(28, BlockPlan(28, {}, {},
                                                    MEMORY_TYPE_HBM),
                                      /*is_sender=*/true)
                  .ok());
  EXPECT_EQ(manager.base()->PlanHostBlocks(28, {0}).status().code(),
            absl::StatusCode::kInvalidArgument);
  ASSERT_TRUE(manager.UnregisterActivePlan(28).ok());
}

TEST(DemandStagingTest, DemandStagedReceiverPlanUnregistersWhenItSettles) {
  TestManager manager(/*timeout_s=*/0.05);
  manager.EnableDemandStaging();
  manager.AttachPlaceholderDeviceHold();
  auto* pool = manager.base()->host_block_manager();
  const int free_before = pool->num_free_blocks();
  ASSERT_TRUE(manager
                  .RegisterActivePlan(27, BlockPlan(27, {0, 1}, {2, 3},
                                                    MEMORY_TYPE_HBM),
                                      /*is_sender=*/false)
                  .ok());
  ASSERT_TRUE(manager.base()->PlanHostBlocks(27, {2, 3}).ok());

  // Nobody unregisters; the plan still goes when the receive settles (here
  // by timeout), leaving neither a stale mapping nor held blocks behind.
  absl::SleepFor(absl::Milliseconds(120));
  const auto [done_sending, done_recving, failed_recving] =
      manager.CompleteReadRaw();
  EXPECT_THAT(failed_recving, Contains("block_plan_req_27"));
  EXPECT_EQ(manager.base()->PlanHostBlocks(27, {2, 3}).status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(pool->num_free_blocks(), free_before);
  EXPECT_EQ(pool->num_locked_blocks(), 0);
}

}  // namespace
}  // namespace tpu_raiden
