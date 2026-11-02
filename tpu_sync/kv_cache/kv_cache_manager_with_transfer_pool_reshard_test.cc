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
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/kv_cache/reshard_receive_session.h"
#include "tpu_sync/kv_cache/reshard_send_session.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/mock_metrics_backend.h"
#include "tpu_sync/transport/transfer_session.h"

namespace tpu_raiden {

struct PoolReshardRecvTestPeer {
  static void FinishPoolH2d(ReshardReceiveSession& session, size_t pool_idx,
                            const absl::Status& status) {
    session.FinishPoolH2d(pool_idx, status);
  }
};

namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::Contains;
using ::testing::Ge;
using ::testing::IsEmpty;
using ::tpu_sync::rpc::MEMORY_TYPE_DRAM;
using ::tpu_sync::rpc::MEMORY_TYPE_HBM;
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
      session = FindSessionLocked<ReshardReceiveSession>(uuid);
      if (session == nullptr) return;
    }
    PoolReshardRecvTestPeer::FinishPoolH2d(*session, pool_idx, status);
  }
  absl::Status AddPoolReshardSend(const StartTransferRequest& plan,
                                  absl::Span<const int64_t> src_block_ids) {
    absl::StatusOr<std::shared_ptr<ReshardSendSession>> session =
        ReshardSendSession::Create(base(), staging_allocator_.get(),
                                   src_block_ids,
                                   /*parallelism=*/1, DeadlineFromNow(), plan);
    if (!session.ok()) {
      return session.status();
    }
    absl::MutexLock lock(mu_);
    sessions_[plan.uuid()] = *std::move(session);
    return absl::OkStatus();
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
    staging_allocator_ = CreateStagingBlockAllocator(
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

// A receiver plan for `uuid` landing `dst_blocks` on pool 0 in one group:
// every block but the last is covered in full, the last one gets ValidPlan's
// 16-byte head slice.
StartTransferRequest MultiBlockPlan(int64_t uuid,
                                    const std::vector<int64_t>& dst_blocks) {
  StartTransferRequest plan = ValidPlan(uuid);
  auto* group = plan.mutable_pool_groups(0);
  group->clear_dst_device_block_ids();
  group->clear_dst_expected_extent_bytes();
  auto* schedule = &(*plan.mutable_shard_push_schedules())[0];
  schedule->clear_entries();
  for (size_t i = 0; i < dst_blocks.size(); ++i) {
    const int64_t bytes = i + 1 == dst_blocks.size() ? 16 : 128;
    group->add_dst_device_block_ids(dst_blocks[i]);
    group->add_dst_expected_extent_bytes(bytes);
    auto* entry = schedule->add_entries();
    entry->set_dst_peer("127.0.0.1:1");
    entry->set_dst_shard_idx(0);
    entry->set_src_block_id(static_cast<int32_t>(i));
    entry->set_dst_block_id(static_cast<int32_t>(dst_blocks[i]));
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(bytes);
    entry->set_src_stride_bytes(0);
    entry->set_dst_stride_bytes(0);
    entry->set_count(1);
  }
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

// A sender plan's send session belongs to whoever registered the plan: the
// CompleteReadWithDetails sweep never reports it, neither while it is
// registered nor after unregistering cancels it.
TEST(ActivePlanTest, SenderPlanIsNeverReportedAsATransfer) {
  TestManager manager;
  StartTransferRequest send_plan;
  send_plan.set_uuid(7);
  send_plan.set_req_id("sender_plan");
  send_plan.set_is_sender(true);
  send_plan.set_use_block_chunks(true);
  auto* entry = (*send_plan.mutable_shard_push_schedules())[0].add_entries();
  entry->set_dst_peer("127.0.0.1:1");
  entry->set_src_block_id(0);
  entry->set_dst_block_id(0);
  entry->set_size_bytes(16);
  entry->set_count(1);
  ASSERT_TRUE(
      manager.RegisterActivePlan(7, send_plan, /*is_sender=*/true).ok());

  CompleteReadResult registered = manager.CompleteReadWithDetails();
  EXPECT_THAT(registered.sent, IsEmpty());
  EXPECT_THAT(registered.failed, IsEmpty());

  ASSERT_TRUE(manager.UnregisterActivePlan(7).ok());
  EXPECT_THAT(manager.GetSession(7), StatusIs(absl::StatusCode::kNotFound));
  CompleteReadResult unregistered = manager.CompleteReadWithDetails();
  EXPECT_THAT(unregistered.sent, IsEmpty());
  EXPECT_THAT(unregistered.failed, IsEmpty());
  EXPECT_FALSE(unregistered.details.contains("sender_plan"));
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


TEST(DemandStagingTest, UnregisteringIdleReceiverReleasesPlanAtOnce) {
  TestManager manager;
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

  // No push was ever admitted, so unregistering cancels the receive and it
  // settles at once: the plan and its staging go immediately instead of
  // waiting for the deadline sweep.
  ASSERT_TRUE(manager.UnregisterActivePlan(24).ok());
  EXPECT_EQ(manager.base()->PlanHostBlocks(24, {2, 3}).status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(pool->num_free_blocks(), free_before);
  EXPECT_EQ(pool->num_locked_blocks(), 0);

  const auto [done_sending, done_recving, failed_recving] =
      manager.CompleteReadRaw();
  EXPECT_THAT(failed_recving, Contains("block_plan_req_24"));
}

TEST(DemandStagingTest, UnregisteringInFlightReceiverDefersUntilItSettles) {
  TestManager manager;
  manager.EnableDemandStaging();
  manager.AttachPlaceholderDeviceHold();
  auto* pool = manager.base()->host_block_manager();
  const int free_before = pool->num_free_blocks();
  ASSERT_TRUE(manager
                  .RegisterActivePlan(26, BlockPlan(26, {0, 1}, {2, 3},
                                                    MEMORY_TYPE_HBM),
                                      /*is_sender=*/false)
                  .ok());
  const auto staged = manager.base()->PlanHostBlocks(26, {2, 3});
  ASSERT_TRUE(staged.ok()) << staged.status();
  // The transport has admitted a push that is still writing.
  TF_ASSERT_OK_AND_ASSIGN(auto session, manager.base()->GetSession(26));
  EXPECT_TRUE(session->TryBeginOp());

  // Unregistering cancels the receive, but the admitted push keeps the
  // plan's mapping and its staging, so its writes still land in the plan's
  // blocks ...
  ASSERT_TRUE(manager.UnregisterActivePlan(26).ok());
  const auto still_staged = manager.base()->PlanHostBlocks(26, {2, 3});
  ASSERT_TRUE(still_staged.ok()) << still_staged.status();
  EXPECT_EQ(*still_staged, *staged);
  EXPECT_EQ(pool->num_free_blocks(), free_before - 2);
  // ... while no further push is admitted.
  EXPECT_FALSE(session->TryBeginOp());

  // When the admitted push ends the receive settles, and the plan, its
  // staging and the receive session go together.
  session->EndOp();
  EXPECT_TRUE(session->Done());
  EXPECT_EQ(session->GetStatus().code(), absl::StatusCode::kCancelled);
  EXPECT_EQ(manager.base()->PlanHostBlocks(26, {2, 3}).status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(pool->num_free_blocks(), free_before);
  EXPECT_EQ(pool->num_locked_blocks(), 0);

  const auto [done_sending, done_recving, failed_recving] =
      manager.CompleteReadRaw();
  EXPECT_THAT(failed_recving, Contains("block_plan_req_26"));
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

// Declares storage 0 device-backed so RegisterPools configures bounded staging
// without TPU hardware.
class DeviceBackedTestBase : public kv_cache::KVCacheManagerBase {
 public:
  DeviceBackedTestBase(size_t slice_byte_size, int host_blocks,
                       size_t physical_size)
      : KVCacheManagerBase(/*num_layers=*/1, /*num_shards=*/1, slice_byte_size,
                           /*local_port=*/std::nullopt,
                           std::make_optional(host_blocks)) {
    buffer_holds_.resize(1);
    buffer_holds_[0].holds.resize(1);
    buffer_holds_[0].physical_size = physical_size;
    major_dim_size_ = 1;
    layers_[0].shards[0].device_size = physical_size;
  }
};

class BoundedPoolTestManager : public KVCacheManagerWithTransfer {
 public:
  BoundedPoolTestManager(size_t slice_byte_size, int host_blocks,
                         size_t physical_size)
      : KVCacheManagerWithTransfer(
            std::make_unique<DeviceBackedTestBase>(slice_byte_size, host_blocks,
                                                   physical_size),
            /*node_id=*/0, /*local_control_port=*/-1, /*max_blocks=*/0,
            /*num_slots=*/0, /*timeout_s=*/30.0) {}

  using KVCacheManagerWithTransfer::PoolReshardRegisterRecv;
};

TEST(PoolStagingRecvRegistrationTest,
     DuplicatePoolRecvRegistrationKeepsOriginalStagingLease) {
  constexpr int64_t kStride = 128;
  constexpr int64_t kNumBlocks = 16;
  BoundedPoolTestManager manager(/*slice_byte_size=*/kStride, /*host_blocks=*/1,
                                 /*physical_size=*/kStride * kNumBlocks);

  kv_cache::PoolSpec pool = DensePool("fa", kStride, "bf16");
  pool.num_blocks = kNumBlocks;
  pool.staging_blocks_per_request = 1;
  ABSL_ASSERT_OK(manager.base()->RegisterPools({pool}, /*staging_leases=*/2));
  ASSERT_TRUE(manager.base()->PoolStorageStagingBounded(0));
  ASSERT_EQ(manager.base()->PoolStagingSummary()[0].free_slots, 2);

  const StartTransferRequest plan_a = ValidPlan(/*uuid=*/4001);
  ABSL_ASSERT_OK(
      manager.PoolReshardRegisterRecv(plan_a, std::vector<int64_t>{0}));

  EXPECT_EQ(manager.base()->PoolStagingSummary()[0].free_slots, 1);
  const auto chunks_a_before = manager.base()->GetBlockChunks(
      /*layer_idx=*/0, /*shard_idx=*/0, std::vector<int64_t>{0},
      /*total_bytes=*/64, /*uuid=*/4001, /*sender_node_id=*/0);
  ASSERT_EQ(chunks_a_before.size(), 1u);
  const uint8_t* slot_a_ptr = chunks_a_before[0].ptr;
  ASSERT_NE(slot_a_ptr, nullptr);

  // The duplicate registration is refused and must not release 4001's lease.
  const absl::Status duplicate =
      manager.PoolReshardRegisterRecv(plan_a, std::vector<int64_t>{0});
  EXPECT_THAT(duplicate, StatusIs(absl::StatusCode::kAlreadyExists));

  EXPECT_EQ(manager.base()->PoolStagingSummary()[0].free_slots, 1);
  const auto chunks_a_after = manager.base()->GetBlockChunks(
      /*layer_idx=*/0, /*shard_idx=*/0, std::vector<int64_t>{0},
      /*total_bytes=*/64, /*uuid=*/4001, /*sender_node_id=*/0);
  EXPECT_EQ(chunks_a_after.size(), 1u);

  // The next transfer must get the other free slot, not alias 4001's slot.
  StartTransferRequest plan_b = ValidPlan(/*uuid=*/4002);
  plan_b.mutable_pool_groups(0)->set_dst_device_block_ids(0, 5);
  (*plan_b.mutable_shard_push_schedules())[0]
      .mutable_entries(0)
      ->set_dst_block_id(5);
  ABSL_ASSERT_OK(
      manager.PoolReshardRegisterRecv(plan_b, std::vector<int64_t>{5}));
  const auto chunks_b = manager.base()->GetBlockChunks(
      /*layer_idx=*/0, /*shard_idx=*/0, std::vector<int64_t>{5},
      /*total_bytes=*/64, /*uuid=*/4002, /*sender_node_id=*/0);
  ASSERT_EQ(chunks_b.size(), 1u);
  EXPECT_NE(chunks_b[0].ptr, slot_a_ptr);
}

TEST(PoolStagingRecvRegistrationTest,
     ConcurrentDuplicatePoolRecvRegistrationKeepsWinnerStagingLease) {
  constexpr int64_t kStride = 128;
  constexpr int64_t kNumBlocks = 16;
  BoundedPoolTestManager manager(/*slice_byte_size=*/kStride, /*host_blocks=*/1,
                                 /*physical_size=*/kStride * kNumBlocks);

  kv_cache::PoolSpec pool = DensePool("fa", kStride, "bf16");
  pool.num_blocks = kNumBlocks;
  pool.staging_blocks_per_request = 1;
  ABSL_ASSERT_OK(manager.base()->RegisterPools({pool}, /*staging_leases=*/2));
  ASSERT_TRUE(manager.base()->PoolStorageStagingBounded(0));
  ASSERT_EQ(manager.base()->PoolStagingSummary()[0].free_slots, 2);

  for (int i = 0; i < 2; ++i) {
    const uint64_t blocker_uuid = 3998 + i;
    const int64_t blocker_block = 14 + i;
    StartTransferRequest blocker = ValidPlan(blocker_uuid);
    blocker.mutable_pool_groups(0)->set_dst_device_block_ids(0, blocker_block);
    (*blocker.mutable_shard_push_schedules())[0]
        .mutable_entries(0)
        ->set_dst_block_id(blocker_block);
    ABSL_ASSERT_OK(manager.PoolReshardRegisterRecv(
        blocker, std::vector<int64_t>{blocker_block}));
  }
  ASSERT_EQ(manager.base()->PoolStagingSummary()[0].free_slots, 0);

  const StartTransferRequest plan_a = ValidPlan(/*uuid=*/4001);
  absl::Status status_1;
  absl::Status status_2;
  std::thread t1([&]() {
    status_1 = manager.PoolReshardRegisterRecv(plan_a, std::vector<int64_t>{0});
  });
  std::thread t2([&]() {
    status_2 = manager.PoolReshardRegisterRecv(plan_a, std::vector<int64_t>{0});
  });

  absl::SleepFor(absl::Milliseconds(50));
  ABSL_ASSERT_OK(manager.UnregisterActivePlan(3998));
  ABSL_ASSERT_OK(manager.UnregisterActivePlan(3999));

  t1.join();
  t2.join();

  const bool t1_won = status_1.ok();
  const bool t2_won = status_2.ok();
  ASSERT_NE(t1_won, t2_won)
      << "status_1=" << status_1 << " status_2=" << status_2;
  const absl::Status& loser_status = t1_won ? status_2 : status_1;
  EXPECT_THAT(loser_status, StatusIs(absl::StatusCode::kAlreadyExists));

  EXPECT_EQ(manager.base()->PoolStagingSummary()[0].free_slots, 1);
  const auto chunks_a = manager.base()->GetBlockChunks(
      /*layer_idx=*/0, /*shard_idx=*/0, std::vector<int64_t>{0},
      /*total_bytes=*/64, /*uuid=*/4001, /*sender_node_id=*/0);
  EXPECT_EQ(chunks_a.size(), 1u);
}

TEST(PoolStagingRecvRegistrationTest,
     StagingAcquireFailureReleasesUuidClaim) {
  constexpr int64_t kStride = 128;
  constexpr int64_t kNumBlocks = 16;
  BoundedPoolTestManager manager(/*slice_byte_size=*/kStride, /*host_blocks=*/1,
                                 /*physical_size=*/kStride * kNumBlocks);

  kv_cache::PoolSpec pool = DensePool("fa", kStride, "bf16");
  pool.num_blocks = kNumBlocks;
  pool.staging_blocks_per_request = 1;
  ABSL_ASSERT_OK(manager.base()->RegisterPools({pool}, /*staging_leases=*/2));
  ASSERT_TRUE(manager.base()->PoolStorageStagingBounded(0));
  ASSERT_EQ(manager.base()->PoolStagingSummary()[0].free_slots, 2);

  // Three distinct destination blocks on a two-slot arena: Create claims the
  // uuid, then AcquireStagingLeases fails outright (no wait) and must roll the
  // claim back without touching the arena.
  const std::vector<int64_t> too_many_blocks = {0, 1, 2};
  const StartTransferRequest oversized = MultiBlockPlan(/*uuid=*/4001,
                                                        too_many_blocks);
  EXPECT_THAT(manager.PoolReshardRegisterRecv(oversized, too_many_blocks),
              StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_FALSE(manager.base()->HasActivePlan(4001));
  EXPECT_EQ(manager.base()->PoolStagingSummary()[0].free_slots, 2);

  // The uuid is free again: a plan that fits registers normally.
  ABSL_ASSERT_OK(manager.PoolReshardRegisterRecv(ValidPlan(/*uuid=*/4001),
                                                 std::vector<int64_t>{0}));
  EXPECT_TRUE(manager.base()->HasActivePlan(4001));
  EXPECT_EQ(manager.base()->PoolStagingSummary()[0].free_slots, 1);
}

TEST(PoolReshardRecvTest,
     FailedIncomingPushReleasesStagingLeaseBeforeDeadline) {
  constexpr int64_t kStride = 128;
  constexpr int64_t kNumBlocks = 16;
  BoundedPoolTestManager manager(/*slice_byte_size=*/kStride, /*host_blocks=*/1,
                                 /*physical_size=*/kStride * kNumBlocks);

  kv_cache::PoolSpec pool = DensePool("fa", kStride, "bf16");
  pool.num_blocks = kNumBlocks;
  pool.staging_blocks_per_request = 1;
  ABSL_ASSERT_OK(manager.base()->RegisterPools({pool}, /*staging_leases=*/2));
  ASSERT_TRUE(manager.base()->PoolStorageStagingBounded(0));
  ASSERT_EQ(manager.base()->PoolStagingSummary()[0].free_slots, 2);

  const StartTransferRequest plan = ValidPlan(/*uuid=*/3003);
  ABSL_ASSERT_OK(
      manager.PoolReshardRegisterRecv(plan, std::vector<int64_t>{0}));
  EXPECT_EQ(manager.base()->PoolStagingSummary()[0].free_slots, 1);
  EXPECT_TRUE(manager.base()->HasActivePlan(3003));

  // A push that fails mid-stream (socket timeout, peer disconnect) is terminal
  // for the receive: the session must settle now, releasing its staging lease
  // and plan claim, rather than at the session deadline.
  TF_ASSERT_OK_AND_ASSIGN(auto session, manager.base()->GetSession(/*uuid=*/3003));
  EXPECT_TRUE(session->TryBeginOp());
  session->Finish(absl::DeadlineExceededError("Socket read timed out"));
  session->EndOp();
  EXPECT_TRUE(session->Done());
  EXPECT_THAT(session->GetStatus(),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_EQ(manager.base()->PoolStagingSummary()[0].free_slots, 2);
  EXPECT_FALSE(manager.base()->HasActivePlan(3003));

  // The failure is reported on the next poll, not after the deadline sweep.
  const auto [done_sending, done_recving, failed_recving] =
      manager.CompleteReadRaw();
  EXPECT_THAT(done_recving, IsEmpty());
  EXPECT_THAT(failed_recving, Contains("pool_reshard_req_3003"));
}


// ---------------------------------------------------------------------------
// Sub-plan chains: a request coordinated as several pool-reshard plans
// (bounded-staging windows) reports one terminal per role.

StartTransferRequest ChainedRecvPlan(int64_t uuid, int32_t index,
                                     int32_t count) {
  StartTransferRequest plan = ValidPlan(uuid);
  plan.set_req_id("chained_req");
  plan.set_sub_plan_index(index);
  plan.set_sub_plan_count(count);
  // Each sub-plan lands its own destination block.
  plan.mutable_pool_groups(0)->set_dst_device_block_ids(0, index);
  (*plan.mutable_shard_push_schedules())[0]
      .mutable_entries(0)
      ->set_dst_block_id(index);
  return plan;
}

TEST(SubPlanChainTest, ReceiveTerminalWaitsForTheLastSubPlan) {
  TestManager manager;
  ASSERT_TRUE(manager.base()->RegisterPools({DensePool("fa")}).ok());
  manager.AttachPlaceholderDeviceHold();

  ABSL_ASSERT_OK(manager.PoolReshardRegisterRecv(
      ChainedRecvPlan(/*uuid=*/5001, /*index=*/0, /*count=*/3),
      std::vector<int64_t>{0}));
  manager.FinishPoolReshardRecvPool(5001, /*pool_idx=*/0, absl::OkStatus());
  {
    const auto [sent, received, failed] = manager.CompleteReadRaw();
    EXPECT_THAT(received, IsEmpty());
    EXPECT_THAT(failed, IsEmpty());
  }
  // Sub-plans may arrive and settle in any order; the request completes
  // when every one of them has.
  ABSL_ASSERT_OK(manager.PoolReshardRegisterRecv(
      ChainedRecvPlan(5003, /*index=*/2, /*count=*/3), std::vector<int64_t>{2}));
  manager.FinishPoolReshardRecvPool(5003, /*pool_idx=*/0, absl::OkStatus());
  {
    const auto [sent, received, failed] = manager.CompleteReadRaw();
    EXPECT_THAT(received, IsEmpty());
  }
  ABSL_ASSERT_OK(manager.PoolReshardRegisterRecv(
      ChainedRecvPlan(5002, /*index=*/1, /*count=*/3), std::vector<int64_t>{1}));
  manager.FinishPoolReshardRecvPool(5002, /*pool_idx=*/0, absl::OkStatus());
  {
    const auto [sent, received, failed] = manager.CompleteReadRaw();
    EXPECT_THAT(received, Contains("chained_req"));
    EXPECT_THAT(failed, IsEmpty());
  }
  // Reported exactly once.
  {
    const auto [sent, received, failed] = manager.CompleteReadRaw();
    EXPECT_THAT(received, IsEmpty());
  }
}

TEST(SubPlanChainTest, FirstSubPlanFailureFailsTheRequestOnce) {
  TestManager manager;
  ASSERT_TRUE(manager.base()->RegisterPools({DensePool("fa")}).ok());
  manager.AttachPlaceholderDeviceHold();

  ABSL_ASSERT_OK(manager.PoolReshardRegisterRecv(
      ChainedRecvPlan(5011, /*index=*/0, /*count=*/2), std::vector<int64_t>{0}));
  manager.FinishPoolReshardRecvPool(5011, /*pool_idx=*/0,
                                    absl::InternalError("push failed"));
  CompleteReadResult first = manager.CompleteReadWithDetails();
  EXPECT_THAT(first.failed, Contains("chained_req"));
  ASSERT_TRUE(first.details.contains("chained_req"));
  EXPECT_THAT(first.details.at("chained_req").reason.value_or(""),
              ::testing::HasSubstr("sub-plan 0 of 2"));

  // A later sub-plan of the failed request settling cleanly is swallowed:
  // the connector already recomputed the request.
  ABSL_ASSERT_OK(manager.PoolReshardRegisterRecv(
      ChainedRecvPlan(5012, /*index=*/1, /*count=*/2), std::vector<int64_t>{1}));
  manager.FinishPoolReshardRecvPool(5012, /*pool_idx=*/0, absl::OkStatus());
  const auto [sent, received, failed] = manager.CompleteReadRaw();
  EXPECT_THAT(received, IsEmpty());
  EXPECT_THAT(failed, IsEmpty());
}

TEST(SubPlanChainTest, RetryUnderTheSameReqIdIsTrackedAsANewChain) {
  TestManager manager;
  ASSERT_TRUE(manager.base()->RegisterPools({DensePool("fa")}).ok());
  manager.AttachPlaceholderDeviceHold();

  // The first chain fails at sub-plan 0 and is reported.
  ABSL_ASSERT_OK(manager.PoolReshardRegisterRecv(
      ChainedRecvPlan(6001, /*index=*/0, /*count=*/2), std::vector<int64_t>{0}));
  manager.FinishPoolReshardRecvPool(6001, /*pool_idx=*/0,
                                    absl::InternalError("push failed"));
  {
    const auto [sent, received, failed] = manager.CompleteReadRaw();
    EXPECT_THAT(failed, Contains("chained_req"));
  }
  // Its sub-plan 1 is still in flight when the consumer retries the request
  // under the same req_id (a new uuid).
  ABSL_ASSERT_OK(manager.PoolReshardRegisterRecv(
      ChainedRecvPlan(6002, /*index=*/1, /*count=*/2), std::vector<int64_t>{1}));
  ABSL_ASSERT_OK(manager.PoolReshardRegisterRecv(
      ChainedRecvPlan(7001, /*index=*/0, /*count=*/2), std::vector<int64_t>{0}));

  // The stale sub-plan settling neither completes nor fails the retry.
  manager.FinishPoolReshardRecvPool(6002, /*pool_idx=*/0, absl::OkStatus());
  {
    const auto [sent, received, failed] = manager.CompleteReadRaw();
    EXPECT_THAT(received, IsEmpty());
    EXPECT_THAT(failed, IsEmpty());
  }
  manager.FinishPoolReshardRecvPool(7001, /*pool_idx=*/0, absl::OkStatus());
  {
    const auto [sent, received, failed] = manager.CompleteReadRaw();
    EXPECT_THAT(received, IsEmpty());
    EXPECT_THAT(failed, IsEmpty());
  }
  // The retry completes on its own last sub-plan.
  ABSL_ASSERT_OK(manager.PoolReshardRegisterRecv(
      ChainedRecvPlan(7002, /*index=*/1, /*count=*/2), std::vector<int64_t>{1}));
  manager.FinishPoolReshardRecvPool(7002, /*pool_idx=*/0, absl::OkStatus());
  {
    const auto [sent, received, failed] = manager.CompleteReadRaw();
    EXPECT_THAT(received, Contains("chained_req"));
    EXPECT_THAT(failed, IsEmpty());
  }
}

TEST(SubPlanChainTest, StalledChainFailsAfterTheTransferTimeout) {
  TestManager manager(/*timeout_s=*/0.05);
  ASSERT_TRUE(manager.base()->RegisterPools({DensePool("fa")}).ok());
  manager.AttachPlaceholderDeviceHold();

  ABSL_ASSERT_OK(manager.PoolReshardRegisterRecv(
      ChainedRecvPlan(5021, /*index=*/0, /*count=*/2), std::vector<int64_t>{0}));
  manager.FinishPoolReshardRecvPool(5021, /*pool_idx=*/0, absl::OkStatus());
  {
    const auto [sent, received, failed] = manager.CompleteReadRaw();
    EXPECT_THAT(received, IsEmpty());
    EXPECT_THAT(failed, IsEmpty());
  }
  // The second sub-plan never arrives: once the request has had no session
  // for a whole transfer timeout, the sweep fails it so the consumer can
  // recompute instead of waiting forever.
  absl::SleepFor(absl::Milliseconds(120));
  CompleteReadResult swept = manager.CompleteReadWithDetails();
  EXPECT_THAT(swept.failed, Contains("chained_req"));
  ASSERT_TRUE(swept.details.contains("chained_req"));
  EXPECT_THAT(swept.details.at("chained_req").reason.value_or(""),
              ::testing::HasSubstr("sub-plan chain stalled"));
}

TEST(SubPlanChainTest, SinglePlanRequestsReportAtOnce) {
  TestManager manager;
  ASSERT_TRUE(manager.base()->RegisterPools({DensePool("fa")}).ok());
  manager.AttachPlaceholderDeviceHold();

  // sub_plan_count 0 (legacy) and 1 both mean "the whole request".
  StartTransferRequest plan = ValidPlan(/*uuid=*/5031);
  ABSL_ASSERT_OK(
      manager.PoolReshardRegisterRecv(plan, std::vector<int64_t>{0}));
  manager.FinishPoolReshardRecvPool(5031, /*pool_idx=*/0, absl::OkStatus());
  const auto [sent, received, failed] = manager.CompleteReadRaw();
  EXPECT_THAT(received, Contains("pool_reshard_req_5031"));
}

TEST(PoolReshardSessionLookupTest,
     GetSessionResolvesActiveReshardRecvUntilRetired) {
  TestManager manager;
  ABSL_ASSERT_OK(manager.base()->RegisterPools({DensePool("fa")}));
  manager.AttachPlaceholderDeviceHold();

  EXPECT_THAT(manager.GetSession(/*uuid=*/8001),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(manager.base()->GetSession(/*uuid=*/8001),
              StatusIs(absl::StatusCode::kNotFound));

  ABSL_ASSERT_OK(manager.PoolReshardRegisterRecv(ValidPlan(/*uuid=*/8001),
                                                 std::vector<int64_t>{0}));
  absl::StatusOr<std::shared_ptr<TransferSession>> recv_session =
      manager.GetSession(/*uuid=*/8001);
  ABSL_ASSERT_OK(recv_session);
  ASSERT_NE(*recv_session, nullptr);
  EXPECT_FALSE((*recv_session)->Done());
  EXPECT_THAT(manager.base()->GetSession(/*uuid=*/8001),
              IsOkAndHolds(*recv_session));

  manager.FinishPoolReshardRecvPool(/*uuid=*/8001, /*pool_idx=*/0,
                                    absl::OkStatus());
  {
    const auto [sent, received, failed] = manager.CompleteReadRaw();
    EXPECT_THAT(received, Contains("pool_reshard_req_8001"));
    EXPECT_THAT(failed, IsEmpty());
  }
  EXPECT_THAT(manager.GetSession(/*uuid=*/8001),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(manager.base()->GetSession(/*uuid=*/8001),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(PoolReshardSessionLookupTest,
     GetSessionResolvesActiveReshardSendUntilRetired) {
  TestManager manager;
  ABSL_ASSERT_OK(manager.base()->RegisterPools({DensePool("fa")}));
  manager.AttachPlaceholderDeviceHold();

  EXPECT_THAT(manager.GetSession(/*uuid=*/8002),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(manager.base()->GetSession(/*uuid=*/8002),
              StatusIs(absl::StatusCode::kNotFound));

  ABSL_ASSERT_OK(manager.AddPoolReshardSend(ValidPlan(/*uuid=*/8002),
                                            std::vector<int64_t>{0}));
  absl::StatusOr<std::shared_ptr<TransferSession>> send_session =
      manager.GetSession(/*uuid=*/8002);
  ABSL_ASSERT_OK(send_session);
  ASSERT_NE(*send_session, nullptr);
  EXPECT_FALSE((*send_session)->Done());
  EXPECT_THAT(manager.base()->GetSession(/*uuid=*/8002),
              IsOkAndHolds(*send_session));

  (*send_session)->Finish();
  {
    const auto [sent, received, failed] = manager.CompleteReadRaw();
    EXPECT_THAT(sent, Contains("pool_reshard_req_8002"));
    EXPECT_THAT(failed, IsEmpty());
  }
  EXPECT_THAT(manager.GetSession(/*uuid=*/8002),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(manager.base()->GetSession(/*uuid=*/8002),
              StatusIs(absl::StatusCode::kNotFound));
}

// A pool completion always belongs to an armed receiver: with no session for
// the uuid it fails with kNotFound naming the uuid, while the same completion
// for a registered receiver reaches its session.
TEST(PoolReshardSessionLookupTest, PoolCompletionWithoutReceiverIsNotFound) {
  TestManager manager;
  ABSL_ASSERT_OK(manager.base()->RegisterPools({DensePool("fa")}));

  EXPECT_THAT(
      manager.base()->OnPoolReceived(/*pool_idx=*/0, /*uuid=*/8004),
      StatusIs(absl::StatusCode::kNotFound,
               ::testing::HasSubstr("no active receiver for UUID 8004")));

  ABSL_ASSERT_OK(manager.RegisterActivePlan(
      /*uuid=*/8004, ValidPlan(/*uuid=*/8004), /*is_sender=*/false));
  EXPECT_THAT(manager.base()->OnPoolReceived(/*pool_idx=*/0, /*uuid=*/8004),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       ::testing::HasSubstr("armed on the legacy path")));
  ABSL_EXPECT_OK(manager.UnregisterActivePlan(/*uuid=*/8004));
}

// A pool plan armed through RegisterActivePlan lands on a legacy
// TransferReceiveSession, which cannot execute pool completions: the
// completion is rejected before any H2D work, leaving the receive untouched.
TEST(PoolReshardSessionLookupTest,
     PoolCompletionOnLegacyArmedReceiverFailsWithoutH2d) {
  TestManager manager;
  ABSL_ASSERT_OK(manager.base()->RegisterPools({DensePool("fa")}));
  ABSL_ASSERT_OK(manager.RegisterActivePlan(
      /*uuid=*/8003, ValidPlan(/*uuid=*/8003), /*is_sender=*/false));
  absl::StatusOr<std::shared_ptr<TransferSession>> recv_session =
      manager.GetSession(/*uuid=*/8003);
  ABSL_ASSERT_OK(recv_session);
  ASSERT_NE(*recv_session, nullptr);

  EXPECT_THAT(manager.base()->OnPoolReceived(/*pool_idx=*/0, /*uuid=*/8003),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       ::testing::HasSubstr("armed on the legacy path")));

  // The rejection precedes ExecuteLayerH2d, so the receive is still live and
  // unfailed.
  EXPECT_FALSE((*recv_session)->Done());
  EXPECT_FALSE((*recv_session)->IsDraining());
  ABSL_EXPECT_OK((*recv_session)->GetStatus());
  ABSL_EXPECT_OK(manager.UnregisterActivePlan(/*uuid=*/8003));
}

}  // namespace
}  // namespace tpu_raiden
