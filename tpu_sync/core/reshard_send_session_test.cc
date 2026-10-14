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

#include "tpu_sync/core/reshard_send_session.h"

#include <chrono>  // NOLINT(build/c++11)
#include <cstdint>
#include <cstdlib>
#include <limits>
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
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/block_transport.h"

namespace tpu_raiden {

struct ReshardSendSessionTestPeer {
  static absl::Status ValidatePlan(
      const kv_cache::KVCacheManagerBase& base,
      const ::tpu_sync::rpc::StartTransferRequest& plan,
      absl::Span<const int64_t> src_block_ids) {
    return ReshardSendSession::ValidatePlan(base, plan, src_block_ids);
  }

  // Counts one (pool, peer) push of `session` in flight on `manager`, as
  // ExecutePush and StartPoolPush do before issuing it. The executor's device
  // copies are device-only, so deviceless tests drive the op accounting here.
  static void BeginPush(ReshardSendSession& session,
                        KVCacheManagerWithTransfer& manager) {
    absl::MutexLock lock(session.mu_);
    session.manager_ = &manager;
    ++session.in_flight_;
  }

  // Ends that push the way its AsyncPush completion callback does.
  static void CompletePush(ReshardSendSession& session,
                           KVCacheManagerWithTransfer& manager,
                           const absl::Status& status) {
    session.RecordPushCompletion(manager, status);
    session.EndOp();
  }
};

namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;
using ::tpu_sync::rpc::MEMORY_TYPE_HBM;
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

void ExpectInvalid(const absl::Status& status, const std::string& fragment) {
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument)
      << status.ToString();
  EXPECT_NE(std::string(status.message()).find(fragment), std::string::npos)
      << status.ToString();
}

TEST(ReshardSendSessionTest, AcceptsCanonicalPlanOnExplicitPools) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1001);

  EXPECT_TRUE(ReshardSendSessionTestPeer::ValidatePlan(base, plan,
                                                       std::vector<int64_t>{0})
                  .ok());
}

TEST(ReshardSendSessionTest, AcceptsImplicitPools) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  StartTransferRequest plan = ValidPlan(/*uuid=*/1002, /*dtype_tags=*/{""});

  EXPECT_TRUE(ReshardSendSessionTestPeer::ValidatePlan(base, plan,
                                                       std::vector<int64_t>{0})
                  .ok());
}

TEST(ReshardSendSessionTest, RejectsMissingIdentityAndPoolFields) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());

  StartTransferRequest plan = ValidPlan(/*uuid=*/1005);
  plan.clear_req_id();
  ExpectInvalid(ReshardSendSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "req_id");

  plan = ValidPlan(/*uuid=*/0);
  ExpectInvalid(ReshardSendSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "uuid must be positive");

  plan = ValidPlan(/*uuid=*/1006);
  plan.clear_transfer_pool_indices();
  ExpectInvalid(ReshardSendSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "transfer_pool_indices");

  plan = ValidPlan(/*uuid=*/1007);
  plan.clear_pool_groups();
  ExpectInvalid(ReshardSendSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "must declare pool_groups");

  plan = ValidPlan(/*uuid=*/1008);
  plan.set_use_block_chunks(false);
  ExpectInvalid(ReshardSendSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "use_block_chunks");
}

TEST(ReshardSendSessionTest, RejectsOverflowingSenderSpan) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1016);
  auto* entry = plan.mutable_shard_push_schedules()->at(0).mutable_entries(0);
  entry->set_src_offset_bytes(96);
  entry->set_src_stride_bytes(std::numeric_limits<int64_t>::max());
  entry->set_dst_stride_bytes(1);
  entry->set_size_bytes(16);
  entry->set_count(2);

  ExpectInvalid(ReshardSendSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "source span exceeds declared pool 0");
}

TEST(ReshardSendSessionTest,
     RejectsSenderSpanBetweenPackedTokenRegionsInAliasedStorage) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  kv_cache::PoolSpec pool = DensePool("fa");
  pool.regions = {
      kv_cache::RegionSpec{
          .name = "head_group_0",
          .offset_bytes = 0,
          .stride_bytes = 8,
          .unit_bytes = 4,
          .num_units = 4,
          .units_per_stride = 2,
      },
      kv_cache::RegionSpec{
          .name = "head_group_1",
          .offset_bytes = 64,
          .stride_bytes = 8,
          .unit_bytes = 4,
          .num_units = 4,
          .units_per_stride = 2,
      },
  };
  ASSERT_TRUE(base.RegisterPools({pool}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1019);
  EXPECT_TRUE(ReshardSendSessionTestPeer::ValidatePlan(base, plan,
                                                       std::vector<int64_t>{0})
                  .ok());
  auto* entry = plan.mutable_shard_push_schedules()->at(0).mutable_entries(0);
  entry->set_src_offset_bytes(32);
  entry->set_size_bytes(16);

  ExpectInvalid(ReshardSendSessionTestPeer::ValidatePlan(
                    base, plan, std::vector<int64_t>{0}),
                "source span exceeds declared pool 0 live regions");
}

TEST(ReshardSendSessionTest, IgnoresReceiverCoverageGap) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1101);
  plan.mutable_pool_groups(0)->clear_dst_expected_extent_bytes();
  plan.mutable_pool_groups(0)->add_dst_expected_extent_bytes(48);
  auto* entry = (*plan.mutable_shard_push_schedules())[0].add_entries();
  entry->set_dst_peer("127.0.0.1:1");
  entry->set_dst_shard_idx(0);
  entry->set_src_block_id(0);
  entry->set_dst_block_id(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(32);
  entry->set_size_bytes(16);
  entry->set_src_stride_bytes(0);
  entry->set_dst_stride_bytes(0);
  entry->set_count(1);

  EXPECT_TRUE(ReshardSendSessionTestPeer::ValidatePlan(base, plan,
                                                       std::vector<int64_t>{0})
                  .ok());
}

// Sets TPU_RAIDEN_DATA_CONN_REUSE for one test (unsets it when `value` is
// null) and restores it afterwards. Tests construct it before any manager or
// session, so no other thread reads the environment meanwhile.
class ScopedDataConnReuseEnv {
 public:
  explicit ScopedDataConnReuseEnv(const char* value) {
    if (const char* prev = std::getenv(kName)) previous_ = prev;
    if (value == nullptr) {
      unsetenv(kName);
    } else {
      setenv(kName, value, 1);
    }
  }
  ~ScopedDataConnReuseEnv() {
    if (previous_.has_value()) {
      setenv(kName, previous_->c_str(), 1);
    } else {
      unsetenv(kName);
    }
  }

 private:
  static constexpr char kName[] = "TPU_RAIDEN_DATA_CONN_REUSE";
  std::optional<std::string> previous_;
};

// Completes one real push of block 0 from `sender` to `peer`, which leaves the
// push stream's socket idle in `sender`'s connection pool.
absl::Status SeedIdleDataConnection(transport::BlockTransport& sender,
                                    const std::string& peer) {
  return sender.SyncPush({peer}, /*src_block_ids=*/{0}, /*dst_block_ids=*/{0})
      .status();
}

// A CPU-only manager; its base is the sender of the session under test.
KVCacheManagerWithTransfer MakeTestManager() {
  return KVCacheManagerWithTransfer(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/std::make_optional(4));
}

TEST(ReshardSendSessionTest,
     PerTransferPolicyClosesIdlePeerConnectionsBeforeSettling) {
  ScopedDataConnReuseEnv env("per_transfer");
  KVCacheManagerWithTransfer manager = MakeTestManager();
  kv_cache::KVCacheManagerBase receiver = MakeTestBase();
  const std::string peer =
      absl::StrCat("127.0.0.1:", receiver.transport_server()->local_port());
  transport::BlockTransport& transport = *manager.base()->transport_server();
  ABSL_ASSERT_OK(SeedIdleDataConnection(transport, peer));
  ASSERT_EQ(transport.IdleConnectionCount(peer), 1u);
  manager.base()->AttachPlaceholderDeviceHoldForTest();
  std::unique_ptr<StagingBlockAllocator> allocator =
      StagingBlockAllocator::Create(manager.base(), /*num_slots=*/0,
                                  /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(/*uuid=*/1201, /*dtype_tags=*/{""});
  (*plan.mutable_shard_push_schedules())[0].mutable_entries(0)->set_dst_peer(
      peer);
  absl::StatusOr<std::shared_ptr<ReshardSendSession>> session =
      ReshardSendSession::Create(
          manager.base(), allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::seconds(10), plan);
  ABSL_ASSERT_OK(session);
  ReshardSendSessionTestPeer::BeginPush(**session, manager);

  // The plan's only push completes on another thread, as its AsyncPush
  // callback would. That push is the send's last op, so it closes the idle
  // connections to its peer before the send settles: none is left once the
  // send is done. This thread polls Done() instead of blocking in
  // AwaitForDone(): a blocked waiter wakes too late to catch a release that
  // trails the settle.
  std::thread last_push([&session, &manager] {
    ReshardSendSessionTestPeer::CompletePush(**session, manager,
                                             absl::OkStatus());
  });
  while (!(*session)->Done()) {
  }
  EXPECT_EQ(transport.IdleConnectionCount(peer), 0u);
  last_push.join();
  ABSL_EXPECT_OK((*session)->AwaitForDone());
}

TEST(ReshardSendSessionTest,
     PerTransferPolicyClosesIdlePeerConnectionsWhenCancelledMidPush) {
  ScopedDataConnReuseEnv env("per_transfer");
  KVCacheManagerWithTransfer manager = MakeTestManager();
  kv_cache::KVCacheManagerBase receiver = MakeTestBase();
  const std::string peer =
      absl::StrCat("127.0.0.1:", receiver.transport_server()->local_port());
  transport::BlockTransport& transport = *manager.base()->transport_server();
  ABSL_ASSERT_OK(SeedIdleDataConnection(transport, peer));
  ASSERT_EQ(transport.IdleConnectionCount(peer), 1u);
  manager.base()->AttachPlaceholderDeviceHoldForTest();
  std::unique_ptr<StagingBlockAllocator> allocator =
      StagingBlockAllocator::Create(manager.base(), /*num_slots=*/0,
                                  /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(/*uuid=*/1204, /*dtype_tags=*/{""});
  (*plan.mutable_shard_push_schedules())[0].mutable_entries(0)->set_dst_peer(
      peer);
  absl::StatusOr<std::shared_ptr<ReshardSendSession>> session =
      ReshardSendSession::Create(
          manager.base(), allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::seconds(10), plan);
  ABSL_ASSERT_OK(session);
  ReshardSendSessionTestPeer::BeginPush(**session, manager);

  // Cancelling the send while its push runs starts finalizing it, but the
  // push in flight keeps it from settling, and nothing is released yet.
  (*session)->Finish(absl::CancelledError("transfer cancelled"));
  EXPECT_FALSE((*session)->Done());
  EXPECT_EQ(transport.IdleConnectionCount(peer), 1u);
  EXPECT_TRUE(manager.base()->HasActivePlan(plan.uuid()));

  // The push's completion is the send's last op: it drops the plan and
  // closes the idle connections to its peer, then the send settles.
  ReshardSendSessionTestPeer::CompletePush(**session, manager,
                                           absl::OkStatus());

  EXPECT_TRUE((*session)->Done());
  EXPECT_EQ(transport.IdleConnectionCount(peer), 0u);
  EXPECT_FALSE(manager.base()->HasActivePlan(plan.uuid()));
}

TEST(ReshardSendSessionTest, PooledPolicyKeepsIdlePeerConnections) {
  ScopedDataConnReuseEnv env(/*value=*/nullptr);
  KVCacheManagerWithTransfer manager = MakeTestManager();
  kv_cache::KVCacheManagerBase receiver = MakeTestBase();
  const std::string peer =
      absl::StrCat("127.0.0.1:", receiver.transport_server()->local_port());
  transport::BlockTransport& transport = *manager.base()->transport_server();
  ABSL_ASSERT_OK(SeedIdleDataConnection(transport, peer));
  ASSERT_EQ(transport.IdleConnectionCount(peer), 1u);
  manager.base()->AttachPlaceholderDeviceHoldForTest();
  std::unique_ptr<StagingBlockAllocator> allocator =
      StagingBlockAllocator::Create(manager.base(), /*num_slots=*/0,
                                  /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(/*uuid=*/1202, /*dtype_tags=*/{""});
  (*plan.mutable_shard_push_schedules())[0].mutable_entries(0)->set_dst_peer(
      peer);
  absl::StatusOr<std::shared_ptr<ReshardSendSession>> session =
      ReshardSendSession::Create(
          manager.base(), allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::seconds(10), plan);
  ABSL_ASSERT_OK(session);

  ReshardSendSessionTestPeer::BeginPush(**session, manager);
  ReshardSendSessionTestPeer::CompletePush(**session, manager,
                                           absl::OkStatus());

  EXPECT_TRUE((*session)->Done());
  EXPECT_EQ(transport.IdleConnectionCount(peer), 1u);
}

TEST(ReshardSendSessionTest, CreateRejectsUnknownDataConnReuseValue) {
  ScopedDataConnReuseEnv env("per-transfer");
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  base.AttachPlaceholderDeviceHoldForTest();
  std::unique_ptr<StagingBlockAllocator> allocator =
      StagingBlockAllocator::Create(&base, /*num_slots=*/0, /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(/*uuid=*/1203, /*dtype_tags=*/{""});

  EXPECT_THAT(
      ReshardSendSession::Create(
          &base, allocator.get(), std::vector<int64_t>{0}, /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::seconds(10), plan),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("TPU_RAIDEN_DATA_CONN_REUSE"),
                     HasSubstr("per-transfer"))));
  EXPECT_FALSE(base.HasActivePlan(plan.uuid()));
}

}  // namespace
}  // namespace tpu_raiden
