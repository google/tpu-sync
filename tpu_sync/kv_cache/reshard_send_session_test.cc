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

#include "tpu_sync/kv_cache/reshard_send_session.h"

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
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
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/kv_cache/transfer_plan.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/block_transport.h"
#include "tpu_sync/transport/transfer_session.h"

namespace tpu_raiden {

struct ReshardSendSessionTestPeer {
  static absl::StatusOr<size_t> AcquireStagingSlot(ReshardSendSession& session,
                                                   size_t pool_idx) {
    return session.AcquireStagingSlot(pool_idx);
  }

  static absl::Status ValidatePlan(
      const kv_cache::KVCacheManagerBase& base,
      const ::tpu_sync::rpc::StartTransferRequest& plan,
      absl::Span<const int64_t> src_block_ids) {
    return ReshardSendSession::ValidatePlan(base, plan, src_block_ids);
  }

  static std::optional<size_t> ReleaseStagingSlot(ReshardSendSession& session,
                                                  size_t pool_idx) {
    return session.staging_layer_manager_.Release(pool_idx);
  }

  // Issues `pool_idx`'s peer pushes as the D2H completion callback does.
  static void StartPoolPush(ReshardSendSession& session, size_t pool_idx) {
    session.StartPoolPush(pool_idx);
  }

  // Counts one (pool, peer) push of `session` in flight on the manager it was
  // created with, as Start and StartPoolPush do before issuing it. The
  // executor's device copies are device-only, so deviceless tests drive the op
  // accounting here.
  static void BeginPush(ReshardSendSession& session) {
    absl::MutexLock lock(session.mu_);
    ++session.in_flight_;
  }

  // Ends that push the way its AsyncPush completion callback does.
  static void CompletePush(ReshardSendSession& session,
                           const absl::Status& status) {
    session.RecordPushCompletion(status);
    session.EndOp();
  }

  static void RecordPushCompletion(ReshardSendSession& session,
                                   const absl::Status& status) {
    session.RecordPushCompletion(status);
  }

  static void EndOp(ReshardSendSession& session) { session.EndOp(); }
};

namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;
using ::tpu_sync::rpc::MEMORY_TYPE_HBM;
using ::tpu_sync::rpc::StartTransferRequest;

class StubRecvSession : public TransferSession {
 public:
  absl::Status Start() override { return absl::OkStatus(); }
  bool Done() const override { return false; }
  void Finish(const absl::Status& status) override {}
  bool TryBeginOp() override { return true; }
  void EndOp() override {}
  absl::Status GetStatus() const override { return absl::OkStatus(); }
  absl::Status AwaitForDone() override { return absl::OkStatus(); }
  bool IsDraining() const override { return false; }
  absl::StatusOr<size_t> ResolveStagingSlot(size_t unit) override {
    return unit;
  }
  absl::Status OnDataReceived(size_t major_dimension_idx,
                              absl::Span<const int> minor_dimension_indices,
                              absl::Span<const int> shard_ids) override {
    return absl::OkStatus();
  }
  std::optional<std::shared_ptr<const TransferPlan>> plan() const override {
    return std::nullopt;
  }
};

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
  return sender.AsyncPush({peer}, /*src_block_ids=*/{0}, /*dst_block_ids=*/{0})
      .Await()
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
      CreateStagingBlockAllocator(manager.base(), /*num_slots=*/0,
                                  /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(/*uuid=*/1201, /*dtype_tags=*/{""});
  (*plan.mutable_shard_push_schedules())[0].mutable_entries(0)->set_dst_peer(
      peer);
  absl::StatusOr<std::shared_ptr<ReshardSendSession>> session =
      ReshardSendSession::Create(
          manager.base(), &manager, allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::seconds(10), plan);
  ABSL_ASSERT_OK(session);
  ReshardSendSessionTestPeer::BeginPush(**session);

  // The plan's only push completes on another thread, as its AsyncPush
  // callback would. That push is the send's last op, so it closes the idle
  // connections to its peer before the send settles: none is left once the
  // send is done. This thread polls Done() instead of blocking in
  // AwaitForDone(): a blocked waiter wakes too late to catch a release that
  // trails the settle.
  std::thread last_push([&session] {
    ReshardSendSessionTestPeer::CompletePush(**session, absl::OkStatus());
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
      CreateStagingBlockAllocator(manager.base(), /*num_slots=*/0,
                                  /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(/*uuid=*/1204, /*dtype_tags=*/{""});
  (*plan.mutable_shard_push_schedules())[0].mutable_entries(0)->set_dst_peer(
      peer);
  absl::StatusOr<std::shared_ptr<ReshardSendSession>> session =
      ReshardSendSession::Create(
          manager.base(), &manager, allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::seconds(10), plan);
  ABSL_ASSERT_OK(session);
  ReshardSendSessionTestPeer::BeginPush(**session);

  // Cancelling the send while its push runs starts finalizing it, but the
  // push in flight keeps it from settling, and nothing is released yet.
  (*session)->Finish(absl::CancelledError("transfer cancelled"));
  EXPECT_FALSE((*session)->Done());
  EXPECT_EQ(transport.IdleConnectionCount(peer), 1u);
  EXPECT_TRUE((*session)->plan().has_value());

  // The push's completion is the send's last op: it drops the plan and
  // closes the idle connections to its peer, then the send settles.
  ReshardSendSessionTestPeer::CompletePush(**session, absl::OkStatus());

  EXPECT_TRUE((*session)->Done());
  EXPECT_EQ(transport.IdleConnectionCount(peer), 0u);
  EXPECT_FALSE((*session)->plan().has_value());
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
      CreateStagingBlockAllocator(manager.base(), /*num_slots=*/0,
                                  /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(/*uuid=*/1202, /*dtype_tags=*/{""});
  (*plan.mutable_shard_push_schedules())[0].mutable_entries(0)->set_dst_peer(
      peer);
  absl::StatusOr<std::shared_ptr<ReshardSendSession>> session =
      ReshardSendSession::Create(
          manager.base(), &manager, allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::seconds(10), plan);
  ABSL_ASSERT_OK(session);

  ReshardSendSessionTestPeer::BeginPush(**session);
  ReshardSendSessionTestPeer::CompletePush(**session, absl::OkStatus());

  EXPECT_TRUE((*session)->Done());
  EXPECT_EQ(transport.IdleConnectionCount(peer), 1u);
}

TEST(ReshardSendSessionTest, CreateRejectsUnknownDataConnReuseValue) {
  ScopedDataConnReuseEnv env("per-transfer");
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  base.AttachPlaceholderDeviceHoldForTest();
  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(&base, /*num_slots=*/0, /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(/*uuid=*/1203, /*dtype_tags=*/{""});

  EXPECT_THAT(
      ReshardSendSession::Create(
          &base, /*manager=*/nullptr, allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::seconds(10), plan),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("TPU_RAIDEN_DATA_CONN_REUSE"),
                     HasSubstr("per-transfer"))));
  EXPECT_FALSE(base.HasActivePlan(plan.uuid()));
}

TEST(ReshardSendSessionTest, BoundedStagingSlotAcquisitionAndRelease) {
  kv_cache::KVCacheManagerBase base(
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
      std::shared_ptr<ReshardSendSession> session,
      ReshardSendSession::Create(
          &base, /*manager=*/nullptr, allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::seconds(2), plan));

  // Pool 0 leases slot 0 (the only allocated host layer).
  EXPECT_THAT(ReshardSendSessionTestPeer::AcquireStagingSlot(*session, 0),
              IsOkAndHolds(0u));
  EXPECT_THAT(session->ResolveStagingSlot(0), IsOkAndHolds(0u));

  // Pool 1 blocks waiting for slot 0 until pool 0 releases its lease.
  absl::StatusOr<size_t> slot1;
  std::thread release_waiter([&]() {
    slot1 = ReshardSendSessionTestPeer::AcquireStagingSlot(*session, 1);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_EQ(ReshardSendSessionTestPeer::ReleaseStagingSlot(*session, 0),
            std::make_optional(0u));
  release_waiter.join();
  EXPECT_THAT(slot1, IsOkAndHolds(0u));
  EXPECT_THAT(session->ResolveStagingSlot(1), IsOkAndHolds(0u));

  // Cancelling the session while pool 0 is blocked in AcquireStagingSlot wakes
  // the waiter with CancelledError and releases all staging slots on settle.
  absl::StatusOr<size_t> slot0_cancelled;
  std::thread cancel_waiter([&]() {
    slot0_cancelled =
        ReshardSendSessionTestPeer::AcquireStagingSlot(*session, 0);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  session->Finish(absl::CancelledError("cancelled"));
  cancel_waiter.join();
  EXPECT_THAT(slot0_cancelled, StatusIs(absl::StatusCode::kCancelled));
  EXPECT_TRUE(session->Done());
  EXPECT_THAT(session->ResolveStagingSlot(1),
              StatusIs(absl::StatusCode::kCancelled));

  // Timeout when no slot is released before the session deadline.
  StartTransferRequest timeout_plan = ValidPlan(
      /*uuid=*/1302, /*dtype_tags=*/{"bf16", "bf16"},
      /*transferred_pools=*/{0, 1});
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<ReshardSendSession> timeout_session,
      ReshardSendSession::Create(
          &base, /*manager=*/nullptr, allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::milliseconds(30),
          timeout_plan));
  EXPECT_THAT(
      ReshardSendSessionTestPeer::AcquireStagingSlot(*timeout_session, 0),
      IsOkAndHolds(0u));
  EXPECT_THAT(
      ReshardSendSessionTestPeer::AcquireStagingSlot(*timeout_session, 1),
      StatusIs(absl::StatusCode::kDeadlineExceeded));
}

// A precondition that needs no waiting fails Start synchronously, so
// the START_TRANSFER reply carries it, and the send settles with its plan
// dropped rather than waiting for the deadline sweep.
TEST(ReshardSendSessionTest, StartFailsFastWithoutTransportServer) {
  KVCacheManagerWithTransfer manager = MakeTestManager();
  manager.base()->AttachPlaceholderDeviceHoldForTest();
  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(manager.base(), /*num_slots=*/0,
                                  /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(/*uuid=*/1401, /*dtype_tags=*/{""});
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<ReshardSendSession> session,
      ReshardSendSession::Create(
          manager.base(), &manager, allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::seconds(10), plan));
  ASSERT_TRUE(session->plan().has_value());

  manager.base()->StopTransportServer();
  EXPECT_THAT(session->Start(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                         HasSubstr("transport server")));
  EXPECT_TRUE(session->Done());
  EXPECT_THAT(session->GetStatus(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_FALSE(session->plan().has_value());
  // A finished send refuses further pushes with its terminal status.
  EXPECT_THAT(session->Start(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

// A failure on the pipeline thread (here: the staging slot wait running
// into the session deadline) is reported out of band: Start returns
// OK, the send settles with the error, every lease is released, and the plan
// is dropped.
TEST(ReshardSendSessionTest, PipelineSlotWaitTimeoutSettlesSendOutOfBand) {
  KVCacheManagerWithTransfer manager(
      /*num_layers=*/2, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/std::make_optional(4));
  ASSERT_EQ(manager.base()->num_host_layers(), 2u);
  std::vector<kv_cache::PoolSpec> pools = {
      DensePool("fa", 128),
      DensePool("state", 128),
  };
  pools[1].storage_index = 1;
  ABSL_ASSERT_OK(manager.base()->RegisterPools(pools));
  manager.base()->AttachPlaceholderDeviceHoldForTest();
  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(manager.base(), /*num_slots=*/0,
                                  /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(
      /*uuid=*/1402, /*dtype_tags=*/{"bf16", "bf16"},
      /*transferred_pools=*/{0, 1});
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<ReshardSendSession> session,
      ReshardSendSession::Create(
          manager.base(), &manager, allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::milliseconds(200),
          plan));

  // Park every slot on units the pipeline never releases, so pool 0's
  // AcquireStagingSlot can only end at the deadline.
  constexpr size_t kParkedUnits[] = {7, 8};
  EXPECT_THAT(
      ReshardSendSessionTestPeer::AcquireStagingSlot(*session, kParkedUnits[0]),
      IsOkAndHolds(0u));
  EXPECT_THAT(
      ReshardSendSessionTestPeer::AcquireStagingSlot(*session, kParkedUnits[1]),
      IsOkAndHolds(1u));

  ABSL_ASSERT_OK(session->Start());
  EXPECT_THAT(session->AwaitForDone(),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_TRUE(session->Done());
  EXPECT_FALSE(session->plan().has_value());
  // SettleLocked released the parked leases along with everything else.
  for (size_t unit : kParkedUnits) {
    EXPECT_THAT(session->ResolveStagingSlot(unit),
                StatusIs(absl::StatusCode::kDeadlineExceeded));
  }
  EXPECT_THAT(session->ResolveStagingSlot(0),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
}

// A pool pushing to several peers keeps its staging slot until the last
// peer push completes, then the send settles clean.
TEST(ReshardSendSessionTest, StagingSlotReleasedAfterLastPeerPush) {
  KVCacheManagerWithTransfer manager = MakeTestManager();
  kv_cache::KVCacheManagerBase receiver_a = MakeTestBase();
  kv_cache::KVCacheManagerBase receiver_b = MakeTestBase();
  auto recv_session = std::make_shared<StubRecvSession>();
  for (kv_cache::KVCacheManagerBase* receiver : {&receiver_a, &receiver_b}) {
    kv_cache::KVCacheManagerBase::TransferEventHooks hooks;
    hooks.get_session =
        [recv_session](
            uint64_t uuid) -> absl::StatusOr<std::shared_ptr<TransferSession>> {
      return recv_session;
    };
    receiver->SetTransferEventHooks(std::move(hooks));
  }
  const std::string peer_a =
      absl::StrCat("127.0.0.1:", receiver_a.transport_server()->local_port());
  const std::string peer_b =
      absl::StrCat("127.0.0.1:", receiver_b.transport_server()->local_port());
  manager.base()->AttachPlaceholderDeviceHoldForTest();
  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(manager.base(), /*num_slots=*/0,
                                  /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(/*uuid=*/1403, /*dtype_tags=*/{""});
  auto* schedule = &(*plan.mutable_shard_push_schedules())[0];
  schedule->mutable_entries(0)->set_dst_peer(peer_a);
  *schedule->add_entries() = schedule->entries(0);
  schedule->mutable_entries(1)->set_dst_peer(peer_b);
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<ReshardSendSession> session,
      ReshardSendSession::Create(
          manager.base(), /*manager=*/nullptr, allocator.get(),
          std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::seconds(10), plan));

  // Stage pool 0 as the pipeline would, then issue its two peer pushes.
  EXPECT_THAT(ReshardSendSessionTestPeer::AcquireStagingSlot(*session, 0),
              IsOkAndHolds(0u));
  ReshardSendSessionTestPeer::StartPoolPush(*session, /*pool_idx=*/0);

  ABSL_EXPECT_OK(session->AwaitForDone());
  EXPECT_TRUE(session->Done());
  EXPECT_FALSE(session->plan().has_value());
  EXPECT_THAT(session->ResolveStagingSlot(0),
              StatusIs(absl::StatusCode::kCancelled));
}

TEST(ReshardSendSessionTest, FinishAfterLastPushFinalizingPreservesOkStatus) {
  KVCacheManagerWithTransfer manager = MakeTestManager();
  manager.base()->AttachPlaceholderDeviceHoldForTest();
  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(manager.base(), /*num_slots=*/0,
                                  /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(/*uuid=*/1404, /*dtype_tags=*/{""});
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<ReshardSendSession> session,
      ReshardSendSession::Create(
          manager.base(), &manager, allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::seconds(10), plan));

  ReshardSendSessionTestPeer::BeginPush(*session);
  ReshardSendSessionTestPeer::RecordPushCompletion(*session, absl::OkStatus());
  EXPECT_FALSE(session->Done());

  // A concurrent Finish(error) while finalizing is already true (before EndOp
  // settles the session) must not overwrite the completed OK status.
  session->Finish(absl::CancelledError("receive plan unregistered"));
  ReshardSendSessionTestPeer::EndOp(*session);

  EXPECT_TRUE(session->Done());
  ABSL_EXPECT_OK(session->GetStatus());
}

}  // namespace
}  // namespace tpu_raiden
