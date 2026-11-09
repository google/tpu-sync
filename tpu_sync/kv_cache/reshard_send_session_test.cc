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
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/kv_cache/reshard_receive_session.h"
#include "tpu_sync/kv_cache/transfer_plan.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/block_transport.h"
#include "tpu_sync/transport/transfer_session.h"

namespace tpu_raiden {

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
  tsl::Future<> GetDataReadyFuture(size_t major_dimension_idx) override {
    return tsl::Future<>(absl::OkStatus());
  }
  std::optional<std::shared_ptr<const TransferPlan>> plan() const override {
    return std::nullopt;
  }
  std::chrono::steady_clock::time_point deadline() const override {
    return std::chrono::steady_clock::time_point::max();
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

// Returns the status of creating a send session for `plan` on `base`. Create
// validates the plan before anything else that can fail here; the placeholder
// device hold passes its device-attachment check for plans that validate.
absl::Status CreateStatus(kv_cache::KVCacheManagerBase& base,
                          const StartTransferRequest& plan,
                          absl::Span<const int64_t> src_block_ids) {
  base.AttachPlaceholderDeviceHoldForTest();
  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(&base, /*num_slots=*/0, /*max_blocks=*/0);
  return CreateReshardSendSession(
             &base, allocator.get(), src_block_ids, /*parallelism=*/1,
             std::chrono::steady_clock::now() + std::chrono::seconds(10), plan)
      .status();
}

TEST(ReshardSendSessionTest, AcceptsCanonicalPlanOnExplicitPools) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  ASSERT_TRUE(base.RegisterPools({DensePool("fa")}).ok());
  StartTransferRequest plan = ValidPlan(/*uuid=*/1001);

  EXPECT_TRUE(CreateStatus(base, plan, std::vector<int64_t>{0}).ok());
}

TEST(ReshardSendSessionTest, AcceptsImplicitPools) {
  kv_cache::KVCacheManagerBase base = MakeTestBase();
  StartTransferRequest plan = ValidPlan(/*uuid=*/1002, /*dtype_tags=*/{""});

  EXPECT_TRUE(CreateStatus(base, plan, std::vector<int64_t>{0}).ok());
}

TEST(ReshardSendSessionTest, RejectsMissingIdentityAndPoolFields) {
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

  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
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
  EXPECT_TRUE(CreateStatus(base, plan, std::vector<int64_t>{0}).ok());
  auto* entry = plan.mutable_shard_push_schedules()->at(0).mutable_entries(0);
  entry->set_src_offset_bytes(32);
  entry->set_size_bytes(16);

  ExpectInvalid(CreateStatus(base, plan, std::vector<int64_t>{0}),
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

  EXPECT_TRUE(CreateStatus(base, plan, std::vector<int64_t>{0}).ok());
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

// A CPU-only sender base whose pool device-to-host copies need no device. A
// copy of a pool passed to HoldNextCopy() stays pending until CompleteCopy();
// every other copy is ready as soon as it is issued.
class FakeD2hBase : public kv_cache::KVCacheManagerBase {
 public:
  using kv_cache::KVCacheManagerBase::KVCacheManagerBase;

  void HoldNextCopy(size_t pool_idx) {
    absl::MutexLock lock(fake_mu_);
    hold_next_[pool_idx] = true;
  }

  void FillOnCopy(size_t pool_idx, uint8_t fill_byte) {
    absl::MutexLock lock(fake_mu_);
    fill_byte_[pool_idx] = fill_byte;
  }

  // Completes the pending copy of `pool_idx`.
  void CompleteCopy(size_t pool_idx) {
    xla::Promise<> promise;
    {
      absl::MutexLock lock(fake_mu_);
      auto it = pending_.find(pool_idx);
      ASSERT_NE(it, pending_.end()) << "no pending copy of pool " << pool_idx;
      promise = std::move(it->second);
      pending_.erase(it);
    }
    promise.Set();
  }

  int CopiesIssued(size_t pool_idx) const {
    absl::MutexLock lock(fake_mu_);
    auto it = copies_issued_.find(pool_idx);
    return it == copies_issued_.end() ? 0 : it->second;
  }

  // Waits up to 10s for `count` copies of `pool_idx` to have been issued.
  bool WaitForCopies(size_t pool_idx, int count) const {
    const absl::Time give_up = absl::Now() + absl::Seconds(10);
    while (CopiesIssued(pool_idx) < count) {
      if (absl::Now() > give_up) return false;
      absl::SleepFor(absl::Milliseconds(1));
    }
    return true;
  }

  absl::StatusOr<raiden::PjRtCopyFuture> D2hPoolBlocks(
      size_t pool_idx, absl::Span<const int64_t> block_ids,
      std::optional<size_t> shard_idx, std::optional<uint64_t> uuid,
      std::optional<size_t> host_layer_idx) override {
    auto [promise, future] = xla::MakePromise();
    absl::MutexLock lock(fake_mu_);
    if (auto it = fill_byte_.find(pool_idx);
        it != fill_byte_.end() && host_layer_idx.has_value()) {
      const kv_cache::PoolSpec* spec = pool(pool_idx);
      if (spec != nullptr) {
        uint8_t* dst = GetHostPointer(*host_layer_idx, 0);
        if (dst != nullptr) {
          std::memset(dst + spec->base_offset_bytes, it->second, 16);
        }
      }
    }
    ++copies_issued_[pool_idx];
    if (hold_next_[pool_idx]) {
      hold_next_[pool_idx] = false;
      pending_[pool_idx] = std::move(promise);
    } else {
      promise.Set();
    }
    return raiden::PjRtCopyFuture(std::move(future), raiden::BufferHolders{});
  }

 private:
  mutable absl::Mutex fake_mu_;
  absl::flat_hash_map<size_t, bool> hold_next_ ABSL_GUARDED_BY(fake_mu_);
  absl::flat_hash_map<size_t, uint8_t> fill_byte_ ABSL_GUARDED_BY(fake_mu_);
  absl::flat_hash_map<size_t, xla::Promise<>> pending_
      ABSL_GUARDED_BY(fake_mu_);
  absl::flat_hash_map<size_t, int> copies_issued_ ABSL_GUARDED_BY(fake_mu_);
};

// Routes `sender`'s transport session lookups to `session`, as the manager
// that registers a session does, so pushes address its leased staging slots.
void RouteSessionLookups(kv_cache::KVCacheManagerBase& sender,
                         const std::shared_ptr<TransferSession>& session) {
  kv_cache::KVCacheManagerBase::TransferEventHooks hooks;
  hooks.get_session =
      [weak = std::weak_ptr<TransferSession>(session)](
          uint64_t uuid) -> absl::StatusOr<std::shared_ptr<TransferSession>> {
    if (std::shared_ptr<TransferSession> live = weak.lock()) return live;
    return absl::NotFoundError(absl::StrCat("no session for uuid ", uuid));
  };
  sender.SetTransferEventHooks(std::move(hooks));
}

// Makes `receiver` accept every push into `recv_session`.
void AcceptPushes(kv_cache::KVCacheManagerBase& receiver,
                  const std::shared_ptr<TransferSession>& recv_session) {
  kv_cache::KVCacheManagerBase::TransferEventHooks hooks;
  hooks.get_session =
      [recv_session](
          uint64_t uuid) -> absl::StatusOr<std::shared_ptr<TransferSession>> {
    return recv_session;
  };
  receiver.SetTransferEventHooks(std::move(hooks));
}

// Waits up to `timeout` for `session` to settle.
bool SettlesWithin(const TransferSession& session, absl::Duration timeout) {
  const absl::Time give_up = absl::Now() + timeout;
  while (!session.Done()) {
    if (absl::Now() > give_up) return false;
    absl::SleepFor(absl::Milliseconds(1));
  }
  return true;
}

// Waits up to 10s for `session` to start draining.
bool DrainsWithin10s(const TransferSession& session) {
  const absl::Time give_up = absl::Now() + absl::Seconds(10);
  while (!session.IsDraining()) {
    if (absl::Now() > give_up) return false;
    absl::SleepFor(absl::Milliseconds(1));
  }
  return true;
}

TEST(ReshardSendSessionTest,
     PerTransferPolicyClosesIdlePeerConnectionsBeforeSettling) {
  ScopedDataConnReuseEnv env("per_transfer");
  FakeD2hBase sender(/*num_layers=*/1, /*num_shards=*/1,
                     /*slice_byte_size=*/128, /*local_port=*/std::nullopt,
                     /*host_blocks_to_allocate=*/std::make_optional(4));
  kv_cache::KVCacheManagerBase receiver = MakeTestBase();
  const std::string peer =
      absl::StrCat("127.0.0.1:", receiver.transport_server()->local_port());
  transport::BlockTransport& transport = *sender.transport_server();
  ABSL_ASSERT_OK(SeedIdleDataConnection(transport, peer));
  ASSERT_EQ(transport.IdleConnectionCount(peer), 1u);
  AcceptPushes(receiver, std::make_shared<StubRecvSession>());
  sender.AttachPlaceholderDeviceHoldForTest();
  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(&sender, /*num_slots=*/0, /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(/*uuid=*/1201, /*dtype_tags=*/{""});
  (*plan.mutable_shard_push_schedules())[0].mutable_entries(0)->set_dst_peer(
      peer);
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<TransferSession> session,
      CreateReshardSendSession(
          &sender, allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::seconds(10), plan));
  RouteSessionLookups(sender, session);

  // The plan's only push completes on a transport thread. The send's last op
  // closes the idle connections to its peer before the send settles: none is
  // left once the send is done. This thread polls Done() instead of blocking
  // in AwaitForDone(): a blocked waiter wakes too late to catch a release
  // that trails the settle.
  ABSL_ASSERT_OK(session->Start());
  const absl::Time give_up = absl::Now() + absl::Seconds(10);
  while (!session->Done()) {
    ASSERT_LT(absl::Now(), give_up) << "send did not settle";
  }
  EXPECT_EQ(transport.IdleConnectionCount(peer), 0u);
  ABSL_EXPECT_OK(session->AwaitForDone());
}

TEST(ReshardSendSessionTest,
     PerTransferPolicyClosesIdlePeerConnectionsWhenCancelledMidCopy) {
  ScopedDataConnReuseEnv env("per_transfer");
  FakeD2hBase sender(/*num_layers=*/1, /*num_shards=*/1,
                     /*slice_byte_size=*/128, /*local_port=*/std::nullopt,
                     /*host_blocks_to_allocate=*/std::make_optional(4));
  kv_cache::KVCacheManagerBase receiver = MakeTestBase();
  const std::string peer =
      absl::StrCat("127.0.0.1:", receiver.transport_server()->local_port());
  transport::BlockTransport& transport = *sender.transport_server();
  ABSL_ASSERT_OK(SeedIdleDataConnection(transport, peer));
  ASSERT_EQ(transport.IdleConnectionCount(peer), 1u);
  sender.AttachPlaceholderDeviceHoldForTest();
  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(&sender, /*num_slots=*/0, /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(/*uuid=*/1204, /*dtype_tags=*/{""});
  (*plan.mutable_shard_push_schedules())[0].mutable_entries(0)->set_dst_peer(
      peer);
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<TransferSession> session,
      CreateReshardSendSession(
          &sender, allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::seconds(10), plan));
  RouteSessionLookups(sender, session);
  sender.HoldNextCopy(/*pool_idx=*/0);
  ABSL_ASSERT_OK(session->Start());
  ASSERT_TRUE(sender.WaitForCopies(/*pool_idx=*/0, /*count=*/1));

  // Cancelling the send while its pool copy runs starts finalizing it, but
  // the copy in flight keeps it from settling, and nothing is released yet.
  session->Finish(absl::CancelledError("transfer cancelled"));
  EXPECT_FALSE(session->Done());
  EXPECT_EQ(transport.IdleConnectionCount(peer), 1u);
  EXPECT_TRUE(session->plan().has_value());

  // The copy's completion is the send's last op: it drops the plan and closes
  // the idle connections to its peer, then the send settles.
  sender.CompleteCopy(/*pool_idx=*/0);

  EXPECT_THAT(session->AwaitForDone(), StatusIs(absl::StatusCode::kCancelled));
  EXPECT_TRUE(session->Done());
  EXPECT_EQ(transport.IdleConnectionCount(peer), 0u);
  EXPECT_FALSE(session->plan().has_value());
}

TEST(ReshardSendSessionTest, PooledPolicyKeepsIdlePeerConnections) {
  ScopedDataConnReuseEnv env(/*value=*/nullptr);
  FakeD2hBase sender(/*num_layers=*/1, /*num_shards=*/1,
                     /*slice_byte_size=*/128, /*local_port=*/std::nullopt,
                     /*host_blocks_to_allocate=*/std::make_optional(4));
  kv_cache::KVCacheManagerBase receiver = MakeTestBase();
  const std::string peer =
      absl::StrCat("127.0.0.1:", receiver.transport_server()->local_port());
  transport::BlockTransport& transport = *sender.transport_server();
  ABSL_ASSERT_OK(SeedIdleDataConnection(transport, peer));
  ASSERT_EQ(transport.IdleConnectionCount(peer), 1u);
  AcceptPushes(receiver, std::make_shared<StubRecvSession>());
  sender.AttachPlaceholderDeviceHoldForTest();
  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(&sender, /*num_slots=*/0, /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(/*uuid=*/1202, /*dtype_tags=*/{""});
  (*plan.mutable_shard_push_schedules())[0].mutable_entries(0)->set_dst_peer(
      peer);
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<TransferSession> session,
      CreateReshardSendSession(
          &sender, allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::seconds(10), plan));
  RouteSessionLookups(sender, session);

  ABSL_ASSERT_OK(session->Start());
  ABSL_EXPECT_OK(session->AwaitForDone());

  EXPECT_TRUE(session->Done());
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
      CreateReshardSendSession(
          &base, allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::seconds(10), plan),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("TPU_RAIDEN_DATA_CONN_REUSE"),
                     HasSubstr("per-transfer"))));
  EXPECT_FALSE(base.HasActivePlan(plan.uuid()));
}

TEST(ReshardSendSessionTest, BoundedStagingSlotAcquisitionAndRelease) {
  // Three pools share the sender's only host layer buffer.
  FakeD2hBase sender(
      /*num_layers=*/3, /*num_shards=*/1,
      /*slice_byte_size=*/128,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/std::make_optional(4),
      /*parallelism=*/1, /*host_allocator=*/nullptr,
      /*bind_ip=*/std::nullopt, /*num_staging_layers=*/1);
  sender.AttachPlaceholderDeviceHoldForTest();
  ASSERT_EQ(sender.num_host_layers(), 1u);
  ASSERT_EQ(sender.num_layers(), 3u);
  std::vector<kv_cache::PoolSpec> pools = {
      DensePool("fa", 128),
      DensePool("state", 128),
      DensePool("conv", 128),
  };
  pools[1].storage_index = 1;
  pools[2].storage_index = 2;
  ABSL_ASSERT_OK(sender.RegisterPools(pools));
  // A receiver without registered pools accepts plan-less pool pushes.
  kv_cache::KVCacheManagerBase receiver(
      /*num_layers=*/3, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/std::make_optional(4));
  AcceptPushes(receiver, std::make_shared<StubRecvSession>());
  const std::string peer =
      absl::StrCat("127.0.0.1:", receiver.transport_server()->local_port());

  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(&sender, /*num_slots=*/0, /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(
      /*uuid=*/1301, /*dtype_tags=*/{"bf16", "bf16", "bf16"},
      /*transferred_pools=*/{0, 1, 2});
  (*plan.mutable_shard_push_schedules())[0].mutable_entries(0)->set_dst_peer(
      peer);
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<TransferSession> session,
      CreateReshardSendSession(
          &sender, allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::seconds(10), plan));
  RouteSessionLookups(sender, session);
  sender.HoldNextCopy(/*pool_idx=*/0);
  sender.HoldNextCopy(/*pool_idx=*/1);
  ABSL_ASSERT_OK(session->Start());

  // Pool 0 leases slot 0 (the only allocated host layer) for its copy.
  ASSERT_TRUE(sender.WaitForCopies(/*pool_idx=*/0, /*count=*/1));
  EXPECT_THAT(session->ResolveStagingSlot(0), IsOkAndHolds(0u));

  // Pool 1 waits for slot 0 until pool 0's push releases its lease.
  absl::SleepFor(absl::Milliseconds(10));
  EXPECT_EQ(sender.CopiesIssued(1), 0);
  EXPECT_THAT(session->ResolveStagingSlot(1),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  sender.CompleteCopy(/*pool_idx=*/0);
  ASSERT_TRUE(sender.WaitForCopies(/*pool_idx=*/1, /*count=*/1));
  EXPECT_THAT(session->ResolveStagingSlot(1), IsOkAndHolds(0u));

  // Pool 2 now waits for slot 0. Cancelling the session wakes that wait with
  // CancelledError, long before the deadline, and releases every staging slot
  // once the session settles.
  absl::SleepFor(absl::Milliseconds(10));
  session->Finish(absl::CancelledError("cancelled"));
  sender.CompleteCopy(/*pool_idx=*/1);
  EXPECT_TRUE(SettlesWithin(*session, absl::Seconds(5)));
  EXPECT_THAT(session->GetStatus(), StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(sender.CopiesIssued(2), 0);
  EXPECT_THAT(session->ResolveStagingSlot(1),
              StatusIs(absl::StatusCode::kCancelled));

  // Timeout when no slot is released before the session deadline: pool 0
  // holds slot 0 for its copy and pool 1's wait runs into the deadline.
  StartTransferRequest timeout_plan = ValidPlan(
      /*uuid=*/1302, /*dtype_tags=*/{"bf16", "bf16", "bf16"},
      /*transferred_pools=*/{0, 1, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<TransferSession> timeout_session,
      CreateReshardSendSession(
          &sender, allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::milliseconds(30),
          timeout_plan));
  RouteSessionLookups(sender, timeout_session);
  sender.HoldNextCopy(/*pool_idx=*/0);
  ABSL_ASSERT_OK(timeout_session->Start());
  ASSERT_TRUE(sender.WaitForCopies(/*pool_idx=*/0, /*count=*/2));
  EXPECT_THAT(timeout_session->ResolveStagingSlot(0), IsOkAndHolds(0u));
  ASSERT_TRUE(DrainsWithin10s(*timeout_session));
  sender.CompleteCopy(/*pool_idx=*/0);
  EXPECT_THAT(timeout_session->AwaitForDone(),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_EQ(sender.CopiesIssued(1), 1);
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
      std::shared_ptr<TransferSession> session,
      CreateReshardSendSession(
          manager.base(), allocator.get(), std::vector<int64_t>{0},
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
  FakeD2hBase sender(
      /*num_layers=*/2, /*num_shards=*/1,
      /*slice_byte_size=*/128,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/std::make_optional(4),
      /*parallelism=*/1, /*host_allocator=*/nullptr,
      /*bind_ip=*/std::nullopt, /*num_staging_layers=*/1);
  ASSERT_EQ(sender.num_host_layers(), 1u);
  std::vector<kv_cache::PoolSpec> pools = {
      DensePool("fa", 128),
      DensePool("state", 128),
  };
  pools[1].storage_index = 1;
  ABSL_ASSERT_OK(sender.RegisterPools(pools));
  sender.AttachPlaceholderDeviceHoldForTest();
  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(&sender, /*num_slots=*/0, /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(
      /*uuid=*/1402, /*dtype_tags=*/{"bf16", "bf16"},
      /*transferred_pools=*/{0, 1});
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<TransferSession> session,
      CreateReshardSendSession(
          &sender, allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::milliseconds(200),
          plan));
  RouteSessionLookups(sender, session);

  // Pool 0's copy holds the only slot and never completes before the
  // deadline, so pool 1's slot wait can only end at the deadline.
  sender.HoldNextCopy(/*pool_idx=*/0);
  ABSL_ASSERT_OK(session->Start());
  ASSERT_TRUE(sender.WaitForCopies(/*pool_idx=*/0, /*count=*/1));
  ASSERT_TRUE(DrainsWithin10s(*session));
  EXPECT_FALSE(session->Done());
  sender.CompleteCopy(/*pool_idx=*/0);

  EXPECT_THAT(session->AwaitForDone(),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_TRUE(session->Done());
  EXPECT_FALSE(session->plan().has_value());
  EXPECT_EQ(sender.CopiesIssued(1), 0);
  // SettleLocked released every lease along with everything else.
  EXPECT_THAT(session->ResolveStagingSlot(0),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_THAT(session->ResolveStagingSlot(1),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
}

// A pool pushing to several peers keeps its staging slot until the last
// peer push completes, then the send settles clean.
TEST(ReshardSendSessionTest, StagingSlotReleasedAfterLastPeerPush) {
  FakeD2hBase sender(/*num_layers=*/1, /*num_shards=*/1,
                     /*slice_byte_size=*/128, /*local_port=*/std::nullopt,
                     /*host_blocks_to_allocate=*/std::make_optional(4));
  kv_cache::KVCacheManagerBase receiver_a = MakeTestBase();
  kv_cache::KVCacheManagerBase receiver_b = MakeTestBase();
  auto recv_session = std::make_shared<StubRecvSession>();
  AcceptPushes(receiver_a, recv_session);
  AcceptPushes(receiver_b, recv_session);
  const std::string peer_a =
      absl::StrCat("127.0.0.1:", receiver_a.transport_server()->local_port());
  const std::string peer_b =
      absl::StrCat("127.0.0.1:", receiver_b.transport_server()->local_port());
  sender.AttachPlaceholderDeviceHoldForTest();
  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(&sender, /*num_slots=*/0, /*max_blocks=*/0);
  StartTransferRequest plan = ValidPlan(/*uuid=*/1403, /*dtype_tags=*/{""});
  auto* schedule = &(*plan.mutable_shard_push_schedules())[0];
  schedule->mutable_entries(0)->set_dst_peer(peer_a);
  *schedule->add_entries() = schedule->entries(0);
  schedule->mutable_entries(1)->set_dst_peer(peer_b);
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<TransferSession> session,
      CreateReshardSendSession(
          &sender, allocator.get(), std::vector<int64_t>{0},
          /*parallelism=*/1,
          std::chrono::steady_clock::now() + std::chrono::seconds(10), plan));
  RouteSessionLookups(sender, session);

  // Pool 0 leases its staging slot for the copy, then issues its two peer
  // pushes once the copy lands.
  sender.HoldNextCopy(/*pool_idx=*/0);
  ABSL_ASSERT_OK(session->Start());
  ASSERT_TRUE(sender.WaitForCopies(/*pool_idx=*/0, /*count=*/1));
  EXPECT_THAT(session->ResolveStagingSlot(0), IsOkAndHolds(0u));
  sender.CompleteCopy(/*pool_idx=*/0);

  ABSL_EXPECT_OK(session->AwaitForDone());
  EXPECT_TRUE(session->Done());
  EXPECT_FALSE(session->plan().has_value());
  EXPECT_THAT(session->ResolveStagingSlot(0),
              StatusIs(absl::StatusCode::kCancelled));
}

class RecordingH2dReceiverBase : public kv_cache::KVCacheManagerBase {
 public:
  explicit RecordingH2dReceiverBase(size_t num_layers)
      : KVCacheManagerBase(num_layers, /*num_shards=*/1,
                           /*slice_byte_size=*/128,
                           /*local_port=*/std::nullopt,
                           /*host_blocks_to_allocate=*/std::make_optional(4)) {}

  absl::StatusOr<raiden::PjRtCopyFuture> H2dPoolBlocks(
      size_t pool_idx, absl::Span<const int64_t> block_ids,
      std::optional<size_t> shard_idx, std::optional<uint64_t> uuid,
      std::optional<size_t> host_layer_idx) override {
    if (!host_layer_idx.has_value() || block_ids.size() != 1) {
      return absl::InvalidArgumentError(
          "expected a single resolved host staging block");
    }
    const kv_cache::PoolSpec* spec = pool(pool_idx);
    if (spec == nullptr) {
      return absl::InvalidArgumentError("unknown pool_idx");
    }
    const uint8_t* src = GetHostPointer(*host_layer_idx, /*shard_idx=*/0) +
                         spec->base_offset_bytes +
                         static_cast<size_t>(block_ids[0]) *
                             static_cast<size_t>(spec->block_stride_bytes);
    {
      absl::MutexLock lock(mu_);
      uploaded_bytes_[pool_idx].assign(src, src + 16);
      uploaded_host_slots_[pool_idx] = *host_layer_idx;
    }
    auto [promise, future] = xla::MakePromise();
    promise.Set();
    return raiden::PjRtCopyFuture(std::move(future), raiden::BufferHolders{});
  }

  std::vector<uint8_t> uploaded_bytes(size_t pool_idx) const {
    absl::MutexLock lock(mu_);
    auto it = uploaded_bytes_.find(pool_idx);
    if (it == uploaded_bytes_.end()) {
      return {};
    }
    return it->second;
  }

  std::optional<size_t> uploaded_host_slot(size_t pool_idx) const {
    absl::MutexLock lock(mu_);
    auto it = uploaded_host_slots_.find(pool_idx);
    if (it == uploaded_host_slots_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<size_t, std::vector<uint8_t>> uploaded_bytes_
      ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<size_t, size_t> uploaded_host_slots_ ABSL_GUARDED_BY(mu_);
};

TEST(ReshardSendSessionTest,
     PoolOrderInvertedFromStorageOrderPassesTransportValidateChunks) {
  std::vector<kv_cache::PoolSpec> pools = {DensePool("fa"), DensePool("state")};
  pools[0].storage_index = 1;
  pools[0].base_offset_bytes = 16;
  pools[0].num_blocks = 3;
  pools[1].storage_index = 0;
  pools[1].base_offset_bytes = 32;
  pools[1].num_blocks = 3;

  FakeD2hBase sender(/*num_layers=*/2, /*num_shards=*/1,
                     /*slice_byte_size=*/128, /*local_port=*/std::nullopt,
                     /*host_blocks_to_allocate=*/std::make_optional(4));
  sender.set_node_id(0);
  ABSL_ASSERT_OK(sender.RegisterPools(pools));
  sender.FillOnCopy(/*pool_idx=*/0, 0xA5);
  sender.FillOnCopy(/*pool_idx=*/1, 0x5A);

  RecordingH2dReceiverBase receiver(/*num_layers=*/2);
  ABSL_ASSERT_OK(receiver.RegisterPools(pools));
  const std::string peer =
      absl::StrCat("127.0.0.1:", receiver.transport_server()->local_port());

  sender.AttachPlaceholderDeviceHoldForTest();
  receiver.AttachPlaceholderDeviceHoldForTest();
  std::unique_ptr<StagingBlockAllocator> send_allocator =
      CreateStagingBlockAllocator(&sender, /*num_slots=*/0, /*max_blocks=*/0);
  std::unique_ptr<StagingBlockAllocator> recv_allocator =
      CreateStagingBlockAllocator(&receiver, /*num_slots=*/0, /*max_blocks=*/0);

  StartTransferRequest plan = ValidPlan(
      /*uuid=*/1404, /*dtype_tags=*/{"bf16", "bf16"},
      /*transferred_pools=*/{0, 1});
  (*plan.mutable_shard_push_schedules())[0].mutable_entries(0)->set_dst_peer(
      peer);

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<TransferSession> recv_session,
      CreateReshardReceiveSession(&receiver, recv_allocator.get(), plan,
                                  std::vector<int64_t>{0}, deadline));
  RouteSessionLookups(receiver, recv_session);
  ABSL_ASSERT_OK(recv_session->Start());

  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<TransferSession> send_session,
      CreateReshardSendSession(&sender, send_allocator.get(),
                               std::vector<int64_t>{0}, /*parallelism=*/2,
                               deadline, plan));
  RouteSessionLookups(sender, send_session);

  ABSL_ASSERT_OK(send_session->Start());
  ABSL_EXPECT_OK(send_session->AwaitForDone());
  ABSL_EXPECT_OK(recv_session->AwaitForDone());

  EXPECT_EQ(receiver.uploaded_bytes(0), std::vector<uint8_t>(16, 0xA5));
  EXPECT_EQ(receiver.uploaded_bytes(1), std::vector<uint8_t>(16, 0x5A));
  // A full mirror stages each pool in its own storage's buffer.
  EXPECT_EQ(receiver.uploaded_host_slot(0), std::make_optional<size_t>(1));
  EXPECT_EQ(receiver.uploaded_host_slot(1), std::make_optional<size_t>(0));
}

}  // namespace
}  // namespace tpu_raiden
