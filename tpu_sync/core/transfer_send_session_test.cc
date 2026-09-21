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

#include "tpu_sync/core/transfer_send_session.h"

#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/future.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/transfer_session.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::StatusIs;

kv_cache::KVCacheManagerBase MakeTestBase(size_t num_layers = 1) {
  return kv_cache::KVCacheManagerBase(
      /*num_layers=*/num_layers, /*num_shards=*/1,
      /*slice_byte_size=*/128,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/std::make_optional(4));
}

class FakeSendBase : public kv_cache::KVCacheManagerBase {
 public:
  explicit FakeSendBase(size_t num_layers = 1, int64_t host_blocks = 4)
      : kv_cache::KVCacheManagerBase(num_layers, /*num_shards=*/1,
                                     std::vector<size_t>(num_layers, 128),
                                     /*local_port=*/std::nullopt,
                                     /*host_blocks_to_allocate=*/host_blocks,
                                     /*parallelism=*/1, nullptr) {}

  void SetManualD2h(bool manual) {
    absl::MutexLock lock(mu_);
    manual_d2h_ = manual;
  }

  void SetManualH2h(bool manual) {
    absl::MutexLock lock(mu_);
    manual_h2h_ = manual;
  }

  void CompleteD2h(size_t index, absl::Status status = absl::OkStatus()) {
    xla::Promise<> promise;
    {
      absl::MutexLock lock(mu_);
      promise = std::move(d2h_promises_.at(index));
    }
    promise.Set(status);
  }

  void CompleteH2h(size_t index, absl::StatusOr<std::vector<int>> res) {
    std::function<void(absl::StatusOr<std::vector<int>>)> cb;
    {
      absl::MutexLock lock(mu_);
      cb = std::move(h2h_callbacks_.at(index));
    }
    cb(std::move(res));
  }

  int d2h_calls() const {
    absl::MutexLock lock(mu_);
    return d2h_calls_;
  }

  int h2h_calls() const {
    absl::MutexLock lock(mu_);
    return h2h_calls_;
  }

  bool WaitForH2hCalls(int expected, absl::Duration timeout) const {
    absl::MutexLock lock(mu_);
    auto cond = [this, expected]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return h2h_calls_ >= expected;
    };
    return mu_.AwaitWithTimeout(absl::Condition(&cond), timeout);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> D2hSyncDispatch(
      const std::vector<int64_t>& src_offsets_major_dim,
      const std::vector<int64_t>& dst_offsets_major_dim,
      const std::vector<int64_t>& copy_sizes_major_dim,
      std::optional<int64_t> slot_idx, std::optional<size_t> layer_idx,
      std::optional<size_t> shard_idx) override {
    absl::MutexLock lock(mu_);
    ++d2h_calls_;
    auto [promise, future] = xla::MakePromise<>();
    if (manual_d2h_) {
      d2h_promises_.push_back(std::move(promise));
    } else {
      promise.Set(absl::OkStatus());
    }
    return raiden::PjRtCopyFuture(std::move(future), raiden::BufferHolders{});
  }

  void H2hWriteDirectAsync(const std::vector<std::string>& peers,
                           const std::vector<int>& src_block_ids,
                           const std::vector<int>& dst_block_ids, uint64_t uuid,
                           int layer_idx,
                           std::function<void(absl::StatusOr<std::vector<int>>)>
                               on_complete) override {
    {
      absl::MutexLock lock(mu_);
      ++h2h_calls_;
      if (manual_h2h_) {
        h2h_callbacks_.push_back(std::move(on_complete));
        return;
      }
    }
    on_complete(src_block_ids);
  }

 private:
  mutable absl::Mutex mu_;
  bool manual_d2h_ ABSL_GUARDED_BY(mu_) = false;
  bool manual_h2h_ ABSL_GUARDED_BY(mu_) = false;
  int d2h_calls_ ABSL_GUARDED_BY(mu_) = 0;
  int h2h_calls_ ABSL_GUARDED_BY(mu_) = 0;
  std::vector<xla::Promise<>> d2h_promises_ ABSL_GUARDED_BY(mu_);
  std::vector<std::function<void(absl::StatusOr<std::vector<int>>)>>
      h2h_callbacks_ ABSL_GUARDED_BY(mu_);
};

TEST(TransferSendSessionTest, SendSessionImplementsTransferSessionInterface) {
  FakeSendBase base(/*num_layers=*/1, /*host_blocks=*/4);
  base.SetManualD2h(true);
  std::unique_ptr<StagingBlockAllocator> allocator =
      StagingBlockAllocator::Create(&base, /*num_slots=*/2, /*max_blocks=*/2);
  const auto now = std::chrono::steady_clock::now();
  std::shared_ptr<TransferSendSession> session = *TransferSendSession::Create(
      &base, allocator.get(), "req1", /*uuid=*/42, /*block_ids=*/{0, 1},
      /*deadline=*/now + std::chrono::seconds(10), /*register_start=*/now);
  TransferSession* base_session = session.get();

  session->ValidateAndBeginPull({0, 1}, now);
  session->StartPush({"127.0.0.1:9000"}, /*src_block_ids=*/{0, 1},
                     /*dst_block_ids=*/{0, 1});
  EXPECT_TRUE(session->HasStaging());
  EXPECT_FALSE(base_session->Done());
  EXPECT_FALSE(base_session->IsDraining());
  ABSL_EXPECT_OK(base_session->GetStatus());

  base_session->Finish(absl::UnavailableError("peer disconnected"));
  EXPECT_TRUE(base_session->IsDraining());
  EXPECT_FALSE(base_session->Done());
  EXPECT_THAT(base_session->GetStatus(),
              StatusIs(absl::StatusCode::kUnavailable));

  base.CompleteD2h(0, absl::OkStatus());
  EXPECT_TRUE(base_session->Done());
  EXPECT_FALSE(session->HasStaging());
  EXPECT_THAT(base_session->AwaitForDone(),
              StatusIs(absl::StatusCode::kUnavailable));
}

TEST(TransferSendSessionTest,
     FinishingOneSendSessionDoesNotShutdownSharedStagingAllocator) {
  FakeSendBase base(/*num_layers=*/1, /*host_blocks=*/4);
  std::unique_ptr<StagingBlockAllocator> allocator =
      StagingBlockAllocator::Create(&base, /*num_slots=*/2, /*max_blocks=*/1);
  const auto now = std::chrono::steady_clock::now();

  // Session 1 completes a full send push and retires.
  std::shared_ptr<TransferSendSession> session1 = *TransferSendSession::Create(
      &base, allocator.get(), "req1", /*uuid=*/101, /*block_ids=*/{0},
      /*deadline=*/now + std::chrono::seconds(5), /*register_start=*/now);
  session1->ValidateAndBeginPull({0}, now);
  session1->StartPush({"127.0.0.1:9000"}, /*src_block_ids=*/{0},
                      /*dst_block_ids=*/{0});
  ABSL_EXPECT_OK(session1->AwaitForDone());
  EXPECT_TRUE(session1->Done());
  EXPECT_FALSE(session1->HasStaging());
  EXPECT_EQ(allocator->num_free_slots(), 2);

  // Session 2 shares the same StagingBlockAllocator and must still be able to
  // acquire staging and complete a push without CancelledError.
  std::shared_ptr<TransferSendSession> session2 = *TransferSendSession::Create(
      &base, allocator.get(), "req2", /*uuid=*/102, /*block_ids=*/{1},
      /*deadline=*/now + std::chrono::seconds(5), /*register_start=*/now);
  session2->ValidateAndBeginPull({1}, now);
  session2->StartPush({"127.0.0.1:9000"}, /*src_block_ids=*/{1},
                      /*dst_block_ids=*/{0});
  ABSL_EXPECT_OK(session2->AwaitForDone());
  EXPECT_TRUE(session2->Done());
  EXPECT_FALSE(session2->HasStaging());
  EXPECT_EQ(allocator->num_free_slots(), 2);
}

TEST(TransferSendSessionTest,
     CancelledWhileWaitingForStagingExitsWithoutShuttingDownAllocator) {
  FakeSendBase base(/*num_layers=*/1, /*host_blocks=*/2);
  std::unique_ptr<StagingBlockAllocator> allocator =
      StagingBlockAllocator::Create(&base, /*num_slots=*/1, /*max_blocks=*/1);
  const auto now = std::chrono::steady_clock::now();

  // Exhaust the only staging slot.
  absl::StatusOr<StagingAllocation> held_slot = allocator->Acquire(1);
  ABSL_ASSERT_OK(held_slot);
  ASSERT_EQ(allocator->num_free_slots(), 0);

  // Create a session with a 60-second deadline and block it in StartPush.
  std::shared_ptr<TransferSendSession> session = *TransferSendSession::Create(
      &base, allocator.get(), "req_blocked", /*uuid=*/104, /*block_ids=*/{0},
      /*deadline=*/now + std::chrono::seconds(60), /*register_start=*/now);
  session->ValidateAndBeginPull({0}, now);

  auto push_future = std::async(std::launch::async, [session]() {
    session->StartPush({"127.0.0.1:9000"}, /*src_block_ids=*/{0},
                       /*dst_block_ids=*/{0});
  });

  absl::SleepFor(absl::Milliseconds(20));
  EXPECT_FALSE(session->Done());

  // Cancelling the session must wake AcquireWithTimeout promptly without
  // waiting for the 60-second deadline and without poisoning the allocator.
  const absl::Time cancel_start = absl::Now();
  session->Finish(absl::CancelledError("peer aborted"));
  ASSERT_EQ(push_future.wait_for(std::chrono::seconds(2)),
            std::future_status::ready);
  EXPECT_LT(absl::Now() - cancel_start, absl::Seconds(1));

  EXPECT_TRUE(session->Done());
  EXPECT_FALSE(session->HasStaging());

  // Returning the held slot leaves the shared allocator healthy.
  held_slot->Reset();
  EXPECT_EQ(allocator->num_free_slots(), 1);
  absl::StatusOr<StagingAllocation> reacquired =
      allocator->AcquireWithTimeout(1, now + std::chrono::seconds(1));
  ABSL_EXPECT_OK(reacquired);
}

TEST(TransferSendSessionTest,
     FinishBeforeStartPushDoesNotAcquireStagingOrAccessHbm) {
  FakeSendBase base(/*num_layers=*/1, /*host_blocks=*/4);
  std::unique_ptr<StagingBlockAllocator> allocator =
      StagingBlockAllocator::Create(&base, /*num_slots=*/2, /*max_blocks=*/1);
  const auto now = std::chrono::steady_clock::now();

  std::shared_ptr<TransferSendSession> session = *TransferSendSession::Create(
      &base, allocator.get(), "req_race", /*uuid=*/103, /*block_ids=*/{0},
      /*deadline=*/now + std::chrono::seconds(5), /*register_start=*/now);
  session->ValidateAndBeginPull({0}, now);
  EXPECT_FALSE(session->Done());

  // Finish occurs after ValidateAndBeginPull but before StartPush executes on
  // the detached worker thread. AcquireStagingWithRetry checks draining_/done_
  // and must not assign staging_ or issue any D2H/H2H operations.
  session->Finish(absl::CancelledError("peer disconnected before StartPush"));
  EXPECT_TRUE(session->IsDraining());
  EXPECT_TRUE(session->Done());

  session->StartPush({"127.0.0.1:9000"}, /*src_block_ids=*/{0},
                     /*dst_block_ids=*/{0});
  EXPECT_TRUE(session->Done());
  EXPECT_FALSE(session->HasStaging());
  EXPECT_EQ(base.d2h_calls(), 0);
  EXPECT_EQ(base.h2h_calls(), 0);
  EXPECT_EQ(allocator->num_free_slots(), 2);
}

TEST(TransferSendSessionTest,
     DoneGuaranteesAllResourcesReleasedAndNoHbmOrTransportAccessAfterDone) {
  FakeSendBase base(/*num_layers=*/2, /*host_blocks=*/4);
  base.SetManualD2h(true);
  base.SetManualH2h(true);
  std::unique_ptr<StagingBlockAllocator> allocator =
      StagingBlockAllocator::Create(&base, /*num_slots=*/1, /*max_blocks=*/1);
  const auto now = std::chrono::steady_clock::now();

  std::shared_ptr<TransferSendSession> session = *TransferSendSession::Create(
      &base, allocator.get(), "req_drain", /*uuid=*/105, /*block_ids=*/{0},
      /*deadline=*/now + std::chrono::seconds(10), /*register_start=*/now);
  session->ValidateAndBeginPull({0}, now);
  session->StartPush({"127.0.0.1:9000"}, /*src_block_ids=*/{0},
                     /*dst_block_ids=*/{0});

  // Both D2H copies (layers 0 and 1) have been issued; slot is held.
  ASSERT_EQ(base.d2h_calls(), 2);
  EXPECT_TRUE(session->HasStaging());
  EXPECT_EQ(allocator->num_free_slots(), 0);
  EXPECT_FALSE(session->Done());

  // Complete Layer 0 D2H so Layer 0 H2H starts on push_pool.
  base.CompleteD2h(0, absl::OkStatus());
  ASSERT_TRUE(base.WaitForH2hCalls(1, absl::Seconds(2)));
  EXPECT_FALSE(session->Done());

  // Fail/timeout the session while Layer 0 H2H and Layer 1 D2H are in flight.
  session->Finish(absl::DeadlineExceededError("send deadline exceeded"));
  EXPECT_TRUE(session->IsDraining());
  EXPECT_FALSE(session->Done());
  EXPECT_TRUE(session->HasStaging());
  EXPECT_EQ(allocator->num_free_slots(), 0);

  // Layer 0 H2H finishes first: Done() MUST remain false and staging MUST
  // remain pinned because Layer 1 D2H is still accessing HBM & host staging.
  base.CompleteH2h(0, std::vector<int>{0});
  EXPECT_FALSE(session->Done());
  EXPECT_TRUE(session->HasStaging());
  EXPECT_EQ(allocator->num_free_slots(), 0);

  // Layer 1 D2H finishes: because the session is draining, Layer 1 H2H is NOT
  // started, staging is immediately released, and Done() becomes true.
  base.CompleteD2h(1, absl::OkStatus());
  EXPECT_THAT(session->AwaitForDone(),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_TRUE(session->Done());
  EXPECT_FALSE(session->HasStaging());
  EXPECT_EQ(allocator->num_free_slots(), 1);
  EXPECT_EQ(base.h2h_calls(), 1);

  // Any subsequent StartPush call after Done() is a strict no-op.
  session->StartPush({"127.0.0.1:9000"}, /*src_block_ids=*/{0},
                     /*dst_block_ids=*/{0});
  EXPECT_EQ(base.d2h_calls(), 2);
  EXPECT_EQ(base.h2h_calls(), 1);
  EXPECT_EQ(allocator->num_free_slots(), 1);
}

TEST(TransferSendSessionTest,
     AsyncCallbacksKeepSessionAliveAfterCallerDropsSharedPtr) {
  FakeSendBase base(/*num_layers=*/1, /*host_blocks=*/2);
  base.SetManualD2h(true);
  base.SetManualH2h(true);
  std::unique_ptr<StagingBlockAllocator> allocator =
      StagingBlockAllocator::Create(&base, /*num_slots=*/1, /*max_blocks=*/1);
  const auto now = std::chrono::steady_clock::now();

  std::shared_ptr<TransferSendSession> session = *TransferSendSession::Create(
      &base, allocator.get(), "req_lifetime", /*uuid=*/106, /*block_ids=*/{0},
      /*deadline=*/now + std::chrono::seconds(10), /*register_start=*/now);
  session->ValidateAndBeginPull({0}, now);
  session->StartPush({"127.0.0.1:9000"}, /*src_block_ids=*/{0},
                     /*dst_block_ids=*/{0});

  ASSERT_EQ(base.d2h_calls(), 1);
  EXPECT_EQ(allocator->num_free_slots(), 0);

  // Drop the caller's shared_ptr while async D2H and H2H callbacks are pending.
  std::weak_ptr<TransferSendSession> weak_session = session;
  session.reset();
  EXPECT_FALSE(weak_session.expired());

  // Complete D2H: H2H starts and continues holding the session alive.
  base.CompleteD2h(0, absl::OkStatus());
  ASSERT_TRUE(base.WaitForH2hCalls(1, absl::Seconds(2)));
  EXPECT_FALSE(weak_session.expired());
  EXPECT_EQ(allocator->num_free_slots(), 0);

  // Complete H2H: the session finishes, releases its staging slot, and then
  // destroys itself cleanly once the last callback unwinds.
  base.CompleteH2h(0, std::vector<int>{0});
  EXPECT_EQ(allocator->num_free_slots(), 1);
  EXPECT_TRUE(weak_session.expired());
}

TEST(TransferSendSessionTest,
     ZeroLayerSessionCompletesAndReleasesStagingWithoutHang) {
  FakeSendBase base(/*num_layers=*/0, /*host_blocks=*/4);
  std::unique_ptr<StagingBlockAllocator> allocator =
      StagingBlockAllocator::Create(&base, /*num_slots=*/2, /*max_blocks=*/1);
  const auto now = std::chrono::steady_clock::now();

  std::shared_ptr<TransferSendSession> session = *TransferSendSession::Create(
      &base, allocator.get(), "req_zero_layers", /*uuid=*/107,
      /*block_ids=*/{0}, /*deadline=*/now + std::chrono::seconds(5),
      /*register_start=*/now);
  session->ValidateAndBeginPull({0}, now);
  session->StartPush({"127.0.0.1:9000"}, /*src_block_ids=*/{0},
                     /*dst_block_ids=*/{0});

  ABSL_EXPECT_OK(session->AwaitForDone());
  EXPECT_TRUE(session->Done());
  EXPECT_FALSE(session->HasStaging());
  EXPECT_EQ(allocator->num_free_slots(), 2);
}

TEST(TransferSendSessionTest,
     InvalidStartPushArgumentsFailSessionImmediatelyWithoutHang) {
  FakeSendBase base(/*num_layers=*/1, /*host_blocks=*/4);
  std::unique_ptr<StagingBlockAllocator> allocator =
      StagingBlockAllocator::Create(&base, /*num_slots=*/2, /*max_blocks=*/1);
  const auto now = std::chrono::steady_clock::now();

  std::shared_ptr<TransferSendSession> session = *TransferSendSession::Create(
      &base, allocator.get(), "req_bad_args", /*uuid=*/108, /*block_ids=*/{0},
      /*deadline=*/now + std::chrono::seconds(5), /*register_start=*/now);
  session->ValidateAndBeginPull({0}, now);
  session->StartPush(/*remote_data_endpoints=*/{}, /*src_block_ids=*/{0},
                     /*dst_block_ids=*/{0});

  EXPECT_THAT(session->AwaitForDone(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_TRUE(session->Done());
  EXPECT_FALSE(session->HasStaging());
  EXPECT_EQ(allocator->num_free_slots(), 2);
}

TEST(TransferSendSessionTest, StatusIsFrozenOnceSessionIsDrainingOrDone) {
  FakeSendBase base(/*num_layers=*/1, /*host_blocks=*/4);
  base.SetManualD2h(true);
  std::unique_ptr<StagingBlockAllocator> allocator =
      StagingBlockAllocator::Create(&base, /*num_slots=*/2, /*max_blocks=*/1);
  const auto now = std::chrono::steady_clock::now();

  std::shared_ptr<TransferSendSession> session = *TransferSendSession::Create(
      &base, allocator.get(), "req_freeze", /*uuid=*/305, /*block_ids=*/{0},
      /*deadline=*/now + std::chrono::seconds(10), /*register_start=*/now);
  session->ValidateAndBeginPull({0}, now);
  session->StartPush({"127.0.0.1:9000"}, /*src_block_ids=*/{0},
                     /*dst_block_ids=*/{0});

  session->Finish(absl::OkStatus());
  EXPECT_TRUE(session->IsDraining());
  EXPECT_FALSE(session->Done());
  ABSL_EXPECT_OK(session->GetStatus());

  // Subsequent error while draining must not overwrite the frozen status_.
  session->Finish(absl::CancelledError("manager shutdown"));
  ABSL_EXPECT_OK(session->GetStatus());

  base.CompleteD2h(0, absl::OkStatus());
  ABSL_EXPECT_OK(session->AwaitForDone());
  EXPECT_TRUE(session->Done());

  // Subsequent error after done must not overwrite the frozen status_.
  session->Finish(absl::InternalError("late error"));
  ABSL_EXPECT_OK(session->GetStatus());
}

TEST(TransferSendSessionTest, ReleaseStagingBlocksClearsD2hLayerFuturesOnDone) {
  FakeSendBase base(/*num_layers=*/1, /*host_blocks=*/4);
  base.SetManualD2h(true);
  std::unique_ptr<StagingBlockAllocator> allocator =
      StagingBlockAllocator::Create(&base, /*num_slots=*/2, /*max_blocks=*/1);
  const auto now = std::chrono::steady_clock::now();

  std::shared_ptr<TransferSendSession> session = *TransferSendSession::Create(
      &base, allocator.get(), "req_futures", /*uuid=*/306, /*block_ids=*/{0},
      /*deadline=*/now + std::chrono::seconds(10), /*register_start=*/now);
  session->ValidateAndBeginPull({0}, now);
  session->StartPush({"127.0.0.1:9000"}, /*src_block_ids=*/{0},
                     /*dst_block_ids=*/{0});

  EXPECT_TRUE(
      session->OwnsBlockWithReadyFuture(/*block_id=*/0, /*layer_idx=*/0));
  EXPECT_TRUE(session->HasStaging());

  base.CompleteD2h(0, absl::OkStatus());
  ABSL_EXPECT_OK(session->AwaitForDone());
  EXPECT_TRUE(session->Done());
  EXPECT_FALSE(session->HasStaging());
  EXPECT_FALSE(
      session->OwnsBlockWithReadyFuture(/*block_id=*/0, /*layer_idx=*/0));
}

}  // namespace
}  // namespace tpu_raiden
