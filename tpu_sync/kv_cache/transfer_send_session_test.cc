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

#include "tpu_sync/kv_cache/transfer_send_session.h"

#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/future.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/staging_window.h"
#include "tpu_sync/kv_cache/transfer_session.h"
#include "tpu_sync/transport/block_transport.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::AllOf;
using ::testing::HasSubstr;

class FakeSendBase : public kv_cache::KVCacheManagerBase {
 public:
  explicit FakeSendBase(size_t num_layers = 1, int64_t host_blocks = 4,
                        size_t block_byte_size = 128)
      : kv_cache::KVCacheManagerBase(
            num_layers, /*num_shards=*/1,
            std::vector<size_t>(num_layers, block_byte_size),
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

  struct DispatchedD2h {
    std::vector<int64_t> src_offsets;
    std::vector<int64_t> dst_offsets;
    std::vector<int64_t> sizes;
    std::optional<size_t> layer_idx;
    std::thread::id thread_id;
  };

  struct DispatchedH2h {
    std::vector<int> src_block_ids;
    std::vector<int> dst_block_ids;
    int layer_idx;
    std::thread::id thread_id;
    std::optional<int> parallelism = std::nullopt;
  };

  void SetParallelism(int p) { parallelism_ = p; }

  void SetNumNumaGroups(int g) {
    absl::MutexLock lock(mu_);
    num_numa_groups_override_ = g;
  }

  int num_numa_groups() const override {
    absl::MutexLock lock(mu_);
    if (num_numa_groups_override_.has_value()) {
      return *num_numa_groups_override_;
    }
    return kv_cache::KVCacheManagerBase::num_numa_groups();
  }

  int d2h_calls() const {
    absl::MutexLock lock(mu_);
    return d2h_calls_;
  }

  int h2h_calls() const {
    absl::MutexLock lock(mu_);
    return h2h_calls_;
  }

  bool WaitForD2hCalls(int expected, absl::Duration timeout) const {
    absl::MutexLock lock(mu_);
    auto cond = [this, expected]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return d2h_calls_ >= expected;
    };
    return mu_.AwaitWithTimeout(absl::Condition(&cond), timeout);
  }

  bool WaitForH2hCalls(int expected, absl::Duration timeout) const {
    absl::MutexLock lock(mu_);
    auto cond = [this, expected]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return h2h_calls_ >= expected;
    };
    return mu_.AwaitWithTimeout(absl::Condition(&cond), timeout);
  }

  std::vector<DispatchedD2h> d2h_dispatched() const {
    absl::MutexLock lock(mu_);
    return d2h_dispatched_;
  }

  std::vector<DispatchedH2h> h2h_dispatched() const {
    absl::MutexLock lock(mu_);
    return h2h_dispatched_;
  }

  absl::StatusOr<raiden::PjRtCopyFuture> D2hSyncDispatch(
      const std::vector<int64_t>& src_offsets_major_dim,
      const std::vector<int64_t>& dst_offsets_major_dim,
      const std::vector<int64_t>& copy_sizes_major_dim,
      std::optional<int64_t> slot_idx, std::optional<size_t> layer_idx,
      std::optional<size_t> shard_idx) override {
    absl::MutexLock lock(mu_);
    ++d2h_calls_;
    d2h_dispatched_.push_back({src_offsets_major_dim, dst_offsets_major_dim,
                               copy_sizes_major_dim, layer_idx,
                               std::this_thread::get_id()});
    auto [promise, future] = xla::MakePromise<>();
    if (manual_d2h_) {
      d2h_promises_.push_back(std::move(promise));
    } else {
      promise.Set(absl::OkStatus());
    }
    return raiden::PjRtCopyFuture(std::move(future), raiden::BufferHolders{});
  }

  void H2hWriteDirectAsync(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids, uint64_t uuid, int layer_idx,
      std::function<void(absl::StatusOr<std::vector<int>>)> on_complete,
      std::optional<int> parallelism = std::nullopt) override {
    {
      absl::MutexLock lock(mu_);
      ++h2h_calls_;
      h2h_dispatched_.push_back({src_block_ids, dst_block_ids, layer_idx,
                                 std::this_thread::get_id(), parallelism});
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
  std::optional<int> num_numa_groups_override_ ABSL_GUARDED_BY(mu_) =
      std::nullopt;
  std::vector<xla::Promise<>> d2h_promises_ ABSL_GUARDED_BY(mu_);
  std::vector<std::function<void(absl::StatusOr<std::vector<int>>)>>
      h2h_callbacks_ ABSL_GUARDED_BY(mu_);
  std::vector<DispatchedD2h> d2h_dispatched_ ABSL_GUARDED_BY(mu_);
  std::vector<DispatchedH2h> h2h_dispatched_ ABSL_GUARDED_BY(mu_);
};

// Sets TPU_RAIDEN_DATA_CONN_REUSE for one test (unsets it when `value` is
// null) and restores it afterwards. Tests construct it before any manager
// base or session, so no other thread reads the environment meanwhile.
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

TEST(TransferSendSessionTest,
     PerTransferPolicyClosesIdlePeerConnectionsBeforeDone) {
  ScopedDataConnReuseEnv env("per_transfer");
  FakeSendBase base(/*num_layers=*/1, /*host_blocks=*/4);
  FakeSendBase receiver(/*num_layers=*/1, /*host_blocks=*/4);
  const std::string peer =
      absl::StrCat("127.0.0.1:", receiver.transport_server()->local_port());
  ABSL_ASSERT_OK(SeedIdleDataConnection(*base.transport_server(), peer));
  ASSERT_EQ(base.transport_server()->IdleConnectionCount(peer), 1u);
  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(&base, /*num_slots=*/2, /*max_blocks=*/2);
  const auto now = std::chrono::steady_clock::now();
  std::shared_ptr<TransferSendSession> session = *TransferSendSession::Create(
      &base, allocator.get(), "req_release", /*uuid=*/77, /*block_ids=*/{0},
      /*deadline=*/now + std::chrono::seconds(10), /*register_start=*/now);

  session->ValidateAndBeginPull({0}, now);
  session->StartPush({peer}, /*src_block_ids=*/{0}, /*dst_block_ids=*/{0});
  ABSL_ASSERT_OK(session->AwaitForDone());

  // The send's last op closes the idle connections to its peer before the
  // send is done, so nothing is left once AwaitForDone() returns.
  EXPECT_EQ(base.transport_server()->IdleConnectionCount(peer), 0u);
}

TEST(TransferSendSessionTest, PooledPolicyKeepsIdlePeerConnections) {
  ScopedDataConnReuseEnv env(/*value=*/nullptr);
  FakeSendBase base(/*num_layers=*/1, /*host_blocks=*/4);
  FakeSendBase receiver(/*num_layers=*/1, /*host_blocks=*/4);
  const std::string peer =
      absl::StrCat("127.0.0.1:", receiver.transport_server()->local_port());
  ABSL_ASSERT_OK(SeedIdleDataConnection(*base.transport_server(), peer));
  ASSERT_EQ(base.transport_server()->IdleConnectionCount(peer), 1u);
  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(&base, /*num_slots=*/2, /*max_blocks=*/2);
  const auto now = std::chrono::steady_clock::now();
  std::shared_ptr<TransferSendSession> session = *TransferSendSession::Create(
      &base, allocator.get(), "req_pooled", /*uuid=*/79, /*block_ids=*/{0},
      /*deadline=*/now + std::chrono::seconds(10), /*register_start=*/now);

  session->ValidateAndBeginPull({0}, now);
  session->StartPush({peer}, /*src_block_ids=*/{0}, /*dst_block_ids=*/{0});
  ABSL_ASSERT_OK(session->AwaitForDone());

  EXPECT_EQ(base.transport_server()->IdleConnectionCount(peer), 1u);
}

TEST(TransferSendSessionTest,
     PerTransferPolicyClosesIdlePeerConnectionsWhenCancelledMidPush) {
  ScopedDataConnReuseEnv env("per_transfer");
  FakeSendBase base(/*num_layers=*/1, /*host_blocks=*/4);
  base.SetManualH2h(true);
  FakeSendBase receiver(/*num_layers=*/1, /*host_blocks=*/4);
  const std::string peer =
      absl::StrCat("127.0.0.1:", receiver.transport_server()->local_port());
  ABSL_ASSERT_OK(SeedIdleDataConnection(*base.transport_server(), peer));
  ASSERT_EQ(base.transport_server()->IdleConnectionCount(peer), 1u);
  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(&base, /*num_slots=*/2, /*max_blocks=*/2);
  const auto now = std::chrono::steady_clock::now();
  std::shared_ptr<TransferSendSession> session = *TransferSendSession::Create(
      &base, allocator.get(), "req_cancel_release", /*uuid=*/80,
      /*block_ids=*/{0}, /*deadline=*/now + std::chrono::seconds(10),
      /*register_start=*/now);
  session->ValidateAndBeginPull({0}, now);
  session->StartPush({peer}, /*src_block_ids=*/{0}, /*dst_block_ids=*/{0});
  ASSERT_TRUE(base.WaitForH2hCalls(1, absl::Seconds(2)));

  // Cancelling the send while its push runs starts draining it, but the
  // push in flight keeps it from finishing, and nothing is released yet.
  session->Finish(absl::CancelledError("consumer cancelled"));
  EXPECT_FALSE(session->Done());
  EXPECT_EQ(base.transport_server()->IdleConnectionCount(peer), 1u);

  // Completing the push ends the send: its last op closes the idle
  // connections to the consumer before the send is done.
  base.CompleteH2h(0, std::vector<int>{0});
  EXPECT_THAT(session->AwaitForDone(), StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(base.transport_server()->IdleConnectionCount(peer), 0u);
}

TEST(TransferSendSessionTest,
     PooledPolicyKeepsIdlePeerConnectionsWhenCancelledMidPush) {
  ScopedDataConnReuseEnv env(/*value=*/nullptr);
  FakeSendBase base(/*num_layers=*/1, /*host_blocks=*/4);
  base.SetManualH2h(true);
  FakeSendBase receiver(/*num_layers=*/1, /*host_blocks=*/4);
  const std::string peer =
      absl::StrCat("127.0.0.1:", receiver.transport_server()->local_port());
  ABSL_ASSERT_OK(SeedIdleDataConnection(*base.transport_server(), peer));
  ASSERT_EQ(base.transport_server()->IdleConnectionCount(peer), 1u);
  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(&base, /*num_slots=*/2, /*max_blocks=*/2);
  const auto now = std::chrono::steady_clock::now();
  std::shared_ptr<TransferSendSession> session = *TransferSendSession::Create(
      &base, allocator.get(), "req_cancel_pooled", /*uuid=*/81,
      /*block_ids=*/{0}, /*deadline=*/now + std::chrono::seconds(10),
      /*register_start=*/now);
  session->ValidateAndBeginPull({0}, now);
  session->StartPush({peer}, /*src_block_ids=*/{0}, /*dst_block_ids=*/{0});
  ASSERT_TRUE(base.WaitForH2hCalls(1, absl::Seconds(2)));

  session->Finish(absl::CancelledError("consumer cancelled"));
  base.CompleteH2h(0, std::vector<int>{0});
  EXPECT_THAT(session->AwaitForDone(), StatusIs(absl::StatusCode::kCancelled));

  EXPECT_EQ(base.transport_server()->IdleConnectionCount(peer), 1u);
}

TEST(TransferSendSessionTest, CreateRejectsUnknownDataConnReuseValue) {
  ScopedDataConnReuseEnv env("per-transfer");
  FakeSendBase base(/*num_layers=*/1, /*host_blocks=*/4);
  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(&base, /*num_slots=*/2, /*max_blocks=*/2);
  const auto now = std::chrono::steady_clock::now();

  EXPECT_THAT(TransferSendSession::Create(
                  &base, allocator.get(), "req_bad_policy", /*uuid=*/78,
                  /*block_ids=*/{0},
                  /*deadline=*/now + std::chrono::seconds(10),
                  /*register_start=*/now),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("TPU_RAIDEN_DATA_CONN_REUSE"),
                             HasSubstr("per-transfer"))));
}

TEST(TransferSendSessionTest, SendSessionImplementsTransferSessionInterface) {
  FakeSendBase base(/*num_layers=*/1, /*host_blocks=*/4);
  base.SetManualD2h(true);
  std::unique_ptr<StagingBlockAllocator> allocator =
      CreateStagingBlockAllocator(&base, /*num_slots=*/2, /*max_blocks=*/2);
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
      CreateStagingBlockAllocator(&base, /*num_slots=*/2, /*max_blocks=*/1);
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
      CreateStagingBlockAllocator(&base, /*num_slots=*/1, /*max_blocks=*/1);
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
      CreateStagingBlockAllocator(&base, /*num_slots=*/2, /*max_blocks=*/1);
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
      CreateStagingBlockAllocator(&base, /*num_slots=*/1, /*max_blocks=*/1);
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
      CreateStagingBlockAllocator(&base, /*num_slots=*/1, /*max_blocks=*/1);
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
      CreateStagingBlockAllocator(&base, /*num_slots=*/2, /*max_blocks=*/1);
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
      CreateStagingBlockAllocator(&base, /*num_slots=*/2, /*max_blocks=*/1);
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
      CreateStagingBlockAllocator(&base, /*num_slots=*/2, /*max_blocks=*/1);
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
      CreateStagingBlockAllocator(&base, /*num_slots=*/2, /*max_blocks=*/1);
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

class TransferSendSessionPeer {
 public:
  static int32_t d2h_issued_units(const TransferSendSession& s, size_t l) {
    absl::MutexLock lock(s.mu_);
    return l < s.d2h_issued_units_.size() ? s.d2h_issued_units_[l] : 0;
  }
  static int32_t h2h_issued_units(const TransferSendSession& s, size_t l) {
    absl::MutexLock lock(s.mu_);
    return l < s.h2h_issued_units_.size() ? s.h2h_issued_units_[l] : 0;
  }
  static int32_t h2h_acked_units(const TransferSendSession& s, size_t l) {
    absl::MutexLock lock(s.mu_);
    return l < s.h2h_acked_units_.size() ? s.h2h_acked_units_[l] : 0;
  }
  static size_t remaining_h2h_units(const TransferSendSession& s) {
    return s.remaining_h2h_units_.load(std::memory_order_acquire);
  }
  static absl::string_view CurrentPhase(const TransferSendSession& s) {
    absl::MutexLock lock(s.mu_);
    return s.CurrentPhaseLocked();
  }
  static bool IsWindowed(const TransferSendSession& s) {
    absl::MutexLock lock(s.mu_);
    return s.IsWindowedLocked();
  }
  static int32_t window_chunks(const TransferSendSession& s) {
    absl::MutexLock lock(s.mu_);
    return s.window_chunks_;
  }
  static int32_t n_chunks(const TransferSendSession& s) {
    absl::MutexLock lock(s.mu_);
    return s.n_chunks_;
  }
};

namespace {

struct WindowedTestContext {
  std::unique_ptr<FakeSendBase> base;
  std::unique_ptr<StagingBlockAllocator> allocator;
  std::shared_ptr<TransferSendSession> session;
};

WindowedTestContext MakeWindowedSession(int layers, int chunks, int window,
                                        uint64_t uuid = 400,
                                        bool manual_d2h = true) {
  auto base =
      std::make_unique<FakeSendBase>(layers, /*host_blocks=*/chunks * 4);
  base->SetManualD2h(manual_d2h);
  base->SetManualH2h(true);
  auto allocator = CreateStagingBlockAllocator(base.get(), /*num_slots=*/2,
                                               /*max_blocks=*/chunks * 2);
  const auto now = std::chrono::steady_clock::now();
  std::vector<int64_t> blocks(chunks * 2);
  std::iota(blocks.begin(), blocks.end(), 0);
  auto session =
      *TransferSendSession::Create(base.get(), allocator.get(), "req", uuid,
                                   blocks, now + std::chrono::seconds(10), now);
  std::vector<std::vector<CopySpec>> specs(chunks,
                                           std::vector<CopySpec>(layers));
  std::vector<std::vector<int32_t>> src_ids(chunks), dst_ids(chunks);
  for (int c = 0; c < chunks; ++c) {
    src_ids[c] = {c * 2, c * 2 + 1};
    dst_ids[c] = {100 + c * 2, 100 + c * 2 + 1};
    for (int l = 0; l < layers; ++l) {
      specs[c][l].src_offsets = {c * 2, c * 2 + 1};
      specs[c][l].dst_offsets = {c * 2, c * 2 + 1};
      specs[c][l].sizes = {1, 1};
    }
  }
  ABSL_CHECK_OK(
      session->ConfigureWindowedSend(window, specs, src_ids, dst_ids));
  return WindowedTestContext{std::move(base), std::move(allocator),
                             std::move(session)};
}

TEST(TransferSendSessionTest, WindowedSendPerLayerUnitOrderingAndD2hLookahead) {
  auto ctx = MakeWindowedSession(/*layers=*/2, /*chunks=*/4, /*window=*/2, 401);
  const auto now = std::chrono::steady_clock::now();
  std::vector<int64_t> src = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int64_t> dst = {100, 101, 102, 103, 104, 105, 106, 107};
  ctx.session->ValidateAndBeginPull(src, now);
  ctx.session->StartPush({"127.0.0.1:9000"}, src, dst);

  EXPECT_EQ(ctx.base->d2h_calls(), 4);
  EXPECT_EQ(ctx.base->h2h_calls(), 0);
  EXPECT_EQ(TransferSendSessionPeer::remaining_h2h_units(*ctx.session), 8);
  EXPECT_EQ(TransferSendSessionPeer::d2h_issued_units(*ctx.session, 0), 2);
  EXPECT_EQ(TransferSendSessionPeer::window_chunks(*ctx.session), 2);
  EXPECT_EQ(TransferSendSessionPeer::n_chunks(*ctx.session), 4);
  EXPECT_TRUE(TransferSendSessionPeer::IsWindowed(*ctx.session));

  ctx.base->CompleteD2h(0, absl::OkStatus());
  ASSERT_TRUE(ctx.base->WaitForH2hCalls(1, absl::Seconds(2)));
  EXPECT_EQ(TransferSendSessionPeer::h2h_issued_units(*ctx.session, 0), 1);

  ctx.base->CompleteD2h(1, absl::OkStatus());
  absl::SleepFor(absl::Milliseconds(20));
  EXPECT_EQ(ctx.base->h2h_calls(), 1);
  EXPECT_EQ(ctx.base->d2h_calls(), 4);

  ctx.base->CompleteH2h(0, std::vector<int>{0, 1});
  ASSERT_TRUE(ctx.base->WaitForD2hCalls(5, absl::Seconds(2)));
  ASSERT_TRUE(ctx.base->WaitForH2hCalls(2, absl::Seconds(2)));
  EXPECT_EQ(TransferSendSessionPeer::h2h_acked_units(*ctx.session, 0), 1);
  EXPECT_EQ(TransferSendSessionPeer::h2h_issued_units(*ctx.session, 0), 2);

  ctx.base->CompleteD2h(2, absl::OkStatus());
  ASSERT_TRUE(ctx.base->WaitForH2hCalls(3, absl::Seconds(2)));
  ctx.base->CompleteD2h(3, absl::OkStatus());
  ctx.base->CompleteH2h(2, std::vector<int>{0, 1});
  ASSERT_TRUE(ctx.base->WaitForD2hCalls(6, absl::Seconds(2)));
  ASSERT_TRUE(ctx.base->WaitForH2hCalls(4, absl::Seconds(2)));

  ctx.base->CompleteD2h(4, absl::OkStatus());
  ctx.base->CompleteH2h(1, std::vector<int>{2, 3});
  ASSERT_TRUE(ctx.base->WaitForD2hCalls(7, absl::Seconds(2)));
  ASSERT_TRUE(ctx.base->WaitForH2hCalls(5, absl::Seconds(2)));

  ctx.base->CompleteD2h(5, absl::OkStatus());
  ctx.base->CompleteH2h(3, std::vector<int>{2, 3});
  ASSERT_TRUE(ctx.base->WaitForD2hCalls(8, absl::Seconds(2)));
  ASSERT_TRUE(ctx.base->WaitForH2hCalls(6, absl::Seconds(2)));

  ctx.base->CompleteD2h(6, absl::OkStatus());
  ctx.base->CompleteH2h(4, std::vector<int>{4, 5});
  ASSERT_TRUE(ctx.base->WaitForH2hCalls(7, absl::Seconds(2)));

  ctx.base->CompleteD2h(7, absl::OkStatus());
  ctx.base->CompleteH2h(5, std::vector<int>{4, 5});
  ASSERT_TRUE(ctx.base->WaitForH2hCalls(8, absl::Seconds(2)));

  ctx.base->CompleteH2h(6, std::vector<int>{6, 7});
  ctx.base->CompleteH2h(7, std::vector<int>{6, 7});

  ABSL_EXPECT_OK(ctx.session->AwaitForDone());
  EXPECT_TRUE(ctx.session->Done());

  const auto d2h_disp = ctx.base->d2h_dispatched();
  const auto h2h_disp = ctx.base->h2h_dispatched();
  ASSERT_EQ(d2h_disp.size(), 8);
  ASSERT_EQ(h2h_disp.size(), 8);
  for (const auto& d : d2h_disp) {
    EXPECT_EQ(d.sizes, (std::vector<int64_t>{1, 1}));
  }
  for (const auto& h : h2h_disp) {
    EXPECT_EQ(h.src_block_ids.size(), 2);
    EXPECT_EQ(h.dst_block_ids.size(), 2);
  }
}

TEST(TransferSendSessionTest, WindowedSendSynchronousD2hNoDoubleDispatch) {
  auto ctx = MakeWindowedSession(/*layers=*/2, /*chunks=*/4, /*window=*/2, 405,
                                 /*manual_d2h=*/false);
  const auto now = std::chrono::steady_clock::now();
  std::vector<int64_t> src(8), dst(8);
  std::iota(src.begin(), src.end(), 0);
  std::iota(dst.begin(), dst.end(), 100);
  ctx.session->ValidateAndBeginPull(src, now);
  ctx.session->StartPush({"127.0.0.1:9000"}, src, dst);
  for (int i = 0; i < 8; ++i) {
    ASSERT_TRUE(ctx.base->WaitForH2hCalls(i + 1, absl::Seconds(2)));
    ctx.base->CompleteH2h(i, std::vector<int>{0});
  }
  ABSL_EXPECT_OK(ctx.session->AwaitForDone());
  EXPECT_EQ(ctx.base->d2h_calls(), 8);
  EXPECT_EQ(ctx.base->h2h_calls(), 8);
}

TEST(TransferSendSessionTest, WindowedSendPushPoolThreadHop) {
  const std::thread::id caller_thread_id = std::this_thread::get_id();
  auto ctx = MakeWindowedSession(/*layers=*/1, /*chunks=*/2, /*window=*/1, 402);
  const auto now = std::chrono::steady_clock::now();
  ctx.session->ValidateAndBeginPull({0, 1}, now);
  ctx.session->StartPush({"127.0.0.1:9000"}, {0, 1}, {100, 101});

  ctx.base->CompleteD2h(0, absl::OkStatus());
  ASSERT_TRUE(ctx.base->WaitForH2hCalls(1, absl::Seconds(2)));
  EXPECT_NE(ctx.base->h2h_dispatched()[0].thread_id, caller_thread_id);

  ctx.base->CompleteH2h(0, std::vector<int>{0});
  ASSERT_TRUE(ctx.base->WaitForD2hCalls(2, absl::Seconds(2)));
  EXPECT_NE(ctx.base->d2h_dispatched()[1].thread_id, caller_thread_id);

  ctx.base->CompleteD2h(1, absl::OkStatus());
  ASSERT_TRUE(ctx.base->WaitForH2hCalls(2, absl::Seconds(2)));
  EXPECT_NE(ctx.base->h2h_dispatched()[1].thread_id, caller_thread_id);

  ctx.base->CompleteH2h(1, std::vector<int>{1});
  ABSL_EXPECT_OK(ctx.session->AwaitForDone());
}

TEST(TransferSendSessionTest,
     WindowedSendFailureMidWindowDrainsAndAckWaitPhase) {
  for (bool fail_d2h : {false, true}) {
    auto ctx =
        MakeWindowedSession(/*layers=*/2, /*chunks=*/2, /*window=*/2, 403);
    const auto now = std::chrono::steady_clock::now();
    ctx.session->ValidateAndBeginPull({0, 1, 2, 3}, now);
    ctx.session->StartPush({"127.0.0.1:9000"}, {0, 1, 2, 3},
                           {100, 101, 102, 103});
    ASSERT_EQ(ctx.base->d2h_calls(), 4);

    ctx.base->CompleteD2h(0, absl::OkStatus());
    ASSERT_TRUE(ctx.base->WaitForH2hCalls(1, absl::Seconds(2)));

    if (!fail_d2h) {
      ctx.base->CompleteD2h(1, absl::OkStatus());
      ctx.base->CompleteD2h(2, absl::OkStatus());
      ctx.base->CompleteD2h(3, absl::OkStatus());
      ASSERT_TRUE(ctx.base->WaitForH2hCalls(2, absl::Seconds(2)));
      EXPECT_EQ(TransferSendSessionPeer::CurrentPhase(*ctx.session),
                TransferSendSession::kAckWaitPhase);
      ctx.base->CompleteH2h(0, absl::UnavailableError("peer socket reset"));
      EXPECT_TRUE(ctx.session->IsDraining());
      EXPECT_FALSE(ctx.session->Done());
      EXPECT_TRUE(ctx.session->HasStaging());
      ctx.base->CompleteH2h(1, std::vector<int>{0});
      EXPECT_THAT(ctx.session->AwaitForDone(),
                  StatusIs(absl::StatusCode::kUnavailable,
                           HasSubstr("peer socket reset")));
      EXPECT_EQ(ctx.session->Details().phase, "push");
    } else {
      ctx.base->CompleteD2h(1, absl::InternalError("D2H hardware error"));
      EXPECT_TRUE(ctx.session->IsDraining());
      EXPECT_FALSE(ctx.session->Done());
      EXPECT_TRUE(ctx.session->HasStaging());
      ctx.base->CompleteD2h(2, absl::OkStatus());
      ctx.base->CompleteD2h(3, absl::OkStatus());
      ctx.base->CompleteH2h(0, std::vector<int>{0});
      EXPECT_THAT(ctx.session->AwaitForDone(),
                  StatusIs(absl::StatusCode::kInternal,
                           HasSubstr("D2H hardware error")));
      EXPECT_EQ(ctx.session->Details().phase, "d2h");
    }
    EXPECT_TRUE(ctx.session->Done());
    EXPECT_FALSE(ctx.session->HasStaging());
  }
}

TEST(TransferSendSessionTest, WindowedSendValidationAndPullRefusal) {
  FakeSendBase base(/*num_layers=*/1, /*host_blocks=*/4);
  auto allocator = CreateStagingBlockAllocator(&base, 1, 2);
  const auto now = std::chrono::steady_clock::now();
  auto session =
      *TransferSendSession::Create(&base, allocator.get(), "req_val", 404,
                                   {0, 1}, now + std::chrono::seconds(10), now);

  std::vector<std::vector<CopySpec>> specs(2, std::vector<CopySpec>(1));
  std::vector<std::vector<int32_t>> src = {{0}, {1}}, dst = {{10}, {11}};

  EXPECT_THAT(session->ConfigureWindowedSend(0, specs, src, dst),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(session->ConfigureWindowedSend(1, {}, src, dst),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(session->ConfigureWindowedSend(1, specs, {{0}}, dst),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(session->ConfigureWindowedSend(1, specs, src, {{10}}),
              StatusIs(absl::StatusCode::kInvalidArgument));

  ABSL_ASSERT_OK(session->ConfigureWindowedSend(1, specs, src, dst));
  EXPECT_FALSE(session->OwnsBlockWithReadyFuture(0, 0));

  absl::Status cb_status;
  session->RegisterLayerReadinessCallback(0, [&](auto s) { cb_status = s; });
  EXPECT_THAT(cb_status, StatusIs(absl::StatusCode::kFailedPrecondition));

  session->ValidateAndBeginPull({0, 1}, now);
  EXPECT_THAT(session->ConfigureWindowedSend(1, specs, src, dst),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(TransferSendSessionTest, WindowedSendComputesAndPassesPerUnitStreamCount) {
  const int layers = 1;
  const int chunks = 2;
  const int window = 2;
  const uint64_t uuid = 501;

  auto base = std::make_unique<FakeSendBase>(layers, /*host_blocks=*/16);
  base->SetParallelism(8);
  base->SetManualD2h(false);
  base->SetManualH2h(true);

  auto allocator = CreateStagingBlockAllocator(base.get(), /*num_slots=*/2,
                                               /*max_blocks=*/8);
  const auto now = std::chrono::steady_clock::now();
  std::vector<int64_t> blocks = {0, 1, 2, 3, 4, 5};
  auto session =
      *TransferSendSession::Create(base.get(), allocator.get(), "req_nu", uuid,
                                   blocks, now + std::chrono::seconds(10), now);

  std::vector<std::vector<CopySpec>> specs(chunks,
                                           std::vector<CopySpec>(layers));
  std::vector<std::vector<int32_t>> src_ids = {{0, 1, 2, 3}, {4, 5}};
  std::vector<std::vector<int32_t>> dst_ids = {{100, 101, 102, 103},
                                               {104, 105}};

  for (int c = 0; c < chunks; ++c) {
    for (int l = 0; l < layers; ++l) {
      specs[c][l].src_offsets = {src_ids[c].begin(), src_ids[c].end()};
      specs[c][l].dst_offsets = {dst_ids[c].begin(), dst_ids[c].end()};
      specs[c][l].sizes = std::vector<int64_t>(src_ids[c].size(), 1);
    }
  }

  // 2 MiB per block, 4 MiB stream bytes target, P = 8.
  // Chunk 0: 4 blocks * 2 MiB = 8 MiB / 4 MiB -> n_u = 2.
  // Strictly satisfies non-degenerate case: 1 < n_u (2) < min(P=8, C_0=4) = 4.
  // Chunk 1: 2 blocks * 2 MiB = 4 MiB / 4 MiB -> n_u = 1.
  // Strictly satisfies: 1 <= n_u (1) < min(P=8, C_1=2) = 2.
  const int64_t block_bytes = 2 << 20;
  const int64_t stream_bytes_target = 4 << 20;
  ABSL_CHECK_OK(session->ConfigureWindowedSend(
      window, specs, src_ids, dst_ids, block_bytes, stream_bytes_target));

  session->StartPush({"127.0.0.1:1234"}, {0, 1, 2, 3, 4, 5},
                     {100, 101, 102, 103, 104, 105});

  ASSERT_TRUE(base->WaitForH2hCalls(1, absl::Seconds(2)));
  auto h2h = base->h2h_dispatched();
  ASSERT_EQ(h2h.size(), 1);
  EXPECT_EQ(h2h[0].src_block_ids, (std::vector<int>{0, 1, 2, 3}));
  EXPECT_EQ(h2h[0].parallelism, 2);

  base->CompleteH2h(0, std::vector<int>{0, 1, 2, 3});

  ASSERT_TRUE(base->WaitForH2hCalls(2, absl::Seconds(2)));
  h2h = base->h2h_dispatched();
  ASSERT_EQ(h2h.size(), 2);
  EXPECT_EQ(h2h[1].src_block_ids, (std::vector<int>{4, 5}));
  EXPECT_EQ(h2h[1].parallelism, 1);

  base->CompleteH2h(1, std::vector<int>{4, 5});
  ABSL_EXPECT_OK(session->AwaitForDone());
}

TEST(TransferSendSessionTest, WindowedSendDerivesBlockBytesFromBaseWhenZero) {
  const int layers = 1;
  const int chunks = 1;
  const int window = 1;
  const uint64_t uuid = 504;
  const size_t block_size = 2 << 20;  // 2 MiB per block in base_

  auto base =
      std::make_unique<FakeSendBase>(layers, /*host_blocks=*/16, block_size);
  base->SetParallelism(8);
  base->SetManualD2h(false);
  base->SetManualH2h(true);

  auto allocator = CreateStagingBlockAllocator(base.get(), /*num_slots=*/2,
                                               /*max_blocks=*/8);
  const auto now = std::chrono::steady_clock::now();
  std::vector<int64_t> blocks = {0, 1, 2, 3};
  auto session = *TransferSendSession::Create(
      base.get(), allocator.get(), "req_base_bytes", uuid, blocks,
      now + std::chrono::seconds(10), now);

  std::vector<std::vector<CopySpec>> specs(chunks,
                                           std::vector<CopySpec>(layers));
  std::vector<std::vector<int32_t>> src_ids = {{0, 1, 2, 3}};
  std::vector<std::vector<int32_t>> dst_ids = {{100, 101, 102, 103}};

  specs[0][0].src_offsets = {src_ids[0].begin(), src_ids[0].end()};
  specs[0][0].dst_offsets = {dst_ids[0].begin(), dst_ids[0].end()};
  specs[0][0].sizes = std::vector<int64_t>(src_ids[0].size(), 1);

  // block_bytes_per_layer_per_group = 0 -> derived from base_: 2 MiB * 1 shard
  // / 1 group = 2 MiB. 4 blocks * 2 MiB = 8 MiB / 4 MiB stream_bytes_target ->
  // n_u = 2.
  const int64_t stream_bytes_target = 4 << 20;
  ABSL_CHECK_OK(session->ConfigureWindowedSend(
      window, specs, src_ids, dst_ids, /*block_bytes_per_layer_per_group=*/0,
      stream_bytes_target));

  session->StartPush({"127.0.0.1:1234"}, {0, 1, 2, 3}, {100, 101, 102, 103});

  ASSERT_TRUE(base->WaitForH2hCalls(1, absl::Seconds(2)));
  auto h2h = base->h2h_dispatched();
  ASSERT_EQ(h2h.size(), 1);
  EXPECT_EQ(h2h[0].src_block_ids, (std::vector<int>{0, 1, 2, 3}));
  EXPECT_EQ(h2h[0].parallelism, 2);

  base->CompleteH2h(0, std::vector<int>{0, 1, 2, 3});
  ABSL_EXPECT_OK(session->AwaitForDone());
}

TEST(TransferSendSessionTest, NonWindowedSendPassesFullParallelism) {
  const int layers = 1;
  const uint64_t uuid = 502;

  auto base = std::make_unique<FakeSendBase>(layers, /*host_blocks=*/16);
  base->SetParallelism(4);
  base->SetManualD2h(false);
  base->SetManualH2h(true);

  auto allocator = CreateStagingBlockAllocator(base.get(), /*num_slots=*/2,
                                               /*max_blocks=*/4);
  const auto now = std::chrono::steady_clock::now();
  std::vector<int64_t> blocks = {0, 1};
  auto session = *TransferSendSession::Create(
      base.get(), allocator.get(), "req_non_win", uuid, blocks,
      now + std::chrono::seconds(10), now);

  EXPECT_FALSE(session->IsWindowed());

  session->StartPush({"127.0.0.1:1234"}, {0, 1}, {100, 101});

  ASSERT_TRUE(base->WaitForH2hCalls(1, absl::Seconds(2)));
  auto h2h = base->h2h_dispatched();
  ASSERT_EQ(h2h.size(), 1);
  EXPECT_EQ(h2h[0].parallelism, 4);

  base->CompleteH2h(0, std::vector<int>{0, 1});
  ABSL_EXPECT_OK(session->AwaitForDone());
}

TEST(TransferSendSessionTest,
     WindowedSendEmitsParallelismAndStreamSlotWarnings) {
  const int layers = 1;
  const int chunks = 1;
  const int window = 1;
  const uint64_t uuid = 503;

  auto base = std::make_unique<FakeSendBase>(layers, /*host_blocks=*/16);
  // P = 4 < 16, G = 2 -> P * G = 8. L = 1, n_u = 1 -> L * n_u = 1 < 8.
  base->SetParallelism(4);
  base->SetNumNumaGroups(2);
  base->SetManualD2h(false);
  base->SetManualH2h(false);

  auto allocator = CreateStagingBlockAllocator(base.get(), /*num_slots=*/2,
                                               /*max_blocks=*/4);
  const auto now = std::chrono::steady_clock::now();
  std::vector<int64_t> blocks = {0};
  auto session = *TransferSendSession::Create(
      base.get(), allocator.get(), "req_warn", uuid, blocks,
      now + std::chrono::seconds(10), now);

  std::vector<std::vector<CopySpec>> specs(chunks,
                                           std::vector<CopySpec>(layers));
  specs[0][0].src_offsets = {0};
  specs[0][0].dst_offsets = {100};
  specs[0][0].sizes = {1};

  ABSL_CHECK_OK(session->ConfigureWindowedSend(window, specs, {{0}}, {{100}}));

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(
      log, Log(absl::LogSeverity::kWarning, _,
               HasSubstr("Windowed send running with parallelism = 4 < 16")));
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _,
                       HasSubstr("cannot saturate all peer stream slots")));
  log.StartCapturingLogs();

  session->StartPush({"127.0.0.1:1234"}, {0}, {100});
  ABSL_EXPECT_OK(session->AwaitForDone());
}

TEST(TransferSendSessionTest,
     WindowedSendEmitsNoWarningsWhenParallelismAndSlotsSaturated) {
  // P = 16 (no underutilization warning), G = 1 -> peer_slots = 16.
  // L = 16 layers, 1 block chunk with n_u = 1 -> total_active_streams = 16 * 1
  // = 16 >= peer_slots (16). Neither warning should be emitted.
  const int layers = 16;
  const int chunks = 1;
  const int window = 1;
  const uint64_t uuid = 505;

  auto base = std::make_unique<FakeSendBase>(layers, /*host_blocks=*/16);
  base->SetParallelism(16);
  base->SetNumNumaGroups(1);
  base->SetManualD2h(false);
  base->SetManualH2h(false);

  auto allocator = CreateStagingBlockAllocator(base.get(), /*num_slots=*/2,
                                               /*max_blocks=*/4);
  const auto now = std::chrono::steady_clock::now();
  std::vector<int64_t> blocks = {0};
  auto session = *TransferSendSession::Create(
      base.get(), allocator.get(), "req_no_warn", uuid, blocks,
      now + std::chrono::seconds(10), now);

  std::vector<std::vector<CopySpec>> specs(chunks,
                                           std::vector<CopySpec>(layers));
  for (int l = 0; l < layers; ++l) {
    specs[0][l].src_offsets = {0};
    specs[0][l].dst_offsets = {100};
    specs[0][l].sizes = {1};
  }

  ABSL_CHECK_OK(session->ConfigureWindowedSend(window, specs, {{0}}, {{100}}));

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _,
                       HasSubstr("Windowed send running with parallelism")))
      .Times(0);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _,
                       HasSubstr("cannot saturate all peer stream slots")))
      .Times(0);
  log.StartCapturingLogs();

  session->StartPush({"127.0.0.1:1234"}, {0}, {100});
  ABSL_EXPECT_OK(session->AwaitForDone());
}

TEST(TransferSendSessionTest, WindowedSendPrefillRealWRingReuseOrder) {
  const int layers = 1;
  const uint64_t uuid = 601;
  constexpr int64_t k = 7;
  constexpr int32_t chunk_blocks = 2;
  constexpr int32_t window_chunks = 2;

  auto base = std::make_unique<FakeSendBase>(layers, /*host_blocks=*/4);
  base->SetParallelism(4);
  base->SetManualD2h(true);
  base->SetManualH2h(true);

  // Staging pool of only W * C = 4 blocks (1 slot with max_blocks = 4).
  auto allocator = CreateStagingBlockAllocator(base.get(), /*num_slots=*/1,
                                               /*max_blocks=*/4);
  const auto now = std::chrono::steady_clock::now();
  std::vector<int64_t> src_block_ids(k);
  std::iota(src_block_ids.begin(), src_block_ids.end(), 0);
  std::vector<int64_t> dst_block_ids(k);
  std::iota(dst_block_ids.begin(), dst_block_ids.end(), 100);
  auto session = *TransferSendSession::Create(
      base.get(), allocator.get(), "req_real_w_reuse", uuid, src_block_ids,
      now + std::chrono::seconds(10), now);

  StagingWindowConfig win_cfg{
      .window_chunks = window_chunks,
      .serve_windowed_pulls = true,
      .chunk_blocks = chunk_blocks,
  };

  session->StartPush({"127.0.0.1:1234"}, src_block_ids, dst_block_ids, win_cfg);

  // Initially: Chunks 0 and 1 issue D2H immediately into slots 0 and 1.
  // Chunk 2 D2H is blocked by the window (c < h2h_acked + W => 2 < 0 + 2 is
  // false).
  ASSERT_TRUE(base->WaitForD2hCalls(2, absl::Seconds(2)));
  EXPECT_EQ(base->d2h_calls(), 2);
  EXPECT_EQ(base->h2h_calls(), 0);

  auto d2h = base->d2h_dispatched();
  ASSERT_EQ(d2h.size(), 2);
  // Chunk 0 targets ring blocks 0, 1 (coalesced: offset 0, size 2)
  EXPECT_EQ(d2h[0].dst_offsets, (std::vector<int64_t>{0}));
  EXPECT_EQ(d2h[0].sizes, (std::vector<int64_t>{2}));
  // Chunk 1 targets ring blocks 2, 3 (coalesced: offset 2, size 2)
  EXPECT_EQ(d2h[1].dst_offsets, (std::vector<int64_t>{2}));
  EXPECT_EQ(d2h[1].sizes, (std::vector<int64_t>{2}));

  // Complete chunk 0 D2H -> chunk 0 H2H is dispatched.
  base->CompleteD2h(0, absl::OkStatus());
  ASSERT_TRUE(base->WaitForH2hCalls(1, absl::Seconds(2)));
  EXPECT_EQ(base->h2h_calls(), 1);
  EXPECT_EQ(base->d2h_calls(), 2);  // Chunk 2 still not issued.

  // Chunk 0 H2H completes (push acked).
  base->CompleteH2h(0, std::vector<int>{100, 101});

  // Now that chunk 0 push is acked, chunk 2 D2H issues into reused slot 0
  // (blocks 0, 1).
  ASSERT_TRUE(base->WaitForD2hCalls(3, absl::Seconds(2)));
  EXPECT_EQ(base->d2h_calls(), 3);
  d2h = base->d2h_dispatched();
  ASSERT_EQ(d2h.size(), 3);
  EXPECT_EQ(d2h[2].dst_offsets, (std::vector<int64_t>{0}));
  EXPECT_EQ(d2h[2].sizes, (std::vector<int64_t>{2}));
  // Chunk 3 D2H still not issued (3 < 1 + 2 is false).
  EXPECT_EQ(base->h2h_calls(), 1);

  // Complete chunk 1 D2H -> chunk 1 H2H is dispatched.
  base->CompleteD2h(1, absl::OkStatus());
  ASSERT_TRUE(base->WaitForH2hCalls(2, absl::Seconds(2)));
  EXPECT_EQ(base->h2h_calls(), 2);
  EXPECT_EQ(base->d2h_calls(), 3);  // Chunk 3 still not issued.

  // Chunk 1 H2H completes (push acked).
  base->CompleteH2h(1, std::vector<int>{102, 103});

  // Now that chunk 1 push is acked, chunk 3 D2H issues into reused slot 1
  // (block 2).
  ASSERT_TRUE(base->WaitForD2hCalls(4, absl::Seconds(2)));
  EXPECT_EQ(base->d2h_calls(), 4);
  d2h = base->d2h_dispatched();
  ASSERT_EQ(d2h.size(), 4);
  EXPECT_EQ(d2h[3].dst_offsets, (std::vector<int64_t>{2}));
  EXPECT_EQ(d2h[3].sizes, (std::vector<int64_t>{1}));

  // Drain remaining chunks: chunk 2 and chunk 3.
  base->CompleteD2h(2, absl::OkStatus());
  ASSERT_TRUE(base->WaitForH2hCalls(3, absl::Seconds(2)));
  base->CompleteH2h(2, std::vector<int>{104, 105});

  base->CompleteD2h(3, absl::OkStatus());
  ASSERT_TRUE(base->WaitForH2hCalls(4, absl::Seconds(2)));
  base->CompleteH2h(3, std::vector<int>{106});

  ABSL_EXPECT_OK(session->AwaitForDone());
  EXPECT_TRUE(session->Done());
  EXPECT_FALSE(session->HasStaging());
  EXPECT_EQ(allocator->num_free_slots(), 1);
}

}  // namespace
}  // namespace tpu_raiden
