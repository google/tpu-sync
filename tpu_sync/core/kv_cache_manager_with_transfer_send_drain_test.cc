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

// A pull-serve send that expires or fails while its device-to-host copies
// are still running, exercised without a device: the copies complete when
// the test says so.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/future.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/transfer_receive_session.h"
#include "tpu_sync/core/transfer_send_session.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden {
namespace {

using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

constexpr int64_t kSlots = 2;
constexpr double kTimeoutS = 0.05;

// A producer whose device-to-host copies complete when the test says so.
class TestManager : public KVCacheManagerWithTransfer {
 public:
  explicit TestManager(size_t num_layers, double timeout_s = kTimeoutS)
      : KVCacheManagerWithTransfer(std::make_unique<TestBase>(num_layers, this),
                                   /*node_id=*/0,
                                   /*local_control_port=*/-1, /*max_blocks=*/1,
                                   /*num_slots=*/kSlots, timeout_s) {}

  // Serves a pull for `uuid` the way ProcessPullStream does once the
  // consumer is acknowledged: the push runs on this thread and returns
  // with its copies issued.
  void ServePull(uint64_t uuid) {
    std::shared_ptr<TransferSendSession> session;
    {
      absl::MutexLock lock(mu_);
      session = send_sessions_.at(uuid);
    }
    session->StartPush({"127.0.0.1:1"}, /*src_block_ids=*/{0},
                       /*dst_block_ids=*/{0});
  }

  size_t copies_issued() {
    absl::MutexLock lock(copies_mu_);
    return copies_.size();
  }

  // Completes the `index`-th copy issued.
  void FinishCopy(size_t index, absl::Status status) {
    absl::MutexLock lock(copies_mu_);
    copies_.at(index).Set(std::move(status));
  }

  size_t free_slots() { return staging_allocator_->num_free_slots(); }
  StagingBlockAllocator* staging_allocator() {
    return staging_allocator_.get();
  }

  // White-box construction for the send retirement state machine. Tests that
  // need registration and dispatch use NotifyForRead and ServePull instead.
  std::shared_ptr<TransferSendSession> AddSyntheticSend(
      const std::string& req_id, uint64_t uuid, int in_flight) {
    CHECK_GT(NotifyForRead(req_id, uuid, {0}), 0);
    std::shared_ptr<TransferSendSession> session;
    {
      absl::MutexLock lock(mu_);
      session = send_sessions_.at(uuid);
    }
    if (in_flight > 0) {
      ServePull(uuid);
    }
    return session;
  }

  void Decide(const std::shared_ptr<TransferSendSession>& session,
              bool failed) {
    session->Finish(failed ? absl::InternalError("send failed")
                           : absl::OkStatus());
  }

  void End(const std::shared_ptr<TransferSendSession>& session) {
    session->EndSendOp();
  }

  bool has_send(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    auto it = send_sessions_.find(uuid);
    return it != send_sessions_.end() && !it->second->Done();
  }

 private:
  class TestBase : public kv_cache::KVCacheManagerBase {
   public:
    TestBase(size_t num_layers, TestManager* owner)
        : kv_cache::KVCacheManagerBase(num_layers, /*num_shards=*/1,
                                       std::vector<size_t>(num_layers, 128),
                                       /*local_port=*/std::nullopt,
                                       /*host_blocks_to_allocate=*/kSlots,
                                       /*parallelism=*/1, nullptr),
          owner_(owner) {}

    absl::StatusOr<raiden::PjRtCopyFuture> D2hSyncDispatch(
        const std::vector<int64_t>& src_offsets_major_dim,
        const std::vector<int64_t>& dst_offsets_major_dim,
        const std::vector<int64_t>& copy_sizes_major_dim,
        std::optional<int64_t> slot_idx, std::optional<size_t> layer_idx,
        std::optional<size_t> shard_idx) override {
      auto [promise, future] = xla::MakePromise<>();
      absl::MutexLock lock(owner_->copies_mu_);
      owner_->copies_.push_back(std::move(promise));
      return raiden::PjRtCopyFuture(std::move(future), raiden::BufferHolders{});
    }

   private:
    TestManager* owner_;
  };

  absl::Mutex copies_mu_;
  std::vector<xla::Promise<>> copies_;
};

// A consumer whose host-to-device copies complete when the test says so.
class RecvTestManager : public KVCacheManagerWithTransfer {
 public:
  explicit RecvTestManager(size_t num_layers, double timeout_s = 5.0)
      : KVCacheManagerWithTransfer(std::make_unique<RecvBase>(num_layers, this),
                                   /*node_id=*/0,
                                   /*local_control_port=*/-1, /*max_blocks=*/1,
                                   /*num_slots=*/kSlots, timeout_s) {}

  void AddRecv(const std::string& req_id, uint64_t uuid,
               int32_t blocks_per_layer = 1,
               std::optional<std::chrono::steady_clock::time_point> deadline =
                   std::nullopt) {
    absl::MutexLock lock(mu_);
    active_recv_sessions_[uuid] = *TransferReceiveSession::Create(
        base(), staging_allocator_.get(), uuid, req_id, blocks_per_layer,
        deadline.value_or(DeadlineFromNow()), /*acquire_staging=*/true);
  }

  absl::Status ReceiveLayer(size_t layer, uint64_t uuid) {
    return base()->OnLayerReceived(layer, uuid);
  }

  absl::Status ReceiveBlocks(const std::vector<int>& blocks, uint64_t uuid) {
    return OnBlocksReceived(blocks, uuid);
  }

  void FinishCopy(size_t index, absl::Status status) {
    absl::MutexLock lock(copies_mu_);
    copies_.at(index).Set(std::move(status));
  }

  size_t copies_issued() {
    absl::MutexLock lock(copies_mu_);
    return copies_.size();
  }

  size_t free_slots() { return staging_allocator_->num_free_slots(); }
  StagingBlockAllocator* staging_allocator() {
    return staging_allocator_.get();
  }

  bool has_recv(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    auto it = active_recv_sessions_.find(uuid);
    return it != active_recv_sessions_.end() && !it->second->Done();
  }

  void FailRecv(uint64_t uuid, absl::Status status) {
    std::shared_ptr<TransferReceiveSession> session;
    {
      absl::MutexLock lock(mu_);
      auto it = active_recv_sessions_.find(uuid);
      if (it == active_recv_sessions_.end()) return;
      session = it->second;
    }
    session->Finish(status);
  }

  void BlockH2dDispatch() { block_dispatch_.store(true); }

  bool WaitForH2dDispatch(absl::Duration timeout) {
    return dispatch_entered_.WaitForNotificationWithTimeout(timeout);
  }

  void ReleaseH2dDispatch() {
    if (!release_dispatch_.HasBeenNotified()) release_dispatch_.Notify();
  }

 private:
  class RecvBase : public kv_cache::KVCacheManagerBase {
   public:
    RecvBase(size_t num_layers, RecvTestManager* owner)
        : kv_cache::KVCacheManagerBase(num_layers, /*num_shards=*/1,
                                       std::vector<size_t>(num_layers, 128),
                                       /*local_port=*/std::nullopt,
                                       /*host_blocks_to_allocate=*/kSlots,
                                       /*parallelism=*/1, nullptr),
          owner_(owner) {}

    absl::StatusOr<raiden::PjRtCopyFuture> H2dSyncDispatch(
        const std::vector<int64_t>& src_offsets_major_dim,
        const std::vector<int64_t>& dst_offsets_major_dim,
        const std::vector<int64_t>& copy_sizes_major_dim,
        std::optional<int64_t> slot_idx, std::optional<size_t> layer_idx,
        std::optional<size_t> shard_idx) override {
      if (owner_->block_dispatch_.load()) {
        owner_->dispatch_entered_.Notify();
        owner_->release_dispatch_.WaitForNotification();
      }
      auto [promise, future] = xla::MakePromise<>();
      absl::MutexLock lock(owner_->copies_mu_);
      owner_->copies_.push_back(std::move(promise));
      return raiden::PjRtCopyFuture(std::move(future), raiden::BufferHolders{});
    }

   private:
    RecvTestManager* owner_;
  };

  absl::Mutex copies_mu_;
  std::vector<xla::Promise<>> copies_;
  std::atomic<bool> block_dispatch_{false};
  absl::Notification dispatch_entered_;
  absl::Notification release_dispatch_;
};

class DispatchReleaseGuard {
 public:
  explicit DispatchReleaseGuard(RecvTestManager* manager) : manager_(manager) {}
  ~DispatchReleaseGuard() { Release(); }

  void Release() {
    if (manager_ == nullptr) return;
    manager_->ReleaseH2dDispatch();
    manager_ = nullptr;
  }

 private:
  RecvTestManager* manager_;
};

using Reports = std::tuple<std::vector<std::string>, std::vector<std::string>,
                           std::vector<std::string>>;

const std::vector<std::string>& DoneSending(const Reports& r) {
  return std::get<0>(r);
}
const std::vector<std::string>& DoneReceiving(const Reports& r) {
  return std::get<1>(r);
}
const std::vector<std::string>& FailedRecving(const Reports& r) {
  return std::get<2>(r);
}

TEST(SendDrainTest, ExpiredSendKeepsItsStagingUntilTheCopyEnds) {
  TestManager producer(/*num_layers=*/1);
  ASSERT_GT(producer.NotifyForRead("req", /*uuid=*/7, {0},
                                   std::chrono::steady_clock::now() -
                                       std::chrono::milliseconds(1)),
            0);
  producer.ServePull(7);
  ASSERT_EQ(producer.copies_issued(), 1);
  EXPECT_EQ(producer.free_slots(), kSlots - 1);

  // The deadline passes while the copy runs: the send is not reported and
  // its slot stays out of the pool.
  Reports during = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(during), IsEmpty());
  EXPECT_THAT(FailedRecving(during), IsEmpty());
  EXPECT_EQ(producer.free_slots(), kSlots - 1);

  // The copy lands after the deadline: the send is reported failed, not
  // done, and only now hands its slot back.
  producer.FinishCopy(0, absl::OkStatus());
  Reports after = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(after), IsEmpty());
  EXPECT_THAT(FailedRecving(after), Contains("req"));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendDrainTest, FailedLayerWaitsForTheOtherLayersCopies) {
  TestManager producer(/*num_layers=*/2);
  ASSERT_GT(producer.NotifyForRead("req", /*uuid=*/8, {0}), 0);
  producer.ServePull(8);
  ASSERT_EQ(producer.copies_issued(), 2);

  // Layer 0's copy fails while layer 1's still runs: the failure and the
  // slot are held back.
  producer.FinishCopy(0, absl::InternalError("copy failed"));
  Reports during = producer.CompleteReadRaw();
  EXPECT_THAT(FailedRecving(during), IsEmpty());
  EXPECT_EQ(producer.free_slots(), kSlots - 1);

  producer.FinishCopy(1, absl::OkStatus());
  Reports after = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(after), IsEmpty());
  EXPECT_THAT(FailedRecving(after), Contains("req"));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendDrainTest, SendNobodyPulledFailsAtItsDeadline) {
  TestManager producer(/*num_layers=*/1);
  ASSERT_GT(producer.NotifyForRead("req", /*uuid=*/9, {0},
                                   std::chrono::steady_clock::now() -
                                       std::chrono::milliseconds(1)),
            0);
  Reports swept = producer.CompleteReadRaw();
  EXPECT_THAT(FailedRecving(swept), Contains("req"));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendLifecycleTest, DuplicateRegistrationCannotReplaceLiveOffer) {
  TestManager producer(/*num_layers=*/1);
  ASSERT_GT(producer.NotifyForRead("first", /*uuid=*/10, {0},
                                   std::chrono::steady_clock::now() -
                                       std::chrono::milliseconds(1)),
            0);
  EXPECT_EQ(producer.NotifyForRead("replacement", /*uuid=*/10, {0}), 0);
  EXPECT_TRUE(producer.has_send(10));

  Reports reports = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(reports), IsEmpty());
  EXPECT_THAT(DoneReceiving(reports), IsEmpty());
  EXPECT_THAT(FailedRecving(reports), ElementsAre("first"));
  Reports repeated = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(repeated), IsEmpty());
  EXPECT_THAT(DoneReceiving(repeated), IsEmpty());
  EXPECT_THAT(FailedRecving(repeated), IsEmpty());
}

TEST(SendLifecycleTest, DuplicateBlocksAreRejectedAtRegistration) {
  TestManager producer(/*num_layers=*/1);

  EXPECT_EQ(producer.NotifyForRead("req", /*uuid=*/15, {0, 0}), 0);
  EXPECT_FALSE(producer.has_send(15));
  Reports reports = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(reports), IsEmpty());
  EXPECT_THAT(DoneReceiving(reports), IsEmpty());
  EXPECT_THAT(FailedRecving(reports), IsEmpty());
}

TEST(SendLifecycleTest, SuccessfulSendWithoutWorkSettlesImmediately) {
  TestManager producer(/*num_layers=*/1);
  auto session = producer.AddSyntheticSend("req", /*uuid=*/10, /*in_flight=*/0);

  producer.Decide(session, /*failed=*/false);
  Reports reports = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(reports), Contains("req"));
  EXPECT_THAT(FailedRecving(reports), IsEmpty());
  EXPECT_FALSE(producer.has_send(10));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendLifecycleTest, FailedSendWithoutWorkSettlesImmediately) {
  TestManager producer(/*num_layers=*/1);
  auto session = producer.AddSyntheticSend("req", /*uuid=*/11, /*in_flight=*/0);

  producer.Decide(session, /*failed=*/true);
  Reports reports = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(reports), IsEmpty());
  EXPECT_THAT(FailedRecving(reports), Contains("req"));
  EXPECT_FALSE(producer.has_send(11));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendLifecycleTest, SuccessfulSendWaitsForEveryOperation) {
  TestManager producer(/*num_layers=*/2);
  auto session = producer.AddSyntheticSend("req", /*uuid=*/12, /*in_flight=*/2);

  producer.Decide(session, /*failed=*/false);
  producer.FinishCopy(0, absl::OkStatus());
  Reports during = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(during), IsEmpty());
  EXPECT_THAT(FailedRecving(during), IsEmpty());
  EXPECT_TRUE(producer.has_send(12));
  EXPECT_EQ(producer.free_slots(), kSlots - 1);

  producer.FinishCopy(1, absl::OkStatus());
  Reports after = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(after), Contains("req"));
  EXPECT_THAT(FailedRecving(after), IsEmpty());
  EXPECT_FALSE(producer.has_send(12));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendLifecycleTest, FailureCannotOverrideAnEarlierSuccess) {
  TestManager producer(/*num_layers=*/1);
  auto session = producer.AddSyntheticSend("req", /*uuid=*/13, /*in_flight=*/1);

  producer.Decide(session, /*failed=*/false);
  producer.Decide(session, /*failed=*/true);
  producer.FinishCopy(0, absl::OkStatus());
  Reports reports = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(reports), Contains("req"));
  EXPECT_THAT(FailedRecving(reports), IsEmpty());
  EXPECT_FALSE(producer.has_send(13));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendLifecycleTest, SuccessCannotOverrideAnEarlierFailure) {
  TestManager producer(/*num_layers=*/1);
  auto session = producer.AddSyntheticSend("req", /*uuid=*/14, /*in_flight=*/1);

  producer.Decide(session, /*failed=*/true);
  producer.Decide(session, /*failed=*/false);
  producer.FinishCopy(0, absl::OkStatus());
  Reports reports = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(reports), IsEmpty());
  EXPECT_THAT(FailedRecving(reports), Contains("req"));
  EXPECT_FALSE(producer.has_send(14));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(RecvLifecycleTest, NetworkCompletionWaitsForH2d) {
  RecvTestManager consumer(/*num_layers=*/1);
  consumer.AddRecv("req", /*uuid=*/20);
  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/20).ok());
  ASSERT_EQ(consumer.copies_issued(), 1);
  ASSERT_TRUE(consumer.ReceiveBlocks({0}, /*uuid=*/20).ok());

  Reports during = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(during), IsEmpty());
  EXPECT_TRUE(consumer.has_recv(20));
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  consumer.FinishCopy(0, absl::OkStatus());
  Reports after = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(after), Contains("req"));
  EXPECT_THAT(FailedRecving(after), IsEmpty());
  EXPECT_FALSE(consumer.has_recv(20));
  EXPECT_EQ(consumer.free_slots(), kSlots);
}

TEST(RecvLifecycleTest, InvalidReadShapeDoesNotLeakStaging) {
  RecvTestManager consumer(/*num_layers=*/1);
  const size_t free_before = consumer.free_slots();

  EXPECT_THROW(consumer.StartRead("req", /*uuid=*/25,
                                  /*remote_endpoint=*/"unused:1",
                                  /*remote_block_ids=*/{0, 1},
                                  /*local_block_ids=*/{0}),
               std::invalid_argument);
  EXPECT_EQ(consumer.free_slots(), free_before);
  EXPECT_FALSE(consumer.has_recv(25));
}

TEST(RecvLifecycleTest, LateBlockAccountingAfterRetirementIsANoOp) {
  RecvTestManager consumer(/*num_layers=*/1);
  consumer.AddRecv("req", /*uuid=*/21);
  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/21).ok());
  consumer.FinishCopy(0, absl::OkStatus());

  Reports reports = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(reports), Contains("req"));
  EXPECT_THAT(FailedRecving(reports), IsEmpty());
  EXPECT_FALSE(consumer.has_recv(21));
  EXPECT_EQ(consumer.free_slots(), kSlots);

  // BlockTransport calls this immediately after OnLayerReceived. A fast H2D
  // callback may already have retired the receive, so the late accounting is
  // deliberately harmless.
  EXPECT_TRUE(consumer.ReceiveBlocks({0}, /*uuid=*/21).ok());
  Reports after_late_accounting = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(after_late_accounting), IsEmpty());
  EXPECT_THAT(DoneReceiving(after_late_accounting), IsEmpty());
  EXPECT_THAT(FailedRecving(after_late_accounting), IsEmpty());
  EXPECT_FALSE(consumer.has_recv(21));
  EXPECT_EQ(consumer.free_slots(), kSlots);
}

TEST(RecvLifecycleTest, OutOfOrderLayersSettleAfterEveryH2d) {
  RecvTestManager consumer(/*num_layers=*/2);
  consumer.AddRecv("req", /*uuid=*/22);

  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/1, /*uuid=*/22).ok());
  ASSERT_TRUE(consumer.ReceiveBlocks({0}, /*uuid=*/22).ok());
  consumer.FinishCopy(0, absl::OkStatus());
  Reports after_one = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(after_one), IsEmpty());
  EXPECT_TRUE(consumer.has_recv(22));

  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/22).ok());
  ASSERT_TRUE(consumer.ReceiveBlocks({0}, /*uuid=*/22).ok());
  consumer.FinishCopy(1, absl::OkStatus());
  Reports after_both = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(after_both), Contains("req"));
  EXPECT_THAT(FailedRecving(after_both), IsEmpty());
  EXPECT_FALSE(consumer.has_recv(22));
  EXPECT_EQ(consumer.free_slots(), kSlots);
}

TEST(RecvLifecycleTest, SingleFailedH2dReportsFailureAndReturnsStaging) {
  RecvTestManager consumer(/*num_layers=*/1);
  consumer.AddRecv("req", /*uuid=*/23);
  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/23).ok());
  consumer.FinishCopy(0, absl::InternalError("copy failed"));

  Reports reports = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(reports), IsEmpty());
  EXPECT_THAT(FailedRecving(reports), Contains("req"));
  EXPECT_FALSE(consumer.has_recv(23));
  EXPECT_EQ(consumer.free_slots(), kSlots);
}

TEST(RecvLifecycleTest, ReceiveWithoutTrafficFailsAtItsDeadline) {
  RecvTestManager consumer(/*num_layers=*/1, /*timeout_s=*/kTimeoutS);
  consumer.AddRecv(
      "req", /*uuid=*/24, /*blocks_per_layer=*/1,
      std::chrono::steady_clock::now() - std::chrono::milliseconds(1));

  Reports reports = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(reports), IsEmpty());
  EXPECT_THAT(FailedRecving(reports), Contains("req"));
  EXPECT_FALSE(consumer.has_recv(24));
  EXPECT_EQ(consumer.free_slots(), kSlots);
}

TEST(RecvDrainTest, ExpiredReceiveKeepsStagingUntilH2dEnds) {
  RecvTestManager consumer(/*num_layers=*/1, /*timeout_s=*/kTimeoutS);
  consumer.AddRecv(
      "req", /*uuid=*/25, /*blocks_per_layer=*/1,
      std::chrono::steady_clock::now() + std::chrono::milliseconds(10));
  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/25).ok());
  ASSERT_EQ(consumer.copies_issued(), 1);

  absl::SleepFor(absl::Milliseconds(15));
  Reports during = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(during), IsEmpty());
  EXPECT_THAT(DoneReceiving(during), IsEmpty());
  EXPECT_THAT(FailedRecving(during), IsEmpty());
  EXPECT_TRUE(consumer.has_recv(25));
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  consumer.FinishCopy(0, absl::OkStatus());
  Reports after = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(after), IsEmpty());
  EXPECT_THAT(DoneReceiving(after), IsEmpty());
  EXPECT_THAT(FailedRecving(after), ElementsAre("req"));
  EXPECT_FALSE(consumer.has_recv(25));
  EXPECT_EQ(consumer.free_slots(), kSlots);
  Reports repeated = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(repeated), IsEmpty());
  EXPECT_THAT(DoneReceiving(repeated), IsEmpty());
  EXPECT_THAT(FailedRecving(repeated), IsEmpty());
}

TEST(RecvDrainTest, DuplicateUuidIsRejectedUntilExpiredReceiveDrains) {
  RecvTestManager consumer(/*num_layers=*/1, /*timeout_s=*/kTimeoutS);
  consumer.AddRecv(
      "old", /*uuid=*/28, /*blocks_per_layer=*/1,
      std::chrono::steady_clock::now() + std::chrono::milliseconds(10));
  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/28).ok());
  ASSERT_EQ(consumer.copies_issued(), 1);

  absl::SleepFor(absl::Milliseconds(15));
  Reports during = consumer.CompleteReadRaw();
  ASSERT_THAT(DoneSending(during), IsEmpty());
  ASSERT_THAT(DoneReceiving(during), IsEmpty());
  ASSERT_THAT(FailedRecving(during), IsEmpty());
  ASSERT_TRUE(consumer.has_recv(28));

  // Reusing a UUID while callbacks can still arrive would let old traffic
  // mutate the replacement. The public registration API must keep the old
  // session authoritative until its issued work has drained.
  EXPECT_FALSE(consumer
                   .RegisterRecv(/*uuid=*/28, "retry",
                                 /*expected_block_count=*/1)
                   .ok());
  EXPECT_TRUE(consumer.has_recv(28));
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  consumer.FinishCopy(0, absl::OkStatus());
  Reports retired = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(retired), IsEmpty());
  EXPECT_THAT(DoneReceiving(retired), IsEmpty());
  EXPECT_THAT(FailedRecving(retired), ElementsAre("old"));
  EXPECT_FALSE(consumer.has_recv(28));
  EXPECT_EQ(consumer.free_slots(), kSlots);

  EXPECT_TRUE(consumer
                  .RegisterRecv(/*uuid=*/28, "retry",
                                /*expected_block_count=*/1,
                                std::chrono::steady_clock::now() -
                                    std::chrono::milliseconds(1))
                  .ok());
  Reports retry = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(retry), IsEmpty());
  EXPECT_THAT(DoneReceiving(retry), IsEmpty());
  EXPECT_THAT(FailedRecving(retry), ElementsAre("retry"));
  EXPECT_FALSE(consumer.has_recv(28));

  Reports repeated = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(repeated), IsEmpty());
  EXPECT_THAT(DoneReceiving(repeated), IsEmpty());
  EXPECT_THAT(FailedRecving(repeated), IsEmpty());
}

TEST(RecvDrainTest, FailedLayerWaitsForOtherH2dCopies) {
  RecvTestManager consumer(/*num_layers=*/2);
  consumer.AddRecv("req", /*uuid=*/26);
  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/26).ok());
  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/1, /*uuid=*/26).ok());
  ASSERT_EQ(consumer.copies_issued(), 2);

  consumer.FinishCopy(0, absl::InternalError("copy failed"));
  Reports during = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(during), IsEmpty());
  EXPECT_THAT(DoneReceiving(during), IsEmpty());
  EXPECT_THAT(FailedRecving(during), IsEmpty());
  EXPECT_TRUE(consumer.has_recv(26));
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  consumer.FinishCopy(1, absl::OkStatus());
  Reports after = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(after), IsEmpty());
  EXPECT_THAT(DoneReceiving(after), IsEmpty());
  EXPECT_THAT(FailedRecving(after), ElementsAre("req"));
  EXPECT_FALSE(consumer.has_recv(26));
  EXPECT_EQ(consumer.free_slots(), kSlots);
  Reports repeated = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(repeated), IsEmpty());
  EXPECT_THAT(DoneReceiving(repeated), IsEmpty());
  EXPECT_THAT(FailedRecving(repeated), IsEmpty());
}

TEST(RecvDrainTest, TimeoutDuringH2dDispatchKeepsStaging) {
  RecvTestManager consumer(/*num_layers=*/1, /*timeout_s=*/kTimeoutS);
  consumer.AddRecv(
      "req", /*uuid=*/27, /*blocks_per_layer=*/1,
      std::chrono::steady_clock::now() + std::chrono::milliseconds(10));
  consumer.BlockH2dDispatch();
  auto receive = std::async(std::launch::async, [&consumer] {
    return consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/27);
  });
  DispatchReleaseGuard release_dispatch(&consumer);
  ASSERT_TRUE(consumer.WaitForH2dDispatch(absl::Seconds(5)));

  absl::SleepFor(absl::Milliseconds(15));
  Reports during = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(during), IsEmpty());
  EXPECT_THAT(DoneReceiving(during), IsEmpty());
  EXPECT_THAT(FailedRecving(during), IsEmpty());
  EXPECT_TRUE(consumer.has_recv(27));
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  release_dispatch.Release();
  ASSERT_EQ(receive.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  ASSERT_TRUE(receive.get().ok());
  ASSERT_EQ(consumer.copies_issued(), 1);
  consumer.FinishCopy(0, absl::OkStatus());
  Reports after = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(after), IsEmpty());
  EXPECT_THAT(DoneReceiving(after), IsEmpty());
  EXPECT_THAT(FailedRecving(after), ElementsAre("req"));
  EXPECT_FALSE(consumer.has_recv(27));
  EXPECT_EQ(consumer.free_slots(), kSlots);
  Reports repeated = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(repeated), IsEmpty());
  EXPECT_THAT(DoneReceiving(repeated), IsEmpty());
  EXPECT_THAT(FailedRecving(repeated), IsEmpty());
}

TEST(SendDrainTest, WaitForPendingWorkWaitsForActiveSendSession) {
  TestManager producer(/*num_layers=*/1, /*timeout_s=*/5.0);
  auto session = producer.AddSyntheticSend("req", /*uuid=*/77, /*in_flight=*/1);
  producer.Decide(session, /*failed=*/false);
  ASSERT_EQ(producer.copies_issued(), 1);

  absl::Notification wait_finished;
  absl::Status wait_status;
  std::thread waiter([&]() {
    wait_status = producer.WaitForPendingWork();
    wait_finished.Notify();
  });

  EXPECT_FALSE(
      wait_finished.WaitForNotificationWithTimeout(absl::Milliseconds(150)));

  producer.FinishCopy(0, absl::OkStatus());
  EXPECT_TRUE(wait_finished.WaitForNotificationWithTimeout(absl::Seconds(2)));
  EXPECT_THAT(wait_status, ::absl_testing::IsOk());
  waiter.join();
}

TEST(RecvLifecycleTest,
     SessionRemainsValidAcrossAsyncH2dCallbacksAfterCallerDropsPtr) {
  RecvTestManager consumer(/*num_layers=*/1);
  std::weak_ptr<TransferReceiveSession> weak_session;
  {
    std::shared_ptr<TransferReceiveSession> session =
        *TransferReceiveSession::Create(
            consumer.base(), consumer.staging_allocator(),
            /*uuid=*/88, "req_lifetime",
            /*total_blocks=*/1,
            std::chrono::steady_clock::now() + std::chrono::seconds(10),
            /*acquire_staging=*/true);
    weak_session = session;
    EXPECT_TRUE(session->HasStaging());
    EXPECT_EQ(consumer.free_slots(), kSlots - 1);
    ASSERT_THAT(session->ExecuteLayerH2d(consumer, /*layer_idx=*/0),
                ::absl_testing::IsOk());
    ASSERT_EQ(consumer.copies_issued(), 1);
  }
  // Caller dropped its std::shared_ptr; the pending H2D callback retains the
  // session and its staging slot until the copy completes.
  EXPECT_FALSE(weak_session.expired());
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  consumer.FinishCopy(0, absl::OkStatus());
  EXPECT_TRUE(weak_session.expired());
  EXPECT_EQ(consumer.free_slots(), kSlots);
}

TEST(RecvLifecycleTest,
     IncomingPushLeasePinsStagingDuringWriteAndRejectsWhenDraining) {
  RecvTestManager consumer(/*num_layers=*/1, /*timeout_s=*/10.0);
  consumer.AddRecv("req_ingress_lease", /*uuid=*/91, /*blocks_per_layer=*/1);
  ASSERT_EQ(consumer.free_slots(), kSlots - 1);

  EXPECT_THAT(consumer.base()->BeginIncomingPush(/*uuid=*/999),
              ::absl_testing::StatusIs(absl::StatusCode::kNotFound));

  ASSERT_THAT(consumer.base()->BeginIncomingPush(/*uuid=*/91),
              ::absl_testing::IsOk());

  // Failing/timing out mid-write marks the session draining and keeps staging
  // pinned until EndIncomingPush finishes.
  consumer.FailRecv(/*uuid=*/91, absl::InternalError("simulated failure"));
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  EXPECT_THAT(consumer.base()->BeginIncomingPush(/*uuid=*/91),
              ::absl_testing::StatusIs(absl::StatusCode::kCancelled));

  EXPECT_THAT(consumer.base()->EndIncomingPush(/*uuid=*/91),
              ::absl_testing::StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(consumer.free_slots(), kSlots);
}

TEST(RecvLifecycleTest,
     NonSessionTransfersSucceedOverBlockTransportWhileStaleUuidIsRejected) {
  constexpr size_t kSliceBytes = 128;
  KVCacheManagerWithTransfer sender(
      /*num_layers=*/1, /*num_shards=*/1, kSliceBytes,
      /*local_port=*/0, /*host_blocks_to_allocate=*/4,
      /*parallelism=*/1, /*node_id=*/0, /*local_control_port=*/-1,
      /*max_blocks=*/1, /*num_slots=*/1, /*timeout_s=*/10.0);
  KVCacheManagerWithTransfer receiver(
      /*num_layers=*/1, /*num_shards=*/1, kSliceBytes,
      /*local_port=*/0, /*host_blocks_to_allocate=*/4,
      /*parallelism=*/1, /*node_id=*/1, /*local_control_port=*/-1,
      /*max_blocks=*/1, /*num_slots=*/1, /*timeout_s=*/10.0);

  const std::string peer =
      absl::StrCat("127.0.0.1:", *receiver.base()->local_port());

  // 1. Raw H2H push with dynamic block allocation (op = 1, uuid = 0).
  std::memset(sender.base()->GetBlockHostPointer(0, 0, 0), 0x5A, kSliceBytes);
  absl::StatusOr<std::vector<int>> dyn_ids =
      sender.base()->H2hWriteDirect(peer, /*src_block_ids=*/{0});
  ASSERT_THAT(dyn_ids, ::absl_testing::IsOk());
  ASSERT_EQ(dyn_ids->size(), 1);
  EXPECT_EQ(receiver.base()->GetBlockHostPointer(0, 0, (*dyn_ids)[0])[0], 0x5A);

  // 2. Raw H2H push with explicit destination block (op = 6, uuid = 0).
  std::memset(sender.base()->GetBlockHostPointer(0, 0, 0), 0xA5, kSliceBytes);
  absl::StatusOr<std::vector<int>> exp_ids = sender.base()->H2hWriteDirect(
      peer, /*src_block_ids=*/{0}, /*dst_block_ids=*/{2}, /*uuid=*/0);
  ASSERT_THAT(exp_ids, ::absl_testing::IsOk());
  EXPECT_EQ(receiver.base()->GetBlockHostPointer(0, 0, 2)[0], 0xA5);

  // 3. Host-only (MEMORY_TYPE_DRAM) active plan push (op = 6, uuid = 42,
  //    no TransferReceiveSession in active_recv_sessions_).
  ::tpu_sync::rpc::StartTransferRequest dram_plan;
  dram_plan.set_uuid(42);
  dram_plan.set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  dram_plan.set_use_block_chunks(true);
  auto* entry = (*dram_plan.mutable_shard_push_schedules())[0].add_entries();
  entry->set_dst_peer(peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_block_id(0);
  entry->set_dst_block_id(3);
  entry->set_size_bytes(kSliceBytes);
  entry->set_count(1);
  ASSERT_THAT(receiver.RegisterActivePlan(42, dram_plan, /*is_sender=*/false),
              ::absl_testing::IsOk());

  std::memset(sender.base()->GetBlockHostPointer(0, 0, 0), 0x3C, kSliceBytes);
  absl::StatusOr<std::vector<int>> plan_ids = sender.base()->H2hWriteDirect(
      peer, /*src_block_ids=*/{0}, /*dst_block_ids=*/{3}, /*uuid=*/42);
  ASSERT_THAT(plan_ids, ::absl_testing::IsOk());
  EXPECT_EQ(receiver.base()->GetBlockHostPointer(0, 0, 3)[0], 0x3C);

  // 4. Once the DRAM plan is unregistered, pushes for uuid = 42 are rejected
  //    before writing to host memory, preserving the existing bytes at block 3.
  ASSERT_THAT(receiver.UnregisterActivePlan(42), ::absl_testing::IsOk());
  std::memset(sender.base()->GetBlockHostPointer(0, 0, 0), 0xFF, kSliceBytes);
  absl::StatusOr<std::vector<int>> rejected = sender.base()->H2hWriteDirect(
      peer, /*src_block_ids=*/{0}, /*dst_block_ids=*/{3}, /*uuid=*/42);
  EXPECT_FALSE(rejected.ok());
  EXPECT_EQ(receiver.base()->GetBlockHostPointer(0, 0, 3)[0], 0x3C);
}
}  // namespace
}  // namespace tpu_raiden
