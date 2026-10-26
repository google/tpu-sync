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
#include <cstring>
#include <future>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/future.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/fault_injection/fault_injector.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"
#include "tpu_sync/kv_cache/staging_window.h"
#include "tpu_sync/kv_cache/transfer_receive_session.h"
#include "tpu_sync/kv_cache/transfer_send_session.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace {

using ::testing::_;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Ge;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Le;
using ::testing::Optional;

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
    CHECK_LT(index, copies_.size())
        << "FinishCopy index " << index << " out of bounds (" << copies_.size()
        << ")";
    copies_[index].Set(std::move(status));
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

  bool send_draining(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    auto it = send_sessions_.find(uuid);
    return it != send_sessions_.end() && it->second->IsDraining();
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
  explicit RecvTestManager(
      size_t num_layers, double timeout_s = 5.0, size_t num_slots = kSlots,
      size_t num_shards = 1,
      std::optional<StagingWindowConfig> staging_window_config = std::nullopt)
      : KVCacheManagerWithTransfer(
            std::make_unique<RecvBase>(num_layers, num_slots, num_shards, this),
            /*node_id=*/0, /*local_control_port=*/-1, /*max_blocks=*/1,
            num_slots, timeout_s, /*metrics_collector=*/nullptr,
            staging_window_config) {}

  void AddRecv(const std::string& req_id, uint64_t uuid,
               int32_t blocks_per_layer = 1,
               std::optional<std::chrono::steady_clock::time_point> deadline =
                   std::nullopt) {
    absl::MutexLock lock(mu_);
    active_recv_sessions_[uuid] = *TransferReceiveSession::Create(
        base(), staging_allocator_.get(), uuid, req_id, blocks_per_layer,
        deadline.value_or(DeadlineFromNow()), /*acquire_staging=*/true);
  }

  absl::StatusOr<std::shared_ptr<TransferReceiveSession>> AddWindowedRecv(
      const std::string& req_id, uint64_t uuid,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids,
      StagingWindowConfig staging_window_config,
      std::optional<std::chrono::steady_clock::time_point> deadline =
          std::nullopt) {
    ABSL_ASSIGN_OR_RETURN(
        std::shared_ptr<TransferReceiveSession> session,
        TransferReceiveSession::Create(
            base(), staging_allocator_.get(), uuid, req_id, remote_block_ids,
            local_block_ids, /*local_host_block_ids=*/std::nullopt,
            deadline.value_or(DeadlineFromNow()), staging_window_config));
    session->EndRecvOp();
    absl::MutexLock lock(mu_);
    active_recv_sessions_[uuid] = session;
    return session;
  }

  void InsertPoolReshardRecv(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    active_pool_reshard_recvs_[uuid] = nullptr;
  }

  void ErasePoolReshardRecv(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    active_pool_reshard_recvs_.erase(uuid);
  }

  absl::Status ReceiveLayer(size_t layer, uint64_t uuid) {
    return base()->OnLayerReceived(layer, uuid);
  }

  absl::Status ReceiveBlocks(const std::vector<int>& blocks, uint64_t uuid) {
    std::vector<int> all_shards(base()->num_shards());
    std::iota(all_shards.begin(), all_shards.end(), 0);
    return OnBlockShardsReceived(blocks, all_shards, uuid);
  }

  void FinishCopy(size_t index, absl::Status status) {
    absl::MutexLock lock(copies_mu_);
    CHECK_LT(index, copies_.size())
        << "FinishCopy index " << index << " out of bounds (" << copies_.size()
        << ")";
    copies_[index].Set(std::move(status));
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

  std::shared_ptr<TransferReceiveSession> GetRecvSession(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    auto it = active_recv_sessions_.find(uuid);
    return it != active_recv_sessions_.end() ? it->second : nullptr;
  }

  struct IssuedCopy {
    std::vector<int64_t> src_offsets;
    std::vector<int64_t> dst_offsets;
    std::vector<int64_t> sizes;
    std::optional<size_t> layer_idx;
  };

  std::vector<IssuedCopy> issued_copies() {
    absl::MutexLock lock(copies_mu_);
    return issued_copies_;
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
    RecvBase(size_t num_layers, size_t num_slots, size_t num_shards,
             RecvTestManager* owner)
        : kv_cache::KVCacheManagerBase(num_layers, num_shards,
                                       std::vector<size_t>(num_layers, 128),
                                       /*local_port=*/std::nullopt,
                                       /*host_blocks_to_allocate=*/num_slots,
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
      {
        absl::MutexLock lock(owner_->copies_mu_);
        owner_->copies_.push_back(std::move(promise));
        owner_->issued_copies_.push_back({src_offsets_major_dim,
                                          dst_offsets_major_dim,
                                          copy_sizes_major_dim, layer_idx});
      }
      return raiden::PjRtCopyFuture(std::move(future), raiden::BufferHolders{});
    }

   private:
    RecvTestManager* owner_;
  };

  absl::Mutex copies_mu_;
  std::vector<xla::Promise<>> copies_ ABSL_GUARDED_BY(copies_mu_);
  std::vector<IssuedCopy> issued_copies_ ABSL_GUARDED_BY(copies_mu_);
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

int64_t SteadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
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
  EXPECT_THROW(producer.NotifyForRead("replacement", /*uuid=*/10, {0}),
               std::runtime_error);
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

  // Returning 0 would be indistinguishable from "nothing to send"; the caller
  // must receive an exception identifying the duplicate block and its indices.
  std::string error_message;
  try {
    producer.NotifyForRead("req", /*uuid=*/15, {5, 9, 5});
    FAIL() << "a duplicate block must be rejected";
  } catch (const std::invalid_argument& e) {
    error_message = e.what();
  }
  EXPECT_THAT(error_message, HasSubstr("duplicate block 5"));
  EXPECT_THAT(error_message, HasSubstr("req_id=req, uuid=15"));
  EXPECT_THAT(error_message, HasSubstr("positions 0 and 2 of 3 blocks"));
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

TEST(SendLifecycleTest, DetailsReportWhenTheSendCompleted) {
  TestManager producer(/*num_layers=*/1);
  auto session = producer.AddSyntheticSend("req", /*uuid=*/16, /*in_flight=*/1);
  producer.Decide(session, /*failed=*/false);

  const int64_t before = SteadyNowNs();
  producer.FinishCopy(0, absl::OkStatus());
  const int64_t completed = SteadyNowNs();
  // A late poll must not move the reported completion time.
  absl::SleepFor(absl::Milliseconds(20));
  CompleteReadResult completions = producer.CompleteReadWithDetails();

  EXPECT_THAT(completions.sent, ElementsAre("req"));
  EXPECT_THAT(completions.details["req"].completed_ns,
              Optional(AllOf(Ge(before), Le(completed))));
  EXPECT_THAT(producer.CompleteReadWithDetails().details, IsEmpty());
}

TEST(SendLifecycleTest, DetailsSayAnUnpulledSendTimedOutAwaitingItsPull) {
  TestManager producer(/*num_layers=*/1);
  ASSERT_GT(producer.NotifyForRead("req", /*uuid=*/17, {0},
                                   std::chrono::steady_clock::now() -
                                       std::chrono::milliseconds(1)),
            0);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _,
                       HasSubstr("Transfer failed: req_id=req "
                                 "code=DEADLINE_EXCEEDED phase=await_pull")));
  log.StartCapturingLogs();

  CompleteReadResult completions = producer.CompleteReadWithDetails();

  EXPECT_THAT(completions.failed, ElementsAre("req"));
  const TransferDetails& details = completions.details["req"];
  EXPECT_EQ(details.code, "DEADLINE_EXCEEDED");
  EXPECT_EQ(details.reason, "Send session timed out");
  EXPECT_EQ(details.phase, "await_pull");
  EXPECT_EQ(details.peer, std::nullopt);
  EXPECT_THAT(details.elapsed_ns, Optional(Ge(0)));
}

TEST(SendDrainTest, DetailsSayAnExpiredSendWasStillCopying) {
  TestManager producer(/*num_layers=*/1);
  ASSERT_GT(producer.NotifyForRead("req", /*uuid=*/18, {0},
                                   std::chrono::steady_clock::now() -
                                       std::chrono::milliseconds(1)),
            0);
  producer.ServePull(18);
  ASSERT_THAT(producer.CompleteReadWithDetails().failed, IsEmpty());

  producer.FinishCopy(0, absl::OkStatus());
  CompleteReadResult completions = producer.CompleteReadWithDetails();

  EXPECT_THAT(completions.failed, ElementsAre("req"));
  const TransferDetails& details = completions.details["req"];
  EXPECT_EQ(details.code, "DEADLINE_EXCEEDED");
  EXPECT_EQ(details.phase, "d2h");
  EXPECT_EQ(details.peer, "127.0.0.1:1");
}

TEST(SendDrainTest, DetailsCarryTheErrorOfAFailedCopy) {
  TestManager producer(/*num_layers=*/1);
  ASSERT_GT(producer.NotifyForRead("req", /*uuid=*/19, {0}), 0);
  producer.ServePull(19);
  producer.FinishCopy(0, absl::InternalError("copy failed"));

  CompleteReadResult completions = producer.CompleteReadWithDetails();

  EXPECT_THAT(completions.failed, ElementsAre("req"));
  const TransferDetails& details = completions.details["req"];
  EXPECT_EQ(details.code, "INTERNAL");
  EXPECT_EQ(details.reason, "copy failed");
  EXPECT_EQ(details.phase, "d2h");
}

TEST(SendDrainTest, DetailsSayAFailedPushFailedInPushWhileCopiesRun) {
  GetFaultInjector().Reset();
  absl::Cleanup reset_faults = [] { GetFaultInjector().Reset(); };
  FaultInjectionRule failed_push;
  failed_push.hook = std::string(hooks::kRaidenManagerBaseH2hWrite);
  failed_push.probability = 1.0;
  ABSL_ASSERT_OK(GetFaultInjector().Install({failed_push}));
  TestManager producer(/*num_layers=*/2, /*timeout_s=*/5.0);
  ASSERT_GT(producer.NotifyForRead("req", /*uuid=*/27, {0}), 0);
  producer.ServePull(27);
  ASSERT_EQ(producer.copies_issued(), 2);

  // Layer 0's push fails on a push worker while layer 1 is still copying.
  producer.FinishCopy(0, absl::OkStatus());
  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (!producer.send_draining(27) && absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(1));
  }
  ASSERT_TRUE(producer.send_draining(27));
  producer.FinishCopy(1, absl::OkStatus());
  CompleteReadResult completions = producer.CompleteReadWithDetails();

  EXPECT_THAT(completions.failed, ElementsAre("req"));
  const TransferDetails& details = completions.details["req"];
  EXPECT_EQ(details.code, "INTERNAL");
  EXPECT_THAT(details.reason, Optional(HasSubstr("injected fault")));
  EXPECT_EQ(details.phase, "push");
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

TEST(RecvLifecycleTest, DetailsReportWhenTheLastH2dLanded) {
  RecvTestManager consumer(/*num_layers=*/1);
  consumer.AddRecv("req", /*uuid=*/26);
  ASSERT_TRUE(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/26).ok());
  ASSERT_TRUE(consumer.ReceiveBlocks({0}, /*uuid=*/26).ok());

  const int64_t before = SteadyNowNs();
  consumer.FinishCopy(0, absl::OkStatus());
  const int64_t completed = SteadyNowNs();
  absl::SleepFor(absl::Milliseconds(20));
  CompleteReadResult completions = consumer.CompleteReadWithDetails();

  EXPECT_THAT(completions.received, ElementsAre("req"));
  EXPECT_THAT(completions.details["req"].completed_ns,
              Optional(AllOf(Ge(before), Le(completed))));
}

TEST(RecvLifecycleTest, DetailsSayAReceiveTimedOutWaitingForData) {
  RecvTestManager consumer(/*num_layers=*/1, /*timeout_s=*/kTimeoutS);
  consumer.AddRecv(
      "req", /*uuid=*/28, /*blocks_per_layer=*/1,
      std::chrono::steady_clock::now() - std::chrono::milliseconds(1));

  CompleteReadResult completions = consumer.CompleteReadWithDetails();

  EXPECT_THAT(completions.failed, ElementsAre("req"));
  const TransferDetails& details = completions.details["req"];
  EXPECT_EQ(details.code, "DEADLINE_EXCEEDED");
  EXPECT_EQ(details.reason, "Receive session timed out");
  EXPECT_EQ(details.phase, "receive");
}

TEST(RecvLifecycleTest, DetailsSayAFailedUploadFailedInH2dWhileDataArrives) {
  RecvTestManager consumer(/*num_layers=*/1);
  consumer.AddRecv("req", /*uuid=*/29);
  ABSL_ASSERT_OK(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/29));
  consumer.FinishCopy(0, absl::InternalError("copy failed"));

  CompleteReadResult completions = consumer.CompleteReadWithDetails();

  EXPECT_THAT(completions.failed, ElementsAre("req"));
  const TransferDetails& details = completions.details["req"];
  EXPECT_EQ(details.code, "INTERNAL");
  EXPECT_EQ(details.reason, "copy failed");
  EXPECT_EQ(details.phase, "h2d");
  EXPECT_EQ(details.peer, std::nullopt);
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
     IncomingPushLeaseSpansLayerH2dAndBlockAccountingBeforeReleasing) {
  RecvTestManager consumer(/*num_layers=*/1, /*timeout_s=*/10.0);
  consumer.AddRecv("req_handoff", /*uuid=*/92, /*blocks_per_layer=*/1);
  ASSERT_EQ(consumer.free_slots(), kSlots - 1);

  // Simulate BlockTransport::HandleIncomingPush holding the incoming push lease
  // across OnLayerReceived, synchronous H2D completion, and
  // OnBlockShardsReceived.
  ASSERT_THAT(consumer.base()->BeginIncomingPush(/*uuid=*/92),
              ::absl_testing::IsOk());
  ASSERT_THAT(consumer.ReceiveLayer(/*layer=*/0, /*uuid=*/92),
              ::absl_testing::IsOk());
  consumer.FinishCopy(0, absl::OkStatus());
  // Even though the H2D copy finished, staging must remain pinned until
  // OnBlockShardsReceived and EndIncomingPush complete.
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  ASSERT_THAT(consumer.ReceiveBlocks({0}, /*uuid=*/92), ::absl_testing::IsOk());
  EXPECT_EQ(consumer.free_slots(), kSlots - 1);

  EXPECT_THAT(consumer.base()->EndIncomingPush(/*uuid=*/92),
              ::absl_testing::IsOk());
  EXPECT_EQ(consumer.free_slots(), kSlots);
  Reports reports = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(reports), Contains("req_handoff"));
  EXPECT_THAT(FailedRecving(reports), IsEmpty());
}

TEST(
    RecvLifecycleTest,
    FailedIncomingPushImmediatelyFailsSessionAndReleasesStagingBeforeDeadline) {
  RecvTestManager consumer(/*num_layers=*/1, /*timeout_s=*/120.0);
  consumer.AddRecv("req_push_died", /*uuid=*/93, /*blocks_per_layer=*/1);
  ASSERT_EQ(consumer.free_slots(), kSlots - 1);

  ABSL_ASSERT_OK(consumer.base()->BeginIncomingPush(/*uuid=*/93));

  // When an incoming push fails (e.g. socket read timeout or peer disconnect),
  // EndIncomingPush receives the non-OK push status and immediately fails the
  // session, releasing the staging slot without waiting for the 120s deadline.
  EXPECT_THAT(
      consumer.base()->EndIncomingPush(
          /*uuid=*/93, absl::DeadlineExceededError("Socket read timed out")),
      ::absl_testing::StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(consumer.free_slots(), kSlots);

  // Subsequent incoming pushes for the same uuid are immediately rejected.
  EXPECT_THAT(consumer.base()->BeginIncomingPush(/*uuid=*/93),
              ::absl_testing::StatusIs(absl::StatusCode::kCancelled));

  // CompleteReadRaw immediately reports the request in failed_recving and
  // retires the session before the 120s deadline.
  Reports reports = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(reports), IsEmpty());
  EXPECT_THAT(FailedRecving(reports), ElementsAre("req_push_died"));
  EXPECT_FALSE(consumer.has_recv(93));
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

TEST(RecvWindowAccountingTest, UnitCountReadinessAndPerChunkH2dSubSpec) {
  RecvTestManager consumer(/*num_layers=*/2);
  const uint64_t uuid = 101;
  consumer.AddRecv("req_test_1", uuid, /*blocks_per_layer=*/2);
  auto session = consumer.GetRecvSession(uuid);
  ASSERT_NE(session, nullptr);

  CopySpec spec_c0_l0{{100}, {200}, {1}};
  CopySpec spec_c0_l1{{101}, {201}, {1}};
  CopySpec spec_c1_l0{{110}, {210}, {1}};
  CopySpec spec_c1_l1{{111}, {211}, {1}};
  std::vector<std::vector<CopySpec>> specs = {
      {spec_c0_l0, spec_c0_l1},
      {spec_c1_l0, spec_c1_l1},
  };
  ASSERT_THAT(session->SetChunkH2dSpecs(/*n_chunks=*/2, specs),
              ::absl_testing::IsOk());
  EXPECT_EQ(session->n_chunks(), 2);
  EXPECT_EQ(session->ExpectedPushesPerLayer(), 2);
  EXPECT_EQ(consumer.base()->ExpectedPushesPerLayer(uuid), 2);

  // Unit 0 across all layers.
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid), ::absl_testing::IsOk());
  consumer.FinishCopy(0, absl::OkStatus());
  ASSERT_THAT(consumer.ReceiveLayer(1, uuid), ::absl_testing::IsOk());
  consumer.FinishCopy(1, absl::OkStatus());

  EXPECT_EQ(session->units_complete(), 2);
  EXPECT_EQ(session->h2d_issued(), 2);
  EXPECT_EQ(session->h2d_ready(), 2);
  EXPECT_FALSE(session->IsReadyToComplete());
  EXPECT_FALSE(session->Done());

  // Unit 1 across all layers.
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid), ::absl_testing::IsOk());
  consumer.FinishCopy(2, absl::OkStatus());
  ASSERT_THAT(consumer.ReceiveLayer(1, uuid), ::absl_testing::IsOk());
  consumer.FinishCopy(3, absl::OkStatus());

  EXPECT_EQ(session->units_complete(), 4);
  EXPECT_EQ(session->h2d_issued(), 4);
  EXPECT_EQ(session->h2d_ready(), 4);
  EXPECT_TRUE(session->IsReadyToComplete());
  EXPECT_TRUE(session->Done());
  EXPECT_THAT(session->GetStatus(), ::absl_testing::IsOk());

  auto copies = consumer.issued_copies();
  ASSERT_EQ(copies.size(), 4);
  EXPECT_EQ(copies[0].src_offsets, spec_c0_l0.src_offsets);
  EXPECT_EQ(copies[0].dst_offsets, spec_c0_l0.dst_offsets);
  EXPECT_EQ(copies[1].src_offsets, spec_c0_l1.src_offsets);
  EXPECT_EQ(copies[1].dst_offsets, spec_c0_l1.dst_offsets);
  EXPECT_EQ(copies[2].src_offsets, spec_c1_l0.src_offsets);
  EXPECT_EQ(copies[2].dst_offsets, spec_c1_l0.dst_offsets);
  EXPECT_EQ(copies[3].src_offsets, spec_c1_l1.src_offsets);
  EXPECT_EQ(copies[3].dst_offsets, spec_c1_l1.dst_offsets);
}

TEST(RecvWindowAccountingTest, NetworkCompleteTrueBeforeLastH2dIssuedNotReady) {
  RecvTestManager consumer(/*num_layers=*/1);
  const uint64_t uuid = 102;
  consumer.AddRecv("req_test_2", uuid, /*blocks_per_layer=*/2);
  auto session = consumer.GetRecvSession(uuid);
  ASSERT_NE(session, nullptr);
  ASSERT_THAT(session->SetChunkH2dSpecs(/*n_chunks=*/2),
              ::absl_testing::IsOk());
  EXPECT_EQ(session->n_chunks(), 2);

  // Unit 0 arrives and finishes H2D.
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid), ::absl_testing::IsOk());
  consumer.FinishCopy(0, absl::OkStatus());
  EXPECT_EQ(session->units_complete(), 1);
  EXPECT_EQ(session->h2d_issued(), 1);
  EXPECT_EQ(session->h2d_ready(), 1);

  // Unit 1 arrives, but pause inside H2dSyncDispatch so units_complete_ is 2
  // while h2d_issued_ is still 1 and h2d_ready_ is 1.
  consumer.BlockH2dDispatch();
  auto receive_unit1 = std::async(std::launch::async, [&consumer] {
    return consumer.ReceiveLayer(0, uuid);
  });
  DispatchReleaseGuard release_guard(&consumer);
  ASSERT_TRUE(consumer.WaitForH2dDispatch(absl::Seconds(5)));

  // While paused inside H2dSyncDispatch, units_complete_ == total_units (2)
  // and h2d_ready_ == h2d_issued_ (1 == 1), isolating the h2d_issued_ ==
  // total_units clause.
  EXPECT_EQ(session->units_complete(), 2);
  EXPECT_EQ(session->h2d_issued(), 1);
  EXPECT_EQ(session->h2d_ready(), 1);

  // All blocks received on transport -> network_completed_ becomes true.
  ASSERT_THAT(consumer.ReceiveBlocks({0, 1}, uuid), ::absl_testing::IsOk());
  EXPECT_FALSE(session->IsReadyToComplete());
  EXPECT_FALSE(session->Done());

  // Release H2D dispatch and wait for unit 1's ReceiveLayer to return.
  release_guard.Release();
  ASSERT_EQ(receive_unit1.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  ASSERT_THAT(receive_unit1.get(), ::absl_testing::IsOk());
  EXPECT_EQ(session->units_complete(), 2);
  EXPECT_EQ(session->h2d_issued(), 2);
  EXPECT_EQ(session->h2d_ready(), 1);
  EXPECT_FALSE(session->IsReadyToComplete());
  EXPECT_FALSE(session->Done());

  // Finish unit 1's H2D -> now all units and H2D done.
  consumer.FinishCopy(1, absl::OkStatus());
  EXPECT_EQ(session->h2d_ready(), 2);
  EXPECT_TRUE(session->IsReadyToComplete());
  EXPECT_TRUE(session->Done());
  EXPECT_THAT(session->GetStatus(), ::absl_testing::IsOk());
}

TEST(RecvWindowAccountingTest, OutOfOrderUnitH2dCompletion) {
  RecvTestManager consumer(/*num_layers=*/1);
  const uint64_t uuid = 103;
  consumer.AddRecv("req_test_3", uuid, /*blocks_per_layer=*/2);
  auto session = consumer.GetRecvSession(uuid);
  ASSERT_NE(session, nullptr);
  ASSERT_THAT(session->SetChunkH2dSpecs(/*n_chunks=*/2),
              ::absl_testing::IsOk());

  // Issue unit 0 and unit 1 so both copies are in flight.
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid), ::absl_testing::IsOk());
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid), ::absl_testing::IsOk());
  EXPECT_EQ(session->units_complete(), 2);
  EXPECT_EQ(session->h2d_issued(), 2);
  EXPECT_EQ(session->h2d_ready(), 0);

  // Complete unit 1's copy first (out of order).
  consumer.FinishCopy(1, absl::OkStatus());
  EXPECT_EQ(session->h2d_ready(), 1);
  EXPECT_FALSE(session->IsReadyToComplete());
  EXPECT_FALSE(session->Done());

  // Complete unit 0's copy.
  consumer.FinishCopy(0, absl::OkStatus());
  EXPECT_EQ(session->h2d_ready(), 2);
  EXPECT_TRUE(session->IsReadyToComplete());
  EXPECT_TRUE(session->Done());
  EXPECT_THAT(session->GetStatus(), ::absl_testing::IsOk());
}

TEST(RecvWindowAccountingTest, LateH2dCallbackIsSafeNoOp) {
  RecvTestManager consumer(/*num_layers=*/1);
  const uint64_t uuid = 104;
  consumer.AddRecv("req_test_4", uuid, /*blocks_per_layer=*/1);
  auto session = consumer.GetRecvSession(uuid);
  ASSERT_NE(session, nullptr);

  // Issue H2D for layer 0.
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid), ::absl_testing::IsOk());
  EXPECT_EQ(consumer.copies_issued(), 1);

  // Fail/cancel the session while H2D is in flight.
  const absl::Status cancel_status =
      absl::CancelledError("cancelled during test");
  consumer.FailRecv(uuid, cancel_status);
  EXPECT_EQ(session->GetStatus().code(), absl::StatusCode::kCancelled);

  // Resolve the in-flight H2D future afterwards.
  consumer.FinishCopy(0, absl::OkStatus());

  // Terminal status is preserved and not overwritten; no crash.
  EXPECT_EQ(session->GetStatus().code(), absl::StatusCode::kCancelled);
  EXPECT_THAT(session->GetStatus().message(),
              ::testing::HasSubstr("cancelled during test"));
}

TEST(RecvWindowAccountingTest, WindowOffDefaultsToOnePushPerLayer) {
  RecvTestManager consumer(/*num_layers=*/1);
  const uint64_t uuid = 105;
  consumer.AddRecv("req_test_5", uuid, /*blocks_per_layer=*/1);
  auto session = consumer.GetRecvSession(uuid);
  ASSERT_NE(session, nullptr);

  EXPECT_EQ(session->ExpectedPushesPerLayer(), 1);
  EXPECT_EQ(consumer.base()->ExpectedPushesPerLayer(uuid), 1);
  EXPECT_EQ(consumer.base()->ExpectedPushesPerLayer(0), 1);
  EXPECT_EQ(consumer.base()->ExpectedPushesPerLayer(99999), 1);
}

TEST(RecvWindowAccountingTest, ExtraReceiveLayerBeyondNChunksFails) {
  RecvTestManager consumer(/*num_layers=*/2);
  const uint64_t uuid = 106;
  consumer.AddRecv("req_test_extra", uuid, /*blocks_per_layer=*/1);
  auto session = consumer.GetRecvSession(uuid);
  ASSERT_NE(session, nullptr);
  ASSERT_THAT(session->SetChunkH2dSpecs(/*n_chunks=*/1),
              ::absl_testing::IsOk());
  EXPECT_EQ(session->n_chunks(), 1);

  // First unit succeeds.
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid), ::absl_testing::IsOk());
  consumer.FinishCopy(0, absl::OkStatus());

  // Session remains live because layer 1 has not completed yet.
  EXPECT_FALSE(session->Done());

  // Extra ReceiveLayer call beyond n_chunks_ (1) fails with
  // kFailedPrecondition.
  absl::Status status = consumer.ReceiveLayer(0, uuid);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("already received all 1 units"));
  EXPECT_TRUE(session->Done());
  EXPECT_EQ(session->GetStatus().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(RecvWindowAccountingTest, SetChunkH2dSpecsValidationAndMidFlight) {
  RecvTestManager consumer(/*num_layers=*/2);
  const uint64_t uuid = 107;
  consumer.AddRecv("req_test_specs", uuid, /*blocks_per_layer=*/2);
  auto session = consumer.GetRecvSession(uuid);
  ASSERT_NE(session, nullptr);

  // n_chunks < 1 returns kInvalidArgument.
  EXPECT_EQ(session->SetChunkH2dSpecs(/*n_chunks=*/0).code(),
            absl::StatusCode::kInvalidArgument);

  // Outer dimension mismatch: n_chunks = 2, but specs has 1 chunk.
  CopySpec spec{{100}, {200}, {1}};
  EXPECT_EQ(session->SetChunkH2dSpecs(/*n_chunks=*/2, {{spec, spec}}).code(),
            absl::StatusCode::kInvalidArgument);

  // Inner dimension mismatch: n_chunks = 2, chunk 0 has 2 layers, but chunk 1
  // has 1 layer (expected 2).
  EXPECT_EQ(
      session->SetChunkH2dSpecs(/*n_chunks=*/2, {{spec, spec}, {spec}}).code(),
      absl::StatusCode::kInvalidArgument);

  // Valid specs config succeeds.
  std::vector<std::vector<CopySpec>> valid_specs = {
      {spec, spec},
      {spec, spec},
  };
  ASSERT_THAT(session->SetChunkH2dSpecs(/*n_chunks=*/2, valid_specs),
              ::absl_testing::IsOk());
  EXPECT_EQ(session->n_chunks(), 2);

  // Start receiving a unit (mid-flight).
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid), ::absl_testing::IsOk());

  // Mutating chunk config mid-flight (units_complete_ > 0 || h2d_issued_ > 0)
  // returns kFailedPrecondition.
  EXPECT_EQ(session->SetChunkH2dSpecs(/*n_chunks=*/2, valid_specs).code(),
            absl::StatusCode::kFailedPrecondition);

  consumer.FinishCopy(0, absl::OkStatus());
}

TEST(RecvWindowGateTest, ShardGroupG2AdmittedAndClaimResetOnUnitComplete) {
  constexpr size_t kNumLayers = 1;
  constexpr size_t kNumShards = 2;
  constexpr size_t kNumSlots = 64;
  RecvTestManager consumer(
      kNumLayers, /*timeout_s=*/5.0, kNumSlots, kNumShards,
      StagingWindowConfig{.window_chunks = 2, .chunk_blocks = 2});
  const uint64_t uuid = 201;

  // Windowed session: k = 4, C = 2, W = 2 -> 2 chunks.
  const std::vector<int64_t> remote_bids = {0, 1, 2, 3};
  const std::vector<int64_t> local_bids = {10, 11, 12, 13};
  ASSERT_THAT(consumer.AddWindowedRecv(
                  "req_g2", uuid, remote_bids, local_bids,
                  StagingWindowConfig{.window_chunks = 2, .chunk_blocks = 2}),
              ::absl_testing::IsOk());
  auto session = consumer.GetRecvSession(uuid);
  ASSERT_NE(session, nullptr);
  EXPECT_TRUE(session->IsWindowed());

  const std::vector<int64_t> transport_host_bids =
      session->transport_host_block_ids();
  ASSERT_EQ(transport_host_bids.size(), 4);

  // Unit 0: chunk 0 maps to slot 0 -> blocks {transport_host_bids[0],
  // transport_host_bids[1]}.
  const std::vector<int64_t> chunk0_bids = {transport_host_bids[0],
                                            transport_host_bids[1]};
  const absl::Time deadline = absl::InfiniteFuture();

  // Stream 0: shard 0 in G=2. Admitted.
  EXPECT_THAT(
      consumer.base()->AwaitIncomingPushSlot(uuid, /*layer_idx=*/0, chunk0_bids,
                                             /*stream_shards=*/{0}, deadline),
      ::absl_testing::IsOk());

  // Stream 1: shard 1 in G=2. Disjoint shard, identical dst_ids. Admitted.
  EXPECT_THAT(
      consumer.base()->AwaitIncomingPushSlot(uuid, /*layer_idx=*/0, chunk0_bids,
                                             /*stream_shards=*/{1}, deadline),
      ::absl_testing::IsOk());

  // Duplicate claim within unit 0 for stream 0 (shard 0) is rejected.
  absl::Status dup_status = consumer.base()->AwaitIncomingPushSlot(
      uuid, /*layer_idx=*/0, chunk0_bids, /*stream_shards=*/{0}, deadline);
  EXPECT_EQ(dup_status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(
      dup_status.message(),
      ::testing::HasSubstr("re-claims staging (position, shard) pairs"));
  EXPECT_THAT(dup_status.message(),
              ::testing::HasSubstr("peer window mismatch"));
  EXPECT_TRUE(session->Done());
  EXPECT_EQ(session->Details().phase, "gate");

  // Reset/test unit 1 on a fresh session: verify claim set resets on unit
  // completion.
  const uint64_t uuid2 = 202;
  ASSERT_THAT(consumer.AddWindowedRecv(
                  "req_g2_unit1", uuid2, remote_bids, local_bids,
                  StagingWindowConfig{.window_chunks = 2, .chunk_blocks = 2}),
              ::absl_testing::IsOk());
  auto session2 = consumer.GetRecvSession(uuid2);
  ASSERT_NE(session2, nullptr);

  const std::vector<int64_t> host_bids2 = session2->transport_host_block_ids();
  const std::vector<int64_t> chunk0_bids2 = {host_bids2[0], host_bids2[1]};
  const std::vector<int64_t> chunk1_bids2 = {host_bids2[2], host_bids2[3]};

  // Unit 0 admissions for G=2.
  ASSERT_THAT(consumer.base()->AwaitIncomingPushSlot(
                  uuid2, /*layer_idx=*/0, chunk0_bids2, {0}, deadline),
              ::absl_testing::IsOk());
  ASSERT_THAT(consumer.base()->AwaitIncomingPushSlot(
                  uuid2, /*layer_idx=*/0, chunk0_bids2, {1}, deadline),
              ::absl_testing::IsOk());

  // Complete unit 0.
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid2), ::absl_testing::IsOk());
  consumer.FinishCopy(consumer.copies_issued() - 1, absl::OkStatus());

  // Unit 1: claim set has reset; stream 0 and 1 admitted for chunk 1.
  EXPECT_THAT(consumer.base()->AwaitIncomingPushSlot(
                  uuid2, /*layer_idx=*/0, chunk1_bids2, {0}, deadline),
              ::absl_testing::IsOk());
  EXPECT_THAT(consumer.base()->AwaitIncomingPushSlot(
                  uuid2, /*layer_idx=*/0, chunk1_bids2, {1}, deadline),
              ::absl_testing::IsOk());

  // Duplicate for unit 1 rejected.
  absl::Status dup1 = consumer.base()->AwaitIncomingPushSlot(
      uuid2, /*layer_idx=*/0, chunk1_bids2, {0}, deadline);
  EXPECT_EQ(dup1.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(dup1.message(), ::testing::HasSubstr(
                                  "re-claims staging (position, shard) pairs"));
  EXPECT_TRUE(session2->Done());
  EXPECT_EQ(session2->Details().phase, "gate");
}

TEST(RecvWindowGateTest, IntraStreamDuplicatePositionClaimRejected) {
  constexpr size_t kNumLayers = 1;
  constexpr size_t kNumShards = 1;
  constexpr size_t kNumSlots = 64;
  RecvTestManager consumer(
      kNumLayers, /*timeout_s=*/5.0, kNumSlots, kNumShards,
      StagingWindowConfig{.window_chunks = 2, .chunk_blocks = 2});
  const uint64_t uuid = 203;

  // Windowed session: k = 4, C = 2, W = 2 -> 2 chunks.
  const std::vector<int64_t> remote_bids = {0, 1, 2, 3};
  const std::vector<int64_t> local_bids = {10, 11, 12, 13};
  ASSERT_THAT(consumer.AddWindowedRecv(
                  "req_intra_dup", uuid, remote_bids, local_bids,
                  StagingWindowConfig{.window_chunks = 2, .chunk_blocks = 2}),
              ::absl_testing::IsOk());
  auto session = consumer.GetRecvSession(uuid);
  ASSERT_NE(session, nullptr);
  EXPECT_TRUE(session->IsWindowed());

  const std::vector<int64_t> host_bids = session->transport_host_block_ids();
  ASSERT_EQ(host_bids.size(), 4);

  // The first stream of the unit passes duplicate destination block IDs
  // belonging to ring[0] with |dst_ids| <= C_c ({host_bids[0], host_bids[0]}).
  // This exercises the intra-stream duplicate claim branch
  // (!new_claims.insert(key).second) at transfer_receive_session.cc:760.
  const std::vector<int64_t> duplicate_dst_bids = {host_bids[0], host_bids[0]};
  const absl::Time deadline = absl::InfiniteFuture();

  absl::Status status = consumer.base()->AwaitIncomingPushSlot(
      uuid, /*layer_idx=*/0, duplicate_dst_bids, /*stream_shards=*/{0},
      deadline);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(
      status.message(),
      ::testing::HasSubstr("re-claims staging (position, shard) pairs"));
  EXPECT_TRUE(session->Done());
  EXPECT_EQ(session->Details().phase, "gate");
}

TEST(RecvWindowGateTest, LegacySenderFailSafeW1C64K256) {
  constexpr size_t kNumLayers = 1;
  constexpr size_t kNumSlots = 512;
  const absl::Time deadline = absl::InfiniteFuture();

  for (int p_parallelism : {4, 2, 1}) {
    for (int g_shards : {1, 2}) {
      RecvTestManager consumer(
          kNumLayers, /*timeout_s=*/5.0, kNumSlots,
          /*num_shards=*/g_shards,
          StagingWindowConfig{.window_chunks = 1, .chunk_blocks = 64});
      const uint64_t uuid = 300 + p_parallelism * 10 + g_shards;
      std::vector<int64_t> remote_bids(256);
      std::vector<int64_t> local_bids(256);
      std::iota(remote_bids.begin(), remote_bids.end(), 0);
      std::iota(local_bids.begin(), local_bids.end(), 1000);

      ASSERT_THAT(
          consumer.AddWindowedRecv(
              absl::StrCat("req_fail_safe_w1_p", p_parallelism, "_g", g_shards),
              uuid, remote_bids, local_bids,
              StagingWindowConfig{.window_chunks = 1, .chunk_blocks = 64}),
          ::absl_testing::IsOk());
      auto session = consumer.GetRecvSession(uuid);
      ASSERT_NE(session, nullptr);
      const std::vector<int64_t> host_bids =
          session->transport_host_block_ids();
      ASSERT_EQ(host_bids.size(), 256);

      const size_t stream_size = 256 / p_parallelism;
      if (p_parallelism == 4) {
        // stream_size = 64 == C_0.
        // Stream 0 (or G streams for chunk 0) admitted.
        const std::vector<int64_t> chunk0_bids(host_bids.begin(),
                                               host_bids.begin() + 64);
        for (int g = 0; g < g_shards; ++g) {
          EXPECT_THAT(consumer.base()->AwaitIncomingPushSlot(
                          uuid, /*layer_idx=*/0, chunk0_bids, {g}, deadline),
                      ::absl_testing::IsOk())
              << "Failed for P=" << p_parallelism << " G=" << g_shards
              << " shard=" << g;
        }

        // Remaining streams (e.g. stream for chunk 1: host_bids[64..127]) are
        // rejected because the receiver is still at unit 0.
        const std::vector<int64_t> chunk1_bids(host_bids.begin() + 64,
                                               host_bids.begin() + 128);
        absl::Status rej_status = consumer.base()->AwaitIncomingPushSlot(
            uuid, /*layer_idx=*/0, chunk1_bids, {0}, deadline);
        EXPECT_EQ(rej_status.code(), absl::StatusCode::kFailedPrecondition);
        EXPECT_THAT(rej_status.message(),
                    ::testing::HasSubstr("peer window mismatch"));
        EXPECT_TRUE(session->Done());
        EXPECT_EQ(session->Details().phase, "gate");
        EXPECT_EQ(consumer.copies_issued(), 0);
      } else {
        // P in {2, 1}: stream_size > C_0 (128 or 256 > 64). Zero streams
        // admitted.
        const std::vector<int64_t> oversized_bids(
            host_bids.begin(), host_bids.begin() + stream_size);
        absl::Status rej_status = consumer.base()->AwaitIncomingPushSlot(
            uuid, /*layer_idx=*/0, oversized_bids, {0}, deadline);
        EXPECT_EQ(rej_status.code(), absl::StatusCode::kFailedPrecondition);
        EXPECT_THAT(rej_status.message(),
                    ::testing::HasSubstr("peer window mismatch"));
        EXPECT_TRUE(session->Done());
        EXPECT_EQ(session->Details().phase, "gate");
        EXPECT_EQ(consumer.copies_issued(), 0);
      }
    }
  }
}

TEST(RecvWindowGateTest, LegacySenderFailSafeW2C64K256) {
  constexpr size_t kNumLayers = 1;
  constexpr size_t kNumSlots = 512;
  const absl::Time deadline = absl::InfiniteFuture();
  RecvTestManager consumer(
      kNumLayers, /*timeout_s=*/5.0, kNumSlots,
      /*num_shards=*/1,
      StagingWindowConfig{.window_chunks = 2, .chunk_blocks = 64});
  const uint64_t uuid = 401;

  std::vector<int64_t> remote_bids(256);
  std::vector<int64_t> local_bids(256);
  std::iota(remote_bids.begin(), remote_bids.end(), 0);
  std::iota(local_bids.begin(), local_bids.end(), 1000);

  ASSERT_THAT(consumer.AddWindowedRecv(
                  "req_fail_safe_w2", uuid, remote_bids, local_bids,
                  StagingWindowConfig{.window_chunks = 2, .chunk_blocks = 64}),
              ::absl_testing::IsOk());
  auto session = consumer.GetRecvSession(uuid);
  ASSERT_NE(session, nullptr);
  const std::vector<int64_t> host_bids = session->transport_host_block_ids();
  ASSERT_EQ(host_bids.size(), 256);

  // Stream 0: chunk 0 in slot 0 (blocks 0..63) -> admitted.
  const std::vector<int64_t> stream0_bids(host_bids.begin(),
                                          host_bids.begin() + 64);
  EXPECT_THAT(consumer.base()->AwaitIncomingPushSlot(
                  uuid, /*layer_idx=*/0, stream0_bids, {0}, deadline),
              ::absl_testing::IsOk());

  // Stream 1: chunk 1 in slot 1 (blocks 64..127).
  // Rejected because receiver is still at unit 0 (unit 0 H2D has not
  // completed).
  const std::vector<int64_t> stream1_bids(host_bids.begin() + 64,
                                          host_bids.begin() + 128);
  absl::Status status = consumer.base()->AwaitIncomingPushSlot(
      uuid, /*layer_idx=*/0, stream1_bids, {0}, deadline);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("peer window mismatch"));
  EXPECT_TRUE(session->Done());
  EXPECT_EQ(session->Details().phase, "gate");
}

TEST(RecvWindowGateTest, HookNoOpForUuidZeroPlanAndPoolReshard) {
  RecvTestManager consumer(/*num_layers=*/1);
  const absl::Time deadline = absl::InfiniteFuture();
  const std::vector<int64_t> arbitrary_bids = {123, 456};
  const std::vector<int32_t> shards = {0};

  // 1. uuid == 0: always returns OkStatus().
  EXPECT_THAT(
      consumer.base()->AwaitIncomingPushSlot(
          /*uuid=*/0, /*layer_idx=*/0, arbitrary_bids, shards, deadline),
      ::absl_testing::IsOk());

  // 2. Unknown uuid (no session registered): returns OkStatus().
  EXPECT_THAT(
      consumer.base()->AwaitIncomingPushSlot(
          /*uuid=*/99999, /*layer_idx=*/0, arbitrary_bids, shards, deadline),
      ::absl_testing::IsOk());

  // 3. Active-plan push: returns OkStatus().
  constexpr uint64_t kPlanUuid = 777;
  ::tpu_sync::rpc::StartTransferRequest plan;
  plan.set_uuid(kPlanUuid);
  plan.set_dst_mem_type(::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  ASSERT_THAT(consumer.RegisterActivePlan(kPlanUuid, plan, /*is_sender=*/false),
              ::absl_testing::IsOk());
  EXPECT_THAT(consumer.base()->AwaitIncomingPushSlot(
                  kPlanUuid, /*layer_idx=*/0, arbitrary_bids, shards, deadline),
              ::absl_testing::IsOk());
  ASSERT_THAT(consumer.UnregisterActivePlan(kPlanUuid), ::absl_testing::IsOk());

  // 4. Pool-reshard push: returns OkStatus().
  constexpr uint64_t kPoolUuid = 888;
  consumer.InsertPoolReshardRecv(kPoolUuid);
  EXPECT_THAT(consumer.base()->AwaitIncomingPushSlot(
                  kPoolUuid, /*layer_idx=*/0, arbitrary_bids, shards, deadline),
              ::absl_testing::IsOk());
  consumer.ErasePoolReshardRecv(kPoolUuid);
}

TEST(RecvWindowGateTest, ReceiverGateBlocksUntilH2dDone) {
  constexpr size_t kNumLayers = 1;
  constexpr size_t kNumShards = 1;
  constexpr size_t kNumSlots = 64;
  RecvTestManager consumer(
      kNumLayers, /*timeout_s=*/5.0, kNumSlots, kNumShards,
      StagingWindowConfig{.window_chunks = 2, .chunk_blocks = 2});
  const uint64_t uuid = 701;

  // Windowed session: k = 7, C = 2, W = 2 -> 4 chunks. Ring = 4 blocks.
  const std::vector<int64_t> remote_bids = {0, 1, 2, 3, 4, 5, 6};
  const std::vector<int64_t> local_bids = {10, 11, 12, 13, 14, 15, 16};
  ASSERT_THAT(consumer.AddWindowedRecv(
                  "req_gate_block", uuid, remote_bids, local_bids,
                  StagingWindowConfig{.window_chunks = 2, .chunk_blocks = 2}),
              ::absl_testing::IsOk());
  auto session = consumer.GetRecvSession(uuid);
  ASSERT_NE(session, nullptr);
  EXPECT_TRUE(session->IsWindowed());

  const std::vector<int64_t> host_bids = session->transport_host_block_ids();
  ASSERT_EQ(host_bids.size(), 7);

  // Chunk 0: slot 0 (blocks host_bids[0], host_bids[1])
  // Chunk 1: slot 1 (blocks host_bids[2], host_bids[3])
  // Chunk 2: slot 0 (blocks host_bids[0], host_bids[1])
  const std::vector<int64_t> chunk0_bids = {host_bids[0], host_bids[1]};
  const std::vector<int64_t> chunk1_bids = {host_bids[2], host_bids[3]};
  const std::vector<int64_t> chunk2_bids = {host_bids[0], host_bids[1]};
  const absl::Time deadline = absl::InfiniteFuture();

  // Chunk 0 passes gate without waiting (needed_done = 0).
  EXPECT_THAT(consumer.base()->AwaitIncomingPushSlot(
                  uuid, /*layer_idx=*/0, chunk0_bids, {0}, deadline),
              ::absl_testing::IsOk());
  // Land and complete unit 0: issues H2D for chunk 0 into copies_[0].
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid), ::absl_testing::IsOk());

  // Chunk 1 passes gate without waiting (slot 1, needed_done = 0).
  EXPECT_THAT(consumer.base()->AwaitIncomingPushSlot(
                  uuid, /*layer_idx=*/0, chunk1_bids, {0}, deadline),
              ::absl_testing::IsOk());
  // Land and complete unit 1: advances recv_gen_[0] to 2 so chunk 2 can be
  // gated.
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid), ::absl_testing::IsOk());

  // Chunk 2 requires h2d_done_[0][0] >= 1. Currently 0, so it must block.
  std::atomic<bool> gate_returned{false};
  absl::Status gate_status;
  std::thread t([&]() {
    gate_status = consumer.base()->AwaitIncomingPushSlot(
        uuid, /*layer_idx=*/0, chunk2_bids, {0}, deadline);
    gate_returned.store(true);
  });

  // Verify gate is blocked.
  absl::SleepFor(absl::Milliseconds(50));
  EXPECT_FALSE(gate_returned.load());

  // Complete chunk 0 H2D: FinishCopy(0).
  consumer.FinishCopy(0, absl::OkStatus());

  t.join();
  EXPECT_TRUE(gate_returned.load());
  EXPECT_THAT(gate_status, ::absl_testing::IsOk());
  EXPECT_EQ(session->h2d_done(0, 0), 1);
  consumer.FinishCopy(1, absl::OkStatus());
}

TEST(RecvWindowGateTest, ReceiverGateCancelledWhenDraining) {
  constexpr size_t kNumLayers = 1;
  constexpr size_t kNumShards = 1;
  constexpr size_t kNumSlots = 64;
  RecvTestManager consumer(
      kNumLayers, /*timeout_s=*/5.0, kNumSlots, kNumShards,
      StagingWindowConfig{.window_chunks = 2, .chunk_blocks = 2});
  const uint64_t uuid = 702;

  const std::vector<int64_t> remote_bids = {0, 1, 2, 3, 4, 5, 6};
  const std::vector<int64_t> local_bids = {10, 11, 12, 13, 14, 15, 16};
  ASSERT_THAT(consumer.AddWindowedRecv(
                  "req_gate_cancel", uuid, remote_bids, local_bids,
                  StagingWindowConfig{.window_chunks = 2, .chunk_blocks = 2}),
              ::absl_testing::IsOk());
  auto session = consumer.GetRecvSession(uuid);
  ASSERT_NE(session, nullptr);

  const std::vector<int64_t> host_bids = session->transport_host_block_ids();
  const std::vector<int64_t> chunk0_bids = {host_bids[0], host_bids[1]};
  const std::vector<int64_t> chunk1_bids = {host_bids[2], host_bids[3]};
  const std::vector<int64_t> chunk2_bids = {host_bids[0], host_bids[1]};
  const absl::Time deadline = absl::InfiniteFuture();

  ASSERT_THAT(consumer.base()->AwaitIncomingPushSlot(
                  uuid, /*layer_idx=*/0, chunk0_bids, {0}, deadline),
              ::absl_testing::IsOk());
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid), ::absl_testing::IsOk());

  ASSERT_THAT(consumer.base()->AwaitIncomingPushSlot(
                  uuid, /*layer_idx=*/0, chunk1_bids, {0}, deadline),
              ::absl_testing::IsOk());
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid), ::absl_testing::IsOk());

  // Chunk 2 blocks at gate.
  std::atomic<bool> gate_returned{false};
  absl::Status gate_status;
  std::thread t([&]() {
    gate_status = consumer.base()->AwaitIncomingPushSlot(
        uuid, /*layer_idx=*/0, chunk2_bids, {0}, deadline);
    gate_returned.store(true);
  });

  absl::SleepFor(absl::Milliseconds(50));
  EXPECT_FALSE(gate_returned.load());

  // Cancel / drain session while chunk 2 is waiting at the gate.
  consumer.FailRecv(uuid, absl::CancelledError("cancelled during test"));

  t.join();
  EXPECT_TRUE(gate_returned.load());
  EXPECT_EQ(gate_status.code(), absl::StatusCode::kCancelled);

  // In-flight H2D copies from chunks 0 and 1 pin staging; Done() remains false.
  EXPECT_FALSE(session->Done());
  consumer.FinishCopy(0, absl::OkStatus());
  EXPECT_FALSE(session->Done());
  consumer.FinishCopy(1, absl::OkStatus());
  EXPECT_TRUE(session->Done());

  // Originating failure was the external cancel; first error wins per Q7,
  // so phase reflects the cancel time ("connect"), not the gate waiter wake.
  EXPECT_EQ(session->Details().phase, "connect");
}

TEST(RecvWindowGateTest, ReceiverGateTimeoutPhaseGate) {
  constexpr size_t kNumLayers = 1;
  constexpr size_t kNumShards = 1;
  constexpr size_t kNumSlots = 64;
  RecvTestManager consumer(
      kNumLayers, /*timeout_s=*/5.0, kNumSlots, kNumShards,
      StagingWindowConfig{.window_chunks = 2, .chunk_blocks = 2});
  const uint64_t uuid = 703;

  const std::vector<int64_t> remote_bids = {0, 1, 2, 3, 4, 5, 6};
  const std::vector<int64_t> local_bids = {10, 11, 12, 13, 14, 15, 16};
  ASSERT_THAT(consumer.AddWindowedRecv(
                  "req_gate_timeout", uuid, remote_bids, local_bids,
                  StagingWindowConfig{.window_chunks = 2, .chunk_blocks = 2}),
              ::absl_testing::IsOk());
  auto session = consumer.GetRecvSession(uuid);
  ASSERT_NE(session, nullptr);

  const std::vector<int64_t> host_bids = session->transport_host_block_ids();
  const std::vector<int64_t> chunk0_bids = {host_bids[0], host_bids[1]};
  const std::vector<int64_t> chunk1_bids = {host_bids[2], host_bids[3]};
  const std::vector<int64_t> chunk2_bids = {host_bids[0], host_bids[1]};
  const absl::Time deadline = absl::InfiniteFuture();

  ASSERT_THAT(consumer.base()->AwaitIncomingPushSlot(
                  uuid, /*layer_idx=*/0, chunk0_bids, {0}, deadline),
              ::absl_testing::IsOk());
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid), ::absl_testing::IsOk());

  ASSERT_THAT(consumer.base()->AwaitIncomingPushSlot(
                  uuid, /*layer_idx=*/0, chunk1_bids, {0}, deadline),
              ::absl_testing::IsOk());
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid), ::absl_testing::IsOk());

  // Chunk 2 arrives with a short 50ms deadline.
  const absl::Time short_deadline = absl::Now() + absl::Milliseconds(50);
  absl::Status status = consumer.base()->AwaitIncomingPushSlot(
      uuid, /*layer_idx=*/0, chunk2_bids, {0}, short_deadline);
  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_EQ(session->Details().phase, "gate");

  // Outstanding H2D copies for chunks 0 and 1 keep Done() false until drained.
  EXPECT_FALSE(session->Done());
  consumer.FinishCopy(0, absl::OkStatus());
  EXPECT_FALSE(session->Done());
  consumer.FinishCopy(1, absl::OkStatus());
  EXPECT_TRUE(session->Done());
}

TEST(RecvWindowGateTest, ReceiverGateWaitOver1msCounted) {
  constexpr size_t kNumLayers = 1;
  constexpr size_t kNumShards = 1;
  constexpr size_t kNumSlots = 64;
  RecvTestManager consumer(
      kNumLayers, /*timeout_s=*/5.0, kNumSlots, kNumShards,
      StagingWindowConfig{.window_chunks = 2, .chunk_blocks = 2});
  const uint64_t uuid = 704;

  const std::vector<int64_t> remote_bids = {0, 1, 2, 3, 4, 5, 6};
  const std::vector<int64_t> local_bids = {10, 11, 12, 13, 14, 15, 16};
  ASSERT_THAT(consumer.AddWindowedRecv(
                  "req_gate_wait_count", uuid, remote_bids, local_bids,
                  StagingWindowConfig{.window_chunks = 2, .chunk_blocks = 2}),
              ::absl_testing::IsOk());
  auto session = consumer.GetRecvSession(uuid);
  ASSERT_NE(session, nullptr);

  const std::vector<int64_t> host_bids = session->transport_host_block_ids();
  const std::vector<int64_t> chunk0_bids = {host_bids[0], host_bids[1]};
  const std::vector<int64_t> chunk1_bids = {host_bids[2], host_bids[3]};
  const std::vector<int64_t> chunk2_bids = {host_bids[0], host_bids[1]};
  const absl::Time deadline = absl::InfiniteFuture();

  ASSERT_THAT(consumer.base()->AwaitIncomingPushSlot(
                  uuid, /*layer_idx=*/0, chunk0_bids, {0}, deadline),
              ::absl_testing::IsOk());
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid), ::absl_testing::IsOk());

  ASSERT_THAT(consumer.base()->AwaitIncomingPushSlot(
                  uuid, /*layer_idx=*/0, chunk1_bids, {0}, deadline),
              ::absl_testing::IsOk());
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid), ::absl_testing::IsOk());

  // Chunk 2 waits at the gate. Sleep for > 1ms (10ms) before completing H2D.
  std::atomic<bool> gate_returned{false};
  absl::Status gate_status;
  std::thread t([&]() {
    gate_status = consumer.base()->AwaitIncomingPushSlot(
        uuid, /*layer_idx=*/0, chunk2_bids, {0}, deadline);
    gate_returned.store(true);
  });

  absl::SleepFor(absl::Milliseconds(10));
  EXPECT_FALSE(gate_returned.load());

  consumer.FinishCopy(0, absl::OkStatus());

  t.join();
  EXPECT_TRUE(gate_returned.load());
  EXPECT_THAT(gate_status, ::absl_testing::IsOk());
  EXPECT_GE(session->gate_waits_over_1ms(), 1);
  EXPECT_EQ(session->h2d_done(0, 0), 1);
  consumer.FinishCopy(1, absl::OkStatus());
}

TEST(RecvWindowGateTest, ReceiverGateLockOrderWithPollAndH2dCompletion) {
  constexpr size_t kNumLayers = 1;
  constexpr size_t kNumShards = 1;
  constexpr size_t kNumSlots = 64;
  RecvTestManager consumer(
      kNumLayers, /*timeout_s=*/5.0, kNumSlots, kNumShards,
      StagingWindowConfig{.window_chunks = 2, .chunk_blocks = 2});
  const uint64_t uuid = 705;

  const std::vector<int64_t> remote_bids = {0, 1, 2, 3, 4, 5, 6};
  const std::vector<int64_t> local_bids = {10, 11, 12, 13, 14, 15, 16};
  ASSERT_THAT(consumer.AddWindowedRecv(
                  "req_gate_lock_order", uuid, remote_bids, local_bids,
                  StagingWindowConfig{.window_chunks = 2, .chunk_blocks = 2}),
              ::absl_testing::IsOk());
  auto session = consumer.GetRecvSession(uuid);
  ASSERT_NE(session, nullptr);

  const std::vector<int64_t> host_bids = session->transport_host_block_ids();
  const std::vector<int64_t> chunk0_bids = {host_bids[0], host_bids[1]};
  const std::vector<int64_t> chunk1_bids = {host_bids[2], host_bids[3]};
  const std::vector<int64_t> chunk2_bids = {host_bids[0], host_bids[1]};
  const absl::Time deadline = absl::InfiniteFuture();

  ASSERT_THAT(consumer.base()->AwaitIncomingPushSlot(
                  uuid, /*layer_idx=*/0, chunk0_bids, {0}, deadline),
              ::absl_testing::IsOk());
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid), ::absl_testing::IsOk());

  ASSERT_THAT(consumer.base()->AwaitIncomingPushSlot(
                  uuid, /*layer_idx=*/0, chunk1_bids, {0}, deadline),
              ::absl_testing::IsOk());
  ASSERT_THAT(consumer.ReceiveLayer(0, uuid), ::absl_testing::IsOk());

  std::atomic<bool> stop_polling{false};
  // Thread 1: polls Details(), Done(), h2d_done(), gate_waits_over_1ms().
  std::thread poller([&]() {
    while (!stop_polling.load(std::memory_order_relaxed)) {
      (void)session->Details();
      (void)session->Done();
      (void)session->h2d_done(0, 0);
      (void)session->gate_waits_over_1ms();
      (void)session->last_unit_progress();
      absl::SleepFor(absl::Microseconds(100));
    }
  });

  // Thread 2: enters AwaitIncomingPushSlot and blocks on gate.
  std::atomic<bool> gate_returned{false};
  absl::Status gate_status;
  std::thread waiter([&]() {
    gate_status = consumer.base()->AwaitIncomingPushSlot(
        uuid, /*layer_idx=*/0, chunk2_bids, {0}, deadline);
    gate_returned.store(true);
  });

  absl::SleepFor(absl::Milliseconds(10));
  EXPECT_FALSE(gate_returned.load());

  // Thread 3: finishes H2D completion callback for chunk 0.
  consumer.FinishCopy(0, absl::OkStatus());

  waiter.join();
  EXPECT_TRUE(gate_returned.load());
  EXPECT_THAT(gate_status, ::absl_testing::IsOk());

  stop_polling.store(true, std::memory_order_relaxed);
  poller.join();

  EXPECT_EQ(session->h2d_done(0, 0), 1);
  consumer.FinishCopy(1, absl::OkStatus());
}

}  // namespace
}  // namespace tpu_raiden
