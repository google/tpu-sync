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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "xla/future.h"

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
  explicit TestManager(size_t num_layers)
      : KVCacheManagerWithTransfer(num_layers, /*num_shards=*/1,
                                   /*slice_byte_size=*/128,
                                   /*local_port=*/std::nullopt,
                                   /*host_blocks_to_allocate=*/std::nullopt,
                                   /*parallelism=*/1, /*node_id=*/0,
                                   /*local_control_port=*/-1, /*max_blocks=*/1,
                                   /*num_slots=*/kSlots, kTimeoutS) {
    CHECK_OK(ConfigureHostStagingSlots(kSlots, /*max_major_per_slot=*/1));
    CHECK_OK(InitializeSlotPool(kSlots));
  }

  // Serves a pull for `uuid` the way ProcessPullStream does once the
  // consumer is acknowledged: the push runs on this thread and returns
  // with its copies issued.
  void ServePull(uint64_t uuid) {
    {
      absl::MutexLock lock(mu_);
      send_entries_.at(uuid)->pull_started = true;
    }
    StartPushInternal(uuid, {"127.0.0.1:1"}, /*src_block_ids=*/{0},
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

  size_t free_slots() {
    absl::MutexLock lock(mu_);
    return free_slots_.size();
  }

  void ExpireSend(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    send_entries_.at(uuid)->deadline =
        std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
  }

  // White-box construction for the send retirement state machine. Tests that
  // need registration and dispatch use NotifyForRead and ServePull instead.
  std::shared_ptr<SendEntry> AddSyntheticSend(const std::string& req_id,
                                              uint64_t uuid, int in_flight) {
    absl::MutexLock lock(mu_);
    auto entry = std::make_shared<SendEntry>();
    entry->req_id = req_id;
    entry->uuid = uuid;
    entry->slot_idx = AcquireSlotLocked().slot_idx;
    entry->in_flight = in_flight;
    send_entries_[uuid] = entry;
    return entry;
  }

  void Decide(const std::shared_ptr<SendEntry>& entry, bool failed) {
    absl::MutexLock lock(mu_);
    FinishSendLocked(entry, failed);
  }

  void End(const std::shared_ptr<SendEntry>& entry) {
    absl::MutexLock lock(mu_);
    EndSendOpLocked(entry);
  }

  bool has_send(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    return send_entries_.contains(uuid);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> D2hSyncDispatch(
      const std::vector<int64_t>& src_offsets_major_dim,
      const std::vector<int64_t>& dst_offsets_major_dim,
      const std::vector<int64_t>& copy_sizes_major_dim,
      std::optional<int64_t> slot_idx, std::optional<size_t> layer_idx,
      std::optional<size_t> shard_idx) override {
    auto [promise, future] = xla::MakePromise<>();
    absl::MutexLock lock(copies_mu_);
    copies_.push_back(std::move(promise));
    return raiden::PjRtCopyFuture(std::move(future), raiden::BufferHolders{});
  }

 private:
  absl::Mutex copies_mu_;
  std::vector<xla::Promise<>> copies_;
};

// A consumer whose host-to-device copies complete when the test says so.
class RecvTestManager : public KVCacheManagerWithTransfer {
 public:
  explicit RecvTestManager(size_t num_layers, double timeout_s = 5.0)
      : KVCacheManagerWithTransfer(num_layers, /*num_shards=*/1,
                                   /*slice_byte_size=*/128,
                                   /*local_port=*/std::nullopt,
                                   /*host_blocks_to_allocate=*/std::nullopt,
                                   /*parallelism=*/1, /*node_id=*/0,
                                   /*local_control_port=*/-1, /*max_blocks=*/1,
                                   /*num_slots=*/kSlots, timeout_s) {
    CHECK_OK(ConfigureHostStagingSlots(kSlots, /*max_major_per_slot=*/1));
    CHECK_OK(InitializeSlotPool(kSlots));
  }

  void AddRecv(const std::string& req_id, uint64_t uuid,
               int32_t blocks_per_layer = 1) {
    absl::MutexLock lock(mu_);
    RecvEntry entry;
    entry.req_id = req_id;
    entry.slot_idx = AcquireSlotLocked().slot_idx;
    entry.total_blocks = blocks_per_layer;
    entry.deadline = DeadlineFromNow();
    entry.start_time = std::chrono::steady_clock::now();
    active_recv_entries_[uuid] = std::move(entry);
  }

  absl::Status ReceiveLayer(size_t layer, uint64_t uuid) {
    return OnLayerReceived(layer, uuid);
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

  size_t free_slots() {
    absl::MutexLock lock(mu_);
    return free_slots_.size();
  }

  bool has_recv(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    return active_recv_entries_.contains(uuid);
  }

  void ExpireRecv(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    active_recv_entries_.at(uuid).deadline =
        std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2dSyncDispatch(
      const std::vector<int64_t>& src_offsets_major_dim,
      const std::vector<int64_t>& dst_offsets_major_dim,
      const std::vector<int64_t>& copy_sizes_major_dim,
      std::optional<int64_t> slot_idx, std::optional<size_t> layer_idx,
      std::optional<size_t> shard_idx) override {
    auto [promise, future] = xla::MakePromise<>();
    absl::MutexLock lock(copies_mu_);
    copies_.push_back(std::move(promise));
    return raiden::PjRtCopyFuture(std::move(future), raiden::BufferHolders{});
  }

 private:
  absl::Mutex copies_mu_;
  std::vector<xla::Promise<>> copies_;
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
  ASSERT_GT(producer.NotifyForRead("req", /*uuid=*/7, {0}), 0);
  producer.ServePull(7);
  ASSERT_EQ(producer.copies_issued(), 1);
  EXPECT_EQ(producer.free_slots(), kSlots - 1);

  // The deadline passes while the copy runs: the send is not reported and
  // its slot stays out of the pool.
  producer.ExpireSend(/*uuid=*/7);
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
  ASSERT_GT(producer.NotifyForRead("req", /*uuid=*/9, {0}), 0);
  producer.ExpireSend(/*uuid=*/9);
  Reports swept = producer.CompleteReadRaw();
  EXPECT_THAT(FailedRecving(swept), Contains("req"));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendLifecycleTest, DuplicateRegistrationCannotReplaceLiveOffer) {
  TestManager producer(/*num_layers=*/1);
  ASSERT_GT(producer.NotifyForRead("first", /*uuid=*/10, {0}), 0);
  EXPECT_EQ(producer.NotifyForRead("replacement", /*uuid=*/10, {0}), 0);
  EXPECT_TRUE(producer.has_send(10));

  producer.ExpireSend(/*uuid=*/10);
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
  auto entry = producer.AddSyntheticSend("req", /*uuid=*/10, /*in_flight=*/0);

  producer.Decide(entry, /*failed=*/false);
  Reports reports = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(reports), Contains("req"));
  EXPECT_THAT(FailedRecving(reports), IsEmpty());
  EXPECT_FALSE(producer.has_send(10));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendLifecycleTest, FailedSendWithoutWorkSettlesImmediately) {
  TestManager producer(/*num_layers=*/1);
  auto entry = producer.AddSyntheticSend("req", /*uuid=*/11, /*in_flight=*/0);

  producer.Decide(entry, /*failed=*/true);
  Reports reports = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(reports), IsEmpty());
  EXPECT_THAT(FailedRecving(reports), Contains("req"));
  EXPECT_FALSE(producer.has_send(11));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendLifecycleTest, SuccessfulSendWaitsForEveryOperation) {
  TestManager producer(/*num_layers=*/1);
  auto entry = producer.AddSyntheticSend("req", /*uuid=*/12, /*in_flight=*/2);

  producer.Decide(entry, /*failed=*/false);
  producer.End(entry);
  Reports during = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(during), IsEmpty());
  EXPECT_THAT(FailedRecving(during), IsEmpty());
  EXPECT_TRUE(producer.has_send(12));
  EXPECT_EQ(producer.free_slots(), kSlots - 1);

  producer.End(entry);
  Reports after = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(after), Contains("req"));
  EXPECT_THAT(FailedRecving(after), IsEmpty());
  EXPECT_FALSE(producer.has_send(12));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendLifecycleTest, FailureWinsWhileSuccessfulSendIsDraining) {
  TestManager producer(/*num_layers=*/1);
  auto entry = producer.AddSyntheticSend("req", /*uuid=*/13, /*in_flight=*/1);

  producer.Decide(entry, /*failed=*/false);
  producer.Decide(entry, /*failed=*/true);
  producer.End(entry);
  Reports reports = producer.CompleteReadRaw();
  EXPECT_THAT(DoneSending(reports), IsEmpty());
  EXPECT_THAT(FailedRecving(reports), Contains("req"));
  EXPECT_FALSE(producer.has_send(13));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

TEST(SendLifecycleTest, SuccessCannotOverrideAnEarlierFailure) {
  TestManager producer(/*num_layers=*/1);
  auto entry = producer.AddSyntheticSend("req", /*uuid=*/14, /*in_flight=*/1);

  producer.Decide(entry, /*failed=*/true);
  producer.Decide(entry, /*failed=*/false);
  producer.End(entry);
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
  consumer.AddRecv("req", /*uuid=*/24);
  consumer.ExpireRecv(/*uuid=*/24);

  Reports reports = consumer.CompleteReadRaw();
  EXPECT_THAT(DoneReceiving(reports), IsEmpty());
  EXPECT_THAT(FailedRecving(reports), Contains("req"));
  EXPECT_FALSE(consumer.has_recv(24));
  EXPECT_EQ(consumer.free_slots(), kSlots);
}

}  // namespace
}  // namespace tpu_raiden
