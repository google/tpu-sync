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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/future.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/raw_transfer_core.h"

namespace tpu_raiden {
namespace {

using ::testing::Contains;
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

using Reports = std::tuple<std::vector<std::string>, std::vector<std::string>,
                           std::vector<std::string>>;

const std::vector<std::string>& DoneSending(const Reports& r) {
  return std::get<0>(r);
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
  absl::SleepFor(absl::Milliseconds(120));
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
  absl::SleepFor(absl::Milliseconds(120));
  Reports swept = producer.CompleteReadRaw();
  EXPECT_THAT(FailedRecving(swept), Contains("req"));
  EXPECT_EQ(producer.free_slots(), kSlots);
}

}  // namespace
}  // namespace tpu_raiden
