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

#include "tpu_sync/kv_cache/staging_manager.h"

#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <optional>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Optional;

std::chrono::steady_clock::time_point FutureDeadline() {
  return std::chrono::steady_clock::now() + std::chrono::seconds(30);
}

TEST(StagingLayerManagerTest, ZeroSlotManagerFailsAcquire) {
  StagingLayerManager default_manager;
  EXPECT_EQ(default_manager.num_free(), 0);
  EXPECT_EQ(default_manager.Lookup(/*unit=*/0), std::nullopt);
  EXPECT_EQ(default_manager.Release(/*unit=*/0), std::nullopt);
  EXPECT_THAT(default_manager.Acquire(/*unit=*/0, FutureDeadline()),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  default_manager.ReleaseAll();
  default_manager.Shutdown();

  StagingLayerManager zero_manager(/*num_slots=*/0);
  EXPECT_EQ(zero_manager.num_free(), 0);
  EXPECT_THAT(zero_manager.Acquire(/*unit=*/1, FutureDeadline()),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(StagingLayerManagerTest, AcquireLeasesSlotsInOrderAndLookupMatches) {
  StagingLayerManager manager(/*num_slots=*/3);
  EXPECT_EQ(manager.num_free(), 3);
  EXPECT_EQ(manager.Lookup(/*unit=*/10), std::nullopt);

  EXPECT_THAT(manager.Acquire(/*unit=*/10, FutureDeadline()), IsOkAndHolds(0));
  EXPECT_EQ(manager.num_free(), 2);
  EXPECT_THAT(manager.Lookup(/*unit=*/10), Optional(0));

  EXPECT_THAT(manager.Acquire(/*unit=*/20, FutureDeadline()), IsOkAndHolds(1));
  EXPECT_EQ(manager.num_free(), 1);
  EXPECT_THAT(manager.Lookup(/*unit=*/20), Optional(1));

  EXPECT_THAT(manager.Acquire(/*unit=*/30, FutureDeadline()), IsOkAndHolds(2));
  EXPECT_EQ(manager.num_free(), 0);
  EXPECT_THAT(manager.Lookup(/*unit=*/30), Optional(2));
}

TEST(StagingLayerManagerTest, AcquireIsIdempotentForAlreadyLeasedUnit) {
  StagingLayerManager manager(/*num_slots=*/2);
  EXPECT_THAT(manager.Acquire(/*unit=*/5, FutureDeadline()), IsOkAndHolds(0));
  EXPECT_THAT(manager.Acquire(/*unit=*/7, FutureDeadline()), IsOkAndHolds(1));
  EXPECT_EQ(manager.num_free(), 0);

  // Re-acquiring an already leased unit returns its existing slot immediately
  // even when all slots are leased and the deadline has passed.
  const auto past_deadline =
      std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
  EXPECT_THAT(manager.Acquire(/*unit=*/5, past_deadline), IsOkAndHolds(0));
  EXPECT_THAT(manager.Acquire(/*unit=*/7, past_deadline), IsOkAndHolds(1));
  EXPECT_EQ(manager.num_free(), 0);
}

TEST(StagingLayerManagerTest, ReleaseReturnsSlotToTailOfFreeQueue) {
  StagingLayerManager manager(/*num_slots=*/3);
  EXPECT_THAT(manager.Acquire(/*unit=*/0, FutureDeadline()), IsOkAndHolds(0));
  EXPECT_THAT(manager.Acquire(/*unit=*/1, FutureDeadline()), IsOkAndHolds(1));
  EXPECT_THAT(manager.Acquire(/*unit=*/2, FutureDeadline()), IsOkAndHolds(2));
  EXPECT_EQ(manager.num_free(), 0);

  EXPECT_EQ(manager.Release(/*unit=*/99), std::nullopt);

  EXPECT_THAT(manager.Release(/*unit=*/1), Optional(1));
  EXPECT_EQ(manager.num_free(), 1);
  EXPECT_EQ(manager.Lookup(/*unit=*/1), std::nullopt);
  EXPECT_EQ(manager.Release(/*unit=*/1), std::nullopt);

  EXPECT_THAT(manager.Release(/*unit=*/0), Optional(0));
  EXPECT_EQ(manager.num_free(), 2);

  // Free queue order is [1, 0].
  EXPECT_THAT(manager.Acquire(/*unit=*/3, FutureDeadline()), IsOkAndHolds(1));
  EXPECT_THAT(manager.Acquire(/*unit=*/4, FutureDeadline()), IsOkAndHolds(0));
  EXPECT_EQ(manager.num_free(), 0);
}

TEST(StagingLayerManagerTest, ReleaseAllEndsEveryLease) {
  StagingLayerManager manager(/*num_slots=*/3);
  EXPECT_THAT(manager.Acquire(/*unit=*/1, FutureDeadline()), IsOkAndHolds(0));
  EXPECT_THAT(manager.Acquire(/*unit=*/2, FutureDeadline()), IsOkAndHolds(1));
  EXPECT_EQ(manager.num_free(), 1);

  manager.ReleaseAll();
  EXPECT_EQ(manager.num_free(), 3);
  EXPECT_EQ(manager.Lookup(/*unit=*/1), std::nullopt);
  EXPECT_EQ(manager.Lookup(/*unit=*/2), std::nullopt);
  EXPECT_EQ(manager.Release(/*unit=*/1), std::nullopt);

  EXPECT_THAT(manager.Acquire(/*unit=*/10, FutureDeadline()), IsOk());
  EXPECT_THAT(manager.Acquire(/*unit=*/11, FutureDeadline()), IsOk());
  EXPECT_THAT(manager.Acquire(/*unit=*/12, FutureDeadline()), IsOk());
  EXPECT_EQ(manager.num_free(), 0);
}

TEST(StagingLayerManagerTest, AcquireBlocksUntilSlotReleased) {
  StagingLayerManager manager(/*num_slots=*/1);
  EXPECT_THAT(manager.Acquire(/*unit=*/0, FutureDeadline()), IsOkAndHolds(0));

  absl::Notification started;
  absl::Notification acquired;
  absl::StatusOr<size_t> waiter_slot;
  std::thread waiter([&]() {
    started.Notify();
    waiter_slot = manager.Acquire(/*unit=*/1, FutureDeadline());
    acquired.Notify();
  });

  started.WaitForNotification();
  EXPECT_FALSE(acquired.WaitForNotificationWithTimeout(absl::Milliseconds(20)));
  EXPECT_EQ(manager.Lookup(/*unit=*/1), std::nullopt);

  EXPECT_THAT(manager.Release(/*unit=*/0), Optional(0));
  acquired.WaitForNotification();
  waiter.join();

  EXPECT_THAT(waiter_slot, IsOkAndHolds(0));
  EXPECT_THAT(manager.Lookup(/*unit=*/1), Optional(0));
  EXPECT_EQ(manager.num_free(), 0);
}

TEST(StagingLayerManagerTest, AcquireTimesOutWhenAllSlotsLeased) {
  StagingLayerManager manager(/*num_slots=*/1);
  EXPECT_THAT(manager.Acquire(/*unit=*/0, FutureDeadline()), IsOkAndHolds(0));

  EXPECT_THAT(manager.Acquire(/*unit=*/1, std::chrono::steady_clock::now() -
                                              std::chrono::milliseconds(1)),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_THAT(manager.Acquire(/*unit=*/1, std::chrono::steady_clock::now() +
                                              std::chrono::milliseconds(10)),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_EQ(manager.Lookup(/*unit=*/1), std::nullopt);
  EXPECT_EQ(manager.num_free(), 0);

  EXPECT_THAT(manager.Release(/*unit=*/0), Optional(0));
  EXPECT_THAT(manager.Acquire(/*unit=*/1, FutureDeadline()), IsOkAndHolds(0));
}

TEST(StagingLayerManagerTest, ShutdownFailsFutureAcquiresWithCancelled) {
  StagingLayerManager manager(/*num_slots=*/2);
  EXPECT_THAT(manager.Acquire(/*unit=*/0, FutureDeadline()), IsOkAndHolds(0));

  manager.Shutdown();
  EXPECT_THAT(manager.Acquire(/*unit=*/1, FutureDeadline()),
              StatusIs(absl::StatusCode::kCancelled));
  EXPECT_THAT(manager.Acquire(/*unit=*/0, FutureDeadline()),
              StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(manager.num_free(), 1);
}

TEST(StagingLayerManagerTest, ShutdownWakesBlockedAcquireWithCancelled) {
  StagingLayerManager manager(/*num_slots=*/1);
  EXPECT_THAT(manager.Acquire(/*unit=*/0, FutureDeadline()), IsOkAndHolds(0));

  absl::Notification started;
  absl::Notification finished;
  absl::StatusOr<size_t> waiter_slot;
  std::thread waiter([&]() {
    started.Notify();
    waiter_slot = manager.Acquire(/*unit=*/1, FutureDeadline());
    finished.Notify();
  });

  started.WaitForNotification();
  EXPECT_FALSE(finished.WaitForNotificationWithTimeout(absl::Milliseconds(20)));

  manager.Shutdown();
  finished.WaitForNotification();
  waiter.join();

  EXPECT_THAT(waiter_slot, StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(manager.Lookup(/*unit=*/1), std::nullopt);
  EXPECT_EQ(manager.num_free(), 0);
}

TEST(StagingLayerManagerTest, ExternalCancelledFlagFailsAcquireImmediately) {
  StagingLayerManager manager(/*num_slots=*/1);
  std::atomic<bool> cancelled{true};

  EXPECT_THAT(manager.Acquire(/*unit=*/0, FutureDeadline(), &cancelled),
              StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(manager.num_free(), 1);
  EXPECT_EQ(manager.Lookup(/*unit=*/0), std::nullopt);
}

TEST(StagingLayerManagerTest, ExternalCancelledFlagAbortsWaitWithoutLeasing) {
  StagingLayerManager manager(/*num_slots=*/1);
  EXPECT_THAT(manager.Acquire(/*unit=*/0, FutureDeadline()), IsOkAndHolds(0));

  std::atomic<bool> cancelled{false};
  absl::Notification started;
  absl::Notification finished;
  absl::StatusOr<size_t> waiter_slot;
  std::thread waiter([&]() {
    started.Notify();
    waiter_slot = manager.Acquire(/*unit=*/1, FutureDeadline(), &cancelled);
    finished.Notify();
  });

  started.WaitForNotification();
  EXPECT_FALSE(finished.WaitForNotificationWithTimeout(absl::Milliseconds(20)));

  cancelled.store(true, std::memory_order_relaxed);
  EXPECT_THAT(manager.Release(/*unit=*/0), Optional(0));
  finished.WaitForNotification();
  waiter.join();

  EXPECT_THAT(waiter_slot, StatusIs(absl::StatusCode::kCancelled));
  EXPECT_EQ(manager.Lookup(/*unit=*/1), std::nullopt);
  EXPECT_EQ(manager.num_free(), 1);
}

TEST(StagingLayerManagerTest, ConcurrentWorkersMaintainExclusiveSlotOwnership) {
  constexpr size_t kNumSlots = 3;
  constexpr size_t kNumWorkers = 8;
  constexpr size_t kIterations = 50;

  StagingLayerManager manager(kNumSlots);
  std::vector<std::atomic<int>> active_owners(kNumSlots);
  for (std::atomic<int>& owner : active_owners) {
    owner.store(0, std::memory_order_relaxed);
  }

  std::vector<std::thread> workers;
  workers.reserve(kNumWorkers);
  for (size_t worker_id = 0; worker_id < kNumWorkers; ++worker_id) {
    workers.emplace_back([&, worker_id]() {
      for (size_t iter = 0; iter < kIterations; ++iter) {
        absl::StatusOr<size_t> slot =
            manager.Acquire(worker_id, FutureDeadline());
        ASSERT_THAT(slot, IsOk());
        ASSERT_LT(*slot, kNumSlots);
        EXPECT_THAT(manager.Lookup(worker_id), Optional(*slot));
        EXPECT_EQ(active_owners[*slot].fetch_add(1, std::memory_order_acq_rel),
                  0);
        std::this_thread::yield();
        EXPECT_EQ(active_owners[*slot].fetch_sub(1, std::memory_order_acq_rel),
                  1);
        EXPECT_THAT(manager.Release(worker_id), Optional(*slot));
      }
    });
  }

  for (std::thread& worker : workers) {
    worker.join();
  }

  EXPECT_EQ(manager.num_free(), kNumSlots);
  for (const std::atomic<int>& owner : active_owners) {
    EXPECT_EQ(owner.load(std::memory_order_acquire), 0);
  }
}

}  // namespace
}  // namespace tpu_raiden
