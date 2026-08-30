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

#include "tpu_sync/kv_cache/completion_executor.h"

#include <atomic>
#include <cstdint>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gtest/gtest.h>
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

TEST(CompletionExecutorTest, RunsTheTaskOffTheSubmittingThread) {
  absl::Notification ran;
  std::thread::id ran_on;
  const std::thread::id submitter = std::this_thread::get_id();

  CompletionExecutor::Instance().Execute([&] {
    ran_on = std::this_thread::get_id();
    ran.Notify();
  });

  ASSERT_TRUE(ran.WaitForNotificationWithTimeout(absl::Seconds(10)));
  EXPECT_NE(ran_on, submitter)
      << "a completion that runs on the submitting thread re-enters the store "
         "from inside its own submit path";
}

TEST(CompletionExecutorTest, RunsEveryTaskExactlyOnce) {
  constexpr int kTasks = 500;
  std::atomic<int> ran{0};
  for (int i = 0; i < kTasks; ++i) {
    CompletionExecutor::Instance().Execute([&ran] { ran.fetch_add(1); });
  }
  CompletionExecutor::Instance().DrainForTesting();
  EXPECT_EQ(ran.load(), kTasks);
}

// The guard on the claim that this design REPLACES a poll rather than
// relocating one.
//
// The counting point is the whole test: wakeups are recorded on every return
// from the worker's Await, so a worker that woke on a deadline would tick the
// counter with an empty queue. Counting tasks executed instead would pass just
// as happily against a 10 ms timed wait -- which is exactly the regression
// worth catching, since the sweep this replaced cost 100 wakeups a second per
// store with nothing in flight.
TEST(CompletionExecutorTest, IdleExecutorDoesNotWake) {
  CompletionExecutor& executor = CompletionExecutor::Instance();
  executor.DrainForTesting();
  const int64_t before = executor.WakeupsForTesting();

  absl::SleepFor(absl::Milliseconds(250));
  EXPECT_EQ(executor.WakeupsForTesting(), before)
      << "an idle executor woke up; it is polling, not parking";

  constexpr int kTasks = 8;
  std::atomic<int> ran{0};
  for (int i = 0; i < kTasks; ++i) {
    executor.Execute([&ran] { ran.fetch_add(1); });
  }
  executor.DrainForTesting();
  EXPECT_EQ(ran.load(), kTasks);
  EXPECT_EQ(executor.WakeupsForTesting(), before + kTasks)
      << "one wakeup per task and no others";
}

// Four workers, so one slow completion cannot hold up the rest. A single
// worker would make every store in the process wait behind whichever
// completion is currently taking a contended backend lock.
TEST(CompletionExecutorTest, SlowTaskDoesNotBlockOthers) {
  absl::Notification release;
  absl::Notification blocked;
  CompletionExecutor::Instance().Execute([&] {
    blocked.Notify();
    release.WaitForNotificationWithTimeout(absl::Seconds(10));
  });
  ASSERT_TRUE(blocked.WaitForNotificationWithTimeout(absl::Seconds(10)));

  absl::Notification second;
  CompletionExecutor::Instance().Execute([&] { second.Notify(); });
  EXPECT_TRUE(second.WaitForNotificationWithTimeout(absl::Seconds(10)))
      << "a second completion queued behind a blocked one";

  release.Notify();
  CompletionExecutor::Instance().DrainForTesting();
}

TEST(CompletionExecutorTest, ConcurrentSubmittersAllLand) {
  constexpr int kThreads = 8;
  constexpr int kPerThread = 100;
  std::atomic<int> ran{0};
  std::vector<std::thread> submitters;
  submitters.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    submitters.emplace_back([&ran] {
      for (int i = 0; i < kPerThread; ++i) {
        CompletionExecutor::Instance().Execute([&ran] { ran.fetch_add(1); });
      }
    });
  }
  for (std::thread& s : submitters) s.join();
  CompletionExecutor::Instance().DrainForTesting();
  EXPECT_EQ(ran.load(), kThreads * kPerThread);
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
