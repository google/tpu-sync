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

#include "tpu_sync/common/detached_thread_group.h"

#include <atomic>

#include <gtest/gtest.h>
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tpu_raiden {
namespace {

TEST(DetachedThreadGroupTest, RunsEverySpawnedTask) {
  constexpr int kTasks = 64;
  std::atomic<int> ran{0};
  {
    DetachedThreadGroup group("test");
    for (int i = 0; i < kTasks; ++i) {
      EXPECT_TRUE(group.Spawn([&ran] { ++ran; }));
    }
    group.AwaitAllDone();
    EXPECT_EQ(ran.load(), kTasks);
  }
}

TEST(DetachedThreadGroupTest, AwaitAllDoneBlocksUntilTasksReturn) {
  absl::Notification release;
  std::atomic<bool> task_returned{false};

  DetachedThreadGroup group("test");
  ASSERT_TRUE(group.Spawn([&release, &task_returned] {
    release.WaitForNotification();
    task_returned = true;
  }));

  // The task cannot have returned yet, so AwaitAllDone() must not either.
  EXPECT_FALSE(task_returned.load());
  release.Notify();
  group.AwaitAllDone();
  EXPECT_TRUE(task_returned.load());
}

TEST(DetachedThreadGroupTest, AwaitAllDoneReturnsImmediatelyWhenIdle) {
  DetachedThreadGroup group("test");
  group.AwaitAllDone();
  group.AwaitAllDone();
}

TEST(DetachedThreadGroupTest, DestructorWaitsForInFlightTasks) {
  absl::Notification started;
  std::atomic<bool> task_returned{false};
  {
    DetachedThreadGroup group("test");
    ASSERT_TRUE(group.Spawn([&started, &task_returned] {
      started.Notify();
      absl::SleepFor(absl::Milliseconds(50));
      task_returned = true;
    }));
    started.WaitForNotification();
  }
  EXPECT_TRUE(task_returned.load());
}

}  // namespace
}  // namespace tpu_raiden
