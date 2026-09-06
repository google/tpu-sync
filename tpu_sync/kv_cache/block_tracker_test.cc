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

#include "tpu_sync/kv_cache/block_tracker.h"

#include <atomic>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tpu_raiden::kv_cache {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(BlockTrackerTest, AddAndCheckPending) {
  BlockTracker tracker;
  EXPECT_TRUE(tracker.Empty());
  EXPECT_EQ(tracker.PendingCount(), 0);

  tracker.AddPending("block1");
  EXPECT_TRUE(tracker.IsPending("block1"));
  EXPECT_TRUE(tracker.Contains("block1"));
  EXPECT_FALSE(tracker.IsPending("block2"));
  EXPECT_EQ(tracker.PendingCount(), 1);
  EXPECT_THAT(tracker.GetPending(), ElementsAre("block1"));

  // Duplicate add should be a no-op and preserve order.
  std::vector<std::string> batch = {"block2", "block1", "block3"};
  tracker.AddPending(batch);
  EXPECT_EQ(tracker.PendingCount(), 3);
  EXPECT_THAT(tracker.GetPending(), ElementsAre("block1", "block2", "block3"));

  // Remove pending single.
  tracker.RemovePending("block2");
  EXPECT_FALSE(tracker.IsPending("block2"));
  EXPECT_EQ(tracker.PendingCount(), 2);
  EXPECT_THAT(tracker.GetPending(), ElementsAre("block1", "block3"));

  // Remove pending batch.
  std::vector<std::string> to_remove = {"block1",
                                        "block4"};  // block4 does not exist
  tracker.RemovePending(to_remove);
  EXPECT_EQ(tracker.PendingCount(), 1);
  EXPECT_THAT(tracker.GetPending(), ElementsAre("block3"));
}

TEST(BlockTrackerTest, MarkDoneAndPoll) {
  BlockTracker tracker;
  std::vector<std::string> blocks = {"b1", "b2", "b3"};
  tracker.AddPending(blocks);

  tracker.MarkDone("b1");
  EXPECT_FALSE(tracker.IsPending("b1"));
  EXPECT_TRUE(tracker.IsPending("b2"));
  EXPECT_EQ(tracker.DoneCount(), 1);
  EXPECT_EQ(tracker.PendingCount(), 2);

  std::vector<std::string> more_done = {"b2", "b4"};  // b4 was not pending
  tracker.MarkDone(more_done);
  EXPECT_FALSE(tracker.IsPending("b2"));
  EXPECT_EQ(tracker.DoneCount(), 3);
  EXPECT_EQ(tracker.PendingCount(), 1);

  auto res = tracker.Poll();
  EXPECT_THAT(res.pending, ElementsAre("b3"));
  EXPECT_THAT(res.done, ElementsAre("b1", "b2", "b4"));
  EXPECT_THAT(res.failed, IsEmpty());
  EXPECT_THAT(res.existing, IsEmpty());
  EXPECT_THAT(res.unregistered, IsEmpty());

  // Second Poll should see drained done, but pending remains.
  auto res2 = tracker.Poll();
  EXPECT_THAT(res2.pending, ElementsAre("b3"));
  EXPECT_THAT(res2.done, IsEmpty());
}

TEST(BlockTrackerTest, MarkFailedAndPoll) {
  BlockTracker tracker;
  std::vector<std::string> blocks = {"b1", "b2"};
  tracker.AddPending(blocks);

  tracker.MarkFailed("b1");
  EXPECT_FALSE(tracker.IsPending("b1"));
  EXPECT_EQ(tracker.FailedCount(), 1);

  tracker.MarkFailed(std::vector<std::string>{"b2"});
  EXPECT_FALSE(tracker.IsPending("b2"));
  EXPECT_EQ(tracker.FailedCount(), 2);
  EXPECT_EQ(tracker.PendingCount(), 0);

  auto res = tracker.Poll();
  EXPECT_THAT(res.pending, IsEmpty());
  EXPECT_THAT(res.failed, ElementsAre("b1", "b2"));
  EXPECT_THAT(res.done, IsEmpty());

  // Poll drains failed.
  EXPECT_EQ(tracker.FailedCount(), 0);
  EXPECT_TRUE(tracker.Empty());
}

TEST(BlockTrackerTest, MarkExistingAndUnregistered) {
  BlockTracker tracker;
  tracker.MarkExisting("e1");
  tracker.MarkExisting(std::vector<std::string>{"e2", "e3"});

  tracker.MarkUnregistered("u1");
  tracker.MarkUnregistered(std::vector<std::string>{"u2"});

  auto res = tracker.Poll();
  EXPECT_THAT(res.existing, ElementsAre("e1", "e2", "e3"));
  EXPECT_THAT(res.unregistered, ElementsAre("u1", "u2"));

  // Drained on Poll.
  auto res2 = tracker.Poll();
  EXPECT_THAT(res2.existing, IsEmpty());
  EXPECT_THAT(res2.unregistered, IsEmpty());
}

TEST(BlockTrackerTest, Update) {
  BlockTracker tracker;
  tracker.AddPending(std::vector<std::string>{"b1", "b2", "b3"});

  std::vector<std::string> done = {"b1"};
  std::vector<std::string> failed = {"b2"};
  tracker.Update(done, failed);

  EXPECT_EQ(tracker.PendingCount(), 1);
  EXPECT_EQ(tracker.DoneCount(), 1);
  EXPECT_EQ(tracker.FailedCount(), 1);

  auto res = tracker.Poll();
  EXPECT_THAT(res.pending, ElementsAre("b3"));
  EXPECT_THAT(res.done, ElementsAre("b1"));
  EXPECT_THAT(res.failed, ElementsAre("b2"));
}

TEST(BlockTrackerTest, Merge) {
  BlockTracker tracker1;
  tracker1.AddPending(std::vector<std::string>{"b1", "b2", "b3"});

  BlockTracker tracker2;
  tracker2.AddPending("b4");
  tracker2.MarkDone("b1");
  tracker2.MarkFailed("b2");
  tracker2.MarkExisting("b2_ex");
  tracker2.MarkUnregistered("b2_unreg");

  tracker1.Merge(std::move(tracker2));

  EXPECT_TRUE(tracker2.Empty());          // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(tracker2.PendingCount(), 0);  // NOLINT(bugprone-use-after-move)

  EXPECT_EQ(tracker1.PendingCount(), 2);
  EXPECT_EQ(tracker1.DoneCount(), 1);
  EXPECT_EQ(tracker1.FailedCount(), 1);

  auto res = tracker1.Poll();
  EXPECT_THAT(res.pending, ElementsAre("b3", "b4"));
  EXPECT_THAT(res.done, ElementsAre("b1"));
  EXPECT_THAT(res.failed, ElementsAre("b2"));
  EXPECT_THAT(res.existing, ElementsAre("b2_ex"));
  EXPECT_THAT(res.unregistered, ElementsAre("b2_unreg"));
}

TEST(BlockTrackerTest, PollWithTimeout) {
  BlockTracker tracker;
  auto res = tracker.Poll(absl::Milliseconds(10));
  EXPECT_THAT(res.pending, IsEmpty());
  EXPECT_THAT(res.done, IsEmpty());

  std::thread producer([&tracker]() {
    absl::SleepFor(absl::Milliseconds(20));
    tracker.MarkDone("b1");
  });

  auto res2 = tracker.Poll(absl::Seconds(1));
  EXPECT_THAT(res2.done, ElementsAre("b1"));
  producer.join();
}

TEST(BlockTrackerTest, ClearAndEmpty) {
  BlockTracker tracker;
  tracker.AddPending("b1");
  tracker.MarkDone("b2");
  tracker.MarkFailed("b3");
  tracker.MarkExisting("b4");
  tracker.MarkUnregistered("b5");

  EXPECT_FALSE(tracker.Empty());
  tracker.Clear();
  EXPECT_TRUE(tracker.Empty());
  EXPECT_EQ(tracker.PendingCount(), 0);
  EXPECT_EQ(tracker.DoneCount(), 0);
  EXPECT_EQ(tracker.FailedCount(), 0);

  auto res = tracker.Poll();
  EXPECT_THAT(res.pending, IsEmpty());
  EXPECT_THAT(res.done, IsEmpty());
  EXPECT_THAT(res.failed, IsEmpty());
  EXPECT_THAT(res.existing, IsEmpty());
  EXPECT_THAT(res.unregistered, IsEmpty());
}

TEST(BlockTrackerTest, ConcurrentAccess) {
  BlockTracker tracker;
  constexpr int kNumProducers = 4;
  constexpr int kOpsPerProducer = 500;
  std::atomic<bool> start{false};
  std::atomic<bool> done{false};

  // Pre-populate some pending.
  std::vector<std::string> initial_pending;
  for (int i = 0; i < kNumProducers * kOpsPerProducer; ++i) {
    initial_pending.push_back(absl::StrCat("block_", i));
  }
  tracker.AddPending(initial_pending);

  std::vector<std::thread> producers;
  producers.reserve(kNumProducers);
  for (int p = 0; p < kNumProducers; ++p) {
    producers.emplace_back([&, p]() {
      while (!start.load()) {
      }
      for (int i = 0; i < kOpsPerProducer; ++i) {
        std::string hash = absl::StrCat("block_", p * kOpsPerProducer + i);
        if (i % 2 == 0) {
          tracker.MarkDone(hash);
        } else {
          tracker.MarkFailed(hash);
        }
      }
    });
  }

  std::vector<std::thread> consumers;
  consumers.reserve(2);
  for (int c = 0; c < 2; ++c) {
    consumers.emplace_back([&]() {
      while (!start.load()) {
      }
      while (!done.load()) {
        auto res = tracker.Poll();
      }
      // One final poll after completion.
      (void)tracker.Poll();
    });
  }

  start.store(true);
  for (auto& t : producers) {
    t.join();
  }
  done.store(true);
  for (auto& t : consumers) {
    t.join();
  }

  EXPECT_EQ(tracker.PendingCount(), 0);
}

}  // namespace
}  // namespace tpu_raiden::kv_cache
