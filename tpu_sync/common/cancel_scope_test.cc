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

#include "tpu_sync/common/cancel_scope.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <optional>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::StatusIs;

TEST(CancelScopeTest, CancelIsStickyAndIdempotent) {
  CancelScope scope;
  int runs = 0;
  absl::StatusOr<CancelScope::Registration> registration =
      scope.OnCancel([&runs] { ++runs; });
  ABSL_ASSERT_OK(registration);
  EXPECT_FALSE(scope.cancelled());

  scope.Cancel();
  EXPECT_TRUE(scope.cancelled());
  EXPECT_EQ(runs, 1);
  scope.Cancel();
  EXPECT_TRUE(scope.cancelled());
  EXPECT_EQ(runs, 1);
}

TEST(CancelScopeTest, ConcurrentCancelsAreSeenByEveryThread) {
  constexpr int kWaiters = 4;
  constexpr int kCancellers = 4;
  CancelScope scope;
  std::vector<std::thread> threads;
  // Each waiter returns only once some canceller's Cancel() is visible to it.
  for (int i = 0; i < kWaiters; ++i) {
    threads.emplace_back([&scope] {
      while (!scope.cancelled()) std::this_thread::yield();
    });
  }
  for (int i = 0; i < kCancellers; ++i) {
    threads.emplace_back([&scope] { scope.Cancel(); });
  }
  for (std::thread& thread : threads) thread.join();

  EXPECT_TRUE(scope.cancelled());
}

TEST(CancelScopeTest, ConcurrentCancelRunsNoOtherCallback) {
  CancelScope scope;
  std::atomic<int> started{0};
  absl::Notification first_started;
  absl::Notification release_first;
  // Whichever callback runs first blocks until released.
  const auto callback = [&] {
    if (started.fetch_add(1) == 0) {
      first_started.Notify();
      release_first.WaitForNotification();
    }
  };
  absl::StatusOr<CancelScope::Registration> first = scope.OnCancel(callback);
  ABSL_ASSERT_OK(first);
  absl::StatusOr<CancelScope::Registration> second = scope.OnCancel(callback);
  ABSL_ASSERT_OK(second);
  std::thread canceller([&scope] { scope.Cancel(); });
  first_started.WaitForNotification();

  // Returns at once and leaves the other callback to `canceller`.
  scope.Cancel();
  EXPECT_EQ(started.load(), 1);
  release_first.Notify();
  canceller.join();
  EXPECT_EQ(started.load(), 2);
}

TEST(CancelScopeTest, OnCancelAfterCancelFailsWithoutRunningTheCallback) {
  CancelScope scope;
  scope.Cancel();
  bool ran = false;

  EXPECT_THAT(scope.OnCancel([&ran] { ran = true; }),
              StatusIs(absl::StatusCode::kCancelled));
  EXPECT_FALSE(ran);
}

TEST(CancelScopeTest, OnCancelWithAnEmptyCallbackFails) {
  CancelScope scope;
  absl::AnyInvocable<void() &&> empty_callback;

  absl::StatusOr<CancelScope::Registration> registration =
      scope.OnCancel(std::move(empty_callback));
  EXPECT_THAT(registration, StatusIs(absl::StatusCode::kInvalidArgument));
  scope.Cancel();  // Would call the empty callback had it been registered.
}

TEST(CancelScopeTest, DeregisteredCallbackNeverRuns) {
  CancelScope scope;
  bool dropped_ran = false;
  bool kept_ran = false;
  std::optional<absl::StatusOr<CancelScope::Registration>> dropped;
  dropped.emplace(scope.OnCancel([&dropped_ran] { dropped_ran = true; }));
  ABSL_ASSERT_OK(*dropped);
  absl::StatusOr<CancelScope::Registration> kept =
      scope.OnCancel([&kept_ran] { kept_ran = true; });
  ABSL_ASSERT_OK(kept);

  dropped.reset();
  scope.Cancel();
  EXPECT_FALSE(dropped_ran);
  EXPECT_TRUE(kept_ran);
}

TEST(CancelScopeTest, CapturesOfADroppedCallbackCanUseTheScope) {
  CancelScope scope;
  std::optional<absl::StatusOr<CancelScope::Registration>> other;
  other.emplace(scope.OnCancel([] {}));
  ABSL_ASSERT_OK(*other);
  // Destroying this callback drops `other`, which locks the scope again.
  std::optional<absl::StatusOr<CancelScope::Registration>> dropped;
  dropped.emplace(scope.OnCancel(
      [drop_other = absl::MakeCleanup([&other] { other.reset(); })] {}));
  ABSL_ASSERT_OK(*dropped);
  ASSERT_TRUE(other.has_value());

  dropped.reset();  // Deadlocks if the callback is destroyed under the lock.
  EXPECT_FALSE(other.has_value());
}

TEST(CancelScopeTest, CapturesOfACancelledCallbackCanUseTheScope) {
  CancelScope scope;
  std::optional<absl::StatusOr<CancelScope::Registration>> other;
  other.emplace(scope.OnCancel([] {}));
  ABSL_ASSERT_OK(*other);
  // Destroying this callback after it ran drops `other`, which locks the scope
  // again.
  absl::StatusOr<CancelScope::Registration> registration = scope.OnCancel(
      [drop_other = absl::MakeCleanup([&other] { other.reset(); })] {});
  ABSL_ASSERT_OK(registration);
  ASSERT_TRUE(other.has_value());

  scope.Cancel();  // Deadlocks if the callback is destroyed under the lock.
  EXPECT_FALSE(other.has_value());
}

TEST(CancelScopeTest, CapturesOfARefusedCallbackCanUseTheScope) {
  CancelScope scope;
  scope.Cancel();
  bool saw_cancelled = false;

  // Destroying the refused callback calls cancelled(), which locks the scope
  // again: deadlocks if the callback is destroyed under the lock.
  EXPECT_THAT(scope.OnCancel([check = absl::MakeCleanup([&] {
                                saw_cancelled = scope.cancelled();
                              })] {}),
              StatusIs(absl::StatusCode::kCancelled));
  EXPECT_TRUE(saw_cancelled);
}

TEST(CancelScopeTest, CallbacksRegisteredDuringCancelRunExactlyOnceOrNever) {
  constexpr int kThreads = 8;
  CancelScope scope;
  // A callback that OnCancel() accepts, and whose registration lives until
  // after Cancel(), must run exactly once; a refused one must never run; one
  // deregistered at once may run at most once.
  std::vector<std::atomic<int>> kept_runs(kThreads);
  std::vector<std::atomic<int>> dropped_runs(kThreads);
  std::vector<std::atomic<bool>> dropped_refused(kThreads);
  std::vector<std::optional<absl::StatusOr<CancelScope::Registration>>> kept(
      kThreads);
  // Released once every thread exists, so registrations race with Cancel().
  absl::Notification start;
  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i] {
      start.WaitForNotification();
      kept[i].emplace(scope.OnCancel([&kept_runs, i] { ++kept_runs[i]; }));
      absl::StatusOr<CancelScope::Registration> dropped =
          scope.OnCancel([&dropped_runs, i] { ++dropped_runs[i]; });
      dropped_refused[i] = absl::IsCancelled(dropped.status());
    });
  }
  threads.emplace_back([&start, &scope] {
    start.WaitForNotification();
    scope.Cancel();
  });
  start.Notify();
  for (std::thread& thread : threads) thread.join();

  for (int i = 0; i < kThreads; ++i) {
    if (kept[i]->ok()) {
      EXPECT_EQ(kept_runs[i].load(), 1) << "thread " << i;
    } else {
      EXPECT_THAT(*kept[i], StatusIs(absl::StatusCode::kCancelled))
          << "thread " << i;
      EXPECT_EQ(kept_runs[i].load(), 0) << "thread " << i;
    }
    EXPECT_LE(dropped_runs[i].load(), dropped_refused[i] ? 0 : 1)
        << "thread " << i;
  }
}

TEST(CancelScopeTest, DeregistrationWaitsForTheRunningCallback) {
  // How long the deregistration must still be blocked once it has started.
  constexpr absl::Duration kStillBlockedWindow = absl::Milliseconds(100);
  CancelScope scope;
  absl::Notification callback_started;
  absl::Notification release_callback;
  std::optional<absl::StatusOr<CancelScope::Registration>> registration;
  registration.emplace(scope.OnCancel([&] {
    callback_started.Notify();
    release_callback.WaitForNotification();
  }));
  ABSL_ASSERT_OK(*registration);
  std::thread canceller([&scope] { scope.Cancel(); });
  callback_started.WaitForNotification();

  absl::Notification deregistered;
  std::thread deregisterer([&] {
    registration.reset();
    deregistered.Notify();
  });
  // This window cannot fail falsely; on a stalled machine it can only pass
  // falsely.
  EXPECT_FALSE(
      deregistered.WaitForNotificationWithTimeout(kStillBlockedWindow));
  release_callback.Notify();
  deregistered.WaitForNotification();
  canceller.join();
  deregisterer.join();
}

TEST(CancelScopeTest, CallbackCanDestroyItsOwnRegistration) {
  CancelScope scope;
  bool ran = false;
  std::optional<absl::StatusOr<CancelScope::Registration>> registration;
  registration.emplace(scope.OnCancel([&ran, &registration] {
    // Runs on the cancelling thread, so it does not wait for itself.
    registration.reset();
    ran = true;
  }));
  ABSL_ASSERT_OK(*registration);

  scope.Cancel();
  EXPECT_TRUE(ran);
  EXPECT_FALSE(registration.has_value());
}

TEST(CancelScopeTest, CallbackCanCallBackIntoTheScope) {
  CancelScope scope;
  bool cancelled_in_callback = false;
  absl::Status inner_status;
  bool inner_ran = false;
  absl::StatusOr<CancelScope::Registration> outer = scope.OnCancel([&] {
    cancelled_in_callback = scope.cancelled();
    scope.Cancel();  // Returns at once.
    inner_status = scope.OnCancel([&inner_ran] { inner_ran = true; }).status();
  });
  ABSL_ASSERT_OK(outer);

  scope.Cancel();
  EXPECT_TRUE(cancelled_in_callback);
  EXPECT_THAT(inner_status, StatusIs(absl::StatusCode::kCancelled));
  EXPECT_FALSE(inner_ran);
}

TEST(CancelScopeTest, NestedCancelRunsNoOtherCallback) {
  CancelScope scope;
  int runs = 0;
  int depth = 0;
  int max_depth = 0;
  // The nested Cancel() returns at once, so the other callback runs after this
  // one returns, never inside it.
  const auto cancel_again = [&] {
    ++runs;
    max_depth = std::max(max_depth, ++depth);
    scope.Cancel();
    --depth;
  };
  absl::StatusOr<CancelScope::Registration> first =
      scope.OnCancel(cancel_again);
  ABSL_ASSERT_OK(first);
  absl::StatusOr<CancelScope::Registration> second =
      scope.OnCancel(cancel_again);
  ABSL_ASSERT_OK(second);

  scope.Cancel();
  EXPECT_EQ(runs, 2);
  EXPECT_EQ(max_depth, 1);
}

TEST(CancelScopeTest, DestructionCancelsAndWaitsForRegistrations) {
  // How long the destruction must still be blocked once the callback ran.
  constexpr absl::Duration kStillBlockedWindow = absl::Milliseconds(100);
  auto scope = std::make_unique<CancelScope>();
  absl::Notification callback_ran;
  std::optional<absl::StatusOr<CancelScope::Registration>> registration;
  registration.emplace(scope->OnCancel([&] { callback_ran.Notify(); }));
  ABSL_ASSERT_OK(*registration);

  absl::Notification destroyed;
  std::thread destroyer([&] {
    scope.reset();
    destroyed.Notify();
  });
  // The destructor cancels the scope, then waits for the registration.
  callback_ran.WaitForNotification();
  // This window cannot fail falsely; on a stalled machine it can only pass
  // falsely.
  EXPECT_FALSE(destroyed.WaitForNotificationWithTimeout(kStillBlockedWindow));
  registration.reset();
  destroyed.WaitForNotification();
  destroyer.join();
}

TEST(CancelScopeTest, DestructionWaitsForCancelOnAnotherThread) {
  // How long the destruction must still be blocked once it has started.
  constexpr absl::Duration kStillBlockedWindow = absl::Milliseconds(100);
  auto scope = std::make_unique<CancelScope>();
  absl::Notification callback_started;
  absl::Notification release_callback;
  std::optional<absl::StatusOr<CancelScope::Registration>> registration;
  registration.emplace(scope->OnCancel([&] {
    registration.reset();  // Leaves no registration to wait for.
    callback_started.Notify();
    release_callback.WaitForNotification();
  }));
  ABSL_ASSERT_OK(*registration);
  CancelScope* const raw_scope = scope.get();
  std::thread canceller([raw_scope] { raw_scope->Cancel(); });
  callback_started.WaitForNotification();

  absl::Notification destroyed;
  std::thread destroyer([&] {
    scope.reset();
    destroyed.Notify();
  });
  // This window cannot fail falsely; on a stalled machine it can only pass
  // falsely.
  EXPECT_FALSE(destroyed.WaitForNotificationWithTimeout(kStillBlockedWindow));
  release_callback.Notify();
  destroyed.WaitForNotification();
  canceller.join();
  destroyer.join();
}

TEST(CancelScopeTest, DestructionWaitsForTheCapturesOfADroppedCallback) {
  // How long the destruction must still be blocked once it has started.
  constexpr absl::Duration kStillBlockedWindow = absl::Milliseconds(100);
  auto scope = std::make_unique<CancelScope>();
  CancelScope* const raw_scope = scope.get();
  absl::Notification capture_destroying;
  absl::Notification release_capture;
  bool saw_cancelled = false;
  std::optional<absl::StatusOr<CancelScope::Registration>> registration;
  // Dropping the registration destroys this callback's capture, which blocks
  // and then uses the scope.
  registration.emplace(scope->OnCancel([cleanup = absl::MakeCleanup([&] {
                                          capture_destroying.Notify();
                                          release_capture.WaitForNotification();
                                          saw_cancelled =
                                              raw_scope->cancelled();
                                        })] {}));
  ABSL_ASSERT_OK(*registration);
  std::thread dropper([&] { registration.reset(); });
  capture_destroying.WaitForNotification();

  absl::Notification destroyed;
  std::thread destroyer([&] {
    scope.reset();
    destroyed.Notify();
  });
  // This window cannot fail falsely; on a stalled machine it can only pass
  // falsely.
  EXPECT_FALSE(destroyed.WaitForNotificationWithTimeout(kStillBlockedWindow));
  release_capture.Notify();
  destroyed.WaitForNotification();
  dropper.join();
  destroyer.join();
  EXPECT_TRUE(saw_cancelled);
}

TEST(CancelScopeDeathTest, DestroyingTheScopeFromItsOwnCallbackFails) {
  EXPECT_DEATH(
      {
        auto scope = std::make_unique<CancelScope>();
        std::optional<absl::StatusOr<CancelScope::Registration>> registration;
        registration.emplace(scope->OnCancel([&] {
          registration.reset();
          scope.reset();
        }));
        scope->Cancel();
      },
      "CancelScope destroyed by one of its own callbacks");
}

}  // namespace
}  // namespace tpu_raiden
