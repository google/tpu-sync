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

#include "tpu_sync/common/accept_backoff.h"

#include <cerrno>

#include <gtest/gtest.h>
#include "absl/time/time.h"

namespace tpu_raiden {
namespace {

TEST(ClassifyAcceptErrorTest, RoutineErrorsRetryImmediately) {
  EXPECT_EQ(ClassifyAcceptError(EINTR), AcceptErrorAction::kRetry);
  EXPECT_EQ(ClassifyAcceptError(ECONNABORTED), AcceptErrorAction::kRetry);
}

TEST(ClassifyAcceptErrorTest, ResourceExhaustionBacksOff) {
  EXPECT_EQ(ClassifyAcceptError(EMFILE), AcceptErrorAction::kBackoff);
  EXPECT_EQ(ClassifyAcceptError(ENFILE), AcceptErrorAction::kBackoff);
  EXPECT_EQ(ClassifyAcceptError(ENOBUFS), AcceptErrorAction::kBackoff);
  EXPECT_EQ(ClassifyAcceptError(ENOMEM), AcceptErrorAction::kBackoff);
}

TEST(ClassifyAcceptErrorTest, UnusableDescriptorStops) {
  EXPECT_EQ(ClassifyAcceptError(EBADF), AcceptErrorAction::kStop);
  EXPECT_EQ(ClassifyAcceptError(ENOTSOCK), AcceptErrorAction::kStop);
  // What shutdown() on a listening socket reports, so teardown relies on it.
  EXPECT_EQ(ClassifyAcceptError(EINVAL), AcceptErrorAction::kStop);
}

TEST(ClassifyAcceptErrorTest, UnknownErrorsBackOffRatherThanSpin) {
  EXPECT_EQ(ClassifyAcceptError(EIO), AcceptErrorAction::kBackoff);
}

TEST(NextAcceptBackoffTest, GrowsFromZero) {
  const absl::Duration first = NextAcceptBackoff(absl::ZeroDuration());
  EXPECT_GT(first, absl::ZeroDuration());
  EXPECT_GT(NextAcceptBackoff(first), first);
}

TEST(NextAcceptBackoffTest, CapsLowEnoughNotToDelayTeardown) {
  absl::Duration delay = absl::ZeroDuration();
  for (int i = 0; i < 30; ++i) {
    delay = NextAcceptBackoff(delay);
  }
  EXPECT_EQ(delay, NextAcceptBackoff(delay));
  EXPECT_LE(delay, absl::Seconds(1));
}

TEST(AcceptBackoffTest, KeepsLoopRunningForRoutineErrors) {
  AcceptBackoff backoff("test");
  EXPECT_TRUE(backoff.OnError(EINTR));
  // Routine errors must not slow the loop down.
  EXPECT_EQ(backoff.delay_for_testing(), absl::ZeroDuration());
}

TEST(AcceptBackoffTest, LeavesLoopWhenDescriptorIsUnusable) {
  AcceptBackoff backoff("test");
  EXPECT_FALSE(backoff.OnError(EBADF));
}

TEST(AcceptBackoffTest, GrowsDelayWhileFailingAndResetsOnSuccess) {
  AcceptBackoff backoff("test");

  EXPECT_TRUE(backoff.OnError(EMFILE));
  const absl::Duration first = backoff.delay_for_testing();
  EXPECT_GT(first, absl::ZeroDuration());

  EXPECT_TRUE(backoff.OnError(EMFILE));
  EXPECT_GT(backoff.delay_for_testing(), first);

  backoff.OnSuccess();
  EXPECT_EQ(backoff.delay_for_testing(), absl::ZeroDuration());
}

}  // namespace
}  // namespace tpu_raiden
