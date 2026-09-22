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

#include "tpu_sync/transport/lib/test_only_rate_limiter.h"

#include <thread>  // NOLINT

#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tpu_raiden::transport::lib {
namespace {

TEST(TestOnlyRateLimiterTest, ZeroBandwidthDoesNotSleep) {
  TestOnlyRateLimiter limiter(0);
  absl::Time start = absl::Now();
  limiter.Consume(1024 * 1024);
  absl::Duration elapsed = absl::Now() - start;
  EXPECT_LT(elapsed, absl::Milliseconds(10));
}

TEST(TestOnlyRateLimiterTest, ZeroBytesDoesNotSleep) {
  TestOnlyRateLimiter limiter(1000);
  absl::Time start = absl::Now();
  limiter.Consume(0);
  absl::Duration elapsed = absl::Now() - start;
  EXPECT_LT(elapsed, absl::Milliseconds(10));
}

TEST(TestOnlyRateLimiterTest, SequentialPacingAccuracy) {
  // 100,000 bytes per second = 100 bytes / millisecond
  TestOnlyRateLimiter limiter(100000);
  absl::Time start = absl::Now();
  limiter.Consume(5000);  // 50ms
  absl::Duration elapsed = absl::Now() - start;
  EXPECT_GE(elapsed, absl::Milliseconds(40));
  EXPECT_LE(elapsed, absl::Milliseconds(150));
}

TEST(TestOnlyRateLimiterTest, ConcurrentTimelineAdvancement) {
  // 1,000,000 bytes per second. Two threads consume 25,000 bytes each (50,000
  // bytes total = 50ms).
  TestOnlyRateLimiter limiter(1000000);
  absl::Time start = absl::Now();
  std::thread t1([&] { limiter.Consume(25000); });
  std::thread t2([&] { limiter.Consume(25000); });
  t1.join();
  t2.join();
  absl::Duration elapsed = absl::Now() - start;
  EXPECT_GE(elapsed, absl::Milliseconds(40));
  EXPECT_LE(elapsed, absl::Milliseconds(200));
}

}  // namespace
}  // namespace tpu_raiden::transport::lib
