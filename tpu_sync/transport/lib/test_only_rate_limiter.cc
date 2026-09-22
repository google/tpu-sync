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

#include <cstddef>

#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tpu_raiden::transport::lib {

absl::Time TestOnlyRateLimiter::Update(size_t bytes) {
  const absl::Duration required_time =
      absl::Seconds(static_cast<double>(bytes) * sec_per_byte_);
  absl::MutexLock lock(mu_);
  const absl::Time now = absl::Now();
  if (next_available_time_ < now) {
    next_available_time_ = now;
  }
  next_available_time_ += required_time;
  return next_available_time_;
}

void TestOnlyRateLimiter::Consume(size_t bytes) {
  if (bandwidth_bytes_per_sec_ == 0 || bytes == 0) {
    return;
  }
  const absl::Time sleep_until = Update(bytes);
  const absl::Time now = absl::Now();
  if (sleep_until > now) {
    absl::SleepFor(sleep_until - now);
  }
}

}  // namespace tpu_raiden::transport::lib
