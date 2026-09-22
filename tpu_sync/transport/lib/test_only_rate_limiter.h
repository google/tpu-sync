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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_TEST_ONLY_RATE_LIMITER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_TEST_ONLY_RATE_LIMITER_H_

#include <cstddef>
#include <cstdint>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace tpu_raiden::transport::lib {

class TestOnlyRateLimiter {
 public:
  explicit TestOnlyRateLimiter(uint64_t bandwidth_bytes_per_sec)
      : bandwidth_bytes_per_sec_(bandwidth_bytes_per_sec),
        sec_per_byte_(bandwidth_bytes_per_sec > 0
                          ? 1.0 / static_cast<double>(bandwidth_bytes_per_sec)
                          : 0.0) {}

  uint64_t bandwidth_bytes_per_sec() const { return bandwidth_bytes_per_sec_; }

  void Consume(size_t bytes);

 private:
  absl::Time Update(size_t bytes);

  const uint64_t bandwidth_bytes_per_sec_;
  const double sec_per_byte_;
  absl::Mutex mu_;
  absl::Time next_available_time_ ABSL_GUARDED_BY(mu_) = absl::InfinitePast();
};

}  // namespace tpu_raiden::transport::lib

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_TEST_ONLY_RATE_LIMITER_H_
