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

#ifndef THIRD_PARTY_PEREGRINE_SRC_API_TRANSPORT_METRICS_H_
#define THIRD_PARTY_PEREGRINE_SRC_API_TRANSPORT_METRICS_H_

#include <array>
#include <cstdint>
#include <numeric>
#include <string>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

namespace peregrine {

// A log2-spaced metric histogram.
// Sample value vs bucket mapping (N = #buckets):
//    Value Rrange      Bucket
//    [0, 1)            0
//    [1, 2)            1
//    [2, 4)            2
//    ...
//    [2^(i-1), 2^i)    i
//    ...
//    [2^(N-2), inf)    N-1
template <int N = 32>
struct Log2Histogram final {
  uint64_t sum = 0;
  std::array<uint64_t, N> buckets{};

  constexpr int NumBuckets() const { return N; }

  constexpr uint64_t Count() const {
    return std::accumulate(buckets.begin(), buckets.end(), uint64_t{0});
  }

  std::string ToString() const {
    return absl::StrFormat("Log2Histogram(sum: %d, count: %d, buckets: [%s])",
                           sum, Count(), absl::StrJoin(buckets, ", "));
  }
};

// High-level end-to-end metrics for transport users.
struct TransportMetrics final {
  // End-to-end write duration in microseconds.
  Log2Histogram<32> e2e_write_latency_us{};
  // Total payload bytes sent across all data channels.
  uint64_t bytes_sent = 0;
  // Total user transfer requests submitted
  uint64_t requests_posted = 0;
  // Total transfer write failures.
  uint64_t e2e_write_errors = 0;

  // Transport Pipeline Breakdown
  // ---------------------------------------------------------------------------
  // TODO(yyd): Add tier 2 metrics here

  // Hardware & Subsystem Internals
  // ---------------------------------------------------------------------------
  // Number of failed TCP connection attempts to peers
  uint64_t tcp_connect_failures = 0;
  // Number of RPC requests received by the control plane
  uint64_t rpc_requests_received = 0;
};

}  // namespace peregrine

#endif  // THIRD_PARTY_PEREGRINE_SRC_API_TRANSPORT_METRICS_H_
