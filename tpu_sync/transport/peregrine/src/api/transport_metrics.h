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

// Per-operation (read or write) end-to-end metrics.
struct OpMetrics final {
  // End-to-end transfer duration in microseconds.
  Log2Histogram<32> e2e_latency_us{};
  // Request per-op size in bytes.
  Log2Histogram<32> request_size_bytes{};
  // Total payload bytes transferred across all data channels.
  uint64_t bytes = 0;
  // Total transfer failures.
  uint64_t errors = 0;
};

// High-level end-to-end metrics for transport users.
struct TransportMetrics final {
  OpMetrics write{};
  OpMetrics read{};

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
