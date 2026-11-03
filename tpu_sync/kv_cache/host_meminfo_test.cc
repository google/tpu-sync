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

#include "tpu_sync/kv_cache/host_meminfo.h"

#include <cstdint>
#include <optional>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

TEST(HostMeminfoTest, ParseHostMemoryBreakdownComputesComponentsAndClamps) {
  constexpr absl::string_view kSyntheticMeminfo = R"(
MemTotal:       16384000 kB
MemFree:         4096000 kB
MemAvailable:    6144000 kB
Buffers:          512000 kB
Cached:          2048000 kB
AnonPages:       3072000 kB
)";
  constexpr uint64_t kKvPoolBytes = 1024ULL * 1024ULL * 1024ULL;  // 1 GiB
  const std::optional<HostMemoryBreakdown> breakdown =
      ParseHostMemoryBreakdown(kSyntheticMeminfo, kKvPoolBytes);
  ASSERT_TRUE(breakdown.has_value());
  EXPECT_EQ(breakdown->kv_pool_bytes, kKvPoolBytes);
  EXPECT_EQ(breakdown->process_heap_bytes, 3072000ULL * 1024ULL - kKvPoolBytes);
  EXPECT_EQ(breakdown->page_cache_bytes, (512000ULL + 2048000ULL) * 1024ULL);
  EXPECT_EQ(breakdown->free_bytes, 4096000ULL * 1024ULL);

  // When kv_pool_bytes exceeds AnonPages (e.g. shared memory backed pool),
  // process_heap_bytes is clamped to 0.
  const std::optional<HostMemoryBreakdown> clamped =
      ParseHostMemoryBreakdown(kSyntheticMeminfo, 4000000ULL * 1024ULL);
  ASSERT_TRUE(clamped.has_value());
  EXPECT_EQ(clamped->kv_pool_bytes, 4000000ULL * 1024ULL);
  EXPECT_EQ(clamped->process_heap_bytes, 0ULL);

  // Missing required fields or malformed numbers return std::nullopt.
  EXPECT_FALSE(ParseHostMemoryBreakdown("MemFree: 100 kB\nCached: 200 kB\n", 0)
                   .has_value());
  EXPECT_FALSE(
      ParseHostMemoryBreakdown(
          "MemFree: invalid kB\nCached: 200 kB\nAnonPages: 300 kB\n", 0)
          .has_value());
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
