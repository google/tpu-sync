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

TEST(HostMeminfoTest, ParseNumaNodeMeminfoComputesStatesAndFallback) {
  // Test case 1: Standard sysfs meminfo containing MemTotal, MemFree, MemUsed.
  constexpr absl::string_view kStandardMeminfo = R"(
Node 0 MemTotal:       131919728 kB
Node 0 MemFree:        120394812 kB
Node 0 MemUsed:         11524916 kB
Node 0 Active:           1000000 kB
)";
  const std::optional<NumaNodeMemory> mem0 =
      ParseNumaNodeMeminfo(/*node_id=*/0, kStandardMeminfo);
  ASSERT_TRUE(mem0.has_value());
  EXPECT_EQ(mem0->node_id, 0);
  EXPECT_EQ(mem0->used_bytes, 11524916ULL * 1024ULL);
  EXPECT_EQ(mem0->free_bytes, 120394812ULL * 1024ULL);
  EXPECT_EQ(mem0->total_bytes, 131919728ULL * 1024ULL);

  // Test case 2: MemUsed absent -> fallback to MemTotal - MemFree.
  constexpr absl::string_view kFallbackMeminfo = R"(
Node 1 MemTotal:        1000 kB
Node 1 MemFree:          400 kB
)";
  const std::optional<NumaNodeMemory> mem1 =
      ParseNumaNodeMeminfo(/*node_id=*/1, kFallbackMeminfo);
  ASSERT_TRUE(mem1.has_value());
  EXPECT_EQ(mem1->node_id, 1);
  EXPECT_EQ(mem1->used_bytes, 600ULL * 1024ULL);
  EXPECT_EQ(mem1->free_bytes, 400ULL * 1024ULL);
  EXPECT_EQ(mem1->total_bytes, 1000ULL * 1024ULL);

  // Test case 3: MemUsed absent and MemFree > MemTotal -> fallback
  // clamped to 0.
  constexpr absl::string_view kClampedMeminfo = R"(
Node 2 MemTotal:         200 kB
Node 2 MemFree:          500 kB
)";
  const std::optional<NumaNodeMemory> mem2 =
      ParseNumaNodeMeminfo(/*node_id=*/2, kClampedMeminfo);
  ASSERT_TRUE(mem2.has_value());
  EXPECT_EQ(mem2->node_id, 2);
  EXPECT_EQ(mem2->used_bytes, 0ULL);
  EXPECT_EQ(mem2->free_bytes, 500ULL * 1024ULL);
  EXPECT_EQ(mem2->total_bytes, 200ULL * 1024ULL);

  // Test case 4: Missing MemFree -> returns std::nullopt.
  constexpr absl::string_view kMissingFree = "Node 3 MemTotal:       1000 kB\n";
  EXPECT_FALSE(ParseNumaNodeMeminfo(3, kMissingFree).has_value());

  // Test case 5: Missing MemTotal -> returns std::nullopt.
  constexpr absl::string_view kMissingTotal = "Node 4 MemFree:        500 kB\n";
  EXPECT_FALSE(ParseNumaNodeMeminfo(4, kMissingTotal).has_value());

  // Test case 6: Malformed number -> returns std::nullopt.
  constexpr absl::string_view kMalformed = R"(
Node 5 MemTotal:       not_a_number kB
Node 5 MemFree:        500 kB
)";
  EXPECT_FALSE(ParseNumaNodeMeminfo(5, kMalformed).has_value());

  // Test case 7: Empty content -> returns std::nullopt.
  EXPECT_FALSE(ParseNumaNodeMeminfo(6, "").has_value());
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
