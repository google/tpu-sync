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

#include <type_traits>
#include "tpu_sync/kv_cache/lru_cache.h"

#include <optional>
#include <string>
#include <utility>

#include <gtest/gtest.h>

namespace tpu_raiden {
namespace kv_cache {
namespace {

TEST(LRUCacheTest, BasicPutAndGet) {
  LRUCache<int, std::string> cache(3);

  EXPECT_TRUE(cache.empty());
  EXPECT_EQ(cache.size(), 0);
  EXPECT_EQ(cache.capacity(), 3);

  EXPECT_EQ(cache.Put(1, "one"), std::nullopt);
  EXPECT_EQ(cache.Put(2, "two"), std::nullopt);
  EXPECT_EQ(cache.Put(3, "three"), std::nullopt);

  EXPECT_EQ(cache.size(), 3);
  EXPECT_TRUE(cache.Contains(1));
  EXPECT_TRUE(cache.Contains(2));
  EXPECT_TRUE(cache.Contains(3));

  EXPECT_EQ(*cache.Get(1), "one");
  EXPECT_EQ(*cache.Get(2), "two");
  EXPECT_EQ(*cache.Get(3), "three");
}

TEST(LRUCacheTest, EvictionOnOverfull) {
  LRUCache<int, std::string> cache(2);

  cache.Put(1, "one");
  cache.Put(2, "two");

  // Get(1) promotes 1 to MRU, making 2 the LRU
  cache.Get(1);

  // Put(3) should evict 2
  auto evicted = cache.Put(3, "three");
  ASSERT_TRUE(evicted.has_value());
  EXPECT_EQ(evicted->first, 2);
  EXPECT_EQ(evicted->second, "two");

  EXPECT_EQ(cache.size(), 3);
  EXPECT_EQ(cache.active_size(), 2);
  EXPECT_TRUE(cache.Contains(1));
  EXPECT_FALSE(cache.Contains(2));  // 2 is candidate
  EXPECT_TRUE(cache.Contains(3));
  auto candidates = cache.GetEvictCandidateKeys();
  ASSERT_EQ(candidates.size(), 1);
  EXPECT_EQ(candidates[0], 2);
}

TEST(LRUCacheTest, PinningProtectsAgainstEviction) {
  LRUCache<int, std::string> cache(2);

  cache.Put(1, "one");
  cache.Put(2, "two");

  // Pin 2 so it cannot be evicted
  EXPECT_TRUE(cache.Pin(2));
  EXPECT_EQ(cache.GetPinCount(2), 1);

  // Get(2) promotes 2, making 1 the LRU
  // Put(3) should evict 1 because 2 is pinned
  auto evicted = cache.Put(3, "three");
  ASSERT_TRUE(evicted.has_value());
  EXPECT_EQ(evicted->first, 1);
  EXPECT_EQ(evicted->second, "one");

  EXPECT_EQ(cache.size(), 3);
  EXPECT_EQ(cache.active_size(), 2);
  EXPECT_FALSE(cache.Contains(1));  // 1 is candidate
  EXPECT_TRUE(cache.Contains(2));
  EXPECT_TRUE(cache.Contains(3));

  // Unpin 2. Since it is now unpinned, it is promoted to MRU.
  // 3 (inserted earlier) now becomes the LRU.
  EXPECT_TRUE(cache.Unpin(2));
  EXPECT_EQ(cache.GetPinCount(2), 0);

  auto evicted_unpinned = cache.Put(4, "four");
  ASSERT_TRUE(evicted_unpinned.has_value());
  EXPECT_EQ(evicted_unpinned->first,
            3);  // 3 is now LRU because 2 was promoted on unpin
}

TEST(LRUCacheTest, InsertionRejectedWhenAllPinned) {
  LRUCache<int, std::string> cache(2);

  cache.Put(101, "one");
  cache.Put(102, "two");

  EXPECT_TRUE(cache.Pin(101));
  EXPECT_TRUE(cache.Pin(102));

  auto evicted = cache.Put(103, "three");
  EXPECT_EQ(evicted, std::nullopt);
  EXPECT_FALSE(cache.Contains(103));
  EXPECT_EQ(cache.size(), 2);
}

TEST(LRUCacheTest, PeekDoesNotPromote) {
  LRUCache<int, std::string> cache(2);

  cache.Put(1, "one");
  cache.Put(2, "two");

  // Peek(1) should NOT promote 1 to MRU
  EXPECT_EQ(*cache.Peek(1), "one");

  // Put(3) should evict 1 (since 1 is still LRU)
  auto evicted = cache.Put(3, "three");
  ASSERT_TRUE(evicted.has_value());
  EXPECT_EQ(evicted->first, 1);
}

TEST(LRUCacheTest, EraseAndClear) {
  LRUCache<int, std::string> cache(2);

  cache.Put(1, "one");
  cache.Put(2, "two");

  EXPECT_TRUE(cache.Erase(1));
  EXPECT_FALSE(cache.Contains(1));
  EXPECT_EQ(cache.size(), 1);

  cache.Clear();
  EXPECT_TRUE(cache.empty());
  EXPECT_EQ(cache.size(), 0);
}

TEST(LRUCacheTest, PutBack) {
  LRUCache<int, std::string> cache(3);
  cache.Put(1, "one");
  cache.Put(2, "two");
  // MRU to LRU is: 2, 1.

  // PutBack(3) should add 3 to the back (LRU position).
  cache.PutBack(3, "three");
  // Now MRU to LRU should be: 2, 1, 3.
  // Verify by checking Evict(): first evicted item should be 3.
  auto evicted1 = cache.Evict();
  ASSERT_TRUE(evicted1.has_value());
  EXPECT_EQ(evicted1->first, 3);

  auto evicted2 = cache.Evict();
  ASSERT_TRUE(evicted2.has_value());
  EXPECT_EQ(evicted2->first, 1);

  auto evicted3 = cache.Evict();
  ASSERT_TRUE(evicted3.has_value());
  EXPECT_EQ(evicted3->first, 2);
}

TEST(LRUCacheTest, RescueFromCandidateList) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");
  cache.Put(2, "two");

  // Evict will move 1 to candidate list (since 1 is LRU)
  auto evicted = cache.Evict();
  ASSERT_TRUE(evicted.has_value());
  EXPECT_EQ(evicted->first, 1);

  // 1 is in candidate list, size (total including candidates) is still 2
  EXPECT_EQ(cache.size(), 2);
  EXPECT_EQ(cache.active_size(), 1);

  auto candidates = cache.GetEvictCandidateKeys();
  ASSERT_EQ(candidates.size(), 1);
  EXPECT_EQ(candidates[0], 1);

  // Get(1) should rescue 1
  auto* val = cache.Get(1);
  ASSERT_NE(val, nullptr);
  EXPECT_EQ(*val, "one");

  EXPECT_EQ(cache.active_size(), 2);
  EXPECT_TRUE(cache.GetEvictCandidateKeys().empty());

  // Since 1 was rescued, it becomes MRU. 2 is now LRU.
  // Putting 3 should evict 2.
  auto evicted2 = cache.Put(3, "three");
  ASSERT_TRUE(evicted2.has_value());
  EXPECT_EQ(evicted2->first, 2);

  auto candidates2 = cache.GetEvictCandidateKeys();
  ASSERT_EQ(candidates2.size(), 1);
  EXPECT_EQ(candidates2[0], 2);
}

TEST(LRUCacheTest, PinCandidateAndRescue) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");
  cache.Put(2, "two");

  cache.Evict();  // 1 becomes candidate

  auto candidates = cache.GetEvictCandidateKeys();
  ASSERT_EQ(candidates.size(), 1);
  EXPECT_EQ(candidates[0], 1);
  EXPECT_EQ(cache.active_size(), 1);

  // Pinning 1 should rescue and pin it
  EXPECT_TRUE(cache.Pin(1));
  EXPECT_EQ(cache.GetPinCount(1), 1);
  EXPECT_EQ(cache.active_size(), 2);
  EXPECT_TRUE(cache.GetEvictCandidateKeys().empty());

  // 1 is pinned, 2 is unpinned. Putting 3 should evict 2.
  auto evicted = cache.Put(3, "three");
  ASSERT_TRUE(evicted.has_value());
  EXPECT_EQ(evicted->first, 2);
}

TEST(LRUCacheTest, EraseCandidate) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");
  cache.Put(2, "two");

  cache.Evict();  // 1 becomes candidate
  auto candidates = cache.GetEvictCandidateKeys();
  ASSERT_EQ(candidates.size(), 1);
  EXPECT_EQ(candidates[0], 1);
  EXPECT_EQ(cache.size(), 2);

  // Erasing candidate 1
  EXPECT_TRUE(cache.Erase(1));
  EXPECT_EQ(cache.size(), 1);
  EXPECT_TRUE(cache.GetEvictCandidateKeys().empty());
  EXPECT_FALSE(cache.Contains(1));
}

TEST(LRUCacheTest, GetEvictableKeysScanningOrder) {
  LRUCache<int, std::string> cache(3);
  cache.Put(1, "one");
  cache.Put(2, "two");
  cache.Put(3, "three");
  // Active LRU: 3, 2, 1 (MRU to LRU)

  cache.Evict();  // 1 becomes candidate
  // Candidates: 1
  // Active: 3, 2

  cache.Put(4, "four");
  // Candidates: 1
  // Active: 4, 3, 2 (MRU to LRU)

  cache.Evict();  // 2 becomes candidate
  // Candidates: 1, 2 (1 is older)
  // Active: 4, 3 (3 is older active)

  auto keys = cache.GetEvictableKeys(4);
  ASSERT_EQ(keys.size(), 4);
  EXPECT_EQ(keys[0], 1);  // Oldest candidate
  EXPECT_EQ(keys[1], 2);  // Newer candidate
  EXPECT_EQ(keys[2], 3);  // Oldest active
  EXPECT_EQ(keys[3], 4);  // Newer active
}

TEST(LRUCacheTest, CandidateVisibility) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");
  cache.Put(2, "two");

  cache.Evict();  // 1 becomes candidate

  // 1 should not be visible to Contains, Peek, PeekMutable
  EXPECT_FALSE(cache.Contains(1));
  EXPECT_EQ(cache.Peek(1), nullptr);
  EXPECT_EQ(cache.PeekMutable(1), nullptr);

  // 1 should still be visible to PeekIncludingCandidates and
  // PeekMutableIncludingCandidates
  EXPECT_NE(cache.PeekIncludingCandidates(1), nullptr);
  EXPECT_EQ(*cache.PeekIncludingCandidates(1), "one");
  EXPECT_NE(cache.PeekMutableIncludingCandidates(1), nullptr);
  EXPECT_EQ(*cache.PeekMutableIncludingCandidates(1), "one");

  // 2 (active) should be visible to all
  EXPECT_TRUE(cache.Contains(2));
  EXPECT_NE(cache.Peek(2), nullptr);
  EXPECT_EQ(*cache.Peek(2), "two");
  EXPECT_NE(cache.PeekMutable(2), nullptr);
  EXPECT_EQ(*cache.PeekMutable(2), "two");
}

TEST(LRUCacheTest, GetAndPinBasic) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");
  cache.Put(2, "two");

  // Non-existent key
  EXPECT_EQ(cache.GetAndPin(99), nullptr);

  // Existing key: returns value pointer and pins item
  auto* val1 = cache.GetAndPin(1);
  ASSERT_NE(val1, nullptr);
  EXPECT_EQ(*val1, "one");
  EXPECT_EQ(cache.GetPinCount(1), 1);

  // Calling GetAndPin again increments pin count further
  auto* val1_again = cache.GetAndPin(1);
  ASSERT_NE(val1_again, nullptr);
  EXPECT_EQ(*val1_again, "one");
  EXPECT_EQ(cache.GetPinCount(1), 2);

  // 1 is pinned twice, 2 is unpinned. Put(3) evicts 2.
  auto evicted = cache.Put(3, "three");
  ASSERT_TRUE(evicted.has_value());
  EXPECT_EQ(evicted->first, 2);
  // 2 is now a candidate
  EXPECT_FALSE(cache.Contains(2));

  // GetAndPin on candidate 2 rescues and pins it
  auto* val2 = cache.GetAndPin(2);
  ASSERT_NE(val2, nullptr);
  EXPECT_EQ(*val2, "two");
  EXPECT_EQ(cache.GetPinCount(2), 1);
  EXPECT_TRUE(cache.Contains(2));
}

TEST(LRUCacheTest, PutBackAddsAtTheLruEnd) {
  LRUCache<std::string, int> cache(3);
  cache.Put("A", 1);
  cache.Put("B", 2);
  // MRU to LRU is: B, A.

  // PutBack("C", 3) should add C to the back (LRU position).
  cache.PutBack("C", 3);
  // Now MRU to LRU should be: B, A, C.
  // Let's verify by checking Evict(): first evicted item should be C!
  auto evicted1 = cache.Evict();
  ASSERT_TRUE(evicted1.has_value());
  EXPECT_EQ(evicted1->first, "C");

  auto evicted2 = cache.Evict();
  ASSERT_TRUE(evicted2.has_value());
  EXPECT_EQ(evicted2->first, "A");

  auto evicted3 = cache.Evict();
  ASSERT_TRUE(evicted3.has_value());
  EXPECT_EQ(evicted3->first, "B");
}

// available_space() is a GUARD (host_offload_backend.cc:399, :883) and is also
// published to peers as free_blocks (kv_cache_store.cc:335). pinned_list_ can
// exceed capacity_ because pinning an eviction candidate resurrects it without
// a capacity check, so the subtraction must not wrap to SIZE_MAX.
TEST(LRUCacheTest, AvailableSpaceClampsInsteadOfUnderflowing) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "a");
  cache.Put(2, "b");
  cache.Put(3, "c");  // displaces 1 to the candidate list
  cache.Put(4, "d");  // displaces 2 to the candidate list

  cache.Pin(3);
  cache.Pin(4);
  // Pinning a CANDIDATE resurrects it straight into pinned_list_, which is how
  // pinned_list_ grows past capacity_.
  cache.Pin(1);
  cache.Pin(2);
  // Whether or not resurrection is later made capacity-aware, this must hold:
  // the subtraction must clamp, never wrap.
  EXPECT_LE(cache.available_space(), cache.capacity());
}

// Verifies that Erase returns false on pinned entries and succeeds once unpinned.
TEST(LRUCacheTest, EraseRefusesAPinnedEntry) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");
  ASSERT_TRUE(cache.Pin(1));
  ASSERT_EQ(cache.GetPinCount(1), 1);

  EXPECT_FALSE(cache.Erase(1));
  EXPECT_TRUE(cache.Contains(1));
  EXPECT_EQ(*cache.Peek(1), "one");
  EXPECT_EQ(cache.GetPinCount(1), 1);

  // Once the last pin goes, the entry is erasable again.
  EXPECT_TRUE(cache.Unpin(1));
  EXPECT_TRUE(cache.Erase(1));
  EXPECT_FALSE(cache.Contains(1));
}

// Verifies that unpinned eviction candidates remain erasable.
TEST(LRUCacheTest, EraseStillTakesAnUnpinnedCandidate) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");
  cache.Put(2, "two");
  ASSERT_TRUE(cache.Evict().has_value());        // 1 becomes a candidate
  ASSERT_EQ(cache.GetPinCount(1), 0);

  EXPECT_TRUE(cache.Erase(1));
  EXPECT_TRUE(cache.GetEvictCandidateKeys().empty());
}

// Verifies that Peek on a const LRUCache reference returns a const pointer.
TEST(LRUCacheTest, ConstPeekEnforcesConstAccess) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");
  const auto& const_cache = cache;
  const std::string* val = const_cache.Peek(1);
  ASSERT_NE(val, nullptr);
  EXPECT_EQ(*val, "one");
  static_assert(std::is_same_v<decltype(const_cache.Peek(1)), const std::string*>);
  static_assert(std::is_same_v<decltype(cache.Peek(1)), const std::string*>);
}

enum class TestBlockStatus {
  INIT,
  REMOTE,
  HBM,
  HOST,
  HOST_AND_HBM,
  SHARED_STORAGE,
};

struct TestBlockValue {
  int host_block_id = -1;
  TestBlockStatus status = TestBlockStatus::INIT;
};

TEST(LRUCacheTest, NumEvictableHostBlocksTracksUnpinnedHostEntries) {
  LRUCache<std::string, TestBlockValue> cache(2);
  EXPECT_EQ(cache.NumEvictableHostBlocks(), 0);

  // Dead HBM-only entry (e.g. before or after failed save) does not count.
  cache.Put("hbm_only", TestBlockValue{-1, TestBlockStatus::HBM});
  EXPECT_EQ(cache.NumEvictableHostBlocks(), 0);

  // Host-resident entries count when unpinned.
  cache.Put("host_1", TestBlockValue{0, TestBlockStatus::HOST});
  EXPECT_EQ(cache.NumEvictableHostBlocks(), 1);

  // Displacing hbm_only to candidate list and adding host_2.
  cache.Put("host_2", TestBlockValue{1, TestBlockStatus::HOST_AND_HBM});
  EXPECT_EQ(cache.NumEvictableHostBlocks(), 2);

  // Pinning a host-resident entry removes it from evictable count.
  ASSERT_TRUE(cache.Pin("host_1"));
  EXPECT_EQ(cache.NumEvictableHostBlocks(), 1);
  ASSERT_TRUE(cache.Pin("host_1"));
  EXPECT_EQ(cache.NumEvictableHostBlocks(), 1);

  // Unpinning once keeps pin_count=1; second unpin restores evictability.
  ASSERT_TRUE(cache.Unpin("host_1"));
  EXPECT_EQ(cache.NumEvictableHostBlocks(), 1);
  ASSERT_TRUE(cache.Unpin("host_1"));
  EXPECT_EQ(cache.NumEvictableHostBlocks(), 2);

  // Demoting to candidate list preserves evictable host block count until
  // Erase.
  ASSERT_TRUE(cache.Evict().has_value());
  EXPECT_EQ(cache.NumEvictableHostBlocks(), 2);

  // Updating an unpinned entry from HOST_AND_HBM to HBM decrements count.
  ASSERT_TRUE(
      cache.UpdateValue("host_1", TestBlockValue{-1, TestBlockStatus::HBM}));
  EXPECT_EQ(cache.NumEvictableHostBlocks(), 1);
  ASSERT_TRUE(
      cache.UpdateValue("host_1", TestBlockValue{0, TestBlockStatus::HOST}));
  EXPECT_EQ(cache.NumEvictableHostBlocks(), 2);

  // Erasing candidate and active entries decrements count.
  ASSERT_TRUE(cache.Erase("host_2"));
  EXPECT_EQ(cache.NumEvictableHostBlocks(), 1);

  cache.Clear();
  EXPECT_EQ(cache.NumEvictableHostBlocks(), 0);
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
