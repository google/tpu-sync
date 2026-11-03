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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_LRU_CACHE_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_LRU_CACHE_H_

#include <cstddef>
#include <list>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"

namespace tpu_raiden {
namespace kv_cache {

// A highly optimized generic LRU cache tailored for managing Key-Value cache
// block eviction and promotion.
//
// Key features:
// - Supports explicit pinning (`Pin` / `Unpin`) to lock actively executing
//   blocks in memory during transformer attention passes so they cannot be
//   evicted.
// - Features exact MRU promotion on `Get` and fast LRU eviction on `Evict`.
// - Exposes `Peek` to read values without perturbing LRU order.
enum class NodeLocation { kLru, kPinned, kCandidate };

template <typename Key, typename Value>
class LRUCache {
 private:
  struct CacheNode;

 public:
  explicit LRUCache(size_t capacity) : capacity_(capacity) {}

  ~LRUCache() { Clear(); }

  LRUCache(const LRUCache&) = delete;
  LRUCache& operator=(const LRUCache&) = delete;

  LRUCache(LRUCache&& other) noexcept = default;
  LRUCache& operator=(LRUCache&& other) noexcept = default;

  // Returns the maximum capacity of the cache.
  size_t capacity() const { return capacity_; }

  // Returns the total number of elements stored in the cache (including
  // eviction candidates).
  size_t size() const { return map_.size(); }

  // Returns the available space (free + evictable capacity), clamped to 0.
  size_t available_space() const {
    return capacity_ > pinned_list_.size() ? capacity_ - pinned_list_.size()
                                           : 0;
  }
  size_t active_size() const { return lru_list_.size() + pinned_list_.size(); }

  // Returns the number of unpinned (evictable) entries, including eviction
  // candidates, that currently hold a host DRAM block (status is HOST or
  // HOST_AND_HBM with host_block_id >= 0).
  // The count is maintained only by Put/PutBack/UpdateValue and the pin/unpin/
  // erase paths. Mutating `status` or `host_block_id` through a pointer from
  // Get()/PeekMutable()/PeekMutableIncludingCandidates() bypasses it; use
  // UpdateValue() for those changes instead.
  size_t NumEvictableHostBlocks() const { return evictable_host_blocks_; }

  // True if `Value` exposes `status` and `host_block_id` members, i.e. if
  // NumEvictableHostBlocks() tracks it. Otherwise the count is always 0.
  static constexpr bool TracksHostResidency() {
    return HasHostResidency<Value>::value;
  }

  // Returns true if the cache is completely empty.
  bool empty() const { return map_.empty(); }

  // Returns true if the key is present in the cache (and not a candidate).
  bool Contains(const Key& key) const {
    auto it = map_.find(key);
    return it != map_.end() && it->second->location != NodeLocation::kCandidate;
  }

  // Clears all elements from the cache.
  void Clear() {
    map_.clear();
    lru_list_.clear();
    pinned_list_.clear();
    evict_candidate_list_.clear();
    evictable_host_blocks_ = 0;
  }

  // Inserts or updates a key-value pair.
  // If the key already exists, its value is updated and promoted to Most
  // Recently Used (MRU).
  // If the key is new and inserting it exceeds capacity, the unpinned Least
  // Recently Used (LRU) item is demoted to the eviction-candidate list (it
  // stays in the map, invisible to Peek, though Get/GetAndPin/Pin still find
  // it and resurrect it into the active list, until Erase reclaims it) and
  // returned as the "evicted" pair; std::nullopt otherwise.
  // CAVEAT: when nothing is evictable (every entry pinned), the new key is
  // NOT inserted and std::nullopt is still returned -- indistinguishable
  // from a plain success. A caller that must know checks available_space()
  // beforehand or verifies the key afterwards.
  std::optional<std::pair<Key, Value>> Put(Key key, Value value) {
    auto it = map_.find(key);
    if (it != map_.end()) {
      // Update existing item and promote to MRU
      if (it->second->pin_count == 0) {
        OnUnpinnedNodeRemoved(it->second->value);
        OnUnpinnedNodeAdded(value);
      }
      it->second->value = std::move(value);
      Promote(it->second);
      return std::nullopt;
    }

    std::optional<std::pair<Key, Value>> evicted = std::nullopt;
    if (active_size() >= capacity_) {
      evicted = Evict();
      if (!evicted.has_value()) return std::nullopt;
    }

    // Insert new item at the front of the lru_list_ (MRU)
    lru_list_.emplace_front(std::move(key), std::move(value), 0,
                            NodeLocation::kLru);
    OnUnpinnedNodeAdded(lru_list_.front().value);
    map_.emplace_hint(it, lru_list_.front().key, lru_list_.begin());
    return evicted;
  }

  // Inserts a key-value pair at the back of the lru_list_ (least recently
  // used position). If the key already exists, its value is updated and moved
  // to the back of lru_list_.
  void PutBack(Key key, Value value) {
    auto it = map_.find(key);
    if (it != map_.end()) {
      if (it->second->pin_count == 0) {
        OnUnpinnedNodeRemoved(it->second->value);
        OnUnpinnedNodeAdded(value);
      }
      it->second->value = std::move(value);
      PromoteBack(it->second);
      return;
    }

    if (active_size() >= capacity_) {
      Evict();
    }

    lru_list_.emplace_back(std::move(key), std::move(value), 0,
                           NodeLocation::kLru);
    OnUnpinnedNodeAdded(lru_list_.back().value);
    map_.emplace_hint(it, lru_list_.back().key, std::prev(lru_list_.end()));
  }

  // Updates an existing non-candidate key's value in-place without perturbing
  // LRU order. Returns true if the key was found and updated, false if absent
  // or an eviction candidate.
  bool UpdateValue(const Key& key, Value value) {
    auto it = map_.find(key);
    if (it == map_.end() || it->second->location == NodeLocation::kCandidate) {
      return false;
    }
    if (it->second->pin_count == 0) {
      OnUnpinnedNodeRemoved(it->second->value);
      OnUnpinnedNodeAdded(value);
    }
    it->second->value = std::move(value);
    return true;
  }

  // Retrieves a pointer to the stored value for the given key and promotes
  // the item to Most Recently Used (MRU). Returns nullptr if absent.
  // Writing `status`/`host_block_id` through the returned pointer bypasses
  // NumEvictableHostBlocks(); use UpdateValue() instead.
  Value* Get(const Key& key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return nullptr;
    }
    Promote(it->second);
    return &(it->second->value);
  }

  // Retrieves a const pointer to the stored value for the given key without
  // promoting it to MRU (preserves exact LRU order). Returns nullptr if absent
  // or a candidate.
  const Value* Peek(const Key& key) const {
    auto it = map_.find(key);
    if (it == map_.end() || it->second->location == NodeLocation::kCandidate) {
      return nullptr;
    }
    return &(it->second->value);
  }

  // Exposes a mutable pointer via Peek without perturbing LRU order.
  // Returns nullptr if absent or candidate.
  // Writing `status`/`host_block_id` through the returned pointer bypasses
  // NumEvictableHostBlocks(); use UpdateValue() instead.
  Value* PeekMutable(const Key& key) {
    auto it = map_.find(key);
    if (it == map_.end() || it->second->location == NodeLocation::kCandidate) {
      return nullptr;
    }
    return &(it->second->value);
  }

  // Retrieves a pointer to the stored value even if it is a candidate.
  const Value* PeekIncludingCandidates(const Key& key) const {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return nullptr;
    }
    return &(it->second->value);
  }

  // Exposes a mutable pointer even if it is a candidate.
  // Writing `status`/`host_block_id` through the returned pointer bypasses
  // NumEvictableHostBlocks(); use UpdateValue() instead.
  Value* PeekMutableIncludingCandidates(const Key& key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return nullptr;
    }
    return &(it->second->value);
  }

  // Retrieves a pointer to the stored value for the given key and increments
  // its reference pin count. Returns nullptr if absent.
  // Unlike Peek, this FINDS an eviction candidate: pinning one splices it
  // out of the candidate list, un-queueing it from eviction. A caller that
  // must not resurrect candidates Peeks first and pins only on a non-null
  // answer.
  Value* GetAndPin(const Key& key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return nullptr;
    }
    if (it->second->pin_count == 0) {
      OnUnpinnedNodeRemoved(it->second->value);
      if (it->second->location == NodeLocation::kLru) {
        pinned_list_.splice(pinned_list_.begin(), lru_list_, it->second);
      } else if (it->second->location == NodeLocation::kCandidate) {
        pinned_list_.splice(pinned_list_.begin(), evict_candidate_list_,
                            it->second);
      }
      it->second->location = NodeLocation::kPinned;
    }
    it->second->pin_count++;
    return &(it->second->value);
  }

  // Increments the reference pin count for the given key.
  // Pinned items are protected entirely against automated LRU eviction.
  // Returns true if the key exists and was successfully pinned.
  // Delegates to GetAndPin, including its candidate-resurrection behaviour.
  bool Pin(const Key& key) { return GetAndPin(key) != nullptr; }

  // Decrements the reference pin count for the given key, clamping at zero.
  // When the count reaches 0 the item returns to the active list at the MRU
  // position -- so a pin/unpin round trip reorders the LRU -- and becomes
  // eligible for normal eviction again.
  // Returns true whenever the key exists, whether or not a pin was actually
  // dropped: the return value cannot detect an over-release.
  bool Unpin(const Key& key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return false;
    }
    if (it->second->pin_count > 0) {
      it->second->pin_count--;
      if (it->second->pin_count == 0) {
        // Move from pinned_list_ to lru_list_ (promote to MRU)
        lru_list_.splice(lru_list_.begin(), pinned_list_, it->second);
        it->second->location = NodeLocation::kLru;
        OnUnpinnedNodeAdded(it->second->value);
      }
    }
    return true;
  }

  // Returns the pin count for the given key, or 0 if absent.
  int GetPinCount(const Key& key) const {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return 0;
    }
    return it->second->pin_count;
  }

  // Exposes read-only access to the internal map for iteration.
  const absl::flat_hash_map<Key, typename std::list<CacheNode>::iterator>& map()
      const {
    return map_;
  }

  // Returns up to `count` oldest evictable (unpinned) keys in LRU order (oldest
  // first).
  std::vector<Key> GetEvictableKeys(size_t count) const {
    std::vector<Key> keys;
    keys.reserve(
        std::min(count, evict_candidate_list_.size() + lru_list_.size()));
    // 1. Scan evict_candidate_list_ first (oldest first, i.e. front to back)
    for (auto it = evict_candidate_list_.begin();
         it != evict_candidate_list_.end() && keys.size() < count; ++it) {
      keys.push_back(it->key);
    }
    // 2. Scan lru_list_ (oldest first, i.e. back to front)
    if (keys.size() < count) {
      for (auto it = lru_list_.rbegin();
           it != lru_list_.rend() && keys.size() < count; ++it) {
        keys.push_back(it->key);
      }
    }
    return keys;
  }

  // Erases the given key from the cache entirely. Pinned entries cannot be
  // erased and will return false.
  // Returns true if the element was successfully erased, false otherwise.
  bool Erase(const Key& key) {
    auto it = map_.find(key);
    if (it == map_.end() || it->second->pin_count > 0) {
      return false;
    }
    OnUnpinnedNodeRemoved(it->second->value);
    if (it->second->location == NodeLocation::kLru) {
      lru_list_.erase(it->second);
    } else if (it->second->location == NodeLocation::kCandidate) {
      evict_candidate_list_.erase(it->second);
    }
    map_.erase(it);
    return true;
  }

  // Evicts the least recently used, unpinned key-value pair to candidate list.
  // Returns the evicted pair if successful, or std::nullopt.
  std::optional<std::pair<Key, Value>> Evict() {
    if (lru_list_.empty()) {
      return std::nullopt;
    }
    auto it = std::prev(lru_list_.end());
    it->location = NodeLocation::kCandidate;
    evict_candidate_list_.splice(evict_candidate_list_.end(), lru_list_, it);
    return std::make_pair(it->key, it->value);
  }

  // Test-only observability: the candidate list has no production reader.
  // Production reclaims candidates through GetEvictableKeys, which scans this
  // list before the active one.
  std::vector<Key> GetEvictCandidateKeys() const {
    std::vector<Key> keys;
    keys.reserve(evict_candidate_list_.size());
    for (const auto& node : evict_candidate_list_) {
      keys.push_back(node.key);
    }
    return keys;
  }

 private:
  template <typename T, typename = void>
  struct HasHostResidency : std::false_type {};

  template <typename T>
  struct HasHostResidency<
      T, std::void_t<decltype(std::declval<const T&>().status),
                     decltype(std::declval<const T&>().host_block_id)>>
      : std::true_type {};

  static bool IsHostResidentValue(const Value& value) {
    if constexpr (HasHostResidency<Value>::value) {
      using StatusType = std::decay_t<decltype(value.status)>;
      return (value.status == StatusType::HOST ||
              value.status == StatusType::HOST_AND_HBM) &&
             value.host_block_id >= 0;
    } else {
      return false;
    }
  }

  void OnUnpinnedNodeAdded(const Value& value) {
    if (IsHostResidentValue(value)) {
      ++evictable_host_blocks_;
    }
  }

  void OnUnpinnedNodeRemoved(const Value& value) {
    if (!IsHostResidentValue(value)) return;
    DCHECK_GT(evictable_host_blocks_, 0u)
        << "evictable host block count underflow";
    if (evictable_host_blocks_ > 0) {
      --evictable_host_blocks_;
    }
  }

  struct CacheNode {
    Key key;
    Value value;
    int pin_count = 0;
    NodeLocation location = NodeLocation::kLru;

    CacheNode(Key k, Value v, int p, NodeLocation loc = NodeLocation::kLru)
        : key(std::move(k)), value(std::move(v)), pin_count(p), location(loc) {}
  };

  void Promote(typename std::list<CacheNode>::iterator node_it) {
    if (node_it->pin_count > 0) {
      pinned_list_.splice(pinned_list_.begin(), pinned_list_, node_it);
      node_it->location = NodeLocation::kPinned;
    } else {
      if (node_it->location == NodeLocation::kLru) {
        lru_list_.splice(lru_list_.begin(), lru_list_, node_it);
      } else if (node_it->location == NodeLocation::kCandidate) {
        lru_list_.splice(lru_list_.begin(), evict_candidate_list_, node_it);
        node_it->location = NodeLocation::kLru;
      }
    }
  }

  void PromoteBack(typename std::list<CacheNode>::iterator node_it) {
    if (node_it->pin_count > 0) {
      pinned_list_.splice(pinned_list_.end(), pinned_list_, node_it);
      node_it->location = NodeLocation::kPinned;
    } else {
      if (node_it->location == NodeLocation::kLru) {
        lru_list_.splice(lru_list_.end(), lru_list_, node_it);
      } else if (node_it->location == NodeLocation::kCandidate) {
        lru_list_.splice(lru_list_.end(), evict_candidate_list_, node_it);
        node_it->location = NodeLocation::kLru;
      }
    }
  }

  size_t capacity_;
  size_t evictable_host_blocks_ = 0;
  std::list<CacheNode> lru_list_;
  std::list<CacheNode> pinned_list_;
  std::list<CacheNode> evict_candidate_list_;
  absl::flat_hash_map<Key, typename std::list<CacheNode>::iterator> map_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_LRU_CACHE_H_
