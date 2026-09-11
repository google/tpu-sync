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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_HISTOGRAM_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_HISTOGRAM_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/container/inlined_vector.h"

namespace tpu_raiden::transport::lib {

// A light-weighted, histogram / run-length frequency map.
template <typename Key, typename Count = uint32_t, size_t InlinedElements = 4>
class Histogram {
 public:
  Histogram() = default;

  // Increments the count for `key` by `count`.
  // Checks the most recently accessed entry first as an O(1) fast-path for
  // consecutive identical keys, falling back to linear scan if not matching.
  void Add(const Key& key, Count count) {
    if (!entries_.empty() && entries_.back().first == key) {
      entries_.back().second += count;
      return;
    }
    auto it = FindEntry(key);
    if (it != entries_.end()) {
      it->second += count;
    } else {
      entries_.push_back({key, count});
    }
  }

  // Returns the count for `key`, or 0 if `key` is not present in the histogram.
  Count Get(const Key& key) const {
    auto it = FindEntry(key);
    return it != entries_.end() ? it->second : 0;
  }

  // Returns the number of distinct keys in the histogram.
  size_t size() const { return entries_.size(); }

  // Iterators over std::pair<Key, Count>.
  auto begin() const { return entries_.begin(); }
  auto end() const { return entries_.end(); }

 private:
  using Entry = std::pair<Key, Count>;
  // TODO(raiden-dev): Consider using a fixed-size array once we can confirm
  // the upper bound on distinct keys.
  using Container = absl::InlinedVector<Entry, InlinedElements>;

  auto FindEntry(const Key& key) {
    return std::find_if(
        entries_.begin(), entries_.end(),
        [&key](const Entry& entry) { return entry.first == key; });
  }

  auto FindEntry(const Key& key) const {
    return std::find_if(
        entries_.begin(), entries_.end(),
        [&key](const Entry& entry) { return entry.first == key; });
  }

  Container entries_;
};

}  // namespace tpu_raiden::transport::lib

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_HISTOGRAM_H_
