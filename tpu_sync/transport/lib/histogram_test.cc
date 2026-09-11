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

#include "tpu_sync/transport/lib/histogram.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>

namespace tpu_raiden::transport::lib {
namespace {

TEST(HistogramTest, EmptyHistogram) {
  Histogram<int> h;
  EXPECT_EQ(h.Get(42), 0);
}

TEST(HistogramTest, SingleKeyRLE) {
  Histogram<size_t> h;
  for (int i = 0; i < 100; ++i) {
    h.Add(7, 1);
  }
  EXPECT_EQ(h.Get(7), 100);
  EXPECT_EQ(h.Get(0), 0);
}

TEST(HistogramTest, AddWithCustomCount) {
  Histogram<std::string, uint64_t> h;
  h.Add("alpha", 10);
  h.Add("alpha", 5);
  h.Add("beta", 3);

  EXPECT_EQ(h.Get("alpha"), 15);
  EXPECT_EQ(h.Get("beta"), 3);
  EXPECT_EQ(h.Get("gamma"), 0);
}

TEST(HistogramTest, MultipleKeysInterleaved) {
  Histogram<int> h;
  h.Add(1, 1);
  h.Add(1, 1);
  h.Add(2, 1);
  h.Add(2, 1);
  h.Add(1, 1);
  h.Add(3, 1);
  h.Add(2, 1);
  h.Add(3, 1);

  EXPECT_EQ(h.Get(1), 3);
  EXPECT_EQ(h.Get(2), 3);
  EXPECT_EQ(h.Get(3), 2);
  EXPECT_EQ(h.Get(4), 0);
}

TEST(HistogramTest, ExceedsInlinedElements) {
  Histogram<int, int, 2> h;
  for (int i = 0; i < 16; ++i) {
    h.Add(i, i + 1);
  }
  EXPECT_EQ(h.size(), 16);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(h.Get(i), i + 1);
  }
}

TEST(HistogramTest, RangeBasedIteration) {
  Histogram<int, int> h;
  h.Add(10, 1);
  h.Add(20, 2);
  h.Add(30, 3);
  EXPECT_EQ(h.size(), 3);

  int sum_keys = 0;
  int sum_counts = 0;
  for (const auto& [k, c] : h) {
    sum_keys += k;
    sum_counts += c;
  }
  EXPECT_EQ(sum_keys, 60);
  EXPECT_EQ(sum_counts, 6);
}

}  // namespace
}  // namespace tpu_raiden::transport::lib
