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

#include "tpu_sync/common/raiden_id.h"

#include <sstream>

#include <gtest/gtest.h>

namespace tpu_raiden {
namespace {

TEST(RaidenIdTest, DefaultConstructor) {
  RaidenId id;
  EXPECT_TRUE(id.empty());
  EXPECT_EQ(id.job_name, "");
  EXPECT_EQ(id.job_replica_id, "");
  EXPECT_EQ(id.data_name, "");
  EXPECT_EQ(id.data_replica_idx, 0);
}

TEST(RaidenIdTest, NonEmptyAndEquality) {
  RaidenId id1{"trainer", "0", "weights", 0};
  RaidenId id2{"trainer", "0", "weights", 0};
  RaidenId id3{"sampler", "0", "weights", 0};

  EXPECT_FALSE(id1.empty());
  EXPECT_EQ(id1, id2);
  EXPECT_NE(id1, id3);
}

TEST(RaidenIdTest, Hashability) {
  RaidenId id1{"trainer", "0", "weights", 0};
  RaidenId id2{"trainer", "0", "weights", 0};

  RaidenIdHash hasher;
  EXPECT_EQ(hasher(id1), hasher(id2));
}

TEST(RaidenIdTest, StreamOutput) {
  RaidenId id{"trainer", "0", "weights", 1};
  std::ostringstream os;
  os << id;
  EXPECT_EQ(os.str(), "RaidenId{trainer, 0, weights, 1}");
}

TEST(RaidenIdTest, KvCacheNamespaceAlias) {
  kv_cache::RaidenId id{"inference", "1", "kv_cache", 0};
  EXPECT_EQ(id.job_name, "inference");
}

}  // namespace
}  // namespace tpu_raiden
