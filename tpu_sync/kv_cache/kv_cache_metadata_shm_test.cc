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

#include "tpu_sync/kv_cache/kv_cache_metadata_shm.h"

#include <sys/mman.h>
#include <unistd.h>

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/kv_cache/kv_cache_metadata.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::IsEmpty;

// Names a per-test shm segment and unlinks it on scope exit (and up front,
// in case a previously crashed run left one behind).
class ScopedShmKey {
 public:
  explicit ScopedShmKey(absl::string_view test_name)
      : key_(absl::StrCat("/raiden_md_shm_test_", getpid(), "_", test_name)) {
    shm_unlink(key_.c_str());
  }
  ~ScopedShmKey() { shm_unlink(key_.c_str()); }

  const std::string& key() const { return key_; }

 private:
  std::string key_;
};

TEST(KVCacheMetadataShmTest, ColdStartFormatsEmptyTable) {
  ScopedShmKey key("cold");
  TF_ASSERT_OK_AND_ASSIGN(auto region, KVCacheMetadataShmRegion::AttachOrFormat(
                                           key.key(), 8, "model_a"));
  EXPECT_FALSE(region->warm());
  EXPECT_THAT(region->metadata().ValidEntries(), IsEmpty());
}

TEST(KVCacheMetadataShmTest, TableSurvivesProcessRestart) {
  ScopedShmKey key("restart");
  TF_ASSERT_OK_AND_ASSIGN(auto region, KVCacheMetadataShmRegion::AttachOrFormat(
                                           key.key(), 8, "model_a"));
  KVCacheMetadata metadata = region->metadata();
  ABSL_ASSERT_OK(metadata.Set(1, "hash_b", /*seq=*/2));
  ABSL_ASSERT_OK(metadata.Set(3, "hash_d", /*seq=*/1));

  // Dropping the region simulates the engine dying: the mapping goes away,
  // the segment survives.
  region.reset();

  TF_ASSERT_OK_AND_ASSIGN(
      auto revived,
      KVCacheMetadataShmRegion::AttachOrFormat(key.key(), 8, "model_a"));
  EXPECT_TRUE(revived->warm());
  EXPECT_THAT(
      revived->metadata().ValidEntries(),
      ElementsAre(FieldsAre(1, "hash_b", 2), FieldsAre(3, "hash_d", 1)));
}

TEST(KVCacheMetadataShmTest, NumBlocksMismatchRecreates) {
  ScopedShmKey key("num_blocks");
  TF_ASSERT_OK_AND_ASSIGN(auto region, KVCacheMetadataShmRegion::AttachOrFormat(
                                           key.key(), 8, "model_a"));
  KVCacheMetadata metadata = region->metadata();
  ABSL_ASSERT_OK(metadata.Set(1, "hash_b", /*seq=*/2));
  region.reset();

  // A different table geometry must not resurrect the old entries, whether
  // the surviving segment is too small (grown table) or its header disagrees
  // (shrunk table).
  TF_ASSERT_OK_AND_ASSIGN(auto grown, KVCacheMetadataShmRegion::AttachOrFormat(
                                          key.key(), 64, "model_a"));
  EXPECT_FALSE(grown->warm());
  EXPECT_THAT(grown->metadata().ValidEntries(), IsEmpty());
  region = std::move(grown);
  KVCacheMetadata grown_metadata = region->metadata();
  ABSL_ASSERT_OK(grown_metadata.Set(1, "hash_b", /*seq=*/2));
  region.reset();

  TF_ASSERT_OK_AND_ASSIGN(auto shrunk, KVCacheMetadataShmRegion::AttachOrFormat(
                                           key.key(), 8, "model_a"));
  EXPECT_FALSE(shrunk->warm());
  EXPECT_THAT(shrunk->metadata().ValidEntries(), IsEmpty());
}

TEST(KVCacheMetadataShmTest, ModelUidMismatchRecreates) {
  ScopedShmKey key("model_uid");
  TF_ASSERT_OK_AND_ASSIGN(auto region, KVCacheMetadataShmRegion::AttachOrFormat(
                                           key.key(), 8, "model_a"));
  KVCacheMetadata metadata = region->metadata();
  ABSL_ASSERT_OK(metadata.Set(1, "hash_b", /*seq=*/2));
  region.reset();

  // A table recorded under another model must not resurrect its bindings.
  TF_ASSERT_OK_AND_ASSIGN(
      auto other_model,
      KVCacheMetadataShmRegion::AttachOrFormat(key.key(), 8, "model_b"));
  EXPECT_FALSE(other_model->warm());
  EXPECT_THAT(other_model->metadata().ValidEntries(), IsEmpty());
}

TEST(KVCacheMetadataShmTest, ReformatErasesEntries) {
  ScopedShmKey key("reformat");
  TF_ASSERT_OK_AND_ASSIGN(auto region, KVCacheMetadataShmRegion::AttachOrFormat(
                                           key.key(), 8, "model_a"));
  KVCacheMetadata metadata = region->metadata();
  ABSL_ASSERT_OK(metadata.Set(1, "hash_b", /*seq=*/2));

  ABSL_ASSERT_OK(region->Reformat());
  EXPECT_FALSE(region->warm());
  EXPECT_THAT(region->metadata().ValidEntries(), IsEmpty());
  // Copies attached before the reformat observe the emptied table.
  EXPECT_THAT(metadata.ValidEntries(), IsEmpty());
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
