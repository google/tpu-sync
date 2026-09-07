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

#include "tpu_sync/kv_cache/kv_cache_metadata.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/statusor.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::IsEmpty;

// 64-byte aligned backing buffer standing in for the shared memory region.
class Region {
 public:
  explicit Region(int num_blocks)
      : buffer_(KVCacheMetadata::RequiredSizeBytes(num_blocks) + 63) {}

  absl::Span<uint8_t> span() {
    auto addr = reinterpret_cast<uintptr_t>(buffer_.data());
    size_t offset = (64 - addr % 64) % 64;
    return absl::MakeSpan(buffer_.data() + offset, buffer_.size() - offset);
  }

 private:
  std::vector<uint8_t> buffer_;
};

TEST(KVCacheMetadataTest, FormatCreatesEmptyTable) {
  Region region(4);
  TF_ASSERT_OK_AND_ASSIGN(auto metadata,
                          KVCacheMetadata::Format(region.span(), 4));
  EXPECT_EQ(metadata.num_blocks(), 4);
  EXPECT_THAT(metadata.ValidEntries(), IsEmpty());
}

TEST(KVCacheMetadataTest, FormatRejectsInvalidRegions) {
  Region region(4);
  EXPECT_THAT(KVCacheMetadata::Format(region.span(), 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(KVCacheMetadata::Format(absl::Span<uint8_t>(), 4),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // Too small for 8 blocks.
  EXPECT_THAT(KVCacheMetadata::Format(region.span().subspan(0, 128), 8),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // Misaligned.
  EXPECT_THAT(KVCacheMetadata::Format(region.span().subspan(1), 2),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(KVCacheMetadataTest, SetAndClearRoundTrip) {
  Region region(4);
  TF_ASSERT_OK_AND_ASSIGN(auto metadata,
                          KVCacheMetadata::Format(region.span(), 4));

  ABSL_ASSERT_OK(metadata.Set(1, "hash_b", /*seq=*/2));
  ABSL_ASSERT_OK(metadata.Set(3, "hash_d", /*seq=*/1));
  EXPECT_THAT(metadata.ValidEntries(), ElementsAre(FieldsAre(1, "hash_b", 2),
                                                   FieldsAre(3, "hash_d", 1)));

  ABSL_ASSERT_OK(metadata.Clear(1));
  EXPECT_THAT(metadata.ValidEntries(), ElementsAre(FieldsAre(3, "hash_d", 1)));
}

TEST(KVCacheMetadataTest, SetOverwritesPreviousBinding) {
  Region region(2);
  TF_ASSERT_OK_AND_ASSIGN(auto metadata,
                          KVCacheMetadata::Format(region.span(), 2));

  ABSL_ASSERT_OK(metadata.Set(0, "old_hash", /*seq=*/1));
  ABSL_ASSERT_OK(metadata.Set(0, "new", /*seq=*/2));
  EXPECT_THAT(metadata.ValidEntries(), ElementsAre(FieldsAre(0, "new", 2)));
}

TEST(KVCacheMetadataTest, SetValidatesArguments) {
  Region region(2);
  TF_ASSERT_OK_AND_ASSIGN(auto metadata,
                          KVCacheMetadata::Format(region.span(), 2));

  EXPECT_THAT(metadata.Set(-1, "h", 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(metadata.Set(2, "h", 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(metadata.Set(0, "", 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  std::string too_long(KVCacheMetadata::kMaxHashLength + 1, 'x');
  EXPECT_THAT(metadata.Set(0, too_long, 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(metadata.Clear(2), StatusIs(absl::StatusCode::kInvalidArgument));

  std::string max_length(KVCacheMetadata::kMaxHashLength, 'y');
  ABSL_EXPECT_OK(metadata.Set(0, max_length, 0));
  EXPECT_THAT(metadata.ValidEntries(),
              ElementsAre(FieldsAre(0, max_length, 0)));
}

TEST(KVCacheMetadataTest, HashesAreOpaqueBytes) {
  Region region(1);
  TF_ASSERT_OK_AND_ASSIGN(auto metadata,
                          KVCacheMetadata::Format(region.span(), 1));

  std::string binary_hash("\x00\xff\x00raiden\x01", 10);
  ABSL_ASSERT_OK(metadata.Set(0, binary_hash, /*seq=*/7));
  EXPECT_THAT(metadata.ValidEntries(),
              ElementsAre(FieldsAre(0, binary_hash, 7)));
}

TEST(KVCacheMetadataTest, AttachRecoversEntriesFromSurvivingRegion) {
  Region region(4);
  {
    TF_ASSERT_OK_AND_ASSIGN(auto metadata,
                            KVCacheMetadata::Format(region.span(), 4));
    ABSL_ASSERT_OK(metadata.Set(0, "hash_a", /*seq=*/3));
    ABSL_ASSERT_OK(metadata.Set(2, "hash_c", /*seq=*/4));
    // The view is dropped here; the region survives, as shared memory would
    // across an engine crash.
  }

  TF_ASSERT_OK_AND_ASSIGN(auto recovered,
                          KVCacheMetadata::Attach(region.span(), 4));
  EXPECT_THAT(recovered.ValidEntries(), ElementsAre(FieldsAre(0, "hash_a", 3),
                                                    FieldsAre(2, "hash_c", 4)));
}

TEST(KVCacheMetadataTest, AttachRejectsUnformattedOrMismatchedRegions) {
  // Sized for 8 blocks so the mismatch below fails on the header contents,
  // not on the region size.
  Region region(8);

  // Never formatted.
  std::memset(region.span().data(), 0, KVCacheMetadata::RequiredSizeBytes(4));
  EXPECT_THAT(KVCacheMetadata::Attach(region.span(), 4),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  ABSL_ASSERT_OK(KVCacheMetadata::Format(region.span(), 4));
  // Formatted for 4 blocks, attached expecting 8 (block pool resized across
  // the restart): the table no longer matches the pool, treat as cold start.
  EXPECT_THAT(KVCacheMetadata::Attach(region.span(), 8),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  ABSL_EXPECT_OK(KVCacheMetadata::Attach(region.span(), 4));
}

TEST(KVCacheMetadataTest, AttachValidatesModelUid) {
  Region region(4);
  ABSL_ASSERT_OK(KVCacheMetadata::Format(region.span(), 4, "model_a"));

  // A table recorded under another model (or none) must not attach: its
  // bindings describe another model's blocks.
  ABSL_EXPECT_OK(KVCacheMetadata::Attach(region.span(), 4, "model_a"));
  EXPECT_THAT(KVCacheMetadata::Attach(region.span(), 4, "model_b"),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(KVCacheMetadata::Attach(region.span(), 4),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Longer than the header field can record.
  EXPECT_THAT(KVCacheMetadata::Format(region.span(), 4, std::string(64, 'x')),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(KVCacheMetadataTest, FormatWipesSurvivingEntries) {
  Region region(2);
  TF_ASSERT_OK_AND_ASSIGN(auto metadata,
                          KVCacheMetadata::Format(region.span(), 2));
  ABSL_ASSERT_OK(metadata.Set(0, "stale", /*seq=*/1));

  TF_ASSERT_OK_AND_ASSIGN(auto reformatted,
                          KVCacheMetadata::Format(region.span(), 2));
  EXPECT_THAT(reformatted.ValidEntries(), IsEmpty());
}

TEST(KVCacheMetadataTest, UncommittedEntryIsInvisible) {
  Region region(2);
  ABSL_ASSERT_OK(KVCacheMetadata::Format(region.span(), 2));

  // Simulate a crash after the hash bytes landed but before the entry was
  // committed: write the fields directly and leave `valid` unset.
  auto* entry = reinterpret_cast<KVCacheMetadataEntry*>(
      region.span().data() + sizeof(KVCacheMetadataHeader));
  entry->seq = 9;
  entry->hash_len = 4;
  std::memcpy(entry->hash, "torn", 4);

  TF_ASSERT_OK_AND_ASSIGN(auto recovered,
                          KVCacheMetadata::Attach(region.span(), 2));
  EXPECT_THAT(recovered.ValidEntries(), IsEmpty());
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
