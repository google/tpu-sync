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

#include "tpu_sync/transport/lib/chunk_serializer.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/lib/chunk.h"

namespace tpu_raiden::transport::lib {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAreArray;

ChunkHeader MakeSampleHeaderV1() {
  return ChunkHeader{
      .version = 1,
      .op = 0xAB,
      .flags = 0xCD,
      .buffer_id = 0x1234,
      .reserved = 0x5678,
      .metadata_size = 24,
      .remote_id = 0x12345678,
      .local_id = 0x9ABCDEF0,
      .count_or_size = 0x11223344,
      .uuid = 0x0123456789ABCDEFULL,
  };
}

ChunkHeader MakeSampleHeaderV2() {
  return ChunkHeader{
      .version = 2,
      .op = 0xAB,
      .flags = 0xCD,
      .buffer_id = 0x1234,
      .reserved = 0x5678,
      .metadata_size = 40,
      .remote_id = 0x12345678,
      .local_id = 0x9ABCDEF0,
      .count_or_size = 0x11223344,
      .uuid = 0x0123456789ABCDEFULL,
  };
}

ChunkMetadata MakeSampleMetadataV2() {
  return ChunkMetadata{
      .layer_idx = 0x12345678,
      .dst_shard_idx = 0x9ABCDEF0,
      .dst_offset_bytes = 0x1122334455667788ULL,
      .size_bytes = 0x99AABBCCDDEEFF00ULL,
      .dst_stride_bytes = 0x0102030405060708ULL,
      .count = 0x12345678,
  };
}

TEST(ChunkHeaderSerializerTest, SerializeAndDeserializeV1) {
  const ChunkHeader original = MakeSampleHeaderV1();
  const auto bytes = SerializeChunkHeader(original);

  EXPECT_THAT(DeserializeChunkHeader(bytes), IsOkAndHolds(original));
}

TEST(ChunkHeaderSerializerTest, SerializeAndDeserialize) {
  const ChunkHeader original = MakeSampleHeaderV2();
  const auto bytes = SerializeChunkHeader(original);

  EXPECT_THAT(DeserializeChunkHeader(bytes), IsOkAndHolds(original));
}

TEST(ChunkHeaderSerializerTest, SerializeToLittleEndian) {
  const auto wire = SerializeChunkHeader(MakeSampleHeaderV2());
  ASSERT_EQ(wire.size(), kChunkHeaderSize);

  alignas(8) const uint8_t expected_wire[64] = {
      0x52, 0x44, 0x02, 0x00, 0xAB, 0xCD, 0x34, 0x12, 0x78, 0x56, 0x28,
      0x00, 0x78, 0x56, 0x34, 0x12, 0xF0, 0xDE, 0xBC, 0x9A, 0x44, 0x33,
      0x22, 0x11, 0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01,
  };

  EXPECT_THAT(wire, ElementsAreArray(expected_wire));
}

TEST(ChunkHeaderSerializerTest, VerifyMagicBytes) {
  const auto s = SerializeChunkHeader(MakeSampleHeaderV2());
  ASSERT_GE(s.size(), 2);
  ASSERT_EQ(s[0], 'R');
  ASSERT_EQ(s[1], 'D');
}

TEST(ChunkHeaderSerializerTest, DeserializeLittleEndian) {
  alignas(8) const uint8_t raw_wire[64] = {
      0x52, 0x44, 0x02, 0x00, 0xAB, 0xCD, 0x34, 0x12, 0x78, 0x56, 0x28,
      0x00, 0x78, 0x56, 0x34, 0x12, 0xF0, 0xDE, 0xBC, 0x9A, 0x44, 0x33,
      0x22, 0x11, 0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01,
  };

  const auto wire = absl::MakeConstSpan(reinterpret_cast<const char*>(raw_wire),
                                        sizeof(raw_wire));

  EXPECT_THAT(DeserializeChunkHeader(wire), IsOkAndHolds(MakeSampleHeaderV2()));
}

TEST(ChunkHeaderSerializerTest, DeserializeRejectsInvalidMagic) {
  auto bytes = SerializeChunkHeader(MakeSampleHeaderV2());
  bytes[0] ^= 0xFF;  // Corrupt the magic field.

  EXPECT_THAT(DeserializeChunkHeader(bytes),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ChunkHeaderSerializerTest, DeserializeRejectsInvalidVersion) {
  auto bytes = SerializeChunkHeader(MakeSampleHeaderV2());
  // The `ver` field is a little-endian uint16.
  // Version 0 is unsupported.
  bytes[2] = 0x00;
  bytes[3] = 0x00;
  EXPECT_THAT(DeserializeChunkHeader(bytes),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Version 3 is unsupported.
  bytes[2] = 0x03;
  bytes[3] = 0x00;
  EXPECT_THAT(DeserializeChunkHeader(bytes),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(ChunkMetadataSerializerTest, SerializeAndDeserializeV1) {
  const ChunkMetadata original = {
      .layer_idx = 0x12345678,
      .dst_shard_idx = 0x9ABCDEF0,
      .dst_offset_bytes = 0x1122334455667788ULL,
      .size_bytes = 0x99AABBCCDDEEFF00ULL,
      .dst_stride_bytes = 0x99AABBCCDDEEFF00ULL,
      .count = 1,
  };
  const auto bytes = SerializeChunkMetadata(original, /*ver=*/1);
  ASSERT_EQ(bytes.size(), GetChunkMetadataSize(1));

  EXPECT_THAT(DeserializeChunkMetadata(bytes, /*ver=*/1),
              IsOkAndHolds(original));
}

TEST(ChunkMetadataSerializerTest, SerializeAndDeserialize) {
  const ChunkMetadata original = MakeSampleMetadataV2();
  const auto bytes = SerializeChunkMetadata(original, /*ver=*/2);

  EXPECT_THAT(DeserializeChunkMetadata(bytes, /*ver=*/2),
              IsOkAndHolds(original));
}

TEST(ChunkMetadataSerializerTest, SerializeToLittleEndian) {
  const auto wire = SerializeChunkMetadata(MakeSampleMetadataV2(), /*ver=*/2);
  ASSERT_EQ(wire.size(), GetChunkMetadataSize(2));

  alignas(8) const uint8_t expected_wire[40] = {
      0x78, 0x56, 0x34, 0x12, 0xF0, 0xDE, 0xBC, 0x9A, 0x88, 0x77,
      0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00, 0xFF, 0xEE, 0xDD,
      0xCC, 0xBB, 0xAA, 0x99, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03,
      0x02, 0x01, 0x78, 0x56, 0x34, 0x12, 0x00, 0x00, 0x00, 0x00,
  };

  EXPECT_THAT(wire, ElementsAreArray(expected_wire));
}

TEST(ChunkMetadataSerializerTest, DeserializeLittleEndian) {
  alignas(8) const uint8_t raw_wire[40] = {
      0x78, 0x56, 0x34, 0x12, 0xF0, 0xDE, 0xBC, 0x9A, 0x88, 0x77,
      0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00, 0xFF, 0xEE, 0xDD,
      0xCC, 0xBB, 0xAA, 0x99, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03,
      0x02, 0x01, 0x78, 0x56, 0x34, 0x12, 0x00, 0x00, 0x00, 0x00,
  };

  const auto wire = absl::MakeConstSpan(reinterpret_cast<const char*>(raw_wire),
                                        sizeof(raw_wire));

  EXPECT_THAT(DeserializeChunkMetadata(wire, /*ver=*/2),
              IsOkAndHolds(MakeSampleMetadataV2()));
}

TEST(ChunkMetadataSerializerTest, DeserializeV1LittleEndian) {
  alignas(8) const uint8_t raw_wire[24] = {
      0x78, 0x56, 0x34, 0x12, 0xF0, 0xDE, 0xBC, 0x9A, 0x88, 0x77, 0x66, 0x55,
      0x44, 0x33, 0x22, 0x11, 0x00, 0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99,
  };

  const auto wire = absl::MakeConstSpan(reinterpret_cast<const char*>(raw_wire),
                                        sizeof(raw_wire));

  const ChunkMetadata expected = {
      .layer_idx = 0x12345678,
      .dst_shard_idx = 0x9ABCDEF0,
      .dst_offset_bytes = 0x1122334455667788ULL,
      .size_bytes = 0x99AABBCCDDEEFF00ULL,
      .dst_stride_bytes = 0x99AABBCCDDEEFF00ULL,
      .count = 1,
  };

  EXPECT_THAT(DeserializeChunkMetadata(wire, /*ver=*/1),
              IsOkAndHolds(expected));
}

TEST(ChunkMetadataSerializerTest, DeserializeRejectsInvalidVersion) {
  const auto bytes = SerializeChunkMetadata(MakeSampleMetadataV2(), /*ver=*/2);
  // Version 1 has metadata size 24, so 40 bytes does not match.
  EXPECT_THAT(DeserializeChunkMetadata(bytes, /*ver=*/1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // Version 3 has metadata size 0, so 40 bytes does not match.
  EXPECT_THAT(DeserializeChunkMetadata(bytes, /*ver=*/3),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // Version 3 with 0-size input returns FailedPrecondition due to unsupported
  // version.
  EXPECT_THAT(DeserializeChunkMetadata(absl::Span<const char>(), /*ver=*/3),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(BlockIdsSerializerTest, SerializeAndDeserialize) {
  const std::vector<int> original = {0, 1, -1, 42, 0x12345678, -12345678};
  const auto bytes = SerializeBlockIds(original);

  EXPECT_EQ(DeserializeBlockIds(bytes), original);
}

TEST(BlockIdsSerializerTest, SerializeToLittleEndian) {
  const std::vector<int> original = {0x12345678, 0x01020304};
  const auto bytes = SerializeBlockIds(original);
  ASSERT_EQ(bytes.size(), 8);

  const uint8_t expected_wire[8] = {
      0x78, 0x56, 0x34, 0x12, 0x04, 0x03, 0x02, 0x01,
  };
  EXPECT_THAT(bytes, ElementsAreArray(expected_wire));
}

TEST(ChunkSizeSerializerTest, SerializeAndDeserialize) {
  for (uint32_t original : {0u, 1u, 1024u, 0x12345678u, 0xFFFFFFFFu}) {
    const auto bytes = SerializeChunkSize(original);
    EXPECT_EQ(DeserializeChunkSize(bytes), original);
  }
}

TEST(ChunkSizeSerializerTest, SerializeToLittleEndian) {
  const auto bytes = SerializeChunkSize(0x12345678);
  ASSERT_EQ(bytes.size(), 4);

  const uint8_t expected_wire[4] = {0x78, 0x56, 0x34, 0x12};
  EXPECT_THAT(bytes, ElementsAreArray(expected_wire));
}

}  // namespace
}  // namespace tpu_raiden::transport::lib
