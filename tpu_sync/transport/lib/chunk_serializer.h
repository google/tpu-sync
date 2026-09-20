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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_CHUNK_SERIALIZER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_CHUNK_SERIALIZER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/lib/chunk_generated.h"

namespace tpu_raiden::transport::lib {

inline constexpr size_t kChunkHeaderSize = 64;
inline constexpr size_t kMaxMetadataSize = 40;
inline constexpr size_t kChunkSizeFieldSize = sizeof(uint32_t);

inline constexpr uint16_t kRaidenMagic =
    static_cast<uint16_t>(flatbuf::Constant_MAGIC);
static_assert(kRaidenMagic == 0x4452);

static_assert(sizeof(flatbuf::ChunkHeader) == kChunkHeaderSize);
static_assert(sizeof(flatbuf::ChunkMetadataV1) == 24);
static_assert(sizeof(flatbuf::ChunkMetadataV2) == kMaxMetadataSize);

// Returns the size of a chunk metadata for the given version.
constexpr size_t GetChunkMetadataSize(uint16_t ver) {
  switch (ver) {
    case 1:
      return 24;
    case 2:
      return 40;
    default:
      return 0;
  }
}

// Serializes the chunk header to an inlined byte vector.
absl::InlinedVector<char, kChunkHeaderSize> SerializeChunkHeader(
    const ChunkHeader& header);

// Parses the chunk header from its serialized binary bytes.
absl::StatusOr<ChunkHeader> DeserializeChunkHeader(
    absl::Span<const char> bytes);

// Serializes the chunk metadata to an inlined byte vector.
absl::InlinedVector<char, kMaxMetadataSize> SerializeChunkMetadata(
    const ChunkMetadata& meta, uint16_t ver = 1);

// Parses the chunk metadata from its serialized binary bytes.
absl::StatusOr<ChunkMetadata> DeserializeChunkMetadata(
    absl::Span<const char> bytes, uint16_t ver);

// Serializes a span of integer block IDs to a byte vector.
std::vector<uint8_t> SerializeBlockIds(absl::Span<const int> ids);

// Parses a vector of integer block IDs from its serialized binary bytes.
std::vector<int> DeserializeBlockIds(absl::Span<const uint8_t> bytes);

// Serializes a 32-bit chunk size to a 4-byte array.
std::array<uint8_t, kChunkSizeFieldSize> SerializeChunkSize(
    uint32_t size_bytes);

// Parses a 32-bit chunk size from its serialized binary bytes.
uint32_t DeserializeChunkSize(absl::Span<const uint8_t> bytes);

}  // namespace tpu_raiden::transport::lib

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_CHUNK_SERIALIZER_H_
