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

#ifndef THIRD_PARTY_TPU_RAIDEN_TRANSPORT_LIB_CHUNK_H_
#define THIRD_PARTY_TPU_RAIDEN_TRANSPORT_LIB_CHUNK_H_

#include <cstdint>

namespace tpu_raiden::transport::lib {

inline constexpr uint8_t kOpBufferPull = 3;
inline constexpr uint8_t kOpBufferPush = 5;
inline constexpr uint8_t kOpBufferPushBatched = 7;

// Compact 32-byte binary chunk header layout.
struct alignas(8) ChunkHeader {
  // LINT.IfChange
  uint16_t version;  // Header version
  uint8_t op;     // OP code. See kOp* constants.
  uint8_t flags;  // Holds major_order or protocol flags
  uint16_t buffer_id;      // Multidimensional Buffer / Layer ID coordinate
  uint16_t reserved;       // Holds parallelism/expected chunks count
  uint16_t metadata_size;  // Size of metadata item in bytes for batch push
  uint16_t padding;        // Unused padding to align fields
  uint32_t remote_id;      // Remote block ID or linear memory offset
  uint32_t local_id;       // Local block ID or target shard index
  uint32_t count_or_size;  // Number of blocks or continuous payload bytes
  uint64_t uuid;           // Globally unique transaction routing ID
  // LINT.ThenChange(chunk.fbs)

  bool operator==(const ChunkHeader&) const = default;
};
static_assert(sizeof(ChunkHeader) == 32);

struct ChunkMetadata {
  // LINT.IfChange
  uint32_t layer_idx;
  uint32_t dst_shard_idx;
  uint64_t dst_offset_bytes;
  uint64_t size_bytes;
  uint64_t dst_stride_bytes = 0;
  uint32_t count = 1;
  // LINT.ThenChange(chunk.fbs)

  bool operator==(const ChunkMetadata&) const = default;
};

}  // namespace tpu_raiden::transport::lib

#endif  // THIRD_PARTY_TPU_RAIDEN_TRANSPORT_LIB_CHUNK_H_
