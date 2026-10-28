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

#include "tpu_sync/transport/lib/chunk.h"

#include <cstddef>
#include <cstdint>

namespace tpu_raiden::transport::lib {

// Why position-dependent mixing before XOR:
//   Chunks in a transfer can arrive out of order across threads/sockets, so we
//   combine per-byte hashes with XOR (commutative and associative). However,
//   raw XOR on unmixed byte values would self-cancel when two positions hold
//   the same value (x ^ x == 0) and would not detect swapped offsets/layers.
//   By folding (layer_idx, dst_shard_idx, dst_byte_offset, byte_val) through a
//   non-linear 64-bit avalanche mixer first, identical byte values at different
//   positions produce distinct 64-bit hashes that do not cancel out.
//
// Constants from standard Fibonacci hashing and Sebastiano Vigna's splitmix64
// (see //util/random/stable/stable_urbg_base.h):
//   - kGoldenRatio (0x9E3779B97F4A7C15): floor(2^64 / phi), spreads sequential
//     byte offsets uniformly across the 64-bit space.
//   - kSplitMixMul1 (0xBF58476D1CE4E5B9) & kSplitMixMul2 (0x94D049BB133111EB):
//     David Stafford's Mix13 multipliers for 64-bit bit-avalanche finalization.
uint64_t ComputeByteXorHash(uint32_t layer_idx, uint32_t dst_shard_idx,
                            uint64_t dst_byte_offset, uint8_t byte_val) {
  constexpr uint64_t kGoldenRatio = 0x9E3779B97F4A7C15ULL;
  constexpr uint64_t kSplitMixMul1 = 0xBF58476D1CE4E5B9ULL;
  constexpr uint64_t kSplitMixMul2 = 0x94D049BB133111EBULL;

  uint64_t x = (static_cast<uint64_t>(layer_idx) << 48) ^
               ((static_cast<uint64_t>(dst_shard_idx) & 0xFFFFULL) << 32) ^
               (dst_byte_offset * kGoldenRatio) ^
               static_cast<uint64_t>(byte_val);
  x = (x ^ (x >> 30)) * kSplitMixMul1;
  x = (x ^ (x >> 27)) * kSplitMixMul2;
  return x ^ (x >> 31);
}

uint64_t ComputeContiguousXorHash(uint32_t layer_idx, uint32_t dst_shard_idx,
                                  uint64_t dst_offset_bytes,
                                  const uint8_t* data, size_t size_bytes) {
  if (data == nullptr || size_bytes == 0) return 0;
  uint64_t acc = 0;
  for (size_t i = 0; i < size_bytes; ++i) {
    acc ^= ComputeByteXorHash(layer_idx, dst_shard_idx, dst_offset_bytes + i,
                              data[i]);
  }
  return acc;
}

uint64_t ComputeChunkXorHash(const ChunkMetadata& meta, const uint8_t* src_ptr,
                             size_t src_stride_bytes) {
  if (src_ptr == nullptr || meta.size_bytes == 0) return 0;
  const uint32_t count = meta.count > 0 ? meta.count : 1;
  const size_t dst_stride =
      meta.dst_stride_bytes > 0 ? meta.dst_stride_bytes : meta.size_bytes;
  const size_t src_stride =
      src_stride_bytes > 0 ? src_stride_bytes : meta.size_bytes;
  if (count == 1 ||
      (dst_stride == meta.size_bytes && src_stride == meta.size_bytes)) {
    return ComputeContiguousXorHash(meta.layer_idx, meta.dst_shard_idx,
                                    meta.dst_offset_bytes, src_ptr,
                                    count * meta.size_bytes);
  }
  uint64_t acc = 0;
  for (uint32_t chunk_idx = 0; chunk_idx < count; ++chunk_idx) {
    acc ^= ComputeContiguousXorHash(
        meta.layer_idx, meta.dst_shard_idx,
        meta.dst_offset_bytes + chunk_idx * dst_stride,
        src_ptr + chunk_idx * src_stride, meta.size_bytes);
  }
  return acc;
}

}  // namespace tpu_raiden::transport::lib
