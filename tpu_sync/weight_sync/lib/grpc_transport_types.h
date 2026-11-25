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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_LIB_GRPC_TRANSPORT_TYPES_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_LIB_GRPC_TRANSPORT_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tpu_sync/proto/grpc_transport_service.pb.h"

namespace tpu_raiden::weight_sync {

struct ScatterChunkSpec {
  absl::Span<const uint8_t> payload;
  uint64_t byte_offset = 0;
  uint64_t byte_size = 0;
  uint64_t byte_stride = 0;
};

struct GatherChunkSpec {
  std::vector<absl::Span<const uint8_t>> payloads;
  uint64_t byte_offset = 0;
};

using ChunkSpec = std::variant<ScatterChunkSpec, GatherChunkSpec>;

struct PushChunksRequest {
  uint64_t uuid = 0;
  uint64_t buffer_id = 0;
  uint32_t shard_id = 0;
  uint64_t chunk_batch_id = 0;
  std::vector<ChunkSpec> chunks;
};

struct PullChunksRequest {
  uint64_t uuid = 0;
  uint64_t buffer_id = 0;
  uint32_t shard_id = 0;
  uint64_t chunk_batch_id = 0;
};

struct PullChunksResponse {
  std::vector<ChunkSpec> chunks;
  // Optional owner kept alive until the gRPC response write completes.
  std::shared_ptr<const void> owner;
};

// Validates that |chunk| fits within a destination buffer of |host_size| bytes
// without overflow or overlapping strides.
absl::Status ValidateChunkProto(const ::tpu_sync::proto::ChunkProto& chunk,
                                size_t host_size, int chunk_idx);

// Copies |chunk|'s payload into |dst_buffer|. Caller must validate |chunk| with
// ValidateChunkProto first.
void ApplyChunkProto(const ::tpu_sync::proto::ChunkProto& chunk,
                     absl::Span<uint8_t> dst_buffer);

}  // namespace tpu_raiden::weight_sync

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_LIB_GRPC_TRANSPORT_TYPES_H_
