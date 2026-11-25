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

#include "tpu_sync/weight_sync/lib/grpc_transport_types.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/proto/grpc_transport_service.pb.h"

namespace tpu_raiden::weight_sync {
namespace {

absl::Status ValidateScatterChunk(
    const ::tpu_sync::proto::ScatterChunkProto& scatter, size_t host_size,
    int chunk_idx) {
  const uint64_t payload_size = scatter.payload().data().size();
  const uint64_t byte_offset = scatter.byte_offset();
  if (byte_offset > host_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("ScatterChunkProto[", chunk_idx, "] byte_offset ",
                     byte_offset, " exceeds host buffer size ", host_size));
  }
  if (payload_size == 0) {
    return absl::OkStatus();
  }

  const uint64_t slice_size =
      scatter.byte_size() > 0 ? scatter.byte_size() : payload_size;
  if (payload_size % slice_size != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ScatterChunkProto[", chunk_idx, "] payload size ", payload_size,
        " is not divisible by byte_size ", slice_size));
  }

  const uint64_t byte_stride =
      scatter.byte_stride() > 0 ? scatter.byte_stride() : slice_size;
  if (byte_stride < slice_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("ScatterChunkProto[", chunk_idx, "] byte_stride ",
                     byte_stride, " is smaller than byte_size ", slice_size));
  }

  const uint64_t num_slices = payload_size / slice_size;
  const uint64_t remaining = host_size - byte_offset;
  if (slice_size > remaining) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ScatterChunkProto[", chunk_idx,
        "] slice exceeds host buffer bounds: ", "byte_offset=", byte_offset,
        ", byte_size=", slice_size, ", host_size=", host_size));
  }
  if (num_slices > 1 &&
      (num_slices - 1) > (remaining - slice_size) / byte_stride) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ScatterChunkProto[", chunk_idx,
        "] strided span exceeds host buffer bounds: byte_offset=", byte_offset,
        ", byte_size=", slice_size, ", byte_stride=", byte_stride,
        ", num_slices=", num_slices, ", host_size=", host_size));
  }
  return absl::OkStatus();
}

absl::Status ValidateGatherChunk(
    const ::tpu_sync::proto::GatherChunkProto& gather, size_t host_size,
    int chunk_idx) {
  const uint64_t byte_offset = gather.byte_offset();
  if (byte_offset > host_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("GatherChunkProto[", chunk_idx, "] byte_offset ",
                     byte_offset, " exceeds host buffer size ", host_size));
  }
  uint64_t remaining = host_size - byte_offset;
  for (int i = 0; i < gather.payloads_size(); ++i) {
    const uint64_t payload_size = gather.payloads(i).data().size();
    if (payload_size > remaining) {
      return absl::InvalidArgumentError(absl::StrCat(
          "GatherChunkProto[", chunk_idx, "] payloads[", i,
          "] exceeds host buffer bounds: byte_offset=", byte_offset,
          ", host_size=", host_size));
    }
    remaining -= payload_size;
  }
  return absl::OkStatus();
}

void CopyCordToPtr(const absl::Cord& cord, uint8_t* dst) {
  size_t offset = 0;
  for (absl::string_view fragment : cord.Chunks()) {
    if (!fragment.empty()) {
      std::memcpy(dst + offset, fragment.data(), fragment.size());
      offset += fragment.size();
    }
  }
}

void ApplyScatterChunk(const ::tpu_sync::proto::ScatterChunkProto& scatter,
                       absl::Span<uint8_t> dst_buffer) {
  const absl::Cord& payload = scatter.payload().data();
  const size_t payload_size = payload.size();
  if (payload_size == 0) {
    return;
  }

  const size_t byte_offset = static_cast<size_t>(scatter.byte_offset());
  const size_t slice_size = scatter.byte_size() > 0
                                ? static_cast<size_t>(scatter.byte_size())
                                : payload_size;
  const size_t byte_stride = scatter.byte_stride() > 0
                                 ? static_cast<size_t>(scatter.byte_stride())
                                 : slice_size;

  if (byte_stride == slice_size) {
    CopyCordToPtr(payload, dst_buffer.data() + byte_offset);
    return;
  }

  const size_t num_slices = payload_size / slice_size;
  absl::Cord::CharIterator it = payload.char_begin();
  for (size_t s = 0; s < num_slices; ++s) {
    uint8_t* dst = dst_buffer.data() + byte_offset + s * byte_stride;
    size_t remaining = slice_size;
    while (remaining > 0) {
      absl::string_view chunk = absl::Cord::ChunkRemaining(it);
      const size_t to_copy = std::min(remaining, chunk.size());
      std::memcpy(dst, chunk.data(), to_copy);
      dst += to_copy;
      remaining -= to_copy;
      absl::Cord::Advance(&it, to_copy);
    }
  }
}

void ApplyGatherChunk(const ::tpu_sync::proto::GatherChunkProto& gather,
                      absl::Span<uint8_t> dst_buffer) {
  size_t write_offset = static_cast<size_t>(gather.byte_offset());
  for (const ::tpu_sync::proto::ChunkPayload& payload : gather.payloads()) {
    const absl::Cord& data = payload.data();
    CopyCordToPtr(data, dst_buffer.data() + write_offset);
    write_offset += data.size();
  }
}

}  // namespace

absl::Status ValidateChunkProto(const ::tpu_sync::proto::ChunkProto& chunk,
                                size_t host_size, int chunk_idx) {
  switch (chunk.chunk_case()) {
    case ::tpu_sync::proto::ChunkProto::kScatterChunk:
      return ValidateScatterChunk(chunk.scatter_chunk(), host_size, chunk_idx);
    case ::tpu_sync::proto::ChunkProto::kGatherChunk:
      return ValidateGatherChunk(chunk.gather_chunk(), host_size, chunk_idx);
    case ::tpu_sync::proto::ChunkProto::CHUNK_NOT_SET:
      return absl::InvalidArgumentError(
          absl::StrCat("ChunkProto[", chunk_idx, "] has no chunk payload set"));
  }
  return absl::InvalidArgumentError(
      absl::StrCat("ChunkProto[", chunk_idx, "] has unknown chunk type"));
}

void ApplyChunkProto(const ::tpu_sync::proto::ChunkProto& chunk,
                     absl::Span<uint8_t> dst_buffer) {
  if (chunk.has_scatter_chunk()) {
    ApplyScatterChunk(chunk.scatter_chunk(), dst_buffer);
  } else if (chunk.has_gather_chunk()) {
    ApplyGatherChunk(chunk.gather_chunk(), dst_buffer);
  }
}

}  // namespace tpu_raiden::weight_sync
