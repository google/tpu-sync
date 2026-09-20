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

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "flatbuffers/base.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/lib/chunk_generated.h"

namespace tpu_raiden::transport::lib {

namespace {

void DeserializeHeaderV1(const flatbuf::ChunkHeader& h, ChunkHeader& header) {
  DCHECK_EQ(h.ver(), 1);
  header.version = h.ver();
  header.op = h.op();
  header.flags = h.flags();
  header.buffer_id = h.buffer_id();
  header.reserved = h.reserved();
  header.metadata_size = h.metadata_size();
  header.remote_id = h.remote_id();
  header.local_id = h.local_id();
  header.count_or_size = h.count_or_size();
  header.uuid = h.uuid();
}

void DeserializeHeaderV2(const flatbuf::ChunkHeader& h, ChunkHeader& header) {
  DCHECK_EQ(h.ver(), 2);
  header.version = h.ver();
  header.op = h.op();
  header.flags = h.flags();
  header.buffer_id = h.buffer_id();
  header.reserved = h.reserved();
  header.metadata_size = h.metadata_size();
  header.remote_id = h.remote_id();
  header.local_id = h.local_id();
  header.count_or_size = h.count_or_size();
  header.uuid = h.uuid();
}

absl::InlinedVector<char, kMaxMetadataSize> SerializeMetadataV1(
    const ChunkMetadata& meta) {
  const flatbuf::ChunkMetadataV1 m(meta.layer_idx, meta.dst_shard_idx,
                                   meta.dst_offset_bytes, meta.size_bytes);
  absl::InlinedVector<char, kMaxMetadataSize> bytes(sizeof(m));
  std::memcpy(bytes.data(), &m, sizeof(m));
  return bytes;
}

absl::InlinedVector<char, kMaxMetadataSize> SerializeMetadataV2(
    const ChunkMetadata& meta) {
  const flatbuf::ChunkMetadataV2 m(meta.layer_idx, meta.dst_shard_idx,
                                   meta.dst_offset_bytes, meta.size_bytes,
                                   meta.dst_stride_bytes, meta.count,
                                   /*padding=*/0);
  absl::InlinedVector<char, kMaxMetadataSize> bytes(sizeof(m));
  std::memcpy(bytes.data(), &m, sizeof(m));
  return bytes;
}

void DeserializeMetadataV1(const flatbuf::ChunkMetadataV1& m,
                           ChunkMetadata& meta) {
  meta.layer_idx = m.layer_idx();
  meta.dst_shard_idx = m.dst_shard_idx();
  meta.dst_offset_bytes = m.dst_offset_bytes();
  meta.size_bytes = m.size_bytes();
  meta.dst_stride_bytes = m.size_bytes();
  meta.count = 1;
  meta.padding = 0;
}

void DeserializeMetadataV2(const flatbuf::ChunkMetadataV2& m,
                           ChunkMetadata& meta) {
  meta.layer_idx = m.layer_idx();
  meta.dst_shard_idx = m.dst_shard_idx();
  meta.dst_offset_bytes = m.dst_offset_bytes();
  meta.size_bytes = m.size_bytes();
  meta.dst_stride_bytes = m.dst_stride_bytes();
  meta.count = m.count();
  meta.padding = m.padding();
}

}  // namespace

absl::InlinedVector<char, kChunkHeaderSize> SerializeChunkHeader(
    const ChunkHeader& header) {
  const uint16_t ver = header.version == 0 ? 1 : header.version;
  DCHECK(ver == 1 || ver == 2) << "Unsupported chunk header version: " << ver;
  const flatbuf::ChunkHeader h(
      kRaidenMagic, ver, header.op, header.flags, header.buffer_id,
      header.reserved, header.metadata_size, header.remote_id, header.local_id,
      header.count_or_size, header.uuid, /*padding0=*/0, /*padding1=*/0,
      /*padding2=*/0, /*padding3=*/0);

  absl::InlinedVector<char, kChunkHeaderSize> bytes(sizeof(h));
  std::memcpy(bytes.data(), &h, sizeof(h));
  DCHECK_EQ(bytes.size(), kChunkHeaderSize);
  return bytes;
}

absl::StatusOr<ChunkHeader> DeserializeChunkHeader(absl::Span<const char> s) {
  flatbuf::ChunkHeader h;
  DCHECK_EQ(sizeof(h), kChunkHeaderSize);
  if (s.size() != kChunkHeaderSize) {
    return absl::InvalidArgumentError("Invalid chunk header size");
  }

  std::memcpy(&h, s.data(), sizeof(h));

  if (h.magic() != kRaidenMagic) {
    return absl::InvalidArgumentError(
        absl::StrCat("Chunk header magic mismatch: expected ", kRaidenMagic,
                     ", got ", h.magic()));
  }

  const uint16_t ver = h.ver();
  switch (ver) {
    case 1: {
      ChunkHeader header = {};
      DeserializeHeaderV1(h, header);
      return header;
    }
    case 2: {
      ChunkHeader header = {};
      DeserializeHeaderV2(h, header);
      return header;
    }
    default:
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported chunk header flatbuf version: ", ver));
  }
}

absl::InlinedVector<char, kMaxMetadataSize> SerializeChunkMetadata(
    const ChunkMetadata& meta, uint16_t ver) {
  switch (ver) {
    case 1:
      return SerializeMetadataV1(meta);
    case 2:
      return SerializeMetadataV2(meta);
    default:
      DCHECK(false) << "Unsupported chunk metadata version: " << ver;
      return {};
  }
}

absl::StatusOr<ChunkMetadata> DeserializeChunkMetadata(absl::Span<const char> s,
                                                       uint16_t ver) {
  const size_t meta_size = GetChunkMetadataSize(ver);
  if (s.size() != meta_size) {
    return absl::InvalidArgumentError("Invalid chunk metadata size");
  }

  switch (ver) {
    case 1: {
      flatbuf::ChunkMetadataV1 m = {};
      std::memcpy(&m, s.data(), meta_size);
      ChunkMetadata metadata = {};
      DeserializeMetadataV1(m, metadata);
      return metadata;
    }
    case 2: {
      flatbuf::ChunkMetadataV2 m = {};
      std::memcpy(&m, s.data(), meta_size);
      ChunkMetadata metadata = {};
      DeserializeMetadataV2(m, metadata);
      return metadata;
    }
    default:
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported chunk metadata flatbuf version: ", ver));
  }
}

std::vector<uint8_t> SerializeBlockIds(absl::Span<const int> ids) {
  std::vector<uint8_t> buf(ids.size() * sizeof(uint32_t));
  for (size_t i = 0; i < ids.size(); ++i) {
    uint32_t val = flatbuffers::EndianScalar(static_cast<uint32_t>(ids[i]));
    std::memcpy(buf.data() + i * sizeof(uint32_t), &val, sizeof(uint32_t));
  }
  return buf;
}

std::vector<int> DeserializeBlockIds(absl::Span<const uint8_t> bytes) {
  DCHECK_EQ(bytes.size() % sizeof(uint32_t), 0);
  const size_t count = bytes.size() / sizeof(uint32_t);
  std::vector<int> ids;
  ids.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    uint32_t val = 0;
    std::memcpy(&val, bytes.data() + i * sizeof(uint32_t), sizeof(uint32_t));
    ids.push_back(static_cast<int>(flatbuffers::EndianScalar(val)));
  }
  return ids;
}

std::array<uint8_t, kChunkSizeFieldSize> SerializeChunkSize(
    uint32_t size_bytes) {
  std::array<uint8_t, kChunkSizeFieldSize> buf;
  uint32_t val = flatbuffers::EndianScalar(size_bytes);
  std::memcpy(buf.data(), &val, sizeof(uint32_t));
  return buf;
}

uint32_t DeserializeChunkSize(absl::Span<const uint8_t> bytes) {
  DCHECK_EQ(bytes.size(), kChunkSizeFieldSize);
  uint32_t val = 0;
  std::memcpy(&val, bytes.data(), sizeof(uint32_t));
  return flatbuffers::EndianScalar(val);
}

}  // namespace tpu_raiden::transport::lib
