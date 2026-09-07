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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

absl::Status ValidateRegion(absl::Span<uint8_t> region, int num_blocks,
                            absl::string_view model_uid) {
  if (num_blocks <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("num_blocks must be positive, got ", num_blocks));
  }
  if (model_uid.size() >= sizeof(KVCacheMetadataHeader::model_uid)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "model_uid length ", model_uid.size(), " exceeds the maximum of ",
        sizeof(KVCacheMetadataHeader::model_uid) - 1));
  }
  if (region.data() == nullptr) {
    return absl::InvalidArgumentError("Metadata region is null");
  }
  if (reinterpret_cast<uintptr_t>(region.data()) %
          alignof(KVCacheMetadataHeader) !=
      0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Metadata region must be ", alignof(KVCacheMetadataHeader),
        "-byte aligned"));
  }
  size_t required = KVCacheMetadata::RequiredSizeBytes(num_blocks);
  if (region.size() < required) {
    return absl::InvalidArgumentError(
        absl::StrCat("Metadata region too small: ", region.size(),
                     " bytes, need ", required));
  }
  return absl::OkStatus();
}

}  // namespace

size_t KVCacheMetadata::RequiredSizeBytes(int num_blocks) {
  return sizeof(KVCacheMetadataHeader) +
         static_cast<size_t>(num_blocks) * sizeof(KVCacheMetadataEntry);
}

absl::StatusOr<KVCacheMetadata> KVCacheMetadata::Format(
    absl::Span<uint8_t> region, int num_blocks, absl::string_view model_uid) {
  ABSL_RETURN_IF_ERROR(ValidateRegion(region, num_blocks, model_uid));

  std::memset(region.data(), 0, RequiredSizeBytes(num_blocks));
  auto* header = reinterpret_cast<KVCacheMetadataHeader*>(region.data());
  KVCacheMetadataHeader fresh;
  fresh.num_blocks = static_cast<uint32_t>(num_blocks);
  fresh.entry_size = sizeof(KVCacheMetadataEntry);
  std::memcpy(fresh.model_uid, model_uid.data(), model_uid.size());
  std::memcpy(header, &fresh, sizeof(KVCacheMetadataHeader));

  auto* entries = reinterpret_cast<KVCacheMetadataEntry*>(
      region.data() + sizeof(KVCacheMetadataHeader));
  return KVCacheMetadata(header, entries, num_blocks);
}

absl::StatusOr<KVCacheMetadata> KVCacheMetadata::Attach(
    absl::Span<uint8_t> region, int num_blocks, absl::string_view model_uid) {
  ABSL_RETURN_IF_ERROR(ValidateRegion(region, num_blocks, model_uid));

  auto* header = reinterpret_cast<KVCacheMetadataHeader*>(region.data());
  const KVCacheMetadataHeader expected;
  if (header->magic != expected.magic) {
    return absl::FailedPreconditionError("Metadata region magic mismatch");
  }
  if (header->version != expected.version) {
    return absl::FailedPreconditionError(
        absl::StrCat("Metadata region version mismatch: ", header->version,
                     " vs ", expected.version));
  }
  if (header->entry_size != sizeof(KVCacheMetadataEntry)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Metadata entry size mismatch: ", header->entry_size,
                     " vs ", sizeof(KVCacheMetadataEntry)));
  }
  if (header->num_blocks != static_cast<uint32_t>(num_blocks)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Metadata num_blocks mismatch: ", header->num_blocks,
                     " vs ", num_blocks));
  }
  absl::string_view recorded_uid(
      header->model_uid,
      strnlen(header->model_uid, sizeof(header->model_uid)));
  if (recorded_uid != model_uid) {
    return absl::FailedPreconditionError(
        absl::StrCat("Metadata model_uid mismatch: \"", recorded_uid,
                     "\" vs \"", model_uid, "\""));
  }

  auto* entries = reinterpret_cast<KVCacheMetadataEntry*>(
      region.data() + sizeof(KVCacheMetadataHeader));
  return KVCacheMetadata(header, entries, num_blocks);
}

KVCacheMetadata::KVCacheMetadata(KVCacheMetadataHeader* header,
                                 KVCacheMetadataEntry* entries, int num_blocks)
    : header_(header), entries_(entries), num_blocks_(num_blocks) {}

absl::Status KVCacheMetadata::Set(int block_id, absl::string_view hash,
                                  uint64_t seq) {
  if (block_id < 0 || block_id >= num_blocks_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Block ID out of range: ", block_id));
  }
  if (hash.empty()) {
    return absl::InvalidArgumentError("Hash must not be empty");
  }
  if (hash.size() > kMaxHashLength) {
    return absl::InvalidArgumentError(
        absl::StrCat("Hash length ", hash.size(), " exceeds the maximum of ",
                     kMaxHashLength));
  }

  KVCacheMetadataEntry& entry = entries_[block_id];
  // Invalidate first so a crash below leaves a cleanly missing entry, then
  // commit with a release store after all fields are written.
  entry.valid.store(0, std::memory_order_release);
  entry.seq = seq;
  entry.hash_len = static_cast<uint16_t>(hash.size());
  std::memcpy(entry.hash, hash.data(), hash.size());
  entry.valid.store(1, std::memory_order_release);
  return absl::OkStatus();
}

absl::Status KVCacheMetadata::Clear(int block_id) {
  if (block_id < 0 || block_id >= num_blocks_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Block ID out of range: ", block_id));
  }
  entries_[block_id].valid.store(0, std::memory_order_release);
  return absl::OkStatus();
}

std::vector<KVCacheMetadata::Entry> KVCacheMetadata::ValidEntries() const {
  std::vector<Entry> result;
  for (int i = 0; i < num_blocks_; ++i) {
    const KVCacheMetadataEntry& entry = entries_[i];
    if (entry.valid.load(std::memory_order_acquire) == 0) {
      continue;
    }
    if (entry.hash_len == 0 || entry.hash_len > kMaxHashLength) {
      continue;
    }
    result.push_back(Entry{i, std::string(entry.hash, entry.hash_len),
                           entry.seq});
  }
  return result;
}

}  // namespace kv_cache
}  // namespace tpu_raiden
