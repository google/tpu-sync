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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_OFFLOAD_KV_CACHE_METADATA_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_OFFLOAD_KV_CACHE_METADATA_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace tpu_raiden {
namespace kv_cache {

// Fixed-layout header at the start of a KV cache metadata region, aligned
// with SharedMemoryHeader in tpu_sync/core/host_memory_allocator.h.
struct alignas(64) KVCacheMetadataHeader {
  uint64_t magic = 0x52414944454E4B4D;  // "RAIDENKM", offset 0, 8 bytes
  uint32_t version = 1;                 // offset 8, 4 bytes
  uint32_t num_blocks = 0;              // offset 12, 4 bytes
  uint32_t entry_size = 0;              // offset 16, 4 bytes
  // Identifies the model whose KV blocks the table describes, mirroring the
  // model_uid of the shared-memory KV pool (see SharedMemoryHeader). Attach
  // rejects a table recorded under a different model_uid, so a surviving
  // table can never resurrect bindings for another model's blocks.
  char model_uid[64] = {0};  // offset 20, 64 bytes
};

static_assert(sizeof(KVCacheMetadataHeader) == 128,
              "KVCacheMetadataHeader layout is part of the on-memory format");

// One fixed-size slot per block, indexed by block ID. The layout is a
// persistent format read back after a crash, so every byte of the 128-byte
// entry is spelled out explicitly rather than left to compiler padding.
struct alignas(64) KVCacheMetadataEntry {
  // Recency stamp assigned at each LRU cache insert, used to rebuild LRU
  // order on recovery and to pick the newest binding if two slots ever claim
  // the same hash.
  uint64_t seq = 0;       // offset 0, 8 bytes
  uint16_t hash_len = 0;  // offset 8, 2 bytes
  // Distinguishes an occupied slot from an empty/cleared one. Committed last
  // with release ordering, so a crash mid-write leaves the entry invalid
  // instead of exposing a partially written hash.
  std::atomic<uint8_t> valid{0};  // offset 10, 1 byte
  // Explicit padding pinning `hash` to offset 16; available for future
  // fields without a layout change.
  uint8_t reserved[5] = {0};  // offset 11, 5 bytes
  char hash[112] = {0};       // offset 16, 112 bytes
};

static_assert(sizeof(KVCacheMetadataEntry) == 128,
              "KVCacheMetadataEntry layout is part of the on-memory format");
static_assert(std::atomic<uint8_t>::is_always_lock_free);

// Table of prefix-hash-to-block bindings maintained in a caller-provided
// memory region, one fixed-size entry per block. Kept in the same shared
// memory that backs the host KV block pool, it mirrors the KVCacheStore
// LRU cache and survives an engine crash together with the block data, so a
// restarted engine can rebuild its LRU cache locally without consulting the
// global registry.
//
// The class is a non-owning view: the region must outlive it. Plain memory
// writes only — shared memory persists across a process crash, so no flush or
// sync is required, but callers must pass a region that lives in such memory
// for the table to be recoverable.
//
// This class is not thread-safe and requires external synchronization for
// concurrent access.
class KVCacheMetadata {
 public:
  // Maximum hash length one entry can record.
  static constexpr size_t kMaxHashLength = sizeof(KVCacheMetadataEntry::hash);

  // Byte size a region must have to hold a table for `num_blocks` blocks.
  static size_t RequiredSizeBytes(int num_blocks);

  // Initializes `region` as an empty table for `num_blocks` blocks of
  // `model_uid`'s KV data (cold start), overwriting any previous content.
  // The region must be at least RequiredSizeBytes(num_blocks) large and
  // 64-byte aligned.
  static absl::StatusOr<KVCacheMetadata> Format(
      absl::Span<uint8_t> region, int num_blocks,
      absl::string_view model_uid = "");

  // Attaches to a previously formatted region (warm restart). Fails if the
  // header does not carry the expected magic, version, entry size,
  // `num_blocks`, or `model_uid`; callers should treat a failure as a cold
  // start and Format.
  static absl::StatusOr<KVCacheMetadata> Attach(
      absl::Span<uint8_t> region, int num_blocks,
      absl::string_view model_uid = "");

  // Records that `block_id` holds the data identified by `hash`. `seq` is a
  // monotonically increasing recency stamp used to order entries during
  // recovery. Overwrites any previous binding for the block.
  absl::Status Set(int block_id, absl::string_view hash, uint64_t seq);

  // Removes the binding for `block_id`, if any.
  absl::Status Clear(int block_id);

  struct Entry {
    int block_id;
    std::string hash;
    uint64_t seq;
  };

  // Returns all currently valid entries.
  std::vector<Entry> ValidEntries() const;

  int num_blocks() const { return num_blocks_; }

 private:
  KVCacheMetadata(KVCacheMetadataHeader* header, KVCacheMetadataEntry* entries,
                  int num_blocks);

  KVCacheMetadataHeader* header_;
  KVCacheMetadataEntry* entries_;
  int num_blocks_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_OFFLOAD_KV_CACHE_METADATA_H_
