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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_METADATA_SHM_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_METADATA_SHM_H_

#include <cstddef>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/kv_cache_metadata.h"

namespace tpu_raiden {
namespace kv_cache {

// The shm name (sans "/" prefix) of the store's crash-persistent KV metadata
// table: RAIDEN_SHM_KEY + "_metadata", an optional "_<RAIDEN_SHM_SERVER_NAME>"
// suffix, then the store's sanitized RaidenId. The RaidenId keeps the table
// private to its store.
std::string MetadataShmKey(const RaidenId& raiden_id);

// Owns the POSIX shared-memory mapping that backs a KVCacheMetadata table
// and keeps it alive for the lifetime of this object.
//
// AttachOrFormat mirrors the warm-boot pattern of
// SharedMemoryHostMemoryAllocator: it first tries to attach to a table that
// survived a previous engine incarnation under `shm_key`; if the segment is
// missing, undersized, or fails table validation (magic, version, entry
// size, num_blocks, model_uid), the segment is re-created and formatted
// empty (cold start). The segment is never unlinked on destruction — the
// table must outlive the process to be recoverable.
class KVCacheMetadataShmRegion {
 public:
  // Attaches to or creates the shared-memory segment named `shm_key` (a
  // leading '/' is added if missing) sized for a table of `num_blocks`
  // blocks of `model_uid`'s KV data. Never returns a partially initialized
  // region: on error the segment is left for the next attempt.
  static absl::StatusOr<std::unique_ptr<KVCacheMetadataShmRegion>>
  AttachOrFormat(absl::string_view shm_key, int num_blocks,
                 absl::string_view model_uid = "");

  ~KVCacheMetadataShmRegion();

  KVCacheMetadataShmRegion(const KVCacheMetadataShmRegion&) = delete;
  KVCacheMetadataShmRegion& operator=(const KVCacheMetadataShmRegion&) =
      delete;

  // View into the mapped table; valid for the lifetime of this region.
  const KVCacheMetadata& metadata() const { return metadata_; }

  // True when the table survived from a previous incarnation (warm restart);
  // false when it was freshly formatted (cold start).
  bool warm() const { return warm_; }

  // Reformats the mapped table in place, erasing all entries. Used to fall
  // back to a cold start when recovery from a surviving table fails.
  // Existing KVCacheMetadata copies attached to this region stay valid and
  // observe the emptied table.
  absl::Status Reformat();

 private:
  KVCacheMetadataShmRegion(int fd, void* mapped, size_t mapped_size,
                           KVCacheMetadata metadata, bool warm, int num_blocks,
                           std::string model_uid);

  int fd_;
  void* mapped_;
  size_t mapped_size_;
  KVCacheMetadata metadata_;
  bool warm_;
  int num_blocks_;
  std::string model_uid_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_METADATA_SHM_H_
