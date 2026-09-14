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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_POSIX_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_POSIX_BACKEND_H_

#include <fcntl.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"

namespace tpu_raiden {
namespace kv_cache {

class BlockTracker;

namespace backends {
namespace storage {

// PosixKVBackend implements KVBackend for POSIX filesystems (e.g., Lustre,
// local disk).
class PosixKVBackend : public KVBackend {
 public:
  explicit PosixKVBackend(
      std::string name = "PosixKVCacheStoreBackend",
      absl::flat_hash_map<std::string, std::string> properties = {});

  std::string name() const override { return name_; }

  void WriteAsync(const BlockKey& key,
                  absl::Span<const BackendBufferDescriptor> slices,
                  size_t total_bytes,
                  std::function<void(absl::Status)> callback) override;

  void ReadAsync(const BlockKey& key,
                 absl::Span<const BackendBufferDescriptor> slices,
                 size_t total_bytes,
                 std::function<void(absl::Status)> callback) override;

  void BatchExistsAsync(
      absl::Span<const BlockKey> keys,
      std::function<void(std::vector<absl::StatusOr<bool>>)> callback) override;

 private:
  absl::StatusOr<bool> Exists(const BlockKey& key);

  std::string name_;
  std::unique_ptr<NumaThreadPool> thread_pool_;
};

// PosixPathMapper implements rank-partitioned hierarchical directory layout
// mapping:
// `<root_dir>/<model_name>_tp<tp_size>_r<rank>/<l1>/<l2>/<block_hash>.bin`
class PosixPathMapper : public BlockKeyMapper {
 public:
  static absl::string_view GetParentDir(absl::string_view path) {
    size_t last_slash = path.find_last_of('/');
    if (last_slash == absl::string_view::npos) return "";
    return path.substr(0, last_slash);
  }

  PosixPathMapper(absl::string_view root_dir, absl::string_view model_name,
                  int tp_size, int rank);

  BlockKey MapKey(const std::string& block_hash,
                  const KeyMappingOptions& options = {}) const override;
  int tp_size() const override { return tp_size_; }

 private:
  std::string root_dir_;
  std::string model_name_;
  int tp_size_;
  int rank_;
};

inline constexpr size_t kDefaultLookupBatchSize = 32;

// PosixKVCacheStoreBackend probes persistent storage (e.g., Lustre, POSIX).
class PosixKVCacheStoreBackend : public KVCacheStoreBackend {
 public:
  PosixKVCacheStoreBackend(std::shared_ptr<KVBackend> storage_backend,
                           std::string name = "PosixKVCacheStoreBackend",
                           size_t capacity_bytes = 0,
                           size_t lookup_batch_size = kDefaultLookupBatchSize)
      : storage_backend_(std::move(storage_backend)),
        name_(std::move(name)),
        capacity_bytes_(capacity_bytes),
        lookup_batch_size_(lookup_batch_size > 0 ? lookup_batch_size
                                                 : kDefaultLookupBatchSize) {}

  std::string name() const override { return name_; }

  size_t lookup_batch_size() const { return lookup_batch_size_; }

  absl::StatusOr<BlockSliceList> Lookup(
      absl::Span<const std::string> block_hashes,
      const LookupOptions& options = {}) override;

  tsl::Future<> Load(const RaidenId& remote_id,
                     absl::Span<const std::string> block_hashes,
                     absl::Span<const int32_t> device_block_ids,
                     absl::Span<const RaidenBlockId> slices,
                     BlockTracker* absl_nonnull load_tracker) override {
    return tsl::Future<>(absl::OkStatus());
  }

  std::pair<bool, BlockSliceList> Insert(
      absl::Span<const std::string> block_hashes,
      absl::Span<const RaidenBlockId> slices, bool on_host) override {
    return {true, {}};
  }

  bool InsertAndLock(absl::Span<const std::string> block_hashes,
                     absl::Span<const RaidenBlockId> slices,
                     bool on_host) override {
    return true;
  }

  size_t ReleaseAndDelete(absl::Span<const std::string> block_hashes) override {
    return 0;
  }
  void Delete(absl::Span<const std::string> block_hashes,
              absl::Span<const RaidenBlockId> slices) override {}
  bool Pin(absl::Span<const std::string> block_hashes) override { return true; }
  void Release(absl::Span<const std::string> block_hashes) override {}
  int GetPinCount(const std::string& hash) const override { return 0; }
  size_t GetCapacity() const override { return capacity_bytes_; }
  size_t GetSize() const override { return 0; }
  size_t GetAvailableSpace() const override { return capacity_bytes_; }

  std::shared_ptr<KVBackend> storage_backend() const {
    return storage_backend_;
  }

 private:
  std::shared_ptr<KVBackend> storage_backend_;
  std::string name_ = "PosixKVCacheStoreBackend";
  size_t capacity_bytes_ = 0;
  size_t lookup_batch_size_ = 32;
};

}  // namespace storage
}  // namespace backends

namespace storage {
using ::tpu_raiden::kv_cache::backends::storage::kDefaultLookupBatchSize;
using ::tpu_raiden::kv_cache::backends::storage::PosixKVBackend;
using ::tpu_raiden::kv_cache::backends::storage::PosixKVCacheStoreBackend;
using ::tpu_raiden::kv_cache::backends::storage::PosixPathMapper;
}  // namespace storage

using backends::storage::kDefaultLookupBatchSize;
using backends::storage::PosixKVBackend;
using backends::storage::PosixKVCacheStoreBackend;
using backends::storage::PosixPathMapper;

}  // namespace kv_cache

using kv_cache::backends::storage::kDefaultLookupBatchSize;
using kv_cache::backends::storage::PosixKVBackend;
using kv_cache::backends::storage::PosixKVCacheStoreBackend;
using kv_cache::backends::storage::PosixPathMapper;

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_POSIX_BACKEND_H_
