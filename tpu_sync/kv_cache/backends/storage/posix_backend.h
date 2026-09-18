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

// Canonical identifier for the POSIX storage tier.
inline constexpr absl::string_view kPosixBackendName = "posix";

inline constexpr size_t kDefaultLookupBatchSize = 32;

// Prefix-directory widths, in hex characters, over the encoded block hash.
// 3 + 2 yields 16^5 == 1,048,576 leaf directories per topology namespace.
inline constexpr size_t kHashDirL1Width = 3;
inline constexpr size_t kHashDirL2Width = 2;
// Hex encoding doubles the length, and the ".bin" suffix costs 4 more
// characters: 2*125 + 4 == 254 <= NAME_MAX (255).
inline constexpr size_t kMaxBlockHashBytes = 125;

// Typed, validated view of a POSIX backend's configuration. Every POSIX knob
// lives here; `FromProperties` is the single parse-and-validate point.
struct PosixBackendOptions {
  std::string root_dir = "/tmp/raiden_storage";
  std::string model_name = "unknown";
  int tp_size = 1;
  int tp_rank = -1;  // Required; -1 means the caller did not supply it.
  size_t capacity_bytes = 0;
  size_t lookup_batch_size = kDefaultLookupBatchSize;

  // Worker threads in this backend instance's NumaThreadPool. Every async
  // POSIX file operation runs on it -- WriteAsync, ReadAsync and
  // BatchExistsAsync share one pool and one queue, one task per block -- so
  // this is also the ceiling on concurrent pwrite/pread/stat syscalls issued
  // by this backend. Not per-NUMA-node, not per-file-descriptor.
  int storage_io_thread_pool_size = 16;

  // Parses and validates `properties`. Returns InvalidArgumentError for an
  // unparseable value, a negative thread-pool size, or a missing tp_rank.
  static absl::StatusOr<PosixBackendOptions> FromProperties(
      const absl::flat_hash_map<std::string, std::string>& properties);
};

// PosixKVBackend implements KVBackend for POSIX filesystems (e.g., Lustre,
// local disk).
class PosixKVBackend : public KVBackend {
 public:
  // Both arguments are required: `properties` must carry the topology
  // (`tp_rank`, and `tp_size` when sharded), which PosixBackendOptions
  // validates. Defaulting them would turn a missing rank into a runtime
  // LOG(FATAL) instead of a compile error.
  PosixKVBackend(std::string name,
                 absl::flat_hash_map<std::string, std::string> properties);

  std::string name() const override { return name_; }

  const PosixBackendOptions& options() const { return options_; }

  void WriteAsync(const BlockKey& key,
                  absl::Span<const HostBufferDescriptor> slices,
                  size_t total_bytes,
                  std::function<void(absl::Status)> callback) override;

  void ReadAsync(const BlockKey& key,
                 absl::Span<const HostBufferDescriptor> slices,
                 size_t total_bytes,
                 std::function<void(absl::Status)> callback) override;

  void BatchExistsAsync(
      absl::Span<const BlockKey> keys,
      std::function<void(std::vector<absl::StatusOr<bool>>)> callback) override;

 private:
  absl::StatusOr<bool> Exists(const BlockKey& key);

  std::string name_;
  PosixBackendOptions options_;
  // MUST be the last-declared member: destruction runs in reverse declaration
  // order, and ~NumaThreadPool joins its workers after draining the queue, so
  // declaring it last is what guarantees every in-flight task finishes before
  // any member it might touch is destroyed. The scheduled lambdas capture raw
  // `this` and rely on this.
  std::unique_ptr<NumaThreadPool> thread_pool_;
};

// PosixPathMapper implements the filesystem path resolution policy.
// Maps block hash identifiers and tensor-parallel rank to a rank-partitioned
// hierarchical directory layout over the HEX-ENCODED block hash (see
// BlockKeyMapper::MapKey):
//   `<root_dir>/<model_name>/tp<tp_size>_r<tp_rank>/<l1>/<l2>/<hash_hex>.bin`
class PosixPathMapper : public BlockKeyMapper {
 public:
  static absl::string_view GetParentDir(absl::string_view path) {
    size_t last_slash = path.find_last_of('/');
    if (last_slash == absl::string_view::npos) return "";
    return path.substr(0, last_slash);
  }

  PosixPathMapper(absl::string_view root_dir, absl::string_view model_name,
                  int tp_size, int tp_rank);

  absl::StatusOr<BlockKey> MapKey(
      const std::string& block_hash,
      const KeyMappingOptions& options = {}) const override;
  int tp_size() const override { return tp_size_; }

 private:
  std::string root_dir_;
  std::string model_name_;
  int tp_size_;
  int tp_rank_;
};

// PosixKVCacheStoreBackend probes persistent storage (e.g., Lustre, POSIX).
class PosixKVCacheStoreBackend : public KVCacheStoreBackend {
 public:
  PosixKVCacheStoreBackend(std::shared_ptr<KVBackend> storage_backend,
                           std::string name = std::string(kPosixBackendName),
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
  std::string name_ = std::string(kPosixBackendName);
  size_t capacity_bytes_ = 0;
  size_t lookup_batch_size_ = 32;
};

}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_POSIX_BACKEND_H_
