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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_TDS_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_TDS_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tdsul/tdsul.h"
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
#include "tpu_sync/kv_cache/backends/storage/storage_backend_utils.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"

namespace tpu_raiden {
namespace kv_cache {

class BlockTracker;

namespace backends {
namespace storage {

inline constexpr absl::string_view kTdsBackendName = "tds";

// Typed, validated view of a TDS backend's configuration.
struct TdsBackendOptions {
  std::string root_dir = "/tmp/raiden_storage";
  std::string model_name = "unknown";
  int tp_size = 1;
  int tp_rank = -1;  // Required; -1 means the caller did not supply it.
  size_t capacity_bytes = 0;
  size_t lookup_batch_size = 32;
  int storage_io_thread_pool_size = 16;
  bool direct_io = true;

  static absl::StatusOr<TdsBackendOptions> FromProperties(
      const absl::flat_hash_map<std::string, std::string>& properties);
};

// TdsKVBackend implements TPU Raiden's KVBackend interface on top of
// `libtdsul` (`//tpu_sync/tpudirect_storage:tdsul`).
class TdsKVBackend : public KVBackend {
 public:
  explicit TdsKVBackend(
      std::string name,
      absl::flat_hash_map<std::string, std::string> properties = {});

  ~TdsKVBackend() override;

  std::string name() const override { return name_; }

  const TdsBackendOptions& options() const { return options_; }

  bool is_direct_io_supported() const { return direct_io_supported_; }

  static bool ProbeDirectIO(absl::string_view dir);

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
  absl::StatusOr<std::vector<tds_buffer_io_t>> BuildBufferIos(
      absl::Span<const HostBufferDescriptor> slices) const;
  absl::StatusOr<bool> Exists(const BlockKey& key);

  std::string name_;
  TdsBackendOptions options_;
  bool direct_io_supported_ = false;
  // MUST be the last-declared member so in-flight tasks finish before any
  // member they touch is destroyed.
  std::unique_ptr<NumaThreadPool> thread_pool_;
};

// TdsPathMapper maps block hash identifiers and tensor-parallel rank to a
// rank-partitioned hierarchical directory layout over the hex-encoded hash:
//   `<root_dir>/<model_name>/tp<tp_size>_r<tp_rank>/<l1>/<l2>/<hash_hex>.bin`
class TdsPathMapper : public BlockKeyMapper {
 public:
  static absl::string_view GetParentDir(absl::string_view path) {
    size_t last_slash = path.find_last_of('/');
    if (last_slash == absl::string_view::npos) return "";
    return path.substr(0, last_slash);
  }

  TdsPathMapper(absl::string_view root_dir, absl::string_view model_name,
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

// TdsKVCacheStoreBackend probes persistent storage on the coordinator.
class TdsKVCacheStoreBackend : public KVCacheStoreBackend {
 public:
  explicit TdsKVCacheStoreBackend(
      std::shared_ptr<KVBackend> storage_backend,
      std::string name = std::string(kTdsBackendName),
      size_t capacity_bytes = 0, size_t lookup_batch_size = 32);

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
  absl::StatusOr<std::vector<BlockKey>> MapShardKeys(
      const std::string& block_hash) const;

  std::shared_ptr<KVBackend> storage_backend_;
  std::string name_ = std::string(kTdsBackendName);
  size_t capacity_bytes_ = 0;
  size_t lookup_batch_size_ = 32;
};

}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

namespace tkv {
namespace backends = ::tpu_raiden::kv_cache::backends;
}  // namespace tkv

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_TDS_BACKEND_H_
