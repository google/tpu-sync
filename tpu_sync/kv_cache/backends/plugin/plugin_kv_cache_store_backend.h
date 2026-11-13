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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_PLUGIN_PLUGIN_KV_CACHE_STORE_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_PLUGIN_PLUGIN_KV_CACHE_STORE_BACKEND_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/backends/plugin/plugin_kv_backend.h"
#include "tpu_sync/kv_cache/backends/storage/storage_lookup_helper.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {

inline constexpr size_t kDefaultPluginLookupBatchSize = 32;

// Coordinator-side KVCacheStoreBackend for a storage tier that a
// RaidenStorageOutofTreePluginV1 plugin serves.
//
// It answers which blocks exist on the plugin's storage. No block payload
// passes through it: workers write and read blocks through PluginKVBackend.
//
// Lookup probes in one of two modes, chosen by the plugin's vtable:
// - Shard-key mode (`map_key` set): a block is available when the shard key of
//   every (pp_rank, pcp_rank, tp_rank) worker exists.
// - Block-level mode (`map_key` NULL): the raw block hash is the only key, and
//   the plugin decides whether the whole block exists.
//
// The backend keeps no existence cache in this version: Tier 0 host DRAM
// already serves repeated hits, and every Lookup probes the plugin. A block
// that storage evicted is therefore a miss on the next Lookup.
class OutofTreePluginKVCacheStoreBackend : public KVCacheStoreBackend {
 public:
  struct Options {
    size_t capacity_bytes = 0;
    // Number of keys per existence probe. 0 means the default.
    size_t lookup_batch_size = kDefaultPluginLookupBatchSize;
    // Label of the storage_metadata_check_total counter.
    std::string dp_rank = "0";
  };

  // `plugin_backend` is the coordinator's instance of the plugin and must not
  // be null. `name` is the backend type that workers register it under.
  OutofTreePluginKVCacheStoreBackend(
      std::shared_ptr<PluginKVBackend> plugin_backend, std::string name,
      Options options);

  std::string name() const override { return name_; }

  const Options& options() const { return options_; }

  // Queries `block_hashes` on the plugin's storage (longest prefix in kPrefix
  // mode, or all available blocks in kSubset mode; returns empty when
  // host_resident_only is set).
  absl::StatusOr<BlockSliceList> Lookup(
      absl::Span<const std::string> block_hashes,
      const LookupOptions& options = {}) override;

  // No-op on the storage tier (blocks are written via async offload transfers).
  // Returns false for kAdmitStrict, which applies only to host-RAM admission.
  bool Insert(absl::Span<const std::string> block_hashes,
              absl::Span<const RaidenBlockId> slices,
              InsertOption option = InsertOption::kBestEffortUpdate) override {
    return option != InsertOption::kAdmitStrict;
  }

  // Drops nothing and returns 0, because the backend holds no host-resident
  // entry and no cached key. Clears the output lists of `options`.
  size_t Delete(absl::Span<const std::string> block_hashes,
                const DeleteOptions& options = {}) override;

  bool Pin(absl::Span<const std::string> block_hashes) override { return true; }
  void Release(absl::Span<const std::string> block_hashes) override {}
  int GetPinCount(const std::string& hash) const override { return 0; }
  size_t GetCapacity() const override { return options_.capacity_bytes; }
  size_t GetSize() const override { return 0; }
  size_t GetAvailableSpace() const override { return options_.capacity_bytes; }

 private:
  // Returns the keys that must all exist for `block_hash` to be available.
  absl::StatusOr<std::vector<BlockKey>> ExpandKeys(
      const std::string& block_hash) const;

  const std::shared_ptr<PluginKVBackend> plugin_backend_;
  const std::string name_;
  const Options options_;
  const storage::StorageLookupHelper lookup_helper_;
};

}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_PLUGIN_PLUGIN_KV_CACHE_STORE_BACKEND_H_
