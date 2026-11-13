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

#include "tpu_sync/kv_cache/backends/plugin/plugin_kv_cache_store_backend.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/backends/plugin/plugin_kv_backend.h"
#include "tpu_sync/kv_cache/backends/storage/storage_lookup_helper.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {

namespace {

OutofTreePluginKVCacheStoreBackend::Options Normalized(
    OutofTreePluginKVCacheStoreBackend::Options options) {
  if (options.lookup_batch_size == 0) {
    options.lookup_batch_size = kDefaultPluginLookupBatchSize;
  }
  if (options.dp_rank.empty()) options.dp_rank = "0";
  return options;
}

}  // namespace

OutofTreePluginKVCacheStoreBackend::OutofTreePluginKVCacheStoreBackend(
    std::shared_ptr<PluginKVBackend> plugin_backend, std::string name,
    Options options)
    : plugin_backend_(std::move(plugin_backend)),
      name_(std::move(name)),
      options_(Normalized(std::move(options))),
      lookup_helper_(name_, options_.dp_rank, options_.lookup_batch_size) {
  CHECK(plugin_backend_ != nullptr);
}

absl::StatusOr<std::vector<BlockKey>>
OutofTreePluginKVCacheStoreBackend::ExpandKeys(
    const std::string& block_hash) const {
  if (plugin_backend_->HasMapKey()) {
    return storage::StorageLookupHelper::ExpandShardKeys(
        *plugin_backend_->mapper(), block_hash);
  }
  // Without map_key the plugin probes whole blocks, so the raw hash is the
  // only key and it is not expanded per rank.
  if (block_hash.empty()) {
    return absl::InvalidArgumentError("block_hash must not be empty");
  }
  return std::vector<BlockKey>{
      BlockKey{.block_hash = block_hash, .resolved_key = block_hash}};
}

absl::StatusOr<BlockSliceList> OutofTreePluginKVCacheStoreBackend::Lookup(
    absl::Span<const std::string> block_hashes, const LookupOptions& options) {
  return lookup_helper_.Lookup(
      block_hashes, options,
      [this](const std::string& block_hash) { return ExpandKeys(block_hash); },
      [this](absl::Span<const BlockKey> keys) {
        return storage::StorageLookupHelper::ProbeExists(*plugin_backend_,
                                                         keys);
      });
}

size_t OutofTreePluginKVCacheStoreBackend::Delete(
    absl::Span<const std::string> block_hashes, const DeleteOptions& options) {
  if (options.deallocated_host_block_ids != nullptr) {
    options.deallocated_host_block_ids->clear();
  }
  if (options.deleted_hashes != nullptr) {
    options.deleted_hashes->clear();
  }
  return 0;
}

}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden
