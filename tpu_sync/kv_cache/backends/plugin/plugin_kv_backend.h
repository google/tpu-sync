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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_PLUGIN_PLUGIN_KV_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_PLUGIN_PLUGIN_KV_BACKEND_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/backends/plugin/raiden_storage_plugin_abi.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {

absl::Status ToAbslStatus(RaidenStatusCode code, absl::string_view message);

// BlockKeyMapper adapter that delegates key resolution to a C plugin's
// `map_key` callback, or passes raw block hashes through when `map_key` is
// NULL.
class PluginBlockKeyMapper : public BlockKeyMapper {
 public:
  PluginBlockKeyMapper(const RaidenStorageOutofTreePluginV1* vtable, void* ctx,
                       std::shared_ptr<void> ctx_lifetime,
                       ParallelismConfig parallelism);

  absl::StatusOr<BlockKey> MapKey(
      const std::string& block_hash,
      const KeyMappingOptions& options) const override;

  int tp_size() const override { return parallelism_.tp_size; }
  int pcp_size() const override { return parallelism_.pcp_size; }
  int pp_size() const override { return parallelism_.pp_size; }

 private:
  const RaidenStorageOutofTreePluginV1* const vtable_;
  void* const ctx_;
  std::shared_ptr<void> ctx_lifetime_;
  const ParallelismConfig parallelism_;
};

// KVBackend adapter that wraps a dynamically loaded
// `RaidenStorageOutofTreePluginV1` instance and marshals async I/O and
// existence probes across the C ABI.
class PluginKVBackend : public KVBackend {
 public:
  static absl::StatusOr<std::shared_ptr<PluginKVBackend>> Create(
      const RaidenStorageOutofTreePluginV1* vtable,
      absl::flat_hash_map<std::string, std::string> properties,
      const ParallelismConfig& parallelism);

  ~PluginKVBackend() override = default;

  std::string name() const override { return vtable_->type_name; }

  bool HasMapKey() const { return vtable_->map_key != nullptr; }

  absl::Status OnHostMemoryInitialized(
      absl::Span<const HostBufferDescriptor> shard_regions) override;

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
  PluginKVBackend(const RaidenStorageOutofTreePluginV1* vtable, void* ctx,
                  std::shared_ptr<void> ctx_lifetime,
                  absl::flat_hash_map<std::string, std::string> properties);

  const RaidenStorageOutofTreePluginV1* const vtable_;
  void* const ctx_;
  std::shared_ptr<void> ctx_lifetime_;
};

}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_PLUGIN_PLUGIN_KV_BACKEND_H_
