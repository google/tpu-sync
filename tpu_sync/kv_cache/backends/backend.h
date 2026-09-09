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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_BACKEND_H_

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

extern "C" char** environ;

namespace tpu_raiden {
namespace kv_cache {
namespace backends {

// BlockKey represents a resolved key on an offload or persistence backend
// (e.g., local disk, Lustre filesystem, cloud object store, or remote
// memory/network). It bundles:
// - `block_hash`: The content-addressable hash identifying the cache block
// data.
// - `resolved_key`: The target backend locator, URI, or resolved file path.
// - `offset`: The linear byte offset within the backend medium, file, or
// object.
// - `size`: The byte size of this slice.
struct BlockKey {
  std::string block_hash;
  std::string resolved_key;  // Fully resolved backend locator, URI, or path
  // Linear byte offset in the target backend medium. Required when multiple
  // workers write to the same backend entity (e.g. file) to support cross-TP
  // reads and writes.
  int64_t offset = 0;
  int64_t size = 0;          // Byte size of this slice
};

// Encapsulates a single disjoint memory slice within a multi-layer, multi-shard
// block in Host DRAM.
struct BackendBufferDescriptor {
  void* ptr = nullptr;  // Host virtual address
  size_t size = 0;      // Slice size in bytes
  int fd = -1;          // Optional file descriptor (direct/splice I/O)
  int64_t offset = 0;   // Optional byte offset within fd
};

class BlockKeyMapper;

// Stateless backend transfer and I/O interface.
// Represents a generic offload/persistence medium (e.g. POSIX filesystem,
// Lustre, cloud object store, or remote memory pool). Implementations provide
// non-blocking read and write capabilities between host DRAM and the backend.
class KVBackend : public std::enable_shared_from_this<KVBackend> {
 public:
  explicit KVBackend(
      absl::flat_hash_map<std::string, std::string> properties = {})
      : properties_(std::move(properties)) {}

  virtual ~KVBackend() = default;

  virtual std::shared_ptr<BlockKeyMapper> mapper() const { return mapper_; }
  virtual void set_mapper(std::shared_ptr<BlockKeyMapper> mapper) {
    mapper_ = std::move(mapper);
  }

  const absl::flat_hash_map<std::string, std::string>& properties() const {
    return properties_;
  }
  std::string GetProperty(absl::string_view key,
                          absl::string_view default_val = "") const {
    auto it = properties_.find(key);
    return it != properties_.end() ? it->second : std::string(default_val);
  }
  int64_t GetIntProperty(absl::string_view key, int64_t default_val = 0) const {
    auto it = properties_.find(key);
    if (it == properties_.end()) return default_val;
    int64_t val = default_val;
    if (absl::SimpleAtoi(it->second, &val)) return val;
    return default_val;
  }
  bool GetBoolProperty(absl::string_view key, bool default_val = false) const {
    auto it = properties_.find(key);
    if (it == properties_.end()) return default_val;
    return it->second == "true" || it->second == "1";
  }

  // Returns the canonical backend name (e.g. "posix", "lustre").
  virtual std::string name() const = 0;

  // Asynchronously writes a sequence of host DRAM buffer slices into the target
  // key. Works uniformly for a single slice (passed as `{slice}`) or
  // multi-slice blocks.
  virtual void WriteAsync(const BlockKey& key,
                          absl::Span<const BackendBufferDescriptor> slices,
                          size_t total_bytes,
                          std::function<void(absl::Status)> callback) = 0;

  // Asynchronously reads from the target key into a sequence of host DRAM
  // buffer slices. Works uniformly for a single slice (passed as `{slice}`) or
  // multi-slice blocks.
  virtual void ReadAsync(const BlockKey& key,
                         absl::Span<const BackendBufferDescriptor> slices,
                         size_t total_bytes,
                         std::function<void(absl::Status)> callback) = 0;

  // Asynchronously checks whether a batch of blocks exist on the backend.
  virtual void BatchExistsAsync(
      absl::Span<const BlockKey> keys,
      std::function<void(std::vector<absl::StatusOr<bool>>)> callback) = 0;

 protected:
  absl::flat_hash_map<std::string, std::string> properties_;
  std::shared_ptr<BlockKeyMapper> mapper_;
};

// Options for mapping a cache block hash to a backend key/path.
struct KeyMappingOptions {
  int rank =
      -1;  // Target TP rank (-1 indicates using mapper's configured rank_)
  int tp_size =
      -1;  // Target TP size (-1 indicates using mapper's configured tp_size_)
  // Extensible for future metadata without breaking interface (e.g., shard_idx,
  // layer_idx)
};

// BlockKeyMapper defines the coordinator-side locator resolution policy.
// It maps block hash identifiers and rank distributions to backend-specific
// locators and slice boundaries across any offload medium (storage, memory, or
// network).
class BlockKeyMapper {
 public:
  virtual ~BlockKeyMapper() = default;

  // Canonical mapping API taking extensible options.
  virtual BlockKey MapKey(const std::string& block_hash,
                          const KeyMappingOptions& options = {}) const = 0;
  virtual int tp_size() const = 0;
};

inline absl::flat_hash_map<std::string, std::string> ResolveBackendProperties(
    absl::string_view name,
    const absl::flat_hash_map<std::string, std::string>& base_properties = {}) {
  absl::flat_hash_map<std::string, std::string> resolved = base_properties;
  if (::environ == nullptr) return resolved;

  constexpr absl::string_view kPrefix = "RAIDEN_BACKEND_EXTRA_";
  std::string prefix = absl::StrCat(kPrefix, absl::AsciiStrToUpper(name), "_");

  absl::flat_hash_map<std::string, std::string> scoped_env;
  absl::flat_hash_map<std::string, std::string> global_env;

  for (char** env = ::environ; *env != nullptr; ++env) {
    absl::string_view entry(*env);
    if (!absl::StartsWith(entry, kPrefix)) continue;

    std::pair<absl::string_view, absl::string_view> kv =
        absl::StrSplit(entry, absl::MaxSplits('=', 1));
    if (kv.second.empty()) continue;

    if (absl::StartsWith(kv.first, prefix)) {
      std::string key = absl::AsciiStrToLower(kv.first.substr(prefix.size()));
      scoped_env[key] = std::string(kv.second);
    } else {
      std::string key = absl::AsciiStrToLower(kv.first.substr(kPrefix.size()));
      global_env[key] = std::string(kv.second);
    }
  }

  for (const auto& [k, v] : global_env) {
    resolved.try_emplace(k, v);
  }
  for (const auto& [k, v] : scoped_env) {
    if (base_properties.find(k) == base_properties.end()) {
      resolved[k] = v;
    }
  }
  return resolved;
}

}  // namespace backends

namespace storage {
using ::tpu_raiden::kv_cache::backends::BackendBufferDescriptor;
using ::tpu_raiden::kv_cache::backends::BlockKey;
using ::tpu_raiden::kv_cache::backends::BlockKeyMapper;
using ::tpu_raiden::kv_cache::backends::KeyMappingOptions;
using ::tpu_raiden::kv_cache::backends::KVBackend;
using ::tpu_raiden::kv_cache::backends::ResolveBackendProperties;
}  // namespace storage

namespace backend {
using ::tpu_raiden::kv_cache::backends::BackendBufferDescriptor;
using ::tpu_raiden::kv_cache::backends::BlockKey;
using ::tpu_raiden::kv_cache::backends::BlockKeyMapper;
using ::tpu_raiden::kv_cache::backends::KeyMappingOptions;
using ::tpu_raiden::kv_cache::backends::KVBackend;
using ::tpu_raiden::kv_cache::backends::ResolveBackendProperties;
}  // namespace backend

using backends::BackendBufferDescriptor;
using backends::BlockKey;
using backends::BlockKeyMapper;
using backends::KeyMappingOptions;
using backends::KVBackend;
using backends::ResolveBackendProperties;

}  // namespace kv_cache

using kv_cache::backends::BackendBufferDescriptor;
using kv_cache::backends::BlockKey;
using kv_cache::backends::BlockKeyMapper;
using kv_cache::backends::KeyMappingOptions;
using kv_cache::backends::KVBackend;
using kv_cache::backends::ResolveBackendProperties;

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_BACKEND_H_
