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
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

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
// block.
//
// PRECONDITION: `ptr` MUST address Host DRAM.
struct HostBufferDescriptor {
  void* ptr = nullptr;  // Host virtual address
  size_t size = 0;      // Slice size in bytes
  // `fd` and `offset` are optional and reserved for future direct/splice I/O:
  // they let a backend hand the kernel a file-backed source/sink instead of a
  // user buffer. No caller populates them today (callers pass fd=-1,
  // offset=0), so this path is unexercised.
  int fd = -1;         // Optional file descriptor (direct/splice I/O)
  int64_t offset = 0;  // Optional byte offset within fd
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
                          absl::Span<const HostBufferDescriptor> slices,
                          size_t total_bytes,
                          std::function<void(absl::Status)> callback) = 0;

  // Asynchronously reads from the target key into a sequence of host DRAM
  // buffer slices. Works uniformly for a single slice (passed as `{slice}`) or
  // multi-slice blocks.
  virtual void ReadAsync(const BlockKey& key,
                         absl::Span<const HostBufferDescriptor> slices,
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

// Serving parallelism coordinates. Grouped so that adding a dimension later
// does not change the layout of every struct that carries topology.
struct ParallelismConfig {
  // Tensor parallelism. Partitions KV heads: H_rank = H_total / tp_size.
  int tp_size = -1;  // -1 = use the mapper's configured default
  int tp_rank = -1;  // -1 = use the mapper's configured default
};

// Options for mapping a cache block hash to a backend key/path.
struct KeyMappingOptions {
  ParallelismConfig parallelism;
};

// BlockKeyMapper defines the coordinator-side locator resolution policy.
// It maps block hash identifiers and rank distributions to backend-specific
// locators and slice boundaries across any offload medium (storage, memory, or
// network).
class BlockKeyMapper {
 public:
  virtual ~BlockKeyMapper() = default;

  // Maps a logical block hash to a physical backend locator.
  //
  // CONTRACT: `block_hash` is opaque, caller-supplied binary data, NOT a
  // printable string. The public API accepts `list[bytes]`, and the Python
  // binding preserves embedded NULs via std::string(ptr, len). Implementations
  // that embed the hash in a path-like locator MUST first encode it into a
  // closed, filesystem-safe alphabet (e.g. absl::BytesToHexString). Passing raw
  // bytes through allows 0x2F to inject a path separator and 0x00 to truncate
  // the locator, silently aliasing two distinct blocks onto one entity.
  //
  // `BlockKey.block_hash` MUST carry the raw, unencoded bytes: encoding is a
  // storage-representation concern and must not leak into block identity.
  //
  // Returns InvalidArgumentError if the hash cannot be mapped (e.g. empty, or
  // too long to encode within the backend's name-length limit).
  virtual absl::StatusOr<BlockKey> MapKey(
      const std::string& block_hash,
      const KeyMappingOptions& options = {}) const = 0;
  virtual int tp_size() const = 0;
};

}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_BACKEND_H_
