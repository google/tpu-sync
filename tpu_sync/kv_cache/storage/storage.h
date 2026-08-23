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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_STORAGE_STORAGE_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_STORAGE_STORAGE_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace tpu_raiden {
namespace kv_cache {
namespace storage {

// BlockKey represents a resolved key on persistent storage.
// It bundles:
// - `block_hash`: The content-addressable hash identifying the cache block
// data.
// - `resolved_key`: The target absolute storage path (e.g. Lustre mount
// filepath or GCS URI).
struct BlockKey {
  std::string block_hash;
  std::string resolved_key;  // Absolute resolved file path
};

// StorageBufferDescriptor aggregates virtual memory pointer and physical IPC
// segment descriptors (e.g. shared memory file descriptor and offset) for I/O
// operations.
struct StorageBufferDescriptor {
  uint8_t* ptr = nullptr;  // Raw CPU virtual memory address
  int fd = -1;             // Underlying file descriptor (e.g. for IPC/mmap)
  size_t offset = 0;       // Byte offset within the FD segment
};

// Stateless backend I/O driver interface.
// Production drivers (like LustrePOSIXBackend or GcsObjectBackend) will
// implement this interface.
class KVBackend : public std::enable_shared_from_this<KVBackend> {
 public:
  virtual ~KVBackend() = default;

  // Returns the storage scheme string (e.g. "local_disk", "lustre", "k5").
  virtual std::string scheme() const = 0;

  // Reads `size` bytes asynchronously from the backend key into `dst_buffer`.
  // Invokes `callback` upon completion.
  virtual void ReadAsync(const BlockKey& key,
                         StorageBufferDescriptor dst_buffer, size_t size,
                         std::function<void(const absl::Status&)> callback) = 0;

  // Writes `size` bytes asynchronously from `src_buffer` to the backend key.
  // Invokes `callback` upon completion.
  virtual void WriteAsync(
      const BlockKey& key, StorageBufferDescriptor src_buffer, size_t size,
      std::function<void(const absl::Status&)> callback) = 0;

  // Checks whether the block exists in persistent storage.
  virtual absl::StatusOr<bool> Exists(const BlockKey& key) = 0;
};

// PosixBackend implements KVBackend using standard C++ filesystem APIs.
// Simulates a POSIX mount directory on the worker nodes.
class PosixBackend : public KVBackend {
 public:
  explicit PosixBackend(std::string scheme_name = "local_disk")
      : scheme_(std::move(scheme_name)) {}

  std::string scheme() const override { return scheme_; }

  void ReadAsync(const BlockKey& key, StorageBufferDescriptor dst_buffer,
                 size_t size,
                 std::function<void(const absl::Status&)> callback) override;

  void WriteAsync(const BlockKey& key, StorageBufferDescriptor src_buffer,
                  size_t size,
                  std::function<void(const absl::Status&)> callback) override;

  absl::StatusOr<bool> Exists(const BlockKey& key) override;

 private:
  std::string scheme_ = "local_disk";
};

// BlockKeyMapper defines the coordinator-side path resolution policy.
// It maps block hash identifiers and rank distributions to absolute storage
// locations.
class BlockKeyMapper {
 public:
  virtual ~BlockKeyMapper() = default;

  // Maps the block hash and worker rank to a resolved storage path key.
  virtual BlockKey MapKey(const std::string& block_hash, int rank) const = 0;
  virtual int tp_size() const { return 1; }
};

// PosixPathMapper implements the V4 layout mapping policy.
// Resolves files into flat directories structured by model name, tensor
// parallel size, and rank: Layout:
// `<root_dir>/<model_name>_tp<tp_size>_r<rank>/<block_hash>.bin`
class PosixPathMapper : public BlockKeyMapper {
 public:
  PosixPathMapper(absl::string_view root_dir, absl::string_view model_name,
                  int tp_size, int rank);

  BlockKey MapKey(const std::string& block_hash, int rank) const override;
  int tp_size() const override { return tp_size_; }

 private:
  std::string root_dir_;
  std::string model_name_;
  int tp_size_;
  int rank_;
};

}  // namespace storage

// Namespace aliases for convenience and backward compatibility
using storage::BlockKey;
using storage::BlockKeyMapper;
using storage::KVBackend;
using storage::PosixBackend;
using storage::PosixPathMapper;
using storage::StorageBufferDescriptor;

}  // namespace kv_cache

using kv_cache::storage::BlockKey;
using kv_cache::storage::BlockKeyMapper;
using kv_cache::storage::KVBackend;
using kv_cache::storage::PosixBackend;
using kv_cache::storage::PosixPathMapper;
using kv_cache::storage::StorageBufferDescriptor;

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_STORAGE_STORAGE_H_
