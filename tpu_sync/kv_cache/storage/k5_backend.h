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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_STORAGE_K5_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_STORAGE_K5_BACKEND_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/storage/storage.h"

namespace tpu_raiden {
namespace kv_cache {
namespace storage {

class K5BackendMock : public KVBackend {
 public:
  explicit K5BackendMock(std::string uds_socket_path);
  ~K5BackendMock() override = default;

  std::string scheme() const override { return "k5"; }

  void ReadAsync(const BlockKey& key, StorageBufferDescriptor dst_buffer,
                 size_t size,
                 std::function<void(const absl::Status&)> callback) override;

  void WriteAsync(const BlockKey& key, StorageBufferDescriptor src_buffer,
                  size_t size,
                  std::function<void(const absl::Status&)> callback) override;

  absl::StatusOr<bool> Exists(const BlockKey& key) override;

 private:
  absl::Status SendFdAndMetadata(int fd, size_t offset, size_t size,
                                 const std::string& resolved_key,
                                 bool is_write);

  std::string uds_socket_path_;
};

using K5Backend = K5BackendMock;

// K5BlockNameMapper implements the K5 layout mapping policy.
// In production K5 uses logical block/chunk names (keys) instead of files.
// For this mock, we map the logical block name to a file path under the hood
// to simulate persistence in the local sandbox.
class K5BlockNameMapper : public BlockKeyMapper {
 public:
  K5BlockNameMapper(absl::string_view root_dir, absl::string_view model_name,
                    int tp_size, int rank);

  BlockKey MapKey(const std::string& block_hash, int rank) const override;

 private:
  std::string root_dir_;
  std::string model_name_;
  int tp_size_;
  int rank_;
};

}  // namespace storage

using storage::K5Backend;
using storage::K5BackendMock;
using storage::K5BlockNameMapper;

}  // namespace kv_cache

using kv_cache::storage::K5Backend;
using kv_cache::storage::K5BackendMock;
using kv_cache::storage::K5BlockNameMapper;

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_STORAGE_K5_BACKEND_H_
