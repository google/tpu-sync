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

#include "tpu_sync/kv_cache/backends/storage/posix_backend.h"

#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/kv_cache/backends/backend.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace storage {

void PosixKVBackend::WriteAsync(const BlockKey& key,
                                absl::Span<const HostBufferDescriptor> slices,
                                size_t total_bytes,
                                std::function<void(absl::Status)> callback) {
  if (callback) {
    callback(absl::UnimplementedError(
        "PosixKVBackend::WriteAsync not implemented in Part 1"));
  }
}

void PosixKVBackend::ReadAsync(const BlockKey& key,
                               absl::Span<const HostBufferDescriptor> slices,
                               size_t total_bytes,
                               std::function<void(absl::Status)> callback) {
  if (callback) {
    callback(absl::UnimplementedError(
        "PosixKVBackend::ReadAsync not implemented in Part 1"));
  }
}

void PosixKVBackend::BatchExistsAsync(
    absl::Span<const BlockKey> keys,
    std::function<void(std::vector<absl::StatusOr<bool>>)> callback) {
  if (callback) {
    std::vector<absl::StatusOr<bool>> results;
    results.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      results.push_back(absl::UnimplementedError(
          "PosixKVBackend::BatchExistsAsync not implemented in Part 1"));
    }
    callback(std::move(results));
  }
}

PosixPathMapper::PosixPathMapper(absl::string_view root_dir,
                                 absl::string_view model_name, int tp_size,
                                 int tp_rank)
    : root_dir_(root_dir),
      model_name_(model_name),
      tp_size_(tp_size),
      tp_rank_(tp_rank) {}

absl::StatusOr<PosixBackendOptions> PosixBackendOptions::FromProperties(
    const absl::flat_hash_map<std::string, std::string>& properties) {
  return absl::UnimplementedError(
      "PosixBackendOptions::FromProperties not implemented in Part 1");
}

absl::string_view PosixPathMapper::GetParentDir(absl::string_view path) {
  size_t last_slash = path.find_last_of('/');
  if (last_slash == absl::string_view::npos) return "";
  return path.substr(0, last_slash);
}

absl::StatusOr<BlockKey> PosixPathMapper::MapKey(
    const std::string& block_hash, const KeyMappingOptions& options) const {
  return absl::UnimplementedError(
      "PosixPathMapper::MapKey not implemented in Part 1");
}

}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden
