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

#include "tpu_sync/kv_cache/storage/storage.h"

#include <cstddef>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <functional>
#include <ios>
#include <string>
#include <system_error>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace tpu_raiden {
namespace kv_cache {
namespace storage {

namespace fs = std::filesystem;

// --- PosixBackend Implementation ---

// Asynchronously reads block data from the local file system.
// Fires the callback immediately after completing the synchronous POSIX read.
void PosixBackend::ReadAsync(
    const BlockKey& key, StorageBufferDescriptor dst_buffer, size_t size,
    std::function<void(const absl::Status&)> callback) {
  std::ifstream file(key.resolved_key, std::ios::binary);
  if (!file) {
    callback(absl::NotFoundError(
        absl::StrCat("Failed to open file for reading: ", key.resolved_key)));
    return;
  }

  // Read exact byte size into the host DRAM staging buffer
  file.read(reinterpret_cast<char*>(dst_buffer.ptr), size);
  if (!file) {
    callback(absl::DataLossError(
        absl::StrCat("Failed to read expected ", size,
                     " bytes from file: ", key.resolved_key)));
    return;
  }
  callback(absl::OkStatus());
}

// Asynchronously writes block data to the local file system.
// Creates parent directories if missing (simulating directory creation on
// Lustre).
void PosixBackend::WriteAsync(
    const BlockKey& key, StorageBufferDescriptor src_buffer, size_t size,
    std::function<void(const absl::Status&)> callback) {
  std::error_code ec;
  fs::path filepath(key.resolved_key);

  // Ensure the target directory structure exists
  fs::create_directories(filepath.parent_path(), ec);
  if (ec) {
    callback(absl::InternalError(
        absl::StrCat("Failed to create directories for: ", key.resolved_key,
                     ", error: ", ec.message())));
    return;
  }

  std::ofstream file(key.resolved_key, std::ios::binary);
  if (!file) {
    callback(absl::InternalError(
        absl::StrCat("Failed to open file for writing: ", key.resolved_key)));
    return;
  }

  // Write the byte buffer content to disk
  file.write(reinterpret_cast<const char*>(src_buffer.ptr), size);
  if (!file) {
    callback(absl::DataLossError(absl::StrCat(
        "Failed to write ", size, " bytes to file: ", key.resolved_key)));
    return;
  }
  callback(absl::OkStatus());
}

// Verifies if the file exists on the local filesystem.
absl::StatusOr<bool> PosixBackend::Exists(const BlockKey& key) {
  std::error_code ec;
  bool exists = fs::exists(key.resolved_key, ec);
  if (ec) {
    return absl::InternalError(absl::StrCat(
        "fs::exists failed for: ", key.resolved_key, ", msg: ", ec.message()));
  }
  return exists;
}

// --- PosixPathMapper Implementation ---

PosixPathMapper::PosixPathMapper(absl::string_view root_dir,
                                 absl::string_view model_name, int tp_size,
                                 int rank)
    : root_dir_(root_dir),
      model_name_(model_name),
      tp_size_(tp_size),
      rank_(rank) {}

// Resolves a global content hash to a structured local path.
// This matches the layout schema:
// <root_dir>/<model_name>_tp<tp_size>_r<rank>/<block_hash>.bin
BlockKey PosixPathMapper::MapKey(const std::string& block_hash,
                                 int rank) const {
  // If rank is omitted (-1), resolve the path using the current worker's rank.
  int target_rank = (rank == -1) ? rank_ : rank;
  std::string resolved_path =
      absl::StrCat(root_dir_, "/", model_name_, "_tp", tp_size_, "_r",
                   target_rank, "/", block_hash, ".bin");
  return BlockKey{block_hash, resolved_path};
}

}  // namespace storage
}  // namespace kv_cache
}  // namespace tpu_raiden
