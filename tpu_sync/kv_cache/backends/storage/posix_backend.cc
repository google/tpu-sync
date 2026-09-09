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

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace storage {

namespace fs = std::filesystem;

// --- PosixKVBackend Implementation ---

PosixKVBackend::PosixKVBackend(
    std::string name, absl::flat_hash_map<std::string, std::string> properties)
    : KVBackend(std::move(properties)), name_(std::move(name)) {
  std::string root_dir = GetProperty(
      "root_dir", GetProperty("storage_root", "/tmp/raiden_storage"));
  std::string model_name = GetProperty("model_name", "model");
  int tp_size = GetIntProperty("tp_size", 1);
  int rank = GetIntProperty("rank", 0);
  mapper_ =
      std::make_shared<PosixPathMapper>(root_dir, model_name, tp_size, rank);

  int num_threads =
      GetIntProperty("lookup_threads", GetIntProperty("io_threads", 16));
  if (num_threads < 1) num_threads = 1;
  thread_pool_ = std::make_unique<NumaThreadPool>(num_threads);
}

void PosixKVBackend::WriteAsync(
    const BlockKey& key, absl::Span<const BackendBufferDescriptor> slices,
    size_t /*total_bytes*/, std::function<void(absl::Status)> callback) {
  thread_pool_->Schedule(
      std::nullopt, [key,
                     slices = std::vector<BackendBufferDescriptor>(
                         slices.begin(), slices.end()),
                     callback = std::move(callback)]() {
        std::string dir_path =
            std::string(PosixPathMapper::GetParentDir(key.resolved_key));
        if (!dir_path.empty()) {
          std::error_code ec;
          fs::create_directories(dir_path, ec);
          if (ec) {
            if (callback)
              callback(absl::InternalError(
                  absl::StrCat("Failed to create directory: ", dir_path,
                               ", error: ", ec.message())));
            return;
          }
        }

        int flags = O_WRONLY | O_CREAT | (key.offset == 0 ? O_TRUNC : 0);
        int fd = open(key.resolved_key.c_str(), flags, 0644);
        if (fd < 0) {
          if (callback)
            callback(absl::ErrnoToStatus(
                errno, absl::StrCat("Failed to open file for write: ",
                                    key.resolved_key)));
          return;
        }

        off_t current_offset = key.offset;
        for (const auto& slice : slices) {
          if (slice.ptr == nullptr && slice.size > 0) {
            close(fd);
            if (callback)
              callback(absl::InvalidArgumentError(
                  "Null slice pointer in WriteAsync"));
            return;
          }

          size_t bytes_to_write = slice.size;
          const char* buf = static_cast<const char*>(slice.ptr);
          while (bytes_to_write > 0) {
            ssize_t written = pwrite(fd, buf, bytes_to_write, current_offset);
            if (written < 0) {
              if (errno == EINTR) continue;
              int saved_errno = errno;
              close(fd);
              if (callback)
                callback(absl::ErrnoToStatus(
                    saved_errno,
                    absl::StrCat("pwrite failed at offset ", current_offset)));
              return;
            }
            if (written == 0) {
              close(fd);
              if (callback)
                callback(absl::InternalError(absl::StrCat(
                    "pwrite returned 0 bytes at offset ", current_offset)));
              return;
            }
            bytes_to_write -= written;
            buf += written;
            current_offset += written;
          }
        }

        if (close(fd) < 0) {
          if (callback)
            callback(
                absl::ErrnoToStatus(errno, "Failed to close file after write"));
          return;
        }
        if (callback) callback(absl::OkStatus());
      });
}

void PosixKVBackend::ReadAsync(const BlockKey& key,
                               absl::Span<const BackendBufferDescriptor> slices,
                               size_t /*total_bytes*/,
                               std::function<void(absl::Status)> callback) {
  thread_pool_->Schedule(
      std::nullopt, [key,
                     slices = std::vector<BackendBufferDescriptor>(
                         slices.begin(), slices.end()),
                     callback = std::move(callback)]() {
        int fd = open(key.resolved_key.c_str(), O_RDONLY);
        if (fd < 0) {
          if (callback) {
            if (errno == ENOENT) {
              callback(absl::NotFoundError(
                  absl::StrCat("Block file not found: ", key.resolved_key)));
            } else {
              callback(absl::ErrnoToStatus(
                  errno, absl::StrCat("Failed to open file for read: ",
                                      key.resolved_key)));
            }
          }
          return;
        }

        off_t current_offset = key.offset;
        for (const auto& slice : slices) {
          if (slice.ptr == nullptr && slice.size > 0) {
            close(fd);
            if (callback)
              callback(absl::InvalidArgumentError(
                  "Null slice pointer in ReadAsync"));
            return;
          }

          size_t bytes_to_read = slice.size;
          char* buf = static_cast<char*>(slice.ptr);
          while (bytes_to_read > 0) {
            ssize_t bytes_read = pread(fd, buf, bytes_to_read, current_offset);
            if (bytes_read < 0) {
              if (errno == EINTR) continue;
              int saved_errno = errno;
              close(fd);
              if (callback)
                callback(absl::ErrnoToStatus(
                    saved_errno,
                    absl::StrCat("pread failed at offset ", current_offset)));
              return;
            }
            if (bytes_read == 0) {
              close(fd);
              if (callback)
                callback(absl::DataLossError(absl::StrCat(
                    "Unexpected EOF while reading block: ", key.resolved_key)));
              return;
            }
            bytes_to_read -= bytes_read;
            buf += bytes_read;
            current_offset += bytes_read;
          }
        }

        if (close(fd) < 0) {
          if (callback)
            callback(
                absl::ErrnoToStatus(errno, "Failed to close file after read"));
          return;
        }
        if (callback) callback(absl::OkStatus());
      });
}

absl::StatusOr<bool> PosixKVBackend::Exists(const BlockKey& key) {
  std::error_code ec;
  bool exists = fs::exists(key.resolved_key, ec);
  if (ec) {
    return absl::InternalError(absl::StrCat(
        "fs::exists failed for: ", key.resolved_key, ", msg: ", ec.message()));
  }
  return exists;
}

void PosixKVBackend::BatchExistsAsync(
    absl::Span<const BlockKey> keys,
    std::function<void(std::vector<absl::StatusOr<bool>>)> callback) {
  if (keys.empty()) {
    callback({});
    return;
  }

  const size_t num_keys = keys.size();
  struct AsyncBatchState {
    std::vector<absl::StatusOr<bool>> results;
    std::atomic<size_t> remaining;
    std::function<void(std::vector<absl::StatusOr<bool>>)> callback;
  };

  auto state = std::make_shared<AsyncBatchState>();
  state->results.resize(num_keys);
  state->remaining.store(num_keys, std::memory_order_relaxed);
  state->callback = std::move(callback);

  for (size_t i = 0; i < num_keys; ++i) {
    thread_pool_->Schedule(std::nullopt, [this, key = keys[i], i, state]() {
      state->results[i] = Exists(key);
      if (state->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        state->callback(std::move(state->results));
      }
    });
  }
}

// --- PosixPathMapper Implementation ---

PosixPathMapper::PosixPathMapper(absl::string_view root_dir,
                                 absl::string_view model_name, int tp_size,
                                 int rank)
    : root_dir_(root_dir),
      model_name_(model_name),
      tp_size_(tp_size),
      rank_(rank) {}

BlockKey PosixPathMapper::MapKey(const std::string& block_hash,
                                 const KeyMappingOptions& options) const {
  if (block_hash.empty()) {
    LOG(ERROR) << "PosixPathMapper::MapKey: block_hash must not be empty.";
    return BlockKey{};
  }
  int target_rank = (options.rank == -1) ? rank_ : options.rank;
  int target_tp_size = (options.tp_size == -1) ? tp_size_ : options.tp_size;

  std::string l1 = block_hash.size() < 3 ? block_hash : block_hash.substr(0, 3);
  std::string l2 = block_hash.size() >= 5
                       ? block_hash.substr(3, 2)
                       : (block_hash.size() > 3 ? block_hash.substr(3) : "00");

  std::string resolved_path =
      absl::StrCat(root_dir_, "/", model_name_, "_tp", target_tp_size, "_r",
                   target_rank, "/", l1, "/", l2, "/", block_hash, ".bin");
  return BlockKey{block_hash, resolved_path, 0, 0};
}

// --- PosixKVCacheStoreBackend Implementation ---

absl::StatusOr<BlockSliceList> PosixKVCacheStoreBackend::Lookup(
    absl::Span<const std::string> block_hashes, const LookupOptions& options) {
  BlockSliceList results;
  if (!storage_backend_ || !storage_backend_->mapper() || block_hashes.empty())
    return results;

  const size_t total_blocks = block_hashes.size();
  const size_t batch_size =
      lookup_batch_size_ > 0 ? lookup_batch_size_ : kDefaultLookupBatchSize;

  // -------------------------------------------------------------------------
  // Canonical Rank-0 Witness Resolution:
  // In secondary storage, each block is partitioned across all TP workers
  // (r0..rN-1). The existence of the shard at rank 0 confirms the availability
  // of the entire logical block. We explicitly query rank = 0 with the
  // mapper's configured tp_size.
  // -------------------------------------------------------------------------
  backends::KeyMappingOptions lookup_opts{
      .rank = 0,
      .tp_size = storage_backend_->mapper()->tp_size(),
  };

  for (size_t offset = 0; offset < total_blocks; offset += batch_size) {
    size_t current_chunk_len = std::min(batch_size, total_blocks - offset);
    auto chunk_hashes = block_hashes.subspan(offset, current_chunk_len);

    std::vector<BlockKey> chunk_keys;
    chunk_keys.reserve(current_chunk_len);
    for (const auto& hash : chunk_hashes) {
      chunk_keys.push_back(
          storage_backend_->mapper()->MapKey(hash, lookup_opts));
    }

    std::promise<std::vector<absl::StatusOr<bool>>> promise;
    auto future = promise.get_future();
    storage_backend_->BatchExistsAsync(
        chunk_keys, [&promise](std::vector<absl::StatusOr<bool>> res) {
          promise.set_value(std::move(res));
        });
    auto batch_exists = future.get();
    const size_t check_count =
        std::min(batch_exists.size(), chunk_hashes.size());
    bool hit_streak = (batch_exists.size() == chunk_hashes.size());

    for (size_t i = 0; i < check_count; ++i) {
      if (hit_streak && batch_exists[i].ok() && *batch_exists[i]) {
        RaidenBlockId block;
        block.status = BlockStatus::SHARED_STORAGE;
        block.raiden_id.job_replica_id = "shared";
        block.raiden_id.data_name = name_;
        results.push_back(std::make_pair(chunk_hashes[i], block));
      } else {
        hit_streak = false;
        break;
      }
    }

    // Prefix-cache semantics: stop probing subsequent chunks upon first miss.
    if (!hit_streak) {
      break;
    }
  }

  return results;
}

}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden
using ::tpu_raiden::kDefaultLookupBatchSize;
using ::tpu_raiden::PosixKVBackend;
using ::tpu_raiden::PosixKVCacheStoreBackend;
using ::tpu_raiden::PosixPathMapper;

REGISTER_KV_CACHE_STORE_BACKEND(
    "PosixKVCacheStoreBackend",
    [](const ::tpu_raiden::kv_cache::BackendConfig& config,
       ::tpu_raiden::controller::RaidenController* controller)
        -> absl::StatusOr<
            std::shared_ptr<::tpu_raiden::kv_cache::KVCacheStoreBackend>> {
      std::string root_dir = config.GetProperty(
          "root_dir",
          config.GetProperty("storage_root", "/tmp/raiden_storage"));
      std::string model_name = config.GetProperty("model_name", "unknown");
      size_t capacity_bytes =
          static_cast<size_t>(config.GetIntProperty("capacity_bytes", 0));
      size_t lookup_batch_size = static_cast<size_t>(config.GetIntProperty(
          "lookup_batch_size",
          config.GetIntProperty("lookup_chunk_size", kDefaultLookupBatchSize)));

      int tp_size = static_cast<int>(config.GetIntProperty("tp_size", 0));
      if (tp_size <= 0 && controller != nullptr) {
        size_t registered_workers =
            controller->worker_registry()->GetRegisteredWorkers().size();
        if (registered_workers > 0) {
          tp_size = static_cast<int>(registered_workers);
        } else if (controller->num_shards() > 0) {
          tp_size = controller->num_shards();
        }
      }
      if (tp_size <= 0) {
        tp_size = 1;
      }

      std::string backend_name = "PosixKVCacheStoreBackend";
      auto backend =
          std::make_shared<PosixKVBackend>(backend_name, config.properties);
      auto mapper = std::make_shared<PosixPathMapper>(root_dir, model_name,
                                                      tp_size, /*rank=*/0);
      backend->set_mapper(std::move(mapper));
      return std::make_shared<PosixKVCacheStoreBackend>(
          std::move(backend), backend_name, capacity_bytes, lookup_batch_size);
    });
