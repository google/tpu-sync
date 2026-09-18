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
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
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
  absl::StatusOr<PosixBackendOptions> options =
      PosixBackendOptions::FromProperties(properties_);
  if (!options.ok()) {
    LOG(FATAL) << "[PosixKVBackend] invalid configuration for " << name_ << ": "
               << options.status();
  }
  options_ = *std::move(options);
  mapper_ =
      std::make_shared<PosixPathMapper>(options_.root_dir, options_.model_name,
                                        options_.tp_size, options_.tp_rank);
  thread_pool_ =
      std::make_unique<NumaThreadPool>(options_.storage_io_thread_pool_size);
}

void PosixKVBackend::WriteAsync(const BlockKey& key,
                                absl::Span<const HostBufferDescriptor> slices,
                                size_t /*total_bytes*/,
                                std::function<void(absl::Status)> callback) {
  thread_pool_->Schedule(std::nullopt, [key,
                                        slices =
                                            std::vector<HostBufferDescriptor>(
                                                slices.begin(), slices.end()),
                                        callback = std::move(callback)]() {
    // A non-zero offset means this write is one slice of a file several
    // writers share, and a whole-file rename would publish a file
    // containing only this slice. No mapper produces such a key today --
    // PosixPathMapper::MapKey hardcodes offset=0 -- so reject it rather
    // than silently corrupt the file if a future mapper starts
    // partitioning one file across ranks.
    if (key.offset != 0) {
      if (callback)
        callback(absl::InvalidArgumentError(
            absl::StrCat("PosixKVBackend::WriteAsync requires offset 0 (atomic "
                         "whole-file publish); got ",
                         key.offset)));
      return;
    }

    // Write to a temp file and rename into place, so a concurrent Lookup
    // -- which gates on bare existence -- sees either no file or a
    // complete one. Without this, any error below leaves a short file at
    // the final path, which is a permanent phantom hit that fails every
    // future recall with DataLossError.
    //
    // The suffix keeps the temp file in the SAME directory as the final
    // path, because rename(2) is only atomic within a filesystem. The
    // pid distinguishes processes; the counter distinguishes threads and
    // successive writes within one process, which is stricter than a
    // thread id (those get recycled once a thread exits).
    static std::atomic<uint64_t> tmp_seq{0};
    const std::string tmp_path =
        absl::StrCat(key.resolved_key, ".tmp_", getpid(), "_",
                     tmp_seq.fetch_add(1, std::memory_order_relaxed));

    int fd = open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0 && errno == ENOENT) {
      // Optimistic open failed because parent directory does not exist yet.
      // Create directory hierarchy and retry open once.
      std::string dir_path =
          std::string(PosixPathMapper::GetParentDir(key.resolved_key));
      if (!dir_path.empty()) {
        std::error_code ec;
        fs::create_directories(dir_path, ec);
        if (ec) {
          if (callback) {
            callback(absl::InternalError(
                absl::StrCat("Failed to create directory: ", dir_path,
                             ", error: ", ec.message())));
          }
          return;
        }
        fd = open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      }
    }

    if (fd < 0) {
      if (callback) {
        callback(absl::ErrnoToStatus(
            errno, absl::StrCat("Failed to open file for write: ", tmp_path)));
      }
      return;
    }

    // Every failure from here on must remove the temp file: nothing else
    // ever will, since this backend's Delete is a no-op stub.
    auto fail = [&tmp_path, &callback](int fd_to_close, absl::Status s) {
      close(fd_to_close);
      unlink(tmp_path.c_str());
      if (callback) callback(std::move(s));
    };

    // Populate iovec array for scatter-gather write.
    std::vector<struct iovec> iov;
    iov.reserve(slices.size());
    size_t total_slices_bytes = 0;
    for (const auto& slice : slices) {
      if (slice.ptr == nullptr && slice.size > 0) {
        fail(fd,
             absl::InvalidArgumentError("Null slice pointer in WriteAsync"));
        return;
      }
      if (slice.size > 0) {
        iov.push_back(iovec{.iov_base = const_cast<void*>(slice.ptr),
                            .iov_len = slice.size});
        total_slices_bytes += slice.size;
      }
    }

    off_t current_offset = 0;
    size_t iov_idx = 0;
    while (iov_idx < iov.size()) {
      // Chunk batch to stay within UIO_MAXIOV (1024) limit.
      int batch_count = std::min<int>(iov.size() - iov_idx, UIO_MAXIOV);
      size_t batch_bytes = 0;
      for (int i = 0; i < batch_count; ++i) {
        batch_bytes += iov[iov_idx + i].iov_len;
      }

      ssize_t written = pwritev(fd, &iov[iov_idx], batch_count, current_offset);
      if (written < 0) {
        if (errno == EINTR) continue;
        int saved_errno = errno;
        fail(fd, absl::ErrnoToStatus(saved_errno,
                                     absl::StrCat("pwritev failed at offset ",
                                                  current_offset)));
        return;
      }
      if (written == 0) {
        fail(fd, absl::InternalError(absl::StrCat(
                     "pwritev returned 0 bytes at offset ", current_offset)));
        return;
      }
      current_offset += written;

      // Fast Path: Entire batch was written (standard regular-file
      // behavior). Advance iov_idx directly by batch_count.
      if (static_cast<size_t>(written) == batch_bytes) {
        iov_idx += batch_count;
        continue;
      }

      // Slow/Rare Path: Partial write across the batch. Drain the partial
      // bytes:
      size_t bytes_left = static_cast<size_t>(written);
      while (bytes_left > 0 && iov_idx < iov.size()) {
        if (bytes_left >= iov[iov_idx].iov_len) {
          bytes_left -= iov[iov_idx].iov_len;
          ++iov_idx;
        } else {
          iov[iov_idx].iov_base =
              static_cast<char*>(iov[iov_idx].iov_base) + bytes_left;
          iov[iov_idx].iov_len -= bytes_left;
          bytes_left = 0;
        }
      }
    }

    if (close(fd) < 0) {
      int saved_errno = errno;
      unlink(tmp_path.c_str());
      if (callback)
        callback(absl::ErrnoToStatus(saved_errno,
                                     "Failed to close file after write"));
      return;
    }

    if (rename(tmp_path.c_str(), key.resolved_key.c_str()) != 0) {
      int saved_errno = errno;
      unlink(tmp_path.c_str());
      if (callback)
        callback(absl::ErrnoToStatus(
            saved_errno, absl::StrCat("Failed to publish ", key.resolved_key)));
      return;
    }
    if (callback) callback(absl::OkStatus());
  });
}

void PosixKVBackend::ReadAsync(const BlockKey& key,
                               absl::Span<const HostBufferDescriptor> slices,
                               size_t /*total_bytes*/,
                               std::function<void(absl::Status)> callback) {
  thread_pool_->Schedule(std::nullopt, [key,
                                        slices =
                                            std::vector<HostBufferDescriptor>(
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

    // Populate iovec array for scatter-gather read.
    std::vector<struct iovec> iov;
    iov.reserve(slices.size());
    for (const auto& slice : slices) {
      if (slice.ptr == nullptr && slice.size > 0) {
        close(fd);
        if (callback) {
          callback(
              absl::InvalidArgumentError("Null slice pointer in ReadAsync"));
        }
        return;
      }
      if (slice.size > 0) {
        iov.push_back(iovec{.iov_base = slice.ptr, .iov_len = slice.size});
      }
    }

    off_t current_offset = key.offset;
    size_t iov_idx = 0;
    while (iov_idx < iov.size()) {
      int batch_count = std::min<int>(iov.size() - iov_idx, UIO_MAXIOV);
      size_t batch_bytes = 0;
      for (int i = 0; i < batch_count; ++i) {
        batch_bytes += iov[iov_idx + i].iov_len;
      }

      ssize_t bytes_read =
          preadv(fd, &iov[iov_idx], batch_count, current_offset);
      if (bytes_read < 0) {
        if (errno == EINTR) continue;
        int saved_errno = errno;
        close(fd);
        if (callback) {
          callback(absl::ErrnoToStatus(
              saved_errno,
              absl::StrCat("preadv failed at offset ", current_offset)));
        }
        return;
      }
      if (bytes_read == 0) {
        close(fd);
        if (callback) {
          callback(absl::DataLossError(absl::StrCat(
              "Unexpected EOF while reading block: ", key.resolved_key)));
        }
        return;
      }
      current_offset += bytes_read;

      // Fast Path: Entire batch was read. Advance iov_idx directly by
      // batch_count.
      if (static_cast<size_t>(bytes_read) == batch_bytes) {
        iov_idx += batch_count;
        continue;
      }

      // Slow/Rare Path: Partial read. Drain the partial bytes:
      size_t bytes_left = static_cast<size_t>(bytes_read);
      while (bytes_left > 0 && iov_idx < iov.size()) {
        if (bytes_left >= iov[iov_idx].iov_len) {
          bytes_left -= iov[iov_idx].iov_len;
          ++iov_idx;
        } else {
          iov[iov_idx].iov_base =
              static_cast<char*>(iov[iov_idx].iov_base) + bytes_left;
          iov[iov_idx].iov_len -= bytes_left;
          bytes_left = 0;
        }
      }
    }

    if (close(fd) < 0) {
      if (callback)
        callback(absl::ErrnoToStatus(errno, "Failed to close file after read"));
      return;
    }
    if (callback) callback(absl::OkStatus());
  });
}

absl::StatusOr<bool> PosixKVBackend::Exists(const BlockKey& key) {
  if (faccessat(AT_FDCWD, key.resolved_key.c_str(), F_OK, AT_EACCESS) == 0) {
    return true;
  }
  if (errno == ENOENT || errno == ENOTDIR) {
    return false;
  }
  return absl::ErrnoToStatus(
      errno, absl::StrCat("faccessat(F_OK) failed for: ", key.resolved_key));
}

void PosixKVBackend::BatchExistsAsync(
    absl::Span<const BlockKey> keys,
    std::function<void(std::vector<absl::StatusOr<bool>>)> callback) {
  if (keys.empty()) {
    // Completed on the pool, not inline, so this call reports asynchronously
    // for every input. A caller that assumes the callback cannot run before
    // BatchExistsAsync returns -- true for every non-empty batch -- would
    // otherwise be re-entered on this one path, which deadlocks it if it holds
    // a lock across the call.
    thread_pool_->Schedule(
        std::nullopt, [callback = std::move(callback)]() { callback({}); });
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

namespace {

// HuggingFace-style model IDs (e.g. "meta-llama/Llama-3.1-70B") embed path
// separators, which would inject an extra directory level and break the
// fixed-depth layout. Anything outside [A-Za-z0-9._-] is replaced by '_'.
std::string SanitizeModelName(absl::string_view model_name) {
  std::string sanitized;
  sanitized.reserve(model_name.size());
  for (const char c : model_name) {
    const bool is_safe = absl::ascii_isalnum(static_cast<unsigned char>(c)) ||
                         c == '.' || c == '_' || c == '-';
    sanitized.push_back(is_safe ? c : '_');
  }
  if (sanitized.empty()) return "unknown";
  return sanitized;
}

}  // namespace

PosixPathMapper::PosixPathMapper(absl::string_view root_dir,
                                 absl::string_view model_name, int tp_size,
                                 int tp_rank)
    : root_dir_(root_dir),
      model_name_(SanitizeModelName(model_name)),
      tp_size_(tp_size),
      tp_rank_(tp_rank) {}

absl::StatusOr<PosixBackendOptions> PosixBackendOptions::FromProperties(
    const absl::flat_hash_map<std::string, std::string>& properties) {
  PosixBackendOptions options;
  auto str = [&](absl::string_view key, std::string* out) {
    auto it = properties.find(key);
    if (it != properties.end()) *out = it->second;
  };
  auto num = [&](absl::string_view key, int64_t* out) -> absl::Status {
    auto it = properties.find(key);
    if (it == properties.end()) return absl::OkStatus();
    if (!absl::SimpleAtoi(it->second, out)) {
      return absl::InvalidArgumentError(
          absl::StrCat(key, " is not an integer: ", it->second));
    }
    return absl::OkStatus();
  };
  str("root_dir", &options.root_dir);
  str("model_name", &options.model_name);
  int64_t tp_size = options.tp_size;
  int64_t tp_rank = options.tp_rank;
  int64_t capacity = 0;
  int64_t batch = options.lookup_batch_size;
  int64_t threads = options.storage_io_thread_pool_size;
  ABSL_RETURN_IF_ERROR(num("tp_size", &tp_size));
  ABSL_RETURN_IF_ERROR(num("tp_rank", &tp_rank));
  ABSL_RETURN_IF_ERROR(num("capacity_bytes", &capacity));
  ABSL_RETURN_IF_ERROR(num("lookup_batch_size", &batch));
  ABSL_RETURN_IF_ERROR(num("storage_io_thread_pool_size", &threads));
  if (tp_size < 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("tp_size must be >= 1, got ", tp_size));
  }
  if (tp_rank < 0 || tp_rank >= tp_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("tp_rank must be in [0, ", tp_size, "), got ", tp_rank));
  }
  if (threads < 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "storage_io_thread_pool_size must be >= 1, got ", threads));
  }
  options.tp_size = static_cast<int>(tp_size);
  options.tp_rank = static_cast<int>(tp_rank);
  options.capacity_bytes = static_cast<size_t>(capacity);
  options.lookup_batch_size =
      batch > 0 ? static_cast<size_t>(batch) : kDefaultLookupBatchSize;
  options.storage_io_thread_pool_size = static_cast<int>(threads);
  return options;
}

absl::StatusOr<BlockKey> PosixPathMapper::MapKey(
    const std::string& block_hash, const KeyMappingOptions& options) const {
  if (block_hash.empty()) {
    return absl::InvalidArgumentError("block_hash must not be empty.");
  }
  if (block_hash.size() > kMaxBlockHashBytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("block_hash is ", block_hash.size(), " bytes; maximum is ",
                     kMaxBlockHashBytes, " (NAME_MAX after hex encoding)."));
  }
  const int target_rank = (options.parallelism.tp_rank == -1)
                              ? tp_rank_
                              : options.parallelism.tp_rank;
  const int target_tp_size = (options.parallelism.tp_size == -1)
                                 ? tp_size_
                                 : options.parallelism.tp_size;

  // block_hash is opaque binary; encode before it contributes to a path.
  const std::string hash_hex = absl::BytesToHexString(block_hash);
  // Right-pad so the prefix directories are fixed-width for every input
  // length. The filename uses the complete unpadded hash, so short hashes
  // never collide.
  const std::string padded = absl::StrCat(
      hash_hex, std::string(kHashDirL1Width + kHashDirL2Width, '0'));
  const absl::string_view l1(padded.data(), kHashDirL1Width);
  const absl::string_view l2(padded.data() + kHashDirL1Width, kHashDirL2Width);

  std::string resolved_path =
      absl::StrCat(root_dir_, "/", model_name_, "/tp", target_tp_size, "_r",
                   target_rank, "/", l1, "/", l2, "/", hash_hex, ".bin");
  // Block identity stays the raw bytes; only the path is hex-encoded.
  return BlockKey{block_hash, resolved_path, /*offset=*/0, /*size=*/0};
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
  //
  // TODO: Revisit this logic to ensure a consistent lookup based on all
  // shards availability in the storage layer.
  // -------------------------------------------------------------------------
  backends::KeyMappingOptions lookup_opts{
      .parallelism = {.tp_size = storage_backend_->mapper()->tp_size(),
                      .tp_rank = 0},
  };

  for (size_t offset = 0; offset < total_blocks; offset += batch_size) {
    size_t current_chunk_len = std::min(batch_size, total_blocks - offset);
    auto chunk_hashes = block_hashes.subspan(offset, current_chunk_len);

    std::vector<BlockKey> chunk_keys;
    chunk_keys.reserve(current_chunk_len);
    // batch_exists[i] is indexed against chunk_hashes[i] below, so skipping an
    // unmappable key would misattribute existence results to the wrong block.
    // Truncate the probe instead: under-reporting hits is always safe.
    bool mapping_failed = false;
    for (const auto& hash : chunk_hashes) {
      absl::StatusOr<BlockKey> key =
          storage_backend_->mapper()->MapKey(hash, lookup_opts);
      if (!key.ok()) {
        mapping_failed = true;
        break;
      }
      chunk_keys.push_back(*std::move(key));
    }
    if (mapping_failed) break;

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
// These three names are consumed by REGISTER_KV_CACHE_STORE_BACKEND below.
// They are spelled out in full because the header no longer re-exports them
// into the project root namespace.
using ::tpu_raiden::kv_cache::backends::storage::PosixBackendOptions;
using ::tpu_raiden::kv_cache::backends::storage::PosixKVBackend;
using ::tpu_raiden::kv_cache::backends::storage::PosixKVCacheStoreBackend;

REGISTER_KV_CACHE_STORE_BACKEND(
    ::tpu_raiden::kv_cache::backends::storage::kPosixBackendName,
    [](const ::tpu_raiden::kv_cache::BackendConfig& config,
       ::tpu_raiden::controller::RaidenController* controller)
        -> absl::StatusOr<
            std::shared_ptr<::tpu_raiden::kv_cache::KVCacheStoreBackend>> {
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

      // The coordinator resolves block existence through the rank-0 witness
      // (see PosixKVCacheStoreBackend::Lookup), so its mapper is pinned to
      // rank 0. Per-worker tp_rank lives on the worker's own config.
      auto properties = config.properties;
      properties["tp_size"] = absl::StrCat(tp_size);
      properties["tp_rank"] = "0";

      const std::string backend_name = std::string(
          ::tpu_raiden::kv_cache::backends::storage::kPosixBackendName);
      auto backend = std::make_shared<PosixKVBackend>(backend_name, properties);
      const PosixBackendOptions& options = backend->options();
      return std::make_shared<PosixKVCacheStoreBackend>(
          std::move(backend), backend_name, options.capacity_bytes,
          options.lookup_batch_size);
    });
