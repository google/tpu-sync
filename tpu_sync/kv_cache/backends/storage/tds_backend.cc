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

#include "tpu_sync/kv_cache/backends/storage/tds_backend.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
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
#include <system_error>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "tdsul/def.h"
#include "tdsul/tdsul.h"
#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/logging.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/backends/storage/storage_backend_utils.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace storage {

namespace fs = std::filesystem;

namespace {

#ifndef UIO_MAXIOV
#define UIO_MAXIOV 1024
#endif

constexpr size_t kTdsDefaultLookupBatchSize = 32;
constexpr size_t kTdsHashDirL1Width = 3;
constexpr size_t kTdsHashDirL2Width = 2;
constexpr size_t kTdsMaxBlockHashBytes = 125;

// Initializes the process-wide `libtdsul` runtime (`tds_init`) once.
void EnsureTdsulInitialized(int num_worker_threads) {
  static absl::once_flag init_once;
  absl::call_once(init_once, [num_worker_threads]() {
    std::unique_ptr<tds_config_t, decltype(&tds_config_destroy)> cfg(
        tds_config_create(), &tds_config_destroy);
    if (cfg == nullptr) {
      LOG(ERROR) << "tds_config_create failed; libtdsul left uninitialized";
      return;
    }
    tds_config_set_int(cfg.get(), "num_worker_threads",
                       std::max(1, num_worker_threads));
    tds_config_set_int(cfg.get(), "enable_io_uring", 0);
    tds_config_set_int(cfg.get(), "enable_p2p", 0);
    const int rc = tds_init(cfg.get());
    if (rc != TDS_SUCCESS) {
      LOG(ERROR) << "tds_init failed (" << rc << ")";
    }
  });
}

size_t GetBufferIoSize(const tds_buffer_io_t& bio) {
  return (bio.handle != nullptr) ? bio.registered.size : bio.raw_ptr.size;
}

void AdvanceBufferIo(tds_buffer_io_t* bio, size_t consumed) {
  if (bio->handle != nullptr) {
    bio->registered.offset += consumed;
    bio->registered.size -= consumed;
  } else {
    bio->raw_ptr.vptr = static_cast<uint8_t*>(bio->raw_ptr.vptr) + consumed;
    bio->raw_ptr.size -= consumed;
  }
}

absl::Status ExecuteTdsIo(int fd, std::vector<tds_buffer_io_t> buffer_ios,
                          int64_t file_offset, bool is_write,
                          size_t min_required_bytes) {
  if (buffer_ios.empty() && min_required_bytes == 0) {
    return absl::OkStatus();
  }

  std::string fd_uri = absl::StrCat("fd://", fd);
  tds_storage_descr_t storage_descr = {
      .uri = fd_uri.c_str(),
      .options = nullptr,
  };
  tds_storage_handle_t* raw_storage_handle = nullptr;
  if (tds_storage_handle_register(&storage_descr, &raw_storage_handle) !=
          TDS_SUCCESS ||
      raw_storage_handle == nullptr) {
    return absl::InternalError(
        absl::StrCat("tds_storage_handle_register failed for URI: ", fd_uri));
  }
  std::unique_ptr<tds_storage_handle_t,
                  decltype(&tds_storage_handle_deregister)>
      storage_handle(raw_storage_handle, &tds_storage_handle_deregister);

  size_t idx = 0;
  size_t total_transferred = 0;
  int64_t current_offset = file_offset;

  while (idx < buffer_ios.size()) {
    if (GetBufferIoSize(buffer_ios[idx]) == 0) {
      ++idx;
      continue;
    }
    const int iov_cnt = std::min<int>(buffer_ios.size() - idx, UIO_MAXIOV);
    size_t batch_bytes = 0;
    for (int i = 0; i < iov_cnt; ++i) {
      batch_bytes += GetBufferIoSize(buffer_ios[idx + i]);
    }

    tds_storage_io_t storage_io =
        tds_create_file_io(storage_handle.get(), current_offset, batch_bytes);

    ssize_t n = -1;
    if (iov_cnt == 1) {
      n = is_write ? tds_write(&storage_io, &buffer_ios[idx])
                   : tds_read(&storage_io, &buffer_ios[idx]);
    } else {
      n = is_write ? tds_writev(&storage_io, &buffer_ios[idx], iov_cnt)
                   : tds_readv(&storage_io, &buffer_ios[idx], iov_cnt);
    }

    if (n < 0) {
      if (errno == EINTR) continue;
      return absl::ErrnoToStatus(
          errno, is_write ? "libtdsul tds_write/tds_writev failed"
                          : "libtdsul tds_read/tds_readv failed");
    }
    if (n == 0) {
      break;
    }

    size_t step = static_cast<size_t>(n);
    total_transferred += step;
    current_offset += n;
    while (step > 0 && idx < buffer_ios.size()) {
      const size_t cur_len = GetBufferIoSize(buffer_ios[idx]);
      if (step >= cur_len) {
        step -= cur_len;
        ++idx;
      } else {
        AdvanceBufferIo(&buffer_ios[idx], step);
        step = 0;
      }
    }
  }

  if (total_transferred < min_required_bytes) {
    return absl::OutOfRangeError(absl::StrCat(
        "Short transfer in libtdsul I/O: required ", min_required_bytes,
        " bytes, transferred ", total_transferred));
  }
  return absl::OkStatus();
}

}  // namespace

// --- TdsBackendOptions Implementation ---

absl::StatusOr<TdsBackendOptions> TdsBackendOptions::FromProperties(
    const absl::flat_hash_map<std::string, std::string>& properties) {
  TdsBackendOptions options;
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
      batch > 0 ? static_cast<size_t>(batch) : kTdsDefaultLookupBatchSize;
  options.storage_io_thread_pool_size = static_cast<int>(threads);

  if (auto it = properties.find("direct_io"); it != properties.end()) {
    std::string val = absl::AsciiStrToLower(it->second);
    if (val == "true") {
      options.direct_io = true;
    } else if (val == "false") {
      options.direct_io = false;
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid boolean value for direct_io: '", it->second,
                       "'; expected 'true' or 'false'"));
    }
  }
  return options;
}

// --- TdsKVBackend Implementation ---

bool TdsKVBackend::ProbeDirectIO(absl::string_view dir) {
  static std::atomic<uint64_t> probe_seq{0};
  std::string probe_path =
      absl::StrCat(dir, "/.tds_o_direct_probe_", getpid(), "_",
                   probe_seq.fetch_add(1, std::memory_order_relaxed));
  std::error_code ec;
  fs::create_directories(std::string(dir), ec);

  int fd =
      open(probe_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
  if (fd < 0) {
    return false;
  }

  const size_t align = GetStorageDirectIOAlignment();
  void* page = nullptr;
  if (posix_memalign(&page, align, align) != 0) {
    close(fd);
    unlink(probe_path.c_str());
    return false;
  }
  std::memset(page, 0, align);
  ssize_t written = write(fd, page, align);
  free(page);
  close(fd);
  unlink(probe_path.c_str());
  return (written == static_cast<ssize_t>(align));
}

TdsKVBackend::TdsKVBackend(
    std::string name, absl::flat_hash_map<std::string, std::string> properties)
    : KVBackend(std::move(properties)), name_(std::move(name)) {
  if (!properties_.contains("tp_rank")) {
    properties_["tp_rank"] = "0";
  }
  absl::StatusOr<TdsBackendOptions> options =
      TdsBackendOptions::FromProperties(properties_);
  if (!options.ok()) {
    LOG(FATAL) << "[TdsKVBackend] invalid configuration for " << name_ << ": "
               << options.status();
  }
  options_ = *std::move(options);
  EnsureTdsulInitialized(options_.storage_io_thread_pool_size);
  if (options_.direct_io) {
    direct_io_supported_ = ProbeDirectIO(options_.root_dir);
    if (!direct_io_supported_) {
      LOG(WARNING) << "[TdsKVBackend] " << name_
                   << ": O_DIRECT not supported on '" << options_.root_dir
                   << "'; falling back to buffered I/O.";
    }
  }
  mapper_ = std::make_shared<TdsPathMapper>(
      options_.root_dir, options_.model_name, options_.tp_size,
      options_.tp_rank);
  thread_pool_ =
      std::make_unique<NumaThreadPool>(options_.storage_io_thread_pool_size);
}

TdsKVBackend::~TdsKVBackend() { thread_pool_.reset(); }

absl::StatusOr<std::vector<tds_buffer_io_t>> TdsKVBackend::BuildBufferIos(
    absl::Span<const HostBufferDescriptor> slices) const {
  std::vector<tds_buffer_io_t> bios;
  bios.reserve(slices.size());
  for (const auto& slice : slices) {
    if (slice.size == 0) continue;
    if (slice.ptr == nullptr) {
      return absl::InvalidArgumentError(
          "Null slice pointer with non-zero size");
    }
    bios.push_back(tds_create_raw_buffer_io(slice.ptr, slice.size));
  }
  return bios;
}

void TdsKVBackend::WriteAsync(const BlockKey& key,
                              absl::Span<const HostBufferDescriptor> slices,
                              size_t total_bytes,
                              std::function<void(absl::Status)> callback) {
  thread_pool_->Schedule(
      std::nullopt,
      [this, key,
       slices = std::vector<HostBufferDescriptor>(slices.begin(), slices.end()),
       total_bytes, callback = std::move(callback)]() {
        auto run = [&]() -> absl::Status {
          if (key.offset != 0) {
            return absl::InvalidArgumentError(absl::StrCat(
                "TdsKVBackend::WriteAsync requires offset 0 (atomic whole-file "
                "publish); got ",
                key.offset));
          }
          size_t sum_bytes = 0;
          for (const auto& s : slices) sum_bytes += s.size;
          if (sum_bytes != total_bytes) {
            return absl::InvalidArgumentError(absl::StrCat(
                "total_bytes (", total_bytes,
                ") does not match sum of slice sizes (", sum_bytes, ")"));
          }
          ABSL_ASSIGN_OR_RETURN(std::vector<tds_buffer_io_t> bios,
                                BuildBufferIos(slices));

          static std::atomic<uint64_t> tmp_seq{0};
          const std::string tmp_path =
              absl::StrCat(key.resolved_key, ".tmp_", ::getpid(), "_",
                           tmp_seq.fetch_add(1, std::memory_order_relaxed));
          const bool use_direct =
              direct_io_supported_ && AreStorageSlicesDirectIOAligned(slices);
          const int open_flags =
              O_WRONLY | O_CREAT | O_TRUNC | (use_direct ? O_DIRECT : 0);

          int fd = ::open(tmp_path.c_str(), open_flags, 0644);
          if (fd < 0 && errno == ENOENT) {
            std::string dir_path(TdsPathMapper::GetParentDir(key.resolved_key));
            if (!dir_path.empty()) {
              std::error_code ec;
              fs::create_directories(dir_path, ec);
              if (ec && !fs::exists(dir_path, ec)) {
                return absl::InternalError(
                    absl::StrCat("Failed to create directory: ", dir_path,
                                 ", error: ", ec.message()));
              }
              fd = ::open(tmp_path.c_str(), open_flags, 0644);
            }
          }
          if (fd < 0) {
            return absl::ErrnoToStatus(
                errno,
                absl::StrCat("Failed to open file for write: ", tmp_path));
          }

          absl::Status io_status =
              ExecuteTdsIo(fd, std::move(bios),
                           /*file_offset=*/0, /*is_write=*/true, total_bytes);
          if (!io_status.ok()) {
            ::close(fd);
            ::unlink(tmp_path.c_str());
            return io_status;
          }
          if (::close(fd) < 0) {
            int saved_errno = errno;
            ::unlink(tmp_path.c_str());
            return absl::ErrnoToStatus(saved_errno,
                                       "Failed to close file after write");
          }
          if (::rename(tmp_path.c_str(), key.resolved_key.c_str()) != 0) {
            int saved_errno = errno;
            ::unlink(tmp_path.c_str());
            return absl::ErrnoToStatus(
                saved_errno,
                absl::StrCat("Failed to publish ", key.resolved_key));
          }
          return absl::OkStatus();
        };
        absl::Status s = run();
        if (callback) callback(std::move(s));
      });
}

void TdsKVBackend::ReadAsync(const BlockKey& key,
                             absl::Span<const HostBufferDescriptor> slices,
                             size_t total_bytes,
                             std::function<void(absl::Status)> callback) {
  thread_pool_->Schedule(
      std::nullopt,
      [this, key,
       slices = std::vector<HostBufferDescriptor>(slices.begin(), slices.end()),
       total_bytes, callback = std::move(callback)]() {
        auto run = [&]() -> absl::Status {
          if (key.offset < 0) {
            return absl::InvalidArgumentError("Negative key offset");
          }
          size_t sum_bytes = 0;
          for (const auto& s : slices) sum_bytes += s.size;
          if (sum_bytes != total_bytes) {
            return absl::InvalidArgumentError(absl::StrCat(
                "total_bytes (", total_bytes,
                ") does not match sum of slice sizes (", sum_bytes, ")"));
          }
          ABSL_ASSIGN_OR_RETURN(std::vector<tds_buffer_io_t> bios,
                                BuildBufferIos(slices));

          const bool use_direct =
              direct_io_supported_ &&
              ((key.offset & (GetStorageDirectIOAlignment() - 1)) == 0) &&
              AreStorageSlicesDirectIOAligned(slices);
          const int open_flags = O_RDONLY | (use_direct ? O_DIRECT : 0);
          int fd = ::open(key.resolved_key.c_str(), open_flags);
          if (fd < 0) {
            if (errno == ENOENT) {
              return absl::NotFoundError(
                  absl::StrCat("Block file not found: ", key.resolved_key));
            }
            return absl::ErrnoToStatus(
                errno, absl::StrCat("Failed to open file for read: ",
                                    key.resolved_key));
          }

          absl::Status io_status =
              ExecuteTdsIo(fd, std::move(bios), key.offset,
                           /*is_write=*/false, total_bytes);
          if (!io_status.ok()) {
            ::close(fd);
            return io_status;
          }
          if (::close(fd) < 0) {
            return absl::ErrnoToStatus(errno,
                                       "Failed to close file after read");
          }
          return absl::OkStatus();
        };
        absl::Status s = run();
        if (callback) callback(std::move(s));
      });
}

absl::StatusOr<bool> TdsKVBackend::Exists(const BlockKey& key) {
  if (::faccessat(AT_FDCWD, key.resolved_key.c_str(), F_OK, AT_EACCESS) == 0) {
    return true;
  }
  if (errno == ENOENT || errno == ENOTDIR) {
    return false;
  }
  return absl::ErrnoToStatus(
      errno, absl::StrCat("faccessat(F_OK) failed for: ", key.resolved_key));
}

void TdsKVBackend::BatchExistsAsync(
    absl::Span<const BlockKey> keys,
    std::function<void(std::vector<absl::StatusOr<bool>>)> callback) {
  if (keys.empty()) {
    thread_pool_->Schedule(std::nullopt, [callback = std::move(callback)]() {
      if (callback) callback({});
    });
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
      if (state->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1 &&
          state->callback) {
        state->callback(std::move(state->results));
      }
    });
  }
}

// --- TdsPathMapper Implementation ---

TdsPathMapper::TdsPathMapper(absl::string_view root_dir,
                             absl::string_view model_name, int tp_size,
                             int tp_rank)
    : root_dir_(root_dir),
      model_name_(SanitizeStorageModelName(model_name)),
      tp_size_(tp_size),
      tp_rank_(tp_rank) {}

absl::StatusOr<BlockKey> TdsPathMapper::MapKey(
    const std::string& block_hash, const KeyMappingOptions& options) const {
  if (block_hash.empty()) {
    return absl::InvalidArgumentError("block_hash must not be empty.");
  }
  if (block_hash.size() > kTdsMaxBlockHashBytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("block_hash is ", block_hash.size(), " bytes; maximum is ",
                     kTdsMaxBlockHashBytes, " (NAME_MAX after hex encoding)."));
  }
  const int target_rank = (options.parallelism.tp_rank == -1)
                              ? tp_rank_
                              : options.parallelism.tp_rank;
  const int target_tp_size = (options.parallelism.tp_size == -1)
                                 ? tp_size_
                                 : options.parallelism.tp_size;

  const std::string hash_hex = absl::BytesToHexString(block_hash);
  const std::string padded = absl::StrCat(
      hash_hex, std::string(kTdsHashDirL1Width + kTdsHashDirL2Width, '0'));
  const absl::string_view l1(padded.data(), kTdsHashDirL1Width);
  const absl::string_view l2(padded.data() + kTdsHashDirL1Width,
                             kTdsHashDirL2Width);

  std::string resolved_path =
      absl::StrCat(root_dir_, "/", model_name_, "/tp", target_tp_size, "_r",
                   target_rank, "/", l1, "/", l2, "/", hash_hex, ".bin");
  return BlockKey{block_hash, resolved_path, /*offset=*/0, /*size=*/0};
}

// --- TdsKVCacheStoreBackend Implementation ---

TdsKVCacheStoreBackend::TdsKVCacheStoreBackend(
    std::shared_ptr<KVBackend> storage_backend, std::string name,
    size_t capacity_bytes, size_t lookup_batch_size)
    : storage_backend_(std::move(storage_backend)),
      name_(std::move(name)),
      capacity_bytes_(capacity_bytes),
      lookup_batch_size_(lookup_batch_size > 0 ? lookup_batch_size
                                               : kTdsDefaultLookupBatchSize) {}

absl::StatusOr<std::vector<BlockKey>> TdsKVCacheStoreBackend::MapShardKeys(
    const std::string& block_hash) const {
  const std::shared_ptr<BlockKeyMapper> mapper = storage_backend_->mapper();
  const int tp_size = mapper->tp_size();
  if (tp_size < 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("invalid mapper tp_size: ", tp_size));
  }
  std::vector<BlockKey> keys;
  keys.reserve(tp_size);
  for (int rank = 0; rank < tp_size; ++rank) {
    const backends::KeyMappingOptions lookup_opts{
        .parallelism = {.tp_size = tp_size, .tp_rank = rank},
    };
    ABSL_ASSIGN_OR_RETURN(BlockKey key,
                          mapper->MapKey(block_hash, lookup_opts));
    keys.push_back(std::move(key));
  }
  return keys;
}

absl::StatusOr<BlockSliceList> TdsKVCacheStoreBackend::Lookup(
    absl::Span<const std::string> block_hashes, const LookupOptions& options) {
  BlockSliceList results;
  if (!storage_backend_ || !storage_backend_->mapper() || block_hashes.empty())
    return results;

  for (const std::string& hash : block_hashes) {
    absl::StatusOr<std::vector<BlockKey>> shard_keys = MapShardKeys(hash);
    if (!shard_keys.ok()) break;

    std::promise<std::vector<absl::StatusOr<bool>>> promise;
    auto future = promise.get_future();
    storage_backend_->BatchExistsAsync(
        *shard_keys, [&promise](std::vector<absl::StatusOr<bool>> res) {
          promise.set_value(std::move(res));
        });
    const std::vector<absl::StatusOr<bool>> answers = future.get();
    if (answers.size() != shard_keys->size()) break;
    bool all_exist = true;
    for (const auto& ans : answers) {
      if (!ans.ok() || !*ans) {
        all_exist = false;
        break;
      }
    }
    if (!all_exist) break;

    RaidenBlockId block;
    block.status = BlockStatus::SHARED_STORAGE;
    block.raiden_id.job_replica_id = "shared";
    block.raiden_id.data_name = name_;
    results.push_back(std::make_pair(hash, block));
  }

  return results;
}

}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

using ::tpu_raiden::kv_cache::backends::storage::TdsBackendOptions;
using ::tpu_raiden::kv_cache::backends::storage::TdsKVBackend;
using ::tpu_raiden::kv_cache::backends::storage::TdsKVCacheStoreBackend;

REGISTER_KV_CACHE_STORE_BACKEND(
    ::tpu_raiden::kv_cache::backends::storage::kTdsBackendName,
    [](const ::tpu_raiden::kv_cache::BackendConfig& config,
       ::tpu_raiden::controller::RaidenController* /*controller*/)
        -> absl::StatusOr<
            std::shared_ptr<::tpu_raiden::kv_cache::KVCacheStoreBackend>> {
      const int tp_size =
          config.parallelism.tp_size > 0 ? config.parallelism.tp_size : 1;

      ::tpu_raiden::kv_cache::BackendConfig resolved = config;
      ::tpu_raiden::kv_cache::ApplyParallelismToProperties(
          {.tp_size = tp_size, .tp_rank = 0}, &resolved);
      ABSL_ASSIGN_OR_RETURN(
          const TdsBackendOptions options,
          TdsBackendOptions::FromProperties(resolved.properties));

      const std::string backend_name = std::string(
          ::tpu_raiden::kv_cache::backends::storage::kTdsBackendName);
      auto backend =
          std::make_shared<TdsKVBackend>(backend_name, resolved.properties);
      return std::make_shared<TdsKVCacheStoreBackend>(
          std::move(backend), backend_name, options.capacity_bytes,
          options.lookup_batch_size);
    });
