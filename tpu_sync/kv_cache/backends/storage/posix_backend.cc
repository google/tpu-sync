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
#include <limits.h>
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
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
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

// --- Direct I/O alignment ---

size_t DirectIOAlignment() {
  static const size_t kAlign = [] {
    const long page = sysconf(_SC_PAGESIZE);
    CHECK_GT(page, 0) << "sysconf(_SC_PAGESIZE) failed";
    CHECK_EQ(page & (page - 1), 0)
        << "page size is not a power of two: " << page;
    LOG(INFO) << "PosixKVBackend: O_DIRECT alignment = " << page
              << " bytes (sysconf(_SC_PAGESIZE))";
    return static_cast<size_t>(page);
  }();
  return kAlign;
}

bool SlicesAreDirectIOAligned(absl::Span<const HostBufferDescriptor> slices) {
  // The alignment is a power of two, so OR-ing address and size lets one mask
  // test cover both.
  const uintptr_t mask = DirectIOAlignment() - 1;
  for (const auto& slice : slices) {
    if (((reinterpret_cast<uintptr_t>(slice.ptr) | slice.size) & mask) != 0) {
      return false;
    }
  }
  return true;
}

// --- Temporary file naming ---

namespace {

// Drawn once per process from OS entropy. A forked child inherits it, which
// is why callers also mix in getpid().
uint64_t ProcessTempNonce() {
  static const uint64_t nonce = [] {
    absl::BitGen gen;
    return absl::Uniform<uint64_t>(gen);
  }();
  return nonce;
}

// "<prefix><pid>_<nonce>_<seq>", unique across processes and hosts. The
// counter is stricter than a thread id (those get recycled once a thread
// exits) and keeps successive calls from one thread apart.
std::string UniqueSuffix(absl::string_view prefix) {
  static std::atomic<uint64_t> seq{0};
  return absl::StrCat(prefix, getpid(), "_",
                      absl::Hex(ProcessTempNonce(), absl::kZeroPad16), "_",
                      seq.fetch_add(1, std::memory_order_relaxed));
}

}  // namespace

std::string MakeTempPath(absl::string_view final_path) {
  // Same directory as the final path: rename(2) is only atomic within a
  // filesystem, and staying in the leaf directory also keeps the temp file
  // out of any other directory's listings.
  const absl::string_view dir = PosixPathMapper::GetParentDir(final_path);
  return absl::StrCat(dir.empty() ? absl::string_view(".") : dir, "/",
                      UniqueSuffix(".tmp_"));
}

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
  if (options_.direct_io) {
    direct_io_supported_ = ProbeDirectIO(options_.root_dir);
    if (!direct_io_supported_) {
      LOG(FATAL) << "[PosixKVBackend] " << name_
                 << ": direct_io=true but O_DIRECT is not supported on '"
                 << options_.root_dir
                 << "'. Unset direct_io or use a filesystem that supports it.";
    }
  }
  mapper_ = std::make_shared<PosixPathMapper>(
      options_.root_dir, options_.model_name,
      ParallelismConfig{.tp_size = options_.tp_size,
                        .tp_rank = options_.tp_rank,
                        .pcp_size = options_.pcp_size,
                        .pcp_rank = options_.pcp_rank,
                        .pp_size = options_.pp_size,
                        .pp_rank = options_.pp_rank});
  thread_pool_ =
      std::make_unique<NumaThreadPool>(options_.storage_io_thread_pool_size);
}

bool PosixKVBackend::ProbeDirectIO(absl::string_view dir) {
  // Same uniqueness scheme as write temp files: several hosts may probe one
  // shared directory at the same time.
  std::string probe_path =
      absl::StrCat(dir, "/", UniqueSuffix(".o_direct_probe_"));
  std::error_code ec;
  fs::create_directories(std::string(dir), ec);

  int fd = open(probe_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_DIRECT, 0644);
  if (fd < 0) {
    LOG(WARNING) << "PosixKVBackend: O_DIRECT probe open failed on '" << dir
                 << "' (errno: " << errno << " - " << strerror(errno) << ")";
    return false;
  }

  // Write one page, aligned to the page size, to verify whether the underlying
  // filesystem accepts O_DIRECT I/O without failing with EINVAL (as happens on
  // overlayfs or older tmpfs).
  const size_t align = DirectIOAlignment();
  void* page = nullptr;
  if (posix_memalign(&page, align, align) != 0) {
    close(fd);
    unlink(probe_path.c_str());
    LOG(WARNING) << "PosixKVBackend: O_DIRECT probe posix_memalign failed on '"
                 << dir << "'";
    return false;
  }
  std::memset(page, 0, align);
  ssize_t written = write(fd, page, align);
  const int write_errno = errno;
  free(page);
  close(fd);
  unlink(probe_path.c_str());

  bool success = (written == static_cast<ssize_t>(align));
  if (success) {
    LOG(INFO) << "PosixKVBackend: O_DIRECT probe on '" << dir
              << "' succeeded with " << align
              << "-byte alignment; direct I/O is supported.";
  } else {
    LOG(WARNING) << "PosixKVBackend: O_DIRECT probe write failed on '" << dir
                 << "' (written: " << written << ", errno: " << write_errno
                 << " - " << strerror(write_errno) << ")";
  }
  return success;
}

void PosixKVBackend::WriteAsync(const BlockKey& key,
                                absl::Span<const HostBufferDescriptor> slices,
                                size_t /*total_bytes*/,
                                std::function<void(absl::Status)> callback) {
  thread_pool_->Schedule(std::nullopt, [direct_io_supported = direct_io_supported_,
                                        key,
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
    // The temp name (see MakeTempPath) stays in the final file's directory
    // and is unique across hosts: engines on different hosts sharing one
    // storage directory can have equal pids and counters, and with a
    // pid+counter name they would open the same temp file, interleave their
    // writes, and fail or corrupt the publish. It does not embed the hash,
    // so its length is independent of the key.
    const std::string tmp_path = MakeTempPath(key.resolved_key);

    // With direct_io there is no buffered fallback: a misaligned buffer fails
    // the operation before any file is created.
    if (direct_io_supported && !SlicesAreDirectIOAligned(slices)) {
      if (callback)
        callback(absl::InvalidArgumentError(absl::StrCat(
            "PosixKVBackend::WriteAsync: direct_io=true but a slice pointer "
            "or size is not aligned to ",
            DirectIOAlignment(), " bytes for O_DIRECT")));
      return;
    }
    // O_EXCL: the temp name is unique by construction, so an existing file
    // there is another writer's. Fail rather than share its inode.
    const int open_flags =
        O_WRONLY | O_CREAT | O_EXCL | (direct_io_supported ? O_DIRECT : 0);

    int fd = open(tmp_path.c_str(), open_flags, 0644);
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
        fd = open(tmp_path.c_str(), open_flags, 0644);
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
  thread_pool_->Schedule(std::nullopt, [direct_io_supported = direct_io_supported_,
                                        key,
                                        slices =
                                            std::vector<HostBufferDescriptor>(
                                                slices.begin(), slices.end()),
                                        callback = std::move(callback)]() {
    // With direct_io there is no buffered fallback: a misaligned file offset
    // or buffer fails the operation before the file is opened.
    if (direct_io_supported &&
        ((key.offset & (DirectIOAlignment() - 1)) != 0 ||
         !SlicesAreDirectIOAligned(slices))) {
      if (callback)
        callback(absl::InvalidArgumentError(absl::StrCat(
            "PosixKVBackend::ReadAsync: direct_io=true but the file offset, a "
            "slice pointer or a slice size is not aligned to ",
            DirectIOAlignment(), " bytes for O_DIRECT")));
      return;
    }
    const int open_flags = O_RDONLY | (direct_io_supported ? O_DIRECT : 0);

    int fd = open(key.resolved_key.c_str(), open_flags);
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

namespace {

// Joins with '_' one `<axis><size>_r<rank label>` segment per declared axis,
// outermost first in the fixed order pp, pcp, tp (vLLM's rank layout).
// `rank_label` renders the rank part of an axis given its size and rank.
template <typename RankLabel>
std::string JoinAxisSegments(const ParallelismConfig& parallelism,
                             RankLabel rank_label) {
  std::string topology_dir;
  auto append_axis = [&](absl::string_view axis, int size, int rank) {
    if (size == kAxisUndeclared) return;
    absl::StrAppend(&topology_dir, topology_dir.empty() ? "" : "_", axis, size,
                    "_r", rank_label(size, rank));
  };
  append_axis("pp", parallelism.pp_size, parallelism.pp_rank);
  append_axis("pcp", parallelism.pcp_size, parallelism.pcp_rank);
  append_axis("tp", parallelism.tp_size, parallelism.tp_rank);
  return topology_dir;
}

// The topology directories the coordinator probes for `parallelism`, with
// each declared axis's rank written as the range it covers, e.g.
// `pp2_r{0..1}_tp4_r{0..3}`; a size-1 axis has the single rank 0.
std::string FormatProbedTopologyDirPattern(
    const ParallelismConfig& parallelism) {
  return JoinAxisSegments(parallelism, [](int size, int /*rank*/) {
    return size == 1 ? std::string("0") : absl::StrCat("{0..", size - 1, "}");
  });
}

}  // namespace

std::string FormatTopologyDir(const ParallelismConfig& parallelism) {
  return JoinAxisSegments(parallelism,
                          [](int /*size*/, int rank) { return rank; });
}

PosixPathMapper::PosixPathMapper(absl::string_view root_dir,
                                 absl::string_view model_name,
                                 const ParallelismConfig& parallelism)
    : root_dir_(root_dir),
      model_name_(SanitizeModelName(model_name)),
      parallelism_(parallelism) {}

absl::StatusOr<PosixBackendOptions> PosixBackendOptions::FromProperties(
    const absl::flat_hash_map<std::string, std::string>& properties) {
  PosixBackendOptions options;
  auto str = [&](absl::string_view key, std::string* out) {
    auto it = properties.find(key);
    if (it != properties.end()) *out = it->second;
  };
  auto num = [&](absl::string_view key, auto* out) -> absl::Status {
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
  // The sanitized model name is one directory component. Sanitizing only
  // replaces characters, so an over-long name would otherwise fail every
  // write with ENAMETOOLONG once the directory is created.
  if (const size_t len = SanitizeModelName(options.model_name).size();
      len > NAME_MAX) {
    return absl::InvalidArgumentError(
        absl::StrCat("model_name sanitizes to ", len,
                     " characters; the limit is ", NAME_MAX, " (NAME_MAX)"));
  }
  ParallelismConfig parallelism;
  int64_t capacity = 0;
  int64_t batch = options.lookup_batch_size;
  int64_t threads = options.storage_io_thread_pool_size;
  ABSL_RETURN_IF_ERROR(num("tp_size", &parallelism.tp_size));
  ABSL_RETURN_IF_ERROR(num("tp_rank", &parallelism.tp_rank));
  ABSL_RETURN_IF_ERROR(num("pcp_size", &parallelism.pcp_size));
  ABSL_RETURN_IF_ERROR(num("pcp_rank", &parallelism.pcp_rank));
  ABSL_RETURN_IF_ERROR(num("pp_size", &parallelism.pp_size));
  ABSL_RETURN_IF_ERROR(num("pp_rank", &parallelism.pp_rank));
  ABSL_RETURN_IF_ERROR(num("capacity_bytes", &capacity));
  ABSL_RETURN_IF_ERROR(num("lookup_batch_size", &batch));
  ABSL_RETURN_IF_ERROR(num("storage_io_thread_pool_size", &threads));
  ABSL_RETURN_IF_ERROR(kv_cache::ValidateWorkerParallelism(parallelism));
  if (threads < 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "storage_io_thread_pool_size must be >= 1, got ", threads));
  }
  options.tp_size = parallelism.tp_size;
  options.tp_rank = parallelism.tp_rank;
  options.pcp_size = parallelism.pcp_size;
  options.pcp_rank = parallelism.pcp_rank;
  options.pp_size = parallelism.pp_size;
  options.pp_rank = parallelism.pp_rank;
  options.capacity_bytes = static_cast<size_t>(capacity);
  options.lookup_batch_size =
      batch > 0 ? static_cast<size_t>(batch) : kDefaultLookupBatchSize;
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

  // metadata_cache_max_entries: 0 (off), > 0 (cap) or -1 (unbounded).
  ABSL_RETURN_IF_ERROR(
      num("metadata_cache_max_entries", &options.metadata_cache_max_entries));
  if (options.metadata_cache_max_entries < 0 &&
      options.metadata_cache_max_entries != kUnboundedMetadataCache) {
    return absl::InvalidArgumentError(absl::StrCat(
        "metadata_cache_max_entries must be >= 0 or ", kUnboundedMetadataCache,
        ", got ", options.metadata_cache_max_entries));
  }
  // metadata_cache_ttl_secs: > 0 or -1 (never expire).
  ABSL_RETURN_IF_ERROR(
      num("metadata_cache_ttl_secs", &options.metadata_cache_ttl_secs));
  if (options.metadata_cache_ttl_secs <= 0 &&
      options.metadata_cache_ttl_secs != kUnboundedMetadataCache) {
    return absl::InvalidArgumentError(absl::StrCat(
        "metadata_cache_ttl_secs must be > 0 or ", kUnboundedMetadataCache,
        ", got ", options.metadata_cache_ttl_secs));
  }
  if (options.metadata_cache_max_entries != 0) {
    if (options.metadata_cache_max_entries == kUnboundedMetadataCache) {
      LOG(WARNING) << "POSIX metadata cache is unbounded "
                      "(metadata_cache_max_entries=-1); memory grows with the "
                      "number of distinct shard files looked up.";
    }
    if (options.metadata_cache_ttl_secs == kUnboundedMetadataCache) {
      LOG(WARNING) << "POSIX metadata cache entries never expire "
                      "(metadata_cache_ttl_secs=-1); files removed from "
                      "storage out of band are only forgotten after a failed "
                      "recall.";
    }
  }

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
  auto effective = [](int per_call, int configured) {
    return per_call == kAxisUndeclared ? configured : per_call;
  };
  const ParallelismConfig& configured = parallelism_;
  const ParallelismConfig parallelism = {
      .tp_size = effective(options.parallelism.tp_size, configured.tp_size),
      .tp_rank = effective(options.parallelism.tp_rank, configured.tp_rank),
      .pcp_size = effective(options.parallelism.pcp_size, configured.pcp_size),
      .pcp_rank = effective(options.parallelism.pcp_rank, configured.pcp_rank),
      .pp_size = effective(options.parallelism.pp_size, configured.pp_size),
      .pp_rank = effective(options.parallelism.pp_rank, configured.pp_rank),
  };
  ABSL_RETURN_IF_ERROR(kv_cache::ValidateWorkerParallelism(parallelism));

  const std::string topology_dir = FormatTopologyDir(parallelism);

  // block_hash is opaque binary; encode before it contributes to a path.
  const std::string hash_hex = absl::BytesToHexString(block_hash);
  // Right-pad so the prefix directories are fixed-width for every input
  // length. The filename uses the complete unpadded hash, so short hashes
  // never collide.
  const std::string padded = absl::StrCat(
      hash_hex, std::string(kHashDirL1Width + kHashDirL2Width, '0'));
  const absl::string_view l1(padded.data(), kHashDirL1Width);
  const absl::string_view l2(padded.data() + kHashDirL1Width, kHashDirL2Width);

  std::string resolved_path = absl::StrCat(root_dir_, "/", model_name_, "/");
  if (!topology_dir.empty()) absl::StrAppend(&resolved_path, topology_dir, "/");
  absl::StrAppend(&resolved_path, l1, "/", l2, "/", hash_hex, ".bin");
  // Block identity stays the raw bytes; only the path is hex-encoded.
  return BlockKey{block_hash, resolved_path, /*offset=*/0, /*size=*/0};
}

// --- PosixKVCacheStoreBackend Implementation ---

MetadataCacheOptions MetadataCacheOptions::FromPosixOptions(
    const PosixBackendOptions& options) {
  MetadataCacheOptions cache_options;
  cache_options.max_entries =
      options.metadata_cache_max_entries == kUnboundedMetadataCache
          ? std::numeric_limits<size_t>::max()
          : static_cast<size_t>(options.metadata_cache_max_entries);
  cache_options.ttl = options.metadata_cache_ttl_secs == kUnboundedMetadataCache
                          ? absl::InfiniteDuration()
                          : absl::Seconds(options.metadata_cache_ttl_secs);
  return cache_options;
}

PosixKVCacheStoreBackend::PosixKVCacheStoreBackend(
    std::shared_ptr<KVBackend> storage_backend, std::string name,
    size_t capacity_bytes, size_t lookup_batch_size,
    MetadataCacheOptions cache_options)
    : storage_backend_(std::move(storage_backend)),
      name_(std::move(name)),
      capacity_bytes_(capacity_bytes),
      lookup_batch_size_(lookup_batch_size > 0 ? lookup_batch_size
                                               : kDefaultLookupBatchSize),
      cache_options_(cache_options),
      cache_(cache_options.max_entries) {
  if (cache_options_.enabled()) {
    LOG(INFO) << "PosixKVCacheStoreBackend '" << name_
              << "': metadata cache enabled (max_entries="
              << (cache_options_.max_entries ==
                          std::numeric_limits<size_t>::max()
                      ? std::string("unbounded")
                      : absl::StrCat(cache_options_.max_entries))
              << ", ttl=" << cache_options_.ttl << ")";
  }
}

size_t PosixKVCacheStoreBackend::metadata_cache_size() const {
  absl::MutexLock lock(cache_mu_);
  return cache_.size();
}

RaidenBlockId PosixKVCacheStoreBackend::MakeSharedStorageBlock() const {
  RaidenBlockId block;
  block.status = BlockStatus::SHARED_STORAGE;
  block.raiden_id.job_replica_id = "shared";
  block.raiden_id.data_name = name_;
  return block;
}

absl::StatusOr<std::vector<BlockKey>> PosixKVCacheStoreBackend::MapShardKeys(
    const std::string& block_hash) const {
  const std::shared_ptr<BlockKeyMapper> mapper = storage_backend_->mapper();
  const int pp_size = mapper->pp_size();
  const int pcp_size = mapper->pcp_size();
  const int tp_size = mapper->tp_size();
  // In secondary storage every (pp_rank, pcp_rank, tp_rank) worker writes its
  // own shard file of each block. The block is available only if every shard
  // is present. An undeclared axis spans a single shard and is left unset in
  // the mapping options, so it adds no path segment.
  auto declare = [](int size, int rank, int* size_field, int* rank_field) {
    if (size == kAxisUndeclared) return;
    *size_field = size;
    *rank_field = rank;
  };
  std::vector<BlockKey> keys;
  keys.reserve(mapper->shards_per_block());
  for (int pp_rank = 0; pp_rank < std::max(1, pp_size); ++pp_rank) {
    for (int pcp_rank = 0; pcp_rank < std::max(1, pcp_size); ++pcp_rank) {
      for (int tp_rank = 0; tp_rank < std::max(1, tp_size); ++tp_rank) {
        backends::KeyMappingOptions lookup_opts;
        ParallelismConfig& p = lookup_opts.parallelism;
        declare(pp_size, pp_rank, &p.pp_size, &p.pp_rank);
        declare(pcp_size, pcp_rank, &p.pcp_size, &p.pcp_rank);
        declare(tp_size, tp_rank, &p.tp_size, &p.tp_rank);
        ABSL_ASSIGN_OR_RETURN(BlockKey key,
                              mapper->MapKey(block_hash, lookup_opts));
        keys.push_back(std::move(key));
      }
    }
  }
  return keys;
}

std::vector<std::vector<bool>> PosixKVCacheStoreBackend::CachedFresh(
    absl::Span<const std::vector<BlockKey>> shard_keys) {
  std::vector<std::vector<bool>> fresh;
  fresh.reserve(shard_keys.size());
  for (const std::vector<BlockKey>& keys : shard_keys) {
    fresh.emplace_back(keys.size(), false);
  }
  if (!cache_options_.enabled()) return fresh;
  const absl::Time now = absl::Now();
  absl::MutexLock lock(cache_mu_);
  for (size_t i = 0; i < shard_keys.size(); ++i) {
    for (size_t j = 0; j < shard_keys[i].size(); ++j) {
      const std::string& resolved_key = shard_keys[i][j].resolved_key;
      ExistenceEntry* entry = cache_.Get(resolved_key);  // Promotes to MRU.
      if (entry == nullptr) continue;
      if (now - entry->last_used >= cache_options_.ttl) {
        cache_.Erase(resolved_key);
        continue;
      }
      entry->last_used = now;  // Every hit refreshes the TTL.
      fresh[i][j] = true;
    }
  }
  return fresh;
}

void PosixKVCacheStoreBackend::CacheInsert(
    absl::Span<const BlockKey* const> keys) {
  if (!cache_options_.enabled() || keys.empty()) return;
  const absl::Time now = absl::Now();
  absl::MutexLock lock(cache_mu_);
  for (const BlockKey* key : keys) {
    // Confirmed on storage: insert, or refresh the TTL if already cached.
    std::optional<std::pair<std::string, ExistenceEntry>> evicted =
        cache_.Put(key->resolved_key, ExistenceEntry{now});
    // LRUCache parks the evicted entry on its candidate list; reclaim it now
    // so the cache never holds more than max_entries.
    if (evicted.has_value()) cache_.Erase(evicted->first);
  }
}

std::vector<bool> PosixKVCacheStoreBackend::ProbeExists(
    absl::Span<const BlockKey> keys) {
  std::vector<bool> exists(keys.size(), false);
  std::promise<std::vector<absl::StatusOr<bool>>> promise;
  auto future = promise.get_future();
  storage_backend_->BatchExistsAsync(
      keys, [&promise](std::vector<absl::StatusOr<bool>> res) {
        promise.set_value(std::move(res));
      });
  const std::vector<absl::StatusOr<bool>> answers = future.get();
  // A short answer cannot be attributed to keys reliably; treat it as absent.
  if (answers.size() != keys.size()) return exists;
  for (size_t i = 0; i < answers.size(); ++i) {
    exists[i] = answers[i].ok() && *answers[i];
  }
  return exists;
}

// Queries `block_hashes` on storage according to `options`:
// - Returns empty immediately when `options.host_resident_only` is set (POSIX
//   is not host-DRAM-resident) or when `block_hashes` is empty.
// - Each block hash maps via MapShardKeys(hash) to N = max(1, S) * max(1, P) *
//   max(1, T) shard file keys across the (PP, PCP, TP) topology; all shard keys
//   must exist for a block to be available. When the positive-only TTL metadata
//   cache is enabled, fresh shard hits avoid storage probes.
// - Phase 0 (map): maps each block hash to its shard keys (`kPrefix` stops at
//   the first unmappable hash; `kSubset` records an empty key list and
//   continues to preserve index alignment).
// - Phase 1 (cache): marks shard keys with a fresh metadata cache entry as
//   present and refreshes their `last_used` timestamp.
// - Phase 2 (storage): probes cache-missed shard keys via BatchExistsAsync in
//   chunks of `lookup_batch_size_` (`kPrefix` exits early after the first chunk
//   with an absent key; `kSubset` probes all chunks) and caches confirmed keys.
// - Phase 3 (merge): emits SHARED_STORAGE entries (`kPrefix` up to the first
//   unavailable block; `kSubset` for all available blocks).
absl::StatusOr<BlockSliceList> PosixKVCacheStoreBackend::Lookup(
    absl::Span<const std::string> block_hashes, const LookupOptions& options) {
  if (options.match_mode == LookupMatchMode::kSubset && options.enable_global) {
    return absl::InvalidArgumentError(
        "LookupMatchMode::kSubset is not supported with enable_global=true");
  }
  if (options.host_resident_only && options.enable_global) {
    return absl::InvalidArgumentError(
        "host_resident_only is not supported with enable_global=true");
  }
  if (options.include_eviction_candidates && options.pin_found) {
    return absl::InvalidArgumentError(
        "include_eviction_candidates is not supported with pin_found=true");
  }

  BlockSliceList results;
  if (options.host_resident_only || !storage_backend_ ||
      !storage_backend_->mapper() || block_hashes.empty()) {
    return results;
  }

  // Phase 0 (map).
  std::vector<std::vector<BlockKey>> shard_keys;
  shard_keys.reserve(block_hashes.size());
  for (const std::string& hash : block_hashes) {
    absl::StatusOr<std::vector<BlockKey>> keys = MapShardKeys(hash);
    if (!keys.ok()) {
      if (options.match_mode == LookupMatchMode::kSubset) {
        shard_keys.emplace_back();
        continue;
      }
      break;
    }
    shard_keys.push_back(*std::move(keys));
  }
  if (shard_keys.empty()) return results;

  // Phase 1 (cache).
  std::vector<std::vector<bool>> present = CachedFresh(shard_keys);

  // Phase 2 (storage): flat list of cache-missed shard keys across all blocks,
  // probed in chunks of lookup_batch_size_.
  struct ShardRef {
    size_t block;
    size_t shard;
  };
  std::vector<ShardRef> misses;
  for (size_t i = 0; i < shard_keys.size(); ++i) {
    for (size_t j = 0; j < shard_keys[i].size(); ++j) {
      if (!present[i][j]) misses.push_back({i, j});
    }
  }
  std::vector<const BlockKey*> confirmed;
  for (size_t offset = 0; offset < misses.size();
       offset += lookup_batch_size_) {
    const size_t chunk_len =
        std::min(lookup_batch_size_, misses.size() - offset);
    std::vector<BlockKey> chunk_keys;
    chunk_keys.reserve(chunk_len);
    for (size_t k = 0; k < chunk_len; ++k) {
      const ShardRef& miss = misses[offset + k];
      chunk_keys.push_back(shard_keys[miss.block][miss.shard]);
    }
    const std::vector<bool> exists = ProbeExists(chunk_keys);
    bool chunk_has_absent = false;
    for (size_t k = 0; k < chunk_len; ++k) {
      const ShardRef& miss = misses[offset + k];
      if (exists[k]) {
        present[miss.block][miss.shard] = true;
        confirmed.push_back(&shard_keys[miss.block][miss.shard]);
      } else {
        chunk_has_absent = true;
      }
    }
    if (options.match_mode == LookupMatchMode::kPrefix && chunk_has_absent) {
      break;  // Early exit; see Phase 2 above.
    }
  }
  CacheInsert(confirmed);

  // Phase 3 (merge).
  auto all_present = [](const std::vector<bool>& shards) {
    return !shards.empty() && absl::c_all_of(shards, [](bool shard_present) {
      return shard_present;
    });
  };
  if (options.match_mode == LookupMatchMode::kSubset) {
    for (size_t i = 0; i < shard_keys.size(); ++i) {
      if (all_present(present[i])) {
        results.push_back(
            std::make_pair(block_hashes[i], MakeSharedStorageBlock()));
      }
    }
  } else {
    for (size_t i = 0; i < shard_keys.size() && all_present(present[i]); ++i) {
      results.push_back(
          std::make_pair(block_hashes[i], MakeSharedStorageBlock()));
    }
  }
  return results;
}

size_t PosixKVCacheStoreBackend::Delete(
    absl::Span<const std::string> block_hashes, const DeleteOptions& options) {
  if (options.deallocated_host_block_ids != nullptr) {
    options.deallocated_host_block_ids->clear();
  }
  if (options.deleted_hashes != nullptr) {
    options.deleted_hashes->clear();
  }
  if (options.host_resident_only || !cache_options_.enabled() ||
      !storage_backend_ || !storage_backend_->mapper()) {
    return 0;
  }

  struct MappedBlock {
    absl::string_view hash;
    std::vector<std::string> resolved_keys;
  };
  std::vector<MappedBlock> mapped_blocks;
  mapped_blocks.reserve(block_hashes.size());
  for (const std::string& hash : block_hashes) {
    absl::StatusOr<std::vector<BlockKey>> keys = MapShardKeys(hash);
    if (!keys.ok()) continue;  // Never cached; keep invalidating the rest.
    MappedBlock entry{.hash = hash};
    entry.resolved_keys.reserve(keys->size());
    for (BlockKey& key : *keys) {
      entry.resolved_keys.push_back(std::move(key.resolved_key));
    }
    mapped_blocks.push_back(std::move(entry));
  }

  size_t erased_count = 0;
  absl::MutexLock lock(cache_mu_);
  for (const MappedBlock& entry : mapped_blocks) {
    bool erased_any = false;
    for (const std::string& resolved_key : entry.resolved_keys) {
      if (cache_.Erase(resolved_key)) {
        erased_any = true;
      }
    }
    if (erased_any) {
      ++erased_count;
      if (options.deleted_hashes != nullptr) {
        options.deleted_hashes->emplace_back(entry.hash);
      }
    }
  }
  return erased_count;
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
       ::tpu_raiden::controller::RaidenController* /*controller*/)
        -> absl::StatusOr<
            std::shared_ptr<::tpu_raiden::kv_cache::KVCacheStoreBackend>> {
      // The coordinator probes every rank explicitly (see
      // PosixKVCacheStoreBackend::MapShardKeys), so the default mapper rank of
      // each declared axis is pinned to 0. Per-worker ranks live on the
      // worker's own config.
      ABSL_ASSIGN_OR_RETURN(
          const ::tpu_raiden::kv_cache::backends::ParallelismConfig coordinator,
          ::tpu_raiden::kv_cache::ResolveCoordinatorParallelism(
              config.parallelism));
      // With no axis declared every worker would write the same shard files,
      // so a recall could load another worker's KV. KVCacheStore::Create
      // already rejects this; the check stays for direct factory callers.
      ABSL_RETURN_IF_ERROR(::tpu_raiden::kv_cache::RequireDeclaredAxis(
          config.parallelism, coordinator));
      ::tpu_raiden::kv_cache::BackendConfig resolved = config;
      ::tpu_raiden::kv_cache::ApplyParallelismToProperties(coordinator,
                                                           &resolved);
      ABSL_ASSIGN_OR_RETURN(
          const PosixBackendOptions options,
          PosixBackendOptions::FromProperties(resolved.properties));

      const std::string backend_name = std::string(
          ::tpu_raiden::kv_cache::backends::storage::kPosixBackendName);
      auto backend =
          std::make_shared<PosixKVBackend>(backend_name, resolved.properties);
      LOG(INFO) << "[Store] " << backend_name << " backend probes "
                << backend->mapper()->shards_per_block()
                << " shard file(s) per block under " << options.root_dir << "/"
                << ::tpu_raiden::kv_cache::backends::storage::SanitizeModelName(
                       options.model_name)
                << "/"
                << ::tpu_raiden::kv_cache::backends::storage::
                       FormatProbedTopologyDirPattern(coordinator)
                << "/";
      return std::make_shared<PosixKVCacheStoreBackend>(
          std::move(backend), backend_name, options.capacity_bytes,
          options.lookup_batch_size,
          ::tpu_raiden::kv_cache::backends::storage::MetadataCacheOptions::
              FromPosixOptions(options));
    });
