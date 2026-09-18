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
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/logging.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/kv_cache/backends/backend.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace storage {

namespace {

struct LustreLovUserMdV1 {
  uint32_t lmm_magic;
  uint32_t lmm_pattern;
  uint64_t lmm_object_id;
  uint64_t lmm_object_seq;
  uint32_t lmm_stripe_size;
  uint16_t lmm_stripe_count;
  uint16_t lmm_stripe_offset;
};

constexpr uint32_t kLovUserMagicV1 = 0x0BD10BD0;
constexpr uint32_t kLovPatternRaid0 = 0x00000001;

#ifndef LL_IOC_LOV_SETSTRIPE
#define LL_IOC_LOV_SETSTRIPE _IOW('f', 154, long)
#endif

absl::StatusOr<int> OpenFile(const std::string& path, int flags, mode_t mode,
                             bool use_direct_io,
                             const TdsBackendOptions& options = {}) {
  if (flags & O_CREAT) {
    std::filesystem::path fs_path(path);
    std::filesystem::path parent = fs_path.parent_path();
    if (!parent.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(parent, ec);
      if (ec && !std::filesystem::exists(parent, ec)) {
        return absl::InternalError(
            absl::StrCat("Failed to create parent directories for ", path, ": ",
                         ec.message()));
      }
    }
  }

  int fd = -1;
  if (use_direct_io) {
    fd = ::open(path.c_str(), flags | O_DIRECT, mode);
    if (fd < 0 && errno == EINVAL) {
      // Fallback for filesystems like tmpfs in test sandboxes that reject
      // O_DIRECT with EINVAL.
      fd = ::open(path.c_str(), flags & ~O_DIRECT, mode);
    }
  } else {
    fd = ::open(path.c_str(), flags & ~O_DIRECT, mode);
  }

  if (fd < 0) {
    return absl::ErrnoToStatus(errno,
                               absl::StrCat("Failed to open file: ", path));
  }

  if ((flags & O_CREAT) &&
      (options.lustre_stripe_count > 0 || options.lustre_stripe_size > 0)) {
    LustreLovUserMdV1 lum = {};
    lum.lmm_magic = kLovUserMagicV1;
    lum.lmm_pattern = kLovPatternRaid0;
    lum.lmm_stripe_size = static_cast<uint32_t>(options.lustre_stripe_size);
    lum.lmm_stripe_count = static_cast<uint16_t>(options.lustre_stripe_count);
    lum.lmm_stripe_offset = static_cast<uint16_t>(-1);
    if (::ioctl(fd, LL_IOC_LOV_SETSTRIPE, &lum) < 0) {
      VLOG(1) << "Optional LL_IOC_LOV_SETSTRIPE ioctl on " << path
              << " returned: " << std::strerror(errno);
    }
  }

  return fd;
}

}  // namespace

absl::StatusOr<TdsBackendOptions> TdsBackendOptions::FromProperties(
    const absl::flat_hash_map<std::string, std::string>& properties) {
  TdsBackendOptions options;

  if (auto it = properties.find("root_dir"); it != properties.end()) {
    if (it->second.empty()) {
      return absl::InvalidArgumentError("root_dir cannot be empty");
    }
    options.root_dir = it->second;
  }

  if (auto it = properties.find("storage_io_thread_pool_size");
      it != properties.end()) {
    if (!absl::SimpleAtoi(it->second, &options.storage_io_thread_pool_size) ||
        options.storage_io_thread_pool_size <= 0) {
      return absl::InvalidArgumentError(
          "storage_io_thread_pool_size must be a positive integer");
    }
  }

  if (auto it = properties.find("alignment"); it != properties.end()) {
    if (!absl::SimpleAtoi(it->second, &options.alignment) ||
        options.alignment == 0 ||
        (options.alignment & (options.alignment - 1)) != 0) {
      return absl::InvalidArgumentError(
          "alignment must be a positive power of 2");
    }
  }

  if (auto it = properties.find("use_direct_io"); it != properties.end()) {
    if (it->second == "true" || it->second == "1") {
      options.use_direct_io = true;
    } else if (it->second == "false" || it->second == "0") {
      options.use_direct_io = false;
    } else if (!absl::SimpleAtob(it->second, &options.use_direct_io)) {
      return absl::InvalidArgumentError(
          "use_direct_io must be a valid boolean");
    }
  }

  if (auto it = properties.find("lustre_stripe_size"); it != properties.end()) {
    if (!absl::SimpleAtoi(it->second, &options.lustre_stripe_size)) {
      return absl::InvalidArgumentError(
          "lustre_stripe_size must be a non-negative integer");
    }
  }

  if (auto it = properties.find("lustre_stripe_count");
      it != properties.end()) {
    if (!absl::SimpleAtoi(it->second, &options.lustre_stripe_count) ||
        options.lustre_stripe_count < 0) {
      return absl::InvalidArgumentError(
          "lustre_stripe_count must be a non-negative integer");
    }
  }

  return options;
}

TdsKVBackend::TdsKVBackend(
    std::string name, absl::flat_hash_map<std::string, std::string> properties)
    : KVBackend(properties),
      name_(std::move(name)),
      options_(TdsBackendOptions::FromProperties(properties)
                   .value_or(TdsBackendOptions{})),
      thread_pool_(std::make_unique<NumaThreadPool>(
          std::max(1, options_.storage_io_thread_pool_size))) {}

TdsKVBackend::TdsKVBackend(
    std::string name, const TdsBackendOptions& options,
    absl::flat_hash_map<std::string, std::string> properties)
    : KVBackend(std::move(properties)),
      name_(std::move(name)),
      options_(options),
      thread_pool_(std::make_unique<NumaThreadPool>(
          std::max(1, options_.storage_io_thread_pool_size))) {}

TdsKVBackend::~TdsKVBackend() { thread_pool_.reset(); }

absl::Status TdsKVBackend::RegisterBuffer(void* ptr, size_t size) {
  if (ptr == nullptr || size == 0) {
    return absl::InvalidArgumentError(
        "Buffer pointer must be non-null and size must be positive");
  }
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  absl::MutexLock lock(mu_);
  // Check for overlapping registrations.
  auto it = registrations_.upper_bound(addr);
  if (it != registrations_.end() && addr + size > it->first) {
    return absl::AlreadyExistsError(
        "Buffer overlaps with existing registration");
  }
  if (it != registrations_.begin()) {
    auto prev = std::prev(it);
    if (prev->first + prev->second > addr) {
      return absl::AlreadyExistsError(
          "Buffer overlaps with existing registration");
    }
  }
  registrations_[addr] = size;
  return absl::OkStatus();
}

absl::Status TdsKVBackend::UnregisterBuffer(void* ptr) {
  if (ptr == nullptr) {
    return absl::InvalidArgumentError("Buffer pointer must be non-null");
  }
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  absl::MutexLock lock(mu_);
  auto it = registrations_.find(addr);
  if (it == registrations_.end()) {
    return absl::NotFoundError("Buffer not found in registration table");
  }
  registrations_.erase(it);
  return absl::OkStatus();
}

bool TdsKVBackend::IsRegisteredLocked(const void* ptr, size_t size) const {
  if (ptr == nullptr) return false;
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  auto it = registrations_.upper_bound(addr);
  if (it == registrations_.begin()) {
    return false;
  }
  --it;
  return addr >= it->first && (addr + size) <= (it->first + it->second);
}

bool TdsKVBackend::IsDmaAlignedAndRegistered(const void* ptr, size_t size,
                                             int64_t file_offset) const {
  if (ptr == nullptr || file_offset < 0) return false;
  const size_t align = options_.alignment;
  if (align == 0) return false;
  if (static_cast<size_t>(file_offset) % align != 0) return false;
  if (reinterpret_cast<uintptr_t>(ptr) % align != 0) return false;
  if (size % align != 0) return false;

  absl::MutexLock lock(mu_);
  return IsRegisteredLocked(ptr, size);
}

bool TdsKVBackend::CanUseDirectZeroCopy(
    int64_t file_offset, absl::Span<const HostBufferDescriptor> slices) const {
  const size_t align = options_.alignment;
  if (align == 0 || file_offset < 0 ||
      static_cast<size_t>(file_offset) % align != 0) {
    return false;
  }
  absl::MutexLock lock(mu_);
  for (const auto& slice : slices) {
    if (slice.ptr == nullptr && slice.size > 0) {
      return false;
    }
    if (reinterpret_cast<uintptr_t>(slice.ptr) % align != 0) {
      return false;
    }
    if (slice.size % align != 0) {
      return false;
    }
    if (!IsRegisteredLocked(slice.ptr, slice.size)) {
      return false;
    }
  }
  return true;
}

TdsStats TdsKVBackend::stats() const {
  absl::MutexLock lock(mu_);
  return stats_;
}

void TdsKVBackend::ResetStats() {
  absl::MutexLock lock(mu_);
  stats_ = TdsStats{};
}

absl::StatusOr<std::string> TdsKVBackend::ResolvePath(
    const BlockKey& key) const {
  std::string target = key.resolved_key;
  if (target.empty() && mapper_ != nullptr) {
    ABSL_ASSIGN_OR_RETURN(BlockKey mapped, mapper_->MapKey(key.block_hash));
    target = mapped.resolved_key;
  }
  if (target.empty()) {
    target = key.block_hash;
  }
  if (target.empty()) {
    return absl::InvalidArgumentError("BlockKey has empty path and hash");
  }

  // Strip URI scheme if present (e.g., lustre://, nfs://, file://).
  size_t scheme_pos = target.find("://");
  if (scheme_pos != std::string::npos) {
    target = target.substr(scheme_pos + 3);
  }

  if (!target.empty() && target[0] == '/') {
    return target;
  }
  return absl::StrCat(options_.root_dir, "/", target);
}

absl::Status TdsKVBackend::ExecuteWrite(
    const BlockKey& key, absl::Span<const HostBufferDescriptor> slices,
    size_t total_bytes) {
  if (key.offset < 0) {
    return absl::InvalidArgumentError("Negative key offset");
  }

  size_t sum_bytes = 0;
  for (const auto& slice : slices) {
    if (slice.size > 0 && slice.ptr == nullptr) {
      return absl::InvalidArgumentError(
          "Null slice pointer with non-zero size");
    }
    sum_bytes += slice.size;
  }
  if (sum_bytes != total_bytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("total_bytes (", total_bytes,
                     ") does not match sum of slice sizes (", sum_bytes, ")"));
  }

  ABSL_ASSIGN_OR_RETURN(std::string path, ResolvePath(key));

  const bool direct_zero_copy = CanUseDirectZeroCopy(key.offset, slices);
  if (direct_zero_copy) {
    ABSL_ASSIGN_OR_RETURN(int fd, OpenFile(path, O_WRONLY | O_CREAT, 0644,
                                           options_.use_direct_io, options_));
    absl::Cleanup close_fd = [fd]() { ::close(fd); };

    int64_t current_offset = key.offset;
    for (const auto& slice : slices) {
      size_t written = 0;
      const uint8_t* buf = static_cast<const uint8_t*>(slice.ptr);
      while (written < slice.size) {
        ssize_t n = ::pwrite(fd, buf + written, slice.size - written,
                             current_offset + static_cast<int64_t>(written));
        if (n < 0) {
          if (errno == EINTR) continue;
          return absl::ErrnoToStatus(errno, "Direct zero-copy pwrite failed");
        }
        written += static_cast<size_t>(n);
      }
      current_offset += static_cast<int64_t>(slice.size);
    }

    absl::MutexLock lock(mu_);
    stats_.ops++;
    stats_.direct_ops++;
    stats_.bytes += static_cast<int64_t>(total_bytes);
    return absl::OkStatus();
  }

  // 4KB Bounce-Buffer Path.
  ABSL_ASSIGN_OR_RETURN(int fd, OpenFile(path, O_RDWR | O_CREAT, 0644,
                                         options_.use_direct_io, options_));
  absl::Cleanup close_fd = [fd]() { ::close(fd); };

  if (total_bytes == 0) {
    absl::MutexLock lock(mu_);
    stats_.ops++;
    stats_.bounced_ops++;
    return absl::OkStatus();
  }

  struct stat st = {};
  int64_t initial_file_size = 0;
  if (::fstat(fd, &st) == 0) {
    initial_file_size = st.st_size;
  }

  const size_t alignment = options_.alignment;
  const int64_t aligned_start = (key.offset / static_cast<int64_t>(alignment)) *
                                static_cast<int64_t>(alignment);
  const int64_t end_offset = key.offset + static_cast<int64_t>(total_bytes);
  const int64_t aligned_end =
      ((end_offset + static_cast<int64_t>(alignment) - 1) /
       static_cast<int64_t>(alignment)) *
      static_cast<int64_t>(alignment);
  const size_t aligned_size = static_cast<size_t>(aligned_end - aligned_start);
  const size_t head_pad = static_cast<size_t>(key.offset - aligned_start);

  void* bounce_ptr = nullptr;
  if (::posix_memalign(&bounce_ptr, alignment, aligned_size) != 0 ||
      bounce_ptr == nullptr) {
    return absl::ResourceExhaustedError(
        "Failed to allocate aligned bounce buffer");
  }
  std::unique_ptr<void, decltype(&std::free)> bounce_guard(bounce_ptr,
                                                           &std::free);
  std::memset(bounce_ptr, 0, aligned_size);

  // If the file already has data overlapping the bounce window and we have
  // unaligned head or tail padding, pre-read existing blocks to preserve them.
  if (initial_file_size > aligned_start &&
      (head_pad > 0 || end_offset < aligned_end)) {
    const size_t existing_in_window = static_cast<size_t>(
        std::min<int64_t>(aligned_end, initial_file_size) - aligned_start);
    size_t pre_read = 0;
    while (pre_read < existing_in_window) {
      ssize_t n = ::pread(fd, static_cast<uint8_t*>(bounce_ptr) + pre_read,
                          aligned_size - pre_read,
                          aligned_start + static_cast<int64_t>(pre_read));
      if (n < 0) {
        if (errno == EINTR) continue;
        if (errno == EINVAL) {
          int fl = ::fcntl(fd, F_GETFL);
          if (fl >= 0 && (fl & O_DIRECT)) {
            ::fcntl(fd, F_SETFL, fl & ~O_DIRECT);
            continue;
          }
        }
        break;
      }
      if (n == 0) break;
      pre_read += static_cast<size_t>(n);
    }
    if (options_.use_direct_io) {
      int fl = ::fcntl(fd, F_GETFL);
      if (fl >= 0 && !(fl & O_DIRECT)) {
        ::fcntl(fd, F_SETFL, fl | O_DIRECT);
      }
    }
  }

  // Copy slices into the bounce buffer at head_pad.
  size_t offset_in_slices = 0;
  for (const auto& slice : slices) {
    if (slice.size > 0) {
      std::memcpy(
          static_cast<uint8_t*>(bounce_ptr) + head_pad + offset_in_slices,
          slice.ptr, slice.size);
      offset_in_slices += slice.size;
    }
  }

  // Write the aligned bounce buffer to disk.
  size_t written = 0;
  while (written < aligned_size) {
    ssize_t n = ::pwrite(fd, static_cast<const uint8_t*>(bounce_ptr) + written,
                         aligned_size - written,
                         aligned_start + static_cast<int64_t>(written));
    if (n < 0) {
      if (errno == EINTR) continue;
      if (errno == EINVAL) {
        int fl = ::fcntl(fd, F_GETFL);
        if (fl >= 0 && (fl & O_DIRECT)) {
          ::fcntl(fd, F_SETFL, fl & ~O_DIRECT);
          continue;
        }
      }
      return absl::ErrnoToStatus(errno, "Bounce buffer pwrite failed");
    }
    written += static_cast<size_t>(n);
  }

  // Ensure logical file size matches exact written bytes if newly created or
  // extended to less than aligned_end.
  const int64_t target_file_size =
      std::max<int64_t>(initial_file_size, end_offset);
  if (target_file_size < aligned_end) {
    if (::ftruncate(fd, target_file_size) < 0) {
      return absl::ErrnoToStatus(errno, "ftruncate failed after bounce write");
    }
  }

  absl::MutexLock lock(mu_);
  stats_.ops++;
  stats_.bounced_ops++;
  stats_.bytes += static_cast<int64_t>(total_bytes);
  return absl::OkStatus();
}

absl::Status TdsKVBackend::ExecuteRead(
    const BlockKey& key, absl::Span<const HostBufferDescriptor> slices,
    size_t total_bytes) {
  if (key.offset < 0) {
    return absl::InvalidArgumentError("Negative key offset");
  }

  size_t sum_bytes = 0;
  for (const auto& slice : slices) {
    if (slice.size > 0 && slice.ptr == nullptr) {
      return absl::InvalidArgumentError(
          "Null slice pointer with non-zero size");
    }
    sum_bytes += slice.size;
  }
  if (sum_bytes != total_bytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("total_bytes (", total_bytes,
                     ") does not match sum of slice sizes (", sum_bytes, ")"));
  }

  ABSL_ASSIGN_OR_RETURN(std::string path, ResolvePath(key));

  const bool direct_zero_copy = CanUseDirectZeroCopy(key.offset, slices);
  if (direct_zero_copy) {
    ABSL_ASSIGN_OR_RETURN(
        int fd, OpenFile(path, O_RDONLY, 0, options_.use_direct_io, options_));
    absl::Cleanup close_fd = [fd]() { ::close(fd); };

    int64_t current_offset = key.offset;
    for (const auto& slice : slices) {
      size_t bytes_read = 0;
      uint8_t* buf = static_cast<uint8_t*>(slice.ptr);
      while (bytes_read < slice.size) {
        ssize_t n = ::pread(fd, buf + bytes_read, slice.size - bytes_read,
                            current_offset + static_cast<int64_t>(bytes_read));
        if (n < 0) {
          if (errno == EINTR) continue;
          return absl::ErrnoToStatus(errno, "Direct zero-copy pread failed");
        }
        if (n == 0) {
          return absl::OutOfRangeError(absl::StrCat(
              "Unexpected EOF during direct zero-copy read at offset ",
              current_offset + static_cast<int64_t>(bytes_read)));
        }
        bytes_read += static_cast<size_t>(n);
      }
      current_offset += static_cast<int64_t>(slice.size);
    }

    absl::MutexLock lock(mu_);
    stats_.ops++;
    stats_.direct_ops++;
    stats_.bytes += static_cast<int64_t>(total_bytes);
    return absl::OkStatus();
  }

  // 4KB Bounce-Buffer Path.
  ABSL_ASSIGN_OR_RETURN(
      int fd, OpenFile(path, O_RDONLY, 0, options_.use_direct_io, options_));
  absl::Cleanup close_fd = [fd]() { ::close(fd); };

  if (total_bytes == 0) {
    absl::MutexLock lock(mu_);
    stats_.ops++;
    stats_.bounced_ops++;
    return absl::OkStatus();
  }

  const size_t alignment = options_.alignment;
  const int64_t aligned_start = (key.offset / static_cast<int64_t>(alignment)) *
                                static_cast<int64_t>(alignment);
  const int64_t end_offset = key.offset + static_cast<int64_t>(total_bytes);
  const int64_t aligned_end =
      ((end_offset + static_cast<int64_t>(alignment) - 1) /
       static_cast<int64_t>(alignment)) *
      static_cast<int64_t>(alignment);
  const size_t aligned_size = static_cast<size_t>(aligned_end - aligned_start);
  const size_t head_pad = static_cast<size_t>(key.offset - aligned_start);
  const size_t required_bytes = head_pad + total_bytes;

  void* bounce_ptr = nullptr;
  if (::posix_memalign(&bounce_ptr, alignment, aligned_size) != 0 ||
      bounce_ptr == nullptr) {
    return absl::ResourceExhaustedError(
        "Failed to allocate aligned bounce buffer");
  }
  std::unique_ptr<void, decltype(&std::free)> bounce_guard(bounce_ptr,
                                                           &std::free);

  size_t total_read = 0;
  while (total_read < required_bytes) {
    ssize_t n = ::pread(fd, static_cast<uint8_t*>(bounce_ptr) + total_read,
                        aligned_size - total_read,
                        aligned_start + static_cast<int64_t>(total_read));
    if (n < 0) {
      if (errno == EINTR) continue;
      if (errno == EINVAL) {
        int fl = ::fcntl(fd, F_GETFL);
        if (fl >= 0 && (fl & O_DIRECT)) {
          ::fcntl(fd, F_SETFL, fl & ~O_DIRECT);
          continue;
        }
      }
      return absl::ErrnoToStatus(errno, "Bounce buffer pread failed");
    }
    if (n == 0) {
      break;
    }
    total_read += static_cast<size_t>(n);
  }

  if (total_read < required_bytes) {
    return absl::OutOfRangeError(
        absl::StrCat("Short read in bounce buffer: required ", required_bytes,
                     " bytes, got ", total_read));
  }

  size_t offset_in_slices = 0;
  for (const auto& slice : slices) {
    if (slice.size > 0) {
      std::memcpy(
          slice.ptr,
          static_cast<const uint8_t*>(bounce_ptr) + head_pad + offset_in_slices,
          slice.size);
      offset_in_slices += slice.size;
    }
  }

  absl::MutexLock lock(mu_);
  stats_.ops++;
  stats_.bounced_ops++;
  stats_.bytes += static_cast<int64_t>(total_bytes);
  return absl::OkStatus();
}

void TdsKVBackend::WriteAsync(const BlockKey& key,
                              absl::Span<const HostBufferDescriptor> slices,
                              size_t total_bytes,
                              std::function<void(absl::Status)> callback) {
  std::vector<HostBufferDescriptor> owned_slices(slices.begin(), slices.end());
  thread_pool_->Schedule([this, key, owned_slices = std::move(owned_slices),
                          total_bytes, callback = std::move(callback)]() {
    absl::Status status = ExecuteWrite(key, owned_slices, total_bytes);
    if (!status.ok()) {
      absl::MutexLock lock(mu_);
      stats_.ops++;
      stats_.failed_ops++;
    }
    if (callback) {
      callback(std::move(status));
    }
  });
}

void TdsKVBackend::ReadAsync(const BlockKey& key,
                             absl::Span<const HostBufferDescriptor> slices,
                             size_t total_bytes,
                             std::function<void(absl::Status)> callback) {
  std::vector<HostBufferDescriptor> owned_slices(slices.begin(), slices.end());
  thread_pool_->Schedule([this, key, owned_slices = std::move(owned_slices),
                          total_bytes, callback = std::move(callback)]() {
    absl::Status status = ExecuteRead(key, owned_slices, total_bytes);
    if (!status.ok()) {
      absl::MutexLock lock(mu_);
      stats_.ops++;
      stats_.failed_ops++;
    }
    if (callback) {
      callback(std::move(status));
    }
  });
}

void TdsKVBackend::BatchExistsAsync(
    absl::Span<const BlockKey> keys,
    std::function<void(std::vector<absl::StatusOr<bool>>)> callback) {
  if (keys.empty()) {
    thread_pool_->Schedule([callback = std::move(callback)]() {
      if (callback) {
        callback({});
      }
    });
    return;
  }

  struct BatchState {
    explicit BatchState(
        size_t count, std::function<void(std::vector<absl::StatusOr<bool>>)> cb)
        : remaining(count), results(count), callback(std::move(cb)) {}
    std::atomic<size_t> remaining;
    std::vector<absl::StatusOr<bool>> results;
    std::function<void(std::vector<absl::StatusOr<bool>>)> callback;
  };

  auto state = std::make_shared<BatchState>(keys.size(), std::move(callback));
  for (size_t i = 0; i < keys.size(); ++i) {
    BlockKey key = keys[i];
    thread_pool_->Schedule([this, i, key = std::move(key), state]() {
      if (key.offset < 0 || key.size < 0) {
        state->results[i] =
            absl::InvalidArgumentError("Negative offset or size in BlockKey");
      } else {
        absl::StatusOr<std::string> path = ResolvePath(key);
        if (!path.ok()) {
          state->results[i] = path.status();
        } else {
          struct stat st = {};
          if (::stat(path->c_str(), &st) == 0) {
            bool exists =
                S_ISREG(st.st_mode) && (st.st_size >= key.offset + key.size);
            state->results[i] = exists;
          } else {
            if (errno == ENOENT || errno == ENOTDIR) {
              state->results[i] = false;
            } else {
              state->results[i] = absl::ErrnoToStatus(
                  errno, absl::StrCat("stat failed for ", *path));
            }
          }
        }
      }

      if (state->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (state->callback) {
          state->callback(std::move(state->results));
        }
      }
    });
  }
}

}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden
