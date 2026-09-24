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
#include <limits.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
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
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "tdsul/def.h"
#include "tdsul/tdsul.h"
#include "absl/base/call_once.h"
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
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/backends/storage/posix_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_sync/tpudirect_storage/src/def_internal.h"  // NOLINT(misc-include-cleaner)

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace storage {

namespace {

// Lustre user-space stripe descriptor for `ioctl(fd, LL_IOC_LOV_SETSTRIPE)`
// on [shared-storage: Lustre/NFS].
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
#define LL_IOC_LOV_SETSTRIPE _IOW('f', 154, long)  // NOLINT(runtime/int)
#endif

// Lustre flag (`O_LOV_DELAY_CREATE`) instructing the Lustre Metadata Server
// (MDS) to defer OST object allocation on `open(O_CREAT)` until the subsequent
// `ioctl(fd, LL_IOC_LOV_SETSTRIPE)` call specifies the RAID-0 stripe geometry.
#ifndef O_LOV_DELAY_CREATE
#define O_LOV_DELAY_CREATE 0100000000
#endif

#ifndef UIO_MAXIOV
#define UIO_MAXIOV 1024
#endif

constexpr size_t kMaxIovCount = UIO_MAXIOV;

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
      LOG(ERROR) << "tds_init failed (" << rc
                 << "); all subsequent libtdsul registration and I/O will fail";
    }
  });
}

// Returns a unique sibling path `<path>.tmp_<pid>_<seq>`. Keeping the temp file
// in the same directory (same filesystem) lets `::rename(tmp_path, path)`
// replace `path` atomically.
std::string MakeUniqueTempPath(absl::string_view path) {
  static std::atomic<uint64_t> tmp_seq{0};
  return absl::StrCat(path, ".tmp_", ::getpid(), "_",
                      tmp_seq.fetch_add(1, std::memory_order_relaxed));
}

// Opens `path` on [local-storage: NVMe SSD] or [shared-storage: Lustre/NFS].
//   - If `use_direct_io`, tries `O_DIRECT` first and reopens without it if the
//     filesystem returns `EINVAL`.
//   - On `ENOENT` with `O_CREAT`, creates parent directories and retries.
//   - If Lustre striping is configured, opens with `O_LOV_DELAY_CREATE` and
//     then sets the layout via `LL_IOC_LOV_SETSTRIPE` (failure is only logged).
absl::StatusOr<int> OpenFile(const std::string& path, int flags, mode_t mode,
                             bool use_direct_io,
                             const TdsBackendOptions& options = {}) {
  const bool has_lustre_striping =
      (flags & O_CREAT) != 0 &&
      (options.lustre_stripe_count > 0 || options.lustre_stripe_size > 0);

  auto try_open_once = [&](int base_flags) -> int {
    const int open_flags =
        has_lustre_striping ? (base_flags | O_LOV_DELAY_CREATE) : base_flags;
    int fd = -1;
    if (use_direct_io) {
      fd = ::open(path.c_str(), open_flags | O_DIRECT, mode);
      if (fd < 0 && errno == EINVAL) {
        fd = ::open(path.c_str(), base_flags & ~O_DIRECT, mode);
      }
    } else {
      fd = ::open(path.c_str(), open_flags & ~O_DIRECT, mode);
      if (fd < 0 && errno == EINVAL && has_lustre_striping) {
        fd = ::open(path.c_str(), base_flags & ~O_DIRECT, mode);
      }
    }
    return fd;
  };

  int fd = try_open_once(flags);
  if (fd < 0 && errno == ENOENT && (flags & O_CREAT) != 0) {
    std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(parent, ec);
      if (ec && !std::filesystem::exists(parent, ec)) {
        return absl::InternalError(
            absl::StrCat("Failed to create parent directories for ", path, ": ",
                         ec.message()));
      }
      fd = try_open_once(flags);
    }
  }

  if (fd < 0) {
    return absl::ErrnoToStatus(errno,
                               absl::StrCat("Failed to open file: ", path));
  }

  if (has_lustre_striping) {
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

// Reports whether `fd` still carries `O_DIRECT`, i.e. whether I/O on it skips
// [dram: OS Page Cache].
//
// Measured rather than inferred from `TdsBackendOptions`, because
// `OpenFile()` (filesystem rejects `O_DIRECT` at open) and `ExecuteTdsIo()`
// (a `libtdsul` read/write returns `EINVAL`) can both fall back to buffered
// I/O.
bool FdHasDirectIo(int fd) {
  if (fd < 0) return false;
  const int flags = ::fcntl(fd, F_GETFL);
  return flags >= 0 && (flags & O_DIRECT) != 0;
}

// Advances a single `tds_buffer_io_t` descriptor by `consumed` bytes after a
// partial transfer.
void AdvanceBufferIo(tds_buffer_io_t* bio, size_t consumed) {
  if (bio->handle != nullptr) {
    bio->registered.offset += consumed;
    bio->registered.size -= consumed;
  } else {
    bio->raw_ptr.vptr = static_cast<uint8_t*>(bio->raw_ptr.vptr) + consumed;
    bio->raw_ptr.size -= consumed;
  }
}

size_t GetBufferIoSize(const tds_buffer_io_t& bio) {
  return (bio.handle != nullptr) ? bio.registered.size : bio.raw_ptr.size;
}

// Executes a storage transfer between `fd` on [local-storage: NVMe SSD] /
// [shared-storage: Lustre/NFS] and `TDS_MEM_HOST` buffer descriptors in
// [dram: User host_buf] using `libtdsul`:
//   - Registers `fd` via `tds_storage_handle_register("fd://<fd>")`.
//   - Dispatches `tds_write` / `tds_read` when a single buffer descriptor is
//     active, or `tds_writev` / `tds_readv` when multiple scatter-gather slices
//     are active.
//   - If a transfer fails with `EINVAL` and the fd has `O_DIRECT`, clears
//     `O_DIRECT` once via `fcntl(F_SETFL)` and retries, only if the flag is
//     verifiably cleared afterwards (e.g. a filesystem that accepts
//     `O_DIRECT` at open but rejects it at I/O time). Callers detect this via
//     `FdHasDirectIo()`.
absl::Status ExecuteTdsIo(int fd, std::vector<tds_buffer_io_t> buffer_ios,
                          int64_t file_offset, bool is_write,
                          size_t min_required_bytes,
                          size_t lustre_stripe_size = 0) {
  if (buffer_ios.empty() && min_required_bytes == 0) {
    return absl::OkStatus();
  }
  std::unique_ptr<tds_storage_opts_t, decltype(&tds_storage_opts_destroy)>
      storage_opts(nullptr, &tds_storage_opts_destroy);
  if (lustre_stripe_size > 0) {
    storage_opts.reset(tds_storage_opts_create());
    if (storage_opts != nullptr) {
      (void)tds_storage_opts_set_int(storage_opts.get(), "stripe_size",
                                     static_cast<int>(lustre_stripe_size));
    }
  }

  std::string fd_uri = absl::StrCat("fd://", fd);
  tds_storage_descr_t storage_descr = {
      .uri = fd_uri.c_str(),
      .options = storage_opts.get(),
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
  bool tried_clearing_odirect = false;

  while (idx < buffer_ios.size()) {
    if (GetBufferIoSize(buffer_ios[idx]) == 0) {
      ++idx;
      continue;
    }
    const int iov_cnt = static_cast<int>(
        std::min<size_t>(buffer_ios.size() - idx, kMaxIovCount));
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
      if (errno == EINVAL && !tried_clearing_odirect) {
        tried_clearing_odirect = true;
        int fl = ::fcntl(fd, F_GETFL);
        if (fl >= 0 && (fl & O_DIRECT)) {
          if (::fcntl(fd, F_SETFL, fl & ~O_DIRECT) == 0 &&
              (::fcntl(fd, F_GETFL) & O_DIRECT) == 0) {
            continue;
          }
        }
      }
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

// An `alignment`-aligned bounce buffer in [dram: User host_buf]
// (`posix_memalign`, zero-filled) covering [aligned_start, aligned_end) of the
// file, registered with `libtdsul` via
// `tds_buffer_register_vaddr(..., TDS_MEM_HOST)`. Used when a transfer does
// not meet `O_DIRECT` alignment; `libtdsul` itself does not bounce
// (`go/tdsul-api` §1).
struct BounceWindow {
  int64_t aligned_start = 0;
  int64_t end_offset = 0;
  int64_t aligned_end = 0;
  size_t aligned_size = 0;
  size_t head_pad = 0;
  std::unique_ptr<uint8_t, decltype(&std::free)> buffer{nullptr, &std::free};
  std::unique_ptr<tds_buffer_handle_t, decltype(&tds_buffer_deregister)> handle{
      nullptr, &tds_buffer_deregister};
};

absl::StatusOr<BounceWindow> AllocateBounceWindow(int64_t file_offset,
                                                  size_t total_bytes,
                                                  size_t alignment) {
  BounceWindow win;
  const int64_t align_i64 = static_cast<int64_t>(alignment);
  win.aligned_start = (file_offset / align_i64) * align_i64;
  win.end_offset = file_offset + static_cast<int64_t>(total_bytes);
  win.aligned_end = ((win.end_offset + align_i64 - 1) / align_i64) * align_i64;
  win.aligned_size = static_cast<size_t>(win.aligned_end - win.aligned_start);
  win.head_pad = static_cast<size_t>(file_offset - win.aligned_start);

  void* raw = nullptr;
  if (::posix_memalign(&raw, alignment, win.aligned_size) != 0 ||
      raw == nullptr) {
    return absl::ResourceExhaustedError(
        "Failed to allocate aligned bounce buffer in [dram: User host_buf]");
  }
  std::memset(raw, 0, win.aligned_size);
  win.buffer.reset(static_cast<uint8_t*>(raw));

  // Register the bounce buffer with `libtdsul` as `TDS_MEM_HOST` so bounce
  // reads/writes also flow through registered `tds_buffer_handle_t` handles.
  tds_buffer_handle_t* raw_handle = nullptr;
  if (tds_buffer_register_vaddr(win.buffer.get(), TDS_MEM_HOST,
                                win.aligned_size, &raw_handle) != TDS_SUCCESS ||
      raw_handle == nullptr) {
    return absl::InternalError(
        "tds_buffer_register_vaddr(TDS_MEM_HOST) failed for bounce buffer");
  }
  win.handle.reset(raw_handle);
  return win;
}

}  // namespace

// --- TdsBackendOptions Implementation ---

absl::StatusOr<TdsBackendOptions> TdsBackendOptions::FromProperties(
    const absl::flat_hash_map<std::string, std::string>& properties) {
  TdsBackendOptions options;

  auto parse_int = [&](absl::string_view key, auto* out) -> absl::Status {
    auto it = properties.find(key);
    if (it == properties.end()) return absl::OkStatus();
    if (!absl::SimpleAtoi(it->second, out)) {
      return absl::InvalidArgumentError(
          absl::StrCat(key, " is not a valid integer: ", it->second));
    }
    return absl::OkStatus();
  };

  auto parse_bool = [&](absl::string_view key, bool* out) -> absl::Status {
    auto it = properties.find(key);
    if (it == properties.end()) return absl::OkStatus();
    if (it->second == "true" || it->second == "1") {
      *out = true;
    } else if (it->second == "false" || it->second == "0") {
      *out = false;
    } else if (!absl::SimpleAtob(it->second, out)) {
      return absl::InvalidArgumentError(
          absl::StrCat(key, " must be a valid boolean"));
    }
    return absl::OkStatus();
  };

  if (auto it = properties.find("root_dir"); it != properties.end()) {
    if (it->second.empty()) {
      return absl::InvalidArgumentError("root_dir cannot be empty");
    }
    options.root_dir = it->second;
  }
  if (auto it = properties.find("model_name");
      it != properties.end() && !it->second.empty()) {
    options.model_name = it->second;
  }

  ABSL_RETURN_IF_ERROR(parse_int("tp_size", &options.tp_size));
  if (options.tp_size <= 0) {
    return absl::InvalidArgumentError("tp_size must be a positive integer");
  }
  ABSL_RETURN_IF_ERROR(parse_int("tp_rank", &options.tp_rank));
  if (options.tp_rank < 0 || options.tp_rank >= options.tp_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("tp_rank must be in [0, ", options.tp_size, ")"));
  }
  ABSL_RETURN_IF_ERROR(parse_int("capacity_bytes", &options.capacity_bytes));
  ABSL_RETURN_IF_ERROR(
      parse_int("lookup_batch_size", &options.lookup_batch_size));
  if (options.lookup_batch_size == 0) {
    return absl::InvalidArgumentError(
        "lookup_batch_size must be a positive integer");
  }
  ABSL_RETURN_IF_ERROR(parse_int("storage_io_thread_pool_size",
                                 &options.storage_io_thread_pool_size));
  if (options.storage_io_thread_pool_size <= 0) {
    return absl::InvalidArgumentError(
        "storage_io_thread_pool_size must be a positive integer");
  }
  ABSL_RETURN_IF_ERROR(parse_int("alignment", &options.alignment));
  if (options.alignment == 0 ||
      (options.alignment & (options.alignment - 1)) != 0) {
    return absl::InvalidArgumentError(
        "alignment must be a positive power of 2");
  }
  ABSL_RETURN_IF_ERROR(parse_bool("use_direct_io", &options.use_direct_io));
  ABSL_RETURN_IF_ERROR(
      parse_bool("require_registration", &options.require_registration));
  ABSL_RETURN_IF_ERROR(
      parse_int("lustre_stripe_size", &options.lustre_stripe_size));
  ABSL_RETURN_IF_ERROR(
      parse_int("lustre_stripe_count", &options.lustre_stripe_count));
  if (options.lustre_stripe_count < 0) {
    return absl::InvalidArgumentError(
        "lustre_stripe_count must be a non-negative integer");
  }

  return options;
}

// --- TdsKVBackend Implementation ---

namespace {

TdsBackendOptions ParseOptionsOrDefault(
    absl::string_view name,
    const absl::flat_hash_map<std::string, std::string>& properties) {
  absl::StatusOr<TdsBackendOptions> parsed =
      TdsBackendOptions::FromProperties(properties);
  if (!parsed.ok()) {
    LOG(WARNING) << "[TdsKVBackend] Invalid configuration for " << name << ": "
                 << parsed.status() << "; falling back to default options.";
    return TdsBackendOptions{};
  }
  return *std::move(parsed);
}

}  // namespace

TdsKVBackend::TdsKVBackend(
    std::string name, absl::flat_hash_map<std::string, std::string> properties)
    : TdsKVBackend(name, ParseOptionsOrDefault(name, properties), properties) {}

TdsKVBackend::TdsKVBackend(
    std::string name, const TdsBackendOptions& options,
    absl::flat_hash_map<std::string, std::string> properties)
    : KVBackend(std::move(properties)),
      name_(std::move(name)),
      options_(options),
      thread_pool_(std::make_unique<NumaThreadPool>(
          std::max(1, options_.storage_io_thread_pool_size))) {
  EnsureTdsulInitialized(options_.storage_io_thread_pool_size);
  mapper_ =
      std::make_shared<PosixPathMapper>(options_.root_dir, options_.model_name,
                                        options_.tp_size, options_.tp_rank);
}

TdsKVBackend::~TdsKVBackend() {
  thread_pool_.reset();
  absl::MutexLock lock(mu_);
  registrations_.clear();
}

absl::Status TdsKVBackend::CheckNoAddressOverlapLocked(uintptr_t addr,
                                                       size_t size) const {
  auto it = registrations_.upper_bound(addr);
  if (it != registrations_.end() && addr + size > it->first) {
    return absl::AlreadyExistsError(
        "Buffer overlaps with existing registration");
  }
  if (it != registrations_.begin()) {
    auto prev = std::prev(it);
    if (prev->first + prev->second.size > addr) {
      return absl::AlreadyExistsError(
          "Buffer overlaps with existing registration");
    }
  }
  return absl::OkStatus();
}

absl::Status TdsKVBackend::RegisterBuffer(void* ptr, size_t size) {
  if (ptr == nullptr || size == 0) {
    return absl::InvalidArgumentError(
        "Buffer pointer must be non-null and size must be positive");
  }
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  absl::MutexLock lock(mu_);
  ABSL_RETURN_IF_ERROR(CheckNoAddressOverlapLocked(addr, size));

  tds_buffer_handle_t* raw_handle = nullptr;
  tds_result_t res =
      tds_buffer_register_vaddr(ptr, TDS_MEM_HOST, size, &raw_handle);
  if (res != TDS_SUCCESS || raw_handle == nullptr) {
    return absl::InternalError(absl::StrCat(
        "tds_buffer_register_vaddr(TDS_MEM_HOST) failed with code ",
        static_cast<int>(res)));
  }
  RegisteredBuffer reg;
  reg.size = size;
  reg.handle =
      std::shared_ptr<tds_buffer_handle_t>(raw_handle, &tds_buffer_deregister);
  registrations_[addr] = std::move(reg);
  return absl::OkStatus();
}

absl::StatusOr<void*> TdsKVBackend::RegisterDmaBuf(int dma_buf_fd,
                                                   size_t offset, size_t size) {
  if (dma_buf_fd < 0 || size == 0) {
    return absl::InvalidArgumentError(
        "dma_buf_fd must be non-negative and size must be positive");
  }
  if (options_.alignment > 0 && (offset % options_.alignment) != 0) {
    return absl::InvalidArgumentError(absl::StrCat("dma_buf offset (", offset,
                                                   ") must be aligned to ",
                                                   options_.alignment));
  }
  absl::MutexLock lock(mu_);
  for (const auto& [base_addr, reg] : registrations_) {
    if (reg.dma_buf_fd == dma_buf_fd) {
      return absl::AlreadyExistsError(
          absl::StrCat("dma_buf_fd ", dma_buf_fd, " is already registered"));
    }
  }
  tds_buffer_handle_t* raw_handle = nullptr;
  tds_result_t res = tds_buffer_register_dmabuf(
      dma_buf_fd, offset, TDS_MEM_HOST, size, &raw_handle);
  if (res != TDS_SUCCESS || raw_handle == nullptr ||
      raw_handle->vaddr == nullptr) {
    return absl::InternalError(absl::StrCat(
        "tds_buffer_register_dmabuf(TDS_MEM_HOST) failed with code ",
        static_cast<int>(res)));
  }

  void* mapped_vaddr = raw_handle->vaddr;
  uintptr_t addr = reinterpret_cast<uintptr_t>(mapped_vaddr);
  RegisteredBuffer reg;
  reg.size = size;
  reg.handle =
      std::shared_ptr<tds_buffer_handle_t>(raw_handle, &tds_buffer_deregister);
  reg.dma_buf_fd = dma_buf_fd;
  reg.dma_buf_base_offset = offset;
  ABSL_RETURN_IF_ERROR(CheckNoAddressOverlapLocked(addr, size));
  registrations_[addr] = std::move(reg);
  return mapped_vaddr;
}

absl::Status TdsKVBackend::UnregisterDmaBuf(int dma_buf_fd) {
  if (dma_buf_fd < 0) {
    return absl::InvalidArgumentError("dma_buf_fd must be non-negative");
  }
  absl::MutexLock lock(mu_);
  for (auto it = registrations_.begin(); it != registrations_.end(); ++it) {
    if (it->second.dma_buf_fd == dma_buf_fd) {
      registrations_.erase(it);
      return absl::OkStatus();
    }
  }
  return absl::NotFoundError(
      absl::StrCat("dma_buf_fd ", dma_buf_fd, " not found in registrations"));
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

void* TdsKVBackend::ResolveSlicePtrLocked(
    const HostBufferDescriptor& slice) const {
  if (slice.ptr != nullptr) {
    return slice.ptr;
  }
  if (slice.fd >= 0 && slice.offset >= 0) {
    const size_t slice_offset = static_cast<size_t>(slice.offset);
    for (const auto& [base_addr, reg] : registrations_) {
      if (reg.dma_buf_fd == slice.fd &&
          slice_offset >= reg.dma_buf_base_offset &&
          (slice_offset - reg.dma_buf_base_offset) + slice.size <= reg.size) {
        return reinterpret_cast<void*>(
            base_addr + (slice_offset - reg.dma_buf_base_offset));
      }
    }
  }
  return nullptr;
}

std::shared_ptr<tds_buffer_handle_t> TdsKVBackend::LookupRegisteredBufferLocked(
    const void* ptr, size_t size, size_t* offset_in_reg_out) const {
  if (ptr == nullptr) return nullptr;
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  auto it = registrations_.upper_bound(addr);
  if (it == registrations_.begin()) {
    return nullptr;
  }
  --it;
  if (addr >= it->first && (addr + size) <= (it->first + it->second.size)) {
    if (offset_in_reg_out != nullptr) {
      *offset_in_reg_out = addr - it->first;
    }
    return it->second.handle;
  }
  return nullptr;
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
  return LookupRegisteredBufferLocked(ptr, size) != nullptr;
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

absl::StatusOr<TdsKVBackend::PreparedTransfer> TdsKVBackend::PrepareTransfer(
    const BlockKey& key, absl::Span<const HostBufferDescriptor> slices,
    size_t total_bytes) const {
  if (key.offset < 0) {
    return absl::InvalidArgumentError("Negative key offset");
  }
  PreparedTransfer plan;
  plan.resolved_slices.reserve(slices.size());
  plan.buffer_ios.reserve(slices.size());

  const size_t align = options_.alignment;
  bool all_aligned =
      (align > 0) && (static_cast<size_t>(key.offset) % align == 0);
  bool all_registered = true;
  size_t sum_bytes = 0;

  {
    absl::MutexLock lock(mu_);
    for (const auto& slice : slices) {
      sum_bytes += slice.size;
      if (slice.size == 0) continue;
      void* resolved_ptr = ResolveSlicePtrLocked(slice);
      if (resolved_ptr == nullptr) {
        return absl::InvalidArgumentError(
            "Null slice pointer (and unresolved dma_buf_fd) with non-zero "
            "size");
      }
      plan.resolved_slices.push_back(
          ResolvedSlice{.ptr = resolved_ptr, .size = slice.size});

      if (align == 0 ||
          reinterpret_cast<uintptr_t>(resolved_ptr) % align != 0 ||
          slice.size % align != 0) {
        all_aligned = false;
      }

      size_t offset_in_reg = 0;
      std::shared_ptr<tds_buffer_handle_t> handle =
          LookupRegisteredBufferLocked(resolved_ptr, slice.size,
                                       &offset_in_reg);
      if (handle != nullptr) {
        plan.kept_handles.push_back(handle);
        plan.buffer_ios.push_back(tds_create_registered_buffer_io(
            handle.get(), offset_in_reg, slice.size));
      } else {
        all_registered = false;
        plan.buffer_ios.push_back(
            tds_create_raw_buffer_io(resolved_ptr, slice.size));
      }
    }
  }

  if (sum_bytes != total_bytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("total_bytes (", total_bytes,
                     ") does not match sum of slice sizes (", sum_bytes, ")"));
  }

  ABSL_ASSIGN_OR_RETURN(plan.path, ResolvePath(key));

  const bool reg_ok = !options_.require_registration || all_registered;
  // Under `O_DIRECT`, the caller's slices can be passed to `libtdsul` as-is
  // only if the file offset and every slice address/size are aligned (and, if
  // `require_registration`, every slice is registered). Buffered mode has no
  // alignment rule, so only the registration requirement applies.
  const bool is_direct_zero_copy =
      (total_bytes > 0) && options_.use_direct_io && all_aligned && reg_ok;
  plan.can_bypass_bounce = (total_bytes == 0) || is_direct_zero_copy ||
                           (!options_.use_direct_io && reg_ok);
  return plan;
}

void TdsKVBackend::RecordSuccess(bool direct, bool o_direct, size_t bytes) {
  absl::MutexLock lock(mu_);
  stats_.ops++;
  if (direct) {
    stats_.direct_ops++;
  } else {
    stats_.bounced_ops++;
  }
  // Independent of the bounce dimension above: did the storage I/O actually
  // bypass [dram: OS Page Cache]?
  if (o_direct) {
    stats_.o_direct_ops++;
  }
  stats_.bytes += static_cast<int64_t>(bytes);
}

absl::Status TdsKVBackend::ExecuteWrite(
    const BlockKey& key, absl::Span<const HostBufferDescriptor> slices,
    size_t total_bytes) {
  ABSL_ASSIGN_OR_RETURN(PreparedTransfer plan,
                        PrepareTransfer(key, slices, total_bytes));
  const std::string& path = plan.path;

  // When `key.offset == 0` and we are writing the complete file (the standard
  // whole-block write case), write to a sibling `.tmp` file and atomically
  // `::rename()` on completion so concurrent `BatchExistsAsync` lookups on
  // [shared-storage: Lustre/NFS] never observe a partial file.
  struct stat existing_st = {};
  const bool file_exists = (::stat(path.c_str(), &existing_st) == 0);
  const int64_t initial_file_size = file_exists ? existing_st.st_size : 0;
  const bool use_atomic_tmp_publish =
      (key.offset == 0) &&
      (static_cast<int64_t>(total_bytes) >= initial_file_size);

  const std::string write_path =
      use_atomic_tmp_publish ? MakeUniqueTempPath(path) : path;
  bool publish_succeeded = false;
  absl::Cleanup cleanup_tmp = [&]() {
    if (use_atomic_tmp_publish && !publish_succeeded) {
      ::unlink(write_path.c_str());
    }
  };

  auto finalize_publish = [&](int* fd_ptr) -> absl::Status {
    const int fd_to_close = *fd_ptr;
    *fd_ptr = -1;
    // On [shared-storage: Lustre/NFS], ::close(fd) flushes client state and can
    // surface deferred write/quota errors (EIO, ENOSPC, EDQUOT). Verify ::close
    // succeeds before atomically renaming `write_path` into `path`.
    if (fd_to_close >= 0 && ::close(fd_to_close) < 0) {
      return absl::ErrnoToStatus(
          errno, absl::StrCat("close failed before publish for ", write_path));
    }
    if (use_atomic_tmp_publish) {
      if (::rename(write_path.c_str(), path.c_str()) < 0) {
        return absl::ErrnoToStatus(
            errno, absl::StrCat("Atomic rename failed from ", write_path,
                                " to ", path));
      }
      publish_succeeded = true;
    }
    return absl::OkStatus();
  };

  const int open_flags =
      (plan.can_bypass_bounce ? (O_WRONLY | O_CREAT) : (O_RDWR | O_CREAT)) |
      (use_atomic_tmp_publish ? O_TRUNC : 0);
  ABSL_ASSIGN_OR_RETURN(int fd, OpenFile(write_path, open_flags, 0644,
                                         options_.use_direct_io, options_));
  absl::Cleanup close_fd = [&fd]() {
    if (fd >= 0) ::close(fd);
  };

  if (plan.can_bypass_bounce) {
    // Bounce-free path: `tds_write` / `tds_writev` directly from the caller's
    // [dram: User host_buf] slices. With `O_DIRECT` on the fd, the kernel
    // writes those slices straight to [local-storage: NVMe SSD] /
    // [shared-storage: Lustre/NFS]; without it (`use_direct_io == false` or
    // the filesystem rejected `O_DIRECT`), the kernel copies them into
    // [dram: OS Page Cache] first. Either way no `BounceWindow` is used, so
    // this is recorded as `direct = true`.
    ABSL_RETURN_IF_ERROR(ExecuteTdsIo(
        fd, std::move(plan.buffer_ios), key.offset, /*is_write=*/true,
        /*min_required_bytes=*/total_bytes, options_.lustre_stripe_size));
    // Sampled before `finalize_publish()` closes the fd so an `O_DIRECT` strip
    // done by `ExecuteTdsIo()` shows up in `o_direct_ops`.
    const bool o_direct = FdHasDirectIo(fd);
    ABSL_RETURN_IF_ERROR(finalize_publish(&fd));
    RecordSuccess(/*direct=*/true, o_direct, total_bytes);
    return absl::OkStatus();
  }

  // Bounce path: an `alignment`-aligned `BounceWindow` in [dram: User host_buf]
  // (registered as `TDS_MEM_HOST`) covers [aligned_start, aligned_end).

  ABSL_ASSIGN_OR_RETURN(
      BounceWindow win,
      AllocateBounceWindow(key.offset, total_bytes, options_.alignment));

  // Read-modify-write: when updating an existing file in place and the window
  // has head/tail padding that overlaps existing bytes, pre-read those bytes
  // via `tds_read` so the aligned write below does not overwrite them with
  // zeros. Bytes past the current EOF stay zero from `AllocateBounceWindow`.
  if (!use_atomic_tmp_publish && initial_file_size > win.aligned_start &&
      (win.head_pad > 0 || win.end_offset < win.aligned_end)) {
    const size_t existing_in_window = static_cast<size_t>(
        std::min<int64_t>(win.aligned_end, initial_file_size) -
        win.aligned_start);
    tds_buffer_io_t pre_read_bio =
        tds_create_registered_buffer_io(win.handle.get(), 0, win.aligned_size);
    ABSL_RETURN_IF_ERROR(ExecuteTdsIo(fd, {pre_read_bio}, win.aligned_start,
                                      /*is_write=*/false, existing_in_window,
                                      options_.lustre_stripe_size));
  }

  size_t offset_in_slices = 0;
  for (const auto& slice : plan.resolved_slices) {
    std::memcpy(win.buffer.get() + win.head_pad + offset_in_slices, slice.ptr,
                slice.size);
    offset_in_slices += slice.size;
  }

  tds_buffer_io_t write_bio =
      tds_create_registered_buffer_io(win.handle.get(), 0, win.aligned_size);
  ABSL_RETURN_IF_ERROR(ExecuteTdsIo(fd, {write_bio}, win.aligned_start,
                                    /*is_write=*/true, win.aligned_size,
                                    options_.lustre_stripe_size));
  // The bounce write uses the same fd as the fast path (carrying `O_DIRECT`
  // only when `use_direct_io` and the filesystem accepted it), so the
  // page-cache dimension is sampled the same way.
  const bool o_direct = FdHasDirectIo(fd);

  const int64_t target_file_size =
      use_atomic_tmp_publish
          ? win.end_offset
          : std::max<int64_t>(initial_file_size, win.end_offset);
  if (target_file_size < win.aligned_end) {
    if (::ftruncate(fd, target_file_size) < 0) {
      return absl::ErrnoToStatus(errno, "ftruncate failed after bounce write");
    }
  }

  ABSL_RETURN_IF_ERROR(finalize_publish(&fd));
  RecordSuccess(/*direct=*/false, o_direct, total_bytes);
  return absl::OkStatus();
}

absl::Status TdsKVBackend::ExecuteRead(
    const BlockKey& key, absl::Span<const HostBufferDescriptor> slices,
    size_t total_bytes) {
  ABSL_ASSIGN_OR_RETURN(PreparedTransfer plan,
                        PrepareTransfer(key, slices, total_bytes));
  const std::string& path = plan.path;

  ABSL_ASSIGN_OR_RETURN(
      int fd, OpenFile(path, O_RDONLY, 0, options_.use_direct_io, options_));
  absl::Cleanup close_fd = [&fd]() {
    if (fd >= 0) ::close(fd);
  };

  auto close_and_verify = [&]() -> absl::Status {
    const int fd_to_close = fd;
    fd = -1;
    if (fd_to_close >= 0 && ::close(fd_to_close) < 0) {
      return absl::ErrnoToStatus(
          errno, absl::StrCat("Failed to close file after read: ", path));
    }
    return absl::OkStatus();
  };

  if (plan.can_bypass_bounce) {
    // Bounce-free path: `tds_read` / `tds_readv` directly into the caller's
    // [dram: User host_buf] slices. With `O_DIRECT` on the fd, the kernel reads
    // [local-storage: NVMe SSD] / [shared-storage: Lustre/NFS] straight into
    // those slices; without it, data is first staged in
    // [dram: OS Page Cache] and copied by the kernel.
    ABSL_RETURN_IF_ERROR(ExecuteTdsIo(
        fd, std::move(plan.buffer_ios), key.offset, /*is_write=*/false,
        /*min_required_bytes=*/total_bytes, options_.lustre_stripe_size));
    // Sampled before `close_and_verify()` so an `O_DIRECT` strip done by
    // `ExecuteTdsIo()` shows up in `o_direct_ops`.
    const bool o_direct = FdHasDirectIo(fd);
    ABSL_RETURN_IF_ERROR(close_and_verify());
    RecordSuccess(/*direct=*/true, o_direct, total_bytes);
    return absl::OkStatus();
  }

  // Bounce path: read the enclosing `alignment`-aligned window into a
  // `BounceWindow` (registered as `TDS_MEM_HOST`), then memcpy the requested
  // bytes into the caller's slices.
  ABSL_ASSIGN_OR_RETURN(
      BounceWindow win,
      AllocateBounceWindow(key.offset, total_bytes, options_.alignment));
  const size_t required_bytes = win.head_pad + total_bytes;

  tds_buffer_io_t read_bio =
      tds_create_registered_buffer_io(win.handle.get(), 0, win.aligned_size);
  ABSL_RETURN_IF_ERROR(ExecuteTdsIo(fd, {read_bio}, win.aligned_start,
                                    /*is_write=*/false, required_bytes,
                                    options_.lustre_stripe_size));
  const bool o_direct = FdHasDirectIo(fd);
  ABSL_RETURN_IF_ERROR(close_and_verify());

  size_t offset_in_slices = 0;
  for (const auto& slice : plan.resolved_slices) {
    std::memcpy(slice.ptr, win.buffer.get() + win.head_pad + offset_in_slices,
                slice.size);
    offset_in_slices += slice.size;
  }

  RecordSuccess(/*direct=*/false, o_direct, total_bytes);
  return absl::OkStatus();
}

void TdsKVBackend::ScheduleTransfer(
    const BlockKey& key, absl::Span<const HostBufferDescriptor> slices,
    size_t total_bytes, bool is_write,
    std::function<void(absl::Status)> callback) {
  std::vector<HostBufferDescriptor> owned_slices(slices.begin(), slices.end());
  thread_pool_->Schedule(
      std::nullopt, [this, key, owned_slices = std::move(owned_slices),
                     total_bytes, is_write, callback = std::move(callback)]() {
        absl::Status status = is_write
                                  ? ExecuteWrite(key, owned_slices, total_bytes)
                                  : ExecuteRead(key, owned_slices, total_bytes);
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

void TdsKVBackend::WriteAsync(const BlockKey& key,
                              absl::Span<const HostBufferDescriptor> slices,
                              size_t total_bytes,
                              std::function<void(absl::Status)> callback) {
  ScheduleTransfer(key, slices, total_bytes, /*is_write=*/true,
                   std::move(callback));
}

void TdsKVBackend::ReadAsync(const BlockKey& key,
                             absl::Span<const HostBufferDescriptor> slices,
                             size_t total_bytes,
                             std::function<void(absl::Status)> callback) {
  ScheduleTransfer(key, slices, total_bytes, /*is_write=*/false,
                   std::move(callback));
}

absl::StatusOr<bool> TdsKVBackend::Exists(const BlockKey& key) const {
  if (key.offset < 0 || key.size < 0) {
    return absl::InvalidArgumentError("Negative offset or size in BlockKey");
  }
  ABSL_ASSIGN_OR_RETURN(std::string path, ResolvePath(key));
  struct stat st = {};
  if (::stat(path.c_str(), &st) == 0) {
    return S_ISREG(st.st_mode) && (st.st_size >= key.offset + key.size);
  }
  if (errno == ENOENT || errno == ENOTDIR) {
    return false;
  }
  return absl::ErrnoToStatus(errno, absl::StrCat("stat failed for ", path));
}

void TdsKVBackend::BatchExistsAsync(
    absl::Span<const BlockKey> keys,
    std::function<void(std::vector<absl::StatusOr<bool>>)> callback) {
  if (keys.empty()) {
    thread_pool_->Schedule(std::nullopt, [callback = std::move(callback)]() {
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
    thread_pool_->Schedule(std::nullopt, [this, i, key = keys[i], state]() {
      state->results[i] = Exists(key);
      if (state->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1 &&
          state->callback) {
        state->callback(std::move(state->results));
      }
    });
  }
}

}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

using ::tpu_raiden::kv_cache::backends::storage::PosixKVCacheStoreBackend;
using ::tpu_raiden::kv_cache::backends::storage::TdsBackendOptions;
using ::tpu_raiden::kv_cache::backends::storage::TdsKVBackend;

REGISTER_KV_CACHE_STORE_BACKEND(
    ::tpu_raiden::kv_cache::backends::storage::kTdsBackendName,
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
      // via PosixKVCacheStoreBackend::Lookup, so its mapper is pinned to
      // rank 0. Per-worker tp_rank lives on each worker's own config.
      auto properties = config.properties;
      properties["tp_size"] = absl::StrCat(tp_size);
      properties["tp_rank"] = "0";

      const std::string backend_name = std::string(
          ::tpu_raiden::kv_cache::backends::storage::kTdsBackendName);
      ABSL_ASSIGN_OR_RETURN(TdsBackendOptions options,
                            TdsBackendOptions::FromProperties(properties));
      auto backend =
          std::make_shared<TdsKVBackend>(backend_name, options, properties);
      return std::make_shared<PosixKVCacheStoreBackend>(
          std::move(backend), backend_name, options.capacity_bytes,
          options.lookup_batch_size);
    });
