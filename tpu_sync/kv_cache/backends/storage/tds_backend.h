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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_TDS_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_TDS_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tdsul/tdsul.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/backends/storage/posix_backend.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace storage {

inline constexpr absl::string_view kTdsBackendName = "tds";

// Configuration options for TdsKVBackend (backed by `libtdsul` in
// `//tpu_sync/tpudirect_storage:tdsul`).
//
// Controls how KV-cache blocks transfer between:
//   - Storage Tier: [local-storage: NVMe SSD] or [shared-storage: Lustre/NFS]
//   - Host Tier:    [dram: User host_buf] mapped to `TDS_MEM_HOST` via
//                   `tds_buffer_register_vaddr` / `tds_buffer_register_dmabuf`
//
// When `use_direct_io == true`, the storage file is opened with `O_DIRECT` so
// the kernel does not stage data in [dram: OS Page Cache].
struct TdsBackendOptions {
  std::string root_dir = "/tmp/raiden_storage";
  std::string model_name = "unknown";
  int tp_size = 1;
  int tp_rank = 0;
  size_t capacity_bytes = 0;
  size_t lookup_batch_size = kDefaultLookupBatchSize;
  int storage_io_thread_pool_size = 16;
  // `O_DIRECT` alignment (bytes) for the file offset, slice address and slice
  // size.
  size_t alignment = 4096;
  bool use_direct_io = true;
  // When false (default), a transfer whose `key.offset` and slices (address and
  // size) in [dram: User host_buf] are all `alignment`-aligned runs directly on
  // the caller's slices via `tds_read`/`tds_write` (`tds_readv`/`tds_writev`),
  // with no bounce buffer. Each slice uses a registered `tds_buffer_handle_t*`
  // (`TDS_MEM_HOST`) if covered by `RegisterBuffer()`/`RegisterDmaBuf()`, or a
  // raw `tds_buffer_io_t` otherwise. When true, slices must additionally reside
  // inside a registered region to skip the bounce buffer.
  bool require_registration = false;
  size_t lustre_stripe_size = 0;
  int lustre_stripe_count = 0;

  static absl::StatusOr<TdsBackendOptions> FromProperties(
      const absl::flat_hash_map<std::string, std::string>& properties);
};

// Telemetry and I/O statistics for TdsKVBackend.
struct TdsStats {
  int64_t ops = 0;
  int64_t bytes = 0;
  // Successful operations that ran straight on the caller's
  // [dram: User host_buf] slices, with no `BounceWindow` and no memcpy.
  //
  // NOTE: `direct_ops` counts only the bounce dimension. It does NOT say
  // whether [dram: OS Page Cache] was bypassed; see `o_direct_ops`. A transfer
  // can be bounce-free and still be buffered by the kernel (e.g.
  // `use_direct_io == false`, or a filesystem that rejects `O_DIRECT`).
  int64_t direct_ops = 0;
  // Successful operations that went through an `alignment`-aligned
  // `BounceWindow` in [dram: User host_buf] (registered with
  // `tds_buffer_register_vaddr(..., TDS_MEM_HOST)`) plus a memcpy, because the
  // file offset or a slice address/size was not aligned (or a slice was
  // unregistered while `require_registration == true`).
  int64_t bounced_ops = 0;
  int64_t failed_ops = 0;
  // Successful operations whose storage fd still carried `O_DIRECT`
  // (`fcntl(fd, F_GETFL)`) after the `libtdsul` transfer, i.e. the kernel did
  // not stage the data in [dram: OS Page Cache].
  //
  // Measured rather than assumed: `OpenFile()` falls back to a buffered open
  // when the filesystem rejects `O_DIRECT`, and `ExecuteTdsIo()` clears
  // `O_DIRECT` on the fd once when a `libtdsul` read/write returns `EINVAL`.
  // In both cases the data then goes through [dram: OS Page Cache] and a kernel
  // memcpy. When `use_direct_io == true`, `o_direct_ops < ops - failed_ops`
  // means one of these fallbacks happened.
  int64_t o_direct_ops = 0;
};

// TdsKVBackend implements TPU Raiden's KVBackend interface on top of
// `libtdsul` (`//tpu_sync/tpudirect_storage:tdsul`) for
// transfers between [local-storage: NVMe SSD] / [shared-storage: Lustre/NFS]
// and [dram: User host_buf]. Raiden's H2D/D2H path moves data between
// [dram: User host_buf] and [hbm] separately.
//
//   1. `libtdsul` Host Buffer Registration (`TDS_MEM_HOST`):
//      - `RegisterBuffer(ptr, size)` calls
//        `tds_buffer_register_vaddr(ptr, TDS_MEM_HOST, size, &handle)`.
//      - `RegisterDmaBuf(dma_buf_fd, offset, size)` calls
//        `tds_buffer_register_dmabuf(dma_buf_fd, offset, TDS_MEM_HOST, size,
//        &handle)`.
//
//   2. `libtdsul` Storage Handle & Synchronous I/O:
//      Opens the file itself (`O_DIRECT` when `use_direct_io`, plus optional
//      Lustre `LL_IOC_LOV_SETSTRIPE`), wraps the fd via
//      `tds_storage_handle_register("fd://<fd>")`, and transfers a single
//      slice with `tds_read`/`tds_write` or several slices with
//      `tds_readv`/`tds_writev`.
//
//   3. Bounce Buffer & Atomic Whole-Block Publishing:
//      When `O_DIRECT` alignment is not met, the transfer goes through an
//      `alignment`-aligned `BounceWindow` in [dram: User host_buf] registered
//      via `tds_buffer_register_vaddr(..., TDS_MEM_HOST)`. `libtdsul` does not
//      bounce internally (`go/tdsul-api` §1), so this is done here. Writes
//      that cover the whole file (offset 0, size >= current file size) go to
//      `<path>.tmp_<pid>_<seq>` and are `::rename()`d over `<path>` after a
//      successful `::close()`.
class TdsKVBackend : public KVBackend {
 public:
  explicit TdsKVBackend(
      std::string name,
      absl::flat_hash_map<std::string, std::string> properties = {});

  TdsKVBackend(std::string name, const TdsBackendOptions& options,
               absl::flat_hash_map<std::string, std::string> properties = {});

  ~TdsKVBackend() override;

  std::string name() const override { return name_; }

  void WriteAsync(const BlockKey& key,
                  absl::Span<const HostBufferDescriptor> slices,
                  size_t total_bytes,
                  std::function<void(absl::Status)> callback) override;

  void ReadAsync(const BlockKey& key,
                 absl::Span<const HostBufferDescriptor> slices,
                 size_t total_bytes,
                 std::function<void(absl::Status)> callback) override;

  void BatchExistsAsync(
      absl::Span<const BlockKey> keys,
      std::function<void(std::vector<absl::StatusOr<bool>>)> callback) override;

  // Registers a [dram: User host_buf] virtual address range with `libtdsul` via
  // `tds_buffer_register_vaddr(ptr, TDS_MEM_HOST, size, &handle)`.
  absl::Status RegisterBuffer(void* ptr, size_t size);

  // Registers a Linux `dma_buf_fd` region with `libtdsul` via
  // `tds_buffer_register_dmabuf(dma_buf_fd, offset, TDS_MEM_HOST, size,
  // &handle)` and returns the mapped virtual address in [dram: User host_buf].
  // Once registered, callers may pass slices either by
  // `HostBufferDescriptor::ptr` or by `HostBufferDescriptor{.ptr = nullptr,
  // .size = ..., .fd = dma_buf_fd, .offset = ...}`.
  absl::StatusOr<void*> RegisterDmaBuf(int dma_buf_fd, size_t offset,
                                       size_t size);

  // Unregisters a previously registered `dma_buf_fd` region via
  // `tds_buffer_deregister(handle)`.
  absl::Status UnregisterDmaBuf(int dma_buf_fd);

  // Unregisters a previously registered [dram: User host_buf] region via
  // `tds_buffer_deregister(handle)`.
  absl::Status UnregisterBuffer(void* ptr);

  // Returns true if `ptr`, `size` and `file_offset` are all multiples of
  // `options().alignment` (default 4096) and [ptr, ptr + size) in
  // [dram: User host_buf] lies inside a region registered via
  // `RegisterBuffer()` or `RegisterDmaBuf()`.
  bool IsDmaAlignedAndRegistered(const void* ptr, size_t size,
                                 int64_t file_offset) const;

  TdsStats stats() const;

  void ResetStats();

  size_t alignment() const { return options_.alignment; }

  const TdsBackendOptions& options() const { return options_; }

 private:
  struct RegisteredBuffer {
    size_t size = 0;
    std::shared_ptr<tds_buffer_handle_t> handle;
    int dma_buf_fd = -1;
    size_t dma_buf_base_offset = 0;
  };

  struct ResolvedSlice {
    void* ptr = nullptr;
    size_t size = 0;
  };

  struct PreparedTransfer {
    std::string path;
    std::vector<ResolvedSlice> resolved_slices;
    std::vector<tds_buffer_io_t> buffer_ios;
    std::vector<std::shared_ptr<tds_buffer_handle_t>> kept_handles;
    // True when the transfer can run straight on the caller's [dram: User
    // host_buf] slices with no `BounceWindow` scratch copy. This is the
    // `direct_ops` / `bounced_ops` dimension; whether [dram: OS Page Cache] is
    // bypassed is measured independently (`TdsStats::o_direct_ops`).
    bool can_bypass_bounce = false;
  };

  void* ResolveSlicePtrLocked(const HostBufferDescriptor& slice) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  absl::Status CheckNoAddressOverlapLocked(uintptr_t addr, size_t size) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  std::shared_ptr<tds_buffer_handle_t> LookupRegisteredBufferLocked(
      const void* ptr, size_t size, size_t* offset_in_reg_out = nullptr) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  absl::StatusOr<std::string> ResolvePath(const BlockKey& key) const;

  absl::StatusOr<bool> Exists(const BlockKey& key) const;

  // Records one successful transfer. `direct` is the bounce dimension (no
  // `BounceWindow` scratch copy); `o_direct` is the independently measured
  // page-cache dimension (the fd still carried `O_DIRECT` when the transfer
  // completed).
  void RecordSuccess(bool direct, bool o_direct, size_t bytes);

  absl::StatusOr<PreparedTransfer> PrepareTransfer(
      const BlockKey& key, absl::Span<const HostBufferDescriptor> slices,
      size_t total_bytes) const;

  void ScheduleTransfer(const BlockKey& key,
                        absl::Span<const HostBufferDescriptor> slices,
                        size_t total_bytes, bool is_write,
                        std::function<void(absl::Status)> callback);

  absl::Status ExecuteWrite(const BlockKey& key,
                            absl::Span<const HostBufferDescriptor> slices,
                            size_t total_bytes);

  absl::Status ExecuteRead(const BlockKey& key,
                           absl::Span<const HostBufferDescriptor> slices,
                           size_t total_bytes);

  std::string name_;
  TdsBackendOptions options_;
  mutable absl::Mutex mu_;
  std::map<uintptr_t, RegisteredBuffer> registrations_ ABSL_GUARDED_BY(mu_);
  TdsStats stats_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<NumaThreadPool> thread_pool_;
};

}  // namespace storage
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

namespace tkv {
namespace backends = ::tpu_raiden::kv_cache::backends;
}  // namespace tkv

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_STORAGE_TDS_BACKEND_H_
