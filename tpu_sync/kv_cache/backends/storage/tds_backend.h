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

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/kv_cache/backends/backend.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace storage {

// Configuration options for TdsKVBackend.
struct TdsBackendOptions {
  std::string root_dir = "/tmp/raiden_storage";
  int storage_io_thread_pool_size = 16;
  size_t alignment = 4096;
  bool use_direct_io = true;
  size_t lustre_stripe_size = 0;
  int lustre_stripe_count = 0;

  static absl::StatusOr<TdsBackendOptions> FromProperties(
      const absl::flat_hash_map<std::string, std::string>& properties);
};

// Telemetry and I/O statistics for TdsKVBackend.
struct TdsStats {
  int64_t ops = 0;
  int64_t bytes = 0;
  int64_t direct_ops = 0;
  int64_t bounced_ops = 0;
  int64_t failed_ops = 0;
};

// TdsKVBackend implements KVBackend for high-throughput storage targets
// (Lustre, NFS, NVMe) with zero-copy O_DIRECT DMA for aligned/registered
// buffers and an automatic 4KB bounce-buffer engine for unaligned transfers.
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

  // Registers a host memory region for zero-copy O_DIRECT DMA.
  absl::Status RegisterBuffer(void* ptr, size_t size);

  // Unregisters a previously registered host memory region.
  absl::Status UnregisterBuffer(void* ptr);

  // Returns true if the buffer range [ptr, ptr + size) and file_offset satisfy
  // hardware alignment requirements and reside within a registered memory pool.
  bool IsDmaAlignedAndRegistered(const void* ptr, size_t size,
                                 int64_t file_offset) const;

  TdsStats stats() const;

  void ResetStats();

  size_t alignment() const { return options_.alignment; }

  const TdsBackendOptions& options() const { return options_; }

 private:
  bool IsRegisteredLocked(const void* ptr, size_t size) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  bool CanUseDirectZeroCopy(
      int64_t file_offset, absl::Span<const HostBufferDescriptor> slices) const;

  absl::StatusOr<std::string> ResolvePath(const BlockKey& key) const;

  absl::Status ExecuteWrite(const BlockKey& key,
                            absl::Span<const HostBufferDescriptor> slices,
                            size_t total_bytes);

  absl::Status ExecuteRead(const BlockKey& key,
                           absl::Span<const HostBufferDescriptor> slices,
                           size_t total_bytes);

  std::string name_;
  TdsBackendOptions options_;
  mutable absl::Mutex mu_;
  std::map<uintptr_t, size_t> registrations_ ABSL_GUARDED_BY(mu_);
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
