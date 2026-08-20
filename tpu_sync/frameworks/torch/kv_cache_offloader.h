// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TPU_SYNC_FRAMEWORKS_TORCH_KV_CACHE_OFFLOADER_H_
#define TPU_SYNC_FRAMEWORKS_TORCH_KV_CACHE_OFFLOADER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tpu_sync/core/raw_transfer_core.h"

namespace tpu_raiden::torch {

namespace offloader_internal {

inline constexpr char kPosixErrorPayloadUrl[] =
    "type.googleapis.com/tpu_raiden.kv_cache_offloader.PosixError";

// The operating-system and PJRT boundary used by KVCacheOffloader. Production
// installs direct POSIX/PJRT functions; tests replace them to verify ordering
// and registration lifetime without exposing these seams in the Python API.
struct OffloaderPlatform {
  std::function<absl::StatusOr<size_t>()> page_size;
  std::function<absl::Status(void*, size_t)> dma_map;
  std::function<absl::Status(void*)> dma_unmap;
};

class KVCacheOffloaderTestPeer;

}  // namespace offloader_internal

class KVCacheOffloader final {
 public:
  // Splits each same-sized device buffer into fixed-size raw byte pages.
  KVCacheOffloader(const std::vector<at::Tensor>& kv_cache_tensors,
                   size_t page_nbytes);
  KVCacheOffloader(const KVCacheOffloader&) = delete;
  KVCacheOffloader& operator=(const KVCacheOffloader&) = delete;
  ~KVCacheOffloader();

  absl::Status MapSharedMemory(void* mapped_address, size_t pool_size_bytes);
  absl::Status UnmapSharedMemory();
  bool is_shared_memory_mapped() const;

  absl::StatusOr<raiden::PjRtCopyFuture> H2d(
      const std::vector<int64_t>& block_ids,
      const std::vector<at::Tensor>& object_tensors, int64_t rank_id);
  absl::StatusOr<raiden::PjRtCopyFuture> D2h(
      const std::vector<int64_t>& block_ids,
      const std::vector<at::Tensor>& object_tensors, int64_t rank_id);

  size_t num_layers() const;
  size_t num_blocks() const;
  size_t page_nbytes() const;

 private:
  friend class offloader_internal::KVCacheOffloaderTestPeer;

  struct DeviceState;

  enum class CopyDirection {
    kHostToDevice,
    kDeviceToHost,
  };

  explicit KVCacheOffloader(
      std::shared_ptr<offloader_internal::OffloaderPlatform> platform);
  KVCacheOffloader(
      const std::vector<at::Tensor>& kv_cache_tensors, size_t page_nbytes,
      std::shared_ptr<offloader_internal::OffloaderPlatform> platform);

  static std::shared_ptr<DeviceState> PrepareDeviceState(
      const std::vector<at::Tensor>& kv_cache_tensors, size_t page_nbytes);
  absl::StatusOr<raiden::PjRtCopyFuture> CopyBlocks(
      const std::vector<int64_t>& block_ids,
      const std::vector<at::Tensor>& object_tensors, int64_t rank_id,
      CopyDirection direction);

  enum class MappingPhase {
    kUnmapped,
    kMapping,
    kMapped,
    kUnmapping,
  };

  std::shared_ptr<DeviceState> device_state_;
  std::shared_ptr<offloader_internal::OffloaderPlatform> platform_;
  mutable std::mutex mapping_mutex_;
  MappingPhase mapping_phase_ = MappingPhase::kUnmapped;
  void* mapped_address_ = nullptr;
  std::vector<raiden::PjRtCopyFuture> in_flight_copies_;
  absl::Status deferred_copy_error_ = absl::OkStatus();
};

}  // namespace tpu_raiden::torch

#endif  // TPU_SYNC_FRAMEWORKS_TORCH_KV_CACHE_OFFLOADER_H_
