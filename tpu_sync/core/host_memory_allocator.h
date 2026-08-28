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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_HOST_MEMORY_ALLOCATOR_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_HOST_MEMORY_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_client.h"

namespace tpu_raiden {

struct HostBufferAllocation {
  uint8_t* ptr = nullptr;
  size_t size = 0;
  std::shared_ptr<void> owner;
};

// Returns a HostBufferAllocation of at least the requested size for a given
// device. If device is nullptr, it allocates on the default/current NUMA node.
// On failure, returns a non-OK status.
using HostBufferAllocator = std::function<absl::StatusOr<HostBufferAllocation>(
    size_t, const xla::PjRtDevice*)>;

// High-performance host memory allocator that allocates DMA-capable pinned
// memory using PJRT APIs or standard fallback allocations.
class HostMemoryAllocator {
 public:
  virtual ~HostMemoryAllocator() = default;

  static absl::StatusOr<std::unique_ptr<HostMemoryAllocator>> Create(
      xla::PjRtClient* pjrt_client);

  virtual absl::StatusOr<HostBufferAllocation> Allocate(size_t size_bytes) = 0;

  // Allocates host memory registered with the device DMA engine
  // (PjRtClient::DmaMap), so raw C-API D2H/H2D into it run as a true async DMA
  // instead of a synchronous staged copy (the producer-D2H bottleneck). The
  // default implementation falls back to Allocate (e.g. CPU / no DMA support).
  virtual absl::StatusOr<HostBufferAllocation> AllocateDmaMapped(
      size_t size_bytes) {
    return Allocate(size_bytes);
  }

  // Device-aware version of AllocateDmaMapped. Attempts to allocate host
  // memory on the NUMA node local to the given device.
  virtual absl::StatusOr<HostBufferAllocation> AllocateDmaMappedForDevice(
      size_t size_bytes, const xla::PjRtDevice* device) {
    return AllocateDmaMapped(size_bytes);
  }
};

class XlaHostMemoryAllocator : public HostMemoryAllocator {
 public:
  static absl::StatusOr<std::unique_ptr<XlaHostMemoryAllocator>> Create(
      xla::PjRtClient* absl_nonnull pjrt_client);

  absl::StatusOr<HostBufferAllocation> Allocate(size_t size_bytes) override;

  absl::StatusOr<HostBufferAllocation> AllocateDmaMapped(
      size_t size_bytes) override;

  absl::StatusOr<HostBufferAllocation> AllocateDmaMappedForDevice(
      size_t size_bytes, const xla::PjRtDevice* device) override;

 private:
  explicit XlaHostMemoryAllocator(xla::PjRtClient* client);

  xla::PjRtClient* client_ = nullptr;
};

class MallocHostMemoryAllocator : public HostMemoryAllocator {
 public:
  absl::StatusOr<HostBufferAllocation> Allocate(size_t size_bytes) override;
};

struct alignas(64) SharedMemoryHeader {
  uint64_t magic = 0x52414944454E5348;  // "RAIDENSH"
  uint32_t version = 1;
  char model_uid[256] = {0};
  uint32_t global_mesh_shape[5] = {0};
  uint32_t shard_layout[5] = {0};
  uint32_t num_blocks = 0;
  uint32_t block_size = 0;
  uint32_t num_heads = 0;
  uint32_t head_dim = 0;
  uint32_t itemsize = 0;
  uint64_t total_payload_bytes = 0;
  uint32_t reference_count = 0;
};

class SharedMemoryHostMemoryAllocator : public HostMemoryAllocator {
 public:
  static absl::StatusOr<std::unique_ptr<SharedMemoryHostMemoryAllocator>>
  Create(xla::PjRtClient* client, absl::string_view shm_key,
         const SharedMemoryHeader& expected_schema);


  // The shm_open name for this allocator's segment: `shm_key` plus an
  // optional "_<RAIDEN_SHM_SERVER_NAME>" suffix from the environment, plus
  // "_dev_<global_device_id>" when a device is given, "/"-prefixed.
  static std::string ComposeSegmentName(
      absl::string_view shm_key, std::optional<int64_t> global_device_id);

  // Validates the user-supplied segment-name inputs: `shm_key` (a
  // RAIDEN_SHM_KEY value; one leading '/' is tolerated as the shm_open name
  // prefix) and, when set, the RAIDEN_SHM_SERVER_NAME environment value.
  // Each must be non-empty, at most 200 characters (leaving NAME_MAX room
  // for the composed suffixes), and hold only letters, digits, '.', '_' and
  // '-' -- shm names are single path components, so '/' in particular makes
  // shm_open fail. Violations are an InvalidArgumentError naming the
  // variable and the offending character; user configuration is never
  // silently rewritten.
  static absl::Status ValidateShmNameParts(absl::string_view shm_key);

  ~SharedMemoryHostMemoryAllocator() override;

  absl::StatusOr<HostBufferAllocation> Allocate(size_t size_bytes) override;
  absl::StatusOr<HostBufferAllocation> AllocateDmaMapped(
      size_t size_bytes) override;
  absl::StatusOr<HostBufferAllocation> AllocateDmaMappedForDevice(
      size_t size_bytes, const xla::PjRtDevice* device) override;

 private:
  SharedMemoryHostMemoryAllocator(xla::PjRtClient* client,
                                  absl::string_view shm_key,
                                  const SharedMemoryHeader& expected_schema);

  xla::PjRtClient* client_ = nullptr;
  std::string shm_key_;
  SharedMemoryHeader expected_schema_;
  int shm_fd_ = -1;
  void* mapped_ptr_ = nullptr;
  size_t mapped_size_ = 0;
  bool dma_mapped_ = false;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_HOST_MEMORY_ALLOCATOR_H_
