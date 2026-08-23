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
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_client.h"

namespace tpu_raiden {

struct HostBufferAllocation {
  uint8_t* ptr = nullptr;
  size_t size = 0;
  std::shared_ptr<void> owner;
};

struct SharedMemoryInfo {
  std::string shm_key;
  size_t size = 0;
  size_t offset = 0;
  void* base_ptr = nullptr;
  int fd = -1;
};

class HostMemoryAllocator;

// A wrapper class behaving like a callback while keeping allocator references.
class HostBufferAllocator {
 public:
  HostBufferAllocator() = default;
  HostBufferAllocator(std::nullptr_t) {}  // NOLINT

  // Construct from std::function
  HostBufferAllocator(  // NOLINT
      std::function<
          absl::StatusOr<HostBufferAllocation>(size_t, const xla::PjRtDevice*)>
          alloc_fn)
      : alloc_fn_(std::move(alloc_fn)) {}

  // Construct from shared_ptr to HostMemoryAllocator
  HostBufferAllocator(
      std::shared_ptr<HostMemoryAllocator> allocator);  // NOLINT

  // Construct from general lambdas
  template <
      typename F,
      typename = std::enable_if_t<
          !std::is_same_v<std::decay_t<F>, HostBufferAllocator> &&
          !std::is_same_v<std::decay_t<F>,
                          std::shared_ptr<HostMemoryAllocator>> &&
          std::is_invocable_r_v<absl::StatusOr<HostBufferAllocation>, F, size_t,
                                const xla::PjRtDevice*>>>
  HostBufferAllocator(F&& f)  // NOLINT
      : alloc_fn_(std::forward<F>(f)) {}

  absl::StatusOr<HostBufferAllocation> operator()(
      size_t size, const xla::PjRtDevice* device) const {
    if (!alloc_fn_) {
      return absl::FailedPreconditionError("Allocator is not initialized");
    }
    return alloc_fn_(size, device);
  }

  explicit operator bool() const { return static_cast<bool>(alloc_fn_); }

  std::shared_ptr<HostMemoryAllocator> host_memory_allocator() const {
    return allocator_;
  }

  friend bool operator==(const HostBufferAllocator& a, std::nullptr_t) {
    return !a.alloc_fn_;
  }
  friend bool operator==(std::nullptr_t, const HostBufferAllocator& a) {
    return !a.alloc_fn_;
  }
  friend bool operator!=(const HostBufferAllocator& a, std::nullptr_t) {
    return static_cast<bool>(a.alloc_fn_);
  }
  friend bool operator!=(std::nullptr_t, const HostBufferAllocator& a) {
    return static_cast<bool>(a.alloc_fn_);
  }

 private:
  std::function<absl::StatusOr<HostBufferAllocation>(size_t,
                                                     const xla::PjRtDevice*)>
      alloc_fn_;
  std::shared_ptr<HostMemoryAllocator> allocator_;
};

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

  // Returns shared memory info (fd, offset, size, base pointer) for a given
  // host pointer if the memory was allocated via shared memory.
  virtual absl::StatusOr<SharedMemoryInfo> GetSharedMemoryInfo(
      const void* ptr) const {
    return absl::UnimplementedError(
        "GetSharedMemoryInfo is not supported for this allocator.");
  }
};

inline HostBufferAllocator::HostBufferAllocator(
    std::shared_ptr<HostMemoryAllocator> allocator)
    : allocator_(allocator) {
  if (allocator) {
    alloc_fn_ = [allocator](size_t size, const xla::PjRtDevice* device) {
      return allocator->AllocateDmaMappedForDevice(size, device);
    };
  }
}

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

  ~SharedMemoryHostMemoryAllocator() override;

  absl::StatusOr<HostBufferAllocation> Allocate(size_t size_bytes) override;
  absl::StatusOr<HostBufferAllocation> AllocateDmaMapped(
      size_t size_bytes) override;
  absl::StatusOr<HostBufferAllocation> AllocateDmaMappedForDevice(
      size_t size_bytes, const xla::PjRtDevice* device) override;

  absl::StatusOr<SharedMemoryInfo> GetSharedMemoryInfo(
      const void* ptr) const override;

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

// Factory to create allocators wrapped in HostBufferAllocator
HostBufferAllocator CreateHostMemoryAllocator(xla::PjRtClient* client,
                                              size_t num_blocks,
                                              size_t block_size);

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_HOST_MEMORY_ALLOCATOR_H_
