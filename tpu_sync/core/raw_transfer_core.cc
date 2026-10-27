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

#include "tpu_sync/core/raw_transfer_core.h"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tpu_sync/core/tpu_utils.h"

namespace raiden {
namespace {
constexpr size_t kBar2PageAlignment = 4096;

size_t AlignUpBar2(size_t value) {
  return (value + kBar2PageAlignment - 1) & ~(kBar2PageAlignment - 1);
}
}  // namespace

MockBar2DmaBufArena::~MockBar2DmaBufArena() {
  if (base_ptr != nullptr && base_ptr != MAP_FAILED && total_size > 0) {
    munmap(base_ptr, total_size);
    base_ptr = nullptr;
  }
  if (dma_buf_fd >= 0) {
    close(dma_buf_fd);
    dma_buf_fd = -1;
  }
}

absl::StatusOr<std::shared_ptr<MockBar2DmaBufArena>>
MockBar2DmaBufArena::Create(size_t total_size) {
  const size_t aligned_size =
      AlignUpBar2(std::max<size_t>(total_size, kBar2PageAlignment));
  const int fd = memfd_create("raiden_bar2_dmabuf", MFD_CLOEXEC);
  if (fd < 0) {
    return absl::ErrnoToStatus(errno,
                               "memfd_create(raiden_bar2_dmabuf) failed");
  }
  if (ftruncate(fd, static_cast<off_t>(aligned_size)) != 0) {
    const int err = errno;
    close(fd);
    return absl::ErrnoToStatus(err,
                               "ftruncate(raiden_bar2_dmabuf) failed");
  }
  void* mapped =
      mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapped == MAP_FAILED) {
    const int err = errno;
    close(fd);
    return absl::ErrnoToStatus(err, "mmap(raiden_bar2_dmabuf) failed");
  }
  std::memset(mapped, 0, aligned_size);
  auto arena = std::make_shared<MockBar2DmaBufArena>();
  arena->dma_buf_fd = fd;
  arena->base_ptr = mapped;
  arena->total_size = aligned_size;
  return arena;
}

absl::StatusOr<ExternalDmaInfo> RaidenBufferHandle::GetExternalDmaInfo() const {
  if (external_dma_info.dma_buf_fd >= 0 ||
      external_dma_info.dma_buf_vaddr != nullptr) {
    return external_dma_info;
  }
  return absl::UnimplementedError(
      "GetExternalDmaInfo: no BAR2 dma-buf attached to RaidenBufferHandle");
}

absl::Status RaidenBufferHandle::SyncDeviceToBar2Window(int64_t buffer_offset,
                                                        size_t size) const {
  if (!external_dma_info.needs_mock_sync || size == 0 ||
      external_dma_info.dma_buf_vaddr == nullptr) {
    return absl::OkStatus();
  }
  if (buffer_offset < 0 ||
      static_cast<size_t>(buffer_offset) + size > external_dma_info.size) {
    return absl::OutOfRangeError(absl::StrCat(
        "SyncDeviceToBar2Window out of range: buffer_offset=", buffer_offset,
        " size=", size, " layer_size=", external_dma_info.size));
  }
  uint8_t* bar2_dst = static_cast<uint8_t*>(external_dma_info.dma_buf_vaddr) +
                      external_dma_info.offset +
                      static_cast<size_t>(buffer_offset);
  absl::StatusOr<PjRtCopyFuture> fut =
      IssueD2hShard(*this, {D2hCopy{.dst = bar2_dst,
                                    .src_off = buffer_offset,
                                    .size = static_cast<int64_t>(size)}});
  if (!fut.ok()) return fut.status();
  return fut->Await();
}

absl::Status RaidenBufferHandle::SyncBar2WindowToDevice(int64_t buffer_offset,
                                                        size_t size) const {
  if (!external_dma_info.needs_mock_sync || size == 0 ||
      external_dma_info.dma_buf_vaddr == nullptr) {
    return absl::OkStatus();
  }
  if (buffer_offset < 0 ||
      static_cast<size_t>(buffer_offset) + size > external_dma_info.size) {
    return absl::OutOfRangeError(absl::StrCat(
        "SyncBar2WindowToDevice out of range: buffer_offset=", buffer_offset,
        " size=", size, " layer_size=", external_dma_info.size));
  }
  const uint8_t* bar2_src =
      static_cast<const uint8_t*>(external_dma_info.dma_buf_vaddr) +
      external_dma_info.offset + static_cast<size_t>(buffer_offset);
  absl::StatusOr<PjRtCopyFuture> fut =
      IssueH2dShard(*this, {H2dCopy{.src = bar2_src,
                                    .dst_off = buffer_offset,
                                    .size = static_cast<int64_t>(size)}});
  if (!fut.ok()) return fut.status();
  return fut->Await();
}

absl::Status AttachMockBar2DmaBufArena(
    absl::Span<RaidenBufferHandle* const> shard_layer_handles) {
  if (shard_layer_handles.empty()) return absl::OkStatus();
  size_t total_size = 0;
  std::vector<size_t> layer_offsets(shard_layer_handles.size(), 0);
  std::vector<size_t> layer_sizes(shard_layer_handles.size(), 0);
  for (size_t i = 0; i < shard_layer_handles.size(); ++i) {
    RaidenBufferHandle* h = shard_layer_handles[i];
    if (h == nullptr) continue;
    const size_t sz =
        (h->is_common_buffer || h->c_hold != nullptr)
            ? h->GetOnDeviceSizeInBytes()
            : static_cast<size_t>(xla::ShapeUtil::ByteSizeOf(h->shape));
    layer_offsets[i] = total_size;
    layer_sizes[i] = sz;
    total_size += AlignUpBar2(std::max<size_t>(sz, 1));
  }
  absl::StatusOr<std::shared_ptr<MockBar2DmaBufArena>> arena =
      MockBar2DmaBufArena::Create(total_size);
  if (!arena.ok()) return arena.status();

  for (size_t i = 0; i < shard_layer_handles.size(); ++i) {
    RaidenBufferHandle* h = shard_layer_handles[i];
    if (h == nullptr) continue;
    h->mock_bar2_arena = *arena;
    h->external_dma_info = ExternalDmaInfo{
        .dma_buf_fd = (*arena)->dma_buf_fd,
        .offset = layer_offsets[i],
        .size = layer_sizes[i],
        .dma_buf_vaddr = (*arena)->base_ptr,
        .bar2_total_size = (*arena)->total_size,
        .needs_mock_sync = true,
    };
  }
  return absl::OkStatus();
}

std::vector<int> DetectNumaNodes(
    const std::vector<std::vector<RaidenBufferHandle>>& layer_buffers) {
  std::vector<int> unique_numa_nodes;
  for (const auto& layer : layer_buffers) {
    for (const auto& buf : layer) {
      if (buf.device) {
        int node = tpu_raiden::GetPjRtDeviceNumaNode(buf.device);
        if (node >= 0) {
          bool found = false;
          for (int n : unique_numa_nodes) {
            if (n == node) {
              found = true;
              break;
            }
          }
          if (!found) unique_numa_nodes.push_back(node);
        }
      }
    }
  }
  return unique_numa_nodes;
}

std::vector<tpu_raiden::transport::ShardInfo> DetectShards(
    const std::vector<std::vector<RaidenBufferHandle>>& layer_buffers) {
  std::vector<tpu_raiden::transport::ShardInfo> shards;
  if (layer_buffers.empty()) {
    return shards;
  }
  // Every layer places shard `i` on the same device, so layer 0 is
  // representative.
  shards.reserve(layer_buffers[0].size());
  for (size_t sh = 0; sh < layer_buffers[0].size(); ++sh) {
    const RaidenBufferHandle& buf = layer_buffers[0][sh];
    int node = -1;
    if (buf.device) {
      node = tpu_raiden::GetPjRtDeviceNumaNode(buf.device);
    }
    shards.push_back({.local_index = static_cast<int>(sh),
                      .numa_node = node >= 0 ? node : -1});
  }
  return shards;
}

}  // namespace raiden
