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

#include "tpu_sync/frameworks/torch/kv_cache_offloader.h"

#include <errno.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/frameworks/torch/torch_tpu_utils.h"
#include "xla/pjrt/pjrt_client.h"

namespace tpu_raiden::torch {
namespace {

absl::Status FirstError(absl::Status first, const absl::Status& next) {
  return first.ok() ? next : first;
}

absl::Status PosixError(const char* operation, std::string_view guidance = {}) {
  const int saved_errno = errno;
  absl::Status status = absl::InternalError(
      absl::StrCat(operation, " failed: ", std::strerror(saved_errno),
                   guidance.empty() ? "" : absl::StrCat("; ", guidance)));
  status.SetPayload(offloader_internal::kPosixErrorPayloadUrl,
                    absl::Cord(operation));
  return status;
}

absl::StatusOr<size_t> CheckedMultiply(size_t lhs, size_t rhs,
                                       std::string_view context) {
  if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
    return absl::OutOfRangeError(absl::StrCat(context, " overflows size_t"));
  }
  return lhs * rhs;
}

absl::StatusOr<size_t> CheckedAdd(size_t lhs, size_t rhs,
                                  std::string_view context) {
  if (lhs > std::numeric_limits<size_t>::max() - rhs) {
    return absl::OutOfRangeError(absl::StrCat(context, " overflows size_t"));
  }
  return lhs + rhs;
}

int64_t CheckedTransferValue(size_t value, std::string_view context) {
  if (value > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error(
        absl::StrCat(context, " does not fit in int64_t"));
  }
  return static_cast<int64_t>(value);
}

std::shared_ptr<offloader_internal::OffloaderPlatform>
CreateDefaultOffloaderPlatform(xla::PjRtClient* client) {
  assert(client != nullptr &&
         "prepared device state must provide a non-null PJRT client");
  auto platform = std::make_shared<offloader_internal::OffloaderPlatform>();
  platform->page_size = []() -> absl::StatusOr<size_t> {
    const long value = sysconf(_SC_PAGESIZE);
    if (value <= 0) return PosixError("sysconf(_SC_PAGESIZE)");
    return static_cast<size_t>(value);
  };
  platform->dma_map = [client](void* address, size_t size_bytes) {
    return client->DmaMap(address, size_bytes);
  };
  platform->dma_unmap = [client](void* address) {
    return client->DmaUnmap(address);
  };
  return platform;
}

}  // namespace

struct KVCacheOffloader::DeviceState {
  struct Layer {
    at::Tensor tensor;
    std::optional<torch_tpu::DeviceBufferRef> ref;
    raiden::BufferHoldAndAlias buffer;
  };

  size_t num_layers = 0;
  size_t num_blocks = 0;
  size_t page_nbytes = 0;
  xla::PjRtClient* client = nullptr;
  std::vector<Layer> layers;
};

std::shared_ptr<KVCacheOffloader::DeviceState>
KVCacheOffloader::PrepareDeviceState(
    const std::vector<at::Tensor>& kv_cache_tensors, size_t page_nbytes) {
  if (kv_cache_tensors.empty()) {
    throw std::invalid_argument("kv_cache_tensors must not be empty");
  }
  if (page_nbytes == 0) {
    throw std::invalid_argument("page_nbytes must be greater than zero");
  }
  (void)CheckedTransferValue(page_nbytes, "page_nbytes");
  (void)CheckedTransferValue(kv_cache_tensors.size(), "num_layers");

  auto state = std::make_shared<DeviceState>();
  state->num_layers = kv_cache_tensors.size();
  state->page_nbytes = page_nbytes;
  state->layers.reserve(state->num_layers);

  xla::PjRtClient* expected_client = nullptr;
  int expected_global_device_id = -1;
  size_t expected_physical_size = 0;
  for (size_t layer_id = 0; layer_id < state->num_layers; ++layer_id) {
    const at::Tensor& tensor = kv_cache_tensors[layer_id];
    // KV cache offloading intentionally operates on the model's live buffer.
    UnpackedTensor unpacked =
        UnpackTorchTensor(tensor, /*unsafe_skip_buffer_lock=*/true);
    if (unpacked.buffer.buffer == nullptr ||
        unpacked.buffer.buffer->client() == nullptr ||
        unpacked.buffer.device == nullptr) {
      throw std::invalid_argument(
          absl::StrCat("kv_cache_tensors[", layer_id,
                       "] resolved to a null PJRT buffer/client/device"));
    }

    const size_t physical_size = unpacked.buffer.GetOnDeviceSizeInBytes();
    if (physical_size == 0) {
      throw std::invalid_argument(absl::StrCat(
          "kv_cache_tensors[", layer_id, "] has an empty physical buffer"));
    }
    if (physical_size % page_nbytes != 0) {
      throw std::invalid_argument(absl::StrCat(
          "kv_cache_tensors[", layer_id, "] has ", physical_size,
          " physical bytes, which is not divisible by page_nbytes=",
          page_nbytes));
    }

    xla::PjRtClient* client = unpacked.buffer.buffer->client();
    const int global_device_id =
        unpacked.buffer.device->global_device_id().value();
    if (layer_id == 0) {
      expected_physical_size = physical_size;
      expected_client = client;
      expected_global_device_id = global_device_id;
      state->num_blocks = physical_size / page_nbytes;
      state->client = client;
    } else {
      if (physical_size != expected_physical_size) {
        throw std::invalid_argument(absl::StrCat(
            "all kv_cache_tensors must have the same physical size; layer 0 "
            "has ",
            expected_physical_size, " bytes, but layer ", layer_id, " has ",
            physical_size, " bytes"));
      }
      if (client != expected_client ||
          global_device_id != expected_global_device_id) {
        throw std::invalid_argument(absl::StrCat(
            "all kv_cache_tensors must belong to the same PJRT client and "
            "device; layer 0 uses global device ",
            expected_global_device_id, " but layer ", layer_id, " uses ",
            global_device_id));
      }
    }

    state->layers.push_back(DeviceState::Layer{
        .tensor = tensor,
        .ref = std::move(unpacked.ref),
        .buffer = std::move(unpacked.buffer),
    });
  }
  return state;
}

KVCacheOffloader::KVCacheOffloader(
    const std::vector<at::Tensor>& kv_cache_tensors, size_t page_nbytes)
    : KVCacheOffloader(kv_cache_tensors, page_nbytes, /*platform=*/nullptr) {}

KVCacheOffloader::KVCacheOffloader(
    const std::vector<at::Tensor>& kv_cache_tensors, size_t page_nbytes,
    std::shared_ptr<offloader_internal::OffloaderPlatform> platform)
    : device_state_(PrepareDeviceState(kv_cache_tensors, page_nbytes)),
      platform_(platform == nullptr
                    ? CreateDefaultOffloaderPlatform(device_state_->client)
                    : std::move(platform)) {}

KVCacheOffloader::KVCacheOffloader(
    std::shared_ptr<offloader_internal::OffloaderPlatform> platform)
    : platform_(std::move(platform)) {}

KVCacheOffloader::~KVCacheOffloader() {
  bool needs_cleanup = false;
  {
    std::lock_guard<std::mutex> lock(mapping_mutex_);
    needs_cleanup = mapping_phase_ != MappingPhase::kUnmapped;
  }
  if (!needs_cleanup) return;
  const absl::Status status = UnmapSharedMemory();
  if (!status.ok()) {
    LOG(ERROR) << "KVCacheOffloader cleanup failed: " << status;
  }
}

absl::Status KVCacheOffloader::MapSharedMemory(void* mapped_address,
                                               size_t pool_size_bytes) {
  if (mapped_address == nullptr) {
    return absl::InvalidArgumentError("mapped_address must be non-null");
  }
  if (pool_size_bytes == 0) {
    return absl::InvalidArgumentError(
        "pool_size_bytes must be greater than zero");
  }

  absl::StatusOr<size_t> page_size = platform_->page_size();
  if (!page_size.ok()) return page_size.status();
  if (*page_size == 0) {
    return absl::InternalError("system page size must be greater than zero");
  }
  const uintptr_t address_value = reinterpret_cast<uintptr_t>(mapped_address);
  if (address_value % *page_size != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("mapped_address=", address_value,
                     " must be aligned to system page size ", *page_size));
  }
  if (pool_size_bytes % *page_size != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("pool_size_bytes=", pool_size_bytes,
                     " must be aligned to system page size ", *page_size));
  }
  if (pool_size_bytes > std::numeric_limits<uintptr_t>::max() - address_value) {
    return absl::InvalidArgumentError(
        "mapped_address + pool_size_bytes overflows the process address "
        "space");
  }

  {
    std::lock_guard<std::mutex> lock(mapping_mutex_);
    if (mapping_phase_ != MappingPhase::kUnmapped) {
      return absl::FailedPreconditionError(
          "shared memory is already mapped or registration is in progress");
    }
    mapping_phase_ = MappingPhase::kMapping;
  }

  const auto reset_mapping_reservation = [this]() {
    std::lock_guard<std::mutex> lock(mapping_mutex_);
    assert(mapping_phase_ == MappingPhase::kMapping &&
           "map failure must still own the mapping reservation");
    mapping_phase_ = MappingPhase::kUnmapped;
  };

  const absl::Status status =
      platform_->dma_map(mapped_address, pool_size_bytes);
  if (!status.ok()) {
    reset_mapping_reservation();
    return status;
  }

  {
    std::lock_guard<std::mutex> lock(mapping_mutex_);
    assert(mapping_phase_ == MappingPhase::kMapping &&
           "successful map must still own the mapping reservation");
    mapped_address_ = mapped_address;
    mapping_phase_ = MappingPhase::kMapped;
  }
  return absl::OkStatus();
}

absl::Status KVCacheOffloader::UnmapSharedMemory() {
  void* mapped_address = nullptr;
  std::vector<raiden::PjRtCopyFuture> copies_to_await;
  absl::Status copy_status = absl::OkStatus();
  {
    std::lock_guard<std::mutex> lock(mapping_mutex_);
    if (mapping_phase_ == MappingPhase::kUnmapped) {
      return absl::FailedPreconditionError("shared memory is not mapped");
    }
    if (mapping_phase_ == MappingPhase::kMapping) {
      return absl::FailedPreconditionError(
          "shared memory mapping is still in progress");
    }
    if (mapping_phase_ == MappingPhase::kUnmapping) {
      return absl::FailedPreconditionError(
          "shared memory is already being unmapped");
    }
    assert(mapped_address_ != nullptr &&
           "mapped phase must retain the registered address");
    mapped_address = mapped_address_;
    copies_to_await = std::move(in_flight_copies_);
    copy_status = std::move(deferred_copy_error_);
    deferred_copy_error_ = absl::OkStatus();
    mapping_phase_ = MappingPhase::kUnmapping;
  }

  for (raiden::PjRtCopyFuture& future : copies_to_await) {
    copy_status = FirstError(std::move(copy_status), future.Await());
  }

  const absl::Status status = platform_->dma_unmap(mapped_address);
  if (!status.ok()) {
    std::lock_guard<std::mutex> lock(mapping_mutex_);
    deferred_copy_error_ =
        FirstError(std::move(deferred_copy_error_), copy_status);
    mapping_phase_ = MappingPhase::kMapped;
    return status;
  }

  {
    std::lock_guard<std::mutex> lock(mapping_mutex_);
    mapped_address_ = nullptr;
    deferred_copy_error_ = absl::OkStatus();
    mapping_phase_ = MappingPhase::kUnmapped;
  }
  return copy_status;
}

bool KVCacheOffloader::is_shared_memory_mapped() const {
  std::lock_guard<std::mutex> lock(mapping_mutex_);
  return mapping_phase_ == MappingPhase::kMapped;
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheOffloader::H2d(
    const std::vector<int64_t>& block_ids,
    const std::vector<at::Tensor>& object_tensors, int64_t rank_id) {
  return CopyBlocks(block_ids, object_tensors, rank_id,
                    CopyDirection::kHostToDevice);
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheOffloader::D2h(
    const std::vector<int64_t>& block_ids,
    const std::vector<at::Tensor>& object_tensors, int64_t rank_id) {
  return CopyBlocks(block_ids, object_tensors, rank_id,
                    CopyDirection::kDeviceToHost);
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheOffloader::CopyBlocks(
    const std::vector<int64_t>& block_ids,
    const std::vector<at::Tensor>& object_tensors, int64_t rank_id,
    CopyDirection direction) {
  if (device_state_ == nullptr) {
    return absl::FailedPreconditionError(
        "KVCacheOffloader has no registered device KV cache");
  }
  if (block_ids.empty()) {
    return absl::InvalidArgumentError("block_ids must not be empty");
  }
  if (block_ids.size() != object_tensors.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "block_ids and object_tensors must have the same length; got ",
        block_ids.size(), " and ", object_tensors.size()));
  }
  if (rank_id < 0) {
    return absl::InvalidArgumentError("rank_id must be non-negative");
  }

  std::unordered_set<int64_t> unique_blocks;
  unique_blocks.reserve(block_ids.size());
  std::vector<int64_t> device_offsets;
  device_offsets.reserve(block_ids.size());
  for (size_t object_id = 0; object_id < block_ids.size(); ++object_id) {
    const int64_t block_id = block_ids[object_id];
    if (block_id < 0 ||
        static_cast<size_t>(block_id) >= device_state_->num_blocks) {
      return absl::OutOfRangeError(absl::StrCat(
          "block_ids[", object_id, "]=", block_id,
          " is outside block range [0, ", device_state_->num_blocks, ")"));
    }
    if (!unique_blocks.insert(block_id).second) {
      return absl::InvalidArgumentError(absl::StrCat(
          "block_ids contains duplicate block ", block_id,
          "; concurrent writes to one block have undefined ordering"));
    }
    absl::StatusOr<size_t> offset =
        CheckedMultiply(static_cast<size_t>(block_id),
                        device_state_->page_nbytes, "device page offset");
    if (!offset.ok()) return offset.status();
    if (*offset > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
      return absl::OutOfRangeError(
          "device page offset does not fit in int64_t");
    }
    device_offsets.push_back(static_cast<int64_t>(*offset));
  }

  absl::StatusOr<size_t> rank_bytes =
      CheckedMultiply(device_state_->num_layers, device_state_->page_nbytes,
                      "object tensor bytes per rank");
  if (!rank_bytes.ok()) return rank_bytes.status();

  size_t expected_num_ranks = 0;
  size_t expected_object_bytes = 0;
  std::vector<uint8_t*> host_bases;
  host_bases.reserve(object_tensors.size());
  for (size_t object_id = 0; object_id < object_tensors.size(); ++object_id) {
    const at::Tensor& tensor = object_tensors[object_id];
    if (!tensor.device().is_cpu()) {
      return absl::InvalidArgumentError(
          absl::StrCat("object_tensors[", object_id, "] must be a CPU tensor"));
    }
    if (!tensor.is_contiguous()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "object_tensors[", object_id,
          "] must be contiguous; a nonzero storage offset is allowed"));
    }
    if (tensor.dim() != 3 || tensor.scalar_type() != at::ScalarType::Char) {
      return absl::InvalidArgumentError(absl::StrCat(
          "object_tensors[", object_id,
          "] must be a rank-3 CPU int8 tensor with shape [num_ranks, ",
          device_state_->num_layers, ", ", device_state_->page_nbytes, "]"));
    }
    if (tensor.size(0) <= 0 ||
        tensor.size(1) != static_cast<int64_t>(device_state_->num_layers) ||
        tensor.size(2) != static_cast<int64_t>(device_state_->page_nbytes)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "object_tensors[", object_id, "] must have shape [num_ranks, ",
          device_state_->num_layers, ", ", device_state_->page_nbytes,
          "] with num_ranks > 0"));
    }

    const size_t num_ranks = static_cast<size_t>(tensor.size(0));
    if (object_id == 0) {
      expected_num_ranks = num_ranks;
      absl::StatusOr<size_t> object_bytes =
          CheckedMultiply(num_ranks, *rank_bytes, "object tensor byte size");
      if (!object_bytes.ok()) return object_bytes.status();
      expected_object_bytes = *object_bytes;
    } else if (num_ranks != expected_num_ranks) {
      return absl::InvalidArgumentError(
          "all object_tensors must have the same num_ranks dimension");
    }
    if (static_cast<size_t>(rank_id) >= num_ranks) {
      return absl::OutOfRangeError(
          absl::StrCat("rank_id=", rank_id, " is outside object_tensors[",
                       object_id, "] first dimension [0, ", num_ranks, ")"));
    }
    if (static_cast<size_t>(tensor.nbytes()) != expected_object_bytes) {
      return absl::InvalidArgumentError(
          absl::StrCat("object_tensors[", object_id, "] has ", tensor.nbytes(),
                       " bytes, expected ", expected_object_bytes));
    }

    auto* host_base = static_cast<uint8_t*>(tensor.data_ptr());
    host_bases.push_back(host_base);
  }

  absl::StatusOr<size_t> host_rank_offset = CheckedMultiply(
      static_cast<size_t>(rank_id), *rank_bytes, "object tensor rank offset");
  if (!host_rank_offset.ok()) return host_rank_offset.status();
  std::vector<size_t> host_layer_offsets;
  host_layer_offsets.reserve(device_state_->num_layers);
  for (size_t layer_id = 0; layer_id < device_state_->num_layers; ++layer_id) {
    absl::StatusOr<size_t> layer_offset = CheckedMultiply(
        layer_id, device_state_->page_nbytes, "object tensor layer offset");
    if (!layer_offset.ok()) return layer_offset.status();
    absl::StatusOr<size_t> host_offset = CheckedAdd(
        *host_rank_offset, *layer_offset, "object tensor host offset");
    if (!host_offset.ok()) return host_offset.status();
    assert(*host_offset <= expected_object_bytes - device_state_->page_nbytes &&
           "validated rank/layer geometry must stay inside the object view");
    host_layer_offsets.push_back(*host_offset);
  }

  auto tensor_holds = std::make_shared<std::vector<at::Tensor>>(object_tensors);
  std::vector<raiden::PjRtCopyFuture> layer_futures;
  layer_futures.reserve(device_state_->num_layers);

  std::lock_guard<std::mutex> lock(mapping_mutex_);
  if (mapping_phase_ != MappingPhase::kMapped) {
    return absl::FailedPreconditionError(
        "shared memory must be mapped before submitting a copy");
  }
  auto first_pending =
      std::remove_if(in_flight_copies_.begin(), in_flight_copies_.end(),
                     [this](raiden::PjRtCopyFuture& future) {
                       if (!future.IsReady()) return false;
                       deferred_copy_error_ = FirstError(
                           std::move(deferred_copy_error_), future.PollError());
                       return true;
                     });
  in_flight_copies_.erase(first_pending, in_flight_copies_.end());

  const int64_t transfer_size =
      static_cast<int64_t>(device_state_->page_nbytes);
  for (size_t layer_id = 0; layer_id < device_state_->num_layers; ++layer_id) {
    const size_t host_offset = host_layer_offsets[layer_id];

    absl::StatusOr<raiden::PjRtCopyFuture> future;
    if (direction == CopyDirection::kHostToDevice) {
      std::vector<raiden::H2dCopy> copies;
      copies.reserve(object_tensors.size());
      for (size_t object_id = 0; object_id < object_tensors.size();
           ++object_id) {
        copies.push_back(raiden::H2dCopy{
            .src = host_bases[object_id] + host_offset,
            .dst_off = device_offsets[object_id],
            .size = transfer_size,
        });
      }
      future =
          raiden::IssueH2dShard(device_state_->layers[layer_id].buffer, copies);
    } else {
      std::vector<raiden::D2hCopy> copies;
      copies.reserve(object_tensors.size());
      for (size_t object_id = 0; object_id < object_tensors.size();
           ++object_id) {
        copies.push_back(raiden::D2hCopy{
            .dst = host_bases[object_id] + host_offset,
            .src_off = device_offsets[object_id],
            .size = transfer_size,
        });
      }
      future =
          raiden::IssueD2hShard(device_state_->layers[layer_id].buffer, copies);
    }
    if (!future.ok()) {
      if (!layer_futures.empty()) {
        raiden::PjRtCopyFuture submitted =
            raiden::JoinPjRtCopyFutures(layer_futures);
        submitted.AddKeepAlive(tensor_holds);
        submitted.AddKeepAlive(device_state_);
        in_flight_copies_.push_back(std::move(submitted));
      }
      return future.status();
    }
    layer_futures.push_back(std::move(future.value()));
  }

  raiden::PjRtCopyFuture joined = raiden::JoinPjRtCopyFutures(layer_futures);
  joined.AddKeepAlive(std::move(tensor_holds));
  joined.AddKeepAlive(device_state_);
  in_flight_copies_.push_back(joined);
  return joined;
}

size_t KVCacheOffloader::num_layers() const {
  return device_state_ == nullptr ? 0 : device_state_->num_layers;
}

size_t KVCacheOffloader::num_blocks() const {
  return device_state_ == nullptr ? 0 : device_state_->num_blocks;
}

size_t KVCacheOffloader::page_nbytes() const {
  return device_state_ == nullptr ? 0 : device_state_->page_nbytes;
}

}  // namespace tpu_raiden::torch
