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

#include "tpu_sync/frameworks/jax/kv_cache_manager_ffi.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/ffi/api/ffi.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"

namespace tpu_raiden {
namespace kv_cache {

KVCacheManagerWithTransfer* g_kv_cache_managers[32] = {nullptr};
stream_executor::StreamExecutor* g_executors[32] = {nullptr};
std::unique_ptr<stream_executor::Stream> g_streams[32] = {nullptr};

int64_t g_block_byte_size = 0;
int64_t g_local_blocks_per_shard = 0;

// FFI Init custom call implementation (Host CPU Executed)
xla::ffi::Error TriggerRaidenInitImpl(
    xla::ffi::AnyBuffer x, xla::ffi::AnyBuffer shard_idx_buf,
    int64_t slice_byte_size, int32_t local_port, int32_t parallelism,
    int32_t host_blocks_to_allocate, int32_t num_layers,
    xla::ffi::Result<xla::ffi::AnyBuffer> out) {
  (void)x;
  (void)out;
  if (shard_idx_buf.untyped_data() == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInvalidArgument,
                           "shard_idx_buf has null untyped data.");
  }
  int32_t shard_idx =
      *reinterpret_cast<const int32_t*>(shard_idx_buf.untyped_data());
  if (shard_idx < 0 || shard_idx >= 32) {
    return xla::ffi::Error(
        xla::ffi::ErrorCode::kInvalidArgument,
        "shard_idx out of bounds [0, 32): " + std::to_string(shard_idx));
  }

  if (g_kv_cache_managers[shard_idx] == nullptr) {
    VLOG(1) << "[TPU Worker FFI] >>> SERVER-SIDE LAZY INITIALIZATION "
               "TRIGGERED <<< Shard: "
            << shard_idx;
    VLOG(1) << "[TPU Worker FFI] Instantiating pure C++ KVCacheManager on "
               "worker C++ heap for Shard: "
            << shard_idx;
    VLOG(1) << "[TPU Worker FFI] Configuration: layers=" << num_layers
            << ", parallelism=" << parallelism
            << ", global_blocks=" << host_blocks_to_allocate;

    g_kv_cache_managers[shard_idx] = new KVCacheManagerWithTransfer(
        static_cast<size_t>(num_layers), static_cast<size_t>(parallelism),
        static_cast<size_t>(slice_byte_size),
        local_port > 0 ? std::make_optional(local_port) : std::nullopt,
        host_blocks_to_allocate > 0
            ? std::make_optional(host_blocks_to_allocate)
            : std::nullopt,
        parallelism);

    // Allocate the StreamExecutor Stream once per shard, and cache E2E!
    int64_t dev_id = static_cast<int64_t>(shard_idx);
    auto platform_or =
        stream_executor::PlatformManager::PlatformWithName("TPU");
    if (!platform_or.ok()) {
      platform_or =
          stream_executor::PlatformManager::PlatformWithName("Deepsea");
    }
    if (!platform_or.ok()) {
      platform_or = stream_executor::PlatformManager::PlatformWithName("Host");
    }
    if (!platform_or.ok()) {
      return xla::ffi::Error(
          xla::ffi::ErrorCode::kInternal,
          "Failed to resolve physical TPU/Deepsea StreamExecutor platform: " +
              std::string(platform_or.status().message()));
    }
    auto platform = platform_or.value();

    auto executor_or = platform->ExecutorForDevice(dev_id);
    if (!executor_or.ok()) {
      return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                             "Failed to retrieve StreamExecutor for device: " +
                                 std::string(executor_or.status().message()));
    }
    auto executor = executor_or.value();
    g_executors[shard_idx] = executor;  // Cache executor

    auto stream_or = executor->CreateStream();
    if (!stream_or.ok()) {
      return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                             "Failed to allocate execution Stream: " +
                                 std::string(stream_or.status().message()));
    }
    g_streams[shard_idx] = std::move(stream_or.value());

    g_block_byte_size = slice_byte_size;
    g_local_blocks_per_shard = host_blocks_to_allocate / parallelism;
    VLOG(1)
        << "[TPU Worker FFI] KVCacheManager successfully allocated for Shard: "
        << shard_idx << "! block_byte_size=" << g_block_byte_size
        << ", local_blocks_per_shard=" << g_local_blocks_per_shard;
  }
  return xla::ffi::Error::Success();
}

// FFI execution handler for Host-to-Device (H2D) transfers (Host-CPU Executed)
xla::ffi::Error TriggerRaidenH2dImpl(
    xla::ffi::AnyBuffer src_offsets, xla::ffi::AnyBuffer dst_offsets,
    xla::ffi::AnyBuffer copy_sizes, xla::ffi::AnyBuffer shard_idx_buf,
    xla::ffi::AnyBuffer cache_slice_buf, int32_t layer_idx,
    xla::ffi::Result<xla::ffi::AnyBuffer> out) {
  if (src_offsets.untyped_data() == nullptr ||
      dst_offsets.untyped_data() == nullptr ||
      copy_sizes.untyped_data() == nullptr ||
      shard_idx_buf.untyped_data() == nullptr ||
      cache_slice_buf.untyped_data() == nullptr) {
    return xla::ffi::Error(
        xla::ffi::ErrorCode::kInvalidArgument,
        "One or more metadata/buffer arguments have null untyped data.");
  }

  const int32_t* p_shard_idx =
      reinterpret_cast<const int32_t*>(shard_idx_buf.untyped_data());
  int32_t shard_idx = *p_shard_idx;
  if (shard_idx < 0 || shard_idx >= 32) {
    return xla::ffi::Error(
        xla::ffi::ErrorCode::kInvalidArgument,
        "shard_idx out of bounds [0, 32): " + std::to_string(shard_idx));
  }
  if (g_kv_cache_managers[shard_idx] == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kFailedPrecondition,
                           "Raiden FFI KVCacheManager "
                           "has not been initialized for Shard: " +
                               std::to_string(shard_idx));
  }
  int64_t dev_id = static_cast<int64_t>(shard_idx);
  ::xla::se::Stream* stream = g_streams[shard_idx].get();
  if (stream == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kFailedPrecondition,
                           "Stream has not been initialized for Shard: " +
                               std::to_string(shard_idx));
  }

  int64_t num_chunks = src_offsets.element_count();
  const int32_t* h_src_offsets =
      reinterpret_cast<const int32_t*>(src_offsets.untyped_data());
  const int32_t* h_dst_offsets =
      reinterpret_cast<const int32_t*>(dst_offsets.untyped_data());
  const int32_t* h_copy_sizes =
      reinterpret_cast<const int32_t*>(copy_sizes.untyped_data());

  const uint8_t* h_base =
      g_kv_cache_managers[shard_idx]->base()->GetHostPointer(
          static_cast<size_t>(layer_idx), 0);

  LOG(WARNING) << "[TPU Worker FFI] H2D pointers: cache_slice_buf="
               << cache_slice_buf.untyped_data()
               << ", out=" << out->untyped_data();
  uint8_t* d_base = reinterpret_cast<uint8_t*>(out->untyped_data());

  VLOG(1) << "[TPU Worker FFI] >>> EXECUTING HOST-TO-DEVICE (H2D) PCIE "
             "DMA TRANSFER <<<";
  VLOG(1) << "[TPU Worker FFI] Target: layer=" << layer_idx
          << ", device_shard=" << dev_id << ", chunks_count=" << num_chunks
          << ", local_device_base=" << (void*)d_base;

  for (int64_t i = 0; i < num_chunks; ++i) {
    int64_t copy_size = h_copy_sizes[i];
    if (copy_size == 0) continue;

    int64_t global_d_idx = static_cast<int64_t>(h_dst_offsets[i]);
    int64_t owner_dev_id = global_d_idx / g_local_blocks_per_shard;
    if (owner_dev_id != dev_id) {
      continue;
    }

    int64_t local_s_idx =
        static_cast<int64_t>(h_src_offsets[i]) % g_local_blocks_per_shard;
    int64_t local_d_idx =
        static_cast<int64_t>(h_dst_offsets[i]) % g_local_blocks_per_shard;

    int64_t s_offset = local_s_idx * g_block_byte_size;
    int64_t d_offset = local_d_idx * g_block_byte_size;
    size_t size_bytes = copy_size * g_block_byte_size;

    stream_executor::DeviceAddressBase device_dst(d_base + d_offset,
                                                  size_bytes);

    absl::Status status =
        stream->Memcpy(&device_dst, h_base + s_offset, size_bytes);
    if (!status.ok()) {
      return xla::ffi::Error(
          xla::ffi::ErrorCode::kInternal,
          "Stream memcpy H2D failed: " + std::string(status.message()));
    }
  }

  absl::Status sync_status = stream->BlockHostUntilDone();
  if (!sync_status.ok()) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           "Stream BlockHostUntilDone failed: " +
                               std::string(sync_status.message()));
  }

  return xla::ffi::Error::Success();
}

// FFI execution handler for Device-to-Host (D2H) transfers (Host-CPU Executed)
xla::ffi::Error TriggerRaidenD2hImpl(
    xla::ffi::AnyBuffer src_offsets, xla::ffi::AnyBuffer dst_offsets,
    xla::ffi::AnyBuffer copy_sizes, xla::ffi::AnyBuffer shard_idx_buf,
    xla::ffi::AnyBuffer cache_slice_buf, int32_t layer_idx,
    xla::ffi::Result<xla::ffi::AnyBuffer> out) {
  if (src_offsets.untyped_data() == nullptr ||
      dst_offsets.untyped_data() == nullptr ||
      copy_sizes.untyped_data() == nullptr ||
      shard_idx_buf.untyped_data() == nullptr ||
      cache_slice_buf.untyped_data() == nullptr) {
    return xla::ffi::Error(
        xla::ffi::ErrorCode::kInvalidArgument,
        "One or more metadata/buffer arguments have null untyped data.");
  }

  const int32_t* p_shard_idx =
      reinterpret_cast<const int32_t*>(shard_idx_buf.untyped_data());
  int32_t shard_idx = *p_shard_idx;
  if (shard_idx < 0 || shard_idx >= 32) {
    return xla::ffi::Error(
        xla::ffi::ErrorCode::kInvalidArgument,
        "shard_idx out of bounds [0, 32): " + std::to_string(shard_idx));
  }
  if (g_kv_cache_managers[shard_idx] == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kFailedPrecondition,
                           "Raiden FFI KVCacheManager "
                           "has not been initialized for Shard: " +
                               std::to_string(shard_idx));
  }
  int64_t dev_id = static_cast<int64_t>(shard_idx);
  ::xla::se::Stream* stream = g_streams[shard_idx].get();
  if (stream == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kFailedPrecondition,
                           "Stream has not been initialized for Shard: " +
                               std::to_string(shard_idx));
  }
  auto executor = g_executors[shard_idx];
  if (executor == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kFailedPrecondition,
                           "Executor has not been initialized for Shard: " +
                               std::to_string(shard_idx));
  }

  // Synchronize all physical TPU device activity to guarantee that JAX stream
  // execution has completed writing target data to cache_slice_buf before host
  // DMA reading.
  bool sync_ok = executor->SynchronizeAllActivity();
  if (!sync_ok) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           "Executor SynchronizeAllActivity failed!");
  }

  int64_t num_chunks = src_offsets.element_count();
  const int32_t* h_src_offsets =
      reinterpret_cast<const int32_t*>(src_offsets.untyped_data());
  const int32_t* h_dst_offsets =
      reinterpret_cast<const int32_t*>(dst_offsets.untyped_data());
  const int32_t* h_copy_sizes =
      reinterpret_cast<const int32_t*>(copy_sizes.untyped_data());

  uint8_t* h_base = const_cast<uint8_t*>(
      g_kv_cache_managers[shard_idx]->base()->GetHostPointer(
          static_cast<size_t>(layer_idx), 0));

  const uint8_t* d_base =
      reinterpret_cast<const uint8_t*>(cache_slice_buf.untyped_data());

  VLOG(1) << "[TPU Worker FFI] >>> EXECUTING DEVICE-TO-HOST (D2H) PCIE "
             "DMA TRANSFER <<<";
  VLOG(1) << "[TPU Worker FFI] Target: layer=" << layer_idx
          << ", device_shard=" << dev_id << ", chunks_count=" << num_chunks
          << ", local_device_base=" << (void*)d_base;

  for (int64_t i = 0; i < num_chunks; ++i) {
    int64_t copy_size = h_copy_sizes[i];
    if (copy_size == 0) continue;

    int64_t global_s_idx = static_cast<int64_t>(h_src_offsets[i]);
    int64_t owner_dev_id = global_s_idx / g_local_blocks_per_shard;
    if (owner_dev_id != dev_id) {
      continue;
    }

    int64_t local_s_idx =
        static_cast<int64_t>(h_src_offsets[i]) % g_local_blocks_per_shard;
    int64_t local_d_idx =
        static_cast<int64_t>(h_dst_offsets[i]) % g_local_blocks_per_shard;

    int64_t s_offset = local_s_idx * g_block_byte_size;
    int64_t d_offset = local_d_idx * g_block_byte_size;
    size_t size_bytes = copy_size * g_block_byte_size;

    stream_executor::DeviceAddressBase device_src(
        const_cast<uint8_t*>(d_base + s_offset), size_bytes);

    absl::Status status =
        stream->Memcpy(h_base + d_offset, device_src, size_bytes);
    if (!status.ok()) {
      return xla::ffi::Error(
          xla::ffi::ErrorCode::kInternal,
          "Stream memcpy D2H failed: " + std::string(status.message()));
    }
  }

  absl::Status sync_status = stream->BlockHostUntilDone();
  if (!sync_status.ok()) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           "Stream BlockHostUntilDone failed: " +
                               std::string(sync_status.message()));
  }

  return xla::ffi::Error::Success();
}

// Register Stateless initialization handler
XLA_FFI_DEFINE_HANDLER(
    kRaidenInit, TriggerRaidenInitImpl,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()  // anchor JAX input array (Arg 0)
        .Arg<xla::ffi::AnyBuffer>()  // shard_idx JAX input array (Arg 1)
        .Attr<int64_t>("slice_byte_size")
        .Attr<int32_t>("local_port")
        .Attr<int32_t>("parallelism")
        .Attr<int32_t>("host_blocks_to_allocate")
        .Attr<int32_t>("num_layers")
        .Ret<xla::ffi::AnyBuffer>());  // return JAX output array

// Register Stateless H2D execution handlers
XLA_FFI_DEFINE_HANDLER(
    kTriggerRaidenH2d, TriggerRaidenH2dImpl,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()  // src_offsets (Arg 0)
        .Arg<xla::ffi::AnyBuffer>()  // dst_offsets (Arg 1)
        .Arg<xla::ffi::AnyBuffer>()  // copy_sizes (Arg 2)
        .Arg<xla::ffi::AnyBuffer>()  // shard_idx_buf (Arg 3)
        .Arg<xla::ffi::AnyBuffer>()  // cache_slice_buf (Arg 4)
        .Attr<int32_t>("layer_idx")  // layer_idx
        .Ret<xla::ffi::AnyBuffer>()  // out result buffer
);

XLA_FFI_DEFINE_HANDLER(
    kTriggerRaidenD2h, TriggerRaidenD2hImpl,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()  // src_offsets (Arg 0)
        .Arg<xla::ffi::AnyBuffer>()  // dst_offsets (Arg 1)
        .Arg<xla::ffi::AnyBuffer>()  // copy_sizes (Arg 2)
        .Arg<xla::ffi::AnyBuffer>()  // shard_idx_buf (Arg 3)
        .Arg<xla::ffi::AnyBuffer>()  // cache_slice_buf (Arg 4)
        .Attr<int32_t>("layer_idx")  // layer_idx
        .Ret<xla::ffi::AnyBuffer>()  // dummy result
);

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "init_kv_cache_manager",
                         "Host", kRaidenInit);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "init_kv_cache_manager",
                         "TPU", kRaidenInit);

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "h2d", "Host",
                         kTriggerRaidenH2d);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "h2d", "TPU",
                         kTriggerRaidenH2d);

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "d2h", "Host",
                         kTriggerRaidenD2h);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "d2h", "TPU",
                         kTriggerRaidenD2h);

void SyncCopies() {
  LOG(WARNING) << "[C++ FFI] Eager physical TPU PCIe DMA copies "
                  "synchronization complete!";
}

}  // namespace kv_cache
}  // namespace tpu_raiden
