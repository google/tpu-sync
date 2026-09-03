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

#include "tpu_sync/frameworks/jax/weight_synchronizer_ffi.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/ffi/api/ffi.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/frameworks/jax/weight_synchronizer_ffi_internal.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace weight_sync {

WeightSynchronizerBase* g_weight_synchronizers[32] = {nullptr};
std::unique_ptr<stream_executor::Stream> g_streams[32] = {nullptr};

static absl::Mutex ws_mu;
static auto* ws_map =
    new absl::flat_hash_map<int32_t, WeightSynchronizerBase*>();
static auto* ws_shard_to_slot_map =
    new absl::flat_hash_map<WeightSynchronizerBase*,
                            absl::flat_hash_map<int32_t, size_t>>();

void ClearSharedWsMap() {
  absl::MutexLock lock(ws_mu);
  ws_map->clear();
  ws_shard_to_slot_map->clear();
  for (int i = 0; i < 32; ++i) {
    g_weight_synchronizers[i] = nullptr;
    g_streams[i].reset();
  }
}

// Retains and returns a thread-safe singleton instance of
// `WeightSynchronizerBase` for a given listener port across multiple devices on
// the same physical host. In multi-device/multi-host JAX topologies (`pjit`),
// multiple local ranks initialize FFI targets independently. Sharing the
// instance avoids re-binding the same socket (`Address already in use`).
static WeightSynchronizerBase* GetSharedWs(
    int32_t shard_idx, int32_t listener_port, int32_t num_layers,
    int32_t parallelism, const std::vector<size_t>& slice_byte_sizes,
    int32_t local_port, int32_t num_shards) {
  absl::MutexLock lock(ws_mu);
  int32_t submanager_idx = (num_shards > 0) ? (shard_idx / num_shards) : 0;
  int32_t key = (listener_port > 0)
                    ? (listener_port + submanager_idx)
                    : (listener_port == 0 ? -(submanager_idx + 1)
                                          : -(submanager_idx + 1000));
  auto& ws = (*ws_map)[key];
  if (ws == nullptr) {
    std::optional<int> opt_listener_port =
        (listener_port >= 0)
            ? std::make_optional(
                  listener_port > 0 ? (listener_port + submanager_idx) : 0)
            : std::nullopt;
    std::optional<int> opt_local_port =
        (local_port > 0)
            ? std::make_optional(local_port + submanager_idx)
            : (local_port == 0 ? std::make_optional(0) : std::nullopt);

    std::vector<HostNicAddress> host_nics = GetLocalHostNicAddresses();
    std::vector<HostNicAddress> data_nics;
    for (const auto& nic : host_nics) {
      if (nic.classification == NicClassification::kDataPlane) {
        data_nics.push_back(nic);
      }
    }
    std::optional<std::string> sub_bind_ip = std::nullopt;
    if (submanager_idx < static_cast<int>(data_nics.size())) {
      sub_bind_ip = data_nics[submanager_idx].ip_address;
    } else if (!data_nics.empty()) {
      sub_bind_ip = data_nics[submanager_idx % data_nics.size()].ip_address;
    }

    ws = new WeightSynchronizerBase(
        static_cast<size_t>(num_layers), static_cast<size_t>(num_shards),
        slice_byte_sizes, opt_local_port, std::nullopt, parallelism,
        opt_listener_port, sub_bind_ip);
  }
  auto& slot_map = (*ws_shard_to_slot_map)[ws];
  auto [it, inserted] = slot_map.try_emplace(
      shard_idx, static_cast<size_t>(shard_idx) % ws->num_shards());
  if (inserted) {
    std::vector<int64_t> indices(ws->num_shards(), -1);
    for (const auto& [s_id, s_slot] : slot_map) {
      if (s_slot < indices.size()) {
        indices[s_slot] = s_id;
      }
    }
    ws->SetGlobalShardIndices(std::move(indices));
  }
  return ws;
}

static size_t GetLocalSlot(int32_t shard_idx) {
  if (shard_idx < 0 || shard_idx >= 32) {
    return 0;
  }
  WeightSynchronizerBase* ws = g_weight_synchronizers[shard_idx];
  if (ws == nullptr) {
    return static_cast<size_t>(shard_idx);
  }
  absl::MutexLock lock(ws_mu);
  auto it = ws_shard_to_slot_map->find(ws);
  if (it != ws_shard_to_slot_map->end()) {
    auto slot_it = it->second.find(shard_idx);
    if (slot_it != it->second.end()) {
      return slot_it->second;
    }
  }
  return static_cast<size_t>(shard_idx) % ws->num_shards();
}

// FFI Init custom call implementation for WeightSynchronizer (Host CPU
// Executed)
xla::ffi::Error TriggerWeightSynchronizerInitImpl(
    xla::ffi::AnyBuffer x, xla::ffi::AnyBuffer shard_idx_buf,
    xla::ffi::AnyBuffer slice_byte_sizes_buf, int32_t local_port,
    int32_t parallelism, int32_t num_layers, int32_t listener_port,
    int32_t num_shards, xla::ffi::Result<xla::ffi::AnyBuffer> out) {
  (void)x;
  if (shard_idx_buf.untyped_data() == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInvalidArgument,
                           "shard_idx_buf null.");
  }
  int32_t shard_idx =
      *reinterpret_cast<const int32_t*>(shard_idx_buf.untyped_data());
  if (shard_idx < 0 || shard_idx >= 32) {
    return xla::ffi::Error(
        xla::ffi::ErrorCode::kInvalidArgument,
        absl::StrCat("shard_idx out of bounds [0, 32): ", shard_idx));
  }

  if (slice_byte_sizes_buf.untyped_data() == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInvalidArgument,
                           "slice_byte_sizes_buf null.");
  }
  const int32_t* sizes_ptr =
      reinterpret_cast<const int32_t*>(slice_byte_sizes_buf.untyped_data());
  size_t num_sizes = slice_byte_sizes_buf.element_count();
  std::vector<size_t> slice_byte_sizes(num_sizes);
  for (size_t i = 0; i < num_sizes; ++i) {
    slice_byte_sizes[i] = static_cast<size_t>(sizes_ptr[i]);
  }

  if (g_weight_synchronizers[shard_idx] == nullptr) {
    VLOG(1)
        << "[TPU Worker FFI] >>> WS LAZY INITIALIZATION TRIGGERED <<< Shard: "
        << shard_idx;

    g_weight_synchronizers[shard_idx] =
        GetSharedWs(shard_idx, listener_port, num_layers, parallelism,
                    slice_byte_sizes, local_port, num_shards);

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
      LOG(WARNING) << "Failed to resolve platform, skipping stream init "
                      "for CPU test.";
    } else {
      auto platform = platform_or.value();

      auto executor_or = platform->ExecutorForDevice(dev_id);
      if (!executor_or.ok()) {
        return xla::ffi::Error(
            xla::ffi::ErrorCode::kInternal,
            "Failed to retrieve StreamExecutor for device: " +
                std::string(executor_or.status().message()));
      }
      auto executor = executor_or.value();

      auto stream_or = executor->CreateStream();
      if (!stream_or.ok()) {
        return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                               "Failed to allocate execution Stream: " +
                                   std::string(stream_or.status().message()));
      }
      g_streams[shard_idx] = std::move(stream_or.value());
    }
  }

  // Get port
  auto port_opt = g_weight_synchronizers[shard_idx]->local_port();
  if (!port_opt.has_value()) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           "WS has no local port assigned.");
  }
  int32_t port = port_opt.value();

  std::string ip_str = g_weight_synchronizers[shard_idx]->local_ip();
  uint8_t ipv6[16] = {0};
  if (absl::StrContains(ip_str, ".")) {
    struct in_addr ipv4;
    inet_pton(AF_INET, ip_str.c_str(), &ipv4);
    ipv6[10] = 0xff;
    ipv6[11] = 0xff;
    std::memcpy(ipv6 + 12, &ipv4.s_addr, 4);
  } else {
    inet_pton(AF_INET6, ip_str.c_str(), ipv6);
  }

  if (out->element_count() < 5) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInvalidArgument,
                           "Output buffer too small for IPv6 and port.");
  }
  int32_t* out_ptr = reinterpret_cast<int32_t*>(out->untyped_data());
  std::memcpy(out_ptr, ipv6, 16);
  out_ptr[4] = port;
  if (out->element_count() >= 6) {
    out_ptr[5] = g_weight_synchronizers[shard_idx]->listener_port().value_or(0);
  }

  return xla::ffi::Error::Success();
}

// FFI execution handler for WeightSynchronizer Init and D2H (Host CPU Executed)
xla::ffi::Error TriggerWeightSynchronizerInitAndD2hHelper(
    xla::ffi::AnyBuffer shard_idx_buf, xla::ffi::AnyBuffer slice_byte_sizes_buf,
    absl::Span<const xla::ffi::AnyBuffer> jax_arrays, int32_t local_port,
    int32_t parallelism, int32_t num_layers, int32_t listener_port,
    int32_t num_shards, xla::ffi::Result<xla::ffi::AnyBuffer> out) {
  // --- Init Part ---
  if (shard_idx_buf.untyped_data() == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInvalidArgument,
                           "shard_idx_buf has null untyped data.");
  }
  int32_t shard_idx =
      *reinterpret_cast<const int32_t*>(shard_idx_buf.untyped_data());
  if (shard_idx < 0 || shard_idx >= 32) {
    return xla::ffi::Error(
        xla::ffi::ErrorCode::kInvalidArgument,
        absl::StrCat("shard_idx out of bounds [0, 32): ", shard_idx));
  }

  if (slice_byte_sizes_buf.untyped_data() == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInvalidArgument,
                           "slice_byte_sizes_buf null.");
  }
  const int32_t* sizes_ptr =
      reinterpret_cast<const int32_t*>(slice_byte_sizes_buf.untyped_data());
  size_t num_sizes = slice_byte_sizes_buf.element_count();
  std::vector<size_t> slice_byte_sizes(num_sizes);
  for (size_t i = 0; i < num_sizes; ++i) {
    slice_byte_sizes[i] = static_cast<size_t>(sizes_ptr[i]);
  }

  if (g_weight_synchronizers[shard_idx] == nullptr) {
    VLOG(1)
        << "[TPU Worker FFI] >>> WS LAZY INITIALIZATION TRIGGERED <<< Shard: "
        << shard_idx;

    g_weight_synchronizers[shard_idx] =
        GetSharedWs(shard_idx, listener_port, num_layers, parallelism,
                    slice_byte_sizes, local_port, num_shards);

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
      LOG(WARNING) << "Failed to resolve platform, skipping stream init "
                      "for CPU test.";
    } else {
      auto platform = platform_or.value();
      auto executor_or = platform->ExecutorForDevice(dev_id);
      if (!executor_or.ok()) {
        return xla::ffi::Error(
            xla::ffi::ErrorCode::kInternal,
            "Failed to retrieve StreamExecutor for device: " +
                std::string(executor_or.status().message()));
      }
      auto executor = executor_or.value();
      auto stream_or = executor->CreateStream();
      if (!stream_or.ok()) {
        return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                               "Failed to allocate execution Stream: " +
                                   std::string(stream_or.status().message()));
      }
      g_streams[shard_idx] = std::move(stream_or.value());
    }
  }

  // --- Extract IP/Port Part ---
  auto port_opt = g_weight_synchronizers[shard_idx]->local_port();
  if (!port_opt.has_value()) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           "WS has no local port assigned.");
  }
  int32_t port = port_opt.value();

  std::string ip_str = g_weight_synchronizers[shard_idx]->local_ip();
  uint8_t ipv6[16] = {0};
  if (absl::StrContains(ip_str, ".")) {
    struct in_addr ipv4;
    inet_pton(AF_INET, ip_str.c_str(), &ipv4);
    ipv6[10] = 0xff;
    ipv6[11] = 0xff;
    std::memcpy(ipv6 + 12, &ipv4.s_addr, 4);
  } else {
    inet_pton(AF_INET6, ip_str.c_str(), ipv6);
  }

  if (out->element_count() < 5) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInvalidArgument,
                           "Output buffer too small for IPv6 and port.");
  }
  int32_t* out_ptr = reinterpret_cast<int32_t*>(out->untyped_data());
  std::memcpy(out_ptr, ipv6, 16);
  out_ptr[4] = port;
  if (out->element_count() >= 6) {
    out_ptr[5] = g_weight_synchronizers[shard_idx]->listener_port().value_or(0);
  }

  // --- D2H Part (Loop through all passed layers) ---
  size_t local_slot = GetLocalSlot(shard_idx);

  for (size_t i = 0; i < jax_arrays.size(); ++i) {
    auto anchor = jax_arrays[i];

    size_t size = g_weight_synchronizers[shard_idx]->block_bytes(i);
    uint8_t* dst_host_ptr = const_cast<uint8_t*>(
        g_weight_synchronizers[shard_idx]->GetHostBufferPtr(i, local_slot));
    const uint8_t* src_device_ptr =
        reinterpret_cast<const uint8_t*>(anchor.untyped_data());

    if (g_streams[shard_idx] == nullptr) {
      std::memcpy(dst_host_ptr, src_device_ptr, size);
    } else {
      stream_executor::Stream* stream = g_streams[shard_idx].get();
      stream_executor::DeviceAddressBase device_src(
          const_cast<uint8_t*>(src_device_ptr), size);

      absl::Status status = stream->Memcpy(dst_host_ptr, device_src, size);
      if (!status.ok()) {
        return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                               "D2H Memcpy failed for layer " +
                                   std::to_string(i) + ": " +
                                   std::string(status.message()));
      }
    }
  }

  // Sync stream once at the end (if using streams)
  if (g_streams[shard_idx] != nullptr) {
    stream_executor::Stream* stream = g_streams[shard_idx].get();
    absl::Status sync_status = stream->BlockHostUntilDone();
    if (!sync_status.ok()) {
      return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                             "Stream sync failed at the end of D2H copies: " +
                                 std::string(sync_status.message()));
    }
  }

  return xla::ffi::Error::Success();
}

xla::ffi::Error TriggerWeightSynchronizerInitAndD2hImpl(
    xla::ffi::AnyBuffer shard_idx_buf, xla::ffi::AnyBuffer slice_byte_sizes_buf,
    xla::ffi::RemainingArgs jax_arrays, int32_t local_port, int32_t parallelism,
    int32_t num_layers, int32_t listener_port, int32_t num_shards,
    xla::ffi::Result<xla::ffi::AnyBuffer> out) {
  std::vector<xla::ffi::AnyBuffer> arrays;
  arrays.reserve(jax_arrays.size());
  for (size_t i = 0; i < jax_arrays.size(); ++i) {
    auto arr_or = jax_arrays.get<xla::ffi::AnyBuffer>(i);
    if (arr_or.has_error()) {
      return xla::ffi::Error(
          xla::ffi::ErrorCode::kInvalidArgument,
          "Failed to get anchor: " + arr_or.error().message());
    }
    arrays.push_back(arr_or.value());
  }

  return TriggerWeightSynchronizerInitAndD2hHelper(
      shard_idx_buf, slice_byte_sizes_buf, arrays, local_port, parallelism,
      num_layers, listener_port, num_shards, out);
}

// FFI custom call handler executing asynchronous Host-to-Device (H2D) memory
// transfers from local staging buffers (`GetHostBufferPtr`) directly onto
// device memory buffers. Uses multi-host modulo indexing (`shard_idx %
// num_shards()`) to ensure global shard indices map to the correct local
// staging slot.
xla::ffi::Error TriggerH2DImpl(xla::ffi::AnyBuffer shard_idx_buf,
                               int32_t layer_idx,
                               xla::ffi::Result<xla::ffi::AnyBuffer> out) {
  if (shard_idx_buf.untyped_data() == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInvalidArgument,
                           "shard_idx_buf null.");
  }
  int32_t shard_idx =
      *reinterpret_cast<const int32_t*>(shard_idx_buf.untyped_data());
  if (shard_idx < 0 || shard_idx >= 32 ||
      g_weight_synchronizers[shard_idx] == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           "WS not initialized.");
  }

  size_t size = g_weight_synchronizers[shard_idx]->block_bytes(layer_idx);
  size_t local_slot = GetLocalSlot(shard_idx);
  const uint8_t* src_host_ptr =
      g_weight_synchronizers[shard_idx]->GetHostBufferPtr(layer_idx,
                                                          local_slot);

  uint8_t* d_base = reinterpret_cast<uint8_t*>(out->untyped_data());

  if (g_streams[shard_idx] == nullptr) {
    std::memcpy(d_base, src_host_ptr, size);
  } else {
    stream_executor::Stream* stream = g_streams[shard_idx].get();
    stream_executor::DeviceAddressBase device_dst(d_base, size);
    absl::Status status = stream->Memcpy(&device_dst, src_host_ptr, size);
    if (!status.ok()) {
      return xla::ffi::Error(
          xla::ffi::ErrorCode::kInternal,
          "H2D Memcpy failed: " + std::string(status.message()));
    }

    absl::Status sync_status = stream->BlockHostUntilDone();
    if (!sync_status.ok()) {
      return xla::ffi::Error(
          xla::ffi::ErrorCode::kInternal,
          "Stream sync failed: " + std::string(sync_status.message()));
    }
  }
  return xla::ffi::Error::Success();
}

xla::ffi::Error TriggerMultiH2DImpl(xla::ffi::AnyBuffer shard_idx_buf,
                                    xla::ffi::RemainingRets rets) {
  if (shard_idx_buf.untyped_data() == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInvalidArgument,
                           "shard_idx_buf null.");
  }
  int32_t shard_idx =
      *reinterpret_cast<const int32_t*>(shard_idx_buf.untyped_data());
  if (shard_idx < 0 || shard_idx >= 32 ||
      g_weight_synchronizers[shard_idx] == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           "WS not initialized.");
  }

  size_t num_layers = rets.size();
  size_t local_slot = GetLocalSlot(shard_idx);

  for (size_t i = 0; i < num_layers; ++i) {
    auto ret_or = rets.get<xla::ffi::AnyBuffer>(i);
    if (!ret_or.has_value()) {
      return ret_or.error();
    }
    xla::ffi::AnyBuffer out = **ret_or;
    uint8_t* d_base = reinterpret_cast<uint8_t*>(out.untyped_data());

    size_t size = g_weight_synchronizers[shard_idx]->block_bytes(i);
    const uint8_t* src_host_ptr =
        g_weight_synchronizers[shard_idx]->GetHostBufferPtr(i, local_slot);

    if (g_streams[shard_idx] == nullptr) {
      std::memcpy(d_base, src_host_ptr, size);
    } else {
      stream_executor::Stream* stream = g_streams[shard_idx].get();
      stream_executor::DeviceAddressBase device_dst(d_base, size);
      absl::Status status = stream->Memcpy(&device_dst, src_host_ptr, size);
      if (!status.ok()) {
        return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                               "H2D Memcpy failed for layer " +
                                   std::to_string(i) + ": " +
                                   std::string(status.message()));
      }
    }
  }

  if (g_streams[shard_idx] != nullptr) {
    stream_executor::Stream* stream = g_streams[shard_idx].get();
    absl::Status sync_status = stream->BlockHostUntilDone();
    if (!sync_status.ok()) {
      return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                             "Stream sync failed at the end of H2D copies: " +
                                 std::string(sync_status.message()));
    }
  }
  return xla::ffi::Error::Success();
}

// FFI custom call handler executing asynchronous Device-to-Host (D2H) memory
// transfers from device memory directly onto local staging buffers
// (`GetHostBufferPtr`).
xla::ffi::Error TriggerD2HImpl(xla::ffi::AnyBuffer anchor,
                               xla::ffi::AnyBuffer shard_idx_buf,
                               int32_t layer_idx,
                               xla::ffi::Result<xla::ffi::AnyBuffer> out) {
  (void)out;
  if (shard_idx_buf.untyped_data() == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInvalidArgument,
                           "shard_idx_buf null.");
  }
  int32_t shard_idx =
      *reinterpret_cast<const int32_t*>(shard_idx_buf.untyped_data());
  if (shard_idx < 0 || shard_idx >= 32 ||
      g_weight_synchronizers[shard_idx] == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           "WS not initialized.");
  }

  size_t size = g_weight_synchronizers[shard_idx]->block_bytes(layer_idx);
  size_t local_slot = GetLocalSlot(shard_idx);
  uint8_t* dst_host_ptr =
      const_cast<uint8_t*>(g_weight_synchronizers[shard_idx]->GetHostBufferPtr(
          layer_idx, local_slot));
  const uint8_t* src_device_ptr =
      reinterpret_cast<const uint8_t*>(anchor.untyped_data());

  if (g_streams[shard_idx] == nullptr) {
    std::memcpy(dst_host_ptr, src_device_ptr, size);
  } else {
    stream_executor::Stream* stream = g_streams[shard_idx].get();
    stream_executor::DeviceAddressBase device_src(
        const_cast<uint8_t*>(src_device_ptr), size);

    absl::Status status = stream->Memcpy(dst_host_ptr, device_src, size);
    if (!status.ok()) {
      return xla::ffi::Error(
          xla::ffi::ErrorCode::kInternal,
          "D2H Memcpy failed: " + std::string(status.message()));
    }

    absl::Status sync_status = stream->BlockHostUntilDone();
    if (!sync_status.ok()) {
      return xla::ffi::Error(
          xla::ffi::ErrorCode::kInternal,
          "Stream sync failed: " + std::string(sync_status.message()));
    }
  }
  return xla::ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    kWSInit, TriggerWeightSynchronizerInitImpl,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()  // anchor JAX input array (Arg 0)
        .Arg<xla::ffi::AnyBuffer>()  // shard_idx JAX input array (Arg 1)
        .Arg<xla::ffi::AnyBuffer>()  // slice_byte_sizes JAX input array (Arg 2)
        .Attr<int32_t>("local_port")
        .Attr<int32_t>("parallelism")
        .Attr<int32_t>("num_layers")
        .Attr<int32_t>("listener_port")
        .Attr<int32_t>("num_shards")
        .Ret<xla::ffi::AnyBuffer>());  // return result buffer

XLA_FFI_DEFINE_HANDLER(
    kWSInitWeightSynchronizerAndD2h, TriggerWeightSynchronizerInitAndD2hImpl,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()  // shard_idx JAX input array (Arg 0)
        .Arg<xla::ffi::AnyBuffer>()  // slice_byte_sizes JAX input array (Arg 1)
        .RemainingArgs()             // jax_arrays (variable count)
        .Attr<int32_t>("local_port")
        .Attr<int32_t>("parallelism")
        .Attr<int32_t>("num_layers")
        .Attr<int32_t>("listener_port")
        .Attr<int32_t>("num_shards")
        .Ret<xla::ffi::AnyBuffer>());  // return result buffer

XLA_FFI_DEFINE_HANDLER(kH2D, TriggerH2DImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<xla::ffi::AnyBuffer>()  // shard_idx_buf
                           .Attr<int32_t>("layer_idx")
                           .Ret<xla::ffi::AnyBuffer>()  // result buffer
);

XLA_FFI_DEFINE_HANDLER(kWSMultiH2D, TriggerMultiH2DImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<xla::ffi::AnyBuffer>()  // shard_idx_buf
                           .RemainingRets());

XLA_FFI_DEFINE_HANDLER(
    kD2H, TriggerD2HImpl,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()  // anchor
        .Arg<xla::ffi::AnyBuffer>()  // shard_idx_buf
        .Attr<int32_t>("layer_idx")
        .Ret<xla::ffi::AnyBuffer>()  // result (aliased to anchor)
);

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "init_weight_synchronizer",
                         "Host", kWSInit);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "init_weight_synchronizer",
                         "TPU", kWSInit);

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                         "init_weight_synchronizer_and_d2h", "Host",
                         kWSInitWeightSynchronizerAndD2h);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
                         "init_weight_synchronizer_and_d2h", "TPU",
                         kWSInitWeightSynchronizerAndD2h);

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "ws_h2d", "Host", kH2D);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "ws_h2d", "TPU", kH2D);

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "ws_multi_h2d", "Host",
                         kWSMultiH2D);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "ws_multi_h2d", "TPU",
                         kWSMultiH2D);

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "ws_d2h", "Host", kD2H);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "ws_d2h", "TPU", kD2H);

extern "C" void ForceLinkWeightSynchronizerFfi() {}

}  // namespace weight_sync
}  // namespace tpu_raiden
