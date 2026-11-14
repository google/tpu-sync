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

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/ffi/api/ffi.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/frameworks/jax/weight_synchronizer_ffi_internal.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace weight_sync {

std::unique_ptr<stream_executor::Stream> g_streams[kMaxShards] = {nullptr};

void ClearSharedWsMap() {
  WeightSynchronizerFfiManager::Default().Clear(/*delete_instances=*/false);
  for (size_t i = 0; i < kMaxShards; ++i) {
    g_streams[i].reset();
  }
}

namespace {

constexpr size_t kDescriptorBaseWidth = 4;
constexpr size_t kEndpointsMinWidth = 5;
constexpr size_t kEndpointsListenerWidth = 6;
constexpr size_t kEndpointsControlIpWidth = 10;

// `xla::ffi::ErrorCode` mirrors the canonical `absl::StatusCode` values.
xla::ffi::Error StatusToFfiError(const absl::Status& status) {
  return xla::ffi::Error(static_cast<xla::ffi::ErrorCode>(status.code()),
                         std::string(status.message()));
}

absl::StatusOr<absl::Span<uint8_t>> CheckedLocalHostBuffer(
    WeightSynchronizerBase& ws, size_t layer_idx, size_t local_slot,
    size_t required_bytes) {
  ABSL_ASSIGN_OR_RETURN(
      absl::Span<uint8_t> host_buf,
      ws.GetHostBufferWithLocalIndex(layer_idx, local_slot).Await());
  if (required_bytes > host_buf.size()) {
    return absl::OutOfRangeError(
        absl::StrCat("Host buffer smaller than block_bytes for layer ",
                     layer_idx, " local shard ", local_slot, ": ",
                     host_buf.size(), " < ", required_bytes));
  }
  return host_buf;
}

constexpr absl::string_view kClearRegistryHint =
    " Re-initializing with different arguments requires deleting the "
    "instances first (destroy_weight_synchronizer() in Python); a clear that "
    "keeps them also keeps their ports bound.";

// Returns FailedPrecondition for |action| ("register" or "re-initialize")
// |shard_idx| with |field| |requested| while the registry holds |registered|.
template <typename Registered, typename Requested>
absl::Status ReuseMismatchError(absl::string_view action, int32_t shard_idx,
                                absl::string_view field,
                                const Registered& registered,
                                const Requested& requested) {
  return absl::FailedPreconditionError(
      absl::StrCat("Cannot ", action, " shard_idx ", shard_idx, " with ", field,
                   " ", requested, ": the registry holds ", field, " ",
                   registered, ".", kClearRegistryHint));
}

absl::Span<const int32_t> Int32Span(const xla::ffi::AnyBuffer& buf) {
  return absl::MakeConstSpan(
      reinterpret_cast<const int32_t*>(buf.untyped_data()),
      buf.element_count());
}

bool IsVirtualInterface(absl::string_view name) {
  return name == "lo" || absl::StartsWith(name, "docker") ||
         absl::StartsWith(name, "cilium") || absl::StartsWith(name, "lxc") ||
         absl::StartsWith(name, "veth") || absl::StartsWith(name, "br-") ||
         absl::StartsWith(name, "virbr") || absl::StartsWith(name, "cni") ||
         absl::StartsWith(name, "flannel");
}

std::string ProbeDefaultRouteIp() {
  struct sockaddr_in dst4{};
  dst4.sin_family = AF_INET;
  dst4.sin_port = htons(80);
  if (inet_pton(AF_INET, "8.8.8.8", &dst4.sin_addr) == 1) {
    int fd = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (fd >= 0) {
      struct sockaddr_in local4{};
      socklen_t len = sizeof(local4);
      char buf[INET_ADDRSTRLEN] = {0};
      const bool ok =
          connect(fd, reinterpret_cast<const struct sockaddr*>(&dst4),
                  sizeof(dst4)) == 0 &&
          getsockname(fd, reinterpret_cast<struct sockaddr*>(&local4), &len) ==
              0 &&
          inet_ntop(AF_INET, &local4.sin_addr, buf, sizeof(buf)) != nullptr;
      close(fd);
      if (ok && buf[0] != '\0' && std::string(buf) != "0.0.0.0" &&
          !absl::StartsWith(buf, "127.")) {
        return std::string(buf);
      }
    }
  }
  struct sockaddr_in6 dst6{};
  dst6.sin6_family = AF_INET6;
  dst6.sin6_port = htons(80);
  if (inet_pton(AF_INET6, "2001:4860:4860::8888", &dst6.sin6_addr) == 1) {
    int fd = socket(AF_INET6, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (fd >= 0) {
      struct sockaddr_in6 local6{};
      socklen_t len = sizeof(local6);
      char buf[INET6_ADDRSTRLEN] = {0};
      const bool ok =
          connect(fd, reinterpret_cast<const struct sockaddr*>(&dst6),
                  sizeof(dst6)) == 0 &&
          getsockname(fd, reinterpret_cast<struct sockaddr*>(&local6), &len) ==
              0 &&
          inet_ntop(AF_INET6, &local6.sin6_addr, buf, sizeof(buf)) != nullptr;
      close(fd);
      if (ok && buf[0] != '\0' && std::string(buf) != "::" &&
          std::string(buf) != "::1") {
        return std::string(buf);
      }
    }
  }
  return "";
}

}  // namespace

std::string ResolveControlPlaneIp(absl::Span<const HostNicAddress> host_nics,
                                  absl::string_view fallback_ip,
                                  bool use_default_route_probe) {
  if (use_default_route_probe) {
    std::string route_ip = ProbeDefaultRouteIp();
    if (!route_ip.empty()) {
      return route_ip;
    }
  }
  for (const auto& nic : host_nics) {
    if (nic.interface_name == "eth0" && !nic.ip_address.empty()) {
      return nic.ip_address;
    }
  }
  for (const auto& nic : host_nics) {
    if (nic.classification == NicClassification::kControlPlane &&
        !IsVirtualInterface(nic.interface_name) && !nic.ip_address.empty()) {
      return nic.ip_address;
    }
  }
  return std::string(fallback_ip);
}

void PackIpToInt32s(absl::string_view ip_str, int32_t* dst) {
  uint8_t ipv6[16] = {0};
  const std::string ip(ip_str);
  if (absl::StrContains(ip, ".")) {
    struct in_addr ipv4{};
    inet_pton(AF_INET, ip.c_str(), &ipv4);
    ipv6[10] = 0xff;
    ipv6[11] = 0xff;
    std::memcpy(ipv6 + 12, &ipv4.s_addr, 4);
  } else {
    inet_pton(AF_INET6, ip.c_str(), ipv6);
  }
  std::memcpy(dst, ipv6, 16);
}

absl::StatusOr<FfiShardDescriptor> FfiShardDescriptor::FromSpan(
    absl::Span<const int32_t> raw, int32_t num_shards) {
  if (raw.empty()) {
    return absl::InvalidArgumentError(
        "Shard descriptor is empty; expected [shard_idx, local_slot, "
        "submanager_idx, global_shard_idx, numa_0..numa_{num_shards-1}] with "
        "at least shard_idx.");
  }
  FfiShardDescriptor desc;
  desc.shard_idx = raw[0];
  if (raw.size() >= 2) desc.local_slot = raw[1];
  if (raw.size() >= 3) desc.submanager_idx = raw[2];
  if (raw.size() >= 4) desc.global_shard_idx = raw[3];
  if (desc.shard_idx < 0 || static_cast<size_t>(desc.shard_idx) >= kMaxShards) {
    return absl::InvalidArgumentError(absl::StrCat(
        "shard_idx out of bounds [0, ", kMaxShards, "): ", desc.shard_idx));
  }
  const std::pair<absl::string_view, int32_t> optional_fields[] = {
      {"local_slot", desc.local_slot},
      {"submanager_idx", desc.submanager_idx},
      {"global_shard_idx", desc.global_shard_idx}};
  for (const auto& [name, value] : optional_fields) {
    if (value < -1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Shard descriptor for shard_idx ", desc.shard_idx, " has ", name, " ",
          value, "; expected -1 (absent) or a value >= 0."));
    }
  }
  if (raw.size() <= kDescriptorBaseWidth) {
    return desc;
  }
  if (num_shards <= 0 ||
      raw.size() != kDescriptorBaseWidth + static_cast<size_t>(num_shards)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Shard descriptor for shard_idx ", desc.shard_idx, " has ", raw.size(),
        " elements; expected at most ", kDescriptorBaseWidth, " or exactly ",
        kDescriptorBaseWidth, " + num_shards (num_shards=", num_shards, ")."));
  }
  desc.slot_numa_nodes.assign(raw.begin() + kDescriptorBaseWidth, raw.end());
  for (size_t slot = 0; slot < desc.slot_numa_nodes.size(); ++slot) {
    if (desc.slot_numa_nodes[slot] < -1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Shard descriptor for shard_idx ", desc.shard_idx, " has NUMA node ",
          desc.slot_numa_nodes[slot], " for local slot ", slot,
          "; expected -1 (unknown) or a node >= 0."));
    }
  }
  return desc;
}

int32_t FfiShardDescriptor::effective_submanager_idx(int32_t num_shards) const {
  if (submanager_idx >= 0) {
    return submanager_idx;
  }
  return (num_shards > 0) ? (shard_idx / num_shards) : 0;
}

int64_t FfiShardDescriptor::effective_global_shard() const {
  return has_explicit_global() ? static_cast<int64_t>(global_shard_idx)
                               : static_cast<int64_t>(shard_idx);
}

absl::Status ShardEndpoints::PackToSpan(absl::Span<int32_t> out) const {
  if (out.size() < kEndpointsMinWidth) {
    return absl::InvalidArgumentError(
        absl::StrCat("Output buffer too small for IPv6 and port: got ",
                     out.size(), " int32s, need >= ", kEndpointsMinWidth, "."));
  }
  PackIpToInt32s(data_ip, out.data());
  out[4] = data_port;
  if (out.size() >= kEndpointsListenerWidth) {
    out[5] = listener_port;
  }
  if (out.size() >= kEndpointsControlIpWidth) {
    PackIpToInt32s(control_ip, out.data() + 6);
  }
  return absl::OkStatus();
}

WeightSynchronizerFfiManager& WeightSynchronizerFfiManager::Default() {
  static auto* manager = new WeightSynchronizerFfiManager();
  return *manager;
}

WeightSynchronizerFfiManager& WeightSynchronizerFfiManager::ZeroCopy() {
  static auto* manager = new WeightSynchronizerFfiManager();
  return *manager;
}

// Shards of one submanager share an instance: in multi-device/multi-host JAX
// topologies (`pjit`), local ranks initialize FFI targets independently, and
// sharing avoids re-binding the same socket (`Address already in use`).
absl::StatusOr<WeightSynchronizerBase*>
WeightSynchronizerFfiManager::GetOrCreateSynchronizer(
    const FfiShardDescriptor& desc, int32_t listener_port, int32_t num_layers,
    int32_t parallelism, const std::vector<size_t>& slice_byte_sizes,
    int32_t local_port, int32_t num_shards) {
  const int32_t shard_idx = desc.shard_idx;
  if (shard_idx < 0 || static_cast<size_t>(shard_idx) >= kMaxShards) {
    return absl::InvalidArgumentError(absl::StrCat(
        "shard_idx out of bounds [0, ", kMaxShards, "): ", shard_idx));
  }
  if (num_shards <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("num_shards must be positive for shard_idx ", shard_idx,
                     ": ", num_shards));
  }
  if (!desc.slot_numa_nodes.empty() &&
      desc.slot_numa_nodes.size() != static_cast<size_t>(num_shards)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Shard descriptor for shard_idx ", shard_idx, " has ",
        desc.slot_numa_nodes.size(),
        " slot NUMA nodes; expected none or num_shards=", num_shards, "."));
  }
  if (desc.local_slot >= num_shards) {
    return absl::InvalidArgumentError(
        absl::StrCat("Shard descriptor for shard_idx ", shard_idx,
                     " has local_slot ", desc.local_slot,
                     "; expected a slot in [0, num_shards=", num_shards, ")."));
  }
  const size_t slot = static_cast<size_t>(
      desc.local_slot >= 0 ? desc.local_slot : shard_idx % num_shards);
  const int64_t global_shard = desc.effective_global_shard();
  const int32_t submanager_idx = desc.effective_submanager_idx(num_shards);
  const InstanceKey key{.submanager_idx = submanager_idx,
                        .listener_port = std::max(listener_port, -1),
                        .local_port = std::max(local_port, -1)};

  absl::MutexLock lock(mu_);
  auto ws_it = ws_map_.find(key);
  WeightSynchronizerBase* ws = ws_it == ws_map_.end() ? nullptr : ws_it->second;
  WeightSynchronizerBase* bound = shard_table_[shard_idx];
  if (bound != nullptr && bound != ws) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Cannot re-initialize shard_idx ", shard_idx,
        ": it is bound to the instance of ", instances_.at(bound).key,
        " but this call selects ", key, ".", kClearRegistryHint));
  }
  if (ws != nullptr) {
    ABSL_RETURN_IF_ERROR(ValidateReuse(ws, desc, num_layers, num_shards,
                                       parallelism, slice_byte_sizes));
  } else {
    // Positive ports are offset by the submanager index, so another key can
    // derive the same port, which this instance could not bind.
    for (const auto& [other, unused_ws] : ws_map_) {
      if ((listener_port > 0 && other.listener_port > 0 &&
           listener_port + submanager_idx ==
               other.listener_port + other.submanager_idx) ||
          (local_port > 0 && other.local_port > 0 &&
           local_port + submanager_idx ==
               other.local_port + other.submanager_idx)) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Cannot create an instance of ", key, " for shard_idx ", shard_idx,
            ": it derives the same listener or data port as the instance of ",
            other, "."));
      }
    }
    std::optional<int> opt_listener_port =
        (listener_port >= 0)
            ? std::make_optional(
                  listener_port > 0 ? (listener_port + submanager_idx) : 0)
            : std::nullopt;
    std::optional<int> opt_local_port =
        (local_port > 0)
            ? std::make_optional(local_port + submanager_idx)
            : (local_port == 0 ? std::make_optional(0) : std::nullopt);

    // One shard per local slot on its NUMA node. The core derives the distinct
    // NUMA nodes from this placement and binds the data NICs of each; with no
    // known node it binds only the host's first data NIC.
    std::vector<int32_t> slot_numa_nodes =
        desc.slot_numa_nodes.empty()
            ? std::vector<int32_t>(static_cast<size_t>(num_shards), -1)
            : desc.slot_numa_nodes;
    std::vector<transport::ShardInfo> shards;
    shards.reserve(static_cast<size_t>(num_shards));
    for (int32_t s = 0; s < num_shards; ++s) {
      shards.push_back(
          {.local_index = s,
           // Global shard indices are recorded via SetGlobalShardIndices below.
           .global_index = -1,
           .numa_node = slot_numa_nodes[static_cast<size_t>(s)]});
    }

    // Shared instances are intentionally never freed (see ClearSharedWsMap).
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<WeightSynchronizerBase> created,
        WeightSynchronizerBase::Create(
            static_cast<size_t>(num_layers), static_cast<size_t>(num_shards),
            slice_byte_sizes, opt_local_port,
            /*host_blocks_to_allocate=*/std::nullopt, parallelism,
            opt_listener_port, /*bind_ip=*/std::nullopt, /*layer_names=*/{},
            /*auto_h2d=*/false, /*numa_nodes=*/{}, host_nics_(),
            std::move(shards)));
    ws = created.release();
    ws_map_.emplace(key, ws);
    instances_.emplace(
        ws, InstanceInfo{.key = key,
                         .parallelism = parallelism,
                         .slice_byte_sizes = slice_byte_sizes,
                         .slot_numa_nodes = std::move(slot_numa_nodes)});
  }
  InstanceInfo& info = instances_.at(ws);
  if (auto it = info.shards.find(shard_idx); it != info.shards.end()) {
    if (it->second.slot != slot) {
      return ReuseMismatchError("re-initialize", shard_idx, "local slot",
                                it->second.slot, slot);
    }
    if (it->second.global_shard != global_shard) {
      return ReuseMismatchError("re-initialize", shard_idx,
                                "global shard index", it->second.global_shard,
                                global_shard);
    }
    return ws;
  }
  for (const auto& [other_idx, other] : info.shards) {
    if (other.slot == slot) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Cannot register shard_idx ", shard_idx, ": local slot ", slot,
          " of the instance of ", info.key, " is already taken by shard_idx ",
          other_idx, ".", kClearRegistryHint));
    }
  }
  info.shards.emplace(shard_idx, SlotAndGlobal{slot, global_shard});
  initialized_shards_.fetch_add(1, std::memory_order_release);
  std::vector<int64_t> indices(ws->num_shards(), -1);
  std::vector<int> local_indices(ws->num_shards(), -1);
  for (const auto& [s_id, s_info] : info.shards) {
    indices[s_info.slot] = s_info.global_shard;
    local_indices[s_info.slot] = static_cast<int>(s_info.slot);
  }
  ws->SetGlobalShardIndices(std::move(indices));
  ws->SetLocalShardIndices(std::move(local_indices));
  shard_table_[shard_idx] = ws;
  return ws;
}

absl::Status WeightSynchronizerFfiManager::ValidateReuse(
    WeightSynchronizerBase* ws, const FfiShardDescriptor& desc,
    int32_t num_layers, int32_t num_shards, int32_t parallelism,
    const std::vector<size_t>& slice_byte_sizes) const {
  const InstanceInfo& info = instances_.at(ws);
  const int32_t shard_idx = desc.shard_idx;
  const absl::string_view action =
      info.shards.contains(shard_idx) ? "re-initialize" : "register";
  if (static_cast<int64_t>(ws->num_layers()) != num_layers) {
    return ReuseMismatchError(action, shard_idx, "num_layers", ws->num_layers(),
                              num_layers);
  }
  if (static_cast<int64_t>(ws->num_shards()) != num_shards) {
    return ReuseMismatchError(action, shard_idx, "num_shards", ws->num_shards(),
                              num_shards);
  }
  if (info.parallelism != parallelism) {
    return ReuseMismatchError(action, shard_idx, "parallelism",
                              info.parallelism, parallelism);
  }
  if (info.slice_byte_sizes != slice_byte_sizes) {
    return ReuseMismatchError(action, shard_idx, "slice_byte_sizes",
                              absl::StrJoin(info.slice_byte_sizes, ","),
                              absl::StrJoin(slice_byte_sizes, ","));
  }
  if (!desc.slot_numa_nodes.empty() &&
      desc.slot_numa_nodes != info.slot_numa_nodes) {
    return ReuseMismatchError(action, shard_idx, "slot NUMA nodes",
                              absl::StrJoin(info.slot_numa_nodes, ","),
                              absl::StrJoin(desc.slot_numa_nodes, ","));
  }
  return absl::OkStatus();
}

WeightSynchronizerBase* WeightSynchronizerFfiManager::GetSynchronizerForShard(
    int32_t shard_idx) const {
  if (shard_idx < 0 || static_cast<size_t>(shard_idx) >= kMaxShards) {
    return nullptr;
  }
  absl::MutexLock lock(mu_);
  return shard_table_[shard_idx];
}

bool WeightSynchronizerFfiManager::HasActiveListener() const {
  absl::MutexLock lock(mu_);
  for (const WeightSynchronizerBase* ws : shard_table_) {
    if (ws != nullptr && ws->is_listener_active()) {
      return true;
    }
  }
  return false;
}

size_t WeightSynchronizerFfiManager::GetLocalSlot(int32_t shard_idx) const {
  if (shard_idx < 0 || static_cast<size_t>(shard_idx) >= kMaxShards) {
    return 0;
  }
  absl::MutexLock lock(mu_);
  WeightSynchronizerBase* ws = shard_table_[shard_idx];
  if (ws == nullptr) {
    return 0;
  }
  auto it = instances_.find(ws);
  if (it != instances_.end()) {
    auto slot_it = it->second.shards.find(shard_idx);
    if (slot_it != it->second.shards.end()) {
      return slot_it->second.slot;
    }
  }
  return (ws->num_shards() > 0)
             ? (static_cast<size_t>(shard_idx) % ws->num_shards())
             : 0;
}

absl::Status WeightSynchronizerFfiManager::WriteShardEndpoints(
    int32_t shard_idx, absl::Span<int32_t> out) const {
  const WeightSynchronizerBase* ws = GetSynchronizerForShard(shard_idx);
  if (ws == nullptr) {
    return absl::FailedPreconditionError(absl::StrCat(
        "No WeightSynchronizerBase registered for shard_idx ", shard_idx, "."));
  }
  std::optional<int> port = ws->local_port();
  if (!port.has_value()) {
    return absl::InternalError("WS has no local port assigned.");
  }
  const size_t local_slot = GetLocalSlot(shard_idx);
  const std::vector<std::string> slot_ips = ws->local_ips_for_shard(local_slot);
  if (slot_ips.empty()) {
    return absl::InternalError(absl::StrCat("No local IP serves shard_idx ",
                                            shard_idx, " (local slot ",
                                            local_slot, ")."));
  }
  ShardEndpoints endpoints;
  endpoints.data_ip = slot_ips.front();
  endpoints.data_port = *port;
  endpoints.listener_port = ws->listener_port().value_or(0);
  if (out.size() >= kEndpointsControlIpWidth) {
    // Falls back to the instance's bound IP (not the slot's data IP) so that
    // every slot of one instance reports the same control-plane IP.
    endpoints.control_ip = ResolveControlPlaneIp(host_nics_(), ws->local_ip());
  }
  return endpoints.PackToSpan(out);
}

bool WeightSynchronizerFfiManager::RecordMultiH2dCompletion(size_t num_shards) {
  size_t expected_shards =
      std::max({size_t{1}, initialized_shards_.load(std::memory_order_acquire),
                num_shards});
  size_t completed =
      multi_h2d_completed_shards_.fetch_add(1, std::memory_order_acq_rel) + 1;
  return completed % expected_shards == 0;
}

void WeightSynchronizerFfiManager::Clear(bool delete_instances) {
  absl::flat_hash_set<WeightSynchronizerBase*> to_delete;
  {
    absl::MutexLock lock(mu_);
    ws_map_.clear();
    instances_.clear();
    initialized_shards_.store(0, std::memory_order_relaxed);
    multi_h2d_completed_shards_.store(0, std::memory_order_relaxed);
    for (WeightSynchronizerBase*& ws : shard_table_) {
      if (delete_instances && ws != nullptr) {
        to_delete.insert(ws);
      }
      ws = nullptr;
    }
  }
  for (WeightSynchronizerBase* ws : to_delete) {
    delete ws;
  }
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
  absl::StatusOr<FfiShardDescriptor> desc =
      FfiShardDescriptor::FromSpan(Int32Span(shard_idx_buf), num_shards);
  if (!desc.ok()) {
    return StatusToFfiError(desc.status());
  }
  const int32_t shard_idx = desc->shard_idx;

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

  const bool first_init =
      WeightSynchronizerFfiManager::Default().GetSynchronizerForShard(
          shard_idx) == nullptr;
  if (first_init) {
    VLOG(1)
        << "[TPU Worker FFI] >>> WS LAZY INITIALIZATION TRIGGERED <<< Shard: "
        << shard_idx << ", Local Slot: " << desc->local_slot
        << ", Submanager: " << desc->submanager_idx
        << ", Global Shard: " << desc->global_shard_idx;
  }

  absl::Status init_status =
      WeightSynchronizerFfiManager::Default()
          .GetOrCreateSynchronizer(*desc, listener_port, num_layers,
                                   parallelism, slice_byte_sizes, local_port,
                                   num_shards)
          .status();
  if (!init_status.ok()) {
    return StatusToFfiError(init_status);
  }

  if (first_init && g_streams[shard_idx] == nullptr) {
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

  absl::Status endpoints_status =
      WeightSynchronizerFfiManager::Default().WriteShardEndpoints(
          shard_idx, absl::MakeSpan(static_cast<int32_t*>(out->untyped_data()),
                                    out->element_count()));
  if (!endpoints_status.ok()) {
    return StatusToFfiError(endpoints_status);
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
  absl::StatusOr<FfiShardDescriptor> desc =
      FfiShardDescriptor::FromSpan(Int32Span(shard_idx_buf), num_shards);
  if (!desc.ok()) {
    return StatusToFfiError(desc.status());
  }
  const int32_t shard_idx = desc->shard_idx;

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

  const bool first_init =
      WeightSynchronizerFfiManager::Default().GetSynchronizerForShard(
          shard_idx) == nullptr;
  if (first_init) {
    VLOG(1)
        << "[TPU Worker FFI] >>> WS LAZY INITIALIZATION TRIGGERED <<< Shard: "
        << shard_idx << ", Local Slot: " << desc->local_slot
        << ", Submanager: " << desc->submanager_idx
        << ", Global Shard: " << desc->global_shard_idx;
  }

  absl::Status init_status =
      WeightSynchronizerFfiManager::Default()
          .GetOrCreateSynchronizer(*desc, listener_port, num_layers,
                                   parallelism, slice_byte_sizes, local_port,
                                   num_shards)
          .status();
  if (!init_status.ok()) {
    return StatusToFfiError(init_status);
  }

  if (first_init && g_streams[shard_idx] == nullptr) {
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
  absl::Status endpoints_status =
      WeightSynchronizerFfiManager::Default().WriteShardEndpoints(
          shard_idx, absl::MakeSpan(static_cast<int32_t*>(out->untyped_data()),
                                    out->element_count()));
  if (!endpoints_status.ok()) {
    return StatusToFfiError(endpoints_status);
  }

  // --- D2H Part (Loop through all passed layers) ---
  absl::Time d2h_start = absl::Now();
  WeightSynchronizerBase* ws =
      WeightSynchronizerFfiManager::Default().GetSynchronizerForShard(
          shard_idx);
  size_t local_slot =
      WeightSynchronizerFfiManager::Default().GetLocalSlot(shard_idx);

  for (size_t i = 0; i < jax_arrays.size(); ++i) {
    auto anchor = jax_arrays[i];

    size_t size = ws->block_bytes(i);
    absl::StatusOr<absl::Span<uint8_t>> host_buf =
        CheckedLocalHostBuffer(*ws, i, local_slot, size);
    if (!host_buf.ok()) {
      return StatusToFfiError(host_buf.status());
    }
    uint8_t* dst_host_ptr = host_buf->data();
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
  auto& d2h_store = telemetry::RaidenMetricStore::GetGlobalMetricStore();
  if (d2h_store.HasBackends()) {
    double d2h_time_ms = absl::ToDoubleMilliseconds(absl::Now() - d2h_start);
    d2h_store.ObserveHistogram(
        telemetry::metric_names::kWeightSyncD2hTransferTimeMs, {}, d2h_time_ms);
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
// transfers from local staging buffers (`GetHostBufferWithLocalIndex`) directly
// onto device memory buffers. Uses multi-host local slot mapping
// (`GetLocalSlot(shard_idx)`) to ensure global shard indices map to the
// correct local staging slot.
xla::ffi::Error TriggerH2DImpl(xla::ffi::AnyBuffer shard_idx_buf,
                               int32_t layer_idx,
                               xla::ffi::Result<xla::ffi::AnyBuffer> out) {
  if (shard_idx_buf.untyped_data() == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInvalidArgument,
                           "shard_idx_buf null.");
  }
  int32_t shard_idx =
      *reinterpret_cast<const int32_t*>(shard_idx_buf.untyped_data());
  WeightSynchronizerBase* ws =
      WeightSynchronizerFfiManager::Default().GetSynchronizerForShard(
          shard_idx);
  if (ws == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           "WS not initialized.");
  }

  absl::Time h2d_start = absl::Now();
  size_t size = ws->block_bytes(layer_idx);
  size_t local_slot =
      WeightSynchronizerFfiManager::Default().GetLocalSlot(shard_idx);
  absl::StatusOr<absl::Span<uint8_t>> host_buf =
      CheckedLocalHostBuffer(*ws, layer_idx, local_slot, size);
  if (!host_buf.ok()) {
    return StatusToFfiError(host_buf.status());
  }
  const uint8_t* src_host_ptr = host_buf->data();

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
  auto& h2d_store = telemetry::RaidenMetricStore::GetGlobalMetricStore();
  if (h2d_store.HasBackends()) {
    double h2d_time_ms = absl::ToDoubleMilliseconds(absl::Now() - h2d_start);
    h2d_store.ObserveHistogram(
        telemetry::metric_names::kWeightSyncH2dTransferTimeMs, {}, h2d_time_ms);
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
  WeightSynchronizerBase* ws =
      WeightSynchronizerFfiManager::Default().GetSynchronizerForShard(
          shard_idx);
  if (ws == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           "WS not initialized.");
  }

  absl::Time h2d_start = absl::Now();
  size_t num_layers = rets.size();
  size_t local_slot =
      WeightSynchronizerFfiManager::Default().GetLocalSlot(shard_idx);

  for (size_t i = 0; i < num_layers; ++i) {
    auto ret_or = rets.get<xla::ffi::AnyBuffer>(i);
    if (!ret_or.has_value()) {
      return ret_or.error();
    }
    xla::ffi::AnyBuffer out = **ret_or;
    uint8_t* d_base = reinterpret_cast<uint8_t*>(out.untyped_data());

    size_t size = ws->block_bytes(i);
    absl::StatusOr<absl::Span<uint8_t>> host_buf =
        CheckedLocalHostBuffer(*ws, i, local_slot, size);
    if (!host_buf.ok()) {
      return StatusToFfiError(host_buf.status());
    }
    const uint8_t* src_host_ptr = host_buf->data();

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
  auto& h2d_store = telemetry::RaidenMetricStore::GetGlobalMetricStore();
  if (h2d_store.HasBackends()) {
    double h2d_time_ms = absl::ToDoubleMilliseconds(absl::Now() - h2d_start);
    h2d_store.ObserveHistogram(
        telemetry::metric_names::kWeightSyncH2dTransferTimeMs, {}, h2d_time_ms);
    if (WeightSynchronizerFfiManager::Default().RecordMultiH2dCompletion(
            ws->num_shards())) {
      h2d_store.FlushToCloudLogging("destination_h2d", 0);
    }
  }
  return xla::ffi::Error::Success();
}

// FFI custom call handler executing asynchronous Device-to-Host (D2H) memory
// transfers from device memory directly onto local staging buffers
// (`GetHostBufferWithLocalIndex`).
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
  WeightSynchronizerBase* ws =
      WeightSynchronizerFfiManager::Default().GetSynchronizerForShard(
          shard_idx);
  if (ws == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           "WS not initialized.");
  }

  absl::Time d2h_start = absl::Now();
  size_t size = ws->block_bytes(layer_idx);
  size_t local_slot =
      WeightSynchronizerFfiManager::Default().GetLocalSlot(shard_idx);
  absl::StatusOr<absl::Span<uint8_t>> host_buf =
      CheckedLocalHostBuffer(*ws, layer_idx, local_slot, size);
  if (!host_buf.ok()) {
    return StatusToFfiError(host_buf.status());
  }
  uint8_t* dst_host_ptr = host_buf->data();
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
  auto& d2h_store = telemetry::RaidenMetricStore::GetGlobalMetricStore();
  if (d2h_store.HasBackends()) {
    double d2h_time_ms = absl::ToDoubleMilliseconds(absl::Now() - d2h_start);
    d2h_store.ObserveHistogram(
        telemetry::metric_names::kWeightSyncD2hTransferTimeMs, {}, d2h_time_ms);
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
