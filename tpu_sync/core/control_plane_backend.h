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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_CONTROL_PLANE_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_CONTROL_PLANE_BACKEND_H_

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {

enum class ControlPlaneBackendType {
  kTcp,
  kGrpc,
};

struct PullStreamRequestSpec {
  uint64_t uuid = 0;
  uint32_t ep_idx = 0;
  uint32_t consumer_data_port = 0;
  std::vector<std::string> consumer_ips;
  std::vector<int64_t> src_block_ids;
  std::vector<int64_t> dst_block_ids;
  // Consumer host addresses, one per layer. Empty if not reported.
  std::vector<::tpu_sync::rpc::PoolHostAddrsProto> layer_host_addrs;
  int32_t chunk_blocks = 0;
  int32_t window_chunks = 0;
};

inline constexpr int64_t kMaxWindowedPullBlocks = 1 << 20;

struct ControlPlaneStagingWindowAck {
  int32_t chunk_blocks = 0;
  int32_t window_chunks = 0;
  bool operator==(const ControlPlaneStagingWindowAck&) const = default;
};

struct PullStreamResponseSpec {
  int32_t status = 0;
  uint32_t num_layers = 0;
  uint32_t data_port = 0;
  std::string message;
  std::optional<ControlPlaneStagingWindowAck> staging_window_ack;
  std::optional<std::string> staging_window_rejected_reason;
};

struct ControlPlaneCapabilities {
  bool supported = true;
  bool serve_staging_window = false;
  int64_t max_ring_blocks = 0;
  bool operator==(const ControlPlaneCapabilities&) const = default;
};
inline constexpr absl::string_view kStagingWindowRejectedPrefix =
    "staging_window_rejected:";
inline constexpr absl::string_view kStagingWindowAckPrefix = "staging_window=";
inline std::string FormatStagingWindowRejected(absl::string_view reason) {
  return absl::StrCat(kStagingWindowRejectedPrefix, reason);
}
inline std::optional<std::string> ParseStagingWindowRejectedReason(
    absl::string_view msg) {
  if (!absl::StartsWith(msg, kStagingWindowRejectedPrefix) ||
      msg.size() == kStagingWindowRejectedPrefix.size()) {
    return std::nullopt;
  }
  return std::string(msg.substr(kStagingWindowRejectedPrefix.size()));
}
inline std::string FormatStagingWindowAck(int32_t c, int32_t w) {
  return absl::StrCat(kStagingWindowAckPrefix, c, ",", w);
}
inline std::optional<std::pair<int32_t, int32_t>> ParseStagingWindowAck(
    absl::string_view msg) {
  if (!absl::StartsWith(msg, kStagingWindowAckPrefix)) return std::nullopt;
  std::pair<absl::string_view, absl::string_view> p = absl::StrSplit(
      msg.substr(kStagingWindowAckPrefix.size()), absl::MaxSplits(',', 1));
  int32_t c = 0, w = 0;
  if (!absl::SimpleAtoi(p.first, &c) || !absl::SimpleAtoi(p.second, &w) ||
      c <= 0 || w <= 0) {
    return std::nullopt;
  }
  return std::make_pair(c, w);
}
inline std::string FormatCapabilitiesReply(const ControlPlaneCapabilities& c) {
  return c.serve_staging_window
             ? absl::StrCat("caps=staging_window:serve;max_ring_blocks=",
                            c.max_ring_blocks)
             : "caps=";
}
inline absl::StatusOr<ControlPlaneCapabilities> ParseCapabilitiesReply(
    absl::string_view text) {
  if (!absl::StartsWith(text, "caps=")) {
    return absl::InvalidArgumentError(
        absl::StrCat("missing 'caps=' prefix: ", text));
  }
  ControlPlaneCapabilities caps;
  for (absl::string_view token :
       absl::StrSplit(text.substr(5), ';', absl::SkipEmpty())) {
    if (token == "staging_window:serve") {
      caps.serve_staging_window = true;
    } else if (absl::StartsWith(token, "max_ring_blocks=")) {
      if (!absl::SimpleAtoi(token.substr(16), &caps.max_ring_blocks) ||
          caps.max_ring_blocks < 0) {
        return absl::InvalidArgumentError(
            absl::StrCat("invalid max_ring_blocks: ", token));
      }
    }
  }
  return caps;
}
inline bool IsCapabilitiesUnsupportedStatus(const absl::Status& status) {
  return absl::IsUnimplemented(status) ||
         absl::StrContains(status.message(), "unknown control op code");
}

// Callback interface implemented by KVCacheManagerWithTransfer to handle
// incoming control plane requests from remote consumers.
class ControlPlaneHandler {
 public:
  virtual ~ControlPlaneHandler() = default;

  // Handles an incoming PullStream request. Application/validation rejections
  // (such as expired UUIDs or invalid block IDs) are caught and returned as
  // absl::OkStatus() with PullStreamResponseSpec{.status = -1, .message = ...}.
  // Non-OK absl::Status is reserved for unrecoverable internal handler faults.
  virtual absl::StatusOr<PullStreamResponseSpec> OnPullStream(
      const PullStreamRequestSpec& req, absl::string_view fallback_peer_ip) = 0;

  // Handles an incoming empty pull stream / Ack notification (calls AckSend).
  virtual absl::Status OnAck(uint64_t uuid) = 0;

  virtual ControlPlaneCapabilities GetCapabilities() const { return {}; }

  // Returns the maximum number of blocks allowed in a single PullStream
  // request. Used by TcpControlPlaneBackend to reject oversized headers before
  // reading untrusted bodies from sockets.
  virtual uint64_t MaxPullStreamBlocks() const {
    return std::numeric_limits<uint64_t>::max();
  }
};

// Abstract strategy interface for the KV Cache control plane transport.
class ControlPlaneBackend {
 public:
  using TaskExecutor = std::function<void(std::function<void()>)>;

  virtual ~ControlPlaneBackend() = default;

  // Starts the control plane server listening on `requested_port` (0 for
  // ephemeral port assignment). Returns the actual bound port.
  virtual absl::StatusOr<int> StartServer(int requested_port,
                                          ControlPlaneHandler* handler) = 0;

  // Stops the control plane server and waits for all in-flight connection
  // handlers to complete before returning.
  virtual void StopServer() = 0;

  // Sends a PullStream handshake request to `remote_endpoint` ("host:port").
  // The returned future resolves with a non-OK Status on a transport/RPC error
  // (e.g. connection refused or deadline exceeded). Application rejections
  // resolve OK with status != 0.
  //
  // Backends with an async client return without waiting for the response,
  // and the future may then resolve on a transport thread, so callbacks passed
  // to OnReady must not block. Backends without one return a ready future.
  virtual tsl::Future<PullStreamResponseSpec> SendPullRequest(
      absl::string_view remote_endpoint, const PullStreamRequestSpec& req,
      absl::Duration timeout) = 0;

  // Sends an Ack notification to `remote_endpoint` ("host:port").
  virtual absl::Status SendAck(absl::string_view remote_endpoint, uint64_t uuid,
                               absl::Duration timeout) = 0;

  virtual tsl::Future<ControlPlaneCapabilities> GetCapabilities(
      absl::string_view remote_endpoint, absl::Duration timeout) = 0;

  virtual absl::StatusOr<ControlPlaneCapabilities> GetCapabilitiesBlocking(
      absl::string_view remote_endpoint, absl::Duration timeout) {
    auto res = GetCapabilities(remote_endpoint, timeout).Await();
    if (res.ok()) return *res;
    return IsCapabilitiesUnsupportedStatus(res.status())
               ? ControlPlaneCapabilities{.supported = false}
               : absl::StatusOr<ControlPlaneCapabilities>(res.status());
  }

  virtual absl::string_view Name() const = 0;
};

// Resolves the active backend type from optional programmatic override,
// environment variables, or defaults.
ControlPlaneBackendType ResolveControlPlaneBackendType(
    std::optional<ControlPlaneBackendType> override_type = std::nullopt);

// Factory function creating the requested ControlPlaneBackend instance.
// `executor` is used by TcpControlPlaneBackend to dispatch accepted sockets
// onto pull_pool_ with NUMA node affinity.
std::unique_ptr<ControlPlaneBackend> CreateControlPlaneBackend(
    ControlPlaneBackendType type,
    ControlPlaneBackend::TaskExecutor executor = nullptr,
    absl::Duration default_timeout = absl::Seconds(120));

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_CONTROL_PLANE_BACKEND_H_
