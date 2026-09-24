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
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/tsl/concurrency/future.h"

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
};

struct PullStreamResponseSpec {
  int32_t status = 0;
  uint32_t num_layers = 0;
  uint32_t data_port = 0;
  std::string message;
};

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
