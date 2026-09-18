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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_CONTROL_PIPE_TYPES_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_CONTROL_PIPE_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

namespace tpu_raiden {

enum class ControlPipeBackendType {
  kTcp,
  kGrpc,
};

using TaskExecutor = std::function<void(std::function<void()>)>;

// Metadata passed to every registered request handler.
struct ControlContext {
  // Remote client IP address or transport peer identifier.
  std::string peer_ip;

  // Monotonically increasing client-assigned request correlation ID.
  uint64_t request_id = 0;

  // Transport backend over which the request was received.
  ControlPipeBackendType backend_type = ControlPipeBackendType::kTcp;

  // Optional absolute deadline propagated from the caller.
  absl::Time deadline = absl::InfiniteFuture();

  // Optional key-value metadata headers attached to the request envelope.
  absl::flat_hash_map<std::string, std::string> metadata;
};

struct ControlPipeConfig {
  // Active transport backend (`kTcp`, `kGrpc`, or `kZmq`).
  ControlPipeBackendType backend_type = ControlPipeBackendType::kTcp;

  // Local TCP/gRPC/ZMQ port to bind on `Start()` (`0` binds an OS-assigned
  // ephemeral port).
  int requested_port = 0;

  // Default per-request send/receive deadline when callers pass
  // `absl::ZeroDuration()`.
  absl::Duration default_timeout = absl::Seconds(120);

  // Maximum allowed inbound/outbound wire frame size in bytes (Tier-1
  // transport guardrail; defaults to 64 MiB).
  size_t max_frame_bytes = 64 * 1024 * 1024;

  // Whether `TcpControlPipeClient` caches and reuses open sockets per peer
  // endpoint (`ip:port`).
  bool enable_tcp_connection_pooling = true;

  // Maximum number of idle pooled connections retained per target endpoint
  // (`ip:port`).
  size_t max_idle_connections_per_endpoint = 4;

  // Enables 3-way TCP magic-prefix auto-detection (`"CPIP"` envelope, `"RAID"`
  // binary pull-stream/ack frames, and raw length-prefixed `ControlRequest`).
  bool allow_legacy_framing = true;

  // Optional custom or NUMA-aware thread pool callback for offloading accepted
  // connection/request handlers.
  TaskExecutor executor = nullptr;

  // Optional callback returning the maximum allowed block count in legacy
  // `"RAID"` pull-stream headers before vector allocation.
  std::function<uint64_t()> max_legacy_pull_blocks_fn = nullptr;
};

// Resolves the active backend type from:
// 1. Explicit programmatic `override_type` if provided.
// 2. `TPU_RAIDEN_CONTROL_PLANE_BACKEND` environment variable ("grpc" or "tcp").
// 3. `TPU_RAIDEN_USE_GRPC_CONTROL_PLANE` environment variable ("1" or "true").
// 4. Default: ControlPipeBackendType::kTcp.
ControlPipeBackendType ResolveControlPipeBackendType(
    std::optional<ControlPipeBackendType> override_type = std::nullopt);

// Returns a string representation of `type` ("tcp" or "grpc").
absl::string_view ControlPipeBackendTypeName(ControlPipeBackendType type);

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_CONTROL_PIPE_TYPES_H_
