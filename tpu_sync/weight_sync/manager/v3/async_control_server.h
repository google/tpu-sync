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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_ASYNC_CONTROL_SERVER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_ASYNC_CONTROL_SERVER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/proto/control_pipe.pb.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {

// Sends the response to one request. Call it exactly once, from any thread.
// Responses to closed connections are dropped.
using AsyncControlReply =
    absl::AnyInvocable<void(control_pipe::proto::ControlResponseEnvelope) &&>;

// Handles one request. It runs on an event-loop thread and must not block:
// work that may block goes to an executor, and requests that wait for
// something (e.g. a long poll) keep |reply| instead of a thread. The server
// sets `request_id` of the response.
using AsyncControlHandler = std::function<void(
    const ControlContext& ctx, control_pipe::proto::ControlEnvelope request,
    AsyncControlReply reply)>;

// A control-plane server whose handlers answer asynchronously, so that a
// request costs no thread while it waits. It speaks the `ControlPipe` TCP
// envelope framing ("CPIP" + 4-byte big-endian length + `ControlEnvelope`,
// answered by "PIPC" + length + `ControlResponseEnvelope`), so
// `ControlPipeClient` can call it. A connection may carry several requests
// at a time; responses carry the request id and may come in any order.
//
// The controller serves the pull RPCs with it. It can be replaced by
// `ControlPipeServer` once that supports asynchronous handlers.
class AsyncControlServer {
 public:
  struct Stats {
    int64_t accepted_connections = 0;
    int32_t open_connections = 0;
    int64_t requests = 0;
    int64_t responses = 0;
    // CPU time of the event-loop threads.
    int64_t loop_cpu_ns = 0;
  };

  virtual ~AsyncControlServer() = default;

  // Binds |requested_port| (0: an ephemeral port) on all interfaces and
  // starts serving. Returns the bound port.
  virtual absl::StatusOr<int> Start(int requested_port) = 0;
  // Closes every connection and stops the event loops. Later replies are
  // dropped.
  virtual void Stop() = 0;
  virtual int bound_port() const = 0;
  virtual Stats GetStats() const = 0;
};

struct EpollControlServerOptions {
  // Event-loop threads (>= 1). Connections are spread over them.
  int32_t num_threads = 1;
  // Larger frames close the connection.
  size_t max_frame_bytes = 64 << 20;
};

// An `AsyncControlServer` on epoll: non-blocking sockets and one epoll loop
// per thread, woken through an eventfd when a reply is ready.
std::unique_ptr<AsyncControlServer> CreateEpollControlServer(
    EpollControlServerOptions options, AsyncControlHandler handler);

}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_ASYNC_CONTROL_SERVER_H_
