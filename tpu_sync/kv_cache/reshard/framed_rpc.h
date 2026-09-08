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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_FRAMED_RPC_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_FRAMED_RPC_H_

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {

// Client side of the 4-byte big-endian length-framed proto wire spoken by
// the Python RaidenController stack and the C++ KVCacheListener. One
// connection per call, mirroring rpc/raiden_controller.py's
// WorkerRpcClient._send_rpc_sync (connection reuse is
// a future optimization).
class FramedTransport {
 public:
  virtual ~FramedTransport() = default;

  // Connects to "host:port" (bracketed IPv6 allowed), sends the framed
  // payload, and returns the framed response body. `timeout` bounds the
  // whole call including the Python-compatible connect retry loop.
  virtual absl::StatusOr<std::string> Call(absl::string_view address,
                                           absl::string_view payload,
                                           absl::Duration timeout) = 0;
};

class SocketFramedTransport final : public FramedTransport {
 public:
  absl::StatusOr<std::string> Call(absl::string_view address,
                                   absl::string_view payload,
                                   absl::Duration timeout) override;
};

// Framed-TCP server: dual-stack listener, one thread per accepted
// connection, one request/response exchange per connection (the accept
// model of RaidenControllerServer and KVCacheListener). The handler
// receives the raw request body and returns the raw response body.
class FramedServer final {
 public:
  using Handler = std::function<std::string(const std::string&)>;

  // Binds immediately (port 0 selects an ephemeral port); serving starts
  // with Start(). Fatal bind errors surface from Bind(), not the ctor.
  FramedServer(int port, Handler handler);
  ~FramedServer();

  FramedServer(const FramedServer&) = delete;
  FramedServer& operator=(const FramedServer&) = delete;

  // Creates and binds the listener socket. Must be called before Start().
  absl::Status Bind();
  void Start();
  void Stop();

  // The actually bound port (resolves an ephemeral request).
  int port() const { return port_; }

 private:
  // One handler thread per accepted connection; `done` lets the accept loop
  // reap finished threads so a long-lived server does not accumulate one
  // un-joined thread (and its stack) per request.
  struct Connection {
    std::thread thread;
    std::atomic<bool> done{false};
  };

  void AcceptLoop();
  void ServeConnection(int client_fd);

  int requested_port_;
  int port_ = 0;
  int server_fd_ = -1;
  Handler handler_;
  std::atomic<bool> stopping_{false};
  std::thread accept_thread_;
  std::vector<std::unique_ptr<Connection>> connections_;
};

}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_FRAMED_RPC_H_
