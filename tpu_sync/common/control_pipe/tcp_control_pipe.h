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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_TCP_CONTROL_PIPE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_TCP_CONTROL_PIPE_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tpu_sync/common/control_pipe/control_dispatcher.h"
#include "tpu_sync/common/control_pipe/control_pipe_client.h"
#include "tpu_sync/common/control_pipe/control_pipe_server.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/proto/control_pipe.pb.h"

namespace tpu_raiden {

class TcpConnectionPool {
 public:
  TcpConnectionPool(bool enable_pooling, size_t max_idle_per_endpoint);
  ~TcpConnectionPool();

  // Acquires an active TCP socket descriptor to `endpoint` ("host:port").
  // Reuses an idle pooled socket if available and healthy, or dials a new one.
  absl::StatusOr<int> Acquire(absl::string_view endpoint,
                              absl::Duration timeout);

  // Returns a healthy socket descriptor to the pool for future reuse.
  // If pooling is disabled or the endpoint queue is full, closes `fd`.
  void Release(absl::string_view endpoint, int fd);

  // Returns the current number of idle pooled sockets for `endpoint`.
  size_t IdleCount(absl::string_view endpoint) const;

 private:
  bool enable_pooling_;
  size_t max_idle_per_endpoint_;
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::deque<int>> pool_ ABSL_GUARDED_BY(mu_);
};

class TcpControlPipeServer : public ControlPipeServer {
 public:
  static constexpr int kMaxNics = 8;
  static constexpr uint32_t kRaidControlMagic = 0x52414944;   // "RAID"
  static constexpr uint32_t kRaidResponseMagic = 0x44494152;  // "DIAR"
  static constexpr uint32_t kRaidOpAck = 2;
  static constexpr uint32_t kRaidOpPullStream = 3;

  struct alignas(8) LegacyRaidRequestHeader {
    uint32_t magic = kRaidControlMagic;
    uint32_t op = 0;
    uint64_t uuid = 0;
    uint32_t ep_idx = 0;
    uint32_t consumer_data_port = 0;
    uint64_t num_blocks = 0;
    uint32_t num_ips = 0;
    uint8_t consumer_ips[kMaxNics][16] = {{0}};
    uint32_t padding = 0;
  };

  struct alignas(8) LegacyRaidResponseHeader {
    uint32_t magic = kRaidResponseMagic;
    int32_t status = 0;
    uint32_t num_layers = 0;
    uint32_t data_port = 0;
    uint64_t message_len = 0;
  };

  explicit TcpControlPipeServer(const ControlPipeConfig& config);
  ~TcpControlPipeServer() override;

  absl::StatusOr<int> Start(int requested_port) override;
  void Stop() override;

  int bound_port() const override { return bound_port_; }
  ControlDispatcher& dispatcher() override { return dispatcher_; }
  ControlPipeBackendType backend_type() const override {
    return ControlPipeBackendType::kTcp;
  }

 private:
  void AcceptLoop();
  void HandleConnection(int client_fd, std::string peer_ip);
  void HandleLegacyRaidConnection(int client_fd, uint32_t magic_word,
                                  const std::string& peer_ip);
  void SendLegacyRaidErrorResponse(int client_fd, absl::string_view message);

  ControlPipeConfig config_;
  ControlDispatcher dispatcher_;
  int server_fd_ = -1;
  int bound_port_ = 0;
  std::thread accept_thread_;

  mutable absl::Mutex mu_;
  bool stopping_ ABSL_GUARDED_BY(mu_) = false;
  int active_handlers_ ABSL_GUARDED_BY(mu_) = 0;
  absl::flat_hash_set<int> active_client_fds_ ABSL_GUARDED_BY(mu_);
};

class TcpControlPipeClient : public ControlPipeClient {
 public:
  explicit TcpControlPipeClient(const ControlPipeConfig& config);
  ~TcpControlPipeClient() override = default;

  absl::StatusOr<control_pipe::proto::ControlResponseEnvelope> SendRaw(
      absl::string_view endpoint,
      const control_pipe::proto::ControlEnvelope& envelope,
      absl::Duration timeout) override;

  ControlPipeBackendType backend_type() const override {
    return ControlPipeBackendType::kTcp;
  }

  size_t PoolIdleCount(absl::string_view endpoint) const {
    return conn_pool_->IdleCount(endpoint);
  }

 private:
  ControlPipeConfig config_;
  std::unique_ptr<TcpConnectionPool> conn_pool_;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_TCP_CONTROL_PIPE_H_
