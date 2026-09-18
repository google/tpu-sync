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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_ZMQ_CONTROL_PIPE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_ZMQ_CONTROL_PIPE_H_

#include <deque>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
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

class ZmqControlPipeServer final : public ControlPipeServer {
 public:
  explicit ZmqControlPipeServer(const ControlPipeConfig& config);
  ~ZmqControlPipeServer() override;

  absl::StatusOr<int> Start(int requested_port) override;
  void Stop() override;

  int bound_port() const override { return bound_port_; }
  ControlDispatcher& dispatcher() override { return dispatcher_; }
  ControlPipeBackendType backend_type() const override {
    return ControlPipeBackendType::kZmq;
  }

 private:
  void PollLoop();
  std::string ProcessPayload(absl::string_view payload);

  ControlPipeConfig config_;
  ControlDispatcher dispatcher_;
  void* zmq_ctx_ = nullptr;
  void* router_socket_ = nullptr;
  void* inproc_reply_pull_ = nullptr;
  int bound_port_ = 0;
  std::string reply_inproc_addr_;
  std::thread poll_thread_;

  absl::Mutex mu_;
  bool stopping_ ABSL_GUARDED_BY(mu_) = false;
  int active_handlers_ ABSL_GUARDED_BY(mu_) = 0;
};

class ZmqConnectionPool {
 public:
  ZmqConnectionPool(void* zmq_ctx, const ControlPipeConfig& config);
  ~ZmqConnectionPool();

  absl::StatusOr<void*> Acquire(absl::string_view endpoint,
                                absl::Duration timeout);
  void Release(absl::string_view endpoint, void* sock);
  void Discard(void* sock);
  void CloseAll();

 private:
  void* zmq_ctx_;
  ControlPipeConfig config_;
  absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::deque<void*>> pool_
      ABSL_GUARDED_BY(mu_);
};

class ZmqControlPipeClient final : public ControlPipeClient {
 public:
  explicit ZmqControlPipeClient(const ControlPipeConfig& config);
  ~ZmqControlPipeClient() override;

  absl::StatusOr<control_pipe::proto::ControlResponseEnvelope> SendRaw(
      absl::string_view endpoint,
      const control_pipe::proto::ControlEnvelope& envelope,
      absl::Duration timeout) override;

  ControlPipeBackendType backend_type() const override {
    return ControlPipeBackendType::kZmq;
  }

 private:
  ControlPipeConfig config_;
  void* zmq_ctx_ = nullptr;
  std::unique_ptr<ZmqConnectionPool> pool_;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_ZMQ_CONTROL_PIPE_H_
