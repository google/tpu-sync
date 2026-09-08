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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TRANSFER_CONTROL_SERVER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TRANSFER_CONTROL_SERVER_H_

#include <atomic>
#include <memory>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/channel_arguments.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/proto/transfer_control.grpc.pb.h"
#include "tpu_sync/proto/transfer_control.pb.h"

namespace tpu_raiden {

// Pure virtual interface breaking cyclic dependencies between server and
// manager.
class TransferControlDelegate {
 public:
  virtual ~TransferControlDelegate() = default;

  virtual absl::Status HandleGrpcPullStream(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::PullStreamRequest& request,
      ::tpu_sync::proto::PullStreamResponse* response) = 0;

  virtual absl::Status HandleGrpcAck(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::TransferAckRequest& request,
      ::tpu_sync::proto::TransferAckResponse* response) = 0;

  virtual absl::Status HandleGrpcCheckLiveness(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::TransferLivenessRequest& request,
      ::tpu_sync::proto::TransferLivenessResponse* response) = 0;
};

// Implementation of the TransferControlService gRPC service.
class TransferControlServiceImpl final
    : public ::tpu_sync::proto::TransferControlService::Service {
 public:
  explicit TransferControlServiceImpl(TransferControlDelegate* delegate);

  grpc::Status PullStream(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::PullStreamRequest* request,
      ::tpu_sync::proto::PullStreamResponse* response) override;

  grpc::Status Ack(grpc::ServerContext* context,
                   const ::tpu_sync::proto::TransferAckRequest* request,
                   ::tpu_sync::proto::TransferAckResponse* response) override;

  grpc::Status CheckLiveness(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::TransferLivenessRequest* request,
      ::tpu_sync::proto::TransferLivenessResponse* response) override;

 private:
  TransferControlDelegate* const delegate_;
};

// gRPC Server Daemon for the Transfer Control Plane.
class TransferControlServer final {
 public:
  static constexpr int kInProcessPort = -1;
  static constexpr int kDisabledPort = -2;

  TransferControlServer(TransferControlDelegate* delegate, int port);
  ~TransferControlServer();

  TransferControlServer(const TransferControlServer&) = delete;
  TransferControlServer& operator=(const TransferControlServer&) = delete;

  int port() const { return port_; }
  bool is_running() const { return !stopping_.load(); }

  // Returns an in-process channel when running in in-process mode (port == -1).
  std::shared_ptr<grpc::Channel> InProcessChannel(
      const grpc::ChannelArguments& args = grpc::ChannelArguments());

  void Shutdown(absl::Duration timeout = absl::Seconds(5));

 private:
  TransferControlDelegate* delegate_;
  int port_ = 0;
  std::atomic<bool> stopping_{false};

  std::unique_ptr<TransferControlServiceImpl> service_impl_;
  std::unique_ptr<grpc::Server> grpc_server_;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TRANSFER_CONTROL_SERVER_H_
