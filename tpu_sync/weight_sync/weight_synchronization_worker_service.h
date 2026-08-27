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

// Branched from:
// //tpu_sync/weight_sync/weight_synchronizer_listener.h

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SYNCHRONIZATION_WORKER_SERVICE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SYNCHRONIZATION_WORKER_SERVICE_H_

#include <atomic>
#include <functional>
#include <memory>

#include "grpcpp/server.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/rpc/raiden_service.grpc.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace weight_sync {

// Implementation of the WeightSynchronizationWorkerService gRPC service.
class WeightSynchronizationWorkerServiceImpl final
    : public ::tpu_sync::rpc::WeightSynchronizationWorkerService::Service {
 public:
  explicit WeightSynchronizationWorkerServiceImpl(
      WeightSynchronizerBase* engine,
      std::function<void()> shutdown_callback = nullptr);

  grpc::Status HandleControl(
      grpc::ServerContext* context,
      const ::tpu_sync::rpc::ControlRequest* request,
      ::tpu_sync::rpc::ControlResponse* response) override;

 private:
  WeightSynchronizerBase* engine_;
  std::function<void()> shutdown_callback_;
};

// gRPC Server Daemon that runs natively in C++ to accept Control-Plane
// management RPC commands (such as PushWeights and Shutdown) directly over
// gRPC.
class WeightSynchronizationWorkerService final {
 public:
  WeightSynchronizationWorkerService(WeightSynchronizerBase* engine,
                                     int server_port);
  ~WeightSynchronizationWorkerService();

  WeightSynchronizationWorkerService(
      const WeightSynchronizationWorkerService&) = delete;
  WeightSynchronizationWorkerService& operator=(
      const WeightSynchronizationWorkerService&) = delete;

  int server_port() const { return server_port_; }
  int listener_port() const { return server_port_; }
  bool is_active() const { return !stopping_; }

  void Shutdown();

 private:
  WeightSynchronizerBase* engine_;
  int server_port_ = 0;
  std::atomic<bool> stopping_{false};

  std::unique_ptr<WeightSynchronizationWorkerServiceImpl> service_impl_;
  std::unique_ptr<::grpc::Server> grpc_server_;
};

}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SYNCHRONIZATION_WORKER_SERVICE_H_
