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

#include "tpu_sync/core/controller/worker_service_server.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server_builder.h"
#include "tpu_sync/core/controller/worker_service_client.h"
#include "tpu_sync/core/controller/worker_service_impl.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/core/kv_manager_holder.h"

namespace tpu_raiden {
namespace controller {

WorkerServiceServer& WorkerServiceServer::GetInstance() {
  static WorkerServiceServer* instance = new WorkerServiceServer();
  return *instance;
}

std::unique_ptr<WorkerServiceServer> WorkerServiceServer::Create() {
  return std::unique_ptr<WorkerServiceServer>(new WorkerServiceServer());
}

WorkerServiceServer::~WorkerServiceServer() {
  absl::MutexLock lock(mutex_);
  if (grpc_server_) {
    grpc_server_->Shutdown();
  }
}

absl::Status WorkerServiceServer::StartServer(
    std::shared_ptr<HostMemoryAllocator> host_allocator,
    KVManagerHolder transfer_manager, int port) {
  if (port < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid gRPC port: ", port));
  }

  absl::MutexLock lock(mutex_);
  if (started_) {
    if (transfer_manager && worker_service_) {
      worker_service_->SetTransferManager(std::move(transfer_manager));
    }
    return absl::OkStatus();
  }

  worker_service_ = std::make_unique<WorkerServiceImpl>(
      std::move(host_allocator), std::move(transfer_manager));

  std::string server_address = absl::StrCat("[::]:", port);
  grpc::ServerBuilder builder;
  // SubmitTransferProgram carries the fully expanded copy plan (one
  // TransferStep per live chunk, every source rank for receiver arms); long
  // prefills exceed gRPC's 4 MiB default. Match the control-plane channels.
  builder.SetMaxReceiveMessageSize(kMaxWorkerServiceMessageBytes);
  builder.SetMaxSendMessageSize(kMaxWorkerServiceMessageBytes);
  int selected_port = 0;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(),
                           &selected_port);
  builder.RegisterService(worker_service_.get());
  grpc_server_ = builder.BuildAndStart();

  if (!grpc_server_ || selected_port == 0) {
    grpc_server_.reset();
    worker_service_.reset();
    return absl::InternalError(
        absl::StrCat("Failed to start gRPC server on port: ", port, " (",
                     server_address, ")"));
  }

  raiden_worker_port_ = selected_port;
  started_ = true;
  return absl::OkStatus();
}

void WorkerServiceServer::SetTransferManager(KVManagerHolder transfer_manager) {
  absl::MutexLock lock(mutex_);
  if (worker_service_) {
    worker_service_->SetTransferManager(std::move(transfer_manager));
  }
}

int WorkerServiceServer::GetRaidenWorkerPort() const {
  absl::MutexLock lock(mutex_);
  return raiden_worker_port_;
}

}  // namespace controller
}  // namespace tpu_raiden
