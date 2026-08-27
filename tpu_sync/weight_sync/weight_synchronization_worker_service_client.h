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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SYNCHRONIZATION_WORKER_SERVICE_CLIENT_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SYNCHRONIZATION_WORKER_SERVICE_CLIENT_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/rpc/raiden_service.grpc.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace weight_sync {

// Client for interacting with the WeightSynchronizationWorkerService gRPC
// endpoint on a remote weight synchronization worker daemon asynchronously.
class WeightSynchronizationWorkerServiceClient {
 public:
  explicit WeightSynchronizationWorkerServiceClient(
      std::shared_ptr<::grpc::ChannelInterface> channel);
  explicit WeightSynchronizationWorkerServiceClient(
      std::unique_ptr<
          ::tpu_sync::rpc::WeightSynchronizationWorkerService::StubInterface>
          stub);
  explicit WeightSynchronizationWorkerServiceClient(absl::string_view target);

  // Asynchronous HandleControl RPC.
  // Returns a Future that resolves with the ControlResponse upon completion.
  tsl::Future<::tpu_sync::rpc::ControlResponse> HandleControl(
      const ::tpu_sync::rpc::ControlRequest& request,
      absl::Duration timeout = absl::InfiniteDuration());

  // Convenience helper to initiate weight transfer (Sender or Receiver mode)
  // asynchronously.
  tsl::Future<::tpu_sync::rpc::ControlResponse> StartTransfer(
      const ::tpu_sync::rpc::StartTransferRequest& start_transfer_request,
      const std::vector<std::string>& peers = {},
      absl::Duration timeout = absl::InfiniteDuration());

  // Convenience helper to instruct the remote worker to shut down cleanly
  // asynchronously.
  tsl::Future<::tpu_sync::rpc::ControlResponse> Shutdown(
      absl::Duration timeout = absl::InfiniteDuration());

 private:
  std::unique_ptr<
      ::tpu_sync::rpc::WeightSynchronizationWorkerService::StubInterface>
      stub_;
};

}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_WEIGHT_SYNCHRONIZATION_WORKER_SERVICE_CLIENT_H_
