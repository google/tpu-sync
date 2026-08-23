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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_WORKER_SERVICE_CLIENT_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_WORKER_SERVICE_CLIENT_H_

#include <memory>

#include "absl/status/statusor.h"
#include "grpcpp/channel.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/proto/worker_service.grpc.pb.h"
#include "tpu_sync/proto/worker_service.pb.h"

namespace tpu_raiden {
namespace controller {

// Client for interacting with the WorkerService gRPC endpoint on a transfer
// worker asynchronously.
class WorkerServiceClient {
 public:
  explicit WorkerServiceClient(std::shared_ptr<grpc::Channel> channel);

  // Allocates sharded buffers on the remote transfer worker asynchronously.
  tsl::Future<::tpu_sync::proto::CreateBuffersResponse> CreateBuffers(
      const ::tpu_sync::proto::CreateBuffersRequest& request);

  // Deallocates sharded buffers on the remote transfer worker asynchronously.
  tsl::Future<::tpu_sync::proto::DeleteBuffersResponse> DeleteBuffers(
      const ::tpu_sync::proto::DeleteBuffersRequest& request);

  // Transfers (copies) disjoint memory regions across memory spaces on the
  // remote transfer worker asynchronously. The transfer specification applies
  // uniformly across all buffers, i.e., all shards and major dimensions
  // (layers or blocks).
  tsl::Future<> TransferBuffers(
      const ::tpu_sync::proto::TransferBuffersRequest& request);

  // Registers storage backends on the remote transfer worker asynchronously.
  tsl::Future<::tpu_sync::proto::RegisterBackendsResponse> RegisterBackends(
      const ::tpu_sync::proto::RegisterBackendsRequest& request);

  // Submits a transfer program and resolves with the full response. The
  // reshard coordinator needs success + message verbatim for its
  // abandon-claim contract, so admission verdicts are not collapsed into a
  // bare Status here. Transport/dispatch failures, including unsupported
  // completion contracts, surface as error statuses.
  tsl::Future<::tpu_sync::proto::TransferProgramResponse> SubmitTransferProgram(
      const ::tpu_sync::proto::TransferProgramRequest& request);

 private:
  std::unique_ptr<::tpu_sync::proto::WorkerService::Stub> stub_;
};

}  // namespace controller
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_WORKER_SERVICE_CLIENT_H_
