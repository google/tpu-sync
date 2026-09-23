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

#include "tpu_sync/core/controller/controller_client.h"

#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/string_view.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/common/grpc_util.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/proto/controller_service.grpc.pb.h"
#include "tpu_sync/proto/controller_service.pb.h"

namespace tpu_raiden {
namespace core {
namespace controller {

RaidenControllerClient::RaidenControllerClient(
    std::shared_ptr<grpc::Channel> channel)
    : stub_(::tpu_sync::proto::RaidenControllerService::NewStub(channel)) {}

RaidenControllerClient::RaidenControllerClient(absl::string_view endpoint)
    : stub_(::tpu_sync::proto::RaidenControllerService::NewStub(
          grpc::CreateChannel(std::string(endpoint),
                              grpc::InsecureChannelCredentials()))) {}

absl::Status RaidenControllerClient::RegisterWorker(
    absl::string_view worker_id, absl::string_view raiden_worker_endpoint,
    const std::vector<::tpu_raiden::RaidenTransferEndpoint>&
        raiden_transfer_endpoints,
    int64_t node_id, const std::vector<uint64_t>& block_array_bytes,
    int32_t num_kv_shards) {
  ::tpu_sync::proto::RegisterWorkerRequest request;
  request.set_worker_id(std::string(worker_id));
  request.set_raiden_worker_endpoint(std::string(raiden_worker_endpoint));
  request.set_node_id(node_id);
  for (const auto& ep : raiden_transfer_endpoints) {
    auto* desc = request.add_raiden_transfer_endpoints();
    desc->set_endpoint(ep.endpoint);
    for (int64_t shard : ep.shards) {
      desc->add_shards(shard);
    }
  }
  for (uint64_t bytes : block_array_bytes) {
    request.add_block_array_bytes(static_cast<int64_t>(bytes));
  }
  request.set_num_kv_shards(num_kv_shards);

  ::tpu_sync::proto::RegisterWorkerResponse response;
  grpc::ClientContext context;
  grpc::Status status = stub_->RegisterWorker(&context, request, &response);
  ABSL_RETURN_IF_ERROR(FromGrpcStatus(status));
  if (!response.success()) {
    return absl::FailedPreconditionError(response.error_message());
  }
  return absl::OkStatus();
}

}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden
