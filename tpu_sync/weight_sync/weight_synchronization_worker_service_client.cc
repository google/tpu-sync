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

#include "tpu_sync/weight_sync/weight_synchronization_worker_service_client.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/status.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/rpc/raiden_service.grpc.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace weight_sync {
namespace {

void SetContextDeadline(grpc::ClientContext* context, absl::Duration timeout) {
  if (timeout < absl::InfiniteDuration() && timeout > absl::ZeroDuration()) {
    context->set_deadline(absl::ToChronoTime(absl::Now() + timeout));
  }
}

}  // namespace

WeightSynchronizationWorkerServiceClient::
    WeightSynchronizationWorkerServiceClient(
        std::shared_ptr<::grpc::ChannelInterface> channel)
    : stub_(::tpu_sync::rpc::WeightSynchronizationWorkerService::NewStub(
          channel)) {}

WeightSynchronizationWorkerServiceClient::
    WeightSynchronizationWorkerServiceClient(
        std::unique_ptr<
            ::tpu_sync::rpc::WeightSynchronizationWorkerService::StubInterface>
            stub)
    : stub_(std::move(stub)) {}

WeightSynchronizationWorkerServiceClient::
    WeightSynchronizationWorkerServiceClient(absl::string_view target)
    : stub_(::tpu_sync::rpc::WeightSynchronizationWorkerService::NewStub(
          ::grpc::CreateChannel(std::string(target),
                                ::grpc::InsecureChannelCredentials()))) {}

tsl::Future<::tpu_sync::rpc::ControlResponse>
WeightSynchronizationWorkerServiceClient::HandleControl(
    const ::tpu_sync::rpc::ControlRequest& request, absl::Duration timeout) {
  auto [promise, future] = tsl::MakePromise<::tpu_sync::rpc::ControlResponse>();
  auto context = std::make_shared<::grpc::ClientContext>();
  SetContextDeadline(context.get(), timeout);

  auto response = std::make_shared<::tpu_sync::rpc::ControlResponse>();

  stub_->async()->HandleControl(
      context.get(), &request, response.get(),
      [context, response,
       promise = std::move(promise).ToShared()](::grpc::Status status) mutable {
        if (!status.ok()) {
          promise->Set(
              absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                           absl::StrCat("HandleControl RPC failed: ",
                                        status.error_message())));
        } else {
          promise->Set(std::move(*response));
        }
      });

  return future;
}

tsl::Future<::tpu_sync::rpc::ControlResponse>
WeightSynchronizationWorkerServiceClient::StartTransfer(
    const ::tpu_sync::rpc::StartTransferRequest& start_transfer_request,
    const std::vector<std::string>& peers, absl::Duration timeout) {
  ::tpu_sync::rpc::ControlRequest req;
  req.set_command(::tpu_sync::rpc::ControlRequest::COMMAND_START_TRANSFER);
  *req.mutable_start_transfer_request() = start_transfer_request;
  for (const auto& peer : peers) {
    req.add_peers(peer);
  }

  return HandleControl(req, timeout);
}

tsl::Future<::tpu_sync::rpc::ControlResponse>
WeightSynchronizationWorkerServiceClient::Shutdown(absl::Duration timeout) {
  ::tpu_sync::rpc::ControlRequest req;
  req.set_command(::tpu_sync::rpc::ControlRequest::COMMAND_SHUTDOWN);

  return HandleControl(req, timeout);
}

}  // namespace weight_sync
}  // namespace tpu_raiden
