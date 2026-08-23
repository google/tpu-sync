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

#include "tpu_sync/core/controller/worker_service_client.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "grpcpp/client_context.h"
#include "grpcpp/support/status.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/proto/worker_service.grpc.pb.h"
#include "tpu_sync/proto/worker_service.pb.h"

namespace tpu_raiden {
namespace controller {

WorkerServiceClient::WorkerServiceClient(std::shared_ptr<grpc::Channel> channel)
    : stub_(::tpu_sync::proto::WorkerService::NewStub(channel)) {}

tsl::Future<::tpu_sync::proto::CreateBuffersResponse>
WorkerServiceClient::CreateBuffers(
    const ::tpu_sync::proto::CreateBuffersRequest& request) {
  auto [promise, future] =
      tsl::MakePromise<::tpu_sync::proto::CreateBuffersResponse>();
  auto context = std::make_shared<grpc::ClientContext>();
  auto response = std::make_shared<::tpu_sync::proto::CreateBuffersResponse>();

  stub_->async()->CreateBuffers(
      context.get(), &request, response.get(),
      [context, response,
       promise = std::move(promise).ToShared()](grpc::Status status) {
        if (!status.ok()) {
          promise->Set(absl::InternalError(absl::StrCat(
              "CreateBuffers RPC failed: ", status.error_message())));
        } else {
          promise->Set(std::move(*response));
        }
      });
  return future;
}

tsl::Future<::tpu_sync::proto::DeleteBuffersResponse>
WorkerServiceClient::DeleteBuffers(
    const ::tpu_sync::proto::DeleteBuffersRequest& request) {
  auto [promise, future] =
      tsl::MakePromise<::tpu_sync::proto::DeleteBuffersResponse>();
  auto context = std::make_shared<grpc::ClientContext>();
  auto response = std::make_shared<::tpu_sync::proto::DeleteBuffersResponse>();

  stub_->async()->DeleteBuffers(
      context.get(), &request, response.get(),
      [context, response,
       promise = std::move(promise).ToShared()](grpc::Status status) {
        if (!status.ok()) {
          promise->Set(absl::InternalError(absl::StrCat(
              "DeleteBuffers RPC failed: ", status.error_message())));
        } else {
          promise->Set(std::move(*response));
        }
      });
  return future;
}

tsl::Future<::tpu_sync::proto::TransferProgramResponse>
WorkerServiceClient::SubmitTransferProgram(
    const ::tpu_sync::proto::TransferProgramRequest& request) {
  auto [promise, future] =
      tsl::MakePromise<::tpu_sync::proto::TransferProgramResponse>();
  auto context = std::make_shared<grpc::ClientContext>();
  auto response =
      std::make_shared<::tpu_sync::proto::TransferProgramResponse>();

  stub_->async()->SubmitTransferProgram(
      context.get(), &request, response.get(),
      [context, response,
       promise = std::move(promise).ToShared()](grpc::Status status) {
        if (!status.ok()) {
          promise->Set(absl::InternalError(absl::StrCat(
              "SubmitTransferProgram RPC failed: ", status.error_message())));
        } else {
          promise->Set(std::move(*response));
        }
      });
  return future;
}

tsl::Future<> WorkerServiceClient::TransferBuffers(
    const ::tpu_sync::proto::TransferBuffersRequest& request) {
  auto [promise, future] = tsl::MakePromise<>();
  auto context = std::make_shared<grpc::ClientContext>();
  auto response =
      std::make_shared<::tpu_sync::proto::TransferBuffersResponse>();

  stub_->async()->TransferBuffers(
      context.get(), &request, response.get(),
      [context, response,
       promise = std::move(promise).ToShared()](grpc::Status status) {
        if (!status.ok()) {
          promise->Set(absl::InternalError(absl::StrCat(
              "TransferBuffers RPC failed: ", status.error_message())));
        } else if (!response->success()) {
          promise->Set(absl::InternalError(response->message()));
        } else {
          promise->Set(absl::OkStatus());
        }
      });
  return future;
}

tsl::Future<::tpu_sync::proto::RegisterBackendsResponse>
WorkerServiceClient::RegisterBackends(
    const ::tpu_sync::proto::RegisterBackendsRequest& request) {
  auto [promise, future] =
      tsl::MakePromise<::tpu_sync::proto::RegisterBackendsResponse>();
  auto context = std::make_shared<grpc::ClientContext>();
  auto response =
      std::make_shared<::tpu_sync::proto::RegisterBackendsResponse>();

  stub_->async()->RegisterBackends(
      context.get(), &request, response.get(),
      [context, response,
       promise = std::move(promise).ToShared()](grpc::Status status) {
        if (!status.ok()) {
          promise->Set(absl::InternalError(absl::StrCat(
              "RegisterBackends RPC failed: ", status.error_message())));
        } else {
          promise->Set(std::move(*response));
        }
      });
  return future;
}

}  // namespace controller
}  // namespace tpu_raiden
