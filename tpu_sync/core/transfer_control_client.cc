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

#include "tpu_sync/core/transfer_control_client.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "third_party/grpc/include/grpc/impl/channel_arg_names.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/channel_arguments.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/proto/transfer_control.grpc.pb.h"
#include "tpu_sync/proto/transfer_control.pb.h"

namespace tpu_raiden {
namespace {

absl::Status GrpcStatusToAbslStatus(const grpc::Status& s) {
  if (s.ok()) {
    return absl::OkStatus();
  }
  absl::StatusCode code;
  switch (s.error_code()) {
    case grpc::StatusCode::CANCELLED:
      code = absl::StatusCode::kCancelled;
      break;
    case grpc::StatusCode::UNKNOWN:
      code = absl::StatusCode::kUnknown;
      break;
    case grpc::StatusCode::INVALID_ARGUMENT:
      code = absl::StatusCode::kInvalidArgument;
      break;
    case grpc::StatusCode::DEADLINE_EXCEEDED:
      code = absl::StatusCode::kDeadlineExceeded;
      break;
    case grpc::StatusCode::NOT_FOUND:
      code = absl::StatusCode::kNotFound;
      break;
    case grpc::StatusCode::ALREADY_EXISTS:
      code = absl::StatusCode::kAlreadyExists;
      break;
    case grpc::StatusCode::PERMISSION_DENIED:
      code = absl::StatusCode::kPermissionDenied;
      break;
    case grpc::StatusCode::RESOURCE_EXHAUSTED:
      code = absl::StatusCode::kResourceExhausted;
      break;
    case grpc::StatusCode::FAILED_PRECONDITION:
      code = absl::StatusCode::kFailedPrecondition;
      break;
    case grpc::StatusCode::ABORTED:
      code = absl::StatusCode::kAborted;
      break;
    case grpc::StatusCode::OUT_OF_RANGE:
      code = absl::StatusCode::kOutOfRange;
      break;
    case grpc::StatusCode::UNIMPLEMENTED:
      code = absl::StatusCode::kUnimplemented;
      break;
    case grpc::StatusCode::INTERNAL:
      code = absl::StatusCode::kInternal;
      break;
    case grpc::StatusCode::UNAVAILABLE:
      code = absl::StatusCode::kUnavailable;
      break;
    case grpc::StatusCode::DATA_LOSS:
      code = absl::StatusCode::kDataLoss;
      break;
    case grpc::StatusCode::UNAUTHENTICATED:
      code = absl::StatusCode::kUnauthenticated;
      break;
    default:
      code = absl::StatusCode::kUnknown;
      break;
  }
  return absl::Status(code, s.error_message());
}

}  // namespace

TransferControlClient::TransferControlClient(absl::Duration default_timeout)
    : default_timeout_(default_timeout) {}

void TransferControlClient::RegisterInProcessChannel(
    absl::string_view endpoint, std::shared_ptr<grpc::Channel> channel) {
  absl::MutexLock lock(mu_);
  channel_cache_[std::string(endpoint)] = channel;
  if (absl::StartsWith(endpoint, "grpc://")) {
    channel_cache_[std::string(absl::StripPrefix(endpoint, "grpc://"))] =
        channel;
  }
}

std::shared_ptr<grpc::Channel> TransferControlClient::GetOrCreateChannel(
    absl::string_view endpoint) {
  std::string ep(endpoint);
  if (absl::StartsWith(ep, "grpc://")) {
    ep = std::string(absl::StripPrefix(ep, "grpc://"));
  }

  absl::MutexLock lock(mu_);
  auto it = channel_cache_.find(ep);
  if (it != channel_cache_.end()) {
    return it->second;
  }
  if (ep != endpoint) {
    auto orig_it = channel_cache_.find(std::string(endpoint));
    if (orig_it != channel_cache_.end()) {
      return orig_it->second;
    }
  }

  grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 30000);
  args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 10000);
  args.SetInt(GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA, 0);
  args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);

  auto channel =
      grpc::CreateCustomChannel(ep, grpc::InsecureChannelCredentials(), args);
  channel_cache_[ep] = channel;
  return channel;
}

absl::StatusOr<::tpu_sync::proto::PullStreamResponse>
TransferControlClient::PullStream(
    absl::string_view target_endpoint,
    const ::tpu_sync::proto::PullStreamRequest& request,
    absl::Duration timeout) {
  std::shared_ptr<grpc::Channel> channel = GetOrCreateChannel(target_endpoint);
  auto stub = ::tpu_sync::proto::TransferControlService::NewStub(channel);

  grpc::ClientContext context;
  context.set_deadline(absl::ToChronoTime(absl::Now() + timeout));

  ::tpu_sync::proto::PullStreamResponse response;
  grpc::Status rpc_status = stub->PullStream(&context, request, &response);
  if (!rpc_status.ok()) {
    return GrpcStatusToAbslStatus(rpc_status);
  }

  // Translate non-OK application status into absl::Status errors
  if (response.status() != ::tpu_sync::proto::STATUS_OK) {
    std::string err_msg = response.error_message();
    if (err_msg.empty()) {
      err_msg = absl::StrCat(
          "PullStream failed with status: ",
          ::tpu_sync::proto::TransferControlStatusCode_Name(response.status()));
    }
    switch (response.status()) {
      case ::tpu_sync::proto::STATUS_NOT_REGISTERED:
        return absl::NotFoundError(err_msg);
      case ::tpu_sync::proto::STATUS_BLOCK_VALIDATION_FAILED:
        return absl::InvalidArgumentError(err_msg);
      case ::tpu_sync::proto::STATUS_STAGING_UNAVAILABLE:
        return absl::ResourceExhaustedError(err_msg);
      case ::tpu_sync::proto::STATUS_SHUTTING_DOWN:
        return absl::UnavailableError(err_msg);
      case ::tpu_sync::proto::STATUS_INTERNAL_ERROR:
        return absl::InternalError(err_msg);
      default:
        return absl::UnknownError(err_msg);
    }
  }

  if (!response.success()) {
    return absl::InternalError(response.error_message().empty()
                                   ? "PullStream reported failure"
                                   : response.error_message());
  }

  return response;
}

absl::StatusOr<::tpu_sync::proto::TransferAckResponse>
TransferControlClient::Ack(absl::string_view target_endpoint,
                           const ::tpu_sync::proto::TransferAckRequest& request,
                           absl::Duration timeout) {
  std::shared_ptr<grpc::Channel> channel = GetOrCreateChannel(target_endpoint);
  auto stub = ::tpu_sync::proto::TransferControlService::NewStub(channel);

  grpc::ClientContext context;
  context.set_deadline(absl::ToChronoTime(absl::Now() + timeout));

  ::tpu_sync::proto::TransferAckResponse response;
  grpc::Status rpc_status = stub->Ack(&context, request, &response);
  if (!rpc_status.ok()) {
    return GrpcStatusToAbslStatus(rpc_status);
  }

  return response;
}

absl::StatusOr<::tpu_sync::proto::TransferLivenessResponse>
TransferControlClient::CheckLiveness(
    absl::string_view target_endpoint,
    const ::tpu_sync::proto::TransferLivenessRequest& request,
    absl::Duration timeout) {
  std::shared_ptr<grpc::Channel> channel = GetOrCreateChannel(target_endpoint);
  auto stub = ::tpu_sync::proto::TransferControlService::NewStub(channel);

  grpc::ClientContext context;
  context.set_deadline(absl::ToChronoTime(absl::Now() + timeout));

  ::tpu_sync::proto::TransferLivenessResponse response;
  grpc::Status rpc_status = stub->CheckLiveness(&context, request, &response);
  if (!rpc_status.ok()) {
    return GrpcStatusToAbslStatus(rpc_status);
  }

  return response;
}

}  // namespace tpu_raiden
