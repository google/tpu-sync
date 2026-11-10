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

#include "tpu_sync/platform/cloud/cloud_credential_factory.h"

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "tpu_sync/platform/grpc_credential_factory.h"

namespace tpu_raiden {

absl::StatusOr<std::shared_ptr<grpc::ServerCredentials>>
CloudCredentialFactory::CreateServerCredentials() {
  std::shared_ptr<grpc::ServerCredentials> credentials =
      grpc::InsecureServerCredentials();
  if (credentials == nullptr) {
    return absl::FailedPreconditionError(
        "grpc::InsecureServerCredentials() returned nullptr.");
  }
  return credentials;
}

absl::StatusOr<std::shared_ptr<grpc::ChannelCredentials>>
CloudCredentialFactory::CreateClientCredentials() {
  std::shared_ptr<grpc::ChannelCredentials> credentials =
      grpc::InsecureChannelCredentials();
  if (credentials == nullptr) {
    return absl::FailedPreconditionError(
        "grpc::InsecureChannelCredentials() returned nullptr.");
  }
  return credentials;
}

std::shared_ptr<GrpcCredentialFactory> CreateGrpcCredentialFactory() {
  return std::make_shared<CloudCredentialFactory>();
}

}  // namespace tpu_raiden
