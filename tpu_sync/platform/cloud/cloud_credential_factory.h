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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_PLATFORM_CLOUD_CLOUD_CREDENTIAL_FACTORY_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_PLATFORM_CLOUD_CLOUD_CREDENTIAL_FACTORY_H_

#include <memory>

#include "absl/status/statusor.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "tpu_sync/platform/grpc_credential_factory.h"

namespace tpu_raiden {

// Cloud/OSS GrpcCredentialFactory.
//
// Google3 builds use ALTS credentials while the insecure-RPC allowlist request
// is pending; open-source exports use insecure gRPC credentials.
// Thread-safe for concurrent calls across threads.
class CloudCredentialFactory final : public GrpcCredentialFactory {
 public:
  absl::StatusOr<std::shared_ptr<grpc::ServerCredentials>>
  CreateServerCredentials() override;

  absl::StatusOr<std::shared_ptr<grpc::ChannelCredentials>>
  CreateClientCredentials() override;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_PLATFORM_CLOUD_CLOUD_CREDENTIAL_FACTORY_H_
