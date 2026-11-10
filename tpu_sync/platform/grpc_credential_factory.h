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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_PLATFORM_GRPC_CREDENTIAL_FACTORY_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_PLATFORM_GRPC_CREDENTIAL_FACTORY_H_

#include <memory>

#include "absl/status/statusor.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"

namespace tpu_raiden {

// Abstract factory for gRPC server and channel credentials.
//
// Platform selection is resolved by build-configuration `select()`:
// - Google-internal default builds link LoasCredentialFactory (LOAS2).
// - `--config=gce` and open-source/cloud builds link CloudCredentialFactory
//   (insecure gRPC credentials).
//
// Implementations must be thread-safe for concurrent calls across threads.
class GrpcCredentialFactory {
 public:
  virtual ~GrpcCredentialFactory() = default;

  // Creates server credentials for hosting a gRPC server.
  // Thread-safe for concurrent calls across threads.
  virtual absl::StatusOr<std::shared_ptr<grpc::ServerCredentials>>
  CreateServerCredentials() = 0;

  // Creates channel credentials for connecting a gRPC client.
  // Thread-safe for concurrent calls across threads.
  virtual absl::StatusOr<std::shared_ptr<grpc::ChannelCredentials>>
  CreateClientCredentials() = 0;
};

// Returns a platform-appropriate GrpcCredentialFactory (never nullptr).
// Thread-safe for concurrent calls across threads.
std::shared_ptr<GrpcCredentialFactory> CreateGrpcCredentialFactory();

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_PLATFORM_GRPC_CREDENTIAL_FACTORY_H_
