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

#include "tpu_sync/core/control_plane_backend.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/time/time.h"
#include "tpu_sync/core/grpc_control_plane_backend.h"
#include "tpu_sync/core/tcp_control_plane_backend.h"

namespace tpu_raiden {

ControlPlaneBackendType ResolveControlPlaneBackendType(
    std::optional<ControlPlaneBackendType> override_type) {
  if (override_type.has_value()) {
    return *override_type;
  }
  if (const char* env_backend =
          std::getenv("TPU_RAIDEN_CONTROL_PLANE_BACKEND")) {
    std::string val = absl::AsciiStrToLower(env_backend);
    if (val == "grpc") return ControlPlaneBackendType::kGrpc;
    if (val == "tcp") return ControlPlaneBackendType::kTcp;
    LOG(WARNING) << "Unknown TPU_RAIDEN_CONTROL_PLANE_BACKEND='" << env_backend
                 << "', falling back to TCP.";
  }
  if (const char* env_flag = std::getenv("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE")) {
    std::string val = absl::AsciiStrToLower(env_flag);
    if (val == "1" || val == "true") return ControlPlaneBackendType::kGrpc;
  }
  return ControlPlaneBackendType::kTcp;
}

std::unique_ptr<ControlPlaneBackend> CreateControlPlaneBackend(
    ControlPlaneBackendType type, ControlPlaneBackend::TaskExecutor executor,
    absl::Duration default_timeout) {
  switch (type) {
    case ControlPlaneBackendType::kGrpc:
      return std::make_unique<GrpcControlPlaneBackend>();
    case ControlPlaneBackendType::kTcp:
      return std::make_unique<TcpControlPlaneBackend>(std::move(executor),
                                                      default_timeout);
  }
  return std::make_unique<TcpControlPlaneBackend>(std::move(executor),
                                                  default_timeout);
}

}  // namespace tpu_raiden
