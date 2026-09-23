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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_GRPC_UTIL_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_GRPC_UTIL_H_

#include <string>

#include "absl/status/status.h"
#include "grpcpp/support/status.h"

namespace tpu_raiden {

constexpr char kStreamRemovedMessage[] = "Stream removed";

// Identify if the given grpc::Status corresponds to an HTTP stream removed
// error (see chttp2_transport.cc). When auto-reconnecting to a remote worker
// after it restarts, gRPC can return an UNKNOWN error code with a "Stream
// removed" error message, which should be treated as transient/unavailable.
inline bool IsStreamRemovedError(const ::grpc::Status& s) {
  return !s.ok() && s.error_code() == ::grpc::StatusCode::UNKNOWN &&
         s.error_message() == kStreamRemovedMessage;
}

inline absl::Status FromGrpcStatus(const ::grpc::Status& s) {
  if (s.ok()) {
    return absl::OkStatus();
  }
  if (IsStreamRemovedError(s)) {
    return absl::Status(absl::StatusCode::kUnavailable, s.error_message());
  }
  return absl::Status(static_cast<absl::StatusCode>(s.error_code()),
                      s.error_message());
}

inline ::grpc::Status ToGrpcStatus(const absl::Status& s) {
  if (s.ok()) {
    return ::grpc::Status::OK;
  }
  return ::grpc::Status(static_cast<::grpc::StatusCode>(s.code()),
                        std::string(s.message()));
}

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_GRPC_UTIL_H_
