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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_CONTROL_PIPE_CLIENT_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_CONTROL_PIPE_CLIENT_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/proto/control_pipe.pb.h"

namespace tpu_raiden {

class ControlPipeClient {
 public:
  virtual ~ControlPipeClient() = default;

  // Core type-erased transport send method implemented by TCP and gRPC
  // backends.
  virtual absl::StatusOr<control_pipe::proto::ControlResponseEnvelope> SendRaw(
      absl::string_view endpoint,
      const control_pipe::proto::ControlEnvelope& envelope,
      absl::Duration timeout) = 0;

  // Strongly-typed synchronous Request-Response RPC call.
  template <typename ReqProto, typename RespProto>
  absl::StatusOr<RespProto> Call(absl::string_view endpoint,
                                 const ReqProto& request,
                                 absl::Duration timeout = absl::ZeroDuration(),
                                 absl::string_view custom_type_tag = "") {
    control_pipe::proto::ControlEnvelope env;
    env.set_message_type(custom_type_tag.empty()
                             ? std::string(ReqProto::descriptor()->full_name())
                             : std::string(custom_type_tag));
    env.set_request_id(NextRequestId());
    if (!request.SerializeToString(env.mutable_payload())) {
      return absl::InternalError("Failed to serialize request protobuf");
    }

    ABSL_ASSIGN_OR_RETURN(control_pipe::proto::ControlResponseEnvelope resp_env,
                          SendRaw(endpoint, env, timeout));
    if (resp_env.status_code() != 0) {
      return absl::Status(static_cast<absl::StatusCode>(resp_env.status_code()),
                          resp_env.error_message());
    }

    RespProto response;
    if (!response.ParseFromString(resp_env.payload())) {
      return absl::InternalError("Failed to parse response protobuf payload");
    }
    return response;
  }

  // Strongly-typed One-Way / Ack notification call.
  template <typename ReqProto>
  absl::Status SendOneWay(absl::string_view endpoint, const ReqProto& request,
                          absl::Duration timeout = absl::ZeroDuration(),
                          absl::string_view custom_type_tag = "") {
    control_pipe::proto::ControlEnvelope env;
    env.set_message_type(custom_type_tag.empty()
                             ? std::string(ReqProto::descriptor()->full_name())
                             : std::string(custom_type_tag));
    env.set_request_id(NextRequestId());
    if (!request.SerializeToString(env.mutable_payload())) {
      return absl::InternalError(
          "Failed to serialize one-way request protobuf");
    }

    ABSL_ASSIGN_OR_RETURN(control_pipe::proto::ControlResponseEnvelope resp_env,
                          SendRaw(endpoint, env, timeout));
    if (resp_env.status_code() != 0) {
      return absl::Status(static_cast<absl::StatusCode>(resp_env.status_code()),
                          resp_env.error_message());
    }
    return absl::OkStatus();
  }

  virtual ControlPipeBackendType backend_type() const = 0;

 private:
  static uint64_t NextRequestId() {
    static std::atomic<uint64_t> next_id{1};
    return next_id.fetch_add(1, std::memory_order_relaxed);
  }
};

std::unique_ptr<ControlPipeClient> CreateControlPipeClient(
    const ControlPipeConfig& config);

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_CONTROL_PIPE_CLIENT_H_
