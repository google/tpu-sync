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

#include "tpu_sync/common/control_pipe/control_dispatcher.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/proto/control_pipe.pb.h"

namespace tpu_raiden {

void ControlDispatcher::RegisterRawHandler(absl::string_view message_type,
                                           RawHandlerFn handler,
                                           size_t max_payload_bytes) {
  absl::MutexLock lock(mu_);
  handlers_[std::string(message_type)] = std::make_shared<const HandlerEntry>(
      HandlerEntry{std::move(handler), max_payload_bytes});
}

bool ControlDispatcher::HasHandler(absl::string_view message_type) const {
  absl::MutexLock lock(mu_);
  return handlers_.contains(message_type);
}

size_t ControlDispatcher::GetMaxPayloadBytes(absl::string_view message_type,
                                             size_t default_max) const {
  absl::MutexLock lock(mu_);
  auto it = handlers_.find(message_type);
  if (it == handlers_.end()) {
    return default_max;
  }
  return it->second->max_payload_bytes;
}

control_pipe::proto::ControlResponseEnvelope ControlDispatcher::Dispatch(
    const ControlContext& ctx,
    const control_pipe::proto::ControlEnvelope& envelope) const {
  control_pipe::proto::ControlResponseEnvelope resp_env;
  resp_env.set_request_id(envelope.request_id());

  std::shared_ptr<const HandlerEntry> entry;
  {
    absl::MutexLock lock(mu_);
    auto it = handlers_.find(envelope.message_type());
    if (it == handlers_.end()) {
      resp_env.set_status_code(
          static_cast<int32_t>(absl::StatusCode::kUnimplemented));
      resp_env.set_error_message(absl::StrCat(
          "Unregistered control message type: ", envelope.message_type()));
      return resp_env;
    }
    entry = it->second;
  }

  if (envelope.payload().size() > entry->max_payload_bytes) {
    resp_env.set_status_code(
        static_cast<int32_t>(absl::StatusCode::kResourceExhausted));
    resp_env.set_error_message(absl::StrCat(
        "Payload size (", envelope.payload().size(),
        " bytes) exceeds max_payload_bytes (", entry->max_payload_bytes,
        ") for message type: ", envelope.message_type()));
    return resp_env;
  }

  absl::StatusOr<std::string> res = entry->fn(ctx, envelope.payload());
  if (!res.ok()) {
    resp_env.set_status_code(static_cast<int32_t>(res.status().code()));
    resp_env.set_error_message(std::string(res.status().message()));
    return resp_env;
  }

  resp_env.set_status_code(0);
  resp_env.set_payload(*std::move(res));
  return resp_env;
}

}  // namespace tpu_raiden
