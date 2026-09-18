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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_CONTROL_DISPATCHER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_CONTROL_DISPATCHER_H_

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/proto/control_pipe.pb.h"

namespace tpu_raiden {

template <typename ReqProto>
struct HandlerOptions {
  // Maximum allowed byte size for the serialized `ReqProto` payload (defaults
  // to 16 MiB).
  size_t max_payload_bytes = 16 * 1024 * 1024;

  // Optional domain/semantic validation callback invoked immediately after
  // `ReqProto` deserialization (`ParseFromArray`) succeeds and strictly before
  // dispatching to the registered `handler`.
  //
  // Receives `(const ControlContext&, const ReqProto&)` containing the caller's
  // peer/request metadata and the parsed request proto. If `validator` is
  // non-null and returns a non-OK `absl::Status` (e.g.,
  // `absl::InvalidArgumentError` when repeated field counts exceed
  // `MaxPullStreamBlocks()`), `ControlDispatcher` short-circuits without
  // invoking `handler` and propagates that `absl::Status` back to the caller in
  // `ControlResponseEnvelope`. If `nullptr` (default) or if it returns
  // `absl::OkStatus()`, execution proceeds to `handler`.
  absl::AnyInvocable<absl::Status(const ControlContext&, const ReqProto&) const>
      validator = nullptr;

  // Sets the maximum allowed serialized payload size in |bytes| and returns a
  // reference to `*this` for fluent chaining.
  HandlerOptions& WithMaxPayloadBytes(size_t bytes) & {
    max_payload_bytes = bytes;
    return *this;
  }
  HandlerOptions&& WithMaxPayloadBytes(size_t bytes) && {
    max_payload_bytes = bytes;
    return std::move(*this);
  }

  // Sets the domain/semantic validation callback |v| and returns a reference to
  // `*this` for fluent chaining.
  HandlerOptions& WithValidator(
      absl::AnyInvocable<absl::Status(const ControlContext&, const ReqProto&)
                             const>
          v) & {
    validator = std::move(v);
    return *this;
  }
  HandlerOptions&& WithValidator(
      absl::AnyInvocable<absl::Status(const ControlContext&, const ReqProto&)
                             const>
          v) && {
    validator = std::move(v);
    return std::move(*this);
  }
};

class ControlDispatcher {
 public:
  using RawHandlerFn = absl::AnyInvocable<absl::StatusOr<std::string>(
      const ControlContext& ctx, absl::string_view request_bytes) const>;

  ControlDispatcher() = default;

  // Registers a strongly-typed Protobuf request-response handler.
  // `message_type` defaults to `ReqProto::descriptor()->full_name()`.
  template <typename ReqProto, typename RespProto>
  void RegisterHandler(absl::AnyInvocable<absl::StatusOr<RespProto>(
                           const ControlContext&, const ReqProto&) const>
                           handler,
                       HandlerOptions<ReqProto> options = {},
                       absl::string_view custom_type_tag = "") {
    std::string type_name =
        custom_type_tag.empty()
            ? std::string(ReqProto::descriptor()->full_name())
            : std::string(custom_type_tag);

    RawHandlerFn raw_fn =
        [handler = std::move(handler), validator = std::move(options.validator),
         type_name](
            const ControlContext& ctx,
            absl::string_view req_bytes) -> absl::StatusOr<std::string> {
      ReqProto req;
      if (!req.ParseFromArray(req_bytes.data(),
                              static_cast<int>(req_bytes.size()))) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Failed to parse protobuf payload for type: ", type_name));
      }
      if (validator) {
        ABSL_RETURN_IF_ERROR(validator(ctx, req));
      }
      ABSL_ASSIGN_OR_RETURN(RespProto resp, handler(ctx, req));
      std::string out_bytes;
      if (!resp.SerializeToString(&out_bytes)) {
        return absl::InternalError(absl::StrCat(
            "Failed to serialize response protobuf for type: ", type_name));
      }
      return out_bytes;
    };

    RegisterRawHandler(type_name, std::move(raw_fn), options.max_payload_bytes);
  }

  // Registers a one-way / notification handler returning absl::Status.
  template <typename ReqProto>
  void RegisterOneWayHandler(absl::AnyInvocable<absl::Status(
                                 const ControlContext&, const ReqProto&) const>
                                 handler,
                             HandlerOptions<ReqProto> options = {},
                             absl::string_view custom_type_tag = "") {
    std::string type_name =
        custom_type_tag.empty()
            ? std::string(ReqProto::descriptor()->full_name())
            : std::string(custom_type_tag);

    RawHandlerFn raw_fn =
        [handler = std::move(handler), validator = std::move(options.validator),
         type_name](
            const ControlContext& ctx,
            absl::string_view req_bytes) -> absl::StatusOr<std::string> {
      ReqProto req;
      if (!req.ParseFromArray(req_bytes.data(),
                              static_cast<int>(req_bytes.size()))) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Failed to parse protobuf payload for type: ", type_name));
      }
      if (validator) {
        ABSL_RETURN_IF_ERROR(validator(ctx, req));
      }
      ABSL_RETURN_IF_ERROR(handler(ctx, req));
      return std::string();
    };

    RegisterRawHandler(type_name, std::move(raw_fn), options.max_payload_bytes);
  }

  void RegisterRawHandler(absl::string_view message_type, RawHandlerFn handler,
                          size_t max_payload_bytes);

  // Dispatches an incoming ControlEnvelope to the registered handler.
  control_pipe::proto::ControlResponseEnvelope Dispatch(
      const ControlContext& ctx,
      const control_pipe::proto::ControlEnvelope& envelope) const;

  bool HasHandler(absl::string_view message_type) const;

  // Returns the configured max_payload_bytes for a registered message type,
  // or `default_max` if the type is unregistered.
  size_t GetMaxPayloadBytes(absl::string_view message_type,
                            size_t default_max = 16 * 1024 * 1024) const;

 private:
  struct HandlerEntry {
    RawHandlerFn fn;
    size_t max_payload_bytes;
  };

  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::shared_ptr<const HandlerEntry>>
      handlers_ ABSL_GUARDED_BY(mu_);
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CONTROL_PIPE_CONTROL_DISPATCHER_H_
