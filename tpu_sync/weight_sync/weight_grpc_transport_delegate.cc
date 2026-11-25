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

#include "tpu_sync/weight_sync/weight_grpc_transport_delegate.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/transfer_session.h"
#include "tpu_sync/weight_sync/lib/grpc_transport_types.h"
#include "tpu_sync/weight_sync/weight_send_session.h"

namespace tpu_raiden::weight_sync {

WeightGrpcTransportDelegate::WeightGrpcTransportDelegate(
    HostBufferProvider host_buffer_provider)
    : host_buffer_provider_(std::move(host_buffer_provider)) {}

absl::Status WeightGrpcTransportDelegate::RegisterSession(
    uint64_t uuid, std::shared_ptr<::tpu_raiden::TransferSession> session) {
  if (session == nullptr) {
    return absl::InvalidArgumentError(
        "WeightGrpcTransportDelegate requires a non-null TransferSession");
  }
  absl::MutexLock lock(mu_);
  auto [it, inserted] = sessions_.emplace(uuid, std::move(session));
  if (!inserted) {
    return absl::AlreadyExistsError(
        absl::StrCat("TransferSession already registered for uuid=", uuid));
  }
  return absl::OkStatus();
}

absl::Status WeightGrpcTransportDelegate::UnregisterSession(uint64_t uuid) {
  absl::MutexLock lock(mu_);
  if (sessions_.erase(uuid) == 0) {
    return absl::NotFoundError(absl::StrCat("Session ", uuid, " not found"));
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::Span<uint8_t>> WeightGrpcTransportDelegate::GetHostBuffer(
    uint64_t buffer_id, uint32_t shard_id) {
  if (!host_buffer_provider_) {
    return absl::FailedPreconditionError(
        "WeightGrpcTransportDelegate has no HostBufferProvider configured");
  }
  return host_buffer_provider_(buffer_id, shard_id);
}

absl::StatusOr<std::shared_ptr<::tpu_raiden::TransferSession>>
WeightGrpcTransportDelegate::GetSession(uint64_t uuid) const {
  absl::MutexLock lock(mu_);
  auto it = sessions_.find(uuid);
  if (it == sessions_.end()) {
    return absl::NotFoundError(absl::StrCat("Session ", uuid, " not found"));
  }
  return it->second;
}

absl::Status WeightGrpcTransportDelegate::OnPushChunks(
    uint64_t uuid, uint64_t buffer_id, uint32_t shard_id,
    uint64_t chunk_batch_id) {
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<::tpu_raiden::TransferSession> session,
                        GetSession(uuid));

  const int shard = static_cast<int>(shard_id);
  return session->OnDataReceived(static_cast<size_t>(buffer_id), {},
                                 absl::MakeConstSpan(&shard, 1));
}

absl::StatusOr<PullChunksResponse> WeightGrpcTransportDelegate::OnPullChunks(
    const PullChunksRequest& request) {
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<::tpu_raiden::TransferSession> session,
                        GetSession(request.uuid));
  auto send_session = std::dynamic_pointer_cast<WeightSendSession>(session);
  if (send_session == nullptr) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Session for uuid=", request.uuid, " is not a WeightSendSession"));
  }
  return send_session->OnPullChunks(request);
}

}  // namespace tpu_raiden::weight_sync
