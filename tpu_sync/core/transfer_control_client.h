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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TRANSFER_CONTROL_CLIENT_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TRANSFER_CONTROL_CLIENT_H_

#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "tpu_sync/proto/transfer_control.pb.h"

namespace tpu_raiden {

class TransferControlClient {
 public:
  explicit TransferControlClient(
      absl::Duration default_timeout = absl::Seconds(120));
  ~TransferControlClient() = default;

  // Synchronous pull-stream handshake.
  // Translates non-OK application status into absl::Status errors for
  // ABSL_ASSIGN_OR_RETURN.
  absl::StatusOr<::tpu_sync::proto::PullStreamResponse> PullStream(
      absl::string_view target_endpoint,
      const ::tpu_sync::proto::PullStreamRequest& request,
      absl::Duration timeout = absl::Seconds(120));

  // Sends completion or settlement acknowledgment.
  absl::StatusOr<::tpu_sync::proto::TransferAckResponse> Ack(
      absl::string_view target_endpoint,
      const ::tpu_sync::proto::TransferAckRequest& request,
      absl::Duration timeout = absl::Seconds(30));

  // Readiness / health check.
  absl::StatusOr<::tpu_sync::proto::TransferLivenessResponse> CheckLiveness(
      absl::string_view target_endpoint,
      const ::tpu_sync::proto::TransferLivenessRequest& request,
      absl::Duration timeout = absl::Seconds(10));

  // Injects an in-process channel for unit testing.
  void RegisterInProcessChannel(absl::string_view endpoint,
                                std::shared_ptr<grpc::Channel> channel);

 private:
  std::shared_ptr<grpc::Channel> GetOrCreateChannel(absl::string_view endpoint);

  absl::Duration default_timeout_;
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::shared_ptr<grpc::Channel>>
      channel_cache_ ABSL_GUARDED_BY(mu_);
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_CORE_TRANSFER_CONTROL_CLIENT_H_
