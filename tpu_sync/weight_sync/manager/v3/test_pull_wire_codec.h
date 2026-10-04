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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_TEST_PULL_WIRE_CODEC_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_TEST_PULL_WIRE_CODEC_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/weight_sync/manager/v3/controller_v3.h"
#include "tpu_sync/weight_sync/manager/v3/dynamic_pull_engine.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {

// A compact binary encoding of the pull RPCs for tests and benchmarks of the
// pull server, until the wire commands exist. Not a stable format.
class TestPullWireCodec final : public PullWireCodec {
 public:
  static constexpr absl::string_view kMessageType =
      "tpu_raiden.weight_sync.v3.TestPullRpc";

  absl::string_view message_type() const override { return kMessageType; }
  absl::StatusOr<PullServiceRequest> DecodeRequest(
      absl::string_view payload) const override;
  std::string EncodeReply(const PullServiceReply& reply) const override;

  // The client side.
  static std::string EncodeRequest(const PullServiceRequest& request);
  static absl::StatusOr<PullServiceReply> DecodeReply(
      absl::string_view payload);
};

}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_TEST_PULL_WIRE_CODEC_H_
