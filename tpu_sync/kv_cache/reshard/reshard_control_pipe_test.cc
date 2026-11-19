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

#include "tpu_sync/kv_cache/reshard/reshard_control_pipe.h"

#include <deque>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/common/control_pipe/control_pipe_client.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/proto/control_pipe.pb.h"
#include "tpu_sync/rpc/controller_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {
namespace {

// Returns the scripted failures in order (the last one forever when
// `repeat_last`), then a successful ControllerResponse.
class ScriptedTransport final : public ControlPipeClient {
 public:
  ScriptedTransport(std::deque<absl::Status> failures, bool repeat_last)
      : failures_(std::move(failures)), repeat_last_(repeat_last) {}

  absl::StatusOr<control_pipe::proto::ControlResponseEnvelope> SendRaw(
      absl::string_view, const control_pipe::proto::ControlEnvelope& envelope,
      absl::Duration) override {
    ++calls_;
    if (!failures_.empty()) {
      absl::Status status = failures_.front();
      if (!repeat_last_ || failures_.size() > 1) failures_.pop_front();
      return status;
    }
    control_pipe::proto::ControlResponseEnvelope resp_env;
    resp_env.set_request_id(envelope.request_id());
    tpu_sync::rpc::ControllerResponse resp;
    resp.set_success(true);
    resp.SerializeToString(resp_env.mutable_payload());
    return resp_env;
  }
  ControlPipeBackendType backend_type() const override {
    return ControlPipeBackendType::kTcp;
  }
  int calls() const { return calls_; }

 private:
  std::deque<absl::Status> failures_;
  bool repeat_last_;
  int calls_ = 0;
};

absl::Status Refused() {
  return MarkControlPipeNotSent(absl::UnavailableError("connection refused"));
}

absl::StatusOr<tpu_sync::rpc::ControllerResponse> Call(
    ScriptedTransport& transport, absl::Duration timeout,
    absl::Duration connect_retry_budget) {
  return CallReshardControlPipe<tpu_sync::rpc::ControllerRequest,
                                tpu_sync::rpc::ControllerResponse>(
      &transport, "10.0.0.1:27000", tpu_sync::rpc::ControllerRequest(), timeout,
      connect_retry_budget);
}

TEST(CallReshardControlPipeTest, UnreachablePeerFailsWithinConnectRetryBudget) {
  ScriptedTransport transport({Refused()}, /*repeat_last=*/true);
  const absl::Time start = absl::Now();
  auto resp = Call(transport, absl::Seconds(300), absl::Milliseconds(300));
  EXPECT_LT(absl::Now() - start, absl::Seconds(3));
  ASSERT_TRUE(absl::IsUnavailable(resp.status())) << resp.status();
  EXPECT_EQ(resp.status().message(),
            "Control RPC connection failed before sending request to "
            "10.0.0.1:27000");
  EXPECT_GT(transport.calls(), 1);
}

TEST(CallReshardControlPipeTest, PossiblyDeliveredIsNeverReportedNotSent) {
  // An attempt that may have reached the peer, then refusals until the budget.
  ScriptedTransport transport({absl::UnavailableError("EOF"), Refused()},
                              /*repeat_last=*/true);
  auto resp = Call(transport, absl::Seconds(300), absl::Milliseconds(300));
  ASSERT_TRUE(absl::IsDeadlineExceeded(resp.status())) << resp.status();
  EXPECT_THAT(resp.status().message(),
              ::testing::StartsWith("Timeout (0s) failed to connect to robust "
                                    "endpoint 10.0.0.1:27000: "));
  EXPECT_FALSE(IsControlPipeNotSent(resp.status()));
  EXPECT_GT(transport.calls(), 2);
}

TEST(CallReshardControlPipeTest, UnreachablePeerWithoutBudgetIsNotSent) {
  // Every attempt is refused until the timeout: nothing reached the peer.
  ScriptedTransport transport({Refused()}, /*repeat_last=*/true);
  auto resp = Call(transport, absl::Milliseconds(300), absl::ZeroDuration());
  ASSERT_TRUE(absl::IsDeadlineExceeded(resp.status())) << resp.status();
  EXPECT_THAT(resp.status().message(),
              ::testing::StartsWith("Timeout (0s) failed to connect to robust "
                                    "endpoint 10.0.0.1:27000: "));
  EXPECT_TRUE(IsControlPipeNotSent(resp.status()));
  EXPECT_GT(transport.calls(), 1);
}

TEST(CallReshardControlPipeTest, WithoutBudgetEveryUnavailableIsRetried) {
  ScriptedTransport transport({absl::UnavailableError("EOF"), Refused()},
                              /*repeat_last=*/false);
  auto resp = Call(transport, absl::Seconds(30), absl::ZeroDuration());
  ASSERT_TRUE(resp.ok()) << resp.status();
  EXPECT_EQ(transport.calls(), 3);
}

}  // namespace
}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden
