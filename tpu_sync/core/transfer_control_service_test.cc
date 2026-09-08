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

#include <cstdint>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/server_context.h"
#include "tpu_sync/core/transfer_control_client.h"
#include "tpu_sync/core/transfer_control_server.h"
#include "tpu_sync/proto/transfer_control.pb.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::StatusIs;
using ::testing::Eq;

class FakeTransferControlDelegate : public TransferControlDelegate {
 public:
  absl::Status HandleGrpcPullStream(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::PullStreamRequest& request,
      ::tpu_sync::proto::PullStreamResponse* response) override {
    absl::Duration delay;
    ::tpu_sync::proto::TransferControlStatusCode status;
    std::string err_msg;
    uint32_t num_layers;
    uint32_t data_port;
    {
      absl::MutexLock lock(mu_);
      delay = pull_stream_delay_;
      status = pull_stream_status_;
      err_msg = pull_stream_error_message_;
      num_layers = num_layers_;
      data_port = data_port_;
    }

    if (delay > absl::ZeroDuration()) {
      absl::SleepFor(delay);
    }

    if (context != nullptr && context->IsCancelled()) {
      return absl::CancelledError("Request cancelled or deadline exceeded");
    }

    if (status != ::tpu_sync::proto::STATUS_OK) {
      response->set_success(false);
      response->set_status(status);
      response->set_error_message(err_msg);
      return absl::OkStatus();
    }

    response->set_success(true);
    response->set_status(::tpu_sync::proto::STATUS_OK);
    response->set_num_layers(num_layers);
    response->set_data_port(data_port);
    return absl::OkStatus();
  }

  absl::Status HandleGrpcAck(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::TransferAckRequest& request,
      ::tpu_sync::proto::TransferAckResponse* response) override {
    absl::MutexLock lock(mu_);
    response->set_success(ack_success_);
    response->set_message(ack_message_);
    return absl::OkStatus();
  }

  absl::Status HandleGrpcCheckLiveness(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::TransferLivenessRequest& request,
      ::tpu_sync::proto::TransferLivenessResponse* response) override {
    absl::MutexLock lock(mu_);
    response->set_ready(liveness_ready_);
    response->set_active_workers(active_workers_);
    response->set_version(version_);
    return absl::OkStatus();
  }

  void SetPullStreamResponse(
      ::tpu_sync::proto::TransferControlStatusCode status,
      std::string error_message = "") {
    absl::MutexLock lock(mu_);
    pull_stream_status_ = status;
    pull_stream_error_message_ = std::move(error_message);
  }

  void SetPullStreamDelay(absl::Duration delay) {
    absl::MutexLock lock(mu_);
    pull_stream_delay_ = delay;
  }

  void SetAckResponse(bool success, std::string message) {
    absl::MutexLock lock(mu_);
    ack_success_ = success;
    ack_message_ = std::move(message);
  }

  void SetLivenessResponse(bool ready, int32_t active_workers,
                           std::string version) {
    absl::MutexLock lock(mu_);
    liveness_ready_ = ready;
    active_workers_ = active_workers;
    version_ = std::move(version);
  }

  void SetNumLayers(uint32_t num_layers) {
    absl::MutexLock lock(mu_);
    num_layers_ = num_layers;
  }

  void SetDataPort(uint32_t data_port) {
    absl::MutexLock lock(mu_);
    data_port_ = data_port;
  }

 private:
  absl::Mutex mu_;
  ::tpu_sync::proto::TransferControlStatusCode pull_stream_status_
      ABSL_GUARDED_BY(mu_) = ::tpu_sync::proto::STATUS_OK;
  std::string pull_stream_error_message_ ABSL_GUARDED_BY(mu_);
  absl::Duration pull_stream_delay_ ABSL_GUARDED_BY(mu_) = absl::ZeroDuration();
  uint32_t num_layers_ ABSL_GUARDED_BY(mu_) = 8;
  uint32_t data_port_ ABSL_GUARDED_BY(mu_) = 5001;

  bool ack_success_ ABSL_GUARDED_BY(mu_) = true;
  std::string ack_message_ ABSL_GUARDED_BY(mu_) = "ACK_OK";

  bool liveness_ready_ ABSL_GUARDED_BY(mu_) = true;
  int32_t active_workers_ ABSL_GUARDED_BY(mu_) = 4;
  std::string version_ ABSL_GUARDED_BY(mu_) = "1.0.0";
};

TEST(TransferControlServiceTest, InProcessPullStreamSuccess) {
  FakeTransferControlDelegate delegate;
  delegate.SetNumLayers(16);
  delegate.SetDataPort(9001);

  TransferControlServer server(&delegate,
                               TransferControlServer::kInProcessPort);
  EXPECT_EQ(server.port(), TransferControlServer::kInProcessPort);

  auto channel = server.InProcessChannel();
  ASSERT_NE(channel, nullptr);

  TransferControlClient client;
  client.RegisterInProcessChannel("inproc://transfer_control", channel);

  ::tpu_sync::proto::PullStreamRequest request;
  request.set_uuid(1001);
  request.set_req_id("req_1001");
  request.set_ep_idx(0);
  request.set_consumer_data_port(8001);
  request.add_consumer_ips("127.0.0.1");
  request.add_src_block_ids(0);
  request.add_src_block_ids(1);
  request.add_dst_block_ids(10);
  request.add_dst_block_ids(11);
  request.set_timeout_ms(5000);

  ASSERT_OK_AND_ASSIGN(auto pull_response,
                       client.PullStream("inproc://transfer_control", request,
                                         absl::Seconds(5)));

  EXPECT_TRUE(pull_response.success());
  EXPECT_THAT(pull_response.status(), Eq(::tpu_sync::proto::STATUS_OK));
  EXPECT_EQ(pull_response.num_layers(), 16);
  EXPECT_EQ(pull_response.data_port(), 9001);
}

TEST(TransferControlServiceTest, PullStreamStatusTranslation) {
  FakeTransferControlDelegate delegate;
  TransferControlServer server(&delegate,
                               TransferControlServer::kInProcessPort);
  auto channel = server.InProcessChannel();
  ASSERT_NE(channel, nullptr);

  TransferControlClient client;
  client.RegisterInProcessChannel("inproc://transfer_control", channel);

  ::tpu_sync::proto::PullStreamRequest request;
  request.set_uuid(2001);
  request.set_req_id("req_status_test");
  request.set_consumer_data_port(8001);

  // STATUS_NOT_REGISTERED -> NotFoundError
  delegate.SetPullStreamResponse(::tpu_sync::proto::STATUS_NOT_REGISTERED,
                                 "plan not found");
  auto not_found_status =
      client.PullStream("inproc://transfer_control", request, absl::Seconds(5))
          .status();
  EXPECT_THAT(not_found_status, StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(not_found_status.message(), Eq("plan not found"));

  // STATUS_BLOCK_VALIDATION_FAILED -> InvalidArgumentError
  delegate.SetPullStreamResponse(
      ::tpu_sync::proto::STATUS_BLOCK_VALIDATION_FAILED, "block mismatch");
  auto invalid_arg_status =
      client.PullStream("inproc://transfer_control", request, absl::Seconds(5))
          .status();
  EXPECT_THAT(invalid_arg_status, StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(invalid_arg_status.message(), Eq("block mismatch"));

  // STATUS_STAGING_UNAVAILABLE -> ResourceExhaustedError
  delegate.SetPullStreamResponse(::tpu_sync::proto::STATUS_STAGING_UNAVAILABLE,
                                 "host staging pool full");
  auto res_exhausted_status =
      client.PullStream("inproc://transfer_control", request, absl::Seconds(5))
          .status();
  EXPECT_THAT(res_exhausted_status,
              StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_THAT(res_exhausted_status.message(), Eq("host staging pool full"));

  // STATUS_SHUTTING_DOWN -> UnavailableError
  delegate.SetPullStreamResponse(::tpu_sync::proto::STATUS_SHUTTING_DOWN,
                                 "producer shutting down");
  auto unavailable_status =
      client.PullStream("inproc://transfer_control", request, absl::Seconds(5))
          .status();
  EXPECT_THAT(unavailable_status, StatusIs(absl::StatusCode::kUnavailable));
  EXPECT_THAT(unavailable_status.message(), Eq("producer shutting down"));

  // STATUS_INTERNAL_ERROR -> InternalError
  delegate.SetPullStreamResponse(::tpu_sync::proto::STATUS_INTERNAL_ERROR,
                                 "internal crash");
  auto internal_status =
      client.PullStream("inproc://transfer_control", request, absl::Seconds(5))
          .status();
  EXPECT_THAT(internal_status, StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(internal_status.message(), Eq("internal crash"));
}

TEST(TransferControlServiceTest, PullStreamDeadlineTimeout) {
  FakeTransferControlDelegate delegate;
  delegate.SetPullStreamDelay(absl::Milliseconds(500));

  TransferControlServer server(&delegate,
                               TransferControlServer::kInProcessPort);
  auto channel = server.InProcessChannel();
  ASSERT_NE(channel, nullptr);

  TransferControlClient client;
  client.RegisterInProcessChannel("inproc://transfer_control", channel);

  ::tpu_sync::proto::PullStreamRequest request;
  request.set_uuid(3001);
  request.set_req_id("req_timeout");
  request.set_consumer_data_port(8001);

  // Client timeout of 50ms should expire before the 500ms delay.
  auto timeout_status = client
                            .PullStream("inproc://transfer_control", request,
                                        absl::Milliseconds(50))
                            .status();
  EXPECT_THAT(timeout_status, StatusIs(absl::StatusCode::kDeadlineExceeded));
}

TEST(TransferControlServiceTest, AckSuccess) {
  FakeTransferControlDelegate delegate;
  delegate.SetAckResponse(true, "ACK_SETTLED");

  TransferControlServer server(&delegate,
                               TransferControlServer::kInProcessPort);
  auto channel = server.InProcessChannel();
  ASSERT_NE(channel, nullptr);

  TransferControlClient client;
  client.RegisterInProcessChannel("inproc://transfer_control", channel);

  ::tpu_sync::proto::TransferAckRequest request;
  request.set_uuid(4001);
  request.set_req_id("req_ack");
  request.set_success(true);
  request.set_message("Transfer completed successfully");

  ASSERT_OK_AND_ASSIGN(
      auto ack_response,
      client.Ack("inproc://transfer_control", request, absl::Seconds(5)));

  EXPECT_TRUE(ack_response.success());
  EXPECT_THAT(ack_response.message(), Eq("ACK_SETTLED"));
}

TEST(TransferControlServiceTest, CheckLivenessSuccess) {
  FakeTransferControlDelegate delegate;
  delegate.SetLivenessResponse(/*ready=*/true, /*active_workers=*/8,
                               /*version=*/"2.5.0");

  TransferControlServer server(&delegate,
                               TransferControlServer::kInProcessPort);
  auto channel = server.InProcessChannel();
  ASSERT_NE(channel, nullptr);

  TransferControlClient client;
  client.RegisterInProcessChannel("inproc://transfer_control", channel);

  ::tpu_sync::proto::TransferLivenessRequest request;
  request.set_client_id("test_client");

  ASSERT_OK_AND_ASSIGN(auto liveness_response,
                       client.CheckLiveness("inproc://transfer_control",
                                            request, absl::Seconds(5)));

  EXPECT_TRUE(liveness_response.ready());
  EXPECT_EQ(liveness_response.active_workers(), 8);
  EXPECT_THAT(liveness_response.version(), Eq("2.5.0"));
}

TEST(TransferControlServiceTest, DynamicPortAllocation) {
  FakeTransferControlDelegate delegate;
  // Port 0 binds to an ephemeral TCP port.
  TransferControlServer server(&delegate, /*port=*/0);
  EXPECT_GT(server.port(), 0);
  EXPECT_TRUE(server.is_running());

  TransferControlClient client;
  std::string endpoint = absl::StrCat("127.0.0.1:", server.port());

  ::tpu_sync::proto::TransferLivenessRequest request;
  request.set_client_id("tcp_client");

  ASSERT_OK_AND_ASSIGN(
      auto response, client.CheckLiveness(endpoint, request, absl::Seconds(5)));

  EXPECT_TRUE(response.ready());
}

}  // namespace
}  // namespace tpu_raiden
