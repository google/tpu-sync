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

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/proto/control_pipe.pb.h"
#include "tpu_sync/proto/kv_cache_control_plane_service.pb.h"

namespace tpu_raiden {
namespace {

using ::tpu_raiden::control_plane::proto::AckRequest;
using ::tpu_raiden::control_plane::proto::PullStreamRequest;
using ::tpu_raiden::control_plane::proto::PullStreamResponse;

TEST(ControlDispatcherTest, UnregisteredMessageTypeReturnsUnimplemented) {
  ControlDispatcher dispatcher;
  ControlContext ctx;
  ctx.peer_ip = "127.0.0.1";

  control_pipe::proto::ControlEnvelope env;
  env.set_message_type("unknown.package.SomeRequest");
  env.set_request_id(101);
  env.set_payload("dummy");

  control_pipe::proto::ControlResponseEnvelope resp =
      dispatcher.Dispatch(ctx, env);
  EXPECT_EQ(resp.request_id(), 101);
  EXPECT_EQ(resp.status_code(),
            static_cast<int32_t>(absl::StatusCode::kUnimplemented));
  EXPECT_TRUE(absl::StrContains(resp.error_message(),
                                "Unregistered control message type"));
}

TEST(ControlDispatcherTest, OversizedPayloadRejectedBeforeHandler) {
  ControlDispatcher dispatcher;
  std::atomic<bool> handler_invoked{false};

  dispatcher.RegisterOneWayHandler<AckRequest>(
      [&handler_invoked](const ControlContext& ctx, const AckRequest& req) {
        handler_invoked.store(true);
        return absl::OkStatus();
      },
      HandlerOptions<AckRequest>().WithMaxPayloadBytes(64));

  ControlContext ctx;
  control_pipe::proto::ControlEnvelope env;
  env.set_message_type(AckRequest::descriptor()->full_name());
  env.set_request_id(102);
  env.set_payload(std::string(128, 'A'));  // Exceeds 64 bytes cap

  control_pipe::proto::ControlResponseEnvelope resp =
      dispatcher.Dispatch(ctx, env);
  EXPECT_EQ(resp.request_id(), 102);
  EXPECT_EQ(resp.status_code(),
            static_cast<int32_t>(absl::StatusCode::kResourceExhausted));
  EXPECT_TRUE(
      absl::StrContains(resp.error_message(), "exceeds max_payload_bytes"));
  EXPECT_FALSE(handler_invoked.load());
}

TEST(ControlDispatcherTest, ValidatorRejectionPreventsHandlerExecution) {
  ControlDispatcher dispatcher;
  std::atomic<bool> handler_invoked{false};

  dispatcher.RegisterHandler<PullStreamRequest, PullStreamResponse>(
      [&handler_invoked](
          const ControlContext& ctx,
          const PullStreamRequest& req) -> absl::StatusOr<PullStreamResponse> {
        handler_invoked.store(true);
        PullStreamResponse resp;
        resp.set_status(0);
        return resp;
      },
      HandlerOptions<PullStreamRequest>()
          .WithMaxPayloadBytes(1024)
          .WithValidator([](const ControlContext& ctx,
                            const PullStreamRequest& req) -> absl::Status {
            if (req.src_block_ids_size() != req.dst_block_ids_size()) {
              return absl::InvalidArgumentError(
                  "src_block_ids and dst_block_ids count mismatch");
            }
            return absl::OkStatus();
          }));

  PullStreamRequest req;
  req.set_uuid(999);
  req.add_src_block_ids(1);
  req.add_src_block_ids(2);
  req.add_dst_block_ids(10);  // Mismatch

  ControlContext ctx;
  control_pipe::proto::ControlEnvelope env;
  env.set_message_type(PullStreamRequest::descriptor()->full_name());
  env.set_request_id(103);
  ASSERT_TRUE(req.SerializeToString(env.mutable_payload()));

  control_pipe::proto::ControlResponseEnvelope resp =
      dispatcher.Dispatch(ctx, env);
  EXPECT_EQ(resp.request_id(), 103);
  EXPECT_EQ(resp.status_code(),
            static_cast<int32_t>(absl::StatusCode::kInvalidArgument));
  EXPECT_TRUE(absl::StrContains(
      resp.error_message(), "src_block_ids and dst_block_ids count mismatch"));
  EXPECT_FALSE(handler_invoked.load());
}

TEST(ControlDispatcherTest, RequestResponseAndOneWayRoundtrip) {
  ControlDispatcher dispatcher;

  dispatcher.RegisterHandler<PullStreamRequest, PullStreamResponse>(
      [](const ControlContext& ctx,
         const PullStreamRequest& req) -> absl::StatusOr<PullStreamResponse> {
        PullStreamResponse resp;
        resp.set_status(0);
        resp.set_num_layers(32);
        resp.set_data_port(req.consumer_data_port() + 100);
        return resp;
      });

  std::atomic<uint64_t> last_ack_uuid{0};
  dispatcher.RegisterOneWayHandler<AckRequest>(
      [&last_ack_uuid](const ControlContext& ctx, const AckRequest& req) {
        last_ack_uuid.store(req.uuid());
        return absl::OkStatus();
      });

  // Test Request-Response
  PullStreamRequest pull_req;
  pull_req.set_uuid(12345);
  pull_req.set_consumer_data_port(8000);

  ControlContext ctx;
  control_pipe::proto::ControlEnvelope pull_env;
  pull_env.set_message_type(PullStreamRequest::descriptor()->full_name());
  pull_env.set_request_id(1);
  ASSERT_TRUE(pull_req.SerializeToString(pull_env.mutable_payload()));

  control_pipe::proto::ControlResponseEnvelope pull_resp_env =
      dispatcher.Dispatch(ctx, pull_env);
  EXPECT_EQ(pull_resp_env.status_code(), 0);
  PullStreamResponse pull_resp;
  ASSERT_TRUE(pull_resp.ParseFromString(pull_resp_env.payload()));
  EXPECT_EQ(pull_resp.num_layers(), 32);
  EXPECT_EQ(pull_resp.data_port(), 8100);

  // Test One-Way
  AckRequest ack_req;
  ack_req.set_uuid(54321);
  control_pipe::proto::ControlEnvelope ack_env;
  ack_env.set_message_type(AckRequest::descriptor()->full_name());
  ack_env.set_request_id(2);
  ASSERT_TRUE(ack_req.SerializeToString(ack_env.mutable_payload()));

  control_pipe::proto::ControlResponseEnvelope ack_resp_env =
      dispatcher.Dispatch(ctx, ack_env);
  EXPECT_EQ(ack_resp_env.status_code(), 0);
  EXPECT_EQ(last_ack_uuid.load(), 54321);
}

TEST(ControlDispatcherTest, ConcurrentDispatchThreadSafety) {
  ControlDispatcher dispatcher;
  std::atomic<int> total_dispatches{0};

  dispatcher.RegisterOneWayHandler<AckRequest>(
      [&total_dispatches](const ControlContext& ctx, const AckRequest& req) {
        total_dispatches.fetch_add(1, std::memory_order_relaxed);
        return absl::OkStatus();
      });

  AckRequest req;
  req.set_uuid(777);
  control_pipe::proto::ControlEnvelope env;
  env.set_message_type(AckRequest::descriptor()->full_name());
  ASSERT_TRUE(req.SerializeToString(env.mutable_payload()));

  constexpr int kNumThreads = 8;
  constexpr int kCallsPerThread = 100;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&dispatcher, &env]() {
      ControlContext ctx;
      for (int j = 0; j < kCallsPerThread; ++j) {
        control_pipe::proto::ControlResponseEnvelope resp =
            dispatcher.Dispatch(ctx, env);
        EXPECT_EQ(resp.status_code(), 0);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
  EXPECT_EQ(total_dispatches.load(), kNumThreads * kCallsPerThread);
}

}  // namespace
}  // namespace tpu_raiden
