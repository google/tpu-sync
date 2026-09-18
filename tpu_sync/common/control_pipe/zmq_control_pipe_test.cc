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

#include "tpu_sync/common/control_pipe/zmq_control_pipe.h"

#include <atomic>
#include <functional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/common/control_pipe/control_dispatcher.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/proto/control_pipe.pb.h"
#include "tpu_sync/proto/kv_cache_control_plane_service.pb.h"
#include "zmq.h"

namespace tpu_raiden {
namespace {

using ::tpu_raiden::control_plane::proto::AckRequest;
using ::tpu_raiden::control_plane::proto::PullStreamRequest;
using ::tpu_raiden::control_plane::proto::PullStreamResponse;

TEST(ZmqControlPipeTest, SynchronousDispatchRoundtrip) {
  ControlPipeConfig cfg;
  cfg.backend_type = ControlPipeBackendType::kZmq;
  ZmqControlPipeServer server(cfg);

  server.dispatcher().RegisterHandler<PullStreamRequest, PullStreamResponse>(
      [](const ControlContext& ctx,
         const PullStreamRequest& req) -> absl::StatusOr<PullStreamResponse> {
        EXPECT_EQ(ctx.backend_type, ControlPipeBackendType::kZmq);
        PullStreamResponse resp;
        resp.set_status(0);
        resp.set_num_layers(32);
        resp.set_data_port(req.consumer_data_port() + 100);
        return resp;
      });

  TF_ASSERT_OK_AND_ASSIGN(int port, server.Start(0));
  EXPECT_GT(port, 0);
  EXPECT_EQ(server.bound_port(), port);
  std::string endpoint = absl::StrCat("127.0.0.1:", port);

  ZmqControlPipeClient client(cfg);
  PullStreamRequest req;
  req.set_uuid(1234);
  req.set_consumer_data_port(8000);

  TF_ASSERT_OK_AND_ASSIGN(
      PullStreamResponse resp,
      (client.Call<PullStreamRequest, PullStreamResponse>(endpoint, req)));
  EXPECT_EQ(resp.num_layers(), 32);
  EXPECT_EQ(resp.data_port(), 8100);

  server.Stop();
}

TEST(ZmqControlPipeTest, TaskExecutorOffloadedDispatch) {
  std::atomic<int> executor_calls{0};
  ControlPipeConfig cfg;
  cfg.backend_type = ControlPipeBackendType::kZmq;
  cfg.executor = [&executor_calls](std::function<void()> task) {
    ++executor_calls;
    std::thread(std::move(task)).detach();
  };

  ZmqControlPipeServer server(cfg);
  server.dispatcher().RegisterHandler<PullStreamRequest, PullStreamResponse>(
      [](const ControlContext& ctx,
         const PullStreamRequest& req) -> absl::StatusOr<PullStreamResponse> {
        PullStreamResponse resp;
        resp.set_status(0);
        resp.set_num_layers(64);
        resp.set_data_port(req.consumer_data_port() + 1);
        return resp;
      });

  TF_ASSERT_OK_AND_ASSIGN(int port, server.Start(0));
  std::string endpoint = absl::StrCat("127.0.0.1:", port);

  ZmqControlPipeClient client(cfg);
  for (int i = 0; i < 5; ++i) {
    PullStreamRequest req;
    req.set_uuid(i);
    req.set_consumer_data_port(5000 + i);
    TF_ASSERT_OK_AND_ASSIGN(
        PullStreamResponse resp,
        (client.Call<PullStreamRequest, PullStreamResponse>(endpoint, req)));
    EXPECT_EQ(resp.num_layers(), 64);
    EXPECT_EQ(resp.data_port(), 5001 + i);
  }
  EXPECT_EQ(executor_calls.load(), 5);

  server.Stop();
}

TEST(ZmqControlPipeTest, MultiThreadedConcurrentRequests) {
  ControlPipeConfig cfg;
  cfg.backend_type = ControlPipeBackendType::kZmq;
  cfg.executor = [](std::function<void()> task) {
    std::thread(std::move(task)).detach();
  };

  ZmqControlPipeServer server(cfg);
  server.dispatcher().RegisterHandler<PullStreamRequest, PullStreamResponse>(
      [](const ControlContext& ctx,
         const PullStreamRequest& req) -> absl::StatusOr<PullStreamResponse> {
        PullStreamResponse resp;
        resp.set_status(0);
        resp.set_data_port(req.consumer_data_port() * 2);
        return resp;
      });

  TF_ASSERT_OK_AND_ASSIGN(int port, server.Start(0));
  std::string endpoint = absl::StrCat("127.0.0.1:", port);

  ZmqControlPipeClient client(cfg);
  constexpr int kNumThreads = 8;
  constexpr int kCallsPerThread = 10;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&client, &endpoint, t]() {
      for (int i = 0; i < kCallsPerThread; ++i) {
        int port_in = (t + 1) * 100 + i;
        PullStreamRequest req;
        req.set_uuid(port_in);
        req.set_consumer_data_port(port_in);
        absl::StatusOr<PullStreamResponse> resp =
            client.Call<PullStreamRequest, PullStreamResponse>(endpoint, req);
        ASSERT_TRUE(resp.ok()) << resp.status();
        EXPECT_EQ(resp->data_port(), port_in * 2);
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  server.Stop();
}

TEST(ZmqControlPipeTest, OversizedFrameRejection) {
  ControlPipeConfig server_cfg;
  server_cfg.backend_type = ControlPipeBackendType::kZmq;
  server_cfg.max_frame_bytes = 512;  // Small limit for testing

  ZmqControlPipeServer server(server_cfg);
  server.dispatcher().RegisterOneWayHandler<AckRequest>(
      [](const ControlContext& ctx, const AckRequest& req) {
        return absl::OkStatus();
      });

  TF_ASSERT_OK_AND_ASSIGN(int port, server.Start(0));
  std::string endpoint = absl::StrCat("127.0.0.1:", port);

  // 1. Client-side check when client config has small max_frame_bytes
  ZmqControlPipeClient client_small(server_cfg);
  AckRequest huge_ack;
  huge_ack.set_uuid(1);
  control_pipe::proto::ControlEnvelope env;
  env.set_message_type(AckRequest::descriptor()->full_name());
  env.set_payload(std::string(1024, 'x'));
  absl::StatusOr<control_pipe::proto::ControlResponseEnvelope> response =
      client_small.SendRaw(endpoint, env, absl::Seconds(2));
  EXPECT_FALSE(response.ok());
  EXPECT_EQ(response.status().code(), absl::StatusCode::kResourceExhausted);

  // 2. Server-side ZMQ_MAXMSGSIZE check when raw ZMQ client sends > 512 bytes
  void* raw_ctx = zmq_ctx_new();
  void* raw_sock = zmq_socket(raw_ctx, ZMQ_REQ);
  int linger = 0;
  zmq_setsockopt(raw_sock, ZMQ_LINGER, &linger, sizeof(linger));
  int timeout_ms = 500;
  zmq_setsockopt(raw_sock, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
  std::string tcp_ep = absl::StrCat("tcp://127.0.0.1:", port);
  ASSERT_EQ(zmq_connect(raw_sock, tcp_ep.c_str()), 0);

  std::string oversized_payload(2048, 'A');
  (void)zmq_send(raw_sock, oversized_payload.data(), oversized_payload.size(),
                 0);
  zmq_msg_t reply;
  zmq_msg_init(&reply);
  int rc = zmq_msg_recv(&reply, raw_sock, 0);
  // Server drops oversized message due to ZMQ_MAXMSGSIZE, causing recv timeout
  EXPECT_LT(rc, 0);
  zmq_msg_close(&reply);
  zmq_close(raw_sock);
  zmq_ctx_term(raw_ctx);

  server.Stop();
}

TEST(ZmqControlPipeTest, RequestTimeoutHandling) {
  ControlPipeConfig cfg;
  cfg.backend_type = ControlPipeBackendType::kZmq;
  cfg.executor = [](std::function<void()> task) {
    std::thread(std::move(task)).detach();
  };

  ZmqControlPipeServer server(cfg);
  server.dispatcher().RegisterHandler<PullStreamRequest, PullStreamResponse>(
      [](const ControlContext& ctx,
         const PullStreamRequest& req) -> absl::StatusOr<PullStreamResponse> {
        absl::SleepFor(absl::Milliseconds(400));
        PullStreamResponse resp;
        resp.set_status(0);
        return resp;
      });

  TF_ASSERT_OK_AND_ASSIGN(int port, server.Start(0));
  std::string endpoint = absl::StrCat("127.0.0.1:", port);

  ZmqControlPipeClient client(cfg);
  PullStreamRequest req;
  req.set_uuid(999);

  // Request with 100ms timeout should fail with DeadlineExceededError
  absl::StatusOr<PullStreamResponse> timed_out_resp =
      client.Call<PullStreamRequest, PullStreamResponse>(
          endpoint, req, absl::Milliseconds(100));
  EXPECT_FALSE(timed_out_resp.ok());
  EXPECT_EQ(timed_out_resp.status().code(),
            absl::StatusCode::kDeadlineExceeded);

  // Subsequent request on the same client with sufficient timeout should
  // succeed because the timed-out ZMQ_REQ socket was discarded from the pool.
  TF_ASSERT_OK_AND_ASSIGN(PullStreamResponse ok_resp,
                          (client.Call<PullStreamRequest, PullStreamResponse>(
                              endpoint, req, absl::Seconds(5))));
  EXPECT_EQ(ok_resp.status(), 0);

  server.Stop();
}

}  // namespace
}  // namespace tpu_raiden
