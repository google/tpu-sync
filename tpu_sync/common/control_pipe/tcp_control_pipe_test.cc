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

#include "tpu_sync/common/control_pipe/tcp_control_pipe.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <thread>  // NOLINT(build/c++11)

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/common/control_pipe/control_dispatcher.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/proto/control_pipe.pb.h"
#include "tpu_sync/proto/kv_cache_control_plane_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace {

using ::tpu_raiden::control_plane::proto::AckRequest;
using ::tpu_raiden::control_plane::proto::PullStreamRequest;
using ::tpu_raiden::control_plane::proto::PullStreamResponse;

int ConnectLocalhost(int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
  if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(fd);
    return -1;
  }
  return fd;
}

TEST(TcpControlPipeTest, CpipEnvelopeRoundtripAndConnectionPooling) {
  ControlPipeConfig cfg;
  cfg.enable_tcp_connection_pooling = true;

  TcpControlPipeServer server(cfg);
  server.dispatcher().RegisterHandler<PullStreamRequest, PullStreamResponse>(
      [](const ControlContext& ctx,
         const PullStreamRequest& req) -> absl::StatusOr<PullStreamResponse> {
        PullStreamResponse resp;
        resp.set_status(0);
        resp.set_num_layers(16);
        resp.set_data_port(req.consumer_data_port() + 1);
        return resp;
      });

  TF_ASSERT_OK_AND_ASSIGN(int port, server.Start(0));
  std::string endpoint = absl::StrCat("127.0.0.1:", port);

  TcpControlPipeClient client(cfg);
  EXPECT_EQ(client.PoolIdleCount(endpoint), 0);

  PullStreamRequest req;
  req.set_uuid(100);
  req.set_consumer_data_port(5000);

  TF_ASSERT_OK_AND_ASSIGN(
      PullStreamResponse resp1,
      (client.Call<PullStreamRequest, PullStreamResponse>(endpoint, req)));
  EXPECT_EQ(resp1.num_layers(), 16);
  EXPECT_EQ(resp1.data_port(), 5001);
  EXPECT_EQ(client.PoolIdleCount(endpoint), 1);

  // Second call reuses the pooled connection.
  req.set_consumer_data_port(6000);
  TF_ASSERT_OK_AND_ASSIGN(
      PullStreamResponse resp2,
      (client.Call<PullStreamRequest, PullStreamResponse>(endpoint, req)));
  EXPECT_EQ(resp2.data_port(), 6001);
  EXPECT_EQ(client.PoolIdleCount(endpoint), 1);

  server.Stop();
}

TEST(TcpControlPipeTest, LegacyRaidBinaryStructInteroperability) {
  ControlPipeConfig cfg;
  cfg.allow_legacy_framing = true;

  TcpControlPipeServer server(cfg);
  server.dispatcher().RegisterHandler<PullStreamRequest, PullStreamResponse>(
      [](const ControlContext& ctx,
         const PullStreamRequest& req) -> absl::StatusOr<PullStreamResponse> {
        EXPECT_EQ(req.uuid(), 888);
        EXPECT_EQ(req.src_block_ids_size(), 2);
        EXPECT_EQ(req.src_block_ids(0), 10);
        EXPECT_EQ(req.src_block_ids(1), 20);
        PullStreamResponse resp;
        resp.set_status(0);
        resp.set_num_layers(24);
        resp.set_data_port(7777);
        return resp;
      });

  TF_ASSERT_OK_AND_ASSIGN(int port, server.Start(0));

  int fd = ConnectLocalhost(port);
  ASSERT_GE(fd, 0);

  TcpControlPipeServer::LegacyRaidRequestHeader req_hdr{};
  req_hdr.magic = TcpControlPipeServer::kRaidControlMagic;
  req_hdr.op = TcpControlPipeServer::kRaidOpPullStream;
  req_hdr.uuid = 888;
  req_hdr.num_blocks = 2;

  int64_t src_blocks[2] = {10, 20};
  int64_t dst_blocks[2] = {30, 40};

  ASSERT_EQ(send(fd, &req_hdr, sizeof(req_hdr), 0), sizeof(req_hdr));
  ASSERT_EQ(send(fd, src_blocks, sizeof(src_blocks), 0), sizeof(src_blocks));
  ASSERT_EQ(send(fd, dst_blocks, sizeof(dst_blocks), 0), sizeof(dst_blocks));

  TcpControlPipeServer::LegacyRaidResponseHeader resp_hdr{};
  ASSERT_EQ(recv(fd, &resp_hdr, sizeof(resp_hdr), MSG_WAITALL),
            sizeof(resp_hdr));
  EXPECT_EQ(resp_hdr.magic, TcpControlPipeServer::kRaidResponseMagic);
  EXPECT_EQ(resp_hdr.status, 0);
  EXPECT_EQ(resp_hdr.num_layers, 24);
  EXPECT_EQ(resp_hdr.data_port, 7777);

  close(fd);
  server.Stop();
}

TEST(TcpControlPipeTest, Legacy4ByteLengthPrefixedControlRequestInterop) {
  ControlPipeConfig cfg;
  cfg.allow_legacy_framing = true;

  TcpControlPipeServer server(cfg);
  server.dispatcher()
      .RegisterHandler<::tpu_sync::rpc::ControlRequest,
                       ::tpu_sync::rpc::ControlResponse>(
          [](const ControlContext& ctx,
             const ::tpu_sync::rpc::ControlRequest& req)
              -> absl::StatusOr<::tpu_sync::rpc::ControlResponse> {
            ::tpu_sync::rpc::ControlResponse resp;
            resp.set_success(true);
            resp.set_message("pong");
            return resp;
          });

  TF_ASSERT_OK_AND_ASSIGN(int port, server.Start(0));

  int fd = ConnectLocalhost(port);
  ASSERT_GE(fd, 0);

  ::tpu_sync::rpc::ControlRequest rpc_req;
  rpc_req.set_command(::tpu_sync::rpc::ControlRequest::COMMAND_SHUTDOWN);
  std::string req_bytes;
  ASSERT_TRUE(rpc_req.SerializeToString(&req_bytes));

  uint32_t net_len = htonl(static_cast<uint32_t>(req_bytes.size()));
  ASSERT_EQ(send(fd, &net_len, sizeof(net_len), 0), sizeof(net_len));
  ASSERT_EQ(send(fd, req_bytes.data(), req_bytes.size(), 0), req_bytes.size());

  uint32_t resp_net_len = 0;
  ASSERT_EQ(recv(fd, &resp_net_len, sizeof(resp_net_len), MSG_WAITALL),
            sizeof(resp_net_len));
  uint32_t resp_len = ntohl(resp_net_len);
  std::string resp_bytes(resp_len, '\0');
  ASSERT_EQ(recv(fd, resp_bytes.data(), resp_len, MSG_WAITALL), resp_len);

  ::tpu_sync::rpc::ControlResponse rpc_resp;
  ASSERT_TRUE(rpc_resp.ParseFromString(resp_bytes));
  EXPECT_TRUE(rpc_resp.success());
  EXPECT_EQ(rpc_resp.message(), "pong");

  close(fd);
  server.Stop();
}

TEST(TcpControlPipeTest, InstantStopUnblocksIdlePooledClientConnections) {
  ControlPipeConfig cfg;
  cfg.enable_tcp_connection_pooling = true;

  TcpControlPipeServer server(cfg);
  server.dispatcher().RegisterOneWayHandler<AckRequest>(
      [](const ControlContext& ctx, const AckRequest& req) {
        return absl::OkStatus();
      });

  TF_ASSERT_OK_AND_ASSIGN(int port, server.Start(0));
  std::string endpoint = absl::StrCat("127.0.0.1:", port);

  TcpControlPipeClient client(cfg);
  AckRequest ack;
  ack.set_uuid(123);
  absl::Status ack_status = client.SendOneWay(endpoint, ack);
  ASSERT_TRUE(ack_status.ok()) << ack_status;
  EXPECT_EQ(client.PoolIdleCount(endpoint), 1);

  // Server currently has a handler thread blocked in ReadExactOrEof on the
  // pooled socket. Stop() must shut down active_client_fds_ and return rapidly.
  absl::Time start_time = absl::Now();
  server.Stop();
  absl::Duration stop_duration = absl::Now() - start_time;
  EXPECT_LT(stop_duration, absl::Milliseconds(500));
}

TEST(TcpControlPipeTest, OversizedFrameRejectedPriorToAllocation) {
  ControlPipeConfig cfg;
  cfg.max_frame_bytes = 512;

  TcpControlPipeServer server(cfg);
  server.dispatcher().RegisterOneWayHandler<AckRequest>(
      [](const ControlContext& ctx, const AckRequest& req) {
        return absl::OkStatus();
      });

  TF_ASSERT_OK_AND_ASSIGN(int port, server.Start(0));

  int fd = ConnectLocalhost(port);
  ASSERT_GE(fd, 0);

  // Send "CPIP" + 4B length claiming 100 MB payload.
  const char magic[4] = {'C', 'P', 'I', 'P'};
  uint32_t net_len = htonl(100 * 1024 * 1024);
  ASSERT_EQ(send(fd, magic, 4, 0), 4);
  ASSERT_EQ(send(fd, &net_len, sizeof(net_len), 0), sizeof(net_len));

  // Server should send a PIPC error response and close the connection.
  char resp_magic[4] = {0};
  ssize_t n = recv(fd, resp_magic, 4, MSG_WAITALL);
  if (n == 4) {
    EXPECT_EQ(std::memcmp(resp_magic, "PIPC", 4), 0);
  }
  close(fd);
  server.Stop();
}

TEST(TcpControlPipeTest, StopAllowsInFlightHandlerToWriteResponse) {
  ControlPipeConfig cfg;
  TcpControlPipeServer server(cfg);
  absl::Mutex mu;
  bool handler_entered = false;

  server.dispatcher().RegisterRawHandler(
      "test.SlowEcho",
      [&](const ControlContext& /*ctx*/,
          absl::string_view req_bytes) -> absl::StatusOr<std::string> {
        {
          absl::MutexLock lock(mu);
          handler_entered = true;
        }
        absl::SleepFor(absl::Milliseconds(300));
        return std::string(req_bytes);
      },
      cfg.max_frame_bytes);

  TF_ASSERT_OK_AND_ASSIGN(int port, server.Start(0));
  const std::string endpoint = absl::StrCat("127.0.0.1:", port);

  std::thread client_thread([&]() {
    TcpControlPipeClient client(cfg);
    control_pipe::proto::ControlEnvelope env;
    env.set_message_type("test.SlowEcho");
    env.set_request_id(1);
    env.set_payload("in-flight-ok");
    auto resp = client.SendRaw(endpoint, env, absl::Seconds(10));
    ASSERT_TRUE(resp.ok()) << resp.status();
    EXPECT_EQ(resp->status_code(), 0);
    EXPECT_EQ(resp->payload(), "in-flight-ok");
  });

  {
    absl::MutexLock lock(mu);
    mu.Await(absl::Condition(&handler_entered));
  }
  server.Stop();
  client_thread.join();
}

}  // namespace
}  // namespace tpu_raiden
