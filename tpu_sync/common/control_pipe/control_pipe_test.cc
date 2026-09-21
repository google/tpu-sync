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

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/status.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/common/control_pipe/control_dispatcher.h"
#include "tpu_sync/common/control_pipe/control_pipe_client.h"
#include "tpu_sync/common/control_pipe/control_pipe_server.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/common/control_pipe/grpc_control_pipe.h"
#include "tpu_sync/proto/control_pipe.grpc.pb.h"
#include "tpu_sync/proto/control_pipe.pb.h"
#include "tpu_sync/proto/kv_cache_control_plane_service.grpc.pb.h"
#include "tpu_sync/proto/kv_cache_control_plane_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace {

using ::tpu_raiden::control_plane::proto::AckRequest;
using ::tpu_raiden::control_plane::proto::AckResponse;
using ::tpu_raiden::control_plane::proto::KVCacheControlPlaneService;
using ::tpu_raiden::control_plane::proto::PullStreamRequest;
using ::tpu_raiden::control_plane::proto::PullStreamResponse;

bool ReadAll(int fd, void* buf, size_t len) {
  char* p = static_cast<char*>(buf);
  size_t total = 0;
  while (total < len) {
    ssize_t n = recv(fd, p + total, len - total, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    total += static_cast<size_t>(n);
  }
  return true;
}

bool WriteAll(int fd, const void* buf, size_t len) {
  const char* p = static_cast<const char*>(buf);
  size_t total = 0;
  while (total < len) {
    ssize_t n = send(fd, p + total, len - total, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    total += static_cast<size_t>(n);
  }
  return true;
}

absl::StatusOr<::tpu_sync::rpc::ControlResponse> SendLegacyTcpControlRequest(
    int port, const ::tpu_sync::rpc::ControlRequest& req) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return absl::InternalError("socket failed");
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
  if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(fd);
    return absl::UnavailableError("connect failed");
  }
  std::string payload = req.SerializeAsString();
  uint32_t net_len = htonl(static_cast<uint32_t>(payload.size()));
  if (!WriteAll(fd, &net_len, sizeof(net_len)) ||
      !WriteAll(fd, payload.data(), payload.size())) {
    close(fd);
    return absl::UnavailableError("write failed");
  }
  uint32_t resp_net_len = 0;
  if (!ReadAll(fd, &resp_net_len, sizeof(resp_net_len))) {
    close(fd);
    return absl::UnavailableError("read length failed");
  }
  uint32_t resp_len = ntohl(resp_net_len);
  std::string resp_bytes(resp_len, '\0');
  if (resp_len > 0 && !ReadAll(fd, resp_bytes.data(), resp_len)) {
    close(fd);
    return absl::UnavailableError("read payload failed");
  }
  close(fd);
  ::tpu_sync::rpc::ControlResponse resp;
  if (!resp.ParseFromString(resp_bytes)) {
    return absl::InternalError("parse response failed");
  }
  return resp;
}

class LegacyTcpWeightSyncServer {
 public:
  LegacyTcpWeightSyncServer() {
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    socklen_t len = sizeof(addr);
    getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len);
    port_ = ntohs(addr.sin_port);
    listen(listen_fd_, 16);
    thread_ = std::thread([this]() { ServeLoop(); });
  }

  ~LegacyTcpWeightSyncServer() { Stop(); }

  int port() const { return port_; }

  void Stop() {
    if (!stopped_.exchange(true)) {
      shutdown(listen_fd_, SHUT_RDWR);
      close(listen_fd_);
      if (thread_.joinable()) thread_.join();
    }
  }

 private:
  void ServeLoop() {
    while (!stopped_.load()) {
      sockaddr_in client_addr{};
      socklen_t client_len = sizeof(client_addr);
      int client_fd = accept(
          listen_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
      if (client_fd < 0) {
        if (errno == EINTR && !stopped_.load()) continue;
        break;
      }
      uint32_t net_len = 0;
      if (ReadAll(client_fd, &net_len, sizeof(net_len))) {
        uint32_t msg_len = ntohl(net_len);
        // Reject CPIP magic header to prove strict legacy-only framing.
        if (std::memcmp(&net_len, "CPIP", 4) != 0 && msg_len <= 1024 * 1024) {
          std::string buf(msg_len, '\0');
          if (ReadAll(client_fd, buf.data(), msg_len)) {
            ::tpu_sync::rpc::ControlRequest req;
            if (req.ParseFromString(buf)) {
              ::tpu_sync::rpc::ControlResponse resp;
              resp.set_success(true);
              resp.set_message("old_tcp_server_ok");
              std::string out = resp.SerializeAsString();
              uint32_t out_net_len = htonl(static_cast<uint32_t>(out.size()));
              WriteAll(client_fd, &out_net_len, sizeof(out_net_len));
              WriteAll(client_fd, out.data(), out.size());
            }
          }
        }
      }
      close(client_fd);
    }
  }

  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> stopped_{false};
  std::thread thread_;
};

class ControlPipeBackendTest
    : public ::testing::TestWithParam<ControlPipeBackendType> {};

TEST_P(ControlPipeBackendTest, CallAndOneWayRoundtrip) {
  ControlPipeConfig cfg;
  cfg.backend_type = GetParam();

  std::unique_ptr<ControlPipeServer> server = CreateControlPipeServer(cfg);
  ASSERT_NE(server, nullptr);
  EXPECT_EQ(server->backend_type(), GetParam());

  server->dispatcher().RegisterHandler<PullStreamRequest, PullStreamResponse>(
      [](const ControlContext& ctx,
         const PullStreamRequest& req) -> absl::StatusOr<PullStreamResponse> {
        PullStreamResponse resp;
        resp.set_status(0);
        resp.set_num_layers(48);
        resp.set_data_port(req.consumer_data_port() + 10);
        return resp;
      });

  std::atomic<uint64_t> last_ack{0};
  server->dispatcher().RegisterOneWayHandler<AckRequest>(
      [&last_ack](const ControlContext& ctx, const AckRequest& req) {
        last_ack.store(req.uuid());
        return absl::OkStatus();
      });

  TF_ASSERT_OK_AND_ASSIGN(int port, server->Start(0));
  std::string endpoint = absl::StrCat("127.0.0.1:", port);

  std::unique_ptr<ControlPipeClient> client = CreateControlPipeClient(cfg);
  ASSERT_NE(client, nullptr);
  EXPECT_EQ(client->backend_type(), GetParam());

  PullStreamRequest pull_req;
  pull_req.set_uuid(555);
  pull_req.set_consumer_data_port(9000);

  TF_ASSERT_OK_AND_ASSIGN(PullStreamResponse pull_resp,
                          (client->Call<PullStreamRequest, PullStreamResponse>(
                              endpoint, pull_req)));
  EXPECT_EQ(pull_resp.num_layers(), 48);
  EXPECT_EQ(pull_resp.data_port(), 9010);

  AckRequest ack_req;
  ack_req.set_uuid(9999);
  absl::Status ack_status = client->SendOneWay(endpoint, ack_req);
  EXPECT_TRUE(ack_status.ok()) << ack_status;
  EXPECT_EQ(last_ack.load(), 9999);

  server->Stop();
}

TEST_P(ControlPipeBackendTest, RemoteErrorStatusPropagation) {
  ControlPipeConfig cfg;
  cfg.backend_type = GetParam();

  std::unique_ptr<ControlPipeServer> server = CreateControlPipeServer(cfg);
  server->dispatcher().RegisterHandler<PullStreamRequest, PullStreamResponse>(
      [](const ControlContext& ctx,
         const PullStreamRequest& req) -> absl::StatusOr<PullStreamResponse> {
        return absl::InvalidArgumentError("invalid block index");
      });

  TF_ASSERT_OK_AND_ASSIGN(int port, server->Start(0));
  std::string endpoint = absl::StrCat("127.0.0.1:", port);

  std::unique_ptr<ControlPipeClient> client = CreateControlPipeClient(cfg);
  PullStreamRequest req;
  req.set_uuid(1);
  absl::StatusOr<PullStreamResponse> resp =
      client->Call<PullStreamRequest, PullStreamResponse>(endpoint, req);
  EXPECT_FALSE(resp.ok());
  EXPECT_EQ(resp.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(resp.status().message(), "invalid block index");

  server->Stop();
}

INSTANTIATE_TEST_SUITE_P(
    AllBackends, ControlPipeBackendTest,
    ::testing::Values(ControlPipeBackendType::kTcp,
                      ControlPipeBackendType::kGrpc,
                      ControlPipeBackendType::kZmq),
    [](const ::testing::TestParamInfo<ControlPipeBackendType>& info) {
      return std::string(ControlPipeBackendTypeName(info.param));
    });

TEST(GrpcMultiServiceAdapterTest, ServesLegacyKVCacheAndWeightSyncGrpcClients) {
  ControlPipeConfig cfg;
  cfg.backend_type = ControlPipeBackendType::kGrpc;

  std::unique_ptr<ControlPipeServer> server = CreateControlPipeServer(cfg);
  server->dispatcher().RegisterHandler<PullStreamRequest, PullStreamResponse>(
      [](const ControlContext& ctx,
         const PullStreamRequest& req) -> absl::StatusOr<PullStreamResponse> {
        PullStreamResponse resp;
        resp.set_status(0);
        resp.set_num_layers(64);
        resp.set_data_port(1234);
        return resp;
      });
  server->dispatcher().RegisterOneWayHandler<AckRequest>(
      [](const ControlContext& ctx, const AckRequest& req) {
        return absl::OkStatus();
      });

  TF_ASSERT_OK_AND_ASSIGN(int port, server->Start(0));
  std::string endpoint = absl::StrCat("127.0.0.1:", port);

  auto channel =
      grpc::CreateChannel(endpoint, grpc::InsecureChannelCredentials());

  // 1. Call via legacy KVCacheControlPlaneService::Stub
  auto kv_stub = KVCacheControlPlaneService::NewStub(channel);
  {
    grpc::ClientContext ctx;
    PullStreamRequest req;
    req.set_uuid(77);
    PullStreamResponse resp;
    grpc::Status status = kv_stub->PullStream(&ctx, req, &resp);
    ASSERT_TRUE(status.ok()) << status.error_message();
    EXPECT_EQ(resp.status(), 0);
    EXPECT_EQ(resp.num_layers(), 64);
    EXPECT_EQ(resp.data_port(), 1234);
  }
  {
    grpc::ClientContext ctx;
    AckRequest req;
    req.set_uuid(77);
    AckResponse resp;
    grpc::Status status = kv_stub->Ack(&ctx, req, &resp);
    ASSERT_TRUE(status.ok()) << status.error_message();
    EXPECT_EQ(resp.status(), 0);
  }

  // 2. Call bidirectional ControlStream on ControlPipeService::Stub
  auto pipe_stub = control_pipe::proto::ControlPipeService::NewStub(channel);
  {
    grpc::ClientContext ctx;
    auto stream = pipe_stub->ControlStream(&ctx);
    AckRequest ack;
    ack.set_uuid(4242);
    control_pipe::proto::ControlEnvelope env;
    env.set_message_type(AckRequest::descriptor()->full_name());
    env.set_request_id(99);
    ASSERT_TRUE(ack.SerializeToString(env.mutable_payload()));

    ASSERT_TRUE(stream->Write(env));
    control_pipe::proto::ControlResponseEnvelope resp_env;
    ASSERT_TRUE(stream->Read(&resp_env));
    EXPECT_EQ(resp_env.request_id(), 99);
    EXPECT_EQ(resp_env.status_code(), 0);
    stream->WritesDone();
    grpc::Status finish_status = stream->Finish();
    EXPECT_TRUE(finish_status.ok());
  }

  server->Stop();
}

TEST(WeightSyncFourWayInteropTest, TcpFourWayClientServerMatrix) {
  LegacyTcpWeightSyncServer old_server;
  std::string old_endpoint = absl::StrCat("127.0.0.1:", old_server.port());

  ControlPipeConfig new_cfg;
  new_cfg.backend_type = ControlPipeBackendType::kTcp;
  new_cfg.allow_legacy_framing = true;
  std::unique_ptr<ControlPipeServer> new_server =
      CreateControlPipeServer(new_cfg);
  new_server->dispatcher()
      .RegisterHandler<::tpu_sync::rpc::ControlRequest,
                       ::tpu_sync::rpc::ControlResponse>(
          [](const ControlContext& ctx,
             const ::tpu_sync::rpc::ControlRequest& req)
              -> absl::StatusOr<::tpu_sync::rpc::ControlResponse> {
            ::tpu_sync::rpc::ControlResponse resp;
            resp.set_success(true);
            resp.set_message("new_tcp_server_ok");
            return resp;
          });
  TF_ASSERT_OK_AND_ASSIGN(int new_port, new_server->Start(0));
  std::string new_endpoint = absl::StrCat("127.0.0.1:", new_port);

  std::unique_ptr<ControlPipeClient> new_client =
      CreateControlPipeClient(new_cfg);

  ::tpu_sync::rpc::ControlRequest req;
  req.set_command(::tpu_sync::rpc::ControlRequest::COMMAND_START_TRANSFER);

  // 1. Old Client -> Old Server
  TF_ASSERT_OK_AND_ASSIGN(::tpu_sync::rpc::ControlResponse resp_old_old,
                          SendLegacyTcpControlRequest(old_server.port(), req));
  EXPECT_TRUE(resp_old_old.success());
  EXPECT_EQ(resp_old_old.message(), "old_tcp_server_ok");

  // 2. Old Client -> New Server
  TF_ASSERT_OK_AND_ASSIGN(::tpu_sync::rpc::ControlResponse resp_old_new,
                          SendLegacyTcpControlRequest(new_port, req));
  EXPECT_TRUE(resp_old_new.success());
  EXPECT_EQ(resp_old_new.message(), "new_tcp_server_ok");

  // 3. New Client -> Old Server
  TF_ASSERT_OK_AND_ASSIGN(
      ::tpu_sync::rpc::ControlResponse resp_new_old,
      (new_client->Call<::tpu_sync::rpc::ControlRequest,
                        ::tpu_sync::rpc::ControlResponse>(old_endpoint, req)));
  EXPECT_TRUE(resp_new_old.success());
  EXPECT_EQ(resp_new_old.message(), "old_tcp_server_ok");

  // 4. New Client -> New Server
  TF_ASSERT_OK_AND_ASSIGN(
      ::tpu_sync::rpc::ControlResponse resp_new_new,
      (new_client->Call<::tpu_sync::rpc::ControlRequest,
                        ::tpu_sync::rpc::ControlResponse>(new_endpoint, req)));
  EXPECT_TRUE(resp_new_new.success());
  EXPECT_EQ(resp_new_new.message(), "new_tcp_server_ok");

  new_server->Stop();
  old_server.Stop();
}

TEST(GrpcControlPipeClientLruTest,
     EvictsLeastRecentlyUsedStubWhenCapacityExceeded) {
  ControlPipeConfig server_cfg;
  server_cfg.backend_type = ControlPipeBackendType::kGrpc;

  std::unique_ptr<ControlPipeServer> server1 =
      CreateControlPipeServer(server_cfg);
  std::unique_ptr<ControlPipeServer> server2 =
      CreateControlPipeServer(server_cfg);
  std::unique_ptr<ControlPipeServer> server3 =
      CreateControlPipeServer(server_cfg);

  for (ControlPipeServer* srv : {server1.get(), server2.get(), server3.get()}) {
    srv->dispatcher().RegisterOneWayHandler<AckRequest>(
        [](const ControlContext& ctx, const AckRequest& req) {
          return absl::OkStatus();
        });
  }

  TF_ASSERT_OK_AND_ASSIGN(int port1, server1->Start(0));
  TF_ASSERT_OK_AND_ASSIGN(int port2, server2->Start(0));
  TF_ASSERT_OK_AND_ASSIGN(int port3, server3->Start(0));

  std::string ep1 = absl::StrCat("127.0.0.1:", port1);
  std::string ep2 = absl::StrCat("127.0.0.1:", port2);
  std::string ep3 = absl::StrCat("127.0.0.1:", port3);

  ControlPipeConfig client_cfg;
  client_cfg.backend_type = ControlPipeBackendType::kGrpc;
  client_cfg.max_cached_grpc_stubs = 2;
  GrpcControlPipeClient client(client_cfg);
  EXPECT_EQ(client.TEST_CachedStubCount(), 0u);

  AckRequest ack_req;
  ack_req.set_uuid(1);

  // 1. Send request to ep1, then ep2 -> both cached.
  ASSERT_TRUE(client.SendOneWay(ep1, ack_req).ok());
  EXPECT_EQ(client.TEST_CachedStubCount(), 1u);
  EXPECT_TRUE(client.TEST_HasCachedStub(ep1));

  ASSERT_TRUE(client.SendOneWay(ep2, ack_req).ok());
  EXPECT_EQ(client.TEST_CachedStubCount(), 2u);
  EXPECT_TRUE(client.TEST_HasCachedStub(ep1));
  EXPECT_TRUE(client.TEST_HasCachedStub(ep2));

  // 2. Access ep1 again (promotes ep1 to MRU, making ep2 LRU).
  ASSERT_TRUE(client.SendOneWay(ep1, ack_req).ok());
  EXPECT_EQ(client.TEST_CachedStubCount(), 2u);

  // 3. Send request to ep3 -> ep2 is evicted, ep1 and ep3 remain cached.
  ASSERT_TRUE(client.SendOneWay(ep3, ack_req).ok());
  EXPECT_EQ(client.TEST_CachedStubCount(), 2u);
  EXPECT_FALSE(client.TEST_HasCachedStub(ep2));
  EXPECT_TRUE(client.TEST_HasCachedStub(ep1));
  EXPECT_TRUE(client.TEST_HasCachedStub(ep3));

  server1->Stop();
  server2->Stop();
  server3->Stop();
}

TEST(GrpcControlPipeClientLruTest, ZeroCapacityBypassesCache) {
  ControlPipeConfig server_cfg;
  server_cfg.backend_type = ControlPipeBackendType::kGrpc;
  std::unique_ptr<ControlPipeServer> server =
      CreateControlPipeServer(server_cfg);
  server->dispatcher().RegisterOneWayHandler<AckRequest>(
      [](const ControlContext& ctx, const AckRequest& req) {
        return absl::OkStatus();
      });
  TF_ASSERT_OK_AND_ASSIGN(int port, server->Start(0));
  std::string ep = absl::StrCat("127.0.0.1:", port);

  ControlPipeConfig client_cfg;
  client_cfg.backend_type = ControlPipeBackendType::kGrpc;
  client_cfg.max_cached_grpc_stubs = 0;
  GrpcControlPipeClient client(client_cfg);

  AckRequest ack_req;
  ack_req.set_uuid(100);
  ASSERT_TRUE(client.SendOneWay(ep, ack_req).ok());
  EXPECT_EQ(client.TEST_CachedStubCount(), 0u);
  EXPECT_FALSE(client.TEST_HasCachedStub(ep));

  server->Stop();
}

}  // namespace
}  // namespace tpu_raiden
