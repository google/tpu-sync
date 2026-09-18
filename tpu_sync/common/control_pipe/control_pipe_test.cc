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

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

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
#include "tpu_sync/proto/control_pipe.grpc.pb.h"
#include "tpu_sync/proto/control_pipe.pb.h"
#include "tpu_sync/proto/kv_cache_control_plane_service.grpc.pb.h"
#include "tpu_sync/proto/kv_cache_control_plane_service.pb.h"
#include "tpu_sync/rpc/raiden_service.grpc.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace {

using ::tpu_raiden::control_plane::proto::AckRequest;
using ::tpu_raiden::control_plane::proto::AckResponse;
using ::tpu_raiden::control_plane::proto::KVCacheControlPlaneService;
using ::tpu_raiden::control_plane::proto::PullStreamRequest;
using ::tpu_raiden::control_plane::proto::PullStreamResponse;

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
                      ControlPipeBackendType::kGrpc),
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
  server->dispatcher()
      .RegisterHandler<::tpu_sync::rpc::ControlRequest,
                       ::tpu_sync::rpc::ControlResponse>(
          [](const ControlContext& ctx,
             const ::tpu_sync::rpc::ControlRequest& req)
              -> absl::StatusOr<::tpu_sync::rpc::ControlResponse> {
            ::tpu_sync::rpc::ControlResponse resp;
            resp.set_success(true);
            resp.set_message("weight_sync_ok");
            return resp;
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

  // 2. Call via legacy WeightSynchronizationWorkerService::Stub
  auto ws_stub =
      ::tpu_sync::rpc::WeightSynchronizationWorkerService::NewStub(channel);
  {
    grpc::ClientContext ctx;
    ::tpu_sync::rpc::ControlRequest req;
    req.set_command(::tpu_sync::rpc::ControlRequest::COMMAND_SHUTDOWN);
    ::tpu_sync::rpc::ControlResponse resp;
    grpc::Status status = ws_stub->HandleControl(&ctx, req, &resp);
    ASSERT_TRUE(status.ok()) << status.error_message();
    EXPECT_TRUE(resp.success());
    EXPECT_EQ(resp.message(), "weight_sync_ok");
  }

  // 3. Call bidirectional ControlStream on ControlPipeService::Stub
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

}  // namespace
}  // namespace tpu_raiden
