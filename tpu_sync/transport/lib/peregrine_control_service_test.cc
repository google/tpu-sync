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

#include "tpu_sync/transport/lib/peregrine_control_service.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/channel_arguments.h"
#include "grpcpp/support/status.h"
#include "peregrine/src/internal/control/message.pb.h"
#include "peregrine/src/internal/control/message_internal.pb.h"
#include "peregrine/src/internal/control/service.grpc.pb.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/lib/raw_buffer_transport_delegate.h"
#include "tpu_sync/transport/lib/socket/psp_syscall_mock.h"  // NOLINT
#include "tpu_sync/transport/lib/socket/tcp_psp_helper.h"

namespace tpu_raiden::transport::lib {
namespace {

using ::peregrine::internal::control::PeregrineService;
using ::peregrine::internal::proto::ReqMsg;
using ::peregrine::internal::proto::RespMsg;
using ::testing::NotNull;

class FakeRawDelegate : public RawBufferTransportDelegate {
 public:
  uint8_t* GetHostPointer(size_t buffer_id, size_t shard_idx) override {
    return nullptr;
  }
  size_t GetHostSize(size_t buffer_id, size_t shard_idx) override { return 0; }
};

ReqMsg MakeValidReq() {
  ReqMsg req;
  auto* psp_req = req.mutable_psp_tcp_req();
  psp_req->mutable_psp()->set_spi(0x12345678);
  psp_req->mutable_psp()->set_gen(1);
  psp_req->mutable_psp()->set_key(std::string(16, 'z'));
  psp_req->mutable_peer_target()->set_ip_port("127.0.0.1:12345");
  return req;
}

TEST(PeregrineControlServiceTest, ProcessUnary) {
  absl::SetFlag(&FLAGS_require_psp_tcp, true);
  FakeRawDelegate raw_delegate;
  RawBufferTransport transport(&raw_delegate, /*local_port=*/0);
  PeregrineControlServiceImpl service(&transport);

  grpc::ServerBuilder builder;
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  ASSERT_THAT(server, NotNull());

  auto stub = PeregrineService::NewStub(
      server->InProcessChannel(grpc::ChannelArguments()));
  auto call = [&](const ReqMsg& req, RespMsg* resp) {
    grpc::ClientContext ctx;
    return stub->ProcessUnary(&ctx, req, resp);
  };

  RespMsg resp;
  grpc::Status status = call(MakeValidReq(), &resp);
  if (IsPspSupported()) {
    ASSERT_TRUE(status.ok()) << status.error_message();
    ASSERT_TRUE(resp.has_psp_tcp_resp());
    ASSERT_TRUE(resp.psp_tcp_resp().has_psp());
    const auto& resp_psp = resp.psp_tcp_resp().psp();
    EXPECT_NE(resp_psp.spi(), 0);
    EXPECT_TRUE(resp_psp.has_gen());
    EXPECT_EQ(resp_psp.key().size(), 16);
  } else {
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNIMPLEMENTED);
  }

  EXPECT_EQ(call(ReqMsg{}, &resp).error_code(),
            grpc::StatusCode::UNIMPLEMENTED);

  {
    ReqMsg req = MakeValidReq();
    req.mutable_psp_tcp_req()->mutable_psp()->set_key("short_key");
    EXPECT_EQ(call(req, &resp).error_code(),
              grpc::StatusCode::INVALID_ARGUMENT);
  }
  {
    ReqMsg req = MakeValidReq();
    req.mutable_psp_tcp_req()->mutable_psp()->set_spi(0);
    EXPECT_EQ(call(req, &resp).error_code(),
              grpc::StatusCode::INVALID_ARGUMENT);
  }
  {
    ReqMsg req = MakeValidReq();
    req.mutable_psp_tcp_req()->mutable_psp()->clear_gen();
    EXPECT_EQ(call(req, &resp).error_code(),
              grpc::StatusCode::INVALID_ARGUMENT);
  }
  {
    ReqMsg req = MakeValidReq();
    req.mutable_psp_tcp_req()->clear_peer_target();
    EXPECT_EQ(call(req, &resp).error_code(),
              grpc::StatusCode::INVALID_ARGUMENT);
  }
  if (IsPspSupported()) {
    ReqMsg req = MakeValidReq();
    req.mutable_psp_tcp_req()->mutable_peer_target()->set_ip_port("invalid");
    EXPECT_EQ(call(req, &resp).error_code(),
              grpc::StatusCode::INVALID_ARGUMENT);
  }

  server->Shutdown();
}

}  // namespace
}  // namespace tpu_raiden::transport::lib
