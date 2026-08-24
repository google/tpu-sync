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
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/channel_arguments.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/lib/raw_buffer_transport_delegate.h"
#include "tpu_sync/transport/peregrine/src/internal/control/service.grpc.pb.h"
#include "tpu_sync/transport/peregrine/src/internal/control/service.pb.h"

namespace tpu_raiden::transport::lib {
namespace {

using ::testing::NotNull;

class FakeRawDelegate : public RawBufferTransportDelegate {
 public:
  uint8_t* GetHostPointer(size_t buffer_id, size_t shard_idx) override {
    return nullptr;
  }
  size_t GetHostSize(size_t buffer_id, size_t shard_idx) override { return 0; }
};

TEST(PeregrineControlServiceTest, InProcessGrpcExchangePspKey) {
  FakeRawDelegate raw_delegate;
  RawBufferTransport transport(&raw_delegate, /*local_port=*/0);
  PeregrineControlServiceImpl service(&transport);

  grpc::ServerBuilder builder;
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  ASSERT_THAT(server, NotNull());

  std::shared_ptr<grpc::Channel> channel =
      server->InProcessChannel(grpc::ChannelArguments());
  auto stub =
      ::peregrine::internal::control::PeregrineService::NewStub(channel);

  ::peregrine::internal::control::PspKeyExchangeRequest req;
  req.set_client_spi(0x12345678);
  req.set_client_key(std::string(16, 'z'));
  ::peregrine::internal::control::PspKeyExchangeResponse resp;
  grpc::ClientContext ctx;

  grpc::Status status = stub->ExchangePspKey(&ctx, req, &resp);
  // Status is ok or unavailable depending on hardware PSP kernel support.
  if (status.ok()) {
    EXPECT_NE(resp.server_spi(), 0);
    EXPECT_EQ(resp.server_key().size(), 16);
  }

  server->Shutdown();
}

TEST(PeregrineControlServiceTest, RejectsInvalidClientKey) {
  FakeRawDelegate raw_delegate;
  RawBufferTransport transport(&raw_delegate, /*local_port=*/0);
  PeregrineControlServiceImpl service(&transport);

  grpc::ServerBuilder builder;
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  ASSERT_THAT(server, NotNull());

  std::shared_ptr<grpc::Channel> channel =
      server->InProcessChannel(grpc::ChannelArguments());
  auto stub =
      ::peregrine::internal::control::PeregrineService::NewStub(channel);

  ::peregrine::internal::control::PspKeyExchangeRequest req;
  req.set_client_spi(0x12345678);
  req.set_client_key("short_key");  // Not 16 bytes
  ::peregrine::internal::control::PspKeyExchangeResponse resp;
  grpc::ClientContext ctx;

  grpc::Status status = stub->ExchangePspKey(&ctx, req, &resp);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);

  server->Shutdown();
}

TEST(PeregrineControlServiceTest, RejectsZeroClientSpi) {
  FakeRawDelegate raw_delegate;
  RawBufferTransport transport(&raw_delegate, /*local_port=*/0);
  PeregrineControlServiceImpl service(&transport);

  grpc::ServerBuilder builder;
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  ASSERT_THAT(server, NotNull());

  std::shared_ptr<grpc::Channel> channel =
      server->InProcessChannel(grpc::ChannelArguments());
  auto stub =
      ::peregrine::internal::control::PeregrineService::NewStub(channel);

  ::peregrine::internal::control::PspKeyExchangeRequest req;
  req.set_client_spi(0);  // Invalid SPI
  req.set_client_key(std::string(16, 'x'));
  ::peregrine::internal::control::PspKeyExchangeResponse resp;
  grpc::ClientContext ctx;

  grpc::Status status = stub->ExchangePspKey(&ctx, req, &resp);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);

  server->Shutdown();
}

}  // namespace
}  // namespace tpu_raiden::transport::lib
