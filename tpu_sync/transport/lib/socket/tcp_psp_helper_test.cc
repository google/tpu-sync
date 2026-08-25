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

#include "tpu_sync/transport/lib/socket/tcp_psp_helper.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "grpcpp/channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/channel_arguments.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/transport/lib/socket/psp_syscall_mock.h"  // NOLINT
#include "tpu_sync/transport/peregrine/src/internal/control/service.grpc.pb.h"
#include "tpu_sync/transport/peregrine/src/internal/control/service.pb.h"

namespace tpu_raiden::transport::lib {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Field;
using ::testing::Ne;
using ::testing::NotNull;

constexpr absl::string_view kValidKey("0123456789\0\0\0\0\0", 16);  // NOLINT

class FakePeregrineService final
    : public ::peregrine::internal::control::PeregrineService::Service {
 public:
  explicit FakePeregrineService(int server_fd) : server_fd_(server_fd) {}

  grpc::Status ExchangePspKey(
      grpc::ServerContext* context,
      const ::peregrine::internal::control::PspKeyExchangeRequest* request,
      ::peregrine::internal::control::PspKeyExchangeResponse* response)
      override {
    auto rx_key = RegisterPspPeerKey(server_fd_, request->client_spi(),
                                     request->client_key());
    if (!rx_key.ok()) {
      return grpc::Status(rx_key.status());
    }
    response->set_server_spi(rx_key->spi);
    response->set_server_key(rx_key->key);
    return grpc::Status::OK;
  }

 private:
  int server_fd_;
};

TEST(TcpPspHelperTest, RegisterPspKey) {
  if (!IsPspSupported()) {
    GTEST_SKIP() << "PSP-TCP is unimplemented.";
  }
  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(server_fd, 0);

  EXPECT_THAT(RegisterPspPeerKey(server_fd, 0x12345678, kValidKey),
              IsOkAndHolds(Field(&PspPeerKey::spi, Ne(0))));
  EXPECT_THAT(RegisterPspPeerKey(server_fd, 0, kValidKey),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(RegisterPspPeerKey(server_fd, 0x12345678, "short"),
              StatusIs(absl::StatusCode::kInvalidArgument));

  ::close(server_fd);
}

TEST(TcpPspHelperTest, PspEnabledVerification) {
  if (!IsPspSupported()) {
    GTEST_SKIP() << "PSP-TCP is unimplemented.";
  }
  int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(sock_fd, 0);

  EXPECT_TRUE(PspEnabled(sock_fd));
  EXPECT_FALSE(PspEnabled(-1));

  ::close(sock_fd);
}

TEST(TcpPspHelperTest, EndToEndPspKeyExchangeAndConnect) {
  if (!IsPspSupported()) {
    GTEST_SKIP() << "PSP-TCP is unimplemented.";
  }
  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(server_fd, 0);
  struct sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  ASSERT_EQ(bind(server_fd, reinterpret_cast<struct sockaddr*>(&addr),
                 sizeof(addr)),
            0);
  ASSERT_EQ(listen(server_fd, 1), 0);

  socklen_t addrlen = sizeof(addr);
  ASSERT_EQ(getsockname(server_fd, reinterpret_cast<struct sockaddr*>(&addr),
                        &addrlen),
            0);

  FakePeregrineService service(server_fd);
  grpc::ServerBuilder builder;
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  ASSERT_THAT(server, NotNull());

  std::shared_ptr<grpc::Channel> channel =
      server->InProcessChannel(grpc::ChannelArguments());

  int client_fd = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(client_fd, 0);
  ASSERT_OK(TcpPspConnect(client_fd,
                          reinterpret_cast<struct sockaddr*>(&addr),
                          addrlen, channel));

  int accepted_fd = accept(server_fd, nullptr, nullptr);
  ASSERT_GE(accepted_fd, 0);

  EXPECT_TRUE(PspEnabled(client_fd));
  EXPECT_TRUE(PspEnabled(accepted_fd));

  ::close(accepted_fd);
  ::close(client_fd);
  ::close(server_fd);
  server->Shutdown();
}

}  // namespace
}  // namespace tpu_raiden::transport::lib
