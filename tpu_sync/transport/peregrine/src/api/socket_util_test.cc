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

#include "tpu_sync/transport/peregrine/src/api/socket_util.h"

#include <sys/socket.h>
#include <sys/types.h>

#include <cstring>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/peregrine/src/api/transport_types.h"
#include "tpu_sync/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_sync/transport/peregrine/src/internal/base/ipaddr.h"
#include "tpu_sync/transport/peregrine/src/internal/base/types.h"
#include "tpu_sync/transport/peregrine/src/internal/socket/socket_tcp.h"
#include "tpu_sync/transport/peregrine/src/internal/socket/socket_util.h"

namespace peregrine::testing {
namespace {

using ::peregrine::internal::Endpoint;
using ::peregrine::internal::TcpSocket;
using ::testing::Combine;
using ::testing::Eq;
using ::testing::Ne;
using ::testing::Pointwise;
using ::testing::TestParamInfo;
using ::testing::Values;

using Param = std::tuple</*family=*/int, /*riov=*/bool, /*wiov=*/bool>;

std::string ToString(const TestParamInfo<Param>& info) {
  const int family = std::get<0>(info.param);
  const bool read_iovec = std::get<1>(info.param);
  const bool write_iovec = std::get<2>(info.param);
  DCHECK(family == AF_INET || family == AF_INET6);
  return absl::StrFormat("IPv%d_Read%s_Write%s", family == AF_INET ? 4 : 6,
                         read_iovec ? "V" : "", write_iovec ? "V" : "");
}

class SocketUtilTest : public ::testing::TestWithParam<Param> {
 protected:
  SocketUtilTest()
      : family_(std::get<0>(GetParam())),
        read_iovec_(std::get<1>(GetParam())),
        write_iovec_(std::get<2>(GetParam())),
        local_(internal::IpAddr::Create(family_ == AF_INET ? "127.0.0.1" : "::1")
                   .value(),
               0),
        listener_(TcpSocket::Create(family_)),
        connector_(TcpSocket::Create(family_)) {
    CHECK_NE(listener_, nullptr);
    CHECK_NE(connector_, nullptr);
    DCHECK(listener_->IsValid());
    DCHECK(connector_->IsValid());
    DCHECK(!listener_->IsConnected());
    DCHECK(!connector_->IsConnected());
    DCHECK_NE(listener_->fd(), connector_->fd());
  }

 protected:
  void CheckReadWrite(size_t read_iov_count, size_t write_iov_count);

  const int family_;
  const bool read_iovec_;
  const bool write_iovec_;
  const Endpoint local_;
  const std::unique_ptr<TcpSocket> listener_;
  const std::unique_ptr<TcpSocket> connector_;
};

INSTANTIATE_TEST_SUITE_P(, SocketUtilTest,
                         Combine(/*family=*/Values(AF_INET, AF_INET6),
                                 /*riov=*/Values(false, true),
                                 /*wiov=*/Values(false, true)),
                         ToString);

void SocketUtilTest::CheckReadWrite(size_t read_iov_count,
                                  size_t write_iov_count) {
  // Non-uniform, non-zero data detects reordering without test RNG dependencies.
  constexpr size_t kSize = 64UL << 20;
  std::vector<Byte> send_buf(kSize, 0x01);
  std::vector<Byte> recv_buf(kSize, 0x00);
  for (size_t i = 0; i < send_buf.size(); ++i) {
    send_buf[i] = static_cast<Byte>(1 + i % 251);
  }
  ASSERT_THAT(recv_buf, Pointwise(Ne(), send_buf));

  // First, create a server thread.
  absl::Notification server_ready;
  std::thread server([&]() {
    CHECK(listener_->Listen(local_));
    server_ready.Notify();
    DCHECK(listener_->IsBlocking());
    const internal::fd_t new_fd = listener_->Accept();

    CHECK_GE(new_fd.value(), 0);
    auto new_socket = TcpSocket::Create(new_fd, family_);
    DCHECK(new_socket->IsBlocking());
    DCHECK(new_socket->IsConnected());

    if (read_iovec_) {
      std::vector<struct iovec> iovs;
      const size_t partial = kSize / read_iov_count;
      for (size_t i = 0; i < read_iov_count; ++i) {
        const size_t offset = i * partial;
        const size_t size = i + 1 == read_iov_count ? kSize - offset : partial;
        iovs.push_back({recv_buf.data() + offset, size});
      }
      CHECK_OK(ReadVExact(new_socket->fd().value(), iovs));
    } else {
      CHECK_OK(ReadExact(new_socket->fd().value(), recv_buf.data(), kSize));
    }
  });

  // Second, create a client thread.
  std::thread client([&]() {
    server_ready.WaitForNotification();
    // Port zero reserves an ephemeral port, without a free-port race or
    // dependencies on the optional test utility library.
    const Endpoint peer = Endpoint::Create(internal::SelfAddrPort(listener_->fd()));
    CHECK(connector_->Connect(peer));
    DCHECK(connector_->IsBlocking());
    DCHECK(connector_->IsConnected());

    if (write_iovec_) {
      std::vector<struct iovec> iovs;
      const size_t partial = kSize / write_iov_count;
      for (size_t i = 0; i < write_iov_count; ++i) {
        const size_t offset = i * partial;
        const size_t size = i + 1 == write_iov_count ? kSize - offset : partial;
        iovs.push_back({send_buf.data() + offset, size});
      }
      CHECK_OK(WriteVExact(connector_->fd().value(), iovs));
    } else {
      CHECK_OK(WriteExact(connector_->fd().value(), send_buf.data(), kSize));
    }
  });

  // Wait for both threads to finish.
  client.join();
  server.join();

  // Check that the recv buffer has the same data as the send.
  ASSERT_THAT(recv_buf, Pointwise(Eq(), send_buf));
}

TEST_P(SocketUtilTest, ReadWrite) { CheckReadWrite(3, 2); }

TEST_P(SocketUtilTest, ReadWriteAtIovMax) {
  CheckReadWrite(IOV_MAX, IOV_MAX);
}

TEST_P(SocketUtilTest, ReadWriteAboveIovMax) {
  CheckReadWrite(IOV_MAX + 1, 2 * IOV_MAX + 3);
}

TEST_P(SocketUtilTest, ReadWriteManyIovs) {
  // Different sender/receiver batch boundaries must preserve the stream.
  // This exceeds the 14,337 entries observed in hybrid-state resharding.
  CheckReadWrite(16 * IOV_MAX + 3, 14 * IOV_MAX + 1);
}

}  // namespace
}  // namespace peregrine::testing
