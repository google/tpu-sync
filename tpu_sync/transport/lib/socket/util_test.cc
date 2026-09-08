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

#include "tpu_sync/transport/lib/socket/util.h"

#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/flags/reflection.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"

ABSL_DECLARE_FLAG(absl::Duration, raiden_transport_io_timeout);
ABSL_DECLARE_FLAG(absl::Duration, raiden_transport_connect_timeout);

namespace tpu_raiden::transport::lib {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

// A loopback TCP listener on an ephemeral port. Accepts nothing; the tests
// here only need a port that completes a handshake.
class ScopedListener {
 public:
  ScopedListener() {
    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    CHECK_GE(fd_, 0);
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    CHECK_EQ(bind(fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)),
             0);
    CHECK_EQ(listen(fd_, /*backlog=*/8), 0);

    socklen_t len = sizeof(addr);
    CHECK_EQ(
        getsockname(fd_, reinterpret_cast<struct sockaddr*>(&addr), &len), 0);
    port_ = ntohs(addr.sin_port);
  }

  ~ScopedListener() {
    if (fd_ >= 0) close(fd_);
  }

  ScopedListener(const ScopedListener&) = delete;
  ScopedListener& operator=(const ScopedListener&) = delete;

  // Closes the listening socket so its port stops accepting.
  void Close() {
    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
    }
  }

  std::string address() const { return absl::StrCat("127.0.0.1:", port_); }

 private:
  int fd_ = -1;
  int port_ = 0;
};

// Returns the SO_SNDTIMEO currently set on `fd`.
absl::Duration SendTimeoutOf(int fd) {
  struct timeval tv = {};
  socklen_t len = sizeof(tv);
  CHECK_EQ(getsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, &len), 0);
  return absl::DurationFromTimeval(tv);
}

// Returns the SO_RCVTIMEO currently set on `fd`.
absl::Duration RecvTimeoutOf(int fd) {
  struct timeval tv = {};
  socklen_t len = sizeof(tv);
  CHECK_EQ(getsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, &len), 0);
  return absl::DurationFromTimeval(tv);
}

// Returns true iff SO_KEEPALIVE is enabled on `fd`.
bool KeepaliveEnabledOn(int fd) {
  int value = 0;
  socklen_t len = sizeof(value);
  CHECK_EQ(getsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &value, &len), 0);
  return value != 0;
}

// Returns true iff `fd` is in blocking mode.
bool IsBlocking(int fd) {
  const int flags = fcntl(fd, F_GETFL, 0);
  CHECK_GE(flags, 0);
  return (flags & O_NONBLOCK) == 0;
}

class SocketUtilTest : public ::testing::Test {
 protected:
  // Restores every flag this fixture overrides when the test ends.
  absl::FlagSaver flag_saver_;
};

TEST_F(SocketUtilTest, ApplyDataPlaneSocketOptionsBoundsBlockingIo) {
  absl::SetFlag(&FLAGS_raiden_transport_io_timeout, absl::Seconds(7));

  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(SendTimeoutOf(fd), absl::ZeroDuration());

  ApplyDataPlaneSocketOptions(fd);

  EXPECT_EQ(SendTimeoutOf(fd), absl::Seconds(7));
  EXPECT_EQ(RecvTimeoutOf(fd), absl::Seconds(7));
  EXPECT_TRUE(KeepaliveEnabledOn(fd));
  close(fd);
}

TEST_F(SocketUtilTest, ApplyDataPlaneSocketOptionsIsNoOpWhenTimeoutIsZero) {
  // Zero is the documented rollback: it restores the unbounded blocking I/O
  // that predates these options.
  absl::SetFlag(&FLAGS_raiden_transport_io_timeout, absl::ZeroDuration());

  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(fd, 0);

  ApplyDataPlaneSocketOptions(fd);

  EXPECT_EQ(SendTimeoutOf(fd), absl::ZeroDuration());
  EXPECT_EQ(RecvTimeoutOf(fd), absl::ZeroDuration());
  EXPECT_FALSE(KeepaliveEnabledOn(fd));
  close(fd);
}

TEST_F(SocketUtilTest, ConnectToPeerReturnsBoundedBlockingSocket) {
  absl::SetFlag(&FLAGS_raiden_transport_io_timeout, absl::Seconds(11));
  absl::SetFlag(&FLAGS_raiden_transport_connect_timeout, absl::Seconds(5));

  ScopedListener listener;
  absl::StatusOr<int> fd = ConnectToPeer(listener.address());
  ASSERT_THAT(fd, IsOk());

  // The bounded connect switches the socket to non-blocking to poll on it.
  // Send and recv assert blocking mode, so it has to be restored.
  EXPECT_TRUE(IsBlocking(*fd));
  EXPECT_EQ(SendTimeoutOf(*fd), absl::Seconds(11));
  EXPECT_EQ(RecvTimeoutOf(*fd), absl::Seconds(11));
  EXPECT_TRUE(KeepaliveEnabledOn(*fd));
  close(*fd);
}

TEST_F(SocketUtilTest, ConnectToPeerLeavesSocketBlockingWhenDeadlineDisabled) {
  absl::SetFlag(&FLAGS_raiden_transport_connect_timeout, absl::ZeroDuration());

  ScopedListener listener;
  absl::StatusOr<int> fd = ConnectToPeer(listener.address());
  ASSERT_THAT(fd, IsOk());

  EXPECT_TRUE(IsBlocking(*fd));
  close(*fd);
}

TEST_F(SocketUtilTest, ConnectToPeerRefusedStaysUnavailable) {
  absl::SetFlag(&FLAGS_raiden_transport_connect_timeout, absl::Seconds(5));

  // Bind a port and then release it, so the address is well formed and
  // routable but nothing is listening. A refused connection has to stay
  // UNAVAILABLE: the transfer failure metric labels on that code, and
  // callers distinguish it from a connect that ran out of time.
  ScopedListener listener;
  const std::string address = listener.address();
  listener.Close();

  EXPECT_THAT(ConnectToPeer(address),
              StatusIs(absl::StatusCode::kUnavailable));
}

TEST_F(SocketUtilTest, ConnectToPeerRejectsMalformedAddress) {
  EXPECT_THAT(ConnectToPeer("not-a-host-port"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace tpu_raiden::transport::lib
