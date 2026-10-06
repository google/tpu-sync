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

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <cstdlib>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

namespace tpu_raiden::transport::lib {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Ge;
using ::testing::HasSubstr;

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, const char* value) : name_(name) {
    const char* old = std::getenv(name);
    if (old != nullptr) {
      had_old_ = true;
      old_value_ = old;
    }
    if (value != nullptr) {
      setenv(name_, value, 1);
    } else {
      unsetenv(name_);
    }
  }
  ~ScopedEnvVar() {
    if (had_old_) {
      setenv(name_, old_value_.c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char* name_;
  bool had_old_ = false;
  std::string old_value_;
};

// A loopback listener that never accepts by default. The kernel still completes
// the handshake of the connections that fit in its accept queue.
class LoopbackListener {
 public:
  explicit LoopbackListener(int backlog) : backlog_(backlog) {
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t len = sizeof(addr);
    auto* sa = reinterpret_cast<sockaddr*>(&addr);
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    CHECK_GE(listen_fd_, 0);
    CHECK_EQ(bind(listen_fd_, sa, len), 0);
    CHECK_EQ(listen(listen_fd_, backlog_), 0);
    CHECK_EQ(getsockname(listen_fd_, sa, &len), 0);
    addr_ = addr;
  }
  LoopbackListener(const LoopbackListener&) = delete;
  LoopbackListener& operator=(const LoopbackListener&) = delete;
  ~LoopbackListener() {
    for (int fd : filler_fds_) close(fd);
    for (int fd : accepted_fds_) close(fd);
    if (listen_fd_ >= 0) close(listen_fd_);
  }

  // Fills the accept queue, so the kernel drops every further SYN and a
  // connect() can only retry.
  void FillAcceptQueue() {
    for (int i = 0; i <= backlog_ + 1; ++i) {
      const int fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
      CHECK_GE(fd, 0);
      connect(fd, reinterpret_cast<const sockaddr*>(&addr_), sizeof(addr_));
      filler_fds_.push_back(fd);
    }
    absl::SleepFor(absl::Milliseconds(100));
  }

  // Drains all pending connections from the accept queue so subsequent
  // handshakes can complete.
  void DrainAcceptQueue() {
    for (int fd : filler_fds_) close(fd);
    filler_fds_.clear();
    const int flags = fcntl(listen_fd_, F_GETFL, 0);
    CHECK_GE(flags, 0);
    CHECK_EQ(fcntl(listen_fd_, F_SETFL, flags | O_NONBLOCK), 0);
    while (true) {
      const int fd = accept(listen_fd_, nullptr, nullptr);
      if (fd < 0) break;
      accepted_fds_.push_back(fd);
    }
  }

  void CloseListener() {
    if (listen_fd_ >= 0) {
      close(listen_fd_);
      listen_fd_ = -1;
    }
  }

  std::string address() const {
    return absl::StrCat("127.0.0.1:", ntohs(addr_.sin_port));
  }

 private:
  const int backlog_;
  int listen_fd_ = -1;
  sockaddr_in addr_ = {};
  std::vector<int> filler_fds_;
  std::vector<int> accepted_fds_;
};

TEST(ConnectToPeerTest, GivesUpWhenPeerNeverCompletesHandshake) {
  LoopbackListener listener(/*backlog=*/0);
  listener.FillAcceptQueue();

  const absl::Time start = absl::Now();
  EXPECT_THAT(ConnectToPeer(listener.address()),
              StatusIs(absl::StatusCode::kUnavailable, HasSubstr("timed out")));
  // Well under the kernel's ~127 s SYN retry budget.
  EXPECT_LT(absl::Now() - start, absl::Seconds(30));
}

TEST(ConnectToPeerTest, RetriesAndSucceedsWhenAcceptQueueDrains) {
  ScopedEnvVar timeout_env("TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS", "500");
  ScopedEnvVar attempts_env("TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS", "4");
  ScopedEnvVar backoff_env("TPU_RAIDEN_TCP_CONNECT_INITIAL_BACKOFF_MS", "100");

  LoopbackListener listener(/*backlog=*/0);
  listener.FillAcceptQueue();

  // Drain the accept queue after the first 500ms connect attempt times out.
  std::thread drainer([&listener]() {
    absl::SleepFor(absl::Milliseconds(700));
    listener.DrainAcceptQueue();
  });

  const absl::Time start = absl::Now();
  absl::StatusOr<int> fd = ConnectToPeer(listener.address());
  drainer.join();

  ASSERT_THAT(fd, IsOkAndHolds(Ge(0)));
  // Should have taken at least one timeout (500ms) before succeeding on retry.
  EXPECT_GE(absl::Now() - start, absl::Milliseconds(500));
  close(*fd);
}

TEST(ConnectToPeerTest, HandlesSignalInterruptionDuringConnectTimeout) {
  ScopedEnvVar timeout_env("TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS", "400");
  ScopedEnvVar attempts_env("TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS", "2");
  ScopedEnvVar backoff_env("TPU_RAIDEN_TCP_CONNECT_INITIAL_BACKOFF_MS", "50");

  struct sigaction sa = {};
  sa.sa_handler = +[](int) {};
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;  // Ensure connect() is interrupted with EINTR.
  struct sigaction old_sa = {};
  ASSERT_EQ(sigaction(SIGUSR1, &sa, &old_sa), 0);
  absl::Cleanup restore_sig = [&old_sa] {
    sigaction(SIGUSR1, &old_sa, nullptr);
  };

  LoopbackListener listener(/*backlog=*/0);
  listener.FillAcceptQueue();

  const pthread_t main_tid = pthread_self();
  std::thread signaller([main_tid]() {
    absl::SleepFor(absl::Milliseconds(100));
    pthread_kill(main_tid, SIGUSR1);
    absl::SleepFor(absl::Milliseconds(100));
    pthread_kill(main_tid, SIGUSR1);
  });

  EXPECT_THAT(ConnectToPeer(listener.address()),
              StatusIs(absl::StatusCode::kUnavailable, HasSubstr("timed out")));
  signaller.join();
}

TEST(ConnectToPeerTest, DoesNotRetryOnConnectionRefused) {
  LoopbackListener listener(/*backlog=*/0);
  const std::string addr = listener.address();
  listener.CloseListener();

  const absl::Time start = absl::Now();
  EXPECT_THAT(ConnectToPeer(addr), StatusIs(absl::StatusCode::kUnavailable));
  EXPECT_LT(absl::Now() - start, absl::Seconds(1));
}

TEST(ReadWithTimeoutTest, ReadExactTimesOutWhenPeerStalls) {
  int sv[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
  const char partial[] = "ab";
  ASSERT_EQ(write(sv[0], partial, 2), 2);

  char buf[4] = {};
  const absl::Time start = absl::Now();
  EXPECT_THAT(ReadExactWithTimeout(sv[1], buf, sizeof(buf),
                                   absl::Milliseconds(100)),
              StatusIs(absl::StatusCode::kDeadlineExceeded,
                       HasSubstr("timed out")));
  EXPECT_GE(absl::Now() - start, absl::Milliseconds(80));

  close(sv[0]);
  close(sv[1]);
}

TEST(ReadWithTimeoutTest, ReadVExactTimesOutAndReadsAcrossIovecs) {
  int sv[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
  const char full[] = "abcdef";
  ASSERT_EQ(write(sv[0], full, 6), 6);

  char buf1[2] = {};
  char buf2[4] = {};
  struct iovec iovs[2] = {
      {.iov_base = buf1, .iov_len = sizeof(buf1)},
      {.iov_base = buf2, .iov_len = sizeof(buf2)},
  };
  ABSL_EXPECT_OK(ReadVExactWithTimeout(sv[1], absl::MakeConstSpan(iovs),
                                       absl::Seconds(1)));
  EXPECT_EQ(std::string(buf1, 2), "ab");
  EXPECT_EQ(std::string(buf2, 4), "cdef");

  // Write partial data for next scatter read and let it time out.
  ASSERT_EQ(write(sv[0], full, 3), 3);
  EXPECT_THAT(ReadVExactWithTimeout(sv[1], absl::MakeConstSpan(iovs),
                                    absl::Milliseconds(100)),
              StatusIs(absl::StatusCode::kDeadlineExceeded,
                       HasSubstr("timed out")));

  close(sv[0]);
  close(sv[1]);
}

TEST(ReadWithTimeoutTest, NulloptTimeoutUsesPeregrineRead) {
  int sv[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
  const char payload[] = "abcdefgh";
  ASSERT_EQ(write(sv[0], payload, 8), 8);

  char buf[4] = {};
  ABSL_EXPECT_OK(ReadExactWithTimeout(sv[1], buf, sizeof(buf), std::nullopt));
  EXPECT_EQ(std::string(buf, 4), "abcd");

  char vbuf1[2] = {};
  char vbuf2[2] = {};
  struct iovec iovs[2] = {
      {.iov_base = vbuf1, .iov_len = sizeof(vbuf1)},
      {.iov_base = vbuf2, .iov_len = sizeof(vbuf2)},
  };
  ABSL_EXPECT_OK(
      ReadVExactWithTimeout(sv[1], absl::MakeConstSpan(iovs), std::nullopt));
  EXPECT_EQ(std::string(vbuf1, 2), "ef");
  EXPECT_EQ(std::string(vbuf2, 2), "gh");

  close(sv[0]);
  close(sv[1]);
}

TEST(WriteWithTimeoutTest, WriteExactTimesOutWhenPeerStalls) {
  int sv[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
  int sndbuf = 4096;
  ASSERT_EQ(setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf)),
            0);

  std::vector<char> buf(1024 * 1024, 'x');
  const absl::Time start = absl::Now();
  EXPECT_THAT(
      WriteExactWithTimeout(sv[0], buf.data(), buf.size(),
                            absl::Milliseconds(100)),
      StatusIs(absl::StatusCode::kDeadlineExceeded, HasSubstr("timed out")));
  EXPECT_GE(absl::Now() - start, absl::Milliseconds(80));

  close(sv[0]);
  close(sv[1]);
}

TEST(WriteWithTimeoutTest, WriteVExactTimesOutAndWritesAcrossIovecs) {
  int sv[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);

  char part1[] = "ab";
  char part2[] = "cdef";
  struct iovec iovs[2] = {
      {.iov_base = part1, .iov_len = 2},
      {.iov_base = part2, .iov_len = 4},
  };
  ABSL_EXPECT_OK(WriteVExactWithTimeout(sv[0], absl::MakeConstSpan(iovs),
                                        absl::Seconds(1)));

  char recv_buf[6] = {};
  ASSERT_EQ(read(sv[1], recv_buf, sizeof(recv_buf)), 6);
  EXPECT_EQ(std::string(recv_buf, 6), "abcdef");

  int sndbuf = 4096;
  ASSERT_EQ(setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf)),
            0);
  std::vector<char> large1(512 * 1024, 'a');
  std::vector<char> large2(512 * 1024, 'b');
  struct iovec large_iovs[2] = {
      {.iov_base = large1.data(), .iov_len = large1.size()},
      {.iov_base = large2.data(), .iov_len = large2.size()},
  };
  EXPECT_THAT(
      WriteVExactWithTimeout(sv[0], absl::MakeConstSpan(large_iovs),
                             absl::Milliseconds(100)),
      StatusIs(absl::StatusCode::kDeadlineExceeded, HasSubstr("timed out")));

  close(sv[0]);
  close(sv[1]);
}

TEST(WriteWithTimeoutTest, NulloptTimeoutUsesPeregrineWrite) {
  int sv[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);

  const char payload[] = "abcd";
  ABSL_EXPECT_OK(WriteExactWithTimeout(sv[0], payload, 4, std::nullopt));

  char vpart1[] = "ef";
  char vpart2[] = "gh";
  struct iovec iovs[2] = {
      {.iov_base = vpart1, .iov_len = 2},
      {.iov_base = vpart2, .iov_len = 2},
  };
  ABSL_EXPECT_OK(
      WriteVExactWithTimeout(sv[0], absl::MakeConstSpan(iovs), std::nullopt));

  char recv_buf[8] = {};
  ASSERT_EQ(read(sv[1], recv_buf, sizeof(recv_buf)), 8);
  EXPECT_EQ(std::string(recv_buf, 8), "abcdefgh");

  close(sv[0]);
  close(sv[1]);
}
}  // namespace
}  // namespace tpu_raiden::transport::lib
