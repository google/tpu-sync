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
#include <ifaddrs.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sys/socket.h>
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

// First non-loopback IPv4 address on this host, or nullopt if it has none.
std::optional<std::string> FirstNonLoopbackIpv4Address() {
  ifaddrs* ifa_list = nullptr;
  CHECK_EQ(getifaddrs(&ifa_list), 0);
  absl::Cleanup free_list = [ifa_list] { freeifaddrs(ifa_list); };
  for (const ifaddrs* ifa = ifa_list; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr || ifa->ifa_addr->sa_family != AF_INET) {
      continue;
    }
    const auto* sin = reinterpret_cast<const sockaddr_in*>(ifa->ifa_addr);
    if ((ntohl(sin->sin_addr.s_addr) >> 24) == 127) continue;
    char ip[INET_ADDRSTRLEN] = {};
    CHECK_NE(inet_ntop(AF_INET, &sin->sin_addr, ip, sizeof(ip)), nullptr);
    return std::string(ip);
  }
  return std::nullopt;
}

TEST(SourceIpRoutesToPeerTest, LoopbackSourceToLoopbackPeer) {
  ClearSourceIpRouteCacheForTesting();
  EXPECT_THAT(SourceIpRoutesToPeer("127.0.0.3", "127.0.0.1:1"),
              IsOkAndHolds(true));
}

TEST(SourceIpRoutesToPeerTest, UnownedSourceIsNotFound) {
  ClearSourceIpRouteCacheForTesting();
  // TEST-NET-1 is never assigned to a local interface.
  EXPECT_THAT(SourceIpRoutesToPeer("192.0.2.1", "127.0.0.1:1"),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("192.0.2.1")));
}

TEST(SourceIpRoutesToPeerTest, NonLoopbackSourceDoesNotRouteToLoopbackPeer) {
  ClearSourceIpRouteCacheForTesting();
  const std::optional<std::string> nic_ip = FirstNonLoopbackIpv4Address();
  if (!nic_ip.has_value()) {
    GTEST_SKIP() << "host has no non-loopback IPv4 address";
  }
  // The kernel egresses loopback peers via `lo`, not the NIC owning `nic_ip`.
  EXPECT_THAT(SourceIpRoutesToPeer(*nic_ip, "127.0.0.1:1"),
              IsOkAndHolds(false));
}

TEST(SourceIpRoutesToPeerTest, InvalidPeerIsInvalidArgument) {
  ClearSourceIpRouteCacheForTesting();
  EXPECT_THAT(SourceIpRoutesToPeer("127.0.0.1", "nocolon"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SourceIpRoutesToPeerTest, SuccessfulResultsAreMemoizedPerSourceAndPeer) {
  ClearSourceIpRouteCacheForTesting();
  EXPECT_EQ(SourceIpRouteCacheSizeForTesting(), 0);

  EXPECT_THAT(SourceIpRoutesToPeer("127.0.0.3", "127.0.0.1:1"),
              IsOkAndHolds(true));
  EXPECT_EQ(SourceIpRouteCacheSizeForTesting(), 1);

  // An identical call is served from the cache and adds no entry.
  EXPECT_THAT(SourceIpRoutesToPeer("127.0.0.3", "127.0.0.1:1"),
              IsOkAndHolds(true));
  EXPECT_EQ(SourceIpRouteCacheSizeForTesting(), 1);

  // A different peer port is a distinct key.
  EXPECT_THAT(SourceIpRoutesToPeer("127.0.0.3", "127.0.0.1:2"),
              IsOkAndHolds(true));
  EXPECT_EQ(SourceIpRouteCacheSizeForTesting(), 2);
}

TEST(SourceIpRoutesToPeerTest, ErrorsAreNotMemoized) {
  ClearSourceIpRouteCacheForTesting();
  // TEST-NET-1 is never assigned to a local interface, so this fails uncached.
  EXPECT_THAT(SourceIpRoutesToPeer("192.0.2.1", "127.0.0.1:1"),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_EQ(SourceIpRouteCacheSizeForTesting(), 0);

  // Re-evaluated, not served from the cache: still the same error.
  EXPECT_THAT(SourceIpRoutesToPeer("192.0.2.1", "127.0.0.1:1"),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_EQ(SourceIpRouteCacheSizeForTesting(), 0);
}

TEST(SourceIpRoutesToPeerTest, HostnamePeerStillResolves) {
  ClearSourceIpRouteCacheForTesting();
  // The numeric-only fast path must fall back to name resolution. The value
  // is not asserted: it is true if `localhost` resolves to 127.0.0.1 first
  // and false if it resolves only to ::1 (no IPv4 address for the peer).
  ABSL_EXPECT_OK(SourceIpRoutesToPeer("127.0.0.3", "localhost:1"));
}

}  // namespace
}  // namespace tpu_raiden::transport::lib
