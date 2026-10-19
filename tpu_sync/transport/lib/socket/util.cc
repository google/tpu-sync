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
#include <ifaddrs.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "peregrine/src/api/v1/ip_util.h"
#include "tpu_sync/transport/lib/socket/tcp_psp_helper.h"

namespace tpu_raiden::transport::lib {

namespace {
using ::peregrine::v1::GetAddrPortPair;

// Without a bound, connect() to a dead peer blocks for the kernel's SYN
// retries (~127 s by default). Under large fan-out bursts, however, a single
// 3s attempt can time out if two SYN/SYN-ACK packets are dropped (1 s + 2 s
// Linux SYN RTO), so ConnectToPeer retries transient connect timeouts with
// jittered exponential backoff on a fresh socket (new ephemeral port).
constexpr int kDefaultConnectTimeoutMs = 3000;
constexpr int kDefaultMaxConnectAttempts = 4;
constexpr int kDefaultInitialBackoffMs = 200;
constexpr absl::Duration kMaxConnectBackoff = absl::Seconds(4);

int GetPositiveIntFromEnvOrDefault(const char* name, int default_val) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_val;
  }
  int parsed = 0;
  if (!absl::SimpleAtoi(value, &parsed) || parsed <= 0) {
    LOG(WARNING) << name << "=\"" << value
                 << "\" must be a positive integer; using default "
                 << default_val;
    return default_val;
  }
  return parsed;
}

// Sets SO_SNDTIMEO, which on Linux also bounds connect(). Zero means none.
void SetSendTimeout(int fd, absl::Duration timeout) {
  const timeval tv = absl::ToTimeval(timeout);
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

// Returns the socket's SO_SNDTIMEO (zero if unset or unreadable).
absl::Duration GetSendTimeout(int fd) {
  timeval tv = {};
  socklen_t len = sizeof(tv);
  getsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, &len);
  return absl::DurationFromTimeval(tv);
}

// Connects the blocking socket `fd` to `addr`, giving up after `timeout`. A
// short timeout keeps a dead peer from hogging the calling thread, which could
// starve the shared socket worker pool. Transfers on the connected socket keep
// blocking I/O.
absl::Status ConnectWithTimeout(int fd, const addrinfo& addr,
                                absl::Duration timeout, bool require_psp,
                                std::shared_ptr<grpc::Channel> channel,
                                ConnectTiming* timing) {
  const absl::Duration saved_timeout = GetSendTimeout(fd);
  SetSendTimeout(fd, timeout);
  // Restore on every return so the timeout never leaks into transfers.
  absl::Cleanup restore_timeout = [fd, saved_timeout] {
    SetSendTimeout(fd, saved_timeout);
  };
  if (require_psp) {
    double psp_kex_ms = 0.0;
    double tcp_connect_ms = 0.0;
    absl::Status status =
        TcpPspConnect(fd, addr.ai_addr, addr.ai_addrlen, std::move(channel),
                      timing != nullptr ? &psp_kex_ms : nullptr,
                      timing != nullptr ? &tcp_connect_ms : nullptr);
    if (status.ok() && timing != nullptr) {
      timing->psp_key_exchange_ms = psp_kex_ms;
      timing->connect_ms = tcp_connect_ms;
    }
    return status;
  }
  absl::Time connect_start =
      (timing != nullptr) ? absl::Now() : absl::InfinitePast();
  if (connect(fd, addr.ai_addr, addr.ai_addrlen) == 0) {
    if (timing != nullptr) {
      timing->connect_ms =
          absl::ToDoubleMilliseconds(absl::Now() - connect_start);
      timing->psp_key_exchange_ms = 0.0;
    }
    return absl::OkStatus();
  }
  // On a blocking socket, connect() fails with EINPROGRESS when SO_SNDTIMEO
  // expires (see socket(7)), or EINTR if interrupted by a signal while waiting.
  if (errno == EINPROGRESS || errno == ETIMEDOUT || errno == EINTR) {
    return absl::DeadlineExceededError(absl::StrCat(
        "connect timed out after ", absl::FormatDuration(timeout)));
  }
  return absl::ErrnoToStatus(errno, "connect failed");
}

// Splits `peer` ("host:port" or "[v6]:port") into (host, port).
absl::StatusOr<std::pair<std::string, std::string>> SplitHostPort(
    absl::string_view peer) {
  if (!peer.empty() && peer.front() == '[') {
    size_t closing_bracket = peer.find(']');
    if (closing_bracket == absl::string_view::npos ||
        closing_bracket + 1 >= peer.size() ||
        peer[closing_bracket + 1] != ':') {
      return absl::InvalidArgumentError(
          "Invalid IPv6 peer bracket string format");
    }
    return std::make_pair(std::string(peer.substr(1, closing_bracket - 1)),
                          std::string(peer.substr(closing_bracket + 2)));
  }
  std::vector<std::string> parts = absl::StrSplit(peer, ':');
  if (parts.size() != 2) {
    return absl::InvalidArgumentError("Invalid peer string format");
  }
  return std::make_pair(std::move(parts[0]), std::move(parts[1]));
}
}  // namespace

absl::StatusOr<int> ConnectToPeer(absl::string_view peer,
                                  absl::string_view local_ip, bool require_psp,
                                  std::shared_ptr<grpc::Channel> channel,
                                  ConnectTiming* timing) {
  if (require_psp && channel == nullptr) {
    return absl::InvalidArgumentError(
        "gRPC channel is required for PSP connection");
  }

  ABSL_ASSIGN_OR_RETURN(const auto host_port, SplitHostPort(peer));
  const std::string& host = host_port.first;
  const std::string& port_str = host_port.second;

  struct addrinfo hints;
  struct addrinfo* result = nullptr;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  int ret = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result);
  if (ret != 0 || result == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "getaddrinfo failed for host ", host, ": ", gai_strerror(ret)));
  }

  const absl::Duration connect_timeout =
      absl::Milliseconds(GetPositiveIntFromEnvOrDefault(
          "TPU_RAIDEN_TCP_CONNECT_TIMEOUT_MS", kDefaultConnectTimeoutMs));
  const int max_attempts = GetPositiveIntFromEnvOrDefault(
      "TPU_RAIDEN_TCP_CONNECT_MAX_ATTEMPTS", kDefaultMaxConnectAttempts);
  const absl::Duration initial_backoff =
      absl::Milliseconds(GetPositiveIntFromEnvOrDefault(
          "TPU_RAIDEN_TCP_CONNECT_INITIAL_BACKOFF_MS",
          kDefaultInitialBackoffMs));

  int sock_fd = -1;
  int last_errno = 0;
  absl::Status last_status = absl::OkStatus();
  absl::BitGen bitgen;

  for (int attempt = 0; attempt < max_attempts; ++attempt) {
    for (struct addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
      sock_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
      if (sock_fd < 0) {
        last_errno = errno;
        continue;
      }

      int opt = 1;
      setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
      int buf_opt = 16 * 1024 * 1024;  // 16MB
      setsockopt(sock_fd, SOL_SOCKET, SO_SNDBUF, &buf_opt, sizeof(buf_opt));
      setsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &buf_opt, sizeof(buf_opt));

      bool should_bind =
          !local_ip.empty() && local_ip != "0.0.0.0" && local_ip != "::";

      if (should_bind) {
        std::string local_ip_str(local_ip);
        bool is_ipv6 = absl::StrContains(local_ip, ':');
        if (is_ipv6 && rp->ai_family == AF_INET6) {
          struct sockaddr_in6 local_addr;
          std::memset(&local_addr, 0, sizeof(local_addr));
          local_addr.sin6_family = AF_INET6;
          if (inet_pton(AF_INET6, local_ip_str.c_str(), &local_addr.sin6_addr) >
              0) {
            local_addr.sin6_port = 0;
            if (bind(sock_fd, (struct sockaddr*)&local_addr,
                     sizeof(local_addr)) < 0) {
              LOG(WARNING) << "Client bind IPv6 failed to " << local_ip << ": "
                           << std::strerror(errno);
            }
          }
        } else if (!is_ipv6 && rp->ai_family == AF_INET) {
          struct sockaddr_in local_addr;
          std::memset(&local_addr, 0, sizeof(local_addr));
          local_addr.sin_family = AF_INET;
          if (inet_pton(AF_INET, local_ip_str.c_str(), &local_addr.sin_addr) >
              0) {
            local_addr.sin_port = 0;
            if (bind(sock_fd, (struct sockaddr*)&local_addr,
                     sizeof(local_addr)) < 0) {
              LOG(WARNING) << "Client bind IPv4 failed to " << local_ip << ": "
                           << std::strerror(errno);
            }
          }
        }
      }

      const absl::Status connect_status = ConnectWithTimeout(
          sock_fd, *rp, connect_timeout, require_psp, channel, timing);
      if (connect_status.ok()) {
        break; /* Success */
      }

      last_status = connect_status;
      close(sock_fd);
      sock_fd = -1;
    }

    if (sock_fd >= 0) {
      break;
    }
    if (!absl::IsDeadlineExceeded(last_status) || attempt + 1 >= max_attempts) {
      break;
    }

    const absl::Duration base_backoff = std::min(
        initial_backoff * (1 << std::min(attempt, 10)), kMaxConnectBackoff);
    const absl::Duration backoff =
        base_backoff * absl::Uniform(bitgen, 0.5, 1.5);
    LOG(WARNING) << "Connect to peer " << peer << " timed out on attempt "
                 << (attempt + 1) << "/" << max_attempts << " ("
                 << last_status.message() << "); retrying in "
                 << absl::FormatDuration(backoff);
    absl::SleepFor(backoff);
  }

  freeaddrinfo(result);

  if (sock_fd < 0) {
    if (!last_status.ok()) {
      return absl::UnavailableError(absl::StrCat(
          "Failed to connect to peer ", peer, ": ", last_status.message()));
    }
    return absl::UnavailableError(absl::StrCat(
        "Failed to connect to peer ", peer, ": ", std::strerror(last_errno)));
  }

  LOG(INFO) << absl::StrCat("connected tcp socket ", sock_fd, ": ",
                            GetAddrPortPair(sock_fd));
  return sock_fd;
}

}  // namespace tpu_raiden::transport::lib
