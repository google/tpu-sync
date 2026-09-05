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
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "tpu_sync/transport/lib/socket/tcp_psp_helper.h"

ABSL_FLAG(absl::Duration, raiden_transport_io_timeout, absl::Minutes(5),
          "Bounds one blocking send() or recv() on a data plane socket, and "
          "bounds how long the kernel retransmits unacknowledged data before "
          "failing the connection. The default is deliberately generous: it "
          "is a ceiling on a stalled peer, not a latency target. Zero "
          "restores unbounded blocking I/O with no keepalive.");

ABSL_FLAG(absl::Duration, raiden_transport_connect_timeout, absl::Seconds(30),
          "Bounds one connect() attempt to a data plane peer. Without it a "
          "connect to a blackholed address blocks for the kernel SYN retry "
          "window, roughly 127 seconds at Linux defaults. Zero restores a "
          "plain blocking connect().");

namespace tpu_raiden::transport::lib {
namespace {

// Number of unacknowledged keepalive probes before the connection is failed.
constexpr int kKeepaliveProbeCount = 3;

// Sets `opt` on `fd` at `level`, logging rather than failing when the kernel
// does not support it. A socket that keeps one fewer bound still works.
void SetSocketOption(int fd, int level, int opt, absl::string_view name,
                     const void* val, socklen_t len) {
  if (setsockopt(fd, level, opt, val, len) < 0) {
    LOG_EVERY_N_SEC(WARNING, 60)
        << "Failed to set " << name << " on data plane socket " << fd << ": "
        << std::strerror(errno);
  }
}

// Connects `fd` to `addr` within `timeout`, then restores the socket's
// original blocking mode. Falls back to a plain blocking connect() when
// `timeout` is not positive.
absl::Status ConnectWithTimeout(int fd, const struct sockaddr* addr,
                                socklen_t addrlen, absl::Duration timeout) {
  if (timeout <= absl::ZeroDuration()) {
    if (connect(fd, addr, addrlen) < 0) {
      return absl::ErrnoToStatus(errno, "connect failed");
    }
    return absl::OkStatus();
  }

  const int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) {
    return absl::ErrnoToStatus(errno, "fcntl(F_GETFL) failed");
  }
  if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
    return absl::ErrnoToStatus(errno, "fcntl(F_SETFL, O_NONBLOCK) failed");
  }
  // The send and recv paths assert blocking mode, so restore it on every exit
  // including the error paths below.
  absl::Cleanup restore_blocking = [fd, flags] { fcntl(fd, F_SETFL, flags); };

  if (connect(fd, addr, addrlen) == 0) {
    return absl::OkStatus();
  }
  if (errno != EINPROGRESS) {
    return absl::ErrnoToStatus(errno, "connect failed");
  }

  // poll() reports the socket writable once the handshake settles, whether it
  // succeeded or failed; SO_ERROR carries which.
  const absl::Time deadline = absl::Now() + timeout;
  while (true) {
    const absl::Duration remaining = deadline - absl::Now();
    if (remaining <= absl::ZeroDuration()) break;
    // Round up so a sub-millisecond remainder polls once rather than spinning.
    const int remaining_ms = static_cast<int>(
        std::max<int64_t>(1, absl::ToInt64Milliseconds(remaining)));

    struct pollfd pfd = {.fd = fd, .events = POLLOUT};
    const int ret = poll(&pfd, /*nfds=*/1, remaining_ms);
    if (ret < 0) {
      if (errno == EINTR) continue;
      return absl::ErrnoToStatus(errno, "poll during connect failed");
    }
    if (ret == 0) continue;  // Timed out; the deadline check above decides.

    int so_error = 0;
    socklen_t so_error_len = sizeof(so_error);
    if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &so_error_len) < 0) {
      return absl::ErrnoToStatus(errno, "getsockopt(SO_ERROR) failed");
    }
    if (so_error != 0) {
      return absl::ErrnoToStatus(so_error, "connect failed");
    }
    return absl::OkStatus();
  }

  return absl::DeadlineExceededError(
      absl::StrCat("connect did not complete within ",
                   absl::FormatDuration(timeout)));
}

}  // namespace

void ApplyDataPlaneSocketOptions(int fd) {
  const absl::Duration io_timeout =
      absl::GetFlag(FLAGS_raiden_transport_io_timeout);
  if (io_timeout <= absl::ZeroDuration()) return;

  const struct timeval tv = absl::ToTimeval(io_timeout);
  SetSocketOption(fd, SOL_SOCKET, SO_SNDTIMEO, "SO_SNDTIMEO", &tv, sizeof(tv));
  SetSocketOption(fd, SOL_SOCKET, SO_RCVTIMEO, "SO_RCVTIMEO", &tv, sizeof(tv));

#ifdef TCP_USER_TIMEOUT
  const int user_timeout_ms =
      static_cast<int>(absl::ToInt64Milliseconds(io_timeout));
  SetSocketOption(fd, IPPROTO_TCP, TCP_USER_TIMEOUT, "TCP_USER_TIMEOUT",
                  &user_timeout_ms, sizeof(user_timeout_ms));
#endif

  const int enable = 1;
  SetSocketOption(fd, SOL_SOCKET, SO_KEEPALIVE, "SO_KEEPALIVE", &enable,
                  sizeof(enable));

  // Probing starts a quarter of the way into the budget and the probes span
  // another quarter, so an idle peer that vanished is declared dead around
  // halfway through io_timeout, ahead of the TCP_USER_TIMEOUT that governs a
  // connection with data in flight. The tuning knobs are Linux specific; on a
  // platform without them the connection still gets default keepalive.
#if defined(TCP_KEEPIDLE) || defined(TCP_KEEPINTVL)
  const int64_t io_timeout_secs = absl::ToInt64Seconds(io_timeout);
#ifdef TCP_KEEPIDLE
  const int idle_secs =
      std::max<int>(1, static_cast<int>(io_timeout_secs / 4));
  SetSocketOption(fd, IPPROTO_TCP, TCP_KEEPIDLE, "TCP_KEEPIDLE", &idle_secs,
                  sizeof(idle_secs));
#endif
#ifdef TCP_KEEPINTVL
  const int interval_secs = std::max<int>(
      1, static_cast<int>(io_timeout_secs / (4 * kKeepaliveProbeCount)));
  SetSocketOption(fd, IPPROTO_TCP, TCP_KEEPINTVL, "TCP_KEEPINTVL",
                  &interval_secs, sizeof(interval_secs));
#endif
#endif
#ifdef TCP_KEEPCNT
  SetSocketOption(fd, IPPROTO_TCP, TCP_KEEPCNT, "TCP_KEEPCNT",
                  &kKeepaliveProbeCount, sizeof(kKeepaliveProbeCount));
#endif
}

absl::StatusOr<int> ConnectToPeer(
    absl::string_view peer, absl::string_view local_ip, bool require_psp,
    std::shared_ptr<grpc::Channel> channel) {
  if (require_psp && channel == nullptr) {
    return absl::InvalidArgumentError(
        "gRPC channel is required for PSP connection");
  }

  std::string host;
  std::string port_str;

  if (!peer.empty() && peer.front() == '[') {
    size_t closing_bracket = peer.find(']');
    if (closing_bracket == absl::string_view::npos ||
        closing_bracket + 1 >= peer.size() ||
        peer[closing_bracket + 1] != ':') {
      return absl::InvalidArgumentError(
          "Invalid IPv6 peer bracket string format");
    }
    host = std::string(peer.substr(1, closing_bracket - 1));
    port_str = std::string(peer.substr(closing_bracket + 2));
  } else {
    std::vector<std::string> parts = absl::StrSplit(peer, ':');
    if (parts.size() != 2) {
      return absl::InvalidArgumentError("Invalid peer string format");
    }
    host = parts[0];
    port_str = parts[1];
  }

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

  int sock_fd = -1;
  struct addrinfo* rp;
  int last_errno = 0;
  absl::Status last_status = absl::OkStatus();
  for (rp = result; rp != nullptr; rp = rp->ai_next) {
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
          if (bind(sock_fd, (struct sockaddr*)&local_addr, sizeof(local_addr)) <
              0) {
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
          if (bind(sock_fd, (struct sockaddr*)&local_addr, sizeof(local_addr)) <
              0) {
            LOG(WARNING) << "Client bind IPv4 failed to " << local_ip << ": "
                         << std::strerror(errno);
          }
        }
      }
    }

    absl::Status connect_status;
    if (require_psp) {
      // PSP performs its own connect as part of the out-of-band key exchange,
      // so the connect deadline below does not cover it.
      connect_status =
          TcpPspConnect(sock_fd, rp->ai_addr, rp->ai_addrlen, channel);
    } else {
      connect_status = ConnectWithTimeout(
          sock_fd, rp->ai_addr, rp->ai_addrlen,
          absl::GetFlag(FLAGS_raiden_transport_connect_timeout));
    }

    if (connect_status.ok()) {
      ApplyDataPlaneSocketOptions(sock_fd);
      break; /* Success */
    }

    last_status = connect_status;
    close(sock_fd);
    sock_fd = -1;
  }

  freeaddrinfo(result);

  if (sock_fd < 0) {
    if (!last_status.ok()) {
      std::string msg = absl::StrCat("Failed to connect to peer ", peer, ": ",
                                     last_status.message());
      // A connect that ran out of time is reported as such, so it is
      // distinguishable from a refused or unroutable peer both by callers and
      // in the transfer failure metric's error_code label. Every other
      // failure keeps the historical UNAVAILABLE.
      if (last_status.code() == absl::StatusCode::kDeadlineExceeded) {
        return absl::DeadlineExceededError(std::move(msg));
      }
      return absl::UnavailableError(std::move(msg));
    }
    return absl::UnavailableError(absl::StrCat(
        "Failed to connect to peer ", peer, ": ", std::strerror(last_errno)));
  }

  return sock_fd;
}

}  // namespace tpu_raiden::transport::lib
