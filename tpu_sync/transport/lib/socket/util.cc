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
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "tpu_sync/transport/lib/socket/tcp_psp_helper.h"

namespace tpu_raiden::transport::lib {

namespace {

// Without a bound, connect() to a dead peer blocks for the kernel's SYN
// retries (~127 s by default).
constexpr absl::Duration kConnectTimeout = absl::Seconds(3);

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
  // expires (see socket(7)).
  if (errno == EINPROGRESS) {
    return absl::DeadlineExceededError(absl::StrCat(
        "connect timed out after ", absl::FormatDuration(timeout)));
  }
  return absl::ErrnoToStatus(errno, "connect failed");
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
      // Bind only the source address; choose the port at connect() so different
      // destinations can share a source port without sharing a TCP connection.
      if (setsockopt(sock_fd, IPPROTO_IP, IP_BIND_ADDRESS_NO_PORT, &opt,
                     sizeof(opt)) < 0) {
        last_status = absl::ErrnoToStatus(
            errno, "setsockopt(IP_BIND_ADDRESS_NO_PORT) failed");
        close(sock_fd);
        sock_fd = -1;
        continue;
      }
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

    const absl::Status connect_status = ConnectWithTimeout(
        sock_fd, *rp, kConnectTimeout, require_psp, channel, timing);
    if (connect_status.ok()) {
      break; /* Success */
    }

    last_status = connect_status;
    close(sock_fd);
    sock_fd = -1;
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

namespace {
std::string SockAddrToEndpoint(int fd, int (*get_fn)(int, struct sockaddr*,
                                                     socklen_t*)) {
  struct sockaddr_storage ss{};
  socklen_t len = sizeof(ss);
  if (get_fn(fd, reinterpret_cast<struct sockaddr*>(&ss), &len) != 0) {
    return "";
  }
  char host[NI_MAXHOST] = "", serv[NI_MAXSERV] = "";
  if (getnameinfo(reinterpret_cast<const struct sockaddr*>(&ss), len, host,
                  sizeof(host), serv, sizeof(serv),
                  NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
    return "";
  }
  return ss.ss_family == AF_INET6 ? absl::StrCat("[", host, "]:", serv)
                                  : absl::StrCat(host, ":", serv);
}
}  // namespace

std::string GetLocalEndpoint(int fd) {
  return SockAddrToEndpoint(fd, ::getsockname);
}

std::string GetAddrPortPair(int fd) {
  return absl::StrCat(SockAddrToEndpoint(fd, ::getsockname), " <> ",
                      SockAddrToEndpoint(fd, ::getpeername));
}

}  // namespace tpu_raiden::transport::lib
