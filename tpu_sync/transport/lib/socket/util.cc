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
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "grpcpp/channel.h"
#include "tpu_sync/transport/lib/socket/tcp_psp_helper.h"

namespace tpu_raiden::transport::lib {

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
      connect_status =
          TcpPspConnect(sock_fd, rp->ai_addr, rp->ai_addrlen, channel);
    } else {
      if (connect(sock_fd, rp->ai_addr, rp->ai_addrlen) >= 0) {
        connect_status = absl::OkStatus();
      } else {
        last_errno = errno;
        connect_status = absl::ErrnoToStatus(errno, "connect failed");
      }
    }

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

  return sock_fd;
}

}  // namespace tpu_raiden::transport::lib
