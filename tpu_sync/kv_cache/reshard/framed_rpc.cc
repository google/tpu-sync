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

#include "tpu_sync/kv_cache/reshard/framed_rpc.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <thread>        // NOLINT(build/c++11)
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {
namespace {

// Reads exactly n bytes; false on EOF/error.
bool ReadExactly(int fd, char* out, size_t n) {
  size_t total = 0;
  while (total < n) {
    ssize_t r = read(fd, out + total, n - total);
    if (r <= 0) return false;
    total += static_cast<size_t>(r);
  }
  return true;
}

bool SendAll(int fd, const char* data, size_t n) {
  size_t total = 0;
  while (total < n) {
    ssize_t w = send(fd, data + total, n - total, MSG_NOSIGNAL);
    if (w <= 0) return false;
    total += static_cast<size_t>(w);
  }
  return true;
}

// Splits "host:port" at the last colon; strips IPv6 brackets, mirroring
// raiden_controller.connect_socket.
absl::Status SplitAddress(absl::string_view address, std::string* host,
                          int* port) {
  size_t rindex = address.rfind(':');
  if (rindex == absl::string_view::npos) {
    return absl::InvalidArgumentError(
        absl::StrCat("Malformed endpoint (no port): ", address));
  }
  absl::string_view host_view = address.substr(0, rindex);
  if (host_view.size() >= 2 && host_view.front() == '[' &&
      host_view.back() == ']') {
    host_view = host_view.substr(1, host_view.size() - 2);
  }
  int parsed_port = 0;
  if (!absl::SimpleAtoi(address.substr(rindex + 1), &parsed_port)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Malformed endpoint port: ", address));
  }
  *host = std::string(host_view);
  *port = parsed_port;
  return absl::OkStatus();
}

// One connection attempt over every resolved address family.
int TryConnectOnce(const std::string& host, int port,
                   absl::Duration io_timeout) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* result = nullptr;
  const std::string port_str = absl::StrCat(port);
  if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result) != 0) {
    return -1;
  }
  int fd = -1;
  for (addrinfo* res = result; res != nullptr; res = res->ai_next) {
    fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) continue;
    timeval tv;
    tv.tv_sec = absl::ToInt64Seconds(io_timeout);
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    if (connect(fd, res->ai_addr, res->ai_addrlen) == 0) break;
    close(fd);
    fd = -1;
  }
  freeaddrinfo(result);
  return fd;
}

}  // namespace

absl::StatusOr<std::string> SocketFramedTransport::Call(
    absl::string_view address, absl::string_view payload,
    absl::Duration timeout) {
  std::string host;
  int port = 0;
  absl::Status split = SplitAddress(address, &host, &port);
  if (!split.ok()) return split;

  // Python's connect_socket retries every 2 s until the deadline; keep that
  // behavior so a worker that is still binding its listener is awaited, not
  // failed.
  const absl::Time deadline = absl::Now() + timeout;
  int fd = -1;
  while (true) {
    fd = TryConnectOnce(host, port, timeout);
    if (fd >= 0) break;
    if (absl::Now() > deadline) {
      return absl::DeadlineExceededError(
          absl::StrCat("Timeout (", absl::ToInt64Seconds(timeout),
                       "s) failed to connect to robust endpoint ", address));
    }
    absl::SleepFor(absl::Seconds(2));
  }

  std::string response;
  {
    uint32_t net_len = htonl(static_cast<uint32_t>(payload.size()));
    if (!SendAll(fd, reinterpret_cast<const char*>(&net_len),
                 sizeof(net_len)) ||
        !SendAll(fd, payload.data(), payload.size())) {
      close(fd);
      return absl::UnavailableError(
          absl::StrCat("Failed to send framed payload to ", address, ": ",
                       std::strerror(errno)));
    }
    uint32_t resp_net_len = 0;
    if (!ReadExactly(fd, reinterpret_cast<char*>(&resp_net_len),
                     sizeof(resp_net_len))) {
      close(fd);
      return absl::UnavailableError(
          "Remote servicer closed connection while reading response length");
    }
    uint32_t resp_len = ntohl(resp_net_len);
    response.resize(resp_len);
    if (resp_len > 0 && !ReadExactly(fd, response.data(), resp_len)) {
      close(fd);
      return absl::UnavailableError(
          "Remote servicer closed connection while reading response data");
    }
  }
  close(fd);
  return response;
}

FramedServer::FramedServer(int port, Handler handler)
    : requested_port_(port), handler_(std::move(handler)) {}

FramedServer::~FramedServer() { Stop(); }

absl::Status FramedServer::Bind() {
  // Dual-stack IPv6 listener with IPv4 fallback, mirroring
  // raiden_controller.create_server_socket.
  server_fd_ = socket(AF_INET6, SOCK_STREAM, 0);
  if (server_fd_ >= 0) {
    int off = 0;
    setsockopt(server_fd_, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof(off));
    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in6 address{};
    address.sin6_family = AF_INET6;
    address.sin6_port = htons(static_cast<uint16_t>(requested_port_));
    address.sin6_addr = in6addr_any;
    if (bind(server_fd_, reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) == 0 &&
        listen(server_fd_, 128) == 0) {
      socklen_t addr_len = sizeof(address);
      if (getsockname(server_fd_, reinterpret_cast<sockaddr*>(&address),
                      &addr_len) == 0) {
        port_ = ntohs(address.sin6_port);
      }
      return absl::OkStatus();
    }
    close(server_fd_);
    server_fd_ = -1;
  }
  server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd_ < 0) {
    return absl::InternalError(absl::StrCat(
        "Failed to create listener socket: ", std::strerror(errno)));
  }
  int opt = 1;
  setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  sockaddr_in v4{};
  v4.sin_family = AF_INET;
  v4.sin_port = htons(static_cast<uint16_t>(requested_port_));
  v4.sin_addr.s_addr = INADDR_ANY;
  if (bind(server_fd_, reinterpret_cast<sockaddr*>(&v4), sizeof(v4)) != 0 ||
      listen(server_fd_, 128) != 0) {
    int saved_errno = errno;
    close(server_fd_);
    server_fd_ = -1;
    return absl::InternalError(absl::StrCat("Failed to bind port ",
                                            requested_port_, ": ",
                                            std::strerror(saved_errno)));
  }
  sockaddr_in bound{};
  socklen_t bound_len = sizeof(bound);
  if (getsockname(server_fd_, reinterpret_cast<sockaddr*>(&bound),
                  &bound_len) == 0) {
    port_ = ntohs(bound.sin_port);
  }
  return absl::OkStatus();
}

void FramedServer::Start() {
  accept_thread_ = std::thread(&FramedServer::AcceptLoop, this);
}

void FramedServer::Stop() {
  if (stopping_.exchange(true)) {
    return;
  }
  // shutdown() is what breaks the blocking accept(); the descriptor must stay
  // valid until the accept thread is joined, or accept() could be handed a
  // recycled descriptor.
  if (server_fd_ >= 0) {
    shutdown(server_fd_, SHUT_RDWR);
  }
  if (accept_thread_.joinable()) accept_thread_.join();
  if (server_fd_ >= 0) {
    close(server_fd_);
    server_fd_ = -1;
  }
  connection_threads_.AwaitAllDone();
}

void FramedServer::AcceptLoop() {
  while (!stopping_) {
    sockaddr_storage client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int client_fd = accept(
        server_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
    if (client_fd < 0) {
      if (stopping_) break;
      continue;
    }
    if (stopping_) {
      close(client_fd);
      break;
    }
    if (!connection_threads_.Spawn(
            [this, client_fd] { ServeConnection(client_fd); })) {
      close(client_fd);
      // Back off so a persistent spawn failure cannot spin the accept loop.
      absl::SleepFor(absl::Milliseconds(10));
    }
  }
}

void FramedServer::ServeConnection(int client_fd) {
  uint32_t net_len = 0;
  if (!ReadExactly(client_fd, reinterpret_cast<char*>(&net_len),
                   sizeof(net_len))) {
    close(client_fd);
    return;
  }
  uint32_t payload_len = ntohl(net_len);
  std::string request(payload_len, '\0');
  if (payload_len > 0 && !ReadExactly(client_fd, request.data(), payload_len)) {
    close(client_fd);
    return;
  }
  std::string response = handler_(request);
  uint32_t resp_net_len = htonl(static_cast<uint32_t>(response.size()));
  SendAll(client_fd, reinterpret_cast<const char*>(&resp_net_len),
          sizeof(resp_net_len));
  SendAll(client_fd, response.data(), response.size());
  close(client_fd);
}

}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden
