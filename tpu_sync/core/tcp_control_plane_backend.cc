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

#include "tpu_sync/core/tcp_control_plane_backend.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tpu_sync/core/control_plane_backend.h"

namespace tpu_raiden {

TcpControlPlaneBackend::TcpControlPlaneBackend(TaskExecutor executor,
                                               absl::Duration default_timeout)
    : executor_(std::move(executor)), default_timeout_(default_timeout) {}

TcpControlPlaneBackend::~TcpControlPlaneBackend() { StopServer(); }

bool TcpControlPlaneBackend::EncodeIpToIpv6Bytes(const std::string& ip,
                                                 uint8_t out[16]) {
  const std::string mapped = absl::StrContains(ip, ':') ? ip : "::ffff:" + ip;
  if (inet_pton(AF_INET6, mapped.c_str(), out) <= 0) {
    std::memset(out, 0, 16);
    return false;
  }
  return true;
}

absl::StatusOr<std::pair<std::string, int>>
TcpControlPlaneBackend::SplitEndpoint(absl::string_view endpoint) {
  if (endpoint.empty()) {
    return absl::InvalidArgumentError("endpoint is empty");
  }
  std::string host;
  absl::string_view port_str;
  if (absl::StartsWith(endpoint, "[")) {
    if (!absl::StrContains(endpoint, "]:")) {
      return absl::InvalidArgumentError(
          absl::StrCat("invalid IPv6 endpoint: ", endpoint));
    }
    std::pair<absl::string_view, absl::string_view> split =
        absl::StrSplit(endpoint.substr(1), absl::MaxSplits("]:", 1));
    host = std::string(split.first);
    port_str = split.second;
  } else {
    if (!absl::StrContains(endpoint, ':')) {
      return absl::InvalidArgumentError("endpoint must be host:port");
    }
    size_t colon = endpoint.rfind(':');
    host = std::string(endpoint.substr(0, colon));
    port_str = endpoint.substr(colon + 1);
  }
  int port = 0;
  if (!absl::SimpleAtoi(port_str, &port)) {
    return absl::InvalidArgumentError(
        absl::StrCat("invalid port in endpoint: ", endpoint));
  }
  return std::make_pair(std::move(host), port);
}

absl::Status TcpControlPlaneBackend::SetSocketTimeouts(int fd,
                                                       double timeout_s) {
  if (timeout_s <= 0) return absl::OkStatus();
  timeval tv;
  tv.tv_sec = static_cast<time_t>(timeout_s);
  tv.tv_usec = static_cast<suseconds_t>((timeout_s - tv.tv_sec) * 1e6);
  if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0 ||
      setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) < 0) {
    return absl::InternalError(absl::StrCat(
        "setsockopt(SO_RCVTIMEO/SO_SNDTIMEO) failed: ", std::strerror(errno)));
  }
  return absl::OkStatus();
}

absl::StatusOr<int> TcpControlPlaneBackend::ConnectTcp(
    absl::string_view endpoint, double timeout_s) {
  absl::StatusOr<std::pair<std::string, int>> parsed = SplitEndpoint(endpoint);
  if (!parsed.ok()) return parsed.status();
  const auto& [host, port] = *parsed;

  struct addrinfo hints;
  struct addrinfo* res = nullptr;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  std::string port_str = std::to_string(port);
  int err = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res);
  if (err != 0) {
    return absl::UnavailableError(absl::StrCat("Failed to resolve hostname '",
                                               host, "': ", gai_strerror(err)));
  }

  int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  if (fd < 0) {
    int saved_errno = errno;
    freeaddrinfo(res);
    return absl::InternalError(
        absl::StrCat("socket() failed: ", std::strerror(saved_errno)));
  }
  int opt = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
  if (absl::Status status = SetSocketTimeouts(fd, timeout_s); !status.ok()) {
    close(fd);
    freeaddrinfo(res);
    return status;
  }

  if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
    int saved_errno = errno;
    close(fd);
    freeaddrinfo(res);
    if (saved_errno == ETIMEDOUT || saved_errno == EAGAIN ||
        saved_errno == EWOULDBLOCK || saved_errno == EINPROGRESS) {
      return absl::DeadlineExceededError(absl::StrCat(
          "connect(", endpoint, ") timed out: ", std::strerror(saved_errno)));
    }
    return absl::UnavailableError(absl::StrCat(
        "connect(", endpoint, ") failed: ", std::strerror(saved_errno)));
  }
  freeaddrinfo(res);
  return fd;
}

// Smallest budget worth arming the socket with. SetSocketTimeouts splits a
// double into {tv_sec, tv_usec}, and a timeval of {0, 0} means "block forever"
// on Linux, not "give up now" -- so a budget that has all but run out has to
// fail here rather than be rounded down into no bound at all.
constexpr absl::Duration kMinArmableBudget = absl::Milliseconds(1);

// Re-arms the socket to what is left of `deadline` before the next syscall.
// Returns DeadlineExceeded once nothing is left, which is the only thing that
// stops a peer from restarting the socket timeout with every byte it sends.
static absl::Status ArmForRemaining(int fd, absl::Time deadline) {
  if (deadline == absl::InfiniteFuture()) return absl::OkStatus();
  const absl::Duration remaining = deadline - absl::Now();
  if (remaining < kMinArmableBudget) {
    return absl::DeadlineExceededError("control message deadline exceeded");
  }
  return TcpControlPlaneBackend::SetSocketTimeouts(
      fd, absl::ToDoubleSeconds(remaining));
}

absl::Status TcpControlPlaneBackend::WriteExact(int fd, const void* buffer,
                                                size_t length,
                                                absl::Time deadline) {
  const uint8_t* ptr = static_cast<const uint8_t*>(buffer);
  size_t remaining = length;
  while (remaining > 0) {
    if (absl::Status s = ArmForRemaining(fd, deadline); !s.ok()) return s;
    ssize_t written = send(fd, ptr, remaining, MSG_NOSIGNAL);
    if (written < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        return absl::DeadlineExceededError("socket write timed out");
      }
      return absl::InternalError(
          absl::StrCat("socket write failed: ", std::strerror(errno)));
    }
    if (written == 0) {
      return absl::InternalError("socket closed during write");
    }
    ptr += written;
    remaining -= written;
  }
  return absl::OkStatus();
}

absl::Status TcpControlPlaneBackend::ReadExact(int fd, void* buffer,
                                               size_t length,
                                               absl::Time deadline) {
  uint8_t* ptr = static_cast<uint8_t*>(buffer);
  size_t remaining = length;
  while (remaining > 0) {
    if (absl::Status s = ArmForRemaining(fd, deadline); !s.ok()) return s;
    ssize_t bytes_read = read(fd, ptr, remaining);
    if (bytes_read < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        return absl::DeadlineExceededError("socket read timed out");
      }
      return absl::InternalError(
          absl::StrCat("socket read failed: ", std::strerror(errno)));
    }
    if (bytes_read == 0) {
      return absl::InternalError("socket closed during read");
    }
    ptr += bytes_read;
    remaining -= bytes_read;
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> TcpControlPlaneBackend::GetPeerIp(int fd) {
  sockaddr_storage addr;
  socklen_t len = sizeof(addr);
  if (getpeername(fd, reinterpret_cast<sockaddr*>(&addr), &len) < 0) {
    return absl::InternalError(
        absl::StrCat("getpeername() failed: ", std::strerror(errno)));
  }
  char ip_buf[INET6_ADDRSTRLEN];
  if (addr.ss_family == AF_INET) {
    sockaddr_in* s = reinterpret_cast<sockaddr_in*>(&addr);
    if (inet_ntop(AF_INET, &s->sin_addr, ip_buf, sizeof(ip_buf)) == nullptr) {
      return absl::InternalError(
          absl::StrCat("inet_ntop() failed: ", std::strerror(errno)));
    }
  } else if (addr.ss_family == AF_INET6) {
    sockaddr_in6* s = reinterpret_cast<sockaddr_in6*>(&addr);
    if (inet_ntop(AF_INET6, &s->sin6_addr, ip_buf, sizeof(ip_buf)) == nullptr) {
      return absl::InternalError(
          absl::StrCat("inet_ntop() failed: ", std::strerror(errno)));
    }
    absl::string_view ip_view(ip_buf);
    if (absl::StartsWithIgnoreCase(ip_view, "::ffff:")) {
      return std::string(ip_view.substr(7));
    }
  } else {
    return absl::InternalError("unknown socket family");
  }
  return std::string(ip_buf);
}

absl::Status TcpControlPlaneBackend::WriteBlockIds(
    int fd, const std::vector<int64_t>& block_ids, absl::Time deadline) {
  if (block_ids.empty()) return absl::OkStatus();
  return WriteExact(fd, block_ids.data(), block_ids.size() * sizeof(int64_t),
                    deadline);
}

absl::StatusOr<std::vector<int64_t>> TcpControlPlaneBackend::ReadBlockIds(
    int fd, uint64_t num_blocks, absl::Time deadline) {
  if (num_blocks == 0) return std::vector<int64_t>{};
  if (num_blocks > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return absl::InvalidArgumentError("num_blocks is too large");
  }
  std::vector<int64_t> block_ids(static_cast<size_t>(num_blocks));
  if (absl::Status s = ReadExact(
          fd, block_ids.data(), block_ids.size() * sizeof(int64_t), deadline);
      !s.ok()) {
    return s;
  }
  return block_ids;
}

TcpControlPlaneBackend::ControlResponseHeader
TcpControlPlaneBackend::ReadControlResponseHeader(int fd,
                                                  absl::Time deadline) {
  ControlResponseHeader response;
  if (absl::Status s = ReadExact(fd, &response, sizeof(response), deadline);
      !s.ok()) {
    throw std::runtime_error(
        absl::StrCat("control response read failed: ", s.message()));
  }
  if (response.magic != kResponseMagic) {
    throw std::runtime_error("bad control response magic");
  }
  if (response.status != 0) {
    const bool truncated = response.message_len > kMaxControlErrorMessageBytes;
    const size_t message_bytes = static_cast<size_t>(
        std::min(response.message_len, kMaxControlErrorMessageBytes));
    std::string message(message_bytes, '\0');
    if (message_bytes > 0) {
      if (absl::Status s =
              ReadExact(fd, message.data(), message.size(), deadline);
          !s.ok()) {
        throw std::runtime_error(
            absl::StrCat("control error body read failed: ", s.message()));
      }
    }
    if (truncated) {
      absl::StrAppend(&message, " [truncated; peer advertised ",
                      response.message_len, " bytes]");
    }
    throw std::runtime_error("remote Raiden control error: " + message);
  }
  return response;
}

absl::StatusOr<int> TcpControlPlaneBackend::StartServer(
    int requested_port, ControlPlaneHandler* handler) {
  if (control_fd_ >= 0) {
    StopServer();
  }
  handler_ = handler;

  control_fd_ = socket(AF_INET6, SOCK_STREAM, 0);
  if (control_fd_ < 0) {
    return absl::InternalError(
        absl::StrCat("control socket() failed: ", std::strerror(errno)));
  }
  int opt = 1;
  setsockopt(control_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  setsockopt(control_fd_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

  int ipv6only = 0;
  if (setsockopt(control_fd_, IPPROTO_IPV6, IPV6_V6ONLY, &ipv6only,
                 sizeof(ipv6only)) < 0) {
    LOG(WARNING) << "setsockopt IPV6_V6ONLY=0 failed: " << std::strerror(errno);
  }

  sockaddr_in6 addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin6_family = AF_INET6;
  addr.sin6_addr = in6addr_any;
  addr.sin6_port = htons(requested_port);

  if (bind(control_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::string err = std::strerror(errno);
    close(control_fd_);
    control_fd_ = -1;
    return absl::InternalError(
        absl::StrCat("control bind(", requested_port, ") failed: ", err));
  }
  socklen_t len = sizeof(addr);
  if (getsockname(control_fd_, reinterpret_cast<sockaddr*>(&addr), &len) < 0) {
    std::string err = std::strerror(errno);
    close(control_fd_);
    control_fd_ = -1;
    return absl::InternalError(absl::StrCat("getsockname() failed: ", err));
  }
  int bound_port = ntohs(addr.sin6_port);
  if (listen(control_fd_, 128) < 0) {
    std::string err = std::strerror(errno);
    close(control_fd_);
    control_fd_ = -1;
    return absl::InternalError(absl::StrCat("listen() failed: ", err));
  }

  {
    absl::MutexLock lock(handlers_mu_);
    stopping_ = false;
  }
  control_thread_ = std::thread([this]() { ControlServerLoop(); });
  return bound_port;
}

void TcpControlPlaneBackend::StopServer() {
  {
    absl::MutexLock lock(handlers_mu_);
    stopping_ = true;
  }
  if (control_fd_ >= 0) {
    shutdown(control_fd_, SHUT_RDWR);
  }
  if (control_thread_.joinable()) {
    control_thread_.join();
  }
  if (control_fd_ >= 0) {
    close(control_fd_);
    control_fd_ = -1;
  }
  {
    absl::MutexLock lock(handlers_mu_);
    auto no_active_handlers = [this]()
                                  ABSL_EXCLUSIVE_LOCKS_REQUIRED(handlers_mu_) {
                                    return active_handlers_ == 0;
                                  };
    handlers_mu_.Await(absl::Condition(&no_active_handlers));
  }
}

void TcpControlPlaneBackend::ControlServerLoop() {
  while (true) {
    {
      absl::MutexLock lock(handlers_mu_);
      if (stopping_) break;
    }
    pollfd pfd;
    pfd.fd = control_fd_;
    pfd.events = POLLIN;
    int r = poll(&pfd, 1, 200);
    if (r < 0) {
      if (errno == EINTR) continue;
      break;
    }
    if (r == 0) continue;
    int client_fd = accept(control_fd_, nullptr, nullptr);
    if (client_fd < 0) {
      if (errno == EINTR) continue;
      break;
    }
    if (absl::Status status = SetSocketTimeouts(
            client_fd, absl::ToDoubleSeconds(default_timeout_));
        !status.ok()) {
      LOG(WARNING) << "control connection: " << status.message();
    }

    {
      absl::MutexLock lock(handlers_mu_);
      if (stopping_) {
        close(client_fd);
        break;
      }
      ++active_handlers_;
    }

    auto task = [this, client_fd]() {
      HandleControlConnection(client_fd);
      shutdown(client_fd, SHUT_WR);
      close(client_fd);
      absl::MutexLock lock(handlers_mu_);
      --active_handlers_;
    };

    if (executor_) {
      executor_(std::move(task));
    } else {
      task();
    }
  }
}

void TcpControlPlaneBackend::SendErrorResponse(int fd,
                                               absl::string_view message) {
  ControlResponseHeader response;
  response.magic = kResponseMagic;
  response.status = -1;
  std::string msg(message);
  if (msg.size() > kMaxControlErrorMessageBytes) {
    msg.resize(kMaxControlErrorMessageBytes);
  }
  response.message_len = msg.size();
  // A consumer that has stopped reading backs this write up in the kernel, and
  // the handler holds one of a small pool of workers until it drains. Budgeted
  // from now rather than from a deadline shared with the request read, so that
  // whatever the handler spent deciding on an answer does not leave us unable
  // to send it.
  const absl::Time deadline = absl::Now() + default_timeout_;
  if (absl::Status s = WriteExact(fd, &response, sizeof(response), deadline);
      !s.ok()) {
    LOG(WARNING) << "Failed to send control response header: " << s;
    return;
  }
  if (response.message_len > 0) {
    if (absl::Status s = WriteExact(fd, msg.data(), msg.size(), deadline);
        !s.ok()) {
      LOG(WARNING) << "Failed to send control response message: " << s;
      return;
    }
  }
}

void TcpControlPlaneBackend::HandleControlConnection(
    int fd, ControlPlaneHandler* handler) {
  if (!handler) {
    handler = handler_;
  }
  // The mirror of the client-side bound. This handler occupies one of a small
  // pool of workers and the consumer on the other end decides how fast the
  // request arrives, so a consumer that stalls mid-request would otherwise
  // hold the worker indefinitely and delay every other consumer's requests.
  // Covers the request only; the response is budgeted separately, because the
  // handler between them may legitimately wait out the registration grace.
  const absl::Time request_deadline = absl::Now() + default_timeout_;
  ControlRequestHeader req;
  if (absl::Status s = ReadExact(fd, &req, sizeof(req), request_deadline);
      !s.ok()) {
    SendErrorResponse(fd, s.message());
    return;
  }
  if (req.magic != kControlMagic) {
    SendErrorResponse(fd, "bad control request magic");
    return;
  }
  if (!handler) {
    SendErrorResponse(fd, "ControlPlaneHandler not initialized");
    return;
  }

  if (req.op == kOpAck) {
    absl::Status status = handler->OnAck(req.uuid);
    if (!status.ok()) {
      SendErrorResponse(fd, status.message());
      return;
    }
    ControlResponseHeader response;
    response.magic = kResponseMagic;
    response.status = 0;
    (void)WriteExact(fd, &response, sizeof(response),
                     absl::Now() + default_timeout_);
    return;
  }

  if (req.op == kOpPullStream) {
    const uint64_t max_blocks = handler->MaxPullStreamBlocks();
    if (req.num_blocks > max_blocks) {
      SendErrorResponse(
          fd, absl::StrCat("pull stream block count ", req.num_blocks,
                           " exceeds configured maximum ", max_blocks));
      return;
    }

    absl::StatusOr<std::vector<int64_t>> src_blocks =
        ReadBlockIds(fd, req.num_blocks, request_deadline);
    if (!src_blocks.ok()) {
      SendErrorResponse(fd, src_blocks.status().message());
      return;
    }
    absl::StatusOr<std::vector<int64_t>> dst_blocks =
        ReadBlockIds(fd, req.num_blocks, request_deadline);
    if (!dst_blocks.ok()) {
      SendErrorResponse(fd, dst_blocks.status().message());
      return;
    }

    PullStreamRequestSpec spec;
    spec.uuid = req.uuid;
    spec.ep_idx = req.ep_idx;
    spec.consumer_data_port = req.consumer_data_port;
    spec.src_block_ids = std::move(*src_blocks);
    spec.dst_block_ids = std::move(*dst_blocks);

    if (req.num_ips > 0) {
      for (uint32_t i = 0;
           i < std::min(req.num_ips, static_cast<uint32_t>(kMaxNics)); ++i) {
        char ip_str[INET6_ADDRSTRLEN];
        bool is_ipv4_mapped = true;
        for (int j = 0; j < 10; ++j) {
          if (req.consumer_ips[i][j] != 0) {
            is_ipv4_mapped = false;
            break;
          }
        }
        if (req.consumer_ips[i][10] != 0xff ||
            req.consumer_ips[i][11] != 0xff) {
          is_ipv4_mapped = false;
        }

        if (is_ipv4_mapped) {
          struct in_addr ipv4_addr;
          std::memcpy(&ipv4_addr, req.consumer_ips[i] + 12, 4);
          if (inet_ntop(AF_INET, &ipv4_addr, ip_str, sizeof(ip_str)) !=
              nullptr) {
            spec.consumer_ips.push_back(ip_str);
          }
        } else {
          if (inet_ntop(AF_INET6, req.consumer_ips[i], ip_str,
                        sizeof(ip_str)) != nullptr) {
            spec.consumer_ips.push_back(ip_str);
          }
        }
      }
    }

    std::string fallback_peer_ip;
    if (spec.consumer_ips.empty()) {
      if (req.num_ips == 0) {
        LOG(WARNING) << "No consumer IPs specified in ControlRequestHeader.";
      }
      absl::StatusOr<std::string> peer_ip = GetPeerIp(fd);
      if (peer_ip.ok()) {
        fallback_peer_ip = *peer_ip;
      }
    }

    absl::StatusOr<PullStreamResponseSpec> result =
        handler->OnPullStream(spec, fallback_peer_ip);
    if (!result.ok()) {
      SendErrorResponse(fd, result.status().message());
      return;
    }
    if (result->status != 0) {
      SendErrorResponse(fd, result->message);
      return;
    }

    ControlResponseHeader response;
    response.magic = kResponseMagic;
    response.status = 0;
    response.num_layers = result->num_layers;
    response.data_port = result->data_port;
    response.message_len = 0;
    (void)WriteExact(fd, &response, sizeof(response),
                     absl::Now() + default_timeout_);
    return;
  }

  SendErrorResponse(fd, absl::StrCat("unknown control op code: ", req.op));
}

absl::StatusOr<PullStreamResponseSpec> TcpControlPlaneBackend::SendPullRequest(
    absl::string_view remote_endpoint, const PullStreamRequestSpec& req,
    absl::Duration timeout) {
  // `timeout` bounds the handshake as a whole, not each syscall in it: the
  // caller is holding a worker for exactly this long, whether the peer is
  // silent, slow to accept, or answering one byte at a time. Taken before
  // connect() so the SYN phase spends the same budget as the exchange.
  const absl::Time deadline = absl::Now() + timeout;
  double timeout_s = absl::ToDoubleSeconds(timeout);
  absl::StatusOr<int> fd = ConnectTcp(remote_endpoint, timeout_s);
  if (!fd.ok()) return fd.status();
  absl::Cleanup close_fd = [sock = *fd]() { close(sock); };

  ControlRequestHeader stream_request;
  stream_request.magic = kControlMagic;
  stream_request.op = kOpPullStream;
  stream_request.uuid = req.uuid;
  stream_request.ep_idx = req.ep_idx;
  stream_request.num_blocks = static_cast<uint64_t>(req.src_block_ids.size());
  stream_request.consumer_data_port = req.consumer_data_port;

  stream_request.num_ips = static_cast<uint32_t>(
      std::min(req.consumer_ips.size(), static_cast<size_t>(kMaxNics)));
  for (size_t i = 0; i < stream_request.num_ips; ++i) {
    if (!EncodeIpToIpv6Bytes(req.consumer_ips[i],
                             stream_request.consumer_ips[i])) {
      std::memset(stream_request.consumer_ips[i], 0, 16);
    }
  }

  if (absl::Status s =
          WriteExact(*fd, &stream_request, sizeof(stream_request), deadline);
      !s.ok()) {
    return s;
  }
  if (absl::Status s = WriteBlockIds(*fd, req.src_block_ids, deadline);
      !s.ok()) {
    return s;
  }
  if (absl::Status s = WriteBlockIds(*fd, req.dst_block_ids, deadline);
      !s.ok()) {
    return s;
  }

  ControlResponseHeader resp_hdr;
  if (absl::Status s = ReadExact(*fd, &resp_hdr, sizeof(resp_hdr), deadline);
      !s.ok()) {
    return s;
  }
  if (resp_hdr.magic != kResponseMagic) {
    return absl::InternalError(
        absl::StrCat("bad control response magic: ", resp_hdr.magic));
  }

  std::string message;
  if (resp_hdr.message_len > 0) {
    const bool truncated = resp_hdr.message_len > kMaxControlErrorMessageBytes;
    const size_t message_bytes = static_cast<size_t>(
        std::min(resp_hdr.message_len, kMaxControlErrorMessageBytes));
    message.resize(message_bytes);
    if (absl::Status s =
            ReadExact(*fd, message.data(), message.size(), deadline);
        !s.ok()) {
      return s;
    }
    if (truncated) {
      absl::StrAppend(&message, " [truncated; peer advertised ",
                      resp_hdr.message_len, " bytes]");
    }
  }

  return PullStreamResponseSpec{
      .status = resp_hdr.status,
      .num_layers = resp_hdr.num_layers,
      .data_port = resp_hdr.data_port,
      .message = std::move(message),
  };
}

absl::Status TcpControlPlaneBackend::SendAck(absl::string_view remote_endpoint,
                                             uint64_t uuid,
                                             absl::Duration timeout) {
  const absl::Time deadline = absl::Now() + timeout;
  double timeout_s = absl::ToDoubleSeconds(timeout);
  absl::StatusOr<int> fd = ConnectTcp(remote_endpoint, timeout_s);
  if (!fd.ok()) return fd.status();
  absl::Cleanup close_fd = [sock = *fd]() { close(sock); };

  ControlRequestHeader stream_request;
  stream_request.magic = kControlMagic;
  stream_request.op = kOpAck;
  stream_request.uuid = uuid;
  stream_request.ep_idx = 0;
  stream_request.num_blocks = 0;

  if (absl::Status s =
          WriteExact(*fd, &stream_request, sizeof(stream_request), deadline);
      !s.ok()) {
    return s;
  }

  ControlResponseHeader resp_hdr;
  if (absl::Status s = ReadExact(*fd, &resp_hdr, sizeof(resp_hdr), deadline);
      !s.ok()) {
    return s;
  }
  if (resp_hdr.magic != kResponseMagic) {
    return absl::InternalError(
        absl::StrCat("bad control response magic: ", resp_hdr.magic));
  }

  std::string message;
  if (resp_hdr.message_len > 0) {
    const bool truncated = resp_hdr.message_len > kMaxControlErrorMessageBytes;
    const size_t message_bytes = static_cast<size_t>(
        std::min(resp_hdr.message_len, kMaxControlErrorMessageBytes));
    message.resize(message_bytes);
    if (absl::Status s =
            ReadExact(*fd, message.data(), message.size(), deadline);
        !s.ok()) {
      return s;
    }
    if (truncated) {
      absl::StrAppend(&message, " [truncated; peer advertised ",
                      resp_hdr.message_len, " bytes]");
    }
  }

  if (resp_hdr.status != 0) {
    return absl::InternalError(absl::StrCat("Ack rejected by peer: ", message));
  }
  return absl::OkStatus();
}

}  // namespace tpu_raiden
