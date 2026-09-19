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

#include "tpu_sync/common/control_pipe/tcp_control_pipe.h"

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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <limits>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tpu_sync/common/control_pipe/control_dispatcher.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/proto/control_pipe.pb.h"
#include "tpu_sync/proto/kv_cache_control_plane_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace {

constexpr char kCpipMagic[4] = {'C', 'P', 'I', 'P'};
constexpr char kPipcMagic[4] = {'P', 'I', 'P', 'C'};
constexpr uint64_t kMaxLegacyErrorMessageBytes = 4 * 1024;

absl::Status SetSocketTimeouts(int fd, double timeout_s) {
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

absl::Status WriteExact(int fd, const void* buf, size_t len) {
  const char* p = static_cast<const char*>(buf);
  size_t written = 0;
  while (written < len) {
    ssize_t n = send(fd, p + written, len - written, MSG_NOSIGNAL);
    if (n <= 0) {
      if (n < 0 && errno == EINTR) continue;
      if (n < 0 &&
          (errno == EAGAIN || errno == EWOULDBLOCK || errno == ETIMEDOUT)) {
        return absl::DeadlineExceededError(
            absl::StrCat("socket send timeout: ", std::strerror(errno)));
      }
      return absl::UnavailableError(
          absl::StrCat("socket send failed: ", std::strerror(errno)));
    }
    written += static_cast<size_t>(n);
  }
  return absl::OkStatus();
}

absl::Status ReadExact(int fd, void* buf, size_t len) {
  char* p = static_cast<char*>(buf);
  size_t read_bytes = 0;
  while (read_bytes < len) {
    ssize_t n = recv(fd, p + read_bytes, len - read_bytes, 0);
    if (n == 0) {
      return absl::UnavailableError("socket closed by peer (EOF)");
    }
    if (n < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ETIMEDOUT) {
        return absl::DeadlineExceededError(
            absl::StrCat("socket recv timeout: ", std::strerror(errno)));
      }
      return absl::UnavailableError(
          absl::StrCat("socket recv failed: ", std::strerror(errno)));
    }
    read_bytes += static_cast<size_t>(n);
  }
  return absl::OkStatus();
}

absl::Status ReadExactOrEof(int fd, void* buf, size_t len, bool* is_eof) {
  *is_eof = false;
  char* p = static_cast<char*>(buf);
  size_t read_bytes = 0;
  while (read_bytes < len) {
    ssize_t n = recv(fd, p + read_bytes, len - read_bytes, 0);
    if (n == 0) {
      if (read_bytes == 0) {
        *is_eof = true;
        return absl::OkStatus();
      }
      return absl::UnavailableError("unexpected EOF mid-frame");
    }
    if (n < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ETIMEDOUT) {
        return absl::DeadlineExceededError(
            absl::StrCat("socket recv timeout: ", std::strerror(errno)));
      }
      return absl::UnavailableError(
          absl::StrCat("socket recv failed: ", std::strerror(errno)));
    }
    read_bytes += static_cast<size_t>(n);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::pair<std::string, int>> SplitEndpoint(
    absl::string_view endpoint) {
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

absl::StatusOr<int> ConnectSocket(absl::string_view endpoint,
                                  absl::Duration timeout) {
  ABSL_ASSIGN_OR_RETURN(auto host_port, SplitEndpoint(endpoint));
  const std::string& host = host_port.first;
  int port = host_port.second;

  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* result = nullptr;
  std::string port_str = absl::StrCat(port);
  int rc = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result);
  if (rc != 0 || result == nullptr) {
    return absl::UnavailableError(absl::StrCat(
        "getaddrinfo failed for ", endpoint, ": ", gai_strerror(rc)));
  }
  auto cleanup_addr = absl::MakeCleanup([result]() { freeaddrinfo(result); });

  int fd = -1;
  for (addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
    fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd < 0) continue;

    int opt = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    (void)SetSocketTimeouts(fd, absl::ToDoubleSeconds(timeout));

    if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) {
      return fd;
    }
    close(fd);
    fd = -1;
  }
  return absl::UnavailableError(
      absl::StrCat("connect failed to ", endpoint, ": ", std::strerror(errno)));
}

std::string ExtractPeerIp(const sockaddr_storage& peer_addr) {
  char ip_str[INET6_ADDRSTRLEN] = {0};
  if (peer_addr.ss_family == AF_INET) {
    const auto* sin = reinterpret_cast<const sockaddr_in*>(&peer_addr);
    inet_ntop(AF_INET, &sin->sin_addr, ip_str, sizeof(ip_str));
    return std::string(ip_str);
  }
  if (peer_addr.ss_family == AF_INET6) {
    const auto* sin6 = reinterpret_cast<const sockaddr_in6*>(&peer_addr);
    const uint8_t* bytes = sin6->sin6_addr.s6_addr;
    bool is_v4_mapped = true;
    for (int i = 0; i < 10; ++i) {
      if (bytes[i] != 0) {
        is_v4_mapped = false;
        break;
      }
    }
    if (bytes[10] != 0xff || bytes[11] != 0xff) {
      is_v4_mapped = false;
    }
    if (is_v4_mapped) {
      in_addr v4_addr;
      std::memcpy(&v4_addr, bytes + 12, 4);
      inet_ntop(AF_INET, &v4_addr, ip_str, sizeof(ip_str));
    } else {
      inet_ntop(AF_INET6, &sin6->sin6_addr, ip_str, sizeof(ip_str));
    }
    return std::string(ip_str);
  }
  return "127.0.0.1";
}

absl::Status SendCpipResponse(
    int fd, const control_pipe::proto::ControlResponseEnvelope& resp_env) {
  std::string resp_bytes;
  if (!resp_env.SerializeToString(&resp_bytes)) {
    return absl::InternalError("Failed to serialize ControlResponseEnvelope");
  }
  uint32_t net_len = htonl(static_cast<uint32_t>(resp_bytes.size()));
  ABSL_RETURN_IF_ERROR(WriteExact(fd, kPipcMagic, 4));
  ABSL_RETURN_IF_ERROR(WriteExact(fd, &net_len, sizeof(net_len)));
  if (!resp_bytes.empty()) {
    ABSL_RETURN_IF_ERROR(WriteExact(fd, resp_bytes.data(), resp_bytes.size()));
  }
  return absl::OkStatus();
}

void SendCpipError(int fd, uint64_t request_id, absl::StatusCode code,
                   absl::string_view message) {
  control_pipe::proto::ControlResponseEnvelope resp_env;
  resp_env.set_request_id(request_id);
  resp_env.set_status_code(static_cast<int32_t>(code));
  resp_env.set_error_message(std::string(message));
  (void)SendCpipResponse(fd, resp_env);
}

}  // namespace

// =============================================================================
// TcpConnectionPool
// =============================================================================

TcpConnectionPool::TcpConnectionPool(bool enable_pooling,
                                     size_t max_idle_per_endpoint)
    : enable_pooling_(enable_pooling),
      max_idle_per_endpoint_(max_idle_per_endpoint) {}

TcpConnectionPool::~TcpConnectionPool() {
  absl::MutexLock lock(mu_);
  for (auto& [endpoint, fd_deque] : pool_) {
    for (int fd : fd_deque) {
      close(fd);
    }
  }
  pool_.clear();
}

absl::StatusOr<int> TcpConnectionPool::Acquire(absl::string_view endpoint,
                                               absl::Duration timeout) {
  if (enable_pooling_) {
    absl::MutexLock lock(mu_);
    auto it = pool_.find(endpoint);
    if (it != pool_.end()) {
      auto& deque = it->second;
      while (!deque.empty()) {
        int fd = deque.front();
        deque.pop_front();

        pollfd pfd{};
        pfd.fd = fd;
        pfd.events = POLLIN;
        int r = poll(&pfd, 1, 0);
        if (r == 0 &&
            (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL)) == 0) {
          (void)SetSocketTimeouts(fd, absl::ToDoubleSeconds(timeout));
          return fd;
        }
        close(fd);
      }
    }
  }
  return ConnectSocket(endpoint, timeout);
}

void TcpConnectionPool::Release(absl::string_view endpoint, int fd) {
  if (!enable_pooling_ || max_idle_per_endpoint_ == 0) {
    close(fd);
    return;
  }
  absl::MutexLock lock(mu_);
  auto& deque = pool_[std::string(endpoint)];
  if (deque.size() >= max_idle_per_endpoint_) {
    close(fd);
    return;
  }
  deque.push_back(fd);
}

size_t TcpConnectionPool::IdleCount(absl::string_view endpoint) const {
  absl::MutexLock lock(mu_);
  auto it = pool_.find(endpoint);
  if (it == pool_.end()) return 0;
  return it->second.size();
}

// =============================================================================
// TcpControlPipeServer
// =============================================================================

TcpControlPipeServer::TcpControlPipeServer(const ControlPipeConfig& config)
    : config_(config) {}

TcpControlPipeServer::~TcpControlPipeServer() { Stop(); }

absl::StatusOr<int> TcpControlPipeServer::Start(int requested_port) {
  {
    absl::MutexLock lock(mu_);
    if (server_fd_ >= 0) {
      return absl::FailedPreconditionError(
          "TcpControlPipeServer already running");
    }
    stopping_ = false;
  }

  int fd = socket(AF_INET6, SOCK_STREAM, 0);
  if (fd < 0) {
    return absl::InternalError(
        absl::StrCat("socket(AF_INET6) failed: ", std::strerror(errno)));
  }

  int opt = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

  int ipv6only = 0;
  if (setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &ipv6only, sizeof(ipv6only)) <
      0) {
    LOG(WARNING) << "setsockopt IPV6_V6ONLY=0 failed: " << std::strerror(errno);
  }

  sockaddr_in6 addr{};
  addr.sin6_family = AF_INET6;
  addr.sin6_addr = in6addr_any;
  addr.sin6_port = htons(static_cast<uint16_t>(requested_port));

  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::string err = std::strerror(errno);
    close(fd);
    return absl::InternalError(
        absl::StrCat("bind(", requested_port, ") failed: ", err));
  }

  socklen_t len = sizeof(addr);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) < 0) {
    std::string err = std::strerror(errno);
    close(fd);
    return absl::InternalError(absl::StrCat("getsockname() failed: ", err));
  }
  int bound_port = ntohs(addr.sin6_port);

  if (listen(fd, 128) < 0) {
    std::string err = std::strerror(errno);
    close(fd);
    return absl::InternalError(absl::StrCat("listen() failed: ", err));
  }

  {
    absl::MutexLock lock(mu_);
    server_fd_ = fd;
    bound_port_ = bound_port;
  }

  accept_thread_ = std::thread([this]() { AcceptLoop(); });
  return bound_port;
}

void TcpControlPipeServer::Stop() {
  int server_fd_to_close = -1;
  {
    absl::MutexLock lock(mu_);
    stopping_ = true;
    if (server_fd_ >= 0) {
      shutdown(server_fd_, SHUT_RDWR);
      server_fd_to_close = server_fd_;
    }
    for (int client_fd : active_client_fds_) {
      shutdown(client_fd, SHUT_RDWR);
    }
  }

  if (accept_thread_.joinable()) {
    accept_thread_.join();
  }

  if (server_fd_to_close >= 0) {
    absl::MutexLock lock(mu_);
    if (server_fd_ >= 0) {
      close(server_fd_);
      server_fd_ = -1;
    }
  }

  absl::MutexLock lock(mu_);
  auto all_handlers_done = [this]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return active_handlers_ == 0;
  };
  mu_.Await(absl::Condition(&all_handlers_done));
}

void TcpControlPipeServer::StopAccepting() {
  absl::MutexLock lock(mu_);
  if (server_fd_ >= 0) {
    shutdown(server_fd_, SHUT_RDWR);
    close(server_fd_);
    server_fd_ = -1;
  }
}

void TcpControlPipeServer::AcceptLoop() {
  while (true) {
    int current_server_fd = -1;
    {
      absl::MutexLock lock(mu_);
      if (stopping_ || server_fd_ < 0) break;
      current_server_fd = server_fd_;
    }

    pollfd pfd{};
    pfd.fd = current_server_fd;
    pfd.events = POLLIN;
    int r = poll(&pfd, 1, 200);
    if (r < 0) {
      if (errno == EINTR) continue;
      break;
    }
    if (r == 0) continue;

    sockaddr_storage peer_addr{};
    socklen_t peer_len = sizeof(peer_addr);
    int client_fd = accept(current_server_fd,
                           reinterpret_cast<sockaddr*>(&peer_addr), &peer_len);
    if (client_fd < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
      break;
    }

    int opt = 1;
    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    (void)SetSocketTimeouts(client_fd,
                            absl::ToDoubleSeconds(config_.default_timeout));

    std::string peer_ip = ExtractPeerIp(peer_addr);

    {
      absl::MutexLock lock(mu_);
      if (stopping_) {
        close(client_fd);
        break;
      }
      ++active_handlers_;
      active_client_fds_.insert(client_fd);
    }

    auto task = [this, client_fd, peer_ip = std::move(peer_ip)]() mutable {
      auto cleanup = absl::MakeCleanup([this, client_fd]() {
        {
          absl::MutexLock lock(mu_);
          active_client_fds_.erase(client_fd);
          --active_handlers_;
        }
        close(client_fd);
      });
      HandleConnection(client_fd, std::move(peer_ip));
    };

    if (config_.executor) {
      config_.executor(std::move(task));
    } else {
      std::thread(std::move(task)).detach();
    }
  }
}

void TcpControlPipeServer::HandleConnection(int client_fd,
                                            std::string peer_ip) {
  while (true) {
    {
      absl::MutexLock lock(mu_);
      if (stopping_) break;
    }

    uint32_t raw_word = 0;
    bool is_eof = false;
    absl::Status read_status =
        ReadExactOrEof(client_fd, &raw_word, sizeof(raw_word), &is_eof);
    if (!read_status.ok() || is_eof) {
      break;
    }

    uint32_t be_word = ntohl(raw_word);

    if (std::memcmp(&raw_word, kCpipMagic, 4) == 0) {
      // Mode 1: ControlPipe Envelope Framing ("CPIP")
      uint32_t net_len = 0;
      if (!ReadExact(client_fd, &net_len, sizeof(net_len)).ok()) {
        break;
      }
      uint32_t payload_len = ntohl(net_len);
      if (payload_len == 0 || payload_len > config_.max_frame_bytes) {
        LOG(WARNING) << "Rejected oversized/empty CPIP frame (" << payload_len
                     << " bytes, max=" << config_.max_frame_bytes << ") from "
                     << peer_ip;
        SendCpipError(client_fd, 0, absl::StatusCode::kResourceExhausted,
                      "Frame exceeds max_frame_bytes");
        break;
      }

      std::string env_bytes(payload_len, '\0');
      if (!ReadExact(client_fd, env_bytes.data(), payload_len).ok()) {
        break;
      }

      control_pipe::proto::ControlEnvelope env;
      if (!env.ParseFromString(env_bytes)) {
        SendCpipError(client_fd, 0, absl::StatusCode::kInvalidArgument,
                      "Failed to parse ControlEnvelope");
        break;
      }

      ControlContext ctx;
      ctx.peer_ip = peer_ip;
      ctx.request_id = env.request_id();
      ctx.backend_type = ControlPipeBackendType::kTcp;
      for (const auto& [k, v] : env.metadata()) {
        ctx.metadata[k] = v;
      }

      control_pipe::proto::ControlResponseEnvelope resp_env =
          dispatcher_.Dispatch(ctx, env);
      if (!SendCpipResponse(client_fd, resp_env).ok()) {
        break;
      }
      // Mode 1 stays open in loop for TcpConnectionPool reuse.
    } else if (config_.allow_legacy_framing && (raw_word == kRaidControlMagic ||
                                                be_word == kRaidControlMagic)) {
      // Mode 2: Legacy Binary P2P Struct ("RAID")
      HandleLegacyRaidConnection(client_fd, raw_word, peer_ip);
      // Ephemeral single request/response: close immediately.
      shutdown(client_fd, SHUT_WR);
      break;
    } else if (config_.allow_legacy_framing && be_word > 0 &&
               be_word <= config_.max_frame_bytes) {
      // Mode 3: Legacy 4B Big-Endian Length-Prefixed ControlRequest
      uint32_t payload_len = be_word;
      std::string req_bytes(payload_len, '\0');
      if (!ReadExact(client_fd, req_bytes.data(), payload_len).ok()) {
        break;
      }

      control_pipe::proto::ControlEnvelope env;
      env.set_message_type(
          ::tpu_sync::rpc::ControlRequest::descriptor()->full_name());
      env.set_payload(std::move(req_bytes));

      ControlContext ctx;
      ctx.peer_ip = peer_ip;
      ctx.backend_type = ControlPipeBackendType::kTcp;

      control_pipe::proto::ControlResponseEnvelope resp_env =
          dispatcher_.Dispatch(ctx, env);

      std::string out_bytes;
      if (resp_env.status_code() == 0) {
        out_bytes = resp_env.payload();
      } else {
        ::tpu_sync::rpc::ControlResponse err_resp;
        err_resp.set_success(false);
        err_resp.set_message(resp_env.error_message());
        err_resp.SerializeToString(&out_bytes);
      }

      uint32_t resp_net_len = htonl(static_cast<uint32_t>(out_bytes.size()));
      if (!WriteExact(client_fd, &resp_net_len, sizeof(resp_net_len)).ok()) {
        break;
      }
      if (!out_bytes.empty()) {
        if (!WriteExact(client_fd, out_bytes.data(), out_bytes.size()).ok()) {
          break;
        }
      }
      // Mode 3 loops until client EOF.
    } else {
      LOG(WARNING) << "Rejected malformed/oversized frame (be_word=" << be_word
                   << ", max=" << config_.max_frame_bytes << ") from "
                   << peer_ip;
      break;
    }
  }
}

void TcpControlPipeServer::SendLegacyRaidErrorResponse(
    int client_fd, absl::string_view message) {
  LegacyRaidResponseHeader resp_hdr;
  resp_hdr.magic = kRaidResponseMagic;
  resp_hdr.status = -1;
  std::string msg(message);
  if (msg.size() > kMaxLegacyErrorMessageBytes) {
    msg.resize(kMaxLegacyErrorMessageBytes);
  }
  resp_hdr.message_len = msg.size();
  if (!WriteExact(client_fd, &resp_hdr, sizeof(resp_hdr)).ok()) {
    return;
  }
  if (resp_hdr.message_len > 0) {
    (void)WriteExact(client_fd, msg.data(), msg.size());
  }
}

void TcpControlPipeServer::HandleLegacyRaidConnection(
    int client_fd, uint32_t magic_word, const std::string& peer_ip) {
  LegacyRaidRequestHeader req_hdr;
  req_hdr.magic = magic_word;
  char* rem_ptr = reinterpret_cast<char*>(&req_hdr) + sizeof(uint32_t);
  size_t rem_len = sizeof(LegacyRaidRequestHeader) - sizeof(uint32_t);
  if (absl::Status s = ReadExact(client_fd, rem_ptr, rem_len); !s.ok()) {
    SendLegacyRaidErrorResponse(client_fd, s.message());
    return;
  }

  ControlContext ctx;
  ctx.peer_ip = peer_ip;
  ctx.backend_type = ControlPipeBackendType::kTcp;

  if (req_hdr.op == kRaidOpAck) {
    control_plane::proto::AckRequest ack_req;
    ack_req.set_uuid(req_hdr.uuid);

    control_pipe::proto::ControlEnvelope env;
    env.set_message_type(
        control_plane::proto::AckRequest::descriptor()->full_name());
    if (!ack_req.SerializeToString(env.mutable_payload())) {
      SendLegacyRaidErrorResponse(client_fd, "Failed to serialize AckRequest");
      return;
    }

    control_pipe::proto::ControlResponseEnvelope resp_env =
        dispatcher_.Dispatch(ctx, env);
    if (resp_env.status_code() != 0) {
      SendLegacyRaidErrorResponse(client_fd, resp_env.error_message());
      return;
    }

    LegacyRaidResponseHeader resp_hdr;
    resp_hdr.magic = kRaidResponseMagic;
    resp_hdr.status = 0;
    (void)WriteExact(client_fd, &resp_hdr, sizeof(resp_hdr));
    return;
  }

  if (req_hdr.op == kRaidOpPullStream) {
    uint64_t max_blocks = config_.max_legacy_pull_blocks_fn
                              ? config_.max_legacy_pull_blocks_fn()
                              : std::numeric_limits<uint64_t>::max();
    if (req_hdr.num_blocks > max_blocks ||
        req_hdr.num_blocks > config_.max_frame_bytes / (2 * sizeof(int64_t))) {
      SendLegacyRaidErrorResponse(
          client_fd,
          absl::StrCat("pull stream block count ", req_hdr.num_blocks,
                       " exceeds configured maximum ", max_blocks));
      return;
    }

    std::vector<int64_t> src_blocks(req_hdr.num_blocks);
    std::vector<int64_t> dst_blocks(req_hdr.num_blocks);
    size_t blocks_byte_len = req_hdr.num_blocks * sizeof(int64_t);
    if (blocks_byte_len > 0) {
      if (absl::Status s =
              ReadExact(client_fd, src_blocks.data(), blocks_byte_len);
          !s.ok()) {
        SendLegacyRaidErrorResponse(client_fd, s.message());
        return;
      }
      if (absl::Status s =
              ReadExact(client_fd, dst_blocks.data(), blocks_byte_len);
          !s.ok()) {
        SendLegacyRaidErrorResponse(client_fd, s.message());
        return;
      }
    }

    control_plane::proto::PullStreamRequest pull_req;
    pull_req.set_uuid(req_hdr.uuid);
    pull_req.set_ep_idx(req_hdr.ep_idx);
    pull_req.set_consumer_data_port(req_hdr.consumer_data_port);
    for (int64_t b : src_blocks) pull_req.add_src_block_ids(b);
    for (int64_t b : dst_blocks) pull_req.add_dst_block_ids(b);

    uint32_t num_ips =
        std::min(req_hdr.num_ips, static_cast<uint32_t>(kMaxNics));
    for (uint32_t i = 0; i < num_ips; ++i) {
      char ip_str[INET6_ADDRSTRLEN] = {0};
      bool is_v4_mapped = true;
      for (int j = 0; j < 10; ++j) {
        if (req_hdr.consumer_ips[i][j] != 0) {
          is_v4_mapped = false;
          break;
        }
      }
      if (req_hdr.consumer_ips[i][10] != 0xff ||
          req_hdr.consumer_ips[i][11] != 0xff) {
        is_v4_mapped = false;
      }
      if (is_v4_mapped) {
        in_addr ipv4_addr;
        std::memcpy(&ipv4_addr, req_hdr.consumer_ips[i] + 12, 4);
        if (inet_ntop(AF_INET, &ipv4_addr, ip_str, sizeof(ip_str)) != nullptr) {
          pull_req.add_consumer_ips(ip_str);
        }
      } else {
        if (inet_ntop(AF_INET6, req_hdr.consumer_ips[i], ip_str,
                      sizeof(ip_str)) != nullptr) {
          pull_req.add_consumer_ips(ip_str);
        }
      }
    }

    control_pipe::proto::ControlEnvelope env;
    env.set_message_type(
        control_plane::proto::PullStreamRequest::descriptor()->full_name());
    if (!pull_req.SerializeToString(env.mutable_payload())) {
      SendLegacyRaidErrorResponse(client_fd,
                                  "Failed to serialize PullStreamRequest");
      return;
    }

    control_pipe::proto::ControlResponseEnvelope resp_env =
        dispatcher_.Dispatch(ctx, env);
    if (resp_env.status_code() != 0) {
      SendLegacyRaidErrorResponse(client_fd, resp_env.error_message());
      return;
    }

    control_plane::proto::PullStreamResponse pull_resp;
    if (!pull_resp.ParseFromString(resp_env.payload())) {
      SendLegacyRaidErrorResponse(client_fd,
                                  "Failed to parse PullStreamResponse");
      return;
    }

    LegacyRaidResponseHeader resp_hdr;
    resp_hdr.magic = kRaidResponseMagic;
    resp_hdr.status = pull_resp.status();
    resp_hdr.num_layers = pull_resp.num_layers();
    resp_hdr.data_port = pull_resp.data_port();
    resp_hdr.message_len = pull_resp.message().size();
    if (!WriteExact(client_fd, &resp_hdr, sizeof(resp_hdr)).ok()) {
      return;
    }
    if (resp_hdr.message_len > 0) {
      (void)WriteExact(client_fd, pull_resp.message().data(),
                       pull_resp.message().size());
    }
    return;
  }

  SendLegacyRaidErrorResponse(
      client_fd, absl::StrCat("Unsupported RAID op: ", req_hdr.op));
}

// =============================================================================
// TcpControlPipeClient
// =============================================================================

TcpControlPipeClient::TcpControlPipeClient(const ControlPipeConfig& config)
    : config_(config),
      conn_pool_(std::make_unique<TcpConnectionPool>(
          config.enable_tcp_connection_pooling,
          config.max_idle_connections_per_endpoint)) {}

absl::StatusOr<control_pipe::proto::ControlResponseEnvelope>
TcpControlPipeClient::SendRaw(
    absl::string_view endpoint,
    const control_pipe::proto::ControlEnvelope& envelope,
    absl::Duration timeout) {
  absl::Duration effective_timeout =
      timeout > absl::ZeroDuration() ? timeout : config_.default_timeout;

  const bool can_fallback_legacy =
      config_.allow_legacy_framing &&
      envelope.message_type() ==
          ::tpu_sync::rpc::ControlRequest::descriptor()->full_name();

  auto send_legacy_frame =
      [&]() -> absl::StatusOr<control_pipe::proto::ControlResponseEnvelope> {
    ABSL_ASSIGN_OR_RETURN(int fd, ConnectSocket(endpoint, effective_timeout));
    auto fd_closer = absl::MakeCleanup([fd]() { close(fd); });

    uint32_t net_len = htonl(static_cast<uint32_t>(envelope.payload().size()));
    ABSL_RETURN_IF_ERROR(WriteExact(fd, &net_len, sizeof(net_len)));
    if (!envelope.payload().empty()) {
      ABSL_RETURN_IF_ERROR(
          WriteExact(fd, envelope.payload().data(), envelope.payload().size()));
    }
    shutdown(fd, SHUT_WR);

    uint32_t resp_net_len = 0;
    ABSL_RETURN_IF_ERROR(ReadExact(fd, &resp_net_len, sizeof(resp_net_len)));
    uint32_t resp_len = ntohl(resp_net_len);
    if (resp_len > config_.max_frame_bytes) {
      return absl::ResourceExhaustedError(absl::StrCat(
          "Response frame size (", resp_len, ") exceeds max_frame_bytes (",
          config_.max_frame_bytes, ")"));
    }

    control_pipe::proto::ControlResponseEnvelope resp_env;
    resp_env.set_request_id(envelope.request_id());
    resp_env.set_status_code(0);
    std::string* resp_payload = resp_env.mutable_payload();
    resp_payload->resize(resp_len);
    if (resp_len > 0) {
      ABSL_RETURN_IF_ERROR(ReadExact(fd, resp_payload->data(), resp_len));
    }
    return resp_env;
  };

  bool is_verified_cpip = false;
  if (can_fallback_legacy) {
    bool is_known_legacy = false;
    {
      absl::MutexLock lock(legacy_mu_);
      is_known_legacy = legacy_endpoints_.contains(endpoint);
      is_verified_cpip = verified_cpip_endpoints_.contains(endpoint);
    }
    if (is_known_legacy) {
      return send_legacy_frame();
    }
  }

  const bool probe_with_shut_wr = can_fallback_legacy && !is_verified_cpip;

  auto try_cpip =
      [&]() -> absl::StatusOr<control_pipe::proto::ControlResponseEnvelope> {
    int fd = -1;
    if (probe_with_shut_wr) {
      ABSL_ASSIGN_OR_RETURN(fd, ConnectSocket(endpoint, effective_timeout));
    } else {
      ABSL_ASSIGN_OR_RETURN(fd,
                            conn_pool_->Acquire(endpoint, effective_timeout));
    }
    auto fd_closer = absl::MakeCleanup([fd]() { close(fd); });

    std::string env_bytes;
    if (!envelope.SerializeToString(&env_bytes)) {
      return absl::InternalError("Failed to serialize ControlEnvelope");
    }

    uint32_t net_len = htonl(static_cast<uint32_t>(env_bytes.size()));
    ABSL_RETURN_IF_ERROR(WriteExact(fd, kCpipMagic, 4));
    ABSL_RETURN_IF_ERROR(WriteExact(fd, &net_len, sizeof(net_len)));
    if (!env_bytes.empty()) {
      ABSL_RETURN_IF_ERROR(WriteExact(fd, env_bytes.data(), env_bytes.size()));
    }
    if (probe_with_shut_wr) {
      shutdown(fd, SHUT_WR);
    }

    char resp_magic[4] = {0};
    ABSL_RETURN_IF_ERROR(ReadExact(fd, resp_magic, 4));
    if (std::memcmp(resp_magic, kPipcMagic, 4) != 0) {
      return absl::InternalError(
          "Invalid response magic header (expected PIPC)");
    }

    uint32_t resp_net_len = 0;
    ABSL_RETURN_IF_ERROR(ReadExact(fd, &resp_net_len, sizeof(resp_net_len)));
    uint32_t resp_len = ntohl(resp_net_len);
    if (resp_len > config_.max_frame_bytes) {
      return absl::ResourceExhaustedError(absl::StrCat(
          "Response frame size (", resp_len, ") exceeds max_frame_bytes (",
          config_.max_frame_bytes, ")"));
    }

    std::string resp_bytes(resp_len, '\0');
    if (resp_len > 0) {
      ABSL_RETURN_IF_ERROR(ReadExact(fd, resp_bytes.data(), resp_len));
    }

    control_pipe::proto::ControlResponseEnvelope resp_env;
    if (!resp_env.ParseFromString(resp_bytes)) {
      return absl::InternalError("Failed to parse ControlResponseEnvelope");
    }

    if (!probe_with_shut_wr) {
      std::move(fd_closer).Cancel();
      conn_pool_->Release(endpoint, fd);
    }
    return resp_env;
  };

  absl::StatusOr<control_pipe::proto::ControlResponseEnvelope> cpip_res =
      try_cpip();
  if (cpip_res.ok()) {
    if (probe_with_shut_wr) {
      absl::MutexLock lock(legacy_mu_);
      verified_cpip_endpoints_.insert(std::string(endpoint));
    }
    return cpip_res;
  }
  if (can_fallback_legacy) {
    {
      absl::MutexLock lock(legacy_mu_);
      legacy_endpoints_.insert(std::string(endpoint));
    }
    return send_legacy_frame();
  }
  return cpip_res;
}

}  // namespace tpu_raiden
