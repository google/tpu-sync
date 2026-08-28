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

#include "tpu_sync/transport/lib/raw_buffer_transport.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/core/numa_thread_pool.h"
#include "tpu_sync/transport/buffer_push_task.h"
#include "tpu_sync/transport/lib/transport_adapter.h"

#ifndef IOV_MAX
#define IOV_MAX 1024
#endif
#include "tpu_sync/core/status_macros.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/lib/chunk_serializer.h"
#include "tpu_sync/transport/lib/conn/pool.h"
#include "tpu_sync/transport/lib/raw_buffer_transport_delegate.h"
#include "tpu_sync/transport/lib/socket/tcp_psp_helper.h"
#include "tpu_sync/transport/peregrine/src/api/socket_util.h"

namespace tpu_raiden::transport::lib {

using ::peregrine::ReadExact;
using ::peregrine::ReadVExact;
using ::peregrine::WriteExact;
using ::peregrine::WriteVExact;

namespace {

constexpr int kMaxPushThreads = 16;

absl::Status InternalError(absl::string_view msg, int _errno) {
  return absl::InternalError(absl::StrCat(msg, ": ", std::strerror(_errno)));
}

absl::StatusOr<std::pair<int, int>> CreateTcpIPv4Socket(const int port) {
  const int fd = socket(AF_INET, SOCK_STREAM, /*protocol=*/0);
  if (fd < 0) {
    const int last_errno = errno;
    return InternalError("tcp/ipv4 create socket", last_errno);
  }
  int opt = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    const int last_errno = errno;
    close(fd);
    return InternalError("tcp/ipv4 setsockopt SO_REUSEADDR", last_errno);
  }
  struct sockaddr_in sa;
  socklen_t len = sizeof(sa);
  std::memset(&sa, 0, sizeof(sa));
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = INADDR_ANY;
  sa.sin_port = htons(port);
  if (bind(fd, reinterpret_cast<struct sockaddr*>(&sa), sizeof(sa)) < 0) {
    const int last_errno = errno;
    close(fd);
    return InternalError("tcp/ipv4 bind", last_errno);
  }
  if (getsockname(fd, reinterpret_cast<struct sockaddr*>(&sa), &len) < 0) {
    const int last_errno = errno;
    close(fd);
    return InternalError("tcp/ipv4 getsockname", last_errno);
  }
  DCHECK_GE(fd, 0);
  return std::make_pair(fd, ntohs(sa.sin_port));
}

absl::StatusOr<std::pair<int, int>> CreateTcpIPv6Socket(const int port) {
  const int fd = socket(AF_INET6, SOCK_STREAM, /*protocol=*/0);
  if (fd < 0) {
    const int last_errno = errno;
    return InternalError("tcp/ipv6 create socket", last_errno);
  }
  int opt = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    const int last_errno = errno;
    close(fd);
    return InternalError("tcp/ipv6 setsockopt SO_REUSEADDR", last_errno);
  }
  int v6only = 0;
  if (setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &v6only, sizeof(v6only)) < 0) {
    LOG(WARNING) << "Failed to set IPV6_V6ONLY=0: " << std::strerror(errno);
  }
  struct sockaddr_in6 sa;
  socklen_t len = sizeof(sa);
  std::memset(&sa, 0, sizeof(sa));
  sa.sin6_family = AF_INET6;
  sa.sin6_addr = in6addr_any;
  sa.sin6_port = htons(port);
  if (bind(fd, reinterpret_cast<struct sockaddr*>(&sa), sizeof(sa)) < 0) {
    const int last_errno = errno;
    close(fd);
    return InternalError("tcp/ipv6 bind", last_errno);
  }
  if (getsockname(fd, reinterpret_cast<struct sockaddr*>(&sa), &len) < 0) {
    const int last_errno = errno;
    close(fd);
    return InternalError("tcp/ipv6 getsockname", last_errno);
  }
  DCHECK_GE(fd, 0);
  return std::make_pair(fd, ntohs(sa.sin6_port));
}

absl::StatusOr<std::pair<int, int>> CreateSocket(const int port) {
  const auto fd_port = CreateTcpIPv6Socket(port);
  return fd_port.ok() ? fd_port : CreateTcpIPv4Socket(port);
}

inline bool IsValidSocket(int fd) { return fcntl(fd, F_GETFD) >= 0; }
}  // namespace

RawBufferTransport::RawBufferTransport(
    RawBufferTransportDelegate* delegate, int local_port,
    const std::vector<std::string>& local_ips,
    CustomRequestHandler custom_request_handler, size_t coalesce_window_bytes)
    : raw_delegate_(delegate),
      custom_request_handler_(std::move(custom_request_handler)),
      coalesce_window_bytes_(coalesce_window_bytes),
      bound_ip_(local_ips.empty() ? "127.0.0.1" : local_ips[0]),
      local_ips_(local_ips),
      local_port_(local_port),
      require_psp_tcp_(absl::GetFlag(FLAGS_require_psp_tcp)),
      server_fd_(-1),
      stopping_(false) {
  // 1. Setup server listening socket.
  const absl::StatusOr<std::pair<int, int>> fd_port = CreateSocket(local_port_);
  if (!fd_port.ok()) {
    throw std::runtime_error(
        absl::StrCat("Failed to create server tcp listening socket: ",
                     fd_port.status().message()));
  }

  server_fd_ = fd_port->first;
  local_port_ = fd_port->second;
  DCHECK_GE(server_fd_, 0);
  DCHECK(1 <= local_port_ && local_port_ <= 65535);
  DCHECK(IsValidSocket(server_fd_));

  if (listen(server_fd_, 128) < 0) {
    LOG(FATAL) << "Failed to listen on server socket: " << std::strerror(errno);
  }

  // 2. Start listener
  listener_thread_ = std::thread(&RawBufferTransport::ListenerLoop, this);
}

RawBufferTransport::~RawBufferTransport() {
  stopping_ = true;

  // 1. Listener side:
  // 1.1 Shutdown on all active sockets to unblock the threads.
  if (server_fd_ >= 0) {
    DCHECK(IsValidSocket(server_fd_));
    shutdown(server_fd_, SHUT_RDWR);
  }
  {
    absl::MutexLock _(mu_);
    for (int fd : active_client_fds_) {
      shutdown(fd, SHUT_RDWR);
    }
  }
  // 1.2 Join all threads.
  if (listener_thread_.joinable()) {
    listener_thread_.join();
  }
  for (auto& t : worker_threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
  {
    // Each worker thread should have closed its own client_fd.
    absl::MutexLock _(mu_);
    DCHECK(active_client_fds_.empty());
  }

  // 2. Connector side:
  // Close all pooled file descriptors _after_ all the threads are joined.
  conn_pool_.Close();
}

absl::Status RawBufferTransport::ProcessPeerRequest(int client_fd) {
  char header_buf[kChunkHeaderSize];
  RETURN_IF_ERROR(ReadExact(client_fd, header_buf, sizeof(header_buf)));
  ASSIGN_OR_RETURN(const ChunkHeader header,
                   DeserializeChunkHeader(header_buf));

  if ABSL_PREDICT_FALSE (header.op == kOpBufferPull) {  // peer pull request
    const uint32_t src_offset = header.remote_id;
    const uint32_t src_shard_idx = header.local_id;
    const uint32_t size_bytes = header.count_or_size;
    const uint16_t buf_id = header.buffer_id;

    uint8_t* const base_host_ptr =
        raw_delegate_->GetHostPointer(buf_id, src_shard_idx);
    const size_t host_size = raw_delegate_->GetHostSize(buf_id, src_shard_idx);
    if (base_host_ptr == nullptr || src_offset + size_bytes > host_size) {
      return absl::InvalidArgumentError("Source out of bounds");
    }
    uint8_t* const src_ptr = base_host_ptr + src_offset;
    RETURN_IF_ERROR(WriteExact(client_fd, src_ptr, size_bytes));
    return absl::OkStatus();

  } else if (header.op == kOpBufferPush) {  // peer push request
    const uint32_t dst_offset = header.remote_id;
    const uint32_t dst_shard_idx = header.local_id;
    const uint32_t size_bytes = header.count_or_size;
    const uint16_t buf_id = header.buffer_id;

    uint8_t* const base_host_ptr =
        raw_delegate_->GetHostPointer(buf_id, dst_shard_idx);
    const size_t host_size = raw_delegate_->GetHostSize(buf_id, dst_shard_idx);
    if (base_host_ptr == nullptr || dst_offset + size_bytes > host_size) {
      return absl::InvalidArgumentError("Destination out of bounds");
    }
    uint8_t* const dest_ptr = base_host_ptr + dst_offset;
    RETURN_IF_ERROR(ReadExact(client_fd, dest_ptr, size_bytes));

    const uint8_t ack = 1;
    RETURN_IF_ERROR(WriteExact(client_fd, &ack, 1));

    bool trigger_h2d = false;
    std::vector<size_t> layers_to_trigger;
    if (header.uuid > 0) {
      absl::MutexLock lock(raw_progress_mu_);
      auto& prog = raw_progress_[header.uuid];
      prog.completed_chunks++;
      prog.completed_chunks_per_layer[buf_id]++;
      auto it = prog.expected_chunks_per_layer.find(buf_id);
      if (it != prog.expected_chunks_per_layer.end() &&
          prog.completed_chunks_per_layer[buf_id] == it->second &&
          !prog.triggered_layers.contains(buf_id)) {
        prog.triggered_layers.insert(buf_id);
        layers_to_trigger.push_back(buf_id);
      }
      VLOG(1) << "Received chunk for uuid=" << header.uuid
              << " shard=" << dst_shard_idx << " offset=" << dst_offset
              << " size=" << size_bytes << " progress=" << prog.completed_chunks
              << "/"
              << (prog.expected_chunks.has_value()
                      ? std::to_string(*prog.expected_chunks)
                      : "unknown");
      if (prog.expected_chunks.has_value() &&
          prog.completed_chunks == *prog.expected_chunks) {
        raw_progress_.erase(header.uuid);
        trigger_h2d = true;
        VLOG(1) << "Triggering H2D for uuid=" << header.uuid;
      }
    }

    for (size_t l : layers_to_trigger) {
      RETURN_IF_ERROR(raw_delegate_->OnLayerDataReceived(l, header.uuid));
    }
    if (trigger_h2d) {
      RETURN_IF_ERROR(raw_delegate_->OnDataReceived(header.uuid));
    }

    return absl::OkStatus();

  } else if (header.op == kOpBufferPushBatched) {  // peer batched push request
    const uint32_t batch_size = header.count_or_size;
    if (batch_size > IOV_MAX) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Batch size ", batch_size, " exceeds IOV_MAX (", IOV_MAX, ")"));
    }
    const size_t m_size = GetChunkMetadataSize(header.version);
    std::vector<char> meta_buf(m_size * batch_size);
    RETURN_IF_ERROR(ReadExact(client_fd, meta_buf.data(), meta_buf.size()));

    std::vector<ChunkMetadata> metadata(batch_size);
    for (uint32_t i = 0; i < batch_size; ++i) {
      absl::Span<const char> item_bytes(meta_buf.data() + i * m_size, m_size);
      ASSIGN_OR_RETURN(metadata[i],
                       DeserializeChunkMetadata(item_bytes, header.version));
    }

    std::vector<struct iovec> iovs;
    iovs.reserve(batch_size);
    size_t total_bytes = 0;

    for (uint32_t i = 0; i < batch_size; ++i) {
      const auto& meta = metadata[i];
      uint8_t* const base_host_ptr =
          raw_delegate_->GetHostPointer(meta.layer_idx, meta.dst_shard_idx);
      const size_t host_size =
          raw_delegate_->GetHostSize(meta.layer_idx, meta.dst_shard_idx);
      if (base_host_ptr == nullptr ||
          meta.dst_offset_bytes + meta.size_bytes > host_size) {
        return absl::InvalidArgumentError(
            "Destination out of bounds in batched push");
      }
      if (meta.size_bytes > 0) {
        struct iovec iov;
        iov.iov_base = base_host_ptr + meta.dst_offset_bytes;
        iov.iov_len = meta.size_bytes;
        iovs.push_back(iov);
        total_bytes += meta.size_bytes;
      }
    }

    if (total_bytes > 0) {
      RETURN_IF_ERROR(ReadVExact(client_fd, iovs));
    }

    const uint8_t ack = 1;
    RETURN_IF_ERROR(WriteExact(client_fd, &ack, 1));

    bool trigger_h2d = false;
    std::vector<size_t> layers_to_trigger;
    if (header.uuid > 0) {
      absl::MutexLock lock(raw_progress_mu_);
      auto& prog = raw_progress_[header.uuid];
      prog.completed_chunks += batch_size;
      for (uint32_t i = 0; i < batch_size; ++i) {
        size_t l = metadata[i].layer_idx;
        prog.completed_chunks_per_layer[l]++;
        auto it = prog.expected_chunks_per_layer.find(l);
        if (it != prog.expected_chunks_per_layer.end() &&
            prog.completed_chunks_per_layer[l] == it->second &&
            !prog.triggered_layers.contains(l)) {
          prog.triggered_layers.insert(l);
          layers_to_trigger.push_back(l);
        }
      }
      VLOG(1) << "Received batched chunks for uuid=" << header.uuid
              << " batch_size=" << batch_size
              << " progress=" << prog.completed_chunks << "/"
              << (prog.expected_chunks.has_value()
                      ? std::to_string(*prog.expected_chunks)
                      : "unknown");
      if (prog.expected_chunks.has_value() &&
          prog.completed_chunks == *prog.expected_chunks) {
        raw_progress_.erase(header.uuid);
        trigger_h2d = true;
        VLOG(1) << "Triggering H2D for uuid=" << header.uuid;
      }
    }

    for (size_t l : layers_to_trigger) {
      RETURN_IF_ERROR(raw_delegate_->OnLayerDataReceived(l, header.uuid));
    }
    if (trigger_h2d) {
      RETURN_IF_ERROR(raw_delegate_->OnDataReceived(header.uuid));
    }

    return absl::OkStatus();
  } else {
    if (custom_request_handler_) {
      return custom_request_handler_(client_fd, header);
    }
    return absl::UnimplementedError(
        absl::StrCat("Unsupported raw transport op code: ", header.op));
  }
}

void RawBufferTransport::ConnectionWorker(int client_fd) {
  DCHECK_GE(client_fd, 0);
  while (!stopping_) {
    struct pollfd pfd;
    pfd.fd = client_fd;
    pfd.events = POLLIN;
    int ret = poll(&pfd, 1, 50);
    if (ret < 0) {
      // EINTR (interrupted by a signal) and EAGAIN (transient kernel resource
      // pressure) are benign: retry the poll rather than tearing down a healthy
      // connection. Only a genuine error closes the connection.
      if (errno == EINTR || errno == EAGAIN) continue;
      break;
    }
    if (ret == 0) continue;

    if (absl::Status status = ProcessPeerRequest(client_fd); !status.ok()) {
      LOG(ERROR) << "ProcessPeerRequest failed: " << status;
      break;
    }
  }

  DCHECK_GE(client_fd, 0);
  {
    absl::MutexLock _( mu_ );
    active_client_fds_.erase(client_fd);
  }
  close(client_fd);
}

absl::StatusOr<PspPeerKey>
RawBufferTransport::RegisterPspPeer(uint32_t client_spi,
                                    absl::string_view client_key) {
  if (!require_psp_tcp_) {
    return absl::InvalidArgumentError(
        "PSP is not enabled in transport");
  }
  absl::MutexLock lock(psp_mu_);
  if (stopping_ || server_fd_ < 0) {
    return absl::FailedPreconditionError(
        "Transport is stopping or listening socket is not initialized.");
  }

  return RegisterPspPeerKey(server_fd_, client_spi, client_key);
}

void RawBufferTransport::ListenerLoop() {
  while (!stopping_) {
    DCHECK(IsValidSocket(server_fd_));
    struct pollfd pfd;
    pfd.fd = server_fd_;
    pfd.events = POLLIN;
    int ret = poll(&pfd, 1, 50);
    if (ret <= 0) {
      if (stopping_) break;
      continue;
    }

    struct sockaddr_in6 client_addr;
    socklen_t clilen = sizeof(client_addr);
    int client_fd = accept(
        server_fd_, reinterpret_cast<struct sockaddr*>(&client_addr), &clilen);
    if (client_fd < 0) {
      if (stopping_) break;
      continue;
    }
    if (require_psp_tcp_ && !PspEnabled(client_fd)) {
      close(client_fd);
      LOG_EVERY_N_SEC(ERROR, 1)
          << "Unencrypted TCP connection rejected on PSP listener";
      continue;
    }

    int opt = 1;
    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    int buf_opt = 16 * 1024 * 1024;  // 16MB
    setsockopt(client_fd, SOL_SOCKET, SO_SNDBUF, &buf_opt, sizeof(buf_opt));
    setsockopt(client_fd, SOL_SOCKET, SO_RCVBUF, &buf_opt, sizeof(buf_opt));

    DCHECK_GE(client_fd, 0);
    {
      absl::MutexLock _( mu_ );
      active_client_fds_.insert(client_fd);
    }
    worker_threads_.push_back(
        std::thread([this, client_fd]() { ConnectionWorker(client_fd); }));
  }

  DCHECK(IsValidSocket(server_fd_));
  close(server_fd_);
  server_fd_ = -1;
  DCHECK(!IsValidSocket(server_fd_));
}

absl::Status RawBufferTransport::PullBuffer(
    absl::string_view peer, size_t buffer_id, size_t src_shard_idx,
    size_t src_offset_bytes, size_t dst_shard_idx, size_t dst_offset_bytes,
    size_t size_bytes) {
  if (peer.empty()) {
    return absl::InvalidArgumentError("Source peer address cannot be empty");
  }

  const size_t host_size = raw_delegate_->GetHostSize(buffer_id, dst_shard_idx);
  if (dst_offset_bytes + size_bytes > host_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Destination offset out of bounds. Offset: ", dst_offset_bytes,
        ", Size: ", size_bytes, ", Shard Host Size: ", host_size));
  }

  uint8_t* dest_ptr = raw_delegate_->GetHostPointer(buffer_id, dst_shard_idx) +
                      dst_offset_bytes;
  ASSIGN_OR_RETURN(const Request req,
                   BuildBufferRequest(buffer_id, src_shard_idx,
                                      src_offset_bytes, dest_ptr, size_bytes,
                                      /*uuid=*/0, kOpBufferPull));
  return ProcessSocketBufferPull(peer, req);
}

absl::Status RawBufferTransport::ProcessSocketBufferPull(
    absl::string_view peer, const Request& request) {
  if (peer.empty()) {
    return absl::InvalidArgumentError("Source peer address cannot be empty");
  }

  ASSIGN_OR_RETURN(const int fd, BorrowConnection(peer));
  bool ok_to_pool = false;
  auto fd_cleaner =
      absl::MakeCleanup([&] { ReturnConnection(ok_to_pool, fd, peer); });

  ChunkHeader header = {};
  header.version = 1;
  header.op = kOpBufferPull;
  header.buffer_id = static_cast<uint16_t>(request.layer_idx);
  header.remote_id = static_cast<uint32_t>(request.remote_id);
  header.local_id = request.local_id;
  header.count_or_size = static_cast<uint32_t>(request.len);

  const auto s_header = SerializeChunkHeader(header);
  RETURN_IF_ERROR(WriteExact(fd, s_header.data(), s_header.size()));

  if (request.len > 0) {
    if (request.laddr == nullptr) {
      return absl::InvalidArgumentError("Destination host pointer is null");
    }
    RETURN_IF_ERROR(ReadExact(fd, request.laddr, request.len));
  }

  ok_to_pool = true;
  return absl::OkStatus();
}

absl::Status RawBufferTransport::RegisterExpectedChunks(
    uint64_t uuid, uint32_t expected_chunks) {
  if (expected_chunks == 0) {
    return absl::InvalidArgumentError("expected_chunks must be positive");
  }

  bool trigger_h2d = false;
  {
    absl::MutexLock lock(raw_progress_mu_);
    auto& prog = raw_progress_[uuid];
    prog.expected_chunks = expected_chunks;
    VLOG(1) << "RegisterExpectedChunks: uuid=" << uuid
            << " expected_chunks=" << expected_chunks
            << " completed_chunks=" << prog.completed_chunks;
    if (prog.completed_chunks == expected_chunks) {
      raw_progress_.erase(uuid);
      trigger_h2d = true;
      VLOG(1) << "RegisterExpectedChunks triggering H2D for uuid=" << uuid;
    }
  }

  if (trigger_h2d) {
    RETURN_IF_ERROR(raw_delegate_->OnDataReceived(uuid));
  }

  return absl::OkStatus();
}

absl::Status RawBufferTransport::RegisterExpectedLayerChunks(
    uint64_t uuid,
    const absl::flat_hash_map<size_t, uint32_t>& expected_layer_chunks) {
  if (expected_layer_chunks.empty()) {
    return absl::OkStatus();
  }

  std::vector<size_t> layers_to_trigger;
  {
    absl::MutexLock lock(raw_progress_mu_);
    auto& prog = raw_progress_[uuid];
    prog.expected_chunks_per_layer = expected_layer_chunks;
    for (const auto& [layer_idx, expected_count] : expected_layer_chunks) {
      auto it = prog.completed_chunks_per_layer.find(layer_idx);
      if (it != prog.completed_chunks_per_layer.end() &&
          it->second == expected_count &&
          !prog.triggered_layers.contains(layer_idx)) {
        prog.triggered_layers.insert(layer_idx);
        layers_to_trigger.push_back(layer_idx);
      }
    }
  }

  for (size_t layer_idx : layers_to_trigger) {
    RETURN_IF_ERROR(raw_delegate_->OnLayerDataReceived(layer_idx, uuid));
  }

  return absl::OkStatus();
}

absl::Status RawBufferTransport::ProcessSocketBufferPush(
    absl::string_view peer, const Request& request) {
  if (peer.empty()) {
    return absl::InvalidArgumentError(
        "Destination peer address cannot be empty");
  }

  ASSIGN_OR_RETURN(const int fd, BorrowConnection(peer));
  bool ok_to_pool = false;
  auto fd_cleaner =
      absl::MakeCleanup([&] { ReturnConnection(ok_to_pool, fd, peer); });

  const uint8_t opcode = request.socket_opcode;
  const uint64_t uuid = request.uuid;

  if (opcode != kOpBufferPush) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported buffer push opcode: ", opcode));
  }

  if (request.remote_id > std::numeric_limits<uint32_t>::max()) {
    return absl::InvalidArgumentError("Destination offset exceeds uint32 max");
  }

  ChunkHeader header = {};
  header.version = 1;
  header.op = kOpBufferPush;
  header.buffer_id = static_cast<uint16_t>(request.layer_idx);
  header.remote_id = static_cast<uint32_t>(request.remote_id);
  header.local_id = request.local_id;
  header.count_or_size = static_cast<uint32_t>(request.len);
  header.uuid = uuid;

  VLOG(1) << "Pushing chunk to peer=" << peer << " uuid=" << uuid
          << " dst_shard=" << request.local_id
          << " dst_offset=" << request.remote_id << " size=" << request.len;

  const auto s_header = SerializeChunkHeader(header);
  const std::array<struct iovec, 2> iovs = {
      iovec(const_cast<char*>(s_header.data()), s_header.size()),
      iovec(request.laddr, request.len),
  };
  RETURN_IF_ERROR(WriteVExact(fd, iovs));

  uint8_t ack = 0;
  RETURN_IF_ERROR(ReadExact(fd, &ack, 1));
  if (ack != 1) {
    return absl::InternalError("PushBuffer verification failed");
  }

  ok_to_pool = true;
  return absl::OkStatus();
}

absl::StatusOr<Request> BuildBufferRequest(size_t buffer_id, size_t shard_idx,
                                           size_t offset_bytes,
                                           const uint8_t* data_ptr,
                                           size_t size_bytes, uint64_t uuid,
                                           uint8_t socket_opcode) {
  return Request{
      .socket_opcode = socket_opcode,
      .laddr = const_cast<uint8_t*>(data_ptr),
      .raddr = nullptr,
      .len = size_bytes,
      .major_order = 0,
      .layer_idx = static_cast<int>(buffer_id),
      .parallelism = 1,
      .remote_id = offset_bytes,
      .local_id = static_cast<uint32_t>(shard_idx),
      .count_or_size = static_cast<uint32_t>(size_bytes),
      .uuid = uuid,
      .request_id = 0,
  };
}

absl::StatusOr<std::vector<Request>> BuildBufferRequests(
    absl::Span<const BufferPushTask> tasks, uint64_t uuid,
    uint8_t socket_opcode) {
  std::vector<Request> requests;
  requests.reserve(tasks.size());
  for (const auto& task : tasks) {
    ASSIGN_OR_RETURN(auto req,
                     BuildBufferRequest(task.buffer_id, task.dst_shard_idx,
                                        task.dst_offset_bytes, task.data_ptr,
                                        task.size_bytes, uuid, socket_opcode));
    requests.push_back(std::move(req));
  }
  return requests;
}

absl::Status RawBufferTransport::PushBuffers(
    const std::vector<BufferPushTask>& tasks, int parallelism, uint64_t uuid) {
  std::vector<BufferPushTask> grouped_tasks = tasks;
  std::stable_sort(grouped_tasks.begin(), grouped_tasks.end(),
                   [](const BufferPushTask& a, const BufferPushTask& b) {
                     return a.peer < b.peer;
                   });

  struct BatchInfo {
    std::string peer;
    size_t start_idx;
    size_t count;
  };

  std::vector<BatchInfo> batches;
  for (size_t i = 0; i < grouped_tasks.size();) {
    const std::string peer = grouped_tasks[i].peer;
    size_t peer_start = i;
    size_t peer_task_count = 0;
    while (i < grouped_tasks.size() && grouped_tasks[i].peer == peer) {
      peer_task_count++;
      i++;
    }

    const size_t p = std::max(1, parallelism);
    const size_t target_batch_size = (peer_task_count + p - 1) / p;
    const size_t base_batch_size =
        std::clamp(target_batch_size, size_t{1}, static_cast<size_t>(IOV_MAX));

    size_t tasks_processed = 0;
    while (tasks_processed < peer_task_count) {
      const size_t chunk_size =
          std::min(base_batch_size, peer_task_count - tasks_processed);
      const size_t chunk_start = peer_start + tasks_processed;

      if (coalesce_window_bytes_ > 0) {
        // Coalescing enabled: split this chunk into sub-batches that fit in
        // `coalesce_window_bytes_`.
        size_t sub_tasks_added = 0;
        while (sub_tasks_added < chunk_size) {
          size_t sub_batch_bytes = 0;
          size_t sub_batch_count = 0;
          for (size_t j = sub_tasks_added; j < chunk_size; ++j) {
            const size_t task_size = grouped_tasks[chunk_start + j].size_bytes;
            if (sub_batch_count > 0 &&
                sub_batch_bytes + task_size > coalesce_window_bytes_) {
              break;
            }
            sub_batch_bytes += task_size;
            sub_batch_count++;
          }
          batches.push_back(
              {peer, chunk_start + sub_tasks_added, sub_batch_count});
          sub_tasks_added += sub_batch_count;
        }
      } else {
        // Coalescing disabled: keep the chunk as one batch fits in IOV_MAX.
        batches.push_back({peer, chunk_start, chunk_size});
      }
      tasks_processed += chunk_size;
    }
  }

  if (batches.empty()) {
    return absl::OkStatus();
  }

  auto push_batch = [&](const BatchInfo& batch) -> absl::Status {
    ASSIGN_OR_RETURN(
        std::vector<Request> requests,
        BuildBufferRequests(absl::MakeConstSpan(grouped_tasks)
                                .subspan(batch.start_idx, batch.count),
                            uuid, kOpBufferPushBatched));
    return ProcessSocketBufferBatchPush(batch.peer, requests);
  };

  if (parallelism <= 1 || batches.size() == 1) {
    for (const auto& batch : batches) {
      RETURN_IF_ERROR(push_batch(batch));
    }
    return absl::OkStatus();
  }

  tpu_raiden::NumaThreadPool* pool = nullptr;
  {
    absl::MutexLock lock(push_pool_mu_);
    if (!push_pool_) {
      push_pool_ =
          std::make_unique<tpu_raiden::NumaThreadPool>(kMaxPushThreads);
    }
    pool = push_pool_.get();
  }

  absl::BlockingCounter counter(batches.size());
  std::vector<absl::Status> statuses(batches.size(), absl::OkStatus());
  for (size_t i = 0; i < batches.size(); ++i) {
    pool->Schedule(
        [&push_batch, &batch = batches[i], &counter, &status = statuses[i]]() {
          status = push_batch(batch);
          counter.DecrementCount();
        });
  }

  counter.Wait();

  for (const auto& s : statuses) {
    RETURN_IF_ERROR(s);
  }
  return absl::OkStatus();
}

absl::Status RawBufferTransport::ProcessSocketBufferBatchPush(
    absl::string_view peer, absl::Span<const Request> requests) {
  if (peer.empty()) {
    return absl::InvalidArgumentError(
        "Destination peer address cannot be empty");
  }
  if (requests.empty()) {
    return absl::OkStatus();
  }

  ASSIGN_OR_RETURN(const int fd, BorrowConnection(peer));
  bool ok_to_pool = false;
  auto fd_cleaner =
      absl::MakeCleanup([&] { ReturnConnection(ok_to_pool, fd, peer); });

  const size_t batch_size = requests.size();
  const uint64_t uuid = requests.front().uuid;

  ChunkHeader header = {};
  header.version = 1;
  header.op = kOpBufferPushBatched;
  header.buffer_id = 0;
  header.metadata_size = GetChunkMetadataSize(header.version);
  header.remote_id = 0;
  header.local_id = 0;
  header.count_or_size = static_cast<uint32_t>(batch_size);
  header.uuid = uuid;

  std::vector<char> s_metadata_buf(batch_size * header.metadata_size);
  size_t total_bytes = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    const auto& req = requests[i];
    ChunkMetadata meta = {
        .layer_idx = static_cast<uint32_t>(req.layer_idx),
        .dst_shard_idx = req.local_id,
        .dst_offset_bytes = req.remote_id,
        .size_bytes = req.len,
    };
    const auto s_meta = SerializeChunkMetadata(meta);
    DCHECK_EQ(s_meta.size(), header.metadata_size);
    std::memcpy(s_metadata_buf.data() + i * header.metadata_size, s_meta.data(),
                s_meta.size());
    total_bytes += req.len;
  }
  const auto s_header = SerializeChunkHeader(header);
  const std::array<struct iovec, 2> iovs = {
      iovec(const_cast<char*>(s_header.data()), s_header.size()),
      iovec(s_metadata_buf.data(), s_metadata_buf.size()),
  };
  RETURN_IF_ERROR(WriteVExact(fd, iovs));

  if (coalesce_window_bytes_ > 0) {
    // Coalesced path: pack and write
    std::vector<uint8_t> pack_buf(total_bytes);
    size_t pack_offset = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      const auto& req = requests[i];
      if (req.len > 0) {
        if (req.laddr == nullptr) {
          return absl::InvalidArgumentError(
              "Null data pointer in batch push request");
        }
        std::memcpy(pack_buf.data() + pack_offset, req.laddr, req.len);
        pack_offset += req.len;
      }
    }
    RETURN_IF_ERROR(WriteExact(fd, pack_buf.data(), total_bytes));
  } else {
    // Uncoalesced path: gather write (writev) directly from requests.
    std::vector<struct iovec> iovs;
    iovs.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      const auto& req = requests[i];
      if (req.len > 0) {
        if (req.laddr == nullptr) {
          return absl::InvalidArgumentError(
              "Null data pointer in batch push request");
        }
        struct iovec iov;
        iov.iov_base = req.laddr;
        iov.iov_len = req.len;
        iovs.push_back(iov);
      }
    }
    if (!iovs.empty()) {
      RETURN_IF_ERROR(WriteVExact(fd, iovs));
    }
  }

  uint8_t ack = 0;
  RETURN_IF_ERROR(ReadExact(fd, &ack, 1));
  if (ack != 1) {
    return absl::InternalError(
        "ProcessSocketBufferBatchPush verification failed");
  }

  ok_to_pool = true;
  return absl::OkStatus();
}

void RawBufferTransport::ForgetPushProgress(uint64_t uuid) {
  absl::MutexLock lock(raw_progress_mu_);
  raw_progress_.erase(uuid);
}

}  // namespace tpu_raiden::transport::lib
