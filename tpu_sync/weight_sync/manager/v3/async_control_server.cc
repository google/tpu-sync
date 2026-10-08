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

#include "tpu_sync/weight_sync/manager/v3/async_control_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/proto/control_pipe.pb.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {
namespace {

constexpr char kCpipMagic[4] = {'C', 'P', 'I', 'P'};
constexpr char kPipcMagic[4] = {'P', 'I', 'P', 'C'};
constexpr size_t kHeaderBytes = 8;
constexpr uint64_t kWakeToken = 0;
constexpr uint64_t kListenToken = 1;
constexpr uint64_t kFirstConnectionId = 2;
constexpr size_t kReadChunkBytes = 64 << 10;

int64_t ThreadCpuNs(clockid_t clock) {
  timespec ts{};
  if (clock_gettime(clock, &ts) != 0) return 0;
  return static_cast<int64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

std::string PeerIp(const sockaddr_storage& addr) {
  char buf[INET6_ADDRSTRLEN] = {0};
  if (addr.ss_family == AF_INET6) {
    const auto* a6 = reinterpret_cast<const sockaddr_in6*>(&addr);
    inet_ntop(AF_INET6, &a6->sin6_addr, buf, sizeof(buf));
  } else if (addr.ss_family == AF_INET) {
    const auto* a4 = reinterpret_cast<const sockaddr_in*>(&addr);
    inet_ntop(AF_INET, &a4->sin_addr, buf, sizeof(buf));
  }
  return buf;
}

struct Counters {
  std::atomic<int64_t> accepted{0};
  std::atomic<int32_t> open{0};
  std::atomic<int64_t> requests{0};
  std::atomic<int64_t> responses{0};
};

// The part of a connection that replying threads touch.
struct ConnectionShared {
  explicit ConnectionShared(uint64_t id) : id(id) {}
  const uint64_t id;
  absl::Mutex mu;
  bool closed ABSL_GUARDED_BY(mu) = false;
  // Response frames not yet handed to the loop.
  std::string pending ABSL_GUARDED_BY(mu);
};

// The part of a loop that other threads touch. Replies keep it alive, so a
// late reply never writes to a closed eventfd.
class LoopShared {
 public:
  LoopShared() : event_fd_(eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)) {}
  ~LoopShared() {
    if (event_fd_ >= 0) close(event_fd_);
  }

  int event_fd() const { return event_fd_; }

  // Connection |id| has pending responses.
  void NotifyReady(uint64_t id) {
    bool wake = false;
    {
      absl::MutexLock lock(mu_);
      if (stopped_) return;
      wake = ready_.empty() && incoming_.empty();
      ready_.push_back(id);
    }
    if (wake) Wake();
  }

  // Hands an accepted socket to the loop.
  void AddIncoming(int fd, std::string peer_ip) {
    bool wake = false;
    {
      absl::MutexLock lock(mu_);
      if (stopped_) {
        close(fd);
        return;
      }
      wake = ready_.empty() && incoming_.empty();
      incoming_.emplace_back(fd, std::move(peer_ip));
    }
    if (wake) Wake();
  }

  void Stop() {
    {
      absl::MutexLock lock(mu_);
      stopped_ = true;
    }
    Wake();
  }

  // Called by the loop after it consumed the eventfd.
  bool Take(std::vector<uint64_t>& ready,
            std::vector<std::pair<int, std::string>>& incoming) {
    absl::MutexLock lock(mu_);
    ready.swap(ready_);
    incoming.swap(incoming_);
    return stopped_;
  }

 private:
  void Wake() {
    const uint64_t one = 1;
    // EAGAIN means the counter is already non-zero: the loop wakes anyway.
    (void)!write(event_fd_, &one, sizeof(one));
  }

  const int event_fd_;
  absl::Mutex mu_;
  bool stopped_ ABSL_GUARDED_BY(mu_) = false;
  std::vector<uint64_t> ready_ ABSL_GUARDED_BY(mu_);
  std::vector<std::pair<int, std::string>> incoming_ ABSL_GUARDED_BY(mu_);
};

class EpollLoop {
 public:
  EpollLoop(const EpollControlServerOptions& options,
            const AsyncControlHandler* handler,
            std::shared_ptr<Counters> counters, std::atomic<uint64_t>* next_id)
      : options_(options),
        handler_(handler),
        counters_(std::move(counters)),
        next_id_(next_id),
        shared_(std::make_shared<LoopShared>()),
        epoll_fd_(epoll_create1(EPOLL_CLOEXEC)) {}

  ~EpollLoop() {
    if (thread_.joinable()) thread_.join();
    for (auto& [id, conn] : connections_) CloseFd(*conn);
    if (epoll_fd_ >= 0) close(epoll_fd_);
  }

  absl::Status Init() {
    if (epoll_fd_ < 0 || shared_->event_fd() < 0) {
      return absl::InternalError(
          absl::StrCat("epoll/eventfd setup failed: ", std::strerror(errno)));
    }
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.u64 = kWakeToken;
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, shared_->event_fd(), &ev) != 0) {
      return absl::InternalError(
          absl::StrCat("epoll_ctl(eventfd) failed: ", std::strerror(errno)));
    }
    return absl::OkStatus();
  }

  // Makes this loop accept connections on |listen_fd| and spread them over
  // |loops|.
  absl::Status Listen(int listen_fd, std::vector<EpollLoop*> loops) {
    listen_fd_ = listen_fd;
    accept_targets_ = std::move(loops);
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.u64 = kListenToken;
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, listen_fd, &ev) != 0) {
      return absl::InternalError(
          absl::StrCat("epoll_ctl(listen) failed: ", std::strerror(errno)));
    }
    return absl::OkStatus();
  }

  void Start() {
    thread_ = std::thread([this] { Run(); });
  }

  void Stop() {
    shared_->Stop();
    if (thread_.joinable()) thread_.join();
  }

  LoopShared& shared() { return *shared_; }

  int64_t CpuNs() {
    if (thread_.joinable()) {
      clockid_t clock;
      if (pthread_getcpuclockid(thread_.native_handle(), &clock) == 0) {
        return ThreadCpuNs(clock);
      }
    }
    return final_cpu_ns_.load(std::memory_order_relaxed);
  }

 private:
  struct Connection {
    std::shared_ptr<ConnectionShared> shared;
    int fd = -1;
    std::string peer_ip;
    std::string in;
    size_t in_offset = 0;
    std::string out;
    size_t out_offset = 0;
    bool want_write = false;
  };

  void Run() {
    std::vector<epoll_event> events(256);
    std::vector<uint64_t> ready;
    std::vector<std::pair<int, std::string>> incoming;
    bool stopping = false;
    while (!stopping) {
      const int n = epoll_wait(epoll_fd_, events.data(),
                               static_cast<int>(events.size()), -1);
      if (n < 0) {
        if (errno == EINTR) continue;
        LOG(ERROR) << "epoll_wait failed: " << std::strerror(errno);
        break;
      }
      for (int i = 0; i < n; ++i) {
        const uint64_t token = events[i].data.u64;
        if (token == kWakeToken) {
          uint64_t value = 0;
          (void)!read(shared_->event_fd(), &value, sizeof(value));
          stopping = shared_->Take(ready, incoming);
          for (auto& [fd, peer_ip] : incoming) {
            AddConnection(fd, std::move(peer_ip));
          }
          for (uint64_t id : ready) FlushPending(id);
          ready.clear();
          incoming.clear();
        } else if (token == kListenToken) {
          Accept();
        } else {
          auto it = connections_.find(token);
          if (it == connections_.end()) continue;
          Connection& conn = *it->second;
          if (events[i].events & (EPOLLIN | EPOLLHUP | EPOLLERR)) {
            if (!OnReadable(conn)) {
              CloseConnection(token);
              continue;
            }
          }
          if ((events[i].events & EPOLLOUT) && !Flush(conn)) {
            CloseConnection(token);
          }
        }
      }
    }
    final_cpu_ns_.store(ThreadCpuNs(CLOCK_THREAD_CPUTIME_ID),
                        std::memory_order_relaxed);
  }

  void Accept() {
    while (true) {
      sockaddr_storage addr{};
      socklen_t len = sizeof(addr);
      const int fd = accept4(listen_fd_, reinterpret_cast<sockaddr*>(&addr),
                             &len, SOCK_NONBLOCK | SOCK_CLOEXEC);
      if (fd < 0) {
        if (errno == EINTR) continue;
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
          LOG_EVERY_N_SEC(WARNING, 1)
              << "accept failed: " << std::strerror(errno);
        }
        return;
      }
      const int one = 1;
      setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      counters_->accepted.fetch_add(1, std::memory_order_relaxed);
      EpollLoop* target =
          accept_targets_[next_target_++ % accept_targets_.size()];
      target->shared().AddIncoming(fd, PeerIp(addr));
    }
  }

  void AddConnection(int fd, std::string peer_ip) {
    const uint64_t id = next_id_->fetch_add(1, std::memory_order_relaxed);
    auto conn = std::make_unique<Connection>();
    conn->shared = std::make_shared<ConnectionShared>(id);
    conn->fd = fd;
    conn->peer_ip = std::move(peer_ip);
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.u64 = id;
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) != 0) {
      LOG(WARNING) << "epoll_ctl(connection) failed: " << std::strerror(errno);
      close(fd);
      return;
    }
    counters_->open.fetch_add(1, std::memory_order_relaxed);
    connections_.emplace(id, std::move(conn));
  }

  // Reads what is available and dispatches every complete frame. Returns
  // false if the connection must be closed.
  bool OnReadable(Connection& conn) {
    char buf[kReadChunkBytes];
    bool eof = false;
    while (true) {
      const ssize_t n = read(conn.fd, buf, sizeof(buf));
      if (n > 0) {
        conn.in.append(buf, static_cast<size_t>(n));
        if (static_cast<size_t>(n) < sizeof(buf)) break;
        continue;
      }
      if (n == 0) {
        eof = true;
        break;
      }
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) break;
      return false;
    }
    while (conn.in.size() - conn.in_offset >= kHeaderBytes) {
      const char* header = conn.in.data() + conn.in_offset;
      if (std::memcmp(header, kCpipMagic, sizeof(kCpipMagic)) != 0) {
        LOG_EVERY_N_SEC(WARNING, 1)
            << "Closing connection from " << conn.peer_ip
            << ": not a ControlPipe envelope frame";
        return false;
      }
      uint32_t net_len = 0;
      std::memcpy(&net_len, header + 4, sizeof(net_len));
      const size_t len = ntohl(net_len);
      // Like `TcpControlPipe`, reject empty frames: a request envelope is never
      // empty (it carries at least its `message_type`).
      if (len == 0 || len > options_.max_frame_bytes) {
        LOG_EVERY_N_SEC(WARNING, 1)
            << "Closing connection from " << conn.peer_ip << ": frame of "
            << len << " bytes";
        return false;
      }
      if (conn.in.size() - conn.in_offset < kHeaderBytes + len) break;
      control_pipe::proto::ControlEnvelope envelope;
      const bool parsed = envelope.ParseFromString(
          absl::string_view(header + kHeaderBytes, len));
      conn.in_offset += kHeaderBytes + len;
      if (!parsed) {
        LOG_EVERY_N_SEC(WARNING, 1)
            << "Closing connection from " << conn.peer_ip
            << ": malformed ControlEnvelope";
        return false;
      }
      Dispatch(conn, std::move(envelope));
    }
    if (conn.in_offset == conn.in.size()) {
      conn.in.clear();
      conn.in_offset = 0;
    } else if (conn.in_offset >= kReadChunkBytes) {
      conn.in.erase(0, conn.in_offset);
      conn.in_offset = 0;
    }
    return !eof;
  }

  void Dispatch(Connection& conn, control_pipe::proto::ControlEnvelope env) {
    counters_->requests.fetch_add(1, std::memory_order_relaxed);
    ControlContext ctx;
    ctx.peer_ip = conn.peer_ip;
    ctx.request_id = env.request_id();
    ctx.backend_type = ControlPipeBackendType::kTcp;
    for (const auto& [key, value] : env.metadata()) ctx.metadata[key] = value;
    AsyncControlReply reply =
        [conn_shared = conn.shared, loop = shared_, counters = counters_,
         request_id = env.request_id()](
            control_pipe::proto::ControlResponseEnvelope response) {
          response.set_request_id(request_id);
          std::string bytes;
          if (!response.SerializeToString(&bytes)) {
            LOG(ERROR) << "Failed to serialize a ControlResponseEnvelope";
            return;
          }
          const uint32_t net_len = htonl(static_cast<uint32_t>(bytes.size()));
          bool notify = false;
          {
            absl::MutexLock lock(conn_shared->mu);
            if (conn_shared->closed) return;
            notify = conn_shared->pending.empty();
            conn_shared->pending.append(kPipcMagic, sizeof(kPipcMagic));
            conn_shared->pending.append(reinterpret_cast<const char*>(&net_len),
                                        sizeof(net_len));
            conn_shared->pending.append(bytes);
          }
          counters->responses.fetch_add(1, std::memory_order_relaxed);
          if (notify) loop->NotifyReady(conn_shared->id);
        };
    (*handler_)(ctx, std::move(env), std::move(reply));
  }

  void FlushPending(uint64_t id) {
    auto it = connections_.find(id);
    if (it == connections_.end()) return;
    Connection& conn = *it->second;
    {
      absl::MutexLock lock(conn.shared->mu);
      if (conn.out_offset == conn.out.size()) {
        conn.out.clear();
        conn.out_offset = 0;
      }
      conn.out.append(conn.shared->pending);
      conn.shared->pending.clear();
    }
    if (!Flush(conn)) CloseConnection(id);
  }

  // Writes as much of the output as the socket takes. Returns false if the
  // connection must be closed.
  bool Flush(Connection& conn) {
    while (conn.out_offset < conn.out.size()) {
      const ssize_t n = send(conn.fd, conn.out.data() + conn.out_offset,
                             conn.out.size() - conn.out_offset, MSG_NOSIGNAL);
      if (n > 0) {
        conn.out_offset += static_cast<size_t>(n);
        continue;
      }
      if (n < 0 && errno == EINTR) continue;
      if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
      return false;
    }
    const bool want_write = conn.out_offset < conn.out.size();
    if (!want_write) {
      conn.out.clear();
      conn.out_offset = 0;
    }
    if (want_write != conn.want_write) {
      epoll_event ev{};
      ev.events = EPOLLIN | (want_write ? EPOLLOUT : 0);
      ev.data.u64 = conn.shared->id;
      if (epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, conn.fd, &ev) != 0) return false;
      conn.want_write = want_write;
    }
    return true;
  }

  void CloseFd(Connection& conn) {
    {
      absl::MutexLock lock(conn.shared->mu);
      conn.shared->closed = true;
      conn.shared->pending.clear();
    }
    if (conn.fd >= 0) {
      epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, conn.fd, nullptr);
      close(conn.fd);
      conn.fd = -1;
      counters_->open.fetch_sub(1, std::memory_order_relaxed);
    }
  }

  void CloseConnection(uint64_t id) {
    auto it = connections_.find(id);
    if (it == connections_.end()) return;
    CloseFd(*it->second);
    connections_.erase(it);
  }

  const EpollControlServerOptions options_;
  const AsyncControlHandler* const handler_;
  const std::shared_ptr<Counters> counters_;
  std::atomic<uint64_t>* const next_id_;
  const std::shared_ptr<LoopShared> shared_;
  const int epoll_fd_;
  int listen_fd_ = -1;
  std::vector<EpollLoop*> accept_targets_;
  size_t next_target_ = 0;
  // Owned by the loop thread.
  absl::flat_hash_map<uint64_t, std::unique_ptr<Connection>> connections_;
  std::thread thread_;
  std::atomic<int64_t> final_cpu_ns_{0};
};

class EpollControlServer final : public AsyncControlServer {
 public:
  EpollControlServer(EpollControlServerOptions options,
                     AsyncControlHandler handler)
      : options_(options), handler_(std::move(handler)) {
    if (options_.num_threads < 1) options_.num_threads = 1;
  }

  ~EpollControlServer() override { Stop(); }

  absl::StatusOr<int> Start(int requested_port) override {
    absl::MutexLock lock(mu_);
    if (listen_fd_ >= 0) {
      return absl::FailedPreconditionError(
          "AsyncControlServer already running");
    }
    const int fd =
        socket(AF_INET6, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) {
      return absl::InternalError(
          absl::StrCat("socket(AF_INET6) failed: ", std::strerror(errno)));
    }
    const int one = 1;
    const int zero = 0;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &zero, sizeof(zero));
    sockaddr_in6 addr{};
    addr.sin6_family = AF_INET6;
    addr.sin6_addr = in6addr_any;
    addr.sin6_port = htons(static_cast<uint16_t>(requested_port));
    socklen_t len = sizeof(addr);
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0 ||
        listen(fd, SOMAXCONN) != 0) {
      const std::string error = std::strerror(errno);
      close(fd);
      return absl::InternalError(
          absl::StrCat("Cannot listen on port ", requested_port, ": ", error));
    }
    std::vector<std::unique_ptr<EpollLoop>> loops;
    std::vector<EpollLoop*> targets;
    for (int32_t i = 0; i < options_.num_threads; ++i) {
      loops.push_back(std::make_unique<EpollLoop>(options_, &handler_,
                                                  counters_, &next_id_));
      targets.push_back(loops.back().get());
      absl::Status status = loops.back()->Init();
      if (!status.ok()) {
        close(fd);
        return status;
      }
    }
    absl::Status status = loops[0]->Listen(fd, targets);
    if (!status.ok()) {
      close(fd);
      return status;
    }
    for (auto& loop : loops) loop->Start();
    loops_ = std::move(loops);
    listen_fd_ = fd;
    bound_port_ = ntohs(addr.sin6_port);
    return bound_port_;
  }

  void Stop() override {
    absl::MutexLock lock(mu_);
    if (listen_fd_ < 0) return;
    for (auto& loop : loops_) loop->Stop();
    for (auto& loop : loops_) {
      stopped_cpu_ns_ += loop->CpuNs();
    }
    loops_.clear();
    close(listen_fd_);
    listen_fd_ = -1;
  }

  int bound_port() const override {
    absl::MutexLock lock(mu_);
    return bound_port_;
  }

  Stats GetStats() const override {
    Stats stats;
    stats.accepted_connections =
        counters_->accepted.load(std::memory_order_relaxed);
    stats.open_connections = counters_->open.load(std::memory_order_relaxed);
    stats.requests = counters_->requests.load(std::memory_order_relaxed);
    stats.responses = counters_->responses.load(std::memory_order_relaxed);
    absl::MutexLock lock(mu_);
    stats.loop_cpu_ns = stopped_cpu_ns_;
    for (const auto& loop : loops_) stats.loop_cpu_ns += loop->CpuNs();
    return stats;
  }

 private:
  EpollControlServerOptions options_;
  const AsyncControlHandler handler_;
  const std::shared_ptr<Counters> counters_ = std::make_shared<Counters>();
  std::atomic<uint64_t> next_id_{kFirstConnectionId};
  mutable absl::Mutex mu_;
  int listen_fd_ ABSL_GUARDED_BY(mu_) = -1;
  int bound_port_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t stopped_cpu_ns_ ABSL_GUARDED_BY(mu_) = 0;
  std::vector<std::unique_ptr<EpollLoop>> loops_ ABSL_GUARDED_BY(mu_);
};

}  // namespace

std::unique_ptr<AsyncControlServer> CreateEpollControlServer(
    EpollControlServerOptions options, AsyncControlHandler handler) {
  return std::make_unique<EpollControlServer>(options, std::move(handler));
}

}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden
