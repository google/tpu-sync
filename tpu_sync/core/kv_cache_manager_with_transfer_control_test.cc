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

// Control-plane handshake between a consumer's StartRead and a producer's
// pull handler, exercised over loopback without a device.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"

namespace tpu_raiden {
namespace {

using ::testing::Contains;
using ::testing::HasSubstr;

// Both ends of a control handshake share one pool of four workers, so four
// stuck handshakes are enough to starve either side.
constexpr int kPoolSize = 4;
constexpr double kTimeoutS = 0.5;

class TestManager : public KVCacheManagerWithTransfer {
 public:
  TestManager()
      : KVCacheManagerWithTransfer(
            /*num_layers=*/0, /*num_shards=*/1, /*slice_byte_size=*/128,
            /*local_port=*/std::nullopt,
            /*host_blocks_to_allocate=*/std::nullopt,
            /*parallelism=*/1, /*node_id=*/0,
            /*local_control_port=*/0, /*max_blocks=*/1,
            /*num_slots=*/2 * kPoolSize, kTimeoutS) {}

  using KVCacheManagerWithTransfer::ControlRequestHeader;
  using KVCacheManagerWithTransfer::ControlResponseHeader;
  using KVCacheManagerWithTransfer::kControlMagic;
  using KVCacheManagerWithTransfer::kOpPullStream;
  using KVCacheManagerWithTransfer::kResponseMagic;
};

int Connect(int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  EXPECT_GE(fd, 0);
  // A client that would otherwise wait forever fails the test instead.
  timeval tv{.tv_sec = 10, .tv_usec = 0};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  EXPECT_EQ(connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0)
      << std::strerror(errno);
  return fd;
}

void WriteAll(int fd, const void* data, size_t len) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  while (len > 0) {
    ssize_t n = write(fd, p, len);
    if (n < 0 && errno == EINTR) continue;
    ASSERT_GT(n, 0) << std::strerror(errno);
    p += n;
    len -= n;
  }
}

// Returns false when the peer closed or timed out before `len` bytes came.
bool ReadAll(int fd, void* data, size_t len) {
  uint8_t* p = static_cast<uint8_t*>(data);
  while (len > 0) {
    ssize_t n = read(fd, p, len);
    if (n < 0) {
      if (errno == EINTR) continue;
      LOG(ERROR) << "ReadAll failed: " << std::strerror(errno);
      return false;
    }
    if (n == 0) {
      LOG(ERROR) << "ReadAll EOF, remaining bytes: " << len;
      return false;
    }
    p += n;
    len -= n;
  }
  return true;
}

// Sends a one-block pull for `uuid` the way StartRead does.
void SendPull(int fd, uint64_t uuid) {
  TestManager::ControlRequestHeader req;
  req.magic = TestManager::kControlMagic;
  req.op = TestManager::kOpPullStream;
  req.uuid = uuid;
  req.num_blocks = 1;
  WriteAll(fd, &req, sizeof(req));
  int64_t block = 0;
  WriteAll(fd, &block, sizeof(block));  // producer block ids
  WriteAll(fd, &block, sizeof(block));  // consumer host block ids
}

struct Response {
  bool received = false;
  int32_t status = 0;
  std::string message;
};

Response ReadResponse(int fd) {
  Response out;
  TestManager::ControlResponseHeader hdr;
  if (!ReadAll(fd, &hdr, sizeof(hdr))) return out;
  EXPECT_EQ(hdr.magic, TestManager::kResponseMagic);
  out.received = true;
  out.status = hdr.status;
  out.message.resize(hdr.message_len);
  if (hdr.message_len > 0) {
    EXPECT_TRUE(ReadAll(fd, out.message.data(), out.message.size()));
  }
  return out;
}

double SecondsSince(absl::Time start) {
  return absl::ToDoubleSeconds(absl::Now() - start);
}

TEST(ControlHandshakeTest, PullWithoutRegistrationIsRejected) {
  TestManager producer;
  int fd = Connect(producer.local_control_port());
  const absl::Time start = absl::Now();
  SendPull(fd, /*uuid=*/41);
  Response response = ReadResponse(fd);
  close(fd);

  ASSERT_TRUE(response.received);
  EXPECT_NE(response.status, 0);
  EXPECT_THAT(response.message, HasSubstr("no read registered for uuid 41"));
  // Rejected once the registration grace lapses, not at some later deadline.
  EXPECT_LT(SecondsSince(start), 5.0);
}

TEST(ControlHandshakeTest, PullAheadOfRegistrationIsServedOnceRegistered) {
  TestManager producer;
  int fd = Connect(producer.local_control_port());
  SendPull(fd, /*uuid=*/42);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  producer.NotifyForRead("req42", 42, /*block_ids=*/{0});
  Response response = ReadResponse(fd);
  close(fd);

  ASSERT_TRUE(response.received);
  EXPECT_EQ(response.status, 0);
}

TEST(ControlHandshakeTest, HandlersOutliveConsumersThatNeverSpeak) {
  TestManager producer;
  // Every handler is held by a consumer that connected and went silent.
  std::vector<int> idle;
  for (int i = 0; i < kPoolSize; ++i) {
    idle.push_back(Connect(producer.local_control_port()));
  }
  int fd = Connect(producer.local_control_port());
  const absl::Time start = absl::Now();
  SendPull(fd, /*uuid=*/43);
  Response response = ReadResponse(fd);
  close(fd);
  for (int idle_fd : idle) close(idle_fd);

  // The idle connections are dropped at the transfer timeout and the
  // handlers pick this pull up; it is then rejected within the grace.
  ASSERT_TRUE(response.received);
  EXPECT_NE(response.status, 0);
  EXPECT_LT(SecondsSince(start), 2 * kTimeoutS + 5.0);
}

// A producer that accepts control connections and never answers them.
class SilentProducer {
 public:
  SilentProducer() {
    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    EXPECT_EQ(bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    socklen_t len = sizeof(addr);
    EXPECT_EQ(getsockname(fd_, reinterpret_cast<sockaddr*>(&addr), &len), 0);
    port_ = ntohs(addr.sin_port);
    EXPECT_EQ(listen(fd_, 64), 0);
    thread_ = std::thread([this] {
      while (true) {
        int client = accept(fd_, nullptr, nullptr);
        if (client < 0) return;
        ++accepted_;
        clients_.push_back(client);
      }
    });
  }

  ~SilentProducer() {
    shutdown(fd_, SHUT_RDWR);
    close(fd_);
    thread_.join();
    for (int client : clients_) close(client);
  }

  std::string endpoint() const { return absl::StrCat("127.0.0.1:", port_); }
  int accepted() const { return accepted_.load(); }

 private:
  int fd_ = -1;
  int port_ = 0;
  std::atomic<int> accepted_{0};
  std::vector<int> clients_;
  std::thread thread_;
};

TEST(ControlHandshakeTest, ConsumerGivesUpOnProducerThatNeverAnswers) {
  SilentProducer producer;
  TestManager consumer;
  const absl::Time start = absl::Now();
  // One more read than the consumer has handshake workers.
  const int reads = kPoolSize + 1;
  for (int i = 0; i < reads; ++i) {
    consumer.StartRead(absl::StrCat("req", i), /*uuid=*/100 + i,
                       producer.endpoint(), /*remote_block_ids=*/{0},
                       /*local_block_ids=*/{0});
  }

  // The last read connects only after a worker gives up on its silent
  // producer, which happens at the transfer timeout rather than never.
  while (producer.accepted() < reads && SecondsSince(start) < 10.0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  EXPECT_EQ(producer.accepted(), reads);
  EXPECT_LT(SecondsSince(start), 2 * kTimeoutS + 5.0);

  // Every read settles rather than leaking its receive entry. With no
  // layers to receive, the completion sweep can also count an abandoned
  // read as done, so either report settles it here.
  std::vector<std::string> settled;
  while (settled.size() < static_cast<size_t>(reads) &&
         SecondsSince(start) < 20.0) {
    auto [done_sending, done_recving, failed_recving] =
        consumer.CompleteReadRaw();
    settled.insert(settled.end(), done_recving.begin(), done_recving.end());
    settled.insert(settled.end(), failed_recving.begin(), failed_recving.end());
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  for (int i = 0; i < reads; ++i) {
    EXPECT_THAT(settled, Contains(absl::StrCat("req", i)));
  }
}

class DynamicFailingProducer {
 public:
  enum class State {
    HEALTHY,
    SILENT_FREEZE,
    CRASH_WITH_RST,
  };

  DynamicFailingProducer() {
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    EXPECT_GE(listen_fd_, 0);
    int opt = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    EXPECT_EQ(
        bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    socklen_t len = sizeof(addr);
    EXPECT_EQ(getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len),
              0);
    port_ = ntohs(addr.sin_port);
    EXPECT_EQ(listen(listen_fd_, 64), 0);

    listener_thread_ = std::thread([this] { AcceptLoop(); });
  }

  ~DynamicFailingProducer() { Stop(); }

  void SetState(State state) {
    state_.store(state, std::memory_order_relaxed);
    if (state == State::CRASH_WITH_RST) {
      absl::MutexLock lock(mu_);
      for (int client : clients_) {
        linger sl{.l_onoff = 1, .l_linger = 0};
        setsockopt(client, SOL_SOCKET, SO_LINGER, &sl, sizeof(sl));
        close(client);
      }
      clients_.clear();
      if (listen_fd_ >= 0) {
        shutdown(listen_fd_, SHUT_RDWR);
        close(listen_fd_);
        listen_fd_ = -1;
      }
    }
  }

  void Stop() {
    if (listen_fd_ >= 0) {
      shutdown(listen_fd_, SHUT_RDWR);
      close(listen_fd_);
      listen_fd_ = -1;
    }
    if (listener_thread_.joinable()) {
      listener_thread_.join();
    }
    absl::MutexLock lock(mu_);
    for (int client : clients_) {
      close(client);
    }
    clients_.clear();
    for (auto& t : worker_threads_) {
      if (t.joinable()) t.join();
    }
    worker_threads_.clear();
  }

  std::string endpoint() const { return absl::StrCat("127.0.0.1:", port_); }
  int accepted_count() const { return accepted_.load(); }
  int served_count() const { return served_.load(); }

 private:
  void AcceptLoop() {
    while (true) {
      int client = accept(listen_fd_, nullptr, nullptr);
      if (client < 0) return;
      ++accepted_;
      absl::MutexLock lock(mu_);
      clients_.push_back(client);
      worker_threads_.emplace_back([this, client] { HandleClient(client); });
    }
  }

  void HandleClient(int client_fd) {
    State current = state_.load(std::memory_order_relaxed);
    if (current == State::CRASH_WITH_RST) {
      linger sl{.l_onoff = 1, .l_linger = 0};
      setsockopt(client_fd, SOL_SOCKET, SO_LINGER, &sl, sizeof(sl));
      close(client_fd);
      return;
    }
    if (current == State::SILENT_FREEZE) {
      // Hold connection open without writing anything back
      return;
    }

    // State::HEALTHY: Read request header and return valid
    // ControlResponseHeader
    TestManager::ControlRequestHeader req;
    if (!ReadAll(client_fd, &req, sizeof(req))) return;
    std::vector<int64_t> dummy(req.num_blocks * 2);
    if (!ReadAll(client_fd, dummy.data(), dummy.size() * sizeof(int64_t)))
      return;

    TestManager::ControlResponseHeader resp;
    resp.magic = TestManager::kResponseMagic;
    resp.status = 0;
    resp.message_len = 0;
    WriteAll(client_fd, &resp, sizeof(resp));
    ++served_;
  }

  int listen_fd_ = -1;
  int port_ = 0;
  std::atomic<State> state_{State::HEALTHY};
  std::atomic<int> accepted_{0};
  std::atomic<int> served_{0};
  absl::Mutex mu_;
  std::vector<int> clients_ ABSL_GUARDED_BY(mu_);
  std::vector<std::thread> worker_threads_ ABSL_GUARDED_BY(mu_);
  std::thread listener_thread_;
};

TEST(ControlHandshakeTest,
     CircuitBreakerBansFailingPeerWithExponentialBackoff) {
  DynamicFailingProducer p0;
  DynamicFailingProducer p1;
  TestManager consumer;
  consumer.set_circuit_breaker_enabled(true);
  consumer.set_circuit_breaker_initial_backoff(absl::Seconds(2));

  // Freeze p0.
  p0.SetState(DynamicFailingProducer::State::SILENT_FREEZE);

  // Send 2 requests to p0.
  consumer.StartRead("req_p0_0", /*uuid=*/100, p0.endpoint(),
                     /*remote_block_ids=*/{0}, /*local_block_ids=*/{0});
  consumer.StartRead("req_p0_1", /*uuid=*/101, p0.endpoint(),
                     /*remote_block_ids=*/{0}, /*local_block_ids=*/{0});

  // Complete them (they fail after timeout).
  const absl::Time t_fail = absl::Now();
  int fail_count = 0;
  while (SecondsSince(t_fail) < 5.0) {
    auto [done_sending, done_recving, failed_recving] =
        consumer.CompleteReadRaw();
    fail_count += failed_recving.size();
    if (fail_count >= 2) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_GE(fail_count, 2);

  // Assert EXPECT_TRUE(consumer.IsPeerBanned(p0.endpoint()));
  EXPECT_TRUE(consumer.IsPeerBanned(p0.endpoint()));
  const absl::Time banned_until = consumer.GetPeerBannedUntil(p0.endpoint());
  EXPECT_GT(banned_until, absl::Now());

  // Send a 3rd request to p0. Call consumer.CompleteReadRaw(). Assert that it
  // is in failed_recving immediately (<10ms)!
  const absl::Time t_fast_fail = absl::Now();
  consumer.StartRead("req_p0_2", /*uuid=*/102, p0.endpoint(),
                     /*remote_block_ids=*/{0}, /*local_block_ids=*/{0});
  auto [s_3, r_3, f_3] = consumer.CompleteReadRaw();
  const double fast_fail_ms =
      absl::ToDoubleMilliseconds(absl::Now() - t_fast_fail);
  EXPECT_THAT(f_3, Contains("req_p0_2"));
  EXPECT_LT(fast_fail_ms, 10.0);

  // Send a request to healthy p1. Assert that it completes successfully in
  // <20ms!
  const absl::Time t_healthy = absl::Now();
  consumer.StartRead("req_p1_0", /*uuid=*/200, p1.endpoint(),
                     /*remote_block_ids=*/{0}, /*local_block_ids=*/{0});
  bool p1_done = false;
  while (SecondsSince(t_healthy) < 5.0) {
    auto [s_p1, r_p1, f_p1] = consumer.CompleteReadRaw();
    for (const auto& id : r_p1) {
      if (id == "req_p1_0") p1_done = true;
    }
    if (p1_done) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  const double healthy_ms = absl::ToDoubleMilliseconds(absl::Now() - t_healthy);
  EXPECT_TRUE(p1_done);
  EXPECT_LT(healthy_ms, 20.0);

  // Verify straggler protection: call
  // consumer.RecordCircuitBreakerFailure(p0.endpoint()) while banned, assert
  // consumer.GetPeerBannedUntil(p0.endpoint()) was not modified.
  consumer.RecordCircuitBreakerFailure(p0.endpoint());
  EXPECT_EQ(consumer.GetPeerBannedUntil(p0.endpoint()), banned_until);
}

TEST(ControlHandshakeTest, CircuitBreakerDisabledByDefaultDoesNotBan) {
  DynamicFailingProducer p0;
  TestManager consumer;
  EXPECT_FALSE(consumer.circuit_breaker_enabled());

  p0.SetState(DynamicFailingProducer::State::SILENT_FREEZE);
  consumer.StartRead("req_dis_0", /*uuid=*/300, p0.endpoint(),
                     /*remote_block_ids=*/{0}, /*local_block_ids=*/{0});
  consumer.StartRead("req_dis_1", /*uuid=*/301, p0.endpoint(),
                     /*remote_block_ids=*/{0}, /*local_block_ids=*/{0});

  const absl::Time t_fail = absl::Now();
  int fail_count = 0;
  while (SecondsSince(t_fail) < 5.0) {
    auto [done_sending, done_recving, failed_recving] =
        consumer.CompleteReadRaw();
    fail_count += failed_recving.size();
    if (fail_count >= 2) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_GE(fail_count, 2);

  // Because the circuit breaker is disabled by default, the peer is not banned.
  EXPECT_FALSE(consumer.IsPeerBanned(p0.endpoint()));
  EXPECT_EQ(consumer.GetPeerBannedUntil(p0.endpoint()), absl::InfinitePast());
}

void RunFailingProducerWithRst(bool circuit_breaker_enabled,
                               bool expect_peer_banned,
                               bool expect_starvation) {
  DynamicFailingProducer p0;
  DynamicFailingProducer p1;
  TestManager consumer;
  consumer.set_circuit_breaker_enabled(circuit_breaker_enabled);

  p0.SetState(DynamicFailingProducer::State::CRASH_WITH_RST);

  for (int i = 0; i < kPoolSize; ++i) {
    consumer.StartRead(absl::StrCat("req_crash_", i), /*uuid=*/100 + i,
                       p0.endpoint(), /*remote_block_ids=*/{0},
                       /*local_block_ids=*/{0});
  }

  const absl::Time t_healthy_start = absl::Now();
  consumer.StartRead("req_healthy", /*uuid=*/999, p1.endpoint(),
                     /*remote_block_ids=*/{0}, /*local_block_ids=*/{0});

  while (p1.served_count() == 0 && SecondsSince(t_healthy_start) < 5.0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  const double latency = SecondsSince(t_healthy_start);

  EXPECT_EQ(p1.served_count(), 1);
  if (expect_starvation) {
    EXPECT_GE(latency, 0.4);
  } else {
    // Sockets fail immediately with TCP RST, so worker threads are freed
    // in <1ms without starvation.
    EXPECT_LT(latency, 0.2);
  }
  EXPECT_EQ(consumer.IsPeerBanned(p0.endpoint()), expect_peer_banned);
}

TEST(ControlHandshakeTest, FailingProducerWithRst) {
  RunFailingProducerWithRst(/*circuit_breaker_enabled=*/false,
                            /*expect_peer_banned=*/false,
                            /*expect_starvation=*/false);
  RunFailingProducerWithRst(/*circuit_breaker_enabled=*/true,
                            /*expect_peer_banned=*/true,
                            /*expect_starvation=*/false);
}

void RunFailingProducerWithoutRst(bool circuit_breaker_enabled,
                                  bool expect_peer_banned,
                                  bool expect_starvation) {
  DynamicFailingProducer p0;
  DynamicFailingProducer p1;
  TestManager consumer;
  consumer.set_circuit_breaker_enabled(circuit_breaker_enabled);

  p0.SetState(DynamicFailingProducer::State::SILENT_FREEZE);

  // Dispatch 2 reads to p0 and wait for them to fail.
  for (int i = 0; i < 2; ++i) {
    consumer.StartRead(absl::StrCat("req_silent_init_", i), /*uuid=*/100 + i,
                       p0.endpoint(), /*remote_block_ids=*/{0},
                       /*local_block_ids=*/{0});
  }

  const absl::Time t_fail_start = absl::Now();
  int fail_count = 0;
  while (SecondsSince(t_fail_start) < 5.0) {
    auto [done_sending, done_recving, failed_recving] =
        consumer.CompleteReadRaw();
    fail_count += failed_recving.size();
    if (fail_count >= 2) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_GE(fail_count, 2);

  EXPECT_EQ(consumer.IsPeerBanned(p0.endpoint()), expect_peer_banned);

  // Dispatch 4 reads to p0.
  // When banned, these fast-fail in <1us without occupying threads in
  // push_pool_. When unbanned, these saturate all 4 worker threads in
  // push_pool_.
  for (int i = 0; i < kPoolSize; ++i) {
    consumer.StartRead(absl::StrCat("req_silent_batch_", i), /*uuid=*/200 + i,
                       p0.endpoint(), /*remote_block_ids=*/{0},
                       /*local_block_ids=*/{0});
  }

  if (expect_starvation) {
    // When starvation is expected (peer not banned), wait until p0 has accepted
    // the 4 connections so all push_pool_ threads are occupied.
    while (p0.accepted_count() < 2 + kPoolSize) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }

  // Dispatch read to healthy p1.
  const absl::Time t_healthy_start = absl::Now();
  consumer.StartRead("req_healthy", /*uuid=*/999, p1.endpoint(),
                     /*remote_block_ids=*/{0}, /*local_block_ids=*/{0});

  while (p1.served_count() == 0 && SecondsSince(t_healthy_start) < 5.0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  const double latency = SecondsSince(t_healthy_start);

  EXPECT_EQ(p1.served_count(), 1);
  if (expect_starvation) {
    // Healthy producer is starved in queue for >= 0.4s until a silent request
    // times out (kTimeoutS = 0.5s).
    EXPECT_GE(latency, 0.4);
  } else {
    // Healthy producer is serviced immediately (<0.1s) because circuit breaker
    // fast-failed requests to p0 without holding worker threads.
    EXPECT_LT(latency, 0.1);
  }
}

TEST(ControlHandshakeTest, FailingProducerWithoutRst) {
  RunFailingProducerWithoutRst(/*circuit_breaker_enabled=*/false,
                               /*expect_peer_banned=*/false,
                               /*expect_starvation=*/true);
  RunFailingProducerWithoutRst(/*circuit_breaker_enabled=*/true,
                               /*expect_peer_banned=*/true,
                               /*expect_starvation=*/false);
}

TEST(ControlHandshakeTest,
     ConcurrentWorkerFailuresInOpenCircuitIgnoredWhileClosedCircuitCompounds) {
  TestManager consumer;
  consumer.set_circuit_breaker_enabled(true);
  consumer.set_circuit_breaker_initial_backoff(absl::Milliseconds(200));
  const std::string peer = "127.0.0.1:54321";

  // 1. In CLOSED circuit: launch 8 concurrent worker threads recording failure.
  // The first 2 trip the circuit to OPEN (backoff becomes 400ms, banned for
  // 200ms). The remaining 6 run while circuit is already OPEN, so their
  // failures are ignored.
  {
    std::vector<std::thread> workers;
    for (int i = 0; i < 8; ++i) {
      workers.emplace_back(
          [&consumer, &peer]() { consumer.RecordCircuitBreakerFailure(peer); });
    }
    for (auto& t : workers) t.join();
  }

  EXPECT_TRUE(consumer.IsPeerBanned(peer));
  // In open circuit, worker thread failures are ignored and backoff remains
  // unchanged (400ms for next trip)
  EXPECT_EQ(consumer.GetPeerCurrentBackoff(peer), absl::Milliseconds(400));
  const absl::Time banned_until_1 = consumer.GetPeerBannedUntil(peer);

  // Any additional failures while OPEN do not extend banned_until or double
  // backoff
  consumer.RecordCircuitBreakerFailure(peer);
  EXPECT_EQ(consumer.GetPeerBannedUntil(peer), banned_until_1);
  EXPECT_EQ(consumer.GetPeerCurrentBackoff(peer), absl::Milliseconds(400));

  // 2. Wait for the 200ms ban to expire (circuit returns to CLOSED)
  while (absl::Now() < banned_until_1) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_FALSE(consumer.IsPeerBanned(peer));

  // 3. In CLOSED circuit: things compound!
  // 2 consecutive failures now trip with the compounded 400ms backoff,
  // and next backoff compounds to 800ms.
  consumer.RecordCircuitBreakerFailure(peer);
  EXPECT_FALSE(consumer.IsPeerBanned(peer));   // 1 failure: not banned yet
  consumer.RecordCircuitBreakerFailure(peer);  // 2nd failure: trips!
  EXPECT_TRUE(consumer.IsPeerBanned(peer));
  EXPECT_EQ(consumer.GetPeerCurrentBackoff(peer), absl::Milliseconds(800));

  // 4. On success: resets backoff to initial 200ms
  const absl::Time banned_until_2 = consumer.GetPeerBannedUntil(peer);
  while (absl::Now() < banned_until_2) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_FALSE(consumer.IsPeerBanned(peer));
  consumer.RecordCircuitBreakerSuccess(peer);
  EXPECT_EQ(consumer.GetPeerCurrentBackoff(peer), absl::Milliseconds(200));
}

}  // namespace
}  // namespace tpu_raiden
