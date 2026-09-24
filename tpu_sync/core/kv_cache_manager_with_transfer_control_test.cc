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

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/core/control_plane_backend.h"
#include "tpu_sync/core/grpc_control_plane_backend.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/tcp_control_plane_backend.h"
#include "tpu_sync/core/transfer_receive_session.h"
#include "tpu_sync/core/transfer_send_session.h"
#include "tpu_sync/proto/kv_cache_control_plane_service.grpc.pb.h"
#include "tpu_sync/proto/kv_cache_control_plane_service.pb.h"

namespace tpu_raiden {
namespace {

using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

// Both ends of a control handshake share one pool of four workers, so four
// stuck handshakes are enough to starve either side.
constexpr int kPoolSize = 4;
constexpr double kTimeoutS = 0.5;

class TestManager : public KVCacheManagerWithTransfer {
 public:
  explicit TestManager(double timeout_s = kTimeoutS, size_t num_layers = 0,
                       int local_control_port = 0)
      : KVCacheManagerWithTransfer(
            num_layers, /*num_shards=*/1, /*slice_byte_size=*/128,
            /*local_port=*/std::nullopt,
            /*host_blocks_to_allocate=*/std::nullopt,
            /*parallelism=*/1, /*node_id=*/0, local_control_port,
            /*max_blocks=*/8,
            /*num_slots=*/2 * kPoolSize, timeout_s) {}

  using ControlRequestHeader = TcpControlPlaneBackend::ControlRequestHeader;
  using ControlResponseHeader = TcpControlPlaneBackend::ControlResponseHeader;
  static constexpr uint32_t kControlMagic =
      TcpControlPlaneBackend::kControlMagic;
  static constexpr uint32_t kOpPullStream =
      TcpControlPlaneBackend::kOpPullStream;
  static constexpr uint32_t kResponseMagic =
      TcpControlPlaneBackend::kResponseMagic;

  ControlResponseHeader ReadResponseHeaderForTest(int fd) {
    return TcpControlPlaneBackend::ReadControlResponseHeader(fd);
  }

  void HandleControlConnectionForTest(int fd) {
    if (auto* tcp_backend =
            dynamic_cast<TcpControlPlaneBackend*>(control_backend_.get())) {
      tcp_backend->HandleControlConnection(fd, control_handler_.get());
    }
  }

  size_t free_slots() { return staging_allocator_->num_free_slots(); }

  std::optional<std::string> recv_req_id(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    auto it = active_recv_sessions_.find(uuid);
    if (it == active_recv_sessions_.end()) {
      return std::nullopt;
    }
    const std::shared_ptr<TransferReceiveSession>& session = it->second;
    if (session->Done()) {
      return std::nullopt;
    }
    return session->req_id();
  }

  bool has_recv(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    auto it = active_recv_sessions_.find(uuid);
    return it != active_recv_sessions_.end() && !it->second->Done();
  }

  // Pulls waiting for their read to be registered.
  size_t parked_pulls() {
    absl::MutexLock lock(mu_);
    return pull_waiters_.size();
  }

  bool WaitForParkedPulls(size_t count, absl::Duration timeout) {
    const absl::Time give_up = absl::Now() + timeout;
    while (parked_pulls() != count) {
      if (absl::Now() >= give_up) return false;
      absl::SleepFor(absl::Milliseconds(5));
    }
    return true;
  }

  void MarkPullStarted(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    send_sessions_.at(uuid)->ValidateAndBeginPull(
        {0}, std::chrono::steady_clock::now());
  }
};

// These structs are copied directly onto the wire. Keep their ABI explicit so
// a compiler or field-layout change cannot silently break mixed-version peers.
static_assert(std::is_standard_layout_v<TestManager::ControlRequestHeader>);
static_assert(sizeof(TestManager::ControlRequestHeader) == 168);
static_assert(offsetof(TestManager::ControlRequestHeader, magic) == 0);
static_assert(offsetof(TestManager::ControlRequestHeader, uuid) == 8);
static_assert(offsetof(TestManager::ControlRequestHeader, num_blocks) == 24);
static_assert(offsetof(TestManager::ControlRequestHeader, consumer_ips) == 36);
static_assert(std::is_standard_layout_v<TestManager::ControlResponseHeader>);
static_assert(sizeof(TestManager::ControlResponseHeader) == 24);
static_assert(offsetof(TestManager::ControlResponseHeader, status) == 4);
static_assert(offsetof(TestManager::ControlResponseHeader, message_len) == 16);

class ScopedFd {
 public:
  explicit ScopedFd(int fd) : fd_(fd) {}
  ~ScopedFd() {
    if (fd_ >= 0) close(fd_);
  }
  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;

  int get() const { return fd_; }

 private:
  int fd_;
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

void SendRequest(int fd, uint32_t magic, uint32_t op, uint64_t uuid,
                 const std::vector<int64_t>& source_blocks,
                 const std::vector<int64_t>& destination_blocks) {
  ASSERT_EQ(source_blocks.size(), destination_blocks.size());
  TestManager::ControlRequestHeader req;
  req.magic = magic;
  req.op = op;
  req.uuid = uuid;
  req.num_blocks = source_blocks.size();
  WriteAll(fd, &req, sizeof(req));
  if (!source_blocks.empty()) {
    WriteAll(fd, source_blocks.data(),
             source_blocks.size() * sizeof(source_blocks[0]));
    WriteAll(fd, destination_blocks.data(),
             destination_blocks.size() * sizeof(destination_blocks[0]));
  }
}

// Sends a one-block pull for `uuid` the way StartRead does.
void SendPull(int fd, uint64_t uuid) {
  SendRequest(fd, TestManager::kControlMagic, TestManager::kOpPullStream, uuid,
              /*source_blocks=*/{0}, /*destination_blocks=*/{0});
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

TEST(ControlHandshakeTest, RegisteredPullIsAcknowledged) {
  TestManager producer;
  ASSERT_GT(producer.NotifyForRead("req40", /*uuid=*/40, {0}), 0);
  int fd = Connect(producer.local_control_port());
  SendPull(fd, /*uuid=*/40);
  Response response = ReadResponse(fd);
  close(fd);

  ASSERT_TRUE(response.received);
  EXPECT_EQ(response.status, 0);
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

TEST(ControlHandshakeTest, PullAfterRegistrationDeadlineIsRejected) {
  TestManager producer(/*timeout_s=*/0.05);
  ASSERT_GT(producer.NotifyForRead("expired", /*uuid=*/42, {0},
                                   std::chrono::steady_clock::now() -
                                       std::chrono::milliseconds(1)),
            0);

  int fd = Connect(producer.local_control_port());
  const absl::Time start = absl::Now();
  SendPull(fd, /*uuid=*/42);
  Response response = ReadResponse(fd);
  close(fd);

  ASSERT_TRUE(response.received);
  EXPECT_NE(response.status, 0);
  EXPECT_THAT(response.message, HasSubstr("expired"));
  EXPECT_LT(SecondsSince(start), 0.5);
  auto [done_sending, done_recving, failed_recving] =
      producer.CompleteReadRaw();
  (void)done_recving;
  EXPECT_THAT(done_sending, ::testing::IsEmpty());
  EXPECT_THAT(failed_recving, Contains("expired"));
}

TEST(ControlHandshakeTest,
     PullAheadOfRegistrationIsAcknowledgedOnceRegistered) {
  TestManager producer;
  int fd = Connect(producer.local_control_port());
  SendPull(fd, /*uuid=*/42);
  auto response_future = std::async(std::launch::async, ReadResponse, fd);

  // The pull is observably pending before registration; this avoids assuming
  // that a fixed sleep was long enough for a particular worker schedule.
  EXPECT_EQ(response_future.wait_for(std::chrono::milliseconds(100)),
            std::future_status::timeout);
  ASSERT_GT(producer.NotifyForRead("req42", 42, /*block_ids=*/{0}), 0);
  ASSERT_EQ(response_future.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  Response response = response_future.get();
  close(fd);

  ASSERT_TRUE(response.received);
  EXPECT_EQ(response.status, 0);
}

TEST(ControlHandshakeTest, UniqueRegisteredSubsetIsAcknowledged) {
  TestManager producer;
  ASSERT_GT(producer.NotifyForRead("req44", /*uuid=*/44, {0, 1, 2}), 0);
  int fd = Connect(producer.local_control_port());
  SendRequest(fd, TestManager::kControlMagic, TestManager::kOpPullStream,
              /*uuid=*/44, /*source_blocks=*/{2, 0},
              /*destination_blocks=*/{6, 7});
  Response response = ReadResponse(fd);
  close(fd);

  ASSERT_TRUE(response.received);
  EXPECT_EQ(response.status, 0);
}

TEST(ControlHandshakeTest, DuplicatePullIsRejectedBeforeAcknowledgement) {
  TestManager producer;
  ASSERT_GT(producer.NotifyForRead("req", /*uuid=*/52, {0}), 0);
  producer.MarkPullStarted(/*uuid=*/52);

  int duplicate_fd = Connect(producer.local_control_port());
  SendPull(duplicate_fd, /*uuid=*/52);
  Response duplicate = ReadResponse(duplicate_fd);
  close(duplicate_fd);
  ASSERT_TRUE(duplicate.received);
  EXPECT_NE(duplicate.status, 0);
  EXPECT_THAT(duplicate.message, HasSubstr("already"));
}

TEST(ControlHandshakeTest, PullOfUnregisteredBlockIsRejected) {
  TestManager producer;
  ASSERT_GT(producer.NotifyForRead("req45", /*uuid=*/45, {0, 1}), 0);
  int fd = Connect(producer.local_control_port());
  SendRequest(fd, TestManager::kControlMagic, TestManager::kOpPullStream,
              /*uuid=*/45, /*source_blocks=*/{0, 2},
              /*destination_blocks=*/{6, 7});
  Response response = ReadResponse(fd);
  close(fd);

  ASSERT_TRUE(response.received);
  EXPECT_NE(response.status, 0);
  EXPECT_THAT(response.message, HasSubstr("block not registered"));
}

TEST(ControlHandshakeTest, PullWithDuplicateSourceBlockIsRejected) {
  TestManager producer;
  ASSERT_GT(producer.NotifyForRead("req46", /*uuid=*/46, {0, 1}), 0);
  int fd = Connect(producer.local_control_port());
  SendRequest(fd, TestManager::kControlMagic, TestManager::kOpPullStream,
              /*uuid=*/46, /*source_blocks=*/{0, 0},
              /*destination_blocks=*/{6, 7});
  Response response = ReadResponse(fd);
  close(fd);

  ASSERT_TRUE(response.received);
  EXPECT_NE(response.status, 0);
  EXPECT_THAT(response.message, HasSubstr("duplicate producer block"));
}

TEST(ControlHandshakeTest, EmptyPullIsRejected) {
  TestManager producer;
  ASSERT_GT(producer.NotifyForRead("req47", /*uuid=*/47, {0}), 0);
  int fd = Connect(producer.local_control_port());
  SendRequest(fd, TestManager::kControlMagic, TestManager::kOpPullStream,
              /*uuid=*/47, /*source_blocks=*/{}, /*destination_blocks=*/{});
  Response response = ReadResponse(fd);
  close(fd);

  ASSERT_TRUE(response.received);
  EXPECT_NE(response.status, 0);
  EXPECT_THAT(response.message, HasSubstr("requested no blocks"));
}

TEST(ControlHandshakeTest, OversizedPullIsRejectedBeforeReadingItsBody) {
  TestManager producer;
  ScopedFd fd(Connect(producer.local_control_port()));
  TestManager::ControlRequestHeader request;
  request.magic = TestManager::kControlMagic;
  request.op = TestManager::kOpPullStream;
  request.uuid = 48;
  request.num_blocks = 9;  // TestManager is configured for at most 8 blocks.
  WriteAll(fd.get(), &request, sizeof(request));
  ASSERT_EQ(shutdown(fd.get(), SHUT_WR), 0);

  // A hostile peer need not send a body proportional to its untrusted count.
  Response response = ReadResponse(fd.get());
  ASSERT_TRUE(response.received);
  EXPECT_NE(response.status, 0);
  EXPECT_THAT(response.message, HasSubstr("exceeds"));
}

TEST(ControlHandshakeTest, BadMagicIsRejected) {
  TestManager producer;
  int fd = Connect(producer.local_control_port());
  SendRequest(fd, /*magic=*/0, TestManager::kOpPullStream, /*uuid=*/48,
              /*source_blocks=*/{}, /*destination_blocks=*/{});
  Response response = ReadResponse(fd);
  close(fd);

  ASSERT_TRUE(response.received);
  EXPECT_NE(response.status, 0);
  EXPECT_THAT(response.message, HasSubstr("bad control request magic"));
}

TEST(ControlHandshakeTest, UnknownOperationIsRejected) {
  TestManager producer;
  int fd = Connect(producer.local_control_port());
  SendRequest(fd, TestManager::kControlMagic, /*op=*/99, /*uuid=*/49,
              /*source_blocks=*/{}, /*destination_blocks=*/{});
  Response response = ReadResponse(fd);
  close(fd);

  ASSERT_TRUE(response.received);
  EXPECT_NE(response.status, 0);
  EXPECT_THAT(response.message, HasSubstr("unknown control op code"));
}

TEST(ControlHandshakeTest, OversizedErrorResponsePreservesBoundedPrefix) {
  TestManager consumer;
  int sockets[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  ScopedFd writer(sockets[0]);
  ScopedFd reader(sockets[1]);

  TestManager::ControlResponseHeader response;
  response.magic = TestManager::kResponseMagic;
  response.status = -1;
  response.message_len = std::numeric_limits<uint64_t>::max();
  WriteAll(writer.get(), &response, sizeof(response));
  constexpr size_t kExpectedPrefixBytes = 4 * 1024;
  std::string prefix(kExpectedPrefixBytes, 'x');
  constexpr char kDiagnostic[] = "useful remote diagnostic";
  prefix.replace(0, sizeof(kDiagnostic) - 1, kDiagnostic);
  WriteAll(writer.get(), prefix.data(), prefix.size());

  std::string error_message;
  try {
    (void)consumer.ReadResponseHeaderForTest(reader.get());
  } catch (const std::exception& error) {
    error_message = error.what();
  }
  EXPECT_THAT(error_message, HasSubstr(kDiagnostic));
  EXPECT_THAT(error_message, HasSubstr("truncated"));
  EXPECT_THAT(error_message,
              HasSubstr(std::to_string(std::numeric_limits<uint64_t>::max())));
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

TEST(ControlHandshakeTest, MidRequestDisconnectDoesNotRaiseSigpipe) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        std::signal(SIGPIPE, SIG_DFL);
        TestManager producer(/*timeout_s=*/kTimeoutS, /*num_layers=*/0,
                             /*local_control_port=*/-1);
        int sockets[2];
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) _exit(2);

        TestManager::ControlRequestHeader request;
        request.magic = TestManager::kControlMagic;
        request.op = TestManager::kOpPullStream;
        request.uuid = 50;
        request.num_blocks = 1;
        if (write(sockets[0], &request, sizeof(request)) !=
            static_cast<ssize_t>(sizeof(request))) {
          _exit(3);
        }
        const int64_t source_block = 0;
        if (write(sockets[0], &source_block, sizeof(source_block)) !=
            static_cast<ssize_t>(sizeof(source_block))) {
          _exit(4);
        }

        // The handler reads the source block, sees EOF where the destination
        // block should be, and attempts to return an error to a closed peer.
        close(sockets[0]);
        producer.HandleControlConnectionForTest(sockets[1]);
        close(sockets[1]);
        _exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(ControlHandshakeTest, ShutdownUnblocksPendingPull) {
  auto producer = std::make_unique<TestManager>(/*timeout_s=*/10.0);
  int fd = Connect(producer->local_control_port());
  SendPull(fd, /*uuid=*/51);
  auto response_future = std::async(std::launch::async, ReadResponse, fd);

  // Establish the externally visible precondition: the pull is still pending
  // and has neither been acknowledged nor rejected before shutdown begins.
  EXPECT_EQ(response_future.wait_for(std::chrono::milliseconds(100)),
            std::future_status::timeout);

  const absl::Time start = absl::Now();
  producer.reset();
  ASSERT_EQ(response_future.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  (void)response_future.get();
  close(fd);
  EXPECT_LT(SecondsSince(start), 1.0);
}

// A producer that accepts control connections and never answers them.
class SilentProducer {
 public:
  explicit SilentProducer(bool read_request = false)
      : read_request_(read_request) {
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
        {
          std::lock_guard<std::mutex> lock(mu_);
          if (stopping_ || drop_clients_) {
            shutdown(client, SHUT_RDWR);
            close(client);
            continue;
          }
          clients_.push_back(client);
        }
        cv_.notify_all();
        if (!read_request_) continue;

        TestManager::ControlRequestHeader request;
        bool complete = ReadAll(client, &request, sizeof(request));
        constexpr uint64_t kMaxTestBlocks = 64;
        if (complete && request.num_blocks <= kMaxTestBlocks) {
          std::vector<int64_t> block_ids(2 * request.num_blocks);
          complete = ReadAll(client, block_ids.data(),
                             block_ids.size() * sizeof(block_ids[0]));
        } else {
          complete = false;
        }
        std::lock_guard<std::mutex> lock(mu_);
        request_received_ = complete;
        request_read_finished_ = true;
        cv_.notify_all();
        return;
      }
    });
  }

  ~SilentProducer() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopping_ = true;
      for (int client : clients_) shutdown(client, SHUT_RDWR);
    }
    shutdown(fd_, SHUT_RDWR);
    close(fd_);
    fd_ = -1;
    thread_.join();
    for (int client : clients_) close(client);
  }

  std::string endpoint() const { return absl::StrCat("127.0.0.1:", port_); }

  size_t accepted() {
    std::lock_guard<std::mutex> lock(mu_);
    return clients_.size();
  }

  bool WaitUntilAccepted(size_t count, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, timeout, [this, count] {
      return clients_.size() >= count || stopping_;
    }) && clients_.size() >= count;
  }

  bool WaitUntilRequestReceived(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, timeout, [this] {
      return request_read_finished_ || stopping_;
    }) && request_received_;
  }

  void DropClient() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!clients_.empty()) shutdown(clients_.front(), SHUT_RDWR);
  }

  void DropClients() {
    std::lock_guard<std::mutex> lock(mu_);
    drop_clients_ = true;
    for (int client : clients_) shutdown(client, SHUT_RDWR);
  }

 private:
  int fd_ = -1;
  int port_ = 0;
  const bool read_request_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<int> clients_;
  bool request_received_ = false;
  bool request_read_finished_ = false;
  bool drop_clients_ = false;
  bool stopping_ = false;
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
  EXPECT_TRUE(producer.WaitUntilAccepted(reads, std::chrono::seconds(10)));
  EXPECT_LT(SecondsSince(start), 2 * kTimeoutS + 5.0);

  // Every read settles rather than leaking its receive session. With no
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

TEST(ControlHandshakeTest, DuplicateReceiveDoesNotReplaceOrLeakFirstRead) {
  SilentProducer producer(/*read_request=*/true);
  TestManager consumer(/*timeout_s=*/5.0, /*num_layers=*/1);
  const size_t free_before = consumer.free_slots();

  consumer.StartRead("first", /*uuid=*/200, producer.endpoint(),
                     /*remote_block_ids=*/{0}, /*local_block_ids=*/{0});
  ASSERT_TRUE(producer.WaitUntilRequestReceived(std::chrono::seconds(5)));
  ASSERT_EQ(consumer.recv_req_id(200), std::optional<std::string>("first"));
  ASSERT_EQ(consumer.free_slots(), free_before - 1);

  consumer.StartRead("duplicate", /*uuid=*/200, producer.endpoint(),
                     /*remote_block_ids=*/{0}, /*local_block_ids=*/{0});
  EXPECT_EQ(consumer.recv_req_id(200), std::optional<std::string>("first"));
  EXPECT_EQ(consumer.free_slots(), free_before - 1);
  producer.DropClient();

  std::vector<std::string> failed;
  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (failed.size() < 2 && absl::Now() < deadline) {
    auto [done_sending, done_recving, newly_failed] =
        consumer.CompleteReadRaw();
    EXPECT_THAT(done_sending, ::testing::IsEmpty());
    EXPECT_THAT(done_recving, ::testing::IsEmpty());
    failed.insert(failed.end(), newly_failed.begin(), newly_failed.end());
    absl::SleepFor(absl::Milliseconds(1));
  }

  EXPECT_THAT(failed, UnorderedElementsAre("first", "duplicate"));
  EXPECT_EQ(consumer.free_slots(), free_before);
  auto [done_again, received_again, failed_again] = consumer.CompleteReadRaw();
  EXPECT_THAT(done_again, ::testing::IsEmpty());
  EXPECT_THAT(received_again, ::testing::IsEmpty());
  EXPECT_THAT(failed_again, ::testing::IsEmpty());
}

TEST(ControlHandshakeTest, RepeatedReceiveAnnouncementIsIdempotent) {
  SilentProducer producer(/*read_request=*/true);
  TestManager consumer(/*timeout_s=*/5.0, /*num_layers=*/1);
  const size_t free_before = consumer.free_slots();

  consumer.StartRead("req", /*uuid=*/201, producer.endpoint(),
                     /*remote_block_ids=*/{0}, /*local_block_ids=*/{0});
  ASSERT_TRUE(producer.WaitUntilRequestReceived(std::chrono::seconds(5)));
  consumer.StartRead("req", /*uuid=*/201, producer.endpoint(),
                     /*remote_block_ids=*/{0}, /*local_block_ids=*/{0});

  auto [done_before, received_before, failed_before] =
      consumer.CompleteReadRaw();
  EXPECT_THAT(done_before, ::testing::IsEmpty());
  EXPECT_THAT(received_before, ::testing::IsEmpty());
  EXPECT_THAT(failed_before, ::testing::IsEmpty());
  EXPECT_EQ(consumer.free_slots(), free_before - 1);

  producer.DropClient();
  std::vector<std::string> failed;
  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (failed.empty() && absl::Now() < deadline) {
    auto [done_sending, done_recving, newly_failed] =
        consumer.CompleteReadRaw();
    EXPECT_THAT(done_sending, ::testing::IsEmpty());
    EXPECT_THAT(done_recving, ::testing::IsEmpty());
    failed.insert(failed.end(), newly_failed.begin(), newly_failed.end());
    absl::SleepFor(absl::Milliseconds(1));
  }
  EXPECT_THAT(failed, ::testing::ElementsAre("req"));
  EXPECT_EQ(consumer.free_slots(), free_before);

  auto [done_again, received_again, failed_again] = consumer.CompleteReadRaw();
  EXPECT_THAT(done_again, ::testing::IsEmpty());
  EXPECT_THAT(received_again, ::testing::IsEmpty());
  EXPECT_THAT(failed_again, ::testing::IsEmpty());
}

TEST(ControlHandshakeTest, ExpiredReceiveKeepsStagingUntilHandshakeEnds) {
  SilentProducer producer(/*read_request=*/true);
  TestManager consumer(/*timeout_s=*/5.0, /*num_layers=*/1);
  const size_t free_before = consumer.free_slots();
  consumer.StartRead(
      "req", /*uuid=*/201, producer.endpoint(),
      /*remote_block_ids=*/{0}, /*local_block_ids=*/{0},
      /*parallelism=*/1, /*local_host_block_ids=*/std::nullopt,
      std::chrono::steady_clock::now() + std::chrono::milliseconds(20));
  ASSERT_TRUE(producer.WaitUntilRequestReceived(std::chrono::seconds(5)));
  ASSERT_TRUE(consumer.has_recv(201));
  ASSERT_EQ(consumer.free_slots(), free_before - 1);

  absl::SleepFor(absl::Milliseconds(25));
  auto [done_sending, done_recving, failed_during] = consumer.CompleteReadRaw();
  (void)done_sending;
  EXPECT_THAT(done_recving, ::testing::IsEmpty());
  EXPECT_THAT(failed_during, ::testing::IsEmpty());
  EXPECT_EQ(consumer.free_slots(), free_before - 1);

  producer.DropClient();
  std::vector<std::string> done_after;
  std::vector<std::string> failed_after;
  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (failed_after.empty() && absl::Now() < deadline) {
    auto [sent, received, failed] = consumer.CompleteReadRaw();
    (void)sent;
    done_after.insert(done_after.end(), received.begin(), received.end());
    failed_after.insert(failed_after.end(), failed.begin(), failed.end());
    absl::SleepFor(absl::Milliseconds(1));
  }
  EXPECT_THAT(done_after, ::testing::IsEmpty());
  EXPECT_THAT(failed_after, ::testing::ElementsAre("req"));
  EXPECT_FALSE(consumer.has_recv(201));
  EXPECT_EQ(consumer.free_slots(), free_before);

  auto [sent_again, received_again, failed_again] = consumer.CompleteReadRaw();
  EXPECT_THAT(sent_again, ::testing::IsEmpty());
  EXPECT_THAT(received_again, ::testing::IsEmpty());
  EXPECT_THAT(failed_again, ::testing::IsEmpty());
}

// --------------------------------------------------------------------------
// Producer-side pulls on the gRPC control plane (issue #888).
//
// A pull whose read the producer has not registered yet used to wait out the
// registration grace on a server thread, so a burst of early, expired or
// bogus pulls from one consumer could tie up the threads every other consumer
// needs. It is now parked without a thread, capped per peer, and released at
// the consumer's deadline or on cancellation.
// --------------------------------------------------------------------------

// Selects the control-plane backend for managers built while it is in scope.
class ScopedControlPlaneBackend {
 public:
  explicit ScopedControlPlaneBackend(const char* backend) {
    if (const char* old = std::getenv(kVar)) old_ = old;
    setenv(kVar, backend, /*overwrite=*/1);
  }
  ~ScopedControlPlaneBackend() {
    if (old_.has_value()) {
      setenv(kVar, old_->c_str(), /*overwrite=*/1);
    } else {
      unsetenv(kVar);
    }
  }

 private:
  static constexpr char kVar[] = "TPU_RAIDEN_CONTROL_PLANE_BACKEND";
  std::optional<std::string> old_;
};

// The consumer end of the gRPC control plane, reduced to PullStream: issues
// pulls to one producer on the raw stub, so a test can cancel them, and
// records each answer by uuid. Answers the RPC itself fails are recorded as
// the gRPC status.
class GrpcPuller {
 public:
  GrpcPuller(absl::string_view host, int port)
      : stub_(control_plane::proto::KVCacheControlPlaneService::NewStub(
            grpc::CreateChannel(absl::StrCat(host, ":", port),
                                grpc::InsecureChannelCredentials()))) {}

  // Cancels every pull still outstanding and waits for its callback.
  ~GrpcPuller() {
    CancelAll();
    absl::MutexLock lock(mu_);
    mu_.Await(absl::Condition(
        +[](absl::flat_hash_map<uint64_t, std::unique_ptr<Call>>* calls) {
          return calls->empty();
        },
        &calls_));
  }

  void Pull(uint64_t uuid, absl::Duration timeout = absl::Seconds(30)) {
    Pull(uuid, /*src_blocks=*/{0}, /*dst_blocks=*/{0}, timeout);
  }

  void Pull(uint64_t uuid, const std::vector<int64_t>& src_blocks,
            const std::vector<int64_t>& dst_blocks,
            absl::Duration timeout = absl::Seconds(30)) {
    auto call = std::make_unique<Call>();
    call->context.set_deadline(absl::ToChronoTime(absl::Now() + timeout));
    call->request.set_uuid(uuid);
    for (int64_t block : src_blocks) call->request.add_src_block_ids(block);
    for (int64_t block : dst_blocks) call->request.add_dst_block_ids(block);
    Call* raw = call.get();
    {
      absl::MutexLock lock(mu_);
      calls_.insert_or_assign(uuid, std::move(call));
    }
    stub_->async()->PullStream(
        &raw->context, &raw->request, &raw->response,
        [this, uuid, raw](grpc::Status status) {
          absl::StatusOr<PullStreamResponseSpec> answer =
              status.ok()
                  ? absl::StatusOr<PullStreamResponseSpec>(
                        PullStreamResponseSpec{
                            .status = raw->response.status(),
                            .num_layers = raw->response.num_layers(),
                            .data_port = raw->response.data_port(),
                            .message = raw->response.message()})
                  : absl::Status(
                        static_cast<absl::StatusCode>(status.error_code()),
                        status.error_message());
          absl::MutexLock lock(mu_);
          answers_.insert_or_assign(uuid, std::move(answer));
          calls_.erase(uuid);
        });
  }

  void CancelAll() {
    absl::MutexLock lock(mu_);
    for (auto& [uuid, call] : calls_) call->context.TryCancel();
  }

  std::optional<absl::StatusOr<PullStreamResponseSpec>> WaitForAnswer(
      uint64_t uuid, absl::Duration timeout) {
    absl::MutexLock lock(mu_);
    auto answered = [this, uuid]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return answers_.contains(uuid);
    };
    if (!mu_.AwaitWithTimeout(absl::Condition(&answered), timeout)) {
      return std::nullopt;
    }
    return answers_.at(uuid);
  }

 private:
  struct Call {
    grpc::ClientContext context;
    control_plane::proto::PullStreamRequest request;
    control_plane::proto::PullStreamResponse response;
  };

  std::unique_ptr<control_plane::proto::KVCacheControlPlaneService::Stub>
      stub_;
  absl::Mutex mu_;
  absl::flat_hash_map<uint64_t, std::unique_ptr<Call>> calls_
      ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<uint64_t, absl::StatusOr<PullStreamResponseSpec>>
      answers_ ABSL_GUARDED_BY(mu_);
};

int ThreadCount() {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("Threads:", 0) == 0) {
      return std::stoi(line.substr(8));
    }
  }
  return -1;
}

bool HasIpv6Loopback() {
  int fd = socket(AF_INET6, SOCK_STREAM, 0);
  if (fd < 0) return false;
  sockaddr_in6 addr{};
  addr.sin6_family = AF_INET6;
  addr.sin6_addr = in6addr_loopback;
  const bool bound =
      bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
  close(fd);
  return bound;
}

TEST(ControlHandshakeTest,
     GrpcPullAheadOfRegistrationIsAcknowledgedOnceRegistered) {
  ScopedControlPlaneBackend grpc("grpc");
  TestManager producer(/*timeout_s=*/10.0);
  GrpcPuller consumer("127.0.0.1", producer.local_control_port());

  consumer.Pull(/*uuid=*/700);
  ASSERT_TRUE(producer.WaitForParkedPulls(1, absl::Seconds(5)));
  ASSERT_GT(producer.NotifyForRead("req700", /*uuid=*/700, {0}), 0);

  auto answer = consumer.WaitForAnswer(700, absl::Seconds(1));
  ASSERT_TRUE(answer.has_value()) << "registration did not answer the pull";
  ASSERT_TRUE(answer->ok()) << answer->status();
  EXPECT_EQ((*answer)->status, 0) << (*answer)->message;
  EXPECT_EQ(producer.parked_pulls(), 0u);
}

TEST(ControlHandshakeTest, GrpcPullWithoutRegistrationIsRejectedAfterGrace) {
  ScopedControlPlaneBackend grpc("grpc");
  // The grace is min(5 s, timeout_s), so kTimeoutS here.
  TestManager producer;
  GrpcPuller consumer("127.0.0.1", producer.local_control_port());

  const absl::Time start = absl::Now();
  consumer.Pull(/*uuid=*/701);
  auto answer = consumer.WaitForAnswer(701, absl::Seconds(5));
  ASSERT_TRUE(answer.has_value());
  ASSERT_TRUE(answer->ok()) << answer->status();
  EXPECT_NE((*answer)->status, 0);
  EXPECT_THAT((*answer)->message, HasSubstr("no read registered for uuid 701"));
  EXPECT_LT(SecondsSince(start), 4 * kTimeoutS);
  EXPECT_EQ(producer.parked_pulls(), 0u);
}

// Pulls for reads the producer has not registered, whether early, expired or
// bogus, used to wait out the registration grace on a server thread each. A
// burst of them tied up one thread apiece, and on a bounded pool they locked
// out every other consumer. Parked, each is an entry: the thread count does not
// track them, and a pull the producer can answer is answered at once.
TEST(ControlHandshakeTest, GrpcParkedPullsHoldNoThreadsAndDoNotDelayOthers) {
  ScopedControlPlaneBackend grpc("grpc");
  TestManager producer(/*timeout_s=*/30.0);
  GrpcPuller early("127.0.0.1", producer.local_control_port());
  GrpcPuller prompt("127.0.0.1", producer.local_control_port());

  // Warm both channels and the server so lazily started gRPC threads exist
  // before the count is taken.
  ASSERT_GT(producer.NotifyForRead("warm0", /*uuid=*/710, {0}), 0);
  ASSERT_GT(producer.NotifyForRead("warm1", /*uuid=*/711, {0}), 0);
  early.Pull(710);
  prompt.Pull(711);
  ASSERT_TRUE(early.WaitForAnswer(710, absl::Seconds(5)).has_value());
  ASSERT_TRUE(prompt.WaitForAnswer(711, absl::Seconds(5)).has_value());
  const int threads_before = ThreadCount();
  ASSERT_GT(threads_before, 0);

  constexpr size_t kParked = KVCacheManagerWithTransfer::kMaxPullWaitersPerPeer;
  for (size_t i = 0; i < kParked; ++i) {
    early.Pull(/*uuid=*/800 + i);
  }
  ASSERT_TRUE(producer.WaitForParkedPulls(kParked, absl::Seconds(10)))
      << "only " << producer.parked_pulls() << " of " << kParked
      << " unregistered pulls were parked";
  const int grown = ThreadCount() - threads_before;
  EXPECT_LT(grown, static_cast<int>(kParked / 2))
      << kParked << " parked pulls added " << grown
      << " threads; a pull waiting for registration is holding a thread";

  ASSERT_GT(producer.NotifyForRead("req712", /*uuid=*/712, {0}), 0);
  const absl::Time start = absl::Now();
  prompt.Pull(712);
  auto answer = prompt.WaitForAnswer(712, absl::Seconds(10));
  const double waited = SecondsSince(start);
  ASSERT_TRUE(answer.has_value());
  ASSERT_TRUE(answer->ok()) << answer->status();
  EXPECT_EQ((*answer)->status, 0) << (*answer)->message;
  EXPECT_LT(waited, 0.5) << "a registered pull took " << waited << "s behind "
                         << kParked << " parked ones";
  EXPECT_EQ(producer.parked_pulls(), kParked);
}

TEST(ControlHandshakeTest, GrpcPerPeerCapRejectsOnlyTheFloodingPeer) {
  if (!HasIpv6Loopback()) {
    GTEST_SKIP() << "needs ::1 to connect as a second peer";
  }
  ScopedControlPlaneBackend grpc("grpc");
  TestManager producer(/*timeout_s=*/30.0);
  GrpcPuller flooder("127.0.0.1", producer.local_control_port());
  GrpcPuller other("[::1]", producer.local_control_port());

  constexpr size_t kCap = KVCacheManagerWithTransfer::kMaxPullWaitersPerPeer;
  for (size_t i = 0; i < kCap; ++i) {
    flooder.Pull(/*uuid=*/900 + i);
  }
  ASSERT_TRUE(producer.WaitForParkedPulls(kCap, absl::Seconds(10)));

  // One more from the same peer is refused at once rather than parked.
  const absl::Time start = absl::Now();
  flooder.Pull(/*uuid=*/900 + kCap);
  auto refused = flooder.WaitForAnswer(900 + kCap, absl::Seconds(5));
  ASSERT_TRUE(refused.has_value());
  ASSERT_TRUE(refused->ok()) << refused->status();
  EXPECT_NE((*refused)->status, 0);
  EXPECT_THAT((*refused)->message, HasSubstr("too many pulls from 127.0.0.1"));
  EXPECT_LT(SecondsSince(start), 1.0);

  // Another peer's early pull is still parked, and served on registration.
  other.Pull(/*uuid=*/1000);
  ASSERT_TRUE(producer.WaitForParkedPulls(kCap + 1, absl::Seconds(5)))
      << "the other peer's pull was not parked";
  ASSERT_GT(producer.NotifyForRead("req1000", /*uuid=*/1000, {0}), 0);
  auto served = other.WaitForAnswer(1000, absl::Seconds(1));
  ASSERT_TRUE(served.has_value());
  ASSERT_TRUE(served->ok()) << served->status();
  EXPECT_EQ((*served)->status, 0) << (*served)->message;
}

TEST(ControlHandshakeTest, GrpcParkedPullIsReleasedAtConsumerDeadline) {
  ScopedControlPlaneBackend grpc("grpc");
  // A 5 s grace, so a release well before it can only be the deadline.
  TestManager producer(/*timeout_s=*/30.0);
  GrpcPuller consumer("127.0.0.1", producer.local_control_port());

  const absl::Time start = absl::Now();
  consumer.Pull(/*uuid=*/1100, /*timeout=*/absl::Seconds(1));
  ASSERT_TRUE(producer.WaitForParkedPulls(1, absl::Milliseconds(900)));

  // The producer gives up at the consumer's deadline, so its rejection and the
  // consumer's own DEADLINE_EXCEEDED race; either is a prompt release.
  auto answer = consumer.WaitForAnswer(1100, absl::Seconds(5));
  ASSERT_TRUE(answer.has_value());
  if (answer->ok()) {
    EXPECT_NE((*answer)->status, 0);
    EXPECT_THAT((*answer)->message,
                HasSubstr("no read registered for uuid 1100"));
  } else {
    EXPECT_EQ(answer->status().code(), absl::StatusCode::kDeadlineExceeded)
        << answer->status();
  }
  EXPECT_TRUE(producer.WaitForParkedPulls(0, absl::Seconds(1)))
      << "the producer still holds a pull its consumer gave up on";
  EXPECT_LT(SecondsSince(start), 2.5);
}

TEST(ControlHandshakeTest, GrpcCancelledPullIsUnparked) {
  ScopedControlPlaneBackend grpc("grpc");
  TestManager producer(/*timeout_s=*/30.0);
  auto consumer =
      std::make_unique<GrpcPuller>("127.0.0.1", producer.local_control_port());

  consumer->Pull(/*uuid=*/1200, /*timeout=*/absl::Seconds(30));
  ASSERT_TRUE(producer.WaitForParkedPulls(1, absl::Seconds(5)));

  // Destroying the client cancels its outstanding calls.
  const absl::Time start = absl::Now();
  consumer.reset();
  EXPECT_TRUE(producer.WaitForParkedPulls(0, absl::Seconds(2)))
      << "the producer still holds a pull its consumer cancelled";
  EXPECT_LT(SecondsSince(start), 2.0);
}

TEST(ControlHandshakeTest, GrpcShutdownAnswersParkedPulls) {
  ScopedControlPlaneBackend grpc("grpc");
  auto producer = std::make_unique<TestManager>(/*timeout_s=*/30.0);
  GrpcPuller consumer("127.0.0.1", producer->local_control_port());

  constexpr int kParked = 8;
  for (int i = 0; i < kParked; ++i) {
    consumer.Pull(/*uuid=*/1300 + i);
  }
  ASSERT_TRUE(producer->WaitForParkedPulls(kParked, absl::Seconds(5)));

  const absl::Time start = absl::Now();
  producer.reset();
  EXPECT_LT(SecondsSince(start), 2.0);
  for (int i = 0; i < kParked; ++i) {
    auto answer = consumer.WaitForAnswer(1300 + i, absl::Seconds(2));
    ASSERT_TRUE(answer.has_value()) << "parked pull " << i << " never answered";
    ASSERT_TRUE(answer->ok()) << answer->status();
    EXPECT_NE((*answer)->status, 0);
    EXPECT_THAT((*answer)->message, HasSubstr("stopping"));
  }
}

// The TCP backend refuses an oversized pull from its header, before the
// handler sees it; on gRPC the handler's own check is the only one.
TEST(ControlHandshakeTest, GrpcOversizedPullIsRejectedWithoutParking) {
  ScopedControlPlaneBackend grpc("grpc");
  TestManager producer(/*timeout_s=*/30.0);
  GrpcPuller consumer("127.0.0.1", producer.local_control_port());

  // TestManager is configured for at most 8 blocks. The read is not
  // registered, so a pull that got past the check would be parked.
  const std::vector<int64_t> blocks = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  const absl::Time start = absl::Now();
  consumer.Pull(/*uuid=*/1400, blocks, blocks);
  auto answer = consumer.WaitForAnswer(1400, absl::Seconds(5));
  ASSERT_TRUE(answer.has_value());
  ASSERT_TRUE(answer->ok()) << answer->status();
  EXPECT_NE((*answer)->status, 0);
  EXPECT_THAT((*answer)->message,
              HasSubstr("pull stream block count 9 exceeds configured "
                        "maximum 8"));
  EXPECT_LT(SecondsSince(start), 1.0);
  EXPECT_EQ(producer.parked_pulls(), 0u);
}

struct InvalidPullCase {
  const char* name;
  std::vector<int64_t> src_blocks;
  std::vector<int64_t> dst_blocks;
  const char* message;
};

// Each registers blocks {0, 1}.
std::vector<InvalidPullCase> InvalidPullCases() {
  return {
      {"unregistered block", {0, 2}, {6, 7}, "block not registered"},
      {"duplicate block", {0, 0}, {6, 7}, "duplicate producer block"},
      {"no blocks", {}, {}, "requested no blocks"},
  };
}

TEST(ControlHandshakeTest, GrpcInvalidPullOfRegisteredReadIsRejected) {
  ScopedControlPlaneBackend grpc("grpc");
  TestManager producer(/*timeout_s=*/30.0);
  GrpcPuller consumer("127.0.0.1", producer.local_control_port());

  uint64_t uuid = 1500;
  for (const InvalidPullCase& c : InvalidPullCases()) {
    SCOPED_TRACE(c.name);
    ++uuid;
    ASSERT_GT(producer.NotifyForRead(absl::StrCat("req", uuid), uuid, {0, 1}),
              0);
    consumer.Pull(uuid, c.src_blocks, c.dst_blocks);
    auto answer = consumer.WaitForAnswer(uuid, absl::Seconds(5));
    ASSERT_TRUE(answer.has_value());
    ASSERT_TRUE(answer->ok()) << answer->status();
    EXPECT_NE((*answer)->status, 0);
    EXPECT_THAT((*answer)->message, HasSubstr(c.message));
  }
  EXPECT_EQ(producer.parked_pulls(), 0u);
}

// A parked pull is validated only once its read is registered, from
// NotifyForRead rather than the handler, and must still be answered.
TEST(ControlHandshakeTest, GrpcInvalidParkedPullIsRejectedOnRegistration) {
  ScopedControlPlaneBackend grpc("grpc");
  TestManager producer(/*timeout_s=*/30.0);
  GrpcPuller consumer("127.0.0.1", producer.local_control_port());

  const std::vector<InvalidPullCase> cases = InvalidPullCases();
  for (size_t i = 0; i < cases.size(); ++i) {
    consumer.Pull(/*uuid=*/1600 + i, cases[i].src_blocks,
                  cases[i].dst_blocks);
  }
  ASSERT_TRUE(producer.WaitForParkedPulls(cases.size(), absl::Seconds(5)));

  for (size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(cases[i].name);
    const uint64_t uuid = 1600 + i;
    ASSERT_GT(producer.NotifyForRead(absl::StrCat("req", uuid), uuid, {0, 1}),
              0);
    auto answer = consumer.WaitForAnswer(uuid, absl::Seconds(1));
    ASSERT_TRUE(answer.has_value()) << "registration did not answer the pull";
    ASSERT_TRUE(answer->ok()) << answer->status();
    EXPECT_NE((*answer)->status, 0);
    EXPECT_THAT((*answer)->message, HasSubstr(cases[i].message));
  }
  EXPECT_EQ(producer.parked_pulls(), 0u);
}

TEST(ControlHandshakeTest, GrpcPullAfterRegistrationDeadlineIsRejected) {
  ScopedControlPlaneBackend grpc("grpc");
  TestManager producer(/*timeout_s=*/30.0);
  GrpcPuller consumer("127.0.0.1", producer.local_control_port());
  ASSERT_GT(producer.NotifyForRead("expired", /*uuid=*/1700, {0},
                                   std::chrono::steady_clock::now() -
                                       std::chrono::milliseconds(1)),
            0);

  const absl::Time start = absl::Now();
  consumer.Pull(/*uuid=*/1700);
  auto answer = consumer.WaitForAnswer(1700, absl::Seconds(5));
  ASSERT_TRUE(answer.has_value());
  ASSERT_TRUE(answer->ok()) << answer->status();
  EXPECT_NE((*answer)->status, 0);
  EXPECT_THAT((*answer)->message, HasSubstr("expired"));
  EXPECT_LT(SecondsSince(start), 1.0);
  EXPECT_EQ(producer.parked_pulls(), 0u);
}

TEST(ControlHandshakeTest, GrpcParkedPullIsRejectedWhenRegisteredExpired) {
  ScopedControlPlaneBackend grpc("grpc");
  TestManager producer(/*timeout_s=*/30.0);
  GrpcPuller consumer("127.0.0.1", producer.local_control_port());

  consumer.Pull(/*uuid=*/1701);
  ASSERT_TRUE(producer.WaitForParkedPulls(1, absl::Seconds(5)));
  ASSERT_GT(producer.NotifyForRead("expired", /*uuid=*/1701, {0},
                                   std::chrono::steady_clock::now() -
                                       std::chrono::milliseconds(1)),
            0);

  auto answer = consumer.WaitForAnswer(1701, absl::Seconds(1));
  ASSERT_TRUE(answer.has_value()) << "registration did not answer the pull";
  ASSERT_TRUE(answer->ok()) << answer->status();
  EXPECT_NE((*answer)->status, 0);
  EXPECT_THAT((*answer)->message, HasSubstr("expired"));
  EXPECT_EQ(producer.parked_pulls(), 0u);
}

TEST(ControlHandshakeTest, GrpcDuplicatePullIsRejected) {
  ScopedControlPlaneBackend grpc("grpc");
  TestManager producer(/*timeout_s=*/30.0);
  GrpcPuller consumer("127.0.0.1", producer.local_control_port());
  ASSERT_GT(producer.NotifyForRead("req1800", /*uuid=*/1800, {0}), 0);
  producer.MarkPullStarted(/*uuid=*/1800);

  consumer.Pull(/*uuid=*/1800);
  auto answer = consumer.WaitForAnswer(1800, absl::Seconds(5));
  ASSERT_TRUE(answer.has_value());
  ASSERT_TRUE(answer->ok()) << answer->status();
  EXPECT_NE((*answer)->status, 0);
  EXPECT_THAT((*answer)->message, HasSubstr("already"));
}

}  // namespace
}  // namespace tpu_raiden
