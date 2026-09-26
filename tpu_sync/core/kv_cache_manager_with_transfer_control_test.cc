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
#include <cstdlib>
#include <cstring>
#include <exception>
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
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/core/control_plane_backend.h"
#include "tpu_sync/core/grpc_control_plane_backend.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"
#include "tpu_sync/core/tcp_control_plane_backend.h"
#include "tpu_sync/core/transfer_receive_session.h"
#include "tpu_sync/core/transfer_send_session.h"

namespace tpu_raiden {
namespace {

using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

// Both ends of a control handshake share one pool of four workers, so four
// stuck handshakes are enough to starve either side.
constexpr int kPoolSize = 4;
constexpr double kTimeoutS = 0.5;

// Selects the control-plane backend for managers built while it is in scope.
class ScopedControlPlaneBackend {
 public:
  explicit ScopedControlPlaneBackend(const char* backend,
                                     bool overwrite = true) {
    if (const char* old = std::getenv(kVar)) old_ = old;
    if (overwrite || !old_.has_value()) {
      setenv(kVar, backend, /*overwrite=*/1);
    }
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

class TestManager : private ScopedControlPlaneBackend,
                    public KVCacheManagerWithTransfer {
 public:
  explicit TestManager(double timeout_s = kTimeoutS, size_t num_layers = 0,
                       int local_control_port = 0)
      : ScopedControlPlaneBackend("tcp", /*overwrite=*/false),
        KVCacheManagerWithTransfer(
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
// Cross-peer fault isolation on the consumer handshake path (issue #888).
//
// The tests above establish that a silent producer costs its own reads one
// transfer timeout. These establish the part that is a fault-isolation bug:
// the cost must not be paid by reads aimed at *other*, healthy producers.
//
// The handshake used to run as a blocking call on push_pool_, one FIFO queue
// shared by every peer, so kPoolSize handshakes to one unresponsive producer
// held every worker and every other peer queued behind them. On the gRPC
// backend the handshake is now an async RPC that holds no worker while it
// waits. These run on gRPC only: the TCP backend still runs the blocking call
// on push_pool_ and is being retired rather than fixed.
// --------------------------------------------------------------------------

// A handshake timeout long enough that "blocked behind the sick peer" and
// "scheduled promptly" cannot be confused for one another. Reads to the
// healthy peer are expected to start in milliseconds; a worker stuck on the
// sick peer holds on for kStarvationTimeoutS.
constexpr double kStarvationTimeoutS = 4.0;

// The gRPC counterpart of SilentProducer: a real gRPC control server whose
// PullStream handler holds every call until DropClients(), then rejects it.
// accepted() counts calls that reached the handler.
class StalledGrpcProducer : public ControlPlaneHandler {
 public:
  StalledGrpcProducer() {
    absl::StatusOr<int> port = server_.StartServer(0, this);
    EXPECT_TRUE(port.ok()) << port.status();
    port_ = port.value_or(0);
  }

  ~StalledGrpcProducer() override {
    DropClients();
    server_.StopServer();
  }

  std::string endpoint() const { return absl::StrCat("127.0.0.1:", port_); }

  size_t accepted() {
    absl::MutexLock lock(mu_);
    return received_;
  }

  bool WaitUntilAccepted(size_t count, std::chrono::milliseconds timeout) {
    absl::MutexLock lock(mu_);
    auto reached = [this, count]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return received_ >= count;
    };
    return mu_.AwaitWithTimeout(absl::Condition(&reached),
                                absl::FromChrono(timeout));
  }

  void DropClients() {
    absl::MutexLock lock(mu_);
    released_ = true;
  }

  absl::StatusOr<PullStreamResponseSpec> OnPullStream(
      const PullStreamRequestSpec& req,
      absl::string_view fallback_peer_ip) override {
    absl::MutexLock lock(mu_);
    ++received_;
    mu_.Await(absl::Condition(&released_));
    return PullStreamResponseSpec{.status = -1,
                                  .message = "stalled producer released"};
  }

  absl::Status OnAck(uint64_t uuid) override { return absl::OkStatus(); }

 private:
  absl::Mutex mu_;
  size_t received_ ABSL_GUARDED_BY(mu_) = 0;
  bool released_ ABSL_GUARDED_BY(mu_) = false;
  int port_ = 0;
  GrpcControlPlaneBackend server_;
};

// Waits for `count` reads to settle so teardown does not race them.
void DrainReads(TestManager& consumer, size_t count, absl::Duration timeout) {
  const absl::Time drain_deadline = absl::Now() + timeout;
  size_t settled = 0;
  while (settled < count && absl::Now() < drain_deadline) {
    auto [sent, received, failed] = consumer.CompleteReadRaw();
    (void)sent;
    settled += received.size() + failed.size();
    absl::SleepFor(absl::Milliseconds(20));
  }
}

TEST(ControlHandshakeTest, GrpcSickPeerDoesNotDelayHandshakeToHealthyPeer) {
  ScopedControlPlaneBackend grpc("grpc");
  StalledGrpcProducer sick;
  StalledGrpcProducer healthy;
  TestManager consumer(/*timeout_s=*/kStarvationTimeoutS);

  // Put as many handshakes in flight to the sick peer as the consumer has
  // push-pool workers. The sick peer accepts them, so these are not connect()
  // failures: each one is waiting on a response that never comes.
  for (int i = 0; i < kPoolSize; ++i) {
    consumer.StartRead(absl::StrCat("sick", i), /*uuid=*/300 + i,
                       sick.endpoint(), /*remote_block_ids=*/{0},
                       /*local_block_ids=*/{0});
  }
  ASSERT_TRUE(sick.WaitUntilAccepted(kPoolSize, std::chrono::seconds(10)))
      << "precondition: " << kPoolSize
      << " handshakes are outstanding against the sick peer";

  // A read to a peer that is answering normally. Nothing about this request
  // depends on the sick peer; only shared consumer resources couple them.
  const absl::Time start = absl::Now();
  consumer.StartRead("healthy0", /*uuid=*/400, healthy.endpoint(),
                     /*remote_block_ids=*/{0}, /*local_block_ids=*/{0});

  // The healthy peer should see the request promptly. Wait well past the
  // sick peer's timeout so the failure message can report how long it
  // actually took: a delay that tracks kStarvationTimeoutS is the signature
  // of head-of-line blocking in the shared pool, as opposed to mere jitter.
  const bool contacted = healthy.WaitUntilAccepted(1, std::chrono::seconds(20));
  const double waited = SecondsSince(start);
  ASSERT_TRUE(contacted) << "healthy peer was never contacted at all";
  EXPECT_LT(waited, 0.5)
      << "reaching the healthy peer took " << waited
      << "s, against a sick-peer handshake timeout of " << kStarvationTimeoutS
      << "s; a single unresponsive producer is serialising the handshake "
         "path for every other peer";

  sick.DropClients();
  healthy.DropClients();
  DrainReads(consumer, kPoolSize + 1, absl::Seconds(20));
}

// The same coupling, stated as a throughput property rather than a latency
// one: reads to a healthy peer should keep completing while a sick peer is
// being waited on. Uses more sick reads than there are workers so the queue
// stays backed up, which is the production shape -- traffic to the dead peer
// keeps arriving and the pool never drains.
TEST(ControlHandshakeTest,
     GrpcHealthyPeerProgressesWhileSickPeerBacklogDrains) {
  ScopedControlPlaneBackend grpc("grpc");
  StalledGrpcProducer sick;
  StalledGrpcProducer healthy;
  TestManager consumer(/*timeout_s=*/kStarvationTimeoutS);

  // A backlog deeper than the pool, but sized with the healthy reads to stay
  // inside the consumer's staging slots. Overrunning them would make reads
  // fail allocation and never reach the pool at all, which is a different
  // defect (see SickPeerStarvesStagingSlotsForHealthyPeer) and would mask
  // this one.
  constexpr int kHealthyReads = 3;
  constexpr int kSickReads = 5;
  static_assert(kSickReads > kPoolSize, "backlog must exceed the pool");
  static_assert(kSickReads + kHealthyReads <= 2 * kPoolSize,
                "must fit in TestManager's staging slots");

  for (int i = 0; i < kSickReads; ++i) {
    consumer.StartRead(absl::StrCat("sick", i), /*uuid=*/500 + i,
                       sick.endpoint(), /*remote_block_ids=*/{0},
                       /*local_block_ids=*/{0});
  }
  ASSERT_TRUE(sick.WaitUntilAccepted(kPoolSize, std::chrono::seconds(10)));

  // Interleave healthy reads behind the backlog, as a scheduler would.
  const absl::Time start = absl::Now();
  for (int i = 0; i < kHealthyReads; ++i) {
    consumer.StartRead(absl::StrCat("healthy", i), /*uuid=*/600 + i,
                       healthy.endpoint(), /*remote_block_ids=*/{0},
                       /*local_block_ids=*/{0});
  }
  // Guards the precondition: these reads exist as sessions and are waiting on
  // the pool, rather than having been rejected before they got there.
  for (int i = 0; i < kHealthyReads; ++i) {
    ASSERT_TRUE(consumer.has_recv(600 + i))
        << "healthy read " << i << " was dropped before reaching the pool";
  }

  // With isolation the healthy reads are contacted quickly. With one FIFO
  // queue they are strictly behind kSickReads handshakes, so the first contact
  // costs a full timeout and draining the backlog costs two.
  const bool all_contacted =
      healthy.WaitUntilAccepted(kHealthyReads, std::chrono::seconds(30));
  const double waited = SecondsSince(start);
  ASSERT_TRUE(all_contacted)
      << "only " << healthy.accepted() << " of " << kHealthyReads
      << " healthy handshakes ever started";
  EXPECT_LT(waited, 0.5)
      << "draining " << kHealthyReads << " healthy handshakes took " << waited
      << "s while " << kSickReads
      << " reads to an unresponsive peer were outstanding; healthy traffic is "
         "queued strictly behind the sick backlog rather than sharing the pool";

  sick.DropClients();
  healthy.DropClients();
  DrainReads(consumer, kSickReads + kHealthyReads, absl::Seconds(30));
}

// A receive session holds its staging slot for the whole handshake
// (TransferReceiveSession::Create -> AllocateStagingForLoad, called from
// StartRead before the handshake is scheduled). Sessions stuck on an
// unresponsive peer therefore pin the staging pool as well as the thread
// pool, and once it is empty StartRead fails allocation and drops the read
// outright -- a read to a healthy peer is not merely delayed, it is rejected
// and never attempted.
//
// Still DISABLED_ on both backends: not holding a worker does not stop one
// peer from holding every staging slot. That needs per-peer admission at
// StartRead, which is a separate change.
TEST(ControlHandshakeTest, DISABLED_SickPeerStarvesStagingSlotsForHealthyPeer) {
  SilentProducer sick;
  SilentProducer healthy;
  TestManager consumer(/*timeout_s=*/kStarvationTimeoutS);

  // TestManager is built with num_slots = 2 * kPoolSize.
  constexpr int kSlots = 2 * kPoolSize;
  const size_t free_before = consumer.free_slots();
  ASSERT_GE(free_before, static_cast<size_t>(kSlots));

  for (int i = 0; i < kSlots; ++i) {
    consumer.StartRead(absl::StrCat("sick", i), /*uuid=*/800 + i,
                       sick.endpoint(), /*remote_block_ids=*/{0},
                       /*local_block_ids=*/{0});
  }
  ASSERT_TRUE(sick.WaitUntilAccepted(kPoolSize, std::chrono::seconds(10)));
  ASSERT_EQ(consumer.free_slots(), 0u)
      << "precondition: the sick peer is holding every staging slot";

  // A read to a peer that is answering normally.
  consumer.StartRead("healthy0", /*uuid=*/900, healthy.endpoint(),
                     /*remote_block_ids=*/{0}, /*local_block_ids=*/{0});

  EXPECT_TRUE(consumer.has_recv(900))
      << "the read to the healthy peer was rejected outright because an "
         "unresponsive peer holds all "
      << kSlots
      << " staging slots; no request to a healthy producer can even be "
         "attempted while another producer is wedged";

  auto [done_sending, done_recving, failed_recving] =
      consumer.CompleteReadRaw();
  (void)done_sending;
  (void)done_recving;
  EXPECT_THAT(failed_recving, ::testing::Not(Contains("healthy0")))
      << "the healthy read failed immediately rather than being served";

  sick.DropClients();
  healthy.DropClients();
  const absl::Time drain_deadline = absl::Now() + absl::Seconds(30);
  size_t settled = 0;
  while (settled < static_cast<size_t>(kSlots) &&
         absl::Now() < drain_deadline) {
    auto [sent, received, failed] = consumer.CompleteReadRaw();
    (void)sent;
    settled += received.size() + failed.size();
    absl::SleepFor(absl::Milliseconds(20));
  }
}

}  // namespace
}  // namespace tpu_raiden
