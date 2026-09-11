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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

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
  explicit TestManager(double timeout_s = kTimeoutS, size_t num_layers = 0)
      : KVCacheManagerWithTransfer(num_layers, /*num_shards=*/1,
                                   /*slice_byte_size=*/128,
                                   /*local_port=*/std::nullopt,
                                   /*host_blocks_to_allocate=*/std::nullopt,
                                   /*parallelism=*/1, /*node_id=*/0,
                                   /*local_control_port=*/0, /*max_blocks=*/8,
                                   /*num_slots=*/2 * kPoolSize, timeout_s) {}

  using KVCacheManagerWithTransfer::ControlRequestHeader;
  using KVCacheManagerWithTransfer::ControlResponseHeader;
  using KVCacheManagerWithTransfer::kControlMagic;
  using KVCacheManagerWithTransfer::kOpPullStream;
  using KVCacheManagerWithTransfer::kResponseMagic;

  void ExpireRecv(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    active_recv_entries_.at(uuid).deadline =
        std::chrono::steady_clock::now() - std::chrono::seconds(1);
  }

  size_t free_slots() {
    absl::MutexLock lock(mu_);
    return free_slots_.size();
  }

  bool has_recv(uint64_t uuid) {
    absl::MutexLock lock(mu_);
    return active_recv_entries_.contains(uuid);
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
          if (stopping_) {
            close(client);
            return;
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

 private:
  int fd_ = -1;
  int port_ = 0;
  const bool read_request_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<int> clients_;
  bool request_received_ = false;
  bool request_read_finished_ = false;
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

TEST(ControlHandshakeTest, ExpiredReceiveKeepsStagingUntilHandshakeEnds) {
  SilentProducer producer(/*read_request=*/true);
  TestManager consumer(/*timeout_s=*/5.0, /*num_layers=*/1);
  const size_t free_before = consumer.free_slots();
  consumer.StartRead("req", /*uuid=*/201, producer.endpoint(),
                     /*remote_block_ids=*/{0}, /*local_block_ids=*/{0});
  ASSERT_TRUE(producer.WaitUntilRequestReceived(std::chrono::seconds(5)));
  ASSERT_TRUE(consumer.has_recv(201));
  ASSERT_EQ(consumer.free_slots(), free_before - 1);

  consumer.ExpireRecv(201);
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

}  // namespace
}  // namespace tpu_raiden
