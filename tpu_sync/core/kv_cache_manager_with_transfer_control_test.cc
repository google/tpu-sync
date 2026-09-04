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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
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
    if (n <= 0) return false;
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

}  // namespace
}  // namespace tpu_raiden
