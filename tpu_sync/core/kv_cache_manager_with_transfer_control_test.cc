// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <future>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/core/kv_cache_manager_with_transfer.h"

namespace tpu_raiden {
namespace {

class TestManager : public KVCacheManagerWithTransfer {
 public:
  explicit TestManager(double timeout_s)
      : KVCacheManagerWithTransfer(
            /*num_layers=*/1, /*num_shards=*/1,
            /*slice_byte_size=*/128,
            /*local_port=*/std::nullopt,
            /*host_blocks_to_allocate=*/std::make_optional(4),
            /*parallelism=*/1, /*node_id=*/0,
            /*local_control_port=*/0, /*max_blocks=*/1, /*num_slots=*/1,
            timeout_s) {}

  using KVCacheManagerWithTransfer::ControlRequestHeader;
  using KVCacheManagerWithTransfer::ControlResponseHeader;
  using KVCacheManagerWithTransfer::kOpPullStream;
  using KVCacheManagerWithTransfer::kResponseMagic;
};

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

bool WriteExact(int fd, const void* buffer, size_t length) {
  const auto* next = static_cast<const uint8_t*>(buffer);
  while (length > 0) {
    const ssize_t written = send(fd, next, length, MSG_NOSIGNAL);
    if (written < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (written == 0) return false;
    next += written;
    length -= static_cast<size_t>(written);
  }
  return true;
}

enum class PullResult {
  kRejected,
  kAccepted,
  kTimedOut,
  kIoError,
};

PullResult PullMissingSendEntry(int port, uint64_t uuid) {
  ScopedFd fd(socket(AF_INET6, SOCK_STREAM, 0));
  if (fd.get() < 0) return PullResult::kIoError;

  timeval timeout = {};
  timeout.tv_usec = 500000;
  if (setsockopt(fd.get(), SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) <
      0) {
    return PullResult::kIoError;
  }

  sockaddr_in6 address = {};
  address.sin6_family = AF_INET6;
  address.sin6_addr = in6addr_loopback;
  address.sin6_port = htons(port);
  if (connect(fd.get(), reinterpret_cast<sockaddr*>(&address),
              sizeof(address)) < 0) {
    return PullResult::kIoError;
  }

  TestManager::ControlRequestHeader request;
  request.op = TestManager::kOpPullStream;
  request.uuid = uuid;
  request.num_blocks = 1;
  const int64_t block_id = 0;
  if (!WriteExact(fd.get(), &request, sizeof(request)) ||
      !WriteExact(fd.get(), &block_id, sizeof(block_id)) ||
      !WriteExact(fd.get(), &block_id, sizeof(block_id))) {
    return PullResult::kIoError;
  }

  TestManager::ControlResponseHeader response;
  ssize_t bytes_read;
  do {
    bytes_read = recv(fd.get(), &response, sizeof(response), MSG_WAITALL);
  } while (bytes_read < 0 && errno == EINTR);
  if (bytes_read < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return PullResult::kTimedOut;
  }
  // The producer rejects before consuming the block-id body. Depending on TCP
  // timing, its close can surface as either an error response or a reset.
  if (bytes_read == 0 || (bytes_read < 0 && errno == ECONNRESET)) {
    return PullResult::kRejected;
  }
  if (bytes_read != static_cast<ssize_t>(sizeof(response)) ||
      response.magic != TestManager::kResponseMagic) {
    return PullResult::kIoError;
  }
  return response.status == 0 ? PullResult::kAccepted : PullResult::kRejected;
}

TEST(KVCacheManagerWithTransferControlTest, UnknownPullIsRejectedImmediately) {
  TestManager manager(/*timeout_s=*/120.0);
  EXPECT_EQ(PullMissingSendEntry(manager.local_control_port(), 99),
            PullResult::kRejected);
}

TEST(KVCacheManagerWithTransferControlTest,
     ExpiredPullsDoNotExhaustControlWorkers) {
  constexpr uint64_t kFirstUuid = 100;
  constexpr size_t kStalePullCount = 8;
  TestManager manager(/*timeout_s=*/0.05);
  for (uint64_t uuid = kFirstUuid; uuid < kFirstUuid + kStalePullCount;
       ++uuid) {
    ASSERT_GT(manager.NotifyForRead(std::to_string(uuid), uuid, {0}), 0);
  }

  absl::SleepFor(absl::Milliseconds(100));
  const auto completed = manager.CompleteReadRaw();
  ASSERT_TRUE(std::get<0>(completed).empty());
  ASSERT_EQ(std::get<2>(completed).size(), kStalePullCount);

  std::vector<std::future<PullResult>> pulls;
  pulls.reserve(kStalePullCount);
  for (uint64_t uuid = kFirstUuid; uuid < kFirstUuid + kStalePullCount;
       ++uuid) {
    pulls.push_back(std::async(std::launch::async, [&manager, uuid] {
      return PullMissingSendEntry(manager.local_control_port(), uuid);
    }));
  }
  for (auto& pull : pulls) {
    EXPECT_EQ(pull.get(), PullResult::kRejected);
  }

  EXPECT_EQ(PullMissingSendEntry(manager.local_control_port(),
                                 kFirstUuid + kStalePullCount),
            PullResult::kRejected);
}

}  // namespace
}  // namespace tpu_raiden
