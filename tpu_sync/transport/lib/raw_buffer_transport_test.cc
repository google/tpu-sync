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

#include <signal.h>
#include <unistd.h>

#include <chrono>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/support/channel_arguments.h"
#include "peregrine/src/api/socket_util.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/transport/buffer_push_task.h"
#include "tpu_sync/transport/lib/chunk.h"
#include "tpu_sync/transport/lib/chunk_serializer.h"
#include "tpu_sync/transport/lib/conn/pool.h"
#include "tpu_sync/transport/lib/peregrine_control_service.h"
#include "tpu_sync/transport/lib/raw_buffer_transport_delegate.h"
#include "tpu_sync/transport/lib/socket/psp_syscall_mock.h" // NOLINT
#include "tpu_sync/transport/lib/socket/tcp_psp_helper.h"
#include "tpu_sync/transport/lib/socket/util.h"
#include "tpu_sync/transport/lib/transport_adapter.h"

namespace tpu_raiden::transport::lib {
namespace {

bool AllZero(absl::Span<const uint8_t> data) {
  for (uint8_t b : data) {
    if (b != 0) return false;
  }
  return true;
}

void RandomNonZero(absl::Span<uint8_t> data) {
  thread_local std::mt19937 gen(std::random_device{}());
  std::uniform_int_distribution<int> dist(1, 255);
  for (uint8_t& b : data) {
    b = static_cast<uint8_t>(dist(gen));
  }
}

using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Ne;
using ::testing::Pointwise;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;

constexpr int kLocalPort = 0;
constexpr size_t kBufferId = 0;
constexpr size_t kSrcShardIdx = 0;
constexpr size_t kDstShardIdx = 0;

// Generous upper bound for the receiver worker thread to run the delegate
// notifications, which happen after the batch is acknowledged on the wire.
constexpr absl::Duration kNotificationTimeout = absl::Seconds(30);

// Settling time used by negative assertions ("no further notification fires").
constexpr absl::Duration kQuiesceDelay = absl::Milliseconds(200);

std::string GetIpPort(const RawBufferTransport& transport) {
  return "localhost:" + std::to_string(transport.local_port());
}

class RawMockDelegate : public RawBufferTransportDelegate {
 public:
  explicit RawMockDelegate(size_t buffer_size) : buffer_(buffer_size, 0) {
    DCHECK(AllZero(buffer_));
  }

  uint8_t* GetHostPointer(size_t buffer_id, size_t shard_idx) override {
    return buffer_.data();
  }

  size_t GetHostSize(size_t buffer_id, size_t shard_idx) override {
    return buffer_.size();
  }

  absl::Status OnDataReceived(uint64_t uuid = 0) override {
    std::function<void(uint64_t)> cb;
    {
      absl::MutexLock lock(mu_);
      on_data_received_called_ = true;
      cb = on_data_received_callback_;
    }
    if (cb) {
      cb(uuid);
    }
    return absl::OkStatus();
  }

  void SetOnDataReceivedCallback(std::function<void(uint64_t)> cb) {
    absl::MutexLock lock(mu_);
    on_data_received_callback_ = std::move(cb);
  }

  absl::Status OnLayerDataReceived(size_t layer_idx,
                                   uint64_t uuid = 0) override {
    absl::MutexLock lock(mu_);
    layer_triggers_.push_back(layer_idx);
    return absl::OkStatus();
  }

  void SetPeerChannel(absl::string_view peer,
                      std::shared_ptr<grpc::Channel> channel) {
    absl::MutexLock lock(mu_);
    peer_channels_[std::string(peer)] = std::move(channel);
  }

  std::shared_ptr<grpc::Channel> GetPeregrineChannel(
      absl::string_view peer) override {
    absl::MutexLock lock(mu_);
    auto it = peer_channels_.find(peer);
    if (it != peer_channels_.end()) {
      return it->second;
    }
    return default_channel_;
  }

  uint8_t* data() { return buffer_.data(); }
  absl::Span<uint8_t> DataSpan() { return absl::MakeSpan(buffer_); }
  absl::Span<const uint8_t> DataSpan(size_t offset, size_t length) {
    return absl::MakeConstSpan(buffer_.data() + offset, length);
  }

  bool on_data_received() const {
    absl::MutexLock lock(mu_);
    return on_data_received_called_;
  }

  // Layer indices passed to `OnLayerDataReceived()`, in notification order.
  std::vector<size_t> layer_triggers() const {
    absl::MutexLock lock(mu_);
    return layer_triggers_;
  }

  // Waits until at least `count` layer notifications have been observed.
  // Returns false on timeout.
  bool WaitForLayerTriggers(size_t count, absl::Duration timeout) const {
    const auto reached = [this, count]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
      return layer_triggers_.size() >= count;
    };
    absl::MutexLock lock(mu_);
    return mu_.AwaitWithTimeout(absl::Condition(&reached), timeout);
  }

  // Waits until `OnDataReceived()` has been called. Returns false on timeout.
  bool WaitForDataReceived(absl::Duration timeout) const {
    const auto received = [this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
      return on_data_received_called_;
    };
    absl::MutexLock lock(mu_);
    return mu_.AwaitWithTimeout(absl::Condition(&received), timeout);
  }

 private:
  std::vector<uint8_t> buffer_;
  mutable absl::Mutex mu_;
  bool on_data_received_called_ ABSL_GUARDED_BY(mu_) = false;
  std::function<void(uint64_t)> on_data_received_callback_ ABSL_GUARDED_BY(mu_);
  std::vector<size_t> layer_triggers_ ABSL_GUARDED_BY(mu_);
  std::shared_ptr<grpc::Channel> default_channel_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, std::shared_ptr<grpc::Channel>>
      peer_channels_ ABSL_GUARDED_BY(mu_);
};

// Builds one single-byte push task per entry of `layer_ids`, sending
// `payload[i]` to destination offset `i` so that chunks of different layers do
// not overlap in the (shared) mock destination buffer.
std::vector<BufferPushTask> MakeSingleBytePushTasks(
    absl::string_view dst_addr, absl::Span<const size_t> layer_ids,
    const std::vector<uint8_t>& payload) {
  CHECK_EQ(layer_ids.size(), payload.size());
  std::vector<BufferPushTask> tasks;
  tasks.reserve(layer_ids.size());
  for (size_t i = 0; i < layer_ids.size(); ++i) {
    tasks.push_back({
        .peer = std::string(dst_addr),
        .buffer_id = layer_ids[i],
        .dst_shard_idx = kDstShardIdx,
        .dst_offset_bytes = i,
        .data_ptr = &payload[i],
        .size_bytes = 1,
    });
  }
  return tasks;
}

class RawBufferTransportTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    if (GetParam() && !IsPspSupported()) {
      GTEST_SKIP() << "PSP-TCP is unimplemented.";
    }
    absl::SetFlag(&FLAGS_require_psp_tcp, GetParam());
  }

  void TearDown() override {
    for (auto& entry : servers_) {
      if (entry.server) {
        entry.server->Shutdown();
      }
    }
    servers_.clear();
  }

  std::shared_ptr<grpc::Channel> StartControlServer(
      RawBufferTransport* transport) {
    auto service =
        std::make_unique<PeregrineControlServiceImpl>(transport);
    grpc::ServerBuilder builder;
    builder.RegisterService(service.get());
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    std::shared_ptr<grpc::Channel> channel =
        server->InProcessChannel(grpc::ChannelArguments());
    servers_.push_back({std::move(service), std::move(server), channel});
    return channel;
  }

  void BindControlChannels(RawBufferTransport* transport1,
                           RawMockDelegate* delegate1,
                           RawBufferTransport* transport2,
                           RawMockDelegate* delegate2) {
    auto ch1 = StartControlServer(transport1);
    auto ch2 = StartControlServer(transport2);
    delegate1->SetPeerChannel(GetIpPort(*transport2), ch2);
    delegate2->SetPeerChannel(GetIpPort(*transport1), ch1);
  }

 private:
  struct ServerEntry {
    std::unique_ptr<PeregrineControlServiceImpl> service;
    std::unique_ptr<grpc::Server> server;
    std::shared_ptr<grpc::Channel> channel;
  };
  std::vector<ServerEntry> servers_;
};

using ConnPoolTest = RawBufferTransportTest;

TEST_P(RawBufferTransportTest, PullBufferCorrectness) {
  // Set up src/dst buffers.
  constexpr size_t size = 64 * 1024;
  RawMockDelegate src(size);
  RawMockDelegate dst(size);
  RandomNonZero(src.DataSpan());

  // Pre-condition: all the dst bytes are not equal to the src.
  ASSERT_THAT(dst.DataSpan(), Pointwise(Ne(), src.DataSpan()));

  // Create two transports.
  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Pull a buffer segment from src to dst.
  const std::string src_addr = GetIpPort(src_transport);
  constexpr size_t kLen = 62 * 1024;
  constexpr size_t kSrcOffset = 512;
  constexpr size_t kDstOffset = 1024;
  const auto pull_res =
      dst_transport.PullBuffer(src_addr, kBufferId, kSrcShardIdx, kSrcOffset,
                               kDstShardIdx, kDstOffset, kLen);
  ASSERT_OK(pull_res) << pull_res.message();

  // Post-condition: only the copied dst bytes are equal to the src.
  EXPECT_THAT(dst.DataSpan(0, kDstOffset), Each(Eq(0)));
  EXPECT_THAT(dst.DataSpan(kDstOffset, kLen),
              Pointwise(Eq(), src.DataSpan(kSrcOffset, kLen)));
  EXPECT_THAT(dst.DataSpan(kDstOffset + kLen, size - kDstOffset - kLen),
              Each(Eq(0)));
}

TEST_P(RawBufferTransportTest, PushBuffersCorrectness) {
  // Set up src/dst buffers.
  constexpr size_t size = 128 * 1024;
  RawMockDelegate src(size);
  RawMockDelegate dst1(size);
  RawMockDelegate dst2(size);

  // Create transports.
  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport1(&dst1, kLocalPort);
  RawBufferTransport dst_transport2(&dst2, kLocalPort);
  auto ch1 = StartControlServer(&dst_transport1);
  auto ch2 = StartControlServer(&dst_transport2);
  const std::string dst1_addr = GetIpPort(dst_transport1);
  const std::string dst2_addr = GetIpPort(dst_transport2);
  src.SetPeerChannel(dst1_addr, ch1);
  src.SetPeerChannel(dst2_addr, ch2);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Prepare multiple payloads.
  std::vector<uint8_t> payload1(1024);
  std::vector<uint8_t> payload2(2048);
  std::vector<uint8_t> payload3(4096);
  std::vector<uint8_t> payload4(8192);
  RandomNonZero(absl::MakeSpan(payload1));
  RandomNonZero(absl::MakeSpan(payload2));
  RandomNonZero(absl::MakeSpan(payload3));
  RandomNonZero(absl::MakeSpan(payload4));

  uint64_t uuid = 12345;
  // dst1 expects 2 chunks, dst2 expects 2 chunks.
  ASSERT_OK(dst_transport1.RegisterExpectedChunks(uuid, 2));
  ASSERT_OK(dst_transport2.RegisterExpectedChunks(uuid, 2));

  // Interleave tasks between dst1 and dst2 to test sorting.
  std::vector<BufferPushTask> tasks = {
      {.peer = dst1_addr,
       .buffer_id = kBufferId,
       .dst_shard_idx = kDstShardIdx,
       .dst_offset_bytes = 0,
       .data_ptr = payload1.data(),
       .size_bytes = payload1.size()},
      {.peer = dst2_addr,
       .buffer_id = kBufferId,
       .dst_shard_idx = kDstShardIdx,
       .dst_offset_bytes = 0,
       .data_ptr = payload2.data(),
       .size_bytes = payload2.size()},
      {.peer = dst1_addr,
       .buffer_id = kBufferId,
       .dst_shard_idx = kDstShardIdx,
       .dst_offset_bytes = 4096,
       .data_ptr = payload3.data(),
       .size_bytes = payload3.size()},
      {.peer = dst2_addr,
       .buffer_id = kBufferId,
       .dst_shard_idx = kDstShardIdx,
       .dst_offset_bytes = 8192,
       .data_ptr = payload4.data(),
       .size_bytes = payload4.size()},
  };

  const auto push_res =
      src_transport.PushBuffers(tasks, /*parallelism=*/2, uuid);
  EXPECT_OK(push_res) << push_res.message();

  // Post-condition: check payloads at correct offsets for dst1.
  EXPECT_THAT(dst1.DataSpan(0, payload1.size()),
              Pointwise(Eq(), absl::MakeConstSpan(payload1)));
  EXPECT_THAT(dst1.DataSpan(4096, payload3.size()),
              Pointwise(Eq(), absl::MakeConstSpan(payload3)));
  EXPECT_TRUE(dst1.WaitForDataReceived(kNotificationTimeout));

  // Post-condition: check payloads at correct offsets for dst2.
  EXPECT_THAT(dst2.DataSpan(0, payload2.size()),
              Pointwise(Eq(), absl::MakeConstSpan(payload2)));
  EXPECT_THAT(dst2.DataSpan(8192, payload4.size()),
              Pointwise(Eq(), absl::MakeConstSpan(payload4)));
  EXPECT_TRUE(dst2.WaitForDataReceived(kNotificationTimeout));
}

TEST_P(RawBufferTransportTest, PushBuffersPreservesFirstSeenPeerOrder) {
  constexpr size_t size = 16 * 1024;
  RawMockDelegate src(size);
  RawMockDelegate dst1(size);
  RawMockDelegate dst2(size);

  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport1(&dst1, kLocalPort);
  RawBufferTransport dst_transport2(&dst2, kLocalPort);
  auto ch1 = StartControlServer(&dst_transport1);
  auto ch2 = StartControlServer(&dst_transport2);
  const std::string dst1_addr = GetIpPort(dst_transport1);
  const std::string dst2_addr = GetIpPort(dst_transport2);
  src.SetPeerChannel(dst1_addr, ch1);
  src.SetPeerChannel(dst2_addr, ch2);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::vector<uint8_t> payload1(1024);
  std::vector<uint8_t> payload2(1024);
  RandomNonZero(absl::MakeSpan(payload1));
  RandomNonZero(absl::MakeSpan(payload2));

  // Determine which peer address is lexicographically larger.
  // We place the lexicographically LARGER peer first in `tasks`.
  // Alphabetical sorting would schedule the smaller peer first.
  // First-seen ordering must schedule the larger peer first.
  const bool dst1_is_larger = dst1_addr > dst2_addr;
  const std::string& first_peer = dst1_is_larger ? dst1_addr : dst2_addr;
  const std::string& second_peer = dst1_is_larger ? dst2_addr : dst1_addr;
  RawMockDelegate& first_delegate = dst1_is_larger ? dst1 : dst2;
  RawMockDelegate& second_delegate = dst1_is_larger ? dst2 : dst1;
  RawBufferTransport& first_transport =
      dst1_is_larger ? dst_transport1 : dst_transport2;
  RawBufferTransport& second_transport =
      dst1_is_larger ? dst_transport2 : dst_transport1;

  constexpr uint64_t uuid = 998877;
  ASSERT_OK(first_transport.RegisterExpectedChunks(uuid, 2));
  ASSERT_OK(second_transport.RegisterExpectedChunks(uuid, 2));

  absl::Mutex log_mu;
  std::vector<std::string> received_order;
  first_delegate.SetOnDataReceivedCallback([&](uint64_t) {
    absl::MutexLock lock(log_mu);
    received_order.push_back(first_peer);
  });
  second_delegate.SetOnDataReceivedCallback([&](uint64_t) {
    absl::MutexLock lock(log_mu);
    received_order.push_back(second_peer);
  });

  // Interleave tasks starting with `first_peer`.
  std::vector<BufferPushTask> tasks = {
      {.peer = first_peer,
       .buffer_id = kBufferId,
       .dst_shard_idx = kDstShardIdx,
       .dst_offset_bytes = 0,
       .data_ptr = payload1.data(),
       .size_bytes = payload1.size()},
      {.peer = second_peer,
       .buffer_id = kBufferId,
       .dst_shard_idx = kDstShardIdx,
       .dst_offset_bytes = 0,
       .data_ptr = payload2.data(),
       .size_bytes = payload2.size()},
      {.peer = first_peer,
       .buffer_id = kBufferId,
       .dst_shard_idx = kDstShardIdx,
       .dst_offset_bytes = 2048,
       .data_ptr = payload1.data(),
       .size_bytes = payload1.size()},
      {.peer = second_peer,
       .buffer_id = kBufferId,
       .dst_shard_idx = kDstShardIdx,
       .dst_offset_bytes = 2048,
       .data_ptr = payload2.data(),
       .size_bytes = payload2.size()},
  };

  const auto push_res =
      src_transport.PushBuffers(tasks, /*parallelism=*/1, uuid);
  ASSERT_OK(push_res) << push_res.message();

  ASSERT_TRUE(first_delegate.WaitForDataReceived(kNotificationTimeout));
  ASSERT_TRUE(second_delegate.WaitForDataReceived(kNotificationTimeout));

  // Verify that `first_peer` (lexicographically greater) received its batch
  // and triggered OnDataReceived before `second_peer`.
  {
    absl::MutexLock lock(log_mu);
    EXPECT_THAT(received_order, ElementsAre(first_peer, second_peer));
  }
}

TEST_P(RawBufferTransportTest, RejectsOutOfBounds) {
  // Set up src/dst buffers.
  constexpr size_t size = 1024;
  RawMockDelegate src(size);
  RawMockDelegate dst(size);

  // Create two transports.
  RawBufferTransport src_transport(&src, 0);
  RawBufferTransport dst_transport(&dst, 0);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Pulling an out-of-bounds buffer segment from src to dst should fail.
  const std::string src_addr = GetIpPort(src_transport);
  constexpr size_t kSrcOffset = 0;
  constexpr size_t kDstOffset = size / 2;
  constexpr size_t kLen = size / 2 + 1;
  static_assert(kDstOffset + kLen > size);
  const auto pull_res =
      dst_transport.PullBuffer(src_addr, kBufferId, kSrcShardIdx, kSrcOffset,
                               kDstShardIdx, kDstOffset, kLen);
  EXPECT_FALSE(pull_res.ok()) << pull_res.message();
}

TEST_P(RawBufferTransportTest, RejectsWrappingAndOutOfBoundsPushAndPull) {
  // Set up src/dst buffers.
  constexpr size_t size = 1024;
  RawMockDelegate src(size);
  RawMockDelegate dst(size);

  // Create two transports.
  RawBufferTransport src_transport(&src, 0);
  RawBufferTransport dst_transport(&dst, 0);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  const std::string dst_addr = GetIpPort(dst_transport);
  const std::string src_addr = GetIpPort(src_transport);

  // Test wrapping offset:
  // 0xFFFFF000 (about 4 GiB, 8 KiB size): wraps around 32 bits if checked
  // with 32-bit addition (0xFFFFF000 + 0x2000 = 0x1000 <= host_size).
  constexpr uint32_t wrap_offset = 0xFFFFF000;
  constexpr uint32_t kWrapSize = 0x2000;

  // 1. Verify push with wrapping offset + size is rejected on the wire.
  {
    auto client_fd_or = ConnectToPeer(dst_addr);
    ASSERT_OK(client_fd_or);
    const int client_fd = *client_fd_or;

    ChunkHeader header = {};
    header.version = 1;
    header.op = kOpBufferPush;
    header.buffer_id = 0;
    header.remote_id = wrap_offset;
    header.local_id = 0;
    header.count_or_size = kWrapSize;
    header.uuid = 42;

    const auto s_header = SerializeChunkHeader(header);
    ASSERT_OK(
        ::peregrine::WriteExact(client_fd, s_header.data(), s_header.size()));

    // Server should reject destination out of bounds and close socket without
    // reading/writing payload or sending ACK=1.
    uint8_t ack = 0;
    const auto read_res = ::peregrine::ReadExact(client_fd, &ack, 1);
    EXPECT_FALSE(read_res.ok());
    close(client_fd);
  }

  // 2. Verify pull with wrapping offset + size is rejected on the wire.
  {
    auto client_fd_or = ConnectToPeer(src_addr);
    ASSERT_OK(client_fd_or);
    const int client_fd = *client_fd_or;

    ChunkHeader header = {};
    header.version = 1;
    header.op = kOpBufferPull;
    header.buffer_id = 0;
    header.remote_id = wrap_offset;
    header.local_id = 0;
    header.count_or_size = kWrapSize;
    header.uuid = 43;

    const auto s_header = SerializeChunkHeader(header);
    ASSERT_OK(
        ::peregrine::WriteExact(client_fd, s_header.data(), s_header.size()));

    // Server should reject source out of bounds and close socket without
    // sending data.
    std::vector<uint8_t> dummy(kWrapSize);
    const auto read_res =
        ::peregrine::ReadExact(client_fd, dummy.data(), dummy.size());
    EXPECT_FALSE(read_res.ok());
    close(client_fd);
  }

  // 3. Verify PullBuffer with wrapping destination offset is rejected.
  const auto pull_res = dst_transport.PullBuffer(
      src_addr, kBufferId, kSrcShardIdx, /*src_offset_bytes=*/0, kDstShardIdx,
      /*dst_offset_bytes=*/wrap_offset, /*size_bytes=*/kWrapSize);
  EXPECT_FALSE(pull_res.ok());
  EXPECT_THAT(pull_res.message(), ::testing::HasSubstr("out of bounds"));
}

TEST_P(ConnPoolTest, MultiIpPoolingIsolation) {
  // Set up src/dst buffers.
  RawMockDelegate src(1024);

  // Create a transport to serve as the peer.
  RawBufferTransport listener(&src, 0);
  auto channel = StartControlServer(&listener);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Create a ConnPool.
  ConnPool pool;
  const std::string addr = GetIpPort(listener);
  const bool require_psp = absl::GetFlag(FLAGS_require_psp_tcp);

  // 1. Borrow connection with local_ip = "127.0.0.1"
  const auto fd1_or = pool.Borrow(addr, "127.0.0.1", require_psp, channel);
  ASSERT_OK(fd1_or) << fd1_or.status().message();
  const int fd1 = fd1_or.value();

  // Return it. It should be pooled under "127.0.0.1->peer1".
  pool.Return(/*ok=*/true, fd1, addr, "127.0.0.1");

  // 2. Borrow connection with local_ip = "127.0.0.2"
  // This should NOT reuse fd1 because it's a different local IP.
  const auto fd2_or = pool.Borrow(addr, "127.0.0.2", require_psp, channel);
  ASSERT_OK(fd2_or) << fd2_or.status().message();
  const int fd2 = fd2_or.value();
  EXPECT_NE(fd1, fd2);

  // Return it. It should be pooled under "127.0.0.2->peer1".
  pool.Return(/*ok=*/true, fd2, addr, "127.0.0.2");

  // 3. Borrow connection with local_ip = "127.0.0.1" again.
  // This SHOULD reuse fd1.
  const auto fd3_or = pool.Borrow(addr, "127.0.0.1", require_psp, channel);
  ASSERT_OK(fd3_or) << fd3_or.status().message();
  const int fd3 = fd3_or.value();
  EXPECT_EQ(fd1, fd3);
  pool.Return(/*ok=*/true, fd3, addr, "127.0.0.1");

  // 4. Borrow connection with local_ip = "127.0.0.2" again.
  // This SHOULD reuse fd2.
  const auto fd4_or = pool.Borrow(addr, "127.0.0.2", require_psp, channel);
  ASSERT_OK(fd4_or) << fd4_or.status().message();
  const int fd4 = fd4_or.value();
  EXPECT_EQ(fd2, fd4);
  pool.Return(/*ok=*/true, fd4, addr, "127.0.0.2");

  // Close the pool.
  pool.Close();
}

TEST_P(RawBufferTransportTest, PushBuffersCoalescedCorrectness) {
  // Set up src/dst buffers.
  constexpr size_t size = 128 * 1024;
  RawMockDelegate src(size);
  RawMockDelegate dst1(size);
  RawMockDelegate dst2(size);

  // Create transports. Pass 4096 to enable coalescing for the sender.
  RawBufferTransport src_transport(&src, kLocalPort, /*local_ips=*/{},
                                    /*custom_request_handler=*/nullptr,
                                    /*coalesce_window_bytes=*/4096);
  RawBufferTransport dst_transport1(&dst1, kLocalPort);
  RawBufferTransport dst_transport2(&dst2, kLocalPort);
  auto ch1 = StartControlServer(&dst_transport1);
  auto ch2 = StartControlServer(&dst_transport2);
  const std::string dst1_addr = GetIpPort(dst_transport1);
  const std::string dst2_addr = GetIpPort(dst_transport2);
  src.SetPeerChannel(dst1_addr, ch1);
  src.SetPeerChannel(dst2_addr, ch2);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Prepare multiple payloads.
  std::vector<uint8_t> payload1(1024);
  std::vector<uint8_t> payload2(2048);
  std::vector<uint8_t> payload3(4096);
  std::vector<uint8_t> payload4(8192);
  RandomNonZero(absl::MakeSpan(payload1));
  RandomNonZero(absl::MakeSpan(payload2));
  RandomNonZero(absl::MakeSpan(payload3));
  RandomNonZero(absl::MakeSpan(payload4));

  uint64_t uuid = 12345;
  // dst1 expects 2 chunks, dst2 expects 2 chunks.
  ASSERT_OK(dst_transport1.RegisterExpectedChunks(uuid, 2));
  ASSERT_OK(dst_transport2.RegisterExpectedChunks(uuid, 2));

  // Interleave tasks between dst1 and dst2 to test sorting.
  std::vector<BufferPushTask> tasks = {
      {.peer = dst1_addr,
       .buffer_id = kBufferId,
       .dst_shard_idx = kDstShardIdx,
       .dst_offset_bytes = 0,
       .data_ptr = payload1.data(),
       .size_bytes = payload1.size()},
      {.peer = dst2_addr,
       .buffer_id = kBufferId,
       .dst_shard_idx = kDstShardIdx,
       .dst_offset_bytes = 0,
       .data_ptr = payload2.data(),
       .size_bytes = payload2.size()},
      {.peer = dst1_addr,
       .buffer_id = kBufferId,
       .dst_shard_idx = kDstShardIdx,
       .dst_offset_bytes = 4096,
       .data_ptr = payload3.data(),
       .size_bytes = payload3.size()},
      {.peer = dst2_addr,
       .buffer_id = kBufferId,
       .dst_shard_idx = kDstShardIdx,
       .dst_offset_bytes = 8192,
       .data_ptr = payload4.data(),
       .size_bytes = payload4.size()},
  };

  const auto push_res =
      src_transport.PushBuffers(tasks, /*parallelism=*/2, uuid);
  EXPECT_OK(push_res) << push_res.message();

  // Post-condition: check payloads at correct offsets for dst1.
  EXPECT_THAT(dst1.DataSpan(0, payload1.size()),
              Pointwise(Eq(), absl::MakeConstSpan(payload1)));
  EXPECT_THAT(dst1.DataSpan(4096, payload3.size()),
              Pointwise(Eq(), absl::MakeConstSpan(payload3)));
  EXPECT_TRUE(dst1.WaitForDataReceived(kNotificationTimeout));

  // Post-condition: check payloads at correct offsets for dst2.
  EXPECT_THAT(dst2.DataSpan(0, payload2.size()),
              Pointwise(Eq(), absl::MakeConstSpan(payload2)));
  EXPECT_THAT(dst2.DataSpan(8192, payload4.size()),
              Pointwise(Eq(), absl::MakeConstSpan(payload4)));
  EXPECT_TRUE(dst2.WaitForDataReceived(kNotificationTimeout));
}

TEST_P(RawBufferTransportTest, PushBuffersLargeBatchCorrectness) {
  constexpr size_t num_tasks = 1025;  // IOV_MAX (1024) + 1
  constexpr size_t buffer_size = num_tasks;

  RawMockDelegate src(buffer_size);
  RawMockDelegate dst(buffer_size);

  // Create transports. Coalescing is disabled by default (0).
  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  const std::string dst_addr = GetIpPort(dst_transport);

  std::vector<uint8_t> payload(num_tasks);
  RandomNonZero(absl::MakeSpan(payload));

  uint64_t uuid = 99999;
  ASSERT_OK(dst_transport.RegisterExpectedChunks(uuid, num_tasks));

  std::vector<BufferPushTask> tasks;
  tasks.reserve(num_tasks);
  for (size_t i = 0; i < num_tasks; ++i) {
    tasks.push_back({
        .peer = dst_addr,
        .buffer_id = kBufferId,
        .dst_shard_idx = kDstShardIdx,
        .dst_offset_bytes = i,
        .data_ptr = &payload[i],
        .size_bytes = 1,
    });
  }

  const auto push_res =
      src_transport.PushBuffers(tasks, /*parallelism=*/1, uuid);
  EXPECT_OK(push_res) << push_res.message();

  // Verify all bytes were received.
  EXPECT_THAT(dst.DataSpan(0, num_tasks),
              Pointwise(Eq(), absl::MakeConstSpan(payload)));
  EXPECT_TRUE(dst.WaitForDataReceived(kNotificationTimeout));
}

TEST_P(RawBufferTransportTest,
       PushBuffersTriggersOnDataReceivedWhenChunksExceedOrEqualExpected) {
  constexpr size_t num_tasks = 4;
  constexpr size_t buffer_size = num_tasks;

  RawMockDelegate src(buffer_size);
  RawMockDelegate dst(buffer_size);

  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  const std::string dst_addr = GetIpPort(dst_transport);

  std::vector<uint8_t> payload(num_tasks);
  RandomNonZero(absl::MakeSpan(payload));

  uint64_t uuid = 12345;
  // Register 3 expected chunks, but push 4 chunks (completed >= expected).
  ASSERT_OK(dst_transport.RegisterExpectedChunks(uuid, 3));

  std::vector<BufferPushTask> tasks;
  tasks.reserve(num_tasks);
  for (size_t i = 0; i < num_tasks; ++i) {
    tasks.push_back({
        .peer = dst_addr,
        .buffer_id = kBufferId,
        .dst_shard_idx = kDstShardIdx,
        .dst_offset_bytes = i,
        .data_ptr = &payload[i],
        .size_bytes = 1,
    });
  }

  const auto push_res =
      src_transport.PushBuffers(tasks, /*parallelism=*/1, uuid);
  EXPECT_OK(push_res) << push_res.message();

  EXPECT_THAT(dst.DataSpan(0, num_tasks),
              Pointwise(Eq(), absl::MakeConstSpan(payload)));
  EXPECT_TRUE(dst.WaitForDataReceived(kNotificationTimeout));
}

TEST_P(RawBufferTransportTest, PushBuffersStridedCorrectness) {
  constexpr size_t slice_size = 128;
  constexpr size_t count = 1050;  // Exceeds IOV_MAX (1024)
  constexpr size_t dst_stride = 512;
  constexpr size_t total_payload_bytes = count * slice_size;
  constexpr size_t dst_buffer_size = (count - 1) * dst_stride + slice_size;

  RawMockDelegate src(total_payload_bytes);
  RawMockDelegate dst(dst_buffer_size);

  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  const std::string dst_addr = GetIpPort(dst_transport);

  std::vector<uint8_t> payload(total_payload_bytes);
  RandomNonZero(absl::MakeSpan(payload));

  uint64_t uuid = 88888;
  ASSERT_OK(dst_transport.RegisterExpectedChunks(uuid, 1));

  std::vector<BufferPushTask> tasks = {
      BufferPushTask{
          .peer = dst_addr,
          .buffer_id = kBufferId,
          .dst_shard_idx = kDstShardIdx,
          .dst_offset_bytes = 0,
          .data_ptr = payload.data(),
          .size_bytes = slice_size,
          .dst_stride_bytes = dst_stride,
          .count = count,
      },
  };

  const auto push_res =
      src_transport.PushBuffers(tasks, /*parallelism=*/1, uuid);
  EXPECT_OK(push_res) << push_res.message();

  for (size_t c = 0; c < count; ++c) {
    EXPECT_THAT(dst.DataSpan(c * dst_stride, slice_size),
                Pointwise(Eq(), absl::MakeConstSpan(payload).subspan(
                                    c * slice_size, slice_size)))
        << "Mismatch at slice " << c;
  }
  EXPECT_TRUE(dst.WaitForDataReceived(kNotificationTimeout));
}

TEST(BufferPushTaskTest, DefaultInitializers) {
  BufferPushTask task;
  EXPECT_EQ(task.peer, "");
  EXPECT_EQ(task.buffer_id, 0);
  EXPECT_EQ(task.dst_shard_idx, 0);
  EXPECT_EQ(task.dst_offset_bytes, 0);
  EXPECT_EQ(task.data_ptr, nullptr);
  EXPECT_EQ(task.size_bytes, 0);
  EXPECT_EQ(task.dst_stride_bytes, 0);
  EXPECT_EQ(task.count, 1);
  EXPECT_EQ(task.src_stride_bytes, 0);
}

TEST_P(RawBufferTransportTest,
       PushBuffersSourceStridedToContiguousCorrectness) {
  constexpr size_t slice_size = 128;
  constexpr size_t count = 1050;  // Exceeds IOV_MAX (1024)
  constexpr size_t src_stride = 512;
  constexpr size_t src_buffer_size = (count - 1) * src_stride + slice_size;
  constexpr size_t dst_buffer_size = count * slice_size;

  RawMockDelegate src(src_buffer_size);
  RawMockDelegate dst(dst_buffer_size);

  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  const std::string dst_addr = GetIpPort(dst_transport);

  std::vector<uint8_t> src_payload(src_buffer_size, 0x00);
  for (size_t c = 0; c < count; ++c) {
    RandomNonZero(
        absl::MakeSpan(src_payload).subspan(c * src_stride, slice_size));
  }

  uint64_t uuid = 88889;
  // Sent as a single ChunkMetadata (1 expected chunk) despite strided source.
  ASSERT_OK(dst_transport.RegisterExpectedChunks(uuid, 1));

  std::vector<BufferPushTask> tasks = {
      BufferPushTask{
          .peer = dst_addr,
          .buffer_id = kBufferId,
          .dst_shard_idx = kDstShardIdx,
          .dst_offset_bytes = 0,
          .data_ptr = src_payload.data(),
          .size_bytes = slice_size,
          .dst_stride_bytes = slice_size,
          .count = count,
          .src_stride_bytes = src_stride,
      },
  };

  const auto push_res =
      src_transport.PushBuffers(tasks, /*parallelism=*/1, uuid);
  EXPECT_OK(push_res) << push_res.message();

  for (size_t c = 0; c < count; ++c) {
    EXPECT_THAT(dst.DataSpan(c * slice_size, slice_size),
                Pointwise(Eq(), absl::MakeConstSpan(src_payload)
                                    .subspan(c * src_stride, slice_size)))
        << "Mismatch at slice " << c;
  }
  EXPECT_TRUE(dst.WaitForDataReceived(kNotificationTimeout));
}

TEST_P(RawBufferTransportTest, PushBuffersBothSourceAndDestStridedCorrectness) {
  constexpr size_t slice_size = 64;
  constexpr size_t count = 32;
  constexpr size_t src_stride = 256;
  constexpr size_t dst_stride = 192;
  constexpr size_t src_buffer_size = (count - 1) * src_stride + slice_size;
  constexpr size_t dst_buffer_size = (count - 1) * dst_stride + slice_size;

  for (size_t coalesce_window : {size_t{0}, size_t{4096}}) {
    RawMockDelegate src(src_buffer_size);
    RawMockDelegate dst(dst_buffer_size);

    RawBufferTransport src_transport(&src, kLocalPort, /*local_ips=*/{},
                                     /*custom_request_handler=*/nullptr,
                                     coalesce_window);
    RawBufferTransport dst_transport(&dst, kLocalPort);
    BindControlChannels(&src_transport, &src, &dst_transport, &dst);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    const std::string dst_addr = GetIpPort(dst_transport);

    std::vector<uint8_t> src_payload(src_buffer_size, 0x11);
    for (size_t c = 0; c < count; ++c) {
      RandomNonZero(
          absl::MakeSpan(src_payload).subspan(c * src_stride, slice_size));
    }

    uint64_t uuid = 88890 + coalesce_window;
    ASSERT_OK(dst_transport.RegisterExpectedChunks(uuid, 1));

    std::vector<BufferPushTask> tasks = {
        BufferPushTask{
            .peer = dst_addr,
            .buffer_id = kBufferId,
            .dst_shard_idx = kDstShardIdx,
            .dst_offset_bytes = 0,
            .data_ptr = src_payload.data(),
            .size_bytes = slice_size,
            .dst_stride_bytes = dst_stride,
            .count = count,
            .src_stride_bytes = src_stride,
        },
    };

    const auto push_res =
        src_transport.PushBuffers(tasks, /*parallelism=*/1, uuid);
    EXPECT_OK(push_res) << push_res.message();

    for (size_t c = 0; c < count; ++c) {
      EXPECT_THAT(dst.DataSpan(c * dst_stride, slice_size),
                  Pointwise(Eq(), absl::MakeConstSpan(src_payload)
                                      .subspan(c * src_stride, slice_size)))
          << "Mismatch at slice " << c
          << " (coalesce_window=" << coalesce_window << ")";
    }
    EXPECT_TRUE(dst.WaitForDataReceived(kNotificationTimeout));
  }
}

TEST_P(RawBufferTransportTest,
       PushBuffersRejectsOverlappingSourceStridedRequest) {
  constexpr size_t slice_size = 128;
  constexpr size_t count = 2;
  constexpr size_t src_stride = 64;  // Overlapping: src_stride < slice_size

  RawMockDelegate src(1024);
  RawMockDelegate dst(1024);

  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  const std::string dst_addr = GetIpPort(dst_transport);
  std::vector<uint8_t> payload(1024);

  std::vector<BufferPushTask> tasks = {
      BufferPushTask{
          .peer = dst_addr,
          .buffer_id = kBufferId,
          .dst_shard_idx = kDstShardIdx,
          .dst_offset_bytes = 0,
          .data_ptr = payload.data(),
          .size_bytes = slice_size,
          .dst_stride_bytes = slice_size,
          .count = count,
          .src_stride_bytes = src_stride,
      },
  };

  EXPECT_THAT(
      src_transport.PushBuffers(tasks, /*parallelism=*/1, /*uuid=*/77778),
      ::absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_P(RawBufferTransportTest, ProcessSocketBufferPushRejectsStridedRequest) {
  RawMockDelegate src(1024);
  RawMockDelegate dst(1024);

  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  const std::string dst_addr = GetIpPort(dst_transport);

  std::vector<uint8_t> payload(256);
  TF_ASSERT_OK_AND_ASSIGN(
      Request req,
      BuildBufferRequest(kBufferId, kDstShardIdx, /*offset_bytes=*/0,
                         payload.data(), /*size_bytes=*/128, /*uuid=*/1234,
                         kOpBufferPush, /*dst_stride_bytes=*/512,
                         /*stride_count=*/2));

  EXPECT_THAT(src_transport.ProcessSocketBufferPush(dst_addr, req),
              ::absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_P(RawBufferTransportTest, PushBuffersRejectsOverlappingStridedRequest) {
  constexpr size_t slice_size = 128;
  constexpr size_t count = 2;
  constexpr size_t dst_stride = 64;  // Overlapping: dst_stride < slice_size
  constexpr size_t total_payload_bytes = count * slice_size;
  constexpr size_t dst_buffer_size = 1024;

  RawMockDelegate src(total_payload_bytes);
  RawMockDelegate dst(dst_buffer_size);

  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  const std::string dst_addr = GetIpPort(dst_transport);
  std::vector<uint8_t> payload(total_payload_bytes);

  uint64_t uuid = 77777;
  ASSERT_OK(dst_transport.RegisterExpectedChunks(uuid, 1));

  std::vector<BufferPushTask> tasks = {
      BufferPushTask{
          .peer = dst_addr,
          .buffer_id = kBufferId,
          .dst_shard_idx = kDstShardIdx,
          .dst_offset_bytes = 0,
          .data_ptr = payload.data(),
          .size_bytes = slice_size,
          .dst_stride_bytes = dst_stride,
          .count = count,
      },
  };

  const auto push_res =
      src_transport.PushBuffers(tasks, /*parallelism=*/1, uuid);
  EXPECT_FALSE(push_res.ok());
}

TEST_P(RawBufferTransportTest, RegisterExpectedChunksZeroCountOk) {
  RawMockDelegate dst(1024);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  EXPECT_OK(dst_transport.RegisterExpectedChunks(999, 0));
}

TEST_P(RawBufferTransportTest, MultiLayerBatchAccountsChunksPerLayer) {
  // 3 chunks for layer 0, 2 chunks for layer 1 and a single layer-2 chunk, all
  // in one batch. Layer 2's expectation is registered up-front, so its
  // notification proves the whole batch has already been accounted for.
  const std::vector<size_t> layer_ids = {0, 0, 0, 1, 1, 2};
  const size_t num_chunks = layer_ids.size();

  RawMockDelegate src(num_chunks);
  RawMockDelegate dst(num_chunks);
  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  const std::string dst_addr = GetIpPort(dst_transport);
  std::vector<uint8_t> payload(num_chunks);
  RandomNonZero(absl::MakeSpan(payload));

  constexpr uint64_t kUuid = 4242;
  ASSERT_OK(dst_transport.RegisterExpectedLayerChunks(kUuid, {{2, 1}}));

  const auto push_res = src_transport.PushBuffers(
      MakeSingleBytePushTasks(dst_addr, layer_ids, payload),
      /*parallelism=*/1, kUuid);
  ASSERT_OK(push_res) << push_res.message();

  ASSERT_TRUE(dst.WaitForLayerTriggers(1, kNotificationTimeout));
  ASSERT_THAT(dst.layer_triggers(), ElementsAre(2));

  // Upper bound: expecting one chunk more than actually arrived must not
  // trigger, i.e. the accounted counts are at most 3 and 2.
  ASSERT_OK(dst_transport.RegisterExpectedLayerChunks(
      kUuid, {{0, 4}, {1, 3}, {2, 1}}));
  absl::SleepFor(kQuiesceDelay);
  EXPECT_THAT(dst.layer_triggers(), ElementsAre(2));

  // Lower bound: expecting exactly as many chunks as arrived must trigger both
  // layers, i.e. the accounted counts are at least 3 and 2. Combined with the
  // upper bound this pins them to exactly 3 and 2.
  ASSERT_OK(dst_transport.RegisterExpectedLayerChunks(
      kUuid, {{0, 3}, {1, 2}, {2, 1}}));
  absl::SleepFor(kQuiesceDelay);
  // Layer 2 must not fire a second time.
  EXPECT_THAT(dst.layer_triggers(), UnorderedElementsAre(0, 1, 2));
  EXPECT_THAT(dst.DataSpan(0, num_chunks),
              Pointwise(Eq(), absl::MakeConstSpan(payload)));
}

TEST_P(RawBufferTransportTest, InterleavedLayerChunksInBatch) {
  // Alternating layer ids defeat any "same layer as the previous chunk" fast
  // path: every single element is a miss.
  constexpr uint32_t kChunksPerLayer = 8;
  std::vector<size_t> layer_ids;
  layer_ids.reserve(2 * kChunksPerLayer);
  for (size_t i = 0; i < 2 * kChunksPerLayer; ++i) {
    layer_ids.push_back(i % 2);
  }
  const size_t num_chunks = layer_ids.size();

  RawMockDelegate src(num_chunks);
  RawMockDelegate dst(num_chunks);
  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  const std::string dst_addr = GetIpPort(dst_transport);
  std::vector<uint8_t> payload(num_chunks);
  RandomNonZero(absl::MakeSpan(payload));

  constexpr uint64_t kUuid = 777;
  // Layer 1 expects one chunk more than will ever arrive, so it must not fire
  // while layer 0, expecting exactly what arrives, must.
  ASSERT_OK(dst_transport.RegisterExpectedLayerChunks(
      kUuid, {{0, kChunksPerLayer}, {1, kChunksPerLayer + 1}}));

  const auto push_res = src_transport.PushBuffers(
      MakeSingleBytePushTasks(dst_addr, layer_ids, payload),
      /*parallelism=*/1, kUuid);
  ASSERT_OK(push_res) << push_res.message();

  // Both layers are accounted for under the same lock hold, so once layer 0
  // has fired the verdict on layer 1 is final.
  ASSERT_TRUE(dst.WaitForLayerTriggers(1, kNotificationTimeout));
  absl::SleepFor(kQuiesceDelay);
  EXPECT_THAT(dst.layer_triggers(), ElementsAre(0));

  // Lowering layer 1's expectation to what actually arrived must trigger it,
  // pinning its accounted count to exactly `kChunksPerLayer`.
  ASSERT_OK(dst_transport.RegisterExpectedLayerChunks(
      kUuid, {{0, kChunksPerLayer}, {1, kChunksPerLayer}}));
  absl::SleepFor(kQuiesceDelay);
  EXPECT_THAT(dst.layer_triggers(), UnorderedElementsAre(0, 1));
  EXPECT_THAT(dst.DataSpan(0, num_chunks),
              Pointwise(Eq(), absl::MakeConstSpan(payload)));
}

TEST_P(RawBufferTransportTest,
       LayerTriggerFiresExactlyOnceWhenBatchOvershoots) {
  // The layer expects 3 chunks but 10 arrive within the same batch.
  constexpr size_t kNumChunks = 10;
  const std::vector<size_t> layer_ids(kNumChunks, 0);

  RawMockDelegate src(kNumChunks);
  RawMockDelegate dst(kNumChunks);
  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  const std::string dst_addr = GetIpPort(dst_transport);
  std::vector<uint8_t> payload(kNumChunks);
  RandomNonZero(absl::MakeSpan(payload));

  constexpr uint64_t kUuid = 31337;
  ASSERT_OK(dst_transport.RegisterExpectedLayerChunks(kUuid, {{0, 3}}));

  const auto push_res = src_transport.PushBuffers(
      MakeSingleBytePushTasks(dst_addr, layer_ids, payload),
      /*parallelism=*/1, kUuid);
  ASSERT_OK(push_res) << push_res.message();

  ASSERT_TRUE(dst.WaitForLayerTriggers(1, kNotificationTimeout));
  absl::SleepFor(kQuiesceDelay);
  EXPECT_THAT(dst.layer_triggers(), ElementsAre(0));
  EXPECT_THAT(dst.DataSpan(0, kNumChunks),
              Pointwise(Eq(), absl::MakeConstSpan(payload)));
}

TEST_P(RawBufferTransportTest, MultiLayerBatchSpanningIovMax) {
  // 1200 chunks are split by the sender into two batches (IOV_MAX = 1024, then
  // 176). The layout is chosen so that both batches span both layers:
  //   batch 1 = 700 x layer 0 + 324 x layer 1
  //   batch 2 =  76 x layer 1 + 100 x layer 0
  // Layer 0 therefore crosses its threshold in batch 1 and still receives 100
  // more chunks in batch 2, which must not produce a second notification.
  // Layer 1 only crosses in batch 2.
  constexpr uint32_t kLayer0Expected = 600;
  constexpr uint32_t kLayer1Expected = 400;
  constexpr size_t kNumChunks = 1200;
  static_assert(kNumChunks > static_cast<size_t>(IOV_MAX));

  std::vector<size_t> layer_ids;
  layer_ids.reserve(kNumChunks);
  for (size_t i = 0; i < kNumChunks; ++i) {
    layer_ids.push_back((i < 700 || i >= 1100) ? 0 : 1);
  }

  RawMockDelegate src(kNumChunks);
  RawMockDelegate dst(kNumChunks);
  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  const std::string dst_addr = GetIpPort(dst_transport);
  std::vector<uint8_t> payload(kNumChunks);
  RandomNonZero(absl::MakeSpan(payload));

  constexpr uint64_t kUuid = 5150;
  ASSERT_OK(dst_transport.RegisterExpectedLayerChunks(
      kUuid, {{0, kLayer0Expected}, {1, kLayer1Expected}}));

  const auto push_res = src_transport.PushBuffers(
      MakeSingleBytePushTasks(dst_addr, layer_ids, payload),
      /*parallelism=*/1, kUuid);
  ASSERT_OK(push_res) << push_res.message();

  ASSERT_TRUE(dst.WaitForLayerTriggers(2, kNotificationTimeout));
  absl::SleepFor(kQuiesceDelay);
  // Layer 0 fires in batch 1 and layer 1 in batch 2, each exactly once.
  EXPECT_THAT(dst.layer_triggers(), ElementsAre(0, 1));
  EXPECT_THAT(dst.DataSpan(0, kNumChunks),
              Pointwise(Eq(), absl::MakeConstSpan(payload)));
}

TEST_P(RawBufferTransportTest,
       LayerTriggerWhenExpectedRegisteredAfterChunksArrive) {
  // Layers 0 and 1 carry the payload; layer 2 is a barrier chunk whose
  // expectation is registered up-front. Once its notification fires, the whole
  // batch (and therefore layers 0 and 1) has been accounted for, so the
  // expectations below are guaranteed to be registered after arrival.
  const std::vector<size_t> layer_ids = {0, 0, 0, 1, 1, 1, 2};
  const size_t num_chunks = layer_ids.size();

  RawMockDelegate src(num_chunks);
  RawMockDelegate dst(num_chunks);
  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  const std::string dst_addr = GetIpPort(dst_transport);
  std::vector<uint8_t> payload(num_chunks);
  RandomNonZero(absl::MakeSpan(payload));

  constexpr uint64_t kUuid = 2024;
  ASSERT_OK(dst_transport.RegisterExpectedLayerChunks(kUuid, {{2, 1}}));

  const auto push_res = src_transport.PushBuffers(
      MakeSingleBytePushTasks(dst_addr, layer_ids, payload),
      /*parallelism=*/1, kUuid);
  ASSERT_OK(push_res) << push_res.message();

  ASSERT_TRUE(dst.WaitForLayerTriggers(1, kNotificationTimeout));
  EXPECT_THAT(dst.layer_triggers(), ElementsAre(2));

  // Registering the expectations now must retroactively trigger layers 0 and 1
  // while leaving the already-triggered layer 2 alone.
  ASSERT_OK(dst_transport.RegisterExpectedLayerChunks(
      kUuid, {{0, 3}, {1, 3}, {2, 1}}));
  absl::SleepFor(kQuiesceDelay);
  EXPECT_THAT(dst.layer_triggers(), UnorderedElementsAre(0, 1, 2));
  EXPECT_THAT(dst.DataSpan(0, num_chunks),
              Pointwise(Eq(), absl::MakeConstSpan(payload)));
}

TEST_P(RawBufferTransportTest, PerLayerAccountingMatchesNaiveComputation) {
  constexpr size_t kNumChunks = 256;
  // Deliberately larger than the inline capacity of the receiver's per-batch
  // layer histogram, so its heap-allocating path is exercised too.
  constexpr size_t kNumLayers = 6;

  // Deterministic pseudo-random layer assignment.
  std::mt19937 gen(20240607);
  std::uniform_int_distribution<size_t> layer_dist(0, kNumLayers - 1);
  std::vector<size_t> layer_ids;
  layer_ids.reserve(kNumChunks);
  for (size_t i = 0; i < kNumChunks; ++i) {
    layer_ids.push_back(layer_dist(gen));
  }

  // Reference per-layer chunk counts, computed with a naive loop.
  absl::flat_hash_map<size_t, uint32_t> reference_counts;
  for (const size_t layer_idx : layer_ids) {
    ++reference_counts[layer_idx];
  }
  ASSERT_EQ(reference_counts.size(), kNumLayers);

  // Even layers expect exactly the reference count, so they must trigger. Odd
  // layers expect one chunk more than will ever arrive, so they must not.
  // Together this pins the accounted count of every layer exactly.
  absl::flat_hash_map<size_t, uint32_t> expected_layer_chunks;
  std::vector<size_t> expected_triggers;
  for (const auto& [layer_idx, count] : reference_counts) {
    if (layer_idx % 2 == 0) {
      expected_layer_chunks[layer_idx] = count;
      expected_triggers.push_back(layer_idx);
    } else {
      expected_layer_chunks[layer_idx] = count + 1;
    }
  }

  RawMockDelegate src(kNumChunks);
  RawMockDelegate dst(kNumChunks);
  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  const std::string dst_addr = GetIpPort(dst_transport);
  std::vector<uint8_t> payload(kNumChunks);
  RandomNonZero(absl::MakeSpan(payload));

  constexpr uint64_t kUuid = 8675309;
  ASSERT_OK(
      dst_transport.RegisterExpectedLayerChunks(kUuid, expected_layer_chunks));
  // The whole-transfer notification runs after all layer notifications of the
  // batch, so it is a reliable barrier for the negative assertions below.
  ASSERT_OK(dst_transport.RegisterExpectedChunks(kUuid, kNumChunks));

  const auto push_res = src_transport.PushBuffers(
      MakeSingleBytePushTasks(dst_addr, layer_ids, payload),
      /*parallelism=*/1, kUuid);
  ASSERT_OK(push_res) << push_res.message();

  ASSERT_TRUE(dst.WaitForDataReceived(kNotificationTimeout));
  EXPECT_THAT(dst.layer_triggers(),
              UnorderedElementsAreArray(expected_triggers));
  EXPECT_THAT(dst.DataSpan(0, kNumChunks),
              Pointwise(Eq(), absl::MakeConstSpan(payload)));
}

TEST_P(RawBufferTransportTest, ConcurrentBatchesTriggerEachLayerExactlyOnce) {
  // `parallelism = 4` splits the tasks for the single destination into four
  // batches that are pushed concurrently, so four receiver worker threads
  // contend on `raw_progress_mu_` for the same uuid. Layer ids are interleaved
  // so every batch spans every layer.
  constexpr size_t kNumLayers = 5;
  constexpr uint32_t kChunksPerLayer = 40;
  constexpr size_t kNumChunks = kNumLayers * kChunksPerLayer;

  std::vector<size_t> layer_ids;
  layer_ids.reserve(kNumChunks);
  for (size_t i = 0; i < kNumChunks; ++i) {
    layer_ids.push_back(i % kNumLayers);
  }

  absl::flat_hash_map<size_t, uint32_t> expected_layer_chunks;
  std::vector<size_t> expected_triggers;
  for (size_t layer_idx = 0; layer_idx < kNumLayers; ++layer_idx) {
    expected_layer_chunks[layer_idx] = kChunksPerLayer;
    expected_triggers.push_back(layer_idx);
  }

  RawMockDelegate src(kNumChunks);
  RawMockDelegate dst(kNumChunks);
  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  const std::string dst_addr = GetIpPort(dst_transport);
  std::vector<uint8_t> payload(kNumChunks);
  RandomNonZero(absl::MakeSpan(payload));

  constexpr uint64_t kUuid = 606060;
  ASSERT_OK(
      dst_transport.RegisterExpectedLayerChunks(kUuid, expected_layer_chunks));

  const auto push_res = src_transport.PushBuffers(
      MakeSingleBytePushTasks(dst_addr, layer_ids, payload),
      /*parallelism=*/4, kUuid);
  ASSERT_OK(push_res) << push_res.message();

  ASSERT_TRUE(dst.WaitForLayerTriggers(kNumLayers, kNotificationTimeout));
  absl::SleepFor(kQuiesceDelay);
  // No layer may be notified twice, whichever thread wins the races.
  EXPECT_THAT(dst.layer_triggers(),
              UnorderedElementsAreArray(expected_triggers));
  EXPECT_THAT(dst.DataSpan(0, kNumChunks),
              Pointwise(Eq(), absl::MakeConstSpan(payload)));
}

TEST_P(RawBufferTransportTest, TelemetryRecordsSentReceivedAndP2pMetrics) {
  ASSERT_OK(telemetry::RaidenMetricStore::GetGlobalMetricStore()
                .InitializeFromBackendNames({"buffered"}));

  constexpr size_t kNumChunks = 10;
  RawMockDelegate src(kNumChunks);
  RawMockDelegate dst(kNumChunks);
  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  const std::string dst_addr = GetIpPort(dst_transport);
  std::vector<uint8_t> payload(kNumChunks);
  RandomNonZero(absl::MakeSpan(payload));
  std::vector<size_t> layer_ids(kNumChunks, 0);

  constexpr uint64_t kUuid = 777888;
  ASSERT_OK(dst_transport.RegisterExpectedChunks(kUuid, kNumChunks));

  const auto push_res = src_transport.PushBuffers(
      MakeSingleBytePushTasks(dst_addr, layer_ids, payload),
      /*parallelism=*/1, kUuid);
  ASSERT_OK(push_res) << push_res.message();

  ASSERT_TRUE(dst.WaitForDataReceived(kNotificationTimeout));

  auto samples = telemetry::RaidenMetricStore::GetGlobalMetricStore()
                     .GetAndResetMetricSamples();
  EXPECT_FALSE(
      samples[std::string("tpu_raiden_") +
              std::string(telemetry::metric_names::kWeightSyncSentBytesTotal)]
          .empty());
  EXPECT_FALSE(
      samples[std::string("tpu_raiden_") +
              std::string(
                  telemetry::metric_names::kWeightSyncReceivedBytesTotal)]
          .empty());

  telemetry::RaidenMetricStore::GetGlobalMetricStore().SetBackends({});
}

INSTANTIATE_TEST_SUITE_P(
    PspAndPlainTcp, RawBufferTransportTest, ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "PSP" : "PlainTcp";
    });

INSTANTIATE_TEST_SUITE_P(
    PspAndPlainTcp, ConnPoolTest, ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "PSP" : "PlainTcp";
    });

}  // namespace
}  // namespace tpu_raiden::transport::lib
