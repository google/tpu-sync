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

#include <chrono>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
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
#include "absl/types/span.h"
#include "grpcpp/channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/support/channel_arguments.h"
#include "tpu_sync/transport/buffer_push_task.h"
#include "tpu_sync/transport/lib/conn/pool.h"
#include "tpu_sync/transport/lib/peregrine_control_service.h"
#include "tpu_sync/transport/lib/raw_buffer_transport_delegate.h"
#include "tpu_sync/transport/lib/socket/psp_syscall_mock.h" // NOLINT
#include "tpu_sync/transport/lib/socket/tcp_psp_helper.h"
#include "tpu_sync/transport/peregrine/src/util/util.h"

namespace tpu_raiden::transport::lib {
namespace {

using ::peregrine::util::AllZero;
using ::peregrine::util::RandomNonZero;
using ::testing::Each;
using ::testing::Eq;
using ::testing::Ne;
using ::testing::Pointwise;

constexpr int kLocalPort = 0;
constexpr size_t kBufferId = 0;
constexpr size_t kSrcShardIdx = 0;
constexpr size_t kDstShardIdx = 0;

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
    absl::MutexLock lock(mu_);
    on_data_received_called_ = true;
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

 private:
  std::vector<uint8_t> buffer_;
  mutable absl::Mutex mu_;
  bool on_data_received_called_ ABSL_GUARDED_BY(mu_) = false;
  std::shared_ptr<grpc::Channel> default_channel_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, std::shared_ptr<grpc::Channel>>
      peer_channels_ ABSL_GUARDED_BY(mu_);
};

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
  EXPECT_TRUE(dst1.on_data_received());

  // Post-condition: check payloads at correct offsets for dst2.
  EXPECT_THAT(dst2.DataSpan(0, payload2.size()),
              Pointwise(Eq(), absl::MakeConstSpan(payload2)));
  EXPECT_THAT(dst2.DataSpan(8192, payload4.size()),
              Pointwise(Eq(), absl::MakeConstSpan(payload4)));
  EXPECT_TRUE(dst2.on_data_received());
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
  EXPECT_TRUE(dst1.on_data_received());

  // Post-condition: check payloads at correct offsets for dst2.
  EXPECT_THAT(dst2.DataSpan(0, payload2.size()),
              Pointwise(Eq(), absl::MakeConstSpan(payload2)));
  EXPECT_THAT(dst2.DataSpan(8192, payload4.size()),
              Pointwise(Eq(), absl::MakeConstSpan(payload4)));
  EXPECT_TRUE(dst2.on_data_received());
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
  EXPECT_TRUE(dst.on_data_received());
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
  EXPECT_TRUE(dst.on_data_received());
}

TEST_P(RawBufferTransportTest, RegisterExpectedChunksZeroCountOk) {
  RawMockDelegate dst(1024);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  EXPECT_OK(dst_transport.RegisterExpectedChunks(999, 0));
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
