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

#include "tpu_sync/transport/block_transport.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>  // NOLINT
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/support/channel_arguments.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/prometheus_exporter.h"
#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/transport/buffer_push_task.h"
#include "tpu_sync/transport/lib/socket/psp_syscall_mock.h" // NOLINT
#include "tpu_sync/transport/lib/socket/tcp_psp_helper.h"
#include "absl/flags/flag.h"

namespace tpu_raiden {
namespace transport {
namespace {

using ::absl_testing::StatusIs;
using ::testing::Each;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::Pointwise;

constexpr absl::Duration kMetricPollingTimeout = absl::Seconds(5);
constexpr absl::Duration kMetricPollingInterval = absl::Milliseconds(10);

// Polls the global metric store periodically at `kMetricPollingInterval` until
// `expected_metric` is contained in the text snapshot or `timeout` expires.
//
// Returns the metric text snapshot immediately when `expected_metric` is found,
// or returns the latest snapshot observed upon timeout so that downstream
// `EXPECT_THAT(..., HasSubstr(...))` matchers can display informative failure
// diffs.
std::string WaitForMetricSnapshot(
    absl::string_view expected_metric,
    absl::Duration timeout = kMetricPollingTimeout) {
  const absl::Time deadline = absl::Now() + timeout;
  std::string snapshot;
  while (absl::Now() < deadline) {
    snapshot =
        telemetry::RaidenMetricStore::GetGlobalMetricStore().GetTextSnapshot();
    if (absl::StrContains(snapshot, expected_metric)) {
      return snapshot;
    }
    absl::SleepFor(kMetricPollingInterval);
  }
  return snapshot;
}

struct ScopedPrometheusBackend {
  ScopedPrometheusBackend() {
    auto exporter = std::make_unique<telemetry::PrometheusExporter>();
    std::vector<std::unique_ptr<telemetry::MetricsBackend>> backends;
    backends.push_back(std::move(exporter));
    telemetry::RaidenMetricStore::GetGlobalMetricStore().SetBackends(
        std::move(backends));
  }

  ~ScopedPrometheusBackend() {
    telemetry::RaidenMetricStore::GetGlobalMetricStore().SetBackends({});
  }
};

class MockDelegate : public BlockTransportDelegate {
 public:
  MockDelegate(size_t slice_size, int max_blocks = 1, size_t num_layers = 1,
               size_t num_shards = 1)
      : slice_size_(slice_size),
        max_blocks_(max_blocks),
        num_layers_(num_layers),
        num_shards_(num_shards) {
    buffers_.resize(num_layers_ * num_shards_);
    for (auto& buffer : buffers_) {
      buffer.resize(slice_size_ * max_blocks_, 0);
    }
  }

  absl::StatusOr<std::vector<int>> AllocateBlocks(size_t num_blocks,
                                                  uint64_t uuid = 0) override {
    std::vector<int> ids;
    for (size_t i = 0; i < num_blocks; ++i) {
      ids.push_back(i % max_blocks_);
    }
    return ids;
  }

  absl::Status OnDataReceived(uint64_t uuid = 0) override {
    on_data_received_called_ = true;
    return absl::OkStatus();
  }

  absl::Status OnSingleBlockReceived(int block_id, size_t size_bytes) override {
    on_single_block_received_called_ = true;
    received_block_id_ = block_id;
    received_size_bytes_ = size_bytes;
    return OnDataReceived();
  }

  absl::Status OnLayerReceived(size_t layer_idx, uint64_t uuid) override {
    (void)layer_idx;
    (void)uuid;
    ++layer_completion_count_;
    return absl::OkStatus();
  }

  absl::StatusOr<std::optional<PoolPushProgressSpec>> GetPoolPushProgressSpec(
      size_t pool_idx, uint64_t uuid) const override {
    if (!pool_progress_uuid_.has_value() || *pool_progress_uuid_ != uuid) {
      return std::nullopt;
    }
    if (std::find(transfer_pool_indices_.begin(), transfer_pool_indices_.end(),
                  pool_idx) == transfer_pool_indices_.end()) {
      return absl::InvalidArgumentError("pool is outside transfer set");
    }
    return PoolPushProgressSpec{
        .expected_pushes = expected_pushes_per_pool_,
        .expected_pools = transfer_pool_indices_.size(),
    };
  }

  absl::Status OnPoolReceived(size_t pool_idx, uint64_t uuid) override {
    (void)pool_idx;
    (void)uuid;
    ++pool_completion_count_;
    return absl::OkStatus();
  }

  void SetPoolPushProgress(uint64_t uuid, size_t expected_pushes_per_pool,
                           std::vector<size_t> transfer_pool_indices) {
    pool_progress_uuid_ = uuid;
    expected_pushes_per_pool_ = expected_pushes_per_pool;
    transfer_pool_indices_ = std::move(transfer_pool_indices);
  }

  void RegisterBlockReadinessCallback(size_t layer_idx, size_t shard_idx,
                                      int block_id, uint64_t uuid,
                                      HostBlockReadyCallback cb) override {
    {
      absl::MutexLock lock(wait_events_mu_);
      wait_events_.push_back(std::make_tuple(layer_idx, shard_idx, block_id));
    }
    cb(absl::OkStatus());
  }

  void SetPeerChannel(absl::string_view peer,
                      std::shared_ptr<grpc::Channel> channel) {
    absl::MutexLock lock(peer_channels_mu_);
    peer_channels_[std::string(peer)] = std::move(channel);
  }

  std::shared_ptr<grpc::Channel> GetPeregrineChannel(
      absl::string_view peer) override {
    absl::MutexLock lock(peer_channels_mu_);
    auto it = peer_channels_.find(peer);
    if (it != peer_channels_.end()) {
      return it->second;
    }
    return default_channel_;
  }

  bool on_data_received_called() const { return on_data_received_called_; }
  void reset_data_received() { on_data_received_called_ = false; }

  bool on_single_block_received_called() const {
    return on_single_block_received_called_;
  }
  int received_block_id() const { return received_block_id_; }
  size_t received_size_bytes() const { return received_size_bytes_; }

  void reset_single_block_received() {
    on_single_block_received_called_ = false;
    received_block_id_ = -1;
    received_size_bytes_ = 0;
  }

  uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) override {
    return buffers_[BufferIndex(layer_idx, shard_idx)].data();
  }

  size_t GetHostSize(size_t layer_idx, size_t shard_idx) override {
    return buffers_[BufferIndex(layer_idx, shard_idx)].size();
  }

  int layer_completion_count() const { return layer_completion_count_.load(); }
  int pool_completion_count() const { return pool_completion_count_.load(); }

  int GetRemoteReadBlockId(int base_remote_id, int chunk_k) override {
    return base_remote_id + chunk_k;
  }

  size_t num_layers() const override { return num_layers_; }
  size_t num_shards() const override { return num_shards_; }
  size_t slice_byte_size() const override { return slice_size_; }
  size_t shard_factor() const override { return 1; }

  uint8_t* data(size_t layer_idx = 0, size_t shard_idx = 0) {
    return buffers_[BufferIndex(layer_idx, shard_idx)].data();
  }
  uint8_t* block_data(int block_id, size_t layer_idx = 0,
                      size_t shard_idx = 0) {
    return data(layer_idx, shard_idx) + block_id * slice_size_;
  }
  std::vector<std::tuple<size_t, size_t, int>> wait_events() {
    absl::MutexLock lock(wait_events_mu_);
    return wait_events_;
  }

 private:
  size_t BufferIndex(size_t layer_idx, size_t shard_idx) const {
    return layer_idx * num_shards_ + shard_idx;
  }

  size_t slice_size_;
  int max_blocks_;
  size_t num_layers_;
  size_t num_shards_;
  std::vector<std::vector<uint8_t>> buffers_;
  bool on_data_received_called_ = false;
  bool on_single_block_received_called_ = false;
  int received_block_id_ = -1;
  size_t received_size_bytes_ = 0;
  std::optional<uint64_t> pool_progress_uuid_;
  size_t expected_pushes_per_pool_ = 0;
  std::vector<size_t> transfer_pool_indices_;
  std::atomic<int> layer_completion_count_{0};
  std::atomic<int> pool_completion_count_{0};
  absl::Mutex wait_events_mu_;
  std::vector<std::tuple<size_t, size_t, int>> wait_events_;
  mutable absl::Mutex peer_channels_mu_;
  std::shared_ptr<grpc::Channel> default_channel_
      ABSL_GUARDED_BY(peer_channels_mu_);
  absl::flat_hash_map<std::string, std::shared_ptr<grpc::Channel>>
      peer_channels_ ABSL_GUARDED_BY(peer_channels_mu_);
};

class SamePeerFanoutDelegate : public MockDelegate {
 public:
  SamePeerFanoutDelegate() : MockDelegate(/*slice_size=*/256) {}

  std::vector<BlockChunk> GetBlockChunks(size_t layer_idx, size_t shard_idx,
                                         absl::Span<const int64_t> block_ids,
                                         size_t total_bytes, uint64_t uuid,
                                         int64_t sender_node_id = -1,
                                         absl::string_view peer = "",
                                         int64_t src_block_id = -1,
                                         int64_t dst_block_id = -1) override {
    (void)layer_idx;
    (void)shard_idx;
    (void)block_ids;
    (void)total_bytes;
    (void)uuid;
    (void)sender_node_id;
    (void)peer;
    (void)src_block_id;
    if (dst_block_id < 0 || dst_block_id >= 4) {
      return {};
    }
    return {{.ptr = data() + dst_block_id * 64, .size = 64}};
  }
};

// Sender delegate with a fixed node id, so a receiver can tell two senders
// of the same uuid apart.
class NodeDelegate : public MockDelegate {
 public:
  NodeDelegate(int64_t node_id, size_t slice_size, int max_blocks,
               size_t num_layers)
      : MockDelegate(slice_size, max_blocks, num_layers), node_id_(node_id) {}

  int64_t node_id() const override { return node_id_; }

 private:
  int64_t node_id_;
};

// Receiver delegate whose plan for `uuid` declares how many senders assemble
// each block array.
class MultiSenderReceiverDelegate : public MockDelegate {
 public:
  using MockDelegate::MockDelegate;

  void ExpectSenders(uint64_t uuid, size_t senders) {
    expected_uuid_ = uuid;
    expected_senders_ = senders;
  }

  std::optional<size_t> ExpectedPushSenders(uint64_t uuid) const override {
    if (!expected_uuid_.has_value() || *expected_uuid_ != uuid) {
      return std::nullopt;
    }
    return expected_senders_;
  }

 private:
  std::optional<uint64_t> expected_uuid_;
  size_t expected_senders_ = 0;
};

class PlanRequiringDelegate : public MockDelegate {
 public:
  using MockDelegate::MockDelegate;

  bool AcceptsPlanlessExplicitPush(uint64_t uuid) const override {
    return false;
  }
};

class BlockTransportTest : public ::testing::Test {
 protected:
  void SetUp() override { absl::SetFlag(&FLAGS_require_psp_tcp, true); }

  void TearDown() override {
    for (auto& server : servers_) {
      if (server) {
        server->Shutdown();
      }
    }
    servers_.clear();
  }

  std::shared_ptr<grpc::Channel> StartControlServer(
      BlockTransport* transport) {
    grpc::ServerBuilder builder;
    builder.RegisterService(transport->peregrine_control_service());
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    std::shared_ptr<grpc::Channel> channel =
        server->InProcessChannel(grpc::ChannelArguments());
    servers_.push_back(std::move(server));
    return channel;
  }

  void BindControlChannels(BlockTransport* transport1, MockDelegate* delegate1,
                           BlockTransport* transport2,
                           MockDelegate* delegate2) {
    auto ch1 = StartControlServer(transport1);
    auto ch2 = StartControlServer(transport2);
    delegate1->SetPeerChannel(
        absl::StrCat("localhost:", transport2->local_port()), ch2);
    delegate2->SetPeerChannel(
        absl::StrCat("localhost:", transport1->local_port()), ch1);
  }

 private:
  std::vector<std::unique_ptr<grpc::Server>> servers_;
};

TEST_F(BlockTransportTest, PoolModeReceiverRejectsPlanlessExplicitPush) {
  size_t size = 1024;
  MockDelegate sender_delegate(size);
  PlanRequiringDelegate receiver_delegate(size);
  std::memset(sender_delegate.data(), 0xAB, size);
  std::memset(receiver_delegate.data(), 0x00, size);

  BlockTransport sender(&sender_delegate, 0);
  BlockTransport receiver(&receiver_delegate, 0);
  BindControlChannels(&sender, &sender_delegate, &receiver, &receiver_delegate);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Explicit destination (op=6) without a registered plan: the pool-mode
  // receiver drops the push before any payload byte lands in its mirror.
  auto rejected =
      sender.SyncPush({absl::StrCat("localhost:", receiver.local_port())},
                      /*src_block_ids=*/{0}, /*dst_block_ids=*/{0},
                      /*parallelism=*/1, MajorOrder::kLayerMajor, /*uuid=*/77,
                      /*layer_idx=*/-1);
  EXPECT_FALSE(rejected.ok());
  EXPECT_EQ(receiver_delegate.data()[0], 0x00);

  // The plan-less legacy contract (receiver-allocated destinations) is
  // untouched.
  auto legacy =
      sender.SyncPush({absl::StrCat("localhost:", receiver.local_port())},
                      /*src_block_ids=*/{0}, /*dst_block_ids=*/{},
                      /*parallelism=*/1, MajorOrder::kLayerMajor, /*uuid=*/0,
                      /*layer_idx=*/-1);
  ASSERT_TRUE(legacy.ok()) << legacy.status().message();
  EXPECT_EQ(receiver_delegate.data()[0], 0xAB);
}

TEST_F(BlockTransportTest, PushAndPullCorrectness) {
  size_t size = 1024;
  MockDelegate delegate1(size);
  MockDelegate delegate2(size);

  // Populate source with custom pattern
  std::memset(delegate1.data(), 0xAB, size);
  std::memset(delegate2.data(), 0x00, size);

  BlockTransport transport1(&delegate1, 0);
  BlockTransport transport2(&delegate2, 0);
  BindControlChannels(&transport1, &delegate1, &transport2, &delegate2);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Push block 0 from transport1 to transport2
  auto push_res = transport1.SyncPush(
      {absl::StrCat("localhost:", transport2.local_port())},
      /*src_block_ids=*/{0}, /*dst_block_ids=*/{},
      /*parallelism=*/1, MajorOrder::kLayerMajor, /*uuid=*/0, /*layer_idx=*/-1);
  ASSERT_TRUE(push_res.ok()) << push_res.status().message();

  // Verify push parity
  EXPECT_EQ(delegate2.data()[0], 0xAB);
  EXPECT_EQ(delegate2.data()[size - 1], 0xAB);

  // Reset dest to 0
  std::memset(delegate2.data(), 0x00, size);

  // Pull block 0 from transport1 using transport2
  auto pull_res =
      transport2.SyncPull({absl::StrCat("localhost:", transport1.local_port())},
                          /*src_block_ids=*/{0},
                          /*local_block_ids=*/{}, /*explicit_dst_ptrs=*/{},
                          /*parallelism=*/1, MajorOrder::kLayerMajor,
                          /*on_block_received=*/{}, /*uuid=*/0);
  ASSERT_TRUE(pull_res.ok()) << pull_res.status().message();

  // Verify pull parity
  EXPECT_EQ(delegate2.data()[0], 0xAB);
  EXPECT_EQ(delegate2.data()[size - 1], 0xAB);
}

TEST_F(BlockTransportTest, PullNonContiguous) {
  size_t size = 1024;
  // Delegate 1 has 3 blocks capacity
  MockDelegate delegate1(size, 3);
  // Delegate 2 has 2 blocks capacity (we want to pull 2 blocks)
  MockDelegate delegate2(size, 2);

  // Populate source blocks with different patterns
  std::memset(delegate1.block_data(0), 0xAA, size);
  std::memset(delegate1.block_data(1), 0xBB, size);
  std::memset(delegate1.block_data(2), 0xCC, size);

  std::memset(delegate2.block_data(0), 0x00, size);
  std::memset(delegate2.block_data(1), 0x00, size);

  BlockTransport transport1(&delegate1, 0);
  BlockTransport transport2(&delegate2, 0);
  BindControlChannels(&transport1, &delegate1, &transport2, &delegate2);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Pull block 0 and 2 (non-contiguous) from transport1 using transport2
  // We expect they will be written to local block 0 and 1 respectively.
  auto pull_res = transport2.SyncPull(
      {absl::StrCat("localhost:", transport1.local_port())},
      /*src_block_ids=*/{0, 2},
      /*local_block_ids=*/{}, /*explicit_dst_ptrs=*/{}, /*parallelism=*/1,
      MajorOrder::kLayerMajor, /*on_block_received=*/{}, /*uuid=*/0);
  ASSERT_TRUE(pull_res.ok()) << pull_res.status().message();

  // Verify pull parity
  // Local Block 0 should have 0xAA (from remote Block 0)
  EXPECT_EQ(delegate2.block_data(0)[0], 0xAA);
  EXPECT_EQ(delegate2.block_data(0)[size - 1], 0xAA);

  // Local Block 1 should have 0xCC (from remote Block 2)
  EXPECT_EQ(delegate2.block_data(1)[0], 0xCC);
  EXPECT_EQ(delegate2.block_data(1)[size - 1], 0xCC);
}

TEST_F(BlockTransportTest,
       PullExplicitDestPtrsMultiLayerUnevenParallelism) {
  constexpr size_t kSliceSize = 16;
  constexpr int kNumBlocks = 3;
  constexpr size_t kNumLayers = 2;
  MockDelegate source(kSliceSize, kNumBlocks, kNumLayers);
  MockDelegate receiver(kSliceSize, kNumBlocks, kNumLayers);

  for (size_t layer = 0; layer < kNumLayers; ++layer) {
    for (int block = 0; block < kNumBlocks; ++block) {
      std::memset(source.block_data(block, layer),
                  static_cast<int>(0x10 + layer * 0x10 + block), kSliceSize);
    }
  }

  std::vector<uint8_t> layer0(kSliceSize * kNumBlocks, 0);
  std::vector<uint8_t> layer1(kSliceSize * kNumBlocks, 0);
  std::vector<uint8_t*> explicit_dst_ptrs = {layer0.data(), layer1.data()};

  BlockTransport source_transport(&source, 0);
  BlockTransport receiver_transport(&receiver, 0);
  BindControlChannels(&source_transport, &source, &receiver_transport,
                      &receiver);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  auto pull_res = receiver_transport.SyncPull(
      {absl::StrCat("localhost:", source_transport.local_port())}, {0, 1, 2},
      /*local_block_ids=*/{0, 1, 2}, explicit_dst_ptrs, /*parallelism=*/2,
      MajorOrder::kLayerMajor,
      /*on_block_received=*/{}, /*uuid=*/0);
  ASSERT_TRUE(pull_res.ok()) << pull_res.status().message();
  EXPECT_EQ(*pull_res, std::vector<int>({0, 1, 2}));

  for (int block = 0; block < kNumBlocks; ++block) {
    EXPECT_EQ(layer0[block * kSliceSize], 0x10 + block);
    EXPECT_EQ(layer0[(block + 1) * kSliceSize - 1], 0x10 + block);
    EXPECT_EQ(layer1[block * kSliceSize], 0x20 + block);
    EXPECT_EQ(layer1[(block + 1) * kSliceSize - 1], 0x20 + block);
  }
}

TEST_F(BlockTransportTest, PullRejectsOutOfBoundsRemoteBlock) {
  constexpr size_t kSliceSize = 16;
  constexpr int kNumBlocks = 1;
  MockDelegate source(kSliceSize, kNumBlocks);
  MockDelegate receiver(kSliceSize, kNumBlocks);

  BlockTransport source_transport(&source, 0);
  BlockTransport receiver_transport(&receiver, 0);
  BindControlChannels(&source_transport, &source, &receiver_transport,
                      &receiver);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  auto pull_res = receiver_transport.SyncPull(
      {absl::StrCat("localhost:", source_transport.local_port())},
      /*src_block_ids=*/{1},
      /*local_block_ids=*/{0}, /*explicit_dst_ptrs=*/{},
      /*parallelism=*/1, MajorOrder::kLayerMajor,
      /*on_block_received=*/{}, /*uuid=*/0);
  EXPECT_FALSE(pull_res.ok());
}

TEST_F(BlockTransportTest, PullSupportsBlockMajorOrder) {
  constexpr size_t kSliceSize = 16;
  constexpr int kNumBlocks = 2;
  constexpr size_t kNumLayers = 2;
  MockDelegate source(kSliceSize, kNumBlocks, kNumLayers);
  MockDelegate receiver(kSliceSize, kNumBlocks, kNumLayers);

  for (size_t layer = 0; layer < kNumLayers; ++layer) {
    for (int block = 0; block < kNumBlocks; ++block) {
      std::memset(source.block_data(block, layer),
                  static_cast<int>(0x10 + layer * 0x10 + block), kSliceSize);
    }
  }

  BlockTransport source_transport(&source, 0);
  BlockTransport receiver_transport(&receiver, 0);
  BindControlChannels(&source_transport, &source, &receiver_transport,
                      &receiver);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  auto pull_res = receiver_transport.SyncPull(
      {absl::StrCat("localhost:", source_transport.local_port())},
      /*src_block_ids=*/{0, 1},
      /*local_block_ids=*/{0, 1}, /*explicit_dst_ptrs=*/{},
      /*parallelism=*/1, MajorOrder::kBlockMajor,
      /*on_block_received=*/{}, /*uuid=*/0);
  ASSERT_TRUE(pull_res.ok()) << pull_res.status().message();

  for (int block = 0; block < kNumBlocks; ++block) {
    EXPECT_EQ(receiver.block_data(block, 0)[0], 0x10 + block);
    EXPECT_EQ(receiver.block_data(block, 1)[0], 0x20 + block);
  }

  EXPECT_EQ(source.wait_events(),
            (std::vector<std::tuple<size_t, size_t, int>>{
                std::make_tuple(0, 0, 0),
                std::make_tuple(1, 0, 0),
                std::make_tuple(0, 0, 1),
                std::make_tuple(1, 0, 1),
            }));
}

TEST_F(BlockTransportTest, SamePeerFanoutFiltersEachDestinationStream) {
  SamePeerFanoutDelegate sender;
  MockDelegate receiver(/*slice_size=*/64, /*max_blocks=*/4);
  for (int page = 0; page < 4; ++page) {
    std::memset(sender.data() + page * 64, 0x31 + page, 64);
  }

  BlockTransport sender_transport(&sender, 0);
  BlockTransport receiver_transport(&receiver, 0);
  BindControlChannels(&sender_transport, &sender, &receiver_transport,
                      &receiver);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  auto result = sender_transport.SyncPush(
      {absl::StrCat("localhost:", receiver_transport.local_port())},
      /*src_block_ids=*/{0, 0, 0, 0},
      /*dst_block_ids=*/{0, 1, 2, 3}, /*parallelism=*/4,
      MajorOrder::kLayerMajor, /*uuid=*/901, /*layer_idx=*/0);

  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(*result, std::vector<int>({0, 1, 2, 3}));
  for (int page = 0; page < 4; ++page) {
    EXPECT_TRUE(
        std::all_of(receiver.block_data(page), receiver.block_data(page) + 64,
                    [page](uint8_t byte) { return byte == 0x31 + page; }));
  }
}

TEST_F(BlockTransportTest, LayerCompletesOnlyAfterEveryDeclaredSender) {
  constexpr uint64_t kUuid = 905;
  constexpr size_t kSliceSize = 32;
  constexpr size_t kNumLayers = 2;
  NodeDelegate sender_a(/*node_id=*/0, kSliceSize, /*max_blocks=*/1,
                        kNumLayers);
  NodeDelegate sender_b(/*node_id=*/1, kSliceSize, /*max_blocks=*/1,
                        kNumLayers);
  MultiSenderReceiverDelegate receiver(kSliceSize, /*max_blocks=*/1,
                                       kNumLayers);
  receiver.ExpectSenders(kUuid, /*senders=*/2);

  BlockTransport a_transport(&sender_a, 0);
  BlockTransport b_transport(&sender_b, 0);
  BlockTransport receiver_transport(&receiver, 0);
  const std::string peer =
      absl::StrCat("localhost:", receiver_transport.local_port());
  auto ch_recv = StartControlServer(&receiver_transport);
  sender_a.SetPeerChannel(peer, ch_recv);
  sender_b.SetPeerChannel(peer, ch_recv);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  auto push_layer = [&](BlockTransport& transport, int layer_idx) {
    return transport.SyncPush({peer}, /*src_block_ids=*/{0},
                              /*dst_block_ids=*/{0}, /*parallelism=*/1,
                              MajorOrder::kLayerMajor, kUuid, layer_idx);
  };

  // One sender's full push of a layer must not complete it.
  ASSERT_TRUE(push_layer(a_transport, 0).ok());
  EXPECT_EQ(receiver.layer_completion_count(), 0);
  ASSERT_TRUE(push_layer(b_transport, 0).ok());
  EXPECT_EQ(receiver.layer_completion_count(), 1);

  // Arrival order across senders does not matter.
  ASSERT_TRUE(push_layer(b_transport, 1).ok());
  EXPECT_EQ(receiver.layer_completion_count(), 1);
  ASSERT_TRUE(push_layer(a_transport, 1).ok());
  EXPECT_EQ(receiver.layer_completion_count(), 2);

  // Every layer completed, so the uuid's progress retired and can be reused.
  ASSERT_TRUE(push_layer(a_transport, 0).ok());
  EXPECT_EQ(receiver.layer_completion_count(), 2);
  ASSERT_TRUE(push_layer(b_transport, 0).ok());
  EXPECT_EQ(receiver.layer_completion_count(), 3);
}

TEST_F(BlockTransportTest, DeclaredSendersEachCompleteTheirOwnParallelism) {
  constexpr uint64_t kUuid = 906;
  constexpr size_t kSliceSize = 32;
  NodeDelegate sender_a(/*node_id=*/0, kSliceSize, /*max_blocks=*/2,
                        /*num_layers=*/1);
  NodeDelegate sender_b(/*node_id=*/1, kSliceSize, /*max_blocks=*/2,
                        /*num_layers=*/1);
  MultiSenderReceiverDelegate receiver(kSliceSize, /*max_blocks=*/2,
                                       /*num_layers=*/1);
  receiver.ExpectSenders(kUuid, /*senders=*/2);

  BlockTransport a_transport(&sender_a, 0);
  BlockTransport b_transport(&sender_b, 0);
  BlockTransport receiver_transport(&receiver, 0);
  const std::string peer =
      absl::StrCat("localhost:", receiver_transport.local_port());
  auto ch_recv = StartControlServer(&receiver_transport);
  sender_a.SetPeerChannel(peer, ch_recv);
  sender_b.SetPeerChannel(peer, ch_recv);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Sender A splits its two blocks over two streams; sender B uses one.
  ASSERT_TRUE(a_transport
                  .SyncPush({peer}, /*src_block_ids=*/{0, 1},
                            /*dst_block_ids=*/{0, 1}, /*parallelism=*/2,
                            MajorOrder::kLayerMajor, kUuid, /*layer_idx=*/0)
                  .ok());
  EXPECT_EQ(receiver.layer_completion_count(), 0);
  ASSERT_TRUE(b_transport
                  .SyncPush({peer}, /*src_block_ids=*/{0, 1},
                            /*dst_block_ids=*/{0, 1}, /*parallelism=*/1,
                            MajorOrder::kLayerMajor, kUuid, /*layer_idx=*/0)
                  .ok());
  EXPECT_EQ(receiver.layer_completion_count(), 1);
}

TEST_F(BlockTransportTest, SenderOverDeliveryIsRejected) {
  constexpr uint64_t kUuid = 908;
  constexpr size_t kSliceSize = 32;
  constexpr size_t kNumLayers = 2;
  NodeDelegate sender_a(/*node_id=*/0, kSliceSize, /*max_blocks=*/1,
                        kNumLayers);
  NodeDelegate sender_b(/*node_id=*/1, kSliceSize, /*max_blocks=*/1,
                        kNumLayers);
  MultiSenderReceiverDelegate receiver(kSliceSize, /*max_blocks=*/1,
                                       kNumLayers);
  receiver.ExpectSenders(kUuid, /*senders=*/2);

  BlockTransport a_transport(&sender_a, 0);
  BlockTransport b_transport(&sender_b, 0);
  BlockTransport receiver_transport(&receiver, 0);
  const std::string peer =
      absl::StrCat("localhost:", receiver_transport.local_port());
  auto ch_recv = StartControlServer(&receiver_transport);
  sender_a.SetPeerChannel(peer, ch_recv);
  sender_b.SetPeerChannel(peer, ch_recv);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  auto push_layer = [&](BlockTransport& transport, int layer_idx) {
    return transport.SyncPush({peer}, /*src_block_ids=*/{0},
                              /*dst_block_ids=*/{0}, /*parallelism=*/1,
                              MajorOrder::kLayerMajor, kUuid, layer_idx);
  };

  ASSERT_TRUE(push_layer(a_transport, 0).ok());
  ASSERT_TRUE(push_layer(b_transport, 0).ok());
  EXPECT_EQ(receiver.layer_completion_count(), 1);

  // Layer 1 is still open, so the uuid's progress is live: a second stream
  // from sender A for layer 0 exceeds the one stream it declared.
  EXPECT_FALSE(push_layer(a_transport, 0).ok());
  EXPECT_EQ(receiver.layer_completion_count(), 1);
}

TEST_F(BlockTransportTest, SenderChangingDeclaredStreamCountIsRejected) {
  constexpr uint64_t kUuid = 909;
  constexpr size_t kSliceSize = 32;
  NodeDelegate sender_a(/*node_id=*/0, kSliceSize, /*max_blocks=*/2,
                        /*num_layers=*/1);
  MultiSenderReceiverDelegate receiver(kSliceSize, /*max_blocks=*/2,
                                       /*num_layers=*/1);
  receiver.ExpectSenders(kUuid, /*senders=*/2);

  BlockTransport a_transport(&sender_a, 0);
  BlockTransport receiver_transport(&receiver, 0);
  const std::string peer =
      absl::StrCat("localhost:", receiver_transport.local_port());
  auto ch_recv = StartControlServer(&receiver_transport);
  sender_a.SetPeerChannel(peer, ch_recv);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Sender A declares one stream for the array ...
  ASSERT_TRUE(a_transport
                  .SyncPush({peer}, /*src_block_ids=*/{0},
                            /*dst_block_ids=*/{0}, /*parallelism=*/1,
                            MajorOrder::kLayerMajor, kUuid, /*layer_idx=*/0)
                  .ok());
  EXPECT_EQ(receiver.layer_completion_count(), 0);
  // ... and then pushes the same array again declaring two.
  EXPECT_FALSE(a_transport
                   .SyncPush({peer}, /*src_block_ids=*/{0, 1},
                             /*dst_block_ids=*/{0, 1}, /*parallelism=*/2,
                             MajorOrder::kLayerMajor, kUuid, /*layer_idx=*/0)
                   .ok());
  EXPECT_EQ(receiver.layer_completion_count(), 0);
}

TEST_F(BlockTransportTest, UndeclaredUuidKeepsHeaderDeclaredCompletion) {
  constexpr uint64_t kUuid = 907;
  constexpr size_t kSliceSize = 32;
  NodeDelegate sender_a(/*node_id=*/0, kSliceSize, /*max_blocks=*/1,
                        /*num_layers=*/1);
  NodeDelegate sender_b(/*node_id=*/1, kSliceSize, /*max_blocks=*/1,
                        /*num_layers=*/1);
  MultiSenderReceiverDelegate receiver(kSliceSize, /*max_blocks=*/1,
                                       /*num_layers=*/1);
  receiver.ExpectSenders(/*uuid=*/kUuid + 1, /*senders=*/2);

  BlockTransport a_transport(&sender_a, 0);
  BlockTransport b_transport(&sender_b, 0);
  BlockTransport receiver_transport(&receiver, 0);
  const std::string peer =
      absl::StrCat("localhost:", receiver_transport.local_port());
  auto ch_recv = StartControlServer(&receiver_transport);
  sender_a.SetPeerChannel(peer, ch_recv);
  sender_b.SetPeerChannel(peer, ch_recv);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  auto push = [&](BlockTransport& transport) {
    return transport.SyncPush({peer}, /*src_block_ids=*/{0},
                              /*dst_block_ids=*/{0}, /*parallelism=*/1,
                              MajorOrder::kLayerMajor, kUuid,
                              /*layer_idx=*/0);
  };

  // Without a declared sender count each push completes the layer by itself.
  ASSERT_TRUE(push(a_transport).ok());
  EXPECT_EQ(receiver.layer_completion_count(), 1);
  ASSERT_TRUE(push(b_transport).ok());
  EXPECT_EQ(receiver.layer_completion_count(), 2);
}

TEST_F(BlockTransportTest, ForgetPushProgressAllowsUuidReuse) {
  constexpr uint64_t kUuid = 902;
  MockDelegate sender(/*slice_size=*/32, /*max_blocks=*/1,
                      /*num_layers=*/2);
  MockDelegate receiver(/*slice_size=*/32, /*max_blocks=*/1,
                        /*num_layers=*/2);
  BlockTransport sender_transport(&sender, 0);
  BlockTransport receiver_transport(&receiver, 0);
  BindControlChannels(&sender_transport, &sender, &receiver_transport,
                      &receiver);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  auto push_layer = [&](int layer_idx) {
    return sender_transport.SyncPush(
        {absl::StrCat("localhost:", receiver_transport.local_port())},
        /*src_block_ids=*/{0}, /*dst_block_ids=*/{0},
        /*parallelism=*/1, MajorOrder::kLayerMajor, kUuid, layer_idx);
  };

  ASSERT_TRUE(push_layer(0).ok());
  EXPECT_EQ(receiver.layer_completion_count(), 1);
  receiver_transport.ForgetPushProgress(kUuid);
  ASSERT_TRUE(push_layer(0).ok());
  EXPECT_EQ(receiver.layer_completion_count(), 2);

  // Finishing every layer retires progress automatically, so the same UUID
  // starts clean even without an explicit ForgetPushProgress call.
  ASSERT_TRUE(push_layer(1).ok());
  EXPECT_EQ(receiver.layer_completion_count(), 3);
  ASSERT_TRUE(push_layer(0).ok());
  EXPECT_EQ(receiver.layer_completion_count(), 4);
}

TEST_F(BlockTransportTest,
       PoolProgressWaitsForEveryStreamOfEveryDeclaredPoolAndRetires) {
  constexpr uint64_t kUuid = 903;
  MockDelegate sender(/*slice_size=*/32, /*max_blocks=*/1,
                      /*num_layers=*/2);
  MockDelegate receiver(/*slice_size=*/32, /*max_blocks=*/1,
                        /*num_layers=*/2);
  receiver.SetPoolPushProgress(kUuid, /*expected_pushes_per_pool=*/2,
                               /*transfer_pool_indices=*/{0, 1});
  BlockTransport sender_transport(&sender, 0);
  BlockTransport receiver_transport(&receiver, 0);
  BindControlChannels(&sender_transport, &sender, &receiver_transport,
                      &receiver);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  auto push_pool = [&](int pool_idx) {
    return sender_transport.SyncPush(
        {absl::StrCat("localhost:", receiver_transport.local_port())},
        /*src_block_ids=*/{0}, /*dst_block_ids=*/{0},
        /*parallelism=*/1, MajorOrder::kLayerMajor, kUuid, pool_idx);
  };

  ASSERT_TRUE(push_pool(0).ok());
  ASSERT_TRUE(push_pool(1).ok());
  EXPECT_EQ(receiver.pool_completion_count(), 0);
  EXPECT_EQ(receiver.layer_completion_count(), 0);

  ASSERT_TRUE(push_pool(0).ok());
  EXPECT_EQ(receiver.pool_completion_count(), 1);
  ASSERT_TRUE(push_pool(1).ok());
  EXPECT_EQ(receiver.pool_completion_count(), 2);

  // Completing the final declared pool retires all progress for the uuid. The
  // same UUID starts a fresh generation without an explicit Forget call.
  ASSERT_TRUE(push_pool(0).ok());
  EXPECT_EQ(receiver.pool_completion_count(), 2);
  ASSERT_TRUE(push_pool(0).ok());
  EXPECT_EQ(receiver.pool_completion_count(), 3);
}

TEST_F(BlockTransportTest, ForgetPushProgressResetsPartialPoolGeneration) {
  constexpr uint64_t kUuid = 904;
  MockDelegate sender(/*slice_size=*/32);
  MockDelegate receiver(/*slice_size=*/32);
  receiver.SetPoolPushProgress(kUuid, /*expected_pushes_per_pool=*/2,
                               /*transfer_pool_indices=*/{0});
  BlockTransport sender_transport(&sender, 0);
  BlockTransport receiver_transport(&receiver, 0);
  BindControlChannels(&sender_transport, &sender, &receiver_transport,
                      &receiver);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  auto push_once = [&]() {
    return sender_transport.SyncPush(
        {absl::StrCat("localhost:", receiver_transport.local_port())},
        /*src_block_ids=*/{0}, /*dst_block_ids=*/{0},
        /*parallelism=*/1, MajorOrder::kLayerMajor, kUuid, /*layer_idx=*/0);
  };

  ASSERT_TRUE(push_once().ok());
  EXPECT_EQ(receiver.pool_completion_count(), 0);
  receiver_transport.ForgetPushProgress(kUuid);
  ASSERT_TRUE(push_once().ok());
  EXPECT_EQ(receiver.pool_completion_count(), 0);
  ASSERT_TRUE(push_once().ok());
  EXPECT_EQ(receiver.pool_completion_count(), 1);
}

// TODO(yongx): re-enable this test.
#if 0
class MockBlockTransport : public BlockTransport {
 public:
  struct CallRecord {
    std::string peer;
    std::string local_ip;
  };

  MockBlockTransport(BlockTransportDelegate* delegate, int local_port,
                     const std::vector<std::string>& local_ips)
      : BlockTransport(delegate, local_port, local_ips) {}

  absl::StatusOr<int> BorrowConnection(absl::string_view peer,
                                       absl::string_view local_ip) override {
    absl::MutexLock lock(mock_mu_);
    acquire_calls_.push_back({std::string(peer), std::string(local_ip)});
    return absl::InternalError("mock_connection_halt");
  }

  std::vector<CallRecord> acquire_calls() {
    absl::MutexLock lock(mock_mu_);
    return acquire_calls_;
  }

 private:
  absl::Mutex mock_mu_;
  std::vector<CallRecord> acquire_calls_ ABSL_GUARDED_BY(mock_mu_);
};

TEST_F(BlockTransportTest, RoundRobinDistribution) {
  constexpr size_t kSliceSize = 16;
  MockDelegate delegate(kSliceSize);

  std::vector<std::string> local_ips = {"10.0.0.1", "10.0.0.2"};
  std::vector<std::string> peers = {"10.0.0.3:1234", "10.0.0.4:1234",
                                    "10.0.0.5:1234"};

  MockBlockTransport transport(&delegate, 0, local_ips);

  std::vector<int> src_blocks = {0, 1, 2, 3, 4, 5};
  auto res = transport.SyncPush(peers, src_blocks, /*dst_block_ids=*/{},
                                /*parallelism=*/6, MajorOrder::kLayerMajor,
                                /*uuid=*/0, /*layer_idx=*/-1);

  EXPECT_FALSE(res.ok());
  EXPECT_EQ(res.status().message(), "mock_connection_halt");

  auto calls = transport.acquire_calls();
  ASSERT_EQ(calls.size(), 6);

  std::vector<MockBlockTransport::CallRecord> expected = {
      {"10.0.0.3:1234", "10.0.0.1"}, {"10.0.0.4:1234", "10.0.0.2"},
      {"10.0.0.5:1234", "10.0.0.1"}, {"10.0.0.3:1234", "10.0.0.2"},
      {"10.0.0.4:1234", "10.0.0.1"}, {"10.0.0.5:1234", "10.0.0.2"}};

  auto compare = [](const MockBlockTransport::CallRecord& a,
                    const MockBlockTransport::CallRecord& b) {
    if (a.peer != b.peer) return a.peer < b.peer;
    return a.local_ip < b.local_ip;
  };

  std::sort(calls.begin(), calls.end(), compare);
  std::sort(expected.begin(), expected.end(), compare);

  for (size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(calls[i].peer, expected[i].peer);
    EXPECT_EQ(calls[i].local_ip, expected[i].local_ip);
  }
}
#endif

TEST_F(BlockTransportTest, SentBytesTelemetryIncrementedOnPushAndPull) {
  ScopedPrometheusBackend scoped_telemetry;

  constexpr size_t kSize = 1024;
  MockDelegate delegate1(kSize);
  MockDelegate delegate2(kSize);

  std::memset(delegate1.data(), 0xAB, kSize);
  std::memset(delegate2.data(), 0x00, kSize);

  BlockTransport transport1(&delegate1, 0);
  BlockTransport transport2(&delegate2, 0);
  BindControlChannels(&transport1, &delegate1, &transport2, &delegate2);

  auto push_res = transport1.SyncPush(
      {absl::StrCat("localhost:", transport2.local_port())},
      /*src_block_ids=*/{0}, /*dst_block_ids=*/{},
      /*parallelism=*/1, MajorOrder::kLayerMajor, /*uuid=*/0, /*layer_idx=*/-1);
  ASSERT_OK(push_res);

  constexpr absl::string_view kExpectedPushMetric =
      "tpu_raiden_sent_bytes_total{direction=\"push\"} 1024";
  const std::string snapshot1 = WaitForMetricSnapshot(kExpectedPushMetric);
  EXPECT_THAT(snapshot1, HasSubstr(kExpectedPushMetric));

  ASSERT_OK_AND_ASSIGN(
      auto pull_res,
      transport2.SyncPull({absl::StrCat("localhost:", transport1.local_port())},
                          /*src_block_ids=*/{0},
                          /*local_block_ids=*/{}, /*explicit_dst_ptrs=*/{},
                          /*parallelism=*/1, MajorOrder::kLayerMajor,
                          /*on_block_received=*/{}, /*uuid=*/0));

  constexpr absl::string_view kExpectedPullResponseMetric =
      "tpu_raiden_sent_bytes_total{direction=\"pull_response\"} 1024";
  const std::string snapshot2 =
      WaitForMetricSnapshot(kExpectedPullResponseMetric);
  EXPECT_THAT(snapshot2, HasSubstr(kExpectedPullResponseMetric));
}

TEST_F(BlockTransportTest, ReceivedBytesTelemetryIncrementedOnPushAndPull) {
  ScopedPrometheusBackend scoped_telemetry;

  constexpr size_t kSize = 1024;
  MockDelegate delegate1(kSize);
  MockDelegate delegate2(kSize);

  std::memset(delegate1.data(), 0xAB, kSize);
  std::memset(delegate2.data(), 0x00, kSize);

  BlockTransport transport1(&delegate1, 0);
  BlockTransport transport2(&delegate2, 0);
  BindControlChannels(&transport1, &delegate1, &transport2, &delegate2);

  auto push_res = transport1.SyncPush(
      {absl::StrCat("localhost:", transport2.local_port())},
      /*src_block_ids=*/{0}, /*dst_block_ids=*/{},
      /*parallelism=*/1, MajorOrder::kLayerMajor, /*uuid=*/0, /*layer_idx=*/-1);
  ASSERT_OK(push_res);

  constexpr absl::string_view kExpectedPushMetric =
      "tpu_raiden_received_bytes_total{direction=\"push\"} 1024";
  const std::string snapshot1 = WaitForMetricSnapshot(kExpectedPushMetric);
  EXPECT_THAT(snapshot1, HasSubstr(kExpectedPushMetric));
  EXPECT_THAT(snapshot1, Not(HasSubstr("direction=\"pull_response\"")));

  auto pull_res =
      transport2.SyncPull({absl::StrCat("localhost:", transport1.local_port())},
                          /*src_block_ids=*/{0},
                          /*local_block_ids=*/{}, /*explicit_dst_ptrs=*/{},
                          /*parallelism=*/1, MajorOrder::kLayerMajor,
                          /*on_block_received=*/{}, /*uuid=*/0);
  ASSERT_OK(pull_res);

  constexpr absl::string_view kExpectedPullResponseMetric =
      "tpu_raiden_received_bytes_total{direction=\"pull_response\"} 1024";
  const std::string snapshot2 =
      WaitForMetricSnapshot(kExpectedPullResponseMetric);
  EXPECT_THAT(
      snapshot2,
      HasSubstr("tpu_raiden_received_bytes_total{direction=\"push\"} 1024"));
  EXPECT_THAT(snapshot2, HasSubstr(kExpectedPullResponseMetric));
}

TEST_F(BlockTransportTest, TransferFailuresTelemetryPushValidationFailures) {
  ScopedPrometheusBackend scoped_telemetry;

  constexpr size_t kSize = 1024;
  MockDelegate delegate(kSize);
  BlockTransport transport(&delegate, 0);

  // 1. Empty block list (memory error)
  absl::StatusOr<std::vector<int>> res1 = transport.SyncPush(
      {"localhost:1234"}, /*src_block_ids=*/{}, /*dst_block_ids=*/{},
      /*parallelism=*/1, MajorOrder::kLayerMajor, /*uuid=*/0, /*layer_idx=*/-1);
  EXPECT_THAT(res1, StatusIs(absl::StatusCode::kInvalidArgument));

  constexpr absl::string_view kExpectedError1 =
      "tpu_raiden_transfer_failures_total{direction=\"push\","
      "error_code=\"INVALID_ARGUMENT\"} 1";
  EXPECT_THAT(WaitForMetricSnapshot(kExpectedError1),
              HasSubstr(kExpectedError1));

  // 2. Empty peers list (memory error)
  absl::StatusOr<std::vector<int>> res2 = transport.SyncPush(
      /*peers=*/{}, /*src_block_ids=*/{0}, /*dst_block_ids=*/{},
      /*parallelism=*/1, MajorOrder::kLayerMajor, /*uuid=*/0, /*layer_idx=*/-1);
  EXPECT_THAT(res2, StatusIs(absl::StatusCode::kInvalidArgument));

  constexpr absl::string_view kExpectedError2 =
      "tpu_raiden_transfer_failures_total{direction=\"push\","
      "error_code=\"INVALID_ARGUMENT\"} 2";
  EXPECT_THAT(WaitForMetricSnapshot(kExpectedError2),
              HasSubstr(kExpectedError2));

  // 3. Invalid parallelism (P <= 0)
  absl::StatusOr<std::vector<int>> res3 = transport.SyncPush(
      {"localhost:1234"}, /*src_block_ids=*/{0}, /*dst_block_ids=*/{},
      /*parallelism=*/0, MajorOrder::kLayerMajor, /*uuid=*/0, /*layer_idx=*/-1);
  EXPECT_THAT(res3, StatusIs(absl::StatusCode::kInvalidArgument));

  constexpr absl::string_view kExpectedError3 =
      "tpu_raiden_transfer_failures_total{direction=\"push\","
      "error_code=\"INVALID_ARGUMENT\"} 3";
  EXPECT_THAT(WaitForMetricSnapshot(kExpectedError3),
              HasSubstr(kExpectedError3));
}

TEST_F(BlockTransportTest, TransferFailuresTelemetryPushTransferFailure) {
  ScopedPrometheusBackend scoped_telemetry;

  constexpr size_t kSize = 1024;
  MockDelegate delegate(kSize);
  delegate.SetPeerChannel(
      "localhost:1",
      grpc::CreateChannel("localhost:1", grpc::InsecureChannelCredentials()));
  BlockTransport transport(&delegate, 0);

  absl::StatusOr<std::vector<int>> res = transport.SyncPush(
      {"localhost:1"}, /*src_block_ids=*/{0}, /*dst_block_ids=*/{},
      /*parallelism=*/1, MajorOrder::kLayerMajor, /*uuid=*/0, /*layer_idx=*/-1);
  EXPECT_THAT(res, StatusIs(absl::StatusCode::kUnavailable));

  constexpr absl::string_view kExpectedError =
      "tpu_raiden_transfer_failures_total{direction=\"push\","
      "error_code=\"UNAVAILABLE\"} 1";
  EXPECT_THAT(WaitForMetricSnapshot(kExpectedError), HasSubstr(kExpectedError));
}

TEST_F(BlockTransportTest,
       TransferFailuresTelemetryPushBufferAndPushBuffers) {
  ScopedPrometheusBackend scoped_telemetry;

  constexpr size_t kSize = 1024;
  MockDelegate delegate(kSize);
  delegate.SetPeerChannel(
      "localhost:1",
      grpc::CreateChannel("localhost:1", grpc::InsecureChannelCredentials()));
  BlockTransport transport(&delegate, 0);

  // PushBuffer failure to non-existent peer
  uint8_t dummy_data[64] = {0};
  absl::Status status1 = transport.PushBuffer(
      "localhost:1", /*buffer_id=*/0, /*dst_shard_idx=*/0,
      /*dst_offset_bytes=*/0, dummy_data, sizeof(dummy_data));
  EXPECT_THAT(status1, StatusIs(absl::StatusCode::kUnavailable));

  constexpr absl::string_view kExpectedError1 =
      "tpu_raiden_transfer_failures_total{direction=\"push\","
      "error_code=\"UNAVAILABLE\"} 1";
  EXPECT_THAT(WaitForMetricSnapshot(kExpectedError1),
              HasSubstr(kExpectedError1));

  // PushBuffers failure with unreachable peer
  BufferPushTask task = {
      .peer = "localhost:1",
      .buffer_id = 0,
      .dst_shard_idx = 0,
      .dst_offset_bytes = 0,
      .data_ptr = dummy_data,
      .size_bytes = sizeof(dummy_data),
  };
  absl::Status status2 = transport.PushBuffers({task}, /*parallelism=*/1, 0);
  EXPECT_THAT(status2, StatusIs(absl::StatusCode::kUnavailable));

  constexpr absl::string_view kExpectedError2 =
      "tpu_raiden_transfer_failures_total{direction=\"push\","
      "error_code=\"UNAVAILABLE\"} 2";
  EXPECT_THAT(WaitForMetricSnapshot(kExpectedError2),
              HasSubstr(kExpectedError2));
}

TEST_F(BlockTransportTest, TransferFailuresTelemetryPullValidationFailures) {
  ScopedPrometheusBackend scoped_telemetry;

  constexpr size_t kSize = 1024;
  MockDelegate delegate(kSize);
  BlockTransport transport(&delegate, 0);

  // 1. Empty block list
  absl::StatusOr<std::vector<int>> res1 = transport.SyncPull(
      {"localhost:1234"}, /*src_block_ids=*/{},
      /*local_block_ids=*/{}, /*explicit_dst_ptrs=*/{}, /*parallelism=*/1,
      MajorOrder::kLayerMajor, /*on_block_received=*/{}, /*uuid=*/0);
  EXPECT_THAT(res1, StatusIs(absl::StatusCode::kInvalidArgument));

  constexpr absl::string_view kExpectedError1 =
      "tpu_raiden_transfer_failures_total{direction=\"pull\","
      "error_code=\"INVALID_ARGUMENT\"} 1";
  EXPECT_THAT(WaitForMetricSnapshot(kExpectedError1),
              HasSubstr(kExpectedError1));

  // 2. Empty peers
  absl::StatusOr<std::vector<int>> res2 = transport.SyncPull(
      /*peers=*/{}, /*src_block_ids=*/{0},
      /*local_block_ids=*/{}, /*explicit_dst_ptrs=*/{}, /*parallelism=*/1,
      MajorOrder::kLayerMajor, /*on_block_received=*/{}, /*uuid=*/0);
  EXPECT_THAT(res2, StatusIs(absl::StatusCode::kInvalidArgument));

  constexpr absl::string_view kExpectedError2 =
      "tpu_raiden_transfer_failures_total{direction=\"pull\","
      "error_code=\"INVALID_ARGUMENT\"} 2";
  EXPECT_THAT(WaitForMetricSnapshot(kExpectedError2),
              HasSubstr(kExpectedError2));

  // 3. Invalid parallelism (P <= 0)
  absl::StatusOr<std::vector<int>> res3 = transport.SyncPull(
      {"localhost:1234"}, /*src_block_ids=*/{0},
      /*local_block_ids=*/{}, /*explicit_dst_ptrs=*/{}, /*parallelism=*/0,
      MajorOrder::kLayerMajor, /*on_block_received=*/{}, /*uuid=*/0);
  EXPECT_THAT(res3, StatusIs(absl::StatusCode::kInvalidArgument));

  constexpr absl::string_view kExpectedError3 =
      "tpu_raiden_transfer_failures_total{direction=\"pull\","
      "error_code=\"INVALID_ARGUMENT\"} 3";
  EXPECT_THAT(WaitForMetricSnapshot(kExpectedError3),
              HasSubstr(kExpectedError3));

  // 4. explicit_dst_ptrs size mismatch
  uint8_t dummy_dst[16] = {0};
  absl::StatusOr<std::vector<int>> res4 = transport.SyncPull(
      {"localhost:1234"}, /*src_block_ids=*/{0},
      /*local_block_ids=*/{}, /*explicit_dst_ptrs=*/{dummy_dst, dummy_dst},
      /*parallelism=*/1, MajorOrder::kLayerMajor, /*on_block_received=*/{},
      /*uuid=*/0);
  EXPECT_THAT(res4, StatusIs(absl::StatusCode::kInvalidArgument));

  constexpr absl::string_view kExpectedError4 =
      "tpu_raiden_transfer_failures_total{direction=\"pull\","
      "error_code=\"INVALID_ARGUMENT\"} 4";
  EXPECT_THAT(WaitForMetricSnapshot(kExpectedError4),
              HasSubstr(kExpectedError4));

  // 5. local_block_ids size mismatch
  absl::StatusOr<std::vector<int>> res5 = transport.SyncPull(
      {"localhost:1234"}, /*src_block_ids=*/{0},
      /*local_block_ids=*/{0, 1}, /*explicit_dst_ptrs=*/{},
      /*parallelism=*/1, MajorOrder::kLayerMajor, /*on_block_received=*/{},
      /*uuid=*/0);
  EXPECT_THAT(res5, StatusIs(absl::StatusCode::kInvalidArgument));

  constexpr absl::string_view kExpectedError5 =
      "tpu_raiden_transfer_failures_total{direction=\"pull\","
      "error_code=\"INVALID_ARGUMENT\"} 5";
  EXPECT_THAT(WaitForMetricSnapshot(kExpectedError5),
              HasSubstr(kExpectedError5));
}

TEST_F(BlockTransportTest, TransferFailuresTelemetryPullTransferFailure) {
  ScopedPrometheusBackend scoped_telemetry;

  constexpr size_t kSliceSize = 16;
  constexpr int kNumBlocks = 1;
  MockDelegate source(kSliceSize, kNumBlocks);
  MockDelegate receiver(kSliceSize, kNumBlocks);

  BlockTransport source_transport(&source, 0);
  BlockTransport receiver_transport(&receiver, 0);
  BindControlChannels(&source_transport, &source, &receiver_transport,
                      &receiver);

  absl::StatusOr<std::vector<int>> pull_res = receiver_transport.SyncPull(
      {absl::StrCat("localhost:", source_transport.local_port())},
      /*src_block_ids=*/{1},
      /*local_block_ids=*/{0}, /*explicit_dst_ptrs=*/{},
      /*parallelism=*/1, MajorOrder::kLayerMajor,
      /*on_block_received=*/{}, /*uuid=*/0);
  EXPECT_FALSE(pull_res.ok());

  constexpr absl::string_view kExpectedError =
      "tpu_raiden_transfer_failures_total{direction=\"pull\",";
  EXPECT_THAT(WaitForMetricSnapshot(kExpectedError), HasSubstr(kExpectedError));
}

TEST_F(BlockTransportTest, NoTransferFailuresTelemetryOnSuccess) {
  ScopedPrometheusBackend scoped_telemetry;

  constexpr size_t kSize = 1024;
  MockDelegate delegate1(kSize);
  MockDelegate delegate2(kSize);

  std::memset(delegate1.data(), 0xAB, kSize);
  std::memset(delegate2.data(), 0x00, kSize);

  BlockTransport transport1(&delegate1, 0);
  BlockTransport transport2(&delegate2, 0);
  BindControlChannels(&transport1, &delegate1, &transport2, &delegate2);

  ASSERT_OK(
      transport1.SyncPush({absl::StrCat("localhost:", transport2.local_port())},
                          /*src_block_ids=*/{0},
                          /*dst_block_ids=*/{},
                          /*parallelism=*/1, MajorOrder::kLayerMajor,
                          /*uuid=*/0, /*layer_idx=*/-1));

  ASSERT_OK(
      transport2.SyncPull({absl::StrCat("localhost:", transport1.local_port())},
                          /*src_block_ids=*/{0},
                          /*local_block_ids=*/{},
                          /*explicit_dst_ptrs=*/{},
                          /*parallelism=*/1, MajorOrder::kLayerMajor,
                          /*on_block_received=*/{}, /*uuid=*/0));

  constexpr absl::string_view kNotExpectedError =
      "tpu_raiden_transfer_failures_total{";
  EXPECT_THAT(WaitForMetricSnapshot(kNotExpectedError),
              Not(HasSubstr(kNotExpectedError)));
}

TEST_F(BlockTransportTest, MultiShardPushBlockMajor) {
  constexpr size_t kBlockSize = 256;
  constexpr int kNumBlocks = 3;
  constexpr size_t kNumLayers = 2;
  constexpr size_t kNumShards = 2;

  MockDelegate sender_delegate(kBlockSize, kNumBlocks, kNumLayers, kNumShards);
  MockDelegate receiver_delegate(kBlockSize, kNumBlocks, kNumLayers,
                                 kNumShards);

  for (size_t l = 0; l < kNumLayers; ++l) {
    for (size_t sh = 0; sh < kNumShards; ++sh) {
      for (int b = 0; b < kNumBlocks; ++b) {
        std::memset(sender_delegate.block_data(b, l, sh),
                    static_cast<int>(0x10 + l * 0x20 + sh * 0x08 + b),
                    kBlockSize);
        std::memset(receiver_delegate.block_data(b, l, sh), 0, kBlockSize);
      }
    }
  }

  BlockTransport sender(&sender_delegate, 0);
  BlockTransport receiver(&receiver_delegate, 0);
  BindControlChannels(&sender, &sender_delegate, &receiver,
                      &receiver_delegate);

  ASSERT_OK(
      sender.SyncPush({absl::StrCat("localhost:", receiver.local_port())},
                      /*src_block_ids=*/{0, 1, 2}, /*dst_block_ids=*/{0, 1, 2},
                      /*parallelism=*/1, MajorOrder::kBlockMajor, /*uuid=*/0,
                      /*layer_idx=*/-1));

  for (size_t l = 0; l < kNumLayers; ++l) {
    for (size_t sh = 0; sh < kNumShards; ++sh) {
      for (int b = 0; b < kNumBlocks; ++b) {
        int expected = static_cast<int>(0x10 + l * 0x20 + sh * 0x08 + b);
        EXPECT_EQ(receiver_delegate.block_data(b, l, sh)[0], expected);
        EXPECT_EQ(receiver_delegate.block_data(b, l, sh)[kBlockSize - 1],
                  expected);
      }
    }
  }
}

TEST_F(BlockTransportTest, MultiShardPushLayerMajor) {
  constexpr size_t kBlockSize = 256;
  constexpr int kNumBlocks = 3;
  constexpr size_t kNumLayers = 2;
  constexpr size_t kNumShards = 2;

  MockDelegate sender_delegate(kBlockSize, kNumBlocks, kNumLayers, kNumShards);
  MockDelegate receiver_delegate(kBlockSize, kNumBlocks, kNumLayers,
                                 kNumShards);

  for (size_t l = 0; l < kNumLayers; ++l) {
    for (size_t sh = 0; sh < kNumShards; ++sh) {
      for (int b = 0; b < kNumBlocks; ++b) {
        std::memset(sender_delegate.block_data(b, l, sh),
                    static_cast<int>(0x10 + l * 0x20 + sh * 0x08 + b),
                    kBlockSize);
        std::memset(receiver_delegate.block_data(b, l, sh), 0, kBlockSize);
      }
    }
  }

  BlockTransport sender(&sender_delegate, 0);
  BlockTransport receiver(&receiver_delegate, 0);
  BindControlChannels(&sender, &sender_delegate, &receiver,
                      &receiver_delegate);

  ASSERT_OK(
      sender.SyncPush({absl::StrCat("localhost:", receiver.local_port())},
                      /*src_block_ids=*/{0, 1, 2}, /*dst_block_ids=*/{0, 1, 2},
                      /*parallelism=*/1, MajorOrder::kLayerMajor, /*uuid=*/0,
                      /*layer_idx=*/-1));

  for (size_t l = 0; l < kNumLayers; ++l) {
    for (size_t sh = 0; sh < kNumShards; ++sh) {
      for (int b = 0; b < kNumBlocks; ++b) {
        int expected = static_cast<int>(0x10 + l * 0x20 + sh * 0x08 + b);
        EXPECT_EQ(receiver_delegate.block_data(b, l, sh)[0], expected);
        EXPECT_EQ(receiver_delegate.block_data(b, l, sh)[kBlockSize - 1],
                  expected);
      }
    }
  }
}

TEST_F(BlockTransportTest, MultiShardPullBlockMajor) {
  constexpr size_t kBlockSize = 256;
  constexpr int kNumBlocks = 3;
  constexpr size_t kNumLayers = 2;
  constexpr size_t kNumShards = 2;

  MockDelegate sender_delegate(kBlockSize, kNumBlocks, kNumLayers, kNumShards);
  MockDelegate receiver_delegate(kBlockSize, kNumBlocks, kNumLayers,
                                 kNumShards);

  for (size_t l = 0; l < kNumLayers; ++l) {
    for (size_t sh = 0; sh < kNumShards; ++sh) {
      for (int b = 0; b < kNumBlocks; ++b) {
        std::memset(sender_delegate.block_data(b, l, sh),
                    static_cast<int>(0x10 + l * 0x20 + sh * 0x08 + b),
                    kBlockSize);
        std::memset(receiver_delegate.block_data(b, l, sh), 0, kBlockSize);
      }
    }
  }

  BlockTransport sender(&sender_delegate, 0);
  BlockTransport receiver(&receiver_delegate, 0);
  BindControlChannels(&sender, &sender_delegate, &receiver,
                      &receiver_delegate);

  ASSERT_OK(receiver.SyncPull(
      {absl::StrCat("localhost:", sender.local_port())},
      /*src_block_ids=*/{0, 1, 2}, /*local_block_ids=*/{0, 1, 2},
      /*explicit_dst_ptrs=*/{}, /*parallelism=*/1, MajorOrder::kBlockMajor,
      /*on_block_received=*/{}, /*uuid=*/0));

  for (size_t l = 0; l < kNumLayers; ++l) {
    for (size_t sh = 0; sh < kNumShards; ++sh) {
      for (int b = 0; b < kNumBlocks; ++b) {
        int expected = static_cast<int>(0x10 + l * 0x20 + sh * 0x08 + b);
        EXPECT_EQ(receiver_delegate.block_data(b, l, sh)[0], expected);
        EXPECT_EQ(receiver_delegate.block_data(b, l, sh)[kBlockSize - 1],
                  expected);
      }
    }
  }
}

TEST_F(BlockTransportTest, MultiShardPullLayerMajor) {
  constexpr size_t kBlockSize = 256;
  constexpr int kNumBlocks = 3;
  constexpr size_t kNumLayers = 2;
  constexpr size_t kNumShards = 2;

  MockDelegate sender_delegate(kBlockSize, kNumBlocks, kNumLayers, kNumShards);
  MockDelegate receiver_delegate(kBlockSize, kNumBlocks, kNumLayers,
                                 kNumShards);

  for (size_t l = 0; l < kNumLayers; ++l) {
    for (size_t sh = 0; sh < kNumShards; ++sh) {
      for (int b = 0; b < kNumBlocks; ++b) {
        std::memset(sender_delegate.block_data(b, l, sh),
                    static_cast<int>(0x10 + l * 0x20 + sh * 0x08 + b),
                    kBlockSize);
        std::memset(receiver_delegate.block_data(b, l, sh), 0, kBlockSize);
      }
    }
  }

  BlockTransport sender(&sender_delegate, 0);
  BlockTransport receiver(&receiver_delegate, 0);
  BindControlChannels(&sender, &sender_delegate, &receiver,
                      &receiver_delegate);

  ASSERT_OK(receiver.SyncPull(
      {absl::StrCat("localhost:", sender.local_port())},
      /*src_block_ids=*/{0, 1, 2}, /*local_block_ids=*/{0, 1, 2},
      /*explicit_dst_ptrs=*/{}, /*parallelism=*/1, MajorOrder::kLayerMajor,
      /*on_block_received=*/{}, /*uuid=*/0));

  for (size_t l = 0; l < kNumLayers; ++l) {
    for (size_t sh = 0; sh < kNumShards; ++sh) {
      for (int b = 0; b < kNumBlocks; ++b) {
        int expected = static_cast<int>(0x10 + l * 0x20 + sh * 0x08 + b);
        EXPECT_EQ(receiver_delegate.block_data(b, l, sh)[0], expected);
        EXPECT_EQ(receiver_delegate.block_data(b, l, sh)[kBlockSize - 1],
                  expected);
      }
    }
  }
}

TEST_F(BlockTransportTest, PushBufferCorrectness) {
  constexpr size_t size = 64 * 1024;
  MockDelegate src(size);
  MockDelegate dst(size);

  BlockTransport src_transport(&src, 0);
  BlockTransport dst_transport(&dst, 0);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  constexpr size_t kLen = 62 * 1024;
  constexpr size_t kDstOffset = 512;
  std::vector<uint8_t> push_payload(kLen);
  for (size_t i = 0; i < kLen; ++i) {
    push_payload[i] = static_cast<uint8_t>((i % 255) + 1);
  }
  const std::string dst_addr =
      absl::StrCat("localhost:", dst_transport.local_port());
  const auto push_res = src_transport.PushBuffer(
      dst_addr, /*buffer_id=*/0, /*dst_shard_idx=*/0,
      /*dst_offset_bytes=*/kDstOffset, push_payload.data(), push_payload.size(),
      /*uuid=*/0);
  EXPECT_OK(push_res) << push_res.message();

  const uint8_t* dst_buf = dst.GetHostPointer(0, 0);
  EXPECT_THAT(absl::MakeConstSpan(dst_buf, kDstOffset), Each(Eq(0)));
  EXPECT_THAT(absl::MakeConstSpan(dst_buf + kDstOffset, kLen),
              Pointwise(Eq(), absl::MakeConstSpan(push_payload)));
  EXPECT_THAT(absl::MakeConstSpan(dst_buf + kDstOffset + kLen,
                                  size - kDstOffset - kLen),
              Each(Eq(0)));
}

TEST_F(BlockTransportTest, PollEINTRIsBenign) {
  // Set up src/dst buffers.
  constexpr size_t size = 4096;
  MockDelegate src(size);
  MockDelegate dst(size);

  // Create two transports.
  BlockTransport src_transport(&src, 0);
  BlockTransport dst_transport(&dst, 0);
  BindControlChannels(&src_transport, &src, &dst_transport, &dst);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Register a dummy signal handler.
  signal(SIGUSR1, [](int) {});
  // Send a signal to the process to interrupt some poll() calls with EINTR.
  kill(getpid(), SIGUSR1);

  // Perform a push to verify the connection worker didn't die.
  const std::string dst_addr =
      absl::StrCat("localhost:", dst_transport.local_port());
  const std::vector<uint8_t> push_payload(1024, 0xAB);
  constexpr size_t kDstOffset = 512;
  const auto push_res = src_transport.PushBuffer(
      dst_addr, /*buffer_id=*/0, /*dst_shard_idx=*/0,
      /*dst_offset_bytes=*/kDstOffset, push_payload.data(), push_payload.size(),
      /*uuid=*/0);
  EXPECT_OK(push_res) << push_res.message();
}

}  // namespace
}  // namespace transport
}  // namespace tpu_raiden
