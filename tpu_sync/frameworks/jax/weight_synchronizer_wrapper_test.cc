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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/frameworks/jax/weight_synchronizer.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace jax {
namespace {

class MockSubWeightSynchronizer : public weight_sync::WeightSynchronizerBase {
 public:
  MockSubWeightSynchronizer(size_t num_layers, size_t num_shards,
                            size_t slice_byte_size)
      : WeightSynchronizerBase(num_layers, num_shards, slice_byte_size,
                               /*local_port=*/std::nullopt,
                               /*host_blocks_to_allocate=*/std::nullopt,
                               /*parallelism=*/1,
                               /*listener_port=*/std::nullopt,
                               /*bind_ip=*/std::nullopt,
                               /*layer_names=*/{}, /*auto_h2d=*/false) {}

  int d2h_calls = 0;
  int h2d_calls = 0;
  int push_calls = 0;
  int push_resharded_calls = 0;
  std::vector<std::string> last_push_peers;

  absl::StatusOr<raiden::PjRtCopyFuture> D2h(uint64_t uuid = 0) override {
    d2h_calls++;
    return raiden::PjRtCopyFuture();
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2d(uint64_t uuid = 0) override {
    h2d_calls++;
    return raiden::PjRtCopyFuture();
  }

  absl::Status PushWeights(const std::vector<std::string>& peers) override {
    push_calls++;
    last_push_peers = peers;
    return absl::OkStatus();
  }

  absl::Status PushWeightsLocal(
      const std::vector<std::string>& peers) override {
    push_calls++;
    last_push_peers = peers;
    return absl::OkStatus();
  }

  absl::Status PushWeightsResharded(
      const tpu_sync::rpc::StartTransferRequest& request) override {
    push_resharded_calls++;
    return absl::OkStatus();
  }

  absl::Status PushWeightsReshardedLocal(
      const tpu_sync::rpc::StartTransferRequest& request) override {
    push_resharded_calls++;
    return absl::OkStatus();
  }

  uint32_t last_registered_chunks = 0;
  absl::flat_hash_map<size_t, uint32_t> last_registered_layer_chunks;
  int wait_completion_calls = 0;
  int drain_pending_h2d_calls = 0;

  absl::Status RegisterExpectedChunksLocal(uint64_t uuid,
                                           uint32_t expected_chunks) override {
    last_registered_chunks = expected_chunks;
    return absl::OkStatus();
  }

  absl::Status RegisterExpectedLayerChunksLocal(
      uint64_t uuid,
      const absl::flat_hash_map<size_t, uint32_t>& expected_layer_chunks)
      override {
    last_registered_layer_chunks = expected_layer_chunks;
    return absl::OkStatus();
  }

  absl::Status WaitForTransferCompletion(uint64_t uuid) override {
    wait_completion_calls++;
    return absl::OkStatus();
  }

  void DrainPendingH2d() override { drain_pending_h2d_calls++; }

  void SetMockMetrics(const weight_sync::WeightSyncMetrics& m) {
    SetMetricsForTesting(m);
  }
};

TEST(WeightSynchronizerWrapperTest, MockInjectionAndShardMapping) {
  auto sub0 = std::make_unique<MockSubWeightSynchronizer>(2, 4, 1024);
  auto sub1 = std::make_unique<MockSubWeightSynchronizer>(2, 4, 1024);

  std::vector<std::unique_ptr<weight_sync::WeightSynchronizerBase>> subs;
  subs.push_back(std::move(sub0));
  subs.push_back(std::move(sub1));

  WeightSynchronizer ws(std::move(subs));

  EXPECT_EQ(ws.num_layers(), 2);
  EXPECT_EQ(ws.num_shards(), 8);
  EXPECT_EQ(ws.slice_byte_size(), 1024);

  // Verify host pointer access maps across both sub-managers without crashing
  for (size_t l = 0; l < ws.num_layers(); ++l) {
    for (size_t s = 0; s < ws.num_shards(); ++s) {
      EXPECT_NE(ws.GetHostBufferPtr(l, s), nullptr);
    }
  }
}

TEST(WeightSynchronizerWrapperTest, ConcurrentD2hAndH2dExecution) {
  auto sub0_raw = new MockSubWeightSynchronizer(2, 4, 1024);
  auto sub1_raw = new MockSubWeightSynchronizer(2, 4, 1024);

  std::vector<std::unique_ptr<weight_sync::WeightSynchronizerBase>> subs;
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub0_raw));
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub1_raw));

  WeightSynchronizer ws(std::move(subs));

  auto d2h_status = ws.D2h();
  EXPECT_TRUE(d2h_status.ok());
  EXPECT_EQ(sub0_raw->d2h_calls, 1);
  EXPECT_EQ(sub1_raw->d2h_calls, 1);

  auto h2d_status = ws.H2d();
  EXPECT_TRUE(h2d_status.ok());
  EXPECT_EQ(sub0_raw->h2d_calls, 1);
  EXPECT_EQ(sub1_raw->h2d_calls, 1);
}

TEST(WeightSynchronizerWrapperTest, MetricsAggregationAcrossSubManagers) {
  auto sub0_raw = new MockSubWeightSynchronizer(2, 4, 1024);
  auto sub1_raw = new MockSubWeightSynchronizer(2, 4, 1024);

  weight_sync::WeightSyncMetrics m0;
  m0.last_d2h_time_ms = 10.0;
  m0.last_h2h_time_ms = 50.0;
  m0.last_d2h_bytes = 1000;
  m0.last_h2h_bytes = 2000;
  sub0_raw->SetMockMetrics(m0);

  weight_sync::WeightSyncMetrics m1;
  m1.last_d2h_time_ms = 15.0;
  m1.last_h2h_time_ms = 45.0;
  m1.last_d2h_bytes = 1500;
  m1.last_h2h_bytes = 3000;
  sub1_raw->SetMockMetrics(m1);

  std::vector<std::unique_ptr<weight_sync::WeightSynchronizerBase>> subs;
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub0_raw));
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub1_raw));

  WeightSynchronizer ws(std::move(subs));

  weight_sync::WeightSyncMetrics agg = ws.GetMetrics();

  // Max of durations
  EXPECT_DOUBLE_EQ(agg.last_d2h_time_ms, 15.0);
  EXPECT_DOUBLE_EQ(agg.last_h2h_time_ms, 50.0);

  // Sum of bytes
  EXPECT_EQ(agg.last_d2h_bytes, 2500);
  EXPECT_EQ(agg.last_h2h_bytes, 5000);
}

TEST(WeightSynchronizerWrapperTest, PushWeightsReshardedDispatchesToAllSubs) {
  auto sub0_raw = new MockSubWeightSynchronizer(2, 4, 1024);
  auto sub1_raw = new MockSubWeightSynchronizer(2, 4, 1024);

  std::vector<std::unique_ptr<weight_sync::WeightSynchronizerBase>> subs;
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub0_raw));
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub1_raw));

  NumaAwareWeightSynchronizer numa_ws(std::move(subs));

  tpu_sync::rpc::StartTransferRequest req;
  req.set_is_sender(true);

  absl::Status status = numa_ws.PushWeightsResharded(req);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(sub0_raw->push_resharded_calls, 1);
  EXPECT_EQ(sub1_raw->push_resharded_calls, 1);
}

TEST(WeightSynchronizerWrapperTest, GetLocalEndpointsAggregatesSubManagers) {
  auto sub0 = std::make_unique<MockSubWeightSynchronizer>(2, 4, 1024);
  auto sub1 = std::make_unique<MockSubWeightSynchronizer>(2, 4, 1024);

  std::vector<std::unique_ptr<weight_sync::WeightSynchronizerBase>> subs;
  subs.push_back(std::move(sub0));
  subs.push_back(std::move(sub1));

  WeightSynchronizer ws(std::move(subs));
  auto eps = ws.get_local_endpoints();
  ASSERT_EQ(eps.size(), 2);
  EXPECT_EQ(eps[0].shards, (std::vector<int64_t>{0, 1, 2, 3}));
  EXPECT_EQ(eps[1].shards, (std::vector<int64_t>{4, 5, 6, 7}));
}

TEST(WeightSynchronizerWrapperTest,
     RegisterExpectedCountsWithShardPushSchedulesPartitionsCorrectly) {
  auto sub0_raw = new MockSubWeightSynchronizer(2, 4, 1024);
  auto sub1_raw = new MockSubWeightSynchronizer(2, 4, 1024);

  std::vector<std::unique_ptr<weight_sync::WeightSynchronizerBase>> subs;
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub0_raw));
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub1_raw));

  NumaAwareWeightSynchronizer numa_ws(std::move(subs));

  tpu_sync::rpc::StartTransferRequest req;
  req.set_uuid(42);

  // Add schedules: 4 chunks targeting sub0 (shards 0..3) and 2 chunks targeting
  // sub1 (shards 4..5)
  auto* sched_proto = req.mutable_shard_push_schedules();
  auto& s0 = (*sched_proto)[0];
  for (int sh = 0; sh < 4; ++sh) {
    auto* entry = s0.add_entries();
    entry->set_dst_shard_idx(sh);
    entry->set_layer_idx(0);
    entry->set_count(1);
    entry->set_size_bytes(1024);
    entry->set_src_stride_bytes(1024);
    entry->set_dst_stride_bytes(1024);
  }
  for (int sh = 4; sh < 6; ++sh) {
    auto* entry = s0.add_entries();
    entry->set_dst_shard_idx(sh);
    entry->set_layer_idx(1);
    entry->set_count(1);
    entry->set_size_bytes(1024);
    entry->set_src_stride_bytes(1024);
    entry->set_dst_stride_bytes(1024);
  }

  numa_ws.StoreSkipTiling(42, req);

  absl::flat_hash_map<size_t, uint32_t> layer_counts = {{0, 4}, {1, 2}};
  EXPECT_TRUE(numa_ws.RegisterExpectedLayerChunks(42, layer_counts).ok());
  EXPECT_TRUE(numa_ws.RegisterExpectedChunks(42, 6).ok());

  // sub0 should receive 4 chunks for layer 0, total 4 chunks
  EXPECT_EQ(sub0_raw->last_registered_chunks, 4);
  EXPECT_EQ(sub0_raw->last_registered_layer_chunks[0], 4);
  EXPECT_EQ(sub0_raw->last_registered_layer_chunks[1], 0);

  // sub1 should receive 2 chunks for layer 1, total 2 chunks
  EXPECT_EQ(sub1_raw->last_registered_chunks, 2);
  EXPECT_EQ(sub1_raw->last_registered_layer_chunks[0], 0);
  EXPECT_EQ(sub1_raw->last_registered_layer_chunks[1], 2);
}

TEST(WeightSynchronizerWrapperTest,
     RegisterExpectedCountsProportionalFallback) {
  auto sub0_raw = new MockSubWeightSynchronizer(2, 4, 1024);
  auto sub1_raw = new MockSubWeightSynchronizer(2, 4, 1024);

  std::vector<std::unique_ptr<weight_sync::WeightSynchronizerBase>> subs;
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub0_raw));
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub1_raw));

  NumaAwareWeightSynchronizer numa_ws(std::move(subs));

  // No StartTransferRequest schedules stored: test proportional fallback (4
  // shards each of 8 total = 50%)
  absl::flat_hash_map<size_t, uint32_t> layer_counts = {{0, 8}, {1, 16}};
  EXPECT_TRUE(numa_ws.RegisterExpectedLayerChunks(99, layer_counts).ok());
  EXPECT_TRUE(numa_ws.RegisterExpectedChunks(99, 24).ok());

  EXPECT_EQ(sub0_raw->last_registered_chunks, 12);
  EXPECT_EQ(sub0_raw->last_registered_layer_chunks[0], 4);
  EXPECT_EQ(sub0_raw->last_registered_layer_chunks[1], 8);

  EXPECT_EQ(sub1_raw->last_registered_chunks, 12);
  EXPECT_EQ(sub1_raw->last_registered_layer_chunks[0], 4);
  EXPECT_EQ(sub1_raw->last_registered_layer_chunks[1], 8);
}

TEST(WeightSynchronizerWrapperTest,
     WaitForTransferCompletionDispatchesToAllSubs) {
  auto sub0_raw = new MockSubWeightSynchronizer(2, 4, 1024);
  auto sub1_raw = new MockSubWeightSynchronizer(2, 4, 1024);

  std::vector<std::unique_ptr<weight_sync::WeightSynchronizerBase>> subs;
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub0_raw));
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub1_raw));

  NumaAwareWeightSynchronizer numa_ws(std::move(subs));

  EXPECT_TRUE(numa_ws.WaitForTransferCompletion(100).ok());
  EXPECT_EQ(sub0_raw->wait_completion_calls, 1);
  EXPECT_EQ(sub1_raw->wait_completion_calls, 1);
}

TEST(WeightSynchronizerWrapperTest,
     RegisterExpectedCountsWithReplicatedTensors) {
  auto sub0_raw = new MockSubWeightSynchronizer(2, 4, 1024);
  auto sub1_raw = new MockSubWeightSynchronizer(2, 4, 1024);

  std::vector<std::unique_ptr<weight_sync::WeightSynchronizerBase>> subs;
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub0_raw));
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub1_raw));

  NumaAwareWeightSynchronizer numa_ws(std::move(subs));

  // 1 chunk for a replicated layer (e.g. 1D layer norm across 8 shards)
  absl::flat_hash_map<size_t, uint32_t> layer_counts = {{0, 1}};
  EXPECT_TRUE(numa_ws.RegisterExpectedLayerChunks(101, layer_counts).ok());
  EXPECT_TRUE(numa_ws.RegisterExpectedChunks(101, 1).ok());

  // Both sub0 and sub1 must register count 1 for layer 0, not drop to 0
  EXPECT_EQ(sub0_raw->last_registered_chunks, 1);
  EXPECT_EQ(sub0_raw->last_registered_layer_chunks[0], 1);

  EXPECT_EQ(sub1_raw->last_registered_chunks, 1);
  EXPECT_EQ(sub1_raw->last_registered_layer_chunks[0], 1);
}

TEST(WeightSynchronizerWrapperTest, DrainPendingH2dDispatchesToAllSubs) {
  auto sub0_raw = new MockSubWeightSynchronizer(2, 4, 1024);
  auto sub1_raw = new MockSubWeightSynchronizer(2, 4, 1024);

  std::vector<std::unique_ptr<weight_sync::WeightSynchronizerBase>> subs;
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub0_raw));
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub1_raw));

  NumaAwareWeightSynchronizer numa_ws(std::move(subs));

  numa_ws.DrainPendingH2d();
  EXPECT_EQ(sub0_raw->drain_pending_h2d_calls, 1);
  EXPECT_EQ(sub1_raw->drain_pending_h2d_calls, 1);
}

TEST(WeightSynchronizerWrapperTest, GlobalShardIndicesMapping) {
  // Test constructing WeightSynchronizer with global_shard_indices={4, 5, 6, 7}
  WeightSynchronizer ws(
      2, 4, 1024, /*local_port=*/std::nullopt,
      /*parallelism=*/1, /*listener_port=*/std::nullopt,
      /*bind_ip=*/std::nullopt, /*auto_h2d=*/false,
      /*global_shard_indices=*/std::vector<int64_t>{4, 5, 6, 7});

  EXPECT_EQ(ws.num_layers(), 2);
  EXPECT_EQ(ws.num_shards(), 4);
  EXPECT_EQ(ws.slice_byte_size(), 1024);

  auto eps = ws.get_local_endpoints();
  ASSERT_EQ(eps.size(), 1);
  EXPECT_EQ(eps[0].shards, (std::vector<int64_t>{4, 5, 6, 7}));
}

TEST(WeightSynchronizerWrapperTest, NonContiguousGlobalShardIndicesMapping) {
  // Test constructing WeightSynchronizer with non-contiguous
  // global_shard_indices={0, 1, 4, 5}
  WeightSynchronizer ws(
      2, 4, 1024, /*local_port=*/std::nullopt,
      /*parallelism=*/1, /*listener_port=*/std::nullopt,
      /*bind_ip=*/std::nullopt, /*auto_h2d=*/false,
      /*global_shard_indices=*/std::vector<int64_t>{0, 1, 4, 5});

  EXPECT_EQ(ws.num_layers(), 2);
  EXPECT_EQ(ws.num_shards(), 4);
  EXPECT_EQ(ws.slice_byte_size(), 1024);

  auto eps = ws.get_local_endpoints();
  ASSERT_EQ(eps.size(), 1);
  EXPECT_EQ(eps[0].shards, (std::vector<int64_t>{0, 1, 4, 5}));
}

TEST(WeightSynchronizerWrapperTest,
     RegisterExpectedCountsRemainderDistribution) {
  auto sub0_raw = new MockSubWeightSynchronizer(2, 4, 1024);
  auto sub1_raw = new MockSubWeightSynchronizer(2, 4, 1024);

  std::vector<std::unique_ptr<weight_sync::WeightSynchronizerBase>> subs;
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub0_raw));
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub1_raw));

  NumaAwareWeightSynchronizer numa_ws(std::move(subs));

  // 5 chunks total for layer 0 (5 is not evenly divisible by 2
  // sub-synchronizers)
  absl::flat_hash_map<size_t, uint32_t> layer_counts = {{0, 5}};
  EXPECT_TRUE(numa_ws.RegisterExpectedLayerChunks(202, layer_counts).ok());
  EXPECT_TRUE(numa_ws.RegisterExpectedChunks(202, 5).ok());

  // Remainder is distributed: sub0 receives 3, sub1 receives 2, sum is 5
  EXPECT_EQ(sub0_raw->last_registered_chunks, 3);
  EXPECT_EQ(sub0_raw->last_registered_layer_chunks[0], 3);

  EXPECT_EQ(sub1_raw->last_registered_chunks, 2);
  EXPECT_EQ(sub1_raw->last_registered_layer_chunks[0], 2);
  EXPECT_EQ(sub0_raw->last_registered_chunks + sub1_raw->last_registered_chunks,
            5);
}

TEST(WeightSynchronizerWrapperTest, StoreSkipTilingNonContiguousRouting) {
  auto sub0_raw = new MockSubWeightSynchronizer(2, 2, 1024);
  auto sub1_raw = new MockSubWeightSynchronizer(2, 2, 1024);

  std::vector<std::unique_ptr<weight_sync::WeightSynchronizerBase>> subs;
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub0_raw));
  subs.push_back(
      std::unique_ptr<weight_sync::WeightSynchronizerBase>(sub1_raw));

  NumaAwareWeightSynchronizer numa_ws(std::move(subs));
  // Non-contiguous sub-manager assignment: sub0={0, 1}, sub1={4, 5}
  numa_ws.SetSubmanagerShardsForTesting({{0, 1}, {4, 5}});

  tpu_sync::rpc::StartTransferRequest req;
  req.set_uuid(300);

  auto* sched_proto = req.mutable_shard_push_schedules();
  auto& s0 = (*sched_proto)[0];

  // Entry 1: dst_shard_idx=0 (layer 0) -> sub0
  auto* e1 = s0.add_entries();
  e1->set_dst_shard_idx(0);
  e1->set_layer_idx(0);
  e1->set_count(1);
  e1->set_size_bytes(1024);
  e1->set_src_stride_bytes(1024);
  e1->set_dst_stride_bytes(1024);

  // Entry 2: dst_shard_idx=4 (layer 0) -> sub1
  auto* e2 = s0.add_entries();
  e2->set_dst_shard_idx(4);
  e2->set_layer_idx(0);
  e2->set_count(1);
  e2->set_size_bytes(1024);
  e2->set_src_stride_bytes(1024);
  e2->set_dst_stride_bytes(1024);

  // Entry 3: dst_shard_idx=5 (layer 1) -> sub1
  auto* e3 = s0.add_entries();
  e3->set_dst_shard_idx(5);
  e3->set_layer_idx(1);
  e3->set_count(1);
  e3->set_size_bytes(1024);
  e3->set_src_stride_bytes(1024);
  e3->set_dst_stride_bytes(1024);

  // Entry 4: dst_shard_idx=2 (belongs to foreign host) -> MUST NOT route to
  // sub0/sub1
  auto* e4 = s0.add_entries();
  e4->set_dst_shard_idx(2);
  e4->set_layer_idx(0);
  e4->set_count(1);
  e4->set_size_bytes(1024);
  e4->set_src_stride_bytes(1024);
  e4->set_dst_stride_bytes(1024);

  numa_ws.StoreSkipTiling(300, req);

  absl::flat_hash_map<size_t, uint32_t> layer_counts = {{0, 2}, {1, 1}};
  EXPECT_TRUE(numa_ws.RegisterExpectedLayerChunks(300, layer_counts).ok());
  EXPECT_TRUE(numa_ws.RegisterExpectedChunks(300, 3).ok());

  // sub0 must receive 1 chunk for layer 0 only
  EXPECT_EQ(sub0_raw->last_registered_chunks, 1);
  EXPECT_EQ(sub0_raw->last_registered_layer_chunks[0], 1);
  EXPECT_EQ(sub0_raw->last_registered_layer_chunks[1], 0);

  // sub1 must receive 1 chunk for layer 0 and 1 chunk for layer 1 (total 2)
  EXPECT_EQ(sub1_raw->last_registered_chunks, 2);
  EXPECT_EQ(sub1_raw->last_registered_layer_chunks[0], 1);
  EXPECT_EQ(sub1_raw->last_registered_layer_chunks[1], 1);
}

}  // namespace
}  // namespace jax
}  // namespace tpu_raiden
