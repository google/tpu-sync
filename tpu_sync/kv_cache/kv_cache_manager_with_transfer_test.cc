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

#include "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/core/control_plane_backend.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/staging_arena.h"
#include "tpu_sync/core/tpu_pjrt_manager.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/staging_window.h"
#include "tpu_sync/kv_cache/transfer_receive_session.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/mock_metrics_backend.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::IsOk;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::ExplainMatchResult;
using ::testing::Field;
using ::testing::Gt;
using ::testing::IsEmpty;
using ::testing::Ne;
using ::testing::Not;

// Verifies that a MetricLabel has key `key` and a non-empty IP value that
// resolved to a known address (i.e. not kUnknownIp).
MATCHER_P(HasResolvedIpLabel, key,
          absl::StrCat(negation ? "does not have" : "has", " label key '", key,
                       "' with a resolved non-empty IP value")) {
  return ExplainMatchResult(
      AllOf(Field("key", &telemetry::MetricLabel::key, key),
            Field("value", &telemetry::MetricLabel::value,
                  AllOf(Not(IsEmpty()),
                        Ne(telemetry::metric_labels::kUnknownIp)))),
      arg, result_listener);
}

class TestBase : public kv_cache::KVCacheManagerBase {
 public:
  using kv_cache::KVCacheManagerBase::KVCacheManagerBase;

  void SetMockLocalIps(const std::vector<std::string>& ips) {
    mock_local_ips_ = ips;
  }

  std::vector<std::string> local_ips() const override {
    if (mock_local_ips_.has_value()) {
      return *mock_local_ips_;
    }
    return kv_cache::KVCacheManagerBase::local_ips();
  }

 private:
  std::optional<std::vector<std::string>> mock_local_ips_;
};

class TestKVCacheManagerWithTransfer : public KVCacheManagerWithTransfer {
 public:
  TestKVCacheManagerWithTransfer(
      std::vector<std::vector<raiden::RaidenBufferHandle>>
          layer_buffer_handles_in,
      std::optional<int> local_port_in,
      std::optional<int> host_blocks_to_allocate, bool unsafe_skip_buffer_lock,
      int parallelism, HostBufferAllocator host_allocator_in, int node_id,
      int local_control_port, int64_t max_blocks, int64_t num_slots,
      double timeout_s)
      : KVCacheManagerWithTransfer(
            std::make_unique<TestBase>(std::move(layer_buffer_handles_in),
                                       local_port_in, host_blocks_to_allocate,
                                       unsafe_skip_buffer_lock, parallelism,
                                       std::move(host_allocator_in)),
            node_id, local_control_port, max_blocks, num_slots, timeout_s) {}

  void SetMockLocalIps(const std::vector<std::string>& ips) {
    static_cast<TestBase*>(base())->SetMockLocalIps(ips);
  }
};

TEST(KVCacheManagerWithTransferTest, LocalOrchestratedTransfer) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());

  // Rank 3 shape: {2, 32, 32} of float.
  // 2 slices, each slice is 32*32 = 1024 elements.
  std::vector<int64_t> shape_dims = {2, 32, 32};
  int64_t elements_per_slice = 32 * 32;
  int64_t total_elements = 2 * elements_per_slice;

  // Initialize buffer with distinct values:
  // Slice 0 (Block 0): 0, 1, 2, ...
  // Slice 1 (Block 1): 1024, 1025, ...
  std::vector<float> host_data(total_elements);
  for (int i = 0; i < total_elements; ++i) {
    host_data[i] = static_cast<float>(i);
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      pjrt_manager->BufferFromHost(host_data.data(), xla::F32, shape_dims));

  ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

  struct ScopedLocalRank {
    std::optional<std::string> prev;
    ScopedLocalRank() {
      if (const char* p = std::getenv(telemetry::kLocalRankEnvVar)) {
        prev = p;
      }
      unsetenv(telemetry::kLocalRankEnvVar);
    }
    ~ScopedLocalRank() {
      if (prev.has_value()) {
        setenv(telemetry::kLocalRankEnvVar, prev->c_str(), 1);
      } else {
        unsetenv(telemetry::kLocalRankEnvVar);
      }
    }
  } scoped_rank;

  TF_ASSERT_OK_AND_ASSIGN(raiden::RaidenBufferHandle handle,
                          raiden::RaidenBufferHandle::Acquire(buffer.get()));
  std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers = {
      {std::move(handle)}};
  auto engine = std::make_unique<KVCacheManagerWithTransfer>(
      layer_buffers,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/std::nullopt,
      /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/1,
      /*host_allocator=*/nullptr,
      /*node_id=*/0,
      /*local_control_port=*/0,
      /*max_blocks=*/2,
      /*num_slots=*/2,
      /*timeout_s=*/10.0);
  const uint64_t expected_slice_bytes = engine->base()->slice_byte_size();

  const auto pcie_labels_matcher = ElementsAre(
      HasResolvedIpLabel(telemetry::metric_labels::kHostIp),
      telemetry::MetricLabel{.key = telemetry::metric_labels::kLocalRank,
                             .value = "0"});

  auto mock_backend = std::make_unique<telemetry::MockMetricsBackend>();
  telemetry::MockMetricsBackend* raw_mock = mock_backend.get();
  EXPECT_CALL(*raw_mock,
              IncrementCounter(
                  telemetry::metric_names::kSentBytesTotal,
                  ElementsAre(
                      telemetry::MetricLabel{
                          .key = telemetry::metric_labels::kDirection,
                          .value = telemetry::metric_labels::kDirectionPush},
                      HasResolvedIpLabel(telemetry::metric_labels::kSrcIp),
                      HasResolvedIpLabel(telemetry::metric_labels::kDstIp)),
                  Gt(0)))
      .Times(1);
  EXPECT_CALL(
      *raw_mock,
      IncrementCounter(telemetry::metric_names::kReceivedBytesTotal,
                       ElementsAre(telemetry::MetricLabel{
                           .key = telemetry::metric_labels::kDirection,
                           .value = telemetry::metric_labels::kDirectionPush}),
                       Gt(0)))
      .Times(1);
  EXPECT_CALL(*raw_mock,
              ObserveHistogram(telemetry::metric_names::kTransferDurationMs,
                               IsEmpty(), Gt(0.0)))
      .Times(1);
  EXPECT_CALL(
      *raw_mock,
      ObserveHistogram(
          telemetry::metric_names::kP2pTransferTimeMs,
          ElementsAre(
              telemetry::MetricLabel{.key = telemetry::metric_labels::kSrcIp,
                                     .value = "127.0.0.1"},
              HasResolvedIpLabel(telemetry::metric_labels::kDstIp),
              telemetry::MetricLabel{
                  .key = telemetry::metric_labels::kLocalRank, .value = "0"}),
          Gt(0.0)))
      .Times(1);
  EXPECT_CALL(*raw_mock,
              ObserveHistogram(telemetry::metric_names::kD2hTransferTimeMs,
                               IsEmpty(), Gt(0.0)))
      .Times(1);
  EXPECT_CALL(*raw_mock,
              ObserveHistogram(telemetry::metric_names::kD2hDmaTimeMs,
                               pcie_labels_matcher, testing::Ge(0.0)))
      .Times(1);
  EXPECT_CALL(*raw_mock,
              IncrementCounter(telemetry::metric_names::kD2hBytesTotal,
                               pcie_labels_matcher, expected_slice_bytes))
      .Times(1);
  EXPECT_CALL(*raw_mock,
              ObserveHistogram(telemetry::metric_names::kH2dTransferTimeMs,
                               IsEmpty(), Gt(0.0)))
      .Times(1);
  EXPECT_CALL(*raw_mock,
              ObserveHistogram(telemetry::metric_names::kH2dDmaTimeMs,
                               pcie_labels_matcher, testing::Ge(0.0)))
      .Times(1);
  EXPECT_CALL(*raw_mock,
              IncrementCounter(telemetry::metric_names::kH2dBytesTotal,
                               pcie_labels_matcher, expected_slice_bytes))
      .Times(1);
  // Register mock backend
  telemetry::ScopedMetricsBackendReset scoped_metrics_reset(
      std::move(mock_backend));

  // Configure staging slots: 2 slots, max 2 blocks per slot
  ASSERT_THAT(engine->base()->ConfigureHostStagingSlots(2, 2), IsOk());

  // Register Block 0 as ready for read on ourselves (producer)
  uint64_t uuid = 12345;
  std::string req_id = "local_test_req";
  engine->NotifyForRead(req_id, uuid, /*block_ids=*/{0});

  // Get our own control port
  int port = engine->local_control_port();
  ASSERT_GT(port, 0);
  std::string remote_endpoint = "127.0.0.1:" + std::to_string(port);

  // Start read: pull Block 0 (remote) and store it in Block 1 (local)
  engine->StartRead(req_id, uuid, remote_endpoint,
                    /*remote_block_ids=*/{0},
                    /*local_block_ids=*/{1});

  // Wait for transfer to complete by polling CompleteReadRaw
  bool done = false;
  for (int i = 0; i < 100; ++i) {
    auto [done_sending, done_recving, failed_recving] =
        engine->CompleteReadRaw();
    if (std::find(failed_recving.begin(), failed_recving.end(), req_id) !=
        failed_recving.end()) {
      FAIL() << "Transfer failed";
    }
    if (std::find(done_recving.begin(), done_recving.end(), req_id) !=
        done_recving.end()) {
      done = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  ASSERT_TRUE(done) << "Transfer timed out";

  // Verify device buffer: Block 1 should now contain Block 0's original data.
  // i.e. elements 1024..2047 should be equal to 0..1023.
  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
  auto read_back = literal->data<float>();
  ASSERT_EQ(read_back.size(), total_elements);

  // Block 0 (unchanged)
  for (int i = 0; i < elements_per_slice; ++i) {
    EXPECT_EQ(read_back[i], static_cast<float>(i));
  }
  // Block 1 (overwritten with Block 0's data)
  for (int i = 0; i < elements_per_slice; ++i) {
    EXPECT_EQ(read_back[elements_per_slice + i], static_cast<float>(i));
  }
}

TEST(KVCacheManagerWithTransferTest, StartReadAcceptsParallelism) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());
  std::vector<int64_t> shape_dims = {2, 32, 32};
  std::vector<float> host_data(2 * 32 * 32, 0.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      pjrt_manager->BufferFromHost(host_data.data(), xla::F32, shape_dims));
  ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

  auto handle_or = raiden::RaidenBufferHandle::Acquire(buffer.get());
  std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers = {
      {handle_or.value()}};
  auto engine = std::make_unique<KVCacheManagerWithTransfer>(
      layer_buffers,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/std::nullopt,
      /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/1,
      /*host_allocator=*/nullptr,
      /*node_id=*/0,
      /*local_control_port=*/0,
      /*max_blocks=*/2,
      /*num_slots=*/2,
      /*timeout_s=*/10.0);
  ASSERT_THAT(engine->base()->ConfigureHostStagingSlots(2, 2), IsOk());

  // Calling StartRead with a non-existent port throws or returns an op that
  // fails
  engine->StartRead("req_parallel", 99999, "127.0.0.1:8888", {0}, {0},
                    /*parallelism=*/2);
}
TEST(KVCacheManagerWithTransferTest,
     LocalOrchestratedTransferToCustomHostBlock) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());

  // Rank 3 shape: {2, 32, 32} of float.
  // 2 slices, each slice is 32*32 = 1024 elements.
  std::vector<int64_t> shape_dims = {2, 32, 32};
  int64_t elements_per_slice = 32 * 32;
  int64_t total_elements = 2 * elements_per_slice;

  // Initialize buffer with distinct values:
  // Slice 0 (Block 0): 0, 1, 2, ...
  // Slice 1 (Block 1): 1024, 1025, ...
  std::vector<float> host_data(total_elements);
  for (int i = 0; i < total_elements; ++i) {
    host_data[i] = static_cast<float>(i);
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      pjrt_manager->BufferFromHost(host_data.data(), xla::F32, shape_dims));

  ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

  // Create KVCacheManagerWithTransfer
  auto handle_or = raiden::RaidenBufferHandle::Acquire(buffer.get());
  std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers = {
      {handle_or.value()}};
  auto engine = std::make_unique<KVCacheManagerWithTransfer>(
      layer_buffers,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/6,  // Allocate 6 blocks in host
      /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/1,
      /*host_allocator=*/nullptr,
      /*node_id=*/0,
      /*local_control_port=*/0,
      /*max_blocks=*/2,
      /*num_slots=*/2,
      /*timeout_s=*/10.0);

  // Configure staging slots: 2 slots, max 2 blocks per slot
  ASSERT_THAT(engine->base()->ConfigureHostStagingSlots(2, 2), IsOk());

  // Register Block 0 as ready for read on ourselves (producer)
  uint64_t uuid = 12346;
  std::string req_id = "local_test_custom_host_req";
  engine->NotifyForRead(req_id, uuid, /*block_ids=*/{0});

  // Get our own control port
  int port = engine->local_control_port();
  ASSERT_GT(port, 0);
  std::string remote_endpoint = "127.0.0.1:" + std::to_string(port);

  // Start read: pull Remote Block 0 -> Local Device Block 1,
  // but target Local Host Block 0 as the staging/host block.
  engine->StartRead(req_id, uuid, remote_endpoint,
                    /*remote_block_ids=*/{0},
                    /*local_block_ids=*/{1},
                    /*parallelism=*/1,
                    /*local_host_block_ids=*/std::vector<int64_t>{4});

  // Wait for transfer to complete
  bool done = false;
  for (int i = 0; i < 100; ++i) {
    auto [done_sending, done_recving, failed_recving] =
        engine->CompleteReadRaw();
    if (std::find(failed_recving.begin(), failed_recving.end(), req_id) !=
        failed_recving.end()) {
      FAIL() << "Transfer failed";
    }
    if (std::find(done_recving.begin(), done_recving.end(), req_id) !=
        done_recving.end()) {
      done = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  ASSERT_TRUE(done) << "Transfer timed out";

  // Verify device buffer: Block 1 should now contain Block 0's original data.
  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
  auto read_back = literal->data<float>();
  ASSERT_EQ(read_back.size(), total_elements);

  // Block 1 (overwritten with Block 0's data: 0..1023)
  for (int i = 0; i < elements_per_slice; ++i) {
    EXPECT_EQ(read_back[elements_per_slice + i], static_cast<float>(i));
  }

  // Verify host buffer:
  uint8_t* host_block_4 = engine->base()->GetBlockHostPointer(0, 0, 4);
  uint8_t* host_block_5 = engine->base()->GetBlockHostPointer(0, 0, 5);

  float* host_block_4_float = reinterpret_cast<float*>(host_block_4);
  float* host_block_5_float = reinterpret_cast<float*>(host_block_5);

  // Host Block 4 should contain the transferred data in tiled layout.
  // Tile width is 128 floats (512 bytes) due to TPU alignment.
  const int tile_width = 128;
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      int physical_idx = r * tile_width + c;
      int logical_val = r * 32 + c;
      EXPECT_EQ(host_block_4_float[physical_idx],
                static_cast<float>(logical_val));
    }
  }

  // Host Block 5 should still be 0 (untouched, since we targeted Host Block 4)
  for (int i = 0; i < elements_per_slice; ++i) {
    EXPECT_EQ(host_block_5_float[i], 0.0f);
  }
}

TEST(KVCacheManagerWithTransferTest, TreeBroadcastCorrectness8Nodes) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());

  // Rank 3 shape: {2, 32, 32} of float.
  std::vector<int64_t> shape_dims = {2, 32, 32};
  int64_t elements_per_slice = 32 * 32;
  int64_t total_elements = 2 * elements_per_slice;

  // Initialize source data with distinct values
  std::vector<float> src_host_data(total_elements);
  for (int i = 0; i < total_elements; ++i) {
    src_host_data[i] = static_cast<float>(i + 1);
  }

  // Create 8 separate PjRtBuffers representing the TPU buffers for the 8 nodes
  std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers(8);
  for (int i = 0; i < 8; ++i) {
    if (i == 0) {
      TF_ASSERT_OK_AND_ASSIGN(
          buffers[i], pjrt_manager->BufferFromHost(src_host_data.data(),
                                                   xla::F32, shape_dims));
    } else {
      // Destination buffers initialized to zeros
      std::vector<float> zeros(total_elements, 0.0f);
      TF_ASSERT_OK_AND_ASSIGN(
          buffers[i],
          pjrt_manager->BufferFromHost(zeros.data(), xla::F32, shape_dims));
    }
    ASSERT_THAT(buffers[i]->GetReadyFuture().Await(), IsOk());
  }

  // Create 8 KVCacheManagerWithTransfer engines
  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> engines(8);
  std::vector<std::string> endpoints(8);

  for (int i = 0; i < 8; ++i) {
    auto handle_or = raiden::RaidenBufferHandle::Acquire(buffers[i].get());
    ASSERT_OK(handle_or.status());
    std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers = {
        {handle_or.value()}};
    engines[i] = std::make_unique<KVCacheManagerWithTransfer>(
        layer_buffers,
        /*local_port=*/std::nullopt,
        /*host_blocks_to_allocate=*/std::nullopt,
        /*unsafe_skip_buffer_lock=*/true,
        /*parallelism=*/1,
        /*host_allocator=*/nullptr,
        /*node_id=*/i,
        /*local_control_port=*/0,
        /*max_blocks=*/2,
        /*num_slots=*/2,
        /*timeout_s=*/10.0);

    ASSERT_THAT(engines[i]->base()->ConfigureHostStagingSlots(2, 2), IsOk());
    int port = engines[i]->local_control_port();
    ASSERT_GT(port, 0);
    endpoints[i] = "127.0.0.1:" + std::to_string(port);
  }

  uint64_t uuid_1 = 88881;
  uint64_t uuid_2 = 88882;
  uint64_t uuid_3 = 88883;
  uint64_t uuid_4 = 88884;
  uint64_t uuid_5 = 88885;
  uint64_t uuid_6 = 88886;
  uint64_t uuid_7 = 88887;
  std::string req_id_prefix = "tree_broadcast_req_";

  // Phase 1: Node 0 (Source) notifies read. Nodes 1 and 2 start read pulling
  // from Node 0.
  engines[0]->NotifyForRead(absl::StrCat(req_id_prefix, "1"), uuid_1, {0});
  engines[0]->NotifyForRead(absl::StrCat(req_id_prefix, "2"), uuid_2, {0});

  engines[1]->StartRead(absl::StrCat(req_id_prefix, "1"), uuid_1, endpoints[0],
                        {0}, {1});
  engines[2]->StartRead(absl::StrCat(req_id_prefix, "2"), uuid_2, endpoints[0],
                        {0}, {1});

  // Await completion of Phase 1
  auto await_completion = [](KVCacheManagerWithTransfer* engine,
                             const std::string& req_id) {
    bool done = false;
    for (int i = 0; i < 200; ++i) {
      auto [done_sending, done_recving, failed_recving] =
          engine->CompleteReadRaw();
      if (std::find(failed_recving.begin(), failed_recving.end(), req_id) !=
          failed_recving.end()) {
        return false;
      }
      if (std::find(done_recving.begin(), done_recving.end(), req_id) !=
          done_recving.end()) {
        done = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return done;
  };

  ASSERT_TRUE(
      await_completion(engines[1].get(), absl::StrCat(req_id_prefix, "1")));
  ASSERT_TRUE(
      await_completion(engines[2].get(), absl::StrCat(req_id_prefix, "2")));

  // Phase 2: Nodes 0, 1, 2 notify read.
  // Nodes 3, 4 pull from Node 1.
  // Nodes 5, 6 pull from Node 2.
  engines[1]->NotifyForRead(absl::StrCat(req_id_prefix, "3"), uuid_3, {1});
  engines[1]->NotifyForRead(absl::StrCat(req_id_prefix, "4"), uuid_4, {1});
  engines[2]->NotifyForRead(absl::StrCat(req_id_prefix, "5"), uuid_5, {1});
  engines[2]->NotifyForRead(absl::StrCat(req_id_prefix, "6"), uuid_6, {1});

  engines[3]->StartRead(absl::StrCat(req_id_prefix, "3"), uuid_3, endpoints[1],
                        {1}, {1});
  engines[4]->StartRead(absl::StrCat(req_id_prefix, "4"), uuid_4, endpoints[1],
                        {1}, {1});
  engines[5]->StartRead(absl::StrCat(req_id_prefix, "5"), uuid_5, endpoints[2],
                        {1}, {1});
  engines[6]->StartRead(absl::StrCat(req_id_prefix, "6"), uuid_6, endpoints[2],
                        {1}, {1});

  // Await completion of Phase 2
  ASSERT_TRUE(
      await_completion(engines[3].get(), absl::StrCat(req_id_prefix, "3")));
  ASSERT_TRUE(
      await_completion(engines[4].get(), absl::StrCat(req_id_prefix, "4")));
  ASSERT_TRUE(
      await_completion(engines[5].get(), absl::StrCat(req_id_prefix, "5")));
  ASSERT_TRUE(
      await_completion(engines[6].get(), absl::StrCat(req_id_prefix, "6")));

  // Phase 3: Node 3 notifies read. Node 7 pulls from Node 3.
  engines[3]->NotifyForRead(absl::StrCat(req_id_prefix, "7"), uuid_7, {1});
  engines[7]->StartRead(absl::StrCat(req_id_prefix, "7"), uuid_7, endpoints[3],
                        {1}, {1});

  // Await completion of Phase 3
  ASSERT_TRUE(
      await_completion(engines[7].get(), absl::StrCat(req_id_prefix, "7")));

  // Verify all 7 destinations got the data correctly in their device buffers
  // (Block 1)
  for (int i = 1; i < 8; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(auto literal, buffers[i]->ToLiteral().Await());
    auto read_back = literal->data<float>();
    ASSERT_EQ(read_back.size(), total_elements);

    // Block 1 should be overwritten with Block 0's data (values 1..1024)
    for (int j = 0; j < elements_per_slice; ++j) {
      EXPECT_EQ(read_back[elements_per_slice + j], src_host_data[j])
          << "Node " << i << " mismatch at index " << j;
    }
  }
}

TEST(KVCacheManagerWithTransferTest, MultiIpOrchestratedTransfer) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());

  std::vector<int64_t> shape_dims = {2, 32, 32};
  int64_t elements_per_slice = 32 * 32;
  int64_t total_elements = 2 * elements_per_slice;

  std::vector<float> host_data(total_elements);
  for (int i = 0; i < total_elements; ++i) {
    host_data[i] = static_cast<float>(i);
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      pjrt_manager->BufferFromHost(host_data.data(), xla::F32, shape_dims));

  ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

  auto handle_or = raiden::RaidenBufferHandle::Acquire(buffer.get());
  std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers = {
      {handle_or.value()}};

  auto engine = std::make_unique<TestKVCacheManagerWithTransfer>(
      layer_buffers,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/std::nullopt,
      /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/2,
      /*host_allocator=*/nullptr,
      /*node_id=*/0,
      /*local_control_port=*/0,
      /*max_blocks=*/2,
      /*num_slots=*/2,
      /*timeout_s=*/10.0);

  engine->SetMockLocalIps({"127.0.0.1", "127.0.0.2"});

  ASSERT_THAT(engine->base()->ConfigureHostStagingSlots(2, 2), IsOk());

  uint64_t uuid = 12347;
  std::string req_id = "multi_ip_test_req";
  engine->NotifyForRead(req_id, uuid, /*block_ids=*/{0});

  int port = engine->local_control_port();
  ASSERT_GT(port, 0);
  std::string remote_endpoint = "127.0.0.1:" + std::to_string(port);

  engine->StartRead(req_id, uuid, remote_endpoint,
                    /*remote_block_ids=*/{0},
                    /*local_block_ids=*/{1},
                    /*parallelism=*/2);

  bool done = false;
  for (int i = 0; i < 100; ++i) {
    auto [done_sending, done_recving, failed_recving] =
        engine->CompleteReadRaw();
    if (std::find(failed_recving.begin(), failed_recving.end(), req_id) !=
        failed_recving.end()) {
      FAIL() << "Transfer failed";
    }
    if (std::find(done_recving.begin(), done_recving.end(), req_id) !=
        done_recving.end()) {
      done = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  ASSERT_TRUE(done) << "Transfer timed out";

  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
  auto read_back = literal->data<float>();
  ASSERT_EQ(read_back.size(), total_elements);

  for (int i = 0; i < elements_per_slice; ++i) {
    EXPECT_EQ(read_back[i], static_cast<float>(i));
  }
  for (int i = 0; i < elements_per_slice; ++i) {
    EXPECT_EQ(read_back[elements_per_slice + i], static_cast<float>(i));
  }
}

TEST(KVCacheManagerWithTransferTest, LocalDataEndpointsCarryLayerHostAddrs) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());

  std::vector<int64_t> shape_dims = {2, 32, 32};
  int64_t elements_per_slice = 32 * 32;
  int64_t total_elements = 2 * elements_per_slice;
  std::vector<float> host_data(total_elements, 1.0f);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      pjrt_manager->BufferFromHost(host_data.data(), xla::F32, shape_dims));
  ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

  TF_ASSERT_OK_AND_ASSIGN(raiden::RaidenBufferHandle handle,
                          raiden::RaidenBufferHandle::Acquire(buffer.get()));
  std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers = {
      {std::move(handle)}};

  auto engine = std::make_unique<KVCacheManagerWithTransfer>(
      layer_buffers,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/4,
      /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/1,
      /*host_allocator=*/nullptr,
      /*node_id=*/0,
      /*local_control_port=*/0,
      /*max_blocks=*/2,
      /*num_slots=*/2,
      /*timeout_s=*/10.0);

  auto data_eps = engine->get_local_data_endpoints();
  ASSERT_FALSE(data_eps.empty());
  for (const auto& ep : data_eps) {
    ASSERT_EQ(ep.layer_host_addrs.size(), 1u);
    EXPECT_EQ(ep.layer_host_addrs[0].host_base_addrs_size(), 1);
    EXPECT_GT(ep.layer_host_addrs[0].host_base_addrs(0), 0u);
    EXPECT_EQ(ep.layer_host_addrs[0].block_stride_bytes(),
              engine->base()->slice_byte_size());
    EXPECT_EQ(ep.layer_host_addrs[0].num_blocks(), 4u);
  }
}

class KVCacheManagerWithTransferTestWindowAccessor
    : public KVCacheManagerWithTransfer {
 public:
  using KVCacheManagerWithTransfer::base_;
  using KVCacheManagerWithTransfer::control_handler_;
  using KVCacheManagerWithTransfer::KVCacheManagerWithTransfer;
  using KVCacheManagerWithTransfer::staging_allocator_;

  KVCacheManagerWithTransferTestWindowAccessor(
      std::unique_ptr<kv_cache::KVCacheManagerBase> base,
      std::optional<StagingWindowConfig> staging_window_config)
      : KVCacheManagerWithTransfer(
            std::move(base), /*node_id=*/0, /*local_control_port=*/0,
            /*max_blocks=*/4, /*num_slots=*/2, /*timeout_s=*/10.0,
            /*metrics_collector=*/nullptr, staging_window_config) {}
};

std::unique_ptr<kv_cache::KVCacheManagerBase> MakeTestBase(
    int host_blocks = 16) {
  return std::make_unique<kv_cache::KVCacheManagerBase>(
      /*num_layers=*/1, /*num_shards=*/1,
      /*slice_byte_sizes=*/std::vector<size_t>{128},
      /*local_port=*/std::nullopt, host_blocks,
      /*parallelism=*/1, /*host_allocator=*/nullptr);
}

KVCacheManagerWithTransferTestWindowAccessor MakeTestAccessor(
    int64_t max_blocks = 8, int64_t num_slots = 2,
    std::optional<StagingWindowConfig> cfg = std::nullopt) {
  return KVCacheManagerWithTransferTestWindowAccessor(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/std::nullopt, /*host_blocks_to_allocate=*/16,
      /*parallelism=*/1, /*node_id=*/0, /*local_control_port=*/0, max_blocks,
      num_slots, /*timeout_s=*/10.0, /*metrics_collector=*/nullptr, cfg);
}

TEST(KVCacheManagerWithTransferTest, StagingWindowConstructorValidation) {
  EXPECT_THROW(MakeTestAccessor(0, 2), std::invalid_argument);
  EXPECT_THROW(MakeTestAccessor(4, 0), std::invalid_argument);
  EXPECT_THROW(
      MakeTestAccessor(
          8, 2, StagingWindowConfig{.window_chunks = 8, .chunk_blocks = 4}),
      std::invalid_argument);
}

TEST(KVCacheManagerWithTransferTest, StagingWindowDynamicModeForced) {
  struct ScopedEnv {
    const char* v;
    std::optional<std::string> prev;
    explicit ScopedEnv(const char* key, const char* val) : v(key) {
      if (const char* p = std::getenv(v)) prev = p;
      if (val)
        setenv(v, val, 1);
      else
        unsetenv(v);
    }
    ~ScopedEnv() {
      if (prev)
        setenv(v, prev->c_str(), 1);
      else
        unsetenv(v);
    }
  };

  {
    ScopedEnv env("TPU_RAIDEN_DYNAMIC_HOST_STAGING", nullptr);
    EXPECT_FALSE(MakeTestAccessor().staging_allocator_->dynamic_host_staging());
  }
  {
    ScopedEnv env("TPU_RAIDEN_DYNAMIC_HOST_STAGING", "1");
    EXPECT_TRUE(MakeTestAccessor().staging_allocator_->dynamic_host_staging());
  }
  {
    ScopedEnv env("TPU_RAIDEN_DYNAMIC_HOST_STAGING", "0");
    EXPECT_TRUE(
        MakeTestAccessor(
            8, 2, StagingWindowConfig{.window_chunks = 4, .chunk_blocks = 2})
            .staging_allocator_->dynamic_host_staging());
  }
}

TEST(KVCacheManagerWithTransferTest, StagingWindowCoTenancyWarning) {
  auto base = MakeTestBase(16);
  ASSERT_TRUE(base->host_block_manager()->Allocate(1, /*lock=*/true).ok());
  KVCacheManagerWithTransferTestWindowAccessor manager(
      std::move(base),
      StagingWindowConfig{.window_chunks = 2, .chunk_blocks = 2});
  ASSERT_NE(manager.staging_allocator_, nullptr);
}

TEST(KVCacheManagerWithTransferTest, SetStagingWindowConfigRules) {
  StagingWindowConfig init_cfg{
      .window_chunks = 4, .serve_windowed_pulls = true, .chunk_blocks = 2};
  auto manager = MakeTestAccessor(8, 2, init_cfg);

  StagingWindowConfig valid_update = init_cfg;
  valid_update.chunk_blocks = 3;
  EXPECT_TRUE(manager.SetStagingWindowConfig(valid_update).ok());
  EXPECT_EQ(manager.staging_window_config().chunk_blocks, 3);

  StagingWindowConfig bad = valid_update;
  bad.window_chunks = 2;
  EXPECT_FALSE(manager.SetStagingWindowConfig(bad).ok());
  bad = valid_update;
  bad.serve_windowed_pulls = false;
  EXPECT_FALSE(manager.SetStagingWindowConfig(bad).ok());
  bad = valid_update;
  bad.chunk_blocks = 5;
  EXPECT_FALSE(manager.SetStagingWindowConfig(bad).ok());

  // Auto-C re-validation on chunk_bytes_target changes:
  StagingWindowConfig auto_c = init_cfg;
  auto_c.chunk_blocks = 0;
  auto_c.chunk_bytes_target = 8 << 20;
  EXPECT_FALSE(manager.SetStagingWindowConfig(auto_c).ok());
  auto_c.chunk_bytes_target = 256;
  EXPECT_TRUE(manager.SetStagingWindowConfig(auto_c).ok());
  EXPECT_EQ(manager.staging_window_config().chunk_blocks, 0);
  EXPECT_EQ(manager.staging_window_config().chunk_bytes_target, 256);

  manager.NotifyForRead("reg1", /*uuid=*/101, {0});
  EXPECT_EQ(manager.SetStagingWindowConfig(valid_update).code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(KVCacheManagerWithTransferTest,
     TransferReceiveSessionChunkGettersAndReset) {
  auto base = MakeTestBase(16);
  auto allocator = StagingBlockAllocator::Create(
      base->host_block_manager(), /*num_slots=*/2, /*max_blocks=*/4);
  ASSERT_TRUE(allocator.ok());

  StagingWindowConfig window_cfg{.window_chunks = 4, .chunk_blocks = 2};
  auto session_or = TransferReceiveSession::Create(
      base.get(), allocator->get(), /*uuid=*/555, "req_getters",
      /*requested_remote_block_ids=*/{0, 1, 2, 3},
      /*requested_local_block_ids=*/{4, 5, 6, 7},
      /*requested_host_block_ids=*/std::nullopt,
      /*deadline=*/std::chrono::steady_clock::now() + std::chrono::seconds(10),
      window_cfg);
  ASSERT_TRUE(session_or.ok());
  auto session = *session_or;

  EXPECT_EQ(session->chunk_blocks(), 2);
  EXPECT_EQ(session->window_chunks(), 4);
  EXPECT_FALSE(session->SetChunkH2dSpecs(1, {}, -1, 4).ok());
  EXPECT_FALSE(session->SetChunkH2dSpecs(1, {}, 2, -1).ok());
  EXPECT_TRUE(session->SetChunkH2dSpecs(1, {}, 0, 0).ok());
  EXPECT_EQ(session->chunk_blocks(), 0);
  EXPECT_EQ(session->window_chunks(), 0);
}

TEST(KVCacheManagerWithTransferTest, WindowedLoopbackTransferW2C2K7) {
  for (int32_t w : {2, 4, 8}) {
    TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                            TpuPjrtManager::GetDefault());

    constexpr int64_t k = 7;
    constexpr int64_t total_blocks = 2 * k;
    constexpr int64_t elements_per_slice = 32 * 32;
    const int64_t total_elements = total_blocks * elements_per_slice;

    std::vector<float> host_data(total_elements, 0.0f);
    std::iota(host_data.begin(), host_data.begin() + k * elements_per_slice,
              1.0f);

    std::vector<int64_t> shape_dims = {total_blocks, 32, 32};
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<xla::PjRtBuffer> buffer,
        pjrt_manager->BufferFromHost(host_data.data(), xla::F32, shape_dims));
    ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

    TF_ASSERT_OK_AND_ASSIGN(raiden::RaidenBufferHandle handle,
                            raiden::RaidenBufferHandle::Acquire(buffer.get()));
    std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers = {
        {std::move(handle)}};

    StagingWindowConfig window_cfg{
        .window_chunks = w, .serve_windowed_pulls = true, .chunk_blocks = 2};

    auto engine = std::make_unique<KVCacheManagerWithTransfer>(
        layer_buffers, /*local_port=*/std::nullopt,
        /*host_blocks_to_allocate=*/32, /*unsafe_skip_buffer_lock=*/true,
        /*parallelism=*/1, /*host_allocator=*/nullptr, /*node_id=*/0,
        /*local_control_port=*/0, /*max_blocks=*/16, /*num_slots=*/2,
        /*timeout_s=*/10.0, /*metrics_collector=*/nullptr, window_cfg);

    std::vector<int64_t> src_blocks = {0, 1, 2, 3, 4, 5, 6};
    std::vector<int64_t> dst_blocks = {7, 8, 9, 10, 11, 12, 13};
    const uint64_t uuid = 20000 + w;
    const std::string req_id = absl::StrCat("window_loopback_w", w);

    engine->NotifyForRead(req_id, uuid, src_blocks);
    std::string remote_endpoint =
        absl::StrCat("127.0.0.1:", engine->local_control_port());
    engine->StartRead(req_id, uuid, remote_endpoint, src_blocks, dst_blocks);

    bool done = false;
    for (int i = 0; i < 100; ++i) {
      auto [done_sending, done_recving, failed_recving] =
          engine->CompleteReadRaw();
      if (!failed_recving.empty()) FAIL() << "Transfer failed for W=" << w;
      if (!done_recving.empty()) {
        done = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ASSERT_TRUE(done) << "Transfer timed out for W=" << w;

    TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
    auto read_back = literal->data<float>();
    ASSERT_EQ(read_back.size(), total_elements);

    int mismatches = 0;
    for (int64_t b = 0; b < k; ++b) {
      int64_t src_offset = b * elements_per_slice;
      int64_t dst_offset = (k + b) * elements_per_slice;
      for (int64_t i = 0; i < elements_per_slice; ++i) {
        if (read_back[dst_offset + i] != read_back[src_offset + i]) {
          ++mismatches;
        }
      }
    }
    EXPECT_EQ(mismatches, 0) << "Data mismatch for W=" << w;
  }
}

void RunLoopbackHbmTest(bool assume_peer_support, uint64_t uuid,
                        absl::string_view req_id, bool echo_mismatch = false) {
  TF_ASSERT_OK_AND_ASSIGN(auto* pjrt_mgr, TpuPjrtManager::GetDefault());
  constexpr int64_t k = 7, total_blocks = 14, elements_per_slice = 32 * 32;
  const int64_t total_elements = total_blocks * elements_per_slice;

  std::vector<float> host_data(total_elements, 0.0f);
  std::iota(host_data.begin(), host_data.begin() + k * elements_per_slice,
            1.0f);

  TF_ASSERT_OK_AND_ASSIGN(auto buffer,
                          pjrt_mgr->BufferFromHost(host_data.data(), xla::F32,
                                                   {total_blocks, 32, 32}));
  ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

  TF_ASSERT_OK_AND_ASSIGN(auto handle,
                          raiden::RaidenBufferHandle::Acquire(buffer.get()));
  std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers{
      {std::move(handle)}};

  StagingWindowConfig window_cfg{.window_chunks = 2,
                                 .serve_windowed_pulls = echo_mismatch,
                                 .chunk_blocks = 2,
                                 .assume_peer_support = assume_peer_support};

  auto engine = std::make_unique<KVCacheManagerWithTransferTestWindowAccessor>(
      layer_buffers, std::nullopt, 32, /*unsafe_skip_buffer_lock=*/true, 1,
      nullptr, /*node_id=*/0, /*local_control_port=*/0, /*max_blocks=*/16,
      /*num_slots=*/2, /*timeout_s=*/10.0, nullptr, window_cfg);

  std::unique_ptr<ControlPlaneBackend> interceptor;
  std::unique_ptr<ControlPlaneHandler> mismatch_handler;
  std::string remote_endpoint;
  if (echo_mismatch) {
    class MismatchHandler : public ControlPlaneHandler {
     public:
      explicit MismatchHandler(ControlPlaneHandler* d) : delegate_(d) {}
      absl::StatusOr<PullStreamResponseSpec> OnPullStream(
          const PullStreamRequestSpec& req, absl::string_view p) override {
        PullStreamRequestSpec legacy_req = req;
        legacy_req.chunk_blocks = legacy_req.window_chunks = 0;
        return delegate_->OnPullStream(legacy_req, p);
      }
      absl::Status OnAck(uint64_t u) override { return delegate_->OnAck(u); }
      ControlPlaneCapabilities GetCapabilities() const override {
        return delegate_->GetCapabilities();
      }
      uint64_t MaxPullStreamBlocks() const override {
        return delegate_->MaxPullStreamBlocks();
      }

     private:
      ControlPlaneHandler* delegate_;
    };

    mismatch_handler =
        std::make_unique<MismatchHandler>(engine->control_handler_.get());
    interceptor = CreateControlPlaneBackend(
        ResolveControlPlaneBackendType(),
        [](auto t) { std::thread(std::move(t)).detach(); });
    auto intercept_port = interceptor->StartServer(0, mismatch_handler.get());
    ASSERT_TRUE(intercept_port.ok()) << intercept_port.status();
    remote_endpoint = absl::StrCat("127.0.0.1:", *intercept_port);
  } else {
    remote_endpoint = absl::StrCat("127.0.0.1:", engine->local_control_port());
  }

  std::vector<int64_t> src = {0, 1, 2, 3, 4, 5, 6},
                       dst = {7, 8, 9, 10, 11, 12, 13};
  engine->NotifyForRead(std::string(req_id), uuid, src);
  engine->StartRead(std::string(req_id), uuid, remote_endpoint, src, dst);

  if (!echo_mismatch) {
    bool done = false;
    for (int i = 0; i < 100 && !done; ++i) {
      auto [sent, recvd, failed] = engine->CompleteReadRaw();
      if (!failed.empty()) FAIL() << "Transfer failed";
      if (!recvd.empty()) done = true;
      if (!done) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ASSERT_TRUE(done) << "Transfer timed out";
  } else {
    CompleteReadResult details;
    for (int i = 0; i < 100 && details.failed.empty(); ++i) {
      details = engine->CompleteReadWithDetails();
      if (details.failed.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }
    ASSERT_THAT(details.failed, testing::ElementsAre(std::string(req_id)));
    const auto& d = details.details[std::string(req_id)];
    EXPECT_EQ(d.phase.value_or(""), "connect");
    EXPECT_THAT(d.reason.value_or(""),
                testing::HasSubstr("peer ignored staging window"));
    auto* hbm = engine->base_->host_block_manager();
    for (int i = 0; i < 100 && hbm->num_free_blocks() < hbm->total_blocks();
         ++i) {
      (void)engine->CompleteReadWithDetails();
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    EXPECT_EQ(hbm->num_free_blocks(), hbm->total_blocks());
    interceptor->StopServer();
  }

  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
  auto read_back = literal->data<float>();
  ASSERT_EQ(read_back.size(), total_elements);

  int mismatches = 0;
  for (int64_t b = 0; b < k; ++b) {
    int64_t src = b * elements_per_slice, dst = (k + b) * elements_per_slice;
    for (int64_t i = 0; i < elements_per_slice; ++i) {
      if (read_back[dst + i] != (!echo_mismatch ? read_back[src + i] : 0.0f)) {
        ++mismatches;
      }
    }
  }
  EXPECT_EQ(mismatches, 0);
}

TEST(KVCacheManagerWithTransferTest,
     WindowedLoopbackDowngradeToExactKHbmCheck) {
  RunLoopbackHbmTest(/*assume_peer_support=*/false, 21001,
                     "window_loopback_downgrade_hbm");
}

TEST(KVCacheManagerWithTransferTest, WindowedLoopbackMarkerRetryHbmCheck) {
  RunLoopbackHbmTest(/*assume_peer_support=*/true, 21002,
                     "window_loopback_marker_retry_hbm");
}

TEST(KVCacheManagerWithTransferTest,
     WindowedLoopbackEchoMismatchFailsFastAndPreservesHbm) {
  RunLoopbackHbmTest(/*assume_peer_support=*/true, 21003, "mismatch_echo_req",
                     /*echo_mismatch=*/true);
}

void RunLoopbackDataCheckTest(
    int64_t k, int32_t chunk_blocks, int32_t window_chunks, uint64_t uuid,
    absl::string_view req_id,
    const std::optional<std::vector<int64_t>>& local_host_bids = std::nullopt,
    const std::vector<int64_t>& override_dst = {}) {
  TF_ASSERT_OK_AND_ASSIGN(auto* pjrt_mgr, TpuPjrtManager::GetDefault());
  const int64_t total_blocks = 2 * k;
  constexpr int64_t elements_per_slice = 32 * 32;
  const int64_t total_elements = total_blocks * elements_per_slice;

  std::vector<float> host_data(total_elements, 0.0f);
  std::iota(host_data.begin(), host_data.begin() + k * elements_per_slice,
            1.0f);

  TF_ASSERT_OK_AND_ASSIGN(auto buffer,
                          pjrt_mgr->BufferFromHost(host_data.data(), xla::F32,
                                                   {total_blocks, 32, 32}));
  ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

  TF_ASSERT_OK_AND_ASSIGN(auto handle,
                          raiden::RaidenBufferHandle::Acquire(buffer.get()));
  std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers{
      {std::move(handle)}};

  StagingWindowConfig window_cfg{.window_chunks = window_chunks,
                                 .serve_windowed_pulls = true,
                                 .chunk_blocks = chunk_blocks,
                                 .assume_peer_support = true};

  auto engine = std::make_unique<KVCacheManagerWithTransfer>(
      layer_buffers, std::nullopt, 32, /*unsafe_skip_buffer_lock=*/true, 1,
      nullptr, /*node_id=*/0, /*local_control_port=*/0, /*max_blocks=*/16,
      /*num_slots=*/2, /*timeout_s=*/10.0, nullptr, window_cfg);

  std::vector<int64_t> src(k);
  std::iota(src.begin(), src.end(), 0);
  std::vector<int64_t> dst(k);
  if (!override_dst.empty()) {
    dst = override_dst;
  } else {
    std::iota(dst.begin(), dst.end(), k);
  }

  engine->NotifyForRead(std::string(req_id), uuid, src);
  std::string remote_endpoint =
      absl::StrCat("127.0.0.1:", engine->local_control_port());
  engine->StartRead(std::string(req_id), uuid, remote_endpoint, src, dst,
                    /*parallelism=*/1, local_host_bids,
                    /*deadline=*/std::nullopt);

  const bool expect_window_off =
      (local_host_bids.has_value() && !local_host_bids->empty()) ||
      ChunkPlan::HasDuplicates(dst);
  const int32_t expected_pushes =
      expect_window_off
          ? 1
          : static_cast<int32_t>((k + chunk_blocks - 1) / chunk_blocks);
  EXPECT_EQ(engine->base()->ExpectedPushesPerLayer(uuid), expected_pushes);

  bool done = false;
  for (int i = 0; i < 100 && !done; ++i) {
    auto [sent, recvd, failed] = engine->CompleteReadRaw();
    if (!failed.empty()) FAIL() << "Transfer failed for req_id=" << req_id;
    if (!recvd.empty()) done = true;
    if (!done) std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  ASSERT_TRUE(done) << "Transfer timed out for req_id=" << req_id;

  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
  auto read_back = literal->data<float>();
  ASSERT_EQ(read_back.size(), total_elements);

  // Map each unique destination block to the source block that ultimately
  // resides in it. When multiple source blocks target the same destination,
  // the one latest in transport order (sorted by remote_block_ids in
  // BuildLoadCopyPlan) overwrites earlier ones.
  absl::flat_hash_map<int64_t, int64_t> dst_to_src;
  std::vector<size_t> remote_order(k);
  std::iota(remote_order.begin(), remote_order.end(), 0);
  std::stable_sort(remote_order.begin(), remote_order.end(),
                   [&](size_t a, size_t b) { return src[a] < src[b]; });
  for (size_t idx : remote_order) {
    dst_to_src[dst[idx]] = src[idx];
  }

  int mismatches = 0;
  for (const auto& [dst_bid, src_bid] : dst_to_src) {
    int64_t src_offset = src_bid * elements_per_slice;
    int64_t dst_offset = dst_bid * elements_per_slice;
    for (int64_t i = 0; i < elements_per_slice; ++i) {
      if (read_back[dst_offset + i] != read_back[src_offset + i]) {
        ++mismatches;
      }
    }
  }
  EXPECT_EQ(mismatches, 0);
}

TEST(KVCacheManagerWithTransferTest, WindowedLoopbackK4C2W2DataCheck) {
  RunLoopbackDataCheckTest(4, 2, 2, 22001, "loopback_k4_c2_w2");
}

TEST(KVCacheManagerWithTransferTest, WindowedLoopbackK7C2W2DataCheck) {
  RunLoopbackDataCheckTest(7, 2, 2, 22002, "loopback_k7_c2_w2");
}

TEST(KVCacheManagerWithTransferTest,
     WindowDisabledWhenCallerSuppliesLocalHostBlockIds) {
  std::vector<int64_t> host_bids = {0, 1, 2, 3};
  RunLoopbackDataCheckTest(4, 2, 2, 22003, "window_disabled_host_bids",
                           host_bids);
}

TEST(KVCacheManagerWithTransferTest, WindowDisabledWhenDuplicateLocalBlockIds) {
  // src={0, 1, 2, 3}, dst={4, 5, 5, 6} (dst contains duplicate ID 5).
  std::vector<int64_t> duplicate_dst = {4, 5, 5, 6};
  RunLoopbackDataCheckTest(4, 2, 2, 22004, "window_disabled_dup_local_bids",
                           std::nullopt, duplicate_dst);
}

TEST(KVCacheManagerWithTransferTest,
     StagingWindowLoopbackRealWReuseK7C2W2BoundedPool) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());

  constexpr int64_t k = 7;
  constexpr int32_t chunk_blocks = 2;
  constexpr int32_t window_chunks = 2;
  constexpr int64_t ring_pool_blocks =
      window_chunks * chunk_blocks;  // 4 blocks
  constexpr int64_t total_blocks = 2 * k;
  constexpr int64_t elements_per_slice = 32 * 32;
  const int64_t total_elements = total_blocks * elements_per_slice;

  std::vector<float> host_data(total_elements, 0.0f);
  std::iota(host_data.begin(), host_data.begin() + k * elements_per_slice,
            1.0f);

  std::vector<int64_t> shape_dims = {total_blocks, 32, 32};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      pjrt_manager->BufferFromHost(host_data.data(), xla::F32, shape_dims));
  ASSERT_THAT(buffer->GetReadyFuture().Await(), ::absl_testing::IsOk());

  TF_ASSERT_OK_AND_ASSIGN(raiden::RaidenBufferHandle handle,
                          raiden::RaidenBufferHandle::Acquire(buffer.get()));
  std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers = {
      {std::move(handle)}};

  StagingWindowConfig window_cfg{.window_chunks = window_chunks,
                                 .serve_windowed_pulls = true,
                                 .chunk_blocks = chunk_blocks};

  // Host staging pool is sized for both prefill and decode rings in loopback:
  // num_slots = 2, max_blocks = 4 -> 8 host blocks (2 * W * C). Peak staging
  // usage across both rings is bounded by 2 * min(k, W * C).
  const int64_t loopback_host_blocks = 2 * ring_pool_blocks;
  auto engine = std::make_unique<KVCacheManagerWithTransfer>(
      layer_buffers, /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/loopback_host_blocks,
      /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/1, /*host_allocator=*/nullptr, /*node_id=*/0,
      /*local_control_port=*/0, /*max_blocks=*/ring_pool_blocks,
      /*num_slots=*/2, /*timeout_s=*/10.0, /*metrics_collector=*/nullptr,
      window_cfg);

  std::vector<int64_t> src_blocks = {0, 1, 2, 3, 4, 5, 6};
  std::vector<int64_t> dst_blocks = {7, 8, 9, 10, 11, 12, 13};
  constexpr uint64_t uuid = 23001;
  const std::string req_id = "bounded_pool_k7_c2_w2";

  engine->NotifyForRead(req_id, uuid, src_blocks);
  std::string remote_endpoint =
      absl::StrCat("127.0.0.1:", engine->local_control_port());
  engine->StartRead(req_id, uuid, remote_endpoint, src_blocks, dst_blocks);

  bool done = false;
  for (int i = 0; i < 100; ++i) {
    auto [done_sending, done_recving, failed_recving] =
        engine->CompleteReadRaw();
    if (!failed_recving.empty()) {
      FAIL() << "Transfer failed for req_id=" << req_id;
    }
    if (!done_recving.empty()) {
      done = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  ASSERT_TRUE(done) << "Transfer timed out for req_id=" << req_id;

  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
  auto read_back = literal->data<float>();
  ASSERT_EQ(read_back.size(), total_elements);

  int mismatches = 0;
  for (int64_t b = 0; b < k; ++b) {
    int64_t src_offset = b * elements_per_slice;
    int64_t dst_offset = (k + b) * elements_per_slice;
    for (int64_t i = 0; i < elements_per_slice; ++i) {
      if (read_back[dst_offset + i] != read_back[src_offset + i]) {
        ++mismatches;
      }
    }
  }
  EXPECT_EQ(mismatches, 0);
}

TEST(KVCacheManagerWithTransferTest,
     StagingWindowLoopbackRealWReuseBoundedPoolK16) {
  for (int32_t w : {2, 4, 8}) {
    TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                            TpuPjrtManager::GetDefault());

    // k = 16 (total_blocks = 32) is chosen to satisfy PJRT hardware layout
    // alignment (physical_size % major_dim == 0, where 524288 % 32 == 0) while
    // keeping n_chunks = k / chunk_blocks = 8, ensuring multi-chunk windowing
    // and slot reuse across W in {2, 4, 8}.
    constexpr int64_t k = 16;
    constexpr int32_t chunk_blocks = 2;
    const int64_t ring_pool_blocks = w * chunk_blocks;
    const int64_t loopback_host_blocks = 2 * ring_pool_blocks;
    constexpr int64_t total_blocks = 2 * k;
    constexpr int64_t elements_per_slice = 32 * 32;
    const int64_t total_elements = total_blocks * elements_per_slice;

    std::vector<float> host_data(total_elements, 0.0f);
    std::iota(host_data.begin(), host_data.begin() + k * elements_per_slice,
              1.0f);

    std::vector<int64_t> shape_dims = {total_blocks, 32, 32};
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<xla::PjRtBuffer> buffer,
        pjrt_manager->BufferFromHost(host_data.data(), xla::F32, shape_dims));
    ASSERT_THAT(buffer->GetReadyFuture().Await(), ::absl_testing::IsOk());

    TF_ASSERT_OK_AND_ASSIGN(raiden::RaidenBufferHandle handle,
                            raiden::RaidenBufferHandle::Acquire(buffer.get()));
    std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers = {
        {std::move(handle)}};

    StagingWindowConfig window_cfg{.window_chunks = w,
                                   .serve_windowed_pulls = true,
                                   .chunk_blocks = chunk_blocks};

    // Host staging pool is sized for both prefill and decode rings in loopback:
    // num_slots = 2, max_blocks = ring_pool_blocks -> 2 * W * C host blocks.
    auto engine = std::make_unique<KVCacheManagerWithTransfer>(
        layer_buffers, /*local_port=*/std::nullopt,
        /*host_blocks_to_allocate=*/loopback_host_blocks,
        /*unsafe_skip_buffer_lock=*/true,
        /*parallelism=*/1, /*host_allocator=*/nullptr, /*node_id=*/0,
        /*local_control_port=*/0, /*max_blocks=*/ring_pool_blocks,
        /*num_slots=*/2, /*timeout_s=*/10.0, /*metrics_collector=*/nullptr,
        window_cfg);

    std::vector<int64_t> src_blocks(k);
    std::iota(src_blocks.begin(), src_blocks.end(), 0);
    std::vector<int64_t> dst_blocks(k);
    std::iota(dst_blocks.begin(), dst_blocks.end(), k);
    const uint64_t uuid = 24000 + w;
    const std::string req_id = absl::StrCat("bounded_pool_k16_w", w);

    engine->NotifyForRead(req_id, uuid, src_blocks);
    std::string remote_endpoint =
        absl::StrCat("127.0.0.1:", engine->local_control_port());
    engine->StartRead(req_id, uuid, remote_endpoint, src_blocks, dst_blocks);

    bool done = false;
    for (int i = 0; i < 100; ++i) {
      auto [done_sending, done_recving, failed_recving] =
          engine->CompleteReadRaw();
      if (!failed_recving.empty()) {
        FAIL() << "Transfer failed for W=" << w;
      }
      if (!done_recving.empty()) {
        done = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ASSERT_TRUE(done) << "Transfer timed out for W=" << w;

    TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
    auto read_back = literal->data<float>();
    ASSERT_EQ(read_back.size(), total_elements);

    int mismatches = 0;
    for (int64_t b = 0; b < k; ++b) {
      int64_t src_offset = b * elements_per_slice;
      int64_t dst_offset = (k + b) * elements_per_slice;
      for (int64_t i = 0; i < elements_per_slice; ++i) {
        if (read_back[dst_offset + i] != read_back[src_offset + i]) {
          ++mismatches;
        }
      }
    }
    EXPECT_EQ(mismatches, 0) << "Data mismatch for W=" << w;
  }
}

TEST(KVCacheManagerWithTransferTest,
     FixedSlotPrefillServingWindowedDecodeLoopback) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());

  constexpr int64_t k = 7;
  constexpr int32_t chunk_blocks = 2;
  constexpr int64_t elements_per_slice = 32 * 32;
  const int64_t total_elements = k * elements_per_slice;

  std::vector<float> prefill_data(total_elements);
  std::iota(prefill_data.begin(), prefill_data.end(), 1.0f);

  std::vector<float> decode_data(total_elements, 0.0f);

  std::vector<int64_t> shape_dims = {k, 32, 32};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> prefill_buffer,
      pjrt_manager->BufferFromHost(prefill_data.data(), xla::F32, shape_dims));
  ASSERT_THAT(prefill_buffer->GetReadyFuture().Await(), ::absl_testing::IsOk());

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> decode_buffer,
      pjrt_manager->BufferFromHost(decode_data.data(), xla::F32, shape_dims));
  ASSERT_THAT(decode_buffer->GetReadyFuture().Await(), ::absl_testing::IsOk());

  TF_ASSERT_OK_AND_ASSIGN(
      raiden::RaidenBufferHandle prefill_handle,
      raiden::RaidenBufferHandle::Acquire(prefill_buffer.get()));
  std::vector<std::vector<raiden::RaidenBufferHandle>> prefill_layers = {
      {std::move(prefill_handle)}};

  TF_ASSERT_OK_AND_ASSIGN(
      raiden::RaidenBufferHandle decode_handle,
      raiden::RaidenBufferHandle::Acquire(decode_buffer.get()));
  std::vector<std::vector<raiden::RaidenBufferHandle>> decode_layers = {
      {std::move(decode_handle)}};

  // Prefill engine: window_chunks = 0, serve_windowed_pulls = true,
  // fixed-slot max_blocks = 4, num_slots = 1.
  StagingWindowConfig prefill_cfg{.window_chunks = 0,
                                  .serve_windowed_pulls = true,
                                  .chunk_blocks = chunk_blocks};

  auto prefill_engine = std::make_unique<KVCacheManagerWithTransfer>(
      prefill_layers, /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/4, /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/1, /*host_allocator=*/nullptr, /*node_id=*/0,
      /*local_control_port=*/0, /*max_blocks=*/4, /*num_slots=*/1,
      /*timeout_s=*/10.0, /*metrics_collector=*/nullptr, prefill_cfg);

  // Decode engine: W = 2, C = 2, fixed-slot max_blocks = 4, num_slots = 1.
  StagingWindowConfig decode_cfg{.window_chunks = 2,
                                 .serve_windowed_pulls = true,
                                 .chunk_blocks = chunk_blocks};

  auto decode_engine = std::make_unique<KVCacheManagerWithTransfer>(
      decode_layers, /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/4, /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/1, /*host_allocator=*/nullptr, /*node_id=*/1,
      /*local_control_port=*/0, /*max_blocks=*/4, /*num_slots=*/1,
      /*timeout_s=*/10.0, /*metrics_collector=*/nullptr, decode_cfg);

  std::vector<int64_t> src_blocks = {0, 1, 2, 3, 4, 5, 6};
  std::vector<int64_t> dst_blocks = {0, 1, 2, 3, 4, 5, 6};
  constexpr uint64_t uuid = 25001;
  const std::string req_id = "prefill_serving_windowed_decode";

  prefill_engine->NotifyForRead(req_id, uuid, src_blocks);
  std::string prefill_endpoint =
      absl::StrCat("127.0.0.1:", prefill_engine->local_control_port());
  decode_engine->StartRead(req_id, uuid, prefill_endpoint, src_blocks,
                           dst_blocks);

  bool done = false;
  for (int i = 0; i < 100; ++i) {
    auto [sent_p, recvd_p, failed_p] = prefill_engine->CompleteReadRaw();
    auto [sent_d, recvd_d, failed_d] = decode_engine->CompleteReadRaw();
    if (!failed_p.empty() || !failed_d.empty()) {
      FAIL() << "Transfer failed for req_id=" << req_id;
    }
    if (!recvd_d.empty()) {
      done = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  ASSERT_TRUE(done) << "Transfer timed out for req_id=" << req_id;

  TF_ASSERT_OK_AND_ASSIGN(auto literal, decode_buffer->ToLiteral().Await());
  auto read_back = literal->data<float>();
  ASSERT_EQ(read_back.size(), total_elements);

  int mismatches = 0;
  for (int64_t i = 0; i < total_elements; ++i) {
    if (read_back[i] != prefill_data[i]) {
      ++mismatches;
    }
  }
  EXPECT_EQ(mismatches, 0);
}

TEST(KVCacheManagerWithTransferIntegrationTest,
     BoundedWindowWaitAdmissionLoopback) {
  // Verifies that a windowed decode receiver configured with admission = kWait
  // successfully transfers data end-to-end with 0 mismatches.
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());

  const int64_t chunk_blocks = 2;
  const int64_t num_blocks = 4;
  const int64_t elements_per_block = 128;
  const int64_t total_elements = num_blocks * elements_per_block;

  std::vector<int64_t> shape_dims = {num_blocks, elements_per_block};
  std::vector<float> prefill_data(total_elements);
  std::vector<float> decode_data(total_elements, 0.0f);
  for (int64_t i = 0; i < total_elements; ++i) {
    prefill_data[i] = static_cast<float>(i + 1) * 0.5f;
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> prefill_buffer,
      pjrt_manager->BufferFromHost(prefill_data.data(), xla::F32, shape_dims));
  ASSERT_THAT(prefill_buffer->GetReadyFuture().Await(), ::absl_testing::IsOk());

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> decode_buffer,
      pjrt_manager->BufferFromHost(decode_data.data(), xla::F32, shape_dims));
  ASSERT_THAT(decode_buffer->GetReadyFuture().Await(), ::absl_testing::IsOk());

  TF_ASSERT_OK_AND_ASSIGN(
      raiden::RaidenBufferHandle prefill_handle,
      raiden::RaidenBufferHandle::Acquire(prefill_buffer.get()));
  std::vector<std::vector<raiden::RaidenBufferHandle>> prefill_layers = {
      {std::move(prefill_handle)}};

  TF_ASSERT_OK_AND_ASSIGN(
      raiden::RaidenBufferHandle decode_handle,
      raiden::RaidenBufferHandle::Acquire(decode_buffer.get()));
  std::vector<std::vector<raiden::RaidenBufferHandle>> decode_layers = {
      {std::move(decode_handle)}};

  StagingWindowConfig prefill_cfg{.window_chunks = 0,
                                  .serve_windowed_pulls = true,
                                  .chunk_blocks = chunk_blocks};

  auto prefill_engine = std::make_unique<KVCacheManagerWithTransfer>(
      prefill_layers, /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/4, /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/1, /*host_allocator=*/nullptr, /*node_id=*/0,
      /*local_control_port=*/0, /*max_blocks=*/4, /*num_slots=*/1,
      /*timeout_s=*/10.0, /*metrics_collector=*/nullptr, prefill_cfg);

  // Decode engine with W = 1, C = 2, kWait admission.
  StagingWindowConfig decode_cfg{
      .window_chunks = 1,
      .serve_windowed_pulls = true,
      .chunk_blocks = chunk_blocks,
      .admission = StagingWindowConfig::Admission::kWait};

  auto decode_engine = std::make_unique<KVCacheManagerWithTransfer>(
      decode_layers, /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/4, /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/1, /*host_allocator=*/nullptr, /*node_id=*/1,
      /*local_control_port=*/0, /*max_blocks=*/4, /*num_slots=*/1,
      /*timeout_s=*/10.0, /*metrics_collector=*/nullptr, decode_cfg);

  std::vector<int64_t> src_blocks = {0, 1, 2, 3};
  std::vector<int64_t> dst_blocks = {0, 1, 2, 3};
  constexpr uint64_t uuid = 25002;
  const std::string req_id = "wait_admission_loopback";

  prefill_engine->NotifyForRead(req_id, uuid, src_blocks);
  std::string prefill_endpoint =
      absl::StrCat("127.0.0.1:", prefill_engine->local_control_port());
  decode_engine->StartRead(req_id, uuid, prefill_endpoint, src_blocks,
                           dst_blocks);

  bool done = false;
  for (int i = 0; i < 100; ++i) {
    auto [sent_p, recvd_p, failed_p] = prefill_engine->CompleteReadRaw();
    auto [sent_d, recvd_d, failed_d] = decode_engine->CompleteReadRaw();
    if (!failed_p.empty() || !failed_d.empty()) {
      FAIL() << "Transfer failed for req_id=" << req_id;
    }
    if (!recvd_d.empty()) {
      done = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  ASSERT_TRUE(done) << "Transfer timed out for req_id=" << req_id;

  TF_ASSERT_OK_AND_ASSIGN(auto literal, decode_buffer->ToLiteral().Await());
  auto read_back = literal->data<float>();
  ASSERT_EQ(read_back.size(), total_elements);

  int mismatches = 0;
  for (int64_t i = 0; i < total_elements; ++i) {
    if (read_back[i] != prefill_data[i]) {
      ++mismatches;
    }
  }
  EXPECT_EQ(mismatches, 0);
}

}  // namespace
}  // namespace tpu_raiden
