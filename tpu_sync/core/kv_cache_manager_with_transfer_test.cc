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

#include "tpu_sync/core/kv_cache_manager_with_transfer.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/tpu_pjrt_manager.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/telemetry/mock_metrics_backend.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::IsOk;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::Gt;
using ::testing::IsEmpty;
using ::testing::Ne;
using ::testing::Not;

auto HasResolvedIpLabel(absl::string_view key) {
  return AllOf(
      Field(&telemetry::MetricLabel::key, key),
      Field(&telemetry::MetricLabel::value,
            AllOf(Not(IsEmpty()), Ne(telemetry::metric_labels::kUnknownIp))));
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

  auto handle_or = raiden::RaidenBufferHandle::Acquire(buffer.get());
  ASSERT_THAT(handle_or, IsOk());
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
  const uint64_t expected_slice_bytes = engine->base()->slice_byte_size();

  const auto pcie_labels_matcher = ElementsAre(
      HasResolvedIpLabel(telemetry::metric_labels::kHostIp),
      telemetry::MetricLabel{.key = telemetry::metric_labels::kLocalRank,
                             .value = "0"});

  auto mock_backend = std::make_unique<telemetry::MockMetricsBackend>();
  telemetry::MockMetricsBackend* raw_mock = mock_backend.get();
  EXPECT_CALL(*raw_mock,
              IncrementCounter(telemetry::metric_names::kSentBytesTotal,
                               testing::_, Gt(0)))
      .Times(1);
  EXPECT_CALL(*raw_mock,
              IncrementCounter(telemetry::metric_names::kReceivedBytesTotal,
                               testing::_, Gt(0)))
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
              HasResolvedIpLabel(telemetry::metric_labels::kDstIp)),
          Gt(0.0)))
      .Times(1);
  EXPECT_CALL(*raw_mock,
              ObserveHistogram(telemetry::metric_names::kD2hTransferTimeMs,
                               IsEmpty(), Gt(0.0)))
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

}  // namespace
}  // namespace tpu_raiden
