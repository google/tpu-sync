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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "peregrine/src/api/v1/rate_limiter.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/tpu_utils.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/transport/block_transport_delegate.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

ABSL_DECLARE_FLAG(size_t, raiden_weight_sync_host_buffer_scratchpad_size);

namespace tpu_raiden {
namespace weight_sync {
namespace {

class WeightSynchronizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::SetFlag(&FLAGS_raiden_weight_sync_host_buffer_scratchpad_size, 0);
    // Unified physical test parameters representing 64KB weight buffer E2E!
    num_layers_ = 1;
    num_shards_ = 1;
    slice_byte_size_ = 65536;  // 64KB
  }

  size_t num_layers_;
  size_t num_shards_;
  size_t slice_byte_size_;
};

TEST_F(WeightSynchronizerTest, PushWeightsCorrectnessE2e) {
  // 1. Instantiate three independent CPU-only synchronizers locally E2E!
  // ws_source represents the active RL Trainer
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers_, num_shards_, slice_byte_size_,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  // ws_dest1 and ws_dest2 represent inference server peers
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest1,
                          WeightSynchronizerBase::Create(
                              num_layers_, num_shards_, slice_byte_size_,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest2,
                          WeightSynchronizerBase::Create(
                              num_layers_, num_shards_, slice_byte_size_,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest1->local_port().has_value());
  ASSERT_TRUE(ws_dest2->local_port().has_value());

  std::string source_peer =
      "localhost:" + std::to_string(*ws_source->local_port());
  std::string dest1_peer =
      "localhost:" + std::to_string(*ws_dest1->local_port());
  std::string dest2_peer =
      "localhost:" + std::to_string(*ws_dest2->local_port());

  LOG(INFO) << "Launched C++ Weight Syncers: Source=" << source_peer
            << ", Dest1=" << dest1_peer << ", Dest2=" << dest2_peer;

  // 2. Populate the Trainer source buffer with distinct byte pattern (0xAB)
  // E2E!
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_host_buf,
                       ws_source->GetHostBuffer(0, 0));
  uint8_t* src_host_ptr = src_host_buf.data();
  std::memset(src_host_ptr, 0xAB, slice_byte_size_);

  // Populate inference servers with zeros baseline
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dest1_host_buf,
                       ws_dest1->GetHostBuffer(0, 0));
  uint8_t* dest1_host_ptr = dest1_host_buf.data();
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dest2_host_buf,
                       ws_dest2->GetHostBuffer(0, 0));
  uint8_t* dest2_host_ptr = dest2_host_buf.data();
  std::memset(dest1_host_ptr, 0x00, slice_byte_size_);
  std::memset(dest2_host_ptr, 0x00, slice_byte_size_);

  // Assert baseline state (zeros)
  EXPECT_EQ(dest1_host_ptr[0], 0x00);
  EXPECT_EQ(dest2_host_ptr[0], 0x00);

  // ==========================================================================
  // Test Scenario 1: Push weights from Trainer ws_source to both inference
  // peers!
  // ==========================================================================
  absl::Status push_status = ws_source->PushWeights({dest1_peer, dest2_peer});
  ASSERT_TRUE(push_status.ok()) << push_status.message();

  // Assert successful E2E network sockets streaming and exact byte parity!
  for (size_t i = 0; i < slice_byte_size_; ++i) {
    EXPECT_EQ(dest1_host_ptr[i], 0xAB) << "Mismatch at byte " << i;
    EXPECT_EQ(dest2_host_ptr[i], 0xAB) << "Mismatch at byte " << i;
  }
}

TEST_F(WeightSynchronizerTest, PushWeightsHeterogeneousCorrectness) {
  size_t num_layers = 2;
  size_t num_shards = 1;
  std::vector<size_t> slice_byte_sizes = {16384, 32768};

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_sizes,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_sizes,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());

  std::string source_peer =
      "localhost:" + std::to_string(*ws_source->local_port());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  // Populate source layers
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_l0_buf,
                       ws_source->GetHostBuffer(0, 0));
  uint8_t* src_l0_ptr = src_l0_buf.data();
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_l1_buf,
                       ws_source->GetHostBuffer(1, 0));
  uint8_t* src_l1_ptr = src_l1_buf.data();
  std::memset(src_l0_ptr, 0xAA, slice_byte_sizes[0]);
  std::memset(src_l1_ptr, 0xBB, slice_byte_sizes[1]);

  // Populate dest layers with zeros
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dest_l0_buf,
                       ws_dest->GetHostBuffer(0, 0));
  uint8_t* dest_l0_ptr = dest_l0_buf.data();
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dest_l1_buf,
                       ws_dest->GetHostBuffer(1, 0));
  uint8_t* dest_l1_ptr = dest_l1_buf.data();
  std::memset(dest_l0_ptr, 0x00, slice_byte_sizes[0]);
  std::memset(dest_l1_ptr, 0x00, slice_byte_sizes[1]);

  absl::Status push_status = ws_source->PushWeights({dest_peer});
  ASSERT_TRUE(push_status.ok()) << push_status.message();

  for (size_t i = 0; i < slice_byte_sizes[0]; ++i) {
    EXPECT_EQ(dest_l0_ptr[i], 0xAA) << "Mismatch at layer 0 byte " << i;
  }
  for (size_t i = 0; i < slice_byte_sizes[1]; ++i) {
    EXPECT_EQ(dest_l1_ptr[i], 0xBB) << "Mismatch at layer 1 byte " << i;
  }
}

// A layer with no shards has no device buffer to size the host staging buffers
// from, so Create() must fail instead of indexing shard 0. The Torch facade
// passes `[[]]` through to this overload unchanged.
TEST_F(WeightSynchronizerTest, CreateRejectsLayerWithNoShards) {
  const std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers(1);

  absl::StatusOr<std::unique_ptr<WeightSynchronizerBase>> ws =
      WeightSynchronizerBase::Create(layer_buffers);

  EXPECT_EQ(ws.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(WeightSynchronizerTest, CustomLayerNamesInitialization) {
  size_t num_layers = 2;
  size_t num_shards = 1;
  std::vector<size_t> slice_byte_sizes = {16384, 32768};
  std::vector<std::string> custom_names = {"my_layer_0", "my_layer_1"};

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_sizes,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1,
                              /*parallelism=*/1, /*listener_port=*/std::nullopt,
                              /*bind_ip=*/std::nullopt, custom_names));

  EXPECT_EQ(ws->layer_names().size(), 2);
  EXPECT_EQ(ws->layer_names()[0], "my_layer_0");
  EXPECT_EQ(ws->layer_names()[1], "my_layer_1");
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedExactBoundary) {
  size_t num_layers = 1;
  size_t num_shards = 1;
  size_t slice_byte_size = 16384;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_host_buf,
                       ws_source->GetHostBuffer(0, 0));
  uint8_t* src_host_ptr = src_host_buf.data();
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dest_host_buf,
                       ws_dest->GetHostBuffer(0, 0));
  uint8_t* dest_host_ptr = dest_host_buf.data();
  std::memset(src_host_ptr, 0xAB, slice_byte_size);
  std::memset(dest_host_ptr, 0x00, slice_byte_size);

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(12345);

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(slice_byte_size);
  entry->set_count(1);
  entry->set_layer_idx(0);

  ASSERT_OK(ws_dest->RegisterExpectedChunks(request.uuid(), 1));
  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
  ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));

  for (size_t i = 0; i < slice_byte_size; ++i) {
    EXPECT_EQ(dest_host_ptr[i], 0xAB) << "Mismatch at byte " << i;
  }
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedMultiPeerBroadcast) {
  size_t num_layers = 1;
  size_t num_shards = 1;
  size_t slice_byte_size = 16384;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest1,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest2,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest1->local_port().has_value());
  ASSERT_TRUE(ws_dest2->local_port().has_value());
  std::string dest_peer1 =
      "localhost:" + std::to_string(*ws_dest1->local_port());
  std::string dest_peer2 =
      "localhost:" + std::to_string(*ws_dest2->local_port());

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_host_buf,
                       ws_source->GetHostBuffer(0, 0));
  uint8_t* src_host_ptr = src_host_buf.data();
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dest_host_buf1,
                       ws_dest1->GetHostBuffer(0, 0));
  uint8_t* dest_host_ptr1 = dest_host_buf1.data();
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dest_host_buf2,
                       ws_dest2->GetHostBuffer(0, 0));
  uint8_t* dest_host_ptr2 = dest_host_buf2.data();
  std::memset(src_host_ptr, 0xCD, slice_byte_size);
  std::memset(dest_host_ptr1, 0x00, slice_byte_size);
  std::memset(dest_host_ptr2, 0x00, slice_byte_size);

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(67890);

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer1);
  entry->add_dst_peers(dest_peer1);
  entry->add_dst_peers(dest_peer2);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(slice_byte_size);
  entry->set_count(1);
  entry->set_layer_idx(0);

  ASSERT_OK(ws_dest1->RegisterExpectedChunks(request.uuid(), 1));
  ASSERT_OK(ws_dest2->RegisterExpectedChunks(request.uuid(), 1));
  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
  ASSERT_OK(ws_dest1->WaitForTransferCompletion(request.uuid()));
  ASSERT_OK(ws_dest2->WaitForTransferCompletion(request.uuid()));

  for (size_t i = 0; i < slice_byte_size; ++i) {
    EXPECT_EQ(dest_host_ptr1[i], 0xCD) << "Mismatch dest1 at byte " << i;
    EXPECT_EQ(dest_host_ptr2[i], 0xCD) << "Mismatch dest2 at byte " << i;
  }
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedMultiPeerBroadcastStrided) {
  size_t num_layers = 1;
  size_t num_shards = 1;
  size_t slice_byte_size = 16384;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest1,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest2,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest1->local_port().has_value());
  ASSERT_TRUE(ws_dest2->local_port().has_value());
  std::string dest_peer1 =
      "localhost:" + std::to_string(*ws_dest1->local_port());
  std::string dest_peer2 =
      "localhost:" + std::to_string(*ws_dest2->local_port());

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_host_buf,
                       ws_source->GetHostBuffer(0, 0));
  uint8_t* src_host_ptr = src_host_buf.data();
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dest_host_buf1,
                       ws_dest1->GetHostBuffer(0, 0));
  uint8_t* dest_host_ptr1 = dest_host_buf1.data();
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dest_host_buf2,
                       ws_dest2->GetHostBuffer(0, 0));
  uint8_t* dest_host_ptr2 = dest_host_buf2.data();
  std::memset(src_host_ptr, 0xEF, slice_byte_size);
  std::memset(dest_host_ptr1, 0x00, slice_byte_size);
  std::memset(dest_host_ptr2, 0x00, slice_byte_size);

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(67891);

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->add_dst_peers(dest_peer1);
  entry->add_dst_peers(dest_peer2);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(256);
  entry->set_src_stride_bytes(512);
  entry->set_dst_stride_bytes(512);
  entry->set_count(10);
  entry->set_layer_idx(0);

  // Source-and-destination strided chunk now pushes as a single task (1 chunk)
  ASSERT_OK(ws_dest1->RegisterExpectedChunks(request.uuid(), 1));
  ASSERT_OK(ws_dest2->RegisterExpectedChunks(request.uuid(), 1));
  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
  ASSERT_OK(ws_dest1->WaitForTransferCompletion(request.uuid()));
  ASSERT_OK(ws_dest2->WaitForTransferCompletion(request.uuid()));

  for (size_t c = 0; c < 10; ++c) {
    for (size_t b = 0; b < 256; ++b) {
      EXPECT_EQ(dest_host_ptr1[c * 512 + b], 0xEF);
      EXPECT_EQ(dest_host_ptr2[c * 512 + b], 0xEF);
    }
  }
}

TEST_F(WeightSynchronizerTest, PushWeightsResharded3DStridedSuccess) {
  size_t num_layers = 1;
  size_t num_shards = 1;
  // 3D strided layout: outer_counts = [2, 3], inner count = 4, size = 16
  constexpr size_t slice_byte_size = 8192;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_host_buf,
                       ws_source->GetHostBuffer(0, 0));
  uint8_t* src_host_ptr = src_host_buf.data();
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dest_host_buf,
                       ws_dest->GetHostBuffer(0, 0));
  uint8_t* dest_host_ptr = dest_host_buf.data();
  std::memset(src_host_ptr, 0x00, slice_byte_size);
  std::memset(dest_host_ptr, 0x00, slice_byte_size);

  // Populate 3D strided source pattern
  for (size_t d0 = 0; d0 < 2; ++d0) {
    for (size_t d1 = 0; d1 < 3; ++d1) {
      for (size_t c = 0; c < 4; ++c) {
        uint8_t val = static_cast<uint8_t>(1 + d0 * 12 + d1 * 4 + c);
        size_t s_off = d0 * 1024 + d1 * 256 + c * 32;
        std::memset(src_host_ptr + s_off, val, 16);
      }
    }
  }

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(67892);

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->add_dst_peers(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(16);
  entry->set_src_stride_bytes(32);
  entry->set_dst_stride_bytes(48);
  entry->set_count(4);
  entry->add_outer_counts(2);
  entry->add_outer_counts(3);
  entry->add_outer_src_strides_bytes(1024);
  entry->add_outer_src_strides_bytes(256);
  entry->add_outer_dst_strides_bytes(2048);
  entry->add_outer_dst_strides_bytes(512);
  entry->set_layer_idx(0);

  // 2 * 3 = 6 outer coordinates, each pushed as 1 strided BufferPushTask
  ASSERT_OK(ws_dest->RegisterExpectedChunks(request.uuid(), 6));
  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
  ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));

  for (size_t d0 = 0; d0 < 2; ++d0) {
    for (size_t d1 = 0; d1 < 3; ++d1) {
      for (size_t c = 0; c < 4; ++c) {
        uint8_t expected = static_cast<uint8_t>(1 + d0 * 12 + d1 * 4 + c);
        size_t d_off = d0 * 2048 + d1 * 512 + c * 48;
        for (size_t b = 0; b < 16; ++b) {
          EXPECT_EQ(dest_host_ptr[d_off + b], expected);
        }
      }
    }
  }
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedStridedSuccess) {
  size_t num_layers = 1;
  size_t num_shards = 1;
  constexpr size_t count = 1100;  // Exceeds IOV_MAX (1024)
  constexpr size_t slice_size = 32;
  constexpr size_t dst_stride = 128;
  constexpr size_t total_payload = count * slice_size;
  constexpr size_t dst_buffer_size = (count - 1) * dst_stride + slice_size;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, total_payload,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, dst_buffer_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_host_buf,
                       ws_source->GetHostBuffer(0, 0));
  uint8_t* src_host_ptr = src_host_buf.data();
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dest_host_buf,
                       ws_dest->GetHostBuffer(0, 0));
  uint8_t* dest_host_ptr = dest_host_buf.data();

  for (size_t i = 0; i < total_payload; ++i) {
    src_host_ptr[i] = static_cast<uint8_t>((i * 7 + 1) & 0xFF);
  }
  std::memset(dest_host_ptr, 0x00, dst_buffer_size);

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(54321);

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(slice_size);
  entry->set_src_stride_bytes(slice_size);
  entry->set_dst_stride_bytes(dst_stride);
  entry->set_count(count);
  entry->set_layer_idx(0);

  // Approach 1: 1 task is emitted for contiguous source to strided dest
  ASSERT_OK(ws_dest->RegisterExpectedChunks(request.uuid(), 1));
  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
  ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));

  // Verify strided destination contents and unwritten stride gaps
  for (size_t c = 0; c < count; ++c) {
    for (size_t b = 0; b < slice_size; ++b) {
      EXPECT_EQ(dest_host_ptr[c * dst_stride + b],
                src_host_ptr[c * slice_size + b])
          << "Mismatch at slice " << c << ", byte " << b;
    }
    if (c + 1 < count) {
      for (size_t g = slice_size; g < dst_stride; ++g) {
        EXPECT_EQ(dest_host_ptr[c * dst_stride + g], 0x00)
            << "Stride gap corrupted at slice " << c << ", byte " << g;
      }
    }
  }
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedZeroBytes) {
  size_t num_layers = 1;
  size_t num_shards = 1;
  size_t slice_byte_size = 16384;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_host_buf,
                       ws_source->GetHostBuffer(0, 0));
  uint8_t* src_host_ptr = src_host_buf.data();
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dest_host_buf,
                       ws_dest->GetHostBuffer(0, 0));
  uint8_t* dest_host_ptr = dest_host_buf.data();
  std::memset(src_host_ptr, 0xAB, slice_byte_size);
  std::memset(dest_host_ptr, 0x00, slice_byte_size);

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(12345);

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(0);  // 0 bytes request
  entry->set_count(1);
  entry->set_layer_idx(0);

  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();

  for (size_t i = 0; i < slice_byte_size; ++i) {
    EXPECT_EQ(dest_host_ptr[i], 0x00) << "Mismatch at byte " << i;
  }
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedOutOfBoundsError) {
  size_t num_layers = 1;
  size_t num_shards = 1;
  size_t slice_byte_size = 1024;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(12345);

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(500);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(600);  // 500 + 600 = 1100 > 1024
  entry->set_count(1);
  entry->set_layer_idx(0);

  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(WeightSynchronizerTest,
       PushWeightsReshardedUnmappedScheduleKeysMissingGlobalIndicesFails) {
  size_t num_layers = 1;
  size_t num_shards = 1;
  size_t slice_byte_size = 1024;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(12345);

  // Host only has num_shards = 1 (local shard 0). Schedules contain keys
  // {0, 1} broadcast from a 2-shard transfer. Key 1 is not among local shard
  // indices while global_shard_indices_ is empty.
  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry0 = (*schedules)[0].add_entries();
  entry0->set_dst_peer(dest_peer);
  entry0->set_dst_shard_idx(0);
  entry0->set_src_offset_bytes(0);
  entry0->set_dst_offset_bytes(0);
  entry0->set_size_bytes(1024);
  entry0->set_count(1);
  entry0->set_layer_idx(0);

  auto* entry1 = (*schedules)[1].add_entries();
  entry1->set_dst_peer(dest_peer);
  entry1->set_dst_shard_idx(1);
  entry1->set_src_offset_bytes(0);
  entry1->set_dst_offset_bytes(0);
  entry1->set_size_bytes(1024);
  entry1->set_count(1);
  entry1->set_layer_idx(0);

  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(
      absl::StrContains(status.message(), "global_shard_indices_ is empty"));
  EXPECT_TRUE(absl::StrContains(status.message(), "num_shards_=1"));
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedInvalidLayerIndexError) {
  size_t num_layers = 2;
  size_t num_shards = 1;
  size_t slice_byte_size = 1024;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(12345);

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(128);
  entry->set_count(1);
  entry->set_layer_idx(2);  // out of bounds (only 2 layers, indices 0 and 1)

  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedEmptySchedule) {
  size_t num_layers = 1;
  size_t num_shards = 1;
  size_t slice_byte_size = 1024;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(12345);
  // No schedules added

  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedFallbackByNameSuccess) {
  size_t num_layers = 2;
  size_t num_shards = 1;
  std::vector<size_t> slice_byte_sizes = {16384, 16384};
  std::vector<std::string> custom_names = {"layer_A", "layer_B"};

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_sizes,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1,
                              /*parallelism=*/1, /*listener_port=*/std::nullopt,
                              /*bind_ip=*/std::nullopt, custom_names));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_sizes,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1,
                              /*parallelism=*/1, /*listener_port=*/std::nullopt,
                              /*bind_ip=*/std::nullopt, custom_names));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  // Populate source layer_B (index 1) with 0xAB
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_l1_buf,
                       ws_source->GetHostBuffer(1, 0));
  uint8_t* src_l1_ptr = src_l1_buf.data();
  std::memset(src_l1_ptr, 0xAB, slice_byte_sizes[1]);

  // Populate dest layer_B with zeros
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dest_l1_buf,
                       ws_dest->GetHostBuffer(1, 0));
  uint8_t* dest_l1_ptr = dest_l1_buf.data();
  std::memset(dest_l1_ptr, 0x00, slice_byte_sizes[1]);

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(12345);

  auto* src_unit = request.add_src_units();
  src_unit->set_data_name("layer_B");

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(slice_byte_sizes[1]);
  entry->set_count(1);
  // Do not set layer_idx (forces fallback)

  ASSERT_OK(ws_dest->RegisterExpectedChunks(request.uuid(), 1));
  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
  ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));

  for (size_t i = 0; i < slice_byte_sizes[1]; ++i) {
    EXPECT_EQ(dest_l1_ptr[i], 0xAB) << "Mismatch at byte " << i;
  }
}

TEST_F(WeightSynchronizerTest,
       PushWeightsReshardedFallbackByNameNotFoundError) {
  size_t num_layers = 2;
  size_t num_shards = 1;
  std::vector<size_t> slice_byte_sizes = {16384, 16384};
  std::vector<std::string> custom_names = {"layer_A", "layer_B"};

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_sizes,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1,
                              /*parallelism=*/1, /*listener_port=*/std::nullopt,
                              /*bind_ip=*/std::nullopt, custom_names));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_sizes,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1,
                              /*parallelism=*/1, /*listener_port=*/std::nullopt,
                              /*bind_ip=*/std::nullopt, custom_names));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(12345);

  auto* src_unit = request.add_src_units();
  src_unit->set_data_name("unknown_layer");  // Not registered

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(1024);
  entry->set_count(1);
  // Do not set layer_idx

  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedSkipD2h) {
  auto client_status_or = xla::GetXlaPjrtCpuClient(xla::CpuClientOptions());
  ASSERT_TRUE(client_status_or.ok()) << client_status_or.status().message();
  auto client = std::move(client_status_or.value());

  size_t slice_byte_size = 1024;

  std::vector<uint8_t> src_device_data(slice_byte_size, 0xDD);
  auto memory_space_status_or =
      client->addressable_devices()[0]->default_memory_space();
  ASSERT_TRUE(memory_space_status_or.ok())
      << memory_space_status_or.status().message();
  xla::PjRtMemorySpace* memory_space = memory_space_status_or.value();

  auto src_buffer_status_or = client->BufferFromHostBuffer(
      src_device_data.data(), xla::U8, {static_cast<int64_t>(slice_byte_size)},
      /*byte_strides=*/std::nullopt,
      xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
      /*on_done_with_host_buffer=*/nullptr, memory_space,
      /*device_layout=*/nullptr);
  ASSERT_TRUE(src_buffer_status_or.ok())
      << src_buffer_status_or.status().message();
  auto src_pjrt_buffer = std::move(src_buffer_status_or.value());

  std::vector<uint8_t> dest_device_data(slice_byte_size, 0x00);
  auto dest_buffer_status_or = client->BufferFromHostBuffer(
      dest_device_data.data(), xla::U8, {static_cast<int64_t>(slice_byte_size)},
      /*byte_strides=*/std::nullopt,
      xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
      /*on_done_with_host_buffer=*/nullptr, memory_space,
      /*device_layout=*/nullptr);
  ASSERT_TRUE(dest_buffer_status_or.ok())
      << dest_buffer_status_or.status().message();
  auto dest_pjrt_buffer = std::move(dest_buffer_status_or.value());

  auto src_handle_or =
      raiden::RaidenBufferHandle::Acquire(src_pjrt_buffer.get());
  ASSERT_TRUE(src_handle_or.ok()) << src_handle_or.status().message();
  std::vector<std::vector<raiden::RaidenBufferHandle>> src_buffers = {
      {src_handle_or.value()}};

  auto dest_handle_or =
      raiden::RaidenBufferHandle::Acquire(dest_pjrt_buffer.get());
  ASSERT_TRUE(dest_handle_or.ok()) << dest_handle_or.status().message();
  std::vector<std::vector<raiden::RaidenBufferHandle>> dest_buffers = {
      {dest_handle_or.value()}};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> ws_source,
      WeightSynchronizerBase::Create(src_buffers, /*local_port=*/0));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> ws_dest,
      WeightSynchronizerBase::Create(dest_buffers, /*local_port=*/0));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_host_buf,
                       ws_source->GetHostBuffer(0, 0));
  uint8_t* src_host_ptr = src_host_buf.data();
  std::memset(src_host_ptr, 0xAA, slice_byte_size);

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dest_host_buf,
                       ws_dest->GetHostBuffer(0, 0));
  uint8_t* dest_host_ptr = dest_host_buf.data();
  std::memset(dest_host_ptr, 0x00, slice_byte_size);

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(12345);

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(slice_byte_size);
  entry->set_count(1);
  entry->set_layer_idx(0);

  ASSERT_OK(ws_dest->RegisterExpectedChunks(request.uuid(), 1));
  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
  ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));

  for (size_t i = 0; i < slice_byte_size; ++i) {
    EXPECT_EQ(dest_host_ptr[i], 0xAA) << "Mismatch at byte " << i;
  }

  std::memset(dest_host_ptr, 0x00, slice_byte_size);
  request.set_skip_d2h(false);
  request.set_uuid(12346);
  ASSERT_OK(ws_dest->RegisterExpectedChunks(request.uuid(), 1));
  status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
  ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));

  for (size_t i = 0; i < slice_byte_size; ++i) {
    EXPECT_EQ(dest_host_ptr[i], 0xDD) << "Mismatch at byte " << i;
  }
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedMultiLayerPipelineGrouped) {
  size_t num_layers = 4;
  size_t num_shards = 2;
  size_t slice_byte_size = 1024;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  for (size_t l = 0; l < num_layers; ++l) {
    for (size_t s = 0; s < num_shards; ++s) {
      TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_buf,
                           ws_source->GetHostBuffer(l, s));
      uint8_t* src_ptr = src_buf.data();
      TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dst_buf,
                           ws_dest->GetHostBuffer(l, s));
      uint8_t* dst_ptr = dst_buf.data();
      std::memset(src_ptr, static_cast<int>(l * 16 + s + 1), slice_byte_size);
      std::memset(dst_ptr, 0x00, slice_byte_size);
    }
  }

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(99901);

  auto* schedules = request.mutable_shard_push_schedules();
  for (size_t s = 0; s < num_shards; ++s) {
    for (size_t l = 0; l < num_layers; ++l) {
      auto* entry = (*schedules)[static_cast<int32_t>(s)].add_entries();
      entry->set_dst_peer(dest_peer);
      entry->set_dst_shard_idx(s);
      entry->set_src_offset_bytes(0);
      entry->set_dst_offset_bytes(0);
      entry->set_size_bytes(slice_byte_size);
      entry->set_count(1);
      entry->set_layer_idx(static_cast<int32_t>(l));
    }
  }

  // Test 1: Pipeline group size = 2 (2 layers per group)
  ws_source->SetPipelineGroupSize(2);
  EXPECT_EQ(ws_source->GetPipelineGroupSize(), 2);
  request.set_uuid(20001);
  ASSERT_OK(
      ws_dest->RegisterExpectedChunks(request.uuid(), num_layers * num_shards));
  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
  ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));

  for (size_t l = 0; l < num_layers; ++l) {
    for (size_t s = 0; s < num_shards; ++s) {
      TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> dst_buf,
                           ws_dest->GetHostBuffer(l, s));
      const uint8_t* dst_ptr = dst_buf.data();
      uint8_t expected_val = static_cast<uint8_t>(l * 16 + s + 1);
      for (size_t b = 0; b < slice_byte_size; ++b) {
        EXPECT_EQ(dst_ptr[b], expected_val)
            << "Mismatch at layer " << l << " shard " << s << " byte " << b;
      }
    }
  }

  // Test 2: Pipeline group size = 1 (layer-by-layer)
  for (size_t l = 0; l < num_layers; ++l) {
    for (size_t s = 0; s < num_shards; ++s) {
      TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dst_buf,
                           ws_dest->GetHostBuffer(l, s));
      std::memset(dst_buf.data(), 0x00, slice_byte_size);
    }
  }
  ws_source->SetPipelineGroupSize(1);
  EXPECT_EQ(ws_source->GetPipelineGroupSize(), 1);
  request.set_uuid(20002);
  ASSERT_OK(
      ws_dest->RegisterExpectedChunks(request.uuid(), num_layers * num_shards));
  status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
  ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));

  for (size_t l = 0; l < num_layers; ++l) {
    for (size_t s = 0; s < num_shards; ++s) {
      TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> dst_buf,
                           ws_dest->GetHostBuffer(l, s));
      const uint8_t* dst_ptr = dst_buf.data();
      uint8_t expected_val = static_cast<uint8_t>(l * 16 + s + 1);
      for (size_t b = 0; b < slice_byte_size; ++b) {
        EXPECT_EQ(dst_ptr[b], expected_val)
            << "Mismatch at layer " << l << " shard " << s << " byte " << b;
      }
    }
  }

  // Test 3: Pipeline group size = 0 (all layers in 1 group)
  for (size_t l = 0; l < num_layers; ++l) {
    for (size_t s = 0; s < num_shards; ++s) {
      TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dst_buf,
                           ws_dest->GetHostBuffer(l, s));
      std::memset(dst_buf.data(), 0x00, slice_byte_size);
    }
  }
  ws_source->SetPipelineGroupSize(0);
  EXPECT_EQ(ws_source->GetPipelineGroupSize(), 0);
  request.set_uuid(20003);
  ASSERT_OK(
      ws_dest->RegisterExpectedChunks(request.uuid(), num_layers * num_shards));
  status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
  ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));

  for (size_t l = 0; l < num_layers; ++l) {
    for (size_t s = 0; s < num_shards; ++s) {
      TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> dst_buf,
                           ws_dest->GetHostBuffer(l, s));
      const uint8_t* dst_ptr = dst_buf.data();
      uint8_t expected_val = static_cast<uint8_t>(l * 16 + s + 1);
      for (size_t b = 0; b < slice_byte_size; ++b) {
        EXPECT_EQ(dst_ptr[b], expected_val)
            << "Mismatch at layer " << l << " shard " << s << " byte " << b;
      }
    }
  }

  // Test 4: Pipeline group size configured via environment variable
  for (size_t l = 0; l < num_layers; ++l) {
    for (size_t s = 0; s < num_shards; ++s) {
      TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dst_buf,
                           ws_dest->GetHostBuffer(l, s));
      std::memset(dst_buf.data(), 0x00, slice_byte_size);
    }
  }
  ws_source->SetPipelineGroupSize(std::nullopt);
  setenv("RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE", "3", 1);
  EXPECT_EQ(ws_source->GetPipelineGroupSize(), 3);
  request.set_uuid(20004);
  ASSERT_OK(
      ws_dest->RegisterExpectedChunks(request.uuid(), num_layers * num_shards));
  status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
  ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));
  unsetenv("RAIDEN_WEIGHT_SYNC_PIPELINE_GROUP_SIZE");

  for (size_t l = 0; l < num_layers; ++l) {
    for (size_t s = 0; s < num_shards; ++s) {
      TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> dst_buf,
                           ws_dest->GetHostBuffer(l, s));
      const uint8_t* dst_ptr = dst_buf.data();
      uint8_t expected_val = static_cast<uint8_t>(l * 16 + s + 1);
      for (size_t b = 0; b < slice_byte_size; ++b) {
        EXPECT_EQ(dst_ptr[b], expected_val)
            << "Mismatch at layer " << l << " shard " << s << " byte " << b;
      }
    }
  }
}

TEST_F(WeightSynchronizerTest, BindWeights) {
  auto client_status_or = xla::GetXlaPjrtCpuClient(xla::CpuClientOptions());
  ASSERT_TRUE(client_status_or.ok()) << client_status_or.status().message();
  auto client = std::move(client_status_or.value());

  size_t slice_byte_size = 1024;
  auto memory_space_status_or =
      client->addressable_devices()[0]->default_memory_space();
  ASSERT_TRUE(memory_space_status_or.ok())
      << memory_space_status_or.status().message();
  xla::PjRtMemorySpace* memory_space = memory_space_status_or.value();

  // 1. Create ALL buffers first to ensure they outlive the synchronizers
  std::vector<uint8_t> src_device_data(slice_byte_size, 0x11);
  auto src_buffer_status_or = client->BufferFromHostBuffer(
      src_device_data.data(), xla::U8, {static_cast<int64_t>(slice_byte_size)},
      /*byte_strides=*/std::nullopt,
      xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
      /*on_done_with_host_buffer=*/nullptr, memory_space,
      /*device_layout=*/nullptr);
  ASSERT_TRUE(src_buffer_status_or.ok())
      << src_buffer_status_or.status().message();
  auto src_pjrt_buffer = std::move(src_buffer_status_or.value());

  std::vector<uint8_t> dest_device_data(slice_byte_size, 0x00);
  auto dest_buffer_status_or = client->BufferFromHostBuffer(
      dest_device_data.data(), xla::U8, {static_cast<int64_t>(slice_byte_size)},
      /*byte_strides=*/std::nullopt,
      xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
      /*on_done_with_host_buffer=*/nullptr, memory_space,
      /*device_layout=*/nullptr);
  ASSERT_TRUE(dest_buffer_status_or.ok())
      << dest_buffer_status_or.status().message();
  auto dest_pjrt_buffer = std::move(dest_buffer_status_or.value());

  std::vector<uint8_t> new_src_device_data(slice_byte_size, 0x22);
  auto new_src_buffer_status_or = client->BufferFromHostBuffer(
      new_src_device_data.data(), xla::U8,
      {static_cast<int64_t>(slice_byte_size)},
      /*byte_strides=*/std::nullopt,
      xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
      /*on_done_with_host_buffer=*/nullptr, memory_space,
      /*device_layout=*/nullptr);
  ASSERT_TRUE(new_src_buffer_status_or.ok())
      << new_src_buffer_status_or.status().message();
  auto new_src_pjrt_buffer = std::move(new_src_buffer_status_or.value());

  std::vector<uint8_t> new_dest_device_data(slice_byte_size, 0x00);
  auto new_dest_buffer_status_or = client->BufferFromHostBuffer(
      new_dest_device_data.data(), xla::U8,
      {static_cast<int64_t>(slice_byte_size)},
      /*byte_strides=*/std::nullopt,
      xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
      /*on_done_with_host_buffer=*/nullptr, memory_space,
      /*device_layout=*/nullptr);
  ASSERT_TRUE(new_dest_buffer_status_or.ok())
      << new_dest_buffer_status_or.status().message();
  auto new_dest_pjrt_buffer = std::move(new_dest_buffer_status_or.value());

  // 2. Create handles
  auto src_handle_or =
      raiden::RaidenBufferHandle::Acquire(src_pjrt_buffer.get());
  ASSERT_TRUE(src_handle_or.ok()) << src_handle_or.status().message();
  std::vector<std::vector<raiden::RaidenBufferHandle>> src_buffers = {
      {src_handle_or.value()}};

  auto dest_handle_or =
      raiden::RaidenBufferHandle::Acquire(dest_pjrt_buffer.get());
  ASSERT_TRUE(dest_handle_or.ok()) << dest_handle_or.status().message();
  std::vector<std::vector<raiden::RaidenBufferHandle>> dest_buffers = {
      {dest_handle_or.value()}};

  auto new_src_handle_or =
      raiden::RaidenBufferHandle::Acquire(new_src_pjrt_buffer.get());
  ASSERT_TRUE(new_src_handle_or.ok()) << new_src_handle_or.status().message();
  std::vector<std::vector<raiden::RaidenBufferHandle>> new_src_buffers = {
      {new_src_handle_or.value()}};

  auto new_dest_handle_or =
      raiden::RaidenBufferHandle::Acquire(new_dest_pjrt_buffer.get());
  ASSERT_TRUE(new_dest_handle_or.ok()) << new_dest_handle_or.status().message();
  std::vector<std::vector<raiden::RaidenBufferHandle>> new_dest_buffers = {
      {new_dest_handle_or.value()}};

  // 3. Create synchronizers (declared after buffers, so destroyed before them)
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> ws_source,
      WeightSynchronizerBase::Create(src_buffers, /*local_port=*/0));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> ws_dest,
      WeightSynchronizerBase::Create(dest_buffers, /*local_port=*/0));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  // 4. Bind weights
  absl::Status status = ws_source->BindWeights(new_src_buffers);
  ASSERT_TRUE(status.ok()) << status.message();

  status = ws_dest->BindWeights(new_dest_buffers);
  ASSERT_TRUE(status.ok()) << status.message();

  // 5. Sync
  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(false);
  request.set_uuid(12345);

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(slice_byte_size);
  entry->set_count(1);
  entry->set_layer_idx(0);

  ASSERT_OK(ws_dest->RegisterExpectedChunks(request.uuid(), 1));
  status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
  ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));

  auto h2d_future_or = ws_dest->H2d();
  ASSERT_TRUE(h2d_future_or.ok()) << h2d_future_or.status().message();
  status = h2d_future_or.value().Await();
  EXPECT_TRUE(status.ok()) << status.message();

  // 6. Verify
  std::vector<uint8_t> new_dest_readback(slice_byte_size, 0);
  auto copy_status =
      new_dest_pjrt_buffer
          ->CopyRawToHost(new_dest_readback.data(), 0, slice_byte_size)
          .Await();
  ASSERT_TRUE(copy_status.ok()) << copy_status.message();
  for (size_t i = 0; i < slice_byte_size; ++i) {
    EXPECT_EQ(new_dest_readback[i], 0x22)
        << "Mismatch in new dest buffer at byte " << i;
  }

  std::vector<uint8_t> old_dest_readback(slice_byte_size, 0xFF);
  copy_status =
      dest_pjrt_buffer
          ->CopyRawToHost(old_dest_readback.data(), 0, slice_byte_size)
          .Await();
  ASSERT_TRUE(copy_status.ok()) << copy_status.message();
  for (size_t i = 0; i < slice_byte_size; ++i) {
    EXPECT_EQ(old_dest_readback[i], 0x00)
        << "Mismatch in old dest buffer at byte " << i;
  }
}

// CPU-only instances never bind device buffers and must keep their legacy
// no-op D2h/H2d behavior; UnbindWeights() must not be required to use them.
TEST_F(WeightSynchronizerTest, CpuOnlyInstanceD2hH2dRemainNoOp) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws,
                          WeightSynchronizerBase::Create(
                              num_layers_, num_shards_, slice_byte_size_,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  auto d2h_or = ws->D2h();
  ASSERT_TRUE(d2h_or.ok()) << d2h_or.status().message();
  EXPECT_OK(d2h_or.value().Await());

  auto h2d_or = ws->H2d();
  ASSERT_TRUE(h2d_or.ok()) << h2d_or.status().message();
  EXPECT_OK(h2d_or.value().Await());
}

TEST_F(WeightSynchronizerTest, UnbindWeightsFailsTransfersUntilRebound) {
  auto client_status_or = xla::GetXlaPjrtCpuClient(xla::CpuClientOptions());
  ASSERT_TRUE(client_status_or.ok()) << client_status_or.status().message();
  auto client = std::move(client_status_or.value());

  size_t slice_byte_size = 1024;
  auto memory_space_status_or =
      client->addressable_devices()[0]->default_memory_space();
  ASSERT_TRUE(memory_space_status_or.ok())
      << memory_space_status_or.status().message();
  xla::PjRtMemorySpace* memory_space = memory_space_status_or.value();

  auto make_buffer = [&](uint8_t fill) {
    std::vector<uint8_t> data(slice_byte_size, fill);
    auto buf_or = client->BufferFromHostBuffer(
        data.data(), xla::U8, {static_cast<int64_t>(slice_byte_size)},
        /*byte_strides=*/std::nullopt,
        xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
        /*on_done_with_host_buffer=*/nullptr, memory_space,
        /*device_layout=*/nullptr);
    EXPECT_TRUE(buf_or.ok()) << buf_or.status().message();
    return std::move(buf_or.value());
  };

  // Buffers are declared before synchronizers so they outlive them.
  auto src_pjrt_buffer = make_buffer(0x11);
  auto dest_pjrt_buffer = make_buffer(0x00);
  auto new_src_pjrt_buffer = make_buffer(0x33);
  auto new_dest_pjrt_buffer = make_buffer(0x00);

  auto acquire = [](xla::PjRtBuffer* buf) {
    auto h = raiden::RaidenBufferHandle::Acquire(buf);
    EXPECT_TRUE(h.ok()) << h.status().message();
    return std::vector<std::vector<raiden::RaidenBufferHandle>>{{h.value()}};
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> ws_source,
      WeightSynchronizerBase::Create(acquire(src_pjrt_buffer.get()),
                                     /*local_port=*/0));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> ws_dest,
      WeightSynchronizerBase::Create(acquire(dest_pjrt_buffer.get()),
                                     /*local_port=*/0));
  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  auto make_request = [&](uint64_t uuid) {
    tpu_sync::rpc::StartTransferRequest request;
    request.set_skip_d2h(false);
    request.set_uuid(uuid);
    auto* entry = (*request.mutable_shard_push_schedules())[0].add_entries();
    entry->set_dst_peer(dest_peer);
    entry->set_dst_shard_idx(0);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(slice_byte_size);
    entry->set_count(1);
    entry->set_layer_idx(0);
    return request;
  };

  // 1. Baseline transfer works while bound.
  {
    auto request = make_request(1);
    ASSERT_OK(ws_dest->RegisterExpectedChunks(request.uuid(), 1));
    ASSERT_OK(ws_source->PushWeightsResharded(request));
    ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));
    auto h2d_or = ws_dest->H2d();
    ASSERT_TRUE(h2d_or.ok()) << h2d_or.status().message();
    ASSERT_OK(h2d_or.value().Await());
  }

  // 2. Unbind both sides (explicit API on source, empty-bind alias on dest).
  ws_source->UnbindWeights();
  ASSERT_OK(ws_dest->BindWeights({}));

  // 3. Direct device entry points must fail with FailedPrecondition.
  EXPECT_EQ(ws_source->D2h().status().code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(ws_dest->H2d().status().code(),
            absl::StatusCode::kFailedPrecondition);

  // 4. Controller-driven push with skip_d2h=false must also fail rather than
  //    silently shipping stale host staging bytes.
  {
    auto request = make_request(2);
    absl::Status status = ws_source->PushWeightsResharded(request);
    EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition)
        << status.message();
  }

  // 5. Re-binding on the same instances fully recovers (no re-creation).
  ASSERT_OK(ws_source->BindWeights(acquire(new_src_pjrt_buffer.get())));
  ASSERT_OK(ws_dest->BindWeights(acquire(new_dest_pjrt_buffer.get())));
  {
    auto request = make_request(3);
    ASSERT_OK(ws_dest->RegisterExpectedChunks(request.uuid(), 1));
    ASSERT_OK(ws_source->PushWeightsResharded(request));
    ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));
    auto h2d_or = ws_dest->H2d();
    ASSERT_TRUE(h2d_or.ok()) << h2d_or.status().message();
    ASSERT_OK(h2d_or.value().Await());
  }

  std::vector<uint8_t> readback(slice_byte_size, 0);
  ASSERT_OK(
      new_dest_pjrt_buffer->CopyRawToHost(readback.data(), 0, slice_byte_size)
          .Await());
  for (size_t i = 0; i < slice_byte_size; ++i) {
    EXPECT_EQ(readback[i], 0x33) << "Mismatch at byte " << i;
  }
}

TEST_F(WeightSynchronizerTest, TilingSkipScenarios) {
  auto client_status_or = xla::GetXlaPjrtCpuClient(xla::CpuClientOptions());
  ASSERT_TRUE(client_status_or.ok()) << client_status_or.status().message();
  auto client = std::move(client_status_or.value());

  auto memory_space_status_or =
      client->addressable_devices()[0]->default_memory_space();
  ASSERT_TRUE(memory_space_status_or.ok())
      << memory_space_status_or.status().message();
  xla::PjRtMemorySpace* memory_space = memory_space_status_or.value();

  struct TestCaseRunner {
    xla::PjRtClient* client;
    xla::PjRtMemorySpace* memory_space;

    struct WSWrapper {
      std::unique_ptr<xla::PjRtBuffer> pjrt_buffer;
      std::unique_ptr<WeightSynchronizerBase> ws;
    };

    WSWrapper CreateWS(xla::PrimitiveType type, absl::Span<const int64_t> dims,
                       const xla::Layout& layout,
                       std::vector<uint8_t>& placeholder) {
      xla::Shape default_shape = xla::ShapeUtil::MakeShape(type, dims);
      size_t byte_size = xla::ShapeUtil::ByteSizeOf(default_shape);
      placeholder.resize(byte_size, 0);

      auto buffer_status_or =
          client->BufferFromHostBuffer(placeholder.data(), type, dims,
                                       /*byte_strides=*/std::nullopt,
                                       xla::PjRtClient::HostBufferSemantics::
                                           kImmutableUntilTransferCompletes,
                                       /*on_done_with_host_buffer=*/nullptr,
                                       memory_space, /*device_layout=*/nullptr);
      EXPECT_TRUE(buffer_status_or.ok()) << buffer_status_or.status().message();
      auto pjrt_buffer = std::move(buffer_status_or.value());

      auto handle_or = raiden::RaidenBufferHandle::Acquire(pjrt_buffer.get());
      EXPECT_TRUE(handle_or.ok()) << handle_or.status().message();

      handle_or.value().shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
          type, dims, layout.minor_to_major(), layout.tiles());

      std::vector<std::vector<raiden::RaidenBufferHandle>> buffers = {
          {handle_or.value()}};

      absl::StatusOr<std::unique_ptr<WeightSynchronizerBase>> ws =
          WeightSynchronizerBase::Create(buffers, /*local_port=*/0);
      EXPECT_TRUE(ws.ok()) << ws.status().message();
      return WSWrapper{std::move(pjrt_buffer), *std::move(ws)};
    }
  } runner{client.get(), memory_space};

  // Case 1: Tiled Layout, perfect division (no padding), identical layouts.
  {
    xla::Layout layout =
        xla::LayoutUtil::MakeLayout({1, 0}, {xla::Tile({4, 4})});
    std::vector<uint8_t> src_device_placeholder;
    auto src_ws = runner.CreateWS(xla::PrimitiveType::F32, {8, 8}, layout,
                                  src_device_placeholder);

    TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_host_buf,
                         src_ws.ws->GetHostBuffer(0, 0));
    float* src_host = reinterpret_cast<float*>(src_host_buf.data());
    for (int i = 0; i < 64; ++i) {
      src_host[i] = static_cast<float>(i);
    }

    // Test with skip = true
    tpu_sync::rpc::StartTransferRequest req_true;
    (*req_true.mutable_skip_tiling())[0] = true;
    src_ws.ws->StoreSkipTiling(123, req_true);
    absl::StatusOr<raiden::PjRtCopyFuture> h2d_fut = src_ws.ws->H2d(123);
    ASSERT_TRUE(h2d_fut.ok()) << h2d_fut.status().message();
    ASSERT_TRUE(h2d_fut.value().Await().ok());

    std::vector<float> dst_host_raw(64, 0.0f);
    auto src_handle_or =
        raiden::RaidenBufferHandle::Acquire(src_ws.pjrt_buffer.get());
    ASSERT_TRUE(src_handle_or.ok());
    auto raw_d2h_fut = src_handle_or.value().CopyRawDeviceToHost(
        dst_host_raw.data(), 0, 64 * sizeof(float));
    ASSERT_TRUE(raw_d2h_fut.Await().ok());

    // Verify that the data is NOT permuted on device (tiling was skipped)
    for (int i = 0; i < 64; ++i) {
      EXPECT_EQ(dst_host_raw[i], static_cast<float>(i))
          << "Mismatch at index " << i;
    }

    // Now test with skip = false (tiling should occur)
    tpu_sync::rpc::StartTransferRequest req_false;
    (*req_false.mutable_skip_tiling())[0] = false;
    src_ws.ws->StoreSkipTiling(456, req_false);
    h2d_fut = src_ws.ws->H2d(456);
    ASSERT_TRUE(h2d_fut.ok()) << h2d_fut.status().message();
    ASSERT_TRUE(h2d_fut.value().Await().ok());

    raw_d2h_fut = src_handle_or.value().CopyRawDeviceToHost(
        dst_host_raw.data(), 0, 64 * sizeof(float));
    ASSERT_TRUE(raw_d2h_fut.Await().ok());
    EXPECT_NE(dst_host_raw[4], 4.0f);
    EXPECT_EQ(dst_host_raw[16], 4.0f);
  }
}

TEST_F(WeightSynchronizerTest, TilingActiveByDefault) {
  auto client_status_or = xla::GetXlaPjrtCpuClient(xla::CpuClientOptions());
  ASSERT_TRUE(client_status_or.ok()) << client_status_or.status().message();
  auto client = std::move(client_status_or.value());

  auto memory_space_status_or =
      client->addressable_devices()[0]->default_memory_space();
  ASSERT_TRUE(memory_space_status_or.ok())
      << memory_space_status_or.status().message();
  xla::PjRtMemorySpace* memory_space = memory_space_status_or.value();

  struct TestCaseRunner {
    xla::PjRtClient* client;
    xla::PjRtMemorySpace* memory_space;

    struct WSWrapper {
      std::unique_ptr<xla::PjRtBuffer> pjrt_buffer;
      std::unique_ptr<WeightSynchronizerBase> ws;
    };

    WSWrapper CreateWS(xla::PrimitiveType type, absl::Span<const int64_t> dims,
                       const xla::Layout& layout,
                       std::vector<uint8_t>& placeholder) {
      xla::Shape default_shape = xla::ShapeUtil::MakeShape(type, dims);
      size_t byte_size = xla::ShapeUtil::ByteSizeOf(default_shape);
      placeholder.resize(byte_size, 0);

      auto buffer_status_or =
          client->BufferFromHostBuffer(placeholder.data(), type, dims,
                                       /*byte_strides=*/std::nullopt,
                                       xla::PjRtClient::HostBufferSemantics::
                                           kImmutableUntilTransferCompletes,
                                       /*on_done_with_host_buffer=*/nullptr,
                                       memory_space, /*device_layout=*/nullptr);
      EXPECT_TRUE(buffer_status_or.ok()) << buffer_status_or.status().message();
      auto pjrt_buffer = std::move(buffer_status_or.value());

      auto handle_or = raiden::RaidenBufferHandle::Acquire(pjrt_buffer.get());
      EXPECT_TRUE(handle_or.ok()) << handle_or.status().message();

      handle_or.value().shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
          type, dims, layout.minor_to_major(), layout.tiles());

      std::vector<std::vector<raiden::RaidenBufferHandle>> buffers = {
          {handle_or.value()}};

      absl::StatusOr<std::unique_ptr<WeightSynchronizerBase>> ws =
          WeightSynchronizerBase::Create(buffers, /*local_port=*/0);
      EXPECT_TRUE(ws.ok()) << ws.status().message();
      return WSWrapper{std::move(pjrt_buffer), *std::move(ws)};
    }
  } runner{client.get(), memory_space};

  // Case: Tiled Layout, calling H2d() without arguments should run tiling by
  // default.
  {
    xla::Layout layout =
        xla::LayoutUtil::MakeLayout({1, 0}, {xla::Tile({4, 4})});
    std::vector<uint8_t> src_device_placeholder;
    auto src_ws = runner.CreateWS(xla::PrimitiveType::F32, {8, 8}, layout,
                                  src_device_placeholder);

    TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_host_buf,
                         src_ws.ws->GetHostBuffer(0, 0));
    float* src_host = reinterpret_cast<float*>(src_host_buf.data());
    for (int i = 0; i < 64; ++i) {
      src_host[i] = static_cast<float>(i);
    }

    // Call H2d without arguments -> should use tiling by default
    absl::StatusOr<raiden::PjRtCopyFuture> h2d_fut = src_ws.ws->H2d();
    ASSERT_TRUE(h2d_fut.ok()) << h2d_fut.status().message();
    ASSERT_TRUE(h2d_fut.value().Await().ok());

    std::vector<float> dst_host_raw(64, 0.0f);
    auto src_handle_or =
        raiden::RaidenBufferHandle::Acquire(src_ws.pjrt_buffer.get());
    ASSERT_TRUE(src_handle_or.ok());
    auto raw_d2h_fut = src_handle_or.value().CopyRawDeviceToHost(
        dst_host_raw.data(), 0, 64 * sizeof(float));
    ASSERT_TRUE(raw_d2h_fut.Await().ok());

    // Verify that the data IS permuted on device (tiling occurred)
    EXPECT_NE(dst_host_raw[4], 4.0f);
    EXPECT_EQ(dst_host_raw[16], 4.0f);
  }
}

TEST_F(WeightSynchronizerTest, H2dTilesAllShardsAndLayersConcurrently) {
  constexpr int kNumShards = 4;
  constexpr int kNumLayers = 3;
  constexpr int kRows = 8;
  constexpr int kCols = 8;
  constexpr int kElems = kRows * kCols;
  xla::CpuClientOptions options;
  options.cpu_device_count = kNumShards;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          xla::GetXlaPjrtCpuClient(options));
  ASSERT_GE(client->addressable_devices().size(), kNumShards);

  const xla::Layout layout =
      xla::LayoutUtil::MakeLayout({1, 0}, {xla::Tile({4, 4})});
  std::vector<float> placeholder(kElems, 0.0f);
  std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> pjrt_buffers(
      kNumLayers);
  std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers(
      kNumLayers);
  for (int l = 0; l < kNumLayers; ++l) {
    for (int s = 0; s < kNumShards; ++s) {
      TF_ASSERT_OK_AND_ASSIGN(
          xla::PjRtMemorySpace * memory_space,
          client->addressable_devices()[s]->default_memory_space());
      TF_ASSERT_OK_AND_ASSIGN(
          std::unique_ptr<xla::PjRtBuffer> pjrt_buffer,
          client->BufferFromHostBuffer(
              placeholder.data(), xla::PrimitiveType::F32, {kRows, kCols},
              /*byte_strides=*/std::nullopt,
              xla::PjRtClient::HostBufferSemantics::
                  kImmutableUntilTransferCompletes,
              /*on_done_with_host_buffer=*/nullptr, memory_space,
              /*device_layout=*/nullptr));
      TF_ASSERT_OK_AND_ASSIGN(
          raiden::RaidenBufferHandle handle,
          raiden::RaidenBufferHandle::Acquire(pjrt_buffer.get()));
      handle.shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
          xla::PrimitiveType::F32, {kRows, kCols}, layout.minor_to_major(),
          layout.tiles());
      layer_buffers[l].push_back(handle);
      pjrt_buffers[l].push_back(std::move(pjrt_buffer));
    }
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> ws,
      WeightSynchronizerBase::Create(layer_buffers, /*local_port=*/0));

  auto value_at = [](int l, int s, int i) {
    return static_cast<float>(1000 * l + 100 * s + i);
  };
  for (int l = 0; l < kNumLayers; ++l) {
    for (int s = 0; s < kNumShards; ++s) {
      TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> host_buf,
                           ws->GetHostBuffer(l, s));
      float* host = reinterpret_cast<float*>(host_buf.data());
      for (int i = 0; i < kElems; ++i) host[i] = value_at(l, s, i);
    }
  }

  TF_ASSERT_OK_AND_ASSIGN(raiden::PjRtCopyFuture h2d_future, ws->H2d());
  ABSL_ASSERT_OK(h2d_future.Await());

  for (int l = 0; l < kNumLayers; ++l) {
    for (int s = 0; s < kNumShards; ++s) {
      TF_ASSERT_OK_AND_ASSIGN(
          raiden::RaidenBufferHandle handle,
          raiden::RaidenBufferHandle::Acquire(pjrt_buffers[l][s].get()));
      std::vector<float> device(kElems, -1.0f);
      ABSL_ASSERT_OK(
          handle.CopyRawDeviceToHost(device.data(), 0, kElems * sizeof(float))
              .Await());
      for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kCols; ++c) {
          // Row-major 4x4 tiles: tiles are laid out row-major, as are the
          // elements within each tile.
          const int physical =
              ((r / 4) * (kCols / 4) + c / 4) * 16 + (r % 4) * 4 + c % 4;
          EXPECT_EQ(device[physical], value_at(l, s, r * kCols + c))
              << "layer " << l << " shard " << s << " r " << r << " c " << c;
        }
      }
    }
  }
}

TEST_F(WeightSynchronizerTest, LazyTiledBufferAllocationSavesMemory) {
  auto client_status_or = xla::GetXlaPjrtCpuClient(xla::CpuClientOptions());
  ASSERT_TRUE(client_status_or.ok()) << client_status_or.status().message();
  auto client = std::move(client_status_or.value());

  auto memory_space_status_or =
      client->addressable_devices()[0]->default_memory_space();
  ASSERT_TRUE(memory_space_status_or.ok())
      << memory_space_status_or.status().message();
  xla::PjRtMemorySpace* memory_space = memory_space_status_or.value();

  struct TestCaseRunner {
    xla::PjRtClient* client;
    xla::PjRtMemorySpace* memory_space;

    struct WSWrapper {
      std::unique_ptr<xla::PjRtBuffer> pjrt_buffer;
      std::unique_ptr<WeightSynchronizerBase> ws;
    };

    WSWrapper CreateWS(xla::PrimitiveType type, absl::Span<const int64_t> dims,
                       const xla::Layout& layout,
                       std::vector<uint8_t>& placeholder) {
      xla::Shape default_shape = xla::ShapeUtil::MakeShape(type, dims);
      size_t byte_size = xla::ShapeUtil::ByteSizeOf(default_shape);
      placeholder.resize(byte_size, 0);

      auto buffer_status_or =
          client->BufferFromHostBuffer(placeholder.data(), type, dims,
                                       /*byte_strides=*/std::nullopt,
                                       xla::PjRtClient::HostBufferSemantics::
                                           kImmutableUntilTransferCompletes,
                                       /*on_done_with_host_buffer=*/nullptr,
                                       memory_space, /*device_layout=*/nullptr);
      EXPECT_TRUE(buffer_status_or.ok()) << buffer_status_or.status().message();
      auto pjrt_buffer = std::move(buffer_status_or.value());

      auto handle_or = raiden::RaidenBufferHandle::Acquire(pjrt_buffer.get());
      EXPECT_TRUE(handle_or.ok()) << handle_or.status().message();

      handle_or.value().shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
          type, dims, layout.minor_to_major(), layout.tiles());

      std::vector<std::vector<raiden::RaidenBufferHandle>> buffers = {
          {handle_or.value()}};

      absl::StatusOr<std::unique_ptr<WeightSynchronizerBase>> ws =
          WeightSynchronizerBase::Create(buffers, /*local_port=*/0);
      EXPECT_TRUE(ws.ok()) << ws.status().message();
      return WSWrapper{std::move(pjrt_buffer), *std::move(ws)};
    }
  } runner{client.get(), memory_space};

  xla::Layout layout = xla::LayoutUtil::MakeLayout({1, 0}, {xla::Tile({4, 4})});
  std::vector<uint8_t> src_device_placeholder;
  auto src_ws = runner.CreateWS(xla::PrimitiveType::F32, {8, 8}, layout,
                                src_device_placeholder);

  // 1. Verify tiled_ptr is NOT pre-allocated during initialization (lazy
  // allocation saves memory)
  EXPECT_EQ(src_ws.ws->GetTiledPointer(0, 0), nullptr);

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_host_buf,
                       src_ws.ws->GetHostBuffer(0, 0));
  float* src_host = reinterpret_cast<float*>(src_host_buf.data());
  for (int i = 0; i < 64; ++i) {
    src_host[i] = static_cast<float>(i);
  }

  // 2. When skip_tiling is true, H2d executes without allocating tiled_ptr
  tpu_sync::rpc::StartTransferRequest req_true;
  (*req_true.mutable_skip_tiling())[0] = true;
  src_ws.ws->StoreSkipTiling(999, req_true);
  auto h2d_skip = src_ws.ws->H2d(999);
  ASSERT_TRUE(h2d_skip.ok());
  ASSERT_TRUE(h2d_skip.value().Await().ok());

  // tiled_ptr MUST remain nullptr when tiling is skipped (0 bytes wasted)
  EXPECT_EQ(src_ws.ws->GetTiledPointer(0, 0), nullptr);

  // 3. When skip_tiling is false, tiled_ptr is lazily allocated on demand
  // during H2d
  tpu_sync::rpc::StartTransferRequest req_false;
  (*req_false.mutable_skip_tiling())[0] = false;
  src_ws.ws->StoreSkipTiling(1000, req_false);
  auto h2d_tile = src_ws.ws->H2d(1000);
  ASSERT_TRUE(h2d_tile.ok());
  ASSERT_TRUE(h2d_tile.value().Await().ok());

  // Now tiled_ptr is non-null because tiling actually ran!
  EXPECT_NE(src_ws.ws->GetTiledPointer(0, 0), nullptr);

  // 4. Verify that layers on the same shard reuse the exact same shared
  // scratchpad pointer
  EXPECT_EQ(src_ws.ws->GetTiledPointer(0, 0), src_ws.ws->GetTiledPointer(1, 0));
}

TEST_F(WeightSynchronizerTest, OneDimensionalTiledTensorH2dAndD2hRoundtrip) {
  auto client_status_or = xla::GetXlaPjrtCpuClient(xla::CpuClientOptions());
  ASSERT_TRUE(client_status_or.ok()) << client_status_or.status().message();
  auto client = std::move(client_status_or.value());

  auto memory_space_status_or =
      client->addressable_devices()[0]->default_memory_space();
  ASSERT_TRUE(memory_space_status_or.ok())
      << memory_space_status_or.status().message();
  xla::PjRtMemorySpace* memory_space = memory_space_status_or.value();

  struct TestCaseRunner {
    xla::PjRtClient* client;
    xla::PjRtMemorySpace* memory_space;

    struct WSWrapper {
      std::unique_ptr<xla::PjRtBuffer> pjrt_buffer;
      std::unique_ptr<WeightSynchronizerBase> ws;
    };

    WSWrapper CreateWS(xla::PrimitiveType type, absl::Span<const int64_t> dims,
                       const xla::Layout& layout,
                       std::vector<uint8_t>& placeholder) {
      // Allocate device placeholder with full physical tiled size (1024
      // elements = 4096 bytes)
      placeholder.resize(1024 * sizeof(float), 0);

      auto buffer_status_or =
          client->BufferFromHostBuffer(placeholder.data(), type, {1024},
                                       /*byte_strides=*/std::nullopt,
                                       xla::PjRtClient::HostBufferSemantics::
                                           kImmutableUntilTransferCompletes,
                                       /*on_done_with_host_buffer=*/nullptr,
                                       memory_space, /*device_layout=*/nullptr);
      EXPECT_TRUE(buffer_status_or.ok()) << buffer_status_or.status().message();
      auto pjrt_buffer = std::move(buffer_status_or.value());

      auto handle_or = raiden::RaidenBufferHandle::Acquire(pjrt_buffer.get());
      EXPECT_TRUE(handle_or.ok()) << handle_or.status().message();

      // Logical shape is {64}, with physical tile {8, 128}
      handle_or.value().shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
          type, dims, layout.minor_to_major(), layout.tiles());

      std::vector<std::vector<raiden::RaidenBufferHandle>> buffers = {
          {handle_or.value()}};

      absl::StatusOr<std::unique_ptr<WeightSynchronizerBase>> ws =
          WeightSynchronizerBase::Create(buffers, /*local_port=*/0);
      EXPECT_TRUE(ws.ok()) << ws.status().message();
      return WSWrapper{std::move(pjrt_buffer), *std::move(ws)};
    }
  } runner{client.get(), memory_space};

  // 1D tensor: shape {64}, F32, standard hardware 2D tile {8, 128}
  // Logical size = 64 * 4 = 256 bytes.
  // Physical tile size = 8 * 128 * 4 = 4096 bytes.
  xla::Layout layout = xla::LayoutUtil::MakeLayout({0}, {xla::Tile({8, 128})});
  std::vector<uint8_t> device_placeholder;
  auto ws_wrap = runner.CreateWS(xla::PrimitiveType::F32, {64}, layout,
                                 device_placeholder);

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> host_buf,
                       ws_wrap.ws->GetHostBuffer(0, 0));
  float* host_ptr = reinterpret_cast<float*>(host_buf.data());
  for (int i = 0; i < 64; ++i) {
    host_ptr[i] = static_cast<float>(i + 1);
  }

  // 1. Run H2D (linear host buffer -> tiled device buffer)
  absl::StatusOr<raiden::PjRtCopyFuture> h2d_fut = ws_wrap.ws->H2d();
  ASSERT_TRUE(h2d_fut.ok()) << h2d_fut.status().message();
  ASSERT_TRUE(h2d_fut.value().Await().ok());

  // Inspect physical device buffer:
  // For shape {64} with tile {8, 128}, physical buffer has 1024 elements (4096
  // bytes). Element index 32 (value 33.0f) is placed at physical tile index 128
  // (byte offset 512).
  std::vector<float> dev_raw(1024, 0.0f);
  auto handle_or =
      raiden::RaidenBufferHandle::Acquire(ws_wrap.pjrt_buffer.get());
  ASSERT_TRUE(handle_or.ok());
  auto raw_fut = handle_or.value().CopyRawDeviceToHost(dev_raw.data(), 0,
                                                       1024 * sizeof(float));
  ASSERT_TRUE(raw_fut.Await().ok());
  // Verify element 32 (value 33.0f) is written to device memory:
  EXPECT_EQ(dev_raw[32], 33.0f);

  // 2. Zero out host memory to verify D2H roundtrip
  for (int i = 0; i < 64; ++i) {
    host_ptr[i] = 0.0f;
  }

  // 3. Run D2H (tiled device buffer -> linear host buffer)
  absl::StatusOr<raiden::PjRtCopyFuture> d2h_fut = ws_wrap.ws->D2h();
  ASSERT_TRUE(d2h_fut.ok()) << d2h_fut.status().message();
  ASSERT_TRUE(d2h_fut.value().Await().ok());

  // 4. Verify all 64 elements preserved exact numerical values after D2H
  // detiling
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ(host_ptr[i], static_cast<float>(i + 1))
        << "Numerical mismatch at index " << i << " after D2H detiling!";
  }
}

TEST_F(WeightSynchronizerTest, DeferredH2dTilesHostBufferOnDataReceived) {
  auto client_status_or = xla::GetXlaPjrtCpuClient(xla::CpuClientOptions());
  ASSERT_TRUE(client_status_or.ok()) << client_status_or.status().message();
  std::unique_ptr<xla::PjRtClient> client = *std::move(client_status_or);
  absl::StatusOr<xla::PjRtMemorySpace*> memory_space =
      client->addressable_devices()[0]->default_memory_space();
  ASSERT_TRUE(memory_space.ok()) << memory_space.status().message();

  std::vector<float> placeholder(64, 0.0f);
  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> pjrt_buffer =
      client->BufferFromHostBuffer(
          placeholder.data(), xla::PrimitiveType::F32, {8, 8},
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::
              kImmutableUntilTransferCompletes,
          /*on_done_with_host_buffer=*/nullptr, *memory_space,
          /*device_layout=*/nullptr);
  ASSERT_TRUE(pjrt_buffer.ok()) << pjrt_buffer.status().message();
  absl::StatusOr<raiden::RaidenBufferHandle> handle =
      raiden::RaidenBufferHandle::Acquire(pjrt_buffer->get());
  ASSERT_TRUE(handle.ok()) << handle.status().message();
  handle->shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {8, 8}, {1, 0}, {xla::Tile({4, 4})});
  std::vector<std::vector<raiden::RaidenBufferHandle>> buffers = {{*handle}};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> ws,
      WeightSynchronizerBase::Create(
          buffers, /*local_port=*/0, /*external_host_ptrs=*/std::nullopt,
          /*unsafe_skip_buffer_lock=*/false, /*parallelism=*/1,
          /*listener_port=*/std::nullopt, /*bind_ip=*/std::nullopt,
          /*layer_names=*/std::vector<std::string>{}, /*auto_h2d=*/false));

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> host_buf, ws->GetHostBuffer(0, 0));
  float* host = reinterpret_cast<float*>(host_buf.data());
  for (int i = 0; i < 64; ++i) {
    host[i] = static_cast<float>(i);
  }

  // With auto_h2d=false, OnDataReceived of a leaf receiver tiles the host
  // buffer in place.
  constexpr uint64_t kUuid = 777;
  tpu_sync::rpc::StartTransferRequest req;
  req.set_host_tiling_mode(tpu_sync::rpc::HOST_TILING_MODE_ON_ARRIVAL);
  ws->StoreSkipTiling(kUuid, req);
  ASSERT_OK(ws->OnLayerDataReceived(/*layer_idx=*/0, kUuid));
  ASSERT_OK(ws->OnDataReceived(kUuid));
  EXPECT_EQ(host[4], 8.0f);
  EXPECT_EQ(host[16], 4.0f);

  // H2d then performs a raw copy of the already-tiled host buffer.
  absl::StatusOr<raiden::PjRtCopyFuture> h2d_fut = ws->H2d(kUuid);
  ASSERT_TRUE(h2d_fut.ok()) << h2d_fut.status().message();
  ASSERT_OK(h2d_fut->Await());

  std::vector<float> dev_raw(64, -1.0f);
  ASSERT_OK(handle->CopyRawDeviceToHost(dev_raw.data(), 0, 64 * sizeof(float))
                .Await());
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ(dev_raw[i], host[i]) << "Mismatch at index " << i;
  }
  EXPECT_EQ(dev_raw[16], 4.0f);
}

TEST_F(WeightSynchronizerTest, H2dTracksPerLayerHostLayout) {
  auto client_status_or = xla::GetXlaPjrtCpuClient(xla::CpuClientOptions());
  ASSERT_TRUE(client_status_or.ok()) << client_status_or.status().message();
  std::unique_ptr<xla::PjRtClient> client = *std::move(client_status_or);
  absl::StatusOr<xla::PjRtMemorySpace*> memory_space =
      client->addressable_devices()[0]->default_memory_space();
  ASSERT_TRUE(memory_space.ok()) << memory_space.status().message();

  std::vector<float> placeholder(64, 0.0f);
  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> pjrt_buffer =
      client->BufferFromHostBuffer(
          placeholder.data(), xla::PrimitiveType::F32, {8, 8},
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::
              kImmutableUntilTransferCompletes,
          /*on_done_with_host_buffer=*/nullptr, *memory_space,
          /*device_layout=*/nullptr);
  ASSERT_TRUE(pjrt_buffer.ok()) << pjrt_buffer.status().message();
  absl::StatusOr<raiden::RaidenBufferHandle> handle =
      raiden::RaidenBufferHandle::Acquire(pjrt_buffer->get());
  ASSERT_TRUE(handle.ok()) << handle.status().message();
  handle->shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {8, 8}, {1, 0}, {xla::Tile({4, 4})});
  std::vector<std::vector<raiden::RaidenBufferHandle>> buffers = {{*handle}};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> ws,
      WeightSynchronizerBase::Create(
          buffers, /*local_port=*/0, /*external_host_ptrs=*/std::nullopt,
          /*unsafe_skip_buffer_lock=*/false, /*parallelism=*/1,
          /*listener_port=*/std::nullopt, /*bind_ip=*/std::nullopt,
          /*layer_names=*/std::vector<std::string>{}, /*auto_h2d=*/false));

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> host_buf, ws->GetHostBuffer(0, 0));
  float* host = reinterpret_cast<float*>(host_buf.data());
  for (int i = 0; i < 64; ++i) {
    host[i] = static_cast<float>(i);
  }

  auto copy_device_to_host = [&]() {
    std::vector<float> dev_raw(64, -1.0f);
    EXPECT_OK(handle->CopyRawDeviceToHost(dev_raw.data(), 0, 64 * sizeof(float))
                  .Await());
    return dev_raw;
  };
  auto h2d = [&]() {
    absl::StatusOr<raiden::PjRtCopyFuture> h2d_fut = ws->H2d();
    ASSERT_TRUE(h2d_fut.ok()) << h2d_fut.status().message();
    ASSERT_OK(h2d_fut->Await());
  };

  // Untracked logical data: H2d tiles into a scratchpad and leaves the host
  // buffer logical.
  h2d();
  EXPECT_EQ(host[4], 4.0f);
  EXPECT_EQ(host[16], 16.0f);
  std::vector<float> dev_raw = copy_device_to_host();
  EXPECT_EQ(dev_raw[4], 8.0f);
  EXPECT_EQ(dev_raw[16], 4.0f);

  // A tracked transfer to a leaf receiver tiles the host buffer in place, and
  // H2d copies it without tiling again.
  constexpr uint64_t kLeafUuid = 778;
  tpu_sync::rpc::StartTransferRequest leaf_req;
  leaf_req.set_host_tiling_mode(tpu_sync::rpc::HOST_TILING_MODE_ON_ARRIVAL);
  ws->StoreSkipTiling(kLeafUuid, leaf_req);
  ASSERT_OK(ws->OnLayerDataReceived(/*layer_idx=*/0, kLeafUuid));
  ASSERT_OK(ws->OnDataReceived(kLeafUuid));
  EXPECT_EQ(host[4], 8.0f);
  EXPECT_EQ(host[16], 4.0f);
  h2d();
  dev_raw = copy_device_to_host();
  EXPECT_EQ(dev_raw[4], 8.0f);
  EXPECT_EQ(dev_raw[16], 4.0f);

  // Arming a deferred receive marks the layer's host buffer untiled so that the
  // next H2D tiles the newly received logical data.
  for (int i = 0; i < 64; ++i) {
    host[i] = static_cast<float>(i + 100);
  }
  constexpr uint64_t kDeferredUuid = 780;
  tpu_sync::rpc::StartTransferRequest deferred_req;
  ws->StoreSkipTiling(kDeferredUuid, deferred_req);
  ASSERT_OK(ws->RegisterExpectedLayerChunks(kDeferredUuid, {{0, 1}}));
  ASSERT_OK(ws->OnLayerDataReceived(/*layer_idx=*/0, kDeferredUuid));
  ASSERT_OK(ws->OnDataReceived(kDeferredUuid));
  h2d();
  dev_raw = copy_device_to_host();
  EXPECT_EQ(dev_raw[4], 108.0f);
  EXPECT_EQ(dev_raw[16], 104.0f);

  // D2h writes logical-layout bytes back into the host buffer, so the next
  // H2d tiles again.
  ws->StoreSkipTiling(kLeafUuid, leaf_req);
  ASSERT_OK(ws->OnLayerDataReceived(/*layer_idx=*/0, kLeafUuid));
  ASSERT_OK(ws->OnDataReceived(kLeafUuid));
  absl::StatusOr<raiden::PjRtCopyFuture> d2h_fut = ws->D2h();
  ASSERT_TRUE(d2h_fut.ok()) << d2h_fut.status().message();
  ASSERT_OK(d2h_fut->Await());
  EXPECT_EQ(host[4], 104.0f);
  h2d();
  dev_raw = copy_device_to_host();
  EXPECT_EQ(dev_raw[4], 108.0f);
  EXPECT_EQ(dev_raw[16], 104.0f);
}

// A receiver armed with HOST_TILING_MODE_UNSPECIFIED (the default) or
// HOST_TILING_MODE_DEFERRED keeps its host buffers in logical layout and tiles
// into a scratchpad at H2D.
TEST_F(WeightSynchronizerTest, DeferredReceiverKeepsHostBufferLogical) {
  auto client_status_or = xla::GetXlaPjrtCpuClient(xla::CpuClientOptions());
  ASSERT_TRUE(client_status_or.ok()) << client_status_or.status().message();
  std::unique_ptr<xla::PjRtClient> client = *std::move(client_status_or);
  absl::StatusOr<xla::PjRtMemorySpace*> memory_space =
      client->addressable_devices()[0]->default_memory_space();
  ASSERT_TRUE(memory_space.ok()) << memory_space.status().message();

  for (bool auto_h2d : {false, true}) {
    SCOPED_TRACE(absl::StrCat("auto_h2d=", auto_h2d));
    std::vector<float> placeholder(64, 0.0f);
    absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> pjrt_buffer =
        client->BufferFromHostBuffer(
            placeholder.data(), xla::PrimitiveType::F32, {8, 8},
            /*byte_strides=*/std::nullopt,
            xla::PjRtClient::HostBufferSemantics::
                kImmutableUntilTransferCompletes,
            /*on_done_with_host_buffer=*/nullptr, *memory_space,
            /*device_layout=*/nullptr);
    ASSERT_TRUE(pjrt_buffer.ok()) << pjrt_buffer.status().message();
    absl::StatusOr<raiden::RaidenBufferHandle> handle =
        raiden::RaidenBufferHandle::Acquire(pjrt_buffer->get());
    ASSERT_TRUE(handle.ok()) << handle.status().message();
    handle->shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::F32, {8, 8}, {1, 0}, {xla::Tile({4, 4})});
    std::vector<std::vector<raiden::RaidenBufferHandle>> buffers = {{*handle}};

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<WeightSynchronizerBase> ws,
        WeightSynchronizerBase::Create(
            buffers, /*local_port=*/0, /*external_host_ptrs=*/std::nullopt,
            /*unsafe_skip_buffer_lock=*/false, /*parallelism=*/1,
            /*listener_port=*/std::nullopt, /*bind_ip=*/std::nullopt,
            /*layer_names=*/std::vector<std::string>{}, auto_h2d));

    TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> host_buf, ws->GetHostBuffer(0, 0));
    float* host = reinterpret_cast<float*>(host_buf.data());
    for (int i = 0; i < 64; ++i) {
      host[i] = static_cast<float>(i);
    }

    for (tpu_sync::rpc::HostTilingMode mode :
         {tpu_sync::rpc::HOST_TILING_MODE_UNSPECIFIED,
          tpu_sync::rpc::HOST_TILING_MODE_DEFERRED}) {
      constexpr uint64_t kUuid = 779;
      tpu_sync::rpc::StartTransferRequest req;
      req.set_host_tiling_mode(mode);
      ws->StoreSkipTiling(kUuid, req);
      ASSERT_OK(ws->OnLayerDataReceived(/*layer_idx=*/0, kUuid));
      ASSERT_OK(ws->OnDataReceived(kUuid));
      for (int i = 0; i < 64; ++i) {
        EXPECT_EQ(host[i], static_cast<float>(i)) << "Host mismatch at " << i;
      }

      if (!auto_h2d) {
        absl::StatusOr<raiden::PjRtCopyFuture> h2d_fut = ws->H2d(kUuid);
        ASSERT_TRUE(h2d_fut.ok()) << h2d_fut.status().message();
        ASSERT_OK(h2d_fut->Await());
        EXPECT_EQ(host[4], 4.0f);
      }
      std::vector<float> dev_raw(64, -1.0f);
      ASSERT_OK(
          handle->CopyRawDeviceToHost(dev_raw.data(), 0, 64 * sizeof(float))
              .Await());
      EXPECT_EQ(dev_raw[4], 8.0f);
      EXPECT_EQ(dev_raw[16], 4.0f);
    }
  }
}

// A block push whose streams carry shard subsets must not trigger the auto
// H2D (which copies every shard) until every shard of the uuid has landed.
TEST_F(WeightSynchronizerTest, AutoH2dWaitsForEveryShardOfBlockPush) {
  auto client_status_or = xla::GetXlaPjrtCpuClient(xla::CpuClientOptions());
  ASSERT_TRUE(client_status_or.ok()) << client_status_or.status().message();
  std::unique_ptr<xla::PjRtClient> client = *std::move(client_status_or);
  absl::StatusOr<xla::PjRtMemorySpace*> memory_space =
      client->addressable_devices()[0]->default_memory_space();
  ASSERT_TRUE(memory_space.ok()) << memory_space.status().message();

  constexpr int kShards = 2;
  std::vector<float> placeholder(64, 0.0f);
  std::vector<std::unique_ptr<xla::PjRtBuffer>> pjrt_buffers;
  std::vector<raiden::RaidenBufferHandle> shard_handles;
  for (int sh = 0; sh < kShards; ++sh) {
    absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> pjrt_buffer =
        client->BufferFromHostBuffer(
            placeholder.data(), xla::PrimitiveType::F32, {8, 8},
            /*byte_strides=*/std::nullopt,
            xla::PjRtClient::HostBufferSemantics::
                kImmutableUntilTransferCompletes,
            /*on_done_with_host_buffer=*/nullptr, *memory_space,
            /*device_layout=*/nullptr);
    ASSERT_TRUE(pjrt_buffer.ok()) << pjrt_buffer.status().message();
    absl::StatusOr<raiden::RaidenBufferHandle> handle =
        raiden::RaidenBufferHandle::Acquire(pjrt_buffer->get());
    ASSERT_TRUE(handle.ok()) << handle.status().message();
    handle->shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::F32, {8, 8}, {1, 0}, {xla::Tile({4, 4})});
    pjrt_buffers.push_back(*std::move(pjrt_buffer));
    shard_handles.push_back(*handle);
  }
  std::vector<std::vector<raiden::RaidenBufferHandle>> buffers = {
      shard_handles};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> ws,
      WeightSynchronizerBase::Create(
          buffers, /*local_port=*/0, /*external_host_ptrs=*/std::nullopt,
          /*unsafe_skip_buffer_lock=*/false, /*parallelism=*/1,
          /*listener_port=*/std::nullopt, /*bind_ip=*/std::nullopt,
          /*layer_names=*/std::vector<std::string>{}, /*auto_h2d=*/true));
  for (int sh = 0; sh < kShards; ++sh) {
    TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> host_buf,
                         ws->GetHostBuffer(0, sh));
    float* host = reinterpret_cast<float*>(host_buf.data());
    for (int i = 0; i < 64; ++i) {
      host[i] = static_cast<float>(i);
    }
  }

  constexpr uint64_t kUuid = 781;
  auto device_value_at = [&](int sh, int idx) -> float {
    std::vector<float> dev_raw(64, -1.0f);
    absl::Status copied =
        shard_handles[sh]
            .CopyRawDeviceToHost(dev_raw.data(), 0, 64 * sizeof(float))
            .Await();
    EXPECT_OK(copied);
    return dev_raw[idx];
  };

  // Shard 0 alone: H2D is deferred, device still holds the placeholder.
  ASSERT_OK(
      ws->OnBlockShardsReceived(/*block_ids=*/{0}, /*shard_ids=*/{0}, kUuid));
  EXPECT_EQ(device_value_at(0, 4), 0.0f);
  EXPECT_EQ(device_value_at(1, 4), 0.0f);

  // Shard 1 completes the uuid: H2D runs for every shard.
  ASSERT_OK(
      ws->OnBlockShardsReceived(/*block_ids=*/{0}, /*shard_ids=*/{1}, kUuid));
  EXPECT_EQ(device_value_at(0, 4), 8.0f);
  EXPECT_EQ(device_value_at(1, 4), 8.0f);
}

TEST_F(WeightSynchronizerTest, DrainPendingH2dDrainsActivePendingFutures) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> ws,
      WeightSynchronizerBase::Create(
          /*num_layers=*/2, /*num_shards=*/1, slice_byte_size_,
          /*local_port=*/0, /*host_blocks_to_allocate=*/1,
          /*parallelism=*/2, /*listener_port=*/0, /*bind_ip=*/std::nullopt,
          /*layer_names=*/std::vector<std::string>{"layer_0", "layer_1"},
          /*auto_h2d=*/true));

  uint64_t uuid = 54321;
  EXPECT_TRUE(ws->OnLayerDataReceived(0, uuid).ok());
  EXPECT_TRUE(ws->OnLayerDataReceived(1, uuid).ok());

  // DrainPendingH2d should drain all scheduled futures and mark uuid completed
  ws->DrainPendingH2d();

  EXPECT_TRUE(ws->WaitForTransferCompletion(uuid).ok());
  ws->ForgetPushProgress(uuid);
}

TEST_F(WeightSynchronizerTest, OnDataReceivedWaitsForAllExpectedLayers) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> ws,
      WeightSynchronizerBase::Create(
          /*num_layers=*/3, /*num_shards=*/1, slice_byte_size_,
          /*local_port=*/0, /*host_blocks_to_allocate=*/1,
          /*parallelism=*/2, /*listener_port=*/0, /*bind_ip=*/std::nullopt,
          /*layer_names=*/
          std::vector<std::string>{"layer_0", "layer_1", "layer_2"},
          /*auto_h2d=*/true));

  const uint64_t uuid = 65432;
  // Layer 2 has a zero count and never fires OnLayerDataReceived.
  ASSERT_TRUE(
      ws->RegisterExpectedLayerChunks(uuid, {{0, 1}, {1, 1}, {2, 0}}).ok());
  ASSERT_TRUE(ws->OnLayerDataReceived(0, uuid).ok());

  absl::Notification data_received_done;
  absl::Status data_received_status;
  std::thread receiver([&] {
    data_received_status = ws->OnDataReceived(uuid);
    data_received_done.Notify();
  });

  // Layer 1's callback has not arrived yet, so OnDataReceived must block.
  EXPECT_FALSE(
      data_received_done.WaitForNotificationWithTimeout(absl::Seconds(1)));

  ASSERT_TRUE(ws->OnLayerDataReceived(1, uuid).ok());
  receiver.join();
  EXPECT_TRUE(data_received_done.HasBeenNotified());
  EXPECT_TRUE(data_received_status.ok()) << data_received_status;
  EXPECT_TRUE(ws->WaitForTransferCompletion(uuid).ok());
  ws->ForgetPushProgress(uuid);
}

// The transport acks the sender before running the receive callbacks, so a
// caller may start H2D before OnLayerDataReceived has run. H2d must wait for
// in-place tiling of every layer armed by RegisterExpectedLayerChunks, and
// fail if the receive is dropped before the layer arrives.
TEST_F(WeightSynchronizerTest, H2dWaitsForArmedInPlaceTiling) {
  auto client_status_or = xla::GetXlaPjrtCpuClient(xla::CpuClientOptions());
  ASSERT_TRUE(client_status_or.ok()) << client_status_or.status().message();
  std::unique_ptr<xla::PjRtClient> client = *std::move(client_status_or);
  absl::StatusOr<xla::PjRtMemorySpace*> memory_space =
      client->addressable_devices()[0]->default_memory_space();
  ASSERT_TRUE(memory_space.ok()) << memory_space.status().message();

  std::vector<float> placeholder(64, 0.0f);
  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> pjrt_buffer =
      client->BufferFromHostBuffer(
          placeholder.data(), xla::PrimitiveType::F32, {8, 8},
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::
              kImmutableUntilTransferCompletes,
          /*on_done_with_host_buffer=*/nullptr, *memory_space,
          /*device_layout=*/nullptr);
  ASSERT_TRUE(pjrt_buffer.ok()) << pjrt_buffer.status().message();
  absl::StatusOr<raiden::RaidenBufferHandle> handle =
      raiden::RaidenBufferHandle::Acquire(pjrt_buffer->get());
  ASSERT_TRUE(handle.ok()) << handle.status().message();
  handle->shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {8, 8}, {1, 0}, {xla::Tile({4, 4})});
  std::vector<std::vector<raiden::RaidenBufferHandle>> buffers = {{*handle}};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> ws,
      WeightSynchronizerBase::Create(
          buffers, /*local_port=*/0, /*external_host_ptrs=*/std::nullopt,
          /*unsafe_skip_buffer_lock=*/false, /*parallelism=*/1,
          /*listener_port=*/std::nullopt, /*bind_ip=*/std::nullopt,
          /*layer_names=*/std::vector<std::string>{}, /*auto_h2d=*/false));

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> host_buf, ws->GetHostBuffer(0, 0));
  float* host = reinterpret_cast<float*>(host_buf.data());
  for (int i = 0; i < 64; ++i) {
    host[i] = static_cast<float>(i);
  }

  tpu_sync::rpc::StartTransferRequest req;
  req.set_host_tiling_mode(tpu_sync::rpc::HOST_TILING_MODE_ON_ARRIVAL);
  constexpr uint64_t kFirstUuid = 880;
  ws->StoreSkipTiling(kFirstUuid, req);
  ASSERT_OK(ws->RegisterExpectedLayerChunks(kFirstUuid, {{0, 1}}));

  // The data has landed (and been acked) but OnLayerDataReceived has not run
  // yet, so H2d must block.
  absl::Notification h2d_done;
  absl::Status h2d_status;
  std::thread h2d_caller([&] {
    absl::StatusOr<raiden::PjRtCopyFuture> h2d_fut = ws->H2d(kFirstUuid);
    h2d_status = h2d_fut.ok() ? h2d_fut->Await() : h2d_fut.status();
    h2d_done.Notify();
  });
  EXPECT_FALSE(h2d_done.WaitForNotificationWithTimeout(absl::Seconds(1)));

  EXPECT_OK(ws->OnLayerDataReceived(/*layer_idx=*/0, kFirstUuid));
  EXPECT_OK(ws->OnDataReceived(kFirstUuid));
  h2d_caller.join();
  ASSERT_OK(h2d_status);
  // Tiled exactly once.
  std::vector<float> dev_raw(64, -1.0f);
  ASSERT_OK(handle->CopyRawDeviceToHost(dev_raw.data(), 0, 64 * sizeof(float))
                .Await());
  EXPECT_EQ(dev_raw[4], 8.0f);
  EXPECT_EQ(dev_raw[16], 4.0f);

  // Arming the next receive invalidates the previous tiled state; dropping it
  // before the layer arrives fails a pending H2d instead of hanging.
  constexpr uint64_t kSecondUuid = 881;
  ws->StoreSkipTiling(kSecondUuid, req);
  ASSERT_OK(ws->RegisterExpectedLayerChunks(kSecondUuid, {{0, 1}}));
  ws->ForgetPushProgress(kSecondUuid);
  absl::StatusOr<raiden::PjRtCopyFuture> h2d_fut = ws->H2d(kSecondUuid);
  EXPECT_EQ(h2d_fut.status().code(), absl::StatusCode::kCancelled);
  ws->ForgetPushProgress(kFirstUuid);
}

// SHUTDOWN runs DrainPendingH2d() on every receiver. A receiver that was armed
// with an in-place tiling mode but whose data never arrived (sender died,
// partial transfer) must be released by the drain: H2d() fails instead of
// blocking on the layer's readiness signal and WaitForTransferCompletion(uuid)
// returns. Synthetic reproduction; no transport or controller involved.
TEST_F(WeightSynchronizerTest, DrainPendingH2dReleasesArmedLayers) {
  auto client_status_or = xla::GetXlaPjrtCpuClient(xla::CpuClientOptions());
  ASSERT_TRUE(client_status_or.ok()) << client_status_or.status().message();
  std::unique_ptr<xla::PjRtClient> client = *std::move(client_status_or);
  absl::StatusOr<xla::PjRtMemorySpace*> memory_space =
      client->addressable_devices()[0]->default_memory_space();
  ASSERT_TRUE(memory_space.ok()) << memory_space.status().message();

  for (bool auto_h2d : {true, false}) {
    SCOPED_TRACE(absl::StrCat("auto_h2d=", auto_h2d));
    std::vector<float> placeholder(64, 0.0f);
    absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> pjrt_buffer =
        client->BufferFromHostBuffer(
            placeholder.data(), xla::PrimitiveType::F32, {8, 8},
            /*byte_strides=*/std::nullopt,
            xla::PjRtClient::HostBufferSemantics::
                kImmutableUntilTransferCompletes,
            /*on_done_with_host_buffer=*/nullptr, *memory_space,
            /*device_layout=*/nullptr);
    ASSERT_TRUE(pjrt_buffer.ok()) << pjrt_buffer.status().message();
    absl::StatusOr<raiden::RaidenBufferHandle> handle =
        raiden::RaidenBufferHandle::Acquire(pjrt_buffer->get());
    ASSERT_TRUE(handle.ok()) << handle.status().message();
    handle->shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::F32, {8, 8}, {1, 0}, {xla::Tile({4, 4})});
    std::vector<std::vector<raiden::RaidenBufferHandle>> buffers = {{*handle}};

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<WeightSynchronizerBase> ws,
        WeightSynchronizerBase::Create(
            buffers, /*local_port=*/0, /*external_host_ptrs=*/std::nullopt,
            /*unsafe_skip_buffer_lock=*/false, /*parallelism=*/1,
            /*listener_port=*/std::nullopt, /*bind_ip=*/std::nullopt,
            /*layer_names=*/std::vector<std::string>{}, auto_h2d));

    // The controller arms the receiver (START_TRANSFER with is_sender=false),
    // but no chunk ever lands, so OnLayerDataReceived/OnDataReceived never run.
    constexpr uint64_t kUuid = 883;
    tpu_sync::rpc::StartTransferRequest req;
    req.set_host_tiling_mode(tpu_sync::rpc::HOST_TILING_MODE_ON_ARRIVAL);
    ws->StoreSkipTiling(kUuid, req);
    ASSERT_OK(ws->RegisterExpectedLayerChunks(kUuid, {{0, 1}}));

    // SHUTDOWN.
    ws->DrainPendingH2d();

    // What the benchmark client does next, each on its own thread so that a
    // hang shows up as a failure instead of wedging the test.
    absl::Notification wait_done;
    absl::Status wait_status;
    std::thread wait_caller([&] {
      wait_status = ws->WaitForTransferCompletion(kUuid);
      wait_done.Notify();
    });
    absl::Notification h2d_done;
    absl::Status h2d_status;
    std::thread h2d_caller([&] {
      absl::StatusOr<raiden::PjRtCopyFuture> h2d_fut = ws->H2d(kUuid);
      h2d_status = h2d_fut.ok() ? h2d_fut->Await() : h2d_fut.status();
      h2d_done.Notify();
    });

    const bool wait_returned =
        wait_done.WaitForNotificationWithTimeout(absl::Seconds(3));
    const bool h2d_returned =
        h2d_done.WaitForNotificationWithTimeout(absl::Seconds(3));
    EXPECT_TRUE(wait_returned)
        << "WaitForTransferCompletion(uuid) is still blocked 3s after "
           "DrainPendingH2d()";
    EXPECT_TRUE(h2d_returned)
        << "H2d() is still blocked on the armed layer 3s after "
           "DrainPendingH2d()";
    if (wait_returned) {
      EXPECT_OK(wait_status);
    }
    if (h2d_returned) {
      EXPECT_EQ(h2d_status.code(), absl::StatusCode::kCancelled) << h2d_status;
    }

    // Unblock whatever the drain left behind so the threads can be joined.
    if (!h2d_returned) {
      ws->ForgetPushProgress(kUuid);  // Cancels the armed signal.
    }
    h2d_caller.join();
    if (!wait_returned) {
      // Pretend the missing layer arrived after all.
      EXPECT_OK(ws->OnLayerDataReceived(/*layer_idx=*/0, kUuid));
      EXPECT_OK(ws->OnDataReceived(kUuid));
    }
    wait_caller.join();
    ws->ForgetPushProgress(kUuid);
  }
}

// A relay receiver gets bytes that the seed already tiled: it must neither
// tile them again on receive nor at H2D.
TEST_F(WeightSynchronizerTest, PreTiledReceiverCopiesHostBufferAsIs) {
  auto client_status_or = xla::GetXlaPjrtCpuClient(xla::CpuClientOptions());
  ASSERT_TRUE(client_status_or.ok()) << client_status_or.status().message();
  std::unique_ptr<xla::PjRtClient> client = *std::move(client_status_or);
  absl::StatusOr<xla::PjRtMemorySpace*> memory_space =
      client->addressable_devices()[0]->default_memory_space();
  ASSERT_TRUE(memory_space.ok()) << memory_space.status().message();

  for (bool auto_h2d : {false, true}) {
    SCOPED_TRACE(absl::StrCat("auto_h2d=", auto_h2d));
    std::vector<float> placeholder(64, 0.0f);
    absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> pjrt_buffer =
        client->BufferFromHostBuffer(
            placeholder.data(), xla::PrimitiveType::F32, {8, 8},
            /*byte_strides=*/std::nullopt,
            xla::PjRtClient::HostBufferSemantics::
                kImmutableUntilTransferCompletes,
            /*on_done_with_host_buffer=*/nullptr, *memory_space,
            /*device_layout=*/nullptr);
    ASSERT_TRUE(pjrt_buffer.ok()) << pjrt_buffer.status().message();
    absl::StatusOr<raiden::RaidenBufferHandle> handle =
        raiden::RaidenBufferHandle::Acquire(pjrt_buffer->get());
    ASSERT_TRUE(handle.ok()) << handle.status().message();
    handle->shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::F32, {8, 8}, {1, 0}, {xla::Tile({4, 4})});
    std::vector<std::vector<raiden::RaidenBufferHandle>> buffers = {{*handle}};

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<WeightSynchronizerBase> ws,
        WeightSynchronizerBase::Create(
            buffers, /*local_port=*/0, /*external_host_ptrs=*/std::nullopt,
            /*unsafe_skip_buffer_lock=*/false, /*parallelism=*/1,
            /*listener_port=*/std::nullopt, /*bind_ip=*/std::nullopt,
            /*layer_names=*/std::vector<std::string>{}, auto_h2d));

    // The bytes as relayed: already in device layout.
    TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> host_buf, ws->GetHostBuffer(0, 0));
    float* host = reinterpret_cast<float*>(host_buf.data());
    for (int i = 0; i < 64; ++i) {
      host[i] = static_cast<float>(i);
    }

    constexpr uint64_t kUuid = 882;
    tpu_sync::rpc::StartTransferRequest req;
    req.set_host_tiling_mode(tpu_sync::rpc::HOST_TILING_MODE_PRE_TILED);
    ws->StoreSkipTiling(kUuid, req);
    ASSERT_OK(ws->RegisterExpectedLayerChunks(kUuid, {{0, 1}}));
    ASSERT_OK(ws->OnLayerDataReceived(/*layer_idx=*/0, kUuid));
    ASSERT_OK(ws->OnDataReceived(kUuid));
    for (int i = 0; i < 64; ++i) {
      EXPECT_EQ(host[i], static_cast<float>(i)) << "Host mismatch at " << i;
    }

    if (!auto_h2d) {
      absl::StatusOr<raiden::PjRtCopyFuture> h2d_fut = ws->H2d(kUuid);
      ASSERT_TRUE(h2d_fut.ok()) << h2d_fut.status().message();
      ASSERT_OK(h2d_fut->Await());
    }
    std::vector<float> dev_raw(64, -1.0f);
    ASSERT_OK(handle->CopyRawDeviceToHost(dev_raw.data(), 0, 64 * sizeof(float))
                  .Await());
    for (int i = 0; i < 64; ++i) {
      EXPECT_EQ(dev_raw[i], static_cast<float>(i))
          << "Device mismatch at " << i;
    }
    ws->ForgetPushProgress(kUuid);
  }
}

// A tree broadcast re-arms each receiver with the same uuid before every
// pipeline stage. Re-arming must not discard the readiness of layers that have
// already arrived, or a later H2d would wait for them forever.
TEST_F(WeightSynchronizerTest, ReArmKeepsReceivedLayersReady) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> ws,
      WeightSynchronizerBase::Create(
          /*num_layers=*/2, /*num_shards=*/1, slice_byte_size_,
          /*local_port=*/0, /*host_blocks_to_allocate=*/1,
          /*parallelism=*/2, /*listener_port=*/0, /*bind_ip=*/std::nullopt,
          /*layer_names=*/std::vector<std::string>{"layer_0", "layer_1"},
          /*auto_h2d=*/false));

  constexpr uint64_t kUuid = 883;
  tpu_sync::rpc::StartTransferRequest req;
  req.set_host_tiling_mode(tpu_sync::rpc::HOST_TILING_MODE_ON_ARRIVAL);
  ws->StoreSkipTiling(kUuid, req);
  const absl::flat_hash_map<size_t, uint32_t> counts = {{0, 1}, {1, 1}};
  ASSERT_OK(ws->RegisterExpectedLayerChunks(kUuid, counts));
  ASSERT_OK(ws->OnLayerDataReceived(/*layer_idx=*/0, kUuid));
  // Second stage arm.
  ASSERT_OK(ws->RegisterExpectedLayerChunks(kUuid, counts));
  ASSERT_OK(ws->OnLayerDataReceived(/*layer_idx=*/1, kUuid));
  ASSERT_OK(ws->OnDataReceived(kUuid));

  absl::StatusOr<raiden::PjRtCopyFuture> h2d_fut = ws->H2d(kUuid);
  ASSERT_TRUE(h2d_fut.ok()) << h2d_fut.status().message();
  ASSERT_OK(h2d_fut->Await());
  ws->ForgetPushProgress(kUuid);
}

// A seed relays each received shard as is once it has been tiled in place:
// the whole device-size shard, whatever span the schedule entry names, and
// only after the layer's in-place tiling has finished.
TEST_F(WeightSynchronizerTest, PreTiledPushSendsWholeShardAfterTiling) {
  constexpr size_t kSliceBytes = 16384;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_seed,
                          WeightSynchronizerBase::Create(
                              /*num_layers=*/1, /*num_shards=*/1, kSliceBytes,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_relay,
                          WeightSynchronizerBase::Create(
                              /*num_layers=*/1, /*num_shards=*/1, kSliceBytes,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  ASSERT_TRUE(ws_relay->local_port().has_value());
  const std::string relay_peer =
      absl::StrCat("localhost:", *ws_relay->local_port());

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> seed_buf,
                       ws_seed->GetHostBuffer(0, 0));
  uint8_t* seed_host = seed_buf.data();
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> relay_buf,
                       ws_relay->GetHostBuffer(0, 0));
  uint8_t* relay_host = relay_buf.data();
  std::memset(seed_host, 0xAB, kSliceBytes);
  std::memset(relay_host, 0x00, kSliceBytes);

  constexpr uint64_t kUuid = 884;
  // The seed is still receiving (and tiling) layer 0 from the trainer.
  tpu_sync::rpc::StartTransferRequest recv_req;
  recv_req.set_host_tiling_mode(tpu_sync::rpc::HOST_TILING_MODE_ON_ARRIVAL);
  ws_seed->StoreSkipTiling(kUuid, recv_req);
  ASSERT_OK(ws_seed->RegisterExpectedLayerChunks(kUuid, {{0, 1}}));

  tpu_sync::rpc::StartTransferRequest push_req;
  push_req.set_is_sender(true);
  push_req.set_skip_d2h(true);
  push_req.set_uuid(kUuid);
  push_req.set_host_tiling_mode(tpu_sync::rpc::HOST_TILING_MODE_PRE_TILED);
  auto* entry = (*push_req.mutable_shard_push_schedules())[0].add_entries();
  entry->set_dst_peer(relay_peer);
  entry->set_dst_shard_idx(0);
  // A logical span shorter than the shard.
  entry->set_src_offset_bytes(64);
  entry->set_dst_offset_bytes(64);
  entry->set_size_bytes(128);
  entry->set_count(1);
  entry->set_layer_idx(0);

  ASSERT_OK(ws_relay->RegisterExpectedChunks(kUuid, 1));
  absl::Notification push_done;
  absl::Status push_status;
  std::thread pusher([&] {
    push_status = ws_seed->PushWeightsResharded(push_req);
    push_done.Notify();
  });
  EXPECT_FALSE(push_done.WaitForNotificationWithTimeout(absl::Seconds(1)));

  ASSERT_OK(ws_seed->OnLayerDataReceived(/*layer_idx=*/0, kUuid));
  pusher.join();
  ASSERT_OK(push_status);
  ASSERT_OK(ws_relay->WaitForTransferCompletion(kUuid));
  for (size_t i = 0; i < kSliceBytes; ++i) {
    ASSERT_EQ(relay_host[i], 0xAB) << "Mismatch at byte " << i;
  }
  ws_seed->ForgetPushProgress(kUuid);
  ws_relay->ForgetPushProgress(kUuid);
}

TEST_F(WeightSynchronizerTest, GetHostBufferNonContiguousGlobalShards) {
  const size_t num_layers = 2;
  const size_t num_shards = 4;
  const size_t slice_size = 1024;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  // Assign distinct per-slot byte sizes so wrong-slot or wrong-layer
  // resolution fails on `.size()` as well as `.data()`.
  std::vector<const uint8_t*> slot_ptrs;
  std::vector<size_t> slot_sizes;
  for (size_t l = 0; l < num_layers; ++l) {
    for (size_t s = 0; s < num_shards; ++s) {
      slot_ptrs.push_back(ws->GetHostBufferPtr(l, s));
      slot_sizes.push_back((l * num_shards + s + 1) * 128);
    }
  }
  ws->SetExternalHostPointers(slot_ptrs, slot_sizes);

  // Configure non-contiguous global shards, e.g., NUMA node 0 managing shards
  // {0, 2, 4, 6}
  ws->SetGlobalShardIndices({0, 2, 4, 6});

  // Check that GetHostBuffer correctly maps global shard indices to internal
  // slots 0, 1, 2, 3
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> shard0_buf, ws->GetHostBuffer(0, 0));
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> shard2_buf, ws->GetHostBuffer(0, 2));
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> shard4_buf, ws->GetHostBuffer(0, 4));
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> shard6_buf, ws->GetHostBuffer(0, 6));
  uint8_t* ptr_shard0 = shard0_buf.data();
  uint8_t* ptr_shard2 = shard2_buf.data();
  uint8_t* ptr_shard4 = shard4_buf.data();
  uint8_t* ptr_shard6 = shard6_buf.data();

  EXPECT_EQ(ptr_shard0, ws->GetHostBufferPtr(0, 0));
  EXPECT_EQ(ptr_shard2, ws->GetHostBufferPtr(0, 1));
  EXPECT_EQ(ptr_shard4, ws->GetHostBufferPtr(0, 2));
  EXPECT_EQ(ptr_shard6, ws->GetHostBufferPtr(0, 3));

  // Ensure all 4 resolved pointers are distinct
  EXPECT_NE(ptr_shard0, ptr_shard2);
  EXPECT_NE(ptr_shard0, ptr_shard4);
  EXPECT_NE(ptr_shard0, ptr_shard6);
  EXPECT_NE(ptr_shard2, ptr_shard4);
  EXPECT_NE(ptr_shard2, ptr_shard6);
  EXPECT_NE(ptr_shard4, ptr_shard6);

  // Check distinct per-slot sizes
  EXPECT_EQ(shard0_buf.size(), 128);
  EXPECT_EQ(shard2_buf.size(), 256);
  EXPECT_EQ(shard4_buf.size(), 384);
  EXPECT_EQ(shard6_buf.size(), 512);

  // Const overloads
  const WeightSynchronizerBase* const_ws = ws.get();
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> const_shard0_buf,
                       const_ws->GetHostBuffer(0, 0));
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> const_shard2_buf,
                       const_ws->GetHostBuffer(0, 2));
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> const_shard4_buf,
                       const_ws->GetHostBuffer(0, 4));
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> const_shard6_buf,
                       const_ws->GetHostBuffer(0, 6));
  EXPECT_EQ(const_shard0_buf.data(), ws->GetHostBufferPtr(0, 0));
  EXPECT_EQ(const_shard2_buf.data(), ws->GetHostBufferPtr(0, 1));
  EXPECT_EQ(const_shard4_buf.data(), ws->GetHostBufferPtr(0, 2));
  EXPECT_EQ(const_shard6_buf.data(), ws->GetHostBufferPtr(0, 3));
  EXPECT_EQ(const_shard0_buf.size(), 128);
  EXPECT_EQ(const_shard2_buf.size(), 256);
  EXPECT_EQ(const_shard4_buf.size(), 384);
  EXPECT_EQ(const_shard6_buf.size(), 512);

  // Verify memory writes do not collide
  std::memset(ptr_shard0, 0x11, shard0_buf.size());
  std::memset(ptr_shard2, 0x22, shard2_buf.size());
  std::memset(ptr_shard4, 0x33, shard4_buf.size());
  std::memset(ptr_shard6, 0x44, shard6_buf.size());

  EXPECT_EQ(ws->GetHostBufferPtr(0, 0)[0], 0x11);
  EXPECT_EQ(ws->GetHostBufferPtr(0, 1)[0], 0x22);
  EXPECT_EQ(ws->GetHostBufferPtr(0, 2)[0], 0x33);
  EXPECT_EQ(ws->GetHostBufferPtr(0, 3)[0], 0x44);
}

TEST_F(WeightSynchronizerTest, GetHostBufferLocalAndGlobalIndices) {
  const size_t num_layers = 2;
  const size_t num_shards = 4;
  const size_t slice_size = 1024;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  // Assign distinct per-slot byte sizes so wrong-slot or wrong-layer
  // resolution fails on `.size()` as well as `.data()`.
  std::vector<const uint8_t*> slot_ptrs;
  std::vector<size_t> slot_sizes;
  for (size_t l = 0; l < num_layers; ++l) {
    for (size_t s = 0; s < num_shards; ++s) {
      slot_ptrs.push_back(ws->GetHostBufferPtr(l, s));
      slot_sizes.push_back((l * num_shards + s + 1) * 128);
    }
  }
  ws->SetExternalHostPointers(slot_ptrs, slot_sizes);

  // Configure non-contiguous global shards, e.g., Host 1 managing global shards
  // {2, 3, 6, 7} with local slots {0, 1, 2, 3}.
  ws->SetGlobalShardIndices({2, 3, 6, 7});
  ws->SetLocalShardIndices({0, 1, 2, 3});

  // Local slot indices should map directly to internal slots 0, 1, 2, 3
  // without being falsely remapped by global shard indices {2, 3, 6, 7}.
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> slot0_buf, ws->GetHostBuffer(0, 0));
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> slot1_buf, ws->GetHostBuffer(0, 1));
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> slot2_buf, ws->GetHostBuffer(0, 2));
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> slot3_buf, ws->GetHostBuffer(0, 3));
  EXPECT_EQ(slot0_buf.data(), ws->GetHostBufferPtr(0, 0));
  EXPECT_EQ(slot1_buf.data(), ws->GetHostBufferPtr(0, 1));
  EXPECT_EQ(slot2_buf.data(), ws->GetHostBufferPtr(0, 2));
  EXPECT_EQ(slot3_buf.data(), ws->GetHostBufferPtr(0, 3));
  EXPECT_EQ(slot0_buf.size(), 128);
  EXPECT_EQ(slot1_buf.size(), 256);
  EXPECT_EQ(slot2_buf.size(), 384);
  EXPECT_EQ(slot3_buf.size(), 512);

  // Global shard indices not matching local slots (e.g. 6 and 7) should still
  // map to their corresponding internal slots (slots 2 and 3).
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> global6_buf,
                       ws->GetHostBuffer(0, 6));
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> global7_buf,
                       ws->GetHostBuffer(0, 7));
  EXPECT_EQ(global6_buf.data(), ws->GetHostBufferPtr(0, 2));
  EXPECT_EQ(global7_buf.data(), ws->GetHostBufferPtr(0, 3));
  EXPECT_EQ(global6_buf.size(), 384);
  EXPECT_EQ(global7_buf.size(), 512);

  // Const overloads
  const WeightSynchronizerBase* const_ws = ws.get();
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> const_slot2_buf,
                       const_ws->GetHostBuffer(0, 2));
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> const_slot3_buf,
                       const_ws->GetHostBuffer(0, 3));
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> const_global6_buf,
                       const_ws->GetHostBuffer(1, 6));
  EXPECT_EQ(const_slot2_buf.data(), ws->GetHostBufferPtr(0, 2));
  EXPECT_EQ(const_slot3_buf.data(), ws->GetHostBufferPtr(0, 3));
  EXPECT_EQ(const_global6_buf.data(), ws->GetHostBufferPtr(1, 2));
  EXPECT_EQ(const_slot2_buf.size(), 384);
  EXPECT_EQ(const_slot3_buf.size(), 512);
  EXPECT_EQ(const_global6_buf.size(), 896);
}

TEST_F(WeightSynchronizerTest, GetHostBufferMapsGlobalShardIndices) {
  const size_t num_layers = 2;
  const size_t num_shards = 4;
  const size_t slice_size = 1024;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws,
                       WeightSynchronizerBase::Create(
                           num_layers, num_shards, slice_size,
                           /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  std::vector<const uint8_t*> slot_ptrs;
  std::vector<size_t> slot_sizes;
  for (size_t l = 0; l < num_layers; ++l) {
    for (size_t s = 0; s < num_shards; ++s) {
      slot_ptrs.push_back(ws->GetHostBufferPtr(l, s));
      slot_sizes.push_back((l * num_shards + s + 1) * 128);
    }
  }
  ws->SetExternalHostPointers(slot_ptrs, slot_sizes);
  ws->SetGlobalShardIndices({0, 2, 4, 6});
  const WeightSynchronizerBase& const_ws = *ws;

  // Global shard 6 is local slot 3 (modulo 4 would give slot 2).
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> buffer, ws->GetHostBuffer(1, 6));
  EXPECT_EQ(buffer.data(), ws->GetHostBufferPtr(1, 3));
  EXPECT_EQ(buffer.size(), 1024);
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> const_buffer,
                       const_ws.GetHostBuffer(1, 2));
  EXPECT_EQ(const_buffer.data(), ws->GetHostBufferPtr(1, 1));
  EXPECT_EQ(const_buffer.size(), 768);
}

TEST_F(WeightSynchronizerTest, GetHostBufferMapsLocalShardIndices) {
  const size_t num_layers = 1;
  const size_t num_shards = 4;
  const size_t slice_size = 1024;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws,
                       WeightSynchronizerBase::Create(
                           num_layers, num_shards, slice_size,
                           /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  ws->SetExternalHostPointers(
      {ws->GetHostBufferPtr(0, 0), ws->GetHostBufferPtr(0, 1),
       ws->GetHostBufferPtr(0, 2), ws->GetHostBufferPtr(0, 3)},
      {128, 256, 384, 512});
  ws->SetLocalShardIndices({3, 2, 1, 0});
  const WeightSynchronizerBase& const_ws = *ws;

  // Shard 3 is listed at local slot 0 (modulo 4 would give slot 3).
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> buffer, ws->GetHostBuffer(0, 3));
  EXPECT_EQ(buffer.data(), ws->GetHostBufferPtr(0, 0));
  EXPECT_EQ(buffer.size(), 128);
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> const_buffer,
                       const_ws.GetHostBuffer(0, 0));
  EXPECT_EQ(const_buffer.data(), ws->GetHostBufferPtr(0, 3));
  EXPECT_EQ(const_buffer.size(), 512);
}

TEST_F(WeightSynchronizerTest, GetHostBufferLayerOutOfRange) {
  const size_t num_layers = 2;
  const size_t num_shards = 4;
  const size_t slice_size = 1024;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws,
                       WeightSynchronizerBase::Create(
                           num_layers, num_shards, slice_size,
                           /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  ws->SetGlobalShardIndices({0, 2, 4, 6});

  ABSL_EXPECT_OK(ws->GetHostBuffer(1, 6));
  EXPECT_THAT(ws->GetHostBuffer(2, 6),
              ::absl_testing::StatusIs(
                  absl::StatusCode::kOutOfRange,
                  ::testing::HasSubstr("layer_idx 2 out of range")));
}

TEST_F(WeightSynchronizerTest, GetHostBufferMissingBufferIsFailedPrecondition) {
  const size_t num_layers = 1;
  const size_t num_shards = 2;
  const size_t slice_size = 1024;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws,
                       WeightSynchronizerBase::Create(
                           num_layers, num_shards, slice_size,
                           /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  ws->SetGlobalShardIndices({4, 6});
  // Local slot 0 (global shard 4) loses its host buffer; slot 1 keeps its own.
  ws->SetExternalHostPointers({nullptr, ws->GetHostBufferPtr(0, 1)},
                              {slice_size, slice_size});
  const WeightSynchronizerBase& const_ws = *ws;

  ABSL_EXPECT_OK(ws->GetHostBuffer(0, 6));
  EXPECT_THAT(ws->GetHostBuffer(0, 4),
              ::absl_testing::StatusIs(
                  absl::StatusCode::kFailedPrecondition,
                  ::testing::HasSubstr("layer 0 shard 4 has no host buffer")));
  EXPECT_THAT(const_ws.GetHostBuffer(0, 4),
              ::absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(WeightSynchronizerTest,
       GetHostBufferGlobalShardIndexOutOfBoundsLocalSlot) {
  const size_t num_layers = 1;
  const size_t num_shards = 4;
  const size_t slice_size = 1024;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws,
                       WeightSynchronizerBase::Create(
                           num_layers, num_shards, slice_size,
                           /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  ws->SetGlobalShardIndices({0, 1, 2, 3, 4, 5, 6, 7});
  const WeightSynchronizerBase& const_ws = *ws;

  ABSL_EXPECT_OK(ws->GetHostBuffer(0, 3));
  EXPECT_THAT(ws->GetHostBuffer(0, 7),
              ::absl_testing::StatusIs(
                  absl::StatusCode::kOutOfRange,
                  ::testing::AllOf(::testing::HasSubstr("local slot 7"),
                                   ::testing::HasSubstr("4 local shards"))));
  EXPECT_THAT(const_ws.GetHostBuffer(0, 7),
              ::absl_testing::StatusIs(
                  absl::StatusCode::kOutOfRange,
                  ::testing::AllOf(::testing::HasSubstr("local slot 7"),
                                   ::testing::HasSubstr("4 local shards"))));
}

TEST_F(WeightSynchronizerTest,
       GetHostBufferLocalShardIndexOutOfBoundsLocalSlot) {
  const size_t num_layers = 1;
  const size_t num_shards = 4;
  const size_t slice_size = 1024;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws,
                       WeightSynchronizerBase::Create(
                           num_layers, num_shards, slice_size,
                           /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  ws->SetLocalShardIndices({0, 1, 2, 3, 4, 5, 6, 7});
  const WeightSynchronizerBase& const_ws = *ws;

  ABSL_EXPECT_OK(ws->GetHostBuffer(0, 3));
  EXPECT_THAT(ws->GetHostBuffer(0, 7),
              ::absl_testing::StatusIs(
                  absl::StatusCode::kOutOfRange,
                  ::testing::AllOf(::testing::HasSubstr("local slot 7"),
                                   ::testing::HasSubstr("4 local shards"))));
  EXPECT_THAT(const_ws.GetHostBuffer(0, 7),
              ::absl_testing::StatusIs(
                  absl::StatusCode::kOutOfRange,
                  ::testing::AllOf(::testing::HasSubstr("local slot 7"),
                                   ::testing::HasSubstr("4 local shards"))));
}

TEST_F(WeightSynchronizerTest,
       GetHostBufferEmptyLocalIndicesIdentityAndUnownedFailClosed) {
  const size_t num_layers = 1;
  const size_t num_shards = 2;
  const size_t slice_size = 1024;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws,
                       WeightSynchronizerBase::Create(
                           num_layers, num_shards, slice_size,
                           /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  const uint8_t* ptr0 = ws->GetHostBufferPtr(0, 0);
  const uint8_t* ptr1 = ws->GetHostBufferPtr(0, 1);
  ws->SetExternalHostPointers({ptr0, ptr1}, {128, 256});
  ws->SetGlobalShardIndices({4, 6});
  const WeightSynchronizerBase& const_ws = *ws;

  // Empty local_shard_indices_ preserves identity [0, 2) alongside {4, 6}.
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> local0, ws->GetHostBuffer(0, 0));
  EXPECT_EQ(local0.data(), ptr0);
  EXPECT_EQ(local0.size(), 128);
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> local1, ws->GetHostBuffer(0, 1));
  EXPECT_EQ(local1.data(), ptr1);
  EXPECT_EQ(local1.size(), 256);
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> global4, ws->GetHostBuffer(0, 4));
  EXPECT_EQ(global4.data(), ptr0);
  EXPECT_EQ(global4.size(), 128);
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> global6,
                       const_ws.GetHostBuffer(0, 6));
  EXPECT_EQ(global6.data(), ptr1);
  EXPECT_EQ(global6.size(), 256);

  // Unowned shards outside [0, 2) and {4, 6} fail closed instead of wrapping.
  for (size_t unowned : {size_t{2}, size_t{5}, size_t{1} << 32, SIZE_MAX}) {
    EXPECT_THAT(ws->GetHostBuffer(0, unowned),
                ::absl_testing::StatusIs(absl::StatusCode::kOutOfRange,
                                         ::testing::HasSubstr("not mapped")));
  }
}

TEST_F(WeightSynchronizerTest,
       GetHostBufferExplicitLocalIndicesFailClosedOnUnowned) {
  const size_t num_layers = 1;
  const size_t num_shards = 2;
  const size_t slice_size = 1024;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws,
                       WeightSynchronizerBase::Create(
                           num_layers, num_shards, slice_size,
                           /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  const uint8_t* ptr0 = ws->GetHostBufferPtr(0, 0);
  const uint8_t* ptr1 = ws->GetHostBufferPtr(0, 1);
  ws->SetExternalHostPointers({ptr0, ptr1}, {128, 256});
  ws->SetLocalShardIndices({2, 3});

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> local2, ws->GetHostBuffer(0, 2));
  EXPECT_EQ(local2.data(), ptr0);
  EXPECT_EQ(local2.size(), 128);
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> local3, ws->GetHostBuffer(0, 3));
  EXPECT_EQ(local3.data(), ptr1);
  EXPECT_EQ(local3.size(), 256);

  for (size_t unowned : {size_t{0}, size_t{1}, size_t{5}}) {
    EXPECT_THAT(ws->GetHostBuffer(0, unowned),
                ::absl_testing::StatusIs(absl::StatusCode::kOutOfRange,
                                         ::testing::HasSubstr("not mapped")));
  }

  ws->SetGlobalShardIndices({4, 6});
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> global4, ws->GetHostBuffer(0, 4));
  EXPECT_EQ(global4.data(), ptr0);
  EXPECT_EQ(global4.size(), 128);
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> global6, ws->GetHostBuffer(0, 6));
  EXPECT_EQ(global6.data(), ptr1);
  EXPECT_EQ(global6.size(), 256);
  for (size_t unowned : {size_t{0}, size_t{1}, size_t{5}}) {
    EXPECT_THAT(ws->GetHostBuffer(0, unowned),
                ::absl_testing::StatusIs(absl::StatusCode::kOutOfRange,
                                         ::testing::HasSubstr("not mapped")));
  }
}

TEST_F(WeightSynchronizerTest,
       GetHostBufferRejectsNegativeAndTruncatedShardIndices) {
  const size_t num_layers = 1;
  const size_t num_shards = 2;
  const size_t slice_size = 1024;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws,
                       WeightSynchronizerBase::Create(
                           num_layers, num_shards, slice_size,
                           /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  const uint8_t* ptr0 = ws->GetHostBufferPtr(0, 0);
  const uint8_t* ptr1 = ws->GetHostBufferPtr(0, 1);
  ws->SetExternalHostPointers({ptr0, ptr1}, {128, 256});

  ws->SetLocalShardIndices({0, -1});
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> local0, ws->GetHostBuffer(0, 0));
  EXPECT_EQ(local0.data(), ptr0);
  EXPECT_EQ(local0.size(), 128);
  EXPECT_THAT(ws->GetHostBuffer(0, size_t{1} << 32),
              ::absl_testing::StatusIs(absl::StatusCode::kOutOfRange,
                                       ::testing::HasSubstr("not mapped")));
  EXPECT_THAT(ws->GetHostBuffer(0, SIZE_MAX),
              ::absl_testing::StatusIs(absl::StatusCode::kOutOfRange,
                                       ::testing::HasSubstr("not mapped")));

  ws->SetLocalShardIndices({});
  ws->SetGlobalShardIndices({0, -1});
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> global0, ws->GetHostBuffer(0, 0));
  EXPECT_EQ(global0.data(), ptr0);
  EXPECT_EQ(global0.size(), 128);
  EXPECT_THAT(ws->GetHostBuffer(0, size_t{1} << 32),
              ::absl_testing::StatusIs(absl::StatusCode::kOutOfRange,
                                       ::testing::HasSubstr("not mapped")));
  EXPECT_THAT(ws->GetHostBuffer(0, SIZE_MAX),
              ::absl_testing::StatusIs(absl::StatusCode::kOutOfRange,
                                       ::testing::HasSubstr("not mapped")));
}

TEST_F(WeightSynchronizerTest,
       PushWeightsReshardedGlobalShardIndicesPrioritized) {
  const size_t num_layers = 1;
  const size_t num_shards = 4;
  const size_t slice_byte_size = 256;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  // Global shards {2, 3, 6, 7} map to local slots {0, 1, 2, 3}.
  // Schedules are keyed by global shard indices.
  const std::vector<int64_t> global_shards = {2, 3, 6, 7};
  ws_source->SetGlobalShardIndices(global_shards);
  ws_source->SetLocalShardIndices({0, 1, 2, 3});
  ws_dest->SetGlobalShardIndices(global_shards);
  ws_dest->SetLocalShardIndices({0, 1, 2, 3});

  const uint8_t fill_bytes[4] = {0x11, 0x22, 0x33, 0x44};
  for (size_t s = 0; s < num_shards; ++s) {
    TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_buf,
                         ws_source->GetHostBuffer(0, s));
    uint8_t* src_ptr = src_buf.data();
    TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dst_buf,
                         ws_dest->GetHostBuffer(0, s));
    uint8_t* dst_ptr = dst_buf.data();
    std::memset(src_ptr, fill_bytes[s], slice_byte_size);
    std::memset(dst_ptr, 0x00, slice_byte_size);
  }

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(54321);

  auto* schedules = request.mutable_shard_push_schedules();
  for (size_t s = 0; s < num_shards; ++s) {
    auto* entry = (*schedules)[global_shards[s]].add_entries();
    entry->set_dst_peer(dest_peer);
    entry->set_dst_shard_idx(s);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(slice_byte_size);
    entry->set_count(1);
    entry->set_layer_idx(0);
  }

  ASSERT_OK(ws_dest->RegisterExpectedChunks(request.uuid(), num_shards));
  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
  ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));

  for (size_t s = 0; s < num_shards; ++s) {
    const uint8_t* dst_ptr = ws_dest->GetHostBufferPtr(0, s);
    ASSERT_NE(dst_ptr, nullptr);
    for (size_t b = 0; b < slice_byte_size; ++b) {
      EXPECT_EQ(dst_ptr[b], fill_bytes[s])
          << "Mismatch at slot " << s << " byte " << b;
    }
  }
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedLocalShardIndicesFallback) {
  const size_t num_layers = 1;
  const size_t num_shards = 4;
  const size_t slice_byte_size = 256;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  // Global shard indices are unassigned (-1), so schedules match local_shard.
  ws_source->SetGlobalShardIndices({-1, -1, -1, -1});
  ws_source->SetLocalShardIndices({0, 1, 2, 3});
  ws_dest->SetGlobalShardIndices({-1, -1, -1, -1});
  ws_dest->SetLocalShardIndices({0, 1, 2, 3});

  const uint8_t fill_bytes[4] = {0x11, 0x22, 0x33, 0x44};
  for (size_t s = 0; s < num_shards; ++s) {
    TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_buf,
                         ws_source->GetHostBuffer(0, s));
    uint8_t* src_ptr = src_buf.data();
    TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dst_buf,
                         ws_dest->GetHostBuffer(0, s));
    uint8_t* dst_ptr = dst_buf.data();
    std::memset(src_ptr, fill_bytes[s], slice_byte_size);
    std::memset(dst_ptr, 0x00, slice_byte_size);
  }

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(54322);

  auto* schedules = request.mutable_shard_push_schedules();
  for (size_t s = 0; s < num_shards; ++s) {
    auto* entry = (*schedules)[s].add_entries();
    entry->set_dst_peer(dest_peer);
    entry->set_dst_shard_idx(s);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(slice_byte_size);
    entry->set_count(1);
    entry->set_layer_idx(0);
  }

  ASSERT_OK(ws_dest->RegisterExpectedChunks(request.uuid(), num_shards));
  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
  ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));

  for (size_t s = 0; s < num_shards; ++s) {
    const uint8_t* dst_ptr = ws_dest->GetHostBufferPtr(0, s);
    ASSERT_NE(dst_ptr, nullptr);
    for (size_t b = 0; b < slice_byte_size; ++b) {
      EXPECT_EQ(dst_ptr[b], fill_bytes[s])
          << "Mismatch at slot " << s << " byte " << b;
    }
  }
}

TEST_F(WeightSynchronizerTest,
       PushWeightsReshardedLocalKeysWithoutExplicitLocalShardIndices) {
  // One shard per process with a non-zero global index and no explicit local
  // indices (empty local_shard_indices_ == identity). A schedule keyed by the
  // local index 0 must still be resolved rather than looked up by global key.
  const size_t num_layers = 1;
  const size_t num_shards = 1;
  const size_t slice_byte_size = 256;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  ws_source->SetGlobalShardIndices({3});
  ws_dest->SetGlobalShardIndices({3});

  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_buf,
                       ws_source->GetHostBuffer(0, 0));
  uint8_t* src_ptr = src_buf.data();
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dst_buf,
                       ws_dest->GetHostBuffer(0, 0));
  uint8_t* dst_ptr = dst_buf.data();
  std::memset(src_ptr, 0x5A, slice_byte_size);
  std::memset(dst_ptr, 0x00, slice_byte_size);

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(54324);
  auto* entry = (*request.mutable_shard_push_schedules())[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(slice_byte_size);
  entry->set_count(1);
  entry->set_layer_idx(0);

  ASSERT_OK(ws_dest->RegisterExpectedChunks(request.uuid(), 1));
  ASSERT_OK(ws_source->PushWeightsResharded(request));
  ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));

  const uint8_t* out = ws_dest->GetHostBufferPtr(0, 0);
  ASSERT_NE(out, nullptr);
  for (size_t b = 0; b < slice_byte_size; ++b) {
    EXPECT_EQ(out[b], 0x5A) << "Mismatch at byte " << b;
  }
}

TEST_F(WeightSynchronizerTest,
       PushWeightsReshardedLocalShardIndicesNotShadowedByGlobalIndices) {
  const size_t num_layers = 1;
  const size_t num_shards = 4;
  const size_t slice_byte_size = 256;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  // Host 1 of a 2x4 mesh with 2x2 host subgrid: global_shards = {2, 3, 6, 7}
  // and local_shards = {0, 1, 2, 3}. Notice that global shards 2 and 3 overlap
  // with local shard indices 2 and 3.
  // When RaidenController sends a StartTransferRequest keyed by local_shard
  // {0, 1, 2, 3}, slot 0 (global 2) must execute schedule[0] and NOT
  // schedule[2].
  ws_source->SetGlobalShardIndices({2, 3, 6, 7});
  ws_source->SetLocalShardIndices({0, 1, 2, 3});
  ws_dest->SetGlobalShardIndices({0, 1, 2, 3});
  ws_dest->SetLocalShardIndices({0, 1, 2, 3});

  const uint8_t fill_bytes[4] = {0x11, 0x22, 0x33, 0x44};
  for (size_t s = 0; s < num_shards; ++s) {
    uint8_t* src_ptr = const_cast<uint8_t*>(ws_source->GetHostBufferPtr(0, s));
    uint8_t* dst_ptr = const_cast<uint8_t*>(ws_dest->GetHostBufferPtr(0, s));
    ASSERT_NE(src_ptr, nullptr);
    ASSERT_NE(dst_ptr, nullptr);
    std::memset(src_ptr, fill_bytes[s], slice_byte_size);
    std::memset(dst_ptr, 0x00, slice_byte_size);
  }

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(54323);

  auto* schedules = request.mutable_shard_push_schedules();
  for (size_t s = 0; s < num_shards; ++s) {
    auto* entry = (*schedules)[s].add_entries();
    entry->set_dst_peer(dest_peer);
    entry->set_dst_shard_idx(s);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(slice_byte_size);
    entry->set_count(1);
    entry->set_layer_idx(0);
  }

  ASSERT_OK(ws_dest->RegisterExpectedChunks(request.uuid(), num_shards));
  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
  ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));

  for (size_t s = 0; s < num_shards; ++s) {
    const uint8_t* dst_ptr = ws_dest->GetHostBufferPtr(0, s);
    ASSERT_NE(dst_ptr, nullptr);
    for (size_t b = 0; b < slice_byte_size; ++b) {
      EXPECT_EQ(dst_ptr[b], fill_bytes[s])
          << "Mismatch at slot " << s << " byte " << b;
    }
  }
}

TEST_F(WeightSynchronizerTest,
       PushWeightsReshardedMultiHostGlobalScheduleKeys) {
  const size_t num_layers = 1;
  const size_t num_shards = 4;
  const size_t slice_byte_size = 256;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  // Multi-host cluster: Source host is host 1 out of 2, owning global shards
  // {4, 5, 6, 7} with local shard slots {0, 1, 2, 3}.
  // Controller sends the cluster-wide schedule containing keys 0..7.
  ws_source->SetGlobalShardIndices({4, 5, 6, 7});
  ws_source->SetLocalShardIndices({0, 1, 2, 3});
  ws_dest->SetGlobalShardIndices({0, 1, 2, 3});
  ws_dest->SetLocalShardIndices({0, 1, 2, 3});

  const uint8_t fill_bytes[4] = {0xAA, 0xBB, 0xCC, 0xDD};
  for (size_t s = 0; s < num_shards; ++s) {
    uint8_t* src_ptr = const_cast<uint8_t*>(ws_source->GetHostBufferPtr(0, s));
    uint8_t* dst_ptr = const_cast<uint8_t*>(ws_dest->GetHostBufferPtr(0, s));
    ASSERT_NE(src_ptr, nullptr);
    ASSERT_NE(dst_ptr, nullptr);
    std::memset(src_ptr, fill_bytes[s], slice_byte_size);
    std::memset(dst_ptr, 0x00, slice_byte_size);
  }

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(54324);

  auto* schedules = request.mutable_shard_push_schedules();
  // Populate schedules for all 8 cluster shards (0..7).
  // Global shards 0..3 belong to host 0; global shards 4..7 belong to host 1.
  for (size_t g = 0; g < 8; ++g) {
    auto* entry = (*schedules)[g].add_entries();
    entry->set_dst_peer(dest_peer);
    entry->set_dst_shard_idx(g % num_shards);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(slice_byte_size);
    entry->set_count(1);
    entry->set_layer_idx(0);
  }

  ASSERT_OK(ws_dest->RegisterExpectedChunks(request.uuid(), num_shards));
  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
  ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));

  for (size_t s = 0; s < num_shards; ++s) {
    const uint8_t* dst_ptr = ws_dest->GetHostBufferPtr(0, s);
    ASSERT_NE(dst_ptr, nullptr);
    for (size_t b = 0; b < slice_byte_size; ++b) {
      EXPECT_EQ(dst_ptr[b], fill_bytes[s])
          << "Mismatch at slot " << s << " byte " << b;
    }
  }
}

TEST_F(WeightSynchronizerTest,
       PreSlicedWorkerScheduleWithDefaultZeroBasedIndicesAndMultiStepUuid) {
  const size_t num_layers = 1;
  const size_t num_shards = 4;
  const size_t slice_byte_size = 256;

  // Source host is Worker 1 (owning cluster shards 4..7), but initialized with
  // default 0-based local/global shard indices {0, 1, 2, 3} (e.g. PyTorch or
  // default JAX WeightSynchronizer).
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);

  // Controller sends ONLY the pre-sliced schedule keys {4, 5, 6, 7} owned by
  // Worker 1.
  auto* schedules = request.mutable_shard_push_schedules();
  for (size_t s = 0; s < num_shards; ++s) {
    size_t sliced_key = 4 + s;
    auto* entry = (*schedules)[sliced_key].add_entries();
    entry->set_dst_peer(dest_peer);
    entry->set_dst_shard_idx(s);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(slice_byte_size);
    entry->set_count(1);
    entry->set_layer_idx(0);
  }

  // Execute two consecutive steady-state steps (uuid=60001, then uuid=60002)
  // to verify both pre-sliced key mapping and multi-step uuid synchronization.
  const uint8_t step_patterns[2][4] = {
      {0x11, 0x22, 0x33, 0x44},
      {0x55, 0x66, 0x77, 0x88},
  };
  const uint64_t step_uuids[2] = {60001, 60002};

  for (int step = 0; step < 2; ++step) {
    for (size_t s = 0; s < num_shards; ++s) {
      uint8_t* src_ptr =
          const_cast<uint8_t*>(ws_source->GetHostBufferPtr(0, s));
      uint8_t* dst_ptr = const_cast<uint8_t*>(ws_dest->GetHostBufferPtr(0, s));
      ASSERT_NE(src_ptr, nullptr);
      ASSERT_NE(dst_ptr, nullptr);
      std::memset(src_ptr, step_patterns[step][s], slice_byte_size);
      std::memset(dst_ptr, 0x00, slice_byte_size);
    }

    request.set_uuid(step_uuids[step]);
    ASSERT_OK(ws_dest->RegisterExpectedChunks(request.uuid(), num_shards));
    absl::Status status = ws_source->PushWeightsResharded(request);
    EXPECT_TRUE(status.ok()) << status.message();
    ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));

    for (size_t s = 0; s < num_shards; ++s) {
      const uint8_t* dst_ptr = ws_dest->GetHostBufferPtr(0, s);
      ASSERT_NE(dst_ptr, nullptr);
      for (size_t b = 0; b < slice_byte_size; ++b) {
        EXPECT_EQ(dst_ptr[b], step_patterns[step][s])
            << "Step " << step << " mismatch at slot " << s << " byte " << b;
      }
    }
  }
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedMetricsAccumulateAndReset) {
  size_t num_layers = 1;
  size_t num_shards = 1;
  size_t slice_byte_size = 4096;

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = absl::StrCat("127.0.0.1:", *ws_dest->local_port());

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(slice_byte_size);
  entry->set_count(1);
  entry->set_layer_idx(0);

  // Initial metrics should be zero.
  WeightSyncMetrics m0 = ws_source->GetMetrics();
  EXPECT_EQ(m0.total_h2h_bytes, 0);
  EXPECT_DOUBLE_EQ(m0.total_h2h_time_ms, 0.0);
  EXPECT_EQ(m0.push_resharded_call_count, 0);

  // Step 1: execute first push.
  request.set_uuid(70001);
  ASSERT_OK(ws_dest->RegisterExpectedChunks(request.uuid(), 1));
  ASSERT_OK(ws_source->PushWeightsResharded(request));
  ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));

  WeightSyncMetrics m1 = ws_source->GetMetrics();
  EXPECT_EQ(m1.total_h2h_bytes, slice_byte_size);
  EXPECT_GT(m1.total_h2h_time_ms, 0.0);
  EXPECT_EQ(m1.last_h2h_bytes, slice_byte_size);
  EXPECT_EQ(m1.push_resharded_call_count, 1);

  // Step 2: execute second push to verify accumulation.
  request.set_uuid(70002);
  ASSERT_OK(ws_dest->RegisterExpectedChunks(request.uuid(), 1));
  ASSERT_OK(ws_source->PushWeightsResharded(request));
  ASSERT_OK(ws_dest->WaitForTransferCompletion(request.uuid()));

  WeightSyncMetrics m2 = ws_source->GetMetrics();
  EXPECT_EQ(m2.total_h2h_bytes, 2 * slice_byte_size);
  EXPECT_GE(m2.total_h2h_time_ms, m1.total_h2h_time_ms);
  EXPECT_EQ(m2.last_h2h_bytes, slice_byte_size);
  EXPECT_EQ(m2.push_resharded_call_count, 2);

  // Step 3: verify reset.
  ws_source->ResetMetrics();
  WeightSyncMetrics m_reset = ws_source->GetMetrics();
  EXPECT_EQ(m_reset.total_h2h_bytes, 0);
  EXPECT_DOUBLE_EQ(m_reset.total_h2h_time_ms, 0.0);
  EXPECT_DOUBLE_EQ(m_reset.total_push_resharded_time_ms, 0.0);
  EXPECT_EQ(m_reset.push_resharded_call_count, 0);
}

TEST_F(WeightSynchronizerTest,
       PushWeightsReshardedConcurrentOverlappingPushesDoNotDoubleCountTime) {
  size_t num_layers = 1;
  size_t num_shards = 1;
  size_t slice_byte_size = 500000;  // 500 KB

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers, num_shards, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  // One destination shard per push, so the concurrent pushes overlap in time
  // but never write the same destination bytes.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers, /*num_shards=*/2, slice_byte_size,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = absl::StrCat("127.0.0.1:", *ws_dest->local_port());

  // Configure a simulated rate limiter of 10 MB/s (~50 ms per 500KB push)
  // to guarantee substantial concurrency overlap.
  auto egress_limiter =
      std::make_shared<peregrine::v1::TestOnlyRateLimiter>(10'000'000);
  ws_source->SetTestOnlyRateLimiters(egress_limiter, /*ingress=*/nullptr);

  auto make_request = [&](uint64_t uuid, int dst_shard_idx) {
    tpu_sync::rpc::StartTransferRequest req;
    req.set_skip_d2h(true);
    req.set_uuid(uuid);
    auto* schedules = req.mutable_shard_push_schedules();
    auto* entry = (*schedules)[0].add_entries();
    entry->set_dst_peer(dest_peer);
    entry->set_dst_shard_idx(dst_shard_idx);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(slice_byte_size);
    entry->set_count(1);
    entry->set_layer_idx(0);
    return req;
  };

  uint64_t uuid1 = 80001;
  uint64_t uuid2 = 80002;
  ASSERT_OK(ws_dest->RegisterExpectedChunks(uuid1, 1));
  ASSERT_OK(ws_dest->RegisterExpectedChunks(uuid2, 1));

  absl::Notification start_notification;
  double elapsed1_ms = 0.0;
  double elapsed2_ms = 0.0;

  auto fut1 = std::async(std::launch::async, [&]() -> absl::Status {
    start_notification.WaitForNotification();
    auto t0 = absl::Now();
    auto s = ws_source->PushWeightsResharded(
        make_request(uuid1, /*dst_shard_idx=*/0));
    elapsed1_ms = absl::ToDoubleMilliseconds(absl::Now() - t0);
    return s;
  });

  auto fut2 = std::async(std::launch::async, [&]() -> absl::Status {
    start_notification.WaitForNotification();
    auto t0 = absl::Now();
    auto s = ws_source->PushWeightsResharded(
        make_request(uuid2, /*dst_shard_idx=*/1));
    elapsed2_ms = absl::ToDoubleMilliseconds(absl::Now() - t0);
    return s;
  });

  start_notification.Notify();

  ASSERT_OK(fut1.get());
  ASSERT_OK(fut2.get());
  ASSERT_OK(ws_dest->WaitForTransferCompletion(uuid1));
  ASSERT_OK(ws_dest->WaitForTransferCompletion(uuid2));

  WeightSyncMetrics m = ws_source->GetMetrics();
  EXPECT_EQ(m.total_h2h_bytes, 2 * slice_byte_size);
  EXPECT_EQ(m.push_resharded_call_count, 2);
  EXPECT_GT(m.total_h2h_time_ms, 0.0);
  EXPECT_GT(elapsed1_ms, 0.0);
  EXPECT_GT(elapsed2_ms, 0.0);
  EXPECT_LT(m.total_h2h_time_ms, elapsed1_ms + elapsed2_ms);
}

TEST_F(WeightSynchronizerTest, TelemetryRecordsOccupancyAndPushMetrics) {
  ASSERT_OK(telemetry::RaidenMetricStore::GetGlobalMetricStore()
                .InitializeFromBackendNames({"buffered"}));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_source,
                          WeightSynchronizerBase::Create(
                              num_layers_, num_shards_, slice_byte_size_,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest,
                          WeightSynchronizerBase::Create(
                              num_layers_, num_shards_, slice_byte_size_,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));

  // Buffer allocation gauge should be recorded
  auto samples = telemetry::RaidenMetricStore::GetGlobalMetricStore()
                     .GetAndResetMetricSamples();
  EXPECT_FALSE(
      samples[std::string("tpu_raiden_") +
              std::string(
                  telemetry::metric_names::kWeightSyncBufferAllocatedBytes)]
          .empty());
  EXPECT_GT(
      samples[std::string("tpu_raiden_") +
              std::string(
                  telemetry::metric_names::kWeightSyncBufferAllocatedBytes)]
          .back(),
      0.0);

  // Setup simple transfer
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());
  uint64_t uuid = 1234567;

  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(uuid);

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(slice_byte_size_);
  entry->set_count(1);
  entry->set_layer_idx(0);

  ASSERT_OK(ws_dest->RegisterExpectedChunks(uuid, 1));
  ASSERT_OK(ws_source->PushWeightsResharded(request));
  ASSERT_OK(ws_dest->WaitForTransferCompletion(uuid));

  samples = telemetry::RaidenMetricStore::GetGlobalMetricStore()
                .GetAndResetMetricSamples();
  EXPECT_FALSE(
      samples[std::string("tpu_raiden_") +
              std::string(telemetry::metric_names::kWeightSyncPushDurationMs)]
          .empty());
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

TEST_F(WeightSynchronizerTest,
       PushWeightsReshardedConcurrentCoalescedD2hWaitsForLeaderCompletion) {
  auto client_status_or = xla::GetXlaPjrtCpuClient(xla::CpuClientOptions());
  ASSERT_OK(client_status_or.status());
  auto client = std::move(client_status_or.value());

  constexpr size_t kNumLayers = 8;
  constexpr size_t kSliceByteSize = 16384;
  auto memory_space_status_or =
      client->addressable_devices()[0]->default_memory_space();
  ASSERT_OK(memory_space_status_or.status());
  xla::PjRtMemorySpace* memory_space = memory_space_status_or.value();

  std::vector<std::vector<uint8_t>> src_device_data(
      kNumLayers, std::vector<uint8_t>(kSliceByteSize));
  std::vector<std::unique_ptr<xla::PjRtBuffer>> src_pjrt_buffers;
  std::vector<std::vector<raiden::RaidenBufferHandle>> src_buffers;
  for (size_t l = 0; l < kNumLayers; ++l) {
    std::memset(src_device_data[l].data(), static_cast<int>(0x30 + l),
                kSliceByteSize);
    auto buf_or = client->BufferFromHostBuffer(
        src_device_data[l].data(), xla::U8,
        {static_cast<int64_t>(kSliceByteSize)},
        /*byte_strides=*/std::nullopt,
        xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
        /*on_done_with_host_buffer=*/nullptr, memory_space,
        /*device_layout=*/nullptr);
    ASSERT_OK(buf_or.status());
    src_pjrt_buffers.push_back(std::move(buf_or.value()));
    auto handle_or =
        raiden::RaidenBufferHandle::Acquire(src_pjrt_buffers.back().get());
    ASSERT_OK(handle_or.status());
    src_buffers.push_back({handle_or.value()});
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> ws_source,
      WeightSynchronizerBase::Create(src_buffers, /*local_port=*/0));
  ws_source->SetPipelineGroupSize(1);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest1,
                          WeightSynchronizerBase::Create(
                              kNumLayers, /*num_shards=*/1, kSliceByteSize,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> ws_dest2,
                          WeightSynchronizerBase::Create(
                              kNumLayers, /*num_shards=*/1, kSliceByteSize,
                              /*local_port=*/0, /*host_blocks_to_allocate=*/1));
  ASSERT_TRUE(ws_dest1->local_port().has_value());
  ASSERT_TRUE(ws_dest2->local_port().has_value());
  std::string dest1_peer = absl::StrCat("127.0.0.1:", *ws_dest1->local_port());
  std::string dest2_peer = absl::StrCat("127.0.0.1:", *ws_dest2->local_port());

  // Initialize source and destination host staging buffers with zeros so only
  // completed D2H copies can populate 0x30 + l.
  for (size_t l = 0; l < kNumLayers; ++l) {
    TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> src_buf,
                         ws_source->GetHostBuffer(l, 0));
    std::memset(src_buf.data(), 0x00, kSliceByteSize);
    TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dst1_buf,
                         ws_dest1->GetHostBuffer(l, 0));
    std::memset(dst1_buf.data(), 0x00, kSliceByteSize);
    TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> dst2_buf,
                         ws_dest2->GetHostBuffer(l, 0));
    std::memset(dst2_buf.data(), 0x00, kSliceByteSize);
  }

  constexpr uint64_t kSharedUuid = 90001;
  auto make_request = [&](const std::string& peer) {
    tpu_sync::rpc::StartTransferRequest req;
    req.set_skip_d2h(false);
    req.set_uuid(kSharedUuid);
    auto* schedules = req.mutable_shard_push_schedules();
    for (size_t l = 0; l < kNumLayers; ++l) {
      auto* entry = (*schedules)[0].add_entries();
      entry->set_dst_peer(peer);
      entry->set_dst_shard_idx(0);
      entry->set_src_offset_bytes(0);
      entry->set_dst_offset_bytes(0);
      entry->set_size_bytes(kSliceByteSize);
      entry->set_count(1);
      entry->set_layer_idx(static_cast<int32_t>(l));
    }
    return req;
  };

  ASSERT_OK(ws_dest1->RegisterExpectedChunks(kSharedUuid, kNumLayers));
  ASSERT_OK(ws_dest2->RegisterExpectedChunks(kSharedUuid, kNumLayers));

  absl::Notification start;
  auto fut1 = std::async(std::launch::async, [&]() {
    start.WaitForNotification();
    return ws_source->PushWeightsResharded(make_request(dest1_peer));
  });
  auto fut2 = std::async(std::launch::async, [&]() {
    start.WaitForNotification();
    return ws_source->PushWeightsResharded(make_request(dest2_peer));
  });
  start.Notify();

  ASSERT_OK(fut1.get());
  ASSERT_OK(fut2.get());
  ASSERT_OK(ws_dest1->WaitForTransferCompletion(kSharedUuid));
  ASSERT_OK(ws_dest2->WaitForTransferCompletion(kSharedUuid));

  EXPECT_EQ(ws_source->GetMetrics().d2h_call_count, 1);
  for (size_t l = 0; l < kNumLayers; ++l) {
    const uint8_t expected = static_cast<uint8_t>(0x30 + l);
    TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> dst1_buf,
                         ws_dest1->GetHostBuffer(l, 0));
    const uint8_t* dst1_ptr = dst1_buf.data();
    TF_ASSERT_OK_AND_ASSIGN(absl::Span<const uint8_t> dst2_buf,
                         ws_dest2->GetHostBuffer(l, 0));
    const uint8_t* dst2_ptr = dst2_buf.data();
    for (size_t b = 0; b < kSliceByteSize; ++b) {
      EXPECT_EQ(dst1_ptr[b], expected) << "dest1 layer " << l << " byte " << b;
      EXPECT_EQ(dst2_ptr[b], expected) << "dest2 layer " << l << " byte " << b;
    }
  }
}

TEST_F(WeightSynchronizerTest,
       GetLocalEndpointsReturnsPerNumaDataNicEndpointsWithGlobalShardIndices) {
  // RaidenManagerBase binds wildcard [::] when local_ips_ has >1 entries, so
  // any IP strings in HostNicAddress are preserved in local_ips() without
  // per-IP bind.
  std::vector<HostNicAddress> nics = {
      {"eth0", "10.11.3.60", 0, NicClassification::kDataPlane},
      {"eth1", "10.35.3.60", 0, NicClassification::kControlPlane},
      {"eth2", "10.34.3.60", 1, NicClassification::kDataPlane},
  };
  std::vector<transport::ShardInfo> shards = {
      {/*local_index=*/0, /*global_index=*/8, /*numa_node=*/0},
      {/*local_index=*/1, /*global_index=*/9, /*numa_node=*/0},
      {/*local_index=*/2, /*global_index=*/10, /*numa_node=*/1},
      {/*local_index=*/3, /*global_index=*/11, /*numa_node=*/1},
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto ws,
      WeightSynchronizerBase::Create(
          /*num_layers=*/1, /*num_shards=*/4,
          std::vector<size_t>{slice_byte_size_}, /*local_port=*/0,
          /*host_blocks_to_allocate=*/1, /*parallelism=*/1,
          /*listener_port=*/std::nullopt, /*bind_ip=*/std::nullopt,
          /*layer_names=*/std::vector<std::string>{}, /*auto_h2d=*/false,
          /*numa_nodes=*/std::vector<int>{0, 1}, nics, shards));
  ws->SetGlobalShardIndices({8, 9, 10, 11});

  ASSERT_TRUE(ws->local_port().has_value());
  const int port = *ws->local_port();
  std::vector<RaidenTransferEndpoint> endpoints = ws->get_local_endpoints();
  ASSERT_EQ(endpoints.size(), 2u);
  EXPECT_EQ(endpoints[0].endpoint, absl::StrCat("10.11.3.60:", port));
  EXPECT_THAT(endpoints[0].shards, ::testing::ElementsAre(8, 9));
  EXPECT_EQ(endpoints[1].endpoint, absl::StrCat("10.34.3.60:", port));
  EXPECT_THAT(endpoints[1].shards, ::testing::ElementsAre(10, 11));
}

// One single-shard layer of a device-backed synchronizer. An empty `tiles`
// keeps the CPU buffer's default (untiled) layout; otherwise the handle is
// relabeled with a row-major tiled layout so that D2H detiles and H2D tiles on
// the host, as on TPU.
struct DeviceLayerSpec {
  xla::PrimitiveType type;
  std::vector<int64_t> dims;
  std::vector<xla::Tile> tiles;
};

// A synchronizer whose layers are backed by XLA:CPU device buffers. Members are
// destroyed bottom-up, so `ws` goes before the buffers and client it uses.
struct DeviceBackedSynchronizer {
  std::unique_ptr<xla::PjRtClient> client;
  std::vector<std::vector<uint8_t>> placeholders;
  std::vector<std::unique_ptr<xla::PjRtBuffer>> pjrt_buffers;
  std::unique_ptr<WeightSynchronizerBase> ws;
  // Sum of the on-device sizes of every layer, i.e. the bytes one full D2H or
  // H2D moves.
  size_t total_device_bytes = 0;
};

void CreateDeviceBackedSynchronizer(absl::Span<const DeviceLayerSpec> specs,
                                    bool auto_h2d,
                                    DeviceBackedSynchronizer* out) {
  TF_ASSERT_OK_AND_ASSIGN(out->client,
                          xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      out->client->addressable_devices()[0]->default_memory_space());
  std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers;
  for (const DeviceLayerSpec& spec : specs) {
    const xla::Shape shape = xla::ShapeUtil::MakeShape(spec.type, spec.dims);
    std::vector<uint8_t>& placeholder = out->placeholders.emplace_back(
        xla::ShapeUtil::ByteSizeOf(shape), uint8_t{0x5A});
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<xla::PjRtBuffer> buffer,
        out->client->BufferFromHostBuffer(
            placeholder.data(), spec.type, spec.dims,
            /*byte_strides=*/std::nullopt,
            xla::PjRtClient::HostBufferSemantics::
                kImmutableUntilTransferCompletes,
            /*on_done_with_host_buffer=*/nullptr, memory_space,
            /*device_layout=*/nullptr));
    TF_ASSERT_OK_AND_ASSIGN(raiden::RaidenBufferHandle handle,
                            raiden::RaidenBufferHandle::Acquire(buffer.get()));
    if (!spec.tiles.empty()) {
      std::vector<int64_t> minor_to_major(spec.dims.size());
      for (size_t i = 0; i < minor_to_major.size(); ++i) {
        minor_to_major[i] = static_cast<int64_t>(minor_to_major.size() - 1 - i);
      }
      handle.shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
          spec.type, spec.dims, minor_to_major, spec.tiles);
    }
    out->total_device_bytes += handle.GetOnDeviceSizeInBytes();
    out->pjrt_buffers.push_back(std::move(buffer));
    layer_buffers.push_back({std::move(handle)});
  }
  TF_ASSERT_OK_AND_ASSIGN(
      out->ws,
      WeightSynchronizerBase::Create(
          layer_buffers, /*local_port=*/0, /*external_host_ptrs=*/std::nullopt,
          /*unsafe_skip_buffer_lock=*/false, /*parallelism=*/1,
          /*listener_port=*/std::nullopt, /*bind_ip=*/std::nullopt,
          /*layer_names=*/std::vector<std::string>{}, auto_h2d));
}

// A tiny untiled layer followed by a large tiled one. Layer 0 finishes its D2H
// almost immediately, while layer 1 must copy 16 MiB and then detile it on the
// host, so "time until layer 0 is on host" and "time until every layer is on
// host" are far apart.
std::vector<DeviceLayerSpec> SmallUntiledThenLargeTiledLayers() {
  return {
      {xla::U8, {4096}, {}},
      {xla::F32, {2048, 2048}, {xla::Tile({8, 128})}},
  };
}

// Host staging buffers carry extra scratchpad bytes beyond the device size.
// Byte metrics must count the bytes actually copied, not the staging capacity.
constexpr size_t kHostScratchpadBytes = 256 * 1024;

TEST_F(WeightSynchronizerTest, PushWeightsReshardedD2hMetricsCoverEveryLayer) {
  absl::SetFlag(&FLAGS_raiden_weight_sync_host_buffer_scratchpad_size,
                kHostScratchpadBytes);
  ASSERT_OK(telemetry::RaidenMetricStore::GetGlobalMetricStore()
                .InitializeFromBackendNames({"buffered"}));
  absl::Cleanup reset_backends = [] {
    telemetry::RaidenMetricStore::GetGlobalMetricStore().SetBackends({});
  };

  const std::vector<DeviceLayerSpec> specs = SmallUntiledThenLargeTiledLayers();
  DeviceBackedSynchronizer source;
  ASSERT_NO_FATAL_FAILURE(
      CreateDeviceBackedSynchronizer(specs, /*auto_h2d=*/false, &source));
  // The push D2Hs every layer in full; the wire only needs to carry a prefix of
  // each layer for this test.
  constexpr size_t kSentBytesPerLayer = 4096;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> dest,
      WeightSynchronizerBase::Create(specs.size(), /*num_shards=*/1,
                                     kSentBytesPerLayer, /*local_port=*/0));
  ASSERT_TRUE(dest->local_port().has_value());
  const std::string dest_peer = absl::StrCat("127.0.0.1:", *dest->local_port());

  constexpr uint64_t kUuid = 91001;
  tpu_sync::rpc::StartTransferRequest request;
  request.set_skip_d2h(false);
  request.set_uuid(kUuid);
  auto& schedule = (*request.mutable_shard_push_schedules())[0];
  for (size_t l = 0; l < specs.size(); ++l) {
    auto* entry = schedule.add_entries();
    entry->set_dst_peer(dest_peer);
    entry->set_dst_shard_idx(0);
    entry->set_src_offset_bytes(0);
    entry->set_dst_offset_bytes(0);
    entry->set_size_bytes(kSentBytesPerLayer);
    entry->set_count(1);
    entry->set_layer_idx(static_cast<int32_t>(l));
  }
  ASSERT_OK(dest->RegisterExpectedChunks(kUuid, specs.size()));
  // Drop samples recorded by earlier tests.
  telemetry::RaidenMetricStore::GetGlobalMetricStore()
      .GetAndResetMetricSamples();

  ASSERT_OK(source.ws->PushWeightsResharded(request));
  ASSERT_OK(dest->WaitForTransferCompletion(kUuid));

  const WeightSyncMetrics m = source.ws->GetMetrics();
  // Layer 1 was detiled as part of its D2H, so the D2H of the whole push lasts
  // at least as long as that detile. A metric that stops at layer 0 is shorter.
  ASSERT_GT(m.last_detiling_time_ms, 0.0);
  EXPECT_GE(m.last_d2h_time_ms, m.last_detiling_time_ms);
  EXPECT_LE(m.last_d2h_time_ms, m.last_total_push_resharded_time_ms);
  EXPECT_DOUBLE_EQ(m.total_d2h_time_ms, m.last_d2h_time_ms);
  // Bytes are the device bytes copied, not the staging buffer capacity.
  EXPECT_EQ(m.last_d2h_bytes, source.total_device_bytes);
  EXPECT_EQ(m.total_d2h_bytes, source.total_device_bytes);
  EXPECT_EQ(m.d2h_call_count, 1);

  // The struct metric and the exported histogram describe the same interval.
  auto samples = telemetry::RaidenMetricStore::GetGlobalMetricStore()
                     .GetAndResetMetricSamples();
  const std::vector<double>& d2h_samples = samples[absl::StrCat(
      "tpu_raiden_", telemetry::metric_names::kWeightSyncD2hTransferTimeMs)];
  ASSERT_THAT(d2h_samples, ::testing::SizeIs(1));
  EXPECT_DOUBLE_EQ(m.last_d2h_time_ms, d2h_samples[0]);
}

TEST_F(WeightSynchronizerTest, ExplicitD2hRecordsMetrics) {
  absl::SetFlag(&FLAGS_raiden_weight_sync_host_buffer_scratchpad_size,
                kHostScratchpadBytes);
  DeviceBackedSynchronizer s;
  ASSERT_NO_FATAL_FAILURE(CreateDeviceBackedSynchronizer(
      SmallUntiledThenLargeTiledLayers(), /*auto_h2d=*/false, &s));

  const WeightSyncMetrics m0 = s.ws->GetMetrics();
  EXPECT_EQ(m0.d2h_call_count, 0);
  EXPECT_EQ(m0.last_d2h_bytes, 0);
  EXPECT_DOUBLE_EQ(m0.last_d2h_time_ms, 0.0);

  absl::Time start = absl::Now();
  TF_ASSERT_OK_AND_ASSIGN(raiden::PjRtCopyFuture d2h, s.ws->D2h());
  ASSERT_OK(d2h.Await());
  const double wall_ms = absl::ToDoubleMilliseconds(absl::Now() - start);

  // Read right after Await(): the metrics are recorded before the returned
  // future completes, so callers never observe stale values.
  const WeightSyncMetrics m1 = s.ws->GetMetrics();
  EXPECT_EQ(m1.d2h_call_count, 1);
  EXPECT_EQ(m1.last_d2h_bytes, s.total_device_bytes);
  EXPECT_EQ(m1.total_d2h_bytes, s.total_device_bytes);
  ASSERT_GT(m1.last_detiling_time_ms, 0.0);
  EXPECT_GE(m1.last_d2h_time_ms, m1.last_detiling_time_ms);
  EXPECT_LE(m1.last_d2h_time_ms, wall_ms);
  EXPECT_DOUBLE_EQ(m1.total_d2h_time_ms, m1.last_d2h_time_ms);

  TF_ASSERT_OK_AND_ASSIGN(d2h, s.ws->D2h());
  ASSERT_OK(d2h.Await());
  const WeightSyncMetrics m2 = s.ws->GetMetrics();
  EXPECT_EQ(m2.d2h_call_count, 2);
  EXPECT_EQ(m2.last_d2h_bytes, s.total_device_bytes);
  EXPECT_EQ(m2.total_d2h_bytes, 2 * s.total_device_bytes);
  EXPECT_DOUBLE_EQ(m2.total_d2h_time_ms,
                   m1.last_d2h_time_ms + m2.last_d2h_time_ms);
}

TEST_F(WeightSynchronizerTest, ExplicitH2dRecordsMetrics) {
  absl::SetFlag(&FLAGS_raiden_weight_sync_host_buffer_scratchpad_size,
                kHostScratchpadBytes);
  DeviceBackedSynchronizer s;
  ASSERT_NO_FATAL_FAILURE(CreateDeviceBackedSynchronizer(
      SmallUntiledThenLargeTiledLayers(), /*auto_h2d=*/false, &s));

  const WeightSyncMetrics m0 = s.ws->GetMetrics();
  EXPECT_EQ(m0.last_h2d_bytes, 0);
  EXPECT_DOUBLE_EQ(m0.last_h2d_time_ms, 0.0);

  absl::Time start = absl::Now();
  TF_ASSERT_OK_AND_ASSIGN(raiden::PjRtCopyFuture h2d, s.ws->H2d());
  ASSERT_OK(h2d.Await());
  const double wall_ms = absl::ToDoubleMilliseconds(absl::Now() - start);

  const WeightSyncMetrics m1 = s.ws->GetMetrics();
  EXPECT_EQ(m1.last_h2d_bytes, s.total_device_bytes);
  EXPECT_EQ(m1.total_h2d_bytes, s.total_device_bytes);
  // Layer 1 is tiled on the host inside H2d(), so its tiling is part of it.
  ASSERT_GT(m1.last_tiling_time_ms, 0.0);
  EXPECT_GE(m1.last_h2d_time_ms, m1.last_tiling_time_ms);
  EXPECT_LE(m1.last_h2d_time_ms, wall_ms);
  EXPECT_DOUBLE_EQ(m1.total_h2d_time_ms, m1.last_h2d_time_ms);

  TF_ASSERT_OK_AND_ASSIGN(h2d, s.ws->H2d());
  ASSERT_OK(h2d.Await());
  const WeightSyncMetrics m2 = s.ws->GetMetrics();
  EXPECT_EQ(m2.last_h2d_bytes, s.total_device_bytes);
  EXPECT_EQ(m2.total_h2d_bytes, 2 * s.total_device_bytes);
  EXPECT_DOUBLE_EQ(m2.total_h2d_time_ms,
                   m1.last_h2d_time_ms + m2.last_h2d_time_ms);

  s.ws->ResetMetrics();
  const WeightSyncMetrics m_reset = s.ws->GetMetrics();
  EXPECT_EQ(m_reset.last_h2d_bytes, 0);
  EXPECT_EQ(m_reset.total_h2d_bytes, 0);
  EXPECT_DOUBLE_EQ(m_reset.total_h2d_time_ms, 0.0);
}

TEST_F(WeightSynchronizerTest, AutoH2dOnDataReceivedRecordsH2dBytes) {
  absl::SetFlag(&FLAGS_raiden_weight_sync_host_buffer_scratchpad_size,
                kHostScratchpadBytes);
  DeviceBackedSynchronizer s;
  ASSERT_NO_FATAL_FAILURE(CreateDeviceBackedSynchronizer(
      SmallUntiledThenLargeTiledLayers(), /*auto_h2d=*/true, &s));

  constexpr uint64_t kUuid = 91002;
  ASSERT_OK(s.ws->OnLayerDataReceived(0, kUuid));
  ASSERT_OK(s.ws->OnLayerDataReceived(1, kUuid));
  ASSERT_OK(s.ws->OnDataReceived(kUuid));

  const WeightSyncMetrics m = s.ws->GetMetrics();
  EXPECT_EQ(m.last_h2d_bytes, s.total_device_bytes);
  EXPECT_EQ(m.total_h2d_bytes, s.total_device_bytes);
  EXPECT_GT(m.last_h2d_time_ms, 0.0);
  EXPECT_DOUBLE_EQ(m.total_h2d_time_ms, m.last_h2d_time_ms);
}

// Fake PJRT C-API raw buffer with in-memory bytes, optional lifetime token, and
// optional caller-controlled gate on CopyRawDeviceToHost.
class GatedRawBuffer {
 public:
  struct D2hGate {
    xla::Promise<> promise;
    std::shared_ptr<absl::Notification> started;
  };

  explicit GatedRawBuffer(std::vector<uint8_t> initial_bytes,
                          std::shared_ptr<int> lifetime_token = nullptr)
      : data_(std::move(initial_bytes)),
        lifetime_token_(std::move(lifetime_token)) {}

  static std::shared_ptr<GatedRawBuffer> Filled(
      size_t bytes, uint8_t val, std::shared_ptr<int> token = nullptr) {
    return std::make_shared<GatedRawBuffer>(std::vector<uint8_t>(bytes, val),
                                            std::move(token));
  }

  void Fill(uint8_t val) {
    absl::MutexLock lock(mu_);
    std::fill(data_.begin(), data_.end(), val);
  }

  D2hGate ArmGate() {
    auto [promise, future] = xla::MakePromise();
    auto started = std::make_shared<absl::Notification>();
    absl::MutexLock lock(mu_);
    next_d2h_gate_ = std::move(future);
    next_d2h_started_ = started;
    return {std::move(promise), std::move(started)};
  }

  size_t d2h_call_count() const {
    absl::MutexLock lock(mu_);
    return d2h_calls_;
  }

  size_t GetOnDeviceSizeInBytes() const {
    absl::MutexLock lock(mu_);
    return data_.size();
  }

  xla::Future<> CopyRawHostToDevice(const void* src, int64_t offset,
                                    int64_t transfer_size) {
    absl::MutexLock lock(mu_);
    if (offset < 0 || transfer_size < 0 ||
        static_cast<size_t>(offset + transfer_size) > data_.size()) {
      return xla::Future<>(absl::OutOfRangeError("H2D copy out of bounds"));
    }
    std::memcpy(data_.data() + offset, src, static_cast<size_t>(transfer_size));
    return xla::Future<>(absl::OkStatus());
  }

  xla::Future<> CopyRawDeviceToHost(void* dst, int64_t offset,
                                    int64_t transfer_size) {
    std::vector<uint8_t> snapshot;
    xla::Future<> gate;
    std::shared_ptr<absl::Notification> started;
    {
      absl::MutexLock lock(mu_);
      ++d2h_calls_;
      if (offset < 0 || transfer_size < 0 ||
          static_cast<size_t>(offset + transfer_size) > data_.size()) {
        return xla::Future<>(absl::OutOfRangeError("D2H copy out of bounds"));
      }
      snapshot.assign(data_.begin() + offset,
                      data_.begin() + offset + transfer_size);
      gate = std::move(next_d2h_gate_);
      next_d2h_gate_ = xla::Future<>();
      started = std::move(next_d2h_started_);
      next_d2h_started_.reset();
    }
    if (started != nullptr && !started->HasBeenNotified()) started->Notify();
    if (!gate.IsValid()) {
      std::memcpy(dst, snapshot.data(), snapshot.size());
      return xla::Future<>(absl::OkStatus());
    }
    auto [promise, done] = xla::MakePromise();
    std::move(gate).OnReady(
        [dst, snapshot = std::move(snapshot),
         promise = std::move(promise)](absl::Status s) mutable {
          if (s.ok()) std::memcpy(dst, snapshot.data(), snapshot.size());
          promise.Set(std::move(s));
        });
    return done;
  }

  static raiden::RaidenBufferHandle MakeHandle(
      const std::shared_ptr<GatedRawBuffer>& raw, const xla::Shape& shape) {
    static const PJRT_Api kApi = [] {
      PJRT_Api api{};
      api.struct_size = PJRT_Api_STRUCT_SIZE;
      api.PJRT_Error_Message = +[](PJRT_Error_Message_Args* a) {
        absl::string_view m =
            reinterpret_cast<const absl::Status*>(a->error)->message();
        a->message = m.data();
        a->message_size = m.size();
      };
      api.PJRT_Error_GetCode = +[](PJRT_Error_GetCode_Args* a) -> PJRT_Error* {
        int c = static_cast<int>(
            reinterpret_cast<const absl::Status*>(a->error)->code());
        std::memcpy(&a->code, &c, sizeof(a->code));
        return nullptr;
      };
      api.PJRT_Error_Destroy = +[](PJRT_Error_Destroy_Args* a) {
        delete reinterpret_cast<absl::Status*>(a->error);
      };
      api.PJRT_Error_ForEachPayload =
          +[](PJRT_Error_ForEachPayload_Args*) -> PJRT_Error* {
        return nullptr;
      };
      api.PJRT_Event_OnReady = +[](PJRT_Event_OnReady_Args* a) -> PJRT_Error* {
        // Move out onto the stack first: synchronous `cb` invokes
        // `PJRT_Event_Destroy(a->event)` before `OnReady` returns.
        xla::Future<> fut =
            std::move(*reinterpret_cast<xla::Future<>*>(a->event));
        std::move(fut).OnReady(
            [cb = a->callback, arg = a->user_arg](absl::Status s) {
              PJRT_Error* err =
                  s.ok() ? nullptr
                         : reinterpret_cast<PJRT_Error*>(new absl::Status(s));
              cb(err, arg);
            });
        return nullptr;
      };
      api.PJRT_Event_Destroy = +[](PJRT_Event_Destroy_Args* a) -> PJRT_Error* {
        delete reinterpret_cast<xla::Future<>*>(a->event);
        return nullptr;
      };
      return api;
    }();
    static const PJRT_RawBuffer_Extension kExt = [] {
      PJRT_RawBuffer_Extension ext{};
      ext.base.struct_size = PJRT_RawBuffer_Extension_STRUCT_SIZE;
      ext.base.type = PJRT_Extension_Type_RawBuffer;
      ext.PJRT_RawBuffer_Destroy =
          +[](PJRT_RawBuffer_Destroy_Args* a) -> PJRT_Error* {
        delete reinterpret_cast<std::shared_ptr<GatedRawBuffer>*>(a->buffer);
        return nullptr;
      };
      ext.PJRT_RawBuffer_GetOnDeviceSizeInBytes =
          +[](PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args* a) -> PJRT_Error* {
        a->on_device_size_in_bytes =
            Unwrap(a->buffer)->GetOnDeviceSizeInBytes();
        return nullptr;
      };
      ext.PJRT_RawBuffer_CopyRawHostToDevice =
          +[](PJRT_RawBuffer_CopyRawHostToDevice_Args* a) -> PJRT_Error* {
        a->event = reinterpret_cast<PJRT_Event*>(
            new xla::Future<>(Unwrap(a->buffer)->CopyRawHostToDevice(
                a->src, a->offset, a->transfer_size)));
        return nullptr;
      };
      ext.PJRT_RawBuffer_CopyRawDeviceToHost =
          +[](PJRT_RawBuffer_CopyRawDeviceToHost_Args* a) -> PJRT_Error* {
        a->event = reinterpret_cast<PJRT_Event*>(
            new xla::Future<>(Unwrap(a->buffer)->CopyRawDeviceToHost(
                a->dst, a->offset, a->transfer_size)));
        return nullptr;
      };
      return ext;
    }();
    raiden::RaidenBufferHandle result;
    result.shape = shape;
    result.is_common_buffer = false;
    result.c_raw_buffer = reinterpret_cast<PJRT_RawBuffer*>(
        new std::shared_ptr<GatedRawBuffer>(raw));
    result.c_hold = std::make_shared<raiden::RawBufferHolder>(
        &kApi, &kExt, result.c_raw_buffer);
    return result;
  }

  static raiden::RaidenBufferHandle MakeU8Handle(
      const std::shared_ptr<GatedRawBuffer>& raw) {
    return MakeHandle(
        raw,
        xla::ShapeUtil::MakeShape(
            xla::U8, {static_cast<int64_t>(raw->GetOnDeviceSizeInBytes())}));
  }

 private:
  static const std::shared_ptr<GatedRawBuffer>& Unwrap(PJRT_RawBuffer* buf) {
    return *reinterpret_cast<std::shared_ptr<GatedRawBuffer>*>(buf);
  }

  mutable absl::Mutex mu_;
  std::vector<uint8_t> data_ ABSL_GUARDED_BY(mu_);
  xla::Future<> next_d2h_gate_ ABSL_GUARDED_BY(mu_);
  std::shared_ptr<absl::Notification> next_d2h_started_ ABSL_GUARDED_BY(mu_);
  size_t d2h_calls_ ABSL_GUARDED_BY(mu_) = 0;
  std::shared_ptr<int> lifetime_token_;
};

class LayerNotifyingSynchronizer : public WeightSynchronizerBase {
 public:
  static absl::StatusOr<std::unique_ptr<LayerNotifyingSynchronizer>> CreateCpu(
      size_t num_layers, size_t slice_byte_size) {
    std::vector<size_t> sizes(num_layers, slice_byte_size);
    auto ws = absl::WrapUnique(new LayerNotifyingSynchronizer(sizes));
    ABSL_RETURN_IF_ERROR(ws->InitHostBuffers(sizes, std::nullopt));
    return ws;
  }

  absl::Status OnLayerDataReceived(size_t layer_idx, uint64_t uuid) override {
    absl::Status s =
        WeightSynchronizerBase::OnLayerDataReceived(layer_idx, uuid);
    if (layer_idx < layer_count_ && !received_[layer_idx].HasBeenNotified()) {
      received_[layer_idx].Notify();
    }
    return s;
  }

  absl::Notification& layer_received(size_t layer_idx) {
    return received_[layer_idx];
  }

 private:
  explicit LayerNotifyingSynchronizer(const std::vector<size_t>& sizes)
      : WeightSynchronizerBase(sizes.size(), 1, sizes, 0, 1, std::nullopt, {},
                               false),
        layer_count_(sizes.size()),
        received_(std::make_unique<absl::Notification[]>(sizes.size())) {}
  size_t layer_count_;
  std::unique_ptr<absl::Notification[]> received_;
};

tpu_sync::rpc::StartTransferRequest MakeSimplePushRequest(
    uint64_t uuid, bool skip_d2h, const WeightSynchronizerBase& dest,
    size_t num_layers, size_t layer_bytes,
    std::optional<bool> skip_tiling_all = std::nullopt) {
  const std::string peer = absl::StrCat("127.0.0.1:", *dest.local_port());
  tpu_sync::rpc::StartTransferRequest request;
  request.set_uuid(uuid);
  request.set_is_sender(true);
  request.set_skip_d2h(skip_d2h);
  auto& schedule = (*request.mutable_shard_push_schedules())[0];
  for (size_t l = 0; l < num_layers; ++l) {
    auto* entry = schedule.add_entries();
    entry->set_dst_peer(peer);
    entry->set_dst_shard_idx(0);
    entry->set_size_bytes(layer_bytes);
    entry->set_count(1);
    entry->set_layer_idx(static_cast<int32_t>(l));
    if (skip_tiling_all.has_value()) {
      (*request.mutable_skip_tiling())[static_cast<int32_t>(l)] =
          *skip_tiling_all;
    }
  }
  return request;
}

absl::StatusOr<std::unique_ptr<WeightSynchronizerBase>> CreateGatedSource(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& handles,
    std::vector<std::string> layer_names = {}, int parallelism = 1) {
  return WeightSynchronizerBase::Create(handles, 0, std::nullopt, false,
                                        parallelism, std::nullopt, std::nullopt,
                                        std::move(layer_names), false);
}

absl::StatusOr<std::unique_ptr<WeightSynchronizerBase>> CreateU8Source(
    absl::Span<const std::shared_ptr<GatedRawBuffer>> raws,
    std::vector<std::string> layer_names = {}, int parallelism = 1) {
  std::vector<std::vector<raiden::RaidenBufferHandle>> handles;
  for (const auto& r : raws)
    handles.push_back({GatedRawBuffer::MakeU8Handle(r)});
  return CreateGatedSource(handles, std::move(layer_names), parallelism);
}

void ExpectErr(const absl::Status& s, absl::StatusCode code,
               absl::Span<const absl::string_view> substrs) {
  EXPECT_EQ(s.code(), code) << s;
  for (absl::string_view sub : substrs) {
    EXPECT_TRUE(absl::StrContains(s.message(), sub)) << s;
  }
}

std::vector<uint8_t> ReadHostLayerBytes(const WeightSynchronizerBase& ws,
                                        size_t layer_idx, size_t bytes) {
  const uint8_t* ptr = ws.GetHostBufferPtr(layer_idx, 0);
  return std::vector<uint8_t>(ptr, ptr + bytes);
}

std::future<absl::Status> PushAsync(WeightSynchronizerBase& ws,
                                    tpu_sync::rpc::StartTransferRequest req) {
  return std::async(std::launch::async, [&ws, req = std::move(req)]() {
    return ws.PushWeightsResharded(req);
  });
}

xla::Shape MakeTiledF32Shape() {
  return xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::F32, {16, 256}, /*minor_to_major=*/{1, 0}, {xla::Tile({8, 128})});
}

std::vector<uint8_t> MakeTiledPatternBytes(size_t bytes, uint8_t mul) {
  std::vector<uint8_t> out(bytes);
  for (size_t i = 0; i < bytes; ++i) {
    out[i] = static_cast<uint8_t>(((i ^ (i >> 8)) * mul + 11) & 0xFF);
  }
  return out;
}

TEST_F(WeightSynchronizerTest,
       ExplicitD2hSkipD2hPushStreamsPerLayerWithoutSendingUnstagedBytes) {
  constexpr size_t kNumLayers = 2;
  constexpr size_t kBytes = 4096;
  auto raw0 = GatedRawBuffer::Filled(kBytes, 0xA1);
  auto raw1 = GatedRawBuffer::Filled(kBytes, 0xB2);
  TF_ASSERT_OK_AND_ASSIGN(auto source,
                       CreateU8Source({raw0, raw1}, {"layer_0", "layer_1"}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest, LayerNotifyingSynchronizer::CreateCpu(kNumLayers, kBytes));
  auto g0 = raw0->ArmGate();
  auto g1 = raw1->ArmGate();
  TF_ASSERT_OK_AND_ASSIGN(auto d2h_future, source->D2h());
  EXPECT_FALSE(d2h_future.IsReady());

  constexpr uint64_t kUuid = 92001;
  ABSL_ASSERT_OK(dest->RegisterExpectedLayerChunks(kUuid, {{0, 1}, {1, 1}}));
  ABSL_ASSERT_OK(dest->RegisterExpectedChunks(kUuid, 2));
  auto push_future = PushAsync(
      *source, MakeSimplePushRequest(kUuid, true, *dest, kNumLayers, kBytes));
  g0.started->WaitForNotification();
  g1.started->WaitForNotification();
  EXPECT_FALSE(dest->layer_received(0).HasBeenNotified());
  EXPECT_FALSE(dest->layer_received(1).HasBeenNotified());
  EXPECT_EQ(ReadHostLayerBytes(*dest, 0, kBytes),
            std::vector<uint8_t>(kBytes, 0));

  // Release layer 0 only: H2H streams layer 0 while layer 1's D2H stays gated.
  g0.promise.Set(absl::OkStatus());
  ASSERT_TRUE(dest->layer_received(0).WaitForNotificationWithTimeout(
      absl::Seconds(10)));
  EXPECT_FALSE(d2h_future.IsReady());
  EXPECT_FALSE(dest->layer_received(1).HasBeenNotified());
  EXPECT_EQ(ReadHostLayerBytes(*dest, 0, kBytes),
            std::vector<uint8_t>(kBytes, 0xA1));
  EXPECT_EQ(ReadHostLayerBytes(*dest, 1, kBytes),
            std::vector<uint8_t>(kBytes, 0));
  g1.promise.Set(absl::OkStatus());
  ABSL_ASSERT_OK(push_future.get());
  ABSL_ASSERT_OK(d2h_future.Await());
  ABSL_ASSERT_OK(dest->WaitForTransferCompletion(kUuid));
  EXPECT_EQ(ReadHostLayerBytes(*dest, 1, kBytes),
            std::vector<uint8_t>(kBytes, 0xB2));
}

TEST_F(WeightSynchronizerTest,
       ExplicitD2hErrorPropagatesToFuturePushWaitAndH2d) {
  constexpr size_t kNumLayers = 2;
  constexpr size_t kBytes = 2048;
  auto raw0 = GatedRawBuffer::Filled(kBytes, 0x33);
  auto raw1 = GatedRawBuffer::Filled(kBytes, 0x44);
  TF_ASSERT_OK_AND_ASSIGN(auto source, CreateU8Source({raw0, raw1}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest, WeightSynchronizerBase::Create(kNumLayers, 1, kBytes, 0));
  auto g0 = raw0->ArmGate();
  auto g1 = raw1->ArmGate();
  TF_ASSERT_OK_AND_ASSIGN(auto d2h_future, source->D2h());
  g0.promise.Set(absl::OkStatus());
  constexpr absl::string_view kErr = "simulated_d2h_dma_failure_layer_1";
  g1.promise.Set(absl::UnavailableError(kErr));
  ExpectErr(d2h_future.Await(), absl::StatusCode::kUnavailable, {kErr});
  ABSL_EXPECT_OK(source->WaitForExplicitD2h(0));
  TF_ASSERT_OK_AND_ASSIGN(const uint8_t* ptr0,
                       source->WaitAndGetHostBufferPtr(0, 0));
  EXPECT_EQ(ptr0, source->GetHostBufferPtr(0, 0));
  ExpectErr(source->WaitForExplicitD2h(1), absl::StatusCode::kUnavailable,
            {kErr});
  ExpectErr(source->WaitAndGetHostBufferPtr(1, 0).status(),
            absl::StatusCode::kUnavailable, {kErr});
  ExpectErr(source->WaitAndGetHostBufferPtr(2, 0).status(),
            absl::StatusCode::kOutOfRange, {"layer_idx=2", "num_layers=2"});
  ExpectErr(source->H2d().status(), absl::StatusCode::kUnavailable, {kErr});
  ExpectErr(source->PushWeightsResharded(
                MakeSimplePushRequest(92002, true, *dest, kNumLayers, kBytes)),
            absl::StatusCode::kUnavailable, {kErr});

  // Positive recovery: a clean D2h() restores WaitAndGetHostBufferPtr/H2d/push.
  TF_ASSERT_OK_AND_ASSIGN(auto recovery_d2h, source->D2h());
  ABSL_ASSERT_OK(recovery_d2h.Await());
  ABSL_EXPECT_OK(source->WaitAndGetHostBufferPtr(1, 0));
  TF_ASSERT_OK_AND_ASSIGN(auto recovery_h2d, source->H2d());
  ABSL_ASSERT_OK(recovery_h2d.Await());
  ABSL_ASSERT_OK(dest->RegisterExpectedChunks(92003, 2));
  ABSL_ASSERT_OK(source->PushWeightsResharded(
      MakeSimplePushRequest(92003, true, *dest, kNumLayers, kBytes)));
  ABSL_ASSERT_OK(dest->WaitForTransferCompletion(92003));
  EXPECT_EQ(ReadHostLayerBytes(*dest, 1, kBytes),
            std::vector<uint8_t>(kBytes, 0x44));
}

TEST_F(
    WeightSynchronizerTest,
    ExplicitD2hSkipTilingMismatchRestagesWhenBoundAndFailsLoudlyWhenUnbound) {
  const xla::Shape tiled_shape = MakeTiledF32Shape();
  const size_t kBytes = xla::ShapeUtil::ByteSizeOf(tiled_shape);
  ASSERT_EQ(kBytes, 16384u);
  const auto device_tiled_bytes = MakeTiledPatternBytes(kBytes, 37);
  auto raw = std::make_shared<GatedRawBuffer>(device_tiled_bytes);
  auto handle = GatedRawBuffer::MakeHandle(raw, tiled_shape);
  TF_ASSERT_OK_AND_ASSIGN(
      auto source, CreateGatedSource({{handle}}, {"transformer.layer_0.attn"}));
  TF_ASSERT_OK_AND_ASSIGN(auto dest,
                       WeightSynchronizerBase::Create(1, 1, kBytes, 0));
  TF_ASSERT_OK_AND_ASSIGN(auto d2h_bound, source->D2h());
  ABSL_ASSERT_OK(d2h_bound.Await());
  ASSERT_NE(ReadHostLayerBytes(*source, 0, kBytes), device_tiled_bytes);

  // Case 1: Bound weights re-stage automatically on skip_tiling=true plan.
  ABSL_ASSERT_OK(dest->RegisterExpectedChunks(92010, 1));
  ABSL_ASSERT_OK(source->PushWeightsResharded(
      MakeSimplePushRequest(92010, true, *dest, 1, kBytes, true)));
  ABSL_ASSERT_OK(dest->WaitForTransferCompletion(92010));
  EXPECT_EQ(ReadHostLayerBytes(*dest, 0, kBytes), device_tiled_bytes);

  // Case 2: Unbinding before a mismatched skip_tiling=true plan fails loudly.
  ABSL_ASSERT_OK(source->BindWeights({{handle}}));
  source->SetSkipTiling(false);
  TF_ASSERT_OK_AND_ASSIGN(auto d2h_unbound, source->D2h());
  ABSL_ASSERT_OK(d2h_unbound.Await());
  source->UnbindWeights();
  ExpectErr(source->PushWeightsResharded(
                MakeSimplePushRequest(92011, true, *dest, 1, kBytes, true)),
            absl::StatusCode::kFailedPrecondition,
            {"transformer.layer_0.attn", "uuid=92011", "skip_tiling=false",
             "skip_tiling=true", "skip_d2h=false"});
}

TEST_F(WeightSynchronizerTest,
       ExplicitD2hReleasesDeviceHoldsWhileFutureRemainsAlive) {
  // Case 1: Untiled U8[4096], normal verl order (await D2H -> UnbindWeights()).
  {
    auto token = std::make_shared<int>(7);
    std::weak_ptr<int> weak_token = token;
    auto raw = GatedRawBuffer::Filled(4096, 0x7E, std::move(token));
    TF_ASSERT_OK_AND_ASSIGN(auto source, CreateU8Source({raw}));
    auto gate = raw->ArmGate();
    raw.reset();
    ASSERT_FALSE(weak_token.expired());
    TF_ASSERT_OK_AND_ASSIGN(auto d2h_future, source->D2h());
    EXPECT_TRUE(d2h_future.holds.empty());
    EXPECT_EQ(d2h_future.keep_alive, nullptr);
    gate.started->WaitForNotification();
    gate.promise.Set(absl::OkStatus());
    ABSL_ASSERT_OK(d2h_future.Await());
    EXPECT_FALSE(weak_token.expired());
    source->UnbindWeights();
    EXPECT_TRUE(weak_token.expired());
    EXPECT_TRUE(d2h_future.IsReady());
  }
  // Case 2: Tiled F32[16,256] ({8,128} tile) with UnbindWeights() called while
  // D2H+detile is still gated: buffer stays pinned until completion, then
  // drops.
  {
    const xla::Shape tiled_shape = MakeTiledF32Shape();
    const size_t kBytes = xla::ShapeUtil::ByteSizeOf(tiled_shape);
    const auto tiled_bytes = MakeTiledPatternBytes(kBytes, 29);
    auto token = std::make_shared<int>(9);
    std::weak_ptr<int> weak_token = token;
    auto raw = std::make_shared<GatedRawBuffer>(tiled_bytes, std::move(token));
    TF_ASSERT_OK_AND_ASSIGN(
        auto source,
        CreateGatedSource({{GatedRawBuffer::MakeHandle(raw, tiled_shape)}}));
    auto gate = raw->ArmGate();
    raw.reset();
    TF_ASSERT_OK_AND_ASSIGN(auto tiled_fut, source->D2h());
    gate.started->WaitForNotification();
    source->UnbindWeights();
    EXPECT_FALSE(weak_token.expired());
    gate.promise.Set(absl::OkStatus());
    ABSL_ASSERT_OK(tiled_fut.Await());
    EXPECT_TRUE(weak_token.expired());
    EXPECT_TRUE(tiled_fut.IsReady());
    EXPECT_NE(ReadHostLayerBytes(*source, 0, kBytes), tiled_bytes);
  }
}

TEST_F(WeightSynchronizerTest,
       SecondD2hInPushD2hAndGetHostBufferWaitForInFlightExplicitD2h) {
  constexpr size_t kBytes = 4096;
  auto raw = GatedRawBuffer::Filled(kBytes, 0x11);
  TF_ASSERT_OK_AND_ASSIGN(auto source, CreateU8Source({raw}));
  TF_ASSERT_OK_AND_ASSIGN(auto dest,
                       WeightSynchronizerBase::Create(1, 1, kBytes, 0));
  // Part 1a: WaitAndGetHostBufferPtr() waits for an in-flight explicit D2h().
  auto g1 = raw->ArmGate();
  TF_ASSERT_OK_AND_ASSIGN(auto first_d2h, source->D2h());
  g1.started->WaitForNotification();
  EXPECT_EQ(raw->d2h_call_count(), 1u);
  absl::Notification host_read_done;
  uint8_t observed_first_byte = 0;
  std::thread host_reader([&]() {
    TF_ASSERT_OK_AND_ASSIGN(const uint8_t* ptr,
                         source->WaitAndGetHostBufferPtr(0, 0));
    observed_first_byte = ptr[0];
    host_read_done.Notify();
  });
  EXPECT_FALSE(
      host_read_done.WaitForNotificationWithTimeout(absl::Milliseconds(50)));
  g1.promise.Set(absl::OkStatus());
  host_reader.join();
  EXPECT_EQ(observed_first_byte, 0x11);
  ABSL_ASSERT_OK(first_d2h.Await());

  // Part 1b: A second explicit D2h() serializes behind the in-flight D2h().
  raw->Fill(0x22);
  auto g2 = raw->ArmGate();
  TF_ASSERT_OK_AND_ASSIGN(auto second_d2h, source->D2h());
  g2.started->WaitForNotification();
  EXPECT_EQ(raw->d2h_call_count(), 2u);
  raw->Fill(0x33);
  absl::Notification third_d2h_returned;
  auto third_d2h_done = std::async(std::launch::async, [&]() -> absl::Status {
    ABSL_ASSIGN_OR_RETURN(auto f3, source->D2h());
    third_d2h_returned.Notify();
    return f3.Await();
  });
  EXPECT_FALSE(third_d2h_returned.WaitForNotificationWithTimeout(
      absl::Milliseconds(50)));
  EXPECT_EQ(raw->d2h_call_count(), 2u);
  g2.promise.Set(absl::OkStatus());
  ABSL_ASSERT_OK(second_d2h.Await());
  ABSL_ASSERT_OK(third_d2h_done.get());
  EXPECT_EQ(raw->d2h_call_count(), 3u);
  EXPECT_EQ(source->GetHostBufferPtr(0, 0)[0], 0x33);

  // Part 2: In-push D2H (skip_d2h=false) and unawaited D2h() + BindWeights()
  // serialize behind explicit D2h().
  auto g4 = raw->ArmGate();
  TF_ASSERT_OK_AND_ASSIGN(auto fourth_d2h, source->D2h());
  g4.started->WaitForNotification();
  EXPECT_EQ(raw->d2h_call_count(), 4u);
  raw->Fill(0x55);
  constexpr uint64_t kInPushUuid = 92020;
  ABSL_ASSERT_OK(dest->RegisterExpectedChunks(kInPushUuid, 1));
  auto in_push_req =
      MakeSimplePushRequest(kInPushUuid, false, *dest, 1, kBytes);
  absl::Notification in_push_finished;
  auto in_push_done = std::async(std::launch::async, [&]() {
    absl::Status s = source->PushWeightsResharded(in_push_req);
    in_push_finished.Notify();
    return s;
  });
  EXPECT_FALSE(
      in_push_finished.WaitForNotificationWithTimeout(absl::Milliseconds(50)));
  EXPECT_EQ(raw->d2h_call_count(), 4u);
  g4.promise.Set(absl::OkStatus());
  ABSL_ASSERT_OK(fourth_d2h.Await());
  ABSL_ASSERT_OK(in_push_done.get());
  ABSL_ASSERT_OK(dest->WaitForTransferCompletion(kInPushUuid));
  EXPECT_EQ(raw->d2h_call_count(), 5u);
  EXPECT_EQ(dest->GetHostBufferPtr(0, 0)[0], 0x55);
  // Dropping the returned future without awaiting is still awaited by rebind.
  (void)source->D2h();
  ABSL_ASSERT_OK(source->BindWeights({{GatedRawBuffer::MakeU8Handle(raw)}}));
}

TEST_F(WeightSynchronizerTest, PreTiledRelayAndActivePushGuards) {
  constexpr size_t kNumLayers = 2;
  constexpr size_t kBytes = 4096;
  auto raw0 = GatedRawBuffer::Filled(kBytes, 0x11);
  auto raw1 = GatedRawBuffer::Filled(kBytes, 0x22);
  auto h0 = GatedRawBuffer::MakeU8Handle(raw0);
  auto h1 = GatedRawBuffer::MakeU8Handle(raw1);
  TF_ASSERT_OK_AND_ASSIGN(auto relay, CreateGatedSource({{h0}, {h1}}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest, LayerNotifyingSynchronizer::CreateCpu(kNumLayers, kBytes));
  TF_ASSERT_OK_AND_ASSIGN(auto init_d2h, relay->D2h());
  ABSL_ASSERT_OK(init_d2h.Await());
  EXPECT_EQ(raw0->d2h_call_count(), 1u);

  // Arm `relay` as a PRE_TILED relay with planned skip_tiling=true: layer 0
  // exercises the PRE_TILED PushWeightsResharded gate directly; layer 1 parks
  // on tile_ready[1] so active_h2h_pushes=1 is deterministic.
  constexpr uint64_t kRelayUuid = 92500;
  ABSL_ASSERT_OK(
      dest->RegisterExpectedLayerChunks(kRelayUuid, {{0, 1}, {1, 1}}));
  ABSL_ASSERT_OK(dest->RegisterExpectedChunks(kRelayUuid, 2));
  auto relay_req =
      MakeSimplePushRequest(kRelayUuid, true, *dest, kNumLayers, kBytes, true);
  relay_req.set_host_tiling_mode(tpu_sync::rpc::HOST_TILING_MODE_PRE_TILED);
  auto recv_req = relay_req;
  recv_req.set_is_sender(false);
  relay->StoreSkipTiling(kRelayUuid, recv_req);
  relay->StoreSkipTiling(kRelayUuid, relay_req);
  ABSL_ASSERT_OK(relay->RegisterExpectedLayerChunks(kRelayUuid, {{1, 1}}));
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> relay_l0_buf,
                       relay->GetHostBuffer(0, 0));
  std::memset(relay_l0_buf.data(), 0xC1, kBytes);
  auto push_fut = PushAsync(*relay, relay_req);
  ASSERT_TRUE(dest->layer_received(0).WaitForNotificationWithTimeout(
      absl::Seconds(10)));
  EXPECT_EQ(raw0->d2h_call_count(), 1u);
  EXPECT_EQ(ReadHostLayerBytes(*dest, 0, kBytes),
            std::vector<uint8_t>(kBytes, 0xC1));
  ExpectErr(relay->D2h().status(), absl::StatusCode::kFailedPrecondition,
            {"active_h2h_pushes=1",
             "await the in-flight transfer before bind_weights()/d2h()"});
  ExpectErr(relay->BindWeights({{h0}, {h1}}),
            absl::StatusCode::kFailedPrecondition, {"active_h2h_pushes=1"});
  ABSL_EXPECT_OK(relay->BindWeights({}));
  TF_ASSERT_OK_AND_ASSIGN(absl::Span<uint8_t> relay_l1_buf,
                       relay->GetHostBuffer(1, 0));
  std::memset(relay_l1_buf.data(), 0xD2, kBytes);
  ABSL_ASSERT_OK(relay->OnLayerDataReceived(1, kRelayUuid));
  ABSL_ASSERT_OK(relay->OnDataReceived(kRelayUuid));
  ABSL_ASSERT_OK(push_fut.get());
  ABSL_ASSERT_OK(dest->WaitForTransferCompletion(kRelayUuid));
  EXPECT_EQ(ReadHostLayerBytes(*dest, 1, kBytes),
            std::vector<uint8_t>(kBytes, 0xD2));

  ABSL_ASSERT_OK(relay->BindWeights({{h0}, {h1}}));
  relay->SetSkipTiling(false);
  TF_ASSERT_OK_AND_ASSIGN(auto post_d2h, relay->D2h());
  ABSL_ASSERT_OK(post_d2h.Await());
  ABSL_ASSERT_OK(relay->RegisterExpectedChunks(92501, 1));
  relay->UnbindWeights();
  ABSL_ASSERT_OK(dest->RegisterExpectedChunks(92502, 2));
  ABSL_ASSERT_OK(relay->PushWeightsResharded(
      MakeSimplePushRequest(92502, true, *dest, kNumLayers, kBytes, true)));
  ABSL_ASSERT_OK(dest->WaitForTransferCompletion(92502));
}

TEST_F(WeightSynchronizerTest,
       ConcurrentUnbindDuringSkipD2hPushIsDataRaceFree) {
  constexpr size_t kNumLayers = 2;
  constexpr size_t kBytes = 16384;
  auto raw0 = GatedRawBuffer::Filled(kBytes, 0x10);
  auto raw1 = GatedRawBuffer::Filled(kBytes, 0x20);
  auto h0 = GatedRawBuffer::MakeU8Handle(raw0);
  auto h1 = GatedRawBuffer::MakeU8Handle(raw1);
  TF_ASSERT_OK_AND_ASSIGN(auto source,
                       CreateGatedSource({{h0}, {h1}}, {}, /*parallelism=*/2));
  TF_ASSERT_OK_AND_ASSIGN(
      auto dest, WeightSynchronizerBase::Create(kNumLayers, 1, kBytes, 0));
  source->SetTestOnlyRateLimiters(
      std::make_shared<peregrine::v1::TestOnlyRateLimiter>(8 * 1024 * 1024),
      nullptr);

  for (int iter = 0; iter < 8; ++iter) {
    const uint8_t v0 = static_cast<uint8_t>(0x30 + iter);
    const uint8_t v1 = static_cast<uint8_t>(0x80 + iter);
    raw0->Fill(v0);
    raw1->Fill(v1);
    ABSL_ASSERT_OK(source->BindWeights({{h0}, {h1}}));

    const uint64_t uuid = 93000 + static_cast<uint64_t>(iter);
    ABSL_ASSERT_OK(dest->RegisterExpectedChunks(uuid, kNumLayers));
    TF_ASSERT_OK_AND_ASSIGN(auto d2h_fut, source->D2h());
    auto push_fut = PushAsync(
        *source, MakeSimplePushRequest(uuid, true, *dest, kNumLayers, kBytes));

    ABSL_ASSERT_OK(d2h_fut.Await());
    source->UnbindWeights();
    ABSL_ASSERT_OK(push_fut.get());
    ABSL_ASSERT_OK(dest->WaitForTransferCompletion(uuid));
    EXPECT_EQ(dest->GetHostBufferPtr(0, 0)[0], v0);
    EXPECT_EQ(dest->GetHostBufferPtr(1, 0)[0], v1);
  }
}

}  // namespace
}  // namespace weight_sync
}  // namespace tpu_raiden
