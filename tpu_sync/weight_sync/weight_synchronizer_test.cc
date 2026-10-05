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
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/common/control_pipe/control_pipe_server.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"
#include "tpu_sync/transport/lib/test_only_rate_limiter.h"
#include "tpu_sync/weight_sync/swarm_service.h"
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
  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers_, num_shards_, slice_byte_size_,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  // ws_dest1 and ws_dest2 represent inference server peers
  auto ws_dest1 = std::make_unique<WeightSynchronizerBase>(
      num_layers_, num_shards_, slice_byte_size_,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  auto ws_dest2 = std::make_unique<WeightSynchronizerBase>(
      num_layers_, num_shards_, slice_byte_size_,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

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
  uint8_t* src_host_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(0, 0));
  ASSERT_NE(src_host_ptr, nullptr);
  std::memset(src_host_ptr, 0xAB, slice_byte_size_);

  // Populate inference servers with zeros baseline
  uint8_t* dest1_host_ptr =
      const_cast<uint8_t*>(ws_dest1->GetHostPointer(0, 0));
  uint8_t* dest2_host_ptr =
      const_cast<uint8_t*>(ws_dest2->GetHostPointer(0, 0));
  ASSERT_NE(dest1_host_ptr, nullptr);
  ASSERT_NE(dest2_host_ptr, nullptr);
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

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_sizes,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_sizes,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());

  std::string source_peer =
      "localhost:" + std::to_string(*ws_source->local_port());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  // Populate source layers
  uint8_t* src_l0_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(0, 0));
  uint8_t* src_l1_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(1, 0));
  ASSERT_NE(src_l0_ptr, nullptr);
  ASSERT_NE(src_l1_ptr, nullptr);
  std::memset(src_l0_ptr, 0xAA, slice_byte_sizes[0]);
  std::memset(src_l1_ptr, 0xBB, slice_byte_sizes[1]);

  // Populate dest layers with zeros
  uint8_t* dest_l0_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(0, 0));
  uint8_t* dest_l1_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(1, 0));
  ASSERT_NE(dest_l0_ptr, nullptr);
  ASSERT_NE(dest_l1_ptr, nullptr);
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

TEST_F(WeightSynchronizerTest, CustomLayerNamesInitialization) {
  size_t num_layers = 2;
  size_t num_shards = 1;
  std::vector<size_t> slice_byte_sizes = {16384, 32768};
  std::vector<std::string> custom_names = {"my_layer_0", "my_layer_1"};

  auto ws = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_sizes,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*listener_port=*/std::nullopt,
      /*bind_ip=*/std::nullopt, custom_names);

  EXPECT_EQ(ws->layer_names().size(), 2);
  EXPECT_EQ(ws->layer_names()[0], "my_layer_0");
  EXPECT_EQ(ws->layer_names()[1], "my_layer_1");
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedExactBoundary) {
  size_t num_layers = 1;
  size_t num_shards = 1;
  size_t slice_byte_size = 16384;

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  uint8_t* src_host_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(0, 0));
  uint8_t* dest_host_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(0, 0));
  ASSERT_NE(src_host_ptr, nullptr);
  ASSERT_NE(dest_host_ptr, nullptr);
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

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest1 = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest2 = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest1->local_port().has_value());
  ASSERT_TRUE(ws_dest2->local_port().has_value());
  std::string dest_peer1 =
      "localhost:" + std::to_string(*ws_dest1->local_port());
  std::string dest_peer2 =
      "localhost:" + std::to_string(*ws_dest2->local_port());

  uint8_t* src_host_ptr =
      const_cast<uint8_t*>(ws_source->GetHostPointer(0, 0));
  uint8_t* dest_host_ptr1 =
      const_cast<uint8_t*>(ws_dest1->GetHostPointer(0, 0));
  uint8_t* dest_host_ptr2 =
      const_cast<uint8_t*>(ws_dest2->GetHostPointer(0, 0));
  ASSERT_NE(src_host_ptr, nullptr);
  ASSERT_NE(dest_host_ptr1, nullptr);
  ASSERT_NE(dest_host_ptr2, nullptr);
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

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest1 = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest2 = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest1->local_port().has_value());
  ASSERT_TRUE(ws_dest2->local_port().has_value());
  std::string dest_peer1 =
      "localhost:" + std::to_string(*ws_dest1->local_port());
  std::string dest_peer2 =
      "localhost:" + std::to_string(*ws_dest2->local_port());

  uint8_t* src_host_ptr =
      const_cast<uint8_t*>(ws_source->GetHostPointer(0, 0));
  uint8_t* dest_host_ptr1 =
      const_cast<uint8_t*>(ws_dest1->GetHostPointer(0, 0));
  uint8_t* dest_host_ptr2 =
      const_cast<uint8_t*>(ws_dest2->GetHostPointer(0, 0));
  ASSERT_NE(src_host_ptr, nullptr);
  ASSERT_NE(dest_host_ptr1, nullptr);
  ASSERT_NE(dest_host_ptr2, nullptr);
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

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  uint8_t* src_host_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(0, 0));
  uint8_t* dest_host_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(0, 0));
  ASSERT_NE(src_host_ptr, nullptr);
  ASSERT_NE(dest_host_ptr, nullptr);
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

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, total_payload,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, dst_buffer_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  uint8_t* src_host_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(0, 0));
  uint8_t* dest_host_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(0, 0));
  ASSERT_NE(src_host_ptr, nullptr);
  ASSERT_NE(dest_host_ptr, nullptr);

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

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  uint8_t* src_host_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(0, 0));
  uint8_t* dest_host_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(0, 0));
  ASSERT_NE(src_host_ptr, nullptr);
  ASSERT_NE(dest_host_ptr, nullptr);
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

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

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

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

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

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

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

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

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

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_sizes,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*listener_port=*/std::nullopt,
      /*bind_ip=*/std::nullopt, custom_names);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_sizes,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*listener_port=*/std::nullopt,
      /*bind_ip=*/std::nullopt, custom_names);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  // Populate source layer_B (index 1) with 0xAB
  uint8_t* src_l1_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(1, 0));
  ASSERT_NE(src_l1_ptr, nullptr);
  std::memset(src_l1_ptr, 0xAB, slice_byte_sizes[1]);

  // Populate dest layer_B with zeros
  uint8_t* dest_l1_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(1, 0));
  ASSERT_NE(dest_l1_ptr, nullptr);
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

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_sizes,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*listener_port=*/std::nullopt,
      /*bind_ip=*/std::nullopt, custom_names);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_sizes,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*listener_port=*/std::nullopt,
      /*bind_ip=*/std::nullopt, custom_names);

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

  auto ws_source =
      std::make_unique<WeightSynchronizerBase>(src_buffers, /*local_port=*/0);
  auto ws_dest =
      std::make_unique<WeightSynchronizerBase>(dest_buffers, /*local_port=*/0);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  uint8_t* src_host_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(0, 0));
  ASSERT_NE(src_host_ptr, nullptr);
  std::memset(src_host_ptr, 0xAA, slice_byte_size);

  uint8_t* dest_host_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(0, 0));
  ASSERT_NE(dest_host_ptr, nullptr);
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

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  for (size_t l = 0; l < num_layers; ++l) {
    for (size_t s = 0; s < num_shards; ++s) {
      uint8_t* src_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(l, s));
      uint8_t* dst_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(l, s));
      ASSERT_NE(src_ptr, nullptr);
      ASSERT_NE(dst_ptr, nullptr);
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
      uint8_t* dst_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(l, s));
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
      std::memset(const_cast<uint8_t*>(ws_dest->GetHostPointer(l, s)), 0x00,
                  slice_byte_size);
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
      uint8_t* dst_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(l, s));
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
      std::memset(const_cast<uint8_t*>(ws_dest->GetHostPointer(l, s)), 0x00,
                  slice_byte_size);
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
      uint8_t* dst_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(l, s));
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
      std::memset(const_cast<uint8_t*>(ws_dest->GetHostPointer(l, s)), 0x00,
                  slice_byte_size);
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
      uint8_t* dst_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(l, s));
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
  auto ws_source =
      std::make_unique<WeightSynchronizerBase>(src_buffers, /*local_port=*/0);
  auto ws_dest =
      std::make_unique<WeightSynchronizerBase>(dest_buffers, /*local_port=*/0);

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
  auto ws = std::make_unique<WeightSynchronizerBase>(
      num_layers_, num_shards_, slice_byte_size_,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

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

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      acquire(src_pjrt_buffer.get()), /*local_port=*/0);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      acquire(dest_pjrt_buffer.get()), /*local_port=*/0);
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
  ASSERT_OK(new_dest_pjrt_buffer
                ->CopyRawToHost(readback.data(), 0, slice_byte_size)
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

      auto ws =
          std::make_unique<WeightSynchronizerBase>(buffers, /*local_port=*/0);
      return WSWrapper{std::move(pjrt_buffer), std::move(ws)};
    }
  } runner{client.get(), memory_space};

  // Case 1: Tiled Layout, perfect division (no padding), identical layouts.
  {
    xla::Layout layout =
        xla::LayoutUtil::MakeLayout({1, 0}, {xla::Tile({4, 4})});
    std::vector<uint8_t> src_device_placeholder;
    auto src_ws = runner.CreateWS(xla::PrimitiveType::F32, {8, 8}, layout,
                                  src_device_placeholder);

    float* src_host = reinterpret_cast<float*>(
        const_cast<uint8_t*>(src_ws.ws->GetHostPointer(0, 0)));
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

      auto ws =
          std::make_unique<WeightSynchronizerBase>(buffers, /*local_port=*/0);
      return WSWrapper{std::move(pjrt_buffer), std::move(ws)};
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

    float* src_host = reinterpret_cast<float*>(
        const_cast<uint8_t*>(src_ws.ws->GetHostPointer(0, 0)));
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
  auto ws =
      std::make_unique<WeightSynchronizerBase>(layer_buffers, /*local_port=*/0);

  auto value_at = [](int l, int s, int i) {
    return static_cast<float>(1000 * l + 100 * s + i);
  };
  for (int l = 0; l < kNumLayers; ++l) {
    for (int s = 0; s < kNumShards; ++s) {
      float* host = reinterpret_cast<float*>(ws->GetHostPointer(l, s));
      ASSERT_NE(host, nullptr);
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

      auto ws =
          std::make_unique<WeightSynchronizerBase>(buffers, /*local_port=*/0);
      return WSWrapper{std::move(pjrt_buffer), std::move(ws)};
    }
  } runner{client.get(), memory_space};

  xla::Layout layout = xla::LayoutUtil::MakeLayout({1, 0}, {xla::Tile({4, 4})});
  std::vector<uint8_t> src_device_placeholder;
  auto src_ws = runner.CreateWS(xla::PrimitiveType::F32, {8, 8}, layout,
                                src_device_placeholder);

  // 1. Verify tiled_ptr is NOT pre-allocated during initialization (lazy
  // allocation saves memory)
  EXPECT_EQ(src_ws.ws->GetTiledPointer(0, 0), nullptr);

  float* src_host = reinterpret_cast<float*>(
      const_cast<uint8_t*>(src_ws.ws->GetHostPointer(0, 0)));
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

      auto ws =
          std::make_unique<WeightSynchronizerBase>(buffers, /*local_port=*/0);
      return WSWrapper{std::move(pjrt_buffer), std::move(ws)};
    }
  } runner{client.get(), memory_space};

  // 1D tensor: shape {64}, F32, standard hardware 2D tile {8, 128}
  // Logical size = 64 * 4 = 256 bytes.
  // Physical tile size = 8 * 128 * 4 = 4096 bytes.
  xla::Layout layout = xla::LayoutUtil::MakeLayout({0}, {xla::Tile({8, 128})});
  std::vector<uint8_t> device_placeholder;
  auto ws_wrap = runner.CreateWS(xla::PrimitiveType::F32, {64}, layout,
                                 device_placeholder);

  float* host_ptr = reinterpret_cast<float*>(
      const_cast<uint8_t*>(ws_wrap.ws->GetHostPointer(0, 0)));
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

TEST_F(WeightSynchronizerTest, DrainPendingH2dDrainsActivePendingFutures) {
  auto ws = std::make_unique<WeightSynchronizerBase>(
      /*num_layers=*/2, /*num_shards=*/1, slice_byte_size_,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/2, /*listener_port=*/0, /*bind_ip=*/std::nullopt,
      /*layer_names=*/std::vector<std::string>{"layer_0", "layer_1"},
      /*auto_h2d=*/true);

  uint64_t uuid = 54321;
  EXPECT_TRUE(ws->OnLayerDataReceived(0, uuid).ok());
  EXPECT_TRUE(ws->OnLayerDataReceived(1, uuid).ok());

  // DrainPendingH2d should drain all scheduled futures and mark uuid completed
  ws->DrainPendingH2d();

  EXPECT_TRUE(ws->WaitForTransferCompletion(uuid).ok());
  ws->ForgetPushProgress(uuid);
}

TEST_F(WeightSynchronizerTest, OnDataReceivedWaitsForAllExpectedLayers) {
  auto ws = std::make_unique<WeightSynchronizerBase>(
      /*num_layers=*/3, /*num_shards=*/1, slice_byte_size_,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/2, /*listener_port=*/0, /*bind_ip=*/std::nullopt,
      /*layer_names=*/
      std::vector<std::string>{"layer_0", "layer_1", "layer_2"},
      /*auto_h2d=*/true);

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

TEST_F(WeightSynchronizerTest, GetHostPointerAndSizeNonContiguousGlobalShards) {
  const size_t num_layers = 2;
  const size_t num_shards = 4;
  const size_t slice_size = 1024;
  auto ws = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  // Configure non-contiguous global shards, e.g., NUMA node 0 managing shards
  // {0, 2, 4, 6}
  ws->SetGlobalShardIndices({0, 2, 4, 6});

  // Check that GetHostPointer correctly maps global shard indices to internal
  // slots 0, 1, 2, 3
  uint8_t* ptr_shard0 = ws->GetHostPointer(0, 0);
  uint8_t* ptr_shard2 = ws->GetHostPointer(0, 2);
  uint8_t* ptr_shard4 = ws->GetHostPointer(0, 4);
  uint8_t* ptr_shard6 = ws->GetHostPointer(0, 6);

  ASSERT_NE(ptr_shard0, nullptr);
  ASSERT_NE(ptr_shard2, nullptr);
  ASSERT_NE(ptr_shard4, nullptr);
  ASSERT_NE(ptr_shard6, nullptr);

  EXPECT_EQ(ptr_shard0, const_cast<uint8_t*>(ws->GetHostBufferPtr(0, 0)));
  EXPECT_EQ(ptr_shard2, const_cast<uint8_t*>(ws->GetHostBufferPtr(0, 1)));
  EXPECT_EQ(ptr_shard4, const_cast<uint8_t*>(ws->GetHostBufferPtr(0, 2)));
  EXPECT_EQ(ptr_shard6, const_cast<uint8_t*>(ws->GetHostBufferPtr(0, 3)));

  // Ensure all 4 resolved pointers are distinct
  EXPECT_NE(ptr_shard0, ptr_shard2);
  EXPECT_NE(ptr_shard0, ptr_shard4);
  EXPECT_NE(ptr_shard0, ptr_shard6);
  EXPECT_NE(ptr_shard2, ptr_shard4);
  EXPECT_NE(ptr_shard2, ptr_shard6);
  EXPECT_NE(ptr_shard4, ptr_shard6);

  // Check sizes
  EXPECT_EQ(ws->GetHostSize(0, 0), slice_size);
  EXPECT_EQ(ws->GetHostSize(0, 2), slice_size);
  EXPECT_EQ(ws->GetHostSize(0, 4), slice_size);
  EXPECT_EQ(ws->GetHostSize(0, 6), slice_size);

  // Const overloads
  const auto* const_ws = ws.get();
  EXPECT_EQ(const_ws->GetHostPointer(0, 0), ws->GetHostBufferPtr(0, 0));
  EXPECT_EQ(const_ws->GetHostPointer(0, 2), ws->GetHostBufferPtr(0, 1));
  EXPECT_EQ(const_ws->GetHostPointer(0, 4), ws->GetHostBufferPtr(0, 2));
  EXPECT_EQ(const_ws->GetHostPointer(0, 6), ws->GetHostBufferPtr(0, 3));
  EXPECT_EQ(const_ws->GetHostSize(0, 0), slice_size);
  EXPECT_EQ(const_ws->GetHostSize(0, 2), slice_size);

  // Verify memory writes do not collide
  std::memset(ptr_shard0, 0x11, slice_size);
  std::memset(ptr_shard2, 0x22, slice_size);
  std::memset(ptr_shard4, 0x33, slice_size);
  std::memset(ptr_shard6, 0x44, slice_size);

  EXPECT_EQ(ws->GetHostBufferPtr(0, 0)[0], 0x11);
  EXPECT_EQ(ws->GetHostBufferPtr(0, 1)[0], 0x22);
  EXPECT_EQ(ws->GetHostBufferPtr(0, 2)[0], 0x33);
  EXPECT_EQ(ws->GetHostBufferPtr(0, 3)[0], 0x44);
}

TEST_F(WeightSynchronizerTest, GetHostPointerAndSizeLocalAndGlobalIndices) {
  const size_t num_layers = 2;
  const size_t num_shards = 4;
  const size_t slice_size = 1024;
  auto ws = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  // Configure non-contiguous global shards, e.g., Host 1 managing global shards
  // {2, 3, 6, 7} with local slots {0, 1, 2, 3}.
  ws->SetGlobalShardIndices({2, 3, 6, 7});
  ws->SetLocalShardIndices({0, 1, 2, 3});

  // Local slot indices should map directly to internal slots 0, 1, 2, 3
  // without being falsely remapped by global shard indices {2, 3, 6, 7}.
  uint8_t* ptr_slot0 = ws->GetHostPointer(0, 0);
  uint8_t* ptr_slot1 = ws->GetHostPointer(0, 1);
  uint8_t* ptr_slot2 = ws->GetHostPointer(0, 2);
  uint8_t* ptr_slot3 = ws->GetHostPointer(0, 3);

  ASSERT_NE(ptr_slot0, nullptr);
  ASSERT_NE(ptr_slot1, nullptr);
  ASSERT_NE(ptr_slot2, nullptr);
  ASSERT_NE(ptr_slot3, nullptr);

  EXPECT_EQ(ptr_slot0, const_cast<uint8_t*>(ws->GetHostBufferPtr(0, 0)));
  EXPECT_EQ(ptr_slot1, const_cast<uint8_t*>(ws->GetHostBufferPtr(0, 1)));
  EXPECT_EQ(ptr_slot2, const_cast<uint8_t*>(ws->GetHostBufferPtr(0, 2)));
  EXPECT_EQ(ptr_slot3, const_cast<uint8_t*>(ws->GetHostBufferPtr(0, 3)));

  // Global shard indices not matching local slots (e.g. 6 and 7) should still
  // map to their corresponding internal slots (slots 2 and 3).
  uint8_t* ptr_global6 = ws->GetHostPointer(0, 6);
  uint8_t* ptr_global7 = ws->GetHostPointer(0, 7);
  EXPECT_EQ(ptr_global6, const_cast<uint8_t*>(ws->GetHostBufferPtr(0, 2)));
  EXPECT_EQ(ptr_global7, const_cast<uint8_t*>(ws->GetHostBufferPtr(0, 3)));

  // Const overloads
  const auto* const_ws = ws.get();
  EXPECT_EQ(const_ws->GetHostPointer(0, 2), ws->GetHostBufferPtr(0, 2));
  EXPECT_EQ(const_ws->GetHostPointer(0, 3), ws->GetHostBufferPtr(0, 3));
  EXPECT_EQ(const_ws->GetHostSize(0, 2), slice_size);
  EXPECT_EQ(const_ws->GetHostSize(0, 3), slice_size);
}

TEST_F(WeightSynchronizerTest,
       PushWeightsReshardedGlobalShardIndicesPrioritized) {
  const size_t num_layers = 1;
  const size_t num_shards = 4;
  const size_t slice_byte_size = 256;

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

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
    uint8_t* src_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(0, s));
    uint8_t* dst_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(0, s));
    ASSERT_NE(src_ptr, nullptr);
    ASSERT_NE(dst_ptr, nullptr);
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

TEST_F(WeightSynchronizerTest,
       PushWeightsReshardedLocalShardIndicesFallback) {
  const size_t num_layers = 1;
  const size_t num_shards = 4;
  const size_t slice_byte_size = 256;

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

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
    uint8_t* src_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(0, s));
    uint8_t* dst_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(0, s));
    ASSERT_NE(src_ptr, nullptr);
    ASSERT_NE(dst_ptr, nullptr);
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
       PushWeightsReshardedLocalShardIndicesNotShadowedByGlobalIndices) {
  const size_t num_layers = 1;
  const size_t num_shards = 4;
  const size_t slice_byte_size = 256;

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

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

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

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
  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

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

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

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

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = absl::StrCat("127.0.0.1:", *ws_dest->local_port());

  // Configure a simulated rate limiter of 10 MB/s (~50 ms per 500KB push)
  // to guarantee substantial concurrency overlap.
  auto egress_limiter =
      std::make_shared<transport::lib::TestOnlyRateLimiter>(10000000);
  ws_source->SetTestOnlyRateLimiters(egress_limiter, /*ingress=*/nullptr);

  auto make_request = [&](uint64_t uuid) {
    tpu_sync::rpc::StartTransferRequest req;
    req.set_skip_d2h(true);
    req.set_uuid(uuid);
    auto* schedules = req.mutable_shard_push_schedules();
    auto* entry = (*schedules)[0].add_entries();
    entry->set_dst_peer(dest_peer);
    entry->set_dst_shard_idx(0);
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
    auto s = ws_source->PushWeightsResharded(make_request(uuid1));
    elapsed1_ms = absl::ToDoubleMilliseconds(absl::Now() - t0);
    return s;
  });

  auto fut2 = std::async(std::launch::async, [&]() -> absl::Status {
    start_notification.WaitForNotification();
    auto t0 = absl::Now();
    auto s = ws_source->PushWeightsResharded(make_request(uuid2));
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

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers_, num_shards_, slice_byte_size_,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers_, num_shards_, slice_byte_size_,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

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

  auto ws_source =
      std::make_unique<WeightSynchronizerBase>(src_buffers, /*local_port=*/0);
  ws_source->SetPipelineGroupSize(1);

  auto ws_dest1 = std::make_unique<WeightSynchronizerBase>(
      kNumLayers, /*num_shards=*/1, kSliceByteSize,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest2 = std::make_unique<WeightSynchronizerBase>(
      kNumLayers, /*num_shards=*/1, kSliceByteSize,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  ASSERT_TRUE(ws_dest1->local_port().has_value());
  ASSERT_TRUE(ws_dest2->local_port().has_value());
  std::string dest1_peer = absl::StrCat("127.0.0.1:", *ws_dest1->local_port());
  std::string dest2_peer = absl::StrCat("127.0.0.1:", *ws_dest2->local_port());

  // Initialize source and destination host staging buffers with zeros so only
  // completed D2H copies can populate 0x30 + l.
  for (size_t l = 0; l < kNumLayers; ++l) {
    std::memset(const_cast<uint8_t*>(ws_source->GetHostPointer(l, 0)), 0x00,
                kSliceByteSize);
    std::memset(const_cast<uint8_t*>(ws_dest1->GetHostPointer(l, 0)), 0x00,
                kSliceByteSize);
    std::memset(const_cast<uint8_t*>(ws_dest2->GetHostPointer(l, 0)), 0x00,
                kSliceByteSize);
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
    const uint8_t* dst1_ptr = ws_dest1->GetHostPointer(l, 0);
    const uint8_t* dst2_ptr = ws_dest2->GetHostPointer(l, 0);
    for (size_t b = 0; b < kSliceByteSize; ++b) {
      EXPECT_EQ(dst1_ptr[b], expected) << "dest1 layer " << l << " byte " << b;
      EXPECT_EQ(dst2_ptr[b], expected) << "dest2 layer " << l << " byte " << b;
    }
  }
}

TEST_F(WeightSynchronizerTest, StartBundlePullTransferValidation) {
  WeightSynchronizerBase engine(num_layers_, num_shards_, slice_byte_size_,
                                /*local_port=*/0,
                                /*host_blocks_to_allocate=*/1);

  // 1. Request without pull_config.
  tpu_sync::rpc::StartTransferRequest req1;
  absl::Status s1 = engine.StartBundlePullTransfer(req1);
  EXPECT_FALSE(s1.ok());
  EXPECT_EQ(s1.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(s1.message(), ::testing::HasSubstr("enable_bundle_pull"));

  // 2. Request with enable_bundle_pull=false.
  tpu_sync::rpc::StartTransferRequest req2;
  req2.mutable_pull_config()->set_enable_bundle_pull(false);
  absl::Status s2 = engine.StartBundlePullTransfer(req2);
  EXPECT_FALSE(s2.ok());
  EXPECT_EQ(s2.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(s2.message(), ::testing::HasSubstr("enable_bundle_pull"));

  // 3. Request with empty controller_address.
  tpu_sync::rpc::StartTransferRequest req3;
  req3.mutable_pull_config()->set_enable_bundle_pull(true);
  req3.mutable_pull_config()->set_controller_address("");
  absl::Status s3 = engine.StartBundlePullTransfer(req3);
  EXPECT_FALSE(s3.ok());
  EXPECT_EQ(s3.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(s3.message(), ::testing::HasSubstr("controller_address"));
}

TEST_F(WeightSynchronizerTest, BundlePullTransferExecutionE2e) {
  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers_, num_shards_, slice_byte_size_,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers_, num_shards_, slice_byte_size_,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());

  uint8_t* src_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(0, 0));
  ASSERT_NE(src_ptr, nullptr);
  std::memset(src_ptr, 0x5A, slice_byte_size_);

  uint8_t* dst_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(0, 0));
  ASSERT_NE(dst_ptr, nullptr);
  std::memset(dst_ptr, 0x00, slice_byte_size_);

  auto swarm = std::make_shared<SwarmService>();
  ControlPipeConfig cfg;
  cfg.requested_port = 0;
  auto ctrl_server = CreateControlPipeServer(cfg);
  ctrl_server->dispatcher()
      .RegisterHandler<tpu_sync::rpc::ControlRequest,
                       tpu_sync::rpc::ControlResponse>(
          [&swarm](const ControlContext&,
                   const tpu_sync::rpc::ControlRequest& req)
              -> absl::StatusOr<tpu_sync::rpc::ControlResponse> {
            tpu_sync::rpc::ControlResponse resp;
            resp.set_success(true);
            if (req.command() == tpu_sync::rpc::ControlRequest::
                                     COMMAND_ACQUIRE_BUNDLE_PULL_TOKEN) {
              auto tok = swarm->AcquireBundlePullToken(
                  req.acquire_bundle_pull_token_request());
              if (tok.ok()) {
                *resp.mutable_acquire_bundle_pull_token_response() =
                    std::move(*tok);
              } else {
                resp.set_success(false);
                resp.set_message(std::string(tok.status().message()));
              }
            } else if (req.command() ==
                       tpu_sync::rpc::ControlRequest::
                           COMMAND_REGISTER_BUNDLE_AVAILABILITY) {
              auto reg = swarm->RegisterBundleAvailability(
                  req.register_bundle_availability_request());
              if (reg.ok()) {
                *resp.mutable_register_bundle_availability_response() =
                    std::move(*reg);
              } else {
                resp.set_success(false);
                resp.set_message(std::string(reg.status().message()));
              }
            }
            return resp;
          });
  auto port_or = ctrl_server->Start(0);
  ASSERT_TRUE(port_or.ok());
  const std::string ctrl_addr = "127.0.0.1:" + std::to_string(*port_or);

  const uint64_t kUuid = 98765;
  SwarmService::Participant p0;
  p0.unit.set_job_name("sampler");
  p0.unit.set_job_replica_id("0");
  p0.host_data_endpoints.push_back("127.0.0.1:" +
                                   std::to_string(*ws_source->local_port()));

  SwarmService::Participant p1;
  p1.unit.set_job_name("sampler");
  p1.unit.set_job_replica_id("1");
  p1.host_data_endpoints.push_back("127.0.0.1:" +
                                   std::to_string(*ws_dest->local_port()));

  ASSERT_TRUE(swarm
                  ->StartSession("test_bundle_pull", kUuid, /*num_bundles=*/1,
                                 {p0, p1}, SwarmService::SessionConfig{})
                  .ok());

  tpu_sync::rpc::RegisterBundleAvailabilityRequest reg_req;
  reg_req.set_req_id("test_bundle_pull");
  reg_req.set_uuid(static_cast<int64_t>(kUuid));
  reg_req.set_bundle_index(0);
  reg_req.mutable_unit()->set_job_name("sampler");
  reg_req.mutable_unit()->set_job_replica_id("0");
  ASSERT_TRUE(swarm->RegisterBundleAvailability(reg_req).ok());

  tpu_sync::rpc::StartTransferRequest pull_req;
  pull_req.set_req_id("test_bundle_pull");
  pull_req.set_uuid(kUuid);
  pull_req.mutable_pull_config()->set_enable_bundle_pull(true);
  pull_req.mutable_pull_config()->set_controller_address(ctrl_addr);
  pull_req.mutable_pull_config()->mutable_target_unit()->set_job_name(
      "sampler");
  pull_req.mutable_pull_config()->mutable_target_unit()->set_job_replica_id(
      "1");
  pull_req.mutable_pull_config()->set_host_idx(0);

  auto* bundle = pull_req.add_variable_bundles();
  bundle->set_bundle_index(0);
  bundle->set_bundle_name("layer0");
  auto* var = bundle->add_variables();
  var->set_layer_idx(0);

  ASSERT_TRUE(ws_dest->StartBundlePullTransfer(pull_req).ok());
  ASSERT_TRUE(ws_dest->WaitForTransferCompletion(kUuid).ok());

  for (size_t i = 0; i < slice_byte_size_; ++i) {
    EXPECT_EQ(dst_ptr[i], 0x5A) << "Mismatch at byte " << i;
  }

  ctrl_server->Stop();
}

class TestableWeightSynchronizer : public WeightSynchronizerBase {
 public:
  using WeightSynchronizerBase::GetOrCreateControlPipeClient;
  using WeightSynchronizerBase::WeightSynchronizerBase;
};

TEST_F(WeightSynchronizerTest, ControlPipeClientCaching) {
  TestableWeightSynchronizer engine(num_layers_, num_shards_, slice_byte_size_,
                                    /*local_port=*/0,
                                    /*host_blocks_to_allocate=*/1);
  ControlPipeClient* client1 =
      engine.GetOrCreateControlPipeClient("127.0.0.1:10001");
  ControlPipeClient* client1_again =
      engine.GetOrCreateControlPipeClient("127.0.0.1:10001");
  ControlPipeClient* client2 =
      engine.GetOrCreateControlPipeClient("127.0.0.1:10002");

  EXPECT_NE(client1, nullptr);
  EXPECT_NE(client2, nullptr);
  EXPECT_EQ(client1, client1_again);
  EXPECT_NE(client1, client2);
}

}  // namespace
}  // namespace weight_sync
}  // namespace tpu_raiden
