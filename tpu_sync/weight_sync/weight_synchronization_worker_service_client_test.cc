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

#include "tpu_sync/weight_sync/weight_synchronization_worker_service_client.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/weight_synchronization_worker_service.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace weight_sync {
namespace {

using ::absl_testing::IsOk;
using ::tpu_sync::rpc::ControlRequest;
using ::tpu_sync::rpc::ShardPushEntryProto;
using ::tpu_sync::rpc::ShardPushScheduleProto;
using ::tpu_sync::rpc::StartTransferRequest;

TEST(WeightSynchronizationWorkerServiceClientTest, HandleControlSuccess) {
  WeightSynchronizerBase engine(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
      /*parallelism=*/1, /*listener_port=*/std::nullopt);

  WeightSynchronizationWorkerService service(&engine, /*server_port=*/0);
  ASSERT_GT(service.server_port(), 0);

  WeightSynchronizationWorkerServiceClient client(
      absl::StrCat("localhost:", service.server_port()));

  WeightSynchronizerBase dst_engine(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*listener_port=*/std::nullopt);
  ASSERT_TRUE(dst_engine.local_port().has_value());

  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_START_TRANSFER);
  req.add_peers(absl::StrCat("127.0.0.1:", *dst_engine.local_port()));

  auto future = client.HandleControl(req);
  auto response_or = future.Await();
  ASSERT_THAT(response_or, IsOk());
  EXPECT_TRUE(response_or->success());
}

TEST(WeightSynchronizationWorkerServiceClientTest,
     StartTransferConvenienceMethod) {
  WeightSynchronizerBase engine(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
      /*parallelism=*/1, /*listener_port=*/std::nullopt);

  WeightSynchronizationWorkerService service(&engine, /*server_port=*/0);
  ASSERT_GT(service.server_port(), 0);

  WeightSynchronizationWorkerServiceClient client(
      absl::StrCat("localhost:", service.server_port()));

  WeightSynchronizerBase dst_engine(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*listener_port=*/std::nullopt);
  ASSERT_TRUE(dst_engine.local_port().has_value());

  StartTransferRequest start_req;
  start_req.set_is_sender(true);
  std::vector<std::string> peers = {
      absl::StrCat("127.0.0.1:", *dst_engine.local_port())};

  auto future = client.StartTransfer(start_req, peers);
  auto response_or = future.Await();
  ASSERT_THAT(response_or, IsOk());
  EXPECT_TRUE(response_or->success());
}

TEST(WeightSynchronizationWorkerServiceClientTest,
     PushWeightsReshardedUsingClients) {
  WeightSynchronizerBase src_engine(
      /*num_layers=*/1, /*num_shards=*/4, /*slice_byte_size=*/16,
      /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
      /*parallelism=*/1, /*listener_port=*/std::nullopt);

  WeightSynchronizationWorkerService src_service(&src_engine,
                                                 /*server_port=*/0);
  ASSERT_GT(src_service.server_port(), 0);

  WeightSynchronizerBase dst_engine(
      /*num_layers=*/1, /*num_shards=*/4, /*slice_byte_size=*/16,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*listener_port=*/std::nullopt);
  ASSERT_TRUE(dst_engine.local_port().has_value());

  WeightSynchronizationWorkerService dst_service(&dst_engine,
                                                 /*server_port=*/0);
  ASSERT_GT(dst_service.server_port(), 0);

  std::string dst_peer = absl::StrCat("127.0.0.1:", *dst_engine.local_port());

  // Populate source buffers
  std::vector<std::vector<uint8_t>> src_data = {
      {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27},
      {4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31},
      {32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59},
      {36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63},
  };

  for (size_t i = 0; i < 4; ++i) {
    uint8_t* ptr = src_engine.GetHostPointer(0, i);
    ASSERT_NE(ptr, nullptr);
    std::memcpy(ptr, src_data[i].data(), 16);
  }

  uint64_t test_uuid = 123456789;

  WeightSynchronizationWorkerServiceClient dst_client(
      absl::StrCat("localhost:", dst_service.server_port()));
  WeightSynchronizationWorkerServiceClient src_client(
      absl::StrCat("localhost:", src_service.server_port()));

  // 1. Arm destination receiver using client
  StartTransferRequest dst_start_req;
  dst_start_req.set_is_sender(false);
  dst_start_req.set_uuid(test_uuid);
  dst_start_req.set_expected_block_count(8);

  auto dst_resp_or = dst_client.StartTransfer(dst_start_req).Await();
  ASSERT_THAT(dst_resp_or, IsOk());
  EXPECT_TRUE(dst_resp_or->success());

  // 2. Dispatch sender push schedules using client
  StartTransferRequest src_start_req;
  src_start_req.set_is_sender(true);
  src_start_req.set_uuid(test_uuid);

  auto& push_schedules = *src_start_req.mutable_shard_push_schedules();

  // S0 -> D0
  ShardPushScheduleProto s0_sched;
  for (int r = 0; r < 4; ++r) {
    ShardPushEntryProto* e = s0_sched.add_entries();
    e->set_dst_peer(dst_peer);
    e->set_dst_shard_idx(0);
    e->set_src_offset_bytes(r * 4);
    e->set_dst_offset_bytes(r * 2);
    e->set_size_bytes(2);
    e->set_layer_idx(0);
  }
  push_schedules[0] = s0_sched;

  // S2 -> D0
  ShardPushScheduleProto s2_sched;
  for (int r = 0; r < 4; ++r) {
    ShardPushEntryProto* e = s2_sched.add_entries();
    e->set_dst_peer(dst_peer);
    e->set_dst_shard_idx(0);
    e->set_src_offset_bytes(r * 4);
    e->set_dst_offset_bytes(8 + r * 2);
    e->set_size_bytes(2);
    e->set_layer_idx(0);
  }
  push_schedules[2] = s2_sched;

  auto src_resp_or = src_client.StartTransfer(src_start_req).Await();
  ASSERT_THAT(src_resp_or, IsOk());
  EXPECT_TRUE(src_resp_or->success());

  // Verify Destination Shard 0 host memory matches resharded layout
  uint8_t* dst_ptr = dst_engine.GetHostPointer(0, 0);
  ASSERT_NE(dst_ptr, nullptr);

  std::vector<uint8_t> expected_d0 = {0,  1,  8,  9,  16, 17, 24, 25,
                                      32, 33, 40, 41, 48, 49, 56, 57};
  for (size_t k = 0; k < 16; ++k) {
    EXPECT_EQ(dst_ptr[k], expected_d0[k]) << "Mismatch at byte " << k;
  }
}

TEST(WeightSynchronizationWorkerServiceClientTest, ShutdownSuccess) {
  WeightSynchronizerBase engine(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
      /*parallelism=*/1, /*listener_port=*/std::nullopt);

  WeightSynchronizationWorkerService service(&engine, /*server_port=*/0);
  EXPECT_TRUE(service.is_active());

  WeightSynchronizationWorkerServiceClient client(
      absl::StrCat("localhost:", service.server_port()));

  auto response_or = client.Shutdown().Await();
  ASSERT_THAT(response_or, IsOk());
  EXPECT_TRUE(response_or->success());

  for (int i = 0; i < 100 && service.is_active(); ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_FALSE(service.is_active());
}

TEST(WeightSynchronizationWorkerServiceClientTest, ServerErrorPropagation) {
  WeightSynchronizerBase engine(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
      /*parallelism=*/1, /*listener_port=*/std::nullopt);

  WeightSynchronizationWorkerService service(&engine, /*server_port=*/0);
  ASSERT_GT(service.server_port(), 0);

  WeightSynchronizationWorkerServiceClient client(
      absl::StrCat("localhost:", service.server_port()));

  // Send an invalid receiver START_TRANSFER request with invalid
  // expected_block_count (0)
  StartTransferRequest start_req;
  start_req.set_is_sender(false);
  start_req.set_expected_block_count(0);

  auto response_or = client.StartTransfer(start_req).Await();
  EXPECT_TRUE(!response_or.ok() || !response_or->success());
}

}  // namespace
}  // namespace weight_sync
}  // namespace tpu_raiden
