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

// Branched from:
// //tpu_sync/weight_sync/weight_synchronizer_listener_test.cc

#include "tpu_sync/weight_sync/weight_synchronization_worker_service.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/status.h"
#include "tpu_sync/rpc/raiden_service.grpc.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace weight_sync {
namespace {

using ::tpu_sync::rpc::ControlRequest;
using ::tpu_sync::rpc::ControlResponse;
using ::tpu_sync::rpc::ShardPushEntryProto;
using ::tpu_sync::rpc::ShardPushScheduleProto;
using ::tpu_sync::rpc::StartTransferRequest;
using ::tpu_sync::rpc::WeightSynchronizationWorkerService;

std::unique_ptr<WeightSynchronizationWorkerService::Stub> CreateStub(int port) {
  std::string target = absl::StrCat("localhost:", port);
  auto channel =
      grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
  return WeightSynchronizationWorkerService::NewStub(channel);
}

TEST(WeightSynchronizationWorkerServiceTest, PushWeightsCommandSuccess) {
  WeightSynchronizerBase engine(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
      /*parallelism=*/1, /*listener_port=*/std::nullopt);

  tpu_raiden::weight_sync::WeightSynchronizationWorkerService service(
      &engine, /*server_port=*/0);
  ASSERT_GT(service.server_port(), 0);
  EXPECT_TRUE(service.is_active());

  WeightSynchronizerBase dst_engine(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*listener_port=*/std::nullopt);
  ASSERT_TRUE(dst_engine.local_port().has_value());

  auto stub = CreateStub(service.server_port());

  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_START_TRANSFER);
  req.add_peers("127.0.0.1:" + std::to_string(*dst_engine.local_port()));

  ControlResponse resp;
  grpc::ClientContext context;
  grpc::Status status = stub->HandleControl(&context, req, &resp);

  ASSERT_TRUE(status.ok()) << status.error_message();
  EXPECT_TRUE(resp.success()) << resp.message();
}

TEST(WeightSynchronizationWorkerServiceTest, ShutdownCommandStopsService) {
  WeightSynchronizerBase engine(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
      /*parallelism=*/1, /*listener_port=*/std::nullopt);

  tpu_raiden::weight_sync::WeightSynchronizationWorkerService service(
      &engine, /*server_port=*/0);
  EXPECT_TRUE(service.is_active());

  auto stub = CreateStub(service.server_port());

  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_SHUTDOWN);

  ControlResponse resp;
  grpc::ClientContext context;
  grpc::Status status = stub->HandleControl(&context, req, &resp);

  ASSERT_TRUE(status.ok()) << status.error_message();
  EXPECT_TRUE(resp.success());

  for (int i = 0; i < 100 && service.is_active(); ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_FALSE(service.is_active());
}

TEST(WeightSynchronizationWorkerServiceTest, PushWeightsReshardedSuccess) {
  WeightSynchronizerBase src_engine(
      /*num_layers=*/1, /*num_shards=*/4, /*slice_byte_size=*/16,
      /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
      /*parallelism=*/1, /*listener_port=*/std::nullopt);

  tpu_raiden::weight_sync::WeightSynchronizationWorkerService src_service(
      &src_engine, /*server_port=*/0);
  ASSERT_GT(src_service.server_port(), 0);

  WeightSynchronizerBase dst_engine(
      /*num_layers=*/1, /*num_shards=*/4, /*slice_byte_size=*/16,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*listener_port=*/std::nullopt);
  ASSERT_TRUE(dst_engine.local_port().has_value());

  tpu_raiden::weight_sync::WeightSynchronizationWorkerService dst_service(
      &dst_engine, /*server_port=*/0);
  ASSERT_GT(dst_service.server_port(), 0);

  std::string dst_peer =
      "127.0.0.1:" + std::to_string(*dst_engine.local_port());

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

  uint64_t test_uuid = 98765;

  // 1. Send START_TRANSFER (Receiver) to destination service
  auto dst_stub = CreateStub(dst_service.server_port());

  ControlRequest dst_req;
  dst_req.set_command(ControlRequest::COMMAND_START_TRANSFER);
  StartTransferRequest* dst_start_req =
      dst_req.mutable_start_transfer_request();
  dst_start_req->set_is_sender(false);
  dst_start_req->set_uuid(test_uuid);
  dst_start_req->set_expected_block_count(8);

  ControlResponse dst_resp;
  grpc::ClientContext dst_context;
  grpc::Status dst_status =
      dst_stub->HandleControl(&dst_context, dst_req, &dst_resp);

  ASSERT_TRUE(dst_status.ok()) << dst_status.error_message();
  EXPECT_TRUE(dst_resp.success()) << dst_resp.message();

  // 2. Send START_TRANSFER (Sender) to source service
  auto src_stub = CreateStub(src_service.server_port());

  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_START_TRANSFER);

  StartTransferRequest* start_req = req.mutable_start_transfer_request();
  start_req->set_is_sender(true);
  start_req->set_uuid(test_uuid);

  // Construct precise resharding push schedules for S0 and S2 pushing to D0
  auto& push_schedules = *start_req->mutable_shard_push_schedules();

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

  ControlResponse resp;
  grpc::ClientContext src_context;
  grpc::Status src_status = src_stub->HandleControl(&src_context, req, &resp);

  ASSERT_TRUE(src_status.ok()) << src_status.error_message();
  EXPECT_TRUE(resp.success()) << resp.message();

  // Verify Destination Shard 0 final host memory!
  uint8_t* dst_ptr = dst_engine.GetHostPointer(0, 0);
  ASSERT_NE(dst_ptr, nullptr);

  std::vector<uint8_t> expected_d0 = {0,  1,  8,  9,  16, 17, 24, 25,
                                      32, 33, 40, 41, 48, 49, 56, 57};
  for (size_t k = 0; k < 16; ++k) {
    EXPECT_EQ(dst_ptr[k], expected_d0[k]) << "Mismatch at byte " << k;
  }
}

}  // namespace
}  // namespace weight_sync
}  // namespace tpu_raiden
