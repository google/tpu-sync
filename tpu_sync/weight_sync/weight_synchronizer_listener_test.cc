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

#include "tpu_sync/weight_sync/weight_synchronizer_listener.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/weight_synchronization_worker_service_client.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace weight_sync {

using ::tpu_sync::rpc::ControlRequest;
using ::tpu_sync::rpc::ControlResponse;
using ::tpu_sync::rpc::ShardPushEntryProto;
using ::tpu_sync::rpc::ShardPushScheduleProto;
using ::tpu_sync::rpc::StartTransferRequest;

namespace {

int ConnectToListenerPort(int port) {
  int sock = socket(AF_INET6, SOCK_STREAM, 0);
  if (sock < 0) {
    return -1;
  }

  struct sockaddr_in6 addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin6_family = AF_INET6;
  addr.sin6_port = htons(port);
  addr.sin6_addr = in6addr_loopback;

  if (connect(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) <
      0) {
    close(sock);
    return -1;
  }
  return sock;
}

TEST(WeightSynchronizerListenerTest, PushWeightsCommandSuccess) {
  WeightSynchronizerBase engine(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
      /*parallelism=*/1, /*listener_port=*/std::nullopt);

  WeightSynchronizerListener listener(&engine, /*listener_port=*/0);
  ASSERT_GT(listener.listener_port(), 0);
  EXPECT_TRUE(listener.is_active());

  int sock = ConnectToListenerPort(listener.listener_port());
  ASSERT_GE(sock, 0);

  WeightSynchronizerBase dst_engine(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*listener_port=*/std::nullopt);
  ASSERT_TRUE(dst_engine.local_port().has_value());

  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_START_TRANSFER);
  req.add_peers("127.0.0.1:" + std::to_string(*dst_engine.local_port()));

  std::string payload;
  ASSERT_TRUE(req.SerializeToString(&payload));
  uint32_t req_len = htonl(payload.size());

  EXPECT_EQ(write(sock, &req_len, sizeof(req_len)), sizeof(req_len));
  EXPECT_EQ(write(sock, payload.data(), payload.size()), payload.size());

  // Read response
  uint32_t resp_len_net = 0;
  ASSERT_EQ(read(sock, &resp_len_net, sizeof(resp_len_net)),
            sizeof(resp_len_net));
  uint32_t resp_len = ntohl(resp_len_net);

  std::string resp_bytes(resp_len, '\0');
  ASSERT_EQ(read(sock, resp_bytes.data(), resp_len), resp_len);

  ControlResponse resp;
  ASSERT_TRUE(resp.ParseFromString(resp_bytes));
  EXPECT_TRUE(resp.success());

  close(sock);
}

TEST(WeightSynchronizerListenerTest, ShutdownCommandStopsService) {
  WeightSynchronizerBase engine(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
      /*parallelism=*/1, /*listener_port=*/std::nullopt);

  WeightSynchronizerListener listener(&engine, /*listener_port=*/0);
  EXPECT_TRUE(listener.is_active());

  int sock = ConnectToListenerPort(listener.listener_port());
  ASSERT_GE(sock, 0);

  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_SHUTDOWN);

  std::string payload;
  ASSERT_TRUE(req.SerializeToString(&payload));
  uint32_t req_len = htonl(payload.size());

  EXPECT_EQ(write(sock, &req_len, sizeof(req_len)), sizeof(req_len));
  EXPECT_EQ(write(sock, payload.data(), payload.size()), payload.size());

  // Read response
  uint32_t resp_len_net = 0;
  ASSERT_EQ(read(sock, &resp_len_net, sizeof(resp_len_net)),
            sizeof(resp_len_net));
  uint32_t resp_len = ntohl(resp_len_net);

  std::string resp_bytes(resp_len, '\0');
  ASSERT_EQ(read(sock, resp_bytes.data(), resp_len), resp_len);

  ControlResponse resp;
  ASSERT_TRUE(resp.ParseFromString(resp_bytes));
  EXPECT_TRUE(resp.success());

  close(sock);
}

TEST(WeightSynchronizerListenerTest, PushWeightsReshardedSuccess) {
  WeightSynchronizerBase src_engine(
      /*num_layers=*/1, /*num_shards=*/4, /*slice_byte_size=*/16,
      /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
      /*parallelism=*/1, /*listener_port=*/std::nullopt);

  WeightSynchronizerListener listener(&src_engine,
                                      /*listener_port=*/0);
  ASSERT_GT(listener.listener_port(), 0);

  int sock = ConnectToListenerPort(listener.listener_port());
  ASSERT_GE(sock, 0);

  WeightSynchronizerBase dst_engine(
      /*num_layers=*/1, /*num_shards=*/4, /*slice_byte_size=*/16,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*listener_port=*/0);
  ASSERT_TRUE(dst_engine.local_port().has_value());
  ASSERT_TRUE(dst_engine.listener_port().has_value());
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

  // Send START_TRANSFER (Receiver) payload to target peer to register expected
  // chunks before initiating sender's resharded push transfer.
  int dst_sock = ConnectToListenerPort(*dst_engine.listener_port());
  ASSERT_GE(dst_sock, 0);

  ControlRequest dst_req;
  dst_req.set_command(ControlRequest::COMMAND_START_TRANSFER);
  StartTransferRequest* dst_start_req =
      dst_req.mutable_start_transfer_request();
  dst_start_req->set_is_sender(false);
  dst_start_req->set_uuid(test_uuid);
  dst_start_req->set_expected_block_count(8);

  std::string dst_payload;
  ASSERT_TRUE(dst_req.SerializeToString(&dst_payload));
  uint32_t dst_req_len = htonl(dst_payload.size());

  EXPECT_EQ(write(dst_sock, &dst_req_len, sizeof(dst_req_len)),
            sizeof(dst_req_len));
  EXPECT_EQ(write(dst_sock, dst_payload.data(), dst_payload.size()),
            dst_payload.size());

  // Read response
  uint32_t dst_resp_len_net = 0;
  ASSERT_EQ(read(dst_sock, &dst_resp_len_net, sizeof(dst_resp_len_net)),
            sizeof(dst_resp_len_net));
  uint32_t dst_resp_len = ntohl(dst_resp_len_net);

  std::string dst_resp_bytes(dst_resp_len, '\0');
  ASSERT_EQ(read(dst_sock, dst_resp_bytes.data(), dst_resp_len), dst_resp_len);

  ControlResponse dst_resp;
  ASSERT_TRUE(dst_resp.ParseFromString(dst_resp_bytes));
  EXPECT_TRUE(dst_resp.success()) << dst_resp.message();

  close(dst_sock);

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

  std::string payload;
  ASSERT_TRUE(req.SerializeToString(&payload));
  uint32_t req_len = htonl(payload.size());

  EXPECT_EQ(write(sock, &req_len, sizeof(req_len)), sizeof(req_len));
  EXPECT_EQ(write(sock, payload.data(), payload.size()), payload.size());

  // Read response
  uint32_t resp_len_net = 0;
  ASSERT_EQ(read(sock, &resp_len_net, sizeof(resp_len_net)),
            sizeof(resp_len_net));
  uint32_t resp_len = ntohl(resp_len_net);

  std::string resp_bytes(resp_len, '\0');
  ASSERT_EQ(read(sock, resp_bytes.data(), resp_len), resp_len);

  ControlResponse resp;
  ASSERT_TRUE(resp.ParseFromString(resp_bytes));
  EXPECT_TRUE(resp.success()) << resp.message();

  // Verify Destination Shard 0 final host memory!
  uint8_t* dst_ptr = dst_engine.GetHostPointer(0, 0);
  ASSERT_NE(dst_ptr, nullptr);

  std::vector<uint8_t> expected_d0 = {0,  1,  8,  9,  16, 17, 24, 25,
                                      32, 33, 40, 41, 48, 49, 56, 57};
  for (size_t k = 0; k < 16; ++k) {
    EXPECT_EQ(dst_ptr[k], expected_d0[k]) << "Mismatch at byte " << k;
  }

  close(sock);
}

TEST(WeightSynchronizerListenerTest, GrpcModePushWeightsReshardedSuccess) {
  WeightSynchronizerBase src_engine(
      /*num_layers=*/1, /*num_shards=*/4, /*slice_byte_size=*/16,
      /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
      /*parallelism=*/1, /*listener_port=*/0, /*bind_ip=*/std::nullopt,
      /*layer_names=*/{}, /*auto_h2d=*/false, /*use_grpc=*/true);
  ASSERT_TRUE(src_engine.listener_port().has_value());
  EXPECT_TRUE(src_engine.is_listener_active());

  WeightSynchronizerBase dst_engine(
      /*num_layers=*/1, /*num_shards=*/4, /*slice_byte_size=*/16,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*listener_port=*/0, /*bind_ip=*/std::nullopt,
      /*layer_names=*/{}, /*auto_h2d=*/false, /*use_grpc=*/true);
  ASSERT_TRUE(dst_engine.local_port().has_value());
  ASSERT_TRUE(dst_engine.listener_port().has_value());
  EXPECT_TRUE(dst_engine.is_listener_active());

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

  uint64_t test_uuid = 112233;

  WeightSynchronizationWorkerServiceClient dst_client(
      "localhost:" + std::to_string(*dst_engine.listener_port()));
  WeightSynchronizationWorkerServiceClient src_client(
      "localhost:" + std::to_string(*src_engine.listener_port()));

  // 1. Arm destination receiver using gRPC client
  StartTransferRequest dst_start_req;
  dst_start_req.set_is_sender(false);
  dst_start_req.set_uuid(test_uuid);
  dst_start_req.set_expected_block_count(8);

  auto dst_resp_or = dst_client.StartTransfer(dst_start_req).Await();
  ASSERT_TRUE(dst_resp_or.ok());
  EXPECT_TRUE(dst_resp_or->success());

  // 2. Dispatch sender push schedules using gRPC client
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
  ASSERT_TRUE(src_resp_or.ok());
  EXPECT_TRUE(src_resp_or->success());

  // Verify Destination Shard 0 host memory
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
