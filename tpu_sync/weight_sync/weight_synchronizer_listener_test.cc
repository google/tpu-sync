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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/support/status.h"
#include "peregrine/src/api/v1/psp/psp.h"
#include "peregrine/src/internal/control/message.pb.h"
#include "peregrine/src/internal/control/message_internal.pb.h"
#include "peregrine/src/internal/control/service.grpc.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/common/control_pipe/control_pipe_client.h"
#include "tpu_sync/common/control_pipe/control_pipe_types.h"
#include "tpu_sync/common/control_pipe/grpc_control_pipe.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace weight_sync {

// Gives the tests access to WeightSynchronizerBase's private endpoint
// recording.
class WeightSynchronizerBaseTestPeer {
 public:
  static absl::Status RecordPeerControlEndpoints(
      WeightSynchronizerBase& engine, const std::vector<std::string>& peers,
      const tpu_sync::rpc::StartTransferRequest& request) {
    return engine.RecordPeerControlEndpoints(peers, request);
  }
};

using ::peregrine::internal::control::PeregrineService;
using ::peregrine::internal::proto::PspToken;
using ::peregrine::internal::proto::ReqMsg;
using ::peregrine::internal::proto::RespMsg;
using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::status::StatusIs;
using ::tpu_sync::rpc::ControlRequest;
using ::tpu_sync::rpc::ControlResponse;
using ::tpu_sync::rpc::ShardPushEntryProto;
using ::tpu_sync::rpc::ShardPushScheduleProto;
using ::tpu_sync::rpc::StartTransferRequest;

namespace {

// Data-plane peers and listener endpoints for tests that record key-exchange
// channels without using them: creating a gRPC channel does not connect.
constexpr char kPeerA[] = "10.0.0.1:7000";
constexpr char kPeerB[] = "10.0.0.2:7000";
constexpr char kPeerC[] = "10.0.0.3:7000";
constexpr char kPeerD[] = "10.0.0.4:7000";
constexpr char kEndpoint1[] = "10.0.1.1:9000";
constexpr char kEndpoint2[] = "10.0.1.2:9000";

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, std::optional<const char*> value)
      : name_(name) {
    if (const char* old = std::getenv(name_)) {
      previous_ = std::string(old);
    }
    if (value.has_value()) {
      setenv(name_, *value, /*overwrite=*/1);
    } else {
      unsetenv(name_);
    }
  }

  ~ScopedEnvVar() {
    if (previous_.has_value()) {
      setenv(name_, previous_->c_str(), /*overwrite=*/1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char* name_;
  std::optional<std::string> previous_;
};

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

TEST(WeightSynchronizerListenerTest, DefaultBackendIsGrpc) {
  ScopedEnvVar backend_env("TPU_RAIDEN_CONTROL_PLANE_BACKEND", std::nullopt);
  ScopedEnvVar grpc_flag("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", std::nullopt);

  EXPECT_EQ(WeightSynchronizerListener::ResolveBackendType(),
            ControlPipeBackendType::kGrpc);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> engine_ptr,
      WeightSynchronizerBase::Create(
          /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
          /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
          /*parallelism=*/1, /*listener_port=*/std::nullopt));
  WeightSynchronizerListener listener(engine_ptr.get(), /*listener_port=*/0);
  EXPECT_EQ(listener.backend_type(), ControlPipeBackendType::kGrpc);
  EXPECT_GT(listener.listener_port(), 0);
  EXPECT_TRUE(listener.is_active());
}

TEST(WeightSynchronizerListenerTest, ResolveBackendTypeHonorsEnvVars) {
  {
    ScopedEnvVar backend_env("TPU_RAIDEN_CONTROL_PLANE_BACKEND", "grpc");
    ScopedEnvVar grpc_flag("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", "0");
    EXPECT_EQ(WeightSynchronizerListener::ResolveBackendType(),
              ControlPipeBackendType::kGrpc);
  }
  {
    ScopedEnvVar backend_env("TPU_RAIDEN_CONTROL_PLANE_BACKEND", "tcp");
    ScopedEnvVar grpc_flag("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", "1");
    EXPECT_EQ(WeightSynchronizerListener::ResolveBackendType(),
              ControlPipeBackendType::kTcp);
  }
  {
    ScopedEnvVar backend_env("TPU_RAIDEN_CONTROL_PLANE_BACKEND", "unknown");
    ScopedEnvVar grpc_flag("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", "0");
    EXPECT_EQ(WeightSynchronizerListener::ResolveBackendType(),
              ControlPipeBackendType::kGrpc);
  }
  {
    ScopedEnvVar backend_env("TPU_RAIDEN_CONTROL_PLANE_BACKEND", std::nullopt);
    ScopedEnvVar grpc_flag("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", "1");
    EXPECT_EQ(WeightSynchronizerListener::ResolveBackendType(),
              ControlPipeBackendType::kGrpc);
  }
  {
    ScopedEnvVar backend_env("TPU_RAIDEN_CONTROL_PLANE_BACKEND", std::nullopt);
    ScopedEnvVar grpc_flag("TPU_RAIDEN_USE_GRPC_CONTROL_PLANE", "0");
    EXPECT_EQ(WeightSynchronizerListener::ResolveBackendType(),
              ControlPipeBackendType::kTcp);
  }
}

TEST(WeightSynchronizerListenerTest, PushWeightsCommandGrpcSuccess) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> engine_ptr,
      WeightSynchronizerBase::Create(
          /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
          /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
          /*parallelism=*/1, /*listener_port=*/std::nullopt));
  WeightSynchronizerBase& engine = *engine_ptr;

  WeightSynchronizerListener listener(&engine, /*listener_port=*/0);
  ASSERT_GT(listener.listener_port(), 0);
  EXPECT_TRUE(listener.is_active());
  EXPECT_EQ(listener.backend_type(), ControlPipeBackendType::kGrpc);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> dst_engine_ptr,
      WeightSynchronizerBase::Create(
          /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
          /*local_port=*/0, /*host_blocks_to_allocate=*/1,
          /*parallelism=*/1, /*listener_port=*/std::nullopt));
  WeightSynchronizerBase& dst_engine = *dst_engine_ptr;
  ASSERT_TRUE(dst_engine.local_port().has_value());

  ControlPipeConfig cfg;
  cfg.backend_type = ControlPipeBackendType::kGrpc;
  std::unique_ptr<ControlPipeClient> client = CreateControlPipeClient(cfg);
  const std::string endpoint =
      absl::StrCat("127.0.0.1:", listener.listener_port());

  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_START_TRANSFER);
  req.add_peers("127.0.0.1:" + std::to_string(*dst_engine.local_port()));

  TF_ASSERT_OK_AND_ASSIGN(ControlResponse resp,
                       (client->Call<ControlRequest, ControlResponse>(
                           endpoint, req, absl::Seconds(10))));
  EXPECT_TRUE(resp.success()) << resp.message();
}

TEST(WeightSynchronizerListenerTest, ShutdownCommandGrpcStopsService) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> engine_ptr,
      WeightSynchronizerBase::Create(
          /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
          /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
          /*parallelism=*/1, /*listener_port=*/std::nullopt));
  WeightSynchronizerBase& engine = *engine_ptr;

  WeightSynchronizerListener listener(&engine, /*listener_port=*/0);
  EXPECT_TRUE(listener.is_active());
  EXPECT_EQ(listener.backend_type(), ControlPipeBackendType::kGrpc);

  ControlPipeConfig cfg;
  cfg.backend_type = ControlPipeBackendType::kGrpc;
  std::unique_ptr<ControlPipeClient> client = CreateControlPipeClient(cfg);
  const std::string endpoint =
      absl::StrCat("127.0.0.1:", listener.listener_port());

  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_SHUTDOWN);

  TF_ASSERT_OK_AND_ASSIGN(ControlResponse resp,
                       (client->Call<ControlRequest, ControlResponse>(
                           endpoint, req, absl::Seconds(10))));
  EXPECT_TRUE(resp.success());
  EXPECT_FALSE(listener.is_active());

  listener.Shutdown();
  EXPECT_FALSE((client->Call<ControlRequest, ControlResponse>(endpoint, req,
                                                              absl::Seconds(5)))
                   .ok());
}

TEST(WeightSynchronizerListenerTest, PushWeightsCommandTcpLegacySuccess) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> engine_ptr,
      WeightSynchronizerBase::Create(
          /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
          /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
          /*parallelism=*/1, /*listener_port=*/std::nullopt));
  WeightSynchronizerBase& engine = *engine_ptr;

  WeightSynchronizerListener listener(&engine, /*listener_port=*/0,
                                      ControlPipeBackendType::kTcp);
  ASSERT_GT(listener.listener_port(), 0);
  EXPECT_TRUE(listener.is_active());
  EXPECT_EQ(listener.backend_type(), ControlPipeBackendType::kTcp);

  int sock = ConnectToListenerPort(listener.listener_port());
  ASSERT_GE(sock, 0);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> dst_engine_ptr,
      WeightSynchronizerBase::Create(
          /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
          /*local_port=*/0, /*host_blocks_to_allocate=*/1,
          /*parallelism=*/1, /*listener_port=*/std::nullopt));
  WeightSynchronizerBase& dst_engine = *dst_engine_ptr;
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

TEST(WeightSynchronizerListenerTest, ShutdownCommandTcpLegacyStopsService) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> engine_ptr,
      WeightSynchronizerBase::Create(
          /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
          /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
          /*parallelism=*/1, /*listener_port=*/std::nullopt));
  WeightSynchronizerBase& engine = *engine_ptr;

  WeightSynchronizerListener listener(&engine, /*listener_port=*/0,
                                      ControlPipeBackendType::kTcp);
  EXPECT_TRUE(listener.is_active());
  EXPECT_EQ(listener.backend_type(), ControlPipeBackendType::kTcp);

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
  EXPECT_FALSE(listener.is_active());
  int probe = ConnectToListenerPort(listener.listener_port());
  EXPECT_GE(probe, 0);
  if (probe >= 0) {
    close(probe);
  }
  listener.Shutdown();
  EXPECT_LT(ConnectToListenerPort(listener.listener_port()), 0);
}

TEST(WeightSynchronizerListenerTest, StartTransferRejectedAfterShutdown) {
  for (ControlPipeBackendType backend :
       {ControlPipeBackendType::kGrpc, ControlPipeBackendType::kTcp}) {
    SCOPED_TRACE(ControlPipeBackendTypeName(backend));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<WeightSynchronizerBase> engine_ptr,
        WeightSynchronizerBase::Create(
            /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
            /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
            /*parallelism=*/1, /*listener_port=*/std::nullopt));
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<WeightSynchronizerBase> dst_engine_ptr,
        WeightSynchronizerBase::Create(
            /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
            /*local_port=*/0, /*host_blocks_to_allocate=*/1,
            /*parallelism=*/1, /*listener_port=*/std::nullopt));
    ASSERT_TRUE(dst_engine_ptr->local_port().has_value());

    WeightSynchronizerListener listener(engine_ptr.get(), /*listener_port=*/0,
                                        backend);
    ASSERT_GT(listener.listener_port(), 0);
    EXPECT_TRUE(listener.is_active());

    ControlPipeConfig cfg;
    cfg.backend_type = backend;
    std::unique_ptr<ControlPipeClient> client = CreateControlPipeClient(cfg);
    const std::string endpoint =
        absl::StrCat("127.0.0.1:", listener.listener_port());

    ControlRequest transfer_req;
    transfer_req.set_command(ControlRequest::COMMAND_START_TRANSFER);
    transfer_req.mutable_start_transfer_request()->set_is_sender(true);
    transfer_req.add_peers(
        absl::StrCat("127.0.0.1:", *dst_engine_ptr->local_port()));

    TF_ASSERT_OK_AND_ASSIGN(ControlResponse pre_resp,
                         (client->Call<ControlRequest, ControlResponse>(
                             endpoint, transfer_req, absl::Seconds(10))));
    EXPECT_TRUE(pre_resp.success()) << pre_resp.message();

    ControlRequest shutdown_req;
    shutdown_req.set_command(ControlRequest::COMMAND_SHUTDOWN);
    TF_ASSERT_OK_AND_ASSIGN(ControlResponse shutdown_resp,
                         (client->Call<ControlRequest, ControlResponse>(
                             endpoint, shutdown_req, absl::Seconds(10))));
    EXPECT_TRUE(shutdown_resp.success()) << shutdown_resp.message();
    EXPECT_FALSE(listener.is_active());

    // The server keeps accepting connections until Shutdown() / destruction,
    // but new START_TRANSFER requests are rejected once stopping begins.
    TF_ASSERT_OK_AND_ASSIGN(ControlResponse post_resp,
                         (client->Call<ControlRequest, ControlResponse>(
                             endpoint, transfer_req, absl::Seconds(10))));
    EXPECT_FALSE(post_resp.success());
    EXPECT_EQ(post_resp.message(), "WeightSynchronizerListener is stopping");
  }
}

TEST(WeightSynchronizerListenerTest, PushWeightsReshardedSuccess) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> src_engine_ptr,
      WeightSynchronizerBase::Create(
          /*num_layers=*/1, /*num_shards=*/4, /*slice_byte_size=*/16,
          /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
          /*parallelism=*/1, /*listener_port=*/std::nullopt));
  WeightSynchronizerBase& src_engine = *src_engine_ptr;

  WeightSynchronizerListener listener(&src_engine,
                                      /*listener_port=*/0);
  ASSERT_GT(listener.listener_port(), 0);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> dst_engine_ptr,
      WeightSynchronizerBase::Create(
          /*num_layers=*/1, /*num_shards=*/4, /*slice_byte_size=*/16,
          /*local_port=*/0, /*host_blocks_to_allocate=*/1,
          /*parallelism=*/1, /*listener_port=*/0));
  WeightSynchronizerBase& dst_engine = *dst_engine_ptr;
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

  ControlPipeConfig cfg;
  cfg.backend_type = listener.backend_type();
  std::unique_ptr<ControlPipeClient> client = CreateControlPipeClient(cfg);
  // Send START_TRANSFER (Receiver) payload to target peer to register expected
  // chunks before initiating sender's resharded push transfer.
  ControlRequest dst_req;
  dst_req.set_command(ControlRequest::COMMAND_START_TRANSFER);
  StartTransferRequest* dst_start_req =
      dst_req.mutable_start_transfer_request();
  dst_start_req->set_is_sender(false);
  dst_start_req->set_uuid(test_uuid);
  dst_start_req->set_expected_block_count(8);
  (*dst_start_req->mutable_expected_layer_chunk_counts())[0] = 8;

  TF_ASSERT_OK_AND_ASSIGN(
      ControlResponse dst_resp,
      (client->Call<ControlRequest, ControlResponse>(
          absl::StrCat("127.0.0.1:", *dst_engine.listener_port()), dst_req,
          absl::Seconds(10))));
  EXPECT_TRUE(dst_resp.success()) << dst_resp.message();
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

  TF_ASSERT_OK_AND_ASSIGN(ControlResponse src_resp,
                       (client->Call<ControlRequest, ControlResponse>(
                           absl::StrCat("127.0.0.1:", listener.listener_port()),
                           req, absl::Seconds(10))));
  EXPECT_TRUE(src_resp.success()) << src_resp.message();

  // Verify Destination Shard 0 final host memory!
  uint8_t* dst_ptr = dst_engine.GetHostPointer(0, 0);
  ASSERT_NE(dst_ptr, nullptr);

  std::vector<uint8_t> expected_d0 = {0,  1,  8,  9,  16, 17, 24, 25,
                                      32, 33, 40, 41, 48, 49, 56, 57};
  for (size_t k = 0; k < 16; ++k) {
    EXPECT_EQ(dst_ptr[k], expected_d0[k]) << "Mismatch at byte " << k;
  }
}

TEST(WeightSynchronizerListenerTest,
     ExecuteControlRequestBroadcastRoundDestinationsLogging) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> engine_ptr,
      WeightSynchronizerBase::Create(
          /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
          /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
          /*parallelism=*/1, /*listener_port=*/std::nullopt));
  WeightSynchronizerBase& engine = *engine_ptr;

  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_START_TRANSFER);

  StartTransferRequest* start_req = req.mutable_start_transfer_request();
  start_req->set_is_sender(false);
  start_req->set_req_id("test_bcast_dest_req");
  start_req->set_uuid(8888);
  start_req->set_broadcast_round(0);
  start_req->set_expected_block_count(1);
  (*start_req->mutable_expected_layer_chunk_counts())[0] = 1;

  auto* src_unit = start_req->add_src_units();
  src_unit->set_job_name("trainer");
  src_unit->set_job_replica_id("0");
  src_unit->set_data_name("weights_0");

  // Round 0: single destination unit with 1 peer.
  auto* rd0 = start_req->add_broadcast_round_destinations();
  rd0->set_round_idx(0);
  rd0->add_dst_units("sampler:0");
  rd0->add_dst_peers("10.0.0.1:8000");

  // Round 1: multi-host unit with 1 unit and 2 peers (verifying refined
  // formatting).
  auto* rd1 = start_req->add_broadcast_round_destinations();
  rd1->set_round_idx(1);
  rd1->add_dst_units("sampler:1");
  rd1->add_dst_peers("10.0.0.2:8000");
  rd1->add_dst_peers("10.0.0.3:8000");

  // Round 2: 1-to-1 paired units and peers.
  auto* rd2 = start_req->add_broadcast_round_destinations();
  rd2->set_round_idx(2);
  rd2->add_dst_units("sampler:2");
  rd2->add_dst_units("sampler:3");
  rd2->add_dst_peers("10.0.0.4:8000");
  rd2->add_dst_peers("10.0.0.5:8000");

  ControlResponse resp;
  WeightSynchronizerListener::ExecuteControlRequest(&engine, req, &resp,
                                                    []() {});

  EXPECT_TRUE(resp.success());
  EXPECT_EQ(resp.message(), "SUCCESS");
  EXPECT_EQ(start_req->broadcast_round(), 0);
  EXPECT_EQ(start_req->broadcast_round_destinations_size(), 3);
}

TEST(WeightSynchronizerListenerTest,
     ReceiverStartTransferRejectsMissingLayerChunkCounts) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> engine_ptr,
      WeightSynchronizerBase::Create(
          /*num_layers=*/2, /*num_shards=*/1, /*slice_byte_size=*/128,
          /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
          /*parallelism=*/1, /*listener_port=*/std::nullopt));
  WeightSynchronizerBase& engine = *engine_ptr;

  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_START_TRANSFER);
  StartTransferRequest* start_req = req.mutable_start_transfer_request();
  start_req->set_is_sender(false);
  start_req->set_uuid(7777);
  start_req->set_expected_block_count(3);

  ControlResponse resp;
  WeightSynchronizerListener::ExecuteControlRequest(&engine, req, &resp,
                                                    []() {});
  EXPECT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("expected_layer_chunk_counts"));
}

TEST(WeightSynchronizerListenerTest,
     ReceiverStartTransferRejectsLayerChunkCountsNotSummingToTotal) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> engine_ptr,
      WeightSynchronizerBase::Create(
          /*num_layers=*/2, /*num_shards=*/1, /*slice_byte_size=*/128,
          /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
          /*parallelism=*/1, /*listener_port=*/std::nullopt));
  WeightSynchronizerBase& engine = *engine_ptr;

  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_START_TRANSFER);
  StartTransferRequest* start_req = req.mutable_start_transfer_request();
  start_req->set_is_sender(false);
  start_req->set_uuid(7778);
  start_req->set_expected_block_count(3);
  (*start_req->mutable_expected_layer_chunk_counts())[0] = 1;
  (*start_req->mutable_expected_layer_chunk_counts())[1] = 1;

  ControlResponse resp;
  WeightSynchronizerListener::ExecuteControlRequest(&engine, req, &resp,
                                                    []() {});
  EXPECT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), HasSubstr("summing to 2"));
}

ControlPipeConfig GrpcClientConfig() {
  ControlPipeConfig cfg;
  cfg.backend_type = ControlPipeBackendType::kGrpc;
  return cfg;
}

// Sends |request| to PeregrineService over |channel|.
grpc::Status CallPeregrine(const std::shared_ptr<grpc::Channel>& channel,
                           const ReqMsg& request) {
  std::unique_ptr<PeregrineService::Stub> stub =
      PeregrineService::NewStub(channel);
  grpc::ClientContext ctx;
  RespMsg response;
  return stub->ProcessUnary(&ctx, request, &response);
}

// A PSP-TCP request without a token. The transport's PeregrineService rejects
// it during validation, before it reaches the transport.
ReqMsg PspTcpRequestWithoutToken() {
  ReqMsg request;
  request.mutable_psp_tcp_req();
  return request;
}

// A PSP-TCP request that passes every validation of the transport's
// PeregrineService, so it is handed to the transport.
ReqMsg ValidPspTcpRequest() {
  ReqMsg request;
  PspToken* psp = request.mutable_psp_tcp_req()->mutable_psp();
  psp->set_spi(7);
  psp->set_gen(1);
  psp->set_key(std::string(16, 'k'));
  request.mutable_psp_tcp_req()->mutable_peer_target()->set_ip_port(
      "127.0.0.1:56789");
  return request;
}

TEST(WeightSynchronizerListenerTest,
     GrpcListenerServesControlRequestAndPeregrineOnOneChannel) {
  ASSERT_FALSE(absl::GetFlag(FLAGS_peregrine_enable_psp_tcp));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> engine_ptr,
      WeightSynchronizerBase::Create(
          /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
          /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
          /*parallelism=*/1, /*listener_port=*/std::nullopt));
  EXPECT_FALSE(engine_ptr->psp_tcp_required());
  WeightSynchronizerListener listener(engine_ptr.get(), /*listener_port=*/0);
  ASSERT_EQ(listener.backend_type(), ControlPipeBackendType::kGrpc);
  const std::string endpoint =
      absl::StrCat("127.0.0.1:", listener.listener_port());

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> dst_engine_ptr,
      WeightSynchronizerBase::Create(
          /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
          /*local_port=*/0, /*host_blocks_to_allocate=*/1,
          /*parallelism=*/1, /*listener_port=*/std::nullopt));
  ASSERT_TRUE(dst_engine_ptr->local_port().has_value());

  GrpcControlPipeClient client(GrpcClientConfig());
  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_START_TRANSFER);
  req.add_peers(absl::StrCat("127.0.0.1:", *dst_engine_ptr->local_port()));
  TF_ASSERT_OK_AND_ASSIGN(ControlResponse resp,
                       (client.Call<ControlRequest, ControlResponse>(
                           endpoint, req, absl::Seconds(10))));
  EXPECT_TRUE(resp.success()) << resp.message();

  // The channel the control request just used.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<grpc::Channel> channel,
                       client.GetOrCreateChannel(endpoint));
  EXPECT_EQ(client.TEST_CachedStubCount(), 1);

  // Answered by the transport's service: its own validation message.
  grpc::Status no_token_status =
      CallPeregrine(channel, PspTcpRequestWithoutToken());
  EXPECT_EQ(no_token_status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(no_token_status.error_message(),
              HasSubstr("psp token must be present"));

  // A valid request goes through the service into this engine's transport,
  // which refuses it because PSP-TCP is off. Only the transport produces this
  // message.
  grpc::Status valid_status = CallPeregrine(channel, ValidPspTcpRequest());
  EXPECT_EQ(valid_status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(valid_status.error_message(),
              HasSubstr("PSP is not enabled in transport"));
  EXPECT_EQ(client.TEST_CachedStubCount(), 1);
}

TEST(WeightSynchronizerListenerTest,
     GrpcListenerServesPeregrineWhenPspTcpRequired) {
  absl::SetFlag(&FLAGS_peregrine_enable_psp_tcp, true);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> engine_ptr,
      WeightSynchronizerBase::Create(
          /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
          /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
          /*parallelism=*/1, /*listener_port=*/std::nullopt));
  ASSERT_TRUE(engine_ptr->psp_tcp_required());
  WeightSynchronizerListener listener(engine_ptr.get(), /*listener_port=*/0);
  ASSERT_EQ(listener.backend_type(), ControlPipeBackendType::kGrpc);

  GrpcControlPipeClient client(GrpcClientConfig());
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<grpc::Channel> channel,
                       client.GetOrCreateChannel(absl::StrCat(
                           "127.0.0.1:", listener.listener_port())));

  // The service's own validation answers; an unserved service would return
  // UNIMPLEMENTED. No key exchange is attempted here.
  grpc::Status status = CallPeregrine(channel, PspTcpRequestWithoutToken());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), HasSubstr("psp token must be present"));
}

// Builds a PSP-TCP engine and a TCP-backend listener on it.
void ConstructTcpListenerWithPspTcpRequired() {
  absl::SetFlag(&FLAGS_peregrine_enable_psp_tcp, true);
  absl::StatusOr<std::unique_ptr<WeightSynchronizerBase>> engine =
      WeightSynchronizerBase::Create(
          /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
          /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
          /*parallelism=*/1, /*listener_port=*/std::nullopt);
  // Exit cleanly if the engine cannot be built, so that only the listener's
  // own check can satisfy the death expectation.
  if (!engine.ok()) {
    return;
  }
  WeightSynchronizerListener listener(engine->get(), /*listener_port=*/0,
                                      ControlPipeBackendType::kTcp);
}

// The flag-off counterpart is PushWeightsCommandTcpLegacySuccess: a TCP
// listener starts without the service when PSP-TCP is not required.
TEST(WeightSynchronizerListenerDeathTest, TcpListenerDiesWhenPspTcpRequired) {
  EXPECT_DEATH(ConstructTcpListenerWithPspTcpRequired(),
               "PSP-TCP is required but the listener backend 'tcp' cannot "
               "serve PeregrineService");
}

// The assertions below show that the teardown sequence ran: shutdown command,
// then engine destruction with a listener that served PeregrineService. They
// do not show the lifetime property itself (no handler runs against a
// destroyed transport); that is checked only when this test runs under ASan
// and TSan.
TEST(WeightSynchronizerListenerTest,
     EngineOwnedListenerTearsDownAfterShutdownCommand) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WeightSynchronizerBase> engine,
      WeightSynchronizerBase::Create(
          /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
          /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
          /*parallelism=*/1, /*listener_port=*/0));
  ASSERT_TRUE(engine->listener_port().has_value());
  ASSERT_TRUE(engine->is_listener_active());
  const std::string endpoint =
      absl::StrCat("127.0.0.1:", *engine->listener_port());

  GrpcControlPipeClient client(GrpcClientConfig());
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<grpc::Channel> channel,
                       client.GetOrCreateChannel(endpoint));

  // The engine-owned listener serves the transport's PeregrineService.
  grpc::Status served_status =
      CallPeregrine(channel, PspTcpRequestWithoutToken());
  EXPECT_EQ(served_status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(served_status.error_message(),
              HasSubstr("psp token must be present"));

  ControlRequest shutdown_req;
  shutdown_req.set_command(ControlRequest::COMMAND_SHUTDOWN);
  TF_ASSERT_OK_AND_ASSIGN(ControlResponse shutdown_resp,
                       (client.Call<ControlRequest, ControlResponse>(
                           endpoint, shutdown_req, absl::Seconds(10))));
  EXPECT_TRUE(shutdown_resp.success()) << shutdown_resp.message();
  EXPECT_FALSE(engine->is_listener_active());

  engine.reset();

  // Nothing answers on the old channel once the engine is gone.
  grpc::Status after_status =
      CallPeregrine(channel, PspTcpRequestWithoutToken());
  EXPECT_EQ(after_status.error_code(), grpc::StatusCode::UNAVAILABLE)
      << after_status.error_message();
}

// Creates a CPU-only engine with one layer and one shard, and with its own
// listener when |listener_port| is set. Reads --peregrine_enable_psp_tcp.
absl::StatusOr<std::unique_ptr<WeightSynchronizerBase>> CreateCpuEngine(
    std::optional<int> listener_port = std::nullopt) {
  return WeightSynchronizerBase::Create(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/128,
      /*local_port=*/0, /*host_blocks_to_allocate=*/std::nullopt,
      /*parallelism=*/1, listener_port);
}

// Adds an entry pushing two bytes of layer 0 to shard 0's schedule in
// |request| and returns it for the caller to set its peers.
ShardPushEntryProto* AddPushEntry(StartTransferRequest& request) {
  ShardPushEntryProto* entry =
      (*request.mutable_shard_push_schedules())[0].add_entries();
  entry->set_layer_idx(0);
  entry->set_size_bytes(2);
  return entry;
}

TEST(WeightSynchronizerPspPeerChannelTest,
     RecordsNothingWhenPspTcpNotRequired) {
  ASSERT_FALSE(absl::GetFlag(FLAGS_peregrine_enable_psp_tcp));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> engine,
                       CreateCpuEngine());
  ASSERT_FALSE(engine->psp_tcp_required());
  StartTransferRequest request;
  (*request.mutable_peer_control_endpoints())[kPeerA] = kEndpoint1;

  ABSL_EXPECT_OK(WeightSynchronizerBaseTestPeer::RecordPeerControlEndpoints(
      *engine, {kPeerA}, request));
  EXPECT_EQ(engine->GetPeregrineChannel(kPeerA), nullptr);
  // A peer without an entry is not an error either.
  ABSL_EXPECT_OK(WeightSynchronizerBaseTestPeer::RecordPeerControlEndpoints(
      *engine, {kPeerB}, request));
}

TEST(WeightSynchronizerPspPeerChannelTest, ReturnsChannelForRecordedPeer) {
  absl::SetFlag(&FLAGS_peregrine_enable_psp_tcp, true);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> engine,
                       CreateCpuEngine());
  ASSERT_TRUE(engine->psp_tcp_required());
  StartTransferRequest request;
  (*request.mutable_peer_control_endpoints())[kPeerA] = kEndpoint1;

  EXPECT_EQ(engine->GetPeregrineChannel(kPeerA), nullptr);
  ASSERT_OK(WeightSynchronizerBaseTestPeer::RecordPeerControlEndpoints(
      *engine, {kPeerA}, request));
  EXPECT_NE(engine->GetPeregrineChannel(kPeerA), nullptr);
}

TEST(WeightSynchronizerPspPeerChannelTest, SharesChannelPerEndpoint) {
  absl::SetFlag(&FLAGS_peregrine_enable_psp_tcp, true);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> engine,
                       CreateCpuEngine());
  StartTransferRequest request;
  (*request.mutable_peer_control_endpoints())[kPeerA] = kEndpoint1;
  (*request.mutable_peer_control_endpoints())[kPeerB] = kEndpoint1;
  (*request.mutable_peer_control_endpoints())[kPeerC] = kEndpoint2;

  ASSERT_OK(WeightSynchronizerBaseTestPeer::RecordPeerControlEndpoints(
      *engine, {kPeerA, kPeerB, kPeerC}, request));
  std::shared_ptr<grpc::Channel> channel_a =
      engine->GetPeregrineChannel(kPeerA);
  ASSERT_NE(channel_a, nullptr);
  EXPECT_EQ(engine->GetPeregrineChannel(kPeerB), channel_a);
  EXPECT_NE(engine->GetPeregrineChannel(kPeerC), nullptr);
  EXPECT_NE(engine->GetPeregrineChannel(kPeerC), channel_a);
}

TEST(WeightSynchronizerPspPeerChannelTest, NamesEveryPeerMissingFromRequest) {
  absl::SetFlag(&FLAGS_peregrine_enable_psp_tcp, true);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> engine,
                       CreateCpuEngine());
  StartTransferRequest request;
  (*request.mutable_peer_control_endpoints())[kPeerA] = kEndpoint1;

  EXPECT_THAT(
      WeightSynchronizerBaseTestPeer::RecordPeerControlEndpoints(
          *engine, {kPeerA, kPeerB, kPeerC}, request),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               AllOf(HasSubstr("peer_control_endpoints"), HasSubstr(kPeerB),
                     HasSubstr(kPeerC), Not(HasSubstr(kPeerA)))));
  // Nothing was recorded, not even for the peer that has an entry.
  EXPECT_EQ(engine->GetPeregrineChannel(kPeerA), nullptr);
}

TEST(WeightSynchronizerPspPeerChannelTest, RejectsEmptyEndpoint) {
  absl::SetFlag(&FLAGS_peregrine_enable_psp_tcp, true);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> engine,
                       CreateCpuEngine());
  StartTransferRequest request;
  (*request.mutable_peer_control_endpoints())[kPeerA] = kEndpoint1;
  (*request.mutable_peer_control_endpoints())[kPeerB] = "";

  EXPECT_THAT(WeightSynchronizerBaseTestPeer::RecordPeerControlEndpoints(
                  *engine, {kPeerA, kPeerB}, request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr(kPeerB), Not(HasSubstr(kPeerA)))));
  EXPECT_EQ(engine->GetPeregrineChannel(kPeerA), nullptr);
}

TEST(WeightSynchronizerPspPeerChannelTest, DoesNotTrustEarlierRequests) {
  absl::SetFlag(&FLAGS_peregrine_enable_psp_tcp, true);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> engine,
                       CreateCpuEngine());
  StartTransferRequest earlier_request;
  (*earlier_request.mutable_peer_control_endpoints())[kPeerA] = kEndpoint1;
  ASSERT_OK(WeightSynchronizerBaseTestPeer::RecordPeerControlEndpoints(
      *engine, {kPeerA}, earlier_request));
  ASSERT_NE(engine->GetPeregrineChannel(kPeerA), nullptr);

  // The peer is recorded right now, but this request does not map it.
  EXPECT_THAT(
      WeightSynchronizerBaseTestPeer::RecordPeerControlEndpoints(
          *engine, {kPeerA}, StartTransferRequest()),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr(kPeerA)));
}

TEST(WeightSynchronizerPspPeerChannelTest, LaterRequestReplacesEndpoint) {
  absl::SetFlag(&FLAGS_peregrine_enable_psp_tcp, true);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> engine,
                       CreateCpuEngine());
  StartTransferRequest earlier_request;
  (*earlier_request.mutable_peer_control_endpoints())[kPeerA] = kEndpoint1;
  StartTransferRequest later_request;
  (*later_request.mutable_peer_control_endpoints())[kPeerA] = kEndpoint2;

  ASSERT_OK(WeightSynchronizerBaseTestPeer::RecordPeerControlEndpoints(
      *engine, {kPeerA}, earlier_request));
  std::shared_ptr<grpc::Channel> earlier_channel =
      engine->GetPeregrineChannel(kPeerA);
  ASSERT_NE(earlier_channel, nullptr);
  ASSERT_OK(WeightSynchronizerBaseTestPeer::RecordPeerControlEndpoints(
      *engine, {kPeerA}, later_request));
  std::shared_ptr<grpc::Channel> later_channel =
      engine->GetPeregrineChannel(kPeerA);
  ASSERT_NE(later_channel, nullptr);
  EXPECT_NE(later_channel, earlier_channel);
}

TEST(WeightSynchronizerPspPeerChannelTest,
     PushWeightsReshardedNamesUnmappedPeersOfItsEntries) {
  absl::SetFlag(&FLAGS_peregrine_enable_psp_tcp, true);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> engine,
                       CreateCpuEngine());
  StartTransferRequest request;
  request.set_is_sender(true);
  AddPushEntry(request)->add_dst_peers(kPeerA);
  AddPushEntry(request)->set_dst_peer(kPeerB);
  // The engine pushes this entry to dst_peers only, so its dst_peer needs no
  // endpoint.
  ShardPushEntryProto* both_fields_entry = AddPushEntry(request);
  both_fields_entry->add_dst_peers(kPeerC);
  both_fields_entry->set_dst_peer(kPeerD);

  // Every unmapped peer is named, so a peer that is not named is not demanded.
  EXPECT_THAT(engine->PushWeightsResharded(request),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       AllOf(HasSubstr("peer_control_endpoints"),
                             HasSubstr(kPeerA), HasSubstr(kPeerB),
                             HasSubstr(kPeerC), Not(HasSubstr(kPeerD)))));

  (*request.mutable_peer_control_endpoints())[kPeerC] = kEndpoint1;
  EXPECT_THAT(engine->PushWeightsResharded(request),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       AllOf(HasSubstr(kPeerA), HasSubstr(kPeerB),
                             Not(HasSubstr(kPeerC)), Not(HasSubstr(kPeerD)))));
  EXPECT_EQ(engine->GetPeregrineChannel(kPeerC), nullptr);
}

TEST(WeightSynchronizerPspPeerChannelTest,
     ListenerFailsPlanlessSenderRequestNamingUnmappedPeer) {
  absl::SetFlag(&FLAGS_peregrine_enable_psp_tcp, true);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> engine,
                       CreateCpuEngine(/*listener_port=*/0));
  ASSERT_TRUE(engine->listener_port().has_value());
  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_START_TRANSFER);
  req.add_peers(kPeerA);
  req.add_peers(kPeerB);
  StartTransferRequest* start_req = req.mutable_start_transfer_request();
  start_req->set_is_sender(true);
  (*start_req->mutable_peer_control_endpoints())[kPeerA] = kEndpoint1;

  GrpcControlPipeClient client(GrpcClientConfig());
  TF_ASSERT_OK_AND_ASSIGN(ControlResponse resp,
                       (client.Call<ControlRequest, ControlResponse>(
                           absl::StrCat("127.0.0.1:", *engine->listener_port()),
                           req, absl::Seconds(10))));
  EXPECT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), AllOf(HasSubstr("peer_control_endpoints"),
                                    HasSubstr(kPeerB), Not(HasSubstr(kPeerA))));
}

TEST(WeightSynchronizerPspPeerChannelTest,
     ListenerFailsPlannedSenderRequestNamingUnmappedPeer) {
  absl::SetFlag(&FLAGS_peregrine_enable_psp_tcp, true);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> engine,
                       CreateCpuEngine(/*listener_port=*/0));
  ASSERT_TRUE(engine->listener_port().has_value());
  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_START_TRANSFER);
  StartTransferRequest* start_req = req.mutable_start_transfer_request();
  start_req->set_is_sender(true);
  AddPushEntry(*start_req)->add_dst_peers(kPeerA);
  AddPushEntry(*start_req)->add_dst_peers(kPeerB);
  (*start_req->mutable_peer_control_endpoints())[kPeerA] = kEndpoint1;

  GrpcControlPipeClient client(GrpcClientConfig());
  TF_ASSERT_OK_AND_ASSIGN(ControlResponse resp,
                       (client.Call<ControlRequest, ControlResponse>(
                           absl::StrCat("127.0.0.1:", *engine->listener_port()),
                           req, absl::Seconds(10))));
  EXPECT_FALSE(resp.success());
  EXPECT_THAT(resp.message(), AllOf(HasSubstr("peer_control_endpoints"),
                                    HasSubstr(kPeerB), Not(HasSubstr(kPeerA))));
}

TEST(WeightSynchronizerPspPeerChannelTest,
     RecordedChannelReachesPeerPeregrineService) {
  absl::SetFlag(&FLAGS_peregrine_enable_psp_tcp, true);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> receiver,
                       CreateCpuEngine(/*listener_port=*/0));
  ASSERT_TRUE(receiver->local_port().has_value());
  ASSERT_TRUE(receiver->listener_port().has_value());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<WeightSynchronizerBase> sender,
                       CreateCpuEngine());
  const std::string receiver_data_endpoint =
      absl::StrCat("127.0.0.1:", *receiver->local_port());
  StartTransferRequest request;
  (*request.mutable_peer_control_endpoints())[receiver_data_endpoint] =
      absl::StrCat("127.0.0.1:", *receiver->listener_port());

  ASSERT_OK(WeightSynchronizerBaseTestPeer::RecordPeerControlEndpoints(
      *sender, {receiver_data_endpoint}, request));
  // What the sender's transport gets when it connects to the receiver.
  std::shared_ptr<grpc::Channel> channel =
      sender->GetPeregrineChannel(receiver_data_endpoint);
  ASSERT_NE(channel, nullptr);

  // Answered by the PeregrineService the receiver's listener registered: its
  // own validation message. No key exchange is attempted.
  grpc::Status status = CallPeregrine(channel, PspTcpRequestWithoutToken());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), HasSubstr("psp token must be present"));
}

}  // namespace
}  // namespace weight_sync
}  // namespace tpu_raiden
