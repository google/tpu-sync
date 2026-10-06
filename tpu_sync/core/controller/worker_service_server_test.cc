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

#include "tpu_sync/core/controller/worker_service_server.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "tpu_sync/core/controller/worker_service_client.h"
#include "tpu_sync/proto/transfer_program.pb.h"
#include "tpu_sync/proto/worker_service.pb.h"

namespace tpu_raiden {
namespace controller {

namespace {

using ::absl_testing::StatusIs;

TEST(WorkerServiceServerTest, StartServerAndGetPortWorks) {
  WorkerServiceServer& server = WorkerServiceServer::GetInstance();
  ABSL_ASSERT_OK(server.StartServer(/*host_allocator=*/nullptr, /*port=*/0));
  int port = server.GetRaidenWorkerPort();
  EXPECT_GT(port, 0);

  // Connect via client and verify gRPC communication works.
  std::string server_address = "localhost:" + std::to_string(port);
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  WorkerServiceClient client(channel);

  ::tpu_sync::proto::CreateBuffersRequest create_req;
  create_req.mutable_unit()->set_job_name("test_job");
  create_req.mutable_unit()->set_job_replica_id("0");
  create_req.mutable_unit()->set_data_name("test_data");
  auto* spec = create_req.add_buffers();
  spec->set_num_shards(1);
  spec->set_size_bytes(512);

  auto resp_or = client.CreateBuffers(create_req).Await();
  ABSL_ASSERT_OK(resp_or);
  EXPECT_TRUE(resp_or->success());
  ASSERT_EQ(resp_or->buffers_size(), 1);
}

TEST(WorkerServiceServerTest, SingletonIsReused) {
  WorkerServiceServer& server1 = WorkerServiceServer::GetInstance();
  ABSL_ASSERT_OK(server1.StartServer(/*host_allocator=*/nullptr, /*port=*/0));
  int port1 = server1.GetRaidenWorkerPort();
  EXPECT_GT(port1, 0);

  WorkerServiceServer& server2 = WorkerServiceServer::GetInstance();
  ABSL_ASSERT_OK(server2.StartServer(/*host_allocator=*/nullptr, /*port=*/0));
  int port2 = server2.GetRaidenWorkerPort();
  EXPECT_EQ(port1, port2);
}

TEST(WorkerServiceServerTest, StartServerWithInvalidPortFails) {
  WorkerServiceServer& server = WorkerServiceServer::GetInstance();
  absl::Status status =
      server.StartServer(/*host_allocator=*/nullptr, /*port=*/-1);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(WorkerServiceServerTest, SubmitTransferProgramAcceptsMessagesOver4MiB) {
  std::unique_ptr<WorkerServiceServer> server = WorkerServiceServer::Create();
  ABSL_ASSERT_OK(server->StartServer(/*host_allocator=*/nullptr, /*port=*/0));
  const std::string server_address =
      "localhost:" + std::to_string(server->GetRaidenWorkerPort());
  WorkerServiceClient client(CreateWorkerServiceChannel(server_address));

  ::tpu_sync::proto::TransferProgramRequest req;
  req.mutable_envelope()->set_uuid(1);
  req.mutable_envelope()->set_req_id("large_prefill_req");
  req.mutable_envelope()->set_kind(
      ::tpu_sync::proto::TRANSFER_KIND_POOL_RESHARD);
  req.mutable_envelope()->set_role(
      ::tpu_sync::proto::TRANSFER_ROLE_RECEIVER_ARM);
  auto* program = req.mutable_program();
  program->mutable_reshard()->add_transfer_pool_indices(0);
  auto* group = program->mutable_completion()->mutable_fan_in()->add_groups();
  group->add_pool_indices(0);
  group->add_dst_device_block_ids(0);
  group->set_expected_pushes(1);

  // ~130k steps (~7 MiB serialized) reproduces the 6.66 MiB 519k-token
  // receiver-arm program that exceeded gRPC's 4 MiB default.
  constexpr int kNumSteps = 130000;
  program->mutable_steps()->Reserve(kNumSteps);
  for (int i = 0; i < kNumSteps; ++i) {
    auto* step = program->add_steps();
    step->mutable_src()->set_block_id(i % 1024);
    step->mutable_dst()->set_block_id(i % 1024);
    auto* extent = step->add_extents();
    extent->set_src_offset_bytes(i * 4096);
    extent->set_dst_offset_bytes(i * 4096);
    extent->set_size_bytes(4096);
    extent->set_count(1);
    step->set_dst_peer("10.0.0.1:50051");
    step->set_group(0);
    step->set_source_rank(i % 8);
  }
  ASSERT_GT(req.ByteSizeLong(), 6 * 1024 * 1024);

  absl::StatusOr<::tpu_sync::proto::TransferProgramResponse> resp =
      client.SubmitTransferProgram(req).Await();
  ABSL_ASSERT_OK(resp);
  EXPECT_FALSE(resp->success());
  EXPECT_EQ(resp->message(), "Transfer manager is not initialized");
}

}  // namespace
}  // namespace controller
}  // namespace tpu_raiden
