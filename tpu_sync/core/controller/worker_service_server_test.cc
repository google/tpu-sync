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
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "tpu_sync/core/controller/worker_service_client.h"
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

TEST(WorkerServiceServerTest, RegisterBackendsViaClient) {
  WorkerServiceServer& server = WorkerServiceServer::GetInstance();
  ABSL_ASSERT_OK(server.StartServer(/*host_allocator=*/nullptr, /*port=*/0));
  int port = server.GetRaidenWorkerPort();
  EXPECT_GT(port, 0);

  std::string server_address = "localhost:" + std::to_string(port);
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  WorkerServiceClient client(channel);

  ::tpu_sync::proto::RegisterBackendsRequest req;
  auto* config = req.add_configs();
  config->set_name("PosixBackend");
  config->set_scheme("posix_test");

  auto resp_or = client.RegisterBackends(req).Await();
  ABSL_ASSERT_OK(resp_or);
  // Expect false if transfer_manager is null, or true if configured.
  // The RPC should succeed without transport errors.
}

}  // namespace
}  // namespace controller
}  // namespace tpu_raiden
