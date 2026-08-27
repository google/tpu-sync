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

#include "tpu_sync/kv_cache/kv_cache_store_server.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/controller/controller_client.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/core/controller/test_util.h"
#include "tpu_sync/core/kv_manager_holder.h"
#include "tpu_sync/kv_cache/kv_cache_store.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_client.h"
namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::testing::UnorderedElementsAre;

class KVCacheStoreServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 1. Setup mock worker server & transfer manager
    test_worker_server_ = ::tpu_raiden::controller::CreateTestWorkerServer();
    dst_transfer_mock_ = std::make_unique<
        ::tpu_raiden::controller::ShardAwareMockTransferManager>();
    test_worker_server_->service->SetTransferManager(
        ::tpu_raiden::KVManagerHolder(dst_transfer_mock_.get()));

    // 2. Setup src controller server
    src_controller_server_ = core::controller::CreateTestControllerServer();

    RaidenId src_raiden_id{"src_job", "0", "src_data", 0};
    RaidenId dst_raiden_id{"dst_job", "0", "dst_data", 0};

    ASSERT_OK(src_controller_server_->client->RegisterWorker(
        "worker_0", test_worker_server_->server_address,
        {{test_worker_server_->server_address, {}}}));

    src_controller_server_->service->SetReadRemoteHooks(
        [&](absl::Span<const std::string> h)
            -> absl::StatusOr<std::vector<int32_t>> {
          return std::vector<int32_t>(h.size(), 42);
        },
        [&](absl::Span<const std::string> /*h*/) {});

    // 3. Create destination KVCacheStore. These cases drive the store server
    // directly, so no global registry is needed.
    store_ = std::make_unique<KVCacheStore>(
        /*capacity=*/100, /*global_registry_address=*/"", dst_raiden_id,
        /*num_shards=*/1,
        /*shard_size_bytes=*/1024,
        /*store_server_ip=*/"127.0.0.1");

    ::tpu_raiden::core::controller::RaidenControllerClient
        dst_controller_client(store_->raiden_controller_address());
    ASSERT_OK(dst_controller_client.RegisterWorker(
        "dst_worker_0", test_worker_server_->server_address,
        {{test_worker_server_->server_address, {}}}));

    // Pre-populate test blocks
    std::vector<std::string> test_hashes = {"block_hash_1", "block_hash_2"};
    std::vector<RaidenBlockId> slices = {
        RaidenBlockId(src_raiden_id, 10, BlockStatus::HOST),
        RaidenBlockId(src_raiden_id, 11, BlockStatus::HOST),
    };
    ASSERT_TRUE(store_->Insert(test_hashes, slices, /*on_host=*/true).ok());
  }

  void TearDown() override {
    if (server_) {
      server_->Shutdown();
    }
  }

  std::unique_ptr<::tpu_raiden::controller::TestWorkerServer>
      test_worker_server_;
  std::unique_ptr<::tpu_raiden::controller::ShardAwareMockTransferManager>
      dst_transfer_mock_;
  std::unique_ptr<::tpu_raiden::core::controller::TestControllerServer>
      src_controller_server_;
  std::unique_ptr<KVCacheStore> store_;
  std::unique_ptr<KVCacheStoreServer> server_;
};

TEST_F(KVCacheStoreServerTest, StartServerWithRawPointerStore) {
  server_ = KVCacheStoreServer::Create();
  // A wildcard bind reports no publishable address (no in-tree caller ever
  // wildcard-binds), so use a real, dialable host.
  ASSERT_OK(server_->StartServer(store_->backend().get(),
                                 store_->raiden_controller(), "127.0.0.1:0"));

  int port = server_->GetGrpcPort();
  EXPECT_GT(port, 0);
  EXPECT_FALSE(server_->GetServerAddress().empty());

  // Connect via KVCacheStoreClient
  auto channel = ::grpc::CreateChannel(server_->GetServerAddress(),
                                       ::grpc::InsecureChannelCredentials());
  KVCacheStoreClient client(channel);

  std::vector<std::string> hashes = {"block_hash_1", "block_hash_2"};
  std::vector<int32_t> host_ids = {100, 101};
  auto fetch_res =
      client.Fetch(hashes, /*device_block_ids=*/{}, host_ids).Await();
  ASSERT_OK(fetch_res.status());
  EXPECT_THAT(fetch_res->done_block_hashes(),
              UnorderedElementsAre("block_hash_1", "block_hash_2"));

  server_->Shutdown();
  EXPECT_EQ(server_->GetGrpcPort(), 0);
  EXPECT_TRUE(server_->GetServerAddress().empty());
}

TEST_F(KVCacheStoreServerTest, RestartServerAfterShutdown) {
  server_ = KVCacheStoreServer::Create();
  ASSERT_OK(server_->StartServer(store_->backend().get(),
                                 store_->raiden_controller(), "127.0.0.1:0"));
  int first_port = server_->GetGrpcPort();
  EXPECT_GT(first_port, 0);

  server_->Shutdown();
  EXPECT_EQ(server_->GetGrpcPort(), 0);
  EXPECT_TRUE(server_->GetServerAddress().empty());

  // Restart server on a new ephemeral port
  ASSERT_OK(server_->StartServer(store_->backend().get(),
                                 store_->raiden_controller(), "127.0.0.1:0"));
  int second_port = server_->GetGrpcPort();
  EXPECT_GT(second_port, 0);

  // Verify client connection after restart
  auto channel = ::grpc::CreateChannel(server_->GetServerAddress(),
                                       ::grpc::InsecureChannelCredentials());
  KVCacheStoreClient client(channel);

  std::vector<std::string> hashes = {"block_hash_1", "block_hash_2"};
  std::vector<int32_t> host_ids = {100, 101};
  auto fetch_res =
      client.Fetch(hashes, /*device_block_ids=*/{}, host_ids).Await();
  ASSERT_OK(fetch_res.status());
  EXPECT_THAT(fetch_res->done_block_hashes(),
              UnorderedElementsAre("block_hash_1", "block_hash_2"));

  server_->Shutdown();
}

TEST_F(KVCacheStoreServerTest, MultipleServersCanRunConcurrently) {
  auto server1 = KVCacheStoreServer::Create();
  ASSERT_OK(server1->StartServer(store_->backend().get(),
                                 store_->raiden_controller(), "[::]:0"));
  int port1 = server1->GetGrpcPort();

  auto server2 = KVCacheStoreServer::Create();
  ASSERT_OK(server2->StartServer(store_->backend().get(),
                                 store_->raiden_controller(), "[::]:0"));
  int port2 = server2->GetGrpcPort();

  EXPECT_GT(port1, 0);
  EXPECT_GT(port2, 0);
  EXPECT_NE(port1, port2);

  server1->Shutdown();
  server2->Shutdown();
}

TEST_F(KVCacheStoreServerTest, StartServerWithInvalidAddressFails) {
  auto server = KVCacheStoreServer::Create();
  absl::Status status =
      server->StartServer(store_->backend().get(), store_->raiden_controller(),
                          "invalid_address:999999");
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(server->GetGrpcPort(), 0);
}

TEST_F(KVCacheStoreServerTest, StartServerWithHostOnlySucceeds) {
  auto server = KVCacheStoreServer::Create();
  ASSERT_OK(server->StartServer(store_->backend().get(),
                                store_->raiden_controller(), "127.0.0.1"));
  EXPECT_GT(server->GetGrpcPort(), 0);
  server->Shutdown();
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
