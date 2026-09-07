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

#include "tpu_sync/kv_cache/kv_cache_store_client.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/time/time.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/support/status.h"
#include "grpcpp/support/sync_stream.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/core/raiden_future.h"
#include "tpu_sync/kv_cache/block_tracker.h"
#include "tpu_sync/proto/kv_cache_store_service.grpc.pb.h"
#include "tpu_sync/proto/kv_cache_store_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

namespace {

using ::absl_testing::StatusIs;
using ::testing::UnorderedElementsAre;

class TestKVCacheStoreService
    : public ::tpu_raiden::kv_cache::proto::KVCacheStoreService::Service {
 public:
  ::grpc::Status Fetch(
      ::grpc::ServerContext* context,
      const ::tpu_raiden::kv_cache::proto::FetchRequest* request,
      ::tpu_raiden::kv_cache::proto::FetchResponse* response) override {
    if (fail_rpc_) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                            "Simulated RPC error");
    }
    last_request_ = *request;
    for (const auto& hash : request->block_hashes()) {
      response->add_done_block_hashes(hash);
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status WriteRemote(
      ::grpc::ServerContext* context,
      const ::tpu_raiden::kv_cache::proto::WriteRemoteRequest* request,
      ::grpc::ServerWriter<::tpu_raiden::kv_cache::proto::WriteRemoteEvent>*
          writer) override {
    if (fail_write_remote_) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                            "Simulated WriteRemote RPC error");
    }
    ::tpu_raiden::kv_cache::proto::WriteRemoteEvent ack_event;
    ack_event.mutable_ack()->set_operation_id(12345);
    ack_event.mutable_ack()->set_granted_deadline_ms(5000);
    writer->Write(ack_event);

    ::tpu_raiden::kv_cache::proto::WriteRemoteEvent result_event;
    result_event.mutable_result()->set_state(
        ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::COMMITTED);
    writer->Write(result_event);
    return ::grpc::Status::OK;
  }

  void SetFailRpc(bool fail) { fail_rpc_ = fail; }
  void SetFailWriteRemote(bool fail) { fail_write_remote_ = fail; }
  const ::tpu_raiden::kv_cache::proto::FetchRequest& last_request() const {
    return last_request_;
  }

 private:
  bool fail_rpc_ = false;
  bool fail_write_remote_ = false;
  ::tpu_raiden::kv_cache::proto::FetchRequest last_request_;
};

class KVCacheStoreClientTest : public ::testing::Test {
 protected:
  void SetUp() override {
    service_ = std::make_unique<TestKVCacheStoreService>();
    ::grpc::ServerBuilder builder;
    int selected_port = 0;
    builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(),
                             &selected_port);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();

    std::string server_address = "localhost:" + std::to_string(selected_port);
    auto channel = ::grpc::CreateChannel(server_address,
                                         ::grpc::InsecureChannelCredentials());
    client_ = std::make_unique<KVCacheStoreClient>(channel);
  }

  void TearDown() override {
    if (server_) {
      server_->Shutdown();
    }
  }

  std::unique_ptr<TestKVCacheStoreService> service_;
  std::unique_ptr<::grpc::Server> server_;
  std::unique_ptr<KVCacheStoreClient> client_;
};

TEST_F(KVCacheStoreClientTest, FetchReturnsFutureWithFetchResponseSuccess) {
  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<int32_t> host_ids = {10, 11};
  ::tpu_sync::rpc::RaidenIdProto client_id;
  client_id.set_job_name("test_job");
  client_id.set_job_replica_id("0");

  tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse> future =
      client_->Fetch(hashes, /*device_block_ids=*/{}, host_ids, client_id);
  TF_ASSERT_OK_AND_ASSIGN(auto response, future.Await());
  EXPECT_THAT(response.done_block_hashes(),
              UnorderedElementsAre("hash_1", "hash_2"));
}

TEST_F(KVCacheStoreClientTest, FetchReturnsFutureWithErrorStatusOnRPCFailure) {
  service_->SetFailRpc(true);
  std::vector<std::string> hashes = {"hash_1"};
  std::vector<int32_t> host_ids = {10};

  tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse> future =
      client_->Fetch(hashes, /*device_block_ids=*/{}, host_ids);
  auto response = future.Await();
  EXPECT_THAT(response.status(), StatusIs(absl::StatusCode::kInternal));
}

TEST_F(KVCacheStoreClientTest, FetchPopulatesRequestFieldsCorrectly) {
  std::vector<std::string> hashes = {"hash_a", "hash_b"};
  std::vector<int32_t> dev_ids = {100, 101};
  std::vector<int32_t> host_ids = {10, 11};
  ::tpu_sync::rpc::RaidenIdProto client_id;
  client_id.set_job_name("client_job");
  client_id.set_job_replica_id("1");
  client_id.set_data_name("data");
  client_id.set_data_replica_idx(2);

  tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse> future =
      client_->Fetch(hashes, dev_ids, host_ids, client_id);
  TF_ASSERT_OK_AND_ASSIGN(auto response, future.Await());

  const auto& req = service_->last_request();
  EXPECT_THAT(req.block_hashes(), UnorderedElementsAre("hash_a", "hash_b"));
  EXPECT_EQ(req.device_block_ids_size(), 2);
  EXPECT_EQ(req.device_block_ids(0), 100);
  EXPECT_EQ(req.device_block_ids(1), 101);
  EXPECT_EQ(req.host_block_ids_size(), 2);
  EXPECT_EQ(req.host_block_ids(0), 10);
  EXPECT_EQ(req.host_block_ids(1), 11);
  EXPECT_EQ(req.client_raiden_id().job_name(), "client_job");
  EXPECT_EQ(req.client_raiden_id().job_replica_id(), "1");
}

TEST_F(KVCacheStoreClientTest, FetchValidatesMismatchedDeviceBlockIds) {
  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<int32_t> dev_ids = {100};  // Mismatched count

  tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse> future =
      client_->Fetch(hashes, dev_ids);
  auto response = future.Await();
  EXPECT_THAT(response.status(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(KVCacheStoreClientTest, FetchValidatesMismatchedHostBlockIds) {
  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<int32_t> host_ids = {10};  // Mismatched count

  tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse> future =
      client_->Fetch(hashes, /*device_block_ids=*/{}, host_ids);
  auto response = future.Await();
  EXPECT_THAT(response.status(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(KVCacheStoreClientTest, FetchEmptyHashesReturnsEmptyResponse) {
  std::vector<std::string> hashes;
  tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse> future =
      client_->Fetch(hashes);
  TF_ASSERT_OK_AND_ASSIGN(auto response, future.Await());
  EXPECT_EQ(response.done_block_hashes_size(), 0);
}

TEST_F(KVCacheStoreClientTest, WriteRemoteUpdatesBlockTrackerOnSuccess) {
  BlockTracker tracker;
  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<int32_t> host_ids = {10, 11};
  tracker.AddPending(hashes);

  auto call = client_->WriteRemote(
      ::tpu_sync::rpc::RaidenIdProto(), hashes, host_ids, {},
      /*deadline_ms=*/5000, absl::Seconds(10), &tracker);
  auto ack = call.ack.Await();
  ABSL_ASSERT_OK(ack);
  EXPECT_EQ(ack->operation_id(), 12345);

  auto result = call.result.Await();
  ABSL_ASSERT_OK(result);
  EXPECT_EQ(result->state(),
            ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::COMMITTED);

  auto status = tracker.Poll();
  EXPECT_THAT(status.done, UnorderedElementsAre("hash_1", "hash_2"));
  EXPECT_TRUE(status.pending.empty());
}

TEST_F(KVCacheStoreClientTest, WriteRemoteValidatesArguments) {
  BlockTracker tracker;
  // Empty hashes
  auto empty_call =
      client_->WriteRemote(::tpu_sync::rpc::RaidenIdProto(), {}, {}, {}, 5000,
                           absl::Seconds(10), &tracker);
  EXPECT_THAT(empty_call.ack.Await().status(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Mismatched host ids
  auto mismatch_call =
      client_->WriteRemote(::tpu_sync::rpc::RaidenIdProto(), {"a"}, {1, 2}, {},
                           5000, absl::Seconds(10), &tracker);
  EXPECT_THAT(mismatch_call.ack.Await().status(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Non-positive deadline
  auto deadline_call =
      client_->WriteRemote(::tpu_sync::rpc::RaidenIdProto(), {"a"}, {1}, {}, 0,
                           absl::Seconds(10), &tracker);
  EXPECT_THAT(deadline_call.ack.Await().status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
