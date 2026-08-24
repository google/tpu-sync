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

#include "tpu_sync/kv_cache/kv_cache_store_service.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/support/status.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/core/buffer.h"
#include "tpu_sync/core/controller/controller_client.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/core/controller/test_util.h"
#include "tpu_sync/core/kv_manager_holder.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_server.h"
#include "tpu_sync/kv_cache/global_registry/test_util.h"
#include "tpu_sync/kv_cache/host_offload_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_client.h"
#include "tpu_sync/kv_cache/raiden_id.h"
namespace tpu_raiden {
namespace kv_cache {

namespace {

using ::absl_testing::StatusIs;
using ::testing::UnorderedElementsAre;

class KVCacheStoreServiceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up test worker server and mock transfer manager
    test_worker_server_ = ::tpu_raiden::controller::CreateTestWorkerServer();
    dst_transfer_mock_ = std::make_unique<
        ::tpu_raiden::controller::ShardAwareMockTransferManager>();
    test_worker_server_->service->SetTransferManager(
        ::tpu_raiden::KVManagerHolder(dst_transfer_mock_.get()));

    // Set up src controller server
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

    // Create dst KVCacheStore. These cases drive the WriteRemote handler
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

    // Pre-populate and pin test blocks in store so Fetch succeeds
    std::vector<std::string> test_hashes = {
        "block_hash_1", "block_hash_2", "block_hash_dev_1", "block_hash_dev_2"};
    std::vector<RaidenBlockId> slices = {
        RaidenBlockId(src_raiden_id, 10, BlockStatus::HOST),
        RaidenBlockId(src_raiden_id, 11, BlockStatus::HOST),
        RaidenBlockId(src_raiden_id, 12, BlockStatus::HOST),
        RaidenBlockId(src_raiden_id, 13, BlockStatus::HOST),
    };
    ASSERT_TRUE(store_->Insert(test_hashes, slices, /*on_host=*/true).ok());

    // Setup KVCacheStoreServiceImpl & gRPC server
    service_ = std::make_unique<KVCacheStoreServiceImpl>(
        store_->backend().get(), store_->raiden_controller());
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

  std::unique_ptr<::tpu_raiden::controller::TestWorkerServer>
      test_worker_server_;
  std::unique_ptr<::tpu_raiden::controller::ShardAwareMockTransferManager>
      dst_transfer_mock_;
  std::unique_ptr<::tpu_raiden::core::controller::TestControllerServer>
      src_controller_server_;
  std::unique_ptr<KVCacheStore> store_;
  std::unique_ptr<KVCacheStoreServiceImpl> service_;
  std::unique_ptr<::grpc::Server> server_;
  std::unique_ptr<KVCacheStoreClient> client_;
};

TEST_F(KVCacheStoreServiceTest, FetchEmptyRequest) {
  std::vector<std::string> empty_hashes;
  tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse> future =
      client_->Fetch(empty_hashes);
  auto response_or = future.Await();
  ASSERT_OK(response_or.status());
  EXPECT_EQ(response_or->done_block_hashes_size(), 0);
}

TEST_F(KVCacheStoreServiceTest, Fetch5StepWorkflowSuccess) {
  std::vector<std::string> hashes = {"block_hash_1", "block_hash_2"};
  std::vector<int32_t> host_block_ids = {100, 101};
  ::tpu_sync::rpc::RaidenIdProto client_id;
  client_id.set_job_name("client_job");
  client_id.set_job_replica_id("0");

  ::tpu_sync::proto::RaidenWorkerEndpointsProto client_ep;
  client_ep.set_node_id(0);
  client_ep.set_worker_id("dst_worker_0");
  auto* ep = client_ep.add_endpoints();
  ep->set_endpoint(test_worker_server_->server_address);

  tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse> future =
      client_->Fetch(hashes, /*device_block_ids=*/{}, host_block_ids, client_id,
                     {client_ep});
  auto response_or = future.Await();
  ASSERT_OK(response_or.status());
  EXPECT_THAT(response_or->done_block_hashes(),
              UnorderedElementsAre("block_hash_1", "block_hash_2"));
  EXPECT_EQ(response_or->failed_block_hashes_size(), 0);
}

TEST_F(KVCacheStoreServiceTest, FetchCrossNodeMissingEndpointsFails) {
  std::vector<std::string> hashes = {"block_hash_1"};
  std::vector<int32_t> host_block_ids = {100};
  ::tpu_sync::rpc::RaidenIdProto client_id;
  client_id.set_job_name("client_job");
  client_id.set_job_replica_id("0");

  tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse> future =
      client_->Fetch(hashes, /*device_block_ids=*/{}, host_block_ids,
                     client_id);
  auto response_or = future.Await();
  EXPECT_FALSE(response_or.status().ok());
  EXPECT_EQ(response_or.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(KVCacheStoreServiceTest, FetchValidationFailsForMissingHash) {
  std::vector<std::string> hashes = {"non_existent_hash"};
  std::vector<int32_t> host_block_ids = {100};

  tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse> future =
      client_->Fetch(hashes, /*device_block_ids=*/{}, host_block_ids);
  auto response_or = future.Await();
  EXPECT_THAT(response_or.status(), StatusIs(absl::StatusCode::kNotFound));
}

// Real block hashes are raw digests, not text. Every other case in this file
// uses an ASCII hash, which is why nothing here noticed while block_hashes was
// declared `string`: proto3 validates UTF-8 on PARSE, so a real digest
// serialized fine at the client and then failed to decode at the server.
//
// This exercises both directions -- the hash goes out on FetchRequest and
// comes back on FetchResponse -- because only the response side would catch a
// regression that fixed the request field alone. The payload is deliberately
// fixed rather than random: it holds a bare 0xFF, a lone continuation byte and
// an embedded NUL, so it is certainly invalid UTF-8 rather than merely
// almost-certainly.
TEST_F(KVCacheStoreServiceTest, FetchRoundTripsNonUtf8Hash) {
  const std::string binary_hash("\xff\xfe\x80\x00\x01\xc0\xaf\xed\xa0\x80", 10);
  RaidenId src_raiden_id{"src_job", "0", "src_data", 0};
  ASSERT_TRUE(store_->Insert(
      {binary_hash}, {RaidenBlockId(src_raiden_id, 20, BlockStatus::HOST)},
      /*on_host=*/true)
                  .ok());

  ::tpu_sync::rpc::RaidenIdProto client_id;
  client_id.set_job_name("client_job");
  client_id.set_job_replica_id("0");

  ::tpu_sync::proto::RaidenWorkerEndpointsProto client_ep;
  client_ep.set_node_id(0);
  client_ep.set_worker_id("dst_worker_0");
  client_ep.add_endpoints()->set_endpoint(test_worker_server_->server_address);

  tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse> future =
      client_->Fetch({binary_hash}, /*device_block_ids=*/{},
                     /*host_block_ids=*/{102}, client_id, {client_ep});
  auto response_or = future.Await();
  ASSERT_OK(response_or.status());
  EXPECT_THAT(response_or->done_block_hashes(),
              UnorderedElementsAre(binary_hash));
  EXPECT_EQ(response_or->failed_block_hashes_size(), 0);
}

TEST_F(KVCacheStoreServiceTest, FetchValidationFailsForNonHostBlock) {
  RaidenId src_raiden_id{"src_job", "0", "src_data", 0};
  std::vector<std::string> hashes = {"hbm_only_hash"};
  // HBM-only: local, so Insert takes it (REMOTE would be refused at the
  // gate), but not host-resident -- which is what Fetch validates.
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(src_raiden_id, /*host_block_id=*/-1,
                    /*device_block_id=*/50, BlockStatus::HBM),
  };
  ASSERT_TRUE(store_->Insert(hashes, slices, /*on_host=*/false).ok());

  std::vector<int32_t> host_block_ids = {100};
  tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse> future =
      client_->Fetch(hashes, /*device_block_ids=*/{}, host_block_ids);
  auto response_or = future.Await();
  EXPECT_THAT(response_or.status(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(KVCacheStoreServiceTest, FetchMismatchedHostBlockCount) {
  std::vector<std::string> hashes = {"block_hash_1", "block_hash_2"};
  std::vector<int32_t> host_block_ids = {100};  // Mismatched size!

  tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse> future =
      client_->Fetch(hashes, /*device_block_ids=*/{}, host_block_ids);
  auto response_or = future.Await();
  EXPECT_THAT(response_or.status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(KVCacheStoreServiceTest, FetchNullStoreHandling) {
  auto null_service =
      std::make_unique<KVCacheStoreServiceImpl>(nullptr, nullptr);
  ::grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(null_service.get());
  auto null_server = builder.BuildAndStart();

  auto null_client = std::make_unique<KVCacheStoreClient>(
      ::grpc::CreateChannel("localhost:" + std::to_string(port),
                            ::grpc::InsecureChannelCredentials()));

  std::vector<std::string> hashes = {"block_hash_1"};
  std::vector<int32_t> host_block_ids = {100};
  tsl::Future<::tpu_raiden::kv_cache::proto::FetchResponse> future =
      null_client->Fetch(hashes, /*device_block_ids=*/{}, host_block_ids);
  auto response_or = future.Await();
  EXPECT_THAT(response_or.status(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  null_server->Shutdown();
}

TEST_F(KVCacheStoreServiceTest, ConcurrentFetchRPCs) {
  RaidenId src_raiden_id{"src_job", "0", "src_data", 0};
  constexpr int kNumThreads = 8;
  std::vector<std::string> extra_hashes;
  std::vector<RaidenBlockId> extra_slices;
  extra_hashes.reserve(kNumThreads * 2);
  extra_slices.reserve(kNumThreads * 2);

  for (int i = 0; i < kNumThreads * 2; ++i) {
    extra_hashes.push_back("concurrent_hash_" + std::to_string(i));
    extra_slices.push_back(
        RaidenBlockId(src_raiden_id, 100 + i, BlockStatus::HOST));
  }
  ASSERT_TRUE(
      store_->Insert(extra_hashes, extra_slices, /*on_host=*/true).ok());

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  std::vector<absl::StatusOr<::tpu_raiden::kv_cache::proto::FetchResponse>>
      results(kNumThreads);

  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([this, i, &results]() {
      std::vector<std::string> hashes = {
          "concurrent_hash_" + std::to_string(2 * i),
          "concurrent_hash_" + std::to_string(2 * i + 1)};
      std::vector<int32_t> host_ids = {200 + 2 * i, 200 + 2 * i + 1};
      results[i] =
          client_->Fetch(hashes, /*device_block_ids=*/{}, host_ids).Await();
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  for (int i = 0; i < kNumThreads; ++i) {
    ASSERT_OK(results[i].status());
    EXPECT_THAT(
        results[i]->done_block_hashes(),
        UnorderedElementsAre("concurrent_hash_" + std::to_string(2 * i),
                             "concurrent_hash_" + std::to_string(2 * i + 1)));
  }
}

::tpu_sync::proto::RaidenWorkerEndpointsProto MakeGroup(
    int64_t node_id, absl::string_view worker_id, absl::string_view endpoint) {
  ::tpu_sync::proto::RaidenWorkerEndpointsProto group;
  group.set_node_id(node_id);
  group.set_worker_id(std::string(worker_id));
  auto* ep = group.add_endpoints();
  ep->set_endpoint(std::string(endpoint));
  ep->add_shards(0);
  return group;
}

TEST_F(KVCacheStoreServiceTest, FetchRoutesToClientAdvertisedEndpoints) {
  ::tpu_sync::rpc::RaidenIdProto client_id;
  client_id.set_job_name("client_job");
  client_id.set_job_replica_id("0");
  client_id.set_data_name("client_data");
  client_id.set_data_replica_idx(0);

  // The fixture's source controller has one worker, registered with node_id 0.
  std::vector<::tpu_sync::proto::RaidenWorkerEndpointsProto> groups = {
      MakeGroup(/*node_id=*/0, "client_worker_0", "10.0.0.9:44001")};

  std::vector<std::string> hashes = {"block_hash_1"};
  std::vector<int32_t> host_ids = {201};
  auto res =
      client_
          ->Fetch(hashes, /*device_block_ids=*/{}, host_ids, client_id, groups)
          .Await();
  ASSERT_TRUE(res.ok()) << res.status();

  ASSERT_EQ(dst_transfer_mock_->last_write_descriptors.size(), 1);
  EXPECT_EQ(dst_transfer_mock_->last_write_descriptors[0].endpoint,
            "10.0.0.9:44001");
}

TEST_F(KVCacheStoreServiceTest, CrossNodeFetchWithoutEndpointsIsRejected) {
  ::tpu_sync::rpc::RaidenIdProto client_id;
  client_id.set_job_name("client_job");
  client_id.set_job_replica_id("0");
  client_id.set_data_name("client_data");
  client_id.set_data_replica_idx(0);

  dst_transfer_mock_->last_write_descriptors.clear();

  std::vector<std::string> hashes = {"block_hash_1"};
  std::vector<int32_t> host_ids = {201};
  auto res =
      client_->Fetch(hashes, /*device_block_ids=*/{}, host_ids, client_id)
          .Await();

  EXPECT_THAT(res.status(), StatusIs(absl::StatusCode::kInvalidArgument));
  // Nothing was transferred -- in particular nothing was written locally.
  EXPECT_TRUE(dst_transfer_mock_->last_write_descriptors.empty());
}

TEST_F(KVCacheStoreServiceTest, SameNodeFetchNeedsNoEndpoints) {
  std::vector<std::string> hashes = {"block_hash_1"};
  std::vector<int32_t> host_ids = {201};
  auto res = client_->Fetch(hashes, /*device_block_ids=*/{}, host_ids).Await();
  EXPECT_TRUE(res.ok()) << res.status();
}

// Two workers on this node, each with its own transfer manager, and two client
// groups. Each worker must be handed ONLY the group carrying its own node_id.
// raiden_controller_test proves the pairing inside TransferBuffers; this
// proves the Fetch path feeds it correctly, which is what changed.
TEST_F(KVCacheStoreServiceTest, FetchWithMultiWorkerEndpointsRoutesPerWorker) {
  auto worker_a = ::tpu_raiden::controller::CreateTestWorkerServer();
  auto mock_a = std::make_unique<
      ::tpu_raiden::controller::ShardAwareMockTransferManager>();
  worker_a->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(mock_a.get()));

  auto worker_b = ::tpu_raiden::controller::CreateTestWorkerServer();
  auto mock_b = std::make_unique<
      ::tpu_raiden::controller::ShardAwareMockTransferManager>();
  worker_b->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(mock_b.get()));

  // A fresh store so only these two workers are registered, with distinct
  // NON-ZERO node_ids: node_id 0 is exempt from the registry's uniqueness
  // check, and peer_node_id_to_endpoints keeps the first group per node_id, so
  // workers left at the default would collapse into one and every worker would
  // be routed to it.
  RaidenId multi_id{"multi_job", "0", "multi_data", 0};
  KVCacheStore store(/*capacity=*/16, /*global_registry_address=*/"", multi_id,
                     /*num_shards=*/1, /*shard_size_bytes=*/1024,
                     /*store_server_ip=*/"127.0.0.1");
  ::tpu_raiden::core::controller::RaidenControllerClient ctrl_client(
      store.raiden_controller_address());
  ASSERT_OK(ctrl_client.RegisterWorker("w_a", worker_a->server_address,
                                       {{worker_a->server_address, {}}},
                                       /*node_id=*/10));
  ASSERT_OK(ctrl_client.RegisterWorker("w_b", worker_b->server_address,
                                       {{worker_b->server_address, {}}},
                                       /*node_id=*/20));

  std::vector<std::string> hashes = {"multi_hash"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(multi_id, 7, BlockStatus::HOST)};
  ASSERT_TRUE(store.Insert(hashes, slices, /*on_host=*/true).ok());

  KVCacheStoreServiceImpl service(store.backend().get(),
                                  store.raiden_controller());
  ::grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  KVCacheStoreClient client(
      ::grpc::CreateChannel("localhost:" + std::to_string(port),
                            ::grpc::InsecureChannelCredentials()));

  ::tpu_sync::rpc::RaidenIdProto client_id;
  client_id.set_job_name("client_job");
  client_id.set_job_replica_id("0");

  std::vector<::tpu_sync::proto::RaidenWorkerEndpointsProto> groups = {
      MakeGroup(/*node_id=*/10, "client_w_a", "10.0.0.10:45001"),
      MakeGroup(/*node_id=*/20, "client_w_b", "10.0.0.20:45002")};

  auto res = client
                 .Fetch(hashes, /*device_block_ids=*/{},
                        /*host_block_ids=*/{301}, client_id, groups)
                 .Await();
  ASSERT_OK(res.status());

  // Each worker saw its own peer -- not the other's, and not both.
  ASSERT_EQ(mock_a->last_write_descriptors.size(), 1);
  EXPECT_EQ(mock_a->last_write_descriptors[0].endpoint, "10.0.0.10:45001");
  ASSERT_EQ(mock_b->last_write_descriptors.size(), 1);
  EXPECT_EQ(mock_b->last_write_descriptors[0].endpoint, "10.0.0.20:45002");

  server->Shutdown();
}

// Groups matching no local worker fail rather than being broadcast.
//
// NOTE: the rejection is clean here only because the single group is
// unmatched. TransferBuffers validates worker N+1 after dispatching worker N,
// so with several workers the error can arrive with an earlier write already
// in flight.
TEST_F(KVCacheStoreServiceTest, FetchWithUnmatchedNodeIdFails) {
  ::tpu_sync::rpc::RaidenIdProto client_id;
  client_id.set_job_name("client_job");
  client_id.set_job_replica_id("0");

  // The fixture's worker is registered with the default node_id 0.
  std::vector<::tpu_sync::proto::RaidenWorkerEndpointsProto> groups = {
      MakeGroup(/*node_id=*/99, "client_worker_x", "10.0.0.99:46001")};

  auto res = client_
                 ->Fetch({"block_hash_1"}, /*device_block_ids=*/{},
                         /*host_block_ids=*/{202}, client_id, groups)
                 .Await();

  EXPECT_THAT(res.status(), StatusIs(absl::StatusCode::kInternal,
                                     ::testing::HasSubstr("node_id")));
}

// ===========================================================================
// Remote write -- the destination state machine.
// ===========================================================================

// Holds a transfer pending until the test says otherwise. RaidenController has
// no virtual methods, so without the service's transfer seam there is no way
// to observe a destination between "issued the pull" and "the bytes arrived",
// which is where every race in this design lives.
class TransferLatch {
 public:
  tsl::Future<> Issue() {
    auto [promise, future] = tsl::MakePromise<>();
    absl::MutexLock lock(mutex_);
    promise_ = std::move(promise);
    ++issued_;
    return future;
  }

  void Release(absl::Status status) {
    absl::MutexLock lock(mutex_);
    ASSERT_TRUE(promise_.has_value()) << "no transfer was issued";
    promise_->Set(std::move(status));
    promise_.reset();
  }

  int issued() const {
    absl::MutexLock lock(mutex_);
    return issued_;
  }

 private:
  mutable absl::Mutex mutex_;
  std::optional<tsl::Promise<>> promise_;
  int issued_ = 0;
};

class WriteRemoteTest : public ::testing::Test {
 protected:
  static constexpr int kCapacity = 8;

  void SetUp() override {
    dst_id_ = RaidenId{"dst_job", "0", "dst_data", 0};
    // No registry: RegisterBlocksSync then succeeds trivially, which keeps
    // these cases about the state machine rather than about publication.
    store_ = std::make_unique<KVCacheStore>(
        /*capacity=*/kCapacity, /*global_registry_address=*/"", dst_id_,
        /*num_shards=*/1, /*shard_size_bytes=*/1024,
        /*store_server_ip=*/"127.0.0.1");

    service_ = std::make_unique<KVCacheStoreServiceImpl>(
        store_->backend().get(), store_->raiden_controller());
    service_->SetTransferFnForTesting(
        [this](absl::Span<const Buffer>, absl::Span<const Buffer>) {
          return latch_.Issue();
        });

    ::grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(),
                             &port);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    client_ = std::make_unique<KVCacheStoreClient>(
        ::grpc::CreateChannel("localhost:" + std::to_string(port),
                              ::grpc::InsecureChannelCredentials()));
  }

  void TearDown() override {
    if (server_) server_->Shutdown();
  }

  static ::tpu_sync::rpc::RaidenIdProto SrcIdProto() {
    ::tpu_sync::rpc::RaidenIdProto id;
    id.set_job_name("src_job");
    id.set_job_replica_id("0");
    id.set_data_name("src_data");
    id.set_data_replica_idx(0);
    return id;
  }

  static std::vector<::tpu_sync::proto::RaidenWorkerEndpointsProto>
  SrcEndpoints() {
    ::tpu_sync::proto::RaidenWorkerEndpointsProto group;
    group.set_node_id(0);
    group.set_worker_id("src_worker_0");
    group.add_endpoints()->set_endpoint("127.0.0.1:9999");
    return {group};
  }

  absl::StatusOr<::tpu_raiden::kv_cache::proto::WriteRemoteResponse> Offer(
      const std::vector<std::string>& hashes, int64_t deadline_ms = 5000) {
    std::vector<int32_t> src_ids(hashes.size(), 0);
    for (size_t i = 0; i < src_ids.size(); ++i) src_ids[i] = 100 + i;
    return client_
        ->WriteRemote(SrcIdProto(), hashes, src_ids, SrcEndpoints(),
                      deadline_ms)
        .Await();
  }

  absl::StatusOr<::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse> Poll(
      uint64_t op_id) {
    return client_->PollWriteRemote(op_id).Await();
  }

  // Polls until the operation leaves PENDING, or gives up. The completion runs
  // on the transfer's callback thread, so a terminal state is not visible the
  // instant Release() returns.
  ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::State AwaitTerminal(
      uint64_t op_id) {
    for (int i = 0; i < 200; ++i) {
      auto poll = Poll(op_id);
      if (poll.ok() &&
          poll->state() !=
              ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::PENDING) {
        last_poll_ = *poll;
        return poll->state();
      }
      absl::SleepFor(absl::Milliseconds(10));
    }
    return ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::PENDING;
  }

  RaidenId dst_id_;
  TransferLatch latch_;
  std::unique_ptr<KVCacheStore> store_;
  std::unique_ptr<KVCacheStoreServiceImpl> service_;
  std::unique_ptr<::grpc::Server> server_;
  std::unique_ptr<KVCacheStoreClient> client_;
  ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse last_poll_;
};

TEST_F(WriteRemoteTest, RejectsAnEmptyOffer) {
  std::vector<int32_t> no_ids;
  auto response = client_
                      ->WriteRemote(SrcIdProto(), {}, no_ids, SrcEndpoints(),
                                    /*deadline_ms=*/5000)
                      .Await();
  EXPECT_THAT(response.status(), StatusIs(absl::StatusCode::kInvalidArgument));
}

// Only once a transfer is actually going to happen. An offer the destination
// can answer from what it already holds needs no data plane.
TEST_F(WriteRemoteTest, RejectsMissingSourceEndpointsWhenAPullIsNeeded) {
  std::vector<int32_t> src_ids = {100};
  auto response =
      client_->WriteRemote(SrcIdProto(), {"a"}, src_ids, {}, 5000).Await();
  EXPECT_THAT(response.status(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(WriteRemoteTest, AllExistNeedsNoSourceEndpoints) {
  ASSERT_TRUE(store_->backend()->InsertAllOrNothing(
      {"a"}, {RaidenBlockId(dst_id_, 1, BlockStatus::HOST)}));

  std::vector<int32_t> src_ids = {100};
  auto response =
      client_->WriteRemote(SrcIdProto(), {"a"}, src_ids, {}, 5000).Await();
  ASSERT_OK(response.status());
  EXPECT_EQ(response->exist_state(),
            ::tpu_raiden::kv_cache::proto::WRITE_ALL_EXIST);
}

// proto3 gives a scalar no presence, so an unset deadline arrives as 0. If 0
// meant "already expired" every write would fail at the deadline check with a
// message about deadlines, which is not what is wrong.
TEST_F(WriteRemoteTest, RejectsAnUnsetDeadline) {
  std::vector<int32_t> src_ids = {100};
  ::tpu_raiden::kv_cache::proto::WriteRemoteRequest request;
  *request.mutable_src_raiden_id() = SrcIdProto();
  request.add_block_hashes("a");
  request.add_src_host_block_ids(100);
  *request.add_src_worker_endpoints() = SrcEndpoints()[0];
  request.set_deadline_ms(0);

  ::tpu_raiden::kv_cache::proto::WriteRemoteResponse response;
  ::grpc::ServerContext context;
  auto status = service_->WriteRemote(&context, &request, &response);
  EXPECT_EQ(status.error_code(), ::grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("deadline_ms"));
}

TEST_F(WriteRemoteTest, RejectsAnOfferFromItself) {
  ::tpu_sync::rpc::RaidenIdProto self;
  self.set_job_name(dst_id_.job_name);
  self.set_job_replica_id(dst_id_.job_replica_id);
  self.set_data_name(dst_id_.data_name);
  self.set_data_replica_idx(dst_id_.data_replica_idx);

  std::vector<int32_t> src_ids = {100};
  auto response =
      client_->WriteRemote(self, {"a"}, src_ids, SrcEndpoints(), 5000).Await();
  EXPECT_THAT(response.status(), StatusIs(absl::StatusCode::kInvalidArgument));
}

// SUCCESS, not failure: hashes are content-addressed, so the destination
// already having them is the post-condition the source wanted.
TEST_F(WriteRemoteTest, AllExistIsAnImmediateSuccess) {
  ASSERT_TRUE(store_->backend()->InsertAllOrNothing(
      {"a", "b"}, {RaidenBlockId(dst_id_, 1, BlockStatus::HOST),
                   RaidenBlockId(dst_id_, 2, BlockStatus::HOST)}));

  auto response = Offer({"a", "b"});
  ASSERT_OK(response.status());
  EXPECT_EQ(response->operation_id(), 0);
  EXPECT_EQ(response->exist_state(),
            ::tpu_raiden::kv_cache::proto::WRITE_ALL_EXIST);
  EXPECT_EQ(latch_.issued(), 0) << "nothing should have been transferred";
}

// FAILURE, and the destination does not do partial writes: reporting success
// would let an eviction caller free blocks this node does not have.
TEST_F(WriteRemoteTest, PartialExistIsRefusedAndNamesWhatItHas) {
  ASSERT_TRUE(store_->backend()->InsertAllOrNothing(
      {"a"}, {RaidenBlockId(dst_id_, 1, BlockStatus::HOST)}));

  auto response = Offer({"a", "b"});
  ASSERT_OK(response.status());
  EXPECT_EQ(response->operation_id(), 0);
  EXPECT_EQ(response->exist_state(),
            ::tpu_raiden::kv_cache::proto::WRITE_PARTIAL_EXIST);
  EXPECT_THAT(response->existing_hashes(), UnorderedElementsAre("a"));
  EXPECT_EQ(latch_.issued(), 0);
  EXPECT_EQ(store_->backend()->GetSize(), 1) << "nothing should have landed";
}

TEST_F(WriteRemoteTest, AcceptsAndAnswersWithoutWaitingForBytes) {
  auto response = Offer({"a", "b"});
  ASSERT_OK(response.status());
  EXPECT_NE(response->operation_id(), 0);
  EXPECT_EQ(response->exist_state(),
            ::tpu_raiden::kv_cache::proto::WRITE_EXIST_STATE_UNSPECIFIED);
  EXPECT_GT(response->granted_deadline_ms(), 0);
  // The handler returned while the transfer is still outstanding. That is the
  // contract, and it is what stops a slow peer occupying a handler thread.
  EXPECT_EQ(latch_.issued(), 1);
  EXPECT_EQ(Poll(response->operation_id())->state(),
            ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::PENDING);

  latch_.Release(absl::OkStatus());
  EXPECT_EQ(AwaitTerminal(response->operation_id()),
            ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::COMMITTED);
}

// The cap bounds how long this node holds landing blocks for ANY source,
// whatever that source asks for.
TEST_F(WriteRemoteTest, GrantedDeadlineIsClampedToTheLocalCap) {
  auto response =
      Offer({"a"}, /*deadline_ms=*/absl::ToInt64Milliseconds(absl::Hours(1)));
  ASSERT_OK(response.status());
  EXPECT_LE(response->granted_deadline_ms(),
            absl::ToInt64Milliseconds(absl::Seconds(25)));
  latch_.Release(absl::CancelledError("done with this test"));
  AwaitTerminal(response->operation_id());
}

TEST_F(WriteRemoteTest, CommitInsertsTheBlocksAndReportsThem) {
  auto response = Offer({"a", "b"});
  ASSERT_OK(response.status());
  latch_.Release(absl::OkStatus());

  ASSERT_EQ(AwaitTerminal(response->operation_id()),
            ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::COMMITTED);
  EXPECT_THAT(last_poll_.committed_hashes(), UnorderedElementsAre("a", "b"));
  EXPECT_THAT(store_->backend()->AlreadyPresentHostResident({"a", "b"}),
              UnorderedElementsAre("a", "b"));
}

TEST_F(WriteRemoteTest, TransferFailureFreesTheLandingBlocks) {
  auto response = Offer({"a", "b"});
  ASSERT_OK(response.status());
  latch_.Release(absl::InternalError("pull failed"));

  ASSERT_EQ(AwaitTerminal(response->operation_id()),
            ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::FAILED);
  EXPECT_THAT(last_poll_.failed_hashes(), UnorderedElementsAre("a", "b"));
  EXPECT_TRUE(
      store_->backend()->AlreadyPresentHostResident({"a", "b"}).empty());
  // The blocks came back: the whole pool is allocatable again.
  auto reallocated = store_->raiden_controller()->AllocateBlockIds(kCapacity);
  EXPECT_TRUE(reallocated.ok()) << reallocated.status().ToString();
}

// The claim rule, which is the one invariant everything else leans on: a
// transfer that resolves after the deadline must not insert or register
// anything, however it lost the race. The bytes are discarded and the blocks
// come back.
TEST_F(WriteRemoteTest, ATransferThatResolvesPastTheDeadlineNeverCommits) {
  auto response = Offer({"a", "b"}, /*deadline_ms=*/100);
  ASSERT_OK(response.status());
  const uint64_t op_id = response->operation_id();

  absl::SleepFor(absl::Milliseconds(300));
  // The verdict is already FAILED, but the blocks are still held: only the
  // workers reporting may release them.
  ASSERT_EQ(Poll(op_id)->state(),
            ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::FAILED);

  latch_.Release(absl::OkStatus());
  // Give the completion time to run and lose the claim.
  absl::SleepFor(absl::Milliseconds(200));

  EXPECT_EQ(Poll(op_id)->state(),
            ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::FAILED);
  EXPECT_TRUE(store_->backend()->AlreadyPresentHostResident({"a", "b"}).empty())
      << "a post-deadline transfer inserted its bytes anyway";
  auto reallocated = store_->raiden_controller()->AllocateBlockIds(kCapacity);
  EXPECT_TRUE(reallocated.ok()) << "the deferred free never happened: "
                                << reallocated.status().ToString();
}

// Same rule as the case above, but reached WITHOUT the deadline thread having
// run. The claim compares against the wall clock precisely so that commit
// safety does not depend on that thread being scheduled -- a wedged one must
// not be able to let a commit through after the source has unpinned.
TEST_F(WriteRemoteTest, TheClaimRefusesALateTransferEvenIfNoThreadFiredIt) {
  service_->PauseDeadlineFiringForTesting();

  auto response = Offer({"a", "b"}, /*deadline_ms=*/100);
  ASSERT_OK(response.status());
  const uint64_t op_id = response->operation_id();

  absl::SleepFor(absl::Milliseconds(300));
  // Nothing fired the deadline, so the operation still reads as PENDING.
  ASSERT_EQ(Poll(op_id)->state(),
            ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::PENDING);

  latch_.Release(absl::OkStatus());
  ASSERT_EQ(AwaitTerminal(op_id),
            ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::FAILED)
      << "a transfer that resolved past its deadline was allowed to commit";
  EXPECT_TRUE(
      store_->backend()->AlreadyPresentHostResident({"a", "b"}).empty());
  auto reallocated = store_->raiden_controller()->AllocateBlockIds(kCapacity);
  EXPECT_TRUE(reallocated.ok()) << reallocated.status().ToString();
}

// A concurrent writer landed the hashes while the bytes were in flight. The
// same existence rule as at ack time, which is what keeps the two coherent.
// This is the CLAIMED path's free: the operation succeeded at claiming and
// then found nothing to do, so its landing blocks must still come back.
TEST_F(WriteRemoteTest, LosingTheRaceAtInsertTimeReportsAllExistAndFrees) {
  auto response = Offer({"a", "b"});
  ASSERT_OK(response.status());

  ASSERT_TRUE(store_->backend()->InsertAllOrNothing(
      {"a", "b"}, {RaidenBlockId(dst_id_, 6, BlockStatus::HOST),
                   RaidenBlockId(dst_id_, 7, BlockStatus::HOST)}));

  latch_.Release(absl::OkStatus());
  ASSERT_EQ(AwaitTerminal(response->operation_id()),
            ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::ALL_EXIST);
  auto reallocated = store_->raiden_controller()->AllocateBlockIds(kCapacity);
  EXPECT_TRUE(reallocated.ok()) << "the landing blocks were never returned: "
                                << reallocated.status().ToString();
}

// The other arrival path for a PARTIAL_EXIST verdict. At ack time nothing was
// present, so the offer was accepted; by the time the bytes landed a
// concurrent writer had committed one of the two hashes. The source must get
// the same answer, and the same list, whichever reply carried it -- otherwise
// it would have to remember which call to look at.
TEST_F(WriteRemoteTest, PartialExistDiscoveredAtInsertTimeReachesThePoll) {
  auto response = Offer({"a", "b"});
  ASSERT_OK(response.status());

  ASSERT_TRUE(store_->backend()->InsertAllOrNothing(
      {"a"}, {RaidenBlockId(dst_id_, 6, BlockStatus::HOST)}));

  latch_.Release(absl::OkStatus());
  ASSERT_EQ(
      AwaitTerminal(response->operation_id()),
      ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::PARTIAL_EXIST);
  EXPECT_THAT(last_poll_.existing_hashes(), UnorderedElementsAre("a"));
  // "b" was never inserted: this destination does not do partial writes.
  EXPECT_TRUE(store_->backend()->AlreadyPresentHostResident({"b"}).empty());
  auto reallocated = store_->raiden_controller()->AllocateBlockIds(kCapacity);
  EXPECT_TRUE(reallocated.ok()) << reallocated.status().ToString();
}

// Free blocks only -- this path never evicts, which is why a warm destination
// refuses every write until destination-side eviction is implemented. The
// cache must be untouched: that is what makes this a deliberate omission
// rather than an eviction bug.
TEST_F(WriteRemoteTest, RefusesWhenThereAreNoFreeBlocksAndEvictsNothing) {
  ASSERT_TRUE(store_->backend()->InsertAllOrNothing(
      {"victim"}, {RaidenBlockId(dst_id_, 0, BlockStatus::HOST)}));
  auto drained = store_->raiden_controller()->AllocateBlockIds(kCapacity);
  ASSERT_TRUE(drained.ok()) << drained.status().ToString();

  auto response = Offer({"a"});
  EXPECT_THAT(response.status(),
              StatusIs(absl::StatusCode::kResourceExhausted));
  EXPECT_EQ(latch_.issued(), 0);
  EXPECT_THAT(store_->backend()->AlreadyPresentHostResident({"victim"}),
              UnorderedElementsAre("victim"));
}

TEST_F(WriteRemoteTest, PollOfAnUnknownOperationIsUnknown) {
  auto poll = Poll(999999);
  ASSERT_OK(poll.status());
  EXPECT_EQ(poll->state(),
            ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::UNKNOWN);
}

TEST_F(WriteRemoteTest, PollRejectsTheReservedOperationId) {
  ::tpu_raiden::kv_cache::proto::PollWriteRemoteRequest request;
  request.set_operation_id(0);
  ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse response;
  ::grpc::ServerContext context;
  auto status = service_->PollWriteRemote(&context, &request, &response);
  EXPECT_EQ(status.error_code(), ::grpc::StatusCode::INVALID_ARGUMENT);
}

// When the transfer completes, the thread that called Release() on that
// completion must NOT be held while the registry call runs. A stalled registry
// must not delay completing the transfer.
TEST(WriteRemotePublishTest, PublishDoesNotBlockTheTransferCompletion) {
  auto impl = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  auto stalling_service =
      std::make_unique<global_registry::StallingRegistryService>(
          std::move(impl));
  auto* registry_service = stalling_service.get();
  registry_service->EnableStall();

  auto registry_server =
      global_registry::CreateTestGlobalRegistryServerWithService(
          std::move(stalling_service));

  const RaidenId dst_id{"dst_job_publish", "0", "dst_data", 0};
  KVCacheStore store(/*capacity=*/8, registry_server->server_address, dst_id,
                     /*num_shards=*/1, /*shard_size_bytes=*/1024,
                     /*store_server_ip=*/"127.0.0.1");

  KVCacheStoreServiceImpl service(store.backend().get(),
                                  store.raiden_controller());
  TransferLatch latch;
  service.SetTransferFnForTesting(
      [&latch](absl::Span<const Buffer>, absl::Span<const Buffer>) {
        return latch.Issue();
      });

  ::grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  KVCacheStoreClient client(
      ::grpc::CreateChannel("localhost:" + std::to_string(port),
                            ::grpc::InsecureChannelCredentials()));

  ::tpu_sync::rpc::RaidenIdProto src_id;
  src_id.set_job_name("src_job_publish");
  src_id.set_job_replica_id("0");
  src_id.set_data_name("src_data");
  ::tpu_sync::proto::RaidenWorkerEndpointsProto group;
  group.set_node_id(0);
  group.set_worker_id("src_worker_0");
  group.add_endpoints()->set_endpoint("127.0.0.1:9999");

  std::vector<std::string> hashes = {"publish_a", "publish_b"};
  std::vector<int32_t> src_ids = {200, 201};
  auto response =
      client.WriteRemote(src_id, hashes, src_ids, {group}, 5000).Await();
  ASSERT_OK(response.status());
  const uint64_t op_id = response->operation_id();
  ASSERT_NE(op_id, 0);

  // Hold the registry for kStall once it has been reached. A synchronous
  // publish would make Release() below cost that long; an asynchronous one
  // costs nothing. Released from another thread rather than after the timing
  // window, so that a regression fails an assertion instead of deadlocking.
  constexpr absl::Duration kStall = absl::Seconds(2);
  std::thread releaser([registry_service, kStall] {
    registry_service->WaitForStall();
    absl::SleepFor(kStall);
    registry_service->ReleaseStall();
  });

  const absl::Time before = absl::Now();
  latch.Release(absl::OkStatus());
  const absl::Duration completion_cost = absl::Now() - before;

  EXPECT_LT(completion_cost, kStall / 2)
      << "completing the transfer took " << completion_cost
      << ", which means it waited on the registry stalled for " << kStall;

  // ... and the operation has not settled, which is what proves the wait that
  // did not happen was a real one rather than a registry that answered fast.
  auto polled = client.PollWriteRemote(op_id).Await();
  ASSERT_OK(polled.status());
  EXPECT_EQ(polled->state(),
            ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::PENDING);

  releaser.join();

  ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse final_poll;
  for (int i = 0; i < 300; ++i) {
    auto p = client.PollWriteRemote(op_id).Await();
    ASSERT_OK(p.status());
    if (p->state() !=
        ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::PENDING) {
      final_poll = *p;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  // Committed on a gRPC callback thread, not on the thread that released the
  // transfer -- the half of the change that the Lifetime guard protects.
  EXPECT_EQ(final_poll.state(),
            ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::COMMITTED);
  EXPECT_THAT(final_poll.committed_hashes(),
              UnorderedElementsAre("publish_a", "publish_b"));

  auto looked_up = registry_server->client->Lookup(hashes);
  ASSERT_OK(looked_up.status());
  EXPECT_EQ(looked_up->size(), 2)
      << "the landed blocks were never advertised";

  server->Shutdown();
}

// The bytes arrive but the destination cannot publish them.
//
// Not a failure and not a commit. Discarding them would throw away a completed
// transfer to fix nothing -- an unpublished block is invisible, not wrong.
// Reporting COMMITTED would be worse: that is what licenses the source to drop
// its own copy, and the block would go from findable-on-the-source to
// findable-by-nobody. So the destination keeps them, says so, and lets the
// source decide.
//
// A live registry is required at construction (a store that cannot register
// itself does not build), so this kills the registry afterwards, which is also
// how it happens in practice.
TEST(WriteRemoteRegistryFailureTest, StoredButUnregisteredIsReportedAsSuch) {
  auto registry_server = global_registry::CreateTestGlobalRegistryServer();

  const RaidenId dst_id{"dst_job_unreg", "0", "dst_data", 0};
  KVCacheStore store(/*capacity=*/8, registry_server->server_address, dst_id,
                     /*num_shards=*/1, /*shard_size_bytes=*/1024,
                     /*store_server_ip=*/"127.0.0.1");

  KVCacheStoreServiceImpl service(store.backend().get(),
                                  store.raiden_controller());
  TransferLatch latch;
  service.SetTransferFnForTesting(
      [&latch](absl::Span<const Buffer>, absl::Span<const Buffer>) {
        return latch.Issue();
      });

  ::grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  KVCacheStoreClient client(
      ::grpc::CreateChannel("localhost:" + std::to_string(port),
                            ::grpc::InsecureChannelCredentials()));

  ::tpu_sync::rpc::RaidenIdProto src_id;
  src_id.set_job_name("src_job_unreg");
  src_id.set_job_replica_id("0");
  src_id.set_data_name("src_data");
  ::tpu_sync::proto::RaidenWorkerEndpointsProto group;
  group.set_node_id(0);
  group.set_worker_id("src_worker_0");
  group.add_endpoints()->set_endpoint("127.0.0.1:9999");

  std::vector<std::string> hashes = {"unreg_a", "unreg_b"};
  std::vector<int32_t> src_ids = {100, 101};
  auto response =
      client.WriteRemote(src_id, hashes, src_ids, {group}, 5000).Await();
  ASSERT_OK(response.status());
  const uint64_t op_id = response->operation_id();
  ASSERT_NE(op_id, 0);

  // The transfer is in flight; take the registry away before it lands.
  registry_server->server->Shutdown();
  latch.Release(absl::OkStatus());

  ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse poll;
  for (int i = 0; i < 300; ++i) {
    auto polled = client.PollWriteRemote(op_id).Await();
    ASSERT_OK(polled.status());
    if (polled->state() !=
        ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::PENDING) {
      poll = *polled;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }

  EXPECT_EQ(poll.state(), ::tpu_raiden::kv_cache::proto::
                              PollWriteRemoteResponse::STORED_UNREGISTERED);
  EXPECT_THAT(poll.unregistered_hashes(),
              UnorderedElementsAre("unreg_a", "unreg_b"));
  EXPECT_TRUE(poll.committed_hashes().empty())
      << "STORED_UNREGISTERED must not be reported as committed";

  // The bytes were KEPT -- that is the whole point of not rolling back.
  EXPECT_THAT(store.backend()->AlreadyPresentHostResident(hashes),
              UnorderedElementsAre("unreg_a", "unreg_b"));
  // And the landing blocks belong to the cache now, so they are NOT back in
  // the pool: the store holds 8 blocks and 2 are spoken for.
  EXPECT_FALSE(store.raiden_controller()->AllocateBlockIds(8).ok())
      << "the landing blocks were returned to the pool while the cache still "
         "points at them";

  server->Shutdown();
}

// Teardown with a transfer still outstanding must not hang and must not leave
// a callback pointing at freed memory. The bound is what keeps this from
// hanging every GetServerAddress() caller, since the destructor runs under the
// server's own mutex.
TEST_F(WriteRemoteTest, TeardownWithAnOutstandingTransferIsBoundedAndSafe) {
  auto response = Offer({"a", "b"});
  ASSERT_OK(response.status());
  ASSERT_EQ(latch_.issued(), 1);

  server_->Shutdown();
  server_.reset();

  const absl::Time started = absl::Now();
  service_.reset();
  EXPECT_LT(absl::Now() - started, absl::Seconds(30))
      << "teardown waited longer than its own bound";

  // The late completion must find no service and do nothing, rather than
  // touch the object that was just destroyed.
  latch_.Release(absl::OkStatus());
  absl::SleepFor(absl::Milliseconds(100));
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
