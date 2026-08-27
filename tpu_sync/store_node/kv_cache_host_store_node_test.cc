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

#include "tpu_sync/store_node/kv_cache_host_store_node.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/global_registry/global_registry.pb.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_client.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_server.h"
#include "tpu_sync/kv_cache/global_registry/test_util.h"
#include "tpu_sync/store_node/grs_kv_transfer_spec_source.h"
#include "tpu_sync/store_node/kv_transfer_spec_source.h"

namespace tpu_raiden {
namespace store_node {
namespace {

using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

// Yields `status` until `failures` calls have been made, then the spec.
class FlakyKVTransferSpecSource : public KVTransferSpecSource {
 public:
  FlakyKVTransferSpecSource(KVTransferSpec spec, int failures,
                            absl::Status status)
      : spec_(spec), failures_(failures), status_(status) {}

  absl::StatusOr<KVTransferSpec> Get() override {
    ++calls_;
    if (calls_ <= failures_) {
      return status_;
    }
    return spec_;
  }

  int calls() const { return calls_; }

 private:
  KVTransferSpec spec_;
  int failures_;
  absl::Status status_;
  int calls_ = 0;
};

// Two block arrays of 256 bytes each on 1 shard: one block is 512 bytes.
KVTransferSpec TestSpec() {
  return KVTransferSpec{/*block_array_bytes=*/{256, 256},
                        /*num_kv_shards=*/1, /*num_workers=*/1};
}

KVCacheHostStoreNode::Options FastPollOptions() {
  KVCacheHostStoreNode::Options options;
  options.spec_poll_initial = absl::Milliseconds(1);
  options.spec_poll_max = absl::Milliseconds(4);
  return options;
}

TEST(ValidateSpecTest, RejectsEmptyOrZeroQuantities) {
  EXPECT_TRUE(ValidateSpec(TestSpec()).ok());
  EXPECT_TRUE(absl::IsInvalidArgument(
      ValidateSpec(KVTransferSpec{/*block_array_bytes=*/{}, 1, 1})));
  EXPECT_TRUE(absl::IsInvalidArgument(
      ValidateSpec(KVTransferSpec{{256, 256}, 0, 1})));
  EXPECT_TRUE(absl::IsInvalidArgument(
      ValidateSpec(KVTransferSpec{{256, 0}, 1, 1})));
  EXPECT_TRUE(absl::IsInvalidArgument(
      ValidateSpec(KVTransferSpec{{256, 256}, 1, 0})));
}

TEST(WaitForSpecTest, RetriesNotFoundUntilPublished) {
  FlakyKVTransferSpecSource source(TestSpec(), /*failures=*/3,
                                   absl::NotFoundError("not published"));
  absl::StatusOr<KVTransferSpec> spec =
      KVCacheHostStoreNode::WaitForSpec(source, FastPollOptions());
  ASSERT_TRUE(spec.ok()) << spec.status();
  EXPECT_EQ(spec->block_array_bytes, TestSpec().block_array_bytes);
  EXPECT_EQ(source.calls(), 4);
}

TEST(WaitForSpecTest, RetriesUnavailable) {
  FlakyKVTransferSpecSource source(TestSpec(), /*failures=*/2,
                                   absl::UnavailableError("registry not up"));
  absl::StatusOr<KVTransferSpec> spec =
      KVCacheHostStoreNode::WaitForSpec(source, FastPollOptions());
  ASSERT_TRUE(spec.ok()) << spec.status();
  EXPECT_EQ(source.calls(), 3);
}

TEST(WaitForSpecTest, PropagatesFatalErrorWithoutRetry) {
  FlakyKVTransferSpecSource source(TestSpec(), /*failures=*/100,
                                   absl::InternalError("corrupt registry"));
  absl::StatusOr<KVTransferSpec> spec =
      KVCacheHostStoreNode::WaitForSpec(source, FastPollOptions());
  EXPECT_TRUE(absl::IsInternal(spec.status()));
  EXPECT_EQ(source.calls(), 1);
}

TEST(WaitForSpecTest, RejectsInvalidPublishedSpec) {
  FlakyKVTransferSpecSource source(KVTransferSpec{}, /*failures=*/0,
                                   absl::OkStatus());
  absl::StatusOr<KVTransferSpec> spec =
      KVCacheHostStoreNode::WaitForSpec(source, FastPollOptions());
  EXPECT_TRUE(absl::IsInvalidArgument(spec.status()));
}

TEST(WaitForSpecTest, TimesOut) {
  FlakyKVTransferSpecSource source(TestSpec(), /*failures=*/1000000,
                                   absl::NotFoundError("not published"));
  KVCacheHostStoreNode::Options options = FastPollOptions();
  options.spec_wait_timeout = absl::Milliseconds(30);
  absl::StatusOr<KVTransferSpec> spec =
      KVCacheHostStoreNode::WaitForSpec(source, options);
  EXPECT_TRUE(absl::IsDeadlineExceeded(spec.status()));
  EXPECT_GT(source.calls(), 1);
}

TEST(NumBlocksForBudgetTest, FloorsToWholeBlocks) {
  // One block of TestSpec() is 1 worker * 1 shard * (256 + 256) = 512 bytes.
  absl::StatusOr<size_t> blocks =
      KVCacheHostStoreNode::NumBlocksForBudget(4096, TestSpec());
  ASSERT_TRUE(blocks.ok()) << blocks.status();
  EXPECT_EQ(*blocks, 8u);
  blocks = KVCacheHostStoreNode::NumBlocksForBudget(4095, TestSpec());
  ASSERT_TRUE(blocks.ok()) << blocks.status();
  EXPECT_EQ(*blocks, 7u);
}

TEST(NumBlocksForBudgetTest, RejectsBudgetBelowOneBlock) {
  absl::StatusOr<size_t> blocks =
      KVCacheHostStoreNode::NumBlocksForBudget(511, TestSpec());
  EXPECT_TRUE(absl::IsInvalidArgument(blocks.status()));
  EXPECT_THAT(blocks.status().message(), HasSubstr("single block"));
}

TEST(NumBlocksForBudgetTest, ChargesEveryWorkersPool) {
  // Each worker's pool holds its own shards of every block, so doubling the
  // workers doubles what one block costs the budget.
  KVTransferSpec spec = TestSpec();
  spec.num_workers = 2;
  absl::StatusOr<size_t> blocks =
      KVCacheHostStoreNode::NumBlocksForBudget(4096, spec);
  ASSERT_TRUE(blocks.ok()) << blocks.status();
  EXPECT_EQ(*blocks, 4u);
  blocks = KVCacheHostStoreNode::NumBlocksForBudget(1023, spec);
  EXPECT_TRUE(absl::IsInvalidArgument(blocks.status()));
}

// Boot tests stand up real (loopback) gRPC servers; the env var isolates
// each node's controller/worker servers from the process-wide singletons so
// tests do not leak servers into each other.
class KVCacheHostStoreNodeBootTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("RAIDEN_DISABLE_SINGLETON_WORKER", "1", /*overwrite=*/1);
  }
  void TearDown() override { unsetenv("RAIDEN_DISABLE_SINGLETON_WORKER"); }

  KVCacheHostStoreNode::Options BootOptions() {
    KVCacheHostStoreNode::Options options = FastPollOptions();
    options.raiden_id =
        kv_cache::RaidenId{"store_node_test", "0", "kv_pool", 0};
    options.store_server_ip = "localhost";
    options.dram_budget_bytes = 8 * 512;  // 8 blocks of TestSpec().
    return options;
  }
};

TEST_F(KVCacheHostStoreNodeBootTest, RequiresStoreServerIp) {
  KVCacheHostStoreNode::Options options = BootOptions();
  options.store_server_ip.clear();
  StaticKVTransferSpecSource source(TestSpec());
  absl::StatusOr<std::unique_ptr<KVCacheHostStoreNode>> node =
      KVCacheHostStoreNode::Create(options, &source);
  EXPECT_TRUE(absl::IsInvalidArgument(node.status()));
  EXPECT_THAT(node.status().message(), HasSubstr("store_server_ip"));
}

TEST_F(KVCacheHostStoreNodeBootTest, RequiresKVTransferSpecSource) {
  absl::StatusOr<std::unique_ptr<KVCacheHostStoreNode>> node =
      KVCacheHostStoreNode::Create(BootOptions(), nullptr);
  EXPECT_TRUE(absl::IsInvalidArgument(node.status()));
}

TEST_F(KVCacheHostStoreNodeBootTest, RejectsHeterogeneousBlockArraysForNow) {
  // A hybrid-model spec: block arrays with differing strides. Valid as a
  // spec, but the CPU-only manager cannot express it yet, so boot must fail
  // loudly instead of allocating a wrong-shaped pool.
  StaticKVTransferSpecSource source(
      KVTransferSpec{/*block_array_bytes=*/{256, 512}, /*num_kv_shards=*/1,
                     /*num_workers=*/1});
  absl::StatusOr<std::unique_ptr<KVCacheHostStoreNode>> node =
      KVCacheHostStoreNode::Create(BootOptions(), &source);
  EXPECT_TRUE(absl::IsUnimplemented(node.status())) << node.status();
  EXPECT_THAT(node.status().message(), HasSubstr("uniform block arrays"));
}

TEST_F(KVCacheHostStoreNodeBootTest, BootsWithoutRegistryButServesNoPeers) {
  StaticKVTransferSpecSource source(TestSpec());
  absl::StatusOr<std::unique_ptr<KVCacheHostStoreNode>> node =
      KVCacheHostStoreNode::Create(BootOptions(), &source);
  ASSERT_TRUE(node.ok()) << node.status();

  // The budget became whole blocks of the received spec.
  EXPECT_EQ((*node)->num_host_blocks(), 8u);
  EXPECT_EQ((*node)->spec().block_array_bytes, TestSpec().block_array_bytes);

  // KVCacheStore's construction rules make the global registry decide
  // whether the peer-facing plane exists: no registry, no store server.
  EXPECT_EQ((*node)->store()->store_server(), nullptr);
  EXPECT_EQ((*node)->store_server_address(), "");

  // The controller is live and knows exactly our one worker, registered
  // under node_id 0 -- the pairing id the serving hosts' single worker
  // uses, mirrored rather than configured.
  auto workers = (*node)
                     ->store()
                     ->raiden_controller()
                     ->worker_registry()
                     ->GetRegisteredWorkers();
  ASSERT_EQ(workers.size(), 1u);
  EXPECT_EQ(workers[0].node_id, 0);

  // The manager opened its raw-transfer data endpoint.
  ASSERT_EQ((*node)->managers().size(), 1u);
  EXPECT_FALSE((*node)->managers()[0]->get_local_endpoints().empty());
}

TEST_F(KVCacheHostStoreNodeBootTest, MirrorsSpecWorkerTopology) {
  // Two serving-host transfer ranks: the node must field one worker per
  // rank with matching node_ids, or peers fail pairing with
  // FailedPrecondition on their first transfer.
  KVTransferSpec spec = TestSpec();
  spec.num_workers = 2;
  StaticKVTransferSpecSource source(spec);
  absl::StatusOr<std::unique_ptr<KVCacheHostStoreNode>> node =
      KVCacheHostStoreNode::Create(BootOptions(), &source);
  ASSERT_TRUE(node.ok()) << node.status();

  // The same budget now feeds two pools, so half the blocks fit; the block
  // id space stays one flat [0, num_host_blocks) shared by both workers.
  EXPECT_EQ((*node)->num_host_blocks(), 4u);
  ASSERT_EQ((*node)->managers().size(), 2u);

  auto workers = (*node)
                     ->store()
                     ->raiden_controller()
                     ->worker_registry()
                     ->GetRegisteredWorkers();
  ASSERT_EQ(workers.size(), 2u);
  std::vector<int64_t> node_ids = {workers[0].node_id, workers[1].node_id};
  EXPECT_THAT(node_ids, UnorderedElementsAre(0, 1));

  // Each rank owns its own data plane: distinct managers, distinct
  // registered transfer endpoints.
  ASSERT_FALSE(workers[0].raiden_transfer_endpoints.empty());
  ASSERT_FALSE(workers[1].raiden_transfer_endpoints.empty());
  EXPECT_NE(workers[0].raiden_transfer_endpoints[0].endpoint,
            workers[1].raiden_transfer_endpoints[0].endpoint);
}

TEST_F(KVCacheHostStoreNodeBootTest, BootsServesAndPublishesWithRegistry) {
  auto registry_server =
      kv_cache::global_registry::CreateTestGlobalRegistryServer();

  KVCacheHostStoreNode::Options options = BootOptions();
  options.global_registry_address = registry_server->server_address;
  options.kv_pool_group = "prefill_pool";

  StaticKVTransferSpecSource source(TestSpec());
  absl::StatusOr<std::unique_ptr<KVCacheHostStoreNode>> node =
      KVCacheHostStoreNode::Create(options, &source);
  ASSERT_TRUE(node.ok()) << node.status();

  // The peer-facing store server is bound and advertised.
  EXPECT_THAT((*node)->store_server_address(), HasSubstr("localhost:"));
  EXPECT_NE((*node)->store()->store_server(), nullptr);

  // And the node published itself: the registry resolves our RaidenId to
  // the advertised store address.
  auto& registry_client = *registry_server->client;
  auto store_info = registry_client.ResolveStore(options.raiden_id);
  ASSERT_TRUE(store_info.ok()) << store_info.status();
  EXPECT_EQ(store_info->store_server_address(),
            (*node)->store_server_address());

  // Placement can find it: a tier-0 store of the same group asks for targets
  // and gets this node back, which is Options' kv_pool_group and evict_tier
  // plumbed into the registration -- placement never crosses groups and only
  // offers tiers greater than the caller's.
  kv_cache::RaidenId serving_id{"serving_host", "0", "kv_cache", 0};
  ASSERT_TRUE(registry_client
                  .RegisterStore(serving_id, "localhost:1",
                                 /*controller_address=*/"",
                                 /*ttl=*/absl::ZeroDuration(),
                                 /*kv_pool_group=*/"prefill_pool",
                                 /*evict_tier=*/0)
                  .ok());
  auto targets = registry_client.GetPlacementTargets(serving_id);
  ASSERT_TRUE(targets.ok()) << targets.status();
  ASSERT_EQ(targets->size(), 1u);
  EXPECT_EQ((*targets)[0].store_server_address(),
            (*node)->store_server_address());
}

TEST_F(KVCacheHostStoreNodeBootTest, BootsFromGrsPublishedSpec) {
  auto service =
      std::make_unique<kv_cache::global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(service.get());
  std::unique_ptr<grpc::Server> registry_server = builder.BuildAndStart();
  ASSERT_NE(registry_server, nullptr);
  const std::string registry_address = absl::StrCat("localhost:", port);

  // A serving host publishes the spec; the node then boots entirely from
  // the registry, no static spec anywhere.
  kv_cache::global_registry::GlobalRegistryClient registry_client(
      grpc::CreateChannel(registry_address,
                          grpc::InsecureChannelCredentials()));
  kv_cache::global_registry::KVTransferSpec published;
  published.add_block_arrays()->set_block_bytes(256);
  published.add_block_arrays()->set_block_bytes(256);
  published.set_num_kv_shards(1);
  published.set_num_workers(1);
  ASSERT_TRUE(
      registry_client.RegisterKVTransferSpec(published, "prefill_pool").ok());

  KVCacheHostStoreNode::Options options = BootOptions();
  options.global_registry_address = registry_address;
  GrsKVTransferSpecSource source(registry_address, "prefill_pool");
  absl::StatusOr<std::unique_ptr<KVCacheHostStoreNode>> node =
      KVCacheHostStoreNode::Create(options, &source);
  ASSERT_TRUE(node.ok()) << node.status();
  EXPECT_EQ((*node)->spec().block_array_bytes,
            (std::vector<uint64_t>{256, 256}));
  EXPECT_EQ((*node)->num_host_blocks(), 8u);
  EXPECT_THAT((*node)->store_server_address(), HasSubstr("localhost:"));
}

TEST_F(KVCacheHostStoreNodeBootTest, WaitsOutLateSpecThenBoots) {
  FlakyKVTransferSpecSource source(TestSpec(), /*failures=*/2,
                                   absl::NotFoundError("not published"));
  absl::StatusOr<std::unique_ptr<KVCacheHostStoreNode>> node =
      KVCacheHostStoreNode::Create(BootOptions(), &source);
  ASSERT_TRUE(node.ok()) << node.status();
  EXPECT_EQ(source.calls(), 3);
  EXPECT_EQ((*node)->num_host_blocks(), 8u);
}

}  // namespace
}  // namespace store_node
}  // namespace tpu_raiden
