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

#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/support/status.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/kv_cache/global_registry/global_registry.grpc.pb.h"
#include "tpu_sync/kv_cache/global_registry/global_registry.pb.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_client.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_server.h"
#include "tpu_sync/kv_cache/global_registry/test_util.h"
#include "tpu_sync/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {
namespace global_registry {

using namespace ::tpu_raiden::kv_cache::global_registry;  // NOLINT

namespace {

class GlobalRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_server_ = CreateTestGlobalRegistryServer(
        /*default_ttl=*/absl::Seconds(2),
        /*cleanup_interval=*/absl::Seconds(1));
    service_ = test_server_->service.get();
    channel_ = test_server_->channel;
    client_ = test_server_->client.get();
  }

  void TearDown() override { test_server_.reset(); }

  // Calls PullOwned via the raw generated stub and collects all streamed
  // entries. Reports the number of streamed messages via `num_messages` and
  // the final stream status via `status` when provided.
  static std::vector<PullOwnedEntry> PullOwnedOn(
      const std::shared_ptr<grpc::Channel>& channel, const RaidenId& raiden_id,
      grpc::Status* status = nullptr, int* num_messages = nullptr) {
    auto stub = GlobalRegistryService::NewStub(channel);
    grpc::ClientContext context;
    PullOwnedRequest request;
    auto* id_proto = request.mutable_raiden_id();
    id_proto->set_job_name(raiden_id.job_name);
    id_proto->set_job_replica_id(raiden_id.job_replica_id);
    id_proto->set_data_name(raiden_id.data_name);
    id_proto->set_data_replica_idx(raiden_id.data_replica_idx);

    auto reader = stub->PullOwned(&context, request);
    std::vector<PullOwnedEntry> entries;
    PullOwnedResponse response;
    int messages = 0;
    while (reader->Read(&response)) {
      ++messages;
      for (const auto& entry : response.entries()) {
        entries.push_back(entry);
      }
    }
    grpc::Status finish_status = reader->Finish();
    if (status != nullptr) *status = finish_status;
    if (num_messages != nullptr) *num_messages = messages;
    return entries;
  }

  std::vector<PullOwnedEntry> PullOwned(const RaidenId& raiden_id,
                                        grpc::Status* status = nullptr,
                                        int* num_messages = nullptr) {
    return PullOwnedOn(channel_, raiden_id, status, num_messages);
  }

  std::unique_ptr<TestGlobalRegistryServer> test_server_;
  GlobalRegistryServiceImpl* service_ = nullptr;
  std::shared_ptr<grpc::Channel> channel_;
  GlobalRegistryClient* client_ = nullptr;
};

TEST_F(GlobalRegistryTest, BasicRegisterAndLookup) {
  std::string hash = "hash1";
  RaidenId host = {"job1", "replica1", "data1", 0};
  int32_t block = 42;

  auto status = client_->Register({{hash, host, block}});
  EXPECT_TRUE(status.ok()) << status.ToString();

  auto lookup_res = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res.ok()) << lookup_res.status().ToString();
  ASSERT_EQ(lookup_res->size(), 1);
  const auto& meta = (*lookup_res)[0];
  EXPECT_EQ(meta.raiden_id().job_name(), host.job_name);
  EXPECT_EQ(meta.raiden_id().job_replica_id(), host.job_replica_id);
  EXPECT_EQ(meta.raiden_id().data_name(), host.data_name);
  EXPECT_EQ(meta.raiden_id().data_replica_idx(), host.data_replica_idx);
  EXPECT_EQ(meta.block_id(), block);
}

TEST_F(GlobalRegistryTest, BinaryHashRegisterAndLookup) {
  std::string hash("\x93\xff\x00\xa1\xfe\x80zz\x01\xc3\xbf\xed", 12);
  RaidenId host = {"job1", "replica1", "data1", 0};

  auto status = client_->Register({{hash, host, 7}});
  EXPECT_TRUE(status.ok()) << status.ToString();

  auto lookup_res = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res.ok()) << lookup_res.status().ToString();
  ASSERT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].raiden_id().job_name(), host.job_name);
  EXPECT_EQ((*lookup_res)[0].block_id(), 7);

  auto owned = PullOwned(host);
  ASSERT_EQ(owned.size(), 1);
  EXPECT_EQ(owned[0].prefix_hash(), hash);

  auto unreg = client_->Unregister({hash}, host);
  EXPECT_TRUE(unreg.ok()) << unreg.ToString();
  auto after = client_->Lookup({hash});
  ASSERT_TRUE(after.ok());
  EXPECT_TRUE(after->empty());
}

TEST_F(GlobalRegistryTest, BinaryHashUnregisterMismatchReportsHexError) {
  std::string hash("\xde\xad\xbe\xef\xff\x00\xc3\x28", 8);
  RaidenId owner = {"job1", "replica1", "data1", 0};
  RaidenId other = {"job2", "replica2", "data2", 1};

  ASSERT_TRUE(client_->Register({{hash, owner, 3}}).ok());

  absl::Status unreg = client_->Unregister({hash}, other);
  EXPECT_EQ(unreg.code(), absl::StatusCode::kFailedPrecondition)
      << unreg.ToString();
  EXPECT_NE(std::string(unreg.message()).find("deadbeefff00c328"),
            std::string::npos)
      << unreg.message();
}

TEST_F(GlobalRegistryTest, MultiRegistrationAndRoundRobinLookup) {
  std::string hash = "hash1";
  RaidenId host1 = {"job1", "replica1", "data1", 0};
  int32_t block1 = 42;
  RaidenId host2 = {"job1", "replica2", "data1", 1};
  int32_t block2 = 43;

  EXPECT_TRUE(
      client_->Register({{hash, host1, block1}, {hash, host2, block2}}).ok());

  // First lookup should return host1 or host2 depending on internal order.
  // The server implementation iterates over insertion order, so we expect host1
  // then host2.
  auto lookup_res1 = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res1.ok()) << lookup_res1.status().ToString();
  ASSERT_EQ(lookup_res1->size(), 1);
  EXPECT_EQ((*lookup_res1)[0].raiden_id().job_replica_id(),
            host1.job_replica_id);
  EXPECT_EQ((*lookup_res1)[0].block_id(), block1);

  // Second lookup should round-robin to host2.
  auto lookup_res2 = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res2.ok()) << lookup_res2.status().ToString();
  ASSERT_EQ(lookup_res2->size(), 1);
  EXPECT_EQ((*lookup_res2)[0].raiden_id().job_replica_id(),
            host2.job_replica_id);
  EXPECT_EQ((*lookup_res2)[0].block_id(), block2);

  // Third lookup should wrap around to host1.
  auto lookup_res3 = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res3.ok()) << lookup_res3.status().ToString();
  ASSERT_EQ(lookup_res3->size(), 1);
  EXPECT_EQ((*lookup_res3)[0].raiden_id().job_replica_id(),
            host1.job_replica_id);
}

TEST_F(GlobalRegistryTest, OverwriteRegistrationSameHost) {
  std::string hash = "hash1";
  RaidenId host = {"job1", "replica1", "data1", 0};
  int32_t block1 = 42;
  int32_t block2 = 43;

  EXPECT_TRUE(client_->Register({{hash, host, block1}}).ok());
  EXPECT_TRUE(client_->Register({{hash, host, block2}}).ok());

  auto lookup_res = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res.ok()) << lookup_res.status().ToString();
  ASSERT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].block_id(), block2);
}

TEST_F(GlobalRegistryTest, MultiLookupTerminatesOnFirstMiss) {
  EXPECT_TRUE(client_
                  ->Register({{"h1", {"j1", "r1", "d1", 0}, 42},
                              {"h2", {"j1", "r2", "d1", 0}, 43}})
                  .ok());

  // Sequential hits for "h1" and "h2"
  auto res1 = client_->Lookup({"h1", "h2"});
  ASSERT_TRUE(res1.ok());
  EXPECT_EQ(res1->size(), 2);

  // Miss on "h3" stops sequential lookup, "h2" is omitted
  auto res2 = client_->Lookup({"h1", "h3", "h2"});
  ASSERT_TRUE(res2.ok());
  EXPECT_EQ(res2->size(), 1);
  EXPECT_EQ((*res2)[0].block_id(), 42);
}

TEST_F(GlobalRegistryTest, UnregisterSuccess) {
  std::string hash1 = "hash1";
  std::string hash2 = "hash2";
  RaidenId host = {"job1", "replica1", "data1", 0};

  EXPECT_TRUE(client_->Register({{hash1, host, 42}, {hash2, host, 43}}).ok());

  auto status = client_->Unregister({hash1}, host);
  EXPECT_TRUE(status.ok()) << status.ToString();

  auto res = client_->Lookup({hash1, hash2});
  ASSERT_TRUE(res.ok());
  EXPECT_EQ(res->size(), 0);  // Stops at first miss (hash1 is gone)

  auto res2 = client_->Lookup({hash2});
  ASSERT_TRUE(res2.ok());
  EXPECT_EQ(res2->size(), 1);
  EXPECT_EQ((*res2)[0].block_id(), 43);
}

TEST_F(GlobalRegistryTest, ExpirationTtlFilter) {
  std::string hash = "hash1";
  RaidenId host = {"job1", "replica1", "data1", 0};

  auto status = client_->Register({{hash, host, 42, absl::Seconds(1)}});
  EXPECT_TRUE(status.ok());

  absl::SleepFor(absl::Milliseconds(1500));

  auto res = client_->Lookup({hash});
  ASSERT_TRUE(res.ok());
  EXPECT_EQ(res->size(), 0);
}

TEST_F(GlobalRegistryTest, PullOwnedReturnsOnlyOwnedEntries) {
  RaidenId owner_a = {"jobA", "r1", "d1", 0};
  RaidenId owner_b = {"jobB", "r1", "d1", 0};
  ASSERT_TRUE(client_
                  ->Register({{"h1", owner_a, 1, absl::Seconds(300)},
                              {"h2", owner_a, 2, absl::Seconds(300)},
                              {"h2", owner_b, 20, absl::Seconds(300)},
                              {"h3", owner_b, 30, absl::Seconds(300)}})
                  .ok());

  grpc::Status status;
  auto entries = PullOwned(owner_a, &status);
  EXPECT_TRUE(status.ok()) << status.error_message();
  ASSERT_EQ(entries.size(), 2);
  std::map<std::string, int32_t> blocks;
  for (const auto& entry : entries) {
    blocks[entry.prefix_hash()] = entry.block_id();
  }
  // h2 is registered by both owners; owner A must see its own block id.
  EXPECT_EQ(blocks.at("h1"), 1);
  EXPECT_EQ(blocks.at("h2"), 2);

  auto entries_b = PullOwned(owner_b);
  ASSERT_EQ(entries_b.size(), 2);
}

TEST_F(GlobalRegistryTest, PullOwnedEmptyForUnknownOwner) {
  grpc::Status status;
  int num_messages = 0;
  auto entries =
      PullOwned({"unknown", "r1", "d1", 0}, &status, &num_messages);
  EXPECT_TRUE(status.ok()) << status.error_message();
  EXPECT_TRUE(entries.empty());
  EXPECT_EQ(num_messages, 0);
}

TEST_F(GlobalRegistryTest, PullOwnedRequiresRaidenId) {
  grpc::Status status;
  auto entries = PullOwned(RaidenId{}, &status);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_TRUE(entries.empty());
}

TEST_F(GlobalRegistryTest, PullOwnedReflectsUnregisterAndIndexShrinks) {
  RaidenId owner = {"jobU", "r1", "d1", 0};
  ASSERT_TRUE(client_
                  ->Register({{"h1", owner, 1, absl::Seconds(300)},
                              {"h2", owner, 2, absl::Seconds(300)}})
                  .ok());
  EXPECT_EQ(service_->GetOwnerIndexSizeForTest(owner), 2);

  ASSERT_TRUE(client_->Unregister({"h1"}, owner).ok());
  EXPECT_EQ(service_->GetOwnerIndexSizeForTest(owner), 1);

  auto entries = PullOwned(owner);
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].prefix_hash(), "h2");

  ASSERT_TRUE(client_->Unregister({"h2"}, owner).ok());
  EXPECT_EQ(service_->GetOwnerIndexSizeForTest(owner), 0);
  EXPECT_TRUE(PullOwned(owner).empty());
}

TEST_F(GlobalRegistryTest, PullOwnedOmitsExpiredEntries) {
  RaidenId owner = {"jobE", "r1", "d1", 0};
  ASSERT_TRUE(client_
                  ->Register({{"h1", owner, 1, absl::Seconds(1)},
                              {"h2", owner, 2, absl::Seconds(300)}})
                  .ok());

  absl::SleepFor(absl::Milliseconds(1500));

  auto entries = PullOwned(owner);
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].prefix_hash(), "h2");
}

TEST_F(GlobalRegistryTest, CleanupShrinksOwnerIndex) {
  RaidenId owner = {"jobC", "r1", "d1", 0};
  ASSERT_TRUE(client_->Register({{"h1", owner, 1, absl::Seconds(1)}}).ok());
  EXPECT_EQ(service_->GetOwnerIndexSizeForTest(owner), 1);

  absl::SleepFor(absl::Milliseconds(1500));
  service_->CleanupExpiredEntries();

  EXPECT_EQ(service_->GetOwnerIndexSizeForTest(owner), 0);
  EXPECT_TRUE(PullOwned(owner).empty());
}

TEST_F(GlobalRegistryTest, PullOwnedUpsertKeepsSingleEntryWithLatestBlock) {
  RaidenId owner = {"jobO", "r1", "d1", 0};
  ASSERT_TRUE(client_->Register({{"h1", owner, 42, absl::Seconds(300)}}).ok());
  ASSERT_TRUE(client_->Register({{"h1", owner, 43, absl::Seconds(300)}}).ok());

  EXPECT_EQ(service_->GetOwnerIndexSizeForTest(owner), 1);
  auto entries = PullOwned(owner);
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].block_id(), 43);
}

TEST_F(GlobalRegistryTest, PullOwnedStreamsInServerConfiguredBatches) {
  auto test_server = CreateTestGlobalRegistryServer(
      /*default_ttl=*/absl::Seconds(300),
      /*cleanup_interval=*/absl::ZeroDuration(),
      /*pull_owned_batch_size=*/2);

  RaidenId owner = {"jobB", "r1", "d1", 0};
  ASSERT_TRUE(test_server->client
                  ->Register({{"h1", owner, 1},
                              {"h2", owner, 2},
                              {"h3", owner, 3},
                              {"h4", owner, 4},
                              {"h5", owner, 5}})
                  .ok());

  grpc::Status status;
  int num_messages = 0;
  auto entries =
      PullOwnedOn(test_server->channel, owner, &status, &num_messages);
  EXPECT_TRUE(status.ok()) << status.error_message();
  EXPECT_EQ(entries.size(), 5);
  EXPECT_EQ(num_messages, 3);  // ceil(5 entries / batch size 2)
}

TEST_F(GlobalRegistryTest, PullOwnedRemainingTtlForExpiringEntry) {
  RaidenId owner = {"jobT", "r1", "d1", 0};
  ASSERT_TRUE(client_->Register({{"h1", owner, 1, absl::Seconds(300)}}).ok());

  auto entries = PullOwned(owner);
  ASSERT_EQ(entries.size(), 1);
  // Wall-clock time elapses between Register and PullOwned, so the lower
  // bound is deliberately generous: it only needs to prove the value reflects
  // the registered TTL rather than the 0 infinite marker or the >= 1 clamp.
  // The upper bound is timing-safe (remaining TTL only decreases).
  EXPECT_GT(entries[0].remaining_ttl_seconds(), 200);
  EXPECT_LE(entries[0].remaining_ttl_seconds(), 300);
}

TEST_F(GlobalRegistryTest, ClientPullOwnedReturnsEntries) {
  RaidenId owner = {"jobW", "r1", "d1", 0};
  ASSERT_TRUE(client_
                  ->Register({{"h1", owner, 7, absl::Seconds(300)},
                              {"h2", owner, 8, absl::Seconds(300)}})
                  .ok());

  auto pulled_or = client_->PullOwned(owner);
  ASSERT_TRUE(pulled_or.ok()) << pulled_or.status().ToString();
  ASSERT_EQ(pulled_or->size(), 2);
  std::map<std::string, GlobalRegistryClient::PulledEntry> by_hash;
  for (const auto& entry : *pulled_or) {
    by_hash[entry.prefix_hash] = entry;
  }
  EXPECT_EQ(by_hash.at("h1").block_id, 7);
  EXPECT_EQ(by_hash.at("h2").block_id, 8);
  EXPECT_GT(by_hash.at("h1").remaining_ttl_seconds, 0);
}

TEST_F(GlobalRegistryTest, ClientPullOwnedEmptyForUnknownOwner) {
  auto pulled_or = client_->PullOwned({"nobody", "r1", "d1", 0});
  ASSERT_TRUE(pulled_or.ok()) << pulled_or.status().ToString();
  EXPECT_TRUE(pulled_or->empty());
}

TEST_F(GlobalRegistryTest, ClientPullOwnedRejectsEmptyRaidenId) {
  auto pulled_or = client_->PullOwned(RaidenId{});
  EXPECT_FALSE(pulled_or.ok());
}

TEST_F(GlobalRegistryTest, PullOwnedRemainingTtlZeroForInfiniteTtl) {
  auto test_server = CreateTestGlobalRegistryServer(
      /*default_ttl=*/absl::InfiniteDuration(),
      /*cleanup_interval=*/absl::ZeroDuration());

  RaidenId owner = {"jobI", "r1", "d1", 0};
  // No explicit TTL: the server's default (infinite) applies.
  ASSERT_TRUE(test_server->client->Register({{"h1", owner, 1}}).ok());

  auto entries = PullOwnedOn(test_server->channel, owner);
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].remaining_ttl_seconds(), 0);
}

// ---------------------------------------------------------------------------
// Store registrations: RaidenId -> where to reach that store's
// KVCacheStoreService. Independent of the block registry above.
// ---------------------------------------------------------------------------

TEST_F(GlobalRegistryTest, RegisterStoreAndResolveRoundTrip) {
  RaidenId id = {"jobS", "r0", "dataS", 3};

  ASSERT_TRUE(client_
                  ->RegisterStore(id, "10.0.0.7:41337",
                                  /*controller_address=*/"10.0.0.7:9000")
                  .ok());

  auto resolved = client_->ResolveStore(id);
  ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
  EXPECT_EQ(resolved->store_server_address(), "10.0.0.7:41337");
  EXPECT_EQ(resolved->controller_address(), "10.0.0.7:9000");
  EXPECT_EQ(resolved->raiden_id().job_name(), id.job_name);
  EXPECT_EQ(resolved->raiden_id().job_replica_id(), id.job_replica_id);
  EXPECT_EQ(resolved->raiden_id().data_name(), id.data_name);
  EXPECT_EQ(resolved->raiden_id().data_replica_idx(), id.data_replica_idx);
}

// A restarted store comes back on a different ephemeral port; re-registering
// must replace the dead coordinates rather than accumulate a second entry.
TEST_F(GlobalRegistryTest, RegisterStoreReplacesPreviousAddress) {
  RaidenId id = {"jobS", "r0", "dataS", 0};

  ASSERT_TRUE(client_->RegisterStore(id, "10.0.0.7:1111").ok());
  ASSERT_TRUE(client_->RegisterStore(id, "10.0.0.7:2222").ok());

  auto resolved = client_->ResolveStore(id);
  ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
  EXPECT_EQ(resolved->store_server_address(), "10.0.0.7:2222");
}

TEST_F(GlobalRegistryTest, ResolveStoreMissIsNotFound) {
  auto resolved = client_->ResolveStore({"nobody", "r0", "d0", 0});
  EXPECT_TRUE(absl::IsNotFound(resolved.status())) << resolved.status();
}

TEST_F(GlobalRegistryTest, RegisterStoreRejectsEmptyAddress) {
  RaidenId id = {"jobS", "r0", "dataS", 0};
  EXPECT_FALSE(client_->RegisterStore(id, "").ok());
  EXPECT_TRUE(absl::IsNotFound(client_->ResolveStore(id).status()));
}

TEST_F(GlobalRegistryTest, RegisterStoreRejectsEmptyRaidenId) {
  EXPECT_FALSE(client_->RegisterStore({"", "", "", 0}, "10.0.0.7:1").ok());
}

TEST_F(GlobalRegistryTest, ResolveStoreRejectsEmptyRaidenId) {
  EXPECT_FALSE(client_->ResolveStore({"", "", "", 0}).ok());
}

TEST_F(GlobalRegistryTest, UnregisterStoreRemovesRegistration) {
  RaidenId id = {"jobS", "r0", "dataS", 0};
  ASSERT_TRUE(client_->RegisterStore(id, "10.0.0.7:1111").ok());
  ASSERT_TRUE(client_->ResolveStore(id).ok());

  ASSERT_TRUE(client_->UnregisterStore(id).ok());
  EXPECT_TRUE(absl::IsNotFound(client_->ResolveStore(id).status()));
}

// Teardown must not fail because the entry was already gone -- a store that
// never registered still calls this from its destructor.
TEST_F(GlobalRegistryTest, UnregisterStoreIsIdempotent) {
  RaidenId id = {"jobS", "r0", "dataS", 0};
  EXPECT_TRUE(client_->UnregisterStore(id).ok());
  ASSERT_TRUE(client_->RegisterStore(id, "10.0.0.7:1111").ok());
  EXPECT_TRUE(client_->UnregisterStore(id).ok());
  EXPECT_TRUE(client_->UnregisterStore(id).ok());
}

// Deliberate divergence from block entries: a store registration with no
// explicit TTL never expires, so a long-lived store does not silently become
// unresolvable. The fixture's default_ttl is 2s, which a block entry would
// have hit by now.
TEST_F(GlobalRegistryTest, StoreRegistrationDefaultsToNoExpiry) {
  RaidenId id = {"jobS", "r0", "dataS", 0};
  ASSERT_TRUE(client_->RegisterStore(id, "10.0.0.7:1111").ok());

  absl::SleepFor(absl::Milliseconds(2500));
  service_->CleanupExpiredEntries();

  auto resolved = client_->ResolveStore(id);
  ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
  EXPECT_EQ(resolved->store_server_address(), "10.0.0.7:1111");
}

// An explicit TTL is honoured, and an expired record reads as a miss before
// the cleanup thread gets to it (the lazy filter in ResolveStore).
TEST_F(GlobalRegistryTest, StoreRegistrationExpiresWithExplicitTtl) {
  RaidenId id = {"jobS", "r0", "dataS", 0};
  ASSERT_TRUE(client_
                  ->RegisterStore(id, "10.0.0.7:1111",
                                  /*controller_address=*/"",
                                  /*ttl=*/absl::Seconds(1))
                  .ok());
  ASSERT_TRUE(client_->ResolveStore(id).ok());

  absl::SleepFor(absl::Milliseconds(1500));

  EXPECT_TRUE(absl::IsNotFound(client_->ResolveStore(id).status()));
}

TEST_F(GlobalRegistryTest, HeartbeatRefreshesStoreTtl) {
  RaidenId id = {"jobS", "r0", "dataS", 0};
  ASSERT_TRUE(client_
                  ->RegisterStore(id, "10.0.0.7:1111",
                                  /*controller_address=*/"",
                                  /*ttl=*/absl::Seconds(2))
                  .ok());

  // Past the halfway point of the original TTL; the heartbeat must push the
  // expiry out so the registration survives beyond the original deadline.
  absl::SleepFor(absl::Milliseconds(1500));
  StoreStatus status;
  status.set_free_blocks(7);
  ASSERT_TRUE(client_->Heartbeat(id, status).ok());

  absl::SleepFor(absl::Milliseconds(1500));  // 3.0s > the original 2s TTL.
  EXPECT_TRUE(client_->ResolveStore(id).ok());

  // Silence past the refreshed deadline: the registration expires.
  absl::SleepFor(absl::Milliseconds(2500));
  EXPECT_TRUE(absl::IsNotFound(client_->ResolveStore(id).status()));
}

// A heartbeat carries no coordinates, so it must not resurrect or create a
// registration -- the store has to RegisterStore again.
TEST_F(GlobalRegistryTest, HeartbeatWithoutRegistrationIsNotFound) {
  RaidenId id = {"jobS", "r0", "dataS", 0};
  StoreStatus status;
  status.set_free_blocks(1);
  EXPECT_TRUE(absl::IsNotFound(client_->Heartbeat(id, status)));
  EXPECT_TRUE(absl::IsNotFound(client_->ResolveStore(id).status()));
}

TEST_F(GlobalRegistryTest, ExpiredStorePurgesItsBlockEntries) {
  RaidenId dead = {"jobDead", "r0", "dataS", 0};
  RaidenId storeless = {"jobStoreless", "r0", "dataS", 0};
  ASSERT_TRUE(client_
                  ->RegisterStore(dead, "10.0.0.8:1111",
                                  /*controller_address=*/"",
                                  /*ttl=*/absl::Seconds(1))
                  .ok());
  ASSERT_TRUE(client_
                  ->Register({{"hd1", dead, 1, absl::Seconds(300)},
                              {"hshared", dead, 2, absl::Seconds(300)},
                              {"hshared", storeless, 20, absl::Seconds(300)},
                              {"hs1", storeless, 3, absl::Seconds(300)}})
                  .ok());

  absl::SleepFor(absl::Milliseconds(1500));
  service_->CleanupExpiredEntries();

  // The expired store's block entries left with its registration, including
  // its copy of the shared hash...
  EXPECT_EQ(service_->GetOwnerIndexSizeForTest(dead), 0);
  auto res = client_->Lookup({"hd1"});
  ASSERT_TRUE(res.ok());
  EXPECT_EQ(res->size(), 0);
  res = client_->Lookup({"hshared"});
  ASSERT_TRUE(res.ok());
  ASSERT_EQ(res->size(), 1);
  EXPECT_EQ((*res)[0].block_id(), 20);
  // ...while an owner that never registered a store keeps its entries.
  EXPECT_EQ(service_->GetOwnerIndexSizeForTest(storeless), 2);
  res = client_->Lookup({"hs1"});
  ASSERT_TRUE(res.ok());
  EXPECT_EQ(res->size(), 1);
}

TEST_F(GlobalRegistryTest, HeartbeatKeepsStoreBlocksThroughCleanup) {
  RaidenId id = {"jobHb", "r0", "dataS", 0};
  ASSERT_TRUE(client_
                  ->RegisterStore(id, "10.0.0.9:1111",
                                  /*controller_address=*/"",
                                  /*ttl=*/absl::Seconds(2))
                  .ok());
  ASSERT_TRUE(client_->Register({{"hb1", id, 1, absl::Seconds(300)}}).ok());

  // Same timing as HeartbeatRefreshesStoreTtl: refresh past the halfway
  // point, then check beyond the original deadline.
  absl::SleepFor(absl::Milliseconds(1500));
  StoreStatus status;
  status.set_free_blocks(5);
  ASSERT_TRUE(client_->Heartbeat(id, status).ok());
  absl::SleepFor(absl::Milliseconds(1500));
  service_->CleanupExpiredEntries();

  auto res = client_->Lookup({"hb1"});
  ASSERT_TRUE(res.ok());
  EXPECT_EQ(res->size(), 1);
}

TEST_F(GlobalRegistryTest, UnregisterStorePurgesItsBlockEntries) {
  RaidenId id = {"jobBye", "r0", "dataS", 0};
  ASSERT_TRUE(client_
                  ->RegisterStore(id, "10.0.0.10:1111",
                                  /*controller_address=*/"",
                                  /*ttl=*/absl::ZeroDuration())
                  .ok());
  ASSERT_TRUE(client_
                  ->Register({{"hu1", id, 1, absl::Seconds(300)},
                              {"hu2", id, 2, absl::Seconds(300)}})
                  .ok());
  EXPECT_EQ(service_->GetOwnerIndexSizeForTest(id), 2);

  ASSERT_TRUE(client_->UnregisterStore(id).ok());

  EXPECT_EQ(service_->GetOwnerIndexSizeForTest(id), 0);
  auto res = client_->Lookup({"hu1"});
  ASSERT_TRUE(res.ok());
  EXPECT_EQ(res->size(), 0);
}

TEST_F(GlobalRegistryTest, PlacementTargetsComeFromNearestGreaterTier) {
  auto register_store = [&](absl::string_view job, absl::string_view group,
                            int32_t tier) {
    RaidenId id = {std::string(job), "r0", "dataS", 0};
    EXPECT_TRUE(client_
                    ->RegisterStore(id, absl::StrCat(job, ":1111"),
                                    /*controller_address=*/"",
                                    /*ttl=*/absl::ZeroDuration(), group, tier)
                    .ok());
    return id;
  };
  RaidenId peer_t0 = register_store("peer_t0", "groupA", 0);
  RaidenId node_t1 = register_store("node_t1", "groupA", 1);
  RaidenId node_t2 = register_store("node_t2", "groupA", 2);
  register_store("other_group_t1", "groupB", 1);  // Wrong group: skipped.

  // From tier 0 the nearest greater tier is 1: tier 2 and the wrong group
  // are skipped, and so is the caller's own tier.
  auto targets = client_->GetPlacementTargets(peer_t0, /*max_targets=*/8);
  ASSERT_TRUE(targets.ok()) << targets.status().ToString();
  ASSERT_EQ(targets->size(), 1);
  EXPECT_EQ((*targets)[0].raiden_id().job_name(), "node_t1");
  EXPECT_EQ((*targets)[0].store_server_address(), "node_t1:1111");

  // From tier 1 the nearest greater tier is 2.
  targets = client_->GetPlacementTargets(node_t1, /*max_targets=*/8);
  ASSERT_TRUE(targets.ok());
  ASSERT_EQ(targets->size(), 1);
  EXPECT_EQ((*targets)[0].raiden_id().job_name(), "node_t2");

  // The bottom tier has no greater tier: empty is the answer, not an error.
  targets = client_->GetPlacementTargets(node_t2, /*max_targets=*/8);
  ASSERT_TRUE(targets.ok());
  EXPECT_TRUE(targets->empty());
}

TEST_F(GlobalRegistryTest, PlacementTargetsFavorReportedFreeCapacity) {
  RaidenId caller = {"pressured_t0", "r0", "dataS", 0};
  ASSERT_TRUE(client_
                  ->RegisterStore(caller, "pressured_t0:1111",
                                  /*controller_address=*/"",
                                  /*ttl=*/absl::ZeroDuration(), "groupA",
                                  /*evict_tier=*/0)
                  .ok());
  auto register_node = [&](absl::string_view job, int64_t free_blocks) {
    RaidenId id = {std::string(job), "r0", "dataS", 0};
    ASSERT_TRUE(client_
                    ->RegisterStore(id, absl::StrCat(job, ":1111"),
                                    /*controller_address=*/"",
                                    /*ttl=*/absl::ZeroDuration(), "groupA",
                                    /*evict_tier=*/1)
                    .ok());
    StoreStatus status;
    status.set_free_blocks(free_blocks);
    ASSERT_TRUE(client_->Heartbeat(id, status).ok());
  };
  register_node("node_full", 0);
  register_node("node_free", 50);

  // A store with no reported free space is never preferred over one with
  // some: it can only appear after every store with free space.
  auto targets = client_->GetPlacementTargets(caller, /*max_targets=*/1);
  ASSERT_TRUE(targets.ok());
  ASSERT_EQ(targets->size(), 1);
  EXPECT_EQ((*targets)[0].raiden_id().job_name(), "node_free");

  // With room for both, the full store still shows up -- as the last resort.
  targets = client_->GetPlacementTargets(caller, /*max_targets=*/8);
  ASSERT_TRUE(targets.ok());
  ASSERT_EQ(targets->size(), 2);
  EXPECT_EQ((*targets)[0].raiden_id().job_name(), "node_free");
  EXPECT_EQ((*targets)[1].raiden_id().job_name(), "node_full");
}

TEST_F(GlobalRegistryTest, PlacementTargetsHonorCapAndSkipExpired) {
  RaidenId caller = {"pressured_t0", "r0", "dataS", 0};
  ASSERT_TRUE(client_
                  ->RegisterStore(caller, "pressured_t0:1111",
                                  /*controller_address=*/"",
                                  /*ttl=*/absl::ZeroDuration(), "groupA",
                                  /*evict_tier=*/0)
                  .ok());
  auto register_node = [&](absl::string_view job, absl::Duration ttl) {
    RaidenId id = {std::string(job), "r0", "dataS", 0};
    ASSERT_TRUE(client_
                    ->RegisterStore(id, absl::StrCat(job, ":1111"),
                                    /*controller_address=*/"",
                                    ttl, "groupA", /*evict_tier=*/1)
                    .ok());
  };
  register_node("node_a", absl::ZeroDuration());
  register_node("node_b", absl::ZeroDuration());
  register_node("node_c", absl::ZeroDuration());
  register_node("node_d", absl::ZeroDuration());
  register_node("node_dead", absl::Seconds(1));

  absl::SleepFor(absl::Milliseconds(1500));

  auto targets = client_->GetPlacementTargets(caller, /*max_targets=*/8);
  ASSERT_TRUE(targets.ok());
  EXPECT_EQ(targets->size(), 4);  // node_dead expired without heartbeats.

  targets = client_->GetPlacementTargets(caller, /*max_targets=*/1);
  ASSERT_TRUE(targets.ok());
  EXPECT_EQ(targets->size(), 1);

  // An unset max_targets falls back to the server default.
  targets = client_->GetPlacementTargets(caller);
  ASSERT_TRUE(targets.ok());
  EXPECT_EQ(targets->size(),
            GlobalRegistryServiceImpl::kDefaultMaxPlacementTargets);
}

TEST_F(GlobalRegistryTest, PlacementTargetsRequireALiveRegisteredCaller) {
  EXPECT_TRUE(absl::IsInvalidArgument(
      client_->GetPlacementTargets({"", "", "", 0}).status()));

  // Placement is answered from the caller's own registration, so an
  // unregistered caller gets NotFound -- its cue to RegisterStore again.
  RaidenId unregistered = {"ghost", "r0", "dataS", 0};
  EXPECT_TRUE(
      absl::IsNotFound(client_->GetPlacementTargets(unregistered).status()));

  // A store registered without a kv_pool_group opted out of placement.
  RaidenId groupless = {"loner", "r0", "dataS", 0};
  ASSERT_TRUE(client_->RegisterStore(groupless, "loner:1111").ok());
  EXPECT_TRUE(absl::IsFailedPrecondition(
      client_->GetPlacementTargets(groupless).status()));
}

// The two tables are independent: blocks can be registered by a store that
// never published an address (lookup hits, resolve misses), which is exactly
// the state store registration exists to make impossible for stores that do
// publish one.
TEST_F(GlobalRegistryTest, BlockRegistrationDoesNotImplyStoreRegistration) {
  RaidenId id = {"jobS", "r0", "dataS", 0};
  ASSERT_TRUE(client_->Register({{"hashX", id, 7}}).ok());

  auto lookup = client_->Lookup({"hashX"});
  ASSERT_TRUE(lookup.ok());
  EXPECT_EQ(lookup->size(), 1);

  EXPECT_TRUE(absl::IsNotFound(client_->ResolveStore(id).status()));
}

// Real block hashes are raw digests, not text. Every other test in this file
// uses an ASCII hash, which is why nothing here noticed while the prefix-hash
// fields were declared `string`: proto3 validates UTF-8 on PARSE, so a real
// digest serialized fine at the client and then failed to decode at the
// server. The failure surfaced as a broken-looking peer, not as bad data.
//
// The payload below is deliberately fixed rather than random: it contains a
// bare 0xFF, a lone continuation byte and an embedded NUL, so it is certainly
// invalid UTF-8 rather than merely almost-certainly.
TEST_F(GlobalRegistryTest, RoundTripsNonUtf8PrefixHash) {
  const std::string binary_hash("\xff\xfe\x80\x00\x01\xc0\xaf\xed\xa0\x80", 10);
  ASSERT_FALSE(binary_hash.empty());
  RaidenId host = {"jobBin", "r0", "dataBin", 0};

  ASSERT_TRUE(client_->Register({{binary_hash, host, 11}}).ok());

  auto lookup = client_->Lookup({binary_hash});
  ASSERT_TRUE(lookup.ok()) << lookup.status().ToString();
  ASSERT_EQ(lookup->size(), 1);
  EXPECT_EQ((*lookup)[0].block_id(), 11);

  // The bytes must survive the round trip unaltered, not merely be accepted.
  auto owned = PullOwned(host);
  ASSERT_EQ(owned.size(), 1);
  EXPECT_EQ(owned[0].prefix_hash(), binary_hash);

  ASSERT_TRUE(client_->Unregister({binary_hash}, host).ok());
  auto after = client_->Lookup({binary_hash});
  ASSERT_TRUE(after.ok()) << after.status().ToString();
  EXPECT_TRUE(after->empty());
}

// The KV pool group most spec tests publish and read under.
constexpr absl::string_view kGroup = "prefill_pool";

// A well-formed spec: two block arrays (e.g. a unified full-attention array
// plus one state array), 2 shards, two transfer workers.
KVTransferSpec MakeTransferSpec() {
  KVTransferSpec spec;
  spec.add_block_arrays()->set_block_bytes(4096);
  spec.add_block_arrays()->set_block_bytes(512);
  spec.set_num_kv_shards(2);
  spec.set_num_workers(2);
  return spec;
}

TEST_F(GlobalRegistryTest, GetKVTransferSpecBeforePublishIsNotFound) {
  auto spec = client_->GetKVTransferSpec(kGroup);
  EXPECT_TRUE(absl::IsNotFound(spec.status())) << spec.status().ToString();
}

TEST_F(GlobalRegistryTest, RegisterKVTransferSpecRegistersAndGetRoundTrips) {
  const KVTransferSpec published = MakeTransferSpec();
  ASSERT_TRUE(client_->RegisterKVTransferSpec(published, kGroup).ok());

  auto spec = client_->GetKVTransferSpec(kGroup);
  ASSERT_TRUE(spec.ok()) << spec.status().ToString();
  ASSERT_EQ(spec->block_arrays_size(), published.block_arrays_size());
  EXPECT_EQ(spec->block_arrays(0).block_bytes(), 4096);
  EXPECT_EQ(spec->block_arrays(1).block_bytes(), 512);
  EXPECT_EQ(spec->num_kv_shards(), 2);
  EXPECT_EQ(spec->num_workers(), 2);
}

TEST_F(GlobalRegistryTest, RegisterKVTransferSpecIdenticalRepublishIsOk) {
  ASSERT_TRUE(client_->RegisterKVTransferSpec(MakeTransferSpec(), kGroup).ok());
  // The restart path: the same publisher (or an agreeing peer) re-publishes
  // the identical spec.
  EXPECT_TRUE(client_->RegisterKVTransferSpec(MakeTransferSpec(), kGroup).ok());
}

TEST_F(GlobalRegistryTest, RegisterKVTransferSpecRejectsEveryFieldMismatch) {
  ASSERT_TRUE(client_->RegisterKVTransferSpec(MakeTransferSpec(), kGroup).ok());

  KVTransferSpec other = MakeTransferSpec();
  other.mutable_block_arrays(1)->set_block_bytes(1024);
  absl::Status status = client_->RegisterKVTransferSpec(other, kGroup);
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status.ToString();
  EXPECT_TRUE(absl::StrContains(status.message(), "block_arrays[1]"))
      << status.ToString();

  other = MakeTransferSpec();
  other.add_block_arrays()->set_block_bytes(512);
  status = client_->RegisterKVTransferSpec(other, kGroup);
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status.ToString();

  other = MakeTransferSpec();
  other.set_num_kv_shards(4);
  status = client_->RegisterKVTransferSpec(other, kGroup);
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status.ToString();
  EXPECT_TRUE(absl::StrContains(status.message(), "num_kv_shards"))
      << status.ToString();

  other = MakeTransferSpec();
  other.set_num_workers(3);
  status = client_->RegisterKVTransferSpec(other, kGroup);
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status.ToString();
  EXPECT_TRUE(absl::StrContains(status.message(), "num_workers"))
      << status.ToString();

  // None of the rejected attempts replaced the registered spec.
  auto spec = client_->GetKVTransferSpec(kGroup);
  ASSERT_TRUE(spec.ok()) << spec.status().ToString();
  EXPECT_EQ(spec->block_arrays(1).block_bytes(), 512);
  EXPECT_EQ(spec->num_kv_shards(), 2);
  EXPECT_EQ(spec->num_workers(), 2);
}

TEST_F(GlobalRegistryTest, RegisterKVTransferSpecRejectsInvalidSpecs) {
  KVTransferSpec no_arrays = MakeTransferSpec();
  no_arrays.clear_block_arrays();
  EXPECT_TRUE(absl::IsInvalidArgument(
      client_->RegisterKVTransferSpec(no_arrays, kGroup)));

  KVTransferSpec zero_bytes = MakeTransferSpec();
  zero_bytes.mutable_block_arrays(0)->set_block_bytes(0);
  EXPECT_TRUE(absl::IsInvalidArgument(
      client_->RegisterKVTransferSpec(zero_bytes, kGroup)));

  KVTransferSpec zero_shards = MakeTransferSpec();
  zero_shards.set_num_kv_shards(0);
  EXPECT_TRUE(absl::IsInvalidArgument(
      client_->RegisterKVTransferSpec(zero_shards, kGroup)));

  KVTransferSpec zero_workers = MakeTransferSpec();
  zero_workers.set_num_workers(0);
  EXPECT_TRUE(absl::IsInvalidArgument(
      client_->RegisterKVTransferSpec(zero_workers, kGroup)));

  // A rejected spec registered nothing.
  EXPECT_TRUE(absl::IsNotFound(client_->GetKVTransferSpec(kGroup).status()));
}

TEST_F(GlobalRegistryTest, KVTransferSpecsAreIsolatedPerTransferGroup) {
  // Two differently-shaped groups (e.g. heterogeneous prefill and decode
  // jobs) register side by side without colliding.
  const KVTransferSpec prefill = MakeTransferSpec();
  KVTransferSpec decode = MakeTransferSpec();
  decode.set_num_kv_shards(8);
  decode.set_num_workers(1);
  ASSERT_TRUE(client_->RegisterKVTransferSpec(prefill, "prefill_pool").ok());
  ASSERT_TRUE(client_->RegisterKVTransferSpec(decode, "decode_pool").ok());

  auto prefill_read = client_->GetKVTransferSpec("prefill_pool");
  ASSERT_TRUE(prefill_read.ok()) << prefill_read.status().ToString();
  EXPECT_EQ(prefill_read->num_kv_shards(), 2);
  auto decode_read = client_->GetKVTransferSpec("decode_pool");
  ASSERT_TRUE(decode_read.ok()) << decode_read.status().ToString();
  EXPECT_EQ(decode_read->num_kv_shards(), 8);

  EXPECT_TRUE(
      absl::IsNotFound(client_->GetKVTransferSpec("no_such_group").status()));
}

TEST_F(GlobalRegistryTest, KVTransferSpecRejectsEmptyTransferGroup) {
  EXPECT_TRUE(absl::IsInvalidArgument(
      client_->RegisterKVTransferSpec(MakeTransferSpec(), "")));
  EXPECT_TRUE(absl::IsInvalidArgument(client_->GetKVTransferSpec("").status()));
}

TEST_F(GlobalRegistryTest, RegisterAsyncResolvesOk) {
  std::string hash = "async_hash1";
  RaidenId host = {"job1", "replica1", "data1", 0};
  int32_t block = 10;

  auto future = client_->RegisterAsync({{hash, host, block}});
  auto status = future.Await();
  EXPECT_TRUE(status.ok()) << status.ToString();

  auto lookup_res = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res.ok()) << lookup_res.status().ToString();
  ASSERT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].block_id(), block);
}

TEST_F(GlobalRegistryTest, UnregisterAsyncResolvesOk) {
  std::string hash = "async_hash2";
  RaidenId host = {"job1", "replica1", "data1", 0};
  int32_t block = 20;

  ASSERT_TRUE(client_->Register({{hash, host, block}}).ok());
  auto future = client_->UnregisterAsync({hash}, host);
  auto status = future.Await();
  EXPECT_TRUE(status.ok()) << status.ToString();

  auto lookup_res = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res.ok()) << lookup_res.status().ToString();
  EXPECT_EQ(lookup_res->size(), 0);
}

// FailedPrecondition, not Internal: the sync form's contract, which is now
// this fold.
TEST_F(GlobalRegistryTest, RegisterAsyncReportsServerRejection) {
  std::string empty_hash = "";
  RaidenId host = {"job1", "replica1", "data1", 0};

  auto future = client_->RegisterAsync({{empty_hash, host, 5}});
  auto status = future.Await();
  EXPECT_TRUE(absl::IsFailedPrecondition(status)) << status.ToString();
}

TEST(GlobalRegistryAsyncTest, AsyncCallDoesNotBlockTheCaller) {
  auto impl = std::make_unique<GlobalRegistryServiceImpl>();
  auto stalling_service =
      std::make_unique<StallingRegistryService>(std::move(impl));
  auto* service_ptr = stalling_service.get();
  service_ptr->EnableStall();

  auto server =
      CreateTestGlobalRegistryServerWithService(std::move(stalling_service));
  std::string hash = "stall_hash";
  RaidenId host = {"job1", "replica1", "data1", 0};

  // The server sits in Register until ReleaseStall, so a blocking client would
  // not return for at least kStall. Time the call itself.
  constexpr absl::Duration kStall = absl::Milliseconds(500);
  absl::Notification issued;
  std::thread releaser([&] {
    issued.WaitForNotification();
    service_ptr->WaitForStall();
    absl::SleepFor(kStall);
    service_ptr->ReleaseStall();
  });

  absl::Time before = absl::Now();
  auto future = server->client->RegisterAsync({{hash, host, 1}});
  absl::Duration issue_cost = absl::Now() - before;
  issued.Notify();

  EXPECT_LT(issue_cost, kStall / 5)
      << "RegisterAsync blocked for " << issue_cost
      << ", which is not meaningfully shorter than the " << kStall
      << " server stall it was supposed to return ahead of";
  EXPECT_FALSE(future.IsReady());

  auto status = future.Await();
  EXPECT_TRUE(status.ok()) << status.ToString();
  EXPECT_GE(absl::Now() - before, kStall)
      << "the stall did not actually happen, so this test proved nothing";
  releaser.join();
}

TEST(GlobalRegistryAsyncTest, AsyncCallHonoursItsDeadline) {
  auto impl = std::make_unique<GlobalRegistryServiceImpl>();
  auto stalling_service =
      std::make_unique<StallingRegistryService>(std::move(impl));
  auto* service_ptr = stalling_service.get();
  service_ptr->EnableStall();

  auto server =
      CreateTestGlobalRegistryServerWithService(std::move(stalling_service));
  std::string hash = "deadline_hash";
  RaidenId host = {"job1", "replica1", "data1", 0};

  auto future = server->client->RegisterAsync({{hash, host, 1}},
                                              absl::Milliseconds(50));
  service_ptr->WaitForStall();

  auto status = future.Await();
  EXPECT_FALSE(status.ok()) << "Expected deadline timeout error, got OK";

  service_ptr->ReleaseStall();
}

// The callback captures the stub, so the channel outlives the client that
// opened it.
TEST_F(GlobalRegistryTest, AsyncSurvivesClientDestruction) {
  std::string hash = "survive_client_destruction_hash";
  RaidenId host = {"job1", "replica1", "data1", 0};
  int32_t block = 88;

  // The channel is created inside the scope and dropped with the client, so
  // the callback's captured stub is the only thing keeping the transport
  // alive. Reusing the fixture's channel here would let a unique_ptr stub pass
  // this test.
  tsl::Future<> future;
  {
    std::shared_ptr<grpc::Channel> local_channel = grpc::CreateChannel(
        test_server_->server_address, grpc::InsecureChannelCredentials());
    GlobalRegistryClient local_client(local_channel);
    future = local_client.RegisterAsync({{hash, host, block}});
  }

  auto status = future.Await();
  EXPECT_TRUE(status.ok()) << status.ToString();

  auto lookup_res = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res.ok()) << lookup_res.status().ToString();
  ASSERT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].block_id(), block);
}

// Zero means "already expired" here, unlike the ttl parameters beside it.
TEST_F(GlobalRegistryTest, RegisterAsyncZeroTimeout) {
  std::string hash = "zero_timeout_hash";
  RaidenId host = {"job1", "replica1", "data1", 0};

  // Zero is a deadline of *now*, NOT "no deadline" -- unlike Registration::ttl
  // and RegisterStore's ttl, where zero means "never expires". The call must
  // fail rather than block forever, and must leave the registry untouched.
  auto future =
      client_->RegisterAsync({{hash, host, 1}}, absl::ZeroDuration());
  auto status = future.Await();
  ASSERT_FALSE(status.ok())
      << "zero timeout must expire the call, not mean 'never expire'";
  EXPECT_TRUE(absl::IsDeadlineExceeded(status)) << status.ToString();

  // Negative is expired too, and by the same rule rather than by accident.
  auto negative_status =
      client_->RegisterAsync({{hash, host, 1}}, -absl::Seconds(1)).Await();
  EXPECT_TRUE(absl::IsDeadlineExceeded(negative_status))
      << negative_status.ToString();

  auto lookup_res = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res.ok()) << lookup_res.status().ToString();
  EXPECT_EQ(lookup_res->size(), 0)
      << "an expired Register must not reach the registry";

  // And infinity is the other end of the same parameter: it succeeds.
  auto ok_future =
      client_->RegisterAsync({{hash, host, 1}}, absl::InfiniteDuration());
  EXPECT_TRUE(ok_future.Await().ok());
}

TEST_F(GlobalRegistryTest, UnregisterAsyncZeroTimeout) {
  std::string hash = "zero_timeout_unregister_hash";
  RaidenId host = {"job1", "replica1", "data1", 0};
  ASSERT_TRUE(client_->Register({{hash, host, 7}}).ok());

  auto status =
      client_->UnregisterAsync({hash}, host, absl::ZeroDuration()).Await();
  ASSERT_TRUE(absl::IsDeadlineExceeded(status)) << status.ToString();

  // The withdraw expired before dispatch, so the entry must still be there.
  auto lookup_res = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res.ok()) << lookup_res.status().ToString();
  EXPECT_EQ(lookup_res->size(), 1);
}

// Catches non-atomic stub access and any per-call state that is not per-call.
TEST_F(GlobalRegistryTest, ConcurrentAsyncRegistersFromManyThreads) {
  constexpr int kNumThreads = 8;
  constexpr int kEntriesPerThread = 10;
  RaidenId host = {"job1", "replica1", "data1", 0};

  std::vector<std::thread> threads;
  std::vector<tsl::Future<>> futures(kNumThreads * kEntriesPerThread);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([this, t, &futures, &host]() {
      for (int i = 0; i < kEntriesPerThread; ++i) {
        int idx = t * kEntriesPerThread + i;
        std::string hash = absl::StrCat("concurrent_hash_", idx);
        futures[idx] = client_->RegisterAsync({{hash, host, idx}});
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  for (int i = 0; i < kNumThreads * kEntriesPerThread; ++i) {
    auto status = futures[i].Await();
    EXPECT_TRUE(status.ok()) << "Entry " << i << " failed: " << status.ToString();
  }

  for (int i = 0; i < kNumThreads * kEntriesPerThread; ++i) {
    std::string hash = absl::StrCat("concurrent_hash_", i);
    auto lookup_res = client_->Lookup({hash});
    ASSERT_TRUE(lookup_res.ok()) << lookup_res.status().ToString();
    ASSERT_EQ(lookup_res->size(), 1);
    EXPECT_EQ((*lookup_res)[0].block_id(), i);
  }
}

// tsl::Future has no cancellation: dropping one must not drop the RPC.
TEST_F(GlobalRegistryTest, DroppedFutureStillReachesTheServer) {
  std::string hash = "dropped_future_hash";
  RaidenId host = {"job1", "replica1", "data1", 0};
  int32_t block = 99;

  (void)client_->RegisterAsync({{hash, host, block}});

  // Poll until the registration appears on the server
  bool found = false;
  absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (absl::Now() < deadline) {
    auto lookup_res = client_->Lookup({hash});
    if (lookup_res.ok() && !lookup_res->empty()) {
      EXPECT_EQ((*lookup_res)[0].block_id(), block);
      found = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_TRUE(found) << "Dropped future registration never appeared in registry";
}

// A one-entry batch cannot catch a half-built request; an empty one must
// resolve rather than hang.
TEST_F(GlobalRegistryTest, RegisterAsyncBatchAndEmptyBatch) {
  // Empty batch
  auto empty_future = client_->RegisterAsync({});
  EXPECT_TRUE(empty_future.Await().ok());

  // 50-entry batch
  constexpr int kBatchSize = 50;
  RaidenId host = {"job1", "replica1", "data1", 0};
  std::vector<Registration> registrations;
  std::vector<std::string> hashes;
  for (int i = 0; i < kBatchSize; ++i) {
    std::string hash = absl::StrCat("batch_hash_", i);
    hashes.push_back(hash);
    registrations.push_back({hash, host, i});
  }

  auto batch_future = client_->RegisterAsync(registrations);
  auto status = batch_future.Await();
  EXPECT_TRUE(status.ok()) << status.ToString();

  auto lookup_res = client_->Lookup(hashes);
  ASSERT_TRUE(lookup_res.ok()) << lookup_res.status().ToString();
  ASSERT_EQ(lookup_res->size(), kBatchSize);
  for (int i = 0; i < kBatchSize; ++i) {
    EXPECT_EQ((*lookup_res)[i].block_id(), i);
  }
}

}  // namespace
}  // namespace global_registry
}  // namespace kv_cache
}  // namespace tpu_raiden
