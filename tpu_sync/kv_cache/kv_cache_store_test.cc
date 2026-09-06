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

#include "tpu_sync/kv_cache/kv_cache_store.h"

#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/server_callback.h"
#include "grpcpp/support/status.h"
#include "grpcpp/support/sync_stream.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/controller/controller_client.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/core/controller/test_util.h"
#include "tpu_sync/core/kv_manager_holder.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/kv_cache/global_registry/global_registry.grpc.pb.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_client.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_server.h"
#include "tpu_sync/kv_cache/global_registry/test_util.h"
#include "tpu_sync/kv_cache/host_offload_backend.h"
#include "tpu_sync/kv_cache/kv_cache_metadata.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_sync/kv_cache/kv_cache_store_client.h"
#include "tpu_sync/proto/kv_cache_store_service.grpc.pb.h"

#ifndef _WIN32
int ignore_sigpipe = []() {
  std::signal(SIGPIPE, SIG_IGN);
  return 0;
}();

// Every KVCacheStore owns a real
// RaidenController, so this binary mints many controllers in one process.
// A private ControllerServer per controller keeps them from sharing (and
// re-pointing) the process-wide singleton.
int use_private_controller_servers = []() {
  setenv("RAIDEN_DISABLE_SINGLETON_WORKER", "1", /*overwrite=*/1);
  return 0;
}();
#endif

namespace tpu_raiden {
namespace kv_cache {

class KVCacheStoreTest {
 public:
  static size_t Evict(KVCacheStore& store,
                      const std::vector<std::string>& block_hashes) {
    return store.Evict(block_hashes);
  }

  static ::tpu_raiden::controller::RaidenController* GetController(
      const KVCacheStore& store) {
    return store.raiden_controller_.get();
  }

  static std::vector<std::string> GetEvictCandidateKeys(
      const KVCacheStore& store) {
    return store.backend() ? store.backend()->GetEvictCandidateKeys()
                           : std::vector<std::string>{};
  }

  static void PlantIndexEntry(KVCacheStore& store,
                              const std::vector<std::string>& block_hashes,
                              const std::vector<RaidenBlockId>& slices,
                              bool on_host) {
    store.backend()->Insert(block_hashes, slices, on_host);
  }

  static std::optional<KVCacheStore::RemoteWriteState> TakeRemoteWrite(
      KVCacheStore& store, KVCacheStore::OperationKey key) {
    return store.TakeRemoteWrite(key);
  }

  static void OnWriteRemoteVerdict(KVCacheStore& store,
                                   KVCacheStore::RemoteWriteState state,
                                   bool succeeded,
                                   std::vector<std::string> existing = {},
                                   std::vector<std::string> unregistered = {}) {
    store.OnWriteRemoteVerdict(std::move(state), succeeded, std::move(existing),
                               std::move(unregistered));
  }
};

class HostOffloadBackendTest {
 public:
  // Tests construct backends directly; production code goes through
  // HostOffloadBackend::Create.
  class Backend : public HostOffloadBackend {
   public:
    template <typename... Args>
    explicit Backend(Args&&... args)
        : HostOffloadBackend(std::forward<Args>(args)...) {}
  };
};

// Direct-construction shorthand for the backend fixtures below.
using TestHostOffloadBackend = HostOffloadBackendTest::Backend;

namespace {

// Observes what is resident WITHOUT acquiring it. The application-level
// Lookup pins what it returns, which is right for a caller that goes on to
// load or save the answer but wrong for a case that is asserting on pin counts
// or eviction candidates -- there the pin is the thing under test, and a
// lookup taking one of its own would be measuring itself.
absl::StatusOr<BlockSliceList> PeekLookup(
    KVCacheStore& store, const std::vector<std::string>& block_hashes) {
  return store.Lookup(block_hashes, LookupOptions{.enable_global = false});
}

// Populates the cache with ordinary, EVICTABLE residents.
//
// Insert() pins what it takes -- that pin is what a later Save() consumes --
// but most cases below only need the cache populated, not a pin held, and a
// held pin makes an entry unevictable and undeletable. Handing it straight
// back is what the real flow does via a successful save; doing it here keeps
// those cases about their own subject.
//
// Cases whose subject IS insert's pin call Insert() directly.
bool InsertResident(KVCacheStore& store,
                    const std::vector<std::string>& block_hashes,
                    const std::vector<RaidenBlockId>& slices, bool on_host) {
  if (!store.Insert(block_hashes, slices, on_host).ok()) return false;
  store.Release(block_hashes);
  return true;
}

// Publishes a peer so a remote read can resolve its controller. The store
// server address is required by the registry but a read never dials it -- it
// speaks to the controller.
absl::Status PublishPeerController(absl::string_view registry_address,
                                   const RaidenId& peer_id,
                                   absl::string_view controller_address) {
  global_registry::GlobalRegistryClient client(grpc::CreateChannel(
      std::string(registry_address), grpc::InsecureChannelCredentials()));
  return client.RegisterStore(peer_id, "peer.store.unused:1",
                              controller_address);
}

TEST(KVCacheStoreTest, RaidenBlockIdConstructorAndEquality) {
  RaidenId id{"test_job", "0", "test_cache", 0};
  RaidenBlockId block_1(id, 10, 20, BlockStatus::HBM);
  EXPECT_EQ(block_1.raiden_id, id);
  EXPECT_EQ(block_1.host_block_id, 10);
  EXPECT_EQ(block_1.device_block_id, 20);
  EXPECT_EQ(block_1.status, BlockStatus::HBM);

  RaidenBlockId block_2(id, 10, 20, BlockStatus::HBM);
  EXPECT_EQ(block_1, block_2);

  RaidenBlockId block_3(id, 10, 21, BlockStatus::HBM);
  EXPECT_NE(block_1, block_3);

  RaidenBlockId block_4(id, 11, 20, BlockStatus::HBM);
  EXPECT_NE(block_1, block_4);
}

TEST(KVCacheStoreTest, EvictionTracking) {
  KVCacheStore controller(2, "", {}, /*num_shards=*/1,
                          /*shard_size_bytes=*/512,
                          /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes_1_2 = {"101", "102"};
  std::vector<RaidenBlockId> slices_1_2 = {
      RaidenId{"inference_server", "0", "kv_cache", 0},
      RaidenId{"inference_server", "1", "kv_cache", 1}};

  // 1. Insert 101 and 102, filling the cache. Both come back PINNED.
  EXPECT_TRUE(controller.Insert(hashes_1_2, slices_1_2, true).ok());

  // 2. Insert 103. available_space() is capacity minus the PINNED entries, so
  // with both slots pinned there is nothing insert is allowed to reclaim and
  // it refuses rather than evicting a block somebody is holding. All-or-
  // nothing: 103 is not inserted and 101/102 keep their pins.
  std::vector<std::string> hash_3 = {"103"};
  std::vector<RaidenBlockId> slice_3 = {
      RaidenId{"inference_server", "2", "kv_cache", 2}};

  EXPECT_FALSE(controller.Insert(hash_3, slice_3, true).ok());
  EXPECT_EQ(controller.GetPinCount("101"), 1);
  EXPECT_EQ(controller.GetPinCount("102"), 1);

  // 3. Release them -- in the real flow a successful save() does this -- and
  // the same insert now succeeds by evicting.
  //
  // The victim is 102, not 101. Release walks the batch in REVERSE so that the
  // tail of a sequence becomes evictable first, which is what you want for
  // prefix caches: dropping the tail of a shared prefix costs less than
  // dropping its head.
  controller.Release(hashes_1_2);
  EXPECT_TRUE(controller.Insert(hash_3, slice_3, true).ok());

  // 4. Verify that 102 is in candidates.
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(controller),
              ::testing::ElementsAre("102"));

  // 5. Verify that lookup for 102 misses (candidate invisible with Peek).
  auto lookup_res = PeekLookup(controller, {"102"});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_TRUE(lookup_res->empty());
  // 102 should still be in candidates.
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(controller),
              ::testing::ElementsAre("102"));

  EXPECT_EQ(PeekLookup(controller, {"101"})->size(), 1);
  EXPECT_EQ(PeekLookup(controller, {"103"})->size(), 1);
}

TEST(KVCacheStoreTest, GlobalLookupFallback) {
  // 1. Start a local registry server
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = reg_server->server_address;

  // 2. Register some blocks in the registry
  auto& registry_client = *reg_server->client;

  std::string hash1 = "global_hash_1";
  RaidenId host1{"job1", "0", "kv_cache", 0};
  int32_t block1 = 42;

  std::string hash2 = "global_hash_2";
  RaidenId host2{"job2", "0", "kv_cache", 0};
  int32_t block2 = 43;

  std::string hash_shared = "shared_hash";
  RaidenId host_shared_remote{"job_shared", "0", "kv_cache", 0};
  int32_t block_shared_remote = 99;

  ASSERT_TRUE(
      registry_client
          .Register({{hash1, host1, block1},
                     {hash2, host2, block2},
                     {hash_shared, host_shared_remote, block_shared_remote}})
          .ok());

  // 3. Create KVCacheStore with the registry address
  RaidenId store_id{"store_job", "0", "kv_cache", 0};
  KVCacheStore store(50, server_address, store_id, /*num_shards=*/1,
                     /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");

  // Insert blocks locally
  std::vector<std::string> local_hashes = {"local_only_hash", "shared_hash"};
  std::vector<RaidenBlockId> local_slices = {
      RaidenId{"local_job", "0", "kv_cache", 0},
      RaidenId{"local_job", "0", "kv_cache", 1}};
  ASSERT_TRUE(InsertResident(store, local_hashes, local_slices, true));

  // Case 1: Full local hit, no global hit
  {
    auto lookup_res = store.Lookup({"local_only_hash"},
                                   LookupOptions{.enable_global = true});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 1);
    EXPECT_EQ((*lookup_res)[0].first, "local_only_hash");
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.job_name, "local_job");
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.data_replica_idx, 0);
  }

  // Case 2: Both local and global has the same hit, but we return local hit
  // results
  {
    auto lookup_res =
        store.Lookup({"shared_hash"}, LookupOptions{.enable_global = true});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 1);
    EXPECT_EQ((*lookup_res)[0].first, "shared_hash");
    // Should return local info, not remote info from registry
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.job_name, "local_job");
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.data_replica_idx, 1);
  }

  // Case 3: No local hit, only global hits
  {
    auto lookup_res = store.Lookup({"global_hash_1", "global_hash_2"},
                                   LookupOptions{.enable_global = true});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 2);

    EXPECT_EQ((*lookup_res)[0].first, "global_hash_1");
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.job_name, "job1");
    EXPECT_EQ((*lookup_res)[0].second.host_block_id, 42);
    EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::REMOTE);

    EXPECT_EQ((*lookup_res)[1].first, "global_hash_2");
    EXPECT_EQ((*lookup_res)[1].second.raiden_id.job_name, "job2");
    EXPECT_EQ((*lookup_res)[1].second.host_block_id, 43);
    EXPECT_EQ((*lookup_res)[1].second.status, BlockStatus::REMOTE);
  }

  // 4. Lookup with enable_global = false
  // It should stop at the first miss (which is the global hash if we query it)
  // If we query {"local_only_hash", "global_hash_1"}, it should return
  // local_only_hash and stop.
  {
    auto lookup_res = PeekLookup(store, {"local_only_hash", "global_hash_1"});
    ASSERT_TRUE(lookup_res.ok());
    EXPECT_EQ(lookup_res->size(), 1);
    EXPECT_EQ((*lookup_res)[0].first, "local_only_hash");
  }

  // 5. Lookup with enable_global = true
  // It should return both local and global
  {
    auto lookup_res =
        store.Lookup({"local_only_hash", "global_hash_1", "global_hash_2"},
                     LookupOptions{.enable_global = true});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 3);

    EXPECT_EQ((*lookup_res)[0].first, "local_only_hash");
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.job_name, "local_job");

    EXPECT_EQ((*lookup_res)[1].first, "global_hash_1");
    EXPECT_EQ((*lookup_res)[1].second.raiden_id.job_name, "job1");
    EXPECT_EQ((*lookup_res)[1].second.host_block_id, 42);
    EXPECT_EQ((*lookup_res)[1].second.status, BlockStatus::REMOTE);

    EXPECT_EQ((*lookup_res)[2].first, "global_hash_2");
    EXPECT_EQ((*lookup_res)[2].second.raiden_id.job_name, "job2");
    EXPECT_EQ((*lookup_res)[2].second.host_block_id, 43);
    EXPECT_EQ((*lookup_res)[2].second.status, BlockStatus::REMOTE);
  }

  // 6. Lookup with enable_global = true, but registry has a miss
  // It should stop at the first miss in registry
  {
    auto lookup_res = store.Lookup(
        {"local_only_hash", "global_hash_1", "missing_hash", "global_hash_2"},
        LookupOptions{.enable_global = true});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 2);  // local_only_hash, global_hash_1
    EXPECT_EQ((*lookup_res)[0].first, "local_only_hash");
    EXPECT_EQ((*lookup_res)[1].first, "global_hash_1");
  }
}

// Delegates every RPC to a real GlobalRegistryServiceImpl and counts
// ResolveStore. A remote read resolves its peer's controller address through
// that call, so the count is what distinguishes "cached" from "re-resolved" --
// the only externally visible difference between the two.
class CountingRegistryService final
    : public global_registry::GlobalRegistryService::Service {
 public:
  grpc::Status Register(grpc::ServerContext* context,
                        const global_registry::RegisterRequest* request,
                        global_registry::RegisterResponse* response) override {
    return delegate_.Register(context, request, response);
  }
  grpc::Status Lookup(grpc::ServerContext* context,
                      const global_registry::LookupRequest* request,
                      global_registry::LookupResponse* response) override {
    return delegate_.Lookup(context, request, response);
  }
  grpc::Status Unregister(
      grpc::ServerContext* context,
      const global_registry::UnregisterRequest* request,
      global_registry::UnregisterResponse* response) override {
    return delegate_.Unregister(context, request, response);
  }
  grpc::Status PullOwned(
      grpc::ServerContext* context,
      const global_registry::PullOwnedRequest* request,
      grpc::ServerWriter<global_registry::PullOwnedResponse>* writer) override {
    return delegate_.PullOwned(context, request, writer);
  }
  grpc::Status RegisterStore(
      grpc::ServerContext* context,
      const global_registry::RegisterStoreRequest* request,
      global_registry::RegisterStoreResponse* response) override {
    return delegate_.RegisterStore(context, request, response);
  }
  grpc::Status ResolveStore(
      grpc::ServerContext* context,
      const global_registry::ResolveStoreRequest* request,
      global_registry::ResolveStoreResponse* response) override {
    resolve_store_calls.fetch_add(1);
    return delegate_.ResolveStore(context, request, response);
  }
  grpc::Status UnregisterStore(
      grpc::ServerContext* context,
      const global_registry::UnregisterStoreRequest* request,
      global_registry::UnregisterStoreResponse* response) override {
    return delegate_.UnregisterStore(context, request, response);
  }

  std::atomic<int> resolve_store_calls{0};

 private:
  global_registry::GlobalRegistryServiceImpl delegate_;
};

// Delegates every RPC to a real GlobalRegistryServiceImpl except Lookup,
// which always fails. Models a registry that IS reachable -- RegisterStore at
// construction succeeds -- but whose Lookup path is down. The old version of
// this test pointed construction itself at an unreachable address; that no
// longer works because RegisterStore failure now fails construction:
// a registry that cannot be reached AT ALL can no
// longer be distinguished from a misconfigured store, so the only
// representable "down" is Lookup failing after the store is already up.
class LookupFailingRegistryService final
    : public global_registry::GlobalRegistryService::Service {
 public:
  grpc::Status Register(grpc::ServerContext* context,
                        const global_registry::RegisterRequest* request,
                        global_registry::RegisterResponse* response) override {
    return delegate_.Register(context, request, response);
  }
  grpc::Status Lookup(grpc::ServerContext* context,
                      const global_registry::LookupRequest* request,
                      global_registry::LookupResponse* response) override {
    return grpc::Status(grpc::StatusCode::UNAVAILABLE, "registry lookup down");
  }
  grpc::Status Unregister(
      grpc::ServerContext* context,
      const global_registry::UnregisterRequest* request,
      global_registry::UnregisterResponse* response) override {
    return delegate_.Unregister(context, request, response);
  }
  grpc::Status PullOwned(
      grpc::ServerContext* context,
      const global_registry::PullOwnedRequest* request,
      grpc::ServerWriter<global_registry::PullOwnedResponse>* writer) override {
    return delegate_.PullOwned(context, request, writer);
  }
  grpc::Status RegisterStore(
      grpc::ServerContext* context,
      const global_registry::RegisterStoreRequest* request,
      global_registry::RegisterStoreResponse* response) override {
    return delegate_.RegisterStore(context, request, response);
  }
  grpc::Status ResolveStore(
      grpc::ServerContext* context,
      const global_registry::ResolveStoreRequest* request,
      global_registry::ResolveStoreResponse* response) override {
    return delegate_.ResolveStore(context, request, response);
  }
  grpc::Status UnregisterStore(
      grpc::ServerContext* context,
      const global_registry::UnregisterStoreRequest* request,
      global_registry::UnregisterStoreResponse* response) override {
    return delegate_.UnregisterStore(context, request, response);
  }

 private:
  global_registry::GlobalRegistryServiceImpl delegate_;
};

TEST(KVCacheStoreTest, GlobalLookupRegistryDown) {
  LookupFailingRegistryService service;
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);

  // Construction succeeds: RegisterStore is reachable and healthy.
  RaidenId store_id{"store_job", "0", "kv_cache", 0};
  KVCacheStore store(50, server_address, store_id, /*num_shards=*/1,
                     /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");

  // Insert one block locally
  std::vector<std::string> local_hashes = {"local_hash"};
  std::vector<RaidenBlockId> local_slices = {
      RaidenId{"local_job", "0", "kv_cache", 0}};
  ASSERT_TRUE(InsertResident(store, local_hashes, local_slices, true));

  // Lookup with enable_global = true -- explicitly, because the default is
  // false and a defaulted call would never dial the failing registry at all.
  // It should NOT fail even though Lookup RPCs to the registry fail. It
  // should return the local hit. Observation only, so no pin is taken.
  auto lookup_res = store.Lookup({"local_hash", "missing_hash"},
                                 LookupOptions{.enable_global = true});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].first, "local_hash");
  EXPECT_EQ((*lookup_res)[0].second.raiden_id.job_name, "local_job");

  server->Shutdown();
}

// --- ReadRemote All-or-Nothing validate & pin block hashes at the src
// controller: source-side ValidateAndPinHostBlocks ---

TEST(KVCacheStoreTest, ValidateAndPinHostBlocksSuccessReDerivesIdsAndPins) {
  KVCacheStore store(4, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");
  RaidenId rid{"src_job", "0", "src_cache", 0};
  std::vector<std::string> hashes = {"h0", "h1"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, /*host_block_id=*/5, /*device_block_id=*/-1,
                    BlockStatus::HOST),
      RaidenBlockId(rid, /*host_block_id=*/7, /*device_block_id=*/-1,
                    BlockStatus::HOST_AND_HBM)};
  ASSERT_TRUE(InsertResident(store, hashes, slices, /*on_host=*/true));

  auto ids_or = store.ValidateAndPinHostBlocks(hashes);
  ASSERT_TRUE(ids_or.ok()) << ids_or.status().message();
  // Source ids are re-derived from the LRU (not from the request).
  EXPECT_THAT(*ids_or, ::testing::ElementsAre(5, 7));
  EXPECT_EQ(store.GetPinCount("h0"), 1);
  EXPECT_EQ(store.GetPinCount("h1"), 1);

  store.UnpinHostBlocks(hashes);
  EXPECT_EQ(store.GetPinCount("h0"), 0);
  EXPECT_EQ(store.GetPinCount("h1"), 0);
}

TEST(KVCacheStoreTest, ValidateAndPinHostBlocksMissingReturnsNotFound) {
  KVCacheStore store(4, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");
  auto ids_or =
      store.ValidateAndPinHostBlocks(std::vector<std::string>{"missing"});
  EXPECT_TRUE(absl::IsNotFound(ids_or.status())) << ids_or.status();
}

TEST(KVCacheStoreTest, ValidateAndPinHostBlocksWrongStatusFailedPrecondition) {
  KVCacheStore store(4, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");
  RaidenId rid{"src_job", "0", "src_cache", 0};
  std::vector<std::string> hashes = {"hbm_h"};
  std::vector<RaidenBlockId> slices = {RaidenBlockId(
      rid, /*host_block_id=*/-1, /*device_block_id=*/0, BlockStatus::HBM)};
  ASSERT_TRUE(InsertResident(store, hashes, slices, /*on_host=*/false));

  auto ids_or = store.ValidateAndPinHostBlocks(hashes);
  EXPECT_TRUE(absl::IsFailedPrecondition(ids_or.status())) << ids_or.status();
  EXPECT_EQ(store.GetPinCount("hbm_h"), 0);
}

TEST(KVCacheStoreTest, ValidateAndPinHostBlocksAtomicRollbackOnPartialMiss) {
  KVCacheStore store(4, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");
  RaidenId rid{"src_job", "0", "src_cache", 0};
  // "ok" is HOST, "bad" is HBM-only (not host-resident) -> the whole batch
  // must abort and "ok" must NOT remain pinned (all-or-nothing).
  std::vector<std::string> hashes = {"ok", "bad"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, /*host_block_id=*/3, /*device_block_id=*/-1,
                    BlockStatus::HOST),
      RaidenBlockId(rid, /*host_block_id=*/-1, /*device_block_id=*/0,
                    BlockStatus::HBM)};
  ASSERT_TRUE(InsertResident(store, hashes, slices, /*on_host=*/false));

  auto ids_or = store.ValidateAndPinHostBlocks(hashes);
  EXPECT_FALSE(ids_or.ok());
  EXPECT_EQ(store.GetPinCount("ok"), 0);
  EXPECT_EQ(store.GetPinCount("bad"), 0);
}

TEST(KVCacheStoreTest, ValidateAndPinHostBlocksEmptyInputIsOk) {
  KVCacheStore store(4, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");
  auto ids_or = store.ValidateAndPinHostBlocks(std::vector<std::string>{});
  ASSERT_TRUE(ids_or.ok());
  EXPECT_TRUE(ids_or->empty());
}

TEST(KVCacheStoreTest,
     ValidateAndPinHostBlocksIncrementsAndReleasesExistingPin) {
  KVCacheStore store(4, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");
  RaidenId rid{"src_job", "0", "src_cache", 0};
  std::vector<std::string> hashes = {"h0"};
  std::vector<RaidenBlockId> slices = {RaidenBlockId(
      rid, /*host_block_id=*/9, /*device_block_id=*/-1, BlockStatus::HOST)};
  // Insert pins once, and this case counts from that pin.
  ASSERT_TRUE(store.Insert(hashes, slices, /*on_host=*/true).ok());
  EXPECT_EQ(store.GetPinCount("h0"), 1);

  auto ids_or = store.ValidateAndPinHostBlocks(hashes);
  ASSERT_TRUE(ids_or.ok());
  EXPECT_THAT(*ids_or, ::testing::ElementsAre(9));
  EXPECT_EQ(store.GetPinCount("h0"), 2);  // verify added a second pin.

  store.UnpinHostBlocks(hashes);
  EXPECT_EQ(store.GetPinCount("h0"), 1);  // back to the caller's pin.
}

TEST(KVCacheStoreTest, LookupCapLimit) {
  KVCacheStore store(2, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"101", "102"};
  std::vector<RaidenBlockId> slices = {
      RaidenId{"inference_server", "0", "kv_cache", 0},
      RaidenId{"inference_server", "1", "kv_cache", 1}};

  ASSERT_TRUE(InsertResident(store, hashes, slices, true));

  // Lookup 3 hashes, but capacity is 2. It should only return 2.
  std::vector<std::string> lookup_hashes = {"101", "102", "103"};
  auto lookup_res = PeekLookup(store, lookup_hashes);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "101");
  EXPECT_EQ((*lookup_res)[1].first, "102");
}

TEST(KVCacheStoreTest, LookupCapLimitWithGlobal) {
  // 1. Start a local registry server
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = reg_server->server_address;

  // 2. Register some blocks in the registry
  auto& registry_client = *reg_server->client;

  std::string hash1 = "global_hash_1";
  RaidenId host1{"job1", "0", "kv_cache", 0};
  int32_t block1 = 42;

  std::string hash2 = "global_hash_2";
  RaidenId host2{"job2", "0", "kv_cache", 0};
  int32_t block2 = 43;

  std::string hash3 = "global_hash_3";
  RaidenId host3{"job3", "0", "kv_cache", 0};
  int32_t block3 = 44;

  ASSERT_TRUE(registry_client
                  .Register({{hash1, host1, block1},
                             {hash2, host2, block2},
                             {hash3, host3, block3}})
                  .ok());

  // 3. Create KVCacheStore with capacity 2
  RaidenId store_id{"store_job", "0", "kv_cache", 0};
  KVCacheStore store(2, server_address, store_id, /*num_shards=*/1,
                     /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");

  // Lookup 3 hashes, but capacity is 2. It should only return 2.
  std::vector<std::string> lookup_hashes = {"global_hash_1", "global_hash_2",
                                            "global_hash_3"};
  auto lookup_res = store.Lookup(lookup_hashes, /*enable_global=*/true);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "global_hash_1");
  EXPECT_EQ((*lookup_res)[1].first, "global_hash_2");
}

TEST(KVCacheStoreTest, LookupCapLimitMixed) {
  // 1. Start a local registry server
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = reg_server->server_address;

  // 2. Register some blocks in the registry
  auto& registry_client = *reg_server->client;

  std::string hash2 = "global_hash_2";
  RaidenId host2{"job2", "0", "kv_cache", 0};
  int32_t block2 = 43;

  std::string hash3 = "global_hash_3";
  RaidenId host3{"job3", "0", "kv_cache", 0};
  int32_t block3 = 44;

  ASSERT_TRUE(
      registry_client.Register({{hash2, host2, block2}, {hash3, host3, block3}})
          .ok());

  // 3. Create KVCacheStore with capacity 2
  RaidenId store_id{"store_job", "0", "kv_cache", 0};
  KVCacheStore store(2, server_address, store_id, /*num_shards=*/1,
                     /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");

  // Insert 1 block locally
  std::vector<std::string> local_hashes = {"local_hash_1"};
  std::vector<RaidenBlockId> local_slices = {
      RaidenId{"local_job", "0", "kv_cache", 0}};
  ASSERT_TRUE(InsertResident(store, local_hashes, local_slices, true));

  // Lookup 3 hashes, but capacity is 2. It should only return 2 (1 local, 1
  // global).
  std::vector<std::string> lookup_hashes = {"local_hash_1", "global_hash_2",
                                            "global_hash_3"};
  auto lookup_res =
      store.Lookup(lookup_hashes, LookupOptions{.enable_global = true});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "local_hash_1");
  EXPECT_EQ((*lookup_res)[1].first, "global_hash_2");
}

// Decision: enable_global defaults to FALSE -- the registry query is a
// blocking RPC a caller should not pay for by omission. A hash the registry
// resolves but the local index does not is therefore invisible to a
// defaulted lookup, and visible the moment the caller asks globally. This is
// the test that fails if the default ever flips back.
TEST(KVCacheStoreTest, LookupDefaultSkipsTheRegistry) {
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  auto& registry_client = *reg_server->client;

  RaidenId owner{"peer_job", "0", "kv_cache", 0};
  ASSERT_TRUE(registry_client.Register({{"registry_only", owner, 42}}).ok());

  RaidenId store_id{"store_job", "0", "kv_cache", 0};
  KVCacheStore store(4, reg_server->server_address, store_id,
                     /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");

  // Defaulted: local miss, registry never consulted.
  auto defaulted = store.Lookup({"registry_only"});
  ASSERT_TRUE(defaulted.ok());
  EXPECT_TRUE(defaulted->empty());

  // Asked globally, the same hash resolves.
  auto global = store.Lookup({"registry_only"},
                             LookupOptions{.enable_global = true});
  ASSERT_TRUE(global.ok());
  ASSERT_EQ(global->size(), 1);
  EXPECT_EQ((*global)[0].second.status, BlockStatus::REMOTE);
  EXPECT_EQ((*global)[0].second.raiden_id, owner);
}

TEST(KVCacheStoreTest, LookupAvailableSpaceLimit) {
  KVCacheStore store(3, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"101", "102", "103"};
  std::vector<RaidenBlockId> slices = {
      RaidenId{"inference_server", "0", "kv_cache", 0},
      RaidenId{"inference_server", "1", "kv_cache", 1},
      RaidenId{"inference_server", "2", "kv_cache", 2}};

  ASSERT_TRUE(InsertResident(store, hashes, slices, true));

  // Pin 101. Pinned count = 1. Available space = 3 - 1 = 2.
  EXPECT_TRUE(store.Lookup({"101"}).ok());

  // Lookup 4 hashes. Lookup is non-mutating and unbounded by available space,
  // returning all 3 cached blocks up to the first miss ("104").
  std::vector<std::string> lookup_hashes = {"101", "102", "103", "104"};
  auto lookup_res = PeekLookup(store, lookup_hashes);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 3);
  EXPECT_EQ((*lookup_res)[0].first, "101");
  EXPECT_EQ((*lookup_res)[1].first, "102");
  EXPECT_EQ((*lookup_res)[2].first, "103");

  // Give back the deliberate pin on 101.
  store.Release({"101"});
}

TEST(KVCacheStoreTest, InsertPinsExistingAndNewAlike) {
  KVCacheStore store(2, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");

  // Already resident, and unpinned -- the pins under test are the ones the
  // insert below grants.
  std::vector<std::string> local_hashes = {"local_1"};
  std::vector<RaidenBlockId> local_slices = {
      RaidenId{"local_job", "0", "kv_cache", 0}};
  ASSERT_TRUE(InsertResident(store, local_hashes, local_slices, true));

  // One hash already present, one new: both come back pinned exactly once.
  std::vector<RaidenBlockId> slices = {
      RaidenId{"local_job", "0", "kv_cache", 0},
      RaidenId{"remote_job", "0", "kv_cache", 42}};
  EXPECT_TRUE(store.Insert({"local_1", "remote_1"}, slices, true).ok());
  EXPECT_EQ(store.GetPinCount("local_1"), 1);
  EXPECT_EQ(store.GetPinCount("remote_1"), 1);

  // Capacity is 2 and both entries are pinned, so available space is 0 and
  // there is nothing insert is allowed to reclaim: remote_2 is refused rather
  // than evicting a block somebody is holding.
  absl::Status refused = store.Insert(
      {"remote_2"}, {RaidenBlockId(RaidenId{"local_job", "0", "kv_cache", 2},
                                   2, BlockStatus::HOST)},
      true);
  EXPECT_TRUE(absl::IsResourceExhausted(refused)) << refused;
}

// The LRU cache holds LOCAL blocks only: a REMOTE slice names a block on
// another node. Insert refuses the whole batch before touching anything, so
// a bad batch cannot pin or insert a thing -- not even its local members.
TEST(KVCacheStoreTest, InsertRejectsRemoteSlices) {
  KVCacheStore store(4, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");

  RaidenId id{"job", "0", "cache", 0};
  std::vector<std::string> resident = {"h1"};
  ASSERT_TRUE(InsertResident(store, resident,
                             {RaidenBlockId(id, 1, BlockStatus::HOST)}, true));

  // A mixed batch: h1 already resident and local, h2 REMOTE.
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(id, 1, BlockStatus::HOST),
      RaidenBlockId(RaidenId{"peer_job", "0", "cache", 0}, 42,
                    BlockStatus::REMOTE)};
  absl::Status status = store.Insert({"h1", "h2"}, slices, true);
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;

  // Nothing happened: h1 is not pinned and h2 was never inserted.
  EXPECT_EQ(store.GetPinCount("h1"), 0);
  EXPECT_TRUE(PeekLookup(store, {"h2"})->empty());
}

TEST(KVCacheStoreTest, EvictRaceCondition) {
  KVCacheStore store(3, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");

  // Insert local_1 (HOST status)
  std::vector<std::string> local_hashes = {"local_1"};
  std::vector<RaidenBlockId> local_slices = {RaidenBlockId(
      RaidenId{"local_job", "0", "kv_cache", 0}, -1, BlockStatus::HOST)};
  ASSERT_TRUE(InsertResident(store, local_hashes, local_slices, true));

  // Pin local_1
  ASSERT_TRUE(store.Lookup({"local_1"}).ok());
  EXPECT_EQ(store.GetPinCount("local_1"), 1);

  // Attempt Evict on local_1 (which is pinned)
  size_t evicted = KVCacheStoreTest::Evict(store, {"local_1"});
  EXPECT_EQ(evicted, 0);

  // Verify local_1 is still in the cache and pinned. PeekLookup: the pin
  // count above is the subject, so the check must not add one of its own.
  EXPECT_EQ(store.GetPinCount("local_1"), 1);
  auto lookup_res = PeekLookup(store, {"local_1"});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 1);
}

using ::testing::ElementsAre;

// 64-byte aligned backing buffer standing in for the shared memory region
// that would hold the KV cache metadata table.
class MetadataRegion {
 public:
  explicit MetadataRegion(int num_blocks)
      : buffer_(KVCacheMetadata::RequiredSizeBytes(num_blocks) + 63) {}

  absl::Span<uint8_t> span() {
    auto addr = reinterpret_cast<uintptr_t>(buffer_.data());
    size_t offset = (64 - addr % 64) % 64;
    return absl::MakeSpan(buffer_.data() + offset, buffer_.size() - offset);
  }

 private:
  std::vector<uint8_t> buffer_;
};

// Worker-less controller: sufficient for the metadata and recovery tests,
// which only touch the logical block manager.
std::unique_ptr<::tpu_raiden::controller::RaidenController>
MakeRecoveryController(const RaidenId& rid, int num_blocks) {
  ::tpu_sync::rpc::RaidenIdProto unit;
  unit.set_job_name(rid.job_name);
  unit.set_job_replica_id(rid.job_replica_id);
  unit.set_data_name(rid.data_name);
  unit.set_data_replica_idx(rid.data_replica_idx);
  return *::tpu_raiden::controller::RaidenController::Create(
      unit, num_blocks, /*num_shards=*/1, /*shard_size_bytes=*/512,
      /*raiden_controller_address=*/"");
}


TEST(KVCacheStoreTest, MetadataKeepsEvictionCandidates) {
  MetadataRegion region(4);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 4);
  ASSERT_TRUE(metadata_or.ok());

  RaidenId rid{"local_job", "0", "kv_cache", 0};
  KVCacheStore store(2, MakeRecoveryController(rid, 4),
                     /*global_registry_address=*/"", rid, *metadata_or,
                     /*store_server_ip=*/"127.0.0.1");

  ASSERT_TRUE(InsertResident(store, {"host_1", "host_2"},
                             {RaidenBlockId(rid, 0, BlockStatus::HOST),
                              RaidenBlockId(rid, 1, BlockStatus::HOST)},
                             true));

  // Inserting host_3 moves host_2 to the candidate list. Its host block stays
  // allocated, so its metadata entry survives.
  //
  // host_2, the TAIL of the batch, not host_1: releasing a batch walks it in
  // reverse and each unpin promotes to MRU, so the first hash ends up the most
  // recently used and the last is evicted first. That is the documented intent
  // -- dropping the tail of a shared prefix costs less than dropping its head
  // -- and it is what pinning on insert made actually true.
  ASSERT_TRUE(InsertResident(store, {"host_3"},
                             {RaidenBlockId(rid, 2, BlockStatus::HOST)}, true));
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ElementsAre("host_2"));
  EXPECT_THAT(metadata_or->ValidEntries(),
              ElementsAre(::testing::FieldsAre(0, "host_1", 1),
                          ::testing::FieldsAre(1, "host_2", 0),
                          ::testing::FieldsAre(2, "host_3", 2)));

  // Re-inserting host_2 under a new host block reactivates the candidate and
  // overwrites its binding in place: block 1's entry is cleared and block 3's
  // is set with the newest seq.
  ASSERT_TRUE(InsertResident(store, {"host_2"},
                             {RaidenBlockId(rid, 3, BlockStatus::HOST)}, true));
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::IsEmpty());
  EXPECT_THAT(metadata_or->ValidEntries(),
              ElementsAre(::testing::FieldsAre(0, "host_1", 1),
                          ::testing::FieldsAre(2, "host_3", 2),
                          ::testing::FieldsAre(3, "host_2", 3)));
}

TEST(KVCacheStoreTest, EvictClearsMetadataEntries) {
  MetadataRegion region(2);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 2);
  ASSERT_TRUE(metadata_or.ok());

  RaidenId rid{"local_job", "0", "kv_cache", 0};
  KVCacheStore store(2, MakeRecoveryController(rid, 2),
                     /*global_registry_address=*/"", rid, *metadata_or,
                     /*store_server_ip=*/"127.0.0.1");

  ASSERT_TRUE(InsertResident(store, {"host_1", "host_2"},
                             {RaidenBlockId(rid, 0, BlockStatus::HOST),
                              RaidenBlockId(rid, 1, BlockStatus::HOST)},
                             true));
  ASSERT_EQ(metadata_or->ValidEntries().size(), 2);

  EXPECT_EQ(KVCacheStoreTest::Evict(store, {"host_1"}), 1);
  EXPECT_THAT(metadata_or->ValidEntries(),
              ElementsAre(::testing::FieldsAre(1, "host_2", 0)));
}

class KVCacheStoreEmbeddedControllerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_server_ = ::tpu_raiden::controller::CreateTestWorkerServer();
    // The DESTINATION worker executes the copy now (the read is a pull), so it
    // needs a transfer manager. Scripting it to succeed is what lets the
    // commit-side logic run at all on CPU, where a real transfer cannot.
    dst_transfer_mock_ = std::make_unique<
        ::tpu_raiden::controller::ShardAwareMockTransferManager>();
    test_server_->service->SetTransferManager(
        ::tpu_raiden::KVManagerHolder(dst_transfer_mock_.get()));
    unit_.set_job_name("test_job");
    unit_.set_job_replica_id("0");
    unit_.set_data_name("test_data");

    // A remote read resolves the source's controller address through the
    // global registry, so these cases need one even when they exercise nothing
    // else about it.
    registry_service_ =
        std::make_unique<global_registry::GlobalRegistryServiceImpl>();
    grpc::ServerBuilder builder;
    int bound_port = 0;
    builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                             &bound_port);
    builder.RegisterService(registry_service_.get());
    registry_server_ = builder.BuildAndStart();
    registry_address_ = absl::StrCat("localhost:", bound_port);
  }

  void TearDown() override {
    if (registry_server_) {
      registry_server_->Shutdown();
      registry_server_->Wait();
    }
  }

  void RegisterAndInitWorker(
      ::tpu_raiden::controller::RaidenController& controller,
      const std::string& worker_id, const std::string& worker_address) {
    ::tpu_raiden::core::controller::RaidenControllerClient client(
        controller.controller_address());
    auto status = client.RegisterWorker(worker_id, worker_address,
                                        {{worker_address, {}}});
    ASSERT_TRUE(status.ok()) << status.message();
  }

  std::unique_ptr<::tpu_raiden::controller::RaidenController> MakeController(
      int num_blocks = 10, int num_shards = 1, int64_t shard_size_bytes = 512,
      absl::string_view controller_address = "") {
    return *::tpu_raiden::controller::RaidenController::Create(
        unit_, num_blocks, num_shards, shard_size_bytes, controller_address);
  }

  ::tpu_sync::rpc::RaidenIdProto unit_;
  std::unique_ptr<::tpu_raiden::controller::TestWorkerServer> test_server_;
  std::unique_ptr<::tpu_raiden::controller::ShardAwareMockTransferManager>
      dst_transfer_mock_;
  std::unique_ptr<global_registry::GlobalRegistryServiceImpl> registry_service_;
  std::unique_ptr<grpc::Server> registry_server_;
  std::string registry_address_;
};

TEST_F(KVCacheStoreEmbeddedControllerTest, SaveReusesFreedBlocksAfterEvict) {
  // Evicted host blocks return to the free pool even when the directory has
  // nothing left to evict: a full evict empties the directory and deallocates
  // every block, and the save below must be served from those.
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  // A two-block pool, so the first save exhausts it.
  auto controller = MakeController(/*num_blocks=*/2);
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);
  auto* controller_ptr = controller.get();

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(2, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  auto save_and_wait =
      [&store](const std::vector<std::string>& hashes) -> absl::Status {
    absl::Status status = store.Save(hashes);
    if (!status.ok()) return status;
    while (true) {
      auto [done, failed, pending, save_existing,
          save_unregistered] = store.PollSaveStatus();
      if (!failed.empty()) return absl::InternalError("async save failed");
      if (!done.empty()) return absl::OkStatus();
      absl::SleepFor(absl::Milliseconds(10));
    }
  };

  std::vector<std::string> first = {"hash_1", "hash_2"};
  std::vector<RaidenBlockId> first_slices = {
      RaidenBlockId(rid, -1, 0, BlockStatus::HBM),
      RaidenBlockId(rid, -1, 1, BlockStatus::HBM)};
  ASSERT_TRUE(InsertResident(store, first, first_slices, false));
  ASSERT_TRUE(store.Lookup(first).ok());
  ASSERT_TRUE(save_and_wait(first).ok());
  EXPECT_EQ(controller_ptr->block_manager()->num_locked_blocks(), 2);

  // Evict everything: the successful save consumed the pins, so the
  // directory entries are already evictable.
  ASSERT_EQ(KVCacheStoreTest::Evict(store, first), 2);
  EXPECT_EQ(controller_ptr->block_manager()->num_free_blocks(), 2);
  EXPECT_EQ(controller_ptr->block_manager()->num_locked_blocks(), 0);

  // The directory is empty and hash_3 gets pinned, so nothing is evictable:
  // the host block for this save has to come from the freed pool.
  std::vector<std::string> second = {"hash_3"};
  std::vector<RaidenBlockId> second_slices = {
      RaidenBlockId(rid, -1, 0, BlockStatus::HBM)};
  ASSERT_TRUE(InsertResident(store, second, second_slices, false));
  ASSERT_TRUE(store.Lookup(second).ok());
  absl::Status status = save_and_wait(second);
  EXPECT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(controller_ptr->block_manager()->num_locked_blocks(), 1);
}

TEST_F(KVCacheStoreEmbeddedControllerTest, SaveSuccess) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, -1, 0, BlockStatus::HBM),
      RaidenBlockId(rid, -1, 1, BlockStatus::HBM)};

  // Insert them as HBM blocks
  ASSERT_TRUE(InsertResident(store, hashes, slices, false));

  // Pin them
  ASSERT_TRUE(store.Lookup(hashes).ok());

  // Save them
  absl::Status status = store.Save(hashes);
  ASSERT_TRUE(status.ok()) << status.message();

  // Poll for completion
  bool done = false;
  while (!done) {
    auto [save_done, save_failed, save_pending, save_existing,
          save_unregistered] = store.PollSaveStatus();
    if (!save_failed.empty()) {
      FAIL() << "Async Save failed during polling";
    }
    // A local save never produces the remote-only outcomes.
    EXPECT_TRUE(save_existing.empty());
    EXPECT_TRUE(save_unregistered.empty());
    if (!save_done.empty()) {
      EXPECT_THAT(save_done,
                  ::testing::UnorderedElementsAre("hash_1", "hash_2"));
      done = true;
    }
    if (!done) {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }

  // Verify transfer manager was called
  EXPECT_EQ(mock_mgr.d2h_calls, 1);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(0, 1));
  // host_block_ids are allocated starting from 0
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(0, 1));

  // Verify status in store is updated to HOST_AND_HBM
  auto lookup_res = PeekLookup(store, hashes);
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, 0);
  EXPECT_EQ((*lookup_res)[0].second.device_block_id, 0);
  EXPECT_EQ((*lookup_res)[1].second.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ((*lookup_res)[1].second.host_block_id, 1);
  EXPECT_EQ((*lookup_res)[1].second.device_block_id, 1);
}

TEST_F(KVCacheStoreEmbeddedControllerTest, LoadSuccess) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  // Insert as HOST only blocks
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, 0, -1, BlockStatus::HOST),
      RaidenBlockId(rid, 1, -1, BlockStatus::HOST)};

  ASSERT_TRUE(InsertResident(store, hashes, slices, true));

  // Pin them
  ASSERT_TRUE(store.Lookup(hashes).ok());

  // Load them to device block 2 and 3
  absl::Status status = store.Load(hashes, {2, 3});
  ASSERT_TRUE(status.ok()) << status.message();

  // Poll for completion
  bool done = false;
  while (!done) {
    auto [load_done, load_failed, load_pending] = store.PollLoadStatus();
    if (!load_failed.empty()) {
      FAIL() << "Async Load failed during polling";
    }
    if (!load_done.empty()) {
      EXPECT_THAT(load_done,
                  ::testing::UnorderedElementsAre("hash_1", "hash_2"));
      done = true;
    }
    if (!done) {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }

  // Verify transfer manager was called
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_calls, 1);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(0, 1));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(2, 3));

  // Verify status in store is updated to HOST_AND_HBM
  auto lookup_res = PeekLookup(store, hashes);
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, 0);
  EXPECT_EQ((*lookup_res)[0].second.device_block_id, 2);
  EXPECT_EQ((*lookup_res)[1].second.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ((*lookup_res)[1].second.host_block_id, 1);
  EXPECT_EQ((*lookup_res)[1].second.device_block_id, 3);
}

TEST_F(KVCacheStoreEmbeddedControllerTest, LoadWithSlicesSuccess) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, 0, -1, BlockStatus::HOST),
      RaidenBlockId(rid, 1, -1, BlockStatus::HOST)};

  ASSERT_TRUE(InsertResident(store, hashes, slices, true));
  ASSERT_TRUE(store.Lookup(hashes).ok());

  absl::Status status = store.Load(hashes, slices, {2, 3});
  ASSERT_TRUE(status.ok()) << status.message();

  bool done = false;
  while (!done) {
    auto [load_done, load_failed, load_pending] = store.PollLoadStatus();
    if (!load_failed.empty()) {
      FAIL() << "Async Load failed during polling";
    }
    if (!load_done.empty()) {
      EXPECT_THAT(load_done,
                  ::testing::UnorderedElementsAre("hash_1", "hash_2"));
      done = true;
    }
    if (!done) {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }

  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_calls, 1);
  EXPECT_THAT(mock_mgr.last_src_offsets, ::testing::ElementsAre(0, 1));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ::testing::ElementsAre(2, 3));
}

TEST_F(KVCacheStoreEmbeddedControllerTest, LoadWithSlicesSizeMismatch) {
  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, 0, -1, BlockStatus::HOST)};

  absl::Status status = store.Load(hashes, slices, {2, 3});
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_THAT(std::string(status.message()), ::testing::HasSubstr("mismatch"));
}

// The slices form used to accept an unpinned local block, while the no-slices
// form required a pin -- one signature, two pin contracts. It requires the pin
// now, for the same reason the other form always did: a successful local load
// CONSUMES one, so there has to be one to consume.
TEST_F(KVCacheStoreEmbeddedControllerTest, LoadWithSlicesUnpinnedFails) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, 0, -1, BlockStatus::HOST)};

  ASSERT_TRUE(InsertResident(store, hashes, slices, true));

  absl::Status status = store.Load(hashes, slices, {2});
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("not pinned"));

  // With the pin the caller was supposed to hold, it goes through -- and a
  // successful load CONSUMES that pin, same as the no-slices form.
  ASSERT_TRUE(store.Lookup(hashes).ok());
  ASSERT_TRUE(store.Load(hashes, slices, {2}).ok());
  bool done = false;
  for (int attempt = 0; attempt < 100 && !done; ++attempt) {
    auto [load_done, load_failed, load_pending] = store.PollLoadStatus();
    ASSERT_TRUE(load_failed.empty());
    if (!load_done.empty()) done = true;
    if (!done) absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(done);
  EXPECT_EQ(store.GetPinCount("hash_1"), 0)
      << "a successful slices-form local load must consume the caller's pin";
}

// The local half of the same rule. Note the release is sequenced AFTER the
// registry write-through where there is one -- releasing inline would let the
// entry be evicted while the queued Register is still pending, publishing a
// host block id this node had already freed. This store has no registry, so it
// exercises the inline branch; the ordering itself is asserted by the e2e
// suites, which do have one.
TEST_F(KVCacheStoreEmbeddedControllerTest, LocalSaveConsumesTheCallerPin) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller = MakeController(10, 1, 512, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, -1, 3, BlockStatus::HBM)};
  ASSERT_TRUE(InsertResident(store, hashes, slices, /*on_host=*/false));
  ASSERT_TRUE(store.Lookup(hashes).ok());
  ASSERT_EQ(store.GetPinCount("hash_1"), 1);

  ASSERT_OK(store.Save(hashes));

  bool done = false;
  for (int attempt = 0; attempt < 200 && !done; ++attempt) {
    auto [save_done, save_failed, pending, existing, unregistered] =
        store.PollSaveStatus();
    ASSERT_TRUE(save_failed.empty());
    // A local save never produces the remote-only outcomes.
    EXPECT_TRUE(existing.empty());
    EXPECT_TRUE(unregistered.empty());
    if (!save_done.empty()) done = true;
    if (!done) absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(done);

  // The release can land on the write-through pool, so give it a moment.
  for (int attempt = 0; attempt < 200 && store.GetPinCount("hash_1") > 0;
       ++attempt) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_EQ(store.GetPinCount("hash_1"), 0)
      << "a successful local save must consume the caller's pin";

  // The entry survives the unpin, carrying the save's result.
  auto after = PeekLookup(store, hashes);
  ASSERT_TRUE(after.ok());
  ASSERT_EQ(after->size(), 1);
  EXPECT_EQ((*after)[0].second.status, BlockStatus::HOST_AND_HBM);
}

// A successful local load consumes the caller's pin, so the caller never
// releases after a load. A failed one does not: giving up is the caller's
// decision, and an entry silently unpinned under a retry would be evictable
// while the caller still believed it held it.
TEST_F(KVCacheStoreEmbeddedControllerTest, LocalLoadConsumesTheCallerPin) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller = MakeController(10, 1, 512, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, 0, -1, BlockStatus::HOST)};
  ASSERT_TRUE(InsertResident(store, hashes, slices, true));
  ASSERT_TRUE(store.Lookup(hashes).ok());
  ASSERT_EQ(store.GetPinCount("hash_1"), 1);

  ASSERT_OK(store.Load(hashes, {2}));

  bool done = false;
  for (int attempt = 0; attempt < 100 && !done; ++attempt) {
    auto [load_done, load_failed, load_pending] = store.PollLoadStatus();
    ASSERT_TRUE(load_failed.empty());
    if (!load_done.empty()) done = true;
    if (!done) absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(done);

  EXPECT_EQ(store.GetPinCount("hash_1"), 0)
      << "a successful local load must consume the caller's pin";
  // The entry survives the unpin and carries the load's result.
  auto after = PeekLookup(store, hashes);
  ASSERT_TRUE(after.ok());
  ASSERT_EQ(after->size(), 1);
  EXPECT_EQ((*after)[0].second.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ((*after)[0].second.device_block_id, 2);
}

// The failure half of the same contract: a FAILED local load does not spend
// the pin. Giving up is the caller's decision; an entry silently unpinned
// under a retry would be evictable while the caller still believed it held
// it.
TEST_F(KVCacheStoreEmbeddedControllerTest, FailedLocalLoadKeepsTheCallerPin) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  mock_mgr.fail_transfers = true;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller = MakeController(10, 1, 512, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, 0, -1, BlockStatus::HOST)};
  ASSERT_TRUE(InsertResident(store, hashes, slices, true));
  ASSERT_TRUE(store.Lookup(hashes).ok());
  ASSERT_EQ(store.GetPinCount("hash_1"), 1);

  ASSERT_OK(store.Load(hashes, {2}));

  bool failed = false;
  for (int attempt = 0; attempt < 100 && !failed; ++attempt) {
    auto [load_done, load_failed, load_pending] = store.PollLoadStatus();
    ASSERT_TRUE(load_done.empty());
    if (!load_failed.empty()) {
      EXPECT_THAT(load_failed, ::testing::ElementsAre("hash_1"));
      failed = true;
    }
    if (!failed) absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(failed);

  EXPECT_EQ(store.GetPinCount("hash_1"), 1)
      << "a failed load must leave the caller's pin so a retry can hold on";
  // The entry itself is untouched: still HOST, still the same block.
  auto after = PeekLookup(store, hashes);
  ASSERT_TRUE(after.ok());
  ASSERT_EQ(after->size(), 1);
  EXPECT_EQ((*after)[0].second.status, BlockStatus::HOST);
  // The caller decides: retry (the pin is still good) or give up.
  store.Release(hashes);
  EXPECT_EQ(store.GetPinCount("hash_1"), 0);
}

// The failure half for the local save: pin retained, and the host blocks the
// save allocated for its destination go back to the pool rather than leaking
// one block per failed save.
TEST_F(KVCacheStoreEmbeddedControllerTest,
       FailedLocalSaveKeepsPinAndFreesHostBlocks) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  mock_mgr.fail_transfers = true;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller = MakeController(10, 1, 512, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");
  auto* controller_ptr = KVCacheStoreTest::GetController(store);
  ASSERT_NE(controller_ptr, nullptr);

  std::vector<std::string> hashes = {"hash_1"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, -1, 3, BlockStatus::HBM)};
  ASSERT_TRUE(InsertResident(store, hashes, slices, /*on_host=*/false));
  ASSERT_TRUE(store.Lookup(hashes).ok());
  ASSERT_EQ(store.GetPinCount("hash_1"), 1);

  const size_t free_before = controller_ptr->block_manager()->num_free_blocks();
  ASSERT_OK(store.Save(hashes));

  bool failed = false;
  for (int attempt = 0; attempt < 200 && !failed; ++attempt) {
    auto [save_done, save_failed, pending, existing, unregistered] =
        store.PollSaveStatus();
    ASSERT_TRUE(save_done.empty());
    if (!save_failed.empty()) {
      EXPECT_THAT(save_failed, ::testing::ElementsAre("hash_1"));
      failed = true;
    }
    if (!failed) absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(failed);

  EXPECT_EQ(store.GetPinCount("hash_1"), 1)
      << "a failed save must leave the caller's pin";
  EXPECT_EQ(controller_ptr->block_manager()->num_free_blocks(), free_before)
      << "the host blocks a failed save allocated must return to the pool";
  // Still an HBM-only entry: the failed save recorded no host residency.
  auto after = PeekLookup(store, hashes);
  ASSERT_TRUE(after.ok());
  ASSERT_EQ(after->size(), 1);
  EXPECT_EQ((*after)[0].second.status, BlockStatus::HBM);
  store.Release(hashes);
}

// The no-slices load's pin gate, hit directly: resident but unpinned is
// refused. Only the absent-hash flavor was covered before, which conflates
// "no entry" with "no pin".
TEST_F(KVCacheStoreEmbeddedControllerTest, UnpinnedLoadIsRefused) {
  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, 0, -1, BlockStatus::HOST)};
  ASSERT_TRUE(InsertResident(store, hashes, slices, true));
  ASSERT_EQ(store.GetPinCount("hash_1"), 0);

  absl::Status status = store.Load(hashes, {2});
  EXPECT_TRUE(absl::IsFailedPrecondition(status)) << status;
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("not pinned"));
}

// The local save's own preconditions, hit directly: no pin, and no HBM
// residency. Both gates carry the pin algebra -- a save that accepted an
// unpinned block would have nothing to consume on success.
TEST_F(KVCacheStoreEmbeddedControllerTest, UnpinnedSaveIsRefused) {
  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, -1, 3, BlockStatus::HBM)};
  ASSERT_TRUE(InsertResident(store, hashes, slices, /*on_host=*/false));
  ASSERT_EQ(store.GetPinCount("hash_1"), 0);

  absl::Status status = store.Save(hashes);
  EXPECT_TRUE(absl::IsFailedPrecondition(status)) << status;
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("not pinned"));
}

TEST_F(KVCacheStoreEmbeddedControllerTest, SaveRequiresHbmResidency) {
  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  // Host-resident and pinned -- everything a save needs except the bytes in
  // HBM to save from.
  std::vector<std::string> hashes = {"hash_1"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, 0, -1, BlockStatus::HOST)};
  ASSERT_TRUE(store.Insert(hashes, slices, /*on_host=*/true).ok());

  absl::Status status = store.Save(hashes);
  EXPECT_TRUE(absl::IsFailedPrecondition(status)) << status;
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("not in HBM"));
  // The refused save consumed nothing.
  EXPECT_EQ(store.GetPinCount("hash_1"), 1);
  store.Release(hashes);
}

// One call is one source, and a peer source is ONE peer: slices naming two
// different owners are refused before anything moves.
TEST_F(KVCacheStoreEmbeddedControllerTest, LoadWithSlicesTwoPeersFails) {
  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  RaidenId peer_a{"peer_a", "0", "kv", 0};
  RaidenId peer_b{"peer_b", "0", "kv", 0};
  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(peer_a, 7, BlockStatus::REMOTE),
      RaidenBlockId(peer_b, 9, BlockStatus::REMOTE)};

  absl::Status status = store.Load(hashes, slices, {2, 3});
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("Mixed remote node IDs"));
}

TEST_F(KVCacheStoreEmbeddedControllerTest, LoadWithSlicesAlreadyLoadingFails) {
  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, 0, -1, BlockStatus::HOST)};

  ASSERT_TRUE(InsertResident(store, hashes, slices, true));
  ASSERT_TRUE(store.Lookup(hashes).ok());

  absl::Status status1 = store.Load(hashes, slices, {2});
  ASSERT_TRUE(status1.ok());

  absl::Status status2 = store.Load(hashes, slices, {3});
  EXPECT_TRUE(absl::IsFailedPrecondition(status2));
  EXPECT_THAT(std::string(status2.message()),
              ::testing::HasSubstr("Block is already loading"));
}

TEST_F(KVCacheStoreEmbeddedControllerTest, LoadWithSlicesMixedStatusesFails) {
  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  RaidenId remote_rid{"remote_job", "0", "remote_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(remote_rid, 0, -1, BlockStatus::REMOTE),
      RaidenBlockId(rid, 1, -1, BlockStatus::HOST)};

  KVCacheStoreTest::PlantIndexEntry(store, {"hash_1"}, {slices[0]},
                                    /*on_host=*/false);
  ASSERT_TRUE(InsertResident(store, {"hash_2"}, {slices[1]}, /*on_host=*/true));
  ASSERT_TRUE(store.Lookup(hashes).ok());

  absl::Status status = store.Load(hashes, slices, {2, 3});
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("Mixed block statuses"));
}

TEST_F(KVCacheStoreEmbeddedControllerTest, LoadWithSlicesRemoteSuccess) {
  auto registry_server = global_registry::CreateTestGlobalRegistryServer();
  std::string registry_address = registry_server->server_address;

  RaidenId local_rid{"local_job", "0", "local_cache", 0};
  RaidenId remote_rid{"remote_job", "0", "remote_cache", 0};

  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  BackendConfig remote_config;
  remote_config.type = "HostOffloadBackend";
  remote_config.capacity = 100;
  remote_config.global_registry_address = registry_address;
  remote_config.raiden_id = remote_rid;

  auto remote_backend_or =
      HostOffloadBackend::Create(remote_config, controller.get());
  ASSERT_OK(remote_backend_or.status());
  auto remote_backend =
      std::dynamic_pointer_cast<HostOffloadBackend>(*remote_backend_or);
  ASSERT_NE(remote_backend, nullptr);

  std::vector<RaidenBlockId> remote_slices = {
      RaidenBlockId(remote_rid, 42, BlockStatus::HOST),
  };
  remote_backend->Insert({"load_remote_hash_1"}, remote_slices,
                         /*on_host=*/true);

  auto remote_server = KVCacheStoreServer::Create();
  ASSERT_OK(remote_server->StartServer(remote_backend.get(), controller.get(),
                                       "127.0.0.1"));

  auto channel =
      grpc::CreateChannel(registry_address, grpc::InsecureChannelCredentials());
  auto registry_client =
      std::make_shared<global_registry::GlobalRegistryClient>(channel);
  ASSERT_OK(registry_client->RegisterStore(remote_rid,
                                           remote_server->GetServerAddress(),
                                           controller->controller_address()));

  KVCacheStore store(10, std::move(controller), registry_address, local_rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"load_remote_hash_1"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(remote_rid, 42, BlockStatus::REMOTE)};

  KVCacheStoreTest::PlantIndexEntry(store, hashes, slices,
                                    /*on_host=*/false);
  // No pin: a load from a peer requires none and consumes none.

  absl::Status status = store.Load(hashes, slices, {5});
  ASSERT_TRUE(status.ok()) << status.message();

  bool done = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto [load_done, load_failed, load_pending] = store.PollLoadStatus();
    ASSERT_TRUE(load_failed.empty());
    if (!load_done.empty()) {
      EXPECT_THAT(load_done,
                  ::testing::UnorderedElementsAre("load_remote_hash_1"));
      done = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(done);

  // Nothing is recorded for a peer source, so the entry this test inserted up
  // front is left exactly as it was: still REMOTE, still naming the peer.
  auto lookup_res = PeekLookup(store, hashes);
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::REMOTE);
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, 42);
  EXPECT_EQ((*lookup_res)[0].second.raiden_id, remote_rid);
}

// The flow the API actually serves for a peer source: lookup() resolves the
// hash through the registry and hands back a REMOTE slice, load() takes that
// slice directly. Nothing is inserted before, and -- the point of this case --
// nothing is left after. The local cache is untouched from start to finish.
TEST_F(KVCacheStoreEmbeddedControllerTest, LoadRemoteWithSlicesRecordsNothing) {
  auto registry_server = global_registry::CreateTestGlobalRegistryServer();
  std::string registry_address = registry_server->server_address;

  RaidenId local_rid{"local_job", "0", "local_cache", 0};
  RaidenId remote_rid{"remote_job", "0", "remote_cache", 0};

  auto controller = MakeController(10, 1, 512, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  BackendConfig remote_config;
  remote_config.type = "HostOffloadBackend";
  remote_config.capacity = 100;
  remote_config.global_registry_address = registry_address;
  remote_config.raiden_id = remote_rid;

  auto remote_backend_or =
      HostOffloadBackend::Create(remote_config, controller.get());
  ASSERT_OK(remote_backend_or.status());
  auto remote_backend =
      std::dynamic_pointer_cast<HostOffloadBackend>(*remote_backend_or);
  ASSERT_NE(remote_backend, nullptr);
  remote_backend->Insert({"slice_load_hash"},
                         {RaidenBlockId(remote_rid, 42, BlockStatus::HOST)},
                         /*on_host=*/true);

  auto remote_server = KVCacheStoreServer::Create();
  ASSERT_OK(remote_server->StartServer(remote_backend.get(), controller.get(),
                                       "127.0.0.1"));
  auto channel =
      grpc::CreateChannel(registry_address, grpc::InsecureChannelCredentials());
  auto registry_client =
      std::make_shared<global_registry::GlobalRegistryClient>(channel);
  ASSERT_OK(registry_client->RegisterStore(remote_rid,
                                           remote_server->GetServerAddress(),
                                           controller->controller_address()));
  // And advertise the block: indexing it in the remote backend does not, and
  // this case needs the registry to answer for a hash the store lacks.
  ASSERT_OK(registry_client->Register({{"slice_load_hash", remote_rid, 42}}));

  KVCacheStore store(10, std::move(controller), registry_address, local_rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"slice_load_hash"};

  // The registry answers, but a registry-only hit never enters the local index
  // and is never pinned -- so this load has no caller pin to consume either.
  auto resolved = store.Lookup(hashes, /*enable_global=*/true);
  ASSERT_TRUE(resolved.ok());
  ASSERT_EQ(resolved->size(), 1);
  EXPECT_EQ((*resolved)[0].second.status, BlockStatus::REMOTE);
  EXPECT_TRUE(PeekLookup(store, hashes)->empty())
      << "a registry-only hit must not have entered the local index";

  std::vector<RaidenBlockId> slices = {(*resolved)[0].second};
  ASSERT_OK(store.Load(hashes, slices, {5}));

  bool done = false;
  for (int attempt = 0; attempt < 100 && !done; ++attempt) {
    auto [load_done, load_failed, load_pending] = store.PollLoadStatus();
    ASSERT_TRUE(load_failed.empty());
    if (!load_done.empty()) {
      EXPECT_THAT(load_done, ::testing::UnorderedElementsAre("slice_load_hash"));
      done = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(done);

  // The whole point: the bytes are in device block 5 and the cache is as empty
  // as it was before the load.
  EXPECT_TRUE(PeekLookup(store, hashes)->empty())
      << "a load from a peer must leave no local entry";
}

// The no-slices Load is LOCAL-ONLY: a REMOTE index entry is refused with
// InvalidArgument regardless of pin state -- the entry here is deliberately
// UNPINNED, and the refusal must come before the pin gate, because "wrong
// API" is the answer whether or not a pin exists. The slices overload is the
// only peer-load path.
TEST_F(KVCacheStoreEmbeddedControllerTest, LoadRefusesRemoteBlock) {
  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId local_rid{"local_job", "0", "local_cache", 0};
  RaidenId remote_rid{"remote_job", "0", "remote_cache", 0};
  KVCacheStore store(10, std::move(controller), "", local_rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"planted_remote_hash"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(remote_rid, /*host_block_id=*/-1, /*device_block_id=*/-1,
                    BlockStatus::REMOTE)};
  KVCacheStoreTest::PlantIndexEntry(store, hashes, slices,
                                    /*on_host=*/false);

  absl::Status status = store.Load(hashes, {0});
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_THAT(status.message(), ::testing::HasSubstr("local-only"));
}

TEST_F(KVCacheStoreEmbeddedControllerTest, SaveMultiWorkerSuccess) {
  auto test_server_0 = ::tpu_raiden::controller::CreateTestWorkerServer();
  auto test_server_1 = ::tpu_raiden::controller::CreateTestWorkerServer();

  ::tpu_raiden::controller::MockTransferManager mock_mgr_0;
  ::tpu_raiden::controller::MockTransferManager mock_mgr_1;

  test_server_0->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr_0));
  test_server_1->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr_1));

  auto controller = MakeController();

  RegisterAndInitWorker(*controller, "worker_0", test_server_0->server_address);
  RegisterAndInitWorker(*controller, "worker_1", test_server_1->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, -1, 0, BlockStatus::HBM),
      RaidenBlockId(rid, -1, 1, BlockStatus::HBM)};

  ASSERT_TRUE(InsertResident(store, hashes, slices, false));

  // Pin them
  ASSERT_TRUE(store.Lookup(hashes).ok());

  absl::Status status = store.Save(hashes);
  ASSERT_TRUE(status.ok()) << status.message();

  // Poll for completion
  bool done = false;
  while (!done) {
    auto [save_done, save_failed, save_pending, save_existing,
          save_unregistered] = store.PollSaveStatus();
    if (!save_failed.empty()) {
      FAIL() << "Async Save failed during polling";
    }
    // A local save never produces the remote-only outcomes.
    EXPECT_TRUE(save_existing.empty());
    EXPECT_TRUE(save_unregistered.empty());
    if (!save_done.empty()) {
      EXPECT_THAT(save_done,
                  ::testing::UnorderedElementsAre("hash_1", "hash_2"));
      done = true;
    }
    if (!done) {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }

  EXPECT_EQ(mock_mgr_0.d2h_calls, 1);
  EXPECT_EQ(mock_mgr_0.h2d_calls, 0);
  EXPECT_THAT(mock_mgr_0.last_src_offsets, ElementsAre(0, 1));
  EXPECT_THAT(mock_mgr_0.last_dst_offsets, ElementsAre(0, 1));

  EXPECT_EQ(mock_mgr_1.d2h_calls, 1);
  EXPECT_EQ(mock_mgr_1.h2d_calls, 0);
  EXPECT_THAT(mock_mgr_1.last_src_offsets, ElementsAre(0, 1));
  EXPECT_THAT(mock_mgr_1.last_dst_offsets, ElementsAre(0, 1));

  auto lookup_res = PeekLookup(store, hashes);
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, 0);
  EXPECT_EQ((*lookup_res)[0].second.device_block_id, 0);
}

TEST_F(KVCacheStoreEmbeddedControllerTest, LoadMultiWorkerSuccess) {
  auto test_server_0 = ::tpu_raiden::controller::CreateTestWorkerServer();
  auto test_server_1 = ::tpu_raiden::controller::CreateTestWorkerServer();

  ::tpu_raiden::controller::MockTransferManager mock_mgr_0;
  ::tpu_raiden::controller::MockTransferManager mock_mgr_1;

  test_server_0->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr_0));
  test_server_1->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr_1));

  auto controller = MakeController();

  RegisterAndInitWorker(*controller, "worker_0", test_server_0->server_address);
  RegisterAndInitWorker(*controller, "worker_1", test_server_1->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, 0, -1, BlockStatus::HOST),
      RaidenBlockId(rid, 1, -1, BlockStatus::HOST)};

  ASSERT_TRUE(InsertResident(store, hashes, slices, true));

  // Pin them
  ASSERT_TRUE(store.Lookup(hashes).ok());

  absl::Status status = store.Load(hashes, {2, 3});
  ASSERT_TRUE(status.ok()) << status.message();

  // Poll for completion
  bool done = false;
  while (!done) {
    auto [load_done, load_failed, load_pending] = store.PollLoadStatus();
    if (!load_failed.empty()) {
      FAIL() << "Async Load failed during polling";
    }
    if (!load_done.empty()) {
      EXPECT_THAT(load_done,
                  ::testing::UnorderedElementsAre("hash_1", "hash_2"));
      done = true;
    }
    if (!done) {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }

  EXPECT_EQ(mock_mgr_0.d2h_calls, 0);
  EXPECT_EQ(mock_mgr_0.h2d_calls, 1);
  EXPECT_THAT(mock_mgr_0.last_src_offsets, ElementsAre(0, 1));
  EXPECT_THAT(mock_mgr_0.last_dst_offsets, ElementsAre(2, 3));

  EXPECT_EQ(mock_mgr_1.d2h_calls, 0);
  EXPECT_EQ(mock_mgr_1.h2d_calls, 1);
  EXPECT_THAT(mock_mgr_1.last_src_offsets, ElementsAre(0, 1));
  EXPECT_THAT(mock_mgr_1.last_dst_offsets, ElementsAre(2, 3));

  auto lookup_res = PeekLookup(store, hashes);
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, 0);
  EXPECT_EQ((*lookup_res)[0].second.device_block_id, 2);
}

TEST_F(KVCacheStoreEmbeddedControllerTest, SaveWriteThrough) {
  // 1. Start a local mock registry server
  auto server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = server->server_address;

  // 2. Setup mock transfer manager & controller
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  // 3. Initialize KVCacheStore with the registry server address & controller
  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), server_address, rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, -1, 0, BlockStatus::HBM),
      RaidenBlockId(rid, -1, 1, BlockStatus::HBM)};

  // 4. Insert them as HBM blocks locally and pin them
  ASSERT_TRUE(InsertResident(store, hashes, slices, false));
  ASSERT_TRUE(store.Lookup(hashes).ok());

  // 5. Call Save on the store
  absl::Status status = store.Save(hashes);
  ASSERT_TRUE(status.ok()) << status.message();

  // 6. Poll for completion
  bool done = false;
  while (!done) {
    auto [save_done, save_failed, save_pending, save_existing,
          save_unregistered] = store.PollSaveStatus();
    if (!save_failed.empty()) {
      FAIL() << "Async Save failed during polling";
    }
    // A local save never produces the remote-only outcomes.
    EXPECT_TRUE(save_existing.empty());
    EXPECT_TRUE(save_unregistered.empty());
    if (!save_done.empty()) {
      EXPECT_THAT(save_done,
                  ::testing::UnorderedElementsAre("hash_1", "hash_2"));
      done = true;
    }
    if (!done) {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }

  // 7. Verify registry has been updated (need to poll registry since
  // registration is async)
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);

  bool registered = false;
  std::vector<global_registry::KVBlockMetadata> metadata_results;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto lookup_res = registry_client.Lookup(hashes);
    if (lookup_res.ok() && lookup_res->size() == 2) {
      metadata_results = *std::move(lookup_res);
      registered = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }

  ASSERT_TRUE(registered)
      << "Block hashes were not registered in global registry";

  // Verify the metadata results
  EXPECT_EQ(metadata_results[0].raiden_id().job_name(), rid.job_name);
  EXPECT_EQ(metadata_results[0].raiden_id().job_replica_id(),
            rid.job_replica_id);
  EXPECT_EQ(metadata_results[0].raiden_id().data_name(), rid.data_name);
  EXPECT_EQ(metadata_results[0].raiden_id().data_replica_idx(),
            rid.data_replica_idx);
  EXPECT_EQ(metadata_results[0].block_id(),
            0);  // first host block allocated is 0

  EXPECT_EQ(metadata_results[1].raiden_id().job_name(), rid.job_name);
  EXPECT_EQ(metadata_results[1].raiden_id().job_replica_id(),
            rid.job_replica_id);
  EXPECT_EQ(metadata_results[1].raiden_id().data_name(), rid.data_name);
  EXPECT_EQ(metadata_results[1].raiden_id().data_replica_idx(),
            rid.data_replica_idx);
  EXPECT_EQ(metadata_results[1].block_id(),
            1);  // second host block allocated is 1
}

// A registry that connects and never answers must not drain the host block
// pool. Each outstanding write-through pins its batch, and a pinned block is
// invisible to eviction, so without a bound saves keep adding never-settling
// calls until Evict finds nothing and Save fails ResourceExhausted. Past the
// bound the advertisement is dropped instead: invisible to peers, which is
// recoverable, rather than unevictable, which is not.
TEST_F(KVCacheStoreEmbeddedControllerTest,
       WriteThroughInFlightCapSurvivesStalledRegistry) {
  auto impl = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  auto stalling_service =
      std::make_unique<global_registry::StallingRegistryService>(
          std::move(impl));
  auto* service_ptr = stalling_service.get();
  // Only Register stalls, so the store still registers itself at construction.
  service_ptr->EnableStall();
  auto registry = global_registry::CreateTestGlobalRegistryServerWithService(
      std::move(stalling_service));

  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));
  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  constexpr int kCapacity = 10;
  constexpr size_t kCap = 2;
  constexpr int kSaves = 6;

  {
    KVCacheStore store(kCapacity, std::move(controller),
                       registry->server_address, rid, std::nullopt,
                       /*store_server_ip=*/"127.0.0.1");
    // Two blocks rather than the production default of a few thousand, so the
    // bound is reachable without saving thousands of blocks.
    store.SetMaxInFlightWriteThroughBlocksForTesting(kCap);

    std::vector<std::string> saved;
    for (int i = 0; i < kSaves; ++i) {
      const std::string hash = absl::StrCat("capped_hash_", i);
      std::vector<std::string> batch = {hash};
      std::vector<RaidenBlockId> slices = {
          RaidenBlockId(rid, -1, i, BlockStatus::HBM)};
      ASSERT_TRUE(InsertResident(store, batch, slices, false));
      // Lookup takes the pin that Save consumes.
      ASSERT_TRUE(store.Lookup(batch).ok());

      // The pool has not drained, which is the failure this bound prevents.
      absl::Status status = store.Save(batch);
      ASSERT_TRUE(status.ok())
          << "save " << i << " failed: " << status.ToString();
      ASSERT_FALSE(absl::IsResourceExhausted(status));
      saved.push_back(hash);

      bool done = false;
      absl::Time deadline = absl::Now() + absl::Seconds(10);
      while (!done && absl::Now() < deadline) {
        auto [save_done, save_failed, save_pending, save_existing,
              save_unregistered] = store.PollSaveStatus();
        ASSERT_TRUE(save_failed.empty()) << "save " << i << " reported failed";
        if (!save_done.empty()) done = true;
        if (!done) absl::SleepFor(absl::Milliseconds(5));
      }
      ASSERT_TRUE(done) << "save " << i << " never completed";
    }

    // The registry is holding the calls that were admitted, and only those.
    service_ptr->WaitForStall();
    EXPECT_EQ(store.InFlightWriteThroughBlocksForTesting(), kCap)
        << "the outstanding write-throughs are not bounded";

    // Everything past the bound had its pin released, so eviction can still
    // make progress -- which is the whole point.
    auto evictable = store.backend()->GetEvictableKeys(kSaves);
    EXPECT_EQ(evictable.size(), kSaves - kCap)
        << "a stalled registry made " << kSaves - evictable.size()
        << " of " << kSaves << " saved blocks unevictable";
    for (const auto& hash : evictable) {
      EXPECT_EQ(store.backend()->GetPinCount(hash), 0);
    }
    EXPECT_FALSE(store.backend()->Evict(evictable).empty())
        << "Evict found nothing to free while the registry was stalled";

    // Letting the registry answer releases the rest.
    service_ptr->ReleaseStall();
    absl::Time deadline = absl::Now() + absl::Seconds(10);
    while (store.InFlightWriteThroughBlocksForTesting() != 0 &&
           absl::Now() < deadline) {
      absl::SleepFor(absl::Milliseconds(10));
    }
    EXPECT_EQ(store.InFlightWriteThroughBlocksForTesting(), 0);

    // The admitted ones did reach the registry; the dropped ones did not, and
    // that is the trade the bound makes.
    int published = 0;
    for (const auto& hash : saved) {
      auto looked_up = registry->client->Lookup({hash});
      if (looked_up.ok() && !looked_up->empty()) ++published;
    }
    EXPECT_EQ(published, kCap)
        << "expected exactly the admitted write-throughs to be advertised";
  }
}

// Evicting a hash erases the entry, unlocks its host block, and
// unregisters it globally -- identically for a HOST_AND_HBM entry and a
// HOST-only one, which is the point of running both statuses through one
// body.
TEST_F(KVCacheStoreEmbeddedControllerTest, EvictByHashesErasesEitherStatus) {
  struct Case {
    BlockStatus status;
    int device_block_id_0;
    int device_block_id_1;
  };
  for (const Case& c : {Case{BlockStatus::HOST_AND_HBM, 0, 1},
                        Case{BlockStatus::HOST, -1, -1}}) {
    SCOPED_TRACE(static_cast<int>(c.status));

    auto server = global_registry::CreateTestGlobalRegistryServer();
    std::string server_address = server->server_address;

    ::tpu_raiden::controller::MockTransferManager mock_mgr;
    test_server_->service->SetTransferManager(
        ::tpu_raiden::KVManagerHolder(&mock_mgr));

    auto controller = MakeController();
    RegisterAndInitWorker(*controller, "worker_0",
                          test_server_->server_address);

    auto alloc_or = controller->AllocateBlockIds(2);
    ASSERT_TRUE(alloc_or.ok());
    std::vector<int> host_block_ids = *alloc_or;
    ASSERT_EQ(host_block_ids.size(), 2);

    RaidenId rid{"test_job", "0", "test_cache", 0};
    KVCacheStore store(10, std::move(controller), server_address, rid,
                       std::nullopt, /*store_server_ip=*/"127.0.0.1");

    // Register in the global registry first to simulate write-through.
    auto channel =
        grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
    global_registry::GlobalRegistryClient registry_client(channel);
    ASSERT_TRUE(registry_client
                    .Register({{"hash_1", rid, host_block_ids[0]},
                               {"hash_2", rid, host_block_ids[1]}})
                    .ok());

    std::vector<std::string> hashes = {"hash_1", "hash_2"};
    std::vector<RaidenBlockId> slices = {
        RaidenBlockId(rid, host_block_ids[0], c.device_block_id_0, c.status),
        RaidenBlockId(rid, host_block_ids[1], c.device_block_id_1, c.status)};
    ASSERT_TRUE(InsertResident(store, hashes, slices, true));

    auto* controller_ptr = KVCacheStoreTest::GetController(store);
    ASSERT_NE(controller_ptr, nullptr);
    EXPECT_EQ(controller_ptr->block_manager()->num_locked_blocks(), 2);

    // Evict "hash_1".
    size_t evicted = KVCacheStoreTest::Evict(store, {"hash_1"});
    EXPECT_EQ(evicted, 1);

    // "hash_1" is erased whole (Lookup stops at the first miss) and "hash_2"
    // is untouched.
    EXPECT_EQ(PeekLookup(store, {"hash_1", "hash_2"})->size(), 0);
    auto lookup_res = PeekLookup(store, {"hash_2"});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 1);
    EXPECT_EQ((*lookup_res)[0].first, "hash_2");
    EXPECT_EQ((*lookup_res)[0].second.status, c.status);
    EXPECT_EQ((*lookup_res)[0].second.host_block_id, host_block_ids[1]);
    EXPECT_EQ((*lookup_res)[0].second.device_block_id, c.device_block_id_1);

    // The evicted hash's host block is unlocked...
    EXPECT_EQ(controller_ptr->block_manager()->num_locked_blocks(), 1);

    // ...and the registry unregisters it (write-through is async; poll).
    bool unregistered = false;
    for (int attempt = 0; attempt < 100; ++attempt) {
      auto reg_res = registry_client.Lookup({"hash_1"});
      if (reg_res.ok() && reg_res->empty()) {
        unregistered = true;
        break;
      }
      absl::SleepFor(absl::Milliseconds(10));
    }
    EXPECT_TRUE(unregistered);
  }
}

// Verifies that eviction skips pinned blocks and leaves them registered in the
// global registry, while unregistering only the actually evicted blocks.
TEST_F(KVCacheStoreEmbeddedControllerTest, EvictKeepsASkippedHashRegistered) {
  auto server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = server->server_address;

  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0",
                        test_server_->server_address);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<int> host_block_ids,
                       controller->AllocateBlockIds(2));
  ASSERT_EQ(host_block_ids.size(), 2);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), server_address, rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);
  ASSERT_OK(registry_client.Register({{"pinned_1", rid, host_block_ids[0]},
                                      {"free_1", rid, host_block_ids[1]}}));

  std::vector<std::string> hashes = {"pinned_1", "free_1"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, host_block_ids[0], -1, BlockStatus::HOST),
      RaidenBlockId(rid, host_block_ids[1], -1, BlockStatus::HOST)};
  ASSERT_TRUE(InsertResident(store, hashes, slices, true));

  // The pin is what makes "pinned_1" unevictable.
  ASSERT_OK(store.Lookup({"pinned_1"}));
  ASSERT_EQ(store.GetPinCount("pinned_1"), 1);

  // Ask for both. Only the unpinned one can go.
  EXPECT_EQ(KVCacheStoreTest::Evict(store, {"pinned_1", "free_1"}), 1);
  EXPECT_EQ(store.GetPinCount("pinned_1"), 1);

  // Wait for the eviction to reach the registry, then check the survivor:
  // once "free_1" is gone we know the unregister call has been served, so a
  // still-present "pinned_1" is a settled answer rather than a slow one.
  // Note: GlobalRegistryClient communicates with an external gRPC server
  // without client notification/subscription streams, so we poll for
  // unregistration.
  bool free_unregistered = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto reg_res = registry_client.Lookup({"free_1"});
    if (reg_res.ok() && reg_res->empty()) {
      free_unregistered = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_TRUE(free_unregistered);

  TF_ASSERT_OK_AND_ASSIGN(auto pinned_res,
                          registry_client.Lookup({"pinned_1"}));
  EXPECT_FALSE(pinned_res.empty());
}

TEST_F(KVCacheStoreEmbeddedControllerTest, EvictOnSave) {
  auto server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = server->server_address;

  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller = MakeController(/*num_blocks=*/2);
  auto* controller_ptr = controller.get();
  RegisterAndInitWorker(*controller_ptr, "worker_0",
                        test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(3, std::move(controller), server_address, rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<int> host_block_ids,
                       controller_ptr->AllocateBlockIds(2));
  ASSERT_EQ(host_block_ids.size(), 2);

  ASSERT_OK(registry_client.Register({{"block_A", rid, host_block_ids[0]},
                                      {"block_B", rid, host_block_ids[1]}}));

  std::vector<std::string> hashes = {"block_A", "block_B"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, host_block_ids[0], -1, BlockStatus::HOST),
      RaidenBlockId(rid, host_block_ids[1], -1, BlockStatus::HOST)};
  ASSERT_TRUE(InsertResident(store, hashes, slices, true));

  std::vector<std::string> hashes_C = {"block_C"};
  std::vector<RaidenBlockId> slices_C = {
      RaidenBlockId(rid, -1, 0, BlockStatus::HBM)};
  ASSERT_TRUE(InsertResident(store, hashes_C, slices_C, false));
  ASSERT_OK(store.Lookup(hashes_C));

  EXPECT_EQ(controller_ptr->block_manager()->num_free_blocks(), 0);

  ASSERT_OK(store.Save(hashes_C));

  bool done = false;
  while (!done) {
    auto [save_done, save_failed, save_pending, save_existing,
          save_unregistered] = store.PollSaveStatus();
    if (!save_failed.empty()) {
      FAIL() << "Async Save failed during polling";
    }
    if (!save_done.empty()) {
      EXPECT_THAT(save_done, ::testing::ElementsAre("block_C"));
      done = true;
    }
    if (!done) {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }

  // The save needed a host block and there were none free, so it evicted --
  // block_B, the TAIL of the batch, because releasing a batch walks it in
  // reverse and each unpin promotes to MRU. block_A survives.
  EXPECT_EQ(PeekLookup(store, {"block_B"})->size(), 0);
  EXPECT_EQ(PeekLookup(store, {"block_A"})->size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(auto lookup_res, PeekLookup(store, {"block_C"}));
  ASSERT_EQ(lookup_res.size(), 1);
  EXPECT_EQ(lookup_res[0].second.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ(lookup_res[0].second.host_block_id, host_block_ids[1]);
  EXPECT_EQ(lookup_res[0].second.device_block_id, 0);

  // Evicting it locally must also unpublish it, or peers keep being told this
  // node holds a block it has handed back.
  bool unregistered_B = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto lookup_res = registry_client.Lookup({"block_B"});
    if (lookup_res.ok() && lookup_res->empty()) {
      unregistered_B = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_TRUE(unregistered_B);
}

TEST_F(KVCacheStoreEmbeddedControllerTest, ProactiveEvictionWithCandidates) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  // Capacity is 2
  auto controller = MakeController(/*num_blocks=*/2);
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(2, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_A", "hash_B"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, -1, 0, BlockStatus::HBM),
      RaidenBlockId(rid, -1, 1, BlockStatus::HBM)};

  // 1. Insert A and B as HBM blocks
  ASSERT_TRUE(InsertResident(store, hashes, slices, false));

  // 2. Save A and B (allocates host blocks for both)
  ASSERT_TRUE(store.Lookup(hashes).ok());
  ASSERT_TRUE(store.Save(hashes).ok());

  // Poll for Save completion
  bool save_done = false;
  while (!save_done) {
    auto [done, failed, pending, save_existing,
          save_unregistered] = store.PollSaveStatus();
    ASSERT_TRUE(failed.empty());
    if (!done.empty()) {
      save_done = true;
    } else {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }
  // No release: the successful save consumed the pins.

  // Verify both are HOST_AND_HBM. Observed rather than looked up: this case is
  // about which block evicts first, and a pin/unpin round trip moves a node
  // through the pinned list and back, which reorders the very LRU under test.
  {
    auto lookup_res = PeekLookup(store, {"hash_B", "hash_A"});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 2);
    EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST_AND_HBM);
    EXPECT_EQ((*lookup_res)[1].second.status, BlockStatus::HOST_AND_HBM);
  }

  // Active LRU: A, B (A is MRU, B is LRU).

  // 3. Insert C (HBM block). This exceeds store capacity (2) and evicts B.
  std::vector<std::string> hash_C = {"hash_C"};
  std::vector<RaidenBlockId> slice_C = {
      RaidenBlockId(rid, -1, 2, BlockStatus::HBM)};
  ASSERT_TRUE(InsertResident(store, hash_C, slice_C, false));

  // B should now be in candidates.
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::ElementsAre("hash_B"));

  // Active LRU: C, A (C is MRU, A is LRU).
  // 5. Insert D (HBM block). This exceeds capacity and evicts A (since A is
  // LRU).
  std::vector<std::string> hash_D = {"hash_D"};
  std::vector<RaidenBlockId> slice_D = {
      RaidenBlockId(rid, -1, 3, BlockStatus::HBM)};
  ASSERT_TRUE(InsertResident(store, hash_D, slice_D, false));

  // Candidates list should now contain B, then A.
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::ElementsAre("hash_B", "hash_A"));

  // 6. Save D. Requires 1 host block.
  // Controller free host blocks: 0 (used by A and B).
  // It should pick candidate B for eviction and deallocate its host block.
  // A (candidate HOST_AND_HBM) should not be affected.
  ASSERT_TRUE(store.Lookup(hash_D).ok());
  ASSERT_TRUE(store.Save(hash_D).ok());

  // Poll for Save completion
  save_done = false;
  while (!save_done) {
    auto [done, failed, pending, save_existing,
          save_unregistered] = store.PollSaveStatus();
    ASSERT_TRUE(failed.empty());
    if (!done.empty()) {
      save_done = true;
    } else {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }
  // No release: the successful save consumed the pin the Lookup granted.

  // 7. Verify states:
  // - B should be erased (since it was HOST_AND_HBM and got evicted)
  // - A should remain in candidates (HOST_AND_HBM)
  // - D should be HOST_AND_HBM
  {
    auto lookup_res = PeekLookup(store, {"hash_B"});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 0);
  }
  {
    auto lookup_res = PeekLookup(store, {"hash_D"});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 1);
    EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST_AND_HBM);
    EXPECT_NE((*lookup_res)[0].second.host_block_id, -1);
  }
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::ElementsAre("hash_A"));
}

TEST_F(KVCacheStoreEmbeddedControllerTest, ReadRemoteSuccess) {
  auto registry_server = global_registry::CreateTestGlobalRegistryServer();
  std::string registry_address = registry_server->server_address;

  kv_cache::RaidenId src_raiden_id{"src_job", "0", "src_data", 0};
  RaidenId rid{"dst_job", "0", "dst_cache", 0};

  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  BackendConfig src_config;
  src_config.type = "HostOffloadBackend";
  src_config.capacity = 100;
  src_config.global_registry_address = registry_address;
  src_config.raiden_id = src_raiden_id;

  auto src_backend_or =
      HostOffloadBackend::Create(src_config, controller.get());
  ASSERT_OK(src_backend_or.status());
  auto src_backend =
      std::dynamic_pointer_cast<HostOffloadBackend>(*src_backend_or);
  ASSERT_NE(src_backend, nullptr);

  std::vector<RaidenBlockId> src_slices = {
      RaidenBlockId(src_raiden_id, 42, BlockStatus::HOST),
  };
  src_backend->Insert({"hash_0"}, src_slices, /*on_host=*/true);

  auto src_store_server = KVCacheStoreServer::Create();
  ASSERT_OK(src_store_server->StartServer(src_backend.get(), controller.get(),
                                          "127.0.0.1"));

  auto channel =
      grpc::CreateChannel(registry_address, grpc::InsecureChannelCredentials());
  auto registry_client =
      std::make_shared<global_registry::GlobalRegistryClient>(channel);
  ASSERT_OK(registry_client->RegisterStore(src_raiden_id,
                                           src_store_server->GetServerAddress(),
                                           controller->controller_address()));

  KVCacheStore store(10, std::move(controller), registry_address, rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_0"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(src_raiden_id, 42, BlockStatus::REMOTE)};
  const std::vector<int32_t> device_blocks = {7};

  absl::Status status = store.ReadRemote(hashes, slices, device_blocks);
  ASSERT_TRUE(status.ok()) << status.message();

  // Poll for completion
  bool done = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto [done_hashes, failed_hashes, pending_hashes] =
        store.PollRemoteReadStatus();
    ASSERT_TRUE(failed_hashes.empty());
    if (!done_hashes.empty()) {
      EXPECT_THAT(done_hashes, ::testing::ElementsAre("hash_0"));
      done = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(done);

  // A successful read leaves NO local record: the bytes are in the caller's
  // device block and nowhere else. A later local lookup is still a miss.
  auto lookup_res = store.Lookup(hashes);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_TRUE(lookup_res->empty());

  // ...and nothing is advertised to the registry. There is no host-resident
  // copy here to serve to a peer, so publishing one would advertise a block
  // this node does not have.
  auto registry_lookup = registry_client->Lookup(hashes);
  ASSERT_TRUE(registry_lookup.ok());
  EXPECT_TRUE(registry_lookup->empty())
      << "read_remote must not advertise the read block to the registry";
}

// The host blocks a read stages through are a hop, not a destination. Nothing
// in the LRU points at them, so if the poller did not return them to the pool
// each read would burn one host block permanently. Reading more blocks in
// total than the pool holds only works if every read gives its staging back.
TEST_F(KVCacheStoreEmbeddedControllerTest, ReadRemoteReturnsStagingOnSuccess) {
  auto registry_server = global_registry::CreateTestGlobalRegistryServer();
  std::string registry_address = registry_server->server_address;

  kv_cache::RaidenId src_raiden_id{"src_job", "0", "src_data", 0};
  RaidenId rid{"dst_job", "0", "dst_cache", 0};

  constexpr int kHostBlocks = 4;
  auto dst_controller = MakeController(kHostBlocks);
  RegisterAndInitWorker(*dst_controller, "worker_0",
                        test_server_->server_address);

  BackendConfig src_config;
  src_config.type = "HostOffloadBackend";
  src_config.capacity = 100;
  src_config.global_registry_address = registry_address;
  src_config.raiden_id = src_raiden_id;

  auto src_backend_or =
      HostOffloadBackend::Create(src_config, dst_controller.get());
  ASSERT_OK(src_backend_or.status());
  auto src_backend =
      std::dynamic_pointer_cast<HostOffloadBackend>(*src_backend_or);
  ASSERT_NE(src_backend, nullptr);

  std::vector<RaidenBlockId> src_slices = {
      RaidenBlockId(src_raiden_id, 42, BlockStatus::HOST),
      RaidenBlockId(src_raiden_id, 43, BlockStatus::HOST)};
  for (int round = 0; round < 3; ++round) {
    src_backend->Insert({absl::StrCat("hash_", round, "_a"),
                         absl::StrCat("hash_", round, "_b")},
                        src_slices, /*on_host=*/true);
  }

  auto src_store_server = KVCacheStoreServer::Create();
  ASSERT_OK(src_store_server->StartServer(src_backend.get(),
                                          dst_controller.get(), "127.0.0.1"));

  auto channel =
      grpc::CreateChannel(registry_address, grpc::InsecureChannelCredentials());
  auto registry_client =
      std::make_shared<global_registry::GlobalRegistryClient>(channel);
  ASSERT_OK(registry_client->RegisterStore(
      src_raiden_id, src_store_server->GetServerAddress(),
      dst_controller->controller_address()));

  KVCacheStore store(kHostBlocks, std::move(dst_controller), registry_address,
                     rid, std::nullopt, /*store_server_ip=*/"127.0.0.1");

  const std::vector<RaidenBlockId> slices = {
      RaidenBlockId(src_raiden_id, 42, BlockStatus::REMOTE),
      RaidenBlockId(src_raiden_id, 43, BlockStatus::REMOTE)};

  for (int round = 0; round < 3; ++round) {
    std::vector<std::string> hashes = {absl::StrCat("hash_", round, "_a"),
                                       absl::StrCat("hash_", round, "_b")};
    absl::Status status = store.ReadRemote(hashes, slices, {7, 8});
    ASSERT_TRUE(status.ok())
        << "round " << round << " failed to launch: " << status.message()
        << " -- staging blocks from an earlier round were not reclaimed";

    bool done = false;
    for (int attempt = 0; attempt < 200 && !done; ++attempt) {
      auto [done_hashes, failed_hashes, pending] = store.PollRemoteReadStatus();
      ASSERT_TRUE(failed_hashes.empty());
      if (done_hashes.size() == hashes.size()) done = true;
      if (!done) absl::SleepFor(absl::Milliseconds(10));
    }
    ASSERT_TRUE(done) << "round " << round << " never completed";
  }
}

// A remote read needs the global registry to learn where the owning peer's
// controller lives, so a store built without one cannot read remotely at all.
// The error has to say that, or the caller sees an unexplained failure on a
// path that used to work through a separate directory service.
TEST_F(KVCacheStoreEmbeddedControllerTest, ReadRemoteWithoutRegistryFails) {
  auto dst_controller = MakeController();
  RegisterAndInitWorker(*dst_controller, "worker_0",
                        test_server_->server_address);

  RaidenId rid{"dst_job", "0", "dst_cache", 0};
  KVCacheStore store(10, std::move(dst_controller),
                     /*global_registry_address=*/"", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  RaidenId src_raiden_id{"src_job", "0", "src_data", 0};
  std::vector<std::string> hashes = {"hash_0"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(src_raiden_id, 42, BlockStatus::REMOTE)};

  absl::Status status = store.ReadRemote(hashes, slices, {7});
  ASSERT_TRUE(status.ok()) << status.message();

  bool failed = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto [done_hashes, failed_hashes, pending_hashes] =
        store.PollRemoteReadStatus();
    if (!failed_hashes.empty()) {
      EXPECT_THAT(failed_hashes, ::testing::ElementsAre("hash_0"));
      failed = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(failed);
}

// A peer registered by an older binary, or one that never stood up a
// controller, publishes an empty controller address. Dialling it would fail
// inside gRPC with nothing naming the peer, so refuse it here.
TEST_F(KVCacheStoreEmbeddedControllerTest,
       ReadRemotePeerWithoutControllerAddressFails) {
  RaidenId src_raiden_id{"src_job", "0", "src_data", 0};
  ASSERT_OK(PublishPeerController(registry_address_, src_raiden_id,
                                  /*controller_address=*/""));

  auto dst_controller = MakeController();
  RegisterAndInitWorker(*dst_controller, "worker_0",
                        test_server_->server_address);

  RaidenId rid{"dst_job", "0", "dst_cache", 0};
  KVCacheStore store(10, std::move(dst_controller), registry_address_, rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_0"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(src_raiden_id, 42, BlockStatus::REMOTE)};

  absl::Status status = store.ReadRemote(hashes, slices, {7});
  ASSERT_TRUE(status.ok()) << status.message();

  bool failed = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto [done_hashes, failed_hashes, pending_hashes] =
        store.PollRemoteReadStatus();
    if (!failed_hashes.empty()) {
      EXPECT_THAT(failed_hashes, ::testing::ElementsAre("hash_0"));
      failed = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(failed);
}

// The peer's controller address is cached, so a repeat read costs no registry
// round trip -- and every failed read drops it, so a peer that moved is
// re-resolved rather than dialled at its old address forever.
//
// The cost of that design is visible here and is the contract: the FIRST read
// after a peer restarts still fails, against the stale address. The retry is
// what succeeds.
TEST_F(KVCacheStoreEmbeddedControllerTest,
       ReadRemoteCachesPeerControllerAndReresolvesAfterFailure) {
  CountingRegistryService counting_registry;
  grpc::ServerBuilder builder;
  int registry_port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &registry_port);
  builder.RegisterService(&counting_registry);
  auto registry_server = builder.BuildAndStart();
  const std::string registry_address =
      absl::StrCat("localhost:", registry_port);

  RaidenId src_raiden_id{"src_job", "0", "src_data", 0};

  auto dst_controller = MakeController(/*num_blocks=*/20);
  RegisterAndInitWorker(*dst_controller, "worker_0",
                        test_server_->server_address);
  auto* controller_ptr = dst_controller.get();

  BackendConfig src_config;
  src_config.type = "HostOffloadBackend";
  src_config.capacity = 100;
  src_config.global_registry_address = registry_address;
  src_config.raiden_id = src_raiden_id;

  auto src_backend_or = HostOffloadBackend::Create(src_config, controller_ptr);
  ASSERT_OK(src_backend_or.status());
  auto src_backend =
      std::dynamic_pointer_cast<HostOffloadBackend>(*src_backend_or);
  ASSERT_NE(src_backend, nullptr);

  std::vector<RaidenBlockId> src_slices = {
      RaidenBlockId(src_raiden_id, 42, BlockStatus::HOST),
      RaidenBlockId(src_raiden_id, 42, BlockStatus::HOST),
      RaidenBlockId(src_raiden_id, 42, BlockStatus::HOST),
      RaidenBlockId(src_raiden_id, 42, BlockStatus::HOST)};
  src_backend->Insert({"hash_0", "hash_1", "hash_2", "hash_3"}, src_slices,
                      /*on_host=*/true);

  auto old_src = KVCacheStoreServer::Create();
  ASSERT_OK(
      old_src->StartServer(src_backend.get(), controller_ptr, "127.0.0.1"));

  global_registry::GlobalRegistryClient reg_client(grpc::CreateChannel(
      registry_address, grpc::InsecureChannelCredentials()));
  ASSERT_OK(reg_client.RegisterStore(src_raiden_id, old_src->GetServerAddress(),
                                     controller_ptr->controller_address()));

  RaidenId rid{"dst_job", "0", "dst_cache", 0};
  KVCacheStore store(20, std::move(dst_controller), registry_address, rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

  // The store publishing itself resolves nothing; count only what reads do.
  const int resolves_before = counting_registry.resolve_store_calls.load();

  const auto read_and_wait = [&](const std::string& hash) {
    std::vector<std::string> hashes = {hash};
    std::vector<RaidenBlockId> slices = {
        RaidenBlockId(src_raiden_id, 42, BlockStatus::REMOTE)};
    EXPECT_TRUE(store.ReadRemote(hashes, slices, {7}).ok());
    for (int attempt = 0; attempt < 300; ++attempt) {
      auto [done, failed, pending] = store.PollRemoteReadStatus();
      if (!done.empty()) return true;
      if (!failed.empty()) return false;
      absl::SleepFor(absl::Milliseconds(10));
    }
    return false;
  };

  ASSERT_TRUE(read_and_wait("hash_0"));
  const int after_first = counting_registry.resolve_store_calls.load();
  EXPECT_EQ(after_first - resolves_before, 1) << "first read must resolve";

  // Second read, same peer: served from the cache, no registry round trip.
  ASSERT_TRUE(read_and_wait("hash_1"));
  EXPECT_EQ(counting_registry.resolve_store_calls.load(), after_first)
      << "a cached peer must not be re-resolved";

  // The peer restarts: old store server is shut down and new server starts.
  old_src->Shutdown();
  auto new_src = KVCacheStoreServer::Create();
  ASSERT_OK(
      new_src->StartServer(src_backend.get(), controller_ptr, "127.0.0.1"));
  ASSERT_OK(reg_client.RegisterStore(src_raiden_id, new_src->GetServerAddress(),
                                     controller_ptr->controller_address()));

  // The cached address is stale, so this read fails -- and that failure is
  // what evicts it.
  EXPECT_FALSE(read_and_wait("hash_2"));

  // The retry re-resolves and lands on the new store server.
  ASSERT_TRUE(read_and_wait("hash_3"));
  EXPECT_GT(counting_registry.resolve_store_calls.load(), after_first)
      << "a failed read must drop the cached address";

  registry_server->Shutdown();
}

TEST_F(KVCacheStoreEmbeddedControllerTest, ReadRemoteFailure) {
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder registry_builder;
  int registry_port = 0;
  registry_builder.AddListeningPort(
      "localhost:0", grpc::InsecureServerCredentials(), &registry_port);
  registry_builder.RegisterService(service.get());
  auto registry_server = registry_builder.BuildAndStart();
  std::string registry_address = "localhost:" + std::to_string(registry_port);

  auto src_controller_server = core::controller::CreateTestControllerServer();

  ::tpu_sync::rpc::RaidenIdProto src_unit;
  src_unit.set_job_name("src_job");
  src_unit.set_job_replica_id("0");
  src_unit.set_data_name("src_data");
  src_unit.set_data_replica_idx(0);

  kv_cache::RaidenId src_raiden_id;
  src_raiden_id.job_name = "src_job";
  src_raiden_id.job_replica_id = "0";
  src_raiden_id.data_name = "src_data";
  src_raiden_id.data_replica_idx = 0;

  ASSERT_OK(PublishPeerController(registry_address, src_raiden_id,
                                  src_controller_server->server_address));

  auto register_src_worker = [&](const std::string& worker_id,
                                 const std::string& worker_address,
                                 const std::string& transfer_endpoint) {
    auto status = src_controller_server->client->RegisterWorker(
        worker_id, worker_address, {{transfer_endpoint, {}}});
    ASSERT_TRUE(status.ok()) << status.message();
  };
  register_src_worker("worker_0", "src_worker_0_addr", "src_worker_0_transfer");

  // Every read is now validated at the source by construction -- there is no
  // longer any RPC that transfers without verifying and pinning first. Grant
  // the lease and echo back authoritative ids.
  src_controller_server->service->SetReadRemoteHooks(
      [&](absl::Span<const std::string> h)
          -> absl::StatusOr<std::vector<int32_t>> {
        return std::vector<int32_t>(h.size(), 42);
      },
      [&](absl::Span<const std::string> /*h*/) {});
  // NOTE: the source no longer transfers anything. Under the pull
  // design the DESTINATION's own worker (test_server_, backed by a mock
  // transfer manager) executes the copy, and the source only leases.

  auto dst_controller = MakeController();
  RegisterAndInitWorker(*dst_controller, "worker_0",
                        test_server_->server_address);

  RaidenId rid{"dst_job", "0", "dst_cache", 0};
  KVCacheStore store(2, std::move(dst_controller), registry_address, rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_0"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(src_raiden_id, 42, BlockStatus::REMOTE)};

  // The transfer now runs on the DESTINATION, so that is where the failure is
  // injected.
  dst_transfer_mock_->fail_transfers = true;

  absl::Status status = store.ReadRemote(hashes, slices, {7});
  ASSERT_TRUE(status.ok()) << status.message();

  // Poll for failure
  bool failed = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto [done_hashes, failed_hashes, pending_hashes] =
        store.PollRemoteReadStatus();
    ASSERT_TRUE(done_hashes.empty());
    if (!failed_hashes.empty()) {
      EXPECT_THAT(failed_hashes, ::testing::ElementsAre("hash_0"));
      failed = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(failed);

  // A failed read leaves NOTHING to clean up. There is no entry to delete and
  // no candidate to restore -- the read never touched the LRU. This is the
  // whole point of taking the source coordinates as an argument: the old
  // design stranded a REMOTE entry here that only release_and_delete could
  // remove.
  {
    auto lookup_res = store.Lookup(hashes);
    ASSERT_TRUE(lookup_res.ok());
    EXPECT_TRUE(lookup_res->empty());
  }

  // The staging blocks went back to the pool, so the read can be retried. If
  // the failure path leaked them this second launch would be the one to fail.
  dst_transfer_mock_->fail_transfers = false;
  EXPECT_TRUE(store.ReadRemote(hashes, slices, {7}).ok())
      << "a failed read must return its staging blocks";

  registry_server->Shutdown();
}

// ReadRemote step 6a end-to-end at the store level: the source controller's
// verify hook rejects the requested hash -> the destination read fails and its
// pre-allocated host block is reverted (via PollRemoteReadsInternal).
TEST_F(KVCacheStoreEmbeddedControllerTest,
       ReadRemoteSourceVerifyMissingRevertsDestination) {
  RaidenId src_raiden_id{"src_job", "0", "src_data", 0};

  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  BackendConfig src_config;
  src_config.type = "HostOffloadBackend";
  src_config.capacity = 100;
  src_config.global_registry_address = registry_address_;
  src_config.raiden_id = src_raiden_id;

  auto src_backend_or =
      HostOffloadBackend::Create(src_config, controller.get());
  ASSERT_OK(src_backend_or.status());
  auto src_backend =
      std::dynamic_pointer_cast<HostOffloadBackend>(*src_backend_or);
  ASSERT_NE(src_backend, nullptr);

  auto src_store_server = KVCacheStoreServer::Create();
  ASSERT_OK(src_store_server->StartServer(src_backend.get(), controller.get(),
                                          "127.0.0.1"));

  auto channel = grpc::CreateChannel(registry_address_,
                                     grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);
  ASSERT_OK(registry_client.RegisterStore(src_raiden_id,
                                          src_store_server->GetServerAddress(),
                                          controller->controller_address()));

  RaidenId rid{"dst_job", "0", "dst_cache", 0};
  KVCacheStore store(2, std::move(controller), registry_address_, rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_0"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(src_raiden_id, 42, BlockStatus::REMOTE)};

  ASSERT_TRUE(store.ReadRemote(hashes, slices, {7}).ok());

  bool failed = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto [done_hashes, failed_hashes, pending_hashes] =
        store.PollRemoteReadStatus();
    ASSERT_TRUE(done_hashes.empty());
    if (!failed_hashes.empty()) {
      EXPECT_THAT(failed_hashes, ::testing::ElementsAre("hash_0"));
      failed = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(failed);
  // Nothing was recorded locally, on this path as on every other.
  auto lookup_res = PeekLookup(store, hashes);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_TRUE(lookup_res->empty());
}

// The three parallel arrays must agree before anything is allocated.
TEST_F(KVCacheStoreEmbeddedControllerTest, ReadRemoteSizeMismatchFails) {
  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"dst_job", "0", "dst_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  RaidenId peer{"src_job", "0", "src_cache", 0};
  std::vector<std::string> hashes = {"hash_0"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(peer, 42, BlockStatus::REMOTE)};

  // One hash, one slice, two destinations.
  absl::Status status = store.ReadRemote(hashes, slices, {7, 8});
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;

  // One hash, two slices.
  std::vector<RaidenBlockId> two_slices = {
      RaidenBlockId(peer, 42, BlockStatus::REMOTE),
      RaidenBlockId(peer, 43, BlockStatus::REMOTE)};
  status = store.ReadRemote(hashes, two_slices, {7});
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
}

TEST_F(KVCacheStoreEmbeddedControllerTest, ReadRemoteDuplicateFails) {
  auto src_controller_server = core::controller::CreateTestControllerServer();

  ::tpu_sync::rpc::RaidenIdProto src_unit;
  src_unit.set_job_name("src_job");
  src_unit.set_job_replica_id("0");
  src_unit.set_data_name("src_data");
  src_unit.set_data_replica_idx(0);

  kv_cache::RaidenId src_raiden_id;
  src_raiden_id.job_name = "src_job";
  src_raiden_id.job_replica_id = "0";
  src_raiden_id.data_name = "src_data";
  src_raiden_id.data_replica_idx = 0;

  ASSERT_OK(PublishPeerController(registry_address_, src_raiden_id,
                                  src_controller_server->server_address));

  auto register_src_worker = [&](const std::string& worker_id,
                                 const std::string& worker_address,
                                 const std::string& transfer_endpoint) {
    auto status = src_controller_server->client->RegisterWorker(
        worker_id, worker_address, {{transfer_endpoint, {}}});
    ASSERT_TRUE(status.ok()) << status.message();
  };
  register_src_worker("worker_0", "src_worker_0_addr", "src_worker_0_transfer");

  // Every read is now validated at the source by construction -- there is no
  // longer any RPC that transfers without verifying and pinning first. Grant
  // the lease and echo back authoritative ids.
  src_controller_server->service->SetReadRemoteHooks(
      [&](absl::Span<const std::string> h)
          -> absl::StatusOr<std::vector<int32_t>> {
        return std::vector<int32_t>(h.size(), 42);
      },
      [&](absl::Span<const std::string> /*h*/) {});
  // NOTE: the source no longer transfers anything. Under the pull design
  // the DESTINATION's own worker (test_server_, backed by a mock transfer
  // manager) executes the copy; the source only leases.

  auto dst_controller = MakeController();
  RegisterAndInitWorker(*dst_controller, "worker_0",
                        test_server_->server_address);

  RaidenId rid{"dst_job", "0", "dst_cache", 0};
  KVCacheStore store(10, std::move(dst_controller), registry_address_, rid,
                     std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_0"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(src_raiden_id, 42, BlockStatus::REMOTE)};

  // First call succeeds
  absl::Status status1 = store.ReadRemote(hashes, slices, {7});
  ASSERT_TRUE(status1.ok()) << status1.message();

  // Second call fails with FailedPreconditionError
  absl::Status status2 = store.ReadRemote(hashes, slices, {8});
  EXPECT_FALSE(status2.ok());
  EXPECT_EQ(status2.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status2.message(), ::testing::HasSubstr("already loading"));
}


TEST_F(KVCacheStoreEmbeddedControllerTest, ReadRemoteMultipleSources) {
  auto registry_server = global_registry::CreateTestGlobalRegistryServer();
  std::string registry_address = registry_server->server_address;

  kv_cache::RaidenId src_raiden_id_1{"src_job_1", "0", "src_data_1", 0};
  kv_cache::RaidenId src_raiden_id_2{"src_job_2", "0", "src_data_2", 0};

  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  // Source 1
  BackendConfig src_config_1;
  src_config_1.type = "HostOffloadBackend";
  src_config_1.capacity = 100;
  src_config_1.global_registry_address = registry_address;
  src_config_1.raiden_id = src_raiden_id_1;
  auto src_backend_or_1 =
      HostOffloadBackend::Create(src_config_1, controller.get());
  ASSERT_OK(src_backend_or_1.status());
  auto src_backend_1 =
      std::dynamic_pointer_cast<HostOffloadBackend>(*src_backend_or_1);
  src_backend_1->Insert({"hash_0"},
                        {RaidenBlockId(src_raiden_id_1, 10, BlockStatus::HOST)},
                        /*on_host=*/true);
  auto src_server_1 = KVCacheStoreServer::Create();
  ASSERT_OK(src_server_1->StartServer(src_backend_1.get(), controller.get(),
                                      "127.0.0.1"));

  // Source 2
  BackendConfig src_config_2;
  src_config_2.type = "HostOffloadBackend";
  src_config_2.capacity = 100;
  src_config_2.global_registry_address = registry_address;
  src_config_2.raiden_id = src_raiden_id_2;
  auto src_backend_or_2 =
      HostOffloadBackend::Create(src_config_2, controller.get());
  ASSERT_OK(src_backend_or_2.status());
  auto src_backend_2 =
      std::dynamic_pointer_cast<HostOffloadBackend>(*src_backend_or_2);
  src_backend_2->Insert({"hash_1"},
                        {RaidenBlockId(src_raiden_id_2, 20, BlockStatus::HOST)},
                        /*on_host=*/true);
  auto src_server_2 = KVCacheStoreServer::Create();
  ASSERT_OK(src_server_2->StartServer(src_backend_2.get(), controller.get(),
                                      "127.0.0.1"));

  auto channel =
      grpc::CreateChannel(registry_address, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient client(channel);
  ASSERT_OK(client.RegisterStore(src_raiden_id_1,
                                 src_server_1->GetServerAddress(),
                                 controller->controller_address()));
  ASSERT_OK(client.RegisterStore(src_raiden_id_2,
                                 src_server_2->GetServerAddress(),
                                 controller->controller_address()));

  RaidenId rid{"dst_job", "0", "dst_cache", 0};
  KVCacheStore store(10, std::move(controller), registry_address, rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

  // hash_0 lives on one peer and hash_1 on another; the caller names both.
  std::vector<std::string> hashes = {"hash_0", "hash_1"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(src_raiden_id_1, 10, BlockStatus::REMOTE),
      RaidenBlockId(src_raiden_id_2, 20, BlockStatus::REMOTE)};

  // Trigger ReadRemote for both: Load enforces single-peer per batch.
  absl::Status status = store.ReadRemote(hashes, slices, {7, 8});
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Mixed remote node IDs"));
}

TEST_F(KVCacheStoreEmbeddedControllerTest,
       SaveSetsMetadataEntriesOnCompletion) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller = MakeController();
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  MetadataRegion region(10);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 10);
  ASSERT_TRUE(metadata_or.ok());

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, *metadata_or,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, -1, 0, BlockStatus::HBM),
      RaidenBlockId(rid, -1, 1, BlockStatus::HBM)};

  ASSERT_TRUE(InsertResident(store, hashes, slices, false));
  ASSERT_TRUE(store.Lookup(hashes).ok());

  // Insert has already called SetMetadataEntry for both slices, but their HBM
  // status fails its data-lives-in-local-host-memory filter: the data exists
  // only in HBM at this point, so the LRU registration alone must leave the
  // table empty.
  EXPECT_THAT(metadata_or->ValidEntries(), ::testing::IsEmpty());

  absl::Status status = store.Save(hashes);
  ASSERT_TRUE(status.ok()) << status.message();

  // Poll for completion
  bool done = false;
  while (!done) {
    auto [save_done, save_failed, save_pending, save_existing,
          save_unregistered] = store.PollSaveStatus();
    if (!save_failed.empty()) {
      FAIL() << "Async Save failed during polling";
    }
    if (!save_done.empty()) {
      done = true;
    }
    if (!done) {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }

  // Save completion lands the data on host blocks 0 and 1, which is when the
  // bindings enter the table.
  EXPECT_THAT(metadata_or->ValidEntries(),
              ElementsAre(::testing::FieldsAre(0, "hash_1", 0),
                          ::testing::FieldsAre(1, "hash_2", 1)));
}


TEST(KVCacheStoreTest, RecoverFromLocalManifestRebuildsLruCache) {
  RaidenId rid{"manifest_job", "0", "kv_cache", 0};
  MetadataRegion region(10);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 10);
  ASSERT_TRUE(metadata_or.ok());

  // Table left behind by the previous incarnation of this store.
  ASSERT_TRUE(metadata_or->Set(5, "hash_b", 3).ok());
  ASSERT_TRUE(metadata_or->Set(7, "hash_a", 4).ok());
  ASSERT_TRUE(metadata_or->Set(9, "hash_c", 8).ok());

  auto controller = MakeRecoveryController(rid, 10);
  auto* controller_ptr = controller.get();
  KVCacheStore store(10, std::move(controller),
                     /*global_registry_address=*/"", rid, *metadata_or,
                     /*store_server_ip=*/"127.0.0.1");

  auto recovered_or = store.RecoverFromLocalManifest();
  ASSERT_TRUE(recovered_or.ok()) << recovered_or.status().ToString();
  EXPECT_EQ(*recovered_or, 3);

  auto lookup = PeekLookup(store, {"hash_a", "hash_b", "hash_c"});
  ASSERT_TRUE(lookup.ok());
  ASSERT_EQ(lookup->size(), 3);
  EXPECT_EQ((*lookup)[0].second.status, BlockStatus::HOST);
  EXPECT_EQ((*lookup)[0].second.host_block_id, 7);
  EXPECT_EQ((*lookup)[1].second.host_block_id, 5);
  EXPECT_EQ((*lookup)[2].second.host_block_id, 9);

  // Recovered blocks are allocated and locked; new allocations avoid them.
  for (int id : {5, 7, 9}) {
    EXPECT_TRUE(controller_ptr->block_manager()->IsAllocated(id));
    EXPECT_TRUE(controller_ptr->block_manager()->IsLocked(id));
  }

  // The seq counter resumes past the largest recovered stamp: the next host
  // insert is stamped 9, not 0.
  ASSERT_TRUE(
      InsertResident(store, {"hash_d"}, {RaidenBlockId(rid, 0, BlockStatus::HOST)}, true));
  EXPECT_THAT(metadata_or->ValidEntries(),
              ElementsAre(::testing::FieldsAre(0, "hash_d", 9),
                          ::testing::FieldsAre(5, "hash_b", 3),
                          ::testing::FieldsAre(7, "hash_a", 4),
                          ::testing::FieldsAre(9, "hash_c", 8)));
}

TEST(KVCacheStoreTest, RecoverFromLocalManifestRebuildsLruOrder) {
  RaidenId rid{"manifest_job_order", "0", "kv_cache", 0};
  MetadataRegion region(10);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 10);
  ASSERT_TRUE(metadata_or.ok());

  ASSERT_TRUE(metadata_or->Set(5, "hash_b", 3).ok());
  ASSERT_TRUE(metadata_or->Set(7, "hash_a", 4).ok());
  ASSERT_TRUE(metadata_or->Set(9, "hash_c", 8).ok());

  // The table also records eviction candidates, so it may hold more entries
  // than the LRU cache capacity. With capacity 2 the oldest entry overflows
  // into a candidate again, keeping its block and metadata entry.
  KVCacheStore store(2, MakeRecoveryController(rid, 10),
                     /*global_registry_address=*/"", rid, *metadata_or,
                     /*store_server_ip=*/"127.0.0.1");
  auto recovered_or = store.RecoverFromLocalManifest();
  ASSERT_TRUE(recovered_or.ok()) << recovered_or.status().ToString();
  EXPECT_EQ(*recovered_or, 3);

  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ElementsAre("hash_b"));
  EXPECT_EQ(metadata_or->ValidEntries().size(), 3);
}

TEST(KVCacheStoreTest, RecoverFromLocalManifestKeepsNewestDuplicate) {
  RaidenId rid{"manifest_job_dup", "0", "kv_cache", 0};
  MetadataRegion region(10);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 10);
  ASSERT_TRUE(metadata_or.ok());

  ASSERT_TRUE(metadata_or->Set(2, "dup_hash", 1).ok());
  ASSERT_TRUE(metadata_or->Set(4, "other", 3).ok());
  ASSERT_TRUE(metadata_or->Set(6, "dup_hash", 5).ok());

  auto controller = MakeRecoveryController(rid, 10);
  auto* controller_ptr = controller.get();
  KVCacheStore store(10, std::move(controller),
                     /*global_registry_address=*/"", rid, *metadata_or,
                     /*store_server_ip=*/"127.0.0.1");

  auto recovered_or = store.RecoverFromLocalManifest();
  ASSERT_TRUE(recovered_or.ok()) << recovered_or.status().ToString();
  EXPECT_EQ(*recovered_or, 2);

  // The newest binding wins; the stale block is neither tracked nor
  // allocated, and its entry is cleared from the table.
  auto lookup = PeekLookup(store, {"dup_hash"});
  ASSERT_EQ(lookup->size(), 1);
  EXPECT_EQ((*lookup)[0].second.host_block_id, 6);
  EXPECT_TRUE(controller_ptr->block_manager()->IsAllocated(6));
  EXPECT_FALSE(controller_ptr->block_manager()->IsAllocated(2));
  EXPECT_THAT(metadata_or->ValidEntries(),
              ElementsAre(::testing::FieldsAre(4, "other", 3),
                          ::testing::FieldsAre(6, "dup_hash", 5)));
}

// Only reachable through misuse: recovery must run on a fresh store, so a
// conflicting allocation means someone allocated before (or instead of)
// recovering. Verifies the failure is clean — error out, LRU cache and table
// untouched.
TEST(KVCacheStoreTest, RecoverFromLocalManifestFailsOnAllocatorConflict) {
  RaidenId rid{"manifest_job_conflict", "0", "kv_cache", 0};
  MetadataRegion region(10);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 10);
  ASSERT_TRUE(metadata_or.ok());
  ASSERT_TRUE(metadata_or->Set(0, "rh1", 0).ok());

  auto controller = MakeRecoveryController(rid, 10);
  // Block 0 is already taken locally before recovery runs.
  ASSERT_TRUE(controller->AllocateBlockIds(1).ok());

  KVCacheStore store(10, std::move(controller),
                     /*global_registry_address=*/"", rid, *metadata_or,
                     /*store_server_ip=*/"127.0.0.1");
  auto recovered_or = store.RecoverFromLocalManifest();
  EXPECT_EQ(recovered_or.status().code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(store.Lookup({"rh1"})->size(), 0);
  EXPECT_EQ(metadata_or->ValidEntries().size(), 1);
}

TEST(KVCacheStoreTest, RecoverFromLocalManifestPreconditions) {
  RaidenId rid{"manifest_job_pre", "0", "kv_cache", 0};
  MetadataRegion region(10);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 10);
  ASSERT_TRUE(metadata_or.ok());

  // A controller-less store is unrepresentable under the construction rules,
  // so the old no-controller sub-case is gone.

  // No attached metadata table.
  KVCacheStore store_no_metadata(10, MakeRecoveryController(rid, 10),
                                 /*global_registry_address=*/"", rid,
                                 std::nullopt,
                                 /*store_server_ip=*/"127.0.0.1");
  EXPECT_EQ(store_no_metadata.RecoverFromLocalManifest().status().code(),
            absl::StatusCode::kFailedPrecondition);

  // Non-empty LRU cache.
  KVCacheStore store_non_empty(10, MakeRecoveryController(rid, 10),
                               /*global_registry_address=*/"", rid,
                               *metadata_or,
                               /*store_server_ip=*/"127.0.0.1");
  ASSERT_TRUE(
      InsertResident(
              store_non_empty, {"hash_a"}, {RaidenBlockId(rid, 0, BlockStatus::HOST)}, true));
  EXPECT_EQ(store_non_empty.RecoverFromLocalManifest().status().code(),
            absl::StatusCode::kFailedPrecondition);

  // Empty table: recovery succeeds with zero blocks.
  MetadataRegion empty_region(10);
  auto empty_metadata_or = KVCacheMetadata::Format(empty_region.span(), 10);
  ASSERT_TRUE(empty_metadata_or.ok());
  KVCacheStore store_empty(10, MakeRecoveryController(rid, 10),
                           /*global_registry_address=*/"", rid,
                           *empty_metadata_or,
                           /*store_server_ip=*/"127.0.0.1");
  auto recovered_or = store_empty.RecoverFromLocalManifest();
  ASSERT_TRUE(recovered_or.ok()) << recovered_or.status().ToString();
  EXPECT_EQ(*recovered_or, 0);
}

TEST(KVCacheStoreTest, MultiBackendPriorityLookupChain) {
  auto b1 = std::make_shared<TestHostOffloadBackend>(/*capacity=*/4);
  auto b2 = std::make_shared<TestHostOffloadBackend>(/*capacity=*/2);

  RaidenId id{"job", "0", "cache", 0};
  std::vector<std::string> b1_hashes = {"h1", "h2"};
  std::vector<RaidenBlockId> b1_slices = {
      RaidenBlockId(id, 1, BlockStatus::HOST),
      RaidenBlockId(id, 2, BlockStatus::HOST)};
  b1->Insert(b1_hashes, b1_slices, /*on_host=*/true);

  std::vector<std::string> b2_hashes = {"h3", "h4"};
  std::vector<RaidenBlockId> b2_slices = {
      RaidenBlockId(id, 3, BlockStatus::HOST),
      RaidenBlockId(id, 4, BlockStatus::HOST)};
  b2->Insert(b2_hashes, b2_slices, /*on_host=*/true);

  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{b1, b2},
                     RaidenId{}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");

  auto lookup_res =
      store.Lookup({"h1", "h2", "h3", "h4"}, /*enable_global=*/true);
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 4);
  EXPECT_EQ((*lookup_res)[0].first, "h1");
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, 1);
  EXPECT_EQ((*lookup_res)[1].first, "h2");
  EXPECT_EQ((*lookup_res)[1].second.host_block_id, 2);
  EXPECT_EQ((*lookup_res)[2].first, "h3");
  EXPECT_EQ((*lookup_res)[2].second.host_block_id, 3);
  EXPECT_EQ((*lookup_res)[3].first, "h4");
  EXPECT_EQ((*lookup_res)[3].second.host_block_id, 4);
}

TEST(KVCacheStoreTest, MultiBackendLocalLookupWhenGlobalDisabled) {
  auto b1 = std::make_shared<TestHostOffloadBackend>(/*capacity=*/2);
  auto b2 = std::make_shared<TestHostOffloadBackend>(/*capacity=*/2);

  RaidenId id{"job", "0", "cache", 0};
  b1->Insert({"h1"}, {RaidenBlockId(id, 1, BlockStatus::HOST)},
             /*on_host=*/true);
  b2->Insert({"h2"}, {RaidenBlockId(id, 2, BlockStatus::HOST)},
             /*on_host=*/true);

  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{b1, b2},
                     RaidenId{}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");

  // enable_global = false => searches all local backends (b1 and b2)
  auto gated_res = store.Lookup({"h1", "h2"}, /*enable_global=*/false);
  ASSERT_TRUE(gated_res.ok());
  ASSERT_EQ(gated_res->size(), 2);
  EXPECT_EQ((*gated_res)[0].first, "h1");
  EXPECT_EQ((*gated_res)[1].first, "h2");

  // enable_global = true => queries all backends as well
  auto ungated_res = store.Lookup({"h1", "h2"}, /*enable_global=*/true);
  ASSERT_TRUE(ungated_res.ok());
  ASSERT_EQ(ungated_res->size(), 2);
  EXPECT_EQ((*ungated_res)[0].first, "h1");
  EXPECT_EQ((*ungated_res)[1].first, "h2");
}

TEST(KVCacheStoreTest, MultiBackendInsertAndLockRollback) {
  auto b1 = std::make_shared<TestHostOffloadBackend>(/*capacity=*/2);
  auto b2 = std::make_shared<TestHostOffloadBackend>(/*capacity=*/1);

  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{b1, b2},
                     RaidenId{}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");

  RaidenId id{"job", "0", "cache", 0};
  std::vector<std::string> hashes = {"h1", "h2"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(id, /*host_block_id=*/-1, /*device_block_id=*/1,
                    BlockStatus::HBM),
      RaidenBlockId(id, /*host_block_id=*/-1, /*device_block_id=*/2,
                    BlockStatus::HBM)};

  // b1 capacity=2 supports 2 blocks, but b2 capacity=1 fails on 2 blocks.
  absl::Status status = store.Insert(hashes, slices, /*on_host=*/true);
  EXPECT_TRUE(absl::IsResourceExhausted(status)) << status;

  // Verify rollback on b1: no locks or entries remain.
  EXPECT_EQ(b1->GetSize(), 0);
  EXPECT_EQ(b1->GetPinCount("h1"), 0);
  EXPECT_EQ(b1->GetPinCount("h2"), 0);
  EXPECT_EQ(b2->GetSize(), 0);
}


using ::testing::EndsWith;
using ::testing::Not;
using ::testing::StartsWith;

// ---------------------------------------------------------------------------
// Store-server discovery: a store publishes where peers reach it, keyed by
// its RaidenId, so a global-registry Lookup result becomes dialable.
// ---------------------------------------------------------------------------

// Owns a registry server on an ephemeral port for the tests below.
class StoreDiscoveryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    service_ = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                             &port);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    registry_address_ = "localhost:" + std::to_string(port);
    channel_ = grpc::CreateChannel(registry_address_,
                                   grpc::InsecureChannelCredentials());
    client_ = std::make_unique<global_registry::GlobalRegistryClient>(channel_);
  }

  void TearDown() override {
    client_.reset();
    channel_.reset();
    if (server_) {
      server_->Shutdown(std::chrono::system_clock::now() +
                        std::chrono::milliseconds(200));
    }
  }

  std::string registry_address_;
  std::unique_ptr<global_registry::GlobalRegistryServiceImpl> service_;
  std::unique_ptr<grpc::Server> server_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<global_registry::GlobalRegistryClient> client_;
};

TEST_F(StoreDiscoveryTest, PublishesStoreAddressToTheRegistry) {
  RaidenId rid{"disco_job", "0", "kv_cache", 0};
  KVCacheStore store(/*capacity=*/16, registry_address_, rid, /*num_shards=*/1,
                     /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1",
                     /*raiden_controller_port=*/0);

  // Advertised host is exactly what was supplied; the port is gRPC's choice.
  ASSERT_FALSE(store.store_server_address().empty());
  EXPECT_THAT(store.store_server_address(), StartsWith("127.0.0.1:"));
  ASSERT_NE(store.store_server(), nullptr);
  EXPECT_EQ(store.store_server_address(),
            absl::StrCat("127.0.0.1:", store.store_server()->GetGrpcPort()));

  auto resolved = client_->ResolveStore(rid);
  ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
  EXPECT_EQ(resolved->store_server_address(), store.store_server_address());
  // The controller address rides along: it is what a peer dials to acquire a
  // read lease against this store.
  EXPECT_EQ(resolved->controller_address(), store.raiden_controller_address());
}

// store_server_ip is bind-and-advertise, so the published address is
// actually connectable -- the property the old "localhost:<port>" lacked.
TEST_F(StoreDiscoveryTest, PublishedAddressIsConnectable) {
  RaidenId rid{"disco_job", "0", "kv_cache", 0};
  KVCacheStore store(/*capacity=*/16, registry_address_, rid, /*num_shards=*/1,
                     /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");

  auto resolved = client_->ResolveStore(rid);
  ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();

  auto peer_channel = grpc::CreateChannel(resolved->store_server_address(),
                                          grpc::InsecureChannelCredentials());
  ASSERT_TRUE(peer_channel->WaitForConnected(std::chrono::system_clock::now() +
                                             std::chrono::seconds(10)));
}

// Teardown must retract the registration, or peers keep dialling a dead port.
TEST_F(StoreDiscoveryTest, DestructorUnpublishes) {
  RaidenId rid{"disco_job_gone", "0", "kv_cache", 0};
  {
    KVCacheStore store(/*capacity=*/16, registry_address_, rid,
                       /*num_shards=*/1, /*shard_size_bytes=*/512,
                       /*store_server_ip=*/"127.0.0.1");
    ASSERT_TRUE(client_->ResolveStore(rid).ok());
  }
  EXPECT_TRUE(absl::IsNotFound(client_->ResolveStore(rid).status()));
}

// A restarted store comes back on a different ephemeral port. Because the
// registration is keyed by RaidenId, the new address replaces the old one
// rather than accumulating beside it.
TEST_F(StoreDiscoveryTest, RestartReplacesPublishedAddress) {
  RaidenId rid{"disco_job_restart", "0", "kv_cache", 0};

  std::string first_address;
  {
    KVCacheStore store(/*capacity=*/16, registry_address_, rid,
                       /*num_shards=*/1, /*shard_size_bytes=*/512,
                       /*store_server_ip=*/"127.0.0.1");
    first_address = store.store_server_address();
  }

  KVCacheStore restarted(/*capacity=*/16, registry_address_, rid,
                         /*num_shards=*/1, /*shard_size_bytes=*/512,
                         /*store_server_ip=*/"127.0.0.1");
  auto resolved = client_->ResolveStore(rid);
  ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
  EXPECT_EQ(resolved->store_server_address(), restarted.store_server_address());
  EXPECT_NE(resolved->store_server_address(), first_address);
}

// When a backend already hosts a store server, the store must adopt and
// publish THAT server rather than stand up a second one, so a node serves peers
// from exactly one port. The store must also get there before anything else
// starts that server, because StartServer never rebinds a running one -- the
// factory path GlobalMemoryPoolingBackend::Create(config, controller) does
// start it, and the single-argument Create this test goes through does not.
TEST_F(StoreDiscoveryTest, AdoptsAndPublishesTheBackendsServer) {
  RaidenId rid{"disco_job_tiered", "0", "kv_cache", 0};

  BackendConfig host_config;
  host_config.type = "HostOffloadBackend";
  host_config.capacity = 16;
  host_config.raiden_id = rid;

  BackendConfig pooling_config;
  pooling_config.type = "HostOffloadBackend";
  pooling_config.capacity = 16;
  pooling_config.global_registry_address = registry_address_;
  pooling_config.raiden_id = rid;

  const BackendConfig configs[] = {host_config, pooling_config};
  auto store_or = KVCacheStore::Create(
      absl::MakeConstSpan(configs), /*capacity=*/16, registry_address_, rid,
      /*num_shards=*/1, /*shard_size_bytes=*/512,
      /*store_server_ip=*/"127.0.0.1");
  ASSERT_TRUE(store_or.ok()) << store_or.status().ToString();
  auto& store = **store_or;

  auto* pooling = dynamic_cast<HostOffloadBackend*>(store.backends()[1].get());
  ASSERT_NE(pooling, nullptr);

  // Exactly one server, created and published by the store.
  ASSERT_NE(store.store_server(), nullptr);

  // Published under the supplied ip, not the backend's hardcoded wildcard.
  EXPECT_THAT(store.store_server_address(), StartsWith("127.0.0.1:"));
  auto resolved = client_->ResolveStore(rid);
  ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
  EXPECT_EQ(resolved->store_server_address(), store.store_server_address());
}

// A backend that already hosts a KVCacheStoreServer (because something started
// one on it explicitly before store construction) is adopted by the store
// rather than standing up a second server.
TEST_F(StoreDiscoveryTest, AdoptsTheBackendsServerRatherThanOwningASecond) {
  RaidenId rid{"disco_job_adopt", "0", "kv_cache", 0};

  // Nothing starts a backend's server implicitly: the only way a backend can
  // already host one by the time the store wires up is a caller starting it
  // explicitly beforehand, as here.
  // The bootstrap controller only needs to be non-null for StartServer to
  // succeed; the store below wires its own controller in afterward.
  auto bootstrap_controller = MakeRecoveryController(rid, /*num_blocks=*/16);
  auto backend = std::make_shared<TestHostOffloadBackend>(
      /*capacity=*/16, std::nullopt, rid, bootstrap_controller.get());
  ASSERT_TRUE(backend->StartServer("127.0.0.1").ok());
  ASSERT_NE(backend->store_server(), nullptr);

  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{backend},
                     rid,
                     /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1",
                     /*raiden_controller_port=*/0, registry_address_);

  ASSERT_NE(store.store_server(), nullptr);
  ASSERT_FALSE(store.backends().empty());
  EXPECT_EQ(store.store_server(), store.backends()[0]->store_server());
}

// The controller is addressed from the same ip, with its port either chosen by
// gRPC (0) or taken verbatim.
TEST_F(StoreDiscoveryTest, ControllerAddressComposedFromIpAndPort) {
  RaidenId rid{"disco_job_ctrl", "0", "kv_cache", 0};
  KVCacheStore store(/*capacity=*/16, registry_address_, rid, /*num_shards=*/1,
                     /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1",
                     /*raiden_controller_port=*/0);

  EXPECT_THAT(store.raiden_controller_address(), StartsWith("127.0.0.1:"));
  // Port 0 means "gRPC picks"; the advertised address carries the real port.
  EXPECT_THAT(store.raiden_controller_address(), Not(EndsWith(":0")));
}

// With the monitor enabled the registration carries a TTL and lives on
// heartbeats alone, and the placement attributes from the BackendConfig ride
// along -- so the store shows up as a placement target for a smaller-tier
// caller in its group.
TEST_F(StoreDiscoveryTest, StoreMonitorHeartbeatsTheRegistration) {
  RaidenId rid{"disco_job_monitored", "0", "kv_cache", 0};
  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 16;
  config.raiden_id = rid;
  config.global_registry_address = registry_address_;
  config.kv_pool_group = "groupA";
  config.evict_tier = 1;
  config.monitor_config.enable = true;
  config.monitor_config.heartbeat_period = absl::Milliseconds(300);

  auto store_or = KVCacheStore::Create(config, /*capacity=*/16,
                                       registry_address_, rid,
                                       /*num_shards=*/1,
                                       /*shard_size_bytes=*/512,
                                       /*store_server_ip=*/"127.0.0.1");
  ASSERT_TRUE(store_or.ok()) << store_or.status().ToString();

  // The registration's TTL is three heartbeat periods (900ms here); well
  // past it, only the heartbeats keep the store resolvable.
  absl::SleepFor(absl::Seconds(2));
  auto resolved = client_->ResolveStore(rid);
  ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
  EXPECT_EQ(resolved->store_server_address(),
            (*store_or)->store_server_address());

  // A tier-0 caller in the same group is offered this store: the
  // kv_pool_group and evict_tier from the BackendConfig were published.
  RaidenId caller{"disco_caller", "0", "kv_cache", 0};
  ASSERT_TRUE(client_
                  ->RegisterStore(caller, "10.0.0.7:1111",
                                  /*controller_address=*/"",
                                  /*ttl=*/absl::ZeroDuration(), "groupA",
                                  /*evict_tier=*/0)
                  .ok());
  auto targets = client_->GetPlacementTargets(caller, /*max_targets=*/8);
  ASSERT_TRUE(targets.ok()) << targets.status().ToString();
  ASSERT_EQ(targets->size(), 1);
  EXPECT_EQ((*targets)[0].raiden_id().job_name(), "disco_job_monitored");

  // Destruction stops the monitor before unpublishing, so no late heartbeat
  // re-registers the store after this.
  store_or->reset();
  EXPECT_TRUE(absl::IsNotFound(client_->ResolveStore(rid).status()));
}

// A registration lapse means the registry may have purged this store's block
// entries (its dead-store cascade). The heartbeat that discovers the lapse
// re-registers and must then republish the store's host-resident inventory.
TEST_F(StoreDiscoveryTest, LapsedRegistrationRepublishesBlockEntries) {
  RaidenId rid{"disco_job_republish", "0", "kv_cache", 0};
  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 16;
  config.raiden_id = rid;
  config.global_registry_address = registry_address_;
  config.kv_pool_group = "groupA";
  config.evict_tier = 1;
  config.monitor_config.enable = true;
  config.monitor_config.heartbeat_period = absl::Milliseconds(300);

  auto store_or = KVCacheStore::Create(config, /*capacity=*/16,
                                       registry_address_, rid,
                                       /*num_shards=*/1,
                                       /*shard_size_bytes=*/512,
                                       /*store_server_ip=*/"127.0.0.1");
  ASSERT_TRUE(store_or.ok()) << store_or.status().ToString();
  KVCacheStore& store = **store_or;

  // Host-resident blocks with no registry entries: exactly the state the
  // dead-store cascade leaves behind (and what Insert alone produces --
  // publishing normally happens when a save completes).
  const std::vector<std::string> hashes = {"rp_h1", "rp_h2"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(rid, /*host_id=*/5, BlockStatus::HOST),
      RaidenBlockId(rid, /*host_id=*/6, BlockStatus::HOST)};
  ASSERT_TRUE(store.Insert(hashes, slices, /*on_host=*/true).ok());
  auto looked = client_->Lookup(hashes);
  ASSERT_TRUE(looked.ok());
  ASSERT_EQ(looked->size(), 0);

  // Drop the store registration; the heartbeat has to discover the lapse.
  ASSERT_TRUE(client_->UnregisterStore(rid).ok());

  // The next heartbeat gets NotFound, re-registers, and republishes.
  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  size_t republished = 0;
  while (absl::Now() < deadline) {
    auto again = client_->Lookup(hashes);
    if (again.ok() && again->size() == hashes.size()) {
      republished = again->size();
      EXPECT_EQ((*again)[0].block_id(), 5);
      EXPECT_EQ((*again)[1].block_id(), 6);
      break;
    }
    absl::SleepFor(absl::Milliseconds(100));
  }
  EXPECT_EQ(republished, hashes.size());
  EXPECT_TRUE(client_->ResolveStore(rid).ok());
}

// The flag promises heartbeats; without a registry there is no registration
// to refresh, and Create says so instead of quietly not monitoring.
TEST_F(StoreDiscoveryTest, StoreMonitorWithoutARegistryIsAnError) {
  RaidenId rid{"disco_job_unmonitorable", "0", "kv_cache", 0};
  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 16;
  config.raiden_id = rid;
  config.monitor_config.enable = true;

  auto store_or = KVCacheStore::Create(config, /*capacity=*/16,
                                       /*global_registry_address=*/"", rid,
                                       /*num_shards=*/1,
                                       /*shard_size_bytes=*/512,
                                       /*store_server_ip=*/"127.0.0.1");
  EXPECT_TRUE(absl::IsFailedPrecondition(store_or.status()))
      << store_or.status().ToString();
}

// A tier-0 backend whose Lookup parks, so a peer's Fetch can be held inside the
// store's own service while the store is being destroyed.
class ParkingBackend : public TestHostOffloadBackend {
 public:
  using TestHostOffloadBackend::TestHostOffloadBackend;

  absl::StatusOr<BlockSliceList> Lookup(
      absl::Span<const std::string> block_hashes,
      const LookupOptions& options = {}) override {
    if (!entered.HasBeenNotified()) entered.Notify();
    release.WaitForNotification();
    handler_finished.store(true);
    return HostOffloadBackend::Lookup(block_hashes, options);
  }

  absl::Notification entered;
  absl::Notification release;
  std::atomic<bool> handler_finished{false};
};

// Destroying a store must not return while a peer's RPC is still executing
// inside its service.
//
// Scope, stated because it is easy to over-read: this covers a store-OWNED
// server, where the drain is guaranteed twice over -- by the destructor's
// explicit Shutdown, and by owned_store_server_ being declared after
// raiden_controller_ and so destroyed before it. Removing the explicit
// Shutdown does NOT fail this test for that reason.
//
// The case the explicit Shutdown alone covers is a server ADOPTED from a
// backend: backends_ is declared before raiden_controller_, so member
// destruction frees the controller first and leaves a backend-hosted server
// answering RPCs that dereference it. That path has no regression test -- see
// the teardown notes in the store-server discovery doc.
TEST_F(StoreDiscoveryTest, TeardownDrainsAnInFlightPeerRpc) {
  RaidenId rid{"disco_job_teardown", "0", "kv_cache", 0};
  auto parking = std::make_shared<ParkingBackend>(
      /*capacity=*/16, std::nullopt, rid, /*raiden_controller=*/nullptr);

  auto store = std::make_unique<KVCacheStore>(
      parking, rid, /*num_shards=*/1, /*shard_size_bytes=*/512,
      /*store_server_ip=*/"127.0.0.1",
      /*raiden_controller_port=*/0, registry_address_);
  ASSERT_FALSE(store->store_server_address().empty());

  auto peer_channel = grpc::CreateChannel(store->store_server_address(),
                                          grpc::InsecureChannelCredentials());
  KVCacheStoreClient peer(peer_channel);
  auto fetch_future = peer.Fetch({"h1"}, /*device_block_ids=*/{},
                                 /*host_block_ids=*/{7});

  // The peer's Fetch is now parked inside our service, holding the handler.
  parking->entered.WaitForNotification();

  std::thread destroyer([&store] { store.reset(); });
  // The destructor should be blocked draining that handler, not racing past it.
  absl::SleepFor(absl::Milliseconds(200));
  EXPECT_FALSE(parking->handler_finished.load());

  parking->release.Notify();
  destroyer.join();

  // Teardown outlived the handler rather than pulling the controller out from
  // under it.
  EXPECT_TRUE(parking->handler_finished.load());
  (void)fetch_future.Await();  // whatever it reports, it must not crash
}

// Constructing a store with no tier-0 backend (or a null tier-0 pointer) is
// now a construction-rule violation. Catching it at construction guarantees
// that a store configured with a registry is always registered by preventing
// the un-registered state from being constructed.
TEST_F(StoreDiscoveryTest, NoBackendIsAConstructionRuleViolation) {
  // Required, not stylistic: this fixture has a running gRPC registry server,
  // so the process is multithreaded by the time the test body runs. The
  // default "fast" death test style forks, and fork clones only the calling
  // thread -- any lock a gRPC thread happened to hold is inherited locked and
  // never released, so the child wedges building its own registry channel and
  // the parent waits forever for a child that cannot die. "threadsafe" re-execs
  // the binary instead of forking, so it starts with no inherited locks.
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  RaidenId rid{"disco_job_nobackend", "0", "kv_cache", 0};
  EXPECT_DEATH(
      {
        KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{},
                           rid,
                           /*num_shards=*/1, /*shard_size_bytes=*/512,
                           /*store_server_ip=*/"127.0.0.1",
                           /*raiden_controller_port=*/0, registry_address_);
      },
      "requires at least one backend");

  EXPECT_DEATH(
      {
        KVCacheStore store(
            std::vector<std::shared_ptr<KVCacheStoreBackend>>{nullptr}, rid,
            /*num_shards=*/1, /*shard_size_bytes=*/512,
            /*store_server_ip=*/"127.0.0.1",
            /*raiden_controller_port=*/0, registry_address_);
      },
      "tier-0 backend must not be null");

  EXPECT_FALSE(KVCacheStore::Create(std::vector<BackendConfig>{},
                                    /*capacity=*/16, registry_address_, rid,
                                    /*num_shards=*/1, /*shard_size_bytes=*/512,
                                    /*store_server_ip=*/"127.0.0.1",
                                    /*raiden_controller_port=*/0)
                   .ok());
}

// Giving those two constructors a real registry client is a behaviour change,
// not just a repair: their backends now take part in the global tier. Pinned
// here so it stays a decision on the record rather than something a later
// reader "fixes" back.
TEST_F(StoreDiscoveryTest, CapacityConstructedStoreJoinsTheGlobalTier) {
  RaidenId rid{"disco_job_globaltier", "0", "kv_cache", 0};
  KVCacheStore store(/*capacity=*/16, registry_address_, rid, /*num_shards=*/1,
                     /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1",
                     /*raiden_controller_port=*/0);

  auto* backend = dynamic_cast<HostOffloadBackend*>(store.backend().get());
  ASSERT_NE(backend, nullptr);

  // The backend holds a real registry client, so what it publishes reaches
  // the global tier. Inserting does not publish; a completed save does,
  // through this path -- so drive it directly.
  ASSERT_OK(backend->RegisterBlocksAsync({"tiered_hash"}, {3}).Await());
  auto looked_up = client_->Lookup({"tiered_hash"});
  ASSERT_TRUE(looked_up.ok()) << looked_up.status().ToString();
  ASSERT_EQ(looked_up->size(), 1);
  EXPECT_EQ((*looked_up)[0].block_id(), 3);

  // A local miss consults tier 1 and comes back with the owning peer.
  RaidenId peer{"some_peer", "0", "kv_cache", 0};
  ASSERT_TRUE(client_->Register({{"peer_hash", peer, 9}}).ok());
  auto result = backend->Lookup({"peer_hash"});
  ASSERT_TRUE(result.ok()) << result.status().ToString();
  ASSERT_EQ(result->size(), 1);
  EXPECT_EQ((*result)[0].second.raiden_id, peer);
  EXPECT_EQ((*result)[0].second.status, BlockStatus::REMOTE);
}

// A backend's server can be started AFTER the store was constructed -- which
// is too late for the store's adoption pass to have seen it -- and with no
// registry the store never adopts one at all. That server holds the store's
// RaidenController in a pointer it cannot re-seat, so leaving it running past
// the store's destructor gives a live service dereferencing a freed
// controller. Waiting for ~HostOffloadBackend is not enough: backends_ holds
// shared_ptrs, so a caller that keeps its own reference outlives the store,
// which is exactly what this test does.
//
// GetGrpcPort() going to 0 is Shutdown()'s observable effect, so this fails
// deterministically without a sanitizer.
TEST_F(StoreDiscoveryTest, DestructorShutsDownABackendStartedServer) {
  RaidenId rid{"disco_job_h1", "0", "kv_cache", 0};
  std::shared_ptr<KVCacheStoreBackend> backend_ref;
  {
    KVCacheStore store(/*capacity=*/16, /*global_registry_address=*/"", rid,
                       /*num_shards=*/1, /*shard_size_bytes=*/512,
                       /*store_server_ip=*/"127.0.0.1",
                       /*raiden_controller_port=*/0);
    // No registry, so the store owns and adopts nothing.
    ASSERT_EQ(store.store_server(), nullptr);

    backend_ref = store.backend();
    auto* backend = dynamic_cast<HostOffloadBackend*>(backend_ref.get());
    ASSERT_NE(backend, nullptr);
    ASSERT_TRUE(backend->StartServer("127.0.0.1").ok());
    ASSERT_NE(backend->store_server(), nullptr);
    ASSERT_GT(backend->store_server()->GetGrpcPort(), 0);
  }

  auto* backend = dynamic_cast<HostOffloadBackend*>(backend_ref.get());
  ASSERT_NE(backend->store_server(), nullptr);
  EXPECT_EQ(backend->store_server()->GetGrpcPort(), 0)
      << "the store's destructor left a backend-hosted server running, still "
         "holding the controller it just destroyed";
}

// Same sweep, but with a registry, so the store DID adopt the backend's
// server. Covers the de-duplication: the adopted server is shut once, not
// twice.
TEST_F(StoreDiscoveryTest, DestructorSweepSkipsTheAdoptedServer) {
  RaidenId rid{"disco_job_h1_adopt", "0", "kv_cache", 0};
  auto bootstrap_controller = MakeRecoveryController(rid, /*num_blocks=*/16);
  auto backend = std::make_shared<TestHostOffloadBackend>(
      /*capacity=*/16, std::nullopt, rid, bootstrap_controller.get());
  ASSERT_TRUE(backend->StartServer("127.0.0.1").ok());
  ASSERT_GT(backend->store_server()->GetGrpcPort(), 0);

  {
    KVCacheStore store(
        std::vector<std::shared_ptr<KVCacheStoreBackend>>{backend}, rid,
        /*num_shards=*/1, /*shard_size_bytes=*/512,
        /*store_server_ip=*/"127.0.0.1",
        /*raiden_controller_port=*/0, registry_address_);
    ASSERT_EQ(store.store_server(), backend->store_server());
  }

  EXPECT_EQ(backend->store_server()->GetGrpcPort(), 0);
}

// store_clients_ caches one client per peer and nothing ever erased it, so a
// peer that restarted on a new port stayed undialable for the life of THIS
// process -- even after the registry had healed. The invalidation call was
// lost when GlobalMemoryPoolingBackend was folded into HostOffloadBackend.
//
// The two failures are deliberately different so the assertion has teeth:
// dialling a closed port is UNAVAILABLE, while reaching a real peer that does
// not hold the hash is NOT_FOUND. Without invalidation the second Load
// redials the dead address and stays UNAVAILABLE.
TEST_F(StoreDiscoveryTest, FailedLoadDropsTheCachedPeerClient) {
  RaidenId rid{"disco_job_invalidate", "0", "kv_cache", 0};
  RaidenId peer_rid{"disco_job_invalidate_peer", "0", "kv_cache", 0};

  KVCacheStore store(/*capacity=*/16, registry_address_, rid, /*num_shards=*/1,
                     /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1",
                     /*raiden_controller_port=*/0);
  auto* backend = dynamic_cast<HostOffloadBackend*>(store.backend().get());
  ASSERT_NE(backend, nullptr);

  // Port 1 is reserved and never listening, so this connect always fails.
  ASSERT_TRUE(client_
                  ->RegisterStore(peer_rid, "127.0.0.1:1",
                                  /*controller_address=*/"")
                  .ok());
  // Load requires a destination for every hash. The device block is never written 
  // as the loads below fail while resolving or querying the peer.
  auto first = backend->Load(peer_rid, {"h"}, {0}).Await();
  ASSERT_FALSE(first.ok());
  ASSERT_TRUE(absl::IsUnavailable(first)) << first.ToString();

  // The peer comes back on a real port and republishes, replacing its entry.
  KVCacheStore peer(/*capacity=*/16, registry_address_, peer_rid,
                    /*num_shards=*/1, /*shard_size_bytes=*/512,
                    /*store_server_ip=*/"127.0.0.1",
                    /*raiden_controller_port=*/0);

  ASSERT_FALSE(peer.store_server_address().empty());

  // What this pins is that the peer ANSWERED: a dead cached client would still
  // be UNAVAILABLE. The answer itself is about the request rather than about
  // the hash, because Fetch validates its arguments before it looks anything
  // up, and this store registers no workers to name as endpoints.
  auto second = backend->Load(peer_rid, {"h"}, {0}).Await();
  EXPECT_TRUE(absl::IsInvalidArgument(second))
      << "expected the restarted peer to answer; got " << second.ToString()
      << " -- the cached client still points at the address it had";
  EXPECT_FALSE(absl::IsUnavailable(second));
}

// ===========================================================================
// Remote write -- source side.
// ===========================================================================

// A fake destination that answers whatever the test tells it to, so each
// verdict can be driven directly instead of by arranging a real
// destination's internal state.
class FakeDestinationService
    : public ::tpu_raiden::kv_cache::proto::KVCacheStoreService::CallbackService {
 public:
  static constexpr uint64_t kOperationId = 4242;

  class FakeServerReactor
      : public ::grpc::ServerWriteReactor<
            ::tpu_raiden::kv_cache::proto::WriteRemoteEvent> {
   public:
    FakeServerReactor(FakeDestinationService* parent,
                      proto::WriteRemoteEvent ack_event,
                      std::optional<proto::WriteRemoteEvent> result_event,
                      bool is_ack_only = false)
        : parent_(parent),
          ack_event_(std::move(ack_event)) {
      if (result_event.has_value()) {
        result_event_ = std::move(*result_event);
        has_result_ = true;
      }
      if (is_ack_only) {
        finished_ = true;
        StartWriteAndFinish(&ack_event_, grpc::WriteOptions(),
                            ::grpc::Status::OK);
      } else {
        StartWrite(&ack_event_);
      }
    }

    void FinishWithError(::grpc::Status status) {
      bool should_finish = false;
      {
        absl::MutexLock lock(mu_);
        if (!finished_) {
          finished_ = true;
          should_finish = true;
        }
      }
      if (should_finish) {
        Finish(status);
      }
    }

    void SendResult(proto::WriteRemoteEvent result_event) {
      bool should_write = false;
      {
        absl::MutexLock lock(mu_);
        result_event_ = std::move(result_event);
        has_result_ = true;
        if (ack_done_ && !finished_) {
          finished_ = true;
          should_write = true;
        }
      }
      if (should_write) {
        StartWriteAndFinish(&result_event_, grpc::WriteOptions(),
                            ::grpc::Status::OK);
      }
    }

    void OnWriteDone(bool ok) override {
      if (!ok) {
        FinishWithError(
            ::grpc::Status(::grpc::StatusCode::CANCELLED, "Write failed"));
        return;
      }
      bool should_write = false;
      {
        absl::MutexLock lock(mu_);
        ack_done_ = true;
        if (has_result_ && !finished_) {
          finished_ = true;
          should_write = true;
        }
      }
      if (should_write) {
        StartWriteAndFinish(&result_event_, grpc::WriteOptions(),
                            ::grpc::Status::OK);
      }
    }

    void OnCancel() override {
      // Without this, a cancelled stream never reaches OnDone (gRPC withholds
      // it until Finish is called) and the reactor stays in active_reactors_
      // forever -- so open_streams() could not observe a cancellation.
      FinishWithError(::grpc::Status::CANCELLED);
    }

    void OnDone() override {
      parent_->OnReactorDone(this);
      delete this;
    }

   private:
    FakeDestinationService* const parent_;
    absl::Mutex mu_;
    proto::WriteRemoteEvent ack_event_;
    proto::WriteRemoteEvent result_event_;
    bool ack_done_ ABSL_GUARDED_BY(mu_) = false;
    bool has_result_ ABSL_GUARDED_BY(mu_) = false;
    bool finished_ ABSL_GUARDED_BY(mu_) = false;
  };

  void CancelAllReactors() {
    absl::MutexLock lock(mutex_);
    for (auto* reactor : active_reactors_) {
      reactor->FinishWithError(::grpc::Status(::grpc::StatusCode::UNAVAILABLE,
                                              "Service shutting down"));
    }
    active_reactors_.clear();
  }

  ~FakeDestinationService() override {
    CancelAllReactors();
  }

  ::grpc::ServerWriteReactor<::tpu_raiden::kv_cache::proto::WriteRemoteEvent>*
  WriteRemote(
      ::grpc::CallbackServerContext* /*context*/,
      const ::tpu_raiden::kv_cache::proto::WriteRemoteRequest* request) override {
    absl::MutexLock lock(mutex_);
    ++write_calls_;
    requested_deadline_ms_ = request->deadline_ms();
    if (!write_status_.ok()) {
      auto* reactor =
          new FakeServerReactor(this, {}, std::nullopt, /*is_ack_only=*/false);
      reactor->Finish(write_status_);
      return reactor;
    }

    proto::WriteRemoteEvent ack_event;
    auto* ack = ack_event.mutable_ack();
    ack->set_operation_id(kOperationId);
    ack->set_exist_state(exist_state_);
    ack->set_granted_deadline_ms(granted_deadline_ms_override_ > 0
                                     ? granted_deadline_ms_override_
                                     : request->deadline_ms());
    for (const auto& hash : existing_hashes_) {
      ack->add_existing_hashes(hash);
    }

    if (exist_state_ == proto::WRITE_ALL_EXIST ||
        exist_state_ == proto::WRITE_PARTIAL_EXIST) {
      return new FakeServerReactor(this, std::move(ack_event), std::nullopt,
                                   /*is_ack_only=*/true);
    }

    std::optional<proto::WriteRemoteEvent> result_event;
    if (poll_response_.state() !=
            proto::PollWriteRemoteResponse::STATE_UNSPECIFIED &&
        poll_response_.state() != proto::PollWriteRemoteResponse::PENDING) {
      proto::WriteRemoteEvent res;
      auto* r = res.mutable_result();
      r->set_state(poll_response_.state());
      for (const auto& hash : poll_response_.existing_hashes()) {
        r->add_existing_hashes(hash);
      }
      for (const auto& hash : poll_response_.unregistered_hashes()) {
        r->add_unregistered_hashes(hash);
      }
      result_event = std::move(res);
    }

    const bool has_result = result_event.has_value();
    auto* reactor =
        new FakeServerReactor(this, std::move(ack_event),
                              std::move(result_event), /*is_ack_only=*/false);
    if (!has_result) {
      active_reactors_.push_back(reactor);
    }
    return reactor;
  }

  ::grpc::ServerUnaryReactor* PollWriteRemote(
      ::grpc::CallbackServerContext* context,
      const ::tpu_raiden::kv_cache::proto::PollWriteRemoteRequest* request,
      ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse* response)
      override {
    auto* reactor = context->DefaultReactor();
    absl::MutexLock lock(mutex_);
    ++poll_calls_;
    if (request->wait_ms() > 0 &&
        (poll_response_.state() ==
             proto::PollWriteRemoteResponse::STATE_UNSPECIFIED ||
         poll_response_.state() ==
             proto::PollWriteRemoteResponse::PENDING)) {
      mutex_.AwaitWithTimeout(
          absl::Condition(
              +[](FakeDestinationService* svc) {
                svc->mutex_.AssertHeld();
                return svc->poll_response_.state() !=
                           proto::PollWriteRemoteResponse::STATE_UNSPECIFIED &&
                       svc->poll_response_.state() !=
                           proto::PollWriteRemoteResponse::PENDING;
              },
              this),
          absl::Milliseconds(request->wait_ms()));
    }
    *response = poll_response_;
    reactor->Finish(::grpc::Status::OK);
    return reactor;
  }

  void BreakActiveStreams(::grpc::Status error_status) {
    absl::MutexLock lock(mutex_);
    for (auto* reactor : active_reactors_) {
      reactor->FinishWithError(error_status);
    }
    active_reactors_.clear();
  }

  void SetPollRecoveryResponse(
      ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse response) {
    absl::MutexLock lock(mutex_);
    poll_response_ = std::move(response);
  }

  void SetPollResponse(
      ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse response) {
    absl::MutexLock lock(mutex_);
    poll_response_ = std::move(response);
    if (poll_response_.state() !=
            proto::PollWriteRemoteResponse::STATE_UNSPECIFIED &&
        poll_response_.state() != proto::PollWriteRemoteResponse::PENDING) {
      proto::WriteRemoteEvent res;
      auto* r = res.mutable_result();
      r->set_state(poll_response_.state());
      for (const auto& hash : poll_response_.existing_hashes()) {
        r->add_existing_hashes(hash);
      }
      for (const auto& hash : poll_response_.unregistered_hashes()) {
        r->add_unregistered_hashes(hash);
      }
      for (auto* reactor : active_reactors_) {
        reactor->SendResult(res);
      }
      active_reactors_.clear();
    }
  }

  void OnReactorDone(FakeServerReactor* reactor) {
    absl::MutexLock lock(mutex_);
    std::erase(active_reactors_, reactor);
  }

  // What the ACK says, as opposed to the later result. ALL_EXIST makes the
  // offer settle synchronously inside Save, with no result message.
  void SetWriteExistState(proto::WriteExistState state) {
    absl::MutexLock lock(mutex_);
    exist_state_ = state;
  }

  void SetExistingHashes(std::vector<std::string> hashes) {
    absl::MutexLock lock(mutex_);
    existing_hashes_ = std::move(hashes);
  }

  // Non-OK makes WriteRemote answer with that status instead of accepting,
  // the way a full destination answers RESOURCE_EXHAUSTED.
  void SetWriteRemoteStatus(::grpc::Status status) {
    absl::MutexLock lock(mutex_);
    write_status_ = std::move(status);
  }

  // Overrides the granted deadline in the ack (0 echoes the request), so a
  // test can make the grant outrun the source's HOLD -- the inversion the
  // destination's own clamp normally prevents.
  void SetGrantedDeadlineMs(int64_t ms) {
    absl::MutexLock lock(mutex_);
    granted_deadline_ms_override_ = ms;
  }

  int write_calls() const {
    absl::MutexLock lock(mutex_);
    return write_calls_;
  }
  int poll_calls() const {
    absl::MutexLock lock(mutex_);
    return poll_calls_;
  }
  // Streams accepted and not yet ended -- each one is an operation this
  // destination is still holding open for its source.
  int open_streams() const {
    absl::MutexLock lock(mutex_);
    return static_cast<int>(active_reactors_.size());
  }
  int64_t requested_deadline_ms() const {
    absl::MutexLock lock(mutex_);
    return requested_deadline_ms_;
  }

 private:
  mutable absl::Mutex mutex_;
  std::vector<FakeServerReactor*> active_reactors_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> existing_hashes_ ABSL_GUARDED_BY(mutex_);
  ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse poll_response_
      ABSL_GUARDED_BY(mutex_);
  ::tpu_raiden::kv_cache::proto::WriteExistState exist_state_
      ABSL_GUARDED_BY(mutex_) =
          ::tpu_raiden::kv_cache::proto::WRITE_EXIST_STATE_UNSPECIFIED;
  ::grpc::Status write_status_ ABSL_GUARDED_BY(mutex_) = ::grpc::Status::OK;
  int write_calls_ ABSL_GUARDED_BY(mutex_) = 0;
  int poll_calls_ ABSL_GUARDED_BY(mutex_) = 0;
  int64_t requested_deadline_ms_ ABSL_GUARDED_BY(mutex_) = 0;
  int64_t granted_deadline_ms_override_ ABSL_GUARDED_BY(mutex_) = 0;
};

class RemoteWriteSourceTest : public StoreDiscoveryTest {
 protected:
  static constexpr int kCapacity = 8;

  std::unique_ptr<KVCacheStore> MakeStore(const RaidenId& id,
                                          bool with_registry = true) {
    return std::make_unique<KVCacheStore>(
        /*capacity=*/kCapacity,
        with_registry ? registry_address_ : std::string(), id,
        /*num_shards=*/1, /*shard_size_bytes=*/1024,
        /*store_server_ip=*/"127.0.0.1",
        /*raiden_controller_port=*/0);
  }

  // Registers one real worker so the source has a data plane to offer from;
  // the destination refuses an offer without one.
  void RegisterWorker(KVCacheStore& store) {
    worker_server_ = ::tpu_raiden::controller::CreateTestWorkerServer();
    transfer_mock_ = std::make_unique<
        ::tpu_raiden::controller::ShardAwareMockTransferManager>();
    worker_server_->service->SetTransferManager(
        ::tpu_raiden::KVManagerHolder(transfer_mock_.get()));

    ::tpu_raiden::core::controller::RaidenControllerClient client(
        store.raiden_controller_address());
    auto status =
        client.RegisterWorker("worker_0", worker_server_->server_address,
                              {{worker_server_->server_address, {}}});
    ASSERT_TRUE(status.ok()) << status.message();
  }

  // Puts `hashes` in `store` as host-resident, which is the precondition for
  // offering them.
  static void Populate(KVCacheStore& store, const RaidenId& id,
                       const std::vector<std::string>& hashes) {
    std::vector<RaidenBlockId> slices;
    for (size_t i = 0; i < hashes.size(); ++i) {
      slices.push_back(
          RaidenBlockId(id, static_cast<int>(i), BlockStatus::HOST));
    }
    // Insert(), not InsertResident(): these cases are about offering blocks to
    // a peer, and a remote save requires -- and consumes -- the caller's pin.
    ASSERT_TRUE(store.Insert(hashes, slices, /*on_host=*/true).ok());
  }

  // Reads PollSaveStatus() until the write leaves the pending set.
  // Returns {done, failed, existing, unregistered}.
  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>, std::vector<std::string>>
  AwaitWriteSettled(KVCacheStore& store, int attempts = 300) {
    for (int i = 0; i < attempts; ++i) {
      auto [done, failed, pending, existing, unregistered] =
          store.PollSaveStatus();
      if (pending.empty() && (!done.empty() || !failed.empty())) {
        return {done, failed, existing, unregistered};
      }
      absl::SleepFor(absl::Milliseconds(10));
    }
    return {{}, {}, {}, {}};
  }

  // Stands the fake up and publishes it under `dst` so the source resolves it
  // through the registry exactly as it would a real peer.
  void StartFakeDestination(const RaidenId& dst,
                            absl::string_view kv_pool_group = "",
                            int32_t evict_tier = 0) {
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(),
                             &port);
    builder.RegisterService(&fake_destination_);
    fake_destination_server_ = builder.BuildAndStart();
    ASSERT_NE(fake_destination_server_, nullptr);
    ASSERT_TRUE(client_
                    ->RegisterStore(dst, "127.0.0.1:" + std::to_string(port),
                                    /*controller_address=*/"",
                                    /*ttl=*/absl::ZeroDuration(), kv_pool_group,
                                    evict_tier)
                    .ok());
  }

  void TearDown() override {
    fake_destination_.CancelAllReactors();
    if (fake_destination_server_) {
      fake_destination_server_->Shutdown(std::chrono::system_clock::now() +
                                         std::chrono::milliseconds(200));
    }
    StoreDiscoveryTest::TearDown();
  }

  FakeDestinationService fake_destination_;
  std::unique_ptr<grpc::Server> fake_destination_server_;
  std::unique_ptr<::tpu_raiden::controller::TestWorkerServer> worker_server_;
  std::unique_ptr<::tpu_raiden::controller::ShardAwareMockTransferManager>
      transfer_mock_;
};

// The registry precondition, reported by the one component that knows: the
// backend, which is what resolves peers.
TEST_F(RemoteWriteSourceTest, RefusesWithoutAGlobalRegistry) {
  RaidenId src{"rw_src_noreg", "0", "kv", 0};
  auto store = MakeStore(src, /*with_registry=*/false);
  Populate(*store, src, {"a"});

  auto status = store->Save({"a"}, RaidenId{"rw_dst", "0", "kv", 0});
  EXPECT_TRUE(absl::IsFailedPrecondition(status)) << status.ToString();
  EXPECT_THAT(status.message(), ::testing::HasSubstr("registry"));
}

// A missing registration means gone, not still-starting: every store that
// exists with a registry is registered before its constructor returns. So
// this fails fast rather than retrying.
TEST_F(RemoteWriteSourceTest, AnUnresolvableDestinationFailsImmediately) {
  RaidenId src{"rw_src_unresolvable", "0", "kv", 0};
  auto store = MakeStore(src);
  Populate(*store, src, {"a"});

  auto status = store->Save({"a"}, RaidenId{"nobody", "0", "kv", 0});
  EXPECT_TRUE(absl::IsNotFound(status)) << status.ToString();

  auto [done, failed, pending, existing, unregistered] =
      store->PollSaveStatus();
  EXPECT_TRUE(pending.empty()) << "a rejected offer must not stay active";
}

TEST_F(RemoteWriteSourceTest, RefusesToOfferBlocksItDoesNotHold) {
  RaidenId src{"rw_src_missing", "0", "kv", 0};
  auto store = MakeStore(src);

  auto status = store->Save({"absent"}, RaidenId{"d", "0", "kv", 0});
  EXPECT_TRUE(absl::IsNotFound(status)) << status.ToString();
}

// A remote save consumes one caller pin on success, so it requires one on
// entry: both halves of Save() share one pin contract behind one name.
TEST_F(RemoteWriteSourceTest, RefusesToOfferAnUnpinnedBlock) {
  RaidenId src{"rw_src_unpinned", "0", "kv", 0};
  auto store = MakeStore(src);
  Populate(*store, src, {"a"});
  // Populate pins; drop it so the block is host-resident but unpinned.
  store->Release({"a"});
  ASSERT_EQ(store->GetPinCount("a"), 0);

  auto status = store->Save({"a"}, RaidenId{"rw_dst_unpinned", "0", "kv", 0});
  EXPECT_TRUE(absl::IsFailedPrecondition(status)) << status.ToString();
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("not pinned"));
}

// A remote save pulls bytes out of host DRAM, so a block that lives only in
// HBM is refused; it must be saved locally first.
TEST_F(RemoteWriteSourceTest, RefusesToOfferAnHbmOnlyBlock) {
  RaidenId src{"rw_src_hbm_only", "0", "kv", 0};
  auto store = MakeStore(src);
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(src, /*host_block_id=*/-1, /*device_block_id=*/0,
                    BlockStatus::HBM)};
  ASSERT_TRUE(store->Insert({"a"}, slices, /*on_host=*/false).ok());

  auto status = store->Save({"a"}, RaidenId{"rw_dst_hbm", "0", "kv", 0});
  EXPECT_TRUE(absl::IsFailedPrecondition(status)) << status.ToString();
  // The refused save consumed nothing.
  EXPECT_EQ(store->GetPinCount("a"), 1);
  store->Release({"a"});
}

TEST_F(RemoteWriteSourceTest, RefusesToOfferToItself) {
  RaidenId src{"rw_src_self", "0", "kv", 0};
  auto store = MakeStore(src);
  Populate(*store, src, {"a"});

  EXPECT_TRUE(absl::IsInvalidArgument(store->Save({"a"}, src)));
}

// All-exist: the destination already had everything. A success that moves no
// bytes and creates no operation.
TEST_F(RemoteWriteSourceTest, AllExistSettlesDoneWithoutATransfer) {
  RaidenId src{"rw_src_allexist", "0", "kv", 0};
  RaidenId dst{"rw_dst_allexist", "0", "kv", 0};
  auto src_store = MakeStore(src);
  auto dst_store = MakeStore(dst);
  RegisterWorker(*src_store);
  Populate(*src_store, src, {"a", "b"});
  ASSERT_TRUE(dst_store->backend()->InsertAllOrNothing(
      {"a", "b"}, {RaidenBlockId(dst, 5, BlockStatus::HOST),
                   RaidenBlockId(dst, 6, BlockStatus::HOST)}));

  ASSERT_TRUE(src_store->Save({"a", "b"}, dst).ok());

  auto [done, failed, pending, existing, unregistered] =
      src_store->PollSaveStatus();
  EXPECT_THAT(done, ::testing::UnorderedElementsAre("a", "b"));
  EXPECT_TRUE(failed.empty());
  EXPECT_TRUE(pending.empty());
  EXPECT_TRUE(existing.empty());
}

// Partial-exist: the destination held a subset. Reported as a failure with
// the list; the store does not reissue the difference on its own.
TEST_F(RemoteWriteSourceTest, PartialExistIsReportedAndNotRetried) {
  RaidenId src{"rw_src_partial", "0", "kv", 0};
  RaidenId dst{"rw_dst_partial", "0", "kv", 0};
  auto src_store = MakeStore(src);
  auto dst_store = MakeStore(dst);
  RegisterWorker(*src_store);
  Populate(*src_store, src, {"a", "b"});
  ASSERT_TRUE(dst_store->backend()->InsertAllOrNothing(
      {"a"}, {RaidenBlockId(dst, 5, BlockStatus::HOST)}));

  ASSERT_TRUE(src_store->Save({"a", "b"}, dst).ok());

  auto [done, failed, pending, existing, unregistered] =
      src_store->PollSaveStatus();
  EXPECT_TRUE(done.empty());
  EXPECT_THAT(failed, ::testing::UnorderedElementsAre("a", "b"));
  EXPECT_TRUE(pending.empty());
  EXPECT_THAT(existing, ::testing::UnorderedElementsAre("a"));

  // No hidden second offer: the destination still holds only what it had.
  EXPECT_EQ(dst_store->backend()->GetSize(), 1);
}

// The full source loop: offer, receive the verdict on the offer's own call,
// and release the internal pin. The transfer fails here (no workers on the
// destination); what matters is that the source settles and lets go.
TEST_F(RemoteWriteSourceTest, AnAcceptedOfferSettlesOnATerminalVerdict) {
  RaidenId src{"rw_src_accept", "0", "kv", 0};
  RaidenId dst{"rw_dst_accept", "0", "kv", 0};
  auto src_store = MakeStore(src);
  auto dst_store = MakeStore(dst);
  RegisterWorker(*src_store);
  Populate(*src_store, src, {"a", "b"});

  ASSERT_TRUE(src_store->Save({"a", "b"}, dst).ok());

  auto [done, failed, existing, unregistered] = AwaitWriteSettled(*src_store);
  EXPECT_TRUE(done.empty());
  EXPECT_THAT(failed, ::testing::UnorderedElementsAre("a", "b"));

  // The internal pin is gone and the hashes are no longer marked as writing,
  // so the same blocks can be offered again.
  EXPECT_TRUE(src_store->Save({"a", "b"}, dst).ok())
      << "the first operation never released its claim on these hashes";
  AwaitWriteSettled(*src_store);
}

// A hash already being offered cannot be offered again concurrently.
TEST_F(RemoteWriteSourceTest, RefusesASecondConcurrentOfferOfTheSameHash) {
  RaidenId src{"rw_src_concurrent", "0", "kv", 0};
  RaidenId dst{"rw_dst_concurrent", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::PENDING);
  fake_destination_.SetPollResponse(verdict);

  ASSERT_TRUE(src_store->Save({"a"}, dst).ok());
  auto second = src_store->Save({"a"}, dst);
  EXPECT_TRUE(absl::IsFailedPrecondition(second)) << second.ToString();
}

// STORED_UNREGISTERED (peer has the bytes but could not publish them) must
// be reported as failed, with the list, so the caller does not drop a copy
// that no peer can find.
TEST_F(RemoteWriteSourceTest, StoredUnregisteredIsFailedAndNamesTheBlocks) {
  RaidenId src{"rw_src_unreg", "0", "kv", 0};
  RaidenId dst{"rw_dst_unreg", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a", "b"});
  StartFakeDestination(dst);

  ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse verdict;
  verdict.set_state(::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::
                        STORED_UNREGISTERED);
  verdict.add_unregistered_hashes("a");
  verdict.add_unregistered_hashes("b");
  fake_destination_.SetPollResponse(verdict);

  ASSERT_TRUE(src_store->Save({"a", "b"}, dst).ok());
  auto [done, failed, existing, unregistered] = AwaitWriteSettled(*src_store);

  EXPECT_TRUE(done.empty())
      << "reporting this as done would let a caller free its only findable "
         "copy";
  EXPECT_THAT(failed, ::testing::UnorderedElementsAre("a", "b"));
  EXPECT_THAT(unregistered, ::testing::UnorderedElementsAre("a", "b"));
  EXPECT_TRUE(existing.empty());

  // The internal pin is released either way, so the caller can act on the
  // list -- including by offering the same blocks somewhere else.
  EXPECT_TRUE(src_store->Save({"a", "b"}, dst).ok());
  AwaitWriteSettled(*src_store);
}

// A COMMITTED verdict is reported as done.
TEST_F(RemoteWriteSourceTest, CommittedIsReportedAsDone) {
  RaidenId src{"rw_src_committed", "0", "kv", 0};
  RaidenId dst{"rw_dst_committed", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a", "b"});
  StartFakeDestination(dst);

  ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse verdict;
  verdict.set_state(
      ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::COMMITTED);
  verdict.add_committed_hashes("a");
  verdict.add_committed_hashes("b");
  fake_destination_.SetPollResponse(verdict);

  ASSERT_TRUE(src_store->Save({"a", "b"}, dst).ok());
  auto [done, failed, existing, unregistered] = AwaitWriteSettled(*src_store);

  EXPECT_THAT(done, ::testing::UnorderedElementsAre("a", "b"));
  EXPECT_TRUE(failed.empty());
  EXPECT_TRUE(unregistered.empty());
  EXPECT_TRUE(existing.empty());
}

// A refusal is an application answer, not a channel failure: the cached
// store client survives, so the next offer skips the resolve and reconnect.
TEST_F(RemoteWriteSourceTest, ARefusalKeepsTheStoreClient) {
  RaidenId src{"rw_src_refusal", "0", "kv", 0};
  RaidenId dst{"rw_dst_refusal", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse verdict;
  verdict.set_state(
      ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::COMMITTED);
  verdict.add_committed_hashes("a");
  fake_destination_.SetPollResponse(verdict);

  fake_destination_.SetWriteRemoteStatus(::grpc::Status(
      ::grpc::StatusCode::RESOURCE_EXHAUSTED, "no free landing blocks"));
  auto refused = src_store->Save({"a"}, dst);
  EXPECT_TRUE(absl::IsResourceExhausted(refused)) << refused.ToString();

  // Unregistering the peer makes re-resolution impossible, so the second
  // offer can only reach the destination through the client kept from the
  // first one.
  ASSERT_TRUE(client_->UnregisterStore(dst).ok());
  fake_destination_.SetWriteRemoteStatus(::grpc::Status::OK);
  EXPECT_TRUE(src_store->Save({"a"}, dst).ok());
  EXPECT_EQ(fake_destination_.write_calls(), 2);
  AwaitWriteSettled(*src_store);
}

// A channel-level failure drops the cached store client: the peer may be
// back under a new address, and only a fresh resolve finds it.
TEST_F(RemoteWriteSourceTest, ATransportErrorInvalidatesTheStoreClient) {
  RaidenId src{"rw_src_transport", "0", "kv", 0};
  RaidenId dst{"rw_dst_transport", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a", "b"});
  StartFakeDestination(dst);

  ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse verdict;
  verdict.set_state(
      ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::COMMITTED);
  verdict.add_committed_hashes("a");
  verdict.add_committed_hashes("b");
  fake_destination_.SetPollResponse(verdict);

  // A first offer establishes the store client for dst.
  ASSERT_TRUE(src_store->Save({"a"}, dst).ok());
  AwaitWriteSettled(*src_store);

  fake_destination_server_->Shutdown();
  auto down = src_store->Save({"b"}, dst);
  EXPECT_TRUE(absl::IsUnavailable(down)) << down.ToString();

  // The destination comes back under the same identity on a fresh port. A
  // client kept across the transport error would still dial the dead
  // one; this offer succeeds only by re-resolving.
  ASSERT_TRUE(client_->UnregisterStore(dst).ok());
  StartFakeDestination(dst);
  EXPECT_TRUE(src_store->Save({"b"}, dst).ok());
  AwaitWriteSettled(*src_store);
}

// UNKNOWN (record aged out, or the destination restarted) lands as a plain
// failure with no annotations.
TEST_F(RemoteWriteSourceTest, UnknownIsReportedAsAPlainFailure) {
  RaidenId src{"rw_src_unknown", "0", "kv", 0};
  RaidenId dst{"rw_dst_unknown", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse verdict;
  verdict.set_state(
      ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::UNKNOWN);
  fake_destination_.SetPollResponse(verdict);

  ASSERT_TRUE(src_store->Save({"a"}, dst).ok());
  auto [done, failed, existing, unregistered] = AwaitWriteSettled(*src_store);

  EXPECT_TRUE(done.empty());
  EXPECT_THAT(failed, ::testing::UnorderedElementsAre("a"));
  EXPECT_TRUE(unregistered.empty());
}

// The source asks for HOLD minus the margin, so the destination's grant
// cannot outlive the source's pin.
TEST_F(RemoteWriteSourceTest, TheOfferAsksForLessThanTheSourceWillHold) {
  RaidenId src{"rw_src_deadline", "0", "kv", 0};
  RaidenId dst{"rw_dst_deadline", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse verdict;
  verdict.set_state(
      ::tpu_raiden::kv_cache::proto::PollWriteRemoteResponse::COMMITTED);
  verdict.add_committed_hashes("a");
  fake_destination_.SetPollResponse(verdict);

  ASSERT_TRUE(src_store->Save({"a"}, dst).ok());
  AwaitWriteSettled(*src_store);

  EXPECT_EQ(fake_destination_.write_calls(), 1);
  // Default HOLD is 30s and the margin 5s.
  EXPECT_EQ(fake_destination_.requested_deadline_ms(),
            absl::ToInt64Milliseconds(absl::Seconds(25)));
}

// A commit settles with zero PollWriteRemote calls: the verdict arrives on
// the offer's own call.
TEST_F(RemoteWriteSourceTest, DestinationCommitsSettlesWithZeroPollCalls) {
  RaidenId src{"rw_src_zero_poll", "0", "kv", 0};
  RaidenId dst{"rw_dst_zero_poll", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::COMMITTED);
  verdict.add_committed_hashes("a");
  fake_destination_.SetPollResponse(verdict);

  ASSERT_TRUE(src_store->Save({"a"}, dst).ok());
  auto [done, failed, existing, unregistered] = AwaitWriteSettled(*src_store);

  EXPECT_THAT(done, ::testing::UnorderedElementsAre("a"));
  EXPECT_TRUE(failed.empty());
  EXPECT_EQ(fake_destination_.write_calls(), 1);
  EXPECT_EQ(fake_destination_.poll_calls(), 0)
      << "Destination commit must be received reactively over the WriteRemote "
         "stream without any PollWriteRemote RPCs";
}

// The STORED_UNREGISTERED and PARTIAL_EXIST hash lists survive the streamed
// result byte-for-byte.
TEST_F(RemoteWriteSourceTest,
       StoredUnregisteredAndPartialExistSurviveStreamByteForByte) {
  RaidenId src{"rw_src_unreg_stream", "0", "kv", 0};
  RaidenId dst{"rw_dst_unreg_stream", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"x", "y"});
  StartFakeDestination(dst);

  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::STORED_UNREGISTERED);
  verdict.add_unregistered_hashes("x");
  verdict.add_unregistered_hashes("y");
  fake_destination_.SetPollResponse(verdict);

  ASSERT_TRUE(src_store->Save({"x", "y"}, dst).ok());
  auto [done, failed, existing, unregistered] = AwaitWriteSettled(*src_store);

  EXPECT_TRUE(done.empty());
  EXPECT_THAT(failed, ::testing::UnorderedElementsAre("x", "y"));
  EXPECT_THAT(unregistered, ::testing::UnorderedElementsAre("x", "y"));
}

// A stream broken mid-flight keeps the pin and recovers the true verdict
// with one waiting PollWriteRemote, triggered by the break itself.
TEST_F(RemoteWriteSourceTest, StreamBrokenMidFlightRecoversViaWaitingPoll) {
  RaidenId src{"rw_src_stream_break", "0", "kv", 0};
  RaidenId dst{"rw_dst_stream_break", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  // Keep the stream pending initially so the ack is read but no result is sent yet.
  proto::PollWriteRemoteResponse pending_verdict;
  pending_verdict.set_state(proto::PollWriteRemoteResponse::PENDING);
  fake_destination_.SetPollResponse(pending_verdict);

  ASSERT_TRUE(src_store->Save({"a"}, dst).ok());
  // The offer was accepted and is in flight. Internal pin is held.
  EXPECT_EQ(src_store->GetPinCount("a"), 2);

  // Set the eventual verdict that the destination will report when asked.
  proto::PollWriteRemoteResponse committed_verdict;
  committed_verdict.set_state(proto::PollWriteRemoteResponse::COMMITTED);
  committed_verdict.add_committed_hashes("a");
  fake_destination_.SetPollRecoveryResponse(committed_verdict);

  // Abruptly break the active stream (simulating network drop / reset).
  fake_destination_.BreakActiveStreams(::grpc::Status(
      ::grpc::StatusCode::UNAVAILABLE, "Stream broken mid-flight"));

  // The source's OnDone catches the break and issues PollWriteRemote(wait_ms),
  // recovering the true verdict without a polling thread or timer loop.
  auto [done, failed, existing, unregistered] = AwaitWriteSettled(*src_store);

  EXPECT_THAT(done, ::testing::UnorderedElementsAre("a"));
  EXPECT_TRUE(failed.empty());
  EXPECT_EQ(fake_destination_.write_calls(), 1);
  EXPECT_EQ(fake_destination_.poll_calls(), 1)
      << "Recovery must issue exactly one PollWriteRemote call";
}

// A stream that breaks during teardown issues no recovery ask: the
// destructor's take wins the operation, so the break finds nothing to act on.
TEST_F(RemoteWriteSourceTest,
       StreamBrokenDuringTeardownDoesNotIssueRecoveryPoll) {
  RaidenId src{"rw_src_break_dtor", "0", "kv", 0};
  RaidenId dst{"rw_dst_break_dtor", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  proto::PollWriteRemoteResponse pending_verdict;
  pending_verdict.set_state(proto::PollWriteRemoteResponse::PENDING);
  fake_destination_.SetPollResponse(pending_verdict);

  ASSERT_TRUE(src_store->Save({"a"}, dst).ok());
  EXPECT_EQ(src_store->InFlightRemoteWritesCountForTesting(), 1);

  // Destroy the store while stream is in flight: destructor takes and cancels.
  src_store.reset();

  // The stream breaks because server/client shuts down.
  fake_destination_.BreakActiveStreams(::grpc::Status(
      ::grpc::StatusCode::UNAVAILABLE, "Stream broken"));

  // OnDone must not issue a recovery poll because the operation was already taken during teardown.
  EXPECT_EQ(fake_destination_.poll_calls(), 0)
      << "Teardown must not leave an orphaned recovery poll in flight";
}

// Settling must claim the operation, not work off a copy: two settles of one
// operation would release the internal pin twice and report every hash
// twice.
TEST_F(RemoteWriteSourceTest, ConcurrentPollsSettleAnOfferExactlyOnce) {
  RaidenId src{"rw_src_race", "0", "kv", 0};
  RaidenId dst{"rw_dst_race", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a", "b"});
  StartFakeDestination(dst);

  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::COMMITTED);
  verdict.add_committed_hashes("a");
  verdict.add_committed_hashes("b");
  fake_destination_.SetPollResponse(verdict);

  ASSERT_TRUE(src_store->Save({"a", "b"}, dst).ok());

  // Each thread drains PollSaveStatus and pools what it sees. The invariant:
  // every hash appears exactly once across all of them.
  absl::Mutex settled_mu;
  std::vector<std::string> settled;
  std::atomic<bool> stop{false};
  std::vector<std::thread> drivers;
  for (int t = 0; t < 4; ++t) {
    drivers.emplace_back([&]() {
      while (!stop.load(std::memory_order_relaxed)) {
        auto [done, failed, pending, existing, unregistered] =
            src_store->PollSaveStatus();
        if (!done.empty() || !failed.empty()) {
          absl::MutexLock lock(settled_mu);
          settled.insert(settled.end(), done.begin(), done.end());
          settled.insert(settled.end(), failed.begin(), failed.end());
        }
      }
    });
  }

  // Long enough for a duplicate settle to show up, not just the first one.
  for (int i = 0; i < 300; ++i) {
    {
      absl::MutexLock lock(settled_mu);
      if (settled.size() >= 2) break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  absl::SleepFor(absl::Milliseconds(200));

  stop.store(true, std::memory_order_relaxed);
  for (auto& driver : drivers) driver.join();

  absl::MutexLock settled_lock(settled_mu);

  EXPECT_THAT(settled, ::testing::UnorderedElementsAre("a", "b"))
      << "the offer was settled more than once";
  // Both pins are spent by the success. The double-settle detector is the
  // `settled` list above, not this count: Unpin clamps at zero.
  EXPECT_EQ(src_store->GetPinCount("a"), 0);
  EXPECT_EQ(src_store->GetPinCount("b"), 0);
}

// A successful remote save consumes the caller's pin, exactly as a local one
// does.
TEST_F(RemoteWriteSourceTest, ASuccessfulRemoteSaveConsumesTheCallerPin) {
  RaidenId src{"rw_src_unpin_ok", "0", "kv", 0};
  RaidenId dst{"rw_dst_unpin_ok", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);
  ASSERT_EQ(src_store->GetPinCount("a"), 1);

  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::COMMITTED);
  verdict.add_committed_hashes("a");
  fake_destination_.SetPollResponse(verdict);

  ASSERT_TRUE(src_store->Save({"a"}, dst).ok());
  auto [done, failed, existing, unregistered] = AwaitWriteSettled(*src_store);
  ASSERT_THAT(done, ::testing::ElementsAre("a"));

  EXPECT_EQ(src_store->GetPinCount("a"), 0)
      << "a successful remote save must consume the caller's pin";
}

// A failed remote save keeps the caller's pin; retrying or releasing is the
// caller's decision.
TEST_F(RemoteWriteSourceTest, AFailedRemoteSaveKeepsTheCallerPin) {
  RaidenId src{"rw_src_unpin_fail", "0", "kv", 0};
  RaidenId dst{"rw_dst_unpin_fail", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::FAILED);
  fake_destination_.SetPollResponse(verdict);

  ASSERT_TRUE(src_store->Save({"a"}, dst).ok());
  auto [done, failed, existing, unregistered] = AwaitWriteSettled(*src_store);
  ASSERT_THAT(failed, ::testing::ElementsAre("a"));

  EXPECT_EQ(src_store->GetPinCount("a"), 1)
      << "a failed remote save must leave the caller's pin alone";
}

// The all-exist path settles synchronously inside Save with no verdict
// message, and still consumes the caller's pin.
TEST_F(RemoteWriteSourceTest, AnAllExistRemoteSaveConsumesTheCallerPinToo) {
  RaidenId src{"rw_src_unpin_allexist", "0", "kv", 0};
  RaidenId dst{"rw_dst_unpin_allexist", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a", "b"});
  StartFakeDestination(dst);
  fake_destination_.SetWriteExistState(proto::WRITE_ALL_EXIST);

  ASSERT_EQ(src_store->GetPinCount("a"), 1);
  ASSERT_TRUE(src_store->Save({"a", "b"}, dst).ok());

  // Settled inside the call: no poll was needed to get here.
  EXPECT_EQ(src_store->GetPinCount("a"), 0);
  EXPECT_EQ(src_store->GetPinCount("b"), 0);

  auto [done, failed, pending, existing, unregistered] =
      src_store->PollSaveStatus();
  EXPECT_THAT(done, ::testing::UnorderedElementsAre("a", "b"));
  EXPECT_TRUE(pending.empty());
}

// A store destroyed with an offer outstanding releases its internal pin: the
// backend can outlive the store, and nothing else would release it.
TEST_F(RemoteWriteSourceTest, DestroyingAStoreMidOfferReleasesItsInternalPin) {
  RaidenId src{"rw_src_dtor_pin", "0", "kv", 0};
  RaidenId dst{"rw_dst_dtor_pin", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  // The destination never reaches a verdict, so the offer stays outstanding.
  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::PENDING);
  fake_destination_.SetPollResponse(verdict);

  ASSERT_TRUE(src_store->Save({"a"}, dst).ok());
  // Populate's pin plus the offer's internal one.
  EXPECT_EQ(src_store->GetPinCount("a"), 2);

  // Outlive the store, exactly as a caller sharing a backend would.
  std::shared_ptr<KVCacheStoreBackend> backend = src_store->backend();
  src_store.reset();

  EXPECT_EQ(backend->GetPinCount("a"), 1)
      << "the offer's internal pin outlived the store that took it";
}

// Destruction must not wait for outstanding offers: that could stall ~30s
// behind a dead peer.
TEST_F(RemoteWriteSourceTest, DestroyingAStoreMidOfferDoesNotWaitForTheHold) {
  RaidenId src{"rw_src_dtor_fast", "0", "kv", 0};
  RaidenId dst{"rw_dst_dtor_fast", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::PENDING);
  fake_destination_.SetPollResponse(verdict);

  ASSERT_TRUE(src_store->Save({"a"}, dst).ok());
  {
    auto [done, failed, pending, existing, unregistered] =
        src_store->PollSaveStatus();
    ASSERT_THAT(pending, ::testing::ElementsAre("a"));
  }

  const absl::Time before = absl::Now();
  src_store.reset();
  const absl::Duration took = absl::Now() - before;

  // The HOLD is 30s; anything near it means the drain waited.
  EXPECT_LT(took, absl::Seconds(10))
      << "destruction blocked on the remote write's hold window";
}

// An offer that dies with DEADLINE_EXCEEDED before any ack releases its pin
// and record right there; no settle path ever fires for an ack-less
// operation, so a kept pin would leak.
TEST_F(RemoteWriteSourceTest, AckDeadlineReleasesThePinAndTheRecord) {
  RaidenId src{"rw_src_ack_fail", "0", "kv", 0};
  RaidenId dst{"rw_dst_ack_fail", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  // Destination times out before ack (simulating lost ack / slow peer).
  fake_destination_.SetWriteRemoteStatus(::grpc::Status(
      ::grpc::StatusCode::DEADLINE_EXCEEDED, "deadline exceeded before ack"));

  auto status = src_store->Save({"a"}, dst);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::IsDeadlineExceeded(status)) << status;

  // The offer is fully undone: only the caller's pin remains, no operation
  // stays on the books, and the hash may be offered again immediately.
  EXPECT_EQ(src_store->GetPinCount("a"), 1);
  EXPECT_EQ(src_store->InFlightRemoteWritesCountForTesting(), 0);

  fake_destination_.SetWriteRemoteStatus(::grpc::Status::OK);
  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::COMMITTED);
  verdict.add_committed_hashes("a");
  fake_destination_.SetPollResponse(verdict);
  EXPECT_TRUE(src_store->Save({"a"}, dst).ok());
  AwaitWriteSettled(*src_store);
}

// A lost answer undoes the offer completely (pin restored, nothing
// outstanding, no verdict filed), so the sweep can re-offer to the next
// target right away. This pins the KNOWN GAP contract stated in SaveRemote.
TEST_F(RemoteWriteSourceTest, ALostAnswerUndoesTheOfferAndAllowsARetry) {
  RaidenId src{"rw_src_lost_ack", "0", "kv", 0};
  RaidenId dst{"rw_dst_lost_ack", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  const int pinned_before = src_store->backend()->GetPinCount("a");

  fake_destination_.SetWriteRemoteStatus(
      ::grpc::Status(::grpc::StatusCode::UNAVAILABLE, "connection reset"));
  auto lost = src_store->Save({"a"}, dst);
  EXPECT_TRUE(absl::IsUnavailable(lost)) << lost.ToString();

  // Nothing outstanding, and no verdict filed: the return value was the
  // report.
  auto [done, failed, pending, existing, unregistered] =
      src_store->PollSaveStatus();
  EXPECT_TRUE(pending.empty()) << "an offer that failed stayed active";
  EXPECT_TRUE(done.empty());
  EXPECT_TRUE(failed.empty());
  EXPECT_EQ(src_store->backend()->GetPinCount("a"), pinned_before)
      << "the internal pin outlived the offer it belonged to";

  // The hash is free to be offered again straight away.
  fake_destination_.SetWriteRemoteStatus(::grpc::Status::OK);
  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::COMMITTED);
  verdict.add_committed_hashes("a");
  fake_destination_.SetPollResponse(verdict);
  EXPECT_TRUE(src_store->Save({"a"}, dst).ok());
  AwaitWriteSettled(*src_store);
}

// Sets RAIDEN_REMOTE_WRITE_HOLD_S for one test and restores it after: the
// default 30s HOLD is too long to wait for in a test.
class ScopedRemoteWriteHold {
 public:
  explicit ScopedRemoteWriteHold(const char* seconds) {
    const char* previous = std::getenv("RAIDEN_REMOTE_WRITE_HOLD_S");
    if (previous != nullptr) {
      had_previous_ = true;
      previous_ = previous;
    }
    setenv("RAIDEN_REMOTE_WRITE_HOLD_S", seconds, /*overwrite=*/1);
  }
  ~ScopedRemoteWriteHold() {
    if (had_previous_) {
      setenv("RAIDEN_REMOTE_WRITE_HOLD_S", previous_.c_str(), /*overwrite=*/1);
    } else {
      unsetenv("RAIDEN_REMOTE_WRITE_HOLD_S");
    }
  }

 private:
  bool had_previous_ = false;
  std::string previous_;
};

// If the destination grants a deadline LONGER than the source's HOLD, the
// call still dies at the HOLD -- but the blocks must stay protected for the
// grant. The deciding clock is hold_expiry, which the ack extends, not the
// call's fixed deadline.
TEST_F(RemoteWriteSourceTest, AGrantPastTheHoldKeepsTheBlocksForTheGrant) {
  ScopedRemoteWriteHold hold("6");
  RaidenId src{"rw_src_grant", "0", "kv", 0};
  RaidenId dst{"rw_dst_grant", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  // Grant far past the six-second HOLD, then stay silent: the call dies at
  // its own deadline with the operation still live on the destination.
  fake_destination_.SetGrantedDeadlineMs(20000);

  ASSERT_TRUE(src_store->Save({"a"}, dst).ok());
  EXPECT_EQ(src_store->GetPinCount("a"), 2);

  // The HOLD runs out. What must NOT happen is the operation being settled
  // here; what must happen is one ask, for what is left of the grant.
  for (int i = 0; i < 1500 && fake_destination_.poll_calls() == 0; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_EQ(fake_destination_.poll_calls(), 1)
      << "the call died at the hold and the operation was settled there, "
         "while the destination still had its grant to run";
  EXPECT_EQ(src_store->GetPinCount("a"), 2)
      << "the source stopped protecting blocks the destination was granted "
         "more time to read";

  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::COMMITTED);
  verdict.add_committed_hashes("a");
  fake_destination_.SetPollResponse(verdict);

  auto [done, failed, existing, unregistered] =
      AwaitWriteSettled(*src_store, /*attempts=*/2000);
  EXPECT_THAT(done, ::testing::UnorderedElementsAre("a"));
  EXPECT_TRUE(failed.empty());
}

// The ordinary end of an unanswered offer: the HOLD runs out, the source
// releases the pin and reports the batch failed, and asks nothing.
TEST_F(RemoteWriteSourceTest, AHoldThatRunsOutReleasesThePinAndFailsTheBatch) {
  ScopedRemoteWriteHold hold("6");
  RaidenId src{"rw_src_holdout", "0", "kv", 0};
  RaidenId dst{"rw_dst_holdout", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  // Accepted, and then nothing: no result, and no grant to outlive the hold.
  ASSERT_TRUE(src_store->Save({"a"}, dst).ok());
  EXPECT_EQ(src_store->GetPinCount("a"), 2);

  auto [done, failed, existing, unregistered] =
      AwaitWriteSettled(*src_store, /*attempts=*/2000);
  EXPECT_TRUE(done.empty());
  EXPECT_THAT(failed, ::testing::UnorderedElementsAre("a"));
  // The internal pin is gone; the caller's stays, because the save failed.
  EXPECT_EQ(src_store->GetPinCount("a"), 1);
  EXPECT_EQ(fake_destination_.poll_calls(), 0)
      << "nothing should have been asked: the hold was over";
}

// Teardown cancels the calls it abandons (without waiting for them), so the
// destination is not left holding a call nobody will answer for the rest of
// the hold window.
TEST_F(RemoteWriteSourceTest, TeardownEndsTheCallsItAbandons) {
  ScopedRemoteWriteHold hold("6");
  RaidenId src{"rw_src_cancel", "0", "kv", 0};
  RaidenId dst{"rw_dst_cancel", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  // Accepted, and then silent: the destination sits on the stream, so within
  // the two seconds observed below, the call ending can only be the
  // cancellation -- the six-second hold has not run out yet.
  ASSERT_TRUE(src_store->Save({"a"}, dst).ok());
  ASSERT_EQ(fake_destination_.open_streams(), 1);

  src_store.reset();

  for (int i = 0; i < 200 && fake_destination_.open_streams() != 0; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_EQ(fake_destination_.open_streams(), 0)
      << "the destination is still holding the call, and its landing blocks "
         "with it, for a source that no longer exists";
}

// The recovery ask is asynchronous, so a store can be destroyed while the
// destination sits on the ask, and the internal pin is still released.
TEST_F(RemoteWriteSourceTest, TeardownDoesNotWaitForARecoveryAskToBeAnswered) {
  RaidenId src{"rw_src_recover_async", "0", "kv", 0};
  RaidenId dst{"rw_dst_recover_async", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  // Accepted, then the stream drops -- and the destination then sits on the
  // recovery ask rather than answering it (the fake's poll parks while its
  // response is unset).
  ASSERT_TRUE(src_store->Save({"a"}, dst).ok());
  fake_destination_.BreakActiveStreams(
      ::grpc::Status(::grpc::StatusCode::UNAVAILABLE, "stream lost"));
  for (int i = 0; i < 500 && fake_destination_.poll_calls() == 0; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_EQ(fake_destination_.poll_calls(), 1)
      << "the stream broke and nothing asked the destination about it";

  std::shared_ptr<KVCacheStoreBackend> backend = src_store->backend();
  absl::Notification destroyed;
  std::thread reaper([&]() {
    src_store.reset();
    destroyed.Notify();
  });
  EXPECT_TRUE(destroyed.WaitForNotificationWithTimeout(absl::Seconds(10)))
      << "destroying the store waited for a peer that has not answered";

  // Unpark the fake's poll so its server thread can end, then collect the
  // reaper -- after the assertion, so a hang is reported rather than waited
  // out.
  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::COMMITTED);
  verdict.add_committed_hashes("a");
  fake_destination_.SetPollResponse(verdict);
  reaper.join();

  // Teardown released the internal pin. The caller's is still there: the save
  // did not succeed, and only success consumes it.
  EXPECT_EQ(backend->GetPinCount("a"), 1);
}

// TakeRemoteWrite is the atomic claim function: across multiple concurrent
// callers or threads, exactly one caller receives the RemoteWriteState and all
// others receive std::nullopt.
TEST_F(RemoteWriteSourceTest, TakeRemoteWriteSettlesExactlyOnce) {
  RaidenId src{"rw_src_take_once", "0", "kv", 0};
  RaidenId dst{"rw_dst_take_once", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::PENDING);
  fake_destination_.SetPollResponse(verdict);

  ASSERT_TRUE(src_store->Save({"a"}, dst).ok());
  ASSERT_EQ(src_store->InFlightRemoteWritesCountForTesting(), 1);

  constexpr int kNumThreads = 10;
  std::vector<std::thread> threads;
  std::atomic<int> success_count{0};
  using ResultType = decltype(KVCacheStoreTest::TakeRemoteWrite(*src_store, 1));
  std::vector<ResultType> results(kNumThreads);

  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&, i]() {
      results[i] = KVCacheStoreTest::TakeRemoteWrite(*src_store, 1);
      if (results[i].has_value()) {
        success_count.fetch_add(1);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(success_count.load(), 1);
  EXPECT_EQ(src_store->InFlightRemoteWritesCountForTesting(), 0);

  // Settle the single taken state.
  for (auto& res : results) {
    if (res.has_value()) {
      KVCacheStoreTest::OnWriteRemoteVerdict(*src_store, std::move(*res),
                                             /*succeeded=*/false, {});
    }
  }
  EXPECT_EQ(src_store->GetPinCount("a"), 1);
}

// A destination that vanished after registering. The source must get a prompt
// error rather than waiting, and must drop the cached client so a restarted
// peer is reachable.
TEST_F(RemoteWriteSourceTest, AStaleButRegisteredDestinationFailsPromptly) {
  RaidenId src{"rw_src_stale", "0", "kv", 0};
  RaidenId dst{"rw_dst_stale", "0", "kv", 0};
  auto src_store = MakeStore(src);
  RegisterWorker(*src_store);
  Populate(*src_store, src, {"a"});

  {
    auto dst_store = MakeStore(dst);
    ASSERT_FALSE(dst_store->store_server_address().empty());
  }
  // The registration outlives the store: store entries never expire, and
  // nothing unpublishes on the way out except the store itself, which has
  // already gone.
  auto status = src_store->Save({"a"}, dst);
  EXPECT_FALSE(status.ok())
      << "offering to a dead peer should fail rather than hang";
}

class EvictSweepTest : public RemoteWriteSourceTest {
 protected:
  // A store with the sweep on. With the 0.5/0.75 watermarks, pressure
  // starts below capacity/2 free blocks and relief comes at 3/4 free.
  absl::StatusOr<std::unique_ptr<KVCacheStore>> MakeSweepStore(
      const RaidenId& id, absl::string_view kv_pool_group,
      size_t capacity = kCapacity) {
    BackendConfig config;
    config.type = "HostOffloadBackend";
    config.capacity = capacity;
    config.raiden_id = id;
    config.global_registry_address = registry_address_;
    config.kv_pool_group = std::string(kv_pool_group);
    config.monitor_config.enable = true;
    config.monitor_config.enable_evict_sweep = true;
    config.monitor_config.evict_sweep_period = absl::Milliseconds(200);
    config.monitor_config.evict_low_watermark = 0.5;
    config.monitor_config.evict_high_watermark = 0.75;
    return KVCacheStore::Create(config, capacity, registry_address_, id,
                                /*num_shards=*/1,
                                /*shard_size_bytes=*/1024,
                                /*store_server_ip=*/"127.0.0.1");
  }

  // Fills `store` with cold (unpinned) host blocks whose ids really come from
  // the block manager, so the free-block count the sweep watches drops and
  // eviction raises it back.
  void PopulateCold(KVCacheStore& store, const RaidenId& id,
                    const std::vector<std::string>& hashes) {
    auto ids_or = store.raiden_controller()->AllocateBlockIds(hashes.size());
    ASSERT_TRUE(ids_or.ok()) << ids_or.status().ToString();
    std::vector<RaidenBlockId> slices;
    for (size_t i = 0; i < hashes.size(); ++i) {
      slices.push_back(RaidenBlockId(id, (*ids_or)[i], BlockStatus::HOST));
    }
    // Insert() pins; the sweep only takes unpinned blocks, so release.
    ASSERT_TRUE(store.Insert(hashes, slices, /*on_host=*/true).ok());
    store.Release(hashes);
  }

  // The same, but KEEPING the pin Insert granted: an application remote save
  // requires the caller's pin and consumes it on success.
  void PopulatePinned(KVCacheStore& store, const RaidenId& id,
                      const std::vector<std::string>& hashes) {
    auto ids_or = store.raiden_controller()->AllocateBlockIds(hashes.size());
    ASSERT_TRUE(ids_or.ok()) << ids_or.status().ToString();
    std::vector<RaidenBlockId> slices;
    for (size_t i = 0; i < hashes.size(); ++i) {
      slices.push_back(RaidenBlockId(id, (*ids_or)[i], BlockStatus::HOST));
    }
    ASSERT_TRUE(store.Insert(hashes, slices, /*on_host=*/true).ok());
  }

  // The sweep runs on the monitor's thread; wait until it has raised the free
  // count to `expected_free`. Free blocks, not cache size, because eviction
  // erases the cache entry a beat before the block id is deallocated.
  void AwaitSweepFreed(KVCacheStore& store, int expected_free) {
    for (int i = 0; i < 1000; ++i) {
      if (store.raiden_controller()->block_manager()->num_free_blocks() ==
          expected_free) {
        break;
      }
      absl::SleepFor(absl::Milliseconds(10));
    }
    EXPECT_EQ(store.raiden_controller()->block_manager()->num_free_blocks(),
              expected_free);
  }

  // How many of `hashes` the store still resolves locally.
  static size_t CountResident(KVCacheStore& store,
                              const std::vector<std::string>& hashes) {
    size_t resident = 0;
    for (const std::string& hash : hashes) {
      auto found = store.backend()->Lookup({hash});
      if (found.ok() && found->size() == 1) {
        ++resident;
      }
    }
    return resident;
  }
};

// The main path: pressure starts an episode, the sweep fetches a same-group
// higher-tier target from the registry and demotes LRU-cold batches to it
// until the high watermark, freeing the local copies.
TEST_F(EvictSweepTest, DemotesColdBlocksToAPlacementTarget) {
  RaidenId src{"sweep_src", "0", "kv", 0};
  RaidenId dst{"sweep_dst", "0", "kv", 0};
  // Large enough that reaching the high watermark takes several batches of
  // the built-in batch cap (128).
  auto store_or = MakeSweepStore(src, "sweepgroup", /*capacity=*/600);
  ASSERT_TRUE(store_or.ok()) << store_or.status().ToString();
  KVCacheStore& store = **store_or;
  ASSERT_EQ(store.raiden_controller()->block_manager()->total_blocks(), 600);

  StartFakeDestination(dst, "sweepgroup", /*evict_tier=*/1);
  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::COMMITTED);
  fake_destination_.SetPollResponse(verdict);

  // 500 of 600 blocks in use: free ratio 1/6, below the 0.5 low watermark.
  std::vector<std::string> hashes;
  for (int i = 0; i < 500; ++i) {
    hashes.push_back(absl::StrCat("h", i));
  }
  PopulateCold(store, src, hashes);

  // The deficit to the 0.75 high watermark is 350 blocks: batches of
  // 128 + 128 + 94, each demoted and freed.
  AwaitSweepFreed(store, 450);
  EXPECT_EQ(fake_destination_.write_calls(), 3);
  EXPECT_EQ(store.backend()->GetSize(), 150);
  EXPECT_EQ(CountResident(store, hashes), 150);
}

// A target that cannot be reached is dropped for the episode and the same
// batch goes to the next target in the ranking.
TEST_F(EvictSweepTest, SkipsAnUnreachableTargetForTheNextOne) {
  RaidenId src{"sweep_src_skip", "0", "kv", 0};
  RaidenId dead{"sweep_dead", "0", "kv", 0};
  RaidenId dst{"sweep_dst_skip", "0", "kv", 0};
  auto store_or = MakeSweepStore(src, "skipgroup");
  ASSERT_TRUE(store_or.ok()) << store_or.status().ToString();
  KVCacheStore& store = **store_or;

  // A dead peer, ranked first: registered at tier 1 with the most reported
  // free blocks, but nothing listens on its address.
  ASSERT_TRUE(client_
                  ->RegisterStore(dead, "127.0.0.1:1",
                                  /*controller_address=*/"",
                                  /*ttl=*/absl::ZeroDuration(), "skipgroup",
                                  /*evict_tier=*/1)
                  .ok());
  global_registry::StoreStatus roomy;
  roomy.set_free_blocks(1000);
  ASSERT_TRUE(client_->Heartbeat(dead, roomy).ok());
  StartFakeDestination(dst, "skipgroup", /*evict_tier=*/1);
  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::COMMITTED);
  fake_destination_.SetPollResponse(verdict);

  PopulateCold(store, src, {"a", "b", "c", "d", "e", "f"});

  // The offer to the dead peer fails, it is dropped, and the batch lands on
  // the live target instead.
  AwaitSweepFreed(store, 6);
  EXPECT_EQ(fake_destination_.write_calls(), 1);
  EXPECT_EQ(store.backend()->GetSize(), 2);
}

// A target that accepts but then fails the transfer is dropped too; with no
// target left, the next step falls back to dropping the blocks locally.
TEST_F(EvictSweepTest, ATransferFailureFallsBackToLocalDrop) {
  RaidenId src{"sweep_src_fail", "0", "kv", 0};
  RaidenId dst{"sweep_dst_fail", "0", "kv", 0};
  auto store_or = MakeSweepStore(src, "failgroup");
  ASSERT_TRUE(store_or.ok()) << store_or.status().ToString();
  KVCacheStore& store = **store_or;

  StartFakeDestination(dst, "failgroup", /*evict_tier=*/1);
  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::FAILED);
  fake_destination_.SetPollResponse(verdict);

  PopulateCold(store, src, {"a", "b", "c", "d", "e", "f"});

  // One offer is accepted and fails; the target is dropped and the free
  // count still recovers, through local drops.
  AwaitSweepFreed(store, 6);
  EXPECT_EQ(fake_destination_.write_calls(), 1);
  EXPECT_EQ(store.backend()->GetSize(), 2);
}

// The bottom tier has no placement targets; under pressure the sweep drops
// cold blocks locally instead -- the same discard the allocation path would
// be forced into later, done early.
TEST_F(EvictSweepTest, DropsLocallyWhenThereAreNoTargets) {
  RaidenId src{"sweep_src_bottom", "0", "kv", 0};
  auto store_or = MakeSweepStore(src, "sweepgroup_bottom");
  ASSERT_TRUE(store_or.ok()) << store_or.status().ToString();
  KVCacheStore& store = **store_or;

  PopulateCold(store, src, {"a", "b", "c", "d", "e", "f"});

  AwaitSweepFreed(store, 6);
  EXPECT_EQ(store.backend()->GetSize(), 2);
  EXPECT_EQ(fake_destination_.write_calls(), 0);
}

// --- Save() ownership ------------------------------------------------------
// A remote save's verdict is addressed to whoever asked for it. Verdicts are
// drained on read, so with one shared queue the two consumers -- the
// application through PollSaveStatus(), the sweep through
// WaitForBatchWriteResult -- take each other's mail, and each direction
// breaks something different. Both directions are covered below.
// ---------------------------------------------------------------------------

// Direction one: the application must not see, or drain, an operation the
// sweep is waiting on. Stealing it strands the sweep permanently -- it waits
// for a verdict that has already been discarded, on the monitor's thread, so
// no later sweep runs either.
TEST_F(EvictSweepTest, AnApplicationPollNeitherSeesNorStealsSweepVerdicts) {
  RaidenId src{"sweep_src_owner", "0", "kv", 0};
  RaidenId dst{"sweep_dst_owner", "0", "kv", 0};
  auto store_or = MakeSweepStore(src, "ownergroup");
  ASSERT_TRUE(store_or.ok()) << store_or.status().ToString();
  KVCacheStore& store = **store_or;

  StartFakeDestination(dst, "ownergroup", /*evict_tier=*/1);
  // Held in flight: the destination keeps saying "still working", so the
  // sweep stays in its drain loop while the assertions below run against a
  // live, unsettled sweep operation.
  proto::PollWriteRemoteResponse held;
  held.set_state(proto::PollWriteRemoteResponse::PENDING);
  fake_destination_.SetPollResponse(held);

  PopulateCold(store, src, {"a", "b", "c", "d", "e", "f"});

  for (int i = 0; i < 1000 && fake_destination_.write_calls() == 0; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_GE(fake_destination_.write_calls(), 1)
      << "the sweep never got an offer in flight";

  // The application is not waiting on this operation and will never receive a
  // verdict for it, so reporting it as pending would describe a wait with no
  // end.
  auto [done, failed, pending, existing, unregistered] =
      store.PollSaveStatus();
  EXPECT_THAT(done, ::testing::IsEmpty());
  EXPECT_THAT(failed, ::testing::IsEmpty());
  EXPECT_THAT(pending, ::testing::IsEmpty());
  EXPECT_THAT(existing, ::testing::IsEmpty());
  EXPECT_THAT(unregistered, ::testing::IsEmpty());

  // Now poll hard while the verdict actually lands, so the application's
  // drain races the sweep's -- the verdict must still be filed under the
  // sweep rather than handed back here.
  std::atomic<bool> stop{false};
  absl::Mutex seen_mu;
  std::vector<std::string> seen;
  std::thread application([&]() {
    while (!stop.load(std::memory_order_relaxed)) {
      auto [d, f, p, e, u] = store.PollSaveStatus();
      absl::MutexLock lock(seen_mu);
      seen.insert(seen.end(), d.begin(), d.end());
      seen.insert(seen.end(), f.begin(), f.end());
    }
  });

  proto::PollWriteRemoteResponse committed;
  committed.set_state(proto::PollWriteRemoteResponse::COMMITTED);
  fake_destination_.SetPollResponse(committed);

  // The sweep completes only if its verdict reached it.
  AwaitSweepFreed(store, 6);
  stop.store(true, std::memory_order_relaxed);
  application.join();

  absl::MutexLock lock(seen_mu);
  EXPECT_THAT(seen, ::testing::IsEmpty())
      << "the application drained a verdict addressed to the sweep";
}

// Direction two: the sweep must not swallow the application's verdicts. This
// one is silent rather than fatal -- the application's save simply never
// reports done or failed -- which is why it is worth a test of its own.
TEST_F(EvictSweepTest, TheSweepDoesNotSwallowAnApplicationVerdict) {
  RaidenId src{"sweep_src_appverdict", "0", "kv", 0};
  RaidenId dst{"sweep_dst_appverdict", "0", "kv", 0};
  auto store_or = MakeSweepStore(src, "appverdictgroup");
  ASSERT_TRUE(store_or.ok()) << store_or.status().ToString();
  KVCacheStore& store = **store_or;

  StartFakeDestination(dst, "appverdictgroup", /*evict_tier=*/1);
  // ALL_EXIST settles an offer synchronously inside Save, so the
  // application's verdict is in its mailbox before the sweep ever runs and
  // the test does not race the verdict delivery.
  fake_destination_.SetWriteExistState(proto::WRITE_ALL_EXIST);

  // Two blocks, still under the low watermark for free blocks (6 of 8 free),
  // so saving them does not itself start a pressure episode.
  PopulatePinned(store, src, {"app0", "app1"});
  ASSERT_TRUE(store.Save({"app0", "app1"}, dst).ok());

  // Now push the store under the watermark and let the sweep run a batch,
  // draining as it goes.
  PopulateCold(store, src, {"c0", "c1", "c2", "c3"});
  AwaitSweepFreed(store, 6);

  // The application's own verdict must have survived the sweep's drains.
  auto [done, failed, pending, existing, unregistered] =
      store.PollSaveStatus();
  EXPECT_THAT(done, ::testing::UnorderedElementsAre("app0", "app1"))
      << "the sweep drained the application's verdict and discarded it";
  EXPECT_THAT(failed, ::testing::IsEmpty());
}

// The sweep flag rides on the monitor; asking for one without the other is a
// contradiction, as are watermarks that never let the sweep stop.
TEST_F(EvictSweepTest, SweepConfigContradictionsAreErrors) {
  RaidenId id{"sweep_misconfigured", "0", "kv", 0};
  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = kCapacity;
  config.raiden_id = id;
  config.global_registry_address = registry_address_;
  config.monitor_config.enable_evict_sweep = true;

  auto no_monitor = KVCacheStore::Create(config, /*capacity=*/kCapacity,
                                         registry_address_, id,
                                         /*num_shards=*/1,
                                         /*shard_size_bytes=*/1024,
                                         /*store_server_ip=*/"127.0.0.1");
  EXPECT_TRUE(absl::IsFailedPrecondition(no_monitor.status()))
      << no_monitor.status().ToString();

  config.monitor_config.enable = true;
  config.monitor_config.evict_low_watermark = 0.5;
  config.monitor_config.evict_high_watermark = 0.25;
  auto inverted = KVCacheStore::Create(config, /*capacity=*/kCapacity,
                                       registry_address_, id,
                                       /*num_shards=*/1,
                                       /*shard_size_bytes=*/1024,
                                       /*store_server_ip=*/"127.0.0.1");
  EXPECT_TRUE(absl::IsFailedPrecondition(inverted.status()))
      << inverted.status().ToString();
}

// ---------------------------------------------------------------------------
// Construction rules (kv_cache_store_construction_rules.md): Create() rejects
// a missing/wildcard store_server_ip and a controller-less configuration with
// InvalidArgument; the raw constructors LOG(FATAL) on the same violations.
// ---------------------------------------------------------------------------

BackendConfig MakeHostBackendConfig() {
  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 4;
  return config;
}

TEST(KVCacheStoreConstructionRulesTest, CreateRejectsEmptyStoreServerIp) {
  auto store_or = KVCacheStore::Create(MakeHostBackendConfig(), /*capacity=*/4,
                                       /*global_registry_address=*/"",
                                       RaidenId{}, /*num_shards=*/1,
                                       /*shard_size_bytes=*/512,
                                       /*store_server_ip=*/"");
  EXPECT_TRUE(absl::IsInvalidArgument(store_or.status())) << store_or.status();
}

TEST(KVCacheStoreConstructionRulesTest, CreateRejectsWildcardStoreServerIp) {
  for (const char* wildcard : {"[::]", "0.0.0.0", "::"}) {
    auto store_or = KVCacheStore::Create(
        MakeHostBackendConfig(), /*capacity=*/4,
        /*global_registry_address=*/"", RaidenId{}, /*num_shards=*/1,
        /*shard_size_bytes=*/512,
        /*store_server_ip=*/wildcard);
    EXPECT_TRUE(absl::IsInvalidArgument(store_or.status()))
        << "wildcard \"" << wildcard << "\": " << store_or.status();
  }
}

TEST(KVCacheStoreConstructionRulesTest, CreateRejectsZeroShards) {
  auto store_or = KVCacheStore::Create(MakeHostBackendConfig(), /*capacity=*/4,
                                       /*global_registry_address=*/"",
                                       RaidenId{}, /*num_shards=*/0,
                                       /*shard_size_bytes=*/512,
                                       /*store_server_ip=*/"127.0.0.1");
  EXPECT_TRUE(absl::IsInvalidArgument(store_or.status())) << store_or.status();
}

TEST(KVCacheStoreConstructionRulesDeathTest, CapacityCtorDiesOnEmptyIp) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      KVCacheStore store(/*capacity=*/4, /*global_registry_address=*/"",
                         RaidenId{}, /*num_shards=*/1,
                         /*shard_size_bytes=*/512,
                         /*store_server_ip=*/""),
      "construction validation failed");
}

TEST(KVCacheStoreConstructionRulesDeathTest, CapacityCtorDiesOnZeroShards) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      KVCacheStore store(/*capacity=*/4, /*global_registry_address=*/"",
                         RaidenId{}, /*num_shards=*/0,
                         /*shard_size_bytes=*/512,
                         /*store_server_ip=*/"127.0.0.1"),
      "construction validation failed");
}

// Regression test: Create() must return a Status on a RegisterStore
// failure, not FATAL. This is the one failure Create() cannot pre-validate --
// it only surfaces once the store actually tries to register itself, so it is
// the whole reason Create() exists as a distinct, recoverable-error path from
// the raw constructors. This caught a real bug once: the constructor Create()
// calls internally already wired the controller (and FATALed on failure)
// before Create() got a chance to do its own recoverable wiring, making
// Create()'s Status-return dead code for every RegisterStore failure.
TEST(KVCacheStoreConstructionRulesTest, CreateFailsWhenRegistryPublishFails) {
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(service.get());
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);

  // A reachable registry that genuinely rejects RegisterStore (empty
  // job_name) -- registered, valid construction args, but a real runtime
  // publish failure.
  auto store_or = KVCacheStore::Create(
      MakeHostBackendConfig(), /*capacity=*/4, server_address, RaidenId{},
      /*num_shards=*/1, /*shard_size_bytes=*/512,
      /*store_server_ip=*/"127.0.0.1");
  EXPECT_FALSE(store_or.ok()) << "expected RegisterStore's rejection to "
                                 "surface as a Create() failure";

  server->Shutdown();
}

class ErrorLookupBackend : public KVCacheStoreBackend {
 public:
  std::string name() const override { return "ErrorLookupBackend"; }
  absl::StatusOr<BlockSliceList> Lookup(
      absl::Span<const std::string> block_hashes,
      const LookupOptions& options = {}) override {
    return absl::InternalError("Backend lookup failed");
  }
  tsl::Future<> Load(const RaidenId& remote_id,
                     absl::Span<const std::string> block_hashes,
                     absl::Span<const int32_t> device_block_ids,
                     absl::Span<const RaidenBlockId> slices = {}) override {
    return {};
  }
  std::pair<bool, BlockSliceList> Insert(
      absl::Span<const std::string> block_hashes,
      absl::Span<const RaidenBlockId> slices, bool on_host) override {
    return {false, {}};
  }
  bool InsertAndLock(absl::Span<const std::string> block_hashes,
                     absl::Span<const RaidenBlockId> slices,
                     bool on_host) override {
    return false;
  }
  size_t ReleaseAndDelete(absl::Span<const std::string> block_hashes) override {
    return 0;
  }
  void Delete(absl::Span<const std::string> block_hashes,
              absl::Span<const RaidenBlockId> slices) override {}
  bool Pin(absl::Span<const std::string> block_hashes) override {
    return false;
  }
  void Release(absl::Span<const std::string> block_hashes) override {}
  int GetPinCount(const std::string& hash) const override { return 0; }
  size_t GetCapacity() const override { return 0; }
  size_t GetSize() const override { return 0; }
  size_t GetAvailableSpace() const override { return 0; }
};

TEST(KVCacheStoreTest, LookupAndPinWorkflow) {
  auto b = std::make_shared<TestHostOffloadBackend>(/*capacity=*/2);
  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{b},
                     RaidenId{}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");

  RaidenId id{"job", "0", "cache", 0};
  std::vector<std::string> hashes = {"h1", "h2"};
  std::vector<RaidenBlockId> slices = {RaidenBlockId(id, 1, BlockStatus::HOST),
                                       RaidenBlockId(id, 2, BlockStatus::HOST)};
  ASSERT_TRUE(store.Insert(hashes, slices, /*on_host=*/true).ok());
  // Insert pins what it takes. This case is about the pin LOOKUP grants, so
  // hand the insert's back first and count from zero.
  store.Release(hashes);

  auto res = store.Lookup(hashes, LookupOptions{.pin_found = true});
  ASSERT_TRUE(res.ok());
  EXPECT_EQ(res->size(), 2);
  EXPECT_EQ(store.GetPinCount("h1"), 1);
  EXPECT_EQ(store.GetPinCount("h2"), 1);

  // With every block pinned there is nothing an insert is allowed to reclaim,
  // so h3 is REFUSED rather than evicting a block somebody is holding.
  std::vector<std::string> new_hash = {"h3"};
  std::vector<RaidenBlockId> new_slice = {
      RaidenBlockId(id, 3, BlockStatus::HOST)};
  EXPECT_FALSE(store.Insert(new_hash, new_slice, /*on_host=*/true).ok());
  EXPECT_EQ(store.GetPinCount("h1"), 1);
  EXPECT_EQ(store.GetPinCount("h2"), 1);
  EXPECT_TRUE(PeekLookup(store, {"h3"})->empty());

  // Release pins
  store.Release(hashes);
  EXPECT_EQ(store.GetPinCount("h1"), 0);
  EXPECT_EQ(store.GetPinCount("h2"), 0);

  // Now inserting h3 succeeds, evicting unpinned h2 (tail of sequence).
  EXPECT_TRUE(store.Insert(new_hash, new_slice, /*on_host=*/true).ok());
  EXPECT_TRUE(PeekLookup(store, {"h2"})->empty());
  EXPECT_EQ(PeekLookup(store, {"h1"})->size(), 1);
}

// The application overload's pin_found=false is the observation mode: the
// answer is the same, but no pin is taken and the LRU order is untouched.
TEST(KVCacheStoreTest, LookupPinFoundFalseObservesWithoutPinning) {
  auto b = std::make_shared<TestHostOffloadBackend>(/*capacity=*/2);
  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{b},
                     RaidenId{}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");

  RaidenId id{"job", "0", "cache", 0};
  ASSERT_TRUE(InsertResident(store, {"h1"},
                             {RaidenBlockId(id, 1, BlockStatus::HOST)}, true));

  auto res =
      store.Lookup({"h1"}, /*enable_global=*/false, /*pin_found=*/false);
  ASSERT_TRUE(res.ok());
  EXPECT_EQ(res->size(), 1);
  EXPECT_EQ(store.GetPinCount("h1"), 0);

  // The default still pins.
  ASSERT_TRUE(store.Lookup({"h1"}).ok());
  EXPECT_EQ(store.GetPinCount("h1"), 1);
  store.Release({"h1"});
}

TEST(KVCacheStoreTest, LookupAndPinErrorRollback) {
  auto b1 = std::make_shared<TestHostOffloadBackend>(/*capacity=*/2);
  RaidenId id{"job", "0", "cache", 0};
  b1->Insert({"h1"}, {RaidenBlockId(id, 1, BlockStatus::HOST)},
             /*on_host=*/true);

  auto b2 = std::make_shared<ErrorLookupBackend>();

  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{b1, b2},
                     RaidenId{}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");

  auto res = store.Lookup({"h1", "h2"}, LookupOptions{.pin_found = true});
  EXPECT_FALSE(res.ok());
  EXPECT_EQ(b1->GetPinCount("h1"), 0);
}

TEST(KVCacheStoreTest, LookupAndPinCapacityTruncation) {
  auto b1 = std::make_shared<TestHostOffloadBackend>(/*capacity=*/2);
  auto b2 = std::make_shared<TestHostOffloadBackend>(/*capacity=*/2);

  RaidenId id{"job", "0", "cache", 0};
  b1->Insert({"h1", "h2"},
             {RaidenBlockId(id, 1, BlockStatus::HOST),
              RaidenBlockId(id, 2, BlockStatus::HOST)},
             /*on_host=*/true);
  b2->Insert({"h3"}, {RaidenBlockId(id, 3, BlockStatus::HOST)},
             /*on_host=*/true);

  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{b1, b2},
                     RaidenId{}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*store_server_ip=*/"127.0.0.1");

  EXPECT_EQ(store.capacity(), 2);

  auto res = store.Lookup({"h1", "h2", "h3"}, LookupOptions{.pin_found = true});
  ASSERT_TRUE(res.ok());
  ASSERT_EQ(res->size(), 2);
  EXPECT_EQ((*res)[0].first, "h1");
  EXPECT_EQ((*res)[1].first, "h2");

  EXPECT_EQ(b1->GetPinCount("h1"), 1);
  EXPECT_EQ(b1->GetPinCount("h2"), 1);
  EXPECT_EQ(b2->GetPinCount("h3"), 0);
}

// The expected_worker_count sequencing inside Create (construction returns
// only once the workers have registered, then the spec is published) is
// covered by the controller-level tests and the deployment e2e test; here a
// worker registers against an already-constructed store, which exercises the
// same snapshot -> validate -> compose -> publish pipeline synchronously.
TEST(KVCacheStoreTest, RegisterKVTransferSpecFromWorkersPublishesToRegistry) {
  auto registry_service =
      std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder registry_builder;
  int registry_port = 0;
  registry_builder.AddListeningPort(
      "localhost:0", grpc::InsecureServerCredentials(), &registry_port);
  registry_builder.RegisterService(registry_service.get());
  auto registry_server = registry_builder.BuildAndStart();
  ASSERT_NE(registry_server, nullptr);
  std::string registry_address = absl::StrCat("localhost:", registry_port);

  BackendConfig config;
  config.type = "HostOffloadBackend";
  auto store_or = KVCacheStore::Create(
      config, /*capacity=*/4, registry_address, RaidenId{"job", "0", "data", 0},
      /*num_shards=*/2, /*shard_size_bytes=*/512,
      /*store_server_ip=*/"127.0.0.1");
  ASSERT_OK(store_or.status());
  KVCacheStore& store = **store_or;

  // A live WorkerService to register: registration probes the worker's
  // endpoint with an empty CreateBuffers RPC and rejects an unreachable one.
  auto worker_server = ::tpu_raiden::controller::CreateTestWorkerServer();
  core::controller::RaidenControllerClient controller_client(
      store.raiden_controller_address());
  ASSERT_OK(controller_client.RegisterWorker(
      "worker_0", worker_server->server_address,
      {::tpu_raiden::RaidenTransferEndpoint{worker_server->server_address, {}}},
      /*node_id=*/0, /*block_array_bytes=*/{4096, 512},
      /*num_kv_shards=*/2));

  auto* backend = dynamic_cast<HostOffloadBackend*>(store.backend().get());
  ASSERT_NE(backend, nullptr);
  ASSERT_OK(backend->RegisterKVTransferSpecFromWorkers());

  // No kv_pool_group was configured, so the publish fell back to the
  // store's raiden_id.job_name as the group.
  global_registry::GlobalRegistryClient registry_client(grpc::CreateChannel(
      registry_address, grpc::InsecureChannelCredentials()));
  auto spec_or = registry_client.GetKVTransferSpec("job");
  ASSERT_OK(spec_or.status());
  ASSERT_EQ(spec_or->block_arrays_size(), 2);
  EXPECT_EQ(spec_or->block_arrays(0).block_bytes(), 4096);
  EXPECT_EQ(spec_or->block_arrays(1).block_bytes(), 512);
  EXPECT_EQ(spec_or->num_kv_shards(), 2);
  EXPECT_EQ(spec_or->num_workers(), 1);

  store_or->reset();
  registry_server->Shutdown();
}

// ===========================================================================
// Interleaved local/remote hits, seen through KVCacheStore::Lookup.
// ===========================================================================

// A store backed by a live registry, with two blocks cached locally and two
// owned by a peer, so a lookup can alternate between the two sources.
struct StoreInterleaveFixture {
  std::unique_ptr<global_registry::TestGlobalRegistryServer> registry;
  std::unique_ptr<KVCacheStore> store;
  RaidenId store_id{"store_job", "0", "kv_cache", 0};
  RaidenId peer_id{"peer_job", "0", "kv_cache", 0};
};

std::unique_ptr<StoreInterleaveFixture> MakeStoreInterleaveFixture(
    size_t capacity = 50) {
  auto f = std::make_unique<StoreInterleaveFixture>();
  f->registry = global_registry::CreateTestGlobalRegistryServer();

  CHECK_OK(f->registry->client->Register(
      {{"r1", f->peer_id, 42}, {"r2", f->peer_id, 43}}));

  f->store = std::make_unique<KVCacheStore>(
      capacity, f->registry->server_address, f->store_id, /*num_shards=*/1,
      /*shard_size_bytes=*/512,
      /*store_server_ip=*/"127.0.0.1");

  InsertResident(*f->store, {"l1", "l2"},
                 {RaidenBlockId(f->store_id, 11, BlockStatus::HOST),
                  RaidenBlockId(f->store_id, 12, BlockStatus::HOST)},
                 /*on_host=*/true);
  // Nothing advertises "l1"/"l2": inserting only indexes them, and no save
  // runs here. So the registry answers for "r1"/"r2" and nothing else, which
  // is what makes each entry's source identifiable.
  auto local = f->registry->client->Lookup({"l1", "l2"});
  CHECK_OK(local.status());
  CHECK(local->empty()) << "inserting a block must not advertise it";
  return f;
}

TEST(KVCacheStoreTest, LookupInterleavesLocalAndRemoteThroughTheStore) {
  auto f = MakeStoreInterleaveFixture();

  auto res = f->store->Lookup({"r1", "l1", "r2", "l2", "nowhere"},
                              /*enable_global=*/true);
  ASSERT_TRUE(res.ok());
  ASSERT_EQ(res->size(), 4);

  EXPECT_EQ((*res)[0].second.status, BlockStatus::REMOTE);
  EXPECT_EQ((*res)[0].second.raiden_id, f->peer_id);
  EXPECT_EQ((*res)[0].second.host_block_id, 42);

  EXPECT_EQ((*res)[1].second.status, BlockStatus::HOST);
  EXPECT_EQ((*res)[1].second.raiden_id, f->store_id);
  EXPECT_EQ((*res)[1].second.host_block_id, 11);

  EXPECT_EQ((*res)[2].second.status, BlockStatus::REMOTE);
  EXPECT_EQ((*res)[2].second.host_block_id, 43);

  EXPECT_EQ((*res)[3].second.status, BlockStatus::HOST);
  EXPECT_EQ((*res)[3].second.host_block_id, 12);
}

TEST(KVCacheStoreTest, LookupInterleavedWithPinFoundPinsOnlyLocalEntries) {
  auto f = MakeStoreInterleaveFixture();

  auto res = f->store->Lookup({"r1", "l1", "r2", "l2", "nowhere"},
                              LookupOptions{.pin_found = true});
  ASSERT_TRUE(res.ok());
  ASSERT_EQ(res->size(), 4);
  EXPECT_EQ(f->store->GetPinCount("l1"), 1);
  EXPECT_EQ(f->store->GetPinCount("l2"), 1);
  // A registry descriptor has no local entry behind it, so there is nothing to
  // pin -- and releasing the whole answer must stay safe despite that.
  EXPECT_EQ(f->store->GetPinCount("r1"), 0);
  EXPECT_EQ(f->store->GetPinCount("r2"), 0);

  f->store->Release({"r1", "l1", "r2", "l2"});
  EXPECT_EQ(f->store->GetPinCount("l1"), 0);
  EXPECT_EQ(f->store->GetPinCount("l2"), 0);
}

TEST(KVCacheStoreTest, LookupInterleavedDisabledThroughTheStore) {
  auto f = MakeStoreInterleaveFixture();

  // The sweep stops at "r1", so "l1" can only be looked for in the registry,
  // which no longer answers for it.
  auto legacy = f->store->Lookup(
      {"r1", "l1"}, LookupOptions{.enable_interleaved_lookup = false});
  ASSERT_TRUE(legacy.ok());
  ASSERT_EQ(legacy->size(), 1);
  EXPECT_EQ((*legacy)[0].first, "r1");

  auto interleaved = f->store->Lookup({"r1", "l1"}, /*enable_global=*/true);
  ASSERT_TRUE(interleaved.ok());
  ASSERT_EQ(interleaved->size(), 2);
  EXPECT_EQ((*interleaved)[1].first, "l1");
  EXPECT_EQ((*interleaved)[1].second.status, BlockStatus::HOST);
}

TEST(KVCacheStoreTest, LookupInterleavedWithoutGlobalStopsAtTheLocalMiss) {
  auto f = MakeStoreInterleaveFixture();

  for (bool interleaved : {true, false}) {
    auto res = f->store->Lookup(
        {"l1", "r1", "l2"},
        LookupOptions{.enable_global = false,
                      .enable_interleaved_lookup = interleaved});
    ASSERT_TRUE(res.ok());
    ASSERT_EQ(res->size(), 1) << "interleaved=" << interleaved;
    EXPECT_EQ((*res)[0].first, "l1");
  }
}

TEST(KVCacheStoreTest, LookupInterleavedTruncatesToCapacityAndUnwindsPins) {
  // Capacity 2 cuts the four-entry interleaved answer in half, so the store has
  // to release pins the backend took for entries it is dropping -- including
  // for a registry descriptor it never pinned, which must stay harmless.
  auto f = MakeStoreInterleaveFixture(/*capacity=*/2);

  auto res = f->store->Lookup({"r1", "l1", "r2", "l2"},
                              LookupOptions{.pin_found = true});
  ASSERT_TRUE(res.ok());
  ASSERT_EQ(res->size(), 2);
  EXPECT_EQ((*res)[0].first, "r1");
  EXPECT_EQ((*res)[1].first, "l1");

  EXPECT_EQ(f->store->GetPinCount("l1"), 1);
  EXPECT_EQ(f->store->GetPinCount("l2"), 0);
  EXPECT_EQ(f->store->GetPinCount("r2"), 0);
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
