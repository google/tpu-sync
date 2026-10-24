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

#include "tpu_sync/kv_cache/host_offload_backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/controller/controller_client.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/core/controller/test_util.h"
#include "tpu_sync/core/controller/worker_registry.h"
#include "tpu_sync/core/kv_manager_holder.h"
#include "tpu_sync/kv_cache/block_tracker.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_client.h"
#include "tpu_sync/kv_cache/global_registry/test_util.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_sync/kv_cache/kv_cache_store_client.h"
#include "tpu_sync/kv_cache/kv_cache_store_server.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

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

  static absl::StatusOr<KVTransferSpecConfig> ComposeKVTransferSpec(
      absl::Span<const core::controller::WorkerRegistration> workers) {
    return HostOffloadBackend::ComposeKVTransferSpec(workers);
  }

  static std::vector<std::string> GetEvictCandidateKeys(
      const HostOffloadBackend& backend) {
    absl::MutexLock lock(backend.mutex_);
    return backend.lru_cache_.GetEvictCandidateKeys();
  }
};

namespace {

using ::testing::UnorderedElementsAre;

TEST(HostOffloadBackendTest, BasicInsertAndLookup) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/2);
  EXPECT_EQ(backend.name(), "HostOffloadBackend");
  EXPECT_EQ(backend.GetCapacity(), 2);
  EXPECT_EQ(backend.GetSize(), 0);

  std::vector<std::string> hashes = {"h1", "h2"};
  RaidenId id{"job", "0", "data", 0};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(id, 10, BlockStatus::HOST),
      RaidenBlockId(id, 11, BlockStatus::HOST)};

  EXPECT_TRUE(backend.Insert(hashes, slices));
  EXPECT_EQ(backend.GetSize(), 2);

  // Lookup both
  TF_ASSERT_OK_AND_ASSIGN(auto lookup_res, backend.Lookup({"h1", "h2"}));
  EXPECT_EQ(lookup_res.size(), 2);
  EXPECT_EQ(lookup_res[0].first, "h1");
  EXPECT_EQ(lookup_res[0].second.host_block_id, 10);
  EXPECT_EQ(lookup_res[1].first, "h2");
  EXPECT_EQ(lookup_res[1].second.host_block_id, 11);

  // Partial miss at end
  TF_ASSERT_OK_AND_ASSIGN(auto partial_res, backend.Lookup({"h1", "h2", "h3"}));
  EXPECT_EQ(partial_res.size(), 2);

  // Miss at start
  TF_ASSERT_OK_AND_ASSIGN(auto miss_res, backend.Lookup({"h3", "h1"}));
  EXPECT_TRUE(miss_res.empty());
}

TEST(HostOffloadBackendTest, SnapshotAndPinHostResidentForRepublish) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/4);
  RaidenId id{"job", "0", "data", 0};
  std::vector<std::string> hashes = {"h1", "h2"};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(id, 10, BlockStatus::HOST),
      RaidenBlockId(id, 11, BlockStatus::HOST_AND_HBM)};
  backend.Insert(hashes, slices);

  TF_ASSERT_OK_AND_ASSIGN(
      auto snapshot, backend.Lookup({}, LookupOptions::SnapshotHostResident()));
  EXPECT_THAT(snapshot,
              UnorderedElementsAre(::testing::Key("h1"), ::testing::Key("h2")));

  // A hash evicted after the snapshot must be skipped, not resurrected --
  // republishing it would advertise a block that is gone.
  backend.Evict({"h2"});
  const int base_pins = backend.GetPinCount("h1");
  TF_ASSERT_OK_AND_ASSIGN(
      auto pinned,
      backend.Lookup(std::vector<std::string>{"h1", "h2", "never_inserted"},
                     LookupOptions::PinPresentHostResident()));
  ASSERT_EQ(pinned.size(), 1);
  EXPECT_EQ(pinned[0].first, "h1");
  EXPECT_EQ(pinned[0].second.host_block_id, 10);
  EXPECT_EQ(pinned[0].second.status, BlockStatus::HOST);
  EXPECT_EQ(backend.GetPinCount("h1"), base_pins + 1);
  backend.Release({"h1"});
  EXPECT_EQ(backend.GetPinCount("h1"), base_pins);
}

TEST(HostOffloadBackendTest, RepublishSkipsEvictionCandidates) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/2);
  RaidenId id{"job", "0", "data", 0};
  backend.Insert({"h1", "h2"}, {RaidenBlockId(id, 10, BlockStatus::HOST),
                                RaidenBlockId(id, 11, BlockStatus::HOST)});
  // A third insert overflows capacity 2 and demotes the LRU entry ("h1") to
  // an eviction candidate: still holding its host block, but on its way out.
  backend.Insert({"h3"}, {RaidenBlockId(id, 12, BlockStatus::HOST)});
  ASSERT_THAT(HostOffloadBackendTest::GetEvictCandidateKeys(backend),
              UnorderedElementsAre("h1"));

  TF_ASSERT_OK_AND_ASSIGN(
      auto snapshot, backend.Lookup({}, LookupOptions::SnapshotHostResident()));
  EXPECT_THAT(snapshot,
              UnorderedElementsAre(::testing::Key("h2"), ::testing::Key("h3")));
  TF_ASSERT_OK_AND_ASSIGN(
      auto pinned, backend.Lookup(std::vector<std::string>{"h1"},
                                  LookupOptions::PinPresentHostResident()));
  EXPECT_TRUE(pinned.empty());
}

TEST(HostOffloadBackendTest, LookupRejectsInvalidOptionCombinations) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/4);
  RaidenId id{"job", "0", "data", 0};
  backend.Insert({"h1"}, {RaidenBlockId(id, 10, BlockStatus::HOST)});

  EXPECT_THAT(
      backend.Lookup({"h1"},
                     LookupOptions{.enable_global = true,
                                   .match_mode = LookupMatchMode::kSubset}),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(backend.Lookup({"h1"}, LookupOptions{.enable_global = true,
                                                   .host_resident_only = true}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(backend.Lookup(
                  {"h1"}, LookupOptions{.enable_global = false,
                                        .pin_found = true,
                                        .include_eviction_candidates = true}),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Positive control: valid local subset lookup succeeds on the same backend.
  TF_ASSERT_OK_AND_ASSIGN(
      auto valid_res,
      backend.Lookup({"h1"}, LookupOptions::HostResidentSubset()));
  ASSERT_EQ(valid_res.size(), 1);
  EXPECT_EQ(valid_res[0].first, "h1");
}

TEST(HostOffloadBackendTest,
     LookupHostResidentSubsetMatchesNonContiguousAndCandidatesWithoutPinning) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/2);
  RaidenId id{"job", "0", "data", 0};
  backend.Insert({"h1", "hbm_only"},
                 {RaidenBlockId(id, 10, BlockStatus::HOST),
                  RaidenBlockId(id, -1, 5, BlockStatus::HBM)});
  // Overflow capacity 2 to demote "h1" to an eviction candidate.
  backend.Insert({"h2"}, {RaidenBlockId(id, 11, 6, BlockStatus::HOST_AND_HBM)});
  ASSERT_THAT(HostOffloadBackendTest::GetEvictCandidateKeys(backend),
              UnorderedElementsAre("h1"));

  std::vector<std::string> query = {"missing1", "h1", "hbm_only", "missing2",
                                    "h2"};
  TF_ASSERT_OK_AND_ASSIGN(
      auto subset_res,
      backend.Lookup(query, LookupOptions::HostResidentSubset()));
  std::vector<std::string> subset_hashes;
  for (const auto& [hash, _] : subset_res) {
    subset_hashes.push_back(hash);
  }
  EXPECT_THAT(subset_hashes, UnorderedElementsAre("h1", "h2"));
  EXPECT_EQ(backend.GetPinCount("h1"), 0);
  EXPECT_EQ(backend.GetPinCount("h2"), 0);
  EXPECT_THAT(HostOffloadBackendTest::GetEvictCandidateKeys(backend),
              UnorderedElementsAre("h1"));
}

TEST(HostOffloadBackendTest,
     LookupSnapshotHostResidentReturnsActiveHostBlocksExcludingCandidates) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/3);
  RaidenId id{"job", "0", "data", 0};
  backend.Insert({"h1", "h2", "hbm_only"},
                 {RaidenBlockId(id, 10, BlockStatus::HOST),
                  RaidenBlockId(id, 11, 4, BlockStatus::HOST_AND_HBM),
                  RaidenBlockId(id, -1, 5, BlockStatus::HBM)});
  // Overflow capacity 3 to demote "h1" to an eviction candidate.
  backend.Insert({"h3"}, {RaidenBlockId(id, 12, BlockStatus::HOST)});
  ASSERT_THAT(HostOffloadBackendTest::GetEvictCandidateKeys(backend),
              UnorderedElementsAre("h1"));

  TF_ASSERT_OK_AND_ASSIGN(
      auto snapshot_res, backend.Lookup(/*block_hashes=*/{},
                                        LookupOptions::SnapshotHostResident()));
  std::vector<std::string> snapshot_hashes;
  for (const auto& [hash, _] : snapshot_res) {
    snapshot_hashes.push_back(hash);
  }
  EXPECT_THAT(snapshot_hashes, UnorderedElementsAre("h2", "h3"));
  EXPECT_EQ(backend.GetPinCount("h2"), 0);
  EXPECT_EQ(backend.GetPinCount("h3"), 0);
}

TEST(HostOffloadBackendTest,
     LookupPinPresentHostResidentPinsActiveHostSubsetAndSkipsCandidates) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/3);
  RaidenId id{"job", "0", "data", 0};
  backend.Insert({"cand", "h1", "hbm_only"},
                 {RaidenBlockId(id, 10, BlockStatus::HOST),
                  RaidenBlockId(id, 11, BlockStatus::HOST),
                  RaidenBlockId(id, -1, 5, BlockStatus::HBM)});
  // Overflow capacity 3 to demote "cand" to an eviction candidate.
  backend.Insert({"h2"}, {RaidenBlockId(id, 12, 6, BlockStatus::HOST_AND_HBM)});
  ASSERT_THAT(HostOffloadBackendTest::GetEvictCandidateKeys(backend),
              UnorderedElementsAre("cand"));

  std::vector<std::string> query = {"missing", "cand", "h1", "hbm_only", "h2"};
  TF_ASSERT_OK_AND_ASSIGN(
      auto pinned_res,
      backend.Lookup(query, LookupOptions::PinPresentHostResident()));
  ASSERT_EQ(pinned_res.size(), 2);
  EXPECT_EQ(pinned_res[0].first, "h1");
  EXPECT_EQ(pinned_res[0].second.host_block_id, 11);
  EXPECT_EQ(pinned_res[1].first, "h2");
  EXPECT_EQ(pinned_res[1].second.host_block_id, 12);
  EXPECT_EQ(backend.GetPinCount("h1"), 1);
  EXPECT_EQ(backend.GetPinCount("h2"), 1);
  EXPECT_EQ(backend.GetPinCount("cand"), 0);
  EXPECT_EQ(backend.GetPinCount("hbm_only"), 0);
  // Candidate must remain in evict_candidate_list_, not resurrected into
  // pinned.
  EXPECT_THAT(HostOffloadBackendTest::GetEvictCandidateKeys(backend),
              UnorderedElementsAre("cand"));

  backend.Release({"h1", "h2"});
}

TEST(HostOffloadBackendTest,
     LookupLocalOnlyPrefixStopsAtFirstMissAndPinsMatchedPrefix) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/4);
  RaidenId id{"job", "0", "data", 0};
  backend.Insert({"h1", "h2", "h3"},
                 {RaidenBlockId(id, 10, BlockStatus::HOST),
                  RaidenBlockId(id, 11, BlockStatus::HOST),
                  RaidenBlockId(id, 12, BlockStatus::HOST)});

  TF_ASSERT_OK_AND_ASSIGN(
      auto res,
      backend.Lookup({"h1", "h2", "missing", "h3"},
                     LookupOptions{.enable_global = false, .pin_found = true}));
  ASSERT_EQ(res.size(), 2);
  EXPECT_EQ(res[0].first, "h1");
  EXPECT_EQ(res[1].first, "h2");
  EXPECT_EQ(backend.GetPinCount("h1"), 1);
  EXPECT_EQ(backend.GetPinCount("h2"), 1);
  EXPECT_EQ(backend.GetPinCount("h3"), 0);
}

TEST(HostOffloadBackendTest, LookupUnboundedByAvailableSpace) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/2);
  std::vector<std::string> hashes = {"h1", "h2"};
  RaidenId id{"job", "0", "data", 0};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(id, 10, BlockStatus::HOST),
      RaidenBlockId(id, 11, BlockStatus::HOST)};

  backend.Insert(hashes, slices);
  EXPECT_TRUE(backend.Pin(hashes));
  EXPECT_EQ(backend.GetAvailableSpace(), 0);

  // Lookup still succeeds completely despite available_space() == 0
  TF_ASSERT_OK_AND_ASSIGN(auto lookup_res, backend.Lookup({"h1", "h2"}));
  EXPECT_EQ(lookup_res.size(), 2);
}

TEST(HostOffloadBackendTest, InsertAndPinRollbackOnCapacityExceeded) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/2);
  std::vector<std::string> hashes = {"h1", "h2", "h3"};
  RaidenId id{"job", "0", "data", 0};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(id, 10, BlockStatus::HOST),
      RaidenBlockId(id, 11, BlockStatus::HOST),
      RaidenBlockId(id, 12, BlockStatus::HOST)};

  // Insert(kInsertAndPin) for 3 items on capacity=2 must fail and rollback
  bool success = backend.Insert(hashes, slices, InsertOption::kInsertAndPin);
  EXPECT_FALSE(success);
  EXPECT_EQ(backend.GetSize(), 0);

  // Partial Insert(kInsertAndPin) up to capacity works
  std::vector<std::string> sub_hashes = {"h1", "h2"};
  std::vector<RaidenBlockId> sub_slices = {slices[0], slices[1]};
  EXPECT_TRUE(
      backend.Insert(sub_hashes, sub_slices, InsertOption::kInsertAndPin));
  EXPECT_EQ(backend.GetPinCount("h1"), 1);
  EXPECT_EQ(backend.GetPinCount("h2"), 1);

  // Attempt to Insert(kInsertAndPin) h3 should fail because available_space()
  // is 0
  EXPECT_FALSE(
      backend.Insert({"h3"}, {slices[2]}, InsertOption::kInsertAndPin));
  EXPECT_EQ(backend.GetPinCount("h1"), 1);
  EXPECT_EQ(backend.GetPinCount("h2"), 1);
}

TEST(HostOffloadBackendTest, InsertAndPinRebindsStaleHbmEntry) {
  // Re-inserting an HBM-only hash with kInsertAndPin updates its device block.
  HostOffloadBackendTest::Backend backend(/*capacity=*/2);
  RaidenId id{"job", "0", "data", 0};

  RaidenBlockId first(id, /*host_block_id=*/-1, /*device_block_id=*/7,
                      BlockStatus::HBM);
  ASSERT_TRUE(backend.Insert({"h1"}, {first}, InsertOption::kInsertAndPin));
  backend.Release({"h1"});

  RaidenBlockId second(id, /*host_block_id=*/-1, /*device_block_id=*/9,
                       BlockStatus::HBM);
  ASSERT_TRUE(backend.Insert({"h1"}, {second}, InsertOption::kInsertAndPin));

  TF_ASSERT_OK_AND_ASSIGN(auto lookup_res, backend.Lookup({"h1"}));
  ASSERT_EQ(lookup_res.size(), 1);
  EXPECT_EQ(lookup_res[0].second.device_block_id, 9);
  EXPECT_EQ(lookup_res[0].second.status, BlockStatus::HBM);
}

TEST(HostOffloadBackendTest, InsertAndPinKeepsHostResidentEntry) {
  // Re-inserting a HOST entry with kInsertAndPin pins it and preserves its
  // existing host block.
  HostOffloadBackendTest::Backend backend(/*capacity=*/2);
  RaidenId id{"job", "0", "data", 0};

  RaidenBlockId resident(id, /*host_block_id=*/3, BlockStatus::HOST);
  ASSERT_TRUE(backend.Insert({"h1"}, {resident}, InsertOption::kInsertAndPin));
  backend.Release({"h1"});

  RaidenBlockId reoffer(id, /*host_block_id=*/-1, /*device_block_id=*/9,
                        BlockStatus::HBM);
  ASSERT_TRUE(backend.Insert({"h1"}, {reoffer}, InsertOption::kInsertAndPin));

  TF_ASSERT_OK_AND_ASSIGN(auto lookup_res, backend.Lookup({"h1"}));
  ASSERT_EQ(lookup_res.size(), 1);
  EXPECT_EQ(lookup_res[0].second.status, BlockStatus::HOST);
  EXPECT_EQ(lookup_res[0].second.host_block_id, 3);
}

TEST(HostOffloadBackendTest,
     InsertBestEffortUpdateOverwritesActiveInPlaceWithoutPromoting) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/2);
  RaidenId id{"job", "0", "data", 0};
  ASSERT_TRUE(backend.Insert({"h1", "h2"},
                             {RaidenBlockId(id, 10, BlockStatus::HOST),
                              RaidenBlockId(id, 11, BlockStatus::HOST)},
                             InsertOption::kBestEffortUpdate));

  // Overwriting active "h1" updates its block ID without promoting it to MRU.
  ASSERT_TRUE(backend.Insert({"h1"}, {RaidenBlockId(id, 20, BlockStatus::HOST)},
                             InsertOption::kBestEffortUpdate));
  TF_ASSERT_OK_AND_ASSIGN(auto res_h1, backend.Lookup({"h1"}));
  ASSERT_EQ(res_h1.size(), 1);
  EXPECT_EQ(res_h1[0].second.host_block_id, 20);

  // Inserting "h3" displaces the LRU entry ("h1", not "h2") into candidates.
  ASSERT_TRUE(backend.Insert({"h3"}, {RaidenBlockId(id, 12, BlockStatus::HOST)},
                             InsertOption::kBestEffortUpdate));
  EXPECT_THAT(HostOffloadBackendTest::GetEvictCandidateKeys(backend),
              UnorderedElementsAre("h1"));
}

TEST(HostOffloadBackendTest,
     InsertAndPinRebindsHbmAndPreservesHostAndHostAndHbm) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/3);
  RaidenId id{"job", "0", "data", 0};

  ASSERT_TRUE(backend.Insert(
      {"hbm", "host", "both"},
      {RaidenBlockId(id, /*host_block_id=*/-1, /*device_block_id=*/1,
                     BlockStatus::HBM),
       RaidenBlockId(id, /*host_block_id=*/10, BlockStatus::HOST),
       RaidenBlockId(id, /*host_block_id=*/11, /*device_block_id=*/2,
                     BlockStatus::HOST_AND_HBM)},
      InsertOption::kInsertAndPin));
  backend.Release({"hbm", "host", "both"});

  ASSERT_TRUE(
      backend.Insert({"hbm", "host", "both"},
                     {RaidenBlockId(id, /*host_block_id=*/-1,
                                    /*device_block_id=*/91, BlockStatus::HBM),
                      RaidenBlockId(id, /*host_block_id=*/-1,
                                    /*device_block_id=*/92, BlockStatus::HBM),
                      RaidenBlockId(id, /*host_block_id=*/-1,
                                    /*device_block_id=*/93, BlockStatus::HBM)},
                     InsertOption::kInsertAndPin));
  EXPECT_EQ(backend.GetPinCount("hbm"), 1);
  EXPECT_EQ(backend.GetPinCount("host"), 1);
  EXPECT_EQ(backend.GetPinCount("both"), 1);

  TF_ASSERT_OK_AND_ASSIGN(auto lookup_res,
                          backend.Lookup({"hbm", "host", "both"}));
  ASSERT_EQ(lookup_res.size(), 3);
  EXPECT_EQ(lookup_res[0].second.status, BlockStatus::HBM);
  EXPECT_EQ(lookup_res[0].second.device_block_id, 91);
  EXPECT_EQ(lookup_res[1].second.status, BlockStatus::HOST);
  EXPECT_EQ(lookup_res[1].second.host_block_id, 10);
  EXPECT_EQ(lookup_res[2].second.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ(lookup_res[2].second.host_block_id, 11);
  EXPECT_EQ(lookup_res[2].second.device_block_id, 2);
}

TEST(HostOffloadBackendTest,
     InsertAndPinReverseOrderPlacesFirstHashAtMruAndRollsBackOnOverflow) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/2);
  RaidenId id{"job", "0", "data", 0};

  ASSERT_TRUE(backend.Insert({"h1", "h2"},
                             {RaidenBlockId(id, 10, BlockStatus::HOST),
                              RaidenBlockId(id, 11, BlockStatus::HOST)},
                             InsertOption::kInsertAndPin));
  EXPECT_EQ(backend.GetPinCount("h1"), 1);
  EXPECT_EQ(backend.GetPinCount("h2"), 1);

  // Batch mixing existing "h1" with 2 new keys ("h3", "h4") exceeds available
  // space (0) and must unpin "h1" back to its prior pin count of 1.
  EXPECT_FALSE(backend.Insert({"h1", "h3", "h4"},
                              {RaidenBlockId(id, 10, BlockStatus::HOST),
                               RaidenBlockId(id, 12, BlockStatus::HOST),
                               RaidenBlockId(id, 13, BlockStatus::HOST)},
                              InsertOption::kInsertAndPin));
  EXPECT_EQ(backend.GetPinCount("h1"), 1);
  EXPECT_EQ(backend.GetPinCount("h2"), 1);

  backend.Release({"h1", "h2"});

  // Because new keys were inserted in reverse order ("h2" then "h1"), "h1" is
  // at the MRU front and "h2" is at the LRU tail. Inserting "h3" displaces
  // "h2" into the eviction-candidates list.
  ASSERT_TRUE(backend.Insert({"h3"}, {RaidenBlockId(id, 12, BlockStatus::HOST)},
                             InsertOption::kInsertAndPin));
  EXPECT_THAT(HostOffloadBackendTest::GetEvictCandidateKeys(backend),
              UnorderedElementsAre("h2"));
}

TEST(HostOffloadBackendTest, LookupReturnsRemoteDescriptors) {
  // Setup local gRPC registry server
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = reg_server->server_address;
  auto registry_client = reg_server->client.get();

  RaidenId remote_node_id{"remote_job", "1", "data", 0};
  std::vector<global_registry::Registration> regs = {
      {.prefix_hash = "r_hash1", .raiden_id = remote_node_id, .block_id = 42},
      {.prefix_hash = "r_hash2", .raiden_id = remote_node_id, .block_id = 43},
  };
  ABSL_ASSERT_OK(registry_client->Register(regs));

  RaidenId local_node_id{"local_job", "0", "data", 0};
  ::tpu_sync::rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(local_node_id.job_name);
  unit_proto.set_job_replica_id(local_node_id.job_replica_id);
  unit_proto.set_data_name(local_node_id.data_name);
  unit_proto.set_data_replica_idx(local_node_id.data_replica_idx);

  TF_ASSERT_OK_AND_ASSIGN(auto controller, controller::RaidenController::Create(
                                               unit_proto, /*num_blocks=*/100,
                                               /*num_shards=*/1,
                                               /*shard_size_bytes=*/1024));

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.global_registry_address = server_address;
  config.raiden_id = local_node_id;

  TF_ASSERT_OK_AND_ASSIGN(auto backend,
                          HostOffloadBackend::Create(config, controller.get()));
  EXPECT_EQ(backend->name(), "HostOffloadBackend");

  TF_ASSERT_OK_AND_ASSIGN(auto lookup_res,
                          backend->Lookup({"r_hash1", "r_hash2"}));
  EXPECT_EQ(lookup_res.size(), 2);
  EXPECT_EQ(lookup_res[0].first, "r_hash1");
  EXPECT_EQ(lookup_res[0].second.status, BlockStatus::REMOTE);
  EXPECT_EQ(lookup_res[0].second.host_block_id, 42);
  EXPECT_EQ(lookup_res[0].second.raiden_id, remote_node_id);

  EXPECT_EQ(lookup_res[1].first, "r_hash2");
  EXPECT_EQ(lookup_res[1].second.status, BlockStatus::REMOTE);
  EXPECT_EQ(lookup_res[1].second.host_block_id, 43);

  // Lookup with miss stops at miss
  TF_ASSERT_OK_AND_ASSIGN(auto partial_res,
                          backend->Lookup({"r_hash1", "missing_hash"}));
  EXPECT_EQ(partial_res.size(), 1);
}

TEST(HostOffloadBackendTest,
     LookupStopsAtStaleLocalRaidenIdFromGlobalRegistry) {
  // Setup local gRPC registry server
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = reg_server->server_address;
  auto registry_client = reg_server->client.get();

  RaidenId local_node_id{"local_job", "0", "data", 0};
  std::vector<global_registry::Registration> regs = {
      {.prefix_hash = "local_g_hash",
       .raiden_id = local_node_id,
       .block_id = 99},
  };
  ABSL_ASSERT_OK(registry_client->Register(regs));

  ::tpu_sync::rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(local_node_id.job_name);
  unit_proto.set_job_replica_id(local_node_id.job_replica_id);
  unit_proto.set_data_name(local_node_id.data_name);
  unit_proto.set_data_replica_idx(local_node_id.data_replica_idx);

  TF_ASSERT_OK_AND_ASSIGN(auto controller, controller::RaidenController::Create(
                                               unit_proto, /*num_blocks=*/100,
                                               /*num_shards=*/1,
                                               /*shard_size_bytes=*/1024));

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.global_registry_address = server_address;
  config.raiden_id = local_node_id;

  TF_ASSERT_OK_AND_ASSIGN(auto backend,
                          HostOffloadBackend::Create(config, controller.get()));

  TF_ASSERT_OK_AND_ASSIGN(auto lookup_res, backend->Lookup({"local_g_hash"}));
  EXPECT_EQ(lookup_res.size(), 0);
}

TEST(HostOffloadBackendTest, CreateRegistersKVTransferSpecFromConfig) {
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = reg_server->server_address;

  RaidenId node_id{"node_job", "0", "data", 0};
  ::tpu_sync::rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  TF_ASSERT_OK_AND_ASSIGN(auto controller, controller::RaidenController::Create(
                                               unit_proto, /*num_blocks=*/100,
                                               /*num_shards=*/1,
                                               /*shard_size_bytes=*/1024));

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.global_registry_address = server_address;
  config.raiden_id = node_id;
  config.kv_transfer_spec = KVTransferSpecConfig{
      .block_array_bytes = {4096, 512}, .num_kv_shards = 2, .num_workers = 2};
  config.kv_pool_group = "node_group";

  TF_ASSERT_OK_AND_ASSIGN(auto backend,
                          HostOffloadBackend::Create(config, controller.get()));

  // The registry now serves the registered spec under the configured group.
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);
  TF_ASSERT_OK_AND_ASSIGN(auto spec,
                          registry_client.GetKVTransferSpec("node_group"));
  ASSERT_EQ(spec.block_arrays_size(), 2);
  EXPECT_EQ(spec.block_arrays(0).block_bytes(), 4096);
  EXPECT_EQ(spec.block_arrays(1).block_bytes(), 512);
  EXPECT_EQ(spec.num_kv_shards(), 2);
  EXPECT_EQ(spec.num_workers(), 2);

  // Creating another backend with the identical spec is a no-op validation;
  // a differing spec fails creation.
  ABSL_EXPECT_OK(HostOffloadBackend::Create(config, controller.get()));
  config.kv_transfer_spec->num_kv_shards = 4;
  EXPECT_TRUE(absl::IsInvalidArgument(
      HostOffloadBackend::Create(config, controller.get()).status()));
}

TEST(HostOffloadBackendTest, KVTransferSpecWithoutRegistryFailsCreation) {
  RaidenId node_id{"node_job", "0", "data", 0};
  ::tpu_sync::rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  TF_ASSERT_OK_AND_ASSIGN(auto controller, controller::RaidenController::Create(
                                               unit_proto, /*num_blocks=*/100,
                                               /*num_shards=*/1,
                                               /*shard_size_bytes=*/1024));

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.kv_transfer_spec = KVTransferSpecConfig{
      .block_array_bytes = {4096}, .num_kv_shards = 1, .num_workers = 1};
  EXPECT_TRUE(absl::IsFailedPrecondition(
      HostOffloadBackend::Create(config, controller.get()).status()));
}

core::controller::WorkerRegistration GeometryWorker(
    const std::string& worker_id, int64_t node_id,
    std::vector<uint64_t> block_array_bytes, int32_t num_kv_shards) {
  core::controller::WorkerRegistration reg;
  reg.worker_id = worker_id;
  reg.node_id = node_id;
  reg.block_array_bytes = std::move(block_array_bytes);
  reg.num_kv_shards = num_kv_shards;
  return reg;
}

TEST(HostOffloadBackendTest, ComposeKVTransferSpecFromUniformWorkers) {
  std::vector<core::controller::WorkerRegistration> workers;
  workers.push_back(GeometryWorker("w1", /*node_id=*/1, {4096, 512}, 2));
  workers.push_back(GeometryWorker("w0", /*node_id=*/0, {4096, 512}, 2));

  auto spec = HostOffloadBackendTest::ComposeKVTransferSpec(workers);
  ABSL_ASSERT_OK(spec.status());
  EXPECT_EQ(spec->block_array_bytes, (std::vector<uint64_t>{4096, 512}));
  EXPECT_EQ(spec->num_kv_shards, 2);
  EXPECT_EQ(spec->num_workers, 2);
}

TEST(HostOffloadBackendTest, ComposeKVTransferSpecRejectsEmptyWorkerList) {
  EXPECT_THAT(HostOffloadBackendTest::ComposeKVTransferSpec({}).status(),
              ::absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(HostOffloadBackendTest, ComposeKVTransferSpecRejectsMissingGeometry) {
  std::vector<core::controller::WorkerRegistration> workers;
  workers.push_back(GeometryWorker("w0", /*node_id=*/0, {4096}, 2));
  workers.push_back(GeometryWorker("w1", /*node_id=*/1,
                                   /*block_array_bytes=*/{},
                                   /*num_kv_shards=*/0));

  EXPECT_THAT(HostOffloadBackendTest::ComposeKVTransferSpec(workers).status(),
              ::absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(HostOffloadBackendTest, ComposeKVTransferSpecRejectsMismatchedGeometry) {
  std::vector<core::controller::WorkerRegistration> workers;
  workers.push_back(GeometryWorker("w0", /*node_id=*/0, {4096, 512}, 2));
  workers.push_back(GeometryWorker("w1", /*node_id=*/1, {4096, 1024}, 2));

  EXPECT_THAT(HostOffloadBackendTest::ComposeKVTransferSpec(workers).status(),
              ::absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(HostOffloadBackendTest, ComposeKVTransferSpecRejectsSparseNodeIds) {
  std::vector<core::controller::WorkerRegistration> workers;
  workers.push_back(GeometryWorker("w0", /*node_id=*/0, {4096}, 2));
  workers.push_back(GeometryWorker("w2", /*node_id=*/2, {4096}, 2));

  EXPECT_THAT(HostOffloadBackendTest::ComposeKVTransferSpec(workers).status(),
              ::absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(HostOffloadBackendTest, ComposeKVTransferSpecRejectsDuplicateNodeIds) {
  std::vector<core::controller::WorkerRegistration> workers;
  workers.push_back(GeometryWorker("w0", /*node_id=*/0, {4096}, 2));
  workers.push_back(GeometryWorker("w0b", /*node_id=*/0, {4096}, 2));

  EXPECT_THAT(HostOffloadBackendTest::ComposeKVTransferSpec(workers).status(),
              ::absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(HostOffloadBackendTest, ServerLifecycleAndControllerInitialization) {
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = reg_server->server_address;

  RaidenId node_id{"node_job", "0", "data", 0};
  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.global_registry_address = server_address;
  config.raiden_id = node_id;

  // Create RaidenController
  ::tpu_sync::rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  TF_ASSERT_OK_AND_ASSIGN(auto controller, controller::RaidenController::Create(
                                               unit_proto, /*num_blocks=*/100,
                                               /*num_shards=*/1,
                                               /*shard_size_bytes=*/1024));

  TF_ASSERT_OK_AND_ASSIGN(auto backend_base,
                          HostOffloadBackend::Create(config, controller.get()));
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(backend_base);
  ASSERT_NE(backend, nullptr);

  auto store_server = KVCacheStoreServer::Create();
  // A wildcard bind reports no publishable address,
  // so bind a real, dialable host.
  ABSL_ASSERT_OK(
      store_server->StartServer(backend.get(), controller.get(), "127.0.0.1"));
  EXPECT_GT(store_server->GetGrpcPort(), 0);
  EXPECT_FALSE(store_server->GetServerAddress().empty());
  store_server->Shutdown();
}

TEST(HostOffloadBackendTest, StartServerStripsControllerPort) {
  RaidenId node_id{"node_job", "0", "data", 0};
  ::tpu_sync::rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  TF_ASSERT_OK_AND_ASSIGN(auto controller,
                          controller::RaidenController::Create(
                              unit_proto, /*num_blocks=*/100, /*num_shards=*/1,
                              /*shard_size_bytes=*/1024,
                              /*raiden_controller_address=*/"127.0.0.1:12345"));

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.global_registry_address = "localhost:0";
  config.raiden_id = node_id;

  TF_ASSERT_OK_AND_ASSIGN(auto backend_base,
                          HostOffloadBackend::Create(config, controller.get()));
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(backend_base);
  ASSERT_NE(backend, nullptr);

  auto store_server = KVCacheStoreServer::Create();
  std::string ctrl_addr = controller->controller_address();
  std::string target_host = ctrl_addr.substr(0, ctrl_addr.rfind(':'));
  ABSL_ASSERT_OK(
      store_server->StartServer(backend.get(), controller.get(), target_host));
  EXPECT_GT(store_server->GetGrpcPort(), 0);
  EXPECT_NE(store_server->GetGrpcPort(), 12345);
  store_server->Shutdown();
}

TEST(HostOffloadBackendTest, EndToEndFetchRPC) {
  // 1. Setup global registry server
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string reg_address = reg_server->server_address;
  auto registry_client = reg_server->client.get();

  // 2. Setup mock worker server & transfer manager
  auto test_worker_server = controller::CreateTestWorkerServer();
  auto dst_transfer_mock =
      std::make_unique<controller::ShardAwareMockTransferManager>();
  test_worker_server->service->SetTransferManager(
      KVManagerHolder(dst_transfer_mock.get()));

  // 3. Setup src controller server
  auto src_controller_server = core::controller::CreateTestControllerServer();

  RaidenId src_raiden_id{"src_job", "0", "src_data", 0};
  RaidenId dst_raiden_id{"dst_job", "0", "dst_data", 0};

  ABSL_ASSERT_OK(src_controller_server->client->RegisterWorker(
      "worker_0", test_worker_server->server_address,
      {{test_worker_server->server_address, {}}}));

  src_controller_server->service->SetReadRemoteHooks(
      [&](absl::Span<const std::string> h)
          -> absl::StatusOr<std::vector<int32_t>> {
        return std::vector<int32_t>(h.size(), 42);
      },
      [&](absl::Span<const std::string> /*h*/) {});

  // 5. Register remote blocks in GlobalRegistry
  std::vector<global_registry::Registration> registrations = {
      {.prefix_hash = "fetch_hash_1",
       .raiden_id = dst_raiden_id,
       .block_id = 101},
      {.prefix_hash = "fetch_hash_2",
       .raiden_id = dst_raiden_id,
       .block_id = 102},
  };
  ABSL_ASSERT_OK(registry_client->Register(registrations));

  // 6. Create destination HostOffloadBackend & RaidenController
  ::tpu_sync::rpc::RaidenIdProto dst_unit_proto;
  dst_unit_proto.set_job_name(dst_raiden_id.job_name);
  dst_unit_proto.set_job_replica_id(dst_raiden_id.job_replica_id);
  dst_unit_proto.set_data_name(dst_raiden_id.data_name);
  dst_unit_proto.set_data_replica_idx(dst_raiden_id.data_replica_idx);

  TF_ASSERT_OK_AND_ASSIGN(
      auto dst_controller,
      controller::RaidenController::Create(dst_unit_proto, /*num_blocks=*/100,
                                           /*num_shards=*/1,
                                           /*shard_size_bytes=*/1024,
                                           /*raiden_controller_address=*/""));

  BackendConfig dst_config;
  dst_config.type = "HostOffloadBackend";
  dst_config.capacity = 100;
  dst_config.global_registry_address = reg_address;
  dst_config.raiden_id = dst_raiden_id;
  TF_ASSERT_OK_AND_ASSIGN(
      auto backend_base,
      HostOffloadBackend::Create(dst_config, dst_controller.get()));
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(backend_base);
  ASSERT_NE(backend, nullptr);

  std::vector<RaidenBlockId> dst_slices = {
      RaidenBlockId(dst_raiden_id, 101, BlockStatus::HOST),
      RaidenBlockId(dst_raiden_id, 102, BlockStatus::HOST),
  };
  backend->Insert({"fetch_hash_1", "fetch_hash_2"}, dst_slices);

  core::controller::RaidenControllerClient dst_controller_client(
      dst_controller->controller_address());
  ABSL_ASSERT_OK(dst_controller_client.RegisterWorker(
      "dst_worker_0", test_worker_server->server_address,
      {{test_worker_server->server_address, {}}}));

  auto store_server = KVCacheStoreServer::Create();
  // A wildcard bind reports no publishable address,
  // so bind a real, dialable host.
  ABSL_ASSERT_OK(store_server->StartServer(backend.get(), dst_controller.get(),
                                           "127.0.0.1"));
  EXPECT_GT(store_server->GetGrpcPort(), 0);

  // 7. Issue Fetch RPC using KVCacheStoreClient
  auto client_channel = grpc::CreateChannel(store_server->GetServerAddress(),
                                            grpc::InsecureChannelCredentials());
  KVCacheStoreClient client(client_channel);

  std::vector<std::string> hashes = {"fetch_hash_1", "fetch_hash_2"};
  std::vector<int32_t> host_ids = {201, 202};
  auto fetch_res = client
                       .Fetch(hashes, /*device_block_ids=*/{}, host_ids,
                              dst_controller->unit())
                       .Await();
  ABSL_ASSERT_OK(fetch_res.status());
  EXPECT_THAT(fetch_res->done_block_hashes(),
              UnorderedElementsAre("fetch_hash_1", "fetch_hash_2"));

  store_server->Shutdown();
}

TEST(HostOffloadBackendTest, LoadMismatchedDeviceBlockCount) {
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = reg_server->server_address;
  RaidenId node_id{"node_job", "0", "data", 0};

  ::tpu_sync::rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  TF_ASSERT_OK_AND_ASSIGN(auto controller, controller::RaidenController::Create(
                                               unit_proto, /*num_blocks=*/100,
                                               /*num_shards=*/1,
                                               /*shard_size_bytes=*/1024));
  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.global_registry_address = server_address;
  config.raiden_id = node_id;

  TF_ASSERT_OK_AND_ASSIGN(auto backend_base,
                          HostOffloadBackend::Create(config, controller.get()));
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(backend_base);
  ASSERT_NE(backend, nullptr);

  std::vector<std::string> hashes = {"hash1", "hash2"};
  std::vector<int32_t> dev_ids = {10};  // Mismatched count
  BlockTracker tracker;
  auto load_future =
      backend->Load(node_id, hashes, dev_ids, /*slices=*/{}, &tracker);
  EXPECT_THAT(load_future.Await(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(HostOffloadBackendTest, LoadSuccess) {
  // Setup GlobalRegistry and register remote block
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = reg_server->server_address;

  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  auto registry_client =
      std::make_shared<global_registry::GlobalRegistryClient>(channel);

  RaidenId remote_node_id{"remote_job", "0", "remote_data", 0};
  RaidenId local_node_id{"local_job", "0", "local_data", 0};

  std::vector<global_registry::Registration> regs = {
      {.prefix_hash = "load_hash_1",
       .raiden_id = remote_node_id,
       .block_id = 42},
  };
  ABSL_ASSERT_OK(registry_client->Register(regs));

  // Setup local RaidenController
  ::tpu_sync::rpc::RaidenIdProto local_unit;
  local_unit.set_job_name(local_node_id.job_name);
  local_unit.set_job_replica_id(local_node_id.job_replica_id);
  local_unit.set_data_name(local_node_id.data_name);
  local_unit.set_data_replica_idx(local_node_id.data_replica_idx);

  TF_ASSERT_OK_AND_ASSIGN(
      auto controller,
      controller::RaidenController::Create(local_unit, /*num_blocks=*/100,
                                           /*num_shards=*/1,
                                           /*shard_size_bytes=*/1024,
                                           /*raiden_controller_address=*/""));

  // Setup fake server for remote node to process Fetch
  BackendConfig remote_config;
  remote_config.type = "HostOffloadBackend";
  remote_config.capacity = 100;
  remote_config.global_registry_address = server_address;
  remote_config.raiden_id = remote_node_id;

  TF_ASSERT_OK_AND_ASSIGN(
      auto remote_backend_base,
      HostOffloadBackend::Create(remote_config, controller.get()));
  auto remote_backend =
      std::dynamic_pointer_cast<HostOffloadBackend>(remote_backend_base);
  ASSERT_NE(remote_backend, nullptr);

  std::vector<RaidenBlockId> remote_slices = {
      RaidenBlockId(remote_node_id, 42, BlockStatus::HOST),
  };
  remote_backend->Insert({"load_hash_1"}, remote_slices);

  auto remote_server = KVCacheStoreServer::Create();
  // A wildcard bind reports no publishable address,
  // so bind a real, dialable host -- this test publishes it below.
  ABSL_ASSERT_OK(remote_server->StartServer(remote_backend.get(),
                                            controller.get(), "127.0.0.1"));

  ABSL_ASSERT_OK(registry_client->RegisterStore(
      remote_node_id, remote_server->GetServerAddress(),
      controller->controller_address()));

  BackendConfig local_config;
  local_config.type = "HostOffloadBackend";
  local_config.capacity = 100;
  local_config.global_registry_address = server_address;
  local_config.raiden_id = local_node_id;

  TF_ASSERT_OK_AND_ASSIGN(
      auto local_backend_base,
      HostOffloadBackend::Create(local_config, controller.get()));
  auto backend =
      std::dynamic_pointer_cast<HostOffloadBackend>(local_backend_base);
  ASSERT_NE(backend, nullptr);

  // Register local worker in controller
  auto test_worker_server = controller::CreateTestWorkerServer();
  auto dst_transfer_mock =
      std::make_unique<controller::ShardAwareMockTransferManager>();
  test_worker_server->service->SetTransferManager(
      KVManagerHolder(dst_transfer_mock.get()));

  core::controller::RaidenControllerClient controller_client(
      controller->controller_address());
  ABSL_ASSERT_OK(controller_client.RegisterWorker(
      "worker_0", test_worker_server->server_address,
      {{test_worker_server->server_address, {}}}));

  // Perform Load
  std::vector<std::string> hashes = {"load_hash_1"};
  std::vector<int32_t> dev_ids = {5};
  BlockTracker tracker;
  auto load_future =
      backend->Load(remote_node_id, hashes, dev_ids, /*slices=*/{}, &tracker);
  ABSL_EXPECT_OK(load_future.Await());

  remote_server->Shutdown();
}

TEST(HostOffloadBackendTest, LoadLocalSuccess) {
  RaidenId node_id{"node_job", "0", "data", 0};
  ::tpu_sync::rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  TF_ASSERT_OK_AND_ASSIGN(auto controller, controller::RaidenController::Create(
                                               unit_proto, /*num_blocks=*/100,
                                               /*num_shards=*/1,
                                               /*shard_size_bytes=*/1024));
  auto test_worker_server = controller::CreateTestWorkerServer();
  auto transfer_mock =
      std::make_unique<controller::ShardAwareMockTransferManager>();
  test_worker_server->service->SetTransferManager(
      KVManagerHolder(transfer_mock.get()));

  core::controller::RaidenControllerClient controller_client(
      controller->controller_address());
  ABSL_ASSERT_OK(controller_client.RegisterWorker(
      "worker_0", test_worker_server->server_address,
      {{test_worker_server->server_address, {}}}));

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.raiden_id = node_id;

  TF_ASSERT_OK_AND_ASSIGN(auto backend_base,
                          HostOffloadBackend::Create(config, controller.get()));
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(backend_base);
  ASSERT_NE(backend, nullptr);

  backend->Insert({"local_hash_1"},
                  {RaidenBlockId(node_id, 10, BlockStatus::HOST)});

  BlockTracker tracker1;
  auto load_future = backend->Load(RaidenId{}, {"local_hash_1"}, {5},
                                   /*slices=*/{}, &tracker1);
  ABSL_EXPECT_OK(load_future.Await());
}

TEST(HostOffloadBackendTest, LoadLocalMissingBlockError) {
  RaidenId node_id{"node_job", "0", "data", 0};
  ::tpu_sync::rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  TF_ASSERT_OK_AND_ASSIGN(auto controller, controller::RaidenController::Create(
                                               unit_proto, /*num_blocks=*/100,
                                               /*num_shards=*/1,
                                               /*shard_size_bytes=*/1024));
  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.raiden_id = node_id;

  TF_ASSERT_OK_AND_ASSIGN(auto backend_base,
                          HostOffloadBackend::Create(config, controller.get()));
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(backend_base);
  ASSERT_NE(backend, nullptr);

  BlockTracker tracker2;
  auto load_future = backend->Load(RaidenId{}, {"missing_hash"}, {5},
                                   /*slices=*/{}, &tracker2);
  EXPECT_THAT(load_future.Await(),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST(HostOffloadBackendTest, LoadLocalNonHostBlockError) {
  RaidenId node_id{"node_job", "0", "data", 0};
  ::tpu_sync::rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  TF_ASSERT_OK_AND_ASSIGN(auto controller, controller::RaidenController::Create(
                                               unit_proto, /*num_blocks=*/100,
                                               /*num_shards=*/1,
                                               /*shard_size_bytes=*/1024));
  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.raiden_id = node_id;

  TF_ASSERT_OK_AND_ASSIGN(auto backend_base,
                          HostOffloadBackend::Create(config, controller.get()));
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(backend_base);
  ASSERT_NE(backend, nullptr);

  RaidenId remote_id{"remote_job", "0", "data", 0};
  backend->Insert({"remote_hash"},
                  {RaidenBlockId(remote_id, 10, BlockStatus::REMOTE)});

  BlockTracker tracker3;
  auto load_future =
      backend->Load(RaidenId{}, {"remote_hash"}, {5}, /*slices=*/{}, &tracker3);
  EXPECT_THAT(load_future.Await(),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

// --- Remote write backend primitives ---------------------------------------

namespace {

::tpu_sync::rpc::RaidenIdProto ToProto(const RaidenId& id) {
  ::tpu_sync::rpc::RaidenIdProto proto;
  proto.set_job_name(id.job_name);
  proto.set_job_replica_id(id.job_replica_id);
  proto.set_data_name(id.data_name);
  proto.set_data_replica_idx(id.data_replica_idx);
  return proto;
}

}  // namespace

// Prefix Lookup stops at the first miss, whereas
// LookupOptions::HostResidentSubset reports a scattered subset of host-resident
// entries.
TEST(HostOffloadBackendWriteRemoteTest, AlreadyPresentReportsAScatteredSubset) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/8);
  RaidenId id{"job", "0", "data", 0};
  backend.Insert({"a", "c"}, {RaidenBlockId(id, 1, BlockStatus::HOST),
                              RaidenBlockId(id, 3, BlockStatus::HOST)});

  TF_ASSERT_OK_AND_ASSIGN(
      auto present,
      backend.Lookup({"a", "b", "c"}, LookupOptions::HostResidentSubset()));
  EXPECT_THAT(present,
              UnorderedElementsAre(::testing::Key("a"), ::testing::Key("c")));
  TF_ASSERT_OK_AND_ASSIGN(
      auto missing, backend.Lookup({"b"}, LookupOptions::HostResidentSubset()));
  EXPECT_TRUE(missing.empty());
}

// A block that is registered elsewhere but not resident here is not something
// this node can serve, so it must not count as present.
TEST(HostOffloadBackendWriteRemoteTest, AlreadyPresentIgnoresNonHostBlocks) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/8);
  RaidenId id{"job", "0", "data", 0};
  backend.Insert({"remote"}, {RaidenBlockId(id, 1, BlockStatus::REMOTE)});

  TF_ASSERT_OK_AND_ASSIGN(
      auto present,
      backend.Lookup({"remote"}, LookupOptions::HostResidentSubset()));
  EXPECT_TRUE(present.empty());
}

TEST(HostOffloadBackendWriteRemoteTest, AdmitStrictInsertsTheWholeBatch) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/8);
  RaidenId id{"job", "0", "data", 0};

  EXPECT_TRUE(backend.Insert({"a", "b"},
                             {RaidenBlockId(id, 1, BlockStatus::HOST),
                              RaidenBlockId(id, 2, BlockStatus::HOST)},
                             InsertOption::kAdmitStrict));
  EXPECT_EQ(backend.GetSize(), 2);
  TF_ASSERT_OK_AND_ASSIGN(
      auto present,
      backend.Lookup({"a", "b"}, LookupOptions::HostResidentSubset()));
  EXPECT_THAT(present,
              UnorderedElementsAre(::testing::Key("a"), ::testing::Key("b")));
}

// kAdmitStrict rejects the entire batch if any hash is already present.
TEST(HostOffloadBackendWriteRemoteTest, AdmitStrictRefusesADuplicate) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/8);
  RaidenId id{"job", "0", "data", 0};
  backend.Insert({"a"}, {RaidenBlockId(id, 1, BlockStatus::HOST)});

  EXPECT_FALSE(backend.Insert({"b", "a"},
                              {RaidenBlockId(id, 2, BlockStatus::HOST),
                               RaidenBlockId(id, 3, BlockStatus::HOST)},
                              InsertOption::kAdmitStrict));
  // "b" is not inserted when the batch is rejected.
  TF_ASSERT_OK_AND_ASSIGN(
      auto present_b,
      backend.Lookup({"b"}, LookupOptions::HostResidentSubset()));
  EXPECT_TRUE(present_b.empty());
  EXPECT_EQ(backend.GetSize(), 1);
}

// kAdmitStrict rejects the batch when all capacity is pinned.
TEST(HostOffloadBackendWriteRemoteTest, AdmitStrictRespectsPinnedSpace) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/2);
  RaidenId id{"job", "0", "data", 0};
  ASSERT_TRUE(backend.Insert({"pinned_a", "pinned_b"},
                             {RaidenBlockId(id, 1, BlockStatus::HOST),
                              RaidenBlockId(id, 2, BlockStatus::HOST)},
                             InsertOption::kInsertAndPin));

  EXPECT_FALSE(backend.Insert({"c"}, {RaidenBlockId(id, 3, BlockStatus::HOST)},
                              InsertOption::kAdmitStrict));
}

TEST(HostOffloadBackendWriteRemoteTest,
     AdmitStrictRejectsEmptyMismatchedCollisionsAndIntraBatchDuplicates) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/2);
  RaidenId id{"job", "0", "data", 0};

  // Empty batch and mismatched span sizes are rejected.
  EXPECT_FALSE(backend.Insert({}, {}, InsertOption::kAdmitStrict));
  EXPECT_FALSE(backend.Insert({"a", "b"},
                              {RaidenBlockId(id, 1, BlockStatus::HOST)},
                              InsertOption::kAdmitStrict));

  // Intra-batch duplicate keys {"a", "a"} are rejected before modifying state.
  EXPECT_FALSE(backend.Insert({"a", "a"},
                              {RaidenBlockId(id, 1, BlockStatus::HOST),
                               RaidenBlockId(id, 2, BlockStatus::HOST)},
                              InsertOption::kAdmitStrict));
  EXPECT_EQ(backend.GetSize(), 0);

  // Valid batch succeeds without pinning.
  ASSERT_TRUE(backend.Insert({"a", "b"},
                             {RaidenBlockId(id, 1, BlockStatus::HOST),
                              RaidenBlockId(id, 2, BlockStatus::HOST)},
                             InsertOption::kAdmitStrict));
  EXPECT_EQ(backend.GetPinCount("a"), 0);
  EXPECT_EQ(backend.GetPinCount("b"), 0);

  // Active key collision is rejected.
  EXPECT_FALSE(backend.Insert({"b"}, {RaidenBlockId(id, 3, BlockStatus::HOST)},
                              InsertOption::kAdmitStrict));

  // Admitting "c" displaces "a" into eviction candidates; a subsequent
  // kAdmitStrict for candidate "a" must also be rejected.
  ASSERT_TRUE(backend.Insert({"c"}, {RaidenBlockId(id, 4, BlockStatus::HOST)},
                             InsertOption::kAdmitStrict));
  ASSERT_THAT(HostOffloadBackendTest::GetEvictCandidateKeys(backend),
              UnorderedElementsAre("a"));
  EXPECT_FALSE(backend.Insert({"a"}, {RaidenBlockId(id, 5, BlockStatus::HOST)},
                              InsertOption::kAdmitStrict));
}

// With no registry there is nothing to advertise, and no way for the blocks to
// be believed reachable when they are not -- so this is success, not an error
// that would fail every write on a registry-less node.
TEST(HostOffloadBackendWriteRemoteTest,
     RegisterBlocksAsyncIsOkWithoutARegistry) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/8);
  ABSL_EXPECT_OK(backend.RegisterBlocksAsync({"a"}, {1}).Await());
}

TEST(HostOffloadBackendWriteRemoteTest,
     RegisterBlocksAsyncPublishesToTheRegistry) {
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();

  RaidenId id{"job_regsync", "0", "data", 0};
  HostOffloadBackendTest::Backend backend(
      /*capacity=*/8, std::nullopt, id, /*raiden_controller=*/nullptr,
      std::make_shared<global_registry::GlobalRegistryClient>(
          reg_server->channel));

  ABSL_ASSERT_OK(backend.RegisterBlocksAsync({"a", "b"}, {7, 8}).Await());

  TF_ASSERT_OK_AND_ASSIGN(auto looked_up,
                          reg_server->client->Lookup({"a", "b"}));
  ASSERT_EQ(looked_up.size(), 2);
  EXPECT_EQ(looked_up[0].block_id(), 7);
  EXPECT_EQ(looked_up[1].block_id(), 8);
}

// COMMITTED is only allowed to mean "globally reachable", so a publish that
// fails has to reach the caller rather than be logged and swallowed.
TEST(HostOffloadBackendWriteRemoteTest, RegisterBlocksAsyncReportsFailure) {
  RaidenId id{"job_regfail", "0", "data", 0};
  // Port 1 is reserved and never listening.
  auto channel =
      grpc::CreateChannel("127.0.0.1:1", grpc::InsecureChannelCredentials());
  HostOffloadBackendTest::Backend backend(
      /*capacity=*/8, std::nullopt, id, /*raiden_controller=*/nullptr,
      std::make_shared<global_registry::GlobalRegistryClient>(channel));

  EXPECT_FALSE(backend.RegisterBlocksAsync({"a"}, {1}).Await().ok());
}

TEST(HostOffloadBackendWriteRemoteTest,
     RegisterBlocksAsyncRejectsMismatchedSizes) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/8);
  EXPECT_TRUE(absl::IsInvalidArgument(
      backend.RegisterBlocksAsync({"a", "b"}, {1}).Await()));
}

TEST(HostOffloadBackendTest, LookupAndPinBasicHits) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/5);
  std::vector<std::string> hashes = {"h1", "h2", "h3"};
  RaidenId id{"job", "0", "data", 0};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(id, 10, BlockStatus::HOST),
      RaidenBlockId(id, 11, BlockStatus::HOST),
      RaidenBlockId(id, 12, BlockStatus::HOST)};

  backend.Insert(hashes, slices);

  TF_ASSERT_OK_AND_ASSIGN(
      auto res, backend.Lookup(hashes, LookupOptions{.pin_found = true}));
  EXPECT_EQ(res.size(), 3);
  EXPECT_EQ(backend.GetPinCount("h1"), 1);
  EXPECT_EQ(backend.GetPinCount("h2"), 1);
  EXPECT_EQ(backend.GetPinCount("h3"), 1);
}

TEST(HostOffloadBackendTest, LookupAndPinPartialMiss) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/5);
  std::vector<std::string> hashes = {"h1", "h2"};
  RaidenId id{"job", "0", "data", 0};
  std::vector<RaidenBlockId> slices = {
      RaidenBlockId(id, 10, BlockStatus::HOST),
      RaidenBlockId(id, 11, BlockStatus::HOST)};

  backend.Insert(hashes, slices);

  TF_ASSERT_OK_AND_ASSIGN(
      auto res,
      backend.Lookup({"h1", "h2", "h3"}, LookupOptions{.pin_found = true}));
  EXPECT_EQ(res.size(), 2);
  EXPECT_EQ(backend.GetPinCount("h1"), 1);
  EXPECT_EQ(backend.GetPinCount("h2"), 1);
  EXPECT_EQ(backend.GetPinCount("h3"), 0);
}

TEST(HostOffloadBackendTest, LookupAndPinRemoteDescriptorsUnpinnedLocally) {
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = reg_server->server_address;
  auto registry_client = reg_server->client.get();

  RaidenId remote_node_id{"remote_job", "0", "data", 0};
  std::vector<global_registry::Registration> regs = {
      {.prefix_hash = "r1", .raiden_id = remote_node_id, .block_id = 42},
  };
  ABSL_ASSERT_OK(registry_client->Register(regs));

  RaidenId local_node_id{"local_job", "0", "data", 0};
  ::tpu_sync::rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(local_node_id.job_name);
  unit_proto.set_job_replica_id(local_node_id.job_replica_id);
  unit_proto.set_data_name(local_node_id.data_name);
  unit_proto.set_data_replica_idx(local_node_id.data_replica_idx);

  TF_ASSERT_OK_AND_ASSIGN(auto controller, controller::RaidenController::Create(
                                               unit_proto, /*num_blocks=*/100,
                                               /*num_shards=*/1,
                                               /*shard_size_bytes=*/1024));

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.global_registry_address = server_address;
  config.raiden_id = local_node_id;

  TF_ASSERT_OK_AND_ASSIGN(auto backend,
                          HostOffloadBackend::Create(config, controller.get()));

  TF_ASSERT_OK_AND_ASSIGN(
      auto lookup_res,
      backend->Lookup({"r1"},
                      LookupOptions{.enable_global = true, .pin_found = true}));
  ASSERT_EQ(lookup_res.size(), 1);
  EXPECT_EQ(lookup_res[0].first, "r1");
  EXPECT_EQ(lookup_res[0].second.status, BlockStatus::REMOTE);
  EXPECT_EQ(backend->GetPinCount("r1"), 0);
}

// ===========================================================================
// Interleaved local/global lookup.
// ===========================================================================

// A backend wired to a live in-process registry, with the pieces the tests
// below need to keep alive for the backend's lifetime.
struct InterleavedFixture {
  std::unique_ptr<global_registry::TestGlobalRegistryServer> registry;
  std::unique_ptr<controller::RaidenController> controller;
  std::shared_ptr<KVCacheStoreBackend> backend;
  RaidenId local_id{"local_job", "0", "data", 0};
  RaidenId peer_id{"peer_job", "0", "data", 0};
};

std::unique_ptr<InterleavedFixture> MakeInterleavedFixture(
    size_t capacity = 100) {
  auto fixture = std::make_unique<InterleavedFixture>();
  fixture->registry = global_registry::CreateTestGlobalRegistryServer();

  ::tpu_sync::rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(fixture->local_id.job_name);
  unit_proto.set_job_replica_id(fixture->local_id.job_replica_id);
  unit_proto.set_data_name(fixture->local_id.data_name);
  unit_proto.set_data_replica_idx(fixture->local_id.data_replica_idx);
  auto created_controller = controller::RaidenController::Create(
      unit_proto, /*num_blocks=*/100, /*num_shards=*/1,
      /*shard_size_bytes=*/1024);
  CHECK_OK(created_controller.status());
  fixture->controller = std::move(*created_controller);

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = capacity;
  config.global_registry_address = fixture->registry->server_address;
  config.raiden_id = fixture->local_id;

  auto created_backend =
      HostOffloadBackend::Create(config, fixture->controller.get());
  CHECK_OK(created_backend.status());
  fixture->backend = std::move(*created_backend);
  return fixture;
}

// Caches `hash` in the local index at `host_block_id`.
void InsertLocal(KVCacheStoreBackend* backend, const RaidenId& local_id,
                 const std::string& hash, int host_block_id) {
  backend->Insert({hash},
                  {RaidenBlockId(local_id, host_block_id, BlockStatus::HOST)});
}

// Publishes `hash` as owned by `owner` at `block_id`.
void RegisterGlobal(global_registry::GlobalRegistryClient* client,
                    const RaidenId& owner, const std::string& hash,
                    int block_id) {
  ABSL_ASSERT_OK(client->Register(
      {{.prefix_hash = hash, .raiden_id = owner, .block_id = block_id}}));
}

TEST(HostOffloadBackendTest, LookupInterleavesLocalAndRemoteHits) {
  auto f = MakeInterleavedFixture();
  // [remote, local, remote, local, miss]
  RegisterGlobal(f->registry->client.get(), f->peer_id, "r1", 42);
  RegisterGlobal(f->registry->client.get(), f->peer_id, "r2", 43);
  InsertLocal(f->backend.get(), f->local_id, "l1", 11);
  InsertLocal(f->backend.get(), f->local_id, "l2", 12);

  auto res = f->backend->Lookup({"r1", "l1", "r2", "l2", "nowhere"});
  ABSL_ASSERT_OK(res.status());
  ASSERT_EQ(res->size(), 4) << "the answer must run to the first hash that "
                               "neither source can resolve";

  EXPECT_EQ((*res)[0].first, "r1");
  EXPECT_EQ((*res)[0].second.status, BlockStatus::REMOTE);
  EXPECT_EQ((*res)[0].second.raiden_id, f->peer_id);
  EXPECT_EQ((*res)[0].second.host_block_id, 42);

  EXPECT_EQ((*res)[1].first, "l1");
  EXPECT_EQ((*res)[1].second.status, BlockStatus::HOST);
  EXPECT_EQ((*res)[1].second.raiden_id, f->local_id);
  EXPECT_EQ((*res)[1].second.host_block_id, 11);

  EXPECT_EQ((*res)[2].first, "r2");
  EXPECT_EQ((*res)[2].second.status, BlockStatus::REMOTE);
  EXPECT_EQ((*res)[2].second.host_block_id, 43);

  EXPECT_EQ((*res)[3].first, "l2");
  EXPECT_EQ((*res)[3].second.status, BlockStatus::HOST);
  EXPECT_EQ((*res)[3].second.host_block_id, 12);
}

TEST(HostOffloadBackendTest,
     LookupInterleavedResolvesLocalHitsAfterARemoteRun) {
  auto f = MakeInterleavedFixture();
  // The case the old two-phase logic could not express at all: the sequence
  // opens with blocks only a peer has, and continues into blocks we hold.
  RegisterGlobal(f->registry->client.get(), f->peer_id, "r1", 42);
  RegisterGlobal(f->registry->client.get(), f->peer_id, "r2", 43);
  InsertLocal(f->backend.get(), f->local_id, "l1", 11);
  InsertLocal(f->backend.get(), f->local_id, "l2", 12);

  auto res = f->backend->Lookup({"r1", "r2", "l1", "l2"});
  ABSL_ASSERT_OK(res.status());
  ASSERT_EQ(res->size(), 4);
  EXPECT_EQ((*res)[2].second.status, BlockStatus::HOST);
  EXPECT_EQ((*res)[2].second.host_block_id, 11);
  EXPECT_EQ((*res)[3].second.host_block_id, 12);
}

TEST(HostOffloadBackendTest,
     LookupInterleavedPrefersTheLocalIndexOverRegistry) {
  auto f = MakeInterleavedFixture();
  RegisterGlobal(f->registry->client.get(), f->peer_id, "r1", 42);
  InsertLocal(f->backend.get(), f->local_id, "l1", 11);
  // A peer is the registry's only owner of a block we hold ourselves, at a
  // different id (inserting does not advertise, so nothing lists us). Consulting
  // the registry for a hash the local index resolved is what used to make this
  // block look remote, read from a block id that is not the one we hold.
  RegisterGlobal(f->registry->client.get(), f->peer_id, "l1", 91);

  auto res = f->backend->Lookup({"r1", "l1"});
  ABSL_ASSERT_OK(res.status());
  ASSERT_EQ(res->size(), 2);
  EXPECT_EQ((*res)[1].first, "l1");
  EXPECT_EQ((*res)[1].second.status, BlockStatus::HOST);
  EXPECT_EQ((*res)[1].second.raiden_id, f->local_id);
  EXPECT_EQ((*res)[1].second.host_block_id, 11);
}

TEST(HostOffloadBackendTest, LookupInterleavedStopsAtTheFirstAbsoluteMiss) {
  auto f = MakeInterleavedFixture();
  InsertLocal(f->backend.get(), f->local_id, "l1", 11);
  InsertLocal(f->backend.get(), f->local_id, "l2", 12);
  // "gap" is in neither the local index nor the registry, so nothing after it
  // may be reported however reachable it is.

  auto res = f->backend->Lookup({"l1", "gap", "l2"});
  ABSL_ASSERT_OK(res.status());
  ASSERT_EQ(res->size(), 1);
  EXPECT_EQ((*res)[0].first, "l1");
}

TEST(HostOffloadBackendTest, LookupInterleavedPinsOnlyTheReturnedPrefix) {
  auto f = MakeInterleavedFixture();
  RegisterGlobal(f->registry->client.get(), f->peer_id, "r1", 42);
  InsertLocal(f->backend.get(), f->local_id, "l1", 11);
  InsertLocal(f->backend.get(), f->local_id, "l2", 12);

  // "gap" ends the answer, so l2 -- a local hit that was pinned during the
  // sweep -- has to be handed back: the caller never sees it and so would
  // never release it.
  auto res = f->backend->Lookup({"r1", "l1", "gap", "l2"},
                                LookupOptions{.pin_found = true});
  ABSL_ASSERT_OK(res.status());
  ASSERT_EQ(res->size(), 2);
  EXPECT_EQ(f->backend->GetPinCount("l1"), 1);
  EXPECT_EQ(f->backend->GetPinCount("l2"), 0);
  // Registry descriptors have no local entry to pin.
  EXPECT_EQ(f->backend->GetPinCount("r1"), 0);
}

TEST(HostOffloadBackendTest, LookupInterleavedDisabledStopsAtFirstLocalMiss) {
  auto f = MakeInterleavedFixture();
  RegisterGlobal(f->registry->client.get(), f->peer_id, "r1", 42);
  InsertLocal(f->backend.get(), f->local_id, "l1", 11);

  auto legacy = f->backend->Lookup(
      {"r1", "l1"}, LookupOptions{.enable_interleaved_lookup = false});
  ABSL_ASSERT_OK(legacy.status());
  ASSERT_EQ(legacy->size(), 1);
  EXPECT_EQ((*legacy)[0].first, "r1");

  auto interleaved = f->backend->Lookup({"r1", "l1"});
  ABSL_ASSERT_OK(interleaved.status());
  ASSERT_EQ(interleaved->size(), 2);
  EXPECT_EQ((*interleaved)[1].first, "l1");
  EXPECT_EQ((*interleaved)[1].second.status, BlockStatus::HOST);
  EXPECT_EQ((*interleaved)[1].second.host_block_id, 11);
}

TEST(HostOffloadBackendTest, LookupInterleavedWithoutGlobalStopsAtLocalMiss) {
  auto f = MakeInterleavedFixture();
  RegisterGlobal(f->registry->client.get(), f->peer_id, "r1", 42);
  InsertLocal(f->backend.get(), f->local_id, "l1", 11);

  auto res =
      f->backend->Lookup({"r1", "l1"}, LookupOptions{.enable_global = false});
  ABSL_ASSERT_OK(res.status());
  EXPECT_TRUE(res->empty()) << "with no registry to consult, a local miss "
                               "still ends the answer";

  auto res2 =
      f->backend->Lookup({"l1", "r1"}, LookupOptions{.enable_global = false});
  ABSL_ASSERT_OK(res2.status());
  ASSERT_EQ(res2->size(), 1);
  EXPECT_EQ((*res2)[0].first, "l1");
}

TEST(HostOffloadBackendTest, LookupInterleavedWithoutRegistryClient) {
  // No registry configured at all: holes are unfillable, so the answer is the
  // leading run of local hits, and nothing is left pinned past it.
  HostOffloadBackendTest::Backend backend(/*capacity=*/5);
  RaidenId id{"job", "0", "data", 0};
  backend.Insert({"l1", "l2"}, {RaidenBlockId(id, 11, BlockStatus::HOST),
                                RaidenBlockId(id, 12, BlockStatus::HOST)});

  auto res =
      backend.Lookup({"l1", "gap", "l2"}, LookupOptions{.pin_found = true});
  ABSL_ASSERT_OK(res.status());
  ASSERT_EQ(res->size(), 1);
  EXPECT_EQ((*res)[0].first, "l1");
  EXPECT_EQ(backend.GetPinCount("l1"), 1);
  EXPECT_EQ(backend.GetPinCount("l2"), 0);
}

TEST(HostOffloadBackendTest, DeleteSkipsPinnedBlocks) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/2);
  RaidenId id{"job", "0", "data", 0};
  backend.Insert({"h1"}, {RaidenBlockId(id, 10, BlockStatus::HOST)});
  ASSERT_TRUE(backend.Pin({"h1"}));
  EXPECT_EQ(backend.GetPinCount("h1"), 1);

  // Deleting a pinned block must skip it without erasing the block.
  EXPECT_EQ(backend.Delete({"h1"}), 0);
  auto lookup_res = backend.Lookup({"h1"});
  ABSL_ASSERT_OK(lookup_res.status());
  EXPECT_EQ(lookup_res->size(), 1);

  backend.Release({"h1"});
  EXPECT_EQ(backend.GetPinCount("h1"), 0);

  // Now that it's unpinned, Delete removes it.
  EXPECT_EQ(backend.Delete({"h1"}), 1);
  auto lookup_after = backend.Lookup({"h1"});
  ABSL_ASSERT_OK(lookup_after.status());
  EXPECT_TRUE(lookup_after->empty());
}

TEST(HostOffloadBackendTest,
     DeleteDefaultOptionsErasesUnpinnedAndPopulatesOutputs) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/3);
  RaidenId id{"job", "0", "data", 0};
  backend.Insert({"cand", "host", "pinned"},
                 {RaidenBlockId(id, 10, BlockStatus::HOST),
                  RaidenBlockId(id, 11, BlockStatus::HOST),
                  RaidenBlockId(id, 12, BlockStatus::HOST)});
  ASSERT_TRUE(backend.Pin({"pinned"}));
  // Overflow capacity 3 to demote "cand" to an eviction candidate.
  backend.Insert({"hbm", "both"},
                 {RaidenBlockId(id, -1, 5, BlockStatus::HBM),
                  RaidenBlockId(id, 13, 6, BlockStatus::HOST_AND_HBM)});

  std::vector<int> deallocated_ids = {999};
  std::vector<std::string> deleted_hashes = {"stale"};
  EXPECT_EQ(backend.Delete(
                {"missing", "pinned", "cand", "hbm", "both"},
                DeleteOptions{.deallocated_host_block_ids = &deallocated_ids,
                              .deleted_hashes = &deleted_hashes}),
            3);
  EXPECT_THAT(deallocated_ids, ::testing::ElementsAre(10, 13));
  EXPECT_THAT(deleted_hashes, ::testing::ElementsAre("cand", "hbm", "both"));
  EXPECT_EQ(backend.GetPinCount("pinned"), 1);
}

TEST(HostOffloadBackendTest,
     DeleteEvictHostResidentSkipsHbmOnlyPinnedAndAbsent) {
  HostOffloadBackendTest::Backend backend(/*capacity=*/3);
  RaidenId id{"job", "0", "data", 0};
  backend.Insert({"cand", "hbm", "pinned"},
                 {RaidenBlockId(id, 10, BlockStatus::HOST),
                  RaidenBlockId(id, -1, 5, BlockStatus::HBM),
                  RaidenBlockId(id, 12, BlockStatus::HOST)});
  ASSERT_TRUE(backend.Pin({"pinned"}));
  // Overflow capacity 3 to demote "cand" to an eviction candidate.
  backend.Insert({"both"},
                 {RaidenBlockId(id, 13, 6, BlockStatus::HOST_AND_HBM)});

  std::vector<int> deallocated_ids = {999};
  std::vector<std::string> evicted_hashes = {"stale"};
  EXPECT_EQ(backend.Delete({"missing", "pinned", "cand", "hbm", "both"},
                           DeleteOptions::EvictHostResident(&deallocated_ids,
                                                            &evicted_hashes)),
            2);
  EXPECT_THAT(deallocated_ids, ::testing::ElementsAre(10, 13));
  EXPECT_THAT(evicted_hashes, ::testing::ElementsAre("cand", "both"));

  // "hbm" and "pinned" must remain in the backend.
  TF_ASSERT_OK_AND_ASSIGN(
      auto remaining,
      backend.Lookup({"hbm", "pinned"},
                     LookupOptions{.enable_global = false,
                                   .match_mode = LookupMatchMode::kSubset}));
  EXPECT_THAT(remaining, UnorderedElementsAre(::testing::Key("hbm"),
                                              ::testing::Key("pinned")));
}

TEST(HostOffloadBackendTest, LoadUpdatesTrackerOnSuccessAndFailure) {
  RaidenId node_id{"node_job", "0", "data", 0};
  ::tpu_sync::rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  TF_ASSERT_OK_AND_ASSIGN(auto controller, controller::RaidenController::Create(
                                               unit_proto, /*num_blocks=*/100,
                                               /*num_shards=*/1,
                                               /*shard_size_bytes=*/1024));
  auto test_worker_server = controller::CreateTestWorkerServer();
  auto transfer_mock =
      std::make_unique<controller::ShardAwareMockTransferManager>();
  test_worker_server->service->SetTransferManager(
      KVManagerHolder(transfer_mock.get()));

  core::controller::RaidenControllerClient controller_client(
      controller->controller_address());
  ABSL_ASSERT_OK(controller_client.RegisterWorker(
      "worker_0", test_worker_server->server_address,
      {{test_worker_server->server_address, {}}}));

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.raiden_id = node_id;

  TF_ASSERT_OK_AND_ASSIGN(auto backend_base,
                          HostOffloadBackend::Create(config, controller.get()));
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(backend_base);
  ASSERT_NE(backend, nullptr);

  // Success path:
  backend->Insert({"load_success_1"},
                  {RaidenBlockId(node_id, 10, BlockStatus::HOST)});

  BlockTracker success_tracker;
  success_tracker.AddPending("load_success_1");
  auto success_future =
      backend->Load(RaidenId{}, {"load_success_1"}, {5}, {}, &success_tracker);
  ABSL_EXPECT_OK(success_future.Await());

  BlockTracker::StatusResult success_res = success_tracker.Poll();
  EXPECT_THAT(success_res.done, UnorderedElementsAre("load_success_1"));
  EXPECT_TRUE(success_res.failed.empty());

  // Failure path (transfer failure):
  backend->Insert({"load_fail_1"},
                  {RaidenBlockId(node_id, 11, BlockStatus::HOST)});
  transfer_mock->fail_transfers = true;

  BlockTracker failure_tracker;
  failure_tracker.AddPending("load_fail_1");
  auto failure_future =
      backend->Load(RaidenId{}, {"load_fail_1"}, {6}, {}, &failure_tracker);
  EXPECT_FALSE(failure_future.Await().ok());

  BlockTracker::StatusResult failure_res = failure_tracker.Poll();
  EXPECT_TRUE(failure_res.done.empty());
  EXPECT_THAT(failure_res.failed, UnorderedElementsAre("load_fail_1"));
}

TEST(HostOffloadBackendTest, SaveUpdatesTrackerOnSuccessAndFailure) {
  RaidenId node_id{"node_job", "0", "data", 0};
  ::tpu_sync::rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  TF_ASSERT_OK_AND_ASSIGN(auto controller, controller::RaidenController::Create(
                                               unit_proto, /*num_blocks=*/100,
                                               /*num_shards=*/1,
                                               /*shard_size_bytes=*/1024));
  auto test_worker_server = controller::CreateTestWorkerServer();
  auto transfer_mock =
      std::make_unique<controller::ShardAwareMockTransferManager>();
  test_worker_server->service->SetTransferManager(
      KVManagerHolder(transfer_mock.get()));

  core::controller::RaidenControllerClient controller_client(
      controller->controller_address());
  ABSL_ASSERT_OK(controller_client.RegisterWorker(
      "worker_0", test_worker_server->server_address,
      {{test_worker_server->server_address, {}}}));

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.raiden_id = node_id;

  TF_ASSERT_OK_AND_ASSIGN(auto backend_base,
                          HostOffloadBackend::Create(config, controller.get()));
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(backend_base);
  ASSERT_NE(backend, nullptr);

  // Success path:
  backend->Insert({"save_success_1"},
                  {RaidenBlockId(node_id, -1, 7, BlockStatus::HBM)});

  BlockTracker success_tracker;
  success_tracker.AddPending("save_success_1");
  auto success_future =
      backend->Save({"save_success_1"}, {7}, {10}, &success_tracker);
  ABSL_EXPECT_OK(success_future.Await());

  BlockTracker::StatusResult success_res = success_tracker.Poll();
  EXPECT_THAT(success_res.done, UnorderedElementsAre("save_success_1"));
  EXPECT_TRUE(success_res.failed.empty());

  auto lookup_res = backend->Lookup({"save_success_1"});
  ABSL_ASSERT_OK(lookup_res.status());
  ASSERT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, 10);
  EXPECT_EQ((*lookup_res)[0].second.device_block_id, 7);

  // Failure path (transfer failure):
  transfer_mock->fail_transfers = true;
  BlockTracker failure_tracker;
  failure_tracker.AddPending("save_fail_1");
  auto failure_future =
      backend->Save({"save_fail_1"}, {8}, {11}, &failure_tracker);
  EXPECT_FALSE(failure_future.Await().ok());

  BlockTracker::StatusResult failure_res = failure_tracker.Poll();
  EXPECT_TRUE(failure_res.done.empty());
  EXPECT_THAT(failure_res.failed, UnorderedElementsAre("save_fail_1"));
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
