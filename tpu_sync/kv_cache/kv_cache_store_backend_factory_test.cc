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

#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/block_tracker.h"
#include "tpu_sync/kv_cache/host_offload_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

namespace {

using ::absl_testing::StatusIs;

class CustomTestBackend : public KVCacheStoreBackend {
 public:
  std::string name() const override { return "CustomTestBackend"; }
  absl::StatusOr<BlockSliceList> Lookup(absl::Span<const std::string> hashes,
                                        const LookupOptions& options) override {
    auto is_host_resident = [](BlockStatus s) {
      return s == BlockStatus::HOST || s == BlockStatus::HOST_AND_HBM;
    };
    BlockSliceList result;
    if (options.match_mode == LookupMatchMode::kSubset && hashes.empty()) {
      for (const auto& [h, id] : entries_) {
        if (options.host_resident_only && !is_host_resident(id.status)) {
          continue;
        }
        if (options.pin_found) {
          ++pin_counts_[h];
        }
        result.push_back({h, id});
      }
      return result;
    }
    for (const auto& h : hashes) {
      auto it = entries_.find(h);
      if (it == entries_.end() || (options.host_resident_only &&
                                   !is_host_resident(it->second.status))) {
        if (options.match_mode == LookupMatchMode::kSubset) {
          continue;
        }
        break;
      }
      if (options.pin_found) {
        ++pin_counts_[h];
      }
      result.push_back({h, it->second});
    }
    return result;
  }
  tsl::Future<> Load(const RaidenId& remote_id,
                     absl::Span<const std::string> block_hashes,
                     absl::Span<const int32_t> device_block_ids,
                     absl::Span<const RaidenBlockId> slices,
                     BlockTracker* absl_nonnull load_tracker) override {
    CHECK(load_tracker != nullptr);
    return tsl::Future<>(absl::UnimplementedError("Load is not supported."));
  }
  bool Insert(absl::Span<const std::string> hashes,
              absl::Span<const RaidenBlockId> slices,
              InsertOption option = InsertOption::kBestEffortUpdate) override {
    if (option == InsertOption::kAdmitStrict) {
      return false;
    }
    for (size_t i = 0; i < hashes.size() && i < slices.size(); ++i) {
      entries_[hashes[i]] = slices[i];
      if (option == InsertOption::kInsertAndPin) {
        ++pin_counts_[hashes[i]];
      }
    }
    return true;
  }
  size_t Delete(absl::Span<const std::string> hashes,
                const DeleteOptions& options = {}) override {
    if (options.deallocated_host_block_ids != nullptr) {
      options.deallocated_host_block_ids->clear();
    }
    if (options.deleted_hashes != nullptr) {
      options.deleted_hashes->clear();
    }
    if (options.host_resident_only) {
      return 0;
    }
    size_t erased = 0;
    for (const auto& h : hashes) {
      if (entries_.erase(h) > 0) {
        ++erased;
        if (options.deleted_hashes != nullptr) {
          options.deleted_hashes->push_back(h);
        }
      }
    }
    return erased;
  }
  bool Pin(absl::Span<const std::string> hashes) override {
    for (const auto& h : hashes) {
      if (!entries_.contains(h)) return false;
    }
    for (const auto& h : hashes) {
      ++pin_counts_[h];
    }
    return true;
  }
  void Release(absl::Span<const std::string> hashes) override {
    for (const auto& h : hashes) {
      auto it = pin_counts_.find(h);
      if (it != pin_counts_.end() && it->second > 0) {
        --it->second;
      }
    }
  }
  int GetPinCount(const std::string& hash) const override {
    auto it = pin_counts_.find(hash);
    return it != pin_counts_.end() ? it->second : 0;
  }
  size_t GetCapacity() const override { return 100; }
  size_t GetSize() const override { return entries_.size(); }
  size_t GetAvailableSpace() const override { return 100 - entries_.size(); }

 private:
  absl::flat_hash_map<std::string, RaidenBlockId> entries_;
  absl::flat_hash_map<std::string, int> pin_counts_;
};

REGISTER_KV_CACHE_STORE_BACKEND(
    "custom_test_backend",
    [](const BackendConfig& config, controller::RaidenController* controller)
        -> absl::StatusOr<std::shared_ptr<KVCacheStoreBackend>> {
      return std::make_shared<CustomTestBackend>();
    });

TEST(BackendConfigTest, PropertyGettersAndSetters) {
  BackendConfig config;
  config.SetProperty("str_key", "hello");
  config.SetProperty("bool_true1", "true");
  config.SetProperty("bool_true2", "1");
  config.SetProperty("bool_false", "false");
  config.SetProperty("int_key", "12345");

  EXPECT_TRUE(config.HasProperty("str_key"));
  EXPECT_FALSE(config.HasProperty("missing_key"));

  EXPECT_EQ(config.GetProperty("str_key"), "hello");
  EXPECT_EQ(config.GetProperty("missing_key", "default"), "default");

  EXPECT_TRUE(config.GetBoolProperty("bool_true1"));
  EXPECT_TRUE(config.GetBoolProperty("bool_true2"));
  EXPECT_FALSE(config.GetBoolProperty("bool_false", true));
  EXPECT_FALSE(config.GetBoolProperty("missing_key", false));
  EXPECT_TRUE(config.GetBoolProperty("missing_key", true));

  EXPECT_EQ(config.GetIntProperty("int_key"), 12345);
  EXPECT_EQ(config.GetIntProperty("missing_key", 99), 99);
  EXPECT_EQ(config.GetIntProperty("str_key", 42), 42);
}

TEST(KVCacheStoreBackendFactoryTest, BuiltInHostOffload) {
  ::tpu_sync::rpc::RaidenIdProto unit_proto;
  TF_ASSERT_OK_AND_ASSIGN(auto ctrl, controller::RaidenController::Create(
                                         unit_proto, /*num_blocks=*/100,
                                         /*num_shards=*/1,
                                         /*shard_size_bytes=*/1024));

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 50;

  TF_ASSERT_OK_AND_ASSIGN(
      auto backend, KVCacheStoreBackendFactory::Create(config, ctrl.get()));
  EXPECT_EQ(backend->name(), "HostOffloadBackend");
  EXPECT_EQ(backend->GetCapacity(), 50);

  // Capacity 0 error test
  config.capacity = 0;
  EXPECT_THAT(KVCacheStoreBackendFactory::Create(config, ctrl.get()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(KVCacheStoreBackendFactoryTest, HostOffloadWithGlobalRegistry) {
  ::tpu_sync::rpc::RaidenIdProto unit_proto;
  TF_ASSERT_OK_AND_ASSIGN(auto ctrl, controller::RaidenController::Create(
                                         unit_proto, /*num_blocks=*/100,
                                         /*num_shards=*/1,
                                         /*shard_size_bytes=*/1024));

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 50;
  config.global_registry_address = "localhost:50051";

  TF_ASSERT_OK_AND_ASSIGN(
      auto backend, KVCacheStoreBackendFactory::Create(config, ctrl.get()));
  EXPECT_EQ(backend->name(), "HostOffloadBackend");
}

TEST(KVCacheStoreBackendFactoryTest, AutoRegistrationMacro) {
  EXPECT_TRUE(KVCacheStoreBackendFactory::Instance().IsRegistered(
      "custom_test_backend"));

  BackendConfig config;
  config.type = "custom_test_backend";

  TF_ASSERT_OK_AND_ASSIGN(auto backend,
                          KVCacheStoreBackendFactory::Create(config));
  EXPECT_EQ(backend->name(), "CustomTestBackend");
}

TEST(KVCacheStoreBackendFactoryTest, DuplicateRegistrationFails) {
  auto status = KVCacheStoreBackendFactory::Instance().RegisterBackend(
      "HostOffloadBackend",
      [](const BackendConfig&, controller::RaidenController*)
          -> absl::StatusOr<std::shared_ptr<KVCacheStoreBackend>> {
        return nullptr;
      });
  EXPECT_EQ(status.code(), absl::StatusCode::kAlreadyExists);
}

TEST(KVCacheStoreBackendFactoryTest, UnknownBackendTypeFails) {
  BackendConfig config;
  config.type = "non_existent_backend";

  EXPECT_THAT(KVCacheStoreBackendFactory::Create(config),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(KVCacheStoreBackendFactoryTest, KVCacheStoreCreateIntegration) {
  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 200;

  TF_ASSERT_OK_AND_ASSIGN(
      auto store,
      KVCacheStore::Create(config, /*capacity=*/0,
                           /*global_registry_address=*/"", RaidenId{},
                           /*num_shards=*/1, /*shard_size_bytes=*/512,
                           /*store_server_ip=*/"127.0.0.1"));
  ASSERT_NE(store, nullptr);
  EXPECT_EQ(store->capacity(), 200);
  EXPECT_EQ(store->backend()->name(), "HostOffloadBackend");
}

TEST(KVCacheStoreBackendFactoryTest, CustomBackendDefaultHostOffloadHooks) {
  BackendConfig config;
  config.type = "custom_test_backend";

  TF_ASSERT_OK_AND_ASSIGN(auto backend,
                          KVCacheStoreBackendFactory::Create(config));
  EXPECT_EQ(backend->name(), "CustomTestBackend");

  EXPECT_FALSE(
      backend->Insert({"h1"}, {RaidenBlockId()}, InsertOption::kAdmitStrict));

  BlockTracker save_tracker;
  EXPECT_THAT(
      backend->BeginWriteRemote(RaidenId{"dst", "0", "kv", 0}, {"h1"}, {0},
                                absl::Seconds(25), absl::Seconds(30),
                                &save_tracker),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               testing::HasSubstr(
                   "A remote save requires a HostOffloadBackend at tier 0")));
  EXPECT_THAT(
      backend->RegisterKVTransferSpecFromWorkers(),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               testing::HasSubstr("registering the KVTransferSpec requires a "
                                  "HostOffloadBackend at tier 0")));
}

TEST(KVCacheStoreBackendFactoryTest,
     KVCacheStoreSaveRemoteWithCustomTier0BackendFailsPrecondition) {
  const RaidenId src_id{"src_job", "0", "kv", 0};
  const RaidenId dst_id{"dst_job", "0", "kv", 0};
  BackendConfig config;
  config.type = "custom_test_backend";
  config.capacity = 100;
  config.raiden_id = src_id;

  TF_ASSERT_OK_AND_ASSIGN(
      auto store,
      KVCacheStore::Create(config, /*capacity=*/0,
                           /*global_registry_address=*/"", src_id,
                           /*num_shards=*/1, /*shard_size_bytes=*/512,
                           /*store_server_ip=*/"127.0.0.1"));
  ASSERT_NE(store, nullptr);
  EXPECT_EQ(store->backend()->name(), "CustomTestBackend");

  ABSL_ASSERT_OK(store->Insert(
      {"h1"}, {RaidenBlockId(src_id, /*host_id=*/0, BlockStatus::HOST)},
      /*on_host=*/true));
  EXPECT_EQ(store->GetPinCount("h1"), 1);

  EXPECT_THAT(
      store->Save({"h1"}, dst_id),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               testing::HasSubstr(
                   "A remote save requires a HostOffloadBackend at tier 0")));
  EXPECT_EQ(store->GetPinCount("h1"), 1);
  EXPECT_EQ(store->InFlightRemoteWritesCountForTesting(), 0);
}

TEST(ApplyParallelismToPropertiesTest, SetsTopologyProperties) {
  BackendConfig config;
  config.type = "posix";
  ApplyParallelismToProperties({.tp_size = 4, .tp_rank = 2}, &config);
  EXPECT_EQ(config.GetProperty("tp_size"), "4");
  EXPECT_EQ(config.GetProperty("tp_rank"), "2");
}

TEST(ApplyParallelismToPropertiesTest, OverridesCallerTopologyProperties) {
  BackendConfig config;
  config.type = "posix";
  config.SetProperty("tp_size", "8");
  config.SetProperty("tp_rank", "7");
  config.SetProperty("root_dir", "/some/dir");
  ApplyParallelismToProperties({.tp_size = 4, .tp_rank = 1}, &config);
  EXPECT_EQ(config.GetProperty("tp_size"), "4");
  EXPECT_EQ(config.GetProperty("tp_rank"), "1");
  EXPECT_EQ(config.GetProperty("root_dir"), "/some/dir");
}

TEST(ApplyParallelismToPropertiesTest, SetsPcpTopologyProperties) {
  BackendConfig config;
  config.type = "posix";
  ApplyParallelismToProperties({.pcp_size = 8, .pcp_rank = 3}, &config);
  EXPECT_EQ(config.GetProperty("pcp_size"), "8");
  EXPECT_EQ(config.GetProperty("pcp_rank"), "3");
  EXPECT_FALSE(config.HasProperty("tp_size"));
  EXPECT_FALSE(config.HasProperty("tp_rank"));
}

TEST(ApplyParallelismToPropertiesTest, UndeclaredFieldsEraseCallerProperties) {
  BackendConfig config;
  config.type = "posix";
  config.SetProperty("tp_size", "8");
  config.SetProperty("tp_rank", "7");
  config.SetProperty("pcp_size", "4");
  config.SetProperty("pcp_rank", "1");
  ApplyParallelismToProperties({}, &config);
  EXPECT_FALSE(config.HasProperty("tp_size"));
  EXPECT_FALSE(config.HasProperty("tp_rank"));
  EXPECT_FALSE(config.HasProperty("pcp_size"));
  EXPECT_FALSE(config.HasProperty("pcp_rank"));
}

TEST(ValidateWorkerParallelismTest, AcceptsDeclaredAndUndeclaredAxes) {
  ABSL_EXPECT_OK(ValidateWorkerParallelism({}));
  ABSL_EXPECT_OK(ValidateWorkerParallelism({.tp_size = 2, .tp_rank = 1}));
  ABSL_EXPECT_OK(ValidateWorkerParallelism({.pcp_size = 8, .pcp_rank = 7}));
  ABSL_EXPECT_OK(ValidateWorkerParallelism(
      {.tp_size = 1, .tp_rank = 0, .pcp_size = 8, .pcp_rank = 3}));
}

TEST(ValidateWorkerParallelismTest, RejectsInvalidPcpAxis) {
  EXPECT_THAT(ValidateWorkerParallelism({.pcp_size = 0}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("pcp_size must be >= 1")));
  EXPECT_THAT(ValidateWorkerParallelism({.pcp_size = -2}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("got -2")));
  EXPECT_THAT(
      ValidateWorkerParallelism({.pcp_size = 4, .pcp_rank = 4}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr(
                   "pcp_rank must be in [0, 4) for pcp_size 4; got 4")));
  EXPECT_THAT(ValidateWorkerParallelism({.pcp_rank = 0}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("pcp_rank 0 was given without")));
}

TEST(ResolveCoordinatorParallelismTest, PinsDeclaredRanksToZero) {
  TF_ASSERT_OK_AND_ASSIGN(
      backends::ParallelismConfig coordinator,
      ResolveCoordinatorParallelism(
          {.tp_size = 2, .tp_rank = 1, .pcp_size = 4, .pcp_rank = 3}));
  EXPECT_EQ(coordinator.tp_size, 2);
  EXPECT_EQ(coordinator.tp_rank, 0);
  EXPECT_EQ(coordinator.pcp_size, 4);
  EXPECT_EQ(coordinator.pcp_rank, 0);
}

TEST(ResolveCoordinatorParallelismTest, LeavesUndeclaredAxesUndeclared) {
  TF_ASSERT_OK_AND_ASSIGN(backends::ParallelismConfig coordinator,
                          ResolveCoordinatorParallelism({}));
  EXPECT_EQ(coordinator.tp_size, backends::kAxisUndeclared);
  EXPECT_EQ(coordinator.tp_rank, backends::kAxisUndeclared);
  EXPECT_EQ(coordinator.pcp_size, backends::kAxisUndeclared);
  EXPECT_EQ(coordinator.pcp_rank, backends::kAxisUndeclared);
}

TEST(ResolveCoordinatorParallelismTest, RejectsPcpRankWithoutPcpSize) {
  EXPECT_THAT(
      ResolveCoordinatorParallelism({.pcp_rank = 3}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr("pcp_rank 3 was given without pcp_size")));
}

TEST(ResolveCoordinatorParallelismTest, RejectsTpRankWithoutTpSize) {
  EXPECT_THAT(
      ResolveCoordinatorParallelism({.tp_rank = 1}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr("tp_rank 1 was given without tp_size")));
}

TEST(ResolveCoordinatorParallelismTest, RejectsZeroPcpSize) {
  EXPECT_THAT(ResolveCoordinatorParallelism({.pcp_size = 0}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("pcp_size must be >= 1")));
}

TEST(ApplyParallelismToPropertiesTest, SetsAndErasesPpTopologyProperties) {
  BackendConfig config;
  config.type = "posix";
  ApplyParallelismToProperties({.pp_size = 4, .pp_rank = 2}, &config);
  EXPECT_EQ(config.GetProperty("pp_size"), "4");
  EXPECT_EQ(config.GetProperty("pp_rank"), "2");
  EXPECT_FALSE(config.HasProperty("pcp_size"));
  EXPECT_FALSE(config.HasProperty("tp_size"));

  ApplyParallelismToProperties({}, &config);
  EXPECT_FALSE(config.HasProperty("pp_size"));
  EXPECT_FALSE(config.HasProperty("pp_rank"));
}

TEST(ValidateWorkerParallelismTest, AcceptsPpWithOtherAxes) {
  ABSL_EXPECT_OK(ValidateWorkerParallelism({.pp_size = 4, .pp_rank = 3}));
  ABSL_EXPECT_OK(ValidateWorkerParallelism({.tp_size = 2,
                                            .tp_rank = 1,
                                            .pcp_size = 2,
                                            .pcp_rank = 0,
                                            .pp_size = 2,
                                            .pp_rank = 1}));
}

TEST(ValidateWorkerParallelismTest, RejectsInvalidPpAxis) {
  EXPECT_THAT(ValidateWorkerParallelism({.pp_size = 0}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("pp_size must be >= 1")));
  EXPECT_THAT(ValidateWorkerParallelism({.pp_size = -2}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("got -2")));
  EXPECT_THAT(ValidateWorkerParallelism({.pp_size = 2, .pp_rank = 2}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr(
                           "pp_rank must be in [0, 2) for pp_size 2; got 2")));
  EXPECT_THAT(ValidateWorkerParallelism({.pp_rank = 0}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("pp_rank 0 was given without")));
}

TEST(ResolveCoordinatorParallelismTest, PinsPpRankToZero) {
  TF_ASSERT_OK_AND_ASSIGN(
      backends::ParallelismConfig coordinator,
      ResolveCoordinatorParallelism({.pp_size = 4, .pp_rank = 3}));
  EXPECT_EQ(coordinator.pp_size, 4);
  EXPECT_EQ(coordinator.pp_rank, 0);
  EXPECT_EQ(coordinator.tp_size, backends::kAxisUndeclared);
  EXPECT_EQ(coordinator.pcp_size, backends::kAxisUndeclared);

  TF_ASSERT_OK_AND_ASSIGN(backends::ParallelismConfig undeclared,
                          ResolveCoordinatorParallelism({}));
  EXPECT_EQ(undeclared.pp_size, backends::kAxisUndeclared);
  EXPECT_EQ(undeclared.pp_rank, backends::kAxisUndeclared);
}

TEST(ResolveCoordinatorParallelismTest, RejectsPpRankWithoutPpSize) {
  EXPECT_THAT(
      ResolveCoordinatorParallelism({.pp_rank = 1}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr("pp_rank 1 was given without pp_size")));
}

TEST(FormatParallelismTest, FormatsEveryAxisOutermostFirst) {
  EXPECT_EQ(FormatParallelism({.tp_size = 4,
                               .tp_rank = 3,
                               .pcp_size = 2,
                               .pcp_rank = 0,
                               .pp_size = 2,
                               .pp_rank = 1}),
            "pp=2/r1 pcp=2/r0 tp=4/r3");
}

TEST(FormatParallelismTest, FormatsUndeclaredAxes) {
  EXPECT_EQ(FormatParallelism({}),
            "pp=undeclared pcp=undeclared tp=undeclared");
  EXPECT_EQ(FormatParallelism(
                {.tp_size = 4, .tp_rank = 3, .pp_size = 2, .pp_rank = 1}),
            "pp=2/r1 pcp=undeclared tp=4/r3");
}

// Configs that fail validation print as given, so a log names the bad value.
TEST(FormatParallelismTest, FormatsUnsetRanksAndInvalidValuesAsGiven) {
  EXPECT_EQ(FormatParallelism({.tp_size = 8}),
            "pp=undeclared pcp=undeclared tp=8");
  EXPECT_EQ(FormatParallelism({.tp_size = 1, .tp_rank = 0, .pcp_rank = 3}),
            "pp=undeclared pcp=undeclared/r3 tp=1/r0");
  EXPECT_EQ(FormatParallelism({.pcp_size = 0, .pcp_rank = 0}),
            "pp=undeclared pcp=0/r0 tp=undeclared");
}

TEST(HasDeclaredAxisTest, TrueWhenAnyAxisSizeIsSet) {
  EXPECT_TRUE(HasDeclaredAxis({.tp_size = 1}));
  EXPECT_TRUE(HasDeclaredAxis({.pcp_size = 8}));
  EXPECT_TRUE(HasDeclaredAxis({.pp_size = 1}));
  // A size of 0 is declared (and invalid); validation reports it.
  EXPECT_TRUE(HasDeclaredAxis({.pcp_size = 0}));
}

TEST(HasDeclaredAxisTest, FalseWhenEverySizeIsUnset) {
  EXPECT_FALSE(HasDeclaredAxis({}));
  // A rank alone does not declare its axis.
  EXPECT_FALSE(HasDeclaredAxis({.tp_rank = 0, .pcp_rank = 1, .pp_rank = 2}));
}

TEST(RequireDeclaredAxisTest, AcceptsAnyDeclaredAxis) {
  ABSL_EXPECT_OK(RequireDeclaredAxis({.tp_size = 1}, {.tp_size = 1}));
  ABSL_EXPECT_OK(RequireDeclaredAxis({.pcp_size = 8}, {.pcp_size = 8}));
  ABSL_EXPECT_OK(RequireDeclaredAxis({.pp_size = 1}, {.pp_size = 1}));
}

TEST(RequireDeclaredAxisTest, RejectsNoAxisAndNamesReceivedTopology) {
  EXPECT_THAT(
      RequireDeclaredAxis({.tp_rank = 2}, {}),
      ::absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::AllOf(
              ::testing::HasSubstr("no parallelism axis is declared"),
              ::testing::HasSubstr("received topology: pp=undeclared "
                                   "pcp=undeclared tp=undeclared/r2"))));
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
