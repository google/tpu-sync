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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/kv_cache/host_offload_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

namespace {

using ::absl_testing::StatusIs;

class CustomTestBackend : public KVCacheStoreBackend {
 public:
  std::string name() const override { return "CustomTestBackend"; }
  absl::StatusOr<BlockSliceList> Lookup(absl::Span<const std::string>,
                                        const LookupOptions&) override {
    return BlockSliceList{};
  }
  tsl::Future<> Load(const RaidenId& remote_id,
                     absl::Span<const std::string> block_hashes,
                     absl::Span<const int32_t> device_block_ids,
                     absl::Span<const RaidenBlockId> slices = {}) override {
    return tsl::Future<>(absl::UnimplementedError("Load is not supported."));
  }
  std::pair<bool, BlockSliceList> Insert(absl::Span<const std::string>,
                                         absl::Span<const RaidenBlockId>,
                                         bool) override {
    return {true, {}};
  }
  bool InsertAndLock(absl::Span<const std::string>,
                     absl::Span<const RaidenBlockId>, bool) override {
    return true;
  }
  size_t ReleaseAndDelete(absl::Span<const std::string>) override { return 0; }
  void Delete(absl::Span<const std::string>,
              absl::Span<const RaidenBlockId>) override {}
  bool Pin(absl::Span<const std::string>) override { return true; }
  void Release(absl::Span<const std::string>) override {}
  int GetPinCount(const std::string&) const override { return 0; }
  size_t GetCapacity() const override { return 100; }
  size_t GetSize() const override { return 0; }
  size_t GetAvailableSpace() const override { return 100; }
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

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
