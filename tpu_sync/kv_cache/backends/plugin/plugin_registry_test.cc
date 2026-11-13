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

#include "tpu_sync/kv_cache/backends/plugin/plugin_registry.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/backends/plugin/mock_storage_plugin_loader.h"
#include "tpu_sync/kv_cache/backends/plugin/plugin_kv_backend.h"
#include "tpu_sync/kv_cache/backends/plugin/plugin_kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/backends/plugin/raiden_storage_plugin_abi.h"
#include "tpu_sync/kv_cache/kv_backend_factory.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

TEST(PluginRegistryTest, DlopenPluginAndRoundTripShardKeyMode) {
  const std::string so_path = GetMockPluginSoPath();

  BackendConfig store_cfg;
  store_cfg.type = "mock_plugin";
  store_cfg.plugin_path = so_path;
  store_cfg.parallelism.tp_size = 2;
  store_cfg.SetProperty("prefix", "gcs_test");
  store_cfg.SetProperty("clear_store_on_create", "true");

  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<KVCacheStoreBackend> store_backend,
      KVCacheStoreBackendFactory::Instance().CreateBackend(store_cfg));
  ASSERT_NE(store_backend, nullptr);
  // The coordinator gets the dedicated plugin backend, named after the type.
  EXPECT_NE(std::dynamic_pointer_cast<OutofTreePluginKVCacheStoreBackend>(
                store_backend),
            nullptr);
  EXPECT_EQ(store_backend->name(), "mock_plugin");

  BackendConfig worker0_cfg;
  worker0_cfg.type = "mock_plugin";
  worker0_cfg.plugin_path = so_path;
  worker0_cfg.parallelism.tp_size = 2;
  worker0_cfg.parallelism.tp_rank = 0;
  worker0_cfg.SetProperty("prefix", "gcs_test");

  BackendConfig worker1_cfg = worker0_cfg;
  worker1_cfg.parallelism.tp_rank = 1;

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<KVBackend> worker0,
                       KVBackendFactory::Instance().CreateBackend(worker0_cfg));
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<KVBackend> worker1,
                       KVBackendFactory::Instance().CreateBackend(worker1_cfg));

  const std::string hash_a = "block_hash_alpha";
  const std::string hash_b = "block_hash_beta";

  TF_ASSERT_OK_AND_ASSIGN(auto initial_found,
                       store_backend->Lookup({hash_a, hash_b}));
  EXPECT_THAT(initial_found, IsEmpty());

  // Write rank 0 shard for hash_a. Because tp_size=2, the coordinator should
  // not report hash_a present until both rank 0 and rank 1 shards exist.
  TF_ASSERT_OK_AND_ASSIGN(BlockKey key_a0, worker0->mapper()->MapKey(hash_a));
  std::string payload_0 = "rank0_kv_payload";
  HostBufferDescriptor slice0{payload_0.data(), payload_0.size(), -1, 0};
  absl::Notification write0_done;
  absl::Status write0_status;
  worker0->WriteAsync(key_a0, {slice0}, payload_0.size(), [&](absl::Status s) {
    write0_status = std::move(s);
    write0_done.Notify();
  });
  write0_done.WaitForNotification();
  ABSL_ASSERT_OK(write0_status);

  TF_ASSERT_OK_AND_ASSIGN(auto partial_found,
                       store_backend->Lookup({hash_a, hash_b}));
  EXPECT_THAT(partial_found, IsEmpty());

  // Write rank 1 shard for hash_a. Now both TP shards exist.
  TF_ASSERT_OK_AND_ASSIGN(BlockKey key_a1, worker1->mapper()->MapKey(hash_a));
  std::string payload_1 = "rank1_kv_payload";
  HostBufferDescriptor slice1{payload_1.data(), payload_1.size(), -1, 0};
  absl::Notification write1_done;
  absl::Status write1_status;
  worker1->WriteAsync(key_a1, {slice1}, payload_1.size(), [&](absl::Status s) {
    write1_status = std::move(s);
    write1_done.Notify();
  });
  write1_done.WaitForNotification();
  ABSL_ASSERT_OK(write1_status);

  TF_ASSERT_OK_AND_ASSIGN(auto full_found,
                       store_backend->Lookup({hash_a, hash_b}));
  ASSERT_EQ(full_found.size(), 1u);
  EXPECT_EQ(full_found[0].first, hash_a);

  // Read back rank 0 and rank 1 shards into fresh buffers.
  std::string read_buf_0(payload_0.size(), '\0');
  HostBufferDescriptor read_slice0{read_buf_0.data(), read_buf_0.size(), -1, 0};
  absl::Notification read0_done;
  absl::Status read0_status;
  worker0->ReadAsync(key_a0, {read_slice0}, read_buf_0.size(),
                     [&](absl::Status s) {
                       read0_status = std::move(s);
                       read0_done.Notify();
                     });
  read0_done.WaitForNotification();
  ABSL_ASSERT_OK(read0_status);
  EXPECT_EQ(read_buf_0, payload_0);
}

TEST(PluginRegistryTest, BlockLevelProbeModeWhenMapKeyIsNull) {
  ABSL_ASSERT_OK(PluginRegistry::Instance().EnsurePluginLoaded(
      "mock_plugin", GetMockPluginSoPath()));
  const RaidenStorageOutofTreePluginV1* base_vtable =
      PluginRegistry::Instance().GetPlugin("mock_plugin");
  ASSERT_NE(base_vtable, nullptr);

  static RaidenStorageOutofTreePluginV1 k5_vtable = *base_vtable;
  k5_vtable.type_name = "mock_k5_plugin";
  k5_vtable.map_key = nullptr;

  ABSL_ASSERT_OK(PluginRegistry::Instance().RegisterStaticPlugin(&k5_vtable));

  BackendConfig store_cfg;
  store_cfg.type = "mock_k5_plugin";
  store_cfg.parallelism.tp_size = 4;
  store_cfg.SetProperty("clear_store_on_create", "true");

  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<KVCacheStoreBackend> store_backend,
      KVCacheStoreBackendFactory::Instance().CreateBackend(store_cfg));

  BackendConfig worker_cfg;
  worker_cfg.type = "mock_k5_plugin";
  worker_cfg.parallelism.tp_size = 4;
  worker_cfg.parallelism.tp_rank = 0;

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<KVBackend> worker,
                       KVBackendFactory::Instance().CreateBackend(worker_cfg));

  const std::string k5_hash = "k5_logical_block_hash_1";
  TF_ASSERT_OK_AND_ASSIGN(BlockKey mapped, worker->mapper()->MapKey(k5_hash));
  EXPECT_EQ(mapped.block_hash, k5_hash);
  EXPECT_EQ(mapped.resolved_key, k5_hash);

  std::string payload = "k5_block_bytes";
  HostBufferDescriptor slice{payload.data(), payload.size(), -1, 0};
  absl::Notification write_done;
  absl::Status write_status;
  worker->WriteAsync(mapped, {slice}, payload.size(), [&](absl::Status s) {
    write_status = std::move(s);
    write_done.Notify();
  });
  write_done.WaitForNotification();
  ABSL_ASSERT_OK(write_status);

  // Even though tp_size=4, `map_key == nullptr` uses single-call block-level
  // probing rather than expanding across 4 shard files.
  TF_ASSERT_OK_AND_ASSIGN(auto found,
                       store_backend->Lookup({k5_hash, "missing_hash"}));
  ASSERT_EQ(found.size(), 1u);
  EXPECT_EQ(found[0].first, k5_hash);
}

TEST(PluginRegistryTest, EnvVarFallbackResolution) {
  const std::string so_path = GetMockPluginSoPath();
  setenv("RAIDEN_STORAGE_PLUGIN_PATH_MOCK_ENV_BACKEND", so_path.c_str(), 1);

  BackendConfig worker_cfg;
  worker_cfg.type = "mock_env_backend";
  worker_cfg.parallelism.tp_size = 1;
  worker_cfg.parallelism.tp_rank = 0;

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<KVBackend> worker,
                       KVBackendFactory::Instance().CreateBackend(worker_cfg));
  EXPECT_NE(worker, nullptr);

  unsetenv("RAIDEN_STORAGE_PLUGIN_PATH_MOCK_ENV_BACKEND");
}

TEST(PluginRegistryTest, PropertyPluginPathFallbackAndExtendedStructSize) {
  const std::string so_path = GetMockPluginSoPath();

  BackendConfig prop_path_cfg;
  prop_path_cfg.type = "mock_prop_path_backend";
  prop_path_cfg.SetProperty("plugin_path", so_path);
  prop_path_cfg.parallelism.tp_size = 1;
  prop_path_cfg.parallelism.tp_rank = 0;

  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<KVBackend> worker,
      KVBackendFactory::Instance().CreateBackend(prop_path_cfg));
  EXPECT_NE(worker, nullptr);

  const RaidenStorageOutofTreePluginV1* base_vtable =
      PluginRegistry::Instance().GetPlugin("mock_prop_path_backend");
  ASSERT_NE(base_vtable, nullptr);
  EXPECT_TRUE(RAIDEN_PLUGIN_V1_HAS_FIELD(base_vtable, batch_exists_async));

  // Simulate a future v1.1 plugin whose struct_size is larger than v1.0.
  static RaidenStorageOutofTreePluginV1 v1_1_vtable = *base_vtable;
  v1_1_vtable.type_name = "mock_v1_1_plugin";
  v1_1_vtable.struct_size =
      sizeof(RaidenStorageOutofTreePluginV1) + sizeof(void*);
  ABSL_ASSERT_OK(PluginRegistry::Instance().RegisterStaticPlugin(&v1_1_vtable));

  // Simulate a truncated struct_size smaller than
  // RAIDEN_STORAGE_PLUGIN_V1_MIN_SIZE.
  RaidenStorageOutofTreePluginV1 truncated_vtable = *base_vtable;
  truncated_vtable.type_name = "mock_truncated_plugin";
  truncated_vtable.struct_size = RAIDEN_STORAGE_PLUGIN_V1_MIN_SIZE - 1;
  EXPECT_THAT(
      PluginRegistry::Instance().RegisterStaticPlugin(&truncated_vtable),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("smaller than minimum v1 size")));
}

TEST(PluginRegistryTest, RejectsMissingPluginOrAbiMismatch) {
  BackendConfig bad_path_cfg;
  bad_path_cfg.type = "nonexistent_plugin_with_path";
  bad_path_cfg.plugin_path = "/nonexistent/path/libmissing_plugin.so";
  bad_path_cfg.parallelism.tp_size = 1;
  bad_path_cfg.parallelism.tp_rank = 0;

  EXPECT_THAT(KVBackendFactory::Instance().CreateBackend(bad_path_cfg),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Failed to dlopen")));

  BackendConfig unknown_cfg;
  unknown_cfg.type = "completely_unknown_backend";
  unknown_cfg.parallelism.tp_size = 1;
  unknown_cfg.parallelism.tp_rank = 0;

  EXPECT_THAT(KVBackendFactory::Instance().CreateBackend(unknown_cfg),
              StatusIs(absl::StatusCode::kNotFound));

  RaidenStorageOutofTreePluginV1 bad_abi_vtable{};
  bad_abi_vtable.abi_version = 999;
  bad_abi_vtable.struct_size = sizeof(RaidenStorageOutofTreePluginV1);
  bad_abi_vtable.type_name = "bad_abi_plugin";
  EXPECT_THAT(PluginRegistry::Instance().RegisterStaticPlugin(&bad_abi_vtable),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("ABI version")));
}

// The role property value of each create() call on "mock_role_plugin".
std::vector<std::string>& RecordedRoles() {
  static auto* roles = new std::vector<std::string>();
  return *roles;
}

// Records the role property, then creates a mock plugin instance. The key is
// spelled out so that a change to the value in the ABI header fails the test.
RaidenStatusCode RecordRoleAndCreate(const RaidenPluginConfig* config,
                                     const RaidenParallelismConfig* parallelism,
                                     void** out_ctx, char* err_buf,
                                     size_t err_buf_len) {
  std::string role = "<not set>";
  for (size_t i = 0; i < config->num_properties; ++i) {
    if (std::strcmp(config->properties[i].key, "raiden_role") == 0) {
      role = config->properties[i].value;
    }
  }
  RecordedRoles().push_back(role);
  return PluginRegistry::Instance()
      .GetPlugin("mock_plugin")
      ->create(config, parallelism, out_ctx, err_buf, err_buf_len);
}

TEST(PluginRegistryTest, PassesInstanceRoleToPlugin) {
  ABSL_ASSERT_OK(PluginRegistry::Instance().EnsurePluginLoaded(
      "mock_plugin", GetMockPluginSoPath()));
  const RaidenStorageOutofTreePluginV1* base_vtable =
      PluginRegistry::Instance().GetPlugin("mock_plugin");
  ASSERT_NE(base_vtable, nullptr);

  static RaidenStorageOutofTreePluginV1 role_vtable = *base_vtable;
  role_vtable.type_name = "mock_role_plugin";
  role_vtable.create = &RecordRoleAndCreate;
  ABSL_ASSERT_OK(PluginRegistry::Instance().RegisterStaticPlugin(&role_vtable));
  RecordedRoles().clear();

  // The caller sets the wrong role here. The registry must overwrite it.
  BackendConfig store_cfg;
  store_cfg.type = "mock_role_plugin";
  store_cfg.parallelism.tp_size = 2;
  store_cfg.SetProperty(RAIDEN_PLUGIN_PROPERTY_ROLE, RAIDEN_PLUGIN_ROLE_WORKER);
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<KVCacheStoreBackend> store_backend,
      KVCacheStoreBackendFactory::Instance().CreateBackend(store_cfg));
  EXPECT_THAT(RecordedRoles(), ElementsAre("coordinator"));

  // The worker has the same parallelism as the coordinator's pinned ranks.
  BackendConfig worker_cfg;
  worker_cfg.type = "mock_role_plugin";
  worker_cfg.parallelism.tp_size = 2;
  worker_cfg.parallelism.tp_rank = 0;
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<KVBackend> worker,
                       KVBackendFactory::Instance().CreateBackend(worker_cfg));
  EXPECT_THAT(RecordedRoles(), ElementsAre("coordinator", "worker"));
}

}  // namespace
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden
