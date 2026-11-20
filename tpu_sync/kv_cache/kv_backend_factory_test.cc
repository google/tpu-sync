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

#include "tpu_sync/kv_cache/kv_backend_factory.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/statusor.h"  // IWYU pragma: keep
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_sync/kv_cache_offload/backends/backend.h"
#include "tpu_sync/kv_cache_offload/backends/plugin/mock_storage_plugin_loader.h"
#include "tpu_sync/kv_cache_offload/backends/plugin/plugin_registry.h"
#include "tpu_sync/kv_cache_offload/backends/plugin/raiden_storage_plugin_abi.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::absl_testing::StatusIs;
using ::testing::Contains;
using ::testing::HasSubstr;

class DummyKVBackend : public backends::KVBackend {
 public:
  explicit DummyKVBackend(std::string name, BackendConfig config)
      : backends::KVBackend(config.properties),
        name_(std::move(name)),
        config_(std::move(config)) {}

  std::string name() const override { return name_; }

  const BackendConfig& config() const { return config_; }

  void WriteAsync(const backends::BlockKey& /*key*/,
                  absl::Span<const backends::HostBufferDescriptor> /*slices*/,
                  size_t /*total_bytes*/,
                  std::function<void(absl::Status)> callback) override {
    if (callback) callback(absl::OkStatus());
  }

  void ReadAsync(const backends::BlockKey& /*key*/,
                 absl::Span<const backends::HostBufferDescriptor> /*slices*/,
                 size_t /*total_bytes*/,
                 std::function<void(absl::Status)> callback) override {
    if (callback) callback(absl::OkStatus());
  }

  void BatchExistsAsync(absl::Span<const backends::BlockKey> keys,
                        std::function<void(std::vector<absl::StatusOr<bool>>)>
                            callback) override {
    std::vector<absl::StatusOr<bool>> out(keys.size(), true);
    if (callback) callback(std::move(out));
  }

 private:
  std::string name_;
  BackendConfig config_;
};

REGISTER_KV_BACKEND(
    "static_dummy_kv",
    [](const ::tpu_raiden::kv_cache::BackendConfig& config)
        -> absl::StatusOr<
            std::shared_ptr<::tpu_raiden::kv_cache::backends::KVBackend>> {
      return std::make_shared<::tpu_raiden::kv_cache::DummyKVBackend>(
          "static_dummy_kv", config);
    });

TEST(KVBackendFactoryTest, StaticRegistrationAndCreate) {
  auto& factory = KVBackendFactory::Instance();
  EXPECT_TRUE(factory.IsRegistered("static_dummy_kv"));
  EXPECT_THAT(factory.GetRegisteredTypes(), Contains("static_dummy_kv"));

  BackendConfig config;
  config.type = "static_dummy_kv";
  config.SetProperty("custom_key", "custom_val");

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<backends::KVBackend> backend,
                       factory.CreateBackend(config));
  ASSERT_NE(backend, nullptr);
  EXPECT_EQ(backend->name(), "static_dummy_kv");
  auto dummy = std::dynamic_pointer_cast<DummyKVBackend>(backend);
  ASSERT_NE(dummy, nullptr);
  EXPECT_EQ(dummy->config().GetProperty("custom_key"), "custom_val");
}

TEST(KVBackendFactoryTest, RegisterDuplicateReturnsAlreadyExists) {
  auto& factory = KVBackendFactory::Instance();
  const std::string type_name = "dynamic_dup_test_kv";
  ABSL_ASSERT_OK(
      factory.RegisterBackend(type_name, [](const BackendConfig& config) {
        return std::make_shared<DummyKVBackend>("dynamic_dup_test_kv", config);
      }));

  EXPECT_THAT(factory.RegisterBackend(type_name,
                                      [](const BackendConfig& config) {
                                        return std::make_shared<DummyKVBackend>(
                                            "dynamic_dup_test_kv", config);
                                      }),
              StatusIs(absl::StatusCode::kAlreadyExists,
                       HasSubstr("dynamic_dup_test_kv")));
}

TEST(KVBackendFactoryTest, CreateUnknownBackendReturnsNotFound) {
  BackendConfig config;
  config.type = "nonexistent_kv_backend_xyz";
  EXPECT_THAT(KVBackendFactory::Instance().CreateBackend(config),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("nonexistent_kv_backend_xyz")));
}

TEST(KVBackendFactoryTest, CreatorErrorPropagates) {
  auto& factory = KVBackendFactory::Instance();
  const std::string type_name = "failing_creator_kv";
  ABSL_ASSERT_OK(factory.RegisterBackend(
      type_name,
      [](const BackendConfig& /*config*/)
          -> absl::StatusOr<std::shared_ptr<backends::KVBackend>> {
        return absl::InvalidArgumentError("missing required endpoint property");
      }));

  BackendConfig config;
  config.type = type_name;
  EXPECT_THAT(factory.CreateBackend(config),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("missing required endpoint property")));
}

TEST(KVBackendFactoryTest, LoadsAndRegistersDynamicPluginOnDemand) {
  const std::string so_path = backends::GetMockPluginSoPath();

  BackendConfig field_cfg;
  field_cfg.type = "mock_factory_field_plugin";
  field_cfg.dynamically_loaded_backend_path = so_path;
  field_cfg.parallelism.tp_size = 1;
  field_cfg.parallelism.tp_rank = 0;

  EXPECT_FALSE(KVBackendFactory::Instance().IsRegistered(field_cfg.type));
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<backends::KVBackend> worker,
                       KVBackendFactory::Instance().CreateBackend(field_cfg));
  ASSERT_NE(worker, nullptr);
  EXPECT_TRUE(KVBackendFactory::Instance().IsRegistered(field_cfg.type));

  BackendConfig prop_cfg;
  prop_cfg.type = "mock_factory_prop_plugin";
  prop_cfg.SetProperty("dynamically_loaded_backend_path", so_path);
  prop_cfg.parallelism.tp_size = 1;
  prop_cfg.parallelism.tp_rank = 0;
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<backends::KVBackend> prop_worker,
                       KVBackendFactory::Instance().CreateBackend(prop_cfg));
  EXPECT_NE(prop_worker, nullptr);

  BackendConfig legacy_prop_cfg;
  legacy_prop_cfg.type = "mock_factory_legacy_prop_plugin";
  legacy_prop_cfg.SetProperty("plugin_path", so_path);
  legacy_prop_cfg.parallelism.tp_size = 1;
  legacy_prop_cfg.parallelism.tp_rank = 0;
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<backends::KVBackend> legacy_worker,
      KVBackendFactory::Instance().CreateBackend(legacy_prop_cfg));
  EXPECT_NE(legacy_worker, nullptr);

  setenv("RAIDEN_STORAGE_PLUGIN_PATH_MOCK_FACTORY_ENV_PLUGIN", so_path.c_str(),
         1);
  auto unset_env = absl::MakeCleanup(
      [] { unsetenv("RAIDEN_STORAGE_PLUGIN_PATH_MOCK_FACTORY_ENV_PLUGIN"); });
  BackendConfig env_cfg;
  env_cfg.type = "mock_factory_env_plugin";
  env_cfg.parallelism.tp_size = 1;
  env_cfg.parallelism.tp_rank = 0;
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<backends::KVBackend> env_worker,
                       KVBackendFactory::Instance().CreateBackend(env_cfg));
  EXPECT_NE(env_worker, nullptr);

  BackendConfig missing_so_cfg;
  missing_so_cfg.type = "mock_factory_missing_so";
  missing_so_cfg.dynamically_loaded_backend_path =
      "/nonexistent/path/libmissing_plugin.so";
  missing_so_cfg.parallelism.tp_size = 1;
  missing_so_cfg.parallelism.tp_rank = 0;
  EXPECT_THAT(KVBackendFactory::Instance().CreateBackend(missing_so_cfg),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Failed to dlopen")));
}

TEST(KVBackendFactoryTest, ValidatesWorkerParallelismWithoutDeclaredAxis) {
  const std::string so_path = backends::GetMockPluginSoPath();

  BackendConfig undeclared_cfg;
  undeclared_cfg.type = "mock_factory_parallelism_plugin";
  undeclared_cfg.dynamically_loaded_backend_path = so_path;
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<backends::KVBackend> undeclared_worker,
      KVBackendFactory::Instance().CreateBackend(undeclared_cfg));
  EXPECT_NE(undeclared_worker, nullptr);

  BackendConfig rank_without_size_cfg = undeclared_cfg;
  rank_without_size_cfg.parallelism.tp_rank = 0;
  EXPECT_THAT(KVBackendFactory::Instance().CreateBackend(rank_without_size_cfg),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("tp_rank 0 was given without tp_size")));
}

std::vector<std::string>& RecordedWorkerRoles() {
  static auto* roles = new std::vector<std::string>();
  return *roles;
}

RaidenStatusCode RecordWorkerRoleAndCreate(
    const RaidenPluginConfig* config,
    const RaidenParallelismConfig* parallelism, void** out_ctx, char* err_buf,
    size_t err_buf_len) {
  std::string role = "<not set>";
  for (size_t i = 0; i < config->num_properties; ++i) {
    if (std::strcmp(config->properties[i].key, "raiden_role") == 0) {
      role = config->properties[i].value;
    }
  }
  RecordedWorkerRoles().push_back(role);
  return backends::LoadMockPlugin()->create(config, parallelism, out_ctx,
                                            err_buf, err_buf_len);
}

TEST(KVBackendFactoryTest, StampsWorkerRoleOnPluginInstance) {
  const RaidenStorageOutofTreePluginV1* base_vtable =
      backends::LoadMockPlugin();
  ASSERT_NE(base_vtable, nullptr);
  static RaidenStorageOutofTreePluginV1 role_vtable = *base_vtable;
  role_vtable.type_name = "mock_worker_role_plugin";
  role_vtable.create = &RecordWorkerRoleAndCreate;
  ABSL_ASSERT_OK(
      backends::PluginRegistry::Instance().RegisterStaticPlugin(&role_vtable));
  RecordedWorkerRoles().clear();

  BackendConfig cfg;
  cfg.type = "mock_worker_role_plugin";
  cfg.parallelism.tp_size = 2;
  cfg.parallelism.tp_rank = 0;
  cfg.SetProperty(RAIDEN_PLUGIN_PROPERTY_ROLE, RAIDEN_PLUGIN_ROLE_COORDINATOR);
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<backends::KVBackend> worker,
                       KVBackendFactory::Instance().CreateBackend(cfg));
  EXPECT_NE(worker, nullptr);
  EXPECT_THAT(RecordedWorkerRoles(), ::testing::ElementsAre("worker"));
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
