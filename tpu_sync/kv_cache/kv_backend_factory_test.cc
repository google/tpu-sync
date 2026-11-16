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
#include <functional>
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
#include "xla/tsl/platform/statusor.h"  // IWYU pragma: keep
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"

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

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
