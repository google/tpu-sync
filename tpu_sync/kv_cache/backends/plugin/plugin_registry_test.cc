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

#include <unistd.h>

#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <string>
#include <system_error>  // NOLINT(build/c++11)

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/statusor.h"  // IWYU pragma: keep
#include "tpu_sync/kv_cache/backends/plugin/mock_storage_plugin_loader.h"
#include "tpu_sync/kv_cache/backends/plugin/raiden_storage_plugin_abi.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(PluginRegistryTest, LoadsPluginByExplicitPathAndCachesDescriptor) {
  const std::string so_path = GetMockPluginSoPath();
  TF_ASSERT_OK_AND_ASSIGN(
      const RaidenStorageOutofTreePluginV1* vtable,
      PluginRegistry::Instance().LoadPlugin("mock_plugin", so_path));
  ASSERT_NE(vtable, nullptr);
  EXPECT_STREQ(vtable->type_name, "mock_plugin");
  EXPECT_EQ(PluginRegistry::Instance().GetPlugin("MOCK_PLUGIN"), vtable);

  // Subsequent calls for the same type return the cached descriptor even when
  // no path is passed.
  TF_ASSERT_OK_AND_ASSIGN(const RaidenStorageOutofTreePluginV1* cached,
                       PluginRegistry::Instance().LoadPlugin("mock_plugin"));
  EXPECT_EQ(cached, vtable);
}

TEST(PluginRegistryTest, ResolvesPathFromEnvironmentVariables) {
  const std::string so_path = GetMockPluginSoPath();
  setenv("RAIDEN_STORAGE_PLUGIN_PATH_MOCK_ENV_BACKEND", so_path.c_str(), 1);
  auto unset_path_env = absl::MakeCleanup(
      [] { unsetenv("RAIDEN_STORAGE_PLUGIN_PATH_MOCK_ENV_BACKEND"); });

  TF_ASSERT_OK_AND_ASSIGN(
      const RaidenStorageOutofTreePluginV1* from_type_env,
      PluginRegistry::Instance().LoadPlugin("mock_env_backend"));
  ASSERT_NE(from_type_env, nullptr);
  EXPECT_EQ(PluginRegistry::Instance().GetPlugin("mock_env_backend"),
            from_type_env);

  const std::string plugin_dir =
      absl::StrCat(::testing::TempDir(), "/plugin_dir_", getpid());
  std::filesystem::create_directories(plugin_dir);
  std::filesystem::copy_file(
      so_path, absl::StrCat(plugin_dir, "/libraiden_plugin_dir_backend.so"));
  setenv("RAIDEN_STORAGE_PLUGIN_DIR", plugin_dir.c_str(), 1);
  auto cleanup_dir = absl::MakeCleanup([&] {
    unsetenv("RAIDEN_STORAGE_PLUGIN_DIR");
    std::error_code ec;
    std::filesystem::remove_all(plugin_dir, ec);
  });

  TF_ASSERT_OK_AND_ASSIGN(const RaidenStorageOutofTreePluginV1* from_dir_env,
                       PluginRegistry::Instance().LoadPlugin("dir_backend"));
  ASSERT_NE(from_dir_env, nullptr);
}

TEST(PluginRegistryTest, AcceptsExtendedStructSizeAndRejectsTruncatedStruct) {
  TF_ASSERT_OK_AND_ASSIGN(const RaidenStorageOutofTreePluginV1* base_vtable,
                       PluginRegistry::Instance().LoadPlugin(
                           "mock_plugin", GetMockPluginSoPath()));
  EXPECT_TRUE(RAIDEN_PLUGIN_V1_HAS_FIELD(base_vtable, batch_exists_async));

  // Simulate a future v1.1 plugin whose struct_size is larger than v1.0.
  static RaidenStorageOutofTreePluginV1 v1_1_vtable = *base_vtable;
  v1_1_vtable.type_name = "mock_v1_1_plugin";
  v1_1_vtable.struct_size =
      sizeof(RaidenStorageOutofTreePluginV1) + sizeof(void*);
  ABSL_ASSERT_OK(PluginRegistry::Instance().RegisterStaticPlugin(&v1_1_vtable));
  TF_ASSERT_OK_AND_ASSIGN(
      const RaidenStorageOutofTreePluginV1* loaded_v1_1,
      PluginRegistry::Instance().LoadPlugin("mock_v1_1_plugin"));
  EXPECT_EQ(loaded_v1_1, &v1_1_vtable);

  RaidenStorageOutofTreePluginV1 truncated_vtable = *base_vtable;
  truncated_vtable.type_name = "mock_truncated_plugin";
  truncated_vtable.struct_size = RAIDEN_STORAGE_PLUGIN_V1_MIN_SIZE - 1;
  EXPECT_THAT(
      PluginRegistry::Instance().RegisterStaticPlugin(&truncated_vtable),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("smaller than minimum v1 size")));
}

TEST(PluginRegistryTest, RejectsMissingPluginOrAbiMismatch) {
  EXPECT_THAT(PluginRegistry::Instance().LoadPlugin(""),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(PluginRegistry::Instance().LoadPlugin(
                  "nonexistent_plugin_with_path",
                  "/nonexistent/path/libmissing_plugin.so"),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Failed to dlopen")));

  EXPECT_THAT(
      PluginRegistry::Instance().LoadPlugin("completely_unknown_backend"),
      StatusIs(absl::StatusCode::kNotFound));

  EXPECT_THAT(PluginRegistry::Instance().RegisterStaticPlugin(nullptr),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("null vtable")));

  RaidenStorageOutofTreePluginV1 bad_abi_vtable{};
  bad_abi_vtable.abi_version = 999;
  bad_abi_vtable.struct_size = sizeof(RaidenStorageOutofTreePluginV1);
  bad_abi_vtable.type_name = "bad_abi_plugin";
  EXPECT_THAT(PluginRegistry::Instance().RegisterStaticPlugin(&bad_abi_vtable),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("ABI version")));
}

}  // namespace
}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden
