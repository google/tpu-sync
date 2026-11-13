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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_PLUGIN_PLUGIN_REGISTRY_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_PLUGIN_PLUGIN_REGISTRY_H_

#include <string>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/kv_cache/backends/plugin/raiden_storage_plugin_abi.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {

// Thread-safe dynamic loader and registry for out-of-tree C ABI storage
// plugins (`RaidenStorageOutofTreePluginV1`).
class PluginRegistry {
 public:
  static PluginRegistry& Instance();

  // Loads the shared library for `type_name` if not already loaded, validates
  // its `RaidenStorageOutofTreePluginV1` descriptor, and registers `type_name`
  // in both `KVBackendFactory` and `KVCacheStoreBackendFactory`.
  //
  // Resolution order when `explicit_plugin_path` is empty:
  // 1. `RAIDEN_STORAGE_PLUGIN_PATH_<UPPER_TYPE>` environment variable.
  // 2. `<RAIDEN_STORAGE_PLUGIN_DIR>/libraiden_plugin_<lower_type>.so`.
  absl::Status EnsurePluginLoaded(absl::string_view type_name,
                                  absl::string_view explicit_plugin_path = "");

  // Registers an in-process `RaidenStorageOutofTreePluginV1` vtable directly
  // without `dlopen` (useful for unit tests and statically linked plugins).
  absl::Status RegisterStaticPlugin(
      const RaidenStorageOutofTreePluginV1* vtable);

  // Returns the loaded plugin descriptor for `type_name`, or nullptr if not
  // loaded.
  const RaidenStorageOutofTreePluginV1* GetPlugin(
      absl::string_view type_name) const;

 private:
  friend class absl::NoDestructor<PluginRegistry>;

  struct LoadedPlugin {
    void* dl_handle = nullptr;
    const RaidenStorageOutofTreePluginV1* vtable = nullptr;
    std::string resolved_path;
  };

  PluginRegistry();
  ~PluginRegistry() = default;

  PluginRegistry(const PluginRegistry&) = delete;
  PluginRegistry& operator=(const PluginRegistry&) = delete;

  absl::StatusOr<std::string> ResolvePluginPath(
      absl::string_view canonical_type,
      absl::string_view explicit_plugin_path) const;

  absl::Status RegisterPluginFactories(
      absl::string_view canonical_type,
      const RaidenStorageOutofTreePluginV1* vtable);

  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, LoadedPlugin> plugins_ ABSL_GUARDED_BY(mu_);
};

}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_PLUGIN_PLUGIN_REGISTRY_H_
