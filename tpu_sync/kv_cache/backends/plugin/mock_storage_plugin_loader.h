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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_PLUGIN_MOCK_STORAGE_PLUGIN_LOADER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_PLUGIN_MOCK_STORAGE_PLUGIN_LOADER_H_

#include <dlfcn.h>

#include <cstdlib>
#include <string>

#include "absl/strings/str_cat.h"
#include "tpu_sync/kv_cache/backends/plugin/raiden_storage_plugin_abi.h"

namespace tpu_raiden {
namespace kv_cache {
namespace backends {

// Returns the path of libmock_storage_plugin.so in the runfiles of the
// running test.
inline std::string GetMockPluginSoPath() {
  const char* srcdir = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (srcdir != nullptr && srcdir[0] != '\0') {
    std::string ws =
        (workspace != nullptr && workspace[0] != '\0') ? workspace : "google3";
    return absl::StrCat(
        srcdir, "/", ws,
        "/tpu_sync/kv_cache/backends/plugin/"
        "libmock_storage_plugin.so");
  }
  return "tpu_sync/kv_cache/backends/plugin/"
         "libmock_storage_plugin.so";
}

// Loads libmock_storage_plugin.so and returns its function table, or nullptr
// if the library or its entry symbol cannot be found. The library stays loaded
// until the process exits.
inline const RaidenStorageOutofTreePluginV1* LoadMockPlugin() {
  void* handle = dlopen(GetMockPluginSoPath().c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) return nullptr;
  void* entry = dlsym(handle, RAIDEN_STORAGE_PLUGIN_ENTRY_SYMBOL);
  if (entry == nullptr) return nullptr;
  return reinterpret_cast<RaidenStoragePluginGetV1Fn>(entry)();
}

}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_BACKENDS_PLUGIN_MOCK_STORAGE_PLUGIN_LOADER_H_
