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

#include <dlfcn.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/kv_cache/backends/backend.h"
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

absl::Status ValidatePluginVtable(const RaidenStorageOutofTreePluginV1* vtable,
                                  absl::string_view source_label) {
  if (vtable == nullptr) {
    return absl::FailedPreconditionError(
        absl::StrCat("Plugin '", source_label, "' returned a null vtable"));
  }
  if (vtable->abi_version != RAIDEN_STORAGE_PLUGIN_ABI_VERSION) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Plugin '", source_label, "' has ABI version ", vtable->abi_version,
        ", expected ", RAIDEN_STORAGE_PLUGIN_ABI_VERSION));
  }
  if (vtable->struct_size < RAIDEN_STORAGE_PLUGIN_V1_MIN_SIZE) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Plugin '", source_label, "' has struct_size ", vtable->struct_size,
        ", smaller than minimum v1 size ", RAIDEN_STORAGE_PLUGIN_V1_MIN_SIZE));
  }
  if (vtable->type_name == nullptr || vtable->type_name[0] == '\0') {
    return absl::FailedPreconditionError(
        absl::StrCat("Plugin '", source_label, "' has an empty type_name"));
  }
  if (vtable->create == nullptr || vtable->destroy == nullptr ||
      vtable->write_async == nullptr || vtable->read_async == nullptr ||
      vtable->batch_exists_async == nullptr) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Plugin '", source_label, "' is missing required function pointers"));
  }
  return absl::OkStatus();
}

// Reads the coordinator-side options of a plugin backend from `config`.
// `capacity_bytes` and `lookup_batch_size` have the names and meanings that the
// built-in storage backends use. The `metadata_cache_*` properties are not
// read, because the plugin backend has no existence cache.
OutofTreePluginKVCacheStoreBackend::Options ParseStoreOptions(
    const BackendConfig& config) {
  OutofTreePluginKVCacheStoreBackend::Options options;
  options.capacity_bytes = static_cast<size_t>(std::max<int64_t>(
      0, config.GetIntProperty("capacity_bytes",
                               static_cast<int64_t>(config.capacity))));
  options.lookup_batch_size = static_cast<size_t>(
      std::max<int64_t>(0, config.GetIntProperty("lookup_batch_size", 0)));
  options.dp_rank = absl::StrCat(config.raiden_id.data_replica_idx);
  return options;
}

}  // namespace

PluginRegistry& PluginRegistry::Instance() {
  static absl::NoDestructor<PluginRegistry> instance;
  return *instance;
}

PluginRegistry::PluginRegistry() {
  auto loader = [this](absl::string_view type_name,
                       absl::string_view plugin_path) -> absl::Status {
    return EnsurePluginLoaded(type_name, plugin_path);
  };
  KVCacheStoreBackendFactory::Instance().SetPluginLoader(loader);
  KVBackendFactory::Instance().SetPluginLoader(loader);
}

absl::StatusOr<std::string> PluginRegistry::ResolvePluginPath(
    absl::string_view canonical_type,
    absl::string_view explicit_plugin_path) const {
  if (!explicit_plugin_path.empty()) {
    return std::string(explicit_plugin_path);
  }
  const std::string env_var = absl::StrCat(
      "RAIDEN_STORAGE_PLUGIN_PATH_", absl::AsciiStrToUpper(canonical_type));
  if (const char* env_path = std::getenv(env_var.c_str());
      env_path != nullptr && env_path[0] != '\0') {
    return std::string(env_path);
  }
  if (const char* env_dir = std::getenv("RAIDEN_STORAGE_PLUGIN_DIR");
      env_dir != nullptr && env_dir[0] != '\0') {
    return absl::StrCat(env_dir, "/libraiden_plugin_", canonical_type, ".so");
  }
  return absl::NotFoundError(absl::StrCat(
      "No plugin path configured for backend type '", canonical_type, "'"));
}

absl::Status PluginRegistry::EnsurePluginLoaded(
    absl::string_view type_name, absl::string_view explicit_plugin_path) {
  if (type_name.empty()) {
    return absl::InvalidArgumentError("Plugin type_name must not be empty");
  }
  const std::string canonical_type = absl::AsciiStrToLower(type_name);
  {
    absl::MutexLock lock(mu_);
    if (plugins_.contains(canonical_type)) {
      return absl::OkStatus();
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      const std::string resolved_path,
      ResolvePluginPath(canonical_type, explicit_plugin_path));

  const RaidenStorageOutofTreePluginV1* vtable = nullptr;
  {
    absl::MutexLock lock(mu_);
    if (plugins_.contains(canonical_type)) {
      return absl::OkStatus();
    }

    dlerror();
    void* handle = dlopen(resolved_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      const char* err = dlerror();
      return absl::FailedPreconditionError(
          absl::StrCat("Failed to dlopen storage plugin '", resolved_path,
                       "' for backend type '", canonical_type,
                       "': ", err != nullptr ? err : "unknown dlopen error"));
    }

    dlerror();
    void* sym = dlsym(handle, RAIDEN_STORAGE_PLUGIN_ENTRY_SYMBOL);
    if (sym == nullptr) {
      const char* err = dlerror();
      dlclose(handle);
      return absl::FailedPreconditionError(absl::StrCat(
          "Storage plugin '", resolved_path, "' is missing entry symbol '",
          RAIDEN_STORAGE_PLUGIN_ENTRY_SYMBOL,
          "': ", err != nullptr ? err : "symbol not found"));
    }

    auto entry_fn = reinterpret_cast<RaidenStoragePluginGetV1Fn>(sym);
    vtable = entry_fn();
    if (absl::Status status = ValidatePluginVtable(vtable, resolved_path);
        !status.ok()) {
      dlclose(handle);
      return status;
    }

    LoadedPlugin loaded;
    loaded.dl_handle = handle;
    loaded.vtable = vtable;
    loaded.resolved_path = resolved_path;
    plugins_[canonical_type] = loaded;

    const std::string vtable_type = absl::AsciiStrToLower(vtable->type_name);
    if (vtable_type != canonical_type) {
      plugins_.try_emplace(vtable_type, loaded);
    }
  }

  ABSL_RETURN_IF_ERROR(RegisterPluginFactories(canonical_type, vtable));
  const std::string vtable_type = absl::AsciiStrToLower(vtable->type_name);
  if (vtable_type != canonical_type) {
    ABSL_RETURN_IF_ERROR(RegisterPluginFactories(vtable_type, vtable));
  }
  return absl::OkStatus();
}

absl::Status PluginRegistry::RegisterStaticPlugin(
    const RaidenStorageOutofTreePluginV1* vtable) {
  ABSL_RETURN_IF_ERROR(ValidatePluginVtable(vtable, "static_plugin"));
  const std::string canonical_type = absl::AsciiStrToLower(vtable->type_name);
  {
    absl::MutexLock lock(mu_);
    LoadedPlugin loaded;
    loaded.dl_handle = nullptr;
    loaded.vtable = vtable;
    loaded.resolved_path = "<static>";
    plugins_[canonical_type] = loaded;
  }
  return RegisterPluginFactories(canonical_type, vtable);
}

const RaidenStorageOutofTreePluginV1* PluginRegistry::GetPlugin(
    absl::string_view type_name) const {
  const std::string canonical_type = absl::AsciiStrToLower(type_name);
  absl::MutexLock lock(mu_);
  auto it = plugins_.find(canonical_type);
  return it != plugins_.end() ? it->second.vtable : nullptr;
}

absl::Status PluginRegistry::RegisterPluginFactories(
    absl::string_view canonical_type,
    const RaidenStorageOutofTreePluginV1* vtable) {
  const std::string type_str(canonical_type);
  if (!KVBackendFactory::Instance().IsRegistered(type_str)) {
    absl::Status status = KVBackendFactory::Instance().RegisterBackend(
        type_str,
        [vtable](const BackendConfig& config)
            -> absl::StatusOr<std::shared_ptr<KVBackend>> {
          BackendConfig resolved = config;
          if (HasDeclaredAxis(config.parallelism)) {
            ABSL_RETURN_IF_ERROR(ValidateWorkerParallelism(config.parallelism));
            ApplyParallelismToProperties(config.parallelism, &resolved);
          }
          resolved.SetProperty(RAIDEN_PLUGIN_PROPERTY_ROLE,
                               RAIDEN_PLUGIN_ROLE_WORKER);
          return PluginKVBackend::Create(vtable, resolved.properties,
                                         resolved.parallelism);
        });
    if (!status.ok() && !absl::IsAlreadyExists(status)) {
      return status;
    }
  }

  if (!KVCacheStoreBackendFactory::Instance().IsRegistered(type_str)) {
    absl::Status status =
        KVCacheStoreBackendFactory::Instance().RegisterBackend(
            type_str,
            [vtable, type_str](const BackendConfig& config,
                               controller::RaidenController* /*controller*/)
                -> absl::StatusOr<std::shared_ptr<KVCacheStoreBackend>> {
              ABSL_ASSIGN_OR_RETURN(
                  const ParallelismConfig coordinator,
                  ResolveCoordinatorParallelism(config.parallelism));
              ABSL_RETURN_IF_ERROR(
                  RequireDeclaredAxis(config.parallelism, coordinator));
              BackendConfig resolved = config;
              ApplyParallelismToProperties(coordinator, &resolved);
              // The pinned ranks make the coordinator instance look like the
              // rank 0 worker, so the role property is what tells them apart.
              resolved.SetProperty(RAIDEN_PLUGIN_PROPERTY_ROLE,
                                   RAIDEN_PLUGIN_ROLE_COORDINATOR);

              ABSL_ASSIGN_OR_RETURN(
                  std::shared_ptr<PluginKVBackend> plugin_backend,
                  PluginKVBackend::Create(vtable, resolved.properties,
                                          coordinator));

              return std::make_shared<OutofTreePluginKVCacheStoreBackend>(
                  std::move(plugin_backend), type_str,
                  ParseStoreOptions(resolved));
            });
    if (!status.ok() && !absl::IsAlreadyExists(status)) {
      return status;
    }
  }

  return absl::OkStatus();
}

namespace {
struct PluginRegistryInitializer {
  PluginRegistryInitializer() { (void)PluginRegistry::Instance(); }
};
static PluginRegistryInitializer g_plugin_registry_init;
}  // namespace

}  // namespace backends
}  // namespace kv_cache
}  // namespace tpu_raiden
