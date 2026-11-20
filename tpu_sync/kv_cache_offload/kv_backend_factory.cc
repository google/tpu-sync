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

#include "tpu_sync/kv_cache_offload/kv_backend_factory.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/kv_cache_offload/backends/backend.h"
#include "tpu_sync/kv_cache_offload/backends/plugin/plugin_kv_backend.h"
#include "tpu_sync/kv_cache_offload/backends/plugin/plugin_registry.h"
#include "tpu_sync/kv_cache_offload/backends/plugin/raiden_storage_plugin_abi.h"
#include "tpu_sync/kv_cache_offload/kv_cache_store_backend_factory.h"

namespace tpu_raiden {
namespace kv_cache {

namespace {

KVBackendFactory::BackendCreator MakePluginWorkerCreator(
    const RaidenStorageOutofTreePluginV1* vtable) {
  return [vtable](const BackendConfig& config)
             -> absl::StatusOr<std::shared_ptr<backends::KVBackend>> {
    ABSL_RETURN_IF_ERROR(ValidateWorkerParallelism(config.parallelism));
    BackendConfig resolved = config;
    if (HasDeclaredAxis(config.parallelism)) {
      ApplyParallelismToProperties(config.parallelism, &resolved);
    }
    resolved.SetProperty(RAIDEN_PLUGIN_PROPERTY_ROLE,
                         RAIDEN_PLUGIN_ROLE_WORKER);
    return backends::PluginKVBackend::Create(vtable, resolved.properties,
                                             resolved.parallelism);
  };
}

}  // namespace

KVBackendFactory& KVBackendFactory::Instance() {
  static absl::NoDestructor<KVBackendFactory> instance;
  return *instance;
}

absl::Status KVBackendFactory::RegisterBackend(absl::string_view type_name,
                                               BackendCreator creator) {
  absl::MutexLock lock(mutex_);
  auto [it, inserted] =
      creators_.try_emplace(std::string(type_name), std::move(creator));
  if (!inserted) {
    return absl::AlreadyExistsError(absl::StrCat("KVBackend type '", type_name,
                                                 "' is already registered."));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<backends::KVBackend>>
KVBackendFactory::CreateBackend(const BackendConfig& config) {
  BackendCreator creator;
  {
    absl::MutexLock lock(mutex_);
    if (auto it = creators_.find(config.type); it != creators_.end()) {
      creator = it->second;
    }
  }
  if (!creator) {
    absl::StatusOr<const RaidenStorageOutofTreePluginV1*> vtable_or =
        backends::PluginRegistry::Instance().LoadPlugin(
            config.type, config.ResolveDynamicallyLoadedBackendPath());
    if (!vtable_or.ok()) {
      if (absl::IsNotFound(vtable_or.status())) {
        return absl::NotFoundError(
            absl::StrCat("Unknown KVBackend type '", config.type,
                         "'. Ensure it is registered in KVBackendFactory."));
      }
      return vtable_or.status();
    }
    BackendCreator plugin_creator = MakePluginWorkerCreator(*vtable_or);
    absl::MutexLock lock(mutex_);
    auto [it, _] = creators_.try_emplace(config.type, plugin_creator);
    const std::string canonical_type = absl::AsciiStrToLower(config.type);
    if (canonical_type != config.type) {
      creators_.try_emplace(canonical_type, plugin_creator);
    }
    creator = it->second;
  }
  // Run the creator outside `mutex_` so nested registrations or slow backend
  // initialization do not block concurrent lookups.
  return creator(config);
}

bool KVBackendFactory::IsRegistered(absl::string_view type_name) const {
  absl::MutexLock lock(mutex_);
  return creators_.contains(type_name);
}

std::vector<std::string> KVBackendFactory::GetRegisteredTypes() const {
  absl::MutexLock lock(mutex_);
  std::vector<std::string> types;
  types.reserve(creators_.size());
  for (const auto& [type, _] : creators_) {
    types.push_back(type);
  }
  return types;
}

}  // namespace kv_cache
}  // namespace tpu_raiden
