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

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace tpu_raiden {
namespace kv_cache {

std::string BackendConfig::GetProperty(absl::string_view key,
                                       absl::string_view default_val) const {
  auto it = properties.find(key);
  return (it != properties.end()) ? it->second : std::string(default_val);
}

bool BackendConfig::GetBoolProperty(absl::string_view key,
                                    bool default_val) const {
  auto it = properties.find(key);
  if (it == properties.end()) {
    return default_val;
  }
  absl::string_view val = it->second;
  if (absl::EqualsIgnoreCase(val, "true") || val == "1") {
    return true;
  }
  if (absl::EqualsIgnoreCase(val, "false") || val == "0") {
    return false;
  }
  return default_val;
}

int64_t BackendConfig::GetIntProperty(absl::string_view key,
                                      int64_t default_val) const {
  auto it = properties.find(key);
  if (it == properties.end()) {
    return default_val;
  }
  int64_t res = 0;
  if (absl::SimpleAtoi(it->second, &res)) {
    return res;
  }
  return default_val;
}

void BackendConfig::SetProperty(absl::string_view key,
                                absl::string_view value) {
  properties[std::string(key)] = std::string(value);
}

bool BackendConfig::HasProperty(absl::string_view key) const {
  return properties.contains(key);
}

KVCacheStoreBackendFactory& KVCacheStoreBackendFactory::Instance() {
  static absl::NoDestructor<KVCacheStoreBackendFactory> instance;
  return *instance;
}

KVCacheStoreBackendFactory::KVCacheStoreBackendFactory() {
  RegisterBuiltInBackends();
}

void KVCacheStoreBackendFactory::RegisterBuiltInBackends() {}

absl::Status KVCacheStoreBackendFactory::RegisterBackend(
    absl::string_view type_name, BackendCreator creator) {
  absl::MutexLock lock(mutex_);
  auto [it, inserted] =
      creators_.try_emplace(std::string(type_name), std::move(creator));
  if (!inserted) {
    return absl::AlreadyExistsError(
        absl::StrCat("Backend type '", type_name, "' is already registered."));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<KVCacheStoreBackend>>
KVCacheStoreBackendFactory::CreateBackend(
    const BackendConfig& config,
    controller::RaidenController* controller) const {
  BackendCreator creator;
  {
    absl::MutexLock lock(mutex_);
    auto it = creators_.find(config.type);
    if (it == creators_.end()) {
      return absl::NotFoundError(absl::StrCat(
          "Unknown backend type '", config.type,
          "'. Ensure it is registered in KVCacheStoreBackendFactory."));
    }
    creator = it->second;
  }
  // Execute creator OUTSIDE the lock to ensure thread safety and avoid
  // deadlock during recursive instantiation of sub-backends.
  return creator(config, controller);
}

bool KVCacheStoreBackendFactory::IsRegistered(
    absl::string_view type_name) const {
  absl::MutexLock lock(mutex_);
  return creators_.contains(type_name);
}

std::vector<std::string> KVCacheStoreBackendFactory::GetRegisteredTypes()
    const {
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
