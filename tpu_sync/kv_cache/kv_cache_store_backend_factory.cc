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
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/kv_cache/backends/backend.h"

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

void ApplyParallelismToProperties(
    const backends::ParallelismConfig& parallelism, BackendConfig* config) {
  auto apply = [config](absl::string_view key, int value) {
    if (value == backends::kAxisUndeclared) {
      config->properties.erase(key);
      return;
    }
    config->SetProperty(key, absl::StrCat(value));
  };
  apply("tp_size", parallelism.tp_size);
  apply("tp_rank", parallelism.tp_rank);
  apply("pcp_size", parallelism.pcp_size);
  apply("pcp_rank", parallelism.pcp_rank);
  apply("pp_size", parallelism.pp_size);
  apply("pp_rank", parallelism.pp_rank);
}

namespace {

absl::Status ValidateWorkerAxis(absl::string_view axis, int size, int rank) {
  if (size == 0 || size < backends::kAxisUndeclared) {
    return absl::InvalidArgumentError(
        absl::StrCat(axis, "_size must be >= 1, or -1 when the ", axis,
                     " axis is undeclared; got ", size));
  }
  if (size == backends::kAxisUndeclared) {
    if (rank == backends::kAxisUndeclared) return absl::OkStatus();
    return absl::InvalidArgumentError(absl::StrCat(
        axis, "_rank ", rank, " was given without ", axis, "_size; declare ",
        axis, "_size >= 1 or leave ", axis, "_rank at -1"));
  }
  if (rank < 0 || rank >= size) {
    return absl::InvalidArgumentError(
        absl::StrCat(axis, "_rank must be in [0, ", size, ") for ", axis,
                     "_size ", size, "; got ", rank));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateWorkerParallelism(
    const backends::ParallelismConfig& parallelism) {
  ABSL_RETURN_IF_ERROR(
      ValidateWorkerAxis("tp", parallelism.tp_size, parallelism.tp_rank));
  ABSL_RETURN_IF_ERROR(
      ValidateWorkerAxis("pcp", parallelism.pcp_size, parallelism.pcp_rank));
  return ValidateWorkerAxis("pp", parallelism.pp_size, parallelism.pp_rank);
}

bool HasDeclaredAxis(const backends::ParallelismConfig& parallelism) {
  return parallelism.tp_size != backends::kAxisUndeclared ||
         parallelism.pcp_size != backends::kAxisUndeclared ||
         parallelism.pp_size != backends::kAxisUndeclared;
}

std::string FormatParallelism(const backends::ParallelismConfig& parallelism) {
  auto format_axis = [](absl::string_view axis, int size, int rank) {
    std::string formatted = absl::StrCat(
        axis, "=",
        size == backends::kAxisUndeclared ? "undeclared" : absl::StrCat(size));
    if (rank != backends::kAxisUndeclared) {
      absl::StrAppend(&formatted, "/r", rank);
    }
    return formatted;
  };
  return absl::StrCat(
      format_axis("pp", parallelism.pp_size, parallelism.pp_rank), " ",
      format_axis("pcp", parallelism.pcp_size, parallelism.pcp_rank), " ",
      format_axis("tp", parallelism.tp_size, parallelism.tp_rank));
}

absl::Status RequireDeclaredAxis(
    const backends::ParallelismConfig& received,
    const backends::ParallelismConfig& coordinator) {
  if (HasDeclaredAxis(coordinator)) return absl::OkStatus();
  return absl::InvalidArgumentError(absl::StrCat(
      "no parallelism axis is declared (received topology: ",
      FormatParallelism(received),
      "); declare the axes and sizes every worker declares (tp_size, "
      "pcp_size, pp_size); a single worker uses tp_size=1"));
}

absl::StatusOr<backends::ParallelismConfig> ResolveCoordinatorParallelism(
    const backends::ParallelismConfig& parallelism) {
  // An undeclared axis must not carry a rank: for size == -1,
  // ValidateWorkerAxis accepts only rank == -1.
  if (parallelism.tp_size == backends::kAxisUndeclared) {
    ABSL_RETURN_IF_ERROR(
        ValidateWorkerAxis("tp", parallelism.tp_size, parallelism.tp_rank));
  }
  if (parallelism.pcp_size == backends::kAxisUndeclared) {
    ABSL_RETURN_IF_ERROR(
        ValidateWorkerAxis("pcp", parallelism.pcp_size, parallelism.pcp_rank));
  }
  if (parallelism.pp_size == backends::kAxisUndeclared) {
    ABSL_RETURN_IF_ERROR(
        ValidateWorkerAxis("pp", parallelism.pp_size, parallelism.pp_rank));
  }
  auto pinned_rank = [](int size) {
    return size >= 1 ? 0 : backends::kAxisUndeclared;
  };
  const backends::ParallelismConfig coordinator = {
      .tp_size = parallelism.tp_size,
      .tp_rank = pinned_rank(parallelism.tp_size),
      .pcp_size = parallelism.pcp_size,
      .pcp_rank = pinned_rank(parallelism.pcp_size),
      .pp_size = parallelism.pp_size,
      .pp_rank = pinned_rank(parallelism.pp_size),
  };
  // With the ranks pinned, only an invalid size can fail.
  ABSL_RETURN_IF_ERROR(ValidateWorkerParallelism(coordinator));
  return coordinator;
}

KVCacheStoreBackendFactory& KVCacheStoreBackendFactory::Instance() {
  static absl::NoDestructor<KVCacheStoreBackendFactory> instance;
  return *instance;
}

KVCacheStoreBackendFactory::KVCacheStoreBackendFactory() {
  RegisterBuiltInBackends();
}

void KVCacheStoreBackendFactory::RegisterBuiltInBackends() {}

void KVCacheStoreBackendFactory::SetPluginLoader(PluginLoader loader) {
  absl::MutexLock lock(mutex_);
  plugin_loader_ = std::move(loader);
}

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
  PluginLoader loader;
  {
    absl::MutexLock lock(mutex_);
    auto it = creators_.find(config.type);
    if (it != creators_.end()) {
      creator = it->second;
    } else {
      loader = plugin_loader_;
    }
  }
  if (!creator && loader) {
    const std::string effective_plugin_path =
        !config.plugin_path.empty() ? config.plugin_path
                                    : config.GetProperty("plugin_path");
    absl::Status load_status = loader(config.type, effective_plugin_path);
    if (!load_status.ok()) {
      if (!absl::IsNotFound(load_status) || !effective_plugin_path.empty()) {
        return load_status;
      }
    } else {
      absl::MutexLock lock(mutex_);
      auto it = creators_.find(config.type);
      if (it != creators_.end()) {
        creator = it->second;
      }
    }
  }
  if (!creator) {
    return absl::NotFoundError(absl::StrCat(
        "Unknown backend type '", config.type,
        "'. Ensure it is registered in KVCacheStoreBackendFactory."));
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
