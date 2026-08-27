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

#include "tpu_sync/kv_cache/kv_cache_store_wrapper.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/kv_cache_metadata.h"
#include "tpu_sync/kv_cache/kv_cache_metadata_shm.h"
#include "tpu_sync/kv_cache/kv_cache_store.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

// Names the metadata segment after the KV pool segments: RAIDEN_SHM_KEY
// plus the server-name suffix (but no per-device suffix — the table spans
// the store, not one device).
std::string MetadataShmKey() {
  const char* shm_key = std::getenv("RAIDEN_SHM_KEY");
  if (shm_key == nullptr || std::strlen(shm_key) == 0) {
    return "";
  }
  std::string key = absl::StrCat(shm_key, "_metadata");
  const char* server_name = std::getenv("RAIDEN_SHM_SERVER_NAME");
  if (server_name != nullptr && std::strlen(server_name) > 0) {
    absl::StrAppend(&key, "_", server_name);
  }
  return key;
}

// "true"/"1" means enabled, anything else (including unset) does not, the
// convention of the codebase's other enable-style switches
// (RAIDEN_DISABLE_SINGLETON_WORKER, ENABLE_RAIDEN_METRICS).
bool BoolFromEnv(const char* name) {
  const char* raw = std::getenv(name);
  return raw != nullptr &&
         (std::strcmp(raw, "true") == 0 || std::strcmp(raw, "1") == 0);
}

// Reads a duration from an env var holding whole seconds. Zero (meaning
// "the built-in default") if unset, unparseable, or non-positive.
absl::Duration DurationFromEnv(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr) return absl::ZeroDuration();
  int64_t seconds = 0;
  if (!absl::SimpleAtoi(raw, &seconds) || seconds <= 0) {
    LOG(ERROR) << name << "=\"" << raw
               << "\" is not a positive integer number of seconds; using "
                  "the built-in default";
    return absl::ZeroDuration();
  }
  return absl::Seconds(seconds);
}

// Reads a ratio in (0, 1). Zero (meaning "the built-in default") if unset,
// unparseable, or out of range.
double RatioFromEnv(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr) return 0.0;
  double ratio = 0.0;
  // The negated form also rejects NaN, whose comparisons are all false.
  if (!absl::SimpleAtod(raw, &ratio) || !(ratio > 0.0 && ratio < 1.0)) {
    LOG(ERROR) << name << "=\"" << raw
               << "\" is not a ratio in (0, 1); using the built-in default";
    return 0.0;
  }
  return ratio;
}

}  // namespace

StoreMonitorConfig StoreMonitorConfigFromEnv() {
  StoreMonitorConfig config;
  config.enable = BoolFromEnv("RAIDEN_ENABLE_STORE_MONITOR");
  config.heartbeat_period = DurationFromEnv("RAIDEN_STORE_MONITOR_HEARTBEAT_S");
  config.enable_evict_sweep = BoolFromEnv("RAIDEN_ENABLE_EVICT_SWEEP");
  config.evict_sweep_period = DurationFromEnv("RAIDEN_EVICT_SWEEP_PERIOD_S");
  config.evict_low_watermark = RatioFromEnv("RAIDEN_EVICT_LOW_WATERMARK");
  config.evict_high_watermark = RatioFromEnv("RAIDEN_EVICT_HIGH_WATERMARK");
  return config;
}

KVCacheStoreWrapper::KVCacheStoreWrapper(
    size_t lru_capacity, std::string global_registry_address,
    RaidenId raiden_id, int num_shards, int64_t shard_size_bytes,
    std::string store_server_ip, int raiden_controller_port,
    int expected_worker_count, std::string kv_pool_group) {
  std::optional<KVCacheMetadata> metadata;
  if (num_shards > 0) {
    std::string shm_key = MetadataShmKey();
    if (!shm_key.empty()) {
      const char* model_uid = std::getenv("RAIDEN_SHM_MODEL_UID");
      auto region_or = KVCacheMetadataShmRegion::AttachOrFormat(
          shm_key, static_cast<int>(lru_capacity),
          model_uid != nullptr ? model_uid : "default_model");
      if (region_or.ok()) {
        metadata_region_ = *std::move(region_or);
        metadata = metadata_region_->metadata();
      } else {
        LOG(WARNING) << "KV metadata table unavailable, serving without "
                        "crash recovery: "
                     << region_or.status().message();
      }
    }
  }

  // Routed through Create() (not the raw constructor) so a misconfigured
  // caller -- e.g. a missing store_server_ip -- gets a Python exception
  // instead of aborting the process.
  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = lru_capacity;
  config.kv_pool_group = std::move(kv_pool_group);
  // Serving hosts sit on placement tier 0 (BackendConfig's default
  // evict_tier), demoting to higher tiers, never receiving.
  config.monitor_config = StoreMonitorConfigFromEnv();
  if (global_registry_address.empty()) {
    // The env block is shared across a fleet's processes, so the switches
    // must not break a registry-less (local-only) store: without a registry
    // the monitor has nothing to heartbeat and the sweep no way to find
    // targets, so they are dropped -- Create rejects the combination.
    config.monitor_config.enable = false;
    config.monitor_config.enable_evict_sweep = false;
  }
  auto created_store = KVCacheStore::Create(
      config, /*capacity=*/lru_capacity, global_registry_address, raiden_id,
      num_shards, shard_size_bytes, store_server_ip, raiden_controller_port,
      metadata, expected_worker_count);
  if (!created_store.ok()) {
    // invalid_argument maps to Python ValueError, runtime_error to
    // RuntimeError: a bad configuration is the caller's mistake, everything
    // else (e.g. the expected-worker barrier timing out) is a runtime failure.
    if (created_store.status().code() == absl::StatusCode::kInvalidArgument ||
        created_store.status().code() ==
            absl::StatusCode::kFailedPrecondition) {
      throw std::invalid_argument(
          std::string(created_store.status().message()));
    }
    throw std::runtime_error(std::string(created_store.status().message()));
  }
  controller_ = std::move(*created_store);

  if (metadata_region_ != nullptr && metadata_region_->warm()) {
    auto recovered_or = controller_->RecoverFromLocalManifest();
    if (recovered_or.ok()) {
      LOG(INFO) << "Recovered " << *recovered_or
                << " blocks from the local KV metadata table";
    } else {
      LOG(WARNING) << "KV metadata recovery failed, falling back to a cold "
                      "start: "
                   << recovered_or.status().message();
      absl::Status reformat_status = metadata_region_->Reformat();
      if (!reformat_status.ok()) {
        LOG(WARNING) << "Failed to reformat the KV metadata table: "
                     << reformat_status.message();
      }
    }
  }
}

}  // namespace kv_cache
}  // namespace tpu_raiden
