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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_WRAPPER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_WRAPPER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/kv_cache_store.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"

namespace tpu_raiden {
namespace kv_cache {

// The StoreMonitorConfig a Python-driven store runs with. The bindings
// expose no monitor arguments; like the shm wiring below, the knobs come
// from the environment:
//   RAIDEN_ENABLE_STORE_MONITOR       "true"/"1" runs the StoreMonitor
//   RAIDEN_ENABLE_EVICT_SWEEP         "true"/"1" runs the evict sweep
//   RAIDEN_STORE_MONITOR_HEARTBEAT_S  heartbeat period, whole seconds
//   RAIDEN_EVICT_SWEEP_PERIOD_S       sweep fallback period, whole seconds
//   RAIDEN_EVICT_LOW_WATERMARK        free-block ratio in (0, 1)
//   RAIDEN_EVICT_HIGH_WATERMARK       free-block ratio in (0, 1)
// A numeric value that is unset -- or invalid, which logs an error -- stays
// zero, which KVCacheStore::Create resolves to the built-in default. The
// wrapper honors the two switches only when constructed with a
// global_registry_address; a registry-less store drops them, so the fleet's
// shared env block never breaks a local-only boot.
StoreMonitorConfig StoreMonitorConfigFromEnv();

// Constructs the KVCacheStore behind the framework Python bindings (the jax
// and torch modules both bind this class).
class KVCacheStoreWrapper {
 public:
  explicit KVCacheStoreWrapper(
      size_t lru_capacity, std::string global_registry_address = "",
      RaidenId raiden_id = {}, int num_shards = 0, int64_t shard_size_bytes = 0,
      std::string store_server_ip = "", int raiden_controller_port = 0,
      int expected_worker_count = 0, std::string kv_pool_group = "");

  explicit KVCacheStoreWrapper(std::unique_ptr<KVCacheStore> store)
      : controller_(std::move(store)) {}

  KVCacheStore* operator->() { return controller_.get(); }
  KVCacheStore& operator*() { return *controller_; }

 private:
  std::unique_ptr<KVCacheStore> controller_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_WRAPPER_H_
