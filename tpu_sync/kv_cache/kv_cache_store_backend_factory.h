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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_BACKEND_FACTORY_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_BACKEND_FACTORY_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/backends/backend.h"
#include "tpu_sync/kv_cache/kv_cache_metadata.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"

namespace tpu_raiden {

namespace controller {
class RaidenController;
}  // namespace controller

namespace kv_cache {

// The deployment's KVTransferSpec (see global_registry.proto), registered
// with the global registry when a backend that supports it is created.
struct KVTransferSpecConfig {
  // Bytes one block occupies in each block array, on one shard.
  std::vector<uint64_t> block_array_bytes;
  // Device shards each block array is split across on a serving host.
  int num_kv_shards = 0;
  // Transfer workers on the serving hosts, node ids [0, num_workers).
  int num_workers = 0;
};

// The knobs behind a store's StoreMonitor: the heartbeat it sends and the
// evict sweep it schedules. Zero values mean the built-in defaults.
struct StoreMonitorConfig {
  // Runs a StoreMonitor thread that heartbeats the store's status to the
  // global registry; the store's registration then carries a TTL and expires
  // when heartbeats stop.
  bool enable = false;
  // Heartbeat period. Zero means the StoreMonitor default. With the monitor
  // enabled this also sets the registration TTL, a fixed multiple of the
  // period.
  absl::Duration heartbeat_period = absl::ZeroDuration();
  // Demotes cold blocks to a peer store on a higher evict_tier whenever free
  // blocks fall below evict_low_watermark. Requires `enable`: the sweep runs
  // on the store monitor's schedule.
  bool enable_evict_sweep = false;
  // Fallback period between sweep pressure checks; allocation pressure wakes
  // the sweep immediately. Zero means the StoreMonitor default.
  absl::Duration evict_sweep_period = absl::ZeroDuration();
  // Free-block ratio (free / total) below which the sweep starts demoting.
  // Zero means the default.
  double evict_low_watermark = 0.0;
  // Free-block ratio at which an active sweep stops; must be >= the low
  // watermark. Zero means the default.
  double evict_high_watermark = 0.0;
};

struct BackendConfig {
  std::string type;
  // Optional filesystem path to a dynamically loaded
  // `RaidenStorageOutofTreePluginV1` shared library (`.so`). When empty,
  // `PluginRegistry` falls back to `RAIDEN_STORAGE_PLUGIN_PATH_<TYPE>` or
  // `RAIDEN_STORAGE_PLUGIN_DIR`.
  std::string plugin_path;
  size_t capacity = 0;
  std::string global_registry_address;
  // Serving topology this backend instance participates in.
  backends::ParallelismConfig parallelism;
  RaidenId raiden_id;
  std::optional<KVCacheMetadata> metadata = std::nullopt;
  std::vector<BackendConfig> sub_backends;
  absl::flat_hash_map<std::string, std::string> properties;
  std::optional<KVTransferSpecConfig> kv_transfer_spec = std::nullopt;
  // KV pool group KVTransferSpecs are registered under (see
  // global_registry.proto). Empty falls back to raiden_id.job_name.
  std::string kv_pool_group;
  // Placement tier the store registers under (see StoreInfo.evict_tier).
  int32_t evict_tier = 0;
  StoreMonitorConfig monitor_config;

  std::string GetProperty(absl::string_view key,
                          absl::string_view default_val = "") const;
  bool GetBoolProperty(absl::string_view key, bool default_val = false) const;
  int64_t GetIntProperty(absl::string_view key, int64_t default_val = 0) const;
  void SetProperty(absl::string_view key, absl::string_view value);
  bool HasProperty(absl::string_view key) const;
};

// Writes the storage topology in `parallelism` into config->properties as
// "tp_size" / "tp_rank" / "pcp_size" / "pcp_rank" / "pp_size" / "pp_rank", the
// keys storage backends consume. Storage topology comes only from
// BackendConfig::parallelism: every field other than -1 overwrites the
// caller-supplied property, and the property of every -1 field is erased. Does
// not validate; the backend validates what it parses.
void ApplyParallelismToProperties(
    const backends::ParallelismConfig& parallelism, BackendConfig* config);

// Validates a worker's topology, one axis (tp, pcp, pp) at a time. An axis is
// declared when its size is set: a declared axis needs size >= 1 and
// 0 <= rank < size; an undeclared axis needs size == rank == -1. Returns
// InvalidArgumentError naming the axis and the offending values otherwise.
absl::Status ValidateWorkerParallelism(
    const backends::ParallelismConfig& parallelism);

// Returns true if at least one axis (tp, pcp, pp) has its size set, i.e. is
// not kAxisUndeclared. A rank alone does not declare its axis.
bool HasDeclaredAxis(const backends::ParallelismConfig& parallelism);

// Formats `parallelism` for logs, outermost axis first, e.g.
// "pp=2/r1 pcp=undeclared tp=4/r3". An unset (-1) size prints as
// "undeclared", and "/r<rank>" is appended whenever the rank is set. Does not
// validate, so an invalid config prints as given (e.g. "pcp=undeclared/r3").
std::string FormatParallelism(const backends::ParallelismConfig& parallelism);

// Returns InvalidArgumentError, naming the `received` topology, when
// `coordinator` declares no parallelism axis: with no axis declared every
// worker writes the same shard files. Returns OK otherwise.
absl::Status RequireDeclaredAxis(
    const backends::ParallelismConfig& received,
    const backends::ParallelismConfig& coordinator);

// Returns the coordinator's topology for `parallelism`: the same axis sizes,
// with the rank of each declared axis pinned to 0 and of each undeclared axis
// left at -1. The coordinator probes every rank explicitly, so the caller's
// ranks (worker-only) of declared axes are ignored. Returns
// InvalidArgumentError for an axis size that is neither -1 nor >= 1, or for a
// rank given without its axis size.
absl::StatusOr<backends::ParallelismConfig> ResolveCoordinatorParallelism(
    const backends::ParallelismConfig& parallelism);

// Factory registry for dynamic creation of KVCacheStoreBackend instances.
class KVCacheStoreBackendFactory {
 public:
  using BackendCreator =
      std::function<absl::StatusOr<std::shared_ptr<KVCacheStoreBackend>>(
          const BackendConfig& config,
          controller::RaidenController* controller)>;
  using PluginLoader = std::function<absl::Status(
      absl::string_view type_name, absl::string_view plugin_path)>;

  // Access the singleton instance of the factory registry.
  static KVCacheStoreBackendFactory& Instance();

  // Registers a fallback loader invoked when `CreateBackend` encounters an
  // unregistered backend type.
  void SetPluginLoader(PluginLoader loader);

  // Registers a creator function for the given type_name.
  // Returns Status::AlreadyExists if type_name is already registered.
  absl::Status RegisterBackend(absl::string_view type_name,
                               BackendCreator creator);

  // Constructs a backend instance based on the provided configuration.
  absl::StatusOr<std::shared_ptr<KVCacheStoreBackend>> CreateBackend(
      const BackendConfig& config,
      controller::RaidenController* controller = nullptr) const;

  // Returns whether a backend type is registered.
  bool IsRegistered(absl::string_view type_name) const;

  // Returns list of all registered backend type names.
  std::vector<std::string> GetRegisteredTypes() const;

  // Convenience static wrappers
  static absl::Status Register(absl::string_view type_name,
                               BackendCreator creator) {
    return Instance().RegisterBackend(type_name, std::move(creator));
  }

  static absl::StatusOr<std::shared_ptr<KVCacheStoreBackend>> Create(
      const BackendConfig& config,
      controller::RaidenController* controller = nullptr) {
    return Instance().CreateBackend(config, controller);
  }

 private:
  friend class absl::NoDestructor<KVCacheStoreBackendFactory>;

  KVCacheStoreBackendFactory();
  ~KVCacheStoreBackendFactory() = default;

  KVCacheStoreBackendFactory(const KVCacheStoreBackendFactory&) = delete;
  KVCacheStoreBackendFactory& operator=(const KVCacheStoreBackendFactory&) =
      delete;

  void RegisterBuiltInBackends();

  mutable absl::Mutex mutex_;
  absl::flat_hash_map<std::string, BackendCreator> creators_
      ABSL_GUARDED_BY(mutex_);
  PluginLoader plugin_loader_ ABSL_GUARDED_BY(mutex_);
};

namespace internal {
// Helper struct for static auto-registration macros
struct BackendRegistrar {
  BackendRegistrar(absl::string_view type_name,
                   KVCacheStoreBackendFactory::BackendCreator creator) {
    (void)KVCacheStoreBackendFactory::Instance().RegisterBackend(
        type_name, std::move(creator));
  }
};
}  // namespace internal

}  // namespace kv_cache
}  // namespace tpu_raiden

// Macro for static auto-registration of custom backend creators.
// Uses an anonymous namespace and __COUNTER__ to guarantee identifier
// uniqueness.
#define REGISTER_KV_CACHE_STORE_BACKEND_CONCAT_INNER(a, b) a##b
#define REGISTER_KV_CACHE_STORE_BACKEND_CONCAT(a, b) \
  REGISTER_KV_CACHE_STORE_BACKEND_CONCAT_INNER(a, b)

#define REGISTER_KV_CACHE_STORE_BACKEND(type_name, creator_func)         \
  namespace {                                                            \
  static ::tpu_raiden::kv_cache::internal::BackendRegistrar              \
      REGISTER_KV_CACHE_STORE_BACKEND_CONCAT(_raiden_backend_reg_,       \
                                             __COUNTER__)(type_name,     \
                                                          creator_func); \
  }  // namespace

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_BACKEND_FACTORY_H_
