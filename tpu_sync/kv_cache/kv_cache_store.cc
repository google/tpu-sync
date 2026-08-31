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

#include "tpu_sync/kv_cache/kv_cache_store.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <tuple>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/buffer.h"
#include "tpu_sync/core/controller/raiden_controller.h"
#include "tpu_sync/core/status_macros.h"
#include "tpu_sync/kv_cache/completion_executor.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_client.h"
#include "tpu_sync/kv_cache/host_offload_backend.h"
#include "tpu_sync/kv_cache/completion_executor.h"
#include "tpu_sync/kv_cache/kv_cache_metadata.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend.h"
#include "tpu_sync/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_sync/kv_cache/reshard/reshard_service.h"
#include "tpu_sync/kv_cache/store_monitor.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

namespace {

// Registration TTL, in heartbeat periods: the registration survives two
// missed heartbeats and expires on the third. A lapse is cheap to recover
// from anyway -- the next heartbeat's NotFound answer makes the monitor
// re-register.
constexpr int kRegistrationTtlInHeartbeats = 3;

// Evict-sweep defaults. The watermarks can sit this low because detection is
// not periodic: the allocation path nudges the sweep the moment free blocks
// dip below the low watermark, so the cushion only has to cover the demotion
// ramp-up, not a polling interval.
constexpr double kDefaultEvictLowWatermark = 0.03;
constexpr double kDefaultEvictHighWatermark = 0.05;
// Cap on one demotion batch. The batch is sized by what the high watermark
// still needs; this cap only keeps a single sweep step short, since Stop()
// and an interleaved heartbeat wait for at most one step.
constexpr int kMaxEvictBatchBlocks = 128;
// Placement targets requested per pressure episode.
constexpr int32_t kMaxPlacementTargets = 4;
// Cap on one inventory-republish batch: bounds both the registry's lock hold
// per Register RPC and how long the batch's blocks stay pinned. Larger than
// kMaxEvictBatchBlocks because a step is one metadata RPC, not a transfer.
constexpr size_t kMaxRepublishBatchBlocks = 512;

// Bind-and-advertise address for this store's RaidenController.
//
// Empty ip preserves the legacy behaviour (RaidenController binds the wildcard
// interface). A port of 0 lets gRPC choose; either way RaidenController splices
// the actually-bound port back into the address it advertises.
std::string ComposeControllerAddress(absl::string_view store_server_ip,
                                     int raiden_controller_port) {
  if (store_server_ip.empty()) {
    return raiden_controller_port > 0
               ? absl::StrCat("[::]:", raiden_controller_port)
               : "";
  }
  return absl::StrCat(store_server_ip, ":", raiden_controller_port);
}

// Every store has a
// controller, and store_server_ip is mandatory and never a wildcard -- it is
// the host of the controller address published to the global registry, which
// is what a peer dials to acquire a read lease, so it must be routable by
// peers. Hostnames are allowed (same-host tests use "localhost"); empty and
// wildcard are not.
absl::Status ValidateConstructionRules(absl::string_view store_server_ip,
                                       int num_shards) {
  if (store_server_ip.empty()) {
    return absl::InvalidArgumentError(
        "store_server_ip is required: it is the host peers use to reach this "
        "store's services. Same-host use: \"127.0.0.1\".");
  }
  if (store_server_ip == "[::]" || store_server_ip == "::" ||
      store_server_ip == "0.0.0.0" || store_server_ip == "0:0:0:0:0:0:0:0") {
    return absl::InvalidArgumentError(absl::StrCat(
        "store_server_ip may not be a wildcard (got \"", store_server_ip,
        "\"): a wildcard binds but cannot be published or dialled."));
  }
  if (num_shards < 1) {
    return absl::InvalidArgumentError(
        "num_shards must be >= 1: every KVCacheStore has a RaidenController; "
        "the controller-less configuration is not supported.");
  }
  return absl::OkStatus();
}

// ValidateBackends checks that backends_ has at least a tier-0 backend and it
// is not null. Catching this at construction guarantees that a store configured
// with a registry is always registered, by preventing a backend-less store from
// being built: previously such a store constructed, warned, and quietly
// declined to serve or publish.
//
// Validated the way every other construction rule fails -- before any
// resource exists, Status from Create() and abort from the raw constructors.
absl::Status ValidateBackends(
    const std::vector<std::shared_ptr<KVCacheStoreBackend>>& backends) {
  if (backends.empty()) {
    return absl::InvalidArgumentError(
        "KVCacheStore requires at least one backend (tier 0, local host "
        "DRAM).");
  }
  if (backends[0] == nullptr) {
    return absl::InvalidArgumentError(
        "KVCacheStore's tier-0 backend must not be null.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<controller::RaidenController>>
MakeRaidenController(const RaidenId& raiden_id, size_t capacity, int num_shards,
                     int64_t shard_size_bytes,
                     absl::string_view store_server_ip,
                     int raiden_controller_port,
                     int expected_worker_count = 0) {
  if (num_shards <= 0) return nullptr;
  ::tpu_sync::rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(raiden_id.job_name);
  unit_proto.set_job_replica_id(raiden_id.job_replica_id);
  unit_proto.set_data_name(raiden_id.data_name);
  unit_proto.set_data_replica_idx(raiden_id.data_replica_idx);
  return ::tpu_raiden::controller::RaidenController::Create(
      unit_proto, capacity, num_shards, shard_size_bytes,
      ComposeControllerAddress(store_server_ip, raiden_controller_port),
      /*preprovision_worker_buffers=*/false, expected_worker_count);
}

}  // namespace

absl::StatusOr<std::unique_ptr<KVCacheStore>> KVCacheStore::Create(
    absl::Span<const BackendConfig> backend_configs, size_t capacity,
    absl::string_view global_registry_address, RaidenId raiden_id,
    int num_shards, int64_t shard_size_bytes, absl::string_view store_server_ip,
    int raiden_controller_port, std::optional<KVCacheMetadata> metadata,
    int expected_worker_count) {
  if (backend_configs.empty()) {
    return absl::InvalidArgumentError("backend_configs must not be empty");
  }
  // Before any resource is created (violation must not leak).
  RETURN_IF_ERROR(ValidateConstructionRules(store_server_ip, num_shards));

  BackendConfig effective_config0 = backend_configs[0];
  if (!effective_config0.global_registry_address.empty() &&
      global_registry_address.empty()) {
    global_registry_address = effective_config0.global_registry_address;
  }
  if (effective_config0.capacity == 0 && capacity > 0) {
    effective_config0.capacity = capacity;
  }
  if (!effective_config0.metadata.has_value() && metadata.has_value()) {
    effective_config0.metadata = metadata;
  }
  if (effective_config0.global_registry_address.empty() &&
      !global_registry_address.empty()) {
    effective_config0.global_registry_address =
        std::string(global_registry_address);
  }
  if (effective_config0.raiden_id.empty() && !raiden_id.empty()) {
    effective_config0.raiden_id = raiden_id;
  }

  RaidenId effective_raiden_id = effective_config0.raiden_id;
  std::unique_ptr<::tpu_raiden::controller::RaidenController> raiden_controller;
  if (num_shards > 0) {
    ASSIGN_OR_RETURN(
        raiden_controller,
        MakeRaidenController(effective_raiden_id, effective_config0.capacity,
                             num_shards, shard_size_bytes, store_server_ip,
                             raiden_controller_port, expected_worker_count));
  }

  std::vector<std::shared_ptr<KVCacheStoreBackend>> backends;
  backends.reserve(backend_configs.size());

  for (size_t i = 0; i < backend_configs.size(); ++i) {
    const auto& config = backend_configs[i];
    BackendConfig effective_config = (i == 0) ? effective_config0 : config;

    // Apply global registry address and raiden_id across all backend
    // configurations
    if (effective_config.global_registry_address.empty() &&
        !global_registry_address.empty()) {
      effective_config.global_registry_address =
          std::string(global_registry_address);
    }
    if (effective_config.raiden_id.empty() && !raiden_id.empty()) {
      effective_config.raiden_id = raiden_id;
    }

    ASSIGN_OR_RETURN(auto backend,
                     KVCacheStoreBackendFactory::Instance().CreateBackend(
                         effective_config, raiden_controller.get()));
    // A custom registration may return OK with a null pointer; that would sail
    // past ValidateBackends for any tier but 0, and crash later.
    if (backend == nullptr) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Backend factory returned a null backend for tier ", i, "."));
    }
    backends.push_back(std::move(backend));
  }

  RETURN_IF_ERROR(ValidateBackends(backends));

  // The private constructor used here deliberately does no controller
  // wiring (the public ones do, and FATAL on failure) -- that would defeat
  // Create()'s whole purpose of returning a recoverable Status. Create()
  // does the wiring below instead.
  auto store = absl::WrapUnique(new KVCacheStore(
      std::move(backends), effective_raiden_id, std::move(raiden_controller),
      store_server_ip, global_registry_address));

  // Before SetRaidenController: the registration published there carries
  // these attributes.
  store->kv_pool_group_ = effective_config0.kv_pool_group;
  store->evict_tier_ = effective_config0.evict_tier;
  StoreMonitorConfig& monitor_config = store->monitor_config_;
  monitor_config = effective_config0.monitor_config;
  if (monitor_config.heartbeat_period <= absl::ZeroDuration()) {
    monitor_config.heartbeat_period = StoreMonitor::kDefaultHeartbeatPeriod;
  }
  if (monitor_config.evict_low_watermark <= 0.0) {
    monitor_config.evict_low_watermark = kDefaultEvictLowWatermark;
  }
  if (monitor_config.evict_high_watermark <= 0.0) {
    monitor_config.evict_high_watermark = kDefaultEvictHighWatermark;
  }
  if (monitor_config.enable_evict_sweep) {
    if (!monitor_config.enable) {
      return absl::FailedPreconditionError(
          "StoreMonitorConfig.enable_evict_sweep requires the monitor "
          "(StoreMonitorConfig.enable): the sweep runs on the store "
          "monitor's schedule.");
    }
    if (monitor_config.evict_high_watermark <
            monitor_config.evict_low_watermark ||
        monitor_config.evict_high_watermark > 1.0) {
      return absl::FailedPreconditionError(absl::StrCat(
          "evict watermarks must satisfy low <= high <= 1, got low=",
          monitor_config.evict_low_watermark,
          " high=", monitor_config.evict_high_watermark));
    }
  }

  if (store->raiden_controller_ != nullptr) {
    RETURN_IF_ERROR(
        store->SetRaidenController(store->raiden_controller_.get()));
    store->RegisterReadRemoteHooks();
    store->poller_thread_ =
        std::make_unique<std::thread>(&KVCacheStore::PollerLoop, store.get());
  }

  // The constructor above returned, so the expected workers have all
  // registered -- with their KV block geometry -- and the deployment's
  // KVTransferSpec can be derived and published. The tier-0 backend does all
  // of that: registry interaction is its job, and the registry side is
  // first-wins idempotent, so every serving host running this same code is
  // safe.
  const bool tier0_has_registry =
      !backend_configs[0].global_registry_address.empty() ||
      !global_registry_address.empty();
  if (expected_worker_count > 0 && tier0_has_registry) {
    auto* host_backend =
        dynamic_cast<HostOffloadBackend*>(store->backend().get());
    if (host_backend == nullptr) {
      return absl::FailedPreconditionError(
          "registering the KVTransferSpec requires a HostOffloadBackend at "
          "tier 0");
    }
    RETURN_IF_ERROR(host_backend->RegisterKVTransferSpecFromWorkers());
  }

  if (monitor_config.enable) {
    // The flag promises heartbeats; a store that never registered has
    // nothing to heartbeat, so this configuration is a contradiction, not a
    // degraded mode.
    if (!store->registered_in_global_registry_) {
      return absl::FailedPreconditionError(
          "StoreMonitorConfig.enable requires a global_registry_address: the "
          "monitor's heartbeats refresh this store's registration there.");
    }
    StoreMonitor::Options monitor_options;
    monitor_options.heartbeat_period = monitor_config.heartbeat_period;
    if (monitor_config.evict_sweep_period > absl::ZeroDuration()) {
      monitor_options.sweep_period = monitor_config.evict_sweep_period;
    }
    StoreMonitor::SweepFn sweep_fn;
    if (monitor_config.enable_evict_sweep) {
      sweep_fn = [s = store.get()] { return s->SweepOnce(); };
    }
    store->store_monitor_ = std::make_unique<StoreMonitor>(
        monitor_options, store->registry_client_, store->raiden_id_,
        /*status_fn=*/
        [s = store.get()] {
          global_registry::StoreStatus status;
          status.set_free_blocks(s->backend()->GetAvailableSpace());
          return status;
        },
        // Re-publishes the registration EnsureStoreServerAndRegister first
        // made; runs only on the monitor thread, so the TTL applies
        // unconditionally.
        /*reregister_fn=*/
        [s = store.get()] {
          return s->registry_client_->RegisterStore(
              s->raiden_id_, s->store_server_address_,
              s->raiden_controller_ != nullptr
                  ? s->raiden_controller_->controller_address()
                  : "",
              kRegistrationTtlInHeartbeats *
                  s->monitor_config_.heartbeat_period,
              !s->kv_pool_group_.empty() ? s->kv_pool_group_
                                         : s->raiden_id_.job_name,
              s->evict_tier_);
        },
        std::move(sweep_fn),
        /*republish_fn=*/[s = store.get()] { return s->RepublishOnce(); });
    store->store_monitor_->Start();
  }

  return store;
}

absl::StatusOr<std::unique_ptr<KVCacheStore>> KVCacheStore::Create(
    const BackendConfig& config, size_t capacity,
    absl::string_view global_registry_address, RaidenId raiden_id,
    int num_shards, int64_t shard_size_bytes, absl::string_view store_server_ip,
    int raiden_controller_port, std::optional<KVCacheMetadata> metadata,
    int expected_worker_count) {
  return KVCacheStore::Create(
      absl::MakeConstSpan(&config, 1), capacity, global_registry_address,
      raiden_id, num_shards, shard_size_bytes, store_server_ip,
      raiden_controller_port, std::move(metadata), expected_worker_count);
}

absl::StatusOr<std::unique_ptr<KVCacheStore>> KVCacheStore::CreateReshardStore(
    RaidenId raiden_id, absl::string_view store_server_ip,
    int raiden_controller_port, int reshard_service_port) {
  if (store_server_ip.empty() || store_server_ip == "0.0.0.0" ||
      store_server_ip == "::") {
    return absl::InvalidArgumentError(
        "store_server_ip must be a concrete IP (not empty or wildcard)");
  }
  // Construct a minimal thin store with dummy backend (capacity=1) so the
  // store owns the RaidenController and can host ReshardService.
  BackendConfig cfg;
  // "local_lru" is not registered in this repository's
  // KVCacheStoreBackendFactory; HostOffloadBackend is the only registered
  // backend and is cheap at capacity=1 (no registry client, no server start,
  // no preallocation) — the reshard store never serves blocks through it.
  cfg.type = "HostOffloadBackend";
  cfg.capacity = 1;
  cfg.raiden_id = raiden_id;

  // num_shards=1 gives the controller an initial partition; workers dynamically
  // register their actual shard assignments via WorkerService.
  ASSIGN_OR_RETURN(auto store, KVCacheStore::Create(
                                   cfg, /*capacity=*/1,
                                   /*global_registry_address=*/"", raiden_id,
                                   /*num_shards=*/1, /*shard_size_bytes=*/0,
                                   store_server_ip, raiden_controller_port,
                                   /*metadata=*/std::nullopt,
                                   /*expected_worker_count=*/0));

  // Initialize ReshardService with WorkerDelivery::Mode::kController.
  reshard::ReshardService::Options reshard_opts;
  reshard_opts.port = reshard_service_port;
  reshard_opts.delivery.mode = reshard::WorkerDelivery::Mode::kController;
  reshard_opts.delivery.controller = store->raiden_controller_.get();

  store->reshard_service_ =
      std::make_unique<reshard::ReshardService>(reshard_opts);
  RETURN_IF_ERROR(store->reshard_service_->StartServer());
  return store;
}

KVCacheStore::KVCacheStore(ReshardSidecarTag)
    : lifetime_(std::make_shared<Lifetime>()) {
  lifetime_->store = this;
}

absl::StatusOr<std::unique_ptr<KVCacheStore>>
KVCacheStore::CreateReshardSidecar(int reshard_port,
                                   double request_registry_ttl_s) {
  // Reshard-only sidecar mode. Deliberately bypasses
  // ValidateConstructionRules and ValidateBackends — this store serves no offload tier, publishes no
  // registry record, and builds no controller submodule; its only surface
  // is the reshard plane's framed listener. The full-store validation
  // paths above stay byte-identical.
  if (request_registry_ttl_s <= 0) {
    return absl::InvalidArgumentError(
        "request_registry_ttl_s must be positive");
  }
  auto store = absl::WrapUnique(new KVCacheStore(ReshardSidecarTag{}));
  reshard::ReshardService::Options options;
  options.port = reshard_port;
  options.request_registry_ttl_s = request_registry_ttl_s;
  store->reshard_service_ = std::make_unique<reshard::ReshardService>(options);
  RETURN_IF_ERROR(store->reshard_service_->StartServer());
  return store;
}

KVCacheStore::KVCacheStore(
    std::vector<std::shared_ptr<KVCacheStoreBackend>> backends,
    RaidenId raiden_id,
    std::unique_ptr<controller::RaidenController> raiden_controller,
    absl::string_view store_server_ip,
    absl::string_view global_registry_address)
    : backends_(std::move(backends)),
      raiden_id_(std::move(raiden_id)),
      raiden_controller_(std::move(raiden_controller)),
      store_server_ip_(store_server_ip),
      lifetime_(std::make_shared<Lifetime>()) {
  lifetime_->store = this;
  if (absl::Status v = ValidateConstructionRules(
          store_server_ip, raiden_controller_ != nullptr ? 1 : 0);
      !v.ok()) {
    LOG(FATAL) << "KVCacheStore construction validation failed: " << v.message()
               << " Use KVCacheStore::Create() for a recoverable error.";
  }
  // Both public backend-taking constructors delegate here, so this is the one
  // place that has to check. Create() checks separately, before constructing,
  // so it can return the error instead of aborting.
  if (absl::Status v = ValidateBackends(backends_); !v.ok()) {
    LOG(FATAL) << "KVCacheStore construction rules violated: " << v.message()
               << " Use KVCacheStore::Create() for a recoverable error.";
  }
  // Created before SetRaidenController: publishing happens there, and needs it.
  if (!global_registry_address.empty()) {
    auto channel = grpc::CreateChannel(std::string(global_registry_address),
                                       grpc::InsecureChannelCredentials());
    registry_client_ =
        std::make_shared<global_registry::GlobalRegistryClient>(channel);
  }
}

KVCacheStore::KVCacheStore(
    std::vector<std::shared_ptr<KVCacheStoreBackend>> backends,
    RaidenId raiden_id, int num_shards, int64_t shard_size_bytes,
    absl::string_view store_server_ip, int raiden_controller_port,
    absl::string_view global_registry_address, int expected_worker_count)
    : KVCacheStore(
          backends, raiden_id,
          MakeRaidenController(raiden_id,
                               (!backends.empty() && backends[0] != nullptr)
                                   ? backends[0]->GetCapacity()
                                   : 0,
                               num_shards, shard_size_bytes, store_server_ip,
                               raiden_controller_port, expected_worker_count)
              .value_or(nullptr),
          store_server_ip, global_registry_address) {
  if (raiden_controller_) {
    if (absl::Status s = SetRaidenController(raiden_controller_.get());
        !s.ok()) {
      LOG(FATAL) << "KVCacheStore failed to start/publish its store server: "
                 << s.message()
                 << " Use KVCacheStore::Create() for a recoverable error.";
    }
    RegisterReadRemoteHooks();
    poller_thread_ =
        std::make_unique<std::thread>(&KVCacheStore::PollerLoop, this);
  }
}

KVCacheStore::KVCacheStore(std::shared_ptr<KVCacheStoreBackend> backend,
                           RaidenId raiden_id, int num_shards,
                           int64_t shard_size_bytes,
                           absl::string_view store_server_ip,
                           int raiden_controller_port,
                           absl::string_view global_registry_address,
                           int expected_worker_count)
    : KVCacheStore(
          std::vector<std::shared_ptr<KVCacheStoreBackend>>{std::move(backend)},
          std::move(raiden_id), num_shards, shard_size_bytes, store_server_ip,
          raiden_controller_port, global_registry_address,
          expected_worker_count) {}

KVCacheStore::KVCacheStore(
    size_t capacity, absl::string_view global_registry_address,
    RaidenId raiden_id, int num_shards, int64_t shard_size_bytes,
    absl::string_view store_server_ip, int raiden_controller_port,
    std::optional<KVCacheMetadata> metadata, int expected_worker_count)
    : raiden_id_(raiden_id),
      store_server_ip_(store_server_ip),
      lifetime_(std::make_shared<Lifetime>()) {
  lifetime_->store = this;
  if (absl::Status v = ValidateConstructionRules(store_server_ip, num_shards);
      !v.ok()) {
    LOG(FATAL) << "KVCacheStore construction validation failed: " << v.message()
               << " Use KVCacheStore::Create() for a recoverable error.";
  }
  if (!global_registry_address.empty()) {
    auto channel = grpc::CreateChannel(std::string(global_registry_address),
                                       grpc::InsecureChannelCredentials());
    registry_client_ =
        std::make_shared<global_registry::GlobalRegistryClient>(channel);
  }
  if (num_shards > 0) {
    auto controller = MakeRaidenController(
        raiden_id_, capacity, num_shards, shard_size_bytes, store_server_ip,
        raiden_controller_port, expected_worker_count);
    if (!controller.ok()) {
      LOG(FATAL) << "KVCacheStore failed to create RaidenController: "
                 << controller.status().message()
                 << " Use KVCacheStore::Create() for a recoverable error.";
    }
    raiden_controller_ = std::move(*controller);
  }

  // The backend must get its own registry client (built by the factory from
  // the same address): peer resolution runs through the BACKEND's registry
  // client, not this store's, and a backend without one registers itself
  // perfectly but can reach nobody.
  BackendConfig backend_config;
  backend_config.type = "HostOffloadBackend";
  backend_config.capacity = capacity;
  backend_config.global_registry_address = std::string(global_registry_address);
  backend_config.raiden_id = raiden_id_;
  backend_config.metadata = std::move(metadata);
  auto backend = KVCacheStoreBackendFactory::Instance().CreateBackend(
      backend_config, raiden_controller_.get());
  if (!backend.ok()) {
    LOG(FATAL) << "KVCacheStore failed to create its tier-0 backend: "
               << backend.status().message()
               << " Use KVCacheStore::Create() for a recoverable error.";
  }
  backends_ = {std::move(*backend)};

  if (raiden_controller_) {
    if (absl::Status s = SetRaidenController(raiden_controller_.get());
        !s.ok()) {
      LOG(FATAL) << "KVCacheStore failed to start/publish its store server: "
                 << s.message()
                 << " Use KVCacheStore::Create() for a recoverable error.";
    }
    RegisterReadRemoteHooks();
    poller_thread_ =
        std::make_unique<std::thread>(&KVCacheStore::PollerLoop, this);
  }
}

KVCacheStore::KVCacheStore(
    size_t capacity,
    std::unique_ptr<::tpu_raiden::controller::RaidenController>
        raiden_controller,
    absl::string_view global_registry_address, RaidenId raiden_id,
    std::optional<KVCacheMetadata> metadata, absl::string_view store_server_ip)
    : raiden_id_(raiden_id),
      raiden_controller_(std::move(raiden_controller)),
      store_server_ip_(store_server_ip),
      lifetime_(std::make_shared<Lifetime>()) {
  lifetime_->store = this;
  if (absl::Status v =
          ValidateConstructionRules(store_server_ip, /*num_shards=*/1);
      !v.ok()) {
    LOG(FATAL) << "KVCacheStore construction validation failed: "
               << v.message();
  }
  if (raiden_controller_ == nullptr) {
    LOG(FATAL) << "KVCacheStore requires a RaidenController; the "
                  "controller-less configuration is not supported. "
                  "(store_server_ip/controller-host consistency is the "
                  "caller's promise on this constructor.)";
  }
  if (!global_registry_address.empty()) {
    auto channel = grpc::CreateChannel(std::string(global_registry_address),
                                       grpc::InsecureChannelCredentials());
    registry_client_ =
        std::make_shared<global_registry::GlobalRegistryClient>(channel);
  }

  // See the capacity constructor: the factory builds the backend its own
  // registry client from the same address.
  BackendConfig backend_config;
  backend_config.type = "HostOffloadBackend";
  backend_config.capacity = capacity;
  backend_config.global_registry_address = std::string(global_registry_address);
  backend_config.raiden_id = raiden_id_;
  backend_config.metadata = std::move(metadata);
  auto backend = KVCacheStoreBackendFactory::Instance().CreateBackend(
      backend_config, raiden_controller_.get());
  if (!backend.ok()) {
    LOG(FATAL) << "KVCacheStore failed to create its tier-0 backend: "
               << backend.status().message()
               << " Use KVCacheStore::Create() for a recoverable error.";
  }
  backends_ = {std::move(*backend)};

  if (raiden_controller_) {
    if (absl::Status s = SetRaidenController(raiden_controller_.get());
        !s.ok()) {
      LOG(FATAL) << "KVCacheStore failed to start/publish its store server: "
                 << s.message()
                 << " Use KVCacheStore::Create() for a recoverable error.";
    }
    RegisterReadRemoteHooks();
    poller_thread_ =
        std::make_unique<std::thread>(&KVCacheStore::PollerLoop, this);
  }
}

absl::Status KVCacheStore::SetRaidenController(
    tpu_raiden::controller::RaidenController* controller) {
  // Runs first so that store_server_ip_ decides the bind address: StartServer
  // never rebinds a running server, so whoever starts it first wins.
  return EnsureStoreServerAndRegister(controller);
}

absl::Status KVCacheStore::EnsureStoreServerAndRegister(
    tpu_raiden::controller::RaidenController* controller) {
  if (store_server_ != nullptr || controller == nullptr) {
    return absl::OkStatus();  // Already done; may be called repeatedly.
  }

  // The global registry decides whether the P2P plane exists at all. No
  // registry client, no server -- not adopted from a backend, not owned by
  // this store. (store_server_ip_ is never empty here: validation
  // makes it mandatory.)
  if (registry_client_ == nullptr) {
    return absl::OkStatus();
  }

  // Reuse a backend's server if one exists, so a node never serves peers from
  // two ports. Otherwise this store owns one -- every store must be reachable,
  // not just those configured with a backend that happens to host a server.
  KVCacheStoreBackend* serving_backend = nullptr;
  for (auto& backend : backends_) {
    if (backend != nullptr && backend->store_server() != nullptr) {
      store_server_ = backend->store_server();
      serving_backend = backend.get();
      break;
    }
  }
  if (store_server_ == nullptr) {
    // Unreachable: ValidateBackends rejects a store with no tier-0 backend at
    // construction. Kept as a last line of defence, and as an ERROR rather
    // than the warning-and-carry-on it used to be -- returning OK here left a
    // store that has a registry configured but never publishes itself, which
    // reads to every peer as "that node does not exist".
    if (backends_.empty() || backends_[0] == nullptr) {
      return absl::FailedPreconditionError(
          "KVCacheStore has no tier-0 backend; cannot serve or publish.");
    }
    owned_store_server_ = KVCacheStoreServer::Create();
    store_server_ = owned_store_server_.get();
    serving_backend = backends_[0].get();
  }

  // store_server_ip_ is mandatory; the port is always gRPC's choice.
  const std::string bind_address = absl::StrCat(store_server_ip_, ":0");

  absl::Status status =
      store_server_->StartServer(serving_backend, controller, bind_address);
  if (!status.ok()) {
    store_server_ = nullptr;
    owned_store_server_.reset();
    return absl::Status(status.code(),
                        absl::StrCat("Failed to start KVCacheStoreServer on ",
                                     bind_address, ": ", status.message()));
  }

  // The server is the single source of truth for its own address -- it
  // remembers the host it was actually started with. This is empty only if
  // something adopted (bound before this call) it on a wildcard/empty host --
  // unreachable for any in-tree caller, but checked rather than
  // publishing a blank address.
  store_server_address_ = store_server_->GetServerAddress();
  if (store_server_address_.empty()) {
    return absl::FailedPreconditionError(
        "KVCacheStoreServer has no publishable address (bound on a wildcard "
        "or empty host); cannot register with the global registry.");
  }

  // A TTL is only safe with a heartbeat refreshing it: an expiring
  // registration on an unmonitored store would just vanish. The kv_pool_group
  // fallback is the same as the KVTransferSpec registration key's
  // (BackendConfig.kv_pool_group).
  absl::Status register_status = registry_client_->RegisterStore(
      raiden_id_, store_server_address_,
      raiden_controller_ != nullptr ? raiden_controller_->controller_address()
                                    : "",
      monitor_config_.enable
          ? kRegistrationTtlInHeartbeats * monitor_config_.heartbeat_period
          : absl::ZeroDuration(),
      !kv_pool_group_.empty() ? kv_pool_group_ : raiden_id_.job_name,
      evict_tier_);
  if (!register_status.ok()) {
    return absl::Status(
        register_status.code(),
        absl::StrCat("Failed to publish store address ", store_server_address_,
                     " to the global registry: ", register_status.message()));
  }
  registered_in_global_registry_ = true;
  LOG(INFO) << "KVCacheStore published at " << store_server_address_;
  return absl::OkStatus();
}

void KVCacheStore::ShutdownBackendStoreServers(
    KVCacheStoreServer* already_shut) {
  for (auto& backend : backends_) {
    if (backend == nullptr) {
      continue;
    }
    KVCacheStoreServer* server = backend->store_server();
    // Shutdown() is idempotent, but skipping the adopted server says why
    // rather than leaving the reader to check.
    if (server != nullptr && server != already_shut) {
      server->Shutdown();
    }
  }
}

KVCacheStore::~KVCacheStore() {
  {
    absl::MutexLock lock(lifetime_->mu);
    lifetime_->store = nullptr;
  }
  // First, before the registry client or the backends its callbacks read go
  // away -- and before UnregisterStore below, which a late heartbeat's
  // re-register would undo.
  if (store_monitor_ != nullptr) {
    store_monitor_->Stop();
  }

  // Stop being discoverable, then stop serving, and do both BEFORE anything the
  // service dereferences is destroyed. `backends_` is declared before
  // `raiden_controller_`, so member destruction would otherwise free the
  // controller while the server is still accepting RPCs that use it.
  //
  // Withdrawing the store record is also what retires this store's block
  // entries: the server purges everything this raiden id owns. Nothing waits
  // for the publishes still in flight.
  if (registered_in_global_registry_ && registry_client_ != nullptr) {
    absl::Status status = registry_client_->UnregisterStore(raiden_id_);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to unpublish store address "
                   << store_server_address_ << ": " << status.message()
                   << ". Peers may dial a dead address until it is replaced.";
    }
    registered_in_global_registry_ = false;
  }
  if (store_server_ != nullptr) {
    // Shutdown() drains in-flight handlers, so no RPC is mid-flight past this
    // point.
    //
    // We shut the server down even when a backend owns it, because the service
    // holds OUR RaidenController in pointers it cannot re-seat -- once we go,
    // it can only answer RPCs with a dangling controller. The consequence is
    // that sharing one backend between two KVCacheStores is not supported: the
    // first store to be destroyed takes the shared server with it.
    KVCacheStoreServer* already_shut = store_server_;
    store_server_->Shutdown();
    store_server_ = nullptr;

    // Fall through to the sweep below with this one excluded.
    ShutdownBackendStoreServers(already_shut);
  } else {
    // store_server_ is null whenever this store has no registry (see
    // EnsureStoreServerAndRegister), but a backend's server can still be
    // running: StartServer is public, and a caller may start one AFTER
    // construction, which is too late for the adoption above to see it. That
    // server holds OUR controller in a pointer it cannot re-seat, and
    // raiden_controller_ is about to be destroyed, so it has to be stopped
    // here -- in the destructor BODY, while the controller is still alive.
    // Relying on ~HostOffloadBackend is not enough: backends_ holds
    // shared_ptrs, so a caller keeping its own reference outlives this store.
    ShutdownBackendStoreServers(/*already_shut=*/nullptr);
  }

  if (poller_thread_) {
    stop_poller_.store(true);
    if (poller_thread_->joinable()) {
      poller_thread_->join();
    }
  }

  // Abandon any remote write still outstanding, now that no poller can race
  // us. This does NOT wait for them: a remote write goes terminal only when
  // the destination answers or the HOLD (~30s) expires, so waiting would make
  // destroying a store block for half a minute behind a slow or dead peer.
  //
  // Releasing the internal pin is the part that has to happen. `backends_`
  // holds shared_ptrs, so a backend can outlive the store that pinned into it;
  // a pin left behind there is a host block nothing can ever reclaim.
  {
    std::vector<RemoteWriteState> abandoned;
    {
      absl::MutexLock lock(mutex_);
      abandoned.reserve(active_remote_writes_.size());
      for (auto& [key, state] : active_remote_writes_) {
        abandoned.push_back(std::move(state));
      }
      active_remote_writes_.clear();
    }
    // Cancelled BEFORE the pins go back, and for the destination's sake
    // rather than this store's. Nothing waits for the calls to end --
    // TryCancel does not block, and each reactor settles itself -- but a
    // destination still holding an open call keeps that operation alive and
    // its landing blocks reserved for the rest of the hold window, waiting on
    // an answer from a source that no longer exists.
    for (const auto& state : abandoned) {
      if (state.cancel != nullptr) {
        state.cancel->TryCancel();
      }
    }
    for (auto& state : abandoned) {
      LOG(WARNING) << "Store destroyed with remote write " << state.operation_id
                   << " still outstanding; releasing its source pin without "
                      "waiting for the destination's verdict.";
      OnWriteRemoteVerdict(std::move(state), /*succeeded=*/false, {});
    }
  }

  std::vector<tsl::Future<>> futures_to_await;
  {
    absl::MutexLock lock(mutex_);
    for (auto& state : active_saves_) {
      futures_to_await.push_back(state.future);
    }
    for (auto& state : active_loads_) {
      futures_to_await.push_back(state.future);
    }
  }
  for (auto& fut : futures_to_await) {
    (void)fut.Await();
  }
}

absl::StatusOr<BlockSliceList> KVCacheStore::Lookup(
    const std::vector<std::string>& block_hashes, bool enable_global,
    bool pin_found) {
  // The application-facing overload PINS what it finds by default. A caller
  // asks what is resident in order to use it, and between the answer and the
  // use the entry would otherwise be evictable -- so the pin comes with the
  // answer, and the operation the caller goes on to perform (load, save)
  // consumes it. pin_found = false is for the caller that only wants to
  // OBSERVE residency: no pin is taken and the LRU order is untouched.
  //
  // Only this overload defaults to pinning. The LookupOptions overload leaves
  // pin_found at its struct default of false, which is what every internal
  // caller goes through.
  return Lookup(block_hashes, LookupOptions{.enable_global = enable_global,
                                            .pin_found = pin_found});
}

absl::StatusOr<BlockSliceList> KVCacheStore::Lookup(
    const std::vector<std::string>& block_hashes,
    const LookupOptions& options) {
  BlockSliceList accumulated_results;
  accumulated_results.reserve(block_hashes.size());

  size_t start_idx = 0;
  for (size_t tier_idx = 0; tier_idx < backends_.size(); ++tier_idx) {
    const auto& backend = backends_[tier_idx];
    if (start_idx >= block_hashes.size()) break;
    if (!backend) continue;

    auto res_or = backend->Lookup(
        absl::MakeSpan(block_hashes).subspan(start_idx), options);
    if (!res_or.ok()) {
      if (!accumulated_results.empty() && absl::IsNotFound(res_or.status())) {
        break;
      }
      if (options.pin_found && !accumulated_results.empty()) {
        std::vector<std::string> matched_hashes;
        matched_hashes.reserve(accumulated_results.size());
        for (const auto& pair : accumulated_results) {
          matched_hashes.push_back(pair.first);
        }
        Release(matched_hashes);
      }
      return res_or.status();
    }

    const auto& res = res_or.value();
    for (const auto& pair : res) {
      accumulated_results.push_back(pair);
      ++start_idx;
    }
  }

  size_t cap = capacity();
  if (cap > 0 && accumulated_results.size() > cap) {
    if (options.pin_found) {
      std::vector<std::string> excess_hashes;
      excess_hashes.reserve(accumulated_results.size() - cap);
      for (size_t i = cap; i < accumulated_results.size(); ++i) {
        excess_hashes.push_back(accumulated_results[i].first);
      }
      Release(excess_hashes);
    }
    accumulated_results.resize(cap);
  }

  return accumulated_results;
}

absl::Status KVCacheStore::Insert(const std::vector<std::string>& block_hashes,
                                  const std::vector<RaidenBlockId>& slices,
                                  bool on_host) {
  if (backends_.empty()) {
    return absl::FailedPreconditionError("KVCacheStore has no backend");
  }

  // The LRU cache holds LOCAL blocks only; refuse the whole batch before any
  // backend is touched, so a bad batch cannot pin or insert anything.
  for (size_t i = 0; i < slices.size(); ++i) {
    if (slices[i].status == BlockStatus::REMOTE) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Insert only takes LOCAL entries; slice ", i, " (hash ",
          i < block_hashes.size() ? absl::BytesToHexString(block_hashes[i])
                                  : "<none>",
          ") has status REMOTE"));
    }
  }

  std::vector<size_t> locked_backends;
  for (size_t i = 0; i < backends_.size(); ++i) {
    if (!backends_[i]) continue;
    if (!backends_[i]->InsertAndLock(block_hashes, slices, on_host)) {
      for (size_t lb : locked_backends) {
        backends_[lb]->ReleaseAndDelete(block_hashes);
      }
      return absl::ResourceExhaustedError(
          "Insert refused: not enough unpinned space, or a hash could not be "
          "pinned");
    }
    locked_backends.push_back(i);
  }
  return absl::OkStatus();
}

void KVCacheStore::Release(const std::vector<std::string>& block_hashes) {
  for (auto& backend : backends_) {
    if (backend) backend->Release(block_hashes);
  }
}

int KVCacheStore::GetPinCount(const std::string& hash) const {
  for (const auto& backend : backends_) {
    if (backend) {
      int count = backend->GetPinCount(hash);
      if (count > 0) return count;
    }
  }
  return 0;
}

size_t KVCacheStore::capacity() const {
  return (!backends_.empty() && backends_[0]) ? backends_[0]->GetCapacity() : 0;
}

std::string KVCacheStore::raiden_controller_address() const {
  if (raiden_controller_) {
    return raiden_controller_->controller_address();
  }
  return "";
}

absl::Status KVCacheStore::Save(
    const std::vector<std::string>& block_hashes,
    const std::optional<RaidenId>& dst_raiden_id) {
  return dst_raiden_id.has_value()
             ? SaveRemote(block_hashes, *dst_raiden_id,
                          SaveOwner::kApplication)
             : SaveLocal(block_hashes);
}

absl::Status KVCacheStore::SaveLocal(
    const std::vector<std::string>& block_hashes) {
  if (!raiden_controller_) {
    return absl::FailedPreconditionError("RaidenController is not initialized");
  }

  std::vector<int64_t> src_device_block_ids;
  src_device_block_ids.reserve(block_hashes.size());

  {
    absl::MutexLock lock(mutex_);
    auto lookup_or = backend()->Lookup(block_hashes);
    if (!lookup_or.ok()) return lookup_or.status();
    const auto& slices = lookup_or.value();
    if (slices.size() < block_hashes.size()) {
      return absl::NotFoundError(
          absl::StrCat("Block hash not found: ", block_hashes[slices.size()]));
    }
    for (size_t i = 0; i < slices.size(); ++i) {
      const auto& hash = block_hashes[i];
      const auto& existing = slices[i].second;
      if (existing.status != BlockStatus::HBM) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is not in HBM status: ", hash));
      }
      if (existing.device_block_id == -1) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block device_block_id is -1: ", hash));
      }
      if (backend()->GetPinCount(hash) <= 0) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is not pinned: ", hash));
      }
      if (saving_hashes_.contains(hash)) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is already saving: ", hash));
      }
      src_device_block_ids.push_back(existing.device_block_id);
    }
    for (const auto& hash : block_hashes) {
      saving_hashes_.insert(hash);
    }
  }

  auto host_blocks_or = AllocateBlockIds(block_hashes.size());
  if (!host_blocks_or.ok()) {
    absl::MutexLock lock(mutex_);
    for (const auto& hash : block_hashes) {
      saving_hashes_.erase(hash);
    }
    return host_blocks_or.status();
  }
  const auto& host_block_ids = host_blocks_or.value();

  std::vector<Buffer> src_buffers;
  src_buffers.reserve(src_device_block_ids.size());
  for (int64_t id : src_device_block_ids) {
    src_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                             ::tpu_sync::rpc::MEMORY_TYPE_HBM);
  }
  std::vector<Buffer> dst_buffers;
  dst_buffers.reserve(host_block_ids.size());
  for (int id : host_block_ids) {
    dst_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                             ::tpu_sync::rpc::MEMORY_TYPE_DRAM);
  }

  tsl::Future<> future = raiden_controller_->TransferBuffers(
      src_buffers, dst_buffers, /*staging_host_buffers=*/{},
      /*copy_sizes=*/{});

  {
    absl::MutexLock lock(mutex_);
    active_saves_.push_back(SaveState{
        .future = std::move(future),
        .block_hashes = block_hashes,
        .host_block_ids = host_block_ids,
    });
  }

  return absl::OkStatus();
}

absl::Status KVCacheStore::Load(absl::Span<const std::string> block_hashes,
                                absl::Span<const int> device_block_ids) {
  if (block_hashes.size() != device_block_ids.size()) {
    return absl::InvalidArgumentError(
        "block_hashes and device_block_ids size mismatch");
  }
  if (!raiden_controller_) {
    return absl::FailedPreconditionError("RaidenController is not initialized");
  }
  if (block_hashes.empty()) {
    return absl::OkStatus();
  }

  {
    absl::MutexLock lock(mutex_);
    auto lookup_or = backend()->Lookup(block_hashes);
    if (!lookup_or.ok()) return lookup_or.status();
    const auto& slices = lookup_or.value();
    if (slices.size() < block_hashes.size()) {
      return absl::NotFoundError(
          absl::StrCat("Block hash not found: ", block_hashes[slices.size()]));
    }

    for (size_t i = 0; i < slices.size(); ++i) {
      const auto& hash = block_hashes[i];
      const auto& existing = slices[i].second;
      // LOCAL ONLY, checked before the pin gate: a REMOTE hash here is the
      // wrong API regardless of its pin state. The slices overload is the
      // only peer-load path.
      if (existing.status == BlockStatus::REMOTE) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Load without slices is local-only, but block is REMOTE: ", hash,
            ". Use Load(block_hashes, slices, device_block_ids) to load from "
            "a peer."));
      }
      if (backend()->GetPinCount(hash) <= 0) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is not pinned: ", hash));
      }
      if (loading_hashes_.contains(hash)) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is already loading: ", hash));
      }
      if (existing.status != BlockStatus::HOST &&
          existing.status != BlockStatus::HOST_AND_HBM) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is not on host: ", hash));
      }
      if (existing.host_block_id == -1) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block host_block_id is -1: ", hash));
      }
    }
    for (const auto& hash : block_hashes) {
      loading_hashes_.insert(hash);
    }
  }

  tsl::Future<> future = backend()->Load(
      RaidenId(), block_hashes,
      absl::Span<const int32_t>(
          reinterpret_cast<const int32_t*>(device_block_ids.data()),
          device_block_ids.size()));

  {
    absl::MutexLock lock(mutex_);
    active_loads_.push_back(LoadState{
        .future = std::move(future),
        .block_hashes =
            std::vector<std::string>(block_hashes.begin(), block_hashes.end()),
        .device_block_ids =
            std::vector<int>(device_block_ids.begin(), device_block_ids.end()),
        .from_remote = false,
    });
  }

  return absl::OkStatus();
}

absl::Status KVCacheStore::Load(absl::Span<const std::string> block_hashes,
                                absl::Span<const RaidenBlockId> slices,
                                absl::Span<const int> device_block_ids) {
  if (block_hashes.size() != slices.size() ||
      slices.size() != device_block_ids.size()) {
    return absl::InvalidArgumentError(
        "block_hashes, slices, and device_block_ids size mismatch");
  }
  if (!raiden_controller_) {
    return absl::FailedPreconditionError("RaidenController is not initialized");
  }
  if (block_hashes.empty()) {
    return absl::OkStatus();
  }

  RaidenId remote_id;
  bool from_remote = false;
  {
    absl::MutexLock lock(mutex_);

    BlockStatus first_status = slices[0].status;
    if (first_status == BlockStatus::REMOTE) {
      remote_id = slices[0].raiden_id;
      from_remote = true;
    }

    for (size_t i = 0; i < slices.size(); ++i) {
      const auto& hash = block_hashes[i];
      const auto& existing = slices[i];
      if (loading_hashes_.contains(hash)) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is already loading: ", hash));
      }

      if (first_status == BlockStatus::REMOTE) {
        if (existing.status != BlockStatus::REMOTE) {
          return absl::InvalidArgumentError(
              "Mixed block statuses in a single Load call");
        }
        if (existing.raiden_id != remote_id) {
          return absl::InvalidArgumentError(
              "Mixed remote node IDs in a single Load call");
        }
      } else {
        // The caller's pin is what a successful local load consumes, so it has
        // to exist. The no-slices form has always required it; this form did
        // not, which left one signature hiding two different pin contracts.
        if (backend()->GetPinCount(hash) <= 0) {
          return absl::FailedPreconditionError(
              absl::StrCat("Block is not pinned: ", hash));
        }
        if (existing.status != BlockStatus::HOST &&
            existing.status != BlockStatus::HOST_AND_HBM) {
          return absl::FailedPreconditionError(
              absl::StrCat("Block is not on host: ", hash));
        }
        if (existing.host_block_id == -1) {
          return absl::FailedPreconditionError(
              absl::StrCat("Block host_block_id is -1: ", hash));
        }
      }
    }
    for (const auto& hash : block_hashes) {
      loading_hashes_.insert(hash);
    }
  }

  tsl::Future<> future = backend()->Load(
      remote_id, block_hashes,
      absl::Span<const int32_t>(
          reinterpret_cast<const int32_t*>(device_block_ids.data()),
          device_block_ids.size()),
      slices);

  {
    absl::MutexLock lock(mutex_);
    active_loads_.push_back(LoadState{
        .future = std::move(future),
        .block_hashes =
            std::vector<std::string>(block_hashes.begin(), block_hashes.end()),
        .device_block_ids =
            std::vector<int>(device_block_ids.begin(), device_block_ids.end()),
        .from_remote = from_remote,
    });
  }

  return absl::OkStatus();
}

void KVCacheStore::RegisterReadRemoteHooks() {
  if (!raiden_controller_) {
    return;
  }
  raiden_controller_->SetReadRemoteHooks(
      [this](absl::Span<const std::string> hashes) {
        return this->ValidateAndPinHostBlocks(hashes);
      },
      [this](absl::Span<const std::string> hashes) {
        this->UnpinHostBlocks(hashes);
      });
}

absl::StatusOr<std::vector<int32_t>> KVCacheStore::ValidateAndPinHostBlocks(
    absl::Span<const std::string> block_hashes) {
  absl::MutexLock lock(mutex_);
  auto lookup_or = backend()->Lookup(block_hashes);
  if (!lookup_or.ok()) return lookup_or.status();
  const auto& slices = lookup_or.value();
  if (slices.size() < block_hashes.size()) {
    return absl::NotFoundError(
        absl::StrCat("BLOCK_HASH_NOT_FOUND: ", block_hashes[slices.size()]));
  }

  std::vector<int32_t> src_host_block_ids;
  src_host_block_ids.reserve(slices.size());
  for (size_t i = 0; i < slices.size(); ++i) {
    const auto& existing = slices[i].second;
    if (existing.status != BlockStatus::HOST &&
        existing.status != BlockStatus::HOST_AND_HBM) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Block not resident in host DRAM (status=",
          static_cast<int>(existing.status), "): ", block_hashes[i]));
    }
    src_host_block_ids.push_back(existing.host_block_id);
  }

  if (!backend()->Pin(block_hashes)) {
    return absl::InternalError("Failed to pin host blocks");
  }

  return src_host_block_ids;
}

void KVCacheStore::UnpinHostBlocks(absl::Span<const std::string> block_hashes) {
  backend()->Release(block_hashes);
}

absl::Status KVCacheStore::ReadRemote(
    const std::vector<std::string>& block_hashes,
    const std::vector<RaidenBlockId>& slices,
    const std::vector<int32_t>& device_block_ids) {
  if (block_hashes.empty()) {
    return absl::OkStatus();
  }

  // Validate before allocating anything: an early return past the allocation
  // owes the cleanup below, and there is nothing to clean up yet here.
  if (slices.size() != block_hashes.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("slices size ", slices.size(),
                     " must match block_hashes size ", block_hashes.size()));
  }
  if (device_block_ids.size() != block_hashes.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "device_block_ids size ", device_block_ids.size(),
        " must match block_hashes size ", block_hashes.size(),
        ": read_remote always reads into local HBM"));
  }

  auto host_blocks_or = AllocateBlockIds(block_hashes.size());
  if (!host_blocks_or.ok()) {
    return host_blocks_or.status();
  }

  // Unwinds everything this call has claimed so far. Every failure below is an
  // early return, and each one owes both the reading marks and the landing
  // blocks: an error path that returned without freeing them leaked N host
  // blocks per call, silently and permanently.
  std::vector<std::string> successfully_marked_as_reading;
  successfully_marked_as_reading.reserve(block_hashes.size());
  auto cleanup = absl::MakeCleanup(
      [this, &successfully_marked_as_reading, &host_blocks_or]() {
        DeallocateBlockIds(host_blocks_or.value());
        absl::MutexLock lock(mutex_);
        for (const auto& hash : successfully_marked_as_reading) {
          reading_hashes_.erase(hash);
        }
      });

  std::vector<int> dst_host_block_ids = host_blocks_or.value();

  struct RemoteReadGroup {
    RaidenId src_raiden_id;
    // The peer's ControllerService address, resolved from the global registry
    // below. The controller holds no directory of its own.
    std::string src_controller_address;
    std::vector<int32_t> src_host_block_ids;
    std::vector<int32_t> dst_host_block_ids;
    std::vector<std::string> block_hashes;
    std::vector<int32_t> device_block_ids;
  };
  std::vector<RemoteReadGroup> groups;

  {
    // The source coordinates come from the caller, not from this store's
    // index: `slices[i].raiden_id` names the owning peer and
    // `slices[i].host_block_id` the block on it. The lock still guards
    // reading_hashes_, which is this store's own in-flight marker.
    absl::MutexLock lock(mutex_);
    for (size_t i = 0; i < block_hashes.size(); ++i) {
      const auto& hash = block_hashes[i];
      if (!reading_hashes_.insert(hash).second) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is already reading remote: ", hash));
      }
      successfully_marked_as_reading.push_back(hash);

      const auto& src_id = slices[i].raiden_id;
      auto it = std::find_if(groups.begin(), groups.end(),
                             [&src_id](const RemoteReadGroup& g) {
                               return g.src_raiden_id == src_id;
                             });
      if (it == groups.end()) {
        groups.push_back(RemoteReadGroup{.src_raiden_id = src_id});
        it = groups.end() - 1;
      }
      it->src_host_block_ids.push_back(slices[i].host_block_id);
      it->dst_host_block_ids.push_back(dst_host_block_ids[i]);
      it->block_hashes.push_back(hash);
      it->device_block_ids.push_back(device_block_ids[i]);
    }
  }

  if (!raiden_controller_) {
    return absl::FailedPreconditionError(
        "RaidenController is not initialized for ReadRemote");
  }

  // Resolve every peer BEFORE issuing anything. Resolving inside the issue loop
  // would leave the first group's lease acquired and its transfer running with
  // nothing tracking it when a later group turns out to be unreachable.
  //
  // Cached per peer (resolved_peer_controllers_), and dropped whenever a read
  // against that peer fails -- see the member's comment for why invalidation is
  // what makes a cache here safe at all.
  if (registry_client_ == nullptr) {
    return absl::FailedPreconditionError(
        "ReadRemote needs a global registry: it is what maps the owning peer "
        "to the controller address this store acquires a read lease from. "
        "Construct this store with a global_registry_address.");
  }
  for (auto& group : groups) {
    {
      absl::MutexLock lock(mutex_);
      auto it = resolved_peer_controllers_.find(group.src_raiden_id);
      if (it != resolved_peer_controllers_.end()) {
        group.src_controller_address = it->second;
      }
    }
    if (!group.src_controller_address.empty()) continue;

    absl::StatusOr<global_registry::StoreInfo> store_info =
        registry_client_->ResolveStore(group.src_raiden_id);
    if (!store_info.ok()) {
      return store_info.status();
    }
    if (store_info->controller_address().empty()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Peer ", group.src_raiden_id.job_name, "/",
          group.src_raiden_id.job_replica_id, "/",
          group.src_raiden_id.data_name, "/",
          group.src_raiden_id.data_replica_idx,
          " is registered but published no controller address, so it cannot "
          "serve a remote read."));
    }
    group.src_controller_address = store_info->controller_address();
    {
      absl::MutexLock lock(mutex_);
      resolved_peer_controllers_[group.src_raiden_id] =
          group.src_controller_address;
    }
  }

  // One lease per owning peer. The per-group futures are joined, so if ANY
  // group fails -- transfer error or a verdict other than HELD -- the whole
  // batch discards, including groups whose bytes landed perfectly. That is
  // fail-closed and consistent with the commit-as-a-unit invariant. Committing
  // only the healthy groups would need per-group RemoteReadState and is
  // exactly where a partial-promote bug would enter; do not "optimise" it
  // without splitting the state first.
  std::vector<tsl::Future<>> futures;
  futures.reserve(groups.size());
  for (const auto& group : groups) {
    futures.push_back(raiden_controller_->ReadRemote(
        group.src_controller_address, group.src_host_block_ids,
        group.dst_host_block_ids, group.block_hashes, group.device_block_ids));
  }

  tsl::Future<> combined_future;
  if (futures.size() == 1) {
    combined_future = std::move(futures[0]);
  } else {
    combined_future = tsl::JoinFutures(futures);
  }

  {
    absl::MutexLock lock(mutex_);
    std::vector<RaidenId> peers;
    peers.reserve(groups.size());
    for (const auto& group : groups) peers.push_back(group.src_raiden_id);
    active_remote_reads_.emplace(std::move(combined_future),
                                 RemoteReadState{
                                     .block_hashes = block_hashes,
                                     .src_raiden_ids = std::move(peers),
                                     .host_block_ids = dst_host_block_ids,
                                 });
  }

  // Issued: the staging blocks now belong to the read, and the reading marks
  // are cleared by the poller when it goes terminal.
  std::move(cleanup).Cancel();
  return absl::OkStatus();
}

KVCacheStore::PollSaveStatusResult KVCacheStore::PollSaveStatus() {
  // Drives the LOCAL saves whose DMA has landed; a remote save needs no
  // driving here -- its verdict arrives on the offer's own call, and this
  // only reads the mailbox it was filed in.
  PollFuturesInternal();
  absl::MutexLock lock(mutex_);
  std::vector<std::string> pending;
  for (const auto& state : active_saves_) {
    for (const auto& hash : state.block_hashes) {
      pending.push_back(hash);
    }
  }
  for (const auto& [key, state] : active_remote_writes_) {
    // The sweep's operations are not this caller's to wait on: it will never
    // receive a verdict for them, so reporting them as pending would describe
    // a wait that never ends. Unaccepted / failed offers (operation_id == 0)
    // already reported their error synchronously to the caller.
    if (state.owner != SaveOwner::kApplication || state.operation_id == 0) {
      continue;
    }
    pending.insert(pending.end(), state.block_hashes.begin(),
                   state.block_hashes.end());
  }

  std::vector<std::string> done = std::move(done_saves_);
  std::vector<std::string> failed = std::move(failed_saves_);
  done_saves_.clear();
  failed_saves_.clear();
  // The application's mailbox only. A sweep verdict drained here would be
  // discarded -- nothing in this result names an owner -- and the sweep would
  // then wait for it forever.
  RemoteWriteVerdicts remote;
  remote.done.swap(application_remote_writes_.done);
  remote.failed.swap(application_remote_writes_.failed);
  remote.existing.swap(application_remote_writes_.existing);
  remote.unregistered.swap(application_remote_writes_.unregistered);

  done.insert(done.end(), remote.done.begin(), remote.done.end());
  failed.insert(failed.end(), remote.failed.begin(), remote.failed.end());

  std::vector<std::string> existing = std::move(remote.existing);
  std::vector<std::string> unregistered = std::move(remote.unregistered);

  return PollSaveStatusResult{
      .done = std::move(done),
      .failed = std::move(failed),
      .pending = std::move(pending),
      .existing = std::move(existing),
      .unregistered = std::move(unregistered),
  };
}

KVCacheStore::PollLoadStatusResult KVCacheStore::PollLoadStatus() {
  PollFuturesInternal();
  absl::MutexLock lock(mutex_);
  std::vector<std::string> pending;
  for (const auto& state : active_loads_) {
    for (const auto& hash : state.block_hashes) {
      pending.push_back(hash);
    }
  }
  std::vector<std::string> done = std::move(done_loads_);
  std::vector<std::string> failed = std::move(failed_loads_);
  done_loads_.clear();
  failed_loads_.clear();
  return PollLoadStatusResult{
      .done = std::move(done),
      .failed = std::move(failed),
      .pending = std::move(pending),
  };
}

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
KVCacheStore::PollRemoteReadStatus() {
  PollFuturesInternal();
  absl::MutexLock lock(mutex_);
  std::vector<std::string> pending;
  for (const auto& [fut, state] : active_remote_reads_) {
    for (const auto& hash : state.block_hashes) {
      pending.push_back(hash);
    }
  }
  std::vector<std::string> done = std::move(done_remote_reads_);
  std::vector<std::string> failed = std::move(failed_remote_reads_);
  done_remote_reads_.clear();
  failed_remote_reads_.clear();
  return std::make_tuple(done, failed, pending);
}

absl::StatusOr<size_t> KVCacheStore::RecoverFromLocalManifest() {
  if (!raiden_controller_) {
    return absl::FailedPreconditionError(
        "RaidenController is required for crash recovery");
  }
  if (backend()->GetSize() > 0) {
    return absl::FailedPreconditionError(
        "RecoverFromLocalManifest can only be called on an empty cache store");
  }
  absl::StatusOr<size_t> recovered = backend()->RecoverFromLocalManifest();
  if (recovered.ok() && *recovered > 0 && store_monitor_ != nullptr) {
    // The recovered blocks predate this process; if the downtime outlived the
    // registration TTL, the registry has purged their entries. Report them.
    store_monitor_->RequestRepublish();
  }
  return recovered;
}

size_t KVCacheStore::Evict(const std::vector<std::string>& block_hashes) {
  if (block_hashes.empty()) {
    return 0;
  }
  // The backend handles thread-safe eviction and unregisters erased blocks
  // from the global registry outside its lock. We do not hold store mutex_
  // across this call to avoid blocking concurrent operations on network I/O.
  const std::vector<int> host_ids_to_deallocate =
      backend()->Evict(block_hashes);

  if (host_ids_to_deallocate.empty()) {
    return 0;
  }

  DeallocateBlockIds(host_ids_to_deallocate);

  return host_ids_to_deallocate.size();
}

bool KVCacheStore::SweepOnce() {
  const int free_blocks =
      raiden_controller_->block_manager()->num_free_blocks();
  const int total_blocks = raiden_controller_->block_manager()->total_blocks();
  if (total_blocks <= 0) {
    return false;
  }
  const double free_ratio = static_cast<double>(free_blocks) / total_blocks;
  // Ends the current pressure episode; the next one re-fetches targets.
  auto end_episode = [this] {
    sweep_active_ = false;
    placement_targets_.clear();
    return false;
  };

  if (!sweep_active_) {
    // Idle and enough free blocks: nothing to do.
    if (free_ratio >= monitor_config_.evict_low_watermark) {
      return false;
    }
    // Free blocks fell below the low watermark: a pressure episode begins
    // (it ends when they recover to the high watermark). The targets are
    // fetched once here and reused for the whole episode, so even a long
    // drain costs one registry read. If fleets of stores hitting pressure
    // together ever make these per-episode reads a load problem for the
    // registry, the fetch could be decoupled from the sweep into its own
    // periodic task maintaining a local target list -- at the cost of
    // staler placement data.
    sweep_active_ = true;
    placement_targets_.clear();
    auto targets_or =
        registry_client_->GetPlacementTargets(raiden_id_, kMaxPlacementTargets);
    if (targets_or.ok()) {
      for (const auto& info : *targets_or) {
        placement_targets_.push_back(RaidenId{
            info.raiden_id().job_name(), info.raiden_id().job_replica_id(),
            info.raiden_id().data_name(),
            static_cast<int>(info.raiden_id().data_replica_idx())});
      }
    } else {
      LOG(WARNING) << "Evict sweep could not fetch placement targets: "
                   << targets_or.status()
                   << ". Dropping cold blocks locally instead.";
    }
  } else if (free_ratio >= monitor_config_.evict_high_watermark) {
    // Recovered to the high watermark: the episode is over.
    return end_episode();
  }

  // One batch per step, sized to what the high watermark still needs and
  // capped so a single step stays short.
  const int deficit =
      static_cast<int>(
          std::ceil(monitor_config_.evict_high_watermark * total_blocks)) -
      free_blocks;
  std::vector<std::string> batch;
  {
    absl::MutexLock lock(mutex_);
    batch =
        backend()->GetEvictableKeys(std::min(deficit, kMaxEvictBatchBlocks));
  }
  if (batch.empty()) {
    // Everything left is pinned; nothing more this episode can free.
    return end_episode();
  }

  // Offer the batch to the episode's targets in order. Every iteration
  // either sends the batch or drops one target, so this ends.
  while (!placement_targets_.empty()) {
    const RaidenId dst = placement_targets_.front();
    // SaveRemote's pin contract: the caller pins, and only success consumes
    // the pin. The sweep is its own caller here; whatever a refusal or a
    // failed transfer leaves pinned is released below, or the blocks would
    // never become evictable again.
    if (!backend()->Pin(batch)) {
      return end_episode();
    }
    absl::Status offered = SaveRemote(batch, dst, SaveOwner::kSweep);
    if (!offered.ok()) {
      backend()->Release(batch);
      // Refused (peer out of free blocks) or unreachable: this target is
      // out for the rest of the episode; try the batch on the next one.
      LOG(INFO) << "Evict sweep target " << dst
                << " declined a demotion batch: " << offered;
      placement_targets_.erase(placement_targets_.begin());
      continue;
    }
    const BatchWriteResult result = WaitForBatchWriteResult(batch);
    // Offers that settled without success kept the sweep's pin; drop it
    // before evicting, which skips pinned blocks.
    if (!result.still_pinned.empty()) {
      backend()->Release(result.still_pinned);
    }
    const size_t evicted = Evict(result.freeable);
    if (result.transfer_failed) {
      // The peer accepted but a transfer failed; the failed blocks stay
      // local. Drop the target so the next step retries them elsewhere.
      placement_targets_.erase(placement_targets_.begin());
    } else if (evicted == 0) {
      // The peer holds the whole batch, yet nothing could be freed locally:
      // readers pinned every block between the offer and here. Re-offering
      // the same keys could spin without raising the free count, so end the
      // episode; the next wake re-evaluates.
      return end_episode();
    }
    return true;
  }

  // No target will take the batch (none exist, all dropped, or the registry
  // was unreachable): drop it locally -- the same discard AllocateBlockIds
  // would be forced into later, done early enough to keep absorbing writes.
  if (Evict(batch) == 0) {
    // The whole batch got pinned since it was picked: end the episode
    // rather than spin on the same keys.
    return end_episode();
  }
  return true;
}

bool KVCacheStore::RepublishOnce() {
  auto* host_backend = dynamic_cast<HostOffloadBackend*>(backend().get());
  if (host_backend == nullptr) {
    return false;
  }

  if (republish_queue_.empty()) {
    republish_queue_ = host_backend->SnapshotHostResidentHashes();
    if (republish_queue_.empty()) {
      return false;
    }
    LOG(INFO) << "Republishing " << republish_queue_.size()
              << " host-resident block(s) to the global registry.";
  }

  const size_t take =
      std::min(kMaxRepublishBatchBlocks, republish_queue_.size());
  std::vector<std::string> batch(republish_queue_.end() - take,
                                 republish_queue_.end());
  republish_queue_.resize(republish_queue_.size() - take);

  // Pin-validate against the live cache: hashes evicted since the snapshot
  // are skipped here, and the pins keep the rest resident until the registry
  // has acknowledged them -- an eviction's Unregister can then only run
  // after this batch's Register, never be overwritten by it.
  const std::vector<std::pair<std::string, int32_t>> pinned =
      host_backend->PinPresentHostResident(batch);
  if (pinned.empty()) {
    return !republish_queue_.empty();
  }
  std::vector<global_registry::Registration> registrations;
  std::vector<std::string> pinned_hashes;
  registrations.reserve(pinned.size());
  pinned_hashes.reserve(pinned.size());
  for (const auto& [hash, host_block_id] : pinned) {
    registrations.push_back({
        .prefix_hash = hash,
        .raiden_id = raiden_id_,
        .block_id = host_block_id,
    });
    pinned_hashes.push_back(hash);
  }

  // The deadline must stay under the heartbeat period: this blocks the
  // monitor thread, and a Register hanging longer than a period could
  // starve the heartbeats whose TTL budget is a small multiple of it.
  const absl::Duration timeout =
      std::min(monitor_config_.heartbeat_period, absl::Seconds(10));
  const absl::Status registered =
      registry_client_->RegisterAsync(registrations, timeout).Await();
  host_backend->Release(pinned_hashes);

  if (!registered.ok()) {
    LOG(WARNING) << "Inventory republish failed with "
                 << republish_queue_.size()
                 << " block(s) still unreported: " << registered
                 << ". Dropping the rest; the next registration lapse "
                    "restarts the report.";
    republish_queue_.clear();
    return false;
  }
  return !republish_queue_.empty();
}

KVCacheStore::RemoteWriteVerdicts KVCacheStore::DrainSweepVerdicts() {
  PollFuturesInternal();
  absl::MutexLock lock(mutex_);
  mutex_.AwaitWithTimeout(
      absl::Condition(
          +[](KVCacheStore* store) {
            store->mutex_.AssertHeld();
            return !store->sweep_remote_writes_.done.empty() ||
                   !store->sweep_remote_writes_.failed.empty() ||
                   !store->sweep_remote_writes_.existing.empty() ||
                   !store->sweep_remote_writes_.unregistered.empty();
          },
          this),
      absl::Seconds(1));
  RemoteWriteVerdicts verdicts;
  verdicts.done.swap(sweep_remote_writes_.done);
  verdicts.failed.swap(sweep_remote_writes_.failed);
  verdicts.existing.swap(sweep_remote_writes_.existing);
  verdicts.unregistered.swap(sweep_remote_writes_.unregistered);
  return verdicts;
}

KVCacheStore::BatchWriteResult KVCacheStore::WaitForBatchWriteResult(
    const std::vector<std::string>& batch) {
  // Everything drained here belongs to this batch. SaveRemote tags the
  // operation with its owner and FinishRemoteWrite files the verdict under
  // that owner, so the application cannot take these and this cannot take the
  // application's; and the monitor's thread, the only caller, runs one batch
  // at a time. SaveRemote's HOLD deadline guarantees each block eventually
  // leaves `pending`.
  absl::flat_hash_set<std::string> pending(batch.begin(), batch.end());
  BatchWriteResult result;
  while (!pending.empty()) {
    const RemoteWriteVerdicts verdicts = DrainSweepVerdicts();
    // Completed transfers: the peer holds these blocks now, and success
    // already consumed the sweep's pin.
    for (const std::string& hash : verdicts.done) {
      pending.erase(hash);
      result.freeable.push_back(hash);
    }
    for (const std::string& hash : verdicts.failed) {
      pending.erase(hash);
      result.still_pinned.push_back(hash);
    }
    // `existing` annotates refusals whose bytes the peer already holds -- as
    // freeable as a completed transfer. It is a SUBSET of `failed`, so those
    // hashes are both unpinned above and freed here: nothing consumed the
    // sweep's pin, yet the local copy is droppable. Everything else that
    // failed (including `unregistered`: landed but unfindable there) keeps its
    // local copy.
    for (const std::string& hash : verdicts.existing) {
      result.freeable.push_back(hash);
    }
    if (verdicts.failed.size() > verdicts.existing.size()) {
      result.transfer_failed = true;
    }
  }
  return result;
}

absl::StatusOr<std::vector<int>> KVCacheStore::AllocateBlockIds(int needed) {
  std::vector<std::string> hashes_to_deallocate;
  bool request_sweep = false;
  {
    absl::MutexLock lock(mutex_);
    int free_count = raiden_controller_->block_manager()->num_free_blocks();
    // Wake the evict sweep the moment this allocation dips below its low
    // watermark, instead of leaving detection to the sweep's fallback
    // period. The drop-evict below stays the last-ditch fallback for
    // allocations the sweep has not made room for.
    if (monitor_config_.enable_evict_sweep && store_monitor_ != nullptr) {
      const int total = raiden_controller_->block_manager()->total_blocks();
      request_sweep =
          total > 0 &&
          free_count - needed < monitor_config_.evict_low_watermark * total;
    }
    int to_free = needed - free_count;
    if (to_free > 0) {
      hashes_to_deallocate = backend()->GetEvictableKeys(to_free);
      if (hashes_to_deallocate.size() < static_cast<size_t>(to_free)) {
        return absl::ResourceExhaustedError(
            absl::StrCat("Insufficient free blocks and not enough evictable "
                         "blocks. Needed: ",
                         needed, ", Free: ", free_count,
                         ", Evictable: ", hashes_to_deallocate.size()));
      }
    }
  }

  if (request_sweep) {
    store_monitor_->RequestSweep();
  }

  if (!hashes_to_deallocate.empty()) {
    Evict(hashes_to_deallocate);
  }

  return raiden_controller_->AllocateBlockIds(needed);
}

void KVCacheStore::DeallocateBlockIds(absl::Span<const int> block_ids) {
  if (raiden_controller_) {
    auto status = raiden_controller_->DeallocateBlockIds(block_ids);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to deallocate host block IDs: "
                   << status.message();
    }
  }
}

// ===========================================================================
// Remote write -- source side.
// ===========================================================================

namespace {

// How long this source keeps its blocks intact and keeps asking. The
// destination is asked for HOLD minus a margin, and clamps that to its own
// cap, so the source always outlives the destination's verdict.
constexpr absl::Duration kDefaultRemoteWriteHold = absl::Seconds(30);

// Covers one-way RPC latency, clock skew between the two hosts, and scheduler
// jitter -- none of which this design controls. Not a tuning knob: the two
// timers live in different processes and are armed at different moments, so
// without a gap the destination's deadline outlives the source's HOLD by the
// time it took the request to arrive. In that window the source has unpinned
// and reported failure while the destination can still commit, and globally
// register, bytes read out of blocks the source may already have reused.
constexpr absl::Duration kRemoteWriteMargin = absl::Seconds(5);

absl::Duration RemoteWriteHold() {
  const char* env = std::getenv("RAIDEN_REMOTE_WRITE_HOLD_S");
  if (env == nullptr) {
    return kDefaultRemoteWriteHold;
  }
  int seconds = 0;
  if (!absl::SimpleAtoi(env, &seconds) || seconds <= 0) {
    LOG(WARNING) << "Ignoring RAIDEN_REMOTE_WRITE_HOLD_S=\"" << env
                 << "\": expected a positive number of seconds.";
    return kDefaultRemoteWriteHold;
  }
  return absl::Seconds(seconds);
}

}  // namespace

size_t KVCacheStore::InFlightRemoteWritesCountForTesting() const {
  absl::MutexLock lock(mutex_);
  return active_remote_writes_.size();
}

absl::Status KVCacheStore::SaveRemote(
    const std::vector<std::string>& block_hashes, const RaidenId& dst_raiden_id,
    SaveOwner owner) {
  if (block_hashes.empty()) {
    return absl::OkStatus();
  }
  if (dst_raiden_id.empty()) {
    return absl::InvalidArgumentError("A remote save requires a destination id");
  }
  if (dst_raiden_id == raiden_id_) {
    return absl::InvalidArgumentError(
        "The save destination is this store; there is nothing to transfer");
  }

  auto* backend = dynamic_cast<HostOffloadBackend*>(this->backend().get());
  if (backend == nullptr) {
    return absl::FailedPreconditionError(
        "A remote save requires a HostOffloadBackend at tier 0");
  }

  // Everything before the RPC is undone by this if we do not get as far as
  // recording the operation.
  std::vector<std::string> marked;
  std::vector<std::string> pinned;
  auto rollback = absl::MakeCleanup([&]() {
    if (!pinned.empty()) {
      backend->Release(pinned);
    }
    if (!marked.empty()) {
      absl::MutexLock lock(mutex_);
      for (const auto& hash : marked) {
        saving_hashes_.erase(hash);
      }
    }
  });

  std::vector<int32_t> src_host_block_ids;
  {
    absl::MutexLock lock(mutex_);
    auto lookup_or = backend->Lookup(block_hashes);
    if (!lookup_or.ok()) return lookup_or.status();
    const auto& slices = lookup_or.value();
    if (slices.size() < block_hashes.size()) {
      return absl::NotFoundError(
          absl::StrCat("Block hash not found: ", block_hashes[slices.size()]));
    }
    src_host_block_ids.reserve(slices.size());
    for (size_t i = 0; i < slices.size(); ++i) {
      const auto& [hash, slice] = slices[i];
      if (slice.status != BlockStatus::HOST &&
          slice.status != BlockStatus::HOST_AND_HBM) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Block is not resident in host DRAM, so there is nothing to "
            "offer: ",
            absl::BytesToHexString(hash)));
      }
      // The caller's pin, which a successful save consumes. The local branch
      // has always required it; this one did not, and its old contract said
      // the caller's pin was separate and might not exist at all -- which made
      // "a save consumes one pin" unstatable for half the API.
      if (backend->GetPinCount(hash) <= 0) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Block is not pinned: ", absl::BytesToHexString(hash)));
      }
      if (!saving_hashes_.insert(hash).second) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is already saving: ",
                         absl::BytesToHexString(hash)));
      }
      marked.push_back(hash);
      src_host_block_ids.push_back(slice.host_block_id);
    }
  }

  // The INTERNAL pin, separate from whatever the caller holds. The block ids
  // we are about to send are only authoritative for as long as this holds.
  if (!backend->Pin(block_hashes)) {
    return absl::ResourceExhaustedError(
        "Failed to pin host blocks for a remote write");
  }
  pinned = block_hashes;

  const absl::Duration hold = RemoteWriteHold();
  if (hold <= kRemoteWriteMargin) {
    return absl::FailedPreconditionError(absl::StrCat(
        "RAIDEN_REMOTE_WRITE_HOLD_S (", absl::FormatDuration(hold),
        ") must exceed the ", absl::FormatDuration(kRemoteWriteMargin),
        " margin, or there is no deadline left to ask the destination for."));
  }
  // Armed HERE, with the pin, rather than after the ack: the pin has to be
  // protected even if the RPC itself hangs. The cost of that ordering is that
  // the margin must also cover the round trip.
  const absl::Time hold_expiry = absl::Now() + hold;

  OperationKey op_key = 0;
  {
    absl::MutexLock lock(mutex_);
    op_key = next_op_key_++;
    active_remote_writes_[op_key] = RemoteWriteState{
        .key = op_key,
        .dst_raiden_id = dst_raiden_id,
        .operation_id = 0,
        .block_hashes = block_hashes,
        .hold_expiry = hold_expiry,
        .owner = owner,
    };
  }
  // Recorded before the call: the internal pin is now managed by
  // active_remote_writes_ and will only be released by a settle path, never
  // by rollback.
  std::move(rollback).Cancel();

  auto on_verdict = [lifetime = lifetime_, op_key, dst_raiden_id](
                        absl::Status status,
                        std::optional<proto::WriteRemoteResult> result,
                        uint64_t stream_op_id) {
    absl::MutexLock lock(lifetime->mu);
    if (lifetime->store == nullptr) {
      return;
    }

    auto settle_verdict = [](KVCacheStore* store, OperationKey key,
                             proto::PollWriteRemoteResponse::State state,
                             std::vector<std::string> existing_hashes,
                             std::vector<std::string> unregistered_hashes) {
      auto taken = store->TakeRemoteWrite(key);
      if (!taken.has_value()) {
        return;
      }
      switch (state) {
        case proto::PollWriteRemoteResponse::COMMITTED:
        case proto::PollWriteRemoteResponse::ALL_EXIST:
          store->OnWriteRemoteVerdict(std::move(*taken),
                                      /*succeeded=*/true, {});
          break;
        case proto::PollWriteRemoteResponse::PARTIAL_EXIST:
          store->OnWriteRemoteVerdict(std::move(*taken), /*succeeded=*/false,
                                      std::move(existing_hashes));
          break;
        case proto::PollWriteRemoteResponse::STORED_UNREGISTERED:
          store->OnWriteRemoteVerdict(std::move(*taken), /*succeeded=*/false, {},
                                      std::move(unregistered_hashes));
          break;
        default:
          store->OnWriteRemoteVerdict(std::move(*taken),
                                      /*succeeded=*/false, {});
          break;
      }
    };

    if (result.has_value()) {
      settle_verdict(
          lifetime->store, op_key, result->state(),
          std::vector<std::string>(result->existing_hashes().begin(),
                                   result->existing_hashes().end()),
          std::vector<std::string>(result->unregistered_hashes().begin(),
                                   result->unregistered_hashes().end()));
    } else {
      // The stream ended without a verdict. Decide on the clock this store
      // owns -- the operation's CURRENT hold_expiry -- not on the status the
      // call carried. The two can disagree: the destination grants its own
      // deadline, the ack may have extended hold_expiry past the deadline this
      // call was armed with, and the lambda's captured copy cannot see that
      // extension. A DEADLINE_EXCEEDED from a call whose deadline the grant
      // outran arrives while the destination is still entitled to pull, and
      // must recover, not settle. Both values are read in one locked lookup.
      uint64_t op_id = stream_op_id;
      absl::Time current_hold_expiry = absl::InfinitePast();
      {
        absl::MutexLock lock(lifetime->store->mutex_);
        auto it = lifetime->store->active_remote_writes_.find(op_key);
        if (it == lifetime->store->active_remote_writes_.end()) {
          // Already settled (all-exist/partial-exist inside save(), or
          // teardown took it). Nothing to do, and nothing to ask about.
          return;
        }
        if (op_id == 0) {
          op_id = it->second.operation_id;
        }
        current_hold_expiry = it->second.hold_expiry;
      }
      const absl::Duration remaining_hold = current_hold_expiry - absl::Now();
      if (op_id == 0 || remaining_hold <= absl::ZeroDuration()) {
        // Either the destination never told this source an operation exists
        // (no ack arrived, so there is nothing to ask about -- see the S5
        // comment in SaveRemote), or the hold this store promised has fully
        // elapsed: the ordinary ending of an unanswered offer.
        settle_verdict(lifetime->store, op_key,
                       proto::PollWriteRemoteResponse::FAILED, {}, {});
        return;
      }
      // The operation is still inside its hold: ask the destination once, and
      // ask it to hold the answer until the operation is terminal or the hold
      // ends. One attempt, ever -- the continuation below always settles.
      auto* host_backend =
          dynamic_cast<HostOffloadBackend*>(lifetime->store->backends_[0].get());
      if (host_backend != nullptr) {
          auto fut = host_backend->PollWriteRemoteAsync(
              dst_raiden_id, op_id, absl::ToInt64Milliseconds(remaining_hold));
          fut.OnReady([lifetime, op_key, settle_verdict](
                          absl::StatusOr<proto::PollWriteRemoteResponse> resp) {
            CompletionExecutor::Schedule(
                [lifetime, op_key, settle_verdict, resp = std::move(resp)]() {
                  absl::MutexLock lock(lifetime->mu);
                  if (lifetime->store == nullptr) {
                    return;
                  }
                  if (resp.ok()) {
                    settle_verdict(
                        lifetime->store, op_key, resp->state(),
                        std::vector<std::string>(resp->existing_hashes().begin(),
                                                 resp->existing_hashes().end()),
                        std::vector<std::string>(resp->unregistered_hashes().begin(),
                                                 resp->unregistered_hashes().end()));
                  } else {
                    settle_verdict(lifetime->store, op_key,
                                   proto::PollWriteRemoteResponse::FAILED, {}, {});
                  }
                });
          });
          return;
      }
      settle_verdict(lifetime->store, op_key,
                     proto::PollWriteRemoteResponse::FAILED, {}, {});
    }
  };

  auto ack_or =
      backend->BeginWriteRemote(dst_raiden_id, block_hashes, src_host_block_ids,
                                hold - kRemoteWriteMargin, hold,
                                std::move(on_verdict));
  if (!ack_or.ok()) {
    // The offer is undone completely: pin released, marks dropped, no verdict
    // filed -- the caller is told by the return value, which is how a
    // synchronous refusal has always been reported. A DEADLINE_EXCEEDED here
    // is no exception: the only deadline on the offer call is the hold window
    // itself, so by the time it fires the hold this store promised has fully
    // elapsed, and there is nothing left to protect by keeping the pin. (An
    // earlier revision kept the pin "until hold_expiry" for this case, but no
    // settle path ever fired for an operation with no ack -- the pin and the
    // record leaked until the store was destroyed.)
    //
    // KNOWN GAP, deliberately left as it is (review finding S5). The
    // destination records the operation and starts pulling BEFORE it answers,
    // so a lost ANSWER is not a lost offer: it may be pulling from these
    // blocks right now, and releasing the pin lets the LRU hand one to another
    // hash underneath it. Holding instead is a one-line change here -- keep
    // the operation, let the HOLD expiring settle it -- and it is not made
    // because of what it costs everywhere else: a peer that is simply DOWN
    // fails the same way, and every offer to it would then freeze its batch
    // for the whole HOLD. The evict sweep is built on the opposite assumption,
    // that an unreachable target is dropped and the same batch goes to the
    // next one immediately. gRPC reports "nothing listens there" and "the
    // connection broke after the server took it" both as UNAVAILABLE, so the
    // frequent, harmless case cannot be told from the rare, dangerous one.
    //
    // Trading a common recovery path for a narrow race needs a decision this
    // code cannot make for itself, so it keeps today's behaviour and states
    // the hazard instead of hiding it.
    if (auto taken = TakeRemoteWrite(op_key); taken.has_value()) {
      backend->Release(taken->block_hashes);
      absl::MutexLock lock(mutex_);
      for (const auto& hash : taken->block_hashes) {
        saving_hashes_.erase(hash);
      }
    }
    return ack_or.status();
  }


  const auto& ack = *ack_or;
  if (ack.all_exist) {
    // SUCCESS with nothing to wait for.
    auto taken = TakeRemoteWrite(op_key);
    if (taken.has_value()) {
      OnWriteRemoteVerdict(std::move(*taken), /*succeeded=*/true, {});
    }
    return absl::OkStatus();
  }
  if (!ack.existing_hashes.empty()) {
    // FAILURE, and this store does not retry the remainder: the caller gets
    // the list and decides.
    auto taken = TakeRemoteWrite(op_key);
    if (taken.has_value()) {
      OnWriteRemoteVerdict(std::move(*taken), /*succeeded=*/false,
                           std::move(ack.existing_hashes));
    }
    return absl::OkStatus();
  }

  // Belt and braces. The destination clamps to its own cap, which is below the
  // default HOLD, so this should be unreachable -- but the two values live in
  // different processes and neither side can check the invariant alone. If it
  // ever inverts, keep the pin until the granted deadline has elapsed rather
  // than releasing it while the destination may still be pulling. The extended
  // hold_expiry written below is what on_verdict reads when the call dies: the
  // call's own deadline cannot be moved once the call is made, so a
  // DEADLINE_EXCEEDED inside the extended hold recovers instead of settling.
  if (ack.granted_deadline >= hold) {
    LOG(ERROR) << "Destination granted a deadline of " << ack.granted_deadline
               << ", which is not shorter than this "
               << "source's HOLD of " << hold
               << ". Extending the hold rather than unpinning while the "
                  "destination may still be reading.";
  }

  {
    absl::MutexLock lock(mutex_);
    auto it = active_remote_writes_.find(op_key);
    if (it != active_remote_writes_.end()) {
      it->second.operation_id = ack.operation_id;
      it->second.cancel = ack.cancel;
      it->second.hold_expiry =
          std::max(hold_expiry, absl::Now() + ack.granted_deadline);
    }
  }
  return absl::OkStatus();
}

std::optional<KVCacheStore::RemoteWriteState> KVCacheStore::TakeRemoteWrite(
    OperationKey key) {
  absl::MutexLock lock(mutex_);
  auto it = active_remote_writes_.find(key);
  if (it == active_remote_writes_.end()) {
    return std::nullopt;
  }
  RemoteWriteState state = std::move(it->second);
  active_remote_writes_.erase(it);
  return state;
}

void KVCacheStore::OnWriteRemoteVerdict(RemoteWriteState state, bool succeeded,
                                        std::vector<std::string> existing,
                                        std::vector<std::string> unregistered) {
  if (auto* backend = this->backend().get(); backend != nullptr) {
    // The INTERNAL pin, which protected the blocks while the destination might
    // still be reading them. Dropped whichever way the offer ended.
    backend->Release(state.block_hashes);
    if (succeeded) {
      // ...and separately the CALLER's, which a successful save consumes. Two
      // releases because they are two different pins: the caller's says "I am
      // using this", the internal one says "a peer may still be pulling it".
      //
      // Doing it here covers BOTH terminal routes: the streamed (or
      // recovered) verdict and the synchronous all-exist path, which settles
      // inside SaveRemote and never produces a verdict message at all. An
      // auto-unpin keyed to arriving verdicts would leak the caller's pin on
      // every batch the destination already held.
      backend->Release(state.block_hashes);
    }
  }
  absl::MutexLock lock(mutex_);
  // Addressed to whoever asked for this save; see SaveOwner.
  RemoteWriteVerdicts& verdicts = state.owner == SaveOwner::kSweep
                                      ? sweep_remote_writes_
                                      : application_remote_writes_;
  for (const auto& hash : state.block_hashes) {
    saving_hashes_.erase(hash);
    (succeeded ? verdicts.done : verdicts.failed).push_back(hash);
  }
  for (auto& hash : existing) {
    verdicts.existing.push_back(std::move(hash));
  }
  for (auto& hash : unregistered) {
    verdicts.unregistered.push_back(std::move(hash));
  }
}

void KVCacheStore::PollerLoop() {
  while (!stop_poller_.load()) {
    PollFuturesInternal();
    absl::SleepFor(absl::Milliseconds(10));
  }
}

void KVCacheStore::PollSavesInternal(std::vector<SaveState> ready_saves) {
  for (auto& state : ready_saves) {
    absl::Status status = state.future.Await();
    absl::MutexLock lock(mutex_);
    if (status.ok()) {
      std::vector<global_registry::Registration> write_through_regs;
      write_through_regs.reserve(state.block_hashes.size());
      // Hoisted out of the lookup scope: the caller's pins on these hashes
      // are consumed below, on a path that may run after this scope ends.
      std::vector<std::string> update_hashes;
      auto lookup_or = backend()->Lookup(state.block_hashes,
                                         LookupOptions{.enable_global = false});
      if (lookup_or.ok()) {
        const auto& slices = lookup_or.value();
        std::vector<RaidenBlockId> update_slices;
        for (size_t i = 0; i < state.block_hashes.size(); ++i) {
          const auto& hash = state.block_hashes[i];
          if (i < slices.size()) {
            RaidenBlockId block = slices[i].second;
            block.host_block_id = state.host_block_ids[i];
            block.status = BlockStatus::HOST_AND_HBM;
            update_hashes.push_back(hash);
            update_slices.push_back(block);
            if (registry_client_) {
              write_through_regs.push_back({
                  .prefix_hash = hash,
                  .raiden_id = raiden_id_,
                  .block_id = state.host_block_ids[i],
              });
            }
          }
          done_saves_.push_back(hash);
        }
        if (!update_hashes.empty()) {
          backend()->Insert(update_hashes, update_slices, /*on_host=*/true);
        }
      } else {
        DeallocateBlockIds(state.host_block_ids);
      }
      if (!write_through_regs.empty() && registry_client_) {
        // The pin is spent in the callback, not here: released inline, the
        // entry could be evicted while the publish is in flight, and the
        // publish would then advertise a host block already freed.
        const size_t num_blocks = write_through_regs.size();
        bool admitted = false;
        {
          absl::MutexLock wt_lock(wt_state_->mutex);
          admitted = wt_state_->in_flight_blocks + num_blocks <=
                     wt_state_->max_in_flight_blocks;
          if (admitted) {
            wt_state_->in_flight_blocks += num_blocks;
          } else {
            LOG_EVERY_N_SEC(WARNING, 5)
                << "Not advertising " << num_blocks
                << " saved block(s) to the global registry: "
                << wt_state_->in_flight_blocks
                << " blocks are already pinned by write-throughs waiting on "
                   "it, at the bound of "
                << wt_state_->max_in_flight_blocks
                << ". Peers will not find them; the alternative is pinning "
                   "them out of reach of eviction until the registry answers, "
                   "which drains the host block pool.";
          }
        }
        if (!admitted) {
          if (!update_hashes.empty()) {
            backend()->Release(update_hashes);
          }
        } else {
          // OnReady can run inline on this thread, which holds `mutex_`, so
          // Release may run under it -- the same order as the branch below.
          registry_client_
              ->RegisterAsync(write_through_regs,
                              global_registry::kUnwaitedMutationTimeout)
              .OnReady([wt_state = wt_state_, backend = this->backend(),
                        saved = update_hashes,
                        num_blocks](absl::Status s) {
                {
                  absl::MutexLock wt_lock(wt_state->mutex);
                  wt_state->in_flight_blocks -= num_blocks;
                }
                if (!s.ok()) {
                  LOG(WARNING) << "Async write-through failed after Save: "
                               << s.message();
                } else {
                  LOG(INFO) << "Async write-through succeeded after Save for "
                            << num_blocks << " blocks";
                }
                // Whether or not publishing worked: the save itself succeeded,
                // so the pin it was granted is spent. Holding it on a registry
                // failure would leak one pin per block with nothing to release
                // it.
                if (backend != nullptr) backend->Release(saved);
              });
        }
      } else if (!update_hashes.empty()) {
        // No write-through to outlive, so the release is correct right here.
        backend()->Release(update_hashes);
      }
    } else {
      LOG(ERROR) << "Async Save failed: " << status.ToString();
      DeallocateBlockIds(state.host_block_ids);
      for (const auto& hash : state.block_hashes) {
        failed_saves_.push_back(hash);
      }
    }
    for (const auto& hash : state.block_hashes) {
      saving_hashes_.erase(hash);
    }
  }
}

void KVCacheStore::PollLoadsInternal(std::vector<LoadState> ready_loads) {
  for (auto& state : ready_loads) {
    absl::Status status = state.future.Await();
    absl::MutexLock lock(mutex_);
    if (status.ok() && state.from_remote) {
      // A load from a peer records NOTHING locally. The bytes went to the
      // caller's device blocks and no local host copy was kept, so there is no
      // residency to describe: an entry here would claim HBM with
      // host_block_id -1, which eviction cannot reclaim (it only takes HOST and
      // HOST_AND_HBM) and which nothing left in the API can delete.
      //
      // The consequence is deliberate: a later lookup() of the same hash is a
      // miss, and a repeat request re-fetches unless the caller's own block
      // manager remembers it already owns the device block.
      for (const auto& hash : state.block_hashes) {
        done_loads_.push_back(hash);
      }
    } else if (status.ok()) {
      // Local source: the entry exists here by construction, so this lookup is
      // purely local -- no registry fallback, which would otherwise put a
      // blocking RPC inside the poller while it holds mutex_.
      auto lookup_or = backend()->Lookup(state.block_hashes,
                                         LookupOptions{.enable_global = false});
      if (lookup_or.ok()) {
        const auto& slices = lookup_or.value();
        std::vector<std::string> update_hashes;
        std::vector<RaidenBlockId> update_slices;
        for (size_t i = 0; i < state.block_hashes.size(); ++i) {
          const auto& hash = state.block_hashes[i];
          if (i < slices.size()) {
            RaidenBlockId block = slices[i].second;
            block.device_block_id = state.device_block_ids[i];
            block.status = BlockStatus::HOST_AND_HBM;
            update_hashes.push_back(hash);
            update_slices.push_back(block);
          }
          done_loads_.push_back(hash);
        }
        if (!update_hashes.empty()) {
          backend()->Insert(update_hashes, update_slices, /*on_host=*/true);
          // The load is done with the block, so the pin the caller acquired to
          // keep it alive across the transfer is consumed here. Released AFTER
          // the index update, so the entry cannot be evicted between the two.
          //
          // Only on success, and only for a local source: a failed load stays
          // pinned so the caller can retry or release deliberately, and a
          // remote load never had a caller pin to consume.
          backend()->Release(update_hashes);
        }
      }
    } else {
      LOG(ERROR) << "Async Load failed: " << status.ToString();
      for (const auto& hash : state.block_hashes) {
        failed_loads_.push_back(hash);
      }
    }
    for (const auto& hash : state.block_hashes) {
      loading_hashes_.erase(hash);
    }
  }
}

void KVCacheStore::PollRemoteReadsInternal(
    std::vector<std::pair<tsl::Future<>, RemoteReadState>> ready_remote_reads) {
  for (auto& [future, state] : ready_remote_reads) {
    absl::Status status = future.Await();
    absl::MutexLock lock(mutex_);

    if (status.ok()) {
      // Nothing to record. The bytes are in the caller's device blocks and
      // this store keeps no account of them: no LRU entry, no registry
      // advertisement. Reporting the hashes done is the whole commit.
      for (const auto& hash : state.block_hashes) {
        done_remote_reads_.push_back(hash);
      }
    } else {
      // The caller's device blocks may hold garbage -- by design: nothing
      // points at them, and the caller treats them as scratch until this
      // reports success.
      LOG(WARNING) << "Async ReadRemote failed: " << status.ToString();
      // Drop these peers' cached controller addresses. Most failures are not
      // address failures, and dropping anyway is the point: re-resolving costs
      // one RPC on the next read, whereas keeping an address that moved leaves
      // the peer unreachable until this process dies.
      for (const auto& peer : state.src_raiden_ids) {
        resolved_peer_controllers_.erase(peer);
      }
      for (const auto& hash : state.block_hashes) {
        failed_remote_reads_.push_back(hash);
      }
    }
    // The staging blocks were a hop, not a destination, so they go back to the
    // pool whichever way the read went. Success is not an exception: no LRU
    // entry points at them, so leaking them here would burn a host block per
    // read with nothing able to reclaim it.
    DeallocateBlockIds(state.host_block_ids);
    for (const auto& hash : state.block_hashes) {
      reading_hashes_.erase(hash);
    }
  }
}

void KVCacheStore::PollFuturesInternal() {
  std::vector<SaveState> ready_saves;
  std::vector<LoadState> ready_loads;
  std::vector<std::pair<tsl::Future<>, RemoteReadState>> ready_remote_reads;

  {
    absl::MutexLock lock(mutex_);
    auto it = active_saves_.begin();
    while (it != active_saves_.end()) {
      if (it->future.IsReady()) {
        ready_saves.push_back(std::move(*it));
        it = active_saves_.erase(it);
      } else {
        ++it;
      }
    }

    auto jt = active_loads_.begin();
    while (jt != active_loads_.end()) {
      if (jt->future.IsReady()) {
        ready_loads.push_back(std::move(*jt));
        jt = active_loads_.erase(jt);
      } else {
        ++jt;
      }
    }

    auto kt = active_remote_reads_.begin();
    while (kt != active_remote_reads_.end()) {
      if (kt->first.IsReady()) {
        ready_remote_reads.push_back({kt->first, std::move(kt->second)});
        active_remote_reads_.erase(kt++);
      } else {
        ++kt;
      }
    }
  }

  PollSavesInternal(std::move(ready_saves));
  PollLoadsInternal(std::move(ready_loads));
  PollRemoteReadsInternal(std::move(ready_remote_reads));
}

}  // namespace kv_cache
}  // namespace tpu_raiden
