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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_ENTITY_REGISTRY_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_ENTITY_REGISTRY_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/manager/v3/logical_types.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {

// Describes a single host belonging to a multi-host or single-host work unit.
struct HostAttachment {
  RaidenId unit;
  int32_t host_idx = 0;
  std::string control_address;
  // Indices into the owning entity's `shards` (for a composite entity, into the
  // concatenated shards of all parts).
  std::vector<int32_t> owned_global_shard_indices;
  // The same shards as indices into the shards of `unit`, the work unit whose
  // worker this host runs: commands sent to that worker address them this way,
  // so they stay relative to the part even inside a composite entity.
  std::vector<int32_t> unit_local_shard_indices;
  std::vector<std::string> data_endpoints;
};

// Represents a registered Trainer or Sampler work unit replica and its
// physical host attachments and logical variable metadata.
struct WorkUnitEntity {
  RaidenId unit;
  std::vector<std::string> shards;
  std::vector<std::string> control_endpoints;
  std::vector<HostAttachment> hosts;
  // Every variable carries one global shard index per entry of `shards`.
  std::vector<VariableSpec> variables;

  int32_t NumHosts() const {
    if (!hosts.empty()) {
      return static_cast<int32_t>(hosts.size());
    }
    if (!control_endpoints.empty()) {
      return static_cast<int32_t>(control_endpoints.size());
    }
    return 1;
  }

  tpu_sync::rpc::RegisterWorkUnitRequest ToMetadataProto() const;
};

// Thread-safe registry managing registered Trainer and Sampler work units,
// host control/data endpoints, and global-to-local shard resolution.
class EntityRegistry {
 public:
  EntityRegistry() = default;

  // Validates and registers (or updates) |req|. Every variable must carry
  // `global_shard_indices` with one entry per shard (the global shard index,
  // in [0, prod(mesh_shape)), held by that shard); otherwise InvalidArgument.
  // Unit-level global_shape/layout/mesh_shape without variables are rejected,
  // and the deprecated mesh_axes, host_subgrid and sharding_spec are ignored.
  // Returns true if the unit's shards, control endpoints or variables changed
  // (requiring plan cache invalidation), false for an identical registration.
  absl::StatusOr<bool> RegisterWorkUnit(
      const tpu_sync::rpc::RegisterWorkUnitRequest& req);

  // Attaches or updates a host endpoint and its owned shards on |unit|.
  absl::Status AttachHost(const RaidenId& unit,
                          absl::string_view control_address,
                          absl::Span<const std::string> host_shards);

  // Returns a snapshot of the `WorkUnitEntity` registered for |unit|.
  absl::StatusOr<WorkUnitEntity> GetEntity(const RaidenId& unit) const;

  // Builds a composite `WorkUnitEntity` across one or more |units| (e.g. when a
  // multi-host trainer is registered as one `RaidenId` per host).
  absl::StatusOr<WorkUnitEntity> BuildCompositeEntity(
      absl::Span<const RaidenId> units) const;

  // Resolves |dst_units| into destination replica entities, automatically
  // grouping per-host destination units that belong to the same multi-host
  // replica mesh into composite replica entities.
  absl::StatusOr<std::vector<WorkUnitEntity>> BuildDestinationReplicaEntities(
      absl::Span<const RaidenId> dst_units) const;

  // Returns true if |unit| is currently registered.
  bool HasUnit(const RaidenId& unit) const;

  // Returns all registered work unit IDs in registration order.
  std::vector<RaidenId> GetRegisteredUnits() const;

  // Returns `RegisterWorkUnitRequest` metadata protos for all registered work
  // units in registration order.
  std::vector<tpu_sync::rpc::RegisterWorkUnitRequest> GetAllMetadata() const;

  // Recomputes host attachments for |entity| from its `shards` and
  // `control_endpoints`.
  static void RebuildHostAttachments(WorkUnitEntity* entity);

 private:
  // Returns the entities of |units|, all read under one lock so they are
  // consistent with each other.
  absl::StatusOr<std::vector<WorkUnitEntity>> GetEntities(
      absl::Span<const RaidenId> units) const;

  mutable absl::Mutex mu_;
  std::vector<RaidenId> registration_order_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<RaidenId, WorkUnitEntity, RaidenIdHash> entities_
      ABSL_GUARDED_BY(mu_);
};

// Converts between `RaidenId` and `tpu_sync::rpc::RaidenIdProto`.
RaidenId RaidenIdFromProto(const tpu_sync::rpc::RaidenIdProto& proto);
tpu_sync::rpc::RaidenIdProto RaidenIdToProto(const RaidenId& unit);

}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_ENTITY_REGISTRY_H_
