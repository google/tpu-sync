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

#include "tpu_sync/weight_sync/manager/v3/entity_registry.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/manager/v3/logical_types.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {
namespace {

// Returns the host of a `host:port` endpoint. IPv6 hosts are bracketed
// (`[2001:db8::1]:8000`) and returned without the brackets.
std::string ExtractEndpointIp(absl::string_view endpoint) {
  if (absl::ConsumePrefix(&endpoint, "[")) {
    const size_t close = endpoint.find(']');
    if (close != absl::string_view::npos) {
      return std::string(endpoint.substr(0, close));
    }
  }
  auto pos = endpoint.rfind(':');
  if (pos == absl::string_view::npos) {
    return std::string(endpoint);
  }
  return std::string(endpoint.substr(0, pos));
}

std::vector<std::string> SplitEndpoints(absl::string_view raw) {
  std::vector<std::string> out;
  for (absl::string_view part : absl::StrSplit(raw, ',', absl::SkipEmpty())) {
    absl::string_view trimmed = absl::StripAsciiWhitespace(part);
    if (!trimmed.empty()) {
      out.emplace_back(trimmed);
    }
  }
  return out;
}

absl::Status UnitNotRegisteredError(const RaidenId& unit) {
  return absl::NotFoundError(
      absl::StrCat("Work unit is not registered: ", unit.job_name, "/",
                   unit.job_replica_id));
}

// Number of global shards of the first variable of |ent|: the product of its
// mesh_shape, widened to cover its global_shard_indices (1 without
// variables).
int64_t GlobalShardCount(const WorkUnitEntity& ent) {
  if (ent.variables.empty()) return 1;
  const VariableSpec& v = ent.variables[0];
  int64_t count = 1;
  for (int64_t d : v.mesh_shape) {
    if (d > 0) count *= d;
  }
  for (int64_t g : v.global_shard_indices) count = std::max(count, g + 1);
  return count;
}

}  // namespace

RaidenId RaidenIdFromProto(const tpu_sync::rpc::RaidenIdProto& proto) {
  return RaidenId{
      .job_name = proto.job_name(),
      .job_replica_id = proto.job_replica_id(),
      .data_name = proto.data_name(),
      .data_replica_idx = proto.data_replica_idx(),
  };
}

tpu_sync::rpc::RaidenIdProto RaidenIdToProto(const RaidenId& unit) {
  tpu_sync::rpc::RaidenIdProto proto;
  proto.set_job_name(unit.job_name);
  proto.set_job_replica_id(unit.job_replica_id);
  proto.set_data_name(unit.data_name);
  proto.set_data_replica_idx(unit.data_replica_idx);
  return proto;
}

tpu_sync::rpc::RegisterWorkUnitRequest WorkUnitEntity::ToMetadataProto() const {
  tpu_sync::rpc::RegisterWorkUnitRequest req;
  *req.mutable_unit() = RaidenIdToProto(unit);
  for (const std::string& s : shards) {
    req.add_shards(s);
  }
  if (!control_endpoints.empty()) {
    req.set_control_plane_rpc_address(absl::StrJoin(control_endpoints, ","));
  }
  for (const VariableSpec& var : variables) {
    auto* vp = req.add_variables();
    vp->set_name(var.name);
    for (int64_t d : var.global_shape) vp->add_shape(d);
    for (int64_t m : var.mesh_shape) vp->add_mesh_shape(m);
    for (int64_t l : var.layout) vp->add_layout(static_cast<int32_t>(l));
    vp->set_item_size(static_cast<int32_t>(var.itemsize));
    vp->set_layer_idx(var.layer_idx);
    for (int64_t g : var.global_shard_indices) vp->add_global_shard_indices(g);
  }
  return req;
}

void EntityRegistry::RebuildHostAttachments(WorkUnitEntity* entity) {
  entity->hosts.clear();
  const int32_t num_shards = static_cast<int32_t>(entity->shards.size());
  if (num_shards == 0) return;

  if (entity->control_endpoints.size() <= 1) {
    HostAttachment h;
    h.unit = entity->unit;
    h.host_idx = 0;
    if (!entity->control_endpoints.empty()) {
      h.control_address = entity->control_endpoints[0];
    }
    h.owned_global_shard_indices.resize(num_shards);
    h.unit_local_shard_indices.resize(num_shards);
    for (int32_t i = 0; i < num_shards; ++i) {
      h.owned_global_shard_indices[i] = i;
      h.unit_local_shard_indices[i] = i;
    }
    h.data_endpoints = entity->shards;
    entity->hosts.push_back(std::move(h));
    return;
  }

  // Multi-host work unit: match control endpoints to shard data endpoints by
  // distinct data endpoint order or IP address, falling back to contiguous
  // partition across hosts.
  std::vector<std::string> distinct_data;
  for (const std::string& s : entity->shards) {
    if (std::find(distinct_data.begin(), distinct_data.end(), s) ==
        distinct_data.end()) {
      distinct_data.push_back(s);
    }
  }

  const size_t num_hosts = entity->control_endpoints.size();
  if (distinct_data.size() == num_hosts) {
    for (size_t h_idx = 0; h_idx < num_hosts; ++h_idx) {
      HostAttachment h;
      h.unit = entity->unit;
      h.host_idx = static_cast<int32_t>(h_idx);
      h.control_address = entity->control_endpoints[h_idx];
      const std::string& target_data = distinct_data[h_idx];
      for (int32_t s_idx = 0; s_idx < num_shards; ++s_idx) {
        if (entity->shards[s_idx] == target_data) {
          h.owned_global_shard_indices.push_back(s_idx);
          h.unit_local_shard_indices.push_back(s_idx);
          h.data_endpoints.push_back(entity->shards[s_idx]);
        }
      }
      entity->hosts.push_back(std::move(h));
    }
    return;
  }

  // Try IP-based matching if distinct IPs match.
  absl::flat_hash_map<std::string, std::vector<int32_t>> shards_by_ip;
  for (int32_t s_idx = 0; s_idx < num_shards; ++s_idx) {
    shards_by_ip[ExtractEndpointIp(entity->shards[s_idx])].push_back(s_idx);
  }
  bool all_ips_found = true;
  for (const std::string& ctrl : entity->control_endpoints) {
    if (!shards_by_ip.contains(ExtractEndpointIp(ctrl))) {
      all_ips_found = false;
      break;
    }
  }
  if (all_ips_found && shards_by_ip.size() == num_hosts) {
    for (size_t h_idx = 0; h_idx < num_hosts; ++h_idx) {
      HostAttachment h;
      h.unit = entity->unit;
      h.host_idx = static_cast<int32_t>(h_idx);
      h.control_address = entity->control_endpoints[h_idx];
      const auto& owned = shards_by_ip[ExtractEndpointIp(h.control_address)];
      h.owned_global_shard_indices.reserve(owned.size());
      h.unit_local_shard_indices.reserve(owned.size());
      h.data_endpoints.reserve(owned.size());
      for (int32_t s_idx : owned) {
        h.owned_global_shard_indices.push_back(s_idx);
        h.unit_local_shard_indices.push_back(s_idx);
        h.data_endpoints.push_back(entity->shards[s_idx]);
      }
      entity->hosts.push_back(std::move(h));
    }
    return;
  }

  // Contiguous even partition across hosts.
  const int32_t base = num_shards / static_cast<int32_t>(num_hosts);
  const int32_t rem = num_shards % static_cast<int32_t>(num_hosts);
  int32_t cur = 0;
  for (size_t h_idx = 0; h_idx < num_hosts; ++h_idx) {
    HostAttachment h;
    h.unit = entity->unit;
    h.host_idx = static_cast<int32_t>(h_idx);
    h.control_address = entity->control_endpoints[h_idx];
    const int32_t count = base + (static_cast<int32_t>(h_idx) < rem ? 1 : 0);
    for (int32_t k = 0; k < count && cur < num_shards; ++k, ++cur) {
      h.owned_global_shard_indices.push_back(cur);
      h.unit_local_shard_indices.push_back(cur);
      h.data_endpoints.push_back(entity->shards[cur]);
    }
    entity->hosts.push_back(std::move(h));
  }
}

absl::StatusOr<bool> EntityRegistry::RegisterWorkUnit(
    const tpu_sync::rpc::RegisterWorkUnitRequest& req) {
  if (req.shards_size() == 0) {
    return absl::InvalidArgumentError(
        "shards must contain at least one non-empty endpoint");
  }
  for (const std::string& s : req.shards()) {
    if (s.empty()) {
      return absl::InvalidArgumentError(
          "shards must contain at least one non-empty endpoint");
    }
  }

  if (req.variables_size() == 0 &&
      (req.global_shape_size() > 0 || req.layout_size() > 0 ||
       req.mesh_shape_size() > 0)) {
    return absl::InvalidArgumentError(
        "Unit-level global_shape/layout/mesh_shape are not supported; describe "
        "each weight as an entry of `variables` with shape, mesh_shape, "
        "layout, item_size and global_shard_indices.");
  }

  WorkUnitEntity next;
  next.unit = RaidenIdFromProto(req.unit());
  next.shards.assign(req.shards().begin(), req.shards().end());
  next.control_endpoints = SplitEndpoints(req.control_plane_rpc_address());

  next.variables.reserve(req.variables_size());
  for (int i = 0; i < req.variables_size(); ++i) {
    const tpu_sync::rpc::VariableMetadataProto& vp = req.variables(i);
    if (vp.global_shard_indices_size() != req.shards_size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Variable '", vp.name(), "' (index ", i, ") of work unit ",
          next.unit.job_name, "/", next.unit.job_replica_id, " has ",
          vp.global_shard_indices_size(), " global_shard_indices for ",
          req.shards_size(),
          " shards. global_shard_indices is required on every variable: "
          "entry i is the global shard index, in [0, prod(mesh_shape)), held "
          "by shards[i]. The deprecated mesh_axes, host_subgrid, unit-level "
          "mesh_shape and sharding_spec are not used to derive it."));
    }
    VariableSpec vs;
    vs.name = vp.name();
    vs.global_shape.assign(vp.shape().begin(), vp.shape().end());
    vs.mesh_shape.assign(vp.mesh_shape().begin(), vp.mesh_shape().end());
    vs.layout.assign(vp.layout().begin(), vp.layout().end());
    vs.itemsize = vp.item_size() > 0 ? vp.item_size() : req.itemsize();
    vs.layer_idx = vp.layer_idx();
    vs.global_shard_indices.assign(vp.global_shard_indices().begin(),
                                   vp.global_shard_indices().end());
    next.variables.push_back(std::move(vs));
  }
  RebuildHostAttachments(&next);

  absl::MutexLock lock(mu_);
  auto it = entities_.find(next.unit);
  bool topology_changed = true;
  if (it != entities_.end()) {
    const WorkUnitEntity& prev = it->second;
    topology_changed = !(prev.shards == next.shards &&
                         prev.control_endpoints == next.control_endpoints &&
                         prev.variables == next.variables);
    it->second = std::move(next);
  } else {
    registration_order_.push_back(next.unit);
    entities_.emplace(next.unit, std::move(next));
  }
  return topology_changed;
}

absl::Status EntityRegistry::AttachHost(
    const RaidenId& unit, absl::string_view control_address,
    absl::Span<const std::string> host_shards) {
  absl::MutexLock lock(mu_);
  auto it = entities_.find(unit);
  if (it == entities_.end()) {
    WorkUnitEntity ent;
    ent.unit = unit;
    ent.shards.assign(host_shards.begin(), host_shards.end());
    ent.control_endpoints = SplitEndpoints(control_address);
    RebuildHostAttachments(&ent);
    registration_order_.push_back(unit);
    entities_.emplace(unit, std::move(ent));
    return absl::OkStatus();
  }
  WorkUnitEntity& ent = it->second;
  for (const std::string& addr : SplitEndpoints(control_address)) {
    if (std::find(ent.control_endpoints.begin(), ent.control_endpoints.end(),
                  addr) == ent.control_endpoints.end()) {
      ent.control_endpoints.push_back(addr);
    }
  }
  for (const std::string& s : host_shards) {
    if (!s.empty() && std::find(ent.shards.begin(), ent.shards.end(), s) ==
                          ent.shards.end()) {
      ent.shards.push_back(s);
    }
  }
  RebuildHostAttachments(&ent);
  return absl::OkStatus();
}

absl::StatusOr<WorkUnitEntity> EntityRegistry::GetEntity(
    const RaidenId& unit) const {
  absl::MutexLock lock(mu_);
  auto it = entities_.find(unit);
  if (it == entities_.end()) return UnitNotRegisteredError(unit);
  return it->second;
}

absl::StatusOr<std::vector<WorkUnitEntity>> EntityRegistry::GetEntities(
    absl::Span<const RaidenId> units) const {
  std::vector<WorkUnitEntity> out;
  out.reserve(units.size());
  absl::MutexLock lock(mu_);
  for (const RaidenId& unit : units) {
    auto it = entities_.find(unit);
    if (it == entities_.end()) return UnitNotRegisteredError(unit);
    out.push_back(it->second);
  }
  return out;
}

absl::StatusOr<WorkUnitEntity> EntityRegistry::BuildCompositeEntity(
    absl::Span<const RaidenId> units) const {
  if (units.empty()) {
    return absl::InvalidArgumentError("units must not be empty");
  }
  // One snapshot of all units, so the composite never mixes registry states.
  absl::StatusOr<std::vector<WorkUnitEntity>> snapshot = GetEntities(units);
  if (!snapshot.ok()) return snapshot.status();
  std::vector<WorkUnitEntity> parts = *std::move(snapshot);
  if (parts.size() == 1) {
    return std::move(parts[0]);
  }

  // If all constituent units have distinct numeric `job_replica_id`s (e.g.,
  // host indices "0", "1", "2", "3" registered in arbitrary pod startup order),
  // sort `parts` by numeric `job_replica_id` so host_idx aligns across all
  // replicas.
  bool all_numeric_distinct = true;
  absl::flat_hash_set<int64_t> seen_ids;
  for (const WorkUnitEntity& part : parts) {
    int64_t parsed_id = 0;
    if (!absl::SimpleAtoi(part.unit.job_replica_id, &parsed_id) ||
        !seen_ids.insert(parsed_id).second) {
      all_numeric_distinct = false;
      break;
    }
  }
  if (all_numeric_distinct) {
    std::sort(parts.begin(), parts.end(),
              [](const WorkUnitEntity& a, const WorkUnitEntity& b) {
                int64_t id_a = 0;
                int64_t id_b = 0;
                (void)absl::SimpleAtoi(a.unit.job_replica_id, &id_a);
                (void)absl::SimpleAtoi(b.unit.job_replica_id, &id_b);
                return id_a < id_b;
              });
  }

  WorkUnitEntity composite = parts[0];
  composite.shards.clear();
  composite.control_endpoints.clear();
  composite.hosts.clear();
  composite.variables = parts[0].variables;

  for (const WorkUnitEntity& part : parts) {
    const int32_t shard_offset = static_cast<int32_t>(composite.shards.size());
    composite.shards.insert(composite.shards.end(), part.shards.begin(),
                            part.shards.end());
    composite.control_endpoints.insert(composite.control_endpoints.end(),
                                       part.control_endpoints.begin(),
                                       part.control_endpoints.end());
    for (const HostAttachment& h : part.hosts) {
      HostAttachment ch = h;
      ch.unit = part.unit;
      ch.host_idx = static_cast<int32_t>(composite.hosts.size());
      ch.unit_local_shard_indices = h.owned_global_shard_indices;
      for (int32_t& g_idx : ch.owned_global_shard_indices) {
        g_idx += shard_offset;
      }
      composite.hosts.push_back(std::move(ch));
    }
  }

  // The composite's local shard order is the concatenation of the parts', so
  // its global_shard_indices are the concatenation of theirs.
  for (size_t l = 0; l < composite.variables.size(); ++l) {
    VariableSpec& cv = composite.variables[l];
    cv.global_shard_indices.clear();
    for (const WorkUnitEntity& part : parts) {
      if (l >= part.variables.size() ||
          part.variables[l].global_shard_indices.size() != part.shards.size()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Work unit ", part.unit.job_name, "/", part.unit.job_replica_id,
            " does not have one global_shard_indices entry per shard for "
            "variable '",
            cv.name, "'"));
      }
      const std::vector<int64_t>& part_gsi =
          part.variables[l].global_shard_indices;
      cv.global_shard_indices.insert(cv.global_shard_indices.end(),
                                     part_gsi.begin(), part_gsi.end());
    }
    // Per-host registrations of a 1-D sharded weight describe only their own
    // slice; widen the mesh and shape to cover every reported global index.
    if (cv.mesh_shape.size() == 1 && cv.global_shape.size() == 1 &&
        !parts[0].shards.empty() &&
        cv.mesh_shape[0] == static_cast<int64_t>(parts[0].shards.size()) &&
        cv.global_shape[0] % cv.mesh_shape[0] == 0) {
      int64_t max_gsi = -1;
      for (int64_t g : cv.global_shard_indices) {
        max_gsi = std::max(max_gsi, g);
      }
      if (max_gsi + 1 > cv.mesh_shape[0]) {
        const int64_t elems_per_shard = cv.global_shape[0] / cv.mesh_shape[0];
        const int64_t full_shards = std::max<int64_t>(
            static_cast<int64_t>(composite.shards.size()), max_gsi + 1);
        cv.mesh_shape[0] = full_shards;
        cv.global_shape[0] = elems_per_shard * full_shards;
      }
    }
  }

  return composite;
}

absl::StatusOr<std::vector<WorkUnitEntity>>
EntityRegistry::BuildDestinationReplicaEntities(
    absl::Span<const RaidenId> dst_units) const {
  if (dst_units.empty()) {
    return absl::InvalidArgumentError("dst_units must not be empty");
  }
  absl::StatusOr<std::vector<WorkUnitEntity>> snapshot = GetEntities(dst_units);
  if (!snapshot.ok()) return snapshot.status();
  std::vector<WorkUnitEntity> raw_entities = *std::move(snapshot);
  bool has_partial_host_units = false;
  for (const WorkUnitEntity& ent : raw_entities) {
    if (!ent.variables.empty() && !ent.shards.empty() &&
        GlobalShardCount(ent) > static_cast<int64_t>(ent.shards.size()) &&
        dst_units.size() > 1) {
      has_partial_host_units = true;
    }
  }

  if (!has_partial_host_units) {
    return raw_entities;
  }

  // Group per-host destination units into full destination replicas by
  // `(job_name, data_name, data_replica_idx)`, preserving first-seen replica
  // order, and chunking by `hosts_per_replica` when multiple replicas share a
  // `job_name`.
  std::vector<std::string> group_keys;
  absl::flat_hash_map<std::string, std::vector<RaidenId>> groups;
  absl::flat_hash_map<std::string, int64_t> group_mesh_prod;
  absl::flat_hash_map<std::string, size_t> group_n_local;

  for (size_t i = 0; i < dst_units.size(); ++i) {
    const RaidenId& u = dst_units[i];
    const WorkUnitEntity& ent = raw_entities[i];
    const std::string key =
        absl::StrCat(u.job_name, "|", u.data_name, "|", u.data_replica_idx);
    std::vector<RaidenId>& group = groups[key];
    if (group.empty()) {
      group_keys.push_back(key);
      group_n_local[key] = std::max<size_t>(1, ent.shards.size());
      group_mesh_prod[key] = 1;
    }
    group_mesh_prod[key] =
        std::max(group_mesh_prod[key], GlobalShardCount(ent));
    group.push_back(u);
  }

  absl::flat_hash_map<std::string, size_t> group_hosts_per_replica;
  for (const std::string& key : group_keys) {
    const int64_t mesh_prod = group_mesh_prod[key];
    const size_t n_local = group_n_local[key];
    const size_t h_per_rep = (mesh_prod > static_cast<int64_t>(n_local))
                                 ? static_cast<size_t>(mesh_prod) / n_local
                                 : 1;
    group_hosts_per_replica[key] = std::max<size_t>(1, h_per_rep);
  }

  std::vector<WorkUnitEntity> replica_entities;
  for (const std::string& key : group_keys) {
    const std::vector<RaidenId>& unit_list = groups[key];
    const size_t h_per_rep = group_hosts_per_replica[key];
    if (h_per_rep <= 1 || unit_list.size() <= h_per_rep) {
      auto comp = BuildCompositeEntity(unit_list);
      if (!comp.ok()) return comp.status();
      replica_entities.push_back(std::move(*comp));
    } else {
      for (size_t offset = 0; offset < unit_list.size(); offset += h_per_rep) {
        const size_t count = std::min(h_per_rep, unit_list.size() - offset);
        absl::Span<const RaidenId> slice(&unit_list[offset], count);
        auto comp = BuildCompositeEntity(slice);
        if (!comp.ok()) return comp.status();
        replica_entities.push_back(std::move(*comp));
      }
    }
  }
  return replica_entities;
}

bool EntityRegistry::HasUnit(const RaidenId& unit) const {
  absl::MutexLock lock(mu_);
  return entities_.contains(unit);
}

std::vector<RaidenId> EntityRegistry::GetRegisteredUnits() const {
  absl::MutexLock lock(mu_);
  return registration_order_;
}

std::vector<tpu_sync::rpc::RegisterWorkUnitRequest>
EntityRegistry::GetAllMetadata() const {
  absl::MutexLock lock(mu_);
  std::vector<tpu_sync::rpc::RegisterWorkUnitRequest> out;
  out.reserve(registration_order_.size());
  for (const RaidenId& u : registration_order_) {
    auto it = entities_.find(u);
    if (it != entities_.end()) {
      out.push_back(it->second.ToMetadataProto());
    }
  }
  return out;
}

}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden
