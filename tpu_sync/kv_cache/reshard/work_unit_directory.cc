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

#include "tpu_sync/kv_cache/reshard/work_unit_directory.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/reshard/declaration_types.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {

absl::Status WorkUnitDirectory::Validate(const WorkUnitRegistration& reg) {
  const bool has_metadata = reg.mesh_shape.has_value() ||
                            reg.layout.has_value() ||
                            reg.global_shape.has_value();
  if (has_metadata) {
    if (!reg.mesh_shape.has_value() || !reg.layout.has_value() ||
        !reg.global_shape.has_value()) {
      return absl::InvalidArgumentError(
          "If any of mesh_shape, layout, or global_shape is provided, "
          "all of them must be provided to enable centralized slice "
          "planning.");
    }
    if (!reg.itemsize.has_value() || *reg.itemsize <= 0) {
      return absl::InvalidArgumentError(
          "itemsize must be provided and must be greater than 0 if "
          "resharding metadata is provided.");
    }
  }

  const bool has_reshard_metadata =
      reg.pool_manifest.has_value() || reg.layout_fingerprint.has_value() ||
      reg.page_tokens.has_value() || reg.transfer_parallelism.has_value() ||
      reg.transfer_rank.has_value();
  if (has_reshard_metadata) {
    if (!reg.pool_manifest.has_value() || !reg.layout_fingerprint.has_value() ||
        !reg.page_tokens.has_value() || !reg.transfer_parallelism.has_value() ||
        !reg.transfer_rank.has_value()) {
      return absl::InvalidArgumentError(
          "pool_manifest, layout_fingerprint, page_tokens, "
          "transfer_parallelism, and transfer_rank must be provided "
          "together for pool resharding");
    }
    if (reg.pool_manifest->empty()) {
      return absl::InvalidArgumentError("pool_manifest must not be empty");
    }
    if (reg.layout_fingerprint->empty()) {
      return absl::InvalidArgumentError("layout_fingerprint must not be empty");
    }
    if (*reg.page_tokens <= 0) {
      return absl::InvalidArgumentError("page_tokens must be positive");
    }
    if (*reg.transfer_parallelism <= 0) {
      return absl::InvalidArgumentError(
          "transfer_parallelism must be positive");
    }
    if (*reg.transfer_rank < 0) {
      return absl::InvalidArgumentError("transfer_rank must be non-negative");
    }
    if (*reg.transfer_rank >= *reg.transfer_parallelism) {
      return absl::InvalidArgumentError(
          "transfer_rank must be less than transfer_parallelism");
    }
  }

  bool any_empty_shard = false;
  for (const std::string& shard : reg.shards) {
    if (shard.empty()) any_empty_shard = true;
  }
  if (reg.shards.empty() || any_empty_shard) {
    return absl::InvalidArgumentError(
        "shards must contain at least one non-empty endpoint");
  }
  return absl::OkStatus();
}

int WorkUnitDirectory::FindLocked(const RaidenId& unit) const {
  for (size_t i = 0; i < order_.size(); ++i) {
    if (order_[i] == unit) return static_cast<int>(i);
  }
  return -1;
}

void WorkUnitDirectory::RegisterLocked(const WorkUnitRegistration& reg) {
  int idx = FindLocked(reg.unit);
  if (idx < 0) {
    order_.push_back(reg.unit);
    records_.emplace_back();
    idx = static_cast<int>(records_.size()) - 1;
  } else {
    // Registration is replacement, not a patch: stale optional metadata
    // must disappear when a unit restarts with a different payload.
    records_[idx] = Record();
  }
  Record& record = records_[idx];
  record.shards = reg.shards;
  record.mesh_shape = reg.mesh_shape;
  record.layout = reg.layout;
  record.global_shape = reg.global_shape;
  record.itemsize = reg.itemsize;
  if (reg.pool_manifest.has_value()) {
    record.pools = reg.pool_manifest;
    record.layout_fingerprint = reg.layout_fingerprint;
    record.page_tokens = reg.page_tokens;
    record.transfer_parallelism = reg.transfer_parallelism;
    record.transfer_rank = reg.transfer_rank;
  }
  record.variables = reg.variables;
  record.control_plane_rpc_address = reg.control_plane_rpc_address.value_or("");
}

absl::StatusOr<tpu_sync::rpc::RegisterWorkUnitRequest>
WorkUnitDirectory::MetadataProtoLocked(const RaidenId& unit) const {
  int idx = FindLocked(unit);
  if (idx < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Work unit is not registered: ", PythonRepr(unit)));
  }
  const Record& record = records_[idx];
  tpu_sync::rpc::RegisterWorkUnitRequest reg_req;
  *reg_req.mutable_unit() = RaidenIdToProto(unit);
  for (const std::string& shard : record.shards) {
    reg_req.add_shards(shard);
  }
  reg_req.set_control_plane_rpc_address(record.control_plane_rpc_address);
  reg_req.set_itemsize(record.itemsize.value_or(0));
  reg_req.set_layout_fingerprint(record.layout_fingerprint.value_or(""));
  reg_req.set_page_tokens(record.page_tokens.value_or(0));
  reg_req.set_transfer_parallelism(record.transfer_parallelism.value_or(0));
  reg_req.set_transfer_rank(record.transfer_rank.value_or(0));
  if (record.mesh_shape.has_value()) {
    for (int64_t v : *record.mesh_shape) reg_req.add_mesh_shape(v);
  }
  if (record.layout.has_value()) {
    for (int32_t v : *record.layout) reg_req.add_layout(v);
  }
  if (record.global_shape.has_value()) {
    for (int64_t v : *record.global_shape) reg_req.add_global_shape(v);
  }
  if (record.pools.has_value()) {
    for (const auto& pool : *record.pools) {
      *reg_req.add_pools() = pool;
    }
  }
  if (record.variables.has_value()) {
    for (const auto& variable : *record.variables) {
      *reg_req.add_variables() = variable;
    }
  }
  return reg_req;
}

std::vector<tpu_sync::rpc::RegisterWorkUnitRequest>
WorkUnitDirectory::AllMetadata() const {
  absl::MutexLock lock(*mu_);
  std::vector<tpu_sync::rpc::RegisterWorkUnitRequest> result;
  result.reserve(order_.size());
  for (const RaidenId& unit : order_) {
    auto proto = MetadataProtoLocked(unit);
    if (proto.ok()) result.push_back(*std::move(proto));
  }
  return result;
}

absl::StatusOr<std::vector<tpu_sync::rpc::RegisterWorkUnitRequest>>
WorkUnitDirectory::LocalMetadata(const std::vector<RaidenId>& units) const {
  absl::MutexLock lock(*mu_);
  std::vector<std::string> missing;
  for (const RaidenId& unit : units) {
    if (FindLocked(unit) < 0) missing.push_back(PythonRepr(unit));
  }
  if (!missing.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Work units are not registered: [", absl::StrJoin(missing, ", "), "]"));
  }
  std::vector<tpu_sync::rpc::RegisterWorkUnitRequest> result;
  result.reserve(units.size());
  for (const RaidenId& unit : units) {
    auto proto = MetadataProtoLocked(unit);
    if (!proto.ok()) return proto.status();
    result.push_back(*std::move(proto));
  }
  return result;
}

bool WorkUnitDirectory::IsRegisteredLocked(const RaidenId& unit) const {
  return FindLocked(unit) >= 0;
}

std::vector<tpu_sync::rpc::PoolSpecProto> WorkUnitDirectory::PoolManifestLocked(
    const RaidenId& unit) const {
  int idx = FindLocked(unit);
  if (idx < 0 || !records_[idx].pools.has_value()) return {};
  return *records_[idx].pools;
}

}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden
