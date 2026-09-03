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

// Byte-exact port of rpc/raiden_controller.py::_build_byte_span_plan_claimed.
// Validation order, emission order, and error strings mirror the Python
// reference; gate G1 diffs the resulting wire bytes against it.

#include "tpu_sync/kv_cache/reshard/pool_reshard_planner.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/kv_cache/reshard/declaration_types.h"
#include "tpu_sync/kv_cache/reshard/request_block_registry.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {
namespace {

constexpr int64_t kMaxLiveSegments = int64_t{1} << 20;

// Python repr of a str, adequate for pool tags (no quotes/escapes inside).
std::string PyStrRepr(const std::string& s) {
  return absl::StrCat("'", s, "'");
}

std::string PyStrListRepr(const std::vector<std::string>& values) {
  std::vector<std::string> reprs;
  reprs.reserve(values.size());
  for (const std::string& value : values) reprs.push_back(PyStrRepr(value));
  return absl::StrCat("[", absl::StrJoin(reprs, ", "), "]");
}

std::string PySortedIntSetRepr(const std::set<int64_t>& values) {
  return absl::StrCat("[", absl::StrJoin(values, ", "), "]");
}

// _pool_geometry_signature: the registration fields defining one pool's
// byte geometry, as a canonical comparable string.
std::string GeometrySignature(const tpu_sync::rpc::PoolSpecProto& pool) {
  std::string sig =
      absl::StrCat(pool.tag(), "\x1f", pool.storage_index(), "\x1f",
                   pool.base_offset_bytes(), "\x1f", pool.block_stride_bytes(),
                   "\x1f", pool.num_blocks(), "\x1f", pool.dtype_tag());
  for (const auto& region : pool.regions()) {
    absl::StrAppend(&sig, "\x1e", region.name(), "\x1f", region.offset_bytes(),
                    "\x1f", region.stride_bytes(), "\x1f", region.unit_bytes(),
                    "\x1f", region.num_units(), "\x1f",
                    region.units_per_stride());
  }
  return sig;
}

// _pool_live_segments via the shared C++ leaf primitive.
absl::StatusOr<std::vector<PoolLiveSegment>> LiveSegments(
    const tpu_sync::rpc::PoolSpecProto& pool_proto) {
  auto pool = PoolSpecFromProto(pool_proto);
  if (!pool.ok()) return pool.status();
  return ExpandPoolLiveSegments(*pool);
}

// _metadata_by_unit: selects exact requested metadata, rejecting
// duplicates and reporting the missing set in request order.
absl::StatusOr<std::map<RaidenId, const tpu_sync::rpc::RegisterWorkUnitRequest*,
                        RequestBlockRegistry::RaidenIdLess>>
MetadataByUnit(
    const std::vector<tpu_sync::rpc::RegisterWorkUnitRequest>& metadata,
    const std::vector<RaidenId>& units) {
  std::set<RaidenId, RequestBlockRegistry::RaidenIdLess> requested(
      units.begin(), units.end());
  std::map<RaidenId, const tpu_sync::rpc::RegisterWorkUnitRequest*,
           RequestBlockRegistry::RaidenIdLess>
      result;
  for (const auto& item : metadata) {
    RaidenId unit = RaidenIdFromProto(item.unit());
    if (requested.find(unit) == requested.end()) continue;
    if (result.find(unit) != result.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Duplicate registration metadata for ", PythonRepr(unit)));
    }
    result.emplace(unit, &item);
  }
  std::vector<std::string> missing;
  for (const RaidenId& unit : units) {
    if (result.find(unit) == result.end()) {
      missing.push_back(PythonRepr(unit));
    }
  }
  if (!missing.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Missing registration metadata for [",
                     absl::StrJoin(missing, ", "), "]"));
  }
  return result;
}

struct TagPrecheck {
  std::vector<int32_t> selected;  // destination pool indices
  // The source unit whose pools define the tag's source geometry, and its
  // pool indices for the tag (aligned 1:1 with `selected`).
  RaidenId src_reference_unit;
  std::vector<int32_t> src_selected;
  int64_t src_live = 0;
  int64_t dst_live = 0;
  std::vector<PoolLiveSegment> src_segments;
  std::vector<PoolLiveSegment> dst_segments;
};

}  // namespace

absl::StatusOr<PoolReshardPlan> BuildPoolReshardPlan(
    const PlanRequest& request, RequestBlockRegistry* registry,
    const void* claim_owner) {
  const std::string& req_id = request.req_id;
  const int64_t uuid = request.uuid;

  if (req_id.empty()) {
    return absl::InvalidArgumentError(
        "req_id must not be empty for pool resharding");
  }
  if (uuid <= 0) {
    return absl::InvalidArgumentError(
        "uuid must be positive for pool resharding");
  }
  if (request.dst_units.empty()) {
    return absl::InvalidArgumentError(
        "Pool resharding requires at least one destination unit");
  }
  {
    std::set<RaidenId, RequestBlockRegistry::RaidenIdLess> unique_src(
        request.src_units.begin(), request.src_units.end());
    if (unique_src.size() != request.src_units.size()) {
      return absl::InvalidArgumentError(
          "src_units must not contain duplicates");
    }
    std::set<RaidenId, RequestBlockRegistry::RaidenIdLess> unique_dst(
        request.dst_units.begin(), request.dst_units.end());
    if (unique_dst.size() != request.dst_units.size()) {
      return absl::InvalidArgumentError(
          "dst_units must not contain duplicates");
    }
  }

  const std::vector<int64_t>& dst_ids = request.dst_device_block_ids;
  if (dst_ids.empty()) {
    return absl::InvalidArgumentError("dst_device_block_ids must not be empty");
  }
  for (int64_t block_id : dst_ids) {
    if (block_id < 0) {
      return absl::InvalidArgumentError(
          "dst_device_block_ids must be non-negative");
    }
  }

  auto src_by_unit_or = MetadataByUnit(request.src_metadata, request.src_units);
  if (!src_by_unit_or.ok()) return src_by_unit_or.status();
  auto& src_by_unit = *src_by_unit_or;
  auto dst_by_unit_or = MetadataByUnit(request.dst_metadata, request.dst_units);
  if (!dst_by_unit_or.ok()) return dst_by_unit_or.status();
  auto& dst_by_unit = *dst_by_unit_or;
  // All destinations must share one local pool geometry (validated below).
  const tpu_sync::rpc::RegisterWorkUnitRequest& dst_meta =
      *dst_by_unit.at(request.dst_units[0]);

  std::vector<const tpu_sync::rpc::RegisterWorkUnitRequest*> all_metadata;
  all_metadata.reserve(request.src_units.size() + request.dst_units.size());
  for (const RaidenId& unit : request.src_units) {
    all_metadata.push_back(src_by_unit.at(unit));
  }
  for (const RaidenId& unit : request.dst_units) {
    all_metadata.push_back(dst_by_unit.at(unit));
  }
  for (const auto* meta : all_metadata) {
    RaidenId unit = RaidenIdFromProto(meta->unit());
    if (meta->layout_fingerprint().empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Missing layout fingerprint for ", PythonRepr(unit)));
    }
    if (meta->pools_size() == 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Missing explicit pool manifest for ", PythonRepr(unit)));
    }
    if (meta->shards_size() == 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Missing data-plane endpoint for ", PythonRepr(unit)));
    }
    if (meta->shards_size() != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Pool reshard planning requires one data-plane endpoint per work "
          "unit; ",
          PythonRepr(unit), " registered ", meta->shards_size()));
    }
    if (meta->control_plane_rpc_address().empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Missing control-plane endpoint for ", PythonRepr(unit)));
    }
  }

  {
    std::set<std::string> fingerprints;
    for (const auto* meta : all_metadata) {
      fingerprints.insert(meta->layout_fingerprint());
    }
    if (fingerprints.size() != 1) {
      return absl::InvalidArgumentError(
          "Layout fingerprint mismatch between source and destination");
    }
  }

  std::vector<std::pair<std::string, std::string>> dst_identity;
  dst_identity.reserve(dst_meta.pools().size());
  for (const auto& pool : dst_meta.pools()) {
    dst_identity.emplace_back(pool.tag(), pool.dtype_tag());
  }
  if (dst_identity.empty()) {
    return absl::InvalidArgumentError(
        "Destination pool manifest must not be empty");
  }
  // Pools pair up by tag. A source unit may register a subset of the
  // destination's tags (a pipeline stage holds a layer subset), but every
  // tag it registers must carry the destination's pool count and dtype for
  // that tag, in manifest order; a source tag the destination lacks is a
  // mismatch.
  std::map<std::string, std::vector<int32_t>> dst_pools_by_tag;
  for (int i = 0; i < dst_meta.pools_size(); ++i) {
    dst_pools_by_tag[dst_meta.pools(i).tag()].push_back(i);
  }
  std::map<RaidenId, std::map<std::string, std::vector<int32_t>>,
           RequestBlockRegistry::RaidenIdLess>
      src_pools_by_tag;
  for (const RaidenId& src_unit : request.src_units) {
    const auto& src_pools = src_by_unit.at(src_unit)->pools();
    auto& by_tag = src_pools_by_tag[src_unit];
    for (int i = 0; i < src_pools.size(); ++i) {
      by_tag[src_pools.Get(i).tag()].push_back(i);
    }
    for (const auto& [tag, src_indices] : by_tag) {
      auto dst_it = dst_pools_by_tag.find(tag);
      bool matches = dst_it != dst_pools_by_tag.end() &&
                     dst_it->second.size() == src_indices.size();
      for (size_t k = 0; matches && k < src_indices.size(); ++k) {
        matches = src_pools.Get(src_indices[k]).dtype_tag() ==
                  dst_meta.pools(dst_it->second[k]).dtype_tag();
      }
      if (!matches) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Canonical pool manifest mismatch between source and destination "
            "for ",
            PythonRepr(src_unit), " at tag ", PyStrRepr(tag)));
      }
    }
  }
  {
    std::vector<std::string> reference_dst_geometry;
    reference_dst_geometry.reserve(dst_meta.pools().size());
    for (const auto& pool : dst_meta.pools()) {
      reference_dst_geometry.push_back(GeometrySignature(pool));
    }
    for (size_t i = 1; i < request.dst_units.size(); ++i) {
      const RaidenId& unit = request.dst_units[i];
      const auto& other_meta = *dst_by_unit.at(unit);
      std::vector<std::pair<std::string, std::string>> other_identity;
      other_identity.reserve(other_meta.pools().size());
      for (const auto& pool : other_meta.pools()) {
        other_identity.emplace_back(pool.tag(), pool.dtype_tag());
      }
      if (other_identity != dst_identity) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Canonical pool manifest mismatch between destinations at ",
            PythonRepr(unit)));
      }
      std::vector<std::string> other_geometry;
      other_geometry.reserve(other_meta.pools().size());
      for (const auto& pool : other_meta.pools()) {
        other_geometry.push_back(GeometrySignature(pool));
      }
      if (other_geometry != reference_dst_geometry) {
        return absl::InvalidArgumentError(
            absl::StrCat("Destination pool geometry differs across units at ",
                         PythonRepr(unit)));
      }
    }
  }

  // Source geometry is compared per tag across the ranks that register the
  // tag; the first such rank (in request order) is the tag's reference.
  std::map<std::string, RaidenId> tag_reference_unit;
  {
    std::map<std::string, std::vector<std::string>> reference_geometry;
    for (const RaidenId& src_unit : request.src_units) {
      const auto& src_pools = src_by_unit.at(src_unit)->pools();
      for (const auto& [tag, src_indices] : src_pools_by_tag.at(src_unit)) {
        std::vector<std::string> geometry;
        for (int32_t idx : src_indices) {
          geometry.push_back(GeometrySignature(src_pools.Get(idx)));
        }
        auto [it, inserted] = reference_geometry.emplace(tag, geometry);
        if (inserted) {
          tag_reference_unit.emplace(tag, src_unit);
        } else if (it->second != geometry) {
          return absl::InvalidArgumentError(
              absl::StrCat("Source pool geometry differs across ranks at ",
                           PythonRepr(src_unit), " for tag ", PyStrRepr(tag)));
        }
      }
    }
  }

  const std::vector<std::string>& requested_tags = request.transfer_pool_tags;
  if (requested_tags.empty()) {
    return absl::InvalidArgumentError(
        "transfer_pool_tags must name at least one pool tag");
  }
  {
    std::set<std::string> unique_tags(requested_tags.begin(),
                                      requested_tags.end());
    if (unique_tags.size() != requested_tags.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("transfer_pool_tags must not repeat tags; got ",
                       PyStrListRepr(requested_tags)));
    }
  }
  std::vector<int64_t> counts = request.dst_block_counts;
  if (!counts.empty()) {
    if (counts.size() != requested_tags.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "dst_block_counts must align 1:1 with transfer_pool_tags: ",
          counts.size(), " counts for ", requested_tags.size(), " tags"));
    }
    int64_t sum = 0;
    bool any_non_positive = false;
    for (int64_t count : counts) {
      if (count <= 0) any_non_positive = true;
      sum += count;
    }
    if (any_non_positive || sum != static_cast<int64_t>(dst_ids.size())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "dst_block_counts must be positive and sum to the destination "
          "block count: counts=",
          PythonReprIntList(counts), ", blocks=", dst_ids.size()));
    }
  } else if (requested_tags.size() == 1) {
    counts = {static_cast<int64_t>(dst_ids.size())};
  } else {
    return absl::InvalidArgumentError(
        "Multi-tag transfers must split dst_device_block_ids with "
        "dst_block_counts");
  }
  std::vector<std::vector<int64_t>> tag_dst_ids;
  {
    size_t split_cursor = 0;
    for (int64_t count : counts) {
      tag_dst_ids.emplace_back(
          dst_ids.begin() + split_cursor,
          dst_ids.begin() + split_cursor + static_cast<size_t>(count));
      split_cursor += static_cast<size_t>(count);
    }
  }
  std::vector<int64_t> tag_skips(requested_tags.size(), 0);
  if (!request.dst_skip_bytes.empty()) {
    if (request.dst_skip_bytes.size() != requested_tags.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "dst_skip_bytes must align 1:1 with transfer_pool_tags: ",
          request.dst_skip_bytes.size(), " skips for ", requested_tags.size(),
          " tags"));
    }
    for (size_t i = 0; i < request.dst_skip_bytes.size(); ++i) {
      if (request.dst_skip_bytes[i] < 0) {
        return absl::InvalidArgumentError(
            absl::StrCat("dst_skip_bytes must be non-negative; tag ",
                         PyStrRepr(requested_tags[i]), " requested ",
                         request.dst_skip_bytes[i]));
      }
      tag_skips[i] = request.dst_skip_bytes[i];
    }
  }

  {
    std::set<std::string> available_tags;
    for (const auto& pool : dst_meta.pools()) {
      available_tags.insert(pool.tag());
    }
    for (const std::string& plan_tag : requested_tags) {
      if (available_tags.find(plan_tag) == available_tags.end()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "transfer_pool_tags do not match any registered pool: [",
            PyStrRepr(plan_tag), "]"));
      }
    }
  }

  std::set<int64_t> admitted_parallelism;
  for (const RaidenId& unit : request.src_units) {
    admitted_parallelism.insert(src_by_unit.at(unit)->transfer_parallelism());
  }
  if (admitted_parallelism.size() != 1 || admitted_parallelism.count(0) > 0) {
    return absl::InvalidArgumentError(
        "All source ranks must declare one consistent positive "
        "transfer_parallelism");
  }
  const int64_t topology_parallelism = *admitted_parallelism.begin();
  std::map<int64_t, RaidenId> ranks;
  for (const RaidenId& unit : request.src_units) {
    ranks[src_by_unit.at(unit)->transfer_rank()] = unit;
  }
  if (ranks.size() != request.src_units.size()) {
    return absl::InvalidArgumentError(
        "Source transfer_rank values must be unique");
  }
  {
    int64_t expected_rank = 0;
    for (const auto& [rank, unit] : ranks) {
      if (rank != expected_rank) {
        return absl::InvalidArgumentError(
            "Source transfer_rank values must be contiguous from zero");
      }
      ++expected_rank;
    }
  }
  const int64_t requested_parallelism =
      request.parallelism.has_value()
          ? static_cast<int64_t>(*request.parallelism)
          : topology_parallelism;
  if (requested_parallelism <= 0) {
    return absl::InvalidArgumentError("parallelism must be positive");
  }
  if (requested_parallelism > topology_parallelism) {
    return absl::InvalidArgumentError(absl::StrCat(
        "parallelism exceeds source admission: requested=",
        requested_parallelism, ", admitted=", topology_parallelism));
  }

  // Cheap per-tag argument validation runs BEFORE the registration claim
  // so invalid requests never consume (and roll back) a claim.
  std::vector<TagPrecheck> tag_precheck;
  for (size_t group_idx = 0; group_idx < requested_tags.size(); ++group_idx) {
    const std::string& plan_tag = requested_tags[group_idx];
    const std::vector<int64_t>& dst_ids_g = tag_dst_ids[group_idx];
    {
      std::set<int64_t> unique_ids(dst_ids_g.begin(), dst_ids_g.end());
      if (unique_ids.size() != dst_ids_g.size()) {
        return absl::InvalidArgumentError(
            absl::StrCat("dst_device_block_ids for tag ", PyStrRepr(plan_tag),
                         " must not contain duplicates"));
      }
    }
    TagPrecheck precheck;
    for (int i = 0; i < dst_meta.pools_size(); ++i) {
      if (dst_meta.pools(i).tag() == plan_tag) {
        precheck.selected.push_back(i);
      }
    }
    {
      auto ref_it = tag_reference_unit.find(plan_tag);
      if (ref_it == tag_reference_unit.end()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "No source unit registers pools for tag ", PyStrRepr(plan_tag)));
      }
      precheck.src_reference_unit = ref_it->second;
      precheck.src_selected =
          src_pools_by_tag.at(ref_it->second).at(plan_tag);
    }
    const tpu_sync::rpc::RegisterWorkUnitRequest& reference_src =
        *src_by_unit.at(precheck.src_reference_unit);

    std::set<int64_t> src_live_values;
    std::set<int64_t> dst_live_values;
    std::vector<std::vector<PoolLiveSegment>> src_segment_maps;
    std::vector<std::vector<PoolLiveSegment>> dst_segment_maps;
    for (size_t k = 0; k < precheck.selected.size(); ++k) {
      const int32_t pool_idx = precheck.selected[k];
      const auto& src_pool = reference_src.pools(precheck.src_selected[k]);
      const auto& dst_pool = dst_meta.pools(pool_idx);
      auto src_segments = LiveSegments(src_pool);
      if (!src_segments.ok()) return src_segments.status();
      auto dst_segments = LiveSegments(dst_pool);
      if (!dst_segments.ok()) return dst_segments.status();
      int64_t src_live = 0;
      for (const auto& segment : *src_segments) src_live += segment.size;
      int64_t dst_live = 0;
      for (const auto& segment : *dst_segments) dst_live += segment.size;
      if (src_live <= 0 || dst_live <= 0) {
        return absl::InvalidArgumentError(
            absl::StrCat("Pool ", pool_idx, " has no declared live bytes"));
      }
      src_live_values.insert(src_live);
      dst_live_values.insert(dst_live);
      bool src_seen = false;
      for (const auto& existing : src_segment_maps) {
        if (existing == *src_segments) {
          src_seen = true;
          break;
        }
      }
      if (!src_seen) src_segment_maps.push_back(*std::move(src_segments));
      bool dst_seen = false;
      for (const auto& existing : dst_segment_maps) {
        if (existing == *dst_segments) {
          dst_seen = true;
          break;
        }
      }
      if (!dst_seen) dst_segment_maps.push_back(*std::move(dst_segments));
    }
    if (src_live_values.size() != 1 || dst_live_values.size() != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Pools tagged ", plan_tag, " must share one geometry per side; ",
          "src=", PySortedIntSetRepr(src_live_values),
          " dst=", PySortedIntSetRepr(dst_live_values)));
    }
    if (src_segment_maps.size() != 1 || dst_segment_maps.size() != 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("Pools tagged ", plan_tag,
                       " must share one live-region map per side"));
    }
    precheck.src_live = *src_live_values.begin();
    precheck.dst_live = *dst_live_values.begin();
    if (tag_skips[group_idx] % precheck.dst_live != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "dst_skip_bytes must be a whole multiple of the destination page "
          "live bytes for tag ",
          PyStrRepr(plan_tag), ": skip=", tag_skips[group_idx],
          ", page_live_bytes=", precheck.dst_live));
    }
    precheck.src_segments = std::move(src_segment_maps[0]);
    precheck.dst_segments = std::move(dst_segment_maps[0]);
    tag_precheck.push_back(std::move(precheck));
  }

  auto registrations_or =
      registry->LookupAndClaim(req_id, uuid, request.src_units, claim_owner);
  if (!registrations_or.ok()) return registrations_or.status();
  auto& registrations = *registrations_or;

  // Per-tag planning: each requested tag selects its own pools, owns its
  // own destination block-id space and coverage validation, and emits one
  // entry group.
  absl::btree_map<RaidenId, std::string, RequestBlockRegistry::RaidenIdLess>
      dst_peers;
  {
    absl::btree_set<std::string> unique_peers;
    for (const RaidenId& unit : request.dst_units) {
      const std::string peer = dst_by_unit.at(unit)->shards(0);
      unique_peers.insert(peer);
      dst_peers.emplace(unit, peer);
    }
    if (unique_peers.size() != request.dst_units.size()) {
      std::vector<std::string> sorted_peers(unique_peers.begin(),
                                            unique_peers.end());
      return absl::InvalidArgumentError(absl::StrCat(
          "Destinations must register distinct data-plane endpoints; got ",
          PyStrListRepr(sorted_peers)));
    }
  }
  std::map<RaidenId, std::vector<ScheduleEntry>,
           RequestBlockRegistry::RaidenIdLess>
      schedules;
  // Python dict-of-schedules is insertion-ordered by first emission; the
  // rank filter below re-derives order, so a sorted map is equivalent.
  std::vector<PlanPoolGroup> pool_groups;
  std::vector<int32_t> union_selected;
  std::vector<int64_t> flat_extents;
  int64_t emitted_chunks = 0;
  for (size_t group_idx = 0; group_idx < requested_tags.size(); ++group_idx) {
    const std::string& plan_tag = requested_tags[group_idx];
    const std::vector<int64_t>& dst_ids_g = tag_dst_ids[group_idx];
    const TagPrecheck& precheck = tag_precheck[group_idx];
    const int64_t src_live = precheck.src_live;
    const int64_t dst_live = precheck.dst_live;
    const int64_t tag_skip = tag_skips[group_idx];

    std::vector<std::pair<RaidenId, const PoolSpanRegistration*>> declared;
    for (const RaidenId& unit : request.src_units) {
      const RequestBlockRegistration& registration = registrations.at(unit);
      for (const PoolSpanRegistration& entry : registration.pool_spans) {
        if (entry.tag != plan_tag) continue;
        if (entry.spans.empty()) continue;
        if (src_pools_by_tag.at(unit).find(plan_tag) ==
            src_pools_by_tag.at(unit).end()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Byte spans are declared for tag ", PyStrRepr(plan_tag),
              " by ", PythonRepr(unit),
              ", which registers no pool with that tag"));
        }
        declared.emplace_back(unit, &entry);
      }
    }
    if (declared.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("No byte-span declarations are registered for tag ",
                       plan_tag, ", req_id=", req_id, ", uuid=", uuid));
    }

    // T3.4: destination-page-agnostic declarations (dst_space_version=1)
    // split at destination page boundaries here, where the destination
    // geometry is known. A page-aligned dst_skip_bytes clip applies in the
    // same pass: spans wholly below the cut are dropped (their source bytes
    // are never read), the straddling span is trimmed, and survivors
    // re-base into the clipped space — skip is a dst_live multiple, so
    // in-page offsets are unchanged and only page indices shift.
    std::vector<PoolSpanRegistration> converted_storage;
    converted_storage.reserve(declared.size());
    std::vector<std::pair<RaidenId, const PoolSpanRegistration*>> converted;
    int64_t tag_clipped_bytes = 0;
    int64_t tag_global_end = 0;
    for (const auto& [unit, entry] : declared) {
      if (entry->dst_space_version == 0) {
        if (tag_skip > 0) {
          return absl::InvalidArgumentError(absl::StrCat(
              "dst_skip_bytes requires destination-page-agnostic "
              "(dst_space_version=1) declarations for tag ",
              PyStrRepr(plan_tag), " (declared by ", PythonRepr(unit), ")"));
        }
        converted.emplace_back(unit, entry);
        continue;
      }
      PoolSpanRegistration split_entry = *entry;
      split_entry.spans.clear();
      split_entry.dst_space_version = 0;
      for (const PoolByteSpan& span : entry->spans) {
        if (span.count > 1 || span.src_stride_bytes != 0 ||
            span.dst_stride_bytes != 0) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Global-space byte spans must be plain contiguous ranges "
              "(declared by ",
              PythonRepr(unit), ")"));
        }
        if (span.dst_block_index != 0) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Global-space byte spans must leave dst_block_index zero "
              "(declared by ",
              PythonRepr(unit), ")"));
        }
        tag_global_end =
            std::max(tag_global_end, span.dst_offset_bytes + span.size_bytes);
        int64_t clip_advance = 0;
        if (span.dst_offset_bytes + span.size_bytes <= tag_skip) {
          tag_clipped_bytes += span.size_bytes;
          continue;
        }
        if (span.dst_offset_bytes < tag_skip) {
          clip_advance = tag_skip - span.dst_offset_bytes;
          tag_clipped_bytes += clip_advance;
        }
        int64_t remaining = span.size_bytes - clip_advance;
        int64_t global_offset =
            span.dst_offset_bytes + clip_advance - tag_skip;
        int64_t src_offset = span.src_offset_bytes + clip_advance;
        while (remaining > 0) {
          const int64_t page_index = global_offset / dst_live;
          const int64_t in_page = global_offset % dst_live;
          const int64_t take = std::min(remaining, dst_live - in_page);
          PoolByteSpan split_span;
          split_span.src_block_ordinal = span.src_block_ordinal;
          split_span.src_offset_bytes = src_offset;
          split_span.dst_block_index = page_index;
          split_span.dst_offset_bytes = in_page;
          split_span.size_bytes = take;
          split_span.dst_unit_ordinal = span.dst_unit_ordinal;
          split_entry.spans.push_back(split_span);
          global_offset += take;
          src_offset += take;
          remaining -= take;
        }
      }
      converted_storage.push_back(std::move(split_entry));
      converted.emplace_back(unit, &converted_storage.back());
    }
    if (tag_skip > 0 && tag_skip >= tag_global_end) {
      return absl::InvalidArgumentError(absl::StrCat(
          "dst_skip_bytes removes the entire tag ", PyStrRepr(plan_tag),
          ": skip=", tag_skip, ", declared_extent=", tag_global_end,
          "; a full local hit must not reach the planner"));
    }
    declared = std::move(converted);

    for (size_t k = 0; k < precheck.selected.size(); ++k) {
      const int32_t pool_idx = precheck.selected[k];
      const int64_t dst_num_blocks = dst_meta.pools(pool_idx).num_blocks();
      for (const auto& [unit, entry] : declared) {
        const int32_t src_pool_idx =
            src_pools_by_tag.at(unit).at(plan_tag)[k];
        const int64_t limit =
            src_by_unit.at(unit)->pools(src_pool_idx).num_blocks();
        for (int64_t block_id : entry->block_ids) {
          if (block_id >= limit) {
            return absl::InvalidArgumentError(
                absl::StrCat("Source block id is out of range for pool ",
                             pool_idx, " at ", PythonRepr(unit)));
          }
        }
      }
      for (int64_t block_id : dst_ids_g) {
        if (block_id >= dst_num_blocks) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Destination block id is out of range for pool ", pool_idx));
        }
      }
    }

    // Validate: expand every span's uniform repeats; for every destination
    // unit, the union of the repeats routed to it must cover each
    // destination block exactly once as a prefix-shaped extent. A span
    // without a dst_unit_ordinal (absent on the wire) is replicated to every
    // destination; a span with an ordinal belongs to that destination only
    // (sharded destinations, e.g. head-split KV caches). Declared/covered
    // byte accounting counts each span repeat once however many
    // destinations it reaches.
    const size_t num_dst = request.dst_units.size();
    std::vector<std::vector<std::vector<std::pair<int64_t, int64_t>>>>
        coverage_by_dst(num_dst,
                        std::vector<std::vector<std::pair<int64_t, int64_t>>>(
                            dst_ids_g.size()));
    int64_t expanded_repeats = 0;
    int64_t declared_total = 0;
    int64_t covered_total = 0;
    int64_t replicated_total = 0;
    for (const auto& [unit, entry] : declared) {
      declared_total += entry->declared_bytes;
      for (const PoolByteSpan& span : entry->spans) {
        if (span.dst_unit_ordinal >= static_cast<int64_t>(num_dst)) {
          return absl::InvalidArgumentError(
              absl::StrCat("Byte span dst_unit_ordinal ", span.dst_unit_ordinal,
                           " exceeds the transfer's ", num_dst,
                           " destination units for tag ", PyStrRepr(plan_tag),
                           " (declared by ", PythonRepr(unit), ")"));
        }
        if (span.dst_block_index >= static_cast<int64_t>(dst_ids_g.size())) {
          return absl::InvalidArgumentError(
              absl::StrCat("Byte span destination index ", span.dst_block_index,
                           " exceeds the transfer's ", dst_ids_g.size(),
                           " destination blocks for tag ", PyStrRepr(plan_tag),
                           " (declared by ", PythonRepr(unit), ")"));
        }
        const int64_t src_end = span.src_offset_bytes +
                                (span.count - 1) * span.src_stride_bytes +
                                span.size_bytes;
        if (src_end > src_live) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Byte span exceeds its source block live bytes: end=", src_end,
              ", live=", src_live, " (declared by ", PythonRepr(unit), ")"));
        }
        const int64_t dst_end = span.dst_offset_bytes +
                                (span.count - 1) * span.dst_stride_bytes +
                                span.size_bytes;
        if (dst_end > dst_live) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Byte span exceeds its destination block live bytes: end=",
              dst_end, ", live=", dst_live, " (declared by ", PythonRepr(unit),
              ")"));
        }
        expanded_repeats += span.count;
        if (expanded_repeats > kMaxLiveSegments) {
          return absl::InvalidArgumentError(
              "Byte span plan exceeds the repeat expansion bound");
        }
        for (int32_t repeat = 0; repeat < span.count; ++repeat) {
          const int64_t start =
              span.dst_offset_bytes + repeat * span.dst_stride_bytes;
          if (span.dst_unit_ordinal < 0) {
            for (size_t d = 0; d < num_dst; ++d) {
              coverage_by_dst[d][span.dst_block_index].emplace_back(
                  start, start + span.size_bytes);
            }
            replicated_total += span.size_bytes;
          } else {
            coverage_by_dst[span.dst_unit_ordinal][span.dst_block_index]
                .emplace_back(start, start + span.size_bytes);
          }
          covered_total += span.size_bytes;
        }
      }
    }
    std::vector<int64_t> extents;
    for (size_t d = 0; d < num_dst; ++d) {
      // Single-destination plans keep the historical error strings; the
      // destination suffix appears only for multi-destination plans.
      const std::string dst_suffix =
          num_dst > 1 ? absl::StrCat(" at destination unit ", d) : "";
      std::vector<int64_t> dst_extents;
      for (size_t index = 0; index < dst_ids_g.size(); ++index) {
        std::vector<std::pair<int64_t, int64_t>>& intervals =
            coverage_by_dst[d][index];
        std::sort(intervals.begin(), intervals.end());
        if (intervals.empty()) {
          return absl::InvalidArgumentError(
              absl::StrCat("Destination block index ", index,
                           " has no declared coverage for tag ",
                           PyStrRepr(plan_tag), dst_suffix));
        }
        int64_t covered_until = 0;
        for (const auto& [start, end] : intervals) {
          if (start != covered_until) {
            const bool overlap = start < covered_until;
            return absl::InvalidArgumentError(absl::StrCat(
                "Declared byte spans have a destination coverage ",
                overlap ? "overlap" : "gap", " at byte ",
                std::min(start, covered_until), " of destination block index ",
                index, dst_suffix));
          }
          covered_until = end;
        }
        if (index != dst_ids_g.size() - 1 && covered_until != dst_live) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Destination block index ", index, " is covered to ",
              covered_until, " of ", dst_live,
              " live bytes; only the final block may be partial", dst_suffix));
        }
        dst_extents.push_back(covered_until);
      }
      if (d == 0) {
        extents = std::move(dst_extents);
      } else if (dst_extents != extents) {
        // One extent vector per group travels to every receiver; sharded
        // destinations must therefore cover the same byte prefix of every
        // destination block (true for head-split caches by construction).
        return absl::InvalidArgumentError(absl::StrCat(
            "Destination units must share uniform coverage extents for tag ",
            PyStrRepr(plan_tag), "; destination unit ", d,
            " differs from destination unit 0"));
      }
    }
    if (declared_total != covered_total + tag_clipped_bytes) {
      if (tag_clipped_bytes == 0) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Declared byte totals disagree with coverage: declared=",
            declared_total, ", covered=", covered_total));
      }
      return absl::InvalidArgumentError(absl::StrCat(
          "Declared byte totals disagree with coverage: declared=",
          declared_total, ", covered=", covered_total,
          ", clipped=", tag_clipped_bytes));
    }
    {
      // Every destination covers extent_sum bytes; replicated spans are
      // counted once in covered_total but reach all num_dst destinations.
      int64_t extent_sum = 0;
      for (int64_t extent : extents) extent_sum += extent;
      if (static_cast<int64_t>(num_dst) * extent_sum !=
          covered_total +
              static_cast<int64_t>(num_dst - 1) * replicated_total) {
        return absl::InternalError("byte coverage accounting failed");
      }
    }

    // Bind ordinals and destination indices to physical ids and emit.
    struct OrderedSpan {
      const PoolByteSpan* span;
      RaidenId unit;
      const PoolSpanRegistration* entry;
    };
    std::vector<OrderedSpan> ordered_spans;
    for (const auto& [unit, entry] : declared) {
      for (const PoolByteSpan& span : entry->spans) {
        ordered_spans.push_back(OrderedSpan{&span, unit, entry});
      }
    }
    std::stable_sort(
        ordered_spans.begin(), ordered_spans.end(),
        [](const OrderedSpan& a, const OrderedSpan& b) {
          return std::tie(a.span->dst_block_index, a.span->dst_offset_bytes) <
                 std::tie(b.span->dst_block_index, b.span->dst_offset_bytes);
        });
    std::map<
        RaidenId,
        std::map<RaidenId, std::set<std::tuple<std::string, int64_t, int64_t>>,
                 RequestBlockRegistry::RaidenIdLess>,
        RequestBlockRegistry::RaidenIdLess>
        transfer_pairs_per_sender;
    for (const OrderedSpan& ordered : ordered_spans) {
      const PoolByteSpan& span = *ordered.span;
      const RaidenId& src_unit = ordered.unit;
      const int64_t src_block_id =
          ordered.entry->block_ids[span.src_block_ordinal];
      const int64_t dst_block_id = dst_ids_g[span.dst_block_index];
      // A span without an ordinal replicates its chunks to every
      // destination (entries differ only in dst_peer); a span with an
      // ordinal reaches that destination only.
      std::vector<RaidenId> targets;
      if (span.dst_unit_ordinal < 0) {
        targets = request.dst_units;
      } else {
        targets.push_back(request.dst_units[span.dst_unit_ordinal]);
      }
      for (int32_t repeat = 0; repeat < span.count; ++repeat) {
        const int64_t src_offset =
            span.src_offset_bytes + repeat * span.src_stride_bytes;
        const int64_t dst_offset =
            span.dst_offset_bytes + repeat * span.dst_stride_bytes;
        auto translated =
            TranslateLiveCopy(precheck.src_segments, precheck.dst_segments,
                              src_offset, dst_offset, span.size_bytes);
        if (!translated.ok()) return translated.status();
        emitted_chunks += static_cast<int64_t>(translated->size()) *
                          static_cast<int64_t>(targets.size());
        if (emitted_chunks > kMaxLiveSegments) {
          return absl::InvalidArgumentError(
              "Byte-span plan exceeds the live-region expansion bound");
        }
        for (const LiveCopyChunk& chunk : *translated) {
          for (const RaidenId& dst_unit_id : targets) {
            ScheduleEntry schedule_entry;
            schedule_entry.dst_peer = dst_peers.at(dst_unit_id);
            schedule_entry.dst_shard_idx = 0;
            schedule_entry.dst_offset_bytes = chunk.dst_physical;
            schedule_entry.src_offset_bytes = chunk.src_physical;
            schedule_entry.size_bytes = chunk.size;
            schedule_entry.src_block_id = src_block_id;
            schedule_entry.dst_block_id = dst_block_id;
            schedule_entry.src_stride_bytes = 0;
            schedule_entry.dst_stride_bytes = 0;
            schedule_entry.count = 1;
            schedule_entry.layer_idx = 0;
            schedule_entry.pool_group = static_cast<int32_t>(group_idx);
            schedules[src_unit].push_back(std::move(schedule_entry));
          }
        }
      }
      auto& sender_pairs = transfer_pairs_per_sender[src_unit];
      for (const RaidenId& dst_unit_id : targets) {
        sender_pairs[dst_unit_id].insert(std::make_tuple(
            dst_peers.at(dst_unit_id), src_block_id, dst_block_id));
      }
    }

    // Computed expected pushes for the receiver.
    std::map<RaidenId, int32_t, RequestBlockRegistry::RaidenIdLess>
        expected_pushes_by_dst;
    for (const RaidenId& dst_unit_id : request.dst_units) {
      int64_t dst_pushes = 0;
      for (const auto& [unit, by_dst] : transfer_pairs_per_sender) {
        auto pairs_it = by_dst.find(dst_unit_id);
        if (pairs_it == by_dst.end()) continue;
        dst_pushes += std::min(requested_parallelism,
                               static_cast<int64_t>(pairs_it->second.size()));
      }
      expected_pushes_by_dst[dst_unit_id] = static_cast<int32_t>(dst_pushes);
    }
    for (const RaidenId& dst_unit_id : request.dst_units) {
      if (expected_pushes_by_dst.at(dst_unit_id) <= 0) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Pool reshard plan contains no source pushes for tag ",
            PyStrRepr(plan_tag),
            num_dst > 1
                ? absl::StrCat(" at destination unit ", PythonRepr(dst_unit_id))
                : ""));
      }
    }
    PlanPoolGroup group;
    group.pool_indices = precheck.selected;
    group.dst_device_block_ids = dst_ids_g;
    group.expected_pushes_by_dst = std::move(expected_pushes_by_dst);
    group.dst_expected_extent_bytes = extents;
    // FA (the first requested tag by connector convention) uploads first;
    // state classes land after it on aliased arena pages.
    group.order_rank = group_idx == 0 ? 0 : 1;
    pool_groups.push_back(std::move(group));
    union_selected.insert(union_selected.end(), precheck.selected.begin(),
                          precheck.selected.end());
    flat_extents.insert(flat_extents.end(), extents.begin(), extents.end());
  }

  {
    std::set<int32_t> unique_selected(union_selected.begin(),
                                      union_selected.end());
    if (unique_selected.size() != union_selected.size()) {
      return absl::InvalidArgumentError(
          "Requested tags select overlapping pool indices; tags must "
          "partition their pools");
    }
  }
  std::map<std::string, int64_t> skipped_pool_counts;
  {
    std::set<int32_t> union_selected_set(union_selected.begin(),
                                         union_selected.end());
    for (int i = 0; i < dst_meta.pools_size(); ++i) {
      if (union_selected_set.find(i) == union_selected_set.end()) {
        skipped_pool_counts[dst_meta.pools(i).tag()] += 1;
      }
    }
  }

  PoolReshardPlan plan;
  for (const auto& [rank, unit] : ranks) {
    if (schedules.find(unit) != schedules.end()) {
      plan.src_units.push_back(unit);
    }
  }
  plan.dst_units = request.dst_units;
  plan.schedules = std::move(schedules);
  for (const auto* meta : all_metadata) {
    plan.worker_rpc_addresses[RaidenIdFromProto(meta->unit())] =
        meta->control_plane_rpc_address();
  }
  plan.dst_peers = std::move(dst_peers);
  plan.uuid = uuid;
  plan.req_id = req_id;
  plan.expected_block_count = static_cast<int64_t>(dst_ids.size());
  plan.expected_pushes_per_pool =
      pool_groups[0].expected_pushes_by_dst.at(request.dst_units[0]);
  plan.transfer_pool_indices = union_selected;
  for (const auto& pool : dst_meta.pools()) {
    plan.pool_dtype_tags.push_back(pool.dtype_tag());
  }
  for (const RaidenId& unit : plan.src_units) {
    const auto& src_pools = src_by_unit.at(unit)->pools();
    std::vector<std::string>& dtype_tags = plan.src_pool_dtype_tags[unit];
    for (const auto& pool : src_pools) {
      dtype_tags.push_back(pool.dtype_tag());
    }
    std::map<int32_t, int32_t>& remap = plan.src_pool_indices[unit];
    const auto& by_tag = src_pools_by_tag.at(unit);
    for (const TagPrecheck& precheck : tag_precheck) {
      auto tag_it = by_tag.find(dst_meta.pools(precheck.selected[0]).tag());
      if (tag_it == by_tag.end()) continue;
      for (size_t k = 0; k < precheck.selected.size(); ++k) {
        remap[precheck.selected[k]] = tag_it->second[k];
      }
    }
  }
  plan.dst_device_block_ids = dst_ids;
  for (size_t ordinal = 0; ordinal < plan.src_units.size(); ++ordinal) {
    plan.src_schedule_keys[plan.src_units[ordinal]] =
        static_cast<int32_t>(ordinal);
  }
  plan.parallelism = static_cast<int32_t>(requested_parallelism);
  plan.num_tokens = std::max<int64_t>(request.num_tokens, 0);
  plan.skipped_pool_counts = std::move(skipped_pool_counts);
  plan.dst_expected_extent_bytes = std::move(flat_extents);
  plan.pool_groups = std::move(pool_groups);
  return plan;
}

}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden
