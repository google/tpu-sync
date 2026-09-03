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

#include "tpu_sync/kv_cache/reshard/request_block_registry.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/kv_cache/reshard/declaration_types.h"
#include "tpu_sync/kv_cache/reshard/work_unit_directory.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {
namespace {

constexpr int64_t kMaxSpanCount = 1 << 20;

// _pool_live_bytes over the shared C++ leaf primitive.
absl::StatusOr<int64_t> PoolLiveBytes(
    const tpu_sync::rpc::PoolSpecProto& pool_proto) {
  auto pool = PoolSpecFromProto(pool_proto);
  if (!pool.ok()) return pool.status();
  auto segments = ExpandPoolLiveSegments(*pool);
  if (!segments.ok()) return segments.status();
  int64_t total = 0;
  for (const PoolLiveSegment& segment : *segments) total += segment.size;
  return total;
}

}  // namespace

RequestBlockRegistry::RequestBlockRegistry(WorkUnitDirectory* directory,
                                           double ttl_s, Clock clock)
    : directory_(directory),
      mu_(directory->mu()),
      ttl_s_(ttl_s),
      clock_(std::move(clock)) {}

void RequestBlockRegistry::PurgeExpiredRequestBlocksLocked(double now) {
  for (auto it = request_blocks_.begin(); it != request_blocks_.end();) {
    if (it->second.expires_at <= now) {
      it = request_blocks_.erase(it);
    } else {
      ++it;
    }
  }
}

void RequestBlockRegistry::PurgeExpiredLifecycleLocked(double now) {
  for (auto* lifecycle : {&claimed_, &cancelled_}) {
    for (auto it = lifecycle->begin(); it != lifecycle->end();) {
      if (it->second <= now) {
        it = lifecycle->erase(it);
      } else {
        ++it;
      }
    }
  }
  for (auto it = completed_units_.begin(); it != completed_units_.end();) {
    if (it->second.expires_at <= now) {
      it = completed_units_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = claimed_units_.begin(); it != claimed_units_.end();) {
    if (claimed_.find(it->first) == claimed_.end()) {
      claimed_owners_.erase(it->first);
      it = claimed_units_.erase(it);
    } else {
      ++it;
    }
  }
}

int RequestBlockRegistry::RetireLocked(const LifecycleKey& key) {
  const std::string& req_id = key.first;
  const int64_t uuid = key.second;
  std::vector<BlockKey> keys;
  for (const auto& [block_key, registration] : request_blocks_) {
    if (block_key.first == req_id && registration.uuid == uuid) {
      keys.push_back(block_key);
    }
  }
  for (const BlockKey& block_key : keys) {
    request_blocks_.erase(block_key);
  }
  claimed_.erase(key);
  claimed_units_.erase(key);
  claimed_owners_.erase(key);
  completed_units_.erase(key);
  cancelled_.erase(key);
  return static_cast<int>(keys.size());
}

absl::Status RequestBlockRegistry::Register(
    const std::string& req_id, int64_t uuid, const RaidenId& unit,
    const std::vector<int64_t>& block_ids,
    const std::vector<PoolSpanRegistration>& pool_spans) {
  if (req_id.empty()) {
    return absl::InvalidArgumentError("req_id must not be empty");
  }
  if (uuid <= 0) {
    return absl::InvalidArgumentError("uuid must be positive");
  }
  std::vector<int64_t> normalized = block_ids;
  for (int64_t block_id : normalized) {
    if (block_id < 0) {
      return absl::InvalidArgumentError("block_ids must be non-negative");
    }
  }
  {
    std::set<int64_t> unique_ids(normalized.begin(), normalized.end());
    if (unique_ids.size() != normalized.size()) {
      return absl::InvalidArgumentError(
          "block_ids must not contain duplicates");
    }
  }

  std::vector<PoolSpanRegistration> normalized_pool_spans;
  if (!pool_spans.empty()) {
    // Python reads the manifest outside the lock; snapshot it briefly here.
    std::vector<tpu_sync::rpc::PoolSpecProto> manifest;
    {
      absl::MutexLock lock(*mu_);
      directory_->mu()->AssertHeld();
      manifest = directory_->PoolManifestLocked(unit);
    }
    std::map<std::string, int64_t> live_bytes_by_tag;
    for (const auto& pool : manifest) {
      const std::string tag = pool.tag();
      auto live = PoolLiveBytes(pool);
      if (!live.ok()) return live.status();
      auto [it, inserted] = live_bytes_by_tag.emplace(tag, *live);
      if (!inserted && it->second != *live) {
        return absl::InvalidArgumentError(absl::StrCat(
            "pools tagged ", tag, " disagree on live bytes for ",
            PythonRepr(unit),
            "; byte-span registration requires one geometry per tag"));
      }
    }
    std::set<std::string> seen_pool_tags;
    for (const PoolSpanRegistration& entry : pool_spans) {
      const std::string& tag = entry.tag;
      if (tag.empty()) {
        return absl::InvalidArgumentError(
            "pool span registration tag must not be empty");
      }
      if (!seen_pool_tags.insert(tag).second) {
        return absl::InvalidArgumentError(
            absl::StrCat("duplicate pool span registration tag: ", tag));
      }
      auto live_it = live_bytes_by_tag.find(tag);
      if (live_it == live_bytes_by_tag.end()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "pool span registration ", tag,
            " does not match any registered pool for ", PythonRepr(unit)));
      }
      const int64_t live = live_it->second;
      const std::vector<int64_t>& entry_ids = entry.block_ids;
      for (int64_t block_id : entry_ids) {
        if (block_id < 0) {
          return absl::InvalidArgumentError(
              absl::StrCat("pool span registration ", tag,
                           " block_ids must be non-negative"));
        }
      }
      {
        std::set<int64_t> unique_ids(entry_ids.begin(), entry_ids.end());
        if (unique_ids.size() != entry_ids.size()) {
          return absl::InvalidArgumentError(
              absl::StrCat("pool span registration ", tag,
                           " block_ids must not contain duplicates"));
        }
      }
      std::vector<PoolByteSpan> entry_spans;
      int64_t span_bytes = 0;
      for (const PoolByteSpan& span : entry.spans) {
        if (span.size_bytes <= 0) {
          return absl::InvalidArgumentError(absl::StrCat(
              "byte span size_bytes must be positive (", tag, ")"));
        }
        if (span.count <= 0) {
          return absl::InvalidArgumentError(
              absl::StrCat("byte span count must be positive (", tag, ")"));
        }
        if (span.count > kMaxSpanCount) {
          return absl::InvalidArgumentError(
              absl::StrCat("byte span count ", span.count,
                           " exceeds the expansion bound (", tag, ")"));
        }
        if (span.src_offset_bytes < 0 || span.dst_offset_bytes < 0 ||
            span.src_stride_bytes < 0 || span.dst_stride_bytes < 0 ||
            span.dst_block_index < 0) {
          return absl::InvalidArgumentError(absl::StrCat(
              "byte span fields must be non-negative (", tag, ")"));
        }
        if (span.dst_unit_ordinal < -1) {
          return absl::InvalidArgumentError(absl::StrCat(
              "byte span dst_unit_ordinal must be -1 (every destination) or "
              "a non-negative destination index (",
              tag, ")"));
        }
        if (span.src_block_ordinal < 0 ||
            span.src_block_ordinal >= static_cast<int64_t>(entry_ids.size())) {
          return absl::InvalidArgumentError(
              absl::StrCat("byte span ordinal ", span.src_block_ordinal,
                           " is outside the registration's ", entry_ids.size(),
                           " block ids (", tag, ")"));
        }
        const int64_t src_end = span.src_offset_bytes +
                                (span.count - 1) * span.src_stride_bytes +
                                span.size_bytes;
        if (src_end > live) {
          return absl::InvalidArgumentError(
              absl::StrCat("byte span exceeds its source block live bytes (",
                           tag, "): end=", src_end, ", live=", live));
        }
        span_bytes += span.size_bytes * span.count;
        entry_spans.push_back(span);
      }
      if (entry.declared_bytes != span_bytes) {
        return absl::InvalidArgumentError(absl::StrCat(
            "pool span registration ", tag, " declared_bytes ",
            entry.declared_bytes, " disagrees with the span sum ", span_bytes));
      }
      PoolSpanRegistration normalized_entry;
      normalized_entry.tag = tag;
      normalized_entry.block_ids = entry_ids;
      normalized_entry.spans = std::move(entry_spans);
      normalized_entry.declared_bytes = span_bytes;
      normalized_entry.dst_space_version = entry.dst_space_version;
      normalized_pool_spans.push_back(std::move(normalized_entry));
    }
  }

  const double now = clock_();
  RequestBlockRegistration registration;
  registration.uuid = uuid;
  registration.block_ids = normalized;
  registration.expires_at = now + ttl_s_;
  registration.pool_spans = normalized_pool_spans;

  const BlockKey key{req_id, unit};
  const LifecycleKey lifecycle_key{req_id, uuid};
  absl::MutexLock lock(*mu_);
  PurgeExpiredRequestBlocksLocked(now);
  PurgeExpiredLifecycleLocked(now);
  directory_->mu()->AssertHeld();
  if (!directory_->IsRegisteredLocked(unit)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Work unit is not registered: ", PythonRepr(unit)));
  }
  if (cancelled_.find(lifecycle_key) != cancelled_.end()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Request block registration was cancelled for req_id=", req_id,
        ", uuid=", uuid));
  }
  if (claimed_.find(lifecycle_key) != claimed_.end()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Request block registration is already claimed for req_id=", req_id,
        ", uuid=", uuid));
  }
  auto existing = request_blocks_.find(key);
  if (existing != request_blocks_.end()) {
    const RequestBlockRegistration& current = existing->second;
    if (current.uuid != uuid || current.block_ids != normalized ||
        current.pool_spans != normalized_pool_spans) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Conflicting request block registration for req_id=", req_id,
          ", unit=", PythonRepr(unit)));
    }
    // Duplicate request-finish notifications are idempotent and refresh
    // the TTL while preserving the exact registered payload.
  }
  request_blocks_[key] = std::move(registration);
  return absl::OkStatus();
}

absl::StatusOr<int> RequestBlockRegistry::Release(const std::string& req_id,
                                                  int64_t uuid) {
  if (req_id.empty()) {
    return absl::InvalidArgumentError("req_id must not be empty");
  }
  if (uuid <= 0) {
    return absl::InvalidArgumentError("uuid must be positive");
  }
  const double now = clock_();
  absl::MutexLock lock(*mu_);
  PurgeExpiredRequestBlocksLocked(now);
  PurgeExpiredLifecycleLocked(now);
  return RetireLocked(LifecycleKey{req_id, uuid});
}

absl::StatusOr<int> RequestBlockRegistry::Complete(const std::string& req_id,
                                                   int64_t uuid,
                                                   const RaidenId& unit) {
  if (req_id.empty()) {
    return absl::InvalidArgumentError("req_id must not be empty");
  }
  if (uuid <= 0) {
    return absl::InvalidArgumentError("uuid must be positive");
  }
  const double now = clock_();
  absl::MutexLock lock(*mu_);
  PurgeExpiredRequestBlocksLocked(now);
  PurgeExpiredLifecycleLocked(now);
  const LifecycleKey lifecycle_key{req_id, uuid};
  auto expected_it = claimed_units_.find(lifecycle_key);
  const bool has_expected = expected_it != claimed_units_.end();
  auto registration_it = request_blocks_.find(BlockKey{req_id, unit});
  const bool has_registration = registration_it != request_blocks_.end();
  if (!has_expected && has_registration &&
      registration_it->second.uuid == uuid &&
      !registration_it->second.block_ids.empty()) {
    return absl::InvalidArgumentError(
        "Only an empty producer registration may complete before the "
        "request block snapshot is claimed");
  }
  const bool registration_matches =
      has_registration && registration_it->second.uuid == uuid;
  const bool unit_expected =
      has_expected && expected_it->second.count(unit) > 0;
  if (!registration_matches && !unit_expected) {
    // Aggregate completion is idempotent after the last rank has already
    // retired the generation.
    return 0;
  }
  auto [completion_it, inserted] = completed_units_.try_emplace(lifecycle_key);
  Completions& completion = completion_it->second;
  if (inserted) {
    completion.expires_at = now + ttl_s_;
  }
  completion.units.insert(unit);
  completion.expires_at = now + ttl_s_;
  if (has_expected) {
    bool all_voted = true;
    for (const RaidenId& expected_unit : expected_it->second) {
      if (completion.units.count(expected_unit) == 0) {
        all_voted = false;
        break;
      }
    }
    if (all_voted) {
      return RetireLocked(lifecycle_key);
    }
  }
  return 0;
}

absl::StatusOr<bool> RequestBlockRegistry::CancelIfUnclaimed(
    const std::string& req_id, int64_t uuid) {
  if (req_id.empty()) {
    return absl::InvalidArgumentError("req_id must not be empty");
  }
  if (uuid <= 0) {
    return absl::InvalidArgumentError("uuid must be positive");
  }
  const double now = clock_();
  const LifecycleKey lifecycle_key{req_id, uuid};
  absl::MutexLock lock(*mu_);
  PurgeExpiredLifecycleLocked(now);
  if (cancelled_.find(lifecycle_key) != cancelled_.end()) {
    return true;
  }
  if (claimed_.find(lifecycle_key) != claimed_.end()) {
    return false;
  }
  cancelled_[lifecycle_key] = now + ttl_s_;
  completed_units_.erase(lifecycle_key);
  claimed_units_.erase(lifecycle_key);
  claimed_owners_.erase(lifecycle_key);
  std::vector<BlockKey> keys;
  for (const auto& [block_key, registration] : request_blocks_) {
    if (block_key.first == req_id && registration.uuid == uuid) {
      keys.push_back(block_key);
    }
  }
  for (const BlockKey& block_key : keys) {
    request_blocks_.erase(block_key);
  }
  return true;
}

absl::StatusOr<std::map<RaidenId, RequestBlockRegistration,
                        RequestBlockRegistry::RaidenIdLess>>
RequestBlockRegistry::LookupAndClaim(const std::string& req_id, int64_t uuid,
                                     const std::vector<RaidenId>& units,
                                     const void* claim_owner) {
  const double now = clock_();
  const LifecycleKey lifecycle_key{req_id, uuid};
  absl::MutexLock lock(*mu_);
  PurgeExpiredRequestBlocksLocked(now);
  PurgeExpiredLifecycleLocked(now);
  if (cancelled_.find(lifecycle_key) != cancelled_.end()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Request block registration was cancelled for req_id=", req_id,
        ", uuid=", uuid));
  }
  std::set<RaidenId, RaidenIdLess> claimed_units(units.begin(), units.end());
  auto existing_units_it = claimed_units_.find(lifecycle_key);
  if (existing_units_it != claimed_units_.end() &&
      existing_units_it->second != claimed_units) {
    return absl::InvalidArgumentError(
        "Request block snapshot was already claimed for a different "
        "source unit set");
  }
  std::map<RaidenId, RequestBlockRegistration, RaidenIdLess> result;
  for (const RaidenId& unit : units) {
    auto it = request_blocks_.find(BlockKey{req_id, unit});
    if (it == request_blocks_.end() || it->second.uuid != uuid) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Missing producer block registration for req_id=", req_id,
          ", uuid=", uuid, ", unit=", PythonRepr(unit)));
    }
    result.emplace(unit, it->second);
  }
  // This is the claim linearization point: every requested rank has been
  // validated and copied while cancellation is excluded by the shared lock.
  claimed_[lifecycle_key] = now + ttl_s_;
  claimed_units_[lifecycle_key] = claimed_units;
  claimed_owners_[lifecycle_key].insert(claim_owner);
  auto completion_it = completed_units_.find(lifecycle_key);
  if (completion_it != completed_units_.end()) {
    completion_it->second.expires_at = now + ttl_s_;
    bool all_voted = true;
    for (const RaidenId& unit : claimed_units) {
      if (completion_it->second.units.count(unit) == 0) {
        all_voted = false;
        break;
      }
    }
    if (all_voted) {
      RetireLocked(lifecycle_key);
    }
  }
  return result;
}

bool RequestBlockRegistry::AbandonClaim(const std::string& req_id, int64_t uuid,
                                        const void* claim_owner) {
  const LifecycleKey lifecycle_key{req_id, uuid};
  absl::MutexLock lock(*mu_);
  auto claimed_it = claimed_.find(lifecycle_key);
  auto owner_it = claimed_owners_.find(lifecycle_key);
  if (claimed_it == claimed_.end() || owner_it == claimed_owners_.end() ||
      owner_it->second.erase(claim_owner) == 0) {
    return false;
  }
  if (owner_it->second.empty()) {
    claimed_.erase(lifecycle_key);
    claimed_units_.erase(lifecycle_key);
    claimed_owners_.erase(lifecycle_key);
  }
  return true;
}

void RequestBlockRegistry::InvalidateUnitLocked(const RaidenId& unit) {
  for (auto it = request_blocks_.begin(); it != request_blocks_.end();) {
    if (it->first.second == unit) {
      it = request_blocks_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden
