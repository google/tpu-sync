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

#include "tpu_sync/weight_sync/versioned_receives.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden::weight_sync {
namespace {

std::string FormatSkipTiling(const absl::btree_map<int32_t, bool>& skip) {
  std::vector<std::string> parts;
  parts.reserve(skip.size());
  for (const auto& [layer_idx, flag] : skip) {
    parts.push_back(absl::StrCat(layer_idx, ":", flag ? "true" : "false"));
  }
  return absl::StrCat("{", absl::StrJoin(parts, ", "), "}");
}

}  // namespace

VersionedReceives::VersionedReceives(size_t max_records,
                                     absl::Duration min_retention,
                                     size_t max_unclaimed_writes)
    : max_records_(max_records),
      min_retention_(min_retention),
      max_unclaimed_writes_(max_unclaimed_writes) {
  CHECK_GE(max_records_, size_t{3})
      << "max_records must accommodate an active claim, a committing copy, and "
         "at least one finished record; got "
      << max_records_;
  CHECK_GE(min_retention_, absl::ZeroDuration())
      << "min_retention must not be negative";
  CHECK_GE(max_unclaimed_writes_, size_t{1})
      << "max_unclaimed_writes must be at least 1";
}

VersionedReceives::PrepareLookup VersionedReceives::LookupForPrepare(
    uint64_t uuid, const ReceiveSpec& spec, absl::Time now) {
  PrepareLookup out;
  const ReceiveRecord* record = FindLive(uuid, now, &out.evicted_uuids);
  if (record == nullptr) {
    out.decision = PrepareDecision::kClaim;
    return out;
  }
  if (record->state != ReceiveState::kActive || record->committing) {
    out.decision = PrepareDecision::kFinished;
    out.status = PrepareStatusForFinished(*record);
    return out;
  }
  if (record->spec.version != spec.version) {
    out.decision = PrepareDecision::kError;
    out.status = absl::InvalidArgumentError(absl::StrCat(
        "Receive uuid=", uuid, " repeated with ", spec.version,
        ", but it claimed the staging buffer for ", record->spec.version));
    return out;
  }
  if (record->spec.host_tiling_mode != spec.host_tiling_mode) {
    out.decision = PrepareDecision::kError;
    out.status = absl::InvalidArgumentError(absl::StrCat(
        "Receive uuid=", uuid, " repeated with host_tiling_mode=",
        tpu_sync::rpc::HostTilingMode_Name(spec.host_tiling_mode),
        ", but it claimed with host_tiling_mode=",
        tpu_sync::rpc::HostTilingMode_Name(record->spec.host_tiling_mode)));
    return out;
  }
  if (record->spec.skip_tiling != spec.skip_tiling) {
    out.decision = PrepareDecision::kError;
    out.status = absl::InvalidArgumentError(absl::StrCat(
        "Receive uuid=", uuid,
        " repeated with skip_tiling=", FormatSkipTiling(spec.skip_tiling),
        ", but it claimed with skip_tiling=",
        FormatSkipTiling(record->spec.skip_tiling)));
    return out;
  }
  out.decision = PrepareDecision::kJoin;
  return out;
}

std::vector<uint64_t> VersionedReceives::StartActive(
    uint64_t uuid, ReceiveSpec spec, absl::Duration receive_timeout,
    absl::Time expires_at, absl::Time now) {
  CHECK_NE(uuid, uint64_t{0}) << "StartActive requires a non-zero uuid";
  CHECK_EQ(active_claim_uuid_, uint64_t{0})
      << "StartActive(uuid=" << uuid
      << ") while receive uuid=" << active_claim_uuid_
      << " still holds the active claim";
  std::vector<uint64_t> evicted;
  const ReceiveRecord* existing = FindLive(uuid, now, &evicted);
  CHECK(existing == nullptr)
      << "StartActive(uuid=" << uuid << ") called for an existing record";
  if (records_.size() >= max_records_) {
    SweepExpiredFinished(now, &evicted);
  }
  while (records_.size() >= max_records_ && !finished_order_.empty()) {
    EvictFinished(FindFinished(finished_order_.front()), &evicted);
  }
  ReceiveRecord record;
  record.uuid = uuid;
  record.spec = std::move(spec);
  record.state = ReceiveState::kActive;
  record.receive_timeout = std::max(absl::ZeroDuration(), receive_timeout);
  record.expires_at = expires_at;
  records_.emplace(uuid, Entry{.record = std::move(record)});
  active_claim_uuid_ = uuid;
  return evicted;
}

void VersionedReceives::MarkReady(uint64_t uuid) {
  if (active_claim_uuid_ != uuid) return;
  auto it = records_.find(uuid);
  if (it != records_.end() &&
      it->second.record.state == ReceiveState::kActive &&
      !it->second.record.committing) {
    it->second.record.ready = true;
  }
}

void VersionedReceives::MarkCommitting(uint64_t uuid) {
  auto it = records_.find(uuid);
  if (it == records_.end() ||
      it->second.record.state != ReceiveState::kActive) {
    return;
  }
  it->second.record.ready = false;
  it->second.record.committing = true;
  if (active_claim_uuid_ == uuid) active_claim_uuid_ = 0;
}

void VersionedReceives::MarkCommitted(uint64_t uuid, absl::Time now) {
  auto it = records_.find(uuid);
  if (it == records_.end() ||
      it->second.record.state != ReceiveState::kActive) {
    return;
  }
  ReceiveRecord& record = it->second.record;
  record.state = ReceiveState::kCommitted;
  record.ready = false;
  record.committing = false;
  record.outcome = absl::OkStatus();
  record.retain_until = now + RetentionFor(record.receive_timeout);
  if (active_claim_uuid_ == uuid) active_claim_uuid_ = 0;
  ClearUnclaimedWrite(uuid);
  it->second.finished_position =
      finished_order_.insert(finished_order_.end(), uuid);
}

void VersionedReceives::Finish(uint64_t uuid, ReceiveState state,
                               absl::Status outcome, absl::Time now,
                               bool allow_committing) {
  CHECK(state == ReceiveState::kAborted || state == ReceiveState::kExpired ||
        state == ReceiveState::kFailed)
      << "Finish(uuid=" << uuid << ") needs kAborted/kExpired/kFailed, got "
      << static_cast<int>(state);
  CHECK(!outcome.ok()) << "Finish(uuid=" << uuid << ") needs a non-OK outcome";
  auto it = records_.find(uuid);
  if (it == records_.end() ||
      it->second.record.state != ReceiveState::kActive ||
      (!allow_committing && it->second.record.committing)) {
    return;
  }
  ReceiveRecord& record = it->second.record;
  record.state = state;
  record.ready = false;
  record.committing = false;
  record.outcome = std::move(outcome);
  record.retain_until = now + RetentionFor(record.receive_timeout);
  if (active_claim_uuid_ == uuid) active_claim_uuid_ = 0;
  ClearUnclaimedWrite(uuid);
  it->second.finished_position =
      finished_order_.insert(finished_order_.end(), uuid);
}

void VersionedReceives::FinishWithFailure(uint64_t uuid, absl::Status outcome,
                                          absl::Time now,
                                          bool allow_committing) {
  const ReceiveState state = StateForFailure(outcome);
  Finish(uuid, state, std::move(outcome), now, allow_committing);
}

absl::StatusOr<bool> VersionedReceives::CheckWriteNotTombstoned(
    uint64_t uuid, absl::Time now, std::vector<uint64_t>* evicted_uuids) {
  const ReceiveRecord* record = FindLive(uuid, now, evicted_uuids);
  if (record == nullptr) return false;
  if (record->state == ReceiveState::kActive && !record->committing) {
    return true;
  }
  return WriteRefusalForFinished(*record);
}

absl::Status VersionedReceives::PrepareStatusForFinished(
    const ReceiveRecord& record) {
  if (record.state == ReceiveState::kCommitted || record.committing) {
    return absl::FailedPreconditionError(
        absl::StrCat("Receive uuid=", record.uuid, " of ", record.spec.version,
                     " already committed"));
  }
  return record.outcome;
}

absl::Status VersionedReceives::WriteRefusalForFinished(
    const ReceiveRecord& record) {
  if (record.state == ReceiveState::kCommitted || record.committing) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Write of uuid=", record.uuid, " refused: receive uuid=", record.uuid,
        " of ", record.spec.version, " already committed"));
  }
  return absl::AbortedError(absl::StrCat(
      "Write of uuid=", record.uuid, " refused: receive uuid=", record.uuid,
      " of ", record.spec.version,
      " already ended: ", record.outcome.ToString()));
}

ReceiveState VersionedReceives::StateForFailure(const absl::Status& outcome) {
  switch (outcome.code()) {
    case absl::StatusCode::kDeadlineExceeded:
      return ReceiveState::kExpired;
    case absl::StatusCode::kAborted:
    case absl::StatusCode::kCancelled:
      return ReceiveState::kAborted;
    default:
      return ReceiveState::kFailed;
  }
}

const ReceiveRecord* VersionedReceives::Find(uint64_t uuid) const {
  auto it = records_.find(uuid);
  return it == records_.end() ? nullptr : &it->second.record;
}

const ReceiveRecord* VersionedReceives::FindLive(
    uint64_t uuid, absl::Time now, std::vector<uint64_t>* evicted_uuids) {
  auto it = records_.find(uuid);
  if (it == records_.end() || EvictIfRetentionExpired(it, now, evicted_uuids)) {
    return nullptr;
  }
  return &it->second.record;
}

bool VersionedReceives::IsReadyActive(uint64_t uuid) const {
  if (uuid == 0 || active_claim_uuid_ != uuid) return false;
  const ReceiveRecord* record = Find(uuid);
  return record != nullptr && record->state == ReceiveState::kActive &&
         record->ready && !record->committing;
}

void VersionedReceives::RecordUnclaimedWrite(uint64_t uuid) {
  if (uuid == 0) return;
  if (auto it = unclaimed_writes_.find(uuid); it != unclaimed_writes_.end()) {
    unclaimed_write_order_.splice(unclaimed_write_order_.end(),
                                  unclaimed_write_order_, it->second);
    return;
  }
  if (unclaimed_writes_.size() >= max_unclaimed_writes_) {
    ClearUnclaimedWrite(unclaimed_write_order_.front());
  }
  unclaimed_writes_.emplace(
      uuid, unclaimed_write_order_.insert(unclaimed_write_order_.end(), uuid));
}

bool VersionedReceives::HasUnclaimedWrite(uint64_t uuid) const {
  return unclaimed_writes_.contains(uuid);
}

void VersionedReceives::ClearUnclaimedWrite(uint64_t uuid) {
  auto it = unclaimed_writes_.find(uuid);
  if (it == unclaimed_writes_.end()) return;
  unclaimed_write_order_.erase(it->second);
  unclaimed_writes_.erase(it);
}

absl::Duration VersionedReceives::RetentionFor(
    absl::Duration receive_timeout) const {
  return std::max(min_retention_, 2 * receive_timeout);
}

bool VersionedReceives::EvictIfRetentionExpired(
    RecordMap::iterator it, absl::Time now,
    std::vector<uint64_t>* evicted_uuids) {
  const ReceiveRecord& record = it->second.record;
  if (record.state == ReceiveState::kActive || now < record.retain_until) {
    return false;
  }
  EvictFinished(it, evicted_uuids);
  return true;
}

VersionedReceives::RecordMap::iterator VersionedReceives::FindFinished(
    uint64_t uuid) {
  auto it = records_.find(uuid);
  CHECK(it != records_.end())
      << "Receive uuid=" << uuid << " has no finished record to evict";
  return it;
}

void VersionedReceives::EvictFinished(RecordMap::iterator it,
                                      std::vector<uint64_t>* evicted_uuids) {
  const uint64_t uuid = it->first;
  CHECK(it->second.finished_position.has_value())
      << "Receive uuid=" << uuid << " has no finished record to evict";
  finished_order_.erase(*it->second.finished_position);
  records_.erase(it);
  ClearUnclaimedWrite(uuid);
  if (evicted_uuids != nullptr) evicted_uuids->push_back(uuid);
}

void VersionedReceives::SweepExpiredFinished(
    absl::Time now, std::vector<uint64_t>* evicted_uuids) {
  for (auto position = finished_order_.begin();
       position != finished_order_.end();) {
    const RecordMap::iterator it = FindFinished(*position);
    ++position;  // Evicting `it` erases its node, so advance first.
    EvictIfRetentionExpired(it, now, evicted_uuids);
  }
}

}  // namespace tpu_raiden::weight_sync
