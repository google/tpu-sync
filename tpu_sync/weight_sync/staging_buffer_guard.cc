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

#include "tpu_sync/weight_sync/staging_buffer_guard.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/rpc/weight_transfer_worker_service.pb.h"
#include "tpu_sync/weight_sync/weight_version.h"

namespace tpu_raiden::weight_sync {

StagingBufferGuard::StagingBufferGuard(absl::Duration claim_timeout,
                                       absl::Duration legacy_write_pin_wait)
    : claim_timeout_(claim_timeout),
      legacy_write_pin_wait_(legacy_write_pin_wait) {}

absl::StatusOr<StagingBufferGuard::EndedClaim> StagingBufferGuard::Claim(
    uint64_t uuid, const WeightVersion& version, absl::Time expires_at) {
  if (uuid == 0) {
    return absl::InvalidArgumentError(
        "A versioned receive needs a non-zero uuid (0 tags untagged pushes); "
        "got uuid=0");
  }
  absl::MutexLock lock(mu_);
  const absl::Time now = absl::Now();
  EndedClaim ended = ExpireClaimIfPastDeadlineLocked(now);
  AbortClaimPastItsDrain();
  if (claim_ != nullptr && claim_->uuid == uuid) {
    if (claim_->version != version) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Receive uuid=", uuid, " repeated with ", version,
          ", but it claimed the staging buffer for ", claim_->version));
    }
    claim_->expires_at = std::min(claim_->expires_at, expires_at);
  } else if (last_aborted_ != nullptr && last_aborted_->uuid == uuid) {
    if (ended.uuid != 0) pending_expired_.push_back(std::move(ended));
    return Refusal("Claim", uuid);
  } else {
    if (claim_ != nullptr) {
      absl::Status superseded = absl::AbortedError(
          absl::StrCat("superseded by receive uuid=", uuid, " of ", version));
      ended = {claim_->uuid, superseded};
      AbortClaim(std::move(superseded));
    }
    claim_ = std::make_shared<ReceiveClaim>(
        ReceiveClaim{uuid, version, now + claim_timeout_, expires_at,
                     writes_in_flight_.empty() && pins_.empty(), std::nullopt});
    staged_ = nullptr;
  }
  return ended;
}

absl::Status StagingBufferGuard::AwaitReady(uint64_t uuid) {
  absl::MutexLock lock(mu_);
  if (claim_ == nullptr || claim_->uuid != uuid) {
    // It claimed before (see Claim()), so its claim ended unready.
    const absl::Status refused = Refusal("Wait", uuid);
    return absl::AbortedError(refused.message());
  }
  const std::shared_ptr<ReceiveClaim> claim = claim_;
  if (absl::Now() >= claim->expires_at) {
    absl::Status expired = ExpiryError(*claim);
    pending_expired_.push_back({claim->uuid, expired});
    AbortClaim(expired);
    return expired;
  }
  if (claim->ready) return absl::OkStatus();
  ++claim->waiters;
  while (!claim->ready && claim == claim_) {
    const absl::Time observed_expiry = claim->expires_at;
    const auto ready_ended_or_tightened =
        [this, &claim, observed_expiry]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
          return claim->ready || claim != claim_ ||
                 claim->expires_at < observed_expiry;
        };
    if (!mu_.AwaitWithDeadline(
            absl::Condition(&ready_ended_or_tightened),
            std::min(claim->drain_deadline, observed_expiry))) {
      break;
    }
  }
  --claim->waiters;
  if (claim->abort_reason.has_value()) {
    return AbortedClaimError("Wait", *claim);
  }
  if (claim == claim_ && absl::Now() >= claim->expires_at) {
    absl::Status expired = ExpiryError(*claim);
    pending_expired_.push_back({claim->uuid, expired});
    AbortClaim(expired);
    return expired;
  }
  if (claim->ready) return absl::OkStatus();
  // Still the current claim: only a commit ends a claim without aborting it,
  // and it needs a ready claim.
  DCHECK_EQ(claim, claim_) << "The claim of uuid=" << uuid
                           << " ended unready without being aborted";
  absl::Status timed_out = DrainTimeoutError(*claim);
  AbortClaim(timed_out);
  return timed_out;
}

absl::StatusOr<StagingBufferGuard::ReadPin>
StagingBufferGuard::CommitReceiveAndPin(
    uint64_t uuid, std::vector<tpu_sync::rpc::ShardExtent> extents) {
  absl::MutexLock lock(mu_);
  if (EndedClaim expired = ExpireClaimIfPastDeadlineLocked(absl::Now());
      expired.uuid != 0) {
    if (expired.uuid == uuid) return expired.reason;
    pending_expired_.push_back(std::move(expired));
  }
  if (claim_ == nullptr || claim_->uuid != uuid) {
    return Refusal("Commit", uuid);
  }
  if (!claim_->ready || !writes_in_flight_.empty()) {
    absl::Status refused = absl::FailedPreconditionError(
        absl::StrCat("Commit of uuid=", uuid, " refused: still in flight (",
                     InFlight(), ")"));
    AbortClaim(refused);
    return refused;
  }
  staged_ = std::make_shared<const StagedCommit>(
      StagedCommit{claim_->version, std::move(extents)});
  claim_ = nullptr;
  return AddPin();
}

void StagingBufferGuard::AbortReceive(uint64_t uuid,
                                      const absl::Status& reason) {
  DCHECK(!reason.ok()) << "AbortReceive(uuid=" << uuid
                       << ") needs a non-OK reason";
  absl::MutexLock lock(mu_);
  if (claim_ != nullptr && claim_->uuid == uuid) AbortClaim(reason);
}

absl::StatusOr<StagingBufferGuard::WriteAdmission>
StagingBufferGuard::BeginWrite(uint64_t uuid, WritePolicy policy) {
  absl::MutexLock lock(mu_);
  const absl::Time now = absl::Now();
  if (EndedClaim expired = ExpireClaimIfPastDeadlineLocked(now);
      expired.uuid != 0) {
    pending_expired_.push_back(std::move(expired));
  }
  AbortClaimPastItsDrain();
  const absl::Time legacy_deadline = now + legacy_write_pin_wait_;
  while (policy == WritePolicy::kAllowLegacy && claim_ == nullptr &&
         !pins_.empty() &&
         (last_aborted_ == nullptr || last_aborted_->uuid != uuid)) {
    ABSL_RETURN_IF_ERROR(AwaitPinsForLegacyWrite(uuid, legacy_deadline));
    if (EndedClaim expired = ExpireClaimIfPastDeadlineLocked(absl::Now());
        expired.uuid != 0) {
      pending_expired_.push_back(std::move(expired));
    }
    AbortClaimPastItsDrain();
  }
  if (claim_ != nullptr && claim_->uuid == uuid) {
    if (!claim_->ready) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Write of uuid=", uuid,
          " refused: its claim on the staging buffer still waits for what "
          "was in flight before it (",
          InFlight(), ")"));
    }
    ++writes_in_flight_[uuid];
    return WriteAdmission::kClaimed;
  }
  if (policy == WritePolicy::kClaimedOnly) {
    return Refusal("Write", uuid);
  }
  if (claim_ != nullptr ||
      (last_aborted_ != nullptr && last_aborted_->uuid == uuid)) {
    return Refusal("Write", uuid);
  }
  DCHECK(pins_.empty()) << "Admitting unclaimed write of uuid=" << uuid
                        << " while staging buffer is still in use ("
                        << InFlight() << ")";
  staged_ = nullptr;
  ++writes_in_flight_[uuid];
  return WriteAdmission::kUnclaimed;
}

void StagingBufferGuard::EndWrite(uint64_t uuid) {
  absl::MutexLock lock(mu_);
  const auto it = writes_in_flight_.find(uuid);
  CHECK(it != writes_in_flight_.end())
      << "EndWrite(uuid=" << uuid << ") without a matching BeginWrite";
  if (--it->second == 0) writes_in_flight_.erase(it);
  UpdateClaimReadiness();
}

absl::StatusOr<StagingBufferGuard::ReadPin>
StagingBufferGuard::TryAcquireReadPin(const WeightVersion& version) {
  absl::MutexLock lock(mu_);
  if (staged_ == nullptr || staged_->version != version) {
    return absl::NotFoundError(absl::StrCat(
        "Read pin of ", version, " refused: the staging buffer holds ",
        StagedCommitName()));
  }
  return AddPin();
}

void StagingBufferGuard::ReleaseReadPin(uint64_t pin_id) {
  absl::MutexLock lock(mu_);
  if (pins_.erase(pin_id) == 0) return;
  UpdateClaimReadiness();
}

std::optional<WeightVersion> StagingBufferGuard::staged_version() const {
  absl::MutexLock lock(mu_);
  if (staged_ == nullptr) return std::nullopt;
  return staged_->version;
}

bool StagingBufferGuard::Claims(uint64_t uuid) const {
  absl::MutexLock lock(mu_);
  return claim_ != nullptr && claim_->uuid == uuid &&
         absl::Now() < claim_->expires_at;
}

absl::Status StagingBufferGuard::CheckClaim(absl::string_view operation,
                                            uint64_t uuid) const {
  absl::MutexLock lock(mu_);
  if (claim_ != nullptr && claim_->uuid == uuid) {
    if (absl::Now() < claim_->expires_at) return absl::OkStatus();
    return ExpiryError(*claim_);
  }
  return Refusal(operation, uuid);
}

absl::Status StagingBufferGuard::CheckUnclaimed(absl::string_view operation,
                                                uint64_t uuid) const {
  absl::MutexLock lock(mu_);
  if (claim_ != nullptr && absl::Now() < claim_->expires_at) {
    return absl::FailedPreconditionError(absl::StrCat(
        operation, " of uuid=", uuid, " refused: receive uuid=", claim_->uuid,
        " of ", claim_->version, " claims the staging buffer"));
  }
  return absl::OkStatus();
}

std::vector<StagingBufferGuard::EndedClaim>
StagingBufferGuard::ExpireIfPastDeadline() {
  absl::MutexLock lock(mu_);
  if (EndedClaim expired = ExpireClaimIfPastDeadlineLocked(absl::Now());
      expired.uuid != 0) {
    pending_expired_.push_back(std::move(expired));
  }
  std::vector<EndedClaim> out;
  out.swap(pending_expired_);
  return out;
}

absl::Time StagingBufferGuard::ClaimExpiry(uint64_t uuid) const {
  absl::MutexLock lock(mu_);
  if (claim_ != nullptr && claim_->uuid == uuid) return claim_->expires_at;
  return absl::InfiniteFuture();
}

void StagingBufferGuard::ForgetAborted(uint64_t uuid) {
  absl::MutexLock lock(mu_);
  if (last_aborted_ != nullptr && last_aborted_->uuid == uuid) {
    last_aborted_ = nullptr;
  }
}

absl::Status StagingBufferGuard::AbortedClaimError(absl::string_view operation,
                                                   const ReceiveClaim& claim) {
  DCHECK(claim.abort_reason.has_value());
  return absl::AbortedError(absl::StrCat(
      operation, " of uuid=", claim.uuid, " refused: receive uuid=", claim.uuid,
      " of ", claim.version, " was aborted: ", claim.abort_reason->ToString()));
}

absl::Status StagingBufferGuard::ExpiryError(const ReceiveClaim& claim) {
  return absl::DeadlineExceededError(
      absl::StrCat("Receive uuid=", claim.uuid, " of ", claim.version,
                   " expired before it committed"));
}

void StagingBufferGuard::AbortClaim(absl::Status reason) {
  claim_->abort_reason = std::move(reason);
  last_aborted_ = std::move(claim_);
}

StagingBufferGuard::EndedClaim
StagingBufferGuard::ExpireClaimIfPastDeadlineLocked(absl::Time now) {
  if (claim_ == nullptr || now < claim_->expires_at) return {};
  absl::Status reason = ExpiryError(*claim_);
  const uint64_t expired_uuid = claim_->uuid;
  AbortClaim(reason);
  return {expired_uuid, std::move(reason)};
}

void StagingBufferGuard::AbortClaimPastItsDrain() {
  // A waiting AwaitReady() aborts it itself, reporting DeadlineExceeded.
  if (claim_ != nullptr && !claim_->ready && claim_->waiters == 0 &&
      absl::Now() >= claim_->drain_deadline) {
    AbortClaim(DrainTimeoutError(*claim_));
  }
}

absl::Status StagingBufferGuard::DrainTimeoutError(
    const ReceiveClaim& claim) const {
  return absl::DeadlineExceededError(absl::StrCat(
      "Receive uuid=", claim.uuid, " of ", claim.version, " timed out after ",
      absl::FormatDuration(claim_timeout_),
      " waiting for what was in flight before its claim (", InFlight(), ")"));
}

StagingBufferGuard::ReadPin StagingBufferGuard::AddPin() {
  const uint64_t pin_id = ++last_pin_id_;
  pins_.emplace(pin_id, staged_);
  return ReadPin{pin_id, staged_};
}

absl::Status StagingBufferGuard::AwaitPinsForLegacyWrite(uint64_t uuid,
                                                         absl::Time deadline) {
  // Also wakes for a claim, which refuses the legacy write, for a new commit,
  // whose pins the write must wait for too, and if `uuid` itself became the
  // last aborted claim while `mu_` was unlocked.
  const auto unblocked_or_changed =
      [this, uuid]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        return pins_.empty() || claim_ != nullptr || staged_ != nullptr ||
               (last_aborted_ != nullptr && last_aborted_->uuid == uuid);
      };
  while (claim_ == nullptr &&
         (last_aborted_ == nullptr || last_aborted_->uuid != uuid) &&
         !pins_.empty()) {
    staged_ = nullptr;  // So no new pins are taken.
    if (!mu_.AwaitWithDeadline(absl::Condition(&unblocked_or_changed),
                               deadline)) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Write of uuid=", uuid, " refused: read pins still held after ",
          absl::FormatDuration(legacy_write_pin_wait_), ": ", PinsHeld()));
    }
  }
  return absl::OkStatus();
}

void StagingBufferGuard::UpdateClaimReadiness() {
  if (claim_ != nullptr && writes_in_flight_.empty() && pins_.empty()) {
    claim_->ready = true;
  }
}

absl::Status StagingBufferGuard::Refusal(absl::string_view operation,
                                         uint64_t uuid) const {
  if (last_aborted_ != nullptr && last_aborted_->uuid == uuid) {
    return AbortedClaimError(operation, *last_aborted_);
  }
  if (claim_ != nullptr) {
    return absl::FailedPreconditionError(absl::StrCat(
        operation, " of uuid=", uuid, " refused: receive uuid=", claim_->uuid,
        " of ", claim_->version, " claims the staging buffer"));
  }
  return absl::FailedPreconditionError(
      absl::StrCat(operation, " of uuid=", uuid,
                   " refused: it holds no claim on the staging buffer"));
}

std::string StagingBufferGuard::WritesInFlight() const {
  return absl::StrJoin(writes_in_flight_, ", ",
                       [](std::string* out, const auto& uuid_and_count) {
                         absl::StrAppend(out, uuid_and_count.second,
                                         " of uuid=", uuid_and_count.first);
                       });
}

std::string StagingBufferGuard::PinsHeld() const {
  return absl::StrJoin(
      pins_, ", ", [](std::string* out, const auto& pin_and_commit) {
        absl::StrAppend(out, "pin ", pin_and_commit.first, " on ",
                        pin_and_commit.second->version);
      });
}

std::string StagingBufferGuard::InFlight() const {
  return absl::StrCat(
      "writes: ", writes_in_flight_.empty() ? "none" : WritesInFlight(),
      "; read pins: ", pins_.empty() ? "none" : PinsHeld());
}

std::string StagingBufferGuard::StagedCommitName() const {
  if (staged_ == nullptr) return "no version";
  return absl::StrCat(staged_->version);
}

}  // namespace tpu_raiden::weight_sync
