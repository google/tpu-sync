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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/rpc/weight_transfer_worker_service.pb.h"
#include "tpu_sync/weight_sync/weight_version.h"

namespace tpu_raiden::weight_sync {

StagingBufferGuard::StagingBufferGuard(absl::Duration legacy_write_pin_wait)
    : legacy_write_pin_wait_(legacy_write_pin_wait) {
  CHECK_GE(legacy_write_pin_wait_, absl::ZeroDuration())
      << "legacy_write_pin_wait must be non-negative; got "
      << legacy_write_pin_wait_;
}

uint64_t StagingBufferGuard::claim_epoch() const {
  absl::MutexLock lock(mu_);
  return claim_epoch_;
}

uint64_t StagingBufferGuard::Claim(uint64_t uuid) {
  CHECK_NE(uuid, uint64_t{0}) << "Claim() requires a non-zero uuid";
  std::shared_ptr<const StagedCommit> dropped_staged;
  absl::MutexLock lock(mu_);
  CHECK(claim_ == nullptr || claim_->uuid != uuid)
      << "Claim(uuid=" << uuid << ") already holds the active claim";
  const uint64_t superseded = claim_ != nullptr ? claim_->uuid : 0;
  ++claim_epoch_;
  dropped_staged = std::exchange(staged_, nullptr);
  claim_ = std::make_shared<ReceiveClaim>(ReceiveClaim{
      uuid, /*ready=*/legacy_writes_in_flight_ == 0 && pins_.empty()});
  return superseded;
}

absl::Status StagingBufferGuard::AwaitReady(uint64_t uuid,
                                            absl::Time deadline) {
  absl::MutexLock lock(mu_);
  if (claim_ == nullptr || claim_->uuid != uuid) return AbortedWait(uuid);
  const std::shared_ptr<ReceiveClaim> claim = claim_;
  if (claim->ready) return absl::OkStatus();
  const auto ready_or_ended = [this, &claim]()
                                  ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                                    return claim->ready || claim != claim_;
                                  };
  mu_.AwaitWithDeadline(absl::Condition(&ready_or_ended), deadline);
  if (claim != claim_) return AbortedWait(uuid);
  if (claim->ready) return absl::OkStatus();
  return absl::DeadlineExceededError(
      absl::StrCat("Wait for claim of uuid=", uuid,
                   " timed out waiting for drain (", InFlight(), ")"));
}

absl::StatusOr<StagingBufferGuard::ReadPin> StagingBufferGuard::CommitAndPin(
    uint64_t uuid, WeightVersion version,
    std::vector<tpu_sync::rpc::ShardExtent> extents) {
  absl::MutexLock lock(mu_);
  if (claim_ == nullptr || claim_->uuid != uuid) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Commit of uuid=", uuid, " refused: ", ClaimOwnerSummary()));
  }
  if (!claim_->ready) {
    claim_.reset();
    return absl::FailedPreconditionError(
        absl::StrCat("Commit of uuid=", uuid, " refused: claim is not ready (",
                     InFlight(), ")"));
  }
  DCHECK_EQ(legacy_writes_in_flight_, 0) << "Ready claim uuid=" << uuid;
  DCHECK(pins_.empty()) << "Ready claim uuid=" << uuid << " has pins ("
                        << PinsHeld() << ")";
  claim_.reset();
  DCHECK(staged_ == nullptr)
      << "Ready claim uuid=" << uuid << " found staged " << staged_->version;
  staged_ = std::make_shared<const StagedCommit>(
      StagedCommit{std::move(version), std::move(extents)});
  return AddPin();
}

void StagingBufferGuard::AbortClaim(uint64_t uuid) {
  absl::MutexLock lock(mu_);
  if (claim_ != nullptr && claim_->uuid == uuid) claim_.reset();
}

absl::Status StagingBufferGuard::BeginLegacyWrite(
    uint64_t uuid, uint64_t expected_claim_epoch) {
  std::shared_ptr<const StagedCommit> dropped_staged;
  absl::MutexLock lock(mu_);
  if (claim_ != nullptr || claim_epoch_ != expected_claim_epoch) {
    return LegacyWriteRefusal(uuid, expected_claim_epoch);
  }
  dropped_staged = std::exchange(staged_, nullptr);
  if (!pins_.empty()) {
    const absl::Time deadline = absl::Now() + legacy_write_pin_wait_;
    const auto unblocked_or_claimed =
        [this, expected_claim_epoch]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
          return pins_.empty() || claim_ != nullptr ||
                 claim_epoch_ != expected_claim_epoch;
        };
    mu_.AwaitWithDeadline(absl::Condition(&unblocked_or_claimed), deadline);
    if (claim_ != nullptr || claim_epoch_ != expected_claim_epoch) {
      return LegacyWriteRefusal(uuid, expected_claim_epoch);
    }
    if (!pins_.empty()) {
      return absl::DeadlineExceededError(
          absl::StrCat("Legacy write of uuid=", uuid, " timed out after ",
                       absl::FormatDuration(legacy_write_pin_wait_),
                       " waiting for read pins (", PinsHeld(), ")"));
    }
  }
  ++legacy_writes_in_flight_;
  return absl::OkStatus();
}

void StagingBufferGuard::EndLegacyWrite() {
  absl::MutexLock lock(mu_);
  CHECK_GT(legacy_writes_in_flight_, 0)
      << "EndLegacyWrite() without a matching BeginLegacyWrite()";
  --legacy_writes_in_flight_;
  UpdateClaimReadiness();
}

absl::StatusOr<StagingBufferGuard::ReadPin>
StagingBufferGuard::TryAcquireReadPin(const WeightVersion& version) {
  absl::MutexLock lock(mu_);
  if (staged_ != nullptr && staged_->version == version) return AddPin();
  return absl::NotFoundError(
      absl::StrCat("Read pin of ", version, " refused: staging buffer holds ",
                   staged_ == nullptr ? std::string("no version")
                                      : absl::StrCat(staged_->version)));
}

void StagingBufferGuard::ReleaseReadPin(uint64_t pin_id) {
  std::shared_ptr<const StagedCommit> released;
  absl::MutexLock lock(mu_);
  const auto it = pins_.find(pin_id);
  if (it == pins_.end()) return;
  released = std::move(it->second);
  pins_.erase(it);
  UpdateClaimReadiness();
}

std::optional<WeightVersion> StagingBufferGuard::staged_version() const {
  absl::MutexLock lock(mu_);
  if (staged_ == nullptr) return std::nullopt;
  return staged_->version;
}

StagingBufferGuard::ReadPin StagingBufferGuard::AddPin() {
  DCHECK(staged_ != nullptr);
  const uint64_t pin_id = ++last_pin_id_;
  pins_.emplace(pin_id, staged_);
  return ReadPin{pin_id, staged_};
}

void StagingBufferGuard::UpdateClaimReadiness() {
  if (claim_ != nullptr && legacy_writes_in_flight_ == 0 && pins_.empty()) {
    claim_->ready = true;
  }
}

absl::Status StagingBufferGuard::AbortedWait(uint64_t uuid) const {
  return absl::AbortedError(absl::StrCat("Wait for claim of uuid=", uuid,
                                         " aborted: ", ClaimOwnerSummary()));
}

absl::Status StagingBufferGuard::LegacyWriteRefusal(
    uint64_t uuid, uint64_t expected_claim_epoch) const {
  if (claim_ != nullptr) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Legacy write of uuid=", uuid, " refused: ", ClaimOwnerSummary(),
        " (claim_epoch=", claim_epoch_, ", expected=", expected_claim_epoch,
        ")"));
  }
  return absl::FailedPreconditionError(
      absl::StrCat("Legacy write of uuid=", uuid,
                   " refused: claim_epoch mismatch (expected ",
                   expected_claim_epoch, ", current ", claim_epoch_, ")"));
}

std::string StagingBufferGuard::ClaimOwnerSummary() const {
  if (claim_ != nullptr) {
    return absl::StrCat("staging buffer is claimed by uuid=", claim_->uuid);
  }
  return "no active claim on the staging buffer";
}

std::string StagingBufferGuard::PinsHeld() const {
  return absl::StrJoin(pins_, ", ", [](std::string* out, const auto& entry) {
    absl::StrAppend(out, "pin ", entry.first, " of ", entry.second->version);
  });
}

std::string StagingBufferGuard::InFlight() const {
  return absl::StrCat(
      "legacy writes in flight: ", legacy_writes_in_flight_,
      "; read pins: ", pins_.empty() ? std::string("none") : PinsHeld());
}

}  // namespace tpu_raiden::weight_sync
