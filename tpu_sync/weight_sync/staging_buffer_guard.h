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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_STAGING_BUFFER_GUARD_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_STAGING_BUFFER_GUARD_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tpu_sync/rpc/weight_transfer_worker_service.pb.h"
#include "tpu_sync/weight_sync/weight_version.h"

namespace tpu_raiden::weight_sync {

// Longest a legacy write waits for read pins (device copies, relay pushes).
// TODO(raiden-eng): Confirm with the first clients.
inline constexpr absl::Duration kLegacyWritePinWait = absl::Seconds(30);

// Session-agnostic guard for one weight synchronizer's shared host staging
// buffer. Owns the receive claim and its monotonic `claim_epoch()`, untagged
// (legacy) writes, the staged version, and read pins. Versioned writes are
// admitted and counted by the receive's TransferSession through the transport.
// Callers must also wait until earlier receives' sessions are Done() before
// letting writes in under a ready claim; AwaitReady() covers only legacy writes
// and read pins. Thread-safe; `mu_` is a leaf mutex.
class StagingBufferGuard final {
 public:
  struct StagedCommit {
    WeightVersion version;
    std::vector<tpu_sync::rpc::ShardExtent> extents;
  };

  struct ReadPin {
    uint64_t pin_id = 0;
    std::shared_ptr<const StagedCommit> commit;
  };

  // `legacy_write_pin_wait` must be non-negative (CHECK-ed).
  explicit StagingBufferGuard(
      absl::Duration legacy_write_pin_wait = kLegacyWritePinWait);
  StagingBufferGuard(const StagingBufferGuard&) = delete;
  StagingBufferGuard& operator=(const StagingBufferGuard&) = delete;

  // Monotonic counter incremented on every Claim() for ABA checks.
  uint64_t claim_epoch() const ABSL_LOCKS_EXCLUDED(mu_);

  // Installs a fresh claim for `uuid` (CHECKs `uuid != 0` and not already the
  // active owner), increments `claim_epoch_`, drops `staged_`, and supersedes
  // any prior claim (waking AwaitReady() with Aborted and BeginLegacyWrite()
  // with FailedPrecondition). Ready immediately iff no legacy writes or pins
  // are in flight. Returns the superseded uuid, or 0 if none.
  uint64_t Claim(uint64_t uuid) ABSL_LOCKS_EXCLUDED(mu_);

  // Non-mutating drain wait for `uuid`'s claim up to `deadline`.
  // Pre-wait: non-owner -> Aborted; ready -> OkStatus.
  // Post-wait: ended -> Aborted; ready -> OkStatus; else DeadlineExceeded
  // naming the in-flight legacy writes and pins, without clearing the claim.
  // Do not call while holding a lock needed by EndLegacyWrite() or
  // ReleaseReadPin(), or while holding an open legacy write or ReadPin.
  absl::Status AwaitReady(uint64_t uuid, absl::Time deadline)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Stages `version` and `extents` for `uuid`'s ready claim, clears the claim,
  // and returns a ReadPin. A non-owner gets FailedPrecondition with no state
  // change; an owner whose claim is not ready loses it and gets
  // FailedPrecondition naming the in-flight legacy writes and pins.
  // Precondition (not checked here): the receive's TransferSession is Done()
  // with an OK status, as checked by the caller under the same lock that
  // serializes Claim(), AbortClaim() and CommitAndPin().
  absl::StatusOr<ReadPin> CommitAndPin(
      uint64_t uuid, WeightVersion version,
      std::vector<tpu_sync::rpc::ShardExtent> extents) ABSL_LOCKS_EXCLUDED(mu_);

  // Clears the active claim iff it belongs to `uuid`; no-op otherwise.
  void AbortClaim(uint64_t uuid) ABSL_LOCKS_EXCLUDED(mu_);

  // Legacy write admission, only for the raw transport's untagged/V1 pushes;
  // `uuid` is used only in messages. Callers check, under the lock that
  // serializes claims, that no versioned receive is active and that earlier
  // receives' sessions are Done(), and read `expected_claim_epoch` from
  // claim_epoch() there. Refuses with FailedPrecondition without touching
  // `staged_` if claimed or `claim_epoch() != expected_claim_epoch` (on entry
  // and on waking). Otherwise drops `staged_`, waits up to
  // `legacy_write_pin_wait_` for active ReadPins (DeadlineExceeded if still
  // held), and admits. Pair OK with EndLegacyWrite(). May block up to
  // `legacy_write_pin_wait_`; do not call while holding the lock that
  // serializes claims or while holding a ReadPin.
  absl::Status BeginLegacyWrite(uint64_t uuid, uint64_t expected_claim_epoch)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Ends one admitted legacy write (CHECKs > 0 in flight); updates readiness.
  void EndLegacyWrite() ABSL_LOCKS_EXCLUDED(mu_);

  // Pins the staged commit iff its version equals `version`; else NotFound.
  absl::StatusOr<ReadPin> TryAcquireReadPin(const WeightVersion& version)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Releases `pin_id` (idempotent no-op if absent) and updates claim readiness.
  void ReleaseReadPin(uint64_t pin_id) ABSL_LOCKS_EXCLUDED(mu_);

  // The version currently staged in the host buffer, if any.
  std::optional<WeightVersion> staged_version() const ABSL_LOCKS_EXCLUDED(mu_);

 private:
  struct ReceiveClaim {
    uint64_t uuid;
    // Set once no legacy writes or pins remain, and never cleared: a held claim
    // admits no legacy write, and Claim() dropped `staged_`, so no new pins.
    bool ready;
  };

  ReadPin AddPin() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void UpdateClaimReadiness() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status AbortedWait(uint64_t uuid) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status LegacyWriteRefusal(uint64_t uuid,
                                  uint64_t expected_claim_epoch) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  std::string ClaimOwnerSummary() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  std::string PinsHeld() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  std::string InFlight() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const absl::Duration legacy_write_pin_wait_;
  mutable absl::Mutex mu_;
  uint64_t claim_epoch_ ABSL_GUARDED_BY(mu_) = 0;
  std::shared_ptr<ReceiveClaim> claim_ ABSL_GUARDED_BY(mu_);
  std::shared_ptr<const StagedCommit> staged_ ABSL_GUARDED_BY(mu_);
  int64_t legacy_writes_in_flight_ ABSL_GUARDED_BY(mu_) = 0;
  absl::flat_hash_map<uint64_t, std::shared_ptr<const StagedCommit>> pins_
      ABSL_GUARDED_BY(mu_);
  uint64_t last_pin_id_ ABSL_GUARDED_BY(mu_) = 0;
};

}  // namespace tpu_raiden::weight_sync

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_STAGING_BUFFER_GUARD_H_
