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
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/rpc/weight_transfer_worker_service.pb.h"
#include "tpu_sync/weight_sync/weight_version.h"

namespace tpu_raiden::weight_sync {

// Admits the writes into one weight synchronizer's host staging buffer, and
// tracks which version the buffer holds.
//
// A versioned receive (transfer `uuid` of one version) claims the buffer with
// Claim(), and waits with AwaitReady() until the claim is ready: once
// everything in flight when it was installed has ended. While a claim exists,
// only the writes of its uuid are admitted, and only once it is ready; while
// none exists, every write is a legacy write, which drops the staged version.
// So a commit stages a version only if nothing but that receive wrote the
// buffer since every earlier write ended. A claim ends when it commits, or
// when it is aborted: superseded by the claim of another uuid, committed with
// writes in flight, or aborted by its owner. The guard keeps the reason of the
// receive aborted last (only that one), and refuses its later claims, writes
// and commit with Aborted.
//
// Readers pin the staged commit; a writer drops the staged version, so no new
// pins are taken, and waits for the pins (device copies, relay pushes).
//
// Thread-safe. The mutex is a leaf: no other lock is taken while holding it.
class StagingBufferGuard final {
 public:
  // A commit to the buffer: its version, and the extent that its writer
  // recorded for each (layer, shard).
  struct StagedCommit {
    WeightVersion version;
    std::vector<tpu_sync::rpc::ShardExtent> extents;
  };

  // A read pin on `commit`, ended by ReleaseReadPin(pin_id).
  struct ReadPin {
    uint64_t pin_id = 0;
    std::shared_ptr<const StagedCommit> commit;
  };

  // A prior claim that ended during Claim().
  struct EndedClaim {
    // 0 if no claim ended.
    uint64_t uuid = 0;
    // Aborted if superseded; OK iff `uuid == 0`.
    absl::Status reason;
  };

  // Controls whether BeginWrite() may fall back to an unclaimed legacy write.
  enum class WritePolicy {
    kAllowLegacy,
    kClaimedOnly,
  };

  // How BeginWrite() admitted a write.
  enum class WriteAdmission {
    kClaimed,    // Under the ready claim of the write's own uuid.
    kUnclaimed,  // With no claim: a legacy write, which dropped the version.
  };

  StagingBufferGuard() = default;
  StagingBufferGuard(const StagingBufferGuard&) = delete;
  StagingBufferGuard& operator=(const StagingBufferGuard&) = delete;

  // Claims the buffer for receive `uuid` of `version`, or joins the claim if
  // `uuid` already holds it, without waiting for the claim to get ready. A new
  // claim supersedes the claim of any other uuid and drops the staged version.
  // Returns the prior claim that ended during this call (superseded by `uuid`,
  // with Aborted); `{0, OK}` if none, as for a join.
  // Fails with
  // - InvalidArgument for uuid 0, which tags untagged pushes, or if `uuid`
  //   holds the claim for another version;
  // - Aborted, with the recorded reason, if `uuid` is the receive aborted last.
  absl::StatusOr<EndedClaim> Claim(uint64_t uuid, const WeightVersion& version)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Waits until the claim of receive `uuid` is ready or ends. Never call it
  // while holding a lock that callers of EndWrite() or ReleaseReadPin() take,
  // or on a thread with an admitted write or a read pin open: the claim would
  // wait for it. Fails with Aborted if the claim of `uuid` ended before it got
  // ready, superseded or aborted, naming the reason if `uuid` is the receive
  // aborted last.
  absl::Status AwaitReady(uint64_t uuid) ABSL_LOCKS_EXCLUDED(mu_);

  // Stages the version of the claim of receive `uuid`, with the `extents` that
  // its writer recorded, and ends the claim. Returns a read pin on the new
  // commit, which writers wait for, for the copy after it. Callers commit only
  // on writes of `uuid` admitted under its latest claim, since the claim did
  // not protect earlier ones.
  // While anything is in flight, aborts the claim instead and fails with
  // FailedPrecondition. Fails with Aborted if `uuid` is the receive aborted
  // last, and with FailedPrecondition if it holds no claim otherwise.
  absl::StatusOr<ReadPin> CommitReceiveAndPin(
      uint64_t uuid, std::vector<tpu_sync::rpc::ShardExtent> extents)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Aborts the claim of receive `uuid` for `reason`, which must not be OK
  // (DCHECK-ed) and which its later claims, writes and commit report. Does
  // nothing if `uuid` holds no claim.
  void AbortReceive(uint64_t uuid, const absl::Status& reason)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Admits a write of transfer `uuid` (0 for an untagged push) into the buffer;
  // each OK must be followed by one EndWrite(uuid) once the write stopped.
  // Refuses the writes of the receive aborted last with Aborted. While a claim
  // exists, admits only the writes of its own uuid once ready (returning
  // WriteAdmission::kClaimed) and refuses other uuids and unready claims with
  // FailedPrecondition.
  //
  // Under WritePolicy::kClaimedOnly, refuses unless `uuid` holds the ready
  // claim without waiting on pins or dropping any staged version.
  //
  // Under WritePolicy::kAllowLegacy, when no claim exists, admits a legacy
  // write (returning WriteAdmission::kUnclaimed), which drops any staged
  // version. While read pins are held, a legacy write first drops the staged
  // version, so no new pins are taken, and waits for them, unless a claim
  // refuses it meanwhile. Never call it on a thread with a read pin open: the
  // write would wait for it.
  absl::StatusOr<WriteAdmission> BeginWrite(
      uint64_t uuid, WritePolicy policy = WritePolicy::kAllowLegacy)
      ABSL_LOCKS_EXCLUDED(mu_);
  void EndWrite(uint64_t uuid) ABSL_LOCKS_EXCLUDED(mu_);

  // Pins the staged commit if it is of `version`, for a reader that writers
  // wait for. Fails with NotFound, naming the staged commit, unless `version`
  // is staged.
  absl::StatusOr<ReadPin> TryAcquireReadPin(const WeightVersion& version)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Ends the read pin `pin_id`; does nothing if it already ended.
  void ReleaseReadPin(uint64_t pin_id) ABSL_LOCKS_EXCLUDED(mu_);

  // The version that the buffer holds, if any.
  std::optional<WeightVersion> staged_version() const ABSL_LOCKS_EXCLUDED(mu_);

 private:
  // A versioned receive's claim. `ready` and `abort_reason` are guarded by
  // `mu_`.
  struct ReceiveClaim {
    uint64_t uuid;
    WeightVersion version;
    // Whether everything in flight when the claim was installed has ended.
    bool ready;
    // Why the claim was aborted, once it was.
    std::optional<absl::Status> abort_reason;
  };

  // Refuses `operation` for the aborted `claim` with Aborted, naming its
  // reason.
  static absl::Status AbortedClaimError(absl::string_view operation,
                                        const ReceiveClaim& claim);

  // Aborts the current claim for `reason`.
  void AbortClaim(absl::Status reason) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Pins the staged commit.
  ReadPin AddPin() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Waits, for a legacy write of `uuid`, until no read pin is held (see
  // BeginWrite()), or a claim refuses the write. Releases `mu_` meanwhile, so
  // callers must re-check the state afterwards.
  void AwaitPinsForLegacyWrite(uint64_t uuid)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Marks the claim ready once nothing earlier is in flight.
  void UpdateClaimReadiness() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // The refusal of `operation` for `uuid`, which holds no claim.
  absl::Status Refusal(absl::string_view operation, uint64_t uuid) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Lists the writes in flight, e.g. "1 of uuid=7, 2 of uuid=0".
  std::string WritesInFlight() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Lists the read pins, e.g. "pin 3 on version(len=2, \"v1\")".
  std::string PinsHeld() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Lists what is in flight, e.g. "writes: 1 of uuid=7; read pins: none".
  std::string InFlight() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  // Describes the staged commit, e.g. "version(len=2, \"v1\")".
  std::string StagedCommitName() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutable absl::Mutex mu_;
  // Null while the buffer holds no version.
  std::shared_ptr<const StagedCommit> staged_ ABSL_GUARDED_BY(mu_);
  // Null while the buffer is unclaimed. Shared with the AwaitReady() calls
  // waiting for the claim, which learn from it how it ended.
  std::shared_ptr<ReceiveClaim> claim_ ABSL_GUARDED_BY(mu_);
  std::shared_ptr<const ReceiveClaim> last_aborted_ ABSL_GUARDED_BY(mu_);
  // Number of writes in flight, by uuid.
  absl::flat_hash_map<uint64_t, int64_t> writes_in_flight_ ABSL_GUARDED_BY(mu_);
  // The commit of each read pin held, by pin id.
  absl::flat_hash_map<uint64_t, std::shared_ptr<const StagedCommit>> pins_
      ABSL_GUARDED_BY(mu_);
  uint64_t last_pin_id_ ABSL_GUARDED_BY(mu_) = 0;
};

}  // namespace tpu_raiden::weight_sync

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_STAGING_BUFFER_GUARD_H_
