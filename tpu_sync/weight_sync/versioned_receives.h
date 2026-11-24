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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_VERSIONED_RECEIVES_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_VERSIONED_RECEIVES_H_

#include <cstddef>
#include <cstdint>
#include <list>
#include <optional>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/weight_sync/weight_version.h"

namespace tpu_raiden::weight_sync {

// Default minimum retention of a finished versioned receive record (tombstone).
// A record that finished with `receive_timeout` is retained for
// max(min_retention, 2 * receive_timeout).
// TODO(raiden-eng): Confirm with production telemetry.
inline constexpr absl::Duration kMinReceiveRecordRetention = absl::Minutes(10);

// Default soft bound on retained receive records. When inserting a new ACTIVE
// record at this bound, expired tombstones are swept first and then the oldest
// finished records are evicted so prepares are never refused for space. Active
// records are never evicted, so they can exceed the bound.
// TODO(raiden-eng): Confirm with production telemetry.
inline constexpr size_t kMaxReceiveRecords = 4096;

// Default upper bound on tracked unclaimed-write uuids. Oldest entries are
// evicted LRU-first so a flood of unclaimed pushes cannot exhaust memory or
// permanently poison future receives.
// TODO(raiden-eng): Confirm cap with production telemetry.
inline constexpr size_t kMaxUnclaimedWrites = 4096;

// Lifecycle state of one versioned receive record.
enum class ReceiveState {
  kActive,     // Claimed (draining, ready, or finishing hooked H2D).
  kCommitted,  // Staged (and copied to device if H2D hooks are registered).
  kAborted,    // Superseded, cancelled (drained/forgotten), or aborted.
  kExpired,    // Past its receive deadline.
  kFailed,     // Failed internally (tiling, registration, commit, or H2D).
};

// Shared contract that every prepare (initial claim and later per-sender joins)
// for a given receive uuid must agree on.
struct ReceiveSpec {
  WeightVersion version;
  tpu_sync::rpc::HostTilingMode host_tiling_mode =
      tpu_sync::rpc::HOST_TILING_MODE_UNSPECIFIED;
  absl::btree_map<int32_t, bool> skip_tiling;
};

// One versioned receive's state, deadline, and terminal outcome.
struct ReceiveRecord {
  uint64_t uuid = 0;
  ReceiveSpec spec;
  ReceiveState state = ReceiveState::kActive;
  // True while this receive holds the ready staging claim and may mutate
  // receive state. False while draining, while finishing a pinned hooked copy
  // after its commit, or once finished.
  bool ready = false;
  // True after the receive committed its staged version, while a pinned hooked
  // copy finishes and before the record transitions to kCommitted or kFailed.
  bool committing = false;
  // Receive timeout used to compute post-finish tombstone retention.
  absl::Duration receive_timeout = absl::ZeroDuration();
  // Receive expiry timestamp (InfiniteFuture if unbounded), kept for the
  // caller; VersionedReceives does not enforce it.
  absl::Time expires_at = absl::InfiniteFuture();
  // OK for kActive and kCommitted; non-OK for kAborted, kExpired, and kFailed.
  absl::Status outcome;
  // When a finished record's tombstone expires; InfiniteFuture while kActive.
  absl::Time retain_until = absl::InfiniteFuture();
};

// Tracks versioned receive records, tombstones, retention/eviction, and
// pre-claim writes for one receiver.
//
// Thread-compatible: const methods may run concurrently; non-const methods need
// external synchronization.
class VersionedReceives final {
 public:
  // Outcome of checking `uuid` on prepare, before the caller claims the staging
  // buffer.
  enum class PrepareDecision {
    kClaim,     // No live record: caller installs a new claim.
    kJoin,      // Active claim record for `uuid` with matching spec: join.
    kFinished,  // Finished (or committing) record: return `status`, no claim.
    kError,     // Active claim record with mismatched spec.
  };

  struct PrepareLookup {
    PrepareDecision decision = PrepareDecision::kClaim;
    // Non-OK for kFinished and kError; OK for kClaim and kJoin.
    absl::Status status;
    // Finished records whose retention expired during lookup.
    std::vector<uint64_t> evicted_uuids;
  };

  // `max_records` must be at least 3 (one active claim, one committing hooked
  // copy, and at least one finished record), `min_retention` must not be
  // negative, and `max_unclaimed_writes` must be at least 1.
  explicit VersionedReceives(
      size_t max_records = kMaxReceiveRecords,
      absl::Duration min_retention = kMinReceiveRecordRetention,
      size_t max_unclaimed_writes = kMaxUnclaimedWrites);

  // Not copyable: records and unclaimed writes hold iterators into this
  // object's lists, which moves keep valid.
  VersionedReceives(const VersionedReceives&) = delete;
  VersionedReceives& operator=(const VersionedReceives&) = delete;
  VersionedReceives(VersionedReceives&&) = default;
  VersionedReceives& operator=(VersionedReceives&&) = default;

  // Decides whether a prepare of `uuid` with `spec` should claim, join, return
  // a recorded finished outcome, or fail on a version or tiling mismatch.
  // Evicts `uuid` first if it holds an expired finished tombstone.
  PrepareLookup LookupForPrepare(uint64_t uuid, const ReceiveSpec& spec,
                                 absl::Time now);

  // Inserts a new ACTIVE (draining, `ready = false`) record for `uuid` with
  // `spec`, enforcing `max_records_` by sweeping expired finished records and
  // then evicting the oldest finished records. A negative `receive_timeout`
  // counts as zero. Returns evicted uuids so the caller can clean up its other
  // state for them. CHECK-fails if `uuid` is 0, another receive still holds the
  // active claim, or `uuid` already has a live record.
  std::vector<uint64_t> StartActive(uint64_t uuid, ReceiveSpec spec,
                                    absl::Duration receive_timeout,
                                    absl::Time expires_at, absl::Time now);

  // Marks the active-claim record of `uuid` as ready to admit writes and
  // receive mutations. Does nothing unless `uuid` is the active claim.
  void MarkReady(uint64_t uuid);

  // Records that `uuid` committed its staged version and is finishing a pinned
  // hooked copy. Releases the active-claim slot so a newer receive may claim
  // and wait on the copy's pin, while keeping `uuid` in kActive until
  // MarkCommitted() or Finish(). Does nothing unless `uuid` is kActive.
  void MarkCommitting(uint64_t uuid);

  // Transitions an ACTIVE record `uuid` to kCommitted at `now`.
  void MarkCommitted(uint64_t uuid, absl::Time now);

  // Transitions an ACTIVE record `uuid` to `state` (must be kAborted, kExpired,
  // or kFailed) with non-OK `outcome` at `now`, CHECK-failing otherwise. First
  // transition wins. When `allow_committing` is false (default), a record that
  // already committed and is finishing its pinned hooked H2D (`committing ==
  // true`) is not overwritten; only the hooked H2D failure path passes true.
  void Finish(uint64_t uuid, ReceiveState state, absl::Status outcome,
              absl::Time now, bool allow_committing = false);

  // Convenience wrapper mapping non-OK `outcome` to kExpired, kAborted, or
  // kFailed via StateForFailure(outcome).
  void FinishWithFailure(uint64_t uuid, absl::Status outcome, absl::Time now,
                         bool allow_committing = false);

  // Rejects incoming host-buffer writes for finished (or committing) records.
  // Evicts `uuid` into `evicted_uuids` if its finished tombstone expired by
  // `now`. Returns true iff `uuid` holds an active non-committing record, false
  // if `uuid` has no live record (unknown or just evicted), or an error if
  // finished or committing.
  absl::StatusOr<bool> CheckWriteNotTombstoned(
      uint64_t uuid, absl::Time now, std::vector<uint64_t>* evicted_uuids);

  // Status a prepare for a finished (or committing) record must return:
  // FailedPrecondition for committed/committing, else `record.outcome`.
  static absl::Status PrepareStatusForFinished(const ReceiveRecord& record);

  // Status an incoming write for a finished (or committing) record must return:
  // FailedPrecondition for committed/committing, else Aborted naming
  // `record.outcome`.
  static absl::Status WriteRefusalForFinished(const ReceiveRecord& record);

  // Maps a failure status to kExpired (DeadlineExceeded), kAborted (Aborted or
  // Cancelled), or kFailed (all others).
  static ReceiveState StateForFailure(const absl::Status& outcome);

  // Looks up `uuid` without evicting. Returned pointers are invalidated by
  // StartActive() and by evicting `uuid`.
  const ReceiveRecord* Find(uint64_t uuid) const;
  // Looks up `uuid`, evicting it into `evicted_uuids` first if it is finished
  // and past `retain_until`.
  const ReceiveRecord* FindLive(uint64_t uuid, absl::Time now,
                                std::vector<uint64_t>* evicted_uuids);

  // Uuid currently holding the active staging claim; 0 if none.
  uint64_t active_claim_uuid() const { return active_claim_uuid_; }
  // True iff `uuid` holds the active staging claim and is marked ready.
  bool IsReadyActive(uint64_t uuid) const;
  bool Contains(uint64_t uuid) const { return records_.contains(uuid); }
  size_t size() const { return records_.size(); }

  // Tracks uuids with writes admitted while not holding a ready active claim;
  // uuid 0 (untagged pushes) is never tracked. Recording a new uuid at the
  // `max_unclaimed_writes` bound evicts the least recently recorded one, so
  // HasUnclaimedWrite() can return false for a uuid whose entry was evicted.
  void RecordUnclaimedWrite(uint64_t uuid);
  bool HasUnclaimedWrite(uint64_t uuid) const;
  void ClearUnclaimedWrite(uint64_t uuid);

 private:
  // A record plus its position in `finished_order_` once it finished.
  struct Entry {
    ReceiveRecord record;
    std::optional<std::list<uint64_t>::iterator> finished_position;
  };
  using RecordMap = absl::flat_hash_map<uint64_t, Entry>;

  absl::Duration RetentionFor(absl::Duration receive_timeout) const;
  // Evicts record `it` if it is finished and past `retain_until`. Returns
  // whether it did, which invalidates `it`.
  bool EvictIfRetentionExpired(RecordMap::iterator it, absl::Time now,
                               std::vector<uint64_t>* evicted_uuids);
  // Returns the record of `uuid`, which `finished_order_` lists.
  RecordMap::iterator FindFinished(uint64_t uuid);
  // Evicts finished record `it` together with its unclaimed write.
  void EvictFinished(RecordMap::iterator it,
                     std::vector<uint64_t>* evicted_uuids);
  void SweepExpiredFinished(absl::Time now,
                            std::vector<uint64_t>* evicted_uuids);

  size_t max_records_;
  absl::Duration min_retention_;
  size_t max_unclaimed_writes_;
  uint64_t active_claim_uuid_ = 0;
  RecordMap records_;
  // Uuids of the finished records in `records_`, oldest finish first.
  std::list<uint64_t> finished_order_;
  // Uuids with unclaimed writes, least recently recorded first.
  std::list<uint64_t> unclaimed_write_order_;
  // Position of each tracked uuid in `unclaimed_write_order_`.
  absl::flat_hash_map<uint64_t, std::list<uint64_t>::iterator>
      unclaimed_writes_;
};

}  // namespace tpu_raiden::weight_sync

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_VERSIONED_RECEIVES_H_
