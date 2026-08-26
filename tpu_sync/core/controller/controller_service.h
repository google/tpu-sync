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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_CONTROLLER_SERVICE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_CONTROLLER_SERVICE_H_

#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/core/buffer.h"
#include "tpu_sync/core/controller/worker_registry.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/proto/controller_service.grpc.pb.h"
#include "tpu_sync/proto/controller_service.pb.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace core {
namespace controller {

class RaidenControllerServiceImpl final
    : public ::tpu_sync::proto::RaidenControllerService::Service {
 public:
  explicit RaidenControllerServiceImpl(
      std::shared_ptr<WorkerRegistry> worker_registry = nullptr);
  // Stops the sweeper and unpins every live lease.
  ~RaidenControllerServiceImpl() override;

  // Disallow copy and assign
  RaidenControllerServiceImpl(const RaidenControllerServiceImpl&) = delete;
  RaidenControllerServiceImpl& operator=(const RaidenControllerServiceImpl&) =
      delete;

  grpc::Status RegisterWorker(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::RegisterWorkerRequest* request,
      ::tpu_sync::proto::RegisterWorkerResponse* response) override;

  // --- Read-lease protocol (source side) ---------------------------------
  //
  // See controller_service.proto for the three-phase contract. Two rules carry
  // the whole guarantee, and both live here:
  //
  //   1. The sweeper MARKS, it does not forget. On expiry it unpins (that is
  //      the point of the TTL) but retains the record as revoked for a
  //      retention window, so a late release learns REVOKED rather than
  //      UNKNOWN. This buys diagnosability, not safety -- UNKNOWN is an
  //      equally mandatory discard.
  //   2. Release is by lease_id and idempotent. Two concurrent readers of the
  //      same hashes hold INDEPENDENT leases over independent pin counts (the
  //      LRU pin is a refcount, so this composes). Never dedupe leases by hash
  //      set: that recreates the release-steals-the-other-reader's-pin bug.
  grpc::Status AcquireReadLease(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::AcquireReadLeaseRequest* request,
      ::tpu_sync::proto::AcquireReadLeaseResponse* response) override;

  grpc::Status RenewReadLease(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::RenewReadLeaseRequest* request,
      ::tpu_sync::proto::RenewReadLeaseResponse* response) override;

  grpc::Status ReleaseReadLease(
      grpc::ServerContext* context,
      const ::tpu_sync::proto::ReleaseReadLeaseRequest* request,
      ::tpu_sync::proto::ReleaseReadLeaseResponse* response) override;

  // Forces `lease_id` to expire immediately, exactly as the sweeper would.
  // Test seam: the lease TTL floor (30 s) is far longer than any unit test,
  // and there is no injectable clock. Returns false if the lease is unknown or
  // already not live.
  bool ForceExpireForTest(uint64_t lease_id);

  // Invoked (outside the service mutex) immediately after a lease is granted.
  // Test seam for the one case a mocked transfer cannot otherwise produce: a
  // transfer that SUCCEEDS while the lease is revoked underneath it, which is
  // precisely what the commit-time verdict exists to catch.
  void SetLeaseGrantedHookForTest(absl::AnyInvocable<void(uint64_t)> hook) {
    absl::MutexLock lock(mutex_);
    lease_granted_hook_ =
        std::make_shared<absl::AnyInvocable<void(uint64_t)>>(std::move(hook));
  }

  // Number of lease records currently retained (live + revoked/released but
  // not yet aged out). Test observability only.
  size_t LeaseCountForTest() const;

  // Total blocks currently leased to `requester`, for the per-peer pressure
  // metric. A quota is deliberately NOT enforced (see rev3 A11a): leases are
  // held only for the span of one read, so the count decreases as transfers
  // complete. This exists so the question is answerable with data.
  int64_t LeasedBlocksForPeerForTest(absl::string_view requester_key) const;

  // ReadRemote All-or-Nothing validate & pin block hashes at the src controller hooks: verify the requested block hashes exist in the
  // source store's LRU with status HOST/HOST_AND_HBM, pin them, and return the
  // authoritative source host_block_ids (re-derived from the LRU). Returns
  // NotFound (missing hash) / FailedPrecondition (wrong status) on failure.
  using ValidateAndPinCallback = absl::AnyInvocable<
      absl::StatusOr<std::vector<int32_t>>(
          absl::Span<const std::string> block_hashes) const>;
  using UnpinCallback =
      absl::AnyInvocable<void(absl::Span<const std::string> block_hashes) const>;

  void SetReadRemoteHooks(ValidateAndPinCallback validate_and_pin,
                          UnpinCallback unpin) {
    absl::MutexLock lock(mutex_);
    validate_and_pin_cb_ =
        std::make_shared<ValidateAndPinCallback>(std::move(validate_and_pin));
    unpin_cb_ = std::make_shared<UnpinCallback>(std::move(unpin));
  }

  // Detaches the store hooks. MUST be called before the store is destroyed if
  // the store does not outlive this service: unlike the push path -- where the
  // callback window was bounded by a live RPC -- the lease sweeper can invoke
  // the unpin hook minutes after the grant, and the destructor invokes it at
  // teardown. Both would use-after-free a dead store.
  void ClearReadRemoteHooks();

  // Updates the underlying WorkerRegistry.
  void SetWorkerRegistry(std::shared_ptr<WorkerRegistry> worker_registry);

  // Returns the current WorkerRegistry.
  std::shared_ptr<WorkerRegistry> worker_registry() const;

  // Lease timing. `retention > deadline > ttl` must hold across the pair of
  // hosts (the destination owns the deadline, see raiden_controller.h):
  //   * deadline > ttl       -- a pull that outlives its lease still gets to
  //                             release and LEARN that it was revoked.
  //   * retention > deadline -- a destination still legitimately waiting can
  //                             never be told UNKNOWN about a lease that
  //                             merely expired.
  // These are compared on two clocks, so the invariant assumes bounded clock
  // RATE skew, not synchronised clocks; the margins absorb any plausible
  // drift. (The commit gate itself stays clock-free by design.)
  static absl::Duration DefaultLeaseTtl();      // RAIDEN_REMOTE_LOCK_TTL_S
  static absl::Duration MinLeaseTtl();          // clamp floor: 30 s
  static absl::Duration MaxLeaseTtl();          // clamp ceiling: 600 s
  static absl::Duration RevokedLeaseRetention();  // RAIDEN_REVOKED_LEASE_RETENTION_S

 private:
  struct ReadLease {
    std::vector<std::string> block_hashes;
    std::string requester_key;  // for the per-peer metric
    absl::Time expiration;
    // A lease is "live" exactly while it holds pins. Release and the sweeper
    // both transition it out of live under mutex_, and only the transitioning
    // party unpins -- which is how the pin count can never underflow.
    bool live = true;
    bool revoked = false;  // sweeper reclaimed the pins (vs. clean release)
    absl::Time retire_at;  // record erased past this
  };

  // Marks `lease_id` not-live and returns its hashes to unpin, or empty if it
  // was already not-live (someone else got there first). Caller MUST invoke
  // the unpin hook OUTSIDE mutex_ -- the hook takes the store's mutex, and
  // taking store-after-service here while the store takes service-after-store
  // elsewhere would close a lock cycle. Marking first is what keeps the gap
  // honest: a Renew landing in it sees the mark and answers REVOKED, never
  // HELD on pins that are about to drop.
  std::vector<std::string> TransitionOutOfLive(uint64_t lease_id, bool revoked)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  void StartSweeperIfNeeded() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void SweeperLoop();
  void RunSweepOnce();

  mutable absl::Mutex mutex_;
  std::shared_ptr<WorkerRegistry> worker_registry_ ABSL_GUARDED_BY(mutex_);
  std::shared_ptr<const ValidateAndPinCallback> validate_and_pin_cb_
      ABSL_GUARDED_BY(mutex_);
  std::shared_ptr<const UnpinCallback> unpin_cb_ ABSL_GUARDED_BY(mutex_);

  absl::flat_hash_map<uint64_t, ReadLease> read_leases_ ABSL_GUARDED_BY(mutex_);
  std::shared_ptr<absl::AnyInvocable<void(uint64_t)>> lease_granted_hook_
      ABSL_GUARDED_BY(mutex_);
  absl::BitGen lease_id_gen_ ABSL_GUARDED_BY(mutex_);

  std::unique_ptr<std::thread> sweeper_ ABSL_GUARDED_BY(mutex_);
  bool sweeper_stop_ ABSL_GUARDED_BY(mutex_) = false;
  mutable absl::CondVar sweeper_cv_;
};

}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_CONTROLLER_SERVICE_H_
