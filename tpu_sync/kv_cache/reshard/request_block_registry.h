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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_REQUEST_BLOCK_REGISTRY_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_REQUEST_BLOCK_REGISTRY_H_

#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/reshard/declaration_types.h"
#include "tpu_sync/kv_cache/reshard/work_unit_directory.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {

// Port of the RaidenController request-block registry: (req_id, uuid)
// generations of per-rank block registrations, the claim → complete
// lifecycle, cancellation tombstones, owner-scoped claim abandonment, and
// TTL purge. Shares the controller-wide mutex with WorkUnitDirectory
// (Python's single self._lock); the injectable clock reproduces
// time.monotonic() for TTL tests.
class RequestBlockRegistry {
 public:
  using Clock = std::function<double()>;

  // Ordered comparator so error strings and sweeps render deterministically
  // (Python dict is insertion-ordered; ordering only affects purge scans,
  // never wire output).
  struct RaidenIdLess {
    bool operator()(const RaidenId& a, const RaidenId& b) const {
      return std::tie(a.job_name, a.job_replica_id, a.data_name,
                      a.data_replica_idx) <
             std::tie(b.job_name, b.job_replica_id, b.data_name,
                      b.data_replica_idx);
    }
  };

  RequestBlockRegistry(WorkUnitDirectory* directory, double ttl_s, Clock clock);

  // register_request_blocks: full validation cascade with byte-identical
  // error strings, then the locked transition.
  absl::Status Register(const std::string& req_id, int64_t uuid,
                        const RaidenId& unit,
                        const std::vector<int64_t>& block_ids,
                        const std::vector<PoolSpanRegistration>& pool_spans);

  // release_request_blocks (force retire). Returns retired row count.
  absl::StatusOr<int> Release(const std::string& req_id, int64_t uuid);

  // complete_request_blocks: one per-rank terminal vote. Returns retired
  // row count (0 until the full claimed set has voted).
  absl::StatusOr<int> Complete(const std::string& req_id, int64_t uuid,
                               const RaidenId& unit);

  // cancel_request_blocks_if_unclaimed.
  absl::StatusOr<bool> CancelIfUnclaimed(const std::string& req_id,
                                         int64_t uuid);

  // _lookup_request_blocks: the claim linearization point. claim_owner is
  // an opaque identity pointer (Python: `object()`).
  absl::StatusOr<std::map<RaidenId, RequestBlockRegistration, RaidenIdLess>>
  LookupAndClaim(const std::string& req_id, int64_t uuid,
                 const std::vector<RaidenId>& units, const void* claim_owner);

  // _abandon_request_blocks_claim: claim rollback before dispatch.
  bool AbandonClaim(const std::string& req_id, int64_t uuid,
                    const void* claim_owner);

  // Worker-restart invalidation (register_work_unit's locked half calls
  // this under the shared mutex).
  void InvalidateUnitLocked(const RaidenId& unit)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Mutex* mu() const ABSL_LOCK_RETURNED(mu_) { return mu_; }

 private:
  using LifecycleKey = std::pair<std::string, int64_t>;
  using BlockKey = std::pair<std::string, RaidenId>;

  struct BlockKeyLess {
    bool operator()(const BlockKey& a, const BlockKey& b) const {
      return std::tie(a.first) < std::tie(b.first) ||
             (a.first == b.first && RaidenIdLess()(a.second, b.second));
    }
  };

  struct Completions {
    std::set<RaidenId, RaidenIdLess> units;
    double expires_at = 0.0;
  };

  void PurgeExpiredRequestBlocksLocked(double now)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void PurgeExpiredLifecycleLocked(double now)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  int RetireLocked(const LifecycleKey& key) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  WorkUnitDirectory* directory_;
  absl::Mutex* mu_;
  const double ttl_s_;
  Clock clock_;

  std::map<BlockKey, RequestBlockRegistration, BlockKeyLess> request_blocks_
      ABSL_GUARDED_BY(mu_);
  std::map<LifecycleKey, double> claimed_ ABSL_GUARDED_BY(mu_);
  std::map<LifecycleKey, std::set<RaidenId, RaidenIdLess>> claimed_units_
      ABSL_GUARDED_BY(mu_);
  std::map<LifecycleKey, const void*> claimed_owners_ ABSL_GUARDED_BY(mu_);
  std::map<LifecycleKey, Completions> completed_units_ ABSL_GUARDED_BY(mu_);
  std::map<LifecycleKey, double> cancelled_ ABSL_GUARDED_BY(mu_);
};

}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RESHARD_REQUEST_BLOCK_REGISTRY_H_
