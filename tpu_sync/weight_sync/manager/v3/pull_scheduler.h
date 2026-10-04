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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_PULL_SCHEDULER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_PULL_SCHEDULER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tpu_sync/weight_sync/manager/v3/logical_types.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {

// Tuning of a `PullScheduler`.
struct PullSchedulerOptions {
  // Maximum number of leases a puller host holds at a time (k_b), which is
  // also the most grants one reply carries. Clamped to >= 1, and to
  // `max_concurrent_uploads_per_source` unless
  // `allow_grant_batch_above_upload_cap`: with more pulls in flight per host
  // than uploads per source, a few pullers hold most upload slots while the
  // others idle (8x slower syncs in simulation with k_b = 8, c = 1).
  int32_t grant_batch_size = 8;
  // Maximum number of concurrent uploads of a source host (c). Clamped to
  // >= 1.
  int32_t max_concurrent_uploads_per_source = 8;
  // Keep `grant_batch_size` above `max_concurrent_uploads_per_source`.
  bool allow_grant_batch_above_upload_cap = false;
  // A puller host must report every lease within this time. When a lease
  // expires and its puller has no parked request, the puller is presumed dead:
  // all of its leases are reclaimed and it stops serving as a source until it
  // contacts the scheduler again. A host that is not done and has not
  // contacted the scheduler for this long is treated the same way.
  absl::Duration lease_timeout = absl::Seconds(30);
  // A request that cannot be granted anything is parked for at most this long
  // and then answered with no grants.
  absl::Duration long_poll_timeout = absl::Seconds(5);
  // A source is excluded for every puller once this many distinct pullers
  // reported a failed pull from it (K). Clamped to >= 1.
  int32_t unhealthy_source_reports = 3;
  // Most requests a shard applies before it matches waiting pullers and
  // replies. Matching only revisits what changed, so small batches cost
  // little throughput, and they answer the first requests of a burst (e.g.
  // a wave of pull completions) without waiting for the whole burst.
  int32_t max_batch_requests = 16;
  // Process requests on the caller's thread instead of one thread per shard.
  // Replies are then delivered before `Submit()` returns (or from a later
  // `Submit()`, `AdvanceTime()`, `MarkSeeded()` ... call). For tests and
  // simulations.
  bool inline_execution = false;
  // Time source for leases, long polls and liveness (default: `absl::Now`).
  std::function<absl::Time()> clock;
  // Threaded execution: runs first on the thread of each shard (host index),
  // e.g. to set its CPU affinity.
  std::function<void(int32_t host)> shard_thread_init;
};

// `options.grant_batch_size` after clamping: the leases a puller host may
// hold at a time.
int32_t EffectiveGrantBatchSize(const PullSchedulerOptions& options);

// One pull decided by the scheduler: host `h` of the requesting replica pulls
// `bundle_id` from host `h` of `source_replica`.
struct PullGrant {
  int32_t bundle_id = 0;
  int32_t source_replica = 0;
  uint64_t lease_id = 0;
  absl::Time expiry;
};

// A request of one puller host (`replica`, `host`): it reports finished leases
// and asks for up to `max_grants` new ones. `AcquirePulls` is a request
// without reports, `ReportPulls` one with reports (piggybacked: its reply
// carries the next grants).
struct PullRequest {
  int32_t replica = 0;
  int32_t host = 0;
  // Leases whose bundle the host now holds.
  absl::InlinedVector<uint64_t, 8> completed;
  // Leases that failed. The source is excluded for this host.
  absl::InlinedVector<uint64_t, 4> failed;
  // New grants the host can take now (capped by `grant_batch_size` minus its
  // leases in flight). 0 answers right away.
  int32_t max_grants = 0;
  // How long the request may stay parked until something can be granted
  // (capped by `long_poll_timeout`); it is then answered with no grants. Zero
  // answers right away.
  absl::Duration long_poll = absl::InfiniteDuration();
  // The host holds its seeded bundles (the Trainer push to it landed).
  bool seeded = false;
  // The host lost everything it received for this transfer (e.g. it
  // restarted). Its leases are dropped and it pulls every bundle again.
  bool lost_data = false;
};

enum class PullReplyKind {
  // `grants` (possibly empty after a long-poll timeout). Ask again.
  kGrants,
  // The host holds every bundle; it keeps serving as a source.
  kDone,
  // The transfer failed or was cancelled (`status`).
  kAborted,
};

struct PullReply {
  PullReplyKind kind = PullReplyKind::kGrants;
  absl::InlinedVector<PullGrant, 8> grants;
  absl::Status status;
};

// Receives the reply of one request, exactly once. It runs on a scheduler
// thread (or inline) and must not block.
using PullReplyCallback = absl::AnyInvocable<void(PullReply) &&>;

// Counters of one shard (one host index). Monotonic unless noted.
struct PullShardStats {
  int32_t host = 0;
  int64_t requests = 0;
  int64_t grants = 0;
  int64_t completions = 0;
  int64_t failures = 0;
  int64_t expired_leases = 0;
  int64_t long_poll_timeouts = 0;
  int64_t suspended_hosts = 0;
  int64_t unhealthy_sources = 0;
  int64_t batches = 0;
  // Largest number of requests applied in one batch.
  int64_t max_batch_requests = 0;
  // Total time spent processing batches (apply, match, reply).
  int64_t busy_ns = 0;
  // Batch processing time percentiles.
  int64_t batch_p50_ns = 0;
  int64_t batch_p99_ns = 0;
  // Percentiles of the time from `Submit()` to the end of the batch that
  // applied the request (queueing plus processing; threaded execution only).
  // Requests that can be granted something are answered then; the others
  // wait parked for a source.
  int64_t request_p50_ns = 0;
  int64_t request_p99_ns = 0;
  // CPU time of the shard thread (threaded execution only).
  int64_t thread_cpu_ns = 0;
  // Current values.
  int32_t parked_requests = 0;
  int32_t done_hosts = 0;
  int32_t leases_in_flight = 0;
};

// Decides every sampler-to-sampler pull of one transfer at request time.
//
// Host `h` of a replica only pulls from host `h` of another replica (all
// replicas share one shard layout), so the scheduler is split into one
// independent shard per host index, without shared state. Each shard is
// owned by a single thread fed by a lock-free queue: it applies a batch of
// requests (reports first), then matches the waiting pullers against the
// sources with a free upload slot, preferring the rarest bundle (by holders
// plus pulls in flight) and the least loaded source, and replies. State is
// indexed (bitsets per replica, bundles sorted by rarity, sources bucketed by
// load, an intrusive FIFO of waiting pullers), so a decision costs O(1)
// amortized plus short scans. Requests that cannot be served are parked
// entries, not blocked threads.
//
// The transfer completes once every host of every replica holds every bundle.
// It aborts when a bundle that some host still misses has no live holder
// left (and no seed that may still report it).
// TODO(b/570356170): Ask the Trainer to re-push such a bundle instead.
//
// Thread-safe.
class PullScheduler {
 public:
  // |seeded_bundles[r]|: bundles the Trainer pushes to replica `r` (empty for
  // non-seeds). They count as potential sources from the start and become
  // real ones when the host reports `seeded` (or on `MarkSeeded`).
  static absl::StatusOr<std::unique_ptr<PullScheduler>> Create(
      int32_t num_replicas, int32_t num_hosts, int32_t num_bundles,
      std::vector<std::vector<int32_t>> seeded_bundles,
      PullSchedulerOptions options);

  // Answers parked requests with `kAborted` (CancelledError) and stops the
  // shard threads.
  ~PullScheduler();

  PullScheduler(const PullScheduler&) = delete;
  PullScheduler& operator=(const PullScheduler&) = delete;

  // Handles |request| asynchronously and answers it through |done|. Requests
  // of one host are applied in submission order; a new request of a host
  // answers its parked one (with no grants) first. The grants of every reply
  // are leases the host must pull and report, including those of a reply to
  // a request it has since superseded (it may have been granted just before
  // the new request arrived).
  void Submit(PullRequest request, PullReplyCallback done);

  // Every host of |replica| holds its seeded bundles.
  void MarkSeeded(int32_t replica);
  // |replica| is gone for good: none of its hosts serves again.
  void MarkReplicaDead(int32_t replica);
  // Fails the transfer with |status| (if it is still running).
  void Abort(absl::Status status);

  // Inline execution: processes expired leases, long polls and liveness at
  // the current `clock()`. No-op for threaded execution.
  void AdvanceTime();
  // Inline execution: the earliest time `AdvanceTime()` has work to do.
  absl::Time NextTimer() const;

  // OK once every host holds every bundle, the abort status if the transfer
  // failed, or `DeadlineExceededError` if |deadline| passed first.
  absl::Status WaitForCompletion(absl::Time deadline);
  bool complete() const;
  // OK while running or complete, else the abort status.
  absl::Status status() const;

  int32_t num_replicas() const { return num_replicas_; }
  int32_t num_hosts() const { return num_hosts_; }
  int32_t num_bundles() const { return num_bundles_; }
  const PullSchedulerOptions& options() const { return options_; }

  // Bundle of a lease id.
  static int32_t LeaseBundle(uint64_t lease_id);

  // `result[r]`: the bundles |layout| seeds replica `r` with.
  static std::vector<std::vector<int32_t>> SeededBundles(
      const SeedLayout& layout, int32_t num_replicas);

  std::vector<PullShardStats> GetStats() const;

  class Shard;
  struct Shared;

 private:
  PullScheduler(int32_t num_replicas, int32_t num_hosts, int32_t num_bundles,
                PullSchedulerOptions options);

  const int32_t num_replicas_;
  const int32_t num_hosts_;
  const int32_t num_bundles_;
  const PullSchedulerOptions options_;
  std::vector<std::vector<int32_t>> seeded_bundles_;
  std::unique_ptr<Shared> shared_;
  std::vector<std::unique_ptr<Shard>> shards_;
};

}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_PULL_SCHEDULER_H_
