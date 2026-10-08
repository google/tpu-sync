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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_FAKE_SAMPLER_FLEET_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_FAKE_SAMPLER_FLEET_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "tpu_sync/weight_sync/manager/v3/logical_types.h"
#include "tpu_sync/weight_sync/manager/v3/pull_scheduler.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {

struct FakeSamplerFleetOptions {
  // NIC bandwidth of every host in each direction, in bundle bytes per
  // second (a bundle's bytes are what one host of a replica receives).
  double bandwidth = 1.0;
  // Per-replica overrides of `bandwidth` (empty: no override).
  std::vector<double> download_bandwidth;
  std::vector<double> upload_bandwidth;
  // Bandwidth of one Trainer push stream. 0: every seed holds its stripe at
  // time 0.
  double trainer_stream_bandwidth = 0;
  // Replica -> time at which it dies for good: it stops serving and pulling.
  absl::flat_hash_map<int32_t, double> dead_at;
  // Delay after which a death is reported to the scheduler (as the
  // controller's health checks would); negative: never (the scheduler only
  // notices through lease expiry and liveness).
  double death_detection_delay = -1;
  // Replica -> time at which it restarts: it loses its data, is down for
  // `restart_downtime` and then asks again with `lost_data`.
  absl::flat_hash_map<int32_t, double> restart_at;
  double restart_downtime = 0.5;
  // A puller gives up on a pull that would take longer than this and
  // reports it failed (0: never), which is how slow sources get avoided.
  double pull_timeout = 0;
};

struct FakeFleetResult {
  // OK if every host of every replica that did not die completed, else the
  // abort status the hosts received (or an error if the simulation got
  // stuck).
  absl::Status status;
  // Time at which the last completing replica completed.
  double finish_time = 0;
  // Per replica: time at which its last host completed, or -1.
  std::vector<double> replica_finish_time;
  int32_t completed_replicas = 0;
  int64_t requests = 0;
  int64_t grants = 0;
  int64_t failed_pulls = 0;
  int64_t rejoins = 0;
  // Largest number of concurrent uploads of any source host.
  int32_t max_concurrent_uploads = 0;
  // Per replica: pulls it served (summed over its hosts).
  std::vector<int64_t> uploads_per_replica;
  // Wall-clock duration of the simulation.
  double wall_seconds = 0;
  // `SimulateScheduledPulls` only: the scheduler's view at the end.
  bool scheduler_complete = false;
  absl::Status scheduler_status;
  std::vector<PullShardStats> scheduler_stats;
};

// Transfer shape the fleet simulates.
struct FakeFleetTopology {
  int32_t num_replicas = 0;
  int32_t num_hosts = 1;
  // Bytes of every bundle (per host).
  std::vector<double> bundle_bytes;
  SeedLayout seed_layout;
  // Must match the scheduler's options.
  int32_t grant_batch_size = 1;
  int32_t max_concurrent_uploads_per_source = 1;
};

// How the fleet reaches the scheduler: |submit| delivers a request whose
// reply arrives through the callback (inline, in virtual time),
// |advance_time| processes the scheduler's timers at the fleet's current time
// and |next_timer| says when that is next needed.
struct FakeFleetDriver {
  std::function<void(PullRequest, PullReplyCallback)> submit;
  std::function<void()> advance_time;
  std::function<absl::Time()> next_timer;
  // Optional: reports a replica death (see `death_detection_delay`).
  std::function<void(int32_t)> mark_dead;
  // Optional: the scheduler's statistics, for diagnostics.
  std::function<std::vector<PullShardStats>()> stats;
};

// In-process stand-in for the sampler hosts of a transfer that pull under a
// `PullScheduler`, for tests and as a simulator. It runs in virtual time and
// is deterministic. Every host keeps up to `grant_batch_size` pulls in flight
// and reports each finished pull in its next request (piggybacked); a
// pull runs at `min(download / grant_batch_size, upload / uploads cap)` of
// the two hosts' bandwidths. Seeds report `seeded` once their Trainer stripe
// landed. A pull from a dead source fails at once, one that would outlast
// `pull_timeout` fails then; restarted replicas ask again with `lost_data`.
// The scheduler must run inline on `clock()`.
class FakeSamplerFleet {
 public:
  explicit FakeSamplerFleet(FakeSamplerFleetOptions options)
      : options_(std::move(options)) {}

  // Virtual time, for `PullSchedulerOptions::clock`.
  std::function<absl::Time()> clock() const {
    return [this] { return Now(); };
  }
  absl::Time Now() const { return absl::UnixEpoch() + absl::Seconds(now_); }

  // Simulates the pull phase from the start of the Trainer push.
  FakeFleetResult Run(const FakeFleetTopology& topology,
                      const FakeFleetDriver& driver);

 private:
  const FakeSamplerFleetOptions options_;
  double now_ = 0;
};

// Runs a `FakeSamplerFleet` against a fresh inline `PullScheduler` created
// with |scheduler_options| (its clock is replaced by the fleet's).
absl::StatusOr<FakeFleetResult> SimulateScheduledPulls(
    const FakeFleetTopology& topology, PullSchedulerOptions scheduler_options,
    FakeSamplerFleetOptions fleet_options);

}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_MANAGER_V3_FAKE_SAMPLER_FLEET_H_
