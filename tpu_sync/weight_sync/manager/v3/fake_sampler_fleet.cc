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

#include "tpu_sync/weight_sync/manager/v3/fake_sampler_fleet.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/weight_sync/manager/v3/logical_types.h"
#include "tpu_sync/weight_sync/manager/v3/pull_scheduler.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kEps = 1e-9;
// Bound on simulation steps, against bugs that would never make progress.
constexpr int64_t kMaxSteps = 200'000'000;
// Steps with only scheduler timers (long polls, liveness) after which the
// simulation counts as stuck.
constexpr int64_t kMaxIdleSteps = 100'000;

enum class EventKind { kPullEnd, kSeed, kDie, kReportDead, kRestart, kRejoin };

struct Event {
  double time = 0;
  EventKind kind = EventKind::kPullEnd;
  int32_t replica = 0;
  int32_t host = 0;
  uint64_t lease = 0;
  // Ordering tie-breaker, for determinism.
  int64_t order = 0;

  bool operator>(const Event& o) const {
    return time != o.time ? time > o.time : order > o.order;
  }
};

struct Pull {
  uint64_t lease = 0;
  int32_t bundle = 0;
  int32_t source = 0;
  // The pull ends by failing (timeout).
  bool fails = false;
};

struct HostSim {
  std::vector<uint8_t> holds;
  absl::InlinedVector<Pull, 8> pulls;
  int32_t uploads = 0;
  uint32_t seq = 0;
  bool outstanding = false;
  bool done = false;
  absl::InlinedVector<uint64_t, 8> completed;
  absl::InlinedVector<uint64_t, 4> failed;
  bool report_seeded = false;
  bool report_lost_data = false;
  bool dirty = false;
};

struct ReplicaSim {
  bool dead = false;
  bool down = false;
  bool restarted = false;
  double down_bw = 1;
  double up_bw = 1;
};

struct PendingReply {
  int32_t replica = 0;
  int32_t host = 0;
  uint32_t seq = 0;
  PullReply reply;
};

}  // namespace

FakeFleetResult FakeSamplerFleet::Run(const FakeFleetTopology& topology,
                                      const FakeFleetDriver& driver) {
  const absl::Time wall_start = absl::Now();
  const int32_t num_dst = topology.num_replicas;
  const int32_t num_hosts = std::max(1, topology.num_hosts);
  const int32_t num_bundles =
      static_cast<int32_t>(topology.bundle_bytes.size());
  const int32_t grant_batch = std::max(1, topology.grant_batch_size);
  const int32_t upload_cap =
      std::max(1, topology.max_concurrent_uploads_per_source);
  const SeedLayout& layout = topology.seed_layout;
  const std::vector<std::vector<int32_t>> seeded =
      PullScheduler::SeededBundles(layout, num_dst);
  now_ = 0;

  FakeFleetResult result;
  result.replica_finish_time.assign(num_dst, -1);
  result.uploads_per_replica.assign(num_dst, 0);
  std::vector<ReplicaSim> reps(num_dst);
  std::vector<HostSim> hosts(static_cast<size_t>(num_dst) * num_hosts);
  auto host_of = [&](int32_t r, int32_t h) -> HostSim& {
    return hosts[static_cast<size_t>(r) * num_hosts + h];
  };
  for (int32_t r = 0; r < num_dst; ++r) {
    ReplicaSim& rep = reps[r];
    rep.down_bw = r < static_cast<int32_t>(options_.download_bandwidth.size())
                      ? options_.download_bandwidth[r]
                      : options_.bandwidth;
    rep.up_bw = r < static_cast<int32_t>(options_.upload_bandwidth.size())
                    ? options_.upload_bandwidth[r]
                    : options_.bandwidth;
    for (int32_t h = 0; h < num_hosts; ++h) {
      host_of(r, h).holds.assign(num_bundles, 0);
    }
  }

  std::priority_queue<Event, std::vector<Event>, std::greater<Event>> events;
  int64_t order = 0;
  auto push = [&](Event e) {
    e.order = order++;
    events.push(e);
  };
  // Trainer waves: the pushes of a wave run concurrently, one stream each.
  std::vector<double> seed_time(num_dst, kInf);
  double wave_start = 0;
  for (const auto& wave : layout.trainer_waves) {
    double wave_end = wave_start;
    for (const auto& [seed, stripe] : wave) {
      if (seed < 0 || seed >= num_dst || stripe < 0 ||
          stripe >= static_cast<int32_t>(layout.stripe_bundles.size())) {
        continue;
      }
      double stripe_bytes = 0;
      for (int32_t b : layout.stripe_bundles[stripe]) {
        if (b >= 0 && b < num_bundles) stripe_bytes += topology.bundle_bytes[b];
      }
      const double done =
          options_.trainer_stream_bandwidth > 0
              ? wave_start + stripe_bytes / options_.trainer_stream_bandwidth
              : 0;
      wave_end = std::max(wave_end, done);
      seed_time[seed] = std::min(seed_time[seed], done);
    }
    wave_start = wave_end;
  }
  for (int32_t r = 0; r < num_dst; ++r) {
    if (!seeded[r].empty() && seed_time[r] > 0 && seed_time[r] < kInf) {
      push(Event{.time = seed_time[r], .kind = EventKind::kSeed, .replica = r});
    }
  }
  for (const auto& [r, t] : options_.dead_at) {
    if (r >= 0 && r < num_dst) {
      push(Event{.time = t, .kind = EventKind::kDie, .replica = r});
    }
  }
  for (const auto& [r, t] : options_.restart_at) {
    if (r >= 0 && r < num_dst) {
      push(Event{.time = t, .kind = EventKind::kRestart, .replica = r});
    }
  }

  std::vector<PendingReply> pending;
  std::vector<std::pair<int32_t, int32_t>> dirty;
  bool aborted = false;
  int32_t hosts_done = 0;
  auto mark_dirty = [&](int32_t r, int32_t h) {
    HostSim& x = host_of(r, h);
    if (x.dirty) return;
    x.dirty = true;
    dirty.emplace_back(r, h);
  };
  auto send = [&](int32_t r, int32_t h) {
    HostSim& x = host_of(r, h);
    PullRequest req;
    req.replica = r;
    req.host = h;
    req.completed = std::move(x.completed);
    req.failed = std::move(x.failed);
    x.completed.clear();
    x.failed.clear();
    req.max_grants = grant_batch - static_cast<int32_t>(x.pulls.size());
    req.seeded = x.report_seeded;
    req.lost_data = x.report_lost_data;
    x.report_seeded = false;
    x.report_lost_data = false;
    const uint32_t seq = ++x.seq;
    x.outstanding = true;
    ++result.requests;
    driver.submit(std::move(req), [&pending, r, h, seq](PullReply reply) {
      pending.push_back(PendingReply{
          .replica = r, .host = h, .seq = seq, .reply = std::move(reply)});
    });
  };
  // Ends the pulls of hosts of |r| as a puller (abandoned) and fails the pulls
  // it serves.
  auto stop_replica = [&](int32_t r) {
    for (int32_t h = 0; h < num_hosts; ++h) {
      HostSim& x = host_of(r, h);
      for (const Pull& pull : x.pulls) --host_of(pull.source, h).uploads;
      x.pulls.clear();
      x.completed.clear();
      x.failed.clear();
      x.outstanding = false;
    }
    for (int32_t p = 0; p < num_dst; ++p) {
      for (int32_t h = 0; h < num_hosts; ++h) {
        HostSim& x = host_of(p, h);
        for (size_t i = 0; i < x.pulls.size();) {
          if (x.pulls[i].source == r) {
            --host_of(r, h).uploads;
            ++result.failed_pulls;
            x.failed.push_back(x.pulls[i].lease);
            x.pulls.erase(x.pulls.begin() + i);
            mark_dirty(p, h);
          } else {
            ++i;
          }
        }
      }
    }
  };
  auto handle = [&](PendingReply& pr) {
    const int32_t r = pr.replica;
    const int32_t h = pr.host;
    HostSim& x = host_of(r, h);
    ReplicaSim& rep = reps[r];
    if (rep.dead || rep.down) return;
    // A reply to a superseded request still carries real grants.
    if (pr.seq == x.seq) x.outstanding = false;
    switch (pr.reply.kind) {
      case PullReplyKind::kAborted:
        if (result.status.ok()) result.status = pr.reply.status;
        aborted = true;
        return;
      case PullReplyKind::kDone:
        if (!x.done) {
          x.done = true;
          ++hosts_done;
          bool all = true;
          for (int32_t k = 0; k < num_hosts; ++k) all &= host_of(r, k).done;
          if (all) result.replica_finish_time[r] = now_;
        }
        return;
      case PullReplyKind::kGrants:
        break;
    }
    for (const PullGrant& g : pr.reply.grants) {
      ++result.grants;
      const int32_t s = g.source_replica;
      const int32_t b = g.bundle_id;
      if (s < 0 || s >= num_dst || b < 0 || b >= num_bundles) {
        x.failed.push_back(g.lease_id);
        mark_dirty(r, h);
        continue;
      }
      HostSim& sx = host_of(s, h);
      if (reps[s].dead || reps[s].down || !sx.holds[b]) {
        ++result.failed_pulls;
        x.failed.push_back(g.lease_id);
        mark_dirty(r, h);
        continue;
      }
      ++sx.uploads;
      ++result.uploads_per_replica[s];
      result.max_concurrent_uploads =
          std::max(result.max_concurrent_uploads, sx.uploads);
      const double rate =
          std::min(rep.down_bw / grant_batch, reps[s].up_bw / upload_cap);
      const double duration = topology.bundle_bytes[b] / rate;
      const bool fails =
          options_.pull_timeout > 0 && duration > options_.pull_timeout;
      x.pulls.push_back(
          Pull{.lease = g.lease_id, .bundle = b, .source = s, .fails = fails});
      push(Event{.time = now_ + (fails ? options_.pull_timeout : duration),
                 .kind = EventKind::kPullEnd,
                 .replica = r,
                 .host = h,
                 .lease = g.lease_id});
    }
    if (!x.done && static_cast<int32_t>(x.pulls.size()) < grant_batch) {
      mark_dirty(r, h);
    }
  };
  // Delivers replies and sends requests until nothing changes. Returns false
  // if the hosts keep asking without anything changing (a livelock).
  auto settle = [&]() {
    const int64_t max_requests =
        result.requests + 64 * int64_t{num_dst} * num_hosts * grant_batch +
        1024;
    while (!pending.empty() || !dirty.empty()) {
      if (result.requests > max_requests) {
        std::string what;
        for (size_t i = 0; i < pending.size() && i < 4; ++i) {
          const PendingReply& pr = pending[i];
          const HostSim& x = host_of(pr.replica, pr.host);
          absl::StrAppend(&what, " [replica ", pr.replica, " host ", pr.host,
                          " reply kind ", static_cast<int>(pr.reply.kind),
                          " grants ", pr.reply.grants.size(), " seq ", pr.seq,
                          "/", x.seq, " pulls ", x.pulls.size(), " done ",
                          x.done, "]");
        }
        if (driver.stats) {
          for (const PullShardStats& s : driver.stats()) {
            int64_t sim_pulls = 0;
            for (int32_t r = 0; r < num_dst; ++r) {
              if (s.host < num_hosts)
                sim_pulls += host_of(r, s.host).pulls.size();
            }
            absl::StrAppend(&what, " [shard ", s.host, " leases ",
                            s.leases_in_flight, " vs pulls ", sim_pulls,
                            " parked ", s.parked_requests, "]");
          }
        }
        result.status = absl::InternalError(
            absl::StrCat("Livelock at time ", now_, ":", what));
        return false;
      }
      while (!pending.empty()) {
        std::vector<PendingReply> batch;
        batch.swap(pending);
        for (PendingReply& pr : batch) handle(pr);
      }
      std::vector<std::pair<int32_t, int32_t>> to_send;
      to_send.swap(dirty);
      for (const auto& [r, h] : to_send) {
        HostSim& x = host_of(r, h);
        x.dirty = false;
        if (aborted || reps[r].dead || reps[r].down) continue;
        const bool reports = !x.completed.empty() || !x.failed.empty() ||
                             x.report_seeded || x.report_lost_data;
        const bool wants =
            !x.done && static_cast<int32_t>(x.pulls.size()) < grant_batch;
        if (reports || (wants && !x.outstanding)) send(r, h);
      }
    }
    return true;
  };

  // Every host asks at time 0 (seeds that hold their stripe already say so).
  for (int32_t r = 0; r < num_dst; ++r) {
    for (int32_t h = 0; h < num_hosts; ++h) {
      HostSim& x = host_of(r, h);
      if (!seeded[r].empty() && seed_time[r] <= 0) {
        for (int32_t b : seeded[r]) x.holds[b] = 1;
        x.report_seeded = true;
      }
      mark_dirty(r, h);
    }
  }

  const int32_t total_hosts = num_dst * num_hosts;
  // Hosts of dead replicas that had not completed.
  int32_t dead_hosts = 0;
  // Consecutive steps with nothing but scheduler timers.
  int64_t idle_steps = 0;
  for (int64_t steps = 0; steps < kMaxSteps; ++steps) {
    if (!settle()) break;
    if (aborted || hosts_done + dead_hosts == total_hosts) break;
    const double timer =
        absl::ToDoubleSeconds(driver.next_timer() - absl::UnixEpoch());
    const double next =
        std::min(events.empty() ? kInf : events.top().time, timer);
    idle_steps = events.empty() ? idle_steps + 1 : 0;
    if (next == kInf || idle_steps > kMaxIdleSteps) {
      result.status = absl::InternalError(
          absl::StrCat("Simulation stuck at time ", now_, " with ", hosts_done,
                       " of ", total_hosts - dead_hosts, " live hosts done"));
      break;
    }
    now_ = std::max(now_, next);
    if (timer <= now_ + kEps) driver.advance_time();
    while (!events.empty() && events.top().time <= now_ + kEps) {
      const Event e = events.top();
      events.pop();
      ReplicaSim& rep = reps[e.replica];
      switch (e.kind) {
        case EventKind::kPullEnd: {
          HostSim& x = host_of(e.replica, e.host);
          auto it =
              std::find_if(x.pulls.begin(), x.pulls.end(),
                           [&](const Pull& p) { return p.lease == e.lease; });
          if (it == x.pulls.end()) break;
          --host_of(it->source, e.host).uploads;
          if (it->fails) {
            ++result.failed_pulls;
            x.failed.push_back(it->lease);
          } else {
            x.holds[it->bundle] = 1;
            x.completed.push_back(it->lease);
          }
          x.pulls.erase(it);
          mark_dirty(e.replica, e.host);
          break;
        }
        case EventKind::kSeed:
          if (rep.dead || rep.down || rep.restarted) break;
          for (int32_t h = 0; h < num_hosts; ++h) {
            HostSim& x = host_of(e.replica, h);
            for (int32_t b : seeded[e.replica]) x.holds[b] = 1;
            x.report_seeded = true;
            mark_dirty(e.replica, h);
          }
          break;
        case EventKind::kDie:
          if (rep.dead) break;
          rep.dead = true;
          stop_replica(e.replica);
          for (int32_t h = 0; h < num_hosts; ++h) {
            if (!host_of(e.replica, h).done) ++dead_hosts;
          }
          if (options_.death_detection_delay >= 0 && driver.mark_dead) {
            push(Event{.time = now_ + options_.death_detection_delay,
                       .kind = EventKind::kReportDead,
                       .replica = e.replica});
          }
          break;
        case EventKind::kReportDead:
          driver.mark_dead(e.replica);
          break;
        case EventKind::kRestart:
          if (rep.dead) break;
          rep.down = true;
          rep.restarted = true;
          stop_replica(e.replica);
          for (int32_t h = 0; h < num_hosts; ++h) {
            HostSim& x = host_of(e.replica, h);
            std::fill(x.holds.begin(), x.holds.end(), 0);
            if (x.done) {
              x.done = false;
              --hosts_done;
            }
          }
          result.replica_finish_time[e.replica] = -1;
          push(Event{.time = now_ + options_.restart_downtime,
                     .kind = EventKind::kRejoin,
                     .replica = e.replica});
          break;
        case EventKind::kRejoin:
          if (!rep.down) break;
          rep.down = false;
          ++result.rejoins;
          for (int32_t h = 0; h < num_hosts; ++h) {
            host_of(e.replica, h).report_lost_data = true;
            mark_dirty(e.replica, h);
          }
          break;
      }
    }
  }
  for (int32_t r = 0; r < num_dst; ++r) {
    if (result.replica_finish_time[r] >= 0) {
      ++result.completed_replicas;
      result.finish_time =
          std::max(result.finish_time, result.replica_finish_time[r]);
    }
  }
  if (result.status.ok() && hosts_done + dead_hosts != total_hosts) {
    result.status = absl::InternalError(absl::StrCat(
        hosts_done, " of ", total_hosts - dead_hosts, " live hosts completed"));
  }
  result.wall_seconds = absl::ToDoubleSeconds(absl::Now() - wall_start);
  return result;
}

absl::StatusOr<FakeFleetResult> SimulateScheduledPulls(
    const FakeFleetTopology& topology, PullSchedulerOptions scheduler_options,
    FakeSamplerFleetOptions fleet_options) {
  FakeSamplerFleet fleet(std::move(fleet_options));
  scheduler_options.inline_execution = true;
  scheduler_options.clock = fleet.clock();
  FakeFleetTopology topo = topology;
  topo.grant_batch_size = EffectiveGrantBatchSize(scheduler_options);
  topo.max_concurrent_uploads_per_source =
      scheduler_options.max_concurrent_uploads_per_source;
  absl::StatusOr<std::unique_ptr<PullScheduler>> scheduler =
      PullScheduler::Create(
          topo.num_replicas, topo.num_hosts,
          static_cast<int32_t>(topo.bundle_bytes.size()),
          PullScheduler::SeededBundles(topo.seed_layout, topo.num_replicas),
          scheduler_options);
  if (!scheduler.ok()) return scheduler.status();
  PullScheduler* s = scheduler->get();
  FakeFleetDriver driver{
      .submit =
          [s](PullRequest req, PullReplyCallback done) {
            s->Submit(std::move(req), std::move(done));
          },
      .advance_time = [s] { s->AdvanceTime(); },
      .next_timer = [s] { return s->NextTimer(); },
      .mark_dead = [s](int32_t replica) { s->MarkReplicaDead(replica); },
      .stats = [s] { return s->GetStats(); },
  };
  FakeFleetResult result = fleet.Run(topo, driver);
  result.scheduler_complete = s->complete();
  result.scheduler_status = s->status();
  result.scheduler_stats = s->GetStats();
  return result;
}

}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden
