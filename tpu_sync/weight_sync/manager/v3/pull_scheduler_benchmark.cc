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

// Benchmark of `PullScheduler`: a synthetic fleet of D replicas x H hosts
// drives the scheduler in process.
//
//   saturate: every grant completes instantly, so the clients ask again as
//     fast as the scheduler answers. Measures the scheduler's capacity
//     (decisions/s), request and batch latency, batch sizes and CPU per
//     decision.
//   realtime: pulls take wall-clock time (`--bundle_time_ms` per bundle at a
//     host's full ingress; with k_b pulls in flight per host and c uploads
//     per source each pull runs at `1 / max(k_b, c)` of it). Measures the full
//     sync time against the ingress lower bound under a real request load.
//     With `--bundle_time_ms=4` the load is 10x the production demand
//     (4 GB bundles, 25 GB/s per host: 40 ms per bundle).
//   simulate: the virtual-time `FakeSamplerFleet` (no wall clock): full sync
//     time in bundle times against the ingress lower bound.
//
// Columns: kb_r/kb = requested and effective k_b (clamped to c unless
// --allow_batch_above_cap); dec/s = grants per second (aggregate, and of the
// slowest shard); req50/99 = scheduler-side time from `Submit()` to the end
// of the batch that applied the request; wait50/99 = client-side time from
// `Submit()` to a reply with grants (includes waiting parked for a source);
// bat50/99 = batch processing time; cpu_ns = shard thread CPU per decision;
// proc_ns = process CPU (scheduler and load generator) per decision;
// sync/lb = sync time over the ingress lower bound.
//
// Run optimized:
//   blaze run -c opt :pull_scheduler_benchmark -- --mode=saturate,realtime

#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/weight_sync/manager/v3/fake_sampler_fleet.h"
#include "tpu_sync/weight_sync/manager/v3/logical_reshard_planner.h"
#include "tpu_sync/weight_sync/manager/v3/logical_types.h"
#include "tpu_sync/weight_sync/manager/v3/pull_scheduler.h"

ABSL_FLAG(std::string, mode, "saturate,realtime,simulate",
          "Comma-separated modes: saturate, realtime, simulate.");
ABSL_FLAG(int32_t, replicas, 1024, "Destination replicas (D).");
ABSL_FLAG(int32_t, hosts, 4, "Hosts per replica (H).");
ABSL_FLAG(std::string, bundles, "128,512,2048", "Bundle counts (B).");
ABSL_FLAG(std::string, grant_batch, "1,8", "Grant batch sizes (k_b).");
ABSL_FLAG(int32_t, upload_cap, 0,
          "Concurrent uploads per source (c); 0 uses k_b.");
ABSL_FLAG(bool, allow_batch_above_cap, false,
          "Keep k_b above c instead of clamping it to c.");
ABSL_FLAG(int32_t, seed_replication, 2, "Seed replication (R).");
ABSL_FLAG(int32_t, max_batch, 0,
          "Most requests a shard applies per batch; 0 uses the default.");
ABSL_FLAG(int32_t, client_threads, 4, "Load generator threads.");
ABSL_FLAG(bool, pin_threads, true,
          "Pin shard threads to CPUs [0, H) and load generator threads to "
          "the other CPUs.");
ABSL_FLAG(double, min_seconds, 2.0,
          "saturate: repeat full syncs for at least this long.");
ABSL_FLAG(double, bundle_time_ms, 4.0,
          "realtime: time of one bundle at a host's full ingress.");

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {
namespace {

std::vector<int32_t> ParseList(absl::string_view s) {
  std::vector<int32_t> out;
  for (absl::string_view part : absl::StrSplit(s, ',', absl::SkipEmpty())) {
    int32_t v = 0;
    CHECK(absl::SimpleAtoi(part, &v)) << part;
    out.push_back(v);
  }
  return out;
}

int64_t ProcessCpuNanos() {
  rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  auto ns = [](const timeval& tv) {
    return static_cast<int64_t>(tv.tv_sec) * 1'000'000'000 +
           static_cast<int64_t>(tv.tv_usec) * 1000;
  };
  return ns(ru.ru_utime) + ns(ru.ru_stime);
}

std::string CpuModel() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("model name", 0) == 0) {
      return line.substr(line.find(':') + 2);
    }
  }
  return "unknown";
}

int32_t NumCpus() {
  return static_cast<int32_t>(
      std::max<long>(1, sysconf(_SC_NPROCESSORS_ONLN)));  // NOLINT
}

// Iterations a client polls its empty inbox before it sleeps.
constexpr int32_t kClientSpinIterations = 2000;

inline void CpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield");
#endif
}

void PinCurrentThread(int32_t cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

// Exact percentiles over recorded samples (in ns).
class Samples {
 public:
  void Add(int64_t v) { values_.push_back(v); }
  void Merge(const Samples& other) {
    values_.insert(values_.end(), other.values_.begin(), other.values_.end());
  }
  int64_t Percentile(double q) {
    if (values_.empty()) return 0;
    const size_t k =
        std::min(values_.size() - 1, static_cast<size_t>(q * values_.size()));
    std::nth_element(values_.begin(), values_.begin() + k, values_.end());
    return values_[k];
  }

 private:
  std::vector<int64_t> values_;
};

struct Config {
  int32_t replicas = 0;
  int32_t hosts = 0;
  int32_t bundles = 0;
  // Requested k_b and the effective one (see `EffectiveGrantBatchSize`).
  int32_t requested_grant_batch = 1;
  int32_t grant_batch = 1;
  int32_t upload_cap = 1;
  bool allow_batch_above_cap = false;
  int32_t max_batch = 0;
  int32_t seed_replication = 2;
  int32_t client_threads = 1;
  bool pin_threads = false;
  bool realtime = false;
  double bundle_time_ms = 4.0;
};

PullSchedulerOptions SchedulerOptions(const Config& cfg) {
  PullSchedulerOptions options;
  options.grant_batch_size = cfg.requested_grant_batch;
  options.max_concurrent_uploads_per_source = cfg.upload_cap;
  options.allow_grant_batch_above_upload_cap = cfg.allow_batch_above_cap;
  if (cfg.max_batch > 0) options.max_batch_requests = cfg.max_batch;
  return options;
}

SeedLayout Layout(const Config& cfg) {
  std::vector<int64_t> bytes(cfg.bundles, 1);
  SeedingOptions seeding;
  seeding.replication = cfg.seed_replication;
  absl::StatusOr<SeedLayout> layout = LogicalReshardPlanner::BuildSeedLayout(
      bytes, cfg.replicas, seeding, /*trainer_streams=*/cfg.replicas);
  CHECK_OK(layout.status());
  return *std::move(layout);
}

// Ingress lower bound in bundle times: the replica with the fewest seeded
// bundles receives all the others at full ingress.
double LowerBound(const Config& cfg,
                  const std::vector<std::vector<int32_t>>& seeds) {
  size_t min_seeded = cfg.bundles;
  for (const auto& s : seeds) min_seeded = std::min(min_seeded, s.size());
  return static_cast<double>(cfg.bundles - static_cast<int32_t>(min_seeded));
}

// One load-generator thread: drives the pullers `(replica, host)` with
// `(replica * H + host) % threads == index`.
class Client {
 public:
  Client(const Config& cfg, PullScheduler* scheduler, int32_t index)
      : cfg_(cfg), scheduler_(scheduler) {
    for (int32_t r = 0; r < cfg.replicas; ++r) {
      for (int32_t h = 0; h < cfg.hosts; ++h) {
        if ((r * cfg.hosts + h) % cfg.client_threads == index) {
          pullers_.push_back(Puller{.replica = r, .host = h});
        }
      }
    }
    pull_ns_ = static_cast<int64_t>(
        cfg.bundle_time_ms * 1e6 *
        static_cast<double>(std::max(cfg.grant_batch, cfg.upload_cap)));
  }

  void Run() {
    for (size_t i = 0; i < pullers_.size(); ++i) {
      Send(static_cast<int32_t>(i), {});
    }
    while (done_ < static_cast<int32_t>(pullers_.size()) && !failed_) {
      EventNode* list = TakeEvents();
      if (list == nullptr && cfg_.realtime) {
        FireTimers(absl::GetCurrentTimeNanos());
        list = TakeEvents();
      }
      if (list == nullptr) {
        Wait();
        continue;
      }
      const int64_t now = absl::GetCurrentTimeNanos();
      while (list != nullptr) {
        EventNode* next = list->next;
        Handle(list->event, now);
        delete list;
        list = next;
      }
      if (cfg_.realtime) FireTimers(absl::GetCurrentTimeNanos());
    }
    // Free replies that arrived after the last puller finished.
    for (EventNode* list = TakeEvents(); list != nullptr;) {
      EventNode* next = list->next;
      delete list;
      list = next;
    }
  }

  Samples& wait() { return wait_; }
  int64_t finish_ns() const { return finish_ns_; }
  bool failed() const { return failed_; }

 private:
  struct Puller {
    int32_t replica = 0;
    int32_t host = 0;
    uint32_t seq = 0;
    int32_t inflight = 0;
    bool done = false;
  };
  struct Event {
    int32_t puller = 0;
    uint32_t seq = 0;
    int64_t submit_ns = 0;
    PullReply reply;
  };
  // Replies reach the client through a lock-free stack, like the completion
  // queue of an asynchronous server: the scheduler's shard thread runs the
  // callbacks and must not block on the client.
  struct EventNode {
    EventNode* next = nullptr;
    Event event;
  };
  struct Timer {
    int64_t due_ns = 0;
    int32_t puller = 0;
    uint64_t lease = 0;
    bool operator>(const Timer& o) const { return due_ns > o.due_ns; }
  };

  void Send(int32_t i, absl::InlinedVector<uint64_t, 8> completed) {
    Puller& p = pullers_[i];
    PullRequest req;
    req.replica = p.replica;
    req.host = p.host;
    req.completed = std::move(completed);
    req.max_grants = cfg_.grant_batch - p.inflight;
    const uint32_t seq = ++p.seq;
    const int64_t submit_ns = absl::GetCurrentTimeNanos();
    scheduler_->Submit(
        std::move(req), [this, i, seq, submit_ns](PullReply reply) mutable {
          Push(new EventNode{.event = Event{.puller = i,
                                            .seq = seq,
                                            .submit_ns = submit_ns,
                                            .reply = std::move(reply)}});
        });
  }

  // Any thread.
  void Push(EventNode* node) {
    EventNode* head = inbox_.load(std::memory_order_relaxed);
    do {
      node->next = head;
    } while (!inbox_.compare_exchange_weak(
        head, node, std::memory_order_seq_cst, std::memory_order_relaxed));
    if (sleeping_.load(std::memory_order_seq_cst)) {
      absl::MutexLock lock(mu_);
      cv_.Signal();
    }
  }

  // Returns the pending events in arrival order.
  EventNode* TakeEvents() {
    EventNode* list = inbox_.exchange(nullptr, std::memory_order_acquire);
    EventNode* fifo = nullptr;
    while (list != nullptr) {
      EventNode* next = list->next;
      list->next = fifo;
      fifo = list;
      list = next;
    }
    return fifo;
  }

  // Waits for an event or the next timer.
  void Wait() {
    for (int32_t i = 0; i < kClientSpinIterations; ++i) {
      if (inbox_.load(std::memory_order_relaxed) != nullptr) return;
      CpuRelax();
    }
    absl::MutexLock lock(mu_);
    sleeping_.store(true, std::memory_order_seq_cst);
    if (inbox_.load(std::memory_order_seq_cst) == nullptr) {
      const int64_t wait_ns =
          timers_.empty() ? 1'000'000'000
                          : timers_.top().due_ns - absl::GetCurrentTimeNanos();
      if (wait_ns > 0) cv_.WaitWithTimeout(&mu_, absl::Nanoseconds(wait_ns));
    }
    sleeping_.store(false, std::memory_order_relaxed);
  }

  void Handle(Event& e, int64_t now) {
    Puller& p = pullers_[e.puller];
    if (e.reply.kind == PullReplyKind::kAborted) {
      if (!p.done) {
        failed_ = true;
        absl::FPrintF(stderr, "aborted: %s\n", e.reply.status.ToString());
      }
      return;
    }
    if (!e.reply.grants.empty()) wait_.Add(now - e.submit_ns);
    // A reply to a superseded request still carries real grants.
    const bool current = e.seq == p.seq;
    if (e.reply.kind == PullReplyKind::kDone) {
      if (!p.done) {
        p.done = true;
        ++done_;
        finish_ns_ = std::max(finish_ns_, now);
      }
      return;
    }
    if (!cfg_.realtime) {
      if (!current && e.reply.grants.empty()) return;
      absl::InlinedVector<uint64_t, 8> completed;
      for (const PullGrant& g : e.reply.grants) completed.push_back(g.lease_id);
      Send(e.puller, std::move(completed));
      return;
    }
    for (const PullGrant& g : e.reply.grants) {
      timers_.push(Timer{now + pull_ns_, e.puller, g.lease_id});
      ++p.inflight;
    }
    if (current && p.inflight < cfg_.grant_batch) Send(e.puller, {});
  }

  void FireTimers(int64_t now) {
    std::vector<std::pair<int32_t, uint64_t>> due;
    while (!timers_.empty() && timers_.top().due_ns <= now) {
      due.emplace_back(timers_.top().puller, timers_.top().lease);
      timers_.pop();
    }
    std::sort(due.begin(), due.end());
    for (size_t i = 0; i < due.size();) {
      const int32_t puller = due[i].first;
      absl::InlinedVector<uint64_t, 8> completed;
      for (; i < due.size() && due[i].first == puller; ++i) {
        completed.push_back(due[i].second);
      }
      pullers_[puller].inflight -= static_cast<int32_t>(completed.size());
      Send(puller, std::move(completed));
    }
  }

  const Config cfg_;
  PullScheduler* const scheduler_;
  std::vector<Puller> pullers_;
  int64_t pull_ns_ = 0;
  int32_t done_ = 0;
  bool failed_ = false;
  int64_t finish_ns_ = 0;
  Samples wait_;
  std::priority_queue<Timer, std::vector<Timer>, std::greater<Timer>> timers_;

  std::atomic<EventNode*> inbox_{nullptr};
  std::atomic<bool> sleeping_{false};
  absl::Mutex mu_;
  absl::CondVar cv_;
};

struct RunResult {
  double wall_s = 0;
  int64_t grants = 0;
  std::vector<int64_t> shard_grants;
  int64_t shard_cpu_ns = 0;
  int64_t process_cpu_ns = 0;
  int64_t batches = 0;
  int64_t requests = 0;
  int64_t max_batch = 0;
  int64_t batch_p50_ns = 0;
  int64_t batch_p99_ns = 0;
  int64_t request_p50_ns = 0;
  int64_t request_p99_ns = 0;
  Samples wait;
  double sync_s = 0;
  bool failed = false;

  void Add(const RunResult& r) {
    wall_s += r.wall_s;
    grants += r.grants;
    if (shard_grants.empty()) {
      shard_grants = r.shard_grants;
    } else {
      for (size_t i = 0; i < r.shard_grants.size(); ++i) {
        shard_grants[i] += r.shard_grants[i];
      }
    }
    shard_cpu_ns += r.shard_cpu_ns;
    process_cpu_ns += r.process_cpu_ns;
    batches += r.batches;
    requests += r.requests;
    max_batch = std::max(max_batch, r.max_batch);
    batch_p50_ns = std::max(batch_p50_ns, r.batch_p50_ns);
    batch_p99_ns = std::max(batch_p99_ns, r.batch_p99_ns);
    request_p50_ns = std::max(request_p50_ns, r.request_p50_ns);
    request_p99_ns = std::max(request_p99_ns, r.request_p99_ns);
    wait.Merge(r.wait);
    sync_s = std::max(sync_s, r.sync_s);
  }
};

RunResult RunOnce(const Config& cfg,
                  const std::vector<std::vector<int32_t>>& seeds) {
  const int32_t cpus = NumCpus();
  PullSchedulerOptions options = SchedulerOptions(cfg);
  options.lease_timeout = absl::Seconds(600);
  options.long_poll_timeout = absl::Seconds(60);
  if (cfg.pin_threads) {
    options.shard_thread_init = [cpus](int32_t host) {
      PinCurrentThread(host % cpus);
    };
  }
  absl::StatusOr<std::unique_ptr<PullScheduler>> scheduler =
      PullScheduler::Create(cfg.replicas, cfg.hosts, cfg.bundles, seeds,
                            options);
  CHECK_OK(scheduler.status());
  for (int32_t r = 0; r < cfg.replicas; ++r) (*scheduler)->MarkSeeded(r);

  std::vector<std::unique_ptr<Client>> clients;
  clients.reserve(cfg.client_threads);
  for (int32_t t = 0; t < cfg.client_threads; ++t) {
    clients.push_back(std::make_unique<Client>(cfg, scheduler->get(), t));
  }
  const int64_t cpu0 = ProcessCpuNanos();
  const int64_t start = absl::GetCurrentTimeNanos();
  std::vector<std::thread> threads;
  threads.reserve(cfg.client_threads);
  for (int32_t t = 0; t < cfg.client_threads; ++t) {
    threads.emplace_back([&cfg, &clients, t, cpus] {
      if (cfg.pin_threads && cpus > cfg.hosts) {
        PinCurrentThread(cfg.hosts + t % (cpus - cfg.hosts));
      }
      clients[t]->Run();
    });
  }
  for (std::thread& t : threads) t.join();
  const int64_t end = absl::GetCurrentTimeNanos();
  RunResult result;
  result.process_cpu_ns = ProcessCpuNanos() - cpu0;
  result.wall_s = static_cast<double>(end - start) * 1e-9;
  int64_t finish = start;
  for (auto& c : clients) {
    result.wait.Merge(c->wait());
    finish = std::max(finish, c->finish_ns());
    result.failed |= c->failed();
  }
  result.sync_s = static_cast<double>(finish - start) * 1e-9;
  // Idle shards publish their final CPU time before they sleep.
  absl::SleepFor(absl::Milliseconds(50));
  for (const PullShardStats& s : (*scheduler)->GetStats()) {
    result.grants += s.grants;
    result.shard_grants.push_back(s.grants);
    result.shard_cpu_ns += s.thread_cpu_ns;
    result.batches += s.batches;
    result.requests += s.requests;
    result.max_batch = std::max(result.max_batch, s.max_batch_requests);
    result.batch_p50_ns = std::max(result.batch_p50_ns, s.batch_p50_ns);
    result.batch_p99_ns = std::max(result.batch_p99_ns, s.batch_p99_ns);
    result.request_p50_ns = std::max(result.request_p50_ns, s.request_p50_ns);
    result.request_p99_ns = std::max(result.request_p99_ns, s.request_p99_ns);
  }
  return result;
}

void PrintHeader(absl::string_view mode) {
  absl::PrintF("\n== %s\n", mode);
  absl::PrintF(
      "%6s %4s %3s %3s %4s %7s %10s %9s %9s %8s %8s %8s %8s %8s %8s %7s %6s "
      "%7s %7s %8s\n",
      "B", "kb_r", "kb", "c", "runs", "wall_s", "grants", "dec/s", "dec/s/sh",
      "req50us", "req99us", "wait50us", "wait99us", "bat50us", "bat99us",
      "req/bat", "maxbat", "cpu_ns", "proc_ns", "sync/lb");
}

void RunConfig(const Config& cfg, double min_seconds) {
  const std::vector<std::vector<int32_t>> seeds =
      PullScheduler::SeededBundles(Layout(cfg), cfg.replicas);
  RunResult total;
  int32_t runs = 0;
  do {
    RunResult r = RunOnce(cfg, seeds);
    if (r.failed) {
      absl::PrintF("run failed\n");
      return;
    }
    ++runs;
    total.Add(r);
  } while (!cfg.realtime && total.wall_s < min_seconds);
  int64_t min_shard = total.shard_grants.empty() ? 0 : total.shard_grants[0];
  for (int64_t g : total.shard_grants) min_shard = std::min(min_shard, g);
  const double grants = static_cast<double>(std::max<int64_t>(1, total.grants));
  const std::string sync =
      cfg.realtime
          ? absl::StrFormat("%.3f", total.sync_s / (LowerBound(cfg, seeds) *
                                                    cfg.bundle_time_ms * 1e-3))
          : std::string("-");
  absl::PrintF(
      "%6d %4d %3d %3d %4d %7.2f %10d %9.0f %9.0f %8.1f %8.1f %8.1f %8.1f "
      "%8.1f %8.1f %7.1f %6d %7.0f %7.0f %8s\n",
      cfg.bundles, cfg.requested_grant_batch, cfg.grant_batch, cfg.upload_cap,
      runs, total.wall_s, total.grants, total.grants / total.wall_s,
      min_shard / total.wall_s, total.request_p50_ns * 1e-3,
      total.request_p99_ns * 1e-3, total.wait.Percentile(0.5) * 1e-3,
      total.wait.Percentile(0.99) * 1e-3, total.batch_p50_ns * 1e-3,
      total.batch_p99_ns * 1e-3,
      static_cast<double>(total.requests) /
          static_cast<double>(std::max<int64_t>(1, total.batches)),
      total.max_batch, static_cast<double>(total.shard_cpu_ns) / grants,
      static_cast<double>(total.process_cpu_ns) / grants, sync);
  std::fflush(stdout);
}

void RunSimulation(const Config& cfg) {
  FakeFleetTopology topology;
  topology.num_replicas = cfg.replicas;
  topology.num_hosts = cfg.hosts;
  topology.bundle_bytes.assign(cfg.bundles, 1.0);
  topology.seed_layout = Layout(cfg);
  PullSchedulerOptions options = SchedulerOptions(cfg);
  options.lease_timeout = absl::Hours(1000);
  options.long_poll_timeout = absl::Hours(1000);
  absl::StatusOr<FakeFleetResult> result =
      SimulateScheduledPulls(topology, options, FakeSamplerFleetOptions());
  CHECK_OK(result.status());
  if (!result->status.ok()) {
    absl::PrintF("%6d %4d %3d %3d simulation failed: %s\n", cfg.bundles,
                 cfg.requested_grant_batch, cfg.grant_batch, cfg.upload_cap,
                 result->status.ToString());
    return;
  }
  const double bound = LowerBound(
      cfg, PullScheduler::SeededBundles(topology.seed_layout, cfg.replicas));
  absl::PrintF("%6d %4d %3d %3d %10.1f %10.1f %8.3f %10d %10d %8.2f\n",
               cfg.bundles, cfg.requested_grant_batch, cfg.grant_batch,
               cfg.upload_cap, result->finish_time, bound,
               result->finish_time / bound, result->grants, result->requests,
               result->wall_seconds);
  std::fflush(stdout);
}

int Main() {
  const std::vector<std::string> modes =
      absl::StrSplit(absl::GetFlag(FLAGS_mode), ',', absl::SkipEmpty());
  absl::PrintF(
      "machine: %s, %d cpus; D=%d H=%d R=%d client_threads=%d pinned=%d\n",
      CpuModel(), NumCpus(), absl::GetFlag(FLAGS_replicas),
      absl::GetFlag(FLAGS_hosts), absl::GetFlag(FLAGS_seed_replication),
      absl::GetFlag(FLAGS_client_threads), absl::GetFlag(FLAGS_pin_threads));
  for (const std::string& mode : modes) {
    if (mode == "simulate") {
      absl::PrintF("\n== simulate (virtual time, bundle time = 1)\n");
      absl::PrintF("%6s %4s %3s %3s %10s %10s %8s %10s %10s %8s\n", "B", "kb_r",
                   "kb", "c", "sync", "bound", "ratio", "grants", "requests",
                   "wall_s");
    } else {
      PrintHeader(mode);
    }
    for (int32_t b : ParseList(absl::GetFlag(FLAGS_bundles))) {
      for (int32_t kb : ParseList(absl::GetFlag(FLAGS_grant_batch))) {
        Config cfg;
        cfg.replicas = absl::GetFlag(FLAGS_replicas);
        cfg.hosts = absl::GetFlag(FLAGS_hosts);
        cfg.bundles = b;
        cfg.requested_grant_batch = kb;
        cfg.upload_cap = absl::GetFlag(FLAGS_upload_cap) > 0
                             ? absl::GetFlag(FLAGS_upload_cap)
                             : kb;
        cfg.allow_batch_above_cap = absl::GetFlag(FLAGS_allow_batch_above_cap);
        cfg.grant_batch = EffectiveGrantBatchSize(SchedulerOptions(cfg));
        cfg.max_batch = absl::GetFlag(FLAGS_max_batch);
        cfg.seed_replication = absl::GetFlag(FLAGS_seed_replication);
        cfg.client_threads = absl::GetFlag(FLAGS_client_threads);
        cfg.pin_threads = absl::GetFlag(FLAGS_pin_threads);
        cfg.bundle_time_ms = absl::GetFlag(FLAGS_bundle_time_ms);
        if (mode == "saturate") {
          RunConfig(cfg, absl::GetFlag(FLAGS_min_seconds));
        } else if (mode == "realtime") {
          cfg.realtime = true;
          RunConfig(cfg, 0);
        } else if (mode == "simulate") {
          RunSimulation(cfg);
        }
      }
    }
  }
  return 0;
}

}  // namespace
}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  return tpu_raiden::weight_sync::v3::Main();
}
