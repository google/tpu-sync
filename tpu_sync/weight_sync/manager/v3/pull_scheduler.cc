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

#include "tpu_sync/weight_sync/manager/v3/pull_scheduler.h"

#include <time.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <queue>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/weight_sync/manager/v3/logical_types.h"

namespace tpu_raiden {
namespace weight_sync {
namespace v3 {
namespace {

// Intersections with at most this many bundles are searched bit by bit for
// the rarest bundle; denser ones by walking the bundles in rarity order.
constexpr int32_t kSparseIntersection = 64;
// Iterations a shard thread polls its empty queue before it sleeps.
constexpr int32_t kSpinIterations = 2000;

// Lease ids: bundle (24 bits) | generation (8 bits) | slot (32 bits). The
// bundle lets a late completion of a reclaimed lease still count.
uint64_t MakeLeaseId(int32_t bundle, uint32_t generation, uint32_t slot) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(bundle)) << 40) |
         (static_cast<uint64_t>(generation & 0xFF) << 32) | slot;
}
uint32_t LeaseSlot(uint64_t id) { return static_cast<uint32_t>(id); }
uint32_t LeaseGeneration(uint64_t id) {
  return static_cast<uint32_t>(id >> 32) & 0xFF;
}

int64_t ThreadCpuNanos() {
  timespec ts;
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) return 0;
  return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

inline void CpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield");
#endif
}

// Node of `MpscQueue`.
struct QueueNode {
  std::atomic<QueueNode*> next{nullptr};
};

// Intrusive, lock-free multi-producer single-consumer queue (Vyukov).
class MpscQueue {
 public:
  MpscQueue() : head_(&stub_), tail_(&stub_) {}

  // Any thread.
  void Push(QueueNode* node) {
    node->next.store(nullptr, std::memory_order_relaxed);
    // seq_cst: pairs with the consumer's sleep protocol (see `Shard`).
    QueueNode* prev = head_.exchange(node, std::memory_order_seq_cst);
    prev->next.store(node, std::memory_order_release);
  }

  // Consumer only. Returns nullptr if the queue is empty or a push is still
  // linking its node.
  QueueNode* Pop() {
    QueueNode* tail = tail_;
    QueueNode* next = tail->next.load(std::memory_order_acquire);
    if (tail == &stub_) {
      if (next == nullptr) return nullptr;
      tail_ = next;
      tail = next;
      next = next->next.load(std::memory_order_acquire);
    }
    if (next != nullptr) {
      tail_ = next;
      return tail;
    }
    if (tail != head_.load(std::memory_order_acquire)) return nullptr;
    Push(&stub_);
    next = tail->next.load(std::memory_order_acquire);
    if (next != nullptr) {
      tail_ = next;
      return tail;
    }
    return nullptr;
  }

  // Consumer only. False if anything was pushed and not popped yet.
  bool Empty() const {
    return tail_ == &stub_ && head_.load(std::memory_order_seq_cst) == &stub_;
  }

 private:
  alignas(64) std::atomic<QueueNode*> head_;
  alignas(64) QueueNode* tail_;
  QueueNode stub_;
};

// Single-writer counter that other threads may read.
class Counter {
 public:
  void Add(int64_t v) {
    value_.store(value_.load(std::memory_order_relaxed) + v,
                 std::memory_order_relaxed);
  }
  void Set(int64_t v) { value_.store(v, std::memory_order_relaxed); }
  void Max(int64_t v) {
    if (v > value_.load(std::memory_order_relaxed)) Set(v);
  }
  int64_t Get() const { return value_.load(std::memory_order_relaxed); }

 private:
  std::atomic<int64_t> value_{0};
};

// Single-writer latency histogram: 8 buckets per power of two.
class LatencyHistogram {
 public:
  void Record(int64_t ns) { buckets_[Bucket(ns)].Add(1); }

  int64_t Percentile(double q) const {
    int64_t total = 0;
    for (const Counter& c : buckets_) total += c.Get();
    if (total == 0) return 0;
    const int64_t rank = static_cast<int64_t>(q * static_cast<double>(total));
    int64_t seen = 0;
    for (int i = 0; i < kBuckets; ++i) {
      seen += buckets_[i].Get();
      if (seen > rank) return UpperBound(i);
    }
    return UpperBound(kBuckets - 1);
  }

 private:
  static constexpr int kBuckets = 64 * 8;

  static int Bucket(int64_t ns) {
    const uint64_t v = static_cast<uint64_t>(std::max<int64_t>(ns, 1));
    const int msb = 63 - std::countl_zero(v);
    if (msb < 3) return static_cast<int>(v);
    const int sub = static_cast<int>((v >> (msb - 3)) & 7);
    return std::min(kBuckets - 1, msb * 8 + sub);
  }

  static int64_t UpperBound(int bucket) {
    if (bucket < 8) return bucket;
    const int msb = bucket / 8;
    const int sub = bucket % 8;
    return static_cast<int64_t>((uint64_t{8} + sub + 1) << (msb - 3));
  }

  Counter buckets_[kBuckets];
};

enum class OpKind { kRequest, kSeeded, kDead, kAbort, kStop };

struct Op : QueueNode {
  OpKind kind = OpKind::kRequest;
  PullRequest request;
  PullReplyCallback done;
  int32_t replica = 0;
  int64_t enqueue_ns = 0;
};

}  // namespace

// State shared by the shards of one scheduler.
struct PullScheduler::Shared {
  std::vector<Shard*> shards;
  std::atomic<bool> propagate_abort{false};

  mutable absl::Mutex mu;
  absl::Status status ABSL_GUARDED_BY(mu);
  int32_t shards_complete ABSL_GUARDED_BY(mu) = 0;
  bool complete ABSL_GUARDED_BY(mu) = false;

  // Records |abort| (unless the transfer already completed or failed) and
  // asks for it to reach every shard. Returns the effective status.
  absl::Status RaiseAbort(const absl::Status& abort) {
    absl::MutexLock lock(mu);
    if (status.ok() && !complete) {
      status = abort;
      propagate_abort.store(true, std::memory_order_release);
    }
    return status.ok() ? abort : status;
  }

  void ShardComplete() {
    absl::MutexLock lock(mu);
    if (++shards_complete == static_cast<int32_t>(shards.size()) &&
        status.ok()) {
      complete = true;
    }
  }

  // Posts the abort to every shard once. Must not be called with a shard
  // lock held.
  void PropagateIfNeeded();
};

// One host index of the transfer. Owns all state of that host index; only
// its thread (or, with inline execution, the holder of `mu_`) touches it.
class PullScheduler::Shard {
 public:
  Shard(int32_t host, int32_t num_replicas, int32_t num_bundles,
        const std::vector<std::vector<int32_t>>& seeded,
        const PullSchedulerOptions& options, Shared* shared)
      : host_(host),
        num_replicas_(num_replicas),
        num_bundles_(num_bundles),
        num_words_((num_bundles + 63) / 64),
        grant_batch_(EffectiveGrantBatchSize(options)),
        upload_cap_(std::max(1, options.max_concurrent_uploads_per_source)),
        unhealthy_reports_(std::max(1, options.unhealthy_source_reports)),
        max_batch_(std::max(1, options.max_batch_requests)),
        lease_timeout_(options.lease_timeout),
        long_poll_timeout_(options.long_poll_timeout),
        sweep_interval_(
            std::max(absl::Milliseconds(1), options.lease_timeout / 4)),
        inline_(options.inline_execution),
        clock_(options.clock),
        thread_init_(options.shard_thread_init),
        shared_(shared),
        seeded_(seeded),
        hosts_(num_replicas),
        have_(static_cast<size_t>(num_replicas) * num_words_, 0),
        want_(static_cast<size_t>(num_replicas) * num_words_, 0),
        have_summary_(num_replicas, 0),
        want_summary_(num_replicas, 0),
        live_(num_bundles, 0),
        pending_(num_bundles, 0),
        copies_(num_bundles, 0),
        order_(num_bundles),
        position_(num_bundles),
        group_start_(num_replicas + 2, num_bundles),
        free_(upload_cap_) {
    group_start_[0] = 0;
    for (int32_t b = 0; b < num_bundles_; ++b) {
      order_[b] = b;
      position_[b] = b;
    }
    const absl::Time now = clock_();
    next_sweep_ = now + sweep_interval_;
    for (int32_t r = 0; r < num_replicas_; ++r) {
      Host& h = hosts_[r];
      h.last_seen = now;
      for (int32_t b = 0; b < num_bundles_; ++b) SetBit(want_, r, b);
      h.want_count = num_bundles_;
      for (int32_t b : seeded_[r]) {
        if (TestBit(want_, r, b)) {
          ClearBit(want_, r, b);
          --h.want_count;
        }
        ++pending_[b];
      }
      h.seed_pending = !seeded_[r].empty();
      RecomputeSummaries(r);
    }
    stats_host_ = host;
  }

  ~Shard() { Stop(); }

  void Start() {
    if (inline_) return;
    thread_ = std::thread([this] {
      if (thread_init_) thread_init_(host_);
      Run();
    });
  }

  void Stop() {
    if (stopped_) return;
    stopped_ = true;
    if (!inline_ && thread_.joinable()) {
      Op* op = new Op;
      op->kind = OpKind::kStop;
      Post(op);
      thread_.join();
    }
    std::vector<std::pair<PullReplyCallback, PullReply>> out;
    {
      absl::MutexLock lock(mu_);
      CancelAll(absl::CancelledError("Pull scheduler shut down"));
      out.swap(outbox_);
    }
    Deliver(out);
  }

  // Takes ownership of |op|.
  void Post(Op* op) {
    if (!inline_) {
      op->enqueue_ns = absl::GetCurrentTimeNanos();
      queue_.Push(op);
      if (sleeping_.load(std::memory_order_seq_cst)) {
        absl::MutexLock lock(wake_mu_);
        wake_cv_.Signal();
      }
      return;
    }
    std::vector<std::pair<PullReplyCallback, PullReply>> out;
    {
      absl::MutexLock lock(mu_);
      const absl::Time now = clock_();
      ProcessTimersIfDue(now);
      Execute(op, now);
      Match();
      FinishBatch();
      out.swap(outbox_);
    }
    Deliver(out);
    shared_->PropagateIfNeeded();
  }

  void AdvanceInline() {
    if (!inline_) return;
    std::vector<std::pair<PullReplyCallback, PullReply>> out;
    {
      absl::MutexLock lock(mu_);
      ProcessTimersIfDue(clock_());
      Match();
      FinishBatch();
      out.swap(outbox_);
    }
    Deliver(out);
    shared_->PropagateIfNeeded();
  }

  absl::Time NextTimerInline() {
    absl::MutexLock lock(mu_);
    return NextTimer();
  }

  PullShardStats Stats() const {
    PullShardStats s;
    s.host = stats_host_;
    s.requests = requests_.Get();
    s.grants = grants_.Get();
    s.completions = completions_.Get();
    s.failures = failures_.Get();
    s.expired_leases = expired_.Get();
    s.long_poll_timeouts = poll_timeouts_.Get();
    s.suspended_hosts = suspended_.Get();
    s.unhealthy_sources = unhealthy_.Get();
    s.batches = batches_.Get();
    s.max_batch_requests = max_batch_seen_.Get();
    s.busy_ns = busy_ns_.Get();
    s.batch_p50_ns = batch_hist_.Percentile(0.5);
    s.batch_p99_ns = batch_hist_.Percentile(0.99);
    s.request_p50_ns = request_hist_.Percentile(0.5);
    s.request_p99_ns = request_hist_.Percentile(0.99);
    s.thread_cpu_ns = cpu_ns_.Get();
    s.parked_requests = static_cast<int32_t>(parked_.Get());
    s.done_hosts = static_cast<int32_t>(done_gauge_.Get());
    s.leases_in_flight = static_cast<int32_t>(leases_gauge_.Get());
    return s;
  }

 private:
  enum class HostState : uint8_t { kActive, kSuspended, kDead };

  // Host `host_` of one replica, as a puller and as a source.
  struct Host {
    // Parked request, if any, and the reply being built for it.
    PullReplyCallback waiter;
    PullReply reply;
    bool parked = false;
    bool in_outbox = false;
    int32_t remaining = 0;
    uint32_t seq = 0;
    // Puller side.
    int32_t inflight = 0;
    int32_t have_count = 0;
    int32_t want_count = 0;
    absl::InlinedVector<uint32_t, 8> leases;
    absl::InlinedVector<int32_t, 4> excluded;
    // Source side.
    int32_t uploads = 0;
    absl::InlinedVector<int32_t, 4> reporters;
    bool unhealthy = false;
    HostState state = HostState::kActive;
    bool done = false;
    bool seed_pending = false;
    absl::Time last_seen;
    // Intrusive FIFO of waiting pullers.
    bool waiting = false;
    int32_t wait_prev = -1;
    int32_t wait_next = -1;
    // Position in `free_[free_load]`, or -1.
    int32_t free_load = -1;
    int32_t free_pos = -1;
    bool dirty_source = false;
    bool dirty_puller = false;
  };

  struct Lease {
    int32_t puller = -1;
    int32_t source = -1;
    int32_t bundle = -1;
    uint32_t generation = 0;
    // Index in `hosts_[puller].leases`.
    int32_t index = -1;
    bool active = false;
  };

  struct TimedLease {
    absl::Time expiry;
    uint64_t id;
  };

  struct TimedPoll {
    absl::Time deadline;
    int32_t puller;
    uint32_t seq;
    // Orders `poll_queue_` by earliest deadline first.
    bool operator>(const TimedPoll& other) const {
      return deadline > other.deadline;
    }
  };

  // ---- Bitsets -------------------------------------------------------------

  uint64_t* Row(std::vector<uint64_t>& bits, int32_t r) {
    return bits.data() + static_cast<size_t>(r) * num_words_;
  }
  bool TestBit(const std::vector<uint64_t>& bits, int32_t r, int32_t b) const {
    return (bits[static_cast<size_t>(r) * num_words_ + (b >> 6)] >> (b & 63)) &
           1;
  }
  void SetBit(std::vector<uint64_t>& bits, int32_t r, int32_t b) {
    bits[static_cast<size_t>(r) * num_words_ + (b >> 6)] |= uint64_t{1}
                                                            << (b & 63);
  }
  void ClearBit(std::vector<uint64_t>& bits, int32_t r, int32_t b) {
    bits[static_cast<size_t>(r) * num_words_ + (b >> 6)] &=
        ~(uint64_t{1} << (b & 63));
  }
  // Bit `w % 64` of a summary is set iff any word `w' == w (mod 64)` is
  // non-zero, so disjoint summaries prove disjoint bitsets.
  uint64_t SummaryBit(const std::vector<uint64_t>& bits, int32_t r,
                      int32_t word) const {
    const uint64_t* row = bits.data() + static_cast<size_t>(r) * num_words_;
    for (int32_t w = word & 63; w < num_words_; w += 64) {
      if (row[w] != 0) return uint64_t{1} << (word & 63);
    }
    return 0;
  }
  void UpdateSummary(std::vector<uint64_t>& bits,
                     std::vector<uint64_t>& summary, int32_t r, int32_t b) {
    const int32_t word = b >> 6;
    summary[r] = (summary[r] & ~(uint64_t{1} << (word & 63))) |
                 SummaryBit(bits, r, word);
  }
  void RecomputeSummaries(int32_t r) {
    have_summary_[r] = 0;
    want_summary_[r] = 0;
    for (int32_t w = 0; w < num_words_; ++w) {
      have_summary_[r] |= SummaryBit(have_, r, w);
      want_summary_[r] |= SummaryBit(want_, r, w);
    }
  }

  void SetWant(int32_t p, int32_t b) {
    if (TestBit(want_, p, b)) return;
    SetBit(want_, p, b);
    want_summary_[p] |= uint64_t{1} << ((b >> 6) & 63);
    ++hosts_[p].want_count;
  }
  void ClearWant(int32_t p, int32_t b) {
    if (!TestBit(want_, p, b)) return;
    ClearBit(want_, p, b);
    UpdateSummary(want_, want_summary_, p, b);
    --hosts_[p].want_count;
  }

  // ---- Rarity order (bundles sorted by `copies_`) --------------------------

  void SwapOrder(int32_t i, int32_t j) {
    const int32_t a = order_[i];
    const int32_t b = order_[j];
    order_[i] = b;
    order_[j] = a;
    position_[b] = i;
    position_[a] = j;
  }
  void IncCopies(int32_t b) {
    const int32_t c = copies_[b];
    SwapOrder(position_[b], group_start_[c + 1] - 1);
    --group_start_[c + 1];
    copies_[b] = c + 1;
  }
  void DecCopies(int32_t b) {
    const int32_t c = copies_[b];
    SwapOrder(position_[b], group_start_[c]);
    ++group_start_[c];
    copies_[b] = c - 1;
  }

  // ---- Roles ---------------------------------------------------------------

  bool IsSource(int32_t r) const {
    return hosts_[r].state == HostState::kActive && !hosts_[r].unhealthy;
  }

  // Lists |r| in `free_` iff it can take another upload.
  void Relist(int32_t r) {
    Host& h = hosts_[r];
    const int32_t want =
        (IsSource(r) && h.have_count > 0 && h.uploads < upload_cap_) ? h.uploads
                                                                     : -1;
    if (want == h.free_load) return;
    if (h.free_load >= 0) {
      std::vector<int32_t>& bucket = free_[h.free_load];
      const int32_t last = bucket.back();
      bucket[h.free_pos] = last;
      hosts_[last].free_pos = h.free_pos;
      bucket.pop_back();
    }
    h.free_load = want;
    if (want >= 0) {
      h.free_pos = static_cast<int32_t>(free_[want].size());
      free_[want].push_back(r);
    } else {
      h.free_pos = -1;
    }
  }

  bool Listed(int32_t r) const { return hosts_[r].free_load >= 0; }

  void MarkDirtySource(int32_t r) {
    if (hosts_[r].dirty_source) return;
    hosts_[r].dirty_source = true;
    dirty_sources_.push_back(r);
  }
  void MarkDirtyPuller(int32_t r) {
    if (hosts_[r].dirty_puller) return;
    hosts_[r].dirty_puller = true;
    dirty_pullers_.push_back(r);
  }

  // Adds (|sign| = 1) or removes (-1) the live-holder and pending-seed
  // contributions of |r| for every bundle it holds or will be seeded with.
  // A host contributes iff `IsSource(r)`: call this right after it became a
  // source or right before it stops being one.
  void Contribute(int32_t r, int32_t sign) {
    Host& h = hosts_[r];
    if (h.have_count > 0) {
      const uint64_t* row = Row(have_, r);
      for (int32_t w = 0; w < num_words_; ++w) {
        for (uint64_t x = row[w]; x != 0; x &= x - 1) {
          const int32_t b = w * 64 + std::countr_zero(x);
          live_[b] += sign;
          if (sign > 0) {
            IncCopies(b);
          } else {
            DecCopies(b);
          }
        }
      }
    }
    if (h.seed_pending) {
      for (int32_t b : seeded_[r]) pending_[b] += sign;
    }
  }

  // Aborts if a bundle |r| holds or will be seeded with has no potential
  // source left.
  void CheckLost(int32_t r) {
    const uint64_t* row = Row(have_, r);
    for (int32_t w = 0; w < num_words_ && !aborted_; ++w) {
      for (uint64_t x = row[w]; x != 0; x &= x - 1) {
        CheckBundle(w * 64 + std::countr_zero(x));
      }
    }
    for (int32_t b : seeded_[r]) CheckBundle(b);
  }

  void CheckBundle(int32_t b) {
    if (aborted_ || live_[b] + pending_[b] > 0) return;
    // Rare: no potential source left. Fine if no live replica misses it.
    bool missing = false;
    for (int32_t r = 0; r < num_replicas_ && !missing; ++r) {
      missing = hosts_[r].state != HostState::kDead && !TestBit(have_, r, b);
    }
    if (!missing) return;
    // TODO(b/570356170): Ask the Trainer to re-push the bundle instead.
    AbortLocal(shared_->RaiseAbort(absl::UnavailableError(absl::StrCat(
        "No live replica holds bundle ", b, " on host ", host_,
        " any more (every seed and every replica that pulled it failed)"))));
  }

  // Marks the shard complete once every replica that is not dead is done.
  void MaybeComplete() {
    if (complete_ || done_count_ + dead_undone_ < num_replicas_) return;
    complete_ = true;
    shared_->ShardComplete();
  }

  // |p| now holds |b|.
  void AddHave(int32_t p, int32_t b) {
    Host& h = hosts_[p];
    if (TestBit(have_, p, b)) return;
    ClearWant(p, b);
    SetBit(have_, p, b);
    have_summary_[p] |= uint64_t{1} << ((b >> 6) & 63);
    ++h.have_count;
    if (IsSource(p)) {
      ++live_[b];
      IncCopies(b);
    }
    Relist(p);
    MarkDirtySource(p);
    if (h.have_count == num_bundles_ && !h.done) {
      h.done = true;
      done_gauge_.Set(++done_count_);
      MaybeComplete();
    }
  }

  // ---- Waiting pullers ------------------------------------------------------

  void RelistWaiting(int32_t p) {
    Host& h = hosts_[p];
    const bool want = h.parked && h.remaining > 0 &&
                      h.inflight < grant_batch_ && h.want_count > 0 &&
                      !h.done && h.state == HostState::kActive;
    if (want == h.waiting) return;
    h.waiting = want;
    if (want) {
      h.wait_prev = wait_tail_;
      h.wait_next = -1;
      if (wait_tail_ >= 0) {
        hosts_[wait_tail_].wait_next = p;
      } else {
        wait_head_ = p;
      }
      wait_tail_ = p;
    } else {
      if (h.wait_prev >= 0) {
        hosts_[h.wait_prev].wait_next = h.wait_next;
      } else {
        wait_head_ = h.wait_next;
      }
      if (h.wait_next >= 0) {
        hosts_[h.wait_next].wait_prev = h.wait_prev;
      } else {
        wait_tail_ = h.wait_prev;
      }
      h.wait_prev = h.wait_next = -1;
    }
  }

  // Answers the parked request of |p|.
  void Reply(int32_t p, PullReplyKind kind,
             absl::Status status = absl::OkStatus()) {
    Host& h = hosts_[p];
    if (!h.parked) return;
    h.reply.kind = kind;
    h.reply.status = std::move(status);
    if (kind != PullReplyKind::kGrants) h.reply.grants.clear();
    outbox_.emplace_back(std::move(h.waiter), std::move(h.reply));
    h.waiter = nullptr;
    h.reply = PullReply();
    h.parked = false;
    h.remaining = 0;
    ++h.seq;
    parked_.Set(--parked_count_);
    RelistWaiting(p);
  }

  static void Deliver(
      std::vector<std::pair<PullReplyCallback, PullReply>>& out) {
    for (auto& [done, reply] : out) std::move(done)(std::move(reply));
    out.clear();
  }

  // ---- Leases --------------------------------------------------------------

  uint32_t AllocLease() {
    if (!free_leases_.empty()) {
      const uint32_t slot = free_leases_.back();
      free_leases_.pop_back();
      return slot;
    }
    leases_.emplace_back();
    return static_cast<uint32_t>(leases_.size() - 1);
  }

  // Ends lease |slot|: frees the source's upload slot and the puller's lease
  // slot. The caller handles the bundle.
  void EndLease(uint32_t slot) {
    Lease& l = leases_[slot];
    Host& puller = hosts_[l.puller];
    const uint32_t last = puller.leases.back();
    puller.leases[l.index] = last;
    leases_[last].index = l.index;
    puller.leases.pop_back();
    --puller.inflight;
    Host& source = hosts_[l.source];
    --source.uploads;
    Relist(l.source);
    MarkDirtySource(l.source);
    l.active = false;
    ++l.generation;
    free_leases_.push_back(slot);
    leases_gauge_.Set(--leases_in_flight_);
  }

  // Returns the active lease |id| of |p|, or nullptr.
  Lease* FindLease(int32_t p, uint64_t id) {
    const uint32_t slot = LeaseSlot(id);
    if (slot >= leases_.size()) return nullptr;
    Lease& l = leases_[slot];
    if (!l.active || l.puller != p ||
        (l.generation & 0xFF) != LeaseGeneration(id)) {
      return nullptr;
    }
    return &l;
  }

  // Ends lease |slot| without the bundle: |b| is wanted again.
  void ReclaimLease(uint32_t slot) {
    const int32_t p = leases_[slot].puller;
    const int32_t b = leases_[slot].bundle;
    EndLease(slot);
    DecCopies(b);
    if (!TestBit(have_, p, b)) SetWant(p, b);
    RelistWaiting(p);
    MarkDirtyPuller(p);
  }

  void Complete(int32_t p, uint64_t id) {
    Lease* l = FindLease(p, id);
    if (l != nullptr) {
      const int32_t b = l->bundle;
      EndLease(LeaseSlot(id));
      DecCopies(b);
      AddHave(p, b);
      completions_.Add(1);
      return;
    }
    // A late completion of a reclaimed lease: the data still arrived.
    const int32_t b = LeaseBundle(id);
    if (b >= 0 && b < num_bundles_ && hosts_[p].state == HostState::kActive) {
      AddHave(p, b);
    }
  }

  void Fail(int32_t p, uint64_t id) {
    Lease* l = FindLease(p, id);
    if (l == nullptr) return;
    const int32_t s = l->source;
    ReclaimLease(LeaseSlot(id));
    failures_.Add(1);
    Host& h = hosts_[p];
    if (std::find(h.excluded.begin(), h.excluded.end(), s) ==
        h.excluded.end()) {
      h.excluded.push_back(s);
    }
    Host& src = hosts_[s];
    if (std::find(src.reporters.begin(), src.reporters.end(), p) ==
        src.reporters.end()) {
      src.reporters.push_back(p);
      if (static_cast<int32_t>(src.reporters.size()) >= unhealthy_reports_ &&
          !src.unhealthy) {
        if (IsSource(s)) Contribute(s, -1);
        src.unhealthy = true;
        Relist(s);
        unhealthy_.Add(1);
        CheckLost(s);
      }
    }
  }

  // Drops every lease of |p| as a puller.
  void ReclaimAll(int32_t p) {
    while (!hosts_[p].leases.empty()) ReclaimLease(hosts_[p].leases.back());
  }

  void Suspend(int32_t r, HostState state) {
    Host& h = hosts_[r];
    if (h.state == state || h.state == HostState::kDead) return;
    if (IsSource(r)) Contribute(r, -1);
    h.state = state;
    if (state == HostState::kDead && !h.done) {
      ++dead_undone_;
      MaybeComplete();
    }
    ReclaimAll(r);
    Relist(r);
    RelistWaiting(r);
    if (state == HostState::kSuspended) suspended_.Add(1);
    CheckLost(r);
  }

  void Resume(int32_t r) {
    Host& h = hosts_[r];
    if (h.state != HostState::kSuspended) return;
    h.state = HostState::kActive;
    if (IsSource(r)) Contribute(r, 1);
    Relist(r);
    MarkDirtySource(r);
  }

  // |r| lost everything it received (e.g. it restarted). It comes back as a
  // fresh, healthy host without data.
  void ResetHost(int32_t r) {
    Host& h = hosts_[r];
    ReclaimAll(r);
    if (IsSource(r)) Contribute(r, -1);
    h.state = HostState::kActive;
    h.unhealthy = false;
    h.reporters.clear();
    h.seed_pending = false;
    std::fill(Row(have_, r), Row(have_, r) + num_words_, 0);
    h.have_count = 0;
    for (int32_t b = 0; b < num_bundles_; ++b) SetBit(want_, r, b);
    h.want_count = num_bundles_;
    RecomputeSummaries(r);
    if (h.done) {
      h.done = false;
      done_gauge_.Set(--done_count_);
    }
    Relist(r);
    for (int32_t b = 0; b < num_bundles_ && !aborted_; ++b) CheckBundle(b);
  }

  void Seeded(int32_t r) {
    Host& h = hosts_[r];
    if (!h.seed_pending || h.state == HostState::kDead) return;
    if (IsSource(r)) {
      for (int32_t b : seeded_[r]) --pending_[b];
    }
    h.seed_pending = false;
    for (int32_t b : seeded_[r]) AddHave(r, b);
  }

  // ---- Matching ------------------------------------------------------------

  bool Excluded(int32_t p, int32_t s) const {
    const auto& ex = hosts_[p].excluded;
    return std::find(ex.begin(), ex.end(), s) != ex.end();
  }

  // Whether |s| holds a bundle that |p| wants, and may serve |p|.
  bool Compatible(int32_t p, int32_t s) {
    if (s == p || (have_summary_[s] & want_summary_[p]) == 0) return false;
    const uint64_t* hv = Row(have_, s);
    const uint64_t* wv = Row(want_, p);
    bool hit = false;
    for (int32_t w = 0; w < num_words_; ++w) {
      if ((hv[w] & wv[w]) != 0) {
        hit = true;
        break;
      }
    }
    return hit && !Excluded(p, s);
  }

  // The rarest bundle that |s| holds and |p| wants (they are compatible).
  int32_t Rarest(int32_t s, int32_t p) {
    const uint64_t* hv = Row(have_, s);
    const uint64_t* wv = Row(want_, p);
    int32_t n = 0;
    for (int32_t w = 0; w < num_words_; ++w) {
      n += std::popcount(hv[w] & wv[w]);
    }
    if (n <= kSparseIntersection) {
      int32_t best = -1;
      // Start at a puller-dependent word, so that ties spread.
      const int32_t start = p % num_words_;
      for (int32_t i = 0; i < num_words_; ++i) {
        const int32_t w = (start + i) % num_words_;
        for (uint64_t x = hv[w] & wv[w]; x != 0; x &= x - 1) {
          const int32_t b = w * 64 + std::countr_zero(x);
          if (best < 0 || copies_[b] < copies_[best]) best = b;
        }
      }
      return best;
    }
    for (int32_t i = 0; i < num_bundles_; ++i) {
      const int32_t b = order_[i];
      const uint64_t bit = uint64_t{1} << (b & 63);
      if ((hv[b >> 6] & wv[b >> 6] & bit) != 0) return b;
    }
    return -1;
  }

  void Grant(int32_t p, int32_t s, int32_t b) {
    Host& h = hosts_[p];
    const uint32_t slot = AllocLease();
    Lease& l = leases_[slot];
    l.puller = p;
    l.source = s;
    l.bundle = b;
    l.index = static_cast<int32_t>(h.leases.size());
    l.active = true;
    h.leases.push_back(slot);
    ++h.inflight;
    ClearWant(p, b);
    ++hosts_[s].uploads;
    Relist(s);
    IncCopies(b);
    const uint64_t id = MakeLeaseId(b, l.generation, slot);
    const absl::Time expiry = now_ + lease_timeout_;
    h.reply.grants.push_back(PullGrant{
        .bundle_id = b, .source_replica = s, .lease_id = id, .expiry = expiry});
    --h.remaining;
    if (!h.in_outbox) {
      h.in_outbox = true;
      granted_.push_back(p);
    }
    lease_queue_.push_back(TimedLease{expiry, id});
    leases_gauge_.Set(++leases_in_flight_);
    grants_.Add(1);
    RelistWaiting(p);
  }

  // Serves waiting puller |p| from the free sources, least loaded first. A
  // grant moves its source to the next bucket, so a puller only gets several
  // grants from one source when no less loaded source can serve it.
  void PullerScan(int32_t p) {
    Host& h = hosts_[p];
    for (int32_t load = 0; load < upload_cap_ && h.waiting; ++load) {
      std::vector<int32_t>& bucket = free_[load];
      if (bucket.empty()) continue;
      const size_t start = scan_cursor_++ % bucket.size();
      size_t i = 0;
      while (h.waiting && i < bucket.size()) {
        const int32_t s = bucket[(start + i) % bucket.size()];
        if (Compatible(p, s)) {
          // Removes |s| from |bucket|.
          Grant(p, s, Rarest(s, p));
        } else {
          ++i;
        }
      }
    }
  }

  // Lets source |s| serve the waiting pullers, in FIFO order.
  void SourceScan(int32_t s) {
    int32_t p = wait_head_;
    while (p >= 0 && Listed(s)) {
      const int32_t next = hosts_[p].wait_next;
      if (Compatible(p, s)) Grant(p, s, Rarest(s, p));
      p = next;
    }
  }

  void Match() {
    if (aborted_) {
      dirty_pullers_.clear();
      dirty_sources_.clear();
      return;
    }
    for (size_t i = 0; i < dirty_pullers_.size(); ++i) {
      const int32_t p = dirty_pullers_[i];
      hosts_[p].dirty_puller = false;
      if (hosts_[p].waiting) PullerScan(p);
    }
    dirty_pullers_.clear();
    for (size_t i = 0; i < dirty_sources_.size(); ++i) {
      const int32_t s = dirty_sources_[i];
      hosts_[s].dirty_source = false;
      if (Listed(s) && wait_head_ >= 0) SourceScan(s);
    }
    dirty_sources_.clear();
  }

  // Answers every parked request that got grants in this batch, and the
  // requests that asked not to be parked.
  void FinishBatch() {
    for (int32_t p : granted_) {
      hosts_[p].in_outbox = false;
      if (hosts_[p].parked) Reply(p, PullReplyKind::kGrants);
    }
    granted_.clear();
    for (const auto& [p, seq] : immediate_) {
      if (hosts_[p].parked && hosts_[p].seq == seq) {
        Reply(p, PullReplyKind::kGrants);
      }
    }
    immediate_.clear();
  }

  // ---- Requests ------------------------------------------------------------

  void Execute(Op* op, absl::Time now) {
    now_ = now;
    switch (op->kind) {
      case OpKind::kRequest:
        ApplyRequest(op);
        break;
      case OpKind::kSeeded:
        if (!aborted_) Seeded(op->replica);
        break;
      case OpKind::kDead:
        if (!aborted_) {
          Reply(op->replica, PullReplyKind::kAborted,
                absl::UnavailableError("Replica was marked dead"));
          Suspend(op->replica, HostState::kDead);
        }
        break;
      case OpKind::kAbort: {
        absl::Status status;
        {
          absl::MutexLock lock(shared_->mu);
          status = shared_->status;
        }
        if (!status.ok()) AbortLocal(status);
        break;
      }
      case OpKind::kStop:
        break;
    }
    delete op;
  }

  void ApplyRequest(Op* op) {
    PullRequest& req = op->request;
    const int32_t p = req.replica;
    Host& h = hosts_[p];
    requests_.Add(1);
    if (aborted_ || h.state == HostState::kDead) {
      PullReply reply;
      reply.kind = PullReplyKind::kAborted;
      reply.status =
          aborted_ ? abort_status_ : absl::UnavailableError("Replica is dead");
      outbox_.emplace_back(std::move(op->done), std::move(reply));
      return;
    }
    h.last_seen = now_;
    // A new request of the host answers its parked one.
    Reply(p, PullReplyKind::kGrants);
    if (req.lost_data) {
      ResetHost(p);
    } else if (h.state == HostState::kSuspended) {
      Resume(p);
    }
    if (req.seeded) Seeded(p);
    for (uint64_t id : req.completed) Complete(p, id);
    for (uint64_t id : req.failed) Fail(p, id);
    h.waiter = std::move(op->done);
    h.reply = PullReply();
    h.parked = true;
    parked_.Set(++parked_count_);
    if (aborted_) {
      Reply(p, PullReplyKind::kAborted, abort_status_);
      return;
    }
    if (h.done) {
      Reply(p, PullReplyKind::kDone);
      return;
    }
    h.remaining = std::min(req.max_grants, grant_batch_ - h.inflight);
    if (h.remaining <= 0) {
      Reply(p, PullReplyKind::kGrants);
      return;
    }
    const absl::Duration poll = std::min(req.long_poll, long_poll_timeout_);
    if (poll > absl::ZeroDuration()) {
      poll_queue_.push(TimedPoll{now_ + poll, p, h.seq});
    } else {
      immediate_.emplace_back(p, h.seq);
    }
    RelistWaiting(p);
    MarkDirtyPuller(p);
  }

  void AbortLocal(const absl::Status& status) {
    if (aborted_) return;
    aborted_ = true;
    abort_status_ = status;
    for (int32_t p = 0; p < num_replicas_; ++p) {
      if (hosts_[p].parked) Reply(p, PullReplyKind::kAborted, status);
    }
  }

  void CancelAll(const absl::Status& status) {
    for (int32_t p = 0; p < num_replicas_; ++p) {
      if (hosts_[p].parked) Reply(p, PullReplyKind::kAborted, status);
    }
  }

  // ---- Timers --------------------------------------------------------------

  absl::Time NextTimer() const {
    absl::Time next = next_sweep_;
    if (!lease_queue_.empty()) {
      next = std::min(next, lease_queue_.front().expiry);
    }
    if (!poll_queue_.empty()) {
      next = std::min(next, poll_queue_.top().deadline);
    }
    return next;
  }

  void ProcessTimersIfDue(absl::Time now) {
    if (now >= NextTimer()) ProcessTimers(now);
  }

  void ProcessTimers(absl::Time now) {
    now_ = now;
    while (!lease_queue_.empty() && lease_queue_.front().expiry <= now) {
      const uint64_t id = lease_queue_.front().id;
      lease_queue_.pop_front();
      const uint32_t slot = LeaseSlot(id);
      if (slot >= leases_.size() || !leases_[slot].active ||
          (leases_[slot].generation & 0xFF) != LeaseGeneration(id)) {
        continue;
      }
      expired_.Add(1);
      const int32_t p = leases_[slot].puller;
      if (hosts_[p].parked) {
        // The puller is alive; only this lease is lost.
        ReclaimLease(slot);
      } else {
        Suspend(p, HostState::kSuspended);
      }
    }
    while (!poll_queue_.empty() && poll_queue_.top().deadline <= now) {
      const TimedPoll poll = poll_queue_.top();
      poll_queue_.pop();
      Host& h = hosts_[poll.puller];
      if (h.parked && h.seq == poll.seq) {
        poll_timeouts_.Add(1);
        Reply(poll.puller, PullReplyKind::kGrants);
      }
    }
    if (now >= next_sweep_) {
      next_sweep_ = now + sweep_interval_;
      for (int32_t r = 0; r < num_replicas_ && !aborted_; ++r) {
        const Host& h = hosts_[r];
        if (!h.done && !h.parked && h.state == HostState::kActive &&
            now - h.last_seen >= lease_timeout_) {
          Suspend(r, HostState::kSuspended);
        }
      }
    }
  }

  // ---- Thread --------------------------------------------------------------

  void Run() {
    int64_t batches_since_cpu = 0;
    while (true) {
      bool stop = false;
      int32_t n = 0;
      const absl::Time now = clock_();
      const int64_t start_ns = absl::GetCurrentTimeNanos();
      while (n < max_batch_) {
        QueueNode* node = queue_.Pop();
        if (node == nullptr) break;
        Op* op = static_cast<Op*>(node);
        if (op->kind == OpKind::kStop) {
          delete op;
          stop = true;
          break;
        }
        if (op->kind == OpKind::kRequest) {
          batch_enqueue_ns_.push_back(op->enqueue_ns);
        }
        Execute(op, now);
        ++n;
      }
      const bool timers = now >= NextTimer();
      if (timers) ProcessTimers(now);
      if (n > 0 || timers) {
        Match();
        FinishBatch();
        Deliver(outbox_);
        const int64_t end_ns = absl::GetCurrentTimeNanos();
        const int64_t busy = end_ns - start_ns;
        busy_ns_.Add(busy);
        batch_hist_.Record(busy);
        for (int64_t enqueue_ns : batch_enqueue_ns_) {
          request_hist_.Record(end_ns - enqueue_ns);
        }
        batch_enqueue_ns_.clear();
        batches_.Add(1);
        max_batch_seen_.Max(n);
        shared_->PropagateIfNeeded();
        if (++batches_since_cpu >= 256) {
          batches_since_cpu = 0;
          cpu_ns_.Set(ThreadCpuNanos());
        }
      }
      if (stop) break;
      if (n == 0 && !timers) Sleep();
    }
    cpu_ns_.Set(ThreadCpuNanos());
    // Answer whatever is still queued.
    while (QueueNode* node = queue_.Pop()) {
      Op* op = static_cast<Op*>(node);
      if (op->kind == OpKind::kRequest) {
        PullReply reply;
        reply.kind = PullReplyKind::kAborted;
        reply.status = absl::CancelledError("Pull scheduler shut down");
        std::move(op->done)(std::move(reply));
      }
      delete op;
    }
  }

  void Sleep() {
    for (int32_t i = 0; i < kSpinIterations; ++i) {
      if (!queue_.Empty()) return;
      CpuRelax();
    }
    cpu_ns_.Set(ThreadCpuNanos());
    const absl::Time next = NextTimer();
    absl::MutexLock lock(wake_mu_);
    sleeping_.store(true, std::memory_order_seq_cst);
    if (queue_.Empty()) {
      // `clock_` may be virtual; never sleep past a real-time bound.
      const absl::Duration wait = std::min(next - clock_(), absl::Seconds(1));
      wake_cv_.WaitWithTimeout(&wake_mu_, std::max(wait, absl::ZeroDuration()));
    }
    sleeping_.store(false, std::memory_order_relaxed);
  }

  const int32_t host_;
  const int32_t num_replicas_;
  const int32_t num_bundles_;
  const int32_t num_words_;
  const int32_t grant_batch_;
  const int32_t upload_cap_;
  const int32_t unhealthy_reports_;
  const int32_t max_batch_;
  const absl::Duration lease_timeout_;
  const absl::Duration long_poll_timeout_;
  const absl::Duration sweep_interval_;
  const bool inline_;
  const std::function<absl::Time()> clock_;
  const std::function<void(int32_t)> thread_init_;
  Shared* const shared_;
  const std::vector<std::vector<int32_t>>& seeded_;

  // Inline execution: guards everything below.
  absl::Mutex mu_;

  std::vector<Host> hosts_;
  std::vector<uint64_t> have_;
  std::vector<uint64_t> want_;
  std::vector<uint64_t> have_summary_;
  std::vector<uint64_t> want_summary_;
  // Per bundle: live holders, seeds that may still report it, and live
  // holders plus leases in flight (the rarity key).
  std::vector<int32_t> live_;
  std::vector<int32_t> pending_;
  std::vector<int32_t> copies_;
  // Bundles sorted by `copies_`; `group_start_[c]` is the first position with
  // `copies_ >= c`.
  std::vector<int32_t> order_;
  std::vector<int32_t> position_;
  std::vector<int32_t> group_start_;
  // `free_[k]`: sources with `k` uploads that can take another one.
  std::vector<std::vector<int32_t>> free_;
  size_t scan_cursor_ = 0;
  int32_t wait_head_ = -1;
  int32_t wait_tail_ = -1;
  std::vector<int32_t> dirty_sources_;
  std::vector<int32_t> dirty_pullers_;
  std::vector<int32_t> granted_;
  std::vector<std::pair<int32_t, uint32_t>> immediate_;
  std::vector<std::pair<PullReplyCallback, PullReply>> outbox_;
  std::vector<Lease> leases_;
  std::vector<uint32_t> free_leases_;
  std::deque<TimedLease> lease_queue_;
  std::priority_queue<TimedPoll, std::vector<TimedPoll>, std::greater<>>
      poll_queue_;
  absl::Time next_sweep_;
  absl::Time now_;
  int32_t done_count_ = 0;
  // Replicas marked dead before they were done.
  int32_t dead_undone_ = 0;
  int32_t parked_count_ = 0;
  int32_t leases_in_flight_ = 0;
  bool complete_ = false;
  bool aborted_ = false;
  absl::Status abort_status_;
  bool stopped_ = false;

  // Threaded execution.
  MpscQueue queue_;
  std::thread thread_;
  absl::Mutex wake_mu_;
  absl::CondVar wake_cv_;
  std::atomic<bool> sleeping_{false};

  // Statistics (written by the shard only).
  int32_t stats_host_ = 0;
  Counter requests_, grants_, completions_, failures_, expired_, poll_timeouts_,
      suspended_, unhealthy_, batches_, max_batch_seen_, busy_ns_, cpu_ns_,
      parked_, done_gauge_, leases_gauge_;
  LatencyHistogram batch_hist_;
  LatencyHistogram request_hist_;
  std::vector<int64_t> batch_enqueue_ns_;
};

int32_t EffectiveGrantBatchSize(const PullSchedulerOptions& options) {
  const int32_t batch = std::max(1, options.grant_batch_size);
  if (options.allow_grant_batch_above_upload_cap) return batch;
  return std::min(batch,
                  std::max(1, options.max_concurrent_uploads_per_source));
}

void PullScheduler::Shared::PropagateIfNeeded() {
  if (!propagate_abort.exchange(false, std::memory_order_acq_rel)) return;
  for (Shard* shard : shards) {
    Op* op = new Op;
    op->kind = OpKind::kAbort;
    shard->Post(op);
  }
}

absl::StatusOr<std::unique_ptr<PullScheduler>> PullScheduler::Create(
    int32_t num_replicas, int32_t num_hosts, int32_t num_bundles,
    std::vector<std::vector<int32_t>> seeded_bundles,
    PullSchedulerOptions options) {
  if (num_replicas <= 0 || num_hosts <= 0 || num_bundles <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Pull scheduler needs replicas, hosts and bundles, got ",
                     num_replicas, ", ", num_hosts, ", ", num_bundles));
  }
  if (num_bundles >= (1 << 23)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Too many bundles: ", num_bundles));
  }
  if (static_cast<int32_t>(seeded_bundles.size()) != num_replicas) {
    return absl::InvalidArgumentError(
        absl::StrCat("Seeded bundles given for ", seeded_bundles.size(),
                     " replicas, expected ", num_replicas));
  }
  std::vector<bool> seeded(num_bundles, false);
  for (std::vector<int32_t>& bundles : seeded_bundles) {
    std::sort(bundles.begin(), bundles.end());
    bundles.erase(std::unique(bundles.begin(), bundles.end()), bundles.end());
    for (int32_t b : bundles) {
      if (b < 0 || b >= num_bundles) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Seeded bundle ", b, " out of range for ", num_bundles));
      }
      seeded[b] = true;
    }
  }
  for (int32_t b = 0; b < num_bundles; ++b) {
    if (!seeded[b]) {
      return absl::InvalidArgumentError(
          absl::StrCat("Bundle ", b, " has no seed"));
    }
  }
  if (!options.clock) options.clock = [] { return absl::Now(); };
  auto scheduler = absl::WrapUnique(
      new PullScheduler(num_replicas, num_hosts, num_bundles, options));
  scheduler->seeded_bundles_ = std::move(seeded_bundles);
  for (int32_t h = 0; h < num_hosts; ++h) {
    scheduler->shards_.push_back(std::make_unique<Shard>(
        h, num_replicas, num_bundles, scheduler->seeded_bundles_,
        scheduler->options_, scheduler->shared_.get()));
    scheduler->shared_->shards.push_back(scheduler->shards_.back().get());
  }
  for (auto& shard : scheduler->shards_) shard->Start();
  return scheduler;
}

PullScheduler::PullScheduler(int32_t num_replicas, int32_t num_hosts,
                             int32_t num_bundles, PullSchedulerOptions options)
    : num_replicas_(num_replicas),
      num_hosts_(num_hosts),
      num_bundles_(num_bundles),
      options_(std::move(options)),
      shared_(std::make_unique<Shared>()) {}

PullScheduler::~PullScheduler() {
  for (auto& shard : shards_) shard->Stop();
}

int32_t PullScheduler::LeaseBundle(uint64_t lease_id) {
  return static_cast<int32_t>(lease_id >> 40);
}

std::vector<std::vector<int32_t>> PullScheduler::SeededBundles(
    const SeedLayout& layout, int32_t num_replicas) {
  std::vector<std::vector<int32_t>> seeded(std::max(0, num_replicas));
  for (size_t g = 0; g < layout.stripe_seeds.size(); ++g) {
    if (g >= layout.stripe_bundles.size()) break;
    for (int32_t r : layout.stripe_seeds[g]) {
      if (r < 0 || r >= num_replicas) continue;
      seeded[r].insert(seeded[r].end(), layout.stripe_bundles[g].begin(),
                       layout.stripe_bundles[g].end());
    }
  }
  return seeded;
}

void PullScheduler::Submit(PullRequest request, PullReplyCallback done) {
  if (request.replica < 0 || request.replica >= num_replicas_ ||
      request.host < 0 || request.host >= num_hosts_) {
    PullReply reply;
    reply.kind = PullReplyKind::kAborted;
    reply.status = absl::InvalidArgumentError(
        absl::StrCat("Pull request from replica ", request.replica, " host ",
                     request.host, " out of range for ", num_replicas_,
                     " replicas with ", num_hosts_, " hosts"));
    std::move(done)(std::move(reply));
    return;
  }
  Op* op = new Op;
  op->kind = OpKind::kRequest;
  const int32_t host = request.host;
  op->request = std::move(request);
  op->done = std::move(done);
  shards_[host]->Post(op);
}

void PullScheduler::MarkSeeded(int32_t replica) {
  if (replica < 0 || replica >= num_replicas_) return;
  for (auto& shard : shards_) {
    Op* op = new Op;
    op->kind = OpKind::kSeeded;
    op->replica = replica;
    shard->Post(op);
  }
}

void PullScheduler::MarkReplicaDead(int32_t replica) {
  if (replica < 0 || replica >= num_replicas_) return;
  for (auto& shard : shards_) {
    Op* op = new Op;
    op->kind = OpKind::kDead;
    op->replica = replica;
    shard->Post(op);
  }
}

void PullScheduler::Abort(absl::Status status) {
  if (status.ok()) status = absl::AbortedError("Transfer aborted");
  shared_->RaiseAbort(status).IgnoreError();
  shared_->PropagateIfNeeded();
}

void PullScheduler::AdvanceTime() {
  if (!options_.inline_execution) return;
  for (auto& shard : shards_) shard->AdvanceInline();
}

absl::Time PullScheduler::NextTimer() const {
  absl::Time next = absl::InfiniteFuture();
  if (!options_.inline_execution) return next;
  for (const auto& shard : shards_) {
    next = std::min(next, shard->NextTimerInline());
  }
  return next;
}

absl::Status PullScheduler::WaitForCompletion(absl::Time deadline) {
  absl::MutexLock lock(shared_->mu);
  auto finished = [this]() ABSL_SHARED_LOCKS_REQUIRED(shared_->mu) {
    return shared_->complete || !shared_->status.ok();
  };
  shared_->mu.AwaitWithDeadline(absl::Condition(&finished), deadline);
  if (shared_->complete) return absl::OkStatus();
  if (!shared_->status.ok()) return shared_->status;
  return absl::DeadlineExceededError(
      "Pull phase did not complete before the deadline");
}

bool PullScheduler::complete() const {
  absl::MutexLock lock(shared_->mu);
  return shared_->complete;
}

absl::Status PullScheduler::status() const {
  absl::MutexLock lock(shared_->mu);
  return shared_->status;
}

std::vector<PullShardStats> PullScheduler::GetStats() const {
  std::vector<PullShardStats> stats;
  stats.reserve(shards_.size());
  for (const auto& shard : shards_) stats.push_back(shard->Stats());
  return stats;
}

}  // namespace v3
}  // namespace weight_sync
}  // namespace tpu_raiden
