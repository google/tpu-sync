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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_STORE_MONITOR_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_STORE_MONITOR_H_

#include <functional>
#include <memory>
#include <thread>  // NOLINT(build/c++11)

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/global_registry/global_registry.pb.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_client.h"

namespace tpu_raiden {
namespace kv_cache {

// The store's periodic reporting thread: heartbeats the store's mutable
// status (see StoreStatus) to the global registry, which both refreshes the
// registration TTL and feeds the registry's capacity ranking. A heartbeat
// answered with NotFound means the registry no longer holds a live
// registration (registry restart or TTL expiry); heartbeats carry no
// coordinates, so the monitor re-registers through the callback instead.
//
// With a SweepFn, the same thread also schedules the store's evict sweep --
// one thread total, a tradeoff to balance the number of threads raiden is
// managing. The sweep runs once per sweep period, immediately when
// NudgeSweep() is called, and step by step while the callback keeps
// returning true; the heartbeat check is interleaved between steps, so
// sweeping delays a heartbeat by at most one bounded step.
//
// With a RepublishFn, a successful re-register also kicks off an inventory
// republish -- the lapse that forced the re-register means the registry may
// have purged (or, after a registry restart, lost) every block entry this
// store owns. It runs on the same bounded-step contract as the sweep, and
// RequestRepublish() starts one without a lapse (the boot-after-recovery
// case).
//
// Owned by KVCacheStore; deliberately talks to the store only through the
// callbacks so it never sees store internals.
class StoreMonitor {
 public:
  static constexpr absl::Duration kDefaultHeartbeatPeriod = absl::Seconds(300);
  static constexpr absl::Duration kDefaultSweepPeriod = absl::Seconds(30);

  struct Options {
    absl::Duration heartbeat_period = kDefaultHeartbeatPeriod;
    absl::Duration sweep_period = kDefaultSweepPeriod;
  };

  // Snapshots the store's current status for one heartbeat.
  using StatusFn = std::function<global_registry::StoreStatus()>;
  // Re-publishes the store's full registration (coordinates and TTL).
  using ReregisterFn = std::function<absl::Status()>;
  // One bounded step of evict-sweep work. Returns true if more work remains,
  // in which case the monitor runs the next step right away; false parks the
  // sweep until the next period or request.
  using SweepFn = std::function<bool()>;
  // One bounded step of inventory republish; same contract as SweepFn.
  using RepublishFn = std::function<bool()>;

  StoreMonitor(const Options& options,
               std::shared_ptr<global_registry::GlobalRegistryClient>
                   registry_client,
               RaidenId raiden_id, StatusFn status_fn,
               ReregisterFn reregister_fn, SweepFn sweep_fn = nullptr,
               RepublishFn republish_fn = nullptr);

  // Stops the threads; the callbacks must outlive this call, not the object.
  ~StoreMonitor();

  StoreMonitor(const StoreMonitor&) = delete;
  StoreMonitor& operator=(const StoreMonitor&) = delete;

  // Starts the monitor thread. The first heartbeat fires one period from
  // now: the caller registers before starting the monitor, so the registry
  // is already fresh. Call at most once.
  void Start();

  // Stops and joins the thread. Idempotent, and terminal: a stopped monitor
  // cannot be restarted.
  void Stop();

  // Request to run the sweep now instead of at its next period. Called from
  // the store's allocation path when free blocks dip below the sweep's low
  // watermark, so pressure is acted on immediately rather than discovered a
  // sweep period later. No-op without a SweepFn or after Stop().
  void RequestSweep();

  // Request an inventory republish without waiting for a registration lapse.
  // Called after a crash restart recovers blocks from the local manifest,
  // whose registry entries a purge may have removed in the meantime. No-op
  // without a RepublishFn or after Stop().
  void RequestRepublish();

 private:
  void Loop();
  // Returns true ONLY when the heartbeat came back NotFound (the registry
  // holds no live registration) AND reregister_fn_ then succeeded. That
  // combination is what triggers the inventory republish: the lapse means
  // the registry may have purged this store's block entries, and the fresh
  // registration means it is ready to take them back. A healthy heartbeat,
  // a transport error, or a failed re-register all return false.
  bool HeartbeatOnce();

  const Options options_;
  const std::shared_ptr<global_registry::GlobalRegistryClient>
      registry_client_;
  const RaidenId raiden_id_;
  const StatusFn status_fn_;
  const ReregisterFn reregister_fn_;
  const SweepFn sweep_fn_;
  const RepublishFn republish_fn_;

  absl::Mutex mu_;
  // Signaled on Stop(), RequestSweep() and RequestRepublish(); the loop waits
  // on it with a deadline, so a signal is its only early wake-up.
  absl::CondVar cv_;
  bool stop_ ABSL_GUARDED_BY(mu_) = false;
  bool sweep_requested_ ABSL_GUARDED_BY(mu_) = false;
  bool republish_requested_ ABSL_GUARDED_BY(mu_) = false;
  std::thread thread_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_STORE_MONITOR_H_
