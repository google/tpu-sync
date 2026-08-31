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

#include "tpu_sync/kv_cache/store_monitor.h"

#include <algorithm>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_client.h"

namespace tpu_raiden {
namespace kv_cache {

StoreMonitor::StoreMonitor(
    const Options& options,
    std::shared_ptr<global_registry::GlobalRegistryClient> registry_client,
    RaidenId raiden_id, StatusFn status_fn, ReregisterFn reregister_fn,
    SweepFn sweep_fn, RepublishFn republish_fn)
    : options_(options),
      registry_client_(std::move(registry_client)),
      raiden_id_(std::move(raiden_id)),
      status_fn_(std::move(status_fn)),
      reregister_fn_(std::move(reregister_fn)),
      sweep_fn_(std::move(sweep_fn)),
      republish_fn_(std::move(republish_fn)) {}

StoreMonitor::~StoreMonitor() { Stop(); }

void StoreMonitor::Start() { thread_ = std::thread(&StoreMonitor::Loop, this); }

void StoreMonitor::Stop() {
  {
    absl::MutexLock lock(mu_);
    stop_ = true;
    cv_.Signal();
  }
  if (thread_.joinable()) {
    thread_.join();
  }
}

void StoreMonitor::RequestSweep() {
  if (sweep_fn_ == nullptr) {
    return;
  }
  absl::MutexLock lock(mu_);
  sweep_requested_ = true;
  cv_.Signal();
}

void StoreMonitor::RequestRepublish() {
  if (republish_fn_ == nullptr) {
    return;
  }
  absl::MutexLock lock(mu_);
  republish_requested_ = true;
  cv_.Signal();
}

void StoreMonitor::Loop() {
  const bool sweeping = sweep_fn_ != nullptr;
  absl::Time next_heartbeat = absl::Now() + options_.heartbeat_period;
  absl::Time next_sweep = absl::Now() + options_.sweep_period;
  // True while a sweep request is pending or the last sweep step reported more
  // work; the loop then skips the wait, but still passes the heartbeat
  // check between steps -- so a long drain delays a heartbeat by at most
  // one bounded step. republish_now works the same way; the sweep step runs
  // first so republishing never delays pressure relief.
  bool sweep_now = false;
  bool republish_now = false;
  while (true) {
    {
      absl::MutexLock lock(mu_);
      const absl::Time wake =
          sweeping ? std::min(next_heartbeat, next_sweep) : next_heartbeat;
      while (!stop_ && !sweep_requested_ && !republish_requested_ &&
             !sweep_now && !republish_now && absl::Now() < wake) {
        cv_.WaitWithDeadline(&mu_, wake);
      }
      if (stop_) {
        return;
      }
      sweep_now = sweep_now || sweep_requested_;
      sweep_requested_ = false;
      republish_now = republish_now || republish_requested_;
      republish_requested_ = false;
    }
    if (absl::Now() >= next_heartbeat) {
      if (HeartbeatOnce() && republish_fn_ != nullptr) {
        republish_now = true;
      }
      next_heartbeat = absl::Now() + options_.heartbeat_period;
    }
    if (sweeping && (sweep_now || absl::Now() >= next_sweep)) {
      sweep_now = sweep_fn_();
      next_sweep = absl::Now() + options_.sweep_period;
    }
    if (republish_fn_ != nullptr && republish_now) {
      republish_now = republish_fn_();
    }
  }
}

bool StoreMonitor::HeartbeatOnce() {
  absl::Status status = registry_client_->Heartbeat(raiden_id_, status_fn_());
  if (status.ok()) {
    return false;
  }
  if (!absl::IsNotFound(status)) {
    LOG(WARNING) << "Store heartbeat to the global registry failed: " << status
                 << ". Retrying next period; the registration expires if this "
                    "keeps failing.";
    return false;
  }
  absl::Status reregistered = reregister_fn_();
  if (reregistered.ok()) {
    LOG(INFO) << "Store registration lapsed at the global registry "
                 "(registry restart or expired TTL); re-registered.";
    return true;
  }
  LOG(WARNING) << "Store registration lapsed at the global registry and "
                  "re-registering failed: "
               << reregistered << ". Retrying next period.";
  return false;
}

}  // namespace kv_cache
}  // namespace tpu_raiden
