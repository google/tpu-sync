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

#include "tpu_sync/core/controller/worker_registry.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tpu_sync/core/controller/worker_service_client.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"

namespace tpu_raiden {
namespace core {
namespace controller {

absl::Status WorkerRegistry::RegisterWorker(const WorkerRegistration& reg) {
  if (reg.worker_id.empty()) {
    return absl::InvalidArgumentError("worker_id cannot be empty");
  }
  if (reg.worker_service_client == nullptr &&
      reg.raiden_worker_endpoint.empty() &&
      reg.raiden_transfer_endpoints.empty()) {
    return absl::InvalidArgumentError(
        "at least one of raiden_worker_endpoint or raiden_transfer_endpoints"
        " must be provided, or worker_service_client must be non-null");
  }

  WorkerRegistration entry = reg;
  if (entry.worker_service_client == nullptr &&
      !entry.raiden_worker_endpoint.empty()) {
    entry.worker_service_client =
        std::make_shared<::tpu_raiden::controller::WorkerServiceClient>(
            ::tpu_raiden::controller::CreateWorkerServiceChannel(
                entry.raiden_worker_endpoint));
  }

  absl::MutexLock lock(mutex_);
  // Enforce unique node_id across distinct workers (node_id 0 == unset is
  // exempt). worker_id uniqueness is inherent: workers_ is keyed by worker_id,
  // so a re-registration under the same worker_id is an update, not a collision.
  if (entry.node_id != 0) {
    for (const auto& [existing_id, existing] : workers_) {
      if (existing_id != entry.worker_id &&
          existing.node_id == entry.node_id) {
        return absl::FailedPreconditionError(absl::StrCat(
            "node_id ", entry.node_id, " is already registered by worker '",
            existing_id, "'; node_id must be unique per controller (rejecting '",
            entry.worker_id, "')"));
      }
    }
  }
  if (on_register_cb_) {
    absl::Status status = on_register_cb_(entry);
    if (!status.ok()) return status;
  }
  workers_[entry.worker_id] = std::move(entry);
  return absl::OkStatus();
}

absl::Status WorkerRegistry::RegisterWorker(
    absl::string_view worker_id, absl::string_view raiden_worker_endpoint,
    const std::vector<::tpu_raiden::RaidenTransferEndpoint>&
        raiden_transfer_endpoints,
    int64_t node_id) {
  return RegisterWorker(WorkerRegistration{
      .worker_id = std::string(worker_id),
      .raiden_worker_endpoint = std::string(raiden_worker_endpoint),
      .raiden_transfer_endpoints = raiden_transfer_endpoints,
      .node_id = node_id,
  });
}

std::vector<WorkerRegistration> WorkerRegistry::GetRegisteredWorkers() const {
  absl::MutexLock lock(mutex_);
  std::vector<WorkerRegistration> result;
  result.reserve(workers_.size());
  for (const auto& [_, reg] : workers_) {
    result.push_back(reg);
  }
  return result;
}

bool WorkerRegistry::AwaitWorkerCount(size_t count,
                                      absl::Duration timeout) const {
  auto reached = [this, count]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return workers_.size() >= count;
  };
  absl::MutexLock lock(mutex_);
  return mutex_.AwaitWithTimeout(absl::Condition(&reached), timeout);
}

absl::StatusOr<WorkerRegistration> WorkerRegistry::GetWorker(
    absl::string_view worker_id) const {
  absl::MutexLock lock(mutex_);
  auto it = workers_.find(worker_id);
  if (it == workers_.end()) {
    return absl::NotFoundError(absl::StrCat("Worker not found: ", worker_id));
  }
  return it->second;
}

}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden
