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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_ACCEPT_BACKOFF_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_ACCEPT_BACKOFF_H_

#include <string>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"

namespace tpu_raiden {

// What an accept loop should do after accept(), or the poll() that guards it,
// reports an error.
enum class AcceptErrorAction {
  // Routine, and the listener is healthy: retry straight away.
  kRetry,
  // The process is out of a resource. The connection that could not be
  // accepted stays queued, so an immediate retry fails the same way and the
  // loop spins at 100% CPU, starving the very threads that would free the
  // resource. Sleep first.
  kBackoff,
  // The listening descriptor is not usable any more: leave the loop.
  kStop,
};

AcceptErrorAction ClassifyAcceptError(int err);

// Delay to wait after |current| (pass zero for the first backoff). Capped, so
// a stop request is never delayed by more than one step.
absl::Duration NextAcceptBackoff(absl::Duration current);

// Backoff state for one accept loop. Not thread-safe: each loop owns one on
// its own stack.
class AcceptBackoff {
 public:
  // |name| identifies the listener in log messages.
  explicit AcceptBackoff(absl::string_view name);

  // Handles a failed accept(). Sleeps for errors worth retrying slowly, and
  // returns false if the caller must leave its accept loop.
  //
  // Callers check their own stopping flag before calling this, so a kStop
  // error here always means the descriptor died unexpectedly and is logged as
  // an error rather than as routine shutdown.
  bool OnError(int err);

  // Resets the delay. Call after a successful accept().
  void OnSuccess();

  absl::Duration delay_for_testing() const { return delay_; }

 private:
  const std::string name_;
  absl::Duration delay_ = absl::ZeroDuration();
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_ACCEPT_BACKOFF_H_
