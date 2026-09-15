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

#include "tpu_sync/common/accept_backoff.h"

#include <algorithm>
#include <cerrno>
#include <cstring>

#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tpu_raiden {
namespace {

constexpr absl::Duration kInitialBackoff = absl::Milliseconds(10);
constexpr absl::Duration kMaxBackoff = absl::Milliseconds(500);

}  // namespace

AcceptErrorAction ClassifyAcceptError(int err) {
  switch (err) {
    // A signal arrived, or the peer went away before we accepted it. Both are
    // routine; backing off would penalise healthy peers that reconnect often.
    case EINTR:
    case ECONNABORTED:
    case EPROTO:
      return AcceptErrorAction::kRetry;
    // Out of descriptors, buffers or memory.
    case EMFILE:
    case ENFILE:
    case ENOBUFS:
    case ENOMEM:
      return AcceptErrorAction::kBackoff;
    // The descriptor is closed, or is no longer a listening socket. On Linux
    // EINVAL is what shutdown() on a listening socket reports, which is how
    // teardown unblocks accept().
    case EBADF:
    case EINVAL:
    case ENOTSOCK:
    case EOPNOTSUPP:
      return AcceptErrorAction::kStop;
    default:
      // Unknown, so assume it can repeat. Backing off is the safe default:
      // the worst case is a slower retry, not a spin.
      return AcceptErrorAction::kBackoff;
  }
}

absl::Duration NextAcceptBackoff(absl::Duration current) {
  if (current <= absl::ZeroDuration()) return kInitialBackoff;
  return std::min(2 * current, kMaxBackoff);
}

AcceptBackoff::AcceptBackoff(absl::string_view name) : name_(name) {}

bool AcceptBackoff::OnError(int err) {
  switch (ClassifyAcceptError(err)) {
    case AcceptErrorAction::kRetry:
      return true;
    case AcceptErrorAction::kBackoff:
      delay_ = NextAcceptBackoff(delay_);
      LOG_EVERY_N_SEC(WARNING, 1)
          << name_ << " accept() failed: " << std::strerror(err)
          << "; retrying in " << delay_;
      absl::SleepFor(delay_);
      return true;
    case AcceptErrorAction::kStop:
      LOG(ERROR) << name_ << " stopped accepting connections: "
                 << std::strerror(err);
      return false;
  }
  return false;
}

void AcceptBackoff::OnSuccess() { delay_ = absl::ZeroDuration(); }

}  // namespace tpu_raiden
