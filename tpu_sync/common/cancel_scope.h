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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CANCEL_SCOPE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CANCEL_SCOPE_H_

#include <cstdint>
#include <thread>  // NOLINT(build/c++11)
#include <utility>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

namespace tpu_raiden {

// Cancels a group of operations together, e.g. the requests that a server is
// handling when it shuts down. Cancellation is sticky: once Cancel() is called,
// cancelled() returns true forever. Each owner has its own scope, so cancelling
// one owner's scope leaves other owners' operations running.
//
// An operation can poll cancelled() between steps:
//
//   for (const Chunk& chunk : chunks) {
//     if (scope.cancelled()) return absl::CancelledError("Shutting down");
//     ABSL_RETURN_IF_ERROR(Send(chunk));
//   }
//
// To interrupt a blocking call instead, register a callback for Cancel() to
// run, and keep the returned Registration alive until the call returns.
// OnCancel() fails if the scope is already cancelled:
//
//   ABSL_ASSIGN_OR_RETURN(CancelScope::Registration registration,
//                         scope.OnCancel([fd] { shutdown(fd, SHUT_RDWR); }));
//   ssize_t n = read(fd, buffer, size);  // Returns early once cancelled.
//   if (scope.cancelled()) return absl::CancelledError("Shutting down");
//
// Destroying a Registration deregisters its callback. If Cancel() is running
// the callback on another thread, the destructor waits for it to return, so
// never destroy a Registration while holding a lock that its callback takes.
//
// Thread-safe. Destroying the scope cancels it (running any remaining callbacks
// on the destroying thread), then waits until every registration is destroyed
// and a Cancel() call running the callbacks on another thread returns. Until
// then, registration holders, callbacks, and captures' destructors may keep
// using the scope, but no other call may start. The destroying thread must not
// hold or be destroying a registration, nor hold anything that a callback or a
// registration's holder waits for (e.g. a lock).
class CancelScope final {
 public:
  // Keeps a callback of OnCancel() registered until destroyed; see the class
  // comment. A callback may destroy its own Registration, which then does not
  // wait for the callback to return.
  class [[nodiscard]] Registration final {
   public:
    Registration(Registration&& other) noexcept
        : scope_(std::exchange(other.scope_, nullptr)), id_(other.id_) {}
    Registration& operator=(Registration&&) = delete;
    ~Registration();

   private:
    friend class CancelScope;
    Registration(CancelScope* scope, uint64_t id) : scope_(scope), id_(id) {}

    CancelScope* scope_;  // Null if moved from.
    uint64_t id_;
  };

  CancelScope() = default;
  CancelScope(const CancelScope&) = delete;
  CancelScope& operator=(const CancelScope&) = delete;
  // Cancels the scope, then waits until every registration is destroyed and a
  // Cancel() call running the callbacks on another thread returns, logging a
  // warning periodically while it waits (see the class comment for when this
  // would deadlock). CHECK-fails if one of the scope's callbacks destroys it,
  // since Cancel() would then use the destroyed scope.
  ~CancelScope();

  // Cancels the scope. The first call runs each registered callback once, on
  // this thread, without holding the scope's lock. Later calls, including ones
  // from a callback, return at once.
  void Cancel() ABSL_LOCKS_EXCLUDED(mu_);

  // Returns true once Cancel() has been called.
  bool cancelled() const ABSL_LOCKS_EXCLUDED(mu_);

  // Registers `on_cancel` for Cancel() to run, unless the returned Registration
  // is destroyed first. Returns a CANCELLED error, without running `on_cancel`,
  // if the scope is already cancelled, and INVALID_ARGUMENT if `on_cancel` is
  // empty. `on_cancel` and its captures are destroyed without the scope's lock
  // held, so their destructors may use the scope.
  absl::StatusOr<Registration> OnCancel(
      absl_nonnull absl::AnyInvocable<void() &&> on_cancel)
      ABSL_LOCKS_EXCLUDED(mu_);

 private:
  void Deregister(uint64_t id) ABSL_LOCKS_EXCLUDED(mu_);

  // Never held while a callback runs or is destroyed, so callbacks may use the
  // scope, and `mu_` may be taken under any other lock.
  mutable absl::Mutex mu_;
  bool cancelled_ ABSL_GUARDED_BY(mu_) = false;
  uint64_t last_id_ ABSL_GUARDED_BY(mu_) = 0;
  // Callbacks that have neither run nor been deregistered, by registration.
  absl::flat_hash_map<uint64_t, absl::AnyInvocable<void() &&>> callbacks_
      ABSL_GUARDED_BY(mu_);
  // Registrations not yet destroyed.
  int64_t num_live_registrations_ ABSL_GUARDED_BY(mu_) = 0;
  // True while the first Cancel() call runs the callbacks.
  bool cancel_running_ ABSL_GUARDED_BY(mu_) = false;
  // The registration whose callback Cancel() is running (0: none), and the
  // thread that runs the callbacks.
  uint64_t running_id_ ABSL_GUARDED_BY(mu_) = 0;
  std::thread::id running_thread_ ABSL_GUARDED_BY(mu_);
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_CANCEL_SCOPE_H_
