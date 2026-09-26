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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FAULT_INJECTION_FAULT_INJECTOR_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FAULT_INJECTION_FAULT_INJECTOR_H_

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tpu_sync/fault_injection/hooks.h"  // IWYU pragma: export

namespace tpu_raiden {

enum class FaultInjectionType {
  kNone = 0,
  kDelay,
  kFail,
};

struct FaultInjectionRule {
  std::string hook;
  FaultInjectionType action = FaultInjectionType::kFail;
  double probability = 0.0;
  uint32_t min_delay_ms = 0;
  uint32_t max_delay_ms = 0;
};

struct FaultInjectionAction {
  FaultInjectionType type = FaultInjectionType::kNone;
  uint32_t delay_ms = 0;
};

using FaultInjectionRules = std::vector<FaultInjectionRule>;

class FaultInjector final {
 public:
  FaultInjector() = default;
  FaultInjector(const FaultInjector&) = delete;
  FaultInjector& operator=(const FaultInjector&) = delete;
  ~FaultInjector() = default;

  // Returns true if any fault injection rules are currently
  // active.
  static bool HasActiveInjections() noexcept {
    return has_active_injections_.load(std::memory_order_relaxed);
  }

  // Returns true if any active fault injection rule matches the specified hook.
  bool IsHookActive(std::string_view hook) const noexcept
      ABSL_LOCKS_EXCLUDED(mu_);

  // Evaluates active rules for the hook and returns the action.
  FaultInjectionAction Evaluate(std::string_view hook) ABSL_LOCKS_EXCLUDED(mu_);

  // Installs fault injection rules. Returns an error if any rule is invalid.
  // A hook ending in '*' is a prefix pattern ("*" matches all hooks,
  // "grpc_control_plane.*" matches hooks with that prefix); it installs the
  // rule at every matching hook that supports the rule's action.
  absl::Status Install(const FaultInjectionRules& rules)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Clears all active fault injection rules and resets hit counters.
  void Reset() ABSL_LOCKS_EXCLUDED(mu_);

  // Returns the hit count for a specific hook, or the total count if hook is
  // empty.
  uint64_t GetHitCount(std::string_view hook = "") const
      ABSL_LOCKS_EXCLUDED(mu_);

  // Returns a snapshot mapping of hook names to their hit counts.
  absl::flat_hash_map<std::string, uint64_t> GetHitCounts() const
      ABSL_LOCKS_EXCLUDED(mu_);

  // Slow-path execution helpers invoked only when HasActiveInjections() is
  // true. A delay always runs to completion.
  void ExecuteDelay(std::string_view hook) ABSL_LOCKS_EXCLUDED(mu_);
  void ExecuteThrow(std::string_view hook) ABSL_LOCKS_EXCLUDED(mu_);
  absl::Status ExecuteStatus(std::string_view hook, absl::StatusCode code)
      ABSL_LOCKS_EXCLUDED(mu_);
  bool ExecuteErrno(std::string_view hook, int err) ABSL_LOCKS_EXCLUDED(mu_);

  // Socket-aware execution, called right before the real I/O on `fd`: the
  // file descriptor of the connected TCP socket the caller is about to read
  // or write (a pooled/connected fd on the sender, the accept()ed fd on the
  // receiver). It identifies one connection, so the fault affects only it:
  //  - kFail: shuts `fd` down (SHUT_RDWR) and returns. The caller's real I/O
  //    then fails with the real error (EPIPE/EOF), and the peer sees FIN.
  //  - kDelay: holds the connection open and silent until the delay expires,
  //    the peer closes, or `fd` is shut down locally. Readable payload data
  //    does not end the delay.
  void ExecuteSocket(std::string_view hook, int fd) ABSL_LOCKS_EXCLUDED(mu_);

 private:
  static inline std::atomic<bool> has_active_injections_{false};

  // Evaluates the hook and sleeps for the delay if a delay action is chosen.
  FaultInjectionAction EvaluateAndSleep(std::string_view hook)
      ABSL_LOCKS_EXCLUDED(mu_);

  mutable absl::Mutex mu_;
  absl::BitGen bitgen_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, std::vector<FaultInjectionRule>> rules_
      ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, uint64_t> rule_hits_ ABSL_GUARDED_BY(mu_);
};

// Returns the process-wide FaultInjector singleton instance.
FaultInjector& GetFaultInjector();

// Evaluates the hook and sleeps for the configured delay if matched.
inline void FaultInjectDelay(std::string_view hook) {
  if (ABSL_PREDICT_FALSE(FaultInjector::HasActiveInjections())) {
    GetFaultInjector().ExecuteDelay(hook);
  }
}

// Evaluates the hook and throws std::runtime_error if a failure is triggered.
inline void FaultInjectThrow(std::string_view hook) {
  if (ABSL_PREDICT_FALSE(FaultInjector::HasActiveInjections())) {
    GetFaultInjector().ExecuteThrow(hook);
  }
}

// Evaluates the hook and returns an error with status `code` if a failure is
// triggered, OK otherwise. `code` must not be kOk, so callers can propagate
// the result directly (e.g. with ABSL_RETURN_IF_ERROR).
inline absl::Status FaultInjectStatus(
    std::string_view hook,
    absl::StatusCode code = absl::StatusCode::kInternal) {
  if (ABSL_PREDICT_FALSE(FaultInjector::HasActiveInjections())) {
    return GetFaultInjector().ExecuteStatus(hook, code);
  }
  return absl::OkStatus();
}

// Evaluates the hook and sets errno to `err` if a failure is triggered.
inline bool FaultInjectErrno(std::string_view hook, int err = EIO) {
  if (ABSL_PREDICT_FALSE(FaultInjector::HasActiveInjections())) {
    return GetFaultInjector().ExecuteErrno(hook, err);
  }
  return false;
}

// Call right before the real I/O on `fd`, the file descriptor of the
// connected TCP socket the caller is about to read or write. See
// FaultInjector::ExecuteSocket for the fail and delay semantics.
inline void FaultInjectSocket(std::string_view hook, int fd) {
  if (ABSL_PREDICT_FALSE(FaultInjector::HasActiveInjections())) {
    GetFaultInjector().ExecuteSocket(hook, fd);
  }
}

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FAULT_INJECTION_FAULT_INJECTOR_H_
