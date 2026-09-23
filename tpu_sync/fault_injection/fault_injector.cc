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

#include "tpu_sync/fault_injection/fault_injector.h"

#include <cerrno>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tpu_raiden {

bool FaultInjector::IsHookActive(std::string_view hook) const noexcept {
  if (!HasActiveInjections()) return false;
  absl::ReaderMutexLock lock(mu_);
  return rules_.contains(hook) || rules_.contains("");
}

FaultInjectionAction FaultInjector::Evaluate(std::string_view hook) {
  if (!HasActiveInjections()) return FaultInjectionAction{};

  absl::MutexLock lock(mu_);
  for (std::string_view key : {hook, std::string_view("")}) {
    auto it = rules_.find(key);
    if (it != rules_.end()) {
      for (const auto& r : it->second) {
        if (r.probability <= 0.0) continue;
        if (r.probability < 1.0 && !absl::Bernoulli(bitgen_, r.probability)) {
          continue;
        }
        rule_hits_[hook]++;
        uint32_t delay_ms = 0;
        if (r.action == FaultInjectionType::kDelay) {
          delay_ms = (r.max_delay_ms > r.min_delay_ms)
                         ? absl::Uniform(absl::IntervalClosed, bitgen_,
                                         r.min_delay_ms, r.max_delay_ms)
                         : r.min_delay_ms;
        }
        return FaultInjectionAction{
            .type = r.action,
            .delay_ms = delay_ms,
        };
      }
    }
    if (key.empty()) break;
  }
  return FaultInjectionAction{};
}

absl::Status FaultInjector::Install(const FaultInjectionRules& rules) {
  absl::flat_hash_map<std::string, std::vector<FaultInjectionRule>> new_rules;
  for (const auto& r : rules) {
    if (r.min_delay_ms > r.max_delay_ms) {
      return absl::InvalidArgumentError(
          "min_delay_ms cannot be greater than max_delay_ms");
    }
    new_rules[r.hook].push_back(r);
  }

  absl::MutexLock lock(mu_);
  rules_ = std::move(new_rules);
  has_active_injections_ = !rules.empty();
  return absl::OkStatus();
}

void FaultInjector::Reset() {
  absl::MutexLock lock(mu_);
  rules_.clear();
  rule_hits_.clear();
  has_active_injections_ = false;
}

uint64_t FaultInjector::GetHitCount(std::string_view hook) const {
  absl::ReaderMutexLock lock(mu_);
  if (hook.empty()) {
    uint64_t total = 0;
    for (const auto& [_, hits] : rule_hits_) {
      total += hits;
    }
    return total;
  }
  auto it = rule_hits_.find(hook);
  return it != rule_hits_.end() ? it->second : 0;
}

absl::flat_hash_map<std::string, uint64_t> FaultInjector::GetHitCounts() const {
  absl::ReaderMutexLock lock(mu_);
  return rule_hits_;
}

void FaultInjector::ExecuteDelay(std::string_view hook) {
  FaultInjectionAction action = Evaluate(hook);
  if (action.type == FaultInjectionType::kDelay && action.delay_ms > 0) {
    absl::SleepFor(absl::Milliseconds(action.delay_ms));
  }
}

void FaultInjector::ExecuteThrow(std::string_view hook) {
  FaultInjectionAction action = Evaluate(hook);
  if (action.type == FaultInjectionType::kDelay && action.delay_ms > 0) {
    absl::SleepFor(absl::Milliseconds(action.delay_ms));
  } else if (action.type == FaultInjectionType::kFail) {
    throw std::runtime_error(absl::StrCat("injected fault at ", hook));
  }
}

absl::Status FaultInjector::ExecuteStatus(std::string_view hook) {
  FaultInjectionAction action = Evaluate(hook);
  if (action.type == FaultInjectionType::kDelay && action.delay_ms > 0) {
    absl::SleepFor(absl::Milliseconds(action.delay_ms));
  } else if (action.type == FaultInjectionType::kFail) {
    return absl::InternalError(absl::StrCat("injected fault at ", hook));
  }
  return absl::OkStatus();
}

bool FaultInjector::ExecuteErrno(std::string_view hook, int err) {
  FaultInjectionAction action = Evaluate(hook);
  if (action.type == FaultInjectionType::kDelay && action.delay_ms > 0) {
    absl::SleepFor(absl::Milliseconds(action.delay_ms));
  } else if (action.type == FaultInjectionType::kFail) {
    errno = err;
    return true;
  }
  return false;
}

FaultInjector& GetFaultInjector() {
  static absl::NoDestructor<FaultInjector> injector;
  return *injector;
}

}  // namespace tpu_raiden
