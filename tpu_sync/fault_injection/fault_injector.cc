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

#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

namespace tpu_raiden {

namespace {

// Returns the hooks where `action` is permitted. A '*' pattern expands only to
// these, e.g. a delay pattern skips fail-only hooks where sleeping is unsafe.
absl::Span<const std::string_view> HooksFor(FaultInjectionType action) {
  switch (action) {
    case FaultInjectionType::kFail:
      return hooks::kFailHooks;
    case FaultInjectionType::kDelay:
      return hooks::kDelayHooks;
    case FaultInjectionType::kNone:
      return {};
  }
  return {};
}

constexpr std::string_view kFaultInjectionFileEnvVar =
    "RAIDEN_FAULT_INJECTION_FILE";

__attribute__((constructor)) void InitFaultInjectorFromEnvAtLoad() {
  if (const char* p = std::getenv(kFaultInjectionFileEnvVar.data());
      p != nullptr && p[0] != '\0') {
    (void)GetFaultInjector();
  }
}

// Plain-text format: one rule per line (`#` comments and blank lines ignored):
//   <hook|pattern*> <fail|delay> <probability> [min_delay_ms [max_delay_ms]]
absl::Status LoadRulesFromText(std::string_view content,
                               FaultInjector& injector) {
  FaultInjectionRules rules;
  for (std::string_view line :
       absl::StrSplit(content, '\n', absl::SkipEmpty())) {
    line = absl::StripAsciiWhitespace(line);
    if (line.empty() || line[0] == '#') continue;
    std::vector<std::string_view> cols =
        absl::StrSplit(line, absl::ByAnyChar(" \t,"), absl::SkipEmpty());
    if (cols.size() < 3 || cols.size() > 5) {
      return absl::InvalidArgumentError(
          absl::StrCat("invalid rule line: ", line));
    }
    FaultInjectionRule r;
    r.hook = std::string(cols[0]);
    if (cols[1] == "delay") {
      r.action = FaultInjectionType::kDelay;
    } else if (cols[1] == "fail") {
      r.action = FaultInjectionType::kFail;
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("unknown action: ", cols[1]));
    }
    if (!absl::SimpleAtod(cols[2], &r.probability)) {
      return absl::InvalidArgumentError(
          absl::StrCat("invalid probability: ", cols[2]));
    }
    if (cols.size() >= 4 && !absl::SimpleAtoi(cols[3], &r.min_delay_ms)) {
      return absl::InvalidArgumentError(
          absl::StrCat("invalid min_delay_ms: ", cols[3]));
    }
    if (cols.size() == 5 && !absl::SimpleAtoi(cols[4], &r.max_delay_ms)) {
      return absl::InvalidArgumentError(
          absl::StrCat("invalid max_delay_ms: ", cols[4]));
    }
    rules.push_back(std::move(r));
  }
  return injector.Install(rules);
}

void WriteStatusFile(std::string_view status_path, bool armed,
                     uint64_t total_hits,
                     const absl::flat_hash_map<std::string, uint64_t>& hits) {
  std::string out_str =
      absl::StrCat("armed=", armed ? 1 : 0, "\ntotal_hits=", total_hits, "\n");
  for (const auto& [hook, count] : hits) {
    absl::StrAppend(&out_str, hook, "=", count, "\n");
  }
  std::string tmp_path = absl::StrCat(status_path, ".tmp");
  if (std::ofstream out(tmp_path); out) {
    out << out_str;
    out.close();
    (void)std::rename(tmp_path.c_str(), std::string(status_path).c_str());
  }
}

}  // namespace

FaultInjector::FaultInjector() {
  if (const char* p = std::getenv(kFaultInjectionFileEnvVar.data());
      p != nullptr && p[0] != '\0') {
    StartFileWatcher(p);
  }
}

FaultInjector::~FaultInjector() { StopFileWatcher(); }

bool FaultInjector::IsHookActive(std::string_view hook) const noexcept {
  if (!HasActiveInjections()) return false;
  absl::ReaderMutexLock lock(mu_);
  return rules_.contains(hook);
}

FaultInjectionAction FaultInjector::Evaluate(std::string_view hook) {
  if (!HasActiveInjections()) return FaultInjectionAction{};

  absl::MutexLock lock(mu_);
  auto it = rules_.find(hook);
  if (it == rules_.end()) return FaultInjectionAction{};
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
  return FaultInjectionAction{};
}

FaultInjectionAction FaultInjector::EvaluateAndSleep(std::string_view hook) {
  const FaultInjectionAction action = Evaluate(hook);
  if (action.type == FaultInjectionType::kDelay && action.delay_ms > 0) {
    absl::SleepFor(absl::Milliseconds(action.delay_ms));
  }
  return action;
}

absl::Status FaultInjector::Install(const FaultInjectionRules& rules) {
  absl::flat_hash_map<std::string, std::vector<FaultInjectionRule>> new_rules;
  for (const auto& r : rules) {
    if (std::isnan(r.probability) || r.probability < 0.0 ||
        r.probability > 1.0) {
      return absl::InvalidArgumentError(
          absl::StrCat("probability must be in [0, 1], got ", r.probability));
    }
    if (r.min_delay_ms > r.max_delay_ms) {
      return absl::InvalidArgumentError(
          "min_delay_ms cannot be greater than max_delay_ms");
    }

    // A trailing '*' is a prefix pattern: install the rule at every hook with
    // that prefix that supports its action.
    if (absl::EndsWith(r.hook, "*")) {
      const std::string_view prefix = absl::StripSuffix(r.hook, "*");
      bool matched = false;
      for (std::string_view hook : HooksFor(r.action)) {
        if (!absl::StartsWith(hook, prefix)) continue;
        new_rules[hook].push_back(r);
        matched = true;
      }
      if (!matched) {
        return absl::InvalidArgumentError(absl::StrCat(
            "pattern '", r.hook, "' matches no hook supporting the action"));
      }
      continue;
    }

    if (!absl::c_linear_search(hooks::kFailHooks, r.hook) &&
        !absl::c_linear_search(hooks::kDelayHooks, r.hook)) {
      return absl::InvalidArgumentError(
          absl::StrCat("unknown hook '", r.hook, "'"));
    }
    if (r.action == FaultInjectionType::kDelay &&
        !absl::c_linear_search(hooks::kDelayHooks, r.hook)) {
      return absl::InvalidArgumentError(
          absl::StrCat("delay is not permitted at hook '", r.hook, "'"));
    }
    if (r.action == FaultInjectionType::kFail &&
        !absl::c_linear_search(hooks::kFailHooks, r.hook)) {
      return absl::InvalidArgumentError(
          absl::StrCat("fail is not permitted at hook '", r.hook, "'"));
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

void FaultInjector::StartFileWatcher(std::string_view file_path,
                                     absl::Duration poll_interval) {
  StopFileWatcher();
  if (file_path.empty()) return;
  {
    absl::MutexLock lock(watcher_mu_);
    watcher_stopping_ = false;
  }
  watcher_thread_ = std::thread(&FaultInjector::WatcherLoop, this,
                                std::string(file_path), poll_interval);
}

void FaultInjector::StopFileWatcher() {
  if (!watcher_thread_.joinable()) return;
  {
    absl::MutexLock lock(watcher_mu_);
    watcher_stopping_ = true;
  }
  watcher_thread_.join();
}

void FaultInjector::WatcherLoop(std::string file_path,
                                absl::Duration poll_interval) {
  const std::string status_path = absl::StrCat(file_path, ".status.", getpid());
  std::string last_content;
  bool status_written = false;
  bool last_armed = false;
  uint64_t last_hits = 0;

  while (true) {
    if (std::ifstream in(file_path); in) {
      std::string content((std::istreambuf_iterator<char>(in)), {});
      if (content != last_content) {
        last_content = std::move(content);
        LoadRulesFromText(last_content, *this).IgnoreError();
      }
    } else if (!last_content.empty()) {
      Install({}).IgnoreError();
      last_content.clear();
    }

    bool armed = HasActiveInjections();
    uint64_t total_hits = GetHitCount();
    if (!status_written || armed != last_armed || total_hits != last_hits) {
      WriteStatusFile(status_path, armed, total_hits, GetHitCounts());
      status_written = true;
      last_armed = armed;
      last_hits = total_hits;
    }

    absl::MutexLock lock(watcher_mu_);
    if (watcher_mu_.AwaitWithTimeout(absl::Condition(&watcher_stopping_),
                                     poll_interval)) {
      break;
    }
  }
}

void FaultInjector::ExecuteDelay(std::string_view hook) {
  EvaluateAndSleep(hook);
}

void FaultInjector::ExecuteThrow(std::string_view hook) {
  FaultInjectionAction action = EvaluateAndSleep(hook);
  if (action.type == FaultInjectionType::kFail) {
    throw std::runtime_error(absl::StrCat("injected fault at ", hook));
  }
}

absl::Status FaultInjector::ExecuteStatus(std::string_view hook,
                                          absl::StatusCode code) {
  ABSL_CHECK_NE(code, absl::StatusCode::kOk)
      << "injected status at " << hook << " must be an error";
  FaultInjectionAction action = EvaluateAndSleep(hook);
  if (action.type == FaultInjectionType::kFail) {
    return absl::Status(code, absl::StrCat("injected fault at ", hook));
  }
  return absl::OkStatus();
}

bool FaultInjector::ExecuteErrno(std::string_view hook, int err) {
  FaultInjectionAction action = EvaluateAndSleep(hook);
  if (action.type == FaultInjectionType::kFail) {
    errno = err;
    return true;
  }
  return false;
}

void FaultInjector::ExecuteSocket(std::string_view hook, int fd) {
  const FaultInjectionAction action = Evaluate(hook);
  if (action.type == FaultInjectionType::kFail) {
    ::shutdown(fd, SHUT_RDWR);
    return;
  }
  if (action.type != FaultInjectionType::kDelay || action.delay_ms == 0) {
    return;
  }

  // Holds the connection open and silent until the delay expires or the
  // connection is torn down. POLLIN is not requested: payload data may already
  // be readable, and only a teardown should end the silence early.
  const absl::Time deadline = absl::Now() + absl::Milliseconds(action.delay_ms);
  struct pollfd pfd = {.fd = fd, .events = POLLRDHUP, .revents = 0};
  int ret;
  do {
    const absl::Duration remaining =
        std::max(deadline - absl::Now(), absl::ZeroDuration());
    ret = ::poll(&pfd, /*nfds=*/1,
                 static_cast<int>(absl::ToInt64Milliseconds(
                     absl::Ceil(remaining, absl::Milliseconds(1)))));
  } while (ret < 0 && errno == EINTR);
}

FaultInjector& GetFaultInjector() {
  static absl::NoDestructor<FaultInjector> injector;
  return *injector;
}

}  // namespace tpu_raiden
