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

#include "tpu_sync/fault_injection/fault_injection_bindings.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>  // IWYU pragma: keep
#include <nanobind/stl/string_view.h>  // IWYU pragma: keep
#include "tpu_sync/fault_injection/fault_injector.h"

namespace tpu_raiden {
namespace {

namespace nb = nanobind;

FaultInjectionRules CompileRules(const nb::list& rule_dicts) {
  FaultInjectionRules compiled;
  compiled.reserve(rule_dicts.size());

  for (nb::handle item : rule_dicts) {
    nb::dict d = nb::cast<nb::dict>(item);
    FaultInjectionRule r;
    if (d.contains("hook")) {
      r.hook = nb::cast<std::string>(d["hook"]);
    }
    if (d.contains("action")) {
      auto act = nb::cast<std::string_view>(d["action"]);
      if (act == "delay") {
        r.action = FaultInjectionType::kDelay;
      } else if (act == "fail") {
        r.action = FaultInjectionType::kFail;
      } else {
        throw nb::value_error(absl::StrCat("unknown action: ", act).c_str());
      }
    }
    if (d.contains("probability")) {
      r.probability = nb::cast<double>(d["probability"]);
    }
    if (d.contains("min_delay_ms")) {
      r.min_delay_ms = nb::cast<uint32_t>(d["min_delay_ms"]);
    }
    if (d.contains("max_delay_ms")) {
      r.max_delay_ms = nb::cast<uint32_t>(d["max_delay_ms"]);
    }
    compiled.push_back(std::move(r));
  }
  return compiled;
}

}  // namespace

void BindFaultInjection(nb::module_& m) {
  m.def(
      "inject_faults",
      [](const nb::list& rule_dicts) {
        absl::Status res =
            GetFaultInjector().Install(CompileRules(rule_dicts));
        if (!res.ok()) {
          throw nb::value_error(std::string(res.message()).c_str());
        }
      },
      nb::arg("rules"), "Installs a list of fault injection rules.");

  m.def(
      "reset_faults", []() { GetFaultInjector().Reset(); },
      "Resets all active fault injection rules and hit counters.");

  m.def(
      "get_fault_status",
      []() -> nb::dict {
        nb::dict d;
        for (const auto& [hook, hits] : GetFaultInjector().GetHitCounts()) {
          d[hook.c_str()] = hits;
        }
        return d;
      },
      "Returns a dictionary mapping hook names to hit counts.");

  m.def(
      "get_hit_count",
      [](std::string_view hook) -> uint64_t {
        return GetFaultInjector().GetHitCount(hook);
      },
      nb::arg("hook") = "",
      "Returns the hit count for a specific hook, or total hits if empty.");

  m.def(
      "is_hook_active",
      [](std::string_view hook) -> bool {
        return GetFaultInjector().IsHookActive(hook);
      },
      nb::arg("hook"),
      "Returns True if any active rule matches the specified hook.");

  m.def(
      "has_active_injections",
      []() -> bool { return FaultInjector::HasActiveInjections(); },
      "Returns True if any fault injection rules are currently active.");
}

}  // namespace tpu_raiden
