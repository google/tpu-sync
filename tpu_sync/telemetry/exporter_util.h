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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_EXPORTER_UTIL_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_EXPORTER_UTIL_H_

#include <string>

#include "absl/strings/string_view.h"

namespace tpu_raiden::telemetry {

// Common metric namespace prefix prepended to all Prometheus metric family
// names.
inline constexpr absl::string_view kPrometheusMetricPrefix = "tpu_raiden_";

// Reserved Prometheus histogram bucket upper-bound label ("less than or
// equal").
inline constexpr absl::string_view kPrometheusLeLabel = "le";

// Formats a host string and port into a valid endpoint address (e.g.
// "127.0.0.1:8080"). If the host contains ':' and is not already bracketed
// (IPv6 address), it wraps the host in brackets (e.g. "[::1]:8080") per RFC
// 3986.
std::string JoinHostPort(absl::string_view host, int port);

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_EXPORTER_UTIL_H_
