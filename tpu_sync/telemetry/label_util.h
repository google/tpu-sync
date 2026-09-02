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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_LABEL_UTIL_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_LABEL_UTIL_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::telemetry {

// ============================================================================
// Shared Memory Compact Semicolon Formatting (k1=v1;k2=v2)
// Escapes: '\', '=', ';'
// ============================================================================

// Formats labels into semicolon-delimited format directly into output_buffer.
// Returns an absl::string_view pointing into output_buffer on success, or
// std::nullopt if output_buffer has insufficient capacity. Guarantees zero heap
// allocations for pre-sorted labels or unsorted label sets up to 8 labels.
// Does not append a trailing null terminator; callers targeting fixed-size
// buffers (such as ShmTocEntry::encoded_labels) should ensure the buffer has
// room for a null terminator if null termination is required.
std::optional<absl::string_view> FormatShmLabelsToBuffer(
    LabelSpan labels, absl::Span<char> output_buffer);

// Parses a compact semicolon-encoded string into key-value label pairs.
// Handles backslash unescaping for '\=', '\;', and '\\'. Terminates parsing
// on the first null byte ('\0') to support fixed-size null-padded buffers.
// Discards malformed tokens (missing '=' or empty keys) on a best-effort
// basis without logging side effects.
std::vector<std::pair<std::string, std::string>> ParseShmLabels(
    absl::string_view encoded_labels);

// ============================================================================
// Prometheus Canonical Formatting ({k1="v1",k2="v2"})
// Escapes: '\', '"', '\n'
// ============================================================================

// Formats labels into Prometheus canonical format directly into output_buffer.
// Returns an absl::string_view pointing into output_buffer on success, or
// std::nullopt if output_buffer has insufficient capacity. Performs raw,
// zero-overhead serialization without runtime key syntax validation or
// deduplication; callers are responsible for label key syntax and uniqueness.
// Guarantees zero heap allocations for pre-sorted labels or unsorted label sets
// up to 8 labels.
std::optional<absl::string_view> FormatPrometheusLabelsToBuffer(
    LabelSpan labels, absl::Span<char> output_buffer);

// Owning std::string wrapper for non-critical paths, testing, and series
// identification. Performs raw serialization without runtime key syntax
// validation.
std::string FormatPrometheusLabels(LabelSpan labels);

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_LABEL_UTIL_H_
