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

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
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

// Default stack buffer capacity (in bytes) for formatted Prometheus labels,
// sized to accommodate standard multi-label metric descriptors without heap
// allocation.
inline constexpr std::size_t kDefaultPrometheusStackBufferSize = 256;

// Zero-allocation RAII stack buffer view for hot-path metric lookup.
// Formats labels directly into an internal 256-byte stack buffer and provides
// an absl::string_view for heterogeneous hash map lookup. Falls back to dynamic
// heap allocation only if the formatted string exceeds 256 bytes.
class PrometheusLabelView {
 public:
  explicit PrometheusLabelView(LabelSpan labels);

  // Non-copyable and non-movable: holds internal pointers to stack_buf_ /
  // heap_fallback_.
  PrometheusLabelView(const PrometheusLabelView&) = delete;
  PrometheusLabelView& operator=(const PrometheusLabelView&) = delete;
  PrometheusLabelView(PrometheusLabelView&&) = delete;
  PrometheusLabelView& operator=(PrometheusLabelView&&) = delete;

  absl::string_view view() const ABSL_ATTRIBUTE_LIFETIME_BOUND { return view_; }

  // Returns an owned string, moving the heap fallback if allocated or
  // constructing from the stack buffer.
  std::string ToOwned() && {
    absl::string_view current_view = view_;
    view_ = "";
    if (!heap_fallback_.empty()) {
      return std::move(heap_fallback_);
    }
    return std::string(current_view);
  }

 private:
  char stack_buf_[kDefaultPrometheusStackBufferSize];
  std::string heap_fallback_;
  absl::string_view view_ = {};
};

// ============================================================================
// Zero-Allocation Label Resolution for Fixed Schemas (Fixed-Arity)
// ============================================================================

inline constexpr absl::string_view kDefaultLabelValue = "unknown";

// Resolves target label keys from LabelSpan in a single pass with zero heap
// allocations for fixed-arity schemas known at compile time. Returns an array
// of resolved values matching target_keys order. Missing keys default to
// `default_val`. Preserves first-match semantics for duplicate keys, and
// terminates early once all N keys are found.
template <std::size_t N>
constexpr std::array<absl::string_view, N> ResolveLabels(
    LabelSpan labels, const std::array<absl::string_view, N>& target_keys,
    absl::string_view default_val = kDefaultLabelValue) {
  if constexpr (N == 0) {
    return {};
  }
  std::array<absl::string_view, N> resolved;
  resolved.fill(default_val);
  std::array<bool, N> found{};
  std::size_t found_count = 0;

  for (const MetricLabel& label : labels) {
    for (std::size_t i = 0; i < N; ++i) {
      if (!found[i] && label.key == target_keys[i]) {
        resolved[i] = label.value;
        found[i] = true;
        if (++found_count == N) {
          return resolved;
        }
        break;
      }
    }
  }
  return resolved;
}

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_LABEL_UTIL_H_
