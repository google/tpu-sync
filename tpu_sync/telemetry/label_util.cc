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

#include "tpu_sync/telemetry/label_util.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::telemetry {
namespace {

constexpr size_t kDefaultInlinedLabelCapacity = 8;
constexpr size_t kDefaultPrometheusStackBufferSize = 256;

// Lightweight buffer writer that bounds-checks appends into a char span.
class BufferWriter {
 public:
  explicit BufferWriter(absl::Span<char> output_buffer)
      : output_buffer_(output_buffer) {}

  bool Append(char c) {
    if (offset_ >= output_buffer_.size()) return false;
    output_buffer_[offset_++] = c;
    return true;
  }

  bool Append(absl::string_view s) {
    if (s.size() > output_buffer_.size() - offset_) return false;
    if (!s.empty()) {
      std::copy(s.begin(), s.end(), output_buffer_.data() + offset_);
      offset_ += s.size();
    }
    return true;
  }

  // Escapes '\', '=', and ';' for compact SHM IPC formatting.
  bool AppendShmEscaped(absl::string_view s) {
    for (char c : s) {
      if (c == '\\' || c == '=' || c == ';') {
        if (!Append('\\')) return false;
      }
      if (!Append(c)) return false;
    }
    return true;
  }

  // Escapes '\', '"', and '\n' for canonical Prometheus label values.
  bool AppendPrometheusEscaped(absl::string_view s) {
    for (char c : s) {
      switch (c) {
        case '\\':
        case '"':
          if (!Append('\\') || !Append(c)) return false;
          break;
        case '\n':
          if (!Append('\\') || !Append('n')) return false;
          break;
        default:
          if (!Append(c)) return false;
          break;
      }
    }
    return true;
  }

  absl::string_view view() const {
    return absl::string_view(output_buffer_.data(), offset_);
  }

 private:
  absl::Span<char> output_buffer_;
  size_t offset_ = 0;
};

// Common stack-allocated sorting helper for multi-label sets.
void SortLabels(
    LabelSpan labels,
    absl::InlinedVector<MetricLabel, kDefaultInlinedLabelCapacity>& out) {
  out.assign(labels.begin(), labels.end());
  absl::c_sort(out);
}

// Returns the byte length of Prometheus label value after escaping ('\', '"',
// '\n').
size_t EscapedPrometheusValueLength(absl::string_view s) {
  size_t len = s.size();
  for (char c : s) {
    if (c == '\\' || c == '"' || c == '\n') ++len;
  }
  return len;
}

// Computes exact serialized byte length for Prometheus format
// ({k1="v1",k2="v2"}).
size_t ComputePrometheusLabelsSize(LabelSpan labels) {
  if (labels.empty()) return 0;
  size_t total = 2;  // '{' and '}'
  for (size_t i = 0; i < labels.size(); ++i) {
    if (i > 0) ++total;  // ','
    total += labels[i].key.size() + 2 /* '="' */ +
             EscapedPrometheusValueLength(labels[i].value) + 1 /* '"' */;
  }
  return total;
}

// Generic 2-stage stack-to-heap allocation fallback orchestrator.
template <size_t StackBufferSize, typename SizeFn, typename BufferFormatFn>
std::string FormatWithStackBufferFallback(LabelSpan labels, SizeFn size_fn,
                                          BufferFormatFn format_fn) {
  if (labels.empty()) return "";

  char stack_buffer[StackBufferSize];
  if (std::optional<absl::string_view> formatted =
          format_fn(labels, absl::MakeSpan(stack_buffer));
      formatted.has_value()) {
    return std::string(*formatted);
  }

  // Exact-size dynamic fallback for label sets exceeding stack buffer.
  std::string result(size_fn(labels), '\0');
  std::optional<absl::string_view> formatted =
      format_fn(labels, absl::MakeSpan(result));
  if (!formatted.has_value()) {
    return "";
  }
  result.resize(formatted->size());
  return result;
}

}  // namespace

std::optional<absl::string_view> FormatShmLabelsToBuffer(
    LabelSpan labels, absl::Span<char> output_buffer) {
  if (labels.empty()) {
    return absl::string_view(output_buffer.data(), 0);
  }

  BufferWriter writer(output_buffer);

  absl::InlinedVector<MetricLabel, kDefaultInlinedLabelCapacity> sorted;
  if (!absl::c_is_sorted(labels)) {
    SortLabels(labels, sorted);
    labels = sorted;
  }

  for (size_t i = 0; i < labels.size(); ++i) {
    if (i > 0 && !writer.Append(';')) return std::nullopt;
    const MetricLabel& label = labels[i];
    if (!writer.AppendShmEscaped(label.key) || !writer.Append('=') ||
        !writer.AppendShmEscaped(label.value)) {
      return std::nullopt;
    }
  }
  return writer.view();
}

std::vector<std::pair<std::string, std::string>> ParseShmLabels(
    absl::string_view encoded_labels) {
  std::vector<std::pair<std::string, std::string>> labels;

  // Terminate parsing on first null byte to safely handle fixed-size buffers
  // with null padding (e.g. ShmTocEntry::encoded_labels).
  const size_t null_pos = encoded_labels.find('\0');
  if (null_pos != absl::string_view::npos) {
    encoded_labels = encoded_labels.substr(0, null_pos);
  }
  if (encoded_labels.empty()) return labels;

  const size_t num_semicolons = absl::c_count(encoded_labels, ';');
  labels.reserve(num_semicolons + 1);

  std::string current_key;
  std::string current_value;
  bool parsing_value = false;
  bool in_escape = false;

  auto commit_pair = [&]() {
    if (!current_key.empty() && parsing_value) {
      labels.emplace_back(std::move(current_key), std::move(current_value));
    }
    current_key.clear();
    current_value.clear();
    parsing_value = false;
    in_escape = false;
  };

  for (char c : encoded_labels) {
    std::string& target = parsing_value ? current_value : current_key;
    if (in_escape) {
      target.push_back(c);
      in_escape = false;
    } else if (c == '\\') {
      in_escape = true;
    } else if (c == '=' && !parsing_value) {
      parsing_value = true;
    } else if (c == ';') {
      commit_pair();
    } else {
      target.push_back(c);
    }
  }
  if (in_escape) {
    std::string& target = parsing_value ? current_value : current_key;
    target.push_back('\\');
  }
  commit_pair();
  return labels;
}

std::optional<absl::string_view> FormatPrometheusLabelsToBuffer(
    LabelSpan labels, absl::Span<char> output_buffer) {
  if (labels.empty()) {
    return absl::string_view(output_buffer.data(), 0);
  }

  BufferWriter writer(output_buffer);

  absl::InlinedVector<MetricLabel, kDefaultInlinedLabelCapacity> sorted;
  if (!absl::c_is_sorted(labels)) {
    SortLabels(labels, sorted);
    labels = sorted;
  }

  if (!writer.Append('{')) return std::nullopt;
  for (size_t i = 0; i < labels.size(); ++i) {
    if (i > 0 && !writer.Append(',')) return std::nullopt;
    const MetricLabel& label = labels[i];
    if (!writer.Append(label.key) || !writer.Append("=\"") ||
        !writer.AppendPrometheusEscaped(label.value) || !writer.Append('"')) {
      return std::nullopt;
    }
  }
  if (!writer.Append('}')) return std::nullopt;
  return writer.view();
}

std::string FormatPrometheusLabels(LabelSpan labels) {
  return FormatWithStackBufferFallback<kDefaultPrometheusStackBufferSize>(
      labels, ComputePrometheusLabelsSize, FormatPrometheusLabelsToBuffer);
}

}  // namespace tpu_raiden::telemetry
