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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_TRACE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_TRACE_H_

#include <cstdint>
#include <cstdio>   // IWYU pragma: keep
#include <cstdlib>  // IWYU pragma: keep
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/strings/str_cat.h"  // IWYU pragma: keep
#include "absl/strings/string_view.h"
#include "perfetto/tracing/event_context.h"
#include "perfetto/tracing/string_helpers.h"
#include "perfetto/tracing/tracing.h"  // IWYU pragma: keep
#include "perfetto/tracing/track_event.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

// Defines Perfetto track event categories in the tpu_raiden namespace.
PERFETTO_DEFINE_CATEGORIES_IN_NAMESPACE(
    tpu_raiden, ::perfetto::Category("tpu_raiden")
                    .SetDescription("Events from TPU Raiden"));

namespace tpu_raiden {

using ::tsl::profiler::TraceMe;
using ::tsl::profiler::TraceMeEncode;
using ::tsl::profiler::TraceMeLevel;

// Supported tracing backends.
enum class TraceBackend : uint8_t {
  kNone = 0,
  kTraceMe = 1 << 0,
  kPerfetto = 1 << 1,
  kBoth = kTraceMe | kPerfetto,
};

constexpr TraceBackend operator|(TraceBackend a, TraceBackend b) {
  return static_cast<TraceBackend>(static_cast<uint8_t>(a) |
                                   static_cast<uint8_t>(b));
}

constexpr TraceBackend operator&(TraceBackend a, TraceBackend b) {
  return static_cast<TraceBackend>(static_cast<uint8_t>(a) &
                                   static_cast<uint8_t>(b));
}

constexpr bool HasBackend(TraceBackend config, TraceBackend target) {
  return (static_cast<uint8_t>(config) & static_cast<uint8_t>(target)) != 0;
}

// Ensures Perfetto client tracing is initialized. Thread-safe.
void EnsurePerfettoInitialized();

// Returns the active tracing backend, determined from environment variables
// (TPU_RAIDEN_TRACE_BACKEND / RAIDEN_TRACE_BACKEND, or TPU_RAIDEN_USE_PERFETTO)
// or an override set via SetTraceBackendForTesting. Thread-safe.
TraceBackend GetTraceBackend();

// Overrides the active tracing backend for unit tests. Pass std::nullopt to
// clear the override. Thread-safe.
void SetTraceBackendForTesting(std::optional<TraceBackend> backend);

// Starts an in-process Perfetto tracing session and writes the collected
// trace data to output_file_path when StopPerfettoTraceToFile() is called.
// Returns true on success, false if a trace session is already running or
// if output_file_path cannot be created. Thread-safe.
bool StartPerfettoTraceToFile(absl::string_view output_file_path);

// Stops the file-based Perfetto tracing session started by
// StartPerfettoTraceToFile(), flushes all buffered events, and writes the
// trace data to the configured file path. Returns true on success. Thread-safe.
bool StopPerfettoTraceToFile();

// Checks if verbose trace logging to stderr is enabled via either
// TPU_RAIDEN_TRACE or RAIDEN_TRACE environment variable.
// Thread-safe.
inline bool IsTraceLoggingEnabled() {
  const char* tpu_raiden_trace = std::getenv("TPU_RAIDEN_TRACE");
  if (tpu_raiden_trace != nullptr && tpu_raiden_trace[0] != '\0' &&
      absl::string_view(tpu_raiden_trace) != "0") {
    return true;
  }
  const char* raiden_trace = std::getenv("RAIDEN_TRACE");
  if (raiden_trace != nullptr && raiden_trace[0] != '\0' &&
      absl::string_view(raiden_trace) != "0") {
    return true;
  }
  return false;
}

inline uint64_t SubTransferStageTrackId(uint64_t uuid, uint32_t stage_tag) {
  uint64_t z =
      uuid ^ (0x9E3779B97F4A7C15ULL * (static_cast<uint64_t>(stage_tag) + 1));
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return (z ^ (z >> 31)) | 1ULL;
}

// RAII scoped trace that emits trace events to TraceMe and/or Perfetto.
class ScopedTrace {
 public:
  template <
      typename NameType,
      std::enable_if_t<!std::is_same_v<std::decay_t<NameType>, TraceBackend>,
                       bool> = true>
  explicit ScopedTrace(NameType&& name, int level = 1,
                       std::optional<uint64_t> track_id = std::nullopt)
      : ScopedTrace(GetTraceBackend(), std::forward<NameType>(name), level,
                    track_id) {}

  template <typename NameType>
  ScopedTrace(TraceBackend backend, NameType&& name, int level = 1,
              std::optional<uint64_t> track_id = std::nullopt)
      : track_id_(track_id) {
    Init(backend, std::forward<NameType>(name), level);
  }

  ~ScopedTrace() { EndPerfettoSlice(); }

  ScopedTrace(ScopedTrace&& other) noexcept
      : traceme_(std::move(other.traceme_)),
        perfetto_active_(other.perfetto_active_),
        track_id_(other.track_id_),
        perfetto_metadata_(std::move(other.perfetto_metadata_)) {
    other.perfetto_active_ = false;
    other.track_id_.reset();
  }

  ScopedTrace& operator=(ScopedTrace&& other) noexcept {
    if (this != &other) {
      EndPerfettoSlice();
      traceme_ = std::move(other.traceme_);
      perfetto_active_ = other.perfetto_active_;
      track_id_ = other.track_id_;
      perfetto_metadata_ = std::move(other.perfetto_metadata_);
      other.perfetto_active_ = false;
      other.track_id_.reset();
    }
    return *this;
  }

  ScopedTrace(const ScopedTrace&) = delete;
  ScopedTrace& operator=(const ScopedTrace&) = delete;

  // Appends metadata to the trace scope.
  template <typename MetadataGenerator>
  void AppendMetadata(MetadataGenerator&& metadata_generator) {
    if (traceme_.has_value()) {
      traceme_->AppendMetadata(
          std::forward<MetadataGenerator>(metadata_generator));
    }
    if (perfetto_active_) {
      PERFETTO_USE_CATEGORIES_FROM_NAMESPACE_SCOPED(tpu_raiden);
      std::string val;
      if constexpr (std::is_invocable_v<MetadataGenerator>) {
        val = std::string(metadata_generator());
      } else {
        val = std::string(metadata_generator);
      }
      absl::string_view sv(val);
      if (!perfetto_metadata_.empty()) {
        perfetto_metadata_.push_back(',');
      }
      perfetto_metadata_.append(sv.data(), sv.size());
      TRACE_EVENT_INSTANT(
          "tpu_raiden", "metadata", [&](::perfetto::EventContext ctx) {
            auto* annotation = ctx.event()->add_debug_annotations();
            annotation->set_name("metadata");
            annotation->set_string_value(val);
          });
    }
  }

 private:
  template <typename NameType>
  void Init(TraceBackend backend, NameType&& name, int level) {
    const bool use_traceme = HasBackend(backend, TraceBackend::kTraceMe);
    const bool use_perfetto = HasBackend(backend, TraceBackend::kPerfetto);

    if (use_traceme) {
      traceme_.emplace(std::forward<NameType>(name), level);
    }

    if (use_perfetto) {
      EnsurePerfettoInitialized();
      PERFETTO_USE_CATEGORIES_FROM_NAMESPACE_SCOPED(tpu_raiden);
      if (TRACE_EVENT_CATEGORY_ENABLED("tpu_raiden")) {
        EmitPerfettoBegin(std::forward<NameType>(name), track_id_);
        perfetto_active_ = true;
      }
    }
  }

  template <typename NameType>
  static void EmitPerfettoBegin(
      NameType&& name, std::optional<uint64_t> track_id = std::nullopt) {
    PERFETTO_USE_CATEGORIES_FROM_NAMESPACE_SCOPED(tpu_raiden);
#define RAIDEN_EMIT_BEGIN_(event_name)                 \
  do {                                                 \
    if (track_id.has_value()) {                        \
      TRACE_EVENT_BEGIN("tpu_raiden", event_name,      \
                        ::perfetto::Track(*track_id)); \
    } else {                                           \
      TRACE_EVENT_BEGIN("tpu_raiden", event_name);     \
    }                                                  \
  } while (0)
    if constexpr (std::is_invocable_v<NameType>) {
      auto dynamic_name = name();
      RAIDEN_EMIT_BEGIN_(
          ::perfetto::DynamicString(dynamic_name.data(), dynamic_name.size()));
    } else if constexpr (std::is_same_v<std::decay_t<NameType>,
                                        ::perfetto::DynamicString> ||
                         std::is_same_v<std::decay_t<NameType>,
                                        ::perfetto::StaticString> ||
                         std::is_array_v<std::remove_reference_t<NameType>>) {
      RAIDEN_EMIT_BEGIN_(name);
    } else if constexpr (std::is_convertible_v<NameType, absl::string_view>) {
      absl::string_view sv = name;
      RAIDEN_EMIT_BEGIN_(::perfetto::DynamicString(sv.data(), sv.size()));
    } else {
      RAIDEN_EMIT_BEGIN_(name);
    }
#undef RAIDEN_EMIT_BEGIN_
  }

  void EndPerfettoSlice() {
    if (!perfetto_active_) return;
    PERFETTO_USE_CATEGORIES_FROM_NAMESPACE_SCOPED(tpu_raiden);
    const ::perfetto::DynamicString meta(perfetto_metadata_.data(),
                                         perfetto_metadata_.size());
    if (track_id_.has_value()) {
      if (!perfetto_metadata_.empty()) {
        TRACE_EVENT_END("tpu_raiden", ::perfetto::Track(*track_id_), "metadata",
                        meta);
      } else {
        TRACE_EVENT_END("tpu_raiden", ::perfetto::Track(*track_id_));
      }
    } else if (!perfetto_metadata_.empty()) {
      TRACE_EVENT_END("tpu_raiden", "metadata", meta);
    } else {
      TRACE_EVENT_END("tpu_raiden");
    }
    perfetto_active_ = false;
  }

  std::optional<::tsl::profiler::TraceMe> traceme_;
  bool perfetto_active_ = false;
  std::optional<uint64_t> track_id_;
  std::string perfetto_metadata_;
};

}  // namespace tpu_raiden

// Concatenation helpers for generating unique variable names per scope line.
#define RAIDEN_TRACE_CONCAT_INNER_(x, y) x##y
#define RAIDEN_TRACE_CONCAT_(x, y) RAIDEN_TRACE_CONCAT_INNER_(x, y)
#define RAIDEN_TRACE_UNIQUE_NAME_(base) RAIDEN_TRACE_CONCAT_(base, __LINE__)

// Defines an RAII trace scope using the globally configured trace backend.
// Negligible overhead when profiling is inactive.
//
// Usage:
//   RAIDEN_TRACE("MyActivity");
//   RAIDEN_TRACE("MyActivity", /*level=*/2);
//   RAIDEN_TRACE([&]() { return absl::StrCat("DynamicOp_", id); });
#define RAIDEN_TRACE(name, ...)                              \
  const ::tpu_raiden::ScopedTrace RAIDEN_TRACE_UNIQUE_NAME_( \
      _raiden_scoped_trace)(name, ##__VA_ARGS__)

// Defines a trace scope for a code block. Alias for RAIDEN_TRACE.
//
// Usage:
//   RAIDEN_TRACE_SCOPE("MyScope");
#define RAIDEN_TRACE_SCOPE(name, ...) RAIDEN_TRACE(name, ##__VA_ARGS__)

// Defines a scoped trace activity with dynamic metadata.
// The metadata callback is only executed if tracing is active.
//
// Usage:
//   RAIDEN_TRACE_FN("MyActivity", [&]() {
//     return ::tpu_raiden::TraceMeEncode({{"batch_size", 32}, {"step", 1}});
//   });
#define RAIDEN_TRACE_FN(name, metadata_fn, ...)                              \
  ::tpu_raiden::ScopedTrace RAIDEN_TRACE_UNIQUE_NAME_(_raiden_scoped_trace)( \
      name, ##__VA_ARGS__);                                                  \
  RAIDEN_TRACE_UNIQUE_NAME_(_raiden_scoped_trace).AppendMetadata(metadata_fn)

// Defines a scoped trace using Perfetto explicitly regardless of default
// backend.
//
// Usage:
//   RAIDEN_PERFETTO_TRACE("MyActivity");
#define RAIDEN_PERFETTO_TRACE(name, ...)                                   \
  const ::tpu_raiden::ScopedTrace RAIDEN_TRACE_UNIQUE_NAME_(               \
      _raiden_perfetto_trace)(::tpu_raiden::TraceBackend::kPerfetto, name, \
                              ##__VA_ARGS__)

// Defines a scoped trace using TraceMe explicitly regardless of default
// backend.
//
// Usage:
//   RAIDEN_TRACEME_TRACE("MyActivity");
#define RAIDEN_TRACEME_TRACE(name, ...)                                  \
  const ::tpu_raiden::ScopedTrace RAIDEN_TRACE_UNIQUE_NAME_(             \
      _raiden_traceme_trace)(::tpu_raiden::TraceBackend::kTraceMe, name, \
                             ##__VA_ARGS__)

// Logs a trace message directly to stderr when TPU_RAIDEN_TRACE or RAIDEN_TRACE
// is set in the environment. Thread-safe.
#define RAIDEN_TRACE_LOG(...)                          \
  do {                                                 \
    if (::tpu_raiden::IsTraceLoggingEnabled()) {       \
      std::fprintf(stderr, "[RAIDEN_TRACE] %s\n",      \
                   absl::StrCat(__VA_ARGS__).c_str()); \
    }                                                  \
  } while (0)

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_TRACE_H_
