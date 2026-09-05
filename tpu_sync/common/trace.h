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

// RAII scoped trace that emits trace events to TraceMe and/or Perfetto.
class ScopedTrace {
 public:
  template <
      typename NameType,
      std::enable_if_t<!std::is_same_v<std::decay_t<NameType>, TraceBackend>,
                       bool> = true>
  explicit ScopedTrace(NameType&& name, int level = 1)
      : ScopedTrace(GetTraceBackend(), std::forward<NameType>(name), level) {}

  template <typename NameType>
  ScopedTrace(TraceBackend backend, NameType&& name, int level = 1) {
    Init(backend, std::forward<NameType>(name), level);
  }

  ~ScopedTrace() {
    if (perfetto_active_) {
      PERFETTO_USE_CATEGORIES_FROM_NAMESPACE_SCOPED(tpu_raiden);
      TRACE_EVENT_END("tpu_raiden");
    }
  }

  ScopedTrace(ScopedTrace&& other) noexcept
      : traceme_(std::move(other.traceme_)),
        perfetto_active_(other.perfetto_active_) {
    other.perfetto_active_ = false;
  }

  ScopedTrace& operator=(ScopedTrace&& other) noexcept {
    if (this != &other) {
      if (perfetto_active_) {
        PERFETTO_USE_CATEGORIES_FROM_NAMESPACE_SCOPED(tpu_raiden);
        TRACE_EVENT_END("tpu_raiden");
      }
      traceme_ = std::move(other.traceme_);
      perfetto_active_ = other.perfetto_active_;
      other.perfetto_active_ = false;
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
      TRACE_EVENT_INSTANT(
          "tpu_raiden", "metadata", [&](::perfetto::EventContext ctx) {
            auto* annotation = ctx.event()->add_debug_annotations();
            annotation->set_name("metadata");
            if constexpr (std::is_invocable_v<MetadataGenerator>) {
              std::string val = std::string(metadata_generator());
              annotation->set_string_value(val);
            }
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
        EmitPerfettoBegin(std::forward<NameType>(name));
        perfetto_active_ = true;
      }
    }
  }

  template <typename NameType>
  static void EmitPerfettoBegin(NameType&& name) {
    PERFETTO_USE_CATEGORIES_FROM_NAMESPACE_SCOPED(tpu_raiden);
    if constexpr (std::is_invocable_v<NameType>) {
      auto dynamic_name = name();
      TRACE_EVENT_BEGIN(
          "tpu_raiden",
          ::perfetto::DynamicString(dynamic_name.data(), dynamic_name.size()));
    } else if constexpr (std::is_same_v<std::decay_t<NameType>,
                                        ::perfetto::DynamicString> ||
                         std::is_same_v<std::decay_t<NameType>,
                                        ::perfetto::StaticString>) {
      TRACE_EVENT_BEGIN("tpu_raiden", name);
    } else if constexpr (std::is_array_v<std::remove_reference_t<NameType>>) {
      TRACE_EVENT_BEGIN("tpu_raiden", name);
    } else if constexpr (std::is_convertible_v<NameType, absl::string_view>) {
      absl::string_view sv = name;
      TRACE_EVENT_BEGIN("tpu_raiden",
                        ::perfetto::DynamicString(sv.data(), sv.size()));
    } else {
      TRACE_EVENT_BEGIN("tpu_raiden", name);
    }
  }

  std::optional<::tsl::profiler::TraceMe> traceme_;
  bool perfetto_active_ = false;
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
