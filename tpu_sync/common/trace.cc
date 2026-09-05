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

#include "tpu_sync/common/trace.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "perfetto/tracing/backend_type.h"
#include "perfetto/tracing/core/data_source_config.h"  // IWYU pragma: keep
#include "perfetto/tracing/core/trace_config.h"  // IWYU pragma: keep
#include "perfetto/tracing/tracing.h"
#include "perfetto/tracing/track_event.h"

PERFETTO_TRACK_EVENT_STATIC_STORAGE_IN_NAMESPACE(tpu_raiden);

namespace tpu_raiden {
namespace {

constexpr int8_t kNoOverride = -1;
std::atomic<int8_t> g_trace_backend_override{kNoOverride};

struct FileTraceSessionState {
  std::unique_ptr<::perfetto::TracingSession> session;
  std::string path;
};

absl::Mutex g_file_trace_mutex(absl::kConstInit);
std::unique_ptr<FileTraceSessionState> g_file_trace_state
    ABSL_GUARDED_BY(g_file_trace_mutex);

bool IsEnvVarTruthy(const char* val) {
  if (val == nullptr || val[0] == '\0') {
    return false;
  }
  absl::string_view sv(val);
  if (sv == "0" || absl::EqualsIgnoreCase(sv, "false") ||
      absl::EqualsIgnoreCase(sv, "off") || absl::EqualsIgnoreCase(sv, "no")) {
    return false;
  }
  return true;
}

bool StartPerfettoTraceToFileInternal(absl::string_view output_file_path) {
  if (output_file_path.empty()) {
    return false;
  }
  std::string file_path(output_file_path);
  {
    std::ofstream test_out(file_path, std::ios::out | std::ios::binary);
    if (!test_out) {
      return false;
    }
  }

  absl::MutexLock lock(g_file_trace_mutex);
  if (g_file_trace_state != nullptr) {
    return false;
  }

  ::perfetto::TraceConfig cfg;
  cfg.add_buffers()->set_size_kb(4096);
  auto* ds_cfg = cfg.add_data_sources()->mutable_config();
  ds_cfg->set_name("track_event");

  auto session = ::perfetto::Tracing::NewTrace(::perfetto::kInProcessBackend);
  if (!session) {
    return false;
  }

  session->Setup(cfg);
  session->StartBlocking();

  auto state = std::make_unique<FileTraceSessionState>();
  state->session = std::move(session);
  state->path = std::string(output_file_path);
  g_file_trace_state = std::move(state);
  return true;
}

}  // namespace

void EnsurePerfettoInitialized() {
  static absl::once_flag once;
  absl::call_once(once, []() {
    if (!::perfetto::Tracing::IsInitialized()) {
      ::perfetto::TracingInitArgs args;
      args.backends =
          ::perfetto::kSystemBackend | ::perfetto::kInProcessBackend;
      ::perfetto::Tracing::Initialize(args);
    }
    ::tpu_raiden::TrackEvent::Register();

    const char* output_file = std::getenv("TPU_RAIDEN_PERFETTO_OUTPUT_FILE");
    if (output_file == nullptr || output_file[0] == '\0') {
      output_file = std::getenv("RAIDEN_PERFETTO_OUTPUT_FILE");
    }
    if (output_file != nullptr && output_file[0] != '\0') {
      if (StartPerfettoTraceToFileInternal(output_file)) {
        std::atexit([]() { StopPerfettoTraceToFile(); });
      }
    }
  });
}

bool StartPerfettoTraceToFile(absl::string_view output_file_path) {
  EnsurePerfettoInitialized();
  return StartPerfettoTraceToFileInternal(output_file_path);
}

bool StopPerfettoTraceToFile() {
  std::unique_ptr<FileTraceSessionState> state;
  {
    absl::MutexLock lock(g_file_trace_mutex);
    if (!g_file_trace_state) {
      return false;
    }
    state = std::move(g_file_trace_state);
  }

  ::tpu_raiden::TrackEvent::Flush();
  state->session->StopBlocking();
  std::vector<char> trace_data = state->session->ReadTraceBlocking();

  std::ofstream out(state->path, std::ios::out | std::ios::binary);
  if (!out) {
    return false;
  }
  if (!trace_data.empty()) {
    out.write(trace_data.data(),
              static_cast<std::streamsize>(trace_data.size()));
  }
  out.close();
  return out.good();
}

void SetTraceBackendForTesting(std::optional<TraceBackend> backend) {
  if (backend.has_value()) {
    g_trace_backend_override.store(static_cast<int8_t>(*backend),
                                   std::memory_order_relaxed);
  } else {
    g_trace_backend_override.store(kNoOverride, std::memory_order_relaxed);
  }
}

TraceBackend GetTraceBackend() {
  int8_t override_val =
      g_trace_backend_override.load(std::memory_order_relaxed);
  if (override_val != kNoOverride) {
    return static_cast<TraceBackend>(override_val);
  }

  const char* backend_env = std::getenv("TPU_RAIDEN_TRACE_BACKEND");
  if (backend_env == nullptr || backend_env[0] == '\0') {
    backend_env = std::getenv("RAIDEN_TRACE_BACKEND");
  }

  if (backend_env != nullptr && backend_env[0] != '\0') {
    std::string s = absl::AsciiStrToLower(backend_env);
    if (s == "perfetto") {
      return TraceBackend::kPerfetto;
    }
    if (s == "traceme") {
      return TraceBackend::kTraceMe;
    }
    if (s == "both") {
      return TraceBackend::kBoth;
    }
    if (s == "none" || s == "off" || s == "0") {
      return TraceBackend::kNone;
    }
  }

  const char* use_perfetto_1 = std::getenv("TPU_RAIDEN_USE_PERFETTO");
  const char* use_perfetto_2 = std::getenv("RAIDEN_USE_PERFETTO");
  const char* use_perfetto_3 = std::getenv("TPU_RAIDEN_TRACE_PERFETTO");
  if (IsEnvVarTruthy(use_perfetto_1) || IsEnvVarTruthy(use_perfetto_2) ||
      IsEnvVarTruthy(use_perfetto_3)) {
    return TraceBackend::kPerfetto;
  }

  return TraceBackend::kTraceMe;
}

}  // namespace tpu_raiden
