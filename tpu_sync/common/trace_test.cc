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

#include <stdlib.h>

#include <fstream>
#include <ios>
#include <iterator>
#include <optional>
#include <string>  // IWYU pragma: keep
#include <vector>  // IWYU pragma: keep

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "perfetto/tracing/backend_type.h"  // IWYU pragma: keep
#include "perfetto/tracing/core/trace_config.h"  // IWYU pragma: keep
#include "perfetto/tracing/tracing.h"  // IWYU pragma: keep

namespace tpu_raiden {
namespace {

class TraceBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    SetTraceBackendForTesting(std::nullopt);
    unsetenv("TPU_RAIDEN_TRACE_BACKEND");
    unsetenv("RAIDEN_TRACE_BACKEND");
    unsetenv("TPU_RAIDEN_USE_PERFETTO");
    unsetenv("RAIDEN_USE_PERFETTO");
    unsetenv("TPU_RAIDEN_TRACE_PERFETTO");
  }

  void TearDown() override {
    SetTraceBackendForTesting(std::nullopt);
    unsetenv("TPU_RAIDEN_TRACE_BACKEND");
    unsetenv("RAIDEN_TRACE_BACKEND");
    unsetenv("TPU_RAIDEN_USE_PERFETTO");
    unsetenv("RAIDEN_USE_PERFETTO");
    unsetenv("TPU_RAIDEN_TRACE_PERFETTO");
  }
};

TEST_F(TraceBackendTest, GetTraceBackendDefault) {
  EXPECT_EQ(GetTraceBackend(), TraceBackend::kTraceMe);
}

TEST_F(TraceBackendTest, SetTraceBackendForTestingOverride) {
  SetTraceBackendForTesting(TraceBackend::kPerfetto);
  EXPECT_EQ(GetTraceBackend(), TraceBackend::kPerfetto);

  SetTraceBackendForTesting(TraceBackend::kBoth);
  EXPECT_EQ(GetTraceBackend(), TraceBackend::kBoth);

  SetTraceBackendForTesting(TraceBackend::kNone);
  EXPECT_EQ(GetTraceBackend(), TraceBackend::kNone);

  SetTraceBackendForTesting(TraceBackend::kTraceMe);
  EXPECT_EQ(GetTraceBackend(), TraceBackend::kTraceMe);

  SetTraceBackendForTesting(std::nullopt);
  EXPECT_EQ(GetTraceBackend(), TraceBackend::kTraceMe);
}

TEST_F(TraceBackendTest, BackendEnvVarsBackendName) {
  setenv("TPU_RAIDEN_TRACE_BACKEND", "perfetto", 1);
  EXPECT_EQ(GetTraceBackend(), TraceBackend::kPerfetto);

  setenv("TPU_RAIDEN_TRACE_BACKEND", "both", 1);
  EXPECT_EQ(GetTraceBackend(), TraceBackend::kBoth);

  setenv("TPU_RAIDEN_TRACE_BACKEND", "none", 1);
  EXPECT_EQ(GetTraceBackend(), TraceBackend::kNone);

  setenv("TPU_RAIDEN_TRACE_BACKEND", "0", 1);
  EXPECT_EQ(GetTraceBackend(), TraceBackend::kNone);

  setenv("TPU_RAIDEN_TRACE_BACKEND", "off", 1);
  EXPECT_EQ(GetTraceBackend(), TraceBackend::kNone);

  setenv("TPU_RAIDEN_TRACE_BACKEND", "traceme", 1);
  EXPECT_EQ(GetTraceBackend(), TraceBackend::kTraceMe);
  unsetenv("TPU_RAIDEN_TRACE_BACKEND");

  // Fallback to RAIDEN_TRACE_BACKEND
  setenv("RAIDEN_TRACE_BACKEND", "perfetto", 1);
  EXPECT_EQ(GetTraceBackend(), TraceBackend::kPerfetto);
  unsetenv("RAIDEN_TRACE_BACKEND");
}

TEST_F(TraceBackendTest, BackendEnvVarsUsePerfettoFlags) {
  setenv("TPU_RAIDEN_USE_PERFETTO", "1", 1);
  EXPECT_EQ(GetTraceBackend(), TraceBackend::kPerfetto);
  unsetenv("TPU_RAIDEN_USE_PERFETTO");

  setenv("RAIDEN_USE_PERFETTO", "true", 1);
  EXPECT_EQ(GetTraceBackend(), TraceBackend::kPerfetto);
  unsetenv("RAIDEN_USE_PERFETTO");

  setenv("TPU_RAIDEN_TRACE_PERFETTO", "1", 1);
  EXPECT_EQ(GetTraceBackend(), TraceBackend::kPerfetto);
  unsetenv("TPU_RAIDEN_TRACE_PERFETTO");

  // Disabled flags should not select Perfetto
  setenv("TPU_RAIDEN_USE_PERFETTO", "0", 1);
  EXPECT_EQ(GetTraceBackend(), TraceBackend::kTraceMe);
  unsetenv("TPU_RAIDEN_USE_PERFETTO");
}

TEST(TraceTest, ScopedTraceMacroWithLiteral) {
  RAIDEN_TRACE("TestLiteral");
  int val = 42;
  EXPECT_EQ(val, 42);
}

TEST(TraceTest, ScopedTraceMacroWithLevel) {
  RAIDEN_TRACE("TestLevel", 2);
  EXPECT_EQ(TraceMeLevel::kInfo, 2);
}

TEST(TraceTest, ScopedTraceMacroWithLambda) {
  int op_id = 1;
  RAIDEN_TRACE([&]() { return absl::StrCat("Op_", op_id); });
  EXPECT_EQ(op_id, 1);
}

TEST(TraceTest, ScopedTraceAliasMacro) {
  RAIDEN_TRACE_SCOPE("TestScope");
  int val = 100;
  EXPECT_EQ(val, 100);
}

TEST(TraceTest, ScopedTraceFnWithMetadata) {
  RAIDEN_TRACE_FN("TestFn", []() { return TraceMeEncode({{"k", "v"}}); });
  int val = 10;
  EXPECT_EQ(val, 10);
}

TEST(TraceTest, NestedScopes) {
  RAIDEN_TRACE("OuterScope");
  int count = 1;
  {
    RAIDEN_TRACE("InnerScope1");
    count += 1;
    {
      RAIDEN_TRACE("InnerScope2");
      count += 1;
    }
  }
  EXPECT_EQ(count, 3);
}

TEST(TraceTest, ExplicitBackendMacros) {
  RAIDEN_PERFETTO_TRACE("ExplicitPerfetto");
  RAIDEN_PERFETTO_TRACE("ExplicitPerfettoLevel", 2);
  RAIDEN_TRACEME_TRACE("ExplicitTraceMe");
  RAIDEN_TRACEME_TRACE("ExplicitTraceMeLevel", 2);
}

TEST(TraceTest, BothAndNoneBackends) {
  SetTraceBackendForTesting(TraceBackend::kBoth);
  {
    RAIDEN_TRACE("BothTrace");
    RAIDEN_TRACE_FN("BothTraceFn", []() { return "key=val"; });
  }

  SetTraceBackendForTesting(TraceBackend::kNone);
  {
    RAIDEN_TRACE("NoneTrace");
    RAIDEN_TRACE_FN("NoneTraceFn", []() { return "key=val"; });
  }

  SetTraceBackendForTesting(std::nullopt);
}

TEST(TraceTest, TraceLogMacro) {
  // Disabled state by default
  unsetenv("TPU_RAIDEN_TRACE");
  unsetenv("RAIDEN_TRACE");
  EXPECT_FALSE(IsTraceLoggingEnabled());
  RAIDEN_TRACE_LOG("This message is suppressed when trace is disabled: ", 123);

  // Enabled via TPU_RAIDEN_TRACE
  setenv("TPU_RAIDEN_TRACE", "1", 1);
  EXPECT_TRUE(IsTraceLoggingEnabled());
  RAIDEN_TRACE_LOG("Trace logging via TPU_RAIDEN_TRACE: ", "active",
                   ", count=", 42);
  unsetenv("TPU_RAIDEN_TRACE");

  // Enabled via RAIDEN_TRACE
  setenv("RAIDEN_TRACE", "1", 1);
  EXPECT_TRUE(IsTraceLoggingEnabled());
  RAIDEN_TRACE_LOG("Trace logging via RAIDEN_TRACE: ", "active");
  unsetenv("RAIDEN_TRACE");

  // Explicitly disabled with "0"
  setenv("TPU_RAIDEN_TRACE", "0", 1);
  EXPECT_FALSE(IsTraceLoggingEnabled());
  unsetenv("TPU_RAIDEN_TRACE");
}

TEST(TraceTest, PerfettoInProcessTraceCollection) {
  EnsurePerfettoInitialized();

  // NOLINTNEXTLINE(misc-include-cleaner)
  perfetto::TraceConfig cfg;
  cfg.add_buffers()->set_size_kb(1024);
  auto* ds_cfg = cfg.add_data_sources()->mutable_config();
  ds_cfg->set_name("track_event");

  auto session = perfetto::Tracing::NewTrace(perfetto::kInProcessBackend);
  session->Setup(cfg);
  session->StartBlocking();

  SetTraceBackendForTesting(TraceBackend::kPerfetto);

  {
    RAIDEN_TRACE("PerfettoScopeEvent");
    RAIDEN_PERFETTO_TRACE("PerfettoExplicitEvent");
    RAIDEN_TRACE([&]() { return absl::StrCat("PerfettoDynamicOp_", 42); });
    RAIDEN_TRACE_FN("PerfettoFnEvent",
                    []() { return "debug_detail=perfetto_ok"; });
  }

  SetTraceBackendForTesting(std::nullopt);

  ::tpu_raiden::TrackEvent::Flush();
  session->StopBlocking();
  std::vector<char> trace_data = session->ReadTraceBlocking();

  EXPECT_FALSE(trace_data.empty());
  absl::string_view trace_view(trace_data.data(), trace_data.size());
  EXPECT_TRUE(absl::StrContains(trace_view, "PerfettoScopeEvent"));
  EXPECT_TRUE(absl::StrContains(trace_view, "PerfettoExplicitEvent"));
  EXPECT_TRUE(absl::StrContains(trace_view, "PerfettoDynamicOp_42"));
}

TEST(TraceTest, StartAndStopPerfettoTraceToFile) {
  std::string temp_dir = testing::TempDir();
  std::string trace_file =
      absl::StrCat(temp_dir, "/raiden_trace_file_test_",
                   absl::ToUnixMicros(absl::Now()), ".pftrace");

  // Ensure no prior session is running
  StopPerfettoTraceToFile();

  ASSERT_TRUE(StartPerfettoTraceToFile(trace_file));

  // A second call should fail while active
  EXPECT_FALSE(StartPerfettoTraceToFile(trace_file));

  SetTraceBackendForTesting(TraceBackend::kPerfetto);
  {
    RAIDEN_TRACE("FileTraceScopedSlice");
    RAIDEN_PERFETTO_TRACE("FileTraceExplicitSlice");
    RAIDEN_TRACE([&]() { return absl::StrCat("FileTraceDynamicSlice_", 99); });
  }
  SetTraceBackendForTesting(std::nullopt);

  ASSERT_TRUE(StopPerfettoTraceToFile());

  // A second stop should return false
  EXPECT_FALSE(StopPerfettoTraceToFile());

  // Read back the file and verify non-empty content containing slice names
  std::ifstream in(trace_file, std::ios::in | std::ios::binary);
  ASSERT_TRUE(in.is_open());
  std::string file_content((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
  in.close();

  EXPECT_FALSE(file_content.empty());
  absl::string_view view(file_content);
  EXPECT_TRUE(absl::StrContains(view, "FileTraceScopedSlice"));
  EXPECT_TRUE(absl::StrContains(view, "FileTraceExplicitSlice"));
  EXPECT_TRUE(absl::StrContains(view, "FileTraceDynamicSlice_99"));
}

}  // namespace
}  // namespace tpu_raiden
