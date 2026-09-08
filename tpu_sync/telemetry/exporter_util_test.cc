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

#include "tpu_sync/telemetry/exporter_util.h"

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace tpu_raiden::telemetry {
namespace {

TEST(ExporterUtilTest, JoinHostPort) {
  struct TestCase {
    absl::string_view host;
    int port;
    absl::string_view expected;
  };

  constexpr TestCase kTestCases[] = {
      // Standard IPv4
      {"127.0.0.1", 8080, "127.0.0.1:8080"},
      {"0.0.0.0", 9090, "0.0.0.0:9090"},
      // IPv6 without brackets (must be bracketed per RFC 3986)
      {"::1", 8080, "[::1]:8080"},
      {"2001:db8::1", 9090, "[2001:db8::1]:9090"},
      // Pre-bracketed IPv6 (must not be double-bracketed)
      {"[::1]", 8080, "[::1]:8080"},
      {"[2001:db8::1]", 9090, "[2001:db8::1]:9090"},
      // Edge cases
      {"", 8080, ":8080"},
      {"localhost", 0, "localhost:0"},
      {"localhost", 65535, "localhost:65535"},
  };

  for (const auto& [host, port, expected] : kTestCases) {
    SCOPED_TRACE(absl::StrCat("host: '", host, "', port: ", port));
    EXPECT_EQ(JoinHostPort(host, port), expected);
  }
}

}  // namespace
}  // namespace tpu_raiden::telemetry
