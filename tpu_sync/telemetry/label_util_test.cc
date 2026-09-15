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

#include <array>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::telemetry {
namespace {

using ::testing::ElementsAre;
using ::testing::Pair;

// ============================================================================
// Shared Memory Compact Semicolon Formatting Tests
// ============================================================================

TEST(LabelUtilTest, FormatShmLabelsToBufferEmpty) {
  char output_buffer[64];
  std::optional<absl::string_view> result =
      FormatShmLabelsToBuffer({}, absl::MakeSpan(output_buffer));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "");
  EXPECT_EQ(result->data(), output_buffer);

  // Empty buffer must also succeed for empty labels.
  std::optional<absl::string_view> empty_buffer_result =
      FormatShmLabelsToBuffer({}, absl::Span<char>());
  ASSERT_TRUE(empty_buffer_result.has_value());
  EXPECT_EQ(*empty_buffer_result, "");
}

TEST(LabelUtilTest, FormatShmLabelsToBufferSingleFastPath) {
  MetricLabel labels[] = {{"direction", "pull"}};

  // Exact-fit buffer: "direction=pull" is 14 bytes.
  char exact_buffer[14];
  std::optional<absl::string_view> exact_result =
      FormatShmLabelsToBuffer(labels, absl::MakeSpan(exact_buffer));
  ASSERT_TRUE(exact_result.has_value());
  EXPECT_EQ(*exact_result, "direction=pull");

  // 1 byte too small must fail.
  char tight_buffer[13];
  EXPECT_FALSE(FormatShmLabelsToBuffer(labels, absl::MakeSpan(tight_buffer))
                   .has_value());

  // Large single-label value formatting into large buffer.
  std::string long_value(128 + 100, 'x');
  MetricLabel large_labels[] = {{"long_key", long_value}};
  std::vector<char> large_buffer(long_value.size() + 20);
  std::optional<absl::string_view> large_result =
      FormatShmLabelsToBuffer(large_labels, absl::MakeSpan(large_buffer));
  ASSERT_TRUE(large_result.has_value());
  EXPECT_EQ(*large_result, absl::StrCat("long_key=", long_value));
}

TEST(LabelUtilTest, FormatShmLabelsToBufferMultiSorted) {
  // Input unsorted by key.
  MetricLabel labels[] = {
      {"tag", "0"},
      {"direction", "pull"},
      {"mode", "direct"},
  };

  char output_buffer[64];
  std::optional<absl::string_view> result =
      FormatShmLabelsToBuffer(labels, absl::MakeSpan(output_buffer));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "direction=pull;mode=direct;tag=0");

  // Multi-labels with identical keys: tie-broken by value.
  MetricLabel same_keys[] = {
      {"replica", "1"},
      {"replica", "0"},
  };
  char same_keys_buffer[32];
  std::optional<absl::string_view> same_result =
      FormatShmLabelsToBuffer(same_keys, absl::MakeSpan(same_keys_buffer));
  ASSERT_TRUE(same_result.has_value());
  EXPECT_EQ(*same_result, "replica=0;replica=1");

  // Large multi-label set (> kDefaultInlinedLabelCapacity = 8).
  MetricLabel ten_labels[] = {
      {"k09", "v9"}, {"k08", "v8"}, {"k07", "v7"}, {"k06", "v6"}, {"k05", "v5"},
      {"k04", "v4"}, {"k03", "v3"}, {"k02", "v2"}, {"k01", "v1"}, {"k00", "v0"},
  };
  char ten_labels_buffer[128];
  std::optional<absl::string_view> ten_result =
      FormatShmLabelsToBuffer(ten_labels, absl::MakeSpan(ten_labels_buffer));
  ASSERT_TRUE(ten_result.has_value());
  EXPECT_EQ(
      *ten_result,
      "k00=v0;k01=v1;k02=v2;k03=v3;k04=v4;k05=v5;k06=v6;k07=v7;k08=v8;k09=v9");
}

TEST(LabelUtilTest, FormatShmLabelsToBufferEscaping) {
  MetricLabel labels[] = {{"key=with;delims\\", "val=with;delims\\"}};

  char output_buffer[128];
  std::optional<absl::string_view> result =
      FormatShmLabelsToBuffer(labels, absl::MakeSpan(output_buffer));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "key\\=with\\;delims\\\\=val\\=with\\;delims\\\\");

  // Multi-byte UTF-8 sequences containing escaped delimiters.
  MetricLabel utf8_labels[] = {{"モデル=名", "値;データ\\"}};
  char utf8_buffer[128];
  std::optional<absl::string_view> utf8_result =
      FormatShmLabelsToBuffer(utf8_labels, absl::MakeSpan(utf8_buffer));
  ASSERT_TRUE(utf8_result.has_value());
  EXPECT_EQ(*utf8_result, "モデル\\=名=値\\;データ\\\\");
}

TEST(LabelUtilTest, FormatShmLabelsToBufferOverflow) {
  MetricLabel multi_labels[] = {{"k1", "v1"}, {"k2", "v2"}};
  char small_buffer[5];
  EXPECT_FALSE(
      FormatShmLabelsToBuffer(multi_labels, absl::MakeSpan(small_buffer))
          .has_value());

  // Exact boundary: "k1=v1;k2=v2" is 11 bytes.
  char exact_boundary_buffer[11];
  EXPECT_TRUE(FormatShmLabelsToBuffer(multi_labels,
                                      absl::MakeSpan(exact_boundary_buffer))
                  .has_value());
  char tight_boundary_buffer[10];
  EXPECT_FALSE(FormatShmLabelsToBuffer(multi_labels,
                                       absl::MakeSpan(tight_boundary_buffer))
                   .has_value());
}

// ============================================================================
// Shared Memory Parsing Tests
// ============================================================================

TEST(LabelUtilTest, ParseShmLabelsNormal) {
  // Empty inputs and delimiter noise return empty results.
  EXPECT_TRUE(ParseShmLabels("").empty());
  EXPECT_TRUE(ParseShmLabels(";;;").empty());

  EXPECT_THAT(ParseShmLabels("direction=pull;mode=direct;tag=0"),
              ElementsAre(Pair("direction", "pull"), Pair("mode", "direct"),
                          Pair("tag", "0")));

  // Leading, trailing, and duplicate semicolons are handled cleanly.
  EXPECT_THAT(ParseShmLabels(";direction=pull;;mode=direct;"),
              ElementsAre(Pair("direction", "pull"), Pair("mode", "direct")));
}

TEST(LabelUtilTest, ParseShmLabelsEscaped) {
  EXPECT_THAT(ParseShmLabels(
                  "key\\=with\\;delims\\\\=val\\=with\\;delims\\\\;simple=ok"),
              ElementsAre(Pair("key=with;delims\\", "val=with;delims\\"),
                          Pair("simple", "ok")));

  // Escaped backslash right before semicolon.
  EXPECT_THAT(ParseShmLabels("k=v\\\\;k2=v2"),
              ElementsAre(Pair("k", "v\\"), Pair("k2", "v2")));
}

TEST(LabelUtilTest, ParseShmLabelsMalformed) {
  // Missing equal sign, valid pair, empty trailing value, trailing backslash.
  EXPECT_THAT(ParseShmLabels("invalid_no_equal;valid=1;trailing="),
              ElementsAre(Pair("valid", "1"), Pair("trailing", "")));

  EXPECT_THAT(ParseShmLabels("key=val\\"), ElementsAre(Pair("key", "val\\")));

  // Empty key with non-empty value, empty key and value.
  EXPECT_THAT(ParseShmLabels("=orphan_value;valid=2;="),
              ElementsAre(Pair("valid", "2")));
}

TEST(LabelUtilTest, ParseShmLabelsNullPaddedBuffer) {
  // Fixed-size TOC buffers (e.g. char[128]) are null-padded.
  char fixed_buffer[128] = {};
  std::memcpy(fixed_buffer, "key=value;tag=1", 15);
  EXPECT_THAT(
      ParseShmLabels(absl::string_view(fixed_buffer, sizeof(fixed_buffer))),
      ElementsAre(Pair("key", "value"), Pair("tag", "1")));
}

TEST(LabelUtilTest, RoundTripInvariance) {
  MetricLabel original_labels[] = {
      {"direction", "pull_response"},
      {"error_code", "RESOURCE_EXHAUSTED"},
      {"meta;key", "value=with\\slash"},
      {"モデル", "日本語データ=ok;test"},
  };
  char formatted_buffer[256];
  std::optional<absl::string_view> formatted = FormatShmLabelsToBuffer(
      original_labels, absl::MakeSpan(formatted_buffer));
  ASSERT_TRUE(formatted.has_value());
  std::vector<std::pair<std::string, std::string>> parsed =
      ParseShmLabels(*formatted);
  EXPECT_THAT(parsed, ElementsAre(Pair("direction", "pull_response"),
                                  Pair("error_code", "RESOURCE_EXHAUSTED"),
                                  Pair("meta;key", "value=with\\slash"),
                                  Pair("モデル", "日本語データ=ok;test")));
}

// ============================================================================
// Prometheus Canonical Formatting Tests
// ============================================================================

TEST(LabelUtilTest, FormatPrometheusLabelsEmpty) {
  EXPECT_EQ(FormatPrometheusLabels({}), "");

  char output_buffer[64];
  std::optional<absl::string_view> result =
      FormatPrometheusLabelsToBuffer({}, absl::MakeSpan(output_buffer));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "");
  EXPECT_EQ(result->data(), output_buffer);

  // Empty buffer must also succeed for empty labels.
  std::optional<absl::string_view> empty_buffer_result =
      FormatPrometheusLabelsToBuffer({}, absl::Span<char>());
  ASSERT_TRUE(empty_buffer_result.has_value());
  EXPECT_EQ(*empty_buffer_result, "");
}

TEST(LabelUtilTest, FormatPrometheusLabelsSingleFastPath) {
  MetricLabel labels[] = {{"direction", "pull"}};
  EXPECT_EQ(FormatPrometheusLabels(labels), "{direction=\"pull\"}");

  // Exact-fit buffer: "{direction=\"pull\"}" is 18 bytes.
  char exact_buffer[18];
  std::optional<absl::string_view> exact_result =
      FormatPrometheusLabelsToBuffer(labels, absl::MakeSpan(exact_buffer));
  ASSERT_TRUE(exact_result.has_value());
  EXPECT_EQ(*exact_result, "{direction=\"pull\"}");

  // 1 byte too small must fail.
  char tight_buffer[17];
  EXPECT_FALSE(
      FormatPrometheusLabelsToBuffer(labels, absl::MakeSpan(tight_buffer))
          .has_value());

  // Very large label set exceeding 256 bytes to exercise heap fallback.
  std::string long_value(300, 'y');
  MetricLabel large_labels[] = {{"long_key", long_value}};
  std::string formatted_large = FormatPrometheusLabels(large_labels);
  EXPECT_EQ(formatted_large, absl::StrCat("{long_key=\"", long_value, "\"}"));
}

TEST(LabelUtilTest, FormatPrometheusLabelsMultiSorted) {
  // Input unsorted by key.
  MetricLabel labels[] = {
      {"tag", "0"},
      {"direction", "pull"},
      {"mode", "direct"},
  };
  EXPECT_EQ(FormatPrometheusLabels(labels),
            "{direction=\"pull\",mode=\"direct\",tag=\"0\"}");

  char output_buffer[64];
  std::optional<absl::string_view> result =
      FormatPrometheusLabelsToBuffer(labels, absl::MakeSpan(output_buffer));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "{direction=\"pull\",mode=\"direct\",tag=\"0\"}");

  // Large multi-label set (> 8 labels).
  MetricLabel ten_labels[] = {
      {"k09", "v9"}, {"k08", "v8"}, {"k07", "v7"}, {"k06", "v6"}, {"k05", "v5"},
      {"k04", "v4"}, {"k03", "v3"}, {"k02", "v2"}, {"k01", "v1"}, {"k00", "v0"},
  };
  EXPECT_EQ(
      FormatPrometheusLabels(ten_labels),
      "{k00=\"v0\",k01=\"v1\",k02=\"v2\",k03=\"v3\",k04=\"v4\",k05=\"v5\","
      "k06=\"v6\",k07=\"v7\",k08=\"v8\",k09=\"v9\"}");
}

TEST(LabelUtilTest, FormatPrometheusLabelsEscapingAndUtf8Value) {
  MetricLabel labels[] = {{"query", "line1\nline2\"quoted\"with\\slash"}};
  EXPECT_EQ(FormatPrometheusLabels(labels),
            "{query=\"line1\\nline2\\\"quoted\\\"with\\\\slash\"}");

  char output_buffer[128];
  std::optional<absl::string_view> result =
      FormatPrometheusLabelsToBuffer(labels, absl::MakeSpan(output_buffer));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "{query=\"line1\\nline2\\\"quoted\\\"with\\\\slash\"}");

  // UTF-8 in value combined with newline and quotes.
  MetricLabel utf8_labels[] = {{"error_detail", "失敗: \"IO\\Error\"\n詳細"}};
  EXPECT_EQ(FormatPrometheusLabels(utf8_labels),
            "{error_detail=\"失敗: \\\"IO\\\\Error\\\"\\n詳細\"}");
}

TEST(LabelUtilTest, FormatPrometheusLabelsRawKeys) {
  // Verifies that FormatPrometheusLabels acts as a raw serializer without
  // enforcing Prometheus key naming restrictions or duplicate key rejection.
  MetricLabel raw_labels[] = {
      {"replica", "1"},
      {"replica", "0"},
  };
  EXPECT_EQ(FormatPrometheusLabels(raw_labels),
            "{replica=\"0\",replica=\"1\"}");

  MetricLabel special_keys[] = {{"bad-key", "value"}};
  EXPECT_EQ(FormatPrometheusLabels(special_keys), "{bad-key=\"value\"}");
}

TEST(LabelUtilTest, FormatPrometheusLabelsToBufferOverflow) {
  MetricLabel multi_labels[] = {{"k1", "v1"}, {"k2", "v2"}};
  char small_buffer[10];
  EXPECT_FALSE(
      FormatPrometheusLabelsToBuffer(multi_labels, absl::MakeSpan(small_buffer))
          .has_value());

  // Exact boundary: "{k1=\"v1\",k2=\"v2\"}" is 17 bytes.
  char exact_boundary_buffer[17];
  EXPECT_TRUE(FormatPrometheusLabelsToBuffer(
                  multi_labels, absl::MakeSpan(exact_boundary_buffer))
                  .has_value());
  char tight_boundary_buffer[16];
  EXPECT_FALSE(FormatPrometheusLabelsToBuffer(
                   multi_labels, absl::MakeSpan(tight_boundary_buffer))
                   .has_value());
}

// ============================================================================
// Zero-Allocation Label Resolution for Fixed Schemas Tests
// ============================================================================

TEST(LabelUtilTest, ResolveLabelsZeroArity) {
  constexpr std::array<absl::string_view, 0> kKeys{};
  const std::array<absl::string_view, 0> resolved = ResolveLabels({}, kKeys);
  EXPECT_TRUE(resolved.empty());

  // Non-empty labels passed to 0-arity (fieldless) metric cleanly return an
  // empty array without evaluating keys.
  const MetricLabel dummy_label{.key = "direction", .value = "pull"};
  const std::array<absl::string_view, 0> resolved_with_labels =
      ResolveLabels({dummy_label}, kKeys);
  EXPECT_TRUE(resolved_with_labels.empty());
}

TEST(LabelUtilTest, ResolveLabelsSingleArity) {
  constexpr std::array<absl::string_view, 1> kKeys{"direction"};
  {
    const MetricLabel label{.key = "direction", .value = "pull"};
    const std::array<absl::string_view, 1> resolved =
        ResolveLabels({label}, kKeys);
    EXPECT_THAT(resolved, ElementsAre("pull"));
  }
  {
    // Missing key defaults to "unknown"
    const MetricLabel label{.key = "other", .value = "val"};
    const std::array<absl::string_view, 1> resolved =
        ResolveLabels({label}, kKeys);
    EXPECT_THAT(resolved, ElementsAre("unknown"));
  }
  {
    // Empty string value is preserved and does not default to "unknown"
    const MetricLabel label{.key = "direction", .value = ""};
    const std::array<absl::string_view, 1> resolved =
        ResolveLabels({label}, kKeys);
    EXPECT_THAT(resolved, ElementsAre(""));
  }
  {
    // Duplicate key preserves first-match semantics
    const std::array<MetricLabel, 2> labels = {
        MetricLabel{.key = "direction", .value = "first"},
        MetricLabel{.key = "direction", .value = "second"},
    };
    const std::array<absl::string_view, 1> resolved =
        ResolveLabels(labels, kKeys);
    EXPECT_THAT(resolved, ElementsAre("first"));
  }
}

TEST(LabelUtilTest, ResolveLabelsMultiArity) {
  constexpr std::array<absl::string_view, 2> kKeys{"direction", "error_code"};
  // In order match
  {
    const std::array<MetricLabel, 2> labels = {
        MetricLabel{.key = "direction", .value = "push"},
        MetricLabel{.key = "error_code", .value = "CANCELLED"},
    };
    const std::array<absl::string_view, 2> resolved =
        ResolveLabels(labels, kKeys);
    EXPECT_THAT(resolved, ElementsAre("push", "CANCELLED"));
  }
  // Reversed out-of-order match
  {
    const std::array<MetricLabel, 2> labels = {
        MetricLabel{.key = "error_code", .value = "UNAVAILABLE"},
        MetricLabel{.key = "direction", .value = "pull"},
    };
    const std::array<absl::string_view, 2> resolved =
        ResolveLabels(labels, kKeys);
    EXPECT_THAT(resolved, ElementsAre("pull", "UNAVAILABLE"));
  }
  // Partial match with fallback to default
  {
    const std::array<MetricLabel, 1> labels = {
        MetricLabel{.key = "error_code", .value = "INTERNAL"},
    };
    const std::array<absl::string_view, 2> resolved =
        ResolveLabels(labels, kKeys);
    EXPECT_THAT(resolved, ElementsAre("unknown", "INTERNAL"));
  }
  // Custom default value fallback
  {
    const std::array<absl::string_view, 2> resolved =
        ResolveLabels({}, kKeys, "none");
    EXPECT_THAT(resolved, ElementsAre("none", "none"));
  }
}

TEST(LabelUtilTest, ResolveLabelsEarlyExitOnAllFound) {
  constexpr std::array<absl::string_view, 3> kKeys{"a", "b", "c"};
  const std::array<MetricLabel, 5> labels = {
      MetricLabel{.key = "b", .value = "2"},
      MetricLabel{.key = "a", .value = "1"},
      MetricLabel{.key = "c", .value = "3"},
      MetricLabel{.key = "d", .value = "4"},
      MetricLabel{.key = "e", .value = "5"},
  };
  const std::array<absl::string_view, 3> resolved =
      ResolveLabels(labels, kKeys);
  EXPECT_THAT(resolved, ElementsAre("1", "2", "3"));
}

TEST(LabelUtilTest, ResolveLabelsConstexprEvaluation) {
  constexpr std::array<absl::string_view, 0> kZeroKeys{};
  static_assert(ResolveLabels({}, kZeroKeys).empty());

  constexpr std::array<absl::string_view, 2> kKeys{"a", "b"};
  constexpr std::array<MetricLabel, 2> kLabels = {
      MetricLabel{.key = "b", .value = "2"},
      MetricLabel{.key = "a", .value = "1"},
  };
  constexpr auto kResolved = ResolveLabels(kLabels, kKeys);
  static_assert(kResolved[0] == "1");
  static_assert(kResolved[1] == "2");

  // Missing label falls back to default value at compile time
  constexpr std::array<MetricLabel, 1> kPartialLabels = {
      MetricLabel{.key = "b", .value = "val_b"},
  };
  constexpr auto kPartialResolved =
      ResolveLabels(kPartialLabels, kKeys, "default");
  static_assert(kPartialResolved[0] == "default");
  static_assert(kPartialResolved[1] == "val_b");
}

}  // namespace
}  // namespace tpu_raiden::telemetry
