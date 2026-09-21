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
#include <type_traits>
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

TEST(LabelUtilTest, FormatPrometheusLabelsToBufferEmpty) {
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

TEST(LabelUtilTest, FormatPrometheusLabelsToBufferSingleFastPath) {
  const MetricLabel labels[] = {{"direction", "pull"}};

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

  // Large label set exceeding 256 bytes.
  std::string long_value(300, 'y');
  const MetricLabel large_labels[] = {{"long_key", long_value}};
  char large_buffer[350];
  std::optional<absl::string_view> large_result =
      FormatPrometheusLabelsToBuffer(large_labels,
                                     absl::MakeSpan(large_buffer));
  ASSERT_TRUE(large_result.has_value());
  EXPECT_EQ(*large_result, absl::StrCat("{long_key=\"", long_value, "\"}"));
}

TEST(LabelUtilTest, FormatPrometheusLabelsToBufferMultiSorted) {
  // Input unsorted by key.
  const MetricLabel labels[] = {
      {"tag", "0"},
      {"direction", "pull"},
      {"mode", "direct"},
  };
  char output_buffer[64];
  std::optional<absl::string_view> result =
      FormatPrometheusLabelsToBuffer(labels, absl::MakeSpan(output_buffer));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "{direction=\"pull\",mode=\"direct\",tag=\"0\"}");

  // Large multi-label set (> 8 labels).
  const MetricLabel ten_labels[] = {
      {"k09", "v9"}, {"k08", "v8"}, {"k07", "v7"}, {"k06", "v6"}, {"k05", "v5"},
      {"k04", "v4"}, {"k03", "v3"}, {"k02", "v2"}, {"k01", "v1"}, {"k00", "v0"},
  };
  char large_output_buffer[256];
  std::optional<absl::string_view> ten_result = FormatPrometheusLabelsToBuffer(
      ten_labels, absl::MakeSpan(large_output_buffer));
  ASSERT_TRUE(ten_result.has_value());
  EXPECT_EQ(
      *ten_result,
      "{k00=\"v0\",k01=\"v1\",k02=\"v2\",k03=\"v3\",k04=\"v4\",k05=\"v5\","
      "k06=\"v6\",k07=\"v7\",k08=\"v8\",k09=\"v9\"}");
}

TEST(LabelUtilTest, FormatPrometheusLabelsToBufferEscapingAndUtf8Value) {
  const MetricLabel labels[] = {{"query", "line1\nline2\"quoted\"with\\slash"}};
  char output_buffer[128];
  std::optional<absl::string_view> result =
      FormatPrometheusLabelsToBuffer(labels, absl::MakeSpan(output_buffer));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "{query=\"line1\\nline2\\\"quoted\\\"with\\\\slash\"}");

  // UTF-8 in value combined with newline and quotes.
  const MetricLabel utf8_labels[] = {
      {"error_detail", "失敗: \"IO\\Error\"\n詳細"}};
  char utf8_buffer[128];
  std::optional<absl::string_view> utf8_result =
      FormatPrometheusLabelsToBuffer(utf8_labels, absl::MakeSpan(utf8_buffer));
  ASSERT_TRUE(utf8_result.has_value());
  EXPECT_EQ(*utf8_result,
            "{error_detail=\"失敗: \\\"IO\\\\Error\\\"\\n詳細\"}");
}

TEST(LabelUtilTest, FormatPrometheusLabelsToBufferRawKeys) {
  // Verifies that FormatPrometheusLabelsToBuffer acts as a raw serializer
  // without enforcing Prometheus key naming restrictions or duplicate key
  // rejection.
  const MetricLabel raw_labels[] = {
      {"replica", "1"},
      {"replica", "0"},
  };
  char raw_buffer[64];
  std::optional<absl::string_view> raw_result =
      FormatPrometheusLabelsToBuffer(raw_labels, absl::MakeSpan(raw_buffer));
  ASSERT_TRUE(raw_result.has_value());
  EXPECT_EQ(*raw_result, "{replica=\"0\",replica=\"1\"}");

  const MetricLabel special_keys[] = {{"bad-key", "value"}};
  char special_buffer[32];
  std::optional<absl::string_view> special_result =
      FormatPrometheusLabelsToBuffer(special_keys,
                                     absl::MakeSpan(special_buffer));
  ASSERT_TRUE(special_result.has_value());
  EXPECT_EQ(*special_result, "{bad-key=\"value\"}");
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

TEST(LabelUtilTest, PrometheusLabelViewEmpty) {
  PrometheusLabelView view({});
  EXPECT_TRUE(view.view().empty());
  EXPECT_TRUE(PrometheusLabelView({}).ToOwned().empty());
}

TEST(LabelUtilTest, PrometheusLabelViewStackFit) {
  const std::array<MetricLabel, 2> labels = {
      MetricLabel{.key = "direction", .value = "push"},
      MetricLabel{.key = "mode", .value = "direct"},
  };
  PrometheusLabelView view(labels);
  EXPECT_EQ(view.view(), "{direction=\"push\",mode=\"direct\"}");
  EXPECT_EQ(PrometheusLabelView(labels).ToOwned(),
            "{direction=\"push\",mode=\"direct\"}");
}

TEST(LabelUtilTest, PrometheusLabelViewDefaultCapacityOverflow) {
  const std::string long_value(kDefaultPrometheusStackBufferSize + 64, 'x');
  const std::array<MetricLabel, 1> labels = {
      MetricLabel{.key = "key", .value = long_value},
  };
  PrometheusLabelView view(labels);
  const std::string expected = absl::StrCat("{key=\"", long_value, "\"}");
  EXPECT_EQ(view.view(), expected);
  const std::string owned = view.ToOwned();
  EXPECT_EQ(owned, expected);
  EXPECT_TRUE(view.view().empty());
}

TEST(LabelUtilTest, PrometheusLabelViewExactStackBufferBoundaries) {
  // "{k=\"...\"}" has 6 bytes of syntax overhead ('{', 'k', '=', '"', '"',
  // '}'). 255 bytes (kDefaultPrometheusStackBufferSize - 1)
  {
    const std::string val_255(kDefaultPrometheusStackBufferSize - 7, 'a');
    const std::array<MetricLabel, 1> labels = {
        MetricLabel{.key = "k", .value = val_255},
    };
    const std::string expected = absl::StrCat("{k=\"", val_255, "\"}");
    ASSERT_EQ(expected.size(), kDefaultPrometheusStackBufferSize - 1);
    PrometheusLabelView view(labels);
    EXPECT_EQ(view.view(), expected);
    EXPECT_EQ(view.ToOwned(), expected);
  }
  // 256 bytes (kDefaultPrometheusStackBufferSize, exact stack fit)
  {
    const std::string val_256(kDefaultPrometheusStackBufferSize - 6, 'b');
    const std::array<MetricLabel, 1> labels = {
        MetricLabel{.key = "k", .value = val_256},
    };
    const std::string expected = absl::StrCat("{k=\"", val_256, "\"}");
    ASSERT_EQ(expected.size(), kDefaultPrometheusStackBufferSize);
    PrometheusLabelView view(labels);
    EXPECT_EQ(view.view(), expected);
    EXPECT_EQ(view.ToOwned(), expected);
  }
  // 257 bytes (kDefaultPrometheusStackBufferSize + 1, minimal heap fallback)
  {
    const std::string val_257(kDefaultPrometheusStackBufferSize - 5, 'c');
    const std::array<MetricLabel, 1> labels = {
        MetricLabel{.key = "k", .value = val_257},
    };
    const std::string expected = absl::StrCat("{k=\"", val_257, "\"}");
    ASSERT_EQ(expected.size(), kDefaultPrometheusStackBufferSize + 1);
    PrometheusLabelView view(labels);
    EXPECT_EQ(view.view(), expected);
    EXPECT_EQ(view.ToOwned(), expected);
  }
}

TEST(LabelUtilTest, PrometheusLabelViewEmptyLabelValue) {
  const std::array<MetricLabel, 1> labels = {
      MetricLabel{.key = "empty_key", .value = ""},
  };
  PrometheusLabelView view(labels);
  EXPECT_EQ(view.view(), "{empty_key=\"\"}");
  EXPECT_EQ(view.ToOwned(), "{empty_key=\"\"}");
}

TEST(LabelUtilTest, PrometheusLabelViewRawKeys) {
  const std::array<MetricLabel, 2> raw_labels = {
      MetricLabel{.key = "replica", .value = "1"},
      MetricLabel{.key = "replica", .value = "0"},
  };
  PrometheusLabelView view(raw_labels);
  EXPECT_EQ(view.view(), "{replica=\"0\",replica=\"1\"}");

  const std::array<MetricLabel, 1> special_keys = {
      MetricLabel{.key = "bad-key", .value = "value"},
  };
  PrometheusLabelView special_view(special_keys);
  EXPECT_EQ(special_view.view(), "{bad-key=\"value\"}");
}

TEST(LabelUtilTest, PrometheusLabelViewUtf8Value) {
  const std::array<MetricLabel, 1> utf8_labels = {
      MetricLabel{.key = "error_detail", .value = "失敗: \"IO\\Error\"\n詳細"},
  };
  PrometheusLabelView view(utf8_labels);
  EXPECT_EQ(view.view(), "{error_detail=\"失敗: \\\"IO\\\\Error\\\"\\n詳細\"}");
}

TEST(LabelUtilTest, PrometheusLabelViewNonCopyableNonMovable) {
  static_assert(!std::is_copy_constructible_v<PrometheusLabelView>);
  static_assert(!std::is_copy_assignable_v<PrometheusLabelView>);
  static_assert(!std::is_move_constructible_v<PrometheusLabelView>);
  static_assert(!std::is_move_assignable_v<PrometheusLabelView>);
}

TEST(LabelUtilTest, PrometheusLabelViewUnsortedLabels) {
  const std::array<MetricLabel, 2> labels = {
      MetricLabel{.key = "z", .value = "1"},
      MetricLabel{.key = "a", .value = "2"},
  };
  PrometheusLabelView view(labels);
  EXPECT_EQ(view.view(), "{a=\"2\",z=\"1\"}");
  EXPECT_EQ(PrometheusLabelView(labels).ToOwned(), "{a=\"2\",z=\"1\"}");
}

TEST(LabelUtilTest, PrometheusLabelViewUnsortedLabelsOverflow) {
  const std::string long_value(kDefaultPrometheusStackBufferSize + 64, 'x');
  const std::array<MetricLabel, 2> labels = {
      MetricLabel{.key = "z", .value = long_value},
      MetricLabel{.key = "a", .value = "first"},
  };
  PrometheusLabelView view(labels);
  const std::string expected =
      absl::StrCat("{a=\"first\",z=\"", long_value, "\"}");
  EXPECT_EQ(view.view(), expected);
  EXPECT_EQ(view.ToOwned(), expected);
}

TEST(LabelUtilTest,
     PrometheusLabelViewUnsortedExceedingInlinedCapacityAndStackBuffer) {
  const std::string val(32, 'v');
  const MetricLabel labels[] = {
      {"k09", val}, {"k08", val}, {"k07", val}, {"k06", val}, {"k05", val},
      {"k04", val}, {"k03", val}, {"k02", val}, {"k01", val}, {"k00", val},
  };
  const std::string expected = absl::StrCat(
      "{k00=\"", val, "\",k01=\"", val, "\",k02=\"", val, "\",k03=\"", val,
      "\",k04=\"", val, "\",k05=\"", val, "\",k06=\"", val, "\",k07=\"", val,
      "\",k08=\"", val, "\",k09=\"", val, "\"}");
  ASSERT_GT(expected.size(), kDefaultPrometheusStackBufferSize);
  PrometheusLabelView view(labels);
  EXPECT_EQ(view.view(), expected);
  EXPECT_EQ(view.ToOwned(), expected);
}

TEST(LabelUtilTest, PrometheusLabelViewEscapedCharacters) {
  const std::array<MetricLabel, 1> labels = {
      MetricLabel{.key = "msg", .value = "hello \"world\"\npath\\to"},
  };
  PrometheusLabelView view(labels);
  EXPECT_EQ(view.view(), "{msg=\"hello \\\"world\\\"\\npath\\\\to\"}");
  EXPECT_EQ(PrometheusLabelView(labels).ToOwned(),
            "{msg=\"hello \\\"world\\\"\\npath\\\\to\"}");
}

TEST(LabelUtilTest, PrometheusLabelViewEscapingOverflow) {
  // 140 quotes: unescaped length is 140, but escaped length is 280 > 256.
  const std::string quotes(140, '"');
  const std::array<MetricLabel, 1> labels = {
      MetricLabel{.key = "q", .value = quotes},
  };
  PrometheusLabelView view(labels);
  std::string expected_val;
  expected_val.reserve(280);
  for (int i = 0; i < 140; ++i) {
    expected_val += "\\\"";
  }
  const std::string expected = absl::StrCat("{q=\"", expected_val, "\"}");
  EXPECT_EQ(view.view(), expected);
  EXPECT_EQ(view.ToOwned(), expected);
}

TEST(LabelUtilTest, PrometheusLabelViewToOwnedStackAndHeap) {
  // Stack path
  const std::array<MetricLabel, 1> stack_labels = {
      MetricLabel{.key = "env", .value = "prod"},
  };
  EXPECT_EQ(PrometheusLabelView(stack_labels).ToOwned(), "{env=\"prod\"}");

  // Heap path
  const std::string long_value(kDefaultPrometheusStackBufferSize + 64, 'x');
  const std::array<MetricLabel, 1> heap_labels = {
      MetricLabel{.key = "env", .value = long_value},
  };
  const std::string expected = absl::StrCat("{env=\"", long_value, "\"}");
  EXPECT_EQ(PrometheusLabelView(heap_labels).ToOwned(), expected);
}
}  // namespace
}  // namespace tpu_raiden::telemetry
