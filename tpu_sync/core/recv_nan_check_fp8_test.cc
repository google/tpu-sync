// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0
#include "tpu_sync/core/recv_nan_check_fp8.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"

namespace tpu_raiden::recv_nan_internal {
namespace {
constexpr std::array<std::string_view, 4> kDtypes = {
    "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"};

// Independent scalar special-value sets from the FP8 format specifications.
bool Reference(uint8_t value, std::string_view dtype) {
  if (dtype.ends_with("fnuz")) return value == 0x80;
  if (dtype == "float8_e4m3fn") return value == 0x7f || value == 0xff;
  return (value >= 0x7c && value <= 0x7f) || value >= 0xfc;
}

struct Cases {
  std::vector<uint32_t> bits;
  std::vector<int64_t> byte, begin, end;
  std::vector<uint8_t> expected;

  void Add(uint32_t word, int64_t offset, int64_t first, int64_t last,
           std::string_view dtype) {
    bits.push_back(word);
    byte.push_back(offset);
    begin.push_back(first);
    end.push_back(last);
    bool bad = false;
    for (int lane = 0; lane < 4; ++lane)
      bad |= offset + lane >= first && offset + lane < last &&
             Reference(static_cast<uint8_t>(word >> (8 * lane)), dtype);
    expected.push_back(bad);
  }
};

void ExpectPredicate(const Cases& cases, std::string_view dtype) {
  SCOPED_TRACE(dtype);
  const auto format = NanFormatForDtype(dtype);
  ASSERT_TRUE(format);
  xla::XlaBuilder builder("fp8_received_word_test");
  const int64_t count = cases.bits.size();
  const auto matrix = [&](xla::XlaOp values) {
    return xla::Reshape(values, {count, 1});
  };
  const auto root = Fp8NonFinite(
      matrix(xla::ConstantR1<uint32_t>(&builder, cases.bits)),
      matrix(xla::ConstantR1<int64_t>(&builder, cases.byte)),
      matrix(xla::ConstantR1<int64_t>(&builder, cases.begin)),
      matrix(xla::ConstantR1<int64_t>(&builder, cases.end)), format->scan_kind);
  auto computation = builder.Build(root);
  ASSERT_TRUE(computation.ok()) << computation.status();
  // The generated scanner must retain the gather's element count; expanding
  // one word into four U32 lanes would multiply its intermediate storage.
  for (const auto& graph : computation->proto().computations())
    for (const auto& instruction : graph.instructions()) {
      int64_t elements = 1;
      for (int64_t dimension : instruction.shape().dimensions())
        elements *= dimension;
      EXPECT_LE(elements, count) << instruction.name();
    }
  auto config = xla::HloModule::CreateModuleConfigFromProto(
      computation->proto(), xla::DebugOptions());
  ASSERT_TRUE(config.ok()) << config.status();
  auto module = xla::HloModule::CreateFromProto(computation->proto(), *config);
  ASSERT_TRUE(module.ok()) << module.status();
  xla::HloEvaluator evaluator;
  auto result = evaluator.Evaluate(**module, {});
  ASSERT_TRUE(result.ok()) << result.status();
  std::vector<uint8_t> actual;
  for (bool value : result->data<bool>()) actual.push_back(value);
  EXPECT_EQ(actual, cases.expected);
}

TEST(RecvNanCheckFp8Test, ClassifiesEveryByteEncodingInAllFourLanes) {
  for (const auto dtype : kDtypes) {
    Cases cases;
    for (int lane = 0; lane < 4; ++lane)
      for (uint32_t value = 0; value < 256; ++value) {
        const uint32_t word =
            (0x25252525U & ~(0xffU << (8 * lane))) | (value << (8 * lane));
        cases.Add(word, 0, 0, 4, dtype);
      }
    ExpectPredicate(cases, dtype);
  }
}

TEST(RecvNanCheckFp8Test, KeepsAdjacentBytesIndependentWhenMaskingOneLane) {
  for (const auto dtype : kDtypes) {
    Cases cases;
    for (int lane = 0; lane < 3; ++lane)
      for (uint32_t pair = 0; pair < 65536; ++pair) {
        const uint32_t word =
            (0x25252525U & ~(0xffffU << (8 * lane))) | (pair << (8 * lane));
        // An abnormal byte just before the valid lane cannot create a false
        // marker there through an inter-byte carry or borrow.
        cases.Add(word, 0, lane + 1, lane + 2, dtype);
      }
    ExpectPredicate(cases, dtype);
  }
}

TEST(RecvNanCheckFp8Test, ClipsEveryByteEndpointIncludingEmptyAndPaddedRows) {
  for (const auto dtype : kDtypes) {
    Cases cases;
    for (int64_t begin = 0; begin < 12; ++begin)
      for (int64_t end = begin; end <= 12; ++end)
        for (int64_t byte = begin / 4 * 4; byte <= 16; byte += 4)
          for (int lane = 0; lane < 4; ++lane) {
            const uint32_t special = dtype.ends_with("fnuz") ? 0x80 : 0x7f;
            cases.Add(special << (8 * lane), byte, begin, end, dtype);
          }
    for (int lane = 0; lane < 4; ++lane)
      for (int64_t byte : {32764, 32768, 32772}) {
        const uint32_t special = dtype.ends_with("fnuz") ? 0x80 : 0x7f;
        cases.Add(special << (8 * lane), byte, 32767, 32769, dtype);
      }
    ExpectPredicate(cases, dtype);
  }
}
}  // namespace
}  // namespace tpu_raiden::recv_nan_internal
