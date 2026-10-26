// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0
#include "tpu_sync/core/recv_nan_check_program.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"

namespace tpu_raiden::recv_nan_internal {
namespace {
void ExpectProgram(const xla::XlaComputation& computation,
                   const std::vector<uint32_t>& words,
                   const std::vector<int64_t>& metadata,
                   const std::vector<uint8_t>& expected) {
  auto config = xla::HloModule::CreateModuleConfigFromProto(
      computation.proto(), xla::DebugOptions());
  ASSERT_TRUE(config.ok()) << config.status();
  auto module = xla::HloModule::CreateFromProto(computation.proto(), *config);
  ASSERT_TRUE(module.ok()) << module.status();
  auto raw = xla::LiteralUtil::CreateR1<uint32_t>(words);
  auto ranges = xla::LiteralUtil::CreateFromDimensions(
      xla::S64, {static_cast<int64_t>(metadata.size() / 3), 3});
  std::copy(metadata.begin(), metadata.end(), ranges.data<int64_t>().begin());
  const std::array<const xla::Literal*, 2> arguments = {&raw, &ranges};
  xla::HloEvaluator evaluator;
  auto result = evaluator.Evaluate(**module, arguments);
  ASSERT_TRUE(result.ok()) << result.status();
  std::vector<uint8_t> actual;
  for (bool value : result->data<bool>()) actual.push_back(value);
  EXPECT_EQ(actual, expected);
}

TEST(RecvNanCheckProgramTest, CachedProgramsKeepFp8ClassifiersDistinct) {
  struct FormatCase {
    std::string_view dtype;
    std::array<uint8_t, 3> expected;
  };
  const std::array<FormatCase, 4> cases = {{
      {"float8_e4m3fn", {0, 0, 1}},
      {"float8_e5m2", {1, 0, 1}},
      {"float8_e4m3fnuz", {0, 1, 0}},
      {"float8_e5m2fnuz", {0, 1, 0}},
  }};
  const std::array<uint32_t, 3> values = {0x7c7c7c7cU, 0x80808080U,
                                          0x7f7f7f7fU};
  std::vector<uint32_t> words(kNanCheckChunkBytes / 4);
  const std::vector<xla::Shape> shapes = {
      xla::ShapeUtil::MakeShape(xla::U32, {static_cast<int64_t>(words.size())}),
      xla::ShapeUtil::MakeShape(xla::S64, {1, 3})};
  std::map<std::string, xla::XlaComputation> programs;
  for (const auto& test : cases) {
    SCOPED_TRACE(test.dtype);
    const std::vector<BatchInput> inputs = {
        {0, 0, 1, *NanFormatForDtype(test.dtype)}};
    const auto batch = BuildNanCheckBatch(inputs);
    const auto key = NanCheckProgramKey(nullptr, inputs, batch, shapes);
    auto it = programs.find(key);
    if (it == programs.end()) {
      auto computation = BuildNanCheckComputation(inputs, batch, shapes);
      ASSERT_TRUE(computation.ok()) << computation.status();
      it = programs.emplace(key, std::move(*computation)).first;
    }
    for (size_t i = 0; i < values.size(); ++i) {
      words[0] = values[i];
      ExpectProgram(it->second, words, {0, 0, 4}, {test.expected[i]});
    }
  }
  // FNUZ variants share classification; the other encodings cannot share it.
  EXPECT_EQ(programs.size(), 3);
}

TEST(RecvNanCheckProgramTest, ReordersMixedPoolsWithIndependentRangesAndFlags) {
  constexpr std::array<std::string_view, 7> dtypes = {"float8_e4m3fn",
                                                      "float8_e5m2",
                                                      "float8_e4m3fnuz",
                                                      "float8_e5m2fnuz",
                                                      "bf16",
                                                      "f16",
                                                      "f32"};
  const std::array<std::vector<NanCheckChunk>, 7> chunks = {{
      {{1, 4, 5}},
      {{2, 8, 9}},
      {{16, 64, 65}, {18, 72, 73}, {20, 80, 81}},
      {{24, 96, 97}, {26, 104, 105}, {28, 112, 113}},
      {{32, 128, 130}},
      {{33, 134, 136}},
      {{34, 136, 140}},
  }};
  constexpr std::array<uint8_t, 7> flags = {0, 1, 1, 0, 1, 0, 1};
  constexpr std::array<std::array<size_t, 7>, 4> orders = {{
      {0, 1, 2, 3, 4, 5, 6},
      {6, 5, 4, 3, 2, 1, 0},
      {2, 0, 4, 1, 3, 6, 5},
      {3, 6, 1, 5, 2, 0, 4},
  }};
  // All gaps, tails and metadata-padding gathers contain FNUZ NaNs.
  std::vector<uint32_t> words(kNanCheckChunkBytes / 4, 0x80808080U);
  const auto put_byte = [&](size_t offset, uint32_t value) {
    const int shift = (offset % 4) * 8;
    words[offset / 4] =
        (words[offset / 4] & ~(0xffU << shift)) | (value << shift);
  };
  put_byte(4, 0x7e);  // E4M3FN finite maximum, next byte is an excluded NaN.
  put_byte(5, 0x7f);
  put_byte(8, 0x7c);  // E5M2 +Inf.
  for (size_t offset : {64, 80, 96, 104, 112}) put_byte(offset, 1);
  words[32] = 0x80807f80U;  // BF16 +Inf in the valid low halfword.
  words[33] = 0x7bff7c00U;  // FP16 +Inf outside the valid high halfword.
  words[34] = 0x7f800000U;  // FP32 +Inf.

  std::map<std::string, xla::XlaComputation> programs;
  for (const auto& order : orders) {
    std::vector<BatchInput> inputs;
    for (size_t index : order)
      inputs.push_back({0, 0, chunks[index].size() == 3 ? 4 : 1,
                        *NanFormatForDtype(dtypes[index])});
    const auto batch = BuildNanCheckBatch(inputs);
    ASSERT_EQ(batch.raw_inputs.size(), 1);
    ASSERT_EQ(batch.groups.size(), 6);
    std::vector<int64_t> metadata;
    std::vector<uint8_t> expected;
    for (const auto& group : batch.groups)
      for (size_t input : group.inputs) {
        const size_t index = order[input];
        for (const auto& chunk : chunks[index])
          metadata.insert(metadata.end(),
                          {chunk.word_begin, chunk.byte_begin, chunk.byte_end});
        metadata.resize(
            metadata.size() + (inputs[input].bucket - chunks[index].size()) * 3,
            0);
        expected.push_back(flags[index]);
      }
    const std::vector<xla::Shape> shapes = {
        xla::ShapeUtil::MakeShape(xla::U32,
                                  {static_cast<int64_t>(words.size())}),
        xla::ShapeUtil::MakeShape(
            xla::S64, {static_cast<int64_t>(metadata.size() / 3), 3})};
    const auto key = NanCheckProgramKey(nullptr, inputs, batch, shapes);
    auto it = programs.find(key);
    if (it == programs.end()) {
      auto computation = BuildNanCheckComputation(inputs, batch, shapes);
      ASSERT_TRUE(computation.ok()) << computation.status();
      it = programs.emplace(key, std::move(*computation)).first;
    }
    ExpectProgram(it->second, words, metadata, expected);
  }
  EXPECT_EQ(programs.size(), 1);
}
}  // namespace
}  // namespace tpu_raiden::recv_nan_internal
