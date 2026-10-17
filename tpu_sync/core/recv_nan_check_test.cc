// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0

#include "tpu_sync/core/recv_nan_check.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <span>

#include <gtest/gtest.h>

namespace tpu_raiden {
namespace {

TEST(RecvNanCheckTest, DistinguishesNanFromInfinityAndFiniteValues) {
  const std::array<uint16_t, 5> bf16 = {0, 0x3f80, 0x7f80, 0xff80, 0xff81};
  const std::array<uint16_t, 5> fp16 = {0, 0x3c00, 0x7c00, 0xfc00, 0x7c01};
  const std::array<uint32_t, 5> fp32 = {
      0, 0x3f800000, 0x7f800000, 0xff800000, 0x7f800001};
  EXPECT_EQ(FindNanByteOffset(std::as_bytes(std::span(bf16)),
                             *NanFormatForDtype("bfloat16")), 8);
  EXPECT_EQ(FindNanByteOffset(std::as_bytes(std::span(fp16)),
                             *NanFormatForDtype("float16")), 8);
  EXPECT_EQ(FindNanByteOffset(std::as_bytes(std::span(fp32)),
                             *NanFormatForDtype("float32")), 16);
  EXPECT_FALSE(FindNanByteOffset(std::as_bytes(std::span(bf16).first(4)),
                                *NanFormatForDtype("bfloat16")));
  EXPECT_FALSE(NanFormatForDtype("uint8"));
  EXPECT_FALSE(NanFormatForDtype("int32"));
}

TEST(RecvNanCheckTest, IgnoresRegionGapsAndUntransferredTail) {
  // Two four-byte live runs separated by padding. Only six live bytes
  // arrived, so the last bf16 is outside this request's received prefix.
  std::array<uint16_t, 6> block = {0, 0, 0x7fc1, 0x7fc1, 0, 0x7fc1};
  const std::array<NanCheckRange, 2> ranges = {{{0, 4}, {8, 4}}};
  const NanFormat format = *NanFormatForDtype("bfloat16");
  EXPECT_FALSE(FindNanInLivePrefix(std::as_bytes(std::span(block)), ranges,
                                 /*live_bytes=*/6, format));
  block[4] = 0xff81;
  EXPECT_EQ(FindNanInLivePrefix(std::as_bytes(std::span(block)), ranges,
                               /*live_bytes=*/6, format), 8);
}

TEST(RecvNanCheckTest, HandlesUnalignedBytesAndEmptyPrefix) {
  std::array<std::byte, 5> bytes{};
  const uint32_t nan = 0x7fc00001;
  std::memcpy(bytes.data() + 1, &nan, sizeof(nan));
  const auto format = *NanFormatForDtype("float32");
  EXPECT_EQ(FindNanByteOffset(std::span(bytes).subspan(1), format), 0);
  EXPECT_FALSE(FindNanByteOffset({}, format));
  EXPECT_FALSE(FindNanInLivePrefix(bytes, {}, 0, format));
}

}  // namespace
}  // namespace tpu_raiden
